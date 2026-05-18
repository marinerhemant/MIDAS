"""Spatial-bias-aware residual trimming for calibration.

Hard top-N% trimming of pseudo-strain residuals is fast and reduces noise,
but it carries a systematic risk: if outlier residuals concentrate on one
panel / ring / η-quadrant (e.g. because that region has a real
mis-calibration), global trim drops them and the calibration converges
using only the *uncontested* regions.  The post-trim mean strain looks
clean, but the calibration fits a partial detector.

This module implements five guards:

1. **Stratified trim** — per-(ring × η-bucket × panel) quotas.  Each cell
   loses the same percentile of its own residuals, so spatial coverage is
   preserved.

2. **MAD-based cutoff** — drop only true outliers (|r| > median + k·MAD)
   instead of a fixed percentile.  Tight residual distributions are not
   trimmed at all.

3. **Floor on cell occupancy** — refuse to trim a cell down to fewer than
   ``min_per_cell`` points.  Prevents over-trimming sparse rings.

4. **Diagnostic** — :func:`trim_diagnostic` reports per-cell rejection
   rates and flags when one cell contributes >2× its uniform share of
   rejections.

5. **Un-trimmed evaluation** — :func:`evaluate_full_strain` reports the
   strain metric on ALL fits (including those trimmed by the LM), so the
   user sees the honest mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class TrimReport:
    """Per-cell rejection diagnostic."""
    n_total: int
    n_kept: int
    n_rejected: int
    cells_with_rejections: int
    flagged_cells: List[Tuple[Tuple[int, int, int], int, float]]
    """(cell key, n_rejected_in_cell, rejection_rate) for cells flagged."""

    def render(self) -> str:
        lines = [
            f"trim diagnostic: {self.n_rejected}/{self.n_total} rejected "
            f"({100.0 * self.n_rejected / max(self.n_total, 1):.1f}%) "
            f"across {self.cells_with_rejections} cells",
        ]
        if self.flagged_cells:
            lines.append(f"  ⚠ {len(self.flagged_cells)} cells with rejection "
                          f"rate > 2× uniform expectation:")
            for key, n, rate in self.flagged_cells[:10]:
                ring, eta_b, panel = key
                lines.append(
                    f"    ring={ring} η-bucket={eta_b} "
                    f"panel={panel}: {n} rejected ({100*rate:.0f}%)"
                )
            if len(self.flagged_cells) > 10:
                lines.append(f"    ... and {len(self.flagged_cells) - 10} more")
        else:
            lines.append("  ✓ no spatial clumping detected (uniform rejection)")
        return "\n".join(lines)


def stratified_trim(
    abs_residuals: torch.Tensor,           # [N]
    ring_idx: torch.Tensor,                # [N] long
    eta_deg: torch.Tensor,                 # [N]
    panel_idx: Optional[torch.Tensor] = None,
    *,
    keep_pct: float = 90.0,
    n_eta_buckets: int = 8,                # 8 quadrants of 45° each
    min_per_cell: int = 3,
    use_mad: bool = True,
    mad_k: float = 5.0,
) -> Tuple[torch.Tensor, TrimReport]:
    """Stratified trim with optional MAD-based cutoff and per-cell quotas.

    The cell key is ``(ring_idx, eta_bucket, panel_idx)``.  Within each cell:

    - Use MAD: drop fits with ``|r| > median(|r|) + mad_k * MAD(|r|)``.
    - OR use percentile: drop top ``(1 - keep_pct/100)`` of ``|r|``.
    - Always keep at least ``min_per_cell`` fits per cell (refuse to over-trim).

    Returns
    -------
    keep : torch.BoolTensor [N]
    report : :class:`TrimReport`
    """
    N = abs_residuals.numel()
    keep = torch.ones(N, dtype=torch.bool, device=abs_residuals.device)
    if N == 0:
        return keep, TrimReport(0, 0, 0, 0, [])

    # Bucketise η to integer keys.
    eta_b = ((eta_deg + 180.0) % 360.0 / (360.0 / n_eta_buckets)).floor().long()
    eta_b = eta_b.clamp(0, n_eta_buckets - 1)

    if panel_idx is None:
        panel_keys = torch.zeros(N, dtype=torch.long, device=abs_residuals.device)
    else:
        panel_keys = panel_idx.clamp(min=-1)

    # Build per-cell index dictionary by hashing (ring, eta_bucket, panel).
    cells: dict = {}
    rid_np = ring_idx.cpu().numpy()
    eb_np = eta_b.cpu().numpy()
    pn_np = panel_keys.cpu().numpy()
    abs_r_np = abs_residuals.detach().cpu().numpy()
    for i in range(N):
        k = (int(rid_np[i]), int(eb_np[i]), int(pn_np[i]))
        cells.setdefault(k, []).append(i)

    flagged: List[Tuple[Tuple[int, int, int], int, float]] = []
    cells_with_rej = 0
    rejected_cells_count: dict = {}

    drop_mask = torch.zeros(N, dtype=torch.bool, device=abs_residuals.device)
    for k, idxs in cells.items():
        if len(idxs) <= min_per_cell:
            continue
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=abs_residuals.device)
        r_cell = abs_residuals.index_select(0, idxs_t)
        if use_mad:
            med = float(r_cell.median())
            mad = float((r_cell - med).abs().median())
            cutoff = med + mad_k * mad
        else:
            cutoff = float(torch.quantile(r_cell, keep_pct / 100.0))
        # Mark drops, but respect the min_per_cell floor.
        drops_in_cell = r_cell > cutoff
        # Sort by residual descending, keep at least min_per_cell.
        if int(drops_in_cell.sum()) > 0:
            keep_count = max(min_per_cell, len(idxs) - int(drops_in_cell.sum()))
            order = torch.argsort(r_cell, descending=False)
            cell_keep_local = torch.zeros(len(idxs), dtype=torch.bool,
                                            device=abs_residuals.device)
            cell_keep_local[order[:keep_count]] = True
            cell_drop_local = ~cell_keep_local
            drop_mask[idxs_t] = cell_drop_local
            n_rej = int(cell_drop_local.sum())
            if n_rej > 0:
                cells_with_rej += 1
                rejected_cells_count[k] = n_rej

    # Spatial-clumping flag: cells with >2× uniform-share rejection.
    if cells:
        total_rej = int(drop_mask.sum())
        uniform_share = total_rej / max(len(cells), 1)
        for k, n in rejected_cells_count.items():
            cell_size = len(cells[k])
            rate = n / cell_size
            if n > 2.0 * uniform_share and cell_size >= min_per_cell:
                flagged.append((k, n, rate))
        flagged.sort(key=lambda t: -t[1])

    keep = ~drop_mask
    report = TrimReport(
        n_total=N,
        n_kept=int(keep.sum()),
        n_rejected=int(drop_mask.sum()),
        cells_with_rejections=cells_with_rej,
        flagged_cells=flagged,
    )
    return keep, report


def multfactor_trim(
    residuals: torch.Tensor,
    *,
    factor: float = 2.0,
    max_iter: int = 10,
) -> Tuple[torch.Tensor, "TrimReport"]:
    """Iterative ``|r| > factor × current_mean`` rejection.

    Mirrors v1 C's ``CalibrationCore::Mean_Difference_Refined`` outlier
    cut (controlled by the ``MultFactor`` parameter file key, default 2.0).
    The cut iterates until the kept set is stable, then reports the mean
    over the survivors.  Aggressive — keeps ~30-40% of fits on Pilatus —
    but produces strain numbers that compare apples-to-apples with v1.

    Returns (keep_mask, report).
    """
    r_abs = residuals.abs()
    keep = torch.ones_like(r_abs, dtype=torch.bool)
    iters_used = 0
    for it in range(max_iter):
        cur = r_abs[keep]
        if cur.numel() == 0:
            break
        m = cur.mean()
        thresh = factor * m
        new_keep = r_abs <= thresh
        iters_used = it + 1
        if torch.equal(new_keep, keep):
            break
        keep = new_keep
    n_total = int(r_abs.numel())
    n_kept = int(keep.sum())
    report = TrimReport(
        n_total=n_total,
        n_kept=n_kept,
        n_rejected=n_total - n_kept,
        cells_with_rejections=0,
        flagged_cells=[],
    )
    return keep, report


def stratified_multfactor_trim(
    residuals: torch.Tensor,
    ring_idx: torch.Tensor,
    eta_deg: torch.Tensor,
    panel_idx: Optional[torch.Tensor] = None,
    *,
    factor: float = 2.0,
    n_eta_buckets: int = 8,
    min_per_cell: int = 3,
    max_iter: int = 10,
) -> Tuple[torch.Tensor, "TrimReport"]:
    """Per-(ring × η-bucket × panel) iterative `|r| > factor × cell_mean`.

    Spatially-aware variant of :func:`multfactor_trim`.  Each cell decides
    its own threshold from its own mean, with a floor of ``min_per_cell``
    points (cells never collapse below this).  Preserves coverage even
    when a region has high residual; the global headline mean is honest
    rather than dominated by the easiest fits.

    Use when v1's metric form (`MultFactor`) is desired AND spatial
    coverage matters (the global :func:`multfactor_trim` will reject
    80%+ on Pilatus and concentrate the kept set in 2-3 panels).
    """
    r_abs = residuals.abs()
    dev = r_abs.device
    # Eta buckets: split into n_eta_buckets equal-sized bins over [-180, 180].
    eta_b = (((eta_deg.to(dev) + 180.0) / 360.0 * n_eta_buckets)
             .clamp(0, n_eta_buckets - 1).long())
    rid = ring_idx.to(dev).long()
    pid = (panel_idx.to(dev).long()
            if panel_idx is not None else torch.zeros_like(rid))

    # Build a unique cell key per fit.
    n_rings = int(rid.max().item()) + 1 if rid.numel() else 1
    n_panels = (int(pid.max().item()) + 2) if pid.numel() else 2  # +1 for -1 (gap)
    pid_pos = pid + 1   # shift -1 (gap) to 0 so we can use as an index
    cell_key = (rid * n_eta_buckets + eta_b) * n_panels + pid_pos

    keep = torch.ones_like(r_abs, dtype=torch.bool)
    n_total = int(r_abs.numel())
    iters_used = 0
    for it in range(max_iter):
        any_change = False
        # Per-cell mean over current kept set.
        # Use scatter-mean: sum of kept, count of kept.
        kept_r = r_abs * keep.float()
        n_cells = int(cell_key.max().item()) + 1
        sum_per_cell = torch.zeros(n_cells, dtype=r_abs.dtype, device=r_abs.device)
        cnt_per_cell = torch.zeros(n_cells, dtype=torch.long, device=r_abs.device)
        sum_per_cell.scatter_add_(0, cell_key, kept_r)
        cnt_per_cell.scatter_add_(0, cell_key, keep.long())
        mean_per_cell = sum_per_cell / cnt_per_cell.clamp(min=1).to(sum_per_cell.dtype)
        thresh_per_fit = factor * mean_per_cell[cell_key]
        proposed = r_abs <= thresh_per_fit
        # Apply floor: in any cell where applying `proposed` would drop
        # below min_per_cell, keep the lowest-|r| points to maintain the
        # floor.
        new_keep = keep.clone()
        for c_id in torch.unique(cell_key).tolist():
            in_cell = (cell_key == c_id) & keep
            n_kept_now = int(in_cell.sum())
            if n_kept_now == 0:
                continue
            # Evaluate per-cell rejection candidates.
            cell_proposed = proposed[in_cell]
            cell_kept_after = int(cell_proposed.sum())
            if cell_kept_after >= min_per_cell:
                # OK to apply the cut as-is.
                idx_in_cell = in_cell.nonzero(as_tuple=False).squeeze(-1)
                # set new_keep[idx_in_cell] = cell_proposed
                new_keep[idx_in_cell] = cell_proposed
            else:
                # Too aggressive — keep the smallest-|r| floor.
                idx_in_cell = in_cell.nonzero(as_tuple=False).squeeze(-1)
                cell_r = r_abs[idx_in_cell]
                keep_n = min(n_kept_now, max(min_per_cell, int(cell_proposed.sum())))
                # smallest keep_n by |r|
                _, order = torch.sort(cell_r)
                keep_idx = idx_in_cell[order[:keep_n]]
                new_keep[idx_in_cell] = False
                new_keep[keep_idx] = True
        if not torch.equal(new_keep, keep):
            any_change = True
        keep = new_keep
        iters_used = it + 1
        if not any_change:
            break

    n_kept = int(keep.sum())

    # Diagnostic: per-cell rejection rate; flag cells where rejection
    # >2× the uniform rate.
    n_rej = n_total - n_kept
    uniform_rate = n_rej / max(n_total, 1)
    flagged = []
    for c_id in torch.unique(cell_key).tolist():
        in_cell = cell_key == c_id
        n_in = int(in_cell.sum())
        if n_in == 0:
            continue
        n_kept_cell = int((in_cell & keep).sum())
        n_rej_cell = n_in - n_kept_cell
        rate = n_rej_cell / n_in
        if rate > 2.0 * max(uniform_rate, 0.01) and n_rej_cell > 1:
            ring = c_id // (n_eta_buckets * n_panels)
            rem = c_id % (n_eta_buckets * n_panels)
            eta_b_id = rem // n_panels
            panel_id = (rem % n_panels) - 1   # undo +1 shift
            flagged.append(((ring, eta_b_id, panel_id), n_rej_cell, rate))
    flagged.sort(key=lambda x: -x[2])
    n_cells_with_rej = sum(1 for _, n, _ in flagged if n > 0)

    report = TrimReport(
        n_total=n_total,
        n_kept=n_kept,
        n_rejected=n_rej,
        cells_with_rejections=n_cells_with_rej,
        flagged_cells=flagged,
    )
    return keep, report


def evaluate_full_strain(
    residual_fn,
    unpacked: dict,
) -> Tuple[float, float, float]:
    """Compute mean / median / RMS pseudo-strain on the full (un-trimmed) set.

    Use this to *report* calibration quality after the LM has converged on
    a trimmed subset.  If `mean(full)` is much larger than `mean(trimmed)`,
    the trim was hiding evidence.
    """
    with torch.no_grad():
        r = residual_fn(unpacked).abs()
        mean = float(r.mean())
        med = float(r.median())
        rms = float((r * r).mean().sqrt())
    return mean, med, rms


__all__ = ["stratified_trim", "multfactor_trim", "stratified_multfactor_trim",
            "evaluate_full_strain", "TrimReport"]
