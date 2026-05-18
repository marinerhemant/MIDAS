"""Distribution-level reporting for calibration residuals.

Mean and median are convenient summaries but they hide the *shape* of the
residual distribution.  For calibration, the shape matters:

- A tight Gaussian = the calibration is at the geometric noise floor.
- A heavy upper tail = a small subset of bins is mis-fit (either bad data
  or a real systematic the model doesn't capture).
- Bimodal = two regions of the detector at different effective
  geometries (e.g. a bad panel).
- Per-cell anomalies (one cell 100× the rest) = a localized bug.

This module provides two diagnostics:

- :func:`distribution_report` — quantile sweep + log-spaced ASCII histogram.
- :func:`per_cell_summary`   — per-(ring × η-bucket × panel) median strain,
  with cells flagged when their residual is anomalously high relative to
  the global median.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def distribution_report(
    abs_residuals_uE: torch.Tensor,
    *,
    label: str = "|r|",
    n_hist_bins: int = 16,
    bar_width: int = 36,
) -> str:
    """Multi-line ASCII summary of the residual distribution.

    Parameters
    ----------
    abs_residuals_uE : tensor
        Absolute pseudo-strain residuals in microstrain.
    """
    if abs_residuals_uE.numel() == 0:
        return f"  {label}: (empty)"
    r = abs_residuals_uE.detach().cpu().numpy().astype(np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return f"  {label}: (no finite values)"

    quantiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
    qvals = np.quantile(r, quantiles)

    lines = [
        f"  {label} (μϵ): n={r.size}, mean={r.mean():.2f}, "
        f"std={r.std():.2f}, max={r.max():.2f}",
        "    quantiles  " + "  ".join(f"q{int(q*100):>3}" for q in quantiles),
        "               " + "  ".join(f"{v:>4.1f}" for v in qvals),
    ]

    # Log-binned ASCII histogram covering [10%-quantile, 99.9%-quantile].
    lo = max(qvals[0], 1e-3)
    hi = max(qvals[-1] * 1.05, lo * 10)
    if hi > lo and r.size >= 8:
        edges = np.geomspace(lo, hi, n_hist_bins + 1)
        counts, _ = np.histogram(r, bins=edges)
        n_below = int((r < edges[0]).sum())
        n_above = int((r > edges[-1]).sum())
        max_c = max(counts.max(), 1)
        lines.append(f"    log-binned histogram of {label} (μϵ):")
        if n_below:
            lines.append(f"      [        , {edges[0]:7.2f}): "
                          f"{n_below} (below)")
        for c, b_lo, b_hi in zip(counts, edges[:-1], edges[1:]):
            bar_len = int(round(bar_width * c / max_c))
            bar = "█" * bar_len + " " * (bar_width - bar_len)
            lines.append(f"      [{b_lo:7.2f}, {b_hi:7.2f}): {bar} {c}")
        if n_above:
            lines.append(f"      [{edges[-1]:7.2f}, ∞       ): {n_above} (above)")
    return "\n".join(lines)


def per_cell_summary(
    abs_residuals_uE: torch.Tensor,
    ring_idx: torch.Tensor,
    eta_deg: torch.Tensor,
    panel_idx: Optional[torch.Tensor] = None,
    *,
    n_eta_buckets: int = 8,
    flag_factor: float = 3.0,
    top_k: int = 10,
) -> str:
    """Per-(ring × η-bucket × panel) median strain summary.

    Cells whose median is more than ``flag_factor × global median`` are
    flagged.  Output is a sorted list of the top ``top_k`` worst cells.
    """
    n = abs_residuals_uE.numel()
    if n == 0:
        return "  per-cell summary: (empty)"
    r = abs_residuals_uE.detach().cpu().numpy().astype(np.float64)
    rid = ring_idx.detach().cpu().numpy().astype(np.int64)
    eta = eta_deg.detach().cpu().numpy().astype(np.float64)
    pid = (panel_idx.detach().cpu().numpy().astype(np.int64) if panel_idx is not None
           else np.zeros(n, dtype=np.int64))

    eta_b = np.clip(((eta + 180.0) % 360.0 / (360.0 / n_eta_buckets)).astype(np.int64),
                     0, n_eta_buckets - 1)

    keys: dict = {}
    for i in range(n):
        k = (int(rid[i]), int(eta_b[i]), int(pid[i]))
        keys.setdefault(k, []).append(r[i])

    rows = []
    for k, vals in keys.items():
        rows.append((k, np.median(vals), len(vals)))
    rows.sort(key=lambda t: -t[1])  # descending median

    global_med = float(np.median(r))
    flagged = [r for r in rows if r[1] > flag_factor * global_med]

    lines = [
        f"  per-cell median |r| (n_cells={len(rows)}, "
        f"global median={global_med:.2f} μϵ, "
        f"flag if cell-median > {flag_factor}× global):"
    ]
    if flagged:
        lines.append(f"    ⚠ {len(flagged)} flagged cells (top {min(top_k, len(flagged))}):")
        for k, med, n_in_cell in flagged[:top_k]:
            ring, eb, panel = k
            lines.append(f"      ring={ring:3d}  η-bucket={eb:2d}  "
                          f"panel={panel:3d}  n={n_in_cell:3d}  "
                          f"median={med:7.2f} μϵ "
                          f"({med / max(global_med, 1e-6):.1f}× global)")
    else:
        lines.append("    ✓ no spatially anomalous cells")
    return "\n".join(lines)


def strain_summary(
    residuals_uE: torch.Tensor,
    ring_idx: Optional[torch.Tensor] = None,
    eta_deg: Optional[torch.Tensor] = None,
    panel_idx: Optional[torch.Tensor] = None,
    *,
    label: str = "|r|",
) -> str:
    """One-call combined summary: distribution + per-cell breakdown."""
    abs_r = residuals_uE.abs() if (residuals_uE < 0).any() else residuals_uE
    parts = [distribution_report(abs_r, label=label)]
    if ring_idx is not None and eta_deg is not None:
        parts.append(per_cell_summary(abs_r, ring_idx, eta_deg, panel_idx))
    return "\n".join(parts)


__all__ = ["distribution_report", "per_cell_summary", "strain_summary"]
