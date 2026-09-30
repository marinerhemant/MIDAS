"""Blocked omega windows: find rotation ranges with no (or few) spots inside the configured OmegaRange.

A load frame's posts, a cryostat or a furnace wall shadows the beam over part of the rotation. Nothing
downstream notices: the indexer still predicts spots inside the shadow and counts each one as a miss,
so completeness drops by the shadowed fraction (unevenly, by orientation) and voxels near the
acceptance gate are lost. On 20-ID-E Fe9Cr pf data (2026-09-28) two ~16 deg posts at omega -90 and +90
cost 0.07 completeness and ~8 % of solvable voxels; excluding them switched the winner in 2.2-2.4 % of
voxels, all near-ties on grain boundaries (manuals/pf-hedm/LAB_NOTEBOOK.md section 10).

The test is on the spots themselves: counts per ``bin_deg`` of omega, runs of bins below ``frac`` x
the median, at least ``min_width_deg`` wide. Placeholder rows (ring 0: dropped spots whose row is kept
for numbering) are ignored. A window is reported, never applied: the fix is an ``OmegaRange`` per
unshadowed span (with a matching ``BoxSize`` line each), and ``suggested_ranges`` gives them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["blocked_omega_windows", "report_blocked_omega"]


def blocked_omega_windows(
    omega: np.ndarray,
    ring: Optional[np.ndarray] = None,
    *,
    ranges: Sequence[Tuple[float, float]] = ((-180.0, 180.0),),
    bin_deg: float = 1.0,
    frac: float = 0.5,
    min_width_deg: float = 2.0,
    pad_deg: float = 1.0,
    merge_gap_deg: float = 2.0,
) -> Dict:
    """Omega windows inside ``ranges`` whose spot density falls below ``frac`` x the median.

    Returns ``{"n_spots", "median_per_bin", "bin_deg", "windows": [(lo, hi, min_count)],
    "suggested_ranges": [(lo, hi)]}``; windows are widened by ``pad_deg`` on each side before the
    suggested ranges (the configured ranges minus the windows) are formed. Low runs separated by at
    most ``merge_gap_deg`` of normal-looking bins are one window (a post shadow is not split by a
    single bin that happens to pass the threshold: 20-ID-E Fe9Cr, +82..+104 deg).
    """
    om = np.asarray(omega, dtype=float).ravel()
    if ring is not None:
        om = om[np.asarray(ring).ravel() > 0]
    rngs = [(float(a), float(b)) for a, b in (ranges or [(-180.0, 180.0)])]
    out = {"n_spots": int(om.size), "bin_deg": bin_deg, "windows": [], "suggested_ranges": rngs, "median_per_bin": 0.0}
    if om.size == 0:
        return out
    per_range = []
    for a, b in rngs:
        edges = np.arange(a, b + 1e-9, bin_deg)
        if edges.size < 2:
            continue
        h, e = np.histogram(om, bins=edges)
        per_range.append((h, e))
    counts = np.concatenate([h for h, _ in per_range]) if per_range else np.zeros(0)
    if counts.size == 0:
        return out
    med = float(np.median(counts)); out["median_per_bin"] = med
    if med <= 0:
        return out
    windows: List[Tuple[float, float, int]] = []
    for h, e in per_range:
        low = h < frac * med
        runs = []
        i = 0
        while i < len(h):
            if not low[i]:
                i += 1; continue
            j = i
            while j + 1 < len(h) and low[j + 1]:
                j += 1
            runs.append([i, j]); i = j + 1
        merged = []
        for r in runs:                                   # bridge short normal-looking gaps between low runs
            if merged and (e[r[0]] - e[merged[-1][1] + 1]) <= merge_gap_deg + 1e-9:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for i, j in merged:
            if e[j + 1] - e[i] >= min_width_deg - 1e-9:
                windows.append((float(e[i]), float(e[j + 1]), int(h[i:j + 1].min())))
    out["windows"] = windows
    cut = [(lo - pad_deg, hi + pad_deg) for lo, hi, _ in windows]
    sugg = []
    for a, b in rngs:
        segs = [(a, b)]
        for lo, hi in cut:
            nxt = []
            for s, t in segs:
                if hi <= s or lo >= t:
                    nxt.append((s, t)); continue
                if lo > s: nxt.append((s, lo))
                if hi < t: nxt.append((hi, t))
            segs = nxt
        sugg.extend((round(s, 3), round(t, 3)) for s, t in segs if t - s > bin_deg)
    out["suggested_ranges"] = sugg
    return out


def report_blocked_omega(spots: np.ndarray, ranges: Sequence[Tuple[float, float]],
                         out_dir: Union[str, Path], *, tag: str = "binning") -> Dict:
    """Run :func:`blocked_omega_windows` on a Spots array (col 2 omega, col 5 ring), write
    ``omega_coverage.json`` in ``out_dir`` and log a warning with paste-ready OmegaRange lines."""
    sp = np.asarray(spots)
    rep = blocked_omega_windows(sp[:, 2], sp[:, 5], ranges=ranges)
    try:
        Path(out_dir, "omega_coverage.json").write_text(json.dumps(rep, indent=1))
    except OSError as e:                                   # a report must never stop the pipeline
        LOG.warning("%s: could not write omega_coverage.json (%s)", tag, e)
    if rep["windows"]:
        lines = "; ".join(f"OmegaRange {a:g} {b:g}" for a, b in rep["suggested_ranges"])
        LOG.warning(
            "%s: %d omega window(s) with < 50%% of the median spot density inside OmegaRange: %s "
            "(median %.0f spots/deg). A shadow (load frame, furnace) makes every predicted spot there a "
            "miss and lowers completeness. If they are shadows, set one OmegaRange per open span (and a "
            "BoxSize line for each): %s. Report: %s",
            tag, len(rep["windows"]),
            ", ".join(f"[{lo:g}, {hi:g}] (min {m}/deg)" for lo, hi, m in rep["windows"]),
            rep["median_per_bin"], lines, Path(out_dir, "omega_coverage.json"))
    return rep
