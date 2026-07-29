"""Detector-fixed artifact screening for diffraction-feature catalogues.

A real Bragg/diffuse reflection from a *rotating* sample appears at a specific
``(detector_row, detector_col, omega)``. A feature pinned to **one detector pixel
across a wide range of omega** is fixed in the lab/detector frame, not the sample
frame -- a hot/dead pixel, a persistent zinger site, or parasitic (non-sample)
scatter. Such features are NOT crystallographic and must be screened out before
any q-space analysis: because ``|q|`` depends only on the detector radius, a fixed
pixel produces a *fixed* ``|q|``, so it masquerades as a sharp recurring "reflection
cluster" in any |q|-only catalogue (this is exactly how a hot pixel faked a
"q/G=1.518" satellite family in the demk Cu-Al analysis).

The screen groups catalogue entries by detector pixel and flags any pixel whose
entries span more omega than a genuine reflection ever could.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["FixedPixelArtifact", "flag_fixed_pixel_artifacts"]


@dataclass
class FixedPixelArtifact:
    """A detector pixel that hosts features across a wide omega range."""

    row: float
    col: float
    label_ids: list
    n_entries: int
    omega_span_deg: float


def flag_fixed_pixel_artifacts(
    label_ids: NDArray[np.integer],
    cen_row: NDArray[np.floating],
    cen_col: NDArray[np.floating],
    cen_frame: NDArray[np.floating],
    *,
    deg_per_frame: float = 0.25,
    pixel_tol: float = 4.0,
    omega_span_min_deg: float = 60.0,
    min_entries: int = 4,
) -> tuple[list[int], list[FixedPixelArtifact]]:
    """Flag catalogue entries that sit at a fixed detector pixel across wide omega.

    Parameters
    ----------
    label_ids, cen_row, cen_col, cen_frame
        Per-feature catalogue: integer id and detector-space centroid
        (row, col in pixels; frame index). ``omega = omega0 - deg_per_frame*frame``
        but only the *span* matters here, so the sign/offset are irrelevant.
    pixel_tol
        Entries within this many pixels (Chebyshev) are treated as the same pixel.
    omega_span_min_deg
        A pixel whose entries span at least this much omega is flagged. A real
        reflection (even a broad mosaic spot) spans far less; only a lab-fixed
        feature spans tens of degrees, up to the full rotation.
    min_entries
        Require at least this many entries at the pixel (guards against a single
        broad spot being flagged).

    Returns
    -------
    flagged_ids : list[int]
        Ids of all entries belonging to a flagged pixel -- exclude these.
    artifacts : list[FixedPixelArtifact]
        One per offending pixel, with its member ids and omega span.

    Notes
    -----
    Greedy single-pass clustering by rounded pixel; adequate because true artifacts
    are tightly localized (a few pixels). Does not need the beam centre or geometry.
    """
    ids = np.asarray(label_ids)
    r = np.asarray(cen_row, dtype=np.float64)
    c = np.asarray(cen_col, dtype=np.float64)
    f = np.asarray(cen_frame, dtype=np.float64)
    if not (len(ids) == len(r) == len(c) == len(f)):
        raise ValueError("label_ids, cen_row, cen_col, cen_frame must be the same length")

    # group by rounded pixel cell of size pixel_tol
    key_r = np.round(r / pixel_tol).astype(np.int64)
    key_c = np.round(c / pixel_tol).astype(np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(len(ids)):
        groups.setdefault((int(key_r[i]), int(key_c[i])), []).append(i)

    flagged_ids: list[int] = []
    artifacts: list[FixedPixelArtifact] = []
    for (_, _), idx in groups.items():
        if len(idx) < min_entries:
            continue
        idx_arr = np.asarray(idx)
        span = float((f[idx_arr].max() - f[idx_arr].min()) * deg_per_frame)
        if span < omega_span_min_deg:
            continue
        member_ids = [int(ids[i]) for i in idx]
        flagged_ids.extend(member_ids)
        artifacts.append(
            FixedPixelArtifact(
                row=float(np.mean(r[idx_arr])),
                col=float(np.mean(c[idx_arr])),
                label_ids=member_ids,
                n_entries=len(idx),
                omega_span_deg=span,
            )
        )
    return flagged_ids, artifacts
