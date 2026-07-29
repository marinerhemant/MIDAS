"""Intensity-budget decomposition — the headline diffuse-metrology capability.

Every above-threshold voxel is assigned to exactly one of four physically
distinct components, and the intensity fractions **sum to 1.0** (closure). This
is what licenses *quantitative* statements ("31 % dislocation asterism, 4 %
fault rods") instead of qualitative ones.

Bins (criteria from the demk production analysis, full-res, tol in 1/Å):

  - ``bragg``        : dist-to-lattice < ``bragg_tol``                  (the grains)
  - ``asterism``     : ``bragg_tol`` ≤ dist < ``near_tol`` and q ≥ ``low_q_cut``
                       (near-Bragg dislocation strain broadening)
  - ``inter_bragg``  : dist ≥ ``near_tol`` and q ≥ ``low_q_cut``
                       (fault rods + smooth background)
  - ``low_q_halo``   : q < ``low_q_cut``                               (beam-stop region)

The bins are a partition (mutually exclusive, exhaustive), so closure is exact
by construction — the test is whether each voxel is counted once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class IntensityBudget:
    """Intensity fractions per component (sum to 1.0) + bookkeeping."""
    fractions: Dict[str, float]
    intensities: Dict[str, float]
    total_intensity: float
    counts: Dict[str, int] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)

    def closes(self, atol: float = 1e-9) -> bool:
        """True if the fractions sum to 1 within ``atol`` (closure check)."""
        return abs(sum(self.fractions.values()) - 1.0) <= atol

    def __str__(self) -> str:
        lines = ["intensity budget (% of total):"]
        for k, v in self.fractions.items():
            lines.append(f"  {k:<12s} {100*v:6.2f} %   ({self.counts.get(k,0)} vox)")
        lines.append(f"  {'sum':<12s} {100*sum(self.fractions.values()):6.2f} %")
        return "\n".join(lines)


def intensity_budget(
    dist_to_lattice: np.ndarray,
    q_mag: np.ndarray,
    intensity: np.ndarray,
    *,
    bragg_tol: float = 0.05,
    near_tol: float = 0.20,
    low_q_cut: float = 1.5,
) -> IntensityBudget:
    """Decompose total scattered intensity into the four components.

    Parameters
    ----------
    dist_to_lattice
        (N,) nearest-predicted-reflection distance in 1/Å
        (from `bragg_diffuse.classify_voxels(...).dist_to_lattice`).
    q_mag
        (N,) |q| in 1/Å per voxel.
    intensity
        (N,) per-voxel intensity.
    bragg_tol, near_tol, low_q_cut
        Bin edges in 1/Å (demk production: 0.05, 0.20, 1.5).

    Returns
    -------
    IntensityBudget — fractions sum to 1.0 exactly (partition of all voxels).
    """
    dist = np.asarray(dist_to_lattice, dtype=np.float64)
    q = np.asarray(q_mag, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)

    low_q = q < low_q_cut
    high_q = ~low_q
    bragg = (dist < bragg_tol) & high_q
    asterism = (dist >= bragg_tol) & (dist < near_tol) & high_q
    inter_bragg = (dist >= near_tol) & high_q
    # NB: voxels with q < low_q_cut go to the halo regardless of dist, so the
    # four masks are a true partition.
    masks = {
        "bragg": bragg,
        "asterism": asterism,
        "inter_bragg": inter_bragg,
        "low_q_halo": low_q,
    }

    tot = float(I.sum())
    inten = {k: float(I[m].sum()) for k, m in masks.items()}
    fracs = {k: (v / tot if tot > 0 else 0.0) for k, v in inten.items()}
    counts = {k: int(m.sum()) for k, m in masks.items()}

    # partition sanity: every voxel counted exactly once
    stacked = np.zeros(len(I), dtype=np.int8)
    for m in masks.values():
        stacked += m.astype(np.int8)
    assert stacked.max() <= 1 and stacked.min() >= 1, "budget bins are not a partition"

    return IntensityBudget(
        fractions=fracs,
        intensities=inten,
        total_intensity=tot,
        counts=counts,
        params={"bragg_tol": bragg_tol, "near_tol": near_tol, "low_q_cut": low_q_cut},
    )
