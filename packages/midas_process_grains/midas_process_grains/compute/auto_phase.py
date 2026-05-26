"""Auto-phase detection from observed ring d-spacings.

Given a list of ring numbers / d-spacings observed in the data and a
library of candidate phases (each with its own ``hkls.csv``-style ring
table), pick the phase whose theoretical d-spacings best match the
observed ones.

This is a **rough** classifier — useful when the user has unlabelled FF
data and wants a starting space-group guess. For production analysis the
user should still verify SG manually.

Approach: for each candidate phase, compute the sum of squared
fractional d-spacing differences for the best-matching theoretical ring
of each observed ring. Lowest score wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PhaseCandidate:
    """One candidate phase for auto-detection."""

    name: str
    space_group: int
    lattice: Tuple[float, float, float, float, float, float]
    d_spacings_A: np.ndarray   # (n_rings,) — theoretical d-spacings


@dataclass
class AutoPhaseResult:
    best: PhaseCandidate
    score: float                            # lower = better
    all_scores: Dict[str, float]


# Common metallic + oxide phases (rough; for production add the user's own)
COMMON_PHASES: List[PhaseCandidate] = [
    PhaseCandidate(
        name="Cu_FCC", space_group=225, lattice=(3.615, 3.615, 3.615, 90, 90, 90),
        d_spacings_A=np.array([2.087, 1.808, 1.278, 1.090, 1.044]),
    ),
    PhaseCandidate(
        name="Ni_FCC", space_group=225, lattice=(3.524, 3.524, 3.524, 90, 90, 90),
        d_spacings_A=np.array([2.035, 1.762, 1.246, 1.063, 1.018]),
    ),
    PhaseCandidate(
        name="Au_FCC", space_group=225, lattice=(4.078, 4.078, 4.078, 90, 90, 90),
        d_spacings_A=np.array([2.355, 2.039, 1.442, 1.230, 1.178]),
    ),
    PhaseCandidate(
        name="alpha_Fe_BCC", space_group=229, lattice=(2.866, 2.866, 2.866, 90, 90, 90),
        d_spacings_A=np.array([2.027, 1.433, 1.170, 1.013, 0.9061]),
    ),
    PhaseCandidate(
        name="austenite_FCC", space_group=225, lattice=(3.594, 3.594, 3.594, 90, 90, 90),
        d_spacings_A=np.array([2.075, 1.797, 1.271, 1.084, 1.038]),
    ),
    PhaseCandidate(
        name="alpha_Ti_HCP", space_group=194, lattice=(2.951, 2.951, 4.685, 90, 90, 120),
        d_spacings_A=np.array([2.555, 2.342, 2.243, 1.726, 1.476]),
    ),
    PhaseCandidate(
        name="Mg_HCP", space_group=194, lattice=(3.209, 3.209, 5.211, 90, 90, 120),
        d_spacings_A=np.array([2.779, 2.605, 2.452, 1.901, 1.605]),
    ),
    PhaseCandidate(
        name="alpha_Al2O3", space_group=167, lattice=(4.763, 4.763, 13.003, 90, 90, 120),
        d_spacings_A=np.array([3.479, 2.552, 2.379, 2.085, 1.740]),
    ),
    PhaseCandidate(
        name="LiMn2O4_LMO", space_group=227, lattice=(8.247, 8.247, 8.247, 90, 90, 90),
        d_spacings_A=np.array([4.762, 2.916, 2.486, 2.060, 1.682]),
    ),
]


def detect_phase(
    observed_d_A: np.ndarray,
    candidates: Optional[List[PhaseCandidate]] = None,
    top_k: int = 3,
) -> AutoPhaseResult:
    """Pick the best-matching phase for an observed set of d-spacings.

    Parameters
    ----------
    observed_d_A : (n_rings,) float
        Observed d-spacings in Å.
    candidates : list of PhaseCandidate, optional
        Library to search. Defaults to ``COMMON_PHASES``.
    top_k : int
        How many top candidates to report by name (the best is also
        returned as ``best``).

    Returns
    -------
    AutoPhaseResult
    """
    obs = np.asarray(observed_d_A, dtype=np.float64)
    cands = candidates if candidates is not None else COMMON_PHASES
    scores: Dict[str, float] = {}
    for c in cands:
        # For each observed d, find best-matching theoretical d
        diffs = []
        for d in obs:
            rel = np.min(np.abs(c.d_spacings_A - d)) / max(d, 1e-9)
            diffs.append(rel)
        scores[c.name] = float(np.mean(diffs))
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    best_name = ordered[0][0]
    best_score = ordered[0][1]
    best = next(c for c in cands if c.name == best_name)
    return AutoPhaseResult(best=best, score=best_score, all_scores=scores)


def detect_phase_from_inputall(
    inputall_df,
    candidates: Optional[List[PhaseCandidate]] = None,
    lsd_um: float = 1_000_000.0,
    wavelength_A: float = 0.172979,
) -> AutoPhaseResult:
    """Convenience: derive observed ring d-spacings from InputAll YLab,ZLab.

    For each spot: 2θ = arctan(√(Y² + Z²) / Lsd); d = λ / (2 sinθ).
    Group by ring and take the median d per ring; pass to
    :func:`detect_phase`.
    """
    import pandas as pd
    df = inputall_df.copy()
    if "RingNumber" not in df.columns:
        raise ValueError("InputAll must have RingNumber column")
    df.columns = [c.lstrip("%") for c in df.columns]
    Y = df["YLab"].to_numpy(np.float64)
    Z = df["ZLab"].to_numpy(np.float64)
    two_th = np.arctan2(np.sqrt(Y * Y + Z * Z), float(lsd_um))
    d = wavelength_A / (2.0 * np.sin(two_th / 2.0))
    df = df.assign(_d=d)
    d_per_ring = df.groupby("RingNumber")["_d"].median().sort_index().to_numpy()
    return detect_phase(d_per_ring, candidates=candidates)
