"""Variant assignment by closest fixed reference orientation.

Complementary to :mod:`kmeans_fz`: when the population is known to be a
mixture of a few canonical orientations (e.g. cube + twin-cube + brass for a
rolled FCC sheet), passing those references explicitly avoids the K-means
seeding variability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import CrystalPhase
from .kmeans_fz import _disorientation_deg, _om_to_quat

# Sigma3 reference axis-angle per phase.
# FCC / BCC: 60 deg about [111] (canonical Sigma3 twin).
# HCP:       86.3 deg about [1 -1 0 0] (Sigma3 analogue for hexagonal).
_SIGMA3_AXIS_ANGLE_DEG = {
    CrystalPhase.FCC: (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), 60.0),
    CrystalPhase.BCC: (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), 60.0),
    CrystalPhase.HCP: (np.array([1.0, -1.0, 0.0, 0.0]), 86.3),  # 4-index axis
}

_PHASE_TO_SPACE_GROUP = {
    CrystalPhase.FCC: 225,
    CrystalPhase.BCC: 229,
    CrystalPhase.HCP: 194,
}


def assign_variants_common_reference(
    OM: NDArray[np.floating],
    reference_OMs: list[NDArray[np.floating]],
    phase: CrystalPhase = CrystalPhase.FCC,
) -> NDArray[np.intp]:
    """Label each grain by the index of the closest reference orientation.

    Distance is the cubic / hexagonal disorientation angle in degrees.
    """
    OM = np.asarray(OM, dtype=float)
    if OM.ndim != 3 or OM.shape[1:] != (3, 3):
        raise ValueError(f"OM must be (n_grains, 3, 3); got {OM.shape}")
    if phase not in _PHASE_TO_SPACE_GROUP:
        raise ValueError(f"unknown phase {phase!r}")
    sg = _PHASE_TO_SPACE_GROUP[phase]

    quats_g = _om_to_quat(OM)
    refs = [np.asarray(R, dtype=float) for R in reference_OMs]
    if any(R.shape != (3, 3) for R in refs):
        raise ValueError("each reference must be a (3, 3) orientation matrix")
    quats_r = _om_to_quat(np.stack(refs, axis=0))

    n_grains, n_refs = quats_g.shape[0], quats_r.shape[0]
    labels = np.empty(n_grains, dtype=int)
    for g in range(n_grains):
        d = [_disorientation_deg(quats_g[g], quats_r[r], sg) for r in range(n_refs)]
        labels[g] = int(np.argmin(d))
    return labels


def build_sigma3_pair(
    R_matrix: NDArray[np.floating],
    phase: CrystalPhase = CrystalPhase.FCC,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(R_matrix, R_matrix @ S)`` where ``S`` is the Σ3 rotation.

    The Σ3 operator is in the **crystal** frame, so the twin orientation is
    obtained by post-multiplying the matrix orientation.
    """
    import midas_stress.orientation as o

    R_matrix = np.asarray(R_matrix, dtype=float)
    if R_matrix.shape != (3, 3):
        raise ValueError(f"R_matrix must be (3, 3); got {R_matrix.shape}")

    if phase in (CrystalPhase.FCC, CrystalPhase.BCC):
        axis_3, angle_deg = _SIGMA3_AXIS_ANGLE_DEG[phase]
        S = np.asarray(o.axis_angle_to_orient_mat(axis_3, angle_deg))
    elif phase is CrystalPhase.HCP:
        axis_4, angle_deg = _SIGMA3_AXIS_ANGLE_DEG[phase]
        # Convert 4-index axis to crystal-Cartesian.
        from ..phases.hcp import direction_bravais_to_cart

        axis_cart = direction_bravais_to_cart(*axis_4, c_over_a=1.624)  # Mg default
        S = np.asarray(o.axis_angle_to_orient_mat(axis_cart, angle_deg))
    else:
        raise ValueError(f"unknown phase {phase!r}")

    return R_matrix, R_matrix @ S


__all__ = ["assign_variants_common_reference", "build_sigma3_pair"]
