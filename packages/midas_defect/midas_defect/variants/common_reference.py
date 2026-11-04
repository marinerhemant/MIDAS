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

# The twin / Sigma3 orientation relationships themselves (operators, axis families, pair
# building and finding) live in :mod:`midas_stress.twin_relations`; this module only picks the
# relation for a phase. FCC / BCC: the Sigma3 twin, 60 deg about <111>. HCP: a deformation twin
# of the hexagonal cell (default the {10-12} tension twin), built from its K1 plane normal in the
# requested crystal frame.
#
# History: the earlier HCP "Sigma3 analogue" was 86.3 deg about the textbook-frame label [1-100] at
# Mg c/a. As a Cartesian vector that axis IS a real a axis of the MIDAS frame (a at -30 deg), so
# for MIDAS orientations it was, by accident, the Mg {10-12} twin (86.3 deg about <11-20>); for
# orientations in the textbook frame it was a <10-10> axis, which is no twin. It was documented as
# <1-100>, fixed to Mg, and its default 55-65 deg window could never admit an 85-86 deg twin. The
# twin is now chosen explicitly (``hcp_twin``) and its axis family is derived from the relation.

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


def _sigma3_relation(phase, *, c_over_a: float = 1.624, hcp_twin: str = "tension_1012",
                     frame: str = "midas"):
    """The ``midas_stress.TwinRelation`` used as the 'Sigma3' relation of ``phase``."""
    from midas_stress.twin_relations import default_fcc_twin_relations, default_hcp_twin_relations
    if phase in (CrystalPhase.FCC, CrystalPhase.BCC):
        return default_fcc_twin_relations()[0]            # 60 deg about <111>
    if phase is CrystalPhase.HCP:
        return default_hcp_twin_relations(c_over_a, systems=(hcp_twin,), frame=frame)[0]
    raise ValueError(f"unknown phase {phase!r}")


def build_sigma3_pair(
    R_matrix: NDArray[np.floating],
    phase: CrystalPhase = CrystalPhase.FCC,
    *,
    frame: str = "midas",
    c_over_a: float = 1.624,
    hcp_twin: str = "tension_1012",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(R_matrix, R_matrix @ T)`` where ``T`` is the phase's twin operator.

    The operator is in the **crystal** frame, so the twin orientation is obtained by
    post-multiplying the matrix orientation. FCC/BCC: 60 deg about [111]. HCP: the ``hcp_twin``
    deformation twin (``"tension_1012"``, ``"compression_1011"``, ``"tension_1121"``,
    ``"compression_1122"``) at ``c_over_a`` (default Mg 1.624, where {10-12} is 86.3 deg),
    built in ``frame`` (``"midas"`` default: the frame of Grains.csv orientations;
    ``"crystallographic"``: textbook a1 along x). Implemented by
    :func:`midas_stress.twin_relations.build_twin_pair`.
    """
    from midas_stress.twin_relations import build_twin_pair
    R_matrix = np.asarray(R_matrix, dtype=float)
    if R_matrix.shape != (3, 3):
        raise ValueError(f"R_matrix must be (3, 3); got {R_matrix.shape}")
    return build_twin_pair(R_matrix, _sigma3_relation(phase, c_over_a=c_over_a,
                                                      hcp_twin=hcp_twin, frame=frame))


__all__ = ["assign_variants_common_reference", "build_sigma3_pair"]
