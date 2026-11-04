"""Find spatially nearest Σ3-misoriented partner for each matrix grain."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import CrystalPhase
from .common_reference import _PHASE_TO_SPACE_GROUP


def find_sigma3_partners(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    variant_labels: NDArray[np.intp],
    k_NN: int = 10,
    misori_low: float | None = None,
    misori_high: float | None = None,
    axis_alignment_min: float = 0.9,
    phase: CrystalPhase = CrystalPhase.FCC,
    matrix_label: int = 0,
    twin_label: int = 1,
    z_reliable: bool = True,
    *,
    frame: str = "midas",
    c_over_a: float = 1.624,
    hcp_twin: str = "tension_1012",
) -> dict:
    """For each matrix grain, identify its nearest twin partner of the phase's Sigma3 relation.

    A candidate (matrix g, twin t) pair passes if
        * variant_labels[g] == matrix_label and variant_labels[t] == twin_label
        * disorientation(g, t) in [misori_low, misori_high] deg (default: 55-65 for FCC/BCC;
          for HCP the relation's own disorientation angle +- 5 deg)
        * |axis . n| >= axis_alignment_min for some n in the relation's axis family
          (FCC/BCC: <111>; HCP: the twin's own axis, e.g. <11-20> for {10-12}), with the
          family derived from the relation in ``frame`` so it cannot disagree with the
          orientations' crystal frame.

    The pair finding is :func:`midas_stress.twin_relations.find_twin_partners`.

    Default-window change for HCP: the former fixed 55-65 deg window is gone. It admitted the
    {10-11} twin (57.24 deg) and rejected {10-12} (85 deg) and {11-21} (35 deg). Pass
    ``hcp_twin="compression_1011"`` to find {10-11} pairs; the default relation is the {10-12} tension twin.

    Selectivity: the angle window does most of the work. For a family of three in-plane axes
    (HCP <11-20> or <10-10>) the alignment gate at 0.9 passes ~86% of in-plane directions, and
    FCC's four <111> lines pass ~40% of random axes; tighten ``axis_alignment_min`` (0.98 keeps
    ~38% of in-plane directions for HCP) when the angle window alone admits chance pairs.

    Parameters
    ----------
    OM : (n_grains, 3, 3)
    pos : (n_grains, 3) sample-frame positions
    variant_labels : (n_grains,) variant assignment
    k_NN : number of spatial neighbours to consider per matrix grain
    misori_low, misori_high : angular window in degrees (None = the default above)
    axis_alignment_min : minimum |dot(disori_axis, axis_family)|
    frame : {"midas", "crystallographic"}
        HCP only: the crystal frame ``OM`` lives in (default ``"midas"``: Grains.csv
        orientations). The disorientation axis is reported in that frame, so the family
        must be built in it; a 30 deg mismatch drops the alignment of an ideal axis to
        0.866, below the default 0.9 gate. Ignored for FCC/BCC.
    c_over_a, hcp_twin : HCP only, which twin ({10-12} tension by default) at which c/a.

    Returns
    -------
    dict with
        pairs            (n_pairs, 2)  matrix_idx, twin_idx
        pair_distances   (n_pairs,)    spatial distance
        pair_misori      (n_pairs,)    deg
        pair_axis        (n_pairs, 3)  disorientation axis (crystal frame)
    """
    from midas_stress.twin_relations import find_twin_partners
    from .common_reference import _sigma3_relation

    sg = _PHASE_TO_SPACE_GROUP[phase]
    rel = _sigma3_relation(phase, c_over_a=c_over_a, hcp_twin=hcp_twin, frame=frame)
    if phase is not CrystalPhase.HCP:
        misori_low = 55.0 if misori_low is None else misori_low
        misori_high = 65.0 if misori_high is None else misori_high
    return find_twin_partners(
        OM, pos, variant_labels, rel, sg, k_NN=k_NN, misori_low=misori_low,
        misori_high=misori_high, axis_alignment_min=axis_alignment_min,
        matrix_label=matrix_label, twin_label=twin_label, z_reliable=z_reliable)


__all__ = ["find_sigma3_partners"]
