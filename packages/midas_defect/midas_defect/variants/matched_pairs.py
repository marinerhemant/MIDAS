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
    misori_low: float = 55.0,
    misori_high: float = 65.0,
    axis_alignment_min: float = 0.9,
    phase: CrystalPhase = CrystalPhase.FCC,
    matrix_label: int = 0,
    twin_label: int = 1,
    z_reliable: bool = True,
) -> dict:
    """For each matrix grain, identify its nearest Σ3-misoriented twin partner.

    A candidate (matrix g, twin t) pair passes if
        * variant_labels[g] == matrix_label and variant_labels[t] == twin_label
        * disorientation(g, t) in [misori_low, misori_high] deg
        * |axis . n_sigma3| >= axis_alignment_min  where n_sigma3 is
          - FCC/BCC: any <111>
          - HCP:     <1-100>

    Parameters
    ----------
    OM : (n_grains, 3, 3)
    pos : (n_grains, 3) sample-frame positions
    variant_labels : (n_grains,) variant assignment
    k_NN : number of spatial neighbours to consider per matrix grain
    misori_low, misori_high : angular window in degrees for the Sigma3 angle
    axis_alignment_min : minimum |dot(disori_axis, sigma3_axis_family)|

    Returns
    -------
    dict with
        pairs            (n_pairs, 2)  matrix_idx, twin_idx
        pair_distances   (n_pairs,)    spatial distance
        pair_misori      (n_pairs,)    deg
        pair_axis        (n_pairs, 3)  disorientation axis (sample frame)
    """
    import midas_stress.orientation as o
    from scipy.spatial import cKDTree

    OM = np.asarray(OM, dtype=float)
    pos = np.asarray(pos, dtype=float)
    var = np.asarray(variant_labels, dtype=int)
    if OM.shape[0] != pos.shape[0] or pos.shape[0] != var.shape[0]:
        raise ValueError(
            f"OM ({OM.shape[0]}), pos ({pos.shape[0]}), variant_labels ({var.shape[0]}) "
            "must agree in length"
        )
    sg = _PHASE_TO_SPACE_GROUP[phase]

    matrix_idx = np.where(var == matrix_label)[0]
    twin_idx = np.where(var == twin_label)[0]
    if matrix_idx.size == 0 or twin_idx.size == 0:
        return {
            "pairs": np.zeros((0, 2), dtype=int),
            "pair_distances": np.zeros(0),
            "pair_misori": np.zeros(0),
            "pair_axis": np.zeros((0, 3)),
        }

    # For FF-HEDM refined Z is unreliable (+/-210 um); spatial pairing in 3D then
    # mis-pairs. With z_reliable=False, pair on the trustworthy in-layer (x,y) only.
    # (AUDIT_2026-06-23.md). Pass the layer index as pos[:,2] and z_reliable=False.
    pos_sp = pos[:, :2] if not z_reliable else pos
    tree = cKDTree(pos_sp[twin_idx])
    k_eff = min(k_NN, twin_idx.size)
    dists, nn = tree.query(pos_sp[matrix_idx], k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        nn = nn[:, None]
    # Map back to global grain indices.
    nn_global = twin_idx[nn]

    pairs: list[tuple[int, int]] = []
    p_dist: list[float] = []
    p_misori: list[float] = []
    p_axis: list[NDArray[np.floating]] = []

    sigma3_axes = _sigma3_axis_family(phase)

    for i, g in enumerate(matrix_idx):
        chosen = None
        chosen_d = np.inf
        chosen_misori = np.nan
        chosen_axis = None
        for j in range(k_eff):
            t = int(nn_global[i, j])
            ang_rad, axis = o.misorientation_om(
                OM[g].ravel(), OM[t].ravel(), space_group=sg
            )
            ang_deg = float(np.degrees(ang_rad))
            if not (misori_low <= ang_deg <= misori_high):
                continue
            axis = np.asarray(axis, dtype=float)
            axis_n = axis / max(np.linalg.norm(axis), 1e-12)
            aligns = np.max(np.abs(sigma3_axes @ axis_n))
            if aligns < axis_alignment_min:
                continue
            if dists[i, j] < chosen_d:
                chosen = t
                chosen_d = float(dists[i, j])
                chosen_misori = ang_deg
                chosen_axis = axis_n
        if chosen is not None:
            pairs.append((int(g), int(chosen)))
            p_dist.append(chosen_d)
            p_misori.append(chosen_misori)
            p_axis.append(chosen_axis)

    return {
        "pairs": np.asarray(pairs, dtype=int).reshape(-1, 2),
        "pair_distances": np.asarray(p_dist, dtype=float),
        "pair_misori": np.asarray(p_misori, dtype=float),
        "pair_axis": np.asarray(p_axis, dtype=float).reshape(-1, 3),
    }


def _sigma3_axis_family(phase: CrystalPhase) -> NDArray[np.floating]:
    """All symmetry-equivalent Sigma3 axes for the phase."""
    if phase in (CrystalPhase.FCC, CrystalPhase.BCC):
        # Four <111> axes, both senses.
        axes = np.array(
            [
                [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            ],
            dtype=float,
        )
        return axes / np.linalg.norm(axes, axis=1, keepdims=True)
    if phase is CrystalPhase.HCP:
        # Three <1-100> axes; convert to Cartesian via HCP basis (uses Mg c/a
        # as a default for axis testing; c/a doesn't affect basal-plane axes).
        from ..phases.hcp import direction_bravais_to_cart

        axes_4 = [(1, -1, 0, 0), (-1, 0, 1, 0), (0, 1, -1, 0)]
        axes_cart = np.stack(
            [direction_bravais_to_cart(*a, c_over_a=1.624) for a in axes_4], axis=0
        )
        return axes_cart
    raise ValueError(f"unknown phase {phase!r}")


__all__ = ["find_sigma3_partners"]
