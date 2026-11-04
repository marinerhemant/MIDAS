"""Twin post-processor.

Given the grains output from Phase 5, walk pairs of grains satisfying a
user-supplied twin orientation relationship (default for FCC: 60° about
⟨111⟩) and merge them with the **same** Phase-2 / Phase-3 logic, but with
a twin-extended permutation table: instead of point-group ops, we use
``T · S_s`` for each twin op ``T`` and each point-group op ``S_s``.

A twin-merge survives the §3.6 paper sanity check ("the framework will only
qualify twins if the grain size calculated from overlapping peaks is within
5 µm of the sum of grain sizes calculated from non-overlapping peaks") only
if the size-consistency check passes. Failing pairs are reported but not
merged.

This module is intentionally a small wrapper: the heavy lifting reuses
``compute.symmetry``, ``compute.canonicalize``, and ``compute.refine_cluster``.
The only twin-specific bits are the OR enumeration and the permutation-table
extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .canonicalize import _quat_mul, _quat_inv
from .symmetry import SymmetryTable


__all__ = [
    "TwinRelation",
    "default_fcc_twin_relations",
    "default_cubic_twin_relations",
    "default_hcp_twin_relations",
    "default_tetragonal_twin_relations",
    "default_trigonal_twin_relations",
    "default_twin_relations_for",
    "HCP_FRAMES",
    "hcp_hkil_to_cartesian_normal",
    "tetragonal_hkl_to_cartesian_normal",
    "find_twin_pairs",
    "extend_symmetry_table_with_twin",
]


# The relation definitions moved to midas_stress (orientation code lives there); re-exported so
# existing imports of these names from this module keep working.
from midas_stress.twin_relations import (  # noqa: F401,E402
    HCP_FRAMES,
    TwinRelation,
    default_cubic_twin_relations,
    default_fcc_twin_relations,
    default_hcp_twin_relations,
    default_tetragonal_twin_relations,
    default_trigonal_twin_relations,
    default_twin_relations_for,
    hcp_hkil_to_cartesian_normal,
    tetragonal_hkl_to_cartesian_normal,
)


def extend_symmetry_table_with_twin(
    sym_table: SymmetryTable,
    twin: TwinRelation,
    hkl_table_real: np.ndarray,
    hkl_table_int: np.ndarray,
    hkl_to_row: dict,
) -> SymmetryTable:
    """Build a new SymmetryTable whose ops are ``T · S_s`` for each ``S_s``
    in the original.

    Used when twin-merging two grains: the alignment search becomes
    ``argmin_S angle(O_rep^T · O_other · T · S)`` over the extended op set.
    """
    from midas_stress.orientation import quat_to_orient_mat
    sym_quats_orig = sym_table.ops_quat.cpu().numpy()
    n_sym = sym_quats_orig.shape[0]
    n_hkls = sym_table.n_hkls
    twin_q_t = torch.from_numpy(twin.quaternion).to(sym_table.ops_quat.dtype)

    # Compose T · S_s for each S_s.
    new_q = np.empty_like(sym_quats_orig)
    new_R = np.empty((n_sym, 3, 3), dtype=np.float64)
    for s in range(n_sym):
        S = torch.from_numpy(sym_quats_orig[s]).to(twin_q_t.dtype)
        composed = _quat_mul(twin_q_t, S).cpu().numpy()
        composed = composed / np.linalg.norm(composed)
        new_q[s] = composed
        new_R[s] = np.asarray(quat_to_orient_mat(composed)).reshape(3, 3)

    # Permutations: same construction as the parent table but with the
    # composed rotation matrix.
    # Rotate the CARTESIAN g vectors (HklTable.real[:, :3]); rotating the integer
    # Miller indices by a Cartesian matrix is only valid for a cubic basis.
    from .symmetry import hkl_row_images
    new_perm = np.full((n_sym, n_hkls), -1, dtype=np.int64)
    for s in range(n_sym):
        new_perm[s] = hkl_row_images(new_R[s], hkl_table_real, hkl_table_int,
                                     hkl_to_row)

    return SymmetryTable(
        space_group=sym_table.space_group,
        n_sym=n_sym,
        ops_quat=torch.from_numpy(new_q).to(sym_table.ops_quat),
        ops_R=torch.from_numpy(new_R).to(sym_table.ops_R),
        hkl_perm=torch.from_numpy(new_perm).to(sym_table.hkl_perm),
        n_hkls=n_hkls,
    )


def find_twin_pairs(
    grain_quats: np.ndarray,
    space_group: int,
    twins: List[TwinRelation],
    *,
    tol_rad: float = math.radians(0.5),
) -> List[Tuple[int, int, str]]:
    """Find pairs ``(i, j, twin_name)`` whose orientations satisfy *some*
    user-supplied twin relation within ``tol_rad``.

    Misorientation is computed via the symmetry-aware reducer; for each
    pair we walk the supplied twin list and keep the smallest residual.
    """
    from midas_stress.orientation import misorientation_quat_batch

    pairs: List[Tuple[int, int, str]] = []
    n = grain_quats.shape[0]
    if n < 2 or not twins:
        return pairs

    # For each twin op T, ask: misorientation between O_i and O_j · T_inv?
    # Equivalent to checking misorientation between (O_i · T) and O_j.
    for tw in twins:
        # Premultiply quats by twin op for one half of the population.
        T = tw.quaternion
        rotated = np.empty_like(grain_quats)
        T_t = torch.from_numpy(T)
        for k in range(n):
            qk = torch.from_numpy(grain_quats[k])
            r = _quat_mul(qk, T_t).numpy()
            rotated[k] = r / np.linalg.norm(r)

        for i in range(n):
            for j in range(i + 1, n):
                m = misorientation_quat_batch(
                    grain_quats[i:i+1], rotated[j:j+1], space_group,
                )
                if float(m[0]) < tol_rad:
                    pairs.append((i, j, tw.name))
    return pairs
