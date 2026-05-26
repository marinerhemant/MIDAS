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
    "hcp_hkil_to_cartesian_normal",
    "tetragonal_hkl_to_cartesian_normal",
    "find_twin_pairs",
    "extend_symmetry_table_with_twin",
]


@dataclass
class TwinRelation:
    """One twin orientation relationship.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"FCC_Sigma3"``).
    quaternion : np.ndarray
        ``(4,)`` rotation quaternion in (w, x, y, z) layout.
    angle_deg : float
        Rotation angle for diagnostics.
    axis : tuple
        Rotation axis for diagnostics.
    """

    name: str
    quaternion: np.ndarray
    angle_deg: float
    axis: Tuple[float, float, float]


def default_fcc_twin_relations() -> List[TwinRelation]:
    """Return the four <111> 60° rotation operators for FCC Σ3 twins.

    Each <111> direction (normalised) gives a single 60° rotation operator.
    The axis-angle quaternion is ``(cos(θ/2), n·sin(θ/2))`` for unit axis ``n``.
    """
    half = math.pi / 6.0   # 60° / 2
    cw = math.cos(half)
    sw = math.sin(half)
    out: List[TwinRelation] = []
    for ax in [
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    ]:
        n = np.array(ax, dtype=np.float64)
        n /= np.linalg.norm(n)
        q = np.array([cw, sw * n[0], sw * n[1], sw * n[2]])
        out.append(TwinRelation(
            name=f"FCC_Sigma3_<{ax[0]}{ax[1]}{ax[2]}>",
            quaternion=q,
            angle_deg=60.0,
            axis=tuple(float(x) for x in n),
        ))
    return out


def default_cubic_twin_relations(
    include: Tuple[str, ...] = ("Sigma3", "Sigma9", "Sigma27a", "Sigma27b"),
    lattice: str = "FCC",
) -> List[TwinRelation]:
    """Common CSL twin operators for any cubic crystal (FCC or BCC).

    CSL grain boundaries are characterised by the misorientation between
    grains, independent of which sub-lattice (FCC vs BCC) populates the
    Bravais lattice. The five "common low-Σ" CSL boundaries observed in
    annealed FCC/BCC metals (Cu, Ni, α-Fe, …):

      Σ3   — 60.00° about <111>     (the FCC annealing twin)
      Σ9   — 38.94° about <110>     (Σ3-Σ3 product)
      Σ11  — 50.48° about <110>
      Σ27a — 31.59° about <110>
      Σ27b — 35.43° about <210>

    Parameters
    ----------
    include : tuple of str, optional
        Which CSL boundaries to enumerate. Defaults to the four most
        common (Σ3, Σ9, Σ27a, Σ27b). Σ11 is less common in HEDM data
        but available on request.
    lattice : str, optional
        ``"FCC"`` (default) or ``"BCC"`` — only affects the operator
        ``name`` prefix. Operators are identical.

    Returns
    -------
    list of :class:`TwinRelation`
        One operator per CSL boundary × variant. Σ3 contributes 4 ops
        (one per <111>); the rest contribute 1 op each (the symmetry-
        equivalent variants are recovered by the symmetry-aware
        misorientation check downstream).
    """
    out: List[TwinRelation] = []

    if "Sigma3" in include:
        for ax in [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]:
            n = np.array(ax, dtype=np.float64) / np.linalg.norm(ax)
            half = math.radians(60.0 / 2)
            cw, sw = math.cos(half), math.sin(half)
            q = np.array([cw, sw * n[0], sw * n[1], sw * n[2]])
            out.append(TwinRelation(
                name=f"{lattice}_Sigma3_<{ax[0]}{ax[1]}{ax[2]}>",
                quaternion=q, angle_deg=60.0,
                axis=tuple(float(x) for x in n),
            ))

    csl_specs = [
        ("Sigma9",  38.94, (1, 1, 0)),
        ("Sigma11", 50.48, (1, 1, 0)),
        ("Sigma27a", 31.59, (1, 1, 0)),
        ("Sigma27b", 35.43, (2, 1, 0)),
    ]
    for sigma_name, angle_deg, axis in csl_specs:
        if sigma_name not in include:
            continue
        n = np.array(axis, dtype=np.float64) / np.linalg.norm(axis)
        half = math.radians(angle_deg / 2.0)
        cw, sw = math.cos(half), math.sin(half)
        q = np.array([cw, sw * n[0], sw * n[1], sw * n[2]])
        out.append(TwinRelation(
            name=f"{lattice}_{sigma_name}",
            quaternion=q, angle_deg=angle_deg,
            axis=tuple(float(x) for x in n),
        ))
    return out


def hcp_hkil_to_cartesian_normal(
    h: int, k: int, i: int, l: int, c_over_a: float,
) -> np.ndarray:
    """Convert a Miller-Bravais ``(h k i l)`` plane to a unit cartesian normal.

    Cartesian basis convention:
        a₁ = (1, 0, 0)
        a₂ = (-1/2, √3/2, 0)
        c  = (0, 0, c/a)

    Reciprocal:
        a₁* = (1, 1/√3, 0) / a
        a₂* = (0, 2/√3, 0) / a
        c*  = (0, 0, a/c)

    The plane normal n = h·a₁* + k·a₂* + l·c* (a is set to 1 in this
    cartesian frame; only c/a matters). Note ``i = -(h + k)`` is
    redundant and only used for naming.
    """
    if i != -(h + k):
        raise ValueError(
            f"Miller-Bravais index i={i} must equal -(h+k)={-(h+k)}"
        )
    n = np.array([
        float(h),
        (float(h) + 2.0 * float(k)) / math.sqrt(3.0),
        float(l) / float(c_over_a),
    ], dtype=np.float64)
    return n / np.linalg.norm(n)


def default_hcp_twin_relations(
    c_over_a: float,
    systems: Tuple[str, ...] = (
        "tension_1012", "compression_1011", "compression_2112",
        "tension_1121",  "compression_1122",
    ),
) -> List[TwinRelation]:
    """Standard HCP twin operators (180° rotation about the K₁ plane normal).

    All five common HCP deformation twins are supported; the c/a-dependent
    actual misorientation (the one that EBSD/HEDM measures) is found by
    the symmetry-aware disorientation against the HCP point group
    (SG 194). For α-Ti (c/a ≈ 1.587) these come out as:

      tension_1012  ≈ 85.0° around <1-210>   (the dominant {10-12} tension twin)
      compression_1011 ≈ 57.4° around <1-210>
      compression_2112 ≈ 64.2° around <1-210>
      tension_1121     ≈ 35.0° around <1-210>
      compression_1122 ≈ 64.4° around <-1100>

    Each twin system is emitted as ONE operator per variant; the
    Miller-Bravais variants ((10-12), (01-12), (-1102), …) are recovered
    by the symmetry-aware misorientation check downstream.

    Parameters
    ----------
    c_over_a : float
        The c/a ratio of the hexagonal cell (e.g. 1.587 for α-Ti, 1.624
        for Mg, 1.593 for Zr, 1.886 for Cd, 1.624 for ideal hard-sphere).
    systems : tuple of str, optional
        Which twin systems to include. Defaults to all five. Subset for
        material-specific analyses (e.g. Mg favours ``tension_1012``).

    Returns
    -------
    list of :class:`TwinRelation`
    """
    representative_K1: Dict[str, Tuple[int, int, int, int]] = {
        "tension_1012":     (1,  0, -1,  2),
        "compression_1011": (1,  0, -1,  1),
        "compression_2112": (2,  1, -3,  2),   # (2 1 -3 2) ≡ {2-1-12}
        "tension_1121":     (1,  1, -2,  1),
        "compression_1122": (1,  1, -2,  2),
    }
    # The 6 symmetry-equivalent K1 variants per system are obtained by
    # applying the hex 6-fold (z) and the (h, k, i) → (-h, -k, -i, l)
    # mirror. For VARIANT-LEVEL labelling we enumerate them so the
    # downstream twin_type column reports which specific K1 variant the
    # observed twin pair matches. The disorientation under SG 194 is
    # the same for all six variants (= ~85° for {10-12}/Ti); the
    # **operator** is the 180° rotation about THAT specific K1 normal,
    # so different variants give labelled-different (i,j) pairs.
    out: List[TwinRelation] = []
    for sys_name in systems:
        if sys_name not in representative_K1:
            raise ValueError(
                f"Unknown HCP twin system {sys_name!r}; valid: "
                f"{sorted(representative_K1)}"
            )
        h0, k0, i0, l0 = representative_K1[sys_name]
        # 6 variants by 6-fold rotation of (h, k) around z + mirror in c
        # Bravais convention: (h k i l) → permutation under 6-fold gives:
        #   v0: (h, k, i, l)
        #   v1: (-k, h+k, -h, l)
        #   v2: (-h-k, h, k, l)
        #   v3: (-h, -k, -i, l)
        #   v4: (k, -h-k, h, l)
        #   v5: (h+k, -h, -k, l)
        variants = []
        for h, k in [
            (h0, k0),
            (-k0, h0 + k0),
            (-(h0 + k0), h0),
            (-h0, -k0),
            (k0, -(h0 + k0)),
            (h0 + k0, -h0),
        ]:
            i = -(h + k)
            variants.append((h, k, i, l0))
        for (h, k, i, l) in variants:
            n_cart = hcp_hkil_to_cartesian_normal(h, k, i, l, c_over_a)
            q = np.array([0.0, n_cart[0], n_cart[1], n_cart[2]])
            # Variant label: paren-tagged Miller-Bravais quad
            def _idx(v):
                return f"{v}" if v >= 0 else f"-{abs(v)}"
            label = f"HCP_{sys_name}_K1({_idx(h)}{_idx(k)}{_idx(i)}{_idx(l)})"
            out.append(TwinRelation(
                name=label,
                quaternion=q,
                angle_deg=180.0,
                axis=tuple(float(x) for x in n_cart),
            ))
    return out


def tetragonal_hkl_to_cartesian_normal(
    h: int, k: int, l: int, c_over_a: float,
) -> np.ndarray:
    """Convert a tetragonal Miller index ``(h k l)`` to a unit cartesian normal.

    Cartesian basis convention (a = b for tetragonal):
        a₁ = (1, 0, 0)
        a₂ = (0, 1, 0)
        c  = (0, 0, c/a)

    Reciprocal:
        a₁* = (1, 0, 0) / a
        a₂* = (0, 1, 0) / a
        c*  = (0, 0, a/c)

    The plane normal is ``n = h·a₁* + k·a₂* + l·c*`` = ``(h, k, l/(c/a))``
    (with ``a`` set to 1 in this cartesian frame; only c/a matters).
    """
    n = np.array([float(h), float(k), float(l) / float(c_over_a)],
                 dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError(f"Degenerate {(h, k, l)} plane (zero normal)")
    return n / norm


def default_tetragonal_twin_relations(
    c_over_a: float,
    systems: Tuple[str, ...] = ("twin_101", "twin_011", "twin_112", "twin_103"),
) -> List[TwinRelation]:
    """Standard tetragonal twin operators (180° about the K₁ plane normal).

    Common deformation/transformation twins in tetragonal crystals
    (P4/mmm, I4/mmm, P4₂/mnm, P4₂/nmc — c/a < 1 in L1₀ FePt/FePd/MnAl,
    c/a > 1 in β-Sn, c/a ≈ 1 in CuPt, etc.):

      twin_101 — {101} polysynthetic twin, dominant in L1₀ FePt/FePd
      twin_011 — {011} variant of {101} by 4-fold symmetry
      twin_112 — {112} twin (observed in tetragonal ZrO₂)
      twin_103 — {103} twin (rarer; observed in some L1₀ alloys)

    Each operator is the 180° rotation about the K₁ plane normal; the
    **measured** disorientation under the tetragonal point group
    (4/mmm = 8 ops) depends on c/a. For FePt c/a = 0.967, the {101}
    twin reads as ~94° around <010>; for ZrO₂ c/a = 1.025 the {112}
    twin reads as ~83° around <-110>.

    Parameters
    ----------
    c_over_a : float
        c/a ratio of the tetragonal cell.
    systems : tuple of str, optional
        Which twin systems to include. Defaults to the four most
        commonly-observed.

    Returns
    -------
    list of :class:`TwinRelation`
    """
    twin_planes: Dict[str, Tuple[int, int, int]] = {
        "twin_101": (1, 0, 1),
        "twin_011": (0, 1, 1),
        "twin_112": (1, 1, 2),
        "twin_103": (1, 0, 3),
        "twin_110": (1, 1, 0),     # {110} twin (90° about c in cubic-limit)
    }
    out: List[TwinRelation] = []
    for sys_name in systems:
        if sys_name not in twin_planes:
            raise ValueError(
                f"Unknown tetragonal twin system {sys_name!r}; valid: "
                f"{sorted(twin_planes)}"
            )
        h, k, l = twin_planes[sys_name]
        n_cart = tetragonal_hkl_to_cartesian_normal(h, k, l, c_over_a)
        q = np.array([0.0, n_cart[0], n_cart[1], n_cart[2]])
        out.append(TwinRelation(
            name=f"Tetragonal_{sys_name}_K1({h}{k}{l})",
            quaternion=q,
            angle_deg=180.0,
            axis=tuple(float(x) for x in n_cart),
        ))
    return out


def default_trigonal_twin_relations() -> List[TwinRelation]:
    """Common trigonal twin: 60° rotation around the c-axis.

    Observed in corundum α-Al₂O₃ (SG 167), calcite, ilmenite, hematite,
    and other rhombohedral oxides. The c-axis 60° rotation is the
    rhombohedral twin (Σ3 of the rhombohedral lattice).
    """
    half = math.radians(60.0 / 2)
    cw, sw = math.cos(half), math.sin(half)
    q = np.array([cw, 0.0, 0.0, sw])
    return [TwinRelation(
        name="Trigonal_c60",
        quaternion=q, angle_deg=60.0,
        axis=(0.0, 0.0, 1.0),
    )]


def default_twin_relations_for(
    space_group: int,
    *,
    c_over_a: Optional[float] = None,
    hcp_systems: Optional[Tuple[str, ...]] = None,
    cubic_systems: Optional[Tuple[str, ...]] = None,
    tetragonal_systems: Optional[Tuple[str, ...]] = None,
) -> List[TwinRelation]:
    """Dispatcher: return the right twin operator set for a given space group.

    Parameters
    ----------
    space_group : int
        IT space group number.
    c_over_a : float, optional
        c/a ratio. Required for hexagonal (SG 168-194) and tetragonal
        (SG 75-142). Ignored otherwise.
    hcp_systems : tuple of str, optional
        Override the default HCP twin system list. See
        :func:`default_hcp_twin_relations`.
    cubic_systems : tuple of str, optional
        Override the default cubic CSL boundary list. See
        :func:`default_cubic_twin_relations`.
    tetragonal_systems : tuple of str, optional
        Override the default tetragonal twin system list. See
        :func:`default_tetragonal_twin_relations`.

    Returns
    -------
    list of :class:`TwinRelation`
        Empty list for crystal systems with no standard twin operators
        in this dispatcher (orthorhombic, monoclinic, triclinic).
    """
    # Cubic: 195-230
    if 195 <= space_group <= 230:
        lattice = "BCC" if space_group in (
            197, 199, 204, 206, 211, 214, 217, 220, 229, 230
        ) else "FCC"
        kw = {"lattice": lattice}
        if cubic_systems is not None:
            kw["include"] = cubic_systems
        return default_cubic_twin_relations(**kw)
    # Hexagonal: 168-194
    if 168 <= space_group <= 194:
        if c_over_a is None:
            raise ValueError(
                f"Hexagonal SG {space_group} needs c_over_a for HCP twin relations"
            )
        kw = {"c_over_a": float(c_over_a)}
        if hcp_systems is not None:
            kw["systems"] = hcp_systems
        return default_hcp_twin_relations(**kw)
    # Trigonal: 143-167
    if 143 <= space_group <= 167:
        return default_trigonal_twin_relations()
    # Tetragonal: 75-142
    if 75 <= space_group <= 142:
        if c_over_a is None:
            raise ValueError(
                f"Tetragonal SG {space_group} needs c_over_a for twin relations"
            )
        kw = {"c_over_a": float(c_over_a)}
        if tetragonal_systems is not None:
            kw["systems"] = tetragonal_systems
        return default_tetragonal_twin_relations(**kw)
    # Orthorhombic / monoclinic / triclinic: no defaults yet
    return []


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
    hkl_int = np.asarray(hkl_table_int[:, :3], dtype=np.int64)
    new_perm = np.full((n_sym, n_hkls), -1, dtype=np.int64)
    for s in range(n_sym):
        rotated = np.rint(hkl_int @ new_R[s].T).astype(np.int64)
        for k in range(n_hkls):
            key = (int(rotated[k, 0]), int(rotated[k, 1]), int(rotated[k, 2]))
            new_perm[s, k] = hkl_to_row.get(key, -1)

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
