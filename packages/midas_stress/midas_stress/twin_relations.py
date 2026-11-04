"""Twin and coincidence-site orientation relationships, and the pair finding built on them.

Orientation relationships between two crystals (FCC Sigma3, the HCP deformation twins,
tetragonal and trigonal twins) belong here, with the other orientation/misorientation
code: ``midas_process_grains`` (twin labelling, cluster merging) and ``midas_defect``
(variant pairing) both build on this module instead of carrying their own copy.

Conventions
-----------
* Orientation matrices are **crystal -> lab**; a twin relation is ``O_B = O_A . T . S``
  with ``T`` the operator below in the CRYSTAL frame and ``S`` any proper symmetry
  rotation of the crystal.
* The crystal Cartesian frame of a hexagonal cell is a choice, see :data:`HCP_FRAMES`:
  ``"midas"`` (default; a* along x, the frame of ``Grains.csv`` and of ``midas_hkls``'s
  ``B`` matrix) or ``"crystallographic"`` (a1 along x). The operators' ANGLES do not
  depend on it, their AXES do (30 deg about c apart).
* Quaternions are ``(w, x, y, z)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TwinRelation",
    "HCP_FRAMES",
    "default_fcc_twin_relations",
    "default_cubic_twin_relations",
    "default_hcp_twin_relations",
    "default_tetragonal_twin_relations",
    "default_trigonal_twin_relations",
    "default_twin_relations_for",
    "hcp_hkil_to_cartesian_normal",
    "tetragonal_hkl_to_cartesian_normal",
    "relation_rotation_matrix",
    "relation_disorientation",
    "relation_axis_family",
    "build_twin_pair",
    "find_twin_partners",
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


#: Crystal Cartesian frames a hexagonal orientation can live in. They differ by
#: a 30° rotation about c, which leaves the 12 hexagonal symmetry operators and
#: every disorientation ANGLE unchanged but moves every plane normal, slip
#: direction and rotation AXIS.
#:
#: ``"midas"``            the frame of the MIDAS ``Grains.csv`` orientation
#:                        matrices (crystal→lab) and of ``midas_hkls``'s
#:                        ``B`` matrix (a port of ``CorrectHKLsLatC``):
#:                        ``G_cart = B·(h, k, l)`` with **a\* along x**, b\* in
#:                        the xy plane, c\* along z. Real-space a sits at
#:                        −30° (150°), b at 90°, c ∥ z.
#: ``"crystallographic"`` the textbook frame, real-space a₁ along x.
HCP_FRAMES = ("midas", "crystallographic")


def hcp_hkil_to_cartesian_normal(
    h: int, k: int, i: int, l: int, c_over_a: float,
    *, frame: str = "midas",
) -> np.ndarray:
    """Convert a Miller-Bravais ``(h k i l)`` plane to a unit cartesian normal.

    Parameters
    ----------
    frame : {"midas", "crystallographic"}
        Which crystal Cartesian frame the normal is expressed in (see
        :data:`HCP_FRAMES`). Orientation matrices read from a MIDAS
        ``Grains.csv`` (or produced by ``midas-index`` / ``midas-fit-grain``)
        live in the ``"midas"`` frame, which is the default. Using the
        ``"crystallographic"`` frame with them puts every normal 30° about c
        away from the true plane, and twin operators built from it find no
        twins (0 of 24 exact twins in a test; ``"midas"`` operators find 24 of
        24).

    ``"midas"`` (a = 1, reciprocal basis of ``midas_hkls`` ``_b_matrix``)::

        a* = (2/√3, 0, 0)    b* = (1/√3, 1, 0)    c* = (0, 0, a/c)
        n ∝ h·a* + k·b* + l·c* = ((2h + k)/√3, k, l/(c/a))

    ``"crystallographic"``::

        a₁ = (1, 0, 0)   a₂ = (-1/2, √3/2, 0)   c = (0, 0, c/a)
        a₁* = (1, 1/√3, 0)   a₂* = (0, 2/√3, 0)   c* = (0, 0, a/c)
        n ∝ h·a₁* + k·a₂* + l·c* = (h, (h + 2k)/√3, l/(c/a))

    Note ``i = -(h + k)`` is redundant and only used for naming.
    """
    if i != -(h + k):
        raise ValueError(
            f"Miller-Bravais index i={i} must equal -(h+k)={-(h+k)}"
        )
    if frame == "midas":
        n = np.array([
            (2.0 * float(h) + float(k)) / math.sqrt(3.0),
            float(k),
            float(l) / float(c_over_a),
        ], dtype=np.float64)
    elif frame == "crystallographic":
        n = np.array([
            float(h),
            (float(h) + 2.0 * float(k)) / math.sqrt(3.0),
            float(l) / float(c_over_a),
        ], dtype=np.float64)
    else:
        raise ValueError(f"frame must be one of {HCP_FRAMES}, got {frame!r}")
    return n / np.linalg.norm(n)


def default_hcp_twin_relations(
    c_over_a: float,
    systems: Tuple[str, ...] = (
        "tension_1012", "compression_1011",
        "tension_1121",  "compression_1122",
    ),
    *,
    frame: str = "midas",
) -> List[TwinRelation]:
    """Standard HCP twin operators (180° rotation about the K₁ plane normal).

    The four common HCP deformation twins are in the default list; the
    c/a-dependent actual misorientation (the one that EBSD/HEDM measures) is
    found by the symmetry-aware disorientation against the HCP point group
    (SG 194). For α-Ti (c/a = 1.587) these come out as (computed, not quoted):

      tension_1012     85.00° about <11-20>   (the dominant {10-12} tension twin)
      compression_1011 57.24° about <11-20>
      tension_1121     34.98° about <10-10>
      compression_1122 64.43° about <10-10>

    ``compression_2112`` (K₁ = {2-1-12}) is accepted by name but is NOT in the
    default list: {2-1-12} is the same plane family as {11-22}, so it
    duplicates ``compression_1122``. (Before this was fixed it was defined
    with the plane (2 1 -3 2), which is not a twin plane: 49.6° about an axis
    inclined 65° from c, and its docstring claimed 64.2°.)

    Each twin system is emitted as ONE operator per variant; the
    Miller-Bravais variants ((10-12), (01-12), (-1102), …) are recovered
    by the symmetry-aware misorientation check downstream.

    Parameters
    ----------
    c_over_a : float
        The c/a ratio of the hexagonal cell (e.g. 1.587 for α-Ti, 1.624
        for Mg, 1.593 for Zr, 1.886 for Cd, 1.624 for ideal hard-sphere).
    systems : tuple of str, optional
        Which twin systems to include. Defaults to the four above. Subset for
        material-specific analyses (e.g. Mg favours ``tension_1012``).
    frame : {"midas", "crystallographic"}, keyword-only
        Crystal Cartesian frame of the operators; see
        :func:`hcp_hkil_to_cartesian_normal` and :data:`HCP_FRAMES`. The
        default ``"midas"`` matches ``Grains.csv`` orientation matrices.
        Disorientation angles do not depend on it; the operator AXES do, and
        the wrong frame finds no twins.

    Returns
    -------
    list of :class:`TwinRelation`
    """
    if frame not in HCP_FRAMES:
        raise ValueError(f"frame must be one of {HCP_FRAMES}, got {frame!r}")
    representative_K1: Dict[str, Tuple[int, int, int, int]] = {
        "tension_1012":     (1,  0, -1,  2),
        "compression_1011": (1,  0, -1,  1),
        "compression_2112": (2, -1, -1,  2),   # {2-1-12} ≡ {11-22}
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
            n_cart = hcp_hkil_to_cartesian_normal(h, k, i, l, c_over_a,
                                                  frame=frame)
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
    hcp_frame: str = "midas",
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
    hcp_frame : {"midas", "crystallographic"}, optional
        Crystal Cartesian frame of the HCP operators (default ``"midas"``,
        the frame of ``Grains.csv`` orientation matrices). See
        :data:`HCP_FRAMES`.
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
        kw = {"c_over_a": float(c_over_a), "frame": hcp_frame}
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


# ===================================================================
#  Using a relation: matrix, disorientation, axis family, pairs
# ===================================================================

def relation_rotation_matrix(relation: TwinRelation) -> np.ndarray:
    """3x3 rotation matrix of ``relation.quaternion`` (the crystal-frame operator ``T``)."""
    from .orientation import quat_to_orient_mat
    q = np.asarray(relation.quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return np.asarray(quat_to_orient_mat(list(q)), dtype=np.float64).reshape(3, 3)


def relation_disorientation(relation: TwinRelation, space_group: int) -> Tuple[float, np.ndarray]:
    """``(angle_deg, axis)`` of the relation after reduction by the crystal point group.

    This is what a measured pair of grains shows, not the raw operator: the HCP twin
    operators are 180 deg rotations about the K1 normal, but the disorientation of a
    {10-12} pair is 85.0 deg (Ti) about a <11-20> axis; the FCC Sigma3 operator is 60 deg
    about <111>. ``axis`` is a unit vector in the CRYSTAL frame of the relation.
    """
    from .orientation import misorientation_om
    ang_rad, axis = misorientation_om(np.eye(3).ravel(), relation_rotation_matrix(relation).ravel(),
                                      space_group=space_group)
    axis = np.asarray(axis, dtype=np.float64)
    return float(np.degrees(ang_rad)), axis / np.linalg.norm(axis)


def relation_axis_family(relation: TwinRelation, space_group: int) -> np.ndarray:
    """All symmetry-equivalent disorientation axes of a relation, ``(n, 3)`` unit vectors,
    one per +-pair (sign fixed so the first non-zero component is positive).

    Built from the relation itself in its own frame, so an axis family can never be in a
    different crystal frame from the operator (the failure mode of a hand-written
    ``<1-100>`` table: 30 deg off for MIDAS orientations).
    """
    from .orientation import make_symmetries, quat_to_orient_mat
    _, axis = relation_disorientation(relation, space_group)
    n_sym, sym = make_symmetries(space_group)
    fam: List[np.ndarray] = []
    for q in sym[:n_sym]:
        R = np.asarray(quat_to_orient_mat(list(q)), dtype=np.float64).reshape(3, 3)
        v = R @ axis
        v = v / np.linalg.norm(v)
        nz = next((c for c in v if abs(c) > 1e-9), 1.0)
        if nz < 0:
            v = -v
        if not any(np.allclose(v, f, atol=1e-6) for f in fam):
            fam.append(v)
    return np.array(fam)


def build_twin_pair(R: np.ndarray, relation: TwinRelation) -> Tuple[np.ndarray, np.ndarray]:
    """``(R, R @ T)``: an orientation and its exact twin (operator on the crystal side)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3, 3); got {R.shape}")
    return R, R @ relation_rotation_matrix(relation)


def find_twin_partners(
    OM: np.ndarray,
    pos: np.ndarray,
    variant_labels: np.ndarray,
    relation: TwinRelation,
    space_group: int,
    *,
    k_NN: int = 10,
    misori_low: Optional[float] = None,
    misori_high: Optional[float] = None,
    window_halfwidth_deg: float = 5.0,
    axis_alignment_min: float = 0.9,
    matrix_label: int = 0,
    twin_label: int = 1,
    z_reliable: bool = True,
) -> Dict[str, np.ndarray]:
    """For each ``matrix_label`` grain, the spatially nearest ``twin_label`` grain whose
    disorientation matches ``relation``.

    A candidate passes if its crystal-symmetry-reduced disorientation angle is inside
    ``[misori_low, misori_high]`` (default: the relation's own disorientation angle
    +- ``window_halfwidth_deg``) AND its disorientation axis lies within
    ``axis_alignment_min`` (|dot|) of the relation's axis family
    (:func:`relation_axis_family`). The axis test is what separates a twin from any
    other pair with the right angle; it is only meaningful when ``OM`` and
    ``relation`` are in the same crystal frame (HCP: ``frame`` of
    :func:`default_hcp_twin_relations`).

    Parameters
    ----------
    OM : (n, 3, 3) crystal -> lab orientation matrices
    pos : (n, 3) positions (um)
    variant_labels : (n,) integer variant assignment
    z_reliable : if False pair on (x, y) only (FF-HEDM refined Z is unreliable)

    Returns
    -------
    dict with ``pairs`` (m, 2) matrix/twin indices, ``pair_distances`` (m,),
    ``pair_misori`` (m,) deg, ``pair_axis`` (m, 3) crystal-frame unit axis.
    """
    from scipy.spatial import cKDTree
    from .orientation import misorientation_om

    OM = np.asarray(OM, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    var = np.asarray(variant_labels, dtype=int)
    if OM.shape[0] != pos.shape[0] or pos.shape[0] != var.shape[0]:
        raise ValueError(f"OM ({OM.shape[0]}), pos ({pos.shape[0]}), variant_labels "
                         f"({var.shape[0]}) must agree in length")
    angle0, _ = relation_disorientation(relation, space_group)
    lo = angle0 - window_halfwidth_deg if misori_low is None else float(misori_low)
    hi = angle0 + window_halfwidth_deg if misori_high is None else float(misori_high)
    family = relation_axis_family(relation, space_group)

    matrix_idx = np.where(var == matrix_label)[0]
    twin_idx = np.where(var == twin_label)[0]
    empty = {"pairs": np.zeros((0, 2), dtype=int), "pair_distances": np.zeros(0),
             "pair_misori": np.zeros(0), "pair_axis": np.zeros((0, 3))}
    if matrix_idx.size == 0 or twin_idx.size == 0:
        return empty

    pos_sp = pos[:, :2] if not z_reliable else pos
    tree = cKDTree(pos_sp[twin_idx])
    k_eff = min(k_NN, twin_idx.size)
    dists, nn = tree.query(pos_sp[matrix_idx], k=k_eff)
    if k_eff == 1:
        dists, nn = dists[:, None], nn[:, None]
    nn_global = twin_idx[nn]

    pairs, p_dist, p_mis, p_axis = [], [], [], []
    for i, g in enumerate(matrix_idx):
        best = None
        for j in range(k_eff):
            t = int(nn_global[i, j])
            ang_rad, axis = misorientation_om(OM[g].ravel(), OM[t].ravel(), space_group=space_group)
            ang = float(np.degrees(ang_rad))
            if not (lo <= ang <= hi):
                continue
            axis = np.asarray(axis, dtype=np.float64)
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            if np.max(np.abs(family @ axis)) < axis_alignment_min:
                continue
            if best is None or dists[i, j] < best[0]:
                best = (float(dists[i, j]), t, ang, axis)
        if best is not None:
            pairs.append((int(g), best[1]))
            p_dist.append(best[0])
            p_mis.append(best[2])
            p_axis.append(best[3])
    return {"pairs": np.asarray(pairs, dtype=int).reshape(-1, 2),
            "pair_distances": np.asarray(p_dist, dtype=float),
            "pair_misori": np.asarray(p_mis, dtype=float),
            "pair_axis": np.asarray(p_axis, dtype=float).reshape(-1, 3)}
