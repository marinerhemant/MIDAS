"""Slip-system / Schmid-factor / resolved-shear-stress analysis.

Built on top of the per-grain stress tensors produced by the rest of
``midas_stress``, this module provides the building blocks for
publication-ready slip-system analysis:

- Curated slip-system families for FCC, BCC, and HCP crystals
  (octahedral, pencil, basal/prismatic/pyramidal etc.).
- Batched computation of slip-plane normals and slip directions in the
  lab frame for a set of grains.
- Schmid factor for a uniaxial load
  :math:`m = (\\hat n \\cdot \\hat\\ell)(\\hat b \\cdot \\hat\\ell)`.
- General resolved shear stress on each (plane, direction) from the
  full stress tensor
  :math:`\\tau = \\hat n^\\top \\,\\sigma\\, \\hat b`
  (works for arbitrary biaxial / shear / non-uniform loads).
- Dominant / top-K active systems per grain with vectorised ranking.
- Activity classification from a CRSS threshold and yield-proximity
  ranking across a polycrystal (i.e. "which grains yield first").
- Taylor factor for uniaxial loading (isostrain, upper bound).

All routines accept arrays of N grains and broadcast cleanly. Inputs
are in SI-compatible units (stress in the same units as the CRSS —
MPa is recommended for CRSS work).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ===================================================================
#  Slip-system databases
# ===================================================================
#
# Each family is a list of (plane_normal, slip_direction) tuples in
# the crystal (hexagonal: orthonormal) frame. Plane normals are given
# in Miller indices and slip directions in direction indices; we
# normalise on construction so the stored vectors are unit vectors.

def _normalise_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


# --- FCC ------------------------------------------------------------
# Populated below via _systems_from_plane_direction after helpers are defined.
_FCC_111_110_RAW = None  # assigned after helpers are defined

# --- BCC ------------------------------------------------------------
#
# BCC slip systems are generated algorithmically to avoid sign /
# permutation mistakes in hand-coded tables. Each family lists the
# *distinct* {hkl}<uvw> systems, with plane normals having unique sign
# (i.e. (1,1,0) and (-1,-1,0) are merged) and a single independent
# Burgers vector per sign equivalence class.

def _unique_directions(candidates, *, antiparallel_equiv=True):
    """Return the sign-reduced set of integer direction vectors.

    Two vectors are considered equivalent if they are antiparallel.
    """
    seen = []
    out = []
    for v in candidates:
        v = np.asarray(v, dtype=int)
        if np.all(v == 0):
            continue
        # Reduce sign: the first non-zero component is made positive.
        for c in v:
            if c > 0:
                key = tuple(v)
                break
            if c < 0:
                key = tuple(-v)
                break
        if key in seen:
            continue
        seen.append(key)
        out.append(np.array(key, dtype=float))
    return out


def _distinct_hkl_perms(indices, *, antiparallel_equiv=True):
    """All unique permutations with independent sign choices for a (|h|,|k|,|l|).

    Returns vectors where the first non-zero entry is positive.
    """
    from itertools import permutations
    h, k, l = [abs(int(x)) for x in indices]
    base_perms = set(permutations((h, k, l)))
    out = []
    seen = set()
    for perm in base_perms:
        for sh in (1, -1):
            for sk in (1, -1):
                for sl in (1, -1):
                    v = np.array([perm[0] * sh, perm[1] * sk, perm[2] * sl])
                    if np.all(v == 0):
                        continue
                    for c in v:
                        if c > 0:
                            key = tuple(v)
                            break
                        if c < 0:
                            key = tuple(-v)
                            break
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(np.array(key, dtype=float))
    return out


def _systems_from_plane_direction(plane_family, dir_family):
    """Enumerate (n_hkl, b_uvw) pairs where n · b = 0.

    For each plane in the sign-reduced {hkl} family, include every
    sign-reduced <uvw> that is orthogonal to it. Antiparallel
    Burgers vectors share a slip *system* (slip direction is
    unpolarised), so only one of each antiparallel pair is kept.
    """
    planes = _distinct_hkl_perms(plane_family)
    dirs = _distinct_hkl_perms(dir_family)
    systems = []
    for n in planes:
        for b in dirs:
            if abs(float(n @ b)) < 1e-10:
                systems.append([n, b])
    return np.array(systems, dtype=np.float64)


_FCC_111_110_RAW = _systems_from_plane_direction((1, 1, 1), (1, 1, 0))
_BCC_110_111_RAW = _systems_from_plane_direction((1, 1, 0), (1, 1, 1))
_BCC_112_111_RAW = _systems_from_plane_direction((1, 1, 2), (1, 1, 1))
_BCC_123_111_RAW = _systems_from_plane_direction((1, 2, 3), (1, 1, 1))


# --- HCP ------------------------------------------------------------
#
# HCP slip systems are built geometrically (not via Miller-Bravais
# index arithmetic) to sidestep the numerous sign-convention pitfalls
# when converting 4-index notation to Cartesian.
#
# Crystal frame convention:
#   a1 = (1, 0, 0)  a
#   a2 = (-1/2, sqrt(3)/2, 0)  a
#   c  = (0, 0, c/a)
#
# The three equivalent basal <a> directions are a1, a2, and -(a1+a2),
# 120° apart in the basal plane. Prismatic-plane normals are the three
# in-basal-plane unit vectors perpendicular to those <a> directions
# (i.e. c × d_i).

_HCP_A_DIRS = np.array([
    (1.0,                0.0,             0.0),
    (-0.5,              np.sqrt(3) / 2,   0.0),
    (-0.5,             -np.sqrt(3) / 2,   0.0),
])


def _basal_normal() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


def _prismatic_normals() -> np.ndarray:
    """Three unit normals to prismatic planes (all in basal plane).

    Each is c × <a>_i for the three basal <a> directions.
    """
    c_hat = _basal_normal()
    out = []
    for d in _HCP_A_DIRS:
        n = np.cross(c_hat, d)
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def _pyramidal_a_systems(c_over_a: float) -> np.ndarray:
    """6 first-order pyramidal <a> systems.

    The first-order pyramidal planes of the {10-11} family share an
    <a> slip direction with prismatic planes but tilt toward ±c. Each
    of the three <a> directions has two associated pyramidal planes
    (tilted toward +c and -c).
    """
    a = 1.0
    c = c_over_a
    out = []
    for d in _HCP_A_DIRS:
        # Prismatic normal perpendicular to d (in basal plane)
        m = np.cross(_basal_normal(), d) / np.linalg.norm(
            np.cross(_basal_normal(), d)
        )
        # Pyramidal plane contains d and is tilted such that its
        # normal is m (in basal plane) combined with ±c component.
        # Mathematically, normal = cos(phi)*m ± sin(phi)*c, where
        # phi is determined by the plane's geometry. For the
        # {10-11} family, the plane passes through (0,0,c) and an
        # a-vertex at (a/sqrt(3), 0, 0) scaled by the prismatic
        # distance — giving tan(phi) = (a * sqrt(3)/2) / c.
        # We construct n directly by requiring orthogonality to d
        # and to a specific in-plane vector (d x c rotated).
        #
        # Simpler construction: pyramidal plane contains d and the
        # c-axis-shifted version d' = d + (a*sqrt(3)/c) * c (pyramid
        # apex). Normal = d × d'.
        d_tilted = d + np.array([0.0, 0.0, np.sqrt(3) * a / c])
        n_plus = np.cross(d, d_tilted)
        n_plus /= np.linalg.norm(n_plus)
        # The -c tilted variant:
        d_tilted_neg = d - np.array([0.0, 0.0, np.sqrt(3) * a / c])
        n_minus = np.cross(d, d_tilted_neg)
        n_minus /= np.linalg.norm(n_minus)
        out.append((n_plus, d))
        out.append((n_minus, d))
    return np.array(out)


def _pyramidal_ca_systems(c_over_a: float) -> np.ndarray:
    """Second-order pyramidal <c+a> systems ({11-22}<11-23>, 12 systems).

    Construction:
    - 6 <c+a> Burgers vectors: each of the 6 basal <a> directions
      (three ``<a>`` directions ± their antiparallels — these six
      vectors are ``a1, a2, a3 = -(a1+a2)`` and ``-a1, -a2, -a3``),
      combined with ``+c`` to give 6 distinct <c+a> slip directions.
      (The ``-c`` versions are antiparallel duplicates and thus the
      same slip *system*.)
    - 6 {11-22}-type planes: for each pair of adjacent <a> vectors,
      the plane containing both of their <c+a> variants. Each such
      plane contains 2 distinct <c+a> Burgers vectors, giving
      6 × 2 = 12 systems.
    """
    c = c_over_a
    # The 6 basal <a> direction vectors (a_i and -a_i are different
    # <c+a> partners when combined with +c).
    six_a = np.concatenate([_HCP_A_DIRS, -_HCP_A_DIRS], axis=0)
    # 6 <c+a> unit Burgers vectors, all with +c component.
    ca_dirs = []
    for d in six_a:
        v = d + np.array([0.0, 0.0, c])
        ca_dirs.append(v / np.linalg.norm(v))
    ca_dirs = np.array(ca_dirs)  # (6, 3)

    # Restrict to pairs of <c+a> directions that are exactly 60°
    # apart in the basal projection. There are 6 such adjacent pairs
    # in the 6-fold ring, giving 6 distinct planes × 2 in-plane
    # Burgers vectors = 12 slip systems — the classical
    # {11-22}<11-23> count.
    basal_angles = np.array([
        np.arctan2(b[1], b[0]) for b in ca_dirs
    ])
    systems = []
    seen_planes = []
    for i in range(6):
        for j in range(i + 1, 6):
            # 60° apart modulo 360°
            diff = (basal_angles[i] - basal_angles[j]) % (2 * np.pi)
            diff = min(diff, 2 * np.pi - diff)
            if abs(diff - np.pi / 3) > 1e-6:
                continue
            b1 = ca_dirs[i]
            b2 = ca_dirs[j]
            n = np.cross(b1, b2)
            n = n / np.linalg.norm(n)
            # Reduce plane-normal sign
            for comp in n:
                if comp > 1e-9:
                    break
                if comp < -1e-9:
                    n = -n
                    break
            # Deduplicate plane
            if any(np.allclose(n, p, atol=1e-8) for p in seen_planes):
                continue
            seen_planes.append(n)
            systems.append([n, b1])
            systems.append([n, b2])
    return np.array(systems)


def _hcp_basal(c_over_a: float) -> np.ndarray:
    n = _basal_normal()
    return np.array([[n, d] for d in _HCP_A_DIRS])


def _hcp_prismatic(c_over_a: float) -> np.ndarray:
    return np.array([
        [np.cross(_basal_normal(), d) / np.linalg.norm(np.cross(_basal_normal(), d)), d]
        for d in _HCP_A_DIRS
    ])


def _hcp_pyramidal_a(c_over_a: float) -> np.ndarray:
    return _pyramidal_a_systems(c_over_a)


def _hcp_pyramidal_ca(c_over_a: float) -> np.ndarray:
    return _pyramidal_ca_systems(c_over_a)


# ===================================================================
#  Public API: slip-system registry
# ===================================================================

_CUBIC_FAMILIES = {
    "fcc":                _FCC_111_110_RAW,
    "fcc_octahedral":     _FCC_111_110_RAW,
    "bcc_110":            _BCC_110_111_RAW,
    "bcc_112":            _BCC_112_111_RAW,
    "bcc_123":            _BCC_123_111_RAW,
    "bcc":                np.concatenate([_BCC_110_111_RAW, _BCC_112_111_RAW], axis=0),
    "bcc_all":            np.concatenate([_BCC_110_111_RAW, _BCC_112_111_RAW,
                                          _BCC_123_111_RAW], axis=0),
}


HCP_RATIOS = {
    # Common room-temperature c/a values (dimensionless)
    "Ti":  1.587,
    "Zr":  1.593,
    "Mg":  1.624,
    "Zn":  1.856,
    "Be":  1.568,
    "Co":  1.623,
}


def get_slip_systems(
    family: str,
    c_over_a: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return slip-system normals and directions in the crystal frame.

    Parameters
    ----------
    family : str
        One of:
        - ``'fcc'`` or ``'fcc_octahedral'`` — 12 {111}<110>.
        - ``'bcc_110'`` — 12 {110}<111>.
        - ``'bcc_112'`` — 12 {112}<111>.
        - ``'bcc_123'`` — 24 {123}<111>.
        - ``'bcc'``     — 24 {110}∪{112}<111>.
        - ``'bcc_all'`` — 48 {110}∪{112}∪{123}<111>.
        - ``'hcp_basal'``           — 3 (0001)<11-20>.
        - ``'hcp_prismatic'``       — 3 {10-10}<11-20>.
        - ``'hcp_pyramidal_a'``     — 6 {10-11}<11-20>.
        - ``'hcp_pyramidal_ca'``    — 12 {11-22}<11-23>.
        - ``'hcp'``                 — basal + prismatic + pyramidal<a>
                                      + pyramidal<c+a> (24 systems).
    c_over_a : float, optional
        HCP c/a ratio (required for any ``hcp*`` family). For convenience
        the name of a metal in :data:`HCP_RATIOS` can be passed via
        :func:`get_slip_systems_for_material`.

    Returns
    -------
    n : ndarray (M, 3)
        Unit plane normals in the crystal (Cartesian orthonormal) frame.
    b : ndarray (M, 3)
        Unit slip directions.
    """
    name = family.lower()

    if name in _CUBIC_FAMILIES:
        raw = _CUBIC_FAMILIES[name]
        n = _normalise_rows(raw[:, 0, :])
        b = _normalise_rows(raw[:, 1, :])
        return n, b

    hcp_funcs = {
        "hcp_basal":         _hcp_basal,
        "hcp_prismatic":     _hcp_prismatic,
        "hcp_pyramidal_a":   _hcp_pyramidal_a,
        "hcp_pyramidal_ca":  _hcp_pyramidal_ca,
    }
    if name in hcp_funcs:
        if c_over_a is None:
            raise ValueError(f"'{family}' requires c_over_a.")
        arr = hcp_funcs[name](c_over_a)
        n = _normalise_rows(arr[:, 0, :])
        b = _normalise_rows(arr[:, 1, :])
        return n, b

    if name == "hcp":
        if c_over_a is None:
            raise ValueError("'hcp' requires c_over_a.")
        parts = [
            _hcp_basal(c_over_a),
            _hcp_prismatic(c_over_a),
            _hcp_pyramidal_a(c_over_a),
            _hcp_pyramidal_ca(c_over_a),
        ]
        arr = np.concatenate(parts, axis=0)
        n = _normalise_rows(arr[:, 0, :])
        b = _normalise_rows(arr[:, 1, :])
        return n, b

    raise ValueError(f"Unknown slip family '{family}'.")


def get_slip_systems_for_material(
    material: str,
    family: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: pick default families by material name.

    Parameters
    ----------
    material : str
        Any material in :data:`midas_stress.materials.STIFFNESS_LIBRARY`.
        Cubic materials default to ``'fcc'`` or ``'bcc'``; HCP (Ti,
        Mg, Zn, etc.) default to ``'hcp'``.
    family : str, optional
        Override the default family.

    Returns
    -------
    (n, b) : see :func:`get_slip_systems`.
    """
    from .materials import STIFFNESS_LIBRARY

    fcc_like = {"Au", "Cu", "Al", "Ni", "Ag", "Pt", "Pb"}
    bcc_like = {"Fe", "W", "Mo", "Ta", "Cr", "Nb", "V"}
    hcp_like = set(HCP_RATIOS.keys())

    if family is None:
        if material in fcc_like:
            family = "fcc"
        elif material in bcc_like:
            family = "bcc"
        elif material in hcp_like:
            family = "hcp"
        elif material in STIFFNESS_LIBRARY:
            sym = STIFFNESS_LIBRARY[material].get("symmetry", "cubic")
            family = "fcc" if sym == "cubic" else "hcp"
        else:
            raise ValueError(f"Unknown material '{material}'.")

    c_over_a = HCP_RATIOS.get(material)
    return get_slip_systems(family, c_over_a=c_over_a)


# ===================================================================
#  Slip-system transforms and Schmid factors
# ===================================================================

def slip_systems_to_lab(
    orient: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate slip-system normals / directions from crystal to lab frame.

    Uses the convention ``v_lab = U @ v_crystal`` where ``U`` is the
    crystal→lab orientation matrix (as returned by MIDAS).

    Parameters
    ----------
    orient : ndarray (N, 3, 3) or (3, 3)
    n_crystal : ndarray (M, 3)
    b_crystal : ndarray (M, 3)

    Returns
    -------
    n_lab : ndarray (N, M, 3) (or (M, 3) if ``orient`` is (3, 3))
    b_lab : same shape
    """
    orient = np.asarray(orient)
    single = orient.ndim == 2
    if single:
        orient = orient[None, :, :]
    # U @ n^T for each grain: einsum over the (M, 3) direction array.
    n_lab = np.einsum('gij,mj->gmi', orient, n_crystal)
    b_lab = np.einsum('gij,mj->gmi', orient, b_crystal)
    if single:
        return n_lab[0], b_lab[0]
    return n_lab, b_lab


def schmid_factor(
    orient: np.ndarray,
    load_dir: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
    absolute: bool = True,
) -> np.ndarray:
    """Schmid factor for a uniaxial load.

    For unit tensile axis :math:`\\hat\\ell`, unit plane normal
    :math:`\\hat n_\\text{lab}`, and unit slip direction
    :math:`\\hat b_\\text{lab}`:

    .. math::

       m = (\\hat n_\\text{lab} \\cdot \\hat\\ell)
           (\\hat b_\\text{lab} \\cdot \\hat\\ell).

    Parameters
    ----------
    orient : ndarray (N, 3, 3)
    load_dir : ndarray (3,)
        Loading direction in the lab frame (will be normalised).
    n_crystal, b_crystal : ndarray (M, 3)
        Slip system normals / directions in the crystal frame.
    absolute : bool
        If True (default) return |m|. Slip is directionless so the
        signed value is rarely the quantity of interest.

    Returns
    -------
    ndarray (N, M) of Schmid factors.
    """
    load_dir = np.asarray(load_dir, dtype=np.float64)
    load_dir = load_dir / np.linalg.norm(load_dir)
    n_lab, b_lab = slip_systems_to_lab(orient, n_crystal, b_crystal)
    # Dot products: (N, M)
    n_dot_l = np.einsum('gmi,i->gm', n_lab, load_dir)
    b_dot_l = np.einsum('gmi,i->gm', b_lab, load_dir)
    m = n_dot_l * b_dot_l
    return np.abs(m) if absolute else m


def resolved_shear_stress(
    stress: np.ndarray,
    orient: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
) -> np.ndarray:
    """General resolved shear stress on (plane, direction) pairs.

    Works for any stress state (uniaxial, biaxial, shear, per-grain
    corrected stress, ...).

    .. math::

       \\tau_s = \\hat n_\\text{lab}^\\top \\,\\sigma\\, \\hat b_\\text{lab}.

    Parameters
    ----------
    stress : ndarray (N, 3, 3)
        Per-grain stress tensors in the lab frame.
    orient : ndarray (N, 3, 3)
    n_crystal, b_crystal : ndarray (M, 3)

    Returns
    -------
    ndarray (N, M) — resolved shear stress for each grain on each
    system, in the same units as ``stress``.
    """
    n_lab, b_lab = slip_systems_to_lab(orient, n_crystal, b_crystal)
    # sigma @ b_lab : (N, M, 3)
    sb = np.einsum('gij,gmj->gmi', stress, b_lab)
    tau = np.einsum('gmi,gmi->gm', n_lab, sb)
    return tau


def _prep_schmid_or_tau(
    *,
    stress: Optional[np.ndarray],
    orient: np.ndarray,
    load_dir: Optional[np.ndarray],
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
) -> np.ndarray:
    """Internal dispatcher: return (N, M) Schmid / tau matrix.

    One of ``stress`` or ``load_dir`` must be provided.
    """
    if stress is None and load_dir is None:
        raise ValueError("provide either stress (N,3,3) or load_dir (3,).")
    if stress is not None:
        return resolved_shear_stress(stress, orient, n_crystal, b_crystal)
    return schmid_factor(orient, load_dir, n_crystal, b_crystal)


def dominant_slip_system(
    orient: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
    stress: Optional[np.ndarray] = None,
    load_dir: Optional[np.ndarray] = None,
    top_k: int = 1,
) -> dict:
    """Rank slip systems per grain by |Schmid factor| or |resolved shear|.

    Pass ``stress`` (N,3,3) for general loading, or ``load_dir`` (3,)
    for uniaxial Schmid-factor analysis.

    Parameters
    ----------
    orient : ndarray (N, 3, 3)
    n_crystal, b_crystal : ndarray (M, 3)
    stress : ndarray (N, 3, 3), optional
    load_dir : ndarray (3,), optional
    top_k : int
        Number of systems to return per grain (ranked by |score|).

    Returns
    -------
    dict with keys:
        'score'       : ndarray (N, M) — |tau| or |Schmid|
        'signed'      : ndarray (N, M) — original signed values
        'rank'        : ndarray (N, top_k) — system indices, largest |score| first
        'top_score'   : ndarray (N, top_k)
        'best_index'  : ndarray (N,) — shortcut for rank[:, 0]
        'best_score'  : ndarray (N,) — shortcut for top_score[:, 0]
    """
    signed = _prep_schmid_or_tau(
        stress=stress, orient=orient, load_dir=load_dir,
        n_crystal=n_crystal, b_crystal=b_crystal,
    )
    score = np.abs(signed)
    N, M = score.shape
    top_k = max(1, min(top_k, M))
    order = np.argsort(-score, axis=1)[:, :top_k]
    top_score = np.take_along_axis(score, order, axis=1)
    return {
        'score': score,
        'signed': signed,
        'rank': order,
        'top_score': top_score,
        'best_index': order[:, 0],
        'best_score': top_score[:, 0],
    }


def active_systems_from_crss(
    stress: np.ndarray,
    orient: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
    crss: np.ndarray | float,
) -> dict:
    """Classify slip-system activity from a CRSS threshold.

    A system is considered "active" when :math:`|\\tau_s| \\ge \\tau_c`.
    CRSS may be a scalar or an array of per-system thresholds (useful
    for HCP where basal/prismatic/pyramidal have different CRSS).

    Parameters
    ----------
    stress : ndarray (N, 3, 3) — in the same units as ``crss``.
    orient : ndarray (N, 3, 3)
    n_crystal, b_crystal : ndarray (M, 3)
    crss : float or ndarray (M,)

    Returns
    -------
    dict with keys:
        'tau'              : (N, M)
        'active'           : (N, M) bool
        'n_active_per_grain': (N,)
        'fraction_active_per_system': (M,)
        'grains_any_active': (N,) bool
        'fraction_grains_yielding': float  — share of grains with >=1 active system
    """
    tau = resolved_shear_stress(stress, orient, n_crystal, b_crystal)
    crss_arr = np.asarray(crss, dtype=np.float64)
    if crss_arr.ndim == 0:
        crss_arr = np.full(tau.shape[1], float(crss_arr))
    if crss_arr.shape != (tau.shape[1],):
        raise ValueError(
            f"crss must be scalar or length-M ({tau.shape[1]}); "
            f"got shape {crss_arr.shape}"
        )
    active = np.abs(tau) >= crss_arr[None, :]
    n_active = active.sum(axis=1)
    frac_sys = active.mean(axis=0)
    any_active = n_active > 0
    return {
        'tau': tau,
        'active': active,
        'n_active_per_grain': n_active,
        'fraction_active_per_system': frac_sys,
        'grains_any_active': any_active,
        'fraction_grains_yielding': float(any_active.mean()),
    }


def yield_proximity(
    stress: np.ndarray,
    orient: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
    crss: np.ndarray | float,
) -> dict:
    """Per-grain "distance to yield" ranking.

    The proximity score is :math:`\\max_s |\\tau_s| / \\tau_c^{(s)}`.
    A grain is yielding when the score is :math:`\\ge 1`.

    Parameters
    ----------
    stress : ndarray (N, 3, 3)
    orient : ndarray (N, 3, 3)
    n_crystal, b_crystal : ndarray (M, 3)
    crss : float or ndarray (M,)

    Returns
    -------
    dict with keys:
        'proximity'          : (N,) — max_s |tau|/crss
        'critical_system'    : (N,) — index s that maximises proximity
        'grains_sorted'      : (N,) — grain indices, most-yielded first
        'yielded'            : (N,) bool — proximity >= 1
    """
    tau = resolved_shear_stress(stress, orient, n_crystal, b_crystal)
    crss_arr = np.asarray(crss, dtype=np.float64)
    if crss_arr.ndim == 0:
        crss_arr = np.full(tau.shape[1], float(crss_arr))
    ratios = np.abs(tau) / crss_arr[None, :]
    prox = ratios.max(axis=1)
    crit = ratios.argmax(axis=1)
    sorted_idx = np.argsort(-prox)
    return {
        'proximity': prox,
        'critical_system': crit,
        'grains_sorted': sorted_idx,
        'yielded': prox >= 1.0,
    }


def taylor_factor(
    orient: np.ndarray,
    load_dir: np.ndarray,
    n_crystal: np.ndarray,
    b_crystal: np.ndarray,
    eps_axial: float = 1.0,
    volumes: Optional[np.ndarray] = None,
) -> dict:
    """Taylor factor for uniaxial loading (isostrain upper bound).

    Estimated as :math:`M = 1 / \\max_s |m_s|` per grain where
    :math:`m_s` is the Schmid factor of system :math:`s`. This is the
    *single-slip* approximation (conservative upper bound on
    yield-stress heterogeneity). The polycrystal Taylor factor is the
    volume-weighted mean.

    Parameters
    ----------
    orient : ndarray (N, 3, 3)
    load_dir : ndarray (3,)
    n_crystal, b_crystal : ndarray (M, 3)
    eps_axial : float
        Axial strain amplitude (only enters the reporting, not the
        dimensionless Taylor factor itself).
    volumes : ndarray (N,), optional
        Grain volumes for the polycrystal average.

    Returns
    -------
    dict with keys:
        'M_per_grain'     : (N,)
        'max_schmid'      : (N,)
        'M_poly'          : float — volume-weighted mean
        'M_uniform'       : float — equal-weighted mean
    """
    m = schmid_factor(orient, load_dir, n_crystal, b_crystal,
                      absolute=True)  # (N, M)
    m_max = m.max(axis=1)
    # Avoid division by zero for grains with all-zero Schmid (unlikely
    # with general orientations, but guard anyway).
    safe = np.where(m_max > 1e-12, m_max, np.nan)
    M_grain = 1.0 / safe
    if volumes is None:
        M_poly = float(np.nanmean(M_grain))
    else:
        w = volumes / volumes.sum()
        M_poly = float(np.nansum(w * M_grain))
    return {
        'M_per_grain': M_grain,
        'max_schmid':  m_max,
        'M_poly':      M_poly,
        'M_uniform':   float(np.nanmean(M_grain)),
    }


# ===================================================================
#  Convenience aliases
# ===================================================================

def list_slip_families() -> List[str]:
    """Return the names of every slip-system family available."""
    return sorted(list(_CUBIC_FAMILIES.keys()) + [
        "hcp_basal", "hcp_prismatic", "hcp_pyramidal_a",
        "hcp_pyramidal_ca", "hcp",
    ])
