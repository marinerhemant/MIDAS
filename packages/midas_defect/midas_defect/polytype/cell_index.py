"""Full polytype-cell reflection indexer (mixed ``hkl``, not just the (00l) rod).

:mod:`ladder` measures the ``n*G/3`` rod along the activated <111> -- i.e. the
**(00l) rod** of the polytype hexagonal cell. This module builds the *whole*
reciprocal lattice of an oriented polytype (9R, 2H, 18R, ...) and indexes
arbitrary observed reflections against it, including the **mixed ``hkl``** that
the rod cannot reach. It is the tool that confirms a forbidden-gap satellite set
is one self-consistent phase rather than a coincidence of unrelated spots.

The 9R as an FCC-derived polytype (Shoji-Nishiyama orientation relationship)
-----------------------------------------------------------------------------
9R = 9-layer rhombohedral close packing (space group R-3m). Built on an FCC
parent of lattice ``a_fcc``:

    a_9R = a_fcc / sqrt(2)         (in-plane nearest-neighbour spacing)
    c_9R = 9 * d_111 = 3*sqrt(3)*a_fcc   (nine close-packed layers)

oriented with **c || one parent <111>** (the active fault-plane normal) and the
basal ``a`` along a parent <1-10> in that plane. Then:

  * the **(00l) rod** with ``l = 3n`` GEOMETRICALLY lands on the FCC <111> ladder:
    (0 0 3)=G/3, (0 0 6)=2G/3, (0 0 9)=111, (0 0 12)=4G/3, (0 0 15)=5G/3,
    (0 0 18)=222 -- but see the structure-factor caveat below;
  * **mixed (hkl)** fill the gaps off the rod -- e.g. for demk Cu-Al the
    "q/G~2.1-2.2" forbidden cluster indexes as 9R (2 0 8), (2 0 11), (1 -1 17),
    (-2 2 8)-type reflections, confirming it is the same 9R phase.

Structure-factor caveat -- the on-axis satellites are EXTINCT for an ideal 9R
----------------------------------------------------------------------------
R-centering only fixes *which* reflections are allowed; it does **not** give their
intensity. Evaluating ``|F(hkl)|^2`` for the explicit close-packed basis
(:func:`structure_factor_intensity`) shows that the **(0 0 l) rod is structurally
extinct for l != 9n**: F(003)=F(006)=F(0012)=F(0015)=0 exactly, while
F(009)=F(0018)=N=9 (|F|^2=81). So an *ideal, infinite* 9R has **no n*G/3 satellites
on the <111> axis** -- the only near-axis reflections are the FCC fundamentals 111
and 222. The genuine 9R superlattice reflections are mixed-hkl, ~83 deg OFF the
axis. The on-axis G/3 ladder seen in data is therefore stacking-DISORDER /
finite-lamella diffuse intensity (Hendricks-Teller / Warren), NOT ideal-crystal
Bragg. Consequence: a satellite/parent intensity ratio has **no simple |F|
volume-fraction correction** (the ideal |F_sat|^2 is zero); the raw ratio is a
lower bound whose conversion needs a fault-statistics model, not a structure
factor. See :func:`structure_factor_intensity`.

R-centering selects reflections by ``-h+k+l = 3n`` (obverse) or ``h-k+l = 3n``
(reverse). The two settings are the two stacking polarities (obverse/reverse),
i.e. the polytype's own twin -- the origin of the satellite doublet
(:mod:`satellite_doublet`).

Orientation convention (same as the rest of the package, see
:func:`ladder.decontaminate_ladder`): ``orientations`` are in the ``U @ G``
convention (sample-frame vector ``= U @ crystal_vector``). A **raw MIDAS
``Grains.csv`` matrix must be transposed** before being passed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PolytypeCell",
    "nine_r_from_fcc",
    "polytype_reflections",
    "index_reflections",
    "structure_factor_intensity",
    "close_packed_basis",
    "NINE_R_SEQUENCE",
]

#: 9R close-packed stacking (Zhdanov (21)_3); intrinsic fault every 3rd layer.
NINE_R_SEQUENCE = "ABCBCACAB"

#: in-plane fractional (x, y) of the three close-packed sites, R-OBVERSE setting
#: (so the nonzero-F reflections satisfy -h+k+l=3n, matching ``PolytypeCell`` default).
_CP_XY_OBVERSE = {"A": (0.0, 0.0), "B": (2.0 / 3.0, 1.0 / 3.0), "C": (1.0 / 3.0, 2.0 / 3.0)}
#: reverse setting (the other stacking polarity / the polytype's own twin) -- swap B<->C.
_CP_XY_REVERSE = {"A": (0.0, 0.0), "B": (1.0 / 3.0, 2.0 / 3.0), "C": (2.0 / 3.0, 1.0 / 3.0)}


def close_packed_basis(
    sequence: str = NINE_R_SEQUENCE,
    *,
    reverse: bool = False,
    spacing_modulation: float = 0.0,
) -> NDArray[np.floating]:
    """``(N, 3)`` fractional coords, one atom per close-packed layer.

    Layer ``j`` sits at height ``z = j/N`` (equal d_111 spacing) with in-plane
    ``(x, y)`` set by its A/B/C stacking letter. Default = the 9R sequence in the
    R-obverse setting; ``reverse=True`` gives the opposite stacking polarity (the
    obverse/reverse pair are the two senses whose superlattice reflections satisfy
    ``-h+k+l=3n`` and ``h-k+l=3n`` respectively).

    ``spacing_modulation`` adds a **period-3 interlayer-spacing modulation** (fault
    relaxation), ``z_j = (j + spacing_modulation*cos(2*pi*j/3)) / N``, in units of the
    layer spacing d_111. With the default 0 the layers are exactly equally spaced and
    the on-axis (0 0 l!=9n) reflections are extinct; a nonzero value turns ON the
    on-axis n*G/3 satellites (see :func:`structure_factor_intensity`).
    """
    seq = sequence.upper()
    table = _CP_XY_REVERSE if reverse else _CP_XY_OBVERSE
    n = len(seq)
    out = np.empty((n, 3), dtype=np.float64)
    for j, ch in enumerate(seq):
        if ch not in table:
            raise ValueError(f"stacking letter {ch!r} not in A/B/C")
        x, y = table[ch]
        dz = spacing_modulation * math.cos(2.0 * math.pi * j / 3.0)
        out[j] = (x, y, (j + dz) / n)
    return out


def structure_factor_intensity(
    hkl,
    *,
    sequence: str = NINE_R_SEQUENCE,
    reverse: bool = False,
    spacing_modulation: float = 0.0,
    comp_modulation: float = 0.0,
):
    """``|F(hkl)|^2`` of an ideal close-packed polytype (one identical atom/layer).

    All atoms are the same element, so the atomic form factor ``f(s)`` is a common
    prefactor that cancels in any reflection-to-reflection ratio; this returns the
    purely geometric ``|sum_j exp(2*pi*i*(h*x_j + k*y_j + l*z_j))|^2`` (i.e. f=1),
    a function of ``(h, k, l)`` and the stacking ``sequence`` only (cell ``a``/``c``
    set |q|, not |F|).

    KEY RESULT for 9R (``sequence='ABCBCACAB'``): the **(0 0 l) rod is structurally
    extinct for ``l != 9n``** -- F(003)=F(006)=F(0012)=F(0015)=0 even though
    R-centering 'allows' ``l = 3n``. Only F(009)=F(0018)=N=9 survive on the axis
    (|F|^2 = 81). So an ideal 9R has **no on-axis n*G/3 satellites**; the only
    near-<111> reflections are the FCC fundamentals 111 and 222. The genuine 9R
    superlattice reflections are mixed-hkl, ~83 deg OFF the axis (|F|^2 ~ 16).

    What turns the on-axis ladder ON (confirmed by 3D finite-stack simulation):
    a **period-3 modulation** -- ``spacing_modulation`` (interlayer-spacing fault
    relaxation, period 3) and/or ``comp_modulation`` (period-3 scattering-power /
    chemical ordering, e.g. Al on every 3rd plane). Either makes the on-axis
    (0 0 3n!=9) satellites appear with intensity **~ (modulation amplitude)^2**
    (second order). So the on-axis satellite is NOT a finite-size / Hendricks-Teller
    feature (finite size only adds Laue fringes at the fundamentals) and NOT ideal
    9R Bragg -- it is a modulation superstructure reflection.

    VOLUME-FRACTION CONSEQUENCE: the on-axis satellite intensity ~ f_9R *
    (modulation amplitude)^2, so it **conflates volume fraction with modulation
    amplitude** -- f_9R cannot be recovered from the on-axis ladder alone, and there
    is no simple |F| correction (the ideal |F_sat|^2 is zero). For a modulation-
    independent volume fraction use the **off-axis FIRST-ORDER 9R reflections**
    (the ~83-deg mixed-hkl, |F|^2 ~ 16), which scale directly with f_9R.

    ``hkl`` may be a single triple or an ``(M, 3)`` array; returns a float or an
    ``(M,)`` array of ``|F|^2`` accordingly. ``comp_modulation`` (in (-1, 1)) sets
    per-layer scattering weight ``1 + comp_modulation*cos(2*pi*j/3)``.
    """
    basis = close_packed_basis(sequence, reverse=reverse, spacing_modulation=spacing_modulation)
    n = basis.shape[0]
    w = 1.0 + comp_modulation * np.cos(2.0 * math.pi * np.arange(n) / 3.0)  # (N,)
    hk = np.asarray(hkl, dtype=np.float64)
    single = hk.ndim == 1
    hk = np.atleast_2d(hk)
    phase = 2.0 * math.pi * (hk @ basis.T)          # (M, N)
    F = (w[None, :] * np.exp(1j * phase)).sum(axis=1)  # (M,)
    inten = np.abs(F) ** 2
    return float(inten[0]) if single else inten


@dataclass
class PolytypeCell:
    """A close-packed polytype hexagonal cell.

    ``a``/``c`` in angstrom; ``centering`` selects the reflection condition:
    ``'R-obverse'`` (-h+k+l=3n), ``'R-reverse'`` (h-k+l=3n), or ``'P'`` (none).
    """

    a: float
    c: float
    centering: str = "R-obverse"
    name: str = "9R"

    def allowed(self, h: int, k: int, l: int) -> bool:
        if self.centering == "R-obverse":
            return (-h + k + l) % 3 == 0
        if self.centering == "R-reverse":
            return (h - k + l) % 3 == 0
        return True


def nine_r_from_fcc(a_fcc: float, *, centering: str = "R-obverse") -> PolytypeCell:
    """9R cell derived from an FCC parent: a=a_fcc/sqrt2, c=3*sqrt3*a_fcc."""
    return PolytypeCell(
        a=a_fcc / math.sqrt(2.0),
        c=3.0 * math.sqrt(3.0) * a_fcc,
        centering=centering,
        name="9R",
    )


def _unit(v) -> NDArray[np.floating]:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _orientations(orientations) -> NDArray[np.floating]:
    U = np.asarray(orientations, dtype=np.float64)
    if U.ndim == 2 and U.shape[-1] == 9:
        U = U.reshape(-1, 3, 3)
    elif U.shape == (3, 3):
        U = U.reshape(1, 3, 3)
    return U


def _reciprocal_basis(
    cell: PolytypeCell,
    c_dir: NDArray[np.floating],
    a_dir: NDArray[np.floating],
) -> tuple[NDArray, NDArray, NDArray]:
    """Hexagonal reciprocal basis (b1,b2,b3) in the sample frame, with the c-axis
    along ``c_dir`` and the first basal vector along the in-plane part of ``a_dir``
    (the second basal vector is 120 deg from it)."""
    c_dir = _unit(c_dir)
    a1 = a_dir - np.dot(a_dir, c_dir) * c_dir
    a1 = _unit(a1)
    a2 = math.cos(math.radians(120.0)) * a1 + math.sin(math.radians(120.0)) * np.cross(c_dir, a1)
    A1 = cell.a * a1
    A2 = cell.a * a2
    C = cell.c * c_dir
    V = float(np.dot(A1, np.cross(A2, C)))
    b1 = 2.0 * math.pi * np.cross(A2, C) / V
    b2 = 2.0 * math.pi * np.cross(C, A1) / V
    b3 = 2.0 * math.pi * np.cross(A1, A2) / V
    return b1, b2, b3


def polytype_reflections(
    cell: PolytypeCell,
    orientation,
    c_axis_hkl,
    *,
    a_axis_hkl=None,
    q_max_inv_A: float = 7.0,
    h_max: int = 4,
    l_max: int = 24,
    with_intensity: bool = False,
    min_rel_intensity: float = 0.0,
    sequence: str = NINE_R_SEQUENCE,
):
    """Allowed reflections of one oriented polytype crystal, in the sample frame.

    ``orientation`` is a single 3x3 (or flat-9) matrix in the ``U @ G`` convention.
    The c-axis is set parallel to ``U @ c_axis_hkl`` (a parent <111> for a fault
    polytype); the basal a-axis to the in-plane part of ``U @ a_axis_hkl`` (default:
    a parent <1-10> perpendicular to ``c_axis_hkl``). Returns ``(hkl, q_sample)``
    for every ``cell.allowed`` reflection with ``|q| <= q_max_inv_A``.

    ``with_intensity`` additionally returns ``|F(hkl)|^2`` (see
    :func:`structure_factor_intensity`) as a third element ``(hkl, q, intensity)``.
    ``min_rel_intensity`` (in (0, 1]) drops STRUCTURALLY EXTINCT / weak reflections
    -- those with ``|F|^2 < min_rel_intensity * max|F|^2``. This is how you remove
    the R-centering-allowed-but-extinct on-axis (0 0 l) ``l != 9n`` satellites that
    an ideal 9R does not actually produce; the default 0.0 keeps the geometric set.
    """
    U = _orientations(orientation)[0]
    c_hkl = np.asarray(c_axis_hkl, dtype=np.float64)
    c_dir = U @ c_hkl
    if a_axis_hkl is None:
        # any <1-10> perpendicular to c_axis_hkl in the crystal
        cand = [
            np.array(v, dtype=np.float64)
            for v in [(1, -1, 0), (0, 1, -1), (1, 0, -1), (-1, 1, 0), (0, -1, 1), (-1, 0, 1)]
        ]
        perp = [v for v in cand if abs(float(np.dot(v, c_hkl))) < 1e-9]
        if not perp:
            raise ValueError(f"no <1-10> perpendicular to c_axis_hkl={tuple(c_axis_hkl)}")
        a_dir = U @ perp[0]
    else:
        a_dir = U @ np.asarray(a_axis_hkl, dtype=np.float64)
    b1, b2, b3 = _reciprocal_basis(cell, c_dir, a_dir)

    hkls = []
    qs = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-l_max, l_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not cell.allowed(h, k, l):
                    continue
                G = h * b1 + k * b2 + l * b3
                q = float(np.linalg.norm(G))
                if q <= q_max_inv_A:
                    hkls.append((h, k, l))
                    qs.append(G)
    hkls = np.array(hkls, dtype=np.int64)
    qs = np.array(qs, dtype=np.float64)
    if not (with_intensity or min_rel_intensity > 0.0):
        return hkls, qs
    if len(hkls) == 0:
        inten = np.array([], dtype=np.float64)
    else:
        # match the structure-factor polarity to the cell's centering setting
        rev = cell.centering == "R-reverse"
        inten = structure_factor_intensity(hkls, sequence=sequence, reverse=rev)
        inten = np.atleast_1d(inten).astype(np.float64)
    if min_rel_intensity > 0.0 and inten.size:
        keep = inten >= min_rel_intensity * float(inten.max())
        hkls, qs, inten = hkls[keep], qs[keep], inten[keep]
    return (hkls, qs, inten) if with_intensity else (hkls, qs)


def _random_orientations(n: int, rng: np.random.Generator) -> NDArray[np.floating]:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def index_reflections(
    q_obs: NDArray[np.floating],
    cell: PolytypeCell,
    orientations,
    c_axis_hkls,
    *,
    tol_inv_A: float = 0.15,
    q_max_inv_A: float = 7.0,
    null_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Index each observed reflection against the oriented polytype lattice.

    For every ``q_obs`` row, search all (grain orientation x ``c_axis_hkls``
    variant) polytype lattices for the nearest reflection (3-D vector distance) and
    record the best ``hkl``, residual, grain, and c-axis. Significance is judged
    against a **null** built from ``null_trials`` random orientations (the 5th
    percentile of the random nearest-distance): a match is ``significant`` only if
    its residual is below that floor -- this defeats the multiplicity floor that
    makes a dense mosaic of orientations match almost anything.

    Returns one dict per observed reflection with keys: ``q``, ``hkl``, ``grain``,
    ``c_axis``, ``residual_inv_A``, ``null_floor_inv_A``, ``indexed`` (residual <
    tol), ``significant`` (residual < null floor).

    Notes
    -----
    ``orientations`` in the ``U @ G`` convention (transpose a raw ``Grains.csv``
    matrix first). ``c_axis_hkls`` is the list of candidate fault-plane normals to
    try per grain, e.g. the four ``<111>`` ``[(1,1,1),(1,1,-1),(1,-1,1),(-1,1,1)]``.
    """
    q_obs = np.asarray(q_obs, dtype=np.float64)
    if q_obs.ndim == 1:
        q_obs = q_obs[None, :]
    U = _orientations(orientations)
    rng = np.random.default_rng() if rng is None else rng

    # precompute every grain x c-axis reflection cloud (positions + provenance)
    pts = []
    prov = []  # (grain_index, c_axis_index, hkl_row_index)
    hkl_tables: list[NDArray[np.integer]] = []
    for gi in range(U.shape[0]):
        for ci, ch in enumerate(c_axis_hkls):
            hkls, G = polytype_reflections(
                cell, U[gi], ch, q_max_inv_A=q_max_inv_A
            )
            if len(G) == 0:
                continue
            base = len(hkl_tables)
            hkl_tables.append(hkls)
            pts.append(G)
            prov.append(np.column_stack([
                np.full(len(G), gi),
                np.full(len(G), ci),
                np.full(len(G), base),
                np.arange(len(G)),
            ]))
    allP = np.concatenate(pts, axis=0)
    allprov = np.concatenate(prov, axis=0)

    # null floor: nearest predicted reflection for random orientations (one c-axis is
    # enough -- the random cloud is isotropic), per observed |q| shell scale.
    Rn = _random_orientations(null_trials, rng)
    out = []
    for qi in range(q_obs.shape[0]):
        c = q_obs[qi]
        d = np.linalg.norm(allP - c, axis=1)
        j = int(np.argmin(d))
        gi, ci, tbl, row = (int(allprov[j, 0]), int(allprov[j, 1]),
                            int(allprov[j, 2]), int(allprov[j, 3]))
        resid = float(d[j])
        # null
        nd = np.empty(null_trials)
        for t in range(null_trials):
            _, Gn = polytype_reflections(cell, Rn[t], c_axis_hkls[0], q_max_inv_A=q_max_inv_A)
            nd[t] = float(np.linalg.norm(Gn - c, axis=1).min())
        floor = float(np.percentile(nd, 5))
        out.append({
            "q": float(np.linalg.norm(c)),
            "hkl": tuple(int(x) for x in hkl_tables[tbl][row]),
            "grain": gi,
            "c_axis": tuple(int(x) for x in np.asarray(c_axis_hkls[ci])),
            "residual_inv_A": resid,
            "null_floor_inv_A": floor,
            "indexed": resid < tol_inv_A,
            "significant": resid < floor,
        })
    return out
