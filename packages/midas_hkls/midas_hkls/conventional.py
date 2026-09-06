"""Primitive cell → conventional cell, because they are not the same number.

An ab-initio indexer recovers the lattice the **reflections** define, which is
the *primitive* cell. Published cells are almost always *conventional* — the
centred setting that displays the symmetry. For a body-centred lattice these
differ by a factor of two in volume, and comparing the wrong pair makes a
correct answer look like a 50 % error.

That is not hypothetical. Ab-initio indexing of La₃Ni₂O₇ (I4/mmm, conventional
a = 3.6116, c = 19.2516 Å) returns a primitive cell of 3.61 / 3.61 / 9.96 Å with
angles 79.6 / 79.6 / 90 — the two a-axes and the body-centring vector
(a/2, a/2, c/2). Read against the conventional 251 Å³ it looks like a "×0.5
failure"; read against the primitive 125.6 Å³ it is right to 1 %.

The search
----------
Every conventional basis vector is an integer combination of the primitive ones,
so enumerate combinations with small coefficients, take triples whose
determinant is a plausible centring index (1–4), and score the resulting metric
against each crystal system. **Highest symmetry wins, and the smallest cell that
achieves it.** The transformation matrix comes back so the choice is auditable.

This reports what the metric supports. It is not a space-group determination —
metric symmetry can exceed the true symmetry (a monoclinic cell can be
metrically orthorhombic by accident), and only the intensities settle that.

Known limit — the setting is not canonical
------------------------------------------
The **lattice** comes back reliably; **which of its equivalent conventional
descriptions** is chosen does not. Measured on ab-initio results for the same
La₃Ni₂O₇ lattice at different indexing tolerances, this returned I-tetragonal
(3.610 / 3.570 / 19.244), F-orthorhombic (5.118 / 5.318 / 19.241, the √2
setting) and a C-monoclinic description — all genuinely the same lattice, none
of them wrong, only one of them the published choice.

The c axis is stable throughout: 19.230-19.244 against a published 19.2516, so
**0.04-0.11 %** in every case. It is the *choice among settings* that wanders,
because that needs full Niggli standardisation plus Bravais-type determination
against the International Tables, which this does not do. Read the returned
``transformation`` and ``volume_ratio``, and compare lattices by volume and by
their reduced form rather than by axis labels.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ub_refine import cell_from_metric

__all__ = ["ConventionalCell", "metric_symmetry", "to_conventional",
           "to_conventional_from_fit"]

#: Crystal systems, most symmetric first — the search prefers earlier entries.
SYSTEM_ORDER = ("cubic", "hexagonal", "rhombohedral", "tetragonal",
                "orthorhombic", "monoclinic", "triclinic")


def metric_symmetry(cell: Sequence[float], *, rel_len: float = 0.02,
                    abs_ang: float = 1.5) -> str:
    """Which crystal system this **metric** is consistent with.

    This is a thin wrapper over :func:`midas_hkls.lattice_symmetry.holohedry`,
    which finds the lattice's actual symmetry group rather than comparing
    lengths and angles against windows. There is deliberately only one crystal
    system determination in this package; this signature is kept because it
    reads naturally at call sites that think in lengths and angles.

    The two tolerances are converted to the single metric tolerance the group
    search uses. The metric goes as length squared, so a relative length window
    ``rel_len`` is worth ``2 * rel_len`` on ``G``; an angle window enters as its
    value in radians, since ``G_ij = a_i a_j cos(theta)``. The looser of the two
    wins.

    If the group order is not crystallographic — which means the tolerance does
    not suit the data — this returns ``"triclinic"``, deliberately under-claiming
    rather than rounding to the nearest plausible system. Call ``holohedry``
    directly if you want to see that happen instead of having it hidden.

    Prefer :func:`midas_hkls.lattice_symmetry.holohedry_from_fit` when the cell
    came from a refinement: it takes the tolerance from the covariance instead of
    from an argument, which is the only way to ask what the data can support.
    """
    from .lattice_symmetry import holohedry as _holohedry

    rel_tol = max(2.0 * float(rel_len), math.radians(float(abs_ang)))
    system = _holohedry(cell, rel_tol=rel_tol).system
    return "triclinic" if system == "unrecognised" else system


@dataclass
class ConventionalCell:
    """A conventional setting for a primitive cell, and how it was reached."""
    cell: Tuple[float, ...]
    system: str
    transformation: np.ndarray        # A_conv = A_prim @ T, integer
    centring_index: int               # |det T| — 1 P, 2 I/C, 3 R, 4 F
    primitive_cell: Tuple[float, ...]
    primitive_system: str
    volume_ratio: float
    #: False when no standard setting was reachable within ``max_index``.
    is_standard: bool = True

    @property
    def centring_hint(self) -> str:
        return {1: "P (primitive)", 2: "I or C (one centring vector)",
                3: "R (rhombohedral, hexagonal setting)",
                4: "F (all-face centred)"}.get(self.centring_index,
                                               f"index {self.centring_index}")

    def __str__(self) -> str:
        c = self.cell
        return (f"conventional {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} | "
                f"{c[3]:.2f} {c[4]:.2f} {c[5]:.2f}  [{self.system}, "
                f"{self.centring_hint}, V x{self.volume_ratio:.0f} on the "
                f"primitive{'' if self.is_standard else ', NOT a standard setting'}]")


def _standardise_axes(B: np.ndarray, T: np.ndarray, system: str,
                      rel_len: float) -> Tuple[np.ndarray, np.ndarray]:
    """Put the axes in the conventional order for the system.

    Without this the unique axis can come out first — a correct cell that reads
    as wrong next to a published one. Tetragonal and hexagonal put the two equal
    axes first; monoclinic puts the non-90° angle on beta (unique axis b);
    orthorhombic sorts a < b < c.
    """
    def leq(x, y):
        return abs(x - y) <= rel_len * max(abs(x), abs(y), 1e-12)

    L = [float(np.linalg.norm(B[:, i])) for i in range(3)]
    order = list(range(3))
    if system in ("tetragonal", "hexagonal"):
        for odd in range(3):
            rest = [i for i in range(3) if i != odd]
            if leq(L[rest[0]], L[rest[1]]):
                order = rest + [odd]
                break
    elif system == "orthorhombic":
        order = sorted(range(3), key=lambda i: L[i])
    elif system == "monoclinic":
        cell = cell_from_metric(B.T @ B)
        angs = cell[3:]
        off = [i for i in range(3) if abs(angs[i] - 90.0) > 1e-6]
        if len(off) == 1 and off[0] != 1:
            order = [1, off[0], 0] if off[0] == 0 else [0, off[0], 1]
            order = sorted(set(order), key=order.index)
            if len(order) != 3:
                order = list(range(3))
    Bs = B[:, order]
    Ts = T[:, order]
    if np.linalg.det(Bs) < 0:                 # keep it right-handed
        Bs = Bs.copy(); Bs[:, 2] *= -1
        Ts = Ts.copy(); Ts[:, 2] *= -1
    return Bs, Ts


def _candidate_vectors(max_coeff: int) -> List[Tuple[int, int, int]]:
    out = []
    seen = set()
    rng = range(-max_coeff, max_coeff + 1)
    for n in itertools.product(rng, rng, rng):
        if n == (0, 0, 0):
            continue
        if tuple(-v for v in n) in seen:      # v and -v span the same axis
            continue
        seen.add(n)
        out.append(n)
    return out


def _is_standard_setting(cell: Sequence[float], system: str, *,
                         rel_len: float, abs_ang: float) -> bool:
    """Does this metric have the **standard shape** for an already-known system?

    This is a different question from "what system is this", and that
    distinction is the whole point. The crystal system is a property of the
    lattice and is determined once, group-theoretically, from the primitive
    cell. Asking it again of every candidate supercell is what used to let a
    genuinely triclinic lattice win an "F-centred monoclinic" description from
    an accidental symmetry in a x4 supercell, and what let the primitive cell of
    a body-centred tetragonal lattice short-circuit the search by being
    (correctly!) labelled tetragonal itself.

    So: system in, shape question out. No labelling here.
    """
    a, b, c, al, be, ga = (float(v) for v in cell)

    def leq(x, y):
        return abs(x - y) <= rel_len * max(abs(x), abs(y), 1e-12)

    def aeq(x, y):
        return abs(x - y) <= abs_ang

    ninety = aeq(al, 90) and aeq(be, 90) and aeq(ga, 90)
    if system == "cubic":
        return ninety and leq(a, b) and leq(b, c)
    if system == "hexagonal":
        return leq(a, b) and aeq(al, 90) and aeq(be, 90) and aeq(ga, 120)
    if system == "rhombohedral":
        # either setting is conventional: rhombohedral axes, or the hexagonal
        # setting with three lattice points per cell
        return ((leq(a, b) and leq(b, c) and aeq(al, be) and aeq(be, ga))
                or (leq(a, b) and aeq(al, 90) and aeq(be, 90) and aeq(ga, 120)))
    if system == "tetragonal":
        return ninety and leq(a, b)          # c is the unique axis
    if system == "orthorhombic":
        return ninety
    if system == "monoclinic":
        return aeq(al, 90) and aeq(ga, 90)   # b unique, beta free
    return True                              # triclinic: any reduced cell


def to_conventional(primitive_cell: Sequence[float], *,
                    max_coeff: int = 2, max_index: int = 4,
                    rel_len: float = 0.02, abs_ang: float = 1.5,
                    rel_tol: Optional[float] = None,
                    min_noncoplanarity: float = 0.20,
                    angle_bounds: Tuple[float, float] = (30.0, 150.0)
                    ) -> ConventionalCell:
    """Find the conventional setting of a primitive cell.

    The crystal system is determined **once**, from the primitive lattice's own
    symmetry group (:func:`midas_hkls.lattice_symmetry.holohedry`). The search
    then looks only for the smallest integer transformation that puts the metric
    into the standard *shape* for that system — it never re-labels a candidate,
    because a supercell of a triclinic lattice can easily look monoclinic by
    accident, and because the primitive cell of a body-centred tetragonal
    lattice is itself tetragonal (the holohedry is order 16 either way) and would
    otherwise short-circuit the search for the I-centred description.

    Ranking is by centring index first, then volume: the smallest cell that is a
    standard setting wins. ``angle_bounds`` and ``min_noncoplanarity`` reject
    squashed bases; no conventional setting has a 15 degree angle.

    If no transformation of index ``<= max_index`` reaches a standard setting,
    the primitive cell is returned with ``is_standard = False`` rather than
    silently pretending otherwise.
    """
    from .lattice import Lattice
    from .lattice_symmetry import holohedry as _holohedry

    pc = tuple(float(v) for v in primitive_cell)
    lat = Lattice(a=pc[0], b=pc[1], c=pc[2], alpha=pc[3], beta=pc[4], gamma=pc[5])
    A = np.asarray(lat.cartesian_vectors(), float).T          # columns = a,b,c
    v_prim = abs(float(np.linalg.det(A)))

    if rel_tol is None:
        rel_tol = max(2.0 * float(rel_len), math.radians(float(abs_ang)))
    prim_sys = _holohedry(pc, rel_tol=float(rel_tol)).system
    if prim_sys == "unrecognised":
        prim_sys = "triclinic"

    if prim_sys == "triclinic":
        # No supercell can improve a triclinic lattice, and every candidate
        # trivially satisfies the (empty) standard-shape test, so a tiebreak
        # would just pick whichever skewed basis it happened to rank first.
        from .niggli import niggli_reduce
        red = niggli_reduce(pc)
        return ConventionalCell(cell=red.cell, system="triclinic",
                                transformation=np.asarray(red.transformation, int),
                                centring_index=1, primitive_cell=pc,
                                primitive_system="triclinic", volume_ratio=1.0)

    best_key = None
    best: ConventionalCell | None = None
    lo, hi = angle_bounds

    vecs = _candidate_vectors(max_coeff)
    cols = [A @ np.array(n, float) for n in vecs]
    for i, j, k in itertools.combinations(range(len(vecs)), 3):
        T0 = np.array([vecs[i], vecs[j], vecs[k]], int).T
        d0 = int(round(np.linalg.det(T0)))
        if d0 == 0 or abs(d0) > max_index:
            continue
        B0 = np.stack([cols[i], cols[j], cols[k]], 1)
        lens = [np.linalg.norm(B0[:, m]) for m in range(3)]
        if min(lens) <= 0:
            continue
        if abs(float(np.linalg.det(B0))) / float(np.prod(lens)) < min_noncoplanarity:
            continue

        # try every axis order: which vector is "c" is a labelling choice, and
        # the standard shape for tetragonal/monoclinic depends on getting it right
        for perm in itertools.permutations(range(3)):
            B = B0[:, list(perm)]
            T = T0[:, list(perm)]
            if float(np.linalg.det(B)) < 0:      # keep it right-handed
                B = B.copy(); T = T.copy()
                B[:, 2] *= -1.0; T[:, 2] *= -1
            try:
                cell = cell_from_metric(B.T @ B)
            except ValueError:
                continue
            if any(not (lo <= ang <= hi) for ang in cell[3:]):
                continue
            if not _is_standard_setting(cell, prim_sys, rel_len=rel_len,
                                        abs_ang=abs_ang):
                continue
            vol = abs(float(np.linalg.det(B)))
            # smallest centring index, then smallest cell, then a<=b<=c-ish
            # Where the standard shape leaves the axis order free, take the
            # ascending one: c - a is largest exactly when a is the shortest
            # axis and c the longest, which is the convention. Minimising it
            # instead puts the long axis first (19.23 / 3.68 / 3.62).
            key = (abs(int(round(np.linalg.det(T)))), round(vol, 6),
                   -round(cell[2] - cell[0], 6))
            if best_key is None or key < best_key:
                best_key = key
                best = ConventionalCell(
                    cell=cell, system=prim_sys, transformation=T,
                    centring_index=abs(int(round(np.linalg.det(T)))),
                    primitive_cell=pc, primitive_system=prim_sys,
                    volume_ratio=vol / v_prim, is_standard=True)

    if best is None:
        return ConventionalCell(cell=pc, system=prim_sys,
                                transformation=np.eye(3, dtype=int),
                                centring_index=1, primitive_cell=pc,
                                primitive_system=prim_sys, volume_ratio=1.0,
                                is_standard=False)
    return best


def to_conventional_from_fit(fit, *, n_sigma: float = 3.0,
                             **kwargs) -> ConventionalCell:
    """Conventional setting with the tolerance taken from the fit's covariance.

    The companion to :func:`midas_hkls.lattice_symmetry.holohedry_from_fit`, and
    the reason it exists: calling :func:`to_conventional` with its default 2 %
    window on a refined cell asks a different question from the one the data can
    answer, and the two can disagree. On the real 2604 domain-1 cell the default
    window (0.0400) gives orthorhombic while the covariance (0.0429) gives
    tetragonal — the truth. Same cell, same code, tolerance straddling the
    boundary. Use this whenever the cell came from a refinement.
    """
    from .lattice_symmetry import tolerance_from_fit
    kwargs.pop("rel_tol", None)
    return to_conventional(fit.cell, rel_tol=tolerance_from_fit(fit, n_sigma=n_sigma),
                           **kwargs)
