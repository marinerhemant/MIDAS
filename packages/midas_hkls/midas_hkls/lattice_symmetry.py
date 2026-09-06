"""The symmetry of a lattice, found from the lattice rather than guessed from it.

(Module named ``lattice_symmetry`` rather than ``holohedry`` on purpose: it
exports a function called ``holohedry``, and ``from .holohedry import holohedry``
in the package ``__init__`` would rebind ``midas_hkls.holohedry`` from the module
to the function, silently shadowing it.)

Deciding a crystal system by comparing lengths and angles against tolerances is
fragile: it asks "is a within 2 % of b" and gets a different answer for the same
lattice at a different noise level. The lattice's **symmetry group** is a better
question, because its answer is an integer.

An operation of the lattice is an integer matrix ``M`` with ``det M = ±1`` that
preserves the metric::

    Mᵀ G M = G          G = Aᵀ A

Every such matrix maps the lattice onto itself. Counting them gives the order of
the holohedry, and the order alone fixes the crystal system:

    ====  ==============
    order  system
    ====  ==============
      2    triclinic
      4    monoclinic
      8    orthorhombic
     12    rhombohedral
     16    tetragonal
     24    hexagonal
     48    cubic
    ====  ==============

The search is over integer matrices with entries in −1…1, which suffices for a
**Niggli-reduced** cell — reduce first or the operations can have larger entries
and be missed. That is enforced rather than assumed.

Tolerance still enters, in deciding whether ``Mᵀ G M`` equals ``G``. But it
enters once, on a quantity with a natural scale (the metric), instead of
separately on three lengths and three angles — and the output is a group order,
which is discrete, so a small change in tolerance either changes the answer
completely or not at all. That is easier to notice than a silent reclassification.

It is still a real dependence, and on noisy data it matters. Measured on a real
ab-initio cell whose true lattice is tetragonal, the answer ran **triclinic →
monoclinic → rhombohedral → tetragonal** as ``rel_tol`` went 1e-3 → 4e-2. Note
"rhombohedral" in the middle: a loose tolerance manufactures accidental
operations that are not a subgroup step on the way to the truth.

**So do not choose the tolerance by hand.** :func:`tolerance_from_fit` derives it
from the cell covariance you already have — the metric goes as length squared,
so a relative uncertainty ``sigma_a/a`` on a length is worth ``2 sigma_a/a`` on
``G``. Ask for the symmetry the data can support, not the symmetry you can reach
by loosening until it appears. On the real 2604 domain-1 cell that rule gives a
3-sigma tolerance of 0.027 and the answer **tetragonal** — which is the truth,
and which agrees independently with the a/b split being only 1.7 sigma.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Holohedry", "lattice_symmetry_operations", "holohedry",
           "tolerance_from_fit", "holohedry_from_fit"]

#: Order of the holohedry → crystal system.
ORDER_TO_SYSTEM: Dict[int, str] = {
    2: "triclinic", 4: "monoclinic", 8: "orthorhombic",
    12: "rhombohedral", 16: "tetragonal", 24: "hexagonal", 48: "cubic",
}


@dataclass
class Holohedry:
    """The lattice's own symmetry, and what it implies."""
    order: int
    system: str
    operations: List[np.ndarray]
    n_fold_axes: Dict[int, int]
    reduced_cell: Tuple[float, ...]

    def __str__(self) -> str:
        folds = ", ".join(f"{n}-fold x{c}" for n, c in sorted(self.n_fold_axes.items())
                          if n > 1) or "none"
        return (f"{self.system} (holohedry order {self.order}; axes: {folds})")


def _order_of(M: np.ndarray, max_n: int = 6) -> int:
    """Multiplicative order of an integer matrix (1..6 for a crystal lattice)."""
    P = np.eye(3, dtype=np.int64)
    for n in range(1, max_n + 1):
        P = P @ M
        if np.array_equal(P, np.eye(3, dtype=np.int64)):
            return n
    return 0


@lru_cache(maxsize=4)
def _unimodular_candidates(entry_max: int) -> np.ndarray:
    """Every integer 3x3 matrix with entries in [-entry_max, entry_max] and
    determinant +-1, as one (N, 3, 3) array.

    Built once per ``entry_max`` and reused. For ``entry_max=1`` this is 3**9 =
    19683 matrices filtered down to a few thousand, which is small enough to
    test against a metric in a single batched einsum -- the loop this replaces
    made the function ~100 ms, which is invisible when you call it once and
    fatal inside ``to_conventional``'s candidate loop.
    """
    rng = range(-entry_max, entry_max + 1)
    M = np.array(list(itertools.product(rng, repeat=9)), np.int64).reshape(-1, 3, 3)
    det = np.rint(np.linalg.det(M.astype(float))).astype(np.int64)
    return np.ascontiguousarray(M[np.abs(det) == 1])


def lattice_symmetry_operations(cell: Sequence[float], *,
                                rel_tol: float = 1e-3,
                                entry_max: int = 1,
                                assume_reduced: bool = False
                                ) -> List[np.ndarray]:
    """All integer ``M`` with ``det = +-1`` and ``M^T G M = G``.

    Unless ``assume_reduced``, the cell is Niggli-reduced first -- the entries of
    the symmetry operations are only bounded by ``entry_max`` for a reduced
    cell, so skipping that silently loses operations and under-reports the
    symmetry.
    """
    from .lattice import Lattice
    from .niggli import niggli_reduce

    c = tuple(float(v) for v in cell)
    if not assume_reduced:
        c = niggli_reduce(c).cell
    lat = Lattice(a=c[0], b=c[1], c=c[2], alpha=c[3], beta=c[4], gamma=c[5])
    A = np.asarray(lat.cartesian_vectors(), float).T
    G = A.T @ A
    # Each entry gets its OWN scale. G_ij = a_i . a_j, so the natural scale for
    # entry (i, j) is sqrt(G_ii G_jj) -- NOT the global max|G|. With a global
    # scale a cell with c >> a is scored against c^2 everywhere, and for
    # La3Ni2O7-like axial ratios (c/a ~ 5) a 4 % tolerance becomes larger than
    # a^2 itself: every operation that scrambles the short axes passes, the
    # order comes out non-crystallographic, and a P-tetragonal lattice is
    # reported as triclinic. Per-entry scaling is what makes the test mean the
    # same thing for isotropic and very anisotropic cells alike.
    diag = np.sqrt(np.abs(np.diag(G)))
    scale = np.outer(diag, diag)
    scale = np.maximum(scale, 1e-30)

    M = _unimodular_candidates(int(entry_max))
    Mf = M.astype(float)
    # M^T G M for every candidate at once
    MtGM = np.einsum("nki,kl,nlj->nij", Mf, G, Mf, optimize=True)
    resid = np.abs(MtGM - G) / scale
    keep = resid.reshape(len(M), -1).max(axis=1) <= rel_tol
    return [np.ascontiguousarray(m) for m in M[keep]]


def holohedry(cell: Sequence[float], *, rel_tol: float = 1e-3) -> Holohedry:
    """Crystal system of a lattice, from its symmetry group order.

    Returns a :class:`Holohedry`. An order not in the crystallographic set
    (2, 4, 8, 12, 16, 24, 48) means the tolerance is wrong for the data — too
    loose and accidental operations creep in, too tight and real ones are lost.
    The system is then reported as ``"unrecognised"`` with the order attached
    rather than being rounded to the nearest plausible answer.
    """
    from .niggli import niggli_reduce

    red = niggli_reduce(cell).cell
    ops = lattice_symmetry_operations(red, rel_tol=rel_tol, assume_reduced=True)
    order = len(ops)
    folds: Dict[int, int] = {}
    for M in ops:
        if int(round(float(np.linalg.det(M)))) != 1:
            continue                       # count proper rotations only
        n = _order_of(M)
        if n:
            folds[n] = folds.get(n, 0) + 1
    return Holohedry(order=order,
                     system=ORDER_TO_SYSTEM.get(order, "unrecognised"),
                     operations=ops, n_fold_axes=folds, reduced_cell=red)


def tolerance_from_fit(fit, *, n_sigma: float = 3.0) -> float:
    """A metric tolerance justified by a cell's own uncertainties.

    ``fit`` is a :class:`~midas_hkls.ub_refine.UBFit`. The metric tensor goes as
    length squared, so a relative length uncertainty ``σ_a/a`` propagates to
    ``2 σ_a/a`` on ``G``; the largest such term over the three axes, times
    ``n_sigma``, is the scale at which two metric entries are indistinguishable.

    Using this instead of a hand-picked number is the difference between "the
    lattice is tetragonal to within what we measured" and "the lattice becomes
    tetragonal if I loosen far enough".

    .. warning::

       **This CANNOT corroborate a significance test that uses the same fit.**
       The two are algebraically one test. For the 4-fold that exchanges a and
       b, the metric residual is ``|b^2 - a^2| / (a b)``, and this function
       returns ``n_sigma * 2 * max(sigma/len)``. So :func:`holohedry` accepts
       the 4-fold **iff the a/b split is within about n_sigma of that very
       sigma** — the classification IS the t-test, restated. Measured on the
       real 2604 nickelate cell: residual 0.03189 against tolerance 0.04291.
       Reporting "the split is 1.8 sigma AND the symmetry is tetragonal" as two
       agreeing findings is reporting one number twice. (Refuted claim
       6f60212112ec, 2026-09-03, all four verify lenses.)

       **It also awards symmetry in proportion to uncertainty.** A noisier fit
       gives a looser tolerance and therefore *higher* apparent symmetry. On
       that same data the cleanest subsets returned orthorhombic and tetragonal
       appeared only once worse-fitting spots were admitted. If you want to know
       whether a lattice is tetragonal, improving the data must be able to
       *change* the answer; here it changes it toward lower symmetry.

       Use it to set a defensible starting tolerance, then **report the group
       order as a function of tolerance** across a range, not the value at one
       derived point. Check where the boundaries sit: on the 2604 cell the
       orthorhombic/tetragonal boundary was at 0.042571 against a used value of
       0.042915, a margin of 0.8 %.
    """
    rel = max(s / v for v, s in zip(fit.cell[:3], fit.cell_sigma[:3]) if v > 0)
    return float(n_sigma * 2.0 * rel)


def holohedry_from_fit(fit, *, n_sigma: float = 3.0) -> Holohedry:
    """:func:`holohedry` at the tolerance the fit's own covariance justifies."""
    return holohedry(fit.cell, rel_tol=tolerance_from_fit(fit, n_sigma=n_sigma))
