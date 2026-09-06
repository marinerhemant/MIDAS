"""Which lattice distortion can a given cell setting actually express?

A B-matrix parameterisation that cannot represent the distortion a space group
implies **does not fail loudly**. It absorbs systematics and reports them as a
lattice parameter. This module makes the question answerable before a fit is
run, rather than after a result has to be retracted.

The Ruddlesden-Popper case, which is the general lesson
--------------------------------------------------------
For a phase whose orthorhombic child is described in the √2 × √2 **supercell**
(Fmmm), the same physical distortion looks entirely different in the tetragonal
**subcell**. With supercell axes A = a + b and B = b − a on the subcell axes::

    |A|² = a² + b² + 2ab·cos γ
    |B|² = a² + b² − 2ab·cos γ

so **A ≠ B if and only if γ ≠ 90°**. A subcell a ≠ b cannot produce a supercell
splitting at all.

The identity that makes this a proof, not an approximation: with γ = 90 the
subcell metric is diagonal, so ``|G(110)|² = |G(-110)|² = 1/a² + 1/b²`` —
*identical for any a and b*. A diagonal B-matrix can therefore **never** split
supercell (200)/(020). That splitting is zero by symmetry, not merely small.

The two modes are complementary, and which pair splits identifies the mode:

================================  ====================  ======================
                                  (100)/(010) subcell   (110)/(-110) subcell
================================  ====================  ======================
a ≠ b, γ = 90                     **SPLITS**            degenerate
a = b, γ ≠ 90                     degenerate            **SPLITS**
================================  ====================  ======================

Telling them apart requires **both** families in the same pattern. A pattern
containing only one of them cannot distinguish the modes at any signal-to-noise,
and no amount of precision repairs that — see :func:`can_distinguish_modes`.

Rule
----
Before fitting any cell distortion: ask which setting the indices are in, and
whether the parameterisation can represent the distortion the space group
implies. :func:`diagonal_B_can_express` answers the second half.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import math
import numpy as np

__all__ = [
    "SubcellSetting", "supercell_to_subcell", "subcell_to_supercell",
    "supercell_hkl_to_subcell", "splits_under",
    "diagonal_B_can_express", "can_distinguish_modes",
]


@dataclass(frozen=True)
class SubcellSetting:
    """A √2 × √2 supercell distortion re-expressed in the tetragonal subcell."""
    a: float
    b: float
    gamma_deg: float
    delta_supercell: float          # (A − B) / (A + B) of the supercell

    def __str__(self) -> str:
        return (f"subcell a = b = {self.a:.4f} A, gamma = {self.gamma_deg:.4f} deg "
                f"(supercell delta = {100*self.delta_supercell:+.3f} %)")


def supercell_to_subcell(A: float, B: float,
                         max_shear_deg: float = 15.0) -> SubcellSetting:
    """Convert an Fmmm supercell (A, B) into the equivalent subcell (a, γ).

    Inverts ``|A|² = a² + b² + 2ab cos γ``, ``|B|² = a² + b² − 2ab cos γ`` for
    the a = b case::

        a² = (A² + B²)/4 ,   cos γ = (A² − B²)/(4a²) = (A² − B²)/(A² + B²)

    An orthorhombic supercell is therefore a **γ shear** of a metrically
    tetragonal subcell — not a subcell a ≠ b.

    Note the closed form: ``cos γ = (A² − B²)/(A² + B²)`` lies in (−1, 1) for
    *any* positive A, B, so there is no domain error to guard against and every
    positive pair "converts". The meaningful check is **plausibility**, not
    validity: a genuine √2 supercell pair differs by well under a percent, so a
    large implied shear means the two numbers are not a supercell pair of one
    subcell at all. ``max_shear_deg`` enforces that; raise it deliberately if
    you really mean a strongly sheared cell.
    """
    if A <= 0 or B <= 0:
        raise ValueError("supercell axes must be positive")
    a2 = (A * A + B * B) / 4.0
    a = math.sqrt(a2)
    cos_g = (A * A - B * B) / (A * A + B * B)
    gamma = math.degrees(math.acos(cos_g))
    if abs(gamma - 90.0) > max_shear_deg:
        raise ValueError(
            f"A={A}, B={B} imply gamma = {gamma:.2f} deg, a shear of "
            f"{abs(gamma-90):.2f} deg. A real sqrt2 supercell pair differs by "
            f"well under a percent; these are probably not a supercell pair of "
            f"one subcell. Pass max_shear_deg if this is intended.")
    return SubcellSetting(a=a, b=a, gamma_deg=gamma,
                          delta_supercell=(A - B) / (A + B))


def subcell_to_supercell(a: float, gamma_deg: float,
                         b: float | None = None) -> Tuple[float, float]:
    """The supercell axes ``(A, B)`` implied by a subcell ``(a, b, γ)``."""
    if b is None:
        b = a
    g = math.radians(gamma_deg)
    s = a * a + b * b
    p = 2.0 * a * b * math.cos(g)
    return (math.sqrt(s + p), math.sqrt(s - p))


#: Supercell → subcell index map, for A* = (a*+b*)/2, B* = (b*−a*)/2.
def supercell_hkl_to_subcell(H: int, K: int, L: int) -> Tuple[float, float, int]:
    """``h = (H − K)/2 , k = (H + K)/2``, L unchanged. Half-integers are real."""
    return ((H - K) / 2.0, (H + K) / 2.0, L)


def _d_star_sq(h, k, l, a, b, c, gamma_deg) -> float:
    """1/d² for a monoclinic-in-γ cell (α = β = 90)."""
    g = math.radians(gamma_deg)
    sg2 = math.sin(g) ** 2
    return ((h / a) ** 2 + (k / b) ** 2 - 2 * h * k * math.cos(g) / (a * b)) / sg2 \
        + (l / c) ** 2


def splits_under(hkl_1: Sequence[float], hkl_2: Sequence[float], *,
                 a: float, b: float, c: float, gamma_deg: float,
                 rel_tol: float = 1e-12) -> bool:
    """Do two reflections have different d under this cell? (subcell indices)"""
    q1 = _d_star_sq(*hkl_1, a, b, c, gamma_deg)
    q2 = _d_star_sq(*hkl_2, a, b, c, gamma_deg)
    return abs(q1 - q2) > rel_tol * max(q1, q2, 1e-30)


def diagonal_B_can_express(mode: str) -> bool:
    """Can ``B = diag(1/a, 1/b, 1/c)`` represent this distortion mode?

    ``"a_ne_b"`` yes; ``"gamma_shear"`` **no** — a diagonal B pins γ = 90.
    Fitting a γ shear with a diagonal B in the subcell fits the mode the
    material does not have, and the fit will report the residual systematic as
    a lattice parameter.
    """
    m = mode.lower().replace("-", "_").replace(" ", "_")
    if m in ("a_ne_b", "a_neq_b", "orthorhombic", "ab_splitting"):
        return True
    if m in ("gamma_shear", "gamma", "shear", "monoclinic_gamma"):
        return False
    raise ValueError(f"unknown distortion mode {mode!r}; expected 'a_ne_b' "
                     "or 'gamma_shear'")


def can_distinguish_modes(subcell_hkls: Iterable[Sequence[int]]) -> dict:
    """Does this reflection set contain BOTH sensitive families?

    Returns a dict with ``has_ab_sensitive`` (an (h0L)/(0kL)-type pair, h ≠ k
    with one index zero), ``has_gamma_sensitive`` (an (hhL)/(h-hL)-type pair),
    and ``can_distinguish``. When only one family is present the two modes are
    **indistinguishable at any signal-to-noise** — a precision argument does not
    rescue it, only more reciprocal space does.
    """
    hk = {(int(h), int(k)) for h, k, *_ in subcell_hkls}
    ab = any((h != 0 and k == 0) or (h == 0 and k != 0) for h, k in hk)
    gam = any(h != 0 and abs(h) == abs(k) for h, k in hk)
    return {"has_ab_sensitive": ab, "has_gamma_sensitive": gam,
            "can_distinguish": ab and gam,
            "note": ("both families present" if (ab and gam) else
                     "only one family present — the modes are indistinguishable "
                     "here at any signal-to-noise")}
