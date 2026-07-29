"""Burgers-vector population analysis for deformed hexagonal crystals.

Implements §3.1 / §5 of Dragomir & Ungár, *J. Appl. Cryst.* **35** (2002) 556-564.
Given the *measured* contrast-factor parameters ``(q₁ᵐ, q₂ᵐ)`` of a deformed
hexagonal specimen (from a modified Williamson-Hall / multiple-whole-profile fit)
together with the numerically-computed per-sub-slip-system parameters from
:mod:`contrast_factor_hex`, solve for the fractions of the three Burgers-vector
types ⟨a⟩, ⟨c⟩, ⟨c+a⟩ (paper eqns 9-13).

The averaged mean-square strain weights each Burgers-vector type by its ``b²``
(paper eqns 3-4); ``C̄_{hk.0} b²`` is only a scale on the absolute dislocation
density, so the *two* independent shape parameters ``q₁ᵐ, q₂ᵐ`` plus the closure
``Σhᵢ = 1`` determine the three fractions. The deformed-Ti worked example
(``q₁ᵐ=-0.05, q₂ᵐ=0.18``) recovers ⟨a⟩:⟨c⟩:⟨c+a⟩ ≈ 75:20:5 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from .contrast_factor_hex import fit_hex_cbar, hex_slip_systems

__all__ = [
    "BURGERS_TYPES",
    "burgers_magnitude_A",
    "TypeContrastParameters",
    "burgers_type_parameters",
    "BurgersPopulation",
    "solve_burgers_population",
]

#: The three hexagonal Burgers-vector types.
BURGERS_TYPES = ("a", "c", "c+a")


def burgers_magnitude_A(crystal, btype: str) -> float:
    """Burgers-vector modulus |b| (Å) for ⟨a⟩, ⟨c⟩ or ⟨c+a⟩ in a hexagonal cell."""
    a = float(crystal.lattice.a)
    c = float(crystal.lattice.c)
    if btype == "a":
        return a
    if btype == "c":
        return c
    if btype == "c+a":
        return math.sqrt(a * a + c * c)
    raise ValueError(f"unknown Burgers type {btype!r}; expected one of {BURGERS_TYPES}")


@dataclass
class TypeContrastParameters:
    """Contrast-factor parameters averaged over the sub-slip-systems of one type."""
    btype: str
    cbar_hk0: float          # C̄_{hk.0} averaged over the type's sub-slip-systems
    q1: float                # C̄_{hk.0}-weighted mean q₁
    q2: float                # C̄_{hk.0}-weighted mean q₂
    burgers_A: float         # |b|
    subsystems: tuple        # contributing sub-slip-system names


def burgers_type_parameters(
    C6,
    crystal,
    *,
    exclude: Iterable[str] = ("S3",),
    n_phi: int = 360,
) -> Dict[str, TypeContrastParameters]:
    """Per-Burgers-type contrast parameters, averaged over sub-slip-systems.

    For each Burgers-vector type, averages ``C̄_{hk.0}`` over its sub-slip-systems
    and forms the ``C̄_{hk.0}``-weighted mean of ``q₁``/``q₂`` (so the averaged
    contrast factor stays the parabola ``C̄_{hk.0}^t(1 + q₁ᵗx + q₂ᵗx²)``).

    Parameters
    ----------
    C6
        Hexagonal 6×6 Voigt stiffness (:func:`contrast_factor_hex.hexagonal_stiffness`).
    crystal
        Hexagonal `midas_hkls.Crystal`.
    exclude
        Sub-slip-system names to drop (default ``("S3",)`` — the c-axis screw,
        whose contrast factor is negligible, following the paper's step (i)).
    """
    excl = set(exclude)
    by_type: Dict[str, list] = {t: [] for t in BURGERS_TYPES}
    for s in hex_slip_systems():
        if s.name in excl:
            continue
        model = fit_hex_cbar(C6, crystal, s, n_phi=n_phi)
        by_type[s.btype].append((s.name, model))

    out: Dict[str, TypeContrastParameters] = {}
    for t, entries in by_type.items():
        if not entries:
            continue
        c0 = np.array([m.cbar_hk0 for _, m in entries])
        q1 = np.array([m.q1 for _, m in entries])
        q2 = np.array([m.q2 for _, m in entries])
        w = c0.sum()
        out[t] = TypeContrastParameters(
            btype=t,
            cbar_hk0=float(c0.mean()),
            q1=float((c0 * q1).sum() / w),
            q2=float((c0 * q2).sum() / w),
            burgers_A=burgers_magnitude_A(crystal, t),
            subsystems=tuple(name for name, _ in entries),
        )
    return out


@dataclass
class BurgersPopulation:
    """Solved Burgers-vector-type fractions for a deformed hexagonal specimen."""
    fractions: Dict[str, float]      # {"a":.., "c":.., "c+a":..}, Σ = 1
    P_A2: float                      # P = Σ hᵢ C̄_{hk.0}ⁱ bᵢ²  (= (b²C̄_{hk.0})ᵐ), Å²
    all_nonnegative: bool            # physical solution iff all fractions ≥ 0
    q1_measured: float
    q2_measured: float

    def dominant(self) -> str:
        return max(self.fractions, key=self.fractions.get)


def solve_burgers_population(
    q1_measured: float,
    q2_measured: float,
    type_params: Dict[str, TypeContrastParameters],
) -> BurgersPopulation:
    """Solve eqns 11-13 for the ⟨a⟩/⟨c⟩/⟨c+a⟩ fractions.

    With ``wᵢ = C̄_{hk.0}ⁱ bᵢ²`` and ``P = Σ wᵢ hᵢ``, eqns 11-12 rearrange to the
    linear conditions ``Σ wᵢ (qₖⁱ − qₖᵐ) hᵢ = 0`` (k=1,2), closed by ``Σ hᵢ = 1``.

    Parameters
    ----------
    q1_measured, q2_measured
        Experimentally fitted contrast-factor parameters of the specimen.
    type_params
        Per-type parameters from :func:`burgers_type_parameters` (all three types
        required).
    """
    missing = [t for t in BURGERS_TYPES if t not in type_params]
    if missing:
        raise ValueError(f"type_params missing Burgers types: {missing}")

    order = list(BURGERS_TYPES)
    w = np.array([type_params[t].cbar_hk0 * type_params[t].burgers_A ** 2 for t in order])
    q1 = np.array([type_params[t].q1 for t in order])
    q2 = np.array([type_params[t].q2 for t in order])

    M = np.array([
        w * (q1 - q1_measured),
        w * (q2 - q2_measured),
        np.ones(3),
    ])
    rhs = np.array([0.0, 0.0, 1.0])
    h = np.linalg.solve(M, rhs)
    P = float((w * h).sum())

    fractions = {t: float(h[i]) for i, t in enumerate(order)}
    return BurgersPopulation(
        fractions=fractions,
        P_A2=P,
        all_nonnegative=bool(np.all(h >= -1e-9)),
        q1_measured=q1_measured,
        q2_measured=q2_measured,
    )
