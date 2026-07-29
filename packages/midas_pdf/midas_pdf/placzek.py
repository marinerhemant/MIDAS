"""Placzek correction — X-ray note.

The Placzek correction is a **neutron** total-scattering correction that
absorbs the O((k'/k)²) shift of the elastic-line Compton response from a
static-approximation Q-grid to the true k-grid, where the recoil momentum
transfer is a substantial fraction of the incident momentum.

**For X-ray total scattering the equivalent correction is subsumed by the
Breit-Dirac recoil factor**, which we already apply to the Hubbell Compton
term in :func:`midas_pdf.composition.Composition.compton` (default
``breit_dirac=True``). The higher-order correction beyond Breit-Dirac is of
order (hν / M c²)² which at 63 keV for a Ni atom is $\sim 10^{-12}$ ---
utterly negligible.

We therefore ship NO separate Placzek function for the X-ray pipeline. If
you are looking for one because you're used to the neutron TOF world:
you don't need it. Egami & Billinge, *Underneath the Bragg Peaks* 2nd ed.
§5.6 discusses this explicitly.

This module remains only to hold the atomic-mass utility used elsewhere
(e.g. mean-mass estimates for cylinder MS scaling).
"""
from __future__ import annotations

from typing import Optional

# atomic masses (u) — used by several downstream modules that need a
# fraction-weighted mean atomic mass for the sample
_ATOMIC_MASS_U = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.380,
    "Ga": 69.723, "Ge": 72.640, "As": 74.922, "Se": 78.960, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.620, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.960, "Ag": 107.868, "Cd": 112.411, "In": 114.818,
    "Sn": 118.710, "Sb": 121.760, "Te": 127.600, "I": 126.904, "Cs": 132.905,
    "Ba": 137.327, "La": 138.906, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Sm": 150.360, "Eu": 151.964, "Gd": 157.250, "Tb": 158.925, "Dy": 162.500,
    "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.054, "Lu": 174.967,
    "Hf": 178.490, "Ta": 180.948, "W": 183.840, "Re": 186.207, "Os": 190.230,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.590, "Tl": 204.383,
    "Pb": 207.200, "Bi": 208.980, "Th": 232.038, "U": 238.029,
}


def mean_atomic_mass_u(composition) -> float:
    """Fraction-weighted mean atomic mass (u) of a ``Composition``."""
    total = 0.0
    for el, c in zip(composition.elements, composition.fractions):
        base = el.rstrip("+-0123456789") if any(k in el for k in "+-") else el
        M = _ATOMIC_MASS_U.get(base)
        if M is None:
            raise ValueError(f"no atomic mass for element {el!r}")
        total += c * M
    return float(total)


__all__ = ["mean_atomic_mass_u"]
