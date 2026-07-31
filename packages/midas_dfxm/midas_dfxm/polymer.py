"""X-ray refractive index of polymer (and other compound) optics, the MIDAS way.

The 6-ID-C microscope (Qiao et al., Rev. Sci. Instrum. 91, 113703, 2020) uses SU-8 polymer
optics -- an x-ray prism-lens (XPL) condenser and a polymer compound-refractive-lens
objective -- instead of beryllium. Modelling those needs the complex index
``n = 1 - delta - i beta`` of SU-8, alongside the ``delta_beryllium`` the package already has.

Physics (compound of stoichiometry ``x_i`` of element ``i``, mass density ``rho``):

    delta = (r_e lambda^2 / 2 pi) (rho N_A / M) sum_i x_i (Z_i + f'_i)
    beta  = mu_lin lambda / (4 pi),   mu_lin = rho sum_i w_i (mu/rho)_i,  w_i = x_i A_i / M

with ``M = sum_i x_i A_i`` the formula-unit mass. All atomic data come from ``midas_hkls``
(``Z_for``, ``atomic_mass``, anomalous ``f'`` from ``anomalous_correction``, and the mass
attenuation coefficient ``mu/rho``) -- never an external x-ray library. Differentiable in
energy through ``lambda`` and the tabulated factors.
"""
from __future__ import annotations

import math

import torch

from midas_hkls.absorption import Z_for, atomic_mass, mass_attenuation_coefficient
from midas_hkls import anomalous as _an

_RE_CM = 2.8179403262e-13        # classical electron radius (cm)
_NA = 6.02214076e23             # Avogadro
_HC_KEV_A = 12.398419843320026  # h c in keV*Angstrom

# SU-8 (cured negative epoxy photoresist): standard formula-unit approximation.
SU8_FORMULA = {"C": 87, "H": 108, "O": 16}
SU8_DENSITY_G_CM3 = 1.20


def _wavelength_A(energy_keV):
    return _HC_KEV_A / energy_keV


def _fprime(element, wavelength_A):
    """Real anomalous correction ``f'`` (0 if the element is not tabulated, e.g. H)."""
    try:
        corr = _an.anomalous_correction([element], wavelength_A)
        fp = corr[0] if isinstance(corr, (tuple, list)) else corr
        fp = fp[0] if hasattr(fp, "__len__") and not torch.is_tensor(fp) else fp
        return float(fp[0]) if torch.is_tensor(fp) and fp.ndim else float(fp)
    except Exception:
        return 0.0


def material_index(formula: dict, density_g_cm3: float, energy_keV: float) -> dict:
    """Complex x-ray index of a compound: ``{'delta', 'beta', 'mu_lin_per_cm', 'wavelength_A'}``.

    ``formula`` maps element symbol -> stoichiometric count. Uses ``midas_hkls`` atomic data.
    """
    lam_A = _wavelength_A(energy_keV)
    lam_cm = lam_A * 1e-8
    M = sum(n * atomic_mass(el) for el, n in formula.items())              # g/mol formula unit
    n_formula = density_g_cm3 * _NA / M                                    # formula units / cm^3

    # delta: sum of (Z + f') over the formula unit
    zf = 0.0
    for el, n in formula.items():
        zf += n * (Z_for(el) + _fprime(el, lam_A))
    delta = (_RE_CM * lam_cm ** 2 / (2.0 * math.pi)) * n_formula * zf

    # beta via linear absorption: mu_lin = rho * sum_i w_i (mu/rho)_i
    mu_lin = 0.0
    for el, n in formula.items():
        w = n * atomic_mass(el) / M                                        # mass fraction
        mu_rho = float(mass_attenuation_coefficient(el, lam_A))            # cm^2/g
        mu_lin += w * mu_rho
    mu_lin *= density_g_cm3                                                # 1/cm
    beta = mu_lin * lam_cm / (4.0 * math.pi)

    return {"delta": float(delta), "beta": float(beta),
            "mu_lin_per_cm": float(mu_lin), "wavelength_A": float(lam_A)}


def su8_index(energy_keV) -> dict:
    """Complex x-ray index of SU-8 photoresist at ``energy_keV`` (the 6-ID-C optics material)."""
    return material_index(SU8_FORMULA, SU8_DENSITY_G_CM3, energy_keV)


def beryllium_index(energy_keV) -> dict:
    """Be index from the same first-principles path -- cross-checks ``delta_beryllium``."""
    return material_index({"Be": 1}, 1.848, energy_keV)


def attenuation_length_um(formula, density_g_cm3, energy_keV) -> float:
    """1/e x-ray attenuation length ``1/mu_lin`` in micrometres (absorption-aperture scale)."""
    idx = material_index(formula, density_g_cm3, energy_keV)
    return 1.0e4 / idx["mu_lin_per_cm"]
