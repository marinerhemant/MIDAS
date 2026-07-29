"""Cylinder (capillary) multiple scattering: analog MC reference, and the
differentiable slab-equivalent (transport with cylinder_effective_tau) validated
against it.
"""
import numpy as np
import torch

from midas_pdf import (Composition, cylinder_effective_tau,
                       multiple_scattering_mc_cylinder, slab_transport_ms)
from midas_pdf.ms import CYLINDER_SLAB_FACTOR

WL = 0.1665


def test_cylinder_mc_beta_increases_with_radius():
    comp = Composition({"Si": 1, "O": 2})
    lo = multiple_scattering_mc_cylinder(comp, wavelength_A=WL, tau_radius=0.2,
                                         albedo=0.85, n_photons=150_000, seed=1)
    hi = multiple_scattering_mc_cylinder(comp, wavelength_A=WL, tau_radius=1.0,
                                         albedo=0.85, n_photons=150_000, seed=1)
    blo = lo["n_multiple"] / (lo["n_single"] + lo["n_multiple"])
    bhi = hi["n_multiple"] / (hi["n_single"] + hi["n_multiple"])
    assert 0.0 <= blo < bhi <= 1.0


def test_cylinder_effective_tau_helper():
    assert cylinder_effective_tau(2e-3, 500.0) == CYLINDER_SLAB_FACTOR * 2e-3 * 500.0


def test_transport_eff_tau_matches_cylinder_mc():
    """slab transport at tau = CYLINDER_SLAB_FACTOR * mu * R reproduces the
    cylinder MC beta(Q) within a few percent."""
    comp = Composition({"Si": 1, "O": 2})
    tau_radius = 0.6
    cyl = multiple_scattering_mc_cylinder(comp, wavelength_A=WL, tau_radius=tau_radius,
                                          albedo=0.85, n_photons=400_000, n_psi=40, seed=1)
    tau_eff = cylinder_effective_tau(1.0, tau_radius)   # mu=1 -> tau_eff = factor*tau_radius
    tr = slab_transport_ms(comp, wavelength_A=WL, tau=tau_eff, albedo=0.85,
                           q_max=20.0, n_mu=24, n_tau=70)
    Qg = np.linspace(5.0, 17.0, 7)
    cb = np.interp(Qg, cyl["Q"].numpy(), cyl["beta"].numpy())
    tb = np.interp(Qg, tr["Q"].numpy(), tr["beta"].numpy())
    rms = float(np.sqrt(np.mean((tb - cb) ** 2)))
    assert rms < 0.05, f"transport-eff-tau vs cylinder MC rms = {rms:.3f}"
