import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.cross_section import differential_cross_section
from midas_pdf.ms import (
    multiple_scattering_mc,
    slab_optical_params,
    slab_single_scattering_factor,
)

WAVELENGTH_A = 0.1665


def test_single_scattering_factor_forward_limit():
    # psi -> 0 (Q -> 0): A1 -> t exp(-mu t)
    q = torch.tensor([1e-6], dtype=torch.float64)
    A1 = slab_single_scattering_factor(q, thickness_um=100.0, mu_um=2e-3,
                                       wavelength_A=WAVELENGTH_A)
    assert abs(float(A1[0]) - 100.0 * np.exp(-2e-3 * 100.0)) < 1e-3


def test_single_scattering_factor_differentiable():
    q = torch.linspace(0.5, 20.0, 50, dtype=torch.float64, requires_grad=True)
    A1 = slab_single_scattering_factor(q, thickness_um=200.0, mu_um=1e-3,
                                       wavelength_A=WAVELENGTH_A)
    A1.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))
    assert torch.all(A1 > 0)


def test_optical_params_sane():
    comp = Composition({"Si": 1, "O": 2})
    mu, tau, albedo = slab_optical_params(
        comp, wavelength_A=WAVELENGTH_A, thickness_um=1000.0, number_density_A3=0.066)
    assert mu > 0 and tau > 0
    assert 0.0 < albedo <= 1.0
    # at 74 keV in a light oxide, scattering dominates -> high albedo
    assert albedo > 0.5


def test_mc_beta_increases_with_optical_depth():
    comp = Composition({"Si": 1, "O": 2})
    r_lo = multiple_scattering_mc(comp, wavelength_A=WAVELENGTH_A, tau=0.2,
                                  albedo=0.9, n_photons=120_000, seed=1)
    r_hi = multiple_scattering_mc(comp, wavelength_A=WAVELENGTH_A, tau=1.5,
                                  albedo=0.9, n_photons=120_000, seed=1)
    beta_lo = r_lo["n_multiple"] / (r_lo["n_single"] + r_lo["n_multiple"])
    beta_hi = r_hi["n_multiple"] / (r_hi["n_single"] + r_hi["n_multiple"])
    assert beta_hi > beta_lo
    assert 0.0 <= beta_lo < beta_hi <= 1.0


def test_mc_thin_limit_negligible_multiple():
    comp = Composition({"Si": 1, "O": 2})
    r = multiple_scattering_mc(comp, wavelength_A=WAVELENGTH_A, tau=0.02,
                               albedo=0.9, n_photons=150_000, seed=2)
    beta = r["n_multiple"] / (r["n_single"] + r["n_multiple"])
    assert beta < 0.05            # thin slab: almost all single scattering


def test_mc_single_channel_matches_analytic_shape():
    """The MC single-scattering angular distribution must match the analytic
    prediction dσ/dΩ(psi)·sin(psi)·A1(psi)."""
    comp = Composition({"Si": 1, "O": 2})
    tau = 0.8
    r = multiple_scattering_mc(comp, wavelength_A=WAVELENGTH_A, tau=tau,
                               albedo=0.85, n_photons=400_000, n_psi=60, seed=3)
    psi = r["psi"]
    Q = r["Q"]
    dsig = differential_cross_section(Q, comp, wavelength_A=WAVELENGTH_A)
    A1 = slab_single_scattering_factor(Q, thickness_um=1.0, mu_um=tau,
                                       wavelength_A=WAVELENGTH_A)  # thickness=1 unit
    analytic = (dsig * torch.sin(psi) * A1).numpy()
    mc = r["I_single"].numpy()
    # compare shapes over bins with enough counts
    m = mc > mc.max() * 0.02
    a = analytic[m] / analytic[m].sum()
    b = mc[m] / mc[m].sum()
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.95, f"MC vs analytic single-scattering corr={corr:.3f}"
