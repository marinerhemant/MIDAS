"""All-orders multiple scattering by discrete-ordinates radiative transfer must
match the analog Monte-Carlo reference pointwise in Q (and far better than the
scalar geometric-series resummation).
"""
import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.ms import multiple_scattering_mc
from midas_pdf.ms_transport import phase_matrix, slab_transport_ms

WL = 0.1665


def test_phase_matrix_columns_normalizable():
    comp = Composition({"Si": 1, "O": 2})
    nodes, weights = np.polynomial.legendre.leggauss(24)
    mu = torch.as_tensor(nodes, dtype=torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64)
    P = phase_matrix(comp, WL, mu)
    assert torch.all(P >= 0)
    assert torch.all(torch.isfinite(P))


def test_transport_matches_mc_pointwise():
    comp = Composition({"Si": 1, "O": 2})
    tau, alb = 1.0, 0.85
    tr = slab_transport_ms(comp, wavelength_A=WL, tau=tau, albedo=alb,
                           q_max=20.0, n_mu=24, n_tau=70)
    mc = multiple_scattering_mc(comp, wavelength_A=WL, tau=tau, albedo=alb,
                                n_photons=500_000, n_psi=40, seed=1)
    Qg = np.linspace(4.0, 18.0, 8)
    tr_b = np.interp(Qg, tr["Q"].numpy(), tr["beta"].numpy())
    mc_b = np.interp(Qg, mc["Q"].numpy(), mc["beta"].numpy())
    ratio = tr_b / np.clip(mc_b, 1e-6, None)
    assert 0.85 < float(np.median(ratio)) < 1.15, f"median transport/MC = {np.median(ratio):.3f}"
    assert float(np.corrcoef(tr_b, mc_b)[0, 1]) > 0.95


def test_transport_beats_geometric_series():
    """At large tau the geometric series under-predicts; transport must be much
    closer to the MC."""
    comp = Composition({"Si": 1, "O": 2})
    tau = 1.5
    tr = slab_transport_ms(comp, wavelength_A=WL, tau=tau, albedo=0.85,
                           q_max=20.0, n_mu=24, n_tau=70)
    mc = multiple_scattering_mc(comp, wavelength_A=WL, tau=tau, albedo=0.85,
                                n_photons=400_000, n_psi=40, seed=2)
    from midas_pdf.ms import slab_double_scattering
    geo = slab_double_scattering(comp, wavelength_A=WL, tau=tau, albedo=0.85,
                                 q_max=20.0, geometric_series=True)
    Qg = np.linspace(5.0, 17.0, 6)
    tr_b = np.interp(Qg, tr["Q"].numpy(), tr["beta"].numpy())
    mc_b = np.interp(Qg, mc["Q"].numpy(), mc["beta"].numpy())
    geo_b = np.interp(Qg, geo["Q"].numpy(), geo["beta"].numpy())
    err_tr = np.mean(np.abs(tr_b - mc_b))
    err_geo = np.mean(np.abs(geo_b - mc_b))
    assert err_tr < err_geo, f"transport err {err_tr:.3f} not < geometric err {err_geo:.3f}"


def test_transport_differentiable_in_albedo():
    comp = Composition({"Si": 1, "O": 2})
    alb = torch.tensor(0.85, dtype=torch.float64, requires_grad=True)
    tr = slab_transport_ms(comp, wavelength_A=WL, tau=0.8, albedo=alb,
                           q_max=20.0, n_mu=12, n_tau=30)
    tr["beta"].sum().backward()
    assert alb.grad is not None and torch.isfinite(alb.grad)
