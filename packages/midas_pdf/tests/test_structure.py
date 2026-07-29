"""Differentiable small-box PDF structure modelling (PDFfit-style)."""
import numpy as np
import torch
from scipy.signal import find_peaks

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_pdf.structure import build_pair_list, pdffit_gr, refine_structure


def _ni(a=3.524, B=0.0):
    return Crystal(lattice=Lattice(a, a, a, 90, 90, 90),
                   space_group=SpaceGroup.from_number(225),
                   atoms=[Atom(element="Ni", fract=(0, 0, 0), B_iso=B)], name="Ni")


def test_fcc_peaks_at_known_shells():
    ct = _ni().to_torch()
    pairs = build_pair_list(ct, r_max=8.0)
    assert pairs.n_uc == 4                                  # FCC has 4 atoms/cell
    r = torch.linspace(1.5, 7.5, 1500, dtype=torch.float64)
    G = pdffit_gr(ct, r, pairs, scale=1.0, u_iso=0.005)
    pk, _ = find_peaks(G.numpy(), height=float(G.max()) * 0.12)
    rp = [round(float(r[i]), 2) for i in pk[:4]]
    for got, exp in zip(rp, [2.49, 3.52, 4.32, 4.98]):
        assert abs(got - exp) < 0.03, f"peak {got} != {exp}"


def test_forward_differentiable_in_lattice_uiso_scale():
    ct = _ni().to_torch()
    pairs = build_pair_list(ct, r_max=6.0)
    r = torch.linspace(1.5, 6.0, 600, dtype=torch.float64)
    lp = torch.tensor([3.524, 3.524, 3.524, 90, 90, 90], dtype=torch.float64, requires_grad=True)
    u = torch.tensor(0.005, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    G = pdffit_gr(ct, r, pairs, scale=s, u_iso=u, lattice_params=lp)
    G.pow(2).sum().backward()
    assert torch.isfinite(lp.grad[:3]).all() and torch.isfinite(u.grad) and torch.isfinite(s.grad)
    assert float(lp.grad[:3].abs().sum()) > 0


def test_refine_recovers_planted_parameters():
    a0, U0, s0 = 3.524, 0.0055, 1.0
    ct = _ni().to_torch()
    pairs = build_pair_list(ct, r_max=10.0)
    r = torch.linspace(1.5, 10.0, 1700, dtype=torch.float64)
    lp0 = torch.tensor([a0, a0, a0, 90, 90, 90], dtype=torch.float64)
    G_true = pdffit_gr(ct, r, pairs, scale=s0, u_iso=U0, lattice_params=lp0)
    rng = np.random.default_rng(0)
    sigma = 0.03 * torch.ones_like(G_true)
    G_obs = G_true + torch.tensor(rng.normal(0, 0.03, size=G_true.shape), dtype=torch.float64)

    res = refine_structure(ct, r, G_obs, pairs, sigma_obs=sigma,
                           init_a=3.60, init_u_iso=0.010, init_scale=0.8, steps=150)
    assert abs(res.fitted["a"] - a0) < 1e-3
    assert abs(res.fitted["u_iso"] - U0) < 5e-4
    assert abs(res.fitted["scale"] - s0) < 0.02
    assert 0.7 < res.chi2_reduced < 1.4            # error-aware chi2 well-calibrated


def test_uncertainty_scales_with_noise():
    a0, U0, s0 = 3.524, 0.0055, 1.0
    ct = _ni().to_torch()
    pairs = build_pair_list(ct, r_max=10.0)
    r = torch.linspace(1.5, 10.0, 1500, dtype=torch.float64)
    lp0 = torch.tensor([a0, a0, a0, 90, 90, 90], dtype=torch.float64)
    G_true = pdffit_gr(ct, r, pairs, scale=s0, u_iso=U0, lattice_params=lp0)
    rng = np.random.default_rng(1)

    def sig_a(noise):
        sigma = noise * torch.ones_like(G_true)
        G_obs = G_true + torch.tensor(rng.normal(0, noise, size=G_true.shape), dtype=torch.float64)
        res = refine_structure(ct, r, G_obs, pairs, sigma_obs=sigma,
                               init_a=3.55, init_u_iso=0.008, init_scale=0.9, steps=120)
        return res.uncertainty["a"]

    s_lo, s_hi = sig_a(0.02), sig_a(0.10)
    assert s_lo > 0 and s_hi > 0
    assert 3.0 < s_hi / s_lo < 7.0                 # ~linear growth (factor ~5)
