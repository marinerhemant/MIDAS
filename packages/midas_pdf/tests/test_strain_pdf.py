"""Strain-sensitive (azimuthally-sliced) PDF: physics limits + recovery."""
import numpy as np
import pytest
import torch

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_pdf.structure import build_pair_list, pdffit_gr
from midas_pdf import strain_pdf as sp


def _ni(a=3.524, B=0.0):
    return Crystal(lattice=Lattice(a, a, a, 90, 90, 90),
                   space_group=SpaceGroup.from_number(225),
                   atoms=[Atom(element="Ni", fract=(0, 0, 0), B_iso=B)],
                   name="Ni")


@pytest.fixture(scope="module")
def ni_setup():
    ct = _ni().to_torch()
    pairs = build_pair_list(ct, r_max=8.0)
    r = torch.linspace(1.8, 7.5, 400, dtype=torch.float64)
    return ct, pairs, r


def test_zero_strain_matches_isotropic(ni_setup):
    ct, pairs, r = ni_setup
    q = sp.probe_directions([0.0, 90.0])
    G = sp.sliced_gr_stack(ct, r, pairs, q, torch.zeros(6, dtype=torch.float64),
                           kernel_m=8, n_quad=64, u_iso=0.005)
    G0 = pdffit_gr(ct, r, pairs, u_iso=0.005)
    for d in range(G.shape[0]):
        assert torch.allclose(G[d], G0, atol=1e-10)


def test_pure_dilation_equals_scaled_lattice(ni_setup):
    """eps = delta*I scales every distance by (1+delta) regardless of kernel."""
    ct, pairs, r = ni_setup
    delta = 5.0e-4
    eps = torch.tensor([delta, delta, delta, 0, 0, 0], dtype=torch.float64)
    q = sp.probe_directions([37.0])
    G = sp.sliced_gr_stack(ct, r, pairs, q, eps, kernel_m=8, n_quad=64,
                           u_iso=0.005)[0]
    a = 3.524 * (1.0 + delta)
    lat = torch.tensor([a, a, a, 90, 90, 90], dtype=torch.float64)
    G_ref = pdffit_gr(ct, r, pairs, lattice_params=lat, u_iso=0.005)
    assert torch.allclose(G, G_ref, atol=1e-9)


def test_uniaxial_strain_shifts_aligned_slice_most(ni_setup):
    """e33 > 0 shifts the eta=0 (q || z) slice's first peak out by ~e33*r,
    and the eta=90 (q || y) slice much less."""
    ct, pairs, r = ni_setup
    e33 = 2.0e-3
    eps = torch.tensor([0, 0, e33, 0, 0, 0], dtype=torch.float64)
    q = sp.probe_directions([0.0, 90.0])
    G = sp.sliced_gr_stack(ct, r, pairs, q, eps, kernel_m=16, n_quad=256,
                           u_iso=0.005)
    G0 = pdffit_gr(ct, r, pairs, u_iso=0.005)

    def first_peak_centroid(g):
        w = g.clamp(min=0.0)
        m = (r > 2.2) & (r < 2.8)                      # FCC first shell ~2.49
        return float((r[m] * w[m]).sum() / w[m].sum())

    shift_para = first_peak_centroid(G[0]) - first_peak_centroid(G0)
    shift_perp = first_peak_centroid(G[1]) - first_peak_centroid(G0)
    expected = e33 * 2.49
    # aligned slice: right sign, majority of the full shift (kernel dilutes)
    assert shift_para > 0.5 * expected
    # perpendicular slice sees far less than the aligned one
    assert abs(shift_perp) < 0.5 * shift_para


def test_isotropic_kernel_is_deviatoric_blind(ni_setup):
    """m=0 (isotropic PDF): rank-1 Fisher -- volumetric only (Prototype-2)."""
    ct, pairs, r = ni_setup
    q = sp.probe_directions([0.0, 45.0, 90.0, 135.0])
    res = sp.strain_crlb(ct, r, pairs, q, sigma_G=0.02, kernel_m=0,
                         n_quad=64, u_iso=0.005)
    assert res["rank"] == 1
    # the one determinable direction is the hydrostatic axis (1,1,1,0,0,0)/sqrt3
    evals, evecs = torch.linalg.eigh(res["fisher"])
    v = evecs[:, -1].abs()
    hydro = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float64) / np.sqrt(3)
    assert float((v * hydro).sum()) > 0.99


def test_slicing_plus_tilts_gives_full_rank(ni_setup):
    """One detector: in-plane subspace only. Adding +-45 deg tilts: rank 6."""
    ct, pairs, r = ni_setup
    etas = [0.0, 45.0, 90.0, 135.0]
    q1 = sp.probe_directions(etas)
    res1 = sp.strain_crlb(ct, r, pairs, q1, sigma_G=0.02, kernel_m=8,
                          n_quad=96, u_iso=0.005, rank_rtol=1e-6)
    q3 = sp.probe_directions(etas, tilts_deg=(-45.0, 0.0, 45.0))
    res3 = sp.strain_crlb(ct, r, pairs, q3, sigma_G=0.02, kernel_m=8,
                          n_quad=96, u_iso=0.005, rank_rtol=1e-6)
    assert res1["rank"] < 6
    assert res3["rank"] == 6
    assert torch.isfinite(res3["per_component_ue"]).all()


def test_forward_differentiable_in_strain(ni_setup):
    ct, pairs, r = ni_setup
    eps = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    q = sp.probe_directions([0.0])
    G = sp.sliced_gr_stack(ct, r, pairs, q, eps, kernel_m=8, n_quad=32,
                           u_iso=0.005)
    G.pow(2).sum().backward()
    assert torch.isfinite(eps.grad).all()
    assert float(eps.grad.abs().sum()) > 0


def test_recover_strain_with_tilts(ni_setup):
    """Full-tensor recovery from noisy sliced PDFs (3 tilts x 4 wedges)."""
    ct, pairs, r = ni_setup
    rng = np.random.default_rng(7)
    truth = torch.tensor(rng.normal(scale=5e-4, size=6), dtype=torch.float64)
    q = sp.probe_directions([0.0, 45.0, 90.0, 135.0],
                            tilts_deg=(-45.0, 0.0, 45.0))
    kw = dict(kernel_m=8, n_quad=96, u_iso=0.005)
    y = sp.sliced_gr_stack(ct, r, pairs, q, truth, **kw)
    y = y + 0.02 * torch.as_tensor(rng.normal(size=y.shape))
    res = sp.recover_strain(ct, r, pairs, q, y, n_iter=3, rank_rtol=1e-6, **kw)
    err_ue = 1e6 * (res["strain"] - truth).abs()
    crlb = sp.strain_crlb(ct, r, pairs, q, sigma_G=0.02, rank_rtol=1e-6, **kw)
    # every component recovered within 4x its CRLB
    assert (err_ue < 4.0 * crlb["per_component_ue"] + 1e-6).all(), \
        f"err={err_ue.tolist()} crlb={crlb['per_component_ue'].tolist()}"
