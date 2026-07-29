"""Rev-15 anisotropic-ADP + partial-occupancy PDF tests."""
from __future__ import annotations

import pytest
import torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup

from midas_pdf.aniso_refine import (
    refine_aniso_occupancy,
    u_matrix_to_vector,
    u_vector_to_matrix,
)
from midas_pdf.structure import build_pair_list, pdffit_gr


def _fcc_ni():
    return Crystal(
        lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))],
    ).to_torch()


# ---------------------------------------------------------------------------
# Vector <-> matrix packers
# ---------------------------------------------------------------------------

def test_u_vec_matrix_round_trip():
    v = torch.tensor([[0.006, 0.005, 0.007, 0.001, 0.0005, 0.0]], dtype=torch.float64)
    U = u_vector_to_matrix(v)
    assert U.shape == (1, 3, 3)
    assert torch.allclose(U[0], U[0].T)                        # symmetric
    v2 = u_matrix_to_vector(U)
    assert torch.allclose(v, v2)


def test_u_vec_shape_check():
    with pytest.raises(ValueError, match="u_vec"):
        u_vector_to_matrix(torch.zeros(5, dtype=torch.float64))


# ---------------------------------------------------------------------------
# pdffit_gr: aniso limit + occupancy
# ---------------------------------------------------------------------------

def test_pairlist_has_indices():
    ni = _fcc_ni()
    pairs = build_pair_list(ni, r_max=8.0)
    assert pairs.i_idx is not None and pairs.j_idx is not None
    assert pairs.i_idx.shape == pairs.j_idx.shape
    assert int(pairs.i_idx.max()) < pairs.n_uc
    assert int(pairs.j_idx.max()) < pairs.n_uc


def test_pdffit_gr_aniso_matches_iso_on_scalar_U():
    """u_aniso = u_iso · I should give identical G(r) to u_iso."""
    ni = _fcc_ni()
    r = torch.linspace(1.5, 8.0, 200, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    U_iso = 0.006
    U = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(pairs.n_uc, 1, 1) * U_iso
    G_aniso = pdffit_gr(ni, r, pairs, scale=1.0, u_aniso=U)
    G_iso = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=U_iso)
    assert torch.allclose(G_aniso, G_iso, atol=1e-10)


def test_pdffit_gr_aniso_differentiable_in_U():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 6.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=7.0)
    U = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(pairs.n_uc, 1, 1) * 0.006
    U = U.clone().requires_grad_(True)
    loss = pdffit_gr(ni, r, pairs, scale=1.0, u_aniso=U).sum()
    loss.backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all()


def test_pdffit_gr_aniso_wrong_shape_raises():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 4.0, 50, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=5.0)
    bad = torch.eye(3, dtype=torch.float64).unsqueeze(0)               # (1, 3, 3)
    with pytest.raises(ValueError, match="u_aniso must be shape"):
        pdffit_gr(ni, r, pairs, scale=1.0, u_aniso=bad)


def test_pdffit_gr_zero_occupancy_gives_zero_G():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 8.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    zero = torch.zeros(pairs.n_uc, dtype=torch.float64)
    G = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.006, occupancy=zero)
    assert float(G.abs().max()) == 0.0


def test_pdffit_gr_occupancy_differentiable():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 6.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=7.0)
    occ = torch.full((pairs.n_uc,), 0.7, dtype=torch.float64).requires_grad_(True)
    loss = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.006, occupancy=occ).sum()
    loss.backward()
    assert (occ.grad != 0).any()


def test_pdffit_gr_occupancy_wrong_length_raises():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 4.0, 50, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=5.0)
    with pytest.raises(ValueError, match="occupancy must have length"):
        pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.006,
                   occupancy=torch.tensor([0.5, 0.5], dtype=torch.float64))


# ---------------------------------------------------------------------------
# refine_aniso_occupancy: end-to-end recovery
# ---------------------------------------------------------------------------

def test_refine_aniso_recovers_lattice_and_scale():
    """LBFGS with aniso ADPs still recovers isotropic-truth lattice + scale."""
    ni_true = _fcc_ni()
    r = torch.linspace(1.5, 8.0, 200, dtype=torch.float64)
    pairs = build_pair_list(ni_true, r_max=9.0)
    U_true = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(
        pairs.n_uc, 1, 1) * 0.006
    G_true = pdffit_gr(ni_true, r, pairs, scale=1.0, u_aniso=U_true)
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.02 * torch.randn(G_true.shape, generator=rng,
                                        dtype=torch.float64)

    res = refine_aniso_occupancy(
        ni_true, r, G_obs, pairs,
        sigma_obs=torch.full_like(G_obs, 0.02),
        init_a=3.60,
        refine_aniso=True,
        refine_occupancy=False,
        steps=150,
    )
    assert abs(res.fitted["a"] - 3.524) < 0.01
    assert 0.5 < res.fitted["scale"] < 1.5
    assert res.chi2_reduced < 5.0
    U_fit = torch.tensor(res.fitted["u_aniso"], dtype=torch.float64)
    assert U_fit.shape == (pairs.n_uc, 3, 3)


def test_refine_occupancy_recovers_scale_times_occ_product():
    """After the Rev-15 partial-occ normalisation fix (/Σocc not /n_uc),
    both the pair-density term and the -4πrρ0 baseline scale LINEARLY with
    the uniform occupancy η — which correctly captures the physics but
    makes η and the global ``scale`` degenerate against a free-scale
    synthetic.  So we test that the identifiable quantity
    ``scale × η`` recovers the truth.

    On real data the absolute normalisation of G(r) pins scale=1, at
    which point η is identifiable — but in a synthetic bench test with
    a floating scale, only the product is."""
    ni_true = _fcc_ni()
    r = torch.linspace(1.5, 8.0, 200, dtype=torch.float64)
    pairs = build_pair_list(ni_true, r_max=9.0)
    occ_true = torch.full((pairs.n_uc,), 0.75, dtype=torch.float64)
    G_true = pdffit_gr(ni_true, r, pairs, scale=1.0, u_iso=0.006,
                        occupancy=occ_true)
    rng = torch.Generator().manual_seed(1)
    G_obs = G_true + 0.01 * torch.randn(G_true.shape, generator=rng,
                                        dtype=torch.float64)

    res = refine_aniso_occupancy(
        ni_true, r, G_obs, pairs,
        sigma_obs=torch.full_like(G_obs, 0.01),
        refine_aniso=False,
        refine_occupancy=True,
        steps=150,
    )
    occ_fit = torch.tensor(res.fitted["occupancy"], dtype=torch.float64)
    scale_fit = float(res.fitted["scale"])
    product = scale_fit * float(occ_fit.mean())
    assert abs(product - 0.75) < 0.05, \
        f"scale={scale_fit}, occ={float(occ_fit.mean())}, product={product}"


def test_refine_aniso_gives_positive_definite_U():
    ni = _fcc_ni()
    r = torch.linspace(1.5, 6.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=7.0)
    U_true = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(
        pairs.n_uc, 1, 1) * 0.005
    G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_aniso=U_true)
    res = refine_aniso_occupancy(
        ni, r, G_true, pairs,
        sigma_obs=torch.full_like(G_true, 0.01),
        refine_aniso=True, steps=100,
    )
    U_fit = torch.tensor(res.fitted["u_aniso"], dtype=torch.float64)
    eigs = torch.linalg.eigvalsh(U_fit)                         # (n_uc, 3)
    assert (eigs > -1e-6).all(), f"aniso U not positive-definite: {eigs}"
