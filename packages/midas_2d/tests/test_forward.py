"""Phase 1 forward-model tests: continuous-q structure factor agrees with the
integer-hkl midas_hkls path at Bragg points; intensity assembly + gradients."""
import pytest
import torch

from midas_2d import build_crystal_tensor, rod_intensity, structure_factor_q

DT = torch.float64


@pytest.mark.unit
def test_continuous_F_matches_midas_hkls_at_integers():
    """structure_factor_q at integer hkl must equal midas_hkls.structure_factors
    (B_iso = 0, non-anomalous) -- proves we reuse the same physics."""
    from midas_hkls.structure_factor import structure_factors

    ct = build_crystal_tensor()  # zinc-blende CdSe, B_iso = 0
    hkl = torch.tensor([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]], dtype=DT)

    F_cont = structure_factor_q(ct, hkl)
    F_ref = structure_factors(ct, hkl.long())

    assert torch.allclose(F_cont, F_ref, atol=1e-8), (F_cont, F_ref)


@pytest.mark.unit
def test_rod_intensity_nonnegative_and_finite():
    ct = build_crystal_tensor()
    l = torch.linspace(0.5, 3.5, 600, dtype=DT)
    hkl = torch.stack([torch.full_like(l, 1.0), torch.full_like(l, 1.0), l], dim=-1)
    N = torch.tensor([1e4, 1e4, 4.0], dtype=DT)
    I = rod_intensity(ct, hkl, N, wavelength_A=1.0, apply_lp=False)
    assert torch.isfinite(I).all()
    assert (I >= 0).all()


@pytest.mark.unit
def test_zincblende_111_allowed_200_weak():
    """Zinc-blende: (111) strong, (200) is the weak/mixed reflection. Verify
    |F(200)|^2 < |F(111)|^2 (the structure-factor contrast that makes CdSe a
    sensible demo)."""
    ct = build_crystal_tensor()
    hkl = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]], dtype=DT)
    F = structure_factor_q(ct, hkl)
    F2 = F.real ** 2 + F.imag ** 2
    assert F2[0] > F2[1]


@pytest.mark.autograd
def test_rod_intensity_grad_in_layer_count():
    """Intensity is differentiable w.r.t. the (continuous) layer count N3."""
    ct = build_crystal_tensor(requires_grad={"lattice": True})
    l = torch.linspace(0.6, 1.4, 40, dtype=DT)
    hkl = torch.stack([torch.full_like(l, 1.0), torch.full_like(l, 1.0), l], dim=-1)
    n3 = torch.tensor(4.0, dtype=DT, requires_grad=True)
    N = torch.stack([torch.tensor(1e4, dtype=DT), torch.tensor(1e4, dtype=DT), n3])
    I = rod_intensity(ct, hkl, N, wavelength_A=1.0, apply_lp=False)
    I.sum().backward()
    assert n3.grad is not None and torch.isfinite(n3.grad)
    assert ct.lattice_params.grad is not None
    assert torch.isfinite(ct.lattice_params.grad).all()
