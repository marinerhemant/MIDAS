"""Phase 3 tests: strain observable, design matrix, identifiability, recovery, UQ."""
import math

import pytest
import torch

from midas_dfxm import make_uniform_field, with_uniform_strain
from midas_dfxm.inverse import (
    normal_strain,
    recover_strain_direct,
    recover_strain_regularised,
    strain_covariance,
    strain_design_matrix,
    strain_identifiability,
)

DT = torch.float64

# A reflection set that spans all six strain components (rank 6).
FULL_SET = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (0, 2, 2), (2, 0, 2), (2, 2, 2)]
# A deficient set (only diagonal strains constrained).
POOR_SET = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]


def _voigt6_to_tensor(v):
    return torch.tensor([[v[0], v[5], v[4]],
                         [v[5], v[1], v[3]],
                         [v[4], v[3], v[2]]], dtype=DT)


# --------------------------------------------------------------------------
# observable
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_normal_strain_recovers_uniaxial():
    e = 1e-3
    field = make_uniform_field(shape=(4, 4, 1), dtype=DT)
    eps = torch.diag(torch.tensor([e, 0.0, 0.0], dtype=DT))
    field = with_uniform_strain(field, eps)
    # g=[2,0,0] is along x -> normal strain == e.
    ns = normal_strain(field, (2, 0, 0))
    assert ns.mean().item() == pytest.approx(e, rel=1e-4)
    # g=[0,2,0] perpendicular -> ~0.
    assert normal_strain(field, (0, 2, 0)).abs().max().item() < 1e-6


# --------------------------------------------------------------------------
# design matrix / identifiability
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_full_set_is_rank6_poor_set_is_not():
    good = strain_identifiability(FULL_SET, dtype=DT)
    poor = strain_identifiability(POOR_SET, dtype=DT)
    assert good["rank"] == 6 and good["recoverable"]
    assert poor["rank"] == 3 and not poor["recoverable"]
    assert math.isinf(poor["cond"])


@pytest.mark.unit
def test_design_row_matches_projection():
    M = strain_design_matrix([(1, 1, 0)], dtype=DT)
    ghat = torch.tensor([1.0, 1.0, 0.0], dtype=DT) / math.sqrt(2)
    eps = _voigt6_to_tensor([1e-3, -5e-4, 0, 0, 0, 2e-4])
    expect = ghat @ eps @ ghat
    got = (M[0] @ torch.tensor([1e-3, -5e-4, 0, 0, 0, 2e-4], dtype=DT))
    assert got.item() == pytest.approx(expect.item(), rel=1e-9)


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------
def _planted_strain_field(n=12):
    # Smooth, band-limited strain field across x (the elastic-field regime a
    # curvature prior is designed for).
    x = torch.linspace(-1, 1, n, dtype=DT)
    eps6 = torch.zeros(n, 6, dtype=DT)
    eps6[:, 0] = 1e-3 * torch.sin(2 * x)     # eps11
    eps6[:, 1] = -5e-4 * x                    # eps22
    eps6[:, 5] = 3e-4 * torch.cos(x)          # eps12
    return eps6  # (n, 6), voxels along x


@pytest.mark.unit
def test_direct_recovery_exact_clean():
    eps6 = _planted_strain_field()
    M = strain_design_matrix(FULL_SET, dtype=DT)
    measured = (eps6 @ M.transpose(0, 1)).transpose(0, 1)  # (K, N) clean
    rec = recover_strain_direct(measured, FULL_SET)
    assert torch.allclose(rec, eps6, atol=1e-9)


@pytest.mark.slow
def test_regularised_beats_direct_at_low_snr():
    # Honest regime statement: at HIGH SNR the direct per-voxel solve is already
    # near-optimal and regularisation is ~neutral; the curvature prior earns its
    # keep at LOW SNR, where it denoises the band-limited field. Averaged over
    # noise realisations to avoid single-draw luck.
    n = 40
    eps6 = _planted_strain_field(n)
    M = strain_design_matrix(FULL_SET, dtype=DT)
    clean = (eps6 @ M.transpose(0, 1)).transpose(0, 1)  # (K, N)
    err_d, err_r = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        measured = clean + 1e-3 * torch.randn_like(clean)
        direct = recover_strain_direct(measured, FULL_SET)
        reg = recover_strain_regularised(measured, FULL_SET, shape=(n, 1, 1),
                                         lambda_smooth=3.0, steps=1500, lr=3e-2)
        err_d.append((direct - eps6).abs().mean())
        err_r.append((reg - eps6).abs().mean())
    mean_direct = torch.stack(err_d).mean()
    mean_reg = torch.stack(err_r).mean()
    assert mean_reg < 0.75 * mean_direct


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_covariance_worse_for_ill_conditioned_set():
    cov_good = strain_covariance(FULL_SET, noise_std=1e-4, dtype=DT)
    # A near-deficient set inflates variances (add one off-diagonal reflection to
    # make it just barely rank-6).
    weak_set = FULL_SET[:5] + [(2, 2, 1), (1, 2, 2)]
    cov_weak = strain_covariance(weak_set, noise_std=1e-4, dtype=DT)
    assert torch.diag(cov_weak).max() > torch.diag(cov_good).max()


@pytest.mark.autograd
def test_normal_strain_differentiable():
    field = make_uniform_field(shape=(3, 3, 1), dtype=DT)
    field.F.requires_grad_(True)
    ns = normal_strain(field, (1, 1, 1))
    ns.sum().backward()
    assert field.F.grad is not None and torch.isfinite(field.F.grad).all()
