"""Coded-aperture depth decode: de Bruijn code, NNLS baseline, regularized upgrade."""
import numpy as np
import pytest
import torch

from midas_dfxm.coded_aperture import (
    coded_forward,
    coded_identifiability,
    coding_matrix,
    de_bruijn,
    decode_nnls,
    decode_regularized,
    depth_axis,
)

DT = torch.float64


def test_de_bruijn_windows_unique():
    seq = de_bruijn(order=8, k=2)
    assert seq.shape[0] == 256
    assert set(np.unique(seq)) <= {0, 1}
    # every 8-bit cyclic window is unique -> 256 distinct windows
    ext = np.concatenate([seq, seq[:7]])
    windows = {tuple(ext[i : i + 8]) for i in range(256)}
    assert len(windows) == 256


def test_coding_matrix_and_forward_shapes():
    code = de_bruijn(order=8)
    A = coding_matrix(code, n_positions=120, n_depth=40, dtype=DT)
    assert A.shape == (120, 40)
    assert set(torch.unique(A).tolist()) <= {0.0, 1.0}
    s = torch.zeros(40, dtype=DT)
    s[10] = 1.0
    d = coded_forward(s, A)
    assert d.shape == (120,)
    # single delta -> measurement is exactly the shifted code column
    assert torch.allclose(d, A[:, 10])


def test_identifiability_full_rank():
    A = coding_matrix(de_bruijn(order=8), n_positions=120, n_depth=40, dtype=DT)
    info = coded_identifiability(A)
    assert info["recoverable"]
    assert info["rank"] == 40
    assert np.isfinite(info["cond"])


@pytest.mark.unit
def test_nnls_recovers_clean_depth_profile():
    A = coding_matrix(de_bruijn(order=8), n_positions=150, n_depth=40, dtype=DT)
    rng = np.random.default_rng(0)
    s_true = torch.tensor(np.clip(rng.normal(0, 1, 40), 0, None), dtype=DT)
    s_true[5:9] += 3.0                                   # a localized feature
    d = coded_forward(s_true, A)
    s_hat = decode_nnls(d, A)
    assert torch.allclose(s_hat, s_true, atol=1e-6)


@pytest.mark.unit
def test_forward_is_differentiable():
    A = coding_matrix(de_bruijn(order=8), n_positions=80, n_depth=30, dtype=DT)
    s = torch.rand(30, dtype=DT, requires_grad=True)
    coded_forward(s, A).sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


@pytest.mark.slow
def test_regularized_beats_nnls_under_noise_and_drift():
    # Few aperture positions + Poisson noise + per-position beam drift -> the
    # regime the paper flags for a regularized/compressive solve.
    rng = np.random.default_rng(1)
    ny, nx, N = 8, 8, 24
    A = coding_matrix(de_bruijn(order=8), n_positions=70, n_depth=N, dtype=DT)
    P = ny * nx
    # smooth ground-truth depth field: a depth ridge that shifts across pixels
    xx = np.linspace(0, 1, nx)
    centers = (6 + 10 * xx)[None, :].repeat(ny, 0).reshape(-1)     # (P,)
    zi = np.arange(N)[:, None]
    S = np.exp(-0.5 * ((zi - centers[None, :]) / 2.0) ** 2) * 5.0  # (N, P), non-neg, smooth
    S_t = torch.tensor(S, dtype=DT)
    d = coded_forward(S_t, A)                                       # (M, P)
    drift = torch.tensor(1.0 + 0.15 * rng.standard_normal(d.shape[0]), dtype=DT)[:, None]
    d_noisy = torch.tensor(rng.poisson(np.clip((d * drift).numpy(), 0, None)), dtype=DT)

    s_nnls = decode_nnls(d_noisy, A)
    s_reg = decode_regularized(d_noisy, A, lambda_depth=2.0, lambda_pixel=2.0,
                               shape=(ny, nx), steps=500, lr=0.05)

    err_nnls = (s_nnls - S_t).pow(2).mean().sqrt()
    err_reg = (s_reg - S_t).pow(2).mean().sqrt()
    assert (s_reg >= -1e-8).all()                                  # non-negative
    assert err_reg < err_nnls                                      # regularization helps


def test_depth_axis_geometry():
    z = depth_axis(21, aperture_step_um=1.0, two_theta_deg=20.0)
    assert z.shape == (21,)
    assert torch.allclose(z.mean(), torch.zeros((), dtype=DT), atol=1e-9)  # centred
    pitch = float(z[1] - z[0])
    assert pitch == pytest.approx(1.0 / np.sin(np.deg2rad(20.0)), rel=1e-6)
