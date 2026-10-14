"""Known-truth test of the omega frame allocation of the two intensity models.

``IntensityModel`` (``fit_intensity`` default ``model_kind="psf"``) splits a spot linearly (a tent)
between the frame centres; ``SplatBlurModel(frame_model="gauss")`` integrates a Gaussian of width
``sigma_omega`` over each frame bin.

Convention (the package's own, ``params.py`` ``omega_step``): ``table.frame = (omega - omega_start)/step``
and frame ``j`` covers ``[j, j+1)`` (LEADING edge); a delta spot at ``f`` lies wholly in ``floor(f)``.
Truth for a spot at continuous ``f`` with omega width ``s`` is the exposure integral
``w_j = integral_j^(j+1) G(w; f, s) dw``, rendered here independently of the models (``erf`` and a
brute-force trapezoid cross-check, plus Lorentzian and box shapes the Gaussian model does not contain).
Metric: ``TV = 0.5 * sum_j |w_model_j - w_true_j|`` = the fraction of a spot's intensity put in the
wrong frame, averaged over stratified sub-frame offsets.

Measured 2026-10-12 (400 offsets, Gaussian truth; sigma_psf 1 px, no median filter):

    sigma_true   psf (tent)   gauss sigma_omega=0.2 (default)   gauss sigma_omega=sigma_true
    0 (delta)      0.250           0.160                             0.0005
    0.16           0.122           0.032                             0.0000
    0.30           0.037           0.080                             0.0001
    0.50           0.142           0.231                             0.0085

so neither model dominates: the tent equals a UNIFORM rocking curve one frame wide (sigma = 0.29
frame) and is right near that width, and wrong by 12-25 % for narrower spots; the gauss model is
right only when ``sigma_omega`` matches the data (it is fitted by ``fit_intensity(model_kind="splat")``).
These tests pin those numbers so that a change to either frame kernel is noticed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import ndtr

from midas_nf_fitorientation.intensity import (
    ContributionTable, IntensityModel, RecordedPixels, SplatBlurModel, fit_intensity,
)

J0, NF, TILE, NU = 10, 24, 16, 100
SIG_PSF = 1.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _truth(f, sigma, shape="gauss", n_frames=NF):
    f = np.asarray(f)[:, None]; lo = np.arange(n_frames)[None, :]; hi = lo + 1.0
    if sigma == 0:
        w = (np.floor(f) == lo).astype(float)
    elif shape == "gauss":
        w = ndtr((hi - f) / sigma) - ndtr((lo - f) / sigma)
    elif shape == "lorentz":
        g = 1.1774100225154747 * sigma
        w = (np.arctan((hi - f) / g) - np.arctan((lo - f) / g)) / math.pi
    else:
        raise ValueError(shape)
    return w / w.sum(1, keepdims=True)


def _model_frame_weights(kind, u, sigma_model=0.2):
    """Per-frame fraction of each single-row spot, from the real model (spots tiled along y)."""
    rng = np.random.default_rng(1)
    N = len(u)
    y = TILE * np.arange(N) + 7.5 + rng.uniform(0, 1, N); z = 20.0 + rng.uniform(0, 1, N)
    t = ContributionTable(grain=np.zeros(N, np.int64), refl=np.zeros(N, np.int64), sol=np.zeros(N, np.int8),
                          frame=J0 + u, y=y, z=z, w_geom=np.ones(N), log_F=np.zeros(1), n_grains=1)
    ny, nz = TILE * N, 40
    if kind == "psf":
        m = IntensityModel(t, NF, ny, nz, psf_radius=4, sigma_psf=SIG_PSF)
    elif kind == "linear":
        m = SplatBlurModel(t, NF, ny, nz, blur_radius=4, sigma_psf=SIG_PSF, frame_model="linear")
    else:
        m = SplatBlurModel(t, NF, ny, nz, blur_radius=4, sigma_psf=SIG_PSF, frame_model="gauss",
                           sigma_omega=sigma_model)
    with torch.no_grad():
        mu = m().numpy()
    fr, rem = np.divmod(m.support_key, ny * nz); yy, _ = np.divmod(rem, nz)
    W = np.zeros((N, NF)); np.add.at(W, (yy // TILE, fr), mu)
    assert np.allclose(W.sum(1), 1.0, atol=1e-9)       # unit mass per row, nothing lost off the support
    return W


def _tv(kind, sigma_true, sigma_model=0.2, shape="gauss"):
    u = (np.arange(NU) + 0.5) / NU
    W = _model_frame_weights(kind, u, sigma_model)
    return float((0.5 * np.abs(W - _truth(J0 + u, sigma_true, shape)).sum(1)).mean())


def test_truth_erf_box_equals_quadrature():
    f = J0 + np.array([0.05, 0.37, 0.5, 0.81, 0.99])
    for s in (0.16, 0.5):
        q = np.zeros((f.size, NF))
        for i, fi in enumerate(f):
            for j in range(int(fi) - 4, int(fi) + 5):
                x = np.linspace(j, j + 1, 20001)
                q[i, j] = _trapz(np.exp(-0.5 * ((x - fi) / s) ** 2) / (s * math.sqrt(2 * math.pi)), x)
        assert np.abs(_truth(f, s) - q).max() < 1e-6


def test_tent_is_linear_frame_model_and_hand_value():
    # both implementations of the tent agree row by row
    u = (np.arange(NU) + 0.5) / NU
    assert np.allclose(_model_frame_weights("psf", u), _model_frame_weights("linear", u), atol=1e-12)
    # hand value: delta spot at u=0.2 puts 0.7 in its own frame (floor) and 0.3 in the one before
    W = _model_frame_weights("psf", np.array([0.2]))[0]
    assert W[J0] == pytest.approx(0.7) and W[J0 - 1] == pytest.approx(0.3)


@pytest.mark.parametrize("sigma_true, tv_psf, tol", [(0.0, 0.250, 0.01), (0.16, 0.122, 0.01),
                                                       (0.30, 0.037, 0.01), (0.50, 0.142, 0.01)])
def test_tent_misallocation_by_omega_width(sigma_true, tv_psf, tol):
    assert _tv("psf", sigma_true) == pytest.approx(tv_psf, abs=tol)


@pytest.mark.parametrize("sigma_true, tv_max", [(0.0, 0.01), (0.16, 1e-3), (0.30, 1e-3), (0.50, 0.02)])
def test_gauss_allocation_with_matched_sigma(sigma_true, tv_max):
    assert _tv("gauss", sigma_true, sigma_model=max(sigma_true, 1e-3)) < tv_max


def test_neither_frame_model_dominates_at_fixed_default_width():
    """gauss at its default sigma_omega=0.2 beats the tent for narrow spots and loses for wide ones."""
    assert _tv("gauss", 0.16, 0.2) < 0.5 * _tv("psf", 0.16)
    assert _tv("gauss", 0.30, 0.2) > _tv("psf", 0.30)
    assert _tv("gauss", 0.50, 0.2) > _tv("psf", 0.50)


def test_non_gaussian_truth_is_not_allocated_better_by_either_model_at_fixed_width():
    """Lorentzian tails (sigma-matched): the matched-width gauss is still wrong by >15 %; sanity bound only."""
    assert _tv("gauss", 0.3, 0.3, "lorentz") > 0.15 and _tv("psf", 0.3, shape="lorentz") > 0.15


# ---- sigma_omega recovery on a censored scene rendered independently of the model code ----

def _scene(seed, sigma_true, nf=40, ny=96, nz=96, G=3, rpg=100, n_refl=5, p_true=2.0, sig_noise=2.0, blanket=5.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), rpg); R = g.size
    t = ContributionTable(grain=g, refl=rng.integers(0, n_refl, R), sol=np.zeros(R, np.int8),
                          frame=rng.uniform(3, nf - 4, R), y=rng.uniform(8, ny - 9, R), z=rng.uniform(8, nz - 9, R),
                          w_geom=rng.uniform(0.5, 2.0, R), log_F=np.log(np.linspace(1.0, 4.0, n_refl)), n_grains=G)
    amp = np.array([4.0, 6.0, 9.0])[g] * np.exp(p_true * t.log_F[t.refl]) * t.w_geom
    img = np.zeros((nf, ny, nz)); W = _truth(t.frame, sigma_true, n_frames=nf)
    for i in range(R):
        iy0, iz0 = int(round(t.y[i])), int(round(t.z[i]))
        iy = np.arange(iy0 - 6, iy0 + 7); iz = np.arange(iz0 - 6, iz0 + 7)
        py = ndtr((iy + 0.5 - t.y[i]) / SIG_PSF) - ndtr((iy - 0.5 - t.y[i]) / SIG_PSF)
        pz = ndtr((iz + 0.5 - t.z[i]) / SIG_PSF) - ndtr((iz - 0.5 - t.z[i]) / SIG_PSF)
        fr = np.nonzero(W[i] > 1e-6)[0]
        img[np.ix_(fr, iy, iz)] += amp[i] * W[i, fr][:, None, None] * (py[:, None] * pz[None, :])[None]
    x = img + rng.normal(0, sig_noise, img.shape); rec = x > blanket
    f_, y_, z_ = np.nonzero(rec)
    return t, RecordedPixels.from_arrays(f_, y_, z_, x[rec] - blanket, ny, nz), nf, ny, nz, blanket


@pytest.mark.parametrize("sigma_true", [0.16, 0.5])
def test_fit_intensity_splat_recovers_sigma_omega_and_p(sigma_true):
    t, R, nf, ny, nz, bl = _scene(2, sigma_true)
    fit = fit_intensity(t, R, nf, ny, nz, blanket=bl, p_init=1.5, sigma_psf_init=1.0, model_kind="splat",
                        psf_radius=5, sigma_omega_init=0.3)
    assert fit.frame_model == "gauss"
    assert fit.sigma_omega == pytest.approx(sigma_true, abs=0.03)
    assert fit.p == pytest.approx(2.0, abs=0.1)
