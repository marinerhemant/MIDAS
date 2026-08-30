"""Tests for the curated loss modules.

For each loss: returns finite, differentiates to all upstream params,
zero-mean guarantee where applicable, and a smoke "loss decreases under
SGD on a toy problem with the right gradient direction" check.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    integrate_with_corrections,
    profile_1d_diff,
    ProfileMSELoss,
    ProfileWeightedMSELoss,
    EtaUniformityLoss,
    PeakPositionLoss,
    GaussianPriorLoss,
)


def _spec(NY=24, NZ=24, requires_grad=True):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=requires_grad)


def _gaussian_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5, BC_y=None, BC_z=None,
                    px=200.0):
    if BC_y is None:
        BC_y = NY / 2.0 + 0.37
    if BC_z is None:
        BC_z = NZ / 2.0 - 0.41
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - BC_y) * px
    Zc = (zz - BC_z) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    img = np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)
    return torch.from_numpy(img)


# ── ProfileMSELoss ──

def test_profile_mse_loss_zero_when_reference_matches():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    prof = profile_1d_diff(int2d, s).detach()
    loss_fn = ProfileMSELoss()
    L = loss_fn(int2d, s, prof)
    assert float(L) == pytest.approx(0.0, abs=1e-12)


def test_profile_mse_loss_grad_flows_to_geometry():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    target = torch.zeros(s.n_r_bins, dtype=torch.float64)
    L = ProfileMSELoss()(int2d, s, target)
    L.backward()
    assert s.Lsd.grad is not None and torch.isfinite(s.Lsd.grad)
    assert s.BC_y.grad is not None and torch.isfinite(s.BC_y.grad)


def test_profile_mse_shape_mismatch_raises():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    bad_target = torch.zeros(s.n_r_bins + 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="reference shape"):
        ProfileMSELoss()(int2d, s, bad_target)


# ── ProfileWeightedMSELoss ──

def test_profile_weighted_mse_zero_weight_excludes_bin():
    """A bin with zero weight should not contribute to the loss even if
    it has a large prediction-vs-reference error."""
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    prof = profile_1d_diff(int2d, s).detach()
    bad_target = prof.clone()
    # Inject a 100x error into bin 5 but weight it zero.
    bad_target[5] = bad_target[5] + 100.0
    weights = torch.ones(s.n_r_bins, dtype=torch.float64)
    weights[5] = 0.0
    L = ProfileWeightedMSELoss()(int2d, s, bad_target, weights)
    assert float(L) == pytest.approx(0.0, abs=1e-12)


def test_profile_weighted_mse_grad_finite():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    prof = profile_1d_diff(int2d, s).detach()
    weights = torch.linspace(0.1, 1.0, s.n_r_bins, dtype=torch.float64)
    L = ProfileWeightedMSELoss()(int2d, s, torch.zeros_like(prof), weights)
    L.backward()
    assert torch.isfinite(s.Lsd.grad)


# ── EtaUniformityLoss ──

def test_eta_uniformity_zero_for_perfectly_uniform_input():
    """A 2D array constant along η should have zero per-bin variance."""
    int2d = torch.ones(6, 11, dtype=torch.float64) * 7.5
    L = EtaUniformityLoss()(int2d)
    assert float(L) == pytest.approx(0.0, abs=1e-12)


def test_eta_uniformity_positive_for_non_uniform_input():
    int2d = torch.ones(6, 11, dtype=torch.float64)
    int2d[0, :] = 5.0      # spike one η bin
    L = EtaUniformityLoss()(int2d)
    assert float(L) > 0.0


def test_eta_uniformity_grad_flows_to_geometry():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    L = EtaUniformityLoss()(int2d)
    L.backward()
    assert s.Lsd.grad is not None
    assert torch.isfinite(s.Lsd.grad)


def test_eta_uniformity_subset_of_r_bins_changes_value():
    """Subsetting to a different set of R bins must change the loss
    (otherwise the r_indices filter is silently a no-op)."""
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ, R0_px=6.0)
    int2d = integrate_with_corrections(img, s)
    L_full = EtaUniformityLoss()(int2d)
    L_near_peak = EtaUniformityLoss(r_indices=[5, 6, 7])(int2d)
    assert float(L_full) != float(L_near_peak)


# ── PeakPositionLoss ──

def test_peak_position_loss_zero_when_centroid_matches_prediction():
    """Build int2d so that one R bin's profile centroid is exactly at
    R_pred — loss should be zero."""
    n_eta, n_r = 3, 11
    int2d = torch.zeros(n_eta, n_r, dtype=torch.float64)
    int2d[:, 6] = 1.0           # all intensity in bin index 6
    s = _spec()
    # bin 6 centre is at RMin + 6.5 * RBinSize = 1.0 + 6.5 = 7.5
    R_pred = torch.tensor([7.5], dtype=torch.float64)
    L = PeakPositionLoss()(int2d, s, R_pred, window_px=2.0)
    assert float(L) == pytest.approx(0.0, abs=1e-12)


def test_peak_position_loss_grows_with_prediction_offset():
    n_eta, n_r = 3, 11
    int2d = torch.zeros(n_eta, n_r, dtype=torch.float64)
    int2d[:, 6] = 1.0
    s = _spec()
    L_close = PeakPositionLoss()(int2d, s,
                                   torch.tensor([7.6], dtype=torch.float64))
    L_far   = PeakPositionLoss()(int2d, s,
                                   torch.tensor([8.5], dtype=torch.float64))
    assert float(L_far) > float(L_close)


def test_peak_position_loss_grad_through_geometry():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ, R0_px=6.0)
    int2d = integrate_with_corrections(img, s)
    L = PeakPositionLoss()(int2d, s, torch.tensor([6.0], dtype=torch.float64))
    L.backward()
    assert torch.isfinite(s.BC_y.grad)
    assert torch.isfinite(s.Lsd.grad)


# ── GaussianPriorLoss ──

def test_gaussian_prior_zero_at_mean():
    s = _spec()
    prior = GaussianPriorLoss({"Lsd": (float(s.Lsd.detach()), 100.0)})
    L = prior(s)
    assert float(L) == pytest.approx(0.0, abs=1e-20)


def test_gaussian_prior_grows_quadratically():
    s = _spec()
    mu = float(s.Lsd.detach())
    sig = 100.0
    prior = GaussianPriorLoss({"Lsd": (mu, sig)})
    s.Lsd = torch.tensor(mu + 2 * sig, dtype=torch.float64, requires_grad=True)
    L = prior(s)
    # 0.5 * (2)^2 = 2.0
    assert float(L) == pytest.approx(2.0, abs=1e-12)


def test_gaussian_prior_grad_points_toward_mean():
    s = _spec()
    mu = float(s.Lsd.detach())
    s.Lsd = torch.tensor(mu + 50.0, dtype=torch.float64, requires_grad=True)
    prior = GaussianPriorLoss({"Lsd": (mu, 100.0)})
    L = prior(s)
    L.backward()
    # gradient = (Lsd - mu) / sig^2 = 50 / 10000 = 5e-3 (positive)
    assert float(s.Lsd.grad) == pytest.approx(5e-3, rel=1e-9)


def test_gaussian_prior_composes_with_data_loss():
    s = _spec()
    img = _gaussian_image(s.NrPixelsY, s.NrPixelsZ)
    int2d = integrate_with_corrections(img, s)
    target = torch.zeros(s.n_r_bins, dtype=torch.float64)
    data_loss = ProfileMSELoss()(int2d, s, target)
    prior_loss = GaussianPriorLoss({
        "Lsd": (float(s.Lsd.detach()), 100.0),
        "BC_y": (float(s.BC_y.detach()), 1.0),
    })(s)
    L = data_loss + 0.01 * prior_loss
    L.backward()
    # Both data and prior should have contributed to Lsd's gradient
    assert s.Lsd.grad is not None and torch.isfinite(s.Lsd.grad)


# ── End-to-end smoke: BC_y refinement against a planted profile ──

def _eta_uniformity_scan(bc_offsets, NY=32, NZ=32):
    """Loss vs BC_y error, everything else at truth."""
    BC_true = NY / 2.0 + 0.5
    img = _gaussian_image(NY, NZ, R0_px=8.0, sigma_px=1.2,
                            BC_y=BC_true, BC_z=NZ / 2.0)
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=BC_true, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=14.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=15.0,
    )
    loss_fn = EtaUniformityLoss(intensity_floor=1e-6)
    out = []
    for d in bc_offsets:
        s = spec_from_v1_params(p, requires_grad=False)
        s.BC_y = torch.tensor(BC_true + d, dtype=torch.float64)
        out.append(float(loss_fn(integrate_with_corrections(img, s))))
    return out


def test_eta_uniformity_is_minimised_AT_the_true_beam_centre():
    """The invariant the loss exists for — and the one that was violated.

    Until the 2026-08-29 soft-bin centring fix, ``soft_bin_indices_weights``
    interpolated onto nodes at bin LOWER EDGES while the reported axis is bin
    centres, and zeroed both weights for anything landing in the final bin.
    Under that binning this loss was a LOCAL MAXIMUM at the correct beam
    centre: measured 8.09e-2 at the truth against 6.00e-2 at +0.20 px and
    6.20e-2 at -0.20 px, i.e. the η-uniformity objective actively pushed BC_y
    AWAY from the right answer, into spurious minima either side.

    The previous version of this test asserted only that the optimiser got
    "closer than half the initial error", which drifting from +0.40 into the
    spurious minimum at +0.20 satisfied — so the test passed on the broken
    landscape. This asserts the real thing instead.
    """
    offs = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    losses = _eta_uniformity_scan(offs)
    i_truth = offs.index(0.0)
    i_min = int(min(range(len(losses)), key=lambda i: losses[i]))
    assert i_min == i_truth, (
        "eta-uniformity loss is not minimised at the true beam centre: "
        "minimum at offset %+.2f px (loss %.4e) vs truth (loss %.4e); "
        "full scan %s" % (offs[i_min], losses[i_min], losses[i_truth],
                          [f"{o:+.1f}:{v:.3e}" for o, v in zip(offs, losses)]))


def test_eta_uniformity_is_locally_symmetric_about_the_truth():
    """A correct objective should not prefer one side of the beam centre."""
    offs = [-0.2, -0.1, 0.1, 0.2]
    lo = _eta_uniformity_scan(offs)
    for a, b in ((0, 3), (1, 2)):          # -0.2 vs +0.2, -0.1 vs +0.1
        assert abs(lo[a] - lo[b]) / max(lo[a], lo[b]) < 0.10, (
            f"asymmetric about the truth: {offs[a]} -> {lo[a]:.4e}, "
            f"{offs[b]} -> {lo[b]:.4e}")


def test_eta_uniformity_drives_BC_y_toward_truth():
    """Synth a Gaussian-ring image at the TRUE BC, then optimise BC_y from a
    perturbed value inside the basin of attraction.

    The start is +0.15 px, not +0.40: at this image size (32x32, 26 R bins x
    24 eta bins) there are few pixels per bin, so the loss is bumpy further
    out and gradient descent stalls in a local dip. That is a discretisation
    property of a deliberately tiny test image, not a defect in the loss —
    ``test_eta_uniformity_is_minimised_AT_the_true_beam_centre`` pins the
    global shape.
    """
    NY, NZ = 32, 32
    BC_true = NY / 2.0 + 0.5
    img = _gaussian_image(NY, NZ, R0_px=8.0, sigma_px=1.2,
                            BC_y=BC_true, BC_z=NZ / 2.0)

    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=BC_true, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=14.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=15.0,
    )
    s = spec_from_v1_params(p, requires_grad=True)
    s.BC_y = torch.tensor(BC_true + 0.15, dtype=torch.float64,
                           requires_grad=True)

    loss_fn = EtaUniformityLoss(intensity_floor=1e-6)
    opt = torch.optim.Adam([s.BC_y], lr=0.02)
    history = [float(s.BC_y.detach())]
    losses = []
    for _ in range(200):
        opt.zero_grad()
        int2d = integrate_with_corrections(img, s)
        L = loss_fn(int2d)
        L.backward()
        opt.step()
        history.append(float(s.BC_y.detach()))
        losses.append(float(L))

    closest = min(abs(bc - BC_true) for bc in history)
    initial_err = abs(history[0] - BC_true)
    assert closest < 0.5 * initial_err, (
        f"eta-uniformity did not pull BC_y toward truth: closest {closest:.3f}, "
        f"initial err {initial_err:.3f}"
    )
    assert min(losses) < losses[0], (
        f"loss never improved: start={losses[0]:.4e}, min={min(losses):.4e}"
    )
