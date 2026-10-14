"""Tests for the opt-in exposure-integrated frame axis of ``splat_rois_3d``.

The default (``frame_integrate=False``) must stay bit-identical to the
pre-change code; the new mode replaces the frame factor
``exp(-df^2 / (2 s^2))`` by ``sqrt(2 pi) s [Phi((df+1/2)/s) - Phi((df-1/2)/s)]``.

Most tests drive the splat with controlled frame positions: the forward model
is run on a tiny real bank, then ``pink_forward_stacked`` is wrapped so that
every kept spot sits exactly on its ROI centre pixel (the (y, z) factor is then
exactly 1 at the central pixel) and at a prescribed sub-frame offset from its
centre frame. The central-pixel frame profile is then compared with an
independent ``math.erf`` implementation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("midas_diffract")
pytest.importorskip("midas_hkls")

import midas_diffract as md  # noqa: E402
import midas_pink as mp  # noqa: E402
import midas_pink.inverse as inv  # noqa: E402
from midas_hkls import Lattice, SpaceGroup  # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
ROI_H = ROI_W = 5
HALF = ROI_H // 2

# Argument sets used to generate tests/data/splat3d_default_ref.npz from the
# ORIGINAL (pre-change) splat_rois_3d.
LOCK_CASES = {
    "default": dict(sigma_psf_px=1.5),
    "sig035": dict(sigma_psf_px=1.2, sigma_frame=0.35, roi_frames=5,
                   intensity_per_spot=2.5),
    "mos": dict(sigma_psf_px=1.5, mosaicity_rad=1e-3),
}


@pytest.fixture(scope="module")
def tiny():
    spec = mp.ParameterisedSpectrum(
        E0_keV=71.6764, half_bw=0.02, n_samples=5,
        init_kind="gaussian", init_rel_bw=2e-4,
        fixed=True, dtype=torch.float64,
    )

    def geom_factory(lam_A):
        return md.HEDMGeometry(
            Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
            omega_start=-180.0, omega_step=0.25, n_frames=1440,
            n_pixels_y=2048, n_pixels_z=2048,
            min_eta=6.0, wavelength=lam_A,
        )

    bank = mp.build_pink_bank(
        spec, space_group=SpaceGroup.from_number(225),
        lattice=Lattice.for_system("cubic", a=4.078),
        geom_factory=geom_factory, two_theta_max_deg=6.0,
        device="cpu", dtype=torch.float64,
    )
    euler = torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64) * math.pi / 180.0
    pos = torch.zeros(3, dtype=torch.float64)
    latc = torch.tensor([4.078] * 3 + [90.0] * 3, dtype=torch.float64)
    plan = mp.plan_rois_from_state(
        bank, euler.unsqueeze(0), pos.unsqueeze(0),
        lattice_params=latc, roi_h=ROI_H, roi_w=ROI_W,
    )
    assert plan.n_kept > 0
    return bank, plan, euler.unsqueeze(0), pos.unsqueeze(0), latc


def _call(tiny, **kw):
    bank, plan, euler, pos, latc = tiny
    return mp.splat_rois_3d(bank, plan, euler, pos, latc, **kw)


def _controlled(monkeypatch, plan, offsets, delta=None):
    """Wrap pink_forward_stacked: each kept spot at its ROI centre pixel, at
    frame = round(original centre frame) + offsets[k] (+ delta), same for all
    energies, all valid."""
    real = inv.pink_forward_stacked
    offsets = torch.as_tensor(offsets, dtype=torch.float64)

    def fake(*a, **k):
        omegas, etas, tt, ys, zs, frames, valids = real(*a, **k)
        N_E = ys.shape[0]
        ki, mi = plan.spot_kidx, plan.spot_midx
        base = torch.round(frames[N_E // 2, ki, mi].detach())
        fpos = base + offsets
        if delta is not None:
            fpos = fpos + delta
        ys2, zs2, fr2, va2 = ys.clone(), zs.clone(), frames.clone(), valids.clone()
        ys2[:, ki, mi] = plan.centres_y.to(ys.dtype).unsqueeze(0).expand(N_E, -1)
        zs2[:, ki, mi] = plan.centres_z.to(zs.dtype).unsqueeze(0).expand(N_E, -1)
        fr2[:, ki, mi] = fpos.unsqueeze(0).expand(N_E, -1)
        va2[:, ki, mi] = torch.ones((), dtype=valids.dtype)
        return omegas, etas, tt, ys2, zs2, fr2, va2

    monkeypatch.setattr(inv, "pink_forward_stacked", fake)


def _expected_profile(offsets, s, integrated, roi_frames=7):
    """Independent (math.erf / math.exp) frame profile, shape (n, roi_frames)."""
    half = roi_frames // 2
    out = np.zeros((len(offsets), roi_frames))
    for i, off in enumerate(offsets):
        for j, f in enumerate(range(-half, half + 1)):
            df = f - off
            if integrated:
                hi = (df + 0.5) / s
                lo = (df - 0.5) / s
                out[i, j] = SQ2PI * s * 0.5 * (
                    math.erf(hi / math.sqrt(2.0)) - math.erf(lo / math.sqrt(2.0)))
            else:
                out[i, j] = math.exp(-df * df / (2.0 * s * s))
    return out


def _offsets(n):
    return np.linspace(-0.45, 0.45, n)


# ----------------------------------------------------------------------------
# T1: analytic profile
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
def test_T1_integrated_profile_matches_erf(tiny, monkeypatch, sigma):
    plan = tiny[1]
    offs = _offsets(plan.n_kept)
    _controlled(monkeypatch, plan, offs)
    r = _call(tiny, sigma_psf_px=1.5, sigma_frame=sigma, frame_integrate=True)
    prof = r[:, :, HALF, HALF].detach().numpy()
    exp = _expected_profile(offs, sigma, integrated=True)
    assert np.max(np.abs(prof - exp)) < 1e-10
    # sanity: the sampled mode follows its own analytic form (separate path)
    r0 = _call(tiny, sigma_psf_px=1.5, sigma_frame=sigma)
    exp0 = _expected_profile(offs, sigma, integrated=False)
    assert np.max(np.abs(r0[:, :, HALF, HALF].detach().numpy() - exp0)) < 1e-10
    # and the two modes really differ (the flag does something)
    assert np.max(np.abs(prof - exp0)) > 1e-3


# ----------------------------------------------------------------------------
# T2: normalisation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("sigma,roi_frames", [(0.3, 7), (0.5, 7), (0.8, 7), (0.8, 3)])
def test_T2_integrated_normalisation(tiny, monkeypatch, sigma, roi_frames):
    plan = tiny[1]
    offs = _offsets(plan.n_kept)
    _controlled(monkeypatch, plan, offs)
    r = _call(tiny, sigma_psf_px=1.5, sigma_frame=sigma,
              roi_frames=roi_frames, frame_integrate=True)
    total = r[:, :, HALF, HALF].sum(dim=1).detach().numpy() / (SQ2PI * sigma)
    # Exact truncation: mass of the Gaussian inside [-(h+1/2)-off, (h+1/2)-off]
    h = roi_frames // 2
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    inside = np.array([Phi(((h + 0.5) - o) / sigma) - Phi((-(h + 0.5) - o) / sigma)
                       for o in offs])
    assert np.max(np.abs(total - inside)) < 1e-12
    deficit = 1.0 - total
    assert np.all(deficit >= -1e-12)
    if roi_frames == 7 and sigma <= 0.5:
        # truncation bound for the default window: tail beyond ~(3.05/sigma)
        # sigmas, ~1e-12 at sigma 0.3 and ~6e-10 at 0.5 (bound 1e-8)
        bound = 1e-12 if sigma == 0.3 else 1e-8
        assert deficit.max() < bound
        # offset independent within the same bound
        assert total.max() - total.min() < bound
    if roi_frames == 7 and sigma == 0.8:
        assert deficit.max() < 1e-4   # 2*Phi(-(3.5-0.45)/0.8) ~ 1.4e-4 worst-case bound
    if roi_frames == 3 and sigma == 0.8:
        assert deficit.max() > 0.05   # truncation is real and is reported, not hidden


def test_T2_sampled_profile_is_offset_dependent_at_sigma_03(tiny, monkeypatch):
    """Documents the old behaviour the flag fixes."""
    plan = tiny[1]
    offs = _offsets(plan.n_kept)
    _controlled(monkeypatch, plan, offs)
    r0 = _call(tiny, sigma_psf_px=1.5, sigma_frame=0.3)
    sampled = r0[:, :, HALF, HALF].sum(dim=1).detach().numpy() / (SQ2PI * 0.3)
    assert sampled.max() - sampled.min() > 0.3
    r1 = _call(tiny, sigma_psf_px=1.5, sigma_frame=0.3, frame_integrate=True)
    integ = r1[:, :, HALF, HALF].sum(dim=1).detach().numpy() / (SQ2PI * 0.3)
    assert integ.max() - integ.min() < 1e-12
    assert abs(integ.mean() - 1.0) < 1e-12


# ----------------------------------------------------------------------------
# T3: default unchanged (bit-identical to the pre-change code)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(LOCK_CASES))
def test_T3_default_equals_flag_off(tiny, name):
    """Omitting ``frame_integrate`` and passing False give the same array.

    Exact equality is safe here: both calls run in this process, on one libm.
    The default branch of ``splat_rois_3d`` is the unchanged pre-change
    expression, so the default output is the legacy output by construction.

    A stored reference array made on another machine was dropped from this
    test: its inputs (the ``tiny`` fixture) did not reproduce on a second
    macOS install (the difference from the stored array was O(1) and was the
    same for the pre-change ``inverse.py`` from git HEAD), so it checked the
    environment, not the code.
    """
    kw = LOCK_CASES[name]
    a = _call(tiny, **kw).detach().numpy()
    b = _call(tiny, frame_integrate=False, **kw).detach().numpy()
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a)) and np.all(a >= 0)


def test_T3_flag_changes_output(tiny):
    kw = LOCK_CASES["default"]
    ref = _call(tiny, **kw).detach().numpy()
    c = _call(tiny, frame_integrate=True, **kw).detach().numpy()
    assert not np.array_equal(c, ref)
    assert np.all(np.isfinite(c)) and np.all(c >= 0)


def test_config_default_and_forwarding(tiny, monkeypatch):
    cfg = mp.RecoveryConfig() if hasattr(mp, "RecoveryConfig") else inv.RecoveryConfig()
    assert cfg.frame_integrate is False
    seen = {}
    real = inv.splat_rois_3d

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(inv, "splat_rois_3d", spy)
    bank, plan, euler, pos, latc = tiny
    obs = real(bank, plan, euler, pos, latc, sigma_psf_px=1.5)
    for flag in (False, True):
        cfg = inv.RecoveryConfig(use_3d_loss=True, frame_integrate=flag,
                                 phase1_steps=1, phase2_steps=0, phase3_steps=0,
                                 lbfgs_max_iter=1, fit_position=False,
                                 fit_lattice=False)
        inv.recover_grain_state(bank, plan, obs, euler[0], pos[0], latc, cfg)
        assert seen["frame_integrate"] is flag


# ----------------------------------------------------------------------------
# T4: gradient w.r.t. the frame position
# ----------------------------------------------------------------------------
def test_T4_gradient_finite_matches_fd_and_nonzero_between_cells(tiny, monkeypatch):
    """Spot (within 1e-4 frames of) the midpoint between two cell centres,
    sigma_eff = 0.4. The ROI window is anchored on round(frame), so the spot is
    placed 1e-4 below the exact midpoint and the finite-difference step (2e-5)
    stays inside one anchor cell; the observation is rendered at a smaller
    offset (same anchor). A window re-anchoring would otherwise shift the
    compared cells and make the loss discontinuous (a property of the ROI
    design, not of this change)."""
    plan = tiny[1]
    n = plan.n_kept
    sigma = 0.4
    off = 0.5 - 1e-4

    def run(offset, delta_val, obs, requires_grad=False):
        delta = torch.tensor(delta_val, dtype=torch.float64, requires_grad=requires_grad)
        _controlled(monkeypatch, plan, np.full(n, offset), delta=delta)
        pred = _call(tiny, sigma_psf_px=1.5, sigma_frame=sigma, frame_integrate=True)
        loss = ((pred - obs) ** 2).sum() if obs is not None else None
        return delta, pred, loss

    with torch.no_grad():
        _, obs, _ = run(off - 0.15, 0.0, None)
    delta, _, loss = run(off, 0.0, obs, requires_grad=True)
    loss.backward()
    g = float(delta.grad)
    assert math.isfinite(g)
    assert abs(g) > 1e-3
    eps = 2e-5
    with torch.no_grad():
        lp = float(run(off, +eps, obs)[2])
        lm = float(run(off, -eps, obs)[2])
    fd = (lp - lm) / (2 * eps)
    # relative 1e-6 (FD roundoff is ~1e-13 / eps from the absolute frame number)
    assert abs(g - fd) < 1e-6 * abs(g), (g, fd)

    # tiny residual shift: the gradient is still clearly non-zero
    with torch.no_grad():
        _, obs2, _ = run(off - 0.01, 0.0, None)
    d2, _, l2 = run(off, 0.0, obs2, requires_grad=True)
    l2.backward()
    assert abs(float(d2.grad)) > 1e-4

    # exactly at the midpoint: autograd gradient finite and non-zero
    d3, _, l3 = run(0.5, 0.0, obs, requires_grad=True)
    l3.backward()
    assert math.isfinite(float(d3.grad)) and abs(float(d3.grad)) > 1e-6


# ----------------------------------------------------------------------------
# T5: mosaicity path gives the same sigma_eff in both modes
# ----------------------------------------------------------------------------
def test_T5_mosaicity_sigma_eff_same_in_both_modes(tiny, monkeypatch):
    bank, plan = tiny[0], tiny[1]
    offs = _offsets(plan.n_kept)
    _controlled(monkeypatch, plan, offs)
    mos = 2e-4
    sigma_frame = 0.5
    omega_step = abs(bank.models[0].omega_step)
    sig_mos_f = mos * 180.0 / math.pi / math.sin(4.0 * math.pi / 180.0) / omega_step
    s_eff = math.sqrt(sigma_frame ** 2 + sig_mos_f ** 2)
    assert s_eff > sigma_frame + 0.2   # mosaicity really widens it

    for integrated in (False, True):
        r = _call(tiny, sigma_psf_px=1.5, sigma_frame=sigma_frame,
                  mosaicity_rad=mos, frame_integrate=integrated)
        prof = r[:, :, HALF, HALF].detach().numpy()
        assert np.max(np.abs(prof - _expected_profile(offs, s_eff, integrated))) < 1e-10
        # and clearly not the no-mosaicity width
        assert np.max(np.abs(prof - _expected_profile(offs, sigma_frame, integrated))) > 1e-2
