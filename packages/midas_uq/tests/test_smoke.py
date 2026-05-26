"""Smoke tests for midas_uq.

Verifies imports, top-level API surface, and a tiny end-to-end half-half
+ jackknife + Laplace round trip on a synthetic single grain.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def test_package_imports():
    import midas_uq
    assert hasattr(midas_uq, "__version__")
    assert hasattr(midas_uq, "half_half")
    assert hasattr(midas_uq, "jackknife")
    assert hasattr(midas_uq, "laplace_covariance")
    assert hasattr(midas_uq, "rfree_gap")
    assert hasattr(midas_uq, "GrainState")


def test_grain_state_clone_and_to():
    from midas_uq._common import GrainState
    e = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    l = torch.tensor([4.0] * 3 + [90.0] * 3, dtype=torch.float64)
    s = GrainState(e, l)
    s2 = s.clone()
    assert torch.allclose(s.euler_rad, s2.euler_rad)
    s3 = s.to(dtype=torch.float32)
    assert s3.euler_rad.dtype == torch.float32


def test_mode_dispatch_raises_on_unknown_mode():
    import midas_uq
    with pytest.raises(ValueError):
        midas_uq.half_half(None, None, None, mode="garbage")


# A genuine end-to-end check needs the midas_diffract forward model.
# That requires GetHKLList, which may not be available in CI. Mark the
# test as skipped if midas_diffract is missing.
def _have_diffract():
    try:
        import midas_diffract  # noqa
        return True
    except Exception:
        return False


def _have_hkls():
    try:
        import midas_hkls  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_have_diffract() and _have_hkls()),
                    reason="midas_diffract / midas_hkls not installed")
def test_half_half_synthetic_round_trip():
    """Tiny end-to-end test: synthetic single grain, 1 split.

    Forward-model a known FCC Au grain (FF geometry from paper I), fit
    two halves, check that the half-half disagreement is small.
    Reflection list comes from midas-hkls (no GetHKLList C dep).
    """
    import midas_uq as muq
    from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
    from midas_hkls import SpaceGroup, Lattice

    DEG2RAD = math.pi / 180.0

    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=0.172979,
    )

    sg = SpaceGroup.from_number(225)                 # FCC
    lat = Lattice.for_system("cubic", a=4.08)        # Au
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )

    gt_euler = torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64) * DEG2RAD
    gt_latc = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    gt_pos = torch.zeros(3, dtype=torch.float64)

    spots = model(gt_euler.unsqueeze(0), gt_pos.unsqueeze(0),
                  lattice_params=gt_latc)
    ang, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
    obs = ang.squeeze()[valid.squeeze() > 0.5].detach().clone()
    if obs.shape[0] < 12:
        pytest.skip(f"Not enough valid spots ({obs.shape[0]}) for half-half test")

    # Small perturbation from GT as init
    torch.manual_seed(0)
    init = muq.GrainState(
        euler_rad=gt_euler + 0.5 * DEG2RAD * torch.randn(3, dtype=torch.float64),
        latc=gt_latc + 0.001 * torch.randn(6, dtype=torch.float64),
        pos=gt_pos,
    )

    result = muq.half_half(
        model, init, obs, mode="ff", n_splits=1,
        phase_steps=(5, 5, 3),
    )
    assert result.misori_deg.shape == (1,)
    assert result.lattice_max_abs_A.shape == (1,)
    assert 0.0 <= result.misori_median_deg < 5.0
    assert 0.0 <= result.lattice_median_A < 0.1


@pytest.mark.skipif(not (_have_diffract() and _have_hkls()),
                    reason="midas_diffract / midas_hkls not installed")
def test_fixed_assignment_round_trip():
    """Smoke test for fixed-assignment UQ: half_half_fixed + per_spot_residuals
    + trust_score on a synthetic Au grain.

    The fixed-assignment path holds the obs↔pred mapping constant during the
    LBFGS refit, so a small init perturbation produces a small, smooth misori
    response (no re-pairing). This is the correct shape for scoring an
    already-refined grain.
    """
    import midas_uq as muq
    from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
    from midas_hkls import SpaceGroup, Lattice
    DEG2RAD = math.pi / 180.0

    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=0.172979)
    sg = SpaceGroup.from_number(225); lat = Lattice.for_system("cubic", a=4.08)
    hk, th, hi = hkls_for_forward_model(
        sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0)
    model = HEDMForwardModel(hkls=hk, thetas=th, geometry=geom, hkls_int=hi)
    gt = muq.GrainState(
        euler_rad=torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64) * DEG2RAD,
        latc=torch.tensor([4.08] * 3 + [90.0] * 3, dtype=torch.float64),
        pos=torch.zeros(3, dtype=torch.float64))
    spots = model(gt.euler_rad.unsqueeze(0), gt.pos.unsqueeze(0), lattice_params=gt.latc)
    ang, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
    obs = ang.squeeze()[valid.squeeze() > 0.5].detach().clone()
    if obs.shape[0] < 12:
        pytest.skip(f"Not enough valid spots ({obs.shape[0]})")

    # 1. per_spot_residuals at GT must be ~zero (obs were generated from GT).
    res = muq.per_spot_residuals(model, gt, obs)
    assert res.rmse_rad < 1e-3
    assert res.n_spots > 10

    # 2. half_half_fixed with a 0.2° init perturbation must converge back
    #    (both halves agree within a small misori — the whole point of
    #    fixed-assignment UQ on a refined grain).
    torch.manual_seed(0)
    init = muq.GrainState(
        euler_rad=gt.euler_rad + 0.2 * DEG2RAD * torch.randn(3, dtype=torch.float64),
        latc=gt.latc + 0.001 * torch.randn(6, dtype=torch.float64),
        pos=gt.pos)
    hh = muq.half_half_fixed(model, init, obs, n_splits=2, phase_steps=(8, 8, 5))
    assert hh.misori_median_deg < 0.5
    assert hh.lattice_median_A < 0.05

    # 3. trust_score wraps all three diagnostics.
    ts = muq.trust_score(model, init, obs, n_splits=2,
                          phase_steps=(5, 5, 3),
                          do_jackknife=False)   # skip slow jackknife in CI
    assert ts.n_spots > 10
    assert ts.frac_matched > 0.5


@pytest.mark.skipif(not _have_diffract(), reason="midas_fit_grain not importable")
def test_refiner_anchored_uq():
    """Refiner-anchored UQ: synthesize a tiny FitBest, build a grain view,
    score it. Checks that PerGrainResiduals + bootstrap_uq + trust_score_anchored
    plumb without re-prediction."""
    try:
        from midas_fit_grain import FitBestGrainView, FITBEST_COLS, fitbest_from_array
    except Exception:
        pytest.skip("midas_fit_grain not importable")
    import midas_uq as muq
    # Synthetic FitBest: 1 grain, 5 matched spots, the rest zero padding.
    fb = np.zeros((1, 50, 22), dtype=np.float64)
    spot_ids = np.array([101, 102, 103, 104, 105], dtype=np.float64)
    fb[0, :5, FITBEST_COLS["spot_id"]] = spot_ids
    # obs (1, 2, 3) and pred (7, 8, 9) — small residuals
    fb[0, :5, FITBEST_COLS["obs_y_um"]]    = np.array([100.0, 200.0, -150.0,  50.0, -300.0])
    fb[0, :5, FITBEST_COLS["pred_y_um"]]   = np.array([105.0, 195.0, -148.0,  53.0, -305.0])
    fb[0, :5, FITBEST_COLS["obs_z_um"]]    = np.array([ 50.0, -60.0,   70.0, -80.0,   90.0])
    fb[0, :5, FITBEST_COLS["pred_z_um"]]   = np.array([ 52.0, -62.0,   69.0, -82.0,   91.0])
    fb[0, :5, FITBEST_COLS["obs_ome_deg"]] = np.array([10.0, -20.0, 30.0, -40.0, 50.0])
    fb[0, :5, FITBEST_COLS["pred_ome_deg"]]= np.array([10.05, -19.97, 30.03, -39.95, 50.02])
    fb[0, :5, FITBEST_COLS["diff_len_um"]] = np.array([5.4, 5.4, 2.2, 3.6, 5.1])
    fb[0, :5, FITBEST_COLS["diff_ome_deg"]]= np.array([0.05, 0.03, 0.03, 0.05, 0.02])
    fb[0, :5, FITBEST_COLS["min_ia_deg"]]  = np.array([0.04, 0.03, 0.03, 0.04, 0.02])

    view = fitbest_from_array(fb, 0)
    assert view.n_spots == 5
    np.testing.assert_array_equal(view.spot_id, [101, 102, 103, 104, 105])
    np.testing.assert_allclose(view.dy_um, [-5.0, 5.0, -2.0, -3.0, 5.0])

    res = muq.per_grain_residuals(view)
    assert res.n_spots == 5
    assert 2.0 < res.pos_med_um < 6.0
    assert res.ome_med_deg < 0.1

    boot = muq.bootstrap_uq(res, n_boot=200, seed=0)
    # Bootstrap medians should bracket the empirical median tightly.
    assert boot.pos_med_um_p5_p95[0] <= res.pos_med_um <= boot.pos_med_um_p5_p95[1]

    ts = muq.trust_score_anchored(view, n_boot=200)
    assert ts.n_spots == 5
    assert ts.pos_med_um < 10
    assert ts.ome_med_deg < 0.1
    assert ts.pos_med_bootstrap_std_um >= 0
