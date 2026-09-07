"""Synthetic wiring test for the frozen-point calibration pipeline.

Paints clean, well-separated Gaussian ring spots at a known, deliberately
tilted geometry, seeds :func:`autocalibrate_frozen_point` from a nearby but
wrong starting geometry, and checks that it recovers the known geometry.
This is a wiring smoke test (point_pick <-> FittedDataset <-> residual <->
LM), not a convergence-basin characterisation -- the seed offset here is
small enough that per-ring 2theta windows don't cross-contaminate.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

from midas_calibrate_v2.pipelines.frozen_point import (
    autocalibrate_frozen_point, iterate_frozen_point_until_stable,
)

pytest.importorskip("midas_integrate")


NY, NZ = 700, 700
PX_UM = 200.0
WAVELENGTH_A = 0.1729

TRUE = dict(Lsd=200_000.0, BC_y=340.0, BC_z=360.0,
            tx=0.0, ty=2.0, tz=5.0)
SEED = dict(Lsd=TRUE["Lsd"] * 1.002, BC_y=TRUE["BC_y"] + 1.5,
            BC_z=TRUE["BC_z"] - 1.2,
            tx=0.0, ty=TRUE["ty"] + 0.05, tz=TRUE["tz"] - 0.05)

# A genuinely blind tz guess (no tilt information at all) -- a single direct
# autocalibrate_frozen_point call from here does NOT recover TRUE (confirmed
# below), so escaping it exercises iterate_frozen_point_until_stable's own
# re-seeding mechanism, not just the underlying one-shot fit.
BLIND_SEED = dict(Lsd=TRUE["Lsd"] * 1.002, BC_y=TRUE["BC_y"] + 1.5,
                   BC_z=TRUE["BC_z"] - 1.2, tx=0.0, ty=0.0, tz=0.0)

# Far enough outside the capture range that the iteration should honestly
# report non-convergence rather than land on a self-consistent wrong basin.
FAR_OFF_SEED = dict(Lsd=TRUE["Lsd"] * 1.002, BC_y=TRUE["BC_y"] + 1.5,
                     BC_z=TRUE["BC_z"] - 1.2, tx=0.0, ty=0.0, tz=-10.0)


def _point_pick_kwargs(image: np.ndarray) -> dict:
    return dict(downsample=2, footprint_px=5, snr_threshold=5.0,
                min_ring_gap_deg=0.3,
                panel_mask=np.ones(image.shape, dtype=bool),
                mask_erode_iter=0)


def _build_v1_params(**geom) -> CalibrationParams:
    return CalibrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=PX_UM, pxZ=PX_UM,
        Wavelength=WAVELENGTH_A,
        SpaceGroup=225, LatticeConstant=(5.4116, 5.4116, 5.4116, 90, 90, 90),
        MaxRingRad=float(min(NY, NZ)) / 2.0 - 5.0,
        RhoD=float(min(NY, NZ)) / 2.0 - 5.0,
        Refine={"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": False for i in range(15)}},
        **geom,
    )


def _paint_synthetic_image(v1_true: CalibrationParams) -> np.ndarray:
    from midas_integrate.geometry import build_tilt_matrix, invert_REta_to_pixel_batch

    rt = build_ring_table(v1_true)
    image = np.zeros((NZ, NY), dtype=np.float64)
    zero15 = {f"p{i}": 0.0 for i in range(15)}
    TRs = build_tilt_matrix(TRUE["tx"], TRUE["ty"], TRUE["tz"])
    rng = np.random.default_rng(1)
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ))

    spacing_px = 18.0
    for tt in sorted(set(rt.two_theta_deg.tolist())):
        R0 = (TRUE["Lsd"] / PX_UM) * np.tan(np.radians(tt))
        n_spots = max(8, int(round(2 * np.pi * R0 / spacing_px)))
        phase = rng.uniform(0, 360.0 / n_spots)
        eta = phase + np.arange(n_spots) * (360.0 / n_spots)
        R_targets = np.full_like(eta, R0)
        Y, Z = invert_REta_to_pixel_batch(
            R_targets, eta, Ycen=TRUE["BC_y"], Zcen=TRUE["BC_z"], TRs=TRs,
            Lsd=TRUE["Lsd"], RhoD=1.0, px=PX_UM, parallax=0.0, **zero15,
        )
        on_det = (Y >= 3) & (Y <= NY - 4) & (Z >= 3) & (Z <= NZ - 4)
        for y0, z0 in zip(Y[on_det], Z[on_det]):
            r2 = (yy - y0) ** 2 + (zz - z0) ** 2
            image += 500.0 * np.exp(-r2 / (2 * 1.2 ** 2))

    image += np.random.default_rng(2).normal(scale=3.0, size=image.shape)
    return np.clip(image, 0, None)


def test_autocalibrate_frozen_point_recovers_known_geometry():
    v1_true = _build_v1_params(**TRUE)
    image = _paint_synthetic_image(v1_true)

    v1_seed = _build_v1_params(**SEED)
    result = autocalibrate_frozen_point(
        v1_seed, image,
        snr_min=5.0,
        point_pick_kwargs=dict(downsample=2, footprint_px=5, snr_threshold=5.0,
                                min_ring_gap_deg=0.3,
                                panel_mask=np.ones(image.shape, dtype=bool),
                                mask_erode_iter=0),
        lm_verbose=False, verbose=False,
    )

    fit = result.history[0]
    lsd_err_pct = 100.0 * abs(fit.Lsd - TRUE["Lsd"]) / TRUE["Lsd"]
    bc_err_px = np.hypot(fit.BC_y - TRUE["BC_y"], fit.BC_z - TRUE["BC_z"])
    tz_err_deg = abs(fit.tz - TRUE["tz"])
    ty_err_deg = abs(fit.ty - TRUE["ty"])

    assert fit.n_fitted > 20
    assert lsd_err_pct < 1.0, f"Lsd error {lsd_err_pct:.4f}% too large"
    assert bc_err_px < 3.0, f"BC error {bc_err_px:.4f}px too large"
    assert tz_err_deg < 0.5, f"tz error {tz_err_deg:.4f}deg too large"
    assert ty_err_deg < 0.5, f"ty error {ty_err_deg:.4f}deg too large"


def test_autocalibrate_frozen_point_freezes_tx_regardless_of_caller_spec():
    from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params

    v1_true = _build_v1_params(**TRUE)
    image = _paint_synthetic_image(v1_true)

    v1_seed = _build_v1_params(**SEED)
    spec = spec_from_v1_params(v1_seed)
    if "tx" in spec.parameters:
        spec.parameters["tx"].refined = True  # caller opts in; pipeline must override

    result = autocalibrate_frozen_point(
        v1_seed, image, spec=spec, snr_min=5.0,
        point_pick_kwargs=dict(downsample=2, footprint_px=5, snr_threshold=5.0,
                                min_ring_gap_deg=0.3,
                                panel_mask=np.ones(image.shape, dtype=bool),
                                mask_erode_iter=0),
        lm_verbose=False, verbose=False,
    )
    assert result.spec.parameters["tx"].refined is False


def test_iterate_frozen_point_until_stable_escapes_blind_seed():
    v1_true = _build_v1_params(**TRUE)
    image = _paint_synthetic_image(v1_true)
    pp_kwargs = _point_pick_kwargs(image)

    # A single direct fit from BLIND_SEED must NOT already solve it --
    # otherwise this test would exercise autocalibrate_frozen_point alone,
    # not the iterative re-seeding mechanism.
    direct = autocalibrate_frozen_point(
        _build_v1_params(**BLIND_SEED), image, snr_min=5.0,
        point_pick_kwargs=pp_kwargs, verbose=False,
    )
    assert abs(direct.history[0].tz - TRUE["tz"]) > 0.5, (
        "a single direct fit from BLIND_SEED already recovered the true tz -- "
        "pick a harder BLIND_SEED so this test actually exercises iteration"
    )

    out = iterate_frozen_point_until_stable(
        _build_v1_params(**BLIND_SEED), image, snr_min=5.0,
        point_pick_kwargs=pp_kwargs,
        max_iter=20, bounds_bc_px=50.0, bounds_lsd_um=10_000.0,
        verbose=False,
    )

    assert out.converged, (
        f"expected the iterative wrapper to escape a blind tz=0 seed within "
        f"20 iterations; got converged=False after {out.n_iter}"
    )
    fit = out.fit
    lsd_err_pct = 100.0 * abs(fit.Lsd - TRUE["Lsd"]) / TRUE["Lsd"]
    bc_err_px = np.hypot(fit.BC_y - TRUE["BC_y"], fit.BC_z - TRUE["BC_z"])
    tz_err_deg = abs(fit.tz - TRUE["tz"])
    ty_err_deg = abs(fit.ty - TRUE["ty"])

    assert lsd_err_pct < 1.0, f"Lsd error {lsd_err_pct:.4f}% too large"
    assert bc_err_px < 3.0, f"BC error {bc_err_px:.4f}px too large"
    assert tz_err_deg < 0.5, f"tz error {tz_err_deg:.4f}deg too large"
    assert ty_err_deg < 0.5, f"ty error {ty_err_deg:.4f}deg too large"


def test_iterate_frozen_point_until_stable_reports_non_convergence_honestly():
    """A seed outside the capture range must come back converged=False, not
    a false-positive stable-but-wrong basin -- the entire point of the
    strict (parameter-stability-only) tolerance is to never claim success
    on a basin it hasn't actually reached."""
    v1_true = _build_v1_params(**TRUE)
    image = _paint_synthetic_image(v1_true)
    pp_kwargs = _point_pick_kwargs(image)

    out = iterate_frozen_point_until_stable(
        _build_v1_params(**FAR_OFF_SEED), image, snr_min=5.0,
        point_pick_kwargs=pp_kwargs,
        max_iter=15, bounds_bc_px=50.0, bounds_lsd_um=10_000.0,
        verbose=False,
    )

    assert not out.converged
    assert out.n_iter == 15
    assert len(out.history) == 15
    # Not converged AND not silently near the truth either.
    assert abs(out.fit.tz - TRUE["tz"]) > 1.0
