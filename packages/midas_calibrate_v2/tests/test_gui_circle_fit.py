"""Regression tests for the ring-picker fitting primitives.

These are pure-numpy and cover:

  * ``kasa_circle_fit``       — closed-form algebraic fit.
  * ``geometric_lm_refine``   — LM on the geometric residual.
  * ``joint_bc_lsd_fit``      — multi-ring (BC, Lsd) fit at known 2θ.

No Tk / matplotlib / midas-hkls is touched here; the GUI itself is left
to interactive smoke-testing.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate_v2.gui import (
    kasa_circle_fit,
    geometric_lm_refine,
    joint_bc_lsd_fit,
)


# ---------------------------------------------------------------------------
# kasa_circle_fit
# ---------------------------------------------------------------------------

def test_kasa_full_circle_exact():
    """Clean full circle → machine precision."""
    cx_t, cy_t, R_t = 100.0, 200.0, 300.0
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    xs = cx_t + R_t * np.cos(theta)
    ys = cy_t + R_t * np.sin(theta)
    cx, cy, R, rms = kasa_circle_fit(xs, ys)
    assert abs(cx - cx_t) < 1e-8
    assert abs(cy - cy_t) < 1e-8
    assert abs(R - R_t) < 1e-8
    assert rms < 1e-8


def test_kasa_partial_arc_recovers_centre():
    """60° arc with no noise → still recovers centre."""
    cx_t, cy_t, R_t = 50.0, -30.0, 150.0
    theta = np.linspace(-np.pi / 6, np.pi / 6, 8)
    xs = cx_t + R_t * np.cos(theta)
    ys = cy_t + R_t * np.sin(theta)
    cx, cy, R, rms = kasa_circle_fit(xs, ys)
    # On a perfect arc, even algebraic Kåsa is exact.
    assert abs(cx - cx_t) < 1e-6
    assert abs(cy - cy_t) < 1e-6
    assert abs(R - R_t) < 1e-6


def test_kasa_too_few_points_raises():
    with pytest.raises(ValueError):
        kasa_circle_fit([0.0, 1.0], [0.0, 1.0])


# ---------------------------------------------------------------------------
# geometric_lm_refine
# ---------------------------------------------------------------------------

def test_lm_refine_improves_kasa_on_noisy_arc():
    """LM should reduce the geometric residual below Kåsa on noisy arcs."""
    rng = np.random.default_rng(0)
    cx_t, cy_t, R_t = 100.0, 200.0, 300.0
    theta = np.linspace(-np.pi / 6, np.pi / 6, 12)
    xs = cx_t + R_t * np.cos(theta) + 0.5 * rng.standard_normal(theta.size)
    ys = cy_t + R_t * np.sin(theta) + 0.5 * rng.standard_normal(theta.size)
    cx_k, cy_k, R_k, rms_k = kasa_circle_fit(xs, ys)
    cx_l, cy_l, R_l, rms_l = geometric_lm_refine(xs, ys, cx_k, cy_k, R_k)
    # LM RMS should be ≤ Kåsa RMS by construction.
    assert rms_l <= rms_k + 1e-12
    # And the parameter recovery should be at least as good.
    err_k = np.hypot(cx_k - cx_t, cy_k - cy_t)
    err_l = np.hypot(cx_l - cx_t, cy_l - cy_t)
    assert err_l <= err_k + 1e-6


def test_lm_refine_clean_arc_machine_precision():
    """On a clean arc, LM converges to machine precision."""
    cx_t, cy_t, R_t = 10.0, 20.0, 100.0
    theta = np.linspace(-np.pi / 4, np.pi / 4, 8)
    xs = cx_t + R_t * np.cos(theta)
    ys = cy_t + R_t * np.sin(theta)
    cx, cy, R, rms = kasa_circle_fit(xs, ys)
    cx, cy, R, rms = geometric_lm_refine(xs, ys, cx, cy, R)
    assert abs(cx - cx_t) < 1e-8
    assert abs(cy - cy_t) < 1e-8
    assert abs(R - R_t) < 1e-8
    assert rms < 1e-8


# ---------------------------------------------------------------------------
# joint_bc_lsd_fit
# ---------------------------------------------------------------------------

def _synth_ring(cx, cy, R_px, arc_span_rad, n_pts, noise_px, rng):
    theta = np.linspace(-arc_span_rad / 2, arc_span_rad / 2, n_pts)
    xs = cx + R_px * np.cos(theta) + noise_px * rng.standard_normal(theta.size)
    ys = cy + R_px * np.sin(theta) + noise_px * rng.standard_normal(theta.size)
    return xs, ys


def test_joint_fit_recovers_truth_clean():
    """Three clean rings at known 2θ → exact (BC, Lsd) recovery."""
    cx_t, cy_t, lsd_t, px = 100.0, 200.0, 470_000.0, 150.0
    rng = np.random.default_rng(0)
    rings = []
    for tth_deg in (3.37, 3.89, 5.50):
        tth = np.radians(tth_deg)
        R_px = lsd_t / px * np.tan(tth)
        xs, ys = _synth_ring(cx_t, cy_t, R_px, np.pi, 30, 0.0, rng)
        rings.append((xs, ys, tth))
    res = joint_bc_lsd_fit(rings, pixel_size_um=px)
    assert abs(res["cx"] - cx_t) < 1e-6
    assert abs(res["cy"] - cy_t) < 1e-6
    assert abs(res["lsd_um"] - lsd_t) < 1e-3        # sub-µm on noiseless data
    assert res["rms_total_px"] < 1e-6
    assert res["n_total"] == 90
    assert res["n_rings"] == 3


def test_joint_fit_tightens_lsd_vs_single_ring():
    """Multi-ring joint fit drives σ(Lsd) tighter than a single-ring estimate."""
    cx_t, cy_t, lsd_t, px = 50.0, 75.0, 470_000.0, 150.0
    noise_px = 0.5
    rng = np.random.default_rng(1)
    tths_deg = [3.37, 3.89, 5.50, 6.45]

    # single-ring estimates (Kåsa per-ring → Lsd from R / tan(2θ))
    single_lsd = []
    rings = []
    for tth_d in tths_deg:
        tth = np.radians(tth_d)
        R_px = lsd_t / px * np.tan(tth)
        xs, ys = _synth_ring(cx_t, cy_t, R_px, 2 * np.pi / 3, 14, noise_px, rng)
        rings.append((xs, ys, tth))
        cx_k, cy_k, R_k, _ = kasa_circle_fit(xs, ys)
        single_lsd.append(R_k * px / np.tan(tth))
    single_spread = float(np.std(single_lsd))

    res = joint_bc_lsd_fit(rings, pixel_size_um=px)
    joint_err = abs(res["lsd_um"] - lsd_t)

    # The joint fit should land closer to truth than the typical
    # single-ring spread — that's the whole point of locking the centre.
    assert joint_err < single_spread, (
        f"joint Lsd error {joint_err:.2f} should be < single-ring spread "
        f"{single_spread:.2f}")
    # And to within ~0.1 % of truth on this noise level (0.5 px / pt).
    assert joint_err / lsd_t < 1e-3
    # Centre within 0.5 px on this noise level.
    assert np.hypot(res["cx"] - cx_t, res["cy"] - cy_t) < 0.5


def test_joint_fit_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        joint_bc_lsd_fit([], pixel_size_um=150.0)
    with pytest.raises(ValueError):
        joint_bc_lsd_fit([([0.0, 1.0], [0.0, 1.0], 0.1)], pixel_size_um=150.0)
    with pytest.raises(ValueError):
        joint_bc_lsd_fit([([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], -0.1)],
                         pixel_size_um=150.0)
