"""Tests for transverse_profile.py.

Each test is built so the failure it guards against would make it fail: a
wrong offset sign, a staircase bias at small width, a censored core biasing the
width, a feature measured differently under motion and censoring, a neighbour
spot absorbed into the width, or a noise-only window passing the gate.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest

from midas_defect.transverse_profile import (collect, excess, fit_transverse, inject_rocking,
                                             interp_reference, offsets, quadratic_term_pvalue,
                                             weighted_line_fit)

SHAPE = (7, 90, 90)
KC = 3
TANGENT = (math.cos(math.radians(23.0)), math.sin(math.radians(23.0)))   # oblique on purpose


def _blank(noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    stack = rng.normal(0.0, noise, SHAPE)
    raw = np.full(SHAPE, 500.0) + stack           # raw = a flat background plus the same noise
    return stack, raw, np.zeros(SHAPE[1:], bool), rng


def _measure(stack, raw, mask, r0, c0, W=2, censor=None):
    out, why = collect(stack, raw, mask, r0, c0, TANGENT, KC, W, censor=censor)
    assert out is not None, why
    return fit_transverse(out["t"], out["v"], n_censored=out["n_censored"]), out


def test_offsets_sign_and_axes():
    s, t = offsets([10.0, 12.0, 10.0], [20.0, 20.0, 23.0], 10.0, 20.0, (0.0, 1.0))
    assert s.tolist() == [0.0, 0.0, 3.0]          # tangent along +col
    assert t.tolist() == [0.0, -2.0, 0.0]         # perpendicular = (-1, 0): +row is -t


@pytest.mark.parametrize("sigma", [0.8, 1.3, 2.5])
def test_pixel_integrated_width_is_sigma2_plus_one_twelfth(sigma):
    stack, raw, mask, rng = _blank()
    r0, c0 = 45.37, 44.81
    inject_rocking(stack, raw, r0, c0, TANGENT, KC, sigma, 20000.0, 20.0, weights=(0, 0, 1, 0, 0),
                   motion_px_per_frame=0.0, rng=rng, poisson=False)
    fit, _ = _measure(stack, raw, mask, r0, c0)
    assert fit.ok, fit.reason
    assert fit.sigma ** 2 == pytest.approx(sigma ** 2 + 1.0 / 12.0, rel=0.05)


def test_subpixel_phase_does_not_move_the_width():
    widths = []
    for phase in np.linspace(0.0, 0.9, 6):
        stack, raw, mask, rng = _blank(seed=int(phase * 10))
        r0, c0 = 45.0 + phase, 44.0 + 0.37 * phase
        inject_rocking(stack, raw, r0, c0, TANGENT, KC, 0.9, 20000.0, 20.0, weights=(0, 0, 1, 0, 0),
                       motion_px_per_frame=0.0, rng=rng, poisson=False)
        fit, _ = _measure(stack, raw, mask, r0, c0)
        assert fit.ok, fit.reason
        widths.append(fit.sigma)
    assert (max(widths) - min(widths)) / np.mean(widths) < 0.03


def test_censored_core_leaves_the_width_unbiased():
    stack, raw, mask, rng = _blank()
    r0, c0 = 45.2, 44.6
    inject_rocking(stack, raw, r0, c0, TANGENT, KC, 1.3, 1.0e6, 20.0, weights=(0, 0, 1, 0, 0),
                   motion_px_per_frame=0.0, rng=rng, poisson=False)
    fit, out = _measure(stack, raw, mask, r0, c0, censor=119000.0)
    assert out["n_censored"] > 5                  # the core really was removed
    assert fit.ok, fit.reason
    assert fit.sigma ** 2 == pytest.approx(1.3 ** 2 + 1.0 / 12.0, rel=0.05)


def test_two_features_with_same_width_give_zero_excess_under_motion_and_censoring():
    """The paired-measurement design's core promise: identical transverse profiles, one a
    bright compact feature (censored), one a faint extended one (not censored), both rocking
    and moving across frames -> excess consistent with zero."""
    stack, raw, mask, rng = _blank()
    a = (30.3, 30.7)
    b = (60.6, 58.2)
    inject_rocking(stack, raw, *a, TANGENT, KC, 1.2, 2.0e6, 2.0, rng=rng)
    inject_rocking(stack, raw, *b, TANGENT, KC, 1.2, 8000.0, 20.0, rng=rng)
    f_a, out = _measure(stack, raw, mask, *a, censor=119000.0)
    f_b, _ = _measure(stack, raw, mask, *b, censor=119000.0)
    assert out["n_censored"] > 0
    assert f_a.ok and f_b.ok, (f_a.reason, f_b.reason)
    ex2, ex2_err, _ = excess(f_b.sigma, f_b.sigma_err, f_a.sigma ** 2, 2 * f_a.sigma * f_a.sigma_err)
    assert abs(ex2) < 3 * ex2_err + 0.15


def test_planted_excess_is_recovered():
    stack, raw, mask, rng = _blank()
    a = (30.3, 30.7)
    b = (60.6, 58.2)
    sig_ref, sig_x = 1.2, 1.5 / 2.354820045       # planted excess FWHM 1.5 px
    inject_rocking(stack, raw, *a, TANGENT, KC, sig_ref, 2.0e6, 2.0, rng=rng)
    inject_rocking(stack, raw, *b, TANGENT, KC, math.hypot(sig_ref, sig_x), 8000.0, 20.0, rng=rng)
    f_a, _ = _measure(stack, raw, mask, *a, censor=119000.0)
    f_b, _ = _measure(stack, raw, mask, *b, censor=119000.0)
    assert f_a.ok and f_b.ok
    _, _, fwhm = excess(f_b.sigma, f_b.sigma_err, f_a.sigma ** 2, 2 * f_a.sigma * f_a.sigma_err)
    assert fwhm == pytest.approx(1.5, abs=0.3)


def test_a_neighbour_spot_is_modelled_not_absorbed():
    stack, raw, mask, rng = _blank()
    r0, c0 = 45.1, 44.4
    inject_rocking(stack, raw, r0, c0, TANGENT, KC, 1.2, 20000.0, 20.0, weights=(0, 0, 1, 0, 0),
                   motion_px_per_frame=0.0, rng=rng, poisson=False)
    # neighbour 7 px away in t, half as bright
    (tr, tc), (pr, pc) = ((TANGENT[0], TANGENT[1]), (-TANGENT[1], TANGENT[0]))
    inject_rocking(stack, raw, r0 + 7 * pr, c0 + 7 * pc, TANGENT, KC, 1.2, 10000.0, 3.0,
                   weights=(0, 0, 1, 0, 0), motion_px_per_frame=0.0, rng=rng, poisson=False)
    fit, _ = _measure(stack, raw, mask, r0, c0)
    assert fit.ok, fit.reason
    assert fit.secondary
    assert fit.sigma ** 2 == pytest.approx(1.2 ** 2 + 1.0 / 12.0, rel=0.06)


def test_noise_only_fails_the_contrast_gate():
    stack, raw, mask, _ = _blank(seed=3)
    fits = [_measure(stack, raw, mask, 45.0 + d, 45.0 - d)[0] for d in (-10, 0, 10)]
    assert not any(f.ok for f in fits)


def test_frame_window_off_the_edge_is_refused():
    stack, raw, mask, _ = _blank()
    out, why = collect(stack, raw, mask, 45.0, 45.0, TANGENT, 1, 2)
    assert out is None and "edge" in why


X_RANGE = np.array([2.0e3, 1.8e4, 5.0e4, 9.5e4, 1.2e5, 1.7e5, 2.1e5, 2.6e5, 3.0e5])


def test_weighted_line_fit_recovers_intercept_and_slope_in_original_units():
    rng = np.random.default_rng(5)
    a, b = 0.8, 1.2e-5
    y_true = a + b * X_RANGE
    var = (0.03 * y_true) ** 2
    hits = 0
    for _ in range(200):
        y = y_true + rng.normal(0.0, np.sqrt(var))
        f = weighted_line_fit(X_RANGE, y, var, sys_rel=0.0)
        ea = math.sqrt(f["cov"][0, 0])
        hits += abs(f["params"][0] - a) <= 1.96 * ea
    assert hits / 200 >= 0.9                        # the intercept CI covers the truth
    f = weighted_line_fit(X_RANGE, y_true, var, sys_rel=0.0)
    assert f["params"][0] == pytest.approx(a, rel=1e-9)
    assert f["params"][1] == pytest.approx(b, rel=1e-9)


def test_quadratic_term_is_found_when_present_and_not_when_absent():
    rng = np.random.default_rng(6)
    var = np.full(X_RANGE.size, 0.02 ** 2)
    y_lin = 0.8 + 1.2e-5 * X_RANGE + rng.normal(0, 0.02, X_RANGE.size)
    y_quad = 0.8 + 1.2e-5 * X_RANGE + 2.0e-11 * X_RANGE ** 2 + rng.normal(0, 0.02, X_RANGE.size)
    assert quadratic_term_pvalue(X_RANGE, y_lin, var, sys_rel=0.0) >= 0.05
    assert quadratic_term_pvalue(X_RANGE, y_quad, var, sys_rel=0.0) < 0.05


def test_reference_interpolation_and_zero_excess():
    var, var_err, w = interp_reference(1.0, 0.05, 100.0, 2.0, 0.05, 200.0, 150.0)
    assert w == 0.5
    assert var == pytest.approx(0.5 * 1.0 + 0.5 * 4.0)
    ex2, _, fwhm = excess(math.sqrt(var), 0.05, var, var_err)
    # sqrt(var)**2 - var is ~4e-16, not 0, in float64; that is a 5e-8 px FWHM
    assert ex2 == pytest.approx(0.0, abs=1e-12) and fwhm < 1e-6
