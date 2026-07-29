"""Tests for the modulation-tilt fit and the Sigma3 landing residual."""

import math

import numpy as np

from midas_defect.polytype import fit_modulation_tilt, sigma3_landing_residual


def test_fit_modulation_tilt_recovers_beta():
    g3 = 0.998
    beta_true = 3.0
    slope = 2.0 * g3 * math.tan(math.radians(beta_true))
    orders = np.array([1, 2, 4, 5], dtype=float)
    splits = slope * orders
    out = fit_modulation_tilt(orders, splits, g3)
    assert abs(out["beta_deg"] - beta_true) < 0.05
    assert out["r2"] > 0.999
    assert out["n_points"] == 4


def test_fit_modulation_tilt_uses_abs_order():
    g3 = 0.998
    slope = 0.10
    orders = np.array([-1, -2, 1, 2], dtype=float)
    splits = slope * np.abs(orders)
    out = fit_modulation_tilt(orders, splits, g3)
    assert abs(out["slope_inv_A_per_order"] - slope) < 1e-6


def test_fit_modulation_tilt_real_g1592_numbers():
    # session values: 0.105 at n=1, 0.20 at n=2 -> beta ~3 deg
    g3 = 0.998
    out = fit_modulation_tilt([1, 2], [0.105, 0.20], g3)
    assert 2.0 < out["beta_deg"] < 4.0


def test_sigma3_landing_maps_twin_pair():
    axis = np.array([1.0, 1.0, -1.0]) / math.sqrt(3)
    # build a transverse basis
    e2 = np.array([0.0, 0.0, 1.0]) - (np.array([0, 0, 1.0]) @ axis) * axis
    e2 /= np.linalg.norm(e2)
    qt = 0.998
    a = qt * axis + 0.10 * e2          # member A: + perp
    b = qt * axis - 0.10 * e2          # member B: - perp (180 about axis)
    out = sigma3_landing_residual(a, b, axis)
    assert out["is_twin_mapped"]
    assert out["residual_inv_A"] < 0.02
    assert out["improvement"] > 5.0


def test_sigma3_landing_rejects_same_side_pair():
    axis = np.array([1.0, 1.0, -1.0]) / math.sqrt(3)
    e2 = np.array([0.0, 0.0, 1.0]) - (np.array([0, 0, 1.0]) @ axis) * axis
    e2 /= np.linalg.norm(e2)
    qt = 0.998
    a = qt * axis + 0.10 * e2
    b = qt * axis + 0.06 * e2          # same side -> NOT a 180-deg twin pair
    out = sigma3_landing_residual(a, b, axis)
    assert not out["is_twin_mapped"]
