"""Adversarial correctness tests for the geometry-honest 2026 replacements.

These deliberately construct cases where the DEPRECATED methods produced artifacts,
and assert the new ones behave correctly (and that the old ones now refuse). This is
the "test correctness, not execution" gap from AUDIT_2026-06-23.md.
"""

import math

import numpy as np
import pytest

import midas_stress.orientation as ori
from midas_defect.attribution import (AttributionError, coincident_axes,
                                       assert_variant_attributable)
from midas_defect.polytype.satellite_excess import satellite_radial_excess
from midas_defect.polytype.aggregate_thickness import (aggregate_lamella_thickness,
                                                       find_satellite_axis)
from midas_defect.polytype.satellite_intensity import polytype_satellite_enhancement
from midas_defect.polytype.lamella_thickness import per_grain_lamella_thickness
from midas_defect.asterism.local_decomposition import per_grain_asterism_local

A = 3.6356
G = 2 * math.pi * math.sqrt(3) / A           # |G(111)|
G3, G23 = G / 3, 2 * G / 3


# --------------------------------------------------------------------------- #
# P0-G: attribution guard
# --------------------------------------------------------------------------- #
def _sigma3_pair():
    P = np.eye(3)
    S = np.asarray(ori.axis_angle_to_orient_mat(np.array([1., 1, 1]) / math.sqrt(3), 60.0))
    return P, P @ S


def test_guard_refuses_shared_plane_feature():
    P, T = _sigma3_pair()
    shared = coincident_axes([P, T], hkl=(1, 1, 1), tol_deg=5.0)
    assert shared.shape[0] >= 1                      # the twin plane is shared
    # the shared <111> (= the twin axis [111] in sample frame) must be refused
    n_sat = (P @ (np.array([1., 1, 1]) / math.sqrt(3)))
    with pytest.raises(AttributionError):
        assert_variant_attributable(n_sat, [P, T], hkl=(1, 1, 1), tol_deg=5.0)


def test_guard_allows_non_shared_direction():
    P, T = _sigma3_pair()
    # a generic direction far from any shared <111> must NOT raise
    d = np.array([1.0, 0.2, 0.1]); d /= np.linalg.norm(d)
    assert_variant_attributable(d, [P, T], hkl=(1, 1, 1), tol_deg=5.0)


# --------------------------------------------------------------------------- #
# P0-1: texture-safe satellite excess + discrete-vs-relrod
# --------------------------------------------------------------------------- #
def _bg(n, rng, q_lo=0.2, q_hi=None):
    q_hi = q_hi or (G + 0.2)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    qm = rng.uniform(q_lo, q_hi, n)
    return d * qm[:, None], np.ones(n)


def test_excess_discrete_9R():
    # dense isotropic bg so the narrow on-axis cone is populated at every position
    rng = np.random.default_rng(0)
    qb, vb = _bg(400000, rng)
    # tight satellite clusters at G/3 and 2G/3 along +z (cone ~2 deg), bright
    def cluster(qmag, n):
        z = np.ones(n); xy = rng.normal(scale=math.radians(2.0), size=(n, 2))
        d = np.stack([xy[:, 0], xy[:, 1], z], 1); d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d * qmag, np.full(n, 50.0)
    q1, v1 = cluster(G3, 6000); q2, v2 = cluster(G23, 6000)
    qs = np.concatenate([qb, q1, q2]); vals = np.concatenate([vb, v1, v2])
    res = satellite_radial_excess(qs, vals, np.array([[0, 0, 1.0]]), G,
                                  tube_deg=8.0, off_deg=20.0, dq=0.04)
    p = res["at_positions"]
    assert res["verdict"] == "9R-periodic"
    assert p["G/3*"] > 5 * p["G/2"] and p["2G/3*"] > 5 * p["G/2"]   # peaks >> controls


def test_excess_relrod_not_called_9R():
    rng = np.random.default_rng(1)
    qb, vb = _bg(400000, rng)
    # continuous rod along +z, intensity rising toward the (111) Bragg
    n = 60000; t = rng.uniform(0.3, G, n)
    xy = rng.normal(scale=math.radians(2.0), size=(n, 2))
    d = np.stack([xy[:, 0], xy[:, 1], np.ones(n)], 1); d /= np.linalg.norm(d, axis=1, keepdims=True)
    qr = d * t[:, None]; vr = 5.0 * (t / G)         # rises inward toward Bragg
    qs = np.concatenate([qb, qr]); vals = np.concatenate([vb, vr])
    res = satellite_radial_excess(qs, vals, np.array([[0, 0, 1.0]]), G,
                                  tube_deg=8.0, off_deg=20.0, dq=0.04)
    assert res["verdict"] != "9R-periodic"          # must NOT mislabel a rod as 9R


def test_excess_isotropic_no_inflation():
    # the test the OLD enhancement metric failed: pure isotropic -> excess ~1, not huge
    rng = np.random.default_rng(2)
    qs, vals = _bg(500000, rng)
    res = satellite_radial_excess(qs, vals, np.array([[0, 0, 1.0]]), G,
                                  tube_deg=8.0, off_deg=20.0, dq=0.04)
    p = res["at_positions"]
    assert abs(p["G/3*"] - 1.0) < 0.4 and abs(p["2G/3*"] - 1.0) < 0.4
    assert res["verdict"] != "9R-periodic"


# --------------------------------------------------------------------------- #
# P0-2: per-spot-local asterism strain vs rotation
# --------------------------------------------------------------------------- #
def test_asterism_radial_vs_tangential():
    rng = np.random.default_rng(3)
    qB = np.array([0.0, 0.0, 3.0]); n = 5000
    # pure RADIAL broadening: offsets along qB_hat (=z)
    off_r = rng.normal(scale=0.03, size=n)
    qs_r = qB[None, :] + np.stack([np.zeros(n), np.zeros(n), off_r], 1)
    # pure TANGENTIAL broadening: offsets perpendicular (x)
    off_t = rng.normal(scale=0.03, size=n)
    qs_t = qB[None, :] + np.stack([off_t, np.zeros(n), np.zeros(n)], 1)
    for qs, expect in [(qs_r, "radial"), (qs_t, "tangential")]:
        out = per_grain_asterism_local(
            qs, np.ones(n), np.zeros(n, int), np.tile(qB, (n, 1)),
            np.ones(n, bool), 1, min_voxels_per_grain=10)
        sr, st = out["sigma_r"][0], out["sigma_t"][0]
        if expect == "radial":
            assert sr > 5 * st
        else:
            assert st > 5 * sr


# --------------------------------------------------------------------------- #
# P0-3: aggregate L recovers a known FWHM; find_satellite_axis finds the axis
# --------------------------------------------------------------------------- #
def test_aggregate_L_recovers_known_fwhm():
    rng = np.random.default_rng(4)
    sigma = 0.05; n = 20000
    proj = G3 + rng.normal(scale=sigma, size=n)
    perp = rng.uniform(0, 0.03, n); ang = rng.uniform(0, 2 * math.pi, n)
    qs = np.stack([perp * np.cos(ang), perp * np.sin(ang), proj], 1)
    res = aggregate_lamella_thickness(qs, np.ones(n), np.array([0, 0, 1.0]), G3)
    fwhm_true = 2.3548 * sigma
    assert abs(res["fwhm"] - fwhm_true) / fwhm_true < 0.15
    assert abs(res["L_angstrom"] - 0.94 * 2 * math.pi / fwhm_true) / res["L_angstrom"] < 0.15


def test_find_satellite_axis():
    rng = np.random.default_rng(5)
    qb, vb = _bg(20000, rng)
    n = 3000; z = np.ones(n); xy = rng.normal(scale=math.radians(1.0), size=(n, 2))
    d = np.stack([xy[:, 0], xy[:, 1], z], 1); d /= np.linalg.norm(d, axis=1, keepdims=True)
    qs = np.concatenate([qb, d * G3]); vals = np.concatenate([vb, np.full(n, 50.0)])
    cands = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    ax = find_satellite_axis(qs, vals, cands, G3)
    assert abs(abs(ax[2]) - 1.0) < 1e-6           # picks +z


# --------------------------------------------------------------------------- #
# deprecation: the artifact generators now refuse by default
# --------------------------------------------------------------------------- #
def test_deprecated_enhancement_raises():
    with pytest.raises(RuntimeError):
        polytype_satellite_enhancement(np.zeros((10, 3)), np.ones(10),
                                       np.array([0, 0, 1.0]), G)


def test_deprecated_lamella_raises():
    with pytest.raises(RuntimeError):
        per_grain_lamella_thickness(np.zeros((10, 3)), np.ones(10),
                                    np.eye(3)[None], np.array([1, 1, 1.0]), G3,
                                    np.zeros(10, int), 1)
