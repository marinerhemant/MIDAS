"""Tests for the periodic-vs-aperiodic fault-balance metric."""

import math

import numpy as np

from midas_defect.polytype import periodic_aperiodic_balance


def _axis():
    a = np.array([1.0, 1.0, -1.0]) / math.sqrt(3)
    return a


def test_pure_satellites_high_periodicity():
    rng = np.random.default_rng(0)
    axis = _axis()
    G = 2.993
    g3 = G / 3
    # sharp satellites only, at n = +-1,2,4,5 and fundamentals 3,6 -- no gap continuum
    qs, vals = [], []
    for n in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        qs.append(rng.normal(scale=0.02, size=(150, 3)) + n * g3 * axis)
        vals.append(np.full(150, 10.0))
    qs = np.concatenate(qs); vals = np.concatenate(vals)
    out = periodic_aperiodic_balance(qs, vals, axis, G)
    assert out["periodicity_fraction"] > 0.9
    assert out["I_relrod_gaps"] < 0.2 * out["I_satellites"]


def test_relrod_continuum_lowers_periodicity():
    rng = np.random.default_rng(1)
    axis = _axis()
    G = 2.993
    g3 = G / 3
    qs, vals = [], []
    # weak sharp satellites
    for n in [1, 2, 4, 5]:
        qs.append(rng.normal(scale=0.02, size=(60, 3)) + n * g3 * axis)
        vals.append(np.full(60, 10.0))
    # strong fundamentals
    for n in [3, 6]:
        qs.append(rng.normal(scale=0.02, size=(200, 3)) + n * g3 * axis)
        vals.append(np.full(200, 40.0))
    # continuous relrod spanning the whole rod (fills ALL half-integer gaps)
    t = rng.uniform(0.5, 6.0, size=3000)
    relrod = (t[:, None]) * g3 * axis + rng.normal(scale=0.02, size=(3000, 3))
    qs.append(relrod); vals.append(np.full(3000, 12.0))
    qs = np.concatenate(qs); vals = np.concatenate(vals)
    out = periodic_aperiodic_balance(qs, vals, axis, G)
    assert out["periodicity_fraction"] < 0.6     # relrod drags it down
    assert out["relrod_to_fundamental"] > 0.0
    assert math.isfinite(out["satellite_to_fundamental"])
