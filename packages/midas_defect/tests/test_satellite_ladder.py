"""Tests for the n*G/3 ladder builder + fundamental-vs-satellite decontamination."""

import math

import numpy as np

from midas_defect.lattice import fcc_cu_crystal
from midas_defect.polytype import (
    build_satellite_ladder,
    decontaminate_ladder,
)


def _plant_ladder(axis, g3, rng, n_max=6, n_per=200):
    axis = axis / np.linalg.norm(axis)
    qs, vals = [], []
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        center = n * g3 * axis
        qs.append(rng.normal(scale=0.02, size=(n_per, 3)) + center)
        vals.append(np.full(n_per, 10.0))
    return np.concatenate(qs), np.concatenate(vals)


def test_build_recovers_all_rungs():
    rng = np.random.default_rng(0)
    cr = fcc_cu_crystal()
    a = float(cr.lattice.a)
    B0 = 2 * math.pi / a
    G = math.sqrt(3) * B0
    g3 = G / 3
    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    qs, vals = _plant_ladder(axis, g3, rng)
    lad = build_satellite_ladder(qs, vals, axis, G)
    ns = sorted(r["n"] for r in lad.rungs)
    assert ns == [n for n in range(-6, 7) if n != 0]
    # q_obs tracks n*g3
    for r in lad.rungs:
        assert abs(abs(r["q_obs"]) - abs(r["n"]) * g3) < 0.02


def test_decontaminate_marks_fundamentals_and_satellites():
    rng = np.random.default_rng(1)
    cr = fcc_cu_crystal()
    a = float(cr.lattice.a)
    B0 = 2 * math.pi / a
    G = math.sqrt(3) * B0
    g3 = G / 3
    # carrier orientation = identity -> its <111> axis is [1,1,1]/sqrt3
    OM = np.eye(3)
    axis = OM @ (np.array([1.0, 1.0, 1.0]) / math.sqrt(3))
    qs, vals = _plant_ladder(axis, g3, rng)
    lad = build_satellite_ladder(qs, vals, axis, G)
    lad = decontaminate_ladder(lad, OM, cr, tol_inv_A=0.15)
    cls = {r["n"]: r["classification"] for r in lad.rungs}
    # n = +-3 (111) and +-6 (222) are the carrier's own fundamentals on the rod
    for n in (3, -3, 6, -6):
        assert cls[n] == "fundamental", (n, lad.rungs)
    # n = +-1,2,4,5 are forbidden-gap genuine 9R satellites
    for n in (1, -1, 2, -2, 4, -4, 5, -5):
        assert cls[n] == "9R-satellite", (n, lad.rungs)
    assert lad.metadata["n_satellites"] == 8
    assert lad.metadata["n_fundamentals"] == 4


def test_5G3_not_220_via_3d_distance():
    """A second grain's 220 that is radially near 5G/3 (|q|~4.9) but points in a
    different 3-D direction must NOT contaminate the 5G/3 rung."""
    rng = np.random.default_rng(2)
    cr = fcc_cu_crystal()
    a = float(cr.lattice.a)
    B0 = 2 * math.pi / a
    G = math.sqrt(3) * B0
    g3 = G / 3
    OM = np.eye(3)
    axis = OM @ (np.array([1.0, 1.0, 1.0]) / math.sqrt(3))
    qs, vals = _plant_ladder(axis, g3, rng)
    lad = build_satellite_ladder(qs, vals, axis, G)
    # craft a second grain whose [2,2,0] points ~30 deg off `axis`
    g220 = B0 * np.array([2.0, 2.0, 0.0])           # |q| = 2 sqrt2 B0 = 4.888
    target = 4.888 * (math.cos(math.radians(30)) * axis
                      + math.sin(math.radians(30)) * np.array([0, 0, 1.0]))
    # rotation sending g220 onto target
    u = g220 / np.linalg.norm(g220)
    t = target / np.linalg.norm(target)
    v = np.cross(u, t); s = np.linalg.norm(v); c = u @ t
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R2 = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    lad = decontaminate_ladder(lad, np.stack([OM, R2]), cr, tol_inv_A=0.15)
    cls = {r["n"]: r["classification"] for r in lad.rungs}
    assert cls[5] == "9R-satellite"     # 220 is radially close but 3-D far
    assert cls[-5] == "9R-satellite"
