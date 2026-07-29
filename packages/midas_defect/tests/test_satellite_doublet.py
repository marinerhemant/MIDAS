"""Tests for the discrete-doublet resolver + Ewald-artifact discriminator.

Three regimes:
  - two compact reflections at DISTINCT omega, straddling the axis (twin polarity)
    => verdict "two-reflections", azimuth ~180, is_twin_polarity True.
  - one broad omega-continuous relrod => NOT "two-reflections" (artifact/ambiguous).
  - honesty: the result exposes no parent/twin (host/lamella) identity.
"""

import numpy as np

from midas_defect.polytype import resolve_satellite_doublet, SatelliteDoublet


def _frame(axis):
    axis = axis / np.linalg.norm(axis)
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e2 = seed - (seed @ axis) * axis
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(axis, e2)
    return axis, e2, e3


def _blob(center, omega0, n, rng, q_scale=0.02, om_scale=0.6):
    qs = rng.normal(scale=q_scale, size=(n, 3)) + center
    om = rng.normal(scale=om_scale, size=n) + omega0
    vals = np.full(n, 10.0)
    return qs, vals, om


def test_two_compact_reflections_at_distinct_omega():
    rng = np.random.default_rng(0)
    axis, e2, e3 = _frame(np.array([1.0, 1.0, -1.0]))
    G = 2.993
    qt = G / 3.0
    # two members straddling the axis along +/- e2 (180 deg apart), distinct omega
    perp = 0.10
    cen_a = axis * qt + perp * e2
    cen_b = axis * qt - perp * e2
    qa, va, oa = _blob(cen_a, omega0=15.0, n=400, rng=rng)
    qb, vb, ob = _blob(cen_b, omega0=22.0, n=300, rng=rng)
    # background near the rung at a third omega
    qbg, vbg, obg = _blob(axis * qt, omega0=40.0, n=80, rng=rng, q_scale=0.18)
    vbg[:] = 0.5
    qs = np.concatenate([qa, qb, qbg]); vals = np.concatenate([va, vb, vbg])
    om = np.concatenate([oa, ob, obg])

    res = resolve_satellite_doublet(qs, vals, om, axis, qt)
    assert isinstance(res, SatelliteDoublet)
    assert res.n_members == 2
    assert res.verdict == "two-reflections"
    assert abs(res.azimuth_deg - 180.0) < 25.0
    assert res.is_twin_polarity
    # omega centers recovered and distinct
    oc = sorted(m["omega_center"] for m in res.members)
    assert abs(oc[0] - 15.0) < 1.5 and abs(oc[1] - 22.0) < 1.5
    assert res.metadata["omega_sep_deg"] > 5.0


def test_single_relrod_broad_omega_is_not_two_reflections():
    rng = np.random.default_rng(1)
    axis, e2, e3 = _frame(np.array([1.0, 1.0, -1.0]))
    G = 2.993
    qt = G / 3.0
    n = 800
    # one relrod-like cloud on the axis, intensity spread CONTINUOUSLY over omega
    qs = rng.normal(scale=0.03, size=(n, 3)) + axis * qt
    om = rng.uniform(10.0, 30.0, size=n)        # broad, continuous in omega
    vals = np.full(n, 10.0)
    res = resolve_satellite_doublet(qs, vals, om, axis, qt)
    assert res.verdict != "two-reflections"


def test_no_parent_twin_identity_leaked():
    rng = np.random.default_rng(2)
    axis, e2, e3 = _frame(np.array([1.0, 1.0, -1.0]))
    qt = 2.993 / 3.0
    qa, va, oa = _blob(axis * qt + 0.10 * e2, 15.0, 300, rng)
    qb, vb, ob = _blob(axis * qt - 0.10 * e2, 22.0, 300, rng)
    qs = np.concatenate([qa, qb]); vals = np.concatenate([va, vb]); om = np.concatenate([oa, ob])
    res = resolve_satellite_doublet(qs, vals, om, axis, qt)
    # members are labeled a/b by transverse position; never given a host/lamella id
    for m in res.members:
        for forbidden in ("parent", "twin", "lamella", "host", "matrix"):
            assert forbidden not in m
    # and the result must carry the explicit attribution disclaimer
    assert "not parent/twin" in res.metadata["note"].lower()


def test_member_ordering_and_perp_offset():
    rng = np.random.default_rng(3)
    axis, e2, e3 = _frame(np.array([0.0, 0.0, 1.0]))
    qt = 1.0
    qa, va, oa = _blob(axis * qt + 0.06 * e2, 12.0, 300, rng)   # less off-axis
    qb, vb, ob = _blob(axis * qt - 0.14 * e2, 20.0, 300, rng)   # more off-axis
    qs = np.concatenate([qa, qb]); vals = np.concatenate([va, vb]); om = np.concatenate([oa, ob])
    res = resolve_satellite_doublet(qs, vals, om, axis, qt)
    assert res.n_members == 2
    offs = sorted(m["perp_offset"] for m in res.members)
    assert offs[0] < offs[1]                  # asymmetric offsets resolved
    assert abs(offs[1] - 0.14) < 0.03
