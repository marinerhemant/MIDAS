"""Cell tracking through a series, and the transition gate.

The gate's whole value is that it refuses to answer when it cannot see. So the
tests are: it fires on a planted transition, stays quiet on a planted drift, and
says INDETERMINATE when the reflection count is too low for either.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.cell_series import (cell_deformation, explained_fraction,
                                    track_cell, detect_transition,
                                    min_detectable_excess, poisson_upper_p)

CELL = (5.2739, 5.2384, 20.5, 90.0, 90.0, 90.0)
RNG = np.random.default_rng(20260902)


def _B(cell):
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T


def _hkls(n=40, hmax=3):
    out = set()
    r = np.random.default_rng(4)
    while len(out) < n:
        t = tuple(int(v) for v in r.integers(-hmax, hmax + 1, 3))
        if t != (0, 0, 0):
            out.add(t)
    return np.array(sorted(out), float)


def _data(cell, hkl, noise, seed=0):
    return hkl @ _B(cell).T + np.random.default_rng(seed).normal(0, noise, (len(hkl), 3))


def _with_background(g, frac, seed):
    """Add unassignable spots -- every real pattern has some."""
    r = np.random.default_rng(seed + 999)
    k = max(1, int(round(frac * len(g))))
    span = np.abs(g).max()
    return np.vstack([g, r.uniform(-span, span, (k, 3))])


# ------------------------------------------------------------- deformation

def test_deformation_of_a_pure_compression():
    d = cell_deformation(CELL, tuple(0.99 * v if i < 3 else v
                                     for i, v in enumerate(CELL)))
    assert d.volume_ratio == pytest.approx(0.99 ** 3, rel=1e-6)
    assert d.is_near_hydrostatic
    assert all(e < 0 for e in d.principal_strains)


def test_deformation_flags_a_non_hydrostatic_step():
    changed = (CELL[0] * 0.97, CELL[1] * 1.00, CELL[2] * 1.00, 90., 90., 90.)
    d = cell_deformation(CELL, changed)
    assert not d.is_near_hydrostatic
    assert "NON-HYDROSTATIC" in str(d)


def test_identity_deformation_is_identity():
    d = cell_deformation(CELL, CELL)
    assert np.allclose(d.F, np.eye(3), atol=1e-12)
    assert d.volume_ratio == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ tracking

def test_tracking_absorbs_a_drift_and_reports_the_strain():
    hkl = _hkls(40)
    drifted = tuple(0.995 * v if i < 3 else v for i, v in enumerate(CELL))
    g = _data(drifted, hkl, 2e-4)
    fit, defo, frac = track_cell(CELL, hkl, g, sigma_g=2e-4)
    assert frac > 0.95
    assert np.allclose(fit.cell[:3], drifted[:3], rtol=2e-3)
    assert defo.volume_ratio == pytest.approx(0.995 ** 3, rel=5e-3)


def test_tracking_is_free_triclinic_not_locked_to_the_old_symmetry():
    """A symmetry-lowering step must be visible, not absorbed into a=b."""
    hkl = _hkls(60)
    lowered = (5.30, 5.18, 20.5, 90., 90., 89.4)      # gamma shear appears
    g = _data(lowered, hkl, 1e-4)
    fit, defo, frac = track_cell(CELL, hkl, g, sigma_g=1e-4)
    assert fit.cell[5] == pytest.approx(89.4, abs=0.05)
    assert abs(fit.cell[0] - fit.cell[1]) > 0.05


# ---------------------------------------------------------------- the gate

def _background(g, k, seed):
    """k unassignable spots at random positions -- every pattern has some."""
    r = np.random.default_rng(seed)
    span = np.abs(g).max()
    return np.vstack([g, r.uniform(-span, span, (k, 3))])


def _second_phase(g, hkl, cell, k, seed, noise=2e-4):
    """k spots from a DIFFERENT lattice: reshaped and reoriented."""
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi); c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])
    B2 = R @ np.linalg.inv(np.diag([1.13, 0.91, 1.06])) @ _B(cell)
    j = r.integers(0, len(hkl), k)
    return np.vstack([g, hkl[j] @ B2.T + r.normal(0, noise, (k, 3))])


# ------------------------------------------------------------ poisson maths

def test_poisson_tail_and_mde_are_correct():
    from scipy import stats
    assert poisson_upper_p(10, 2.0) == pytest.approx(stats.poisson.sf(9, 2.0))
    assert poisson_upper_p(0, 5.0) == pytest.approx(1.0)
    assert poisson_upper_p(1, 0.0) == 0.0


def test_mde_grows_with_the_background_but_sublinearly():
    """More background is harder, but as sqrt -- that is the Poisson scaling."""
    m = [min_detectable_excess(l) for l in (2.0, 20.0, 200.0)]
    assert m[0] < m[1] < m[2]
    assert m[2] < 10 * m[0], f"mde grew faster than sqrt: {m}"


def test_mde_relaxes_with_a_looser_alpha():
    assert min_detectable_excess(20.0, alpha=0.05) < min_detectable_excess(20.0, alpha=1e-4)


# ---------------------------------------------------------------- the gate

def test_gate_stays_quiet_when_the_background_matches_the_baseline():
    hkl = _hkls(60)
    drifted = tuple(0.99 * v if i < 3 else v for i, v in enumerate(CELL))
    g = _data(drifted, hkl, 2e-4, seed=11)
    g_all = _background(g, 6, seed=11)                 # 6 strays on 60 assigned
    v = detect_transition(CELL, hkl, g, g_all, baseline_rate=0.10, sigma_g=2e-4)
    assert v.has_power, str(v)
    assert v.verdict == "CONTINUOUS", str(v)
    assert v.n_expected == pytest.approx(6.0)


def test_gate_FIRES_when_a_second_phase_ADDS_spots():
    """The framing the earlier versions got wrong: a new phase adds spots."""
    hkl = _hkls(60)
    g = _data(CELL, hkl, 2e-4, seed=12)
    g_all = _second_phase(_background(g, 6, 12), hkl, CELL, 30, seed=12)
    v = detect_transition(CELL, hkl, g, g_all, baseline_rate=0.10, sigma_g=2e-4)
    assert v.has_power, str(v)
    assert v.verdict == "TRANSITION", str(v)
    assert v.n_unexplained > v.n_expected + v.min_detectable_excess - 1
    assert v.p_value < 0.01


def test_gate_says_INDETERMINATE_when_the_excess_it_needs_exceeds_what_is_expected():
    hkl = _hkls(60)
    g = _data(CELL, hkl, 2e-4, seed=13)
    g_all = _background(g, 6, 13)
    v = detect_transition(CELL, hkl, g, g_all, baseline_rate=0.10, sigma_g=2e-4,
                          expected_excess=2)
    assert v.verdict == "INDETERMINATE", str(v)
    assert not v.has_power and v.min_detectable_excess > 2
    assert "NO POWER" in str(v)


def test_a_high_background_rate_costs_power():
    hkl = _hkls(60)
    g = _data(CELL, hkl, 2e-4, seed=14)
    quiet = detect_transition(CELL, hkl, g, _background(g, 6, 14),
                              baseline_rate=0.10, sigma_g=2e-4)
    noisy = detect_transition(CELL, hkl, g, _background(g, 60, 14),
                              baseline_rate=1.00, sigma_g=2e-4)
    assert noisy.min_detectable_excess > quiet.min_detectable_excess


def test_a_PURE_HOMOGENEOUS_STRAIN_still_passes_and_the_docs_say_so():
    """The pinned limit: any deformation IS a triclinic cell."""
    hkl = _hkls(60)
    squashed = (4.60, 5.95, 18.9, 90., 90., 90.)      # ~13 % anisotropic
    g = _data(squashed, hkl, 2e-4, seed=20)
    v = detect_transition(CELL, hkl, g, _background(g, 6, 20),
                          baseline_rate=0.10, sigma_g=2e-4)
    assert v.verdict == "CONTINUOUS", str(v)
    assert not v.deformation.is_near_hydrostatic
    assert any("non-hydrostatic" in n.lower() for n in v.notes)
    import midas_hkls.cell_series as cs
    assert "invisible to this test" in cs.__doc__


def test_gate_needs_the_UNASSIGNED_spots():
    hkl = _hkls(60)
    g = _data(CELL, hkl, 2e-4, seed=15)
    g_all = _second_phase(_background(g, 6, 15), hkl, CELL, 30, seed=15)
    full = detect_transition(CELL, hkl, g, g_all, baseline_rate=0.10, sigma_g=2e-4)
    blind = detect_transition(CELL, hkl, g, g, baseline_rate=0.10, sigma_g=2e-4)
    assert full.verdict == "TRANSITION"
    assert blind.verdict == "CONTINUOUS", "dropping them hides the whole signal"
    with pytest.raises(ValueError, match="at least the assigned"):
        detect_transition(CELL, hkl, g, g[:10], baseline_rate=0.1, sigma_g=2e-4)


def test_the_framing_error_is_recorded_so_it_is_not_repeated():
    import midas_hkls.cell_series as cs
    src = pathlib.Path(cs.__file__).read_text()
    assert "ADDS spots" in src
    assert "no scatter" in src or "has no scatter" in src


def test_baseline_rate_is_validated():
    hkl = _hkls(20); g = _data(CELL, hkl, 2e-4, seed=16)
    with pytest.raises(ValueError, match="baseline_rate"):
        detect_transition(CELL, hkl, g, g, baseline_rate=-1.0, sigma_g=2e-4)
