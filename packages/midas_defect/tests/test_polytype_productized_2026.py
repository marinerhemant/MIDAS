"""Tests for the productized demk-9R analyses (2026-06-30):
finite-stack simulator, sample-wide doublet survey, modulation-type classifier,
and the Ewald two-crossing predictor.
"""

import math

import numpy as np
import pytest

from midas_defect.polytype import (
    build_close_packed_slab,
    classify_modulation,
    g_111,
    on_axis_ladder,
    slab_intensity,
    structure_factor_intensity,
    survey_doublets,
)

A_FCC = 3.6356


# --------------------------------------------------------------------------- #
# b1: 3-D finite-stack simulator
# --------------------------------------------------------------------------- #

def test_ideal_9r_on_axis_satellites_are_extinct():
    """An ideal finite 9R slab: n*G/3 satellites (n%3!=0) are numerically zero,
    fundamentals 111 (n=3) and 222 (n=6) are ~1."""
    lad = on_axis_ladder(
        [1, 2, 3, 4, 5, 6], n_inplane=6, n_layers=18, a_fcc=A_FCC,
    )
    assert lad[3.0] == pytest.approx(1.0, abs=1e-6)   # 111
    assert lad[6.0] == pytest.approx(1.0, abs=1e-6)   # 222
    for n in (1.0, 2.0, 4.0, 5.0):
        assert lad[n] < 1e-12                          # satellites extinct


def test_finite_size_does_not_create_satellites():
    """Changing slab thickness leaves the ideal on-axis satellites extinct
    (finite size adds Laue fringes at fundamentals, never satellites)."""
    for n_layers in (9, 18, 36):
        lad = on_axis_ladder([1, 2, 4], n_inplane=4, n_layers=n_layers, a_fcc=A_FCC)
        assert max(lad.values()) < 1e-10


def test_spacing_modulation_turns_on_satellites_as_amplitude_squared():
    """A period-3 spacing modulation turns the on-axis satellite on with
    intensity ~ amplitude^2 (second order)."""
    i_small = on_axis_ladder([1], n_inplane=4, n_layers=18, a_fcc=A_FCC,
                             spacing_modulation=0.02)[1.0]
    i_big = on_axis_ladder([1], n_inplane=4, n_layers=18, a_fcc=A_FCC,
                           spacing_modulation=0.04)[1.0]
    assert i_small > 1e-6
    # doubling amplitude -> ~4x intensity
    assert i_big / i_small == pytest.approx(4.0, rel=0.10)


def test_composition_modulation_turns_on_satellites_flat_in_order():
    """A period-3 composition modulation turns satellites on ~flat in order
    (contrast with the rising displacement signature)."""
    lad = on_axis_ladder([1, 2, 4, 5], n_inplane=4, n_layers=18, a_fcc=A_FCC,
                         comp_modulation=0.05)
    vals = np.array([lad[n] for n in (1.0, 2.0, 4.0, 5.0)])
    assert (vals > 1e-6).all()
    # flat: spread across orders is small relative to the mean
    assert vals.std() / vals.mean() < 0.2


def test_finite_stack_agrees_with_infinite_structure_factor_on_extinction():
    """The finite-stack on-axis result agrees with the infinite-cell structure
    factor on WHICH orders are extinct."""
    for n in (1, 2, 4, 5):
        f_sf = structure_factor_intensity((0, 0, 3 * n))          # infinite cell
        i_slab = on_axis_ladder([n], n_inplane=4, n_layers=9, a_fcc=A_FCC)[float(n)]
        assert f_sf == pytest.approx(0.0, abs=1e-9)
        assert i_slab < 1e-9
    # fundamental l=9 is NOT extinct in either
    assert structure_factor_intensity((0, 0, 9)) == pytest.approx(81.0, abs=1e-6)


def test_slab_intensity_batch_and_single_agree():
    stack = build_close_packed_slab(n_inplane=4, n_layers=9, a_fcc=A_FCC)
    G = g_111(A_FCC)
    q_single = stack.axis * G          # 111
    batch = slab_intensity(stack, np.array([q_single, stack.axis * (G / 3)]))
    assert slab_intensity(stack, q_single) == pytest.approx(batch[0])


# --------------------------------------------------------------------------- #
# b2: Ewald two-crossing predictor
# --------------------------------------------------------------------------- #

def test_ewald_two_crossings_satisfy_the_elastic_condition():
    from midas_defect.geometry import ewald_crossing_omegas

    lam = 0.172979
    k0 = 2.0 * math.pi / lam
    q_s = np.array([3.0, 0.5, 1.0])
    qmag = np.linalg.norm(q_s)
    om = ewald_crossing_omegas(q_s, lam)
    assert om.size == 2
    for w in om:
        c, s = math.cos(w), math.sin(w)
        q_lab = np.array([c * q_s[0] - s * q_s[1], s * q_s[0] + c * q_s[1], q_s[2]])
        # diffraction condition q_lab_x = -|q|^2 / (2 k0)
        assert q_lab[0] == pytest.approx(-(qmag ** 2) / (2 * k0), abs=1e-9)
        # elastic: |k_i + q/k0| == 1
        k = np.array([1.0, 0.0, 0.0]) + q_lab / k0
        assert np.linalg.norm(k) == pytest.approx(1.0, abs=1e-9)
        # |q| and the vertical component are invariant across the crossing
        assert np.linalg.norm(q_lab) == pytest.approx(qmag, abs=1e-9)
        assert q_lab[2] == pytest.approx(q_s[2], abs=1e-12)


def test_ewald_crossings_helper_matches_omegas():
    from midas_defect.geometry import ewald_crossing_omegas, ewald_crossings

    lam = 0.172979
    q_s = np.array([2.5, -0.8, 0.7])
    cr = ewald_crossings(q_s, lam)
    om = ewald_crossing_omegas(q_s, lam)
    assert len(cr) == om.size == 2
    assert [c["omega_rad"] for c in cr] == pytest.approx(list(om))


def test_reflection_along_rotation_axis_never_diffracts():
    """A reflection nearly parallel to the vertical rotation axis is blind."""
    from midas_defect.geometry import ewald_crossing_omegas

    om = ewald_crossing_omegas(np.array([1e-3, 1e-3, 3.0]), 0.172979)
    assert om.size == 0


# --------------------------------------------------------------------------- #
# b3: sample-wide doublet survey
# --------------------------------------------------------------------------- #

def _make_doublet_dataset(rng, n_doublets=8, n_spurious=40, dw=6.5):
    """A synthetic layer: n_doublets true doublets (co-located pixel pairs split by
    ~dw in omega) plus spurious satellites at random pixels/omegas."""
    rows, cols, oms, ext, inten, rung = [], [], [], [], [], []
    for k in range(n_doublets):
        r = rng.uniform(100, 1500)
        c = rng.uniform(100, 1400)
        o = rng.uniform(-170, 170)
        for member, dI in ((0, 1.0), (dw, 0.6)):
            rows.append(r + rng.normal(0, 0.5))
            cols.append(c + rng.normal(0, 0.5))
            oms.append(o + member)
            ext.append(rng.uniform(0.02, 0.06))       # compact
            inten.append(dI)
            rung.append(k % 2)                          # two rungs
    for _ in range(n_spurious):
        rows.append(rng.uniform(100, 1500))
        cols.append(rng.uniform(100, 1400))
        oms.append(rng.uniform(-180, 180))
        ext.append(rng.uniform(0.02, 0.30))
        inten.append(rng.uniform(0.2, 1.0))
        rung.append(rng.integers(0, 2))
    return (np.array(rows), np.array(cols), np.array(oms),
            np.array(ext), np.array(inten), np.array(rung))


def test_survey_detects_synthetic_doublet_over_null():
    rng = np.random.default_rng(0)
    row, col, om, ext, inten, rung = _make_doublet_dataset(rng)
    res = survey_doublets(
        row, col, om, q_extent_along_axis=ext, intensity=inten, rung=rung,
        compact_max=0.10, pixel_tol=2.0, dw_lo=4.5, dw_hi=8.5,
    )
    assert res.verdict == "doublet-present"
    assert res.n_doublets >= 5
    assert res.enrichment > 3.0
    # bright/weak member ratio recovered (~1/0.6)
    ratios = [p["intensity_ratio"] for p in res.pairs]
    assert np.median(ratios) == pytest.approx(1.0 / 0.6, rel=0.15)


def test_survey_null_only_reports_no_doublet():
    """Random satellites with no co-located ω-structure -> no doublet."""
    rng = np.random.default_rng(1)
    row = rng.uniform(100, 1500, 60)
    col = rng.uniform(100, 1400, 60)
    om = rng.uniform(-180, 180, 60)
    ext = rng.uniform(0.02, 0.06, 60)
    res = survey_doublets(row, col, om, q_extent_along_axis=ext,
                          compact_max=0.10, pixel_tol=2.0)
    assert res.verdict in ("no-doublet", "insufficient")


def test_survey_relrod_cut_removes_extended_candidates():
    rng = np.random.default_rng(2)
    row, col, om, ext, inten, rung = _make_doublet_dataset(rng, n_doublets=0,
                                                           n_spurious=30)
    ext[:] = 0.5   # all relrods
    res = survey_doublets(row, col, om, q_extent_along_axis=ext,
                          compact_max=0.10)
    assert res.n_candidates == 0
    assert res.verdict == "insufficient"


# --------------------------------------------------------------------------- #
# b4: modulation-type classifier
# --------------------------------------------------------------------------- #

def test_classify_recovers_displacement():
    orders = np.array([1, 2, 4, 5])
    hkls = np.stack([np.zeros(4), np.zeros(4), 3 * orders], axis=1)
    data = structure_factor_intensity(hkls, spacing_modulation=0.04)
    fit = classify_modulation(orders, data)
    assert fit.verdict == "displacement"
    assert fit.displacement_amplitude == pytest.approx(0.04, abs=0.01)
    assert fit.displacement_residual < 1e-3
    assert fit.order_rise_exponent > 0.5      # rises with order


def test_classify_recovers_composition():
    orders = np.array([1, 2, 4, 5])
    hkls = np.stack([np.zeros(4), np.zeros(4), 3 * orders], axis=1)
    data = structure_factor_intensity(hkls, comp_modulation=0.05)
    fit = classify_modulation(orders, data)
    assert fit.verdict == "composition"
    assert abs(fit.order_rise_exponent) < 0.5   # ~flat


def test_classify_rejects_fundamental_orders():
    with pytest.raises(ValueError):
        classify_modulation(np.array([1, 3]), np.array([1.0, 2.0]))
