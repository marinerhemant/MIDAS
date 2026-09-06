"""Reflection and precision budget versus omega range.

The properties pinned are monotonicity (more omega cannot give fewer
reflections or worse precision) and the thing the module exists to say: the
sensitive count is smaller than the raw count, and precision is NOT 1/sqrt(N).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import itertools
import numpy as np
import pytest

from midas_xaf.rotation_budget import (
    random_orientations, accepted_reflections, sigma_delta, rotation_budget,
)

A, B_, C = 5.2739, 5.2384, 20.5
LAM, LSD, PX = 0.42459, 349_622.0, 172.0
NR, NC, BR, BC = 1679, 1475, 867.75, 738.56


def _hkls(d_min=1.15):
    out = []
    for h, k, l in itertools.product(range(-4, 5), range(-4, 5), range(-12, 13)):
        if (h, k, l) == (0, 0, 0):
            continue
        d = 1.0 / np.sqrt((h / A) ** 2 + (k / B_) ** 2 + (l / C) ** 2)
        if d > d_min:
            out.append((h, k, l))
    return np.array(out)


def test_random_orientations_are_rotations():
    U = random_orientations(50, np.random.default_rng(0))
    assert U.shape == (50, 3, 3)
    for u in U:
        assert np.allclose(u @ u.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(u), 1.0)


def test_more_omega_never_accepts_fewer_reflections():
    hkl = _hkls()
    U = random_orientations(1, np.random.default_rng(4))[0]
    Bm = np.diag([1 / A, 1 / B_, 1 / C])
    counts = []
    for half in (6, 15, 22.5, 30, 45):
        keep = accepted_reflections(hkl, U, Bm, half_range_deg=half,
                                    wavelength_A=LAM, lsd_um=LSD, pixel_um=PX,
                                    det_rows=NR, det_cols=NC,
                                    beam_row=BR, beam_col=BC, beamstop_px=60.)
        counts.append(int(keep.sum()))
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0]


def test_a_wider_window_is_a_superset():
    hkl = _hkls()
    U = random_orientations(1, np.random.default_rng(9))[0]
    Bm = np.diag([1 / A, 1 / B_, 1 / C])
    kw = dict(wavelength_A=LAM, lsd_um=LSD, pixel_um=PX, det_rows=NR,
              det_cols=NC, beam_row=BR, beam_col=BC, beamstop_px=60.)
    narrow = accepted_reflections(hkl, U, Bm, half_range_deg=10.0, **kw)
    wide = accepted_reflections(hkl, U, Bm, half_range_deg=25.0, **kw)
    assert np.all(wide[narrow]), "a reflection accepted at 10 deg was lost at 25"


def test_sigma_delta_refuses_an_underdetermined_set():
    U = random_orientations(1, np.random.default_rng(1))[0]
    assert np.isinf(sigma_delta(np.array([[1, 0, 0], [0, 1, 0]]), U,
                                a=A, b=B_, c=C, sigma_g_inv_A=0.0026))


def test_sigma_delta_scales_with_the_measurement_error():
    hkl = _hkls()
    U = random_orientations(1, np.random.default_rng(2))[0]
    Bm = np.diag([1 / A, 1 / B_, 1 / C])
    keep = accepted_reflections(hkl, U, Bm, half_range_deg=30.0,
                                wavelength_A=LAM, lsd_um=LSD, pixel_um=PX,
                                det_rows=NR, det_cols=NC, beam_row=BR,
                                beam_col=BC, beamstop_px=60.)
    s1 = sigma_delta(hkl[keep], U, a=A, b=B_, c=C, sigma_g_inv_A=0.0026)
    s2 = sigma_delta(hkl[keep], U, a=A, b=B_, c=C, sigma_g_inv_A=0.0052)
    assert s2 == pytest.approx(2.0 * s1, rel=1e-6)


def test_budget_is_monotone_and_sensitive_count_is_the_smaller_one():
    budget = rotation_budget(_hkls(), a=A, b=B_, c=C, wavelength_A=LAM,
                             lsd_um=LSD, pixel_um=PX, det_rows=NR, det_cols=NC,
                             beam_row=BR, beam_col=BC, beamstop_px=60.0,
                             half_ranges_deg=(6, 15, 22.5, 30),
                             n_orientations=25)
    t = budget.table
    assert list(t["reflections_per_grain"]) == sorted(t["reflections_per_grain"])
    assert np.all(t["sensitive_per_grain"] <= t["reflections_per_grain"])
    # precision improves (or holds) as omega grows
    assert list(t["sigma_delta_pct"]) == sorted(t["sigma_delta_pct"], reverse=True)
    assert list(t["frac_reaching_target"]) == sorted(t["frac_reaching_target"])


def test_precision_is_not_root_n():
    """If sigma were 1/sqrt(N) the ratio would match; it must not."""
    budget = rotation_budget(_hkls(), a=A, b=B_, c=C, wavelength_A=LAM,
                             lsd_um=LSD, pixel_um=PX, det_rows=NR, det_cols=NC,
                             beam_row=BR, beam_col=BC, beamstop_px=60.0,
                             half_ranges_deg=(6, 30), n_orientations=25)
    t = budget.table
    n_ratio = t["reflections_per_grain"].iloc[1] / t["reflections_per_grain"].iloc[0]
    s_ratio = t["sigma_delta_pct"].iloc[0] / t["sigma_delta_pct"].iloc[1]
    assert not np.isclose(s_ratio, np.sqrt(n_ratio), rtol=0.05), (
        f"sigma ratio {s_ratio:.3f} matched sqrt(N) {np.sqrt(n_ratio):.3f} — "
        "the joint orientation+cell fit is not being modelled")


def test_required_half_range_reads_off_the_table():
    budget = rotation_budget(_hkls(), a=A, b=B_, c=C, wavelength_A=LAM,
                             lsd_um=LSD, pixel_um=PX, det_rows=NR, det_cols=NC,
                             beam_row=BR, beam_col=BC, beamstop_px=60.0,
                             half_ranges_deg=(6, 15, 22.5, 30),
                             target_delta_pct=0.1, n_orientations=25)
    got = budget.required_half_range(0.90)
    if got is not None:
        row = budget.table.set_index("half_range_deg").loc[got]
        assert row["frac_reaching_target"] >= 0.90
