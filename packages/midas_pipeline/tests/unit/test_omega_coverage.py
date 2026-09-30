"""Blocked omega windows (load-frame shadows): midas_pipeline.diagnostics.omega_coverage."""
import json
import numpy as np
from midas_pipeline.diagnostics.omega_coverage import blocked_omega_windows, report_blocked_omega


def _omegas(gaps, n=72000, seed=0):
    w = np.random.default_rng(seed).uniform(-180, 180, n)
    for lo, hi in gaps:
        w = w[(w < lo) | (w >= hi)]
    return w


def test_two_post_shadows_are_found_and_excluded():
    rep = blocked_omega_windows(_omegas([(-98, -82), (82, 98)]), ranges=[(-180, 180)])
    assert [(lo, hi) for lo, hi, _ in rep["windows"]] == [(-98.0, -82.0), (82.0, 98.0)]
    assert rep["suggested_ranges"] == [(-180.0, -99.0), (-81.0, 81.0), (99.0, 180.0)]


def test_uniform_coverage_reports_nothing():
    rep = blocked_omega_windows(_omegas([]), ranges=[(-180, 180)])
    assert rep["windows"] == [] and rep["suggested_ranges"] == [(-180.0, 180.0)]


def test_a_one_degree_dip_is_not_a_window():
    rep = blocked_omega_windows(_omegas([(10, 11)]), ranges=[(-180, 180)])
    assert rep["windows"] == []


def test_placeholder_ring0_rows_are_ignored():
    w = _omegas([]); ring = np.ones_like(w)
    w = np.r_[w, np.zeros(3000)]; ring = np.r_[ring, np.zeros(3000)]      # 3000 all-zero rows at omega 0
    assert blocked_omega_windows(w, ring, ranges=[(-180, 180)])["n_spots"] == w.size - 3000


def test_only_configured_ranges_are_searched():
    rep = blocked_omega_windows(_omegas([(-98, -82)]), ranges=[(-180, -102), (-77, 180)])
    assert rep["windows"] == []                 # the shadow is already outside the configured ranges


def test_report_writes_json_and_warns(tmp_path, caplog):
    w = _omegas([(-98, -82)]); sp = np.zeros((w.size, 10)); sp[:, 2] = w; sp[:, 5] = 1
    with caplog.at_level("WARNING"):
        rep = report_blocked_omega(sp, [(-180, 180)], tmp_path, tag="t")
    assert json.loads((tmp_path / "omega_coverage.json").read_text())["windows"] == [list(x) for x in rep["windows"]]
    assert "OmegaRange -180 -99" in caplog.text


def test_a_shadow_split_by_one_passing_bin_is_one_window():
    w = _omegas([(82, 90), (91, 98)])            # 1 deg of normal density inside the shadow
    rep = blocked_omega_windows(w, ranges=[(-180, 180)])
    assert [(lo, hi) for lo, hi, _ in rep["windows"]] == [(82.0, 98.0)]
