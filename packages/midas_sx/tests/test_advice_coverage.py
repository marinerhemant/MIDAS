"""advice.coverage: rotation range / Friedel pairs, completeness / redundancy / direction holes, next setting, resolution limit.

Stand-in runs: the observation table is built from the package's own predicted Ewald crossings of a small P 2_1 2_1 2_1 crystal (3 atoms), with F2 = calculated |F|^2 * 1e4 * (1 + 3 % noise)
and sigma = 3 %. Defects are injected by removing rows or shortening the scan, so every positive control has an exactly known cause; the clean control is a complete 360 deg set.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_hkls.structure_factor import f2_normalised
from midas_sx.advice import THRESHOLDS, Acquisition, AdviceReport, Finding
from midas_sx.advice import coverage
from midas_sx.geometry import Scan, make_geometry
from midas_sx.predict import enumerate_hkl, predicted_crossings
from midas_sx.workflow import merge_runs
from midas_sx.advice import assess as _assess

LAT = Lattice(5.8033, 5.95953, 12.2978, 90, 90, 90)
SG = SpaceGroup.from_hm("P 21 21 21")
ATOMS = [Atom("C", (0.10, 0.22, 0.31), 1.0, 0.5), Atom("N", (0.43, 0.31, 0.12), 1.0, 0.5), Atom("O", (0.71, 0.62, 0.52), 1.0, 0.5)]
CRYSTAL = Crystal(LAT, SG, ATOMS, "stand-in")
CRYSTAL_NO_ATOMS = Crystal(LAT, SG, [], "no-atoms")
CRYSTAL_P1 = Crystal(LAT, SpaceGroup.from_hm("P 1"), ATOMS[:2], "P1 stand-in")     # no symmetry beyond Friedel: a cone around the axis cannot be filled by equivalents
COVERAGE_CHECKS = ["scan_range", "completeness_redundancy", "next_setting", "resolution_limit"]


def _geom(n_frames, step):
    scan = Scan(0.0, step, n_frames)
    return make_geometry(wavelength_A=0.689, dist_mm=100.0, pixel_mm=0.172, centre_rc=(300.0, 300.0), shape=(600, 600), scan=scan)


def _run(U, range_deg=360.0, step=0.5, drop=None, sig_rel=0.03, seed=0, refined=False, isig_slope=None, crystal=None):
    """A stand-in Reduction. ``drop(table, U) -> bool mask of rows to remove``; ``isig_slope`` makes sigma grow with 1/d (a signal-limited high-resolution end)."""
    cr = crystal or CRYSTAL
    lat, sg, atoms = cr.lattice, cr.space_group, cr.atoms
    rng = np.random.default_rng(seed)
    nf = int(round(range_deg / step))
    geom = _geom(nf, step)
    scan = Scan(geom.omega_first_deg - 0.5 * step, step, nf)
    hkl = enumerate_hkl(lat, 1.10, sg)
    p = predicted_crossings(U, lat, hkl, geom, scan)
    f2 = f2_normalised(lat, sg, atoms, p[["h", "k", "l"]].to_numpy(int)) * 1e4
    F2 = f2 * (1.0 + 0.03 * rng.standard_normal(len(p)))
    sig = sig_rel * np.abs(F2)
    if isig_slope is not None:
        d = np.array([lat.d_spacing(int(a), int(b), int(c)) for a, b, c in p[["h", "k", "l"]].to_numpy(int)])
        sig = np.abs(F2) / (30.0 * np.exp(-isig_slope * (1.0 / d - 1.0 / d.max())) + 0.05)
    obs = pd.DataFrame(dict(h=p.h.astype(int), k=p.k.astype(int), l=p.l.astype(int), frame=p.frame.to_numpy(float), F2=np.abs(F2) + 1e-3, sigF2=sig + 1e-6))
    obs["L"] = 1.0
    if drop is not None:
        obs = obs[~drop(obs, U)].reset_index(drop=True)
    return SimpleNamespace(geom=geom, U=U, obs=obs, spots=None, counts={}, orientation=(SimpleNamespace() if refined else None))


def _drop_classes(frac, seed, sg=SG):
    """drop(table, U): remove every observation of a random ``frac`` of the unique classes (a reflection missing from the data set, not just thinned)."""
    def f(t, U):
        rep = np.array([sg.asu_representative(int(a), int(b), int(c)) for a, b, c in t[["h", "k", "l"]].to_numpy(int)])
        keys = [tuple(r) for r in rep]
        uniq = sorted(set(keys))
        rng = np.random.default_rng(seed)
        gone = {u for u in uniq if rng.random() < frac}
        return np.array([k in gone for k in keys], bool)
    return f


def _U(seed):
    return R.random(random_state=seed).as_matrix()


def assess(runs, crystal, **kw):
    """advice.assess with the merged data built from the observation tables (merge_runs accepts Reductions or DataFrames; the stand-ins are neither)."""
    merged = merge_runs({k: r.obs for k, r in runs.items()}, crystal.space_group) if runs else None
    return _assess(runs, crystal, merged_runs=merged, **kw)


def _ids(rep):
    return {f.id: f for f in rep.findings}


@pytest.fixture(scope="module")
def clean_runs():
    return {"a": _run(_U(1), seed=1), "b": _run(_U(2), seed=2), "c": _run(_U(3), seed=3)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_clean_complete_360deg_set_has_no_limiting_finding(clean_runs):
    rep = assess(clean_runs, CRYSTAL, checks=COVERAGE_CHECKS)
    assert isinstance(rep, AdviceReport)
    assert rep.ok, rep.text()
    assert not rep.checks_unavailable
    assert "scan_range" not in _ids(rep) and "completeness" not in _ids(rep) and "next_setting" not in _ids(rep) and "redundancy" not in _ids(rep)


def test_short_scan_fires_scan_range_and_is_limiting_only_with_refined_geometry_and_few_pairs():
    runs = {"a": _run(_U(11), range_deg=90.0, seed=11), "b": _run(_U(12), range_deg=90.0, seed=12)}
    f = _ids(assess(runs, CRYSTAL, checks=["scan_range"]))["scan_range"]
    assert f.severity == "note" and f.measured["short_runs"] == ["a", "b"]
    assert f.measured["per_run"]["a"]["pairs_predicted"] < f.measured["pairs_predicted_360deg_same_orientation"] if "pairs_predicted_360deg_same_orientation" in f.measured else True
    refined = {k: _run(_U(11 + i), range_deg=90.0, seed=11 + i, refined=True) for i, k in enumerate("ab")}
    g = _ids(assess(refined, CRYSTAL, checks=["scan_range"]))["scan_range"]
    pairs = min(v["pairs_observed"] for v in g.measured["per_run"].values())
    assert (g.severity == "limiting") == (pairs < THRESHOLDS["friedel_min_pairs"][0])
    # the sample environment sets the range: never limiting
    h = _ids(assess(refined, CRYSTAL, acquisition=Acquisition(max_range_deg=90.0), checks=["scan_range"]))["scan_range"]
    assert h.severity == "note" and "sample environment" in h.action


def test_scan_range_pairs_grow_with_the_range():
    p90 = _run(_U(21), range_deg=90.0, seed=21)
    p360 = _run(_U(21), range_deg=360.0, seed=21)
    assert coverage._observed_pairs(p360.obs) > coverage._observed_pairs(p90.obs) >= 0


def test_half_the_unique_reflections_removed_fires_completeness_limiting(clean_runs):
    keep = {k: _run(r.U, seed=hash(k) % 97, drop=_drop_classes(0.5, 7)) for k, r in clean_runs.items()}
    rep = assess(keep, CRYSTAL, checks=["completeness_redundancy"])
    f = _ids(rep)["completeness"]
    assert f.severity == "limiting"
    assert f.measured["completeness_n2"] < THRESHOLDS["completeness_n2"][0]
    assert "strong_half_completeness_n2" in f.measured and f.measured["expected_unique"] > 20
    assert "geometric" in f.scope
    assert f.measured["reachable_but_not_observed"] > f.measured["expected_but_not_reachable"] and "weak, masked or crowded" in f.action


def test_completeness_without_atoms_skips_the_strong_half_figure_and_says_so(clean_runs):
    keep = {k: _run(r.U, seed=hash(k) % 97, drop=_drop_classes(0.5, 8)) for k, r in clean_runs.items()}
    f = _ids(assess(keep, CRYSTAL_NO_ATOMS, checks=["completeness_redundancy"]))["completeness"]
    assert "not computed" in str(f.measured["strong_half_completeness"])


def test_thinning_observations_but_keeping_every_class_is_not_a_completeness_problem(clean_runs):
    rng = np.random.default_rng(9)
    thin = {k: _run(r.U, seed=hash(k) % 97, drop=lambda t, U: rng.random(len(t)) < 0.5) for k, r in clean_runs.items()}
    assert "completeness" not in _ids(assess(thin, CRYSTAL, checks=["completeness_redundancy"]))


def test_missing_cone_around_the_rotation_axis_fires_direction_holes():
    A = np.asarray(__import__("midas_sx.assign", fromlist=["reciprocal_basis"]).reciprocal_basis(LAT), float)
    U = _U(71)

    def drop_axis(t, U_):
        q = (U_ @ A @ t[["h", "k", "l"]].to_numpy(float).T).T
        psi = np.degrees(np.arccos(np.abs(q[:, 2]) / np.linalg.norm(q, axis=1)))
        return psi < 55.0

    runs = {k: _run(U, seed=71 + i, drop=drop_axis, crystal=CRYSTAL_P1) for i, k in enumerate("ab")}
    f = _ids(assess(runs, CRYSTAL_P1, checks=["completeness_redundancy"])).get("direction_holes")
    assert f is not None, "the cone was not detected"
    assert f.measured["worst_sector"]["psi_deg"] in ("0-30", "30-60")
    assert f.measured["worst_sector"]["n1"] < f.measured["overall_n1"] - THRESHOLDS["direction_gap"][0]
    # one orientation WITHOUT any injected loss still has the physical blind cone about the axis (q nearly parallel to the axis never crosses the Ewald sphere): the check must see it
    one = {k: _run(U, seed=71 + i, crystal=CRYSTAL_P1) for i, k in enumerate("ab")}
    assert "direction_holes" in _ids(assess(one, CRYSTAL_P1, checks=["completeness_redundancy"]))
    # the clean control: a second run with the crystal tilted by 90 deg about X fills the cone
    U2 = R.from_rotvec(np.radians(90.0) * np.array([1.0, 0.0, 0.0])).as_matrix() @ U
    two = {"a": _run(U, seed=71, crystal=CRYSTAL_P1), "b": _run(U2, seed=72, crystal=CRYSTAL_P1)}
    assert "direction_holes" not in _ids(assess(two, CRYSTAL_P1, checks=["completeness_redundancy"]))


def test_low_redundancy_fires_redundancy():
    one = {"a": _run(_U(31), range_deg=60.0, seed=31, crystal=CRYSTAL_P1)}
    f = _ids(assess(one, CRYSTAL_P1, checks=["completeness_redundancy"])).get("redundancy")
    assert f is not None and f.measured["mean_multiplicity"] < THRESHOLDS["mean_multiplicity"][0]
    assert set(f.measured["multiplicity_histogram"]) == {"1", "2", "3", "4", "5-8", "9+"}


def test_next_setting_recommends_a_rotation_that_reaches_what_a_repeat_cannot():
    runs = {"a": _run(_U(41), range_deg=60.0, seed=41)}
    f = _ids(assess(runs, CRYSTAL, checks=["next_setting"], setting_step_deg=30.0))["next_setting"]
    m = f.measured
    assert m["predicted_new_reflections_best_rotation"] > m["repeat_same_orientation_control"]["new_reflections"]
    assert m["predicted_gain_n2_best_rotation"] > m["repeat_same_orientation_control"]["gain_n2"]
    assert m["predicted_completeness_n2_after_one_more_run"] > m["completeness_n2_now"]
    assert "re-oriented" in f.title and "PREDICTED" in f.action and "PREDICTED" in (f.gain or "")
    assert m["best_rotation"]["angle_deg"] % 30.0 == 0 and m["seconds"] < 120


def test_next_setting_says_repeat_when_the_missing_reflections_are_reachable_but_seen_too_often_once():
    # one orientation, whole scan: everything reachable is reached once or twice; thin the data so that classes are seen once, not zero times
    rng = np.random.default_rng(3)
    thin = {"a": _run(_U(43), range_deg=360.0, seed=43, crystal=CRYSTAL_P1, drop=lambda t, U: rng.random(len(t)) < 0.55)}
    rep = assess(thin, CRYSTAL_P1, checks=["completeness_redundancy", "next_setting"], setting_step_deg=45.0)
    f = _ids(rep).get("next_setting")
    assert f is not None and f.measured["repeat_same_orientation_control"]["gain_n2"] >= THRESHOLDS["setting_min_gain"][0]


def test_next_setting_is_silent_on_a_complete_set_and_can_be_switched_off(clean_runs):
    assert "next_setting" not in _ids(assess(clean_runs, CRYSTAL, checks=["next_setting"]))
    rep = assess(clean_runs, CRYSTAL, checks=["next_setting"], suggest_setting=False)
    assert rep.checks_unavailable and "suggest_setting" in rep.checks_unavailable[0]


def test_resolution_limit_signal_limited_vs_detector_limited():
    U = _U(51)
    weak = {"a": _run(U, seed=51), "b": _run(_U(52), seed=52)}
    weak = {k: _run(r.U, seed=hash(k) % 91, isig_slope=9.0) for k, r in weak.items()}
    f = _ids(assess(weak, CRYSTAL, checks=["resolution_limit"]))
    assert "resolution_signal_limited" in f or "resolution_detector_limited" in f
    strong = {"a": _run(U, seed=51), "b": _run(_U(52), seed=52)}
    g = _ids(assess(strong, CRYSTAL, checks=["resolution_limit"]))
    assert "resolution_detector_limited" in g and g["resolution_detector_limited"].severity == "note"
    assert g["resolution_detector_limited"].measured["shells"] and all(sh["median_I_over_sigma"] > 2.0 for sh in g["resolution_detector_limited"].measured["shells"])
    # signal dying well before the detector edge: very steep I/sigma fall
    dead = {k: _run(r.U, seed=hash(k) % 89, isig_slope=40.0) for k, r in weak.items()}
    h = _ids(assess(dead, CRYSTAL, checks=["resolution_limit"]))
    assert "resolution_signal_limited" in h and h["resolution_signal_limited"].severity == "limiting"
    assert "systematic_floor" in h["resolution_signal_limited"].action


def test_no_runs_is_unavailable_not_a_crash():
    rep = assess({}, CRYSTAL, checks=["scan_range", "completeness_redundancy"])
    assert len(rep.checks_unavailable) == 2 and not rep.findings


def test_every_finding_carries_the_contract_fields(clean_runs):
    runs = {"a": _run(_U(61), range_deg=60.0, seed=61, refined=True)}
    rep = assess(runs, CRYSTAL, checks=COVERAGE_CHECKS, setting_step_deg=45.0)
    assert rep.findings
    for f in rep.findings:
        assert isinstance(f, Finding) and f.measured and f.threshold and f.evidence and f.action and f.scope is not None
        assert f.severity in ("blocking", "limiting", "note")
    txt = rep.text()
    assert "measured" in txt and "do " in txt


def test_thresholds_added_by_coverage_are_in_the_shared_table():
    for k in ("friedel_min_pairs", "direction_gap", "multiplicity_n1_fraction", "resolution_signal_ratio", "setting_min_gain"):
        assert k in THRESHOLDS and len(THRESHOLDS[k]) == 2


# --- sample-environment aperture (PREREG_advice.md Amendment 2) ------------------------------------------------------------------------------------------------
def _blocked_drop(dac, range_deg=360.0, step=0.5):
    """drop(table, U): remove the observations of the crossings that the cell opening blocks (they leave no blob in real data)."""
    def f(t, U):
        nf = int(round(range_deg / step))
        geom = _geom(nf, step)
        scan = Scan(geom.omega_first_deg - 0.5 * step, step, nf)
        p = predicted_crossings(U, LAT, enumerate_hkl(LAT, 1.10, SG), geom, scan)
        acc = coverage._accessible(SimpleNamespace(acq=Acquisition(dac=dac)), SimpleNamespace(geom=geom), p)
        blocked = set(zip(p.h[~acc].astype(int), p.k[~acc].astype(int), p.l[~acc].astype(int), np.round(p.frame[~acc].to_numpy(float), 6)))
        return np.array([(int(h), int(k), int(l), round(float(fr), 6)) in blocked for h, k, l, fr in zip(t.h, t.k, t.l, t.frame)])
    return f


def test_a_declared_cell_aperture_removes_blocked_reflections_from_the_denominator():
    from midas_sx.reduce import DacModel
    dac = DacModel(axis0=(1.0, 0.03, 0.03), t_plus_mm=2.5, t_minus_mm=3.0, mu_per_cm=9.0, half_opening_deg=35.0)
    runs = {"a": _run(_U(3), range_deg=360.0, drop=_blocked_drop(dac))}
    plain = assess(runs, CRYSTAL, acquisition=Acquisition(), checks=["completeness_redundancy"])
    cell = assess(runs, CRYSTAL, acquisition=Acquisition(dac=dac), checks=["completeness_redundancy"])
    c0 = [f for f in plain.findings if f.id == "completeness"]
    assert c0 and c0[0].measured["completeness_n1"] < THRESHOLDS["completeness_n1"][0]          # without the aperture the blocked ones read as missing
    num = cell.numbers["completeness"]
    assert num["blocked_by_aperture"] > 0 and num["n_accessible"] < num["n_expected"]
    assert num["frac_n_ge_1"] >= THRESHOLDS["completeness_n1"][0]                              # every accessible class was observed
    assert not [f for f in cell.findings if f.id == "completeness" and f.severity == "limiting"]


def test_without_a_declared_aperture_nothing_is_blocked():
    runs = {"a": _run(_U(3), range_deg=360.0)}
    rep = assess(runs, CRYSTAL, acquisition=Acquisition(), checks=["completeness_redundancy"])
    assert rep.numbers["completeness"]["blocked_by_aperture"] is None
    assert rep.numbers["completeness"]["n_accessible"] == rep.numbers["completeness"]["n_expected"]


def test_resolution_limit_counts_the_reflections_that_were_never_detected():
    """PREREG_advice.md Amendment 3: the observed reflections of the outer shells are strong (survivorship), but most of the expected ones are absent."""
    def far(t, U):
        d = np.array([LAT.d_spacing(int(a), int(b), int(c)) for a, b, c in t[["h", "k", "l"]].to_numpy(int)])
        return (d < 1.4) & (np.random.default_rng(5).random(len(t)) < 0.97)        # the stand-in detector reaches d = 1.1 A: beyond 1.4 A nearly everything is lost
    runs = {"a": _run(_U(61), seed=61, drop=far), "b": _run(_U(62), seed=62, drop=far)}
    f = _ids(assess(runs, CRYSTAL, checks=["resolution_limit"]))
    assert "resolution_signal_limited" in f and f["resolution_signal_limited"].severity == "limiting"
    assert f["resolution_signal_limited"].measured["outer_shell_n1_completeness"] < 0.5
    assert all(sh["median_I_over_sigma"] > 2.0 for sh in [dict(median_I_over_sigma=v) for v in f["resolution_signal_limited"].measured["median_I_over_sigma_of_observed"]])   # the observed ones look fine
    clean = _ids(assess({"a": _run(_U(61), seed=61), "b": _run(_U(62), seed=62)}, CRYSTAL, checks=["resolution_limit"]))
    assert "resolution_signal_limited" not in clean


def test_a_resolution_target_cuts_the_expected_set_and_changes_nothing_by_default():
    runs = {"a": _run(_U(71), seed=71), "b": _run(_U(72), seed=72)}
    base = assess(runs, CRYSTAL, checks=["completeness_redundancy"])
    cut = assess(runs, CRYSTAL, checks=["completeness_redundancy"], d_min=2.0)
    n0 = base.numbers["completeness"]["n_expected"]
    n1 = cut.numbers["completeness"]["n_expected"]
    assert 0 < n1 < n0
    again = assess(runs, CRYSTAL, checks=["completeness_redundancy"])
    assert again.numbers["completeness"]["n_expected"] == n0
