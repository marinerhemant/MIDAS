"""advice.quality: a positive control per check (an injected defect makes it fire), a clean control (nothing limiting), and the unavailable paths."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.advice import Acquisition, assess
from midas_sx.geometry import Scan, make_geometry
from midas_sx.merge import scale_groups, scale_smooth, systematic_floor

SG = SpaceGroup.from_hm("P 21 21 21")
CRYSTAL = Crystal(Lattice(5.8, 5.96, 12.3, 90, 90, 90), SG, [Atom("C", (0.1, 0.2, 0.3), 1.0, 0.5, label="C1")], "toy")
QUALITY = ["systematic_floor", "intensity_drift", "dynamic_range", "omega_sampling", "geometry_consistency", "contamination"]
TRUTH_SEED = 100


def _geom(wedge=0.0, tilts=(0.0, 0.0, 0.0)):
    g = make_geometry(wavelength_A=0.689, dist_mm=170.0, pixel_mm=0.172, centre_rc=(800.0, 400.0), shape=(1679, 1475), scan=Scan(-15.0, 0.2, 850))
    return replace(g, wedge_deg=wedge, tx_deg=tilts[0], ty_deg=tilts[1], tz_deg=tilts[2])


def _truth(n_unique=300):
    rng = np.random.default_rng(TRUTH_SEED)
    grid = np.array([(a, b, c) for a in range(1, 12) for b in range(1, 12) for c in range(1, 12)])
    base = grid[rng.choice(len(grid), n_unique, replace=False)]
    return base, 10 ** rng.uniform(1.5, 4.0, n_unique)


def _run(seed, *, drift=None, floor=0.0, counting=0.03, width=6.0, peak=500.0, peak_hi_frac=0.0, peak_hi=2e5, hi_floor=0.0, n_obs=1200, wedge=0.0,
         n_frames=850, rms=(0.6, 0.5), converged=True, eps=0.0, extra_spots=None, rejected=None, L_floor_slope=0.0):
    rng = np.random.default_rng(seed)
    base, true = _truth()
    u = rng.integers(0, len(base), n_obs)
    sgn = rng.choice([-1, 1], (n_obs, 3))
    fr = rng.uniform(0, n_frames, n_obs)
    L = rng.uniform(1.0, 6.0, n_obs)
    kt = np.array([drift(f) for f in fr]) if drift else np.ones(n_obs)
    hi = rng.random(n_obs) < peak_hi_frac
    sc = np.sqrt(floor ** 2 + (hi_floor ** 2) * hi + (L_floor_slope * (L - 1.0)) ** 2)
    f2 = true[u] * kt * np.exp(sc * rng.standard_normal(n_obs)) * (1 + counting * rng.standard_normal(n_obs))
    obs = pd.DataFrame(dict(h=base[u, 0] * sgn[:, 0], k=base[u, 1] * sgn[:, 1], l=base[u, 2] * sgn[:, 2], frame=fr, row=rng.uniform(50, 1600, n_obs), col=rng.uniform(50, 1400, n_obs),
                            omega=fr * 0.2 - 15.0, L=L, PT=0.9, F2=f2, sigF2=counting * np.abs(f2)))
    obs["I"] = obs.F2 * obs.L * obs.PT
    obs["sigI"] = obs.sigF2 * obs.L * obs.PT
    spots = pd.DataFrame(dict(frame=obs.frame, row=obs.row, col=obs.col, integrated=obs.I, peak_counts=np.where(hi, peak_hi, peak), omega_width_frames=width))
    if extra_spots is not None:
        spots = pd.concat([spots, extra_spots], ignore_index=True)
    rej = rejected if rejected is not None else pd.DataFrame(columns=["i", "reason"])
    orient = SimpleNamespace(rms_px=rms[0], rms_frame=rms[1], converged=converged, lambda_cell_scale=eps, sigma_lambda_cell_scale=1e-5 if eps else float("nan"))
    return SimpleNamespace(obs=obs, spots=spots, rejected=rej, counts={}, geom=_geom(wedge), orientation=orient, U=np.eye(3), profile=None)


def _runs(**kw):
    return {f"r0{i}": _run(i, **{**kw, **({"wedge": kw.get("wedge", 0.0) + 0.0} if False else {})}) for i in (2, 3, 4)}


def _assess(runs, only, **kw):
    return assess(runs, CRYSTAL, checks=only, **kw)


def _ids(rep, sev=None):
    return [f.id for f in rep.findings if sev is None or f.severity == sev]


# --- merge.systematic_floor -----------------------------------------------------------------------------------------------------------------
def test_systematic_floor_helper_recovers_injected_scatter_and_zero_for_counting():
    r = _run(1, floor=0.15)
    tab = systematic_floor(r.obs, SG)
    assert 0.11 < tab["floor"].iloc[0] < 0.19 and tab["counting"].iloc[0] < 0.06
    r0 = _run(1, floor=0.0)
    assert systematic_floor(r0.obs, SG)["floor"].iloc[0] < 0.015


def test_systematic_floor_by_groups_and_errors():
    r = _run(2, L_floor_slope=0.05)
    tab = systematic_floor(r.obs, SG, by=pd.cut(r.obs["L"], [0, 2, 4, 7]), min_obs=30)
    assert tab["floor"].iloc[-1] > 1.8 * tab["floor"].iloc[0]
    assert set(systematic_floor(r.obs.assign(g=np.where(r.obs.frame > 400, "b", "a")), SG, by="g").index) == {"a", "b"}
    with pytest.raises(ValueError):
        systematic_floor(r.obs, SG, by=np.arange(3))
    assert systematic_floor(r.obs.iloc[:0], SG).empty


# --- systematic_floor check -----------------------------------------------------------------------------------------------------------------
def test_floor_check_fires_on_systematic_scatter_and_not_on_counting_limited_data():
    bad = _assess(_runs(floor=0.2), ["systematic_floor"])
    f = [x for x in bad.findings if x.id == "systematic_floor"][0]
    assert f.severity == "limiting" and "MORE EXPOSURE WILL NOT HELP" in f.action and f.measured["ratio"] > 2
    clean = _assess(_runs(floor=0.0), ["systematic_floor"])
    assert clean.ok and clean.findings[0].severity == "note"


def test_floor_check_names_the_lorentz_dependence():
    rep = _assess(_runs(L_floor_slope=0.08), ["systematic_floor"])
    f = rep.findings[0]
    assert f.severity == "limiting" and "Lorentz factor" in f.action


# --- intensity drift ------------------------------------------------------------------------------------------------------------------------
def _drift(f):
    return float(np.exp(0.22 * np.sin(2 * np.pi * f / 850.0 * 1.5)))


def test_drift_fires_on_injected_smooth_drift_and_not_on_clean_data():
    rep = _assess(_runs(drift=_drift), ["intensity_drift"])
    f = rep.findings[0]
    assert f.id == "intensity_drift" and f.severity == "limiting" and "INFERRED" in f.action and f.measured["placebo_gain"] < 0.3 * f.measured["rint_gain"]
    assert _assess(_runs(), ["intensity_drift"]).findings == []
    assert "I0 is recorded" in _assess(_runs(drift=_drift), ["intensity_drift"], acquisition=Acquisition(has_i0=True)).findings[0].action


def test_drift_unavailable_without_redundancy():
    r = {"a": _run(9, n_obs=40)}
    rep = _assess(r, ["intensity_drift"])
    assert any(s.startswith("intensity_drift") for s in rep.checks_unavailable)


# --- dynamic range --------------------------------------------------------------------------------------------------------------------------
def test_dynamic_range_fires_when_strong_peaks_scatter_more():
    runs = _runs(peak_hi_frac=0.1, hi_floor=0.25)
    rep = _assess(runs, ["dynamic_range"], acquisition=Acquisition(count_cutoff=791927.0, exposure_s=0.2))
    f = rep.findings[0]
    assert f.id == "dynamic_range" and f.severity == "limiting" and f.measured["floor_effect"] >= 0.03
    note = _assess(_runs(peak_hi_frac=0.1), ["dynamic_range"], acquisition=Acquisition(count_cutoff=791927.0)).findings[0]
    assert note.severity == "note"


def test_dynamic_range_silent_when_no_peak_is_near_the_limit_and_guess_is_labelled():
    assert _assess(_runs(), ["dynamic_range"], acquisition=Acquisition(count_cutoff=791927.0)).findings == []
    f = [f for f in _assess(_runs(peak_hi_frac=0.05), ["dynamic_range"]).findings if "limit_is_a_guess" in f.measured][0]
    assert f.measured["limit_is_a_guess"] is True


def test_dynamic_range_unavailable_without_peak_counts():
    runs = _runs()
    for r in runs.values():
        r.spots = r.spots.drop(columns=["peak_counts"])
    assert any(s.startswith("dynamic_range") for s in _assess(runs, ["dynamic_range"]).checks_unavailable)


# --- omega sampling -------------------------------------------------------------------------------------------------------------------------
def test_omega_sampling_flags_single_frame_spots_and_oversampled_and_stays_quiet_between():
    f = _assess(_runs(width=0.0), ["omega_sampling"]).findings[0]
    assert f.severity == "note" and "finer rotation step" in f.action and f.measured["single_frame_fraction"] >= 0.99
    # two-frame spots (RMS width 0.5) are not single-frame: the old median<3 rule fired on every scan, this one does not
    assert _assess(_runs(width=0.5), ["omega_sampling"]).findings == []
    assert _assess(_runs(width=8.0), ["omega_sampling"]).findings == []
    assert "oversampled" in _assess(_runs(width=45.0), ["omega_sampling"]).findings[0].title


def test_dynamic_range_clipping_is_seen_only_when_peaks_sit_at_the_cutoff():
    cut = 4e5
    clipped = _runs(peak_hi_frac=0.1, peak_hi=cut)
    unclipped = _runs(peak_hi_frac=0.1, peak_hi=0.5 * cut)
    f_c = [f for f in _assess(clipped, ["dynamic_range"], acquisition=Acquisition(count_cutoff=cut)).findings if "saturated" in f.title]
    assert len(f_c) == 1 and f_c[0].severity == "limiting" and f_c[0].measured["observations_clipped"] > 0
    # same cutoff, same fraction of strong peaks, nothing at the cutoff: no saturation finding (the control the first validation lacked)
    assert not [f for f in _assess(unclipped, ["dynamic_range"], acquisition=Acquisition(count_cutoff=cut)).findings if "saturated" in f.title]


def test_dynamic_range_clipping_without_a_cutoff_needs_ties_at_the_maximum():
    tied = _runs(peak_hi_frac=0.1, peak_hi=3e5)                      # every strong peak identical: clipped
    assert [f for f in _assess(tied, ["dynamic_range"]).findings if "saturated" in f.title]
    rng = np.random.default_rng(0)
    cont = _runs(peak_hi_frac=0.1)
    for r in cont.values():
        r.spots["peak_counts"] = np.where(r.spots["peak_counts"] > 1e3, rng.uniform(1.5e5, 3e5, len(r.spots)), r.spots["peak_counts"])   # continuous: unique maximum
    assert not [f for f in _assess(cont, ["dynamic_range"]).findings if "saturated" in f.title]


# --- geometry -------------------------------------------------------------------------------------------------------------------------------
def test_geometry_notes_a_tilted_axis_and_limits_on_a_bad_fit():
    runs = {f"r0{i}": _run(i, wedge=w) for i, w in zip((2, 3, 4), (0.41, 0.31, 0.29))}
    rep = _assess(runs, ["geometry_consistency"])
    assert _ids(rep) == ["geometry_consistency"] and rep.findings[0].severity == "note" and "calibrant" in rep.findings[0].action
    bad = {f"r0{i}": _run(i, rms=(3.5, 0.4)) for i in (2, 3, 4)}
    assert _assess(bad, ["geometry_consistency"]).findings[0].severity == "limiting"
    assert _assess({f"r0{i}": _run(i, converged=False) for i in (2, 3)}, ["geometry_consistency"]).findings[0].severity == "limiting"


def test_geometry_wavelength_scale_is_not_an_energy_calibration():
    runs = {f"r0{i}": _run(i, eps=e) for i, e in zip((2, 3, 4), (2.5e-3, 1.5e-3, 2.2e-3))}
    rep = _assess(runs, ["geometry_consistency"])
    f = [x for x in rep.findings if x.id == "wavelength_cell_scale"][0]
    assert "NOT an energy calibration" in f.scope and "20-40" in f.scope and f.measured["spread"] > 9e-4


def test_geometry_clean_is_silent_and_header_deviation_is_reported():
    assert _assess({"a": _run(1), "b": _run(2)}, ["geometry_consistency"]).findings == []
    runs = {"a": _run(1, wedge=0.3)}
    f = _assess(runs, ["geometry_consistency"], header_geometry=_geom(0.0)).findings[0]
    assert f.measured["deviation_from_header"]["a"]["wedge_deg"] == pytest.approx(0.3)


# --- contamination --------------------------------------------------------------------------------------------------------------------------
def _ring_spots(n=60, tth_deg=11.0, seed=0):
    g = _geom()
    r_px = g.lsd_um / 1000.0 * np.tan(np.radians(tth_deg)) / 0.172
    az = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return pd.DataFrame(dict(frame=np.random.default_rng(seed).uniform(0, 800, n), row=800.0 + r_px * np.sin(az), col=400.0 + r_px * np.cos(az), integrated=5e5, peak_counts=2e3, omega_width_frames=6.0))


def test_contamination_finds_a_constant_2theta_ring_and_calls_ice():
    runs = {}
    for i in (2, 3):
        extra = _ring_spots(seed=i)
        base = _run(i)
        n0 = len(base.spots)
        rej = pd.DataFrame(dict(i=np.arange(n0, n0 + len(extra)), reason="no_hkl"))
        runs[f"r0{i}"] = _run(i, extra_spots=extra, rejected=rej)
    # 11.0 deg 2theta at 0.689 A is d = 3.59 A, not an ice line; move to the Ih (002) line
    d_ice = 3.669
    tth_ice = float(np.degrees(2 * np.arcsin(0.689 / (2 * d_ice))))
    for i in (2, 3):
        extra = _ring_spots(tth_deg=tth_ice, seed=10 + i)
        base = _run(i)
        n0 = len(base.spots)
        runs[f"r0{i}"] = _run(i, extra_spots=extra, rejected=pd.DataFrame(dict(i=np.arange(n0, n0 + len(extra)), reason="no_hkl")))
    rep = _assess(runs, ["contamination"])
    f = [x for x in rep.findings if x.id == "unexplained_strong_blobs"][0]
    assert f.measured["rings"] and any(r["consistent_with_ice_Ih"] for r in f.measured["rings"]) and "frost" in f.action


def test_contamination_calls_sub_lattice_angle_blobs_near_beam_scattering_not_rings():
    runs = {}
    for i in (2, 3):
        base = _run(i)
        n0 = len(base.spots)
        rows = []
        for t in (0.4, 0.9, 1.3, 1.8):                                  # 2theta well below the first reflection of a 12.3 A cell at 0.689 A (2theta = 3.2 deg)
            rows.append(_ring_spots(n=12, tth_deg=t, seed=20 + i))
        extra = pd.concat(rows, ignore_index=True)
        runs[f"r0{i}"] = _run(i, extra_spots=extra, rejected=pd.DataFrame(dict(i=np.arange(n0, n0 + len(extra)), reason="no_hkl")))
    rep = _assess(runs, ["contamination"])
    ids = _ids(rep)
    assert "near_beam_scattering" in ids and "unexplained_strong_blobs" not in [f.id for f in rep.findings if f.measured.get("rings")]
    f = [x for x in rep.findings if x.id == "near_beam_scattering"][0]
    assert f.severity == "note" and "beamstop" in f.action


def test_contamination_flags_crowded_strong_reflections_and_clean_is_silent():
    runs = {}
    for i in (2, 3):
        base = _run(i)
        n0 = len(base.spots)
        top = np.argsort(-base.spots.integrated.to_numpy())[: int(0.06 * n0)]
        rej = pd.DataFrame(dict(i=top[: int(0.5 * len(top))], reason="crowded"))
        runs[f"r0{i}"] = _run(i, rejected=rej)
    f = _assess(runs, ["contamination"]).findings
    assert f and f[0].id == "crowded_strong_reflections" and f[0].severity == "limiting"
    assert _assess({f"r0{i}": _run(i) for i in (2, 3)}, ["contamination"]).findings == []


# --- clean control --------------------------------------------------------------------------------------------------------------------------
def test_clean_control_has_nothing_blocking_or_limiting():
    rep = _assess(_runs(), QUALITY)
    assert rep.ok, rep.text()
    assert not rep.checks_unavailable


def test_every_defect_at_once_is_reported_and_text_renders():
    runs = {f"r0{i}": _run(i, drift=_drift, floor=0.15, width=0.0, wedge=0.4, peak_hi_frac=0.1) for i in (2, 3, 4)}
    rep = _assess(runs, QUALITY, acquisition=Acquisition(count_cutoff=791927.0, has_i0=False, has_calibrant=False))
    assert not rep.ok
    txt = rep.text()
    for k in ("systematic_floor", "omega_sampling", "geometry_consistency", "do       :", "rule     :", "evidence :"):
        assert k in txt
    assert rep.to_dict()["ok"] is False
