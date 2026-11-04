"""The one-call workflow (midas_sx.workflow): frames -> merged F² -> refined structure, against a small independent simulation."""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup           # noqa: E402
from midas_sx.reduce import DacModel, PolarizationModel             # noqa: E402
from midas_sx.refine import EIGHT_PI2                               # noqa: E402
from midas_sx.workflow import (OrientationNotConverged, ScanSetup, TwinReduction, reduce_scan, refine_structure,   # noqa: E402
                               refine_twin90_structure)


def _crystal(st, cell=None):
    return Crystal(Lattice(*(cell or st.cell)), SpaceGroup.from_hm(st.spacegroup),
                   [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)


@pytest.fixture(scope="module")
def scene():
    exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-38.0, n_frames=304,
                              dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
    st = sim.structure_cu()
    res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=40_000, max_pairs=8_000_000)
    raw = res.frames(seed=5)
    dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
                   mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
    # a header that is a little wrong: 0.5 px beam centre and 1 % distance
    setup = ScanSetup(exp.wavelength_A, exp.dist_mm * 1.01, exp.pixel_mm, (exp.centre_rc[0] + 0.5, exp.centre_rc[1] - 0.3),
                      exp.omega_start, exp.omega_step, dac=dac, pol=PolarizationModel(exp.pol_fraction))
    U0 = R.from_rotvec(np.radians(0.25) * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)).as_matrix() @ res.U
    return exp, st, res, raw, setup, U0


def _mis_deg(A, B):
    return float(np.degrees(np.linalg.norm(R.from_matrix(A @ B.T).as_rotvec())))


def test_reduce_scan_recovers_orientation_and_f2_from_a_wrong_start_and_the_refinement_runs(scene):
    exp, st, res, raw, setup, U0 = scene
    red = reduce_scan(raw, _crystal(st), U0, setup)
    assert red.orientation is not None and red.orientation.converged
    assert _mis_deg(red.U, res.U) < 0.05                                   # from 0.25 deg off
    assert red.counts["input"] == red.counts["kept"] + sum(v for k, v in red.counts.items() if k not in ("input", "kept"))
    assert len(red.merged) >= 5 and np.isfinite(red.r_int)
    sg = SpaceGroup.from_hm(st.spacegroup)
    ft = sim.f2_truth(st, red.merged[["h", "k", "l"]].to_numpy(int))
    strong = (red.merged.F2 / red.merged.sigF2 > 20).to_numpy()
    x, f = red.merged.F2.to_numpy()[strong], ft[strong]
    k = (x * f).sum() / (x * x).sum()
    assert strong.sum() >= 4 and np.median(np.abs(k * x / f - 1.0)) < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                      # a clean fit must not warn
        fit = refine_structure(_crystal(st), red.merged)
    assert fit.converged and fit.goof < 3.0


def test_reduce_scan_can_skip_the_orientation_refinement(scene):
    exp, st, res, raw, setup, U0 = scene
    exact = replace(setup, dist_mm=exp.dist_mm, centre_rc=exp.centre_rc)
    red = reduce_scan(raw, _crystal(st), res.U, exact, refine_orientation=False)
    assert red.orientation is None and np.array_equal(red.U, res.U)
    assert len(red.merged) >= 5


def test_a_wrong_cell_raises_unless_allowed_and_the_reason_is_kept(scene):
    exp, st, res, raw, setup, U0 = scene
    bad = _crystal(st, cell=(st.cell[0] * 1.06,) + tuple(st.cell[1:]))
    with pytest.raises(OrientationNotConverged) as e:
        reduce_scan(raw, bad, U0, setup)
    assert not e.value.result.converged and e.value.result.reason
    red = reduce_scan(raw, bad, U0, setup, allow_unconverged=True)         # the caller took the responsibility
    assert red.orientation is not None and not red.orientation.converged


def test_refine_structure_warns_when_it_does_not_converge():
    st = sim.structure_s2_bilayer()
    hkl = np.array([[0, 0, 2], [1, 0, 3], [1, 1, 4]])
    merged = pd.DataFrame(dict(h=hkl[:, 0], k=hkl[:, 1], l=hkl[:, 2], F2=[100.0, 50.0, 20.0], sigF2=[1.0, 1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match="refine_structure"):
        res = refine_structure(_crystal(st), merged)                       # 3 rows for >10 parameters
    assert not res.converged


def test_refine_twin90_structure_recovers_the_fraction_from_an_exact_twin_table():
    st = sim.structure_s2o()
    sg = SpaceGroup.from_hm(st.spacegroup)
    allh = sim.enumerate_hkl(st, 0.8)
    reps = np.array(sorted({sg.asu_representative(int(h), int(k), int(l)) for h, k, l in allh}))
    f2 = sim.f2_truth(st, reps)
    keep = f2 > 1e-6 * f2.max()
    hkl, f2 = reps[keep], f2[keep]
    f2tw = sim.twin_f2(st, 0.3)(hkl, f2)
    sigma = 0.01 * f2tw.max() * np.ones(len(f2tw))
    part = np.stack([hkl[:, 1], -hkl[:, 0], hkl[:, 2]], axis=1)
    obs = pd.DataFrame(dict(h=hkl[:, 0], k=hkl[:, 1], l=hkl[:, 2], F2=f2tw, sigF2=sigma, kind="blend", f1=1.0, f2=1.0,
                            p0=part[:, 0], p1=part[:, 1], p2=part[:, 2], sep_px=0.0))
    tred = TwinReduction(obs, {}, np.eye(3), np.eye(3), None, None, True, None, pd.DataFrame(), pd.DataFrame(), np.zeros((1, 1), bool), {})
    from midas_sx.refine import perturbed_crystal
    start = perturbed_crystal(_crystal(st), np.random.default_rng(1), shift_A=0.05)
    res = refine_twin90_structure(start, tred, robust=False)
    assert res.converged, res.reason
    assert abs(res.twin_fraction - 0.3) < 5e-3, res.twin_fraction
    with pytest.raises(ValueError):
        refine_twin90_structure(start, TwinReduction(obs.iloc[:5], {}, np.eye(3), np.eye(3), None, None, True, None, pd.DataFrame(),
                                                     pd.DataFrame(), np.zeros((1, 1), bool), {}))


def test_default_d_min_is_the_validated_inscribed_edge_not_the_corner():
    """The workflow's default reflection pool must be the one the validated chain used (simulate.Experiment.tth_max_deg: atan((n/2 - 3) px / distance)); the detector
    corner (2theta 47 against 37 deg on this geometry) doubled the pool and asked for reflections the validated runs never measured."""
    from midas_sx.workflow import _tth_max_deg
    exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-40.0, n_frames=320)
    setup = ScanSetup(exp.wavelength_A, exp.dist_mm, exp.pixel_mm, exp.centre_rc, exp.omega_start, exp.omega_step)
    g = setup.geometry((512, 512), 320)
    assert _tth_max_deg(g) == pytest.approx(exp.tth_max_deg(), rel=1e-12)
    corner = np.degrees(np.arctan(np.hypot(255.5, 255.5) * exp.pixel_mm / exp.dist_mm))
    assert corner > exp.tth_max_deg() + 5.0                           # the corner would have been a different pool
    off = ScanSetup(exp.wavelength_A, exp.dist_mm, exp.pixel_mm, (300.0, 255.5), exp.omega_start, exp.omega_step).geometry((512, 512), 320)
    assert _tth_max_deg(off) > _tth_max_deg(g)                        # an off-centre beam: the farthest edge decides


def test_the_twin_reduction_warns_when_it_falls_back_and_the_refinement_hints_at_a_minority_domain():
    import warnings as _w
    from midas_sx.workflow import _warn
    from types import SimpleNamespace
    res = SimpleNamespace(converged=False, reason="on bound: twin.m", at_bound=["twin.m"], twin_fraction=0.5, goof=1.0, poor_fit=False)
    with pytest.warns(RuntimeWarning, match="MAJORITY"):
        _warn(res, "refine_twin90_structure")
    res2 = SimpleNamespace(converged=False, reason="on bound: Ni.U", at_bound=["Ni.U"], twin_fraction=0.3, goof=1.2, poor_fit=False)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        _warn(res2, "refine_twin90_structure")
    assert rec and "MAJORITY" not in str(rec[0].message) and "GooF 1.20" in str(rec[0].message)


def test_reduction_U_does_not_alias_the_callers_array_without_orientation_refinement(scene):
    exp, st, res, raw, setup, U0 = scene
    U_in = U0.copy()
    red = reduce_scan(raw, _crystal(st), U_in, setup, refine_orientation=False)
    assert np.array_equal(red.U, U_in) and red.U is not U_in
    red.U[0, 0] += 1.0                                                      # a caller editing the result must not change its own input
    assert np.array_equal(U_in, U0)


def test_background_option_reaches_the_integration(scene):
    exp, st, res, raw, setup, U0 = scene
    mean = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False)
    quad = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, background="quadratic")
    assert mean.profile.background == "mean" and quad.profile.background == "quadratic"
    # the option must reach integrate_box (a mutation that dropped background= from the call in reduce.py left this test green: /verify artifact lens)
    o_m, o_q = mean.obs.sort_values(["frame", "row", "col"]).reset_index(drop=True), quad.obs.sort_values(["frame", "row", "col"]).reset_index(drop=True)
    assert len(o_m) == len(o_q) > 5
    assert (np.abs(o_q.bg.to_numpy() - o_m.bg.to_numpy()) > 0).mean() > 0.8
    both = mean.merged.merge(quad.merged, on=["h", "k", "l"], suffixes=("_m", "_q"))
    strong = (both.F2_m / both.sigF2_m > 20).to_numpy()
    assert strong.sum() >= 5
    rel = np.abs(both.F2_q - both.F2_m).to_numpy()[strong] / both.F2_m.to_numpy()[strong]
    assert np.median(rel) < 0.02, np.median(rel)                            # a flat simulated background: the two estimators agree on the strong reflections
    with pytest.raises(ValueError):
        reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, background="plane")


def test_scan_setup_tilts_reach_the_geometry():
    base = ScanSetup(0.71073, 38.0, 0.1353, (512.0, 384.0), 0.0, 1.0)
    assert base.geometry((1024, 768), 10).tz_deg == 0.0 and base.geometry((1024, 768), 10).ty_deg == 0.0         # default: no tilt, as before
    tilted = replace(base, tilts_deg=(0.1, -9.684, 9.684))
    g = tilted.geometry((1024, 768), 10)
    assert (g.tx_deg, g.ty_deg, g.tz_deg) == (0.1, -9.684, 9.684) and g.lsd_um == pytest.approx(38000.0)


def test_reduce_scan_accepts_precomputed_spots(scene):
    from midas_sx.pipeline import find_spots
    exp, st, res, raw, setup, U0 = scene
    geom0 = setup.geometry(raw.shape[1:], raw.shape[0])
    found = find_spots(raw, geom0)
    a = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False)
    b = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, found=found)
    assert len(a.spots) == len(b.spots) and np.allclose(a.merged.F2.to_numpy(), b.merged.F2.to_numpy())


def test_reduce_scan_margin_overrides_the_sweep(scene):
    exp, st, res, raw, setup, U0 = scene
    a = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, margin=0)
    b = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, margin=3)
    assert a.profile.margin == 0 and b.profile.margin == 3
    assert not np.allclose(a.obs.I.to_numpy()[: min(len(a.obs), len(b.obs))], b.obs.I.to_numpy()[: min(len(a.obs), len(b.obs))])
