"""Friedel-pair geometry: the pair residual q_s,1 + q_s,2 vanishes for a true pair at the true geometry and the fit recovers tilts / wedge / centre from a displaced start.
The forward model here is the package's (predicted_crossings), so this is a code test; the independent-forward positive control is validation/real/friedel_sim.py."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from midas_hkls import Lattice
from midas_sx.friedel_geometry import fit_pairs, labelled_pairs, pair_residual, with_params
from midas_sx.geometry import Scan, make_geometry
from midas_sx.predict import enumerate_hkl, predicted_crossings
from midas_stress.orientation_sampling import uniform_rotations

LAT = Lattice(5.8033, 5.9595, 12.2978, 90, 90, 90)
SCAN = Scan(155.0, -0.2, 850)
TRUE = dict(tx=-0.10, ty=-0.19, tz=-20.07, wedge=0.41, bcz=861.9, bcy=401.3, lsd=171.14)


def truth_geom():
    g = make_geometry(wavelength_A=0.6889, dist_mm=170.0, pixel_mm=0.172, centre_rc=(862.0, 401.0), shape=(1679, 1475), scan=SCAN)
    return with_params(g, TRUE)


def synthetic_obs(U):
    g = truth_geom()
    P = predicted_crossings(U, LAT, enumerate_hkl(LAT, 0.8), g, SCAN)
    return g, pd.DataFrame(dict(h=P.h, k=P.k, l=P.l, row=P.row, col=P.col, omega_meas=P.omega, hkl_residual=0.0))


@pytest.fixture(scope="module")
def scene():
    U = uniform_rotations(7)[3]
    return synthetic_obs(U)


def test_true_pairs_have_zero_residual_at_the_true_geometry_and_a_U_independent_one(scene):
    g, obs = scene
    pairs = labelled_pairs(obs)
    assert len(pairs) > 20
    assert np.abs(pair_residual(g, pairs)).max() < 1e-6
    U2 = uniform_rotations(11)[8]                        # another crystal: the relation does not know U
    g2, obs2 = synthetic_obs(U2)
    assert np.abs(pair_residual(g2, labelled_pairs(obs2))).max() < 1e-6


def test_wrong_geometry_gives_a_residual_that_grows_with_tx_and_wedge(scene):
    g, obs = scene
    pairs = labelled_pairs(obs)
    r0 = np.abs(pair_residual(g, pairs)).max()
    rt = np.abs(pair_residual(with_params(g, dict(tx=TRUE["tx"] + 0.3)), pairs)).max()
    rw = np.abs(pair_residual(with_params(g, dict(wedge=TRUE["wedge"] + 0.3)), pairs)).max()
    assert rt > 100 * max(r0, 1e-7) and rw > 100 * max(r0, 1e-7)
    assert rt > 1e-3 and rw > 1e-3                       # 0.3 deg = 5e-3 rad


def test_fit_recovers_the_geometry_from_a_displaced_start(scene):
    g, obs = scene
    pairs = labelled_pairs(obs)
    start = with_params(g, dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.4, bcy=401.8, lsd=170.27))
    fit = fit_pairs(pairs, start)
    got = dict(zip(fit.names, fit.values))
    for k in ("tx", "ty", "tz", "wedge"):
        assert got[k] == pytest.approx(TRUE[k], abs=2e-3), (k, got)
    assert got["bcz"] == pytest.approx(TRUE["bcz"], abs=0.05) and got["bcy"] == pytest.approx(TRUE["bcy"], abs=0.05)
    assert fit.rms_final < 1e-5


def test_scrambled_mates_do_not_recover_it(scene):
    g, obs = scene
    pairs = labelled_pairs(obs)
    rng = np.random.default_rng(0)
    bad = pairs.copy()
    for c in ("row2", "col2", "om2"):
        bad[c] = rng.permutation(bad[c].to_numpy())
    start = with_params(g, dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.4, bcy=401.8, lsd=170.27))
    fit = fit_pairs(bad, start)
    assert fit.rms_final > 1e-3                          # no geometry makes random mates opposite


def test_wrong_labels_are_trimmed_and_do_not_move_the_estimate(scene):
    g, obs = scene
    rng = np.random.default_rng(3)
    o = obs.copy()
    bad = rng.random(len(o)) < 0.12                        # 12 % of the spots carry a random other spot's hkl
    o.loc[bad, ["h", "k", "l"]] = obs[["h", "k", "l"]].to_numpy()[rng.permutation(len(o))[: bad.sum()]]
    pairs = labelled_pairs(o)
    start = with_params(g, dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.4, bcy=401.8, lsd=170.27))
    fit = fit_pairs(pairs, start)
    got = dict(zip(fit.names, fit.values))
    assert fit.n_trimmed > 0
    assert got["tx"] == pytest.approx(TRUE["tx"], abs=5e-3) and got["wedge"] == pytest.approx(TRUE["wedge"], abs=5e-3)
    kept = fit_pairs(pairs, start, trim=None)              # without trimming the same data is visibly worse
    assert abs(dict(zip(kept.names, kept.values))["tx"] - TRUE["tx"]) > abs(got["tx"] - TRUE["tx"])


def test_multirun_shared_geometry_recovered_with_unknown_sample_frame_rotations():
    from midas_sx.friedel_geometry import fit_multirun
    from scipy.spatial.transform import Rotation as Rot
    g = truth_geom()
    U0 = uniform_rotations(7)[3]
    P1, P2 = Rot.from_rotvec([0.4, -1.1, 0.7]).as_matrix(), Rot.from_rotvec([-0.9, 0.2, 1.3]).as_matrix()      # arbitrary phi-stage rotations
    runs = [synthetic_obs(P @ U0)[1] for P in (np.eye(3), P1, P2)]
    start = with_params(g, dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.4, bcy=401.8, lsd=170.27))
    fit = fit_multirun(runs, start)
    got = dict(zip(fit.names, fit.values))
    for k in ("tx", "ty", "tz", "wedge"):
        assert got[k] == pytest.approx(TRUE[k], abs=2e-3), (k, got)
    assert got["bcz"] == pytest.approx(TRUE["bcz"], abs=0.05) and got["bcy"] == pytest.approx(TRUE["bcy"], abs=0.05)
    assert fit.n_cross_run_groups > 50 and fit.rms_final < 1e-5


def test_multirun_per_run_deviation_detects_a_wedge_that_differs_between_runs():
    from midas_sx.friedel_geometry import fit_multirun
    from scipy.spatial.transform import Rotation as Rot
    g = truth_geom()
    U0 = uniform_rotations(7)[3]
    Ps = [np.eye(3), Rot.from_rotvec([0.4, -1.1, 0.7]).as_matrix(), Rot.from_rotvec([-0.9, 0.2, 1.3]).as_matrix()]
    runs = []
    for P, dw in zip(Ps, (0.0, 0.0, -0.15)):                                      # run 2 has a 0.15 deg smaller wedge
        gk = with_params(g, dict(wedge=TRUE["wedge"] + dw))
        pc = predicted_crossings(P @ U0, LAT, enumerate_hkl(LAT, 0.8), gk, SCAN)
        runs.append(pd.DataFrame(dict(h=pc.h, k=pc.k, l=pc.l, row=pc.row, col=pc.col, omega_meas=pc.omega, hkl_residual=0.0)))
    start = with_params(g, dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.4, bcy=401.8, lsd=170.27))
    fit = fit_multirun(runs, start, per_run=("wedge",))
    assert fit.dev.shape == (2, 1)
    assert fit.dev[0, 0] == pytest.approx(0.0, abs=3e-3) and fit.dev[1, 0] == pytest.approx(-0.15, abs=3e-3)
    shared = fit_multirun(runs, start)                                              # forcing one wedge on runs that differ leaves a residual
    assert shared.rms_final > 20 * max(fit.rms_final, 1e-7)
