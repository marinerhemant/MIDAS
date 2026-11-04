"""Predicted crossings against the simulator's own truth, and the twin-pair targets (PREREGISTER A26)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Lattice, SpaceGroup                          # noqa: E402
from midas_sx.geometry import Scan, make_geometry                   # noqa: E402
from midas_sx.predict import TWIN_90_C, enumerate_hkl, predicted_crossings, twin_targets   # noqa: E402


@pytest.fixture(scope="module")
def small():
    exp = sim.make_experiment(det_n=256, centre_rc=(127.5, 127.5), omega_start=-38.0, n_frames=304, dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5,
                              t_minus_mm=3.0, bg0=40.0, bg1=0.0)
    st = sim.structure_cu()
    res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=20_000, max_pairs=4_000_000)
    scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
    geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc,
                         shape=(exp.det_n, exp.det_n), scan=scan)
    return exp, st, res, scan, geom


def test_enumerate_hkl_applies_the_resolution_limit_and_the_absences():
    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    sg = SpaceGroup.from_hm("F m -3 m")
    allh = enumerate_hkl(lat, 1.0)
    fcc = enumerate_hkl(lat, 1.0, sg)
    assert len(fcc) < len(allh) and not any(set(map(lambda v: v % 2, h)) == {0, 1} for h in map(tuple, fcc))   # mixed parity absent
    assert max(1.0 / lat.d_spacing(*map(int, h)) for h in fcc) <= 1.0 / 1.0 + 1e-9


def test_predicted_crossings_land_on_the_simulated_spots(small):
    exp, st, res, scan, geom = small
    lat = Lattice(*st.cell)
    t = res.truth[res.truth.complete]
    hkl = np.unique(t[["h", "k", "l"]].to_numpy(int), axis=0)
    pc = predicted_crossings(res.U, lat, hkl, geom, scan)
    assert len(pc) >= len(t) * 0.9
    m = t.merge(pc, on=["h", "k", "l"], suffixes=("_t", "_p"))
    # nearest predicted crossing in omega for each truth spot
    m["dom"] = (m.omega_t - m.omega_p).abs()
    best = m.loc[m.groupby(["h", "k", "l", "omega_t"]).dom.idxmin()]
    assert len(best) >= 0.9 * len(t)
    assert np.hypot(best.row_t - best.row_p, best.col_t - best.col_p).max() < 1.0     # simulator rounds hit positions to integer pixels (A19)
    assert best.dom.max() < 0.25


def test_twin_partner_sense_makes_the_two_components_coincide_for_a_equal_b():
    """U2 = U1 Rz(+90 deg) maps the first component onto the second at the SAME q for T h = (k, -h, l); Rz(-90) does not."""
    from scipy.spatial.transform import Rotation as R
    from midas_sx.assign import reciprocal_basis
    lat = Lattice(3.83, 3.83, 20.5, 90, 90, 90)
    B = reciprocal_basis(lat)
    U1 = R.random(random_state=np.random.RandomState(1)).as_matrix()
    h = np.array([[2, 1, 5], [3, -2, 7], [1, 4, 2]])
    g = h @ np.asarray(TWIN_90_C).T
    d = {s_: np.abs((U1 @ B @ h.T) - (U1 @ R.from_euler("z", s_ * 90.0, degrees=True).as_matrix() @ B @ g.T)).max() for s_ in (+1, -1)}
    assert d[+1] < 1e-12 and d[-1] > 0.1


def test_blend_and_resolved_targets_follow_the_split(small):
    exp, st, res, scan, geom = small
    lat_eq = Lattice(3.83, 3.83, 20.5, 90, 90, 90)
    lat_sp = Lattice(3.83, 3.83 * 1.03, 20.5, 90, 90, 90)
    from scipy.spatial.transform import Rotation as R
    U1 = res.U
    U2 = U1 @ R.from_euler("z", +90.0, degrees=True).as_matrix()
    hkl = enumerate_hkl(lat_eq, 1.6)
    eq = twin_targets(U1, U2, lat_eq, geom, scan, hkl)
    sp = twin_targets(U1, U2, lat_sp, geom, scan, hkl)
    # a = b: (almost) everything is a blend, separation ~ 0
    assert (eq.kind == "blend").mean() > 0.9 and eq.sep_px.dropna().max() < 0.05
    # a != b by 3 %: the split grows with the in-plane index, so some pairs resolve
    assert (sp.kind == "c1").sum() > 0 and sp.sep_px.dropna().max() > 1.0
    assert set(sp.kind) <= {"blend", "c1", "c2"} and (sp[sp.kind == "blend"][["f1", "f2"]] == 1).all().all()
