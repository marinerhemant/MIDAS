"""Reduction against a small independent simulation (validation/simulate.py): no front end, spots injected."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                           # noqa: E402

from midas_hkls import Lattice, SpaceGroup                       # noqa: E402
from midas_sx.geometry import Scan, make_geometry                # noqa: E402
from midas_sx.integrate import ProfileModel                      # noqa: E402
from midas_sx.reduce import (REJECT_REASONS, DacModel, PolarizationModel, Switches,  # noqa: E402
                             reduce_spots)


@pytest.fixture(scope="module")
def small():
    exp = sim.make_experiment(det_n=256, centre_rc=(127.5, 127.5), omega_start=-38.0, n_frames=304,
                              dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
    st = sim.structure_cu()
    res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=40_000, max_pairs=8_000_000)
    raw = res.frames(seed=5)
    t = res.truth
    spots = pd.DataFrame(dict(frame=(t.omega - exp.omega_start) / exp.omega_step - 0.5, row=t.row, col=t.col,
                              integrated=np.nan))
    scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
    geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm,
                         centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
    return exp, st, res, raw, spots, scan, geom


def _models(exp):
    dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
                   mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
    return dac, PolarizationModel(fraction=exp.pol_fraction), ProfileModel(1.0, 0.6, 0.1, k=3.0)


def _rel_error(obs, st):
    sg = SpaceGroup.from_hm(st.spacegroup)
    ft = sim.f2_truth(st, np.array([sg.asu_representative(int(h), int(k), int(l)) for h, k, l in zip(obs.h, obs.k, obs.l)]))
    x = obs.F2.to_numpy()
    k = (x * ft).sum() / (x * x).sum()
    return np.abs(k * x / ft - 1.0), k


def test_accounting_and_f2_recovery(small):
    exp, st, res, raw, spots, scan, geom = small
    dac, pol, prof = _models(exp)
    obs, rej, counts = reduce_spots(spots, raw, geom, scan, Lattice(*st.cell), res.U, prof, dac=dac, pol=pol)
    assert counts["input"] == len(spots) == counts["kept"] + sum(counts[r] for r in REJECT_REASONS)
    assert len(rej) == sum(counts[r] for r in REJECT_REASONS) and (rej.empty or set(rej.reason) <= set(REJECT_REASONS))
    assert counts["kept"] >= 5
    strong = obs[obs.I / obs.sigI > 20]
    err, _ = _rel_error(strong, st)
    assert len(strong) >= 4 and np.median(err) < 0.04 and err.max() < 0.12


def test_each_correction_matters_here_and_assignment_catches_a_wrong_omega_sign(small):
    exp, st, res, raw, spots, scan, geom = small
    dac, pol, prof = _models(exp)
    lat = Lattice(*st.cell)
    full, _, _ = reduce_spots(spots, raw, geom, scan, lat, res.U, prof, dac=dac, pol=pol)
    none, _, _ = reduce_spots(spots, raw, geom, scan, lat, res.U, prof, dac=dac, pol=pol, switches=Switches(False, False, False))
    s_full = full[full.I / full.sigI > 20]
    s_none = none[none.I / none.sigI > 20]
    assert np.median(_rel_error(s_none, st)[0]) > 5 * np.median(_rel_error(s_full, st)[0])
    # positive control: mirrored omega -> hkl assignment must collapse
    bad = Scan(-exp.omega_end, -exp.omega_step, exp.n_frames)
    obs_bad, _, c_bad = reduce_spots(spots, raw, geom, bad, lat, res.U, prof, dac=dac, pol=pol)
    assert c_bad["no_hkl"] >= 0.6 * len(spots) and len(obs_bad) <= 0.3 * len(full)


def test_a_beamstop_artefact_is_rejected_not_assigned_to_000(small):
    exp, st, res, raw, spots, scan, geom = small
    dac, pol, prof = _models(exp)
    fake = pd.concat([spots, pd.DataFrame(dict(frame=[100.0], row=[127.5], col=[127.5], integrated=[np.nan]))],
                     ignore_index=True)
    obs, rej, counts = reduce_spots(fake, raw, geom, scan, Lattice(*st.cell), res.U, prof, dac=dac, pol=pol)
    assert counts["low_angle"] >= 1 and not ((obs.h == 0) & (obs.k == 0) & (obs.l == 0)).any()


def test_corrections_at_the_calculated_crossing_do_not_depend_on_where_the_centroid_is(small):
    """A skewed spot has a biased centroid; L, P, T must follow the calculated reflection, not the centroid (PREREGISTER A15)."""
    exp, st, res, raw, spots, scan, geom = small
    dac, pol, prof = _models(exp)
    lat = Lattice(*st.cell)
    shifted = spots.copy()
    shifted["frame"] = shifted["frame"] + 0.35                                    # 0.09 deg centroid bias
    kw = dict(dac=dac, pol=pol)
    calc0, _, _ = reduce_spots(spots, raw, geom, scan, lat, res.U, prof, **kw)
    calc1, _, _ = reduce_spots(shifted, raw, geom, scan, lat, res.U, prof, **kw)
    meas0, _, _ = reduce_spots(spots, raw, geom, scan, lat, res.U, prof, eval_at="measured", **kw)
    meas1, _, _ = reduce_spots(shifted, raw, geom, scan, lat, res.U, prof, eval_at="measured", **kw)
    common = sorted(set(map(tuple, calc0[["h", "k", "l"]].to_numpy())) & set(map(tuple, calc1[["h", "k", "l"]].to_numpy()))
                    & set(map(tuple, meas0[["h", "k", "l"]].to_numpy())) & set(map(tuple, meas1[["h", "k", "l"]].to_numpy())))
    assert len(common) >= 5
    def corr(d):
        d = d.drop_duplicates(["h", "k", "l"]).set_index(["h", "k", "l"])
        return (d.L * d.PT).loc[common].to_numpy()
    shift_calc = np.max(np.abs(corr(calc1) / corr(calc0) - 1.0))
    shift_meas = np.max(np.abs(corr(meas1) / corr(meas0) - 1.0))
    assert shift_calc < 1e-9                                                       # the calculated crossing ignores the centroid
    assert shift_meas > 1e-4                                                       # the measured-direction version does (5e-4 for a 0.09 deg bias here)


def test_unknown_eval_at_is_refused(small):
    exp, st, res, raw, spots, scan, geom = small
    dac, pol, prof = _models(exp)
    with pytest.raises(ValueError):
        reduce_spots(spots, raw, geom, scan, Lattice(*st.cell), res.U, prof, dac=dac, pol=pol, eval_at="nowhere")
