"""scale_smooth: recovery of an injected smooth per-run drift, no-drift control, placebo, held-out reflections."""
import numpy as np
import pandas as pd
import pytest

from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups, scale_smooth

SG = SpaceGroup.from_hm("P 21 21 21")


def _make(drift, seed=0, n_unique=250, noise=0.03, runs=("a", "b", "c"), n_frames=850):
    """Observations of mmm-equivalent reflections in several runs; F2 = true * k(run, frame) * (1 + noise)."""
    rng = np.random.default_rng(seed)
    grid = np.array([(a, b, c) for a in range(1, 12) for b in range(1, 12) for c in range(1, 12)])
    base = grid[rng.choice(len(grid), n_unique, replace=False)]
    true = 10 ** rng.uniform(1.0, 4.0, n_unique)
    rows = []
    for u in range(n_unique):
        for _ in range(rng.integers(3, 8)):
            r = runs[rng.integers(len(runs))]
            sgn = rng.choice([-1, 1], 3)
            fr = rng.uniform(0, n_frames)
            kt = drift(r, fr)
            f2 = true[u] * kt * (1 + noise * rng.standard_normal())
            rows.append(dict(h=int(base[u, 0] * sgn[0]), k=int(base[u, 1] * sgn[1]), l=int(base[u, 2] * sgn[2]), run=r, frame=fr, F2=f2, sigF2=noise * f2, u=u, kt=kt))
    return pd.DataFrame(rows), true


def _drift(r, fr):
    ph = {"a": 0.0, "b": 1.3, "c": 2.1}[r]
    return float(np.exp(0.22 * np.sin(2 * np.pi * fr / 850.0 * 1.5 + ph) + {"a": 0.0, "b": 0.1, "c": -0.15}[r]))


def _logk_err(out, obs):
    """Compare the recovered correction with the injected drift: k_smooth should equal 1 / kt up to one global constant."""
    resid = np.log(out.k_smooth.to_numpy()) + np.log(obs.kt.to_numpy())
    return resid - resid.mean()


def test_recovers_injected_drift():
    obs, _ = _make(_drift)
    out, tab = scale_smooth(obs, SG)
    err = _logk_err(out, obs)
    assert np.abs(err).max() < 0.06 and err.std() < 0.025
    assert (tab.n_knots > 1).all()


def test_beats_plain_group_scale_on_rint():
    obs, _ = _make(_drift)
    g, _ = scale_groups(obs, SG)
    s, _ = scale_smooth(obs, SG)
    assert r_int(s, SG) < 0.5 * r_int(g, SG)


def test_no_drift_control_does_not_invent_a_scale():
    obs, _ = _make(lambda r, f: 1.0, seed=3)
    out, tab = scale_smooth(obs, SG)
    assert tab.log_k_range.max() < 0.06
    assert abs(r_int(out, SG) - r_int(obs, SG)) < 0.01


def test_placebo_permuted_frames_gain_little():
    obs, _ = _make(_drift, seed=5)
    rng = np.random.default_rng(9)
    pl = obs.copy()
    for r in pl.run.unique():
        m = (pl.run == r).to_numpy()
        pl.loc[m, "frame"] = rng.permutation(pl.loc[m, "frame"].to_numpy())
    g, _ = scale_groups(obs, SG)
    out, _ = scale_smooth(pl, SG)
    real, _ = scale_smooth(obs, SG)
    assert r_int(real, SG) < 0.5 * r_int(out, SG)                  # the real frame order is what removes the scatter
    assert r_int(out, SG) > 0.7 * r_int(g, SG)                     # permuted labels must not do much better than the plain scale


def test_fit_mask_held_out_reflections_improve():
    obs, _ = _make(_drift, seed=7, n_unique=400)
    key = pd.factorize(pd.Series([tuple(SG.asu_representative(int(a), int(b), int(c))) for a, b, c in obs[["h", "k", "l"]].to_numpy()]))[0]
    half = key % 2 == 0
    out, _ = scale_smooth(obs, SG, fit_mask=half)
    g, _ = scale_groups(obs, SG)
    held = obs[~half].index
    assert r_int(out.loc[held], SG) < 0.6 * r_int(g.loc[held], SG)


def test_scale_applies_to_unshared_and_nonpositive_rows():
    obs, _ = _make(_drift, seed=11, n_unique=60)
    extra = pd.DataFrame([dict(h=11, k=12, l=13, run="a", frame=400.0, F2=-5.0, sigF2=2.0, u=-1, kt=1.0)])
    o = pd.concat([obs, extra], ignore_index=True)
    out, _ = scale_smooth(o, SG)
    assert np.isfinite(out.F2).all() and out.k_smooth.iloc[-1] > 0 and np.sign(out.F2.iloc[-1]) == -1


def test_single_group_with_few_observations_falls_back_to_one_knot():
    obs, _ = _make(lambda r, f: 1.0, seed=13, n_unique=12, runs=("a",))
    out, tab = scale_smooth(obs, SG, min_obs_per_knot=50)
    assert tab.n_knots.iloc[0] == 1


def test_scale_groups_unchanged():
    obs, _ = _make(_drift, seed=2, n_unique=40)
    a, ta = scale_groups(obs, SG)
    b, _ = scale_groups(obs, SG)
    assert np.array_equal(a.F2.to_numpy(), b.F2.to_numpy())
    assert abs(float(np.exp(np.mean(np.log(ta.scale)))) - 1.0) < 1e-6
