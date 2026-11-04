"""workflow.merge_runs: the validated multi-run entry point (smooth scale + robust merge by default)."""
import numpy as np
import pandas as pd
import pytest

from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups, scale_smooth
from midas_sx.workflow import MergedRuns, merge_runs

from test_scale_smooth import SG, _drift, _make


def _split(obs):
    return {r: g.drop(columns=["run"]) for r, g in obs.groupby("run")}


def test_defaults_are_smooth_scale_and_robust_merge():
    obs, _ = _make(_drift, seed=21)
    res = merge_runs(_split(obs), SG)
    assert isinstance(res, MergedRuns) and res.scale == "smooth" and res.robust
    assert "k_smooth" in res.obs and "g" in res.merged and "n_rej" in res.merged


def test_default_beats_group_scale_and_plain_merge_on_a_drifting_scan():
    obs, _ = _make(_drift, seed=22)
    new = merge_runs(_split(obs), SG)
    old = merge_runs(_split(obs), SG, scale="group", robust=False)
    assert new.r_int < 0.5 * old.r_int


def test_matches_the_primitives_step_by_step():
    obs, _ = _make(_drift, seed=23)
    res = merge_runs(obs, SG)
    s, _ = scale_smooth(obs, SG)
    m = merge_observations(s, SG, robust=True)
    assert np.allclose(res.merged.F2.to_numpy(), m.F2.to_numpy()) and np.isclose(res.r_int, r_int(s, SG, merge_kwargs=dict(robust=True)))


def test_scale_none_and_group_options():
    obs, _ = _make(lambda r, f: 1.0, seed=24)
    none = merge_runs(obs, SG, scale=None, robust=False)
    plain = merge_observations(obs, SG)
    assert np.allclose(none.merged.F2.to_numpy(), plain.F2.to_numpy()) and none.scales.empty
    g = merge_runs(obs, SG, scale="group", robust=False)
    assert "scale" in g.scales


def test_bad_arguments():
    obs, _ = _make(_drift, seed=25, n_unique=30)
    with pytest.raises(ValueError):
        merge_runs(obs, SG, scale="bogus")
    with pytest.raises(ValueError):
        merge_runs(obs.drop(columns=["run"]), SG)
    with pytest.raises(ValueError):
        merge_runs(obs.drop(columns=["frame"]), SG)
    with pytest.raises(ValueError):
        merge_runs({}, SG)


def test_single_run_few_observations_does_not_crash():
    obs, _ = _make(lambda r, f: 1.0, seed=26, n_unique=8, runs=("a",))
    res = merge_runs(obs, SG)
    assert np.isfinite(res.merged.F2).all()


def test_scale_smooth_with_no_equivalents_is_identity():
    obs = pd.DataFrame(dict(h=[1, 2, 3], k=[1, 1, 2], l=[1, 3, 4], run=["a", "a", "b"], frame=[10.0, 20.0, 30.0], F2=[100.0, 200.0, 50.0], sigF2=[5.0, 6.0, 3.0]))
    out, tab = scale_smooth(obs, SG)
    assert np.allclose(out.F2, obs.F2) and (tab.n_knots == 1).all()


def test_accepts_run_like_objects_with_an_obs_table():
    from types import SimpleNamespace
    obs, _ = _make(_drift, seed=31)
    runs = {r: SimpleNamespace(obs=g.drop(columns=["run"])) for r, g in obs.groupby("run")}
    res = merge_runs(runs, SG)
    ref = merge_runs(_split(obs), SG)
    assert np.allclose(res.merged.F2.to_numpy(), ref.merged.F2.to_numpy())
