"""Opt-in robust merge estimator (merge_observations(robust=True)): bit-identical default, gross outliers, intensity-dependent sigma bias, clean data, n = 2."""
import numpy as np
import pandas as pd
import pytest

from midas_hkls import SpaceGroup
from midas_sx.merge import estimate_error_model_g, merge_observations, r_int

SG = SpaceGroup.from_hm("P 21 21 21")


def _old_merge(obs, sg, value="F2", sigma="sigF2"):
    """The merge as it was before the robust option (copied verbatim, the reference for bit-identity)."""
    rep = [sg.asu_representative(int(h), int(k), int(l)) for h, k, l in zip(obs["h"], obs["k"], obs["l"])]
    d = obs.copy()
    d[["uh", "uk", "ul"]] = np.array(rep)
    out = []
    for (uh, uk, ul), g in d.groupby(["uh", "uk", "ul"]):
        x = g[value].to_numpy(float)
        sx = np.maximum(g[sigma].to_numpy(float), 1e-12)
        w = 1.0 / sx ** 2
        mean = float((w * x).sum() / w.sum())
        sig_int = float(1.0 / np.sqrt(w.sum()))
        n = len(g)
        sig_ext = float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum() / (n - 1))) if n > 1 else sig_int
        out.append(dict(h=uh, k=uk, l=ul, n=n, **{value: mean, sigma: max(sig_int, sig_ext)}, sig_int=sig_int, sig_ext=sig_ext,
                        R_int_refl=float(np.abs(x - mean).sum() / np.abs(x).sum()) if n > 1 else np.nan))
    return pd.DataFrame(out)


def _table(true_f2, n_obs, rel_err=0.0, poisson=True, seed=0, scale=1.0):
    """n_obs equivalents (h, k, l sign variants) of each of len(true_f2) reflections; Poisson-like sigma = sqrt(counts) mapped to F2 units with ``scale``
    (counts = F2 / scale), plus an optional multiplicative error rel_err that the reported sigma does not know about."""
    rng = np.random.default_rng(seed)
    rows = []
    for r, mu in enumerate(true_f2):
        h, k, l = 1 + r % 5, 1 + (r // 5) % 5, 1 + r // 25
        for j in range(n_obs):
            cnt = mu / scale
            c = rng.poisson(cnt) if poisson else cnt
            x = c * scale * (1.0 + rel_err * rng.standard_normal())
            rows.append(dict(h=h * (1 if j % 2 == 0 else -1), k=k, l=l, F2=x, sigF2=max(np.sqrt(max(c, 1.0)) * scale, 1e-3)))
    return pd.DataFrame(rows)


def test_default_is_bit_identical():
    obs = _table(np.geomspace(10, 1e5, 60), 4, rel_err=0.1, seed=1)
    a, b = merge_observations(obs, SG), _old_merge(obs, SG)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert r_int(obs, SG) == r_int(obs, SG, merge_kwargs=None)


def test_failed_integration_does_not_dominate():
    """The A39 (0 0 8) case: three consistent observations and a failed one with a small sigma."""
    obs = pd.DataFrame(dict(h=[0, 0, 0, 0], k=[0, 0, 0, 0], l=[8, -8, 8, -8], F2=[4015.0, 4183.0, 2825.0, -1.19], sigF2=[74.0, 75.0, 26.0, 15.0]))
    old = float(merge_observations(obs, SG).F2.iloc[0])
    new = merge_observations(obs, SG, robust=True, error_g=0.1)
    assert old < 1500                                   # the old mean is dragged to the failed observation
    assert new.n.iloc[0] == 3 and new.n_rej.iloc[0] == 1
    assert 3300 < new.F2.iloc[0] < 3900


def test_intensity_dependent_sigma_bias_is_removed():
    """Real scatter (20 % multiplicative) larger than the reported sigma, and the reported sigma follows each observation's OWN value (sigma = sqrt(value * scale), as a
    counting error does): the 1/sigma^2 weighted mean is biased low (weights ~ 1/x, a harmonic mean). True mean 5000, n = 4, 3000 reflections."""
    rng = np.random.default_rng(2)
    rows = []
    for r in range(3000):
        for j in range(4):
            x = 5000.0 * (1.0 + 0.2 * rng.standard_normal())
            rows.append(dict(h=1 + r % 20 * (1 if j % 2 == 0 else -1), k=1 + (r // 20) % 20, l=1 + r // 400, F2=x, sigF2=np.sqrt(max(x, 1.0) * 5.0)))
    obs = pd.DataFrame(rows)
    bias_old = float(merge_observations(obs, SG).F2.mean() / 5000.0 - 1.0)
    new = merge_observations(obs, SG, robust=True)
    bias_new = float(new.F2.mean() / 5000.0 - 1.0)
    print("bias old", bias_old, "new", bias_new, "g", new.g.iloc[0])
    assert bias_old < -0.02
    assert abs(bias_new) < 0.3 * abs(bias_old)
    assert 0.1 < new.g.iloc[0] < 0.3


def test_gross_outliers_in_many_reflections():
    rng = np.random.default_rng(3)
    mu = np.full(200, 5000.0)
    obs = _table(mu, 5, rel_err=0.05, seed=4, scale=10.0)
    bad = rng.choice(len(obs), 40, replace=False)
    obs.loc[bad, "F2"] = rng.uniform(-50, 200, 40)      # failed integrations, sigma unchanged (small)
    err_old = float(np.abs(merge_observations(obs, SG).F2 / 5000.0 - 1).mean())
    err_new = float(np.abs(merge_observations(obs, SG, robust=True).F2 / 5000.0 - 1).mean())
    print("mean |rel err| old", err_old, "new", err_new)
    assert err_new < 0.5 * err_old


def test_clean_data_costs_nothing():
    obs = _table(np.geomspace(500, 1e5, 150), 4, seed=5, scale=5.0)       # sigma is right, no extra error
    old, new = merge_observations(obs, SG), merge_observations(obs, SG, robust=True)
    assert new.g.iloc[0] < 0.03 and new.n_rej.sum() <= 2
    rel = np.abs(new.F2.to_numpy() / old.F2.to_numpy() - 1)
    assert rel.mean() < 0.01
    assert abs(r_int(obs, SG, merge_kwargs=dict(robust=True)) - r_int(obs, SG)) < 0.01


def test_error_model_g_recovers_the_injected_term():
    for g0 in (0.08, 0.15):
        obs = _table(np.geomspace(2000, 5e5, 300), 4, rel_err=g0, seed=6, scale=20.0)
        g = estimate_error_model_g(obs, SG)
        assert abs(g - g0) < 0.03, (g0, g)


def test_two_observations_are_never_rejected_and_carry_the_scatter():
    obs = pd.DataFrame(dict(h=[1, -1], k=[2, 2], l=[3, 3], F2=[1000.0, 3000.0], sigF2=[20.0, 30.0]))
    m = merge_observations(obs, SG, robust=True, error_g=0.05)
    assert m.n.iloc[0] == 2 and m.n_rej.iloc[0] == 0
    assert m.sigF2.iloc[0] >= 0.5 * 2000.0 * 0.5          # the external scatter sets the merged sigma
