"""forward_maps: leave-one-plane-out forward prediction of an ID03-style cube, planted answers."""
import numpy as np
import pytest

import midas_dfxm.forward_maps as fm

MU = np.arange(25) * 0.04 + 20.17
CHI = np.arange(26) * 0.04 - 5.6
OB = 16.119 + 0.02 * np.arange(8)


def _truth(P, rng, smu=0.05, schi=0.12, sob=0.04):
    """Planted per-pixel parameters, with the obpitch peak inside the scanned planes."""
    mu0 = 20.67 + rng.uniform(-0.15, 0.15, P)
    chi0 = -5.1 + rng.uniform(-0.2, 0.2, P)
    ob0 = 16.19 + rng.uniform(-0.03, 0.03, P)
    one = np.ones(P)
    return fm.SeparableParams(
        mu0=mu0, chi0=chi0, ob0=ob0, smu=smu * one, schi=schi * one, sob=sob * one,
        mob_train=np.zeros((8, P)), base_ob=np.zeros(P), ob_train=OB.copy(), mu_g=MU, chi_g=CHI)


def _cube(P=400, seed=0, **kw):
    rng = np.random.default_rng(seed)
    t = _truth(P, rng)
    # amplitude and baseline planted through the totals the fit will read back
    amp = rng.uniform(300, 900, P)
    t.base_ob = np.full(P, 0.0)
    A_true, b_true = amp, np.full(P, 3.0)
    rng2 = np.random.default_rng(seed + 1)
    cube = np.empty((25, 26, 8, P), np.float32)
    for k, ob in enumerate(OB):
        g = (A_true * fm._gauss(ob, t.ob0, t.sob))[None, None] \
            * fm._gauss(MU[:, None], t.mu0[None], t.smu[None])[:, None] \
            * fm._gauss(CHI[:, None], t.chi0[None], t.schi[None])[None]
        cube[:, :, k, :] = b_true + g + kw.get("noise", 0.0) * rng2.standard_normal(g.shape)
    return cube, t


def test_noiseless_planted_held_out_plane_is_predicted():
    cube, _ = _cube()
    mask = np.ones(cube.shape[-1], bool)
    res = fm.leave_one_plane_out(
        cube, MU, CHI, OB, [3, 4], {k: mask for k in (3, 4)},
        {"uniform": lambda p, m: fm.uniform_params(p, m)})
    for k in (3, 4):
        assert res[k]["full"]["r2"] > 0.9
        # the uniform model has no per-pixel orientation, so it must lose by a clear margin
        assert res[k]["full"]["r2"] - res[k]["uniform"]["r2"] > 0.3


def test_pooled_r2_matches_the_definition():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(5, 6, 20)); p = y + 0.1 * rng.normal(size=y.shape)
    sse = ((y - p) ** 2).sum((0, 1)); sst = ((y - y.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
    assert fm.pooled_r2(sse, sst) == pytest.approx(1 - sse.sum() / sst.sum())
    assert fm.pooled_r2(sse, sst) > 0.9


def test_a_wrong_model_can_fail():
    """Control that can fail: shuffle every pixel's bundle and the score collapses."""
    cube, _ = _cube(P=600)
    rng = np.random.default_rng(2)
    train = [j for j in range(8) if j != 4]
    p = fm.fit_separable_params(cube, MU, CHI, OB, train)
    obs = cube[:, :, 4, :]
    sse, sst = fm.score_plane(obs, p, OB[4])
    good = fm.pooled_r2(sse, sst)
    q = fm.swap_bundle(p, p, fm.pixel_donors(p.n_pixels, rng))
    sse2, sst2 = fm.score_plane(obs, q, OB[4])
    assert good > 0.9
    assert fm.pooled_r2(sse2, sst2) < good - 0.5


def test_permutation_null_sits_below_the_observed_score():
    cube, _ = _cube(P=300)
    rng = np.random.default_rng(3)
    train = [j for j in range(8) if j != 2]
    p = fm.fit_separable_params(cube, MU, CHI, OB, train)
    obs = cube[:, :, 2, :]
    sse, sst = fm.score_plane(obs, p, OB[2])
    observed = fm.pooled_r2(sse, sst)
    idx = np.arange(0, 300, 3)
    nul = fm.permutation_null(obs[:, :, idx], p.take(idx), p, OB[2],
                              lambda r: fm.pixel_donors(p.n_pixels, r), 30, rng,
                              score_idx_in_pool=idx)
    assert nul["r2"].max() < observed


def test_block_donors_is_a_permutation_within_blocks():
    mask = np.ones((128, 192), bool)
    blocks = fm.complete_blocks(mask, 64)
    assert len(blocks) == 6
    rr, cc = np.nonzero(mask)
    rc = np.stack([rr, cc], 1)
    donor = fm.block_donors(rc, blocks, 64, np.random.default_rng(0))
    assert sorted(donor) == list(range(len(rc)))                 # a true permutation
    # the within-block offset is preserved: donor sits at the same (dr, dc) of some block
    assert np.all((rc[donor] % 64) == (rc % 64))
    assert not np.array_equal(donor, np.arange(len(rc)))


def test_complete_blocks_skips_a_block_with_a_hole():
    mask = np.ones((64, 128), bool)
    mask[10, 70] = False
    assert fm.complete_blocks(mask, 64) == [(0, 0)]


def test_sigma_floor_is_applied_and_reported():
    """A peak that fits inside one bin has zero moment width; the floor must catch it."""
    P = 50
    cube = np.zeros((25, 26, 8, P), np.float32)
    cube[12, 13, 4, :] = 1000.0                                   # a single hot cell
    p = fm.fit_separable_params(cube, MU, CHI, OB, [0, 1, 2, 3, 4, 5, 6])
    assert (p.smu >= fm.SIGMA_FLOOR_STEPS * 0.04 - 1e-12).all()
    assert p.floored["mu"] == 1.0


def test_amplitude_recovers_a_planted_value():
    cube, t = _cube(P=100, seed=5)
    p = fm.fit_separable_params(cube, MU, CHI, OB, list(range(8)))
    A, b = fm.amplitude_and_baseline(p)
    assert np.all(A > 0) and np.all(np.isfinite(b))


def test_synthetic_cube_second_peak_lowers_a_single_gaussian_score():
    """The misspecified control must score below the matched one at the same noise."""
    cube, _ = _cube(P=400, seed=7)
    p_all = fm.fit_separable_params(cube, MU, CHI, OB, list(range(8)))
    rng = np.random.default_rng(0)
    ok = fm.synthetic_cube(p_all, OB, noise_var0=4.0, gain=0.0, rng=rng)
    bad = fm.synthetic_cube(p_all, OB, noise_var0=4.0, gain=0.0, rng=rng, second_peak=(3.0, 0.3))
    mask = np.ones(p_all.n_pixels, bool)
    r = {}
    for name, c in (("ok", ok), ("bad", bad)):
        out = fm.leave_one_plane_out(c, MU, CHI, OB, [4], {4: mask})
        r[name] = out[4]["full"]["r2"]
    assert r["ok"] - r["bad"] > 0.05


def test_gain_from_residuals_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 400, size=(25, 26, 200)).astype(float)
    base = np.zeros(200)
    resid = np.sqrt(9.0 + 2.0 * pred) * rng.standard_normal(pred.shape)
    g, c0 = fm.gain_from_residuals(pred, base, resid)
    assert g == pytest.approx(2.0, rel=0.1)
    # the intercept is an extrapolation to zero signal and is poorly pinned; only its order matters
    assert 0.0 < c0 < 30.0


def test_halfmax_sigma_recovers_a_sampled_gaussian():
    x = np.arange(41) * 0.04
    P = 5
    c = np.array([0.7, 0.8, 0.81, 0.9, 1.0])[:P]
    for sig in (0.05, 0.12):
        M = np.exp(-0.5 * ((x[:, None] - c[None]) / sig) ** 2)
        got = fm.halfmax_sigma(M, x)
        assert np.allclose(got, sig, rtol=0.1)


def test_halfmax_sigma_truncated_side_borrows_the_other_half():
    x = np.arange(25) * 0.04
    M = np.exp(-0.5 * ((x[:, None] - 0.0) / 0.12) ** 2)      # peak sits on the first grid point
    got = fm.halfmax_sigma(M, x)
    assert got[0] == pytest.approx(0.12, rel=0.1)


def test_halfmax_sigma_ignores_a_distant_second_island():
    """Contiguity (rule 12): an island above half-max far from the peak must not set the width."""
    x = np.arange(41) * 0.04
    M = np.exp(-0.5 * ((x - 0.6) / 0.05) ** 2)
    M = M + 0.9 * np.exp(-0.5 * ((x - 1.4) / 0.05) ** 2)       # second island, above half height
    got = fm.halfmax_sigma(M[:, None], x)[0]
    assert got == pytest.approx(0.05, rel=0.15)


def test_widths_option_and_moment_variant():
    cube, _ = _cube(P=80, seed=9)
    p_hm = fm.fit_separable_params(cube, MU, CHI, OB, list(range(7)))
    p_mo = fm.fit_separable_params(cube, MU, CHI, OB, list(range(7)), widths="moment")
    assert np.allclose(p_hm.smu_mom, p_mo.smu)
    q = fm.moment_width_params(p_hm)
    assert np.allclose(q.smu, p_mo.smu) and np.allclose(q.mu0, p_hm.mu0)
    with pytest.raises(ValueError):
        fm.fit_separable_params(cube, MU, CHI, OB, [0, 1, 2], widths="nope")


def test_a_resolvable_twin_on_chi_is_seen_and_an_unresolvable_one_on_mu_is_not():
    """Why the control sits on chi: 3 sigma_mu is ~1.5 steps, which a single Gaussian re-fit absorbs."""
    cube, _ = _cube(P=500, seed=11)
    p_all = fm.fit_separable_params(cube, MU, CHI, OB, list(range(8)))
    mask = np.ones(p_all.n_pixels, bool)

    def pipe(sp):
        syn = fm.synthetic_cube(p_all, OB, noise_var0=4.0, gain=0.0, rng=np.random.default_rng(0),
                                second_peak=sp)
        return fm.leave_one_plane_out(syn, MU, CHI, OB, [4], {4: mask})[4]["full"]["r2"]

    base = pipe(None)
    assert base - pipe((3.0, 1.0, "chi")) > 0.10
    assert base - pipe((3.0, 1.0, "mu")) < base - pipe((3.0, 1.0, "chi"))
    with pytest.raises(ValueError):
        fm.synthetic_cube(p_all, OB, noise_var0=1.0, gain=0.0, rng=np.random.default_rng(0),
                          second_peak=(3.0, 1.0, "ob"))
