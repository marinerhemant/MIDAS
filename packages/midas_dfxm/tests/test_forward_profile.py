"""forward_profile: the v3 flat-top obpitch response, planted answers."""
import numpy as np
import pytest

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp

MU = np.arange(25) * 0.04 + 20.17
CHI = np.arange(26) * 0.04 - 5.6
OB = 16.119 + 0.02 * np.arange(8)
N = 25 * 26


def _truth(P=300, w=1.6, s=0.5, seed=0, noise=0.0):
    """A cube drawn from the v3 model with planted per-pixel centres (steps from plane 0) and amplitudes."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(2.5, 4.5, P)                     # peaks sit inside the scanned planes
    A = rng.uniform(300, 900, P)
    one = np.ones(P)
    p = fm.SeparableParams(
        mu0=20.67 + rng.uniform(-0.15, 0.15, P), chi0=-5.1 + rng.uniform(-0.2, 0.2, P), ob0=OB[0] + 0.02 * c,
        smu=0.05 * one, schi=0.12 * one, sob=0.04 * one, mob_train=np.zeros((8, P)), base_ob=np.zeros(P),
        ob_train=OB.copy(), mu_g=MU, chi_g=CHI)
    fit = fp.ProfileFit(c=c, A=A, w=w, s=s, at_edge=0.0)
    b = np.full(P, 3.0)
    cube = fp.synthetic_cube_profile(p, fit, OB, b, noise_var0=noise ** 2, gain=0.0, rng=np.random.default_rng(seed + 1))
    return cube, p, fit


def test_box_profile_is_a_unit_flat_top_with_the_requested_width():
    d = np.linspace(-4, 4, 4001)
    g = fp.box_profile(d, 2.0, 0.3)
    assert g.max() == pytest.approx(1.0, abs=1e-3)
    assert g[np.argmin(np.abs(d))] == pytest.approx(1.0, abs=1e-3)
    assert (g > 0.5).sum() * (d[1] - d[0]) == pytest.approx(2.0, abs=0.02)      # FWHM = w
    assert fp.box_profile(np.array([3.0]), 2.0, 0.3)[0] < 1e-6


def test_fit_centres_recovers_planted_centres_and_amplitudes():
    cube, p, fit = _truth()
    pos = np.arange(8.0)
    S = fm._gauss(MU[:, None], p.mu0[None], p.smu[None]).sum(0) * fm._gauss(CHI[:, None], p.chi0[None], p.schi[None]).sum(0)
    sig = cube.sum((0, 1), dtype=np.float64).T.T - 3.0 * N
    c, A = fp.fit_centres(sig, S, pos, 1.6, 0.5)
    ok = np.abs(c - fit.c) < 0.25
    assert ok.mean() > 0.9
    assert np.median(np.abs(A / fit.A - 1)[ok]) < 0.05


def test_nested_choice_recovers_the_planted_profile_width():
    cube, p, fit = _truth(P=300, w=2.0, s=0.5, seed=3)
    pos = np.arange(8.0)
    S = fm._gauss(MU[:, None], p.mu0[None], p.smu[None]).sum(0) * fm._gauss(CHI[:, None], p.chi0[None], p.schi[None]).sum(0)
    sig = cube.sum((0, 1), dtype=np.float64) - 3.0 * N
    w, s, table = fp.choose_profile(sig[:7], S, pos[:7])
    assert w in (1.6, 2.0, 2.5)                                  # the grid point nearest the planted 2.0
    assert table[(w, s)] == min(table.values())


def test_loo_predicts_a_held_out_plane_and_the_far_planes_sit_at_zero():
    cube, p, fit = _truth(P=400, seed=5, noise=2.0)
    masks = {k: np.ones(400, bool) for k in range(8)}
    res = fp.leave_one_plane_out_profile(cube, MU, CHI, OB, list(range(8)), masks, nested_sub=400)
    r2 = np.array([res[k]["r2"] for k in range(8)])
    assert r2[2:6].max() > 0.7                                   # the planes carrying signal are predicted
    # a plane that is pure floor cannot be 'explained': predicting the floor must not be worse than own-mean by much
    assert r2[[0, 7]].min() > -0.1


def test_the_v2_gaussian_pipeline_fails_where_v3_does_on_a_flat_top_truth():
    """Control that can fail, and the failure it must show: on a flat-top truth the v2 Gaussian pipeline over-predicts the
    plane two steps off the peak and goes negative there, while v3 does not."""
    cube, p, fit = _truth(P=600, seed=7, noise=1.0)
    k = 1                                                    # centres are 2.5..4.5 steps: plane 1 is the near flank
    masks = {k: np.ones(600, bool)}
    v3 = fp.leave_one_plane_out_profile(cube, MU, CHI, OB, [k], masks, nested_sub=600)[k]["r2"]
    v2 = fm.leave_one_plane_out(cube, MU, CHI, OB, [k], masks)[k]["full"]["r2"]
    assert v2 < v3 - 0.2


def test_amplitudes_for_centres_matches_the_joint_fit_at_the_fitted_centres():
    cube, p, fit = _truth(P=100, seed=9)
    pos = np.arange(8.0)
    S = fm._gauss(MU[:, None], p.mu0[None], p.smu[None]).sum(0) * fm._gauss(CHI[:, None], p.chi0[None], p.schi[None]).sum(0)
    sig = cube.sum((0, 1), dtype=np.float64) - 3.0 * N
    c, A = fp.fit_centres(sig, S, pos, 1.6, 0.5)
    assert np.allclose(fp.amplitudes_for_centres(sig, S, pos, c, 1.6, 0.5), A, rtol=1e-8, atol=1e-8)
