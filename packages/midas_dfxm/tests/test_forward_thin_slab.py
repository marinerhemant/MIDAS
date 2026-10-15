"""forward_thin_slab: the package's own physical forward on a one-voxel-thick slab, planted answers."""
import math

import numpy as np
import pytest

import midas_dfxm.forward_thin_slab as ts

LAM = 0.7293187971522731
TT0 = 16.19
MU = 20.17 + 0.04 * np.arange(25)
CHI = -5.6 + 0.04 * np.arange(26)
OB = 16.119 + 0.02 * np.arange(8)
SIG = dict(sigma_mu_deg=0.03, sigma_chi_deg=0.10, sigma_ob_deg=0.04)


def _slab(tmu, tchi, eps):
    return ts.thin_slab_from_maps(tmu, tchi, eps, two_theta0_deg=TT0, wavelength_A=LAM,
                                  mu_ref_deg=20.67, chi_ref_deg=-5.1)


def test_reference_reflection_sits_on_the_bragg_q_with_the_right_length():
    s = _slab([0.0], [0.0], [0.0])
    G0 = s.field.reference_G(s.hkl)
    assert float(np.linalg.norm(G0.numpy())) == pytest.approx(s.q_mag, rel=1e-9)
    # the Q of the beam deflected by 2 theta0 in the vertical plane
    res = ts.resolution_for_plane(s, TT0, **SIG)
    assert np.allclose(G0.numpy(), res.q_nom.numpy(), atol=1e-9)


def test_the_rendered_curves_return_the_input_maps():
    """The sign chain, fixed algebraically: render the maps, take the moments, recover the maps."""
    rng = np.random.default_rng(0)
    P = 6
    tmu = rng.uniform(-0.04, 0.04, P); tchi = rng.uniform(-0.12, 0.12, P)
    eps = rng.uniform(-3e-4, 3e-4, P)
    s = _slab(tmu, tchi, eps)
    W = ts.render_weights(s, MU, CHI, OB, **SIG)
    cm, cc, co = ts.recovered_centres(W, MU, CHI, OB)
    assert np.allclose(cm, 20.67 + tmu, atol=0.004)               # 0.1 of a 0.04 deg step
    assert np.allclose(cc, -5.1 + tchi, atol=0.004)
    th = math.radians(0.5 * TT0)
    ob_expect = TT0 - 2.0 * eps * math.tan(th) * 180.0 / math.pi
    assert np.allclose(co, ob_expect, atol=0.004)


def test_a_wrong_sign_is_caught_by_the_same_round_trip():
    """Control that can fail: flip the mu tilt and the recovered centre moves to the other side."""
    s = _slab([0.03], [0.0], [0.0])
    s_bad = _slab([-0.03], [0.0], [0.0])
    mu_g = 20.67 + 0.01 * np.arange(-12, 13)
    cm, _, _ = ts.recovered_centres(ts.render_weights(s, mu_g, [-5.1], [TT0], **SIG), mu_g, [-5.1], [TT0])
    cb, _, _ = ts.recovered_centres(ts.render_weights(s_bad, mu_g, [-5.1], [TT0], **SIG), mu_g, [-5.1], [TT0])
    assert cm[0] - 20.67 == pytest.approx(0.03, abs=3e-3)
    assert cb[0] - 20.67 == pytest.approx(-0.03, abs=3e-3)


def test_resolution_widths_become_the_requested_angular_widths():
    """At fixed obpitch the rendered rock and roll curves have the sigma that went in (units check)."""
    s = _slab([0.0], [0.0], [0.0])
    mu_g = 20.67 + 0.005 * np.arange(-30, 31)
    chi_g = -5.1 + 0.02 * np.arange(-25, 26)
    W = ts.render_weights(s, mu_g, chi_g, [TT0], **SIG).astype(np.float64)[:, :, 0, 0]
    sd = lambda m, g: math.sqrt(((m / m.sum()) * (g - (m / m.sum() * g).sum()) ** 2).sum())
    assert sd(W.sum(1), mu_g) == pytest.approx(SIG["sigma_mu_deg"], rel=0.12)
    assert sd(W.sum(0), chi_g) == pytest.approx(SIG["sigma_chi_deg"], rel=0.12)


def test_summing_over_obpitch_widens_the_mu_marginal_and_matching_undoes_it():
    """Obpitch moves the accepted vector partly along the rock direction: a real coupling the
    inversion's marginals contain. match_marginal_widths makes the rendered marginals equal the targets."""
    s = _slab([0.0], [0.0], [0.0])
    mu_g = 20.67 + 0.01 * np.arange(-25, 26)
    chi_g = -5.1 + 0.03 * np.arange(-20, 21)
    ob_g = TT0 + 0.01 * np.arange(-20, 21)
    W = ts.render_weights(s, mu_g, chi_g, ob_g, **SIG)
    raw = ts.marginal_sigmas(W, mu_g, chi_g, ob_g)[:, 0]
    assert raw[0] > 1.1 * SIG["sigma_mu_deg"]                       # coupling widens the mu marginal
    target = [0.03, 0.10, 0.04]
    sig, ach = ts.match_marginal_widths(s, mu_g, chi_g, ob_g, target)
    assert np.allclose(ach, target, rtol=0.03)
    assert sig[0] < target[0]                                       # the input had to shrink


def test_fit_amplitudes_recovers_a_planted_amplitude():
    s = _slab(np.zeros(4), np.zeros(4), np.zeros(4))
    W = ts.render_weights(s, MU, CHI, OB, **SIG)
    A_true = np.array([100.0, 250.0, 400.0, 900.0])
    Wt = W.sum((0, 1)).astype(float)                              # (n_ob, P)
    mob = Wt * A_true[None] + 7.0
    A = ts.fit_amplitudes(W, mob[:7], np.full(4, 7.0), list(range(7)))
    assert np.allclose(A, A_true, rtol=1e-6)


def test_flat_top_acceptance_has_the_requested_obpitch_width_and_edges():
    """Unit test (registered): at fixed mu and chi the rendered obpitch response is an erf box of the requested FWHM."""
    s = _slab([0.0], [0.0], [0.0])
    ob_g = TT0 + 0.002 * np.arange(-60, 61)
    for w_deg in (0.03, 0.05):
        W = ts.render_weights(s, [20.67], [-5.1], ob_g, sigma_mu_deg=0.03, sigma_chi_deg=0.10,
                              flat_top_ob_deg=(w_deg, 0.008))[0, 0, :, 0]
        W = W / W.max()
        # FWHM from linearly interpolated half-maximum crossings (the grid step must not set the answer)
        i0 = int(np.argmax(W)); lo = i0
        while W[lo] >= 0.5: lo -= 1
        hi = i0
        while W[hi] >= 0.5: hi += 1
        x_lo = ob_g[lo] + (0.5 - W[lo]) / (W[lo + 1] - W[lo]) * (ob_g[lo + 1] - ob_g[lo])
        x_hi = ob_g[hi - 1] + (W[hi - 1] - 0.5) / (W[hi - 1] - W[hi]) * (ob_g[hi] - ob_g[hi - 1])
        assert (x_hi - x_lo) == pytest.approx(w_deg, rel=0.05)
        assert W[np.argmin(np.abs(ob_g - TT0))] > 0.97               # flat top at the centre
        assert W[np.argmin(np.abs(ob_g - (TT0 + 1.6 * w_deg)))] < 0.02    # steep edge, nothing like a Gaussian tail


def test_the_default_acceptance_is_unchanged_by_the_new_option():
    s = _slab([0.01], [0.02], [1e-4])
    a = ts.render_weights(s, MU[:5], CHI[:5], OB[:3], **SIG)
    b = ts.render_weights(s, MU[:5], CHI[:5], OB[:3], **SIG, flat_top_ob_deg=None)
    assert np.array_equal(a, b)


def test_resolution_function_par_flat_is_an_erf_box_in_the_longitudinal_direction_only():
    import torch
    from midas_dfxm.resolution import ResolutionFunction
    q = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    r = ResolutionFunction(q_nom=q, sigma_par=1e-2, sigma_perp=1e-2, par_flat=(0.04, 0.004))
    e = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    t = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    w = lambda dpar, dt=0.0: float(r.weight((q + dpar * e + dt * t)[None])[0])
    assert w(0.0) == pytest.approx(1.0, abs=1e-6)
    assert w(0.02) == pytest.approx(0.5, abs=1e-3)               # half height at +-width/2
    assert w(-0.02) == pytest.approx(0.5, abs=1e-3)
    assert w(0.06) < 1e-4                                         # steep edge, not a Gaussian tail
    assert w(0.0, 0.01) == pytest.approx(np.exp(-0.5), rel=1e-6)  # transverse stays Gaussian
    g = ResolutionFunction(q_nom=q, sigma_par=1e-2, sigma_perp=1e-2)
    assert float(g.weight((q + 0.02 * e)[None])[0]) == pytest.approx(np.exp(-2.0), rel=1e-6)   # default untouched


def test_physical_score_predicts_a_held_out_plane_of_its_own_render():
    """Planted: a cube made by rendering a slab with known amplitudes is predicted by physical_score from the other 7 planes."""
    rng = np.random.default_rng(0)
    P = 40
    tmu = rng.uniform(-0.03, 0.03, P); tchi = rng.uniform(-0.1, 0.1, P)
    c = rng.uniform(2.5, 4.5, P)                                     # obpitch centre in steps from plane 0
    step = 0.02
    ob_c = OB[0] + c * step
    th = math.radians(OB[0] + 3 * step) / 2
    eps = -(np.deg2rad(ob_c - (OB[0] + 3.5 * step)) / 2) / np.tan(math.radians(TT0) / 2)
    ft = (0.032, 0.01)
    sl = ts.thin_slab_from_maps(tmu, tchi, eps, two_theta0_deg=TT0, wavelength_A=LAM, mu_ref_deg=20.67, chi_ref_deg=-5.1)
    Wt = ts.render_weights(sl, MU, CHI, OB, sigma_mu_deg=0.03, sigma_chi_deg=0.10, flat_top_ob_deg=ft)
    A = rng.uniform(300, 900, P)
    cube = (3.0 + A[None, None, None] * Wt).astype(np.float32)
    k = 4
    train = [j for j in range(8) if j != k]
    mob = cube[:, :, train, :].sum((0, 1), dtype=np.float64)           # (7, P)
    base = np.full(P, 3.0 * 25 * 26)
    refs = dict(mu_ref=20.67, chi_ref=-5.1, ob_ref=TT0, sigma_mu=0.03, sigma_chi=0.10, sigma_ob=0.0)
    # c_steps for the slab: the obpitch centre that gives the strain eps relative to TT0
    c_for = (TT0 + (-2 * eps * np.tan(math.radians(TT0) / 2) * 180 / math.pi) - OB[0]) / step
    sse, sst = ts.physical_score(cube[:, :, k, :], 20.67 + tmu, -5.1 + tchi, c_for, mob, base, OB, k, MU, CHI, refs, step, LAM,
                                 flat_top_ob_deg=ft)
    assert 1 - sse.sum() / sst.sum() > 0.97


def test_couple_false_removes_the_rock_shift_and_per_voxel_widths_are_honoured():
    s = _slab([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    ob_g = TT0 + 0.01 * np.arange(-12, 13)
    mu_g = 20.67 + 0.004 * np.arange(-40, 41)
    # coupled: at an obpitch off the reference the accepted vector has moved along the rock direction, so the mu peak of a voxel moves
    def mu_peak(couple):
        W = ts.render_weights(s, mu_g, [-5.1], [TT0 + 0.10], sigma_mu_deg=0.03, sigma_chi_deg=0.10,
                              flat_top_ob_deg=(0.4, 0.01), couple=couple)[:, 0, 0, 0]
        return mu_g[np.argmax(W)]
    assert abs(mu_peak(True) - 20.67) > 0.01                      # coupling shifts the rock position
    assert abs(mu_peak(False) - 20.67) < 0.005                    # switched off: it does not
    # per-voxel widths: two voxels with different rock widths get different rendered mu widths
    W2 = ts.render_weights(s, mu_g, [-5.1], [TT0], sigma_mu_deg=np.array([0.02, 0.05]), sigma_chi_deg=np.array([0.1, 0.1]),
                           flat_top_ob_deg=(0.04, 0.008))[:, 0, 0, :]
    fw = lambda y: (y / y.max() >= 0.5).sum()
    assert fw(W2[:, 1]) > 1.8 * fw(W2[:, 0])


def test_fractional_coupling_scales_the_rock_shift_linearly():
    """couple is a strength: the mu position of a voxel at an obpitch off the reference moves in proportion to it."""
    s = _slab([0.0], [0.0], [0.0])
    mu_g = 20.67 + 0.002 * np.arange(-100, 101)
    def mu_peak(c):
        W = ts.render_weights(s, mu_g, [-5.1], [TT0 + 0.10], sigma_mu_deg=0.03, sigma_chi_deg=0.10,
                              flat_top_ob_deg=(0.4, 0.01), couple=c)[:, 0, 0, 0]
        return mu_g[np.argmax(W)] - 20.67
    full, half, zero = mu_peak(True), mu_peak(0.5), mu_peak(False)
    assert abs(zero) < 0.003
    assert half == pytest.approx(0.5 * full, abs=0.004)
    assert mu_peak(1.0) == pytest.approx(full, abs=1e-9)


def test_physical_totals_match_the_totals_physical_score_fits_and_amplitude_override_is_used():
    rng = np.random.default_rng(1)
    P = 30
    tmu = rng.uniform(-0.03, 0.03, P); tchi = rng.uniform(-0.1, 0.1, P); c = rng.uniform(2.5, 4.5, P); step = 0.02
    ft = (0.032, 0.01)
    refs = dict(mu_ref=20.67, chi_ref=-5.1, ob_ref=TT0, sigma_mu=0.03, sigma_chi=0.10, sigma_ob=0.0)
    Wt = ts.physical_totals(20.67 + tmu, -5.1 + tchi, c, OB, MU, CHI, refs, step, LAM, flat_top_ob_deg=ft)
    assert Wt.shape == (8, P) and (Wt >= 0).all() and Wt.max() > 0
    # an explicit amplitude equal to the least-squares one reproduces the default score; a doubled one does not
    sl = ts.thin_slab_from_maps(tmu, tchi, -(np.deg2rad(OB[0] + c * step - TT0) / 2) / np.tan(math.radians(TT0) / 2),
                                two_theta0_deg=TT0, wavelength_A=LAM, mu_ref_deg=20.67, chi_ref_deg=-5.1)
    W = ts.render_weights(sl, MU, CHI, OB, sigma_mu_deg=0.03, sigma_chi_deg=0.10, flat_top_ob_deg=ft)
    A = rng.uniform(300, 900, P)
    cube = (3.0 + A[None, None, None] * W).astype(np.float32)
    k = 4; train = [j for j in range(8) if j != k]
    mob = cube[:, :, train, :].sum((0, 1), dtype=np.float64); base = np.full(P, 3.0 * 25 * 26)
    args = (cube[:, :, k, :], 20.67 + tmu, -5.1 + tchi, c, mob, base, OB, k, MU, CHI, refs, step, LAM)
    s0, t0 = ts.physical_score(*args, flat_top_ob_deg=ft)
    Wtr = Wt[train]; A_ls = np.maximum(((mob - base[None]) * Wtr).sum(0) / np.maximum((Wtr ** 2).sum(0), 1e-30), 0)
    s1, _ = ts.physical_score(*args, flat_top_ob_deg=ft, amplitude=A_ls)
    s2, _ = ts.physical_score(*args, flat_top_ob_deg=ft, amplitude=2 * A_ls)
    assert np.allclose(s0, s1, rtol=1e-6, atol=1e-6)
    assert s2.sum() > 5 * s1.sum()
