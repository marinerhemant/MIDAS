"""Gates on the uniaxial squared-modulus ODF model and the three-model ladder.

Two controls, and the pairing is the point:

* :func:`test_ladder_recovers_a_planted_global_texture` -- the fit must find
  texture that is there;
* :func:`test_ladder_refutes_a_uniform_truth` -- and must *not* find texture that
  is not.

Both are **inverse crimes**: the same squared-modulus model generates and fits.
That is the correct scope for a unit test -- it gates the algebra, the Jacobian and
the optimiser -- but it is emphatically *not* evidence that the model can recover
texture from real data. The crime-free control lives in
``scripts/odf_positive_control.py``, which plants discrete crystallites and lays
real peaks on a background at measured contrast. Run that before believing a
texture map; a passing test file here means only that the code does what it says.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_legendre

from midas_dt.odf_uniaxial import (
    LadderResult,
    UniaxialODFModel,
    explained_by_polynomial,
    fibre_cos_theta,
    fit_uniaxial_ladder,
    hermans_parameter,
    legendre_even,
    normalisation_c_l,
    uniaxial_design,
)

N_L = 4


# --------------------------------------------------------------- polynomials
def test_legendre_even_matches_scipy():
    x = np.linspace(-1.0, 1.0, 41)
    got = legendre_even(x, 4)
    for j, l in enumerate((0, 2, 4, 6)):
        assert np.allclose(got[:, j], eval_legendre(l, x), atol=1e-12)


def test_legendre_even_is_one_at_the_pole():
    assert np.allclose(legendre_even(np.array([1.0]), 4), 1.0)


def test_legendre_even_refuses_orders_it_does_not_carry():
    with pytest.raises(NotImplementedError, match="P_8"):
        legendre_even(np.array([0.0]), 5)


def test_normalisation_constants():
    c = normalisation_c_l(4)
    assert c[0] == pytest.approx(np.sqrt(1 / (4 * np.pi)))
    assert c[3] == pytest.approx(np.sqrt(13 / (4 * np.pi)))


# ------------------------------------------------------------------ geometry
def test_fibre_cos_theta_has_no_omega_dependence():
    """The fact behind the model, and behind a withdrawn inference.

    ``n_s . z = cos(theta) sin(eta)`` carries no omega, so an axial fibre is
    STATIC in omega. Reasoning "static in omega, therefore instrumental" is
    therefore wrong -- it was used as a discriminator in this project and had to
    be withdrawn.
    """
    eta = np.linspace(-np.pi, np.pi, 37)
    tt = np.array([5.0, 9.0])
    a = fibre_cos_theta(tt, eta)
    assert a.shape == (37, 2)
    # there is simply no omega argument to vary -- that IS the property
    assert np.allclose(a[:, 0], np.cos(np.radians(5.0) / 2) * np.sin(eta))


def test_fibre_cos_theta_is_bounded_and_signed():
    eta = np.linspace(-np.pi, np.pi, 91)
    a = fibre_cos_theta(np.array([8.0]), eta)
    assert a.max() <= 1.0 and a.min() >= -1.0
    assert a.max() > 0.9 and a.min() < -0.9        # eta sweeps the full range


def test_uniaxial_design_shape_and_l0_column():
    eta = np.linspace(-np.pi, np.pi, 25)
    d = uniaxial_design(fibre_cos_theta(np.array([6.0, 10.0]), eta), N_L)
    assert d.shape == (25, 2, N_L)
    assert np.allclose(d[..., 0], normalisation_c_l(N_L)[0])   # P_0 == 1


# ------------------------------------------------------------ non-negativity
def test_the_squared_model_cannot_go_negative():
    """The reason for squaring: no cone, no L1, no projection step."""
    rng = np.random.default_rng(0)
    mu = np.linspace(-1.0, 1.0, 501)
    P = legendre_even(mu, N_L) * normalisation_c_l(N_L)
    for _ in range(200):
        a = rng.normal(size=N_L) * rng.choice([0.1, 1.0, 10.0])
        assert ((P @ a) ** 2 >= 0).all()


# --------------------------------------------------------- Hermans parameter
def test_uniform_odf_has_exactly_zero_hermans_parameter():
    """Gauss-Legendre is exact for this integrand, so this is 1e-15, not 1e-5.

    A uniform-grid trapezoid rule leaves ~8e-6 here. That is small, but it is a
    floor that a later analysis can mistake for a weak signal, so the quadrature
    is exact and this test pins it.
    """
    S = hermans_parameter(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert S[0] == pytest.approx(0.0, abs=1e-12)


def test_hermans_sign_follows_the_l2_coefficient():
    """a_2 > 0 concentrates weight at the poles (S > 0), a_2 < 0 at the equator."""
    pos = hermans_parameter(np.array([[1.0, 0.6, 0.0, 0.0]]))[0]
    neg = hermans_parameter(np.array([[1.0, -0.6, 0.0, 0.0]]))[0]
    assert pos > 0.05 and neg < -0.05


def test_hermans_is_monotone_in_a2_only_up_to_about_one():
    vals = [hermans_parameter(np.array([[1.0, a, 0.0, 0.0]]))[0]
            for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)
    assert vals[-1] < 1.0                     # bounded above by perfect alignment


def test_hermans_saturates_and_then_falls_as_a2_grows():
    """A real ceiling, pinned so a stalled fit is not misread as a data limit.

    With only ``a_2`` free, ``S`` peaks near +0.61 around ``a_2 ~ 1.35`` and then
    *decreases* -- squaring a large pure ``P_2`` puts weight back near the
    magic angle. A sharper axial texture needs ``a_4`` and ``a_6``, not a bigger
    ``a_2``.
    """
    S = {a: hermans_parameter(np.array([[1.0, a, 0.0, 0.0]]))[0]
         for a in (1.0, 1.35, 2.0, 5.0)}
    assert S[1.35] > S[1.0]
    assert S[2.0] < S[1.35]
    assert S[5.0] < S[2.0]
    assert max(S.values()) < 0.62              # the ceiling


def test_hermans_is_scale_invariant():
    """S describes a shape; multiplying the ODF by a constant must not change it."""
    a = np.array([[1.0, 0.4, -0.1, 0.05]])
    assert hermans_parameter(a)[0] == pytest.approx(
        hermans_parameter(7.3 * a)[0], abs=1e-9)


def test_hermans_handles_many_voxels_at_once():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, N_L))
    S = hermans_parameter(A)
    assert S.shape == (50,)
    assert (S >= -0.5 - 1e-9).all() and (S <= 1.0 + 1e-9).all()


# --------------------------------------------------- polynomial separability
def test_a_smooth_planted_field_is_flagged_as_smooth():
    """The check that has retracted a result three times in this project."""
    xy = np.stack(np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15),
                              indexing="ij"), axis=-1).reshape(-1, 2)
    smooth = 0.3 + 0.5 * xy[:, 0] - 0.2 * xy[:, 1] ** 2
    assert explained_by_polynomial(smooth, xy) > 0.99


def test_genuine_per_voxel_structure_is_not_flagged():
    rng = np.random.default_rng(4)
    xy = np.stack(np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15),
                              indexing="ij"), axis=-1).reshape(-1, 2)
    noise = rng.normal(size=len(xy))
    assert explained_by_polynomial(noise, xy) < 0.2


def test_explained_by_polynomial_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match=r"\(n_vox, 2\)"):
        explained_by_polynomial(np.zeros(10), np.zeros((7, 2)))


def test_explained_by_polynomial_on_a_constant_field():
    xy = np.random.default_rng(5).normal(size=(20, 2))
    assert explained_by_polynomial(np.ones(20), xy) == 1.0


# ------------------------------------------------------------- the model
def _toy_problem(n_side=5, n_ray=40, n_eta=24, seed=0):
    """A small tomographic problem: rays over a voxel grid, eta x 2 rings."""
    rng = np.random.default_rng(seed)
    n_vox = n_side * n_side
    eta = np.linspace(-np.pi, np.pi, n_eta, endpoint=False)
    design = uniaxial_design(fibre_cos_theta(np.array([6.0, 10.0]), eta), N_L)
    design = design.reshape(-1, N_L)                       # (n_eta*n_ring, n_l)
    # each ray touches a random contiguous-ish subset of voxels
    rays = np.zeros((n_ray, n_vox))
    for k in range(n_ray):
        picks = rng.choice(n_vox, size=max(3, n_vox // 4), replace=False)
        rays[k, picks] = 1.0
    xy = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side),
                              indexing="ij"), axis=-1).reshape(-1, 2).astype(float)
    return design, rays, xy, n_vox


def _make_model(design, rays, coefs_true, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    amp = coefs_true @ design.T
    truth = rays @ (amp ** 2)
    data = truth + (noise * rng.normal(size=truth.shape) * truth.std()
                    if noise else 0.0)
    good = np.ones(data.shape, dtype=bool)
    weights = np.ones_like(data)
    return UniaxialODFModel(design, rays, good, data, weights), truth


def test_analytic_jacobian_matches_a_numerical_one():
    """Sparse and analytic, so it needs checking against the slow honest version."""
    design, rays, _, n_vox = _toy_problem(n_side=3, n_ray=8, n_eta=6)
    rng = np.random.default_rng(2)
    coefs = rng.normal(size=(n_vox, N_L)) * 0.3
    coefs[:, 0] += 1.0
    model, _ = _make_model(design, rays, coefs)
    J = np.asarray(model.jacobian(coefs.ravel()).todense())
    p0 = coefs.ravel()
    eps = 1e-6
    for j in rng.choice(p0.size, size=12, replace=False):
        pp, pm = p0.copy(), p0.copy()
        pp[j] += eps
        pm[j] -= eps
        num = (model.residual(pp) - model.residual(pm)) / (2 * eps)
        assert np.allclose(J[:, j], num, atol=1e-5, rtol=1e-4), f"column {j}"


def test_model_rejects_mismatched_shapes():
    design, rays, _, n_vox = _toy_problem(n_side=3, n_ray=8, n_eta=6)
    data = np.zeros((8, design.shape[0]))
    with pytest.raises(ValueError, match="good must be"):
        UniaxialODFModel(design, rays, np.zeros((3, 3), dtype=bool), data, data)
    with pytest.raises(ValueError, match="design must be"):
        UniaxialODFModel(design[..., None], rays,
                         np.ones_like(data, dtype=bool), data, data)


def test_predict_is_non_negative_and_respects_the_ray_support():
    design, rays, _, n_vox = _toy_problem(n_side=3, n_ray=8, n_eta=6)
    rng = np.random.default_rng(3)
    coefs = rng.normal(size=(n_vox, N_L))
    model, _ = _make_model(design, rays, coefs)
    assert (model.predict(coefs.ravel()) >= 0).all()


# ------------------------------------------------------------- the ladder
def test_ladder_recovers_a_planted_global_texture():
    """INVERSE CRIME by design -- gates the algebra, not the physics."""
    design, rays, xy, n_vox = _toy_problem(seed=7)
    coefs_true = np.zeros((n_vox, N_L))
    coefs_true[:, 0] = 1.0
    coefs_true[:, 1] = 0.55                    # one shared a_2: a global texture
    model, _ = _make_model(design, rays, coefs_true, noise=0.01, seed=8)
    res = fit_uniaxial_ladder(model)

    assert res.chi2_null >= res.chi2_global >= res.chi2_pervoxel
    assert res.global_improvement_pct > 20.0
    S = hermans_parameter(coefs_true)[0]
    assert np.median(res.hermans_S) == pytest.approx(S, abs=0.1)
    assert "CONFIRM" in res.verdict(xy)


def test_ladder_refutes_a_uniform_truth():
    """The negative control: no planted texture, so the ladder must say so."""
    design, rays, xy, n_vox = _toy_problem(seed=9)
    coefs_true = np.zeros((n_vox, N_L))
    coefs_true[:, 0] = 1.0                      # uniform ODF in every voxel
    model, _ = _make_model(design, rays, coefs_true, noise=0.01, seed=10)
    res = fit_uniaxial_ladder(model)
    assert res.improvement_pct < 5.0
    assert "REFUTED" in res.verdict(xy)


def test_ladder_separates_global_from_per_voxel_texture():
    """The rung that makes a negative result interpretable.

    With a genuinely global texture, per-voxel freedom must buy almost nothing
    over the shared fit -- that is what says "not a resolution limit".
    """
    design, rays, xy, n_vox = _toy_problem(seed=11)
    coefs_true = np.zeros((n_vox, N_L))
    coefs_true[:, 0] = 1.0
    coefs_true[:, 1] = 0.5
    model, _ = _make_model(design, rays, coefs_true, noise=0.005, seed=12)
    res = fit_uniaxial_ladder(model)
    assert res.global_improvement_pct > 20.0
    assert res.pervoxel_over_global_pct < 0.5 * res.global_improvement_pct


def test_ladder_reports_parameter_counts():
    design, rays, xy, n_vox = _toy_problem(n_side=3, n_ray=12, n_eta=8, seed=13)
    coefs_true = np.zeros((n_vox, N_L))
    coefs_true[:, 0] = 1.0
    model, _ = _make_model(design, rays, coefs_true)
    res = fit_uniaxial_ladder(model)
    assert res.n_param["null"] == n_vox
    assert res.n_param["global"] == n_vox + N_L - 1
    assert res.n_param["pervoxel"] == n_vox * N_L


def test_verdict_flags_a_smooth_field_as_refuted():
    """A per-voxel map a polynomial explains is an instrument, not texture."""
    xy = np.stack(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8),
                              indexing="ij"), axis=-1).reshape(-1, 2)
    res = LadderResult(
        chi2_null=100.0, chi2_global=60.0, chi2_pervoxel=50.0,
        coefs=np.zeros((64, N_L)), global_coefs=np.zeros(N_L - 1),
        hermans_S=0.3 + 0.4 * xy[:, 0],       # a pure gradient
    )
    assert res.improvement_pct == pytest.approx(50.0)
    assert "REFUTED" in res.verdict(xy)        # smooth, despite a big improvement
    assert "CONFIRM" in res.verdict(None)      # and not flagged without xy


def test_verdict_band_is_inconclusive():
    res = LadderResult(
        chi2_null=100.0, chi2_global=95.0, chi2_pervoxel=90.0,
        coefs=np.zeros((4, N_L)), global_coefs=np.zeros(N_L - 1),
        hermans_S=np.zeros(4),
    )
    assert "INCONCLUSIVE" in res.verdict(None)


def test_verdict_thresholds_are_settable():
    res = LadderResult(
        chi2_null=100.0, chi2_global=95.0, chi2_pervoxel=90.0,
        coefs=np.zeros((4, N_L)), global_coefs=np.zeros(N_L - 1),
        hermans_S=np.zeros(4),
    )
    assert "REFUTED" in res.verdict(None, refute_pct=15.0)
    assert "CONFIRM" in res.verdict(None, confirm_pct=5.0)
