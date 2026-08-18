"""Per-voxel deformation-gradient inversion (the 12-D problem).

These tests pin behaviour, not accuracy: the field inverse is ill-posed by
construction (missing cone of half-angle theta, plus 3-of-9 components per
reflection), so "recovers the planted field" is a *measured, regularisation-
dependent* claim that belongs in dev/paper/runs, not in an assertion. What is
asserted here is that the operator, the regulariser and the guards do what they
say -- including the ones whose absence would silently produce a plausible wrong
answer.
"""
import pytest
import torch

from midas_dct_tt import (PlaneDetector, attach_uniform_field,
                          cross_validate_lambda, fit_deformation_field, local_Q,
                          psi_scan, sphere_grain, smoothness_penalty,
                          topograph_stack, tt_alignment)
from midas_dct_tt.acceptance import tt_resolution_aniso

DT = torch.float64
LAM = 0.172979


@pytest.fixture(scope="module")
def rig():
    g = attach_uniform_field(sphere_grain(1.5, spacing_um=1.0))
    det = PlaneDetector(pixel_um=1.0, shape=(32, 32), distance_um=5000.0)
    hkls = [(2, 0, 0), (0, 2, 0)]
    als = [tt_alignment(g.field.reference_G(h), LAM) for h in hkls]
    res = [tt_resolution_aniso(a) for a in als]
    return g, det, hkls, als, res, psi_scan(6)


def _observe(rig, H, supersample=1):
    g, det, hkls, als, res, psi = rig
    eye = torch.eye(3, dtype=DT).expand_as(H)
    return [topograph_stack(g, a, psi, detector=det, hkl=h, resolution=r,
                            supersample=supersample,
                            Q_sample=local_Q(eye + H, g.field.reference_G(h)))
            for a, h, r in zip(als, hkls, res)]


# --- the regulariser -------------------------------------------------------
def test_smoothness_is_zero_for_a_uniform_field():
    """First differences, so the penalty prefers a UNIFORM F -- the honest null
    for a grain with no measured intragranular structure."""
    H = torch.ones(27, 3, 3, dtype=DT) * 3.7
    assert float(smoothness_penalty(H, (3, 3, 3))) == pytest.approx(0.0, abs=1e-24)


def test_smoothness_grows_with_roughness():
    torch.manual_seed(0)
    smooth = torch.linspace(0, 1, 27, dtype=DT).reshape(27, 1, 1) * torch.ones(3, 3, dtype=DT)
    rough = torch.randn(27, 3, 3, dtype=DT)
    assert float(smoothness_penalty(rough, (3, 3, 3))) > \
        float(smoothness_penalty(smooth, (3, 3, 3)))


def test_smoothness_is_differentiable():
    H = torch.randn(8, 3, 3, dtype=DT, requires_grad=True)
    smoothness_penalty(H, (2, 2, 2)).backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()


def test_smoothness_requires_a_grid_and_reports_why():
    H = torch.zeros(10, 3, 3, dtype=DT)
    with pytest.raises(ValueError, match="needs a regular grid"):
        smoothness_penalty(H, None)
    with pytest.raises(ValueError, match="does not match"):
        smoothness_penalty(H, (2, 2, 2))


def test_smoothness_handles_singleton_axes():
    H = torch.randn(4, 3, 3, dtype=DT)
    assert float(smoothness_penalty(H, (4, 1, 1))) > 0
    assert float(smoothness_penalty(H, (1, 1, 4))) > 0


# --- the fit ---------------------------------------------------------------
def test_recovers_shape_and_reduces_the_loss(rig):
    g = rig[0]
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(g.n_voxels, 3, 3, dtype=DT)
    obs = _observe(rig, H)
    F, info = fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                                    hkls=rig[2], resolutions=rig[4],
                                    lambda_smooth=1e-3, steps=12, lr=1e-4)
    assert F.shape == (g.n_voxels, 3, 3)
    assert info["history"][-1] < info["history"][0]
    assert info["lambda_smooth"] == 1e-3


def test_zero_deformation_data_keeps_F_at_identity(rig):
    """The null: featureless data must not manufacture a field."""
    g = rig[0]
    H0 = torch.zeros(g.n_voxels, 3, 3, dtype=DT)
    obs = _observe(rig, H0)
    F, _ = fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                                 hkls=rig[2], resolutions=rig[4],
                                 lambda_smooth=1e-2, steps=10, lr=1e-4)
    eye = torch.eye(3, dtype=DT).expand_as(F)
    assert float((F - eye).abs().max()) < 1e-5


def test_regularisation_suppresses_roughness(rig):
    """lambda is a PRIOR doing real work: more of it must give a smoother field.

    Pinned because a field inverse with a large null space will fill it with
    whatever the optimiser wandered to, and the only thing stopping that is this
    term.
    """
    g = rig[0]
    torch.manual_seed(1)
    H = 1e-3 * torch.randn(g.n_voxels, 3, 3, dtype=DT)
    obs = _observe(rig, H)
    rough = []
    for lam in (0.0, 1.0):
        F, _ = fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                                     hkls=rig[2], resolutions=rig[4],
                                     lambda_smooth=lam, steps=25, lr=1e-3)
        eye = torch.eye(3, dtype=DT).expand_as(F)
        rough.append(float(smoothness_penalty(F - eye, g.shape)))
    assert rough[1] < rough[0], f"lambda=1 not smoother than lambda=0: {rough}"


def test_scale_profiling_absorbs_a_flux_error(rig):
    """A per-reflection flux error must not become a deformation field.

    Without profiling this is the failure mode that faked 0.0265 deg of rotation
    from a 4.72% flux error alone.

    Measured as the **data residual at the true answer** (``H = 0``, ``lr = 0``),
    not as a fitted field. Fitting would measure the optimiser instead: Adam
    normalises by ``sqrt(v)``, so at a vanishing gradient it still takes steps of
    order ``lr`` and random-walks away from a perfectly correct starting point.
    """
    g = rig[0]
    H0 = torch.zeros(g.n_voxels, 3, 3, dtype=DT)
    obs = [o * 1.0472 for o in _observe(rig, H0)]
    kw = dict(detector=rig[1], hkls=rig[2], resolutions=rig[4],
              lambda_smooth=0.0, steps=1, lr=0.0)
    _, on = fit_deformation_field(obs, g, rig[3], rig[5], profile_scales=True, **kw)
    _, off = fit_deformation_field(obs, g, rig[3], rig[5], profile_scales=False, **kw)
    assert on["data"] < 1e-25, f"flux error not absorbed: {on['data']:.3e}"
    assert off["data"] > 1e-8, f"control is not exercising the confound: {off['data']:.3e}"


def test_psf_enters_the_model_not_only_the_data(rig):
    """If the data are blurred and the model is not, the fit solves a different
    operator -- so psf_px must change the prediction."""
    g = rig[0]
    torch.manual_seed(2)
    H = 5e-4 * torch.randn(g.n_voxels, 3, 3, dtype=DT)
    obs = _observe(rig, H)
    kw = dict(detector=rig[1], hkls=rig[2], resolutions=rig[4],
              lambda_smooth=1e-3, steps=6, lr=1e-4)
    _, a = fit_deformation_field(obs, g, rig[3], rig[5], psf_px=0.0, **kw)
    _, b = fit_deformation_field(obs, g, rig[3], rig[5], psf_px=1.2, **kw)
    assert a["data"] != pytest.approx(b["data"], rel=1e-6)


def test_mismatched_list_lengths_are_rejected(rig):
    g = rig[0]
    obs = _observe(rig, torch.zeros(g.n_voxels, 3, 3, dtype=DT))
    with pytest.raises(ValueError, match="same length"):
        fit_deformation_field(obs[:1], g, rig[3], rig[5], detector=rig[1],
                              hkls=rig[2], resolutions=rig[4], steps=1)


def test_grain_without_a_field_is_rejected(rig):
    """reference_G is undefined without a DeformationField, so fail loudly."""
    g = sphere_grain(1.5, spacing_um=1.0)          # no attach_uniform_field
    obs = _observe(rig, torch.zeros(rig[0].n_voxels, 3, 3, dtype=DT))
    with pytest.raises(ValueError, match="no DeformationField"):
        fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                              hkls=rig[2], resolutions=rig[4], steps=1)


def test_init_H_is_honoured(rig):
    g = rig[0]
    obs = _observe(rig, torch.zeros(g.n_voxels, 3, 3, dtype=DT))
    init = 1e-3 * torch.ones(g.n_voxels, 3, 3, dtype=DT)
    F, _ = fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                                 hkls=rig[2], resolutions=rig[4],
                                 init_H=init, lambda_smooth=0.0, steps=1, lr=0.0)
    eye = torch.eye(3, dtype=DT).expand_as(F)
    assert torch.allclose(F - eye, init, atol=1e-12)


# --- convergence -----------------------------------------------------------
# The 12-D campaign ran for months on an optimiser that returned an iterate
# 19-62% worse in loss than one the same run had already visited
# (dev/paper/runs/c2/verify_p10a/probe9.log). These pin the two guarantees that
# fixed it. The failure was silent -- every number downstream looked fine.
def _fit(rig, H, **kw):
    g = rig[0]
    obs = _observe(rig, H)
    return fit_deformation_field(obs, g, rig[3], rig[5], detector=rig[1],
                                 hkls=rig[2], resolutions=rig[4],
                                 lambda_smooth=1e-3, **kw)


def test_returned_iterate_is_never_worse_than_one_already_evaluated(rig):
    """THE bug. A returned point worse than one already scored is not a
    modelling choice, and a large lr makes Adam overshoot on purpose here."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, info = _fit(rig, H, steps=40, lr=3e-2, lr_schedule="none")
    assert info["returned_best"] is True
    assert info["loss"] <= min(info["history"]) + 1e-15
    assert info["loss"] <= info["final_loss"] + 1e-15


def test_the_overshoot_this_guards_against_is_real(rig):
    """If Adam never overshot at this lr the test above would pass vacuously.

    Asserts the *unscheduled* run genuinely ends above its own minimum, so the
    guard cannot silently stop guarding (the same pattern as the forbidden-
    reflection regression test in test_planning.py)."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, info = _fit(rig, H, steps=40, lr=3e-2, lr_schedule="none")
    assert info["final_over_min"] > 1e-3
    assert info["argmin"] < len(info["history"]) - 1


def test_cosine_schedule_removes_the_overshoot(rig):
    """Same problem, same lr, schedule on: the run must end at its own best.

    Two regimes, because they say different things. At the campaign's own lr the
    schedule must land the run exactly on its minimum. At a pathological lr it
    only has to help -- annealing cannot rescue an lr that is 100x too large for
    the first half of the run, and claiming otherwise would be the overclaim."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, sched = _fit(rig, H, steps=40, lr=3e-4, lr_schedule="cosine")
    assert sched["final_over_min"] <= 1e-3
    _, hot = _fit(rig, H, steps=40, lr=3e-2, lr_schedule="cosine")
    _, fixed = _fit(rig, H, steps=40, lr=3e-2, lr_schedule="none")
    assert hot["final_over_min"] < fixed["final_over_min"]


def test_settled_is_false_for_a_truncated_run(rig):
    """final_over_min alone is not enough: a run stopped while still descending
    has a perfectly smooth tail and a final == min."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, info = _fit(rig, H, steps=25, lr=1e-4, lr_schedule="none")
    assert info["final_over_min"] <= 1e-3      # not oscillating ...
    assert info["tail_improvement"] > 1e-3     # ... but still descending
    assert info["settled"] is False


def test_a_quiet_tail_is_not_evidence_of_a_stationary_point(rig):
    """The caveat behind `settled`, pinned on measured numbers.

    Annealing lr suppresses tail_improvement by ~40x on the SAME truncated
    problem (7.7e-2 -> 2.0e-3 at 150 steps) while leaving the iterate *further*
    from stationary -- grad_max_final rises 4.9e-5 -> 1.2e-4, because the schedule
    stopped it moving rather than brought it to rest. So a quiet tail is partly
    manufactured by the schedule, which is why `settled` is not called
    `converged` and why grad_max_final is reported next to it."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, fixed = _fit(rig, H, steps=150, lr=1e-4, lr_schedule="none")
    _, cos = _fit(rig, H, steps=150, lr=1e-4, lr_schedule="cosine")
    assert cos["tail_improvement"] < fixed["tail_improvement"] / 5.0
    assert cos["grad_max_final"] > fixed["grad_max_final"]


def test_info_reports_the_returned_points_own_loss(rig):
    """Old code evaluated the closure BEFORE opt.step(), so info['loss'] came
    from the previous iterate and the returned H had never been scored."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    F, info = _fit(rig, H, steps=30, lr=1e-3)
    obs = _observe(rig, H)
    eye = torch.eye(3, dtype=DT).expand_as(F)
    from midas_dct_tt.field_inverse import _model_stack, intensity_scales
    data = 0.0
    for a, h, r, o in zip(rig[3], rig[2], rig[4], obs):
        m = _model_stack(rig[0], F - eye, a, h, r, rig[5], rig[1],
                         supersample=1, psf_px=0.0, model="exact")
        s = intensity_scales(m.reshape(1, -1), o.reshape(1, -1)).reshape(())
        data += float((((m * s - o) / o.abs().amax().clamp_min(1e-30)) ** 2).mean())
    assert data == pytest.approx(info["data"], rel=1e-9)


def test_opting_out_restores_the_old_behaviour(rig):
    """The old path stays reachable so an archived result can be reproduced."""
    torch.manual_seed(0)
    H = 5e-4 * torch.randn(rig[0].n_voxels, 3, 3, dtype=DT)
    _, info = _fit(rig, H, steps=40, lr=3e-2, lr_schedule="none",
                   return_best=False)
    assert info["returned_best"] is False
    assert info["loss"] == info["final_loss"]


def test_bad_schedule_name_is_rejected(rig):
    torch.manual_seed(0)
    H = torch.zeros(rig[0].n_voxels, 3, 3, dtype=DT)
    with pytest.raises(ValueError, match="lr_schedule"):
        _fit(rig, H, steps=2, lr=1e-4, lr_schedule="linear")


# --- cross-validated lambda ------------------------------------------------
def test_cv_returns_a_table_over_the_requested_lambdas(rig):
    g = rig[0]
    torch.manual_seed(3)
    obs = _observe(rig, 5e-4 * torch.randn(g.n_voxels, 3, 3, dtype=DT))
    lams = [0.0, 1e-2]
    best, table = cross_validate_lambda(
        obs, g, rig[3], rig[5], detector=rig[1], hkls=rig[2], resolutions=rig[4],
        lambdas=lams, n_folds=2, steps=4, lr=1e-4)
    assert [r["lambda"] for r in table] == lams
    assert best in lams
    assert all({"lambda", "train", "test"} <= set(r) for r in table)
    assert best == min(table, key=lambda r: r["test"])["lambda"]


def test_cv_rejects_impossible_fold_counts(rig):
    g = rig[0]
    obs = _observe(rig, torch.zeros(g.n_voxels, 3, 3, dtype=DT))
    for nf in (1, 99):
        with pytest.raises(ValueError, match="n_folds must be in"):
            cross_validate_lambda(obs, g, rig[3], rig[5], detector=rig[1],
                                  hkls=rig[2], resolutions=rig[4],
                                  lambdas=[1e-2], n_folds=nf, steps=1)


def test_cv_folds_are_strided_not_contiguous(rig):
    """A TT scan is periodic in psi, so a contiguous held-out block is a MISSING
    WEDGE and would measure extrapolation across a gap rather than the quality of
    the regularisation. Strided folds keep it an interpolation test."""
    import inspect

    from midas_dct_tt import field_inverse
    src = inspect.getsource(field_inverse.cross_validate_lambda)
    assert "torch.arange(f, n, n_folds)" in src, "folds are no longer strided"


# --- discrepancy-principle lambda selection --------------------------------
_LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
_RES18 = [1.7138e-6, 1.6920e-6, 1.6954e-6, 1.8468e-6, 1.6906e-6,
          1.7444e-6, 1.8753e-6, 2.1078e-6, 2.5119e-6]
_PURE_FLOOR_18 = 1.6848e-6
_INFLATED_FLOOR_18 = 2.4014e-6      # includes supersample 3-vs-1 mismatch


def test_discrepancy_recovers_the_oracle_lambda():
    """Measured 18-view data; the oracle (max corr with the planted field) is 1e-2."""
    from midas_dct_tt import select_lambda_discrepancy
    assert select_lambda_discrepancy(_LAMS, _RES18, _PURE_FLOOR_18, tol=0.10) == 1e-2
    assert select_lambda_discrepancy(_LAMS, _RES18, _PURE_FLOOR_18, tol=0.05) == 1e-2


def test_an_inflated_floor_silently_picks_the_worst_lambda():
    """The failure that made this look useless: a floor contaminated by
    forward-model mismatch (36% here) makes every lambda 'consistent with noise',
    so the criterion returns the MOST over-smoothed candidate."""
    from midas_dct_tt import select_lambda_discrepancy
    bad = select_lambda_discrepancy(_LAMS, _RES18, _INFLATED_FLOOR_18, tol=0.05)
    assert bad == 3e-1, f"expected the over-smoothed end, got {bad}"
    assert bad != 1e-2


def test_discrepancy_rejects_the_oversmoothed_end():
    from midas_dct_tt import select_lambda_discrepancy
    picked = select_lambda_discrepancy(_LAMS, _RES18, _PURE_FLOOR_18, tol=0.10)
    i = _LAMS.index(picked)
    assert all(r > 1.10 * _PURE_FLOOR_18 for r in _RES18[i + 1:]), \
        "a larger lambda also fit to the noise; the pick is not the largest"


def test_discrepancy_input_guards():
    from midas_dct_tt import select_lambda_discrepancy
    with pytest.raises(ValueError, match="differ"):
        select_lambda_discrepancy([1.0, 2.0], [1e-6], 1e-6)
    with pytest.raises(ValueError, match="no candidates"):
        select_lambda_discrepancy([], [], 1e-6)
    with pytest.raises(ValueError, match="noise_floor must be > 0"):
        select_lambda_discrepancy([1.0], [1e-6], 0.0)
    with pytest.raises(ValueError, match="no lambda reaches the noise floor"):
        select_lambda_discrepancy([1.0], [1e-3], 1e-9)
