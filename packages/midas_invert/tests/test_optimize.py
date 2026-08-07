"""Convergence guarantees of the shared gradient-fitting loop.

``midas_invert.fit`` used to run Adam at a fixed rate and hand back whatever the
final step produced.  Measured in the sister package on its 12-D campaign
(``midas_dct_tt/dev/paper/runs/c2/verify_p10a/probe9.log``) the returned iterate
was 19-62% worse in loss than one the same run had already visited, argmin as
early as step 207 of 600.  The reconstructions moved <1%, so nothing looked
wrong -- the damage was in **model selection**, because the argmin moves with the
hyperparameter and every candidate was therefore scored at a different, arbitrary
degree of convergence.

These pin the two guarantees that fix it, on an objective with the same
structural cause: an L1 kink, so the gradient does not vanish at the optimum and
Adam's scale-free step keeps moving at ~lr forever.  This is not a contrived
shape -- sparsity and TV priors are what the consumers of this loop
(``mixture_deconvolution``, ``midas_dct_tt.reconstruct_differentiable``) actually
minimise.
"""
import math

import pytest
import torch

from midas_invert import fit

DT = torch.float64
LAM = 0.2


@pytest.fixture
def lasso():
    """``mean((A x - b)^2) + LAM |x|_1`` -- returns ``(loss_of, fit_it)``.

    Seeded through an explicit Generator rather than the global RNG so the
    measured numbers below hold whatever else a test session has seeded.
    """
    g = torch.Generator().manual_seed(0)
    A = torch.randn(40, 20, generator=g, dtype=DT)
    x_true = torch.zeros(20, dtype=DT)
    x_true[[2, 7, 13]] = torch.tensor([2.0, -1.5, 1.0], dtype=DT)
    b = A @ x_true + 0.05 * torch.randn(40, generator=g, dtype=DT)

    def loss_of(x):
        return ((A @ x - b) ** 2).mean() + LAM * x.abs().sum()

    def fit_it(**kw):
        x = torch.zeros(20, dtype=DT, requires_grad=True)
        return x, fit([x], lambda: loss_of(x), **kw)

    return loss_of, fit_it


# --- convergence -----------------------------------------------------------
def test_returned_iterate_is_never_worse_than_one_already_evaluated(lasso):
    """THE bug.  A returned point worse than one already scored is not a
    modelling choice, and a fixed rate makes Adam overshoot on purpose here."""
    _, fit_it = lasso
    _, info = fit_it(steps=200, lr=0.05, lr_schedule="none")
    assert info["returned_best"] is True
    assert info["loss"] <= info["loss_min"] + 1e-15
    assert info["loss"] <= info["final_loss"] + 1e-15


def test_the_overshoot_this_guards_against_is_real(lasso):
    """Anti-vacuity.  If Adam never overshot here the test above would pass
    without the fix, and the guard could silently stop guarding.

    Measured: argmin at step 84 of 200, final_over_min 2.07e-2 -- the last
    iterate is 2.1% worse in loss than one the run visited 116 steps earlier,
    and tail_improvement is *negative* (-1.1e-2), i.e. it is climbing.
    """
    _, fit_it = lasso
    _, info = fit_it(steps=200, lr=0.05, lr_schedule="none")
    assert info["final_over_min"] > 1e-3, info["final_over_min"]
    assert info["argmin"] < 200            # 201 scored points, 0..200; not the last
    assert info["argmin"] == 84
    assert info["final_over_min"] == pytest.approx(2.07e-2, rel=0.05)
    assert info["tail_improvement"] < 0.0   # climbing over the last 10%


def test_cosine_schedule_removes_the_overshoot(lasso):
    """Same problem, same lr, schedule on: the run ends on its own best.

    It is also a better point, not merely a tidier one -- 8.5785e-01 against the
    fixed-rate run's best of 8.6681e-01 -- because the fixed rate never stops
    bouncing far enough to reach it.
    """
    _, fit_it = lasso
    _, fixed = fit_it(steps=200, lr=0.05, lr_schedule="none")
    _, sched = fit_it(steps=200, lr=0.05, lr_schedule="cosine")
    assert sched["final_over_min"] <= 1e-3
    assert sched["final_over_min"] < fixed["final_over_min"]
    assert sched["loss_min"] < fixed["loss_min"]


def test_best_iterate_is_restored_into_the_callers_tensors(lasso):
    """``fit`` mutates ``params`` in place and every consumer reads the answer
    off its own tensors, so "return best" has to mean *restore* best.  The dict
    and the tensors must agree exactly, or one of the two is lying."""
    loss_of, fit_it = lasso
    x, info = fit_it(steps=200, lr=0.05, lr_schedule="none")
    with torch.no_grad():
        assert float(loss_of(x)) == info["loss"]
    assert x.requires_grad and x.is_leaf      # still a usable optimisation leaf


def test_settled_is_false_for_a_truncated_run(lasso):
    """final_over_min alone is not enough: a run stopped while still descending
    has a perfectly smooth tail and final == min."""
    _, fit_it = lasso
    _, info = fit_it(steps=40, lr=1e-3, lr_schedule="none")
    assert info["final_over_min"] <= 1e-3      # not oscillating ...
    assert info["tail_improvement"] > 1e-3     # ... but still descending
    assert info["settled"] is False


def test_a_quiet_tail_is_not_evidence_of_a_stationary_point(lasso):
    """The caveat behind ``settled``, pinned on measured numbers.

    On the *same* truncated problem, annealing suppresses tail_improvement 95x
    (2.50e-2 -> 2.62e-4 at 150 steps) while leaving the iterate *further* from
    stationary: grad_max_final rises 1.88 -> 2.09, because the schedule stopped
    it moving rather than brought it to rest.  So a quiet tail is partly
    manufactured by the schedule -- which is why the flag is not called
    ``converged`` and why ``grad_max_final`` is reported next to it.
    """
    _, fit_it = lasso
    _, fixed = fit_it(steps=150, lr=1e-3, lr_schedule="none")
    _, cos = fit_it(steps=150, lr=1e-3, lr_schedule="cosine")
    assert cos["tail_improvement"] < fixed["tail_improvement"] / 5.0
    assert cos["grad_max_final"] > fixed["grad_max_final"]


def test_diagnostics_are_named_for_where_they_are_measured(lasso):
    """``grad_max_final`` is the gradient at the FINAL iterate, which under
    ``return_best`` is not the point left in ``params``.  Calling it ``grad_max``
    would invite exactly the misreading ("gradient at the solution") this change
    exists to stop, so the old name must not come back."""
    _, fit_it = lasso
    _, info = fit_it(steps=60, lr=0.05, lr_schedule="none")
    assert "grad_max_final" in info and "grad_max" not in info
    assert "settled" in info and "converged" not in info


def test_info_reports_the_returned_points_own_loss(lasso):
    """Old code evaluated the loss BEFORE ``opt.step()``, so ``info['loss']``
    came from the previous iterate and the parameters handed back had never been
    scored at all -- true even with the fix opted out."""
    loss_of, fit_it = lasso
    x, info = fit_it(steps=200, lr=0.05, lr_schedule="none", return_best=False)
    with torch.no_grad():
        assert float(loss_of(x)) == info["loss"] == info["final_loss"]


def test_opting_out_restores_the_old_behaviour(lasso):
    """The old path stays reachable so an archived result can be reproduced."""
    _, fit_it = lasso
    _, info = fit_it(steps=200, lr=0.05, lr_schedule="none", return_best=False)
    assert info["returned_best"] is False
    assert info["loss"] == info["final_loss"]
    assert info["final_over_min"] > 1e-3        # the overshoot is back


def test_bad_schedule_name_is_rejected(lasso):
    _, fit_it = lasso
    with pytest.raises(ValueError, match="lr_schedule"):
        fit_it(steps=2, lr=1e-3, lr_schedule="linear")


# --- degenerate budgets ----------------------------------------------------
def test_zero_steps_and_an_empty_history_do_not_crash(lasso):
    """``history`` is only populated when ``log_every`` is truthy, and a caller
    may ask for no steps at all; the diagnostics must degrade, not raise."""
    loss_of, fit_it = lasso
    x, info = fit_it(steps=0, lr=0.05)
    assert info["history"] == []
    with torch.no_grad():
        assert float(loss_of(x)) == info["loss"]     # the starting point, scored
    assert info["final_over_min"] == 0.0
    assert math.isnan(info["tail_improvement"])      # nothing to measure a tail on
    assert info["settled"] is False

    _, short = fit_it(steps=3, lr=0.05)
    assert short["history"] == [] and math.isnan(short["tail_improvement"])


def test_an_all_nan_run_reports_nan_rather_than_a_plausible_number():
    """Nothing ever beats +inf, so there is no best iterate.  The diagnostics
    must say so -- reporting ``loss_min = inf`` would look like a real, if huge,
    minimum and ``returned_best = True`` would be a lie about a restore that
    never happened."""
    x = torch.zeros(2, dtype=DT, requires_grad=True)
    info = fit([x], lambda: (x * float("nan")).sum(), steps=5, lr=0.1)
    assert math.isnan(info["loss"]) and math.isnan(info["loss_min"])
    assert math.isnan(info["final_over_min"])
    assert info["argmin"] == -1
    assert info["returned_best"] is False and info["settled"] is False


def test_a_detached_loss_is_scored_not_crashed_on():
    """The end-of-run scoring is a *new* evaluation, and the LBFGS branch used
    to make it under ``no_grad``.  A loss with nothing to differentiate must
    still report a number and an honest NaN gradient, not raise."""
    x = torch.zeros(3, dtype=DT, requires_grad=True)
    info = fit([x], lambda: (x.detach() ** 2 + 4.0).sum(), steps=0)
    assert info["loss"] == 12.0
    assert math.isnan(info["grad_max_final"])


def test_history_still_honours_log_every(lasso):
    """Unchanged contract: ``history`` is the subsampled user log.  The
    diagnostics are computed on every evaluation instead, so they stay honest
    when a caller logs nothing."""
    _, fit_it = lasso
    _, none = fit_it(steps=50, lr=0.05)
    _, every = fit_it(steps=50, lr=0.05, log_every=1)
    _, tenth = fit_it(steps=50, lr=0.05, log_every=10)
    assert none["history"] == []
    assert len(every["history"]) == 50 and len(tenth["history"]) == 6
    assert none["loss_min"] == every["loss_min"] == tenth["loss_min"]


def test_callback_still_fires_once_per_step(lasso):
    """The extra end-of-run scoring must not leak into the per-step callback."""
    _, fit_it = lasso
    seen = []
    fit_it(steps=17, lr=0.05, callback=lambda s, l: seen.append(s))
    assert seen == list(range(17))


# --- lbfgs branch ----------------------------------------------------------
def test_lbfgs_gets_the_same_guarantee_without_a_schedule(lasso):
    """LBFGS runs as one ``opt.step(closure)`` with ``max_iter=steps``, so a
    scheduler stepped outside it could never fire; ``info`` reports what actually
    happened rather than what was asked for.  ``return_best`` still applies, and
    there ``argmin`` counts closure evaluations, not steps."""
    loss_of, fit_it = lasso
    x, info = fit_it(steps=60, lr=1.0, optimizer="lbfgs", log_every=1)
    assert info["lr_schedule"] == "none"
    assert info["returned_best"] is True
    assert info["loss"] <= info["loss_min"] + 1e-15
    with torch.no_grad():
        assert float(loss_of(x)) == info["loss"]
    assert 0 <= info["argmin"] < len(info["history"])
