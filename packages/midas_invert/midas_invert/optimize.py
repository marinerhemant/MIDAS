"""Gradient-fitting loop and scale-aware losses (domain-agnostic).

Shared by midas_2d, and intended for HEDM / Laue inversions (pf-/grain-ODF,
spectrum recovery, ...).  Nothing here knows about diffraction; the loss
closure decides what is fit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["fit", "relative_l2_loss", "cosine_loss"]


def relative_l2_loss(pred, obs, *, eps=1e-12):
    """Scale-robust loss ``||pred - obs||^2 / (||obs||^2 + eps)``."""
    import torch
    pred = torch.as_tensor(pred)
    obs = torch.as_tensor(obs)
    return ((pred - obs) ** 2).sum() / (obs.pow(2).sum() + eps)


def cosine_loss(pred, obs, *, eps=1e-12):
    """Scale-*invariant* shape loss ``1 - <pred,obs>^2 / (|pred|^2 |obs|^2)``.

    Smooth and free of the argmax kink that peak-normalisation introduces -- the
    right choice when only the *shape* of a curve matters and absolute scale is
    a nuisance (rocking curves, fringe profiles, spectra).
    """
    import torch
    pred = torch.as_tensor(pred).reshape(-1)
    obs = torch.as_tensor(obs).reshape(-1)
    num = (pred @ obs) ** 2
    den = (pred @ pred) * (obs @ obs) + eps
    return 1.0 - num / den


def fit(params, loss_fn, *, steps=400, lr=0.05, optimizer="adam",
        lr_schedule="cosine", return_best=True, callback=None, log_every=0):
    """Gradient-fit a list of leaf tensors to minimise ``loss_fn()``.

    The tensors in ``params`` are updated **in place**: the caller reads the
    answer off its own tensors and the returned dict is diagnostics only.

    Parameters
    ----------
    params : sequence of tensors with requires_grad=True
    loss_fn : callable () -> scalar tensor
    steps, lr : optimisation controls
    optimizer : {"adam", "lbfgs"}
    lr_schedule : {"cosine", "none"}
        ``"cosine"`` anneals ``lr`` to zero over ``steps``; ``"none"`` restores
        the old fixed rate.  Adam only -- see below.
    return_best : bool
        Leave the lowest-loss iterate actually evaluated in ``params``, rather
        than whatever the last update landed on.
    callback : callable(step, loss_float), optional
    log_every : int; if > 0, record loss every ``log_every`` steps.

    Returns
    -------
    dict with ``loss`` (the loss of the point left in ``params``), ``history``
    (subsampled by ``log_every``, so possibly empty) and the diagnostics below.

    Convergence
    -----------
    At a fixed learning rate Adam does not settle, and this loop used to hand
    back whatever the final step produced.  Measured on the sister package's
    12-D campaign (``midas_dct_tt/dev/paper/runs/c2/verify_p10a/probe9.log``)
    the returned iterate was **19-62% worse in loss than the best iterate the
    same run had already visited**, with the argmin as early as step 207 of 600.

    The damage was not in the reconstructions -- those moved <1% -- it was in
    **model selection**: the argmin moves with the hyperparameter, so every
    candidate value was scored at a different and arbitrary degree of
    convergence, which invalidates any comparison *across* hyperparameters.  A
    blind cross-validation selector picked the wrong regularisation weight
    before the fix and the right one after.  Hence two changed defaults:

    * ``lr_schedule="cosine"`` anneals to ``eta_min=0``.  A floor of ``lr/100``
      was tried first and is not enough: it leaves the last steps moving at
      nearly the full rate and the overshoot survives.
    * ``return_best=True`` restores the lowest-loss iterate into ``params``.

    Both are needed.  The schedule stops the oscillation; ``return_best`` is the
    guarantee, since no schedule proves the last step was downhill.

    They are not equally free, and this is a shared primitive.  ``return_best``
    costs nothing: where a run never overshoots it restores the point the run
    already ended on, bit for bit.  ``lr_schedule`` is a genuine trade-off,
    because annealing halves the effective rate over the run -- so on a
    **truncated** fit, one still descending when the budget runs out, it makes
    the answer worse.  Measured on two ``midas_dct_tt`` callers, both of which
    have ``final_over_min = 0.0`` (no overshoot to fix): shape recovery lost
    dice 0.973 -> 0.916 and a uniform-``F`` fit lost ``max|F - F_true|``
    7.03e-4 -> 9.57e-4.  Both now pass ``lr_schedule="none"`` deliberately.
    ``tail_improvement`` is the number that says which regime a fit is in;
    check it before choosing, and prefer more ``steps`` to a schedule when the
    run is merely truncated.

    Diagnostics returned alongside ``loss``: ``loss_min``, ``argmin``,
    ``final_loss``, ``final_over_min`` (relative excess of the last iterate over
    the best), ``tail_improvement`` (relative loss decrease over the last 10% of
    evaluations), ``grad_max_final``, ``settled``, ``lr_schedule`` and
    ``returned_best``.

    ``settled`` is **not** a convergence certificate and is deliberately not
    named one.  It is True when ``final_over_min`` and ``tail_improvement`` are
    both below ``1e-3``, i.e. *the run stopped moving inside its own budget*.
    Under a cosine schedule that is partly manufactured: annealing ``lr`` to
    zero drives ``tail_improvement`` to zero whether or not the iterate is
    anywhere near a stationary point, so a truncated run can be ``settled``.
    Read it next to ``grad_max_final``, which is named for the point it is
    measured at -- the *final* iterate, which with ``return_best`` is not the
    one left in ``params``.

    With ``optimizer="lbfgs"`` the whole run is a single ``opt.step(closure)``
    with ``max_iter=steps``, so a scheduler stepped outside it could never fire
    (and strong-Wolfe picks its own step length anyway).  None is applied there
    and ``info["lr_schedule"]`` reports ``"none"``; ``argmin`` then counts
    closure evaluations rather than steps.
    """
    import torch

    if lr_schedule not in ("cosine", "none"):
        raise ValueError(
            f"lr_schedule must be 'cosine' or 'none', got {lr_schedule!r}"
        )

    params = [p for p in params if p is not None]
    history: list[float] = []
    # Every scored point, in order.  `history` is a user-facing log that
    # `log_every` may subsample to nothing, but the diagnostics have to be
    # computed on what actually happened.
    trace: list[float] = []
    best = {"loss": float("inf"), "at": -1, "params": None}

    def record(value):
        """Score one point, and keep a copy of it if it is the best so far.

        Called from inside the loss evaluation, never from the step loop: that
        is the only place where a loss and the parameters that produced it are
        guaranteed to be the same point.  Adam evaluates before ``opt.step()``
        and LBFGS's line search evaluates several times per step, so a
        loop-level snapshot pairs a loss with the wrong iterate under both.
        """
        trace.append(value)
        if value < best["loss"]:
            best["loss"] = value
            best["at"] = len(trace) - 1
            best["params"] = [p.detach().clone() for p in params]

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            record(float(loss.detach()))
            if log_every:
                history.append(trace[-1])
            return loss

        opt.step(closure)
    else:
        opt = torch.optim.Adam(params, lr=lr)
        # eta_min=0: anneal the whole way.  max(steps, 1) only guards steps=0,
        # where CosineAnnealingLR would divide by T_max.
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=max(steps, 1), eta_min=0.0)
                 if lr_schedule == "cosine" else None)
        for step in range(steps):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            record(float(loss.detach()))
            opt.step()
            if sched is not None:
                sched.step()
            if log_every and (step % log_every == 0 or step == steps - 1):
                history.append(trace[-1])
            if callback is not None:
                callback(step, trace[-1])

    # Score the point the optimiser actually stopped on.  Under both branches
    # the last in-loop evaluation happens BEFORE the update that followed it,
    # so without this the parameters handed back have never been scored and
    # ``loss`` belongs to the previous iterate.  With gradients rather than
    # ``no_grad`` so that ``grad_max_final`` really is measured at the final
    # point, and recorded like any other evaluation so ``return_best`` cannot
    # restore an earlier iterate over a better final one.
    opt.zero_grad()
    final = loss_fn()
    if final.requires_grad:       # a detached loss has no gradient to report,
        final.backward()          # and must not turn into a new crash here
    record(float(final.detach()))
    grad_max_final = max((float(p.grad.abs().max()) for p in params
                          if p.grad is not None), default=float("nan"))

    # `best["params"] is None` means nothing ever beat +inf -- an all-NaN run.
    # Key every diagnostic off that rather than off `best["loss"]`, so a failed
    # run reports NaN instead of a plausible-looking inf.
    have_best = best["params"] is not None
    use_best = bool(return_best and have_best)
    if use_best:
        # In place: callers read the answer off the tensors they passed in, so
        # "return best" here means putting the best values back into those
        # leaves, not returning them.  copy_ keeps leaf identity and grad flags.
        with torch.no_grad():
            for p, b in zip(params, best["params"]):
                p.copy_(b)

    loss_min = best["loss"] if have_best else float("nan")
    final_loss = trace[-1] if trace else float("nan")
    denom = abs(loss_min) if have_best and loss_min != 0.0 else float("nan")
    final_over_min = (final_loss - loss_min) / denom if have_best else float("nan")
    # Loss decrease over the last 10% of evaluations, relative to the best.  A
    # run still improving here was truncated, however smooth its tail looks.
    tail = float("nan")
    if len(trace) >= 20:
        k = max(1, len(trace) // 10)
        tail = (trace[-k - 1] - final_loss) / denom
    settled = bool(trace) and (final_over_min <= 1e-3) and (
        tail <= 1e-3 if len(trace) >= 20 else False)

    return {
        "loss": best["loss"] if use_best else final_loss,
        "history": history,
        "loss_min": loss_min,
        "argmin": best["at"],
        "final_loss": final_loss,
        "final_over_min": final_over_min,
        "tail_improvement": tail,
        "grad_max_final": grad_max_final,
        "settled": settled,
        "lr_schedule": "none" if optimizer == "lbfgs" else lr_schedule,
        "returned_best": use_best,
    }
