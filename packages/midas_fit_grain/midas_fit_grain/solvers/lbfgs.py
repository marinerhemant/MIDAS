"""L-BFGS solver wrapper around ``torch.optim.LBFGS``.

The closure must compute residuals, sum-of-squares them, ``backward()``,
and return the loss tensor — exactly the standard PyTorch LBFGS contract.
"""

from __future__ import annotations

from typing import Callable, List

import torch


def minimize_lbfgs(
    closure: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    *,
    max_iter: int = 200,
    ftol: float = 1e-5,
    xtol: float = 1e-7,
    lbfgs_inner_iter: int = 20,
    history_size: int = 10,
    line_search_fn: str = "strong_wolfe",
    **_,
):
    """Minimize ``loss`` (returned by ``closure``) wrt ``params``.

    Returns a dict with ``final_loss``, ``n_iter``, ``converged``,
    ``history``, plus three diagnostics: ``frozen_steps``, ``grad_inf`` and
    ``grad_inf_0``.

    ``converged`` is True if the relative change in loss is below ``ftol``
    for 8 consecutive outer steps.

    The diagnostics exist because a frozen loss is ambiguous: it means either
    "we are at a minimum" or "``torch.optim.LBFGS``'s strong-Wolfe line search
    could not find an improving step and returned t = 0". Only the second is a
    failure, and ``converged`` cannot tell them apart — an fp32 FF grain fit
    takes one improving step, then repeats the loss bit-for-bit until the
    ``ftol`` counter trips, reporting success with the position still exactly
    on its seed (1-ID GE5 Au3, 2026-07-30).

    Deciding which case you are in needs problem knowledge this solver does
    not have, so it only reports:

    ``frozen_steps``  consecutive trailing outer steps in which NO parameter
                      moved at all;
    ``grad_inf``      ``max |grad|`` at the last evaluated point;
    ``grad_inf_0``    ``max |grad|`` at entry, before any step.

    A large ``frozen_steps`` with ``grad_inf`` still comparable to
    ``grad_inf_0`` is a stalled line search. The authoritative check for FF
    refinement is the domain one in ``refine_block`` — a fitted position that
    is bit-identical to its seed.
    """
    if not params:
        raise ValueError("L-BFGS needs at least one parameter")

    optimizer = torch.optim.LBFGS(
        params,
        max_iter=lbfgs_inner_iter,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )

    history: list[float] = []
    prev_loss = float("inf")
    converged = False
    n_iter = 0
    n_below = 0  # consecutive ftol-tight steps before declaring convergence

    # Last finite iterate. torch.optim.LBFGS's strong-Wolfe line search can
    # emit a NaN/inf step (degenerate cubic interpolation when the bracket
    # collapses) — far more likely for large *batched* problems where one
    # scalar loss + one step size serves thousands of grains. Such a step
    # writes NaN straight into ``params`` *inside* ``optimizer.step``, before
    # any closure-level guard runs, and ``res * mask`` then spreads it
    # (NaN*0 = NaN). We snapshot each accepted iterate and roll back if the
    # next step is non-finite, so the solver can never *return* NaN params.
    last_good = [p.detach().clone() for p in params]

    n_stalled = 0        # consecutive trailing steps that moved NO parameter
    grad_inf = float("nan")      # |grad|_inf at the last evaluated point

    # Entry gradient, measured BEFORE any step. Taking it after the first
    # optimizer.step (as a first cut did) makes it useless as a reference: a
    # seed that already sits near the optimum then shows no drop at all.
    grad_inf_0 = float("nan")
    try:
        for p in params:
            p.grad = None
        closure()
        grad_inf_0 = max(
            (float(p.grad.detach().abs().max())
             for p in params if p.grad is not None),
            default=0.0,
        )
    except Exception:                                        # noqa: BLE001
        # Diagnostics must never break a solve that would otherwise work.
        pass

    for step in range(max_iter):
        before = [p.detach().clone() for p in params]
        loss = optimizer.step(closure)
        params_finite = all(torch.isfinite(p).all() for p in params)
        if (not torch.isfinite(loss)) or (not params_finite):
            # Roll back to the last finite iterate and stop. Good grains keep
            # the refinement they accumulated up to here; the divergent step
            # is discarded for the whole batch.
            with torch.no_grad():
                for p, g in zip(params, last_good):
                    p.copy_(g)
            break
        with torch.no_grad():
            for p, g in zip(params, last_good):
                g.copy_(p)
        loss_v = float(loss.detach())
        history.append(loss_v)
        n_iter = step + 1

        # Diagnostics (see the docstring). Reported, never acted on — the
        # solver cannot tell "at a minimum" from "line search gave up", and
        # guessing here false-positives on legitimate fits whose seed already
        # sits near the optimum.
        moved = max(
            float((p.detach() - b).abs().max()) for p, b in zip(params, before)
        )
        grad_inf = max(
            (float(p.grad.detach().abs().max())
             for p in params if p.grad is not None),
            default=0.0,
        )
        n_stalled = n_stalled + 1 if moved == 0.0 else 0

        # Relative-change check; require it to be tight for 8 consecutive
        # steps so we don't exit while a poorly-conditioned axis is still
        # creeping. Parameter-delta check is omitted: with one of the 12
        # grain params already at GT the min-over-params dx is always
        # near-zero and would always exit.
        rel = abs(loss_v - prev_loss) / max(abs(prev_loss), 1e-12)
        if rel < ftol:
            n_below += 1
            if n_below >= 8:
                converged = True
                break
        else:
            n_below = 0
        prev_loss = loss_v

    return {
        "final_loss": history[-1] if history else float("inf"),
        "n_iter": n_iter,
        "converged": converged,
        "frozen_steps": n_stalled,
        "grad_inf": grad_inf,
        "grad_inf_0": grad_inf_0,
        "history": history,
    }
