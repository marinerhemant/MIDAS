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
    ``history``. ``converged`` is True if the relative change in loss is
    below ``ftol`` for two consecutive outer steps.
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

    for step in range(max_iter):
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
        "history": history,
    }
