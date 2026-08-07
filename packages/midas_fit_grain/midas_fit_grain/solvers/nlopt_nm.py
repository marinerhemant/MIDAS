"""Bounded Nelder-Mead: a line-for-line port of MIDAS's vendored `nelder_mead.c`,
which is itself an exact reimplementation of NLopt's `LN_NELDERMEAD`
(nldrmd.c, Steven G. Johnson, MIT).

WHY A PORT AND NOT `scipy.optimize.minimize(method="Nelder-Mead")`
------------------------------------------------------------------
NLopt's simplex differs from the textbook/scipy one in ways that change the
answer, not just the path:

* **Bound handling is Richardson & Kuester (1973)**: a reflected point is
  PINNED to [lb, ub], and if the pinned point coincides (`close()`, 1e-13
  relative) with either the centroid or the vertex being reflected, the solver
  TERMINATES with XTOL instead of clamping and sliding along the bound face.
  scipy is unbounded-by-construction in its classic mode and clips differently.
* **Initial simplex** is built by offsetting each axis by `xstep`, with a
  fallback when that lands outside a bound (and a further "too close to the
  bound, go the other way" rule at 0.1·|xstep|).
* **The default initial step**, used when no explicit step is supplied — which
  is exactly the FF refiner's situation — is NLopt's
  `nlopt_set_default_initial_step()` heuristic: (ub-lb)·0.25, (ub-x)·0.75,
  (x-lb)·0.75, … For FF position bounds of seed ± Rsample that makes the
  starting simplex hundreds of µm wide, i.e. a coarse quasi-global search.
  scipy's default is a 5 % relative perturbation — utterly different.
* **Convergence** is NLopt's `relstop` on f, then an L1 x-test
  (Σ maxradius < xtol_rel · Σ|centroid|).
* **The returned point is the best EVER evaluated** (NLopt's CHECK_EVAL), not
  a vertex of the final simplex.

Getting any one of these wrong reproduces "a Nelder-Mead", not MIDAS's.

The C uses a red-black tree to track min/max/2nd-max; at the refiner's tiny n
a linear scan is equivalent and is what the C comment says it stands in for.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

# NLopt strategy constants (nldrmd.c:35).
ALPHA, BETA, GAMMA, DELTA = 1.0, 0.5, 2.0, 0.5

SUCCESS, FTOL_REACHED, XTOL_REACHED, MAXEVAL = 1, 3, 4, 5
FAILURE, INVALID_ARGS = -1, -2

_DBL_MAX = float(np.finfo(np.float64).max)


def _close(a: float, b: float) -> bool:
    """NLopt close() (nldrmd.c:47)."""
    return abs(a - b) <= 1e-13 * (abs(a) + abs(b))


def _relstop(vold: float, vnew: float, reltol: float, abstol: float) -> bool:
    """NLopt relstop() (stop.c:81)."""
    return (abs(vnew - vold) < abstol
            or abs(vnew - vold) < reltol * (abs(vnew) + abs(vold)) * 0.5
            or (reltol > 0.0 and vnew == vold))


def _reflectpt(xnew, c, scale, xold, lb, ub) -> bool:
    """NLopt reflectpt() (nldrmd.c:63). False => caller terminates with XTOL."""
    equalc = equalold = True
    for i in range(xnew.shape[0]):
        newx = c[i] + scale * (c[i] - xold[i])
        if lb is not None and newx < lb[i]:
            newx = lb[i]
        if ub is not None and newx > ub[i]:
            newx = ub[i]
        equalc = equalc and _close(newx, c[i])
        equalold = equalold and _close(newx, xold[i])
        xnew[i] = newx
    return not (equalc or equalold)


def default_initial_step(x, lb, ub) -> np.ndarray:
    """NLopt nlopt_set_default_initial_step() (options.c:912)."""
    n = x.shape[0]
    dx = np.empty(n, dtype=np.float64)
    for i in range(n):
        step = _DBL_MAX
        ubf = ub is not None
        lbf = lb is not None
        if ubf and lbf and (ub[i] - lb[i]) * 0.25 < step and ub[i] > lb[i]:
            step = (ub[i] - lb[i]) * 0.25
        if ubf and ub[i] - x[i] < step and ub[i] > x[i]:
            step = (ub[i] - x[i]) * 0.75
        if lbf and x[i] - lb[i] < step and x[i] > lb[i]:
            step = (x[i] - lb[i]) * 0.75
        if step == _DBL_MAX:
            if ubf and abs(ub[i] - x[i]) < abs(step):
                step = (ub[i] - x[i]) * 1.1
            if lbf and abs(x[i] - lb[i]) < abs(step):
                step = (x[i] - lb[i]) * 1.1
        if step == _DBL_MAX or abs(step) < 1e-300:
            step = x[i]
        if step == _DBL_MAX or step == 0.0:
            step = 1.0
        dx[i] = step
    return dx


class _Done(Exception):
    def __init__(self, rc):
        self.rc = rc


def minimize_nm(
    f: Callable[[np.ndarray], float],
    x0: Sequence[float],
    *,
    lb: Optional[Sequence[float]] = None,
    ub: Optional[Sequence[float]] = None,
    step_sizes: Optional[Sequence[float]] = None,
    ftol_rel: float = 1e-5,
    xtol_rel: float = 1e-5,
    maxeval: int = 100000,
    stopval: Optional[float] = None,
):
    """Return ``(x_best, f_best, n_eval, rc)``. ``x0`` is not modified."""
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.shape[0]
    if n == 0 or f is None:
        return x0.copy(), float("inf"), 0, INVALID_ARGS
    np1 = n + 1
    lb = None if lb is None else np.asarray(lb, dtype=np.float64)
    ub = None if ub is None else np.asarray(ub, dtype=np.float64)
    maxeval = maxeval if maxeval > 0 else 100000
    ftol_rel = ftol_rel if ftol_rel > 0.0 else 0.0
    xtol_rel = xtol_rel if xtol_rel > 0.0 else 0.0
    minf_max = stopval if stopval is not None else -_DBL_MAX

    xstep = (np.asarray(step_sizes, dtype=np.float64).copy()
             if step_sizes is not None else default_initial_step(x0, lb, ub))

    pts = np.empty((np1, n), dtype=np.float64)
    fval = np.empty(np1, dtype=np.float64)
    cen = np.empty(n, dtype=np.float64)
    xcur = np.empty(n, dtype=np.float64)

    state = {"nev": 0, "fbest": math.inf, "xbest": x0.copy()}

    def check_eval(xc, fc):
        state["nev"] += 1
        if fc <= state["fbest"]:
            state["fbest"] = fc
            state["xbest"] = np.array(xc, dtype=np.float64, copy=True)
            if state["fbest"] < minf_max:
                raise _Done(SUCCESS)
        if state["nev"] >= maxeval:
            raise _Done(MAXEVAL)

    rc = XTOL_REACHED
    try:
        pts[0] = x0
        fval[0] = f(pts[0])
        state["fbest"] = fval[0]
        state["xbest"] = pts[0].copy()
        state["nev"] = 1
        if state["fbest"] < minf_max:
            raise _Done(SUCCESS)

        # vertices 1..n, with NLopt's out-of-bound fallbacks
        for i in range(1, n + 1):
            ax = i - 1
            xax = x0[ax]
            pts[i] = x0
            p = xax + xstep[ax]
            if ub is not None and p > ub[ax]:
                if ub[ax] - xax > abs(xstep[ax]) * 0.1:
                    p = ub[ax]
                else:
                    p = xax - abs(xstep[ax])
            if lb is not None and p < lb[ax]:
                if xax - lb[ax] > abs(xstep[ax]) * 0.1:
                    p = lb[ax]
                else:
                    p = xax + abs(xstep[ax])
                    if ub is not None and p > ub[ax]:
                        far = ub[ax] if (ub[ax] - xax > xax - lb[ax]) else lb[ax]
                        p = 0.5 * (far + xax)
            pts[i, ax] = p
            if _close(p, xax):
                raise _Done(FAILURE)
            fval[i] = f(pts[i])
            check_eval(pts[i], fval[i])

        ninv = 1.0 / n
        while True:
            low = int(np.argmin(fval))
            high = int(np.argmax(fval))
            second = -1
            for i in range(np1):
                if i == high:
                    continue
                if second < 0 or fval[i] > fval[second]:
                    second = i
            fl, fh = fval[low], fval[high]

            if _relstop(fl, fh, ftol_rel, 0.0):
                raise _Done(FTOL_REACHED)

            cen[:] = 0.0
            for i in range(np1):
                if i == high:
                    continue
                cen += pts[i]
            cen *= ninv

            # L1 x-test on the max radius per coordinate
            xcur[:] = 0.0
            for i in range(np1):
                d = np.abs(pts[i] - cen)
                np.maximum(xcur, d, out=xcur)
            if xtol_rel > 0.0 and xcur.sum() < xtol_rel * np.abs(cen).sum():
                raise _Done(XTOL_REACHED)

            if not _reflectpt(xcur, cen, ALPHA, pts[high], lb, ub):
                raise _Done(XTOL_REACHED)
            fr = f(xcur)
            check_eval(xcur, fr)

            if fr < fl:
                if not _reflectpt(pts[high], cen, GAMMA, pts[high], lb, ub):
                    raise _Done(XTOL_REACHED)
                fh = f(pts[high])
                check_eval(pts[high], fh)
                if fh >= fr:                      # expansion no better
                    fh = fr
                    pts[high] = xcur
                fval[high] = fh
            elif second >= 0 and fr < fval[second]:
                pts[high] = xcur
                fval[high] = fr
            else:
                scale = -BETA if fh <= fr else BETA
                if not _reflectpt(xcur, cen, scale, pts[high], lb, ub):
                    raise _Done(XTOL_REACHED)
                fc = f(xcur)
                check_eval(xcur, fc)
                if fc < fr and fc < fh:
                    pts[high] = xcur
                    fval[high] = fc
                else:
                    for i in range(np1):          # shrink toward the lowest
                        if i == low:
                            continue
                        if not _reflectpt(pts[i], pts[low], -DELTA, pts[i],
                                          lb, ub):
                            raise _Done(XTOL_REACHED)
                        fval[i] = f(pts[i])
                        check_eval(pts[i], fval[i])
                    continue                      # == C's `goto restart`
    except _Done as d:
        rc = d.rc

    return state["xbest"], float(state["fbest"]), state["nev"], rc
