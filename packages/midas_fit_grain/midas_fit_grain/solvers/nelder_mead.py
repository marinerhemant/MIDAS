"""Nelder–Mead simplex solver via SciPy.

This solver exists for **C parity**: it mirrors what
``FitPosOrStrainsOMP.c`` does (NLopt LN_NELDERMEAD). Don't pick it for
production runs — L-BFGS or L-M are faster — but it's invaluable when
diffing Python output against the C reference.

Operates purely on detached numpy values; no autograd.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
from scipy.optimize import minimize


def minimize_nelder_mead(
    closure: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    *,
    max_iter: int = 5000,
    ftol: float = 1e-5,
    xtol: float = 1e-5,
    bounds=None,
    n_restarts: int = 2,
    relative_tol: bool = True,
    **_,
):
    """Minimize ``loss(closure)`` via SciPy ``minimize(method='Nelder-Mead')``.

    The closure must compute the loss at the *current* values of
    ``params`` and return a scalar tensor. Backward is **not** required;
    Nelder–Mead is derivative-free.

    C-PARITY NOTES. This solver exists to mirror ``FitPosOrStrainsOMP.c``
    (nlopt ``LN_NELDERMEAD``). Measured on the real 2-Au-grain dataset, an
    earlier version of it was the WORST of six implementations -- it moved
    228 um from its seed against C's 25 um and ended with a 42 % larger
    position residual (275 vs 194 um). It was wandering, not optimizing.
    Three configuration mismatches caused that, all fixed here:

    * **Bounds.** C calls ``nlopt_set_lower_bounds``/``upper_bounds``; scipy
      got none, so the simplex could leave the physical box entirely.
    * **Relative vs absolute tolerances.** C sets ``ftol_rel``/``xtol_rel``
      = 1e-5. scipy's ``fatol``/``xatol`` are ABSOLUTE. On a loss of order
      1e3-1e5 and positions of order 100 um these are not the same stopping
      rule -- not even close. ``relative_tol=True`` rescales them to match.
    * **Restarts.** C runs the optimizer TWICE in succession, restarting from
      the previous minimum to escape a collapsed simplex. ``n_restarts=2``
      reproduces that; it is the default because C parity is the whole point
      of this solver.

    Note also that scipy runs its simplex in float64 whatever the torch dtype,
    so "float32 Nelder-Mead" is float32 only in the objective evaluation. NM
    compares objective values rather than differencing them, so an ~1e-7
    relative difference never flips a simplex decision -- which is why fp32
    and fp64 runs land in the same place to ~1e-11.
    """
    if not params:
        raise ValueError("Nelder-Mead needs at least one parameter")

    sizes = [p.numel() for p in params]
    shapes = [p.shape for p in params]

    def _read_flat() -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in params])

    def _write_flat(flat: np.ndarray) -> None:
        i = 0
        for p, s, sh in zip(params, sizes, shapes):
            slab = flat[i:i + s].reshape(sh)
            p.detach().copy_(torch.from_numpy(slab).to(dtype=p.dtype, device=p.device))
            i += s

    saved = _read_flat()

    # Relative -> absolute, referenced to the starting point, so the stopping
    # rule means what it means in the C reference.
    if relative_tol:
        with torch.no_grad():
            f0 = abs(float(closure().detach().cpu().item()))
        x_scale = float(np.max(np.abs(saved))) if saved.size else 1.0
        fatol = ftol * max(f0, 1e-30)
        xatol = xtol * max(x_scale, 1e-30)
    else:
        fatol, xatol = ftol, xtol

    history: list[float] = []
    iters = 0

    def _f(flat_np: np.ndarray) -> float:
        nonlocal iters
        _write_flat(flat_np)
        with torch.no_grad():
            loss = closure()
        v = float(loss.detach().cpu().item())
        history.append(v)
        iters += 1
        return v

    x = saved
    res = None
    for _restart in range(max(1, int(n_restarts))):
        res = minimize(
            _f, x, method="Nelder-Mead", bounds=bounds,
            options={
                "maxiter": max_iter,
                "fatol": fatol,
                "xatol": xatol,
                "adaptive": True,
                "disp": False,
            },
        )
        x = res.x            # restart from the minimum, as the C code does

    _write_flat(res.x)

    return {
        "final_loss": float(res.fun),
        "n_iter": int(getattr(res, "nit", iters)),
        "converged": bool(res.success),
        "history": history,
    }
