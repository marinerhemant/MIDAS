"""Vectorised Nelder-Mead simplex optimiser in pure PyTorch.

Runs ``B`` independent simplex optimisations in lockstep on the same
device. The objective ``fn`` is called with a ``(B, n_dim)`` tensor of
candidate points and must return a ``(B,)`` tensor of function values
— so the caller's forward model processes all ``B`` problems in one
batched call. For our use case (``B`` = total ``(voxel, winner)``
fit problems on one block of the grid), this replaces ``B`` individual
``scipy.optimize.minimize`` calls with one batched run, amortising
the GPU-launch and Python overhead and letting the underlying
``HEDMForwardModel`` do its work in one big tensor op per NM iteration.

The simplex update rules below are the classic 1965 Nelder-Mead with
the standard reflection / expansion / contraction / shrink branches.
On any single iteration we always evaluate the four candidate points
``{x_r, x_e, x_co, x_ci}`` for every simplex (regardless of which
branch each individual simplex is in) and use ``torch.where`` to
select per-simplex which result to apply. That's a small constant
factor of wasted compute per iteration in exchange for full
vectorisation.

Bounds are enforced by clipping vertex coordinates after every
update — matching ``scipy.optimize.minimize(..., bounds=)``'s NM
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BatchedNMResult:
    """Outcome of a vectorised :func:`batched_nelder_mead` call.

    Attributes
    ----------
    x : Tensor (B, n_dim)
        Best vertex per simplex (lowest function value).
    fun : Tensor (B,)
        Function value at ``x``.
    converged : Tensor (B,) bool
        True for simplices that hit the ``xatol`` + ``fatol`` test
        before ``max_iter``.
    n_iter : int
        Total iterations the loop ran (one count for the whole batch;
        the loop exits early only if **every** simplex has converged).
    n_evals : int
        Total number of ``fn`` calls (each call processes ``B`` points).
        Multiply by ``B`` for the underlying forward count.
    """
    x: torch.Tensor
    fun: torch.Tensor
    converged: torch.Tensor
    n_iter: int
    n_evals: int


def _clip_to_bounds(
    x: torch.Tensor, bounds: Optional[torch.Tensor],
) -> torch.Tensor:
    """Clip ``x`` element-wise to the per-element ``bounds[..., 0]``,
    ``bounds[..., 1]`` interval.

    Handles two ``x`` shapes used by the optimiser:
      - ``(B, n_dim)``: a single candidate point per simplex
        (reflection / contraction / centroid). Bounds shape
        ``(B, n_dim, 2)`` aligns directly.
      - ``(B, n+1, n_dim)``: a whole simplex at a time (init / shrink).
        We broadcast the bounds along the inserted vertex axis.
    """
    if bounds is None:
        return x
    lo = bounds[..., 0]
    hi = bounds[..., 1]
    # Broadcast lo/hi over any middle (vertex) axis x has that bounds
    # doesn't. ``lo`` has shape ``(B, n_dim)``; if ``x`` has an extra
    # middle dim we unsqueeze.
    while lo.ndim < x.ndim:
        lo = lo.unsqueeze(-2)
        hi = hi.unsqueeze(-2)
    return torch.maximum(lo, torch.minimum(hi, x))


def _build_initial_simplex(
    x0: torch.Tensor,
    bounds: Optional[torch.Tensor],
    init_step: float,
) -> torch.Tensor:
    """Build a ``(B, n+1, n)`` simplex by adding a small step along
    each canonical axis. Step magnitude is ``init_step`` × box width
    if ``bounds`` is supplied, else just ``init_step`` itself.
    """
    B, n = x0.shape
    device, dtype = x0.device, x0.dtype
    if bounds is not None:
        widths = bounds[..., 1] - bounds[..., 0]   # (B, n)
        step = init_step * widths
    else:
        step = torch.full((B, n), init_step, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(0)  # (1, n, n)
    simplex = x0.unsqueeze(1).expand(B, n + 1, n).clone()
    simplex[:, 1:, :] = simplex[:, 1:, :] + eye * step.unsqueeze(1)
    if bounds is not None:
        simplex = _clip_to_bounds(simplex, bounds)
    return simplex


def batched_nelder_mead(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    bounds: Optional[torch.Tensor] = None,
    *,
    max_iter: int = 200,
    xatol: float = 1e-5,
    fatol: float = 1e-5,
    init_step: float = 0.05,
    fixed_B: bool = False,
) -> BatchedNMResult:
    """Run ``B`` simplex optimisations in lockstep.

    Parameters
    ----------
    fn : callable ``fn(x, idx) -> (B_active,)``
        Objective. ``x`` is a ``(B_active, n_dim)`` tensor of candidate
        points; ``idx`` is a ``(B_active,)`` long tensor giving the
        original problem index of each row (so the caller can subset
        per-problem aux data — typically per-grain positions —
        consistently as converged problems are trimmed out of the
        active set). Returns ``(B_active,)`` function values. ``B``
        is the same on the first call but **shrinks** as simplices
        converge and get pulled out of the loop; the caller's forward
        gets cheaper over time.
    x0 : Tensor (B, n_dim)
        Initial guesses, one per simplex.
    bounds : Tensor (B, n_dim, 2) or (n_dim, 2), optional
        Per-element ``[lo, hi]`` bounds. If ``(n_dim, 2)`` is given,
        the same bounds apply to every simplex. Vertices outside the
        box are clipped on every update.
    max_iter : int, default 200
        Hard iteration cap. The C ``FitOrientationOMP`` NLopt run
        uses 5000 evals + 30 s; from a screen-warm seed in 3D, ~50–
        100 iterations is usually enough.
    xatol, fatol : float, default 1e-5
        Convergence tolerances. A simplex is "converged" when
        **both**: the max coordinate range across its vertices is
        below ``xatol`` *and* the spread of function values is below
        ``fatol``. Matches scipy's NM defaults.
    init_step : float, default 0.05
        Initial simplex step as a fraction of bounds width. With
        ``bounds=±OrientTol``, 0.05 puts each axial vertex 10 % of a
        full box-width away from the seed — small enough that the
        seed retains influence, large enough to bracket the optimum.
    fixed_B : bool, default False
        If True, don't trim converged simplices out of the active
        batch — instead mask them out so ``fn`` is always called on
        a fixed ``(B, n_dim)`` shape. Slower per-iteration (we do
        wasted work on already-converged simplices) but lets a
        downstream ``torch.compile`` cache fixed-shape kernels and
        replay them via CUDA Graphs. Use this with
        ``torch.compile(fn, dynamic=False)`` on GPU to amortise the
        per-iteration kernel-launch cost.

    Returns
    -------
    :class:`BatchedNMResult`
    """
    if x0.ndim != 2:
        raise ValueError(f"x0 must be (B, n_dim); got {x0.shape}")
    B, n = x0.shape
    device, dtype = x0.device, x0.dtype

    if bounds is not None:
        if bounds.ndim == 2:
            bounds = bounds.unsqueeze(0).expand(B, n, 2).contiguous()
        if bounds.shape != (B, n, 2):
            raise ValueError(
                f"bounds shape {bounds.shape} doesn't match (B={B}, n={n}, 2)"
            )

    simplex = _build_initial_simplex(x0, bounds, init_step)
    # Initial active-set indices ↔ original problem rows. The NM loop
    # trims this as simplices converge.
    active_idx = torch.arange(B, device=device)
    # Initial f at every vertex. ``fn`` takes ``(B_active, n_dim)`` +
    # ``(B_active,)`` index tensor and returns ``(B_active,)``.
    # Clone each call's output before stacking — see the per-iter
    # candidate-eval block below for why (CUDA Graphs aliasing).
    f_vals = torch.stack(
        [fn(simplex[:, k], active_idx).clone() for k in range(n + 1)],
        dim=1,
    )
    n_evals = n + 1

    # Now that we know fn's actual return dtype, allocate out_f.
    out_f = torch.empty(B, dtype=f_vals.dtype, device=device)

    # Standard NM coefficients (Nelder & Mead 1965).
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    # ``out_x``: the eventual returned best vertex per problem; same
    # dtype as ``x0``. ``out_f`` was already allocated above with
    # ``f_vals``'s actual dtype (which can differ from x0's — e.g.
    # ``fn`` may downcast to float32 internally while ``x0`` is
    # float64).
    out_x = torch.empty_like(x0)
    converged_mask = torch.zeros(B, dtype=torch.bool, device=device)

    # ``active_idx`` already initialised above.
    if bounds is not None:
        active_bounds = bounds
    else:
        active_bounds = None

    last_iter = 0
    for it in range(max_iter):
        last_iter = it + 1
        B_active = simplex.shape[0]

        # Sort each simplex by f ascending so vertex 0 is best.
        order = f_vals.argsort(dim=1)
        simplex = torch.gather(
            simplex, 1, order.unsqueeze(-1).expand(-1, -1, n),
        )
        f_vals = torch.gather(f_vals, 1, order)

        # Convergence: x-range per simplex < xatol AND f-range < fatol.
        x_range = (simplex.max(dim=1).values
                   - simplex.min(dim=1).values).max(dim=1).values
        f_range = f_vals[:, -1] - f_vals[:, 0]
        just_converged = (x_range < xatol) & (f_range < fatol)
        if fixed_B:
            # Fixed-B mode: don't trim. Just record which problems
            # have converged so the simplex update below can skip
            # them. ``fn`` keeps seeing a constant-shape ``(B, n)``
            # input, which lets a downstream ``torch.compile`` /
            # CUDA Graphs cache its kernels.
            converged_mask = converged_mask | just_converged
            if bool(converged_mask.all()):
                break
        elif bool(just_converged.any()):
            done_idx = active_idx[just_converged]
            out_x[done_idx] = simplex[just_converged, 0]
            out_f[done_idx] = f_vals[just_converged, 0]
            converged_mask[done_idx] = True
            keep = ~just_converged
            simplex = simplex[keep]
            f_vals = f_vals[keep]
            active_idx = active_idx[keep]
            if active_bounds is not None:
                active_bounds = active_bounds[keep]
            B_active = simplex.shape[0]
            if B_active == 0:
                break

        # Centroid of the n best vertices (excluding worst).
        centroid = simplex[:, :-1].mean(dim=1)            # (B_active, n)
        x_worst = simplex[:, -1]                          # (B_active, n)

        # Four candidate points per simplex; evaluate each via one
        # ``fn`` call (each shape ``(B_active, n)``).
        x_r = _clip_to_bounds(centroid + alpha * (centroid - x_worst), active_bounds)
        x_e = _clip_to_bounds(centroid + gamma * (centroid - x_worst), active_bounds)
        x_co = _clip_to_bounds(centroid + rho * (centroid - x_worst), active_bounds)
        x_ci = _clip_to_bounds(centroid - rho * (centroid - x_worst), active_bounds)
        # Clone fn outputs because ``torch.compile(mode="reduce-overhead")``
        # uses CUDA Graphs which reuse the output buffer across calls;
        # without ``.clone()`` the four candidate fracs alias the same
        # piece of memory and the where-branch logic below sees stale
        # values. The clone is a noop on the eager path.
        # Evaluate the reflection first: it ALONE decides which branch each
        # simplex takes, and the branches are mutually exclusive, so every
        # simplex needs at most ONE of {x_e, x_co, x_ci}. Evaluating all four
        # meant ~2x more objective work than the algorithm asks for -- and the
        # objective is the expensive part (a full forward projection of every
        # predicted spot for every problem in the batch).
        f_r = fn(x_r, active_idx).clone()
        n_evals += 1

        f_best = f_vals[:, 0]
        f_second_worst = f_vals[:, -2] if n >= 1 else f_vals[:, 0]
        f_worst = f_vals[:, -1]

        # Per-simplex branch flags.
        better_than_best = f_r < f_best
        in_middle = (f_r >= f_best) & (f_r < f_second_worst)
        outside = (f_r >= f_second_worst) & (f_r < f_worst)
        # else: inside (f_r >= f_worst)
        _is_inside = ~(better_than_best | in_middle | outside)

        # The one extra point this simplex actually needs. ``in_middle``
        # accepts the reflection outright and needs none, so it is compacted
        # out of the call entirely.
        x_second = torch.where(
            better_than_best.unsqueeze(-1), x_e,
            torch.where(outside.unsqueeze(-1), x_co, x_ci),
        )
        f_second = torch.full_like(f_r, float("inf"))
        _need_second = better_than_best | outside | _is_inside
        if bool(_need_second.any()):
            _sel = torch.nonzero(_need_second, as_tuple=False).squeeze(-1)
            f_second[_sel] = fn(x_second[_sel], active_idx[_sel]).clone()
            n_evals += 1
        # Each of these is only ever READ under its own branch mask, where
        # f_second holds exactly that branch's value; elsewhere it is +inf,
        # which the ``&`` in each use_* test discards anyway.
        f_e = f_co = f_ci = f_second

        # Pick the new "worst" vertex per simplex.
        new_x = x_worst.clone()
        new_f = f_worst.clone()

        # Branch 1: better than best — try expansion.
        use_e = better_than_best & (f_e < f_r)
        new_x = torch.where(use_e.unsqueeze(-1), x_e, new_x)
        new_f = torch.where(use_e, f_e, new_f)
        use_r1 = better_than_best & ~use_e
        new_x = torch.where(use_r1.unsqueeze(-1), x_r, new_x)
        new_f = torch.where(use_r1, f_r, new_f)

        # Branch 2: in middle — accept reflection.
        new_x = torch.where(in_middle.unsqueeze(-1), x_r, new_x)
        new_f = torch.where(in_middle, f_r, new_f)

        # Branch 3: outside — try outside contraction.
        use_co = outside & (f_co <= f_r)
        new_x = torch.where(use_co.unsqueeze(-1), x_co, new_x)
        new_f = torch.where(use_co, f_co, new_f)

        # Branch 4: inside — try inside contraction.
        is_inside = _is_inside
        use_ci = is_inside & (f_ci < f_worst)
        new_x = torch.where(use_ci.unsqueeze(-1), x_ci, new_x)
        new_f = torch.where(use_ci, f_ci, new_f)

        # Anyone whose contraction failed needs to shrink.
        need_shrink = (
            (outside & ~use_co)
            | (is_inside & ~use_ci)
        )

        # Apply non-shrink updates: replace worst vertex. In fixed-B
        # mode, freeze converged simplices.
        if fixed_B:
            keep_mask = (~converged_mask).unsqueeze(-1)
            simplex[:, -1] = torch.where(
                keep_mask, new_x, simplex[:, -1],
            )
            f_vals[:, -1] = torch.where(
                ~converged_mask, new_f, f_vals[:, -1],
            )
        else:
            simplex[:, -1] = new_x
            f_vals[:, -1] = new_f

        # Shrink branch: keep vertex 0 (the best), pull all others
        # halfway toward it, re-evaluate. In fixed-B mode, also gate
        # on ~converged so already-done simplices don't get shrunk.
        if fixed_B:
            need_shrink = need_shrink & ~converged_mask
        if bool(need_shrink.any()):
            x_best = simplex[:, 0:1]                     # (B_active, 1, n)
            shrunk = x_best + sigma * (simplex - x_best)
            shrunk = _clip_to_bounds(shrunk, active_bounds)
            mask = need_shrink.view(simplex.shape[0], 1, 1)
            simplex = torch.where(mask, shrunk, simplex)

            # Re-evaluate the n vertices we changed (vertex 0 is intact).
            # ``.clone()`` is required for CUDA Graphs aliasing safety;
            # see the per-iter candidate-eval block above.
            # Only the shrinking simplices changed, so only they need
            # re-evaluating. With a large batch at least one simplex shrinks
            # almost every iteration, so the full-batch version paid n extra
            # objective calls per iteration for a subset that is typically
            # about half the batch.
            _sh = torch.nonzero(need_shrink, as_tuple=False).squeeze(-1)
            shrunk_f = torch.full(
                (simplex.shape[0], n), float("inf"),
                device=f_vals.device, dtype=f_vals.dtype,
            )
            for k in range(1, n + 1):
                shrunk_f[_sh, k - 1] = fn(
                    simplex[_sh, k], active_idx[_sh]).clone()
            n_evals += n
            mask_f = need_shrink.unsqueeze(1)            # (B_active, 1)
            f_vals[:, 1:] = torch.where(
                mask_f, shrunk_f, f_vals[:, 1:],
            )

    # Final result assembly. In ``fixed_B`` mode we never trimmed,
    # so simplex/f_vals still have the full ``B`` rows; copy them
    # straight out. Otherwise spill any still-active simplices that
    # hit ``max_iter`` before converging.
    if fixed_B:
        order = f_vals.argsort(dim=1)
        simplex = torch.gather(
            simplex, 1, order.unsqueeze(-1).expand(-1, -1, n),
        )
        f_vals = torch.gather(f_vals, 1, order)
        out_x = simplex[:, 0]
        out_f = f_vals[:, 0]
    elif simplex.shape[0] > 0:
        order = f_vals.argsort(dim=1)
        simplex = torch.gather(
            simplex, 1, order.unsqueeze(-1).expand(-1, -1, n),
        )
        f_vals = torch.gather(f_vals, 1, order)
        out_x[active_idx] = simplex[:, 0]
        out_f[active_idx] = f_vals[:, 0]
        # ``converged_mask`` stays False for these entries.

    return BatchedNMResult(
        x=out_x,
        fun=out_f,
        converged=converged_mask,
        n_iter=last_iter,
        n_evals=n_evals,
    )
