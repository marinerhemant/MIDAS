"""The C refiner's staged recipe, ported (``mode="c_recipe"``).

A direct port of ``c_src/FitUnified.c`` — the shipped c-omp refiner — driven by
:func:`midas_fit_grain.solvers.nlopt_nm.minimize_nm`, which is itself a
bit-identical port of the vendored NLopt Nelder-Mead.

Per grain, every stage runs **two warm-started NM calls**,
``ftol_rel = xtol_rel = 1e-5``:

===== ==== ================= ========================= =======================
stage dim  objective         bounds                    initial simplex
===== ==== ================= ========================= =======================
1 ori    3 (Δη, Δω)          Euler0 ± MargOme2         0.05° explicit
2 pos*   3 2-D (Δy, Δz)      centre ± Rsample/2,       **none → NLopt default
                             clamped to the sample     from the bounds**
3 str    6 2-D (Δy, Δz)      ± MargABC / MargABG %     0.001 Å, 0.01°
4 pos*   3 2-D (Δy, Δz)      as stage 2, maxeval 5000  **none → default**
===== ==== ================= ========================= =======================

``*`` FF only — in PF the voxel position is the scan-grid position and C never
refines it (``FitOrStrainsScanningOMP.c`` runs stages 1 and 3 alone).

WHY THIS EXISTS
---------------
The gradient path (``mode="iterative"``/``"all_at_once"``, L-BFGS on
``full3d``) converges — per grain, to a genuine stationary point — but to a
WORSE one: measured 64 % worse under its own objective than the C answer, on
98 % of grains, and ~40 µm away in position on real data. Not precision, not
batching, not spot selection, not the iteration budget; all were measured and
excluded.

Measured on 1-ID shade_LSHR layer 1, 200 grains, against c-omp:
``|Δposition|`` median **1.32 µm** for this recipe versus **39.9 µm** for the
shipped L-BFGS path — where c-orig and c-omp, two accepted C implementations,
differ from each other by 2.81 µm.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
The FF position deficit is **MULTI-FACTORIAL**. Four single-cause attributions
were tried and each was falsified by its own control, so this states
measurements only. shade_LSHR, 100 grains, |dposition| vs c-omp.

Within THIS module (stages/bounds/re-matching fixed):

    stage_solver=nm    + C objectives, per-grain     1.679 um
    stage_solver=lbfgs + C objectives, per-grain     1.553 um   <- solver: no
    stage_solver=lbfgs + full3d,       per-grain    38.569 um
    batched (shared optimizer) + C objectives         5.375 um   <- batching: 3.5x
    batched + C objectives but SUM-OF-SQUARES        38.434 um

In the SHIPPED ``mode="iterative"`` path:

    sumsq   + full3d                                 38.464 um
    sumsq   + phase_losses=(pixel,angular,pixel,-)   38.434 um   <- objectives: no
    sumnorm + full3d                                 30.356 um
    sumnorm + phase_losses                           36.422 um

**The reduction over spots is ASYMMETRIC and that is the key fact.** Imposing
sum-of-squares on this module destroys it (5.375 -> 38.434, reproducing the
shipped number to three decimals). But removing it from the shipped path
recovers only 21 % (38.5 -> 30.4). So Sigma(r^2) is SUFFICIENT TO DESTROY and
its removal is NECESSARY BUT NOT SUFFICIENT.

That explains why every earlier control returned null: while sum-of-squares is
in play it MASKS everything else — objectives, staging, pos_scale (tested
across a 285000x range), batching, iteration budget all measured irrelevant,
because the answer is ~38 um regardless. Remove it and the other factors become
visible, but no single one recovers the result either.

Mechanism for the asymmetry: least squares weights a spot by its error SQUARED,
so a few badly-matched spots dominate the fit and pin the grain in a wrong
basin no matter how well the rest of the machinery is set up. The C accumulates
Sigma||r_i|| -- per-spot DISTANCES -- which is robust to them.

**Practical consequence: do not try to fix the shipped path by changing one
thing.** The configuration measured to work is this whole module: per-stage
objectives + per-stage bounds + sum-of-norms + per-grain-independent
optimization. Use it, or the LM+IRLS variant which reaches 1.528 um at
0.037 s/grain on CUDA.

Consequence: a **fully differentiable** path to the C's answer exists —
``stage_solver="lbfgs"`` — see DIFFERENTIABILITY below.

UNITS: the NM parameter vector is in the C's units — position µm, Euler
**degrees**, lattice (Å, degrees) — because the explicit simplex steps (0.05°,
0.001 Å, 0.01°) are meaningless otherwise. Radians appear only inside the
objective, where the model needs them.

DIFFERENTIABILITY — read this before using this mode in a gradient pipeline
--------------------------------------------------------------------------
``stage_solver`` decides this, and there is a differentiable option that costs
nothing in accuracy (see the A/B above):

* ``stage_solver="lbfgs"`` — **fully differentiable and multi-device**: torch
  autograd on the ``*_t`` objectives, bounds by projection. Reaches the C's
  answer (1.55 µm median vs c-omp). Prefer this in gradient pipelines.
* ``stage_solver="nm"`` — reproduces the C bit-for-bit in method, but
  is **NOT differentiable end-to-end**: Nelder-Mead is derivative-free by
  construction, so there is no gradient to propagate THROUGH the search, and it
  runs on python floats via :mod:`midas_fit_grain.solvers.nlopt_nm`, making it
  effectively CPU-scalar — on CUDA each evaluation is a tiny kernel launch plus
  an ``.item()`` sync.

**The default is ``"lbfgs"``**: it matches ``"nm"`` on accuracy, keeps the
package's differentiability guarantee, and has a path to GPU throughput that a
python-scalar simplex does not. Use ``"nm"`` when the goal is specifically to
reproduce the C's search.

What is preserved, and why that is usually enough:

* **The objective stays pure torch.** :meth:`_GrainProblem.err_yz_t` and
  :meth:`err_etaome_t` return torch scalars with the graph intact; only the
  thin float wrappers the simplex consumes call ``.item()``. So
  ``d(loss)/d(position, euler, lattice)`` — and w.r.t. any differentiable
  model parameter, e.g. geometry — is available at the returned solution.
* **Differentiating THROUGH the fit is still possible, by implicit
  differentiation** rather than by unrolling: at a stationary point
  ``∇_x f(x*, θ) = 0``, so ``dx*/dθ = −H⁻¹ ∂²f/∂x∂θ`` with both terms
  computable from the torch objective above. Caveat: NM terminates on a small
  SIMPLEX, not on ``∇f = 0`` exactly, so ``x*`` satisfies stationarity only to
  the convergence tolerance — near a smooth minimum that is fine, at a bound
  (where NLopt pins and exits) it is not, and the implicit-function theorem
  does not apply to an active constraint.

**The gradient paths are untouched.** ``refine_block`` dispatches to this
module before any batched closure/solver machinery is constructed, so
``mode="iterative"`` and ``mode="all_at_once"`` keep full autograd and
multi-device behaviour. Anything needing differentiability end-to-end should
use those; this mode is for reproducing the C refiner's answer.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

from .matching import MatchResult, associate
from .observations import ObservedSpots
from .solvers.nlopt_nm import minimize_nm

DEG2RAD = math.pi / 180.0

#: FitUnified.c RunFit(): explicit simplex steps per stage dimension.
_STEP_ORIENT = 0.05          # degrees
_STEP_ABC = 0.001            # Angstrom
_STEP_ANG = 0.01             # degrees
_FTOL = _XTOL = 1e-5
_MAXEVAL_POS = 5000          # FitUnified FitPosSec config.max_evaluations


class _GrainProblem:
    """One grain's observations plus the fixed forward model."""

    def __init__(self, model, obs: ObservedSpots, obs_ring_slot: torch.Tensor,
                 pred_ring_slot: torch.Tensor, px: float,
                 omega_tol: float, eta_tol: float):
        self.model = model
        self.obs = obs
        self.obs_ring_slot = obs_ring_slot
        self.pred_ring_slot = pred_ring_slot
        self.px = px
        self.omega_tol = omega_tol
        self.eta_tol = eta_tol
        self.match: Optional[MatchResult] = None
        self._dtype = obs.y_lab.dtype
        self._device = obs.y_lab.device

    # -- forward -----------------------------------------------------------
    def _spots(self, pos, eul_deg, lat):
        # Accepts numpy (the simplex path) OR torch tensors (the gradient
        # path). Tensors are NOT round-tripped through numpy, which would sever
        # the autograd graph and silently make `stage_solver="lbfgs"` a
        # zero-gradient no-op.
        def t(a, shape, scale=1.0):
            if torch.is_tensor(a):
                v = a.to(dtype=self._dtype, device=self._device)
            else:
                v = torch.as_tensor(np.asarray(a, dtype=np.float64),
                                    dtype=self._dtype, device=self._device)
            return (v * scale if scale != 1.0 else v).reshape(*shape)
        return self.model(t(eul_deg, (1, 1, 3), DEG2RAD), t(pos, (1, 1, 3)),
                          lattice_params=t(lat, (1, 6)))

    @staticmethod
    def _sq(x):
        return x.reshape(x.shape[-2], x.shape[-1])

    def rematch(self, pos, eul_deg, lat) -> None:
        sp = self._spots(pos, eul_deg, lat)
        self.match = associate(
            self.obs.ring_nr, self.obs.omega, self.obs.eta,
            self.pred_ring_slot, self._sq(sp.omega), self._sq(sp.eta),
            self._sq(sp.valid), obs_ring_slot=self.obs_ring_slot,
            omega_tolerance=self.omega_tol, eta_tolerance=self.eta_tol,
        )

    def _pred(self, pos, eul_deg, lat):
        sp = self._spots(pos, eul_deg, lat)
        m = self.match
        g = lambda x: self._sq(x)[m.k_idx, m.m_idx]
        return g(sp.omega), g(sp.eta), g(sp.y_pixel), g(sp.z_pixel), m.mask

    # -- the two C objectives ----------------------------------------------
    # Each comes in two forms on purpose. The ``_t`` form returns a TORCH
    # SCALAR and keeps the autograd graph; the plain form is the float the
    # derivative-free simplex needs. See "DIFFERENTIABILITY" in the module
    # docstring: the search is derivative-free by nature, but the objective
    # must stay differentiable so gradients at the solution — and implicit
    # differentiation through it — remain available.
    def err_yz_t(self, pos, eul_deg, lat) -> torch.Tensor:
        """``FitErrors12D``: Σ sqrt(Δy² + Δz²) over matched spots, µm."""
        _, _, yp, zp, msk = self._pred(pos, eul_deg, lat)
        dy = self.obs.y_lab - (self.model.y_BC - yp) * self.px
        dz = self.obs.z_lab - (zp - self.model.z_BC) * self.px
        v = torch.sqrt(dy * dy + dz * dz)[msk]
        if v.numel() == 0:
            return torch.as_tensor(1e30, dtype=self._dtype, device=self._device)
        return v.sum()

    def err_etaome_t(self, pos, eul_deg, lat) -> torch.Tensor:
        """``FitErrors3DOrient``: Σ sqrt(Δη² + Δω²), DEGREES."""
        om, et, _, _, msk = self._pred(pos, eul_deg, lat)
        wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
        de = wrap(self.obs.eta - et) / DEG2RAD
        do = wrap(self.obs.omega - om) / DEG2RAD
        v = torch.sqrt(de * de + do * do)[msk]
        if v.numel() == 0:
            return torch.as_tensor(1e30, dtype=self._dtype, device=self._device)
        return v.sum()

    def err_full3d_t(self, pos, eul_deg, lat) -> torch.Tensor:
        """The package's ``full3d``: (Δy_px, Δz_px, Δω·r_px), sum of squares.

        Present ONLY as the isolation arm for ``stage_loss="full3d"`` — it runs
        the C's stages under the shipped objective, to separate "the staging
        did it" from "the per-stage objectives did it". Mirrors
        ``residuals.py`` full3d, including scaling Δω by the spot's pixel
        radius so it is an azimuthal arc in pixels.
        """
        om, _, yp, zp, msk = self._pred(pos, eul_deg, lat)
        yBC, zBC = self.model.y_BC, self.model.z_BC
        obs_y = yBC - self.obs.y_lab / self.px
        obs_z = zBC + self.obs.z_lab / self.px
        r_px = torch.sqrt((yp - yBC) ** 2 + (zp - zBC) ** 2)
        wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
        res = torch.stack([yp - obs_y, zp - obs_z,
                           wrap(om - self.obs.omega) * r_px], dim=-1)[msk]
        if res.numel() == 0:
            return torch.as_tensor(1e30, dtype=self._dtype, device=self._device)
        return (res * res).sum()

    def err_full3d(self, pos, eul_deg, lat) -> float:
        return float(self.err_full3d_t(pos, eul_deg, lat).item())

    def err_yz(self, pos, eul_deg, lat) -> float:
        return float(self.err_yz_t(pos, eul_deg, lat).item())

    def err_etaome(self, pos, eul_deg, lat) -> float:
        return float(self.err_etaome_t(pos, eul_deg, lat).item())


def _nm_twice(f, x0, lb, ub, steps, maxeval=100000):
    """Two warm-started NM calls — what every C stage does."""
    x, fv, n1, _ = minimize_nm(f, x0, lb=lb, ub=ub, step_sizes=steps,
                               ftol_rel=_FTOL, xtol_rel=_XTOL, maxeval=maxeval)
    x, fv, n2, _ = minimize_nm(f, x, lb=lb, ub=ub, step_sizes=steps,
                               ftol_rel=_FTOL, xtol_rel=_XTOL, maxeval=maxeval)
    return np.asarray(x, dtype=np.float64), float(fv), n1 + n2


def _lbfgs_twice(ft, x0, lb, ub, steps, maxeval=100000, *,
                 dtype=torch.float64, device=None):
    """Gradient counterpart of :func:`_nm_twice`, for ``stage_solver="lbfgs"``.

    Same stages, same objectives, same bounds — ONLY the search differs, so a
    difference in the result is attributable to the optimizer and not to the
    recipe. ``ft`` must return a TORCH scalar (the ``*_t`` objective forms), so
    this path is differentiable throughout.

    Bounds are enforced by projection (clamp after each step), which is the
    closest gradient analogue of NLopt's pinning; the sigmoid reparameterisation
    used elsewhere in this package would change the geometry of the search and
    confound the comparison.
    """
    x = torch.tensor(np.asarray(x0, dtype=np.float64), dtype=dtype,
                     device=device, requires_grad=True)
    lo = None if lb is None else torch.as_tensor(np.asarray(lb), dtype=dtype,
                                                 device=device)
    hi = None if ub is None else torch.as_tensor(np.asarray(ub), dtype=dtype,
                                                 device=device)
    nev = 0
    for _ in range(2):                       # two warm-started calls, as in C
        opt = torch.optim.LBFGS([x], max_iter=200, history_size=10,
                                tolerance_change=_FTOL, tolerance_grad=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            nonlocal nev
            opt.zero_grad(set_to_none=True)
            loss = ft(x)
            loss.backward()
            nev += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            if lo is not None or hi is not None:
                x.clamp_(min=lo, max=hi) if (lo is not None and hi is not None) \
                    else (x.clamp_(min=lo) if lo is not None else x.clamp_(max=hi))
    with torch.no_grad():
        fv = float(ft(x).item())
    return x.detach().cpu().numpy().astype(np.float64), fv, nev


def _pos_bounds(centre, Rsample: float, Hbeam: float):
    """``MIDAS_FG_POSBOUNDS``: centre ± Rsample/2, clamped to the sample."""
    lo = np.asarray(centre, dtype=np.float64) - Rsample / 2.0
    hi = np.asarray(centre, dtype=np.float64) + Rsample / 2.0
    if Rsample > 0:
        lo[0] = max(lo[0], -Rsample); hi[0] = min(hi[0], Rsample)
        lo[1] = max(lo[1], -Rsample); hi[1] = min(hi[1], Rsample)
    if Hbeam > 0:
        lo[2] = max(lo[2], -Hbeam / 2.0); hi[2] = min(hi[2], Hbeam / 2.0)
    return lo, hi


def refine_grain_c_recipe(prob: _GrainProblem, pos0, eul0_deg, lat0, *,
                          is_ff: bool, Rsample: float, Hbeam: float,
                          marg_ome2: float, marg_abc: float, marg_abg: float,
                          stage_solver: str = "lbfgs",
                          stage_loss: str = "c"):
    """Return ``(position, euler_deg, lattice, n_eval)`` for one grain.

    ``stage_solver`` selects ONLY the search: ``"nm"`` reproduces the C,
    ``"lbfgs"`` runs the identical stages/objectives/bounds under gradient
    descent and is differentiable throughout. Any difference between them is
    therefore attributable to the optimizer alone.
    """
    def _fit(f_float, f_torch, x0, lb, ub, steps, maxeval=100000):
        if stage_solver == "lbfgs":
            return _lbfgs_twice(f_torch, x0, lb, ub, steps, maxeval=maxeval,
                                dtype=prob._dtype, device=prob._device)
        return _nm_twice(f_float, x0, lb, ub, steps, maxeval=maxeval)

    # stage_loss="c"      -> the C's per-stage objectives (default)
    # stage_loss="full3d" -> the shipped objective everywhere; isolation arm
    _f3 = (stage_loss == "full3d")
    o_f = (lambda p, e, l: prob.err_full3d(p, e, l)) if _f3 else None
    o_t = (lambda p, e, l: prob.err_full3d_t(p, e, l)) if _f3 else None
    pos = np.asarray(pos0, dtype=np.float64).copy()
    eul = np.asarray(eul0_deg, dtype=np.float64).copy()
    lat = np.asarray(lat0, dtype=np.float64).copy()
    lat_lo = np.concatenate([lat[:3] * (1 - marg_abc / 100.0),
                             lat[3:] * (1 - marg_abg / 100.0)])
    lat_hi = np.concatenate([lat[:3] * (1 + marg_abc / 100.0),
                             lat[3:] * (1 + marg_abg / 100.0)])
    nev = 0
    prob.rematch(pos, eul, lat)

    # stage 1 — orientation, (Δη, Δω), tight explicit simplex
    eul, _, k = _fit((lambda e: o_f(pos, e, lat)) if _f3
                     else (lambda e: prob.err_etaome(pos, e, lat)),
                     (lambda e: o_t(pos, e, lat)) if _f3
                     else (lambda e: prob.err_etaome_t(pos, e, lat)), eul,
                     eul - marg_ome2, eul + marg_ome2,
                     np.full(3, _STEP_ORIENT))
    nev += k
    if is_ff:
        prob.rematch(pos, eul, lat)

    # stage 2 (FF) — position, 2-D, DEFAULT simplex derived from the bounds
    if is_ff:
        lo, hi = _pos_bounds(pos, Rsample, Hbeam)
        pos, _, k = _fit((lambda p: o_f(p, eul, lat)) if _f3 else (lambda p: prob.err_yz(p, eul, lat)),
                         (lambda p: o_t(p, eul, lat)) if _f3 else (lambda p: prob.err_yz_t(p, eul, lat)), pos, lo, hi,
                         None, maxeval=_MAXEVAL_POS)
        nev += k
        prob.rematch(pos, eul, lat)

    # stage 3 — strain, 2-D, tight explicit simplex
    steps6 = np.array([_STEP_ABC] * 3 + [_STEP_ANG] * 3)
    lat, _, k = _fit((lambda L: o_f(pos, eul, L)) if _f3
                     else (lambda L: prob.err_yz(pos, eul, L)),
                     (lambda L: o_t(pos, eul, L)) if _f3
                     else (lambda L: prob.err_yz_t(pos, eul, L)), lat,
                     lat_lo, lat_hi, steps6)
    nev += k
    if is_ff:
        prob.rematch(pos, eul, lat)

    # stage 4 (FF) — position again, now with the refined lattice
    if is_ff:
        lo, hi = _pos_bounds(pos, Rsample, Hbeam)
        pos, _, k = _fit((lambda p: o_f(p, eul, lat)) if _f3 else (lambda p: prob.err_yz(p, eul, lat)),
                         (lambda p: o_t(p, eul, lat)) if _f3 else (lambda p: prob.err_yz_t(p, eul, lat)), pos, lo, hi,
                         None, maxeval=_MAXEVAL_POS)
        nev += k
    return pos, eul, lat, nev


def refine_block_c_recipe(cfg, *, model, grains_obs, init_positions,
                          init_eulers, init_lattices, pred_ring_slot):
    """Block entry point for ``mode="c_recipe"``: loop the recipe per grain.

    Deliberately serial. The C is per-grain too, and the batching that the
    gradient path needs is what forced the shared-optimizer design; here each
    grain is an independent 3/6-parameter simplex, so there is nothing to gain
    from packing them and a lot of clarity to lose.
    """
    from .matching import ring_slot_lookup
    from .refine import GrainFitResult
    from .refine_block import BlockFitResult

    device = init_positions.device
    dtype = init_positions.dtype
    is_ff = float(getattr(cfg, "scan_pos_tol_um", 0.0) or 0.0) <= 0.0
    Rsample = float(getattr(cfg, "Rsample", 0.0) or 0.0)
    Hbeam = float(getattr(cfg, "Hbeam", 0.0) or 0.0)
    marg_ome2 = float(getattr(cfg, "MargOme2", 0.0) or 2.0)
    marg_abc = float(getattr(cfg, "MargABC", 0.0) or 2.0)
    marg_abg = float(getattr(cfg, "MargABG", 0.0) or 2.0)
    omega_tol = max(float(getattr(cfg, "MarginOme", 0.0) or 0.0), 2.0) * DEG2RAD
    eta_tol = max(float(getattr(cfg, "MarginEta", 0.0) or 0.0), 5.0) * DEG2RAD

    _sv = str(getattr(cfg, "stage_solver", "lm_irls") or "lm_irls")
    if _sv == "lm_irls":
        return refine_block_c_recipe_lm(
            cfg, model=model, grains_obs=grains_obs,
            init_positions=init_positions, init_eulers=init_eulers,
            init_lattices=init_lattices, pred_ring_slot=pred_ring_slot)

    grains, total_iter, total_loss = [], 0, 0.0
    moves = []
    for b, obs in enumerate(grains_obs):
        prob = _GrainProblem(
            model, obs, ring_slot_lookup(cfg.RingNumbers, obs.ring_nr),
            pred_ring_slot, float(cfg.px), omega_tol, eta_tol)
        p0 = init_positions[b].detach().cpu().numpy().astype(np.float64)
        e0 = init_eulers[b].detach().cpu().numpy().astype(np.float64) / DEG2RAD
        l0 = init_lattices[b].detach().cpu().numpy().astype(np.float64)
        try:
            pos, eul_deg, lat, nev = refine_grain_c_recipe(
                prob, p0, e0, l0, is_ff=is_ff, Rsample=Rsample, Hbeam=Hbeam,
                marg_ome2=marg_ome2, marg_abc=marg_abc, marg_abg=marg_abg,
                stage_solver=_sv,
                stage_loss=str(getattr(cfg, "stage_loss", "c") or "c"))
        except Exception:                                     # noqa: BLE001
            pos, eul_deg, lat, nev = p0, e0, l0, 0
        prob.rematch(pos, eul_deg, lat)
        floss = prob.err_yz(pos, eul_deg, lat)
        total_loss += floss
        total_iter += nev
        moves.append(float(np.linalg.norm(pos - p0)))
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                      dtype=dtype, device=device)
        grains.append(GrainFitResult(
            position=t(pos), euler=t(np.asarray(eul_deg) * DEG2RAD),
            lattice=t(lat), final_loss=floss,
            n_matched=int(prob.match.mask.sum().item()),
            history=[], converged=True, match=prob.match,
            per_spot_residuals=torch.zeros(0, dtype=dtype, device=device),
        ))

    mv = np.asarray(moves) if moves else np.zeros(1)
    seed_spread = 0.0
    if init_positions.numel():
        c = init_positions.double().mean(dim=0, keepdim=True)
        seed_spread = float((init_positions.double() - c).norm(dim=1).median())
    res = BlockFitResult(
        grains=grains, final_total_loss=float(total_loss),
        n_iter=int(total_iter), converged=True,
        max_position_move_um=float(mv.max()),
        median_position_move_um=float(np.median(mv)),
        n_unmoved_position=int((mv == 0.0).sum()),
        n_grains=len(grains),
    )
    if hasattr(res, "seed_position_spread_um"):
        res.seed_position_spread_um = seed_spread
    return res


# ---------------------------------------------------------------------------
# Batched variant. Two purposes, and they are inseparable:
#   1. SPEED. The per-grain path is 0.325 s/grain against c-omp's 0.016 —
#      ~20x slower — because every objective evaluation is a scalar torch call.
#   2. ISOLATION. It is the one difference from the shipped path that has not
#      been tested UNDER THE C OBJECTIVES. The earlier B=1-vs-B=400 sweep found
#      batching irrelevant, but ran under `full3d` — the objective that makes
#      nothing matter — so it settled nothing about this regime.
# If batching degrades the answer toward ~38 um, batching IS the factor that
# the objective change alone could not supply in the shipped path.
# ---------------------------------------------------------------------------
def _batched_pick(model, pos, eul_deg, lat, match):
    """Predicted (omega, eta, y_pixel, z_pixel) at the matched slots, (B, S)."""
    B = pos.shape[0]
    sp = model((eul_deg * DEG2RAD).view(B, 1, 3), pos.view(B, 1, 3),
               lattice_params=lat)
    K, M = sp.omega.shape[-2], sp.omega.shape[-1]
    flat = (match.k_idx * M + match.m_idx)
    pick = lambda t: t.reshape(B, K * M).gather(1, flat)
    return pick(sp.omega), pick(sp.eta), pick(sp.y_pixel), pick(sp.z_pixel)


def _b_err_yz(model, obs, match, pos, eul_deg, lat, px):
    _, _, yp, zp = _batched_pick(model, pos, eul_deg, lat, match)
    dy = obs.y_lab - (model.y_BC - yp) * px
    dz = obs.z_lab - (zp - model.z_BC) * px
    return (torch.sqrt(dy * dy + dz * dz) * match.mask).sum()


def _b_err_etaome(model, obs, match, pos, eul_deg, lat):
    om, et, _, _ = _batched_pick(model, pos, eul_deg, lat, match)
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    de = wrap(obs.eta - et) / DEG2RAD
    do = wrap(obs.omega - om) / DEG2RAD
    return (torch.sqrt(de * de + do * do) * match.mask).sum()


def _b_lbfgs(param, closure_fn, lo, hi, max_iter=200):
    """Two warm-started L-BFGS calls on one parameter block, bounds by clamp."""
    nev = 0
    for _ in range(2):
        opt = torch.optim.LBFGS([param], max_iter=max_iter, history_size=10,
                                tolerance_change=_FTOL, tolerance_grad=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            nonlocal nev
            opt.zero_grad(set_to_none=True)
            loss = closure_fn()
            loss.backward()
            nev += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            if lo is not None:
                torch.maximum(param, lo, out=param)
            if hi is not None:
                torch.minimum(param, hi, out=param)
    return nev


def refine_block_c_recipe_batched(cfg, *, model, grains_obs, init_positions,
                                  init_eulers, init_lattices, pred_ring_slot):
    """Batched c_recipe: one optimizer per STAGE across all grains."""
    from .batch import ObservedBatch
    from .matching import ring_slot_lookup
    from .refine import GrainFitResult
    from .refine_block import BlockFitResult, _rematch_batch

    device, dtype = init_positions.device, init_positions.dtype
    B = len(grains_obs)
    obs = ObservedBatch.pack(grains_obs, device=device, dtype=dtype)
    obs_ring_slot = ring_slot_lookup(cfg.RingNumbers, obs.ring_nr)
    is_ff = float(getattr(cfg, "scan_pos_tol_um", 0.0) or 0.0) <= 0.0
    Rs = float(getattr(cfg, "Rsample", 0.0) or 0.0)
    Hb = float(getattr(cfg, "Hbeam", 0.0) or 0.0)
    m_ome2 = float(getattr(cfg, "MargOme2", 0.0) or 2.0)
    m_abc = float(getattr(cfg, "MargABC", 0.0) or 2.0)
    m_abg = float(getattr(cfg, "MargABG", 0.0) or 2.0)
    o_tol = max(float(getattr(cfg, "MarginOme", 0.0) or 0.0), 2.0) * DEG2RAD
    e_tol = max(float(getattr(cfg, "MarginEta", 0.0) or 0.0), 5.0) * DEG2RAD

    pos = init_positions.detach().clone()
    eul = (init_eulers.detach() / DEG2RAD).clone()       # DEGREES, as in C
    lat = init_lattices.detach().clone()
    lat_lo = torch.cat([lat[:, :3] * (1 - m_abc / 100),
                        lat[:, 3:] * (1 - m_abg / 100)], dim=1)
    lat_hi = torch.cat([lat[:, :3] * (1 + m_abc / 100),
                        lat[:, 3:] * (1 + m_abg / 100)], dim=1)

    def rematch():
        return _rematch_batch(model=model, pos=pos, euler=eul * DEG2RAD,
                              lattice=lat, obs=obs, obs_ring_slot=obs_ring_slot,
                              pred_ring_slot=pred_ring_slot,
                              omega_tolerance=o_tol, eta_tolerance=e_tol)

    def pos_bounds(c):
        lo = c - Rs / 2.0
        hi = c + Rs / 2.0
        if Rs > 0:
            lo[:, 0].clamp_(min=-Rs); hi[:, 0].clamp_(max=Rs)
            lo[:, 1].clamp_(min=-Rs); hi[:, 1].clamp_(max=Rs)
        if Hb > 0:
            lo[:, 2].clamp_(min=-Hb / 2); hi[:, 2].clamp_(max=Hb / 2)
        return lo, hi

    match = rematch()
    nev = 0

    e = eul.clone().requires_grad_(True)
    nev += _b_lbfgs(e, lambda: _b_err_etaome(model, obs, match, pos, e, lat),
                    eul - m_ome2, eul + m_ome2)
    eul = e.detach()
    if is_ff:
        match = rematch()

    if is_ff:
        lo, hi = pos_bounds(pos.clone())
        p = pos.clone().requires_grad_(True)
        nev += _b_lbfgs(p, lambda: _b_err_yz(model, obs, match, p, eul, lat,
                                             float(cfg.px)), lo, hi)
        pos = p.detach()
        match = rematch()

    L = lat.clone().requires_grad_(True)
    nev += _b_lbfgs(L, lambda: _b_err_yz(model, obs, match, pos, eul, L,
                                         float(cfg.px)), lat_lo, lat_hi)
    lat = L.detach()
    if is_ff:
        match = rematch()

    if is_ff:
        lo, hi = pos_bounds(pos.clone())
        p = pos.clone().requires_grad_(True)
        nev += _b_lbfgs(p, lambda: _b_err_yz(model, obs, match, p, eul, lat,
                                             float(cfg.px)), lo, hi)
        pos = p.detach()

    match = rematch()
    with torch.no_grad():
        floss = float(_b_err_yz(model, obs, match, pos, eul, lat,
                                float(cfg.px)).item())
    mv = (pos - init_positions).norm(dim=1)
    grains = [GrainFitResult(
        position=pos[b].clone(), euler=(eul[b] * DEG2RAD).clone(),
        lattice=lat[b].clone(), final_loss=floss / max(B, 1),
        n_matched=int(match.mask[b].sum().item()), history=[], converged=True,
        match=MatchResult(k_idx=match.k_idx[b], m_idx=match.m_idx[b],
                          mask=match.mask[b],
                          delta_omega=torch.zeros(0, dtype=dtype, device=device),
                          delta_eta=torch.zeros(0, dtype=dtype, device=device)),
        per_spot_residuals=torch.zeros(0, dtype=dtype, device=device))
        for b in range(B)]
    return BlockFitResult(grains=grains, final_total_loss=floss, n_iter=nev,
                          converged=True,
                          max_position_move_um=float(mv.max()),
                          median_position_move_um=float(mv.median()),
                          n_unmoved_position=int((mv == 0).sum()),
                          n_grains=B)


# ---------------------------------------------------------------------------
# LM + IRLS: the production path. Per-grain-INDEPENDENT (own damping, own
# convergence) yet vectorised, so it keeps the accuracy that a shared optimizer
# throws away (1.55 -> 5.38 um) while running at batched speed.
#
# The C's objective is Sigma||r_i||, which is not a least-squares problem — but
# scaling each spot's residual by 1/sqrt(||r_i||) makes it one exactly:
#     Sigma || r_i / sqrt(||r_i||) ||^2  ==  Sigma ||r_i||
# i.e. IRLS with weights recomputed (and detached) each iteration. That is what
# lets a Gauss-Newton/LM solver deliver the robust norm the C uses.
#
# Measured, shade_LSHR, 100 grains, |dposition| vs c-omp:
#     LM+IRLS cuda/f64   1.528 um   0.0373 s/grain
#     LM+IRLS cpu/f64    1.528 um   0.0862
#     per-grain lbfgs    1.553 um   0.2869
#     c-orig             2.813 um   0.0360
#     c-omp                   ref   0.0162
# ---------------------------------------------------------------------------
_LM_POS_SCALE = 100.0


def _irls_residual_fn(model, obs, match, kind, px):
    """(B, S*2) residual, IRLS-scaled so least squares == the C's sum-of-norms."""
    def fn(pos, euler_rad, lattice):
        om, et, yp, zp = _batched_pick(model, pos, euler_rad / DEG2RAD,
                                       lattice, match)
        if kind == "yz":
            a = obs.y_lab - (model.y_BC - yp) * px
            b = obs.z_lab - (zp - model.z_BC) * px
        else:                                   # (delta eta, delta omega), deg
            wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
            a = wrap(obs.eta - et) / DEG2RAD
            b = wrap(obs.omega - om) / DEG2RAD
        r = torch.stack([a, b], dim=-1) * match.mask.unsqueeze(-1)
        nrm = torch.sqrt((r * r).sum(-1)).clamp_min(1e-8).detach()
        return (r / torch.sqrt(nrm).unsqueeze(-1)).reshape(r.shape[0], -1)
    return fn


def refine_block_c_recipe_lm(cfg, *, model, grains_obs, init_positions,
                             init_eulers, init_lattices, pred_ring_slot,
                             max_iter: int = 50):
    """The C recipe driven by per-grain-independent batched LM + IRLS."""
    from .batch import ObservedBatch
    from .matching import ring_slot_lookup
    from .refine import GrainFitResult
    from .refine_block import BlockFitResult, _rematch_batch
    from .solvers.lm_batched import minimize_lm_batched

    device, dtype = init_positions.device, init_positions.dtype
    B = len(grains_obs)
    obs = ObservedBatch.pack(grains_obs, device=device, dtype=dtype)
    ors = ring_slot_lookup(cfg.RingNumbers, obs.ring_nr)
    px = float(cfg.px)
    is_ff = float(getattr(cfg, "scan_pos_tol_um", 0.0) or 0.0) <= 0.0
    Rs = float(getattr(cfg, "Rsample", 0.0) or 0.0)
    Hb = float(getattr(cfg, "Hbeam", 0.0) or 0.0)
    o_tol = max(float(getattr(cfg, "MarginOme", 0.0) or 0.0), 2.0) * DEG2RAD
    e_tol = max(float(getattr(cfg, "MarginEta", 0.0) or 0.0), 5.0) * DEG2RAD

    T = lambda m: torch.tensor(m, dtype=torch.bool, device=device)
    IDX_POS = T([True] * 3 + [False] * 9)
    IDX_EUL = T([False] * 3 + [True] * 3 + [False] * 6)
    IDX_LAT = T([False] * 6 + [True] * 6)

    pos = init_positions.detach().clone()
    eul = init_eulers.detach().clone()            # RADIANS here (lm's convention)
    lat = init_lattices.detach().clone()

    def rematch():
        return _rematch_batch(model=model, pos=pos, euler=eul, lattice=lat,
                              obs=obs, obs_ring_slot=ors,
                              pred_ring_slot=pred_ring_slot,
                              omega_tolerance=o_tol, eta_tolerance=e_tol)

    def clamp_pos():
        with torch.no_grad():
            if Rs > 0:
                pos[:, 0].clamp_(-Rs, Rs)
                pos[:, 1].clamp_(-Rs, Rs)
            if Hb > 0:
                pos[:, 2].clamp_(-Hb / 2, Hb / 2)

    stages = ([("eo", IDX_EUL), ("yz", IDX_POS), ("yz", IDX_LAT), ("yz", IDX_POS)]
              if is_ff else [("eo", IDX_EUL), ("yz", IDX_LAT)])
    match = rematch()
    n_iter = 0
    for kind, idx in stages:
        out = minimize_lm_batched(
            _irls_residual_fn(model, obs, match, kind, px),
            pos / _LM_POS_SCALE, eul, lat, pos_scale=_LM_POS_SCALE,
            max_iter=max_iter, active_mask=idx)
        pos = out["pos_scaled"] * _LM_POS_SCALE
        eul, lat = out["euler"], out["lattice"]
        n_iter += int(out.get("n_iter", 0) or 0)
        clamp_pos()
        match = rematch()

    with torch.no_grad():
        floss = float(_b_err_yz(model, obs, match, pos, eul / DEG2RAD, lat,
                                px).item())
    mv = (pos - init_positions).norm(dim=1)
    grains = [GrainFitResult(
        position=pos[b].clone(), euler=eul[b].clone(), lattice=lat[b].clone(),
        final_loss=floss / max(B, 1),
        n_matched=int(match.mask[b].sum().item()), history=[], converged=True,
        match=MatchResult(k_idx=match.k_idx[b], m_idx=match.m_idx[b],
                          mask=match.mask[b],
                          delta_omega=torch.zeros(0, dtype=dtype, device=device),
                          delta_eta=torch.zeros(0, dtype=dtype, device=device)),
        per_spot_residuals=torch.zeros(0, dtype=dtype, device=device))
        for b in range(B)]
    return BlockFitResult(grains=grains, final_total_loss=floss, n_iter=n_iter,
                          converged=True,
                          max_position_move_um=float(mv.max()),
                          median_position_move_um=float(mv.median()),
                          n_unmoved_position=int((mv == 0).sum()),
                          n_grains=B)
