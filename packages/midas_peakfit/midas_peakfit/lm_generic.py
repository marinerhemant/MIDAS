"""Generic batched Levenberg-Marquardt solver with arbitrary residual / Jacobian.

Lifts the iteration body of ``lm.lm_solve`` (Marquardt damping, Cholesky_ex,
sigmoid box reparam, per-region masking, fp32+TF32 acceleration on CUDA) into
a problem-agnostic API so the same machinery can be reused for detector
geometry calibration, bundle adjustment, or any non-linear least squares
problem.

Two entry points:

* ``lm_solve_generic`` — standard LM with optional analytical Jacobian.
* ``lm_solve_arrowhead`` — Schur-complement reduction for problems whose
  Jacobian has a block-arrow structure  J = [J_dense | J_block_diag],  e.g.
  joint refinement of a small dense parameter vector (geometry) plus many
  per-region nuisance parameters (per-(ring, η-bin) peak shape).  Solves a
  reduced  n_dense × n_dense  system per iter instead of the full
  (n_dense + n_blocks · n_per_block)³.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
from torch.func import jacrev, vmap

# TF32 is scoped to the matmul that opted into it (``_tf32``), NOT enabled
# process-wide at import — an import must not silently change the precision
# of every other fp32 matmul in the interpreter. Importing this module used
# to re-enable the global flag and so would have undone the same fix in
# ``lm.py``.
from midas_peakfit.lm import _tf32
from midas_peakfit.reparam import u_to_x, x_to_u


@dataclass
class GenericLMConfig:
    max_iter: int = 200
    ftol_rel: float = 1e-6
    xtol_rel: float = 1e-6
    lambda_init: float = 1e-3
    lambda_inc: float = 10.0
    lambda_dec: float = 10.0
    lambda_max: float = 1e10
    lambda_min: float = 1e-12
    matmul_precision: str = "fp64"  # "fp64" | "tf32"  (tf32 only on CUDA fp64)
    # Optional Huber loss reshaping; pass huber_delta>0 to enable.
    # Residuals r are passed through r' = sign(r) · √(2 ρ_h(r) ) where
    # ρ_h(r) = ½ r² for |r|≤δ else δ(|r|−δ/2).  This keeps it a NLS problem.
    huber_delta: Optional[float] = None
    verbose: bool = False


# ----------------------------------------------------------- residual helpers
def _huberise(r: torch.Tensor, delta: float) -> torch.Tensor:
    """In-place-style Huber transform of unsigned residual magnitudes.

    Returns r' such that ½‖r'‖² == Σ ρ_h(r_i) (the Huber loss); standard NLS
    machinery then minimises ½‖r'‖².
    """
    abs_r = r.abs()
    over = abs_r > delta
    rho = torch.where(
        over,
        delta * (abs_r - 0.5 * delta),
        0.5 * r * r,
    )
    rprime_mag = torch.sqrt(2.0 * rho.clamp(min=0.0))
    return torch.sign(r) * rprime_mag


# ============================================================================
# Standard generic LM (single dense Jacobian)
# ============================================================================
ResidualFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""(u, lo, hi) -> r  with shapes  ([B, N], [B, N], [B, N]) -> [B, M]."""

ResidualJacobianFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
"""(u, lo, hi) -> (r [B, M], J [B, M, N])."""


def lm_solve_generic(
    x_init: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    residual_fn: ResidualFn,
    jacobian_fn: Optional[ResidualJacobianFn] = None,
    config: GenericLMConfig = GenericLMConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched LM with sigmoid-box bounds and Marquardt damping.

    Parameters
    ----------
    x_init : [B, N]
        Initial parameter values in bounded space ``[lo, hi]``.
    lo, hi : [B, N]
        Hard box bounds (enforced via sigmoid reparameterization in u-space).
    residual_fn : callable
        ``residual_fn(u, lo, hi) -> r`` returning ``[B, M]`` residuals.  ``u``
        is the unbounded LM variable; convert to bounded x via
        ``midas_peakfit.reparam.u_to_x(u, lo, hi)`` inside if needed.
    jacobian_fn : callable, optional
        ``jacobian_fn(u, lo, hi) -> (r, J)`` returning residuals and Jacobian
        ``[B, M, N]`` w.r.t. ``u``.  When omitted, computed via
        ``torch.func.jacrev`` on ``residual_fn``.
    config : GenericLMConfig

    Returns
    -------
    x_final : [B, N]   bounded-space parameters at termination
    cost    : [B]      final ½‖r‖²
    rc      : [B]      0=converged, 1=max-iter, 2=lambda saturated
    """
    device = x_init.device
    dtype = x_init.dtype
    B, N = x_init.shape

    u = x_to_u(x_init, lo, hi).clone()

    if jacobian_fn is None:
        # Autograd path: compute a per-region Jacobian via torch.autograd on a
        # scalar-batched residual.  Slower than analytic; correctness baseline.
        def _resjac(u_, lo_, hi_):
            B_, N_ = u_.shape
            r_ = residual_fn(u_, lo_, hi_)             # [B, M]
            M_ = r_.shape[-1]
            # Build  J[b, m, n] = ∂ r_[b, m] / ∂ u_[b, n]  by row-by-row backprop.
            # Cheap when N is small (typical of geometry refinement).
            u_req = u_.detach().clone().requires_grad_(True)
            r_req = residual_fn(u_req, lo_, hi_)
            J = torch.zeros(B_, M_, N_, dtype=u_.dtype, device=u_.device)
            for m in range(M_):
                grads = torch.autograd.grad(
                    r_req[:, m].sum(), u_req, retain_graph=(m + 1 < M_), create_graph=False
                )[0]
                J[:, m, :] = grads
            return r_, J
    else:
        _resjac = jacobian_fn

    r = residual_fn(u, lo, hi)
    if config.huber_delta is not None:
        r = _huberise(r, config.huber_delta)
    cost = (r * r).sum(dim=-1)
    lam = torch.full((B,), config.lambda_init, dtype=dtype, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    saturated = torch.zeros(B, dtype=torch.bool, device=device)

    eye = torch.eye(N, dtype=dtype, device=device)

    for it in range(config.max_iter):
        active = ~(converged | saturated)
        if not active.any():
            break

        # Compute residuals + Jacobian for the FULL batch (the user's residual
        # function sees the original layout).  We then mask the update so
        # converged / saturated regions don't move.
        r_b, J = _resjac(u, lo, hi)
        if config.huber_delta is not None:
            r_b = _huberise(r_b, config.huber_delta)

        if (
            config.matmul_precision == "tf32"
            and J.device.type == "cuda"
            and J.dtype == torch.float64
        ):
            with _tf32(True):
                J32 = J.float()
                r32 = r_b.float()
                Jt32 = J32.transpose(-1, -2)
                H = (Jt32 @ J32).double()
                g = (Jt32 @ r32.unsqueeze(-1)).squeeze(-1).double()
        else:
            Jt = J.transpose(-1, -2)
            H = Jt @ J
            g = (Jt @ r_b.unsqueeze(-1)).squeeze(-1)

        diag = torch.diagonal(H, dim1=-2, dim2=-1)
        damp = lam.unsqueeze(-1) * diag + lam.unsqueeze(-1) * 1e-9
        H_damped = H + damp.unsqueeze(-1) * eye

        L_chol, _ = torch.linalg.cholesky_ex(H_damped)
        delta = torch.cholesky_solve(-g.unsqueeze(-1), L_chol).squeeze(-1)

        # Tentative step
        u_new = u + delta
        r_new = residual_fn(u_new, lo, hi)
        if config.huber_delta is not None:
            r_new = _huberise(r_new, config.huber_delta)
        cost_new = (r_new * r_new).sum(dim=-1)

        accept = (cost_new < cost) & active   # only accept on active regions
        u = torch.where(accept.unsqueeze(-1), u_new, u)
        cost_next = torch.where(accept, cost_new, cost)
        lam_next = torch.where(
            accept,
            lam / config.lambda_dec,
            torch.where(active, lam * config.lambda_inc, lam),
        ).clamp(config.lambda_min, config.lambda_max)

        rel_cost = (cost - cost_next).abs() / cost.clamp(min=1e-12)
        u_norm = (u - delta).norm(dim=-1).clamp(min=1e-12)  # pre-step norm
        d_norm = delta.norm(dim=-1)
        rel_x = d_norm / u_norm
        # When the residual is at its noise floor (e.g. an irreducible σ from
        # measurement noise), no further step can decrease cost.  cost_new
        # ends up bit-equal to cost or 1 ULP above; nothing is "accepted",
        # λ doubles repeatedly, and λ_max would otherwise saturate the
        # region as if it had diverged.  But if the proposed step magnitude
        # is already below xtol_rel, the optimisation is parked at the
        # optimum within numerical precision — that's convergence, not
        # divergence.  Treat it as such.
        sat_pending = active & (lam_next >= config.lambda_max * 0.5)
        at_optimum = sat_pending & (rel_x < config.xtol_rel)
        conv = (
            (accept & (rel_cost < config.ftol_rel) & (rel_x < config.xtol_rel))
            | at_optimum
        )
        sat = sat_pending & ~at_optimum

        cost = cost_next
        lam = lam_next
        converged = converged | conv
        saturated = saturated | sat

        if config.verbose and it % 5 == 0:
            print(f"  LM iter {it}: cost={cost.mean():.4e}, "
                  f"active={int(active.sum())}, converged={int(converged.sum())}")

    rc = torch.zeros(B, dtype=torch.int32, device=device)
    rc[~converged] = 1
    rc[saturated] = 2
    return u_to_x(u, lo, hi), cost, rc


# ============================================================================
# Schur-arrowhead variant for joint dense-block + many small block-diagonal
# parameter problems (e.g. detector calibration with per-region peak shapes).
# ============================================================================
ArrowheadJacobianFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
"""(u, lo, hi) -> (r [M_total], J_dense [M_total, N_dense], J_blocks [n_blocks, M_block, N_block]).

Each row of J_blocks corresponds to the residuals of one block-diagonal region.
The mapping from M_total → blocks is via ``block_residual_offsets`` argument
of ``lm_solve_arrowhead``.

This is a single-problem (B=1) solver — it does NOT batch over multiple
arrowhead problems.  Calibration is one problem at a time.
"""


def lm_solve_arrowhead(
    x_init: torch.Tensor,                    # [N_total]
    lo: torch.Tensor,                        # [N_total]
    hi: torch.Tensor,                        # [N_total]
    residual_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    jacobian_fn: ArrowheadJacobianFn,
    *,
    n_dense: int,                            # size of the dense block (geometry)
    block_residual_offsets: torch.Tensor,    # [n_blocks+1] CSR-style offsets into r
    config: GenericLMConfig = GenericLMConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Schur-complement LM for joint dense + block-diagonal least squares.

    Solves
        H δ = -g            with     H = J^T J + λ diag(J^T J),   g = J^T r
    where J has the block-arrow structure
        J = [ J_dense | block_diag(J_block_1, J_block_2, …, J_block_K) ]
    and the parameter vector is [θ_dense | θ_block_1 | … | θ_block_K].

    The Schur complement reduces the n_total × n_total solve to:
      1.  invert each per-block Hessian H_kk = J_block_k^T J_block_k + λ·diag (small, vectorised);
      2.  form  S = H_dd − Σ H_dk H_kk^{-1} H_kd          (n_dense × n_dense);
      3.  solve  S · δ_dense = − (g_dense − Σ H_dk H_kk^{-1} g_k);
      4.  back-substitute    δ_block_k = − H_kk^{-1} (g_k + H_kd · δ_dense).

    Returns (x_final, cost, rc).  ``rc``: 0=converged, 1=max-iter, 2=λ-saturated.

    Notes
    -----
    Designed for the joint differentiable calibration formulation: ~25–60
    geometry parameters (dense block) plus ~5 peak-shape parameters per
    (ring, η-bin) region (per-block).  Cost per LM iteration is dominated
    by the n_blocks per-block 5×5 Cholesky solves, which run as a single
    batched ``torch.linalg.cholesky_ex`` on GPU.
    """
    device = x_init.device
    dtype = x_init.dtype
    N_total = x_init.numel()
    N_block = (N_total - n_dense) // (block_residual_offsets.numel() - 1)
    n_blocks = block_residual_offsets.numel() - 1
    assert n_dense + N_block * n_blocks == N_total, "param vector layout mismatch"

    u = x_to_u(x_init.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0)).squeeze(0).clone()
    lo1 = lo.unsqueeze(0); hi1 = hi.unsqueeze(0)

    def _flat_residual(u1d):
        return residual_fn(u1d.unsqueeze(0), lo1, hi1).squeeze(0)

    r = _flat_residual(u)
    cost = float((r * r).sum().item())
    lam = config.lambda_init
    converged = False
    saturated = False
    rc = 1

    eye_d = torch.eye(n_dense, dtype=dtype, device=device)
    eye_b = torch.eye(N_block, dtype=dtype, device=device)

    for it in range(config.max_iter):
        # Compute residual + Jacobian (split into dense + per-block).
        r_full, J_dense, J_blocks = jacobian_fn(u.unsqueeze(0), lo1, hi1)
        r_full = r_full.squeeze(0) if r_full.dim() == 2 else r_full
        if J_dense.dim() == 3:
            J_dense = J_dense.squeeze(0)
        if J_blocks.dim() == 4:
            J_blocks = J_blocks.squeeze(0)
        # Apply Huber to r if requested.
        if config.huber_delta is not None:
            r_full = _huberise(r_full, config.huber_delta)

        # H_dd = J_dense^T J_dense
        H_dd = J_dense.t() @ J_dense                           # [n_dense, n_dense]
        g_d = J_dense.t() @ r_full                             # [n_dense]

        # Per-block local Hessians and gradients (batched 5×5 here for typical use).
        # J_blocks[k] is the dense Jacobian of block k's M_k residuals w.r.t. its N_block local params.
        # Coupling block H_dk: rows = J_dense corresponding to block k's residual range.
        H_kk = torch.bmm(J_blocks.transpose(-1, -2), J_blocks)        # [n_blocks, N_b, N_b]
        # Per-block g_k = J_block_k^T r_k
        # Slice r per block based on offsets.
        offsets = block_residual_offsets
        r_block_list = [r_full[offsets[k]:offsets[k + 1]] for k in range(n_blocks)]
        g_k = torch.stack([
            J_blocks[k].t() @ r_block_list[k] for k in range(n_blocks)
        ])                                                              # [n_blocks, N_b]

        # H_dk = (J_dense rows for block k)^T  J_blocks[k]
        H_dk = torch.stack([
            J_dense[offsets[k]:offsets[k + 1]].t() @ J_blocks[k]
            for k in range(n_blocks)
        ])                                                              # [n_blocks, n_dense, N_b]

        # Marquardt damping on the diagonal of each Hessian piece.
        diag_d = torch.diagonal(H_dd)
        H_dd_d = H_dd + (lam * (diag_d + 1e-9)).diag_embed() if False else H_dd + (lam * diag_d.unsqueeze(-1) * eye_d) + (lam * 1e-9 * eye_d)
        diag_k = torch.diagonal(H_kk, dim1=-2, dim2=-1)
        H_kk_d = H_kk + (lam * diag_k).unsqueeze(-1) * eye_b + (lam * 1e-9) * eye_b

        # Invert each H_kk_d via Cholesky (batched).
        L_kk, info_k = torch.linalg.cholesky_ex(H_kk_d)              # [n_blocks, N_b, N_b]
        # Solve H_kk_d^{-1}  for both g_k and H_kd  (i.e. we need M_k = H_kk^{-1} H_kd)
        # Note: cholesky_solve expects the RHS as a column vector; batch over n_blocks.
        Mk = torch.cholesky_solve(H_dk.transpose(-1, -2), L_kk)      # [n_blocks, N_b, n_dense]
        Vk = torch.cholesky_solve(g_k.unsqueeze(-1), L_kk).squeeze(-1)  # [n_blocks, N_b]

        # Schur complement S = H_dd − Σ H_dk · Mk
        S = H_dd_d - torch.einsum("kij,kjl->il", H_dk, Mk)            # [n_dense, n_dense]
        rhs = -(g_d - torch.einsum("kij,kj->i", H_dk, Vk))            # [n_dense]

        try:
            L_S = torch.linalg.cholesky(S)
            delta_d = torch.cholesky_solve(rhs.unsqueeze(-1), L_S).squeeze(-1)
        except RuntimeError:
            # S non-PSD due to severe damping shortfall; fall back to LU.
            delta_d = torch.linalg.solve(S, rhs)

        # Back-substitute per-block deltas.
        # δ_block_k = -H_kk^{-1} (g_k + H_kd · δ_dense)
        coupling = torch.einsum("kij,j->ki", H_dk.transpose(-1, -2), delta_d)   # [n_blocks, N_b]
        delta_block = -torch.cholesky_solve(
            (g_k + coupling).unsqueeze(-1), L_kk
        ).squeeze(-1)                                                  # [n_blocks, N_b]

        # Assemble the full delta.
        delta = torch.empty_like(u)
        delta[:n_dense] = delta_d
        delta[n_dense:] = delta_block.flatten()

        u_new = u + delta
        r_new = _flat_residual(u_new)
        if config.huber_delta is not None:
            r_new = _huberise(r_new, config.huber_delta)
        cost_new = float((r_new * r_new).sum().item())

        if cost_new < cost:
            rel_cost = abs(cost - cost_new) / max(cost, 1e-12)
            rel_x = float(delta.norm() / u.norm().clamp(min=1e-12))
            u = u_new
            r = r_new
            lam = max(lam / config.lambda_dec, config.lambda_min)
            if rel_cost < config.ftol_rel and rel_x < config.xtol_rel:
                converged = True
                rc = 0
                cost = cost_new
                break
            cost = cost_new
        else:
            lam = min(lam * config.lambda_inc, config.lambda_max)
            if lam >= config.lambda_max * 0.5:
                saturated = True
                rc = 2
                break

        if config.verbose and it % 5 == 0:
            print(f"  arrowhead-LM iter {it}: cost={cost:.4e}  λ={lam:.2e}")

    x_final = u_to_x(u.unsqueeze(0), lo1, hi1).squeeze(0)
    return x_final, torch.tensor(cost, dtype=dtype, device=device), rc


__all__ = [
    "GenericLMConfig",
    "lm_solve_generic",
    "lm_solve_arrowhead",
    "ResidualFn",
    "ResidualJacobianFn",
    "ArrowheadJacobianFn",
]
