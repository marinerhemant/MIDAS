"""Batched Levenberg-Marquardt over many regions sharing ``(n_peaks, M)``.

For each region we solve

    (J^T J + λ × diag(J^T J)) δ = -J^T r

with autograd-computed Jacobians, then apply

    u ← u + δ

clip-free (bounds are enforced by reparameterization in ``reparam.u_to_x``).
Convergence is per-region; converged regions are masked out of further updates.

Marquardt's diagonal-scaled damping is used (``λ × diag(J^T J)``) rather than
``λ × I`` for better conditioning across mixed-scale parameters.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.func import jacrev, vmap

from midas_peakfit.jacobian import residuals_and_jacobian_u
from midas_peakfit.jacobian_triton import (
    _TRITON_OK,
    residuals_and_jacobian_u_triton,
)
from midas_peakfit.model import residuals
from midas_peakfit.reparam import u_to_x, x_to_u
from midas_peakfit.uncertainty import compute_param_sigma


@contextlib.contextmanager
def _tf32(enabled: bool):
    """Scope TF32 tensor-core matmul to the block that asked for it.

    This used to be a module-level ``allow_tf32 = True`` executed at import.
    Two problems with that:

    1. It is a PROCESS-WIDE side effect of an import. Anything else running
       in the same interpreter — other midas packages, a user's own torch
       code — silently got TF32 fp32 matmuls it never asked for.
    2. The justification in ``LMConfig.matmul_precision`` only covers the
       fp64 path, where J is cast to fp32 for the ``J^T J`` matmul and the
       Cholesky and delta solve stay fp64. But when the LM itself runs in
       fp32 (``midas-pipeline --dtype float32``, the FF default) the global
       flag also caught the *plain* ``Jt @ J``, assembling the normal
       equations at TF32's 10-bit mantissa with no fp64 anywhere to absorb
       it. Combined with batch-size-dependent kernel selection that made the
       peak fit non-reproducible: 1167 of 8599 peaks moved between two runs
       on identical input (1-ID GE5 FF scan, 2026-07-30).

    So: enable it explicitly, only around the matmul that was designed for
    it, and put it back afterwards.
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


@dataclass
class LMConfig:
    max_iter: int = 200
    ftol_rel: float = 1e-5  # relative cost change
    xtol_rel: float = 1e-5  # relative parameter change
    lambda_init: float = 1e-3
    lambda_inc: float = 10.0
    lambda_dec: float = 10.0
    lambda_max: float = 1e10
    lambda_min: float = 1e-12
    use_analytical_jacobian: bool = True  # 2-3× faster than torch.func.jacrev
    # Triton kernel (5-7× faster than eager analytical Jacobian on H100). Falls
    # back to eager on non-CUDA devices or kernel failures.
    use_triton_jacobian: bool = True
    # Mixed precision JtJ accumulation: cast J to fp32 and use TF32 tensor cores
    # for the J^T J matmul, then back to fp64 for Cholesky. On H100 this
    # gives ~10× speedup on matmul while preserving Cholesky stability.
    # Position error from the precision loss is <0.001 px in our experiments.
    matmul_precision: str = "tf32"  # "fp64" | "tf32"
    # ``torch.compile`` is experimental: when it works it fuses the LM
    # iteration body via CUDA Graphs (1.5-3× speedup), but on some
    # Triton/driver combinations it crashes with InductorError. The
    # default-off setting keeps things robust; flip True to opt in.
    use_torch_compile: bool = False
    # Per-parameter 1-σ uncertainty from σ_r² × (J^T J)^{-1} at convergence.
    # Off by default — enabling adds one extra forward+Jacobian + one
    # Cholesky solve per bucket (~5–10% overhead, payable only when needed).
    compute_uncertainty: bool = False


# Cache compiled residual+Jacobian functions per (n_peaks, dtype, device).
# A compile may fail at first call (Triton on certain GPU/driver combos); in
# that case we fall back permanently to eager so we don't keep retrying.
_COMPILED_RESJAC: dict = {}
_COMPILE_BLACKLIST: set = set()

# ``mode="reduce-overhead"`` compiles via CUDA Graphs, and each compiled
# variant retains its OWN private CUDA memory pool for the process lifetime —
# this cache is never evicted. One variant is created per distinct ``n_peaks``,
# so shape-diverse data silently pins VRAM in proportion to that diversity:
# Ni FF box-beam produced 400 distinct n_peaks (1..400) ≈ 41GB pinned and OOM'd
# *regardless of batch size* (it still died at capacity=1). Dense pf scans have
# ~106 and survived, which is why this never showed up before.
#
# Cap the number of graph-compiled variants: the most frequent shapes (compiled
# first, and they dominate region counts) keep the CUDA-graph speedup, and the
# long tail of rare shapes runs eager with no pinned pool.
_MAX_COMPILED_VARIANTS = 32


def _get_compiled_resjac(n_peaks: int, sample: torch.Tensor):
    """Return a wrapped residuals_and_jacobian_u that:
      - tries the torch.compile-d path first
      - falls back to eager (and caches that decision) on any compile or
        runtime failure (e.g. ``RuntimeError: PassManager::run failed``).
      - falls back to eager once ``_MAX_COMPILED_VARIANTS`` graph variants
        exist, to bound retained CUDA-graph memory pools.
    """
    key = (n_peaks, sample.dtype, str(sample.device))
    fn = _COMPILED_RESJAC.get(key)
    if fn is not None:
        return fn

    if key in _COMPILE_BLACKLIST:
        _COMPILED_RESJAC[key] = residuals_and_jacobian_u
        return residuals_and_jacobian_u

    n_compiled = sum(1 for v in _COMPILED_RESJAC.values()
                     if v is not residuals_and_jacobian_u)
    if n_compiled >= _MAX_COMPILED_VARIANTS:
        _COMPILED_RESJAC[key] = residuals_and_jacobian_u
        return residuals_and_jacobian_u

    try:
        compiled = torch.compile(
            residuals_and_jacobian_u,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
        )
    except Exception as e:
        print(f"[lm] torch.compile factory failed ({type(e).__name__}); using eager.")
        _COMPILE_BLACKLIST.add(key)
        _COMPILED_RESJAC[key] = residuals_and_jacobian_u
        return residuals_and_jacobian_u

    def _wrapped(*args, **kwargs):
        # Persistent runtime fallback. After one failure the wrapper
        # rebinds to eager for this (n_peaks, dtype, device).
        try:
            return compiled(*args, **kwargs)
        except Exception as e:
            print(
                f"[lm] torch.compile runtime failure ({type(e).__name__}: {e!s:.80}); "
                f"falling back to eager for n_peaks={n_peaks}, "
                f"dtype={sample.dtype}, device={sample.device}."
            )
            _COMPILE_BLACKLIST.add(key)
            _COMPILED_RESJAC[key] = residuals_and_jacobian_u
            return residuals_and_jacobian_u(*args, **kwargs)

    _COMPILED_RESJAC[key] = _wrapped
    return _wrapped


def _residuals_per_region(u_b, lo_b, hi_b, z_b, Rs_b, Etas_b, mask_b, n_peaks):
    """Per-region residual function suitable for vmap+jacrev.

    All inputs are 1D (no batch dim); we reintroduce a batch of 1 for the
    model call, then squeeze.
    """
    x_b = u_to_x(u_b, lo_b, hi_b)
    return residuals(
        x_b.unsqueeze(0),
        z_b.unsqueeze(0),
        Rs_b.unsqueeze(0),
        Etas_b.unsqueeze(0),
        mask_b.unsqueeze(0),
        n_peaks,
    ).squeeze(0)


def _make_jac_fn(n_peaks: int):
    """Vectorized Jacobian dr/du across the batch dim."""
    return vmap(
        jacrev(_residuals_per_region, argnums=0),
        in_dims=(0, 0, 0, 0, 0, 0, 0, None),
    )


def _make_res_fn(n_peaks: int):
    """Vectorized residual evaluation across the batch dim."""
    return vmap(_residuals_per_region, in_dims=(0, 0, 0, 0, 0, 0, 0, None))


def lm_solve(
    x_init: torch.Tensor,  # [B, N] in bounded space
    lo: torch.Tensor,  # [B, N]
    hi: torch.Tensor,  # [B, N]
    z: torch.Tensor,  # [B, M]
    Rs: torch.Tensor,  # [B, M]
    Etas: torch.Tensor,  # [B, M]
    pixel_mask: torch.Tensor,  # [B, M]
    n_peaks: int,
    config: LMConfig = LMConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched LM. Returns ``(x_final, cost_final, return_code, sigma_x)``.

    ``return_code``: int tensor of shape ``[B]`` — 0 = converged, 1 = max-iter,
    2 = lambda saturated (effectively diverged).

    ``sigma_x``: ``[B, N]`` per-parameter 1-σ in bounded space when
    ``config.compute_uncertainty`` is True, else a zero-shape sentinel
    (``torch.empty(B, 0)``) — callers should check ``.numel()``.
    """
    device = x_init.device
    dtype = x_init.dtype
    B, N = x_init.shape

    u = x_to_u(x_init, lo, hi).clone()
    res_fn = _make_res_fn(n_peaks)
    jac_fn = None if config.use_analytical_jacobian else _make_jac_fn(n_peaks)
    # torch.compile gives the biggest win via CUDA Graph fusion; on CPU
    # macOS the inductor C++ backend trips on g++ vs clang flag mismatch,
    # so restrict to CUDA where the fp + perf payoff is.
    if (
        config.use_analytical_jacobian
        and config.use_triton_jacobian
        and _TRITON_OK
        and u.device.type == "cuda"
    ):
        resjac_fn = residuals_and_jacobian_u_triton
    elif (
        config.use_analytical_jacobian
        and config.use_torch_compile
        and u.device.type == "cuda"
    ):
        resjac_fn = _get_compiled_resjac(n_peaks, u)
    else:
        resjac_fn = residuals_and_jacobian_u

    # Initial residual + cost
    r = res_fn(u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks)
    cost = (r * r).sum(dim=-1)  # [B]
    lam = torch.full((B,), config.lambda_init, dtype=dtype, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    saturated = torch.zeros(B, dtype=torch.bool, device=device)

    eye = torch.eye(N, dtype=dtype, device=device)

    # Optional per-iteration cost trace (for the convergence-figure paper
    # plot). Activated by setting ``config._tracer`` to an LMTracer
    # instance; zero overhead when not present.
    _tracer = getattr(config, "_tracer", None)
    _trace_key = (n_peaks, int(z.shape[-1]), int(B))
    if _tracer is not None:
        _tracer.begin(_trace_key, int(B), int(config.max_iter))
        _tracer.log(_trace_key, 0, cost, int(B))

    for it in range(config.max_iter):
        active = ~(converged | saturated)
        if not active.any():
            break

        # Compute Jacobian only on active regions to save flops.
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        u_a = u[idx]
        lo_a = lo[idx]
        hi_a = hi[idx]
        z_a = z[idx]
        Rs_a = Rs[idx]
        Etas_a = Etas[idx]
        mask_a = pixel_mask[idx]
        cost_a = cost[idx]
        lam_a = lam[idx]

        if config.use_analytical_jacobian:
            r_a, J = resjac_fn(
                u_a, lo_a, hi_a, z_a, Rs_a, Etas_a, mask_a, n_peaks
            )
        else:
            J = jac_fn(u_a, lo_a, hi_a, z_a, Rs_a, Etas_a, mask_a, n_peaks)
            r_a = res_fn(u_a, lo_a, hi_a, z_a, Rs_a, Etas_a, mask_a, n_peaks)

        # Mixed precision J^T J matmul. On H100, TF32 tensor cores deliver
        # ~989 TFLOPs vs 67 TFLOPs for fp64 — ~10× headroom for the
        # dominant J^T J cost. Precision impact: matmul accumulates in
        # fp32, costing ~7 decimal digits of accuracy in JtJ entries.
        # Cholesky and the final delta solve still use fp64 (we cast back)
        # so the LM trajectory remains numerically stable. Empirically the
        # δ vector deviates by <1e-6 relative per iter, which the next
        # iter corrects.
        #
        # NOTE the fp64 guard: this trade is only sound when the LM itself is
        # in fp64, because the Cholesky and delta solve below then absorb the
        # matmul's precision loss. In an fp32 LM there is nothing to absorb
        # it, so the else-branch must run at true fp32 — see ``_tf32``.
        if (
            config.matmul_precision == "tf32"
            and J.device.type == "cuda"
            and J.dtype == torch.float64
        ):
            with _tf32(True):
                J32 = J.float()
                r32 = r_a.float()
                Jt32 = J32.transpose(-1, -2)
                H = (Jt32 @ J32).double()
                g = (Jt32 @ r32.unsqueeze(-1)).squeeze(-1).double()
        else:
            Jt = J.transpose(-1, -2)  # [Ba, N, M]
            H = Jt @ J  # [Ba, N, N]
            g = (Jt @ r_a.unsqueeze(-1)).squeeze(-1)  # [Ba, N]

        diag = torch.diagonal(H, dim1=-2, dim2=-1)  # [Ba, N]
        # Marquardt's scaled damping
        damp = lam_a.unsqueeze(-1) * diag
        # Floor damping at λ × eps so a zero-gradient direction still gets
        # some damping (fallback is λ × I).
        damp = damp + (lam_a.unsqueeze(-1) * 1e-9)
        H_damped = H + damp.unsqueeze(-1) * eye  # broadcasts diag onto identity

        # H_damped is symmetric PSD by construction (J^T J + λ × diag), so
        # Cholesky is the right solver. Use ``cholesky_ex`` (returns info
        # without raising) so a per-region failure doesn't kill the batch,
        # and unconditionally pass the result to ``cholesky_solve`` — for
        # any region where Cholesky failed, the resulting delta is garbage
        # but LM's accept-test rejects it next step → λ grows → eventually
        # marked saturated. No GPU→CPU sync per iteration.
        L_chol, _info = torch.linalg.cholesky_ex(H_damped)
        delta = torch.cholesky_solve(-g.unsqueeze(-1), L_chol).squeeze(-1)

        u_new = u_a + delta
        r_new = res_fn(u_new, lo_a, hi_a, z_a, Rs_a, Etas_a, mask_a, n_peaks)
        cost_new = (r_new * r_new).sum(dim=-1)

        accept = cost_new < cost_a  # [Ba]
        # Apply per-region updates
        u_a_next = torch.where(accept.unsqueeze(-1), u_new, u_a)
        cost_a_next = torch.where(accept, cost_new, cost_a)
        lam_a_next = torch.where(
            accept, lam_a / config.lambda_dec, lam_a * config.lambda_inc
        )
        lam_a_next = lam_a_next.clamp(config.lambda_min, config.lambda_max)

        # Convergence test (per accepted region)
        rel_cost = (cost_a - cost_a_next).abs() / cost_a.clamp(min=1e-12)
        u_norm = u_a.norm(dim=-1).clamp(min=1e-12)
        d_norm = (u_a_next - u_a).norm(dim=-1)
        rel_x = d_norm / u_norm
        conv = accept & (rel_cost < config.ftol_rel) & (rel_x < config.xtol_rel)
        sat = lam_a_next >= (config.lambda_max * 0.5)

        # Scatter back
        u[idx] = u_a_next
        cost[idx] = cost_a_next
        lam[idx] = lam_a_next
        # Update flags
        full_conv = torch.zeros(B, dtype=torch.bool, device=device)
        full_conv[idx] = conv
        converged = converged | full_conv
        full_sat = torch.zeros(B, dtype=torch.bool, device=device)
        full_sat[idx] = sat
        saturated = saturated | full_sat

        if _tracer is not None:
            _tracer.log(_trace_key, it + 1, cost, int(active.sum().item()))

    x_final = u_to_x(u, lo, hi)
    rc = torch.zeros(B, dtype=torch.int32, device=device)
    rc[~converged] = 1  # max-iter
    rc[saturated] = 2

    if config.compute_uncertainty:
        # One extra forward+Jacobian pass at the converged u to assemble the
        # un-damped J^T J. We reuse the same resjac path (Triton kernel when
        # available) so the precision matches the LM trajectory.
        if config.use_analytical_jacobian:
            r_final, J_final = resjac_fn(
                u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks
            )
        else:
            J_final = jac_fn(u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks)
            r_final = res_fn(u, lo, hi, z, Rs, Etas, pixel_mask, n_peaks)
        sigma_x, _ = compute_param_sigma(
            u, lo, hi, J_final, r_final, pixel_mask
        )
    else:
        sigma_x = torch.empty(B, 0, dtype=dtype, device=device)
    return x_final, cost, rc, sigma_x


__all__ = ["LMConfig", "lm_solve"]
