"""Fisher-rank diagnostic for any user-chosen parameter block.

The paper's headline claim: HEDM evidence breaks the rank-1 degeneracy of
per-panel ``(δy, δz)`` that paper-3 §9 proves for single-image powder
calibration.  This module quantifies it: given a residual closure (powder-
only, HEDM-only, or joint), build the Fisher block on a chosen subset of
refined parameters and report its rank, condition number, and σ per
parameter.

Generalised: the block can be any list of parameter names, not just panel
shifts.  The user might ask "does my data identify Lsd?" or "does the
joint loss break the (Lsd, pxY) multiplicative gauge?" — same machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.func import jacfwd, jacrev

from midas_peakfit import (
    ParameterSpec,
    pack_spec,
    refined_bounds,
    refined_indices,
    unpack_spec,
    write_refined_back,
    x_to_u, u_to_x,
)


@dataclass
class FisherBlockReport:
    """Diagnostic on a subset of refined parameters under one residual.

    Attributes
    ----------
    block_names
        Parameter names whose Fisher block was extracted.
    block_indices
        Indices of those parameters within the full refined-parameter
        vector.
    fisher
        Fisher matrix restricted to the block (block_size, block_size).
    rank
        Numerical rank computed via SVD with ``rtol = 1e-8``.
    condition_number
        ``λ_max / max(λ_min, 1e-30)``.
    sigma_per_dim
        Marginal σ from the diagonal of the *full* Fisher inverse,
        sliced to the block.  Reflects the data identifiability of each
        individual parameter accounting for cross-correlations with
        non-block parameters.
    nullspace_directions
        Unit vectors in block-coordinates spanning the null space of the
        block (rank-deficient directions).  Empty if the block is full
        rank.
    """
    block_names: List[str]
    block_indices: List[int]
    fisher: torch.Tensor
    rank: int
    condition_number: float
    sigma_per_dim: torch.Tensor
    nullspace_directions: torch.Tensor   # (n_null, block_size)


def fisher_block_rank(
    spec: ParameterSpec,
    residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    map_unpacked: Dict[str, torch.Tensor],
    block_names: Sequence[str],
    *,
    sigma_r: float = 1.0,
    fallback_span: float = 1.0,
    rtol: float = 1e-8,
    ridge: float = 0.0,
    dtype=torch.float64,
    device="cpu",
) -> FisherBlockReport:
    """Build the Fisher block on ``block_names`` and report rank / σ.

    The Fisher matrix is computed on the *full* refined-parameter vector
    (in u-space), then sliced to the block.  Marginal σ comes from the
    diagonal of the inverse of the full Fisher, sliced to the block —
    this is the standard "marginal posterior σ" that paper-3 reports.

    Pass three different ``residual_fn`` closures (powder-only,
    HEDM-only, joint) to produce the headline three-column comparison.

    The ``rtol`` default ``1e-8`` is consistent with a well-conditioned
    Gauss-Newton model; for ill-conditioned blocks (e.g. single-image
    powder per-panel) the rank-1 deficiency is usually obvious by an
    eigenvalue gap of ~12–14 orders of magnitude.
    """
    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)

    x_full_map = x_full.clone()
    for name, val in map_unpacked.items():
        sl = info.slice(name)
        x_full_map[sl] = val.detach().to(dtype=dtype, device=device).reshape(-1)
    x_ref_map = x_full_map.index_select(0, refined_idx)
    u_map = x_to_u(x_ref_map, lo, hi)

    def r_of_u(u: torch.Tensor) -> torch.Tensor:
        x_ref = u_to_x(u, lo, hi)
        x_full_now = write_refined_back(x_full_map, x_ref, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        return residual_fn(unpacked)

    n = int(u_map.numel())
    m_est = int(r_of_u(u_map.detach().clone()).numel())
    if n < m_est:
        J = jacfwd(r_of_u)(u_map.detach().clone())
    else:
        J = jacrev(r_of_u)(u_map.detach().clone())
    JtJ = J.transpose(0, 1) @ J
    F_full = JtJ / (sigma_r ** 2)
    if ridge > 0.0:
        F_full = F_full + ridge * torch.eye(n, dtype=dtype, device=device)

    # Build refined-parameter name → block-of-indices-in-u map.
    cur = 0
    name_to_uidx: Dict[str, List[int]] = {}
    for nm, sz, ref in zip(info.names, info.sizes, info.refined):
        if not ref:
            continue
        name_to_uidx[nm] = list(range(cur, cur + sz))
        cur += sz

    block_idx: List[int] = []
    for nm in block_names:
        if nm not in name_to_uidx:
            raise ValueError(
                f"block name {nm!r} is not a refined parameter in the spec; "
                f"refined names are {list(name_to_uidx.keys())}")
        block_idx.extend(name_to_uidx[nm])

    block_idx_t = torch.tensor(block_idx, dtype=torch.long, device=device)
    F_block = F_full.index_select(0, block_idx_t).index_select(1, block_idx_t)

    # Rank + condition + nullspace (in u-space, but the block-internal
    # null directions transfer to bounded-x via the local Jacobian which
    # is a positive diagonal scaling — null directions stay null).
    eigvals, eigvecs = torch.linalg.eigh(F_block)
    pos_eigvals = eigvals.clamp(min=0.0)
    max_eig = float(pos_eigvals.max())
    rank = int(((pos_eigvals / max(max_eig, 1e-300)) > rtol).sum())
    cond = float(max_eig / max(float(pos_eigvals.min()), 1e-300))
    null_mask = (pos_eigvals / max(max_eig, 1e-300)) <= rtol
    null_dirs = eigvecs[:, null_mask].transpose(0, 1)

    # Marginal σ from the full-Fisher inverse (Moore-Penrose for safety).
    F_inv = torch.linalg.pinv(F_full + 1e-30 * torch.eye(n, dtype=dtype, device=device))
    diag_full = torch.diag(F_inv).clamp(min=0.0).sqrt()
    # Project u-space σ to bounded-x σ via local Jacobian, sliced to block.
    s = torch.sigmoid(u_map)
    dxdu = (hi - lo) * s * (1.0 - s)
    sigma_x = (diag_full * dxdu).index_select(0, block_idx_t)

    return FisherBlockReport(
        block_names=list(block_names),
        block_indices=block_idx,
        fisher=F_block.detach(),
        rank=rank,
        condition_number=cond,
        sigma_per_dim=sigma_x.detach(),
        nullspace_directions=null_dirs.detach(),
    )


__all__ = ["FisherBlockReport", "fisher_block_rank"]
