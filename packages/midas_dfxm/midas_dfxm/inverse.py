"""Field inverse: recover the strain tensor field from multi-reflection DFXM.

Phase 3 of ``implementation_plan.md``.

A DFXM *strain scan* in reflection ``g`` measures, per voxel, the normal strain
along ``g``: ``eps_gg = ghat^T eps ghat`` (the fractional d-spacing change). One
reflection constrains one linear combination of the six independent strain
components; recovering the **full strain tensor field** requires a diverse set of
reflections. This module builds that linear inverse and its regularised,
uncertainty-quantified differentiable counterpart.

Honest scope (see the novelty discussion in the plan):

* For **clean, well-sampled** data the per-voxel recovery is a plain linear
  least-squares solve (:func:`recover_strain_direct`) — differentiability buys
  nothing there, and we say so.
* Differentiability earns its keep when the inverse is **coupled across voxels**
  (spatial regularisation / compatibility priors — :func:`recover_strain_regularised`),
  carries **uncertainty** (:func:`strain_covariance`), informs **experiment design**
  (:func:`strain_identifiability`, reusing ``midas_invert`` Fisher-info), or feeds a
  **nonlinear full forward** (overlapping signals, joint instrument+field, physics
  models) where no closed form exists.

Reuses ``midas_invert`` for the fit loop and ``midas_dfxm.field`` for the geometry.
Everything is torch-differentiable and device-portable.

Units: strain dimensionless; reciprocal vectors carry the ``2*pi`` convention.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit

from .field import DeformationField, reciprocal_basis, small_strain_from_F

# Voigt-6 order used here: [eps11, eps22, eps33, eps23, eps13, eps12] (symmetric,
# no sqrt(2) / factor-2 convention baked in — the design-matrix rows carry the 2x
# on shear terms explicitly).


def normal_strain(field: DeformationField, hkl) -> torch.Tensor:
    """Per-voxel normal strain along ``g`` — the DFXM strain-scan observable.

    The strain scan measures the fractional d-spacing change along ``g``,
    ``eps_gg = ghat^T eps ghat`` to first order, with ``eps = sym(F) - I`` the
    (small) strain tensor and ``ghat`` the reflection's sample-frame unit vector.
    This linear form is what the design matrix inverts, so recovery is exact for
    clean data. Returns ``(N,)``. Differentiable in the field.
    """
    G0 = field.reference_G(hkl)
    ghat = G0 / torch.linalg.vector_norm(G0)
    eps = small_strain_from_F(field.F)              # (N, 3, 3)
    return torch.einsum("i,nij,j->n", ghat, eps, ghat)


def _ghat_sample(hkl, orientation, lattice_params) -> torch.Tensor:
    """Unit reflection vector ``ghat`` in the sample frame for ``hkl``."""
    hkl_t = torch.as_tensor(hkl, dtype=lattice_params.dtype, device=lattice_params.device)
    B = reciprocal_basis(lattice_params)
    G = orientation @ (B @ hkl_t)
    return G / torch.linalg.vector_norm(G)


def strain_design_row(ghat: torch.Tensor) -> torch.Tensor:
    """Design row mapping the Voigt-6 strain to ``eps_gg`` for one reflection.

    ``eps_gg = ghat^T eps ghat = [g1^2, g2^2, g3^2, 2 g2 g3, 2 g1 g3, 2 g1 g2] . eps6``.
    """
    g = ghat
    return torch.stack([
        g[0] ** 2, g[1] ** 2, g[2] ** 2,
        2 * g[1] * g[2], 2 * g[0] * g[2], 2 * g[0] * g[1],
    ])


def strain_design_matrix(
    reflections,
    *,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Design matrix ``M`` (K, 6): row ``k`` maps Voigt-6 strain to ``eps_gg`` for ``g_k``.

    ``orientation`` is the reference ``OM0`` (crystal->sample); defaults to identity.
    Differentiable in the lattice/orientation.
    """
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    if orientation is None:
        orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        orientation = torch.as_tensor(orientation, device=device, dtype=dtype)
    rows = [strain_design_row(_ghat_sample(hkl, orientation, latc)) for hkl in reflections]
    return torch.stack(rows, dim=0)


def strain_identifiability(reflections, **kw) -> dict:
    """Rank / conditioning of a reflection set for full strain-tensor recovery.

    Returns ``{'rank', 'cond', 'singular_values', 'recoverable'}`` where
    ``recoverable`` is ``True`` iff all six strain components are constrained
    (rank 6). Use to choose reflections before beamtime (cf. ``midas_invert``
    experiment design).
    """
    M = strain_design_matrix(reflections, **kw)
    s = torch.linalg.svdvals(M)
    rank = int((s > s.max() * 1e-10).sum())
    cond = float(s.max() / s.min()) if rank == 6 else float("inf")
    return {
        "rank": rank,
        "cond": cond,
        "singular_values": s,
        "recoverable": rank == 6,
    }


def recover_strain_direct(measured: torch.Tensor, reflections, **kw) -> torch.Tensor:
    """Per-voxel least-squares strain tensor from normal-strain maps.

    ``measured`` is ``(K, N)`` (K reflections, N voxels). Returns ``(N, 6)`` Voigt-6.
    The plain linear solve — no regularisation, no cross-voxel coupling. Exact for
    clean data with a rank-6 reflection set; noise passes straight through.
    """
    M = strain_design_matrix(reflections, dtype=measured.dtype, device=measured.device, **kw)
    sol = torch.linalg.lstsq(M, measured).solution  # (6, N)
    return sol.transpose(0, 1)


def strain_covariance(reflections, *, noise_std: float = 1.0, **kw) -> torch.Tensor:
    """Per-voxel strain covariance ``(M^T M)^-1 sigma^2`` for a reflection set.

    Linear-Gaussian uncertainty propagation: the diagonal gives per-component
    strain variances, off-diagonals the correlations induced by the reflection
    geometry. Returns ``(6, 6)``. (The differentiable/nonlinear analogue is a
    ``midas_invert`` Laplace posterior around the fit.)
    """
    M = strain_design_matrix(reflections, **kw)
    return torch.linalg.inv(M.transpose(0, 1) @ M) * (noise_std ** 2)


def _curvature_penalty(field6: torch.Tensor, shape) -> torch.Tensor:
    """Second-difference (curvature) smoothness penalty over the voxel grid.

    Penalises the discrete Laplacian ``v[i+1] - 2 v[i] + v[i-1]`` along each grid
    axis, for all six strain components. Unlike total variation this leaves smooth
    (locally linear) fields unbiased while suppressing high-frequency noise — the
    right prior when the underlying strain field is band-limited (elastic).
    """
    nx, ny, nz = shape
    vol = field6.reshape(nx, ny, nz, 6)
    pen = torch.zeros((), device=field6.device, dtype=field6.dtype)
    if nx > 2:
        pen = pen + (vol[2:] - 2 * vol[1:-1] + vol[:-2]).pow(2).mean()
    if ny > 2:
        pen = pen + (vol[:, 2:] - 2 * vol[:, 1:-1] + vol[:, :-2]).pow(2).mean()
    if nz > 2:
        pen = pen + (vol[:, :, 2:] - 2 * vol[:, :, 1:-1] + vol[:, :, :-2]).pow(2).mean()
    return pen


def recover_strain_regularised(
    measured: torch.Tensor,
    reflections,
    shape,
    *,
    lambda_smooth: float = 1e-3,
    steps: int = 400,
    lr: float = 1e-2,
    init=None,
    **kw,
) -> torch.Tensor:
    """Spatially-regularised strain-tensor field recovery (differentiable).

    Minimises ``||M eps - measured||^2 + lambda_smooth * curvature(eps)`` over the
    whole field jointly — the cross-voxel coupling a per-voxel solve cannot express.
    Returns ``(N, 6)``. Under realistic noise / limited reflections this lowers the
    recovery error vs :func:`recover_strain_direct` (demonstrated in the example).
    Reuses ``midas_invert.optimize.fit``.
    """
    M = strain_design_matrix(reflections, dtype=measured.dtype, device=measured.device, **kw)
    if init is None:
        init = recover_strain_direct(measured, reflections, **kw).detach().clone()
    eps6 = init.clone().requires_grad_(True)

    def loss_fn():
        pred = eps6 @ M.transpose(0, 1)          # (N, K)
        data = (pred - measured.transpose(0, 1)) ** 2
        return data.mean() + lambda_smooth * _curvature_penalty(eps6, shape)

    fit([eps6], loss_fn, steps=steps, lr=lr)
    return eps6.detach()
