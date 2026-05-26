"""midas_propagate.propagate — delta-method from grain Σ to per-grain stress Σ.

Final paper-1 step. Given the per-grain marginal covariance
:class:`midas_propagate.schur.PerGrainMarginalResult.sigma_gg_calmarg`
(shape ``(G, 12, 12)`` on ``(euler[3], latc[6], pos[3])``) and a known
single-crystal stiffness, this module returns the per-grain Cauchy
stress tensor AND its 6×6 Voigt-Mandel covariance via the delta
method::

    Σ_stress = J · Σ_g · J^T,
    J = ∂σ_voigt(g) / ∂g  evaluated at MAP.

The forward map is built from primitives already validated in
``midas_stress`` (``hooke_stress`` on the torch backend); we only add the
lattice→strain conversion and the Jacobian batching.

Strain convention: standard small-strain Lagrangian, ``ε = sym(F − I)``
where ``F = L_grain · L_ref^{-1}`` and ``L`` is the column-stacked
lattice-vector matrix from the (a, b, c, α, β, γ) parameters. Same
convention as ``midas_stress.diffraction`` and as the FF-HEDM grain
output. Position (``pos``) drops out of the stress map; its three
columns of J are exactly zero — propagation correctly leaves stress
independent of grain position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch


_DEG2RAD = math.pi / 180.0


@dataclass
class PerGrainStressResult:
    """Output of :func:`per_grain_stress_with_cov`.

    Attributes
    ----------
    stress_voigt : (G, 6) tensor
        Per-grain Cauchy stress in Voigt-Mandel order
        ``[xx, yy, zz, sqrt(2)·xy, sqrt(2)·xz, sqrt(2)·yz]``, lab frame.
    stress_cov : (G, 6, 6) tensor
        Per-grain stress covariance in the same Voigt-Mandel ordering.
    sigma_voigt : (G, 6) tensor
        Per-grain marginal 1σ per stress component (=√diag(cov)).
    """
    stress_voigt: torch.Tensor
    stress_cov: torch.Tensor
    sigma_voigt: torch.Tensor


def lattice_vectors(latc: torch.Tensor) -> torch.Tensor:
    """Column-stacked lattice-vector matrix L from ``(a, b, c, α, β, γ)``.

    Standard crystallographic convention: ``a`` along x, ``b`` in the
    xy plane, ``c`` filling out the right-handed cell. Differentiable.
    ``latc`` may be batched (any leading shape); ``latc[..., 3:6]`` are
    angles in degrees.

    Returns ``(..., 3, 3)`` where columns are (a, b, c) in the crystal
    reference frame.
    """
    a = latc[..., 0]
    b = latc[..., 1]
    c = latc[..., 2]
    alpha = latc[..., 3] * _DEG2RAD
    beta = latc[..., 4] * _DEG2RAD
    gamma = latc[..., 5] * _DEG2RAD

    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    cg = torch.cos(gamma)
    sg = torch.sin(gamma)

    # a vector along x.
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)
    # b vector in xy plane.
    bx = b * cg
    by = b * sg
    bz = torch.zeros_like(b)
    # c vector — third row from the metric tensor.
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = (1.0
             - cb * cb
             - ((ca - cb * cg) / sg) ** 2)
    cz = c * torch.sqrt(cz_sq.clamp(min=1e-30))

    L = torch.stack([
        torch.stack([ax, bx, cx], dim=-1),
        torch.stack([ay, by, cy], dim=-1),
        torch.stack([az, bz, cz], dim=-1),
    ], dim=-2)
    return L


def latc_to_strain_grain(
    latc: torch.Tensor,
    latc_ref: torch.Tensor,
) -> torch.Tensor:
    """Lagrangian small strain from grain lattice vs reference, in **grain frame**.

    Returns ``(..., 3, 3)`` symmetric tensor. ``ε = sym(F − I)`` with
    ``F = L_grain · L_ref⁻¹``. Differentiable. Both inputs use the
    ``(a, b, c, α, β, γ)`` layout (lengths in Å, angles in degrees).
    """
    L_g = lattice_vectors(latc)
    L_r = lattice_vectors(latc_ref)
    L_r_inv = torch.linalg.inv(L_r)
    F = L_g @ L_r_inv
    I = torch.eye(3, dtype=F.dtype, device=F.device)
    sym = 0.5 * (F + F.transpose(-1, -2)) - I
    return sym


def euler_zxz_to_matrix(euler_rad: torch.Tensor) -> torch.Tensor:
    """Bunge ZXZ Euler → orientation matrix (3, 3). Differentiable.

    Matches the convention used by ``HEDMForwardModel.euler2mat`` and by
    ``midas_fit_grain``. Active rotation: rotates crystal frame to lab
    frame given (φ1, Φ, φ2).
    """
    phi1 = euler_rad[..., 0]
    Phi = euler_rad[..., 1]
    phi2 = euler_rad[..., 2]
    c1, s1 = torch.cos(phi1), torch.sin(phi1)
    cP, sP = torch.cos(Phi), torch.sin(Phi)
    c2, s2 = torch.cos(phi2), torch.sin(phi2)
    R = torch.stack([
        torch.stack([c1 * c2 - s1 * cP * s2,
                     -c1 * s2 - s1 * cP * c2,
                     s1 * sP], dim=-1),
        torch.stack([s1 * c2 + c1 * cP * s2,
                     -s1 * s2 + c1 * cP * c2,
                     -c1 * sP], dim=-1),
        torch.stack([sP * s2, sP * c2, cP], dim=-1),
    ], dim=-2)
    return R


def _stress_voigt_lab_from_g(
    g_flat: torch.Tensor,         # (12,) — euler[3], latc[6], pos[3]
    latc_ref: torch.Tensor,
    stiffness_voigt: torch.Tensor,  # (6, 6) — crystal frame
) -> torch.Tensor:
    """Per-grain lab-frame Cauchy stress in Voigt-Mandel as a function of
    ``g`` only. Differentiable. ``pos`` is unused (kept in g for shape
    consistency with the rest of paper-1)."""
    from midas_stress.torch_backend import (
        tensor_to_voigt, voigt_to_tensor, rotation_voigt_mandel,
    )
    euler = g_flat[:3]
    latc = g_flat[3:9]
    # pos = g_flat[9:12]  — drops out of stress.
    eps_grain = latc_to_strain_grain(latc, latc_ref)        # (3, 3)
    eps_voigt = tensor_to_voigt(eps_grain)                  # (6,)
    sigma_voigt_grain = stiffness_voigt @ eps_voigt          # (6,)
    R = euler_zxz_to_matrix(euler)                          # (3, 3)
    M = rotation_voigt_mandel(R)                            # (6, 6) lab → grain
    sigma_voigt_lab = M.transpose(-1, -2) @ sigma_voigt_grain
    return sigma_voigt_lab


def per_grain_stress_with_cov(
    g_map: torch.Tensor,           # (G, 12) — MAP states
    sigma_gg: torch.Tensor,        # (G, 12, 12) — from schur (frozen or marg)
    latc_ref: torch.Tensor,        # (6,) — reference lattice
    stiffness_voigt: torch.Tensor, # (6, 6) — single-crystal Cij, Voigt-Mandel
) -> PerGrainStressResult:
    """Per-grain stress + delta-method covariance from grain marginal Σ.

    ``J = ∂σ_voigt_lab / ∂g`` is computed per grain via autograd on the
    closed-form forward map ``g → strain → grain-stress → lab-stress``.
    The Voigt covariance is ``Σ_σ = J · Σ_g · J^T``; per-component σ
    is ``√diag(Σ_σ)`` (clamped to ≥ 0 to absorb fp noise).

    Compatible with both the frozen-cal and calibration-marginalised
    grain covariances returned by
    :func:`midas_propagate.schur.per_grain_schur_marginal`; pass the one
    you want propagated.

    Parameters
    ----------
    g_map : (G, 12)
        MAP state per grain in the layout
        ``[euler_rad[3], latc[6], pos_um[3]]``.
    sigma_gg : (G, 12, 12)
        Per-grain covariance (same parameter layout).
    latc_ref : (6,)
        Reference (unstrained) lattice constants. For Park22 / Ti-7Al:
        the as-measured lattice at the unloaded stage. For synthetic
        tests, the GT lattice.
    stiffness_voigt : (6, 6)
        Single-crystal stiffness ``C`` in Voigt-Mandel, crystal frame,
        in GPa (or any consistent stress unit; result inherits).
    """
    if g_map.ndim != 2 or g_map.shape[-1] != 12:
        raise ValueError(
            f"g_map must be (G, 12); got shape {tuple(g_map.shape)}"
        )
    if sigma_gg.shape[:2] != g_map.shape or sigma_gg.shape[-1] != 12:
        raise ValueError(
            f"sigma_gg must be (G, 12, 12); got shape {tuple(sigma_gg.shape)}"
        )

    G = g_map.shape[0]
    dtype = g_map.dtype
    device = g_map.device
    stress_voigt = torch.empty((G, 6), dtype=dtype, device=device)
    stress_cov = torch.empty((G, 6, 6), dtype=dtype, device=device)

    from torch.func import jacfwd

    def f(g):
        return _stress_voigt_lab_from_g(g, latc_ref, stiffness_voigt)

    for k in range(G):
        g_k = g_map[k].detach().clone()
        # Forward (no grad needed).
        with torch.no_grad():
            stress_voigt[k] = f(g_k)
        # Jacobian via jacfwd (12 forward calls — closed-form fwd is cheap).
        J = jacfwd(f)(g_k)                          # (6, 12)
        S_g = sigma_gg[k]
        S_sig = J @ S_g @ J.transpose(-1, -2)
        stress_cov[k] = 0.5 * (S_sig + S_sig.transpose(-1, -2))

    sigma_voigt = torch.sqrt(
        torch.diagonal(stress_cov, dim1=-2, dim2=-1).clamp(min=0.0)
    )
    return PerGrainStressResult(
        stress_voigt=stress_voigt,
        stress_cov=stress_cov,
        sigma_voigt=sigma_voigt,
    )


__all__ = [
    "PerGrainStressResult",
    "lattice_vectors",
    "latc_to_strain_grain",
    "euler_zxz_to_matrix",
    "per_grain_stress_with_cov",
]
