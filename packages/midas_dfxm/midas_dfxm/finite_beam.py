"""Native finite-beam forward: integrate the DFXM observable along the diffracted ray.

Real DFXM sums the diffraction response along the diffracted ray through the finite beam
thickness (Poulsen 2021 Eqs. 49-57; the source of the Henningsson 2025 back-propagation
bias). This module makes that a native, differentiable part of the forward: for each
voxel the observable is the beam-profile-weighted average of the point observable over a
line segment ``r0 + s * ray_dir``, ``s ~ N(0, sigma_z)``.

Because the field is evaluated at the shifted positions, a curved deformation along the
ray produces exactly the ``~ 0.5 * d^2Q/ds^2 * sigma_z^2`` bias of Henningsson's Appendix
D. Crucially, when the inverse fits a **spatial** model (a dislocation, or a smooth field)
through this beam-integrated forward, the curvature is constrained by the surrounding
voxels, so the bias is removed from a single beam thickness -- the mitigation a
closed-form point back-projection cannot do.

Everything torch-differentiable and device/dtype-portable.
"""
from __future__ import annotations

import torch

from .field_inverse import deformation_observable
from .optics import diffracted_beam_direction


def beam_integrated_observable(
    field_fn,
    positions: torch.Tensor,
    reflections,
    *,
    sigma_z: float,
    ray_dir=None,
    two_theta_deg: float | None = None,
    n_samples: int = 21,
    span: float = 3.0,
) -> torch.Tensor:
    """Beam-integrated per-voxel shifts ``(N, 3m)``: average ``dQ`` along the diffracted ray.

    ``field_fn(pts) -> DeformationField`` evaluates the deformation field at arbitrary
    positions (e.g. an analytic Stroh dislocation, or an interpolated voxel field).
    ``sigma_z`` is the beam thickness; the ray direction defaults to the diffracted-beam
    direction from ``two_theta_deg`` (or pass ``ray_dir`` explicitly). Differentiable in
    the field parameters. As ``sigma_z -> 0`` this reduces to
    :func:`midas_dfxm.deformation_observable`.
    """
    device, dtype = positions.device, positions.dtype
    if ray_dir is None:
        if two_theta_deg is None:
            ray_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        else:
            ray_dir = diffracted_beam_direction(two_theta_deg, device=device, dtype=dtype)
    ray_dir = torch.as_tensor(ray_dir, device=device, dtype=dtype)
    ray_dir = ray_dir / torch.linalg.vector_norm(ray_dir)
    if sigma_z <= 0:
        return deformation_observable(field_fn(positions), reflections)

    s = torch.linspace(-span * sigma_z, span * sigma_z, n_samples, device=device, dtype=dtype)
    w = torch.exp(-0.5 * (s / sigma_z) ** 2)
    w = w / w.sum()
    acc = None
    for si, wi in zip(s, w):
        obs = deformation_observable(field_fn(positions + si * ray_dir), reflections)
        acc = wi * obs if acc is None else acc + wi * obs
    return acc
