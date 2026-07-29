"""Magnifying objective optics: sample voxel -> detector pixel projection.

Phase 1 of ``implementation_plan.md`` — the NEW real-space imaging geometry.

The objective images the illuminated sample volume along the diffracted-beam
optical axis ``k_out`` with magnification ``M`` onto a detector perpendicular to
``k_out``. A sample voxel at lab position ``r`` maps to detector coordinates by
projecting ``r`` onto the plane transverse to ``k_out`` (an orthonormal detector
basis ``(u, v)``), scaling by ``M``, and converting to pixels. The imaging is an
*inclined* projection because the sample plane is tilted relative to the optical
axis — captured here by the choice of ``k_out`` (from ``2*theta``) rather than by
an ad-hoc tilt.

Differentiable in positions, magnification, and ``k_out`` direction; device/dtype
portable. Detector distortion, when needed, composes via ``midas_distortion``
(single source of truth) — omitted from the ideal Phase-1 projection.

Units: positions in micrometers; pixel size in micrometers; pixels dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def diffracted_beam_direction(two_theta_deg: float, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Unit ``k_out`` for a reflection at ``2*theta`` in the vertical diffraction plane.

    Incident beam along lab ``x``; diffraction plane is ``x``-``z`` (vertical).
    ``k_out = (cos 2theta) xhat + (sin 2theta) zhat``.
    """
    tt = torch.deg2rad(torch.as_tensor(two_theta_deg, device=device, dtype=dtype))
    return torch.stack([torch.cos(tt), torch.zeros_like(tt), torch.sin(tt)])


def detector_basis(k_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal detector axes ``(u, v)`` spanning the plane transverse to ``k_out``.

    ``u`` is horizontal (in the lab ``x``-``y`` sense), ``v`` completes a
    right-handed set. Returns two ``(3,)`` unit vectors.
    """
    k = k_out / torch.linalg.vector_norm(k_out)
    up = torch.tensor([0.0, 0.0, 1.0], device=k.device, dtype=k.dtype)
    if torch.abs(torch.dot(k, up)) > 0.9:
        up = torch.tensor([0.0, 1.0, 0.0], device=k.device, dtype=k.dtype)
    u = torch.linalg.cross(up, k)
    u = u / torch.linalg.vector_norm(u)
    v = torch.linalg.cross(k, u)
    return u, v


@dataclass
class ObjectiveOptics:
    """Magnifying objective imaging model.

    Attributes
    ----------
    two_theta_deg : float
        Bragg ``2*theta`` of the imaged reflection (sets the optical axis ``k_out``).
    magnification : float
        Transverse magnification ``M`` of the objective.
    pixel_um : float
        Detector pixel size in micrometers.
    detector_shape : tuple[int, int]
        ``(n_u, n_v)`` detector dimensions in pixels.
    center_px : tuple[float, float]
        Pixel coordinate of the optical axis (defaults to detector centre).
    """

    two_theta_deg: float
    magnification: float = 10.0
    pixel_um: float = 1.0
    detector_shape: tuple[int, int] = (256, 256)
    center_px: tuple[float, float] | None = None

    def project(self, positions_lab: torch.Tensor) -> torch.Tensor:
        """Project lab-frame voxel positions ``(N, 3)`` to pixel coords ``(N, 2)``.

        Returns fractional ``(u_px, v_px)``. Differentiable in positions and ``M``.
        """
        k_out = diffracted_beam_direction(
            self.two_theta_deg, device=positions_lab.device, dtype=positions_lab.dtype
        )
        u, v = detector_basis(k_out)
        cu, cv = self._center(positions_lab)
        pu = self.magnification * (positions_lab @ u) / self.pixel_um + cu
        pv = self.magnification * (positions_lab @ v) / self.pixel_um + cv
        return torch.stack([pu, pv], dim=-1)

    def _center(self, ref: torch.Tensor) -> tuple[float, float]:
        if self.center_px is not None:
            return self.center_px
        nu, nv = self.detector_shape
        return (nu - 1) / 2.0, (nv - 1) / 2.0

    def render(self, positions_lab: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """Accumulate per-voxel ``intensity`` onto the detector via bilinear splat.

        Returns an image ``(n_u, n_v)``. Differentiable in ``intensity`` and
        positions (bilinear weights carry gradients). Voxels off the detector are
        dropped.
        """
        px = self.project(positions_lab)
        nu, nv = self.detector_shape
        img = torch.zeros(nu, nv, device=positions_lab.device, dtype=intensity.dtype)
        u = px[:, 0]
        v = px[:, 1]
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        du = u - u0
        dv = v - v0
        for iu, ju, wu in ((u0, 0, 1 - du), (u0 + 1, 1, du)):
            for iv, jv, wv in ((v0, 0, 1 - dv), (v0 + 1, 1, dv)):
                inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
                if not bool(inb.any()):
                    continue
                w = (wu * wv * intensity)[inb]
                flat = (iu[inb] * nv + iv[inb])
                img.view(-1).index_add_(0, flat, w)
        return img
