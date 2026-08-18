"""Ray-to-plane projection: voxels -> detector, for both TT and DCT.

Phase 1 of ``implementation_plan.md``.

One projector serves both techniques, because they differ only in where the
detector sits:

* **TT** -- detector perpendicular to the (fixed) diffracted beam ``k_h``. A
  topography camera.
* **DCT** -- detector perpendicular to the *incident* beam, downstream of the
  sample, carrying both the direct beam and every grain's spots at once.

so :func:`project_rays_to_plane` takes the detector normal as a parameter and
both fall out.

Per-voxel ray directions, and why they matter
---------------------------------------------
Each voxel is propagated along **its own** diffracted direction
``k_h(r) = k_in + Q(r)``, not along a single nominal beam. For an undeformed
grain those coincide and the result is an ordinary parallel projection. Under a
deformation field they do not: a sub-volume whose local ``Q = F^-T G0`` is tilted
lands somewhere else on the detector, so back-projecting it along the nominal
direction places it at the wrong 3-D position. That displacement *is* the
"pseudo distorted grain volume" the literature linearises away, and carrying it
exactly here is what lets Phase 3 invert through it rather than bound it.

Units and normalisation
-----------------------
The projector returns a **line integral**: ``sum_i f_i * V_voxel / A_pixel``, in
units of ``[f] * um``. That makes the output physical rather than
grid-dependent -- for ``f = chi`` (occupancy) a pixel reads the **path length in
micrometers** through the grain, so the centre of a sphere of radius R reads
``2R``, and the total is conserved:

    sum_pixels image * A_pixel  ==  sum_voxels f * V_voxel

exactly (bilinear weights sum to 1), for whatever mass lands on the detector.
Mass that misses the detector is silently dropped by the splat, so
:func:`offdetector_fraction` exists to measure it; call it rather than assuming
a detector is big enough.
"""
from __future__ import annotations

import torch
from midas_dfxm.optics import detector_basis

__all__ = [
    "default_center_px",
    "offdetector_fraction",
    "parallel_projection",
    "pixel_coordinates",
    "project_rays_to_plane",
    "splat_bilinear",
]


def default_center_px(detector_shape) -> tuple:
    """Detector centre in pixel coordinates, matching ``midas_dfxm.ObjectiveOptics``."""
    nu, nv = detector_shape
    return (nu - 1) / 2.0, (nv - 1) / 2.0


def splat_bilinear(u_px, v_px, weights, detector_shape) -> torch.Tensor:
    """Accumulate ``weights`` at fractional pixel coordinates by bilinear splat.

    Returns an image ``(n_u, n_v)``. Differentiable in ``weights`` and in the
    coordinates (the bilinear weights carry gradients). Contributions outside the
    detector are dropped -- see :func:`offdetector_fraction`.

    This mirrors the splat inside :meth:`midas_dfxm.optics.ObjectiveOptics.render`,
    which is not exposed as a standalone function. It is reimplemented rather
    than re-released upstream only because we need it with per-voxel ray
    directions, which ``render`` cannot express; ``tests/test_project.py`` pins
    the two to agree exactly in the case where both apply, so this cannot drift
    into a fork.
    """
    nu, nv = detector_shape
    img = torch.zeros(nu, nv, device=weights.device, dtype=weights.dtype)
    # Keep the autograd graph alive even when every ray misses the detector.
    # Without this the image is a plain zeros tensor with no grad_fn, and a fit
    # whose parameters momentarily throw the spot off the edge dies with
    # "element 0 of tensors does not require grad" rather than simply seeing a
    # zero image. Numerically a no-op.
    if weights.requires_grad:
        img = img + 0.0 * weights.sum()
    u0 = torch.floor(u_px).long()
    v0 = torch.floor(v_px).long()
    du = u_px - u0
    dv = v_px - v0
    for iu, wu in ((u0, 1 - du), (u0 + 1, du)):
        for iv, wv in ((v0, 1 - dv), (v0 + 1, dv)):
            inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
            if not bool(inb.any()):
                continue
            w = (wu * wv * weights)[inb]
            flat = iu[inb] * nv + iv[inb]
            img.view(-1).index_add_(0, flat, w)
    return img


def pixel_coordinates(
    positions_lab: torch.Tensor,
    directions: torch.Tensor,
    *,
    normal: torch.Tensor,
    distance_um,
    pixel_um: float = 1.0,
    detector_shape=(256, 256),
    center_px=None,
) -> torch.Tensor:
    """Where each voxel's ray lands on the detector, in fractional pixels ``(N, 2)``.

    Ray ``i`` starts at ``positions_lab[i]`` and travels along ``directions[i]``
    (need not be normalised) until it meets the plane with unit ``normal`` at
    ``distance_um`` from the origin along that normal::

        t_i = (L - r_i . n) / (d_i . n)
        p_i = r_i + t_i d_i

    Rays travelling away from, or parallel to, the detector produce non-finite
    coordinates; they are pushed far off-detector so the splat drops them
    instead of poisoning the image with NaNs.

    ``directions`` broadcasts: pass a single ``(3,)`` for a common direction or
    ``(N, 3)`` for per-voxel rays.
    """
    n_hat = normal / torch.linalg.vector_norm(normal)
    d = directions
    if d.dim() == 1:
        d = d.unsqueeze(0).expand_as(positions_lab)
    d_hat = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)

    denom = d_hat @ n_hat
    L = torch.as_tensor(distance_um, device=positions_lab.device, dtype=positions_lab.dtype)
    t = (L - positions_lab @ n_hat) / denom
    hit = positions_lab + t.unsqueeze(-1) * d_hat

    u_vec, v_vec = detector_basis(n_hat)
    cu, cv = center_px if center_px is not None else default_center_px(detector_shape)
    u_px = (hit @ u_vec) / pixel_um + cu
    v_px = (hit @ v_vec) / pixel_um + cv

    # A ray that cannot reach the plane must not land anywhere on it.
    bad = ~torch.isfinite(u_px) | ~torch.isfinite(v_px) | (denom <= 0)
    if bool(bad.any()):
        far = float(max(detector_shape)) * 10.0
        u_px = torch.where(bad, torch.full_like(u_px, -far), u_px)
        v_px = torch.where(bad, torch.full_like(v_px, -far), v_px)
    return torch.stack([u_px, v_px], dim=-1)


def project_rays_to_plane(
    positions_lab: torch.Tensor,
    values: torch.Tensor,
    directions: torch.Tensor,
    *,
    normal: torch.Tensor,
    distance_um,
    voxel_volume_um3: float,
    pixel_um: float = 1.0,
    detector_shape=(256, 256),
    center_px=None,
) -> torch.Tensor:
    """Line-integral image ``(n_u, n_v)`` of ``values`` on a plane detector.

    Units are ``[values] * um`` -- see the module docstring. For ``values = chi``
    the image is the path length through the grain in micrometers.
    """
    px = pixel_coordinates(
        positions_lab, directions,
        normal=normal, distance_um=distance_um, pixel_um=pixel_um,
        detector_shape=detector_shape, center_px=center_px,
    )
    img = splat_bilinear(px[:, 0], px[:, 1], values, detector_shape)
    return img * (voxel_volume_um3 / (pixel_um ** 2))


def parallel_projection(
    positions_lab: torch.Tensor,
    values: torch.Tensor,
    *,
    normal: torch.Tensor,
    voxel_volume_um3: float,
    pixel_um: float = 1.0,
    detector_shape=(256, 256),
    center_px=None,
) -> torch.Tensor:
    """Unmagnified parallel projection along ``normal`` -- the ideal topograph.

    The undeformed limit of :func:`project_rays_to_plane`: every ray travels
    along the detector normal, so the detector distance drops out entirely (a
    parallel projection has no magnification and no propagation shift). Identical
    to ``midas_dfxm.ObjectiveOptics(magnification=1.0)`` projection, which
    ``tests/test_project.py`` asserts.
    """
    return project_rays_to_plane(
        positions_lab, values, normal,
        normal=normal, distance_um=0.0, voxel_volume_um3=voxel_volume_um3,
        pixel_um=pixel_um, detector_shape=detector_shape, center_px=center_px,
    )


def offdetector_fraction(
    positions_lab: torch.Tensor,
    values: torch.Tensor,
    directions: torch.Tensor,
    *,
    normal: torch.Tensor,
    distance_um,
    pixel_um: float = 1.0,
    detector_shape=(256, 256),
    center_px=None,
) -> float:
    """Fraction of ``sum(|values|)`` that the splat drops off the detector.

    The splat discards off-detector contributions silently, which reads as "the
    grain is this bright" when it means "the detector is too small". Call this
    whenever a result depends on capturing the whole grain; a topograph with 5%
    of its mass off the edge is not a projection of the grain.

    Measured by running the *same* splat and comparing what survives, so it is
    exact rather than a bound. Classifying rays as on/off by their centre
    coordinate would overestimate what is kept: a voxel landing at ``u = nu -
    0.4`` is "on detector" but still loses 40% of its weight over the edge.
    """
    px = pixel_coordinates(
        positions_lab, directions,
        normal=normal, distance_um=distance_um, pixel_um=pixel_um,
        detector_shape=detector_shape, center_px=center_px,
    )
    mag = values.detach().abs()
    total = float(mag.sum())
    if total == 0.0:
        return 0.0
    kept = float(splat_bilinear(px[:, 0], px[:, 1], mag, detector_shape).sum())
    return max(0.0, 1.0 - kept / total)
