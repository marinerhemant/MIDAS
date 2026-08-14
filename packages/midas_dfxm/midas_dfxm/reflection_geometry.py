"""Reflection-geometry DFXM: absorption depth-weighting and flank-difference imaging.

Adopted from Yildirim et al., "Spatially resolved elastic strain and lattice rotation at
threading dislocations in HgCdTe/CdZnTe epilayers by dark-field X-ray microscopy"
(arXiv:2608.09841, 2026). Two effects a transmission-geometry forward omits but a
*reflection* (Bragg) DFXM forward needs:

1. **Absorption depth-weighting.** In reflection the beam that diffracts at depth ``z``
   below the entry surface travels an extra ``2 z / sin(theta_B)`` through the crystal
   (in + out), so its contribution to the image is attenuated by
   ``exp(-2 z / (Lambda sin theta_B))`` with ``Lambda`` the linear attenuation length. The
   signal is therefore exponentially surface-weighted (1/e depth ``= Lambda sin(theta_B)/2``),
   NOT a symmetric beam profile. For the paper's MCT(111) at 17 keV, 2theta = 11.11 deg,
   this is 1/e every 1.2 um. This module weights per-voxel intensities by that factor.

2. **Single-frame flank difference.** When two populations at different depths (e.g. an
   epilayer and its substrate) are separated in rocking angle by less than the objective's
   reciprocal-space acceptance, their rocking-curve peaks are *convolved, not resolved*.
   Two weak-beam images on opposite flanks, differenced, exploit the different depth-weight
   / peak-offset to isolate the two families in effectively one measurement.

``Lambda`` comes from :func:`midas_dfxm.polymer.attenuation_length_um` (which uses
midas-hkls absorption) -- never an external x-ray table. Everything torch-differentiable.
"""
from __future__ import annotations

import math

import torch

from .conventions import GoniometerSetting
from .forward import voxel_intensity


def attenuation_1e_depth(two_theta_deg: float, attenuation_length_um: float) -> float:
    """1/e depth of the diffracted intensity in symmetric reflection (Bragg) DFXM.

    ``z_1e = Lambda sin(theta_B) / 2`` (round-trip path ``2 z / sin theta_B``), with
    ``theta_B = two_theta / 2`` and ``Lambda`` the linear attenuation length [um]. Yildirim
    et al. 2026 report 1.2 um for MCT(111), 17 keV, 2theta = 11.11 deg.
    """
    theta_b = math.radians(0.5 * float(two_theta_deg))
    return 0.5 * float(attenuation_length_um) * math.sin(theta_b)


def reflection_depth_weight(depth_um, *, two_theta_deg: float,
                            attenuation_length_um: float) -> torch.Tensor:
    """Diffracted-intensity depth weight ``exp(-2 z / (Lambda sin theta_B)) = exp(-z / z_1e)``.

    ``depth_um`` is the depth of each voxel below the entry surface (>= 0, into the sample);
    ``z_1e`` from :func:`attenuation_1e_depth`. Differentiable in ``depth_um``.
    """
    z = torch.as_tensor(depth_um)
    z1e = attenuation_1e_depth(two_theta_deg, attenuation_length_um)
    return torch.exp(-z / z1e)


def surface_depth(positions: torch.Tensor, surface_normal) -> torch.Tensor:
    """Per-voxel depth below the entry surface: ``(pos . n_hat)`` referenced so the
    shallowest voxel is 0 and depth increases into the sample.

    ``surface_normal`` points from the sample INTO the beam (outward); depth grows opposite
    to it. Returns ``(N,)`` >= 0.
    """
    n = torch.as_tensor(surface_normal, device=positions.device, dtype=positions.dtype)
    n = n / torch.linalg.vector_norm(n)
    proj = positions @ n                       # larger = closer to the surface (outward)
    return proj.max() - proj                   # shallowest -> 0, deeper -> positive


def depth_weighted_intensity(
    field,
    hkl,
    goniometer: GoniometerSetting,
    resolution,
    *,
    surface_normal,
    two_theta_deg: float,
    attenuation_length_um: float,
    **voxel_kwargs,
) -> torch.Tensor:
    """Per-voxel intensity ``(N,)`` weighted by the reflection-geometry absorption profile.

    ``voxel_intensity`` gives the kinematic per-voxel response; each voxel is then scaled by
    :func:`reflection_depth_weight` at its :func:`surface_depth`. Splat through
    ``optics.render`` for a detector image. Differentiable end-to-end.
    """
    inten = voxel_intensity(field, hkl, goniometer, resolution, **voxel_kwargs)
    depth = surface_depth(field.positions, surface_normal)
    w = reflection_depth_weight(depth, two_theta_deg=two_theta_deg,
                                attenuation_length_um=attenuation_length_um)
    return inten * w


def flank_difference_intensity(
    field,
    hkl,
    resolution,
    *,
    center: GoniometerSetting | None = None,
    flank_deg: float = 0.05,
    axis: str = "chi",
    depth_weight: bool = False,
    surface_normal=None,
    two_theta_deg: float | None = None,
    attenuation_length_um: float | None = None,
    **voxel_kwargs,
) -> torch.Tensor:
    """Per-voxel weak-beam flank difference ``I(+flank) - I(-flank)`` (Yildirim et al. 2026).

    Rocks by ``+/- flank_deg`` about ``axis`` (``'chi'`` or ``'phi'``) from ``center`` and
    differences the two weak-beam images. Two depth-separated, rocking-convolved populations
    (layer vs substrate) come out with opposite sign / different weight, isolating them in one
    difference. With ``depth_weight=True`` each image is absorption-weighted first (needs
    ``surface_normal``, ``two_theta_deg``, ``attenuation_length_um``). Differentiable.
    """
    c = center or GoniometerSetting()
    if axis not in ("chi", "phi"):
        raise ValueError("axis must be 'chi' or 'phi'")

    def _img(offset):
        kw = dict(mu=c.mu, omega=c.omega, chi=c.chi, phi=c.phi)
        kw[axis] = kw[axis] + offset
        setting = GoniometerSetting(**kw)
        if depth_weight:
            if surface_normal is None or two_theta_deg is None or attenuation_length_um is None:
                raise ValueError("depth_weight=True needs surface_normal, two_theta_deg, "
                                 "attenuation_length_um")
            return depth_weighted_intensity(
                field, hkl, setting, resolution, surface_normal=surface_normal,
                two_theta_deg=two_theta_deg, attenuation_length_um=attenuation_length_um,
                **voxel_kwargs)
        return voxel_intensity(field, hkl, setting, resolution, **voxel_kwargs)

    return _img(+flank_deg) - _img(-flank_deg)
