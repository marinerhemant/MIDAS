"""DFXM forward model: deformation field -> detector image(s).

Phase 1 of ``implementation_plan.md`` — orchestrates field + resolution + optics.

For a goniometer setting ``s`` and imaged reflection ``hkl``, each voxel's
intensity is

    I(r; s) = |F_hkl|^2 * Lp(2*theta) * Res( G(s) @ Q(r) )

where ``Q(r) = F(r)^{-T} G0`` is the local scattering vector (``field.local_Q``),
``G(s)`` is the sample->lab goniometer rotation, ``Res`` the reciprocal-space
resolution function (:mod:`midas_dfxm.resolution`), ``|F_hkl|^2`` the kinematic
structure-factor intensity (``midas_2d`` / ``midas_hkls``), and ``Lp`` the
Lorentz-polarization factor. Voxel intensities are then splatted onto the detector
by the magnifying objective (:mod:`midas_dfxm.optics`).

The structure factor is treated as constant across voxels (kinematic; the local
lattice distortion's effect on ``|F|`` is negligible next to its effect on the
diffraction *geometry*). Everything is torch-differentiable and device-portable.

Units: wavelength Angstrom; angles degrees; intensities arbitrary (scale ``C=1``).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from midas_2d.forward import structure_factor_q
from midas_hkls.crystal_torch import crystal_to_tensor
from midas_hkls.intensity import lorentz_polarization

from .conventions import GoniometerSetting
from .field import DeformationField
from .optics import ObjectiveOptics
from .resolution import ResolutionFunction
from .scan import bragg_two_theta_deg


def structure_factor_intensity(
    crystal,
    hkl,
    *,
    wavelength_A: float,
    polarization: float = 0.5,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Kinematic ``|F_hkl|^2 * Lp(2*theta)`` for one reflection (scalar tensor).

    Reuses ``midas_2d.forward.structure_factor_q`` (continuous-q F, itself built on
    ``midas_hkls`` form factors + DWF) and ``midas_hkls.intensity.lorentz_polarization``.
    """
    crystal_t = crystal_to_tensor(crystal, device=device, dtype=dtype)
    hkl_t = torch.as_tensor(hkl, device=device, dtype=dtype)
    F = structure_factor_q(crystal_t, hkl_t)
    f2 = (F.real ** 2 + F.imag ** 2)
    # 2-theta from |Q| = 2*pi/d; d from the reciprocal metric via midas_hkls.
    from midas_hkls.lattice_torch import d_spacing

    latc = crystal_t.lattice_params
    d = d_spacing(hkl_t, latc)
    q_mag = 2.0 * math.pi / float(d)
    tt = math.radians(bragg_two_theta_deg(q_mag, wavelength_A))
    lp = lorentz_polarization(torch.tensor(tt, device=device, dtype=dtype), polarization=polarization)
    return f2 * lp


def voxel_intensity(
    field: DeformationField,
    hkl,
    goniometer: GoniometerSetting,
    resolution: ResolutionFunction,
    *,
    sf2: Optional[torch.Tensor] = None,
    Q_sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-voxel intensity ``(N,)`` at one goniometer setting.

    ``Q_sample`` (per-voxel sample-frame scattering vector) and ``sf2`` (scalar
    ``|F|^2 Lp``) may be precomputed and passed in to avoid recomputation across a
    scan. Differentiable in the field, resolution, and (if tensors) motor angles.
    """
    if Q_sample is None:
        Q_sample = field.local_Q(hkl)
    G = goniometer.sample_rotation(device=Q_sample.device, dtype=Q_sample.dtype)
    Q_lab = Q_sample @ G.transpose(-1, -2)  # (N,3) rows rotated: (G @ q)^T
    w = resolution.weight(Q_lab)
    if sf2 is None:
        sf2 = torch.tensor(1.0, device=Q_sample.device, dtype=Q_sample.dtype)
    return sf2 * w


def dfxm_image(
    field: DeformationField,
    hkl,
    goniometer: GoniometerSetting,
    resolution: ResolutionFunction,
    optics: ObjectiveOptics,
    *,
    sf2: Optional[torch.Tensor] = None,
    Q_sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render one DFXM detector image ``(n_u, n_v)`` for a goniometer setting.

    Voxel positions are rotated into the lab frame and splatted through the
    objective. Differentiable end-to-end.
    """
    if Q_sample is None:
        Q_sample = field.local_Q(hkl)
    inten = voxel_intensity(field, hkl, goniometer, resolution, sf2=sf2, Q_sample=Q_sample)
    G = goniometer.sample_rotation(device=Q_sample.device, dtype=Q_sample.dtype)
    pos_lab = field.positions @ G.transpose(-1, -2)
    return optics.render(pos_lab, inten)


def dfxm_stack(
    field: DeformationField,
    hkl,
    settings: list[GoniometerSetting],
    resolution: ResolutionFunction,
    optics: ObjectiveOptics,
    *,
    sf2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stack of DFXM images over a scan: ``(S, n_u, n_v)``.

    ``Q_sample`` and ``sf2`` are computed once and reused across settings.
    """
    Q_sample = field.local_Q(hkl)
    imgs = [
        dfxm_image(field, hkl, s, resolution, optics, sf2=sf2, Q_sample=Q_sample)
        for s in settings
    ]
    return torch.stack(imgs, dim=0)


def mosaicity_curve(
    field: DeformationField,
    hkl,
    settings: list[GoniometerSetting],
    resolution: ResolutionFunction,
    *,
    sf2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrated intensity vs setting ``(S,)`` — the rocking / mosaicity curve.

    Sums per-voxel intensity over the field at each setting (detector-integrated).
    Differentiable; the cheap observable for identifiability / FWHM checks.
    """
    Q_sample = field.local_Q(hkl)
    vals = [
        voxel_intensity(field, hkl, s, resolution, sf2=sf2, Q_sample=Q_sample).sum()
        for s in settings
    ]
    return torch.stack(vals, dim=0)
