"""Intensity helpers for X-ray diffraction.

The pf-HEDM peak-fitting workflow needs:
  - Multiplicity m (already on Reflection from generate_hkls).
  - Lorentz-polarization factor Lp(2θ).
  - Optional polarization fraction (synchrotron beamlines).
  - Combined I = m · |F|² · Lp.

All helpers are pure-tensor functions; gradient flow through F, lattice
parameters, wavelength, and the polarization fraction is preserved.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from .crystal_torch import CrystalTensor
from .hkl_gen import Reflection
from .lattice_torch import two_theta as _two_theta_torch
from .structure_factor import structure_factor_intensity, structure_factors

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "lorentz_polarization",
    "powder_intensity",
    "intensity_from_crystal",
    "attach_intensities",
]


# ----------------------------------------------------------------- LP factor

def lorentz_polarization(two_theta_rad, *, polarization: float = 0.5):
    """Lorentz-polarization factor for a *full Debye ring* (azimuthally averaged).

    Lp(2θ) = ½(1 + cos²2θ) / (sin²θ cosθ)

    The polarization term is the average over the whole ring, and the ring
    average does NOT depend on how polarized the beam is: for any linear
    polarization fraction f, ⟨½(1 + cos²2θ − f cos 2η sin²2θ)⟩_η = ½(1 + cos²2θ).
    So ``polarization`` only has one meaningful value here, 0.5, and anything
    else warns. (Before 2026-10-15 the term was ``(1-K) + K cos²2θ``; for
    K != 0.5 that is the value in ONE direction, the scattering plane
    containing the polarization vector, not the ring average, and at K=1 it
    under-weighted 2θ=60° by 2.5x. No caller in the repo passed K != 0.5.)

    For a single reflection at a known azimuth (rotation-method, DFXM, any
    spot-wise use) use :func:`midas_hkls.corrections.polarization_factor`,
    which takes η. Inputs and outputs are torch tensors; radians for 2θ.
    """
    import torch
    tt = torch.as_tensor(two_theta_rad)
    if float(polarization) != 0.5:
        warnings.warn(
            "lorentz_polarization is the full-ring average, which does not depend on "
            f"the polarization fraction; polarization={polarization} is ignored. "
            "Use midas_hkls.corrections.polarization_factor(two_theta, eta, ...) "
            "for a per-azimuth factor.", UserWarning, stacklevel=2)
    cos_tt = torch.cos(tt)
    sin_th = torch.sin(tt / 2.0)
    cos_th = torch.cos(tt / 2.0)
    pol = 0.5 + 0.5 * cos_tt * cos_tt
    lor = 1.0 / (sin_th * sin_th * cos_th)
    return pol * lor


# --------------------------------------------------------------- powder I_hkl

def powder_intensity(F, multiplicity, two_theta_rad, *, polarization: float = 0.5):
    """I_hkl ∝ m · |F|² · Lp(2θ).

    All inputs may be tensors; the output preserves gradients.  No absolute
    scale (the proportionality constant absorbs the unit-cell, beam, and
    detector factors that are usually fit as a single scalar).
    """
    import torch
    m = torch.as_tensor(multiplicity, dtype=F.real.dtype if F.is_complex() else F.dtype,
                        device=F.real.device if F.is_complex() else F.device)
    F2 = structure_factor_intensity(F) if F.is_complex() else (F * F)
    Lp = lorentz_polarization(two_theta_rad, polarization=polarization)
    return m * F2 * Lp


# ------------------------------------------------------- end-to-end convenience

def intensity_from_crystal(
    crystal_t: CrystalTensor,
    refs: Sequence[Reflection],
    *,
    wavelength_A,
    polarization: float = 0.5,
    anomalous: bool = False,
):
    """One-shot: F_hkl + powder intensity for a list of reflections.

    Returns ``(F, I)``: complex F (M,) and real I (M,), both torch tensors.
    Differentiable through every parameter on ``crystal_t`` plus ``wavelength_A``.
    """
    import torch
    hkls = [(r.h, r.k, r.l) for r in refs]
    F = structure_factors(crystal_t, hkls,
                           wavelength_A=wavelength_A, anomalous=anomalous)
    hkl_t = torch.tensor(hkls, dtype=crystal_t.lattice_params.dtype,
                          device=crystal_t.lattice_params.device)
    tt = _two_theta_torch(hkl_t, crystal_t.lattice_params, wavelength_A)
    m = torch.tensor([r.multiplicity for r in refs],
                     dtype=crystal_t.lattice_params.dtype,
                     device=crystal_t.lattice_params.device)
    I = powder_intensity(F, m, tt, polarization=polarization)
    return F, I


def attach_intensities(
    refs: Iterable[Reflection],
    F,
    I=None,
) -> list[Reflection]:
    """Return a NEW list of Reflections with F_real, F_imag, intensity attached.

    ``F`` and ``I`` may be torch tensors or numpy arrays — they're detached and
    converted to floats on assignment.
    """
    F_arr = _to_numpy(F)
    I_arr = _to_numpy(I) if I is not None else None
    out: list[Reflection] = []
    for k, r in enumerate(refs):
        new = replace(r)
        new.F_real = float(F_arr.real[k])
        new.F_imag = float(F_arr.imag[k]) if np.iscomplexobj(F_arr) else 0.0
        new.intensity = float(I_arr[k]) if I_arr is not None else None
        out.append(new)
    return out


def _to_numpy(arr):
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(arr)
