"""Forward model: assemble finite-size diffraction intensity.

    I(q) = |F(q)|^2 . |S(q)|^2 . Lp(2theta)        (summed over the spectrum)

* ``F(q)``  -- continuous-q structure factor (this module; reuses the
  ``midas_hkls`` form-factor / lattice / DWF primitives but does NOT round the
  Miller indices, which is required to sample along a rod / RSM / fringe).
* ``S(q)``  -- finite-size interference factor (:mod:`midas_2d.shape_factor`).
* ``Lp``    -- Lorentz-polarization (:mod:`midas_hkls.intensity`).

Everything is torch-differentiable and device-portable.

Why a continuous-q structure factor?
-------------------------------------
``midas_hkls.structure_factor.structure_factors`` casts ``hkl`` to integers --
correct for Bragg-peak lists, but a thickness fringe lives *between* integer
``l``.  We therefore evaluate ``F`` at real-valued Miller indices here, reusing
the exact same physics (Cromer-Mann ``form_factor_batch``, the reciprocal
metric, and the isotropic Debye-Waller convention) so the two agree at integer
``hkl``.  Anisotropic / time-dependent disorder is Phase 2 (``disorder.py``).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from midas_hkls.form_factors import form_factor_batch
from midas_hkls.lattice_torch import reciprocal_metric_tensor, s_squared, two_theta

from .shape_factor import interference_factor

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from midas_hkls.crystal_torch import CrystalTensor

__all__ = ["structure_factor_q", "rod_intensity"]

_TWOPI = 2.0 * math.pi


def structure_factor_q(crystal_t, hkl_cont, *, b_iso_override=None):
    """Continuous-q structure factor F(q), complex tensor shape (..., ) matching
    the leading shape of ``hkl_cont`` (which is (..., 3)).

    Reuses ``midas_hkls`` primitives; isotropic Debye-Waller only (Phase 1).
    Differentiable through lattice parameters, fractional coordinates,
    occupancies, and B-factors.
    """
    import torch

    hkl_cont = torch.as_tensor(hkl_cont)
    lattice = crystal_t.lattice_params
    dtype = lattice.dtype
    hkl_cont = hkl_cont.to(dtype=dtype, device=lattice.device)

    fract, occ, B_iso, _U = crystal_t.unit_cell_view()   # (N,3),(N,),(N,),_

    # s^2 = 1/(4 d^2) is purely geometric (lambda-independent) and works for
    # real-valued hkl through the reciprocal metric tensor.
    s2 = s_squared(hkl_cont, lattice)                    # (...)

    f0 = form_factor_batch(s2, crystal_t.elements)        # (..., N)

    # Phase exp(2 pi i  h . r) for continuous h.
    phase = _TWOPI * (hkl_cont @ fract.T)                 # (..., N)
    geom = torch.complex(torch.cos(phase), torch.sin(phase))

    # Isotropic Debye-Waller exp(-B s^2).
    B = (torch.as_tensor(b_iso_override, dtype=dtype, device=lattice.device)
         if b_iso_override is not None else B_iso.to(dtype))
    DWF = torch.exp(-B.reshape((1,) * (s2.dim()) + (-1,)) * s2.unsqueeze(-1))  # (...,N)

    f_complex = torch.complex(f0, torch.zeros_like(f0))
    occ_c = torch.complex(occ.to(dtype), torch.zeros_like(occ.to(dtype)))
    F = (f_complex * DWF.to(geom.dtype) * geom * occ_c).sum(dim=-1)            # (...)
    return F


def rod_intensity(crystal_t, hkl_cont, N, *, wavelength_A, polarization=0.5,
                  apply_lp=True):
    """Finite-size intensity I(q) = |F|^2 . |S|^2 . Lp along an arbitrary set of
    continuous Miller indices.

    Parameters
    ----------
    crystal_t : CrystalTensor
        From ``midas_hkls.crystal_torch.crystal_to_tensor``.
    hkl_cont : tensor, shape (..., 3)
        Continuous Miller indices to evaluate (a rod scan, an RSM grid, ...).
    N : sequence/tensor length 3
        Unit-cell counts (N1, N2, N3); the finite dimension drives the fringes.
    wavelength_A : float or tensor
        For the Lorentz-polarization 2-theta only (geometry is in reciprocal
        units; the shape factor is lambda-independent).
    polarization : float
        Polarization fraction K (0.5 = unpolarized).
    apply_lp : bool
        Multiply by the Lorentz-polarization factor.

    Returns
    -------
    tensor, shape (...,)
        Real, non-negative intensity (arbitrary scale).
    """
    import torch
    from midas_hkls.intensity import lorentz_polarization

    F = structure_factor_q(crystal_t, hkl_cont)           # (...) complex
    F2 = (F.real * F.real + F.imag * F.imag)              # |F|^2
    S2 = interference_factor(hkl_cont, N)                 # (...)
    I = F2 * S2
    if apply_lp:
        tt = two_theta(hkl_cont, crystal_t.lattice_params, wavelength_A)  # rad
        I = I * lorentz_polarization(tt, polarization=polarization)
    return I
