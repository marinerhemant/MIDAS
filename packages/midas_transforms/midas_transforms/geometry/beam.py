"""Beam-profile abstraction for PF/FF V-map and soft attribution.

A :class:`BeamProfile` is a :class:`torch.nn.Module` so that refinable beam
parameters (FWHM, offset, ...) are :class:`torch.nn.Parameter` and integrate
cleanly with :func:`torch.optim.LBFGS` / :func:`torch.optim.Adam`.

The core API is::

    fraction = beam.fraction_over_voxel(scan_pos_um, voxel_center_um, voxel_size_um)

Inputs are broadcast against each other, so the same call works for the
forward-model contract (``(n_spots, n_voxels)`` matrix of weights) and for
unit tests on 0-d scalars.

All three profiles below are device-portable (CPU / CUDA / MPS) and
autograd-differentiable in every continuous argument.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = ["BeamProfile", "TopHat", "Gaussian"]


def _to_tensor(x, *, dtype: Optional["torch.dtype"] = None,
               device: Optional["torch.device"] = None) -> "torch.Tensor":
    import torch
    if torch.is_tensor(x):
        if dtype is not None or device is not None:
            return x.to(dtype=dtype or x.dtype, device=device or x.device)
        return x
    return torch.as_tensor(x, dtype=dtype or torch.float64, device=device)


class BeamProfile:
    """Abstract base class.

    Not a real :class:`torch.nn.Module` at the base level — concrete subclasses
    subclass :class:`torch.nn.Module` directly because importing ``torch`` at
    class-creation time would prevent base ``midas-transforms`` (which is
    torch-free) from importing this module.  We avoid that by deferring the
    ``nn.Module`` inheritance into each concrete profile.
    """

    def fraction_over_voxel(
        self,
        scan_pos_um: "torch.Tensor",
        voxel_center_um: "torch.Tensor",
        voxel_size_um: "torch.Tensor",
    ) -> "torch.Tensor":  # pragma: no cover - abstract
        raise NotImplementedError


def _make_module():
    """Return ``torch.nn.Module`` if torch is available, else a no-op stand-in
    so ``isinstance`` checks elsewhere still work without torch imported."""
    try:
        import torch
        return torch.nn.Module
    except ImportError:  # pragma: no cover
        class _Stub:
            def __init__(self, *a, **kw): pass
        return _Stub


_Module = _make_module()


class TopHat(_Module, BeamProfile):
    """Rectangular beam profile (PF default).

    The beam illuminates ``[scan_pos - width/2, scan_pos + width/2]`` uniformly.
    Fraction over a voxel of size ``voxel_size`` centered at ``voxel_center``
    is the overlap of the two intervals, normalized by the voxel size.

    Parameters
    ----------
    width_um : float | Tensor
        Beam width along the scan axis (µm).
    refine   : bool
        If ``True``, ``width_um`` becomes an :class:`torch.nn.Parameter` and
        is refined by upstream optimizers.  Default ``False`` (Wenxi PF).
    """

    def __init__(
        self,
        width_um,
        *,
        refine: bool = False,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ):
        super().__init__()
        import torch
        w = _to_tensor(width_um, dtype=dtype or torch.float64, device=device)
        if refine:
            self.width_um = torch.nn.Parameter(w.clone())
        else:
            self.register_buffer("width_um", w)

    def fraction_over_voxel(
        self,
        scan_pos_um: "torch.Tensor",
        voxel_center_um: "torch.Tensor",
        voxel_size_um: "torch.Tensor",
    ) -> "torch.Tensor":
        """Analytic overlap of the beam window with a voxel window."""
        import torch
        w = self.width_um
        beam_lo = scan_pos_um - w / 2.0
        beam_hi = scan_pos_um + w / 2.0
        vox_lo = voxel_center_um - voxel_size_um / 2.0
        vox_hi = voxel_center_um + voxel_size_um / 2.0
        overlap = torch.clamp(
            torch.minimum(beam_hi, vox_hi) - torch.maximum(beam_lo, vox_lo),
            min=0.0,
        )
        return overlap / voxel_size_um


class Gaussian(_Module, BeamProfile):
    """Gaussian beam profile (FF default).

    Beam intensity is modeled as ``I(x) = exp(-(x - center)^2 / (2 σ²))``
    (peak amplitude 1, NOT a unit-area PDF) so that the returned ``fraction``
    has the same physical interpretation as :class:`TopHat` — at the limit
    of a voxel small compared to FWHM, sitting under the beam peak,
    ``fraction → 1``.

    Concretely::

        fraction = (σ √(2π) / voxel_size) · 0.5 · (erf(b) - erf(a))

    with ``σ = fwhm / (2 √(2 ln 2))``, ``a = (voxel_lo - center) / (σ √2)``,
    ``b = (voxel_hi - center) / (σ √2)``.  The ``σ √(2π)`` prefactor converts
    the PDF integral back to the peak-1 amplitude integral.

    Parameters
    ----------
    fwhm_um         : full-width half-maximum along the scan axis (µm).
    center_offset_um: rigid offset between nominal and true beam position.
    refine_fwhm     : flag to make ``fwhm_um`` an ``nn.Parameter`` (FF refine).
    refine_offset   : same for ``center_offset_um``.
    """

    _LN2_2_SQRT2 = float(2.0 * math.sqrt(2.0 * math.log(2.0)))

    def __init__(
        self,
        fwhm_um,
        center_offset_um=0.0,
        *,
        refine_fwhm: bool = True,
        refine_offset: bool = False,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ):
        super().__init__()
        import torch
        f = _to_tensor(fwhm_um, dtype=dtype or torch.float64, device=device)
        o = _to_tensor(center_offset_um, dtype=dtype or torch.float64, device=device)
        if refine_fwhm:
            self.fwhm_um = torch.nn.Parameter(f.clone())
        else:
            self.register_buffer("fwhm_um", f)
        if refine_offset:
            self.center_offset_um = torch.nn.Parameter(o.clone())
        else:
            self.register_buffer("center_offset_um", o)

    def fraction_over_voxel(
        self,
        scan_pos_um: "torch.Tensor",
        voxel_center_um: "torch.Tensor",
        voxel_size_um: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch
        sigma = self.fwhm_um / Gaussian._LN2_2_SQRT2
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=sigma.dtype, device=sigma.device))
        sqrt_2pi = torch.sqrt(
            torch.tensor(2.0 * math.pi, dtype=sigma.dtype, device=sigma.device)
        )
        center = scan_pos_um + self.center_offset_um
        vox_lo = voxel_center_um - voxel_size_um / 2.0
        vox_hi = voxel_center_um + voxel_size_um / 2.0
        a = (vox_lo - center) / (sigma * sqrt2)
        b = (vox_hi - center) / (sigma * sqrt2)
        # ∫_voxel exp(-(x-c)²/(2σ²)) dx = σ √(2π) · 0.5 · (erf(b) - erf(a))
        peak_integral = sigma * sqrt_2pi * 0.5 * (torch.erf(b) - torch.erf(a))
        return peak_integral / voxel_size_um
