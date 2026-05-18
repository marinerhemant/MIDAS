"""Differentiable-by-default radial integration for v2 specs.

This module sits on top of v1's CSR kernel
(:func:`midas_integrate.kernels.integrate`) which is already torch-based
and runs on CPU/CUDA/MPS. We add:

- An :class:`IntegrationGeometry` that wraps v1's ``CSRGeometry`` plus
  the per-pixel R / η values reported by ``build_map``. The R / η values
  are kept around because Phase 2's autograd path needs them.

- :func:`build_geometry` — single call from a v2 :class:`IntegrationSpec`
  to a ready-to-integrate geometry, taking care of map-build → CSR
  construction.

For Phase 1 the forward pass is bit-identical to v1; Phase 2 layers an
:class:`autograd.Function` on top so a loss on the integrated profile
flows gradient back to refinable spec tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from midas_integrate.bin_io import PixelMap
from midas_integrate.kernels import (
    CSRGeometry,
    build_csr as _v1_build_csr,
    integrate as _v1_integrate,
    profile_1d as _v1_profile_1d,
)

from ..binning import MapCache, build_map
from ..binning.build import BuiltMap
from ..spec import IntegrationSpec


@dataclass
class IntegrationGeometry:
    """Everything the integration kernel needs at call time.

    Carries the v1 :class:`CSRGeometry` for the actual integration plus
    the geometry inputs that built it, so Phase 2 autograd can compute
    ``dProfile/dParams`` analytically without re-traversing the map.
    """
    csr: CSRGeometry
    built_map: BuiltMap
    spec: IntegrationSpec

    @property
    def n_r_bins(self) -> int:
        return self.csr.n_r

    @property
    def n_eta_bins(self) -> int:
        return self.csr.n_eta

    @property
    def device(self) -> torch.device:
        return self.csr.bin_indices.device

    @property
    def dtype(self) -> torch.dtype:
        return self.csr.values.dtype


def build_geometry(
    spec: IntegrationSpec,
    *,
    cache: Optional[MapCache] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    mask: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    **build_kwargs,
) -> IntegrationGeometry:
    """Build a ready-to-integrate :class:`IntegrationGeometry` from a spec.

    A ``cache`` is recommended in refinement loops — when you only change
    the parameters that don't affect bin assignments (e.g. tightening a
    correction), the cache returns the existing map.
    """
    spec.validate()
    if cache is None:
        cache = MapCache(max_entries=1)
    bm = cache.get(spec, mask=mask, flat=flat, **build_kwargs)

    pm = PixelMap(
        pxList=bm.result.pxList,
        counts=bm.result.counts,
        offsets=bm.result.offsets,
        map_header=None,
        nmap_header=None,
    )
    csr = _v1_build_csr(
        pm,
        n_r=spec.n_r_bins, n_eta=spec.n_eta_bins,
        n_pixels_y=spec.NrPixelsY, n_pixels_z=spec.NrPixelsZ,
        device=str(device) if not isinstance(device, torch.device) else device,
        dtype=dtype,
        bc_y=float(spec.BC_y), bc_z=float(spec.BC_z),
    )
    return IntegrationGeometry(csr=csr, built_map=bm, spec=spec)


def integrate(
    image: torch.Tensor,
    geom: IntegrationGeometry,
    *,
    mode: str = "floor",
    normalize: Optional[bool] = None,
) -> torch.Tensor:
    """Integrate ``image`` against ``geom``.

    Returns the 2D ``(n_eta, n_r)`` integrated array, same as v1.
    Forward is bit-identical to :func:`midas_integrate.kernels.integrate`.
    """
    if normalize is None:
        normalize = bool(geom.spec.Normalize)
    return _v1_integrate(image, geom.csr, mode=mode, normalize=normalize)


def profile_1d(
    int2d: torch.Tensor,
    geom: IntegrationGeometry,
    *,
    mode: str = "area_weighted",
) -> torch.Tensor:
    """Reduce a 2D integrated array to a 1D profile over R."""
    return _v1_profile_1d(int2d, geom.csr, mode=mode)


__all__ = [
    "IntegrationGeometry",
    "build_geometry",
    "integrate",
    "profile_1d",
]
