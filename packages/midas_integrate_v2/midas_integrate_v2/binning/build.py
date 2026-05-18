"""Build the per-pixel → bin mapping from an :class:`IntegrationSpec`.

For Phase 1 we delegate to v1's ``build_map`` (numba-backed, well-tested,
fast). The v2 contribution at this layer is:

- An :class:`IntegrationSpec`-aware front door.
- A :class:`MapCache` that hashes the geometry-relevant subset of the
  spec and only rebuilds when something actually changed — same hash as
  v1's :func:`midas_integrate.bin_io.compute_param_hash` so the two
  packages can share a ``Map.bin`` cache.

The map's pxList/counts/offsets are returned both as numpy (for I/O)
and as torch tensors of the spec's dtype/device (for downstream
integration).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from midas_integrate.bin_io import compute_param_hash
from midas_integrate.detector_mapper import build_map as _v1_build_map, BuildMapResult

from ..compat.to_v1 import v1_params_from_spec
from ..spec import IntegrationSpec


@dataclass
class BuiltMap:
    """Output of :func:`build_map` — the v1 numpy result plus its hash."""
    result: BuildMapResult            # holds pxList, counts, offsets (numpy)
    param_hash: bytes                 # 32-byte SHA-256 of geometry inputs


def _spec_param_hash(spec: IntegrationSpec) -> bytes:
    """SHA-256 over the same fields v1 hashes — keeps caches interchangeable."""
    p = v1_params_from_spec(spec)
    return compute_param_hash(
        Lsd=p.Lsd, Ycen=p.BC_y, Zcen=p.BC_z,
        pxY=p.pxY, pxZ=p.pxZ,
        tx=p.tx, ty=p.ty, tz=p.tz,
        p0=p.p0, p1=p.p1, p2=p.p2, p3=p.p3, p4=p.p4,
        p5=p.p5, p6=p.p6, p7=p.p7, p8=p.p8, p9=p.p9,
        p10=p.p10, p11=p.p11, p12=p.p12, p13=p.p13, p14=p.p14,
        Parallax=p.Parallax,
        RhoD=p.RhoD,
        RBinSize=p.RBinSize, EtaBinSize=p.EtaBinSize,
        RMin=p.RMin, RMax=p.RMax,
        EtaMin=p.EtaMin, EtaMax=p.EtaMax,
        NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
        TransOpt=tuple(p.TransOpt),
        qMode=int(p.q_mode_active),
        Wavelength=p.Wavelength,
    )


def build_map(
    spec: IntegrationSpec,
    *,
    mask: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    panels: Optional[Sequence] = None,
    distortion_y: Optional[np.ndarray] = None,
    distortion_z: Optional[np.ndarray] = None,
    residual_corr=None,
    auto_load: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
    use_numba: Optional[bool] = None,
    per_row_max_entries: Optional[int] = None,
) -> BuiltMap:
    """Build the pixel → bin mapping from a v2 :class:`IntegrationSpec`.

    Parameters mirror :func:`midas_integrate.detector_mapper.build_map`;
    geometry is read from the spec via the v1 adapter.
    """
    p = v1_params_from_spec(spec)
    res = _v1_build_map(
        p, mask=mask, flat=flat, panels=panels,
        distortion_y=distortion_y, distortion_z=distortion_z,
        residual_corr=residual_corr,
        auto_load=auto_load, n_jobs=n_jobs, verbose=verbose,
        use_numba=use_numba, per_row_max_entries=per_row_max_entries,
    )
    return BuiltMap(result=res, param_hash=_spec_param_hash(spec))


class MapCache:
    """Cache of (param_hash → BuiltMap) so repeated calls with the same
    geometry don't rebuild. Single-entry by default — most refinement
    loops only ever need the latest geometry.

    Use ``cache.get(spec)`` everywhere; it rebuilds only when the
    geometry-relevant inputs to v1's ``compute_param_hash`` change.
    """
    def __init__(self, max_entries: int = 1):
        self.max_entries = max_entries
        self._store: dict[bytes, BuiltMap] = {}
        self._order: list[bytes] = []

    def get(self, spec: IntegrationSpec, **build_kwargs) -> BuiltMap:
        h = _spec_param_hash(spec)
        if h in self._store:
            return self._store[h]
        bm = build_map(spec, **build_kwargs)
        self._store[h] = bm
        self._order.append(h)
        while len(self._order) > self.max_entries:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        return bm

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


__all__ = ["build_map", "BuiltMap", "MapCache", "_spec_param_hash"]
