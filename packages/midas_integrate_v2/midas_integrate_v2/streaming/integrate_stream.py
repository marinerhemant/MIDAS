"""Out-of-core streaming integration for sweep-mode workflows.

:func:`integrate_stream` builds the binning geometry once, then iterates
over a :class:`FrameSource` integrating each frame and writing its
profile via a user-supplied writer. Memory usage is constant in the
number of frames.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch

from ..binning import (
    HardBinGeometry, integrate_hard,
    PolygonBinGeometry, integrate_polygon,
    SubpixelBinGeometry, integrate_subpixel,
    SoftBinGeometry, integrate_soft,
)
from ..spec import IntegrationSpec
from .frame_source import FrameSource
from .normalize import FrameNormalizer


_INTEGRATE_FUNCS = {
    "hard":     integrate_hard,
    "subpixel": integrate_subpixel,
    "polygon":  integrate_polygon,
    "soft":     integrate_soft,
}


def _build_default_geometry(spec: IntegrationSpec, mode: str, K: int,
                              mask=None):
    if mode == "hard":
        return HardBinGeometry.from_spec(spec, mask=mask)
    if mode == "subpixel":
        return SubpixelBinGeometry.from_spec(spec, K=K, mask=mask)
    if mode == "polygon":
        return PolygonBinGeometry.from_spec(spec, mask=mask, n_jobs=-1)
    if mode == "soft":
        return SoftBinGeometry.from_spec(spec)
    raise ValueError(f"unknown mode {mode!r}")


def integrate_stream(
    spec: IntegrationSpec,
    source: FrameSource,
    *,
    mode: str = "polygon",
    K: int = 2,
    mask: Optional[np.ndarray] = None,
    normaliser: Optional[FrameNormalizer] = None,
    writer: Optional[Callable[[str, np.ndarray, np.ndarray], None]] = None,
    progress_every: int = 0,
) -> dict:
    """Build geometry once; integrate every frame in ``source`` against it.

    Parameters
    ----------
    spec :
        :class:`IntegrationSpec`. Geometry is built once from this; if
        you change the spec mid-stream the geometry won't update.
    source :
        Any :class:`FrameSource` (TIFF glob, HDF5, Zarr, in-memory).
    mode :
        Binning kernel: ``"hard"``, ``"subpixel"``, ``"polygon"``
        (default), or ``"soft"``.
    K :
        Subpixel oversampling K (only when mode='subpixel').
    mask :
        Optional bad-pixel mask; passed to the geometry build.
    normaliser :
        Optional :class:`FrameNormalizer`; applied per frame before
        integration.
    writer :
        Optional callable ``(frame_id, r_axis_px, profile_1d) -> None``
        called once per frame. Use this to emit CSV/XYE per frame
        without holding profiles in memory. If None, profiles are
        accumulated in the returned dict.
    progress_every :
        Print "processed N / total" every this many frames. 0 = silent.

    Returns
    -------
    dict with keys:
      - ``n_processed``: number of frames integrated
      - ``profiles``: ``(n_frames, n_r)`` ndarray (only if writer is None)
      - ``frame_ids``: list of N strings
      - ``r_axis_px``: ``(n_r,)`` ndarray
    """
    spec.validate()
    if source.frame_shape != (spec.NrPixelsZ, spec.NrPixelsY):
        raise ValueError(
            f"frame source shape {source.frame_shape} != spec detector "
            f"({spec.NrPixelsZ}, {spec.NrPixelsY})"
        )
    if mode not in _INTEGRATE_FUNCS:
        raise ValueError(
            f"unknown mode {mode!r}; valid: {list(_INTEGRATE_FUNCS)}"
        )
    geom = _build_default_geometry(spec, mode, K, mask=mask)
    integrate_fn = _INTEGRATE_FUNCS[mode]

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    profiles = []
    frame_ids = []
    n_total = source.n_frames
    for i, (fid, img) in enumerate(source):
        if normaliser is not None:
            img = normaliser(fid, img)
        img_t = torch.from_numpy(img.astype(np.float64))
        if mode == "soft":
            int2d = integrate_fn(img_t, geom)
        else:
            int2d = integrate_fn(img_t, geom, normalize=True)
        prof = int2d.mean(dim=0).detach().cpu().numpy()
        if writer is not None:
            writer(fid, r_axis, prof)
        else:
            profiles.append(prof)
        frame_ids.append(fid)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  integrated {i + 1}/{n_total} frames")

    out = {
        "n_processed": len(frame_ids),
        "frame_ids": frame_ids,
        "r_axis_px": r_axis,
    }
    if writer is None:
        out["profiles"] = np.asarray(profiles)
    return out


__all__ = ["integrate_stream"]
