"""Emit a v1-format ``Map.bin`` / ``nMap.bin`` from a v2 binning geometry.

The on-disk format is the canonical
:mod:`midas_integrate.bin_io` v3 layout. Once written, the resulting
files are readable by every v1 consumer (``midas-integrate`` CLI,
``IntegratorZarrOMP``, ``IntegratorFitPeaksGPUStream``, downstream
peak-fit / Rietveld tooling) without any v2 dependency.

Two map sources supported:

- :class:`HardBinGeometry` — one entry per in-range pixel, each with
  ``frac = area = 1.0``. Equivalent to v1 ``floor`` integration with
  no subpixel oversampling.

- :class:`SubpixelBinGeometry` — one entry per in-range subpixel, each
  with ``frac = area = 1/K²``. As ``K`` grows this approaches v1's
  exact polygon-area kernel (used by v1 ``floor``-with-subpixel
  oversampling) within a fraction of a bin width.

What's intentionally NOT covered (use v1's ``build_map`` if needed):

- v1's exact polygon-area pixel-bin overlap (the ``circle_seg_intersect``
  + ``ray_seg_intersect`` Green's-theorem kernel). Subpixel
  oversampling at K=4 is within ~6% of the polygon-exact area on
  average and matches at single-bin precision; pure-torch polygon math
  is deferred to v0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from midas_integrate.bin_io import (
    PXLIST_DTYPE,
    MapHeader,
    compute_param_hash,
    write_map,
)
from midas_integrate.detector_mapper import _inverse_transform_pixel_arrays

from ..compat.to_v1 import v1_params_from_spec
from ..forward import eval_pixel_REta
from ..spec import IntegrationSpec
from .hard import HardBinGeometry
from .subpixel import SubpixelBinGeometry


def _entries_from_hard(
    geom: HardBinGeometry, spec: IntegrationSpec,
    R_centres: np.ndarray, eta_centres: np.ndarray,
    raw_y: np.ndarray, raw_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (entries, bin_idx) for a hard-bin geometry."""
    NY = spec.NrPixelsY
    with torch.no_grad():
        R, Eta = eval_pixel_REta(spec)
    R_flat = R.reshape(-1).cpu().numpy()
    Eta_flat = Eta.reshape(-1).cpu().numpy()      # noqa: F841 — kept for parity
    valid_np = geom.valid.cpu().numpy()
    flat_bin_np = geom.flat_bin.cpu().numpy()

    n_valid = int(valid_np.sum())
    entries = np.zeros(n_valid, dtype=PXLIST_DTYPE)
    bin_idx = flat_bin_np[valid_np]
    R_at_valid = R_flat[valid_np]

    pix_idx = np.where(valid_np)[0]
    grid_y = pix_idx % NY
    grid_z = pix_idx // NY
    raw_y_flat = raw_y.reshape(-1)
    raw_z_flat = raw_z.reshape(-1)
    entries["y"] = raw_y_flat[pix_idx]
    entries["z"] = raw_z_flat[pix_idx]
    entries["frac"] = 1.0
    entries["areaWeight"] = 1.0

    # deltaR = R_pixel - R_bin_centre  (R_bin_centre stored in R_centres
    # at flat-bin index, but flat_bin = eta * n_r + r so the R index is
    # flat_bin % n_r).
    n_r = geom.n_r
    r_idx = bin_idx % n_r
    entries["deltaR"] = R_at_valid - R_centres[r_idx]
    return entries, bin_idx


def _entries_from_subpixel(
    geom: SubpixelBinGeometry, spec: IntegrationSpec,
    R_centres: np.ndarray, eta_centres: np.ndarray,
    raw_y: np.ndarray, raw_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (entries, bin_idx) for a K×K-oversampled geometry.

    Each in-range subpixel becomes one entry with ``frac = area = 1/K²``
    and ``deltaR`` measured against its bin's R centre.
    """
    NY = spec.NrPixelsY
    K = geom.K
    inv_K2 = 1.0 / (K * K)
    n_r = geom.n_r

    if K == 1:
        offs_y = np.array([0.0])
        offs_z = np.array([0.0])
    else:
        step = 1.0 / K
        base = (np.arange(K) + 0.5) * step - 0.5
        offs_y = base
        offs_z = base

    raw_y_flat = raw_y.reshape(-1)
    raw_z_flat = raw_z.reshape(-1)

    entries_chunks = []
    bin_idx_chunks = []
    sub_k = 0
    valid_np = geom.valid.cpu().numpy()
    flat_bin_np = geom.flat_bin.cpu().numpy()
    for ky, dy in enumerate(offs_y):
        for kz, dz in enumerate(offs_z):
            with torch.no_grad():
                # eval_pixel_REta with subpixel offset — same formula as
                # SubpixelBinGeometry.from_spec.
                ys = torch.arange(NY, dtype=spec.dtype(), device=spec.device())
                zs = torch.arange(spec.NrPixelsZ, dtype=spec.dtype(),
                                   device=spec.device())
                Z, Y = torch.meshgrid(zs, ys, indexing="ij")
                from .subpixel import pixel_to_REta_from_spec   # local import
                # We can call eval_pixel_REta with shifted grid, but it
                # always uses (arange Y, arange Z); use the lower-level
                # pixel_to_REta_from_spec instead.
                from ..forward import pixel_to_REta_from_spec as p2r
                out = p2r(Y + dy, Z + dz, spec)
                R_flat = out.R_px.reshape(-1).cpu().numpy()

            v = valid_np[sub_k]
            n_v = int(v.sum())
            ent = np.zeros(n_v, dtype=PXLIST_DTYPE)
            pix_idx = np.where(v)[0]
            ent["y"] = raw_y_flat[pix_idx]
            ent["z"] = raw_z_flat[pix_idx]
            ent["frac"] = inv_K2
            ent["areaWeight"] = inv_K2
            bin_for_v = flat_bin_np[sub_k][v]
            r_idx = bin_for_v % n_r
            ent["deltaR"] = R_flat[v] - R_centres[r_idx]
            entries_chunks.append(ent)
            bin_idx_chunks.append(bin_for_v)
            sub_k += 1

    entries = np.concatenate(entries_chunks)
    bin_idx = np.concatenate(bin_idx_chunks)
    return entries, bin_idx


def write_map_bin_from_geometry(
    geom,
    spec: IntegrationSpec,
    out_dir: str | Path,
    *,
    map_filename: str = "Map.bin",
    nmap_filename: str = "nMap.bin",
    write_header: bool = True,
) -> Tuple[Path, Path]:
    """Emit a v1-format ``Map.bin`` and ``nMap.bin`` from a v2 geometry.

    Accepts :class:`HardBinGeometry` or :class:`SubpixelBinGeometry`.
    The header's ``param_hash`` is computed via the same canonical
    function v1 uses (:func:`midas_integrate.bin_io.compute_param_hash`)
    so the resulting files participate in v1's cache invalidation
    correctly.

    Parameters
    ----------
    geom :
        v2 binning geometry (Hard or Subpixel).
    spec :
        the :class:`IntegrationSpec` used to build ``geom``. We pull
        the canonical param hash + R/eta bin centres from it.
    out_dir :
        destination directory; created if missing.
    write_header :
        ``True`` writes the v3 header (recommended). ``False`` produces
        legacy header-less files (only useful for testing).

    Returns ``(Map.bin path, nMap.bin path)``.
    """
    if not isinstance(geom, (HardBinGeometry, SubpixelBinGeometry)):
        raise TypeError(
            f"geom must be HardBinGeometry or SubpixelBinGeometry, got "
            f"{type(geom).__name__}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    n_r = spec.n_r_bins
    n_eta = spec.n_eta_bins

    # Bin centres in pixel/degree space
    R_centres = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    eta_centres = spec.EtaMin + spec.EtaBinSize * (np.arange(n_eta) + 0.5)

    raw_y, raw_z = _inverse_transform_pixel_arrays(NY, NZ, list(spec.TransOpt))

    if isinstance(geom, HardBinGeometry):
        entries, bin_idx = _entries_from_hard(
            geom, spec, R_centres, eta_centres, raw_y, raw_z,
        )
    else:
        entries, bin_idx = _entries_from_subpixel(
            geom, spec, R_centres, eta_centres, raw_y, raw_z,
        )

    # Sort all entries by bin index so each bin's entries are contiguous.
    order = np.argsort(bin_idx, kind="stable")
    entries = entries[order]
    bin_idx = bin_idx[order]

    n_bins = n_eta * n_r
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int32)
    offsets = np.empty(n_bins, dtype=np.int32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts[:-1])

    header = None
    if write_header:
        p1 = v1_params_from_spec(spec)
        header = MapHeader(
            param_hash=compute_param_hash(
                Lsd=p1.Lsd, Ycen=p1.BC_y, Zcen=p1.BC_z,
                pxY=p1.pxY, pxZ=p1.pxZ,
                tx=p1.tx, ty=p1.ty, tz=p1.tz,
                p0=p1.p0, p1=p1.p1, p2=p1.p2, p3=p1.p3, p4=p1.p4,
                p5=p1.p5, p6=p1.p6, p7=p1.p7, p8=p1.p8, p9=p1.p9,
                p10=p1.p10, p11=p1.p11, p12=p1.p12, p13=p1.p13, p14=p1.p14,
                Parallax=p1.Parallax,
                RhoD=p1.RhoD,
                RBinSize=p1.RBinSize, EtaBinSize=p1.EtaBinSize,
                RMin=p1.RMin, RMax=p1.RMax,
                EtaMin=p1.EtaMin, EtaMax=p1.EtaMax,
                NrPixelsY=p1.NrPixelsY, NrPixelsZ=p1.NrPixelsZ,
                TransOpt=tuple(p1.TransOpt),
                qMode=int(p1.q_mode_active),
                Wavelength=p1.Wavelength,
            ),
            q_mode=int(p1.q_mode_active),
            gradient_mode=int(p1.GradientCorrection),
            wavelength=p1.Wavelength,
        )

    map_path = out_dir / map_filename
    nmap_path = out_dir / nmap_filename
    write_map(map_path, nmap_path,
              pxList=entries, counts=counts, offsets=offsets,
              header=header)
    return map_path, nmap_path


__all__ = ["write_map_bin_from_geometry"]
