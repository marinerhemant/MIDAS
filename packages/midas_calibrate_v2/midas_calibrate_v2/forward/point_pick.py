"""One-shot, geometry-independent 2D local-maximum peak picking, directly on
the raw detector image.

Unlike the cake/azimuthal-bin extraction used by ``pipelines.single_pv``
(which re-samples intensity along a predicted arc at whatever geometry is
current), this module never bins, windows, or profiles intensity along a
predicted curve. It finds genuine local-intensity maxima anywhere inside a
coarse annular mask, then reads their (Y, Z) pixel coordinates back
directly. The seed geometry is used exactly once, only to decide which
pixels are eligible for which ring -- it never distorts what "the peak" at
an accepted pixel means, so the picked point cloud can be frozen and reused
across an entire fit with no re-extraction feedback loop.

Called once per calibration run by :mod:`pipelines.frozen_point`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import ndimage

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import RingTable

from ..compat.from_v1 import spec_from_v1_params
from ..parameters.pack import pack_spec, unpack_spec
from ..seed.mask import apply_mask_for_arcs
from .distortion import build_p_coeffs
from .geometry import pixel_to_REta

_RAD2DEG = 57.29577951308232


@dataclass
class PickedPoints:
    """A frozen point cloud: picked (Y, Z, ring_id) triples."""

    Y_pix: np.ndarray
    Z_pix: np.ndarray
    ring_idx: np.ndarray     # int, row index into the RingTable arrays
    snr: np.ndarray
    n_by_ring: dict          # ring_idx -> count, for reporting


def _seed_two_theta_map(
    v1_seed: V1Params,
    *,
    npy: int, npz: int,
    downsample: int,
    dtype=torch.float64, device="cpu",
) -> np.ndarray:
    """One-time 2θ map (deg) from the seed geometry, evaluated on a coarse
    grid and upsampled -- only used to place the coarse annular mask, so
    boundary precision at the ``downsample`` scale is immaterial (the
    accepted peak pixels are always read back from the full-resolution
    image, never from this map)."""
    spec_seed = spec_from_v1_params(v1_seed)
    x, info = pack_spec(spec_seed, dtype=dtype, device=device)
    unpacked = unpack_spec(x, info, spec_seed)
    p_coeffs = build_p_coeffs(unpacked, dtype=dtype, device=device)

    yc = np.arange(0, npy, downsample, dtype=np.float64)
    zc = np.arange(0, npz, downsample, dtype=np.float64)
    ZZ, YY = np.meshgrid(zc, yc, indexing="ij")     # [nz_d, ny_d]
    Y_t = torch.as_tensor(YY, dtype=dtype, device=device)
    Z_t = torch.as_tensor(ZZ, dtype=dtype, device=device)

    px = 0.5 * (v1_seed.pxY + v1_seed.pxZ) if v1_seed.pxZ > 0 else v1_seed.pxY
    rho_d_um = (v1_seed.RhoD if v1_seed.RhoD > 0
                else v1_seed.MaxRingRad * px)

    out = pixel_to_REta(
        Y_t, Z_t,
        Lsd=unpacked["Lsd"], BC_y=unpacked["BC_y"], BC_z=unpacked["BC_z"],
        tx=unpacked.get("tx", torch.zeros((), dtype=dtype, device=device)),
        ty=unpacked["ty"], tz=unpacked["tz"],
        p_coeffs=p_coeffs,
        parallax=unpacked.get("Parallax", torch.zeros((), dtype=dtype, device=device)),
        pxY=unpacked["pxY"], pxZ=unpacked.get("pxZ", unpacked["pxY"]),
        rho_d=torch.as_tensor(rho_d_um, dtype=dtype, device=device),
    )
    tt_deg_small = (out.two_theta_rad * _RAD2DEG).detach().cpu().numpy()

    zoom_z = npz / tt_deg_small.shape[0]
    zoom_y = npy / tt_deg_small.shape[1]
    tt_deg = ndimage.zoom(tt_deg_small, (zoom_z, zoom_y), order=1)
    if tt_deg.shape != (npz, npy):
        tt_full = np.empty((npz, npy), dtype=np.float64)
        h = min(npz, tt_deg.shape[0])
        w = min(npy, tt_deg.shape[1])
        tt_full[:h, :w] = tt_deg[:h, :w]
        if h < npz:
            tt_full[h:, :] = tt_full[h - 1, :]
        if w < npy:
            tt_full[:, w:] = tt_full[:, w - 1:w]
        tt_deg = tt_full
    return tt_deg


def _ring_windows_deg(rt: RingTable, *, min_window_deg: float,
                       max_window_deg: float,
                       neighbor_frac: float = 0.4) -> "tuple[np.ndarray, np.ndarray]":
    """Per-ring half-window in 2θ deg, capped at a fraction of the gap to
    the nearest neighbouring ring, so masks are guaranteed non-overlapping
    (a pixel is never eligible for two rings at once). Also returns the raw
    neighbour gap per ring (unclipped), so callers can additionally decide
    to drop hopelessly close multiplets outright rather than merely
    narrowing their window -- see ``min_ring_gap_deg`` in :func:`pick_points`.
    """
    tt = np.asarray(rt.two_theta_deg, dtype=np.float64)
    order = np.argsort(tt)
    tt_sorted = tt[order]
    half = np.full(len(tt), max_window_deg, dtype=np.float64)
    neighbor_gap_out = np.full(len(tt), np.inf, dtype=np.float64)
    for j, i in enumerate(order):
        gaps = []
        if j > 0:
            gaps.append(tt_sorted[j] - tt_sorted[j - 1])
        if j < len(tt_sorted) - 1:
            gaps.append(tt_sorted[j + 1] - tt_sorted[j])
        neighbor_gap = min(gaps) if gaps else np.inf
        neighbor_gap_out[i] = neighbor_gap
        half[i] = float(np.clip(neighbor_frac * neighbor_gap,
                                  min_window_deg, max_window_deg))
    return half, neighbor_gap_out


def _refine_subpixel(img: np.ndarray, z0: int, y0: int, baseline: float,
                      half: int) -> "tuple[float, float]":
    """Intensity-weighted centroid in a small window around an integer
    local-max pixel. A raw integer pixel index alone carries up to ~0.5px
    of quantisation bias against the true sub-pixel peak location."""
    npz, npy = img.shape
    zlo, zhi = max(0, z0 - half), min(npz, z0 + half + 1)
    ylo, yhi = max(0, y0 - half), min(npy, y0 + half + 1)
    patch = img[zlo:zhi, ylo:yhi]
    w = np.clip(patch - baseline, 0.0, None)
    if w.sum() <= 0:
        return float(z0), float(y0)
    zz_, yy_ = np.mgrid[zlo:zhi, ylo:yhi]
    zc = float((zz_ * w).sum() / w.sum())
    yc = float((yy_ * w).sum() / w.sum())
    return zc, yc


def pick_points(
    image: np.ndarray,
    v1_seed: V1Params,
    rt: RingTable,
    *,
    downsample: int = 4,
    footprint_px: int = 7,
    snr_threshold: float = 5.0,
    min_window_deg: float = 0.03,
    max_window_deg: float = 0.5,
    min_ring_gap_deg: float = 0.0,
    subpixel_half_px: int = 2,
    edge_margin_px: int = 5,
    panel_mask: Optional[np.ndarray] = None,
    mask_erode_iter: int = 2,
    dtype=torch.float64, device="cpu",
) -> PickedPoints:
    """Pick one-shot, geometry-independent local-maximum peak points.

    Parameters
    ----------
    image : raw detector image, shape ``(NrPixelsZ, NrPixelsY)`` (row=Z,
        col=Y -- the convention used throughout this package's seed
        modules).
    v1_seed : any reasonably-priced starting geometry -- used ONLY to place
        the coarse per-ring annular mask; never to bin, window, or
        otherwise distort what counts as "the peak" inside it.
    rt : the calibrant's RingTable (from ``build_ring_table``).
    min_ring_gap_deg : rings whose gap to their nearest neighbour (in 2θ)
        is below this are DROPPED ENTIRELY, not merely narrow-windowed.
        Narrowing a window still leaves a ring vulnerable to
        cross-contamination when the combined seed-geometry error and
        per-pixel discretisation noise approach the gap itself; unlike a
        binned/windowed extraction, a point-pick has no mechanism to
        jointly resolve two overlapping rings, so the safer choice is to
        not trust hopelessly close ones at all. Default 0.0 keeps every
        ring (narrow-window only) for backward compatibility; callers
        targeting multiplet-heavy calibrants (e.g. CeO2's closely-spaced
        high-index reflections) should set this explicitly (e.g. a few
        tenths of a degree).

    Returns
    -------
    :class:`PickedPoints` -- ``ring_idx`` values are row indices into
    ``rt``'s arrays, matching the convention ``FittedDataset.ring_idx``
    already uses elsewhere in this package.
    """
    npz, npy = image.shape
    img, valid_mask = apply_mask_for_arcs(image, mask=panel_mask,
                                            erode_iter=mask_erode_iter)

    if edge_margin_px > 0:
        valid_mask[:edge_margin_px, :] = False
        valid_mask[-edge_margin_px:, :] = False
        valid_mask[:, :edge_margin_px] = False
        valid_mask[:, -edge_margin_px:] = False

    tt_seed_deg = _seed_two_theta_map(
        v1_seed, npy=npy, npz=npz, downsample=downsample,
        dtype=dtype, device=device,
    )

    # Global local-maximum mask -- computed ONCE (a pixel is a candidate
    # peak iff it is the max value in its own footprint neighbourhood). No
    # ring-specific binning or profiling: just "is this pixel a real local
    # intensity maximum", evaluated everywhere on the raw image.
    local_max = (img == ndimage.maximum_filter(
        img, size=footprint_px, mode="nearest")) & valid_mask

    half_windows, neighbor_gap = _ring_windows_deg(
        rt, min_window_deg=min_window_deg, max_window_deg=max_window_deg,
    )

    Y_all, Z_all, ring_all, snr_all = [], [], [], []
    n_by_ring = {}
    n_rings = len(rt.two_theta_deg)
    for i in range(n_rings):
        if neighbor_gap[i] < min_ring_gap_deg:
            n_by_ring[i] = 0
            continue
        tt_ring = float(rt.two_theta_deg[i])
        half = float(half_windows[i])
        ring_mask = (np.abs(tt_seed_deg - tt_ring) < half) & valid_mask
        if not np.any(ring_mask):
            n_by_ring[i] = 0
            continue
        ring_vals = img[ring_mask]
        baseline = float(np.median(ring_vals))
        mad = float(np.median(np.abs(ring_vals - baseline)))
        noise = max(1.4826 * mad, 1e-6)

        cand_mask = local_max & ring_mask
        zz, yy = np.nonzero(cand_mask)
        if zz.size == 0:
            n_by_ring[i] = 0
            continue
        peak_vals = img[zz, yy]
        snr = (peak_vals - baseline) / noise
        keep = snr >= snr_threshold
        n_by_ring[i] = int(np.count_nonzero(keep))
        if not np.any(keep):
            continue
        zz_k, yy_k, snr_k = zz[keep], yy[keep], snr[keep]
        if subpixel_half_px > 0:
            refined = [_refine_subpixel(img, int(z), int(y), baseline,
                                          subpixel_half_px)
                       for z, y in zip(zz_k, yy_k)]
            Z_ref = np.array([r[0] for r in refined], dtype=np.float64)
            Y_ref = np.array([r[1] for r in refined], dtype=np.float64)
        else:
            Z_ref = zz_k.astype(np.float64)
            Y_ref = yy_k.astype(np.float64)
        Y_all.append(Y_ref)
        Z_all.append(Z_ref)
        ring_all.append(np.full(int(np.count_nonzero(keep)), i, dtype=np.int64))
        snr_all.append(snr_k)

    if not Y_all:
        return PickedPoints(
            Y_pix=np.zeros(0), Z_pix=np.zeros(0),
            ring_idx=np.zeros(0, dtype=np.int64), snr=np.zeros(0),
            n_by_ring=n_by_ring,
        )
    return PickedPoints(
        Y_pix=np.concatenate(Y_all),
        Z_pix=np.concatenate(Z_all),
        ring_idx=np.concatenate(ring_all),
        snr=np.concatenate(snr_all),
        n_by_ring=n_by_ring,
    )


__all__ = ["PickedPoints", "pick_points"]
