"""E-step: integrate calibrant image, extract per-(ring, η-bin) peak positions.

Strategy:
  1. Build a uniform-R, uniform-η bin grid spanning the calibrant ring range
     (subsetting per-ring windows after integration).
  2. Build midas_integrate's PixelMap + CSR from the current geometry.
  3. Integrate the image into a 2D (R, η) cake.
  4. For each (ring, η-bin), compute a weighted centroid in the radial window
     to get R_fit (px).  v0.1 uses centroid; future versions can swap in a
     pseudo-Voigt LM via midas_peakfit.lm_solve_generic.
  5. Convert (R_fit, η_bin_center) → (Y_pix, Z_pix) via midas_integrate's
     Newton-Raphson inverse.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from midas_integrate.detector_mapper import build_map
from midas_integrate.geometry import (
    build_tilt_matrix,
    invert_REta_to_pixel,
    invert_REta_to_pixel_batch,
)
from midas_integrate.kernels import build_csr, integrate
from midas_integrate.params import IntegrationParams

from .params import CalibrationParams
from .refine import FittedPoint
from .rings import RingTable


# ── adaptive, memory-aware cake binning mode ─────────────────────────────────
# The bilinear (sub-pixel) cake gives the most accurate ring centroids, but its
# CSR build expands every pixel→bin entry into 4 corner weights, so for a large
# detector (e.g. 2880²) it transiently needs several GB and OOMs on a
# memory-limited machine. When ``cake_mode="auto"`` (the default) we estimate
# the bilinear build's peak from the entry count and fall back to floor binning
# (≈4× lighter, only marginally less sub-pixel-accurate — negligible for the
# centroid-based calibration here) when available RAM can't safely cover it.
# Calibrated on a 2880² CeO2 frame (≈33 M pixel-bin entries): the FULL bilinear
# calibrate() pipeline peaks ~12.5 GB → ~400 B/entry. We size the estimate to
# the full-pipeline peak (not just the CSR build) and keep a generous headroom,
# so bilinear is only chosen with real margin. Floor binning gave a bit-for-bit
# identical calibration on this data (same Lsd, same strain) at a lower peak, so
# biasing toward floor on large detectors / modest machines costs no accuracy.
_BILINEAR_BYTES_PER_ENTRY = 400       # full-pipeline bytes per pixel-bin entry
_RAM_HEADROOM_BYTES = 3 * 1024 ** 3   # room for the image, torch/LM state, OS


def _available_ram_bytes() -> Optional[int]:
    """Best-effort available physical RAM in bytes; None if undetectable.

    Tries psutil (cross-platform, a declared dependency), then POSIX
    ``os.sysconf`` (Linux/macOS), then the Win32 ``GlobalMemoryStatusEx`` via
    ctypes (so detection still works on Windows even if psutil is absent).
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import os
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, AttributeError, OSError):
        pass
    try:  # Windows native fallback
        import ctypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def _choose_cake_mode(n_entries: int, requested: str = "auto") -> str:
    """Resolve the cake integration mode.

    ``requested`` is ``"auto"`` (RAM-checked bilinear→floor fallback),
    ``"bilinear"`` (force, maximum accuracy), or ``"floor"`` (force, lightest).
    """
    if requested != "auto":
        return requested
    avail = _available_ram_bytes()
    if avail is None:
        return "bilinear"          # can't tell — keep the accurate default
    needed = n_entries * _BILINEAR_BYTES_PER_ENTRY + _RAM_HEADROOM_BYTES
    if avail < needed:
        warnings.warn(
            f"midas_calibrate: insufficient memory for bilinear caking "
            f"(~{needed / 1024**3:.1f} GB estimated for {n_entries:,} pixel-bin "
            f"entries, ~{avail / 1024**3:.1f} GB available) — falling back to "
            f"floor binning. Pass cake_mode='bilinear' to force it, or run on a "
            f"higher-RAM machine for maximum sub-pixel accuracy.",
            RuntimeWarning, stacklevel=3,
        )
        return "floor"
    return "bilinear"


def _calibration_to_integration_params(
    params: CalibrationParams, *, R_min: float, R_max: float, R_bin_size: float, eta_bin_size: float,
) -> IntegrationParams:
    ip = IntegrationParams()
    ip.NrPixelsY = params.NrPixelsY
    ip.NrPixelsZ = params.NrPixelsZ
    ip.pxY = params.pxY
    ip.pxZ = params.pxZ if params.pxZ > 0 else params.pxY
    ip.Lsd = params.Lsd
    ip.BC_y = params.BC_y
    ip.BC_z = params.BC_z
    ip.tx = params.tx
    ip.ty = params.ty
    ip.tz = params.tz
    for i in range(15):
        setattr(ip, f"p{i}", getattr(params, f"p{i}"))
    ip.RhoD = params.RhoD if params.RhoD > 0 else params.MaxRingRad
    ip.Parallax = params.Parallax
    ip.Wavelength = params.Wavelength
    ip.RMin = float(R_min)
    ip.RMax = float(R_max)
    ip.RBinSize = float(R_bin_size)
    ip.EtaMin = -180.0
    ip.EtaMax = 180.0
    ip.EtaBinSize = float(eta_bin_size)
    ip.SolidAngleCorrection = 0
    ip.PolarizationCorrection = 0
    return ip


@dataclass
class CakeProfile:
    R_centers: np.ndarray
    eta_centers: np.ndarray
    intensity: np.ndarray  # [n_R, n_eta]
    #: Per-cell detector coverage — the summed pixel weight binned into each
    #: (R, η) cell, i.e. what you get by integrating an image of ones through
    #: the same map. Zero means NO detector pixel reaches that cell: it is off
    #: the edge, in a module gap, or masked.
    #:
    #: This exists so ``extract_fitted_points`` can tell an uncovered cell from
    #: a covered one that happens to read zero. Distinguishing those matters:
    #: a radial window containing uncovered cells yields a truncated peak and a
    #: biased centroid, and that bias is what dragged the geometry.
    #:
    #: ``None`` for callers that build a CakeProfile by hand; the consumer then
    #: falls back to treating an exactly-zero intensity as uncovered, which is
    #: right for real data but cannot tell the two cases apart.
    coverage: Optional[np.ndarray] = None  # [n_R, n_eta]


def integrate_cake(
    params: CalibrationParams,
    image: np.ndarray,
    rt: RingTable,
    *, dark: Optional[np.ndarray] = None,
    cake_mode: str = "auto",
    mask: Optional[np.ndarray] = None,
) -> CakeProfile:
    """Build CSR + integrate the image into a uniform (R, η) cake.

    ``cake_mode`` selects the binning kernel: ``"auto"`` (default) uses
    bilinear sub-pixel binning but falls back to floor binning when available
    RAM can't safely cover the bilinear CSR build (large detectors on
    memory-limited machines); ``"bilinear"`` / ``"floor"`` force the choice.

    Parameters
    ----------
    mask :
        Optional bad-pixel mask, same shape and ORIENTATION as ``image``
        (apply the same ``ImTransOpt`` to both — this function does not
        transform it for you). **Nonzero means BAD**, matching
        ``midas_integrate.detector_mapper.build_map`` (``mask == 1.0`` is
        masked) and the shipped ``mask_upd.tif``, whose nonzero pixels
        coincide exactly with the dead pixels of the Pilatus frame.
        (Note the comment in the example ``parameters.txt`` says "0 = masked";
        that is wrong — it would mask 92 % of the detector.)

        Added 2026-08-29. There was previously **no way to pass a mask into
        the calibration at all**: ``CalibrationParams`` has no mask field and
        neither this function nor :func:`run_estep` nor
        ``midas_calibrate_v2.calibrate`` accepted one, even though
        ``IntegrationParams`` carries ``MaskFile`` and ``build_map`` honours a
        mask. Bad pixels therefore entered the cake as genuine zeros and
        diluted every cell they touched, dragging the intensity-weighted
        radial centroid off the ring.

        Masked pixels are removed from the numerator AND the denominator.
        Zeroing them alone would not help — that is exactly the state the code
        was already in.
    """
    if dark is not None:
        image = image - dark
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} != image shape {image.shape}; the "
                f"mask must already carry the same ImTransOpt as the image")
        bad = np.asarray(mask != 0, dtype=bool)
        if not bad.any():
            # An all-good mask must be an EXACT no-op. The masked path
            # normalises by ``CSR @ good`` while the unmasked one uses the
            # map's own ``area_per_bin``; those agree only to ~5e-8 relative,
            # so without this the 9th significant figure would shift merely
            # because a mask was supplied.
            mask = None
        else:
            image = np.where(bad, 0.0, image)

    # R range: half-Width margin around min/max ring radius.
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    R_min = max(0.0, float(rt.r_ideal_px.min()) - half_px - 1.0)
    R_max = float(rt.r_ideal_px.max()) + half_px + 1.0
    ip = _calibration_to_integration_params(
        params, R_min=R_min, R_max=R_max,
        R_bin_size=params.RBinSize, eta_bin_size=params.EtaBinSize,
    )

    pmap_result = build_map(ip, verbose=False)
    from midas_integrate.bin_io import PixelMap as _PixelMap
    pmap = _PixelMap(
        pxList=pmap_result.pxList,
        counts=pmap_result.counts,
        offsets=pmap_result.offsets,
        map_header=None, nmap_header=None,
    )
    # Choose bilinear vs floor from available RAM (large detectors can OOM the
    # bilinear CSR build); the entry count drives the estimate.
    mode = _choose_cake_mode(int(pmap.counts.sum()), cake_mode)
    geom = build_csr(
        pmap,
        n_r=ip.n_r_bins, n_eta=ip.n_eta_bins,
        n_pixels_y=ip.NrPixelsY, n_pixels_z=ip.NrPixelsZ,
        bc_y=ip.BC_y, bc_z=ip.BC_z,
        device="cpu", dtype=torch.float64,
        build_modes=(mode,),
    )

    img_t = torch.as_tensor(image, dtype=torch.float64).contiguous()

    # Per-cell detector coverage: the summed weight of LIVE pixels reaching
    # each cell. Zero means no live pixel does — off the detector, in a module
    # gap, or masked.
    #
    # ``geom.area_per_bin`` is the Σ areaWeight per bin that ``normalize=True``
    # would divide by. Using it rather than integrating an image of ones costs
    # nothing and cannot drift from the normaliser — ``intensity`` and
    # ``coverage`` are then guaranteed to be the same ratio's numerator and
    # denominator. (Measured equal to ``CSR @ ones`` to 4.7e-8 relative, with
    # exact agreement on which bins are zero, in both floor and bilinear modes.)
    coverage = geom.area_per_bin.reshape(ip.n_r_bins, ip.n_eta_bins).numpy()

    if mask is None:
        cake = integrate(img_t, geom, mode=mode, normalize=True).numpy()
    else:
        # Masked pixels must leave the DENOMINATOR too. Normalising by the full
        # ``area_per_bin`` would divide live signal by dead area — precisely
        # the dilution this exists to remove.
        #
        # Push the GOOD-pixel indicator through the CSR rather than subtracting
        # the bad one from ``area_per_bin``. Subtracting cancels two quantities
        # computed by different routes (the map's accumulated area vs an SpMV),
        # which agree only to ~5e-8 relative — so a fully masked cell came out
        # at ~9e-8 instead of 0 and still read as "covered". Integrating the
        # good indicator is exact: no live pixel, no coverage.
        good_t = torch.as_tensor((~bad).astype(np.float64)).contiguous()
        coverage = integrate(good_t, geom, mode=mode, normalize=False).numpy()
        raw = integrate(img_t, geom, mode=mode, normalize=False).numpy()
        cake = np.where(coverage > 0.0, raw / np.where(coverage > 0.0,
                                                       coverage, 1.0), 0.0)

    R_edges = np.linspace(ip.RMin, ip.RMin + ip.RBinSize * ip.n_r_bins, ip.n_r_bins + 1)
    eta_edges = np.linspace(ip.EtaMin, ip.EtaMax, ip.n_eta_bins + 1)
    return CakeProfile(
        R_centers=0.5 * (R_edges[:-1] + R_edges[1:]),
        eta_centers=0.5 * (eta_edges[:-1] + eta_edges[1:]),
        intensity=cake,
        coverage=coverage,
    )


def extract_fitted_points(
    cake: CakeProfile, rt: RingTable, params: CalibrationParams,
    *, snr_min: float = 1.0, min_cell_coverage: float = 0.5,
) -> List[FittedPoint]:
    """Per (ring × η-bin): centroid in the radial window → (R_fit, η) → (Y_pix, Z_pix).

    Vectorised: the η-axis centroid + SNR filter run as numpy array ops
    per ring, and the Newton inversion runs as a single batched call
    over all surviving (ring, η) pairs. This replaces ~1000 scalar calls
    into the inverter (3-5 rings × 360 η-bins) with one array call.
    """
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    eta_centers = np.asarray(cake.eta_centers, dtype=np.float64)

    R_chunks: List[np.ndarray] = []
    Eta_chunks: List[np.ndarray] = []
    ring_idx_chunks: List[np.ndarray] = []
    snr_chunks: List[np.ndarray] = []
    snr_base_chunks: List[np.ndarray] = []

    for ring_i, r_ideal in enumerate(rt.r_ideal_px):
        idx = np.where(np.abs(cake.R_centers - r_ideal) <= half_px)[0]
        if idx.size < 3:
            continue
        R_window = cake.R_centers[idx].astype(np.float64)        # (n_R,)
        I_block = cake.intensity[idx, :].astype(np.float64)      # (n_R, n_eta)
        cov_block = (cake.coverage[idx, :].astype(np.float64)
                     if getattr(cake, "coverage", None) is not None else None)
        # Per-η baseline subtract across the radial window (matches the
        # ``I - I.min()`` per-η-bin step of the previous scalar loop).
        I = np.maximum(I_block - I_block.min(axis=0, keepdims=True), 0.0)
        tot = I.sum(axis=0)                                      # (n_eta,)
        valid_tot = tot > 0.0
        if not valid_tot.any():
            continue
        # Centroid R_fit per η; safe-divide on bins with zero total.
        safe_tot = np.where(valid_tot, tot, 1.0)
        R_fit = (I * R_window[:, None]).sum(axis=0) / safe_tot
        peak = I.max(axis=0)
        mean = I.mean(axis=0) + 1e-12
        snr = peak / mean
        # Baseline-referenced SNR for the per-ring quality filter: subtract a
        # straight line through the two window ends, then compare the peak
        # height to the scatter at those ends.  A weak ring riding a sloping
        # background scores high on peak/mean but low here, which is the
        # discrimination the ring filter needs.
        n_win = I_block.shape[0]
        if n_win >= 5:
            edge = max(2, n_win // 8)
            lo = I_block[:edge].mean(axis=0)
            hi = I_block[-edge:].mean(axis=0)
            ramp = np.linspace(0.0, 1.0, n_win)[:, None]
            base_lin = lo[None, :] + (hi - lo)[None, :] * ramp
            height = (I_block - base_lin).max(axis=0)
            # Noise floor from counting statistics.  The scatter at the window
            # ends alone goes to zero wherever the ends happen to be flat (a
            # masked gap, an off-panel region, a saturated shelf) and the ratio
            # then explodes — measured maxima of ~1e7 on a real frame, which
            # made any SNR threshold inert because almost every ring "passed".
            # A peak cannot be known better than sqrt(N) on its own baseline.
            ends = np.concatenate([I_block[:edge], I_block[-edge:]], axis=0)
            base_level = np.maximum(ends.mean(axis=0), 0.0)
            noise = np.maximum(ends.std(axis=0), np.sqrt(np.maximum(base_level, 1.0)))
            snr_base = height / noise
        else:
            snr_base = np.zeros_like(snr)

        # Reject η-bins whose radial window is CLIPPED — by the detector edge,
        # a module gap, or the mask.
        #
        # R_fit above is an intensity-weighted centroid over a FIXED radial
        # window. Where the window runs off the detector the missing cake cells
        # integrate to exactly 0.0, so the peak is truncated and the centroid is
        # dragged toward the surviving side. Nothing caught that: ``valid_tot``
        # only requires SOME signal in the window, so a half-truncated peak
        # passed and contributed a biased point.
        #
        # This is why the OUTER rings were poisoning the fit — they are the ones
        # whose windows leave the detector — and it is why simply capping the
        # ring radius appeared to "fix" it. That cap is the wrong remedy: the
        # outer rings carry the tilt and distortion leverage, and a partial arc
        # is perfectly good data. Only the clipped BINS are bad, so only those
        # are dropped; the rest of a partial ring is kept and used.
        #
        # MEASURED on a real 48-panel Pilatus against the C reference, this
        # recovers most of what the radius cap bought while keeping the outer
        # rings (see SCIENCE_AUDIT_integrate_calibrate.md).
        #
        # Coverage is the honest marker. ``cake.coverage`` is the same map
        # applied to an image of ones, so a cell's value is the pixel weight
        # that actually reached it; zero means no detector pixel does. That
        # separates "uncovered" from "covered but reads zero", which an
        # intensity test cannot do.
        #
        # A cell that is only PARTIALLY covered — the detector edge cuts
        # through it, or a module gap clips it — is also bad: its intensity is
        # the mean over the covered fraction, so it under-represents that
        # radius and still skews the centroid, just less violently than an
        # empty cell. Require each cell in the window to reach
        # ``min_cell_coverage`` of the best-covered cell in the same window,
        # which is scale-free (bin area grows with R, so an absolute threshold
        # would not travel across the detector).
        #
        # The two tests are applied TOGETHER, not one instead of the other.
        # They fail on different things and neither implies the other:
        #   * a cell can be geometrically covered and still read exactly zero
        #     (dead pixels, a mask applied to the image rather than to the map,
        #     a panel that was not read out) — coverage cannot see that;
        #   * a cell can carry stray counts and still lie off the detector
        #     (scatter, a hot pixel bleeding into a corner bin) — the intensity
        #     test cannot see that.
        window_complete = ~(I_block <= 0.0).any(axis=0)
        if cov_block is not None:
            cov_ref = cov_block.max(axis=0, keepdims=True)
            cov_ref = np.where(cov_ref > 0.0, cov_ref, 1.0)
            window_complete &= (cov_block >= min_cell_coverage * cov_ref).all(axis=0)
        keep = valid_tot & (snr >= snr_min) & window_complete
        if not keep.any():
            continue
        R_chunks.append(R_fit[keep])
        Eta_chunks.append(eta_centers[keep])
        ring_idx_chunks.append(np.full(int(keep.sum()), ring_i, dtype=np.int64))
        snr_chunks.append(snr[keep])
        snr_base_chunks.append(snr_base[keep])

    if not R_chunks:
        return []

    R_targets = np.concatenate(R_chunks)
    Eta_targets = np.concatenate(Eta_chunks)
    ring_idxs = np.concatenate(ring_idx_chunks)
    snrs = np.concatenate(snr_chunks)
    snr_bases = np.concatenate(snr_base_chunks)

    Y_pix, Z_pix = invert_REta_to_pixel_batch(
        R_targets, Eta_targets,
        Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
        Lsd=params.Lsd, RhoD=(params.RhoD if params.RhoD > 0 else params.MaxRingRad),
        px=px, parallax=params.Parallax,
    )

    return [
        FittedPoint(
            Y_pix=float(Y_pix[i]), Z_pix=float(Z_pix[i]),
            ring_idx=int(ring_idxs[i]), snr=float(snrs[i]),
            snr_baseline=float(snr_bases[i]),
        )
        for i in range(R_targets.shape[0])
    ]


def run_estep(
    params: CalibrationParams,
    image: np.ndarray,
    rt: RingTable,
    *, dark: Optional[np.ndarray] = None,
    cake_mode: str = "auto",
    mask: Optional[np.ndarray] = None,
) -> Tuple[CakeProfile, List[FittedPoint]]:
    cake = integrate_cake(params, image, rt, dark=dark, cake_mode=cake_mode,
                          mask=mask)
    fits = extract_fitted_points(cake, rt, params, snr_min=params.SNRMin)
    return cake, fits
