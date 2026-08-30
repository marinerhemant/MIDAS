"""Pure-Python DetectorMapper.

Mirrors ``mapper_build_map`` in MapperCore.c plus the surrounding setup in
DetectorMapper.c. Produces Map.bin / nMap.bin in the v3 format with the
same parameter-hash header used by the C version.

Performance strategy:
- Vectorized pre-pass with numpy: compute (R, Eta) for all four corners
  of every pixel in one shot. This gives bounding (RMi, RMa, EtaMi, EtaMa)
  per pixel which selects candidate (R, Eta) bins.
- The hot inner loop (per-pixel candidate-bin sweep + Green's-theorem
  polygon-arc intersection) is JIT'd with numba (``parallel=True``) and
  uses ``prange`` over rows for OpenMP-equivalent parallelism. With numba
  available, throughput matches the C OMP DetectorMapper on the same
  hardware.
- Pure-Python fallback (``_use_numba=False`` or numba unavailable) is
  retained for portability and testing — slow on large detectors.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from midas_integrate.bin_io import (
    MapHeader,
    PXLIST_DTYPE,
    PixelMap,
    compute_param_hash,
    write_map,
)
from midas_integrate.geometry import (
    DEG2RAD,
    PIXEL_CORNER_OFFSETS,
    REta_to_YZ_scalar,
    build_bin_edges,
    build_q_bin_edges_in_R,
    build_tth_bin_edges_in_R,
    build_tilt_matrix,
    pixel_to_REta,
    pixel_bin_intersect,
    solid_angle_factor,
)
from midas_integrate.panel import (
    Panel,
    empty_panel_array,
    generate_panels,
    load_panel_shifts,
    panels_to_array,
)
from midas_integrate.params import IntegrationParams
from midas_integrate.residual_corr import (
    ResidualCorrection,
    empty_residual_corr_array,
    load_residual_correction_map,
)

try:
    from midas_integrate._mapper_numba import (
        map_kernel as _numba_map_kernel,
        map_kernel_subpixel as _numba_map_kernel_subpixel,
    )
    HAVE_NUMBA = True
except ImportError:
    _numba_map_kernel = None
    _numba_map_kernel_subpixel = None
    HAVE_NUMBA = False


class MapTruncationWarning(UserWarning):
    """Map entries were dropped because a detector row filled its buffer.

    Raised (as a warning) by ``build_map``. It is deliberately not gated on
    ``verbose``: truncation drops ``frac`` and ``areaWeight`` together, so a
    normalised profile can still look correct while absolute flux, per-bin
    area, and bin occupancy are wrong.
    """


#: Default ceiling on the transient per-row map buffer
#: (NrPixelsZ × per_row_max × 6 × 8 bytes).
DEFAULT_MAP_BUFFER_BYTES = 4 * 1024 ** 3

#: Smallest (pixel, bin) overlap area kept in the map, in pixel².
#:
#: This was 1e-5, which discarded the thin slivers a pixel contributes to the
#: bins it barely reaches — and those slivers are exactly what makes the map
#: area-conserving. MEASURED on a 48x48 detector, comparing each pixel's summed
#: area against its own mapped quad area (which is 1 px² only when there is no
#: tilt; under tilt the mapped quad carries the Jacobian):
#:
#:     threshold   entries   max |sum - quad|, no tilt / ty=2 deg
#:       1e-5        3309      9.6e-06  /  8.6e-06
#:       1e-7        3316      2.9e-14  /  3.1e-08
#:       1e-9        3316      2.9e-14  /  4.2e-14      <- saturated
#:      1e-12        3316      2.9e-14  /  4.2e-14
#:
#: 1e-9 is where conservation reaches machine precision, and the entry count is
#: already the same as at 1e-12 — the extra headroom buys nothing.
#:
#: Do NOT lower it further. Two mathematically equivalent configurations (zero
#: distortion vs no distortion; panel shifts of zero vs no panels) evaluate the
#: corners in different orders and so disagree at ~1e-13 of IEEE rounding
#: noise. A threshold at or below that noise makes the map's SPARSITY PATTERN
#: differ between them — the areas still agree, but the entry counts do not,
#: which is both non-deterministic and a real headache for the parity tests
#: that compare maps entry by entry. 1e-9 sits four orders above that floor.
MIN_PIXEL_BIN_AREA = 1e-9


def estimate_per_row_max(params,
                         max_bytes: int = DEFAULT_MAP_BUFFER_BYTES) -> int:
    """Per-detector-row entry budget implied by the binning geometry.

    One pixel maps onto roughly ``(1/RBinSize)`` radial bins and, at the
    innermost radius, ``(deg subtended by one pixel) / EtaBinSize`` azimuthal
    bins; sub-pixel splitting multiplies that by ``SubPixelLevel**2``. A fixed
    guess (the old ``max(NrPixelsY*4, 4000)``) underestimates this badly at
    fine ``RBinSize`` and drops entries silently.

    The result is padded for corner effects and then clamped so the transient
    buffer stays under ``max_bytes``. When the clamp bites — most easily with
    ``SubPixelLevel > 1``, where a detector row lying along a cardinal azimuth
    is split everywhere — the build emits ``MapTruncationWarning`` rather than
    dropping entries quietly.
    """
    n_y = max(int(params.NrPixelsY), 1)
    n_z = max(int(params.NrPixelsZ), 1)

    r_bin = float(getattr(params, "RBinSize", 0.0) or 0.0)
    # +2 for the two partially-covered bins at each end of the span
    r_bins_per_px = (1.0 / r_bin + 2.0) if r_bin > 0 else 3.0

    eta_bin = float(getattr(params, "EtaBinSize", 0.0) or 0.0)
    r_min = float(getattr(params, "RMin", 0.0) or 0.0)
    if eta_bin > 0 and r_min > 0:
        # angle subtended by one pixel at the innermost ring, in degrees
        deg_per_px = math.degrees(1.0 / r_min)
        eta_bins_per_px = deg_per_px / eta_bin + 2.0
    else:
        eta_bins_per_px = 3.0

    spl = max(int(getattr(params, "SubPixelLevel", 1) or 1), 1)

    per_px = r_bins_per_px * eta_bins_per_px * (spl * spl)
    est = int(math.ceil(n_y * per_px * 1.30))          # 30 % headroom

    # Clamp to the memory budget: buffer is n_z × per_row × 6 float64.
    budget = max(int(max_bytes // (n_z * 6 * 8)), 1)
    return int(min(max(est, 4000), budget))


@dataclass
class BuildMapResult:
    """Output of build_map — ready to pass to bin_io.write_map."""
    pxList: np.ndarray              # PXLIST_DTYPE structured, shape (n_entries,)
    counts: np.ndarray              # int32, shape (n_bins,)
    offsets: np.ndarray             # int32, shape (n_bins,)
    bin_mask_flag: np.ndarray       # int32, shape (n_bins,)
    n_continued: tuple[int, int, int] = (0, 0, 0)
    elapsed_s: float = 0.0
    n_bins_filled: int = 0
    backend: str = "numba"          # "numba" | "python"

    def as_pixel_map(self,
                     map_header: Optional[MapHeader] = None) -> PixelMap:
        return PixelMap(
            pxList=self.pxList,
            counts=self.counts,
            offsets=self.offsets,
            map_header=map_header,
            nmap_header=map_header,
        )


def _apply_trans_opt_forward(arr: np.ndarray,
                             trans_opt: Sequence[int],
                             NrPixelsY: int, NrPixelsZ: int) -> np.ndarray:
    """Forward FlipLR/FlipUD/transpose, mirroring midas_image_transform in C.

    ``arr`` has shape ``(NrPixelsZ, NrPixelsY)`` and is returned with the same
    shape (transpose is only legal when NrPixelsY == NrPixelsZ).
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    for opt in trans_opt:
        if opt == 0:
            continue
        if opt == 1:
            out = np.ascontiguousarray(out[:, ::-1])
        elif opt == 2:
            out = np.ascontiguousarray(out[::-1, :])
        elif opt == 3:
            if NrPixelsY != NrPixelsZ:
                raise ValueError(
                    "TransOpt=3 (transpose) requires NrPixelsY == NrPixelsZ "
                    f"(got {NrPixelsY} x {NrPixelsZ})"
                )
            out = np.ascontiguousarray(out.T)
    return out


def load_distortion_maps(
    filename: str | Path, params: IntegrationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a binary distortion file and apply the same TransOpt as the image.

    File format (matches FF_HEDM/src/DetectorMapper.c lines 661-674):

      [NrPixelsY × NrPixelsZ float64] — Δy map, in pixel-Y units, row-major
      [NrPixelsY × NrPixelsZ float64] — Δz map, in pixel-Z units, row-major

    Returns ``(distortion_y, distortion_z)`` each with shape
    ``(NrPixelsZ, NrPixelsY)`` after TransOpt is applied.
    """
    NY = params.NrPixelsY
    NZ = params.NrPixelsZ
    expected = NY * NZ * 8
    raw = Path(filename).read_bytes()
    if len(raw) != 2 * expected:
        raise ValueError(
            f"distortion file {filename}: got {len(raw)} bytes, "
            f"expected {2 * expected} (two {NY}x{NZ} float64 arrays)"
        )
    arr = np.frombuffer(raw, dtype=np.float64)
    dy_raw = arr[:NY * NZ].reshape(NZ, NY).copy()
    dz_raw = arr[NY * NZ:].reshape(NZ, NY).copy()
    dy = _apply_trans_opt_forward(dy_raw, params.TransOpt, NY, NZ)
    dz = _apply_trans_opt_forward(dz_raw, params.TransOpt, NY, NZ)
    return dy, dz


def build_panels_from_params(params: IntegrationParams) -> List[Panel]:
    """Generate panels and load their shifts according to ``params``.

    Returns an empty list when ``NPanelsY`` / ``NPanelsZ`` are not configured.
    """
    if params.NPanelsY <= 0 or params.NPanelsZ <= 0:
        return []
    panels = generate_panels(
        n_panels_y=params.NPanelsY,
        n_panels_z=params.NPanelsZ,
        panel_size_y=params.PanelSizeY,
        panel_size_z=params.PanelSizeZ,
        gaps_y=params.PanelGapsY,
        gaps_z=params.PanelGapsZ,
    )
    if params.PanelShiftsFile:
        load_panel_shifts(params.PanelShiftsFile, panels)
    return panels


def load_residual_corr_from_params(
    params: IntegrationParams,
) -> Optional[ResidualCorrection]:
    """Load the residual correction map referenced by params, if any."""
    if not params.ResidualCorrectionMap:
        return None
    return load_residual_correction_map(
        params.ResidualCorrectionMap,
        NrPixelsY=params.NrPixelsY,
        NrPixelsZ=params.NrPixelsZ,
    )


def _inverse_transform_pixel_arrays(
    NrPixelsY: int, NrPixelsZ: int,
    trans_opt: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of mapper_inverse_transform_pixel for ALL pixels."""
    Yidx, Zidx = np.meshgrid(np.arange(NrPixelsY, dtype=np.int64),
                             np.arange(NrPixelsZ, dtype=np.int64),
                             indexing="xy")        # (Z, Y)
    raw_y = Yidx.copy()
    raw_z = Zidx.copy()
    for opt in reversed(list(trans_opt)):
        if opt == 1:           # flip LR
            raw_y = NrPixelsY - 1 - raw_y
        elif opt == 2:         # flip TB
            raw_z = NrPixelsZ - 1 - raw_z
        elif opt == 3:         # transpose
            raw_y, raw_z = raw_z, raw_y
    return raw_y, raw_z


def _per_pixel_corners_REta(
    *, NrPixelsY: int, NrPixelsZ: int,
    Ycen: float, Zcen: float, Lsd: float, RhoD: float, px: float,
    TRs: np.ndarray,
    p: tuple,
    parallax: float,
):
    """Compute (R, Eta) of all 4 corners of every pixel — vectorized."""
    Yidx, Zidx = np.meshgrid(np.arange(NrPixelsY, dtype=np.float64),
                             np.arange(NrPixelsZ, dtype=np.float64),
                             indexing="xy")
    R_corners = np.empty((NrPixelsZ, NrPixelsY, 4), dtype=np.float64)
    Eta_corners = np.empty((NrPixelsZ, NrPixelsY, 4), dtype=np.float64)
    for c, (dy, dz) in enumerate(PIXEL_CORNER_OFFSETS):
        Y = Yidx + dy
        Z = Zidx + dz
        R_c, E_c = pixel_to_REta(
            Y, Z,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
            p0=p[0], p1=p[1], p2=p[2], p3=p[3], p4=p[4], p5=p[5],
            p6=p[6], p7=p[7], p8=p[8], p9=p[9], p10=p[10], p11=p[11],
            p12=p[12], p13=p[13], p14=p[14],
            parallax=parallax,
        )
        R_corners[..., c] = R_c
        Eta_corners[..., c] = E_c
    Rt_center, Eta_center = pixel_to_REta(
        Yidx, Zidx,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
        p0=p[0], p1=p[1], p2=p[2], p3=p[3], p4=p[4], p5=p[5],
        p6=p[6], p7=p[7], p8=p[8], p9=p[9], p10=p[10], p11=p[11],
        p12=p[12], p13=p[13], p14=p[14],
        parallax=parallax,
    )
    return R_corners, Eta_corners, Rt_center, Eta_center


def _build_pixel_quads(R_corners: np.ndarray, Eta_corners: np.ndarray) -> np.ndarray:
    """(R, Eta) corners → (Y, Z) quad coordinates — shape (NZ, NY, 4, 2)."""
    Y = -R_corners * np.sin(Eta_corners * DEG2RAD)
    Z = R_corners * np.cos(Eta_corners * DEG2RAD)
    return np.stack([Y, Z], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python fallback (slow; mostly for portability + tests)
# ─────────────────────────────────────────────────────────────────────────────
def _build_map_python(
    *, params, R_corners, Eta_corners, cornerYZ, Rt_center, Eta_center,
    raw_y_arr, raw_z_arr,
    r_lo, r_hi, eta_lo, eta_hi, n_r_bins, n_eta_bins,
    NrPixelsY, NrPixelsZ, Lsd, px,
    solid_angle, polarization, pol_fraction, pol_plane_eta_rad, sa_factor,
    mask, flat,
):
    n_bins = n_r_bins * n_eta_bins
    bin_entries: list[list[tuple[float, float, float, float, float]]] = \
        [[] for _ in range(n_bins)]
    bin_mask_flag = np.zeros(n_bins, dtype=np.int32)

    RMi = R_corners.min(axis=-1)
    RMa = R_corners.max(axis=-1)
    EtaMi = Eta_corners.min(axis=-1)
    EtaMa = Eta_corners.max(axis=-1)

    for j in range(NrPixelsZ):
        for i in range(NrPixelsY):
            is_masked = mask is not None and mask[j, i] == 1.0
            rmi = float(RMi[j, i]); rma = float(RMa[j, i])
            emi = float(EtaMi[j, i]); ema = float(EtaMa[j, i])
            r_idx = np.where((r_hi >= rmi) & (r_lo <= rma))[0]
            if r_idx.size == 0:
                continue
            eta_lo_s, eta_hi_s = emi, ema
            if (ema - emi) > 180.0:
                eta_lo_s, eta_hi_s = ema, 360.0 + emi
            e_mask = (
                ((eta_hi >= eta_lo_s) & (eta_lo <= eta_hi_s))
                | ((eta_hi >= eta_lo_s + 360) & (eta_lo <= eta_hi_s + 360))
                | ((eta_hi >= eta_lo_s - 360) & (eta_lo <= eta_hi_s - 360))
            )
            e_idx = np.where(e_mask)[0]
            if e_idx.size == 0:
                continue
            if is_masked:
                for k in r_idx:
                    for l in e_idx:
                        bin_mask_flag[int(k) * n_eta_bins + int(l)] = 1
                continue

            cornerYZ_pix = cornerYZ[j, i]
            Rt_c = float(Rt_center[j, i])
            for k in r_idx:
                RMin = float(r_lo[k]); RMax = float(r_hi[k])
                for l in e_idx:
                    EtaMin = float(eta_lo[l]); EtaMax = float(eta_hi[l])
                    area = pixel_bin_intersect(cornerYZ_pix, RMin, RMax,
                                               EtaMin, EtaMax)
                    if area < MIN_PIXEL_BIN_AREA:
                        continue
                    corrected = area
                    if solid_angle:
                        sa = float(sa_factor[j, i])
                        if sa > 1e-12:
                            corrected /= sa
                    if polarization:
                        # Standard partially-polarized factor (Kahn 1982), the
                        # form pyFAI uses:
                        #     P = 0.5 (1 + cos^2 2th - PF cos(2 deta) sin^2 2th)
                        #
                        # This was 1 - PF sin^2(2th) cos^2(deta), which SCALES
                        # the fully-polarized correction rather than MIXING the
                        # two orthogonal polarization states. The two are
                        # algebraically identical at PF = 1 and diverge below
                        # it: 1.48% at the shipped PF = 0.99, 6.98% at 0.95,
                        # 42.9% at 0.5 (2th <= 60 deg). The physical tell is
                        # that an unpolarized beam has no preferred azimuth, so
                        # its correction cannot depend on eta -- the old form's
                        # did, by 0.375 at 2th = 60 deg with PF = 0.5.
                        #
                        # Changed 2026-08-29. Integrated INTENSITIES move by up
                        # to ~1.5% at high 2theta; peak positions do not.
                        twoTheta = math.atan(Rt_c * px / Lsd)
                        s2t = math.sin(twoTheta)
                        c2t = math.cos(twoTheta)
                        eta_mid = ((EtaMin + EtaMax) * 0.5) * DEG2RAD
                        c2e = math.cos(2.0 * (eta_mid - pol_plane_eta_rad))
                        polFactor = 0.5 * (1.0 + c2t * c2t
                                           - pol_fraction * c2e * s2t * s2t)
                        if polFactor > 1e-6:
                            corrected /= polFactor
                    if flat is not None:
                        f = float(flat[j, i])
                        if f > 1e-12:
                            corrected /= f
                    bin_idx = int(k) * n_eta_bins + int(l)
                    R_bin_center = (RMin + RMax) * 0.5
                    deltaR = Rt_c - R_bin_center
                    bin_entries[bin_idx].append(
                        (float(raw_y_arr[j, i]), float(raw_z_arr[j, i]),
                         float(corrected), float(deltaR), float(area))
                    )
    return bin_entries, bin_mask_flag


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_map(
    params: IntegrationParams,
    *,
    mask: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    panels: Optional[Sequence[Panel]] = None,
    distortion_y: Optional[np.ndarray] = None,
    distortion_z: Optional[np.ndarray] = None,
    residual_corr: Optional[ResidualCorrection] = None,
    auto_load: bool = True,
    n_jobs: int = -1,        # ignored when numba is used (numba uses its own thread pool)
    verbose: bool = False,
    use_numba: Optional[bool] = None,
    per_row_max_entries: Optional[int] = None,
) -> BuildMapResult:
    """Build the full pixel→(R, Eta) bin mapping from an IntegrationParams.

    Args:
        params: Parsed parameter file.
        mask:   Optional (NrPixelsZ, NrPixelsY) float array; pixels where
                mask == 1.0 are flagged as masked and propagated to
                ``bin_mask_flag``.
        flat:   Optional (NrPixelsZ, NrPixelsY) float array of per-pixel
                relative sensitivities. Each ``frac`` (corrected weight) is
                divided by ``flat[pix]`` at CSR-build time, so flat-fielding
                costs nothing at integration time. ``areaWeight`` is left
                unchanged (it is the geometric normaliser).
        panels: Optional list of ``Panel`` objects describing per-panel
                shifts/rotations/Lsd offsets. Mirrors ``Panel.h``. When
                ``None`` and ``auto_load=True`` panels are generated from
                ``params.NPanelsY`` etc.
        distortion_y, distortion_z: Optional (NrPixelsZ, NrPixelsY) float64
                arrays added to the pixel coordinates before the forward
                transform (mirrors ``MapperCore.c`` lines 104-105). When
                both are ``None`` and ``auto_load=True``, they are loaded
                from ``params.DistortionFile`` if present.
        residual_corr: Optional ``ResidualCorrection``; ``Rt += map[Y, Z]``
                is added at the end of the forward transform. When ``None``
                and ``auto_load=True``, loaded from
                ``params.ResidualCorrectionMap`` if present.
        auto_load: If ``True`` and the corresponding kwarg is ``None``,
                load panels / distortion / residual correction from the
                paths in ``params``. Set to ``False`` to require explicit
                arguments.
        n_jobs: Reserved (numba uses ``numba.set_num_threads`` /
                ``NUMBA_NUM_THREADS`` env var).
        use_numba: True/False to force, None to auto-detect.
        per_row_max_entries: maximum bins a single detector row can populate.
                When None, it is derived from the binning geometry by
                ``estimate_per_row_max`` — a fixed guess silently drops
                entries at fine ``RBinSize``. Override only to cap memory,
                and check the truncation warning if you do.
        verbose: Print progress.
    """
    params.validate()
    t0 = time.perf_counter()

    NY = params.NrPixelsY
    NZ = params.NrPixelsZ
    px = params.pxY
    n_eta_bins = params.n_eta_bins
    n_r_bins = params.n_r_bins

    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    p = (params.p0, params.p1, params.p2, params.p3, params.p4, params.p5,
         params.p6, params.p7, params.p8, params.p9, params.p10, params.p11,
         params.p12, params.p13, params.p14)

    if params.q_mode_active:
        r_lo, r_hi, _ = build_q_bin_edges_in_R(
            params.QMin, params.QMax, params.QBinSize,
            params.Lsd, px, params.Wavelength,
        )
        eta_lo = (params.EtaMin
                  + params.EtaBinSize * np.arange(n_eta_bins, dtype=np.float64))
        eta_hi = eta_lo + params.EtaBinSize
    elif params.tth_mode_active:
        r_lo, r_hi, _ = build_tth_bin_edges_in_R(
            params.TthMin, params.TthMax, params.TthBinSize,
            params.Lsd, px,
        )
        eta_lo = (params.EtaMin
                  + params.EtaBinSize * np.arange(n_eta_bins, dtype=np.float64))
        eta_hi = eta_lo + params.EtaBinSize
    else:
        r_lo, r_hi, eta_lo, eta_hi = build_bin_edges(
            params.RMin, params.EtaMin, n_r_bins, n_eta_bins,
            params.RBinSize, params.EtaBinSize,
        )

    if verbose:
        backend = ("numba" if (use_numba if use_numba is not None
                               else HAVE_NUMBA) else "python")
        print(f"[mapper] {NZ}×{NY} px, {n_r_bins}×{n_eta_bins} bins  "
              f"backend={backend}")
        print("[mapper] computing per-pixel (R, Eta) corners…")

    R_corners, Eta_corners, Rt_center, Eta_center = _per_pixel_corners_REta(
        NrPixelsY=NY, NrPixelsZ=NZ,
        Ycen=params.BC_y, Zcen=params.BC_z,
        Lsd=params.Lsd, RhoD=params.RhoD, px=px,
        TRs=TRs, p=p, parallax=params.Parallax,
    )
    cornerYZ = _build_pixel_quads(R_corners, Eta_corners)
    raw_y_arr, raw_z_arr = _inverse_transform_pixel_arrays(NY, NZ, params.TransOpt)

    # Tilt-aware solid-angle factor Ω_pix / Ω_ref = Lsd² · (n̂·r) / |r|³
    # for every detector pixel center. Reduces to cos³(2θ) for zero tilt.
    yy, zz = np.meshgrid(np.arange(NY, dtype=np.float64),
                         np.arange(NZ, dtype=np.float64))
    sa_factor = solid_angle_factor(
        yy, zz,
        Ycen=params.BC_y, Zcen=params.BC_z,
        TRs=TRs, Lsd=params.Lsd, px=px,
    )
    sa_factor = np.ascontiguousarray(sa_factor, dtype=np.float64)
    # Fallback matches IntegrationParams' default: 90 deg = HORIZONTAL in MIDAS
    # eta. A hand-built params object without the attribute must not silently
    # fall back to the old, wrong, vertical plane.
    pol_plane_eta_rad = float(
        getattr(params, "PolarizationPlaneEtaDeg", 90.0)) * (math.pi / 180.0)

    # ── Auto-load panel / distortion / residual-correction arrays ──────
    if panels is None and auto_load:
        loaded_panels = build_panels_from_params(params)
        if loaded_panels:
            panels = loaded_panels
    if (distortion_y is None and distortion_z is None
            and auto_load and params.DistortionFile):
        distortion_y, distortion_z = load_distortion_maps(
            params.DistortionFile, params,
        )
    if residual_corr is None and auto_load:
        residual_corr = load_residual_corr_from_params(params)

    if panels:
        panels_arr = panels_to_array(panels)
    else:
        panels_arr = empty_panel_array()
    n_panels = panels_arr.shape[0]

    has_distortion = (distortion_y is not None and distortion_z is not None)
    if has_distortion:
        distortion_y = np.ascontiguousarray(distortion_y, dtype=np.float64)
        distortion_z = np.ascontiguousarray(distortion_z, dtype=np.float64)
        if (distortion_y.shape != (NZ, NY)
                or distortion_z.shape != (NZ, NY)):
            raise ValueError(
                f"distortion arrays must be ({NZ}, {NY}); got "
                f"{distortion_y.shape} / {distortion_z.shape}"
            )
        distortion_present = 1
    else:
        distortion_y = np.zeros((1, 1), dtype=np.float64)
        distortion_z = np.zeros((1, 1), dtype=np.float64)
        distortion_present = 0

    if residual_corr is not None and residual_corr.map.size > 0:
        corr_map = np.ascontiguousarray(residual_corr.map, dtype=np.float64)
        corr_n_y = int(residual_corr.NrPixelsY)
        corr_n_z = int(residual_corr.NrPixelsZ)
    else:
        corr_map = empty_residual_corr_array()
        corr_n_y = 0
        corr_n_z = 0

    has_corrections = (n_panels > 0) or has_distortion or (corr_n_y > 0)

    use_n = HAVE_NUMBA if use_numba is None else bool(use_numba)
    if use_n and not HAVE_NUMBA:
        raise RuntimeError("use_numba=True but numba is not installed")

    if use_n:
        if per_row_max_entries is None:
            per_row_max_entries = estimate_per_row_max(params)
        if mask is not None:
            mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
            mask_present = 1
        else:
            mask_arr = np.zeros((1, 1), dtype=np.float64)
            mask_present = 0
        if flat is not None:
            flat_arr = np.ascontiguousarray(flat, dtype=np.float64)
            flat_present = 1
        else:
            flat_arr = np.zeros((1, 1), dtype=np.float64)
            flat_present = 0

        # Choose between the precomputed-corners kernel (fast, no
        # sub-pixel splitting) and the in-line subpixel kernel.
        # The subpixel kernel is required whenever SubPixelLevel > 1
        # OR any per-pixel correction (panel / distortion / residual
        # ΔR map) is configured — mirrors C MapperCore.c lines 94-105
        # which apply panel + distortion regardless of SubPixelLevel.
        use_subpixel = int(params.SubPixelLevel) > 1 or has_corrections

        if verbose:
            kernel_name = "subpixel" if use_subpixel else "fast"
            print(f"[mapper] entering numba kernel ({kernel_name}, "
                  f"SubPixelLevel={params.SubPixelLevel}, "
                  f"panels={n_panels}, distortion={bool(has_distortion)}, "
                  f"residual_corr={bool(corr_n_y)}, "
                  f"per_row_max={per_row_max_entries:,}, "
                  f"buffer={NZ * per_row_max_entries * 6 * 8 / 1e9:.2f} GB)…")

        if use_subpixel:
            out_arr, per_row_count, bin_mask_flag = _numba_map_kernel_subpixel(
                r_lo, r_hi, eta_lo, eta_hi,
                n_r_bins, n_eta_bins,
                NY, NZ,
                params.BC_y, params.BC_z, params.Lsd, params.RhoD, px,
                np.ascontiguousarray(TRs, dtype=np.float64),
                params.p0, params.p1, params.p2, params.p3, params.p4,
                params.p5, params.p6, params.p7, params.p8, params.p9,
                params.p10, params.p11, params.p12, params.p13, params.p14,
                params.Parallax,
                int(params.SubPixelLevel),
                float(params.SubPixelCardinalWidth),
                int(bool(params.SolidAngleCorrection)),
                int(bool(params.PolarizationCorrection)),
                params.PolarizationFraction,
                pol_plane_eta_rad,
                mask_arr, mask_present,
                flat_arr, flat_present,
                raw_y_arr.astype(np.float64),
                raw_z_arr.astype(np.float64),
                distortion_y, distortion_z, distortion_present,
                panels_arr, n_panels,
                corr_map, corr_n_y, corr_n_z,
                int(per_row_max_entries),
            )
        else:
            out_arr, per_row_count, bin_mask_flag = _numba_map_kernel(
                R_corners, Eta_corners, cornerYZ,
                Rt_center, Eta_center,
                r_lo, r_hi, eta_lo, eta_hi,
                n_r_bins, n_eta_bins,
                NY, NZ,
                params.Lsd, px,
                int(bool(params.SolidAngleCorrection)),
                int(bool(params.PolarizationCorrection)),
                params.PolarizationFraction,
                pol_plane_eta_rad,
                sa_factor,
                mask_arr, mask_present,
                flat_arr, flat_present,
                raw_y_arr.astype(np.float64),
                raw_z_arr.astype(np.float64),
                int(per_row_max_entries),
            )
        # Truncation silently drops map entries. Both `frac` and `areaWeight`
        # go together so a normalised profile can still look plausible, while
        # absolute flux, bin area, and whole bins are wrong. Never hide this
        # behind `verbose`.
        truncated_rows = int((per_row_count == per_row_max_entries).sum())
        if truncated_rows > 0:
            warnings.warn(
                f"map truncated: {truncated_rows} detector row(s) hit "
                f"per_row_max_entries={per_row_max_entries:,}, so map entries "
                f"were dropped. Normalised intensity may still look sane while "
                f"absolute flux and bin area are wrong, and whole bins can go "
                f"missing. Rebuild with a larger per_row_max_entries "
                f"(try {per_row_max_entries * 4:,}).",
                MapTruncationWarning, stacklevel=2,
            )

        # Flatten + bin-major regroup
        # Each entry is (bin_idx, raw_y, raw_z, frac, deltaR, area)
        if verbose:
            print(f"[mapper] flattening {int(per_row_count.sum()):,} entries…")
        all_entries = []
        for j in range(NZ):
            n = int(per_row_count[j])
            if n > 0:
                all_entries.append(out_arr[j, :n])
        if all_entries:
            flat = np.concatenate(all_entries, axis=0)
        else:
            flat = np.zeros((0, 6), dtype=np.float64)

        bin_idx = flat[:, 0].astype(np.int64)
        # Sort by bin to make pxList contiguous per bin
        order = np.argsort(bin_idx, kind="stable")
        flat = flat[order]
        bin_idx = bin_idx[order]

        n_bins = n_r_bins * n_eta_bins
        counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int32)
        offsets = np.zeros(n_bins, dtype=np.int32)
        if n_bins > 0:
            offsets[1:] = np.cumsum(counts[:-1], dtype=np.int32)

        pxList = np.empty(flat.shape[0], dtype=PXLIST_DTYPE)
        pxList["y"]          = flat[:, 1].astype(np.float32)
        pxList["z"]          = flat[:, 2].astype(np.float32)
        pxList["frac"]       = flat[:, 3].astype(np.float64)
        pxList["deltaR"]     = flat[:, 4].astype(np.float32)
        pxList["areaWeight"] = flat[:, 5].astype(np.float32)
        backend_label = "numba"
    else:
        # Pure-Python fallback
        bin_entries, bin_mask_flag = _build_map_python(
            params=params,
            R_corners=R_corners, Eta_corners=Eta_corners, cornerYZ=cornerYZ,
            Rt_center=Rt_center, Eta_center=Eta_center,
            raw_y_arr=raw_y_arr, raw_z_arr=raw_z_arr,
            r_lo=r_lo, r_hi=r_hi, eta_lo=eta_lo, eta_hi=eta_hi,
            n_r_bins=n_r_bins, n_eta_bins=n_eta_bins,
            NrPixelsY=NY, NrPixelsZ=NZ, Lsd=params.Lsd, px=px,
            solid_angle=bool(params.SolidAngleCorrection),
            polarization=bool(params.PolarizationCorrection),
            pol_fraction=params.PolarizationFraction,
            pol_plane_eta_rad=pol_plane_eta_rad,
            sa_factor=sa_factor,
            mask=mask, flat=flat,
        )
        n_bins = n_r_bins * n_eta_bins
        counts = np.array([len(b) for b in bin_entries], dtype=np.int32)
        offsets = np.zeros(n_bins, dtype=np.int32)
        if n_bins > 0:
            offsets[1:] = np.cumsum(counts[:-1], dtype=np.int32)
        total = int(counts.sum())
        pxList = np.empty(total, dtype=PXLIST_DTYPE)
        cur = 0
        for b in range(n_bins):
            for (ry, rz, frac, deltaR, area) in bin_entries[b]:
                pxList[cur] = (np.float32(ry), np.float32(rz),
                               np.float64(frac), np.float32(deltaR),
                               np.float32(area))
                cur += 1
        backend_label = "python"

    elapsed = time.perf_counter() - t0
    n_filled = int((counts > 0).sum())
    if verbose:
        print(f"[mapper] {int(counts.sum()):,} entries across "
              f"{n_filled:,} non-empty bins in {elapsed:.2f}s "
              f"(backend={backend_label})")
    return BuildMapResult(
        pxList=pxList,
        counts=counts,
        offsets=offsets,
        bin_mask_flag=bin_mask_flag,
        elapsed_s=elapsed,
        n_bins_filled=n_filled,
        backend=backend_label,
    )


def build_and_write_map(
    params: IntegrationParams,
    *,
    output_dir: str | Path,
    mask: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    panels: Optional[Sequence[Panel]] = None,
    distortion_y: Optional[np.ndarray] = None,
    distortion_z: Optional[np.ndarray] = None,
    residual_corr: Optional[ResidualCorrection] = None,
    auto_load: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
    use_numba: Optional[bool] = None,
    per_row_max_entries: Optional[int] = None,
) -> tuple[Path, Path]:
    """Build the map and write Map.bin + nMap.bin with a v3 header."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_map(params, mask=mask, flat=flat,
                       panels=panels,
                       distortion_y=distortion_y, distortion_z=distortion_z,
                       residual_corr=residual_corr,
                       auto_load=auto_load,
                       n_jobs=n_jobs,
                       verbose=verbose,
                       use_numba=use_numba,
                       per_row_max_entries=per_row_max_entries)

    header = MapHeader(
        param_hash=compute_param_hash(
            Lsd=params.Lsd, Ycen=params.BC_y, Zcen=params.BC_z,
            pxY=params.pxY, pxZ=params.pxZ,
            tx=params.tx, ty=params.ty, tz=params.tz,
            # All 15 distortion coefficients — fix for the v1 bug that
            # only hashed p0, p1, p2, p3, p4, p6 and silently used a
            # stale Map.bin when any other coefficient changed.
            p0=params.p0,   p1=params.p1,   p2=params.p2,
            p3=params.p3,   p4=params.p4,   p5=params.p5,
            p6=params.p6,   p7=params.p7,   p8=params.p8,
            p9=params.p9,   p10=params.p10, p11=params.p11,
            p12=params.p12, p13=params.p13, p14=params.p14,
            Parallax=params.Parallax,
            RhoD=params.RhoD,
            RBinSize=params.RBinSize, EtaBinSize=params.EtaBinSize,
            RMin=params.RMin, RMax=params.RMax,
            EtaMin=params.EtaMin, EtaMax=params.EtaMax,
            NrPixelsY=params.NrPixelsY, NrPixelsZ=params.NrPixelsZ,
            TransOpt=params.TransOpt,
            qMode=int(params.q_mode_active),
            Wavelength=params.Wavelength,
        ),
        q_mode=int(params.q_mode_active),
        gradient_mode=int(params.GradientCorrection),
        wavelength=params.Wavelength,
    )

    map_path = output_dir / "Map.bin"
    nmap_path = output_dir / "nMap.bin"
    write_map(map_path, nmap_path,
              pxList=result.pxList,
              counts=result.counts,
              offsets=result.offsets,
              header=header)
    return map_path, nmap_path
