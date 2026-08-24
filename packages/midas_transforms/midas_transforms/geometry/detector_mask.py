"""Push a detector pixel mask forward into the ideal-lab bitset the C forward
model already knows how to read.

Why this exists
---------------
``midas_ck_calc_diffraction_spots`` (``midas_ckernel/c_src/forward.c:181-190``)
already drops a *predicted* reflection whose detector cell fails a bit test::

    YCInt = (int)floor((big_det_size / 2) - (int)(-yl / pixelsize));
    ZCInt = (int)floor((int)(zl / pixelsize) + (big_det_size / 2));
    idx   = YCInt + big_det_size * ZCInt;
    if (!TestBit(bigdet->mask, idx)) KeepSpot = 0;

A spot dropped there leaves the numerator **and** the denominator of the
completeness ratio together, which is exactly the semantics wanted for a
reflection that landed on dead silicon: not "we failed to find it" but "it
could never have been found".

The hook has had no producer. Every writer of ``BigDetectorMask.bin`` lives in
``FF_HEDM/`` or ``gui/archive/`` (soft-deprecated), so ``BigDetSize`` is 0 in
every modern run and the test is dead in the refiner as well as the indexer.
Meanwhile the *real* pixel mask goes in through a different door entirely --
``MaskFile`` -> ``exchange/mask`` in the zarr (``midas_zipper/ff_zip.py:889``)
-> ``midas_peakfit`` -- and is used for exactly one thing: setting the
per-region ``maskTouched`` flag on *observed* peaks (``seeds.py:147-156``).

So the detector mask currently has no effect whatsoever on completeness. This
module closes that gap by writing the bitset from the mask that actually
exists.

Frames, and why this direction
------------------------------
The mask is in **raw analysis-frame pixels**, indexed ``mask[Z, Y]``. The
predicted spot is in **ideal, tilt- and distortion-free lab micrometres**.
Going from a predicted spot back to a pixel would need the distortion
polynomial inverted (a Newton solve, per-spot, inside the hot loop).

We push the *set* forward instead: every masked pixel goes through
:func:`midas_transforms.fit_setup.transform.apply_tilt_distortion` -- the same
function that places observed spots -- and is rasterised into the ideal-lab
grid. No inversion, evaluated once per run rather than once per candidate
orientation, and it lands in the frame the predicted spots already live in
because both end with ``Y = -R sin(eta), Z = R cos(eta)``
(``transform.py:96-97`` and ``forward.c:24-25``).

Known limitation, quantified: ``Spots.bin`` col 0 is the **wedge-corrected**
lab Y, while this push-forward produces the pre-wedge value. The wedge
correction is omega-dependent, so a detector-fixed mask has no single exact
image under it. At ``Wedge == 0`` the two frames are identical by construction
(``correct_wedge_no_op``); across the parameter files sampled on 2026-08-22 the
largest non-zero ``Wedge`` was -0.0126 deg, which displaces the outermost ring
by ``R sin(wedge)`` ~ 42 um ~ 0.28 px at a 150 um pitch. Sub-pixel, and smaller
than the one-cell dilation applied below. It is not correct for a large wedge,
and :func:`build_active_area_bitset` warns past ``wedge_warn_deg``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "BIGDET_WORD_BITS",
    "bigdet_cell_index",
    "build_active_area_bitset",
    "build_active_area_bitset_from_zarr",
    "pack_bitset",
    "write_big_detector_mask",
]

#: ``TestBit(A, k) = A[k / 32] & (1 << (k % 32))`` -- 32-bit words, LSB first.
BIGDET_WORD_BITS = 32


def bigdet_cell_index(
    y_lab_um: np.ndarray, z_lab_um: np.ndarray, big_det_size: int, px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``(YCInt, ZCInt)`` for ideal-lab micrometre coordinates.

    A literal transcription of ``forward.c:183-187``. Two C details are load
    bearing and are reproduced rather than tidied:

    * ``big_det_size / 2`` is **integer** division;
    * ``(int)(-yl / pixelsize)`` truncates toward zero, which is *not* a floor
      for negative values. ``ndarray.astype(np.int64)`` truncates toward zero
      too, so the cast matches.

    The outer ``floor`` in the C is applied to an already-integral expression
    and is therefore a no-op; it is not reproduced.
    """
    half = int(big_det_size) // 2
    yc = half - (-np.asarray(y_lab_um, dtype=np.float64) / px).astype(np.int64)
    zc = (np.asarray(z_lab_um, dtype=np.float64) / px).astype(np.int64) + half
    return yc, zc


def _dilate(grid: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation with a 3x3 structuring element, via shifts.

    Local rather than ``scipy.ndimage`` so this module keeps ``midas-transforms``
    free of a SciPy dependency it does not otherwise carry.
    """
    out = grid
    for _ in range(int(iterations)):
        acc = out.copy()
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dz == 0 and dy == 0:
                    continue
                acc |= np.roll(np.roll(out, dz, axis=0), dy, axis=1)
        out = acc
    return out


def pack_bitset(grid: np.ndarray) -> np.ndarray:
    """``(S, S)`` boolean grid -> the uint32 bitset ``TestBit`` indexes.

    ``grid[z, y]`` is bit ``y + S*z``, matching ``idx = YCInt + big_det_size *
    ZCInt``: C row-major ravel of a ``[ZCInt][YCInt]`` array gives exactly that
    linear index.

    The word count matches the C allocation in ``FitUnified.c:1470-1475``
    (``S*S/32 + 1``), including its deliberate ``+1`` slack word.
    """
    flat = np.ascontiguousarray(grid, dtype=bool).ravel()
    n_words = flat.size // BIGDET_WORD_BITS + 1
    padded = np.zeros(n_words * BIGDET_WORD_BITS, dtype=bool)
    padded[: flat.size] = flat
    weights = (np.uint32(1) << np.arange(BIGDET_WORD_BITS, dtype=np.uint32))
    words = padded.reshape(n_words, BIGDET_WORD_BITS)
    return np.bitwise_or.reduce(
        np.where(words, weights, np.uint32(0)), axis=1
    ).astype(np.uint32)


def build_active_area_bitset(
    mask_bad: np.ndarray,
    *,
    Lsd: float,
    BC_y: float,
    BC_z: float,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    p_coeffs: Optional[np.ndarray] = None,
    px: float,
    rho_d: float,
    parallax: float = 0.0,
    residual_corr_map: Optional[np.ndarray] = None,
    wedge_deg: float = 0.0,
    big_det_size: Optional[int] = None,
    off_detector: str = "drop",
    dilate_masked: int = 1,
    chunk_rows: int = 128,
    wedge_warn_deg: float = 0.05,
) -> Tuple[np.ndarray, int, dict]:
    """Build the ideal-lab active-area grid from a raw detector mask.

    Parameters
    ----------
    mask_bad
        ``(NrPixelsZ, NrPixelsY)``; **non-zero means BAD**, matching
        ``exchange/mask`` and ``midas_peakfit.seeds`` (``mvals == 1`` -> masked).
        Note the polarity is the opposite of the bitset produced here, where a
        **set bit means KEEP** -- ``forward.c`` drops the spot when the bit is
        clear.
    off_detector
        ``"drop"`` (default) -- a bit is set only where a real, unmasked pixel
        maps, so a reflection predicted off the detector *or* on a masked pixel
        leaves the completeness ratio. This is the honest "could it have been
        observed?" question.
        ``"keep"`` -- everything outside the masked cells is set, isolating the
        mask's effect from the detector-extent effect. Useful to attribute a
        change to one cause or the other; both counts are reported in ``stats``
        either way.
    dilate_masked
        Grow the masked set by this many cells before subtracting it. Forward
        mapping pixel centres can leave one-cell gaps wherever tilt or
        distortion stretches the grid; dilating the *masked* set errs toward
        excluding a spot, which is the conservative direction. 0 disables.

    Returns
    -------
    (bitset, big_det_size, stats)
        ``bitset`` is uint32, ready for :func:`write_big_detector_mask`.
    """
    import torch

    if off_detector not in {"drop", "keep"}:
        raise ValueError(
            f"off_detector must be 'drop' or 'keep'; got {off_detector!r}"
        )
    if abs(float(wedge_deg)) > float(wedge_warn_deg):
        warnings.warn(
            f"Wedge = {wedge_deg:g} deg exceeds {wedge_warn_deg:g} deg. This "
            "mask is pushed forward in the PRE-wedge lab frame, while "
            "Spots.bin carries wedge-corrected coordinates. The wedge "
            "correction is omega-dependent, so a detector-fixed mask has no "
            "single exact image under it; at this magnitude the mismatch may "
            "exceed the one-cell dilation. See the module docstring.",
            RuntimeWarning,
            stacklevel=2,
        )

    mask_bad = np.asarray(mask_bad)
    if mask_bad.ndim != 2:
        raise ValueError(f"mask must be 2-D (Z, Y); got shape {mask_bad.shape}")
    n_z, n_y = mask_bad.shape
    bad = mask_bad != 0

    if p_coeffs is None:
        p_coeffs = np.zeros(15, dtype=np.float64)
    p_coeffs = np.asarray(p_coeffs, dtype=np.float64).ravel()
    if p_coeffs.size != 15:
        raise ValueError(f"p_coeffs must have 15 entries; got {p_coeffs.size}")

    dt = torch.float64
    geom = dict(
        Lsd=torch.tensor(float(Lsd), dtype=dt),
        BC_y=torch.tensor(float(BC_y), dtype=dt),
        BC_z=torch.tensor(float(BC_z), dtype=dt),
        tx=torch.tensor(float(tx), dtype=dt),
        ty=torch.tensor(float(ty), dtype=dt),
        tz=torch.tensor(float(tz), dtype=dt),
        p_coeffs=torch.tensor(p_coeffs, dtype=dt),
        px=torch.tensor(float(px), dtype=dt),
        rho_d=torch.tensor(float(rho_d), dtype=dt),
        parallax=torch.tensor(float(parallax), dtype=dt),
        residual_corr_map=(
            None if residual_corr_map is None
            else torch.tensor(np.asarray(residual_corr_map), dtype=dt)
        ),
    )

    from ..fit_setup.transform import apply_tilt_distortion

    def _to_lab(y_pix: np.ndarray, z_pix: np.ndarray):
        y_lab, z_lab = apply_tilt_distortion(
            torch.tensor(y_pix, dtype=dt), torch.tensor(z_pix, dtype=dt), **geom
        )
        return y_lab.numpy(), z_lab.numpy()

    # --- size the grid from the detector perimeter -------------------------
    # The pixel -> lab map is smooth and radially monotone, so the extreme
    # |Y|, |Z| are attained on the border. 11k points instead of n_z*n_y, and
    # the main pass asserts nothing exceeded the bound anyway.
    if big_det_size is None:
        yy = np.arange(n_y, dtype=np.float64)
        zz = np.arange(n_z, dtype=np.float64)
        edge_y = np.concatenate([yy, yy, np.zeros(n_z), np.full(n_z, n_y - 1.0)])
        edge_z = np.concatenate([np.zeros(n_y), np.full(n_y, n_z - 1.0), zz, zz])
        ey, ez = _to_lab(edge_y, edge_z)
        reach_px = max(np.abs(ey).max(), np.abs(ez).max()) / float(px)
        # +2 cells of slack on each side, then force even so that S//2 is an
        # exact centre (the C uses integer division on it).
        big_det_size = 2 * (int(math.ceil(reach_px)) + 2)
    big_det_size = int(big_det_size)
    if big_det_size % 2:
        big_det_size += 1
    s = big_det_size

    active = np.zeros((s, s), dtype=bool)
    masked = np.zeros((s, s), dtype=bool)

    # --- rasterise by pixel CORNERS, not centres ---------------------------
    # A pixel centre maps, in the no-tilt/no-distortion limit, exactly onto a
    # cell boundary -- and `apply_tilt_distortion` reaches it through a polar
    # round trip (sqrt -> atan2 -> sin/cos), so the result lands at the
    # integer plus or minus ~1e-13. Truncation toward zero then throws the
    # value a WHOLE CELL when it lands a hair low. Measured on a 256x256
    # identity geometry: 9388 masked pixels collapsed onto 7652 cells, and a
    # 30 deg wedge that should remove 0.1667 of a ring removed 0.1325.
    #
    # Mapping the four corners and filling the cell rectangle they span
    # removes the knife edge entirely: a pixel sitting on a boundary has
    # corners on both sides and claims both cells. Over-covering is the safe
    # direction for both sets -- it grows the active footprint (permissive,
    # and intersected with the mask anyway) and grows the masked set
    # (conservative: drops a borderline spot rather than trusting it).
    corners = ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
    n_oob = 0
    max_span = 0
    for z0 in range(0, n_z, chunk_rows):
        z1 = min(z0 + chunk_rows, n_z)
        zg, yg = np.meshgrid(
            np.arange(z0, z1, dtype=np.float64),
            np.arange(n_y, dtype=np.float64),
            indexing="ij",
        )
        yg, zg = yg.ravel(), zg.ravel()
        yc_c = np.empty((4, yg.size), dtype=np.int64)
        zc_c = np.empty((4, yg.size), dtype=np.int64)
        for i, (dy, dz) in enumerate(corners):
            y_lab, z_lab = _to_lab(yg + dy, zg + dz)
            yc_c[i], zc_c[i] = bigdet_cell_index(y_lab, z_lab, s, float(px))

        y_lo, y_hi = yc_c.min(axis=0), yc_c.max(axis=0)
        z_lo, z_hi = zc_c.min(axis=0), zc_c.max(axis=0)
        max_span = max(max_span, int((y_hi - y_lo).max()), int((z_hi - z_lo).max()))
        if max_span > 3:
            raise RuntimeError(
                f"a single detector pixel spans {max_span + 1} ideal-lab cells; "
                "the distortion is stretching the grid far more than expected. "
                "Check px / rho_d / p_coeffs before trusting this mask."
            )

        bad_chunk = bad[z0:z1, :].ravel()
        n_oob += int((
            (y_lo < 0) | (y_hi >= s) | (z_lo < 0) | (z_hi >= s)
        ).sum())
        # Fill each pixel's cell rectangle. The span is 0..3 cells per axis,
        # so this is a handful of fully vectorised passes, not a Python loop
        # over pixels.
        for dy in range(max_span + 1):
            for dz in range(max_span + 1):
                yy = y_lo + dy
                zz = z_lo + dz
                ok = (yy <= y_hi) & (zz <= z_hi) & \
                     (yy >= 0) & (yy < s) & (zz >= 0) & (zz < s)
                sel_a = ok & ~bad_chunk
                sel_m = ok & bad_chunk
                active[zz[sel_a], yy[sel_a]] = True
                masked[zz[sel_m], yy[sel_m]] = True

    if n_oob:
        raise RuntimeError(
            f"{n_oob} detector pixels mapped outside the {s}x{s} ideal-lab "
            "grid. The perimeter-based sizing under-estimated the reach, "
            "which means the pixel->lab map is not radially monotone for this "
            "geometry. Pass big_det_size explicitly."
        )

    masked_dil = _dilate(masked, dilate_masked) if dilate_masked else masked

    if off_detector == "drop":
        keep = active & ~masked_dil
    else:
        keep = ~masked_dil

    stats = {
        "big_det_size": s,
        "n_cells": s * s,
        "n_detector_pixels": int(n_z * n_y),
        "n_bad_pixels": int(bad.sum()),
        "n_cells_active": int(active.sum()),
        "n_cells_masked": int(masked.sum()),
        "n_cells_masked_after_dilation": int(masked_dil.sum()),
        "n_cells_keep": int(keep.sum()),
        # Both semantics reported regardless of which was chosen, so a change
        # in completeness can be attributed to the mask or to detector extent.
        "n_cells_keep_if_drop": int((active & ~masked_dil).sum()),
        "n_cells_keep_if_keep": int((~masked_dil).sum()),
        "off_detector": off_detector,
        "dilate_masked": int(dilate_masked),
        "wedge_deg": float(wedge_deg),
        # Cells spanned by the widest single pixel, minus one. 0 means the map
        # is effectively a bijection; a large value means distortion is
        # stretching the grid and the mask is being over-covered accordingly.
        "max_pixel_cell_span": int(max_span),
    }
    return pack_bitset(keep), s, stats


def build_active_area_bitset_from_zarr(zarr_path, **kwargs):
    """Convenience: pull geometry and the mask straight out of a MIDAS zarr.

    Resolves geometry exactly as :mod:`midas_transforms.fit_setup.core` does --
    ``RhoD`` with a ``MaxRingRad`` fallback, distortion coefficients shimmed
    from the v2 basis back to the legacy 15-slot v1 order, ``PixelSize`` as the
    pitch -- because the whole point is for the mask to land in the same frame
    as the observed spots. Diverging here would be silent.

    The mask array is taken through ``midas_peakfit.preprocess.prepare_mask``
    rather than square-padded locally, so it is byte-identical to the array
    ``midas_peakfit.seeds`` used to set ``maskTouched``. That import is soft:
    ``midas-transforms`` does not depend on ``midas-peakfit`` (nor the reverse),
    and duplicating the padding would give two implementations of one
    convention -- exactly the axis question that silently mirrors a mask.

    Returns ``(bitset, big_det_size, stats)``.
    """
    import zarr

    from ..params import read_zarr_params

    zp = read_zarr_params(zarr_path)

    root = zarr.open(str(zarr_path), mode="r")
    if "exchange/mask" not in root:
        raise FileNotFoundError(
            f"{zarr_path} has no 'exchange/mask'. The mask is baked in at zip "
            "time from the MaskFN/MaskFile key (midas_zipper/ff_zip.py:889); "
            "without it there is nothing to push forward."
        )
    raw_mask = np.asarray(root["exchange/mask"][0])

    try:
        from midas_peakfit.preprocess import prepare_mask
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "midas-peakfit is needed to square-pad the mask identically to the "
            "peak search (preprocess.prepare_mask). Install it, or call "
            "build_active_area_bitset directly with an already-padded "
            "(Z, Y) array."
        ) from e

    mask_bad = prepare_mask(
        raw_mask, zp.NrPixels, zp.NrPixelsY, zp.NrPixelsZ, zp.TransOpt
    )

    from midas_distortion import v1_to_v2_coeffs, v2_to_v1_coeffs

    coeffs_v2 = zp.dist_coeffs_v2
    if coeffs_v2 is None:
        coeffs_v2 = v1_to_v2_coeffs(
            np.array([getattr(zp, f"p{i}") for i in range(15)], dtype=np.float64)
        )
    p_v1 = np.asarray(v2_to_v1_coeffs(np.asarray(coeffs_v2, dtype=np.float64)))

    defaults = dict(
        Lsd=zp.Lsd, BC_y=zp.YCen, BC_z=zp.ZCen,
        tx=zp.tx, ty=zp.ty, tz=zp.tz,
        p_coeffs=p_v1, px=zp.PixelSize,
        rho_d=(zp.RhoD if zp.RhoD > 0 else zp.MaxRingRad),
        wedge_deg=zp.Wedge,
    )
    defaults.update(kwargs)
    return build_active_area_bitset(mask_bad, **defaults)


def write_big_detector_mask(
    path: Union[str, Path], bitset: np.ndarray
) -> Path:
    """Write the uint32 bitset where ``ReadBigDet`` mmaps it.

    ``FitUnified.c:1041`` builds the name as ``<cwd>/BigDetectorMask.bin`` and
    mmaps the whole file read-only, so the on-disk layout is simply the raw
    little-endian uint32 words.
    """
    path = Path(path)
    arr = np.ascontiguousarray(bitset, dtype="<u4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.tobytes())
    return path
