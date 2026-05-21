"""PyTorch sparse-CSR integration kernels — full parity with the C/CUDA codes.

Three integration modes selectable per call. All three are reduced to a single
SpMV against a precomputed sparse matrix:

- ``mode='floor'``     — integer-floor pixel lookup using the truncated
                         ``y, z`` stored in pxList. Matches
                         ``integrate_noMapMask`` in
                         ``IntegratorFitPeaksGPUStream.cu``.
                         Uses ``geom.csr_floor``: 1 nonzero per pxList entry.

- ``mode='bilinear'``  — bilinear interpolation at the float ``(y, z)``
                         sub-pixel coordinates stored in pxList. Matches
                         the per-pixel loop in ``IntegratorZarrOMP.c``
                         lines 1733–1744. Uses ``geom.csr_bilinear``: each
                         pxList entry is expanded into 4 (bin, pixel, weight)
                         triples (one per bilinear corner), with duplicate
                         (bin, pixel) pairs coalesced via summation.

- ``mode='gradient'``  — bilinear at the gradient-corrected coordinates
                         ``(y - deltaR·dy/R, z - deltaR·dz/R)``. Matches
                         the ``GradientCorrection=1`` branch of the GPU
                         stream kernel. Uses ``geom.csr_gradient``: same
                         4-corner expansion as bilinear but with the
                         gradient-resampled coordinates.

The precomputed bilinear/gradient CSRs cost ~3× the memory of the floor CSR
(after deduplication; the raw 4× expansion coalesces well in practice) but
deliver SpMV-equivalent throughput in all three modes — no scatter_add, no
manual gather, no per-iteration coordinate math.

All matrices live on the same device as ``geom.csr_floor`` so swapping
backends is a single ``device='cuda'`` change to ``build_csr``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
import torch

from midas_integrate.bin_io import PixelMap

AREA_THRESHOLD = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CSR geometry container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CSRGeometry:
    """Precomputed sparse geometry for a fixed detector + bin layout.

    Three precomputed sparse matrices, one per integration mode. All three
    have shape ``(n_bins, n_pixels_y * n_pixels_z)``.

    Attributes:
        csr_floor: SpMV for ``integrate(..., mode='floor')``. Each pxList
            entry contributes 1 nonzero whose value is its ``frac``.
        csr_bilinear: SpMV for ``integrate(..., mode='bilinear')``. Each
            pxList entry is expanded into 4 nonzeros, one per bilinear
            corner pixel, with values ``(1-fy)(1-fz)·frac``, ``fy(1-fz)·frac``,
            ``(1-fy)·fz·frac``, ``fy·fz·frac``. Duplicate (bin, pixel) pairs
            are summed at construction so the resulting CSR is canonical.
        csr_gradient: same as ``csr_bilinear`` but evaluated at the
            gradient-corrected coordinates.

        csr_floor_sq, csr_bilinear_sq, csr_gradient_sq: per-mode CSRs whose
            values are the element-wise squares of the corresponding
            integration matrices. Used by ``integrate_with_variance`` to
            propagate per-pixel variance under the assumption of
            uncorrelated pixels: ``Var(I_b) = Σ_l W_bl² · Var(I_l)``.
            ``None`` when ``compute_variance=False`` was passed to
            ``build_csr`` (default).

        area_per_bin: dense, shape (n_bins,) — Σ areaWeight per bin, used
            as the normalizer when ``normalize=True``.

        n_r, n_eta, n_pixels_y, n_pixels_z: the geometry shape. Flat pixel
            index is ``z * n_pixels_y + y``.
        bc_y, bc_z: beam-center pixels (recorded for reference).
    """
    csr_floor: torch.Tensor
    csr_bilinear: torch.Tensor
    csr_gradient: torch.Tensor
    area_per_bin: torch.Tensor
    n_r: int
    n_eta: int
    n_pixels_y: int
    n_pixels_z: int
    bc_y: float = 0.0
    bc_z: float = 0.0
    csr_floor_sq: Optional[torch.Tensor] = None
    csr_bilinear_sq: Optional[torch.Tensor] = None
    csr_gradient_sq: Optional[torch.Tensor] = None

    @property
    def n_bins(self) -> int:
        return self.n_r * self.n_eta

    @property
    def device(self) -> torch.device:
        return self.csr_floor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.csr_floor.dtype

    # Back-compat alias: the previous CSRGeometry exposed ``csr_frac`` as the
    # floor matrix. Keep the name working so older code doesn't break.
    @property
    def csr_frac(self) -> torch.Tensor:
        return self.csr_floor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _scipy_csr_to_torch(
    sp: scipy.sparse.csr_matrix,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a torch.sparse_csr_tensor from a scipy CSR (already coalesced)."""
    sp = sp.tocsr()
    sp.sort_indices()                       # required by torch SpMV
    indptr = torch.from_numpy(sp.indptr.astype(np.int64))
    indices = torch.from_numpy(sp.indices.astype(np.int64))
    values = torch.from_numpy(np.ascontiguousarray(sp.data)).to(dtype)
    out = torch.sparse_csr_tensor(indptr, indices, values, size=sp.shape)
    return out.to(device)


def _build_floor_csr(
    *,
    bin_index: np.ndarray,        # (n_entries,) int64
    pixel_y: np.ndarray,          # (n_entries,) float
    pixel_z: np.ndarray,          # (n_entries,) float
    frac: np.ndarray,             # (n_entries,) float64
    n_bins: int,
    n_pixels_y: int, n_pixels_z: int,
    device, dtype,
) -> torch.Tensor:
    """Floor-mode CSR — one nonzero per pxList entry."""
    iy = pixel_y.astype(np.int64)
    iz = pixel_z.astype(np.int64)
    in_bounds = ((iy >= 0) & (iy < n_pixels_y)
                 & (iz >= 0) & (iz < n_pixels_z))
    col = iz * n_pixels_y + iy
    if not in_bounds.all():
        col = np.where(in_bounds, col, 0)
        vals = np.where(in_bounds, frac, 0.0)
    else:
        vals = frac
    sp = scipy.sparse.csr_matrix(
        (vals, (bin_index, col)),
        shape=(n_bins, n_pixels_y * n_pixels_z),
    )
    sp.sum_duplicates()
    return _scipy_csr_to_torch(sp, device=device, dtype=dtype)


def _build_bilinear_csr(
    *,
    bin_index: np.ndarray,        # (n_entries,) int64
    sample_y: np.ndarray,         # (n_entries,) float64 — possibly resampled
    sample_z: np.ndarray,         # (n_entries,) float64
    frac: np.ndarray,             # (n_entries,) float64
    n_bins: int,
    n_pixels_y: int, n_pixels_z: int,
    device, dtype,
) -> torch.Tensor:
    """4-CSR bilinear: each entry → 4 (bin, pixel, weight·frac) triples.

    Mirrors the bilinear sampler of ``IntegratorZarrOMP.c`` lines 1733–1744:
    the same float→int floor + boundary clamp, applied once at CSR build time.
    """
    n_e = bin_index.shape[0]

    # Floor + fractional part
    iy = np.floor(sample_y).astype(np.int64)
    iz = np.floor(sample_z).astype(np.int64)
    fy = sample_y - iy.astype(np.float64)
    fz = sample_z - iz.astype(np.float64)

    # Boundary clamp — exact match to the C version
    over_y = iy >= (n_pixels_y - 1)
    under_y = iy < 0
    iy = np.where(over_y, n_pixels_y - 2, iy)
    fy = np.where(over_y, 1.0, fy)
    iy = np.where(under_y, 0, iy)
    fy = np.where(under_y, 0.0, fy)

    over_z = iz >= (n_pixels_z - 1)
    under_z = iz < 0
    iz = np.where(over_z, n_pixels_z - 2, iz)
    fz = np.where(over_z, 1.0, fz)
    iz = np.where(under_z, 0, iz)
    fz = np.where(under_z, 0.0, fz)

    base = iz * n_pixels_y + iy
    ny = n_pixels_y
    shape = (n_bins, n_pixels_y * n_pixels_z)

    # Memory-bounded assembly for large detectors. The earlier version built one
    # COO of 4·n_e triples (concatenating rows, cols, vals) — for a 2880² panel
    # that is three ~130 M-element arrays plus scipy's COO→CSR copies, several GB
    # of transient peak with a *single ~1 GB contiguous* allocation for `vals`
    # that OOMs on modest machines (this is the exact line that failed in the
    # field). Instead we build the four bilinear corners as separate CSRs and
    # sum them: the largest single allocation drops to ~n_e (one corner's
    # weights, ~4× smaller), scipy coalesces each corner on construction, and the
    # additions coalesce across corners. Indices are int32 (every flat pixel/bin
    # index < 2³¹ here), halving the index footprint. Values stay float64; the
    # only difference vs the single concatenation is summation order (≲1e-12, far
    # inside the 1e-9 bilinear tolerance and below the float32 cast downstream).
    rows32 = bin_index.astype(np.int32)
    del over_y, under_y, over_z, under_z, iy, iz   # free clamp temporaries

    def _corner(col_idx: np.ndarray, w: np.ndarray) -> "scipy.sparse.csr_matrix":
        return scipy.sparse.csr_matrix(
            (w, (rows32, col_idx.astype(np.int32))), shape=shape,
        )

    one_fy = 1.0 - fy
    one_fz = 1.0 - fz
    sp = _corner(base,            one_fy * one_fz * frac)
    sp = sp + _corner(base + 1,     fy * one_fz * frac)
    sp = sp + _corner(base + ny,    one_fy * fz * frac)
    sp = sp + _corner(base + ny + 1, fy * fz * frac)
    sp.sum_duplicates()
    sp.eliminate_zeros()
    return _scipy_csr_to_torch(sp, device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Public: build_csr
# ─────────────────────────────────────────────────────────────────────────────
def build_csr(
    pixmap: PixelMap,
    *,
    n_r: int, n_eta: int,
    n_pixels_y: int, n_pixels_z: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    bc_y: float = 0.0, bc_z: float = 0.0,
    build_modes: tuple[str, ...] = ("floor", "bilinear", "gradient"),
    compute_variance: bool = False,
) -> CSRGeometry:
    """Pack a PixelMap into precomputed CSR matrices for fast integration.

    Args:
        pixmap: loaded Map.bin / nMap.bin.
        n_r, n_eta: bin counts (must equal pixmap.n_bins jointly).
        n_pixels_y, n_pixels_z: detector shape; flat index = z*NY + y.
        device, dtype: forwarded to the produced sparse tensors.
        bc_y, bc_z: beam-center pixels (only used by ``mode='gradient'``).
        build_modes: which CSR matrices to precompute. Default builds all
            three. Pass e.g. ``('floor',)`` to skip the bilinear ones if you
            know you won't use them — saves ~3× memory for the CSR.
        compute_variance: if True, also build a squared-weight CSR for each
            requested mode. The squared CSRs share the sparsity pattern of
            the originals; values are element-wise squares of the canonical
            (sum-coalesced) per-(bin, pixel) weights. Doubles the precomputed
            CSR memory but makes ``integrate_with_variance`` available.
    """
    expected_bins = n_r * n_eta
    if pixmap.n_bins != expected_bins:
        raise ValueError(
            f"PixelMap has {pixmap.n_bins} bins; n_r * n_eta = {expected_bins}"
        )
    n_pixels = n_pixels_y * n_pixels_z

    counts = pixmap.counts.astype(np.int64)
    offsets = pixmap.offsets.astype(np.int64)
    if (counts < 0).any():
        raise ValueError("nMap.bin contains negative pixel counts")
    end_idx = offsets + counts
    bad = (end_idx > pixmap.n_entries) & (counts > 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} bin(s) reference past end of pxList "
            f"(n_entries={pixmap.n_entries})"
        )

    total = int(counts.sum())

    # Bin-major gather (matches PrecomputeOffsets_kernel layout)
    gather = np.empty(total, dtype=np.int64)
    cur = 0
    for b in range(pixmap.n_bins):
        c = int(counts[b])
        if c == 0:
            continue
        s = int(offsets[b])
        gather[cur:cur + c] = np.arange(s, s + c, dtype=np.int64)
        cur += c
    assert cur == total

    rec = pixmap.pxList[gather]
    bin_index = np.repeat(np.arange(pixmap.n_bins, dtype=np.int64), counts)
    pixel_y = rec["y"].astype(np.float64)
    pixel_z = rec["z"].astype(np.float64)
    frac    = rec["frac"].astype(np.float64)
    area    = rec["areaWeight"].astype(np.float64)
    delta_r = rec["deltaR"].astype(np.float64)

    # area_per_bin: matches the C initialize_PerFrameArr_Area_kernel which IS
    # bounds-checked, so we sum only in-bounds entries here.
    iy_int = pixel_y.astype(np.int64)
    iz_int = pixel_z.astype(np.int64)
    in_bounds = ((iy_int >= 0) & (iy_int < n_pixels_y)
                 & (iz_int >= 0) & (iz_int < n_pixels_z))
    area_per_bin = np.zeros(pixmap.n_bins, dtype=np.float64)
    np.add.at(area_per_bin, bin_index, np.where(in_bounds, area, 0.0))

    # ── Build the three CSRs ────────────────────────────────────────────
    csr_floor = _build_floor_csr(
        bin_index=bin_index, pixel_y=pixel_y, pixel_z=pixel_z, frac=frac,
        n_bins=pixmap.n_bins, n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
        device=device, dtype=dtype,
    ) if "floor" in build_modes else _empty_csr(pixmap.n_bins, n_pixels, device, dtype)

    if "bilinear" in build_modes:
        csr_bilinear = _build_bilinear_csr(
            bin_index=bin_index,
            sample_y=pixel_y, sample_z=pixel_z, frac=frac,
            n_bins=pixmap.n_bins,
            n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
            device=device, dtype=dtype,
        )
    else:
        csr_bilinear = _empty_csr(pixmap.n_bins, n_pixels, device, dtype)

    if "gradient" in build_modes:
        # Gradient resampling: shift each entry's sample point toward the
        # R-bin centroid along the radial direction. Mirror of the C
        # `gradientCorrection` branch in integrate_noMapMask.
        dy = pixel_y - bc_y
        dz = pixel_z - bc_z
        R = np.sqrt(dy * dy + dz * dz)
        scale = np.where(R > 1.0, delta_r / R, 0.0)
        sample_y_g = pixel_y - scale * dy
        sample_z_g = pixel_z - scale * dz
        csr_gradient = _build_bilinear_csr(
            bin_index=bin_index,
            sample_y=sample_y_g, sample_z=sample_z_g, frac=frac,
            n_bins=pixmap.n_bins,
            n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
            device=device, dtype=dtype,
        )
    else:
        csr_gradient = _empty_csr(pixmap.n_bins, n_pixels, device, dtype)

    area_per_bin_t = torch.from_numpy(area_per_bin).to(device=device, dtype=dtype)

    csr_floor_sq = _square_csr(csr_floor) if compute_variance and "floor" in build_modes else None
    csr_bilinear_sq = _square_csr(csr_bilinear) if compute_variance and "bilinear" in build_modes else None
    csr_gradient_sq = _square_csr(csr_gradient) if compute_variance and "gradient" in build_modes else None

    return CSRGeometry(
        csr_floor=csr_floor,
        csr_bilinear=csr_bilinear,
        csr_gradient=csr_gradient,
        area_per_bin=area_per_bin_t,
        n_r=n_r, n_eta=n_eta,
        n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
        bc_y=bc_y, bc_z=bc_z,
        csr_floor_sq=csr_floor_sq,
        csr_bilinear_sq=csr_bilinear_sq,
        csr_gradient_sq=csr_gradient_sq,
    )


def _square_csr(sp: torch.Tensor) -> torch.Tensor:
    """Return a torch CSR with the same indptr/indices but values squared.

    The input CSR's values are the canonical per-(bin, pixel) weights after
    sum_duplicates() coalesced any expansion (e.g. bilinear's 4-corner fan-out).
    For uncorrelated pixels, ``Var(Σ_l W_bl·I_l) = Σ_l W_bl²·Var(I_l)``,
    so the variance SpMV uses the squared values against the per-pixel
    variance vector.
    """
    if sp._nnz() == 0:
        return torch.sparse_csr_tensor(
            sp.crow_indices().clone(),
            sp.col_indices().clone(),
            sp.values().clone(),
            size=sp.shape,
        )
    vals_sq = sp.values() * sp.values()
    return torch.sparse_csr_tensor(
        sp.crow_indices().clone(),
        sp.col_indices().clone(),
        vals_sq,
        size=sp.shape,
    )


def _empty_csr(n_bins: int, n_pixels: int,
               device, dtype) -> torch.Tensor:
    """Empty placeholder CSR for modes the user opted out of."""
    indptr = torch.zeros(n_bins + 1, dtype=torch.int64)
    indices = torch.zeros(0, dtype=torch.int64)
    values = torch.zeros(0, dtype=dtype)
    return torch.sparse_csr_tensor(indptr, indices, values,
                                   size=(n_bins, n_pixels)).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Integration — all three modes are now a single SpMV
# ─────────────────────────────────────────────────────────────────────────────
def integrate(
    image: torch.Tensor,
    geom: CSRGeometry,
    *,
    mode: str = "floor",
    normalize: bool = True,
) -> torch.Tensor:
    """Integrate one detector image into the (R, Eta) bin grid.

    Args:
        image:    shape (n_pixels_z, n_pixels_y) or flat (n_pixels,)
        geom:     output of ``build_csr``
        mode:     'floor' | 'bilinear' | 'gradient' — see module docstring
        normalize: if True, divide each bin sum by the bin's area sum
                   (matches the C ``Normalize=1`` flag).

    Returns:
        2D tensor of shape (n_r, n_eta).
    """
    if image.ndim == 2:
        image_flat = image.reshape(-1)
    else:
        image_flat = image
    expected = geom.n_pixels_y * geom.n_pixels_z
    if image_flat.numel() != expected:
        raise ValueError(
            f"image numel {image_flat.numel()} != n_pixels {expected}"
        )
    image_flat = image_flat.to(dtype=geom.dtype, device=geom.device)

    if mode == "floor":
        sp = geom.csr_floor
    elif mode == "bilinear":
        sp = geom.csr_bilinear
    elif mode == "gradient":
        sp = geom.csr_gradient
    else:
        raise ValueError(f"unknown integration mode {mode!r}; "
                         "expected 'floor' | 'bilinear' | 'gradient'")
    if sp._nnz() == 0 and mode != "floor":
        raise RuntimeError(
            f"build_csr was called without {mode!r} in build_modes; "
            "rebuild with build_modes including this mode."
        )

    # Single SpMV — same path for all three modes.
    raw = torch.matmul(sp, image_flat.unsqueeze(1)).squeeze(1)

    if normalize:
        valid = geom.area_per_bin > AREA_THRESHOLD
        out = torch.where(
            valid,
            raw / torch.clamp(geom.area_per_bin, min=AREA_THRESHOLD),
            torch.zeros_like(raw),
        )
    else:
        out = raw
    return out.reshape(geom.n_r, geom.n_eta)


def integrate_with_variance(
    image: torch.Tensor,
    geom: CSRGeometry,
    *,
    mode: str = "floor",
    normalize: bool = True,
    variance_image: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate one frame and propagate per-pixel variance.

    Returns ``(intensity, variance)``. Both are 2D tensors of shape
    ``(n_r, n_eta)``.

    Args:
        image: detector image, shape (n_pixels_z, n_pixels_y) or flat.
        geom: output of ``build_csr(..., compute_variance=True)``.
        mode: 'floor' | 'bilinear' | 'gradient' — see ``integrate``.
        normalize: divide intensity by Σ areaWeight per bin (and variance
            by the square of that quantity, so the propagation matches the
            pyFAI/azint convention ``I = Σ(w·I/C) / Σw``).
        variance_image: per-pixel variance map matching ``image.shape``.
            If None, Poisson statistics are assumed and the image itself
            is used as the variance vector (``Var(I_l) = I_l``). For
            calibrated detectors with explicit gain/read-noise models, pass
            the appropriate variance map.

    The returned variance is the propagated bin variance under the
    assumption of *uncorrelated* per-pixel measurements. For most
    photon-counting and integrating detectors this is the standard
    assumption; for detectors with significant inter-pixel charge sharing
    a correlation correction must be applied externally.
    """
    if image.ndim == 2:
        image_flat = image.reshape(-1)
    else:
        image_flat = image
    expected = geom.n_pixels_y * geom.n_pixels_z
    if image_flat.numel() != expected:
        raise ValueError(
            f"image numel {image_flat.numel()} != n_pixels {expected}"
        )
    image_flat = image_flat.to(dtype=geom.dtype, device=geom.device)

    if mode == "floor":
        sp = geom.csr_floor
        sp_sq = geom.csr_floor_sq
    elif mode == "bilinear":
        sp = geom.csr_bilinear
        sp_sq = geom.csr_bilinear_sq
    elif mode == "gradient":
        sp = geom.csr_gradient
        sp_sq = geom.csr_gradient_sq
    else:
        raise ValueError(f"unknown integration mode {mode!r}")
    if sp_sq is None:
        raise RuntimeError(
            f"build_csr was called without compute_variance=True (mode={mode!r}); "
            "rebuild with compute_variance=True to use integrate_with_variance."
        )

    if variance_image is None:
        var_flat = image_flat
    else:
        if variance_image.ndim == 2:
            var_flat = variance_image.reshape(-1)
        else:
            var_flat = variance_image
        if var_flat.numel() != expected:
            raise ValueError(
                f"variance_image numel {var_flat.numel()} != n_pixels {expected}"
            )
        var_flat = var_flat.to(dtype=geom.dtype, device=geom.device)

    raw_int = torch.matmul(sp, image_flat.unsqueeze(1)).squeeze(1)
    raw_var = torch.matmul(sp_sq, var_flat.unsqueeze(1)).squeeze(1)

    if normalize:
        valid = geom.area_per_bin > AREA_THRESHOLD
        denom = torch.clamp(geom.area_per_bin, min=AREA_THRESHOLD)
        intensity = torch.where(valid, raw_int / denom, torch.zeros_like(raw_int))
        variance = torch.where(valid, raw_var / (denom * denom), torch.zeros_like(raw_var))
    else:
        intensity = raw_int
        variance = raw_var
    return (intensity.reshape(geom.n_r, geom.n_eta),
            variance.reshape(geom.n_r, geom.n_eta))


# ─────────────────────────────────────────────────────────────────────────────
# 1D profile reductions
# ─────────────────────────────────────────────────────────────────────────────
def profile_1d(
    int2d: torch.Tensor,
    geom: CSRGeometry,
    *,
    mode: str = "area_weighted",
) -> torch.Tensor:
    """Reduce (n_r, n_eta) → (n_r,) over the η axis.

    Modes:
        'area_weighted' : Σ(I·A) / Σ(A) per R bin
                           (matches calculate_1D_profile_kernel).
        'simple_mean'   : Σ(I for A>0) / count(A>0)
                           (matches calculate_1D_profile_simple_mean_kernel).
    """
    area_2d = geom.area_per_bin.reshape(geom.n_r, geom.n_eta)
    valid = area_2d > AREA_THRESHOLD
    if mode == "area_weighted":
        weighted_sum = (int2d * area_2d * valid).sum(dim=1)
        area_sum = (area_2d * valid).sum(dim=1)
        return torch.where(
            area_sum > AREA_THRESHOLD,
            weighted_sum / torch.clamp(area_sum, min=AREA_THRESHOLD),
            torch.zeros_like(weighted_sum),
        )
    elif mode == "simple_mean":
        valid_f = valid.to(int2d.dtype)
        sum_int = (int2d * valid_f).sum(dim=1)
        n_valid = valid_f.sum(dim=1)
        return torch.where(
            n_valid > 0,
            sum_int / torch.clamp(n_valid, min=1.0),
            torch.zeros_like(sum_int),
        )
    else:
        raise ValueError(f"unknown profile_1d mode {mode!r}")


def profile_1d_with_variance(
    int2d: torch.Tensor,
    var2d: torch.Tensor,
    geom: CSRGeometry,
    *,
    mode: str = "area_weighted",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce 2D (R, Eta) intensity + variance to 1D radial profile.

    Returns ``(I_1d, Var_1d)``, both shape (n_r,).

    For ``mode='area_weighted'``:
        I_1d(R)   = Σ_η (A·I) / Σ_η A
        Var_1d(R) = Σ_η (A² · Var) / (Σ_η A)²

    The denominator is squared so that σ_1d propagates correctly through
    the area-weighted mean. ``simple_mean`` divides Var by N² where N is
    the number of valid η bins for that R.
    """
    area_2d = geom.area_per_bin.reshape(geom.n_r, geom.n_eta)
    valid = area_2d > AREA_THRESHOLD
    if mode == "area_weighted":
        area_v = area_2d * valid
        weighted_sum = (int2d * area_v).sum(dim=1)
        weighted_var = (var2d * area_v * area_v).sum(dim=1)
        area_sum = area_v.sum(dim=1)
        denom = torch.clamp(area_sum, min=AREA_THRESHOLD)
        ok = area_sum > AREA_THRESHOLD
        I = torch.where(ok, weighted_sum / denom, torch.zeros_like(weighted_sum))
        V = torch.where(ok, weighted_var / (denom * denom), torch.zeros_like(weighted_var))
        return I, V
    if mode == "simple_mean":
        valid_f = valid.to(int2d.dtype)
        sum_int = (int2d * valid_f).sum(dim=1)
        sum_var = (var2d * valid_f).sum(dim=1)
        n_valid = valid_f.sum(dim=1)
        denom = torch.clamp(n_valid, min=1.0)
        ok = n_valid > 0
        I = torch.where(ok, sum_int / denom, torch.zeros_like(sum_int))
        V = torch.where(ok, sum_var / (denom * denom), torch.zeros_like(sum_var))
        return I, V
    raise ValueError(f"unknown profile_1d_with_variance mode {mode!r}")


def r_axis(*, n_r: int, RMin: float, RBinSize: float) -> np.ndarray:
    """R values at bin centers, shape (n_r,)."""
    return RMin + RBinSize * (np.arange(n_r, dtype=np.float64) + 0.5)


def eta_axis(*, n_eta: int, EtaMin: float, EtaBinSize: float) -> np.ndarray:
    """Eta values at bin centers, shape (n_eta,)."""
    return EtaMin + EtaBinSize * (np.arange(n_eta, dtype=np.float64) + 0.5)
