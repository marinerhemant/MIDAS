"""Decoded observation bitmap, kept on the autograd device.

The C path stores the on-detector "did anything fire here?" bitmap as a
packed array of 32-bit ints in ``SpotsInfo.bin`` (one bit per pixel,
flattened over ``layer * frame * y * z``). For the differentiable fit we
need to evaluate it at fractional pixel coordinates, which means
unpacking it once into a dense float32 tensor that ``torch.nn.functional``
can sample.

Memory at 2048 × 2048 × 1440 × 1 distance × 4 bytes = ~24 GB, which is
larger than most consumer GPUs. Two strategies:

- Default: keep the dense volume in CPU pinned memory and stream it to
  the GPU as needed via ``grid_sample``.
- Sparse: convert to a sparse COO tensor where the indices are the
  pixels that fired. For high-quality NF data this is typically <1 % of
  the volume, fitting easily on a single GPU.

This module provides both representations behind a single
:class:`ObsVolume` interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _reduce_over_spots(
    hits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Sum hits and valid counts across the spot axes.

    The forward model produces ``(..., K, M)``-shaped tensors where
    ``K = 2`` (omega solutions) and ``M`` is the number of HKL
    reflections; the C ``CalcFracOverlap`` treats both as part of the
    same flat spot list, so we collapse ``K`` and ``M`` together.

    Cases handled:

    - ``valid.ndim == 1``: flat ``(M,)`` spot list (test inputs) →
      returns a 0-D scalar.
    - ``valid.ndim >= 2``: last two dims are ``(K, M)`` → returns a
      tensor with leading shape ``valid.shape[:-2]``. For per-voxel
      single calls this is a 0-D scalar; for screening across a batch
      of orientations it is the per-orientation fraction vector.
    """
    if valid.ndim == 1:
        return hits.sum() / valid.sum().clamp(min=1.0)
    flat_hits = hits.flatten(start_dim=-2)
    flat_valid = valid.flatten(start_dim=-2)
    return flat_hits.sum(dim=-1) / flat_valid.sum(dim=-1).clamp(min=1.0)


def _unpack_bits(packed: np.ndarray, total_bits: int) -> np.ndarray:
    """Unpack a flat array of 32-bit ints into a flat uint8 bit array.

    The C ``SetBit`` macro stores bit ``k`` in word ``k // 32`` at offset
    ``k % 32`` (LSB-first within the word). NumPy's ``unpackbits``
    treats ``uint8`` words MSB-first, so we go via a ``uint8`` view and
    a per-word reverse to recover LSB-first ordering.
    """
    if packed.dtype != np.uint32:
        packed = packed.view(np.uint32)
    # View as uint8, 4 bytes per word, little-endian on x86/ARM.
    bytes_view = packed.view(np.uint8)
    # unpackbits is MSB-first; reverse the order within each byte to
    # turn it LSB-first, which matches the C bit layout because on a
    # little-endian host the low byte holds bits 0..7 of the uint32.
    bits = np.unpackbits(bytes_view, bitorder="little")
    return bits[:total_bits]


class ObsVolume:
    """Observation bitmap usable as a differentiable target.

    Two storage modes:

    - **Dense** (``dense`` set, ``packed`` None): the bitmap is held as
      a ``(D, F, H, W)`` tensor of dtype ``float32`` (for the soft /
      grid_sample path) or ``uint8`` (for the hard path). Used when
      ``soft_fraction`` is on the hot path or when reproducibility
      checks need direct access to the bitmap.
    - **Packed** (``packed`` set, ``dense`` None): the bitmap is held
      as a flat ``(total_bytes,) uint8`` tensor matching the layout
      of the C ``SpotsInfo.bin`` (1 bit per pixel, little-endian,
      ``bit_idx = (((d * F + f) * H + y) * W + z)``). 8× smaller than
      ``uint8`` dense, 32× smaller than ``float32`` dense — the entire
      Au-example obs at 750 MB instead of 24 GB. ``hard_fraction``
      uses :func:`torch.bitwise_right_shift` + AND to extract the bit.

    The packed mode is the default since v0.4 — it's faster (better
    cache locality on the H100 SM L2) and matches the C storage
    format. Pass ``packed=False`` at construction if you need
    ``soft_fraction``.

    Parameters
    ----------
    dense : torch.Tensor (D, F, H, W), optional
        Dense bitmap (uint8 or float32). Mutually exclusive with
        ``packed``.
    packed : torch.Tensor (total_bytes,) uint8, optional
        Bit-packed bitmap. Mutually exclusive with ``dense``.
    n_distances, n_frames, n_y, n_z : int
        Required when ``packed`` is supplied (the packed tensor has no
        intrinsic shape).
    device : torch.device
        Device the volume lives on.

    Notes
    -----
    The convention for indexing is **(distance, frame, y, z)**, matching
    the SoA layout the forward model produces (``y_pixel``, ``z_pixel``,
    ``frame_nr`` for each predicted spot, optionally with a leading
    distance dim under ``multi_mode="layered"``).
    """

    def __init__(
        self,
        dense: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        *,
        packed: Optional[torch.Tensor] = None,
        n_distances: Optional[int] = None,
        n_frames: Optional[int] = None,
        n_y: Optional[int] = None,
        n_z: Optional[int] = None,
    ):
        if (dense is None) == (packed is None):
            raise ValueError(
                "exactly one of `dense` or `packed` must be supplied"
            )

        if dense is not None:
            if dense.ndim != 4:
                raise ValueError(
                    f"ObsVolume `dense` expects (D, F, H, W); "
                    f"got shape {dense.shape}"
                )
            self.dense = dense
            self.packed = None
            self.n_distances = int(dense.shape[0])
            self.n_frames = int(dense.shape[1])
            self.n_y = int(dense.shape[2])
            self.n_z = int(dense.shape[3])
            self.device = device or dense.device
        else:
            if (n_distances is None or n_frames is None
                    or n_y is None or n_z is None):
                raise ValueError(
                    "packed mode requires explicit "
                    "n_distances/n_frames/n_y/n_z"
                )
            if packed.dtype != torch.uint8:
                raise ValueError(
                    f"packed obs must be uint8, got {packed.dtype}"
                )
            total_bits = n_distances * n_frames * n_y * n_z
            expected_bytes = (total_bits + 7) // 8
            if packed.numel() < expected_bytes:
                raise ValueError(
                    f"packed obs has {packed.numel()} bytes, "
                    f"need at least {expected_bytes}"
                )
            self.dense = None
            self.packed = packed
            self.n_distances = int(n_distances)
            self.n_frames = int(n_frames)
            self.n_y = int(n_y)
            self.n_z = int(n_z)
            self.device = device or packed.device

    # ------------------------------------------------------------------
    @classmethod
    def from_spotsinfo(
        cls,
        path: str | Path,
        n_distances: int,
        n_frames: int,
        n_y: int,
        n_z: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.uint8,
        packed: bool = True,
    ) -> "ObsVolume":
        """Build an :class:`ObsVolume` from a packed ``SpotsInfo.bin``.

        The C code's bit ordering (per ``ProcessImagesCombined.c``) is::

            bitIdx = (((layer * n_frames) + frame) * n_y + y) * n_z + z
            TestBit(A, k) = A[k/32] & (1 << (k%32))

        i.e. the linear pixel index uses ``(layer, frame, y, z)`` in
        row-major order, packed into 32-bit words LSB-first. On a
        little-endian host this is bit-equivalent to packing into
        ``uint8`` bytes with bit 0 in the LSB of byte 0.

        Parameters
        ----------
        packed : bool, default True
            If True (the default), keep the bits packed in a
            ``(total_bytes,) uint8`` tensor — 1 bit per pixel, the same
            layout the C code consumes. 8× smaller than ``uint8`` dense,
            32× smaller than ``float32`` dense; only ``hard_fraction``
            works in this mode. If False, unpack to ``(D, F, H, W)``
            with the requested ``dtype`` (use this when you also need
            the soft / grid_sample path).
        dtype : torch.dtype
            Only consulted when ``packed=False``.
        """
        total_bits = n_distances * n_frames * n_y * n_z
        if packed:
            # Read bytes directly — the file is already in C
            # ``TestBit`` order on little-endian. Trim to the exact
            # number of bytes needed (the file may have trailing
            # padding bits in the last uint32 word).
            need_bytes = (total_bits + 7) // 8
            file_bytes = np.fromfile(path, dtype=np.uint8)
            if file_bytes.size < need_bytes:
                raise ValueError(
                    f"SpotsInfo.bin {path} has {file_bytes.size * 8} bits, "
                    f"need {total_bits}"
                )
            packed_t = torch.from_numpy(file_bytes[:need_bytes].copy()).to(
                device=device,
            )
            return cls(
                packed=packed_t, device=torch.device(device),
                n_distances=n_distances, n_frames=n_frames,
                n_y=n_y, n_z=n_z,
            )

        # Dense path (legacy). Unpack the bits and reshape.
        packed_arr = np.fromfile(path, dtype=np.uint32)
        if packed_arr.size * 32 < total_bits:
            raise ValueError(
                f"SpotsInfo.bin {path} has {packed_arr.size * 32} bits, "
                f"need {total_bits}"
            )
        bits = _unpack_bits(packed_arr, total_bits)
        bits = bits.reshape(n_distances, n_frames, n_y, n_z)
        return cls(
            dense=torch.from_numpy(bits).to(
                dtype=dtype, device=device, copy=True,
            ),
            device=torch.device(device),
        )

    @classmethod
    def from_dense_array(
        cls,
        arr: np.ndarray,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.uint8,
        packed: bool = False,
    ) -> "ObsVolume":
        """Wrap an already-decoded ``(D, F, H, W)`` ndarray (test helper).

        When ``packed=True``, packs the array's bits into the same
        layout :func:`from_spotsinfo` produces — useful for parity
        tests of the bit-extraction path against the dense-path baseline.
        """
        if arr.ndim != 4:
            raise ValueError(f"dense array must be 4D, got {arr.shape}")
        D, F_, H, W = arr.shape
        if packed:
            flat_bits = arr.astype(np.uint8).flatten() != 0   # bool flat
            # packbits: pack 8 bits into 1 byte LSB-first.
            packed_arr = np.packbits(flat_bits.view(np.uint8), bitorder="little")
            packed_t = torch.from_numpy(packed_arr.copy()).to(device=device)
            return cls(
                packed=packed_t, device=torch.device(device),
                n_distances=D, n_frames=F_, n_y=H, n_z=W,
            )
        return cls(
            dense=torch.from_numpy(arr.astype(np.float32, copy=False)).to(
                dtype=dtype, device=device, copy=True,
            ),
            device=torch.device(device),
        )

    # ------------------------------------------------------------------
    def lookup(
        self,
        d_idx: torch.Tensor,
        f_idx: torch.Tensor,
        y_idx: torch.Tensor,
        z_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Integer-coordinate single-pixel lookup, dtype-agnostic.

        Indexes into either the dense ``(D, F, H, W)`` tensor or the
        packed ``(total_bytes,)`` bit array, returning the same 0/1
        ``uint8`` result either way. Used by the screen kernel; the
        per-pixel accumulator multiplies by these and the final
        fraction reduction handles the dtype promotion.

        All four index tensors must broadcast to the same shape and
        contain valid (in-range) integer indices — bounds checking
        is the caller's job.
        """
        if self.packed is not None:
            F_, H, W = self.n_frames, self.n_y, self.n_z
            bit_idx = (
                ((d_idx.to(torch.int64) * F_ + f_idx) * H + y_idx)
                * W + z_idx
            )
            byte_idx = bit_idx >> 3
            bit_pos = (bit_idx & 7).to(torch.uint8)
            return (self.packed[byte_idx] >> bit_pos) & 1
        return self.dense[d_idx, f_idx, y_idx, z_idx]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hard_fraction(
        self,
        frame_nr: torch.Tensor,
        y_pixel: torch.Tensor,
        z_pixel: torch.Tensor,
        valid: torch.Tensor,
        refl_weight: "torch.Tensor | None" = None,
        bounds_already_applied: bool = False,
    ) -> torch.Tensor:
        """Discrete (non-differentiable) ``CalcFracOverlap`` for screening.

        Parameters
        ----------
        frame_nr : Tensor (..., M)
            Fractional frame indices (will be ``.long()``-truncated, the
            C convention via ``(int) frame_nr``).
        y_pixel, z_pixel : Tensor (D, ..., M) or (..., M)
            Per-distance pixel indices. Single-distance input is
            broadcast across the leading ``D`` dim.
        valid : Tensor (..., M)
            Validity mask (1 = predicted spot).
        refl_weight : Tensor (M,) or None
            Per-reflection weight applied to BOTH numerator and denominator.
            ``None`` (default) means all-ones, i.e. exactly the historical
            ``CalcFracOverlap``. This one hook gives all three metrics:

            ==============  ==========================================
            ones            C_raw    -- every listed reflection counts
            (f2 > 0)        C_filt   -- basis-forbidden ones excluded
            f2              C_weight -- weighted by |F|^2
            ==============  ==========================================

            Why this matters: space-group extinction rules do not see
            basis-dependent extinctions, so ``hkls.csv`` can list reflections
            with |F|^2 = 0 that no crystal can produce. They land in the
            denominator and cap the achievable fraction -- 126 of 736 for dhcp
            a DHCP polytype, a ceiling of 0.829, while fcc (nothing extra extinct) can
            still reach 1.000.

            NOT Lorentz-weighted, deliberately: for per-frame threshold
            detection the rotation-method Lorentz factor cancels
            (Domega = w_rlp*L so I_peak = I_int/Domega is L-free). Verified on
            nf_sampleB_htB_s2: spot peak height is flat in eta where 1/|sin eta|
            predicts 4.3x, and spot density along a ring rises as sin eta, the
            complementary half of the same model.

        Returns
        -------
        Tensor (...)
            Fraction = (weighted matches over all distances) / (weighted valid
            predicted spots), with zero-denominator → 0.
        """
        D, F_, H, W = self.n_distances, self.n_frames, self.n_y, self.n_z
        # Promote (..., M) -> (1, ..., M) -> (D, ..., M) where needed.
        if y_pixel.ndim == frame_nr.ndim:
            y_pixel = y_pixel.unsqueeze(0).expand(D, *y_pixel.shape)
            z_pixel = z_pixel.unsqueeze(0).expand(D, *z_pixel.shape)
        elif y_pixel.shape[0] != D:
            raise ValueError(
                f"y_pixel leading dim {y_pixel.shape[0]} != D={D}"
            )

        f_idx = frame_nr.long().clamp(0, F_ - 1)
        y_idx = y_pixel.long().clamp(0, H - 1)
        z_idx = z_pixel.long().clamp(0, W - 1)

        # Out-of-bounds spots count as miss.
        #
        # ``bounds_already_applied`` skips this when the caller's ``valid``
        # ALREADY encodes it. project_to_detector folds exactly these six
        # comparisons into ``valid`` (midas_diffract/forward.py, layer_bounds_ok
        # -> layer_valid -> prod over layers), so on the fit path recomputing
        # them changes nothing: measured on real data, valid * in_bounds ==
        # valid exactly, with zero spots dropped. It is still the default for
        # any other caller, since this is a public method and a caller passing
        # a looser ``valid`` genuinely needs the check.
        if bounds_already_applied:
            in_bounds = None
        else:
            in_bounds = (
                (frame_nr >= 0) & (frame_nr < F_)
                & (y_pixel >= 0).all(dim=0) & (y_pixel < H).all(dim=0)
                & (z_pixel >= 0).all(dim=0) & (z_pixel < W).all(dim=0)
            )

        # Per-distance hit lookup → product across D.
        # f_idx is (..., M); broadcast to (D, ..., M) for the lookup.
        f_idx_d = f_idx.unsqueeze(0).expand(D, *f_idx.shape)
        device = self.dense.device if self.dense is not None else self.packed.device

        if self.packed is not None:
            # Bit-extraction lookup. Compute the linear ``bit_idx`` for
            # every (d, f, y, z) tuple, split into ``byte_idx``
            # (= bit_idx // 8) and ``bit_pos`` (= bit_idx % 8), gather
            # the byte, then extract the bit. Matches the C
            # ``TestBit`` macro on a little-endian host.
            d_idx = torch.arange(D, device=device).reshape(
                D, *([1] * f_idx.ndim)
            ).expand_as(f_idx_d)
            bit_idx = (
                ((d_idx.to(torch.int64) * F_ + f_idx_d) * H + y_idx)
                * W + z_idx
            )                                                # (D, ..., M)
            byte_idx = bit_idx >> 3                          # divide by 8
            bit_pos = (bit_idx & 7).to(torch.uint8)
            byte_val = self.packed[byte_idx]                 # uint8 (D, ..., M)
            hits_d = (byte_val >> bit_pos) & 1               # uint8, 0/1
        else:
            d_idx = torch.arange(D, device=device).reshape(
                D, *([1] * f_idx.ndim)
            ).expand_as(f_idx_d)
            hits_d = self.dense[d_idx, f_idx_d, y_idx, z_idx]   # (D, ..., M)

        # The volume yields uint8 0/1 values; ``prod`` over uint8
        # works because there's no overflow on 0/1. Promotion to
        # float happens at the multiplication with ``weight``.
        hits_all = hits_d.prod(dim=0)                       # (..., M)

        # Match C ``CalcFracOverlap`` (SharedFuncsFit.c:565-649) exactly:
        # the denominator is ``TotalPixels`` — only spots that survive
        # ALL the bounds checks at ALL distances contribute to it.
        weight = (valid if in_bounds is None
                  else valid * in_bounds.to(valid.dtype))
        if refl_weight is not None:
            # (M,) broadcasts against the trailing reflection axis of (..., M)
            weight = weight * refl_weight.to(
                dtype=weight.dtype, device=weight.device)
        hits_all = hits_all.to(weight.dtype) * weight
        return _reduce_over_spots(hits_all, weight)

    # ------------------------------------------------------------------
    def soft_fraction(
        self,
        frame_nr: torch.Tensor,
        y_pixel: torch.Tensor,
        z_pixel: torch.Tensor,
        valid: torch.Tensor,
        sigma_px: float,
    ) -> torch.Tensor:
        """Differentiable Gaussian-splat surrogate for ``CalcFracOverlap``.

        Each predicted spot is treated as a 3-D Gaussian centred at
        (frame_nr, y_pixel, z_pixel) with std-dev (1, σ_px, σ_px), and the
        overlap with the obs volume is the trilinear sample of a
        pre-blurred ``self.dense`` evaluated at that centre. ``σ_z`` (the
        omega axis) is fixed at 1 frame; widening it on physical grounds
        would just add cost without changing the basin.

        Implementation note
        -------------------
        Strict math says: convolve ``dense`` with the same Gaussian and
        sample at the centre. Equivalent and far cheaper:
        :func:`torch.nn.functional.grid_sample` with ``mode='bilinear'``
        already gives a piecewise-linear (i.e. roughly Gaussian-with-σ-1)
        sample of ``dense``. We therefore:

        - if ``sigma_px <= 1`` use bare trilinear sampling (fast path);
        - if ``sigma_px > 1`` apply a ``sigma_px``-radius Gaussian blur
          to ``dense`` once, cache it, then trilinear sample.

        This is good enough for the small-voxel case the user runs in
        practice. The triangle-shape-aware path is documented in the
        plan as deferred.
        """
        if self.dense is None:
            raise TypeError(
                "soft_fraction needs a dense floating-point ObsVolume; "
                "this volume is in packed-bit storage. Pass "
                "``packed=False, dtype=torch.float32`` to "
                "``ObsVolume.from_spotsinfo`` for the soft / "
                "Gaussian-splat path."
            )
        D, F_, H, W = self.dense.shape
        device = self.dense.device

        # ``grid_sample`` requires a float volume. The uint8 fast-path
        # (the default since v0.3) only supports the hard-frac kernel,
        # so error early with a clear message rather than try to
        # upcast 24 GB of obs to float on demand.
        if not self.dense.dtype.is_floating_point:
            raise TypeError(
                f"soft_fraction needs a floating-point ObsVolume; "
                f"got dtype={self.dense.dtype}. Pass "
                "``dtype=torch.float32`` to ``ObsVolume.from_spotsinfo`` "
                "if you want the soft / Gaussian-splat path."
            )

        # ALWAYS blur -- there is no bare-trilinear fast path any more.
        #
        # The old code took bare trilinear whenever sigma_px <= 1.0, on the
        # docstring's claim that bilinear sampling is "roughly Gaussian-with-
        # sigma-1". It is not: bilinear is a tent that reaches ZERO one pixel
        # away, so on a sparse binary volume the score collapses with sub-pixel
        # offset even when the spot lands in the CORRECT pixel and the hard
        # fraction is 1.0. Measured: offset 0.25 px -> 0.316, 0.50 px -> 0.0625,
        # 0.75 px -> 0.0039, and the per-distance product squares the penalty.
        #
        # This mattered in practice because auto_sigma_px() clamps at exactly
        # 1.0, so typical NF configs took this branch every time.
        sampled = self._sample_dense(
            self._blurred(sigma_px), frame_nr, y_pixel, z_pixel,
        )

        sampled = sampled * valid

        return _reduce_over_spots(sampled, valid)

    # ------------------------------------------------------------------
    def _sample_dense(
        self,
        volume: torch.Tensor,
        frame_nr: torch.Tensor,
        y_pixel: torch.Tensor,
        z_pixel: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable trilinear sample of ``volume`` at the predicted
        per-distance pixel coordinates.

        ``y_pixel`` / ``z_pixel`` may be broadcastable to ``frame_nr``'s
        shape (single distance) or have a leading ``D`` dim
        (multi-distance, layered semantics — sample each distance and
        ``prod`` across them).

        Returns the per-spot scalar sample, broadcast back to the
        ``frame_nr`` shape ``(..., M)`` so the caller can multiply by
        ``valid`` directly.
        """
        D, F_, H, W = volume.shape
        if y_pixel.ndim == frame_nr.ndim:
            y_pixel_d = y_pixel.unsqueeze(0).expand(D, *y_pixel.shape)
            z_pixel_d = z_pixel.unsqueeze(0).expand(D, *z_pixel.shape)
        else:
            y_pixel_d = y_pixel
            z_pixel_d = z_pixel

        # Build (D, ..., M, 3) normalised coords for grid_sample.
        # grid_sample expects coords in [-1, 1] with order (x=z, y=y, z=frame).
        f_norm = 2.0 * (frame_nr + 0.5) / F_ - 1.0
        y_norm = 2.0 * (y_pixel_d + 0.5) / H - 1.0
        z_norm = 2.0 * (z_pixel_d + 0.5) / W - 1.0

        # f_norm shape: (..., M); expand to (D, ..., M)
        f_norm_d = f_norm.unsqueeze(0).expand(D, *f_norm.shape)

        # Stack to (D, ..., M, 3)
        grid = torch.stack([z_norm, y_norm, f_norm_d], dim=-1)

        # grid_sample wants input shape (N, C, D, H, W) and grid (N, ..., 3).
        # We sample per-distance then product. Treat each D as an input
        # channel slice in a single batched grid_sample call.
        # Reshape volume from (D, F, H, W) -> (1, D, F, H, W) so D is
        # the channel axis.
        vol5 = volume.unsqueeze(0)
        # grid as (N=1, ..., M, D, 3)? grid_sample for 5D wants
        # grid (N, D_out, H_out, W_out, 3). We'll flatten the per-spot
        # axes and use (N=1, 1, 1, P, 3) for P spots, sampling all D
        # input channels.
        flat = grid.reshape(D, -1, 3)        # (D, P, 3)
        P = flat.shape[1]
        grid5 = flat.permute(1, 0, 2).reshape(1, P, D, 1, 3)  # (1, P, D, 1, 3)

        # grid_sample requires the grid and input dtypes to match. The
        # forward path runs in float64 by default; the obs volume is
        # often float32 (memory). Cast the grid to the volume dtype so
        # the kernel accepts it. The result has the same dtype as the
        # input volume; the multiplications upstream will promote back
        # if the eulers are in float64.
        if grid5.dtype != vol5.dtype:
            grid5 = grid5.to(vol5.dtype)

        sampled = F.grid_sample(
            vol5, grid5, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        # sampled shape: (1, D, P, D, 1)? Let's untangle: input has
        # C=D channels; output has shape (N, C, D_out, H_out, W_out)
        # = (1, D, P, D, 1). For each spot p we want diag in (C, H_out)
        # — channel d evaluated at spatial slot d.
        sampled = sampled.squeeze(0).squeeze(-1)              # (D, P, D)
        # Take diagonal across (C, H_out=D) → (D, P) per detector.
        diag_idx = torch.arange(D, device=volume.device)
        sampled = sampled[diag_idx, :, diag_idx]              # (D, P)

        # Restore (D, ..., M)
        sampled = sampled.reshape(D, *frame_nr.shape)

        # Per-distance product → (..., M).
        return sampled.prod(dim=0)

    # ------------------------------------------------------------------
    _blurred_cache: Optional["torch.Tensor"] = None
    _blurred_sigma: float = -1.0

    def _blurred(self, sigma_px: float) -> torch.Tensor:
        """Cached Gaussian-blurred dense volume."""
        if (self._blurred_cache is not None
                and abs(self._blurred_sigma - sigma_px) < 1e-6):
            return self._blurred_cache

        radius = max(1, int(round(3.0 * sigma_px)))
        # 1-D kernel
        offsets = torch.arange(
            -radius, radius + 1, device=self.dense.device, dtype=self.dense.dtype,
        )
        kernel_1d = torch.exp(-0.5 * (offsets / sigma_px) ** 2)
        # PEAK-normalise (max == 1), NOT sum-normalise.
        #
        # A sum-normalised kernel conserves MASS, which is right for smoothing an
        # image but wrong for an overlap score on a SPARSE BINARY volume: it
        # spreads each lit pixel over the kernel footprint, so the value AT a lit
        # pixel collapses to ~1/(2*pi*sigma^2). Measured on a synthetic volume,
        # a spot sitting exactly on a lit pixel scored 0.0050 at sigma=1.5 and
        # 0.0016 at sigma=2.0 instead of ~1.0 -- and that is precisely the
        # ~0.001-0.005 "overlap" the multipoint driver reported on real data at a
        # geometry whose HARD confidence is 1.0.
        #
        # With max == 1 the blurred volume still reads ~1 at a lit pixel and
        # falls off smoothly around it, so soft_fraction stays comparable to the
        # hard fraction it is a surrogate for, and the optimiser gets a gradient
        # that points the same way.
        kernel_1d = kernel_1d / kernel_1d.max()

        D, F_, H, W = self.dense.shape
        kH = kernel_1d.reshape(1, 1, -1, 1)
        kW = kernel_1d.reshape(1, 1, 1, -1)

        # Blur IN PLACE, in chunks along the (distance, frame) axis.
        #
        # A real NF obs volume is huge -- 2 x 1800 x 2048^2 float32 is 56 GiB --
        # so the old out-of-place `x = conv2d(x, ...)` held the original AND the
        # result simultaneously and OOM'd at 112 GiB even on a 140 GiB card.
        #
        # Each (H, W) frame blurs independently of every other frame, so the
        # separable H-then-W pass can be done per chunk and written straight
        # back over the input. Only a chunk-sized temporary is live at a time.
        #
        # NOTE this MUTATES self.dense. That is safe here because the soft path
        # is the only consumer of the dense volume (the hard path uses packed
        # bits), and a run uses a single sigma -- which `_blurred_sigma` records.
        # Re-blurring at a different sigma would compound, so that is refused.
        if self._blurred_sigma > 0.0:
            raise RuntimeError(
                f"ObsVolume was already blurred in place at sigma="
                f"{self._blurred_sigma}; cannot re-blur at {sigma_px}. Build a "
                f"fresh ObsVolume if you need a different splat width."
            )

        x = self.dense.reshape(D * F_, 1, H, W)
        n = x.shape[0]
        itemsize = x.element_size()
        # keep the transient under ~2 GiB
        chunk = max(1, int((2 << 30) // max(H * W * itemsize * 2, 1)))
        for i in range(0, n, chunk):
            sl = x[i:i + chunk]
            t = F.conv2d(sl, kH, padding=(radius, 0))
            t = F.conv2d(t, kW, padding=(0, radius))
            sl.copy_(t)
            del t

        self._blurred_cache = self.dense
        self._blurred_sigma = sigma_px
        return self._blurred_cache
