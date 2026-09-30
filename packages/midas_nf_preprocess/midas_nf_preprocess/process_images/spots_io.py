"""Bit-packed ``SpotsInfo.bin`` writer/reader.

Compatible with ``FitOrientationOMP``, ``simulateNF``, and the C
``ProcessImagesCombined`` writer.

File layout (from ``ProcessImagesCombined.c`` L902-L934):

  - Total bits  = nDistances * NrPixelsY * NrPixelsZ * NrFilesPerLayer
  - Total bytes = ceil(total_bits / 32) * 4   (int32 word array)
  - Bit index for (layer, frame, y, z):
        BinNr = layer * (NrFilesPerLayer * NrPixelsY * NrPixelsZ)
              + frame * (NrPixelsY * NrPixelsZ)
              + y * NrPixelsZ
              + z
    where ``layer`` is 0-indexed and ``y``, ``z`` use the C's flipped
    convention ``y_C = NrPixelsY-1 - y_raw``, ``z_C = NrPixelsZ-1 - z_raw``
    (see C L1063-L1077).

The bit-set macro is ``SetBit(A, k) = A[k/32] |= 1 << (k%32)`` (C L28). We use
the same little-endian-within-word layout, which is what the reader macros
``TestBit`` (L30) expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch


def _flip_yz(
    y_raw: torch.Tensor, z_raw: torch.Tensor, nr_y: int, nr_z: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the C coordinate flip from L1063-L1064.

    The C does:
        ys = NrPixelsY - 1 - (i % NrPixelsY)
        zs = NrPixelsZ - 1 - (i / NrPixelsY)
    which is just a 180-degree rotation of the index space.
    """
    return (nr_y - 1 - y_raw, nr_z - 1 - z_raw)


def _bin_indices(
    layer: int,
    frame: int,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nr_files_per_layer: int,
    nr_pixels_y: int,
    nr_pixels_z: int,
) -> torch.Tensor:
    """Compute the bit indices into the SpotsInfo.bin bitstream.

    Mirrors C L1074-L1077:
        BinNr = layer * NrFilesPerLayer * NrOfPixels
              + frame * NrOfPixels
              + ythis * NrPixelsZ + zthis
    where ``ythis``, ``zthis`` are already flipped.
    """
    nr_pixels = nr_pixels_y * nr_pixels_z
    layer_off = int(layer) * int(nr_files_per_layer) * nr_pixels
    frame_off = int(frame) * nr_pixels
    return (
        torch.as_tensor(layer_off + frame_off, dtype=torch.int64)
        + y.to(torch.int64) * int(nr_pixels_z)
        + z.to(torch.int64)
    )


class SpotsBitMask:
    """Bit-packed spot mask backed by an int32 buffer (mmap-able)."""

    def __init__(
        self,
        n_layers: int,
        nr_files_per_layer: int,
        nr_pixels_y: int,
        nr_pixels_z: int,
        *,
        backing: Optional[np.ndarray] = None,
    ):
        self.n_layers = int(n_layers)
        self.nr_files_per_layer = int(nr_files_per_layer)
        self.nr_pixels_y = int(nr_pixels_y)
        self.nr_pixels_z = int(nr_pixels_z)
        self.total_bits = (
            self.n_layers
            * self.nr_files_per_layer
            * self.nr_pixels_y
            * self.nr_pixels_z
        )
        self.n_words = (self.total_bits + 31) // 32

        if backing is None:
            # Use uint32 internally to avoid NumPy 2.x signed-overflow on bit 31.
            # On disk the bytes are identical to int32 (the C code reads it as int *).
            self._buf = np.zeros(self.n_words, dtype=np.uint32)
        else:
            # Accept int32 or uint32; reinterpret the bytes as uint32 for arithmetic.
            if backing.dtype == np.uint32:
                self._buf = backing
            elif backing.dtype == np.int32:
                self._buf = backing.view(np.uint32)
            else:
                raise ValueError(
                    f"Backing buffer must be int32 or uint32, got {backing.dtype}"
                )
            if self._buf.shape[0] < self.n_words:
                raise ValueError(
                    f"Backing too small: {self._buf.shape[0]} words < {self.n_words} required"
                )

    # ------------------------------------------------------------------
    # Per-bit ops (slow Python path; used only for testing).
    # ------------------------------------------------------------------

    def set_bit_yz(self, layer: int, frame: int, y_raw: int, z_raw: int) -> None:
        y, z = (
            self.nr_pixels_y - 1 - int(y_raw),
            self.nr_pixels_z - 1 - int(z_raw),
        )
        bin_nr = (
            layer * self.nr_files_per_layer * self.nr_pixels_y * self.nr_pixels_z
            + frame * self.nr_pixels_y * self.nr_pixels_z
            + y * self.nr_pixels_z
            + z
        )
        if not (0 <= bin_nr < self.total_bits):
            raise IndexError(f"bit index {bin_nr} out of range [0, {self.total_bits})")
        self._buf[bin_nr // 32] |= np.uint32(1) << np.uint32(bin_nr % 32)

    def test_bit_yz(self, layer: int, frame: int, y_raw: int, z_raw: int) -> bool:
        y, z = (
            self.nr_pixels_y - 1 - int(y_raw),
            self.nr_pixels_z - 1 - int(z_raw),
        )
        bin_nr = (
            layer * self.nr_files_per_layer * self.nr_pixels_y * self.nr_pixels_z
            + frame * self.nr_pixels_y * self.nr_pixels_z
            + y * self.nr_pixels_z
            + z
        )
        return bool(self._buf[bin_nr // 32] & (np.uint32(1) << np.uint32(bin_nr % 32)))

    # ------------------------------------------------------------------
    # Vectorized: write a whole frame's labels in one shot.
    # ------------------------------------------------------------------

    def set_frame_from_labels(
        self,
        layer: int,
        frame: int,
        labels: torch.Tensor,
    ) -> int:
        """Set bits for every nonzero pixel in a label map.

        ``labels`` has shape ``[NrPixelsZ, NrPixelsY]`` (matching the in-memory
        image layout used by this package). The C convention's coordinate flip
        is applied here.

        Returns the number of bits set.
        """
        if labels.shape != (self.nr_pixels_z, self.nr_pixels_y):
            raise ValueError(
                f"labels shape {tuple(labels.shape)} != "
                f"({self.nr_pixels_z}, {self.nr_pixels_y})"
            )
        # Find the (z_raw, y_raw) coordinates of every nonzero pixel.
        nz = (labels != 0).nonzero(as_tuple=False)  # [N, 2] = (z_raw, y_raw)
        if nz.numel() == 0:
            return 0
        z_raw = nz[:, 0]
        y_raw = nz[:, 1]
        y, z = _flip_yz(y_raw, z_raw, self.nr_pixels_y, self.nr_pixels_z)
        bins = _bin_indices(
            layer,
            frame,
            y,
            z,
            nr_files_per_layer=self.nr_files_per_layer,
            nr_pixels_y=self.nr_pixels_y,
            nr_pixels_z=self.nr_pixels_z,
        ).cpu().numpy()
        # Bounds check (should always pass; assertion-level)
        if (bins < 0).any() or (bins >= self.total_bits).any():
            raise IndexError("bit index out of range; check pixel/layer params")
        # Bit-set into the uint32 buffer
        word_idx = bins // 32
        bit_pos = bins % 32
        # OR-combine via np.bitwise_or.at for duplicate-safe writes (none expected
        # within a single frame, but cheap insurance).
        np.bitwise_or.at(
            self._buf, word_idx, np.uint32(1) << bit_pos.astype(np.uint32)
        )
        return int(bins.size)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> None:
        """Write the buffer to disk in a format readable by FitOrientationOMP.

        On disk the bytes are identical regardless of int32 vs uint32 dtype
        (the C code reads it as ``int *``; only bit positions are checked).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self._buf.tobytes(order="C"))

    @classmethod
    def open_mmap(
        cls,
        path: Union[str, Path],
        n_layers: int,
        nr_files_per_layer: int,
        nr_pixels_y: int,
        nr_pixels_z: int,
        *,
        mode: str = "r+",
    ) -> "SpotsBitMask":
        """Open an existing SpotsInfo.bin as a memory-mapped buffer.

        ``mode='r+'`` for read/write (creates if absent and grows to the right
        size); ``mode='r'`` for read-only.
        """
        path = Path(path)
        total_bits = n_layers * nr_files_per_layer * nr_pixels_y * nr_pixels_z
        n_words = (total_bits + 31) // 32
        n_bytes = n_words * 4
        if mode == "r+":
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with open(path, "wb") as f:
                    f.truncate(n_bytes)
            elif path.stat().st_size < n_bytes:
                with open(path, "r+b") as f:
                    f.truncate(n_bytes)
            mm = np.memmap(path, dtype=np.uint32, mode="r+", shape=(n_words,))
        elif mode == "r":
            if not path.exists():
                raise FileNotFoundError(path)
            mm = np.memmap(path, dtype=np.uint32, mode="r", shape=(n_words,))
        else:
            raise ValueError(f"Unsupported mode '{mode}', use 'r' or 'r+'")
        return cls(
            n_layers=n_layers,
            nr_files_per_layer=nr_files_per_layer,
            nr_pixels_y=nr_pixels_y,
            nr_pixels_z=nr_pixels_z,
            backing=mm,
        )

    def flush(self) -> None:
        """If the backing is an mmap, flush it to disk."""
        if isinstance(self._buf, np.memmap):
            self._buf.flush()

    def count_bits(self) -> int:
        """Return the total number of set bits (slow; for tests/diagnostics)."""
        x = self._buf.astype(np.uint32, copy=True)
        # Hamming weight (vectorized SWAR popcount).
        x = x - ((x >> 1) & np.uint32(0x55555555))
        x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
        x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
        return int(((x * np.uint32(0x01010101)) >> 24).sum())

    @property
    def buffer(self) -> np.ndarray:
        """Direct access to the underlying uint32 buffer (for parity tests)."""
        return self._buf


class SparseGreyRecorder:
    """Grey levels of every pixel that ``SpotsBitMask`` sets, in the SAME coordinates.

    ``SpotsInfo.bin`` keeps one bit per pixel, so every NF fitter downstream compares
    against a binarised image and the measured intensity is gone. This keeps it: for each
    lit pixel ``(layer, frame, y, z, value)`` with ``y, z`` already in the C's flipped
    convention (``y = NrPixelsY-1-y_raw``, ``z = NrPixelsZ-1-z_raw``, :func:`_flip_yz`), so
    a row here and the matching bit in ``SpotsInfo.bin`` index the same detector pixel.

    ``value`` is the frame's ``filtered`` image at that pixel: the median-corrected residual
    after the blanket subtraction, the clamp and the spatial median (or, on the matched-filter
    path, the untouched residual). The blanket actually applied is recorded per layer, so
    ``value + blanket[layer]`` recovers the pre-blanket residual on the LoG / no-LoG paths.

    Opt-in via ``WriteGreyResidual 1``; with it absent the reduction is unchanged.
    """

    def __init__(self, nr_pixels_y: int, nr_pixels_z: int, spot_detect: str = "log"):
        self.nr_pixels_y = int(nr_pixels_y)
        self.nr_pixels_z = int(nr_pixels_z)
        self.spot_detect = str(spot_detect)
        self._chunks: list = []
        self.blanket: dict = {}

    def add(self, layer: int, frame: int, labels: torch.Tensor,
            values: torch.Tensor, blanket: float) -> int:
        self.blanket[int(layer)] = float(blanket)
        nz = (labels != 0).nonzero(as_tuple=False)            # (z_raw, y_raw)
        if nz.numel() == 0:
            return 0
        z_raw, y_raw = nz[:, 0], nz[:, 1]
        v = values.detach()[z_raw, y_raw].float().cpu().numpy()
        y, z = _flip_yz(y_raw, z_raw, self.nr_pixels_y, self.nr_pixels_z)
        n = int(v.size)
        self._chunks.append((np.full(n, layer, np.uint8), np.full(n, frame, np.int32),
                             y.cpu().numpy().astype(np.uint16), z.cpu().numpy().astype(np.uint16),
                             v.astype(np.float32)))
        return n

    @property
    def n_pixels(self) -> int:
        return int(sum(c[0].size for c in self._chunks))

    def write(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        cols = [np.concatenate([c[i] for c in self._chunks]) if self._chunks
                else np.zeros(0, dt) for i, dt in
                enumerate((np.uint8, np.int32, np.uint16, np.uint16, np.float32))]
        lay = np.array(sorted(self.blanket), np.int32)
        np.savez_compressed(
            path, layer=cols[0], frame=cols[1], y=cols[2], z=cols[3], value=cols[4],
            blanket_layer=lay, blanket_value=np.array([self.blanket[k] for k in lay], np.float32),
            nr_pixels_y=self.nr_pixels_y, nr_pixels_z=self.nr_pixels_z,
            spot_detect=self.spot_detect,
            convention="y=NrPixelsY-1-y_raw, z=NrPixelsZ-1-z_raw (same as SpotsInfo.bin)",
            value_note="filtered image at lit pixel; + blanket[layer] = pre-blanket residual (LoG/no-LoG paths)")
        return path
