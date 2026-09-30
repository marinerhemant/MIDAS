"""Vectorized merger for multiple block-sharded ``AllPeaks_PS.bin`` /
``AllPeaks_PX.bin`` files.

Each input file already has the block-sharded format: a full-length header
with ``nPeaks=0`` for frames outside its block, populated rows for frames
inside its block. Merging is "for each frame, pick the (one) input that
has nPeaks > 0".

Implementation reads each input by parsing the header + slicing the data
section by frame, then writes the merged output via a single bulk
``np.tofile`` per section. No per-peak Python loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from midas_peakfit.postfit import N_PEAK_COLS


def _parse_ps_header(path: Path) -> Tuple[int, np.ndarray, np.ndarray, int]:
    """Read the PS.bin header. Returns (nFrames, nPeaks_per_frame,
    offsets_per_frame, header_size_bytes). The data section starts at
    ``header_size_bytes`` from the start of the file.
    """
    with open(path, "rb") as f:
        n_frames = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_peaks = np.frombuffer(f.read(n_frames * 4), dtype=np.int32).copy()
        offsets = np.frombuffer(f.read(n_frames * 8), dtype=np.int64).copy()
    header_size = 4 + n_frames * 4 + n_frames * 8
    return n_frames, n_peaks, offsets, header_size


def _parse_px_header(path: Path) -> Tuple[int, int, np.ndarray, np.ndarray, int]:
    """Read the PX.bin header. Returns (nFrames, NrPixels, nPeaks_per_frame,
    offsets_per_frame, header_size_bytes).
    """
    with open(path, "rb") as f:
        n_frames = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        nr_pixels = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_peaks = np.frombuffer(f.read(n_frames * 4), dtype=np.int32).copy()
        offsets = np.frombuffer(f.read(n_frames * 8), dtype=np.int64).copy()
    header_size = 4 + 4 + n_frames * 4 + n_frames * 8
    return n_frames, nr_pixels, n_peaks, offsets, header_size


def _read_frame_ps_bytes(path: Path, off: int, n_peaks: int) -> bytes:
    """Read ``n_peaks × N_PEAK_COLS × 8`` bytes starting at file offset ``off``."""
    nbytes = n_peaks * N_PEAK_COLS * 8
    if nbytes == 0:
        return b""
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(nbytes)


def _read_frame_px_bytes(path: Path, off: int, owning_offset_next: int) -> bytes:
    """Read the per-frame PX block from ``off`` up to ``owning_offset_next``.

    Each block has the same layout per frame (int32 nPx + int16 yz pairs per
    peak) so the byte slice between two consecutive offsets is exactly that
    frame's PX content.
    """
    nbytes = owning_offset_next - off
    if nbytes <= 0:
        return b""
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(nbytes)


def merge_block_outputs(
    block_dirs: List[Path],
    *,
    out_folder: Path,
    n_total_frames: int = None,  # ignored; derived from inputs
    nr_pixels: int = None,        # ignored; derived from inputs
) -> tuple[Path, Path]:
    """Merge ``len(block_dirs)`` block outputs into one ``out_folder``.

    Each block_dir is expected to contain ``Temp/AllPeaks_PS.bin`` and
    ``Temp/AllPeaks_PX.bin``. The output is written to
    ``out_folder/Temp/AllPeaks_{PS,PX}.bin``.

    Block frames are partitioned (each frame is owned by exactly one
    block); we just pick the owner for each frame.

    The saturated-region siblings (``AllPeaks_{PS,PX}_sat.bin``) are merged
    the same way when every block has them.
    """
    out = _merge_block_files(block_dirs, out_folder, "")
    if all((Path(d) / "Temp" / "AllPeaks_PS_sat.bin").exists()
           and (Path(d) / "Temp" / "AllPeaks_PX_sat.bin").exists()
           for d in block_dirs):
        _merge_block_files(block_dirs, out_folder, "_sat")
    return out


def _merge_block_files(block_dirs, out_folder, suffix: str) -> tuple[Path, Path]:
    out_dir = Path(out_folder) / "Temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ps = out_dir / f"AllPeaks_PS{suffix}.bin"
    out_px = out_dir / f"AllPeaks_PX{suffix}.bin"

    # ── Parse all block headers ────────────────────────────────────
    ps_paths = [Path(d) / "Temp" / f"AllPeaks_PS{suffix}.bin" for d in block_dirs]
    px_paths = [Path(d) / "Temp" / f"AllPeaks_PX{suffix}.bin" for d in block_dirs]

    block_ps = [_parse_ps_header(p) for p in ps_paths]
    block_px = [_parse_px_header(p) for p in px_paths]

    nF = block_ps[0][0]  # all blocks share nFrames
    nrPix = block_px[0][1]

    # ── Determine owner per frame and collect counts ───────────────
    n_peaks_per_frame = np.zeros(nF, dtype=np.int32)
    owner = np.full(nF, -1, dtype=np.int8)
    for bi, (_, n_peaks_b, _, _) in enumerate(block_ps):
        has_data = n_peaks_b > 0
        # Frames already owned by an earlier block keep their owner; only
        # claim un-owned ones. (Disjoint by construction, but defensive.)
        claim = has_data & (owner < 0)
        owner[claim] = bi
        n_peaks_per_frame[claim] = n_peaks_b[claim]

    # ── Compute merged PS.bin offsets ──────────────────────────────
    ps_header_size = 4 + nF * 4 + nF * 8
    ps_offsets = np.empty(nF, dtype=np.int64)
    cur = ps_header_size
    for f in range(nF):
        ps_offsets[f] = cur
        cur += int(n_peaks_per_frame[f]) * N_PEAK_COLS * 8

    # ── Compute merged PX.bin offsets ──────────────────────────────
    # For each frame, the PX byte size is the same as in its owning block
    # (same layout). We compute using the owning block's adjacent offsets.
    px_header_size = 4 + 4 + nF * 4 + nF * 8
    px_offsets = np.empty(nF, dtype=np.int64)
    cur = px_header_size
    px_byte_sizes = np.zeros(nF, dtype=np.int64)
    for f in range(nF):
        px_offsets[f] = cur
        ow = int(owner[f])
        if ow < 0:
            continue
        own_offsets = block_px[ow][3]
        # Next non-block offset gives the upper bound for this frame's bytes.
        # In the block file, frames after the block also have valid offsets;
        # their nPeaks is 0 so size is 0. We just look at offsets[f+1] (if any).
        if f + 1 < nF:
            byte_size = int(own_offsets[f + 1] - own_offsets[f])
        else:
            # Last frame — read till EOF.
            byte_size = (
                ps_paths[ow].stat().st_size  # placeholder; will fix below
            )
        # Actually use the actual file size for the last frame
        px_byte_sizes[f] = byte_size
        cur += byte_size
    # Fix last frame: use actual file size minus offsets[-1]
    last_f = nF - 1
    if owner[last_f] >= 0:
        ow = int(owner[last_f])
        own_offsets = block_px[ow][3]
        px_byte_sizes[last_f] = (
            px_paths[ow].stat().st_size - int(own_offsets[last_f])
        )
        # Recompute offsets for last frame chain
        if last_f > 0:
            px_offsets[last_f] = px_offsets[last_f - 1] + px_byte_sizes[last_f - 1]
        else:
            px_offsets[last_f] = px_header_size

    # ── Write merged PS.bin ────────────────────────────────────────
    with open(out_ps, "wb") as f_out:
        f_out.write(np.int32(nF).tobytes())
        f_out.write(n_peaks_per_frame.tobytes())
        f_out.write(ps_offsets.tobytes())
        # For each frame, copy the data bytes from the owning block.
        for fr in range(nF):
            ow = int(owner[fr])
            if ow < 0:
                continue
            own_n_peaks = block_ps[ow][1][fr]
            own_offsets = block_ps[ow][2]
            data = _read_frame_ps_bytes(
                ps_paths[ow], int(own_offsets[fr]), int(own_n_peaks)
            )
            if data:
                f_out.write(data)
    print(f"Wrote {out_ps} ({nF} frames)")

    # ── Write merged PX.bin ────────────────────────────────────────
    with open(out_px, "wb") as f_out:
        f_out.write(np.int32(nF).tobytes())
        f_out.write(np.int32(nrPix).tobytes())
        f_out.write(n_peaks_per_frame.tobytes())
        f_out.write(px_offsets.tobytes())
        for fr in range(nF):
            ow = int(owner[fr])
            if ow < 0:
                continue
            own_offsets = block_px[ow][3]
            byte_size = int(px_byte_sizes[fr])
            if byte_size <= 0:
                continue
            with open(px_paths[ow], "rb") as f_in:
                f_in.seek(int(own_offsets[fr]))
                data = f_in.read(byte_size)
                f_out.write(data)
    print(f"Wrote {out_px} ({nF} frames)")

    return out_ps, out_px


__all__ = ["merge_block_outputs"]
