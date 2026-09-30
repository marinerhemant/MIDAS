"""Binary writers for AllPeaks_PS.bin and AllPeaks_PX.bin.

Layout matches ``WriteConsolidatedPeakFiles`` in
``FF_HEDM/src/PeaksFittingConsolidatedIO.h`` exactly:

  AllPeaks_PS.bin
  ───────────────
    int32  nFrames
    int32  nPeaks[nFrames]
    int64  offset[nFrames]
    double peakRow[29]   ×   nPeaks_f   (concatenated per frame)

  AllPeaks_PX.bin
  ───────────────
    int32  nFrames
    int32  NrPixels
    int32  nPeaks[nFrames]
    int64  offset[nFrames]
    int32  nPx_p
    int16  y_p, z_p   (interleaved, nPx pairs)   per peak per frame

Multi-block runs: every block writes a full-length header. Frames outside
``[startFrame, endFrame)`` are slotted with ``nPeaks=0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from midas_peakfit.fit import FitOutput, N_UNC_COLS, N_MOMENT_COLS
from midas_peakfit.postfit import N_PEAK_COLS


@dataclass
class FrameAccumulator:
    """Mirror of the C ``FrameAccumulator`` struct.

    Collects all peak rows + pixel coords for one frame.
    """

    rows: List[np.ndarray] = field(default_factory=list)  # each (nPk, 29)
    rows_unc: List[np.ndarray] = field(default_factory=list)  # each (nPk, 9) or None
    rows_moment: List[np.ndarray] = field(default_factory=list)  # each (nPk, 6) or None
    pixel_y: List[np.ndarray] = field(default_factory=list)  # each (nPx,)
    pixel_z: List[np.ndarray] = field(default_factory=list)  # each (nPx,)
    n_px_per_peak: List[int] = field(default_factory=list)

    @property
    def n_peaks(self) -> int:
        return sum(r.shape[0] for r in self.rows)

    @property
    def has_unc(self) -> bool:
        return any(u is not None for u in self.rows_unc)

    @property
    def has_moment(self) -> bool:
        return any(m is not None for m in self.rows_moment)

    def add(self, fo: FitOutput) -> None:
        if fo.rows.shape[0] == 0:
            return
        self.rows.append(fo.rows)
        self.rows_unc.append(fo.rows_unc)
        self.rows_moment.append(fo.rows_moment)
        # Each peak in this region shares the same pixel set
        for _ in range(fo.rows.shape[0]):
            self.pixel_y.append(fo.pixel_y)
            self.pixel_z.append(fo.pixel_z)
            self.n_px_per_peak.append(int(fo.pixel_y.size))

    def stacked_rows(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, N_PEAK_COLS), dtype=np.float64)
        return np.concatenate(self.rows, axis=0)

    def stacked_unc(self) -> np.ndarray:
        """Stack ``rows_unc`` for all regions; missing rows become NaN."""
        if not self.rows:
            return np.zeros((0, N_UNC_COLS), dtype=np.float64)
        chunks = []
        for r, u in zip(self.rows, self.rows_unc):
            if u is None:
                chunks.append(np.full((r.shape[0], N_UNC_COLS), np.nan, dtype=np.float64))
            else:
                chunks.append(u)
        return np.concatenate(chunks, axis=0)

    def stacked_moment(self) -> np.ndarray:
        """Stack ``rows_moment`` for all regions; missing rows become NaN."""
        if not self.rows:
            return np.zeros((0, N_MOMENT_COLS), dtype=np.float64)
        chunks = []
        for r, m in zip(self.rows, self.rows_moment):
            if m is None:
                chunks.append(np.full((r.shape[0], N_MOMENT_COLS), np.nan, dtype=np.float64))
            else:
                chunks.append(m)
        return np.concatenate(chunks, axis=0)


def write_consolidated_peak_files(
    accumulators: List[FrameAccumulator],
    *,
    n_total_frames: int,
    start_frame: int,
    end_frame: int,
    nr_pixels: int,
    out_folder: str | Path,
    abs_frames: list[int] | None = None,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Write ``AllPeaks_PS{suffix}.bin`` and ``AllPeaks_PX{suffix}.bin``.

    ``accumulators[i]`` is the data for block-local position i. The absolute
    frame number for that position is ``abs_frames[i]`` if provided, else
    ``start_frame + i`` (contiguous mode).

    ``suffix="_sat"`` writes the saturated-region siblings
    (``AllPeaks_PS_sat.bin`` / ``AllPeaks_PX_sat.bin``) in the identical
    layout; readers of the main files never see them.
    """
    if abs_frames is None:
        abs_frames = list(range(start_frame, end_frame))
    assert len(abs_frames) == len(accumulators), (
        f"abs_frames ({len(abs_frames)}) must match accumulators ({len(accumulators)})"
    )
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    ps_path = out_folder / f"AllPeaks_PS{suffix}.bin"
    px_path = out_folder / f"AllPeaks_PX{suffix}.bin"

    # Per-frame peak counts (full length, zero outside this block)
    n_peaks_arr = np.zeros(n_total_frames, dtype=np.int32)
    for i, acc in enumerate(accumulators):
        f = abs_frames[i]
        if 0 <= f < n_total_frames:
            n_peaks_arr[f] = acc.n_peaks

    # ── AllPeaks_PS.bin ────────────────────────────────────────────
    ps_header_size = 4 + n_total_frames * 4 + n_total_frames * 8
    ps_offsets = np.zeros(n_total_frames, dtype=np.int64)
    data_off = ps_header_size
    for f in range(n_total_frames):
        ps_offsets[f] = data_off
        data_off += int(n_peaks_arr[f]) * N_PEAK_COLS * 8

    # Map absolute frame → block-local accumulator index (handles both
    # contiguous and interleaved sharding).
    abs_to_loc = {abs_frames[i]: i for i in range(len(abs_frames))}

    with open(ps_path, "wb") as f:
        f.write(np.int32(n_total_frames).tobytes())
        f.write(n_peaks_arr.tobytes())
        f.write(ps_offsets.tobytes())
        for v in range(n_total_frames):
            i = abs_to_loc.get(v, -1)
            if i >= 0:
                acc = accumulators[i]
                if acc.n_peaks > 0:
                    rows = acc.stacked_rows()
                    f.write(np.ascontiguousarray(rows, dtype=np.float64).tobytes())

    # ── AllPeaks_PX.bin ────────────────────────────────────────────
    px_header_size = 4 + 4 + n_total_frames * 4 + n_total_frames * 8
    px_offsets = np.zeros(n_total_frames, dtype=np.int64)
    data_off = px_header_size
    for f in range(n_total_frames):
        px_offsets[f] = data_off
        i = abs_to_loc.get(f, -1)
        if i >= 0:
            acc = accumulators[i]
            for n_px in acc.n_px_per_peak:
                data_off += 4 + n_px * 2 * 2

    with open(px_path, "wb") as f:
        f.write(np.int32(n_total_frames).tobytes())
        f.write(np.int32(nr_pixels).tobytes())
        f.write(n_peaks_arr.tobytes())
        f.write(px_offsets.tobytes())
        for v in range(n_total_frames):
            i = abs_to_loc.get(v, -1)
            if i >= 0:
                acc = accumulators[i]
                for pk_idx in range(len(acc.n_px_per_peak)):
                    n_px = acc.n_px_per_peak[pk_idx]
                    f.write(np.int32(n_px).tobytes())
                    yz = np.empty(n_px * 2, dtype=np.int16)
                    yz[0::2] = acc.pixel_y[pk_idx]
                    yz[1::2] = acc.pixel_z[pk_idx]
                    f.write(yz.tobytes())

    print(f"Wrote {ps_path} ({n_total_frames} frames)")
    print(f"Wrote {px_path} ({n_total_frames} frames)")

    # ── AllPeaks_PS_unc.bin (per-peak σ; sibling file, optional) ────
    # Same header layout as AllPeaks_PS.bin but ``N_UNC_COLS`` doubles per peak
    # rather than ``N_PEAK_COLS``. Skipped if no accumulator carries σ data.
    if any(acc.has_unc for acc in accumulators):
        unc_path = out_folder / "AllPeaks_PS_unc.bin"
        unc_header_size = 4 + n_total_frames * 4 + n_total_frames * 8
        unc_offsets = np.zeros(n_total_frames, dtype=np.int64)
        data_off = unc_header_size
        for f in range(n_total_frames):
            unc_offsets[f] = data_off
            data_off += int(n_peaks_arr[f]) * N_UNC_COLS * 8
        with open(unc_path, "wb") as f:
            f.write(np.int32(n_total_frames).tobytes())
            f.write(n_peaks_arr.tobytes())
            f.write(unc_offsets.tobytes())
            for v in range(n_total_frames):
                i = abs_to_loc.get(v, -1)
                if i >= 0:
                    acc = accumulators[i]
                    if acc.n_peaks > 0:
                        unc = acc.stacked_unc()
                        f.write(np.ascontiguousarray(unc, dtype=np.float64).tobytes())
        print(f"Wrote {unc_path} ({n_total_frames} frames, {N_UNC_COLS} σ-cols/peak)")

    # ── AllPeaks_PS_moment.bin (model-free moment σ + quality flag) ─
    # Same header layout as AllPeaks_PS.bin but ``N_MOMENT_COLS`` doubles per
    # peak. Per Modregger et al., J. Appl. Cryst. 58, 1653 (2025). Skipped
    # if no accumulator carries moment data (i.e. --moment-uncertainty off).
    if any(acc.has_moment for acc in accumulators):
        moment_path = out_folder / "AllPeaks_PS_moment.bin"
        moment_header_size = 4 + n_total_frames * 4 + n_total_frames * 8
        moment_offsets = np.zeros(n_total_frames, dtype=np.int64)
        data_off = moment_header_size
        for f in range(n_total_frames):
            moment_offsets[f] = data_off
            data_off += int(n_peaks_arr[f]) * N_MOMENT_COLS * 8
        with open(moment_path, "wb") as f:
            f.write(np.int32(n_total_frames).tobytes())
            f.write(n_peaks_arr.tobytes())
            f.write(moment_offsets.tobytes())
            for v in range(n_total_frames):
                i = abs_to_loc.get(v, -1)
                if i >= 0:
                    acc = accumulators[i]
                    if acc.n_peaks > 0:
                        mom = acc.stacked_moment()
                        f.write(np.ascontiguousarray(mom, dtype=np.float64).tobytes())
        print(f"Wrote {moment_path} ({n_total_frames} frames, {N_MOMENT_COLS} moment-cols/peak)")
    return ps_path, px_path


__all__ = ["FrameAccumulator", "write_consolidated_peak_files"]
