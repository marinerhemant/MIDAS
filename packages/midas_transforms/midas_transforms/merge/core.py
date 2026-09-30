"""merge_overlapping_peaks: drop-in replacement for ``MergeOverlappingPeaksAllZarr``.

Direct port of ``FF_HEDM/src/MergeOverlappingPeaksAllZarr.c:494-1060``.

Reads ``AllPeaks_PS.bin`` (the ``ConsolidatedPeakReader`` blob written by
``midas-peakfit`` / ``PeaksFittingOMPZarrRefactor``) and produces:

- ``Result_StartNr_<S>_EndNr_<E>.csv`` — 17 cols, in the order the C code
  writes (Radius and Eta are the LATEST match's values, not recomputed).
- ``MergeMap.csv`` — ``MergedSpotID FrameNr PeakID`` triples.

Algorithm (mirrors C exactly):

1. Initialise ``CurrentIDs`` from frame 0's filtered+sorted peaks.
2. For each subsequent frame:
   a. Read frame's peaks, drop ``IntegratedIntensity < 1``, sort by Eta
      with stable secondary key (original index) for tie-break determinism.
   b. Mutual-nearest match in (YCen, ZCen) within ``MarginOmegaOverlap``,
      filtered by ``|Radius_new - Radius_cur| <= MarginOmegaOverlap`` (ring guard).
   c. Matched: update ``CurrentIDs[i]`` (accumulate weighted Omega/Y/Z by
      IntInt, keep min/max omega, max IMax/SigmaR/SigmaEta/FitRMSE,
      sum NrPx/NrPxTot/RawSum, OR maskTouched, **replace Radius/Eta/Y/Z
      with the new peak's values**).
   d. Unmatched current: finalise immediately, write to OutFile + MergeMap.
   e. Unmatched new: become new ``CurrentIDs`` entries for next iteration.
3. End: write all remaining ``CurrentIDs``.

The 19-column ``CurrentIDs`` working buffer mapping is at
``MergeOverlappingPeaksAllZarr.c:639-678`` (init) and ``:806-833`` /
``:858-885`` (update); the 17-col output is written at ``:914-933`` and
``:1042-1057``.

GPU acceleration: per-frame mutual-nearest is a small dense problem
(~10⁴ peaks × ~10⁴ peaks). The torch port computes the pairwise distance
matrix via broadcast and resolves mutual-best in a fully vectorised pass.
The frame loop itself remains Python-driven (sequential by construction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import csv as csv_io
from ..io import zarr_io
from ..params import ZarrParams, read_zarr_params

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass
class MergeResult:
    """In-memory result of the merge stage."""

    peaks: torch.Tensor                            # (N, 18) — Result_*.csv layout (17 legacy + ReturnCode)
    merge_map: List[Tuple[int, int, int]] = field(default_factory=list)
    # Each tuple: (merged_spot_id, frame_nr, original_peak_id)
    # Saturated reflections merged among themselves ((M, 18), ReturnCode -2),
    # or None when peakfit recorded no saturation / predates recording it.
    saturated: Optional[np.ndarray] = None
    # Rows of ``peaks`` with ReturnCode -2 (contain / touch saturation).
    n_saturated_flagged: int = 0


# AllPeaks_PS.bin column layout (from PeaksFittingConsolidatedIO.h:31-44).
N_PEAK_COLS = 29
COL_SPOTID = 0
COL_II = 1
COL_OMEGA = 2
COL_YCEN = 3
COL_ZCEN = 4
COL_IMAX = 5
COL_RADIUS = 6
COL_ETA = 7
COL_SIGMAR = 8
COL_SIGMAETA = 9
COL_NRPX = 10
COL_NRPXTOT = 11
COL_MAXY = 13
COL_MAXZ = 14
COL_RETCODE = 18          # peakfit per-peak returnCode (postfit.py:85)
COL_RAWSUM = 26
COL_MASKTOUCHED = 27
COL_FITRMSE = 28


def _read_sort_filter_frame(
    peaks: np.ndarray, use_maxima_positions: bool,
) -> np.ndarray:
    """Mirror ``ReadSortFiles`` from MergeOverlappingPeaksAllZarr.c:311-355.

    - Optionally swap (YCen, ZCen) for (maxY, maxZ).
    - Drop peaks with ``IntegratedIntensity < 1``.
    - qsort by Eta (stable; secondary key = original index for tie-break).
    """
    arr, _ = _read_sort_filter_frame_with_pixels(
        peaks, pixels=None, use_maxima_positions=use_maxima_positions,
    )
    return arr


def _read_sort_filter_frame_with_pixels(
    peaks: np.ndarray,
    pixels: Optional[List[np.ndarray]],
    use_maxima_positions: bool,
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """Mirror ``ReadSortFiles`` and apply the same permutation to pixel data.

    The C ``MergeOverlappingPeaksAllZarr.c:741-755`` reads pixel data into
    ``newPixels`` in PS-file order, then accesses ``newPixels[j]`` with j
    from the Eta-sorted+filtered ``NewIDs[]`` index space — an indexing
    bug that just happens to be benign on sparse-peak datasets. We fix
    it here by tracking the same sort+filter permutation across both
    arrays so ``pixels_sorted[j]`` actually corresponds to ``peaks_sorted[j]``.
    """
    if peaks.size == 0:
        return peaks, ([] if pixels is not None else None)
    arr = peaks.copy()
    if use_maxima_positions:
        arr[:, COL_YCEN] = arr[:, COL_MAXY]
        arr[:, COL_ZCEN] = arr[:, COL_MAXZ]
    keep = arr[:, COL_II] >= 1.0
    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        return arr[keep], ([] if pixels is not None else None)
    arr = arr[keep_idx]
    pix_kept = (
        [pixels[i] for i in keep_idx]
        if pixels is not None
        else None
    )
    # Stable sort by Eta with original-index tie-break.
    order = np.lexsort((np.arange(arr.shape[0]), arr[:, COL_ETA]))
    arr = arr[order]
    if pix_kept is not None:
        pix_kept = [pix_kept[i] for i in order]
    return arr, pix_kept


def _seed_current_from_frame(
    frame_peaks: np.ndarray, frame_nr: int,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """Initialise ``CurrentIDs`` (20 cols: 19 C-layout + ReturnCode) from a
    frame's filtered peaks.

    Mirrors ``MergeOverlappingPeaksAllZarr.c:644-679``; col 19 additionally
    carries the peakfit per-peak returnCode (N2 — previously dropped here,
    making fit success/failure unrecoverable downstream).
    """
    n = frame_peaks.shape[0]
    cur = np.zeros((n, 20), dtype=np.float64)
    if n == 0:
        return cur, []
    ii = frame_peaks[:, COL_II]
    cur[:, 0] = frame_peaks[:, COL_SPOTID]
    cur[:, 1] = ii
    cur[:, 2] = frame_peaks[:, COL_OMEGA] * ii
    cur[:, 3] = frame_peaks[:, COL_YCEN] * ii
    cur[:, 4] = frame_peaks[:, COL_ZCEN] * ii
    cur[:, 5] = frame_peaks[:, COL_IMAX]
    cur[:, 6] = frame_peaks[:, COL_RADIUS]
    cur[:, 7] = frame_peaks[:, COL_ETA]
    cur[:, 8] = frame_peaks[:, COL_YCEN]
    cur[:, 9] = frame_peaks[:, COL_ZCEN]
    cur[:, 10] = frame_peaks[:, COL_OMEGA]
    cur[:, 11] = frame_peaks[:, COL_OMEGA]
    cur[:, 12] = frame_peaks[:, COL_SIGMAR]
    cur[:, 13] = frame_peaks[:, COL_SIGMAETA]
    cur[:, 14] = frame_peaks[:, COL_NRPX]
    cur[:, 15] = frame_peaks[:, COL_NRPXTOT]
    cur[:, 16] = frame_peaks[:, COL_RAWSUM]
    cur[:, 17] = frame_peaks[:, COL_MASKTOUCHED]
    cur[:, 18] = frame_peaks[:, COL_FITRMSE]
    cur[:, 19] = frame_peaks[:, COL_RETCODE]
    cons = [[(frame_nr, int(frame_peaks[i, COL_SPOTID]))] for i in range(n)]
    return cur, cons


def _mutual_nearest(
    cur: np.ndarray, new: np.ndarray, margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy mutual-nearest matching, mirroring the C semantics exactly.

    The C algorithm (``MergeOverlappingPeaksAllZarr.c:835-906``) iterates
    current peaks in cur-index order. For each ``i``:

      1. Among UNCLAIMED new peaks ``j`` within (Y, Z) distance ``margin``
         and ``|R_cur - R_new| <= margin``, find the closest one
         (distance ``minLen``).
      2. Reverse check: among UNCLAIMED current peaks ``k`` within
         ``minLen`` of new ``j`` and same ring, find the closest. If
         ``k != i``, abandon this match (another cur is closer).
      3. Otherwise: pair (i, j); mark both as taken; move on.

    Because peaks claimed earlier are removed from later searches, this is
    **greedy** (not globally optimal) and **order-sensitive**. We replicate
    the order exactly: cur in index order (which matches the Eta-sorted
    seed order from the previous frame).

    Returns ``(best_for_cur, has_match_cur)``. The matcher is sequential
    in Python; the inner per-cur lookup is vectorised over new peaks.
    """
    nC = cur.shape[0]
    nN = new.shape[0]
    out_best = np.full(nC, -1, dtype=np.int64)
    out_has = np.zeros(nC, dtype=bool)
    if nC == 0 or nN == 0:
        return out_best, out_has

    cur_y = cur[:, 8]
    cur_z = cur[:, 9]
    cur_r = cur[:, 6]
    new_y = new[:, COL_YCEN]
    new_z = new[:, COL_ZCEN]
    new_r = new[:, COL_RADIUS]

    new_taken = np.zeros(nN, dtype=bool)
    cur_taken = np.zeros(nC, dtype=bool)

    # The C ``shash_find_nearest`` (MergeOverlappingPeaksAllZarr.c:459-491)
    # initialises ``best = radius`` so peaks outside the search radius are
    # excluded — the bucketing is a pure performance optimisation. We
    # replicate with a strict ``dist < radius`` filter.

    for i in range(nC):
        # Step 1: forward — closest unclaimed new peak with d < margin.
        free_new = ~new_taken
        if not free_new.any():
            break
        dy = cur_y[i] - new_y
        dz = cur_z[i] - new_z
        dist = np.sqrt(dy * dy + dz * dz)
        ring_ok = np.abs(cur_r[i] - new_r) <= margin
        candidate = free_new & ring_ok & (dist < margin)
        if not candidate.any():
            continue
        d = np.where(candidate, dist, np.inf)
        j = int(np.argmin(d))
        min_len = float(d[j])

        # Step 2: reverse — closest unclaimed cur to j with d < min_len.
        # If this returns some k != i, abandon (another cur is closer).
        # Note that ``cur_taken`` does NOT yet include i — it's still
        # in the pool, so we don't filter it out (and i↔j distance is
        # exactly min_len, NOT < min_len, so i can't be the reverse
        # winner; k != i with k != -1 means we abandon).
        free_cur = ~cur_taken
        rdy = cur_y - new_y[j]
        rdz = cur_z - new_z[j]
        rdist = np.sqrt(rdy * rdy + rdz * rdz)
        rring_ok = np.abs(cur_r - new_r[j]) <= margin
        rcand = free_cur & rring_ok & (rdist < min_len)
        if rcand.any():
            k = int(np.argmin(np.where(rcand, rdist, np.inf)))
            if k != i:
                continue

        out_best[i] = j
        out_has[i] = True
        new_taken[j] = True
        cur_taken[i] = True
    return out_best, out_has


def _pixel_overlap_match(
    cur_pixels: List[np.ndarray],
    new_pixels: List[np.ndarray],
    nr_pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame pixel-overlap matching, mirroring the C semantics in
    ``MergeOverlappingPeaksAllZarr.c:756-810``.

    Algorithm:
      1. Build a label map (``nr_pixels x nr_pixels`` int32, initialised to 0)
         by painting each cur peak's pixel list with its 1-based index.
      2. Forward pass: for each new peak j, look up its pixel list in the
         label map and tally counts per non-zero label. Best label
         (highest count) is the cur peak it most overlaps with.
      3. Reverse pass: for each cur peak i, scan all new peaks claiming
         cur i and pick the one with the highest overlap count. That's
         the match.

    Returns ``(best_for_cur, has_match_cur)``: parallel to ``_mutual_nearest``.
    """
    n_cur = len(cur_pixels)
    n_new = len(new_pixels)
    out_best = np.full(n_cur, -1, dtype=np.int64)
    out_has = np.zeros(n_cur, dtype=bool)
    if n_cur == 0 or n_new == 0:
        return out_best, out_has

    # Build label map. Use raveled (z * nr + y) to match the C
    # ``BuildLabelMap`` indexing: ``idx = (int)pp[pk].y[i] * nrPixels + (int)pp[pk].z[i]``
    # That's actually y * nr_pixels + z (y is the row coordinate in C's
    # convention here). Match it exactly.
    label_map = np.zeros(nr_pixels * nr_pixels, dtype=np.int32)
    painted_indices: List[np.ndarray] = []
    for k, pix in enumerate(cur_pixels):
        if pix is None or pix.size == 0:
            painted_indices.append(np.empty(0, dtype=np.int64))
            continue
        # Bounds-check: drop pixels outside the detector (shouldn't happen,
        # but the C code can write garbage if y/z are oob).
        y = pix[:, 0].astype(np.int64)
        z = pix[:, 1].astype(np.int64)
        in_bounds = (y >= 0) & (y < nr_pixels) & (z >= 0) & (z < nr_pixels)
        idx = (y[in_bounds] * nr_pixels + z[in_bounds])
        label_map[idx] = k + 1   # 1-based
        painted_indices.append(idx)

    # Forward pass.
    forward_label = np.full(n_new, 0, dtype=np.int64)   # 0 = no match
    forward_count = np.zeros(n_new, dtype=np.int64)
    for j, pix in enumerate(new_pixels):
        if pix is None or pix.size == 0:
            continue
        y = pix[:, 0].astype(np.int64)
        z = pix[:, 1].astype(np.int64)
        in_bounds = (y >= 0) & (y < nr_pixels) & (z >= 0) & (z < nr_pixels)
        if not in_bounds.any():
            continue
        idx = (y[in_bounds] * nr_pixels + z[in_bounds])
        labels = label_map[idx]
        # Count non-zero labels and pick the most frequent.
        nz = labels[labels > 0]
        if nz.size == 0:
            continue
        # ``np.bincount`` over the small set of distinct labels.
        unique, counts = np.unique(nz, return_counts=True)
        best_pos = int(np.argmax(counts))
        forward_label[j] = int(unique[best_pos])
        forward_count[j] = int(counts[best_pos])

    # Reverse pass: for each cur i (1-based label = i+1), find new peak with
    # max forward_count among those whose forward_label points at cur i.
    for i in range(n_cur):
        label_i = i + 1
        candidates = np.flatnonzero(forward_label == label_i)
        if candidates.size == 0:
            continue
        best_j = int(candidates[np.argmax(forward_count[candidates])])
        out_best[i] = best_j
        out_has[i] = True

    return out_best, out_has


def _accumulate_match(cur_row: np.ndarray, new_row: np.ndarray) -> None:
    """Update one ``CurrentIDs`` row in-place with one matched new peak.

    Mirrors ``MergeOverlappingPeaksAllZarr.c:806-833`` (in distance branch).
    """
    ii = new_row[COL_II]
    cur_row[1] += ii
    cur_row[2] += new_row[COL_OMEGA] * ii
    cur_row[3] += new_row[COL_YCEN] * ii
    cur_row[4] += new_row[COL_ZCEN] * ii
    if cur_row[5] < new_row[COL_IMAX]:
        cur_row[5] = new_row[COL_IMAX]
    cur_row[6] = new_row[COL_RADIUS]   # latest Radius
    cur_row[7] = new_row[COL_ETA]      # latest Eta
    cur_row[8] = new_row[COL_YCEN]     # latest YCen
    cur_row[9] = new_row[COL_ZCEN]     # latest ZCen
    if cur_row[10] > new_row[COL_OMEGA]:
        cur_row[10] = new_row[COL_OMEGA]   # MinOmega
    if cur_row[11] < new_row[COL_OMEGA]:
        cur_row[11] = new_row[COL_OMEGA]   # MaxOmega
    if cur_row[12] < new_row[COL_SIGMAR]:
        cur_row[12] = new_row[COL_SIGMAR]
    if cur_row[13] < new_row[COL_SIGMAETA]:
        cur_row[13] = new_row[COL_SIGMAETA]
    cur_row[14] += new_row[COL_NRPX]
    cur_row[15] += new_row[COL_NRPXTOT]
    cur_row[16] += new_row[COL_RAWSUM]
    if new_row[COL_MASKTOUCHED] > 0:
        cur_row[17] = 1.0
    if new_row[COL_FITRMSE] > cur_row[18]:
        cur_row[18] = new_row[COL_FITRMSE]
    # ReturnCode: sticky-first-nonzero — a merged spot records the first
    # failing constituent fit (0 stays "all constituents fit OK").
    if cur_row[19] == 0.0:
        cur_row[19] = new_row[COL_RETCODE]


def _finalise_row(
    cur_row: np.ndarray, spot_id: int,
) -> np.ndarray:
    """Convert one ``CurrentIDs`` row into one 17-col Result.csv row.

    Column order matches ``MergeOverlappingPeaksAllZarr.c:914-924`` /
    ``:1042-1051``::

        [0]  SpotIDNr
        [1]  IntegratedIntensity
        [2]  Omega = wOmega / IntInt
        [3]  YCen  = wYCen  / IntInt
        [4]  ZCen  = wZCen  / IntInt
        [5]  IMax
        [6]  MinOme
        [7]  MaxOme
        [8]  SigmaR
        [9]  SigmaEta
        [10] NrPx
        [11] NrPxTot
        [12] Radius (LATEST)
        [13] Eta    (LATEST)
        [14] RawSumIntensity
        [15] maskTouched
        [16] FitRMSE
        [17] ReturnCode  (appended; N2 — sticky-first-nonzero over merged
             constituents; legacy 17-col files simply lack it)
    """
    out = np.zeros(18, dtype=np.float64)
    intInt = cur_row[1] if cur_row[1] != 0 else 1.0
    out[0] = spot_id
    out[1] = cur_row[1]
    out[2] = cur_row[2] / intInt
    out[3] = cur_row[3] / intInt
    out[4] = cur_row[4] / intInt
    out[5] = cur_row[5]
    out[6] = cur_row[10]   # MinOme
    out[7] = cur_row[11]   # MaxOme
    out[8] = cur_row[12]
    out[9] = cur_row[13]
    out[10] = cur_row[14]
    out[11] = cur_row[15]
    out[12] = cur_row[6]   # Radius (latest)
    out[13] = cur_row[7]   # Eta (latest)
    out[14] = cur_row[16]
    out[15] = cur_row[17]
    out[16] = cur_row[18]
    out[17] = cur_row[19]
    return out


def _merge_core(
    frames: List[np.ndarray],
    *,
    overlap_length: float,
    use_maxima_positions: bool = False,
    skip_frame: int = 0,
    pixel_frames: Optional[List[List[np.ndarray]]] = None,
    nr_pixels: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[List[Tuple[int, int]]], List[np.ndarray]]:
    """Frame-by-frame merge. Returns ``(Result_csv_array, merge_map,
    constituents_per_row, processed_frames)``; the last two let the
    saturation pass look up every merged spot's per-frame members.

    If ``pixel_frames`` is provided (and ``nr_pixels > 0``), pixel-overlap
    matching is used per frame; otherwise centroid-distance.

    ``pixel_frames`` must be in PS-file order (one list per frame, one
    ``(n_px, 2)`` int16 array per peak); this routine applies the same
    Eta-sort + II>=1 permutation as ``_read_sort_filter_frame`` so that
    ``proc[fi][j]`` and ``pix[fi][j]`` correspond.
    """
    if not frames:
        return np.empty((0, 18), dtype=np.float64), [], [], []

    # The C code does `EndNr -= skipFrame` then iterates [StartNr, EndNr];
    # we mirror by truncating the frame list.
    if skip_frame > 0:
        frames = frames[: max(len(frames) - skip_frame, 0)]
        if pixel_frames is not None:
            pixel_frames = pixel_frames[: max(len(pixel_frames) - skip_frame, 0)]
    if not frames:
        return np.empty((0, 18), dtype=np.float64), [], [], []

    use_pixel_overlap = pixel_frames is not None and nr_pixels > 0

    # Pre-process every frame: filter II<1, sort by Eta. Apply the same
    # permutation to pixel data so peaks and pixels stay aligned.
    proc: List[np.ndarray] = []
    pix: List[Optional[List[np.ndarray]]] = []
    for fi, f in enumerate(frames):
        pf = pixel_frames[fi] if use_pixel_overlap else None
        peaks_s, pix_s = _read_sort_filter_frame_with_pixels(
            f, pf, use_maxima_positions,
        )
        proc.append(peaks_s)
        pix.append(pix_s)

    # Find first non-empty frame.
    first = 0
    while first < len(proc) and proc[first].shape[0] == 0:
        first += 1
    if first >= len(proc):
        return np.empty((0, 18), dtype=np.float64), [], [], proc

    cur, constituents = _seed_current_from_frame(proc[first], frame_nr=first + 1)
    cur_pixels: List[np.ndarray] = list(pix[first]) if use_pixel_overlap else []
    finalised: List[np.ndarray] = []
    final_cons: List[List[Tuple[int, int]]] = []
    merge_map: List[Tuple[int, int, int]] = []
    spot_id_nr = 1

    # Main frame loop. Mirrors MergeOverlappingPeaksAllZarr.c:735-1029.
    for fi in range(first + 1, len(proc)):
        new_peaks = proc[fi]
        new_pix = pix[fi] if use_pixel_overlap else None
        frame_nr = fi + 1   # 1-based, like the C code

        if new_peaks.shape[0] == 0:
            # No new peaks this frame: every current row finalises.
            for i in range(cur.shape[0]):
                finalised.append(_finalise_row(cur[i], spot_id_nr))
                final_cons.append(constituents[i])
                for (fn, pid) in constituents[i]:
                    merge_map.append((spot_id_nr, fn, pid))
                spot_id_nr += 1
            cur = np.zeros((0, 20), dtype=np.float64)
            constituents = []
            cur_pixels = []
            continue

        # Choose matcher: pixel-overlap-first when pixel data was supplied
        # (only for UsePixelOverlap 2 via merge_overlapping_peaks) and both
        # sides have pixels, else centroid distance.
        if use_pixel_overlap and len(cur_pixels) > 0 and new_pix is not None and len(new_pix) > 0:
            best_for_cur, matched = _pixel_overlap_match(
                cur_pixels, new_pix, nr_pixels,
            )
        else:
            best_for_cur, matched = _mutual_nearest(cur, new_peaks, overlap_length)

        # Update matched current rows (cur-index order, matches C).
        new_taken = np.zeros(new_peaks.shape[0], dtype=bool)
        for i in np.flatnonzero(matched):
            j = int(best_for_cur[i])
            new_taken[j] = True
            _accumulate_match(cur[i], new_peaks[j])
            constituents[i].append((frame_nr, int(new_peaks[j, COL_SPOTID])))

        # Finalise current rows that did NOT match this frame
        # (MergeOverlappingPeaksAllZarr.c:916-947).
        unmatched_cur = ~matched
        if unmatched_cur.any():
            for i in np.flatnonzero(unmatched_cur):
                finalised.append(_finalise_row(cur[i], spot_id_nr))
                final_cons.append(constituents[i])
                for (fn, pid) in constituents[i]:
                    merge_map.append((spot_id_nr, fn, pid))
                spot_id_nr += 1

        # Build next-iteration CurrentIDs:
        #   matched rows (continuing) followed by unmatched new peaks.
        # Matches C order: line 949-998 (matched cur first), then
        # 1000-1029 (unmatched new appended).
        keep_cur = np.flatnonzero(matched)
        new_cur = cur[keep_cur].copy()
        new_cons = [constituents[i] for i in keep_cur]
        # Pixel data tracking: for matched cur, pixels become the new peak's
        # pixels (since cur's location was just updated to new's). For
        # unmatched new, pixels stay as their own.
        new_cur_pixels: List[np.ndarray] = []
        if use_pixel_overlap and new_pix is not None:
            for i in keep_cur:
                new_cur_pixels.append(new_pix[int(best_for_cur[i])])

        unmatched_new = np.flatnonzero(~new_taken)
        if unmatched_new.size > 0:
            seed_arr, seed_cons = _seed_current_from_frame(
                new_peaks[unmatched_new], frame_nr=frame_nr,
            )
            new_cur = np.concatenate([new_cur, seed_arr], axis=0)
            new_cons.extend(seed_cons)
            if use_pixel_overlap and new_pix is not None:
                for j in unmatched_new:
                    new_cur_pixels.append(new_pix[int(j)])

        cur = new_cur
        constituents = new_cons
        cur_pixels = new_cur_pixels

    # Final flush: every remaining cur row gets a SpotIDNr and is written.
    for i in range(cur.shape[0]):
        finalised.append(_finalise_row(cur[i], spot_id_nr))
        final_cons.append(constituents[i])
        for (fn, pid) in constituents[i]:
            merge_map.append((spot_id_nr, fn, pid))
        spot_id_nr += 1

    if not finalised:
        return np.empty((0, 18), dtype=np.float64), [], [], proc
    return np.stack(finalised, axis=0), merge_map, final_cons, proc


# ---------------------------------------------------------------------------
# Saturated regions
# ---------------------------------------------------------------------------
#
# midas-peakfit does not fit a region with a pixel over IntSat. It used to drop
# it silently; it now writes it (one intensity-weighted-centroid row, returnCode
# -2) to the sibling ``AllPeaks_PS_sat.bin``, leaving ``AllPeaks_PS.bin``
# byte-identical. What went wrong before: the faint edges of a saturated
# multi-frame reflection, in the unsaturated frames around its core, survived
# as tiny UNFLAGGED spots and grain matching accepted them as the reflection
# (AlON 1-ID 2026-09-22: 4.5 % of large-grain spots, ~30x too faint, and alone
# a -0.4 slope of log(I/LP) vs log L).
#
# "Touches": a spot constituent in frame f lies within ``r_sat + OverlapLength``
# px of a saturated region's centroid in frame f-1, f or f+1, with
# ``r_sat = sqrt(NrPx / pi)`` the region's equivalent-disc radius.

SATURATED_RETURN_CODE = -2.0      # == midas_peakfit.seeds.SATURATED_RETURN_CODE


def _unique_sat_ids(frames: List[np.ndarray], sat_frames: List[np.ndarray]) -> List[np.ndarray]:
    """Copy of ``sat_frames`` with returnCode forced to -2 and SpotIDs that do
    not collide with the same frame's fitted peaks (peakfit already writes them
    that way; this keeps hand-built / in-memory inputs honest)."""
    out = []
    for fi, s in enumerate(sat_frames):
        s = np.array(s, dtype=np.float64, copy=True).reshape(-1, N_PEAK_COLS)
        if s.shape[0]:
            s[:, COL_RETCODE] = SATURATED_RETURN_CODE
            f = frames[fi] if fi < len(frames) else np.zeros((0, N_PEAK_COLS))
            if f.shape[0] and np.isin(s[:, COL_SPOTID], f[:, COL_SPOTID]).any():
                s[:, COL_SPOTID] = f[:, COL_SPOTID].max() + 1 + np.arange(s.shape[0])
        out.append(s)
    return out


def _sat_by_frame(sat_proc: List[np.ndarray]):
    """Per-frame (Y, Z, r_sat) arrays of saturated centroids (0-based index)."""
    return [
        (s[:, COL_YCEN], s[:, COL_ZCEN], np.sqrt(np.maximum(s[:, COL_NRPX], 1.0) / math.pi))
        for s in sat_proc
    ]


def _row_lookup(proc: List[np.ndarray]):
    """``lookup[fi][spot_id] -> row`` over the processed frames."""
    return [{int(r[COL_SPOTID]): r for r in f} for f in proc]


def _touching_sat(cons, lookup, sat_idx, margin: float) -> List[Tuple[int, int]]:
    """Saturated regions ``(frame_index, k)`` touched by any constituent."""
    hits = []
    n = len(sat_idx)
    for (frame_nr, pid) in cons:
        fi = frame_nr - 1
        row = lookup[fi].get(int(pid))
        if row is None:
            continue
        y, z = row[COL_YCEN], row[COL_ZCEN]
        for fj in (fi - 1, fi, fi + 1):
            if fj < 0 or fj >= n:
                continue
            sy, sz, sr = sat_idx[fj]
            if sy.size == 0:
                continue
            d = np.hypot(sy - y, sz - z)
            for k in np.flatnonzero(d <= sr + margin):
                hits.append((fj, int(k)))
    return hits


def _flag_touching(out, cons_rows, proc, sat_proc, margin) -> np.ndarray:
    """Flag mode: set ReturnCode = -2 on every merged spot touching saturation."""
    out = out.copy()
    lookup = _row_lookup(proc)
    sat_idx = _sat_by_frame(sat_proc)
    for i, cons in enumerate(cons_rows):
        if _touching_sat(cons, lookup, sat_idx, margin):
            out[i, 17] = SATURATED_RETURN_CODE
    return out


def _collapse_saturated(out, merge_map, cons_rows, proc, margin):
    """Include mode: fold every merged spot that contains or touches a
    saturated region into ONE flagged spot per connected saturated blob.

    ``proc`` holds fitted and saturated rows together (saturated ones carry
    returnCode -2). Members are re-accumulated from their per-frame rows in
    frame order (same arithmetic as the merge itself), then SpotIDs and the
    merge map are renumbered 1..N in first-member order.
    """
    n = out.shape[0]
    if n == 0:
        return out, merge_map
    lookup = _row_lookup(proc)
    sat_proc = [f[f[:, COL_RETCODE] == SATURATED_RETURN_CODE] if f.shape[0] else f for f in proc]
    sat_idx = _sat_by_frame(sat_proc)
    # (frame_index, k) of each saturated region -> the merged spot holding it
    owner = {}
    for i, cons in enumerate(cons_rows):
        for (frame_nr, pid) in cons:
            fi = frame_nr - 1
            row = lookup[fi].get(int(pid))
            if row is not None and row[COL_RETCODE] == SATURATED_RETURN_CODE:
                k = int(np.flatnonzero(sat_proc[fi][:, COL_SPOTID] == pid)[0])
                owner[(fi, k)] = i
    if not owner:
        return out, merge_map

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, cons in enumerate(cons_rows):
        for hit in _touching_sat(cons, lookup, sat_idx, margin):
            j = owner.get(hit)
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    sat_members = set(owner.values())

    rows, new_map = [], []
    for root in sorted(groups):
        members = groups[root]
        if not any(m in sat_members for m in members):
            assert len(members) == 1
            cons = cons_rows[members[0]]
            r = out[members[0]].copy()
        else:
            cons = sorted((c for m in members for c in cons_rows[m]), key=lambda c: (c[0], c[1]))
            first = lookup[cons[0][0] - 1][int(cons[0][1])]
            cur, _ = _seed_current_from_frame(first[None, :], frame_nr=cons[0][0])
            for (frame_nr, pid) in cons[1:]:
                _accumulate_match(cur[0], lookup[frame_nr - 1][int(pid)])
            r = _finalise_row(cur[0], 0)
            r[17] = SATURATED_RETURN_CODE
        sid = len(rows) + 1
        r[0] = sid
        rows.append(r)
        new_map.extend((sid, fn, pid) for (fn, pid) in cons)
    return np.stack(rows, axis=0), new_map


def _merge_frames(
    frames: List[np.ndarray],
    *,
    overlap_length: float,
    use_maxima_positions: bool = False,
    skip_frame: int = 0,
    pixel_frames: Optional[List[List[np.ndarray]]] = None,
    nr_pixels: int = 0,
    sat_frames: Optional[List[np.ndarray]] = None,
    sat_pixel_frames: Optional[List[List[np.ndarray]]] = None,
    include_saturated: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Frame-by-frame merge. Returns (Result_csv_array, merge_map).

    Without ``sat_frames`` (or with no saturated rows in them) this is exactly
    the legacy C-parity merge.

    With saturated rows (``AllPeaks_PS_sat.bin``):

    * ``include_saturated=False`` (default, ``IncludeSaturatedSpots 0``): the
      merge runs on the fitted peaks ONLY, so every row, position and SpotID is
      identical to before -- indexing and refinement see exactly the same
      spots. Every merged spot that touches a saturated region gets
      ``ReturnCode = -2`` so intensity consumers can drop it. The saturated
      regions themselves stay out (see :func:`merge_saturated_only`).
    * ``include_saturated=True`` (``IncludeSaturatedSpots 1``): saturated
      regions join the merge and everything touching them is folded into one
      flagged spot per saturated reflection, positioned at the II-weighted
      mean. Spot list, IDs and hence grains change.
    """
    has_sat = sat_frames is not None and any(np.size(s) for s in sat_frames)
    if not has_sat:
        out, mm, _, _ = _merge_core(
            frames, overlap_length=overlap_length,
            use_maxima_positions=use_maxima_positions, skip_frame=skip_frame,
            pixel_frames=pixel_frames, nr_pixels=nr_pixels)
        return out, mm

    n = max(len(frames), len(sat_frames))
    frames = list(frames) + [np.zeros((0, N_PEAK_COLS))] * (n - len(frames))
    sat_frames = _unique_sat_ids(frames, list(sat_frames) + [np.zeros((0, N_PEAK_COLS))] * (n - len(sat_frames)))

    if not include_saturated:
        out, mm, cons, proc = _merge_core(
            frames, overlap_length=overlap_length,
            use_maxima_positions=use_maxima_positions, skip_frame=skip_frame,
            pixel_frames=pixel_frames, nr_pixels=nr_pixels)
        if out.shape[0] == 0:
            return out, mm
        sat_kept = sat_frames[: max(len(sat_frames) - skip_frame, 0)] if skip_frame > 0 else sat_frames
        sat_proc = [_read_sort_filter_frame(s, use_maxima_positions) for s in sat_kept]
        return _flag_touching(out, cons, proc, sat_proc, overlap_length), mm

    combined = [np.concatenate([f.reshape(-1, N_PEAK_COLS), s], axis=0)
                for f, s in zip(frames, sat_frames)]
    comb_pix = None
    if pixel_frames is not None:
        sp = sat_pixel_frames if sat_pixel_frames is not None else [[] for _ in sat_frames]
        comb_pix = []
        for fi in range(n):
            base = list(pixel_frames[fi]) if fi < len(pixel_frames) else []
            extra = list(sp[fi]) if fi < len(sp) else []
            extra += [np.zeros((0, 2), dtype=np.int16)] * (sat_frames[fi].shape[0] - len(extra))
            comb_pix.append(base + extra)
    out, mm, cons, proc = _merge_core(
        combined, overlap_length=overlap_length,
        use_maxima_positions=use_maxima_positions, skip_frame=skip_frame,
        pixel_frames=comb_pix, nr_pixels=nr_pixels)
    return _collapse_saturated(out, mm, cons, proc, overlap_length)


def merge_saturated_only(
    sat_frames: List[np.ndarray],
    *,
    overlap_length: float,
    use_maxima_positions: bool = False,
    skip_frame: int = 0,
) -> np.ndarray:
    """Merge the saturated regions among themselves (one row per saturated
    reflection, ReturnCode -2). Diagnostic output, ``SaturatedSpots.csv``;
    never enters indexing unless ``IncludeSaturatedSpots 1``."""
    empty = [np.zeros((0, N_PEAK_COLS))] * len(sat_frames)
    out, _ = _merge_frames(
        empty, overlap_length=overlap_length,
        use_maxima_positions=use_maxima_positions, skip_frame=skip_frame,
        sat_frames=sat_frames, include_saturated=True)
    return out


def _write_result_csv_c_format(path: Path, data: np.ndarray) -> None:
    """Write Result_*.csv in the exact format the C MergeOverlappingPeaksAllZarr
    emits: SpotID as ``%d``, all other cols as ``%lf`` (six decimals)."""
    header = csv_io.RESULT_HEADER
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in data:
            f.write(f"{int(row[0])}")
            for v in row[1:]:
                f.write(f" {v:.6f}")
            f.write("\n")


def _write_merge_map_csv(path: Path, rows: List[Tuple[int, int, int]]) -> None:
    """Write MergeMap.csv in the C format (tab-separated, leading '%' header)."""
    with open(path, "w") as f:
        f.write("%MergedSpotID\tFrameNr\tPeakID\n")
        for (sid, fn, pid) in rows:
            f.write(f"{sid}\t{fn}\t{pid}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_overlapping_peaks(
    zarr_path: Optional[Union[str, Path]] = None,
    *,
    allpeaks_ps_bin: Optional[Union[str, Path]] = None,
    allpeaks_px_bin: Optional[Union[str, Path]] = None,
    result_folder: Union[str, Path] = ".",
    overlap_length: Optional[float] = None,
    skip_frame: int = 0,
    use_pixel_overlap: Optional[Union[bool, int]] = None,
    use_maxima_positions: bool = False,
    nr_pixels: Optional[int] = None,
    start_nr: int = 1,
    end_nr: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
    frames: Optional[List[np.ndarray]] = None,
    pixel_frames: Optional[List[List[np.ndarray]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
    include_saturated: Optional[bool] = None,
    sat_frames: Optional[List[np.ndarray]] = None,
    sat_pixel_frames: Optional[List[List[np.ndarray]]] = None,
) -> MergeResult:
    """Drop-in replacement for ``MergeOverlappingPeaksAllZarr``.

    Saturated regions: if ``AllPeaks_PS_sat.bin`` sits next to
    ``AllPeaks_PS.bin`` (or ``sat_frames=`` is given), merged spots that
    contain or touch a saturated region get ``ReturnCode = -2``; the
    saturated reflections themselves are written to ``SaturatedSpots.csv``.
    ``include_saturated`` (``IncludeSaturatedSpots``, default 0) decides
    whether they also enter ``Result_*.csv`` and hence indexing; see
    :func:`_merge_frames`.

    Inputs are typically read from ``<result_folder>/Temp/AllPeaks_PS.bin``
    and (in pixel-overlap mode) ``<result_folder>/Temp/AllPeaks_PX.bin``,
    the ``ConsolidatedPeakReader`` / ``ConsolidatedPixelReader`` blobs
    written by midas-peakfit / PeaksFittingOMPZarrRefactor. Override paths
    via ``allpeaks_ps_bin=`` / ``allpeaks_px_bin=`` / ``frames=`` /
    ``pixel_frames=`` for in-memory or relocated inputs.

    ``use_pixel_overlap`` / ``UsePixelOverlap`` selects the frame-to-frame
    matcher:

    * ``0`` -- centroid mutual-nearest within ``OverlapLength`` (the C
      default).
    * ``1`` -- the C ``UsePixelOverlap 1`` output. On the only C golden we
      have (``tests/data/c_goldens_px``, generated with UsePixelOverlap=1)
      the C Result.csv and merge groups are identical to centroid matching:
      every one of the 29049 rows, token for token. So ``1`` runs the
      centroid matcher and does not need ``AllPeaks_PX.bin``. (Issue #13:
      before ``UsePixelOverlap`` was read from the archive this mode was
      never reached; once it was, overlap-first matching produced 28751
      rows against C's 29049.)
    * ``2`` -- opt-in pixel-overlap-FIRST matching (:func:`_pixel_overlap_match`):
      each new peak goes to the current spot it shares most pixels with,
      with no centroid-distance or ring gate. Pixel lists are kept aligned
      with the Eta-sorted, II>=1-filtered peaks. Not C-compatible: on the
      golden dataset it merges 297 extra consecutive-frame pairs whose
      centroids are 2.0-12.8 px apart, 56 % of them with ``|dR| > 2 SigmaR``,
      and in 82 % of them BOTH constituents carry a full single-spot
      intensity, i.e. they are two reflections rather than one split across
      frames. Use it only where overlap-first matching has been validated.
        """
    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    # Load Zarr params if a Zarr path was provided (the C code reads
    # OverlapLength / SkipFrame / UseMaximaPositions / UsePixelOverlap
    # from there).
    if zarr_path is not None:
        zp = read_zarr_params(zarr_path)
        if overlap_length is None:
            overlap_length = zp.OverlapLength
        if not skip_frame:
            skip_frame = zp.SkipFrame
        if not use_maxima_positions:
            use_maxima_positions = bool(zp.UseMaximaPositions)
        if nr_pixels is None:
            nr_pixels = zp.NrPixels
        if use_pixel_overlap is None:
            use_pixel_overlap = int(zp.UsePixelOverlap)
    if overlap_length is None:
        # C default: ``MarginOmegaOverlap = sqrt(4) = 2.0``
        # (MergeOverlappingPeaksAllZarr.c:524).
        overlap_length = 2.0
    if use_pixel_overlap is None:
        use_pixel_overlap = 0
    px_mode = int(use_pixel_overlap)
    if px_mode not in (0, 1, 2):
        raise ValueError(
            f"UsePixelOverlap must be 0 (centroid), 1 (C-compatible) or "
            f"2 (pixel-overlap-first), got {use_pixel_overlap!r}."
        )
    # Only mode 2 consults pixel lists; mode 1 reproduces the C output,
    # which matches by centroid (see docstring).
    overlap_first = px_mode == 2

    # Load frames.
    if frames is None:
        if allpeaks_ps_bin is None:
            allpeaks_ps_bin = rf / "Temp" / "AllPeaks_PS.bin"
            if not Path(allpeaks_ps_bin).exists():
                allpeaks_ps_bin = rf / "AllPeaks_PS.bin"
        allpeaks_ps_bin = Path(allpeaks_ps_bin)
        if not allpeaks_ps_bin.exists():
            raise FileNotFoundError(
                f"AllPeaks_PS.bin not found at {allpeaks_ps_bin}. "
                f"This file is written by midas-peakfit / PeaksFittingOMPZarrRefactor."
            )
        frames = zarr_io.read_allpeaks_ps_frames(allpeaks_ps_bin)

    # Load pixel data when pixel-overlap-first matching is requested.
    if overlap_first and pixel_frames is None:
        if allpeaks_px_bin is None:
            allpeaks_px_bin = rf / "Temp" / "AllPeaks_PX.bin"
            if not Path(allpeaks_px_bin).exists():
                allpeaks_px_bin = rf / "AllPeaks_PX.bin"
        allpeaks_px_bin = Path(allpeaks_px_bin)
        if not allpeaks_px_bin.exists():
            raise FileNotFoundError(
                f"AllPeaks_PX.bin not found at {allpeaks_px_bin}. "
                f"This file is written by midas-peakfit and is required when "
                f"UsePixelOverlap=2."
            )
        nr_pixels_from_file, pixel_frames = zarr_io.read_allpeaks_px_frames(
            allpeaks_px_bin,
        )
        if not nr_pixels:
            nr_pixels = nr_pixels_from_file

    if end_nr is None:
        end_nr = max(start_nr + len(frames) - 1, start_nr)

    # Saturated regions (midas-peakfit's AllPeaks_PS_sat.bin sibling). Absent
    # file == legacy peakfit == legacy merge.
    if include_saturated is None:
        include_saturated = bool(zp.IncludeSaturatedSpots) if zarr_path is not None else False
    if sat_frames is None and allpeaks_ps_bin is not None:
        sat_ps = Path(allpeaks_ps_bin).with_name("AllPeaks_PS_sat.bin")
        if sat_ps.exists():
            sat_frames = zarr_io.read_allpeaks_ps_frames(sat_ps)
            sat_px = Path(allpeaks_ps_bin).with_name("AllPeaks_PX_sat.bin")
            if overlap_first and include_saturated and sat_px.exists():
                _, sat_pixel_frames = zarr_io.read_allpeaks_px_frames(sat_px)

    out, merge_map = _merge_frames(
        frames,
        overlap_length=float(overlap_length),
        use_maxima_positions=use_maxima_positions,
        skip_frame=skip_frame,
        pixel_frames=pixel_frames if overlap_first else None,
        nr_pixels=int(nr_pixels) if nr_pixels else 0,
        sat_frames=sat_frames,
        sat_pixel_frames=sat_pixel_frames,
        include_saturated=bool(include_saturated),
    )
    sat_out = None
    if sat_frames is not None and any(np.size(s) for s in sat_frames):
        sat_out = merge_saturated_only(
            sat_frames, overlap_length=float(overlap_length),
            use_maxima_positions=use_maxima_positions, skip_frame=skip_frame)
    n_flagged = int((out[:, 17] == SATURATED_RETURN_CODE).sum()) if out.size else 0
    if sat_out is not None:
        print(f"[merge] saturated reflections: {sat_out.shape[0]}; "
              f"{'included (IncludeSaturatedSpots 1)' if include_saturated else 'kept out of indexing (IncludeSaturatedSpots 0)'}; "
              f"{n_flagged} merged spots flagged ReturnCode -2 (intensity unreliable).")

    if write:
        out_path = out_dir / f"Result_StartNr_{start_nr}_EndNr_{end_nr}.csv"
        _write_result_csv_c_format(out_path, out)
        _write_merge_map_csv(out_dir / "MergeMap.csv", merge_map)
        if sat_out is not None:
            _write_result_csv_c_format(out_dir / "SaturatedSpots.csv", sat_out)

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)
    return MergeResult(
        peaks=torch.from_numpy(out).to(device=dev, dtype=dt),
        merge_map=merge_map,
        saturated=sat_out,
        n_saturated_flagged=n_flagged,
    )
