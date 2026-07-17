"""Per-frame producer + GPU consumer pipeline.

Frame decompression, dark/flood/threshold, connected components, and
seed initialization run on CPU. Fitting runs on the chosen device
(CPU or CUDA). Frames are processed in order; the GPU stage operates
on a stream of regions (one bucket-flush per frame for now).

A future optimization is to pool regions across multiple frames before
flushing the bucket dispatcher; for clarity and determinism in the
initial port we flush per frame.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from midas_peakfit.connected import find_regions, filter_regions_by_size
from midas_peakfit.fit import FitOutput
from midas_peakfit.geometry import compute_good_coords, load_ring_radii
from midas_peakfit.lm import LMConfig
from midas_peakfit.output import FrameAccumulator, write_consolidated_peak_files
from midas_peakfit.panels import generate_panels, load_panel_shifts
from midas_peakfit.params import ZarrParams, resolve_do_peak_fit, resolve_result_folder
from midas_peakfit.pool import RegionPool
from midas_peakfit.preprocess import prepare_dark, prepare_flood, prepare_mask, preprocess_frame
from midas_peakfit.seeds import seed_region
from midas_peakfit.zarr_io import frame_omega, load_corrections, parse_zarr_params, read_frame


def _build_panels(p: ZarrParams):
    panels = generate_panels(
        p.NPanelsY, p.NPanelsZ, p.PanelSizeY, p.PanelSizeZ,
        p.PanelGapsY, p.PanelGapsZ,
    )
    if panels and p.PanelShiftsFile:
        ok = load_panel_shifts(p.PanelShiftsFile, panels)
        print(
            f"{'Loaded' if ok else 'Failed to load'} panel shifts from "
            f"{p.PanelShiftsFile}"
        )
    return panels


def run(
    data_file: str,
    block_nr: int,
    n_blocks: int,
    num_procs: int,
    *,
    result_folder_cli: str | None = None,
    fit_peaks_cli: int | None = None,
    device: str = "cpu",
    dtype: str = "float64",
    batch_size: int = 4096,  # currently advisory; per-frame buckets used
    deterministic: bool = False,
    producer: str = "process",  # "process" | "thread"
    interleave_blocks: bool = False,
    compute_uncertainty: bool = False,
    compute_moments: bool = False,
) -> dict:
    """Run the full pipeline for ``[block_nr, n_blocks)`` slice of ``data_file``.

    Returns a dict with summary stats.
    """
    t0 = time.time()

    # ── 1. Parse parameters & set up corrections ──────────────────────
    p = parse_zarr_params(data_file)
    p.ResultFolder = resolve_result_folder(result_folder_cli, p.ResultFolder)
    p.doPeakFit = resolve_do_peak_fit(fit_peaks_cli, p.doPeakFit)

    print(p.dump())

    panels = _build_panels(p)
    load_corrections(data_file, p)
    ring_rads = load_ring_radii(p, p.ResultFolder)
    good_coords = compute_good_coords(p, panels, ring_rads)

    # Pre-pad+transform dark/flood/mask once
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    flood = prepare_flood(p.flood, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    mask = prepare_mask(p.mask, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)

    # ── 2. Block frame range ──────────────────────────────────────────
    # Two sharding modes:
    #   contiguous (default): block N gets frames [N*chunk, (N+1)*chunk).
    #     Matches C semantics, easy to merge.
    #   interleaved (--interleave-blocks): block N gets frames where
    #     idx % n_blocks == N. Spreads omega-correlated peak density
    #     evenly across all GPUs at the cost of needing per-frame indexing
    #     in the merger.
    block_frames = p.block_frame_indices(block_nr, n_blocks, interleave=interleave_blocks)
    if interleave_blocks:
        print(
            f"Processing {len(block_frames)} frames (interleaved: every "
            f"{n_blocks}th starting at {block_nr})"
        )
        # For per-frame metadata indexing we still use a "linear position"
        # in this block; the absolute frame for omega is in block_frames.
        start_frame = block_frames[0] if block_frames else 0
        end_frame = block_frames[-1] + 1 if block_frames else 0
    else:
        start_frame = block_frames[0] if block_frames else 0
        end_frame = block_frames[-1] + 1 if block_frames else 0
        print(f"Processing frames {start_frame} to {end_frame}")

    # ── 3. Set up torch backend ───────────────────────────────────────
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        # Apple-Silicon MPS only supports float32; if the caller asked for
        # float64 we transparently downgrade and warn (matching the user-
        # facing semantics of every other midas-* CLI on MPS).
        if torch_dtype == torch.float64:
            print("MPS requested but does not support float64; downgrading to float32")
            torch_dtype = torch.float32
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
        if device == "cuda":
            print("CUDA requested but unavailable; falling back to CPU")
        elif device == "mps":
            print("MPS requested but unavailable; falling back to CPU")

    if deterministic and torch_dtype == torch.float64:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"deterministic mode unavailable: {e}")

    # Tolerances scale with dtype: fp64 supports 1e-5 cleanly; in fp32 the
    # underlying epsilon is ~1e-7, so 1e-4 still leaves comfortable headroom.
    # max_iter=100: Adam fallback kicks in ~35% sooner for the slow tail of
    # regions that LM can't converge on. Saves wallclock at no quality cost
    # because those regions wouldn't have converged at iter 200 either.
    if torch_dtype == torch.float32:
        lm_config = LMConfig(max_iter=100, ftol_rel=1e-4, xtol_rel=1e-4,
                             compute_uncertainty=compute_uncertainty)
    else:
        lm_config = LMConfig(max_iter=100, ftol_rel=1e-5, xtol_rel=1e-5,
                             compute_uncertainty=compute_uncertainty)

    # Cross-frame region pool with async GPU consumer thread. Producers
    # only enqueue; the consumer flushes whichever buckets are over their
    # memory cap, in parallel with CPU producers.
    pool = RegionPool(
        device=torch_device,
        dtype=torch_dtype,
        Ycen=p.Ycen,
        Zcen=p.Zcen,
        do_peak_fit=p.doPeakFit,
        local_maxima_only=p.localMaximaOnly,
        lm_config=lm_config,
    )
    pool.start()

    # ── 4. Iterate frames; CPU-side preprocessing only, push to pool ──
    nr_files_done = 0
    n_frames_total = len(block_frames)
    # ``block_frames[i]`` is the absolute frame index for block-local position i.
    # The pool keys frame_outputs by block-local position (0..n-1); the
    # final output writer places them at absolute frame slots.
    abs_to_local = {abs_f: i for i, abs_f in enumerate(block_frames)}

    def _process_frame(frame_nr: int):
        """CPU-side worker: decompress + preprocess + CC + seed for one frame.

        ``frame_nr`` is the absolute frame index (post-block-sharding).
        Returns ``(frame_nr, omega, n_regions_total, seeded_list)`` or
        empty result on read failure.
        """
        omega_local = frame_omega(p, frame_nr + p.skipFrame)
        try:
            raw = read_frame(data_file, frame_nr + p.skipFrame)
        except Exception as e:
            print(f"Frame {frame_nr}: failed to read ({e}); skipping")
            return frame_nr, omega_local, 0, []
        img_corr = preprocess_frame(
            raw,
            NrPixels=p.NrPixels,
            NrPixelsY=p.NrPixelsY,
            NrPixelsZ=p.NrPixelsZ,
            transform_options=p.TransOpt,
            dark=dark,
            flood=flood,
            good_coords=good_coords,
            bc=p.bc,
            bad_px_intensity=p.BadPxIntensity,
            make_map=p.makeMap,
        )
        regions_all = find_regions(img_corr, good_coords)
        regions = filter_regions_by_size(regions_all, p.minNrPx, p.maxNrPx)
        seeded_list = []
        for reg in regions:
            sr = seed_region(
                reg, img_corr, mask,
                Ycen=p.Ycen, Zcen=p.Zcen,
                int_sat=p.IntSat, max_n_peaks=p.maxNPeaks,
                panels=panels,
                compute_moments=compute_moments,
            )
            if sr is not None:
                seeded_list.append(sr)
        return (frame_nr, omega_local, len(regions_all), seeded_list)

    n_workers = max(1, num_procs)

    # N10: cap per-worker BLAS/OpenMP threads OURSELVES. Without this,
    # n_workers × (OMP/BLAS default = all cores) oversubscribes the box —
    # observed on the Ni Layer-3 run as load 28/64 with the frame rate
    # collapsing 22 → 4-5 f/s until the caller exported OMP_NUM_THREADS=1.
    # setdefault: an explicit user setting always wins. Children inherit
    # the env (fork) or re-read it at import (spawn/thread paths).
    if n_workers > 1:
        for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(_var, "1")

    # The bulk-frames-via-fork-COW pattern only works under the 'fork' start
    # method. On macOS / Windows ('spawn' default) workers can't see the
    # parent module global, so fall back to the threaded producer there.
    if producer == "process" and multiprocessing.get_start_method() != "fork":
        print(
            f"[orch] start method is "
            f"'{multiprocessing.get_start_method()}'; falling back to threaded "
            f"producer (fork-COW unavailable)"
        )
        producer = "thread"

    print(
        f"[orch] CPU producer: {n_workers} {producer}s × {n_frames_total} frames"
    )

    cpu_t0 = time.time()
    omega_per_frame = [0.0] * n_frames_total
    n_regions_per_frame = [0] * n_frames_total
    n_filtered_per_frame = [0] * n_frames_total

    completed = 0

    def _ingest(result):
        """Common collector: scatter result into per-frame metadata + push
        the seeded regions to the (async) GPU consumer pool."""
        nonlocal completed
        frame_nr, omega, n_regs, seeded_list = result
        f_local = abs_to_local.get(frame_nr, -1)
        if 0 <= f_local < n_frames_total:
            omega_per_frame[f_local] = omega
            n_regions_per_frame[f_local] = n_regs
            n_filtered_per_frame[f_local] = len(seeded_list)
            pool.add_frame(f_local, omega, seeded_list)
        completed += 1
        if completed % 100 == 0 or completed == n_frames_total:
            elapsed = time.time() - cpu_t0
            rate = completed / max(elapsed, 1e-3)
            print(
                f"  CPU stage progress: {completed}/{n_frames_total} frames, "
                f"{rate:.1f} f/s, elapsed {elapsed:.1f}s"
            )

    if producer == "process":
        # Multi-process producer. Each worker opens the Zarr archive ONCE
        # at init and caches the data array; per-frame reads are then
        # one Blosc decompression each. With many workers running in
        # parallel, this beats a single bulk read in main on this hardware.
        from midas_peakfit._producer_worker import (
            init_worker, process_frame_in_worker,
        )

        p_for_pickle = type(p)(**{**p.__dict__, "dark": None, "flood": None,
                                  "mask": None, "residualMap": None})
        params_pickle = pickle.dumps(p_for_pickle)
        panels_pickle = pickle.dumps(panels)

        # Workers receive zarr-absolute indices (already adjusted for
        # skipFrame). For contiguous mode this is a contiguous range; for
        # interleaved mode it's a strided list. Same downstream code path.
        skip = p.skipFrame
        zarr_indices = [f + skip for f in block_frames]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(
                str(data_file), params_pickle, dark, flood, mask,
                good_coords, panels_pickle, compute_moments,
            ),
        ) as ex:
            for result in ex.map(
                process_frame_in_worker, zarr_indices, chunksize=4,
            ):
                zarr_idx, _omega_unused, n_regs, seeded_list = result
                abs_frame = zarr_idx - skip  # back to "absolute frame number"
                omega = frame_omega(p, zarr_idx)
                _ingest((abs_frame, omega, n_regs, seeded_list))
                nr_files_done += 1
    else:
        # Threaded producer: lower startup cost, but may be GIL-limited.
        # Iterate over the absolute frames this block owns (handles both
        # contiguous and interleaved sharding identically).
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(
                _process_frame, block_frames, chunksize=8
            ):
                # _process_frame returns (local_idx, omega, n_regs, seeded_list)
                # where local_idx is whatever was passed in (here: absolute frame).
                local_idx, omega, n_regs, seeded_list = result
                _ingest((local_idx, omega, n_regs, seeded_list))
                nr_files_done += 1

    print(f"[orch] CPU stage done in {time.time() - cpu_t0:.1f}s; "
          f"signaling consumer end-of-stream and waiting for drain…")
    flush_t0 = time.time()
    pool.stop()
    print(f"[orch] Consumer drain done in {time.time() - flush_t0:.1f}s")

    # Build accumulators in frame order
    accumulators: List[FrameAccumulator] = []
    for f_local in range(n_frames_total):
        acc = FrameAccumulator()
        for fo in pool.frame_outputs.get(f_local, []):
            acc.add(fo)
        accumulators.append(acc)
        if f_local < 5 or f_local % 100 == 0:
            print(
                f"FrameNr: {start_frame + f_local}, "
                f"NrOfRegions: {n_regions_per_frame[f_local] if f_local < len(n_regions_per_frame) else 0}, "
                f"Filtered regions: {n_filtered_per_frame[f_local] if f_local < len(n_filtered_per_frame) else 0}, "
                f"Number of peaks: {acc.n_peaks}"
            )

    # ── 5. Write consolidated outputs ─────────────────────────────────
    out_temp = Path(p.ResultFolder) / "Temp"
    print("Writing consolidated peak files...")
    ps_path, px_path = write_consolidated_peak_files(
        accumulators,
        n_total_frames=p.nFrames,
        start_frame=start_frame,
        end_frame=end_frame,
        nr_pixels=p.NrPixels,
        out_folder=out_temp,
        abs_frames=block_frames,
    )

    total_time = time.time() - t0
    print(
        f"Finished, time elapsed: {total_time:.3f} seconds, "
        f"nrFramesDone: {nr_files_done}."
    )

    return {
        "ps_path": str(ps_path),
        "px_path": str(px_path),
        "n_frames_done": nr_files_done,
        "total_time": total_time,
    }


__all__ = ["run"]
