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
import threading
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
from midas_peakfit.preprocess import (
    apply_threshold, correct_frame, prepare_dark, prepare_flood, prepare_mask,
)
from midas_peakfit.seeds import seed_region
from midas_peakfit.zarr_io import frame_omega, load_corrections, parse_zarr_params, read_frame


_CUDA_INIT_LOCK = threading.Lock()
_CUDA_READY: set = set()


def ensure_cuda_ready(torch_device) -> None:
    """Serialise first-touch CUDA initialisation across threads.

    midas_pipeline fans PF scans out over a ThreadPoolExecutor and gives each
    worker a different device (``cuda:0``..``cuda:3``). torch initialises a
    device lazily on first use, and that path is NOT safe against several
    threads touching different devices at once: the losers die with
    ``RuntimeError: lazy wrapper should be called at most once``.

    That failure was invisible in the worst way. midas_pipeline logs a failed
    scan as a WARNING and finishes the layer from whatever scans survived, so
    the run reports "ok" with a plausible voxel count built from less data.
    Measured on bt_1id_jun25b s4/L1 (9 scans) 2026-08-22: 0 failed -> 61 voxels,
    1 -> 45-52, 2 -> 20-30, 3 -> 25. Every apparent "GPU vs CPU" and "fp32 vs
    fp64" discrepancy in that investigation was this race, not arithmetic.

    Warming each device once, under a lock, in whatever thread gets there
    first, removes the race for every caller rather than only for the one
    fan-out that happened to expose it.
    """
    if getattr(torch_device, "type", None) != "cuda":
        return
    idx = torch_device.index if torch_device.index is not None else 0
    if idx in _CUDA_READY:                      # fast path, no lock
        return
    with _CUDA_INIT_LOCK:
        if idx in _CUDA_READY:                  # double-checked
            return
        torch.cuda.init()
        torch.zeros(1, device=torch_device)     # forces the per-device context
        # ...and the LINALG backend, which is the one that actually races.
        # torch creates its cuSOLVER/MAGMA handles on the FIRST linalg call,
        # not at device init, and that creation is not thread-safe. The real
        # failure was here:
        #     lm.py: torch.linalg.cholesky_ex(H_damped)
        #     RuntimeError: lazy wrapper should be called at most once
        # Warming the device with torch.zeros alone left this untouched, which
        # is why the first attempt at this fix changed nothing. Warm both
        # dtypes: fp32 and fp64 take different cuSOLVER paths.
        for _dt in (torch.float32, torch.float64):
            torch.linalg.cholesky_ex(torch.eye(2, device=torch_device, dtype=_dt))
        torch.cuda.synchronize(torch_device)
        _CUDA_READY.add(idx)


class NonDeterministicFit(RuntimeError):
    """Determinism was requested but torch cannot guarantee it."""


def enable_determinism(deterministic: bool, torch_device) -> None:
    """Make the fit bit-reproducible. Default ON, every device, every dtype.

    A peak fit that cannot be re-derived has no provenance: two runs of the
    same data give different spots, so every downstream number — grain
    positions, strain — silently inherits run-to-run noise that no amount of
    care downstream can remove.

    Three things were wrong before:

    * it was gated on ``torch_dtype == torch.float64``, so a CUDA run (which
      defaults to fp32 under ``--dtype auto``) never got it at all;
    * ``deterministic`` defaulted to False and **midas_pipeline never passed
      it**, so the production path ran non-deterministic on GPU always;
    * failure was swallowed with a ``print``, which is the worst outcome —
      the run continues and looks fine while being irreproducible.

    On CUDA, cuBLAS also needs a fixed workspace or its GEMMs are not
    reproducible run-to-run; torch requires the variable be set before the
    first cuBLAS handle exists, hence ``setdefault`` here at setup time.

    Raises :class:`NonDeterministicFit` rather than warning: if the caller
    asked for determinism, silently not providing it is the bug being fixed.
    """
    if not deterministic:
        print("*** WARNING: deterministic=False — this fit is NOT reproducible "
              "run-to-run. Every spot, grain and strain derived from it "
              "inherits that. Do not use for production. ***")
        return

    if getattr(torch_device, "type", str(torch_device)) == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:                       # pragma: no cover - torch-version dependent
        raise NonDeterministicFit(
            f"determinism was requested but torch refused it on {torch_device}: "
            f"{e}. Fix the environment rather than proceeding — an "
            f"irreproducible peak fit invalidates the provenance of every "
            f"number derived from it."
        ) from e


def resolve_torch_device(device: str, dtype: str):
    """Map a caller's device/dtype request onto (torch.device, torch.dtype).

    Accepts ``"cuda"`` AND ``"cuda:N"``. midas_pipeline shards PF scans
    round-robin across GPUs (``stages/peakfit.py::_device_for`` returns
    ``f"cuda:{(scan_nr - 1) % n_gpus}"``), so on a multi-GPU host this arrives
    as ``"cuda:0"``..``"cuda:3"``.

    This used to be an exact ``device == "cuda"`` test, which rejected every
    sharded string and fell through to CPU — and the "falling back" warning was
    gated on the same exact test, so the fallback was **silent**. The perverse
    result: peakfit used the GPU on a single-GPU host (where ``_device_for``
    returns plain ``"cuda"``) but quietly ran on CPU on a multi-GPU host, i.e.
    the more GPUs present, the less GPU was used. Measured on alleppey
    2026-08-22: 4x H100 at 0% utilisation while peakfit burned 93 CPU cores.

    Returns the resolved device and dtype; MPS forces float32 because it has no
    float64 support.
    """
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device), torch_dtype
    if device == "mps" and torch.backends.mps.is_available():
        # Apple-Silicon MPS only supports float32; if the caller asked for
        # float64 we transparently downgrade and warn (matching the user-
        # facing semantics of every other midas-* CLI on MPS).
        if torch_dtype == torch.float64:
            print("MPS requested but does not support float64; downgrading to float32")
            torch_dtype = torch.float32
        return torch.device("mps"), torch_dtype
    if device.startswith("cuda"):
        print(f"CUDA requested ({device}) but unavailable; falling back to CPU")
    elif device == "mps":
        print("MPS requested but unavailable; falling back to CPU")
    return torch.device("cpu"), torch_dtype


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
    deterministic: bool = True,   # provenance requirement: see enable_determinism
    producer: str = "process",  # "process" | "thread"
    interleave_blocks: bool = False,
    compute_uncertainty: bool = False,
    compute_moments: bool = False,
    progress_cb=None,
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

    # Opt-in local background subtraction (BgSubtract 1). None => legacy/C path.
    # Cells are needed by BgSubtract AND by the MinPeakSNR filter.
    need_bins = (getattr(p, "BgSubtract", 0) == 1
                 or float(getattr(p, "MinPeakSNR", 0.0)) > 0.0)
    snr_bins = None
    bg_bins = None
    if need_bins:
        from midas_peakfit.background import bins_from_params

        snr_bins = bins_from_params(p, panels, ring_rads,
                                    n_sectors=int(getattr(p, "BgNSectors", 36)))
        if getattr(p, "BgSubtract", 0) == 1:
            bg_bins = snr_bins
        if float(getattr(p, "MinPeakSNR", 0.0)) > 0.0:
            print(f"MinPeakSNR={float(p.MinPeakSNR):g}: peaks below this local "
                  f"SNR will be rejected at detection.")
        if snr_bins is None:
            print("No ring bands are available (DoFullImage, or no "
                  "rings/radii) -- background subtraction and the MinPeakSNR "
                  "filter are both DISABLED for this run.")
        else:
            thin = snr_bins.thin_cells()
            print(f"{snr_bins.n_bins} background cells "
                  f"({snr_bins.n_sectors} sectors/ring)"
                  + (f", {len(thin)} thin cells fall back to the ring median"
                     if len(thin) else ""))

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
    min_peak_snr = float(getattr(p, "MinPeakSNR", 0.0))

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
    torch_device, torch_dtype = resolve_torch_device(device, dtype)

    ensure_cuda_ready(torch_device)
    enable_determinism(deterministic, torch_device)

    # Tolerances scale with dtype: fp64 supports 1e-5 cleanly; in fp32 the
    # underlying epsilon is ~1e-7, so 1e-4 still leaves comfortable headroom.
    # max_iter stays at 100, but NOT for the reason originally given here. The
    # old note claimed the slow tail "wouldn't have converged at iter 200
    # either" -- an assertion about the data, so it was measured on one
    # bt_1id_jun25b s1 scan (8069 peaks, fp64, determinism on) by overriding
    # MIDAS_PEAKFIT_MAX_ITER. Regions where LM runs out of iterations fall
    # through to the Adam recovery path and are flagged return_code = -1:
    #
    #   max_iter   rc=-1 CPU     rc=-1 GPU     CPU wall    GPU wall
    #      100     615  (7.6%)   756  (9.4%)     407 s        64 s
    #      200     342  (4.2%)   360  (4.5%)     584 s        77 s
    #      400     154  (1.9%)   127  (1.6%)     901 s       100 s
    #
    # (1440 frames, 8 procs; wall from the orchestrator's own timer. Absolute
    # times are inflated -- the host was running a campaign -- but all six ran
    # inside 37 min under the same load, so the ratios hold.)
    #
    # The claim is false: more iterations DO convert the tail. Keeping 100 is
    # justified by a different measurement -- the fallback count is a
    # diagnostic, not an error. Going 100 -> 400 moves the fitted centre by a
    # median of 0.018 px for the 7.6 % that were rescued (0.000 px over all
    # peaks; p99 0.132 px) while costing 2.2x CPU wall time. It cannot move it
    # much further: the centre is box-bounded to R +/- 1 px and Eta +/- dEta by
    # the seeder (see ``seeds.py``), so Adam already lands essentially where
    # LM@400 lands. Raise this only when the fallback rate is itself the object
    # of study, or on GPU where the same sweep costs only 64 s -> 100 s.
    _max_iter = int(os.environ.get("MIDAS_PEAKFIT_MAX_ITER", "100"))
    if torch_dtype == torch.float32:
        lm_config = LMConfig(max_iter=_max_iter, ftol_rel=1e-4, xtol_rel=1e-4,
                             compute_uncertainty=compute_uncertainty)
    else:
        lm_config = LMConfig(max_iter=_max_iter, ftol_rel=1e-5, xtol_rel=1e-5,
                             compute_uncertainty=compute_uncertainty)
    if _max_iter != 100:
        print(f"[orch] LM max_iter={_max_iter} (default 100)")

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
        Returns ``(frame_nr, omega, n_regions_total, seeded_list, n_saturated)``
        or an empty result on read failure. ``n_saturated`` is the number of
        regions discarded whole for containing a pixel over ``IntSat``.
        """
        omega_local = frame_omega(p, frame_nr + p.skipFrame)
        try:
            raw = read_frame(data_file, frame_nr + p.skipFrame)
        except Exception as e:
            print(f"Frame {frame_nr}: failed to read ({e}); skipping")
            return frame_nr, omega_local, 0, [], 0
        corrected = correct_frame(
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
            bg_bins=bg_bins,
        )
        img_corr = apply_threshold(corrected, good_coords)
        regions_all = find_regions(img_corr, good_coords)
        regions = filter_regions_by_size(regions_all, p.minNrPx, p.maxNrPx)
        if min_peak_snr > 0.0 and snr_bins is not None:
            from midas_peakfit.background import filter_regions_by_snr

            # SNR is measured on `corrected` (UNGATED): on the thresholded
            # frame every sub-threshold pixel is 0, so the background and its
            # MAD collapse and the SNR is meaningless.
            regions, _ = filter_regions_by_snr(
                regions, corrected, snr_bins, min_peak_snr)
        seeded_list = []
        n_saturated = 0
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
            else:
                # Saturation is the only None case (seeds.seed_region), and it
                # silently deletes a STRONG reflection. Count it so the loss
                # reaches the log instead of vanishing into the gap between
                # `NrOfRegions` and `Filtered regions`, which also contains
                # the size and SNR cuts.
                n_saturated += 1
        return (frame_nr, omega_local, len(regions_all), seeded_list,
                n_saturated)

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

    # CUDA and fork do not mix. This process initialises CUDA for the consumer,
    # and a forked child inherits a CUDA context it cannot use; any torch CUDA
    # touch in the child then raises "lazy wrapper should be called at most
    # once". midas_pipeline catches that per scan, logs a WARNING, and finishes
    # the layer from whatever scans survived -- so the run reports "ok" with
    # fewer voxels and nothing marks the loss. Measured on bt_1id_jun25b s4/L1
    # (9 scans) 2026-08-22/23: 1-3 scans lost per GPU run, voxels 61 -> 20-52,
    # and EVERY apparent "GPU vs CPU" and "fp32 vs fp64" discrepancy in that
    # investigation was this, not arithmetic.
    # 'spawn' is not an option: the fork-COW producer needs the child to see a
    # parent module global. So on CUDA, produce with threads.
    if producer == "process" and torch_device.type == "cuda":
        print("[orch] CUDA device: using the threaded producer — fork is "
              "unsafe once CUDA is initialised in the parent")
        producer = "thread"

    print(
        f"[orch] CPU producer: {n_workers} {producer}s × {n_frames_total} frames"
    )

    cpu_t0 = time.time()
    omega_per_frame = [0.0] * n_frames_total
    n_regions_per_frame = [0] * n_frames_total
    n_filtered_per_frame = [0] * n_frames_total
    n_saturated_per_frame = [0] * n_frames_total

    completed = 0

    def _ingest(result):
        """Common collector: scatter result into per-frame metadata + push
        the seeded regions to the (async) GPU consumer pool."""
        nonlocal completed
        # Tolerate a 4-tuple as well as a 5-tuple.
        #
        # The saturation counter widened this contract, and orchestrator.py /
        # _producer_worker.py are deployed as separate files into a
        # site-packages env that a 47-layer campaign is running off. A
        # half-applied deploy breaks in BOTH directions: a 5-tuple unpack
        # against a 4-tuple return, or the reverse. Neither is worth killing a
        # scan over, because n_saturated is a DIAGNOSTIC — degrading it to 0
        # loses a log line, while raising loses the layer. midas_pipeline logs
        # a failed scan as a WARNING and finishes from whatever survived, so
        # the failure would be near-invisible.
        frame_nr, omega, n_regs, seeded_list, *_sat = result
        n_sat = _sat[0] if _sat else 0
        f_local = abs_to_local.get(frame_nr, -1)
        if 0 <= f_local < n_frames_total:
            omega_per_frame[f_local] = omega
            n_regions_per_frame[f_local] = n_regs
            n_filtered_per_frame[f_local] = len(seeded_list)
            n_saturated_per_frame[f_local] = n_sat
            pool.add_frame(f_local, omega, seeded_list)
        completed += 1
        # Every 10 frames, not 100: on a slow dataset 100 frames can be
        # minutes of silence, and this counter is what a caller uses to tell
        # a slow run from a hung one.
        if completed % 10 == 0 or completed == n_frames_total:
            elapsed = time.time() - cpu_t0
            rate = completed / max(elapsed, 1e-3)
            if progress_cb is not None:
                # Reporting must never be able to kill a peak search.
                try:
                    progress_cb(completed, n_frames_total, "frames", rate)
                except Exception:
                    pass
            if completed % 100 == 0 or completed == n_frames_total:
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
                good_coords, panels_pickle, compute_moments, bg_bins, snr_bins,
            ),
        ) as ex:
            for result in ex.map(
                process_frame_in_worker, zarr_indices, chunksize=4,
            ):
                zarr_idx, _omega_unused, n_regs, seeded_list, *_sat = result
                n_sat = _sat[0] if _sat else 0   # 4-tuple tolerated: see _ingest
                abs_frame = zarr_idx - skip  # back to "absolute frame number"
                omega = frame_omega(p, zarr_idx)
                _ingest((abs_frame, omega, n_regs, seeded_list, n_sat))
                nr_files_done += 1
    else:
        # Threaded producer: lower startup cost, but may be GIL-limited.
        # Iterate over the absolute frames this block owns (handles both
        # contiguous and interleaved sharding identically).
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(
                _process_frame, block_frames, chunksize=8
            ):
                # _process_frame returns
                # (local_idx, omega, n_regs, seeded_list, n_saturated) where
                # local_idx is whatever was passed in (here: absolute frame).
                local_idx, omega, n_regs, seeded_list, *_sat = result
                n_sat = _sat[0] if _sat else 0   # 4-tuple tolerated: see _ingest
                _ingest((local_idx, omega, n_regs, seeded_list, n_sat))
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
                f"Saturated (dropped): {n_saturated_per_frame[f_local] if f_local < len(n_saturated_per_frame) else 0}, "
                f"Number of peaks: {acc.n_peaks}"
            )

    # Saturation deletes whole regions, and a saturated reflection is a strong
    # one — so the loss shows up downstream as incompleteness AND as an
    # inflated grain size (it was the brightest contributor to that ring's
    # powder normalisation). Neither is attributable without this line.
    n_sat_total = sum(n_saturated_per_frame)
    if n_sat_total:
        n_reg_total = sum(n_regions_per_frame)
        print(
            f"[orch] Saturated regions dropped: {n_sat_total} of "
            f"{n_reg_total} ({100.0 * n_sat_total / max(n_reg_total, 1):.2f}%), "
            f"IntSat={p.IntSat:g}. These emit NO peaks and carry no flag; "
            f"raise UpperBoundThreshold or attenuate if this fraction is "
            f"large."
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
