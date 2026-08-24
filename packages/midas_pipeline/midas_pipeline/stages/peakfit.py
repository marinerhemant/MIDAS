"""Stage: peakfit.

For PF mode: per-scan invocation of ``midas_peakfit.orchestrator.run``
on each scan's ``.MIDAS.zip``. Writes ``Temp/AllPeaks_PS.bin`` (plus
``AllPeaks_PX.bin`` when pixel-overlap is enabled) into the scan dir,
which the per-scan ``transforms`` stage then consumes.

Sequential per-scan today (matches scope-locked plan); parsl-parallel
fan-out is a follow-up that swaps the for-loop for ``parsl.map`` with
no other changes.

Skips a scan if its ``AllPeaks_PS.bin`` is already present
(resume-friendly). Skips the whole stage if midas_peakfit isn't
importable.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def _peakfit_device(cfg) -> str:
    """Device for the peak-fit backend, honouring ``--peak-fit-gpu``.

    ``--peak-fit-gpu`` exists so peakfit can use the torch GPU backend while
    the rest of the run stays where it is -- peakfit is the long pole, and
    indexing/refinement go to the c-omp binaries on CPU regardless. The flag
    was parsed (``cli.py``) and stored on the config (``config.py``) but never
    read by any stage, so it silently did nothing and every run landed on CPU
    even on 4-GPU hosts.

    ``--device`` still wins when it already names a GPU, so the flag can only
    ever add GPU, never take it away.
    """
    device = cfg.device if cfg.device else "cpu"
    if getattr(cfg, "peak_fit_gpu", False) and not device.startswith("cuda"):
        return "cuda"
    return device


def _aggregate_frames(frames_done: dict, frames_total: dict, n_scans: int):
    """Total frames done / expected across a layer's scans, or None.

    ``None`` means no scan has reported a frame count yet, so the caller has
    nothing better than a scan count to show.

    Scans that have not started are charged the frame count of one that has:
    the scans of a PF layer come from a single acquisition and share a frame
    count. That keeps the denominator fixed from the first report onward, so
    the bar cannot jump backwards as later scans join.
    """
    if not frames_total:
        return None
    per = max(frames_total.values())
    if per <= 0:
        return None
    return sum(frames_done.values()), per * n_scans


def run(ctx: StageContext) -> StageResult:
    started = time.time()

    try:
        from midas_peakfit.orchestrator import run as peakfit_run  # type: ignore
    except ImportError as e:
        LOG.warning("peakfit: midas_peakfit not importable (%s); skipping.", e)
        return stub_run("peakfit", ctx)

    if ctx.is_pf:
        return _run_pf(ctx, started, peakfit_run)
    return _run_ff(ctx, started, peakfit_run)


def _run_ff(ctx: StageContext, started: float, peakfit_run) -> StageResult:
    """FF single-zip path."""
    cfg = ctx.config
    layer_dir = ctx.layer_dir
    zip_path = _resolve_ff_zip(ctx)
    if zip_path is None or not zip_path.exists():
        LOG.info("peakfit(FF): no zarr/zip at %s; skip.", zip_path)
        return stub_run("peakfit", ctx)
    target = layer_dir / "Temp" / "AllPeaks_PS.bin"
    if target.exists():
        LOG.info("peakfit(FF): %s already exists; skip.", target)
        return _result(started, [target], 1, 0)
    target.parent.mkdir(parents=True, exist_ok=True)

    if cfg.run_sr:
        from .. import sr_midas
        sr_midas.log_status(LOG, run_sr=True)
        _run_sr_subprocess(ctx, layer_dir)
        if not target.exists():
            raise RuntimeError(
                f"peakfit(FF): SR-MIDAS reported success but {target} "
                "was not written."
            )
        LOG.info("peakfit(FF): SR-MIDAS wrote %s", target)
        return _result(started, [target], 1, 0)

    peakfit_run(
        data_file=str(zip_path),
        block_nr=0, n_blocks=1, num_procs=max(1, cfg.n_cpus_local),
        result_folder_cli=str(layer_dir),
        fit_peaks_cli=1,
        device=_peakfit_device(cfg), dtype=cfg.dtype,
        # peakfit is the long pole of an FF run (88.5 % of a 2652-grain gamma
        # reconstruction); this is what makes that visible while it runs.
        progress_cb=(ctx.progress.update if ctx.progress else None),
    )
    LOG.info("peakfit(FF): wrote %s", target)
    return _result(started, [target], 1, 0)


def _run_sr_subprocess(ctx: StageContext, layer_dir: Path) -> None:
    """Run SR-MIDAS peak search in a throwaway subprocess.

    Isolation, not convenience: sr-midas's CNN cascade + GPU peak-fit hold
    a large CUDA context (empirically ~20GB+) that is never released while
    the interpreter stays alive. Running it in-process (as midas_ff_pipeline
    currently does) leaves that memory resident into the indexing/refinement
    stages that follow, which then OOM. A subprocess guarantees the context
    is torn down when it exits, matching how stages/indexing.py and
    stages/refinement.py already isolate their own GPU-heavy FF work.
    """
    import sys
    from ._base import run_subprocess

    cfg = ctx.config
    cmd = [
        sys.executable, "-m", "midas_pipeline._sr_worker",
        str(layer_dir),
        "--srfac", str(cfg.srfac),
        "--save-sr-patches", "1" if cfg.save_sr_patches else "0",
        "--save-frame-good-coords", "1" if cfg.save_frame_good_coords else "0",
        "--use-gpu", "1" if cfg.device.startswith("cuda") else "0",
    ]
    if cfg.sr_config_path and cfg.sr_config_path != "auto":
        cmd += ["--sr-config", cfg.sr_config_path]
    run_subprocess(
        cmd, cwd=layer_dir,
        stdout_path=ctx.log_dir / "peakfit_sr_out.log",
        stderr_path=ctx.log_dir / "peakfit_sr_err.log",
    )


def _run_pf(ctx: StageContext, started: float, peakfit_run) -> StageResult:
    """PF per-scan: call peakfit on each scan's zip, sequentially."""
    from .._pf_scans import iter_pf_scans

    cfg = ctx.config
    layer_dir = ctx.layer_dir

    try:
        scans = iter_pf_scans(
            params_file=cfg.params_file,
            layer_dir=layer_dir,
            layer_nr=ctx.layer_nr,
            raw_dir=cfg.raw_dir,
            n_scans_hint=cfg.scan.n_scans,
            work_dir=getattr(cfg, "scan_work_dir", None),
        )
    except FileNotFoundError as e:
        # P0-2: missing positions.csv in PF mode is a HARD error. Every
        # early PF stage used to soft-skip here, so a missing file made
        # the whole run exit 0 having done nothing. (FF never enters this
        # path — _run_pf is dispatched only when ctx.is_pf; the pipeline
        # materializes positions.csv at layer setup, so this fires only
        # for manually-driven stages or a deleted file.)
        raise RuntimeError(
            f"peakfit(PF): scan discovery failed: {e}. Refusing to "
            "soft-skip in PF mode."
        ) from e
    except ValueError as e:
        # Incomplete Parameters.txt (no FileStem / StartFileNrFirstLayer):
        # tolerated for smoke/partial runs; the missing-positions case
        # above is the silent-corruption one.
        LOG.warning("peakfit(PF): scan discovery failed (%s); skip.", e)
        return stub_run("peakfit", ctx)

    from .._pf_scans import fan_out_scans

    # N5: fan out scans across workers/GPUs. Each worker claims its scan
    # (two independent runners can no longer double-process the same
    # scan) and gets a round-robin CUDA device; n_cpus_local is split
    # between concurrent scans so peakfit's internal frame pool doesn't
    # oversubscribe.
    n_workers = max(1, int(getattr(cfg, "scan_workers", 1)))
    num_procs = max(1, cfg.n_cpus_local // n_workers)
    base_device = _peakfit_device(cfg)
    dtype = cfg.dtype if cfg.dtype else "float64"
    n_gpus = 0
    if base_device.startswith("cuda"):
        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 0

    def _device_for(scan_nr: int) -> str:
        if n_workers > 1 and n_gpus > 1 and base_device == "cuda":
            return f"cuda:{(scan_nr - 1) % n_gpus}"
        return base_device

    # FRAMES, aggregated across the scans running concurrently -- not scans.
    # A scan count is far too coarse: 19 scans against 12 workers leaves the
    # bar reading 1/19 for forty minutes and then jumping, which is the same
    # "everything is in flight, nothing has finished" problem the c-omp
    # indexer has with voxels. Each scan reports its own frames every 10, so
    # summing the per-scan counts gives a monotone, fine-grained total.
    frames_done: dict = {}
    frames_total: dict = {}
    frames_lock = threading.Lock()

    def _aggregate_locked():
        return _aggregate_frames(frames_done, frames_total, len(scans))

    def _frames_cb(scan_nr: int):
        def cb(done, total, unit="frames", rate=None):
            with frames_lock:
                frames_done[scan_nr] = int(done)
                frames_total[scan_nr] = int(total)
                agg = _aggregate_locked()
            if agg is not None and ctx.progress is not None:
                ctx.progress.update(agg[0], agg[1], "frames", rate)
        return cb

    def _fit_one_scan(s):
        if s.allpeaks_ps_bin.exists():
            return "cached"
        if not s.zip_path.exists():
            LOG.warning("peakfit(PF): scan %d zip missing at %s; skip.",
                        s.scan_nr, s.zip_path)
            return "failed"
        if not s.hkls_csv.exists():
            LOG.warning("peakfit(PF): scan %d missing hkls.csv at %s — "
                        "the hkl stage didn't run; skip.",
                        s.scan_nr, s.hkls_csv)
            return "failed"
        s.temp_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        peakfit_run(
            data_file=str(s.zip_path),
            block_nr=0, n_blocks=1, num_procs=num_procs,
            result_folder_cli=str(s.scan_dir),
            fit_peaks_cli=1,
            device=_device_for(s.scan_nr), dtype=dtype,
            progress_cb=_frames_cb(s.scan_nr),
        )
        if not s.allpeaks_ps_bin.exists():
            LOG.warning("peakfit(PF): scan %d ran but %s missing.",
                        s.scan_nr, s.allpeaks_ps_bin)
            return "failed"
        LOG.info("peakfit(PF): scan %d/%d done in %.1fs (%s)",
                 s.scan_nr, len(scans), time.time() - t0,
                 s.allpeaks_ps_bin.name)
        return "ok"

    # Scans completed, NOT frames. Up to n_workers scans run concurrently, so
    # frame counts from different scans would interleave into the one sink and
    # go backwards; "14/19 scans" is both monotone and what an operator wants.
    # Counted in a finally so a scan that raises still advances the bar.
    n_done = [0]
    done_lock = threading.Lock()

    def _do_scan(s):
        try:
            return _fit_one_scan(s)
        finally:
            if ctx.progress is not None:
                with done_lock:
                    n_done[0] += 1
                    n_scans_done = n_done[0]
                with frames_lock:
                    # Charge the scan its full frame count on the way out. A
                    # CACHED scan never reports a frame, so without this a
                    # resumed layer would show the bar going backwards as
                    # fresh scans dilute the total.
                    per = max(frames_total.values()) if frames_total else 0
                    if per > 0:
                        frames_done[s.scan_nr] = frames_total.get(s.scan_nr, per)
                    agg = _aggregate_locked()
                if agg is not None:
                    ctx.progress.update(agg[0], agg[1], "frames")
                else:
                    # Nothing has reported frames (every scan cached, or the
                    # first has not got going): scans are all we know.
                    ctx.progress.update(n_scans_done, len(scans), "scans")

    outcomes = fan_out_scans(scans, _do_scan, layer_dir=layer_dir,
                             stage="peakfit", n_workers=n_workers)
    written: list[Path] = []
    skipped_cached = 0
    failed = 0
    for s, out in outcomes:
        if isinstance(out, Exception):
            LOG.warning("peakfit(PF): scan %d (%s) failed: %s",
                        s.scan_nr, s.zip_path.name, out)
            failed += 1
        elif out == "cached":
            skipped_cached += 1
            written.append(s.allpeaks_ps_bin)
        elif out == "ok":
            written.append(s.allpeaks_ps_bin)
        elif out == "failed":
            failed += 1
        # "claimed-elsewhere": another runner owns it; count as neither.

    LOG.info("peakfit(PF): %d processed + %d cached + %d failed (total %d scans)",
             len(written) - skipped_cached, skipped_cached, failed, len(scans))
    return _result(started, written, len(written) - skipped_cached + skipped_cached, failed)


def _resolve_ff_zip(ctx: StageContext):
    if ctx.config.zarr_path:
        return Path(ctx.config.zarr_path)
    for p in ctx.layer_dir.glob("*.MIDAS.zip"):
        return p
    return None


def _result(started: float, written, n_ok: int, n_failed: int) -> StageResult:
    finished = time.time()
    return StageResult(
        stage_name="peakfit",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"allpeaks_ps_bin": [str(p) for p in written]},
        metrics={"n_scans_ok": n_ok, "n_scans_failed": n_failed},
    )
