"""Stage: refinement.

Two paths, one orchestrator:

- **PF mode** (``scan_mode='pf'``): invokes
  ``midas_fit_grain.scan_driver.refine_scanning_block`` on the
  consolidated ``Output/IndexBest_all.bin`` produced by the indexing
  stage. Each voxel's top candidate is refined under the scan-aware
  filter; per-voxel ``Results/Result_OrientPos_voxel_N.csv`` written
  for ``consolidation_pf`` to aggregate.
- **FF mode** (``scan_mode='ff'``): shells out to ``python -m midas_fit_grain``
  matching ``midas-ff-pipeline.stages.refine`` byte-for-byte. Produces
  ``Output/FitBest.bin`` + ``Results/OrientPosFit.bin``.

Both modes ultimately invoke the same ``midas-fit-grain`` kernels.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from midas_fit_grain.losses import MULTIDET_LOSS, PANEL_DEPENDENT_LOSSES

from .._logging import LOG
from ..results import RefineResult, StageResult
from ._base import run_checked_streamed, StageContext
from ._stub import stub_run

# midas_fit_grain's driver emits this marker when a refined grain position
# came back bit-identical to its seed — i.e. the fit never moved it.
_UNREFINED_MARKER = "UNREFINED-POSITIONS:"


def _surface_unrefined_positions(log_dir: Path) -> None:
    """Re-log the refiner's unrefined-position warning into the run log.

    FF refinement runs in a subprocess whose output goes to
    ``refinement_{out,err}.csv``, which nobody reads. A run where the solver
    silently returned seed positions therefore looked completely normal in
    ``ff_run.log`` — that is how ~158 µm of grain-position error shipped
    unnoticed (1-ID GE5 Au3, 2026-07-30). Promote it to the log people
    actually read.
    """
    for name in ("refinement_err.csv", "refinement_out.csv"):
        path = log_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if _UNREFINED_MARKER in line:
                LOG.warning("refinement(FF): %s",
                            line.split(_UNREFINED_MARKER, 1)[1].strip())
                return


def _expose_legacy_seeds_for_c_refiner(layer_dir: Path) -> None:
    """Make python-indexer seeds visible where the C refiner looks for them.

    FitUnified.c probes ``Output/IndexBest_all.bin`` (the consolidated family
    that BOTH modern indexer backends write) and otherwise falls back to the
    legacy pair at ``<OutputFolder>/IndexBest.bin`` + ``IndexBestFull.bin``.

    This is a no-op for the modern backends, which write the consolidated
    family into ``Output/`` where the C refiner already looks. It exists for
    the legacy pair — from the classical ``IndexerOMP``, or a run seeded from
    one — which lands bare in *layer_dir* while the comp paramstest points
    OutputFolder at ``<layer_dir>/Output``; without this the C refiner would
    find neither seed source and refine nothing.

    Symlinked, not copied: IndexBestFull.bin is ~1.8 GB per layer.
    """
    out_dir = layer_dir / "Output"
    for name in ("IndexBest.bin", "IndexBestFull.bin"):
        src, dst = layer_dir / name, out_dir / name
        if src.exists() and not dst.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)


def _run_ff(ctx: StageContext) -> StageResult:
    """FF (single-scan) refinement — shell out to ``python -m midas_fit_grain``.

    Mirrors ``midas_ff_pipeline.stages.refine.run`` argument-for-argument,
    including the multi-detector pixel→angular loss swap.
    """
    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    paramstest = layer_dir / "paramstest.txt"
    spots_to_index = layer_dir / "SpotsToIndex.csv"
    if not paramstest.exists() or not spots_to_index.exists():
        LOG.info("refinement(FF): missing paramstest or SpotsToIndex.csv → skip.")
        return stub_run("refinement", ctx)

    n_seeds = sum(1 for line in spots_to_index.open() if line.strip())
    output_dir = layer_dir / "Output"
    results_dir = layer_dir / "Results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Clear refiner outputs left by a previous run, from BOTH the c-omp
    # (Output//Results/) and python (bare layer_dir) conventions, before
    # regenerating them — so process_grains' backend-agnostic readers can't
    # resolve a stale file from the other backend's location and mix it with
    # this run's fresh records. IndexBest_all.bin is refinement's *input* and
    # is deliberately left intact.
    for _stale in (results_dir / "OrientPosFit.bin", results_dir / "Key.bin",
                   results_dir / "ProcessKey.bin", output_dir / "FitBest.bin",
                   layer_dir / "OrientPosFit.bin", layer_dir / "Key.bin",
                   layer_dir / "ProcessKey.bin", layer_dir / "FitBest.bin"):
        _stale.unlink(missing_ok=True)

    # The 2D 'pixel' loss is removed (it omitted omega and gave poor,
    # under-determined fits); refinement always uses a full 3D / angular loss.
    loss = ctx.config.refinement.loss

    # Multi-detector: a pixel-based loss is per-panel (own beam centre + Lsd),
    # so one global residual mixes incompatible frames. This stage's docstring
    # has promised "the multi-detector pixel→angular loss swap" throughout,
    # but the code was lost when 'pixel' was retired — and the default became
    # 'full3d', which is pixel-based too (y_pixel, z_pixel, Δω·r_px). So every
    # multi-panel run since has silently refined on a meaningless residual.
    if loss in PANEL_DEPENDENT_LOSSES and "\nDetParams " in (
            "\n" + paramstest.read_text()):
        LOG.info("refinement(FF): multi-detector paramstest → switching loss "
                 "%r → %r (pixel-based losses are per-panel)",
                 loss, MULTIDET_LOSS)
        loss = MULTIDET_LOSS

    refine_dtype = ctx.config.refinement.dtype
    if refine_dtype != ctx.config.dtype:
        LOG.info("refinement(FF): using dtype=%s (run dtype=%s) — conservative "
                 "default; see RefinementConfig.dtype",
                 refine_dtype, ctx.config.dtype)

    # c-omp backend writes its IndexBest*_all.bin into <layer_dir>/Output; hand
    # fit-grain the matching paramstest so it reads them from there.
    #
    # This must key on the REFINER too, not just the indexer: the C refiner
    # locates the binned inputs via dirname(OutputFolder) exactly as the C
    # indexer does, so with indexer=python (which writes a bare
    # OutputFolder <layer_dir>) it looked one level too high and died on
    # `open <layer_dir>/../ExtraInfo.bin`. Measured on shade_LSHR: the
    # python-indexer + c-omp-refiner combination could not complete a layer.
    fit_paramstest = paramstest
    if "c-omp" in (ctx.config.indexer_backend, ctx.config.refine_backend):
        from ._comp_params import comp_backend_paramstest
        fit_paramstest = comp_backend_paramstest(paramstest, layer_dir)
        if ctx.config.refine_backend == "c-omp":
            _expose_legacy_seeds_for_c_refiner(layer_dir)

    log_dir = Path(ctx.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if ctx.config.refine_backend == "c-omp":
        # Bundled unified C refiner (midas_fitgrain / FitUnified): FF mode
        # auto-detected (no/1-row positions.csv), refines position via the
        # spatial objective. Reads the same comp paramstest + seeds.
        from midas_fit_grain import backend_c
        if not backend_c.available():
            raise RuntimeError(
                "refine_backend='c-omp' but the midas_fitgrain binary is not "
                "available. Re-install midas-fit-grain with an OpenMP toolchain "
                "(macOS: `brew install libomp`), or use --refine-backend python."
            )
        LOG.info("refinement(FF, c-omp): %s  [%d seeds]",
                 backend_c.binary_path(), n_seeds)
        proc = backend_c.run_refiner(
            fit_paramstest, block_nr=0, n_blocks=1, n_work=n_seeds,
            num_procs=ctx.config.n_cpus, cwd=layer_dir,
        )
        (log_dir / "refinement_out.csv").write_bytes(proc.stdout or b"")
        (log_dir / "refinement_err.csv").write_bytes(proc.stderr or b"")
        if proc.returncode != 0:
            raise RuntimeError(
                f"midas_fitgrain (c-omp) exited {proc.returncode}; see "
                f"{log_dir / 'refinement_err.csv'}"
            )
    else:
        cmd = [
            sys.executable, "-m", "midas_fit_grain",
            str(fit_paramstest),
            "0", "1",                              # block_nr, n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
            "--solver", ctx.config.refinement.solver,
            "--loss", loss,
            # Forward the run's device so the refiner doesn't auto-select MPS
            # (which can't do float64 → crash on Apple Silicon). Honors
            # --device cpu/cuda from the pipeline invocation.
            "--device", str(ctx.config.device),
            # dtype comes from RefinementConfig, NOT the run's global dtype:
            # `--dtype auto` resolves to float32 on cuda for peak-fitting
            # throughput, and fp32 costs ~158 µm of grain position here. See
            # the note on RefinementConfig.dtype.
            "--dtype", str(refine_dtype),
        ]
        if ctx.config.refinement.mode:
            cmd += ["--mode", ctx.config.refinement.mode]
        LOG.info("refinement(FF): %s", " ".join(cmd))
        with (log_dir / "refinement_out.csv").open("w") as out_fp, \
             (log_dir / "refinement_err.csv").open("w") as err_fp:
            run_checked_streamed(
                cmd, cwd=layer_dir, out_fp=out_fp, err_fp=err_fp,
                line_cb=(ctx.progress.feed_line if ctx.progress else None),
            )
        _surface_unrefined_positions(log_dir)

    finished = time.time()
    orient_pos_fit = results_dir / "OrientPosFit.bin"
    n_grains_refined = 0
    if orient_pos_fit.exists():
        n_grains_refined = orient_pos_fit.stat().st_size // 8
    # A refiner that exits 0 having written nothing is a failure, and it used to
    # be a silent one: process-grains then died two stages later on
    # `np.memmap` of an empty file, with a traceback pointing at
    # midas_process_grains/io/binary.py and no hint the real problem was here
    # (github.com/marinerhemant/MIDAS issues/68).
    if n_grains_refined == 0:
        LOG.warning(
            "refinement(FF): %s is %s — the %s refiner exited 0 but refined "
            "NOTHING. Nothing downstream can succeed; check %s and confirm the "
            "indexing stage produced seeds.",
            orient_pos_fit,
            "absent" if not orient_pos_fit.exists() else "empty (0 bytes)",
            ctx.config.refine_backend, log_dir)
    return RefineResult(
        stage_name="refinement",
        started_at=started, finished_at=finished, duration_s=finished - started,
        orient_pos_fit_bin=str(orient_pos_fit),
        results_dir=str(results_dir),
        n_grains_refined=int(n_grains_refined),
        n_voxels_refined=0,
        outputs={
            str(orient_pos_fit): "",
            str(output_dir / "FitBest.bin"): "",
        },
        metrics={"scan_mode": "ff",
                 "refine_backend": ctx.config.refine_backend,
                 "loss": loss,
                 "solver": ctx.config.refinement.solver,
                 "mode": ctx.config.refinement.mode or "all_at_once"},
    )


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        return _run_ff(ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    results_dir = layer_dir / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    index_best_all = layer_dir / "Output" / "IndexBest_all.bin"
    positions_csv = layer_dir / "positions.csv"
    paramstest = layer_dir / "paramstest.txt"
    # Soft skip when upstream artefacts aren't present (smoke runs /
    # partial pipelines).
    if not index_best_all.exists() or not positions_csv.exists() or not paramstest.exists():
        LOG.info("refinement(PF): missing upstream artefacts → skip.")
        return stub_run("refinement", ctx)

    LOG.info("refinement(PF): index_best_all=%s, results_dir=%s",
             index_best_all, results_dir)

    if ctx.config.refine_backend == "c-omp":
        # Bundled unified C refiner (midas_fitgrain / FitUnified): PF mode
        # auto-detected (positions.csv > 1 row), position FIXED to the voxel grid.
        # The C reads the seed for each voxel from IndexBest_all.bin, indexed by a
        # 5-col SpotsToIndex.csv (voxNr SpId nSpotsBest _ bestSolIdx). The python
        # indexer never emits that file, so without it the C refiner silently
        # refines nothing. We (1) synthesise the 5-col seed from IndexBest_all.bin
        # (same highest-completeness pick as the python refiner), (2) run the C
        # refiner into a dedicated FitBest dir, (3) adapt FitBest_*.csv ->
        # Result_OrientPos_voxel_*.csv so pf-odf + consolidation_pf can read it.
        from midas_fit_grain import backend_c
        from midas_fit_grain.scan_seed import write_pf_seed_file
        from midas_fit_grain.fitbest_adapter import fitbest_to_result_orientpos
        from ._comp_params import comp_backend_paramstest
        if not backend_c.available():
            raise RuntimeError(
                "refine_backend='c-omp' but the midas_fitgrain binary is not "
                "available. Re-install midas-fit-grain with an OpenMP toolchain, "
                "or use --refine-backend python."
            )
        # (1) 5-col seed (the C binary opens 'SpotsToIndex.csv' in cwd=layer_dir).
        spots_to_index = layer_dir / "SpotsToIndex.csv"
        n_vox = write_pf_seed_file(index_best_all, spots_to_index)
        # (2) point FitBest output at a dedicated dir (not Results/, which the
        # adapter fills — same-dir FitBest + Result_OrientPos would double-count
        # in consolidation_pf).
        fitbest_dir = layer_dir / "FitBest_comp"
        comp_pt = comp_backend_paramstest(paramstest, layer_dir,
                                          result_folder=fitbest_dir)
        log_dir = Path(ctx.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("refinement(PF, c-omp): %s  [%d voxels; seed synthesised]",
                 backend_c.binary_path(), n_vox)
        proc = backend_c.run_refiner(
            comp_pt, block_nr=0, n_blocks=1, n_work=n_vox,
            num_procs=ctx.config.n_cpus, cwd=layer_dir,
            # Live seed counts into progress.txt; the streaming path still
            # returns stdout, so the log written below is unchanged.
            line_cb=(ctx.progress.feed_line if ctx.progress else None),
        )
        (log_dir / "refinement_out.csv").write_bytes(proc.stdout or b"")
        (log_dir / "refinement_err.csv").write_bytes(proc.stderr or b"")
        if proc.returncode != 0:
            raise RuntimeError(
                f"midas_fitgrain (c-omp PF) exited {proc.returncode}; see "
                f"{log_dir / 'refinement_err.csv'}"
            )
        # (3) FitBest_*.csv -> Result_OrientPos_voxel_*.csv.
        n_written = fitbest_to_result_orientpos(fitbest_dir, results_dir)
        LOG.info("refinement(PF, c-omp): adapted %d FitBest -> "
                 "Result_OrientPos_voxel", n_written)
        # (4) Per-voxel SpotMatrix.csv, including the reflections each voxel was
        #     PREDICTED to produce and which were never found. PF had no
        #     SpotMatrix at all, and Result_OrientPos_voxel carries completeness
        #     only as a number — the deficit itself was recorded nowhere.
        #     Built from SpotDiagnostics.bin, which already holds observed and
        #     predicted positions, the residuals and the scan, so it needs no
        #     per-scan InputAll join (PF has one such file per scan, and getting
        #     that join wrong is silent).
        try:
            from midas_process_grains.io.spot_diag import (
                load_spot_diag, write_pf_spot_matrix,
            )
            diag = load_spot_diag(fitbest_dir)
            sm = results_dir.parent / "SpotMatrix.csv"
            n_rows = write_pf_spot_matrix(diag, sm)
            n_un = int((diag.n_theor - diag.n_matched).sum())
            LOG.info("refinement(PF, c-omp): SpotMatrix.csv %d rows "
                     "(%d matched + %d predicted-but-NOT-found) -> %s",
                     n_rows, n_rows - n_un, n_un, sm)
            if not diag.col5_is_theor_spot_id:
                LOG.warning("refinement(PF): SpotDiagnostics is v1 — "
                            "theorSpotID is blank on matched rows (the writer "
                            "stored theorGx there). Re-run with a refiner "
                            ">= midas-fit-grain 0.9.0 to populate it.")
        except FileNotFoundError:
            LOG.info("refinement(PF): no SpotDiagnostics.bin -> no "
                     "SpotMatrix.csv (refiner older than 2026-08-21?)")
        except Exception as exc:                       # noqa: BLE001
            # A diagnostic must never take down a refinement that succeeded.
            LOG.warning("refinement(PF): SpotMatrix.csv not written (%s)", exc)
        finished = time.time()
        return RefineResult(
            stage_name="refinement",
            started_at=started, finished_at=finished, duration_s=finished - started,
            orient_pos_fit_bin="", results_dir=str(results_dir),
            n_grains_refined=0, n_voxels_refined=int(n_written),
            outputs={str(results_dir): ""},
            metrics={"scan_mode": "pf", "refine_backend": "c-omp",
                     "n_voxels_processed": n_vox, "n_voxels_written": n_written},
        )

    # Lazy imports to keep FF runs lean.
    from midas_fit_grain.config import FitConfig
    from midas_fit_grain.driver import _build_model, _read_hkls_csv
    from midas_fit_grain.observations import ObservedSpots
    from midas_fit_grain.io_binary import read_extra_info
    from midas_fit_grain.scan_driver import refine_scanning_block
    import torch

    # Build FitConfig from paramstest. The legacy reader in
    # midas-fit-grain.config.from_param_file handles the canonical keys.
    cfg = FitConfig.from_param_file(paramstest)
    cfg.scan_pos_tol_um = (
        ctx.config.scan.scan_pos_tol_um
        if ctx.config.scan.scan_pos_tol_um > 0
        else (ctx.config.scan.beam_size_um / 2.0)
    )
    cfg.friedel_symmetric_scan_filter = ctx.config.scan.friedel_symmetric_scan_filter
    cfg.beam_size_um = ctx.config.scan.beam_size_um
    cfg.position_mode = ctx.config.refinement.position_mode
    cfg.mode = ctx.config.refinement.mode or "all_at_once"
    cfg.solver = ctx.config.refinement.solver
    cfg.loss = ctx.config.refinement.loss
    cfg.use_bounds = ctx.config.refinement.use_bounds
    cfg.bound_euler_deg = ctx.config.refinement.bound_euler_deg
    cfg.bound_lat_abc_pct = ctx.config.refinement.bound_lat_abc_pct
    cfg.bound_lat_angle_deg = ctx.config.refinement.bound_lat_angle_deg
    # NB the real cure for the per-voxel ~20° orientation drift was the loss,
    # not bounds: the old 'pixel' loss was 2D (y,z) and omitted omega, leaving
    # the crystal free to rotate in ω. The default is now the full 3D 'angular'
    # loss (2θ,η,ω) and 'pixel' is disabled. See dev/REFINEMENT_DRIFT_FIX.md.

    # Build the forward model + observations once for the whole voxel loop.
    device = torch.device(ctx.config.device)
    dtype = torch.float64 if ctx.config.dtype == "float64" else torch.float32

    extra_info_path = layer_dir / "ExtraInfo.bin"
    if not extra_info_path.exists():
        raise FileNotFoundError(
            f"refinement(PF): missing {extra_info_path}; transforms didn't run."
        )
    extra = read_extra_info(extra_info_path, mmap=True)
    # PF refinement needs obs for ALL spots that might match any voxel's
    # candidate orientations. Load every spot from ExtraInfo by passing
    # its full SpotID column. (FF refinement could subset by
    # SpotsToIndex.csv; PF can't because matched_ids vary per voxel.)
    all_spot_ids = extra[:, 4].astype(np.int64)
    obs = ObservedSpots.from_extra_info(
        extra, spot_ids=all_spot_ids, device=device, dtype=dtype,
    )

    hkls_path = layer_dir / "hkls.csv"
    if cfg.RhoD > 0.0 and cfg.Lsd > 0.0:
        import math
        max_two_theta_deg = 2.0 * math.degrees(math.atan(cfg.RhoD / cfg.Lsd))
    else:
        max_two_theta_deg = 180.0
    hkls_int, thetas_deg, ring_nr = _read_hkls_csv(
        hkls_path, cfg.RingNumbers, max_two_theta_deg=max_two_theta_deg,
    )
    model, pred_ring_slot = _build_model(
        cfg, device=device, dtype=dtype,
        hkls_int=hkls_int, thetas_deg=thetas_deg, ring_nr=ring_nr,
    )

    voxel_results = refine_scanning_block(
        cfg,
        index_best_all=index_best_all,
        positions_csv=positions_csv,
        results_dir=results_dir,
        model=model,
        obs=obs,
        pred_ring_slot=pred_ring_slot,
        voxel_block_nr=0, voxel_n_blocks=1,
    )

    finished = time.time()
    return RefineResult(
        stage_name="refinement",
        started_at=started, finished_at=finished, duration_s=finished - started,
        orient_pos_fit_bin="",
        results_dir=str(results_dir),
        n_grains_refined=0,
        n_voxels_refined=int(len(voxel_results)),
        outputs={str(results_dir): ""},
        metrics={"scan_mode": "pf",
                 "n_voxels_processed": len(voxel_results),
                 "position_mode": cfg.position_mode,
                 "mode": cfg.mode},
    )
