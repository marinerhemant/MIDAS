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

from .._logging import LOG
from ..results import RefineResult, StageResult
from ._base import StageContext
from ._stub import stub_run


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

    # The 2D 'pixel' loss is removed (it omitted omega and gave poor,
    # under-determined fits); refinement always uses a full 3D / angular loss.
    loss = ctx.config.refinement.loss

    # c-omp backend writes its IndexBest*_all.bin into <layer_dir>/Output; hand
    # fit-grain the matching paramstest so it reads them from there.
    fit_paramstest = paramstest
    if ctx.config.indexer_backend == "c-omp":
        from ._comp_params import comp_backend_paramstest
        fit_paramstest = comp_backend_paramstest(paramstest, layer_dir)

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
        ]
        if ctx.config.refinement.mode:
            cmd += ["--mode", ctx.config.refinement.mode]
        LOG.info("refinement(FF): %s", " ".join(cmd))
        with (log_dir / "refinement_out.csv").open("w") as out_fp, \
             (log_dir / "refinement_err.csv").open("w") as err_fp:
            subprocess.run(
                cmd, cwd=str(layer_dir), check=True,
                stdout=out_fp, stderr=err_fp,
            )

    finished = time.time()
    orient_pos_fit = results_dir / "OrientPosFit.bin"
    n_grains_refined = 0
    if orient_pos_fit.exists():
        n_grains_refined = orient_pos_fit.stat().st_size // 8
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
        # auto-detected (positions.csv > 1 row), position FIXED to the voxel
        # grid. Reads consolidated IndexBest_all.bin. NB: downstream
        # consolidation_pf must read midas_fitgrain's per-voxel output format.
        from midas_fit_grain import backend_c
        if not backend_c.available():
            raise RuntimeError(
                "refine_backend='c-omp' but the midas_fitgrain binary is not "
                "available. Re-install midas-fit-grain with an OpenMP toolchain, "
                "or use --refine-backend python."
            )
        spots_to_index = layer_dir / "SpotsToIndex.csv"
        n_vox = sum(1 for ln in spots_to_index.open() if ln.strip()) \
            if spots_to_index.exists() else 0
        log_dir = Path(ctx.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("refinement(PF, c-omp): %s  [%d voxels]",
                 backend_c.binary_path(), n_vox)
        proc = backend_c.run_refiner(
            paramstest, block_nr=0, n_blocks=1, n_work=n_vox,
            num_procs=ctx.config.n_cpus, cwd=layer_dir,
        )
        (log_dir / "refinement_out.csv").write_bytes(proc.stdout or b"")
        (log_dir / "refinement_err.csv").write_bytes(proc.stderr or b"")
        if proc.returncode != 0:
            raise RuntimeError(
                f"midas_fitgrain (c-omp PF) exited {proc.returncode}; see "
                f"{log_dir / 'refinement_err.csv'}"
            )
        finished = time.time()
        return RefineResult(
            stage_name="refinement",
            started_at=started, finished_at=finished, duration_s=finished - started,
            orient_pos_fit_bin="", results_dir=str(results_dir),
            n_grains_refined=0, n_voxels_refined=int(n_vox),
            outputs={str(results_dir): ""},
            metrics={"scan_mode": "pf", "refine_backend": "c-omp",
                     "n_voxels_processed": n_vox},
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
