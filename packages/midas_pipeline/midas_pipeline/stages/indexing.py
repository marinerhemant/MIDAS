"""Stage: indexing.

Two paths, one orchestrator:

- **PF mode** (``scan_mode='pf'``): invokes ``midas_index.Indexer.run_scanning``
  on the per-voxel grid from ``positions.csv``. Writes the consolidated
  ``Output/IndexBest_all.bin`` consumed by ``find_grains`` and refinement.
- **FF mode** (``scan_mode='ff'``): shells out to ``python -m midas_index``
  with the standard FF arguments (matches ``midas-ff-pipeline.stages.index``
  byte-for-byte). Produces ``Output/IndexBest.bin`` + ``IndexBestFull.bin``.

Both modes ultimately invoke the same ``midas-index`` kernels — that is
the single-source contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import IndexResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _run_ff(ctx: StageContext) -> StageResult:
    """FF (single-scan) indexing — shell out to ``python -m midas_index``.

    Same arguments as ``midas_ff_pipeline.stages.index.run`` so the FF
    parity gate is preserved bit-for-bit.
    """
    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    paramstest = layer_dir / "paramstest.txt"
    spots_to_index = layer_dir / "SpotsToIndex.csv"
    if not paramstest.exists() or not spots_to_index.exists():
        LOG.info("indexing(FF): missing paramstest or SpotsToIndex.csv → skip.")
        return stub_run("indexing", ctx)

    n_seeds = sum(1 for line in spots_to_index.open() if line.strip())
    out_dir = layer_dir / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx.config.indexer_backend == "c-omp":
        from midas_index import backend_c
        if not backend_c.available():
            raise RuntimeError(
                f"indexer_backend='c-omp' but the C binary is not built. "
                f"Re-install midas-index with OpenMP, or set "
                f"indexer_backend='python'. (looked for "
                f"{backend_c.binary_path()})"
            )
        cmd = [
            str(backend_c.binary_path()),
            str(paramstest),
            "0",                               # block_nr
            "1",                               # n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
        ]
    else:
        cmd = [
            sys.executable, "-m", "midas_index",
            str(paramstest),
            "0",                               # block_nr
            "1",                               # n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
            "--device", ctx.config.device,
            "--dtype", ctx.config.dtype,
            "--group-size", str(ctx.config.indexer_group_size),
        ]
    LOG.info("indexing(FF, %s): %s", ctx.config.indexer_backend, " ".join(cmd))
    log_dir = Path(ctx.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "indexing_out.csv").open("w") as out_fp, \
         (log_dir / "indexing_err.csv").open("w") as err_fp:
        subprocess.run(
            cmd, cwd=str(layer_dir), check=True,
            stdout=out_fp, stderr=err_fp,
        )

    finished = time.time()
    index_best = out_dir / "IndexBest.bin"
    if not index_best.exists():
        index_best = layer_dir / "IndexBest.bin"
    index_best_full = index_best.with_name("IndexBestFull.bin")
    n_indexed = 0
    if index_best.exists():
        arr = np.fromfile(index_best, dtype=np.float64)
        if arr.size % 15 == 0:
            arr = arr.reshape(-1, 15)
            n_indexed = int((arr[:, 14] > 0).sum())
    LOG.info("indexing(FF): %d / %d seeds with non-zero data",
             n_indexed, n_seeds)
    return IndexResult(
        stage_name="indexing",
        started_at=started, finished_at=finished, duration_s=finished - started,
        index_best_bin=str(index_best),
        index_best_all_bin="",
        n_voxels_indexed=0,
        outputs={str(index_best): "", str(index_best_full): ""},
        metrics={"scan_mode": "ff",
                 "n_seeds_attempted": n_seeds,
                 "n_seeds_indexed": n_indexed},
    )


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        return _run_ff(ctx)

    # PF (scanning) path follows.
    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    out_dir = layer_dir / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "IndexBest_all.bin"

    paramstest = layer_dir / "paramstest.txt"
    positions_csv = layer_dir / "positions.csv"
    # Soft fail: when upstream stages haven't produced their artefacts
    # (e.g. running a smoke-test or partial pipeline), skip cleanly so
    # the orchestrator can continue. Hard errors only fire from inside
    # the indexer body once we know we *should* be indexing.
    if not paramstest.exists() or not positions_csv.exists():
        LOG.info("indexing(PF): missing paramstest or positions.csv → skip.")
        return stub_run("indexing", ctx)

    LOG.info("indexing(PF): paramstest=%s positions=%s out=%s",
             paramstest, positions_csv, out_path)

    # Lazy-import so FF runs that never touch this stage don't pay the
    # midas-index import cost.
    from midas_index.indexer import Indexer

    scan_positions = np.loadtxt(positions_csv, dtype=np.float64).reshape(-1)
    n_scans = int(scan_positions.size)
    if n_scans < 2:
        raise ValueError(
            f"indexing(PF): positions.csv has {n_scans} entries; "
            "scan mode needs n_scans >= 2."
        )

    # Change to layer_dir so load_observations resolves hkls.csv etc.
    cwd0 = Path.cwd()
    os.chdir(layer_dir)
    try:
        ind = Indexer.from_param_file(paramstest, device=ctx.config.device,
                                      dtype=ctx.config.dtype)
        ind.params.multi_solution_output = True
        ind.params.friedel_symmetric_scan_filter = (
            ctx.config.scan.friedel_symmetric_scan_filter
        )
        if ctx.config.scan.scan_pos_tol_um > 0:
            ind.params.scan_pos_tol_um = ctx.config.scan.scan_pos_tol_um
        elif ctx.config.scan.beam_size_um > 0:
            ind.params.scan_pos_tol_um = ctx.config.scan.beam_size_um / 2.0
        ind.params.OutputFolder = str(layer_dir)

        # P6/P8: soft beam attribution.  Build the weight fn from config and
        # attach to the Indexer; the IndexerContext picks it up in
        # run_scanning() and forwards via scan_kwargs() to compare_spots.
        soft_cfg = ctx.config.soft_attribution
        if soft_cfg.enable:
            from midas_index.compute.soft_attribution import (
                soft_gaussian_fn, soft_top_hat_fn,
            )
            fwhm = soft_cfg.fwhm_um or ctx.config.scan.beam_size_um
            if soft_cfg.profile == "gaussian":
                fn = soft_gaussian_fn(
                    fwhm_um=fwhm, truncate_at=soft_cfg.truncate_at_um,
                )
            elif soft_cfg.profile in ("tophat", "tophat-ramp"):
                fn = soft_top_hat_fn(
                    beam_width_um=fwhm,
                    fall_off_um=soft_cfg.tophat_fall_off_um,
                )
            else:
                raise ValueError(
                    f"unknown soft_attribution.profile={soft_cfg.profile!r}"
                )
            ind.soft_beam_weight_fn = fn

        # c-omp backend skips Python-side observation loading (the C binary
        # mmaps the files itself), but for the python path we need them in
        # memory.
        if ctx.config.indexer_backend == "python":
            ind.load_observations(cwd=layer_dir)
        n_processed = ind.run_scanning(
            scan_positions=scan_positions,
            out_path=out_path,
            num_procs=ctx.config.n_cpus,
            seed_group_size=ctx.config.indexer_group_size,
            backend=ctx.config.indexer_backend,
            paramstest_path=paramstest,
        )
    finally:
        os.chdir(cwd0)

    finished = time.time()
    return IndexResult(
        stage_name="indexing",
        started_at=started, finished_at=finished, duration_s=finished - started,
        index_best_bin="",
        index_best_all_bin=str(out_path),
        n_voxels_indexed=int(n_processed),
        outputs={str(out_path): ""},
        metrics={"scan_mode": "pf", "n_voxels": n_processed,
                 "n_scans": n_scans},
    )
