"""Indexing via ``midas-index``.

Reads the merged ``paramstest.txt`` + ``Spots.bin`` and writes
``Output/IndexBest.bin`` + ``Output/IndexBestFull.bin``.

The pipeline default is fp64 + group_size=4 for parity with C
IndexerOMP. Override via PipelineConfig.

Multi-GPU sharding is supported via ``PipelineConfig.shard_gpus``
(``--shard-gpus 0,1`` on the CLI). When set, the indexer is fanned
out across the named devices, each handling a disjoint slice of
seeds via midas-index's existing ``--block-nr/--n-blocks`` shard
protocol. All shards pwrite into the same IndexBest.bin file at
non-overlapping byte offsets — block 0 truncates + zero-fills the
full output ahead of writes, the rest pwrite into pre-allocated slots.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ._base import StageContext, env_for_index_refine, run_subprocess
from .._logging import LOG, stage_timer
from ..results import IndexResult


def _parse_shard_gpus(raw: str | None) -> list[int]:
    """Parse the ``shard_gpus`` config string into a list of CUDA indices.

    Empty / None → ``[]`` (single-GPU path). Anything else is a
    comma-separated list of integer device indices, e.g. ``"0,1"``.
    """
    if not raw:
        return []
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _resolve_output_dir(paramstest: Path) -> Path:
    """Read ``OutputFolder`` out of the paramstest, falling back to its dir."""
    try:
        for raw in paramstest.read_text().splitlines():
            line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
            if line.startswith("OutputFolder"):
                toks = line.split(None, 1)
                if len(toks) == 2:
                    return Path(toks[1].rstrip(";"))
    except Exception:
        pass
    return paramstest.parent / "Output"


def _preallocate_index_outputs(out_dir: Path, n_total_seeds: int) -> None:
    """Create + ftruncate IndexBest.bin / IndexBestFull.bin to the full size.

    Avoids the multi-shard race in midas-index's ``open_output_files`` where
    a late-finishing shard's ``O_TRUNC`` wipes earlier shards' pwritten
    data. Each shard then runs with ``MIDAS_INDEX_PREALLOCATED=1`` so it
    skips its own truncate/ftruncate.

    Sizes mirror midas_index.io.output:
      IndexBest.bin       n_total * 15 doubles (= 120 bytes per row)
      IndexBestFull.bin   n_total * MAX_N_HKLS * 2 doubles
    """
    from midas_index.io.output import (   # type: ignore
        INDEX_BEST_RECORD_BYTES,
        INDEX_BEST_FULL_RECORD_BYTES,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, record_bytes in (
        ("IndexBest.bin", INDEX_BEST_RECORD_BYTES),
        ("IndexBestFull.bin", INDEX_BEST_FULL_RECORD_BYTES),
    ):
        path = out_dir / name
        # Open O_TRUNC so we get a clean slate; ftruncate to full size; close.
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, n_total_seeds * record_bytes)
        finally:
            os.close(fd)


def _run_cpu_shards(ctx: StageContext, paramstest: Path, n_seeds: int,
                    n_shards: int) -> None:
    """Spawn N concurrent ``midas-index`` processes over disjoint seed slices.

    Each shard gets ``n_cpus // n_shards`` threads via ``num_procs`` (which
    midas-index passes to ``torch.set_num_threads``). Intra-op threading
    scales poorly past ~16 threads on the small per-seed ops, so 6 × 16
    beats 1 × 96 by 3-5× on real hexagonal data.

    Uses the same pwrite-safety pattern as ``_run_shards`` (the GPU path):
    preallocate IndexBest.bin + IndexBestFull.bin, then have every shard
    run with ``MIDAS_INDEX_PREALLOCATED=1`` so none of them truncate.
    """
    out_dir = _resolve_output_dir(paramstest)
    _preallocate_index_outputs(out_dir, n_seeds)
    LOG.info("  pre-allocated %s/IndexBest.bin + IndexBestFull.bin "
             "(%d slots, %d cpu shards × %d threads)", out_dir, n_seeds,
             n_shards, max(1, ctx.config.n_cpus // n_shards))

    threads_per_shard = max(1, ctx.config.n_cpus // n_shards)
    procs: list[tuple[int, subprocess.Popen]] = []
    for block_nr in range(n_shards):
        env = {
            **os.environ,
            **env_for_index_refine(ctx.config),
            "MIDAS_INDEX_PREALLOCATED": "1",
            # Pin BLAS / OpenMP thread pools to match torch's intra-op pool;
            # without this, MKL and OpenMP each spawn their own pool inside
            # every shard process, leading to massive oversubscription on the
            # box (e.g. 6 procs × 126 threads on a 96-core machine).
            "OMP_NUM_THREADS": str(threads_per_shard),
            "MKL_NUM_THREADS": str(threads_per_shard),
            "OPENBLAS_NUM_THREADS": str(threads_per_shard),
            "NUMEXPR_NUM_THREADS": str(threads_per_shard),
        }
        cmd = [
            sys.executable, "-m", "midas_index",
            str(paramstest),
            str(block_nr),
            str(n_shards),
            str(n_seeds),
            str(threads_per_shard),
            "--device", "cpu",
            "--dtype", ctx.config.dtype,
            "--group-size", str(ctx.config.indexer_group_size),
        ]
        out_path = ctx.log_dir / f"indexing_shard{block_nr}_out.csv"
        err_path = ctx.log_dir / f"indexing_shard{block_nr}_err.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(ctx.layer_dir),
            env=env,
            stdout=open(out_path, "w"),
            stderr=open(err_path, "w"),
        )
        LOG.info("  cpu shard %d/%d (pid=%d)", block_nr, n_shards, proc.pid)
        procs.append((block_nr, proc))

    failures: list[str] = []
    for block_nr, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append(f"cpu shard {block_nr} rc={rc}")
    if failures:
        raise RuntimeError(
            "indexing cpu shards failed: " + "; ".join(failures)
            + f" — see {ctx.log_dir}/indexing_shard*_err.csv"
        )


def _run_shards(ctx: StageContext, paramstest: Path, n_seeds: int,
                shard_gpus: list[int]) -> None:
    """Spawn one ``midas-index`` per GPU and wait for all to finish.

    Multi-shard pwrite-safety: midas-index's ``open_output_files`` truncates
    on block_nr==0, which races destructively against other shards' already-
    pwritten data when shards finish near-simultaneously. We pre-allocate
    the output files here and set ``MIDAS_INDEX_PREALLOCATED=1`` so every
    shard skips its own truncate. All shards just pwrite into their slots.
    """
    n_blocks = len(shard_gpus)
    out_dir = _resolve_output_dir(paramstest)
    _preallocate_index_outputs(out_dir, n_seeds)
    LOG.info("  pre-allocated %s/IndexBest.bin + IndexBestFull.bin "
             "(%d slots)", out_dir, n_seeds)

    procs: list[tuple[int, int, subprocess.Popen]] = []  # (block_nr, gpu_idx, proc)
    cpus_per_shard = max(1, ctx.config.n_cpus // n_blocks)

    for block_nr, gpu_idx in enumerate(shard_gpus):
        env = {
            **os.environ,
            **env_for_index_refine(ctx.config),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_idx),
            "MIDAS_INDEX_PREALLOCATED": "1",
        }
        cmd = [
            sys.executable, "-m", "midas_index",
            str(paramstest),
            str(block_nr),
            str(n_blocks),
            str(n_seeds),
            str(cpus_per_shard),
            "--device", "cuda",                      # always cuda when sharded
            "--dtype", ctx.config.dtype,
            "--group-size", str(ctx.config.indexer_group_size),
        ]
        out_path = ctx.log_dir / f"indexing_shard{block_nr}_out.csv"
        err_path = ctx.log_dir / f"indexing_shard{block_nr}_err.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(ctx.layer_dir),
            env=env,
            stdout=open(out_path, "w"),
            stderr=open(err_path, "w"),
        )
        LOG.info("  shard %d → GPU %d (pid=%d)", block_nr, gpu_idx, proc.pid)
        procs.append((block_nr, gpu_idx, proc))

    failures: list[str] = []
    for block_nr, gpu_idx, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append(f"shard {block_nr} (GPU {gpu_idx}) rc={rc}")
    if failures:
        raise RuntimeError(
            "indexing shards failed: " + "; ".join(failures)
            + f" — see {ctx.log_dir}/indexing_shard*_err.csv"
        )


def run(ctx: StageContext) -> IndexResult:
    started = time.time()

    paramstest = ctx.layer_dir / "paramstest.txt"
    spots_to_index = ctx.layer_dir / "SpotsToIndex.csv"

    if not spots_to_index.exists():
        raise FileNotFoundError(
            f"binning did not produce {spots_to_index} — cannot index"
        )

    n_seeds = sum(1 for _ in spots_to_index.open() if _.strip())
    shard_gpus = _parse_shard_gpus(ctx.config.shard_gpus)

    cpu_shards = int(getattr(ctx.config, "cpu_shards", 1) or 1)

    with stage_timer("indexing"):
        if shard_gpus:
            LOG.info("  multi-GPU shard: %d shards across GPUs %s",
                     len(shard_gpus), shard_gpus)
            _run_shards(ctx, paramstest, n_seeds, shard_gpus)
        elif ctx.config.device == "cpu" and cpu_shards > 1:
            LOG.info("  multi-CPU shard: %d concurrent midas-index procs "
                     "(%d threads each)", cpu_shards,
                     max(1, ctx.config.n_cpus // cpu_shards))
            _run_cpu_shards(ctx, paramstest, n_seeds, cpu_shards)
        else:
            cmd = [
                sys.executable, "-m", "midas_index",
                str(paramstest),
                "0",                                   # block_nr
                "1",                                   # n_blocks
                str(n_seeds),
                str(ctx.config.n_cpus),
                "--device", ctx.config.device,
                "--dtype", ctx.config.dtype,
                "--group-size", str(ctx.config.indexer_group_size),
            ]
            run_subprocess(
                cmd,
                cwd=ctx.layer_dir,
                stdout_path=ctx.log_dir / "indexing_out.csv",
                stderr_path=ctx.log_dir / "indexing_err.csv",
                env=env_for_index_refine(ctx.config),
            )

    finished = time.time()
    # midas-index writes IndexBest.bin / IndexBestFull.bin to OutputFolder
    # (set by transforms stage to layer_dir/Output, otherwise cwd=layer_dir).
    output_dir = ctx.layer_dir / "Output"
    index_best = output_dir / "IndexBest.bin"
    if not index_best.exists():
        index_best = ctx.layer_dir / "IndexBest.bin"
    index_best_full = index_best.with_name("IndexBestFull.bin")
    n_indexed = 0
    if index_best.exists():
        arr = np.fromfile(index_best, dtype=np.float64)
        if arr.size % 15 == 0:
            arr = arr.reshape(-1, 15)
            n_indexed = int((arr[:, 14] > 0).sum())
    LOG.info("  midas-index: %d / %d seeds with non-zero data", n_indexed, n_seeds)

    return IndexResult(
        stage_name="indexing",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={
            str(index_best): "",
            str(index_best_full): "",
        },
        index_best_bin=str(index_best),
        n_seeds_attempted=n_seeds,
        n_seeds_indexed=n_indexed,
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [
        ctx.layer_dir / "IndexBest.bin",
        ctx.layer_dir / "IndexBestFull.bin",
    ]
