"""Stage: indexing.

Two paths, one orchestrator:

- **PF mode** (``scan_mode='pf'``): invokes ``midas_index.Indexer.run_scanning``
  on the per-voxel grid from ``positions.csv``. Writes the consolidated
  ``Output/IndexBest_all.bin`` consumed by ``find_grains`` and refinement.
- **FF mode** (``scan_mode='ff'``): shells out to ``python -m midas_index``
  (or the bundled ``midas_indexer`` for ``--indexer-backend c-omp``) with the
  standard FF arguments.

Both modes ultimately invoke the same ``midas-index`` kernels — that is
the single-source contract.

Seed-file formats, FF mode — there are TWO, and this stage must read both:

===============================  ==========================================
writer                           files
===============================  ==========================================
classical C ``IndexerOMP``       ``IndexBest.bin`` + ``IndexBestFull.bin``
``midas_index`` (python)         ``IndexBest_all.bin`` + ``IndexKey_all.bin``
``midas_indexer`` (c-omp)          + ``IndexBest_IDs_all.bin``
                                   + ``IndexBest_weights_all.bin``
===============================  ==========================================

Measured on the datasetA Ni layer: ``py_run`` (python backend, FF) wrote only
the consolidated family, ``c_run`` (classical ``IndexerOMP``) only the legacy
pair. **The consolidated family is the current FF contract for both modern
backends**; the legacy pair now comes only from the classical binary. This
docstring used to claim FF "Produces Output/IndexBest.bin + IndexBestFull.bin",
which sent a bug report (issues/68) chasing a non-existent branch bug in
``IndexerUnified.c`` — the writer was right, the reader and the docs were wrong.

Both consumers already handle both: ``midas_fit_grain.driver`` adapts the
consolidated family to the legacy seed shapes, and ``FitUnified.c`` probes
``IndexBest_all.bin`` before falling back to the legacy pair.
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
from ._base import run_checked_streamed, StageContext
from ._stub import stub_run


def _seed_grains_file(ctx: StageContext, layer_dir: Path) -> "Path | None":
    """Resolve the Grains.csv-format seed file for the active seeding mode.

    - ``ff``        → the user-supplied ``SeedingConfig.grains_file``.
    - ``merged-ff`` → ``<layer_dir>/Grains.csv`` synthesised by the seeding
                      stage (ff_index → ProcessGrains).
    - ``unseeded``  → ``None``.
    """
    mode = getattr(ctx.config.seeding, "mode", "unseeded")
    if mode == "ff":
        gf = getattr(ctx.config.seeding, "grains_file", None)
        return Path(gf) if gf else None
    if mode == "merged-ff":
        cand = layer_dir / "Grains.csv"
        return cand if cand.exists() else None
    return None


def _ensure_grains_seed_in_paramstest(
    ctx: StageContext, paramstest: Path, layer_dir: Path
) -> None:
    """Append a ``GrainsFile <path>`` line to *paramstest* when seeding is
    active and it is not already present. No-op for unseeded runs."""
    seed = _seed_grains_file(ctx, layer_dir)
    if seed is None:
        return
    if not seed.exists():
        LOG.warning("indexing(PF): seed grains file %s not found; "
                    "indexer will run UNSEEDED (full grid).", seed)
        return
    existing = paramstest.read_text() if paramstest.exists() else ""
    for ln in existing.splitlines():
        if ln.strip().split()[:1] == ["GrainsFile"]:
            return                                    # already wired
    with paramstest.open("a") as fh:
        if existing and not existing.endswith("\n"):
            fh.write("\n")
        fh.write(f"GrainsFile {seed.resolve()}\n")
    LOG.info("indexing(PF): seeded from %s (GrainsFile wired into paramstest)",
             seed)


def _count_indexed_seeds(out_dir: Path, legacy_path: Path):
    """How many seeds actually got a solution, whichever backend wrote them.

    Returns ``(n_indexed, path_counted_from)``; the path is ``None`` when no
    recognised seed file exists at all.

    The two backends write different families and this must handle both:
      * legacy / python  ``IndexBest.bin``      (nSeeds, 15), col 14 = n_observed
      * c-omp            ``IndexBest_all.bin``  int32 nVox, int32 nSol[nVox],
                                                int64 off[nVox], then (nSol, 16)
                                                records with col 15 = n_observed

    Counting only the legacy name reported "0 / N seeds with non-zero data" on
    every c-omp run (observed on the datasetA Ni layer: 0 / 56196 logged while
    the run went on to produce 11475 grains). A diagnostic that always reads
    zero cannot warn when the value is genuinely zero, which is exactly the
    failure it exists to catch.
    """
    if legacy_path.exists():
        arr = np.fromfile(legacy_path, dtype=np.float64)
        if arr.size and arr.size % 15 == 0:
            return int((arr.reshape(-1, 15)[:, 14] > 0).sum()), legacy_path

    consolidated = out_dir / "IndexBest_all.bin"
    if consolidated.exists():
        raw = consolidated.read_bytes()
        if len(raw) >= 4:
            n_vox = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
            head = 4 + 4 * n_vox + 8 * n_vox
            n_sol = np.frombuffer(raw[4:4 + 4 * n_vox], dtype=np.int32)
            vals = np.frombuffer(raw[head:], dtype=np.float64)
            if vals.size % 16 == 0:
                vals = vals.reshape(-1, 16)
                n_indexed, pos = 0, 0
                for n in n_sol:
                    if n > 0 and vals[pos:pos + n, 15].max() > 0:
                        n_indexed += 1
                    pos += int(n)
                return n_indexed, consolidated
    return 0, None


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

    # Clear IndexBest outputs left by a previous run, from BOTH the c-omp
    # (Output/) and python (bare layer_dir) conventions, before regenerating
    # them. process_grains' backend-agnostic readers resolve subfolder-first
    # then fall back to the bare path; without this a leftover file from the
    # *other* backend's location would shadow (or mix with) this run's fresh
    # output and silently feed stale records downstream.
    for _stale in (out_dir / "IndexBest.bin", out_dir / "IndexBestFull.bin",
                   layer_dir / "IndexBest.bin", layer_dir / "IndexBestFull.bin"):
        _stale.unlink(missing_ok=True)

    if ctx.config.indexer_backend == "c-omp":
        from midas_index import backend_c
        from ._comp_params import comp_backend_paramstest
        if not backend_c.available():
            raise RuntimeError(
                f"indexer_backend='c-omp' but the C binary is not built. "
                f"Re-install midas-index with OpenMP, or set "
                f"indexer_backend='python'. (looked for "
                f"{backend_c.binary_path()})"
            )
        # The C binary locates binned inputs via dirname(OutputFolder) and
        # emits into OutputFolder; hand it OutputFolder=<layer_dir>/Output.
        comp_paramstest = comp_backend_paramstest(paramstest, layer_dir)
        cmd = [
            str(backend_c.binary_path()),
            str(comp_paramstest),
            "0",                               # block_nr
            "1",                               # n_blocks
            str(n_seeds),
            str(ctx.config.n_cpus),
        ]
    else:
        # The python backend reads its input folders out of the paramstest, so
        # the keys must name THIS layer dir -- they arrive naming whichever
        # machine built the zarr. See localised_paramstest.
        from ._comp_params import localised_paramstest
        py_paramstest = localised_paramstest(paramstest, layer_dir)
        if py_paramstest != paramstest:
            LOG.info("indexing(FF, python): paramstest folder keys did not name "
                     "this layer; using %s", py_paramstest.name)
        cmd = [
            sys.executable, "-m", "midas_index",
            str(py_paramstest),
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
        run_checked_streamed(
            cmd, cwd=layer_dir, out_fp=out_fp, err_fp=err_fp,
            line_cb=(ctx.progress.feed_line if ctx.progress else None),
        )

    finished = time.time()
    index_best = out_dir / "IndexBest.bin"
    if not index_best.exists():
        index_best = layer_dir / "IndexBest.bin"

    n_indexed, counted_from = _count_indexed_seeds(out_dir, index_best)
    if counted_from is None:
        # The indexer exited 0 and wrote nothing we recognise. That is a broken
        # contract, not a scientific result, and it must stop the run here:
        # letting it through means refinement writes an empty OrientPosFit.bin
        # and process-grains dies on `cannot mmap an empty file` two stages
        # later, in a package with no visibility of the real fault
        # (github.com/marinerhemant/MIDAS issues/68).
        #
        # Deliberately distinct from "the file exists and says zero", below,
        # which IS a legitimate if disappointing outcome and only warns.
        raise RuntimeError(
            f"indexing(FF): the indexer exited 0 but produced no recognisable "
            f"seed file in {out_dir}. Expected IndexBest.bin (legacy "
            f"IndexerOMP) or IndexBest_all.bin (python and c-omp backends). "
            f"Nothing downstream can run; see the indexing log in {log_dir}."
        )
    if n_indexed == 0:
        LOG.warning(
            "indexing(FF): 0 / %d seeds indexed, read from %s. Nothing "
            "downstream can succeed; check the indexing log in %s.",
            n_seeds, counted_from.name, log_dir)
    else:
        LOG.info("indexing(FF): %d / %d seeds with non-zero data (from %s)",
                 n_indexed, n_seeds, counted_from.name)

    # Report the seed files that ACTUALLY exist. FF used to hardcode the legacy
    # pair into both the result fields and `outputs`, so a python- or c-omp-
    # backed run advertised `index_best_bin=<...>/IndexBest.bin` for a file that
    # was never written, left `index_best_all_bin` empty for the file that was,
    # and listed two non-existent paths as its outputs. Anything consuming the
    # manifest — provenance, resume, a downstream stage resolving its input —
    # was reading fiction (github.com/marinerhemant/MIDAS issues/68).
    consolidated = counted_from.name == "IndexBest_all.bin"
    if consolidated:
        index_best_bin = ""
        index_best_all_bin = str(counted_from)
        # the c-omp / python FF family; only the first two are always present
        family = [counted_from,
                  counted_from.with_name("IndexKey_all.bin"),
                  counted_from.with_name("IndexBest_IDs_all.bin"),
                  counted_from.with_name("IndexBest_weights_all.bin")]
        outputs = {str(p): "" for p in family if p.exists()}
    else:
        index_best_bin = str(counted_from)
        index_best_all_bin = ""
        family = [counted_from, counted_from.with_name("IndexBestFull.bin")]
        outputs = {str(p): "" for p in family if p.exists()}

    return IndexResult(
        stage_name="indexing",
        started_at=started, finished_at=finished, duration_s=finished - started,
        index_best_bin=index_best_bin,
        index_best_all_bin=index_best_all_bin,
        # IndexResult declares these as fields and FF only ever wrote them into
        # `metrics`, so `result.n_seeds_indexed` read 0 on every FF run no
        # matter what the indexer did — the same "advertised value is fiction"
        # defect as the paths above.
        n_seeds_attempted=int(n_seeds),
        n_seeds_indexed=int(n_indexed),
        n_voxels_indexed=0,
        outputs=outputs,
        metrics={"scan_mode": "ff",
                 "n_seeds_attempted": n_seeds,
                 "n_seeds_indexed": n_indexed,
                 "seed_format": "consolidated" if consolidated else "legacy"},
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
    # P0-2: a missing positions.csv in PF mode is a HARD error (the
    # pipeline materializes it at layer setup — absence means the run is
    # broken, and soft-skipping made whole runs exit 0 doing nothing).
    if not positions_csv.exists():
        raise RuntimeError(
            f"indexing(PF): missing {positions_csv}. Refusing to soft-skip "
            "in PF mode — positions.csv is materialized at layer setup; "
            "if driving stages manually, pre-seed it (one Y per line, "
            "acquisition order)."
        )
    # Soft fail on missing upstream artefacts other than positions
    # (e.g. running a smoke-test or partial pipeline): skip cleanly so
    # the orchestrator can continue. Hard errors only fire from inside
    # the indexer body once we know we *should* be indexing.
    if not paramstest.exists():
        LOG.info("indexing(PF): missing paramstest.txt → skip.")
        return stub_run("indexing", ctx)

    # Wire the FF/merged-FF seed grains into the paramstest the indexer reads.
    # The ``seeding`` stage produces ``UniqueOrientations.csv`` (used by the
    # python backend + find_grains), but the c-omp binary keys its seeded PF
    # path off a ``GrainsFile <path>`` line in its paramstest (isGrainsInput=1,
    # DoIndexing_Seeded). Without it the C binary silently runs the FULL
    # orientation grid — indistinguishable from unseeded, and just as slow.
    # ``_emit_c_omp_paramstest`` copies arbitrary lines through, so injecting it
    # here reaches the binary. The seed file must be Grains.csv format (ID + 9
    # OM in cols 1..9), NOT UniqueOrientations.csv (OM in cols 5..13).
    _ensure_grains_seed_in_paramstest(ctx, paramstest, layer_dir)

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
            # Live voxel counts into progress.txt. This is the longest stage in
            # a PF run by a wide margin (8 h+ on a dense s1 layer) and was
            # completely silent: the C emits `progress: N/M voxels` ~200 times,
            # but nothing consumed it, so the file sat at "indexing RUNNING
            # 0.0s" for the whole run.
            line_cb=(ctx.progress.feed_line if ctx.progress else None),
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
