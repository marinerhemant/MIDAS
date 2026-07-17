"""Stage: hkl.

For PF mode: iterate over per-scan ``.MIDAS.zip`` archives and emit
``hkls.csv`` into each scan dir via
``midas_hkls.zarr_compat.generate_hkls_from_zarr`` (a Python port of
the legacy ``GetHKLListZarr`` C binary). Also copies the layer-shared
``hkls.csv`` up to ``layer_dir`` so downstream stages (indexing,
refinement, find_grains) can read it without per-scan lookups.

For FF mode: single-zip path is identical — generate once into
``layer_dir``.

No-op if every required ``hkls.csv`` already exists (resume-friendly).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


def run(ctx: StageContext) -> StageResult:
    started = time.time()

    try:
        from midas_hkls.zarr_compat import generate_hkls_from_zarr
    except ImportError as e:
        LOG.warning("hkl: midas_hkls not importable (%s); skipping.", e)
        return stub_run("hkl", ctx)

    if ctx.is_pf:
        return _run_pf(ctx, started, generate_hkls_from_zarr)
    return _run_ff(ctx, started, generate_hkls_from_zarr)


def _run_ff(ctx: StageContext, started: float, gen_fn) -> StageResult:
    """FF single-zip: one HKL list at the layer dir."""
    layer_dir = ctx.layer_dir
    out_csv = layer_dir / "hkls.csv"
    zip_path = _resolve_ff_zip(ctx)
    if zip_path is None or not zip_path.exists():
        LOG.info("hkl(FF): no zarr/zip available at %s; skip.", zip_path)
        return stub_run("hkl", ctx)
    if out_csv.exists():
        LOG.info("hkl(FF): %s already exists; skip.", out_csv)
        return _result(started, [out_csv])
    gen_fn(zip_path, result_folder=layer_dir)
    LOG.info("hkl(FF): wrote %s", out_csv)
    return _result(started, [out_csv])


def _run_pf(ctx: StageContext, started: float, gen_fn) -> StageResult:
    """PF per-scan: write hkls.csv into each scan dir + share at layer dir."""
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
            f"hkl(PF): scan discovery failed: {e}. Refusing to "
            "soft-skip in PF mode."
        ) from e
    except ValueError as e:
        # Incomplete Parameters.txt (no FileStem / StartFileNrFirstLayer):
        # tolerated for smoke/partial runs; the missing-positions case
        # above is the silent-corruption one.
        LOG.warning("hkl(PF): scan discovery failed (%s); skip.", e)
        return stub_run("hkl", ctx)

    written: list[Path] = []
    skipped = 0
    representative: Optional[Path] = None
    for s in scans:
        s.scan_dir.mkdir(parents=True, exist_ok=True)
        if s.hkls_csv.exists():
            skipped += 1
            representative = s.hkls_csv
            continue
        if not s.zip_path.exists():
            LOG.warning("hkl(PF): scan %d zip missing at %s; skip scan.",
                        s.scan_nr, s.zip_path)
            continue
        try:
            gen_fn(s.zip_path, result_folder=s.scan_dir)
        except Exception as e:
            LOG.warning("hkl(PF): scan %d generate_hkls_from_zarr failed "
                        "(%s); skipping scan.", s.scan_nr, e)
            continue
        written.append(s.hkls_csv)
        representative = s.hkls_csv

    # Promote one HKL csv to layer_dir so downstream stages have a
    # single canonical hkls.csv (matches pf_MIDAS.py:909).
    layer_hkls = layer_dir / "hkls.csv"
    if not layer_hkls.exists() and representative is not None and representative.exists():
        shutil.copy2(representative, layer_hkls)
        written.append(layer_hkls)

    LOG.info("hkl(PF): %d new + %d cached scan-dir hkls.csv (of %d scans); "
             "layer hkls.csv: %s",
             len(written) - (1 if layer_hkls in written else 0),
             skipped, len(scans),
             "ok" if layer_hkls.exists() else "missing")
    return _result(started, written)


def _resolve_ff_zip(ctx: StageContext) -> Optional[Path]:
    """For FF mode, the zarr is the single ``--zarr`` arg or the layer dir's."""
    if ctx.config.zarr_path:
        return Path(ctx.config.zarr_path)
    for p in ctx.layer_dir.glob("*.MIDAS.zip"):
        return p
    return None


def _result(started: float, written: list[Path]) -> StageResult:
    finished = time.time()
    return StageResult(
        stage_name="hkl",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"hkls": [str(p) for p in written]},
        metrics={"n_written": len(written)},
    )
