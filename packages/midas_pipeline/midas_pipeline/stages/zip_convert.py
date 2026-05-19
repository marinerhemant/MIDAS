"""Stage: zip_convert.

Build (or verify) one MIDAS-format ``.MIDAS.zip`` per scan position
(PF) or for the single layer (FF). Wraps ``utils/ffGenerateZipRefactor.py``,
which is already pure Python — we shell out to it to keep its
self-contained CLI surface and rich raw-frame handling (TIFF / CBF /
GE / HDF5) rather than re-engineering a new entry.

Resume-friendly: if every required ``.MIDAS.zip`` already exists, the
stage is a no-op. PF datasets like Wenxi that ship pre-built zips
fall through this stage in ~50 ms (just stat + log).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run

_FF_GEN_ZIP_REL = "utils/ffGenerateZipRefactor.py"


def run(ctx: StageContext) -> StageResult:
    started = time.time()
    if ctx.is_pf:
        return _run_pf(ctx, started)
    return _run_ff(ctx, started)


def _run_pf(ctx: StageContext, started: float) -> StageResult:
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
        )
    except (FileNotFoundError, ValueError) as e:
        LOG.warning("zip_convert(PF): scan discovery failed (%s); skip.", e)
        return stub_run("zip_convert", ctx)

    n_present = 0
    n_built = 0
    n_failed = 0
    midas_root = _midas_root()

    for s in scans:
        if s.zip_path.exists():
            n_present += 1
            continue
        if not cfg.convert_files:
            LOG.warning("zip_convert(PF): scan %d zip missing (%s) and "
                        "--convert-files is off; skip.",
                        s.scan_nr, s.zip_path)
            n_failed += 1
            continue
        s.scan_dir.mkdir(parents=True, exist_ok=True)
        if not _generate_zip(
            midas_root=midas_root,
            param_file=cfg.params_file,
            scan_dir=s.scan_dir,
            scan_within_layer=s.scan_nr,
            num_frame_chunks=cfg.num_frame_chunks,
            preproc_thresh=cfg.preproc_thresh,
            num_files_per_scan=cfg.num_files_per_scan,
        ):
            n_failed += 1
            continue
        if s.zip_path.exists():
            n_built += 1
        else:
            LOG.warning("zip_convert(PF): scan %d generator returned ok "
                        "but %s is missing.", s.scan_nr, s.zip_path)
            n_failed += 1

    LOG.info("zip_convert(PF): %d pre-existing + %d built + %d failed "
             "(of %d scans)", n_present, n_built, n_failed, len(scans))
    finished = time.time()
    return StageResult(
        stage_name="zip_convert",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"zips": [str(s.zip_path) for s in scans if s.zip_path.exists()]},
        metrics={"n_present": n_present, "n_built": n_built,
                 "n_failed": n_failed, "n_scans": len(scans)},
    )


def _run_ff(ctx: StageContext, started: float) -> StageResult:
    """FF: single zip; build if missing and --convert-files is on."""
    cfg = ctx.config
    layer_dir = ctx.layer_dir
    if cfg.zarr_path and Path(cfg.zarr_path).exists():
        LOG.info("zip_convert(FF): using --zarr %s", cfg.zarr_path)
        return _ff_result(started, [Path(cfg.zarr_path)], present=1, built=0, failed=0)
    existing = list(layer_dir.glob("*.MIDAS.zip"))
    if existing:
        LOG.info("zip_convert(FF): found %s", existing[0])
        return _ff_result(started, existing, present=1, built=0, failed=0)
    if not cfg.convert_files:
        LOG.info("zip_convert(FF): no zip in %s and --convert-files off; skip.",
                 layer_dir)
        return stub_run("zip_convert", ctx)
    midas_root = _midas_root()
    if not _generate_zip(
        midas_root=midas_root,
        param_file=cfg.params_file,
        scan_dir=layer_dir,
        scan_within_layer=ctx.layer_nr,
        num_frame_chunks=cfg.num_frame_chunks,
        preproc_thresh=cfg.preproc_thresh,
        num_files_per_scan=cfg.num_files_per_scan,
    ):
        return _ff_result(started, [], present=0, built=0, failed=1)
    new_zips = list(layer_dir.glob("*.MIDAS.zip"))
    return _ff_result(started, new_zips,
                      present=0, built=len(new_zips), failed=0 if new_zips else 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _midas_root() -> Path:
    """Best-effort MIDAS install root (env var > package parent chain)."""
    if env := os.environ.get("MIDAS_INSTALL_DIR"):
        return Path(env)
    # midas_pipeline lives at <root>/packages/midas_pipeline/midas_pipeline/
    # so root = parents[3].
    import midas_pipeline
    return Path(midas_pipeline.__file__).resolve().parents[3]


def _generate_zip(
    *,
    midas_root: Path,
    param_file: str,
    scan_dir: Path,
    scan_within_layer: int,
    num_frame_chunks: Optional[int] = None,
    preproc_thresh: Optional[int] = None,
    num_files_per_scan: int = 1,
) -> bool:
    """Invoke ``utils/ffGenerateZipRefactor.py`` for one scan/layer.

    The script's CLI takes ``-LayerNr``, which inside pf_MIDAS.py is
    overloaded as the scan-within-layer index (1..n_scans). We mirror
    that mapping here.
    """
    script = midas_root / _FF_GEN_ZIP_REL
    if not script.exists():
        LOG.warning("zip_convert: missing %s — cannot generate zip.", script)
        return False
    cmd = [
        sys.executable, str(script),
        "-resultFolder", str(scan_dir),
        "-paramFN", str(param_file),
        "-LayerNr", str(scan_within_layer),
    ]
    if num_frame_chunks is not None and num_frame_chunks > 0:
        cmd += ["-numFrameChunks", str(num_frame_chunks)]
    if preproc_thresh is not None and preproc_thresh > 0:
        cmd += ["-preProcThresh", str(preproc_thresh)]
    if num_files_per_scan and num_files_per_scan > 1:
        cmd += ["-numFilesPerScan", str(num_files_per_scan)]
    log_dir = scan_dir / "midas_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_dir / "ZipOut.txt", "w") as f_out, \
             open(log_dir / "ZipErr.txt", "w") as f_err:
            res = subprocess.run(cmd, cwd=str(scan_dir),
                                 stdout=f_out, stderr=f_err, check=False)
        if res.returncode != 0:
            LOG.warning("zip_convert: ffGenerateZipRefactor exit=%d for %s",
                        res.returncode, scan_dir)
            return False
        return True
    except Exception as e:
        LOG.warning("zip_convert: ffGenerateZipRefactor crashed for %s: %s",
                    scan_dir, e)
        return False


def _ff_result(started: float, zips, *, present: int, built: int, failed: int):
    finished = time.time()
    return StageResult(
        stage_name="zip_convert",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"zips": [str(z) for z in zips]},
        metrics={"n_present": present, "n_built": built, "n_failed": failed},
    )
