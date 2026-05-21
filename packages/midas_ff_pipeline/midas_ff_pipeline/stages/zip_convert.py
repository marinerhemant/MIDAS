"""Raw → ``.MIDAS.zip`` conversion (gap #1).

Delegates to the pip-installed ``midas_zipper.ff_zip`` module
(``python -m midas_zipper.ff_zip``) so the new pipeline can ingest
GE / HDF5 / TIFF / CBF directly without a separate manual step and
without requiring a MIDAS source tree. Runs once per detector before
``hkl``. No-op when the detector already has a valid zarr (``--zarr
<path>`` or detectors.json explicit zarr_path or ``--no-convert``).

The produced zarr path follows the convention used by ``midas_zipper``::

    {result_dir}/LayerNr_<N>/{FileStem}_{fNr:0{Padding}d}.MIDAS.zip
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import LOG, stage_timer
from ..results import StageResult


def _read_param(path: Path, key: str, default: str | None = None) -> str | None:
    if not path.exists():
        return default
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
        if not line:
            continue
        toks = line.split()
        if toks[0] == key and len(toks) >= 2:
            return toks[1].rstrip(";")
    return default


def _zarr_path_for_layer(ctx: StageContext, params_file: Path) -> Path:
    file_stem = _read_param(params_file, "FileStem", "file") or "file"
    padding = int(_read_param(params_file, "Padding", "6") or "6")
    start_fn = int(_read_param(params_file, "StartFileNrFirstLayer", "1") or "1")
    scan_step = int(
        _read_param(params_file, "ScanStep",
                    _read_param(params_file, "NrFilesPerSweep", "1") or "1")
        or "1"
    )
    f_nr = start_fn + (ctx.layer_nr - 1) * scan_step
    return ctx.layer_dir / f"{file_stem}_{f_nr:0{padding}d}.MIDAS.zip"


def run(ctx: StageContext) -> StageResult:
    started = time.time()
    outputs: dict[str, str] = {}

    params_file = Path(ctx.config.params_file)

    # This layer's expected zarr (file number = start_fn + (layer-1)*scan_step).
    # The skip check MUST be against THIS layer's path, not the persisted
    # ``det.zarr_path`` — that points at the PREVIOUS layer's zarr after the
    # first layer in a multi-layer run, which caused every subsequent layer to
    # reuse the first layer's data. Always (re)point det.zarr_path at the
    # per-layer target so the skip branch is safe for downstream stages too.
    layer_target = _zarr_path_for_layer(ctx, params_file)

    # Skip when conversion isn't wanted.
    if not ctx.config.convert_files:
        LOG.info("zip_convert: --no-convert set, skipping")
        for det in ctx.detectors:
            det.zarr_path = str(layer_target)
        return StageResult(stage_name="zip_convert",
                           started_at=started, finished_at=started,
                           duration_s=0.0,
                           metrics={"skipped": True})

    # Skip only if THIS layer's zarr already exists.
    if layer_target.exists():
        LOG.info("zip_convert: layer %d zarr already exists, skipping (%s)",
                 ctx.layer_nr, layer_target)
        for det in ctx.detectors:
            det.zarr_path = str(layer_target)
        return StageResult(stage_name="zip_convert",
                           started_at=started, finished_at=started,
                           duration_s=0.0,
                           metrics={"skipped": True})

    with stage_timer("zip_convert"):
        for det in ctx.detectors:
            target = _zarr_path_for_layer(ctx, params_file)
            cmd = [
                sys.executable, "-m", "midas_zipper.ff_zip",
                "-resultFolder", str(ctx.layer_dir),
                "-paramFN", str(params_file),
                "-LayerNr", str(ctx.layer_nr),
            ]
            if ctx.config.num_frame_chunks != -1:
                cmd += ["-numFrameChunks", str(ctx.config.num_frame_chunks)]
            if ctx.config.preproc_thresh != -1:
                cmd += ["-preProcThresh", str(ctx.config.preproc_thresh)]
            if ctx.config.num_files_per_scan > 1:
                cmd += ["-numFilesPerScan", str(ctx.config.num_files_per_scan)]
            if ctx.config.file_name:
                cmd += ["-dataFN", ctx.config.file_name]

            run_subprocess(
                cmd,
                cwd=ctx.layer_dir,
                stdout_path=ctx.log_dir / f"zip_convert_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"zip_convert_det{det.det_id}_err.csv",
            )
            if not target.exists():
                # The script may have produced something with a different
                # FileStem (e.g. when --file-name is set). Pick the newest
                # *.MIDAS.zip in layer_dir.
                cands = sorted(
                    ctx.layer_dir.glob("*.MIDAS.zip"),
                    key=lambda p: p.stat().st_mtime,
                )
                if not cands:
                    raise FileNotFoundError(
                        f"zip_convert produced no .MIDAS.zip in {ctx.layer_dir}"
                    )
                target = cands[-1]
            det.zarr_path = str(target)
            outputs[str(target)] = ""
            LOG.info("zip_convert: det %d → %s", det.det_id, target)

    finished = time.time()
    return StageResult(
        stage_name="zip_convert",
        started_at=started, finished_at=finished,
        duration_s=finished - started,
        outputs=outputs,
        metrics={"n_detectors": len(ctx.detectors)},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [Path(d.zarr_path) for d in ctx.detectors if d.zarr_path]
