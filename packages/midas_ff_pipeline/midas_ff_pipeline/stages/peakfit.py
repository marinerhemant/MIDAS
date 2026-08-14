"""Per-detector peak finding via ``peakfit_torch``.

Each detector's zarr is processed independently. Output stays in
the zarr (peak finding writes back into the ``analysis/peaks`` group)
plus auxiliary CSVs in the per-detector working directory.
"""
from __future__ import annotations

import time
from pathlib import Path

from ._base import StageContext, run_subprocess
from .._logging import stage_timer
from ..results import PeakFitResult, PerDetPeakFitResult


def _det_zip(ctx: StageContext, det) -> Path | None:
    """Resolve a detector's zarr, falling back to the layer directory.

    ``det.zarr_path`` comes back empty on the resume path, and ``Path("")``
    renders as ``"."`` — so the bare conversion handed ``peakfit_torch`` the
    current directory as its DataFile and it died on ``IsADirectoryError``,
    while the zip zip_convert had just written sat unused in the layer dir.
    Mirrors ``midas_pipeline.stages.peakfit._resolve_ff_zip``, which is the
    live implementation; this package is a retirement shim (see cli.py).
    """
    if det.zarr_path:
        return Path(det.zarr_path)
    for p in sorted(ctx.layer_dir.glob("*.MIDAS.zip")):
        return p
    return None


def run(ctx: StageContext) -> PeakFitResult:
    started = time.time()
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    per_det: list[PerDetPeakFitResult] = []

    # sr-midas branch (gap #9): replaces the conventional peak search
    # with a super-resolution pass. Requires `pip install sr-midas`.
    if ctx.config.run_sr:
        from .. import sr_midas
        from .._logging import LOG
        sr_midas.log_status(LOG, run_sr=True)
        for det in ctx.detectors:
            zip_path = _det_zip(ctx, det)
            if zip_path is None or not zip_path.exists():
                LOG.info("peakfit(sr): no zarr/zip for det %s; skip.", det.det_id)
                continue
            inputs[str(zip_path)] = ""
            sr_midas.run_sr_peak_search(
                result_dir=str(ctx.stage_dir(det)),
                srfac=ctx.config.srfac,
                sr_config_path=ctx.config.sr_config_path,
                save_sr_patches=ctx.config.save_sr_patches,
                save_frame_good_coords=ctx.config.save_frame_good_coords,
                use_gpu=(ctx.config.device.startswith("cuda")),
                logger=LOG,
            )
            per_det.append(PerDetPeakFitResult(
                det_id=det.det_id, n_peaks=-1, zip_path=str(zip_path),
            ))
            outputs[str(zip_path)] = ""
        finished = time.time()
        return PeakFitResult(
            stage_name="peakfit",
            started_at=started, finished_at=finished,
            duration_s=finished - started,
            inputs=inputs, outputs=outputs,
            per_detector=per_det,
            metrics={"n_detectors": len(ctx.detectors), "engine": "sr-midas"},
        )

    with stage_timer("peakfit"):
        for det in ctx.detectors:
            zip_path = _det_zip(ctx, det)
            if zip_path is None or not zip_path.exists():
                from .._logging import LOG as _LOG
                _LOG.info("peakfit: no zarr/zip for det %s; skip.", det.det_id)
                continue
            inputs[str(zip_path)] = ""
            stage_dir = ctx.stage_dir(det)
            stage_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "peakfit_torch",
                str(zip_path),
                "0",                      # block_nr
                "1",                      # n_blocks
                str(ctx.config.n_cpus),
                str(stage_dir),           # ResultFolder (positional override)
                "--device", ctx.config.device,
                "--dtype", ctx.config.dtype,
            ]
            run_subprocess(
                cmd,
                cwd=stage_dir,
                stdout_path=ctx.log_dir / f"peakfit_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"peakfit_det{det.det_id}_err.csv",
            )
            # Spot count is encoded in the radius_data group of the zarr;
            # we don't deserialize it here for speed. Downstream stages
            # do that check.
            per_det.append(PerDetPeakFitResult(
                det_id=det.det_id,
                n_peaks=-1,           # filled in by merge_overlaps
                zip_path=str(zip_path),
            ))
            outputs[str(zip_path)] = ""    # zarr is rewritten in place

    finished = time.time()
    return PeakFitResult(
        stage_name="peakfit",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs=inputs, outputs=outputs,
        per_detector=per_det,
        metrics={"n_detectors": len(ctx.detectors)},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    """Peak fitting writes back into the zarr — we use the zarr's mtime
    as a proxy for completion; no explicit auxiliary file.

    Resolved the same way as the input: an empty ``zarr_path`` here would make
    the resume check stat ``"."``, and a directory that always exists reads as
    "this stage is done".
    """
    return [p for p in (_det_zip(ctx, d) for d in ctx.detectors) if p is not None]
