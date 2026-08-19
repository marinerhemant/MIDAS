"""Stage: zip_convert.

Build (or verify) one MIDAS-format ``.MIDAS.zip`` per scan position
(PF) or for the single layer (FF). Delegates to the pip-installed
``midas_zipper.ff_zip`` module (``python -m midas_zipper.ff_zip``) — no
MIDAS source tree required. We shell out to keep its self-contained CLI
surface and rich raw-frame handling (TIFF / CBF / GE / HDF5).

Resume-friendly: if every required ``.MIDAS.zip`` already exists, the
stage is a no-op. PF datasets like datasetC that ship pre-built zips
fall through this stage in ~50 ms (just stat + log).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run


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
            f"zip_convert(PF): scan discovery failed: {e}. Refusing to "
            "soft-skip in PF mode."
        ) from e
    except ValueError as e:
        # Incomplete Parameters.txt (no FileStem / StartFileNrFirstLayer):
        # tolerated for smoke/partial runs; the missing-positions case
        # above is the silent-corruption one.
        LOG.warning("zip_convert(PF): scan discovery failed (%s); skip.", e)
        return stub_run("zip_convert", ctx)

    from .._pf_scans import fan_out_scans

    # N6: I/O-bound subprocess per scan — thread fan-out with per-scan
    # claims. zip_workers=1 is the serial legacy behaviour.
    def _do_scan(s):
        if s.zip_path.exists():
            # A pre-existing scan zip carries whatever parameters it was built
            # with; re-read the parameter file into it (see _param_refresh).
            _refresh(ctx, s.zip_path, s.scan_dir,
                     f"zip_convert(PF) scan {s.scan_nr}")
            return "present"
        if not cfg.convert_files:
            LOG.warning("zip_convert(PF): scan %d zip missing (%s) and "
                        "--convert-files is off; skip.",
                        s.scan_nr, s.zip_path)
            return "failed"
        s.scan_dir.mkdir(parents=True, exist_ok=True)
        if not _generate_zip(
            param_file=cfg.params_file,
            scan_dir=s.scan_dir,
            scan_within_layer=s.scan_nr,
            num_frame_chunks=cfg.num_frame_chunks,
            preproc_thresh=cfg.preproc_thresh,
            num_files_per_scan=cfg.num_files_per_scan,
        ):
            return "failed"
        if s.zip_path.exists():
            return "built"
        LOG.warning("zip_convert(PF): scan %d generator returned ok "
                    "but %s is missing.", s.scan_nr, s.zip_path)
        return "failed"

    outcomes = fan_out_scans(
        scans, _do_scan, layer_dir=layer_dir, stage="zip_convert",
        n_workers=max(1, int(getattr(cfg, "zip_workers", 1))),
    )
    n_present = sum(1 for _s, o in outcomes if o == "present")
    n_built = sum(1 for _s, o in outcomes if o == "built")
    n_failed = sum(1 for _s, o in outcomes
                   if o == "failed" or isinstance(o, Exception))

    LOG.info("zip_convert(PF): %d pre-existing + %d built + %d failed "
             "(of %d scans)", n_present, n_built, n_failed, len(scans))
    # N9: when EVERY scan failed, the run used to march on with a WARNING
    # ("0 built + N failed") and "succeed" — e.g. a missing tqdm in a
    # fresh env made every midas_zipper invocation exit 1. Fail hard.
    if scans and (n_present + n_built) == 0:
        raise RuntimeError(
            f"zip_convert(PF): all {len(scans)} scans failed to produce a "
            ".MIDAS.zip. Check midas_log/ZipErr.txt in a scan dir — a "
            "broken environment (e.g. missing dependency) fails every "
            "scan identically."
        )
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
        rep = _refresh(ctx, Path(cfg.zarr_path), layer_dir, "zip_convert(FF)")
        return _ff_result(started, [Path(cfg.zarr_path)], present=1, built=0,
                          failed=0, refresh=rep)
    existing = list(layer_dir.glob("*.MIDAS.zip"))
    if existing:
        LOG.info("zip_convert(FF): found %s", existing[0])
        rep = _refresh(ctx, existing[0], layer_dir, "zip_convert(FF)")
        return _ff_result(started, existing, present=1, built=0, failed=0,
                          refresh=rep)
    if not cfg.convert_files:
        LOG.info("zip_convert(FF): no zip in %s and --convert-files off; skip.",
                 layer_dir)
        return stub_run("zip_convert", ctx)
    # A failed conversion is a HARD error in FF, mirroring the PF P0-2 fix
    # above. It used to return a "failed" StageResult, after which every
    # downstream stage skipped with "no zarr/zip available at None" and the
    # run still printed "done. layers processed: 1" and exited 0 -- which
    # under nohup is indistinguishable from success.
    if not _generate_zip(
        param_file=cfg.params_file,
        scan_dir=layer_dir,
        scan_within_layer=ctx.layer_nr,
        num_frame_chunks=cfg.num_frame_chunks,
        preproc_thresh=cfg.preproc_thresh,
        num_files_per_scan=cfg.num_files_per_scan,
    ):
        raise RuntimeError(
            f"zip_convert(FF): failed to build the .MIDAS.zip for {layer_dir}. "
            f"See {layer_dir / 'midas_log' / 'ZipErr.txt'} for the converter's "
            f"own error. Refusing to continue -- every later stage would skip "
            f"and the run would exit 0 having produced nothing."
        )
    new_zips = list(layer_dir.glob("*.MIDAS.zip"))
    if not new_zips:
        raise RuntimeError(
            f"zip_convert(FF): converter reported success but no *.MIDAS.zip "
            f"was written in {layer_dir}. Refusing to continue."
        )
    return _ff_result(started, new_zips, present=0, built=len(new_zips), failed=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_zip(
    *,
    param_file: str,
    scan_dir: Path,
    scan_within_layer: int,
    num_frame_chunks: Optional[int] = None,
    preproc_thresh: Optional[int] = None,
    num_files_per_scan: int = 1,
) -> bool:
    """Generate one scan/layer ``*.MIDAS.zip`` via the ``midas_zipper`` package.

    The ``-LayerNr`` CLI arg, which inside pf_MIDAS.py is overloaded as the
    scan-within-layer index (1..n_scans), is mirrored here. Runs the
    pip-installed ``midas_zipper.ff_zip`` module — no MIDAS source tree
    needed (formerly shelled out to ``utils/ffGenerateZipRefactor.py``).
    """
    cmd = [
        sys.executable, "-m", "midas_zipper.ff_zip",
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
            LOG.error("zip_convert: midas_zipper.ff_zip exit=%d for %s",
                      res.returncode, scan_dir)
            for line in _tail(log_dir / "ZipErr.txt"):
                LOG.error("    ZipErr: %s", line)
            return False
        return True
    except Exception as e:
        LOG.error("zip_convert: midas_zipper.ff_zip crashed for %s: %s",
                  scan_dir, e)
        return False


def _tail(path: Path, n: int = 8) -> list[str]:
    """Last n non-blank lines of the converter's stderr, for the error message.

    Without this the only signal was 'exit=1' and the user had to know to go
    looking in midas_log/ZipErr.txt.
    """
    try:
        lines = [ln.rstrip() for ln in path.read_text(errors="replace").splitlines()
                 if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def _refresh(ctx: StageContext, zip_path: Path, work_dir: Path, label: str):
    """Re-read Parameters.txt into an already-built archive.

    Off by ``--no-refresh-params``. Newly built archives are skipped -- they
    were written from this very parameter file moments ago.
    """
    if not getattr(ctx.config, "refresh_params", True):
        LOG.info("%s: --no-refresh-params; leaving the archive's parameters "
                 "as built", label)
        return None
    from ._param_refresh import refresh_zip_params
    return refresh_zip_params(
        zip_path=zip_path,
        param_file=ctx.config.params_file,
        work_dir=work_dir,
        force=bool(getattr(ctx.config, "force_param_refresh", False)),
        label=label,
    )


def _ff_result(started: float, zips, *, present: int, built: int, failed: int,
               refresh=None):
    finished = time.time()
    metrics = {"n_present": present, "n_built": built, "n_failed": failed}
    if refresh is not None:
        # Into midas_state.h5, so "which parameters did this run actually use"
        # is answerable after the fact rather than from memory.
        metrics.update(refresh.to_metrics())
    return StageResult(
        stage_name="zip_convert",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"zips": [str(z) for z in zips]},
        metrics=metrics,
    )
