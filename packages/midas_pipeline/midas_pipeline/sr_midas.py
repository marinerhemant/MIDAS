"""sr-midas integration shim (gap #9).

Adapted from ``FF_HEDM/workflows/_sr_midas_shim.py``. When the optional
``sr-midas`` package is installed, ``run_sr_peak_search`` runs the
super-resolution pipeline on a layer's ``*.MIDAS.zip`` and replaces
the conventional peak-search outputs.

The pipeline triggers it from inside ``stages/peakfit.py`` when
``config.run_sr`` is True. ``config.run_sr`` requires the conventional
peak search to be skipped (mirrors ``ff_MIDAS.py``'s ``-doPeakSearch 0``
contract).
"""
from __future__ import annotations

import os


# Match the upstream shim — torch's bundled libomp collides with MKL's on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _detect() -> tuple[bool, str | None]:
    try:
        import sr_midas  # noqa: F401
    except Exception:
        return False, None
    version = getattr(sr_midas, "__version__", None)
    if version is None:
        try:
            from sr_midas._version import __version__ as version  # type: ignore
        except Exception:
            version = "unknown"
    return True, version


SR_MIDAS_AVAILABLE, SR_MIDAS_VERSION = _detect()


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def log_status(logger, run_sr: bool = False) -> None:
    if not SR_MIDAS_AVAILABLE:
        logger.info(
            "SR-MIDAS: not available (`pip install sr-midas` to enable)."
        )
        return
    logger.info(f"SR-MIDAS: available (version {SR_MIDAS_VERSION}).")
    if not _cuda_available():
        msg = ("SR-MIDAS: no CUDA GPU detected — inference will run on CPU "
               "and be significantly slower.")
        if run_sr:
            logger.warning(msg)
        else:
            logger.info(msg)


def _resolve_config_path(sr_config_path: str | None) -> str | None:
    if sr_config_path is None or sr_config_path == "auto":
        return None
    return sr_config_path


def run_sr_peak_search(
    *,
    result_dir: str,
    srfac: int,
    sr_config_path: str,
    save_sr_patches: bool,
    save_frame_good_coords: bool,
    use_gpu: bool,
    logger,
) -> None:
    """Run ``sr_midas.pipeline.sr_process.run_sr_process`` on a MIDAS zarr dir.

    ``result_dir`` must contain a ``*.MIDAS.zip``. Raises ``RuntimeError``
    on failure so the caller can decide whether to abort.
    """
    if not SR_MIDAS_AVAILABLE:
        raise RuntimeError(
            "sr-midas is not installed; cannot run super-resolution peak search."
        )
    if not _cuda_available():
        logger.warning(
            "SR-MIDAS: running inference on CPU — this is very slow."
        )

    from sr_midas.pipeline.sr_process import run_sr_process  # type: ignore

    if not os.path.isdir(result_dir):
        raise RuntimeError(f"SR-MIDAS result_dir not found: {result_dir}")
    zips = [f for f in os.listdir(result_dir) if f.endswith(".MIDAS.zip")]
    if not zips:
        raise RuntimeError(
            f"SR-MIDAS: no '*.MIDAS.zip' found in {result_dir}. "
            "Ensure the zarr exists (zip_convert) before enabling --run-sr."
        )

    kwargs = dict(
        midasZarrDir=result_dir,
        srfac=int(srfac),
        saveSRpatches=int(save_sr_patches),
        saveFrameGoodCoords=int(save_frame_good_coords),
        use_gpu=int(use_gpu),
    )
    resolved = _resolve_config_path(sr_config_path)
    if resolved is not None:
        kwargs["SRconfig_path"] = resolved

    logger.info(
        f"SR-MIDAS: starting run_sr_process on {result_dir} "
        f"(srfac={srfac}, config={'default' if resolved is None else resolved})."
    )
    try:
        run_sr_process(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"sr-midas run_sr_process failed in {result_dir}: {exc}"
        ) from exc
    logger.info("SR-MIDAS: run_sr_process completed.")
