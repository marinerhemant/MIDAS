"""Auto-detect and wire in the optional `sr-midas` pip package.

When `sr-midas` (import name `sr_midas`) is importable, this shim adds five CLI
flags (`-runSR`, `-srfac`, `-SRconfig_path`, `-saveSRpatches`,
`-saveFrameGoodCoords`) to ff_MIDAS.py / pf_MIDAS.py and provides helpers that
replace the MIDAS peak-search stage with sr-midas's super-resolution pipeline.
When it isn't, the helpers are no-ops and the flags are never registered, so
the drivers' CLI surface is unchanged.

The contract matches the manual recipe in sr-midas's README for ff-HEDM.
pf-HEDM integration (`run_sr_peak_search_pf`) is not documented upstream; it
infers the per-scan directory layout from pf_MIDAS.py and must be validated
end-to-end against a reference run before being relied on.
"""

from __future__ import annotations

import os
import sys


# On macOS the torch wheel bundles its own libomp, which collides with the
# libomp that numpy/MKL has already initialized in midas_env and aborts the
# process at `import sr_midas`. Setting this env var before the import is the
# documented (unsafe-but-functional) workaround; MIDAS's own C workers run in
# separate processes and are unaffected. Only applied when not already set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _detect_sr_midas():
    """Return (available, version)."""
    try:
        import sr_midas  # noqa: F401
    except ImportError:
        return False, None
    except Exception:
        # Any other import-time failure (linker errors, torch init, etc.)
        # is treated as "not available" so the driver can still run without SR.
        return False, None
    version = getattr(sr_midas, "__version__", None)
    if version is None:
        try:
            from sr_midas._version import __version__ as version
        except Exception:
            version = "unknown"
    return True, version


SR_MIDAS_AVAILABLE, SR_MIDAS_VERSION = _detect_sr_midas()


def _cuda_available():
    """True when torch.cuda.is_available() succeeds; False on any failure."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def add_sr_midas_cli_args(parser):
    """Register SR-MIDAS CLI flags only when sr-midas is installed."""
    if not SR_MIDAS_AVAILABLE:
        return
    parser.add_argument(
        "-runSR", type=int, required=False, default=0,
        help="(default=0) To enable super-resolution workflow (sr-midas), set to 1. Requires -doPeakSearch 0.",
    )
    parser.add_argument(
        "-srfac", type=int, required=False, default=8, choices=[2, 4, 8],
        help="(default=8) Super-resolution factor. Options: 2, 4, 8.",
    )
    parser.add_argument(
        "-SRconfig_path", type=str, required=False, default="auto",
        help="(default='auto') Full path to sr-midas config (.json). 'auto' uses the bundled default.",
    )
    parser.add_argument(
        "-saveSRpatches", type=int, required=False, default=0,
        help="(default=0) Set to 1 to save predicted SR patches to disk.",
    )
    parser.add_argument(
        "-saveFrameGoodCoords", type=int, required=False, default=0,
        help="(default=0) Set to 1 to save per-frame goodCoords maps to disk.",
    )


def log_sr_midas_status(logger, run_sr=False):
    """Log one-line detection banner and a GPU warning when relevant.

    `run_sr` should be True when the user has actually asked for the SR path;
    in that case the GPU warning is elevated to a stronger message because the
    next step will run inference.
    """
    if not SR_MIDAS_AVAILABLE:
        logger.info("SR-MIDAS: not available (install `pip install sr-midas` to enable super-resolution peak search).")
        return
    logger.info(f"SR-MIDAS: available (version {SR_MIDAS_VERSION}).")
    if not _cuda_available():
        if run_sr:
            logger.warning(
                "SR-MIDAS: no CUDA GPU detected — inference will run on CPU and is "
                "VERY slow (expect 10-100x slowdown vs. GPU). Install a CUDA-capable "
                "PyTorch build for acceptable performance."
            )
        else:
            logger.warning(
                "SR-MIDAS: no CUDA GPU detected. If you enable -runSR 1 on this host, "
                "inference will fall back to CPU and be significantly slower."
            )


def validate_sr_midas_flags(args, do_peak_search, logger):
    """Fail fast on incompatible flag combinations.

    Call this from ff_MIDAS.py / pf_MIDAS.py after args are parsed. If sr-midas
    is not installed but the user passed -runSR 1 by some other means (e.g.
    echoed from a config), we also bail out with a clear install hint.
    """
    run_sr = int(getattr(args, "runSR", 0) or 0)
    if run_sr != 1:
        return
    if not SR_MIDAS_AVAILABLE:
        logger.error(
            "-runSR 1 requested but sr-midas is not installed. "
            "Run `pip install sr-midas` (requires Python 3.12.4) and retry."
        )
        sys.exit(1)
    if int(do_peak_search) != 0:
        logger.error(
            "-runSR 1 requires -doPeakSearch 0 (sr-midas replaces MIDAS peak search). "
            "Either set -doPeakSearch 0 or disable -runSR."
        )
        sys.exit(1)


def _resolve_sr_config_path(sr_config_path):
    """Translate the 'auto' sentinel to the None value sr-midas expects."""
    if sr_config_path is None or sr_config_path == "auto":
        return None
    return sr_config_path


def run_sr_peak_search(
    result_dir,
    srfac,
    sr_config_path,
    save_sr_patches,
    save_frame_good_coords,
    use_gpu,
    logger,
):
    """Run sr-midas's `run_sr_process` on a MIDAS zarr directory (ff-HEDM).

    `result_dir` must contain a `*.MIDAS.zip` file. Raises RuntimeError on
    failure so the caller can decide whether to abort the layer.
    """
    if not SR_MIDAS_AVAILABLE:
        raise RuntimeError("sr-midas is not installed; cannot run super-resolution peak search.")

    if not _cuda_available():
        logger.warning(
            "SR-MIDAS: running inference on CPU — this will be very slow. "
            "Cancel and install a CUDA-capable PyTorch build if this is a long run."
        )

    from sr_midas.pipeline.sr_process import run_sr_process

    if not os.path.isdir(result_dir):
        raise RuntimeError(f"SR-MIDAS result_dir not found: {result_dir}")
    zips = [f for f in os.listdir(result_dir) if f.endswith(".MIDAS.zip")]
    if not zips:
        raise RuntimeError(
            f"SR-MIDAS: no '*.MIDAS.zip' found in {result_dir}. "
            "Run with -convertFiles 1 (or ensure the zarr exists) before enabling -runSR."
        )

    kwargs = dict(
        midasZarrDir=result_dir,
        srfac=int(srfac),
        saveSRpatches=int(save_sr_patches),
        saveFrameGoodCoords=int(save_frame_good_coords),
        use_gpu=int(use_gpu),
    )
    resolved = _resolve_sr_config_path(sr_config_path)
    if resolved is not None:
        kwargs["SRconfig_path"] = resolved

    logger.info(
        f"SR-MIDAS: starting run_sr_process on {result_dir} "
        f"(srfac={srfac}, config={'default' if resolved is None else resolved})."
    )
    try:
        run_sr_process(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"sr-midas run_sr_process failed in {result_dir}: {exc}") from exc
    logger.info("SR-MIDAS: run_sr_process completed.")


