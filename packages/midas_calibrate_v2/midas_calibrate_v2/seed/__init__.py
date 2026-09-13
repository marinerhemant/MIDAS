"""Seed-from-image helpers — chord-bisector BC, multi-hypothesis Lsd,
auto max-ring detection.  Ported from v1 ``AutoCalibrateZarr.py`` with
minor cleanup; the math is identical.
"""
import sys

# CRITICAL: import diplib FIRST so its libomp.dylib loads before
# numpy/scipy on macOS.  When numpy is loaded first the system's
# OpenMP runtime claims the slot and diplib.MedianFilter silently
# hangs (no error, no return).  ``KMP_DUPLICATE_LIB_OK=TRUE`` is also
# needed at process start.  See project memory `_safe_median_filter`
# in AutoCalibrateZarr.py for the same dance.
#
# macOS-ONLY (2026-09-13): this exact workaround, run unconditionally on every
# platform, hung a Windows student's kernel at exactly this import -- zero CPU,
# uninterruptible, matching a classic Windows DLL-loader-lock deadlock (a
# native module's own thread-pool init racing the OS loader lock inside an
# already-multithreaded process, which a Jupyter kernel always is). The bug
# this preload prevents is specific to macOS's dylib load order; forcing the
# same import order on Windows bought a NEW platform-specific hang instead of
# fixing anything there, since diplib is not used by default anyway
# (``use_diplib=False`` everywhere in this module). Gate it to the one
# platform it is known to help.
if sys.platform == "darwin":
    try:
        import diplib as _diplib_preload    # noqa: F401
    except ImportError:
        pass

from .auto_max_ring import auto_detect_max_ring
from .circle_fit import pratt_circle_fit, fit_arcs_for_bc
from .cone import (
    cone_aware_bc_refine,
    cone_aware_bc_refine_with_tilt_prior,
    fit_ellipse,
)
from .from_image import seed_from_image, SeedResult
from .hough import hough_circle_bc, hough_seed_bc_lsd
from .mask import detect_panel_mask, erode_mask, apply_mask_for_arcs
from .refine import refine_seed_geometry

__all__ = ["auto_detect_max_ring", "seed_from_image", "SeedResult",
           "refine_seed_geometry", "pratt_circle_fit", "fit_arcs_for_bc",
           "hough_circle_bc", "hough_seed_bc_lsd",
           "cone_aware_bc_refine", "cone_aware_bc_refine_with_tilt_prior",
           "fit_ellipse",
           "detect_panel_mask", "erode_mask", "apply_mask_for_arcs"]
