"""Robust calibration pipeline — wraps :func:`autocalibrate_pv` with
diagnostic gates and an optional Hough auto-seed.

Usage::

    res, diag = autocalibrate_robust(v1_params, image, dark=dark, ...)
    print(summarise(diag.results))
    if diag.severity == "fail":
        raise RuntimeError("calibration rejected by safety gates")

The wrapper runs three of the four foolproofing recommendations from
the paper-changes review:

  1. **Auto-seed** (optional, on by default) — re-derives BC from the
     image via :func:`seed_from_image` if the seed quality is suspect.
     Handles the v1-stale-RhoD failure mode that pushes calibrations
     into a side basin.
  2. **Diagnostic gates** — strain-cap, basin-check, cross-validation.
     See :mod:`pipelines.diagnostics`.
  3. **Severity policy** — caller picks ``raise_on="fail"`` to abort
     unsafe calibrations, ``"warn"`` to abort on any non-OK, or
     ``None`` to never abort.

The fourth recommendation (BIC basis search) is exposed separately
in :mod:`pipelines.bic_search` so it can be composed with this
wrapper without bloating the default path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..forward.panels import PanelLayout
from ..parameters.spec import CalibrationSpec
from .diagnostics import (
    DiagnosticResult, run_all_gates, summarise, worst_severity,
)
from .single_pv import autocalibrate_pv, PVCalibrationResult


@dataclass
class RobustCalibrationDiagnostics:
    severity: str                       # "ok" | "warn" | "fail"
    results: List[DiagnosticResult] = field(default_factory=list)
    auto_seeded: bool = False
    seed_v1: Optional[V1Params] = None
    seed_drift_px: float = 0.0          # |BC_seed - BC_user|

    def __str__(self) -> str:
        head = f"severity={self.severity}"
        if self.auto_seeded:
            head += f"  auto_seeded=True  drift={self.seed_drift_px:.2f}px"
        return head + "\n" + summarise(self.results)


def _maybe_auto_seed(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    panel_mask: Optional[np.ndarray] = None,
    drift_threshold_px: float = 30.0,
) -> Tuple[V1Params, bool, float]:
    """Run :func:`seed_from_image`; if the recovered BC is far from the
    user-supplied seed, replace the seed and return ``auto_seeded=True``.

    Returns ``(possibly_modified_v1, did_replace, drift_px)``.
    """
    try:
        from ..seed.from_image import seed_from_image
        from midas_calibrate.rings import build_ring_table
    except ImportError as e:
        return v1_params, False, 0.0

    # Build ring table at the user's seed Lsd to get nominal radii.
    try:
        rt = build_ring_table(v1_params)
    except Exception:
        return v1_params, False, 0.0
    radii_px = np.array(getattr(rt, "r_ideal_px",
                                  getattr(rt, "radius_px", None)),
                         dtype=np.float64)
    if radii_px.size == 0 or not np.isfinite(radii_px).all():
        return v1_params, False, 0.0

    try:
        seed = seed_from_image(
            image, sim_radii_px=radii_px,
            initial_lsd=float(v1_params.Lsd),
            npy=int(v1_params.NrPixelsY), npz=int(v1_params.NrPixelsZ),
            bc_guess=(float(v1_params.BC_y), float(v1_params.BC_z)),
            panel_mask=panel_mask,
        )
    except Exception:
        return v1_params, False, 0.0

    bc_recovered = (float(seed.bc_y), float(seed.bc_z))
    drift = float(((bc_recovered[0] - v1_params.BC_y) ** 2
                   + (bc_recovered[1] - v1_params.BC_z) ** 2) ** 0.5)
    if drift > drift_threshold_px:
        # Replace seed.  Make a copy to avoid mutating user's V1Params.
        v1_new = V1Params.from_file.__self__.__class__(**v1_params.__dict__) \
                 if hasattr(V1Params.from_file, "__self__") \
                 else _shallow_copy_v1(v1_params)
        v1_new.BC_y = float(bc_recovered[0])
        v1_new.BC_z = float(bc_recovered[1])
        if hasattr(seed, "lsd") and seed.lsd is not None and seed.lsd > 0:
            v1_new.Lsd = float(seed.lsd)
        return v1_new, True, drift
    return v1_params, False, drift


def _shallow_copy_v1(v1: V1Params) -> V1Params:
    """Best-effort shallow copy of a V1Params instance.  V1Params
    doesn't ship a ``copy`` method, so we read its dict and build a
    new instance from-default + overwrite."""
    import copy
    return copy.copy(v1)


def autocalibrate_robust(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    auto_seed: bool = False,                # opt-in: arc detection is slow
                                             # on large images (~5–10 min on
                                             # 2048² panels).  Recommended for
                                             # first-time setups, off for
                                             # routine calibration.
    auto_seed_drift_threshold_px: float = 30.0,
    raise_on: Optional[str] = "fail",       # "fail" | "warn" | None
    strain_threshold_uE: float = 100.0,
    strain_warn_uE: float = 50.0,
    cv_n_train_rings: Optional[int] = None,
    verbose: bool = True,
    **autocalibrate_kwargs,
) -> Tuple[PVCalibrationResult, RobustCalibrationDiagnostics]:
    """Robust calibration with auto-seed + safety gates.

    Parameters
    ----------
    v1_params : V1Params
        Seed geometry.  May be replaced by the Hough auto-seed if
        ``auto_seed=True`` and the image disagrees with the seed by
        more than ``auto_seed_drift_threshold_px``.
    image : np.ndarray
        Calibrant image.
    dark : optional np.ndarray
        Dark frame.
    spec : optional CalibrationSpec
        Refined parameter spec.  Defaults applied per ``autocalibrate_pv``.
    panel_layout : optional PanelLayout
        For multi-panel detectors.  The auto-seed step uses
        ``panel_layout.panel_index_mask`` as a panel mask if available.
    auto_seed : bool
        If True (default), re-derive BC from the image via Hough +
        chord-bisector.  Replaces ``v1_params.BC_y``/``BC_z`` if the
        drift exceeds ``auto_seed_drift_threshold_px``.
    raise_on : "fail" | "warn" | None
        ``"fail"`` (default) raises ``RuntimeError`` if any gate
        returns severity ``"fail"``.  ``"warn"`` also raises on
        warnings.  ``None`` never raises — caller inspects diagnostics.
    strain_threshold_uE : float
        Calibrant strain cap.  Default 100 μϵ catches every B6 basin
        escape (which all show ≥ 800 μϵ).
    cv_n_train_rings : optional int
        Train/test split for the cross-validation gate.  Default:
        floor(n_rings * 2/3).  Only meaningful if all rings end up in
        ``fits_final``; otherwise the gate degenerates.

    Returns
    -------
    (result, diagnostics) : tuple
        ``result`` is the standard :class:`PVCalibrationResult` from
        :func:`autocalibrate_pv`.  ``diagnostics`` is a
        :class:`RobustCalibrationDiagnostics` with the gate results.
    """
    if raise_on not in (None, "fail", "warn"):
        raise ValueError(f"raise_on must be 'fail', 'warn', or None; "
                         f"got {raise_on!r}")
    diag = RobustCalibrationDiagnostics(severity="ok")

    # ---- Auto-seed.
    v1_used = v1_params
    if auto_seed:
        panel_mask = None
        if panel_layout is not None and panel_layout.panel_index_mask is not None:
            panel_mask = (panel_layout.panel_index_mask.cpu().numpy() >= 0)
        v1_used, did, drift = _maybe_auto_seed(
            v1_params, image,
            panel_mask=panel_mask,
            drift_threshold_px=auto_seed_drift_threshold_px,
        )
        diag.auto_seeded = bool(did)
        diag.seed_v1 = v1_used if did else None
        diag.seed_drift_px = drift
        if verbose and did:
            print(f"[robust] auto-seed replaced BC: drift {drift:.2f} px "
                  f"→ BC=({v1_used.BC_y:.2f}, {v1_used.BC_z:.2f})", flush=True)
        elif verbose:
            print(f"[robust] auto-seed kept user BC (drift {drift:.2f} px "
                  f"≤ {auto_seed_drift_threshold_px:.1f})", flush=True)

    # ---- Run the underlying calibration.
    res = autocalibrate_pv(
        v1_used, image,
        dark=dark, spec=spec, panel_layout=panel_layout,
        verbose=verbose,
        **autocalibrate_kwargs,
    )

    # ---- Run the gates.
    diag.results = run_all_gates(
        v1_init=v1_params,                    # compare to USER's seed
        unpacked=res.unpacked,
        history=res.history,
        fits=res.fits_final,
        panel_layout=panel_layout,
        n_train_rings=cv_n_train_rings,
        strain_threshold_uE=strain_threshold_uE,
        strain_warn_uE=strain_warn_uE,
    )
    diag.severity = worst_severity(diag.results)

    if verbose:
        print(summarise(diag.results), flush=True)

    if raise_on == "fail" and diag.severity == "fail":
        raise RuntimeError(
            "calibration rejected by safety gates:\n" + summarise(diag.results)
        )
    if raise_on == "warn" and diag.severity in ("warn", "fail"):
        raise RuntimeError(
            "calibration produced a warning or failure:\n"
            + summarise(diag.results)
        )

    return res, diag


__all__ = [
    "autocalibrate_robust",
    "RobustCalibrationDiagnostics",
]
