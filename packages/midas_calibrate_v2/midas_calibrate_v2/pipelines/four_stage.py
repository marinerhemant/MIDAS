"""4-stage calibration runner — matches v1 C's published workflow.

The v1 paper (paper3) describes the calibration as four stages:

  Stage 1 (geom-only): refine Lsd, BC, ty, tz with all distortion p₀..p₁₄
                       frozen at zero and panels frozen at zero.  Provides
                       a stable warm-start for stage 2.

  Stage 2 (full): release distortion harmonics + per-panel corrections,
                  re-run with stage 1 geometry as init.

  Stage 3 (spline): fit a thin-plate spline to the remaining ΔR residuals
                    on the cake at stage-2 geometry.  Captures localised
                    distortion the analytical basis can't represent.

  Stage 4 (eval): re-evaluate calibration metric with stage 2 + stage 3
                  spline applied (no additional refinement).  Reports the
                  final pseudo-strain floor.

This module implements all four stages on top of v2's :func:`pipelines.
single_pv.autocalibrate_pv`.  Stage 3's TPS spline is provided via scipy's
``RBFInterpolator``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params
from ..forward.bragg import R_ideal_px
from ..forward.geometry import pixel_to_REta
from ..forward.panels import PanelLayout
from ..parameters.spec import CalibrationSpec
from .single_pv import autocalibrate_pv, PVCalibrationResult


@dataclass
class StageResult:
    name: str
    unpacked: Dict[str, torch.Tensor]
    history: list
    final_strain_uE: float
    spec: CalibrationSpec


@dataclass
class FourStageResult:
    stage1: StageResult
    stage2: StageResult
    stage3_spline_fn: Optional[object] = None       # callable (Y, Z) -> ΔR
    stage3_train_rms_um: float = float("nan")
    stage3_test_rms_um: float = float("nan")        # the honest metric
    stage4_strain_uE: float = float("nan")          # full-set mean
    stage4_strain_uE_med: float = float("nan")      # full-set median
    stage4_strain_uE_test: float = float("nan")     # test-set mean (honest)
    stage4_strain_uE_test_med: float = float("nan") # test-set median (honest)


def _freeze_all_distortion(spec: CalibrationSpec, *,
                            stage1_tilt_tol_deg: float = 0.5,
                            stage1_lsd_tol_um: Optional[float] = None) -> dict:
    """Stage 1: freeze p₀..p₁₄ + panel corrections.

    Returns a dict of bound overrides applied so the caller can restore
    them at Stage 2.

    With distortion frozen at zero, the LM tends to absorb radial
    distortion residuals into the tilts ty/tz (over-tilt artefact
    observed on Varex Ceria 4-stage at -1.4° vs MAP -0.31°).  We tighten
    the tilt tolerance for Stage 1 — the user's larger ``tolTilts``
    bound is restored before Stage 2.

    Same logic for Lsd: with distortion frozen, the LM may inflate Lsd
    to compensate for missing isotropic-radial p₂.  Tighten if asked.
    """
    saved: dict = {}
    from ..forward.distortion import P_COEF_NAMES
    for n in P_COEF_NAMES:
        if n in spec.parameters:
            saved[(n, "refined")] = spec.parameters[n].refined
            spec.parameters[n].refined = False
    for n in ("panel_delta_yz", "panel_delta_theta",
              "panel_delta_lsd", "panel_delta_p2"):
        if n in spec.parameters:
            saved[(n, "refined")] = spec.parameters[n].refined
            spec.parameters[n].refined = False
    # Tilt tightening — preserves original bounds for Stage 2 restore.
    for n in ("ty", "tz"):
        p = spec.parameters.get(n)
        if p is None:
            continue
        saved[(n, "bounds")] = p.bounds
        if p.bounds is not None:
            mid = 0.5 * (p.bounds[0] + p.bounds[1])
        else:
            mid = float(p.init) if isinstance(p.init, (int, float)) else 0.0
        # Re-centre on current init (post-perturbation).
        cur = float(p.init) if isinstance(p.init, (int, float)) else mid
        p.bounds = (cur - stage1_tilt_tol_deg, cur + stage1_tilt_tol_deg)
        # Refresh the transform since bounds changed.
        from ..parameters.transforms import Logit
        p.transform = Logit(*p.bounds)
    if stage1_lsd_tol_um is not None and "Lsd" in spec.parameters:
        p = spec.parameters["Lsd"]
        saved[("Lsd", "bounds")] = p.bounds
        cur = float(p.init)
        p.bounds = (cur - stage1_lsd_tol_um, cur + stage1_lsd_tol_um)
        from ..parameters.transforms import Logit
        p.transform = Logit(*p.bounds)
    return saved


def _restore_bounds(spec: CalibrationSpec, saved: dict) -> None:
    """Restore Stage-1 bound overrides before Stage 2."""
    from ..parameters.transforms import Logit, Identity
    for (name, what), value in saved.items():
        p = spec.parameters.get(name)
        if p is None:
            continue
        if what == "refined":
            p.refined = value
        elif what == "bounds":
            p.bounds = value
            if value is not None:
                p.transform = Logit(*value)
            else:
                p.transform = Identity()


def _thaw_distortion(spec: CalibrationSpec, *, include_panels: bool = True) -> None:
    """Stage 2: thaw the harmonics; optionally also panels."""
    from ..forward.distortion import P_COEF_NAMES
    for n in P_COEF_NAMES:
        if n in spec.parameters:
            spec.parameters[n].refined = True
    if include_panels:
        for n in ("panel_delta_yz", "panel_delta_theta"):
            if n in spec.parameters:
                spec.parameters[n].refined = True


def _fit_spline_residual(
    Y_pix: torch.Tensor, Z_pix: torch.Tensor, dR_um: torch.Tensor,
    *, smoothing: Optional[float] = None,
):
    """Fit thin-plate spline ΔR(Y, Z) → callable.

    Uses scipy's RBFInterpolator with a thin-plate kernel.  When
    ``smoothing`` is None, defaults to v1's heuristic ``max(1.0, n × 1e-3)``.
    For overfit-resistant fits use :func:`_fit_spline_residual_cv`.
    """
    from scipy.interpolate import RBFInterpolator
    Y = Y_pix.detach().cpu().numpy()
    Z = Z_pix.detach().cpu().numpy()
    dR = dR_um.detach().cpu().numpy()
    coords = np.stack([Y, Z], axis=-1)
    n = len(Y)
    if smoothing is None:
        smoothing = max(1.0, n * 1e-3)
    rbf = RBFInterpolator(coords, dR, kernel="thin_plate_spline",
                            smoothing=smoothing, neighbors=200)
    def predict(Y_q, Z_q):
        q = np.stack([np.asarray(Y_q), np.asarray(Z_q)], axis=-1)
        return rbf(q)
    return predict


def _fit_spline_residual_cv(
    Y_pix: torch.Tensor, Z_pix: torch.Tensor, dR_um: torch.Tensor,
    *,
    candidate_smoothings=None,
    n_folds: int = 5,
    seed: int = 0,
    verbose: bool = False,
):
    """Pick the spline smoothing that minimises k-fold CV test RMS.

    Sweeps a candidate list (default geometric, 1e-1..1e6) and chooses
    the smoothing minimising mean test-fold RMS ΔR.  Returns
    ``(predict_fn, best_smoothing, cv_rms_um)``.

    Closes the train/test gap that the fixed ``max(1.0, n×1e-3)`` heuristic
    leaves wide open (see Stage 3 overfit warning in
    ``parity_test_2026-05-06.md``).
    """
    from scipy.interpolate import RBFInterpolator

    Y = Y_pix.detach().cpu().numpy()
    Z = Z_pix.detach().cpu().numpy()
    dR = dR_um.detach().cpu().numpy()
    coords = np.stack([Y, Z], axis=-1)
    n = len(Y)
    if candidate_smoothings is None:
        # Geometric grid spanning under- to over-smoothed.
        candidate_smoothings = np.geomspace(1e-1, 1e6, 14)

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=n)

    cv_rms_per = []
    for s in candidate_smoothings:
        rms = 0.0
        for k in range(n_folds):
            train_mask = fold != k
            test_mask = ~train_mask
            if test_mask.sum() == 0 or train_mask.sum() < 6:
                continue
            try:
                rbf = RBFInterpolator(
                    coords[train_mask], dR[train_mask],
                    kernel="thin_plate_spline", smoothing=s, neighbors=200,
                )
                pred = rbf(coords[test_mask])
                rms += float(np.std(dR[test_mask] - np.asarray(pred)))
            except Exception:
                rms += 1e30
                break
        cv_rms_per.append(rms / n_folds)
    cv_rms_per = np.asarray(cv_rms_per)
    best_i = int(np.argmin(cv_rms_per))
    best_s = float(candidate_smoothings[best_i])
    best_rms = float(cv_rms_per[best_i])
    if verbose:
        print(f"  CV smoothing search (n={n}, {n_folds}-fold):", flush=True)
        for s, rms in zip(candidate_smoothings, cv_rms_per):
            mark = " ← min" if abs(s - best_s) < 1e-12 else ""
            print(f"    smoothing={s:.3e}  cv_rms={rms:.3f} μm{mark}", flush=True)

    rbf = RBFInterpolator(coords, dR, kernel="thin_plate_spline",
                            smoothing=best_s, neighbors=200)
    def predict(Y_q, Z_q):
        q = np.stack([np.asarray(Y_q), np.asarray(Z_q)], axis=-1)
        return rbf(q)
    return predict, best_s, best_rms


def autocalibrate_four_stage(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter_stage1: int = 2,
    n_iter_stage2: int = 3,
    stage1_tilt_tol_deg: float = 0.5,
    stage1_lsd_tol_um: Optional[float] = None,
    spline_smoothing: Optional[float] = None,
    spline_cv: bool = True,            # k-fold CV for smoothing selection
    spline_cv_folds: int = 5,
    enable_stage3_spline: bool = True,
    spline_holdout_frac: float = 0.2,
    spline_holdout_seed: int = 0,
    common_kwargs: Optional[dict] = None,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> FourStageResult:
    """Run the full 4-stage calibration sequence.

    Stage 1 → 2 → 3 → 4 mirroring v1 paper3's published workflow.
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)
    if common_kwargs is None:
        common_kwargs = {}

    # ------------------------------ Stage 1
    if verbose:
        print("\n========== Stage 1: geometry only (distortion / panels frozen) ==========",
              flush=True)
    s1_overrides = _freeze_all_distortion(
        spec, stage1_tilt_tol_deg=stage1_tilt_tol_deg,
        stage1_lsd_tol_um=stage1_lsd_tol_um,
    )
    s1 = autocalibrate_pv(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_stage1, dtype=dtype, device=device, verbose=verbose,
        **common_kwargs,
    )
    s1_strain = s1.history[-1].mean_strain_uE if s1.history else float("nan")
    if verbose:
        print(f"\n[Stage 1] strain = {s1_strain:.2f} μϵ  Lsd={float(s1.unpacked['Lsd']):.2f}",
              flush=True)
    stage1 = StageResult(
        name="stage1_geom_only",
        unpacked=s1.unpacked, history=s1.history,
        final_strain_uE=s1_strain, spec=s1.spec,
    )

    # ------------------------------ Stage 2
    if verbose:
        print("\n========== Stage 2: full distortion + panels ==========", flush=True)
    spec = s1.spec     # carry MAP forward
    _restore_bounds(spec, s1_overrides)
    _thaw_distortion(spec, include_panels=(panel_layout is not None))
    s2 = autocalibrate_pv(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_stage2, dtype=dtype, device=device, verbose=verbose,
        **common_kwargs,
    )
    s2_strain = s2.history[-1].mean_strain_uE if s2.history else float("nan")
    if verbose:
        print(f"\n[Stage 2] strain = {s2_strain:.2f} μϵ  "
              f"Lsd={float(s2.unpacked['Lsd']):.2f}", flush=True)
    stage2 = StageResult(
        name="stage2_full",
        unpacked=s2.unpacked, history=s2.history,
        final_strain_uE=s2_strain, spec=s2.spec,
    )

    # ------------------------------ Stage 3 (optional)
    spline_predict = None
    s3_train_rms_um = float("nan")
    s3_test_rms_um = float("nan")
    test_mask = None
    if enable_stage3_spline and s2.fits_final is not None:
        if verbose:
            print("\n========== Stage 3: thin-plate spline residual map ==========",
                  flush=True)
        # Compute per-fit ΔR (in μm) at stage-2 geometry.
        with torch.no_grad():
            from ..loss.pseudo_strain import pseudo_strain_residual
            r = pseudo_strain_residual(
                s2.fits_final.Y_pix, s2.fits_final.Z_pix,
                s2.fits_final.ring_two_theta_deg, s2.unpacked,
                rho_d=s2.fits_final.rho_d, weights=None,
                panel_layout=panel_layout,
                panel_idx=s2.fits_final.panel_idx,
            )
            R_pred = R_ideal_px(s2.fits_final.ring_two_theta_deg,
                                 s2.unpacked["Lsd"],
                                 0.5 * (s2.unpacked["pxY"]
                                         + s2.unpacked.get("pxZ", s2.unpacked["pxY"])))
            dR_px = -r * R_pred                          # px
            px_mean = float(0.5 * (s2.unpacked["pxY"]
                                     + s2.unpacked.get("pxZ", s2.unpacked["pxY"])))
            dR_um = dR_px * px_mean                       # μm

        # Train/test split: hold out ``spline_holdout_frac`` of fits to give
        # an honest spline metric.  Without this the spline fits + evaluates
        # on the same points and returns a meaninglessly tight residual.
        n_fits = int(s2.fits_final.Y_pix.numel())
        rng = np.random.default_rng(spline_holdout_seed)
        all_idx = np.arange(n_fits)
        rng.shuffle(all_idx)
        n_test = max(int(round(spline_holdout_frac * n_fits)), 1)
        test_idx = np.sort(all_idx[:n_test])
        train_idx = np.sort(all_idx[n_test:])
        test_mask = torch.zeros(n_fits, dtype=torch.bool)
        test_mask[torch.from_numpy(test_idx)] = True

        Y_train = s2.fits_final.Y_pix[train_idx]
        Z_train = s2.fits_final.Z_pix[train_idx]
        dR_train = dR_um[train_idx]
        if spline_cv and spline_smoothing is None:
            spline_predict, chosen_s, cv_rms = _fit_spline_residual_cv(
                Y_train, Z_train, dR_train,
                n_folds=spline_cv_folds, verbose=verbose,
            )
            if verbose:
                print(f"[Stage 3] CV chose smoothing={chosen_s:.3e}  "
                      f"cv_rms={cv_rms:.3f} μm", flush=True)
        else:
            spline_predict = _fit_spline_residual(
                Y_train, Z_train, dR_train, smoothing=spline_smoothing,
            )
        # Residual on the train set: should be small (not informative).
        dR_train_pred = spline_predict(
            Y_train.detach().cpu().numpy(),
            Z_train.detach().cpu().numpy(),
        )
        train_resid = dR_train.detach().cpu().numpy() - np.asarray(dR_train_pred)
        s3_train_rms_um = float(train_resid.std())
        # Residual on the held-out test set: this is the honest metric.
        Y_test = s2.fits_final.Y_pix[test_idx]
        Z_test = s2.fits_final.Z_pix[test_idx]
        dR_test = dR_um[test_idx]
        dR_test_pred = spline_predict(
            Y_test.detach().cpu().numpy(),
            Z_test.detach().cpu().numpy(),
        )
        test_resid = dR_test.detach().cpu().numpy() - np.asarray(dR_test_pred)
        s3_test_rms_um = float(test_resid.std())
        if verbose:
            print(f"[Stage 3] fitted TPS on {len(train_idx)}-fit training set  "
                  f"input RMS ΔR = {float(dR_um.std()):.3f} μm", flush=True)
            print(f"[Stage 3]   train residual RMS = {s3_train_rms_um:.3f} μm  "
                  f"(low = expected, not informative)", flush=True)
            print(f"[Stage 3]   test  residual RMS = {s3_test_rms_um:.3f} μm  "
                  f"(this is the honest spline metric)", flush=True)
            ratio = s3_test_rms_um / max(s3_train_rms_um, 1e-12)
            if ratio > 5.0:
                print(f"[Stage 3]   ⚠ test/train ratio {ratio:.1f}× > 5 — "
                      f"likely spline overfitting; consider raising "
                      f"spline_smoothing", flush=True)

    # ------------------------------ Stage 4: eval-only with spline
    s4_mean_uE = float("nan")
    s4_med_uE = float("nan")
    s4_test_mean_uE = float("nan")
    s4_test_med_uE = float("nan")
    if s2.fits_final is not None:
        if verbose:
            print("\n========== Stage 4: evaluate (no refit) with spline applied ==========",
                  flush=True)
        with torch.no_grad():
            from ..loss.pseudo_strain import pseudo_strain_residual
            r_base = pseudo_strain_residual(
                s2.fits_final.Y_pix, s2.fits_final.Z_pix,
                s2.fits_final.ring_two_theta_deg, s2.unpacked,
                rho_d=s2.fits_final.rho_d, weights=None,
                panel_layout=panel_layout,
                panel_idx=s2.fits_final.panel_idx,
            )
            if spline_predict is not None:
                Y_np = s2.fits_final.Y_pix.detach().cpu().numpy()
                Z_np = s2.fits_final.Z_pix.detach().cpu().numpy()
                dR_um_pred = spline_predict(Y_np, Z_np)
                px_mean = float(0.5 * (s2.unpacked["pxY"]
                                         + s2.unpacked.get("pxZ", s2.unpacked["pxY"])))
                dR_px_pred = torch.tensor(
                    np.asarray(dR_um_pred) / px_mean,
                    dtype=r_base.dtype, device=r_base.device,
                )
                R_pred = R_ideal_px(s2.fits_final.ring_two_theta_deg,
                                     s2.unpacked["Lsd"],
                                     0.5 * (s2.unpacked["pxY"]
                                             + s2.unpacked.get("pxZ",
                                                                 s2.unpacked["pxY"])))
                r_corrected = r_base + dR_px_pred / R_pred
            else:
                r_corrected = r_base
            r_abs = r_corrected.abs()
            s4_mean_uE = float(r_abs.mean()) * 1e6
            s4_med_uE = float(r_abs.median()) * 1e6
            # Honest evaluation on the held-out test set.
            if test_mask is not None:
                r_abs_test = r_abs[test_mask.to(r_abs.device)]
                s4_test_mean_uE = float(r_abs_test.mean()) * 1e6
                s4_test_med_uE = float(r_abs_test.median()) * 1e6
        if verbose:
            print(f"[Stage 4] full-set strain     : mean={s4_mean_uE:7.2f}  "
                  f"med={s4_med_uE:7.2f} μϵ  ⚠ includes spline training data",
                  flush=True)
            if test_mask is not None:
                print(f"[Stage 4] held-out test strain: mean={s4_test_mean_uE:7.2f}  "
                      f"med={s4_test_med_uE:7.2f} μϵ  ✓ honest spline metric",
                      flush=True)

    return FourStageResult(
        stage1=stage1, stage2=stage2,
        stage3_spline_fn=spline_predict,
        stage3_train_rms_um=s3_train_rms_um,
        stage3_test_rms_um=s3_test_rms_um,
        stage4_strain_uE=s4_mean_uE,
        stage4_strain_uE_med=s4_med_uE,
        stage4_strain_uE_test=s4_test_mean_uE,
        stage4_strain_uE_test_med=s4_test_med_uE,
    )


__all__ = ["FourStageResult", "StageResult", "autocalibrate_four_stage"]
