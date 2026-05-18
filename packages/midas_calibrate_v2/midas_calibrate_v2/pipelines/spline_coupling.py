"""Joint geometry + Stage-3 spline refinement.

The 4-stage workflow currently fits a thin-plate spline post-hoc on
Stage 2's residual.  This module exposes the spline weights and
polynomial-tail coefficients as refinable :class:`Parameter` entries
on the spec, so an LM step can refine them jointly with geometry.

Relative to v1, this is a v2-only capability — scipy's RBFInterpolator
isn't differentiable, so v1 cannot do it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..forward.bragg import R_ideal_px
from ..forward.spline import TPSpline, fit_tps
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.parameter import Parameter
from ..parameters.spec import CalibrationSpec


def add_spline_parameters(
    spec: CalibrationSpec,
    X_ctrl: torch.Tensor,            # [n_ctrl, 2] (Y, Z) control points (data, fixed)
    w_init: torch.Tensor,            # [n_ctrl] kernel weights init
    c_init: torch.Tensor,            # [3] polynomial tail init
    *,
    w_bound: float = 1e6,            # sigmoid-box width for w (μm scale)
    c_bound: float = 1e6,            # for c[0] (constant) — μm
) -> None:
    """Inject ``spline_w`` and ``spline_c`` as refinable Parameters.

    The control points ``X_ctrl`` are stored on the spec under
    ``spec.spline_X`` (data, not a refined parameter).  The kernel φ used
    is ``r² log(r)`` — same as scipy's ``thin_plate_spline``.
    """
    # Stash control points and a TPS instance with placeholder weights;
    # the residual closure will re-bind them to the unpacked params.
    spec.spline_X = X_ctrl.detach().clone()
    spec.add(Parameter(
        name="spline_w",
        init=w_init.detach().clone().to(torch.float64),
        refined=True,
        bounds=(-w_bound, w_bound),
    ))
    spec.add(Parameter(
        name="spline_c",
        init=c_init.detach().clone().to(torch.float64),
        refined=True,
        bounds=(-c_bound, c_bound),
    ))


def remove_spline_parameters(spec: CalibrationSpec) -> None:
    """Tear down what :func:`add_spline_parameters` added."""
    for nm in ("spline_w", "spline_c"):
        spec.parameters.pop(nm, None)
    if hasattr(spec, "spline_X"):
        delattr(spec, "spline_X")


def pseudo_strain_residual_with_spline(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    ring_two_theta_deg: torch.Tensor,
    p: dict,
    *,
    rho_d: torch.Tensor,
    spline_X: torch.Tensor,           # [n_ctrl, 2] fixed control points
    weights: Optional[torch.Tensor] = None,
    panel_layout=None, panel_idx=None,
) -> torch.Tensor:
    """Geometry residual MINUS the spline ΔR correction (in strain units).

    Reads ``spline_w`` and ``spline_c`` from ``p`` (unpacked dict), so the
    autograd graph runs through the spline parameters too — they refine
    jointly with geometry under any LM/L-BFGS step.
    """
    r_geom = pseudo_strain_residual(
        Y_pix, Z_pix, ring_two_theta_deg, p,
        rho_d=rho_d, weights=weights,
        panel_layout=panel_layout, panel_idx=panel_idx,
    )
    if "spline_w" not in p:
        return r_geom

    # Build TPS prediction from the current params.
    spline_w = p["spline_w"]
    spline_c = p["spline_c"]
    spline = TPSpline(X=spline_X, w=spline_w, c=spline_c, smoothing=0.0)
    dR_um_pred = spline.predict(Y_pix, Z_pix)         # μm

    # Convert ΔR (μm) → strain perturbation: strain ≈ ΔR / (px · R_pred).
    pxY = p["pxY"]; pxZ = p.get("pxZ", pxY)
    px_mean = 0.5 * (pxY + pxZ)
    R_pred = R_ideal_px(ring_two_theta_deg, p["Lsd"], px_mean)
    dR_px_pred = dR_um_pred / px_mean
    return r_geom + dR_px_pred / R_pred


@dataclass
class SplineCouplingResult:
    map_unpacked: dict
    cost: float
    rc: int
    n_kept: int
    full_set_mean_uE: float
    full_set_med_uE: float
    test_set_mean_uE: float
    test_set_med_uE: float


def joint_refine_geom_spline(
    spec: CalibrationSpec,
    fits_ds,
    *,
    panel_layout=None,
    mult_factor: float = 5.0,
    n_ctrl: int = 200,
    spline_smoothing: Optional[float] = None,
    test_frac: float = 0.2,
    test_seed: int = 0,
    dtype=torch.float64,
    device: str = "cpu",
    verbose: bool = True,
) -> SplineCouplingResult:
    """Run a single LM over (geometry + spline_w + spline_c).

    Steps:
      1.  Compute the Stage-2 residual at current spec.
      2.  Hold out ``test_frac`` of fits for an honest test metric.
      3.  Pick ``n_ctrl`` random control points from the train fits.
      4.  Solve a one-shot TPS fit on the train fits → init ``w``, ``c``.
      5.  Inject ``spline_w`` and ``spline_c`` as Parameters; run LM.
      6.  Report kept-set / full-set / held-out test residuals.
    """
    from ..parameters.pack import pack_spec, unpack_spec
    from ..loss.robust_trim import multfactor_trim
    from ..inference.lm import lm_minimise

    # --- (1) Initial residual + ΔR target.
    x0, info = pack_spec(spec, dtype=dtype, device=device)
    with torch.no_grad():
        unp0 = unpack_spec(x0, info, spec)
        r_init = pseudo_strain_residual(
            fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg, unp0,
            rho_d=fits_ds.rho_d, weights=None,
            panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
        )
        keep, _ = multfactor_trim(r_init, factor=mult_factor)
        R_pred = R_ideal_px(fits_ds.ring_two_theta_deg, unp0["Lsd"],
                             0.5 * (unp0["pxY"] + unp0.get("pxZ", unp0["pxY"])))
        dR_px = -r_init * R_pred
        px_mean = float(0.5 * (unp0["pxY"] + unp0.get("pxZ", unp0["pxY"])))
        dR_um = (dR_px * px_mean).detach()

    # --- (2) Train/test split.
    n_fits = int(fits_ds.Y_pix.numel())
    rng = torch.Generator().manual_seed(test_seed)
    perm = torch.randperm(n_fits, generator=rng)
    n_test = max(int(round(test_frac * n_fits)), 1)
    test_mask_full = torch.zeros(n_fits, dtype=torch.bool)
    test_mask_full[perm[:n_test]] = True
    train_mask_full = ~test_mask_full
    train_keep = train_mask_full & keep

    Yk = fits_ds.Y_pix[train_keep]
    Zk = fits_ds.Z_pix[train_keep]
    ttk = fits_ds.ring_two_theta_deg[train_keep]
    pidxk = (fits_ds.panel_idx[train_keep]
              if fits_ds.panel_idx is not None else None)
    rho_d = fits_ds.rho_d
    n_train = int(train_keep.sum())

    # --- (3) Random control points from the train fits.
    n_ctrl = min(n_ctrl, n_train)
    ctrl_idx = torch.randperm(n_train, generator=rng)[:n_ctrl]
    Y_ctrl = Yk[ctrl_idx]
    Z_ctrl = Zk[ctrl_idx]
    dR_ctrl_init = dR_um[train_keep][ctrl_idx]

    # --- (4) One-shot TPS solve for init.
    if spline_smoothing is None:
        spline_smoothing = max(1.0, n_train * 1e-3)
    sp_init = fit_tps(Y_ctrl, Z_ctrl, dR_ctrl_init,
                       smoothing=spline_smoothing,
                       dtype=dtype)
    X_ctrl = sp_init.X
    w_init = sp_init.w
    c_init = sp_init.c
    if verbose:
        print(f"  [coupling] init TPS: {n_ctrl} control points, "
              f"smoothing={spline_smoothing:.3e}", flush=True)
        # Pre-coupling residual on test set (post-hoc evaluation).
        with torch.no_grad():
            dR_um_pred_init = sp_init.predict(fits_ds.Y_pix, fits_ds.Z_pix)
            r_corr_pre = r_init + (dR_um_pred_init / px_mean) / R_pred
        r_pre_uE = (r_corr_pre.abs() * 1e6).cpu().numpy()
        import numpy as _np
        test_idx_np = test_mask_full.cpu().numpy()
        print(f"  [coupling] post-hoc spline (Stage 4 baseline): "
              f"full mean={r_pre_uE.mean():.3f}  med={_np.median(r_pre_uE):.3f}  "
              f"test mean={r_pre_uE[test_idx_np].mean():.3f}  "
              f"med={_np.median(r_pre_uE[test_idx_np]):.3f} μϵ",
              flush=True)

    # --- (5) Inject spline params and run LM.
    add_spline_parameters(spec, X_ctrl, w_init, c_init)

    def residual_fn(unp):
        return pseudo_strain_residual_with_spline(
            Yk, Zk, ttk, unp,
            rho_d=rho_d,
            spline_X=spec.spline_X,
            panel_layout=panel_layout,
            panel_idx=pidxk,
        )

    if verbose:
        n_ref = len(spec.refined_names())
        print(f"  [coupling] LM over {n_ref} params "
              f"(geom + {n_ctrl + 3} spline DOFs) on {n_train} train fits…",
              flush=True)
    import time
    t0 = time.time()
    map_unp, cost, rc = lm_minimise(spec, residual_fn,
                                      dtype=dtype, device=device)
    elapsed = time.time() - t0
    if verbose:
        print(f"  [coupling] LM converged: rc={rc} cost={cost:.4e}  "
              f"({elapsed:.1f}s)", flush=True)

    # --- (6) Evaluate at converged params on FULL set + held-out test.
    with torch.no_grad():
        r_corr = pseudo_strain_residual_with_spline(
            fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg, map_unp,
            rho_d=rho_d, spline_X=spec.spline_X,
            panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
        )
    r_uE = (r_corr.abs() * 1e6).cpu().numpy()
    test_idx_np = test_mask_full.cpu().numpy()
    import numpy as _np
    return SplineCouplingResult(
        map_unpacked=map_unp,
        cost=cost, rc=rc, n_kept=int(train_keep.sum()),
        full_set_mean_uE=float(r_uE.mean()),
        full_set_med_uE=float(_np.median(r_uE)),
        test_set_mean_uE=float(r_uE[test_idx_np].mean()),
        test_set_med_uE=float(_np.median(r_uE[test_idx_np])),
    )


__all__ = [
    "add_spline_parameters", "remove_spline_parameters",
    "pseudo_strain_residual_with_spline",
    "joint_refine_geom_spline", "SplineCouplingResult",
]
