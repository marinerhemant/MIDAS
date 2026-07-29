"""Per-grain ODF inversion: outer Delta iteration around inner ODF fit.

Implements §6.2 stages 1-3:
  Stage 1: build Delta table from grain-averaged single-orientation prediction.
  Stage 2: shape-only ODF fit using Adam over the ODF parameters.
  Stage 3: refresh Delta from ODF-weighted predicted centroid; refit.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


def _resolve_optimizer(name: Optional[str]) -> str:
    """Resolve `optimizer_name`: explicit value wins; otherwise read
    $MIDAS_OPTIMIZER (defaults to 'adam')."""
    if name is None:
        return os.environ.get("MIDAS_OPTIMIZER", "adam").lower()
    return name.lower()

from midas_grain_odf.forward_helpers import (
    forward_orientations,
    grain_avg_predicted_spots,
)
from midas_grain_odf.losses import (
    compute_delta_table,
    odf_weighted_predicted_centroid,
    shape_mse_loss,
)
from midas_grain_odf.odf import GrainODF
from midas_grain_odf.spot_extract import SpotPatchSpec


@dataclass
class GrainFitResult:
    """Output of a single-grain ODF fit."""

    odf: GrainODF
    delta_y: torch.Tensor
    delta_z: torch.Tensor
    delta_f: torch.Tensor
    keep: torch.Tensor
    losses: list = field(default_factory=list)
    delta_iters_run: int = 0
    converged: bool = False
    # Recovered per-grain microstrain (radial) spread scalar from the
    # refine_strain_spread DOF. In microstrain (dimensionless) units when
    # strain_spread_microstrain_units=True (default), else in patch pixels.
    # None when refine_strain_spread was disabled.
    strain_spread_fit: Optional[torch.Tensor] = None
    # Recovered per-grain orientation (eta+ω) spread scalar from the
    # refine_orientation_spread DOF. In patch pixels (broadens sigma_eta)
    # and frames (broadens sigma_f_per_spot via the same scalar).
    # None when refine_orientation_spread was disabled.
    orientation_spread_fit: Optional[torch.Tensor] = None
    # E4a: True when the corresponding spread DOF finished PINNED at its
    # physical ceiling — the fit is INVALID for that parameter (observed
    # on emerson at default LRs: both DOFs pin and the loss worsens).
    strain_spread_pinned: bool = False
    orientation_spread_pinned: bool = False

    def spread_stats(self, *, within_deg: float = 1.0) -> dict:
        """E4c: robust particle-spread statistics (weighted median +
        weight-within alongside wRMS); see
        :func:`midas_grain_odf.odf.particle_spread_stats`."""
        from midas_grain_odf.odf import particle_spread_stats
        return particle_spread_stats(self.odf, within_deg=within_deg)


def _make_optimizer(odf: GrainODF, lr_axis_angle: float, lr_logits: float,
                     optimizer: str = "adam",
                     lbfgs_max_iter: int = 20,
                     lbfgs_history_size: int = 10):
    """Build an optimizer over the ODF parameters.

    optimizer="adam" (default): two parameter groups with separate LRs for
    axis-angle (radians) and logits (dimensionless), since their gradient
    scales differ by orders of magnitude.

    optimizer="lbfgs": single-group L-BFGS with strong-Wolfe line search.
    Uses the axis-angle LR as the L-BFGS step size; logits ride along.
    L-BFGS does not natively support parameter-group LRs, but the line
    search adapts the effective step magnitude per direction so the lr
    mismatch is mostly compensated.
    """
    if optimizer == "lbfgs":
        all_params = [p for _, p in odf.named_parameters()]
        return torch.optim.LBFGS(
            all_params,
            lr=lr_axis_angle,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

    aa_params = []
    other_params = []
    for name, p in odf.named_parameters():
        if "axis_angle" in name or "mode_axis_angle" in name:
            aa_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if aa_params:
        groups.append({"params": aa_params, "lr": lr_axis_angle})
    if other_params:
        groups.append({"params": other_params, "lr": lr_logits})
    return torch.optim.Adam(groups)


def _maybe_autoscale_spread_lr(opt, param, target_step):
    """E4b: lazy LR auto-scale for a spread SGD optimizer.

    When the caller passed lr="auto", the optimizer starts with the lr=0
    sentinel; after the FIRST backward we set lr = target_step / |grad|
    so the first step moves the parameter by ~target_step (2% of its
    physical ceiling). The emerson failure: synthetic-tuned default LRs
    were orders of magnitude off on real data — both spread DOFs shot to
    their ceilings and the correlation loss WORSENED (−25.4 → −4.0).
    """
    if target_step is None or opt is None or param is None:
        return
    if opt.param_groups[0]["lr"] != 0.0:
        return                                  # already scaled
    g = param.grad
    if g is None:
        return
    gmag = float(g.detach().abs().max())
    if gmag <= 0.0:
        return
    opt.param_groups[0]["lr"] = float(target_step) / gmag


def _run_inner_optim(
    odf: GrainODF,
    model,
    position: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
    spec: SpotPatchSpec,
    measured_patches: torch.Tensor,
    delta_y: torch.Tensor,
    delta_z: torch.Tensor,
    delta_f: torch.Tensor,
    keep: torch.Tensor,
    n_steps: int,
    lr_axis_angle: float,
    lr_logits: float,
    n_distances: int,
    spot_indexer: torch.Tensor,
    verbose: bool = False,
    loss_norm: str = "mean",
    optimizer_name: Optional[str] = None,
    pixel_mask: Optional[torch.Tensor] = None,
    # Microstrain (radial) spread DOF — optional. When set, raw_strain_spread
    # is a scalar nn.Parameter and we stamp the per-spot anisotropic kernel
    # widths on the spec each step:
    #   σ_radial[s] = sqrt(σ_yz² + (gain · σ_ε · c_s[s])²)
    #   σ_eta[s]    = σ_yz                       (instrumental floor)
    # so the splat broadens radially per-spot through the autograd graph and
    # the SGD optimiser updates σ_ε from the loss gradient.
    raw_strain_spread: Optional[torch.nn.Parameter] = None,
    strain_spread_opt: Optional[torch.optim.Optimizer] = None,
    c_s: Optional[torch.Tensor] = None,
    strain_spread_gain: float = 1.0,
    strain_spread_clamp_max: Optional[float] = None,
    radial_y_buf: Optional[torch.Tensor] = None,
    radial_z_buf: Optional[torch.Tensor] = None,
    sigma_eta_const: Optional[torch.Tensor] = None,
    # σ_θ orientation spread DOF (broadens eta + ω). Same SGD pattern as σ_ε.
    raw_orientation_spread: Optional[torch.nn.Parameter] = None,
    orientation_spread_opt: Optional[torch.optim.Optimizer] = None,
    orientation_spread_clamp_max: Optional[float] = None,
    spread_gain_yz: float = 1.0,
    spread_gain_f: float = 1.0,
    sigma_f_const: Optional[torch.Tensor] = None,
    # E4b: non-None → the optimizer was built with the lr=0 "auto"
    # sentinel; scale from the first gradient (see
    # _maybe_autoscale_spread_lr).
    strain_spread_lr_target: Optional[float] = None,
    orientation_spread_lr_target: Optional[float] = None,
) -> list:
    """Adam or L-BFGS optimization of the ODF parameters with Delta fixed.

    When ``raw_strain_spread`` is supplied, an independent SGD step on the
    per-grain microstrain scalar runs alongside the ODF optimiser.
    """
    optimizer_name = _resolve_optimizer(optimizer_name)
    optimizer = _make_optimizer(
        odf, lr_axis_angle, lr_logits, optimizer=optimizer_name,
    )
    losses = []
    fit_strain_spread = (raw_strain_spread is not None
                          and strain_spread_opt is not None
                          and c_s is not None)
    fit_orientation_spread = (raw_orientation_spread is not None
                               and orientation_spread_opt is not None)
    if fit_strain_spread or fit_orientation_spread:
        # σ_yz / σ_f scalars from spec → tensors for the quadrature formulae.
        # Detached: these are instrumental floors, not fit parameters.
        sigma_yz_t = torch.tensor(
            float(spec.sigma_yz),
            dtype=spec.anchor_y.dtype, device=spec.anchor_y.device,
        )
        sigma_f_t = torch.tensor(
            float(spec.sigma_f),
            dtype=spec.anchor_y.dtype, device=spec.anchor_y.device,
        )
        # Per-spot baseline σ_radial / σ_eta floors — captured BEFORE the
        # σ_ε / σ_θ stamps overwrite them. When the caller pre-set them
        # (e.g. via `--anisotropic-sigma` which puts MIDAS-measured per-spot
        # SigmaR/SigmaEta on the spec), σ_ε broadens on top of those rather
        # than on top of a single sigma_yz scalar — gives the right
        # "additional strain broadening beyond instrument" interpretation
        # and matches v15-class fit settings.
        sigma_radial_floor = (
            spec.sigma_radial.detach().clone()
            if spec.sigma_radial is not None else None
        )
        sigma_eta_floor = (
            spec.sigma_eta.detach().clone()
            if spec.sigma_eta is not None else None
        )
        sigma_f_floor_per_spot = (
            spec.sigma_f_per_spot.detach().clone()
            if spec.sigma_f_per_spot is not None else None
        )

    def _eval_loss():
        # Stamp per-spot σ_radial (through σ_ε) and per-spot σ_eta+σ_f
        # (through σ_θ) on the spec each call so autograd connects the
        # respective raw params through the splat to the loss.
        if fit_strain_spread:
            if strain_spread_clamp_max is not None:
                sp = raw_strain_spread.clamp(min=0.0, max=strain_spread_clamp_max)
            else:
                sp = raw_strain_spread.clamp(min=0.0)
            # Per-spot baseline (sigma_radial_floor when caller pre-set it,
            # else scalar sigma_yz) — σ_ε broadens on top of this floor.
            if sigma_radial_floor is not None:
                base_sq = sigma_radial_floor * sigma_radial_floor
            else:
                base_sq = sigma_yz_t * sigma_yz_t
            spec.sigma_radial = torch.sqrt(
                base_sq + (strain_spread_gain * sp * c_s) ** 2
            )
            spec.radial_y = radial_y_buf
            spec.radial_z = radial_z_buf
            if not fit_orientation_spread:
                # Preserve caller's per-spot sigma_eta floor if set, else fall
                # back to the scalar floor.
                spec.sigma_eta = (sigma_eta_floor
                                   if sigma_eta_floor is not None
                                   else sigma_eta_const)
        elif fit_orientation_spread:
            # σ_radial stays at instrumental floor (no σ_ε DOF active).
            spec.sigma_radial = torch.full_like(
                sigma_eta_const, float(spec.sigma_yz),
            )
            spec.radial_y = radial_y_buf
            spec.radial_z = radial_z_buf
        if fit_orientation_spread:
            if orientation_spread_clamp_max is not None:
                op = raw_orientation_spread.clamp(min=0.0,
                                                    max=orientation_spread_clamp_max)
            else:
                op = raw_orientation_spread.clamp(min=0.0)
            # σ_θ adds in quadrature to the per-spot eta and frame floors
            # (when caller pre-set them) or to the scalar instrumental
            # floor. Using per-spot baseline preserves the MIDAS-measured
            # azimuthal width while σ_θ captures the *additional* orientation
            # spread the splat can't explain.
            if sigma_eta_floor is not None:
                eta_base_sq = sigma_eta_floor * sigma_eta_floor
            else:
                eta_base_sq = (sigma_yz_t * sigma_yz_t) * torch.ones_like(sigma_eta_const)
            if sigma_f_floor_per_spot is not None:
                f_base_sq = sigma_f_floor_per_spot * sigma_f_floor_per_spot
            else:
                f_base_sq = (sigma_f_t * sigma_f_t) * torch.ones_like(sigma_f_const)
            spec.sigma_eta = torch.sqrt(
                eta_base_sq + (spread_gain_yz * op) ** 2
            )
            spec.sigma_f_per_spot = torch.sqrt(
                f_base_sq + (spread_gain_f * op) ** 2
            )
        R, w = odf.sample()
        spots = forward_orientations(model, R, position, lattice_params=lattice_params)
        spot_y_full, spot_z_full, spot_f_full, valid_full = _flatten_spots(
            spots, n_distances
        )
        spot_y = spot_y_full[:, spot_indexer]
        spot_z = spot_z_full[:, spot_indexer]
        spot_f = spot_f_full[:, spot_indexer]
        valid = valid_full[:, spot_indexer]
        return shape_mse_loss(
            spec, spot_y, spot_z, spot_f, w, valid,
            measured_patches, delta_y, delta_z, delta_f, keep,
            loss_norm=loss_norm,
            pixel_mask=pixel_mask,
        )

    if optimizer_name == "lbfgs":
        # L-BFGS converges in much fewer outer steps than Adam (each step
        # internally runs a strong-Wolfe line search with up to max_iter
        # iterations). Cap the outer-loop count so callers passing
        # Adam-tuned inner_steps (200-800) don't waste compute.
        n_steps = min(n_steps, int(os.environ.get("MIDAS_LBFGS_STEPS", "50")))
        last_loss_box = [None]

        def closure():
            optimizer.zero_grad()
            if fit_strain_spread:
                strain_spread_opt.zero_grad()
            if fit_orientation_spread:
                orientation_spread_opt.zero_grad()
            loss = _eval_loss()
            loss.backward()
            last_loss_box[0] = float(loss.detach())
            return loss

        for step in range(n_steps):
            optimizer.step(closure)
            if fit_strain_spread:
                # σ_ε grad accumulated by the last closure call inside the
                # line search — single SGD step per outer L-BFGS iteration.
                _maybe_autoscale_spread_lr(strain_spread_opt,
                                           raw_strain_spread,
                                           strain_spread_lr_target)
                strain_spread_opt.step()
            if fit_orientation_spread:
                _maybe_autoscale_spread_lr(orientation_spread_opt,
                                           raw_orientation_spread,
                                           orientation_spread_lr_target)
                orientation_spread_opt.step()
            losses.append(last_loss_box[0])
            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
                print(f"    step {step:4d}  loss={last_loss_box[0]:.6e}")
        return losses

    for step in range(n_steps):
        optimizer.zero_grad()
        if fit_strain_spread:
            strain_spread_opt.zero_grad()
        if fit_orientation_spread:
            orientation_spread_opt.zero_grad()
        loss = _eval_loss()
        loss.backward()
        optimizer.step()
        if fit_strain_spread:
            _maybe_autoscale_spread_lr(strain_spread_opt, raw_strain_spread,
                                       strain_spread_lr_target)
            strain_spread_opt.step()
        if fit_orientation_spread:
            _maybe_autoscale_spread_lr(orientation_spread_opt,
                                       raw_orientation_spread,
                                       orientation_spread_lr_target)
            orientation_spread_opt.step()
        losses.append(float(loss.item()))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"    step {step:4d}  loss={loss.item():.6e}")
    return losses


def _flatten_spots(spots, n_distances: int):
    """Squeeze multi-distance dim if D=1 and flatten (2N=2 om branches, M)
    into a single spot list (S = 2 * M)."""
    y = spots.y_pixel
    z = spots.z_pixel
    f = spots.frame_nr
    v = spots.valid
    # Single-distance: shape is (..., 2, M)
    # Multi-distance: shape is (D, ..., 2, M); only support single-distance MVP.
    if n_distances != 1:
        raise NotImplementedError("MVP supports single-distance FF only.")
    # Reshape (..., 2, M) -> (..., 2*M)
    K = y.shape[0]
    y_flat = y.reshape(K, -1)
    z_flat = z.reshape(K, -1)
    f_flat = f.reshape(K, -1)
    v_flat = v.reshape(K, -1)
    return y_flat, z_flat, f_flat, v_flat


def fit_grain_odf(
    odf: GrainODF,
    model,
    position: torch.Tensor,
    measured_y: torch.Tensor,
    measured_z: torch.Tensor,
    measured_f: torch.Tensor,
    measured_patches: torch.Tensor,
    spot_indexer: torch.Tensor,
    lattice_params: Optional[torch.Tensor] = None,
    patch_F: int = 5,
    patch_P: int = 15,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    delta_iters: int = 3,
    delta_tol: float = 0.05,
    inner_steps: int = 200,
    lr_axis_angle: float = 5e-5,
    lr_logits: float = 0.1,
    outlier_alpha: float = 5.0,
    verbose: bool = False,
    loss_norm: str = "mean",
    optimizer_name: Optional[str] = None,
    pixel_mask: Optional[torch.Tensor] = None,
    sigma_radial: Optional[torch.Tensor] = None,
    sigma_eta: Optional[torch.Tensor] = None,
    sigma_f_per_spot: Optional[torch.Tensor] = None,
    radial_y: Optional[torch.Tensor] = None,
    radial_z: Optional[torch.Tensor] = None,
    # Per-grain fitted radial-microstrain σ_ε DOF. When True, broadens the
    # splat along the per-spot radial direction with rigorous Bragg coupling
    # σ_radial[s] = sqrt(σ_yz² + (gain·σ_ε·c_s[s])²), where
    # c_s = 2·Lsd·tan(θ_s)/px_um is pre-computed once from model geometry.
    # σ_ε is a single scalar per grain (cf. pf_odf which fits per-region).
    refine_strain_spread: bool = False,
    strain_spread_init: Optional[float] = None,
    lr_strain_spread: float = 5e-4,
    strain_spread_gain: float = 1.0,
    # When True (default, rigorous) σ_ε is dimensionless microstrain and the
    # per-spot pixel width is built via c_s. When False, σ_ε is a flat pixel
    # width applied uniformly to all spots (legacy / debugging mode).
    strain_spread_microstrain_units: bool = True,
    # Physical upper bound on σ_ε to prevent saturated runaway on grains
    # where the ODF model can't fit the data (low-pearson cohort). In
    # microstrain mode 5e-2 = 5% strain ceiling (any FF-HEDM analysis beyond
    # this is unphysical for metals); in pixel mode 20 px is a generous PSF×
    # multiplier. Clamped each forward step via `clamp(min=floor, max=ceil)`.
    strain_spread_max: Optional[float] = None,
    # Per-spot average dislocation contrast factor ⟨C̄⟩_hkl (shape (S,)) for
    # mWH-style strain anisotropy. When provided, multiplies c_s by
    # sqrt(⟨C̄⟩) per spot so that σ_ε becomes the modified-Williamson-Hall
    # "effective microstrain × sqrt(⟨C̄⟩)" parameter; dividing by the mean
    # sqrt(⟨C̄⟩) recovers true microstrain. Built via midas_defect's
    # ``average_contrast_factor`` (e.g. FCC, screw character) — the caller
    # is responsible for the slip-system / character / stiffness choices.
    contrast_factor_per_spot: Optional[torch.Tensor] = None,
    # Per-grain fitted orientation-spread σ_θ DOF. Orthogonal to σ_ε:
    # σ_θ broadens the splat along the per-spot eta (azimuthal) and ω
    # directions; σ_ε broadens radial. Together they decompose per-grain
    # peak width into orientation vs microstrain components. In pixel units
    # (broadens sigma_eta + sigma_f_per_spot quadratically). SGD-only, same
    # rationale as σ_ε (Adam collapses it to 0). Default OFF.
    # lr scale: pf_odf default 2e3 is for per-voxel σ_θ where the loss
    # averages over G voxels (grad ∝ 1/G). grain_odf is per-grain (one
    # scalar) so the grad is G× larger; default 50 keeps SGD steps bounded
    # and avoids overshoot through the lower-clamp pin.
    refine_orientation_spread: bool = False,
    orientation_spread_init: Optional[float] = None,
    lr_orientation_spread: float = 50.0,
    spread_gain_yz: float = 1.0,
    spread_gain_f: float = 1.0,
    orientation_spread_max: Optional[float] = None,
) -> GrainFitResult:
    """Fit a per-grain ODF with the fixed-point iterated Delta strategy.

    Parameters
    ----------
    odf : GrainODF
        Pre-constructed ODF with R_avg already set.
    model : HEDMForwardModel
    position : Tensor (3,)
        Grain centroid in micrometers.
    measured_y, measured_z, measured_f : Tensor (S,)
        Measured spot centroids in detector pixels and fractional frames.
    measured_patches : Tensor (S, F, P, P)
        Measured intensity patches centered on the measured spot centroid.
    spot_indexer : Tensor (S,) long
        Mapping from measurement-spot-index (0..S-1) to flattened forward-model
        spot index (0..2M-1). Created by the caller's match step.
    lattice_params : Tensor (6,) optional
        Strained lattice; if None, model uses nominal hkls.
    patch_F, patch_P : int
        Patch dimensions (must match measured_patches shape).
    sigma_yz, sigma_f : float
        Splat kernel widths.
    delta_iters : int
        Maximum number of outer Delta refresh iterations.
    delta_tol : float
        Convergence threshold on max |Delta change| (pixels).
    inner_steps : int
        Adam steps per outer iteration.
    inner_lr : float
        Adam learning rate for the ODF parameters.
    outlier_alpha : float
        Outlier rejection multiple (see compute_delta_table).
    verbose : bool

    Returns
    -------
    GrainFitResult
    """
    n_distances = model.n_distances

    # Stage 1: Delta table from single-orientation prediction.
    with torch.no_grad():
        spots_avg = grain_avg_predicted_spots(model, odf.R_avg, position, lattice_params)
        py_full, pz_full, pf_full, pv_full = _flatten_spots(spots_avg, n_distances)
        # spot_indexer maps measurement-index -> flat-spot index (length 2M).
        py = py_full[0, spot_indexer]
        pz = pz_full[0, spot_indexer]
        pf = pf_full[0, spot_indexer]
        pv = pv_full[0, spot_indexer]
    delta_y, delta_z, delta_f, keep = compute_delta_table(
        py, pz, pf, measured_y, measured_z, measured_f, pv, outlier_alpha
    )

    # Patch spec: anchors at single-orientation predicted centroids; Delta is
    # added at loss time so the splatter compares to the measured frame.
    spec = SpotPatchSpec(
        n_spots=int(measured_patches.shape[0]),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=py.detach().clone(),
        anchor_z=pz.detach().clone(),
        anchor_f=pf.detach().clone(),
        sigma_radial=sigma_radial,
        sigma_eta=sigma_eta,
        sigma_f_per_spot=sigma_f_per_spot,
        radial_y=radial_y,
        radial_z=radial_z,
    )

    # Per-grain fitted microstrain DOF: σ_ε scalar, broadens splat radially
    # via per-spot Bragg-law scale c_s = 2·Lsd·tan(θ_s)/px_um (microstrain
    # mode) or directly as a flat pixel width (legacy mode). SGD-only on
    # σ_ε; Adam's scale invariance collapses it to 0 here (hit in pf_odf).
    raw_strain_spread = None
    strain_spread_opt = None
    c_s = None
    radial_y_buf = None
    radial_z_buf = None
    sigma_eta_const = None
    if refine_strain_spread:
        dtype = py.dtype
        device = py.device
        S_meas = int(measured_patches.shape[0])
        # The 0.05 clamp prevents starting at the σ²-gradient zero-trap. In
        # microstrain units 5% strain is catastrophic; the chain-rule via
        # c_s amplifies the gradient so a much smaller floor is needed.
        _floor = 5e-7 if strain_spread_microstrain_units else 0.05
        # Physical upper bound. Without this, σ_ε in low-pearson grains
        # (where the ODF model can't fit the data) drifts into the saturated
        # regime (σ_radial > patch_P) — the loss is flat there so SGD's
        # signed drift wanders into unphysical values. The clamp keeps the
        # aggregate stats meaningful even when individual grain fits are
        # poor; downstream analysis still filters by pearson > 0.4 to pick
        # the trustworthy cohort.
        _ceil = (strain_spread_max if strain_spread_max is not None
                 else (5e-2 if strain_spread_microstrain_units else 20.0))
        if strain_spread_init is None:
            ssp0_val = 1e-4 if strain_spread_microstrain_units else 0.5
        else:
            ssp0_val = float(strain_spread_init)
        ssp0 = torch.tensor(
            min(max(ssp0_val, _floor), _ceil), dtype=dtype, device=device,
        )
        raw_strain_spread = torch.nn.Parameter(ssp0.reshape(()))
        # SGD (NOT Adam) for the strain spread — Adam's per-parameter
        # variance normalisation collapses σ_ε to 0 in this regime. lr is
        # ~c_s² (≈1e5) larger effective gradient than pixel mode, so the
        # default 5e-4 is roughly right for typical FF Cu geometry — but
        # ONLY for synthetic-like data: on emerson both spread DOFs shot
        # to their ceilings at the defaults (E4). Pass
        # lr_strain_spread="auto" to scale from the first gradient so the
        # first step moves σ_ε by ~2% of its ceiling.
        _ss_auto = isinstance(lr_strain_spread, str) and lr_strain_spread == "auto"
        # First-step target: 20% of the parameter's own scale (its init,
        # floored at 0.5% of the ceiling). A ceiling-relative target is
        # too coarse — 2% of the 5e-2 ceiling is a FULL typical σ_ε and
        # overshoots straight into the zero-clamp dead zone.
        _ss_lr_target = (0.2 * max(ssp0_val, 0.005 * _ceil)
                         if _ss_auto else None)
        strain_spread_opt = torch.optim.SGD(
            [raw_strain_spread],
            lr=(0.0 if _ss_auto else lr_strain_spread), momentum=0.0,
        )
        # Per-spot Bragg-law scale c_s — pre-computed once from geometry.
        # grain_odf is far-field single-Σ so c_s shape is just (S=2M,);
        # subselect via spot_indexer to (S_meas,).
        if strain_spread_microstrain_units:
            Lsd_um = float(model.Lsd if not isinstance(model.Lsd, (list, tuple))
                           else model.Lsd[0])
            px_um = float(model.px if not isinstance(model.px, (list, tuple))
                          else model.px[0])
            thetas_rad = model.thetas.to(dtype).to(device)              # (M,)
            tan_per_hkl = torch.tan(thetas_rad)                         # (M,)
            tan_per_spot = tan_per_hkl.repeat(2)                        # (2M,)
            c_s_full = (2.0 * Lsd_um * tan_per_spot / px_um)            # (2M,)
            c_s = c_s_full[spot_indexer].detach()                       # (S_meas,)
        else:
            # Pixel mode: c_s = 1 → σ_radial[s] = sqrt(σ_yz² + (gain·σ_ε)²)
            # uniformly across spots (no per-spot Bragg coupling).
            c_s = torch.ones(S_meas, dtype=dtype, device=device)
        # mWH per-spot dislocation contrast weighting. σ_ε is then the
        # effective parameter `ε · sqrt(⟨C̄⟩)`; divide by sqrt(⟨C̄⟩_mean) to
        # recover true microstrain (or, with paper3's mWH chain, derive
        # dislocation density per grain).
        if contrast_factor_per_spot is not None:
            cbar = contrast_factor_per_spot.to(dtype).to(device).reshape(-1)
            if cbar.numel() != c_s.numel():
                raise ValueError(
                    f"contrast_factor_per_spot shape {tuple(cbar.shape)} "
                    f"does not match c_s ({c_s.numel()},)"
                )
            c_s = (c_s * torch.sqrt(cbar.clamp(min=1e-12))).detach()
        # Per-spot detector-frame radial unit vectors from beam center.
        y_BC = float(model.y_BC if not isinstance(model.y_BC, (list, tuple))
                     else model.y_BC[0])
        z_BC = float(model.z_BC if not isinstance(model.z_BC, (list, tuple))
                     else model.z_BC[0])
        dyR = spec.anchor_y.to(dtype) - y_BC
        dzR = spec.anchor_z.to(dtype) - z_BC
        rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
        radial_y_buf = (dyR / rR).detach()
        radial_z_buf = (dzR / rR).detach()
        # σ_eta floor: constant per-spot tensor at sigma_yz (until σ_θ DOF
        # below overwrites it on each step when refine_orientation_spread is
        # also on).
        sigma_eta_const = torch.full(
            (S_meas,), float(sigma_yz), dtype=dtype, device=device,
        )

    # Per-grain fitted orientation-spread DOF — scalar σ_θ that broadens
    # the splat along eta + ω. When co-active with σ_ε this gives a clean
    # decomposition: σ_θ → (eta, f), σ_ε → radial.
    raw_orientation_spread = None
    orientation_spread_opt = None
    sigma_f_const = None
    if refine_orientation_spread:
        dtype = py.dtype
        device = py.device
        S_meas = int(measured_patches.shape[0])
        _floor_o = 0.05                       # px (instrumental sub-floor)
        _ceil_o = (orientation_spread_max if orientation_spread_max is not None
                    else 20.0)                # px — generous physical ceiling
        osp0_val = (float(orientation_spread_init) if orientation_spread_init is not None
                    else 0.5)                  # px, same as pf_odf default
        osp0 = torch.tensor(
            min(max(osp0_val, _floor_o), _ceil_o), dtype=dtype, device=device,
        )
        raw_orientation_spread = torch.nn.Parameter(osp0.reshape(()))
        _os_auto = (isinstance(lr_orientation_spread, str)
                    and lr_orientation_spread == "auto")
        _os_lr_target = (0.2 * max(osp0_val, 0.005 * _ceil_o)
                         if _os_auto else None)
        orientation_spread_opt = torch.optim.SGD(
            [raw_orientation_spread],
            lr=(0.0 if _os_auto else lr_orientation_spread), momentum=0.0,
        )
        sigma_f_const = torch.full(
            (S_meas,), float(sigma_f), dtype=dtype, device=device,
        )
        # If σ_ε is not running, populate radial geometry here so the
        # splatter still takes the anisotropic path (it requires all four
        # per-spot fields: sigma_radial, sigma_eta, radial_y, radial_z).
        if not refine_strain_spread:
            y_BC = float(model.y_BC if not isinstance(model.y_BC, (list, tuple))
                         else model.y_BC[0])
            z_BC = float(model.z_BC if not isinstance(model.z_BC, (list, tuple))
                         else model.z_BC[0])
            dyR = spec.anchor_y.to(dtype) - y_BC
            dzR = spec.anchor_z.to(dtype) - z_BC
            rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
            radial_y_buf = (dyR / rR).detach()
            radial_z_buf = (dzR / rR).detach()
            # σ_radial stays at the instrumental floor (no σ_ε DOF active).
            c_s = torch.zeros(S_meas, dtype=dtype, device=device)

    losses_all = []
    converged = False
    delta_y_prev = delta_y.detach().clone()
    delta_z_prev = delta_z.detach().clone()
    delta_f_prev = delta_f.detach().clone()
    final_iter = 0

    for outer in range(delta_iters):
        if verbose:
            print(f"  Delta-iter {outer}: keep={int(keep.sum())}/{keep.numel()}")
        losses = _run_inner_optim(
            odf, model, position, lattice_params,
            spec, measured_patches,
            delta_y, delta_z, delta_f, keep,
            n_steps=inner_steps,
            lr_axis_angle=lr_axis_angle, lr_logits=lr_logits,
            n_distances=n_distances,
            spot_indexer=spot_indexer,
            verbose=verbose,
            loss_norm=loss_norm,
            optimizer_name=optimizer_name,
            pixel_mask=pixel_mask,
            raw_strain_spread=raw_strain_spread,
            strain_spread_opt=strain_spread_opt,
            c_s=c_s,
            strain_spread_gain=strain_spread_gain,
            strain_spread_clamp_max=(_ceil if refine_strain_spread else None),
            strain_spread_lr_target=(_ss_lr_target
                                     if refine_strain_spread else None),
            radial_y_buf=radial_y_buf,
            radial_z_buf=radial_z_buf,
            sigma_eta_const=sigma_eta_const,
            raw_orientation_spread=raw_orientation_spread,
            orientation_spread_opt=orientation_spread_opt,
            orientation_spread_clamp_max=(_ceil_o if refine_orientation_spread else None),
            orientation_spread_lr_target=(_os_lr_target
                                          if refine_orientation_spread else None),
            spread_gain_yz=spread_gain_yz,
            spread_gain_f=spread_gain_f,
            sigma_f_const=sigma_f_const,
        )
        losses_all.extend(losses)
        final_iter = outer + 1

        # Stage 3: refresh Delta from ODF-weighted centroid.
        with torch.no_grad():
            R, w = odf.sample()
            spots = forward_orientations(model, R, position,
                                         lattice_params=lattice_params)
            sy_full, sz_full, sf_full, sv_full = _flatten_spots(spots, n_distances)
            sy = sy_full[:, spot_indexer]
            sz = sz_full[:, spot_indexer]
            sf = sf_full[:, spot_indexer]
            sv = sv_full[:, spot_indexer]
            cy, cz, cf = odf_weighted_predicted_centroid(sy, sz, sf, w, sv)

        delta_y_new, delta_z_new, delta_f_new, keep_new = compute_delta_table(
            cy, cz, cf, measured_y, measured_z, measured_f,
            (sv.sum(dim=0) > 0).to(sv.dtype), outlier_alpha,
        )

        max_change = torch.max(torch.stack([
            (delta_y_new - delta_y_prev).abs().max(),
            (delta_z_new - delta_z_prev).abs().max(),
            (delta_f_new - delta_f_prev).abs().max(),
        ]))
        if verbose:
            print(f"    Delta refresh: max change = {float(max_change):.4f}")
        delta_y, delta_z, delta_f, keep = delta_y_new, delta_z_new, delta_f_new, keep_new
        delta_y_prev = delta_y.detach().clone()
        delta_z_prev = delta_z.detach().clone()
        delta_f_prev = delta_f.detach().clone()

        if max_change < delta_tol:
            converged = True
            break

    strain_spread_pinned = False
    orientation_spread_pinned = False
    if raw_strain_spread is not None:
        strain_spread_fit = raw_strain_spread.detach().clamp(min=0.0).clone()
        if refine_strain_spread:
            # Apply the same physical upper bound the inner loop uses so the
            # reported value matches what entered σ_radial.
            strain_spread_fit = strain_spread_fit.clamp(max=_ceil)
            # E4a: a spread parameter sitting at (or beyond) its clamp
            # means the optimiser wanted MORE broadening than physics
            # allows — the fit for this DOF is invalid (wrong LR, wrong
            # model, or the grain genuinely violates the ceiling).
            if float(raw_strain_spread.detach()) >= 0.999 * _ceil:
                strain_spread_pinned = True
                warnings.warn(
                    f"fit_grain_odf: σ_ε finished PINNED at its ceiling "
                    f"({_ceil:g}) — the strain-spread fit is invalid. "
                    "Reduce lr_strain_spread (or pass "
                    "lr_strain_spread='auto') and check the model fits "
                    "the data at all.",
                    stacklevel=2,
                )
    else:
        strain_spread_fit = None
    if raw_orientation_spread is not None:
        orientation_spread_fit = raw_orientation_spread.detach().clamp(min=0.0).clone()
        if refine_orientation_spread:
            orientation_spread_fit = orientation_spread_fit.clamp(max=_ceil_o)
            if float(raw_orientation_spread.detach()) >= 0.999 * _ceil_o:
                orientation_spread_pinned = True
                warnings.warn(
                    f"fit_grain_odf: σ_θ finished PINNED at its ceiling "
                    f"({_ceil_o:g} px) — the orientation-spread fit is "
                    "invalid. Reduce lr_orientation_spread (or pass "
                    "lr_orientation_spread='auto').",
                    stacklevel=2,
                )
    else:
        orientation_spread_fit = None

    return GrainFitResult(
        odf=odf,
        delta_y=delta_y,
        delta_z=delta_z,
        delta_f=delta_f,
        keep=keep,
        losses=losses_all,
        delta_iters_run=final_iter,
        converged=converged,
        strain_spread_fit=strain_spread_fit,
        orientation_spread_fit=orientation_spread_fit,
        strain_spread_pinned=strain_spread_pinned,
        orientation_spread_pinned=orientation_spread_pinned,
    )


def fit_grain_odf_with_select(
    odf_factory,
    model,
    position: torch.Tensor,
    measured_y: torch.Tensor,
    measured_z: torch.Tensor,
    measured_f: torch.Tensor,
    measured_patches: torch.Tensor,
    spot_indexer: torch.Tensor,
    *,
    holdout_frac: float = 0.20,
    holdout_seed: int = 0,
    optimizers: tuple = ("adam", "lbfgs"),
    fit_kwargs_per_optim: Optional[dict] = None,
    **shared_kwargs,
):
    """Cross-validated optimizer selection.

    Holds out ``holdout_frac`` of the spots, fits the ODF with each
    optimizer in ``optimizers`` on the remaining training spots, scores
    each recovery on the held-out spots, and returns the
    ``GrainFitResult`` with the lower held-out per-spot image-MSE.

    Parameters
    ----------
    odf_factory : callable
        ``odf_factory()`` should return a freshly-initialized GrainODF
        (one per optimizer trial). Each optimizer needs its own copy.
    fit_kwargs_per_optim : dict, optional
        Per-optimizer overrides for fit_grain_odf kwargs (e.g.
        ``{"adam": {"inner_steps": 400, "lr_axis_angle": 1e-4},
           "lbfgs": {"inner_steps": 50, "lr_axis_angle": 1.0}}``).
    shared_kwargs : dict
        Extra kwargs passed through to fit_grain_odf for both runs.

    Returns
    -------
    best_result : GrainFitResult
        The recovery with the lower held-out MSE; carries the optimizer
        name in ``best_result.losses``'s last element via the wrapper:
        we attach ``selected_optimizer`` and ``holdout_mse`` attributes.
    """
    fit_kwargs_per_optim = fit_kwargs_per_optim or {
        "adam":  {"inner_steps": 400, "lr_axis_angle": 1e-4, "lr_logits": 0.1},
        "lbfgs": {"inner_steps": 50,  "lr_axis_angle": 1.0,  "lr_logits": 1.0},
    }

    n_spots = int(spot_indexer.numel())
    n_holdout = max(1, int(round(holdout_frac * n_spots)))
    g = torch.Generator(device="cpu").manual_seed(int(holdout_seed))
    perm = torch.randperm(n_spots, generator=g)
    val_pos = perm[:n_holdout]
    train_pos = perm[n_holdout:]

    val_idx = spot_indexer[val_pos]
    train_idx = spot_indexer[train_pos]
    val_patches = measured_patches[val_pos]
    train_patches = measured_patches[train_pos]
    val_y = measured_y[val_pos]; train_y = measured_y[train_pos]
    val_z = measured_z[val_pos]; train_z = measured_z[train_pos]
    val_f = measured_f[val_pos]; train_f = measured_f[train_pos]

    best = None
    best_score = float("inf")
    best_optim = None
    for optim in optimizers:
        odf = odf_factory()
        kw = dict(shared_kwargs)
        kw.update(fit_kwargs_per_optim.get(optim, {}))
        kw["optimizer_name"] = optim
        result = fit_grain_odf(
            odf, model, position,
            train_y, train_z, train_f, train_patches, train_idx,
            **kw,
        )
        # Score on held-out spots: forward at the recovered ODF, splat
        # into patches anchored at the val measured centroids, compute
        # image-MSE.
        score = _holdout_image_mse(
            result.odf, model, position,
            val_y, val_z, val_f, val_patches, val_idx,
            patch_F=kw.get("patch_F", 5),
            patch_P=kw.get("patch_P", 15),
            sigma_yz=kw.get("sigma_yz", 1.0),
            sigma_f=kw.get("sigma_f", 0.6),
        )
        if score < best_score:
            best_score = score
            best = result
            best_optim = optim

    best.selected_optimizer = best_optim
    best.holdout_mse = float(best_score)
    return best


def fit_grain_odf_multistart(
    odf_factory: Callable[[int], GrainODF],
    model,
    position: torch.Tensor,
    measured_y: torch.Tensor,
    measured_z: torch.Tensor,
    measured_f: torch.Tensor,
    measured_patches: torch.Tensor,
    spot_indexer: torch.Tensor,
    *,
    n_restarts: int = 16,
    score: str = "train_loss",
    holdout_frac: float = 0.20,
    holdout_seed: int = 0,
    base_seed: int = 0,
    seed_step: int = 1009,
    **fit_kwargs,
) -> GrainFitResult:
    """Run ``n_restarts`` independent fits with different ODF inits
    and return the best by ``score``.

    Picks up after the 2026-04-30 density-recovery investigation: with a
    K-particle ParticleODF and L-BFGS, a single random init reliably
    converges to a basin where one mode dominates, but *which* mode it
    captures is seed-dependent. Multi-start with training-loss picking
    drove KS distance from 0.236 → 0.147 (38% reduction) and rel-L2
    from 0.55 → 0.36 (35% reduction) at K=24, spread=2°.

    Parameters
    ----------
    odf_factory : Callable[[int], GrainODF]
        Called once per restart with a unique integer seed; must return
        a freshly initialized ODF (same R_avg / theta_max / K, but
        different particle positions and logits).
    measured_y, measured_z, measured_f, measured_patches, spot_indexer :
        Same as ``fit_grain_odf``.
    n_restarts : int
        Number of independent fits to run.
    score : {"train_loss", "holdout_mse"}
        How to rank restarts. ``"train_loss"`` picks the lowest final
        training loss across all spots. ``"holdout_mse"`` holds out
        ``holdout_frac`` of spots, fits on the rest, and ranks by the
        recovered ODF's image-MSE on the held-out spots — more honest
        but noisier at small spot counts.
    holdout_frac, holdout_seed :
        Used only when ``score="holdout_mse"``.
    base_seed, seed_step : int
        Restart seeds are ``base_seed + r * seed_step`` for r in
        range(n_restarts).
    fit_kwargs :
        Passed through to each ``fit_grain_odf`` call (patch_F, patch_P,
        delta_iters, inner_steps, optimizer_name, loss_norm, …).

    Returns
    -------
    best_result : GrainFitResult
        Carries ``selected_restart`` (the winning seed index) and
        ``restart_scores`` (the per-restart scores) as added attributes.
    """
    if score not in ("train_loss", "holdout_mse"):
        raise ValueError(
            f"score must be 'train_loss' or 'holdout_mse'; got {score!r}"
        )

    if score == "holdout_mse":
        n_spots = int(spot_indexer.numel())
        n_holdout = max(1, int(round(holdout_frac * n_spots)))
        g = torch.Generator(device="cpu").manual_seed(int(holdout_seed))
        perm = torch.randperm(n_spots, generator=g)
        val_pos = perm[:n_holdout]
        train_pos = perm[n_holdout:]
        val_idx = spot_indexer[val_pos]
        train_idx = spot_indexer[train_pos]
        val_patches = measured_patches[val_pos]
        train_patches = measured_patches[train_pos]
        val_y = measured_y[val_pos]; train_y = measured_y[train_pos]
        val_z = measured_z[val_pos]; train_z = measured_z[train_pos]
        val_f = measured_f[val_pos]; train_f = measured_f[train_pos]
    else:
        train_idx = spot_indexer
        train_patches = measured_patches
        train_y, train_z, train_f = measured_y, measured_z, measured_f

    best = None
    best_score = float("inf")
    best_idx = -1
    scores = []
    for r in range(n_restarts):
        seed = int(base_seed + r * seed_step)
        odf = odf_factory(seed)
        result = fit_grain_odf(
            odf, model, position,
            train_y, train_z, train_f, train_patches, train_idx,
            **fit_kwargs,
        )
        if score == "train_loss":
            s = float(result.losses[-1]) if result.losses else float("inf")
        else:
            s = _holdout_image_mse(
                result.odf, model, position,
                val_y, val_z, val_f, val_patches, val_idx,
                patch_F=fit_kwargs.get("patch_F", 5),
                patch_P=fit_kwargs.get("patch_P", 15),
                sigma_yz=fit_kwargs.get("sigma_yz", 1.0),
                sigma_f=fit_kwargs.get("sigma_f", 0.6),
            )
        scores.append(s)
        if s < best_score:
            best_score = s
            best = result
            best_idx = r

    best.selected_restart = best_idx
    best.restart_scores = scores
    best.restart_score_kind = score
    return best


def _holdout_image_mse(odf, model, position, val_y, val_z, val_f,
                        val_patches, val_idx, *, patch_F, patch_P,
                        sigma_yz, sigma_f):
    """Compute mean image-MSE of the recovered ODF against the
    held-out measured patches."""
    from midas_grain_odf.spot_extract import (
        SpotPatchSpec, splat_spots_to_patches,
    )
    with torch.no_grad():
        R, w = odf.sample()
        spots = forward_orientations(model, R, position)
        sy_full, sz_full, sf_full, sv_full = _flatten_spots(
            spots, model.n_distances if hasattr(model, "n_distances") else 1,
        )
        sy = sy_full[:, val_idx]
        sz = sz_full[:, val_idx]
        sf = sf_full[:, val_idx]
        sv = sv_full[:, val_idx]
        spec = SpotPatchSpec(
            n_spots=int(val_idx.numel()),
            patch_F=patch_F, patch_P=patch_P,
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            anchor_y=val_y.detach().clone(),
            anchor_z=val_z.detach().clone(),
            anchor_f=val_f.detach().clone(),
        )
        pred = splat_spots_to_patches(spec, sy, sz, sf, w, sv)
        diff2 = (pred - val_patches) ** 2
        return float(diff2.flatten(1).mean(dim=1).mean())
