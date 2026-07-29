"""Centroid-only baseline inverter — the head-to-head reference for
the peak-shape claim.

Architecturally parallel to ``fit_grain_peakshape``: same warm-start,
same identifiability modes, same Adam/L-BFGS optimizers, same locked-
parameter handling, same smooth Rodrigues. The ONLY difference is the
loss:

    Peak-shape:  ‖c_s* · pred_patch[s, σ] − meas_patch[s, σ]‖²
                 (image MSE on the full 3D patch)

    Centroid:    ‖pred_centroid[s, σ] − meas_centroid[s, σ]‖²
                 (3-DoF MSE on the first moment of the patch)

The *measured* centroid is the first moment of the measured patch; this
is exactly what a peak-finder pipeline emits in production pf-HEDM
workflows. The *predicted* centroid is the gate-weighted, voxel-summed
mean of the per-voxel predicted spot positions.

This isolates the peak-shape contribution: same data, same warm-start,
same optimization machinery — only the loss differs. Recovery RMSE
deltas attributable cleanly to information density.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from midas_diffract.forward import HEDMForwardModel

from midas_pf_odf.forward import soft_beam_gate
from midas_pf_odf.simulate import GrainPatchData
from midas_pf_odf.inversion import (
    GrainPeakFitResult,
    IdentifiabilityMode,
    _aa_to_R,
    _project_mean_zero,
    _resolve_optimizer_name,
    _build_optimizer,
    _neighbor_smooth_loss,
)


def measured_centroids_from_patches(
    data: GrainPatchData,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-moment centroids of each (spot, scan) measured patch.

    Returns
    -------
    cy, cz, cf : Tensor (S, Σ)  — centroid in patch-local coordinates
        offset by the patch anchor (so e.g. cy = anchor_y + first-moment
        deviation in pixels). NaN-safe: cells with zero patch mass get
        the anchor coordinate as a fallback.
    valid : Tensor (S, Σ) bool  — True where the patch has > floor mass
        (cell trustworthy as a centroid).
    """
    M = data.measured_patches.to(torch.float64)                  # (S, Σ, F, P, P)
    S, Sigma, F, P, _ = M.shape
    device = M.device
    dtype = M.dtype

    f_idx = torch.arange(F, dtype=dtype, device=device)
    y_idx = torch.arange(P, dtype=dtype, device=device)
    z_idx = torch.arange(P, dtype=dtype, device=device)
    half_F = 0.5 * (F - 1)
    half_P = 0.5 * (P - 1)

    mass = M.sum(dim=(2, 3, 4)).clamp(min=eps)                  # (S, Σ)
    f_local = (M.sum(dim=(3, 4)) * f_idx.view(1, 1, F)).sum(dim=2) / mass
    y_local = (M.sum(dim=(2, 4)) * y_idx.view(1, 1, P)).sum(dim=2) / mass
    z_local = (M.sum(dim=(2, 3)) * z_idx.view(1, 1, P)).sum(dim=2) / mass

    anchor_y = data.anchor_y.to(dtype).to(device).unsqueeze(-1)  # (S, 1)
    anchor_z = data.anchor_z.to(dtype).to(device).unsqueeze(-1)
    anchor_f = data.anchor_f.to(dtype).to(device).unsqueeze(-1)

    cy = anchor_y + (y_local - half_P)
    cz = anchor_z + (z_local - half_P)
    cf = anchor_f + (f_local - half_F)

    floor = mass.max() * 1e-3
    valid = mass > floor
    # Where invalid, fall back to anchor (no information)
    cy = torch.where(valid, cy, anchor_y.expand_as(cy))
    cz = torch.where(valid, cz, anchor_z.expand_as(cz))
    cf = torch.where(valid, cf, anchor_f.expand_as(cf))
    return cy, cz, cf, valid


def predicted_centroids(
    model: HEDMForwardModel,
    R_voxel: torch.Tensor,        # (G, 3, 3)
    eps_voxel: torch.Tensor,      # (G, 6)
    lattice: torch.Tensor,        # (6,)
    voxel_pos: torch.Tensor,      # (G, 3)
    *,
    n_scans: int,
    gate_tau_um: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxel-summed predicted centroids per (spot, scan).

    Returns
    -------
    cy, cz, cf : Tensor (S, Σ)  — centroid in detector pixel / frame coords
    has_signal : Tensor (S, Σ) — True where some voxel illuminated this
        (s, σ) cell (gate × validity > 0).
    """
    G = R_voxel.shape[0]
    dtype = R_voxel.dtype
    device = R_voxel.device

    lat_per_v = lattice.to(dtype).unsqueeze(0).expand(G, 6)
    hkls_cart, thetas = model.correct_hkls_latc(lat_per_v, strain=eps_voxel)
    OM = R_voxel.unsqueeze(1)                                   # (G, 1, 3, 3)
    omega, eta, two_theta, valid = model.calc_bragg_geometry(
        OM, hkls_cart, thetas
    )
    pos = voxel_pos.unsqueeze(1).to(dtype)
    spots = model.project_to_detector(omega, eta, two_theta, pos, valid)
    M = int(omega.shape[-1])
    S = 2 * M
    sy = spots.y_pixel.reshape(G, S)
    sz = spots.z_pixel.reshape(G, S)
    sf = spots.frame_nr.reshape(G, S)
    sv = spots.valid.reshape(G, S).to(dtype)
    sw = spots.omega.reshape(G, S)

    sc = model.scan_config
    if sc is None:
        gate = torch.ones(G, S, n_scans, dtype=dtype, device=device)
    else:
        gate = soft_beam_gate(
            voxel_pos.to(dtype), sw,
            sc.beam_positions.to(device).to(dtype),
            float(sc.beam_size), float(gate_tau_um),
        )                                                        # (G, S, Σ)

    weight = gate * sv.unsqueeze(-1)                             # (G, S, Σ)
    norm = weight.sum(dim=0).clamp(min=1e-12)                    # (S, Σ)
    cy = (weight * sy.unsqueeze(-1)).sum(dim=0) / norm
    cz = (weight * sz.unsqueeze(-1)).sum(dim=0) / norm
    cf = (weight * sf.unsqueeze(-1)).sum(dim=0) / norm
    has_signal = norm > 1e-9
    return cy, cz, cf, has_signal


def fit_grain_centroid_baseline(
    data: GrainPatchData,
    model: HEDMForwardModel,
    *,
    voxel_pos: torch.Tensor,
    R_init: torch.Tensor,
    eps_init: torch.Tensor,
    lattice_init: torch.Tensor,
    grid_shape: Optional[tuple] = None,
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    optimizer: Optional[str] = None,
    inner_steps: int = 100,
    lr_aa: float = 1e-3,
    lr_eps: float = 1e-4,
    lr_lat: float = 1e-4,
    lambda_smooth: float = 0.0,
    gate_tau_um: float = 0.5,
    radial_only: bool = False,
    verbose: bool = False,
) -> GrainPeakFitResult:
    """Centroid-only joint per-grain inversion. Same API surface as
    :func:`fit_grain_peakshape`; only the loss differs.

    The measured centroid for each (spot, scan) is the first moment of
    the measured patch (computed once before the optimization). The
    predicted centroid is the gate-weighted voxel-summed mean of
    per-voxel predicted spot positions. Loss is the per-cell MSE on the
    (y, z, f) coordinate tuple summed across (spot, scan).

    radial_only : bool
        Reproduces the canonical Henningsson 2020 SCR formulation by
        restricting the loss to the radial (y, i.e. 2θ direction) channel
        only. z (azimuth) and f (omega) contributions are zeroed. Useful
        for a strict head-to-head comparison; expected to over-smooth even
        more than the full-CoM variant because fewer constraints survive.
    """
    optimizer_name = _resolve_optimizer_name(optimizer)
    G = int(R_init.shape[0])
    dtype = R_init.dtype
    device = R_init.device
    Sigma = int(data.scan_positions.numel())

    delta = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    eps_param = torch.nn.Parameter(eps_init.detach().clone().to(dtype).to(device))
    lattice_param = torch.nn.Parameter(
        lattice_init.detach().clone().to(dtype).to(device)
    )
    R_init_buf = R_init.detach().to(dtype).to(device)
    voxel_pos_use = voxel_pos.to(dtype).to(device)

    # Pre-compute measured centroids (data is fixed across iterations).
    meas_cy, meas_cz, meas_cf, meas_valid = measured_centroids_from_patches(data)
    meas_cy = meas_cy.to(dtype).to(device)
    meas_cz = meas_cz.to(dtype).to(device)
    meas_cf = meas_cf.to(dtype).to(device)
    meas_valid = meas_valid.to(device)

    optimizer_obj = _build_optimizer(
        optimizer_name, delta, eps_param, lattice_param,
        lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=lr_lat,
    )

    losses: List[float] = []
    last_loss_box: List[Optional[float]] = [None]

    def _eps_used() -> torch.Tensor:
        if identifiability == IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO:
            return _project_mean_zero(eps_param)
        return eps_param

    def _forward_and_loss():
        R_v = R_init_buf @ _aa_to_R(delta)
        cy, cz, cf, has_sig = predicted_centroids(
            model, R_v, _eps_used(), lattice_param, voxel_pos_use,
            n_scans=Sigma, gate_tau_um=gate_tau_um,
        )
        keep = (meas_valid & has_sig).to(dtype)
        n_keep = keep.sum().clamp(min=1.0)
        if radial_only:
            # Henningsson 2020 strict: only y (radial / 2θ direction) channel.
            diff = ((cy - meas_cy) ** 2) * keep
        else:
            diff = (
                (cy - meas_cy) ** 2
                + (cz - meas_cz) ** 2
                + (cf - meas_cf) ** 2
            ) * keep
        data_mse = diff.sum() / n_keep

        if lambda_smooth > 0.0:
            assert grid_shape is not None
            smooth = _neighbor_smooth_loss(delta, _eps_used(), grid_shape)
            return data_mse + lambda_smooth * smooth
        return data_mse

    if optimizer_obj is None:
        with torch.no_grad():
            losses.append(float(_forward_and_loss().item()))
    elif optimizer_name == "lbfgs":
        for step in range(inner_steps):
            def closure():
                optimizer_obj.zero_grad()
                loss = _forward_and_loss()
                loss.backward()
                last_loss_box[0] = float(loss.detach())
                return loss
            optimizer_obj.step(closure)
            losses.append(last_loss_box[0])
            if verbose and (step % max(1, inner_steps // 10) == 0
                            or step == inner_steps - 1):
                print(f"  centroid step {step:4d}  loss={last_loss_box[0]:.6e}")
    else:
        for step in range(inner_steps):
            optimizer_obj.zero_grad()
            loss = _forward_and_loss()
            loss.backward()
            optimizer_obj.step()
            losses.append(float(loss.item()))
            if verbose and (step % max(1, inner_steps // 10) == 0
                            or step == inner_steps - 1):
                print(f"  centroid step {step:4d}  loss={losses[-1]:.6e}")

    with torch.no_grad():
        R_v_final = R_init_buf @ _aa_to_R(delta)
        eps_final = _eps_used()
        # No per-spot scale in centroid loss — return ones for c.
        cy, cz, cf, _ = predicted_centroids(
            model, R_v_final, eps_final, lattice_param, voxel_pos_use,
            n_scans=Sigma, gate_tau_um=gate_tau_um,
        )
        n_spots = int(data.measured_patches.shape[0])
        c_per_spot = torch.ones(n_spots, dtype=dtype, device=device)

    return GrainPeakFitResult(
        R_fit=R_v_final.detach(),
        eps_fit=eps_final.detach(),
        lattice_fit=lattice_param.detach().clone(),
        c_per_spot=c_per_spot,
        losses=losses,
        holdout_score=None,
        converged=False,
        optimizer_used=optimizer_name,
        voxel_holdout_r2=None,
        metadata={
            "loss_kind": "centroid",
            "identifiability": str(identifiability),
            "lambda_smooth": float(lambda_smooth),
            "gate_tau_um": float(gate_tau_um),
        },
    )
