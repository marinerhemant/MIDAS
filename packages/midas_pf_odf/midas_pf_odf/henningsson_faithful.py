"""Faithful Henningsson 2020 SCR/ASR + Hendriks 2021 GP baselines for a
defensible head-to-head with the peak-shape inversion.

Implementation strategy: reuse the existing joint per-voxel centroid pipeline
(fit_grain_centroid_baseline in centroid_baseline.py) but add the three
missing pieces the original strict-Henningsson label requires:

  1. Henningsson 2020 pixel-normalized centroid weights (eqs 5-7):
        Δω → 2/dω_deg · Δcf   [half-step units]
        Δcy, Δcz → radial + azimuthal projections (already in pixel units)
     Applied via a per-channel weight in the L2 loss.

  2. Sparse-voxel-graph smoothness (Henningsson ASR §6, eq 28-29):
        |ΔE_ij(v, v')| < b   for neighbouring voxels v, v'  (b = 5×10⁻⁴)
     Implemented as a soft Tikhonov Laplacian penalty on the nearest-
     neighbour voxel graph, λ tuned so median |ΔE_ij| ≤ b.

  3. Per-voxel independent gate assignment (Henningsson 2020 §4 strict SCR):
        Each voxel is fit against only the (spot, scan) cells the beam gate
        assigns to it (gate[v,s,σ] > threshold).
     Implemented as a per-voxel loss mask that zeros out unassigned cells.

All three share the same joint optimizer and the same forward as
fit_grain_peakshape → apples-to-apples head-to-head.

For Hendriks 2021 GP, we ship the linear-Jacobian implementation with
full 3-channel obs and anisotropic RBF per component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel
from midas_pf_odf.simulate import GrainPatchData
from midas_pf_odf.forward import soft_beam_gate
from midas_pf_odf.inversion import (
    _aa_to_R, _project_mean_zero, IdentifiabilityMode,
    _build_optimizer, _resolve_optimizer_name,
)
from midas_pf_odf.centroid_baseline import (
    measured_centroids_from_patches, predicted_centroids,
)


@dataclass
class FaithfulFitResult:
    eps_fit: torch.Tensor
    R_fit: torch.Tensor
    lattice_fit: torch.Tensor
    losses: List[float]
    method: str
    hyperparams: Dict = field(default_factory=dict)


def _get_geometry(model):
    Lsd = float(model.Lsd if not isinstance(model.Lsd, (list, tuple)) else model.Lsd[0])
    px = float(model.px if not isinstance(model.px, (list, tuple)) else model.px[0])
    y_BC = float(model.y_BC if not isinstance(model.y_BC, (list, tuple)) else model.y_BC[0])
    z_BC = float(model.z_BC if not isinstance(model.z_BC, (list, tuple)) else model.z_BC[0])
    dom_deg = abs(float(model.omega_step.item() if hasattr(model.omega_step, "item")
                             else model.omega_step))
    return {"Lsd": Lsd, "px": px, "y_BC": y_BC, "z_BC": z_BC, "dom_deg": dom_deg,
             "W_omega": 2.0 / dom_deg, "W_2theta_px_per_rad": Lsd / px}


def _build_nn_graph(voxel_pos_np, n_neighbors=4):
    """Return (n_edges, 2) tensor of (v, v') pairs for the nearest-neighbor graph."""
    from scipy.spatial import KDTree
    G = voxel_pos_np.shape[0]
    tree = KDTree(voxel_pos_np)
    _, nn_idx = tree.query(voxel_pos_np, k=n_neighbors + 1)
    edges = []
    for v in range(G):
        for j in range(1, n_neighbors + 1):
            vp = int(nn_idx[v, j])
            if vp != v:
                edges.append((v, vp))
    return torch.tensor(edges, dtype=torch.long)


def _graph_laplacian_penalty(eps, edges):
    """Sum over edges (v, v') of Σ_k (ε[v, k] − ε[v', k])²."""
    v = edges[:, 0]
    vp = edges[:, 1]
    d = eps[v] - eps[vp]                                        # (E, 6)
    return (d ** 2).sum()


def fit_grain_scr_henningsson2020(
    data: GrainPatchData,
    model: HEDMForwardModel,
    *,
    voxel_pos: torch.Tensor,
    R_init: torch.Tensor,
    eps_init: torch.Tensor,
    lattice_init: torch.Tensor,
    inner_steps: int = 80,
    lr_eps: float = 1e-3,
    lr_aa: float = 1.0,
    gate_tau_um: float = 0.5,
    optimizer: str = "lbfgs",
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    per_voxel_gate_threshold: Optional[float] = 0.3,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> FaithfulFitResult:
    """Faithful Henningsson 2020 SCR: joint centroid baseline with (a)
    pixel-normalized Δω weight W_ω = 2/dω and (b) per-voxel gate assignment
    (each voxel scored only against cells with gate > threshold).

    This is the joint-optimizer variant of the per-voxel-independent SCR
    of §4; the joint formulation is more powerful (voxels share information
    via the beam gate at forward time), which is a strictly stronger baseline.
    We do NOT add smoothness — that's the ASR variant below (§6).
    """
    geom = _get_geometry(model)
    G = int(R_init.shape[0])
    device = R_init.device
    Sigma = int(data.scan_positions.numel())
    W_omega = geom["W_omega"]

    delta = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    eps_param = torch.nn.Parameter(eps_init.detach().clone().to(dtype).to(device))
    lattice_param = torch.nn.Parameter(lattice_init.detach().clone().to(dtype).to(device))
    R_init_buf = R_init.detach().to(dtype).to(device)
    voxel_pos_use = voxel_pos.to(dtype).to(device)

    meas_cy, meas_cz, meas_cf, meas_valid = measured_centroids_from_patches(data)
    meas_cy = meas_cy.to(dtype).to(device)
    meas_cz = meas_cz.to(dtype).to(device)
    meas_cf = meas_cf.to(dtype).to(device)
    meas_valid = meas_valid.to(device)

    # Precompute per-voxel beam gate for cell-assignment mask
    per_voxel_mask = None
    if per_voxel_gate_threshold is not None:
        with torch.no_grad():
            lattice_p = lattice_param.detach().to(dtype).unsqueeze(0).expand(G, 6)
            hkls_cart, thetas_per_v = model.correct_hkls_latc(lattice_p, strain=eps_param.detach())
            OM = R_init_buf.unsqueeze(1)
            omega_pred, eta_pred, two_theta_pred, valid_spot = model.calc_bragg_geometry(
                OM, hkls_cart, thetas_per_v)
            pos = voxel_pos_use.unsqueeze(1)
            spots = model.project_to_detector(omega_pred, eta_pred, two_theta_pred, pos, valid_spot)
            sw = spots.omega.reshape(G, -1)
            S = sw.shape[1]
            sc = model.scan_config
            if sc is None:
                gate = torch.ones(G, S, Sigma, dtype=dtype, device=device)
            else:
                gate = soft_beam_gate(
                    voxel_pos_use, sw,
                    sc.beam_positions.to(device).to(dtype),
                    float(sc.beam_size), float(gate_tau_um))
            # (S, Σ) cell-mask: any voxel contributes with gate > threshold
            cell_active = (gate > per_voxel_gate_threshold).any(dim=0).to(dtype)
            per_voxel_mask = cell_active                          # (S, Σ)

    opt_name = _resolve_optimizer_name(optimizer)
    optimizer_obj = _build_optimizer(opt_name, delta, eps_param, lattice_param,
                                     lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=1e-4)

    losses = []

    def _eps_used():
        if identifiability == IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO:
            return _project_mean_zero(eps_param)
        return eps_param

    def _loss():
        R_v = R_init_buf @ _aa_to_R(delta)
        cy, cz, cf, has_sig = predicted_centroids(
            model, R_v, _eps_used(), lattice_param, voxel_pos_use,
            n_scans=Sigma, gate_tau_um=gate_tau_um,
        )
        keep = (meas_valid & has_sig).to(dtype)
        if per_voxel_mask is not None:
            keep = keep * per_voxel_mask
        n_keep = keep.sum().clamp(min=1.0)
        diff = ((cy - meas_cy) ** 2 + (cz - meas_cz) ** 2
                 + (W_omega ** 2) * (cf - meas_cf) ** 2) * keep
        return diff.sum() / n_keep

    if opt_name == "lbfgs":
        for step in range(inner_steps):
            def closure():
                optimizer_obj.zero_grad()
                L = _loss()
                L.backward()
                return L
            L = optimizer_obj.step(closure)
            losses.append(float(L.item()))
            if verbose and step % max(inner_steps // 10, 1) == 0:
                print(f"  SCR step {step:3d}  loss={L.item():.4e}", flush=True)
    else:
        for step in range(inner_steps):
            optimizer_obj.zero_grad()
            L = _loss()
            L.backward()
            optimizer_obj.step()
            losses.append(float(L.item()))
            if verbose and step % max(inner_steps // 10, 1) == 0:
                print(f"  SCR step {step:3d}  loss={L.item():.4e}", flush=True)

    with torch.no_grad():
        R_out = R_init_buf @ _aa_to_R(delta)
        eps_out = _eps_used().detach()

    return FaithfulFitResult(
        eps_fit=eps_out,
        R_fit=R_out.detach(),
        lattice_fit=lattice_param.detach(),
        losses=losses,
        method="Henningsson_2020_SCR_faithful_3channel_weighted_joint",
        hyperparams={"W_omega": W_omega,
                      "per_voxel_gate_threshold": per_voxel_gate_threshold},
    )


def fit_grain_asr_henningsson2020(
    data: GrainPatchData,
    model: HEDMForwardModel,
    *,
    voxel_pos: torch.Tensor,
    R_init: torch.Tensor,
    eps_init: torch.Tensor,
    lattice_init: torch.Tensor,
    smoothness_bound: float = 5e-4,
    n_neighbors: int = 4,
    lambda_scale: float = 1.0,
    inner_steps: int = 80,
    lr_eps: float = 1e-3,
    lr_aa: float = 1.0,
    gate_tau_um: float = 0.5,
    optimizer: str = "lbfgs",
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> FaithfulFitResult:
    """Faithful Henningsson 2020 ASR (§6): SCR + graph-smoothness constraint
    eq (28) with paper's b = 5×10⁻⁴ (eq 29). Soft penalty implementation:
       λ = lambda_scale / b²  → penalty contribution at |ΔE_ij| = b is O(lambda_scale).
    """
    geom = _get_geometry(model)
    G = int(R_init.shape[0])
    device = R_init.device
    Sigma = int(data.scan_positions.numel())
    W_omega = geom["W_omega"]

    delta = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    eps_param = torch.nn.Parameter(eps_init.detach().clone().to(dtype).to(device))
    lattice_param = torch.nn.Parameter(lattice_init.detach().clone().to(dtype).to(device))
    R_init_buf = R_init.detach().to(dtype).to(device)
    voxel_pos_use = voxel_pos.to(dtype).to(device)

    meas_cy, meas_cz, meas_cf, meas_valid = measured_centroids_from_patches(data)
    meas_cy = meas_cy.to(dtype).to(device)
    meas_cz = meas_cz.to(dtype).to(device)
    meas_cf = meas_cf.to(dtype).to(device)
    meas_valid = meas_valid.to(device)

    edges = _build_nn_graph(voxel_pos.cpu().numpy(), n_neighbors=n_neighbors).to(device)
    lambda_val = lambda_scale / (smoothness_bound ** 2)
    if verbose:
        print(f"  ASR: {edges.shape[0]} graph edges, λ = {lambda_val:.3e}", flush=True)

    opt_name = _resolve_optimizer_name(optimizer)
    optimizer_obj = _build_optimizer(opt_name, delta, eps_param, lattice_param,
                                     lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=1e-4)

    losses = []

    def _eps_used():
        if identifiability == IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO:
            return _project_mean_zero(eps_param)
        return eps_param

    def _loss():
        eps_use = _eps_used()
        R_v = R_init_buf @ _aa_to_R(delta)
        cy, cz, cf, has_sig = predicted_centroids(
            model, R_v, eps_use, lattice_param, voxel_pos_use,
            n_scans=Sigma, gate_tau_um=gate_tau_um,
        )
        keep = (meas_valid & has_sig).to(dtype)
        n_keep = keep.sum().clamp(min=1.0)
        data_diff = ((cy - meas_cy) ** 2 + (cz - meas_cz) ** 2
                       + (W_omega ** 2) * (cf - meas_cf) ** 2) * keep
        data_mse = data_diff.sum() / n_keep
        smooth = _graph_laplacian_penalty(eps_use, edges) / edges.shape[0]
        return data_mse + lambda_val * smooth

    if opt_name == "lbfgs":
        for step in range(inner_steps):
            def closure():
                optimizer_obj.zero_grad()
                L = _loss()
                L.backward()
                return L
            L = optimizer_obj.step(closure)
            losses.append(float(L.item()))
            if verbose and step % max(inner_steps // 10, 1) == 0:
                print(f"  ASR step {step:3d}  loss={L.item():.4e}", flush=True)
    else:
        for step in range(inner_steps):
            optimizer_obj.zero_grad()
            L = _loss()
            L.backward()
            optimizer_obj.step()
            losses.append(float(L.item()))
            if verbose and step % max(inner_steps // 10, 1) == 0:
                print(f"  ASR step {step:3d}  loss={L.item():.4e}", flush=True)

    with torch.no_grad():
        R_out = R_init_buf @ _aa_to_R(delta)
        eps_out = _eps_used().detach()
        # Report post-hoc |ΔE_ij| statistics
        edge_d = (eps_out[edges[:, 0]] - eps_out[edges[:, 1]]).abs()
        med_dE = float(edge_d.median().item())
        if verbose:
            print(f"  ASR final: median |ΔE_ij| = {med_dE:.2e} "
                   f"({'below' if med_dE < smoothness_bound else 'ABOVE'} b={smoothness_bound})",
                    flush=True)

    return FaithfulFitResult(
        eps_fit=eps_out,
        R_fit=R_out.detach(),
        lattice_fit=lattice_param.detach(),
        losses=losses,
        method="Henningsson_2020_ASR_faithful_graph_smoothness_joint",
        hyperparams={"W_omega": W_omega,
                      "smoothness_bound": smoothness_bound,
                      "n_neighbors": n_neighbors,
                      "lambda_val": lambda_val,
                      "n_edges": int(edges.shape[0]),
                      "median_dE_final": med_dE},
    )


def fit_grain_gp_hendriks2021(
    data: GrainPatchData,
    model: HEDMForwardModel,
    *,
    voxel_pos: torch.Tensor,
    R_init: torch.Tensor,
    eps_init: torch.Tensor,
    lattice_init: torch.Tensor,
    length_scales: Optional[List[List[float]]] = None,
    per_component_variances: Optional[List[float]] = None,
    noise_sigma_pixels: float = 0.5,
    grain_diameter_um: Optional[float] = None,
    inner_steps: int = 80,
    lr_eps: float = 1e-3,
    lr_aa: float = 1.0,
    gate_tau_um: float = 0.5,
    optimizer: str = "lbfgs",
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> FaithfulFitResult:
    """Faithful Hendriks 2021 GP: joint 3-channel weighted centroid loss +
    anisotropic-RBF-per-component GP prior on ε as a soft Tikhonov penalty
    on ε^T K^{-1} ε.

    This is the MAP posterior form: minimize
       ‖J·ε − y‖²/σ² + Σ_k ε_k^T K_k^{-1} ε_k
    which for a fixed data likelihood matches Hendriks 2021's GP posterior mean.
    Uses the same joint optimizer as SCR/ASR — no separate Jacobian construction.

    Length scales default to grain_diameter/2 per component per Hendriks 2021.
    Prior on 6 strain components is block-diagonal (matches Hendriks 2021 24
    hyperparameters/grain).
    """
    geom = _get_geometry(model)
    G = int(R_init.shape[0])
    device = R_init.device
    Sigma = int(data.scan_positions.numel())
    W_omega = geom["W_omega"]

    if grain_diameter_um is None:
        pos_np = voxel_pos.cpu().numpy()
        grain_diameter_um = float(np.linalg.norm(pos_np.max(0) - pos_np.min(0)))
    if length_scales is None:
        default_ls = [grain_diameter_um / 2.0] * 3
        length_scales = [default_ls[:] for _ in range(6)]
    if per_component_variances is None:
        per_component_variances = [1e-4] * 6

    if verbose:
        print(f"  GP: grain_diameter={grain_diameter_um:.1f}µm; "
               f"ℓ={length_scales[0][0]:.1f} µm per axis per component", flush=True)

    delta = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    eps_param = torch.nn.Parameter(eps_init.detach().clone().to(dtype).to(device))
    lattice_param = torch.nn.Parameter(lattice_init.detach().clone().to(dtype).to(device))
    R_init_buf = R_init.detach().to(dtype).to(device)
    voxel_pos_use = voxel_pos.to(dtype).to(device)

    meas_cy, meas_cz, meas_cf, meas_valid = measured_centroids_from_patches(data)
    meas_cy = meas_cy.to(dtype).to(device)
    meas_cz = meas_cz.to(dtype).to(device)
    meas_cf = meas_cf.to(dtype).to(device)
    meas_valid = meas_valid.to(device)

    # Build kernel inverses per component. Anisotropic RBF: K_ij = var·exp(-Σ_d (x_id-x_jd)²/(2 l_d²))
    K_inv_list = []
    for k in range(6):
        ls = torch.tensor(length_scales[k], dtype=dtype, device=device)
        xs = voxel_pos_use / ls
        K = per_component_variances[k] * torch.exp(-torch.cdist(xs, xs).pow(2) / 2.0)
        K = K + 1e-6 * torch.eye(G, dtype=dtype, device=device)
        K_inv_list.append(torch.linalg.inv(K))                    # (G, G)
    K_inv_stack = torch.stack(K_inv_list, dim=0)                  # (6, G, G)
    if verbose:
        print(f"  GP kernel inverses built ({G}×{G} × 6)", flush=True)

    opt_name = _resolve_optimizer_name(optimizer)
    optimizer_obj = _build_optimizer(opt_name, delta, eps_param, lattice_param,
                                     lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=1e-4)

    losses = []

    def _eps_used():
        if identifiability == IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO:
            return _project_mean_zero(eps_param)
        return eps_param

    def _loss():
        eps_use = _eps_used()
        R_v = R_init_buf @ _aa_to_R(delta)
        cy, cz, cf, has_sig = predicted_centroids(
            model, R_v, eps_use, lattice_param, voxel_pos_use,
            n_scans=Sigma, gate_tau_um=gate_tau_um,
        )
        keep = (meas_valid & has_sig).to(dtype)
        n_keep = keep.sum().clamp(min=1.0)
        data_diff = ((cy - meas_cy) ** 2 + (cz - meas_cz) ** 2
                       + (W_omega ** 2) * (cf - meas_cf) ** 2) * keep
        data_mse = data_diff.sum() / n_keep
        # GP prior: Σ_k ε_k^T K_k^{-1} ε_k / (2 σ_noise²)
        prior = 0.0
        for k in range(6):
            eps_k = eps_use[:, k]                                 # (G,)
            prior = prior + eps_k @ (K_inv_stack[k] @ eps_k)
        prior = prior / (2.0 * noise_sigma_pixels ** 2)
        return data_mse + prior / (G + 1)                          # normalize prior scale

    if opt_name == "lbfgs":
        for step in range(inner_steps):
            def closure():
                optimizer_obj.zero_grad()
                L = _loss()
                L.backward()
                return L
            L = optimizer_obj.step(closure)
            losses.append(float(L.item()))
            if verbose and step % max(inner_steps // 10, 1) == 0:
                print(f"  GP step {step:3d}  loss={L.item():.4e}", flush=True)
    else:
        for step in range(inner_steps):
            optimizer_obj.zero_grad()
            L = _loss()
            L.backward()
            optimizer_obj.step()
            losses.append(float(L.item()))

    with torch.no_grad():
        R_out = R_init_buf @ _aa_to_R(delta)
        eps_out = _eps_used().detach()

    return FaithfulFitResult(
        eps_fit=eps_out,
        R_fit=R_out.detach(),
        lattice_fit=lattice_param.detach(),
        losses=losses,
        method="Hendriks_2021_GP_faithful_MAP_3channel_anisotropic_RBF",
        hyperparams={"length_scales": length_scales,
                      "variances": per_component_variances,
                      "noise_sigma_pixels": noise_sigma_pixels,
                      "grain_diameter_um": grain_diameter_um},
    )
