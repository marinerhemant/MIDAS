"""Synthetic microstructure plant + forward synthesis for pf-HEDM.

Produces a planted ``(R_V, ε_V)`` field on a regular voxel grid and
forward-simulates the joint per-grain measured peak patches the
inversion driver consumes. All-torch, all-differentiable, hermetic.

The plant covers the canonical Henningsson-style cases:
- Constant (R_avg, ε_avg)
- Linear orientation gradient (axis-angle vs (x, y))
- Linear / polynomial / sinusoidal strain gradient (per-Voigt-component)

Real-data ingest (zarr / ExtraInfo.bin) is deferred to ``io.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from midas_diffract.forward import HEDMGeometry, ScanConfig, HEDMForwardModel
from midas_stress.orientation import axis_angle_to_orient_mat


def _aa_to_R(axis_angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) → rotation matrix (..., 3, 3) via midas_stress."""
    eps = 1e-9
    norm = axis_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / norm
    angle_deg = norm.squeeze(-1) * (180.0 / math.pi)
    R = axis_angle_to_orient_mat(axis, angle_deg)
    near_zero = (norm.squeeze(-1) < 10.0 * eps).unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return torch.where(near_zero, I.expand_as(R), R)


@dataclass
class SinglePhaseGrainPlant:
    """Ground-truth state for one grain plant.

    All tensors are on CPU and fp64 by default; the forward layer moves
    to the requested device/dtype at simulation time.
    """
    voxel_pos: torch.Tensor       # (G, 3) sample-frame positions in µm
    R_voxel: torch.Tensor         # (G, 3, 3) per-voxel orientation
    eps_voxel: torch.Tensor       # (G, 6) per-voxel Voigt strain (crystal frame)
    lattice: torch.Tensor         # (6,) reference lattice [a,b,c,α,β,γ] (Å, deg)
    R_avg: torch.Tensor           # (3, 3) grain-average orientation (initial guess proxy)
    grid_shape: Tuple[int, int]   # (G_x, G_y) for diagnostic plotting
    metadata: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_pos.shape[0])

    def to(self, device, dtype=None) -> "SinglePhaseGrainPlant":
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        return SinglePhaseGrainPlant(
            voxel_pos=self.voxel_pos.to(**kw),
            R_voxel=self.R_voxel.to(**kw),
            eps_voxel=self.eps_voxel.to(**kw),
            lattice=self.lattice.to(**kw),
            R_avg=self.R_avg.to(**kw),
            grid_shape=self.grid_shape,
            metadata=dict(self.metadata),
        )


def plant_single_grain(
    grid_shape: Tuple[int, int] = (10, 10),
    voxel_size_um: float = 2.0,
    lattice: Tuple[float, ...] = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0),
    R_avg: Optional[torch.Tensor] = None,
    R_gradient_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    R_gradient_amp_deg: float = 0.0,
    R_gradient_dir: str = "x",
    eps_avg: Tuple[float, float, float, float, float, float] = (0., 0., 0., 0., 0., 0.),
    eps_gradient_voigt: int = 0,
    eps_gradient_amp: float = 0.0,
    eps_gradient_dir: str = "y",
    *,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
) -> SinglePhaseGrainPlant:
    """Plant a single-grain microstructure with smooth gradients.

    The voxel grid sits at (i * voxel_size, j * voxel_size, 0) for
    i,j ∈ [0, G_x) × [0, G_y), centered at the origin so the grain
    spans roughly ``[-G_x/2, G_x/2] * voxel_size``.

    Parameters
    ----------
    grid_shape : (G_x, G_y)
        Voxels per side. Default 10x10 for fast unit tests.
    voxel_size_um : float
        Side length of one voxel in µm (sample frame).
    lattice : 6-tuple
        Reference [a, b, c, α, β, γ] in Å / degrees.
    R_avg : (3, 3) tensor, optional
        Mean orientation. None → identity.
    R_gradient_axis : 3-tuple (unit-ish)
        Axis-angle direction along which the per-voxel orientation
        deviates from R_avg.
    R_gradient_amp_deg : float
        Maximum axis-angle magnitude across the grid (deg).
    R_gradient_dir : 'x' | 'y' | 'diag'
        Spatial direction of the orientation gradient.
    eps_avg : 6-tuple
        Mean Voigt strain (e_11, e_12, e_13, e_22, e_23, e_33).
    eps_gradient_voigt : int in {0..5}
        Which Voigt component varies across the grid.
    eps_gradient_amp : float
        Amplitude of the linear strain gradient.
    eps_gradient_dir : 'x' | 'y' | 'diag'
        Spatial direction of the strain gradient.
    seed : int, optional
        Random seed (currently only used if R_avg is None and we
        wanted random orientations — kept for future extensions).
    """
    Gx, Gy = grid_shape
    G = Gx * Gy

    # Voxel positions on a regular grid, centered.
    xs = (torch.arange(Gx, dtype=dtype) - 0.5 * (Gx - 1)) * voxel_size_um
    ys = (torch.arange(Gy, dtype=dtype) - 0.5 * (Gy - 1)) * voxel_size_um
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    voxel_pos = torch.stack([XX.flatten(), YY.flatten(),
                              torch.zeros(G, dtype=dtype)], dim=-1)  # (G, 3)

    # Normalised position coords ∈ [-0.5, 0.5] for gradient interpolation.
    if R_gradient_dir == "x":
        u_R = (XX - XX.min()) / (XX.max() - XX.min() + 1e-12) - 0.5
    elif R_gradient_dir == "y":
        u_R = (YY - YY.min()) / (YY.max() - YY.min() + 1e-12) - 0.5
    elif R_gradient_dir == "diag":
        D = (XX + YY); u_R = (D - D.min()) / (D.max() - D.min() + 1e-12) - 0.5
    else:
        raise ValueError(f"unknown R_gradient_dir: {R_gradient_dir}")

    if eps_gradient_dir == "x":
        u_e = (XX - XX.min()) / (XX.max() - XX.min() + 1e-12) - 0.5
    elif eps_gradient_dir == "y":
        u_e = (YY - YY.min()) / (YY.max() - YY.min() + 1e-12) - 0.5
    elif eps_gradient_dir == "diag":
        D = (XX + YY); u_e = (D - D.min()) / (D.max() - D.min() + 1e-12) - 0.5
    else:
        raise ValueError(f"unknown eps_gradient_dir: {eps_gradient_dir}")

    u_R = u_R.flatten()
    u_e = u_e.flatten()

    # Per-voxel axis-angle perturbation around R_avg.
    R_axis = torch.tensor(R_gradient_axis, dtype=dtype)
    R_axis = R_axis / R_axis.norm().clamp_min(1e-12)
    aa_amp_rad = math.radians(R_gradient_amp_deg)
    aa_per_voxel = R_axis.unsqueeze(0) * (aa_amp_rad * u_R).unsqueeze(-1)  # (G, 3)
    if R_avg is None:
        R_avg = torch.eye(3, dtype=dtype)
    else:
        R_avg = R_avg.to(dtype)
    delta_R = _aa_to_R(aa_per_voxel)                                # (G, 3, 3)
    R_voxel = R_avg.unsqueeze(0) @ delta_R                          # (G, 3, 3)

    # Per-voxel Voigt strain.
    eps_avg_t = torch.tensor(eps_avg, dtype=dtype)                  # (6,)
    eps_voxel = eps_avg_t.unsqueeze(0).repeat(G, 1)
    if eps_gradient_amp != 0.0:
        eps_voxel[:, eps_gradient_voigt] += eps_gradient_amp * u_e

    lattice_t = torch.tensor(lattice, dtype=dtype)                  # (6,)

    return SinglePhaseGrainPlant(
        voxel_pos=voxel_pos,
        R_voxel=R_voxel,
        eps_voxel=eps_voxel,
        lattice=lattice_t,
        R_avg=R_avg,
        grid_shape=grid_shape,
        metadata={
            "voxel_size_um": float(voxel_size_um),
            "R_gradient_axis": tuple(map(float, R_gradient_axis)),
            "R_gradient_amp_deg": float(R_gradient_amp_deg),
            "R_gradient_dir": R_gradient_dir,
            "eps_avg": tuple(map(float, eps_avg)),
            "eps_gradient_voigt": int(eps_gradient_voigt),
            "eps_gradient_amp": float(eps_gradient_amp),
            "eps_gradient_dir": eps_gradient_dir,
            "seed": seed,
        },
    )


@dataclass
class GrainPatchData:
    """Output of :func:`simulate_grain_patches` — what the inversion
    driver consumes.

    Shapes (S = #spots = 2M, Σ = #scans):
        anchor_y/z/f : (S,)
        scan_positions: (Σ,)
        spot_valid_v : (G, S) — which voxels' v-th spot was valid (on-detector + eta + frame)
        measured_patches: (S, Σ, F, P, P)
        spot_indexer: (S,) long — into the flat 2M layout
    """
    anchor_y: torch.Tensor
    anchor_z: torch.Tensor
    anchor_f: torch.Tensor
    scan_positions: torch.Tensor
    measured_patches: torch.Tensor
    spot_valid: torch.Tensor                 # (G, S) — predicted valid per voxel-per-spot
    spot_observed: torch.Tensor              # (S,)  — bool: any voxel reached this spot
    spot_indexer: torch.Tensor               # (S,) long
    sigma_yz: float
    sigma_f: float
    patch_F: int
    patch_P: int
    # P2-7: per-pixel saturation mask, same shape as measured_patches;
    # True/1 = pixel VALID (below the detector clamp), 0 = saturated →
    # weight 0 in the data MSE and in the per-spot scale fit. None = no
    # saturation handling (synthetic data / threshold unknown).
    saturation_mask: "Optional[torch.Tensor]" = None


def _voxel_summed_spots_eager(
    model: HEDMForwardModel,
    R_voxel: torch.Tensor,
    eps_voxel: torch.Tensor,
    lattice: torch.Tensor,
    voxel_pos: torch.Tensor,
    apply_scan_filter: bool,
):
    """Tensor-only inner of _voxel_summed_spots (compile target).

    The ``apply_scan_filter`` branch is static (passed by the caller) so
    dynamo specialises the two paths separately — fine for compile.
    """
    G = R_voxel.shape[0]
    lat_per_v = lattice.to(R_voxel.dtype).unsqueeze(0).expand(G, 6)
    hkls_cart, thetas = model.correct_hkls_latc(lat_per_v, strain=eps_voxel)
    OM = R_voxel.unsqueeze(1)                                   # (G, 1, 3, 3)
    omega, eta, two_theta, valid = model.calc_bragg_geometry(OM, hkls_cart, thetas)
    pos = voxel_pos.unsqueeze(1).to(R_voxel.dtype)              # (G, 1, 3)
    spots = model.project_to_detector(omega, eta, two_theta, pos, valid)
    if apply_scan_filter and model.scan_config is not None:
        spots = model.filter_by_scan(spots, pos)
    return spots


def _voxel_summed_spots(
    model: HEDMForwardModel,
    R_voxel: torch.Tensor,            # (G, 3, 3)
    eps_voxel: torch.Tensor,          # (G, 6)
    lattice: torch.Tensor,            # (6,)
    voxel_pos: torch.Tensor,          # (G, 3)
    *,
    apply_scan_filter: bool = False,
):
    """Forward all G voxels with their own (R, ε) at one shared lattice.

    Returns
    -------
    SpotDescriptors with leading dim G; y_pixel/z_pixel/frame_nr/valid
    have shape (G, 2, M). When ``apply_scan_filter`` is True and the
    model has a ``scan_config``, the SpotDescriptors gets a
    ``scan_mask`` of shape (Σ, G, 2, M).

    Routed through ``_maybe_compile`` so callers who built the model
    with ``compile=True`` get inductor fusion across the diffraction core.
    """
    from midas_grain_odf.forward_helpers import _maybe_compile
    fn = _maybe_compile(model, "pf_voxel_summed_spots", _voxel_summed_spots_eager)
    return fn(model, R_voxel, eps_voxel, lattice, voxel_pos, apply_scan_filter)


def simulate_grain_patches(
    plant: SinglePhaseGrainPlant,
    model: HEDMForwardModel,
    *,
    patch_F: int = 5,
    patch_P: int = 15,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    gate_tau_um: float = 0.5,
    add_noise_sigma: float = 0.0,
    seed: Optional[int] = None,
    splat_radius_yz: int = 2,
    splat_radius_f: int = 1,
    voxel_spread: Optional[torch.Tensor] = None,
    voxel_strain_spread: Optional[torch.Tensor] = None,
    spread_gain_yz: float = 1.0,
    spread_gain_f: float = 1.0,
    strain_spread_gain: float = 1.0,
    chunk_size_g: Optional[int] = None,
) -> GrainPatchData:
    """Forward-simulate the planted grain into a measured-patch tensor.

    Runs the joint forward (per-voxel R and ε with the plant's shared
    lattice), splats voxel-summed contributions into per-(spot, scan)
    patches, optionally adds Gaussian noise.

    Parameters
    ----------
    plant : SinglePhaseGrainPlant
    model : HEDMForwardModel
        Pre-built model with geometry, scan_config (mandatory for pf),
        hkls/thetas/hkls_int loaded. The simulator and the inverter
        should share the same model instance (or build matching ones)
        so the forward layers agree by construction.
    patch_F, patch_P : int
        Patch dimensions (frames × pixels²).
    sigma_yz, sigma_f : float
        Splat kernel widths.
    gate_tau_um : float
        Soft beam-membership transition width.
    add_noise_sigma : float
        Gaussian noise σ added to each pixel of ``measured_patches``.
        For Poisson-like behavior at low counts, scale measured first
        and add Gaussian post-hoc.
    seed : int, optional
        For noise reproducibility.

    Returns
    -------
    GrainPatchData
    """
    if model.scan_config is None:
        raise ValueError(
            "simulate_grain_patches requires model.scan_config (pf-HEDM)."
        )
    device = next(iter(model.buffers())).device
    dtype = plant.R_voxel.dtype
    plant = plant.to(device, dtype=dtype)

    # The simulator returns detached data — never need an autograd graph.
    # Wrapping the entire forward in no_grad cuts peak memory ~50% and is
    # essential for production-scale plants (50×50+) on GPU.
    with torch.no_grad():
        return _simulate_grain_patches_no_grad(
            plant, model,
            patch_F=patch_F, patch_P=patch_P,
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            gate_tau_um=gate_tau_um,
            add_noise_sigma=add_noise_sigma,
            seed=seed,
            splat_radius_yz=splat_radius_yz,
            splat_radius_f=splat_radius_f,
            voxel_spread=voxel_spread,
            voxel_strain_spread=voxel_strain_spread,
            spread_gain_yz=spread_gain_yz,
            spread_gain_f=spread_gain_f,
            strain_spread_gain=strain_spread_gain,
            chunk_size_g=chunk_size_g,
            device=device, dtype=dtype,
        )


def _simulate_grain_patches_no_grad(
    plant, model, *, patch_F, patch_P, sigma_yz, sigma_f,
    gate_tau_um, add_noise_sigma, seed, splat_radius_yz, splat_radius_f,
    voxel_spread, spread_gain_yz, spread_gain_f,
    chunk_size_g, device, dtype,
    voxel_strain_spread=None, strain_spread_gain=1.0,
):
    """Inner body of :func:`simulate_grain_patches`. Always called inside
    a ``torch.no_grad()`` block; trades autograd for memory."""
    # Forward all voxels.
    spots = _voxel_summed_spots(
        model, plant.R_voxel, plant.eps_voxel, plant.lattice,
        plant.voxel_pos, apply_scan_filter=False,
    )
    # y_pixel etc. shape: (G, 2, M). Flatten to (G, 2*M) = (G, S).
    G = plant.n_voxels
    M = int(spots.y_pixel.shape[-1])
    S = 2 * M
    sy = spots.y_pixel.reshape(G, S)
    sz = spots.z_pixel.reshape(G, S)
    sf = spots.frame_nr.reshape(G, S)
    sv = spots.valid.reshape(G, S).to(dtype)
    sw = spots.omega.reshape(G, S)                               # (G, S)

    # Soft beam gate per (V, s, σ): sigmoid((BeamSize/2 − |y_rot − pos[σ]|) / τ)
    # y_rot = px·sin(ω) + py·cos(ω) per filter_by_scan convention.
    px = plant.voxel_pos[:, 0:1].expand(G, S)                    # (G, S)
    py = plant.voxel_pos[:, 1:2].expand(G, S)
    y_rot = px * torch.sin(sw) + py * torch.cos(sw)              # (G, S)
    sc = model.scan_config
    beam_y = sc.beam_positions.to(device).to(dtype)              # (Σ,)
    half = float(sc.beam_size) / 2.0
    diff = y_rot.unsqueeze(-1) - beam_y                          # (G, S, Σ)
    gate = torch.sigmoid((half - diff.abs()) / max(gate_tau_um, 1e-6))   # (G, S, Σ)

    # Anchor each (s, σ) cell at the voxel-mean predicted (y, z, f).
    # In the pristine plant case anchors come from the planted truth; with
    # noise / mis-calibration the inverter receives them as observed.
    sv_sum = sv.sum(dim=0).clamp(min=1e-9)                       # (S,)
    anchor_y = (sy * sv).sum(dim=0) / sv_sum                     # (S,)
    anchor_z = (sz * sv).sum(dim=0) / sv_sum
    anchor_f = (sf * sv).sum(dim=0) / sv_sum
    spot_observed = sv_sum > 0.5

    # Splat: K = G voxels, S* = S × Σ targets. Per-(V, s, σ) weight =
    # gate(V,s,σ) · sv(V,s); fold into ``valid`` so ``weights = ones(G)``.
    Sigma = beam_y.numel()
    Sstar = S * Sigma
    sy_flat = sy.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    sz_flat = sz.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    sf_flat = sf.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    valid_eff = (gate * sv.unsqueeze(-1)).reshape(G, Sstar)

    # Anchors per (s, σ): same anchor across σ (peak position is fixed by ω,h).
    a_y = anchor_y.unsqueeze(-1).expand(S, Sigma).reshape(Sstar)
    a_z = anchor_z.unsqueeze(-1).expand(S, Sigma).reshape(Sstar)
    a_f = anchor_f.unsqueeze(-1).expand(S, Sigma).reshape(Sstar)

    from midas_grain_odf.spot_extract import (
        SpotPatchSpec, splat_spots_to_patches_sparse,
    )
    spec = SpotPatchSpec(
        n_spots=Sstar,
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=a_y, anchor_z=a_z, anchor_f=a_f,
    )
    # Per-voxel intra-voxel spread → quadrature-broadened, intensity-conserving.
    s_yz_row = s_f_row = s_radial_row = s_eta_row = None
    if voxel_strain_spread is not None:
        # Anisotropic regime: σ_θ broadens (eta, ω); σ_ε broadens radial.
        # Needs spec.radial_y/z computed from anchor positions + beam center.
        sp_theta = (voxel_spread.to(dtype).to(device).reshape(G)
                    if voxel_spread is not None
                    else torch.zeros(G, dtype=dtype, device=device))
        sp_eps = voxel_strain_spread.to(dtype).to(device).reshape(G)
        s_radial_row = torch.sqrt(sigma_yz ** 2
                                  + (strain_spread_gain * sp_eps) ** 2)
        s_eta_row = torch.sqrt(sigma_yz ** 2
                               + (spread_gain_yz * sp_theta) ** 2)
        s_f_row = torch.sqrt(sigma_f ** 2 + (spread_gain_f * sp_theta) ** 2)
        y_BC = float(model.y_BC if not isinstance(model.y_BC, (list, tuple))
                     else model.y_BC[0])
        z_BC = float(model.z_BC if not isinstance(model.z_BC, (list, tuple))
                     else model.z_BC[0])
        dyR = a_y - y_BC
        dzR = a_z - z_BC
        rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
        spec.radial_y = (dyR / rR)
        spec.radial_z = (dzR / rR)
    elif voxel_spread is not None:
        sp = voxel_spread.to(dtype).to(device).reshape(G)
        s_yz_row = torch.sqrt(sigma_yz ** 2 + (spread_gain_yz * sp) ** 2)
        s_f_row = torch.sqrt(sigma_f ** 2 + (spread_gain_f * sp) ** 2)

    eff_chunk = chunk_size_g if (chunk_size_g is not None
                                  and chunk_size_g < G) else None
    if eff_chunk is None:
        weights_one = torch.ones(G, dtype=dtype, device=device)
        pred_patches = splat_spots_to_patches_sparse(
            spec, sy_flat, sz_flat, sf_flat, weights_one, valid_eff,
            radius_yz=splat_radius_yz, radius_f=splat_radius_f,
            sigma_yz_row=s_yz_row, sigma_f_row=s_f_row,
            sigma_radial_row=s_radial_row, sigma_eta_row=s_eta_row,
        )
    else:
        # Chunk over G voxels and sum partial-splats. No autograd needed
        # in the simulator; saves the K·S·L intermediate memory.
        pred_patches = None
        for g0 in range(0, G, eff_chunk):
            g1 = min(g0 + eff_chunk, G)
            wt_chunk = torch.ones(g1 - g0, dtype=dtype, device=device)
            chunk_pred = splat_spots_to_patches_sparse(
                spec, sy_flat[g0:g1], sz_flat[g0:g1], sf_flat[g0:g1],
                wt_chunk, valid_eff[g0:g1],
                radius_yz=splat_radius_yz, radius_f=splat_radius_f,
                sigma_yz_row=(s_yz_row[g0:g1] if s_yz_row is not None else None),
                sigma_f_row=(s_f_row[g0:g1] if s_f_row is not None else None),
                sigma_radial_row=(s_radial_row[g0:g1]
                                  if s_radial_row is not None else None),
                sigma_eta_row=(s_eta_row[g0:g1]
                                if s_eta_row is not None else None),
            )
            pred_patches = (chunk_pred if pred_patches is None
                            else pred_patches + chunk_pred)
    measured = pred_patches.reshape(S, Sigma, patch_F, patch_P, patch_P)

    if add_noise_sigma > 0.0:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(int(seed))
        noise = torch.randn(measured.shape, generator=gen,
                            dtype=dtype, device=device) * add_noise_sigma
        measured = measured + noise

    return GrainPatchData(
        anchor_y=anchor_y.detach(),
        anchor_z=anchor_z.detach(),
        anchor_f=anchor_f.detach(),
        scan_positions=beam_y.detach(),
        measured_patches=measured.detach(),
        spot_valid=sv.detach().to(torch.bool),
        spot_observed=spot_observed.detach(),
        spot_indexer=torch.arange(S, dtype=torch.long, device=device),
        sigma_yz=float(sigma_yz),
        sigma_f=float(sigma_f),
        patch_F=int(patch_F),
        patch_P=int(patch_P),
    )
