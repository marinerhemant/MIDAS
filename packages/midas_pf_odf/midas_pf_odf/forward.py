"""Joint per-grain forward — voxel-summed splat per (spot, scan).

Parallel to ``simulate._voxel_summed_spots`` but factored as a pure
function to be called repeatedly inside the optimizer. Output shape
matches ``GrainPatchData.measured_patches``: ``(S, Σ, F, P, P)``.

The forward chain at one optimizer step:
    R_voxel       (G, 3, 3)          — R_init @ exp(skew(δ))
    eps_voxel     (G, 6)             — Voigt strain (after identifiability projection)
    lattice       (6,)               — global per-grain lattice
    voxel_pos     (G, 3)             — fixed sample positions
        ↓ correct_hkls_latc
    hkls_cart     (G, M, 3), thetas (G, M)
        ↓ calc_bragg_geometry
    omega/eta/2θ/valid               (G, 2, M)
        ↓ project_to_detector
    y_pixel/z_pixel/frame_nr/valid   (G, 2, M)
        ↓ soft beam gate per (V, s, σ)
    gate          (G, S, Σ)          — S = 2M
        ↓ voxel-summed splat (folded into ``valid_eff``)
    pred_patches  (S, Σ, F, P, P)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from midas_diffract.forward import HEDMForwardModel, ScanConfig
from midas_grain_odf.forward_helpers import _maybe_compile
from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches


def soft_beam_gate(
    voxel_pos: torch.Tensor,
    omega: torch.Tensor,
    beam_positions: torch.Tensor,
    beam_size_um: float,
    tau_um: float = 0.5,
) -> torch.Tensor:
    """Differentiable beam-membership gate.

    Returns a value in (0, 1) that approximates the hard gate
    ``|y_rot - beam_pos| < BeamSize/2`` from
    ``FF_HEDM/src/FitOrStrainsScanningOMP.c:1050-1058`` while staying
    differentiable in the voxel position (and hence in any orientation
    parameter that enters the forward through ``omega``).

    Convention matches ``HEDMForwardModel.filter_by_scan``:
        y_rot = px sin(ω) + py cos(ω)

    Parameters
    ----------
    voxel_pos : Tensor (..., G, 3)
    omega     : Tensor (..., G, S)        — radians
    beam_positions : Tensor (Σ,)
    beam_size_um   : float
    tau_um         : float                — transition width in µm

    Returns
    -------
    gate : Tensor (..., G, S, Σ)
    """
    px = voxel_pos[..., 0:1]                                     # (..., G, 1)
    py = voxel_pos[..., 1:2]
    y_rot = px * torch.sin(omega) + py * torch.cos(omega)        # (..., G, S)
    diff = y_rot.unsqueeze(-1) - beam_positions                  # (..., G, S, Σ)
    half = beam_size_um / 2.0
    return torch.sigmoid((half - diff.abs()) / max(float(tau_um), 1e-6))


@dataclass
class JointForwardOutput:
    pred_patches: torch.Tensor          # (S, Σ, F, P, P)
    spot_valid: torch.Tensor            # (G, S) per-voxel-per-spot validity
    omega_g: torch.Tensor               # (G, S) — for diagnostics


def _pf_diffraction_core_eager(
    model: HEDMForwardModel,
    R_voxel: torch.Tensor,
    eps_voxel: torch.Tensor,
    lattice: torch.Tensor,
    voxel_pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-only diffraction core of joint_grain_forward (compile target).

    Returns (sy, sz, sf, sv, sw) each of shape (G, S=2M) where M is the
    reflection count. The post-projection beam gate and patch splat live
    outside the compiled scope so the heavy scatter/checkpoint machinery
    is not retraced per shape.
    """
    G = R_voxel.shape[0]
    dtype = R_voxel.dtype
    lat_per_v = lattice.to(dtype).unsqueeze(0).expand(G, 6)
    hkls_cart, thetas = model.correct_hkls_latc(lat_per_v, strain=eps_voxel)
    OM = R_voxel.unsqueeze(1)                                     # (G, 1, 3, 3)
    omega, eta, two_theta, valid = model.calc_bragg_geometry(
        OM, hkls_cart, thetas
    )                                                              # (G, 2, M)
    pos = voxel_pos.unsqueeze(1).to(dtype)                         # (G, 1, 3)
    spots = model.project_to_detector(omega, eta, two_theta, pos, valid)
    M = int(omega.shape[-1])
    S = 2 * M
    sy = spots.y_pixel.reshape(G, S)
    sz = spots.z_pixel.reshape(G, S)
    sf = spots.frame_nr.reshape(G, S)
    sv = spots.valid.reshape(G, S).to(dtype)
    sw = spots.omega.reshape(G, S)
    return sy, sz, sf, sv, sw


def joint_grain_forward(
    model: HEDMForwardModel,
    R_voxel: torch.Tensor,              # (G, 3, 3)
    eps_voxel: torch.Tensor,            # (G, 6)
    lattice: torch.Tensor,              # (6,)
    voxel_pos: torch.Tensor,            # (G, 3)
    *,
    spec: SpotPatchSpec,                # n_spots = S × Σ
    n_scans: int,
    scan_config: Optional[ScanConfig] = None,
    gate_tau_um: float = 0.5,
    splat_radius_yz: int = 2,
    splat_radius_f: int = 1,
    voxel_spread: Optional[torch.Tensor] = None,
    voxel_strain_spread: Optional[torch.Tensor] = None,
    spread_gain_yz: float = 1.0,
    spread_gain_f: float = 1.0,
    strain_spread_gain: float = 1.0,
    strain_spread_per_spot_scale: Optional[torch.Tensor] = None,
    chunk_size_g: Optional[int] = None,
    spot_offset: Optional[torch.Tensor] = None,
) -> JointForwardOutput:
    """Run the joint forward for one optimizer step.

    The splatter spec must already have anchors ``anchor_y/z/f`` of
    shape ``(S × Σ,)`` with the per-(s, σ) target patch centers.

    The voxel-summed splat output is reshaped to ``(S, Σ, F, P, P)``.
    """
    G = R_voxel.shape[0]
    dtype = R_voxel.dtype
    device = R_voxel.device

    # The diffraction core (correct_hkls_latc + calc_bragg_geometry +
    # project_to_detector) is a long elementwise pipeline; route it through
    # _maybe_compile so that callers who built the model with compile=True
    # get inductor fusion here. The post-projection splat is left eager.
    diff_fn = _maybe_compile(model, "pf_diffraction_core", _pf_diffraction_core_eager)
    sy, sz, sf, sv, sw = diff_fn(model, R_voxel, eps_voxel, lattice, voxel_pos)
    M = sy.shape[-1] // 2
    S = 2 * M

    # Per-spot detector position offset (Δy, Δz), shared across all G voxels:
    # absorbs the systematic per-reflection position residual (distortion / tilt /
    # beam-center / lattice-reference) the way midas-fit-grain records per-spot
    # residuals. Shared per-spot (NOT per-voxel) so it soaks up the grain-MEAN
    # residual without competing with the per-voxel strain (which is the
    # voxel-to-voxel DEVIATION). None => no-op (existing behaviour).
    if spot_offset is not None:
        sy = sy + spot_offset[:, 0].reshape(1, S)
        sz = sz + spot_offset[:, 1].reshape(1, S)

    # Soft beam gate
    sc = scan_config if scan_config is not None else model.scan_config
    if sc is None:
        # No scan config — emit a fake single-scan gate of all ones.
        gate = torch.ones(G, S, n_scans, dtype=dtype, device=device)
    else:
        gate = soft_beam_gate(
            voxel_pos.to(dtype), sw,
            sc.beam_positions.to(device).to(dtype),
            float(sc.beam_size), float(gate_tau_um),
        )                                                          # (G, S, Σ)

    Sigma = n_scans
    Sstar = S * Sigma

    sy_flat = sy.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    sz_flat = sz.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    sf_flat = sf.unsqueeze(-1).expand(G, S, Sigma).reshape(G, Sstar)
    valid_eff = (gate * sv.unsqueeze(-1)).reshape(G, Sstar)

    # Per-voxel peak-width spread (intra-voxel strain/orientation distribution):
    # adds in quadrature to the instrumental width, conserving integrated
    # intensity (handled in the splatter). spread is in patch px/frame units.
    #
    # Two regimes:
    #  (a) isotropic — voxel_spread set, voxel_strain_spread=None: ONE spread
    #      scalar per voxel, broadens (y, z, ω) uniformly.
    #  (b) anisotropic — voxel_strain_spread set: σ_θ (orientation) broadens
    #      tangential (eta) + ω; σ_ε (microstrain) broadens radial only.
    #      Requires spec.radial_y / radial_z populated (per-spot detector
    #      unit vectors from beam center → spot).
    s_yz_row = s_f_row = s_radial_row = s_eta_row = None
    anisotropic = (
        voxel_strain_spread is not None
        and spec.radial_y is not None and spec.radial_z is not None
    )
    if anisotropic:
        sp_theta = (voxel_spread.to(dtype).to(device).reshape(G)
                    if voxel_spread is not None
                    else torch.zeros(G, dtype=dtype, device=device))
        sp_eps = voxel_strain_spread.to(dtype).to(device).reshape(G)
        # σ_ε regime:
        #  (i)  pixel-units (default, backward-compat) — strain_spread_per_spot_scale
        #       is None; sp_eps is a per-voxel pixel width applied uniformly across
        #       all spots. Ring-averaged ⟨tan θ⟩ post-hoc conversion to microstrain.
        #  (ii) microstrain-rigorous — strain_spread_per_spot_scale (S,) is the
        #       per-spot pixel-width-per-unit-strain factor 2·Lsd·tan(θ_s)/px_um.
        #       sp_eps is then dimensionless microstrain; per-(voxel,spot) pixel
        #       width = sp_eps[k] · c_s[s], rigorous Bragg-law coupling.
        if strain_spread_per_spot_scale is not None:
            # The splat operates on Sstar = S × Σ; caller pre-expands.
            c_s = strain_spread_per_spot_scale.to(dtype).to(device).reshape(Sstar)
            # per-(K, Sstar) microstrain pixel width
            sp_eps_kS = (strain_spread_gain * sp_eps).reshape(G, 1) * c_s.reshape(1, Sstar)
            # (G, Sstar) per-(voxel, spot) σ_radial
            s_radial_row = torch.sqrt(spec.sigma_yz ** 2 + sp_eps_kS ** 2)
            # σ_θ unchanged: per-voxel only
            s_eta_row = torch.sqrt(spec.sigma_yz ** 2
                                   + (spread_gain_yz * sp_theta) ** 2)
        else:
            s_radial_row = torch.sqrt(spec.sigma_yz ** 2
                                      + (strain_spread_gain * sp_eps) ** 2)
            s_eta_row = torch.sqrt(spec.sigma_yz ** 2
                                   + (spread_gain_yz * sp_theta) ** 2)
        s_f_row = torch.sqrt(spec.sigma_f ** 2
                             + (spread_gain_f * sp_theta) ** 2)
    elif voxel_spread is not None:
        # Width depends on spread² (even), so sign is irrelevant — no clamp,
        # which avoids an optimizer boundary trap at spread→0.
        sp = voxel_spread.to(dtype).to(device).reshape(G)
        s_yz_row = torch.sqrt(spec.sigma_yz ** 2 + (spread_gain_yz * sp) ** 2)
        s_f_row = torch.sqrt(spec.sigma_f ** 2 + (spread_gain_f * sp) ** 2)

    from midas_grain_odf.spot_extract import splat_spots_to_patches_sparse
    eff_chunk = chunk_size_g if (chunk_size_g is not None
                                  and chunk_size_g < G) else None
    if eff_chunk is None:
        weights_one = torch.ones(G, dtype=dtype, device=device)
        pred_flat = splat_spots_to_patches_sparse(
            spec, sy_flat, sz_flat, sf_flat, weights_one, valid_eff,
            radius_yz=splat_radius_yz, radius_f=splat_radius_f,
            sigma_yz_row=s_yz_row, sigma_f_row=s_f_row,
            sigma_radial_row=s_radial_row, sigma_eta_row=s_eta_row,
        )                                                          # (Sstar, F, P, P)
    else:
        # Chunk over G voxels and sum partial-splats. Use torch.utils.checkpoint
        # so each chunk's intermediate (chunk_size × S* × L) is freed after the
        # forward and rematerialised in backward — peak memory drops from
        # O(G·S*·L) to O(chunk·S*·L).
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _chunk_splat(sy_c, sz_c, sf_c, ve_c, syr_c, sfr_c, srad_c, seta_c):
            wt_c = torch.ones(sy_c.shape[0], dtype=dtype, device=device)
            return splat_spots_to_patches_sparse(
                spec, sy_c, sz_c, sf_c, wt_c, ve_c,
                radius_yz=splat_radius_yz, radius_f=splat_radius_f,
                sigma_yz_row=syr_c, sigma_f_row=sfr_c,
                sigma_radial_row=srad_c, sigma_eta_row=seta_c,
            )

        pred_flat = None
        for g0 in range(0, G, eff_chunk):
            g1 = min(g0 + eff_chunk, G)
            sy_c = sy_flat[g0:g1].contiguous()
            sz_c = sz_flat[g0:g1].contiguous()
            sf_c = sf_flat[g0:g1].contiguous()
            ve_c = valid_eff[g0:g1].contiguous()
            syr_c = s_yz_row[g0:g1].contiguous() if s_yz_row is not None else None
            sfr_c = s_f_row[g0:g1].contiguous() if s_f_row is not None else None
            srad_c = (s_radial_row[g0:g1].contiguous()
                      if s_radial_row is not None else None)
            seta_c = (s_eta_row[g0:g1].contiguous()
                      if s_eta_row is not None else None)
            chunk_pred = _ckpt(
                _chunk_splat, sy_c, sz_c, sf_c, ve_c, syr_c, sfr_c, srad_c, seta_c,
                use_reentrant=False,
            )
            pred_flat = chunk_pred if pred_flat is None else pred_flat + chunk_pred
    pred = pred_flat.reshape(S, Sigma, spec.patch_F, spec.patch_P, spec.patch_P)

    return JointForwardOutput(pred_patches=pred, spot_valid=sv, omega_g=sw)


def closed_form_per_spot_scale(
    pred: torch.Tensor,
    meas: torch.Tensor,
    spot_keep: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    pixel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Closed-form per-spot intensity scale c_s* (profile likelihood).

    For each spot s, sums numerator/denominator across (Σ, F, P, P)
    and returns ``c_s* = ⟨pred·meas⟩ / ⟨pred·pred⟩``. Shape (S,).

    ``spot_keep`` (S,) optional — when supplied, spots with keep=0 get
    c_s* = 1 (so the loss treats their patches as raw residual, no
    scale absorption).

    ``pixel_weight`` optional, broadcastable to ``meas`` — per-pixel
    weight (P2-7: 0 at saturated pixels). Without it, a 96%-saturated
    ring's flat-top clamps drag c_s* down and the whole fit chases an
    unattainable amplitude.
    """
    flat_dims = (1, 2, 3, 4)                                       # (Σ, F, P, P)
    if pixel_weight is not None:
        num = (pred * meas * pixel_weight).sum(dim=flat_dims)
        den = (pred * pred * pixel_weight).sum(dim=flat_dims).clamp(min=eps)
    else:
        num = (pred * meas).sum(dim=flat_dims)
        den = (pred * pred).sum(dim=flat_dims).clamp(min=eps)
    c = num / den
    if spot_keep is not None:
        c = torch.where(spot_keep > 0.5, c, torch.ones_like(c))
    return c
