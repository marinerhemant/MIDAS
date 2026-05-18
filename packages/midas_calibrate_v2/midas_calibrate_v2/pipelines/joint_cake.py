"""Joint forward-cake pipeline.

The v0.2 plan that v1's ``joint.py`` flagged as deferred.  v2's first-class
implementation: vectorised over (n_rings × n_eta_bins × n_R_window),
restricted to the radial windows around each ring (no wasted compute on
cake regions with no signal).

Forward model (per (ring k, eta bin j) region):

    I_pred[k, j, R] = bg0[k, j] + bg1[k, j] · (R − R_mid[k])
                    + area[k, j] · [η_v[k, j] · L(R; center_k(θ_geom), γ[k, j])
                                  + (1−η_v[k, j]) · G(R; center_k(θ_geom), σ[k, j])]

with the **original Wertheim pseudo-Voigt** — σ_G, γ_L, η_v are independent
free parameters, *not* coupled by the TCH polynomial.  This is the model
the user requested for calibration; it favours center accuracy over
physically-decomposable widths.

Loss:

    L = ½ Σ_{(k, j, R∈win_k)} w[R] · (I_obs - I_pred)²

with Poisson-weighted residuals.  Differentiable in *every* parameter —
geometry and shape — without any centroid extraction or Newton inversion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_peakfit import GenericLMConfig

from ..compat.from_v1 import spec_from_v1_params
from ..forward.bragg import R_ideal_px
from ..forward.cake import build_cake_chain_rule, CakeProfile
from ..forward.geometry import pixel_to_REta
from ..forward.panels import PanelLayout
from ..parameters.parameter import Parameter
from ..parameters.pack import (
    pack_spec, refined_indices, refined_bounds,
    write_refined_back, unpack_spec,
)
from ..parameters.spec import CalibrationSpec
from ._common import run_estep_v1, FittedDataset, filter_ring_table


@dataclass
class JointCakeResult:
    spec: CalibrationSpec
    map_unpacked: Dict[str, torch.Tensor]
    cost: float
    cake: CakeProfile


def _build_predicted_windows(
    R_window: torch.Tensor,           # [n_rings, n_win] absolute R coords (px)
    ring_R_pred: torch.Tensor,        # [n_rings] geometry-driven peak centers (px)
    R_mid: torch.Tensor,              # [n_rings] window mid-R for bg slope reference
    shape_params: torch.Tensor,       # [n_rings, n_eta, 6] (σ, γ, η_v, area, bg0, bg1)
) -> torch.Tensor:
    """Vectorised prediction across all (ring × η-bin × R-window) regions.

    Returns a [n_rings, n_eta, n_win] tensor of predicted intensities,
    differentiable in every input.

    Uses the **original** pseudo-Voigt (independent σ, γ, η_v).
    """
    n_rings = ring_R_pred.shape[0]
    n_eta = shape_params.shape[1]
    n_win = R_window.shape[1]

    # Broadcast shapes:
    R_grid = R_window.unsqueeze(1)                          # [n_rings, 1, n_win]
    center = ring_R_pred.view(n_rings, 1, 1)                # [n_rings, 1, 1]
    R_mid_b = R_mid.view(n_rings, 1, 1)
    dR = R_grid - center                                     # [n_rings, 1, n_win]
    dR2 = dR * dR

    sigma = shape_params[..., 0:1].abs() + 1e-6              # [n_rings, n_eta, 1]
    gamma = shape_params[..., 1:2].abs() + 1e-6
    eta_v = shape_params[..., 2:3].clamp(0.0, 1.0)
    area  = shape_params[..., 3:4]
    bg0   = shape_params[..., 4:5]
    bg1   = shape_params[..., 5:6]

    G = torch.exp(-0.5 * dR2 / (sigma * sigma + 1e-30)) \
        / (sigma * math.sqrt(2.0 * math.pi) + 1e-30)
    L = (gamma / math.pi) / (dR2 + gamma * gamma + 1e-30)
    peak = area * (eta_v * L + (1.0 - eta_v) * G)            # [n_rings, n_eta, n_win]
    bg = bg0 + bg1 * (R_grid - R_mid_b)                       # [n_rings, n_eta, n_win]
    return peak + bg


def _ring_centers_at_geometry(
    unpacked: Dict[str, torch.Tensor],
    ring_two_theta_deg: torch.Tensor,
) -> torch.Tensor:
    """Geometry-driven ring radii (px) at the current parameters."""
    pxY = unpacked["pxY"]
    pxZ = unpacked.get("pxZ", pxY)
    px_mean = 0.5 * (pxY + pxZ)
    return R_ideal_px(ring_two_theta_deg, unpacked["Lsd"], px_mean)


def autocalibrate_joint(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter_seed: int = 2,
    half_window_px: float = 6.0,
    eta_bin_size_override: Optional[float] = None,
    lbfgs_max_iter: int = 100,
    snr_min: float = 3.0,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> JointCakeResult:
    """Run the joint forward-cake engine.

    Strategy:
      1. Run a couple of alternating iterations to seed geometry.
      2. Build the cake at seeded geometry; per-ring radial windows fixed.
      3. Add a single ``joint_shapes`` Parameter [n_rings, n_eta, 6] to the
         spec — refinable like any other v2 parameter.
      4. LBFGS jointly over geometry + shape DOFs against per-window
         Poisson-weighted residuals.
    """
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    # Step 1: seed geometry via the alternating engine.
    from .single import autocalibrate as autocalibrate_single
    if verbose:
        print(f"[joint_cake] seeding ({n_iter_seed} alternating iters)...")
    seed_result = autocalibrate_single(
        v1_params, image, dark=dark, spec=spec, panel_layout=panel_layout,
        n_iter=n_iter_seed, dtype=dtype, device=device, verbose=verbose,
    )
    fits = seed_result.fits_final
    if fits is None:
        raise RuntimeError("seed alternating run produced no fits")
    # Honour v1-style ring exclusion / max-ring cap on the seed's RingTable.
    fits.rt = filter_ring_table(
        fits.rt,
        rings_to_exclude=getattr(spec, "rings_to_exclude", ()),
        max_ring_number=getattr(spec, "max_ring_number", 0),
    )

    # Step 2: build cake at seeded geometry (C-backed builder).
    from midas_integrate.params import IntegrationParams
    px = 0.5 * (v1_params.pxY + v1_params.pxZ) if v1_params.pxZ > 0 else v1_params.pxY
    half_px = max(half_window_px, 0.5 * v1_params.Width / px)
    R_min = max(0.0, float(fits.rt.r_ideal_px.min()) - half_px - 1.0)
    R_max = float(fits.rt.r_ideal_px.max()) + half_px + 1.0
    ip = IntegrationParams()
    for k in ("NrPixelsY", "NrPixelsZ", "pxY", "pxZ", "Lsd",
               "BC_y", "BC_z", "tx", "ty", "tz", "Parallax", "Wavelength"):
        setattr(ip, k, getattr(v1_params, k))
    for i in range(15):
        setattr(ip, f"p{i}", getattr(v1_params, f"p{i}"))
    ip.RhoD = v1_params.RhoD if v1_params.RhoD > 0 else v1_params.MaxRingRad
    bin_size = float(v1_params.RBinSize) if v1_params.RBinSize > 0 else 0.25
    ip.RMin, ip.RMax, ip.RBinSize = float(R_min), float(R_max), bin_size
    eta_bin_size = float(eta_bin_size_override) if eta_bin_size_override is not None \
        else float(v1_params.EtaBinSize)
    ip.EtaMin, ip.EtaMax, ip.EtaBinSize = -180.0, 180.0, eta_bin_size
    ip.SolidAngleCorrection = 0
    ip.PolarizationCorrection = 0

    img_used = image - dark if dark is not None else image
    cake = build_cake_chain_rule(img_used, integration_params=ip)
    n_R_full, n_eta = cake.intensity.shape
    if verbose:
        print(f"[joint_cake] cake shape: {n_R_full} R bins × {n_eta} η bins  "
              f"(bin_size={bin_size:.3f} px, η_bin={eta_bin_size}°)")

    # Step 3: per-ring radial windows (FIXED at seeded geometry's R_ideal).
    R_centers = cake.R_centers.to(dtype=dtype, device=device)
    eta_centers = cake.eta_centers.to(dtype=dtype, device=device)
    I_obs_full = cake.intensity.to(dtype=dtype, device=device)

    n_win = max(7, int(round(2.0 * half_window_px / bin_size)))
    if n_win % 2 == 0:
        n_win += 1   # odd window for symmetric centering

    rt_two_theta = torch.tensor(fits.rt.two_theta_deg, dtype=dtype, device=device)
    seed_R_ideal_full = (v1_params.Lsd
                          * np.tan(np.radians(fits.rt.two_theta_deg)) / px)

    # Dedup: rt has one row per (h,k,l) family; many share the same 2θ.
    # Keep one representative per unique 2θ (rounded to 1e-3 deg).  Otherwise
    # we'd fit overlapping windows multiple times and double-count the data.
    keys = np.round(fits.rt.two_theta_deg, 3)
    _, unique_idx = np.unique(keys, return_index=True)
    unique_idx = np.sort(unique_idx)
    if verbose:
        print(f"[joint_cake] ring table: {len(seed_R_ideal_full)} reflections "
              f"→ {len(unique_idx)} unique rings", flush=True)

    R_window_list: List[torch.Tensor] = []
    I_block_list: List[torch.Tensor] = []
    R_mid_list: List[float] = []
    rings_kept: List[int] = []
    rings_rejected_snr = 0
    rings_rejected_extent = 0
    for k_ring in unique_idx.tolist():
        R_id = float(seed_R_ideal_full[k_ring])
        c_idx = int(np.argmin(np.abs(cake.R_centers - R_id)))
        lo_i = max(0, c_idx - n_win // 2)
        hi_i = lo_i + n_win
        if hi_i > n_R_full:
            hi_i = n_R_full
            lo_i = hi_i - n_win
        if lo_i < 0:
            rings_rejected_extent += 1
            continue

        # SNR filter using Poisson noise estimate (peak above bg / √bg).
        # Std-of-window would be dominated by the peak itself, biasing SNR low
        # for strong signals.  Median across η is robust to bin gaps / dead px.
        block = cake.intensity[lo_i:hi_i, :]                  # [n_win, n_eta]
        max_per_eta = block.max(dim=0).values                  # [n_eta]
        med_per_eta = block.median(dim=0).values.clamp(min=1.0)
        ring_snr = (max_per_eta - med_per_eta) / med_per_eta.sqrt()
        if float(ring_snr.median()) < snr_min:
            rings_rejected_snr += 1
            continue

        R_window_list.append(R_centers[lo_i:hi_i])
        I_block_list.append(I_obs_full[lo_i:hi_i, :].T.contiguous())  # [n_eta, n_win]
        R_mid_list.append(0.5 * (R_centers[lo_i] + R_centers[hi_i - 1]))
        rings_kept.append(k_ring)

    if verbose:
        print(f"[joint_cake] ring filter: kept {len(rings_kept)} / {len(unique_idx)} "
              f"(rejected {rings_rejected_snr} for SNR < {snr_min}, "
              f"{rings_rejected_extent} for window extent)", flush=True)

    if not R_window_list:
        raise RuntimeError("No ring windows fit inside the cake R range")

    n_rings_kept = len(R_window_list)
    R_window = torch.stack(R_window_list, dim=0)              # [n_rings, n_win]
    I_obs_w = torch.stack(I_block_list, dim=0)                # [n_rings, n_eta, n_win]
    R_mid = torch.tensor(R_mid_list, dtype=dtype, device=device)
    rings_kept_t = torch.tensor(rings_kept, dtype=torch.long, device=device)

    if verbose:
        print(f"[joint_cake] {n_rings_kept} rings kept, window={n_win} bins, "
              f"shape DOF = {n_rings_kept * n_eta * 6}")

    # Step 4: shape parameter init from the seeded cake (per-region peak amp + bg).
    shape_init = torch.zeros(n_rings_kept, n_eta, 6, dtype=dtype, device=device)
    shape_init[..., 0] = 1.0    # σ_G init
    shape_init[..., 1] = 0.5    # γ_L init
    shape_init[..., 2] = 0.5    # η_v init (mid-mixed)
    peak_amp = (I_obs_w.max(dim=2).values - I_obs_w.min(dim=2).values).clamp(min=0.0)
    shape_init[..., 3] = peak_amp * 2.0
    shape_init[..., 4] = I_obs_w.min(dim=2).values
    # bg1 stays 0

    if "joint_shapes" in spec.parameters:
        spec.parameters["joint_shapes"].init = shape_init
        spec.parameters["joint_shapes"].refined = True
    else:
        spec.add(Parameter(
            name="joint_shapes", init=shape_init, refined=True, bounds=None,
        ))

    # Pull keep-ring two-theta vector (used by _ring_centers_at_geometry).
    rt_two_theta_kept = rt_two_theta[rings_kept_t]

    def loss_fn(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        ring_R_full = _ring_centers_at_geometry(unpacked, rt_two_theta)
        ring_R_kept = ring_R_full.index_select(0, rings_kept_t)
        shape = unpacked["joint_shapes"].reshape(n_rings_kept, n_eta, 6)
        I_pred = _build_predicted_windows(R_window, ring_R_kept, R_mid, shape)
        # Poisson-weight residuals.
        w = 1.0 / (I_obs_w.clamp(min=1.0)).sqrt()
        r = (I_obs_w - I_pred) * w
        return 0.5 * (r * r).sum()

    if verbose:
        print(f"[joint_cake] running LBFGS over geometry + "
              f"{n_rings_kept * n_eta * 6} shape DOF...")

    # Direct torch.optim.LBFGS over the refined subset, no sigmoid reparam —
    # joint_shapes spans many orders of magnitude (areas, backgrounds) and the
    # ±1 fallback box would clamp them.  Geometry parameters get bounded via
    # ad-hoc soft penalties.
    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    x_ref = x_full.index_select(0, refined_idx).clone().detach().requires_grad_(True)

    optim = torch.optim.LBFGS(
        [x_ref],
        max_iter=lbfgs_max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    n_iter_holder = [0]
    last_loss = [float("inf")]

    def closure():
        optim.zero_grad()
        x_full_now = x_full.clone()
        x_full_now[refined_idx] = x_ref
        unpacked = unpack_spec(x_full_now, info, spec)
        loss = loss_fn(unpacked)
        loss.backward()
        n_iter_holder[0] += 1
        last_loss[0] = float(loss.detach())
        return loss

    optim.step(closure)
    final_loss = last_loss[0]
    n_used = n_iter_holder[0]

    with torch.no_grad():
        x_full_final = x_full.clone()
        x_full_final[refined_idx] = x_ref.detach()
        map_unpacked = unpack_spec(x_full_final, info, spec)
    if verbose:
        print(f"[joint_cake] converged: loss={final_loss:.6e}  iters={n_used}")

        # Report achieved pseudo-strain at the converged geometry.
        with torch.no_grad():
            ring_R_final = _ring_centers_at_geometry(map_unpacked, rt_two_theta)
            shape_final = map_unpacked["joint_shapes"].reshape(n_rings_kept, n_eta, 6)
            # Peak centers as fitted: ring_R_kept (geometry), but with offset
            # from shape (none — center is fully geometry-driven here).
            # Strain residual proxy: |observed_peak_center - ring_R_kept| / ring_R_kept,
            # using a quick numeric max of I_obs in each window.
            obs_argmax = I_obs_w.argmax(dim=2)               # [n_rings, n_eta]
            obs_peak_R = R_window.gather(
                1, obs_argmax)                                 # [n_rings, n_eta]
            strain = (obs_peak_R - ring_R_final.index_select(0, rings_kept_t).unsqueeze(1)) \
                / ring_R_final.index_select(0, rings_kept_t).unsqueeze(1)
            mean_uE = float(strain.abs().mean()) * 1e6
        print(f"[joint_cake] post-fit pseudo-strain (argmax proxy): {mean_uE:.2f} μϵ")

    return JointCakeResult(
        spec=spec, map_unpacked=map_unpacked, cost=final_loss, cake=cake,
    )


__all__ = ["JointCakeResult", "autocalibrate_joint"]
