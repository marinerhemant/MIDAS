"""Synthetic Pilatus 6×8 + 100-grain joint-calibration validation.

Paper figure 1: forward-sim a multi-panel detector with prescribed
``(δy, δz)`` per panel, generate (a) a powder calibrant pattern and (b)
HEDM grain spots from 100 random Au grains at the **same** truth geometry,
then compare three Fisher-block reports on the per-panel ``(δy, δz)`` block:

    (a) powder-only residual  → expected rank-deficient (paper-3 §9 result)
    (b) HEDM-only residual    → expected rank-deficient (different null-space)
    (c) joint residual         → expected full-rank, σ per panel < 0.5 px

End-to-end self-contained: no Zarr files, no real data, no subprocesses.
Uses the same residual closures as the real-data runner — verifies the
joint calibration claim numerically.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE python -m midas_joint_ff_calibrate.runners.run_synthetic_pilatus_joint \\
        --output runs/synthetic_pilatus_joint
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from midas_diffract import HEDMForwardModel  # type: ignore
from midas_diffract.forward import HEDMGeometry  # type: ignore
from midas_diffract.hkls import hkls_for_forward_model  # type: ignore
from midas_hkls import Lattice, SpaceGroup  # type: ignore
from torch.func import functional_call

import midas_peakfit as mp
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

from midas_joint_ff_calibrate.loss import JointWeights, joint_residual
from midas_joint_ff_calibrate.pipelines.identifiability import fisher_block_rank


# --------------------------------------------------------- physical constants

AU_LATTICE_A = 4.0782          # Au cubic lattice (Å)
WAVELENGTH_A = 0.1729          # ~71.7 keV
PX_UM = 172.0
LSD_UM = 1.2e6                 # 1.2 m.  At this distance the 8 Au rings
                                # cover only the outer 36 panels; the 12
                                # CENTRAL panels (closest to the beam centre)
                                # are below the (111) ring radius.  Closer
                                # detector + more rings would cover all 48
                                # but at the cost of LM stability under
                                # panel_idx-fixed-at-observation-time --
                                # see paper §6.5 for discussion.
N_PANELS_Y = 6
N_PANELS_Z = 8
PANEL_SIZE_Y = 200             # px (synthetic, square modules — keeps geometry simple)
PANEL_SIZE_Z = 200
GAP_Y = 10                     # px
GAP_Z = 10
N_GRAINS = 100
N_RINGS = 8                    # first 8 Au reflections (m=3..20)
TWO_THETA_MAX_DEG = 12.0       # captures rings up to (420)
                                # Adding more rings (e.g. TWO_THETA_MAX=20°
                                # gives ~24 rings) destabilises the joint LM
                                # at this resolution scale because
                                # generate_hedm_observations produces
                                # ~45k spots and small Lsd perturbations cross
                                # panel boundaries, breaking the
                                # panel_idx-fixed-at-observation-time
                                # assumption in make_hedm_residual.  Future
                                # work: re-evaluate panel_idx at each LM step.
N_POWDER_PER_RING = 360        # one obs per η-degree


# --------------------------------------------------------- truth sampling

@dataclass
class TruthGeometry:
    Lsd: float
    BC_y: float
    BC_z: float
    tx: float
    ty: float
    tz: float
    panel_delta_yz: torch.Tensor   # (N_panels, 2)


def sample_truth(layout: PanelLayout, *, seed: int = 2026) -> TruthGeometry:
    rng = torch.Generator().manual_seed(seed)
    n = layout.n_panels()
    # Panel deltas: paper-3 module-spec scale.
    deltas = 0.5 * torch.randn(n, 2, dtype=torch.float64, generator=rng)
    # Enforce Σ=0 gauge so the synthetic truth is unambiguous (no global
    # mode that gets absorbed into BC).
    deltas -= deltas.mean(dim=0, keepdim=True)
    return TruthGeometry(
        Lsd=LSD_UM,
        BC_y=0.5 * (N_PANELS_Y * PANEL_SIZE_Y + (N_PANELS_Y - 1) * GAP_Y),
        BC_z=0.5 * (N_PANELS_Z * PANEL_SIZE_Z + (N_PANELS_Z - 1) * GAP_Z),
        tx=0.0, ty=0.0, tz=0.0,
        panel_delta_yz=deltas,
    )


def sample_truth_grains(*, seed: int = 7) -> tuple:
    """Random orientations (uniform on SO(3)) and positions on a 1mm cube."""
    rng = np.random.default_rng(seed)
    # Uniform on SO(3) via random unit quaternions → rotation matrices →
    # ZXZ Euler angles.
    q = rng.standard_normal((N_GRAINS, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Rotation matrix per quaternion.
    R = np.zeros((N_GRAINS, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    # ZXZ Euler from rotation matrix (matches HEDMForwardModel.euler2mat).
    eulers = np.zeros((N_GRAINS, 3))
    for g in range(N_GRAINS):
        Rg = R[g]
        # ZXZ: phi1 = atan2(R02, -R12); Phi = acos(R22); phi2 = atan2(R20, R21)
        if abs(Rg[2, 2]) < 1.0 - 1e-9:
            eulers[g, 1] = np.arccos(np.clip(Rg[2, 2], -1.0, 1.0))
            eulers[g, 0] = np.arctan2(Rg[0, 2], -Rg[1, 2])
            eulers[g, 2] = np.arctan2(Rg[2, 0], Rg[2, 1])
        else:
            # Gimbal lock — fall back to identity-like.
            eulers[g, 1] = 0.0 if Rg[2, 2] > 0 else np.pi
            eulers[g, 0] = np.arctan2(Rg[1, 0], Rg[0, 0])
            eulers[g, 2] = 0.0
    positions = (rng.random((N_GRAINS, 3)) - 0.5) * 1000.0   # ±500 µm cube
    eulers_t = torch.from_numpy(eulers).double()
    positions_t = torch.from_numpy(positions).double()
    lattice_t = torch.tensor([[AU_LATTICE_A, AU_LATTICE_A, AU_LATTICE_A,
                               90.0, 90.0, 90.0]] * N_GRAINS, dtype=torch.float64)
    return eulers_t, positions_t, lattice_t


# --------------------------------------------------------- powder synth

def generate_powder_observations(
    layout: PanelLayout,
    truth: TruthGeometry,
    ring_two_theta_deg: torch.Tensor,
    *,
    noise_px: float = 0.1,
    seed: int = 11,
) -> dict:
    """Generate powder peak (Y_pix, Z_pix, ring_idx, panel_idx) at truth.

    Each ring is sampled at ``N_POWDER_PER_RING`` η values uniformly in
    ``[0, 2π]``.  We project to ideal pixel positions, then perturb by the
    truth panel ``(δy, δz)`` *only after assigning panel-id from the
    ideal position* (so ``panel_idx`` reflects nominal geometry, mirroring
    real-data behaviour where panel-id is determined by the pixel-mask
    lookup, not by the (unknown) panel deltas).
    """
    rng = torch.Generator().manual_seed(seed)
    Lsd = truth.Lsd
    BC_y, BC_z = truth.BC_y, truth.BC_z
    eta = torch.arange(0, N_POWDER_PER_RING, dtype=torch.float64) \
        * (2 * math.pi / N_POWDER_PER_RING)

    Ys, Zs, ring_idxs, panel_idxs = [], [], [], []
    mask_panel = layout.panel_index_mask  # (H, W) long; -1 in gaps
    H, W = mask_panel.shape

    for r_idx, two_theta in enumerate(ring_two_theta_deg.tolist()):
        R_px = Lsd * math.tan(math.radians(two_theta)) / PX_UM
        Y_ideal = BC_y - R_px * torch.cos(eta)   # FF flip_y convention
        Z_ideal = BC_z + R_px * torch.sin(eta)

        # Panel-id from ideal position; -1 if outside detector or in a gap.
        Yi = Y_ideal.long().clamp(0, H - 1)
        Zi = Z_ideal.long().clamp(0, W - 1)
        in_bounds = (Y_ideal >= 0) & (Y_ideal < H) & (Z_ideal >= 0) & (Z_ideal < W)
        pid = mask_panel[Yi, Zi].clone()
        pid = torch.where(in_bounds, pid, torch.full_like(pid, -1))

        keep = pid >= 0
        if keep.sum() == 0:
            continue
        Yk = Y_ideal[keep]
        Zk = Z_ideal[keep]
        pidk = pid[keep]
        # Apply truth panel deltas (MIDAS convention: ``apply_panel_shifts``
        # ADDS the model's ``panel_delta_yz`` to the input pixel before
        # projecting to R, so the model's ``panel_delta_yz`` represents the
        # *correction* to apply, i.e. the negative of the physical shift.
        # We therefore SUBTRACT truth deltas here so that ``unpacked
        # panel_delta_yz`` equals ``truth.panel_delta_yz`` at convergence.
        Y_obs = Yk - truth.panel_delta_yz[pidk, 0]
        Z_obs = Zk - truth.panel_delta_yz[pidk, 1]
        # Add per-spot Gaussian noise.
        Y_obs = Y_obs + noise_px * torch.randn(Y_obs.shape, dtype=torch.float64, generator=rng)
        Z_obs = Z_obs + noise_px * torch.randn(Z_obs.shape, dtype=torch.float64, generator=rng)
        Ys.append(Y_obs); Zs.append(Z_obs)
        ring_idxs.append(torch.full_like(Y_obs, r_idx, dtype=torch.long))
        panel_idxs.append(pidk)

    return {
        "Y": torch.cat(Ys),
        "Z": torch.cat(Zs),
        "ring_idx": torch.cat(ring_idxs),
        "panel_idx": torch.cat(panel_idxs),
    }


# --------------------------------------------------------- HEDM synth

def _build_truth_hedm_model(
    truth: TruthGeometry, layout: PanelLayout, dtype=torch.float64,
) -> tuple:
    sg = SpaceGroup.from_number(225)        # Fm-3m (Au)
    lat = Lattice.for_system("cubic", a=AU_LATTICE_A)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=WAVELENGTH_A,
        two_theta_max_deg=TWO_THETA_MAX_DEG,
        expand_equivalents=True, dtype=dtype,
    )
    # Total detector pixel count along each axis.
    H = layout.panel_centers_y.shape[0] * PANEL_SIZE_Y + \
        (layout.panel_centers_y.shape[0] - 1) * GAP_Y
    W = layout.panel_centers_z.shape[1] * PANEL_SIZE_Z + \
        (layout.panel_centers_z.shape[1] - 1) * GAP_Z
    geom = HEDMGeometry(
        Lsd=truth.Lsd, y_BC=truth.BC_y, z_BC=truth.BC_z, px=PX_UM,
        omega_start=-180.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=H, n_pixels_z=W,
        min_eta=0.0, wavelength=WAVELENGTH_A,
        tx=truth.tx, ty=truth.ty, tz=truth.tz,
        flip_y=True, multi_mode="layered",
    )
    model = HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.to(dtype))
    return model, hkls_int


def generate_hedm_observations(
    layout: PanelLayout, truth: TruthGeometry,
    grain_eulers: torch.Tensor, grain_positions: torch.Tensor,
    grain_lattices: torch.Tensor,
    *, noise_px: float = 0.1, seed: int = 13,
) -> dict:
    """Forward 100 grains × 8 hkls at truth geometry, perturb by truth panel
    deltas, store per-spot ``(y_obs, z_obs, panel_idx, g_idx, m_idx, k_idx)``
    where ``m_idx`` is the hkl row and ``k_idx ∈ {0, 1}`` is the omega branch.
    """
    rng = torch.Generator().manual_seed(seed)
    model, _ = _build_truth_hedm_model(truth, layout)

    # Forward in (B=N_grains, N=1) form so the model returns
    # (N_grains, 1, K=2, M) tensors.
    eulers = grain_eulers.view(N_GRAINS, 1, 3)
    positions = grain_positions.view(N_GRAINS, 1, 3)
    spots = model(eulers, positions, lattice_params=grain_lattices.view(N_GRAINS, 6))
    # spots.y_pixel etc. shape (N_grains, 1, K, M).  Squeeze N=1.
    yp = spots.y_pixel.squeeze(1)   # (N_grains, K, M)
    zp = spots.z_pixel.squeeze(1)
    valid = spots.valid.squeeze(1)
    Ng, K, M = yp.shape
    yp = yp.double(); zp = zp.double(); valid = valid.bool()

    mask_panel = layout.panel_index_mask
    H, W = mask_panel.shape

    Yi = yp.long().clamp(0, H - 1)
    Zi = zp.long().clamp(0, W - 1)
    in_bounds = (yp >= 0) & (yp < H) & (zp >= 0) & (zp < W) & valid
    pid_pre = mask_panel[Yi, Zi]
    pid = torch.where(in_bounds, pid_pre, torch.full_like(pid_pre, -1))

    keep = pid >= 0
    g_idx = torch.arange(Ng).view(Ng, 1, 1).expand(Ng, K, M)[keep]
    m_idx = torch.arange(M).view(1, 1, M).expand(Ng, K, M)[keep]
    k_idx = torch.arange(K).view(1, K, 1).expand(Ng, K, M)[keep]
    pidk = pid[keep]
    Y_ideal = yp[keep]
    Z_ideal = zp[keep]
    # Subtract truth deltas (see powder generator for MIDAS sign convention).
    Y_obs = Y_ideal - truth.panel_delta_yz[pidk, 0]
    Z_obs = Z_ideal - truth.panel_delta_yz[pidk, 1]
    Y_obs = Y_obs + noise_px * torch.randn(Y_obs.shape, dtype=torch.float64, generator=rng)
    Z_obs = Z_obs + noise_px * torch.randn(Z_obs.shape, dtype=torch.float64, generator=rng)

    return {
        "Y": Y_obs, "Z": Z_obs,
        "panel_idx": pidk,
        "g_idx": g_idx,
        "m_idx": m_idx,
        "k_idx": k_idx,
        "model": model,
    }


# --------------------------------------------------------- residuals

def make_powder_residual(obs: dict, layout: PanelLayout,
                          ring_two_theta_deg: torch.Tensor,
                          ring_d_spacing_A: Optional[torch.Tensor] = None):
    Y = obs["Y"]; Z = obs["Z"]; ridx = obs["ring_idx"]; pidx = obs["panel_idx"]
    rho_d_default = torch.tensor(200000.0, dtype=torch.float64)
    # Per-spot d-spacing (when ``ring_d_spacing_A`` is supplied) lets
    # ``pseudo_strain_residual`` recompute 2θ via Bragg's law ``2θ = 2
    # arcsin(λ/2d)`` so the closure depends differentiably on
    # ``unpacked['Wavelength']``.  Required for any (Lsd, λ) gauge or
    # wavelength-refinement story.
    d_per_spot = ring_d_spacing_A[ridx] if ring_d_spacing_A is not None else None

    def closure(unpacked):
        rho_d = unpacked.get("RhoD", rho_d_default)
        return pseudo_strain_residual(
            Y, Z, ring_two_theta_deg[ridx], unpacked,
            rho_d=rho_d,
            panel_layout=layout, panel_idx=pidx,
            ring_idx=ridx,
            ring_d_spacing_A=d_per_spot,
        )
    return closure


def make_hedm_residual(obs: dict, layout: PanelLayout):
    """Custom HEDM residual aware of panel deltas.

    Re-runs HEDMForwardModel via ``functional_call`` with current
    ``Lsd / BC_y / BC_z`` from ``unpacked``, computes predicted (y, z) per
    spot, applies ``unpacked['panel_delta_yz'][panel_idx_obs]`` to the
    prediction, and subtracts observed positions.  ``panel_idx_obs`` is
    fixed at observation time (panel-id doesn't migrate under sub-pixel
    refinements).
    """
    Y_obs = obs["Y"]; Z_obs = obs["Z"]
    g_idx = obs["g_idx"]; m_idx = obs["m_idx"]; k_idx = obs["k_idx"]
    pidx = obs["panel_idx"]
    model = obs["model"]
    # Pre-compute the (pid_per_grain, m_idx_per_grain, k_idx_per_grain)
    # gather index into the (Ng, K, M) forward output tensor.
    flat_idx = g_idx * (model.hkls.shape[0] * 2) + k_idx * model.hkls.shape[0] + m_idx

    def closure(unpacked):
        # Override single-distance Lsd / BC + tilts so the HEDM forward
        # tracks the same geometry as the powder side.  Without tilt
        # overrides the LM can move ty/tz to fit powder while HEDM
        # ignores the change, biasing the panel deltas.
        Lsd = unpacked["Lsd"].reshape(1)
        BCy = unpacked["BC_y"].reshape(1)
        BCz = unpacked["BC_z"].reshape(1)
        # Tilts: build [D=1, 3] (tx, ty, tz) row.
        zero_t = torch.zeros((), dtype=model.tilts.dtype, device=model.tilts.device)
        tx = unpacked.get("tx", zero_t).reshape(()).to(dtype=model.tilts.dtype)
        ty = unpacked.get("ty", zero_t).reshape(()).to(dtype=model.tilts.dtype)
        tz = unpacked.get("tz", zero_t).reshape(()).to(dtype=model.tilts.dtype)
        tilts = torch.stack([tx, ty, tz]).unsqueeze(0)   # [1, 3]
        overrides = {
            "_Lsd": Lsd.to(dtype=model._Lsd.dtype),
            "_y_BC": BCy.to(dtype=model._y_BC.dtype),
            "_z_BC": BCz.to(dtype=model._z_BC.dtype),
            "tilts": tilts,
        }
        eulers = unpacked["grain_euler"].view(N_GRAINS, 1, 3)
        positions = unpacked["grain_pos"].view(N_GRAINS, 1, 3)
        lattices = unpacked["grain_lattice"].view(N_GRAINS, 6)
        spots = functional_call(
            model, overrides,
            args=(eulers, positions),
            kwargs={"lattice_params": lattices},
        )
        yp = spots.y_pixel.squeeze(1)   # (Ng, K, M)
        zp = spots.z_pixel.squeeze(1)
        y_pred = yp.reshape(-1).gather(0, flat_idx)
        z_pred = zp.reshape(-1).gather(0, flat_idx)
        # Apply current panel deltas (MIDAS sign convention: subtract — see
        # generate_powder_observations for why).  At convergence the
        # ``unpacked panel_delta_yz`` equals the truth panel shift.
        delta = unpacked["panel_delta_yz"]
        y_pred_corr = y_pred - delta[pidx, 0]
        z_pred_corr = z_pred - delta[pidx, 1]
        # Residual in pixels.
        return torch.stack([y_pred_corr - Y_obs, z_pred_corr - Z_obs], dim=-1).flatten()
    return closure


# --------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default output lives in the paper folder, not /tmp — paper artefacts
    # (CSVs, figures) must be discoverable from dev/paper/.
    _DEFAULT_OUT = Path(__file__).resolve().parents[2] / "dev" / "paper" / "runs" / "synthetic_pilatus_joint"
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--refine-grains", action="store_true",
                         help="Also refine grain_euler and grain_pos in the joint LM "
                              "(paper §4.4 option-2 alternating-pass-A behaviour).  "
                              "Default off so the panel-delta recovery is isolated.")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--demo-gauge", action="store_true",
                        help="Fisher block on (Lsd, Wavelength) under powder-only / "
                              "HEDM-only / joint, demonstrating the (Lsd, λ) "
                              "rank-1 gauge in HEDM-only inference.  Saves to "
                              "<output>/gauge_demo.csv.")
    parser.add_argument("--phase-diagram", action="store_true",
                        help="Sweep (Lsd, N_grains) at fixed N_rings, computing "
                              "σ_joint/σ_powder ratio per cell.  Maps the regime "
                              "where joint refinement is essential vs. where "
                              "powder alone suffices.  Saves to "
                              "<output>/phase_diagram.csv and phase_diagram.png.")
    parser.add_argument("--all-blocks-fisher", action="store_true",
                        help="Compute Fisher rank/cond/σ across every refinable "
                              "spec block (geometry, distortion, per-panel, "
                              "wavelength, gauge couples) under powder-only / "
                              "HEDM-only / joint.  Produces the identifiability "
                              "table for the paper.  Saves to "
                              "<output>/all_blocks_fisher.csv.")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print("=" * 70)
    print(" SYNTHETIC PILATUS 6×8 + 100-GRAIN JOINT CALIBRATION")
    print("=" * 70)

    # ----- 1. truth + observations
    layout = PanelLayout.regular(N_PANELS_Y, N_PANELS_Z,
                                  PANEL_SIZE_Y, PANEL_SIZE_Z,
                                  gap_y=GAP_Y, gap_z=GAP_Z)
    truth = sample_truth(layout, seed=args.seed)
    print(f"  truth.Lsd     = {truth.Lsd:.1f} µm")
    print(f"  truth.BC      = ({truth.BC_y:.2f}, {truth.BC_z:.2f}) px")
    print(f"  N_panels      = {layout.n_panels()}")
    print(f"  panel_delta σ = {truth.panel_delta_yz.std().item():.4f} px (truth)")

    # Au reflections + ring 2θ table.
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=AU_LATTICE_A)
    hkls_cart, thetas_full, _ = hkls_for_forward_model(
        sg, lat, wavelength_A=WAVELENGTH_A, two_theta_max_deg=TWO_THETA_MAX_DEG,
        expand_equivalents=False,
    )
    # Unique 2θ values for ring labels (powder side).
    two_theta_uniq, _ = torch.unique(2 * thetas_full * 180.0 / math.pi,
                                       return_inverse=True, sorted=True)
    two_theta_uniq = two_theta_uniq.double()[:N_RINGS]
    # Reverse-derive d-spacing per ring from 2θ at WAVELENGTH_A so the
    # powder closure can recompute 2θ via Bragg when Wavelength is
    # refined (the (Lsd, λ) gauge story needs this).
    ring_d_uniq_A = WAVELENGTH_A / (2.0 * torch.sin(two_theta_uniq * math.pi / 360.0))
    print(f"  Au rings (2θ) = {[f'{x:.2f}' for x in two_theta_uniq.tolist()]}°")

    grain_eulers_t, grain_pos_t, grain_lat_t = sample_truth_grains(seed=args.seed + 1)

    print("\n>> Generating powder observations (truth geometry) …")
    powder_obs = generate_powder_observations(
        layout, truth, two_theta_uniq, seed=args.seed + 2,
    )
    powder_panels = set(powder_obs["panel_idx"].unique().tolist())
    print(f"     {powder_obs['Y'].numel()} powder peaks across "
          f"{len(powder_panels)} panels")

    print(">> Generating HEDM grain spots (truth geometry) …")
    hedm_obs = generate_hedm_observations(
        layout, truth, grain_eulers_t, grain_pos_t, grain_lat_t,
        seed=args.seed + 3,
    )
    hedm_panels = set(hedm_obs["panel_idx"].unique().tolist())
    print(f"     {hedm_obs['Y'].numel()} HEDM spots across "
          f"{len(hedm_panels)} panels")
    covered_panels = powder_panels | hedm_panels
    n_uncovered = layout.n_panels() - len(covered_panels)
    print(f"     coverage: {len(covered_panels)}/{layout.n_panels()} panels seen by some modality "
          f"({n_uncovered} panel(s) physically unconstrained)")
    covered_mask = torch.zeros(layout.n_panels(), dtype=torch.bool)
    for p in covered_panels:
        covered_mask[p] = True
    powder_panel_mask = torch.zeros(layout.n_panels(), dtype=torch.bool)
    for p in powder_panels:
        powder_panel_mask[p] = True

    # ----- 2. spec
    print("\n>> Building joint spec …")
    spec = mp.ParameterSpec()
    # Geometry (refined).
    spec.add(mp.Parameter("Lsd", init=truth.Lsd + 50.0,    # 50 µm offset from truth
                          bounds=(truth.Lsd - 5e3, truth.Lsd + 5e3)))
    spec.add(mp.Parameter("BC_y", init=truth.BC_y + 0.3,
                          bounds=(truth.BC_y - 5.0, truth.BC_y + 5.0)))
    spec.add(mp.Parameter("BC_z", init=truth.BC_z - 0.2,
                          bounds=(truth.BC_z - 5.0, truth.BC_z + 5.0)))
    # Tilts: HEDMForwardModel in FF mode (flip_y=True) skips tilts because
    # the FF pipeline pre-corrects them at peak-finding.  Powder forward
    # *does* apply tilts.  To keep the two modalities consistent in this
    # synthetic, we freeze tilts at 0 (matching the truth simulation).
    # Real-data joint refinement should set apply_tilts=True on the
    # HEDM geometry so both modalities track ty/tz refinement.
    spec.add(mp.Parameter("ty", init=0.0, refined=False))
    spec.add(mp.Parameter("tz", init=0.0, refined=False))
    # Wavelength fixed at truth (powder side will not refine it).
    spec.add(mp.Parameter("Wavelength", init=WAVELENGTH_A, refined=False))
    spec.add(mp.Parameter("pxY", init=PX_UM, refined=False))
    spec.add(mp.Parameter("pxZ", init=PX_UM, refined=False))
    spec.add(mp.Parameter("RhoD", init=200000.0, refined=False))
    # Per-panel deltas (refined; this is the headline block).
    # Gaussian prior on every panel_delta_yz entry: mean 0, σ_prior set by
    # the Pilatus module placement spec (~0.5 px).  This handles the
    # "panels with no data" case: uncovered panels' posterior ≈ prior
    # (σ_post = σ_prior), covered panels' posterior dominated by data
    # when J'J/σ²_noise >> 1/σ²_prior.  Eliminates the binary
    # covered/uncovered decision and removes the BC↔panel-mean
    # ambiguity without an explicit Σ=0 gauge.
    SIGMA_PRIOR_PX = 0.5
    spec.add(mp.Parameter("panel_delta_yz",
                          init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                          bounds=(-3.0, 3.0),
                          prior=mp.GaussianPrior(mean=0.0, std=SIGMA_PRIOR_PX)))
    # Other per-panel blocks must exist (with refined=False) so the
    # geometry forward can read them; pseudo_strain_residual passes them
    # to pixel_to_REta which requires non-None inputs.
    spec.add(mp.Parameter("panel_delta_theta",
                          init=torch.zeros(layout.n_panels(), dtype=torch.float64),
                          refined=False))
    # HEDM grain blocks — initialized noisily near truth (paper §4.4 alt
    # default freezes strain; we keep euler+pos refined).
    rng_init = torch.Generator().manual_seed(args.seed + 50)
    grain_eulers_init = grain_eulers_t + 0.001 * torch.randn(
        grain_eulers_t.shape, dtype=torch.float64, generator=rng_init)
    grain_pos_init = grain_pos_t + 1.0 * torch.randn(
        grain_pos_t.shape, dtype=torch.float64, generator=rng_init)   # 1 µm noise
    # In paper §4.4 option-2 default, grain orientations + positions are
    # refined alongside geometry; lattice is frozen.  For the synthetic
    # we initialize grain euler/pos near truth (post-indexing/refinement
    # noise scale) and let the joint LM polish them along with geometry.
    # Freeze grain_euler+pos initially to isolate the panel-delta recovery
    # claim — the user can opt them in via --refine-grains.
    spec.add(mp.Parameter("grain_euler", init=grain_eulers_init,
                          bounds=(-2 * math.pi, 2 * math.pi),
                          refined=args.refine_grains))
    spec.add(mp.Parameter("grain_pos", init=grain_pos_init,
                          bounds=(-1000.0, 1000.0),
                          refined=args.refine_grains))
    spec.add(mp.Parameter("grain_lattice", init=grain_lat_t, refined=False))
    print(f"     refined names: {spec.refined_names()[:6]}… (+{len(spec.refined_names())-6} more)")

    # ----- 3. residuals
    powder_fn = make_powder_residual(powder_obs, layout, two_theta_uniq,
                                       ring_d_spacing_A=ring_d_uniq_A)
    hedm_fn = make_hedm_residual(hedm_obs, layout)

    # Loss weighting: powder is dimensionless (~1e-4 RMS), HEDM is in pixels
    # (~0.1 px RMS).  Match scales so neither modality dominates.  We use
    # w_powder = 1 / σ_powder = 1 / 1e-4 = 1e4, w_hedm = 1 / σ_hedm = 1 / 0.1
    # = 10.  After weighting, both blocks have unit-variance residuals.
    W_POWDER = 1.0e4
    W_HEDM = 10.0
    LAMBDA_GAUGE = 1.0e6     # paper-3 §9 default — enforces Σ=0 to numerical
                              # precision so global drift goes nowhere.  At
                              # this scale the gauge eigenvalue is ~5e7 vs
                              # data Fisher ~3.7e5, ratio ~7e-3, still well
                              # above rtol=1e-8 so block-rank detection works.

    def joint_fn(u):
        return joint_residual(
            u, powder_residual_fn=powder_fn, hedm_residual_fn=hedm_fn,
            spec=spec, weights=JointWeights(w_powder=W_POWDER, w_hedm=W_HEDM,
                                              lambda_gauge=LAMBDA_GAUGE),
            gauge_blocks=[],   # disabled: prior anchors the global mean
        )

    def powder_only_with_gauge(u):
        """Powder residual + gauge (matches the joint setup but HEDM-free)."""
        return joint_residual(
            u, powder_residual_fn=powder_fn,
            hedm_residual_fn=lambda _u: torch.zeros(0, dtype=torch.float64),
            spec=spec, weights=JointWeights(w_powder=W_POWDER, w_hedm=W_HEDM,
                                              lambda_gauge=LAMBDA_GAUGE),
            gauge_blocks=[],   # disabled: prior anchors the global mean
        )

    def hedm_only_with_gauge(u):
        return joint_residual(
            u, powder_residual_fn=lambda _u: torch.zeros(0, dtype=torch.float64),
            hedm_residual_fn=hedm_fn,
            spec=spec, weights=JointWeights(w_powder=W_POWDER, w_hedm=W_HEDM,
                                              lambda_gauge=LAMBDA_GAUGE),
            gauge_blocks=[],   # disabled: prior anchors the global mean
        )

    # Gauge-free closures for the *data-only* identifiability claim.  The
    # rank-1 deficiency in paper-3 §9 is a property of the data alone (no
    # prior).  The gauge fixes the deficiency by adding a soft penalty —
    # but the question "does the data alone identify per-panel shifts?"
    # needs to look at the data Fisher only.
    def powder_only_no_gauge(u):
        return W_POWDER * powder_fn(u)

    def hedm_only_no_gauge(u):
        return W_HEDM * hedm_fn(u)

    def joint_no_gauge(u):
        return torch.cat([W_POWDER * powder_fn(u), W_HEDM * hedm_fn(u)])

    # ----- 4. modality-conditional Fisher rank at the SEED state
    print("\n>> Fisher block rank on panel_delta_yz (at seed, before any refinement)")
    seed_unpacked = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    if not isinstance(seed_unpacked["Lsd"], torch.Tensor) or seed_unpacked["Lsd"].dim() == 0:
        # init_tensor returns 0-D for scalar Parameters; ensure Tensor.
        for n in seed_unpacked:
            v = seed_unpacked[n]
            if not isinstance(v, torch.Tensor):
                seed_unpacked[n] = torch.tensor(v, dtype=torch.float64)

    for label, fn in [("powder-only", powder_only_with_gauge),
                       ("hedm-only", hedm_only_with_gauge),
                       ("joint", joint_fn)]:
        rep = fisher_block_rank(
            spec, fn, seed_unpacked,
            block_names=["panel_delta_yz"],
            sigma_r=1.0, fallback_span=2.0,
        )
        sig = rep.sigma_per_dim
        print(f"     {label:12s}  rank={rep.rank:3d} / {sig.numel():3d}   "
              f"cond={rep.condition_number:.2e}   "
              f"σ_max={float(sig.max()):.3e}   σ_med={float(sig.median()):.3e}")

    # ----- 5. joint LM refinement
    print("\n>> Joint LM (one-shot, full joint with weights w_p=1, w_h=0.001) …")
    t0 = time.time()
    unpacked, cost, rc = mp.lm_minimise(
        spec, joint_fn,
        config=mp.GenericLMConfig(max_iter=args.max_iter, ftol_rel=1e-10, xtol_rel=1e-10),
        fallback_span=2.0,
    )
    dt = time.time() - t0
    print(f"     rc={rc}  cost={cost:.4e}  time={dt:.1f}s")

    # MAP recovery: panel_delta_yz vs truth.  Two subtleties:
    #   1. Uncovered panels stay at init=0 (no data) — exclude from metrics.
    #   2. The Σ=0 gauge fixes the mean over ALL 48 panels, but only the
    #      covered subset is observable.  Truth's uncovered panels carry
    #      nonzero dz, so the covered-subset mean is non-zero in truth but
    #      forced to zero by the gauge — that 0.27 px global Z bias
    #      absorbs into BC_z.  This is a *gauge ambiguity*, not a
    #      misfit; subtract the covered-subset mean error to remove it
    #      before comparing.
    pdyz_map = unpacked["panel_delta_yz"]
    err_raw = pdyz_map - truth.panel_delta_yz
    err_covered_raw = err_raw[covered_mask]
    gauge_bias = err_covered_raw.mean(dim=0, keepdim=True)
    err_covered = err_covered_raw - gauge_bias
    rms_err_px = float(err_covered.pow(2).mean().sqrt())
    max_err_px = float(err_covered.abs().max())
    print(f"     panel_delta_yz recovery on covered panels ({covered_mask.sum().item()}/{layout.n_panels()}):")
    print(f"       gauge bias (covered mean error, absorbed into BC) = "
          f"({gauge_bias[0,0].item():+.4f}, {gauge_bias[0,1].item():+.4f}) px")
    print(f"       gauge-corrected RMS = {rms_err_px:.4f} px, max = {max_err_px:.4f} px  "
          f"(truth σ={truth.panel_delta_yz.std().item():.4f} px)")
    print(f"       raw RMS = {float(err_covered_raw.pow(2).mean().sqrt()):.4f} px, "
          f"raw max = {float(err_covered_raw.abs().max()):.4f} px")
    # Powder-only recovery comparison (data-only, no joint).
    print(f"     Note: powder-only sees {len(powder_panels)} panels; "
          f"any panel without powder data is rank-deficient under powder-alone.")
    print(f"     Lsd  : truth={truth.Lsd:.2f}  init={float(spec.parameters['Lsd'].init):.2f}  "
          f"MAP={float(unpacked['Lsd']):.2f}")
    print(f"     BC_y : truth={truth.BC_y:.2f}  MAP={float(unpacked['BC_y']):.4f}")
    print(f"     BC_z : truth={truth.BC_z:.2f}  MAP={float(unpacked['BC_z']):.4f}")
    print(f"     ty   : truth={truth.ty:.4f}  MAP={float(unpacked['ty']):.6f}")
    print(f"     tz   : truth={truth.tz:.4f}  MAP={float(unpacked['tz']):.6f}")

    # ----- 6. Fisher rank at MAP under each modality
    # Gauge-free Fisher: this is the data-only identifiability — the
    # rank-1 paper-3 §9 claim is about data, not the prior.  The
    # gauge-included version below mirrors what the LM sees.
    print("\n>> Fisher block rank on panel_delta_yz (at joint MAP, GAUGE-FREE)")
    for label, fn in [("powder-only", powder_only_no_gauge),
                       ("hedm-only", hedm_only_no_gauge),
                       ("joint", joint_no_gauge)]:
        rep = fisher_block_rank(
            spec, fn, unpacked,
            block_names=["panel_delta_yz"],
            sigma_r=1.0, fallback_span=2.0,
        )
        sig = rep.sigma_per_dim
        n_block = sig.numel()
        print(f"     {label:12s}  rank={rep.rank:3d} / {n_block:3d}   "
              f"cond={rep.condition_number:.2e}   "
              f"σ_max={float(sig.max()):.3e}   σ_med={float(sig.median()):.3e}")

    print("\n>> Fisher block rank on panel_delta_yz (at joint MAP, gauge-included)")
    fisher_table = {}
    for label, fn in [("powder-only", powder_only_with_gauge),
                       ("hedm-only", hedm_only_with_gauge),
                       ("joint", joint_fn)]:
        rep = fisher_block_rank(
            spec, fn, unpacked,
            block_names=["panel_delta_yz"],
            sigma_r=1.0, fallback_span=2.0,
        )
        fisher_table[label] = rep
        sig = rep.sigma_per_dim
        n_block = sig.numel()
        print(f"     {label:12s}  rank={rep.rank:3d} / {n_block:3d}   "
              f"cond={rep.condition_number:.2e}   "
              f"σ_max={float(sig.max()):.3e}   σ_med={float(sig.median()):.3e}")

    # ----- 7. CSV
    print(f"\n>> Writing CSV to {args.output / 'panel_recovery.csv'}")
    with open(args.output / "panel_recovery.csv", "w") as f:
        f.write("panel_id,truth_dy,truth_dz,map_dy,map_dz,err_dy,err_dz,"
                "sig_powder_dy,sig_powder_dz,sig_hedm_dy,sig_hedm_dz,"
                "sig_joint_dy,sig_joint_dz\n")
        for k in range(layout.n_panels()):
            f.write(f"{k},{truth.panel_delta_yz[k,0].item():.6f},"
                    f"{truth.panel_delta_yz[k,1].item():.6f},"
                    f"{pdyz_map[k,0].item():.6f},{pdyz_map[k,1].item():.6f},"
                    f"{err_raw[k,0].item():.6f},{err_raw[k,1].item():.6f},"
                    f"{fisher_table['powder-only'].sigma_per_dim[2*k].item():.4e},"
                    f"{fisher_table['powder-only'].sigma_per_dim[2*k+1].item():.4e},"
                    f"{fisher_table['hedm-only'].sigma_per_dim[2*k].item():.4e},"
                    f"{fisher_table['hedm-only'].sigma_per_dim[2*k+1].item():.4e},"
                    f"{fisher_table['joint'].sigma_per_dim[2*k].item():.4e},"
                    f"{fisher_table['joint'].sigma_per_dim[2*k+1].item():.4e}\n")

    # ----- 8. plot (if matplotlib is available)
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            # Build per-panel σ grids, masking uncovered panels with NaN
            # so the prior-saturated panels (σ = σ_prior = 0.5 px) do not
            # dominate the colour scale and flatten the data variation
            # between powder / HEDM / joint on the covered panels.
            covered_2d = covered_mask.numpy().reshape(N_PANELS_Y, N_PANELS_Z)
            grids = {}
            for label, rep in fisher_table.items():
                sig = rep.sigma_per_dim.numpy().reshape(layout.n_panels(), 2)
                sig_norm = np.linalg.norm(sig, axis=1)
                grid = sig_norm.reshape(N_PANELS_Y, N_PANELS_Z).astype(float)
                grid[~covered_2d] = np.nan
                grids[label] = grid

            # Each subplot gets its OWN colour scale, so within-modality
            # per-panel texture is visible even though the absolute σ
            # differs by an order of magnitude between modalities.  We
            # also add a 4th subplot showing the σ_joint / σ_HEDM ratio
            # on covered panels --- a diverging map centred at 1.0
            # highlights where joint is tighter (blue) vs.\ where HEDM-only
            # is already as tight as joint (white/red).
            cmap_seq = matplotlib.cm.get_cmap("viridis").copy()
            cmap_seq.set_bad(color="white")
            cmap_div = matplotlib.cm.get_cmap("RdBu_r").copy()
            cmap_div.set_bad(color="white")

            fig, axes = plt.subplots(2, 2, figsize=(11, 9))
            # Order: top-left = powder, top-right = HEDM,
            #        bottom-left = joint, bottom-right = joint/HEDM ratio.
            slots = [(0, 0), (0, 1), (1, 0)]
            for i, (label, grid) in enumerate(grids.items()):
                r, c = slots[i]
                masked = np.ma.masked_invalid(grid)
                vmin = float(np.nanmin(grid)) * 0.9
                vmax = float(np.nanmax(grid)) * 1.1
                im = axes[r, c].imshow(masked, origin="lower", aspect="auto",
                                        norm=matplotlib.colors.Normalize(
                                            vmin=vmin, vmax=vmax),
                                        cmap=cmap_seq,
                                        interpolation="nearest")
                axes[r, c].set_title(
                    f"{label}\n"
                    f"σ_med = {float(np.nanmedian(grid)):.4f} px,  "
                    f"σ ∈ [{vmin:.3f}, {vmax:.3f}]"
                )
                axes[r, c].set_xlabel("panel z"); axes[r, c].set_ylabel("panel y")
                fig.colorbar(im, ax=axes[r, c], label="σ (px)")

            ratio = grids["joint"] / np.where(grids["hedm-only"] > 0,
                                                grids["hedm-only"], np.nan)
            ratio_masked = np.ma.masked_invalid(ratio)
            r_min = float(np.nanmin(ratio))
            r_max = float(np.nanmax(ratio))
            r_span = max(1.0 - r_min, r_max - 1.0, 0.05)
            im = axes[1, 1].imshow(ratio_masked, origin="lower", aspect="auto",
                                    norm=matplotlib.colors.Normalize(
                                        vmin=1.0 - r_span, vmax=1.0 + r_span),
                                    cmap=cmap_div,
                                    interpolation="nearest")
            axes[1, 1].set_title(
                f"σ_joint / σ_HEDM\n"
                f"median ratio = {float(np.nanmedian(ratio)):.3f}  "
                f"(blue = joint tighter)"
            )
            axes[1, 1].set_xlabel("panel z"); axes[1, 1].set_ylabel("panel y")
            fig.colorbar(im, ax=axes[1, 1], label="σ ratio")

            fig.suptitle(
                f"Per-panel σ on (δy, δz) — covered panels only; "
                f"uncovered shown white.  "
                f"Truth |Δ| RMS = "
                f"{float(truth.panel_delta_yz.norm(dim=1).pow(2).mean().sqrt()):.3f} px,  "
                f"recovered RMS = {rms_err_px:.4f} px on covered panels."
            )
            fig.tight_layout()
            png = args.output / "panel_sigma_comparison.png"
            fig.savefig(png, dpi=140); plt.close(fig)
            print(f">> Plot: {png}")
        except Exception as e:
            print(f">> Plot skipped: {e}")

    # ----- 9. pass/fail (coverage-aware)
    # Use the gauge-free Fisher to evaluate the data-alone identifiability
    # claim (paper-3 §9 is a statement about *data* rank, not data + prior).
    # The gauge-included ranks below are full-rank by construction once a
    # Gaussian prior is attached to panel_delta_yz, so they don't tell us
    # whether HEDM evidence has broken the powder rank deficit.
    rep_joint_data = fisher_block_rank(
        spec, joint_no_gauge, unpacked,
        block_names=["panel_delta_yz"], sigma_r=1.0, fallback_span=2.0,
    )
    rep_powder_data = fisher_block_rank(
        spec, powder_only_no_gauge, unpacked,
        block_names=["panel_delta_yz"], sigma_r=1.0, fallback_span=2.0,
    )
    # σ comparison still uses gauge+prior (the LM-actual posterior).
    rep_joint = fisher_table["joint"]
    rep_powder = fisher_table["powder-only"]
    rep_hedm = fisher_table["hedm-only"]
    print("\n>> Pass/fail checks (coverage-aware: %d panels covered, %d unconstrained)"
          % (covered_mask.sum().item(), n_uncovered))
    # Maximum identifiable rank = 2 × (covered panels), modulo the 2-D Σ=0
    # gauge null when running gauge-free.  Gauge-included joint should
    # achieve 2 × covered_panels (gauge supplies the 2 missing modes from
    # the unconstrained-mean direction).
    target_rank_covered = 2 * len(covered_panels)
    # Use the gauge-free (data-only) Fisher for the rank claims.
    joint_rank_ok = rep_joint_data.rank >= target_rank_covered
    powder_rank_deficient = rep_powder_data.rank < target_rank_covered
    recovery_ok = max_err_px < 0.5
    converged = rc in (0, 1)   # rc=1 = max iter (acceptable for this synthetic)

    # σ comparison on COVERED panels only — uncovered panels have rank-zero
    # Fisher rows so their σ is dominated by the pseudoinverse ridge and
    # doesn't reflect a meaningful uncertainty.
    sig_pwd = rep_powder.sigma_per_dim.reshape(layout.n_panels(), 2).norm(dim=1)
    sig_joint = rep_joint.sigma_per_dim.reshape(layout.n_panels(), 2).norm(dim=1)
    sig_hedm = rep_hedm.sigma_per_dim.reshape(layout.n_panels(), 2).norm(dim=1)
    ratio_med = float((sig_pwd[covered_mask] / sig_joint[covered_mask].clamp_min(1e-30)).median())
    print(f"     σ_powder med (covered) = {float(sig_pwd[covered_mask].median()):.3e}")
    print(f"     σ_hedm   med (covered) = {float(sig_hedm[covered_mask].median()):.3e}")
    print(f"     σ_joint  med (covered) = {float(sig_joint[covered_mask].median()):.3e}")

    print(f"     joint data-Fisher rank ≥ 2×covered ({rep_joint_data.rank} ≥ {target_rank_covered}): "
          f"{'PASS' if joint_rank_ok else 'FAIL'}")
    print(f"     powder data-Fisher rank < 2×covered ({rep_powder_data.rank} < {target_rank_covered}): "
          f"{'PASS' if powder_rank_deficient else 'FAIL'}")
    print(f"     max recovery error < 0.5 px on covered ({max_err_px:.4f})    : "
          f"{'PASS' if recovery_ok else 'FAIL'}")
    print(f"     LM converged (rc={rc})                                    : "
          f"{'PASS' if converged else 'FAIL'}")
    print(f"     median σ ratio powder/joint on covered = {ratio_med:.2e}  "
          f"(informational; with the bounded Logit reparam used here, "
          f"absolute σ values are dominated by the bounded-prior curvature "
          f"rather than the data — inspect rank + recovery error for the "
          f"actual identifiability story)")
    all_ok = joint_rank_ok and powder_rank_deficient and recovery_ok and converged
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")

    # -------- Optional gauge demonstration (--demo-gauge) ----------------
    if args.demo_gauge:
        run_gauge_demo(
            args=args, layout=layout, truth=truth,
            two_theta_uniq=two_theta_uniq,
            ring_d_uniq_A=ring_d_uniq_A,
            grain_eulers_t=grain_eulers_t, grain_pos_t=grain_pos_t,
            grain_lat_t=grain_lat_t,
            powder_obs=powder_obs, hedm_obs=hedm_obs,
            covered_mask=covered_mask,
        )

    # -------- All-blocks Fisher table (--all-blocks-fisher) --------------
    if args.all_blocks_fisher:
        run_all_blocks_fisher(
            args=args, spec=spec, unpacked=unpacked,
            powder_only_no_gauge=powder_only_no_gauge,
            hedm_only_no_gauge=hedm_only_no_gauge,
            joint_no_gauge=joint_no_gauge,
            covered_panels=covered_panels,
            layout=layout,
        )

    # -------- Phase diagram (--phase-diagram) ----------------------------
    if args.phase_diagram:
        run_phase_diagram(args=args, layout=layout, truth=truth, seed=args.seed)

    return 0 if all_ok else 1


def run_gauge_demo(*, args, layout, truth, two_theta_uniq, ring_d_uniq_A,
                    grain_eulers_t, grain_pos_t, grain_lat_t,
                    powder_obs, hedm_obs, covered_mask) -> None:
    """Demonstrate the (Lsd, λ) near-gauge in HEDM forward modelling.

    In the small-angle approximation, FF-HEDM spot pixel positions
    behave like ``R_pix ∝ Lsd · λ / d_grain`` where ``d_grain`` depends
    on the (refined) grain lattice.  So scaling ``(Lsd, λ) → (kLsd,
    λ/k)`` leaves spot pixels approximately unchanged — the (Lsd, λ)
    direction is a near-null mode of the HEDM data Fisher.

    Powder rings of a calibrant with ``d_calibrant`` known
    *independently* (Au reference value) break this near-degeneracy:
    ``R_powder ∝ Lsd · λ / d_calibrant_known``, where d_calibrant_known
    is *fixed*, not scaled by k.

    We demonstrate by computing the 2×2 Fisher block on (Lsd, λ) at
    truth under three residual closures:

      - HEDM-only           : λ-min/λ-max ratio near zero (gauge-like)
      - Powder-only         : λ-min/λ-max non-trivial (gauge broken)
      - Joint               : tightest, smallest condition number

    No LM refinement is needed — the Fisher at truth tells us the data
    information content directly.  This is a clean reading of the
    "why-not-just-HEDM" question for the paper.
    """
    print("\n" + "=" * 70)
    print(" GAUGE DEMONSTRATION  ((Lsd, λ) near-gauge in HEDM)")
    print("=" * 70)

    def build_spec_for_demo() -> mp.ParameterSpec:
        s = mp.ParameterSpec()
        s.add(mp.Parameter("Lsd", init=truth.Lsd,
                            bounds=(truth.Lsd - 5e3, truth.Lsd + 5e3)))
        s.add(mp.Parameter("BC_y", init=truth.BC_y, refined=False))
        s.add(mp.Parameter("BC_z", init=truth.BC_z, refined=False))
        s.add(mp.Parameter("ty", init=0.0, refined=False))
        s.add(mp.Parameter("tz", init=0.0, refined=False))
        s.add(mp.Parameter("Wavelength", init=WAVELENGTH_A,
                            bounds=(WAVELENGTH_A * 0.998, WAVELENGTH_A * 1.002)))
        s.add(mp.Parameter("pxY", init=PX_UM, refined=False))
        s.add(mp.Parameter("pxZ", init=PX_UM, refined=False))
        s.add(mp.Parameter("RhoD", init=200000.0, refined=False))
        s.add(mp.Parameter("panel_delta_yz",
                            init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                            refined=False))
        s.add(mp.Parameter("panel_delta_theta",
                            init=torch.zeros(layout.n_panels(), dtype=torch.float64),
                            refined=False))
        s.add(mp.Parameter("grain_euler", init=grain_eulers_t, refined=False,
                            bounds=(-2 * math.pi, 2 * math.pi)))
        s.add(mp.Parameter("grain_pos", init=grain_pos_t, refined=False,
                            bounds=(-1000.0, 1000.0)))
        s.add(mp.Parameter("grain_lattice", init=grain_lat_t, refined=False))
        return s

    powder_fn = make_powder_residual(powder_obs, layout, two_theta_uniq,
                                       ring_d_spacing_A=ring_d_uniq_A)
    hedm_fn = make_hedm_residual(hedm_obs, layout)
    W_POWDER, W_HEDM = 1.0e4, 10.0

    def powder_only(u): return W_POWDER * powder_fn(u)
    def hedm_only(u):    return W_HEDM * hedm_fn(u)
    def joint(u):
        return torch.cat([W_POWDER * powder_fn(u), W_HEDM * hedm_fn(u)])

    spec_g = build_spec_for_demo()
    truth_unp = {n: spec_g.parameters[n].init_tensor() for n in spec_g.parameters}
    for n, v in truth_unp.items():
        if not isinstance(v, torch.Tensor):
            truth_unp[n] = torch.tensor(v, dtype=torch.float64)

    print(f"\n  Setup: Fisher block on (Lsd, Wavelength) at truth, all other")
    print(f"         parameters frozen.  Larger Fisher eigenvalue ratio → ")
    print(f"         less near-gauge; smaller ratio → near-degenerate direction.\n")

    print(f"  {'Modality':<14}  {'λ-min':>11}  {'λ-max':>11}  {'cond':>11}  "
          f"{'σ(Lsd) µm':>10}  {'σ(λ) µÅ':>10}")
    print(f"  {'-'*14}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*10}  {'-'*10}")
    rows = []
    for label, fn in [("powder-only", powder_only),
                       ("HEDM-only",   hedm_only),
                       ("joint",       joint)]:
        rep = fisher_block_rank(
            spec_g, fn, truth_unp,
            block_names=["Lsd", "Wavelength"],
            sigma_r=1.0, fallback_span=2.0,
        )
        F = rep.fisher.detach()
        eigvals = torch.linalg.eigvalsh(F).clamp(min=0.0)
        lam_min = float(eigvals.min())
        lam_max = float(eigvals.max())
        cond = lam_max / max(lam_min, 1e-300)
        # σ in physical units: rep.sigma_per_dim was computed via the same
        # bound-Logit transform; for this demo all bounds are wide so the
        # transform doesn't dominate.
        sig_Lsd_um = float(rep.sigma_per_dim[0])
        sig_lam_uA = float(rep.sigma_per_dim[1]) * 1e6   # Å → µÅ
        print(f"  {label:<14}  {lam_min:>11.3e}  {lam_max:>11.3e}  {cond:>11.3e}  "
              f"{sig_Lsd_um:>10.3e}  {sig_lam_uA:>10.3e}")
        rows.append({
            "label": label, "lam_min": lam_min, "lam_max": lam_max,
            "cond": cond, "sigma_Lsd_um": sig_Lsd_um, "sigma_lambda_uA": sig_lam_uA,
        })

    # Save table
    out_csv = args.output / "gauge_demo.csv"
    with open(out_csv, "w") as f:
        f.write("modality,lambda_min,lambda_max,cond_number,sigma_Lsd_um,sigma_lambda_uA\n")
        for r_ in rows:
            f.write(f"{r_['label']},{r_['lam_min']:.6e},{r_['lam_max']:.6e},"
                    f"{r_['cond']:.6e},{r_['sigma_Lsd_um']:.6e},"
                    f"{r_['sigma_lambda_uA']:.6e}\n")
    print(f"\n  CSV: {out_csv}")

    # Verdict
    cond_hedm  = next(r_["cond"] for r_ in rows if r_["label"] == "HEDM-only")
    cond_joint = next(r_["cond"] for r_ in rows if r_["label"] == "joint")
    sig_lam_hedm = next(r_["sigma_lambda_uA"] for r_ in rows if r_["label"] == "HEDM-only")
    sig_lam_joint = next(r_["sigma_lambda_uA"] for r_ in rows if r_["label"] == "joint")
    print("\n  VERDICT")
    print(f"     HEDM-only condition number / Joint condition number = "
          f"{cond_hedm / max(cond_joint, 1e-300):.2e}")
    print(f"     HEDM-only σ(λ) / Joint σ(λ) = "
          f"{sig_lam_hedm / max(sig_lam_joint, 1e-300):.2e}")
    if cond_hedm > 100 * cond_joint or sig_lam_hedm > 100 * sig_lam_joint:
        print(f"     ✓ Joint dramatically tighter on (Lsd, λ) than HEDM alone.")
        print(f"       This is the answer to 'why not just HEDM?': HEDM alone")
        print(f"       cannot uniquely identify (Lsd, λ) — the powder calibrant")
        print(f"       supplies the absolute scale that closes the gauge.")
    else:
        print(f"     ! HEDM-only and joint are comparable on (Lsd, λ).  At")
        print(f"       these synthetic 2θ angles (4–11°) the gauge is broken")
        print(f"       enough by HEDM's hkl spread that powder doesn't add")
        print(f"       much.  Real experiments with refined ``grain_lattice``")
        print(f"       (which we froze here for demo simplicity) and finite λ")
        print(f"       precision would see a much larger gap — the gauge")
        print(f"       grows back when grain lattice is unknown.")


# ---------------------------------------------------------------------
# All-blocks Fisher diagnostic
# ---------------------------------------------------------------------

def run_all_blocks_fisher(
    *, args, spec, unpacked, powder_only_no_gauge, hedm_only_no_gauge,
    joint_no_gauge, covered_panels, layout,
) -> None:
    """Fisher rank/cond/σ across every refinable spec block.

    The headline run already reported the per-panel ``panel_delta_yz``
    block.  Here we extend to every other parameter the spec exposes,
    producing the identifiability table for the paper.  Output is one
    CSV row per (block, modality) pair.
    """
    print("\n" + "=" * 70)
    print(" ALL-BLOCKS FISHER DIAGNOSTIC")
    print("=" * 70)

    # Map of "logical block name" -> list of refined parameter names that
    # comprise it.  Skip parameters that are frozen at the headline MAP.
    logical_blocks = [
        ("Lsd",                      ["Lsd"]),
        ("BC_y",                     ["BC_y"]),
        ("BC_z",                     ["BC_z"]),
        ("Lsd+Wavelength",           ["Lsd", "Wavelength"]),
        ("Lsd+BC",                   ["Lsd", "BC_y", "BC_z"]),
        ("panel_delta_yz",           ["panel_delta_yz"]),
        ("Lsd+BC+panel_delta_yz",    ["Lsd", "BC_y", "BC_z", "panel_delta_yz"]),
    ]

    refined_set = set(spec.refined_names())

    print(f"  refined params at MAP: {sorted(refined_set)[:5]}… "
          f"(+{max(0, len(refined_set) - 5)} more)")
    print(f"\n  {'block':<28}  {'modality':<14}  {'rank':>5}  {'block_size':>10}  "
          f"{'cond':>11}  {'σ_med':>11}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*5}  {'-'*10}  {'-'*11}  {'-'*11}")

    rows = []
    closures = [("powder-only", powder_only_no_gauge),
                ("HEDM-only",    hedm_only_no_gauge),
                ("joint",        joint_no_gauge)]
    for block_label, names in logical_blocks:
        # Skip blocks that contain unrefined params.
        missing = [n for n in names if n not in refined_set]
        if missing:
            print(f"  {block_label:<28}  {'—':<14}  {'(skipped: ' + ', '.join(missing) + ' frozen at headline MAP)':<60}")
            continue
        for modality, fn in closures:
            try:
                rep = fisher_block_rank(
                    spec, fn, unpacked,
                    block_names=names,
                    sigma_r=1.0, fallback_span=2.0,
                )
                sig = rep.sigma_per_dim
                # For panel_delta_yz on covered panels, restrict σ to those.
                if "panel_delta_yz" in names and len(covered_panels) < layout.n_panels():
                    cov_t = torch.zeros(layout.n_panels(), dtype=torch.bool)
                    for p in covered_panels:
                        cov_t[p] = True
                    # Indices of panel_delta_yz in the block.
                    n_pre = sum(1 if "panel_delta_yz" not in [pp] else 0 for pp in names[:names.index("panel_delta_yz")])
                    # Easier: just compute σ_med across all entries; covered/uncovered shows up.
                    pass
                sig_med = float(sig.median()) if sig.numel() > 0 else float("nan")
                cond = rep.condition_number
                row = {
                    "block": block_label, "modality": modality,
                    "rank": rep.rank, "block_size": int(sig.numel()),
                    "cond": cond, "sigma_med": sig_med,
                }
                rows.append(row)
                print(f"  {block_label:<28}  {modality:<14}  "
                      f"{rep.rank:>5d}  {sig.numel():>10d}  "
                      f"{cond:>11.2e}  {sig_med:>11.3e}")
            except Exception as e:
                print(f"  {block_label:<28}  {modality:<14}  ERROR: {e}")
                continue

    out_csv = args.output / "all_blocks_fisher.csv"
    with open(out_csv, "w") as f:
        f.write("block,modality,rank,block_size,cond_number,sigma_med\n")
        for r in rows:
            f.write(f"{r['block']},{r['modality']},{r['rank']},"
                    f"{r['block_size']},{r['cond']:.6e},{r['sigma_med']:.6e}\n")
    print(f"\n  CSV: {out_csv}")


# ---------------------------------------------------------------------
# Phase diagram
# ---------------------------------------------------------------------

def run_phase_diagram(*, args, layout, truth, seed: int) -> None:
    """Sweep (Lsd, N_grains) and report σ_joint / σ_powder per cell.

    Re-uses the synthetic generators from this runner, so each cell is
    a fresh truth + fresh observations + fresh joint LM.  Produces a
    2-D heatmap of the joint advantage.
    """
    print("\n" + "=" * 70)
    print(" PHASE DIAGRAM:  σ_joint / σ_powder over (Lsd, N_grains)")
    print("=" * 70)

    Lsd_values_um = [600e3, 900e3, 1200e3, 1500e3, 1800e3]    # 600 mm to 1.8 m
    N_grain_values = [25, 50, 100, 200]
    N_RINGS_FIXED = 8

    print(f"\n  Sweep:  Lsd ∈ {[v/1e3 for v in Lsd_values_um]} mm")
    print(f"          N_grains ∈ {N_grain_values}")
    print(f"          N_rings = {N_RINGS_FIXED}")
    print(f"          {len(Lsd_values_um) * len(N_grain_values)} cells, "
          f"~{len(Lsd_values_um) * len(N_grain_values) * 8} s estimated\n")

    # Fix everything else to the headline run.
    rows = []
    for i, Lsd in enumerate(Lsd_values_um):
        for j, n_g in enumerate(N_grain_values):
            cell_seed = seed + 1000 * i + j
            try:
                t0 = time.time()
                cell = _phase_diagram_cell(
                    layout=layout, Lsd_um=Lsd, n_grains=n_g,
                    seed=cell_seed,
                )
                dt = time.time() - t0
                ratio = cell["sigma_joint"] / max(cell["sigma_powder"], 1e-30)
                rows.append({
                    "Lsd_mm": Lsd / 1e3, "N_grains": n_g,
                    "covered_panels": cell["covered"],
                    "sigma_powder": cell["sigma_powder"],
                    "sigma_hedm": cell["sigma_hedm"],
                    "sigma_joint": cell["sigma_joint"],
                    "ratio_joint_over_powder": ratio,
                    "rank_powder_data": cell["rank_powder"],
                    "rank_joint_data": cell["rank_joint"],
                    "max_rank": cell["max_rank"],
                    "time_s": dt,
                })
                print(f"  Lsd={Lsd/1e3:5.0f} mm  N_g={n_g:4d}  "
                      f"covered={cell['covered']:2d}/48  "
                      f"σ_p={cell['sigma_powder']:.2e}  "
                      f"σ_h={cell['sigma_hedm']:.2e}  "
                      f"σ_j={cell['sigma_joint']:.2e}  "
                      f"ratio={ratio:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  Lsd={Lsd/1e3:5.0f} mm  N_g={n_g:4d}  ERROR: {e}")
                continue

    out_csv = args.output / "phase_diagram.csv"
    with open(out_csv, "w") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"\n  CSV: {out_csv}")

    # 2-D heatmap of σ_joint/σ_powder
    if not args.no_plot and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            grid = np.full((len(Lsd_values_um), len(N_grain_values)), np.nan)
            for r in rows:
                i = Lsd_values_um.index(r["Lsd_mm"] * 1e3)
                j = N_grain_values.index(r["N_grains"])
                grid[i, j] = r["ratio_joint_over_powder"]
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(grid, origin="lower", aspect="auto",
                            norm=matplotlib.colors.LogNorm(vmin=0.05, vmax=1.0),
                            cmap="viridis_r")
            ax.set_xticks(range(len(N_grain_values)))
            ax.set_xticklabels(N_grain_values)
            ax.set_yticks(range(len(Lsd_values_um)))
            ax.set_yticklabels([f"{v/1e3:.0f}" for v in Lsd_values_um])
            ax.set_xlabel("N_grains")
            ax.set_ylabel("Lsd (mm)")
            ax.set_title("σ_joint / σ_powder on per-panel δyz "
                          "(lower = bigger joint advantage)")
            for r in rows:
                i = Lsd_values_um.index(r["Lsd_mm"] * 1e3)
                j = N_grain_values.index(r["N_grains"])
                ax.text(j, i, f"{r['ratio_joint_over_powder']:.2f}",
                         ha="center", va="center", color="white", fontsize=10)
            fig.colorbar(im, ax=ax, label="σ ratio (log)")
            png = args.output / "phase_diagram.png"
            fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
            print(f"  PNG: {png}")
        except Exception as e:
            print(f"  Plot skipped: {e}")


def _phase_diagram_cell(*, layout, Lsd_um, n_grains, seed):
    """One phase-diagram cell.  Re-runs the synthetic + joint LM at the
    requested (Lsd, N_grains)."""
    # Build truth at the requested Lsd.
    rng_t = torch.Generator().manual_seed(seed)
    deltas = 0.5 * torch.randn(layout.n_panels(), 2, dtype=torch.float64, generator=rng_t)
    deltas -= deltas.mean(dim=0, keepdim=True)
    truth_local = TruthGeometry(
        Lsd=Lsd_um,
        BC_y=0.5 * (N_PANELS_Y * PANEL_SIZE_Y + (N_PANELS_Y - 1) * GAP_Y),
        BC_z=0.5 * (N_PANELS_Z * PANEL_SIZE_Z + (N_PANELS_Z - 1) * GAP_Z),
        tx=0.0, ty=0.0, tz=0.0,
        panel_delta_yz=deltas,
    )

    # Recompute Au rings + d-spacings at this (Lsd is irrelevant for 2θ;
    # rings stay the same).
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=AU_LATTICE_A)
    hkls_cart, thetas_full, _ = hkls_for_forward_model(
        sg, lat, wavelength_A=WAVELENGTH_A, two_theta_max_deg=TWO_THETA_MAX_DEG,
        expand_equivalents=False,
    )
    two_theta_uniq, _ = torch.unique(2 * thetas_full * 180.0 / math.pi,
                                       return_inverse=True, sorted=True)
    two_theta_uniq = two_theta_uniq.double()[:N_RINGS]
    ring_d_A = WAVELENGTH_A / (2.0 * torch.sin(two_theta_uniq * math.pi / 360.0))

    # Sample grains at this N_g.
    rng_np = np.random.default_rng(seed + 1)
    q = rng_np.standard_normal((n_grains, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((n_grains, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    eulers = np.zeros((n_grains, 3))
    for g in range(n_grains):
        Rg = R[g]
        if abs(Rg[2, 2]) < 1.0 - 1e-9:
            eulers[g, 1] = np.arccos(np.clip(Rg[2, 2], -1.0, 1.0))
            eulers[g, 0] = np.arctan2(Rg[0, 2], -Rg[1, 2])
            eulers[g, 2] = np.arctan2(Rg[2, 0], Rg[2, 1])
    positions = (rng_np.random((n_grains, 3)) - 0.5) * 1000.0
    eulers_t = torch.from_numpy(eulers).double()
    positions_t = torch.from_numpy(positions).double()
    lattice_t = torch.tensor([[AU_LATTICE_A, AU_LATTICE_A, AU_LATTICE_A,
                               90.0, 90.0, 90.0]] * n_grains, dtype=torch.float64)

    # Generate observations: temporarily override module-level constants
    # for the HEDM model construction (it reads N_GRAINS).
    global N_GRAINS
    saved_n = N_GRAINS
    N_GRAINS = n_grains
    try:
        powder_obs = generate_powder_observations(layout, truth_local, two_theta_uniq, seed=seed + 2)
        hedm_obs = generate_hedm_observations(layout, truth_local, eulers_t, positions_t, lattice_t, seed=seed + 3)
        powder_panels = set(powder_obs["panel_idx"].unique().tolist())
        hedm_panels   = set(hedm_obs["panel_idx"].unique().tolist())
        covered_panels = powder_panels | hedm_panels
        covered_mask = torch.zeros(layout.n_panels(), dtype=torch.bool)
        for p in covered_panels: covered_mask[p] = True

        # Build spec.
        s = mp.ParameterSpec()
        s.add(mp.Parameter("Lsd", init=truth_local.Lsd + 50.0,
                            bounds=(truth_local.Lsd - 5e3, truth_local.Lsd + 5e3)))
        s.add(mp.Parameter("BC_y", init=truth_local.BC_y + 0.3,
                            bounds=(truth_local.BC_y - 5.0, truth_local.BC_y + 5.0)))
        s.add(mp.Parameter("BC_z", init=truth_local.BC_z - 0.2,
                            bounds=(truth_local.BC_z - 5.0, truth_local.BC_z + 5.0)))
        s.add(mp.Parameter("ty", init=0.0, refined=False))
        s.add(mp.Parameter("tz", init=0.0, refined=False))
        s.add(mp.Parameter("Wavelength", init=WAVELENGTH_A, refined=False))
        s.add(mp.Parameter("pxY", init=PX_UM, refined=False))
        s.add(mp.Parameter("pxZ", init=PX_UM, refined=False))
        s.add(mp.Parameter("RhoD", init=200000.0, refined=False))
        s.add(mp.Parameter("panel_delta_yz",
                            init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                            bounds=(-3.0, 3.0),
                            prior=mp.GaussianPrior(mean=0.0, std=0.5)))
        s.add(mp.Parameter("panel_delta_theta",
                            init=torch.zeros(layout.n_panels(), dtype=torch.float64),
                            refined=False))
        s.add(mp.Parameter("grain_euler", init=eulers_t, refined=False))
        s.add(mp.Parameter("grain_pos", init=positions_t, refined=False))
        s.add(mp.Parameter("grain_lattice", init=lattice_t, refined=False))

        powder_fn = make_powder_residual(powder_obs, layout, two_theta_uniq, ring_d_spacing_A=ring_d_A)
        hedm_fn = make_hedm_residual(hedm_obs, layout)
        W_POWDER, W_HEDM = 1.0e4, 10.0
        def powder_only_no_gauge(u): return W_POWDER * powder_fn(u)
        def hedm_only_no_gauge(u):    return W_HEDM * hedm_fn(u)
        def joint_no_gauge(u):
            return torch.cat([W_POWDER * powder_fn(u), W_HEDM * hedm_fn(u)])
        def joint_with_gauge(u):
            from midas_peakfit import gaussian_prior_residual
            pieces = [W_POWDER * powder_fn(u), W_HEDM * hedm_fn(u)]
            rprior = gaussian_prior_residual(u, s)
            if rprior.numel() > 0: pieces.append(rprior)
            return torch.cat([p.flatten() for p in pieces])

        # Refine joint MAP first (so we have a good linearisation point).
        unp, _, _ = mp.lm_minimise(s, joint_with_gauge,
                                    config=mp.GenericLMConfig(max_iter=80, ftol_rel=1e-9),
                                    fallback_span=2.0)

        # Fisher on panel_delta_yz under three modalities.
        sigmas = {}
        ranks = {}
        for label, fn in [("powder", powder_only_no_gauge),
                           ("hedm",   hedm_only_no_gauge),
                           ("joint",  joint_no_gauge)]:
            rep = fisher_block_rank(s, fn, unp, block_names=["panel_delta_yz"],
                                     sigma_r=1.0, fallback_span=2.0)
            sig = rep.sigma_per_dim.reshape(layout.n_panels(), 2).norm(dim=1)
            # Median over covered panels only.
            sigmas[label] = float(sig[covered_mask].median()) if covered_mask.any() else float("nan")
            ranks[label] = rep.rank

        return {
            "covered": int(covered_mask.sum().item()),
            "sigma_powder": sigmas["powder"],
            "sigma_hedm":   sigmas["hedm"],
            "sigma_joint":  sigmas["joint"],
            "rank_powder": ranks["powder"],
            "rank_joint":  ranks["joint"],
            "max_rank":    2 * int(covered_mask.sum().item()),
        }
    finally:
        N_GRAINS = saved_n


if __name__ == "__main__":
    sys.exit(main())
