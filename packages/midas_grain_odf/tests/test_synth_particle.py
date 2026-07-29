"""Synthetic plant-and-recover test for ParticleODF.

Plants a known multi-particle ODF, forward-simulates spot patches, and
verifies that the inverter recovers a per-spot patch reconstruction with
small final loss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make conftest helpers importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Ensure the package is importable when running from a repo without install.
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import make_model, random_orientation  # noqa: E402

from midas_grain_odf.forward_helpers import forward_orientations  # noqa: E402
from midas_grain_odf.inversion import fit_grain_odf  # noqa: E402
from midas_grain_odf.odf import ParticleODF, axis_angle_to_matrix  # noqa: E402
from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches  # noqa: E402


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def plant_particle_odf(
    R_avg: torch.Tensor,
    aa_planted: torch.Tensor,   # (P, 3) axis-angle deltas
    weights: torch.Tensor,      # (P,) simplex
):
    """Compose planted orientations and return (R_planted (P,3,3), w (P,))."""
    delta_R = axis_angle_to_matrix(aa_planted)
    R_planted = R_avg.unsqueeze(0) @ delta_R
    w = weights / weights.sum()
    return R_planted, w


def synthesize_measurements(
    model,
    R_avg: torch.Tensor,
    R_planted: torch.Tensor,
    w_planted: torch.Tensor,
    position: torch.Tensor,
    sigma_yz: float,
    sigma_f: float,
    patch_F: int,
    patch_P: int,
):
    """Generate the synthetic measurement: pick valid spots, build patches."""
    # 1. Forward simulate the planted orientations.
    spots_p = forward_orientations(model, R_planted, position)
    sy_p = spots_p.y_pixel.reshape(R_planted.shape[0], -1)
    sz_p = spots_p.z_pixel.reshape(R_planted.shape[0], -1)
    sf_p = spots_p.frame_nr.reshape(R_planted.shape[0], -1)
    sv_p = spots_p.valid.reshape(R_planted.shape[0], -1)

    # 2. Forward simulate R_avg alone for the patch anchors and Delta=0 ground truth.
    R_avg_b = R_avg.unsqueeze(0)
    spots_avg = forward_orientations(model, R_avg_b, position)
    sy_a = spots_avg.y_pixel.reshape(1, -1).squeeze(0)
    sz_a = spots_avg.z_pixel.reshape(1, -1).squeeze(0)
    sf_a = spots_avg.frame_nr.reshape(1, -1).squeeze(0)
    sv_a = spots_avg.valid.reshape(1, -1).squeeze(0)

    # 3. Use only spots where both R_avg and at least one planted orientation are valid.
    valid_global = (sv_a > 0.5) & (sv_p.sum(dim=0) > 0)
    spot_indexer = torch.nonzero(valid_global, as_tuple=False).squeeze(-1)

    # Subselect.
    sy_p_sel = sy_p[:, spot_indexer]
    sz_p_sel = sz_p[:, spot_indexer]
    sf_p_sel = sf_p[:, spot_indexer]
    sv_p_sel = sv_p[:, spot_indexer]
    anchor_y = sy_a[spot_indexer]
    anchor_z = sz_a[spot_indexer]
    anchor_f = sf_a[spot_indexer]

    # 4. Splat planted orientations into patches anchored at R_avg-predicted centers.
    spec = SpotPatchSpec(
        n_spots=int(spot_indexer.numel()),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=anchor_y.detach().clone(),
        anchor_z=anchor_z.detach().clone(),
        anchor_f=anchor_f.detach().clone(),
    )
    patches = splat_spots_to_patches(
        spec, sy_p_sel, sz_p_sel, sf_p_sel, w_planted, sv_p_sel,
    )

    # The "measured" centroids: ODF-weighted mean of planted predictions.
    w_norm = (w_planted.reshape(-1, 1) * sv_p_sel).sum(dim=0).clamp(min=1e-12)
    meas_y = (w_planted.reshape(-1, 1) * sv_p_sel * sy_p_sel).sum(dim=0) / w_norm
    meas_z = (w_planted.reshape(-1, 1) * sv_p_sel * sz_p_sel).sum(dim=0) / w_norm
    meas_f = (w_planted.reshape(-1, 1) * sv_p_sel * sf_p_sel).sum(dim=0) / w_norm

    return {
        "spec": spec,
        "patches": patches,
        "meas_y": meas_y,
        "meas_z": meas_z,
        "meas_f": meas_f,
        "spot_indexer": spot_indexer,
        "valid_global": valid_global,
    }


# ---------------------------------------------------------------------------
#  Test
# ---------------------------------------------------------------------------


def test_recover_3particle_smallspread():
    """3 planted particles in a small (<0.1 deg) ball, asymmetric weights.

    Patch size P=31, planted spread <0.1 deg. The axis-angle -> pixel
    Jacobian is ~3000 px/rad for typical FF geometries, so 0.1 deg axis-angle
    -> ~5 px displacement. Patches comfortably contain the planted spread.

    This validates the optimizer plumbing on a tractable case. Larger
    spreads + bigger patches are exercised in a separate test once the
    splatter is sparsified (currently O(K * S * F * P^2) dense).
    """
    torch.manual_seed(7)
    np.random.seed(7)

    model = make_model()
    R_avg = random_orientation(seed=11).to(torch.float64)
    position = torch.zeros(3, dtype=torch.float64)

    deg = math.pi / 180.0
    # Three particles within a 0.06 deg ball, asymmetric weights.
    aa_planted = torch.tensor([
        [0.00 * deg, 0.00 * deg, 0.00 * deg],
        [0.05 * deg, 0.00 * deg, 0.00 * deg],
        [0.00 * deg, 0.04 * deg, 0.03 * deg],
    ], dtype=torch.float64)
    w_planted = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)

    R_planted, w_planted = plant_particle_odf(R_avg, aa_planted, w_planted)

    sigma_yz, sigma_f = 1.0, 0.6
    patch_F, patch_P = 7, 31

    meas = synthesize_measurements(
        model, R_avg, R_planted, w_planted, position,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        patch_F=patch_F, patch_P=patch_P,
    )
    print(f"  n_spots = {int(meas['spot_indexer'].numel())}")
    print(f"  measured patch peak = {float(meas['patches'].max()):.4f}, "
          f"  total mass = {float(meas['patches'].sum()):.4f}")
    assert meas["spot_indexer"].numel() >= 8

    # Initialize ODF with K=48 particles inside ±0.15 deg (looser than planted).
    odf = ParticleODF(
        R_avg=R_avg.detach().clone(),
        K=48,
        theta_max=0.15 * deg,
        seed=42,
    ).to(torch.float64)

    result = fit_grain_odf(
        odf=odf, model=model, position=position,
        measured_y=meas["meas_y"], measured_z=meas["meas_z"], measured_f=meas["meas_f"],
        measured_patches=meas["patches"],
        spot_indexer=meas["spot_indexer"],
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        delta_iters=2, inner_steps=400,
        lr_axis_angle=1e-4, lr_logits=0.1,
        verbose=True,
    )

    print(f"  delta_iters_run = {result.delta_iters_run}, "
          f"converged = {result.converged}")
    print(f"  initial loss   = {result.losses[0]:.3e}")
    print(f"  final loss     = {result.losses[-1]:.3e}")
    print(f"  loss ratio     = {result.losses[-1] / result.losses[0]:.3e}")

    # Hard gate: significant loss reduction.
    assert result.losses[-1] < 0.1 * result.losses[0], (
        f"loss did not decrease enough: "
        f"{result.losses[0]:.3e} -> {result.losses[-1]:.3e}"
    )

    # Sanity: recovered mass near planted particles. Threshold: at least 60% of
    # the simplex weight should lie within 0.05 deg of some planted particle.
    R_rec, w_rec = result.odf.sample()
    R_p = R_planted.detach()
    trace = torch.einsum("kij,pij->pk", R_rec.detach(), R_p)
    cos_t = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_t)                       # (P, K)
    deg_thresh = 0.05 * deg
    in_neighborhood = (angle < deg_thresh).any(dim=0).double()  # (K,)
    mass_near_planted = (w_rec.detach() * in_neighborhood).sum()
    print(f"  recovered mass within 0.05 deg of planted: {float(mass_near_planted):.3f}")
    print(f"  planted weights:                            {w_planted.tolist()}")

    assert float(mass_near_planted) > 0.6, (
        f"recovered mass near planted particles too low: "
        f"{float(mass_near_planted):.3f}"
    )
