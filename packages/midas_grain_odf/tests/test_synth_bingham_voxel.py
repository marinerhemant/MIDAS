"""Synthetic plant-and-recover tests for BinghamMixtureODF and VoxelGridODF.

Same planted ODF as the particle test, but solved with the other two
parameterizations. Hard MVP gate from the implementation plan: each ODF
class must pass plant-and-recover or the MVP isn't done.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import make_model, random_orientation  # noqa: E402

from midas_grain_odf.inversion import fit_grain_odf  # noqa: E402
from midas_grain_odf.odf import (
    BinghamMixtureODF,
    VoxelGridODF,
    axis_angle_to_matrix,
)  # noqa: E402
from test_synth_particle import (  # noqa: E402
    plant_particle_odf, synthesize_measurements,
)


def _shared_setup():
    torch.manual_seed(7)
    np.random.seed(7)

    model = make_model()
    R_avg = random_orientation(seed=11).to(torch.float64)
    position = torch.zeros(3, dtype=torch.float64)

    deg = math.pi / 180.0
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
    return {
        "model": model, "R_avg": R_avg, "position": position,
        "R_planted": R_planted, "w_planted": w_planted,
        "meas": meas,
        "sigma_yz": sigma_yz, "sigma_f": sigma_f,
        "patch_F": patch_F, "patch_P": patch_P,
        "deg": deg,
    }


def test_bingham_mixture_recovers_planted():
    """BinghamMixtureODF: fit modes + concentrations + mixture weights."""
    s = _shared_setup()
    odf = BinghamMixtureODF(
        R_avg=s["R_avg"].detach().clone(),
        n_modes=4, K_per_mode=24,
        theta_max=0.15 * s["deg"], seed=42,
    ).to(torch.float64)

    result = fit_grain_odf(
        odf=odf, model=s["model"], position=s["position"],
        measured_y=s["meas"]["meas_y"],
        measured_z=s["meas"]["meas_z"],
        measured_f=s["meas"]["meas_f"],
        measured_patches=s["meas"]["patches"],
        spot_indexer=s["meas"]["spot_indexer"],
        patch_F=s["patch_F"], patch_P=s["patch_P"],
        sigma_yz=s["sigma_yz"], sigma_f=s["sigma_f"],
        delta_iters=2, inner_steps=400,
        lr_axis_angle=1e-4, lr_logits=0.1,
        verbose=False,
    )

    print(f"  initial loss = {result.losses[0]:.3e}")
    print(f"  final loss   = {result.losses[-1]:.3e}")
    print(f"  loss ratio   = {result.losses[-1] / result.losses[0]:.3e}")

    # The Bingham mixture is smoother than the particle ODF; it concentrates
    # density near each planted location but doesn't put the entire simplex
    # mass within a 0.05 deg ball. We use a looser shape-fit gate.
    assert result.losses[-1] < 0.05 * result.losses[0], (
        f"loss did not decrease enough: {result.losses[0]:.3e} -> "
        f"{result.losses[-1]:.3e}"
    )

    # Sanity: recovered density should put non-trivial mass within 0.1 deg of
    # each planted particle.
    R_rec, w_rec = result.odf.sample()
    R_p = s["R_planted"].detach()
    trace = torch.einsum("kij,pij->pk", R_rec.detach(), R_p)
    cos_t = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_t)
    deg_thresh = 0.1 * s["deg"]
    near_any = (angle < deg_thresh).any(dim=0).double()
    mass_near = (w_rec.detach() * near_any).sum()
    print(f"  recovered mass within 0.1 deg of planted: {float(mass_near):.3f}")

    assert float(mass_near) > 0.4, (
        f"recovered mass near planted too low: {float(mass_near):.3f}"
    )


def test_voxel_grid_recovers_planted():
    """VoxelGridODF: 7^3 = 343 grid nodes, fit logits only."""
    s = _shared_setup()
    odf = VoxelGridODF(
        R_avg=s["R_avg"].detach().clone(),
        n_per_axis=7, theta_max=0.10 * s["deg"],
    ).to(torch.float64)

    result = fit_grain_odf(
        odf=odf, model=s["model"], position=s["position"],
        measured_y=s["meas"]["meas_y"],
        measured_z=s["meas"]["meas_z"],
        measured_f=s["meas"]["meas_f"],
        measured_patches=s["meas"]["patches"],
        spot_indexer=s["meas"]["spot_indexer"],
        patch_F=s["patch_F"], patch_P=s["patch_P"],
        sigma_yz=s["sigma_yz"], sigma_f=s["sigma_f"],
        delta_iters=2, inner_steps=400,
        lr_axis_angle=1e-4, lr_logits=0.1,
        verbose=False,
    )

    print(f"  initial loss = {result.losses[0]:.3e}")
    print(f"  final loss   = {result.losses[-1]:.3e}")
    print(f"  loss ratio   = {result.losses[-1] / result.losses[0]:.3e}")

    assert result.losses[-1] < 0.1 * result.losses[0], (
        f"loss did not decrease enough: {result.losses[0]:.3e} -> "
        f"{result.losses[-1]:.3e}"
    )

    R_rec, w_rec = result.odf.sample()
    R_p = s["R_planted"].detach()
    trace = torch.einsum("kij,pij->pk", R_rec.detach(), R_p)
    cos_t = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_t)
    # Voxel-grid resolution is theta_max / (n_per_axis-1) = 0.0167 deg per cell.
    # The grid won't have a node exactly at a planted location, so use a
    # one-cell tolerance.
    deg_thresh = (0.1 / 6) * s["deg"] * 1.5
    near_any = (angle < deg_thresh).any(dim=0).double()
    mass_near = (w_rec.detach() * near_any).sum()
    print(f"  recovered mass within {deg_thresh / s['deg']:.4f} deg of planted: "
          f"{float(mass_near):.3f}")
    assert float(mass_near) > 0.4, (
        f"recovered mass near planted too low: {float(mass_near):.3f}"
    )
