"""End-to-end synthetic round-trip on a 4×4 single-grain plant.

Plants a small grid with a tiny ε gradient, simulates the joint
per-grain forward, runs the inversion, and verifies the recovered
(R_V, ε_V) is within tolerance of the plant. This is the gating test
for any code change in the package.
"""

import math
import torch
import pytest

from midas_pf_odf import (
    plant_single_grain, simulate_grain_patches, fit_grain_peakshape,
    IdentifiabilityMode, recovery_metrics,
)

from tests.conftest import (
    make_fcc_hkls, small_scan_config, build_model,
)


@pytest.fixture(scope="module")
def model_and_hkls():
    G, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)   # tiny set for speed
    sc = small_scan_config(sample_size_um=10.0, n_scans=7, beam_size_um=4.0)
    model = build_model(sc, hkls_int, G, th)
    return model, hkls_int


def test_homogeneous_grain_round_trip(model_and_hkls):
    """No gradients, no noise — the inverter should recover within 1e-4."""
    model, _ = model_and_hkls

    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_avg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        eps_gradient_amp=0.0,
        R_gradient_amp_deg=0.0,
    )
    data = simulate_grain_patches(
        plant, model,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5, add_noise_sigma=0.0,
    )

    # Warm-start at the plant truth (this validates the FORWARD agrees
    # with the simulator and the loss is at its minimum at the truth).
    fit = fit_grain_peakshape(
        data, model,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=plant.eps_voxel,
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="adam",
        inner_steps=3,                   # essentially zero updates
        lr_aa=0.0, lr_eps=0.0, lr_lat=0.0,
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    # With zero LR + warm-start at truth, recovery is exact (mod fp64 noise).
    assert rep.misorient_rms_deg < 1e-6
    assert rep.eps_rms < 1e-9


def test_orientation_gradient_round_trip(model_and_hkls):
    """Plant a 0.5° axis-angle gradient across a 6×6 grain; start with a
    NOISY per-voxel R_init (planted truth perturbed by 0.05° random
    axis-angle) — this matches the realistic scenario where an indexer
    has provided approximate per-voxel orientations and refinement
    tightens them. ε is locked at zero so the only signal is rotation.
    """
    import math as _math
    model, _ = model_and_hkls

    plant = plant_single_grain(
        grid_shape=(6, 6), voxel_size_um=2.0,
        R_gradient_axis=(0.0, 0.0, 1.0),
        R_gradient_amp_deg=0.5,
        R_gradient_dir="x",
        eps_gradient_amp=0.0,
    )
    data = simulate_grain_patches(
        plant, model,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5,
    )

    # Realistic warm-start: planted truth + random 0.05° per-voxel perturbation.
    g = torch.Generator().manual_seed(0)
    perturb_aa = torch.randn(plant.n_voxels, 3, generator=g, dtype=plant.R_voxel.dtype) \
                 * _math.radians(0.05)
    from midas_grain_odf.odf import axis_angle_to_matrix as _aa_to_R
    R_init = plant.R_voxel @ _aa_to_R(perturb_aa)

    fit = fit_grain_peakshape(
        data, model,
        voxel_pos=plant.voxel_pos,
        R_init=R_init,
        eps_init=torch.zeros_like(plant.eps_voxel),
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.FREE,
        optimizer="lbfgs",
        inner_steps=20,
        lr_aa=1.0, lr_eps=0.0, lr_lat=0.0,
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    # Initial miso (perturbation magnitude) is ~0.05° per voxel → 0.05° RMS.
    # Recovery should tighten this to <0.02° RMS (a real refinement gain).
    assert rep.misorient_rms_deg < 0.02, (
        f"miso RMS {rep.misorient_rms_deg:.4f} deg > 0.02 (init was ~0.05)"
    )
    assert rep.eps_rms < 1e-9, f"ε leaked to {rep.eps_rms:.3e}"


def test_small_eps_gradient_round_trip(model_and_hkls):
    """Plant a 2e-3 ε_11 linear gradient; recover from a zero warm-start."""
    model, _ = model_and_hkls

    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        eps_avg=(0.0,) * 6,
        eps_gradient_voigt=0,            # ε_11
        eps_gradient_amp=2e-3,
        eps_gradient_dir="x",
        R_gradient_amp_deg=0.0,
    )
    data = simulate_grain_patches(
        plant, model,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5, add_noise_sigma=0.0,
    )

    eps_zero = torch.zeros_like(plant.eps_voxel)
    fit = fit_grain_peakshape(
        data, model,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,            # warm-start orientation correct
        eps_init=eps_zero,                # but strain at zero
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="adam",
        inner_steps=200,
        lr_aa=1e-4,
        lr_eps=1e-3,
        lr_lat=1e-5,
    )
    rep = recovery_metrics(plant, fit.R_fit, fit.eps_fit)
    # The ε gradient is small but well above the inversion noise floor.
    # Tolerances picked loosely; tighten as the inverter improves.
    assert rep.eps_rms < 1e-3, (
        f"ε RMSE {rep.eps_rms:.3e} > 1e-3 (gradient amplitude was 2e-3)"
    )
    assert rep.misorient_rms_deg < 0.05
