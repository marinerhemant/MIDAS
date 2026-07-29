"""Synthetic identifiability test for joint σ_θ + σ_ε disentanglement.

Plants four scenarios on a single-grain phantom, simulates patches with the
anisotropic forward (radial broadening from σ_ε, eta+ω broadening from σ_θ),
runs the joint inversion with refine_spread + refine_strain_spread, and
checks recovery + crosstalk.

Scenarios:
  (a) σ_θ = 0,  σ_ε = 0    — recovery floor (both stay near zero)
  (b) σ_θ = 1, σ_ε = 0    — orientation-only (σ_θ recovered, σ_ε stays low)
  (c) σ_θ = 0,  σ_ε = 1   — strain-only (σ_ε recovered, σ_θ stays low)
  (d) σ_θ = 1, σ_ε = 1   — both planted (both recovered, crosstalk small)
"""
import math
import torch
import pytest

from midas_pf_odf import (
    plant_single_grain, simulate_grain_patches, fit_grain_peakshape,
    IdentifiabilityMode,
)

from tests.conftest import (
    make_fcc_hkls, small_scan_config, build_model,
)


@pytest.fixture(scope="module")
def model_and_plant():
    G, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)
    sc = small_scan_config(sample_size_um=10.0, n_scans=7, beam_size_um=4.0)
    model = build_model(sc, hkls_int, G, th)
    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_avg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        eps_gradient_amp=0.0,
        R_gradient_amp_deg=0.0,
    )
    return model, plant


def _run_scenario(model, plant, sigma_theta_px, sigma_eps_px,
                   refine_th=True, refine_eps=True,
                   inner_steps=80, lr_sp=2e3, lr_eps_sp=2e3):
    """Plant scenario, simulate with anisotropic σ, run joint fit, return
    (final σ_θ_fit_mean, σ_ε_fit_mean)."""
    G = plant.n_voxels
    dtype = plant.R_voxel.dtype
    dev = plant.R_voxel.device
    sp_theta = torch.full((G,), float(sigma_theta_px), dtype=dtype, device=dev)
    sp_eps = torch.full((G,), float(sigma_eps_px), dtype=dtype, device=dev)
    data = simulate_grain_patches(
        plant, model,
        patch_F=5, patch_P=15, sigma_yz=0.6, sigma_f=0.4,
        gate_tau_um=0.5, add_noise_sigma=0.0,
        voxel_spread=sp_theta if sigma_theta_px > 0 else None,
        voxel_strain_spread=sp_eps if (sigma_eps_px > 0 or refine_eps) else None,
    )

    fit = fit_grain_peakshape(
        data, model,
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=plant.eps_voxel,
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="adam",
        inner_steps=inner_steps,
        lr_aa=0.0, lr_eps=0.0, lr_lat=0.0,                # freeze (R, ε)
        refine_spread=refine_th,
        spread_init=(torch.full((G,), 0.5, dtype=dtype, device=dev)
                     if refine_th else None),
        lr_spread=lr_sp,
        refine_strain_spread=refine_eps,
        strain_spread_init=(torch.full((G,), 0.5, dtype=dtype, device=dev)
                             if refine_eps else None),
        lr_strain_spread=lr_eps_sp,
    )
    th_fit = (fit.spread_fit.mean().item() if fit.spread_fit is not None else 0.0)
    eps_fit = (fit.strain_spread_fit.mean().item()
                if fit.strain_spread_fit is not None else 0.0)
    return th_fit, eps_fit, fit.losses[-1]


@pytest.mark.parametrize("planted_theta,planted_eps", [
    (0.0, 0.0),     # (a) recovery floor
    (1.0, 0.0),     # (b) orientation only
    (0.0, 1.0),     # (c) strain only
    (1.0, 1.0),     # (d) both
])
def test_aniso_identifiability(model_and_plant, planted_theta, planted_eps):
    """For each planted scenario the recovered (σ_θ, σ_ε) match the truth
    to within 30 %, with the orthogonal coordinate well below the planted one."""
    model, plant = model_and_plant
    th_fit, eps_fit, loss = _run_scenario(
        model, plant, planted_theta, planted_eps,
        refine_th=True, refine_eps=True, inner_steps=120,
    )
    print(f"\nplanted (σ_θ, σ_ε)=({planted_theta:.2f}, {planted_eps:.2f}) px → "
          f"recovered ({th_fit:.3f}, {eps_fit:.3f}) px  final loss {loss:.3e}")

    # Recovery: the planted parameter should land within 30% of the truth.
    if planted_theta > 0:
        assert abs(th_fit - planted_theta) / planted_theta < 0.30, (
            f"σ_θ recovery off: planted {planted_theta} → recovered {th_fit}"
        )
    if planted_eps > 0:
        assert abs(eps_fit - planted_eps) / planted_eps < 0.30, (
            f"σ_ε recovery off: planted {planted_eps} → recovered {eps_fit}"
        )

    # Crosstalk: when one is planted ZERO, the recovered value should be
    # << the other planted one (orthogonal observables ⇒ small crosstalk).
    if planted_theta == 0 and planted_eps > 0:
        assert th_fit < 0.5 * planted_eps, (
            f"crosstalk σ_θ from σ_ε too large: {th_fit} ≥ 0.5 × {planted_eps}"
        )
    if planted_eps == 0 and planted_theta > 0:
        assert eps_fit < 0.5 * planted_theta, (
            f"crosstalk σ_ε from σ_θ too large: {eps_fit} ≥ 0.5 × {planted_theta}"
        )
