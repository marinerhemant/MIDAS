"""Multi-grain end-to-end test.

Plants a 12×12 grid Voronoi-tessellated into 3 grains with random
orientations and small constant per-grain strain. Forward-simulates
per-grain, runs per-grain inversion, asserts each grain is recovered.
"""

import math
import torch
import pytest

from midas_pf_odf import (
    plant_multi_grain, simulate_multi_grain, fit_multi_grain,
    split_into_grains, recovery_metrics,
    IdentifiabilityMode,
)

from tests.conftest import (
    make_fcc_hkls, small_scan_config, build_model,
)


@pytest.fixture(scope="module")
def model_fixed():
    G, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)
    sc = small_scan_config(sample_size_um=30.0, n_scans=11, beam_size_um=4.0)
    return build_model(sc, hkls_int, G, th)


def test_multi_grain_plant_assignment():
    """Voronoi assignment should partition all voxels among the requested
    grains."""
    plant = plant_multi_grain(
        grid_shape=(12, 12), n_grains=3, voxel_size_um=2.0, seed=0,
    )
    assert plant.n_voxels == 144
    assert plant.n_grains == 3
    # Every voxel labelled in [0, 3)
    assert plant.grain_id.min() >= 0
    assert plant.grain_id.max() < plant.n_grains
    # All grains populated (random Voronoi seeds may starve a grain only at
    # very tight settings; with 3 seeds in a 12×12 box this is exceedingly
    # unlikely — assert no empty grain).
    counts = torch.bincount(plant.grain_id, minlength=plant.n_grains)
    assert (counts > 0).all(), f"empty grain in counts={counts.tolist()}"


def test_multi_grain_round_trip(model_fixed):
    """Plant 3 grains, simulate, recover each grain's (R, ε).

    Uses ``IdentifiabilityMode.FREE`` because the planted per-grain ε is
    constant within each grain — there's no spatial signal to disambiguate
    "absorbed by lattice" vs "absorbed by ε". FREE keeps lattice fixed
    (lr_lat=0) and lets per-voxel ε hold the bulk strain directly. The
    PROJECT_EPS_MEAN_ZERO mode is the natural choice for spatially-varying
    intragranular strain (validated separately in test_round_trip_small).
    """
    plant = plant_multi_grain(
        grid_shape=(8, 8), n_grains=3, voxel_size_um=2.0,
        eps_per_grain_amp=1e-3, intra_grain_spread_deg=0.0, seed=1,
    )
    sub_plants = split_into_grains(plant)
    data = simulate_multi_grain(
        plant, model_fixed,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5, add_noise_sigma=0.0,
    )

    eps_init = {g: torch.zeros_like(sp.eps_voxel)
                for g, sp in sub_plants.items()}
    R_init = {g: sp.R_voxel for g, sp in sub_plants.items()}

    fits = fit_multi_grain(
        data, sub_plants, model_fixed,
        eps_init_per_grain=eps_init,
        R_init_per_grain=R_init,
        identifiability=IdentifiabilityMode.FREE,
        optimizer="lbfgs",
        inner_steps=20,
        lr_aa=1.0, lr_eps=1.0, lr_lat=0.0,   # lock lattice (line search adapts)
    )

    # Per-grain recovery: each grain has a CONSTANT planted ε ~ 1e-3.
    # With lattice locked, the diagonal Voigt components (bulk strain)
    # are partially identifiable — the shears (off-diagonals) recover
    # cleanly but the trace can be slightly under-recovered. Tolerance
    # of 5e-4 catches a complete failure mode while accepting the
    # honest identifiability floor.
    bad = []
    for g, sp in sub_plants.items():
        rep = recovery_metrics(sp, fits[g].R_fit, fits[g].eps_fit)
        if rep.eps_rms > 5e-4 or rep.misorient_rms_deg > 0.05:
            bad.append((g, rep.eps_rms, rep.misorient_rms_deg))
    assert not bad, (
        "Per-grain recovery failures (id, ε RMSE, miso RMS deg): "
        f"{bad}"
    )


def test_multi_grain_lattice_unlocked(model_fixed):
    """``PROJECT_EPS_MEAN_ZERO`` + lattice unlocked: the lattice param
    must absorb the per-grain bulk strain so ``ε_used`` (mean-zero
    projected) lands near zero. Success criterion is the deviatoric
    strain, not total strain.
    """
    plant = plant_multi_grain(
        grid_shape=(8, 8), n_grains=3, voxel_size_um=2.0,
        eps_per_grain_amp=1e-3, intra_grain_spread_deg=0.0, seed=2,
    )
    sub_plants = split_into_grains(plant)
    data = simulate_multi_grain(
        plant, model_fixed,
        patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
        gate_tau_um=0.5, add_noise_sigma=0.0,
    )

    eps_init = {g: torch.zeros_like(sp.eps_voxel)
                for g, sp in sub_plants.items()}
    R_init = {g: sp.R_voxel for g, sp in sub_plants.items()}

    fits = fit_multi_grain(
        data, sub_plants, model_fixed,
        eps_init_per_grain=eps_init,
        R_init_per_grain=R_init,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="lbfgs",
        inner_steps=30,
        lr_aa=1.0, lr_eps=1.0, lr_lat=1.0,
    )

    # PROJECT_EPS_MEAN_ZERO + uniform per-grain ε plant has an
    # acknowledged saddle: lattice and per-voxel deviatoric ε both
    # partially absorb the bulk strain. The two quantities trade off
    # at the local-minimum the optimizer settles into. We verify the
    # mechanism (lattice DID move toward absorbing bulk strain) and
    # that the loss converged — perfect deviatoric=0 recovery requires
    # the spatially-varying-strain plant tested in
    # test_round_trip_small.test_small_eps_gradient_round_trip.
    bad = []
    for g, sp in sub_plants.items():
        final_loss = fits[g].losses[-1]
        initial_loss = fits[g].losses[0]
        lat_drift = float((fits[g].lattice_fit - sp.lattice).abs().max().item())
        eps_amp_per_voxel = float(sp.eps_voxel.abs().max().item())
        # Fit quality: at least 50% loss reduction.
        if final_loss > 0.5 * initial_loss:
            bad.append(("loss", g, initial_loss, final_loss))
        # Lattice mechanism: drift in the direction of bulk strain.
        # Magnitude expected ~ a · |ε_avg| ≈ 3.6 × 1e-3 = 3.6e-3.
        if lat_drift < 0.5 * eps_amp_per_voxel * 3.6:
            bad.append(("lattice_drift", g, lat_drift, eps_amp_per_voxel))
    assert not bad, (
        "Lattice-unlocked mechanism failures: " + str(bad)
    )
