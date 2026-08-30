"""Parity test: chunked splatter matches the unchunked path.

For the same plant + warm-start + optimizer, the recovery numbers must
agree (within fp32/fp64 noise) regardless of chunk_size_g.
"""

import math
import torch
import pytest

from midas_pf_odf import (
    plant_single_grain, simulate_grain_patches,
    fit_grain_peakshape, IdentifiabilityMode, recovery_metrics,
)
from midas_pf_odf.forward import joint_grain_forward
from midas_grain_odf.spot_extract import SpotPatchSpec

from tests.conftest import make_fcc_hkls, small_scan_config, build_model


def test_simulator_chunked_parity():
    """Simulator output identical with and without chunking (no_grad path)."""
    G_cart, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)
    sc = small_scan_config(sample_size_um=10.0, n_scans=7, beam_size_um=4.0)
    model = build_model(sc, hkls_int, G_cart, th)

    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        eps_gradient_voigt=0, eps_gradient_amp=2e-3, eps_gradient_dir="x",
    )

    data_full = simulate_grain_patches(
        plant, model, patch_F=5, patch_P=15,
        sigma_yz=1.0, sigma_f=0.6, gate_tau_um=0.5,
        chunk_size_g=None,
    )
    data_chunk = simulate_grain_patches(
        plant, model, patch_F=5, patch_P=15,
        sigma_yz=1.0, sigma_f=0.6, gate_tau_um=0.5,
        chunk_size_g=4,           # 16 voxels in 4 chunks of 4
    )
    assert torch.allclose(
        data_full.measured_patches, data_chunk.measured_patches, atol=1e-10,
    )


def test_inverter_chunked_parity():
    """Inverter recovery identical with and without chunking."""
    G_cart, th, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)
    sc = small_scan_config(sample_size_um=10.0, n_scans=7, beam_size_um=4.0)
    model = build_model(sc, hkls_int, G_cart, th)

    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        eps_gradient_voigt=0, eps_gradient_amp=2e-3, eps_gradient_dir="x",
    )
    data = simulate_grain_patches(
        plant, model, patch_F=5, patch_P=15,
        sigma_yz=1.0, sigma_f=0.6, gate_tau_um=0.5,
    )

    kw = dict(
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=torch.zeros_like(plant.eps_voxel),
        lattice_init=plant.lattice,
        identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
        optimizer="adam",
        inner_steps=20,
        lr_aa=1e-4, lr_eps=1e-3, lr_lat=1e-5,
    )
    fit_full = fit_grain_peakshape(data, model, chunk_size_g=None, **kw)
    fit_chunk = fit_grain_peakshape(data, model, chunk_size_g=4, **kw)

    # The first-step loss must match (same forward, same gradient).
    assert abs(fit_full.losses[0] - fit_chunk.losses[0]) < 1e-10, (
        f"step-0 loss diverged: full={fit_full.losses[0]:.6e} "
        f"vs chunk={fit_chunk.losses[0]:.6e}"
    )
    # Final tensors agree to a generous fp64 tolerance — Adam's gradient-
    # squared accumulator amplifies machine-eps gradient differences across
    # 20 steps; we accept a small drift but require recovery quality is
    # equivalent.
    # Forward+single-step backward agree to ~machine eps (5e-11 in our
    # diagnostic). Multi-step Adam amplifies this through the v-sqrt
    # update-rule normalization; we accept a small drift across 20 steps
    # while requiring recovery quality is equivalent — both should land
    # within the same order of magnitude relative to the planted ε
    # gradient amplitude.
    eps_diff = (fit_full.eps_fit - fit_chunk.eps_fit).abs().max().item()
    R_diff = (fit_full.R_fit - fit_chunk.R_fit).abs().max().item()
    loss_rel = abs(fit_full.losses[-1] - fit_chunk.losses[-1]) \
                / max(abs(fit_full.losses[-1]), 1e-15)
    assert eps_diff < 1e-3, f"ε diverged: max abs = {eps_diff:.3e}"
    assert R_diff < 1e-3, f"R diverged: max abs = {R_diff:.3e}"
    # The recovery quantities (ε, R) above are the physically meaningful gate.
    #
    # The final-loss SCALAR is not a gate at all -- it is a chaotic Adam
    # trajectory. The v-sqrt update rule amplifies machine-eps forward
    # differences across 20 steps, and MEASURED by sweeping the planted strain
    # amplitude over 1.6e-3 .. 3.0e-3 (a physically irrelevant knob), the
    # relative drift swings over three orders of magnitude and reaches 32% at
    # some amplitudes -- on BOTH sides of the 2026-08-28 omega-solver change:
    #
    #     eps_gradient_amp    1.6e-3  1.8e-3  2.0e-3  2.2e-3  2.4e-3
    #     loss_rel (before)   6.0e-2  3.2e-1  6.1e-2  3.2e-1  2.7e-2
    #     loss_rel (after)    4.2e-3  7.7e-7  2.8e-1  8.7e-4  2.5e-3
    #
    # The old 10% gate therefore only ever passed because this test hard-codes
    # amp = 2.0e-3; it would have failed at 1.8e-3 or 2.2e-3 with the code it
    # was written against. Do not re-tighten it -- that pins a lottery draw,
    # not a property. This bound exists only to catch a gross blow-up.
    # Reproduce with autograd_audit_probes/pf_odf_chunk_parity_chaos.py.
    assert loss_rel < 0.5, f"final loss diverged grossly: rel = {loss_rel:.3e}"
