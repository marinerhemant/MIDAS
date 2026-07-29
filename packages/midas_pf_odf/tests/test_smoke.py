"""Smoke test: import paths + a tiny plant.

Catches packaging / import / scaffolding regressions in <1 second.
"""

import math
import torch

from midas_pf_odf import (
    SinglePhaseGrainPlant, plant_single_grain,
    simulate_grain_patches, fit_grain_peakshape,
    IdentifiabilityMode, recovery_metrics,
)
from midas_pf_odf.forward import joint_grain_forward, soft_beam_gate

from tests.conftest import (
    make_fcc_hkls, standard_pf_geometry, small_scan_config, build_model,
)


def test_imports_smoke():
    # Re-importing the public surface should not error.
    assert SinglePhaseGrainPlant is not None
    assert IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO == "project_eps_mean_zero"


def test_plant_smoke():
    plant = plant_single_grain(grid_shape=(4, 4), voxel_size_um=2.0)
    assert plant.n_voxels == 16
    assert plant.R_voxel.shape == (16, 3, 3)
    assert plant.eps_voxel.shape == (16, 6)
    # Default plant: no orientation gradient → all R == R_avg.
    R0 = plant.R_voxel[0]
    for v in range(plant.n_voxels):
        assert torch.allclose(plant.R_voxel[v], R0, atol=1e-12)


def test_rodrigues_gradient_at_zero():
    """Regression: ``midas_pf_odf.inversion._aa_to_R`` must have a
    non-zero gradient at axis_angle = 0. The old
    ``midas_grain_odf.odf.axis_angle_to_matrix`` failed this because it
    used ``torch.where(near_zero, I, R)`` which kills the gradient.
    Without this property the orientation refinement starting at δ=0
    is dead in the water — see Session 2 RESTART.md.
    """
    from midas_pf_odf.inversion import _aa_to_R

    aa = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    R = _aa_to_R(aa)
    # At zero, R = identity; gradient of e.g. R[0, 1] wrt aa[2] should
    # be -1 (the linear sin(θ)/θ → 1 limit on the K_z component).
    R[0, 1].backward()
    assert aa.grad is not None
    # The skew-symmetric K matrix at axis_angle = (0,0,1)·θ has K[0,1] = -1.
    # Gradient of R[0,1] = sin(θ)/θ * K[0,1] wrt aa[2], at θ=0 (where
    # sin(θ)/θ → 1), is -1.
    assert torch.allclose(aa.grad, torch.tensor([0.0, 0.0, -1.0],
                                                  dtype=torch.float64),
                          atol=1e-8), (
        f"Expected ∇R[0,1]/∇aa = (0, 0, -1) at zero; got {aa.grad.tolist()}"
    )


def test_soft_gate_basic():
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64)
    omega = torch.zeros(2, 3, dtype=torch.float64)             # at ω=0, y_rot = py
    beam_pos = torch.tensor([0.0, 5.0], dtype=torch.float64)
    gate = soft_beam_gate(pos, omega, beam_pos, beam_size_um=2.0, tau_um=0.1)
    assert gate.shape == (2, 3, 2)
    # Voxel 0 at y=0 is centered on beam 0 (pos 0): gate ≈ 1 at scan 0.
    assert gate[0, 0, 0].item() > 0.99
    # Voxel 0 at y=0 is far from beam 1 (pos 5, |diff| = 5 > beam/2): gate ≈ 0.
    assert gate[0, 0, 1].item() < 1e-3
