"""Regression: axis_angle_to_matrix gradient at axis_angle = 0.

The current ``midas_grain_odf.odf.axis_angle_to_matrix`` uses
``torch.where(near_zero, I, R)`` which kills the gradient at zero.
A sibling package (midas_pf_odf) hit this when initializing
per-voxel orientation perturbations at zero — Adam and L-BFGS saw
exactly zero gradient on the perturbation parameters and never moved.

This test pins the expected behavior of a smooth Rodrigues so that if
the implementation is later replaced (with the smooth ratio-form used
in midas_pf_odf.inversion._aa_to_R), the assertion passes silently.
On the current master implementation it is expected to FAIL — marked
as xfail(strict=True) with a documented reason.

Decision (2026-04-30 audit, see commit message): the bug is dormant
in production grain-odf paths — ParticleODF/BinghamMixtureODF inits
draw axis-angle magnitudes from continuous distributions that have
measure-zero probability of landing within the ``10*eps = 10^-8``
threshold, and VoxelGridODF stores its axis-angle grid as a buffer
(not a parameter), so the dead branch is never on the autograd path
for trained parameters in current usage. The test is here to lock
the bug down so it cannot wake up undetected.
"""
from __future__ import annotations

import pytest
import torch

from midas_grain_odf.odf import axis_angle_to_matrix


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Dormant bug: torch.where(near_zero, I, R) zeroes the gradient at "
        "axis_angle=0. Audit 2026-04-30 found no production grain-odf path "
        "currently exercises this branch (random inits never land within "
        "10*eps = 1e-8). Fix deferred until after the Nat Comm submission "
        "to avoid churn; this test is the canary."
    ),
)
def test_axis_angle_grad_at_zero():
    """At axis_angle = 0, R = identity, and the differential of R wrt
    axis_angle is the linearized Rodrigues formula:
        dR / d(aa_k) = K_k  (the basic skew-symmetric generator)
    Specifically dR[0,1] / d(aa[2]) = -1.
    """
    aa = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    R = axis_angle_to_matrix(aa)
    # At zero, R should be identity (up to round-off).
    assert torch.allclose(
        R.detach(),
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
    )
    R[0, 1].backward()
    assert aa.grad is not None, "gradient must be defined"
    assert torch.allclose(
        aa.grad,
        torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
        atol=1e-9,
    ), f"expected aa.grad = [0, 0, -1], got {aa.grad.tolist()}"


def test_axis_angle_grad_just_above_zero():
    """Gradient at small-but-nonzero axis_angle should already be
    well-defined (this is the ``norm > 10*eps`` branch in master).
    Sanity check that this path still works under any future fix.
    """
    aa = torch.tensor(
        [0.0, 0.0, 1e-3], dtype=torch.float64, requires_grad=True
    )
    R = axis_angle_to_matrix(aa)
    R[0, 1].backward()
    assert aa.grad is not None
    # For small θ, dR[0,1]/d(aa[2]) ≈ -cos(θ) ≈ -1
    assert torch.allclose(
        aa.grad,
        torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
        atol=1e-5,
    ), f"expected aa.grad ≈ [0, 0, -1], got {aa.grad.tolist()}"
