"""Roadmap #3 tests: dense generators + external-field adapters."""
import pytest
import torch

from midas_dfxm.dislocation import cubic_stiffness, dislocation_deformation_field
from midas_dfxm.generators import (
    dislocation_pileup,
    ensemble_density_per_um2,
    field_from_deformation_gradient,
    field_from_strain,
    random_dislocation_ensemble,
)
from midas_dfxm.inverse import normal_strain

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)


def _grid(n=10, half=8.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


# --------------------------------------------------------------------------
# external-field adapters (external-field drop-in)
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_field_from_F_roundtrips():
    pts = _grid(6)
    F = torch.eye(3, dtype=DT).expand(pts.shape[0], 3, 3).clone()
    field = field_from_deformation_gradient(F, pts)
    assert field.n_voxels == pts.shape[0]
    assert torch.allclose(field.F, F)


@pytest.mark.unit
def test_field_from_strain_matches_normal_strain():
    pts = _grid(5)
    n = pts.shape[0]
    e = torch.zeros(n, 6, dtype=DT)
    e[:, 0] = 1e-3          # eps11 along x
    field = field_from_strain(e, pts)
    # normal strain along [1,0,0] must equal the planted eps11.
    ns = normal_strain(field, (1, 0, 0))
    assert torch.allclose(ns, torch.full((n,), 1e-3, dtype=DT), atol=1e-9)


@pytest.mark.autograd
def test_field_from_strain_differentiable():
    pts = _grid(4)
    e = torch.zeros(pts.shape[0], 6, dtype=DT, requires_grad=True)
    field = field_from_strain(e, pts)
    normal_strain(field, (1, 1, 0)).sum().backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()


# --------------------------------------------------------------------------
# dense structures
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_pileup_spacing_grows():
    pile = dislocation_pileup(CU, burgers=(1, -1, 0), slip_normal=(1, 1, 1),
                              n=5, first_spacing_um=1.0, growth=1.5)
    ys = [float(d.core_position[1]) for d in pile]
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    assert all(gaps[i + 1] > gaps[i] for i in range(len(gaps) - 1))


@pytest.mark.unit
def test_random_ensemble_deterministic_and_finite():
    bbox = ((-8.0, 8.0), (-8.0, 8.0))
    e1 = random_dislocation_ensemble(CU, bbox_um=bbox, n=15, seed=3)
    e2 = random_dislocation_ensemble(CU, bbox_um=bbox, n=15, seed=3)
    assert len(e1) == 15
    # deterministic in the seed
    assert torch.allclose(e1[0].core_position, e2[0].core_position)
    # builds a finite field
    pts = _grid(12)
    field = dislocation_deformation_field(pts, e1)
    assert torch.isfinite(field.F).all()
    dens = ensemble_density_per_um2(e1, bbox)
    assert dens == pytest.approx(15 / (16 * 16))


@pytest.mark.unit
def test_random_ensemble_density_scales():
    bbox = ((-10.0, 10.0), (-10.0, 10.0))
    low = random_dislocation_ensemble(CU, bbox_um=bbox, n=5, seed=1)
    high = random_dislocation_ensemble(CU, bbox_um=bbox, n=40, seed=1)
    assert ensemble_density_per_um2(high, bbox) > ensemble_density_per_um2(low, bbox)
