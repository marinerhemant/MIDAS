"""CP<->DFXM coupling tests: per-slip-system GND-density decomposition."""
import pytest
import torch

from midas_dfxm.dislocation import fcc_slip_systems
from midas_dfxm.plasticity import (
    gnd_identifiability,
    multislip_gnd_field,
    nye_from_densities,
    recover_gnd_densities,
    slip_dislocation_types,
)

DT = torch.float64


@pytest.mark.unit
def test_fcc_dislocation_types_geometry():
    labels, bh, xh = slip_dislocation_types(dtype=DT)
    assert len(labels) == 24  # 12 systems x {edge, screw}
    for (normal, burgers, ch), b, xi in zip(labels, bh, xh):
        if ch == "edge":
            assert abs(float(b @ xi)) < 1e-9        # edge line perpendicular to b
        else:
            assert abs(abs(float(b @ xi)) - 1.0) < 1e-9  # screw line parallel to b


@pytest.mark.unit
def test_full_catalog_is_underdetermined():
    # Honest identifiability: a 3x3 Nye tensor (9 numbers) cannot resolve 24 types.
    _, bh, xh = slip_dislocation_types(dtype=DT)
    info = gnd_identifiability(bh, xh)
    assert info["n_types"] == 24
    assert info["rank"] == 9
    assert info["null_dim"] == 15


@pytest.mark.unit
def test_small_candidate_set_recovers_exactly():
    # The realistic CP use: a few candidate active systems -> well-posed inverse.
    sys3 = fcc_slip_systems()[:3]
    labels, bh, xh = slip_dislocation_types(systems=sys3, characters=("edge",), dtype=DT)
    assert gnd_identifiability(bh, xh)["null_dim"] == 0
    rho_true = torch.tensor([2.0, 0.0, 1.0], dtype=DT)
    alpha = nye_from_densities(rho_true, bh, xh)
    rec = recover_gnd_densities(alpha, bh, xh, lambda_sparse=1e-3, steps=2000, lr=0.1)
    assert torch.allclose(rec, rho_true, atol=0.02)


@pytest.mark.unit
def test_full_catalog_reproduces_nye_tensor():
    # Even under-determined, the recovered densities reproduce the (identifiable) Nye
    # tensor -- the labels may be non-unique, the tensor is not.
    labels, bh, xh = slip_dislocation_types(dtype=DT)
    rho_true = torch.zeros(len(labels), dtype=DT)
    rho_true[5] = 3.0
    alpha = nye_from_densities(rho_true, bh, xh)
    rec = recover_gnd_densities(alpha, bh, xh, lambda_sparse=1e-2, steps=1500, lr=0.1)
    rel = (nye_from_densities(rec, bh, xh) - alpha).norm() / alpha.norm()
    assert float(rel) < 0.05


@pytest.mark.autograd
def test_nye_from_densities_differentiable():
    _, bh, xh = slip_dislocation_types(dtype=DT)
    rho = torch.zeros(24, dtype=DT, requires_grad=True)
    alpha = nye_from_densities(rho + 1.0, bh, xh)
    alpha.abs().sum().backward()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()


@pytest.mark.unit
def test_multislip_field_builds_and_is_differentiable():
    labels, bh, xh = slip_dislocation_types(dtype=DT)
    xs = torch.linspace(-5, 5, 12, dtype=DT)
    pts = torch.stack([xs, torch.zeros(12, dtype=DT), torch.zeros(12, dtype=DT)], dim=-1)
    rho = torch.zeros(24, dtype=DT, requires_grad=True)
    rho2 = rho + 0.0
    rho2 = rho2.clone()
    rho2.data[5] = 5.0
    field = multislip_gnd_field(rho2, pts, bh, xh)
    assert field.F.shape == (12, 3, 3)
    assert torch.isfinite(field.F).all()
