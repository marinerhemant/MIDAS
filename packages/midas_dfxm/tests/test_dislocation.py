"""Phase 4 tests: per-dislocation Stroh forward + defect typing."""
import math

import pytest
import torch

from midas_dfxm.dislocation import (
    cubic_stiffness,
    dislocation_deformation_field,
    dislocation_dipole,
    edge_dislocation_wall,
    stroh_dislocation,
)
from midas_dfxm.typing import (
    classify_character,
    dilatation_ratio,
    g_dot_b,
    recover_burgers,
    visibility_series,
)

DT = torch.float64
# Copper (GPa), Zener ~3.2 (strongly anisotropic -> Stroh well-conditioned).
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
# FCC {111}<110>: an in-plane Burgers on the (111) plane.
PLANE = (1, 1, 1)
B_LINE = (1, -1, 0)  # perpendicular to (111)


def _ring(radius, n=64, dtype=DT):
    th = torch.linspace(0, 2 * math.pi, n + 1, dtype=dtype)[:-1]
    return torch.stack([radius * torch.cos(th), radius * torch.sin(th),
                        torch.zeros_like(th)], dim=-1)


# --------------------------------------------------------------------------
# displacement-gradient field physics
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_distortion_decays_as_one_over_r():
    d = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="edge",
                          core_radius_um=1e-3)
    b1 = d.displacement_gradient(_ring(1.0))
    b2 = d.displacement_gradient(_ring(2.0))
    ratio = b1.abs().mean() / b2.abs().mean()
    assert ratio.item() == pytest.approx(2.0, rel=0.02)


@pytest.mark.unit
def test_screw_is_pure_shear_edge_has_dilatation():
    screw = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="screw",
                              core_radius_um=1e-3)
    edge = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="edge",
                             core_radius_um=1e-3)
    pts = _ring(1.0)
    r_screw = float(dilatation_ratio(screw, pts))
    r_edge = float(dilatation_ratio(edge, pts))
    assert r_screw < 1e-6            # pure shear
    assert r_edge > 0.1              # tension/compression dipole
    assert classify_character(screw, pts) == "screw"
    assert classify_character(edge, pts) == "edge"


@pytest.mark.unit
def test_deformation_gradient_is_identity_plus_beta():
    d = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="edge")
    pts = _ring(5.0)
    F = d.deformation_gradient(pts)
    beta = d.displacement_gradient(pts)
    eye = torch.eye(3, dtype=DT)
    assert torch.allclose(F, eye + beta, atol=1e-12)
    # Far from the core the distortion is small -> F ~ I.
    assert (F - eye).abs().max() < 1e-2


@pytest.mark.unit
def test_core_cutoff_keeps_field_finite_at_origin():
    d = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="edge",
                          core_radius_um=0.1)
    origin = torch.zeros(1, 3, dtype=DT)
    beta = d.displacement_gradient(origin)
    assert torch.isfinite(beta).all()
    assert beta.abs().max() < 1.0


# --------------------------------------------------------------------------
# g.b invisibility
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_g_dot_b_integer():
    assert g_dot_b((2, 2, 0), (1, -1, 0)) == pytest.approx(0.0)
    assert g_dot_b((2, 0, 0), (1, -1, 0)) == pytest.approx(2.0)


@pytest.mark.unit
def test_screw_invisible_when_g_dot_b_zero():
    # Screw with b=[1,-1,0]: reflections with g.b=0 should extinguish.
    screw = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="screw",
                              core_radius_um=0.2)
    pts = torch.stack(torch.meshgrid(
        torch.linspace(-8, 8, 24, dtype=DT),
        torch.linspace(-8, 8, 24, dtype=DT),
        torch.zeros(1, dtype=DT), indexing="ij"), dim=-1).reshape(-1, 3)
    # b = [1,-1,0]:  g.b = 0 -> invisible;  g.b != 0 -> visible.
    #   (2,2,0).b=0  (0,0,2).b=0  (1,1,1).b=0     [invisible]
    #   (2,-2,0).b=4 (2,0,0).b=2  (0,2,0).b=-2    [visible]
    refl = [(2, 2, 0), (0, 0, 2), (1, 1, 1),
            (2, -2, 0), (2, 0, 0), (0, 2, 0)]
    contrasts = visibility_series(screw, pts, refl, rocking_offset_deg=0.05)
    invisible = [contrasts[r] for r in [(2, 2, 0), (0, 0, 2), (1, 1, 1)]]
    visible = [contrasts[r] for r in [(2, -2, 0), (2, 0, 0), (0, 2, 0)]]
    assert max(invisible) < 0.01 * min(visible)  # extinction is essentially total


@pytest.mark.unit
def test_recover_burgers_from_visibility():
    screw = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="screw",
                              core_radius_um=0.2)
    pts = torch.stack(torch.meshgrid(
        torch.linspace(-8, 8, 20, dtype=DT),
        torch.linspace(-8, 8, 20, dtype=DT),
        torch.zeros(1, dtype=DT), indexing="ij"), dim=-1).reshape(-1, 3)
    refl = [(2, 2, 0), (2, -2, 0), (0, 0, 2), (2, 0, 0), (0, 2, 0), (1, 1, 1), (0, 2, 2)]
    contrasts = visibility_series(screw, pts, refl, mode="intrinsic")
    candidates = [(1, -1, 0), (1, 1, 0), (1, 0, -1), (0, 1, -1)]
    ranked = recover_burgers(contrasts, candidates)
    assert ranked[0][0] == (1, -1, 0)   # true Burgers wins (fewest mismatches)
    assert ranked[0][1] == 0


# --------------------------------------------------------------------------
# multi-dislocation structures
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_tilt_boundary_builds_and_superposes():
    wall = edge_dislocation_wall(CU, burgers=B_LINE, slip_normal=PLANE,
                                 n_dislocations=5, spacing_um=2.0)
    assert len(wall) == 5
    pts = torch.stack(torch.meshgrid(
        torch.linspace(-6, 6, 12, dtype=DT),
        torch.linspace(-6, 6, 12, dtype=DT),
        torch.zeros(1, dtype=DT), indexing="ij"), dim=-1).reshape(-1, 3)
    field = dislocation_deformation_field(pts, wall)
    assert field.F.shape == (pts.shape[0], 3, 3)
    assert torch.isfinite(field.F).all()


@pytest.mark.unit
def test_dipole_far_field_cancels():
    dip = dislocation_dipole(CU, burgers=B_LINE, slip_normal=PLANE, separation_um=2.0)
    single = stroh_dislocation(CU, burgers=B_LINE, slip_normal=PLANE, character="edge")
    # Screening improves with distance (dipole far-field ~ separation/r vs 1/r).
    near = torch.tensor([[40.0, 40.0, 0.0]], dtype=DT)
    far = torch.tensor([[160.0, 160.0, 0.0]], dtype=DT)
    def ratio(pt):
        bd = (dislocation_deformation_field(pt, dip).F - torch.eye(3, dtype=DT)).abs().max()
        bo = single.displacement_gradient(pt).abs().max()
        return (bd / bo).item()
    r_near, r_far = ratio(near), ratio(far)
    assert r_far < r_near          # dipole screens ever better with distance
    assert r_far < 0.1             # and is strongly screened in the far field


# --------------------------------------------------------------------------
# differentiability + device
# --------------------------------------------------------------------------
@pytest.mark.autograd
def test_distortion_differentiable_in_stiffness():
    c11 = torch.tensor(168.4, dtype=DT, requires_grad=True)
    c12 = torch.tensor(121.4, dtype=DT, requires_grad=True)
    c44 = torch.tensor(75.4, dtype=DT, requires_grad=True)
    C = cubic_stiffness(c11, c12, c44, dtype=DT)
    d = stroh_dislocation(C, burgers=B_LINE, slip_normal=PLANE, character="edge")
    beta = d.displacement_gradient(_ring(3.0))
    beta.abs().sum().backward()
    for leaf in (c11, c12, c44):
        assert leaf.grad is not None and torch.isfinite(leaf.grad)
    assert (c11.grad.abs() + c12.grad.abs() + c44.grad.abs()) > 0


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dislocation_device_portable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    # NOTE: Stroh uses torch.linalg.eig (complex) -> CPU/CUDA only, not MPS.
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT).to(device)
    d = stroh_dislocation(C, burgers=B_LINE, slip_normal=PLANE, character="edge")
    pts = _ring(2.0).to(device)
    beta = d.displacement_gradient(pts)
    assert beta.device.type == device
    assert torch.isfinite(beta).all()
