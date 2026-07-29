"""Roadmap #2 (independent oracle) + #1 (physics-model coupling) tests."""
import pytest
import torch

from midas_dfxm import (
    GoniometerSetting,
    aligned_resolution,
    make_uniform_field,
    reference_q_nom,
    rocking_scan,
    voxel_intensity,
    with_orientation_gradient,
    with_screw_dislocation,
)
from midas_dfxm.dislocation import (
    cubic_stiffness,
    dislocation_deformation_field,
    stroh_dislocation,
)
from midas_dfxm.physics import fit_gnd_density, fit_wall_spacing, gnd_curvature_field
from midas_dfxm.validate import cross_check_voxel_intensity, voxel_intensity_numpy

DT = torch.float64


# --------------------------------------------------------------------------
# #2 independent numpy oracle
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_numpy_oracle_matches_torch_forward():
    field = make_uniform_field(shape=(10, 10, 1), dtype=DT)
    field = with_screw_dislocation(field, burgers_A=2.556, core_radius_um=0.5)
    setting = GoniometerSetting(chi=0.03, phi=-0.01)
    q_nom = reference_q_nom(field, (1, 1, 1), GoniometerSetting())
    res = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=8e-3)
    out = cross_check_voxel_intensity(field, (1, 1, 1), setting, res)
    assert out["max_abs_diff"] < 1e-12


@pytest.mark.unit
def test_numpy_oracle_on_curved_crystal():
    field = make_uniform_field(shape=(16, 4, 1), dtype=DT)
    field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
    setting = GoniometerSetting(chi=0.05)
    q_nom = reference_q_nom(field, (2, 0, 0), GoniometerSetting())
    res = aligned_resolution(q_nom, sigma_par=1e-2, sigma_perp=5e-3)
    assert cross_check_voxel_intensity(field, (2, 0, 0), setting, res)["max_abs_diff"] < 1e-12


# --------------------------------------------------------------------------
# #1 physics-model coupling: GND density through the DFXM forward
# --------------------------------------------------------------------------
def _line_grid(n=40, half=10.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    return torch.stack([xs, torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)], dim=-1)


@pytest.mark.unit
def test_gnd_curvature_matches_nye_relation():
    # kappa = rho * b : the planted curvature equals the Nye prediction.
    pts = _line_grid()
    rho = 3.0
    field = gnd_curvature_field(rho, pts, curvature_axis=(0, 0, 1), along=0,
                                burgers_length_A=2.556)
    R, U = field.local_rotation_stretch()

    def z_angle(Rm):
        import math
        return math.degrees(math.atan2(Rm[1, 0].item(), Rm[0, 0].item()))
    span_deg = z_angle(R[-1]) - z_angle(R[0])
    b_um = 2.556e-4
    expect_deg = rho * b_um * 20.0 * 57.29577951308232  # kappa * length * deg/rad
    assert span_deg == pytest.approx(expect_deg, rel=1e-4)


@pytest.mark.autograd
def test_gnd_field_differentiable_in_density():
    pts = _line_grid(12)
    rho = torch.tensor(2.0, dtype=DT, requires_grad=True)
    field = gnd_curvature_field(rho, pts)
    field.F.sum().backward()
    assert rho.grad is not None and torch.isfinite(rho.grad) and rho.grad.abs() > 0


@pytest.mark.slow
def test_fit_gnd_density_recovers_planted():
    pts = _line_grid(40)
    rho_true = 2.5
    field = gnd_curvature_field(rho_true, pts, curvature_axis=(0, 0, 1), along=0)
    hkl, center = (1, 1, 1), GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=4e-3)
    b_um = 2.556e-4
    span_deg = rho_true * b_um * 20.0 * 57.29577951308232
    settings = rocking_scan(center, axis="chi", span=(-span_deg, span_deg), n=15)
    obs = torch.stack([voxel_intensity(field, hkl, s, res) for s in settings], dim=0)
    out = fit_gnd_density(obs, settings, pts, hkl, rho_init=1.0,
                          sigma_par=8e-3, sigma_perp=4e-3, steps=300, lr=0.1)
    assert out["rho"] == pytest.approx(rho_true, rel=0.05)


@pytest.mark.slow
def test_fit_wall_spacing_recovers_dislocation_density():
    # Discrete-dislocation DDD parameter: recover the tilt-boundary spacing D.
    CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
    PLANE, B = (1, 1, 1), (1, -1, 0)
    line = torch.linalg.cross(torch.tensor(PLANE, dtype=DT), torch.tensor(B, dtype=DT))
    xs = torch.linspace(-10, 10, 40, dtype=DT)
    pts = torch.stack([torch.zeros(40, dtype=DT), xs, torch.zeros(40, dtype=DT)], dim=-1)

    def builder(D):
        n = 5
        c0 = (n - 1) / 2.0
        out = []
        for k in range(n):
            pos = torch.stack([torch.zeros((), dtype=DT), (k - c0) * D, torch.zeros((), dtype=DT)])
            out.append(stroh_dislocation(CU, burgers=B, slip_normal=PLANE, line=line,
                                         core_position=pos, core_radius_um=0.5))
        return out

    D_true = torch.tensor(3.0, dtype=DT)
    from midas_dfxm import reference_q_nom as _rqn, aligned_resolution as _ar
    f0 = dislocation_deformation_field(pts, builder(D_true))
    q = _rqn(f0, (2, -2, 0), GoniometerSetting())
    res = _ar(q, sigma_par=8e-3, sigma_perp=6e-3)
    settings = rocking_scan(GoniometerSetting(), axis="chi", span=(-0.15, 0.15), n=21)
    obs = torch.stack([voxel_intensity(f0, (2, -2, 0), s, res) for s in settings], dim=0)  # (S,N)
    out = fit_wall_spacing(obs, settings, pts, (2, -2, 0), builder,
                           resolution=res, spacing_bounds=(1.5, 6.0), n_coarse=15,
                           steps=150, lr=0.05)
    assert out["spacing"] == pytest.approx(3.0, rel=0.1)
