"""Native finite-beam forward: reduces to point, biases curved fields, beam-model mitigates."""
import torch
import pytest

from midas_dfxm import (cubic_stiffness, stroh_dislocation, dislocation_deformation_field,
                        DeformationField)
from midas_dfxm.field_inverse import deformation_observable
from midas_dfxm.finite_beam import beam_integrated_observable
from midas_invert.optimize import fit

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
PLANE, B = (1, 1, 1), (1, -1, 0)
REFL = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3), (3, 1, 1), (1, 3, 1)]
LATC = torch.tensor([3.6156] * 3 + [90.] * 3, dtype=DT)
RAY = (0., 0., 1.)


def _grid(n=21, half=4.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    return torch.stack([gx, gy, gz], -1).reshape(-1, 3)


@pytest.mark.unit
def test_reduces_to_point_at_zero_sigma():
    pts = _grid()
    disl = stroh_dislocation(CU, burgers=B, slip_normal=PLANE, character="edge",
                             core_position=(0.6, -0.4, 0.0), core_radius_um=0.3)
    ffn = lambda p: dislocation_deformation_field(p, disl)
    pt = deformation_observable(ffn(pts), REFL)
    b0 = beam_integrated_observable(ffn, pts, REFL, sigma_z=1e-6, ray_dir=RAY)
    assert float((b0 - pt).abs().max()) < 1e-6
    # a finite beam smears a curved field (nonzero difference)
    bz = beam_integrated_observable(ffn, pts, REFL, sigma_z=0.5, ray_dir=RAY, n_samples=31)
    assert float((bz - pt).norm() / pt.norm()) > 0.05


@pytest.mark.slow
def test_beam_model_removes_amplitude_bias():
    torch.manual_seed(0)
    pts = _grid()
    TC = torch.tensor([0.6, -0.4, 0.0], dtype=DT)
    disl = stroh_dislocation(CU, burgers=B, slip_normal=PLANE, character="edge",
                             core_position=TC, core_radius_um=0.3)
    meas = beam_integrated_observable(lambda p: dislocation_deformation_field(p, disl),
                                      pts, REFL, sigma_z=0.5, ray_dir=RAY, n_samples=31)
    disl0 = stroh_dislocation(CU, burgers=B, slip_normal=PLANE, character="edge",
                              core_position=(0., 0., 0.), core_radius_um=0.3)

    def predicted(core, amp, sigma_z):
        def field_fn(p):
            fld = dislocation_deformation_field(p - core, disl0)
            F = torch.eye(3, dtype=DT) + amp * (fld.F - torch.eye(3, dtype=DT))
            return DeformationField(positions=p, F=F, reference_orientation=torch.eye(3, dtype=DT),
                                    lattice_params=LATC)
        if sigma_z == 0:
            return deformation_observable(field_fn(pts), REFL)
        return beam_integrated_observable(field_fn, pts, REFL, sigma_z=sigma_z, ray_dir=RAY, n_samples=31)

    init = pts[meas.abs().sum(-1).argmax()]
    amps = {}
    for tag, sz in [("point", 0), ("beam", 0.5)]:
        core = init.clone().requires_grad_(True); amp = torch.tensor(1.0, dtype=DT, requires_grad=True)
        fit([core, amp], lambda: (predicted(core, amp, sz) - meas).pow(2).mean(), steps=600, lr=1e-1)
        amps[tag] = float(amp.detach())
    # beam-aware model recovers the amplitude (Burgers magnitude) far better than point-model
    assert abs(amps["beam"] - 1.0) < 0.5 * abs(amps["point"] - 1.0)
