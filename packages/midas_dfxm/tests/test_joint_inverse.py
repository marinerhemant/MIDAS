"""Joint one-model inverse: F as physics-regularized intermediate (core + amplitude fit)."""
import torch
import pytest

from midas_dfxm import cubic_stiffness, dislocation_deformation_field, stroh_dislocation
from midas_dfxm.field_inverse import deformation_observable
from midas_dfxm.joint_inverse import fit_dislocation_field

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
PLANE, BURGERS = (1, 1, 1), (1, -1, 0)
REFL = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3), (3, 1, 1), (1, 3, 1)]


def _setup(core=(1.2, -0.8, 0.0), n=25, half=6.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    disl = stroh_dislocation(CU, burgers=BURGERS, slip_normal=PLANE, character="edge",
                             core_position=core, core_radius_um=0.3)
    fld = dislocation_deformation_field(pts, disl, shape=(n, n, 1))
    return pts, fld, (n, n, 1)


@pytest.mark.slow
def test_recovers_core_and_amplitude_under_noise():
    torch.manual_seed(0)
    true_core = torch.tensor([1.2, -0.8, 0.0], dtype=DT)
    pts, fld, shape = _setup(tuple(true_core.tolist()))
    meas = deformation_observable(fld, REFL)
    meas = meas + 0.1 * meas.abs().mean() * torch.randn_like(meas)
    init = pts[meas.abs().sum(-1).argmax()]                    # coarse-localize
    out = fit_dislocation_field(meas, REFL, CU, burgers=BURGERS, slip_normal=PLANE,
                                character="edge", positions=pts, init_core=init.tolist(),
                                steps=800, lr=2e-1, shape=shape)
    dx = float(pts[1, 1] - pts[0, 1])                          # grid spacing
    assert float((out["core"] - true_core).norm()) < 0.3 * dx  # sub-grid core
    assert abs(out["amp"] - 1.0) < 0.05                        # Burgers magnitude


@pytest.mark.slow
def test_physics_intermediate_beats_free_under_noise():
    # FAIR baseline: free-F best is plain direct LSQ (regularization over-smooths the sharp
    # core). Physics one-model still beats it comfortably (~20-50x), not the inflated ~1e4.
    from midas_dfxm.field_inverse import recover_deformation_direct
    torch.manual_seed(1)
    pts, fld, shape = _setup()
    F_true = fld.F
    meas = deformation_observable(fld, REFL)
    meas = meas + 0.1 * meas.abs().mean() * torch.randn_like(meas)
    F_free = recover_deformation_direct(meas, REFL)            # best-tuned free baseline
    init = pts[meas.abs().sum(-1).argmax()]
    out = fit_dislocation_field(meas, REFL, CU, burgers=BURGERS, slip_normal=PLANE, character="edge",
                                positions=pts, init_core=init.tolist(), steps=800, lr=2e-1, shape=shape)
    e_free = (F_free - F_true).pow(2).mean().sqrt()
    e_phys = (out["F"] - F_true).pow(2).mean().sqrt()
    assert e_phys < 0.2 * e_free                               # physics manifold >5x more robust (honest)


@pytest.mark.slow
def test_wrong_character_fits_worse():
    torch.manual_seed(2)
    pts, fld, shape = _setup()
    meas = deformation_observable(fld, REFL)
    meas = meas + 0.05 * meas.abs().mean() * torch.randn_like(meas)
    init = pts[meas.abs().sum(-1).argmax()]
    kw = dict(positions=pts, init_core=init.tolist(), steps=600, lr=2e-1, shape=shape,
              burgers=BURGERS, slip_normal=PLANE)
    edge = fit_dislocation_field(meas, REFL, CU, character="edge", **kw)
    screw = fit_dislocation_field(meas, REFL, CU, character="screw", **kw)
    assert screw["loss"] > 5 * edge["loss"]                   # correct model selected by fit quality


@pytest.mark.slow
def test_ensemble_recovers_multiple_dislocations():
    # multi-dislocation one-model: recovers cores + signed amplitudes and beats free-F,
    # GIVEN typing-seeded amplitude signs (the discrete sign is not gradient-recoverable).
    from midas_dfxm import (dislocation_deformation_field, DeformationField)
    from midas_dfxm.field_inverse import deformation_observable, recover_deformation_direct
    from midas_dfxm.joint_inverse import fit_dislocation_ensemble, signal_peaks
    import numpy as np
    torch.manual_seed(0)
    xs = torch.linspace(-8, 8, 41, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    d0 = stroh_dislocation(CU, burgers=BURGERS, slip_normal=PLANE, character="edge",
                           core_position=(0., 0., 0.), core_radius_um=0.3)
    true_cores = torch.tensor([[3.5, 2.0, 0.], [-3.0, 3.0, 0.], [-1.0, -4.0, 0.]], dtype=DT)
    true_amps = [1.0, -1.0, 1.0]
    eye = torch.eye(3, dtype=DT); F = eye.expand(pts.shape[0], 3, 3).clone()
    for c, a in zip(true_cores, true_amps):
        F = F + a * (dislocation_deformation_field(pts - c, d0).F - eye)
    fld = DeformationField(positions=pts, F=F, reference_orientation=eye,
                           lattice_params=torch.tensor([3.6156] * 3 + [90.] * 3, dtype=DT))
    meas = deformation_observable(fld, REFL)
    meas = meas + 0.05 * meas.abs().mean() * torch.randn_like(meas)
    init = signal_peaks(meas, pts, 3, min_sep_um=2.0)
    signs = [float(np.sign(true_amps[np.argmin(np.linalg.norm(true_cores.numpy() - p.numpy(), axis=1))]))
             for p in init]
    out = fit_dislocation_ensemble(meas, REFL, CU, [(BURGERS, PLANE, "edge")] * 3,
                                   positions=pts, init_cores=init, init_amps=signs, steps=1500, lr=8e-2)
    e_free = float((recover_deformation_direct(meas, REFL) - F).pow(2).mean().sqrt())
    e_ens = float((out["F"] - F).pow(2).mean().sqrt())
    assert e_ens < 0.5 * e_free                       # ensemble physics-model beats free-F
    # all three cores localized sub-grid
    dx = float(xs[1] - xs[0])
    for c in true_cores:
        assert min(float((rc - c).norm()) for rc in out["cores"]) < 0.5 * dx
