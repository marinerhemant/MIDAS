"""Issue #11: the fused Triton kernel must rotate voxels about the wedge axis.

``fused_hard_frac`` carries its own inlined projection. With ``HAS_WEDGE``
the voxel must rotate like G, ``pos_lab = R_y(-W) R_z(omega) pos`` (MIDAS
Wedge convention, ``midas_diffract.forward`` module doc), and its lab z must
enter ``zdet`` -- the same as the eager
``HEDMForwardModel.project_to_detector`` (which
``packages/midas_diffract/tests/test_wedge_positions.py`` holds to an
independent rigid-body ray trace). This file holds the kernel to the eager
path with the wedge ON.

The geometry is chosen so the wedge moves spots by ~2 px (the about-z
rotation would give a different fraction -- checked in-test), and the Euler
set so that every eager (fp64) spot sits well clear of a pixel / frame
boundary, so the fp32 kernel cannot straddle an integer floor.

Run with (CUDA + Triton required, otherwise the module skips)::

    cd packages/midas_nf_fitorientation
    PYTHONPATH=../midas_diffract python -m pytest tests/test_triton_wedge.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.soft_overlap import forward_batched_grains

try:
    from midas_nf_fitorientation.triton_kernels import HAS_TRITON, fused_hard_frac
except Exception:                                        # pragma: no cover
    HAS_TRITON = False
    fused_hard_frac = None

_CUDA = torch.cuda.is_available()

LSD = [20_000.0, 22_000.0]
YBC = [256.0, 256.0]
ZBC = [256.0, 256.0]
PX = 20.0
OMEGA_START = -180.0
OMEGA_STEP = 10.0
N_FRAMES = 36
N_PIX = 512
MIN_ETA = 6.0
WAVELENGTH = 0.172979
LATTICE_A = 4.08
WEDGE_DEG = 3.0
D = len(LSD)
TOL = 1e-6          # kernel fp32 vs eager fp64
MARGIN = 2e-3       # min distance (px / frames) of any eager spot from a boundary

HKLS_INT = torch.tensor(
    [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]], dtype=torch.float64
)

EULERS = torch.tensor([
    [5.542, 1.079, 3.750],
    [4.745, 2.883, 4.283],
    [1.662, 2.804, 6.201],
    [0.749, 1.061, 3.756],
], dtype=torch.float64)

POSITIONS = torch.tensor([
    [650.0, -300.0, 0.0],
    [-500.0, 450.0, 0.0],
    [120.0, 700.0, 0.0],
    [-600.0, -550.0, 0.0],
], dtype=torch.float64)


def _model(device, wedge_deg=WEDGE_DEG) -> HEDMForwardModel:
    geom = HEDMGeometry(
        Lsd=list(LSD), y_BC=list(YBC), z_BC=list(ZBC), px=PX,
        omega_start=OMEGA_START, omega_step=OMEGA_STEP,
        n_frames=N_FRAMES, n_pixels_y=N_PIX, n_pixels_z=N_PIX,
        min_eta=MIN_ETA, wavelength=WAVELENGTH,
        flip_y=False, multi_mode="layered", wedge=wedge_deg,
    )
    B0 = torch.eye(3, dtype=torch.float64) / LATTICE_A
    hkls_cart = HKLS_INT @ B0.T
    thetas = torch.asin(torch.linalg.norm(hkls_cart, dim=-1) * WAVELENGTH / 2.0)
    return HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom,
        hkls_int=HKLS_INT, device=device,
    )


def _forward(model, eulers=EULERS, positions=POSITIONS):
    dev = model.hkls.device
    return forward_batched_grains(
        model,
        eulers.to(device=dev, dtype=torch.float64),
        positions.to(device=dev, dtype=torch.float64),
    )


def boundary_margin(model) -> float:
    """Smallest distance of any valid eager spot from a pixel/frame edge."""
    frame_nr, valid, y_pixel, z_pixel = _forward(model)
    sel = valid > 0.5
    vals = [frame_nr[sel]]
    for d in range(D):
        vals += [y_pixel[d][sel], z_pixel[d][sel]]
    v = torch.cat([t.reshape(-1) for t in vals]).double()
    frac = v - torch.floor(v)
    return float(torch.minimum(frac, 1.0 - frac).min())


def _obs_ones_with_holes(model, hole_mask, device) -> ObsVolume:
    frame_nr, valid, y_pixel, z_pixel = _forward(model)
    arr = np.ones((D, N_FRAMES, N_PIX, N_PIX), dtype=np.uint8)
    f = frame_nr.long()[hole_mask].cpu().numpy()
    y = y_pixel[0].long()[hole_mask].cpu().numpy()
    z = z_pixel[0].long()[hole_mask].cpu().numpy()
    arr[0, f, y, z] = 0
    return ObsVolume.from_dense_array(arr, device=device, packed=True)


def _every_nth_valid(valid, n):
    flat = valid.reshape(-1) > 0.5
    idx = torch.nonzero(flat, as_tuple=True)[0][::n]
    out = torch.zeros_like(flat)
    out[idx] = True
    return out.reshape(valid.shape)


def _eager_frac(model, obs):
    frame_nr, valid, y_pixel, z_pixel = _forward(model)
    return obs.hard_fraction(frame_nr, y_pixel, z_pixel, valid).double().cpu()


def _triton_frac(model, obs_packed):
    dev = model.hkls.device
    out = fused_hard_frac(
        EULERS.to(device=dev, dtype=torch.float32).contiguous(),
        POSITIONS.to(device=dev, dtype=torch.float32).contiguous(),
        model.hkls.contiguous().to(torch.float32),
        model.thetas.contiguous().to(torch.float32),
        torch.tensor(LSD, device=dev, dtype=torch.float32),
        torch.tensor(YBC, device=dev, dtype=torch.float32),
        torch.tensor(ZBC, device=dev, dtype=torch.float32),
        torch.zeros(D, 9, device=dev, dtype=torch.float32),
        obs_packed,
        px=PX,
        wedge_rad=WEDGE_DEG * math.pi / 180.0,
        omega_start_deg=OMEGA_START,
        omega_step_deg=OMEGA_STEP,
        min_eta_rad=MIN_ETA * math.pi / 180.0,
        n_frames=N_FRAMES, n_y=N_PIX, n_z=N_PIX,
        has_tilts=False, has_wedge=True,
    )
    return out.double().cpu()


def test_fixture_is_well_posed_on_cpu():
    """Runs everywhere: the fixture's spots are clear of every boundary, and
    the wedge moves enough spots for the parity test to be discriminating."""
    m = _model(torch.device("cpu"))
    assert boundary_margin(m) > MARGIN
    _fn, valid, yp, zp = _forward(m)
    m_old = _model(torch.device("cpu"))
    m_old._has_wedge = False          # about-z position rotation, G still wedged
    _fo, valid_o, yo, zo = _forward(m_old)
    sel = valid > 0.5
    moved = ((yp.floor() != yo.floor()) | (zp.floor() != zo.floor()))[:, sel]
    assert int(moved.any(dim=0).sum()) >= 5


@pytest.mark.skipif(not (_CUDA and HAS_TRITON),
                    reason=f"needs CUDA + Triton (cuda={_CUDA}, triton={HAS_TRITON})")
def test_triton_matches_eager_with_wedge():
    dev = torch.device("cuda")
    m = _model(dev)
    _fn, valid, _yp, _zp = _forward(m)
    holes = _every_nth_valid(valid, 3)
    obs = _obs_ones_with_holes(m, holes, dev)

    eager = _eager_frac(m, obs)
    trit = _triton_frac(m, obs.packed)
    assert torch.all(eager > 0.0) and torch.all(eager < 1.0), eager.tolist()
    torch.testing.assert_close(trit, eager, atol=TOL, rtol=0.0)

    # Discrimination: the about-z position rotation gives a different
    # fraction on this obs, so parity cannot hold by accident.
    m_old = _model(dev)
    m_old._has_wedge = False
    old = _eager_frac(m_old, obs)
    assert not torch.allclose(old, eager, atol=TOL, rtol=0.0)
