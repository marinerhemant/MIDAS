"""An FF model with detector tilts must invert the pipeline's own pixel -> lab
conversion (midas_calibrate.geometry_torch.pixel_to_REta_torch, which is what
calibration fits tx/ty/tz in and what transforms applies to every spot).

Before 2026-10-07 the tilt convention was chosen by ``multi_mode == "panel"``,
so a single-detector FF model (``flip_y=True``, ``apply_tilts=True``, default
``multi_mode="layered"``) silently used the NF ray-plane tilt: tx with the
opposite sign and a ty-dependent radial error (154 ue at ty -0.46 deg).
"""
import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

pytest.importorskip("midas_calibrate")
from midas_calibrate.geometry_torch import pixel_to_REta_torch  # noqa: E402

DT = torch.float64
LSD, YBC, ZBC, PX, N = 998809.737, 1430.51, 1342.48, 150.0, 2880


def _model(tx, ty, tz, mode):
    g = HEDMGeometry(Lsd=LSD, y_BC=YBC, z_BC=ZBC, px=PX, omega_start=-180.0, omega_step=0.25,
                     n_frames=1440, n_pixels_y=N, n_pixels_z=N, min_eta=6.0, wavelength=0.173,
                     tx=tx, ty=ty, tz=tz, flip_y=True, apply_tilts=True, multi_mode=mode)
    h = torch.eye(3, dtype=DT)
    return HEDMForwardModel(hkls=h, thetas=torch.full((3,), 0.05, dtype=DT), geometry=g, hkls_int=h).to(DT)


@pytest.mark.parametrize("tilts", [(-0.0285, -0.4629, -0.2221), (-0.5, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, -0.6)])
def test_ff_tilt_inverts_the_pipeline_pixel_to_lab(tilts):
    rng = np.random.default_rng(0)
    R = rng.uniform(60e3, 210e3, 300)
    eta = np.radians(rng.uniform(-180, 180, 300))
    y0 = torch.tensor(-R * np.sin(eta), dtype=DT)
    z0 = torch.tensor(R * np.cos(eta), dtype=DT)
    m = _model(*tilts, "layered")
    yd, zd = m._apply_ff_panel_tilt(y0, z0, torch.tensor(LSD, dtype=DT), d_idx=0)
    ypx, zpx = YBC - yd / PX, ZBC + zd / PX
    Rc, e = pixel_to_REta_torch(
        ypx, zpx, Lsd=torch.tensor(LSD, dtype=DT), BC_y=torch.tensor(YBC, dtype=DT),
        BC_z=torch.tensor(ZBC, dtype=DT), tx=torch.tensor(tilts[0], dtype=DT),
        ty=torch.tensor(tilts[1], dtype=DT), tz=torch.tensor(tilts[2], dtype=DT),
        p_coeffs=torch.zeros(15, dtype=DT), parallax=torch.zeros((), dtype=DT),
        px=torch.tensor(PX, dtype=DT), rho_d=torch.tensor(2000.0, dtype=DT))
    yb = -Rc * PX * torch.sin(torch.deg2rad(e))
    zb = Rc * PX * torch.cos(torch.deg2rad(e))
    assert float(torch.hypot(yb - y0, zb - z0).max()) < 0.01   # um


def test_dispatch_uses_ff_tilt_for_single_detector_ff():
    """project_to_detector itself: a single-detector FF model (default multi_mode
    "layered") must place spots exactly where the FF panel convention does."""
    m = _model(-0.0285, -0.4629, -0.2221, "layered")
    ref = _model(-0.0285, -0.4629, -0.2221, "panel")
    om = torch.tensor([[[0.3]]], dtype=DT)
    eta = torch.tensor([[[0.7]]], dtype=DT)
    tt = torch.tensor([[[0.12]]], dtype=DT)
    pos = torch.zeros(1, 1, 3, dtype=DT)
    v = torch.ones(1, 1, 1, dtype=DT)
    a = m.project_to_detector(om, eta, tt, pos, v)
    b = ref.project_to_detector(om, eta, tt, pos, v)
    assert torch.allclose(a.y_pixel, b.y_pixel, atol=1e-9)
    assert torch.allclose(a.z_pixel, b.z_pixel, atol=1e-9)
