"""The forward distortion must INVERT the pipeline's correction: a predicted spot, run back through
midas_calibrate.pixel_to_REta_torch with the same coefficients, lands on the ideal ring radius.
Before 2026-10-13 the forward applied the correction itself (a 20-ID Varex calibration: 0.11 px median, 0.94 px max error,
twice the distortion)."""
import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

pytest.importorskip("midas_calibrate")
from midas_calibrate.geometry_torch import pixel_to_REta_torch  # noqa: E402
from midas_distortion import v2_coeffs_from_named  # noqa: E402

DT = torch.float64
LSD, YBC, ZBC, PX, N, RHOD = 998809.737, 1430.51, 1342.48, 150.0, 2880, 316753.95
TILT = (-0.0285, -0.4629, -0.2221)
PV1 = {"p0": 3.166e-4, "p1": 1.188e-4, "p2": 3.843e-4, "p3": 39.96}


def test_forward_distortion_inverts_the_pipeline_correction():
    g = HEDMGeometry(Lsd=LSD, y_BC=YBC, z_BC=ZBC, px=PX, omega_start=-180.0, omega_step=0.25, n_frames=1440,
                     n_pixels_y=N, n_pixels_z=N, min_eta=6.0, wavelength=0.173, tx=TILT[0], ty=TILT[1], tz=TILT[2],
                     flip_y=True, apply_tilts=True, apply_distortion=True,
                     p_distortion=[float(x) for x in v2_coeffs_from_named(PV1)], rho_d=RHOD)
    h = torch.eye(3, dtype=DT)
    m = HEDMForwardModel(hkls=h, thetas=torch.full((3,), 0.05, dtype=DT), geometry=g, hkls_int=h).to(DT)
    rng = np.random.default_rng(0); n = 300
    tt = torch.tensor(rng.uniform(0.06, 0.21, n), dtype=DT).reshape(1, 1, -1)
    eta = torch.tensor(rng.uniform(-np.pi, np.pi, n), dtype=DT).reshape(1, 1, -1)
    sp = m.project_to_detector(torch.zeros_like(eta), eta, tt, torch.zeros(1, 1, 3, dtype=DT), torch.ones_like(eta))
    yp, zp = sp.y_pixel.reshape(-1, n)[0], sp.z_pixel.reshape(-1, n)[0]
    p15 = torch.zeros(15, dtype=DT); p15[:4] = torch.tensor([PV1[f"p{i}"] for i in range(4)], dtype=DT)
    t = lambda v: torch.tensor(v, dtype=DT)
    Rc, e = pixel_to_REta_torch(yp, zp, Lsd=t(LSD), BC_y=t(YBC), BC_z=t(ZBC), tx=t(TILT[0]), ty=t(TILT[1]),
                                tz=t(TILT[2]), p_coeffs=p15, parallax=t(0.0), px=t(PX), rho_d=t(RHOD))
    dR_px = (Rc * PX - LSD * torch.tan(tt.reshape(-1))) / PX
    assert float(dR_px.abs().max()) < 1e-3
