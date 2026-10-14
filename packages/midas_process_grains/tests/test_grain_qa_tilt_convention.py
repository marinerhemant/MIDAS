"""grain-QA (and physics mode) predict RAW pixels with the detector tilts
applied. Those pixels must invert through the pipeline's own pixel -> lab
conversion (midas_calibrate pixel_to_REta_torch), the convention calibration
fits tx/ty/tz in.

Before midas_diffract's fix (2026-10-07) an FF model with apply_tilts and the
default multi_mode used the NF ray-plane tilt: at real 20-ID Varex tilts the predicted
pixels were a median 277 um off, tx with the wrong sign.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("midas_calibrate")
from midas_calibrate.geometry_torch import pixel_to_REta_torch  # noqa: E402

from midas_process_grains.grain_qa import _geometry  # noqa: E402
from midas_diffract.forward import HEDMForwardModel  # noqa: E402

DT = torch.float64


def test_grain_qa_geometry_matches_pipeline_tilt_convention():
    b = SimpleNamespace(LsdFit=998809.737, OmegaStep=0.25, YBCFit=1430.51, ZBCFit=1342.48, px=150.0,
                        OmegaStart=-180.0, NrPixelsY=2880, NrPixelsZ=2880, Wavelength=0.173,
                        txFit=-0.0285, tyFit=-0.4629, tzFit=-0.2221, ExcludePoleAngle=6.0)
    geo = _geometry(SimpleNamespace(base=b))[0]
    h = torch.eye(3, dtype=DT)
    m = HEDMForwardModel(hkls=h, thetas=torch.full((3,), 0.05, dtype=DT), geometry=geo, hkls_int=h).to(DT)
    rng = np.random.default_rng(1)
    tt = torch.tensor(rng.uniform(0.06, 0.2, 200), dtype=DT).reshape(1, 1, -1)
    eta = torch.tensor(rng.uniform(-np.pi, np.pi, 200), dtype=DT).reshape(1, 1, -1)
    om = torch.zeros_like(eta)
    sp = m.project_to_detector(om, eta, tt, torch.zeros(1, 1, 3, dtype=DT), torch.ones_like(eta))
    t = lambda v: torch.tensor(v, dtype=DT)
    yp, zp = sp.y_pixel.reshape(-1, 200)[0], sp.z_pixel.reshape(-1, 200)[0]   # (1, 2 branches, M): rows identical here
    Rc, e = pixel_to_REta_torch(yp, zp, Lsd=t(b.LsdFit), BC_y=t(b.YBCFit),
                                BC_z=t(b.ZBCFit), tx=t(b.txFit), ty=t(b.tyFit), tz=t(b.tzFit),
                                p_coeffs=torch.zeros(15, dtype=DT), parallax=torch.zeros((), dtype=DT),
                                px=t(b.px), rho_d=t(2000.0))
    R_ideal = b.LsdFit * torch.tan(tt.reshape(-1))
    assert float((Rc * b.px - R_ideal).abs().max()) < 0.01                       # um
    d_eta = (e - torch.rad2deg(eta.reshape(-1)) + 180) % 360 - 180
    assert float(d_eta.abs().max()) < 1e-4                                       # deg (acos path ~3e-6)
