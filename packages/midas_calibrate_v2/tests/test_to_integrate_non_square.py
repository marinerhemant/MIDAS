"""spec_from_calibration_result: default RMax / RhoD for non-square pixels use the PHYSICAL farthest-corner distance."""
import math
from types import SimpleNamespace

import pytest

from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_result


def _result(pxy, pxz, NY=400, NZ=300, by=150.5, bz=120.25):
    return SimpleNamespace(NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxy, pxZ=pxz, Lsd=500000.0, BC_y=by, BC_z=bz, tx=0.0, ty=0.0, tz=0.0,
                           wavelength_A=0.18, distortion={}, im_trans=())


def _corner_um(r):
    return math.hypot(max(r.BC_y, r.NrPixelsY - 1 - r.BC_y) * r.pxY, max(r.BC_z, r.NrPixelsZ - 1 - r.BC_z) * r.pxZ)


def test_non_square_default_range_covers_the_physical_corner():
    r = _result(172.0, 200.0)
    s = spec_from_calibration_result(r, RBinSize=1.0)
    assert float(s.RhoD) == pytest.approx(_corner_um(r), rel=1e-12)
    assert s.RMax * s.mean_pixel_um() == pytest.approx(_corner_um(r), rel=1e-12)


def test_square_default_range_is_unchanged():
    r = _result(150.0, 150.0)
    s = spec_from_calibration_result(r, RBinSize=1.0)
    corner_px = math.sqrt(max(r.BC_y, r.NrPixelsY - 1 - r.BC_y) ** 2 + max(r.BC_z, r.NrPixelsZ - 1 - r.BC_z) ** 2)
    assert s.RMax == corner_px and float(s.RhoD) == corner_px * 150.0
