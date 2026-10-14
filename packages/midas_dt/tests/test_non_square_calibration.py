"""midas_dt's geometry is single-pitch; a calibration with pxY != pxZ must not be silently collapsed to pxY."""
from types import SimpleNamespace

import pytest

from midas_dt.geometry import from_calibration


def _result(pxz):
    return SimpleNamespace(Lsd=470000.0, BC_y=100.0, BC_z=120.0, pxY=172.0, pxZ=pxz, wavelength_A=0.18,
                           NrPixelsY=256, NrPixelsZ=256, tx=0.0, ty=0.0, tz=0.0, distortion={})


def test_from_calibration_rejects_non_square_pixels():
    with pytest.raises(ValueError, match="single-pitch"):
        from_calibration(_result(200.0))


def test_from_calibration_accepts_square_and_unset_pxZ():
    for pxz in (172.0, 0.0):
        assert from_calibration(_result(pxz)).px_um == 172.0
