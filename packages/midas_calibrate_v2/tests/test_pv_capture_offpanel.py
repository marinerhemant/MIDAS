"""``autocalibrate_pv`` reaches an off-panel, large-tilt geometry from a seed its fine window cannot see.

A 1024 px frame at 200 um, beam centre 56 px beyond the panel edge, tz 12 deg. Seed: Lsd -1.3 %,
BC_y -12 px, ty and tz off by 0.3 and 1 deg. Measured: with the capture phase the fit lands on the
same point a truth-seeded run does (28.6 ue); with ``capture_window_px=0`` -- the fine-window loop
alone -- it ends 21.6 mm short in Lsd with ty railed at its bound, at 4746 ue, and warns.

The truth-seeded run itself lands 0.08 deg / 143 um from the painted geometry, identically with and
without capture (the renderer and the pipeline model differ slightly), so the tolerances below are
absolute and contain that offset while sitting far inside the failure.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

from midas_calibrate_v2.pipelines.single_pv import CaptureRangeWarning, autocalibrate_pv

NY = NZ = 1024
PX = 200.0
TRUE = dict(Lsd=1_000_000.0, BC_y=1080.0, BC_z=500.0, ty=-0.3, tz=12.0)
SEED = dict(Lsd=987_000.0, BC_y=1068.0, BC_z=500.0, ty=0.0, tz=13.0)


def _params(g) -> CalibrationParams:
    p = CalibrationParams()
    p.NrPixelsY = NY; p.NrPixelsZ = NZ; p.pxY = PX; p.pxZ = PX
    p.Lsd = g["Lsd"]; p.BC_y = g["BC_y"]; p.BC_z = g["BC_z"]
    p.tx = 0.0; p.ty = g["ty"]; p.tz = g["tz"]
    p.Wavelength = 0.173; p.SpaceGroup = 225
    p.LatticeConstant = (5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0)
    p.MaxRingRad = max(math.hypot(cy - g["BC_y"], cz - g["BC_z"])
                       for cy, cz in [(0, 0), (NY, 0), (0, NZ), (NY, NZ)]) + 20.0
    p.RhoD = p.MaxRingRad * PX
    p.EtaBinSize = 10.0; p.RBinSize = 0.5
    p.tolLsd = 30000.0; p.tolBC = 30.0; p.tolTilts = 3.0
    p.Refine = {"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": False for i in range(15)}}
    return p


@pytest.fixture(scope="module")
def image():
    p = _params(TRUE)
    Y, Z = np.meshgrid(np.arange(NY, dtype=np.float64), np.arange(NZ, dtype=np.float64))
    R, _ = pixel_to_REta(Y, Z, Ycen=TRUE["BC_y"], Zcen=TRUE["BC_z"],
                         TRs=build_tilt_matrix(0.0, TRUE["ty"], TRUE["tz"]),
                         Lsd=TRUE["Lsd"], RhoD=p.RhoD, px=PX, parallax=0.0)
    rng = np.random.default_rng(0)
    img = 50.0 + rng.normal(0, 5.0, R.shape)
    for r in build_ring_table(p).r_ideal_px:
        img += (1000.0 / (1.0 + r / 300.0)) * np.exp(-0.5 * ((R - r) / 1.5) ** 2)
    return np.clip(img, 0, None)


def _run(image, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = autocalibrate_pv(_params(SEED), image, verbose=False, **kw)
    g = {k: float(res.unpacked[k]) for k in ("Lsd", "BC_y", "BC_z", "ty", "tz")}
    warned = any(issubclass(w.category, CaptureRangeWarning) for w in caught)
    return g, float(res.history[-1].mean_strain_uE), warned, res


@pytest.fixture(scope="module")
def with_capture(image):
    return _run(image)


@pytest.fixture(scope="module")
def fine_only(image):
    return _run(image, capture_window_px=0.0)


def test_capture_reaches_the_geometry(with_capture):
    g, strain, warned, res = with_capture
    assert abs(g["tz"] - TRUE["tz"]) < 0.25, g
    assert abs(g["Lsd"] - TRUE["Lsd"]) < 1000.0, g
    assert math.hypot(g["BC_y"] - TRUE["BC_y"], g["BC_z"] - TRUE["BC_z"]) < 1.0, g
    assert strain < 100.0, strain
    assert len(res.capture_history) >= 1
    # it had to travel: the seed is 13 mm and 12 px away
    assert abs(g["Lsd"] - SEED["Lsd"]) > 10_000.0, g


def test_the_fine_window_alone_does_not(fine_only):
    """The null. If this passes the capture phase is not what the test above measures."""
    g, strain, warned, res = fine_only
    assert len(res.capture_history) == 0
    assert strain > 1000.0, (strain, g)


def test_capture_range_warning_flags_only_the_failure(with_capture, fine_only):
    assert fine_only[2], "the fine-window failure did not warn"
    assert not with_capture[2], "the capture run warned although it reached the rings"
