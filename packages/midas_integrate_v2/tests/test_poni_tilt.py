"""PONI <-> MIDAS geometry under tilt, pinned per pixel against pyFAI's own 2theta.

A PONI is the foot of the perpendicular from the sample to the detector plane, not the direct-beam point, and ``dist`` is the
perpendicular distance, not ``Lsd``. ``poni = (BC + 0.5) * px`` is exact for an untilted detector only; with tilt it is off by
``dist * tan(rot)`` (about 1 px per degree, 12.8 px for the real HPCAT file below). These tests check ``midas_to_poni`` /
``poni_to_midas`` / ``poni_file_to_midas`` / ``make_pyfai_integrator`` against pyFAI's per-pixel 2theta and against the MIDAS pixel
definition itself, and pin the row/column assignment (``image[row=Z, col=Y]``), which ``make_pyfai_integrator`` used to transpose.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    make_pyfai_integrator, midas_to_poni, poni_to_midas, poni_file_to_midas, poni_file_to_bc, spec_from_v1_params,
)
from midas_integrate_v2.compat.pyfai import _midas_tilt_matrix, read_poni

NY, NZ = 90, 60                 # columns, rows
BC_Y, BC_Z = 61.3, 17.6         # deliberately unequal: a transposed beam centre must fail
PXY, PXZ = 200.0, 150.0
LSD = 400_000.0


def midas_tth(Lsd_um, bc_y, bc_z, tx, ty, tz, pxy, pxz, ny=NY, nz=NZ):
    """2theta of every pixel from the MIDAS definition: P = (Lsd,0,0) + TRs . (0, (BC_y - Y) pxY, (Z - BC_z) pxZ); image[row=Z, col=Y]."""
    Y, Z = np.meshgrid(np.arange(ny, dtype=float), np.arange(nz, dtype=float))
    TRs = _midas_tilt_matrix(tx, ty, tz)
    v = np.stack([np.zeros_like(Y), (bc_y - Y) * pxy, (Z - bc_z) * pxz])
    P = np.array([Lsd_um, 0.0, 0.0])[:, None, None] + np.tensordot(TRs, v, axes=(1, 0))
    return np.arctan2(np.hypot(P[1], P[2]), P[0])


def pyfai_tth(g, pxy, pxz, ny=NY, nz=NZ):
    from pyFAI.detectors import Detector
    det = Detector(pixel1=pxz * 1e-6, pixel2=pxy * 1e-6, max_shape=(nz, ny))
    ai = pyFAI.AzimuthalIntegrator(dist=g["dist"], poni1=g["poni1"], poni2=g["poni2"], rot1=g["rot1"], rot2=g["rot2"], rot3=g["rot3"],
                                   detector=det, wavelength=1e-10)
    if hasattr(ai, "center_array"):
        return np.asarray(ai.center_array(shape=(nz, ny), unit="2th_rad"), float)
    return np.asarray(ai.twoThetaArray(shape=(nz, ny)), float)


def dev_px_eq(a, b, Lsd_um, pxy):
    """max |2theta difference| in pixel-equivalents at the local radius, d(R_px) = Lsd/px * d(2theta) / cos^2(2theta)."""
    return float(np.max(np.abs(a - b) * Lsd_um / pxy / np.cos(a) ** 2))


TILTS = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 3.0),
    (0.0, 2.0, 0.0),
    (0.0, -0.327, 14.044),          # the 14 deg Varex frame of the integrate paper
    (1.2, -4.0, 5.0),
    (0.0, 20.0, -8.0),
    (-0.5, 30.0, 40.0),
    (0.0, 0.0, 45.0),
]


@pytest.mark.parametrize("tx,ty,tz", TILTS)
def test_midas_to_poni_reproduces_the_midas_pixel_2theta_in_pyfai(tx, ty, tz):
    g = midas_to_poni(LSD, BC_Y, BC_Z, tx, ty, tz, PXY, PXZ)
    d = dev_px_eq(midas_tth(LSD, BC_Y, BC_Z, tx, ty, tz, PXY, PXZ), pyfai_tth(g, PXY, PXZ), LSD, PXY)
    assert d < 1e-4, d


@pytest.mark.parametrize("tx,ty,tz", TILTS)
def test_round_trip_is_identity(tx, ty, tz):
    g = midas_to_poni(LSD, BC_Y, BC_Z, tx, ty, tz, PXY, PXZ)
    m = poni_to_midas(g["dist"], g["poni1"], g["poni2"], g["rot1"], g["rot2"], g["rot3"], PXY, PXZ)
    assert m["Lsd_um"] == pytest.approx(LSD, rel=1e-12)
    assert (m["BC_y"], m["BC_z"]) == pytest.approx((BC_Y, BC_Z), abs=1e-8)
    assert (m["tx"], m["ty"], m["tz"]) == pytest.approx((tx, ty, tz), abs=1e-9)


@pytest.mark.parametrize("dist,p1,p2,r1,r2,r3", [
    (0.35, 0.012, 0.0090, 0.0063, 0.0035, 1e-5),
    (0.60, 0.011, 0.0150, -0.02, 0.01, 0.003),
    (1.00, 0.020, 0.0120, 0.15, -0.12, 0.30),
])
def test_poni_to_midas_reproduces_the_pyfai_pixel_2theta(dist, p1, p2, r1, r2, r3):
    """From the pyFAI side: the MIDAS geometry derived from an arbitrary PONI gives the same per-pixel 2theta."""
    g = dict(dist=dist, poni1=p1, poni2=p2, rot1=r1, rot2=r2, rot3=r3)
    m = poni_to_midas(dist, p1, p2, r1, r2, r3, PXY, PXZ)
    d = dev_px_eq(pyfai_tth(g, PXY, PXZ), midas_tth(m["Lsd_um"], m["BC_y"], m["BC_z"], m["tx"], m["ty"], m["tz"], PXY, PXZ), m["Lsd_um"], PXY)
    assert d < 1e-4, d


def test_small_angle_limit_matches_the_closed_form_lsd_ty_tz():
    """rot3 = 0: Lsd = dist / (cos rot1 cos rot2), tz = -rot1, ty = +rot2 (pyFAI axis 1 is the image ROW = Z, axis 2 the COLUMN = Y),
    and BC is where pyFAI's 2theta is zero."""
    dist, p1, p2, r1, r2 = 0.50, 0.0100, 0.0123, math.radians(0.9), math.radians(-1.4)
    m = poni_to_midas(dist, p1, p2, r1, r2, 0.0, 172.0, 172.0)
    assert m["Lsd_um"] == pytest.approx(dist / (math.cos(r1) * math.cos(r2)) * 1e6, rel=1e-12)
    assert m["tz"] == pytest.approx(-math.degrees(r1), abs=1e-3)
    assert m["ty"] == pytest.approx(math.degrees(r2), abs=1e-3)
    assert abs(m["tx"]) < 1e-9


def test_beam_centre_is_not_the_poni_when_tilted():
    g = midas_to_poni(LSD, BC_Y, BC_Z, 0.0, 0.0, 2.0, 172.0, 172.0)
    naive_y = g["poni2"] / 172e-6 - 0.5
    naive_z = g["poni1"] / 172e-6 - 0.5
    assert math.hypot(naive_y - BC_Y, naive_z - BC_Z) > 5.0          # 400 mm * tan(2 deg) / 172 um = 81 px: far more than the half pixel


def test_untilted_reduces_to_the_half_pixel_shift():
    g = midas_to_poni(LSD, BC_Y, BC_Z, 0.0, 0.0, 0.0, PXY, PXZ)
    assert g["dist"] == pytest.approx(LSD * 1e-6, rel=1e-12)
    assert g["poni1"] == pytest.approx((BC_Z + 0.5) * PXZ * 1e-6, abs=1e-15)      # rows <-> Z
    assert g["poni2"] == pytest.approx((BC_Y + 0.5) * PXY * 1e-6, abs=1e-15)      # columns <-> Y
    assert (g["rot1"], g["rot2"], g["rot3"]) == (0.0, 0.0, 0.0)


def _spec(tx=0.0, ty=0.0, tz=0.0, pxy=PXY, pxz=PXZ):
    p = IntegrationParams(NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxy, pxZ=pxz, Lsd=LSD, BC_y=BC_Y, BC_z=BC_Z, tx=tx, ty=ty, tz=tz,
                          RhoD=1.0e6, RMin=2.0, RMax=30.0, RBinSize=0.5, EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0, Wavelength=0.172979)
    return spec_from_v1_params(p, requires_grad=False)


def _ai_tth(ai, ny=NY, nz=NZ):
    if hasattr(ai, "center_array"):
        return np.asarray(ai.center_array(shape=(nz, ny), unit="2th_rad"), float)
    return np.asarray(ai.twoThetaArray(shape=(nz, ny)), float)


@pytest.mark.parametrize("tx,ty,tz", [(0.0, 0.0, 0.0), (0.0, -0.327, 14.044), (1.2, -4.0, 5.0)])
def test_make_pyfai_integrator_matches_midas_per_pixel_incl_a_non_symmetric_beam_centre(tx, ty, tz):
    """Regression: BC_y was passed as the ROW, transposing the beam centre (60 px-equivalent here). Non-square pixels too."""
    ai = make_pyfai_integrator(_spec(tx, ty, tz))
    d = dev_px_eq(midas_tth(LSD, BC_Y, BC_Z, tx, ty, tz, PXY, PXZ), _ai_tth(ai), LSD, PXY)
    assert d < 1e-4, d


def test_make_pyfai_integrator_matches_the_v2_forward_model():
    """Independent oracle: v2's own pixel -> R_px map (square pixels, tilt), tan(2theta) = R_px * px / Lsd."""
    torch = pytest.importorskip("torch")
    from midas_integrate_v2.forward import pixel_to_REta_from_spec
    spec = _spec(0.4, -1.1, 6.0, pxy=172.0, pxz=172.0)
    Y, Z = np.meshgrid(np.arange(NY, dtype=float), np.arange(NZ, dtype=float))
    R_px = pixel_to_REta_from_spec(torch.tensor(Y.ravel()), torch.tensor(Z.ravel()), spec).R_px
    tth_v2 = np.arctan(R_px.detach().numpy().reshape(NZ, NY) * 172.0 / LSD)
    d = dev_px_eq(tth_v2, _ai_tth(make_pyfai_integrator(spec)), LSD, 172.0)
    assert d < 1e-3, d


def test_make_pyfai_integrator_tilted_with_a_flipping_orientation_is_refused():
    with pytest.raises(NotImplementedError):
        make_pyfai_integrator(_spec(0.0, 0.0, 3.0), orientation=2, shape=(NZ, NY))


# a real calibration: HPCAT 16-ID-B CeO2, rot1 = 0.36 deg, dist = 350 mm  ->  direct beam 2.2 mm = 12.8 px from the PONI
REAL_PONI = """\
# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
poni_version: 2.1
Detector: Detector
Detector_config: {"pixel1": 0.00017199999999999998, "pixel2": 0.00017199999999999998, "orientation": 3, "max_shape": [1679, 1475]}
Distance: 0.3496217525840799
Poni1: 0.148050995243481
Poni2: 0.1292328649679976
Rot1: 0.006293819821405356
Rot2: 0.003454694798076228
Rot3: 1.0871659998468175e-05
Wavelength: 4.245900000000001e-11
"""


def test_poni_file_to_midas_on_a_real_calibration(tmp_path):
    f = tmp_path / "hpcat.poni"
    f.write_text(REAL_PONI)
    m = poni_file_to_midas(f)
    naive_row, naive_col = poni_file_to_bc(f)                      # pyFAI-named: (row, column) of the PONI
    assert math.hypot(m["BC_z"] - naive_row, m["BC_y"] - naive_col) > 10.0      # the old reading was ~13 px off the direct beam
    p = read_poni(f)
    g = dict(dist=p["Distance"], poni1=p["Poni1"], poni2=p["Poni2"], rot1=p["Rot1"], rot2=p["Rot2"], rot3=p["Rot3"])
    ny, nz = 1475, 1679
    d = dev_px_eq(pyfai_tth(g, 172.0, 172.0, ny, nz),
                  midas_tth(m["Lsd_um"], m["BC_y"], m["BC_z"], m["tx"], m["ty"], m["tz"], 172.0, 172.0, ny, nz), m["Lsd_um"], 172.0)
    assert d < 1e-4, d


def test_poni_file_to_midas_untilted_applies_the_orientation_flip(tmp_path):
    f = tmp_path / "flat.poni"
    f.write_text(REAL_PONI.replace("1.0871659998468175e-05", "0").replace("0.006293819821405356", "0").replace("0.003454694798076228", "0")
                 .replace('"orientation": 3', '"orientation": 2'))
    m = poni_file_to_midas(f)
    assert m["BC_z"] == pytest.approx((1679 - 1) - (0.148050995243481 / 172e-6 - 0.5), abs=1e-6)      # TopRight flips rows
    assert m["BC_y"] == pytest.approx(0.1292328649679976 / 172e-6 - 0.5, abs=1e-6)
    assert (m["tx"], m["ty"], m["tz"]) == (0.0, 0.0, 0.0)
    g = f.read_text().replace('"orientation": 2', '"orientation": 2').replace("Rot1: 0", "Rot1: 0.001")
    (tmp_path / "tilted_flip.poni").write_text(g)
    with pytest.raises(NotImplementedError):
        poni_file_to_midas(tmp_path / "tilted_flip.poni")
