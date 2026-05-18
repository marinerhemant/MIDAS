"""Regression test pinning the BC ↔ PONI 0.5 px shift.

The single most common bug when comparing v2 to pyFAI is dropping the
0.5 px difference between MIDAS's pixel-centre convention and pyFAI's
pixel-corner convention. This test pins:

1. The numerical convention (`(BC + 0.5) · pixel_size = poni`).
2. The round-trip BC → PONI → BC is the identity.
3. The headline test: integrating the same image with pyFAI (using
   the correct +0.5 conversion) and v2 should produce profiles whose
   first-ring centroid agrees within sub-bin precision. Skip this
   test gracefully if pyFAI is not installed.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    bc_to_poni, poni_to_bc, make_pyfai_integrator,
    PolygonBinGeometry, integrate_polygon,
)


def test_bc_to_poni_includes_half_pixel_shift():
    """``poni = (BC + 0.5) · pixel_size`` (not ``BC · pixel_size``)."""
    BC_y, BC_z = 100.0, 200.0
    pxY, pxZ = 150.0, 200.0
    p1, p2 = bc_to_poni(BC_y, BC_z, pxY, pxZ)
    # poni in metres: pixel-centre BC=100 with px=150µm is 100.5 px from
    # the pixel-corner origin, then × 150µm = 15075 µm = 0.015075 m
    assert p1 == pytest.approx((BC_y + 0.5) * pxY * 1e-6, rel=1e-12)
    assert p2 == pytest.approx((BC_z + 0.5) * pxZ * 1e-6, rel=1e-12)
    # And NOT just BC · pxY · 1e-6 (the wrong common conversion)
    assert p1 != pytest.approx(BC_y * pxY * 1e-6, rel=1e-12)


def test_poni_to_bc_round_trip_is_identity():
    """BC → PONI → BC must give back the original BC exactly."""
    BC_y, BC_z = 685.49, 921.03
    pxY, pxZ = 172.0, 172.0
    p1, p2 = bc_to_poni(BC_y, BC_z, pxY, pxZ)
    BC_y_back, BC_z_back = poni_to_bc(p1, p2, pxY, pxZ)
    assert BC_y_back == pytest.approx(BC_y, abs=1e-9)
    assert BC_z_back == pytest.approx(BC_z, abs=1e-9)


def test_bc_to_poni_handles_non_square_pixels():
    BC_y, BC_z = 100.0, 200.0
    pxY, pxZ = 150.0, 100.0       # non-square
    p1, p2 = bc_to_poni(BC_y, BC_z, pxY, pxZ)
    BC_y_back, BC_z_back = poni_to_bc(p1, p2, pxY, pxZ)
    assert BC_y_back == pytest.approx(BC_y, abs=1e-9)
    assert BC_z_back == pytest.approx(BC_z, abs=1e-9)


def test_make_pyfai_integrator_uses_correct_shift():
    pyfai = pytest.importorskip("pyFAI")

    p = IntegrationParams(
        NrPixelsY=1475, NrPixelsZ=1679,
        pxY=172.0, pxZ=172.0, Lsd=657_437.0,
        BC_y=685.49, BC_z=921.03,
        RhoD=1000.0, RMin=10.0, RMax=200.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=5.0,
        Wavelength=0.172979,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    ai = make_pyfai_integrator(s)
    # Pull poni back out and check
    expected_p1 = (685.49 + 0.5) * 172.0 * 1e-6
    expected_p2 = (921.03 + 0.5) * 172.0 * 1e-6
    assert abs(ai.poni1 - expected_p1) < 1e-12
    assert abs(ai.poni2 - expected_p2) < 1e-12
    # And NOT the wrong (no-shift) conversion
    wrong_p1 = 685.49 * 172.0 * 1e-6
    assert abs(ai.poni1 - wrong_p1) > 1e-9


def test_make_pyfai_integrator_raises_when_pyfai_missing(monkeypatch):
    """Sanity-check the ImportError path."""
    import sys
    real_pyfai = sys.modules.pop("pyFAI", None)
    monkeypatch.setattr(sys, "modules",
                          {**sys.modules, "pyFAI": None})
    p = IntegrationParams(
        NrPixelsY=24, NrPixelsZ=24, pxY=200.0, pxZ=200.0, Lsd=1e6,
        BC_y=12.0, BC_z=12.0, RhoD=24.0,
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    with pytest.raises(ImportError, match="pyFAI"):
        make_pyfai_integrator(s)
    if real_pyfai is not None:
        sys.modules["pyFAI"] = real_pyfai


def test_pyfai_vs_v2_first_ring_centroid_agrees():
    """Headline: when the +0.5 shift is applied, pyFAI's profile and
    v2's polygon profile have the same first-ring centroid to
    sub-bin precision."""
    pyfai = pytest.importorskip("pyFAI")

    NY = NZ = 64
    BC = (NY / 2.0 + 0.37, NZ / 2.0 - 0.41)

    # Synth image with one ring at R=15 px around BC
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R_pix = np.sqrt((yy - BC[0]) ** 2 + (zz - BC[1]) ** 2)
    img = np.exp(-((R_pix - 15.0) ** 2) / (2 * 1.0 ** 2))

    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=BC[0], BC_z=BC[1],
        RhoD=float(NY), RMin=2.0, RMax=28.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Wavelength=0.172979,
    )
    s = spec_from_v1_params(p, requires_grad=False)

    # v2 polygon profile
    geom = PolygonBinGeometry.from_spec(s)
    int2d = integrate_polygon(torch.from_numpy(img), geom, normalize=True)
    n_r = s.n_r_bins
    r_axis = s.RMin + s.RBinSize * (np.arange(n_r) + 0.5)
    prof_v2 = int2d.mean(dim=0).numpy()

    # pyFAI profile (with the corrected conversion baked in)
    ai = make_pyfai_integrator(s)
    res = ai.integrate1d(img, n_r, unit="r_mm", error_model=None)
    # pyFAI's r_mm axis is in mm — convert to px for centroid comparison
    r_pyfai_px = res.radial * 1000.0 / p.pxY      # mm → µm → px

    def centroid(r_axis, prof, lo, hi):
        m = (r_axis > lo) & (r_axis < hi)
        if not m.any() or prof[m].sum() <= 0:
            return float("nan")
        return float((r_axis[m] * np.maximum(prof[m], 0)).sum()
                      / np.maximum(prof[m], 0).sum())

    c_v2 = centroid(r_axis, prof_v2, 12, 18)
    c_pyfai = centroid(r_pyfai_px, res.intensity, 12, 18)
    # Sub-bin agreement (RBinSize = 0.5 px)
    assert abs(c_v2 - c_pyfai) < 0.5, (
        f"v2 vs pyFAI first-ring centroid drift {abs(c_v2 - c_pyfai):.3f} > "
        f"0.5 px (likely the 0.5 px BC ↔ PONI shift was wrong)"
    )


def test_wrong_conversion_recovers_BC_shifted_by_half_pixel():
    """The canonical proof: if you skip the +0.5 in BC → PONI, then
    converting BACK from that wrong PONI gives a BC shifted by exactly
    0.5 px from the original. That's the entire convention difference,
    isolated."""
    BC_y, BC_z = 685.49, 921.03
    pxY, pxZ = 172.0, 172.0
    # Wrong conversion (skip +0.5)
    poni1_wrong = BC_y * pxY * 1e-6
    poni2_wrong = BC_z * pxZ * 1e-6
    # Convert back via the (correct) reverse helper
    BC_y_back, BC_z_back = poni_to_bc(poni1_wrong, poni2_wrong, pxY, pxZ)
    # The BC came back exactly 0.5 px BELOW the original — quantitative
    # proof the +0.5 matters.
    assert BC_y_back == pytest.approx(BC_y - 0.5, abs=1e-9)
    assert BC_z_back == pytest.approx(BC_z - 0.5, abs=1e-9)
