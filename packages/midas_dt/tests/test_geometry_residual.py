"""The geometry conversions that were losing terms silently.

Each of these was a real defect found while measuring a strain error floor on
real 11-ID-C data, and each failed the same way: a term the calibration had
just refined was dropped on the way into the reduction, with nothing louder
than a log line to say so. A reconstruction built on the result looks entirely
normal -- which is why they need tests rather than care.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt import Channel, DTGeometry, from_calibration


V2_TERMS = {
    "iso_R2": -2.537067e-03, "iso_R4": 3.552942e-03, "iso_R6": -2.477731e-03,
    "a1": -2.134449e-03, "phi1": -1.825965,
    "a2": -2.050454e-04, "phi2": -68.261969,
    "a3": -1.641486e-04, "phi3": -79.370585,
    "a4": 6.940275e-04, "phi4": 4.667391,
    "a5": 3.864312e-04, "phi5": -28.565661,
    "a6": 4.587806e-04, "phi6": -12.930405,
}


class _FakeResult:
    """The fields `from_calibration` reads off an AutoCalibrationResult."""

    Lsd = 1632201.337
    BC_y, BC_z = 1445.976, 1438.596
    pxY = pxZ = 150.0
    NrPixelsY = NrPixelsZ = 2880
    wavelength_A = 0.11595
    tx, ty, tz = 0.0, 0.010798, 0.205049
    distortion = dict(V2_TERMS)
    residual_corr_bin_path = "/somewhere/residual_corr.bin"


def test_v2_distortion_accepts_v2_names_unchanged():
    """Handing it a calibration's own dict used to drop every term."""
    geo = DTGeometry(lsd_um=1e6, bc_y_px=1000, bc_z_px=1000, px_um=150,
                     n_pixels_y=2880, n_pixels_z=2880, wavelength_a=0.116,
                     distortion=dict(V2_TERMS))
    out = geo.v2_distortion()
    assert set(out) == set(V2_TERMS), "v2 named terms must survive unchanged"
    for k, v in V2_TERMS.items():
        assert out[k] == pytest.approx(v)


def test_v2_distortion_still_maps_legacy_p_names():
    """The legacy path must keep working, and by the canonical permutation."""
    from midas_distortion import V1_TO_V2_DISTORTION

    legacy = {f"p{i}": float(i + 1) for i in V1_TO_V2_DISTORTION}
    geo = DTGeometry(lsd_um=1e6, bc_y_px=1000, bc_z_px=1000, px_um=150,
                     n_pixels_y=2880, n_pixels_z=2880, wavelength_a=0.116,
                     distortion=legacy)
    out = geo.v2_distortion()
    for i, name in V1_TO_V2_DISTORTION.items():
        assert out[name] == pytest.approx(float(i + 1)), (
            f"p{i} must land on {name}; a positional guess sends it elsewhere")


def test_from_calibration_carries_distortion_rhod_and_residual():
    """All three of the terms that were being dropped, in one place."""
    geo = from_calibration(_FakeResult())

    assert geo.v2_distortion(), "distortion was dropped on the way in"
    assert set(geo.v2_distortion()) == set(V2_TERMS)

    # RhoD must describe THIS detector, not the class default. For a 2880 px
    # detector at 150 um it is order 3e5 um; the 1.5e5 default would halve the
    # scale the distortion polynomial is defined against.
    assert geo.rho_d_um > 2.0e5, (
        f"RhoD {geo.rho_d_um} looks like the class default, not this detector")

    assert geo.residual_corr_path == "/somewhere/residual_corr.bin"


def test_residual_map_reaches_the_integration_spec():
    """The map is useless unless it survives into the spec the reducer uses."""
    pytest.importorskip("midas_integrate_v2")
    geo = from_calibration(_FakeResult())
    spec = geo.to_integration_spec(
        Channel(350.0, 1200.0, r_bin=1.0, eta_bin=10.0))
    assert getattr(spec, "ResidualCorrectionMap", "") == \
        "/somewhere/residual_corr.bin"


def test_no_residual_map_leaves_the_field_empty():
    """Absence must stay absent -- not a path to a file that does not exist."""
    pytest.importorskip("midas_integrate_v2")
    r = _FakeResult()
    r.residual_corr_bin_path = ""
    geo = from_calibration(r)
    assert geo.residual_corr_path is None
    spec = geo.to_integration_spec()
    assert not getattr(spec, "ResidualCorrectionMap", "")
