"""Tests for the shared RhoD unit self-consistency guard."""
import warnings

import pytest

from midas_distortion.rhod import (
    detector_max_corner_dist_um,
    check_rho_d_um,
    resolve_rho_d_um,
    resolve_rho_d_um_warn,
)

# A leighanne-like 2880^2 Varex: px=150 µm, BC near centre.
GEO = dict(NrPixelsY=2880, NrPixelsZ=2880, BC_y=1431.7, BC_z=1472.5, pxY=150.0)
DMAX = detector_max_corner_dist_um(**GEO)          # ~310000 µm
CORNER_PX = DMAX / 150.0                            # ~2065 px


def test_dmax_is_um_scale():
    assert 250_000 < DMAX < 360_000   # micrometres, not pixels


def test_resolve_passes_through_correct_um():
    val, how = resolve_rho_d_um(DMAX, **GEO)
    assert abs(val - DMAX) < 1.0
    assert how.startswith("as-is")


def test_resolve_corrects_pixels_to_um():
    # RhoD given in pixels (~2065) must be detected and scaled by px.
    val, how = resolve_rho_d_um(CORNER_PX, **GEO)
    assert abs(val - DMAX) < 1.0
    assert "pixels" in how


def test_resolve_defaults_when_missing():
    val, how = resolve_rho_d_um(0.0, **GEO)
    assert abs(val - DMAX) < 1.0
    assert "default" in how


def test_check_raises_on_pixel_value():
    with pytest.raises(ValueError):
        check_rho_d_um(CORNER_PX, **GEO)            # px value, strict
    assert check_rho_d_um(DMAX, **GEO) is None      # µm value, healthy


def test_warn_helper_warns_on_correction_only():
    # pixel value -> corrected + warns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = resolve_rho_d_um_warn(CORNER_PX, **GEO)
    assert abs(out - DMAX) < 1.0
    assert any("RhoD" in str(x.message) for x in w)
    # already-µm value -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = resolve_rho_d_um_warn(DMAX, **GEO)
    assert abs(out - DMAX) < 1.0
    assert not w
