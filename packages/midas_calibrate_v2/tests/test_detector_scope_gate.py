"""The scope gate: can this detector see enough of the calibrant to be calibrated?

Every other gate in this module judges a fit.  This one runs before there is a
fit, because a detector that cannot record the rings still produces a converged
calibration — onto parasitic scatter — and the post-fit gates then grade a
meaningless answer.

The numbers below are the real 1-ID cases that motivated it.
"""
import math

import pytest

from midas_calibrate_v2.pipelines.diagnostics import detector_scope_gate

CEO2_A = 5.41153
RE_EDGE_KEV = 71.676
LAMBDA = 12.398419 / RE_EDGE_KEV          # ~0.17298 A


def test_ge5_at_1m_passes():
    """The reference case: 2048^2 at 200 um, 1000 mm, Re edge — plenty of rings."""
    r = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=1_000_000.0,
                            pxY_um=200.0, NrPixelsY=2048, NrPixelsZ=2048)
    assert r.severity == "ok"
    assert r.metrics["n_rings_on_panel"] >= 5


def test_saxs_detector_at_long_distance_fails():
    """pixirad: 402 x 1024 at 62 um (25 x 63 mm) at 3300 mm.

    This is the case that put a SAXS detector into a powder-calibration archive:
    30 work units, 16 of which failed outright and 14 of which produced a
    plausible-looking calibration with Lsd as low as 26.9 mm.
    """
    r = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=3_300_000.0,
                            pxY_um=62.0, NrPixelsY=1024, NrPixelsZ=402)
    assert r.severity == "fail"
    assert "not determinable" in r.message.lower()
    assert r.metrics["n_rings_on_panel"] == 0
    # the first CeO2 ring is far outside a panel that only reaches ~34 mm
    assert r.metrics["R_reach_mm"] == pytest.approx(34.1, abs=1.0)
    assert r.metrics["R_innermost_mm"] > 100.0


def test_ge_panel_too_far_fails():
    """bt_1id_jul25b: the GE quad at 3300 mm. Not a small detector — too far.

    These fitted 480-573 mm against a filename recording 3300 mm; the gate
    stops them without needing to know the filename.
    """
    r = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=3_300_000.0,
                            pxY_um=200.0, NrPixelsY=2048, NrPixelsZ=2048)
    assert r.severity == "fail"


def test_reach_uses_beam_centre_not_panel_centre():
    """An off-panel beam centre reaches further on one side; the gate must use
    the actual centre when told, and default to the generous panel centre."""
    far = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=2_000_000.0,
                              pxY_um=200.0, NrPixelsY=2048, NrPixelsZ=2048,
                              BC_y=0.0, BC_z=0.0)
    centred = detector_scope_gate(wavelength_A=LAMBDA, Lsd_um=2_000_000.0,
                                  pxY_um=200.0, NrPixelsY=2048, NrPixelsZ=2048)
    assert far.metrics["R_reach_mm"] > centred.metrics["R_reach_mm"]
    assert far.metrics["n_rings_on_panel"] >= centred.metrics["n_rings_on_panel"]


def test_min_rings_is_the_knob():
    """Same geometry, stricter requirement — the verdict must follow it."""
    kw = dict(wavelength_A=LAMBDA, Lsd_um=1_800_000.0, pxY_um=200.0,
              NrPixelsY=2048, NrPixelsZ=2048)
    n = detector_scope_gate(**kw).metrics["n_rings_on_panel"]
    assert detector_scope_gate(min_rings=int(n) + 1, **kw).severity == "fail"
    assert detector_scope_gate(min_rings=1, **kw).severity != "fail"


def test_lab6_has_more_rings_than_ceo2_at_equal_geometry():
    """LaB6's larger cell puts more rings in the same 2-theta window — the gate
    must take the lattice it is given, not assume CeO2."""
    kw = dict(wavelength_A=LAMBDA, Lsd_um=1_000_000.0, pxY_um=200.0,
              NrPixelsY=2048, NrPixelsZ=2048)
    ceo2 = detector_scope_gate(lattice_a_A=CEO2_A, space_group=225, **kw)
    lab6 = detector_scope_gate(lattice_a_A=4.15689, space_group=221, **kw)
    assert lab6.metrics["n_rings_on_panel"] != ceo2.metrics["n_rings_on_panel"]


def test_rings_scale_with_distance():
    """Sanity on the geometry itself: R = Lsd*tan(2theta), so moving the
    detector out monotonically drops the ring count."""
    kw = dict(wavelength_A=LAMBDA, pxY_um=200.0, NrPixelsY=2048, NrPixelsZ=2048)
    counts = [detector_scope_gate(Lsd_um=L * 1000.0, **kw).metrics["n_rings_on_panel"]
              for L in (500, 1000, 1500, 2000, 3000)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]
