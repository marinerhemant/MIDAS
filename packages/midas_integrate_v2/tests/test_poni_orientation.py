"""pyFAI detector-orientation handling in the PONI ↔ BC conversion.

Ignoring ``Detector_config.orientation`` puts the beam centre
``N - 1 - 2*BC`` pixels wrong — the far side of the detector — and raises
nothing. These tests pin the flip table, and pin it against the installed
pyFAI rather than against documentation, so a semantics change upstream
fails here instead of silently in someone's calibration.

They also pin the trap that cost a real analysis a week: orientation 3 is
pyFAI's DEFAULT and is geometrically identical to 0, so a PONI declaring 3
says nothing at all.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_integrate_v2 import (
    bc_to_poni, poni_to_bc, read_poni, poni_file_to_bc,
    orientation_flips, describe_orientation,
)
from midas_integrate_v2.compat.pyfai import (
    _ORIENTATION_FLIPS, NO_FLIP_ORIENTATIONS,
)

PX = 172.0                     # µm, Pilatus
SHAPE = (1679, 1475)           # rows, cols — Pilatus 2M

# A real PONI, from an HPCAT Sector 16 CeO2 calibration (pyFAI 2.1).
REAL_PONI = """\
# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
# Calibration done on Fri Apr 24 15:01:02 2026
poni_version: 2.1
Detector: Detector
Detector_config: {"pixel1": 0.00017199999999999998, "pixel2": \
0.00017199999999999998, "orientation": 3, "max_shape": [1679, 1475]}
Distance: 0.3496217525840799
Poni1: 0.148050995243481
Poni2: 0.1292328649679976
Rot1: 0.006293819821405356
Rot2: 0.003454694798076228
Rot3: 1.0871659998468175e-05
Wavelength: 4.245900000000001e-11
"""


# --------------------------------------------------------------------------
# backwards compatibility: the historical no-orientation calls are untouched
# --------------------------------------------------------------------------

def test_no_orientation_reproduces_historical_behaviour():
    BC_y, BC_z = 685.49, 921.03
    assert bc_to_poni(BC_y, BC_z, PX, PX) == (
        (BC_y + 0.5) * PX * 1e-6, (BC_z + 0.5) * PX * 1e-6)
    p1, p2 = bc_to_poni(BC_y, BC_z, PX, PX)
    y, z = poni_to_bc(p1, p2, PX, PX)
    assert y == pytest.approx(BC_y, abs=1e-9)
    assert z == pytest.approx(BC_z, abs=1e-9)


@pytest.mark.parametrize("orientation", sorted(_ORIENTATION_FLIPS))
def test_round_trip_is_identity_for_every_orientation(orientation):
    BC_y, BC_z = 685.49, 921.03
    p1, p2 = bc_to_poni(BC_y, BC_z, PX, PX,
                        orientation=orientation, shape=SHAPE)
    y, z = poni_to_bc(p1, p2, PX, PX,
                      orientation=orientation, shape=SHAPE)
    assert y == pytest.approx(BC_y, abs=1e-9)
    assert z == pytest.approx(BC_z, abs=1e-9)


# --------------------------------------------------------------------------
# the trap
# --------------------------------------------------------------------------

def test_orientation_3_and_0_are_identical_and_mean_no_flip():
    """A PONI declaring orientation 3 is declaring pyFAI's default."""
    assert orientation_flips(0) == orientation_flips(3) == (False, False)
    assert 0 in NO_FLIP_ORIENTATIONS and 3 in NO_FLIP_ORIENTATIONS
    assert orientation_flips(None) == (False, False)
    assert "implies nothing" in describe_orientation(3)


def test_ignoring_a_flip_lands_on_the_far_side_of_the_detector():
    """Quantify the harm: the error is N - 1 - 2*BC px, not a rounding slip."""
    BC_y, BC_z = 685.49, 921.03
    p1, p2 = bc_to_poni(BC_y, BC_z, PX, PX, orientation=2, shape=SHAPE)
    naive_y, naive_z = poni_to_bc(p1, p2, PX, PX)          # orientation dropped
    assert naive_y == pytest.approx((SHAPE[0] - 1) - BC_y, abs=1e-9)
    assert naive_z == pytest.approx(BC_z, abs=1e-9)        # TopRight: rows only
    assert abs(naive_y - BC_y) == pytest.approx(
        abs(SHAPE[0] - 1 - 2 * BC_y), abs=1e-9)
    assert abs(naive_y - BC_y) > 300.0


def test_flipping_orientation_without_shape_raises():
    for o in (1, 2, 4):
        with pytest.raises(ValueError, match="shape"):
            poni_to_bc(0.1, 0.1, PX, PX, orientation=o)
    # ...but the no-flip ones do not need it
    poni_to_bc(0.1, 0.1, PX, PX, orientation=3)


def test_axis_swapping_orientations_refuse_rather_than_guess():
    for o in (5, 6, 7, 8):
        with pytest.raises(NotImplementedError, match="transposes"):
            orientation_flips(o)
    with pytest.raises(ValueError, match="unknown"):
        orientation_flips(99)


# --------------------------------------------------------------------------
# the oracle: pyFAI itself defines the table
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", sorted(_ORIENTATION_FLIPS))
def test_flip_table_matches_installed_pyfai(orientation):
    """Re-derive (flip_rows, flip_cols) from pyFAI's own pixel positions."""
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector

    det = Detector(pixel1=PX * 1e-6, pixel2=PX * 1e-6,
                   max_shape=SHAPE, orientation=orientation)
    p1, p2, _ = det.calc_cartesian_positions()
    p1 = p1.reshape(SHAPE); p2 = p2.reshape(SHAPE)
    measured = (bool(p1[0, 0] > p1[-1, 0]), bool(p2[0, 0] > p2[0, -1]))
    assert measured == _ORIENTATION_FLIPS[orientation], (
        f"pyFAI {pyFAI.version} orientation {orientation} flips {measured}, "
        f"table says {_ORIENTATION_FLIPS[orientation]}"
    )


def test_pyfai_default_orientation_is_still_3():
    pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector
    det = Detector(pixel1=PX * 1e-6, pixel2=PX * 1e-6, max_shape=SHAPE)
    assert int(det.orientation) == 3


@pytest.mark.parametrize("orientation", sorted(_ORIENTATION_FLIPS))
def test_poni_to_bc_names_the_pixel_pyfai_names(orientation):
    """End to end: our BC must be the array element pyFAI puts the PONI on."""
    pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector

    P1, P2 = 0.148050995243481, 0.1292328649679976
    det = Detector(pixel1=PX * 1e-6, pixel2=PX * 1e-6,
                   max_shape=SHAPE, orientation=orientation)
    p1, p2, _ = det.calc_cartesian_positions()
    k = int(np.argmin((p1.ravel() - P1) ** 2 + (p2.ravel() - P2) ** 2))
    row, col = divmod(k, SHAPE[1])

    BC_y, BC_z = poni_to_bc(P1, P2, PX, PX,
                            orientation=orientation, shape=SHAPE)
    assert round(BC_y) == row
    assert round(BC_z) == col


# --------------------------------------------------------------------------
# reading the file
# --------------------------------------------------------------------------

def test_read_poni_parses_config_orientation_and_shape(tmp_path):
    f = tmp_path / "refined.poni"
    f.write_text(REAL_PONI)
    p = read_poni(f)
    assert p["Distance"] == pytest.approx(0.3496217525840799)
    assert p["Wavelength"] == pytest.approx(4.2459e-11)
    assert p["Rot3"] == pytest.approx(1.0871659998468175e-05)
    assert p["orientation"] == 3
    assert p["shape"] == (1679, 1475)
    assert p["Detector_config"]["pixel1"] == pytest.approx(172e-6)
    assert p["Detector"] == "Detector"          # unknown keys kept, not dropped


def test_read_poni_survives_a_missing_orientation(tmp_path):
    f = tmp_path / "old.poni"
    f.write_text(REAL_PONI.replace(', "orientation": 3', ""))
    p = read_poni(f)
    assert p["orientation"] is None
    assert p["shape"] == (1679, 1475)
    assert orientation_flips(p["orientation"]) == (False, False)


def test_poni_file_to_bc_uses_the_files_own_pixels(tmp_path):
    f = tmp_path / "refined.poni"
    f.write_text(REAL_PONI)
    BC_y, BC_z = poni_file_to_bc(f)
    # orientation 3 -> no flip, so this is just poni/px - 0.5
    assert BC_y == pytest.approx(0.148050995243481 / (PX * 1e-6) - 0.5, abs=1e-6)
    assert BC_z == pytest.approx(0.1292328649679976 / (PX * 1e-6) - 0.5, abs=1e-6)
