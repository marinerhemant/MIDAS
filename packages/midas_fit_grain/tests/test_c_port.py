"""Regression test for the DiffPos/DiffOme/DiffAngle column-mislabeling bug.

``calc_angle_errors`` returns its three-element error tuple as
``(mean_angle_deg, mean_pos_um, mean_ome_deg)`` (see its docstring and
``FitPosOrStrainsOMP.c::CalcAngleErrors`` lines 686-719). ``driver.py`` must
unpack that tuple into ``GrainResult.ErrorPos``/``ErrorOme``/``ErrorAngle``
so they land in ``OrientPosFit.bin`` cols 22/23/24 — which
``midas_process_grains`` reads back as Grains.csv's
``DiffPos``/``DiffOme``/``DiffAngle`` respectively. A prior version of
``driver.py`` assigned ``err_ini[0..2]`` straight into
``ErrorPos/ErrorOrient/ErrorStrain`` in tuple order, cyclically swapping all
three columns' meanings without changing their header labels.

This test builds one fully-controlled synthetic matched spot with a known,
mutually distinguishable magnitude for each of the three residuals — a
position error (250 um, unbounded by any filter), an omega error (2.5 deg,
bounded < 5 deg by the matching window), and an internal angle error
(0.05 deg, bounded < 1 deg by the match-acceptance threshold) — and checks
that each value ends up in the OrientPosFit.bin column its header claims,
not just that values round-trip unchanged through the binary layer (that
part is already covered by test_io_binary.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_fit_grain import c_port
from midas_fit_grain.io_binary import GrainResult

LSD = 1_000_000.0       # um
WAVELENGTH = 0.1729     # Angstrom
RING_NR = 1

POS_ERR_UM = 250.0      # deliberately large / unbounded-scale
OME_ERR_DEG = 2.5       # deliberately mid-scale, < 5 deg matching window
ANGLE_ERR_DEG = 0.05    # deliberately small, < 1 deg match-accept threshold


def _rotate_by_small_angle(v: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 3-vector ``v`` by ``angle_deg`` about an axis perpendicular to
    it, preserving ``|v|`` and making the angle between input and output
    exactly ``angle_deg`` (up to floating point)."""
    u = v / np.linalg.norm(v)
    w = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, w)) > 0.9:
        w = np.array([0.0, 1.0, 0.0])
    k = np.cross(u, w)
    k /= np.linalg.norm(k)
    theta = np.deg2rad(angle_deg)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta)


def test_calc_angle_errors_returns_angle_pos_ome_order(monkeypatch):
    """err_ini must be (angle, pos, ome) with each value distinguishable by
    its known, filter-implied bound — not silently permuted."""
    y_orig, z_orig, omega_ini = 300.0, 150.0, 12.0

    # Ground-truth "observed, corrected" spot: grain position (0,0,0) makes
    # _displacement_in_the_spot exactly (0, 0), so the corrected spot is
    # just _correct_for_ome applied directly to the raw (y_orig, z_orig).
    ys0, zs0, omega_corr0, g1_0, g2_0, g3_0 = c_port._correct_for_ome(
        y_orig, z_orig, LSD, omega_ini, WAVELENGTH, wedge_deg=0.0,
    )

    g_obs = np.array([g1_0, g2_0, g3_0])
    g_th = _rotate_by_small_angle(g_obs, ANGLE_ERR_DEG)

    theor_spots = np.array([[
        ys0 + POS_ERR_UM, zs0, omega_corr0 + OME_ERR_DEG,
        g_th[0], g_th[1], g_th[2],
        LSD, float(RING_NR), 0.0,
    ]])

    monkeypatch.setattr(
        c_port, "calc_diffr_spots_furnace",
        lambda *a, **k: theor_spots,
    )

    spots_yzo = np.array([[
        ys0, zs0, omega_corr0, 1.0,       # YLab, ZLab, Omega, SpotID
        omega_ini, y_orig, z_orig,         # OmegaIni, YOrig, ZOrig
        float(RING_NR), 0.0, 0.0,          # RingNr, maskTouched, FitRMSE
    ]])

    spots_comp, spots_yzog, err_ini, n_matched = c_port.calc_angle_errors(
        pos=[0.0, 0.0, 0.0],
        orient_mat=np.eye(3),
        lat_c=[4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
        spots_yzo=spots_yzo,
        hkls_int=np.array([[1, 0, 0]], dtype=np.int64),
        ring_nr_per_hkl=np.array([RING_NR], dtype=np.int64),
        lsd=LSD, wavelength=WAVELENGTH,
        omega_ranges=np.zeros((1, 2)), box_sizes=np.zeros((1, 4)),
        min_eta=0.0, wedge_deg=0.0,
    )

    assert n_matched == 1
    mean_angle_deg, mean_pos_um, mean_ome_deg = err_ini
    assert mean_angle_deg == pytest.approx(ANGLE_ERR_DEG, abs=1e-6)
    assert mean_pos_um == pytest.approx(POS_ERR_UM, abs=1e-6)
    assert mean_ome_deg == pytest.approx(OME_ERR_DEG, abs=1e-6)

    # Boundedness the driver relies on to tell the three apart: angle < 1,
    # ome < 5, position unbounded by any matching filter.
    assert mean_angle_deg < 1.0
    assert mean_ome_deg < 5.0
    assert mean_pos_um > 5.0

    # The driver's GrainResult mapping (mirrors driver.py's post-fix
    # unpacking) must land each value in the OrientPosFit.bin column whose
    # Grains.csv header claims it — this is what a naive
    # ErrorPos=err_ini[0]/ErrorOme=err_ini[1]/ErrorAngle=err_ini[2] mapping
    # would get wrong.
    grain_result = GrainResult(
        SpotID=1, OrientMat=np.eye(3).reshape(-1),
        Position=np.zeros(3), LatticeFit=np.array([4.0, 4.0, 4.0, 90, 90, 90]),
        ErrorPos=mean_pos_um, ErrorOme=mean_ome_deg, ErrorAngle=mean_angle_deg,
        meanRadius=0.0, completeness=1.0,
    )
    row = grain_result.to_row()
    # col 22 = DiffPos in Grains.csv: must be the (unbounded) position error.
    assert row[22] == pytest.approx(POS_ERR_UM, abs=1e-6)
    # col 23 = DiffOme in Grains.csv: must be the omega error (< 5 deg).
    assert row[23] == pytest.approx(OME_ERR_DEG, abs=1e-6)
    # col 24 = DiffAngle in Grains.csv: must be the internal angle (< 1 deg).
    assert row[24] == pytest.approx(ANGLE_ERR_DEG, abs=1e-6)
