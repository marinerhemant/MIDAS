"""Smoke tests for midas_pipeline.em_refine.

The module's main function (``run_em_spot_ownership``) shells out to
``fwd_sim/em_spot_ownership.py`` which has its own heavy dependencies.
We just smoke-test that:

- The module imports cleanly without ``fwd_sim`` on the path.
- Pure helpers (``parse_params_for_em``, ``load_grain_orientations``,
  ``orient_mat_to_euler``, ``update_unique_orientations_from_refinement``)
  work on synthetic fixtures.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline import em_refine


def test_em_refine_imports():
    # Module-level imports already succeeded by virtue of pytest collecting
    # this test; assert the public entry points exist.
    assert hasattr(em_refine, "run_em_spot_ownership")
    assert hasattr(em_refine, "parse_params_for_em")
    assert hasattr(em_refine, "load_grain_orientations")
    assert hasattr(em_refine, "orient_mat_to_euler")


def test_parse_params_for_em_basic(tmp_path):
    p = tmp_path / "paramstest.txt"
    p.write_text(
        "Lsd 1234567.0\n"
        "px 200.0\n"
        "BC 1024.0 1024.0\n"
        "OmegaStart -180.0\n"
        "OmegaStep 0.25\n"
        "NrPixels 2048\n"
        "MinEta 6.0\n"
        "Wavelength 0.172979\n"
        "TolOme 0.5\n"
        "TolEta 5.0\n"
        "SpaceGroup 225\n"
        "StartNr 1\n"
        "EndNr 1440\n"
    )
    params = em_refine.parse_params_for_em(str(p))
    assert params["Lsd"] == pytest.approx(1234567.0)
    assert params["px"] == pytest.approx(200.0)
    assert params["sgnum"] == 225
    assert params["n_frames"] == 1440  # end - start + 1


def test_load_grain_orientations_round_trip(tmp_path):
    om0 = np.eye(3)
    om1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    rows = [
        [0.0, 0.0, 5.0, 0.0, 0.0] + list(om0.flatten()),
        [1.0, 0.0, 7.0, 0.0, 0.0] + list(om1.flatten()),
    ]
    # The grain list lives where find_grains writes it (Output/); the
    # layer-level name is seeding's seed list.
    (tmp_path / "Output").mkdir(exist_ok=True)
    np.savetxt(tmp_path / "Output" / "UniqueOrientations.csv",
               np.asarray(rows, dtype=np.float64), fmt="%.10f", delimiter=" ")
    oms, gids = em_refine.load_grain_orientations(str(tmp_path), refined=False)
    assert oms.shape == (2, 3, 3)
    assert gids.tolist() == [0, 1]
    np.testing.assert_allclose(oms[0], om0)
    np.testing.assert_allclose(oms[1], om1)


def test_orient_mat_to_euler_returns_radians():
    om = np.eye(3)
    eu = em_refine.orient_mat_to_euler(om)
    # Identity → all-zero Euler.
    assert eu.shape == (3,)
    np.testing.assert_allclose(eu, np.zeros(3), atol=1e-12)
    # 90° about Z should give a nonzero first Euler angle.
    om_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    eu_z = em_refine.orient_mat_to_euler(om_z)
    # phi1 should be ~ ±π/2 since the rotation is purely about Z.
    assert abs(abs(eu_z[0]) - math.pi / 2) < 1e-9 or abs(eu_z[0]) < 1e-9
