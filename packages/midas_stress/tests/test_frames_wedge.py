"""Tests for the Wedge (rotation-stage tilt) term in frames.py.

Issue #24: ``lab_to_sample_rotation`` / ``grains_midas_to_sample`` rotated
orientation and position by a pure ``R_z(omega)`` (midas) / ``R_y(omega)``
(aps) with no wedge term, which disagrees with the tilted-axis convention
used everywhere else in MIDAS (``midas_diffract.forward``, the FF/NF/pf C
code) once Wedge != 0.

These tests check:
  (a) ``wedge_deg=0.0`` (the default) is bit-identical to the pre-fix
      behaviour -- no observable change for any existing caller.
  (b) the new tilted-axis rotation, for BOTH the "midas" (rotation nominally
      about Z) and "aps" (rotation nominally about Y) conventions, agrees to
      ~1e-9 with ``midas_diffract.forward.HEDMForwardModel``'s own
      stage-to-lab rotation (``_rotate_positions`` / ``calc_bragg_geometry``,
      which implement ``pos_lab = R_y(-Wedge) @ R_z(omega) @ pos`` in the
      MIDAS frame -- the "ONE MIDAS convention" documented at the top of
      ``midas_diffract/forward.py``).
  (c) orientation and position keep getting the SAME rotation as each other
      (the invariant ``grains_midas_to_sample`` is built to preserve --
      see its docstring and ``TestGrainsPipeline`` in test_frames.py), now
      also under a nonzero wedge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.frames import (
    R_MIDAS_TO_APS,
    grains_midas_to_sample,
    lab_to_sample_rotation,
)

torch = pytest.importorskip("torch")
diffract_forward = pytest.importorskip("midas_diffract.forward")


def _build_model(wedge_deg: float):
    """Minimal HEDMForwardModel, just to reach its wedge rotation code."""
    geom = diffract_forward.HEDMGeometry(
        Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=1.0, n_frames=180,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=2.0,
        wavelength=0.1729, wedge=wedge_deg,
    )
    hkls = torch.zeros((1, 3), dtype=torch.float64)
    thetas = torch.zeros((1,), dtype=torch.float64)
    return diffract_forward.HEDMForwardModel(hkls, thetas, geom)


def _diffract_stage_to_lab_midas(model, omega_deg: float, vec: np.ndarray) -> np.ndarray:
    """``pos_lab = R_y(-Wedge) @ R_z(omega) @ vec`` via the real forward model.

    Uses ``HEDMForwardModel._rotate_positions``, the exact map the module
    doc calls "the ONE MIDAS convention, shared by FF, NF and PF" and which
    it applies identically to G-vectors and grain positions.
    """
    omega_rad = torch.tensor(math.radians(omega_deg), dtype=torch.float64)
    cos_w = torch.cos(omega_rad)
    sin_w = torch.sin(omega_rad)
    px_t = torch.tensor(float(vec[0]), dtype=torch.float64)
    py_t = torch.tensor(float(vec[1]), dtype=torch.float64)
    pz_t = torch.tensor(float(vec[2]), dtype=torch.float64)
    x, y, z = model._rotate_positions(px_t, py_t, pz_t, cos_w, sin_w)
    return np.array([float(x), float(y), float(z)])


# ---------------------------------------------------------------------
# (a) wedge_deg=0.0 is bit-identical to the pre-fix behaviour
# ---------------------------------------------------------------------

class TestWedgeZeroBackwardCompatible:
    @pytest.mark.parametrize("frame", ["midas", "aps", "esrf"])
    @pytest.mark.parametrize("omega_deg", [0.0, 1.0, -37.5, 90.0, 179.9, -123.4])
    def test_default_matches_explicit_zero(self, frame, omega_deg):
        R_default = lab_to_sample_rotation(omega_deg, frame)
        R_explicit = lab_to_sample_rotation(omega_deg, frame, wedge_deg=0.0)
        np.testing.assert_array_equal(R_default, R_explicit)

    @pytest.mark.parametrize("frame", ["midas", "aps"])
    def test_matches_pre_fix_formula(self, frame):
        """Reproduce the exact pre-fix (no-wedge) matrices and compare."""
        rng = np.random.default_rng(1)
        for omega_deg in rng.uniform(-180, 180, size=20):
            c = math.cos(math.radians(omega_deg))
            s = math.sin(math.radians(omega_deg))
            if frame == "aps":
                R_old = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
            else:
                R_old = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            R_new = lab_to_sample_rotation(omega_deg, frame, wedge_deg=0.0)
            np.testing.assert_allclose(R_new, R_old, atol=1e-15)

    def test_grains_midas_to_sample_default_matches_explicit_zero(self):
        rng = np.random.default_rng(2)
        N = 6
        U = Rotation.random(N, random_state=3).as_matrix()
        pos = rng.normal(0, 100, (N, 3))
        eps = rng.normal(0, 1e-3, (N, 3, 3))
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))

        out_default = grains_midas_to_sample(U, pos, eps, omega_deg=33.0, target_frame="aps")
        out_explicit = grains_midas_to_sample(
            U, pos, eps, omega_deg=33.0, target_frame="aps", wedge_deg=0.0
        )
        for key in ("orientations", "positions", "strains"):
            np.testing.assert_array_equal(out_default[key], out_explicit[key])


# ---------------------------------------------------------------------
# (b) nonzero wedge cross-checked against midas_diffract to ~1e-9
# ---------------------------------------------------------------------

class TestWedgeCrossCheckMidasDiffract:
    def test_midas_frame_roundtrip_against_diffract(self):
        """R_lab_to_sample(midas) must exactly undo midas_diffract's
        stage-to-lab rotation for the SAME (omega, wedge)."""
        rng = np.random.default_rng(4)
        max_err = 0.0
        for omega_deg in rng.uniform(-180, 180, size=15):
            wedge_deg = float(rng.uniform(-15, 15))
            model = _build_model(wedge_deg)
            vec = rng.uniform(-500, 500, size=3)

            pos_lab = _diffract_stage_to_lab_midas(model, float(omega_deg), vec)

            R_l2s = lab_to_sample_rotation(float(omega_deg), "midas", wedge_deg)
            pos_sample = R_l2s @ pos_lab
            err = np.max(np.abs(pos_sample - vec))
            max_err = max(max_err, err)

        assert max_err < 1e-9, f"midas-frame wedge round-trip error {max_err:.3e}"

    def test_aps_frame_roundtrip_against_diffract(self):
        """Same check, but going through the MIDAS->APS permutation: the
        physical stage tilt is the SAME wedge; midas_diffract only knows
        the MIDAS-frame formula, so permute its output into APS coordinates
        before comparing against frames.py's 'aps' convention."""
        rng = np.random.default_rng(5)
        max_err = 0.0
        for omega_deg in rng.uniform(-180, 180, size=15):
            wedge_deg = float(rng.uniform(-15, 15))
            model = _build_model(wedge_deg)
            vec_midas = rng.uniform(-500, 500, size=3)

            pos_lab_midas = _diffract_stage_to_lab_midas(model, float(omega_deg), vec_midas)
            # Permute both the stage vector and the lab result into APS coords.
            vec_aps = R_MIDAS_TO_APS @ vec_midas
            pos_lab_aps = R_MIDAS_TO_APS @ pos_lab_midas

            R_l2s_aps = lab_to_sample_rotation(float(omega_deg), "aps", wedge_deg)
            pos_sample_aps = R_l2s_aps @ pos_lab_aps
            err = np.max(np.abs(pos_sample_aps - vec_aps))
            max_err = max(max_err, err)

        assert max_err < 1e-9, f"aps-frame wedge round-trip error {max_err:.3e}"

    def test_zero_wedge_axis_is_unperturbed(self):
        """Sanity check the documented axis (-sin W, 0, cos W): at omega=0
        the midas-frame lab_to_sample rotation reduces to R_y(wedge)."""
        wedge_deg = 12.0
        R = lab_to_sample_rotation(0.0, "midas", wedge_deg)
        w = math.radians(wedge_deg)
        R_y_expected = np.array([
            [math.cos(w), 0.0, math.sin(w)],
            [0.0, 1.0, 0.0],
            [-math.sin(w), 0.0, math.cos(w)],
        ])
        np.testing.assert_allclose(R, R_y_expected, atol=1e-14)


# ---------------------------------------------------------------------
# (c) orientation and position remain consistently related under wedge
# ---------------------------------------------------------------------

class TestOrientationPositionInvariantUnderWedge:
    def test_same_R_total_applied_to_orientation_and_position(self):
        """grains_midas_to_sample must apply the identical R_total to the
        orientation matrix and to the position vector (the self-consistency
        the issue explicitly asks to preserve)."""
        rng = np.random.default_rng(6)
        N = 4
        U = Rotation.random(N, random_state=8).as_matrix()
        pos = rng.normal(0, 100, (N, 3))
        eps = rng.normal(0, 1e-3, (N, 3, 3))
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))
        omega_deg, wedge_deg = 22.5, 6.0

        out = grains_midas_to_sample(
            U, pos, eps, omega_deg=omega_deg, target_frame="aps", wedge_deg=wedge_deg
        )

        R_frame = R_MIDAS_TO_APS
        R_lab2sam = lab_to_sample_rotation(omega_deg, "aps", wedge_deg)
        R_total = R_lab2sam @ R_frame

        np.testing.assert_allclose(out["orientations"], R_total @ U, atol=1e-13)
        expected_pos = np.einsum("ij,nj->ni", R_total, pos)
        np.testing.assert_allclose(out["positions"], expected_pos, atol=1e-11)

    def test_orientation_and_position_use_same_rotation_regardless_of_wedge(self):
        """Applying R_total (recovered from the position transform) to a
        pure-rotation identity orientation should reproduce the orientation
        output exactly -- i.e. one rotation drives both outputs."""
        rng = np.random.default_rng(9)
        N = 1
        U = np.eye(3)[np.newaxis].repeat(N, axis=0)
        pos = rng.normal(0, 100, (N, 3))
        eps = np.zeros((N, 3, 3))

        for wedge_deg in (0.0, -8.0, 15.0):
            out = grains_midas_to_sample(
                U, pos, eps, omega_deg=50.0, target_frame="midas", wedge_deg=wedge_deg
            )
            R_lab2sam = lab_to_sample_rotation(50.0, "midas", wedge_deg)
            # target_frame="midas" => R_frame = I, so R_total == R_lab2sam.
            np.testing.assert_allclose(out["orientations"][0], R_lab2sam, atol=1e-13)
            np.testing.assert_allclose(out["positions"][0], R_lab2sam @ pos[0], atol=1e-11)
