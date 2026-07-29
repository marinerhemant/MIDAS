import numpy as np
import pytest

import midas_stress.orientation as o

from midas_defect.gnd import (
    per_grain_nye_tensor,
    scalar_gnd_from_inter_grain_misorientation,
    ssd_gnd_decomposition,
)
from midas_defect.types import CrystalPhase


_BURGERS_CU = 2.57e-10  # m


def _identity_population(n: int):
    """All grains at identity, evenly spaced -- zero GND ground truth."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, size=(n, 3))
    OM = np.tile(np.eye(3)[None], (n, 1, 1))
    return OM, pos


def _tilt_gradient_population(n: int, tilt_per_um_deg: float):
    """Continuous tilt about z axis as a linear function of x position."""
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 50, size=(n, 3))
    angles_deg = pos[:, 0] * tilt_per_um_deg
    axis_z = np.array([0.0, 0.0, 1.0])
    OM = np.stack(
        [np.asarray(o.axis_angle_to_orient_mat(axis_z, a)) for a in angles_deg],
        axis=0,
    )
    return OM, pos


# -- Scalar GND -------------------------------------------------------------

def test_scalar_gnd_zero_for_identity_population():
    OM, pos = _identity_population(30)
    rho = scalar_gnd_from_inter_grain_misorientation(
        OM, pos, burgers_length=_BURGERS_CU, k_NN=5, phase=CrystalPhase.FCC
    )
    finite = np.isfinite(rho)
    assert finite.any()
    np.testing.assert_allclose(rho[finite], 0.0, atol=1e-3)


def test_scalar_gnd_nonzero_under_tilt_gradient():
    OM, pos = _tilt_gradient_population(50, tilt_per_um_deg=0.05)
    rho = scalar_gnd_from_inter_grain_misorientation(
        OM, pos, burgers_length=_BURGERS_CU, k_NN=5, phase=CrystalPhase.FCC
    )
    finite = np.isfinite(rho)
    assert finite.any()
    assert (rho[finite] > 0).all()
    # Order of magnitude: tilt ~ 0.05 deg/um * 1 um = 0.05 deg = 8.7e-4 rad
    # rho ~ ang / (d * b) = 8.7e-4 / (1e-6 * 2.57e-10) ~ 3e12 m^-2; expect ~ 10^12.
    assert 1e11 < np.median(rho[finite]) < 1e14


def test_scalar_gnd_variant_filter_restricts_neighbours():
    rng = np.random.default_rng(0)
    n = 40
    pos = rng.uniform(0, 100, size=(n, 3))
    OM = np.tile(np.eye(3)[None], (n, 1, 1))
    var = (np.arange(n) >= n // 2).astype(int)
    # Identity orientations -> rho should still be ~0 regardless of variant.
    rho = scalar_gnd_from_inter_grain_misorientation(
        OM, pos, burgers_length=_BURGERS_CU, variant_labels=var
    )
    finite = np.isfinite(rho)
    assert finite.any()
    np.testing.assert_allclose(rho[finite], 0.0, atol=1e-3)


# -- Nye tensor -------------------------------------------------------------

def test_nye_tensor_zero_for_identity_population():
    OM, pos = _identity_population(40)
    out = per_grain_nye_tensor(OM, pos, burgers_length=_BURGERS_CU, k_NN=8)
    rho = out["rho_GND_per_grain"]
    finite = np.isfinite(rho)
    # Up to numerical noise on the LS fit.
    assert finite.any()
    assert np.median(rho[finite]) < 1e8  # essentially zero on the GND scale


def test_nye_tensor_nonzero_under_tilt_gradient():
    OM, pos = _tilt_gradient_population(80, tilt_per_um_deg=0.05)
    out = per_grain_nye_tensor(OM, pos, burgers_length=_BURGERS_CU, k_NN=10)
    rho = out["rho_GND_per_grain"]
    finite = np.isfinite(rho)
    assert finite.any()
    assert np.median(rho[finite]) > 1e11


def test_nye_tensor_too_few_grains_returns_all_nan():
    OM = np.tile(np.eye(3)[None], (3, 1, 1))
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    out = per_grain_nye_tensor(OM, pos, burgers_length=_BURGERS_CU, k_NN=5)
    assert np.isnan(out["alpha_per_grain"]).all()


# -- SSD/GND decomposition --------------------------------------------------

def test_ssd_decomposition_basic():
    total = np.array([1e14, 5e13, 2e13])
    gnd = np.array([6e13, 1e13, 3e13])  # last > total -> clipped at 0
    out = ssd_gnd_decomposition(total, gnd)
    np.testing.assert_allclose(out["rho_SSD_per_grain"], [4e13, 4e13, 0.0])
    np.testing.assert_allclose(out["ssd_fraction_per_grain"], [0.4, 0.8, 0.0])
    np.testing.assert_allclose(out["gnd_fraction_per_grain"], [0.6, 0.2, 1.5])


def test_ssd_decomposition_nan_when_total_zero():
    out = ssd_gnd_decomposition(np.array([0.0]), np.array([0.0]))
    assert np.isnan(out["ssd_fraction_per_grain"][0])
    assert np.isnan(out["gnd_fraction_per_grain"][0])


def test_ssd_decomposition_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        ssd_gnd_decomposition(np.zeros(3), np.zeros(4))
