"""Tests for recover_d0_anisotropic — symmetry-aware strain-free reference recovery.

The isotropic recover_d0 scales a, b and c by one factor, which is exact only for
cubic phases.  These tests plant a genuinely anisotropic reference error (a too
small while c is too large — the case that no single scale can absorb) and check
that it comes back.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.equilibrium import (
    recover_d0, recover_d0_anisotropic,
)
from midas_stress.materials import hexagonal_stiffness, cubic_stiffness
from midas_stress.tensor import lattice_params_to_strain


# NMC811-like hexagonal stiffness (GPa); values only need to be a valid
# positive-definite hexagonal tensor for these tests.
C_HEX = hexagonal_stiffness(C11=242.0, C12=76.0, C13=48.0, C33=196.0, C44=46.0)


def _orients(n, seed=0):
    """Uniform random orientations -> weak texture -> well conditioned."""
    return Rotation.random(n, random_state=seed).as_matrix()


def _cells(reference, n):
    """n grains all sitting exactly at `reference` (so <sigma> = 0 exactly)."""
    return np.tile(np.asarray(reference, float), (n, 1))


def test_recovers_planted_anisotropic_reference():
    """a too small AND c too large — the case isotropic recovery cannot fix."""
    true_ref = np.array([2.8500, 2.8500, 14.3200, 90.0, 90.0, 120.0])
    # assumed: a +0.63 % high, c -0.75 % low  (opposite signs)
    assumed = true_ref.copy()
    assumed[0] = assumed[1] = true_ref[0] * 1.0063
    assumed[2] = true_ref[2] * 0.9925

    n = 400
    orients = _orients(n, seed=1)
    lat = _cells(true_ref, n)

    out = recover_d0_anisotropic(
        lat, assumed, C_HEX, orients, crystal_system="hexagonal")

    assert out["well_conditioned"], out["condition_number"]
    rec = out["reference_recovered"]
    assert rec[0] == pytest.approx(true_ref[0], rel=2e-4)
    assert rec[1] == pytest.approx(true_ref[1], rel=2e-4)
    assert rec[2] == pytest.approx(true_ref[2], rel=2e-4)
    # angles are never fitted
    assert rec[3:] == pytest.approx(assumed[3:])
    # and equilibrium is better satisfied afterwards
    assert out["residual_norm_after"] < 1e-3 * out["residual_norm_before"]


def test_isotropic_version_cannot_fix_anisotropic_error():
    """Motivation check: recover_d0 leaves a large residual where this does not."""
    true_ref = np.array([2.8500, 2.8500, 14.3200, 90.0, 90.0, 120.0])
    assumed = true_ref.copy()
    assumed[0] = assumed[1] = true_ref[0] * 1.0063
    assumed[2] = true_ref[2] * 0.9925

    n = 300
    orients = _orients(n, seed=2)
    lat = _cells(true_ref, n)
    vols = np.ones(n)

    iso = recover_d0(lat, assumed, C_HEX, orients, vols)
    ani = recover_d0_anisotropic(lat, assumed, C_HEX, orients, vols,
                                 crystal_system="hexagonal")

    assert ani["residual_norm_after"] < iso["residual_norm_after"]
    # the isotropic fit cannot bring both lengths home
    iso_err = max(abs(iso["reference_recovered"][0] / true_ref[0] - 1),
                  abs(iso["reference_recovered"][2] / true_ref[2] - 1))
    ani_err = max(abs(ani["reference_recovered"][0] / true_ref[0] - 1),
                  abs(ani["reference_recovered"][2] / true_ref[2] - 1))
    assert ani_err < iso_err / 10


def test_zero_error_is_a_fixed_point():
    """A correct reference must be returned unchanged."""
    ref = np.array([2.8500, 2.8500, 14.3200, 90.0, 90.0, 120.0])
    n = 200
    out = recover_d0_anisotropic(
        _cells(ref, n), ref, C_HEX, _orients(n, seed=3),
        crystal_system="hexagonal")
    assert out["reference_recovered"] == pytest.approx(ref, rel=1e-9, abs=1e-9)
    for v in out["deltas"].values():
        assert abs(v) < 1e-9


def test_cubic_matches_isotropic_recover_d0():
    """For a cubic phase the one-parameter solve must agree with recover_d0."""
    C = cubic_stiffness(240.0, 150.0, 120.0)
    true_ref = np.array([3.6000, 3.6000, 3.6000, 90.0, 90.0, 90.0])
    assumed = true_ref * np.array([1.004, 1.004, 1.004, 1, 1, 1])
    assumed[3:] = 90.0

    n = 250
    orients = _orients(n, seed=4)
    lat = _cells(true_ref, n)

    iso = recover_d0(lat, assumed, C, orients, np.ones(n))
    ani = recover_d0_anisotropic(lat, assumed, C, orients,
                                 crystal_system="cubic")
    assert ani["reference_recovered"][:3] == pytest.approx(
        iso["reference_recovered"][:3], rel=1e-6)
    assert ani["reference_recovered"][0] == pytest.approx(true_ref[0], rel=2e-4)


def test_weak_texture_is_the_ill_conditioned_case_not_sharp():
    """Uniform texture washes out the a/c split; a single orientation does not.

    Averaging the Mandel rotation over uniform orientations projects onto the
    isotropic subspace, so both columns tend to C{I} and become collinear —
    and the effect grows as the average converges with N.  A single crystal
    separates them cleanly because its stiffness is anisotropic.
    """
    ref = np.array([2.8500, 2.8500, 14.3200, 90.0, 90.0, 120.0])

    single = recover_d0_anisotropic(
        _cells(ref, 100), ref, C_HEX, np.tile(np.eye(3), (100, 1, 1)),
        crystal_system="hexagonal")
    uni_small = recover_d0_anisotropic(
        _cells(ref, 100), ref, C_HEX, _orients(100, seed=7),
        crystal_system="hexagonal")
    uni_large = recover_d0_anisotropic(
        _cells(ref, 1000), ref, C_HEX, _orients(1000, seed=7),
        crystal_system="hexagonal")

    # sharp texture is the easy case
    assert single["condition_number"] < 10
    assert single["well_conditioned"]
    # uniform texture is worse, and worsens as the orientation average converges
    assert uni_small["condition_number"] > single["condition_number"]
    assert uni_large["condition_number"] > uni_small["condition_number"]


def test_recovery_survives_realistic_per_grain_scatter():
    """Uniform texture + 500 ue of real strain must still recover the reference."""
    true_ref = np.array([2.8500, 2.8500, 14.3200, 90.0, 90.0, 120.0])
    assumed = true_ref.copy()
    assumed[0] = assumed[1] = true_ref[0] * 1.0063
    assumed[2] = true_ref[2] * 0.9925

    n = 1000
    rng = np.random.default_rng(0)
    lat = np.tile(true_ref, (n, 1)).copy()
    lat[:, :3] *= 1 + rng.normal(0, 500e-6, (n, 3))

    out = recover_d0_anisotropic(
        lat, assumed, C_HEX, _orients(n, seed=11), crystal_system="hexagonal")
    rec = out["reference_recovered"]
    # the planted error was ~6300 / 7500 ue; recovery must land far inside it
    assert abs(rec[0] / true_ref[0] - 1) < 1e-3
    assert abs(rec[2] / true_ref[2] - 1) < 1e-3


def test_orthorhombic_three_parameters():
    ref = np.array([4.0, 5.0, 6.0, 90.0, 90.0, 90.0])
    C = cubic_stiffness(240.0, 150.0, 120.0)        # any valid tensor
    assumed = ref * np.array([1.003, 0.997, 1.005, 1, 1, 1])
    assumed[3:] = 90.0
    n = 400
    out = recover_d0_anisotropic(
        _cells(ref, n), assumed, C, _orients(n, seed=5),
        crystal_system="orthorhombic")
    assert set(out["deltas"]) == {"a", "b", "c"}
    assert out["reference_recovered"][:3] == pytest.approx(ref[:3], rel=5e-4)


def test_rejects_unknown_system_and_too_few_grains():
    ref = np.array([2.85, 2.85, 14.32, 90.0, 90.0, 120.0])
    with pytest.raises(ValueError, match="unknown crystal_system"):
        recover_d0_anisotropic(_cells(ref, 10), ref, C_HEX, _orients(10),
                               crystal_system="dodecahedral")
    with pytest.raises(ValueError, match="must be determined"):
        recover_d0_anisotropic(
            _cells(ref, 10), ref, C_HEX, _orients(10),
            crystal_system="hexagonal",
            confidences=np.zeros(10), min_confidence=0.5)
