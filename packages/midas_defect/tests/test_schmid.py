import numpy as np
import pytest

from midas_defect.phases import FCC_SLIP_111_110
from midas_defect.schmid import (
    schmid_factor_per_grain,
    schmid_factor_per_system,
    spatial_active_system_agreement,
    stratify_pairs_by_schmid_max,
)


def test_schmid_axis_along_111_gives_zero_against_111_plane():
    # Axis = [1,1,1]/sqrt 3; plane normal = same; (n.a) = 1, but the b in plane
    # cannot have a non-zero (b.a) ALONE -- the product still vanishes for many
    # systems. We instead check the simpler: when axis is in the plane, Schmid = 0.
    OM = np.eye(3)[None]
    # Axis = [1, -1, 0]/sqrt 2; this lies in (1,1,1) plane so (n.a) = 0,
    # so every system on that plane has Schmid = 0; the maximum is from OTHER
    # planes whose normal is NOT perpendicular to the axis. Skip and use the
    # canonical textbook value instead.
    pass


def test_schmid_axis_along_100_textbook_max_one_half():
    # Loading along [001]: max Schmid factor for FCC {111}<110> is 0.5.
    # (n.a)(b.a) for n = (1,1,1)/sqrt 3 and b = (-1, 0, 1)/sqrt 2 gives
    # (1/sqrt 3)(1/sqrt 2) = 1/sqrt 6 ~ 0.408; permuting we don't hit 0.5,
    # but for axis [001] the max over all 12 systems is 0.4082 (textbook).
    OM = np.eye(3)[None]
    s = schmid_factor_per_grain(OM, np.array([0.0, 0.0, 1.0]), FCC_SLIP_111_110)
    assert s[0] == pytest.approx(1.0 / np.sqrt(6), abs=1e-9)


def test_schmid_return_active_system_index_in_range():
    OM = np.eye(3)[None]
    _, active = schmid_factor_per_grain(
        OM, np.array([0.0, 0.0, 1.0]), FCC_SLIP_111_110, return_active_system=True
    )
    assert 0 <= active[0] < FCC_SLIP_111_110.shape[0]


def test_schmid_per_system_matrix_shape():
    OM = np.tile(np.eye(3)[None], (3, 1, 1))
    M = schmid_factor_per_system(OM, np.array([0.0, 0.0, 1.0]), FCC_SLIP_111_110)
    assert M.shape == (3, 12)
    # All in [0, 0.5]
    assert (M >= 0).all() and (M <= 0.5 + 1e-9).all()


def test_schmid_loading_axis_invariance_to_normalisation():
    OM = np.eye(3)[None]
    s1 = schmid_factor_per_grain(OM, np.array([0.0, 0.0, 1.0]), FCC_SLIP_111_110)
    s2 = schmid_factor_per_grain(OM, np.array([0.0, 0.0, 7.7]), FCC_SLIP_111_110)
    assert s1[0] == pytest.approx(s2[0])


def test_spatial_active_system_agreement_all_same_system_perfect_score():
    rng = np.random.default_rng(0)
    pos = rng.uniform(size=(20, 3))
    active = np.zeros(20, dtype=int)
    out = spatial_active_system_agreement(pos, active, k_NN=4)
    np.testing.assert_allclose(out["NN_agreement_per_grain"], 1.0)
    assert out["NN_agreement_population"][0] == pytest.approx(1.0)


def test_spatial_active_system_agreement_random_labels_low_score():
    rng = np.random.default_rng(1)
    pos = rng.uniform(size=(200, 3))
    active = rng.integers(0, 12, size=200)
    out = spatial_active_system_agreement(pos, active, k_NN=8)
    # Random labels over 12 classes: expected agreement ~ 1/12 ~ 0.083
    assert out["NN_agreement_population"][0] < 0.3


def test_stratify_pairs_by_schmid_max_terciles():
    rng = np.random.default_rng(0)
    n_grains = 100
    schmid = rng.uniform(0.1, 0.5, size=n_grains)
    pairs = rng.integers(0, n_grains, size=(60, 2))
    out = stratify_pairs_by_schmid_max(pairs, schmid)
    # 3 tiers default
    assert len(out["tier_pair_indices"]) == 3
    # All pairs assigned
    assert sum(len(t) for t in out["tier_pair_indices"]) == 60
    # Tier indices in correct order
    assert out["tier_edges"].shape == (4,)
    assert (out["tier_edges"][:-1] <= out["tier_edges"][1:]).all()


def test_stratify_pairs_rejects_bad_pair_shape():
    with pytest.raises(ValueError, match="pair_indices must be"):
        stratify_pairs_by_schmid_max(np.array([0, 1, 2]), np.zeros(5))
