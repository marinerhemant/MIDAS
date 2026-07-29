import numpy as np
import pytest

import midas_stress.orientation as o

from midas_defect.types import CrystalPhase
from midas_defect.variants import (
    assign_variants_common_reference,
    assign_variants_kmeans,
    build_sigma3_pair,
    find_sigma3_partners,
)


def _random_unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _quat_to_om(q):
    return np.asarray(o.quat_to_orient_mat(q)).reshape(3, 3)


def _two_cluster_population(rng, n=40, jitter_deg=2.0):
    """Two clusters, one around identity, one around Sigma3."""
    # Cluster A: small rotations around identity.
    axis_a = rng.normal(size=(n, 3))
    axis_a /= np.linalg.norm(axis_a, axis=1, keepdims=True)
    om_a = np.stack(
        [np.asarray(o.axis_angle_to_orient_mat(axis_a[i], rng.uniform(0, jitter_deg)))
         for i in range(n)], axis=0,
    )

    # Cluster B: 60 deg about [111] then small jitter.
    axis_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    om_sigma3 = np.asarray(o.axis_angle_to_orient_mat(axis_111, 60.0))
    axis_b = rng.normal(size=(n, 3))
    axis_b /= np.linalg.norm(axis_b, axis=1, keepdims=True)
    om_b = np.stack(
        [om_sigma3 @ np.asarray(o.axis_angle_to_orient_mat(axis_b[i], rng.uniform(0, jitter_deg)))
         for i in range(n)], axis=0,
    )

    OM = np.concatenate([om_a, om_b], axis=0)
    true_labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return OM, true_labels


# -- K-means ----------------------------------------------------------------

def test_assign_variants_kmeans_recovers_two_clusters_synthetically():
    rng = np.random.default_rng(0)
    OM, true_labels = _two_cluster_population(rng, n=30, jitter_deg=2.0)
    out = assign_variants_kmeans(OM, n_variants=2, n_init=5, random_state=0)
    assert out["labels"].shape == (60,)
    # Accuracy invariant to label permutation: take max over the two
    # possible labellings.
    acc1 = (out["labels"] == true_labels).mean()
    acc2 = (out["labels"] == 1 - true_labels).mean()
    assert max(acc1, acc2) > 0.95
    # Sigma3 disorientation: between cluster centroids, should be ~60 deg.
    assert 55.0 < float(out["disorientations"][0, 1]) < 65.0
    # Counts add up
    assert out["counts"].sum() == 60


def test_assign_variants_kmeans_rejects_too_few_grains():
    OM = np.tile(np.eye(3)[None], (3, 1, 1))
    with pytest.raises(ValueError, match="at least n_variants"):
        assign_variants_kmeans(OM, n_variants=5)


def test_assign_variants_kmeans_unknown_phase_raises():
    OM = np.tile(np.eye(3)[None], (4, 1, 1))
    class FakePhase:
        pass

    with pytest.raises(ValueError, match="unknown phase"):
        assign_variants_kmeans(OM, n_variants=2, phase=FakePhase())  # type: ignore[arg-type]


# -- Common reference -------------------------------------------------------

def test_assign_variants_common_reference_picks_closest():
    rng = np.random.default_rng(1)
    OM, true_labels = _two_cluster_population(rng, n=20, jitter_deg=2.0)
    R_matrix = np.eye(3)
    R_twin = np.asarray(
        o.axis_angle_to_orient_mat(np.array([1, 1, 1]) / np.sqrt(3), 60.0)
    )
    labels = assign_variants_common_reference(
        OM, [R_matrix, R_twin], phase=CrystalPhase.FCC
    )
    acc = (labels == true_labels).mean()
    assert acc > 0.95


def test_build_sigma3_pair_fcc_disorientation_is_60_deg():
    R = np.eye(3)
    Rm, Rt = build_sigma3_pair(R, phase=CrystalPhase.FCC)
    np.testing.assert_array_equal(Rm, R)
    ang_rad, _ = o.misorientation_om(Rm.ravel(), Rt.ravel(), space_group=225)
    assert np.degrees(ang_rad) == pytest.approx(60.0, abs=1e-6)


def test_build_sigma3_pair_bcc_disorientation_is_60_deg():
    Rm, Rt = build_sigma3_pair(np.eye(3), phase=CrystalPhase.BCC)
    ang_rad, _ = o.misorientation_om(Rm.ravel(), Rt.ravel(), space_group=229)
    assert np.degrees(ang_rad) == pytest.approx(60.0, abs=1e-6)


def test_build_sigma3_pair_hcp_angle_in_expected_range():
    Rm, Rt = build_sigma3_pair(np.eye(3), phase=CrystalPhase.HCP)
    ang_rad, _ = o.misorientation_om(Rm.ravel(), Rt.ravel(), space_group=194)
    ang_deg = float(np.degrees(ang_rad))
    # HCP Sigma3-analogue is 86.3 deg about <1-100>; allow 1 deg slack in case
    # of basis-convention drift on midas_stress side.
    assert 84.0 < ang_deg < 89.0


# -- Matched pairs ----------------------------------------------------------

def test_find_sigma3_partners_synthetic_pair():
    # Two-grain system: matrix at identity, twin at Sigma3, near each other.
    axis_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    om_sigma3 = np.asarray(o.axis_angle_to_orient_mat(axis_111, 60.0))
    OM = np.stack([np.eye(3), om_sigma3], axis=0)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    var = np.array([0, 1])
    out = find_sigma3_partners(
        OM, pos, var, k_NN=1, phase=CrystalPhase.FCC, matrix_label=0, twin_label=1
    )
    assert out["pairs"].shape == (1, 2)
    assert tuple(out["pairs"][0]) == (0, 1)
    assert out["pair_misori"][0] == pytest.approx(60.0, abs=1e-6)
    assert out["pair_distances"][0] == pytest.approx(1.0)


def test_find_sigma3_partners_filters_out_off_sigma3_pair():
    # Twin orientation is 30 deg, not 60 -> should fail the angle filter.
    axis_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    om_not = np.asarray(o.axis_angle_to_orient_mat(axis_111, 30.0))
    OM = np.stack([np.eye(3), om_not], axis=0)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    var = np.array([0, 1])
    out = find_sigma3_partners(
        OM, pos, var, k_NN=1, phase=CrystalPhase.FCC, matrix_label=0, twin_label=1
    )
    assert out["pairs"].shape == (0, 2)
