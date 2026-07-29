"""Tests for `midas_defect.subgrain`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_defect.asterism_fit import AsterismFit
from midas_defect.subgrain import (
    SubGrain,
    decompose_asterism_patches,
    _discrete_dbscan_patch,
)


@pytest.mark.unit
def test_dbscan_finds_two_well_separated_clusters():
    rng = np.random.default_rng(0)
    A = rng.normal(loc=[0, 0, 0], scale=0.005, size=(30, 3))
    B = rng.normal(loc=[0.05, 0, 0], scale=0.005, size=(30, 3))
    pts = np.vstack([A, B])
    ints = np.full(60, 100.0)
    clusters = _discrete_dbscan_patch(
        pts, ints, eps=0.015, min_intensity_frac=0.5, min_cluster_size=5,
    )
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes[0] >= 20 and sizes[1] >= 20


@pytest.mark.unit
def test_dbscan_merges_one_dense_cluster():
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=0, scale=0.01, size=(50, 3))
    ints = np.full(50, 100.0)
    clusters = _discrete_dbscan_patch(
        pts, ints, eps=0.04, min_intensity_frac=0.5, min_cluster_size=5,
    )
    assert len(clusters) == 1
    assert len(clusters[0]) >= 40


@pytest.mark.integration
def test_decompose_finds_two_subgrains_per_patch():
    """Build a fake asterism patch containing two distinct sub-clusters."""
    rng = np.random.default_rng(0)
    center = np.array([1.0, 0.0, 0.5])
    A_off = np.array([0.02, 0.0, 0.0])
    B_off = np.array([-0.02, 0.0, 0.0])
    A = (center + A_off)[None, :] + rng.normal(scale=0.005, size=(40, 3))
    B = (center + B_off)[None, :] + rng.normal(scale=0.005, size=(40, 3))
    qx = np.concatenate([A[:, 0], B[:, 0]])
    qy = np.concatenate([A[:, 1], B[:, 1]])
    qz = np.concatenate([A[:, 2], B[:, 2]])
    inten = np.full(80, 120.0)

    fake_fit = AsterismFit(
        hkl=(1, 0, 0), q_pred=center, q_fit=center,
        amplitude=120.0, baseline=0.0,
        sigma_eig=np.array([0.03, 0.01, 0.01]),
        sigma_axes=np.eye(3),
        integrated_intensity=80 * 120.0, n_voxels=80,
        final_loss=0.0, converged=True,
    )
    subs = decompose_asterism_patches(
        qx, qy, qz, inten, [fake_fit],
        eps=0.012, min_intensity_frac=0.4, min_cluster_size=10,
    )
    assert len(subs) == 2, f"expected 2 sub-grains, got {len(subs)}"
    # Centers should be near the planted off-positions
    centers = np.array([s.q_center for s in subs])
    expected = np.stack([center + A_off, center + B_off])
    # Match each detected center to its nearest expected
    used = set()
    for c in centers:
        best = -1; best_d = 999
        for i, e in enumerate(expected):
            if i in used:
                continue
            d = np.linalg.norm(c - e)
            if d < best_d:
                best_d = d; best = i
        assert best != -1 and best_d < 0.005, (
            f"sub-grain center {c} too far ({best_d}) from any planted spot"
        )
        used.add(best)
