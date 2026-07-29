"""Tests for intensity_budget: 4-bin partition that closes to 100%."""
import numpy as np

from midas_defect.intensity_budget import intensity_budget


def _bins(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0, 0.5, n)
    qmag = rng.uniform(0.5, 5.0, n)
    I = rng.uniform(1, 100, n)
    return dist, qmag, I


def test_budget_closes_exactly():
    b = intensity_budget(*_bins())
    assert b.closes(atol=1e-12)
    assert abs(sum(b.fractions.values()) - 1.0) < 1e-12


def test_budget_is_partition_counts():
    dist, qmag, I = _bins()
    b = intensity_budget(dist, qmag, I)
    assert sum(b.counts.values()) == len(I)


def test_bins_assigned_by_criteria():
    # one voxel per intended bin
    dist = np.array([0.01, 0.10, 0.30, 0.30])
    qmag = np.array([3.0, 3.0, 3.0, 1.0])   # last is low-q halo
    I = np.array([1.0, 1.0, 1.0, 1.0])
    b = intensity_budget(dist, qmag, I, bragg_tol=0.05, near_tol=0.20, low_q_cut=1.5)
    assert b.counts == {"bragg": 1, "asterism": 1, "inter_bragg": 1, "low_q_halo": 1}
    assert all(abs(f - 0.25) < 1e-12 for f in b.fractions.values())


def test_low_q_takes_precedence_over_dist():
    # a low-q voxel is halo even if it sits on the lattice
    b = intensity_budget(np.array([0.0]), np.array([1.0]), np.array([7.0]),
                         low_q_cut=1.5)
    assert b.counts["low_q_halo"] == 1 and b.counts["bragg"] == 0
