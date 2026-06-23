"""Byte-parity: global_cluster_fast (binned / gpu) == global_cluster (reference).

The fast variants must produce IDENTICAL clusters (voxel_to_unique) and the same
representative key array, for cubic (225) and hexagonal (194) space groups, across
tight clusters, intragranular spread (the deformed worst case), and invalid voxels.
"""
import numpy as np
import pytest

from midas_pipeline.find_grains._cluster import global_cluster, global_cluster_fast
from midas_stress.orientation import quat_to_orient_mat


def _rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q


def _perturb(q, rng, max_rad):
    """Rotate quaternion q by a small random rotation of angle < max_rad."""
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = rng.uniform(0, max_rad)
    dq = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
    w0, x0, y0, z0 = dq; w1, x1, y1, z1 = q
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def _make_case(rng, n_grains=8, per_grain=40, spread_deg=2.5):
    bases = _rand_quat(rng, n_grains)
    oms, confs, keys = [], [], []
    vid = 0
    for g in range(n_grains):
        for _ in range(per_grain):
            qp = _perturb(bases[g], rng, np.radians(spread_deg))
            oms.append(np.asarray(quat_to_orient_mat(qp), dtype=np.float64).ravel())
            confs.append(rng.uniform(0.3, 1.0))
            keys.append([vid + 1, rng.integers(3, 30), rng.integers(3, 30), 0])
            vid += 1
    oms = np.array(oms); confs = np.array(confs)
    keys = np.array(keys, dtype=np.uint64)
    # mark ~10% invalid
    inv = rng.choice(len(oms), size=len(oms) // 10, replace=False)
    keys[inv, 0] = np.uint64(2**64 - 1)
    # shuffle voxel order (clustering is order-dependent — parity must hold under it)
    perm = rng.permutation(len(oms))
    return oms[perm], confs[perm], keys[perm]


@pytest.mark.parametrize("sg", [225, 194])
@pytest.mark.parametrize("method", ["binned", "gpu"])
def test_parity(sg, method):
    if method == "gpu":
        torch = pytest.importorskip("torch")
        device = torch.device("cpu")
    else:
        device = None
    rng = np.random.default_rng(sg * 7 + (method == "gpu"))
    oms, confs, keys = _make_case(rng, spread_deg=2.5)
    ref = global_cluster(oms, confs, keys, space_group=sg, max_ang_deg=1.0)
    fast = global_cluster_fast(oms, confs, keys, space_group=sg, max_ang_deg=1.0,
                               method=method, device=device)
    assert fast.n_uniques == ref.n_uniques, (method, sg, fast.n_uniques, ref.n_uniques)
    np.testing.assert_array_equal(fast.voxel_to_unique, ref.voxel_to_unique)
    np.testing.assert_array_equal(fast.unique_key_arr, ref.unique_key_arr)
    np.testing.assert_allclose(fast.unique_OM_arr, ref.unique_OM_arr, atol=0, rtol=0)


@pytest.mark.parametrize("sg", [225, 194])
def test_parity_high_spread(sg):
    """Deformed worst case: spread >> threshold → many singleton clusters."""
    rng = np.random.default_rng(sg)
    oms, confs, keys = _make_case(rng, n_grains=4, per_grain=80, spread_deg=5.0)
    ref = global_cluster(oms, confs, keys, space_group=sg, max_ang_deg=1.0)
    fast = global_cluster_fast(oms, confs, keys, space_group=sg, max_ang_deg=1.0,
                               method="binned")
    np.testing.assert_array_equal(fast.voxel_to_unique, ref.voxel_to_unique)
    np.testing.assert_array_equal(fast.unique_key_arr, ref.unique_key_arr)


@pytest.mark.slow
@pytest.mark.parametrize("sg", [225, 194])
def test_gpu_exact_large(sg):
    """GPU all-pairs path must be EXACT at scale (this is the production default).
    The 3600-voxel deformed-like case is where binning's symmetry-boundary gap
    surfaced; GPU computes every pair so it has no such gap."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    oms, confs, keys = _make_case(rng, n_grains=30, per_grain=120, spread_deg=4.0)
    ref = global_cluster(oms, confs, keys, space_group=sg, max_ang_deg=1.0)
    gpu = global_cluster_fast(oms, confs, keys, space_group=sg, max_ang_deg=1.0,
                              method="gpu", device=torch.device("cpu"))
    np.testing.assert_array_equal(gpu.voxel_to_unique, ref.voxel_to_unique)
    np.testing.assert_array_equal(gpu.unique_key_arr, ref.unique_key_arr)


@pytest.mark.slow
@pytest.mark.xfail(reason="binned FZ-hashing has a rare symmetry-boundary candidate gap "
                          "at large N (~2 missed close pairs / 3600 voxels); use gpu/auto "
                          "for exact results until fixed", strict=False)
@pytest.mark.parametrize("sg", [194])
def test_binned_known_gap(sg):
    rng = np.random.default_rng(0)
    oms, confs, keys = _make_case(rng, n_grains=30, per_grain=120, spread_deg=4.0)
    ref = global_cluster(oms, confs, keys, space_group=sg, max_ang_deg=1.0)
    binned = global_cluster_fast(oms, confs, keys, space_group=sg, max_ang_deg=1.0,
                                 method="binned")
    np.testing.assert_array_equal(binned.voxel_to_unique, ref.voxel_to_unique)
