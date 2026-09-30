"""Unit tests for midas_pipeline.fuse.

Two scenarios:

- ``test_bayesian_fusion_boundary_disambiguates`` — a synthetic 2-grain
  shape map with a boundary voxel; orientation likelihood favors one
  grain at the boundary. Assert the posterior picks the right grain.
- ``test_mask_sino_friedel_keeps_both`` — synthetic Friedel pair; the
  dual-sign filter must keep cells satisfying either branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.fuse import bayesian_fusion, mask_sino_by_assignment


def _write_indexbest_fixture(tmp_path: Path, n_vox: int, sols_per_vox, all_records):
    out_dir = tmp_path / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sol_arr = np.asarray(sols_per_vox, dtype=np.int32)
    header_bytes = 4 + 4 * n_vox + 8 * n_vox
    off_arr = np.empty(n_vox, dtype=np.int64)
    cum = 0
    for v in range(n_vox):
        off_arr[v] = header_bytes + cum * 8
        cum += n_sol_arr[v] * 16

    flat_records = np.asarray(all_records, dtype=np.float64).reshape(-1)
    path = out_dir / "IndexBest_all.bin"
    with open(path, "wb") as f:
        np.asarray([n_vox], dtype=np.int32).tofile(f)
        n_sol_arr.tofile(f)
        off_arr.tofile(f)
        flat_records.tofile(f)
    return path


def _write_unique_orientations(tmp_path: Path, oms_3x3):
    rows = []
    for g, om in enumerate(oms_3x3):
        rows.append([float(g), 0.0, 0.0, 0.0, 0.0] + list(np.asarray(om).flatten()))
    data = np.asarray(rows, dtype=np.float64)
    # The grain list lives where find_grains writes it (Output/), not at the
    # layer level, where seeding writes the seed list.
    path = tmp_path / "Output" / "UniqueOrientations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.10f", delimiter=" ")
    return path


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_bayesian_fusion_boundary_disambiguates(tmp_path):
    """At a voxel where shape favors grain 0 strongly, the fused posterior
    should favor grain 0 even if grain 1 has a similar orient-score there."""
    n_scans = 2
    n_vox = 4
    om_g0 = np.eye(3)
    om_g1 = _rot_z(np.deg2rad(30.0))

    def _record(om, conf):
        r = np.zeros(16, dtype=np.float64)
        r[2:11] = np.asarray(om).flatten()
        r[14] = 10.0
        r[15] = 10.0 * conf
        return r

    # Each voxel has two candidates (one matching each grain). Voxel 0 is
    # the "boundary": both grains have similar orient-score.
    records = [
        # v0 (boundary): candidates for both grains
        _record(om_g0, 0.8), _record(om_g1, 0.8),
        # v1: only grain 0 candidate
        _record(om_g0, 0.9),
        # v2: only grain 1 candidate
        _record(om_g1, 0.95),
        # v3: only grain 1 candidate
        _record(om_g1, 0.85),
    ]
    sols = [2, 1, 1, 1]
    _write_indexbest_fixture(tmp_path, n_vox, sols, records)
    _write_unique_orientations(tmp_path, [om_g0, om_g1])

    # Per-grain shape (recon): grain 0 occupies upper half, grain 1 the lower.
    # The boundary voxel v0 has stronger shape support for grain 0.
    shape = np.zeros((2, n_scans, n_scans), dtype=np.float32)
    shape[0, 0, 0] = 1.0       # v0 — boundary
    shape[0, 0, 1] = 0.9       # v1
    shape[1, 1, 0] = 1.0       # v2
    shape[1, 1, 1] = 0.95      # v3

    posterior = bayesian_fusion(
        shape, tmp_path, sgnum=225, n_grains=2,
        max_ang_deg=1.0, min_conf=0.5,
    )
    # At the boundary v0, posterior[grain 0] should win because of shape.
    assert posterior[0, 0, 0] > posterior[1, 0, 0]
    # v2 should still be grain 1 (no grain 0 candidate there).
    assert posterior[1, 1, 0] > posterior[0, 1, 0]


def test_mask_sino_friedel_keeps_both():
    """A Friedel pair: a voxel V and its omega+180° image should both pass
    the mask. The s_proj for omega and omega+180° are negatives of each other,
    so the dual-sign filter (|s − ypos| < tol or |−s − ypos| < tol) must
    keep both."""
    n_grains = 1
    n_scans = 3
    max_nhkl = 2
    # spatial_pos: [−1, 0, 1] um
    spatial_pos = np.array([-1.0, 0.0, 1.0])

    # One grain assigned to voxel (row=2, col=0) → x=spatial_pos[2]=1, y=spatial_pos[0]=-1
    max_id = -np.ones((n_scans, n_scans), dtype=np.int32)
    max_id[2, 0] = 0

    # Two HKLs at omega=0° and omega=180° (Friedel pair).
    # s = x*sin(omega) + y*cos(omega)   (find_grains._geom.scan_projection_um)
    # omega=0:   s = y = -1
    # omega=180: s = -y = 1
    omegas = np.array([[0.0, 180.0]])

    # Build sinos that put intensity at the scan position the Friedel pair maps to
    sinos = np.zeros((n_grains, max_nhkl, n_scans), dtype=np.float64)
    sinos[0, :, :] = 1.0     # every scan filled — mask will pick out the ones that match
    nr_hkls = np.array([2], dtype=np.int32)

    masked = mask_sino_by_assignment(
        sinos, omegas, nr_hkls, max_id, n_grains, n_scans,
        spatial_pos, scan_tol=0.5,
    )
    # HKL0 (omega=0): s_proj=-1 → scan_pos=-1 by the primary branch
    # HKL1 (omega=180): s_proj=1 → scan_pos=1 by the primary branch
    # Friedel filter also adds |-s_proj - scan_pos| < 0.5, so for HKL0
    # |−1 − scan_pos| < 0.5 → scan_pos=−1 also kept; HKL1 similarly keeps scan_pos=1.
    # → both rows should have BOTH endpoints kept.
    assert masked[0, 0, 0] > 0   # HKL0, scan_pos=-1 kept by primary branch
    assert masked[0, 0, 2] > 0   # HKL0, scan_pos=1 kept by Friedel branch
    assert masked[0, 1, 0] > 0   # HKL1, scan_pos=-1 kept by Friedel branch
    assert masked[0, 1, 2] > 0   # HKL1, scan_pos=1 kept by primary branch
    # Middle scan (scan_pos=0) is not within tol of ±1, so it should be zero.
    assert masked[0, 0, 1] == 0
    assert masked[0, 1, 1] == 0


def test_fusion_reads_grain_list_not_seed_list(tmp_path):
    """Seeded layout: a layer-level seed list with MORE rows, in another order,
    must not be read. Grain g is row g of Output/UniqueOrientations.csv."""
    n_scans, n_vox = 2, 4
    om_g0, om_g1 = np.eye(3), _rot_z(np.deg2rad(30.0))

    def _record(om, conf):
        r = np.zeros(16, dtype=np.float64)
        r[2:11] = np.asarray(om).flatten(); r[14] = 10.0; r[15] = 10.0 * conf
        return r

    _write_indexbest_fixture(tmp_path, n_vox, [1, 1, 1, 1],
                             [_record(om_g0, .9), _record(om_g0, .9),
                              _record(om_g1, .9), _record(om_g1, .9)])
    _write_unique_orientations(tmp_path, [om_g0, om_g1])
    # decoy seed list at layer level: reversed order plus an extra row
    rows = [[float(g), 0, 0, 0, 0] + list(np.asarray(om).flatten())
            for g, om in enumerate([om_g1, _rot_z(1.0), om_g0])]
    np.savetxt(tmp_path / "UniqueOrientations.csv", np.asarray(rows), fmt="%.10f")

    shape = np.zeros((2, n_scans, n_scans), dtype=np.float32)
    shape[0, 0, :] = 1.0; shape[1, 1, :] = 1.0
    post = bayesian_fusion(shape, tmp_path, sgnum=225, n_grains=2,
                           max_ang_deg=1.0, min_conf=0.5)
    assert post[0, 0, 0] > 0 and post[1, 1, 0] > 0
    assert post[0, 1, 0] == 0 and post[1, 0, 0] == 0


def test_fusion_without_grain_list_says_so(tmp_path):
    r = np.zeros(16); r[2:11] = np.eye(3).flatten(); r[14] = 10.0; r[15] = 9.0
    _write_indexbest_fixture(tmp_path, 1, [1], [r])
    np.savetxt(tmp_path / "UniqueOrientations.csv", np.zeros((1, 14)))  # seed list only
    with pytest.raises(FileNotFoundError, match="find_grains"):
        bayesian_fusion(np.ones((1, 1, 1), np.float32), tmp_path, sgnum=225,
                        n_grains=1, max_ang_deg=1.0, min_conf=0.5)



def _asym_grain_sinogram(n=31, conv="indexer"):
    """An off-centre, asymmetric grain (an L of voxels) and its sinogram in the given convention."""
    pos = np.linspace(-150.0, 150.0, n)                 # spatial (ascending) positions, 10 um apart
    max_id = -np.ones((n, n), dtype=np.int32)
    max_id[20:26, 5:9] = 0; max_id[23:26, 9:14] = 0      # x = pos[row] 50..100, y = pos[col] -100..-10
    rows, cols = np.nonzero(max_id == 0)
    om = np.arange(-180.0, 180.0, 7.0); w = np.deg2rad(om)[:, None]
    x, y = pos[rows][None], pos[cols][None]
    s = x * np.sin(w) + y * np.cos(w) if conv == "indexer" else x * np.sin(w) - y * np.cos(w)
    sino = np.zeros((1, len(om), n))
    for h in range(len(om)):
        for v in np.rint((s[h] - pos[0]) / 10.0).astype(int):
            sino[0, h, v] += 1.0
    return sino, om[None], np.array([len(om)], np.int32), max_id, pos


def test_mask_keeps_a_grain_projected_with_the_indexer_convention():
    """Off-centre asymmetric grain: the indexer's projection (x = pos[row], y = pos[col], s = x sin + y cos)
    is the convention of the sinograms; the mask must keep (almost) all of the grain's own intensity.
    The old convention (y mirrored) kept ~24 % on real 20-ID-E data; a y-mirrored sinogram must lose most."""
    sino, om, nr, max_id, pos = _asym_grain_sinogram(conv="indexer")
    kept = mask_sino_by_assignment(sino, om, nr, max_id, 1, len(pos), pos, scan_tol=5.0).sum() / sino.sum()
    assert kept > 0.99
    sino_m, *_ = _asym_grain_sinogram(conv="mirrored")
    kept_m = mask_sino_by_assignment(sino_m, om, nr, max_id, 1, len(pos), pos, scan_tol=5.0).sum() / sino_m.sum()
    assert kept_m < 0.8
