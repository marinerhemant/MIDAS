"""Own-grain density check (midas_pipeline.recon.own_grain_density)."""

import json

import numpy as np
import pytest

from midas_pipeline.recon.own_grain_density import own_grain_density


def _case():
    n = 10
    R = np.zeros((3, n, n), np.float32)
    R[0, :, :6] = 1.0                       # grain 0 has density in columns 0-5
    R[1, :, 6:] = 1.0                       # grain 1 has density in columns 6-9
    R[2, :, :] = 1.0                        # grain 2 (few rows) has density everywhere
    lab = np.full((n, n), -1)
    lab[:, :8] = 0                          # PBP gives grain 0 columns 0-7: columns 6-7 are 'inherited' (no own density)
    lab[:, 8:] = 1
    lab[0, 0] = 2                           # a voxel of the low-row grain 2
    nr = np.array([120, 90, 5])
    return R, lab, nr


def test_flags_voxels_with_no_own_density_and_keeps_the_rest():
    R, lab, nr = _case()
    d = own_grain_density(R, lab, nr)
    flag = d["flag"].reshape(10, 10)
    assert flag[:, 6:8].all()               # grain 0's voxels with no own density
    assert not flag[1:, :6].any()           # grain 0's real voxels are kept
    assert not flag[:, 8:].any()            # grain 1 is fully backed by its own density


def test_grains_below_min_rows_are_not_scored():
    R, lab, nr = _case()
    d = own_grain_density(R, lab, nr)
    assert np.isnan(d["rho"].reshape(10, 10)[0, 0])
    assert not d["flag"].reshape(10, 10)[0, 0] and not d["scored"].reshape(10, 10)[0, 0]
    assert d["grains_below_min_rows"] == 1 and d["unscored_solved_voxels"] == 1
    d2 = own_grain_density(R, lab, nr, min_rows=1)
    assert d2["scored"].reshape(10, 10)[0, 0]


def test_unsolved_voxels_are_nan_and_never_flagged():
    R, lab, nr = _case()
    lab = lab.copy(); lab[5, 5] = -1
    d = own_grain_density(R, lab, nr)
    assert np.isnan(d["rho"].reshape(10, 10)[5, 5]) and not d["flag"].reshape(10, 10)[5, 5]


def test_grain_with_no_density_anywhere_is_all_flagged():
    R, lab, nr = _case()
    R = R.copy(); R[1] = 0
    d = own_grain_density(R, lab, nr)
    assert d["flag"].reshape(10, 10)[:, 8:].all()


def test_shape_mismatch_raises():
    R, lab, nr = _case()
    with pytest.raises(ValueError):
        own_grain_density(R, lab[:5], nr)


def test_graded_density_pins_the_p90_reference_the_threshold_and_the_row_order():
    """Binary fixtures pass for any percentile and any tau in (0, 1); a ramp does not. Values 1..100 in row-major
    order: P90 = 90.1, so rho < 0.15 <=> value < 13.515 <=> exactly the first 13 voxels (P75 would give 11, P95 14,
    tau 0.14 gives 12, tau 0.16 gives 14; a transposed layout flags the first column instead)."""
    ramp = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    R = ramp[None].copy()
    d = own_grain_density(R, np.zeros((10, 10), int), np.array([80]))
    expect = np.zeros((10, 10), bool)
    expect.ravel()[:13] = True
    np.testing.assert_array_equal(d["flag"].reshape(10, 10), expect)
    np.testing.assert_allclose(d["rho"].reshape(10, 10), ramp / np.percentile(ramp, 90), rtol=1e-6)


def test_row_floor_is_inclusive_at_min_rows():
    R = np.ones((2, 4, 4), np.float32)
    lab = np.zeros((4, 4), int); lab[:, 2:] = 1
    d = own_grain_density(R, lab, np.array([30, 29]))
    assert d["scored"].reshape(4, 4)[:, :2].all() and not d["scored"].reshape(4, 4)[:, 2:].any()


def test_reconstruct_all_hook_uses_voxel_grid_labels_and_the_unmasked_stack(tmp_path, monkeypatch):
    """The density check must see voxels a sample mask removes and must score the grain assignment find_grains wrote
    (voxel_grid.csv), not the min_conf-filtered voxel-map labels (both differed in the first version of the hook)."""
    from midas_pipeline.stages import reconstruct as R
    n, n_g = 12, 3
    truth = np.full((n, n), -1); truth[2:10, 2:6] = 0; truth[2:10, 6:10] = 1
    mask = np.zeros((n, n), bool); mask[2:10, 2:6] = True                 # sample = columns 2-5 only
    ml = np.zeros((n_g, n, n), np.float32)
    ml[0, 2:10, 2:6] = 1.0                                               # grain 0: real density in columns 2-5
    ml[0, 2:10, 6:8] = 0.5                                               # ... and a weaker leak in columns 6-7 (outside the mask)
    ml[1, 2:10, 8:10] = 1.0
    (tmp_path / "Output").mkdir(); (tmp_path / "Output" / "IndexBest_all.bin").write_bytes(b"")
    gid = np.full((n, n), -1); gid[2:10, 2:8] = 0; gid[2:10, 8:10] = 1    # find_grains gave grain 0 columns 2-7
    lines = ["voxel_idx x_um y_um z_um grain_id"] + [f"{v} 0 0 0 {gid.ravel()[v]}" for v in range(n * n)]
    (tmp_path / "Output" / "voxel_grid.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(R, "_read_sinograms", lambda *a, **k: (np.zeros((n_g, 40, n)), np.zeros((n_g, 40)), np.array([40, 40, 40], np.int32)))

    def fake(m, *a, **k):
        if m == "voxelmap":                                              # the voxel-map labels DROP columns 6-7 (as a min_conf cut would)
            return _stack_lab(np.where(np.isin(np.arange(n)[None, :], [6, 7]), -1, truth), n_g)
        return ml.copy()
    monkeypatch.setattr(R, "_reconstruct", fake)

    class C:
        class recon: sino_type = "raw"
    R._reconstruct_all(tmp_path, C, n, tmp_path, mask)
    rho = np.load(tmp_path / "OwnGrainDensity.npy")
    assert rho.shape == (n, n)
    assert np.isfinite(rho[2:10, 6:8]).all()                              # scored although the mask and the voxel-map labels drop them
    np.testing.assert_allclose(rho[2:10, 6:8], 0.5, rtol=1e-6)            # from the UNMASKED stack: 0.5 / P90 (=1)
    assert not (rho[2:10, 6:8] < 0.15).any()
    rep = json.loads((tmp_path / "ReconQuality.json").read_text())
    assert rep["own_grain_density"]["scored_voxels"] == (6 + 2) * 8 and rep["own_grain_density"]["flagged_voxels"] == 0


def _stack_lab(lab, n_g):
    R_ = np.zeros((n_g,) + lab.shape, np.float32)
    for g in range(n_g):
        R_[g][lab == g] = 1.0
    return R_
