"""Reconstruction quality report (recon/quality.py) and the reconstruct stage's method='all' + sample mask."""
import json
import numpy as np
import pytest
from midas_pipeline.recon.quality import agreement, half_rows, labels_from_stack, quality_entry


def _stack(lab, n_g):
    R = np.zeros((n_g,) + lab.shape, np.float32)
    for g in range(n_g):
        R[g][lab == g] = 1.0
    return R


def test_labels_are_minus_one_where_empty_or_masked():
    lab = np.array([[0, 1], [1, 0]]); R = _stack(lab, 2); R[:, 0, 0] = 0
    mask = np.array([[True, True], [False, True]])
    assert labels_from_stack(R, mask).tolist() == [[-1, 1], [-1, 0]]


def test_agreement_counts_unassigned_as_disagreement_and_uses_support():
    a = np.array([[0, -1], [1, 1]]); b = np.array([[0, 0], [1, 0]]); sup = np.ones((2, 2), bool)
    assert agreement(a, b, sup) == 0.5
    assert agreement(a, b, np.array([[True, False], [True, False]])) == 1.0


def test_half_rows_split_and_pack():
    s = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3); o = np.arange(8.0).reshape(2, 4); nr = np.array([4, 3])
    s0, o0, n0 = half_rows(s, o, nr, 0); s1, o1, n1 = half_rows(s, o, nr, 1)
    assert n0.tolist() == [2, 2] and n1.tolist() == [2, 1]
    assert np.array_equal(s0[0, :2], s[0, [0, 2]]) and np.array_equal(o1[1, :1], o[1, [1]])


def test_quality_entry_flags_a_transposed_map():
    rng = np.random.default_rng(0); pbp = rng.integers(0, 6, (20, 20)); sup = np.ones((20, 20), bool)
    good = quality_entry(pbp.copy(), sup, pbp=pbp, halves=(pbp, pbp))
    bad = quality_entry(pbp.T.copy(), sup, pbp=pbp)
    assert good["agreement_vs_pbp"] == 1.0 and good["half_split"] == 1.0 and good["grid_convention_ok"]
    assert bad["best_transform"] == 4 and not bad["grid_convention_ok"]


def test_reconstruct_all_writes_report_and_masks_vacuum(tmp_path, monkeypatch):
    from midas_pipeline.stages import reconstruct as R
    n, n_g = 12, 3
    truth = np.full((n, n), -1); truth[2:10, 2:6] = 0; truth[2:10, 6:10] = 1
    mask = truth >= 0
    noisy = _stack(np.where(truth >= 0, truth, 2), n_g) + 0.01          # a recon that paints vacuum as grain 2
    (tmp_path / "Output").mkdir(); (tmp_path / "Output" / "IndexBest_all.bin").write_bytes(b"")
    monkeypatch.setattr(R, "_read_sinograms", lambda *a, **k: (np.zeros((n_g, 4, n)), np.zeros((n_g, 4)), np.array([4, 4, 4], np.int32)))
    monkeypatch.setattr(R, "_reconstruct", lambda m, *a, **k: _stack(truth.clip(0), n_g) if m == "voxelmap" else noisy.copy())
    class C:  # minimal config
        class recon: sino_type = "raw"
    stack, rep = R._reconstruct_all(tmp_path, C, n, tmp_path, mask)
    assert rep["support"] == "sample mask" and set(rep["methods"]) == {"fbp", "mlem", "voxelmap"}
    assert rep["methods"]["fbp"]["agreement_vs_pbp"] == 1.0 and rep["methods"]["fbp"]["half_split"] == 1.0
    lab = np.load(tmp_path / "labels_fbp.npy")
    assert (lab[~mask] == -1).all() and json.loads((tmp_path / "ReconQuality.json").read_text())["support"] == "sample mask"
    # without a mask the support falls back to the PBP-solved voxels, and says so
    _, rep2 = R._reconstruct_all(tmp_path, C, n, tmp_path, None)
    assert rep2["support"].startswith("voxels the per-voxel (PBP) map solved")


def test_count_grains_skips_the_grain_list_header(tmp_path):
    from midas_pipeline.stages.reconstruct import _count_grains
    (tmp_path / "Output").mkdir()
    (tmp_path / "Output" / "UniqueOrientations.csv").write_text(
        "# GrainID RowNr nSpots StartRowNr ListStartPos OM1 OM2 OM3 OM4 OM5 OM6 OM7 OM8 OM9\n"
        "1 2 3 4 5 1 0 0 0 1 0 0 0 1 \n7 8 9 10 11 1 0 0 0 1 0 0 0 1 \n")
    assert _count_grains(tmp_path) == 2
