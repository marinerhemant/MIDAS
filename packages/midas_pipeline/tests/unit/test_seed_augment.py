"""Null-gated seed augmentation (midas_pipeline.seeding.augment)."""
from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.seeding.augment import (
    Candidates, augment_seed, distinct_far, null_gate, shuffle_ff_inputs, write_augmented)

SG = 167


def _rz(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _grains(path, oms):
    head = "%NumGrains {}\n%GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33 X Y Z\n".format(len(oms))
    rows = []
    for k, om in enumerate(oms):
        rows.append("\t".join([str(k + 1)] + [f"{v:.6f}" for v in om.ravel()] + ["1.0", "2.0", "3.0", "0.1"]) + "\n")
    path.write_text(head + "".join(rows))


def test_distinct_far_dedups_itself_and_drops_seed_neighbours():
    seed = np.array([np.eye(3).ravel()])
    om = np.array([_rx(20).ravel(), (_rx(20) @ _rz(0.3)).ravel(),   # near-duplicates of each other
                   _rx(0.4).ravel(),                                 # within 1 deg of the seed
                   _rx(40).ravel()])
    c = Candidates(om, np.array([0.90, 0.95, 0.99, 0.85]), np.zeros(4))
    d = distinct_far(c, seed, SG, 1.0)
    np.testing.assert_allclose(sorted(d.completeness), [0.85, 0.95])  # best of the pair kept


def test_gate_is_the_null_maximum():
    a = Candidates(np.zeros((2, 9)), np.array([0.7, 0.81]), np.zeros(2))
    b = Candidates(np.zeros((1, 9)), np.array([0.84]), np.zeros(1))
    assert null_gate([a, b]) == pytest.approx(0.84)
    assert null_gate([]) == float("inf")


def test_write_augmented_clones_rows_and_zeroes_position(tmp_path):
    g = tmp_path / "Grains.csv"; _grains(g, [np.eye(3), _rx(30)])
    out = tmp_path / "aug.csv"
    n = write_augmented(g, out, np.array([_rx(50).ravel()]))
    txt = out.read_text().splitlines()
    assert n == 3 and txt[0] == "%NumGrains 3"
    new = txt[-1].split("\t")
    assert new[0] == "3" and new[10:13] == ["0.000000"] * 3
    np.testing.assert_allclose([float(x) for x in new[1:10]], _rx(50).ravel(), atol=1e-6)


def test_augment_requires_a_null(tmp_path):
    with pytest.raises(ValueError, match="null"):
        augment_seed(tmp_path / "x", [], tmp_path / "g", tmp_path / "o", sg=SG)


def test_shuffle_keeps_each_rings_omegas_and_row_alignment(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    rng = np.random.default_rng(0); n = 60
    ring = np.repeat([1, 2, 3], 20).astype(float); sid = np.arange(1, n + 1, dtype=float)
    om = rng.uniform(-90, 90, n)
    A = np.c_[rng.random(n), rng.random(n), om, rng.random(n), sid, ring, rng.random(n), rng.random(n)]
    X = np.zeros((n, 18)); X[:, :8] = A; X[:, 8] = om; X[:, 13] = om
    np.savetxt(src / "InputAll.csv", A, fmt="%.6f", header="Y Z Omega R SpotID Ring Eta Tth", comments="")
    np.savetxt(src / "InputAllExtraInfoFittingAll.csv", X, fmt="%.6f", header=" ".join(f"c{i}" for i in range(18)), comments="")
    rep = shuffle_ff_inputs(src, tmp_path / "dst", seed=1)
    As = np.loadtxt(tmp_path / "dst/InputAll.csv", skiprows=1); Xs = np.loadtxt(tmp_path / "dst/InputAllExtraInfoFittingAll.csv", skiprows=1)
    assert rep["moved_fraction"] > 0.5
    for r in (1, 2, 3):
        m = ring == r
        np.testing.assert_allclose(np.sort(As[m, 2]), np.sort(A[m, 2]), atol=1e-6)   # same omegas per ring
    np.testing.assert_allclose(As[:, 2], Xs[:, 2], atol=1e-6)                          # both files permuted together
    np.testing.assert_allclose(Xs[:, 8], Xs[:, 2], atol=1e-6); np.testing.assert_allclose(Xs[:, 13], Xs[:, 2], atol=1e-6)
    np.testing.assert_allclose(As[:, [0, 1, 3, 4, 5]], A[:, [0, 1, 3, 4, 5]], atol=1e-6)  # nothing else moved


def _ff_layer(tmp_path):
    L = tmp_path / "ff" / "LayerNr_1"; L.mkdir(parents=True)
    n = 12; rng = np.random.default_rng(3)
    ring = np.repeat([1, 2], 6).astype(float); sid = np.arange(1, n + 1, dtype=float)
    om = rng.uniform(-90, 90, n)
    A = np.c_[rng.random(n), rng.random(n), om, rng.random(n), sid, ring, rng.random(n), rng.random(n)]
    X = np.zeros((n, 18)); X[:, :8] = A; X[:, 8] = om; X[:, 13] = om
    np.savetxt(L / "InputAll.csv", A, fmt="%.6f", header="Y Z Omega R SpotID Ring Eta Tth", comments="")
    np.savetxt(L / "InputAllExtraInfoFittingAll.csv", X, fmt="%.6f", header=" ".join(f"c{i}" for i in range(18)), comments="")
    for f in ("paramstest.txt", "hkls.csv", "positions.csv"):
        (L / f).write_text("x\n")
    (L / "SpotsToIndex.csv").write_text("1\n2\n3\n")
    (L / "paramstest_index_comp.txt").write_text(
        f"OutputFolder {L}/Output\nResultFolder {L}/Results\nSomeFile {L}/Spots.bin\nMinNHKLs 4\n")
    return L


def test_build_null_run_rewrites_paths_and_indexes(tmp_path, monkeypatch):
    import importlib
    bd = importlib.import_module("midas_transforms.bin_data")
    import subprocess
    from midas_pipeline.seeding import augment as A
    L = _ff_layer(tmp_path); dst = tmp_path / "null"
    binned = {}
    monkeypatch.setattr(bd, "bin_data", lambda **kw: binned.update(kw))
    calls = {}
    def fake_run(cmd, cwd, env, stdout, stderr, check):
        calls.update(cmd=cmd, cwd=cwd, omp=env["OMP_NUM_THREADS"])
        (dst / "Output" / "IndexBest_all.bin").write_bytes(b"\0")
    monkeypatch.setattr(subprocess, "run", fake_run)
    ib = A.build_null_run(L, dst, seed=5, n_cpus=3, indexer="/bin/idx")
    p = (dst / "paramstest_index_comp.txt").read_text()
    assert str(L) not in p and f"OutputFolder {dst}/Output" in p and f"SomeFile {dst}/Spots.bin" in p
    assert binned["result_folder"] == dst
    assert calls["cmd"] == ["/bin/idx", "paramstest_index_comp.txt", "0", "1", "3", "3"] and calls["omp"] == "3"
    assert ib == dst / "Output" / "IndexBest_all.bin"
    assert not np.allclose(np.loadtxt(dst / "InputAll.csv", skiprows=1)[:, 2], np.loadtxt(L / "InputAll.csv", skiprows=1)[:, 2])


def test_build_null_run_refuses_without_real_params(tmp_path):
    from midas_pipeline.seeding.augment import build_null_run
    L = _ff_layer(tmp_path); (L / "paramstest_index_comp.txt").unlink()
    with pytest.raises(FileNotFoundError, match="indexer parameters"):
        build_null_run(L, tmp_path / "null")


def test_seeding_stage_hands_off_the_augmented_seed(tmp_path, monkeypatch):
    import midas_pipeline.seeding as S
    import midas_pipeline.seeding.augment as A
    from midas_pipeline.config import PipelineConfig, ScanGeometry, SeedingConfig
    from midas_pipeline.stages import seeding as stage
    from midas_pipeline.stages._base import StageContext
    g = tmp_path / "Grains.csv"; g.write_text("%orig\n")
    ffl = tmp_path / "ffl"; ffl.mkdir()
    params = tmp_path / "P.txt"; params.write_text("SpaceGroup 167\n")
    cfg = PipelineConfig(result_dir=str(tmp_path / "run"), params_file=str(params),
                         scan=ScanGeometry.pf_uniform(n_scans=5, scan_step_um=1.0, beam_size_um=1.0),
                         device="cpu", seeding=SeedingConfig(mode="ff", grains_file=str(g), augment_ff_layer=str(ffl)))
    layer = tmp_path / "Layer1"; (layer / "midas_log").mkdir(parents=True)
    (layer / "paramstest.txt").write_text("SpaceGroup 167\n")
    ctx = StageContext(config=cfg, layer_nr=1, layer_dir=layer, log_dir=layer / "midas_log")
    seen = {}
    def fake_aug(ff_layer, work_dir, *, sg, n_null, n_cpus):
        seen.update(ff_layer=ff_layer, sg=sg, n_null=n_null)
        out = work_dir / "Grains_augmented.csv"; work_dir.mkdir(parents=True, exist_ok=True); out.write_text("%aug\n")
        return dict(out_csv=str(out), added=7, gate=0.84, null_distinct_far=[100])
    monkeypatch.setattr(A, "augment_from_ff_layer", fake_aug)
    monkeypatch.setattr(S, "grains_csv_to_unique_orientations",
                        lambda src, dst, space_group: (seen.update(handoff=str(src)), 5)[1])
    res = stage.run(ctx)
    assert seen["ff_layer"] == str(ffl) and seen["sg"] == 167 and seen["n_null"] == 1
    assert seen["handoff"].endswith("Grains_augmented.csv")
    assert res.metrics["augment_added"] == 7


def test_build_null_run_refuses_a_different_bin_layout(tmp_path, monkeypatch):
    import importlib
    from midas_pipeline.seeding import augment as A
    bd = importlib.import_module("midas_transforms.bin_data")
    L = _ff_layer(tmp_path); (L / "nData.bin").write_bytes(b"\0" * 64)
    dst = tmp_path / "null"
    monkeypatch.setattr(bd, "bin_data", lambda **kw: (dst / "nData.bin").write_bytes(b"\0" * 16))
    with pytest.raises(RuntimeError, match="bin layout"):
        A.build_null_run(L, dst, indexer="/bin/idx")
