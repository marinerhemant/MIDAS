"""End-to-end pipeline smoke + I/O round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_pipeline_runs_on_tiny_run_dir(tiny_run_dir: Path):
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="spot_aware")
    assert result.n_grains >= 1
    assert result.orient_mat.shape == (result.n_grains, 3, 3)
    assert result.positions.shape == (result.n_grains, 3)
    assert result.strain_lab.shape == (result.n_grains, 3, 3)


def test_pipeline_writes_csvs(tiny_run_dir: Path, tmp_path: Path):
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="spot_aware")
    out = tmp_path / "out"
    result.write(out, h5=True, diagnostics_h5=True)
    assert (out / "Grains.csv").exists()
    assert (out / "GrainIDsKey.csv").exists()
    assert (out / "data_consolidated.h5").exists()
    assert (out / "processgrains_diagnostics.h5").exists()


def test_pipeline_legacy_mode_runs(tiny_run_dir: Path, tmp_path: Path):
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="legacy")
    assert result.mode == "legacy"
    assert result.n_grains >= 1


def test_pipeline_paper_claim_mode_runs(tiny_run_dir: Path):
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="paper_claim")
    assert result.mode == "paper_claim"


def test_pipeline_adaptive_mode_runs(tiny_run_dir: Path):
    """Adaptive mode derives MisoriTol from the antimode of the pairwise
    misorientation histogram at run-time. It should run end-to-end on the
    tiny synthetic dataset and emit at least one grain."""
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="adaptive")
    assert result.mode == "adaptive"
    assert result.n_grains >= 1


def test_consolidated_h5_has_expected_groups(tiny_run_dir: Path, tmp_path: Path):
    import h5py
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="spot_aware")
    out = tmp_path / "out"
    result.write(out, h5=True, diagnostics_h5=True)
    with h5py.File(out / "data_consolidated.h5", "r") as f:
        assert "grains" in f
        for key in ("ids", "orient_mat", "positions", "lattice",
                    "grain_radius", "confidence",
                    "strain_lab", "strain_grain"):
            assert key in f["grains"]
        assert "attrs" in f
        assert f["attrs"].attrs["sg_nr"] == 225
        assert f["attrs"].attrs["mode"].decode() if isinstance(
            f["attrs"].attrs["mode"], bytes
        ) else f["attrs"].attrs["mode"] == "spot_aware"


def test_diagnostics_h5_records_per_grain_metadata(tiny_run_dir: Path, tmp_path: Path):
    import h5py
    from midas_process_grains.pipeline import ProcessGrains
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt", device="cpu")
    result = pg.run(mode="spot_aware")
    out = tmp_path / "out"
    result.write(out, h5=False, diagnostics_h5=True)
    with h5py.File(out / "processgrains_diagnostics.h5", "r") as f:
        for key in (
            "cluster_sizes",
            "n_resolved_hkls",
            "n_majority_hkls",
            "n_residual_tie_hkls",
            "n_forward_sim_hkls",
        ):
            assert key in f["diagnostics"]
            assert f["diagnostics"][key].shape == (result.n_grains,)


def test_cli_smoke(tiny_run_dir: Path, tmp_path: Path, capsys, monkeypatch):
    """The CLI can run end-to-end."""
    from midas_process_grains.cli import main
    out = tmp_path / "out"
    rc = main([
        str(tiny_run_dir / "paramstest.txt"),
        "1",
        "--mode", "spot_aware",
        "--device", "cpu",
        "--dtype", "float64",
        "--out-dir", str(out),
    ])
    assert rc == 0
    assert (out / "Grains.csv").exists()
