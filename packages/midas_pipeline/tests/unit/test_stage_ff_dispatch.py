"""FF-mode dispatch tests for indexing + refinement stages.

The single-source contract: ``midas-pipeline run --scan-mode ff`` invokes
``python -m midas_index`` (indexing) and ``python -m midas_fit_grain``
(refinement) — the same kernels ``midas-ff-pipeline`` uses. These tests
mock ``subprocess.run`` and assert the command line + cwd are right;
they don't actually run the subprocess.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midas_pipeline.config import (
    PipelineConfig, ScanGeometry, RefinementConfig,
)
from midas_pipeline.stages._base import StageContext
from midas_pipeline.stages import indexing, refinement


def _ff_ctx(tmp_path: Path, *, has_files: bool, n_seeds: int = 3) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
        device="cpu", dtype="float64",
        n_cpus=4,
        refinement=RefinementConfig(solver="lbfgs", loss="angular", mode="all_at_once"),
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    if has_files:
        (layer_dir / "paramstest.txt").write_text("RingNumbers 1\n")
        (layer_dir / "SpotsToIndex.csv").write_text(
            "\n".join(str(i) for i in range(n_seeds)) + "\n"
        )
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                        log_dir=log_dir)


def test_indexing_ff_skips_when_artifacts_missing(tmp_path: Path):
    """No paramstest/SpotsToIndex → soft skip (smoke / partial-run path)."""
    result = indexing.run(_ff_ctx(tmp_path, has_files=False))
    assert result.skipped is True


def test_indexing_ff_invokes_midas_index_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When inputs are present, dispatch shells to ``python -m midas_index``."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=5)
    indexing.run(ctx)

    assert seen["cmd"][1:3] == ["-m", "midas_index"]
    # Positional args: paramstest, block_nr, n_blocks, n_seeds, n_cpus
    assert seen["cmd"][3].endswith("paramstest.txt")
    assert seen["cmd"][4:7] == ["0", "1", "5"]
    assert seen["cmd"][7] == "4"  # n_cpus
    assert seen["cwd"] == str(ctx.layer_dir)


def test_refinement_ff_skips_when_artifacts_missing(tmp_path: Path):
    result = refinement.run(_ff_ctx(tmp_path, has_files=False))
    assert result.skipped is True


def test_refinement_ff_invokes_midas_fit_grain_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When inputs are present, dispatch shells to ``python -m midas_fit_grain``."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True, n_seeds=7)
    refinement.run(ctx)

    assert seen["cmd"][1:3] == ["-m", "midas_fit_grain"]
    assert seen["cmd"][3].endswith("paramstest.txt")
    assert seen["cmd"][4:7] == ["0", "1", "7"]
    assert seen["cmd"][7] == "4"  # n_cpus
    assert "--solver" in seen["cmd"]
    assert "lbfgs" in seen["cmd"]
    assert "--loss" in seen["cmd"]
    assert "angular" in seen["cmd"]
    assert seen["cwd"] == str(ctx.layer_dir)


def test_refinement_ff_swaps_pixel_to_angular_for_multidet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Multi-detector paramstest → pixel loss swapped to angular."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _ff_ctx(tmp_path, has_files=True)
    (ctx.layer_dir / "paramstest.txt").write_text(
        "RingNumbers 1\nDetParams 0\n"
    )
    refinement.run(ctx)
    # pixel → angular swap
    loss_idx = seen["cmd"].index("--loss")
    assert seen["cmd"][loss_idx + 1] == "angular"
