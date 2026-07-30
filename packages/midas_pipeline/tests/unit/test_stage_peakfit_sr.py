"""FF `peakfit` stage's sr-midas branch.

``run_sr=True`` replaces the conventional ``midas_peakfit`` in-process call
with a subprocess invocation of ``python -m midas_pipeline._sr_worker`` —
isolation so the SR CNN cascade + GPU peak-fit's CUDA context is torn down
before indexing/refinement claim the GPU (see stages/peakfit.py::
_run_sr_subprocess). These tests mock ``subprocess.run`` and never actually
import sr_midas or touch a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.stages import peakfit
from midas_pipeline.stages._base import StageContext


def _sr_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **cfg_kwargs) -> StageContext:
    # PipelineConfig.__post_init__ fails fast unless sr_midas is importable.
    monkeypatch.setitem(sys.modules, "sr_midas", SimpleNamespace())
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
        device="cpu", dtype="float64",
        n_cpus=4,
        run_sr=True, srfac=8,
        **cfg_kwargs,
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    (layer_dir / "dummy.MIDAS.zip").write_text("")
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir, log_dir=log_dir)


def test_run_sr_invokes_worker_subprocess_not_conventional_peakfit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    seen = {}
    conventional_calls = []

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        # The SR worker is responsible for producing this in a real run;
        # simulate that here so _run_ff's post-condition check passes.
        (tmp_path / "Layer1" / "Temp").mkdir(parents=True, exist_ok=True)
        (tmp_path / "Layer1" / "Temp" / "AllPeaks_PS.bin").write_bytes(b"")
        return SimpleNamespace(returncode=0)

    def fake_peakfit_run(**kwargs):
        conventional_calls.append(kwargs)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _sr_ctx(tmp_path, monkeypatch, save_sr_patches=True)

    # Call _run_ff directly (bypassing peakfit.run()'s `midas_peakfit`
    # importability check) since only the SR branch is under test here.
    peakfit._run_ff(ctx, started=0.0, peakfit_run=fake_peakfit_run)

    assert not conventional_calls, "conventional peakfit_run must not be called when run_sr=True"
    assert seen["cmd"][1:3] == ["-m", "midas_pipeline._sr_worker"]
    assert seen["cmd"][3] == str(ctx.layer_dir)
    assert "--srfac" in seen["cmd"] and "8" in seen["cmd"]
    assert "--save-sr-patches" in seen["cmd"]
    save_idx = seen["cmd"].index("--save-sr-patches")
    assert seen["cmd"][save_idx + 1] == "1"
    assert seen["cwd"] == str(ctx.layer_dir)


def test_run_sr_raises_if_target_missing_after_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A worker subprocess that exits 0 but writes nothing is a bug, not a
    soft-skip — the stage must surface it loudly."""
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    ctx = _sr_ctx(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="was not written"):
        peakfit._run_ff(ctx, started=0.0, peakfit_run=lambda **kw: None)
