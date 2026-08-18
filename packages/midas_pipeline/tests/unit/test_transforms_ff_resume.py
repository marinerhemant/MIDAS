"""Unit tests for the FF ``transforms`` resume guard.

Regression guard: ``_run_ff`` had no existence check, so FF transforms re-ran
unconditionally on every resume while ``peakfit(FF)``, ``hkl(FF)`` and the PF
branch all skipped when their outputs were present.

That is not merely wasted work. transforms rewrites ``InputAll.csv`` and
``InputAllExtraInfoFittingAll.csv``, which invalidates ``binning`` and hence
``indexing`` -- so a resume aimed at a LATER stage (refinement, process_grains)
silently forced a full re-index. Measured on a 20-ID alumina layer
(24900 seeds, 471k spots): ~90 minutes of re-indexing per retry, twice, when the
only stage that needed to re-run was refinement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline.stages import transforms as T


class _Cfg:
    def __init__(self, zarr_path):
        self.zarr_path = str(zarr_path)
        self.device = "cpu"
        self.dtype = "float32"


class _Ctx:
    """Minimal StageContext stand-in: _run_ff touches only these."""

    is_pf = False

    def __init__(self, layer_dir: Path, zarr_path: Path):
        self.layer_dir = layer_dir
        self.config = _Cfg(zarr_path)


def _layer(tmp_path: Path) -> tuple[Path, Path]:
    layer_dir = tmp_path / "LayerNr_1"
    layer_dir.mkdir()
    zip_path = layer_dir / "sample.MIDAS.zip"
    zip_path.write_bytes(b"not a real zarr")
    return layer_dir, zip_path


def test_ff_transforms_skips_when_outputs_exist(tmp_path: Path, monkeypatch):
    """Both outputs present -> skip without constructing the Pipeline."""
    layer_dir, zip_path = _layer(tmp_path)
    (layer_dir / "InputAll.csv").write_text("YLab ZLab Omega\n")
    (layer_dir / "InputAllExtraInfoFittingAll.csv").write_text("YLab ZLab Omega\n")

    def _boom(*a, **k):                      # must never be reached
        raise AssertionError("transforms(FF) re-ran despite existing outputs")

    monkeypatch.setattr("midas_transforms.Pipeline.from_zarr", _boom)

    res = T._run_ff(_Ctx(layer_dir, zip_path), started=0.0)
    assert res.stage_name == "transforms"
    assert res.metrics["n_scans_ok"] == 1
    assert res.metrics["n_scans_failed"] == 0


@pytest.mark.parametrize(
    "present",
    [(), ("InputAll.csv",), ("InputAllExtraInfoFittingAll.csv",)],
    ids=["neither", "only-InputAll", "only-ExtraInfo"],
)
def test_ff_transforms_runs_when_outputs_incomplete(tmp_path: Path, monkeypatch,
                                                    present):
    """A partial result must NOT be inherited -- half a transform is worse than none."""
    layer_dir, zip_path = _layer(tmp_path)
    for name in present:
        (layer_dir / name).write_text("YLab ZLab Omega\n")

    calls = []

    class _FakePipe:
        def run(self):
            calls.append("run")

        def dump(self, d):
            calls.append("dump")

    monkeypatch.setattr("midas_transforms.Pipeline.from_zarr",
                        lambda *a, **k: _FakePipe())

    T._run_ff(_Ctx(layer_dir, zip_path), started=0.0)
    assert calls == ["run", "dump"], (
        f"expected a real transform with {present or 'no'} output(s) present"
    )
