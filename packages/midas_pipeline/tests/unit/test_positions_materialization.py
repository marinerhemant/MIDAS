"""P0-2 regression tests: positions.csv materialization + PF hard error.

Both the SOH and Ni campaigns hit the same silent no-op: every early PF
stage soft-skipped on a missing ``LayerNr_N/positions.csv`` and the run
exited 0 having done nothing. The pipeline now materializes the file at
layer setup from ``ScanGeometry`` (file order = acquisition order, sign
per ``--scan-step``), and a missing file is a hard error in PF mode.
FF is untouched (it writes its own 1-row sentinel late, in the
midas-transforms dump).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline import LayerSelection, Pipeline, PipelineConfig, ScanGeometry
from midas_pipeline._pf_scans import _positions_for_layer


def _pf_config(tmp_path: Path, *, scan_step_um: float, n_scans: int = 5) -> PipelineConfig:
    params = tmp_path / "P.txt"
    params.write_text(f"nScans {n_scans}\nBeamSize 5.0\n")
    return PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.pf_uniform(
            n_scans=n_scans, scan_step_um=scan_step_um, beam_size_um=4.0),
        device="cpu",
        dtype="float64",
        layer_selection=LayerSelection(start=1, end=1),
    )


def test_pf_materializes_positions_at_layer_setup(tmp_path: Path):
    cfg = _pf_config(tmp_path, scan_step_um=2.0)
    pipe = Pipeline(cfg)
    ctx = pipe._make_context(1)
    pcsv = ctx.layer_dir / "positions.csv"
    assert pcsv.exists(), "positions.csv must be materialized at layer setup"
    got = np.loadtxt(pcsv).reshape(-1)
    np.testing.assert_allclose(got, cfg.scan.scan_positions, atol=1e-6)
    # Root copy for tools that read <result>/positions.csv.
    assert (Path(cfg.result_path) / "positions.csv").exists()


def test_pf_negative_step_keeps_acquisition_order(tmp_path: Path):
    """Ni-run geometry: 129 → −129 at −1 µm/scan. File order must be the
    acquisition (descending) order, NOT sorted ascending."""
    cfg = _pf_config(tmp_path, scan_step_um=-1.0, n_scans=5)
    pipe = Pipeline(cfg)
    ctx = pipe._make_context(1)
    got = np.loadtxt(ctx.layer_dir / "positions.csv").reshape(-1)
    assert got[0] > got[-1], "acquisition order (descending) must be preserved"
    np.testing.assert_allclose(got, cfg.scan.scan_positions, atol=1e-6)
    # _positions_for_layer must NOT re-sort (it used to, silently
    # reversing the scan↔Y pairing for descending acquisitions).
    arr = _positions_for_layer(ctx.layer_dir)
    np.testing.assert_allclose(arr, cfg.scan.scan_positions, atol=1e-6)


def test_pf_preseeded_positions_not_overwritten(tmp_path: Path):
    cfg = _pf_config(tmp_path, scan_step_um=2.0)
    layer_dir = Path(cfg.result_path) / "LayerNr_1"
    layer_dir.mkdir(parents=True)
    preseeded = "10.5\n8.5\n6.5\n4.5\n2.5\n"
    (layer_dir / "positions.csv").write_text(preseeded)
    Pipeline(cfg)._make_context(1)
    assert (layer_dir / "positions.csv").read_text() == preseeded


def test_ff_does_not_materialize_positions(tmp_path: Path):
    params = tmp_path / "P.txt"
    params.write_text("")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
        device="cpu", dtype="float64",
        layer_selection=LayerSelection(start=1, end=1),
    )
    ctx = Pipeline(cfg)._make_context(1)
    assert not (ctx.layer_dir / "positions.csv").exists(), (
        "FF must not get a layer-setup positions.csv — the transforms dump "
        "writes its own 1-row sentinel"
    )


def test_pf_missing_positions_is_hard_error(tmp_path: Path):
    """Deleting positions.csv after setup must abort the PF stage, not
    soft-skip it (the silent-no-op failure both campaigns hit)."""
    from midas_pipeline.stages import zip_convert

    cfg = _pf_config(tmp_path, scan_step_um=2.0)
    pipe = Pipeline(cfg)
    ctx = pipe._make_context(1)
    (ctx.layer_dir / "positions.csv").unlink()
    (Path(cfg.result_path) / "positions.csv").unlink()
    with pytest.raises(RuntimeError, match="positions.csv"):
        zip_convert.run(ctx)


def test_pf_missing_positions_hard_error_indexing(tmp_path: Path):
    from midas_pipeline.stages import indexing

    cfg = _pf_config(tmp_path, scan_step_um=2.0)
    pipe = Pipeline(cfg)
    ctx = pipe._make_context(1)
    (ctx.layer_dir / "positions.csv").unlink()
    with pytest.raises(RuntimeError, match="positions.csv"):
        indexing.run(ctx)
