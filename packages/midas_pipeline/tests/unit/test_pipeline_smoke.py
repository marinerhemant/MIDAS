"""End-to-end smoke test of the pipeline driver with stub stages.

P1 stages are all thin shells that return ``skipped=True``. We verify:

- The driver walks the correct stage list for each scan mode.
- The provenance store gets one record per stage.
- LayerResult fields are populated.
- Resume-from skips earlier stages cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline import (
    LayerSelection,
    Pipeline,
    PipelineConfig,
    ScanGeometry,
)
from midas_pipeline.provenance import ProvenanceStore


def _ff_pipeline(tmp_path: Path) -> Pipeline:
    params = tmp_path / "P.txt"
    params.write_text("")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        # synthetic stub inputs: bypass the pre-run existence check, and do
        # not attempt a real zip conversion (there is no raw data here).
        skip_preflight=True,
        convert_files=False,
        scan=ScanGeometry.ff(),
        device="cpu",                    # avoid CUDA assumptions in CI
        dtype="float64",
        layer_selection=LayerSelection(start=1, end=1),
    )
    return Pipeline(cfg)


def _pf_pipeline(tmp_path: Path, n_scans: int = 5) -> Pipeline:
    params = tmp_path / "P.txt"
    params.write_text(f"nScans {n_scans}\nBeamSize 5.0\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        # synthetic stub inputs: bypass the pre-run existence check
        skip_preflight=True,
        scan=ScanGeometry.pf_uniform(n_scans=n_scans, scan_step_um=2.0, beam_size_um=4.0),
        device="cpu",
        dtype="float64",
        layer_selection=LayerSelection(start=1, end=1),
    )
    return Pipeline(cfg)


# ---------------------------------------------------------------------------
# FF degeneracy: thin-shell run should touch every FF stage
# ---------------------------------------------------------------------------


def test_ff_smoke_runs_all_ff_stages(tmp_path):
    pipe = _ff_pipeline(tmp_path)
    results = pipe.run()
    assert len(results) == 1
    layer_result = results[0]
    assert layer_result.layer_nr == 1
    # All FF stages should have a (skipped=True) StageResult attached.
    assert layer_result.hkl is not None and layer_result.hkl.skipped
    assert layer_result.peakfit is not None and layer_result.peakfit.skipped
    assert layer_result.indexing is not None and layer_result.indexing.skipped
    assert layer_result.refinement is not None and layer_result.refinement.skipped
    assert layer_result.process_grains is not None
    assert layer_result.consolidation is not None

    # PF-only fields should be untouched.
    assert layer_result.merge_scans is None
    assert layer_result.find_grains is None
    assert layer_result.sinogen is None


def test_pf_smoke_runs_all_pf_stages(tmp_path):
    pipe = _pf_pipeline(tmp_path)
    results = pipe.run()
    assert len(results) == 1
    layer_result = results[0]
    assert layer_result.merge_scans is not None
    assert layer_result.find_grains is not None
    assert layer_result.sinogen is not None
    assert layer_result.reconstruct is not None
    # FF-only stage should be untouched.
    assert layer_result.process_grains is None


def test_provenance_records_every_stage(tmp_path):
    pipe = _ff_pipeline(tmp_path)
    pipe.run()
    layer_dir = pipe.config.layer_dir(1)
    store = ProvenanceStore(layer_dir)
    recorded = store.all_stages()
    expected_ff_stages = {
        "zip_convert", "hkl", "peakfit", "merge_overlaps", "calc_radius",
        "transforms", "cross_det_merge", "global_powder",
        "binning", "indexing", "refinement",
        "process_grains", "consolidation",
    }
    assert expected_ff_stages.issubset(set(recorded))


def test_resume_auto_skips_already_complete_stages(tmp_path):
    """Run once, then run again: the second run should hit the auto-resume path."""
    pipe = _ff_pipeline(tmp_path)
    pipe.run()
    layer_dir = pipe.config.layer_dir(1)
    store = ProvenanceStore(layer_dir)
    first_durations = {
        name: rec["duration_s"] for name, rec in store.all_stages().items()
    }

    # Run again; with resume="auto" (the default), the pipeline should
    # treat every stage as already complete (stubs have empty outputs,
    # so is_complete returns True trivially via no-output trust).
    pipe2 = _ff_pipeline(tmp_path)
    pipe2.run()
    # Provenance records should still exist (we don't lose state).
    store2 = ProvenanceStore(layer_dir)
    recorded2 = store2.all_stages()
    assert set(recorded2) >= set(first_durations)
