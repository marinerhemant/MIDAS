"""N4 regression tests: ``--only`` allowlist dependency validation.

Both the SOH and Ni campaigns independently produced broken recons from
an ``--only`` set (from the same handoff doc) that omitted
merge_scans/seeding/refinement — every omitted stage soft-skipped and
the run exited 0. The pipeline now hard-errors unless each selected
stage's upstream dependencies are selected too, already complete in the
provenance store, or explicitly --skip'ed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline import LayerSelection, Pipeline, PipelineConfig, ScanGeometry
from midas_pipeline.pipeline import stage_deps_for


def _pf_cfg(tmp_path: Path, **kw) -> PipelineConfig:
    params = tmp_path / "P.txt"
    params.write_text("nScans 5\nBeamSize 5.0\n")
    return PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=4.0),
        device="cpu", dtype="float64",
        layer_selection=LayerSelection(start=1, end=1),
        **kw,
    )


def test_only_missing_deps_is_hard_error(tmp_path: Path):
    """The campaign failure mode: --only binning..find_grains without
    merge_scans/seeding."""
    cfg = _pf_cfg(tmp_path, only_stages=["binning", "indexing", "find_grains"])
    with pytest.raises(RuntimeError, match="omits required upstream"):
        Pipeline(cfg).run()


def test_only_complete_set_passes_guard(tmp_path: Path):
    """A front-to-back --only set has no unmet deps and must run."""
    cfg = _pf_cfg(tmp_path, only_stages=["zip_convert", "hkl"])
    results = Pipeline(cfg).run()          # both stages stub/skip harmlessly
    assert len(results) == 1


def test_only_satisfied_by_provenance(tmp_path: Path):
    """Resume workflow: deps already complete in the provenance store."""
    from midas_pipeline.provenance import ProvenanceStore

    cfg = _pf_cfg(tmp_path, only_stages=["indexing", "refinement"])
    pipe = Pipeline(cfg)
    # Mark the upstream deps complete, as a previous run would have.
    layer_dir = cfg.layer_dir(1)
    layer_dir.mkdir(parents=True, exist_ok=True)
    store = ProvenanceStore(layer_dir)
    for dep in ("binning", "indexing"):
        store.record(dep, started_at=0.0, finished_at=1.0,
                     inputs={}, outputs={})
    results = pipe.run()
    assert len(results) == 1


def test_skip_form_never_triggers_guard(tmp_path: Path):
    """The recommended --skip form (doc correction 3) is untouched."""
    cfg = _pf_cfg(tmp_path, skip_stages=["voxel_cleanup", "sinogen",
                                         "reconstruct", "fuse", "potts",
                                         "em_refine"])
    results = Pipeline(cfg).run()
    assert len(results) == 1


def test_stage_deps_mode_scoped():
    # binning depends on merge_scans+seeding in PF but transforms in FF.
    assert "merge_scans" in stage_deps_for("binning", "pf")
    assert stage_deps_for("binning", "ff") == ("transforms",)
    # PF-only deps are filtered out of FF dep lists.
    assert stage_deps_for("seeding", "ff") == ()
