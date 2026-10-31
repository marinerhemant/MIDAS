"""LayerResult.n_grains must not raise when process_grains was skipped (#75 follow-up).

A run that indexes nothing leaves a plain ``StageResult`` (skipped) in
``process_grains``; ``n_grains`` used to read ``.n_grains`` off it and raise
AttributeError, hiding the real outcome (0 grains) behind a traceback.
"""
from __future__ import annotations

from midas_pipeline.results import LayerResult, ProcessGrainsResult, StageResult


def _stage(name, **kw):
    return StageResult(stage_name=name, started_at=0.0, finished_at=0.0,
                       duration_s=0.0, **kw)


def test_skipped_process_grains_counts_zero():
    r = LayerResult(layer_nr=1, layer_dir="x")
    r.process_grains = _stage("process_grains", skipped=True)
    assert r.n_grains == 0


def test_a_real_count_is_returned():
    r = LayerResult(layer_nr=1, layer_dir="x")
    r.process_grains = ProcessGrainsResult(
        stage_name="process_grains", started_at=0.0, finished_at=0.0,
        duration_s=0.0, grains_csv="g.csv", n_grains=7)
    assert r.n_grains == 7


def test_no_stage_counts_zero():
    assert LayerResult(layer_nr=1, layer_dir="x").n_grains == 0


def test_count_is_read_from_the_stage_metrics():
    r = LayerResult(layer_nr=1, layer_dir="x")
    r.process_grains = _stage("process_grains", metrics={"scan_mode": "ff", "n_grains": 12})
    assert r.n_grains == 12


def test_grains_df_reads_grains_csv_and_is_empty_without_grains(tmp_path):
    r = LayerResult(layer_nr=1, layer_dir=str(tmp_path))
    assert r.grains_df().empty                                  # no file
    (tmp_path / "Grains.csv").write_text(
        "%NumGrains 2\n%GrainID\tX\tY\tZ\n1\t1.0\t2.0\t3.0\n2\t4.0\t5.0\t6.0\n")
    df = r.grains_df()
    assert list(df.columns) == ["GrainID", "X", "Y", "Z"] and len(df) == 2
    (tmp_path / "Grains.csv").write_text("%NumGrains 0\n%GrainID\tX\tY\tZ\n")
    assert r.grains_df().empty and list(r.grains_df().columns) == ["GrainID", "X", "Y", "Z"]
