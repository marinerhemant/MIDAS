"""The FF c-omp refinement stage writes RelFitRMSE.bin only when RelFitRMSEWeightR0 > 0 (last occurrence wins)."""
import numpy as np

from midas_pipeline.stages import refinement


def test_hook_writes_only_when_requested(tmp_path, monkeypatch):
    calls = []
    import midas_transforms.io.csv as tcsv
    monkeypatch.setattr(tcsv, "write_rel_fit_rmse_bin",
                        lambda d: calls.append(d) or (d / "RelFitRMSE.bin", 1, 1))
    p = tmp_path / "paramstest_refine_comp.txt"
    p.write_text("Lsd 1000;\nRelFitRMSEWeightR0 0.69;\nRelFitRMSEWeightR0 0;\n")
    refinement._write_rel_fit_rmse_if_requested(p, tmp_path)
    assert calls == []                       # last occurrence (0) wins -> off
    p.write_text("RelFitRMSEWeightR0 0;\nRelFitRMSEWeightR0 0.69;\n")
    refinement._write_rel_fit_rmse_if_requested(p, tmp_path)
    assert calls == [tmp_path]
    p.write_text("Lsd 1000;\n")
    refinement._write_rel_fit_rmse_if_requested(p, tmp_path)
    assert calls == [tmp_path]               # absent -> off


def test_key_is_propagated_to_the_comp_paramstest():
    from midas_pipeline.stages import _comp_params
    assert "RelFitRMSEWeightR0" in _comp_params._INDEXER_KEYS
