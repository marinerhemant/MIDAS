"""decoy_test / feature_in_raw -- dispositioned into honesty on 2026-09-01, landed 2026-09-10."""
import numpy as np
from midas_defect.honesty import decoy_test, inflated_cell, feature_in_raw

CELL = (3.6116, 3.6116, 19.2516, 90.0, 90.0, 90.0)


def test_a_decoy_that_also_passes_makes_the_acceptance_uninformative():
    r = decoy_test(lambda cell: 20.0, CELL, {"a +3 %": inflated_cell(CELL, 0.03)}, threshold=8)
    assert r["verdict"] == "uninformative" and r["decoys_passing"] == ["a +3 %"]


def test_only_the_real_cell_passing_is_informative():
    sharp = lambda cell: 20.0 if abs(cell[0] - CELL[0]) < 1e-9 else 3.0
    decoys = {"a +3 %": inflated_cell(CELL, 0.03), "a -3 %": inflated_cell(CELL, -0.03)}
    r = decoy_test(sharp, CELL, decoys, threshold=8)
    assert r["verdict"] == "informative" and r["decoys_passing"] == []


def test_a_failing_real_model_is_reported_not_hidden():
    assert decoy_test(lambda cell: 2.0, CELL, {"a +3 %": inflated_cell(CELL, 0.03)}, threshold=8)["verdict"] == "real_fails"


def test_inflated_cell_touches_only_the_named_axes():
    d = inflated_cell(CELL, 0.03)
    assert d[0] == CELL[0] * 1.03 and d[1] == CELL[1] and d[2] == CELL[2]


def _path():
    return np.full(60, 100.0), np.arange(40.0, 160.0, 2.0), np.ones(60), np.zeros(60)


def test_a_ridge_present_in_the_raw_frames_is_found_there():
    raw = np.random.default_rng(1).poisson(400.0, (200, 200)).astype(float)
    raw[98:103, :] += 300.0
    r = feature_in_raw(raw, raw - 400.0, *_path())
    assert r["verdict"] == "in_raw" and r["raw_sigma"] > 5


def test_a_ridge_only_the_processing_shows_is_flagged():
    raw = np.random.default_rng(2).poisson(400.0, (200, 200)).astype(float)
    proc = raw - 400.0
    proc[98:103, :] += 300.0                      # manufactured by the "background" step
    r = feature_in_raw(raw, proc, *_path())
    assert r["verdict"] == "processing_only" and not r["in_raw"]
