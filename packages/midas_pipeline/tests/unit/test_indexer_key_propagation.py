"""The keys the C indexer reads and paramstest never carried.

Measured on the Ce dhcp run, 2026-08-24: a ``Parameters.txt`` setting
``ConfidenceMetric weighted`` produced a ``Grains.csv`` byte-identical to raw,
because the key never reached ``paramstest_comp.txt`` -- the file the binary is
actually invoked with. ``BigDetSize`` (Phase 1's detector mask) and
``MinSeedGrainRadius`` had the same problem. Each feature was implemented and
unit-gated in the C, and each was unreachable from a parameter file.

A silently-ignored key is worse than an unimplemented one: the run looks like
it honoured the request.
"""
from __future__ import annotations

import pytest

from midas_pipeline.stages._comp_params import (
    _INDEXER_KEYS,
    comp_backend_paramstest,
)


@pytest.fixture
def files(tmp_path):
    ps = tmp_path / "paramstest.txt"
    ps.write_text("OutputFolder /old\nResultFolder /old\nLsd 1666219.6;\n")
    pf = tmp_path / "Parameters.txt"
    pf.write_text(
        "# a comment line\n"
        "ConfidenceMetric weighted\n"
        "ForbiddenF2Threshold 0.01\n"
        "BigDetSize 2048\n"
        "MinSeedGrainRadius 1.5\n"
        "Completeness 0.5\n"          # a process-grains key, not an indexer one
    )
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    return ps, pf, layer


def test_the_keys_reach_the_file_the_binary_is_invoked_with(files):
    ps, pf, layer = files
    txt = comp_backend_paramstest(ps, layer, params_file=pf).read_text()
    assert "ConfidenceMetric weighted" in txt
    assert "ForbiddenF2Threshold 0.01" in txt
    assert "BigDetSize 2048" in txt
    assert "MinSeedGrainRadius 1.5" in txt


def test_without_the_params_file_nothing_is_invented(files):
    """The old behaviour, kept for any caller that has no user file: propagate
    nothing rather than guess."""
    ps, _, layer = files
    txt = comp_backend_paramstest(ps, layer).read_text()
    for k in _INDEXER_KEYS:
        assert k not in txt


def test_only_indexer_keys_are_propagated(files):
    """Completeness is a process-grains threshold and has its own propagation
    path; duplicating it here would be two sources for one number."""
    ps, pf, layer = files
    txt = comp_backend_paramstest(ps, layer, params_file=pf).read_text()
    assert "Completeness" not in txt


def test_a_value_already_in_paramstest_is_not_overridden(files):
    """FitSetup's own value wins, matching how every MIDAS parser treats a
    duplicated key (first occurrence)."""
    ps, pf, layer = files
    ps.write_text("OutputFolder /old\nResultFolder /old\n"
                  "ConfidenceMetric raw\n")
    txt = comp_backend_paramstest(ps, layer, params_file=pf).read_text()
    assert txt.count("ConfidenceMetric") == 1
    assert "ConfidenceMetric raw" in txt


def test_the_folder_rewrite_still_happens(files):
    """The function's original job must not regress."""
    ps, pf, layer = files
    txt = comp_backend_paramstest(ps, layer, params_file=pf).read_text()
    assert f"OutputFolder {layer / 'Output'}" in txt
    assert f"ResultFolder {layer / 'Results'}" in txt
    assert "Lsd 1666219.6;" in txt
    assert "/old" not in txt


def test_comments_in_the_user_file_are_not_propagated(files):
    ps, pf, layer = files
    txt = comp_backend_paramstest(ps, layer, params_file=pf).read_text()
    assert "a comment line" not in txt


def test_a_missing_params_file_is_not_an_error(files):
    ps, _, layer = files
    txt = comp_backend_paramstest(ps, layer,
                                  params_file=layer / "nope.txt").read_text()
    assert "ConfidenceMetric" not in txt
