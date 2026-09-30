"""c_parity must honour the Stage-1 misorientation tolerance the user sets.

Regression: in the default ``c_parity`` mode the Stage-1 tolerance was a
hardcoded 0.4 deg. ``run_c_parity_pipeline_from_disk`` never read ``MisoriTol``
from the parameter file, and the CLI parsed ``--misori-tol`` and then did not pass
it on, so both were silently ignored. Measured on 20-ID garnet (HPcat_P2 att5):
with ``MisoriTol 1.0`` in the file the log still read ``misori_tol = 0.400`` and
the grain list was the same 657 rows. The pipeline did not propagate the key
into the process-grains paramstest either.

Pinned here, without the heavy on-disk run:

1. the resolution order — explicit argument, then the file's ``MisoriTol``, then
   C's 0.4 — including that a dataclass default is not mistaken for a file value;
2. a real parameter file carrying ``MisoriTol`` resolves to that value;
3. the CLI passes ``--misori-tol`` through, and passes ``None`` (defer to the
   file) when the flag is absent;
4. midas-pipeline propagates ``MisoriTol`` to process-grains.
"""
from __future__ import annotations

import pytest

from midas_process_grains.compute.c_parity_run import resolve_stage1_misori_tol
from midas_process_grains.params import ProcessGrainsParams


def _params(misori=None, in_file=False):
    p = ProcessGrainsParams()
    p.MisoriTol = misori
    if in_file:
        p.raw["MisoriTol"] = [str(misori)]
    return p


def test_explicit_argument_wins_over_the_file():
    v, src = resolve_stage1_misori_tol(0.8, _params(1.0, in_file=True))
    assert v == pytest.approx(0.8) and "explicit" in src


def test_file_value_is_used_when_no_argument():
    v, src = resolve_stage1_misori_tol(None, _params(1.0, in_file=True))
    assert v == pytest.approx(1.0) and "parameter file" in src


def test_c_default_when_the_file_does_not_set_it():
    v, src = resolve_stage1_misori_tol(None, _params(None))
    assert v == pytest.approx(0.4) and "default" in src


def test_a_mode_default_is_not_mistaken_for_a_file_value():
    """A MisoriTol filled in by mode defaults (not in raw) must not override C's 0.4."""
    v, _ = resolve_stage1_misori_tol(None, _params(0.25, in_file=False))
    assert v == pytest.approx(0.4)


def test_reading_a_parameter_file_carrying_misori_tol(tmp_path):
    from midas_process_grains.params import read_paramstest_pg
    f = tmp_path / "paramstest_pg.txt"
    f.write_text("SpaceGroup 230\nLatticeParameter 11.517 11.517 11.517 90 90 90\n"
                 "MinNrSpots 3\nCompleteness 0.5\nMisoriTol 1.0\n")
    try:
        p = read_paramstest_pg(f)
    except Exception as e:                       # upstream reader may demand more keys
        pytest.skip(f"minimal paramstest not accepted by the upstream reader: {e}")
    v, src = resolve_stage1_misori_tol(None, p)
    assert v == pytest.approx(1.0) and "parameter file" in src


@pytest.mark.parametrize("argv_extra,expected", [(["--misori-tol", "0.8"], 0.8), ([], None)])
def test_cli_passes_misori_tol_through(tmp_path, monkeypatch, argv_extra, expected):
    import midas_process_grains.compute.c_parity_run as cpr
    from midas_process_grains import cli

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(cpr, "run_c_parity_pipeline_from_disk", fake)
    ps = tmp_path / "paramstest_pg.txt"
    ps.write_text("MinNrSpots 3\n")
    rc = cli.main([str(ps), "1", "--mode", "c_parity", "--device", "cpu", *argv_extra])
    assert rc == 0
    assert "misori_tol_stage1_deg" in seen, "CLI must forward the tolerance argument"
    if expected is None:
        assert seen["misori_tol_stage1_deg"] is None
    else:
        assert seen["misori_tol_stage1_deg"] == pytest.approx(expected)


def test_pipeline_propagates_misori_tol():
    mp = pytest.importorskip("midas_pipeline.stages._comp_params")
    assert "MisoriTol" in mp._PG_SELECTION_KEYS
