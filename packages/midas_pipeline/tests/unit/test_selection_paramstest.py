"""The user's grain-selection thresholds must reach process-grains.

``paramstest.txt`` is written for the indexer and the refiner. Neither has any
use for ``MinNrSpots`` or ``Completeness``, so FitSetup does not write them --
and the pipeline then handed that same file to process-grains, which fell back
to its own defaults. The user's parameter file said ``MinNrSpots 3`` and
``Completeness 0.5``; process-grains ran with neither.

Measured on the datasetA Ni layer, one refiner output, one binary:

    ProcessGrains -paramFN <ps_ni.txt keys present> ....  6147 grains
    ProcessGrains -paramFN paramstest.txt ............... 23710 grains
    classical chain (handed the zarr, which carries them)  6132 grains

The 4x gap read as an algorithmic difference between the C and python
implementations of process-grains for most of a session. It was a parameter
that never arrived.
"""

from __future__ import annotations

from pathlib import Path

from midas_pipeline.stages._comp_params import selection_paramstest

_PS = ("LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
       "SpaceGroup 225;\n"
       "OutputFolder /run/LayerNr_1\n"
       "ResultFolder /run/LayerNr_1\n")

_USER = ("SpaceGroup 225\n"
         "Completeness 0.5\n"
         "MinNrSpots 3\n"
         "RawFolder /somewhere/else\n")


def _write(d: Path, name: str, text: str) -> Path:
    p = d / name
    p.write_text(text)
    return p


def test_the_missing_thresholds_are_propagated(tmp_path):
    ps = _write(tmp_path, "paramstest.txt", _PS)
    user = _write(tmp_path, "ps_ni.txt", _USER)

    out = selection_paramstest(ps, user, tmp_path)
    text = out.read_text()

    assert "Completeness 0.5" in text
    assert "MinNrSpots 3" in text
    assert out != ps, "a new file must be written, not the original mutated"
    assert ps.read_text() == _PS, "the source paramstest must be left alone"


def test_unrelated_keys_are_not_dragged_along(tmp_path):
    """Only grain-selection keys move; RawFolder would break the run."""
    ps = _write(tmp_path, "paramstest.txt", _PS)
    user = _write(tmp_path, "ps_ni.txt", _USER)
    assert "RawFolder" not in selection_paramstest(ps, user, tmp_path).read_text()


def test_a_value_already_in_paramstest_is_not_overridden(tmp_path):
    ps = _write(tmp_path, "paramstest.txt", _PS + "MinNrSpots 7\n")
    user = _write(tmp_path, "ps_ni.txt", _USER)
    text = selection_paramstest(ps, user, tmp_path).read_text()
    assert "MinNrSpots 7" in text
    assert "MinNrSpots 3" not in text


def test_nothing_to_add_returns_the_original_unchanged(tmp_path):
    """A well-formed run must not gain a file or a new path."""
    ps = _write(tmp_path, "paramstest.txt", _PS + "MinNrSpots 3\nCompleteness 0.5\n")
    user = _write(tmp_path, "ps_ni.txt", _USER)
    assert selection_paramstest(ps, user, tmp_path) == ps
    assert not (tmp_path / "paramstest_pg.txt").exists()


def test_missing_or_absent_user_file_is_survivable(tmp_path):
    ps = _write(tmp_path, "paramstest.txt", _PS)
    assert selection_paramstest(ps, None, tmp_path) == ps
    assert selection_paramstest(ps, tmp_path / "nope.txt", tmp_path) == ps


def test_comments_and_semicolons_are_handled(tmp_path):
    ps = _write(tmp_path, "paramstest.txt", _PS)
    user = _write(tmp_path, "ps_ni.txt",
                  "# thresholds\nCompleteness 0.5;  # keep good grains\n")
    assert "Completeness 0.5;" in selection_paramstest(ps, user, tmp_path).read_text()


def test_the_stage_calls_it(tmp_path):
    """A helper nothing calls is the bug it was written to fix."""
    import inspect
    from midas_pipeline.stages import process_grains
    src = inspect.getsource(process_grains)
    assert "selection_paramstest(" in src
    assert "params_file" in src
