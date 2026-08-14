"""c_parity must read the parameter file it was GIVEN.

Third instance of one defect. ``run_v4_pipeline`` hardcoded
``layer_dir/"paramstest.txt"`` and discarded the file named on the command
line (see test_v4_paramstest_arg.py); ``run_c_parity_pipeline_from_disk`` did
the same, and died outright with

    FileNotFoundError: No such file or directory: 'paramstest.txt'

when handed a parameter file under any other name. A hardcoded input name is
either a crash or -- worse, as in the v4 case -- a silently wrong answer.

The second half: ``confidence_min`` was hardcoded to 0.05 rather than read
from the parameter file's ``Completeness``, so c_parity could not reproduce
the C run it exists to replicate.
"""

from __future__ import annotations

import inspect

from midas_process_grains.compute import c_parity_run
from midas_process_grains.compute.c_parity_run import run_c_parity_pipeline_from_disk


def test_accepts_a_paramstest_argument():
    sig = inspect.signature(run_c_parity_pipeline_from_disk)
    assert "paramstest" in sig.parameters
    assert sig.parameters["paramstest"].default is None, (
        "must default to None so existing callers keep the old behaviour"
    )


def test_the_hardcoded_name_is_only_a_fallback():
    src = inspect.getsource(run_c_parity_pipeline_from_disk)
    assert 'Path(paramstest) if paramstest else' in src
    assert 'read_paramstest_pg(ps_path)' in src, (
        "the resolved path must be what is actually read"
    )


def test_cli_passes_the_users_file_through():
    from midas_process_grains import cli
    assert "paramstest=args.param_file" in inspect.getsource(cli)


def test_confidence_min_defaults_to_the_parameter_files_completeness():
    sig = inspect.signature(run_c_parity_pipeline_from_disk)
    assert sig.parameters["confidence_min"].default is None, (
        "a hardcoded 0.05 cannot reproduce a run that asked for Completeness 0.5"
    )
    src = inspect.getsource(run_c_parity_pipeline_from_disk)
    assert "params.Completeness" in src


def test_an_explicit_confidence_min_still_wins():
    """The gate is derived only when the caller did not specify one."""
    src = inspect.getsource(run_c_parity_pipeline_from_disk)
    assert "if confidence_min is None:" in src


# --------------------------------------------------------- MinNrSpots ------

def test_min_nr_spots_defaults_to_the_parameter_file():
    """Found by the end-to-end validation run, not by a unit test.

    The CLI hardcoded ``min_nr_spots=1`` whenever ``--min-nr-spots`` was
    absent, so a parameter file saying ``MinNrSpots 3`` — which the pipeline
    goes out of its way to propagate (stages/_comp_params.selection_paramstest)
    — was read into ProcessGrainsParams and then thrown away.

    Measured on the datasetA Ni layer, one pipeline run, all defaults:
    23138 grains against ~6180 expected. C ProcessGrains splits the same way
    on the same input: 23710 at MinNrSpots=1, 6147 at MinNrSpots=3.
    """
    sig = inspect.signature(run_c_parity_pipeline_from_disk)
    assert sig.parameters["min_nr_spots"].default is None, (
        "must default to None so the parameter file is consulted"
    )
    src = inspect.getsource(run_c_parity_pipeline_from_disk)
    assert "if min_nr_spots is None:" in src
    assert "params.MinNrSpots" in src


def test_cli_no_longer_substitutes_one():
    from midas_process_grains import cli
    src = inspect.getsource(cli)
    assert "min_nr_spots=args.min_nr_spots," in src, (
        "the CLI must pass None through, not a hardcoded floor"
    )
    assert "if args.min_nr_spots is not None else 1" not in src


def test_an_explicit_flag_still_wins():
    src = inspect.getsource(run_c_parity_pipeline_from_disk)
    assert "if min_nr_spots is None:" in src, (
        "the file is consulted only when the caller gave nothing"
    )
