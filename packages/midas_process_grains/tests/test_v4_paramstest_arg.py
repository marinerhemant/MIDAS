"""v4 must read the parameter file it was GIVEN, not one it reconstructs.

``run_v4_pipeline`` hardcoded ``layer_dir/"paramstest.txt"``, so the file named
on the command line was discarded. Running with a paramstest carrying a
corrected ``Vsample 1440000`` produced byte-identical grain radii and logged
"no Vsample in paramstest" — it had read the hardcoded name, which had no
Vsample. Silent, and it invalidated the experiment it was run for.
"""

import inspect
from pathlib import Path

from midas_process_grains import v4_pipeline
from midas_process_grains.v4_pipeline import _grab_paramstest_vsample


def test_run_v4_pipeline_accepts_a_paramstest_argument():
    sig = inspect.signature(v4_pipeline.run_v4_pipeline)
    assert "paramstest" in sig.parameters, (
        "v4 must be able to be told which parameter file to read"
    )
    assert sig.parameters["paramstest"].default is None, (
        "it must default to None so existing callers keep the old behaviour"
    )


def test_the_hardcoded_name_is_only_a_fallback():
    src = inspect.getsource(v4_pipeline.run_v4_pipeline)
    assert 'Path(paramstest) if paramstest else' in src, (
        "the caller's file must win; layer_dir/paramstest.txt is the fallback"
    )


def test_cli_passes_the_users_file_through():
    from midas_process_grains import cli
    src = inspect.getsource(cli)
    assert "paramstest=args.param_file" in src, (
        "the CLI must forward the file the user named, not just its directory"
    )


def test_vsample_is_read_from_the_named_file(tmp_path):
    """The concrete failure: a corrected Vsample in a differently-named file."""
    (tmp_path / "paramstest.txt").write_text(
        "Lsd 958874.75\nSpaceGroup 225\nRsample 1800\nHbeam 800\n")
    named = tmp_path / "paramstest_vsample.txt"
    named.write_text(
        "Lsd 958874.75\nSpaceGroup 225\nRsample 1800\nHbeam 800\n"
        "Vsample 1440000\n")

    assert _grab_paramstest_vsample(tmp_path / "paramstest.txt")["Vsample"] == 0.0
    assert _grab_paramstest_vsample(named)["Vsample"] == 1440000.0


def test_vsample_parsing_tolerates_midas_punctuation(tmp_path):
    """MIDAS parameter files carry trailing ';' and '#' comments."""
    p = tmp_path / "ps.txt"
    p.write_text("Vsample 1440000;   # true illuminated volume\n"
                 "Rsample 1800\n")
    assert _grab_paramstest_vsample(p)["Vsample"] == 1440000.0
