"""``c_parity`` must be reachable from the pipeline.

midas_process_grains ships a mode whose whole purpose is to reproduce C
ProcessGrains, and on the datasetA Ni layer it does: 6150 grains against C's
6138, DiffPos median 125.56 um against 125.51, over five different refiner
outputs. The pipeline could not select it -- ``ProcessGrainsMode`` and
``--pg-mode`` both omitted it -- so every pipeline run got ``spot_aware``,
which on the same inputs returns 11514 grains at DiffPos median 273.53 um.

The second half: the stage downgrades any non-legacy mode to ``legacy`` when
FitBest.bin is missing (the c-omp refiner does not write it). c_parity handles
a missing FitBest itself, and downgrading it would substitute a different
algorithm for the one that was explicitly asked for.
"""

from __future__ import annotations

import inspect
import typing


def test_c_parity_is_a_valid_pipeline_mode():
    from midas_pipeline.config import ProcessGrainsMode
    assert "c_parity" in typing.get_args(ProcessGrainsMode)


def _find_action(parser, flag):
    """``--pg-mode`` lives on the ``run`` subparser, not the top-level one."""
    import argparse
    for action in parser._actions:                     # noqa: SLF001
        if flag in getattr(action, "option_strings", []):
            return action
        if isinstance(action, argparse._SubParsersAction):   # noqa: SLF001
            for sub in action.choices.values():
                found = _find_action(sub, flag)
                if found is not None:
                    return found
    return None


def test_cli_offers_c_parity():
    from midas_pipeline import cli
    action = _find_action(cli._build_parser(), "--pg-mode")   # noqa: SLF001
    assert action is not None, "--pg-mode not found on any subparser"
    assert "c_parity" in action.choices


def test_c_parity_is_the_default():
    """EBSD picked it, on shade_LSHR layer 1 against 4328 segmented grains:

        C ProcessGrains  3491 grains  precision 79.8%  recall 64.3%
        c_parity         3492 grains  precision 79.8%  recall 64.4%
        spot_aware       4128 grains  precision 68.2%  recall 65.0%

    Of the 691 grains spot_aware adds, 7.2% have an EBSD partner against
    80.4% for the shared population.
    """
    from midas_pipeline import cli
    from midas_pipeline.config import PipelineConfig
    import dataclasses

    action = _find_action(cli._build_parser(), "--pg-mode")   # noqa: SLF001
    assert action.default == "c_parity"

    fields = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    assert fields["process_grains_mode"].default == "c_parity", (
        "the config default and the CLI default must not disagree"
    )


def test_c_parity_is_not_downgraded_to_legacy():
    from midas_pipeline.stages import process_grains
    src = inspect.getsource(process_grains)
    assert 'mode not in ("legacy", "c_parity")' in src, (
        "a missing FitBest.bin must not silently replace c_parity with legacy"
    )
