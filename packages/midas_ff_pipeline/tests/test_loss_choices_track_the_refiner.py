"""A caller's --loss choices must be a subset of what the refiner accepts.

Reported from a real run: ``midas-ff-pipeline run`` could not complete
refinement on a single-detector dataset with default flags. It passed
``--loss pixel``; ``midas-fit-grain`` accepts only
{full3d, angular, internal_angle}. The lists had diverged:

    midas_fit_grain    --loss  {full3d, angular, internal_angle}  default full3d
    midas_ff_pipeline  --loss  {pixel,  angular, internal_angle}  default pixel

``pixel`` was retired from the refiner (2-D in y/z, omitted omega, so
orientation drifted freely) and this package kept offering it AND defaulting
to it. Multi-detector runs survived only because a separate branch rewrote
pixel→angular before dispatch; single-detector runs went straight into the
refiner's argparse and died.

The cure is structural: the choices now come from
``midas_fit_grain.losses``, so there is no second list to rot. These tests
hold that line for any future caller.
"""

from __future__ import annotations

from midas_fit_grain.losses import (
    DEFAULT_LOSS,
    DEPRECATED_LOSSES,
    LOSS_CHOICES,
    MULTIDET_LOSS,
    PANEL_DEPENDENT_LOSSES,
    resolve,
)


def _loss_action(parser, flag="--loss"):
    import argparse
    for a in parser._actions:                          # noqa: SLF001
        if flag in getattr(a, "option_strings", []):
            return a
        if isinstance(a, argparse._SubParsersAction):  # noqa: SLF001
            for sub in a.choices.values():
                got = _loss_action(sub, flag)
                if got is not None:
                    return got
    return None


def test_ff_pipeline_no_longer_dispatches_anything_itself():
    """midas-ff-pipeline is a shim: it rewrites argv and delegates.

    The divergence this file is named for is now impossible for this package,
    because it has no dispatch of its own left to diverge.
    """
    from midas_ff_pipeline.cli import translate_argv
    out = translate_argv(["run", "--params", "p.txt", "--loss", "pixel"])
    assert out[:3] == ["run", "--scan-mode", "ff"]
    assert "--refine-loss" in out
    assert "full3d" in out, "a retired loss name must be resolved, not forwarded"
    assert "--loss" not in out and "pixel" not in out


def test_ff_pipeline_renames_every_flag_that_needs_it():
    from midas_ff_pipeline.cli import translate_argv, _FLAG_RENAMES
    out = translate_argv(["run", "--loss", "angular",
                          "--mode", "c_recipe", "--solver", "lm"])
    for old, new in _FLAG_RENAMES.items():
        assert old not in out, f"{old} was not translated"
        assert new in out
    # values survive
    assert "angular" in out and "c_recipe" in out and "lm" in out


def test_ff_pipeline_only_injects_scan_mode_where_it_is_accepted():
    from midas_ff_pipeline.cli import translate_argv
    assert "--scan-mode" not in translate_argv(["status", "--result", "out/"])
    # and never twice
    twice = translate_argv(["run", "--scan-mode", "ff", "--params", "p.txt"])
    assert twice.count("--scan-mode") == 1


def test_the_unified_pipeline_offers_nothing_the_refiner_rejects():
    from midas_pipeline import cli as mp_cli
    from midas_fit_grain import cli as fg_cli

    caller = set(_loss_action(mp_cli._build_parser(), "--refine-loss").choices)
    refiner = set(_loss_action(fg_cli.build_parser()).choices)  # noqa: SLF001
    assert caller <= refiner


def test_every_caller_default_is_runnable():
    """The reported failure was a DEFAULT, so no flag was needed to hit it."""
    from midas_ff_pipeline import cli as ff_cli
    from midas_fit_grain import cli as fg_cli
    from midas_ff_pipeline.config import PipelineConfig as FFConfig
    from midas_pipeline.config import RefinementConfig

    refiner = set(_loss_action(fg_cli.build_parser()).choices)  # noqa: SLF001
    cli_default = _loss_action(ff_cli._build_parser()).default
    assert resolve(cli_default)[0] in refiner
    assert FFConfig.__dataclass_fields__["refine_loss"].default in refiner
    assert RefinementConfig.__dataclass_fields__["loss"].default in refiner


def test_config_and_cli_defaults_agree():
    from midas_ff_pipeline import cli as ff_cli
    from midas_ff_pipeline.config import PipelineConfig as FFConfig
    assert (_loss_action(ff_cli._build_parser()).default
            == FFConfig.__dataclass_fields__["refine_loss"].default
            == DEFAULT_LOSS)


# ------------------------------------------------------- deprecated names ---

def test_a_retired_name_is_substituted_not_forwarded():
    resolved, note = resolve("pixel")
    assert resolved == "full3d"
    assert note and "retired" in note


def test_a_current_name_passes_through_untouched():
    for name in LOSS_CHOICES:
        assert resolve(name) == (name, None)


def test_an_unknown_name_is_left_for_the_refiner_to_reject():
    """Don't invent a second error message for a typo."""
    assert resolve("banana") == ("banana", None)


# --------------------------------------------------------- multi-detector ---

def test_the_default_loss_is_panel_dependent():
    """Which is why the multi-detector swap has to still fire on it.

    The swap used to test `loss == "pixel"`. full3d is pixel-based too
    (y_pixel, z_pixel, Δω·r_px), and is now the default — so a name-based
    guard would silently stop firing exactly when it started mattering.
    """
    assert DEFAULT_LOSS in PANEL_DEPENDENT_LOSSES
    assert MULTIDET_LOSS not in PANEL_DEPENDENT_LOSSES


def test_both_stages_key_the_swap_on_the_set_not_a_name():
    import inspect
    from midas_ff_pipeline.stages import refine as ff_refine
    from midas_pipeline.stages import refinement as mp_refine

    for mod in (ff_refine, mp_refine):
        src = inspect.getsource(mod)
        assert "PANEL_DEPENDENT_LOSSES" in src, (
            f"{mod.__name__} must test the set, not a single loss name"
        )
