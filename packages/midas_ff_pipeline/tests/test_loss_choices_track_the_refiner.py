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
    # --loss is DROPPED now, not renamed: midas-pipeline >=0.15.0 has no
    # --refine-loss to rename it to.
    assert "--loss" not in out and "--refine-loss" not in out
    assert "pixel" not in out


def test_a_dropped_flag_takes_its_value_with_it():
    """Both `--flag value` and `--flag=value`.

    Dropping the flag but leaving its value behind would hand midas-pipeline a
    bare positional -- which argparse reads as the subcommand, so the run fails
    somewhere far from the cause.
    """
    import warnings as _w
    from midas_ff_pipeline.cli import translate_argv, _FLAG_DROPPED
    for flag in _FLAG_DROPPED:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            spaced = translate_argv(["run", "--params", "p.txt", flag, "somevalue"])
            equals = translate_argv(["run", "--params", "p.txt", f"{flag}=somevalue"])
        for out in (spaced, equals):
            assert flag not in out
            assert "somevalue" not in out, f"{flag}'s value was left behind"
            assert out.count("--params") == 1 and "p.txt" in out


def test_a_dropped_flag_warns():
    """Silence would look like the flag still worked."""
    import warnings as _w
    from midas_ff_pipeline.cli import translate_argv, _FLAG_DROPPED
    for flag in _FLAG_DROPPED:
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            translate_argv(["run", flag, "x"])
        assert any(flag in str(r.message) for r in rec), f"{flag} dropped silently"


def test_the_shim_cannot_emit_a_flag_the_pipeline_rejects():
    """The structural guard, and the reason this file exists.

    midas-pipeline 0.15.0 removed --refine-loss / --refine-mode /
    --refine-solver along with the Python refiner they configured. The shim was
    still translating --loss/--mode/--solver into them, so every
    `midas-ff-pipeline run --loss ...` died with "unrecognized arguments".

    Rather than fix the three by name, assert the invariant over the whole flag
    surface: anything this shim can emit, midas-pipeline must accept.
    """
    import argparse
    from midas_ff_pipeline import cli as ff_cli
    from midas_pipeline import cli as mp_cli
    from midas_ff_pipeline.cli import _FLAG_RENAMES, _FLAG_DROPPED

    def option_strings(parser, want_sub=None):
        found = set()
        for a in parser._actions:                          # noqa: SLF001
            if isinstance(a, argparse._SubParsersAction):  # noqa: SLF001
                for name, sub in a.choices.items():
                    if want_sub is None or name == want_sub:
                        found |= option_strings(sub)
            else:
                found |= set(a.option_strings)
        return found

    ff_flags = option_strings(ff_cli._build_parser(), "run")   # noqa: SLF001
    mp_flags = option_strings(mp_cli._build_parser(), "run")   # noqa: SLF001

    emitted = {_FLAG_RENAMES.get(f, f) for f in ff_flags if f not in _FLAG_DROPPED}
    orphans = sorted(f for f in emitted if f.startswith("--") and f not in mp_flags)
    assert not orphans, (
        "midas-ff-pipeline can emit flags midas-pipeline does not accept: "
        f"{orphans}. Add them to _FLAG_DROPPED (with the reason) or to "
        "_FLAG_RENAMES."
    )


def test_the_shim_cannot_emit_a_VALUE_the_pipeline_rejects():
    """Matching flag names is not enough -- the choice lists must agree too.

    Caught two live breaks that the name-level check above passes clean:
    ``--indexer-backend python`` and ``--pg-mode spot_aware``. Both flags exist
    on both sides; midas-pipeline 0.15.0 removed the VALUE from each. spot_aware
    was this shim's default, so a user who never passed --pg-mode was the one
    most likely to hit it.
    """
    import argparse
    from midas_ff_pipeline import cli as ff_cli
    from midas_pipeline import cli as mp_cli
    from midas_ff_pipeline.cli import _FLAG_RENAMES, _FLAG_DROPPED

    def actions_for(parser, want_sub):
        found = {}
        for a in parser._actions:                          # noqa: SLF001
            if isinstance(a, argparse._SubParsersAction):  # noqa: SLF001
                for name, sub in a.choices.items():
                    if name == want_sub:
                        found.update(actions_for(sub, want_sub))
            else:
                for opt in a.option_strings:
                    found[opt] = a
        return found

    ff_acts = actions_for(ff_cli._build_parser(), "run")    # noqa: SLF001
    mp_acts = actions_for(mp_cli._build_parser(), "run")    # noqa: SLF001

    bad = []
    for flag, a in ff_acts.items():
        if flag in _FLAG_DROPPED or not flag.startswith("--"):
            continue
        b = mp_acts.get(_FLAG_RENAMES.get(flag, flag))
        if b is None or not getattr(a, "choices", None) or not getattr(b, "choices", None):
            continue
        rejected = sorted(c for c in a.choices if c not in b.choices)
        if rejected:
            bad.append(f"{flag} offers {rejected}, midas-pipeline accepts {sorted(b.choices)}")
        if a.default is not None and a.default not in b.choices:
            bad.append(f"{flag} DEFAULTS to {a.default!r}, which midas-pipeline rejects")
    assert not bad, "shim offers values midas-pipeline rejects:\n  " + "\n  ".join(bad)


def test_ff_pipeline_only_injects_scan_mode_where_it_is_accepted():
    from midas_ff_pipeline.cli import translate_argv
    assert "--scan-mode" not in translate_argv(["status", "--result", "out/"])
    # and never twice
    twice = translate_argv(["run", "--scan-mode", "ff", "--params", "p.txt"])
    assert twice.count("--scan-mode") == 1


def test_the_unified_pipeline_offers_no_loss_at_all():
    """The original invariant, now satisfied structurally rather than by subset.

    This used to assert midas-pipeline's --refine-loss choices were a subset of
    what midas-fit-grain accepts. Since 0.15.0 midas-pipeline offers no loss
    flag at all -- refinement is c-omp only, and the c-omp refiner has no
    configurable loss -- so there is no list left to diverge.

    Kept as an assertion rather than deleted: if a --refine-loss ever comes
    back, this fails and whoever adds it has to restore the subset check
    against midas_fit_grain.losses instead of hand-maintaining a second list.
    """
    from midas_pipeline import cli as mp_cli

    assert _loss_action(mp_cli._build_parser(), "--refine-loss") is None, (
        "midas-pipeline grew a --refine-loss again -- restore the subset "
        "check against midas_fit_grain.losses"
    )


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
