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

    **Reformulated at 0.7.0.** This used to enumerate the shim's own argparse
    parser and check every flag it could emit. The shim has no parser any more
    -- it rewrites argv as strings and hands it to midas-pipeline -- so the
    surface it can *introduce* is exactly the values of ``_FLAG_RENAMES``.
    Those are what must exist on the other side.
    """
    import argparse
    from midas_pipeline import cli as mp_cli
    from midas_ff_pipeline.cli import _FLAG_RENAMES

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

    mp_flags = option_strings(mp_cli._build_parser(), "run")   # noqa: SLF001
    orphans = sorted(t for t in _FLAG_RENAMES.values()
                     if t.startswith("--") and t not in mp_flags)
    assert not orphans, (
        "_FLAG_RENAMES rewrites to flags midas-pipeline does not accept: "
        f"{orphans}. A rename must point at a flag that exists."
    )


def test_a_dropped_flag_is_one_the_pipeline_really_rejects():
    """The same invariant from the other side, and it is the new one.

    Dropping a flag silently discards what the user asked for, so each entry in
    ``_FLAG_DROPPED`` has to be a flag midas-pipeline genuinely will not take.
    If the pipeline ever grows one back, this fails and the entry should go --
    otherwise we swallow a flag that would have worked.

    The old value-level twin of this test (``--indexer-backend python``,
    ``--pg-mode spot_aware``) is gone with the shim's parser: it has no choice
    lists and no defaults of its own any more, so it cannot originate a bad
    VALUE -- only pass one through for midas-pipeline to reject on its own
    terms, which is the correct owner of that check.
    """
    import argparse
    from midas_pipeline import cli as mp_cli
    from midas_ff_pipeline.cli import _FLAG_DROPPED

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

    mp_flags = option_strings(mp_cli._build_parser(), "run")   # noqa: SLF001
    resurrected = sorted(f for f in _FLAG_DROPPED if f in mp_flags)
    assert not resurrected, (
        "these flags are in _FLAG_DROPPED but midas-pipeline accepts them "
        f"again: {resurrected}. Remove them from _FLAG_DROPPED rather than "
        "silently swallowing a flag the user asked for."
    )


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


# NOTE: three tests were removed at 0.7.0 when the shim's stage tree was
# deleted -- `test_every_caller_default_is_runnable` and
# `test_config_and_cli_defaults_agree` both asserted against
# `midas_ff_pipeline.config.PipelineConfig`, and
# `test_both_stages_key_the_swap_on_the_set_not_a_name` against
# `midas_ff_pipeline.stages.refine`. None of those modules exists any more;
# the midas_pipeline half of each assertion lives in that package's own
# suite. What remains here is the shim's real contract: argv in, argv out,
# and never a flag or value the pipeline would reject.
