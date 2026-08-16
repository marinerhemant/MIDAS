"""Every subcommand's --help must render.

argparse runs each help string through %-formatting, so a single bare '%'
raises ValueError and takes down the WHOLE --help output for that
subcommand -- not just the offending line. It has happened twice: once
with "unsupported format character 'v'", and again with 't' from
"18.5% to 1.2%" in --refine-mode, which broke `midas-pipeline run --help`
in a released version. Literal percents must be doubled.
"""

from __future__ import annotations

import pytest

from midas_pipeline.cli import _build_parser


def _subparser_names(parser):
    for action in parser._actions:
        if isinstance(action, __import__("argparse")._SubParsersAction):
            return sorted(action.choices)
    return []


def test_top_level_help_renders():
    assert _build_parser().format_help()


@pytest.mark.parametrize("name", _subparser_names(_build_parser()))
def test_subcommand_help_renders(name):
    parser = _build_parser()
    sub = next(
        a for a in parser._actions
        if isinstance(a, __import__("argparse")._SubParsersAction)
    ).choices[name]
    # This is the call that raises on a bare '%'.
    assert sub.format_help()


def test_no_bare_percent_in_any_help_string():
    """Catch it at the source, with a message that says where."""
    import argparse
    parser = _build_parser()
    subs = next(
        a for a in parser._actions
        if isinstance(a, argparse._SubParsersAction)
    ).choices
    bad = []
    for name, sub in subs.items():
        for action in sub._actions:
            h = action.help
            if not h:
                continue
            try:
                h % {"default": "", "prog": "", "choices": ""}
            except (ValueError, KeyError, TypeError):
                bad.append(f"{name} {'/'.join(action.option_strings) or action.dest}")
    assert not bad, (
        "bare '%' in help text (double it to '%%'): " + ", ".join(bad)
    )
