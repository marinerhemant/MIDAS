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


def test_every_midas_console_script_renders_help_on_stdout():
    """`--help` must exit 0 and print to STDOUT, for every MIDAS command.

    Two distinct bugs motivated this, and neither was caught by building a
    parser in a test:

    * an unescaped ``%`` made argparse raise while FORMATTING the help, so
      ``midas-pipeline run --help`` traced back instead of printing (live in
      0.12.0-0.14.0);
    * ``midas-transforms --help`` fell through to the unknown-subcommand
      branch, printing to stderr and returning 2, so ``--help > file`` wrote
      nothing; three ``midas-nf-fit-*`` commands shared one branch between
      "help requested" and "too few arguments" and returned 1 for both.

    Help is a request, not an error: stdout, exit 0. A usage error stays on
    stderr with a non-zero exit, which is asserted separately.
    """
    import re
    import shutil
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    pkgs = root / "packages"
    if not pkgs.is_dir():
        pytest.skip("not running from a source checkout")

    names = set()
    for pj in sorted(pkgs.glob("midas_*/pyproject.toml")):
        blk = re.search(r"\[project\.scripts\](.*?)(\n\[|\Z)", pj.read_text(), re.S)
        if blk:
            names |= set(re.findall(r'^\s*"?([A-Za-z0-9_.-]+)"?\s*=',
                                    blk.group(1), re.M))
    installed = sorted(n for n in names if shutil.which(n))
    if not installed:
        pytest.skip("no MIDAS console scripts on PATH")

    broken = []
    for n in installed:
        p = subprocess.run([n, "--help"], capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not p.stdout.strip():
            broken.append(f"{n}: exit={p.returncode} stdout={len(p.stdout)}B")
    assert not broken, "commands whose --help is broken:\n  " + "\n  ".join(broken)
