"""``midas-ff-pipeline`` console script — a RETIRED shim.

This package no longer orchestrates anything. ``main()`` rewrites the argv it
is given into ``midas-pipeline`` form and hands it to
:func:`midas_pipeline.cli.main`; every stage, the ``Pipeline`` class and the
whole parallel stage tree were deleted at 0.7.0 rather than maintained in two
places.

**Why the tree went rather than being kept "just in case".** ``main()`` had
already delegated since 0.6.x, so ``_cmd_run``/``_cmd_resume`` — the only
callers of ``Pipeline`` — were unreachable from the console script. What was
left was ~2 500 lines that no CLI user could execute, thirteen stage modules
each already diverged from their ``midas_pipeline`` twin, and a standing
invitation to fix a bug in one copy and not the other. Every module had a twin
in ``midas_pipeline`` (``seeding`` as a package rather than a module), so
nothing was lost by deleting rather than porting.

The only surface that remains is argv translation, plus the
``midas_ff_pipeline.testing`` re-export shim that
``midas_pipeline/notebooks/_build.py`` still imports.

This whole package goes away at 1.0.0. Use ``midas-pipeline run --scan-mode ff``.
"""
from __future__ import annotations

import sys
import warnings
from typing import Optional

_FLAG_RENAMES: dict[str, str] = {}

_FLAG_DROPPED = {
    "--loss":   "the c-omp refiner has no configurable loss",
    "--mode":   "the c-omp refiner has no configurable strategy",
    "--solver": "the c-omp refiner uses a vendored Nelder-Mead",
}

_SCAN_MODE_SUBCOMMANDS = {"run"}


def translate_argv(argv: list[str]) -> list[str]:
    """Rewrite a midas-ff-pipeline argv into a midas-pipeline argv.

    Handles both ``--flag value`` and ``--flag=value``. Flags in
    ``_FLAG_DROPPED`` are removed along with their value, and warned about --
    midas-pipeline no longer accepts them at all.
    """
    out: list[str] = []
    skip_value_for: Optional[str] = None
    for i, tok in enumerate(argv):
        flag, sep, value = tok.partition("=")
        if skip_value_for is not None:
            # this token is the dropped flag's value
            skip_value_for = None
            continue
        if flag in _FLAG_DROPPED:
            warnings.warn(
                f"{flag} is no longer supported and has been dropped: "
                f"{_FLAG_DROPPED[flag]}. midas-pipeline >=0.15.0 runs "
                f"refinement on c-omp only.",
                DeprecationWarning, stacklevel=2,
            )
            if not sep:
                # `--flag value` form: also swallow the value that follows
                skip_value_for = flag
            continue
        if flag in _FLAG_RENAMES:
            new = _FLAG_RENAMES[flag]
            out.append(f"{new}{sep}{value}" if sep else new)
            continue
        out.append(tok)

    # inject --scan-mode ff for the subcommands that accept it
    for i, tok in enumerate(out):
        if not tok.startswith("-"):
            if tok in _SCAN_MODE_SUBCOMMANDS and "--scan-mode" not in out:
                out.insert(i + 1, "ff")
                out.insert(i + 1, "--scan-mode")
            break
    return out


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    new_argv = translate_argv(argv)

    print(
        "\n"
        "  ─────────────────────────────────────────────────────────────────\n"
        "  midas-ff-pipeline is RETIRED and no longer runs its own pipeline.\n"
        "  Delegating to:\n"
        f"      midas-pipeline {' '.join(new_argv)}\n"
        "\n"
        "  Update your scripts — this shim goes away at 1.0.0. The old path\n"
        "  returned ~4x too many grains (no MinNrSpots/Completeness gate) and\n"
        "  used spot_aware rather than c_parity; your results will change,\n"
        "  for the better.\n"
        "  ─────────────────────────────────────────────────────────────────\n",
        file=sys.stderr,
    )

    from midas_pipeline.cli import main as _midas_pipeline_main
    return _midas_pipeline_main(new_argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
