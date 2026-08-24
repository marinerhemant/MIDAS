#!/usr/bin/env python
"""Keep the vendored C in sync with its canonical copy in ``midas_ckernel``.

Why the duplication exists — do not "fix" it by centralising
--------------------------------------------------------------
``midas-index`` and ``midas-fit-grain`` compile their C at install time
(scikit-build-core + CMake) from **their own sdists**. An sdist contains only
that package's directory, so a build that reached into
``packages/midas_ckernel/c_src/`` would work in a git checkout and fail for
every pip user. ``midas-ckernel`` is also deliberately unpublished, so it cannot
be a build-time dependency either. Both CMakeLists say this outright:

    "All required C sources are vendored under c_src/ — no reach into
     FF_HEDM/src/ at build time."

So each package must physically carry the sources. What that costs is drift: a
fix applied to one copy leaves the three packages computing different forward
models, with every test still green, because nothing compares them.

This script is the missing half. ``midas_ckernel/c_src`` is the canonical copy
(named as such in midas_fit_grain/CMakeLists.txt); edit there, then run this.

Usage
-----
    python utils/sync_vendored_c.py            # copy canonical -> the mirrors
    python utils/sync_vendored_c.py --check    # verify only, exit 1 on drift

``--check`` is what CI and the test call; it never writes.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import shutil
import sys
from pathlib import Path

#: The canonical copy. Every other package mirrors this one.
CANONICAL_PKG = "midas_ckernel"

#: Packages that vendor a copy of the shared C.
MIRROR_PKGS = ("midas_fit_grain", "midas_index")

#: Files that must be byte-identical across all three ``c_src`` dirs.
#:
#: Derived from what is actually duplicated today, not from a wish list: adding
#: a name here without the file existing in the canonical dir is an error, so
#: the list cannot silently describe something that is not there.
SHARED = (
    "forward.c",
    "forward.h",
    "MIDAS_Math.c",
    "MIDAS_Math.h",
    "GetMisorientation.c",
    "GetMisorientation.h",
    "IndexerConsolidatedIO.h",
    "nelder_mead.c",
)


def repo_root() -> Path:
    """The repository root, from this file's location."""
    return Path(__file__).resolve().parent.parent


def c_src(pkg: str) -> Path:
    return repo_root() / "packages" / pkg / "c_src"


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def check() -> list[str]:
    """Return a list of human-readable drift descriptions (empty = in sync)."""
    problems: list[str] = []
    src_dir = c_src(CANONICAL_PKG)
    for name in SHARED:
        src = src_dir / name
        if not src.exists():
            problems.append(f"{CANONICAL_PKG}/c_src/{name} is MISSING "
                            f"(it is the canonical copy)")
            continue
        for pkg in MIRROR_PKGS:
            dst = c_src(pkg) / name
            if not dst.exists():
                problems.append(f"{pkg}/c_src/{name} is MISSING")
            elif not filecmp.cmp(src, dst, shallow=False):
                problems.append(
                    f"{pkg}/c_src/{name} DIFFERS from {CANONICAL_PKG} "
                    f"({_digest(dst)} vs canonical {_digest(src)})")
    return problems


def sync() -> list[str]:
    """Copy canonical -> mirrors. Returns the list of files actually written."""
    written: list[str] = []
    src_dir = c_src(CANONICAL_PKG)
    for name in SHARED:
        src = src_dir / name
        if not src.exists():
            raise SystemExit(f"canonical source missing: {src}")
        for pkg in MIRROR_PKGS:
            dst = c_src(pkg) / name
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            written.append(f"{pkg}/c_src/{name}")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sync_vendored_c",
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--check", action="store_true",
                    help="verify only; exit 1 if any mirror has drifted")
    args = ap.parse_args(argv)

    if args.check:
        problems = check()
        if problems:
            print("vendored-C drift detected:")
            for p in problems:
                print(f"  {p}")
            print(f"\nFix by editing packages/{CANONICAL_PKG}/c_src/ and running:")
            print("  python utils/sync_vendored_c.py")
            return 1
        print(f"vendored C in sync: {len(SHARED)} file(s) x "
              f"{len(MIRROR_PKGS)} mirror(s)")
        return 0

    written = sync()
    if written:
        print(f"synced {len(written)} file(s) from {CANONICAL_PKG}:")
        for w in written:
            print(f"  {w}")
    else:
        print("already in sync; nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
