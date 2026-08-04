#!/usr/bin/env python3
"""Fail if a personal name, PI-named beamtime, or user material reaches a tracked file.

The MIDAS repo is public. Beamtime directories are named after the PI, sample
names carry the material, and Parsl configs carry allocation codes -- all of which
identify whose data a given number came from. This gate stops new ones landing.

    python utils/scrub_check.py            # scan tracked files, exit 1 on a hit
    python utils/scrub_check.py --staged   # scan only staged files (pre-commit)
    python utils/scrub_check.py --install-hook

Resolve pseudonyms with the private BEAMTIME_KEY.md (git-excluded, never committed).

Two traps this deliberately handles, both hit during the original scrub:

1. **Accented spellings.** ``borbel`` is NOT a substring of ``Borbély`` -- the
   accented character sits between ``borb`` and ``ly``. A naive substring
   deny-list silently passes the accented form. Patterns here are written to
   match both, and NAME_PATTERNS is unicode-aware.

2. **base64 in notebooks.** Embedded PNGs in ``.ipynb`` outputs contain arbitrary
   letter runs; ``CE5Y`` occurs inside one by chance. Notebooks are parsed as
   JSON and only cell *source* is scanned, never ``outputs`` or ``attachments``.
   Scanning notebooks as flat text produces false positives and, worse, tempts a
   blanket sed that corrupts the image.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Personal names that have appeared as DATA SOURCES (beamtime dirs, sample
# filenames, dataset labels). Case-insensitive.
#
# NOT listed, deliberately -- these are method attributions and citations, and
# scrubbing them would be a misattribution rather than a privacy win:
#   Kenesei   -- the Kenesei per-spot strain solver (paper Eq. 8-11), analogous
#                to Fable-Beaudoin. Also a public API name
#                (``solve_strain_kenesei_bounded``), a CLI choice
#                (``--strain-method kenesei``) and an on-disk HDF5 dataset name
#                (``strain_kenesei``). Renaming it breaks the API and file compat.
#   Shastri   -- author on the MIDAS methodology papers (Acta Cryst. A82).
#   Borbely   -- Ungar & Borbely 1996 / Borbely et al. 2003 (ANIZC), see ALLOWLIST.
# A beamtime named after any of them is still caught, by BEAMTIME_RE below.
NAME_PATTERNS = [
    r"pokharel",
    r"xzhang",
    r"indrajeet",
    r"bucsek",          # also catches the SLURM account 'abucsek0'
    r"wenxi|wenxli",
    r"emerson",
    r"borb[eé]ly",      # BOTH spellings -- see docstring trap 1
    r"preuss",
    r"stubbins",
    r"smaddali",
    r"okasinski",
]

# User sample materials. CeO2 / ceria / LaB6 are calibration standards used at
# every beamline -- they identify nobody and are deliberately NOT listed.
MATERIAL_PATTERNS = [
    r"ce5y",
    r"nf_ce[_0-9]",
    r"beta-ce\b",
    r"gamma-ce\b",
    r"β-ce\b",
    r"γ-ce\b",
    r"ce_jul26",
    r"s6061",
]

# Any name_monYY / nameMonYY beamtime that is not on the institutional allow-list.
BEAMTIME_RE = re.compile(
    r"\b([a-z0-9]+)_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[0-9]{2}\b",
    re.IGNORECASE,
)
ALLOWED_BEAMTIME_PREFIXES = {
    # pseudonyms
    "bt", "dataset",
    # non-personal: facility, sector, programme, or descriptive
    "mpe", "afrl", "hpldrd", "nfdev",
}

# (path substring, regex) pairs that are known-good and must not fail the gate.
ALLOWLIST = [
    # Ungar & Borbely 1996 / Borbely et al. 2003 (ANIZC) are LITERATURE CITATIONS.
    # Scrubbing a citation would be a misattribution, not a privacy win.
    ("midas_defect/", r"borb[eé]ly"),
    # This file names the patterns it searches for.
    ("utils/scrub_check.py", r".*"),
]

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".ge1", ".ge2", ".ge3",
    ".ge5", ".edf", ".tif", ".tiff", ".h5", ".hdf5", ".bin", ".pptx", ".docx",
    ".so", ".dylib", ".o", ".a", ".pyc",
}


def tracked_files(staged: bool) -> list[Path]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [Path(p) for p in out.splitlines() if p]


def notebook_source_lines(path: Path):
    """Yield (lineno, text) for notebook cell SOURCE only -- never outputs."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return
    for ci, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        blob = src if isinstance(src, str) else "".join(src)
        for li, line in enumerate(blob.splitlines(), start=1):
            yield f"cell{ci}:{li}", line


def text_lines(path: Path):
    try:
        for i, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            yield str(i), line
    except OSError:
        return


def allowlisted(path: Path, line: str) -> bool:
    p = str(path)
    return any(
        frag in p and re.search(rx, line, re.IGNORECASE) for frag, rx in ALLOWLIST
    )


def scan(paths: list[Path]) -> list[tuple[str, str, str, str]]:
    name_re = re.compile("|".join(NAME_PATTERNS), re.IGNORECASE)
    mat_re = re.compile("|".join(MATERIAL_PATTERNS), re.IGNORECASE)
    hits: list[tuple[str, str, str, str]] = []

    for path in paths:
        # Filenames leak too: a CHECKPOINT_<pi>_<beamtime>.md or a
        # <PROJECT>_<PI>_NOTES.md discloses just as much as its contents, and a
        # content-only scan never sees it.
        posix = path.as_posix()
        for kind, rx in (("name-in-path", name_re), ("material-in-path", mat_re)):
            m = rx.search(posix)
            if m and not allowlisted(path, posix):
                hits.append((posix, "<filename>", kind, m.group(0)))

        if path.suffix.lower() in BINARY_SUFFIXES or not path.exists():
            continue
        reader = notebook_source_lines if path.suffix == ".ipynb" else text_lines
        for loc, line in reader(path):
            if allowlisted(path, line):
                continue
            for kind, rx in (("name", name_re), ("material", mat_re)):
                m = rx.search(line)
                if m:
                    hits.append((str(path), loc, kind, m.group(0)))
            for m in BEAMTIME_RE.finditer(line):
                prefix = m.group(1).lower()
                if not any(prefix.startswith(a) for a in ALLOWED_BEAMTIME_PREFIXES):
                    hits.append((str(path), loc, "beamtime", m.group(0)))
    return hits


def install_hook() -> int:
    hook = Path(".git/hooks/pre-commit")
    body = (
        "#!/bin/sh\n"
        "# Added by utils/scrub_check.py --install-hook\n"
        'exec python3 "$(git rev-parse --show-toplevel)/utils/scrub_check.py" --staged\n'
    )
    if hook.exists() and "scrub_check.py" not in hook.read_text():
        print(f"refusing to overwrite existing hook: {hook}", file=sys.stderr)
        print("add this line to it yourself:\n  " + body.splitlines()[-1])
        return 1
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(body)
    hook.chmod(0o755)
    print(f"installed {hook}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", action="store_true", help="scan staged files only")
    ap.add_argument("--install-hook", action="store_true")
    args = ap.parse_args()

    if args.install_hook:
        return install_hook()

    hits = scan(tracked_files(args.staged))
    if not hits:
        print("scrub-check: clean")
        return 0

    print("scrub-check: FAILED -- identifying strings in tracked files\n", file=sys.stderr)
    for path, loc, kind, tok in hits:
        print(f"  {path}:{loc}  [{kind}]  {tok}", file=sys.stderr)
    print(
        f"\n{len(hits)} hit(s). Replace with a pseudonym and record the mapping in "
        "BEAMTIME_KEY.md (git-excluded). If this is a literature citation rather "
        "than a data source, add it to ALLOWLIST in utils/scrub_check.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
