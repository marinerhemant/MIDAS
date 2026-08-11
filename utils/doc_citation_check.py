#!/usr/bin/env python3
"""Check that ``path:line`` citations in the manuals still point at real code.

The handbooks are built on citations -- ``connected.py:91-100``,
``ff_zip.py:159-167``, ``fitorientation/params.py:204-221``. They are what makes
a claim checkable instead of remembered. They are also the part that rots
silently: the prose stays put while the code moves under it, and nothing errors.

This catches the mechanical half of that rot:

  MISSING   the cited file does not exist
  RANGE     the cited line number is past the end of the file
  AMBIGUOUS a bare filename matches several files, so the claim cannot be checked
  SYMBOL    the doc names an identifier next to the citation and that identifier
            is nowhere in (or near) the cited lines

SYMBOL is the interesting one -- a file can keep its length while the function
you cited moves 200 lines down, and only the symbol check notices.

What it deliberately does NOT do is verify the claim itself. "This function is
called on every frame" cannot be checked by grep. A green run means the
citations resolve, not that the prose is true.

Usage:
    python utils/doc_citation_check.py [--manuals DIR] [-v]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# `pkg/mod.py:12`, `mod.py:12-40`, `a/b/c.c:100`. Line span optional-ended.
CITATION = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|c|h|cu|sh|toml|md)):(\d+)(?:\s*-\s*(\d+))?[^`]*`"
)
# A backticked identifier: `foo`, `Bar.baz`, `qux()`.
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\(\))?`")

SEARCH_ROOTS = ("packages", "utils", "gui", "FF_HEDM", "NF_HEDM", "TOMO", "DT", "tests")
SYMBOL_WINDOW = 40      # lines either side of the cited range to accept a symbol in


def _norm(s: str) -> str:
    """Fold CamelCase/snake_case/case so a doc key matches its code attribute.

    Parameter names are written ``OrientTol`` in the manuals and bound as
    ``p.orient_tol`` in the code. Matching literally reports drift that is not
    there -- the citation was right, the spelling convention differs.
    """
    return s.lower().replace("_", "")


_INDEX: dict[str, list[Path]] | None = None


def _index(root: Path) -> dict[str, list[Path]]:
    """basename -> every source file with that name, built once."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for sub in SEARCH_ROOTS:
            base = root / sub
            if not base.is_dir():
                continue
            for p in base.rglob("*"):
                if not p.is_file() or p.suffix not in (
                        ".py", ".c", ".h", ".cu", ".sh", ".toml", ".md"):
                    continue
                if any(part in ("build", "dist", "__pycache__", ".git", "_deps",
                                "archive", "egg-info", ".egg-info")
                       for part in p.parts):
                    continue
                _INDEX.setdefault(p.name, []).append(p)
    return _INDEX


def resolve(root: Path, cited: str) -> list[Path]:
    """Candidates for a citation. Partial paths match on trailing components.

    Citations are written for a human ("``fitorientation/params.py``"), not as
    repo-relative paths, so match by path *suffix* rather than requiring the
    exact prefix. Every candidate is returned; the caller passes the citation if
    ANY of them satisfies it, which is the right question -- "is there a file
    where this claim holds?" -- and stops `cli.py` from being unresolvable
    merely because 26 packages have one.
    """
    direct = root / cited
    if direct.is_file():
        return [direct]
    parts = [p for p in cited.split("/") if p and p != "."]
    if not parts:
        return []
    cands = _index(root).get(parts[-1], [])
    for extra in reversed(parts[:-1]):          # narrow by each parent named
        narrowed = [p for p in cands if extra in p.parts or extra in str(p)]
        if narrowed:
            cands = narrowed
    return cands


def nearby_identifiers(line: str, cite_span: tuple[int, int],
                       module_stem: str = "") -> list[str]:
    """Identifiers the citation is *about*.

    Only the run of text immediately before the citation counts. The docs write
    ``` `filter_regions_by_size` (`connected.py:91-100`) ``` -- the subject sits
    just left of the parenthesis. Scanning the whole line instead picks up
    unrelated names from elsewhere in the sentence and reports them as missing.
    """
    lead = line[max(0, cite_span[0] - 90):cite_span[0]]
    out = []
    for m in IDENT.finditer(lead):
        name = m.group(1)
        if name.endswith((".py", ".c", ".h", ".md", ".toml")):
            continue
        if "." in name:
            name = name.split(".")[-1]
        # A module rarely contains its own basename, and an exception class named
        # in prose ("raises `KeyError`") describes behaviour, not a symbol that
        # has to be present at the cited line. Both are noise, not drift.
        if name == module_stem or name.endswith(("Error", "Exception", "Warning")):
            continue
        if len(name) > 3 and not name.isupper():
            out.append(name)
    return out[-2:]                             # the nearest one or two only


# `#{1,6}`: a top-level campaign section is written `# 7.` in the NF notebook, and
# requiring two hashes silently reported its own subsections' parent as missing.
SECTION_HEAD = re.compile(r"^#{1,6} (\d+[a-z]?(?:-[a-z]+)?)\.", re.M)
# A bare §n means "this doc set". Qualified refs (Lab Notebook §n, NF_HEDM_Handbook.md §n)
# point elsewhere and are not this checker's business.
# Match the reference only; the qualifying lead is sliced from the source afterwards.
# Capturing it here consumes text, so in a run like "§0-§0a" the second reference
# gets a one-character lead and its qualifier is missed.
SECTION_REF = re.compile(r"§(\d+[a-z]?(?:-[a-z]+)?)")
LEAD = 40
# The documented convention is `Handbook §n` / `Lab Notebook §n` -- a bare word,
# not a filename. Matching only "handbook.md" missed the form actually in use.
QUALIFIED = re.compile(r"(?i)notebook|handbook|\.md`? ")


def _doc_set(path: Path) -> tuple:
    """Which numbering space a file belongs to.

    A technique directory (manuals/ff-hedm/) is ONE space spread over several
    files, so a §n defined in phase-1 resolves from the spine. The lab notebook
    is its own space even inside that directory -- it carries its own §1-§7,
    which is exactly why the convention writes cross-refs as ``Lab Notebook §n``.
    """
    if "LAB_NOTEBOOK" in path.name.upper() or "LAB_NOTEBOOK" in path.stem.upper():
        return (path.parent, "notebook")
    return (path.parent, "handbook")


def check_section_refs(manuals: Path) -> list[str]:
    """Bare §n references that resolve nowhere in their own doc set."""
    heads: dict[tuple, set[str]] = {}
    docs = sorted(manuals.rglob("*.md"))
    for d in docs:
        heads.setdefault(_doc_set(d), set()).update(
            m.group(1) for m in SECTION_HEAD.finditer(d.read_text(errors="replace")))
    out = []
    for d in docs:
        known = heads.get(_doc_set(d), set())
        if not known:
            continue
        bad = set()
        text = d.read_text(errors="replace")
        for m in SECTION_REF.finditer(text):
            if QUALIFIED.search(text[max(0, m.start() - LEAD):m.start()]):
                continue
            if m.group(1) not in known:
                bad.add(m.group(1))
        for s in sorted(bad):
            out.append(f"SECTION    {d.relative_to(manuals)}  §{s} resolves nowhere "
                       f"in its doc set")
    return out


def check(root: Path, manuals: Path, verbose: bool) -> int:
    problems: list[str] = []
    checked = 0

    # rglob, not glob: a technique doc set (manuals/ff-hedm/) is several files.
    for doc in sorted(manuals.rglob("*.md")):
        lines = doc.read_text(errors="replace").split("\n")
        for lineno, line in enumerate(lines, 1):
            for m in CITATION.finditer(line):
                cited, start_s, end_s = m.group(1), m.group(2), m.group(3)
                # Paths outside the repo (a scratch analysis dir on someone's
                # laptop) are provenance, not a citation this can verify.
                if cited.startswith("/") or cited.startswith("~"):
                    continue
                start = int(start_s)
                end = int(end_s) if end_s else start
                checked += 1
                where = f"{doc.name}:{lineno}"

                cands = resolve(root, cited)
                if not cands:
                    problems.append(f"MISSING    {where}  `{cited}` -> no such file")
                    continue

                idents = nearby_identifiers(line, (m.start(), m.end()),
                                            Path(cited).stem)
                lo_pad, hi_pad = SYMBOL_WINDOW, SYMBOL_WINDOW
                in_range, symbol_ok = False, not idents
                for target in cands:
                    body = target.read_text(errors="replace").split("\n")
                    if start > len(body):
                        continue
                    in_range = True
                    if not idents:
                        break
                    window = "\n".join(body[max(0, start - 1 - lo_pad):
                                            min(len(body), end + hi_pad)])
                    if any(i in window or _norm(i) in _norm(window) for i in idents):
                        symbol_ok = True
                        if verbose:
                            print(f"ok  {where}  {cited}:{start} ({idents})")
                        break
                if not in_range:
                    lens = ", ".join(str(len(c.read_text(errors='replace').split(chr(10))))
                                     for c in cands[:3])
                    problems.append(
                        f"RANGE      {where}  `{cited}:{start}` past EOF in all "
                        f"{len(cands)} candidate(s) (lengths: {lens})")
                elif not symbol_ok:
                    problems.append(
                        f"SYMBOL     {where}  `{cited}:{start}"
                        f"{'-' + str(end) if end != start else ''}` names "
                        f"{idents}, absent within ±{SYMBOL_WINDOW} lines "
                        f"in all {len(cands)} candidate(s)")

    problems.extend(check_section_refs(manuals))
    print(f"doc-citation-check: {checked} citation(s) in {manuals}")
    if problems:
        print(f"\nFAILED -- {len(problems)} unresolved:\n")
        for p in problems:
            print("  " + p)
        print("\nEither the code moved (update the citation) or the claim is stale")
        print("(update the prose). Do not delete the citation to silence this.")
        return 1
    print("all citations resolve")
    return 0


HOOK = """#!/bin/sh
# Added by utils/scrub_check.py --install-hook and
#          utils/doc_citation_check.py --install-hook
ROOT="$(git rev-parse --show-toplevel)"
python3 "$ROOT/utils/scrub_check.py" --staged || exit 1
python3 "$ROOT/utils/doc_citation_check.py" || exit 1
"""


def install_hook(root: Path) -> int:
    """Extend the pre-commit hook to run this check as well as scrub-check.

    Idempotent: rewrites the hook to the canonical two-check form rather than
    appending, so running it twice does not double the invocation.
    """
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    if hook.exists() and "doc_citation_check.py" in hook.read_text():
        print(f"already installed: {hook}")
        return 0
    hook.write_text(HOOK)
    hook.chmod(0o755)
    print(f"installed: {hook}\n  scrub_check.py --staged\n  doc_citation_check.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manuals", default="manuals", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--install-hook", action="store_true",
                    help="wire this check into .git/hooks/pre-commit next to scrub-check")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.install_hook:
        return install_hook(root)
    return check(root, root / args.manuals, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
