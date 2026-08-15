#!/usr/bin/env python3
"""Audit packages/ against what is actually published on PyPI.

Two independent ways a package can be out of date, only one of which is visible
by reading version numbers:

  A. VERSION MISMATCH -- repo version != PyPI latest. Loud and obvious: either a
     bump was never released, or PyPI is ahead of the tree.

  B. STALE RELEASE -- repo version == PyPI latest, but source commits landed
     AFTER the commit that last set that version. The version number is then a
     lie: it claims to describe a released artifact whose contents it no longer
     matches. Nothing warns you, and `pip install` silently hands users old code.

Class B is the dangerous one. It is what broke midas-diffract: HEDMGeometry
gained omega_ranges/box_sizes on 2026-07-30, five weeks after the 0.6.0 bump, so
the published 0.6.0 lacked fields that in-repo callers passed unconditionally.
Anyone resolving midas-diffract from PyPI got a TypeError at construction, while
`version` looked perfectly in sync.

Usage:
    python utils/pypi_audit.py                      # report, exit 0
    python utils/pypi_audit.py --fail-on-stale      # exit 1 if any class-B hit
    python utils/pypi_audit.py --ignore midas-ckernel
    python utils/pypi_audit.py --json               # machine-readable

Exit codes: 0 clean (or report-only), 1 stale releases found with
--fail-on-stale, 2 a PyPI query failed.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:                                        # 3.11+
    import tomllib
except ModuleNotFoundError:                 # pragma: no cover - 3.9/3.10
    try:
        import tomli as tomllib             # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None                      # type: ignore[assignment]

PYPI_JSON = "https://pypi.org/pypi/{name}/json"
_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"', re.M)
_VER_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def repo_root() -> Path:
    """Repo root, derived from this file rather than assumed."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def git(root: Path, *args: str) -> str:
    """Run git and return stdout. Never raises: a missing history is not fatal."""
    return subprocess.run(["git", *args], cwd=root,
                          capture_output=True, text=True).stdout.strip()


def read_project(pyproject: Path) -> tuple[str, str]:
    """(name, version) from a pyproject.toml, with a regex fallback for <3.11."""
    text = pyproject.read_text()
    if tomllib is not None:
        meta = tomllib.loads(text)["project"]
        return meta["name"], meta["version"]
    name, ver = _NAME_RE.search(text), _VER_RE.search(text)
    if not (name and ver):
        raise ValueError(f"cannot parse name/version from {pyproject}")
    return name.group(1), ver.group(1)


def pypi_latest(name: str) -> str:
    """Latest published version, 'UNPUBLISHED' on 404, 'ERR:...' otherwise."""
    try:
        with urllib.request.urlopen(PYPI_JSON.format(name=name), timeout=30) as fh:
            return json.load(fh)["info"]["version"]
    except urllib.error.HTTPError as exc:
        return "UNPUBLISHED" if exc.code == 404 else f"ERR:HTTP{exc.code}"
    except Exception as exc:                        # noqa: BLE001 - report, don't crash
        return f"ERR:{type(exc).__name__}"


def version_key(v: str) -> tuple:
    """Loose version ordering; good enough to say ahead/behind/equal."""
    parts = []
    for chunk in v.replace("-", ".").split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _strip_docstrings(tree):
    """Drop docstring expressions in place, so comment/doc edits compare equal."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _behaviour_signature(src: str) -> str:
    """AST dump with docstrings removed. Raises SyntaxError on unparseable input."""
    return ast.dump(_strip_docstrings(ast.parse(src)))


def _identifier_re():
    """Scrub patterns, borrowed from utils/scrub_check.py so there is one list.

    Falls back to None if scrub_check cannot be imported, in which case
    redacts_identifiers() conservatively reports False and the caller keeps its
    old behaviour-only judgement.
    """
    global _IDENT_RE
    try:
        return _IDENT_RE
    except NameError:
        pass
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import scrub_check                                    # noqa: PLC0415
        parts = list(scrub_check.NAME_PATTERNS) + list(scrub_check.MATERIAL_PATTERNS)
        _IDENT_RE = re.compile("|".join(f"(?:{p})" for p in parts), re.I)
    except Exception:                                         # noqa: BLE001
        _IDENT_RE = None
    return _IDENT_RE


def redacts_identifiers(root: Path, sha: str, prefix: str) -> bool:
    """Did ``sha`` REMOVE an identifying string under ``prefix``?

    This is the hole changes_behaviour() leaves open, and it is not theoretical.
    The 2026-08-03 anonymisation scrub was comment-only, so every package whose
    sole unreleased change was the scrub read as in-sync -- while PyPI kept
    serving the pre-scrub code. Five packages sat that way for nine days,
    including a SLURM account name and a user's home path, until the published
    artifacts were downloaded and grepped by hand.

    A comment is not executable, but it IS published. For privacy the question
    is not "would a wheel behave differently" but "does the wheel still contain
    the string", so this counts a redaction as release-worthy on its own.
    """
    rx = _identifier_re()
    if rx is None:
        return False
    diff = git(root, "show", "--format=", "--unified=0", sha, "--", prefix)
    for line in diff.splitlines():
        # removed lines only: a line that GAINED an identifier is a new leak,
        # caught by scrub_check at commit time, not a release trigger here.
        if line.startswith("-") and not line.startswith("---") and rx.search(line):
            return True
    return False


def changes_behaviour(root: Path, sha: str, prefix: str) -> bool:
    """Did ``sha`` change anything under ``prefix`` that a wheel would execute?

    Comments and docstrings are not executable content: the 2026-08-03 scrub
    renamed dataset identifiers across ~30 packages' docstrings, and every one
    of them then reported as a stale release needing a re-cut, for a diff that
    could not change a single result. Compare the docstring-stripped AST rather
    than the text.

    Conservative in every ambiguous case -- a merge, a root commit, a
    non-Python file, an added or deleted module, or anything that fails to
    parse counts as a behaviour change.
    """
    parents = git(root, "rev-list", "--parents", "-n", "1", sha).split()
    if len(parents) != 2:                       # merge, or no parent
        return True
    for entry in git(root, "show", "--name-status", "--format=", sha, "--",
                     prefix).splitlines():
        if not entry.strip():
            continue
        status, _, path = entry.partition("\t")
        path = path.strip()
        if not path.endswith(".py") or status.strip() != "M":
            return True                          # data file, add, delete, rename
        try:
            before = git(root, "show", f"{sha}^:{path}")
            after = git(root, "show", f"{sha}:{path}")
            if _behaviour_signature(before) != _behaviour_signature(after):
                return True
        except (SyntaxError, ValueError, RecursionError):
            return True
    return False


def inspect(root: Path, pkg_dir: Path) -> dict:
    """Version metadata plus how many commits touched the source since the bump."""
    name, version = read_project(pkg_dir / "pyproject.toml")
    rel = f"packages/{pkg_dir.name}"

    # Where the published artifact was actually built from.
    #
    # First choice is the RELEASE TAG. The workflow checks out the tag, so the
    # tag's commit -- not the version-bump commit -- is what PyPI holds. These
    # differ whenever a release is fixed by retargeting the tag onto a later
    # commit, which is the only recovery path when a release run fails (a rerun
    # re-checks-out the same tag). Anchoring at the bump commit then counts the
    # fix itself as "source moved since release" and the package reads stale
    # forever. Seen twice: midas-dfxm 0.4.0 (tag moved onto the NaN-guard fix)
    # and midas-calibrate-v2 0.5.5 (tag moved onto a revert, so the add+revert
    # pair had zero net diff yet counted as two commits).
    #
    # Requires local tags to be current -- run `git fetch --tags --force` first,
    # or a missing tag silently falls back to the bump commit. --force is not
    # optional: a plain `git fetch --tags` REFUSES to move a tag that already
    # exists locally ("would clobber existing tag") and exits non-zero, which is
    # exactly the retargeted-tag case this anchoring exists to handle.
    tag = f"{name}-v{version}"
    anchor = git(root, "rev-list", "-1", tag)
    anchored_at = "tag" if anchor else "bump"

    # Fallback: the commit that last introduced this exact version string.
    # Pathspec-limited to pyproject.toml so an unrelated file mentioning the
    # version cannot match.
    bump = git(root, "log", "-1", "--format=%H", "-S", f'version = "{version}"',
               "--", f"{rel}/pyproject.toml")
    if not anchor:
        anchor = bump

    since_src = since_all = cosmetic = privacy = 0
    bump_date = ""
    if anchor:
        bump_date = git(root, "log", "-1", "--format=%ad", "--date=short", anchor)
        rng = f"{anchor}..HEAD"
        since_all = len(git(root, "log", "--format=%H", rng, "--", rel).split())
        # Only the importable module dir counts as "shipped code" -- test and doc
        # churn does not make a release stale.
        #
        # Compiled bytecode is excluded: __pycache__/*.pyc can be committed (and
        # later untracked) without any change to what a wheel contains, and a
        # commit that only removes them would otherwise report the package as
        # stale forever. Seen for real -- untracking midas_stress's stale .pyc
        # files flagged midas-stress as needing a re-release.
        #
        # Comments and docstrings are excluded for the same reason, one level
        # up: a commit can touch every .py file in the package and still change
        # nothing a wheel executes. See changes_behaviour().
        #
        # EXCEPT when the comment change is a redaction. A scrub is comment-only
        # and still must reach PyPI, so redacts_identifiers() overrides the
        # cosmetic verdict -- see its docstring for what this cost last time.
        if (pkg_dir / pkg_dir.name).is_dir():
            prefix = f"{rel}/{pkg_dir.name}"
            # Short-circuit on the net diff. Counting commits answers "did anyone
            # touch this?"; the question that decides a re-release is "does the
            # tree differ from what shipped?". They part company on an add+revert
            # pair, which is two commits and zero difference.
            net_changed = bool(git(root, "diff", "--name-only", anchor, "HEAD",
                                   "--", prefix,
                                   f":(exclude){prefix}/**/__pycache__/**",
                                   f":(exclude){prefix}/**/*.pyc"))
            touched = [] if not net_changed else git(root, "log", "--format=%H", rng, "--", prefix,
                          f":(exclude){prefix}/**/__pycache__/**",
                          f":(exclude){prefix}/**/*.pyc").split()
            for s in touched:
                if changes_behaviour(root, s, prefix):
                    since_src += 1
                elif redacts_identifiers(root, s, prefix):
                    privacy += 1
                else:
                    cosmetic += 1

    return {"name": name, "dir": pkg_dir.name, "repo": version, "bump": bump[:8],
            "anchor": anchor[:8], "anchored_at": anchored_at,
            "bump_date": bump_date, "commits_since_bump_src": since_src,
            "commits_since_bump_all": since_all,
            "commits_cosmetic": cosmetic,
            "commits_privacy": privacy}


def classify(row: dict, ignore: set[str]) -> str:
    pypi = row["pypi"]
    if row["name"] in ignore:
        return "ignored"
    if pypi.startswith("ERR:"):
        return "error"
    if pypi == "UNPUBLISHED":
        return "mismatch"
    if version_key(row["repo"]) != version_key(pypi):
        return "mismatch"
    if row["commits_since_bump_src"] > 0 or row.get("commits_privacy", 0) > 0:
        return "stale"
    return "sync"


def render(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 92}\n{title}  ({len(rows)})\n{'=' * 92}")
    if not rows:
        print("  none")
        return
    for r in sorted(rows, key=lambda r: -r["commits_since_bump_src"]):
        # cosmetic = touched the module dir but changed no executable content
        # (comments/docstrings only). Shown so a suppressed commit stays visible.
        cos = r.get("commits_cosmetic", 0)
        cos_s = f" +{cos} cosmetic" if cos else ""
        # privacy = comment-only, but REDACTS an identifier. Not executable, but
        # still published, so it is its own reason to re-release. Flagged loudly
        # because this class was invisible until it had shipped five packages.
        priv = r.get("commits_privacy", 0)
        priv_s = f"  *** +{priv} REDACTION" if priv else ""
        # "bump=" is the anchor date: the release tag's commit when the tag is
        # present locally, else the version-bump commit. A "(no tag)" marks the
        # fallback, which is the weaker measurement -- see inspect().
        anch = "" if r.get("anchored_at") == "tag" else " (no tag)"
        print(f"  {r['name']:<30} repo={r['repo']:<9} pypi={r['pypi']:<12}"
              f" src_commits_since_bump={r['commits_since_bump_src']:<3}"
              f" (all={r['commits_since_bump_all']:<3}){cos_s} bump={r['bump_date']}{anch}"
              f"{priv_s}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit packages/ against PyPI (version drift + stale releases).")
    ap.add_argument("--ignore", action="append", default=[], metavar="PKG",
                    help="distribution name to skip, e.g. an intentionally "
                         "unpublished package (repeatable)")
    ap.add_argument("--fail-on-stale", action="store_true",
                    help="exit 1 if any package's source moved since its bump")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the full table as JSON instead of text")
    ap.add_argument("--jobs", type=int, default=12,
                    help="concurrent PyPI queries (default: 12)")
    args = ap.parse_args()

    root = repo_root()
    pkg_dirs = sorted(p for p in (root / "packages").glob("midas_*")
                      if (p / "pyproject.toml").is_file())
    if not pkg_dirs:
        print(f"no packages found under {root / 'packages'}", file=sys.stderr)
        return 2

    rows = [inspect(root, p) for p in pkg_dirs]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row, latest in zip(rows, pool.map(lambda r: pypi_latest(r["name"]), rows)):
            row["pypi"] = latest

    ignore = set(args.ignore)
    for row in rows:
        row["status"] = classify(row, ignore)

    if args.as_json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        buckets = {k: [r for r in rows if r["status"] == k]
                   for k in ("mismatch", "stale", "sync", "ignored", "error")}
        render(buckets["mismatch"], "A. VERSION MISMATCH (repo vs PyPI)")
        render(buckets["stale"],
               "B. STALE RELEASE -- version matches PyPI, but SOURCE moved since the bump")
        render(buckets["sync"], "C. IN SYNC")
        if buckets["ignored"]:
            render(buckets["ignored"], "IGNORED (--ignore)")
        if buckets["error"]:
            render(buckets["error"], "QUERY ERRORS")
        print(f"\n{len(rows)} packages scanned")

    if any(r["status"] == "error" for r in rows):
        return 2
    if args.fail_on_stale and any(r["status"] == "stale" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
