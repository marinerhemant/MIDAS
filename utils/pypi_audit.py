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


def inspect(root: Path, pkg_dir: Path) -> dict:
    """Version metadata plus how many commits touched the source since the bump."""
    name, version = read_project(pkg_dir / "pyproject.toml")
    rel = f"packages/{pkg_dir.name}"

    # The commit that last introduced this exact version string. Pathspec-limited
    # to pyproject.toml so an unrelated file mentioning the version cannot match.
    bump = git(root, "log", "-1", "--format=%H", "-S", f'version = "{version}"',
               "--", f"{rel}/pyproject.toml")

    since_src = since_all = 0
    bump_date = ""
    if bump:
        bump_date = git(root, "log", "-1", "--format=%ad", "--date=short", bump)
        rng = f"{bump}..HEAD"
        since_all = len(git(root, "log", "--format=%H", rng, "--", rel).split())
        # Only the importable module dir counts as "shipped code" -- test and doc
        # churn does not make a release stale.
        #
        # Compiled bytecode is excluded: __pycache__/*.pyc can be committed (and
        # later untracked) without any change to what a wheel contains, and a
        # commit that only removes them would otherwise report the package as
        # stale forever. Seen for real -- untracking midas_stress's stale .pyc
        # files flagged midas-stress as needing a re-release.
        if (pkg_dir / pkg_dir.name).is_dir():
            since_src = len(git(root, "log", "--format=%H", rng, "--",
                                f"{rel}/{pkg_dir.name}",
                                f":(exclude){rel}/{pkg_dir.name}/**/__pycache__/**",
                                f":(exclude){rel}/{pkg_dir.name}/**/*.pyc").split())

    return {"name": name, "dir": pkg_dir.name, "repo": version, "bump": bump[:8],
            "bump_date": bump_date, "commits_since_bump_src": since_src,
            "commits_since_bump_all": since_all}


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
    return "stale" if row["commits_since_bump_src"] > 0 else "sync"


def render(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 92}\n{title}  ({len(rows)})\n{'=' * 92}")
    if not rows:
        print("  none")
        return
    for r in sorted(rows, key=lambda r: -r["commits_since_bump_src"]):
        print(f"  {r['name']:<30} repo={r['repo']:<9} pypi={r['pypi']:<12}"
              f" src_commits_since_bump={r['commits_since_bump_src']:<3}"
              f" (all={r['commits_since_bump_all']:<3}) bump={r['bump_date']}")


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
