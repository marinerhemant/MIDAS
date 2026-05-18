#!/usr/bin/env python3
"""Output JSON list of packages whose tests should run, given the current
diff range, and the package dep graph in ``packages/``.

Usage
-----
    BASE_REF=origin/master python3 changed_packages.py
    python3 changed_packages.py --all      # every package, ignore diff

Affected = directly-changed packages ∪ every package that transitively
depends on a directly-changed package.

If any "global" file changed (this script, the workflow yaml), every
package is affected — there is no way to predict which test the change
might break.

Hidden packages (``midas_grain_odf``, ``midas_pf_odf``) are dropped: the
existing CI does not test or publish them.

Used by ``.github/workflows/python-packages.yml`` to populate the
``test`` job's matrix dynamically.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = ROOT / "packages"

# Packages that are hidden from CI / PyPI per the user's directive.
HIDDEN: set[str] = {
    "midas_grain_odf",
    "midas_joint_ff_calibrate",
    "midas_pf_odf",
    "midas_pink",
    "midas_propagate",
    "midas_uq",
}

# Files whose mutation invalidates the entire matrix (since they affect
# how every package's CI runs).
GLOBAL_TRIGGERS: tuple[str, ...] = (
    ".github/workflows/python-packages.yml",
    ".github/scripts/changed_packages.py",
)


def discover_packages() -> dict[str, Path]:
    """Return ``{pkg_dir_name: pyproject_path}``."""
    out: dict[str, Path] = {}
    if not PKG_DIR.is_dir():
        return out
    for d in sorted(PKG_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("midas_"):
            continue
        if d.name in HIDDEN:
            continue
        pp = d / "pyproject.toml"
        if pp.exists():
            out[d.name] = pp
    return out


def sibling_deps(pyproject_path: Path) -> list[str]:
    """Extract MIDAS sibling package directory names from a pyproject.toml.

    Looks at the ``dependencies = [...]`` block only — optional-dependencies
    do not gate test runs.
    """
    deps: list[str] = []
    in_block = False
    for line in pyproject_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("dependencies") and s.endswith("["):
            in_block = True
            continue
        if in_block and s.startswith("]"):
            break
        if in_block:
            m = re.match(r'"(midas-[a-z0-9-]+)', s)
            if m:
                deps.append(m.group(1).replace("-", "_"))
    return deps


def reverse_dep_graph(packages: dict[str, Path]) -> dict[str, set[str]]:
    """Return ``rev[b] = {a1, a2, ...}`` where ``a1, a2, ...`` depend on b."""
    rev: dict[str, set[str]] = {p: set() for p in packages}
    for src, pp in packages.items():
        for dep in sibling_deps(pp):
            if dep in packages:
                rev[dep].add(src)
    return rev


def transitive_closure(start: set[str], graph: dict[str, set[str]]) -> set[str]:
    """BFS reachable set from ``start`` over ``graph``."""
    seen = set(start)
    stack = list(start)
    while stack:
        n = stack.pop()
        for m in graph.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def changed_files(base_ref: str) -> list[str]:
    """``git diff --name-only base_ref...HEAD``. Empty list on failure."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, check=True,
        )
        return [p for p in result.stdout.splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def package_for_file(path: str, packages: dict[str, Path]) -> str | None:
    """Map a file path to the owning package directory (or ``None``)."""
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "packages" and parts[1] in packages:
        return parts[1]
    return None


def main(argv: list[str]) -> int:
    packages = discover_packages()
    all_pkgs = sorted(packages.keys())

    if "--all" in argv or os.environ.get("CI_TEST_ALL") == "1":
        print(json.dumps(all_pkgs))
        return 0

    base = os.environ.get("BASE_REF") or "HEAD~1"
    files = changed_files(base)

    if not files:
        # No diff base or empty diff (e.g. force-push, fresh clone, scheduled
        # run). Be conservative: test everything.
        print(json.dumps(all_pkgs), file=sys.stdout)
        print(f"[changed_packages] no diff vs {base} — testing all packages",
              file=sys.stderr)
        return 0

    if any(f in GLOBAL_TRIGGERS for f in files):
        print(json.dumps(all_pkgs))
        print(f"[changed_packages] global trigger touched — testing all "
              f"{len(all_pkgs)} packages", file=sys.stderr)
        return 0

    direct = {pkg for f in files
              if (pkg := package_for_file(f, packages)) is not None}
    rev = reverse_dep_graph(packages)
    affected = transitive_closure(direct, rev)

    print(json.dumps(sorted(affected)))
    print(f"[changed_packages] {len(direct)} directly changed → "
          f"{len(affected)} affected (transitive closure)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
