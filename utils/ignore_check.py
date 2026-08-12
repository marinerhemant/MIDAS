#!/usr/bin/env python
"""Fail if any tracked file matches a .gitignore rule.

A `.gitignore` rule has no effect on a path git is already tracking. So a rule
added after the file was committed is inert: the intent is recorded, the file
keeps being published, and nothing ever says so. Six files were found in that
state in this repository on 2026-08-12 -- `implementation_plan.md`, `fwd_sim/`
(17 files), three more planning documents and a parsl run log -- against rules
that had been in `.gitignore` for months. `git rm --cached` is the only thing
that fixes it, and nothing was prompting anyone to run it.

    python utils/ignore_check.py                 # check
    python utils/ignore_check.py --install-hook  # add to .git/hooks/pre-commit

Checks the whole tracked set rather than only staged paths: the question is a
property of the repository, not of one commit, and a file that slipped in three
commits ago is exactly the case this exists to surface. Use `--no-verify` for the
rare deliberate `git add -f`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HOOK_LINE = 'python3 "$(git rev-parse --show-toplevel)/utils/ignore_check.py" || exit 1\n'


def tracked_but_ignored() -> list[str]:
    """Tracked paths that .gitignore says should not be tracked."""
    out = subprocess.run(
        ["git", "ls-files", "-i", "-c", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def matching_rule(path: str) -> str:
    """Which rule catches this path, so the report names it rather than implying it."""
    r = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", path],
        capture_output=True, text=True,
    )
    return r.stdout.strip().split("\t")[0] if r.stdout.strip() else "?"


def install_hook() -> int:
    hook = Path(".git/hooks/pre-commit")
    if hook.exists():
        body = hook.read_text()
        if "ignore_check.py" in body:
            print(f"already installed in {hook}")
            return 0
        # Append rather than refuse: this repo's hook already carries scrub_check
        # and doc_citation_check, and overwriting it would silently drop both.
        if not body.endswith("\n"):
            body += "\n"
        hook.write_text(body + "# Added by utils/ignore_check.py --install-hook\n" + HOOK_LINE)
    else:
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\n# Added by utils/ignore_check.py --install-hook\n" + HOOK_LINE)
    hook.chmod(0o755)
    print(f"installed {hook}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install-hook", action="store_true")
    args = ap.parse_args()

    if args.install_hook:
        return install_hook()

    hits = tracked_but_ignored()
    if not hits:
        print("ignore-check: clean")
        return 0

    print(f"ignore-check: FAILED -- {len(hits)} tracked file(s) match a .gitignore rule\n",
          file=sys.stderr)
    for p in hits:
        print(f"  {p}\n      caught by {matching_rule(p)}", file=sys.stderr)
    print("\nThe rule cannot take effect while the file is tracked. To honour it:\n"
          f"  git rm --cached {hits[0]}\n"
          "(the file stays on disk; the content stays in history)\n"
          "If the file SHOULD be tracked, remove or narrow the rule instead.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
