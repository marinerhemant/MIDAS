#!/usr/bin/env bash
# Staging guard for MIDAS release commits.  Source it, then call `commit`.
#
#   source manuals/release/commit_helper.sh
#   commit "subject" "body" path/one path/two
#
# Why this exists: `git add <dir>` swept unrelated edits into the wrong commit
# three times in one session, and switching to a `__init__.py$` glob did it a
# fourth -- that one produced a commit that raised ImportError on its own,
# because the glob grouped by path shape rather than by dependency.
#
# The guard asserts the staged set EQUALS the intended set and aborts otherwise.
# It cannot check that a commit stands alone; do that separately (see
# `standalone_check` at the bottom).
set -uo pipefail

_RELEASE_FAIL=0

commit() {
    local subject="$1"; shift
    local body="$1"; shift
    local files=("$@")

    if [ "$_RELEASE_FAIL" -ne 0 ]; then
        echo "SKIP (earlier failure): $subject"; return 1
    fi

    git add -- "${files[@]}" || {
        echo "ADD FAILED: $subject"; _RELEASE_FAIL=1; return 1; }

    local staged want
    staged=$(git diff --cached --name-only | sort)
    want=$(printf '%s\n' "${files[@]}" | sort)
    if [ "$staged" != "$want" ]; then
        echo "GUARD FAILED (staged != intended): $subject"
        diff <(echo "$want") <(echo "$staged") | sed 's/^/    /'
        git reset -q; _RELEASE_FAIL=1; return 1
    fi

    # The hooks (scrub-check, doc-citation-check, ignore-check) run on commit and
    # are load-bearing -- do not bypass them with --no-verify.
    git commit -q -m "$subject" -m "$body" || {
        echo "COMMIT FAILED: $subject"; _RELEASE_FAIL=1; return 1; }

    printf "  %-9s %s\n" "$(git rev-parse --short HEAD)" "$subject"
}

# Assert a symbol exported by an __init__ is DEFINED at the same commit.
#
#   standalone_check <sha> <init-path> <module-path> <SymbolName>
#
# Catches the class of break where a re-export lands in an earlier commit than
# its definition: both files are "correctly staged", and the earlier commit
# raises ImportError on import.
standalone_check() {
    local sha="$1" init="$2" module="$3" sym="$4"
    local n_ref n_def
    n_ref=$(git show "$sha:$init"   2>/dev/null | grep -c "$sym" || true)
    n_def=$(git show "$sha:$module" 2>/dev/null | grep -c "$sym" || true)
    if [ "${n_ref:-0}" -gt 0 ] && [ "${n_def:-0}" -eq 0 ]; then
        echo "  *** $sha: $init references $sym but $module does not define it"
        return 1
    fi
    printf "  %s  %s standalone OK\n" "$sha" "$sym"
}
