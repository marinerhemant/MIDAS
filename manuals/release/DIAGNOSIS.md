# Release diagnosis — symptom → test → cause → lever

Companion to `.claude/skills/release/SKILL.md`. Every entry below happened; the
numbers are measured, not illustrative.

**Last checked:** 2026-08-21 · **Owner:** MIDAS maintainers

---

## A published package cannot be installed

**Test.** In a clean venv:
`pip install --dry-run --no-cache-dir <pkg>==<ver>`. Do **not** settle for PyPI's
`info.version` — that only says the file exists, not that its dependencies do.

**Cause.** It floors a version that never published. A release event tests
dependents, so one failing dependent blocks several publishes at once; whichever
packages already went out are left naming versions that are not there.

**Measured.** A `midas_ff_pipeline` failure blocked five releases. Four
already-published packages (`pipeline 0.15.0`, `suite 0.10.0`, `calibrate-v2
0.8.0`, `integrate-v2 0.4.0`) were uninstallable for ~40 minutes.

**Lever.** Get the blocked packages out — fix, retarget their tags, re-fire.
**Prevent** it by bumping every floor target before the first tag, and by testing
the affected set (`.github/scripts/changed_packages.py`), not the touched set.

---

## A release run is green but nothing published

**Test.** `gh run view <id> --json jobs` and look for the `publish` job
specifically. Then read PyPI.

**Cause, one of two.**

- The tag matched nothing in the workflow's `elif` chain — a **silent green
  no-op**. Note `midas-dct-tt-v*` must be matched *before* `midas-dct-v*`,
  because the shorter prefix glob-matches the longer tag.
- `gh release create` no-opped or 503'd, so no release event ever fired.

**Lever.** Add the tag to the publish map; always confirm creation with
`gh release view`, never the exit code.

---

## A publish job failed — rerun, or retarget?

**Test.** Read the failing job's log and classify the failure.

**If a test failed** → the tag points at broken code. `gh run rerun` re-tests the
same commit and fails identically. Fix, delete the release **and** the tag, push
the deletion, recreate the tag at the fix commit, recreate the release.

**If the infrastructure failed** — `ConnectionResetError` against Sigstore/Rekor
while signing the attestation, a 503, a network reset — the artifact and the tag
are correct. **`gh run rerun --failed`.**

**Measured.** `midas-nf-pipeline 0.6.4` died with
`ConnectionResetError(104)` in `attestations.py`. Attestation runs *before*
upload, so nothing partial landed (confirmed: 0.6.4 absent from PyPI). A plain
rerun published it.

---

## The audit says a package is in sync, but the fix is not shipped

**Test.** `git diff --name-only <last-tag> HEAD -- packages/<pkg>` and compare
against the audit's anchor, `packages/<pkg>/<pkg>` — the **inner** directory.

**Cause.** The change is outside that prefix. `tests/` being outside is correct
(nothing shipped changed). **`c_src/` being outside is a trap**: it is compiled
into the wheel by scikit-build-core, so the wheel changes and the audit cannot
see it.

**Measured.** A `MargStrain` change to `FitUnified.c` was the only change in
`midas_fit_grain`. The audit reported in sync; trusting it would have stranded
the C fix at the old version indefinitely.

**Lever.** Bump on any `c_src/` change regardless of what the audit says.

---

## A commit does not import on its own

**Test.** For each commit in the batch, for any symbol a moved/edited
`__init__.py` exports, check the defining module is in the *same or an earlier*
commit:

```bash
git show <sha>:path/to/__init__.py | grep -c NewSymbol
git show <sha>:path/to/module.py   | grep -c 'class NewSymbol'
```

**Cause.** Staging by pattern. A glob or a directory groups by *path shape*, not
by *dependency*.

**Measured.** `io/__init__.py` (exports `BadPixelSentinelWarning`) landed one
commit before `readers.py` (defines it). `import midas_calibrate_v2.io` raised
`ImportError` at that commit — breaking bisect and any CI run landing on it.

**Lever.** Explicit paths, and a per-commit standalone check. Both commits were
unpushed, so `git reset --mixed HEAD~2` and regrouping was clean.

---

## The canonical pull will not merge

**Test.** Before anything destructive, for every dirty path:

```bash
diff <(git show origin/master:"$f") "$f"
```

Read the **direction**. Lines only in canonical mean canonical is ahead.

**Cause A — canonical is ahead.** Real work exists there and nowhere else.

> Measured: 30 lines of GPU device-placement work (`--device`, `.to(spec.device())`,
> a `_geom_device()` helper) lived only in canonical, uncommitted. A plain pull
> would have destroyed it. Recovered by diffing, confirming canonical's copies
> were strict supersets, and committing them.

**Cause B — untracked files block the merge.** `git checkout -- .` cleans tracked
files only, so a pre-check on tracked paths passes and the pull still aborts with
*"untracked working tree files would be overwritten"*.

**Lever.** Hash each blocking untracked file against origin; delete only the
byte-identical ones and let the pull restore them. If it differs, stop.

---

## The environment reports the old version but runs new code

**Test.** `importlib.metadata.version()` against the behaviour — call the new
function, check the new flag.

**Cause.** Editable installs freeze metadata at install time. The code tracks the
tree; the recorded version does not.

**Lever.** Re-run `pip install --no-deps --no-build-isolation -e <pkg>`. Never
without `--no-deps` — pip will otherwise pull PyPI wheels over the editable
installs and sever the canonical linkage.

---

## pip reports success but installed almost nothing

**Test.** Re-query the environment after every install. Never accept
`PIP EXIT: 0` plus a clean `pip check` as evidence.

**Cause.** The pin list was edited by a pattern that silently matched nothing —
e.g. an anchor of `^midas-suite==0.10.1$` against a line ending in a quote.

**Measured.** 2 of 3 packages were never added to the list. pip installed one,
exited 0, and `pip check` was clean.

**Lever.** Count the pins before running (`[ ${#args[@]} -ne 19 ] && exit 1`), and
diff the environment afterwards.

---

## The scrub hook blocks a commit

**Test.** `python utils/scrub_check.py --staged` after staging.

**Cause.** A real beamtime, user, sample or DM-group name in a **tracked** file.
It fires on test files and code comments, not just manuals.

**Lever.** Pseudonymise per the convention in `BEAMTIME_KEY.md` (git-excluded),
then verify **both** sides — the name is gone from the tree *and* the mapping
actually landed.

> A `python` insert once failed on a bad anchor and reported nothing. The manual
> was scrubbed and the key silently had no mapping — the worst of both.

**It does not scan commit messages.** One real path reached a public commit
message this way.

---

## A doc citation stops resolving

**Test.** `python utils/doc_citation_check.py`.

**Causes, in order of likelihood.**

1. The code moved. **Read the new line and confirm it still supports the claim**
   before repointing.
2. The path is not repo-root-relative. A citation written bare as
   ENVELOPE.md:99 resolves against the **repo root**, not the citing file's own
   folder, even when a file of that name sits beside it. Write the full path:
   `manuals/xrd-ct/ENVELOPE.md:99`.
3. The target is outside the repo (`~/.claude/known-limits.md`). Cite it by name
   without a line range; a repo-relative cite can never resolve.
4. A bare `§5b-ter` that lives in a sibling file — qualify it with the filename.

**Lever.** Fix the citation or the prose. Never delete it to silence the hook.

---

## `git stash -u` and the owner's untracked files

Isolating the committed state with `git stash -u` touches untracked files —
including the owner's local scratch. If used, verify the restore immediately:
stash list empty, every untracked file back, spot-check a version string. Say
that it was used; do not let them find out from `git stash list`.
