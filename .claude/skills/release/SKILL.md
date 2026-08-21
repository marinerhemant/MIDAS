---
name: release
description: >-
  Take uncommitted MIDAS work from a dirty tree to committed, released on PyPI,
  and live in all three environments: survey what changed, test the AFFECTED set
  (not the touched set), group into commits that each stand alone, bump every
  floor target before the first tag, publish and verify from PyPI rather than
  from run status, then pull canonical and refresh prod, dev and the Mac. Use
  when asked to commit, release, publish, bump, cut a version, or "update
  everything", when `git status` is dirty in this repo and the work is ready,
  or when a release went wrong — a blocked publish, a package that installs but
  cannot resolve, a stale environment, a canonical pull that will not merge.
  Covers the `packages/` monorepo; stops and asks outside it.
---

# MIDAS release

Point at the repo. The work is already written; this gets it out safely.

**Nothing here is optional because it is tedious.** Every rule below is in this
file because skipping it produced a wrong answer that did not announce itself —
a blocked publish, an uninstallable package, an environment silently running old
code. The measured consequence is quoted next to each one so it can be weighed,
not just obeyed.

## The commit gate

**Never commit without the owner picking the message.** Draft two — a detailed
body and a `<72` char subject — and wait. "Please commit" authorises the *work*,
not a message they have not read. Once they pick, commit the whole batch.

**No AI co-author or Generated-with trailer.** Office policy.

---

## Phase 0 — survey, before touching anything

```bash
git rev-parse --short HEAD && git status --porcelain
git diff --stat
```

Read every diff. You are about to write commit messages claiming what the change
does; you cannot do that from a filename. Specifically look for:

- **C sources** (`packages/*/c_src/`) — compiled into the wheel, and invisible to
  the audit (§4).
- **A package already version-bumped by the owner.** Do not re-bump it.
- **Loose files at the repo root or under `packages/`** — those are the owner's
  local work. Never commit them, never add them to `.gitignore`, do not ask.

If the owner is still editing, **stop and wait**. A sweep run against a
half-written file reports failures that are not real, and chasing them wastes
the run.

## Phase 1 — test the AFFECTED set, not the touched set

```bash
python .github/scripts/changed_packages.py <last-release-commit> HEAD
```

That prints every package that changed **or depends on one that changed**. Test
all of them.

> Skipping this shipped a break. Nine packages were touched; the script reports
> 33. `midas_ff_pipeline` — untouched, but a dependent — had a cross-package test
> asserting `midas-pipeline` never offers a loss the refiner rejects. The flags
> had been removed. It failed CI, blocked **five** releases, and the four that
> had already published then floored versions that never appeared, leaving
> `pip install midas-pipeline` unresolvable for ~40 minutes.

Run each package separately — a single pytest across the monorepo collides on
duplicate test basenames (`test_diagnostics.py` exists in two packages).

Record the baseline. A failure that is **byte-identical to the previous sweep**
is pre-existing, not yours — say so, with the evidence, rather than fixing or
hiding it.

## Phase 2 — commit

**Explicit paths only.** Not `git add <dir>`, and not a glob.

> `git add <dir>` swept unrelated edits into the wrong commit three times in one
> session. Switching to a `__init__.py$` glob then did it a fourth.

After staging, assert the staged set **equals** the intended set, and refuse if
not. A ready `commit()` guard is in `manuals/release/commit_helper.sh`.

**Every commit must stand alone.** The check is not "did the right files get
staged" — it is "does this commit import, build and test by itself".

> A glob put `io/__init__.py` (which exports a new symbol) in an earlier commit
> than `readers.py` (which defines it). Both were "correctly staged". The first
> commit raised `ImportError` on `import midas_calibrate_v2.io` — breaking bisect
> and any CI run landing there.

Group by *theme*, not by package. When one file legitimately carries two themes
of the same release, keep them in one commit and say so in the message.

**The pre-commit hooks are load-bearing.** They will block you:

- **scrub-check** — a real beamtime, user or sample name in a tracked file.
  Pseudonymise it, record the mapping in `BEAMTIME_KEY.md` (git-excluded), and
  **verify both sides**: the name is gone from the tree *and* the mapping landed.
  It also fires on **test files and code comments**, not just manuals.
- **doc-citation-check** — a `path:line` citation that no longer resolves. Read
  the cited line and confirm it still supports the claim *before* editing the
  citation. Do not delete a citation to silence it.

**The scrub hook does not scan commit messages.** Keep real names out of them
yourself.

## Phase 3 — versions and floors, all before the first tag

Bump **every** package whose shipped content changed, and every floor target,
**in one commit, before any tag exists**.

> CI installs all in-repo packages in a single `uv` resolve. A floor naming a
> version no package in the tree carries yet fails the resolve for *every*
> release event, and a `release: published` event expands to **all** packages.

Put a new floor in the package that needs it, with the reason and the
consequence of being below it. A floor is not a preference; it is a version
below which the answer is wrong and does not say so.

## Phase 4 — what needs a bump (the audit cannot tell you)

`utils/pypi_audit.py` anchors on `packages/<pkg>/<pkg>` — the **inner** dir:

| changed | audit sees it | bump? |
|---|---|---|
| `packages/<pkg>/<pkg>/**.py` | yes | yes |
| `packages/<pkg>/tests/**` | **no** | no — nothing shipped changed |
| `packages/<pkg>/c_src/**` | **no** | **YES** — compiled into the wheel |
| a notebook inside the package dir | yes (non-`.py` ⇒ behaviour change) | yes |

> The `c_src` row is the dangerous one. A `MargStrain` fix to `FitUnified.c` was
> the only change in `midas_fit_grain`; the audit reported it in sync. Trusting
> it would have stranded the C fix at the old version, silently and forever.

Then, in dependency order:

```bash
git fetch --tags --force origin        # --force is NOT optional: a plain fetch
                                       # refuses to move a retargeted tag
( cd packages/<pkg> && ./release.sh <version> )    # prepare only, no --publish
```

`release.sh` skips its own bump commit when the version on disk is already
correct and tags `HEAD` — which is why Phase 3 works. Guard each one: refuse on
untracked files in the package dir (setuptools builds from **disk**, so they land
in the wheel), and assert the tag's tree carries the expected version.

## Phase 5 — publish, and verify from PyPI

```bash
git push origin master --follow-tags
gh release create "$tag" packages/<pkg>/dist/*<ver>* --title "..." --generate-notes
gh release view "$tag" --json tagName -q .tagName      # CONFIRM. Do not trust rc.
```

> `gh release create` has silently no-opped and has returned 503 twice.

A green tag is **not** a published package — the workflow gates upload behind the
full matrix. Verify two ways, because they disagree:

1. PyPI `info.version` — the JSON API
2. `pip install --dry-run --no-cache-dir <pkg>==<ver>` in a clean venv

> The JSON reported `integrate-v2 0.5.0` live while pip's index could not yet
> resolve it, and the prod install failed outright. pip resolves everything
> before installing, so nothing was half-installed — but "the JSON says it's
> there" is not "users can install it".

**A release event tests dependents.** The `midas-stress` release ran 45 test
jobs. Do not conclude otherwise from one green run of a package that happens to
have none.

### When a publish fails

Diagnose *what* failed, not *that* it failed:

| failure | fix |
|---|---|
| **code** — a test failed | the tag points at broken code. Fix, then **retarget the tag** (delete release + tag, recreate at the fix commit). `gh run rerun` re-tests the same broken commit. |
| **infrastructure** — network, Sigstore/Rekor reset, 503 | the artifact and tag are fine. **`gh run rerun --failed`.** Retargeting is pointless churn. |

## Phase 6 — a release is not finished until the environments are

Never ask, never announce — do it.

- **canonical** `~s1iduser/opt/MIDAS_canonical` — pull it
- **remote prod** `…/envs/midas/bin/python` — PyPI-pinned wheels
- **remote dev** `…/envs/midas-dev/bin/python` — editable on canonical
- **local Mac `midas_env`** — the one that gets forgotten

**Canonical can be genuinely AHEAD.** Hash every dirty file against
`git show origin/master:<path>` before pulling, and check the *direction*.

> Canonical once held 30 lines of GPU device-placement work that existed nowhere
> else, uncommitted. A plain pull would have destroyed it.

Untracked files block the pull too, and `git checkout -- .` does not clear them.
Hash them, then remove only the ones byte-identical to origin.

**prod** — pin exactly, never bare `-U`:

```bash
pip install --no-cache-dir --upgrade midas-x==1.2.3 midas-y==4.5.6
```

> `-U` on an already-satisfied floor is a silent no-op. `--no-cache-dir` matters
> because pip's cache serves the *previous* version's metadata.

**dev and Mac** — re-run the editable install:

```bash
pip install --no-deps --no-build-isolation -e packages/<pkg>
```

> Editable installs freeze their metadata at install time: the code tracks the
> tree but `importlib.metadata.version()` keeps reporting the old number.
> `--no-deps` is mandatory — without it pip pulls PyPI wheels over the editable
> installs and severs the canonical linkage.

Packages with C (`midas_fit_grain`, `midas_index`) recompile on reinstall. Include
them or the C fix is not live.

## Phase 7 — verify by behaviour, not by version string

A version number is a claim. Probe the thing the release exists to guarantee, in
the installed artifact, on **all three** environments:

```python
report("--mask reaches every entry point", "--mask" in help_text)
report("spot_aware rejected", raises(ValueError))
report("c-omp indexer available", backend_c.available())
```

> `spec.fix_panel_id == 28` asserted true the entire time that value was failing
> to reach the forward model. Plumbing tests do not catch this class; asserting
> the parameter has an **effect** does.

Finish with `git fetch --tags --force && python utils/pypi_audit.py --ignore midas-ckernel`
and confirm **0 mismatch / 0 stale**. Note the audit reads the *working tree*, so
in-progress edits show as a mismatch that is not a release problem.

---

## Verify after every edit and every install

An exit code is not evidence the change landed.

> Three silent under-applications in one session: a `perl` anchor that matched
> nothing, a citation fix aimed at the wrong file, and a prod pin list where 2 of
> 3 packages were never added — pip then reported `PIP EXIT: 0` and `pip check`
> clean, having installed one of three.

Re-read the file, re-query the environment, diff the result. Every time.

## Diagnosis

Symptom → discriminating test → cause → lever:
**`manuals/release/DIAGNOSIS.md`**.
