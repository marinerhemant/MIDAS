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

**A green local sweep does NOT prove the dependency declarations are complete.**
Your environment satisfies imports that a clean install does not, so a missing
`dependencies` entry is invisible locally and fails only on CI's fresh runner.

> `midas_calibrate_v2` imported `skimage` at the top of `make_seed` — the normal
> autocalibrate path — and never declared `scikit-image`. Every local sweep
> passed because the env had it via another package. A clean
> `pip install midas-calibrate-v2` had raised ModuleNotFoundError the first time
> anyone seeded a geometry, for as long as the seeder had existed. It surfaced
> only when new tests became the first to call `make_seed` on CI.

For any NEW top-level import added this batch, check it is declared. Blocking the
module at the import hook and re-running the entry point is the cheap proof.

**A green local sweep says nothing about platform NUMERICS either.** This Mac is
the only machine in the loop running Apple's libm; CI is glibc. Any assertion of
*exact* floating-point equality is therefore a platform assertion, whether or not
it looks like one.

> Measured: `midas_stress` passed 410/410 locally and failed CI on both 3.11 and
> 3.12. `test_euler_values_unchanged` asserted
> `np.abs(ref - got).max() == 0.0` — but `ref` is the NumPy/`math` backend and
> `got` is the torch backend, two different implementations. glibc and Apple
> libm differ by a few ULP: 1.609823385706477e-15 rad on 3.11,
> 2.220446049250313e-15 on 3.12, in one or two of 1200 components. It blocked a
> release that was otherwise finished. The same difference is already documented
> for the `scanning_5grain_golden` fixture, whose Linux BASELINE differs from
> the stored macOS golden at ~3e-15 while every discrete decision matches.

Scan the batch's new tests for `== 0.0`, `== 0`, and bare `assert a == b` on
floats **before** tagging. Where the two sides come from different backends or
different libraries, bound it — and say in the docstring that the bound is
*platform tolerance*, not a defect being admitted. That sentence is load-bearing:
this project has caught two tests relaxing a tolerance to hide a real problem,
and a later reader cannot distinguish the two cases without being told which
one they are looking at.

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

  **Scrub the WHOLE batch before the first commit, not commit by commit.**
  `scrub_check` only sees *staged* files, so committing in groups reveals the
  hits one group at a time and each fix is a fresh interruption.

  ```bash
  git add -A -- packages/ manuals/ utils/ .claude/   # stage everything
  python utils/scrub_check.py --staged               # one pass, all hits
  git reset -q                                       # unstage, then commit properly
  ```

  > Measured: a first pass reported 5 hits. Doing it group-by-group turned into
  > **six rounds** — 5 in one package, then 22 files of a new capability, then a
  > *filename* (`phase3_<name>.py`, where `\b` does not match after `_`), then
  > **uppercase constants** a case-sensitive pattern missed, then two more
  > beamtimes one at a time. 26 hits total.

  Then stop iterating: sweep the staged tree yourself for the whole token shape,
  so the remaining set is known rather than discovered.

  ```bash
  git diff --cached --name-only | while read -r f; do [ -f "$f" ] && \
    grep -ohE '\b[a-zA-Z][a-zA-Z0-9]*_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[0-9]{2}[a-z]?\b' "$f"; \
  done | sort -u
  ```

  Match **case-insensitively and inside identifiers** — one name appeared as
  lowercase prose, an ALL-CAPS test constant, Title-case, and embedded in a
  script filename. A `\bname\b` pattern misses the last two.

  **Disambiguate before mapping.** Two beamtimes can share a surname, differing
  only in the month suffix; read the full path in context and pick the right
  pseudonym rather than the first matching row in the key.

  **Re-run the affected test suites afterwards.** A scrub edits source and test
  files — renaming a constant inside a test is exactly the kind of change that
  breaks quietly.
- **doc-citation-check** — a `path:line` citation that no longer resolves. Read
  the cited line and confirm it still supports the claim *before* editing the
  citation. Do not delete a citation to silence it.

  Two ways a *newly written* citation fails on its first run, both cheap to avoid:

  - **Path-qualify anything whose basename is not unique.** A bare
    `registry.py:892` is AMBIGUOUS — `midas_params` and `midas_parsl_configs`
    both have one, so the claim cannot be checked. Write the path.
  - **The SYMBOL check reads backticked identifiers NEXT to the citation**, and
    fails if none of them appears within ±40 lines of the cited range. Naming the
    package — ``` `midas_params` (`registry.py:892`) ``` — fails, because
    `midas_params` is not *in* the file. Put a symbol that is actually there
    (`typical=0.8`) beside it, or drop the identifier.

  **If a commit MOVES cited lines, the citations move in that same commit** —
  the hook runs per commit, against the working tree, so a C insertion
  invalidates every citation pointing past it *inside the commit that makes it*.
  Two consequences, and they reorder the batch:

  - **Commit the docs FIRST.** Otherwise staging a manual for a one-line
    citation fix sweeps in all its unrelated content changes, and the guard
    cannot help you — the file genuinely is in both sets.
  - **One new line number may not cover every intermediate state.** The ±40
    window is generous but finite. Measured: a `DetTx` citation moved
    2567 → 2857 → 2906 → 2920 across three C commits. One value covered the last
    two; the first needed its own. Compute where the symbol lands in EACH state
    before deciding how many edits you need.

  **A green run means the pointer resolves, NOT that it points at the claim.**
  The ±40-line symbol window is wide on purpose, and it will happily accept a
  citation that is simply wrong.

  > Measured, in this skill's own batch: a new citation read `registry.py:893`
  > for the shipped default `typical=0.8`. Line 893 is `zarr_rename=…`; the value
  > is on **892**. The checker passed it, because `typical` was inside the
  > window. Open the cited line and read it — every new citation, once.

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

**Vendored C is duplicated on purpose — sync it, never centralise it.** Eight
files (`forward.{c,h}`, `MIDAS_Math.{c,h}`, `GetMisorientation.{c,h}`,
`IndexerConsolidatedIO.h`, `nelder_mead.c`) exist byte-identically in
`midas_ckernel`, `midas_fit_grain` and `midas_index`. They must: the latter two
compile from their OWN sdists, so a build reaching into a sibling package works
in a checkout and fails for every pip user, and `midas-ckernel` is deliberately
unpublished so it cannot be a build-time dependency either.

`midas_ckernel/c_src` is canonical. Edit there, then:

```bash
python utils/sync_vendored_c.py            # canonical -> mirrors
python utils/sync_vendored_c.py --check    # CI/test mode, exit 1 on drift
```

A one-copy fix leaves three packages computing different forward models with
every test green — `test_forward_parity.py` compares ckernel against the LEGACY
bodies, a different axis, and stays green with the mirrors diverged.

**An UNPUBLISHED package gets the bump and the commit, never the tag.**
`midas-ckernel` is deliberately not on PyPI (404, and it is the standing
`--ignore` for the audit). It nonetheless has a `release.sh` AND an entry in the
CI publish map, so everything about it looks releasable — and pushing its tag
would publish it.

> Caught with the tag already created locally: the tagging loop had bumped and
> tagged all 14 changed packages, ckernel among them. Deleted before the push;
> it had never reached origin.

Its C still ships, because the published mirrors vendor a byte-identical copy —
ckernel is the canonical *source*, not a delivery vehicle. Confirm that with
`cmp` rather than assuming it.

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

**Neither is authoritative, and they disagree in BOTH directions.** The JSON
reported `integrate-v2 0.5.0` live while pip's index could not yet resolve it,
and the prod install failed outright. Six weeks later the reverse: `stress
0.12.0` was `JSON=no, pip-resolve=yes`, the JSON cache trailing the index.

> So do not learn "the JSON runs ahead". Learn that **pip-resolve is the one
> that decides**, because it is what a user's install actually does. The JSON is
> a second opinion worth having — a disagreement means one of them is stale and
> you should wait — but it never overrules a clean resolve, and a clean JSON
> never substitutes for one.

**A release event tests dependents.** The `midas-stress` release ran 45 test
jobs. Do not conclude otherwise from one green run of a package that happens to
have none.

### When a publish fails

Diagnose *what* failed, not *that* it failed:

Read the failing JOB, not just the run. There are **three** classes, not two:

| failure | meaning | fix |
|---|---|---|
| **code** — a `test` job failed | the tag points at broken code | fix, then **retarget the tag** (delete release + tag, recreate at the fix commit). `gh run rerun` re-tests the same broken commit. |
| **infrastructure** — network, Sigstore/Rekor reset, 503 | artifact and tag are fine | **`gh run rerun --failed`** |
| **`400 File already exists`** | the package **ALREADY PUBLISHED**; this is a second, redundant upload being refused | **nothing.** A rerun fails identically. Confirm it is live and move on. |

**Measure the blast radius before you retarget — do not assume it is the batch.**
A release event tests the released package and its **dependents**, so most
matrices never run a given package's own suite. Which releases come back GREEN
tells you how far the failure actually reaches.

> Measured: `midas-stress 0.12.0` failed on a bad test and 13 other tags were
> pointing at the same commit. Retargeting all 13 looked like the obvious move.
> Then `midas-suite 0.11.0` came back green from that same commit — `midas_stress`
> is a dependency of suite, not a dependent, so suite's matrix never ran its
> tests. Real blast radius: **one package**. And two of the 13 had already
> published, so retargeting them would have fired duplicate uploads that can
> only fail.

Wait for enough of the matrix to report that you can see the shape, then retarget
only what actually failed AND has not published.

**Re-check publication state immediately before EACH retarget, not once for the
batch.** PyPI files are immutable, so retargeting a tag whose version already
published fires a duplicate upload that can only fail.

> Measured: a batch-wide "nothing has published" check, then several minutes
> rebuilding 16 wheels, then the deletes. Packages published during that window.
> Result: 7 red runs, every one a duplicate-upload refusal on a version that had
> already succeeded. Zero broken packages, but the wall of red is indistinguish-
> able at a glance from a real cascade.

When a retarget does collide, prove the damage is nil rather than asserting it:
`git diff --name-only <old> <new>` to show which packages' content actually
differs, and download a published wheel to check CRC, RECORD, and that it
contains the feature it was released for.

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

**Check the API the probe uses; do not guess it.** A probe that reports a false
FAIL costs the same investigation as a real one, and erodes trust in the whole
set.

> `backend_c.BINARY` does not exist — the accessor is `binary_path()`. The probe
> reported "c-omp binary located: False" on all three environments. The binaries
> were fine.

**For C, probe the binary, not the version.** C recompiles only on reinstall, so
a stale binary reports the correct package version while running the old parser.
Grep the executable for the new key:

```python
raw = open(backend_c.binary_path(), "rb").read()
report("refiner binary parses MargStrain", b"MargStrain" in raw)
```

and force the reinstall rather than trusting the version check:
`pip install --no-cache-dir --force-reinstall --no-deps midas-index==X midas-fit-grain==Y`.

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

**Count the occurrences before AND after any bulk edit**, and make the loop print
both. It is the only check that catches a rewrite which ran and changed nothing.

> `FILES=$(grep -rl ...)` then `for f in $FILES` edited **zero** of 13 files:
> **zsh does not word-split an unquoted parameter**, so `perl` was handed all
> thirteen paths concatenated as a single filename. The failure was one
> `Can't open` line in a wall of output, every exit code was 0, and the sweep
> looked like it had worked. The before/after counts were identical — 26 paths
> before, 26 after — which is what exposed it. Use
> `while IFS= read -r f; do … done < list`.

## If this batch touches a SKILL

A skill's `description` is a **routing surface** — it decides when the skill
fires, and it reads as a support claim. Check it against that doc set's own
`ENVELOPE.md` before committing: a class listed there flatly is one the model
will accept work for.

> Measured: `calibrate-integrate` listed EIGER2 alongside GE (67 verified) and
> Pilatus (32 verified). Its own envelope records EIGER2 at **0 verified of 6**,
> 1.078 px median ring scatter against 0.07–0.2 px elsewhere, and says in
> writing "recoverable, not yet to spec". Qualified in the frontmatter rather
> than dropped, so the skill still engages but does not imply support.

**Check the description against the envelope's HEADER, not only its rows.** The
header is where a doc set names the configurations it covers, and it is the line
that gets forgotten when a second instrument is added.

> Measured: the `nf-hedm` description was widened to "1-ID and 20-ID-D HT-HEDM"
> on the strength of real work — an HDF5 reader, a measured beamstop, a
> re-derived ω sign. But `ENVELOPE.md` still opened **"Instrument: 1-ID
> near-field HEDM"** while six of its own rows were tagged `[20-ID]`. The sibling
> `ff-hedm` header had been updated in the same batch and this one was missed.
> The rows were right and the header was stale — which is the direction that
> reads as an overclaim.

Also verify every path and section the skill cites actually resolves, and that
counts it quotes ("13 rules") match. Those are cheap to check and silently rot.
Bound the numbered cross-references too: the highest `hard rule N` referenced
anywhere in the set must exist in that set's own rule table (measured this batch:
ff 17 defined / 14 referenced, nf 23 / 22, pf 12 / 12).

**A path that names one machine is a delivery defect.** These doc sets are meant
to be run from a beamline host, so `/Users/<someone>/…` and `~/Desktop/…` are
broken instructions there, not cosmetics. `$MIDAS` is the reader's own checkout;
`$ANALYSIS` is a campaign directory that is deliberately **not** in this repo.

> Do not *delete* such a path to fix it — for a harness that produced a quoted
> number, the path IS the provenance, and deleting it leaves a number with
> nothing behind it. Strip the machine-specific prefix, keep the filename, and
> say in the doc set's README that a `$ANALYSIS/...` target is provenance, not a
> link — it names the script a number came from and promises nothing about
> reaching it. Measured: 29 such paths across 16 files, in four doc sets.

## Diagnosis

Symptom → discriminating test → cause → lever:
**`manuals/release/DIAGNOSIS.md`**.
