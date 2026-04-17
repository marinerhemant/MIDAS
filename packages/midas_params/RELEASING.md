# Releasing midas-params

This document describes how to cut a new release of the
`midas-params` package to PyPI and GitHub.

## Quick reference

```bash
cd packages/midas_params
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.1` | **Prepare locally only** (default, safest). Version bump + tests + build + commit + tag. You push/publish manually. |
| `./release.sh 0.1.1 --publish` | **Fully automated**: prepare + push + GitHub release. The CI workflow (`python-packages.yml`) then runs tests and auto-publishes to PyPI via trusted publishing (OIDC). One command. |
| `./release.sh 0.1.1 --dry-run` | **Prepare but don't commit or tag**. For testing the build. Easy to undo with `git checkout -- pyproject.toml midas_params/__init__.py`. |

## Step-by-step (prepare-only mode)

This is the default. The script stops after building artifacts so you
can review before publishing.

```bash
cd packages/midas_params
./release.sh 0.1.1
```

The script will:
1. Verify you're on `master` with a clean working tree.
2. Verify the tag `midas-params-v0.1.1` does not already exist.
3. Bump the version in `pyproject.toml` and `midas_params/__init__.py`.
4. Run the test suite (aborts release on failure, rolls back version).
5. Clean `dist/`, `build/`, and `*.egg-info/`, then build the sdist
   and wheel via `python -m build`.
6. Commit the version bump.
7. Create an annotated git tag `midas-params-v0.1.1`.
8. Print the remaining manual commands to push + publish.

After the script completes, you run:

```bash
git push origin master --follow-tags      # push commit + all annotated tags

gh release create midas-params-v0.1.1 dist/* \
    --title "midas-params v0.1.1" \
    --generate-notes                       # creates GitHub release
```

The CI workflow publishes to PyPI automatically when the release is created.

## One-shot mode (`--publish`)

For routine releases where you trust the script:

```bash
cd packages/midas_params
./release.sh 0.1.1 --publish
```

This does the local + GitHub part end-to-end:
prepare → commit → tag → push → GitHub release.

The GitHub Actions workflow (`.github/workflows/python-packages.yml`)
takes over from there:
1. Runs tests on Linux and macOS across Python 3.9/3.11/3.12.
2. Builds the sdist + wheel.
3. Uploads to PyPI via trusted publishing (OIDC) — no API token
   needed.

The script prints the GitHub release URL when the local phase is
done, plus a link to the Actions page so you can watch progress.
Once the workflow finishes (typically 3-5 minutes), the package is
live on PyPI.

## Dry-run mode (`--dry-run`)

For testing the build without modifying git history:

```bash
cd packages/midas_params
./release.sh 0.1.1 --dry-run
```

The script bumps version, runs tests, and builds, but does NOT commit
or tag. To undo the local version bump:

```bash
git checkout -- pyproject.toml midas_params/__init__.py
```

Useful for first-time testing of the release pipeline or verifying
a build on a new machine.

## Safety features

The script refuses to release in unsafe situations:

- **Not on master**: release must be cut from `master`.
- **Uncommitted changes**: release must start from a clean tree.
- **Tag collision**: aborts if the tag already exists locally
  (and in `--publish` mode, if it exists on `origin`). Prevents the
  "file already exists on PyPI" error, because PyPI refuses
  re-uploads of the same version.
- **Test failure rollback**: if tests fail, the version bump is
  automatically reverted.
- **Build failure rollback**: if the build step fails, the version
  bump is automatically reverted.
- **Dependency auto-install**: `build` and `twine` are installed
  automatically if missing.
- **Prerequisite checks**: `--publish` mode verifies `gh` (GitHub CLI)
  is available before touching anything.

## Prerequisites (one-time setup)

### For all modes
- Python environment with `pip` working.
- Working `git` config and push access to `origin`.

### For `--publish` mode additionally
- `gh` (GitHub CLI) installed and authenticated (`gh auth login`).
- GitHub Actions workflow `python-packages.yml` configured (already
  present in this repo).
- PyPI Trusted Publisher configured at
  https://pypi.org/manage/account/publishing/ with:
  - **PyPI project name**: `midas-params`
  - **Owner**: `marinerhemant`
  - **Repository name**: `MIDAS`
  - **Workflow name**: `python-packages.yml`
  - **Environment name**: `pypi`
- GitHub environment named `pypi` exists in the repo settings
  (Settings → Environments) and is shared with `midas-stress`.

The CI workflow handles the PyPI upload automatically when a release
is created, using OIDC trusted publishing. No API token needed.

## Version numbering

Follow [Semantic Versioning](https://semver.org):

| Change | Bump |
|--------|------|
| Bug fix, doc tweak, no API change | `0.1.0` → `0.1.1` (patch) |
| New feature, backwards compatible | `0.1.1` → `0.2.0` (minor) |
| Breaking API change | `0.2.0` → `1.0.0` (major) |

## Troubleshooting

### "tag already exists"
Either pick a higher version, or delete the local tag:
```bash
git tag -d midas-params-v0.1.1
```

### "file already exists on PyPI"
PyPI rejects re-uploads of the same version, ever (even after
deletion). Bump to the next patch version and retry.

### Tests failed, release aborted
The version bump was rolled back. Fix the tests, commit, and retry.

### Build failed, release aborted
The version bump was rolled back. Check `python -m build` output
for the error (usually a missing dependency in `pyproject.toml`).

### GitHub release created but PyPI upload failed (CI workflow error)
Check the workflow logs: https://github.com/marinerhemant/MIDAS/actions

Common causes:
- **"Trusted publishing exchange failure"**: the pending publisher
  at PyPI isn't configured (see Prerequisites above), or the `pypi`
  GitHub environment doesn't exist.
- **Tag prefix mismatch**: the workflow publishes based on tag
  prefix (`midas-params-v*` → midas-params, `midas-stress-v*` →
  midas-stress). Make sure the tag was created by `./release.sh`
  and follows that naming.
- **Tests failed in CI**: something works locally but not in the
  matrix (e.g., Python 3.9 incompatibility). Fix the test, bump
  the version, and retry.

If you need to publish manually as a fallback (e.g., CI is broken
and you need to ship urgently):
```bash
cd packages/midas_params
twine upload dist/*
```
Requires a PyPI API token in `~/.pypirc`.

## Related packages

- [`midas_stress`](../midas_stress/RELEASING.md) — same release
  flow, different package. Both publish via the shared
  `python-packages.yml` workflow.
