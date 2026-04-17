# Releasing midas-stress

This document describes how to cut a new release of the
`midas-stress` package to PyPI and GitHub.

## Quick reference

```bash
cd packages/midas_stress
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.3` | **Prepare locally only** (default, safest). Version bump + tests + build + commit + tag. You push/publish manually. |
| `./release.sh 0.1.3 --publish` | **Fully automated**: prepare + push + GitHub release + PyPI upload. One command. |
| `./release.sh 0.1.3 --dry-run` | **Prepare but don't commit or tag**. For testing the build. Easy to undo with `git checkout -- pyproject.toml midas_stress/__init__.py`. |

## Step-by-step (prepare-only mode)

This is the default. The script stops after building artifacts so you
can review before publishing.

```bash
cd packages/midas_stress
./release.sh 0.1.3
```

The script will:
1. Verify you're on `master` with a clean working tree.
2. Verify the tag `midas-stress-v0.1.3` does not already exist.
3. Bump the version in `pyproject.toml` and `midas_stress/__init__.py`.
4. Run the test suite (aborts release on failure, rolls back version).
5. Clean `dist/`, `build/`, and `*.egg-info/`, then build the sdist
   and wheel via `python -m build`.
6. Commit the version bump.
7. Create an annotated git tag `midas-stress-v0.1.3`.
8. Print the remaining manual commands to push + publish.

After the script completes, you run:

```bash
git push origin master --follow-tags      # push commit + all annotated tags

gh release create midas-stress-v0.1.3 dist/* \
    --title "midas-stress v0.1.3" \
    --generate-notes                       # creates GitHub release

twine upload dist/*                        # upload to PyPI
```

## One-shot mode (`--publish`)

For routine releases where you trust the script:

```bash
cd packages/midas_stress
./release.sh 0.1.3 --publish
```

This does everything end-to-end:
prepare → commit → tag → push → GitHub release → PyPI upload.

It prints the PyPI and GitHub release URLs when done, plus a
verification command to confirm the new version is installable.

## Dry-run mode (`--dry-run`)

For testing the build without modifying git history:

```bash
cd packages/midas_stress
./release.sh 0.1.3 --dry-run
```

The script bumps version, runs tests, and builds, but does NOT commit
or tag. To undo the local version bump:

```bash
git checkout -- pyproject.toml midas_stress/__init__.py
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
  and `twine` are available before touching anything.

## Prerequisites (one-time setup)

### For all modes
- Python environment with `pip` working.
- Working `git` config and push access to `origin`.

### For `--publish` mode additionally
- `gh` (GitHub CLI) installed and authenticated (`gh auth login`).
- `twine` installed (auto-installed by the script if missing).
- PyPI credentials configured. Two options:
  - **API token**: create at https://pypi.org/manage/account/token/
    and put in `~/.pypirc`.
  - **Trusted publisher (OIDC)**: configure at
    https://pypi.org/manage/account/publishing/ — then the upload
    step is not needed because GitHub Actions publishes automatically
    when the release is created.

## Version numbering

Follow [Semantic Versioning](https://semver.org):

| Change | Bump |
|--------|------|
| Bug fix, doc tweak, no API change | `0.1.2` → `0.1.3` (patch) |
| New feature, backwards compatible | `0.1.3` → `0.2.0` (minor) |
| Breaking API change | `0.2.0` → `1.0.0` (major) |

## Troubleshooting

### "tag already exists"
Either pick a higher version, or delete the local tag:
```bash
git tag -d midas-stress-v0.1.3
```

### "file already exists on PyPI"
PyPI rejects re-uploads of the same version, ever (even after
deletion). Bump to the next patch version and retry.

### Tests failed, release aborted
The version bump was rolled back. Fix the tests, commit, and retry.

### Build failed, release aborted
The version bump was rolled back. Check `python -m build` output
for the error (usually a missing dependency in `pyproject.toml`).

### GitHub release created but PyPI upload failed
Run just the PyPI step manually:
```bash
cd packages/midas_stress
twine upload dist/*
```
The PyPI upload is idempotent per-file, so this is safe to retry.
