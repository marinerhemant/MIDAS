# Releasing midas-auto-calibrate

How to cut a new release of `midas-auto-calibrate` to PyPI + GitHub.

## Quick reference

```bash
cd packages/midas_auto_calibrate
./release.sh <new_version> [--publish | --dry-run]
```

## Three modes

| Command | What it does |
|---------|--------------|
| `./release.sh 0.1.1` | **Prepare locally only** (default). Version bump + tests + sdist build + commit + tag. Push/publish manually. |
| `./release.sh 0.1.1 --publish` | **Fully automated**: prepare + push + GitHub release. CI (`wheels.yml`) then builds manylinux + macOS wheels and publishes via trusted publishing (OIDC). |
| `./release.sh 0.1.1 --dry-run` | **Prepare but don't commit or tag**. For testing. |

## Differences from the pure-Python packages

Unlike `midas-stress` / `midas-params`, this package compiles C. A few
extra considerations:

- **Local build uses the system toolchain.** `release.sh` builds only the
  **sdist** locally via `python -m build --sdist`. Binary wheels are built
  on CI using `cibuildwheel` (see [`.github/workflows/wheels.yml`](../../.github/workflows/wheels.yml)).
- **Dependencies**: the sdist build needs `cmake`, a C toolchain, and
  **NLopt** + **libomp** headers on the machine running `release.sh`.
  On macOS: `brew install cmake nlopt libomp`. On Linux (apt):
  `apt-get install -y cmake libnlopt-dev libomp-dev`.
- **Tests**: `pytest tests/ -q` is the release gate. The import and
  binary-discovery tests run without the C build (the binary-dependent
  paths are `skipif`-gated).

## Step-by-step (prepare-only mode)

```bash
cd packages/midas_auto_calibrate
./release.sh 0.1.1
```

Script steps:
1. Verify on `master` with clean working tree.
2. Verify tag `midas-auto-calibrate-v0.1.1` does not exist.
3. Verify `cmake`, C compiler, and NLopt headers are present locally.
4. Bump version in `pyproject.toml`, `midas_auto_calibrate/__init__.py`,
   and the `cmake.define.PACKAGE_VERSION` line in `pyproject.toml`.
5. Run `pytest tests/ -q --tb=short` (aborts release on failure, rolls back).
6. Clean `dist/`, `build/`, `*.egg-info/` and build the **sdist** via
   `python -m build --sdist`. Binary wheels are built on CI, not here.
7. Commit the version bump.
8. Create annotated git tag `midas-auto-calibrate-v0.1.1`.
9. Print remaining manual publish commands.

After the script completes:

```bash
git push origin master --follow-tags
gh release create midas-auto-calibrate-v0.1.1 dist/* \
    --title "midas-auto-calibrate v0.1.1" \
    --generate-notes
```

CI then runs `cibuildwheel`, produces Linux + macOS wheels, and uploads
to PyPI via trusted publishing.

## One-shot mode (`--publish`)

```bash
cd packages/midas_auto_calibrate
./release.sh 0.1.1 --publish
```

Does prepare + push + GitHub release in one command. The `wheels.yml`
workflow takes over.

## Prerequisites (one-time)

- Python env with `pip build`, `cmake`, C toolchain, NLopt + libomp headers.
- `gh` (GitHub CLI) authenticated (`gh auth login`) for `--publish`.
- PyPI Trusted Publisher configured at
  <https://pypi.org/manage/account/publishing/>:
  - Owner: `marinerhemant`
  - Repository: `MIDAS`
  - Workflow: `wheels.yml`
  - Environment: `pypi`
- GitHub environment `pypi` exists in repo settings.

## Version numbering

Semantic Versioning.

| Change | Bump |
|--------|------|
| Bug fix, doc tweak | `0.1.1` → `0.1.2` (patch) |
| New feature, backwards-compatible | `0.1.2` → `0.2.0` (minor) |
| Breaking API or C ABI | `0.2.0` → `1.0.0` (major) |

**Note on C ABI**: if this release changes any header file in `src/c/`
that `midas-integrate` links against, bump the minor version and update
the `midas-auto-calibrate>=X.Y,<X.Y+1` pin in `midas-integrate`'s
`pyproject.toml` to match.

## Troubleshooting

### "NLopt not found" during sdist build
Install it locally: `brew install nlopt` or `apt-get install libnlopt-dev`.

### cibuildwheel fails on manylinux with missing nlopt
Check [`scripts/install_deps_linux.sh`](scripts/install_deps_linux.sh).
This runs inside the `manylinux_2_28_x86_64` container before each wheel.

### macOS arm64 wheel fails to link libomp
Check [`scripts/install_deps_macos.sh`](scripts/install_deps_macos.sh)
exports `OpenMP_ROOT` from Homebrew.

### "file already exists on PyPI"
PyPI rejects re-uploads of the same version. Bump to the next patch.
