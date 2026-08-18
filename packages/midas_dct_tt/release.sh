#!/usr/bin/env bash
# Release a new version of midas-dct-tt.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release + PyPI
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag
#
# Example:
#   ./release.sh 0.1.1 --publish
#
# NOTE: this package is PRIVATE. The guard in step 0 makes every mode except
# --dry-run refuse to run while it is git-excluded. That is deliberate: see
# RELEASE_CHECKLIST.md.

set -e

# --- Arg parsing ---
if [ -z "$1" ]; then
    echo "Usage: $0 <new_version> [--publish | --dry-run]"
    echo "  <new_version>    e.g. 0.1.1"
    echo "  --publish        push to GitHub + create release + upload to PyPI"
    echo "  --dry-run        prepare artifacts but don't commit/tag"
    exit 1
fi

NEW_VERSION="$1"
MODE="${2:-prepare}"   # default: prepare only

if [ "$MODE" != "prepare" ] && [ "$MODE" != "--publish" ] && [ "$MODE" != "--dry-run" ]; then
    echo "ERROR: unknown flag '$MODE'. Use --publish or --dry-run."
    exit 1
fi

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG_DIR"
TAG="midas-dct-tt-v${NEW_VERSION}"
WORKFLOW="../../.github/workflows/python-packages.yml"

echo "=== Releasing midas-dct-tt v${NEW_VERSION} (mode: ${MODE}) ==="
echo

# --- 0. PRIVATE guard -------------------------------------------------------
# While packages/midas_dct_tt/ is git-excluded (.git/info/exclude), nothing here
# is committable and a "release" would either be an empty commit or -- worse, if
# someone force-adds the tree -- an unannounced disclosure of the whole project.
# Refuse. (The rule is in .git/info/exclude, not .gitignore: a tracked ignore
# entry naming this package would itself leak it. See RELEASE_CHECKLIST.md.)
# --dry-run is allowed through: it only builds artifacts locally.
REL_PATH="$(git rev-parse --show-prefix 2>/dev/null || true)"
if [ -n "$REL_PATH" ] && git check-ignore -q "$REL_PATH" 2>/dev/null; then
    if [ "$MODE" = "--dry-run" ]; then
        echo "NOTE: package is still private (git-excluded). --dry-run is allowed;"
        echo "      it builds into dist/ and touches nothing else."
        echo
    else
        echo "ERROR: this package is still PRIVATE -- '${REL_PATH%/}' is git-excluded."
        echo
        echo "Releasing it means making the project public. Work through"
        echo "RELEASE_CHECKLIST.md first (remove the .git/info/exclude entry, add the"
        echo "tag prefix to the CI publish map, register the PyPI trusted publisher)."
        echo
        echo "To build artifacts locally without releasing:  $0 ${NEW_VERSION} --dry-run"
        exit 1
    fi
fi

# --- 1. Safety checks ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "ERROR: not on master (on $CURRENT_BRANCH). Switch branches first."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/midas_dct_tt/. Commit or stash first."
    git status -s -- .
    exit 1
fi

# Tag must not exist (local or remote)
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag $TAG already exists locally. Pick a different version or delete it:"
    echo "  git tag -d $TAG"
    exit 1
fi

if [ "$MODE" = "--publish" ] && git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
    echo "ERROR: tag $TAG already exists on origin. Pick a different version."
    exit 1
fi

# The publish job's tag -> package map is a hand-maintained elif chain; a tag it
# does not match produces a silent no-op build and nothing reaches PyPI. Check
# before pushing rather than discovering it in the Actions log.
if [ "$MODE" = "--publish" ] && ! grep -q 'midas-dct-tt-v\*' "$WORKFLOW"; then
    echo "ERROR: python-packages.yml has no branch for 'midas-dct-tt-v*'."
    echo "       The release would build nothing and publish nothing."
    echo "       Add to the 'Identify package from tag' step (see RELEASE_CHECKLIST.md):"
    echo
    echo "         elif [[ \"\$TAG\" == midas-dct-tt-v* ]]; then"
    echo "           echo \"package=midas_dct_tt\" >> \"\$GITHUB_OUTPUT\""
    echo "           echo \"pypi_name=midas-dct-tt\" >> \"\$GITHUB_OUTPUT\""
    echo
    echo "       It MUST come before any 'midas-dct-v*' branch (prefix collision,"
    echo "       same trap as midas-integrate-v2 / midas-calibrate-v2)."
    exit 1
fi

# --- 2. Bump version ---
# Keep the pre-bump copies. `git checkout --` cannot restore these while the
# package is excluded (git has no committed version of it to restore), so
# every abort path below would otherwise leave the bumped version on disk.
BACKUP_DIR="$(mktemp -d)"
cp pyproject.toml "$BACKUP_DIR/pyproject.toml"
cp midas_dct_tt/__init__.py "$BACKUP_DIR/__init__.py"

restore_version() {
    cp "$BACKUP_DIR/pyproject.toml" pyproject.toml
    cp "$BACKUP_DIR/__init__.py" midas_dct_tt/__init__.py
    echo "  Version restored to $(grep '^version = ' pyproject.toml | cut -d'"' -f2)."
}

echo "[1/7] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_dct_tt/__init__.py
rm -f pyproject.toml.bak midas_dct_tt/__init__.py.bak

PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_dct_tt/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    exit 1
fi

# --- 3. Run tests ---
echo "[2/7] Running tests..."
# macOS conda envs ship duplicate libomp.dylib (numba + torch); SIGABRTs at
# import without this. Tests are not OpenMP-stress and run cleanly with it.
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q --tb=short || {
    echo "ERROR: tests failed. Aborting."
    restore_version
    exit 1
}

# --- 4. Build ---
echo "[3/7] Building package..."
rm -rf dist/ build/ *.egg-info/

if ! python -c "import build" 2>/dev/null; then
    echo "  Installing 'build' and 'twine'..."
    pip install --quiet build twine
fi

set -o pipefail
python -m build 2>&1 | tail -5
set +o pipefail

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "ERROR: build did not produce artifacts."
    restore_version
    exit 1
fi

# --- 5. If dry-run, stop here ---
if [ "$MODE" = "--dry-run" ]; then
    echo
    echo "=== Dry run complete ==="
    echo "Artifacts in dist/:"
    ls -1 dist/
    echo
    echo "Reverting the version bump (a dry run must leave no trace):"
    restore_version
    exit 0
fi

# --- 6. Commit + tag ---
echo "[4/7] Committing version bump..."
# Both the diff check and the commit are pathspec-limited to the two version
# files. Without that, anything the user happened to have staged before running
# this script gets swept into the "bump version" commit -- and, under --publish,
# pushed and tagged with it. A bare `git diff --cached --quiet` has the same bug
# in reverse: unrelated staged files make it report changes even when the
# version on disk is already correct.
VERSION_FILES=(pyproject.toml midas_dct_tt/__init__.py)

git add -- "${VERSION_FILES[@]}"
if git diff --cached --quiet -- "${VERSION_FILES[@]}"; then
    echo "  Version was already at ${NEW_VERSION} on disk; skipping commit."
else
    git commit -m "midas-dct-tt: bump version to ${NEW_VERSION}" \
        -- "${VERSION_FILES[@]}"
fi

echo "[5/7] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-dct-tt v${NEW_VERSION}"

# --- 7. If --publish, push + GitHub release (CI auto-uploads to PyPI) ---
if [ "$MODE" = "--publish" ]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "ERROR: 'gh' (GitHub CLI) not installed. Install: brew install gh"
        exit 1
    fi

    echo "[6/6] Pushing to GitHub..."
    git push origin master --follow-tags

    echo "[6b/6] Creating GitHub release..."
    gh release create "$TAG" dist/* \
        --title "midas-dct-tt v${NEW_VERSION}" \
        --generate-notes

    echo
    echo "=== Release prepared ==="
    echo "GitHub: https://github.com/marinerhemant/MIDAS/releases/tag/${TAG}"
    echo
    echo "The python-packages.yml workflow will now:"
    echo "  1. Run tests on Linux (Python 3.11/3.12)"
    echo "  2. Build the sdist + wheel"
    echo "  3. Publish to PyPI via trusted publishing (OIDC)"
    echo
    echo "Watch progress: https://github.com/marinerhemant/MIDAS/actions"
    echo
    echo "When workflow completes:"
    echo "  PyPI: https://pypi.org/project/midas-dct-tt/${NEW_VERSION}/"
    echo "  Verify: pip install -U midas-dct-tt && \\"
    echo "          python -c 'import midas_dct_tt; print(midas_dct_tt.__version__)'"
    exit 0
fi

# --- Default (prepare only): show next steps ---
echo
echo "=== Release prepared locally ==="
echo
echo "Artifacts in dist/:"
ls -1 dist/
echo
echo "To publish, run:"
echo
echo "  git push origin master --follow-tags"
echo "  gh release create ${TAG} dist/* \\"
echo "    --title 'midas-dct-tt v${NEW_VERSION}' \\"
echo "    --generate-notes"
echo
echo "The GitHub Actions workflow will build and upload to PyPI"
echo "automatically when the release is created."
echo
echo "Or re-run with --publish next time to do all of this automatically:"
echo "  ./release.sh ${NEW_VERSION} --publish"
echo
