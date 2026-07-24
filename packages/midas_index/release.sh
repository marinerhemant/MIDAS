#!/usr/bin/env bash
# Release a new version of midas-index.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release + PyPI
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <new_version> [--publish | --dry-run]"
    exit 1
fi

NEW_VERSION="$1"
MODE="${2:-prepare}"

if [ "$MODE" != "prepare" ] && [ "$MODE" != "--publish" ] && [ "$MODE" != "--dry-run" ]; then
    echo "ERROR: unknown flag '$MODE'. Use --publish or --dry-run."
    exit 1
fi

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG_DIR"
TAG="midas-index-v${NEW_VERSION}"

echo "=== Releasing midas-index v${NEW_VERSION} (mode: ${MODE}) ==="

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "ERROR: not on master (on $CURRENT_BRANCH)."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/midas_index/."
    git status -s -- .
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag $TAG already exists locally."
    exit 1
fi

if [ "$MODE" = "--publish" ] && git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
    echo "ERROR: tag $TAG already exists on origin."
    exit 1
fi

echo "[1/7] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_index/__init__.py
rm -f pyproject.toml.bak midas_index/__init__.py.bak

PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_index/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    exit 1
fi

echo "[2/7] Running tests..."
# KMP_DUPLICATE_LIB_OK=TRUE works around the macOS-only OpenMP runtime
# collision (libomp from torch vs numpy) that aborts process startup with
# "Abort trap: 6". Harmless no-op on Linux/CI where the env var is ignored.
KMP_DUPLICATE_LIB_OK=TRUE \
python -m pytest tests/ -q --tb=short -m "not slow and not gpu and not mps" || {
    echo "ERROR: tests failed. Aborting."
    git checkout -- pyproject.toml midas_index/__init__.py
    exit 1
}

echo "[3/7] Building package..."
rm -rf dist/ build/ *.egg-info/

if ! python -c "import build" 2>/dev/null; then
    pip install --quiet build twine
fi

set -o pipefail
python -m build 2>&1 | tail -5
set +o pipefail

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "ERROR: build did not produce artifacts."
    git checkout -- pyproject.toml midas_index/__init__.py
    exit 1
fi

if [ "$MODE" = "--dry-run" ]; then
    echo "=== Dry run complete ==="
    ls -1 dist/
    echo "Undo: git checkout -- pyproject.toml midas_index/__init__.py"
    exit 0
fi

echo "[4/7] Committing version bump..."
# Both the diff check and the commit are pathspec-limited to the two version
# files. Without that, anything the user happened to have staged before running
# this script gets swept into the "bump version" commit -- and, under --publish,
# pushed and tagged with it. A bare `git diff --cached --quiet` has the same bug
# in reverse: unrelated staged files make it report changes even when the
# version on disk is already correct.
VERSION_FILES=(pyproject.toml midas_index/__init__.py)

git add -- "${VERSION_FILES[@]}"
if git diff --cached --quiet -- "${VERSION_FILES[@]}"; then
    echo "  Version was already at ${NEW_VERSION}; skipping commit."
else
    git commit -m "midas-index: bump version to ${NEW_VERSION}" \
        -- "${VERSION_FILES[@]}"
fi

echo "[5/7] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-index v${NEW_VERSION}"

if [ "$MODE" = "--publish" ]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "ERROR: 'gh' (GitHub CLI) not installed."
        exit 1
    fi

    echo "[6/7] Pushing to GitHub..."
    git push origin master --follow-tags

    echo "[7/7] Creating GitHub release..."
    gh release create "$TAG" dist/* \
        --title "midas-index v${NEW_VERSION}" \
        --generate-notes

    echo "=== Release prepared ==="
    echo "GitHub: https://github.com/marinerhemant/MIDAS/releases/tag/${TAG}"
    exit 0
fi

echo "=== Release prepared locally ==="
ls -1 dist/
echo "To publish:"
echo "  git push origin master --follow-tags"
echo "  gh release create ${TAG} dist/* --title 'midas-index v${NEW_VERSION}' --generate-notes"
