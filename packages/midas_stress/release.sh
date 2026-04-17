#!/usr/bin/env bash
# Release a new version of midas-stress.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release + PyPI
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag
#
# Example:
#   ./release.sh 0.1.3 --publish

set -e

# --- Arg parsing ---
if [ -z "$1" ]; then
    echo "Usage: $0 <new_version> [--publish | --dry-run]"
    echo "  <new_version>    e.g. 0.1.3"
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
TAG="midas-stress-v${NEW_VERSION}"

echo "=== Releasing midas-stress v${NEW_VERSION} (mode: ${MODE}) ==="
echo

# --- 1. Safety checks ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "ERROR: not on master (on $CURRENT_BRANCH). Switch branches first."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/midas_stress/. Commit or stash first."
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

# --- 2. Bump version ---
echo "[1/7] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_stress/__init__.py
rm -f pyproject.toml.bak midas_stress/__init__.py.bak

PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_stress/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    exit 1
fi

# --- 3. Run tests ---
echo "[2/7] Running tests..."
python -m pytest tests/ -q --tb=short || {
    echo "ERROR: tests failed. Aborting."
    git checkout -- pyproject.toml midas_stress/__init__.py
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
    git checkout -- pyproject.toml midas_stress/__init__.py
    exit 1
fi

# --- 5. If dry-run, stop here ---
if [ "$MODE" = "--dry-run" ]; then
    echo
    echo "=== Dry run complete ==="
    echo "Artifacts in dist/:"
    ls -1 dist/
    echo
    echo "To undo the version bump:"
    echo "  git checkout -- pyproject.toml midas_stress/__init__.py"
    exit 0
fi

# --- 6. Commit + tag ---
echo "[4/7] Committing version bump..."
git add pyproject.toml midas_stress/__init__.py
git commit -m "midas-stress: bump version to ${NEW_VERSION}"

echo "[5/7] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-stress v${NEW_VERSION}"

# --- 7. If --publish, push + GitHub release + PyPI ---
if [ "$MODE" = "--publish" ]; then
    # Check prerequisites
    if ! command -v gh >/dev/null 2>&1; then
        echo "ERROR: 'gh' (GitHub CLI) not installed. Install: brew install gh"
        exit 1
    fi
    if ! command -v twine >/dev/null 2>&1; then
        pip install --quiet twine
    fi

    echo "[6/7] Pushing to GitHub..."
    git push origin master --follow-tags

    echo "[6b/7] Creating GitHub release..."
    gh release create "$TAG" dist/* \
        --title "midas-stress v${NEW_VERSION}" \
        --generate-notes

    echo "[7/7] Uploading to PyPI..."
    twine upload dist/*

    echo
    echo "=== Release complete ==="
    echo "PyPI: https://pypi.org/project/midas-stress/${NEW_VERSION}/"
    echo "GitHub: https://github.com/marinerhemant/MIDAS/releases/tag/${TAG}"
    echo
    echo "Verify:"
    echo "  pip install -U midas-stress"
    echo "  python -c 'import midas_stress; print(midas_stress.__version__)'"
    exit 0
fi

# --- Default (prepare only): show next steps ---
echo
echo "=== Release prepared locally ==="
echo
echo "Artifacts in dist/:"
ls -1 dist/
echo
echo "To publish everything now, run:"
echo
echo "  git push origin master --follow-tags"
echo "  gh release create ${TAG} dist/* \\"
echo "    --title 'midas-stress v${NEW_VERSION}' \\"
echo "    --generate-notes"
echo "  twine upload dist/*"
echo
echo "Or re-run with --publish next time to do all of this automatically:"
echo "  ./release.sh ${NEW_VERSION} --publish"
echo
