#!/usr/bin/env bash
# Release a new version of midas-stress.
#
# Usage: ./release.sh <new_version>
# Example: ./release.sh 0.1.2

set -e  # exit on any error

if [ -z "$1" ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.1.2"
    exit 1
fi

NEW_VERSION="$1"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG_DIR"

echo "=== Releasing midas-stress v${NEW_VERSION} ==="
echo

# --- 1. Safety checks ---
# Must be on master, no uncommitted changes
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

# --- 2. Bump version in both files ---
echo "[1/6] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_stress/__init__.py
rm -f pyproject.toml.bak midas_stress/__init__.py.bak

# Verify the bump
PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_stress/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    echo "  pyproject.toml: $PYPROJ_VER"
    echo "  __init__.py:    $INIT_VER"
    exit 1
fi

# --- 3. Run tests ---
echo "[2/6] Running tests..."
python -m pytest tests/ -q --tb=short || {
    echo "ERROR: tests failed. Aborting release."
    git checkout -- pyproject.toml midas_stress/__init__.py
    exit 1
}

# --- 4. Clean build and build fresh ---
echo "[3/6] Building package..."
rm -rf dist/ build/ *.egg-info/

# Make sure 'build' is installed
if ! python -c "import build" 2>/dev/null; then
    echo "  Installing 'build' and 'twine' (required for release)..."
    pip install --quiet build twine
fi

# Use pipefail so the exit status of 'python -m build' is preserved
# when piping through tail
set -o pipefail
python -m build 2>&1 | tail -5
set +o pipefail

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "ERROR: build did not produce artifacts in dist/. Aborting."
    # Undo the version bump commit would be ideal but already committed;
    # leave it to the user to fix.
    exit 1
fi

# --- 5. Commit + tag ---
echo "[4/6] Committing version bump..."
git add pyproject.toml midas_stress/__init__.py
git commit -m "midas-stress: bump version to ${NEW_VERSION}"

TAG="midas-stress-v${NEW_VERSION}"
echo "[5/6] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-stress v${NEW_VERSION}"

# --- 6. Show what to do next ---
echo
echo "=== Release prepared locally ==="
echo
echo "Artifacts in dist/:"
ls -1 dist/
echo
echo "Next steps (review, then run):"
echo
echo "  # Push commit + tag to GitHub (triggers CI workflow if configured):"
echo "  git push origin master"
echo "  git push origin ${TAG}"
echo
echo "  # Or upload to PyPI manually:"
echo "  twine upload dist/*"
echo
echo "  # Or create a GitHub release (triggers trusted-publishing workflow):"
echo "  gh release create ${TAG} dist/* \\"
echo "    --title 'midas-stress v${NEW_VERSION}' \\"
echo "    --generate-notes"
echo
