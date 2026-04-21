#!/usr/bin/env bash
# Release a new version of midas-integrate-gpu.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag
#
# This package compiles C via scikit-build-core. Local build produces the
# sdist only; binary wheels (manylinux + macOS) are built on CI by
# cibuildwheel via .github/workflows/wheels.yml.
#
# Example:
#   ./release.sh 0.1.1 --publish

set -e

# --- Arg parsing ---
if [ -z "$1" ]; then
    echo "Usage: $0 <new_version> [--publish | --dry-run]"
    echo "  <new_version>    e.g. 0.1.1"
    echo "  --publish        push to GitHub + create release (CI builds wheels + uploads to PyPI)"
    echo "  --dry-run        prepare artifacts but don't commit/tag"
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
TAG="midas-integrate-gpu-v${NEW_VERSION}"

echo "=== Releasing midas-integrate-gpu v${NEW_VERSION} (mode: ${MODE}) ==="
echo

# --- 1. Safety checks ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "ERROR: not on master (on $CURRENT_BRANCH). Switch branches first."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/midas_integrate_gpu/. Commit or stash first."
    git status -s -- .
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag $TAG already exists locally. Pick a different version or delete:"
    echo "  git tag -d $TAG"
    exit 1
fi

if [ "$MODE" = "--publish" ] && git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
    echo "ERROR: tag $TAG already exists on origin. Pick a different version."
    exit 1
fi

# --- 2. Pre-flight: C toolchain + NLopt + OpenMP headers present locally ---
echo "[1/8] Checking C toolchain..."
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found. Install: brew install cmake  (macOS) or apt-get install cmake (Linux)"
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not on PATH. The GPU wheel needs CUDA ≥ 12.0 to build."
    echo "  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  Or run release.sh inside an nvidia/cuda:12.4.1-devel-ubuntu22.04 container."
    exit 1
fi

# Sanity: midas-integrate must be installed at the SAME version — both
# packages share C ABI and version mismatch = runtime crash for users.
NEEDED_PIN=$(grep -E '^\s+"midas-integrate==' pyproject.toml | head -1 | sed -E 's/.*==([0-9.]+).*/\1/')
python -c "import midas_integrate; \
    assert midas_integrate.__version__ == '${NEEDED_PIN}', \
    f'pinned to ${NEEDED_PIN} but installed {midas_integrate.__version__}'" \
    2>/dev/null || {
    echo "ERROR: midas-integrate ${NEEDED_PIN} not installed in this env."
    echo "  pip install -e ../midas_integrate"
    exit 1
}

# --- 3. Bump version (pyproject.toml + __init__.py + cmake.define) ---
echo "[2/8] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^cmake.define.PACKAGE_VERSION = \".*\"/cmake.define.PACKAGE_VERSION = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_integrate_gpu/__init__.py
rm -f pyproject.toml.bak midas_integrate_gpu/__init__.py.bak

PYPROJ_VER=$(grep '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_integrate_gpu/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    exit 1
fi

# --- 4. Run tests ---
echo "[3/8] Running tests..."
python -m pytest tests/ -q --tb=short || {
    echo "ERROR: tests failed. Aborting."
    git checkout -- pyproject.toml midas_integrate_gpu/__init__.py
    exit 1
}

# --- 5. Build sdist only (wheels handled on CI) ---
echo "[4/8] Building sdist..."
rm -rf dist/ build/ *.egg-info/

if ! python -c "import build" 2>/dev/null; then
    echo "  Installing 'build' and 'twine'..."
    pip install --quiet build twine
fi

set -o pipefail
python -m build --sdist 2>&1 | tail -10
set +o pipefail

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "ERROR: sdist build did not produce artifacts."
    git checkout -- pyproject.toml midas_integrate_gpu/__init__.py
    exit 1
fi

# --- 6. If dry-run, stop ---
if [ "$MODE" = "--dry-run" ]; then
    echo
    echo "=== Dry run complete ==="
    echo "Artifacts in dist/:"
    ls -1 dist/
    echo
    echo "To undo:"
    echo "  git checkout -- pyproject.toml midas_integrate_gpu/__init__.py"
    exit 0
fi

# --- 7. Commit + tag ---
echo "[5/8] Committing version bump..."
git add pyproject.toml midas_integrate_gpu/__init__.py
git commit -m "midas-integrate-gpu: bump version to ${NEW_VERSION}"

echo "[6/8] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-integrate-gpu v${NEW_VERSION}"

# --- 8. If --publish, push + GitHub release (CI builds wheels + uploads) ---
if [ "$MODE" = "--publish" ]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "ERROR: 'gh' (GitHub CLI) not installed. Install: brew install gh"
        exit 1
    fi

    echo "[7/8] Pushing to GitHub..."
    git push origin master --follow-tags

    echo "[8/8] Creating GitHub release..."
    gh release create "$TAG" dist/* \
        --title "midas-integrate-gpu v${NEW_VERSION}" \
        --generate-notes

    echo
    echo "=== Release prepared ==="
    echo "GitHub: https://github.com/marinerhemant/MIDAS/releases/tag/${TAG}"
    echo
    echo "The wheels.yml workflow will now:"
    echo "  1. Run cibuildwheel on Linux (manylinux_2_28) and macOS (x86_64 + arm64)"
    echo "  2. Build the sdist again for completeness"
    echo "  3. Upload all artifacts to PyPI via trusted publishing (OIDC)"
    echo
    echo "Watch: https://github.com/marinerhemant/MIDAS/actions"
    echo
    echo "When complete (~10-15 min):"
    echo "  https://pypi.org/project/midas-integrate-gpu/${NEW_VERSION}/"
    echo "  pip install -U midas-integrate-gpu && \\"
    echo "    python -c 'import midas_integrate_gpu; print(midas_integrate_gpu.__version__)'"
    exit 0
fi

# --- Default (prepare only) ---
echo
echo "=== Release prepared locally ==="
echo
echo "Artifacts in dist/:"
ls -1 dist/
echo
echo "To publish:"
echo
echo "  git push origin master --follow-tags"
echo "  gh release create ${TAG} dist/* \\"
echo "    --title 'midas-integrate-gpu v${NEW_VERSION}' \\"
echo "    --generate-notes"
echo
echo "CI will build the wheels and upload to PyPI automatically."
echo
echo "Or re-run with --publish next time:"
echo "  ./release.sh ${NEW_VERSION} --publish"
echo
