#!/usr/bin/env bash
# Release a new version of midas-dt.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release + PyPI
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag
#
# Example:
#   ./release.sh 0.1.1 --publish

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
TAG="midas-dt-v${NEW_VERSION}"

echo "=== Releasing midas-dt v${NEW_VERSION} (mode: ${MODE}) ==="
echo

# --- 1. Safety checks ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "ERROR: not on master (on $CURRENT_BRANCH). Switch branches first."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/midas_dt/. Commit or stash first."
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

# midas-dt reconstructs through midas-tomo. Releasing a midas-dt that floors a
# midas-tomo version PyPI does not have yet gives users an uninstallable
# package -- the failure appears at their `pip install`, not here.
TOMO_FLOOR=$(grep -o 'midas-tomo>=[0-9][0-9.]*' pyproject.toml | head -1 | cut -d= -f3)
if [ -n "$TOMO_FLOOR" ] && [ "$MODE" = "--publish" ]; then
    echo "[0/7] Checking midas-tomo>=${TOMO_FLOOR} is on PyPI..."
    if ! python - "$TOMO_FLOOR" <<'PY'
import json, sys, urllib.request
floor = sys.argv[1]
try:
    with urllib.request.urlopen(
        "https://pypi.org/pypi/midas-tomo/json", timeout=15) as fh:
        rel = json.load(fh)["releases"]
except Exception as exc:                       # offline: warn, don't block
    print(f"  WARNING: could not reach PyPI ({exc}); skipping the check")
    sys.exit(0)
sys.exit(0 if floor in rel else 1)
PY
    then
        echo "ERROR: midas-tomo ${TOMO_FLOOR} is not on PyPI. Release it first."
        exit 1
    fi
fi

# --- 2. Bump version ---
echo "[1/7] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" midas_dt/__init__.py
rm -f pyproject.toml.bak midas_dt/__init__.py.bak

PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' midas_dt/__init__.py | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    exit 1
fi

# --- 3. Run tests ---
echo "[2/7] Running tests..."
# KMP_DUPLICATE_LIB_OK: macOS conda envs ship duplicate libomp.dylib (numba +
# torch); imports SIGABRT without it. Harmless elsewhere.
#
# -rs prints skip reasons. This package has no C of its own, but it inherits
# midas-tomo's skip trap: test_recon.py and test_branches_e2e.py skip at module
# level when the engine is not built -- 15 tests, covering reconstruction and
# the branch comparison. A green run without them means the wrapper was tested
# and nothing else. Expected counts:
#   midas-tomo engine built : 180 passed
#   no engine at all        : 165 passed, 2 skipped
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q --tb=short -rs || {
    echo "ERROR: tests failed. Aborting."
    git checkout -- pyproject.toml midas_dt/__init__.py
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
    git checkout -- pyproject.toml midas_dt/__init__.py
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
    echo "  git checkout -- pyproject.toml midas_dt/__init__.py"
    exit 0
fi

# --- 6. Commit + tag ---
echo "[4/7] Committing version bump..."
# Pathspec-limited on both the diff check and the commit: without it, anything
# the user happened to have staged gets swept into the bump commit and -- under
# --publish -- pushed and tagged with it.
VERSION_FILES=(pyproject.toml midas_dt/__init__.py)

git add -- "${VERSION_FILES[@]}"
if git diff --cached --quiet -- "${VERSION_FILES[@]}"; then
    echo "  Version was already at ${NEW_VERSION} on disk; skipping commit."
else
    git commit -m "midas-dt: bump version to ${NEW_VERSION}" \
        -- "${VERSION_FILES[@]}"
fi

echo "[5/7] Tagging as ${TAG}..."
git tag -a "$TAG" -m "midas-dt v${NEW_VERSION}"

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
        --title "midas-dt v${NEW_VERSION}" \
        --generate-notes

    echo
    echo "=== Release prepared ==="
    echo "GitHub: https://github.com/marinerhemant/MIDAS/releases/tag/${TAG}"
    echo
    echo "The python-packages.yml workflow will now:"
    echo "  1. Run tests on Python 3.11/3.12"
    echo "  2. Build the sdist + wheel"
    echo "  3. Publish to PyPI via trusted publishing (OIDC)"
    echo
    echo "Watch progress: https://github.com/marinerhemant/MIDAS/actions"
    echo
    echo "When workflow completes:"
    echo "  PyPI: https://pypi.org/project/midas-dt/${NEW_VERSION}/"
    echo "  Verify: pip install -U midas-dt && \\"
    echo "          python -c 'import midas_dt; print(midas_dt.__version__)'"
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
echo "    --title 'midas-dt v${NEW_VERSION}' \\"
echo "    --generate-notes"
echo
echo "The GitHub Actions workflow will build and upload to PyPI"
echo "automatically when the release is created."
echo
echo "Or re-run with --publish next time to do all of this automatically:"
echo "  ./release.sh ${NEW_VERSION} --publish"
echo
