#!/usr/bin/env bash
# Guarded PARALLEL release preparation, serial dependency-ordered publish.
#
# Built 2026-09-14 after a 2-package release (midas-hkls -> midas-defect) took ~2 hours
# end to end, almost entirely from running everything strictly sequentially when only
# PART of it actually needed to be. Breakdown of that run: ~38 min pre-commit sweep,
# then hkls's OWN test suite ran a SECOND time inside its release.sh (~3 min), then
# defect's ran a second time inside ITS release.sh (~10 min) -- while the OTHER package
# sat idle the whole time, even though neither package's local test+build+tag step
# depends on the other's. Only the PUBLISH order is a real constraint (a dependent
# package's floor must resolve on PyPI, so hkls must be live before defect publishes).
#
# This script separates the two: Phase A runs `release.sh <version>` PREPARE mode
# (test + build + tag, no push/publish) for every given package CONCURRENTLY, with the
# same guards as parallel_sweep.sh. Phase B pushes tags and publishes ONE AT A TIME, IN
# THE ORDER GIVEN -- waiting for each package's publish workflow and verifying it live
# on PyPI before starting the next. Do not parallelize Phase B: that ordering is what
# manuals/release/SKILL.md Phase 5 exists to protect, not a wall-clock choice.
#
# Phase A deliberately does NOT wait for the push's own GitHub CI to go green first, and
# never should -- `release.sh` prepare mode only tests, builds and tags LOCALLY, so it has
# no dependency on the remote CI result, and a local-only tag is cheap to delete and redo
# if CI turns out red. Confirmed-green CI is a real gate on Phase B's tag PUSH and publish
# (SKILL.md Phase 4b/5), not on local prepare -- if you're calling `release.sh` by hand
# instead of through this script, run it WHILE CI runs, not after (measured 2026-09-14:
# waiting for CI before even starting a single package's local prepare cost ~15 idle
# minutes for no reason -- the two were testing the same commit independently).
#
# Usage:
#   manuals/release/parallel_release.sh <concurrency> <pkg1>:<ver1> <pkg2>:<ver2> ...
#   manuals/release/parallel_release.sh 2 midas_hkls:0.16.0 midas_defect:0.6.0
#
# The package ORDER you pass IS the publish/dependency order -- put the package other
# packages' floors point at first. Concurrency only affects Phase A (local prepare);
# Phase B is always fully serial regardless of the concurrency value.
#
# What this does NOT automate, on purpose:
#   - Version/floor bumps (SKILL.md Phase 3) -- do those, and commit them, first.
#   - Content-specific PyPI verification (unzip -p checking for the actual new symbol) --
#     this script only confirms the version RESOLVES and downloads; grep the wheel
#     yourself for the specific thing this release is about, per SKILL.md Phase 5/7.
#   - Environment refresh (SKILL.md Phase 6) -- run that after this script succeeds.
#   - Stashing foreign uncommitted work in a package directory release.sh refuses on --
#     that needs a human judgment call (whose work is it, is it safe to set aside),
#     not a script silently doing it. If release.sh's own safety check fires, this
#     script surfaces the error and stops; resolve it by hand (see SKILL.md Phase 0).
set -uo pipefail

CONCURRENCY="${1:?usage: parallel_release.sh <concurrency> <pkg1>:<ver1> [pkg2:ver2 ...]}"
shift
SPECS=("$@")
[ "${#SPECS[@]}" -ge 1 ] || { echo "no package:version pairs given"; exit 1; }

ROOT="/Users/hsharma/opt/MIDAS"
PY="/Users/hsharma/miniconda3/envs/midas_env/bin/python"
OUT_DIR="${PARALLEL_RELEASE_OUT:-$HOME/Desktop/analysis/midas_release_audit_2026-09/parallel_release_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR/logs"

PHYS_CORES=$(sysctl -n hw.physicalcpu)
PER_PROC_OMP=$(( PHYS_CORES / CONCURRENCY ))
[ "$PER_PROC_OMP" -lt 1 ] && PER_PROC_OMP=1

PKGS=(); VERS=()
for spec in "${SPECS[@]}"; do
    PKGS+=("${spec%%:*}")
    VERS+=("${spec##*:}")
done

echo "=== parallel_release: concurrency=$CONCURRENCY (Phase A only), ${#PKGS[@]} package(s) ==="
echo "order (= publish/dependency order): ${PKGS[*]}"
echo "guards: DIPLIB_NUM_THREADS=1 OMP_NUM_THREADS=$PER_PROC_OMP KMP_DUPLICATE_LIB_OK=TRUE"
echo "out dir: $OUT_DIR"

# ---------------------------------------------------------------------------
# Phase A: local prepare (test + build + tag) for every package, CONCURRENTLY.
# ---------------------------------------------------------------------------
CRASH_BASELINE=$(ls -1 ~/Library/Logs/DiagnosticReports/*.ips 2>/dev/null | wc -l | tr -d ' ')

prepare_one() {
    local pkg="$1" ver="$2"
    local log="$OUT_DIR/logs/prepare_$pkg.log"
    ( cd "$ROOT/packages/$pkg" && \
      KMP_DUPLICATE_LIB_OK=TRUE DIPLIB_NUM_THREADS=1 OMP_NUM_THREADS="$PER_PROC_OMP" \
      VECLIB_MAXIMUM_THREADS="$PER_PROC_OMP" \
      ./release.sh "$ver" ) > "$log" 2>&1
    echo $? > "$OUT_DIR/logs/prepare_$pkg.rc"
}

echo "--- Phase A: preparing ${PKGS[*]} concurrently (concurrency=$CONCURRENCY) ---"
idx=0
n=${#PKGS[@]}
PREPARE_FAILED=0
while [ "$idx" -lt "$n" ]; do
    batch_pkgs=("${PKGS[@]:$idx:$CONCURRENCY}")
    batch_vers=("${VERS[@]:$idx:$CONCURRENCY}")
    echo "  batch: ${batch_pkgs[*]}"
    pids=()
    for i in "${!batch_pkgs[@]}"; do
        prepare_one "${batch_pkgs[$i]}" "${batch_vers[$i]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done

    crash_now=$(ls -1 ~/Library/Logs/DiagnosticReports/*.ips 2>/dev/null | wc -l | tr -d ' ')
    if [ "$crash_now" -gt "$CRASH_BASELINE" ]; then
        echo "*** NEW CRASH REPORT during prepare batch [${batch_pkgs[*]}] ***"
        PREPARE_FAILED=1
    fi
    for pkg in "${batch_pkgs[@]}"; do
        rc=$(cat "$OUT_DIR/logs/prepare_$pkg.rc" 2>/dev/null || echo "?")
        echo "  $pkg: prepare rc=$rc"
        tail -8 "$OUT_DIR/logs/prepare_$pkg.log" | sed 's/^/    /'
        [ "$rc" != "0" ] && PREPARE_FAILED=1
    done
    idx=$((idx + CONCURRENCY))
done

if [ "$PREPARE_FAILED" -ne 0 ]; then
    echo "=== Phase A FAILED for at least one package. Nothing pushed, nothing tagged remotely. ==="
    echo "Fix the failure, then re-run (release.sh skips its own bump commit when already correct)."
    exit 1
fi
echo "=== Phase A complete: all ${#PKGS[@]} package(s) tested, built, and tagged locally ==="

# ---------------------------------------------------------------------------
# Phase B: push all tags once, then publish ONE PACKAGE AT A TIME, IN ORDER.
# This part is deliberately NOT parallel -- see the header comment.
# ---------------------------------------------------------------------------
echo "--- Phase B: pushing master + all tags ---"
( cd "$ROOT" && git push origin master --follow-tags )

for i in "${!PKGS[@]}"; do
    pkg="${PKGS[$i]}"; ver="${VERS[$i]}"
    kebab="${pkg//_/-}"
    tag="${kebab}-v${ver}"
    echo "--- Phase B: releasing $pkg ($tag), $((i+1))/${#PKGS[@]} ---"

    ( cd "$ROOT/packages/$pkg" && \
      gh release create "$tag" dist/* --title "$kebab v$ver" --generate-notes )

    echo "  waiting for the release-triggered publish workflow..."
    sleep 20
    run_id=$(gh run list --limit 5 --json databaseId,event,status \
             --jq '[.[] | select(.event=="release")][0].databaseId' 2>/dev/null)
    if [ -z "$run_id" ] || [ "$run_id" = "null" ]; then
        echo "  WARNING: could not find the release-triggered run automatically."
        echo "  Check 'gh run list' by hand and wait for it before releasing the next package."
        exit 1
    fi
    for _ in $(seq 1 60); do
        st=$(gh run view "$run_id" --json status -q .status 2>/dev/null)
        [ "$st" = "completed" ] && break
        sleep 15
    done
    concl=$(gh run view "$run_id" --json conclusion -q .conclusion 2>/dev/null)
    echo "  publish workflow $run_id: $concl"
    if [ "$concl" != "success" ]; then
        echo "*** $pkg's publish workflow did NOT succeed ($concl) -- stopping before the next package ***"
        echo "Diagnose per SKILL.md Phase 5's 'When a publish fails' table before continuing."
        exit 1
    fi

    echo "  verifying $pkg==$ver resolves on PyPI (generic check only -- also grep the wheel yourself)..."
    for _ in $(seq 1 20); do
        pip index versions "$pkg" 2>/dev/null | grep -q "$ver" && break
        sleep 10
    done
    tmp_dl=$(mktemp -d)
    if python -m pip download "${pkg//_/-}==$ver" --no-deps --only-binary=:all: -d "$tmp_dl" \
        > "$OUT_DIR/logs/verify_$pkg.log" 2>&1; then
        echo "  OK: $pkg==$ver downloads from PyPI"
    else
        echo "*** $pkg==$ver did NOT resolve from PyPI after waiting -- stopping before the next package ***"
        cat "$OUT_DIR/logs/verify_$pkg.log"
        exit 1
    fi
    rm -rf "$tmp_dl"
done

echo "=== ALL PACKAGES RELEASED AND VERIFIED LIVE ON PyPI: ${PKGS[*]} ==="
echo "Next: environment refresh (SKILL.md Phase 6) and content-specific behavioral probes (Phase 7)."
