#!/usr/bin/env bash
# Guarded PARALLEL package-test sweep for this Mac.
#
# Background (2026-09-13 investigation, see project memory
# reference_mac_parallel_pytest_sigbus.md for the full writeup):
#   - The one CONFIRMED, reproducible concurrency hazard on this machine is diplib's own
#     OpenMP-parallelized filters (MedianFilter/PercentileFilter) deadlocking -- not crashing,
#     hanging forever at 0% CPU -- when 2+ processes call them at the same time. Only
#     midas_calibrate_v2/seed/ imports diplib, and only when use_diplib=True is actually
#     exercised (no test in this repo does that by default as of this writing). Fixed by
#     capping diplib's OWN thread count to 1 via DIPLIB_NUM_THREADS below -- verified: even
#     DIPLIB_NUM_THREADS=2 still deadlocked with 2 concurrent processes, only =1 was safe in
#     repeated trials.
#   - The ORIGINAL documented incident (2026-09-10, midas_dfxm dying with a real SIGBUS while
#     midas_defect + midas_dct_tt ran alongside it) could NOT be reproduced on 2026-09-13
#     despite deliberately re-running that exact trio's real test suites concurrently multiple
#     times (all passed, ~31% faster than serial, zero crashes, zero new entries in
#     ~/Library/Logs/DiagnosticReports/). It may have been a rare/transient race, or something
#     that has since changed. Because it COULD NOT be ruled out, this script keeps concurrency
#     modest (default 2-way) and aborts to serial the moment anything looks wrong, rather than
#     trusting a single clean reproduction.
#
# Usage:
#   manuals/release/parallel_sweep.sh <concurrency> <package1> <package2> ...
#   manuals/release/parallel_sweep.sh 2 midas_dfxm midas_defect midas_dct_tt midas_hkls
#
# Guards applied to EVERY worker:
#   - DIPLIB_NUM_THREADS=1        (the confirmed deadlock fix)
#   - OMP_NUM_THREADS scaled down so concurrency * OMP_NUM_THREADS stays <= physical cores
#     (oversubscription is the generic mechanism every OpenMP-heavy package here shares --
#     even without a proven crash from it, there is no reason to invite it)
#   - KMP_DUPLICATE_LIB_OK=TRUE   (already required per-package; set here too for belt & braces)
#
# Safety net: watches ~/Library/Logs/DiagnosticReports/ for any NEW crash report for the
# duration of the run. If one appears, or any worker exits via a signal (bash reports this as
# exit code > 128), the sweep stops launching new batches immediately, reports which package(s)
# were in flight, and tells you to fall back to the strictly-serial run_sweep.sh pattern for
# the remainder -- it does NOT try to be clever about which one was actually at fault.
set -uo pipefail

CONCURRENCY="${1:?usage: parallel_sweep.sh <concurrency> <pkg1> [pkg2 ...]}"
shift
PACKAGES=("$@")
[ "${#PACKAGES[@]}" -ge 1 ] || { echo "no packages given"; exit 1; }

ROOT="/Users/hsharma/opt/MIDAS"
PY="/Users/hsharma/miniconda3/envs/midas_env/bin/python"
OUT_DIR="${PARALLEL_SWEEP_OUT:-$HOME/Desktop/analysis/midas_release_audit_2026-09/parallel_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR/logs"

PHYS_CORES=$(sysctl -n hw.physicalcpu)
PER_PROC_OMP=$(( PHYS_CORES / CONCURRENCY ))
[ "$PER_PROC_OMP" -lt 1 ] && PER_PROC_OMP=1

echo "=== parallel_sweep: concurrency=$CONCURRENCY, ${#PACKAGES[@]} package(s), OMP_NUM_THREADS/worker=$PER_PROC_OMP (of $PHYS_CORES physical cores) ==="
echo "guards: DIPLIB_NUM_THREADS=1 OMP_NUM_THREADS=$PER_PROC_OMP KMP_DUPLICATE_LIB_OK=TRUE"
echo "out dir: $OUT_DIR"

CRASH_BASELINE=$(ls -1 ~/Library/Logs/DiagnosticReports/*.ips 2>/dev/null | wc -l | tr -d ' ')
ABORT=0
: > "$OUT_DIR/summary.txt"

run_one() {
    local pkg="$1"
    local log="$OUT_DIR/logs/$pkg.log"
    ( cd "$ROOT/packages/$pkg" && \
      KMP_DUPLICATE_LIB_OK=TRUE DIPLIB_NUM_THREADS=1 OMP_NUM_THREADS="$PER_PROC_OMP" \
      VECLIB_MAXIMUM_THREADS="$PER_PROC_OMP" \
      "$PY" -m pytest tests -q -p no:cacheprovider ) > "$log" 2>&1
    echo $? > "$OUT_DIR/logs/$pkg.rc"
}

idx=0
n=${#PACKAGES[@]}
while [ "$idx" -lt "$n" ] && [ "$ABORT" -eq 0 ]; do
    batch=("${PACKAGES[@]:$idx:$CONCURRENCY}")
    echo "--- batch: ${batch[*]} ---" | tee -a "$OUT_DIR/summary.txt"
    pids=()
    for pkg in "${batch[@]}"; do
        run_one "$pkg" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Check for a new crash report BEFORE trusting any individual worker's own exit code --
    # a genuine SIGBUS can sometimes still let the shell wrapper report a clean-looking status.
    crash_now=$(ls -1 ~/Library/Logs/DiagnosticReports/*.ips 2>/dev/null | wc -l | tr -d ' ')
    if [ "$crash_now" -gt "$CRASH_BASELINE" ]; then
        echo "*** NEW CRASH REPORT DETECTED during batch [${batch[*]}] -- aborting to serial ***" \
            | tee -a "$OUT_DIR/summary.txt"
        ABORT=1
    fi

    for pkg in "${batch[@]}"; do
        rc=$(cat "$OUT_DIR/logs/$pkg.rc" 2>/dev/null || echo "?")
        tail -3 "$OUT_DIR/logs/$pkg.log" | tee -a "$OUT_DIR/summary.txt"
        echo "$pkg: rc=$rc" | tee -a "$OUT_DIR/summary.txt"
        if [ "$rc" != "?" ] && [ "$rc" -gt 128 ]; then
            sig=$((rc - 128))
            echo "*** $pkg exited via SIGNAL $sig (rc=$rc) -- aborting to serial ***" \
                | tee -a "$OUT_DIR/summary.txt"
            ABORT=1
        fi
    done

    idx=$((idx + CONCURRENCY))
done

if [ "$ABORT" -eq 1 ]; then
    remaining=("${PACKAGES[@]:$idx}")
    echo "=== ABORTED. Remaining package(s), run these SERIALLY instead: ${remaining[*]:-none} ===" \
        | tee -a "$OUT_DIR/summary.txt"
    exit 1
fi

echo "SWEEP_DONE (parallel, no crash detected)" | tee -a "$OUT_DIR/summary.txt"
