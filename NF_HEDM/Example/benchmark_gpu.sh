#!/bin/bash
#
# GPU vs CPU benchmark for NF-HEDM FitOrientation
#
# Usage: bash benchmark_gpu.sh [nCPUs] [--screen-only]
#   nCPUs:        number of CPU threads (default: 96)
#   --screen-only: skip Phase 2 fitting (pure screening benchmark)
#
# Run this from the NF_HEDM/Example/sim directory (where SpotsInfo.bin
# and the diffraction images live).
#
set -euo pipefail

NCPUS=96
SCREEN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --screen-only) SCREEN_ONLY=1 ;;
    [0-9]*) NCPUS=$arg ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIDAS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_DIR="$MIDAS_DIR/NF_HEDM/bin"
SEED_FILE="$MIDAS_DIR/NF_HEDM/cubicSeed.txt"
PARAM_FILE="ps_au.txt"

echo "=== NF-HEDM GPU Benchmark ==="
echo "MIDAS_DIR:  $MIDAS_DIR"
echo "Working in: $(pwd)"
echo "SEED_FILE:  $SEED_FILE"
echo "nCPUs:      $NCPUS"
[ "$SCREEN_ONLY" = 1 ] && echo "MODE:       screen-only (Phase 2 skipped)"

# Set env vars
export MIDAS_SCREEN_ONLY=$SCREEN_ONLY

# Check we're in a directory with SpotsInfo.bin
if [ ! -f SpotsInfo.bin ]; then
  echo "ERROR: SpotsInfo.bin not found in $(pwd)"
  echo "Run this script from the sim/ directory that has processed image data."
  exit 1
fi

# --- Regenerate orientation-dependent binary files ---
echo ""
echo "=== STEP 1: Regenerating orientation files with cubicSeed.txt ==="

# Update NrOrientations in param file
NR_ORIENT=$(wc -l < "$SEED_FILE")
echo "Seed orientations: $NR_ORIENT"
sed -i.bak "s/^NrOrientations .*/NrOrientations $NR_ORIENT/" "$PARAM_FILE"

# Add SeedOrientations if missing
if ! grep -q "^SeedOrientations " "$PARAM_FILE"; then
  echo "SeedOrientations $SEED_FILE" >> "$PARAM_FILE"
else
  sed -i.bak "s|^SeedOrientations .*|SeedOrientations $SEED_FILE|" "$PARAM_FILE"
fi

# Ensure DataDirectory is set to current dir
if ! grep -q "^DataDirectory " "$PARAM_FILE"; then
  echo "DataDirectory ." >> "$PARAM_FILE"
else
  sed -i.bak 's|^DataDirectory .*|DataDirectory .|' "$PARAM_FILE"
fi

# HKLs
"$BIN_DIR/GetHKLListNF" "$PARAM_FILE"
echo "  HKLs generated"

# Grid
"$BIN_DIR/MakeHexGrid" "$PARAM_FILE"
NVOXELS=$(head -1 grid.txt | awk '{print $1}')
echo "  Grid: $NVOXELS voxels"

# Simulated diffraction spots (generates OrientMat.bin, Key.bin, DiffractionSpots.bin)
echo "  Generating diffraction spots for $NR_ORIENT orientations..."
"$BIN_DIR/MakeDiffrSpots" "$PARAM_FILE" "$NCPUS"
echo "  Done."

# Verify binary files
for f in OrientMat.bin Key.bin DiffractionSpots.bin SpotsInfo.bin; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found"
    exit 1
  fi
  echo "  $f: $(du -h "$f" | cut -f1)"
done

# --- Run CPU benchmark ---
echo ""
echo "=== STEP 2: CPU Benchmark ($NCPUS threads) ==="
time "$BIN_DIR/FitOrientationOMP" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee cpu_bench.log
if [ "$SCREEN_ONLY" = 0 ]; then
  cp Au_bin_Reconstructed.mic cpu_benchmark.mic
fi
echo "CPU done."

# --- Run GPU benchmark ---
echo ""
echo "=== STEP 3: GPU Benchmark ==="
time "$BIN_DIR/FitOrientationGPU" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee gpu_bench.log
if [ "$SCREEN_ONLY" = 0 ]; then
  cp Au_bin_Reconstructed.mic gpu_benchmark.mic
fi
echo "GPU done."

# --- Parity check (only if Phase 2 ran) ---
if [ "$SCREEN_ONLY" = 0 ]; then
  echo ""
  echo "=== STEP 4: Parity Check ==="
  python3 -c "
import struct
def read_mic(fn):
    with open(fn,'rb') as f: data=f.read()
    n=len(data)//(11*8)
    return [struct.unpack_from('11d',data,i*11*8) for i in range(n)]
cpu=read_mic('cpu_benchmark.mic')
gpu=read_mic('gpu_benchmark.mic')
n=min(len(cpu),len(gpu))
match=mismatch=both0=0
for i in range(n):
    cf,gf=cpu[i][10],gpu[i][10]
    if cf==0 and gf==0: both0+=1
    elif abs(cf-gf)<0.02: match+=1
    else: mismatch+=1
active=match+mismatch
print(f'Rows:       {n}')
print(f'Both zero:  {both0}')
print(f'Match(<2%): {match}')
print(f'Mismatch:   {mismatch}')
print(f'Match rate: {match/active*100:.1f}%' if active>0 else 'Match rate: N/A')
"
fi

# --- Timing summary ---
echo ""
echo "========================================"
echo "           TIMING SUMMARY"
echo "========================================"
echo "--- CPU ---"
grep -E "^NF CPU:|^=== (CPU|END CPU)" cpu_bench.log || true
echo ""
echo "--- GPU ---"
grep -E "^NF GPU:|screen-only|^=== (GPU|END GPU)" gpu_bench.log || true
echo "========================================"
