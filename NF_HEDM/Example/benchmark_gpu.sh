#!/bin/bash
#
# GPU vs CPU benchmark for NF-HEDM FitOrientation
#
# Usage: bash benchmark_gpu.sh [nCPUs] [--screen-only] [--gpu-only] [--small] [--gpu-fit] [--gpu-double]
#   nCPUs:        number of CPU threads (default: 96)
#   --screen-only: skip Phase 2 fitting (pure screening benchmark)
#   --gpu-only:    skip preprocessing and CPU benchmark
#   --small:       use grs.csv (4 orientations) instead of cubicSeed.txt
#   --gpu-fit:     enable GPU Phase 2 NM fitting (MIDAS_GPU_FIT=1)
#   --gpu-double:  use double-precision objective (MIDAS_GPU_DOUBLE=1, slower but matches CPU)
#
# Run this from the NF_HEDM/Example/sim directory (where SpotsInfo.bin
# and the diffraction images live).
#
set -euo pipefail

NCPUS=96
SCREEN_ONLY=0
GPU_ONLY=0
USE_SMALL=0
GPU_FIT=0
GPU_DOUBLE=0
for arg in "$@"; do
  case "$arg" in
    --screen-only) SCREEN_ONLY=1 ;;
    --gpu-only) GPU_ONLY=1 ;;
    --small) USE_SMALL=1 ;;
    --gpu-fit) GPU_FIT=1 ;;
    --gpu-double) GPU_DOUBLE=1; GPU_FIT=1 ;;
    [0-9]*) NCPUS=$arg ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIDAS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_DIR="$MIDAS_DIR/NF_HEDM/bin"
if [ "$USE_SMALL" = 1 ]; then
  SEED_FILE="$SCRIPT_DIR/grs.csv"
else
  SEED_FILE="$MIDAS_DIR/NF_HEDM/cubicSeed.txt"
fi
PARAM_FILE="ps_au.txt"

echo "=== NF-HEDM GPU Benchmark ==="
echo "MIDAS_DIR:  $MIDAS_DIR"
echo "Working in: $(pwd)"
echo "SEED_FILE:  $SEED_FILE"
echo "nCPUs:      $NCPUS"
[ "$SCREEN_ONLY" = 1 ] && echo "MODE:       screen-only (Phase 2 skipped)"
[ "$GPU_ONLY" = 1 ] && echo "MODE:       gpu-only (skipping preprocessing + CPU)"
[ "$USE_SMALL" = 1 ] && echo "MODE:       small dataset (grs.csv, 4 orientations)"
[ "$GPU_FIT" = 1 ] && echo "MODE:       GPU Phase 2 fitting enabled"

# Set env vars
export MIDAS_SCREEN_ONLY=$SCREEN_ONLY
if [ "$GPU_FIT" = 1 ]; then
  export MIDAS_GPU_FIT=1
fi
if [ "$GPU_DOUBLE" = 1 ]; then
  export MIDAS_GPU_DOUBLE=1
  echo "MODE:       GPU double-precision objective (CPU-matching)"
fi

# Check we're in a directory with SpotsInfo.bin
if [ ! -f SpotsInfo.bin ]; then
  echo "ERROR: SpotsInfo.bin not found in $(pwd)"
  echo "Run this script from the sim/ directory that has processed image data."
  exit 1
fi

if [ "$GPU_ONLY" = 0 ]; then
# --- Regenerate orientation-dependent binary files ---
echo ""
echo "=== STEP 1: Regenerating orientation files ==="

# For --small, convert grs.csv to seedOrientations.txt format
if [ "$USE_SMALL" = 1 ]; then
  echo "  Converting grs.csv to seedOrientations.txt..."
  "$BIN_DIR/GenSeedOrientationsFF2NFHEDM" "$SEED_FILE" seedOrientations.txt
  SEED_FILE="$(pwd)/seedOrientations.txt"
  # Also set GrainsFile
  if ! grep -q "^GrainsFile " "$PARAM_FILE"; then
    echo "GrainsFile $SCRIPT_DIR/grs.csv" >> "$PARAM_FILE"
  else
    sed -i.bak "s|^GrainsFile .*|GrainsFile $SCRIPT_DIR/grs.csv|" "$PARAM_FILE"
  fi
fi

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

fi  # end if GPU_ONLY=0 (preprocessing + CPU)

if [ "$GPU_ONLY" = 0 ]; then
# --- Run CPU benchmark ---
echo ""
echo "=== STEP 2: CPU Benchmark ($NCPUS threads) ==="
time "$BIN_DIR/FitOrientationOMP" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee cpu_bench.log
if [ "$SCREEN_ONLY" = 0 ]; then
  cp Au_bin_Reconstructed.mic cpu_benchmark.mic
fi
echo "CPU done."
fi  # end CPU benchmark

# --- Run GPU benchmark ---
echo ""
echo "=== STEP 3: GPU Benchmark ==="

if [ "$GPU_DOUBLE" = 1 ] && [ "$SCREEN_ONLY" = 0 ]; then
  # Run FLOAT mode first
  echo "--- STEP 3a: GPU Float Mode ---"
  unset MIDAS_GPU_DOUBLE
  export MIDAS_GPU_FIT=1
  time "$BIN_DIR/FitOrientationGPU" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee gpu_bench_float.log
  cp Au_bin_Reconstructed.mic gpu_float.mic
  echo "GPU float done."

  # Run DOUBLE mode
  echo ""
  echo "--- STEP 3b: GPU Double Mode ---"
  export MIDAS_GPU_DOUBLE=1
  time "$BIN_DIR/FitOrientationGPU" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee gpu_bench_double.log
  cp Au_bin_Reconstructed.mic gpu_double.mic
  cp Au_bin_Reconstructed.mic gpu_benchmark.mic  # default for parity check
  echo "GPU double done."
else
  time "$BIN_DIR/FitOrientationGPU" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee gpu_bench.log
  if [ "$SCREEN_ONLY" = 0 ]; then
    cp Au_bin_Reconstructed.mic gpu_benchmark.mic
  fi
  echo "GPU done."
fi

# --- Parity check (only if Phase 2 ran) ---
if [ "$SCREEN_ONLY" = 0 ]; then
  echo ""
  echo "=== STEP 4: Parity Check ==="

  if [ "$GPU_DOUBLE" = 1 ] && [ -f gpu_float.mic ] && [ -f gpu_double.mic ]; then
    # Parity check for both modes
    for MODE_NAME in float double; do
      echo "--- ${MODE_NAME} mode ---"
      python3 -c "
import struct
def read_mic(fn):
    with open(fn,'rb') as f: data=f.read()
    n=len(data)//(11*8)
    return [struct.unpack_from('11d',data,i*11*8) for i in range(n)]
cpu=read_mic('cpu_benchmark.mic')
gpu=read_mic('gpu_${MODE_NAME}.mic')
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
    done
  else
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

  # Spatial parity maps
  echo ""
  echo "=== STEP 5: Spatial Parity Maps ==="
  PARITY_SCRIPT="$SCRIPT_DIR/parity_maps.py"
  if [ -f "$PARITY_SCRIPT" ]; then
    if [ "$GPU_DOUBLE" = 1 ] && [ -f gpu_float.mic ] && [ -f gpu_double.mic ]; then
      python3 "$PARITY_SCRIPT" cpu_benchmark.mic gpu_float.mic gpu_double.mic 225
    else
      python3 "$PARITY_SCRIPT" cpu_benchmark.mic gpu_benchmark.mic 225
    fi
  else
    echo "WARNING: parity_maps.py not found at $PARITY_SCRIPT"
  fi
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

# --- Diagnostic CSV comparison ---
if [ -f screen_cpu.csv ] && [ -f screen_gpu.csv ]; then
  echo ""
  echo "========================================"
  echo "       SCREENING PARITY CHECK"
  echo "========================================"
  CPU_COUNT=$(tail -n +2 screen_cpu.csv | wc -l | tr -d ' ')
  GPU_COUNT=$(tail -n +2 screen_gpu.csv | wc -l | tr -d ' ')
  echo "CPU winners: $CPU_COUNT"
  echo "GPU winners: $GPU_COUNT"

  # Compare sorted CSVs
  if diff -q screen_cpu.csv screen_gpu.csv > /dev/null 2>&1; then
    echo "RESULT: ✓ EXACT MATCH"
  else
    echo "RESULT: ✗ MISMATCH"
    echo ""
    echo "--- CPU only (not in GPU) ---"
    comm -23 <(tail -n +2 screen_cpu.csv | sort) <(tail -n +2 screen_gpu.csv | sort) | head -20
    echo ""
    echo "--- GPU only (not in CPU) ---"
    comm -13 <(tail -n +2 screen_cpu.csv | sort) <(tail -n +2 screen_gpu.csv | sort) | head -20
  fi
  echo "========================================"
fi
