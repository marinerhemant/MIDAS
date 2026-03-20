#!/bin/bash
#
# GPU vs CPU benchmark for NF-HEDM FitOrientation
#
# Usage: bash benchmark_gpu.sh [nCPUs] [--screen-only]
#   nCPUs:        number of CPU threads (default: 96)
#   --screen-only: skip Phase 2 fitting (pure screening benchmark)
#
# Prerequisites:
#   - MIDAS built with CUDA (FitOrientationGPU + FitOrientationOMP in bin/)
#   - Run from NF_HEDM/Example/ directory (or a copy of it)
#   - ps_au.txt and associated data files present
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

# Work in a separate benchmark directory to avoid clobbering existing data
BENCH_DIR="$SCRIPT_DIR/benchmark_run"
echo "=== NF-HEDM GPU Benchmark ==="
echo "MIDAS_DIR:  $MIDAS_DIR"
echo "BIN_DIR:    $BIN_DIR"
echo "SEED_FILE:  $SEED_FILE"
echo "nCPUs:      $NCPUS"
[ "$SCREEN_ONLY" = 1 ] && echo "MODE:       screen-only (Phase 2 skipped)"

# --- Setup benchmark directory ---
if [ -d "$BENCH_DIR" ]; then
  echo "Cleaning previous benchmark run..."
  rm -rf "$BENCH_DIR"
fi
mkdir -p "$BENCH_DIR"

# Copy parameter file and any needed data
cp "$SCRIPT_DIR/$PARAM_FILE" "$BENCH_DIR/"

# Link data files (images etc) from the Example directory
for f in "$SCRIPT_DIR"/*.tif "$SCRIPT_DIR"/*.ge* "$SCRIPT_DIR"/*.h5; do
  [ -e "$f" ] && ln -sf "$f" "$BENCH_DIR/" 2>/dev/null || true
done
# Link any subdirectories with reduced data
for d in "$SCRIPT_DIR"/*/; do
  dname=$(basename "$d")
  [ "$dname" = "benchmark_run" ] && continue
  [ -d "$d" ] && ln -sf "$d" "$BENCH_DIR/$dname" 2>/dev/null || true
done

cd "$BENCH_DIR"

# --- Update parameter file for benchmark ---
# Use cubicSeed.txt as the seed orientations file
if ! grep -q "^SeedOrientations " "$PARAM_FILE"; then
  echo "SeedOrientations $SEED_FILE" >> "$PARAM_FILE"
else
  sed -i.bak "s|^SeedOrientations .*|SeedOrientations $SEED_FILE|" "$PARAM_FILE"
fi

# Ensure OnlySpotsInfo is off so we do full reconstruction
sed -i.bak 's/^OnlySpotsInfo .*/OnlySpotsInfo 0/' "$PARAM_FILE"

# Count orientations
NR_ORIENT=$(wc -l < "$SEED_FILE")
echo "Seed orientations: $NR_ORIENT"

# Update NrOrientations in param file
sed -i.bak "s/^NrOrientations .*/NrOrientations $NR_ORIENT/" "$PARAM_FILE"

# Read grid info
GRID_SIZE=$(grep "^GridSize " "$PARAM_FILE" | awk '{print $2}')
RSAMPLE=$(grep "^Rsample " "$PARAM_FILE" | awk '{print $2}')
echo "GridSize: $GRID_SIZE, Rsample: $RSAMPLE"

# Set environment for screen-only mode
export MIDAS_SCREEN_ONLY=$SCREEN_ONLY

# --- Run preprocessing (generates binary files) ---
echo ""
echo "=== STEP 1: Preprocessing ==="
echo "Running nf_MIDAS.py preprocessing..."
python3 "$MIDAS_DIR/NF_HEDM/workflows/nf_MIDAS.py" \
  -paramFN "$PARAM_FILE" \
  -nCPUs "$NCPUS" \
  -machineName local \
  -doImageProcessing 0 \
  -restartFrom preprocessing

# Check binary files were created
for f in OrientMat.bin Key.bin DiffractionSpots.bin SpotsInfo.bin; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found after preprocessing"
    exit 1
  fi
  echo "  $f: $(du -h "$f" | cut -f1)"
done

# Count grid points
NVOXELS=$(head -1 grid.txt | awk '{print $1}')
echo "Grid voxels: $NVOXELS"
echo "Orientation matrix file: $(du -h OrientMat.bin | cut -f1)"

# --- Run CPU benchmark ---
echo ""
echo "=== STEP 2: CPU Benchmark ==="
echo "Running FitOrientationOMP with $NCPUS CPUs..."
time "$BIN_DIR/FitOrientationOMP" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee cpu_output.log
echo "CPU benchmark complete."

# --- Run GPU benchmark ---
echo ""
echo "=== STEP 3: GPU Benchmark ==="
echo "Running FitOrientationGPU with $NCPUS CPUs..."
time "$BIN_DIR/FitOrientationGPU" "$PARAM_FILE" 0 1 "$NCPUS" 2>&1 | tee gpu_output.log
echo "GPU benchmark complete."

# --- Compare results (only if Phase 2 ran) ---
if [ "$SCREEN_ONLY" = 0 ]; then
  echo ""
  echo "=== STEP 4: Parity Check ==="
  cp Au_bin_Reconstructed.mic gpu_benchmark.mic
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
else
  echo ""
  echo "(Parity check skipped in screen-only mode)"
fi

# --- Extract timing summary ---
echo ""
echo "=== TIMING SUMMARY ==="
echo "--- CPU ---"
grep -E "^NF CPU:|^=== (CPU|END CPU)" cpu_output.log || echo "(no CPU timing found)"
echo ""
echo "--- GPU ---"
grep -E "^NF GPU:|^=== (GPU|END GPU)" gpu_output.log || echo "(no GPU timing found)"

echo ""
echo "=== Benchmark complete ==="
