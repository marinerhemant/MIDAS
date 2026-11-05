#!/bin/bash
# EXPERIMENTAL. Zip + hkl + peakfit for ONE pf scan file, with a per-scan pseudo-dark. Run one or two at a time (see README: the zipper holds ~44 GB).
#   export EXP=<experiment>; source ../kit_env.sh; ./pf_peak_one.sh <scan_folder> <file_nr> <gpu> <ncpus> <ring-thresh e.g. 1:75,2:150> <preproc_thr>
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../kit_env.sh" || exit 1
FOLDER=${1:?}; NR=${2:?}; GPU=${3:?}; NC=${4:?}; RT=${5:?}; THR=${6:?}
D=$WORK/pf/sub/$NR; mkdir -p $D/results; cd $D
export CUDA_VISIBLE_DEVICES=$GPU
F=$RAW/varexD/$FOLDER/${FOLDER}_$(printf %06d $NR).vrx.h5
T0=$(date +%s); echo "$(date +%T) start nr=$NR gpu=$GPU"
[ -f bg.h5 ] || $PY -B -I "$KIT/ff/make_bg_dark.py" $F bg.h5 120 12 > bg.log 2>&1 || { echo "bg FAILED"; exit 1; }
EXTRA=""; [ -f "$(dirname "$CAL_PARAMS")/residual_corr.bin" ] && EXTRA="ResidualCorrectionMap=$(dirname "$CAL_PARAMS")/residual_corr.bin"
$PY -B -I "$KIT/ff/make_params.py" "$CAL_PARAMS" Parameters.txt SpaceGroup=$TARGET_SG "LatticeConstant=$TARGET_LAT" FileStem=$FOLDER \
  StartFileNrFirstLayer=$NR NrFilesPerSweep=1 RawFolder=$RAW/varexD/$FOLDER/ MarginOme=0.5 Completeness=0.5 Wedge=0 Dark=$D/bg.h5 darkLoc=/exchange/dark \
  RingThresh="$RT" OverAllRingToIndex=${PF_SEED_RING:-1} $EXTRA > params.log 2>&1
midas-pipeline run --scan-mode ff --params Parameters.txt --result results --layers 1-1 --only zip_convert --only hkl --only peakfit \
   --preproc-thresh $THR --n-cpus $NC > chain.log 2>&1
echo "exit=$? nr=$NR elapsed $(( $(date +%s) - T0 )) s"
