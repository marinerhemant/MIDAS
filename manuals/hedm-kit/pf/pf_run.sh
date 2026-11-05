#!/bin/bash
# EXPERIMENTAL pf-HEDM pass over scans whose peakfit pf_peak_one.sh already did. Status (reference beamtime, 11 scans at 8 um): peakfit, transforms,
# binning and seeded indexing ran, but the map had ONE grain at 1 of 11 positions. Treat as a recipe to debug, not a result.
#   export EXP=<experiment>; source ../kit_env.sh; PF_FIRST=585 PF_STRIDE=4 PF_N=11 PF_STEP_UM=-8 PF_START_UM=40 PF_BEAM_UM=2 ./pf_run.sh prep|seeded
# experiment.env needs CAL_PARAMS; PF_GRAINS = an FF Grains.csv for seeding (stage seeded); PF_SCAN_FOLDER, PF_RT (ring-thresh), PF_THR.
# Positions: scan k is file PF_FIRST + k*PF_STRIDE at y = PF_START_UM + k*PF_STEP_UM; the DESCENDING direction is a convention taken from the
# scan call, not a measurement (pf manual 1.2); the beam half-width PF_BEAM_UM is a guess until the beam is measured.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../kit_env.sh" || exit 1
P0=$WORK/pf/run; WK=$P0/work; mkdir -p $P0/results $WK; cd $P0
for k in $(seq 0 $((PF_N-1))); do
  NR=$((PF_FIRST + k*PF_STRIDE)); mkdir -p $WK/$NR; S=$WORK/pf/sub/$NR/results/LayerNr_1
  for f in $S/*.MIDAS.zip $S/hkls.csv; do ln -sf $f $WK/$NR/; done; [ -e $WK/$NR/Temp ] || ln -s $S/Temp $WK/$NR/Temp
done
[ -f Parameters.txt ] || $PY -B -I "$KIT/ff/make_params.py" "$CAL_PARAMS" Parameters.txt SpaceGroup=$TARGET_SG "LatticeConstant=$TARGET_LAT" FileStem=$PF_SCAN_FOLDER \
  StartFileNrFirstLayer=$PF_FIRST NrFilesPerSweep=$PF_STRIDE RawFolder=$RAW/varexD/$PF_SCAN_FOLDER/ MarginOme=0.5 Completeness=0.5 Wedge=0 \
  Dark=$WORK/pf/sub/$PF_FIRST/bg.h5 darkLoc=/exchange/dark RingThresh="$PF_RT" OverAllRingToIndex=${PF_SEED_RING:-1}
# NrFilesPerSweep is the scan-to-scan FILE STRIDE here, but the prebuilt zips were made with 1: the parameter-refresh guard refuses that, hence --no-refresh-params.
COMMON="--scan-mode pf --params Parameters.txt --result results --layers 1-1 --n-scans $PF_N --scan-step $PF_STEP_UM --scan-start-um $PF_START_UM --beam-size $PF_BEAM_UM --scan-work-dir $WK --preproc-thresh $PF_THR --n-cpus $NCPUS --one-sol-per-vox 0 --no-refresh-params --indexer-backend c-omp --refine-backend c-omp --binning-device cpu"
if [ "${1:-}" = prep ]; then
  midas-pipeline run $COMMON --skip seeding --skip binning --skip indexing --skip refinement --skip find_grains --skip voxel_cleanup --skip sinogen --skip reconstruct --skip fuse --skip potts --skip em_refine --skip-preflight > chain_prep.log 2>&1
else
  midas-pipeline run $COMMON --seeding-mode ff --grains-file "$PF_GRAINS" --resume auto > chain_seeded.log 2>&1
fi
echo "exit=$?"
