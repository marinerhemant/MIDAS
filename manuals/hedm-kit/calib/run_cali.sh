#!/bin/bash
# CeO2/LaB6 calibration with midas-calibrate-v2 at the REAL beam energy, one fit per calibrant file in CAL_IMAGES.
# Usage:  export EXP=<experiment>; source ../kit_env.sh; ./run_cali.sh
# experiment.env must define: ENERGY_KEV (explog value, NOT the file name), LSD_MM, NPIX, PX_UM, IMTRANS, CAL_IMAGES, and the
# placeholder sample keys TARGET_SG / TARGET_LAT. Optional: CAL_CALIBRANT (default CeO2), CAL_RAW_FOLDER + CAL_FILE_STEM + CAL_START_NR
# (a real scan of the sample: calibrate-v2 wants one to write the sample keys), CAL_DARK_GROUP (default exchange/dark).
# WARNING (reference beamtime): calibrant files whose exchange/dark is all zeros are refused by the tool (a zero dark subtracts nothing).
# 20-ID-D stores the dark frames in exchange/bright of the dark_before file; point CAL_DARK_GROUP at a group that holds them, or
# build a dark yourself; the 63 keV Ceria files in that beamtime were refused for this reason.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../kit_env.sh" || exit 1
kit_check || exit 1
CAL=${CAL_CALIBRANT:-CeO2}; DG=${CAL_DARK_GROUP:-exchange/dark}
LAM=$($PY -B -I -c "print(12.398419843320026/$ENERGY_KEV)")
mkdir -p "$WORK/calib"; cd "$WORK/calib"
for t in $CAL_IMAGES; do
  tag=${t%%:*}; f=${t#*:}; mkdir -p $tag
  say "calibrate $tag  $f  lambda=$LAM A (E=$ENERGY_KEV keV, $ENERGY_SOURCE)"
  extra=""
  [ -n "${CAL_RAW_FOLDER:-}" ] && extra="--raw-folder $CAL_RAW_FOLDER --file-stem $CAL_FILE_STEM --start-file-nr $CAL_START_NR --ext .vrx.h5 --padding 6"
  midas-calibrate-v2 --mode ff --image "$f" --image-group exchange/data --dark-group $DG --reduce median \
     --calibrant $CAL --wavelength $LAM --initial-lsd $(( ${LSD_MM%.*} * 1000 )) --px $PX_UM --n-pixels $NPIX --im-trans $IMTRANS --skip-frame $SKIPFRAME \
     $extra --omega-start 180 --omega-step -0.25 --space-group $TARGET_SG --lattice $TARGET_LAT \
     --ring-thresh 1:100 --ring-thresh 2:100 --ring-to-index 1 \
     --output $tag/ps_calibrated.txt --verbose > $tag/run.log 2>&1
  echo "exit=$?"; tail -3 $tag/run.log | cut -c1-200
done
echo "Read each run.log: strain in microstrain and PASS/FAIL; agree the Lsd between exposures (reference beamtime: four fits 1039.60 to 1039.66 mm, 4.7 to 12.2 ue)."
echo "Then set CAL_PARAMS in experiment.env to the fit you use, and look at the ring overlay PNG the tool writes (a converged fit can sit on the wrong ring assignment)."
