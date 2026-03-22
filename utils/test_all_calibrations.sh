#!/bin/bash
# Comprehensive calibration test for 3 detector types
# Run from MIDAS root: bash utils/test_all_calibrations.sh

set -e
SCRIPT=/Users/hsharma/opt/MIDAS/utils/AutoCalibrateZarr.py
CALDIR=/Users/hsharma/opt/MIDAS/FF_HEDM/Example/Calibration

echo "============================================================"
echo "  TEST 1: Pilatus (CeO2, 71.676 keV, 650 mm)"
echo "============================================================"
python3 $SCRIPT \
  --data $CALDIR/CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif \
  --dark $CALDIR/dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif \
  --mask $CALDIR/mask_upd.tif \
  --material ceo2 \
  --wavelength 0.17297 \
  --im-trans 2 \
  --n-iterations 10

echo ""
echo "============================================================"
echo "  TEST 2: Varex (Ceria, 63 keV, 900 mm, px=150µm)"
echo "============================================================"
python3 $SCRIPT \
  --data $CALDIR/Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif \
  --material ceo2 \
  --im-trans 2 \
  --px 150 \
  --n-iterations 10

echo ""
echo "============================================================"
echo "  TEST 3: GE5 (Ceria, ~71.7 keV, Lieghanne)"
echo "============================================================"
GEDIR=/Users/hsharma/Desktop/analysis/lieghanne/LCG_MIDAS_TEST_03092026-selected_no_sum
python3 $SCRIPT \
  --data $GEDIR/ceria_1dfocusbeam_0deg_10f0p2s_000417.ge5.h5 \
  --dark $GEDIR/dark_ceria_1dfocusbeam_0deg_10f0p2s_000418.ge5.h5 \
  --data-loc /exchange/data \
  --dark-loc /exchange/data \
  --material ceo2 \
  --wavelength 0.173058 \
  --max-ring 21 \
  --n-iterations 10

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE"
echo "============================================================"
