#!/bin/bash
echo "Spot ID:"
echo $1
/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/bin/IndexerLinuxArgsShm paramstest.txt $1
/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/bin/FitPosOrStrains paramstest.txt $1
