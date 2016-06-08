#!/bin/bash

/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/bin/MergeOverlappingPeaks $1 $2
/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/bin/CalcRadius $1 $2
/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/bin/FitTiltBCLsdSample $1
