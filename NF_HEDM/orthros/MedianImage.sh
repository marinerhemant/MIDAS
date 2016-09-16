#!/bin/bash

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clhome/TOMO1/PeaksAnalysisHemant/DIPLIB/Linuxa64/lib/
export LD_LIBRARY_PATH

/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/NF_HEDM/bin/MedianImage $1 $2
