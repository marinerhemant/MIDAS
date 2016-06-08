#!/bin/bash
source ${HOME}/.MIDAS/paths
${BINFOLDER}/MergeOverlappingPeaks $1 $2
${BINFOLDER}/CalcRadius $1 $2
${BINFOLDER}/FitTiltBCLsdSample $1
