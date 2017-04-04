#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
echo $1
paramfile=$1 # always relative path
CHART=/
flr=${paramfile%$CHART*}
cd $flr
${BINFOLDER}/MergeOverlappingPeaks $1 $2
${BINFOLDER}/CalcRadius $1 $2
${BINFOLDER}/FitTiltBCLsdSample $1
