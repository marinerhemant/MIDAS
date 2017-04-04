#!/bin/bash -eux

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
paramfile=$1 # always relative path
CHART=/
flr=${paramfile%$CHART*}
cd $flr
echo $paramfile
echo $flr
echo $( pwd )
echo "Ring Nr is $2"
${BINFOLDER}/MergeOverlappingPeaks $1 $2

${BINFOLDER}/CalcRadius $1 $2

${BINFOLDER}/FitTiltBCLsdSample $1
