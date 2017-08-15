#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
echo $( pwd )
cd $4
echo $( pwd )
lineNR=$3
argumentToGive=$( sed "${lineNR}q;d" grid.txt )
echo $argumentToGive
${BINFOLDER}/IndexScanningHEDM $1 $2 ${argumentToGive}
${BINFOLDER}/FitPosOrStrainsScanningHEDM $1 ${argumentToGive}
