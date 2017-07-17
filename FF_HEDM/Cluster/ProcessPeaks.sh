#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
paramfile=$1 # always fill path
CHART=/
flr=${paramfile%$CHART*}
cd $flr
echo $paramfile
echo $flr
echo $( pwd )
echo "Ring Nr is $2"
filestem=$( awk '$1 ~ /^FileStem/ { print $2 } ' ${paramfile} )
layernr=$( awk '$1 ~ /^LayerNr/ { print $2 } ' ${paramfile} )
startnr=$( awk '$1 ~ /^StartNr/ { print $2 } ' ${paramfile} )
endnr=$( awk '$1 ~ /^EndNr/ { print $2 } ' ${paramfile} )
mkdir -p $flr/Ring${2}/PeakSearch/${filestem}_${layernr}
nfiles=$( ls -l output | wc -l )
nfilesreq=$(( $endnr - $startnr + 2 ))
echo nfiles
echo nfilesreq
while [[ $nfiles != $nfilesreq ]];
do
	sleep 1
	nfiles=$( ls -l output | wc -l )
	echo nfiles
done
${BINFOLDER}/MergeOverlappingPeaks $1 $2
${BINFOLDER}/CalcRadius $1 $2
${BINFOLDER}/FitTiltBCLsdSample $1
