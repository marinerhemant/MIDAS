#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

foldername=$1
layernr=$2
paramfile=${foldername}/Layer${layernr}_MultiRing_ps.txt
for (( detNr=1; detNr<=4; detNr++ ))
do
	cp ${foldername}/Detector${detNr}/InputAll.csv ${foldername}/InputAll${detNr}.csv
	cp ${foldername}/Detector${detNr}/InputAllExtraInfoFittingAll.csv ${foldername}/InputAllExtraInfoFittingAll${detNr}.csv
	cp ${foldername}/Detector${detNr}/paramstext.txt ${foldername}/paramstest${detNr}.txt
	cp ${foldername}/Detector${detNr}/Layer${layernr}_MultiRing_ps.txt ${foldername}/Layer${layernr}_MultiRing_ps${detNr}.txt
	cp ${foldername}/Layer${layernr}_MultiRing_ps${detNr}.txt ${paramfile}
	cp ${foldername}/paramstest${detNr}.txt ${foldername}/paramstest.txt
	cp ${foldername}/Detector${detNr}/IDsHash.csv ${foldername}/IDsHash${detNr}.csv
done

sed -i '/^Lsd /d' ${paramfile}
${BINFOLDER}/GetHKLList ${paramfile}

