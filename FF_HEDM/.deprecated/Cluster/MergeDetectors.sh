#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

foldername=$1
cd ${foldername}
layernr=$2
FULLPFNAME=$( cat ${foldername}/Detector1/PFNames.txt )
echo $FULLPFNAME
PFNAME="${FULLPFNAME##*/}"
paramfile=${foldername}/${PFNAME}
for (( detNr=1; detNr<=4; detNr++ ))
do
	cp ${foldername}/Detector${detNr}/InputAllExtraInfoFittingAll.csv ${foldername}/InputAllExtraInfoFittingAll${detNr}.csv
	cp ${foldername}/Detector${detNr}/paramstest.txt ${foldername}/paramstest${detNr}.txt
	cp ${foldername}/Detector${detNr}/${PFNAME} ${foldername}/Layer${layernr}_MultiRing_ps${detNr}.txt
	cp ${foldername}/Layer${layernr}_MultiRing_ps${detNr}.txt ${paramfile}
	cp ${foldername}/paramstest${detNr}.txt ${foldername}/paramstest.txt
	cp ${foldername}/Detector${detNr}/IDsHash.csv ${foldername}/IDsHash${detNr}.csv
	cp ${foldername}/Detector${detNr}/SpotsToIndex.csv ${foldername}/SpotsToIndex${detNr}.csv
done

sed -i '/^Lsd /d' ${paramfile}
${BINFOLDER}/GetHKLList ${paramfile}
/clhome/epd/bin/python ${PFDIR}/MergeDetectors.py ${paramfile}
