#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
FOLDERNAME=$2
for (( detNr=1; detNr<=4; detNr++ ))
do
	date
	python ${PFDIR}/runPeaks.py ${BINFOLDER} $1 ${FOLDERNAME}/Detector${detNr}/ParamFileNames.txt $3
done
date
