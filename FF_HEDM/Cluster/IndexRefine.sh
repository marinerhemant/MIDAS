#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv
fldr=$( pwd )

nNODES=${1}
export nNODES
MACHINE_NAME=$3
echo "MACHINE NAME is ${MACHINE_NAME}"
if [[ ${MACHINE_NAME} == *"edison"* ]] || [[ ${MACHINE_NAME} == *"cori"* ]]; then
	echo "We are in NERSC"
	hn=$( hostname )
	hn=${hn: -2}
	hn=$(( hn+20 ))
	intHN=128.55.143.${hn}
	export intHN
	echo intHN
fi 
${PFDIR}/SHMOperators.sh
mkdir -p Output
mkdir -p Results
mkdir -p logs
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/IndexRefine.swift \
 -folder=${fldr}
${BINFOLDER}/ProcessGrains $2
ls -lh
