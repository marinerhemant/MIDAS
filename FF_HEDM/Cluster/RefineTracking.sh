#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

rm SpotsToIndex.csv

nNODES=${1}
export nNODES
MACHINE_NAME=$4
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
${PFDIR}/SHMOperatorsTracking.sh
mkdir -p Output
mkdir -p Results
mkdir -p logs
${BINFOLDER}/GrainTracking $3 paramstest.txt
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/RefineTracking.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
