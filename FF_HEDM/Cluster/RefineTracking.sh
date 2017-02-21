#!/bin/bash -eu

if [[ ${#*} != 3 ]]
then
  exit 1
fi

source ${HOME}/.MIDAS/paths

rm SpotsToIndex.csv

nNODES=${2}
if [ ${nNODES} == 7 ] && [ ${MACHINE_NAME} == 'ort' ]
then
	MACHINE_NAME="ortextra"
fi
echo "MACHINE NAME is ${MACHINE_NAME}"

${PFDIR}/SHMOperatorsTracking.sh
mkdir -p Output
mkdir -p Results
mkdir -p logs
${BINFOLDER}/GrainTracking $3 paramstest.txt
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}${1} ${PFDIR}/RefineTracking.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
