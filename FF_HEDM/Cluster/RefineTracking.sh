#!/bin/bash -eu

if [[ ${#*} != 3 ]]
then
  exit 1
fi

source ${HOME}/.MIDAS/paths

rm SpotsToIndex.csv

${PFDIR}/SHMOperatorsTracking.sh $1
mkdir -p Output
mkdir -p Results
mkdir -p logs
${BINFOLDER}/GrainTracking $3 paramstest.txt
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}_shm${1},${MACHINE_NAME}_noshm${1} ${PFDIR}/RefineTracking.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
