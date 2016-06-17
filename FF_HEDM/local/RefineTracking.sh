#!/bin/bash -eu

if [[ ${#*} != 2 ]]
then
  exit 1
fi

source ${HOME}/.MIDAS/paths

rm SpotsToIndex.csv

${PFDIR}/SHMOperators.sh
mkdir -p Output
mkdir -p Results
mkdir -p logs
${BINFOLDER}/GrainTracking $2 paramstest.txt
${SWIFTDIR}/swift -sites.file ${PFDIR}/sites.xml -tc.file ${PFDIR}/tc.data -config ${PFDIR}/cf.local ${PFDIR}/RefineTracking.swift
${BINFOLDER}/ProcessGrains $1
ls -lh
