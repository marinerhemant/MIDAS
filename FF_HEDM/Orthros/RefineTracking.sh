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
${SWIFTDIR}/swift -sites.file ${PFDIR}/sites$1.xml -tc.file ${PFDIR}/tc.data -config ${PFDIR}/cf.local ${PFDIR}/RefineTracking.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
