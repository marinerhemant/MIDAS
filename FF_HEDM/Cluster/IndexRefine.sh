#!/bin/bash -eu

if [[ ${#*} != 2 ]]
then
  echo "Provide the number of CPUs to use!"
  echo "EG. ./IndexRefine.sh 320 Params.txt"
  exit 1
fi

source ${HOME}/.MIDAS/paths

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv

${PFDIR}/SHMOperators.sh $1

mkdir -p Output
mkdir -p Results
mkdir -p logs
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}${1} ${PFDIR}/IndexRefine.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
