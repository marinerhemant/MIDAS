#!/bin/bash -eu

if [[ ${#*} != 2 ]]
then
  echo "Provide the number of Nodes to use!"
  echo "EG. ./IndexRefine.sh 6 Params.txt"
  exit 1
fi

source ${HOME}/.MIDAS/paths

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv

${PFDIR}/SHMOperators.sh

nNODES=${2}
if [ ${nNODES} == 7 ] && [ ${MACHINE_NAME} == 'ort' ]
then
	MACHINE_NAME="ortextra"
fi
echo "MACHINE NAME is ${MACHINE_NAME}"

mkdir -p Output
mkdir -p Results
mkdir -p logs
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}${1} ${PFDIR}/IndexRefine.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
