#!/bin/bash -eu

source ${HOME}/.bashrc

if [[ ${#*} != 2 ]]
then
  echo "Provide the number of CPUs to use!"
  echo "EG. ./IndexRefine.sh 320 Params.txt"
  exit 1
fi

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv


${PFDIR}/SHMOperators.sh

echo SWIFT is
which swift
if [[ $? == 1 ]];
then
	echo "SWIFT not found in path"
	echo "Exiting."
	exit
fi

mkdir -p Output
mkdir -p Results
mkdir -p logs
swift -sites.file ${PFDIR}sites$1.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}IndexRefine.swift
${BINFOLDER}/ProcessGrains $2
ls -lh
