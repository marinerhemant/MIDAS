#!/bin/bash -eu

if [[ ${#*} != 2 ]]
then
  echo "Provide the number of CPUs to use!"
  echo "EG. ./IndexRefine.sh 320 Params.txt"
  exit 1
fi

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv
INDEXERSTRAINS=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/
${INDEXERSTRAINS}/SHMOperators.sh
# PATH=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/SWIFT/swift-0.95-RC7/bin:$PATH
PATH=~wilde/swift/rev/swift-0.95-RC6/bin:$PATH
mkdir -p Output
mkdir -p Results
mkdir -p logs
swift -sites.file ${INDEXERSTRAINS}sites$1.xml -tc.file ${INDEXERSTRAINS}tc.data -config ${INDEXERSTRAINS}cf ${INDEXERSTRAINS}IndexRefine.swift
${INDEXERSTRAINS}bin/ProcessGrains $2
ls -lh
