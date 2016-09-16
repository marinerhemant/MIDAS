#!/bin/bash -eu

#PATH=~wilde/swift/rev/swift-0.94.1/bin:$PATH
PATH=~wilde/swift/rev/swift-0.95-RC6/bin:$PATH

BINfolder=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/NF_HEDM/

if [[ ${#*} != 2 ]]
then
  echo "Provide a parameters file and number of CPUs!"
  exit 1
fi

PARAMETERS_FILE=$1

NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${PARAMETERS_FILE} )
NRFILESPERLAYER=$( awk '$1 ~ /^NrFilesPerLayer/ { print $2 }' ${PARAMETERS_FILE} )

echo "PARAMETERS_FILE: ${PARAMETERS_FILE}"
echo "NDISTANCES:      ${NDISTANCES}"
echo "NRFILESPERLAYER: ${NRFILESPERLAYER}"

set -x
swift -sites.file ${BINfolder}sites${2}.xml -tc.file ${BINfolder}tc -config ${BINfolder}cf ${BINfolder}ProcessImages.swift \
  -paramfile=${PARAMETERS_FILE} \
  -NrLayers=${NDISTANCES}       \
  -NrFilesPerLayer=${NRFILESPERLAYER} 
