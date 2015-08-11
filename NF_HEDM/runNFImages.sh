#!/bin/bash -eu

if [[ ${#*} != 2 ]];
then
  echo "Usage: ./runNFImages.sh top_parameter_file(full_path) nCPUS"
  echo "Eg. ./runNFImages.sh ParametersFile.txt 384"
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
NCPUS=$2

BINfolder=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/NF_HEDM/

# Go to the right folder
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
cd ${DataDirectory}

STEM=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
CHART=/
FOLDER=${STEM%$CHART*}
mkdir -p ${FOLDER}

# Proess Images
PATH=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/SWIFT/swift-0.95-RC7/bin:$PATH
NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
NRPIXELS=$( awk '$1 ~ /^NrPixels/ { print $2 }' ${TOP_PARAM_FILE} )
echo "Median"
swift -sites.file ${BINfolder}sites${NCPUS}.xml -tc.file ${BINfolder}tc -config ${BINfolder}cf ${BINfolder}ProcessMedianParallel.swift \
  -paramfile=${TOP_PARAM_FILE} -NrLayers=${NDISTANCES} -NrFilesPerLayer=${NRFILESPERDISTANCE} -NrPixels=${NRPIXELS}
echo "Image"
swift -sites.file ${BINfolder}sites${NCPUS}.xml -tc.file ${BINfolder}tc -config ${BINfolder}cf ${BINfolder}ProcessImagesParallel.swift \
  -paramfile=${TOP_PARAM_FILE} -NrLayers=${NDISTANCES} -NrFilesPerLayer=${NRFILESPERDISTANCE} -NrPixels=${NRPIXELS}
