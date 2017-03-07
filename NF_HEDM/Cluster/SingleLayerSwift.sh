#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF
cmdname=$(basename $0)

if [[ ${#*} != 6 ]];
then
  echo "Usage: ${cmdname} top_parameter_file(full_path) FFSeedOrientations ProcessImages nNODEs MachineName EmailAddress"
  echo "Eg. ${cmdname} ParametersFile.txt 1(or 0) 1(or 0) 6 orthros hsharma@anl.gov"
  echo "This will produce the output in the run folder, called Microstructure.mic."
  echo "FFSeedOrientations is when either Orientations exist already (0) or when you provide a FF Orientation file (1)."
  echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
  echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
  echo "At least the parameters file should be in the folder from where the command is executed."
  echo "For FF Seeding, add a parameter MinConfidence and FullSeedFile, this will take the result and repeat analysis"
  echo "with all orientations." 
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
NCPUS=$4
FFSeedOrientations=$2
ProcessImages=$3
MACHINE_NAME=$5
nNODES=${NCPUS}
export nNODES

# Go to the right folder
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
cd ${DataDirectory}

STEM=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
CHART=/
FOLDER=${STEM%$CHART*}
mkdir -p ${FOLDER}

# Make hkls.csv
${BINFOLDER}/GetHKLList ${TOP_PARAM_FILE}

SeedOrientations=$( awk '$1 ~ /^SeedOrientations/ { print $2 }' ${TOP_PARAM_FILE} )
Micf=$(awk '$1 ~ /^MicFileBinary/ { print $2 }' ${TOP_PARAM_FILE})
MicfT=$(awk '$1 ~ /^MicFileText/ { print $2 }' ${TOP_PARAM_FILE})
MICFN=${DataDirectory}/${Micf}

if [[ -n $MICFN ]];
then 
  rm -rf $MICFN
fi

# Generate SeedOrientations
if [ ${FFSeedOrientations} == 1 ];
then
  GrainsFile=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${TOP_PARAM_FILE} )
  MINCONFIDENCE=$(awk '$1 ~ /^MinConfidence/ { print $2 }' ${TOP_PARAM_FILE})
  ${BINFOLDER}/GenSeedOrientationsFF2NFHEDM $GrainsFile $SeedOrientations
fi

NrOrientations=$( wc -l ${SeedOrientations} | awk '{print $1}' )
echo "NrOrientations ${NrOrientations}" >> ${TOP_PARAM_FILE}

# Make HexGrid and DiffractionSpots
${BINFOLDER}/MakeHexGrid $TOP_PARAM_FILE

# Filter HexGrid
if [[ -n $( awk '$1 ~ /^GridMask/ { print }' ${TOP_PARAM_FILE} ) ]];
then
  Xmin=$( awk '$1 ~ /^GridMask/ { print $2 }' ${TOP_PARAM_FILE} )
  Xmax=$( awk '$1 ~ /^GridMask/ { print $3 }' ${TOP_PARAM_FILE} )
  Ymin=$( awk '$1 ~ /^GridMask/ { print $4 }' ${TOP_PARAM_FILE} )
  Ymax=$( awk '$1 ~ /^GridMask/ { print $5 }' ${TOP_PARAM_FILE} )
  awk ' { if (($3 >= '$Xmin') && ($3 <= '$Xmax') && ($4 >= '$Ymin') && ($4 <= '$Ymax'))  print }' < grid.txt >grid_new.txt
  mv grid.txt grid_old.txt
  wc -l <grid_new.txt >grid.txt
  cat grid_new.txt >>grid.txt
fi

${BINFOLDER}/MakeDiffrSpots $TOP_PARAM_FILE

# Calculate nGridPoints
nGridPoints=$( wc -l grid.txt | awk '{print $1}' )
(( nGridPoints = nGridPoints - 1 ))
STARTNR=1
ENDNR=2000 #${nGridPoints}
NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
NRPIXELS=$( awk '$1 ~ /^NrPixels/ { print $2 }' ${TOP_PARAM_FILE} )

# Do Processing
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
	-StartNumber=${STARTNR} -EndNumber=${ENDNR} -ParameterFileName=${TOP_PARAM_FILE} -MicFileName=${MICFN} \
	-NrLayers=${NDISTANCES} -NrFilesPerLayer=${NRFILESPERDISTANCE} -DoPeakSearch=${ProcessImages}

# Parse Mic file
${BINFOLDER}/ParseMic ${TOP_PARAM_FILE}
