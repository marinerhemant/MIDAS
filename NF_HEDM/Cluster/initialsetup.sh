#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF

TOP_PARAM_FILE=$1
FFSeedOrientations=$2
DoGrid=$3
# Go to the right folder and make right output folder
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
cd ${DataDirectory}
STEM=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
CHART=/
FOLDER=${STEM%$CHART*}
mkdir -p ${FOLDER}

hostname=$( hostname )
if [[ ${hostname} == *'orthros.xray.aps.anl.gov'* ]]; then
	echo "Exporting the correct python on orthros."
	export PATH=/clhome/TOMO1/opt/midasconda/bin:$PATH
fi

# Make hkls.csv
doHKLs=$( awk '$1 ~ /^supplyHKLs/ { print $2 } ' ${TOP_PARAM_FILE} )
echo $doHKLs
if [ -z "$doHKLs" ]
then
	echo "We will generate hkls."
	${BINFOLDER}/GetHKLList ${TOP_PARAM_FILE}
else
	echo "We did not calculate HKLs, they were supplied."
fi

# Generate SeedOrientations
SeedOrientations=$( awk '$1 ~ /^SeedOrientations/ { print $2 }' ${TOP_PARAM_FILE} )
if [ ${FFSeedOrientations} == 1 ];
then
  GrainsFile=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${TOP_PARAM_FILE} )
  MINCONFIDENCE=$(awk '$1 ~ /^MinConfidence/ { print $2 }' ${TOP_PARAM_FILE})
  ${BINFOLDER}/GenSeedOrientationsFF2NFHEDM $GrainsFile $SeedOrientations
fi
NrOrientations=$( wc -l ${SeedOrientations} | awk '{print $1}' )
echo "NrOrientations ${NrOrientations}" >> ${TOP_PARAM_FILE}

# Make HexGrid and DiffractionSpots
if [ ${DoGrid} == 1 ];
then
  ${BINFOLDER}/MakeHexGrid $TOP_PARAM_FILE

  # if tomo file was supplied
  if [[ -n $( awk '$1 ~ /^TomoImage/ { print }' ${TOP_PARAM_FILE} ) ]];
  then
    tomoPxSize=$( awk '$1 ~ /^TomoPixelSize/ { print $2 }' ${TOP_PARAM_FILE} )
    TomoFN=$( awk '$1 ~ /^TomoImage/ { print $2 }' ${TOP_PARAM_FILE} )
    echo ${tomoPxSize}
    echo ${TomoFN}
    ${BINFOLDER}/filterGridfromTomo ${TomoFN} ${tomoPxSize}
    mv grid.txt grid_unfilt.txt
    mv gridNew.txt grid.txt
  # Filter HexGrid
  else
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
  fi
fi

# Make diffraction spots
${BINFOLDER}/MakeDiffrSpots $TOP_PARAM_FILE
