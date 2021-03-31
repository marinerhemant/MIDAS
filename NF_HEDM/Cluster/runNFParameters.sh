#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF

cmdname=$(basename $0)


if [[ ${#*} != 6 ]];
then
  echo "Usage: ${cmdname} parameterfile FFSeedOrientations processImages MultiGridPoints nNODEs MachineName"
  echo "Eg. ${cmdname} ParametersFile.txt 0 0 0 6 orthros"
  echo "FFSeedOrientations is when either Orientations exist already (0) or when you provide a FF Orientation file (1)."
  echo "MultiGridPoints is 0 when you just want to process one spot, otherwise if it is 1, then provide the multiple points"
  echo "in the parameter file."
  echo "processImages = 1 if you want to reduce raw files, 0 otherwise"
  echo "**********NOTE: For local runs, nNodes should be nCPUs.**********"
  exit 1
fi

hostname=$( hostname )
if [[ ${hostname} == *'orthros.xray.aps.anl.gov'* ]]; then
	echo "Exporting the correct python on orthros."
	export PATH=/clhome/TOMO1/opt/midasconda/bin:$PATH
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
NCPUS=$5
processImages=$3
FFSeedOrientations=$2
MultiGridPoints=$4
export MACHINE_NAME=$6
export nNODES=${NCPUS}
if [[ ${MACHINE_NAME} == *"edison"* ]]; then
	echo "We are in NERSC EDISON"
	hn=$( hostname )
	hn=${hn: -2}
	hn=${hn#0}
	hn=$(( hn+20 ))
	intHN=128.55.203.${hn}
	export intHN
	echo "IP address of login node: $intHN"
elif [[ ${MACHINE_NAME} == *"cori"* ]]; then
	echo "We are in NERSC CORI"
	hn=$( hostname )
	hn=${hn: -2}
	hn=${hn#0}
	hn=$(( hn+30 ))
	intHN=128.55.224.${hn}
	export intHN
	echo "IP address of login node: $intHN"
else
	intHN=10.10.10.100
	export intHN
fi
# Go to the right folder
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
cd ${DataDirectory}
outFldr=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
outdir=$( dirname ${outFldr} )
mkdir -p ${outdir}
rm -rf output

# Make hkls.csv
doHKLs=$( awk '$1 ~ /^supplyHKLs/ { print $2 } ' ${TOP_PARAM_FILE} )
if [ -z "$doHKLs" ]
then
	${BINFOLDER}/GetHKLList ${TOP_PARAM_FILE}
fi

echo "Making hexgrid."
${BINFOLDER}/MakeHexGrid $TOP_PARAM_FILE
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
if [[ ${MultiGridPoints} == 0 ]];
then
  echo "Now enter the x,y coordinates to optimize, no space, separated by a comma"
  read POS
  echo "You entered: ${POS}"
  GRIDPOINTNR=`python <<END
xy = ${POS}
x = xy[0]
y = xy[1]
grid = open('grid.txt').readline()
nrElements = int(grid)
import numpy as np
grid = np.genfromtxt('grid.txt',skip_header=1)
distt = np.power((grid[:,2]-x),2) + np.power((grid[:,3]-y),2)
idx = (distt).argmin()
print(idx+1)
END`
  echo "This is the line in the grid.txt file, which will be read: " ${GRIDPOINTNR}
fi

echo "Making diffraction spots."

GrainsFile=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${TOP_PARAM_FILE} )
SeedOrientations=$( awk '$1 ~ /^SeedOrientations/ { print $2 }' ${TOP_PARAM_FILE} )

if [[ ${FFSeedOrientations} == 1 ]];
then
    ${BINFOLDER}/GenSeedOrientationsFF2NFHEDM $GrainsFile $SeedOrientations
fi

NrOrientations=$( wc -l ${SeedOrientations} | awk '{print $1}' )

echo "NrOrientations ${NrOrientations}" >> ${TOP_PARAM_FILE}
${BINFOLDER}/MakeDiffrSpots $TOP_PARAM_FILE

if [[ ${processImages} == 1 ]];
then
  echo "Reducing images."
  NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
  NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
  NRPIXELS=$( awk '$1 ~ /^NrPixels/ { print $2 }' ${TOP_PARAM_FILE} )
  tmpfn=${DataDirectory}/fns.txt
  echo "paramfn datadir" > ${tmpfn}
  echo "${TOP_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}
  export JAVA_HOME=$HOME/.MIDAS/jre1.8.0_181/
  export PATH="$JAVA_HOME/bin:$PATH"
  ${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
    -FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
    -DoPeakSearch=${processImages} -FFSeedOrientations=${FFSeedOrientations} -DoFullLayer=0 -DoGrid=0
fi

echo "Finding parameters."
if [[ ${MultiGridPoints} == 0 ]];
then
  ${BINFOLDER}/FitOrientationParameters $TOP_PARAM_FILE ${GRIDPOINTNR}
else
  ${BINFOLDER}/FitOrientationParametersMultiPoint $TOP_PARAM_FILE
fi
