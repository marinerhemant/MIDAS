#!/bin/bash -eu

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

NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
tmpfn=${DataDirectory}/fns.txt
echo "paramfn datadir" > ${tmpfn}
echo "${TOP_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}

# Do Processing
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
	-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
	-DoPeakSearch=${ProcessImages} -FFSeedOrientations=${FFSeedOrientations} -DoFullLayer=1
