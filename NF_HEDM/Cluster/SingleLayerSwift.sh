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
  echo "If FFSeedOrientations is 1, please add 2 lines to your analysis: FullSeedFile and MinConfidence to redo the analysis"
  echo "using all orientations and minconfidence." 
  echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
  echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
  echo "At least the parameters file should be in the folder from where the command is executed."
  echo "For FF Seeding, add parameters MinConfidence and MinConfidenceLowerBound and FullSeedFile, this will"
  echo "take the result and repeat analysis with all orientations." 
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
nNODES=$4
FFSeedOrientations=$2
ProcessImages=$3
MACHINE_NAME=$5
export nNODES
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
NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
tmpfn=${DataDirectory}/fns.txt
echo "paramfn datadir" > ${tmpfn}
echo "${TOP_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}

# Do Processing
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
	-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
	-DoPeakSearch=${ProcessImages} -FFSeedOrientations=${FFSeedOrientations} -DoFullLayer=1 \
	-DoGrid=1

${BINFOLDER}/ParseMic ${TOP_PARAM_FILE}

if [[ ${FFSeedOrientations} == 1 ]]
then
	NEW_PARAM_FILE=${TOP_PARAM_FILE}_AllOrientations.txt
	cp ${TOP_PARAM_FILE} ${NEW_PARAM_FILE}
	AllOrientationsFile=$( awk '$1 ~ /^FullSeedFile/ { print $2 }' ${NEW_PARAM_FILE} )
	if [[ ${AllOrientationsFile} != '' ]]; then
		MINCONFIDENCE=$(awk '$1 ~ /^MinConfidence/ { print $2 }' ${NEW_PARAM_FILE})
		LOWMINCONFIDENCE=$(awk '$1 ~ /^MinConfidenceLowerBound/ { print $2 }' ${NEW_PARAM_FILE})
		echo "Now checking all orientations for all voxels with low confidence."
		Micf=$(awk '$1 ~ /^MicFileBinary/ { print $2 }' ${NEW_PARAM_FILE})
		MicfT=$(awk '$1 ~ /^MicFileText/ { print $2 }' ${NEW_PARAM_FILE})
		sed -i "/SeedOrientations/c\SeedOrientations ${AllOrientationsFile}" ${NEW_PARAM_FILE}
		sed -i "/MinFracAccept/c\MinFracAccept 0.03" ${NEW_PARAM_FILE}
		sed -i "/MicFileBinary/c\MicFileBinary ${Micf}_AllOrientations" ${NEW_PARAM_FILE}
		sed -i "/MicFileText/c\MicFileText ${MicfT}_AllOrientations" ${NEW_PARAM_FILE}
		mv grid.txt grid_all.txt
		python ${PFDIR}/GridSorter.py ${MicfT} ${MINCONFIDENCE} ${LOWMINCONFIDENCE}
		echo "paramfn datadir" > ${tmpfn}
		echo "${NEW_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}
		# Do Processing
		${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
			-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
			-DoPeakSearch=0 -FFSeedOrientations=0 -DoFullLayer=1 -DoGrid=0
		${BINFOLDER}/ParseMic ${NEW_PARAM_FILE}
	fi
fi

EmailAdd=$6
echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
