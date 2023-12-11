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
  echo "If FFSeedOrientations is 1, please add 4 lines to your analysis:"
  echo "				FinalGridSize, FinalEdgeLength, FullSeedFile, MinConfidenceLowerBound and MinConfidence to redo the analysis"
  echo "				using all orientations and minconfidence." 
  echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
  echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
  echo "At least the parameters file should be in the folder from where the command is executed."
  echo "For FF Seeding, add parameters FinalGridSize, FinalEdgeLength, FullSeedFile, MinConfidenceLowerBound and MinConfidence"
  echo "this will take the result and repeat analysis with all orientations." 
  echo "**********NOTE: For local runs, nNodes should be nCPUs.**********"
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
export nNODES=$4
FFSeedOrientations=$2
ProcessImages=$3
export MACHINE_NAME=$5
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
cd ${DataDirectory}
outFldr=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
grFile=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${TOP_PARAM_FILE} )
outdir=$( dirname ${outFldr} )
mkdir -p ${outdir}
rm -rf output
BinFN=$( awk '$1 ~ /^MicFileBinary/ { print $2 }' ${TOP_PARAM_FILE} )
sg=$( awk '$1 ~ /^SpaceGroup/ { print $2 }' ${TOP_PARAM_FILE} )
tmpfn=${DataDirectory}/fns.txt
echo "paramfn datadir" > ${tmpfn}
echo "${TOP_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}

rm -f ${BinFN} ${BinFN}.AllMatches
# Do Processing
export JAVA_HOME=$HOME/.MIDAS/jre1.8.0_181/
export PATH="$JAVA_HOME/bin:$PATH"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
	-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
	-DoPeakSearch=${ProcessImages} -FFSeedOrientations=${FFSeedOrientations} -DoFullLayer=1 \
	-DoGrid=1 -MachineName=${MACHINE_NAME}

${BINFOLDER}/ParseMic ${TOP_PARAM_FILE}

if [[ ${FFSeedOrientations} == 1 ]]
then
	AllOrientationsFile=$( awk '$1 ~ /^FullSeedFile/ { print $2 }' ${TOP_PARAM_FILE} )
	if [[ ${AllOrientationsFile} != '' ]]; then
		NEW_PARAM_FILE=${TOP_PARAM_FILE}_AllOrientations.txt
		cp ${TOP_PARAM_FILE} ${NEW_PARAM_FILE}
		MINCONFIDENCE=$(awk '$1 ~ /^MinConfidence/ { print $2 }' ${NEW_PARAM_FILE})
		FINALGRIDSIZE=$(awk '$1 ~ /^FinGS/ { print $2 }' ${NEW_PARAM_FILE})
		FINALEDGELENGTH=$(awk '$1 ~ /^FinEL/ { print $2 }' ${NEW_PARAM_FILE})
		LOWMINCONFIDENCE=$(awk '$1 ~ /^MinConfidenceLowerBound/ { print $2 }' ${NEW_PARAM_FILE})
		echo "Now checking all orientations for all voxels with low confidence."
		Micf=$(awk '$1 ~ /^MicFileBinary/ { print $2 }' ${NEW_PARAM_FILE})
		MicfT=$(awk '$1 ~ /^MicFileText/ { print $2 }' ${NEW_PARAM_FILE})
		sed -i "/SeedOrientations/c\SeedOrientations ${AllOrientationsFile}" ${NEW_PARAM_FILE}
		sed -i "/MinFracAccept/c\MinFracAccept 0.06" ${NEW_PARAM_FILE}
		sed -i "/MicFileBinary/c\MicFileBinary ${Micf}_AllOrientations" ${NEW_PARAM_FILE}
		sed -i "/MicFileText/c\MicFileText ${MicfT}_AllOrientations" ${NEW_PARAM_FILE}
		mv grid.txt grid_all.txt
		python ${PFDIR}/GridSorter.py ${MicfT} ${MINCONFIDENCE} ${LOWMINCONFIDENCE}
		echo "paramfn datadir" > ${tmpfn}
		echo "${NEW_PARAM_FILE} ${DataDirectory}" >> ${tmpfn}
		# Do Processing
		${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
			-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
			-DoPeakSearch=0 -FFSeedOrientations=0 -DoFullLayer=1 -DoGrid=0 -MachineName=${MACHINE_NAME}
		${BINFOLDER}/ParseMic ${NEW_PARAM_FILE}

		# Try to find the new orientations, then add them to the original Grains.csv, run everything again.
		export PYTHONPATH=${HOME}/opt/MIDAS/utils/
		python ${HOME}/opt/MIDAS/utils/findUniqueOrientationsNF.py ${MicfT}_AllOrientations ${sg} ${grFile} 2
		echo "paramfn datadir" > ${tmpfn}
		NEWPF2=${TOP_PARAM_FILE}_FinConfig.txt
		cp ${TOP_PARAM_FILE} ${NEWPF2}
		echo "${NEWPF2} ${DataDirectory}" >> ${tmpfn}
		sed -i "/GridSize/c\GridSize ${FINALGRIDSIZE}" ${NEWPF2}
		sed -i "/EdgeLength/c\EdgeLength ${FINALEDGELENGTH}" ${NEWPF2}
		sed -i "/MicFileBinary/c\MicFileBinary ${Micf}_FinResult" ${NEWPF2}
		sed -i "/MicFileText/c\MicFileText ${MicfT}_FinResult" ${NEWPF2}
		${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/processLayer.swift \
			-FileData=${tmpfn} -NrDistances=${NDISTANCES} -NrFilesPerDistance=${NRFILESPERDISTANCE} \
			-DoPeakSearch=0 -FFSeedOrientations=1 -DoFullLayer=1 -DoGrid=1 -MachineName=${MACHINE_NAME}
		${BINFOLDER}/ParseMic ${NEWPF2}
	fi
fi



EmailAdd=$6
echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
