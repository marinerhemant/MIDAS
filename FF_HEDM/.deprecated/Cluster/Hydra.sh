#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

cmdname=$(basename $0)

echo "Analysis code for Hydra:"
echo "Version: 1, 2017/03/30, in case of problems contact hsharma@anl.gov"

if [[ ${#*} != 7 ]]
then
	echo "Provide ParametersFile StartLayerNr EndLayerNr DoPeakSearch Number of NODES to use MachineName EmailAddress!"
	echo "EG. ${cmdname} parameters.txt 1 1 1 (or 0) 6 orthros(orthrosextra,local,rice) hsharma@anl.gov"
	echo "the source parameter file MUST not have ring numbers and layer numbers in it."
	echo "Parameter file name MUST not be full path, analysis should be run from the directory where the parameter file is."
	echo "********SeedFolder MUST NOT BE IN THE PARAMETER FILE.**********"
	echo "**********NOTE: For local runs, nNodes should be nCPUs.**********"
	exit 1
fi

ParamsFile=$1
STARTLAYERNR=$2
ENDLAYERNR=$3
DOPEAKSEARCH=$4
NCPUS=$5
export MACHINE_NAME=$6
EM=$7
StartNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${ParamsFile} )
EndNr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${ParamsFile} )
echo "Peaks:"
export nNODES=${NCPUS}
echo "MACHINE NAME is ${MACHINE_NAME}"
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
filestem=$( awk '$1 ~ /^FileStem/ { print $2 } ' ${ParamsFile} )
seedfolder=$( pwd ) # Call from the seed folder always
rm -rf ${seedfolder}/FolderNames.txt
# Make BigMask
echo "Creating Detector Mask"
${BINFOLDER}/MapMultipleDetectors ${ParamsFile}
for (( LAYERNR=$STARTLAYERNR; LAYERNR<=$ENDLAYERNR; LAYERNR++ ))
do
	ide=$( date +%Y_%m_%d_%H_%M_%S )
	outfolder=${seedfolder}/${filestem}_Layer${LAYERNR}_Analysis_Time_${ide}
	echo "${outfolder}" >> ${seedfolder}/FolderNames.txt
	rm -rf ${outfolder}/PFNames.txt
	for (( DETNR=1; DETNR<=4; DETNR++ ))
	do
		${PFDIR}/InitialSetupHydra.sh $ParamsFile $LAYERNR $DETNR $outfolder
		if [[ $? != 0 ]]; then
			exit 1
		fi
	done
	cd $outfolder
	python ${PFDIR}/prepareFilesHydra.py $ParamsFile 1 # This appends stuff to the param files, this is called only to do an initial setup!!
	cd ../
	export JAVA_HOME=$HOME/.MIDAS/jre1.8.0_181/
	export PATH="$JAVA_HOME/bin:$PATH"
	${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} \
		${PFDIR}/processLayersHydra.swift -ringfile=${seedfolder}/RingInfo.txt \
		-startnr=${StartNr} -endnr=${EndNr} -SeedFolder=${seedfolder} \
		-startLayer=${STARTLAYERNR} -endLayer=${ENDLAYERNR} -DoPeakSearch=${DOPEAKSEARCH}
	cd ${outfolder}
	pfname=${ParamsFile}
	${BINFOLDER}/ProcessGrains ${pfname}
	cd ../
done

