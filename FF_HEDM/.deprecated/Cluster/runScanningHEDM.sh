#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
${BINFOLDER}/GetHKLList $1

${PFDIR}/MergeMultipleScans.py $1
${PFDIR}/MakeMeshGridScanning.py $1

export nNODES=10
export MACHINE_NAME=orthrosnew
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
outdirpath=$( awk '$1 ~ /^OutDirPath/ { print $2 }' ${1} )
origdir=$( pwd )
cd ${outdirpath}
tar -cvzf bins_${MACHINE_NAME}.tar.gz ExtraInfo.bin
cp bins_${MACHINE_NAME}.tar.gz ${HOME}/swiftwork/bins/.
cd ${origdir}
nrelements=$( wc -l < grid.txt )
GrainsFN=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${1} )

export JAVA_HOME=$HOME/.MIDAS/jre1.8.0_181/
export PATH="$JAVA_HOME/bin:$PATH"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} \
 ${PFDIR}/processScanningHEDM.swift -ParamsFile=$1 -GrainsFile=${GrainsFN} \
 -nrelements=${nrelements} -Folder=$( pwd )

${BINFOLDER}/ProcessGrainsScanningHEDM $1 ${nrelements}
python ${PFDIR}/filterGrainsScanning.py $1
