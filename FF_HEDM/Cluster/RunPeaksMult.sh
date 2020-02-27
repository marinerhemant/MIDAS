#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
if [[ $1 == /* ]]; then ParamsFile=$1; else ParamsFile=$(pwd)/$1; fi
StartNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${ParamsFile} )
EndNr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${ParamsFile} )
echo "Peaks:"
export nNODES=${2}
export MACHINE_NAME=$6
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
export JAVA_HOME=$HOME/.MIDAS/jre1.8.0_181/
export PATH="$JAVA_HOME/bin:$PATH"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/RunPeaksMultPeaksOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Process Peaks"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/RunPeaksMultProcessOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
