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
nNODES=${2}
export nNODES
MACHINE_NAME=$6
echo "MACHINE NAME is ${MACHINE_NAME}"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/RunPeaksMultPeaksOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Process Peaks"
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/RunPeaksMultProcessOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
