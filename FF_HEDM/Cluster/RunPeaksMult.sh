#!/bin/bash -eux
source ${HOME}/.MIDAS/paths
if [[ $1 == /* ]]; then ParamsFile=$1; else ParamsFile=$(pwd)/$1; fi
StartNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${ParamsFile} )
EndNr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${ParamsFile} )
echo "Peaks:"
echo ${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}_noshm${2} ${PFDIR}/RunPeaksMultPeaksOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Process Peaks"
echo ${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME}_noshm${2} ${PFDIR}/RunPeaksMultProcessOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
lskdjfklshjklfdhklsfhkljhdf
