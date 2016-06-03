#!/bin/bash -eu
source ${HOME}/.bashrc
echo SWIFT is
which swift
if [[ $? == 1 ]];
then
	echo "SWIFT not found in path"
	echo "Exiting."
	exit
fi
if [[ $1 == /* ]]; then ParamsFile=$1; else ParamsFile=$(pwd)/$1; fi
StartNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${ParamsFile} )
EndNr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${ParamsFile} )
echo "Peaks:"
swift -sites.file ${PFDIR}sites.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaksMultPeaksOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Process Peaks"
swift -sites.file ${PFDIR}sites.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaksMultProcessOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
