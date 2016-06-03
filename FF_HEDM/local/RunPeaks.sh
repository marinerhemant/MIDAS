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
RingNr=$( awk '$1 ~ /^RingNumbers/ { print $2 }' ${ParamsFile} )
swift -sites.file ${PFDIR}sites.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaks.swift -paramsfile=${ParamsFile} -ringnr=${RingNr} -startnr=${StartNr} -endnr=${EndNr}
