#!/bin/bash -eu

if [[ ${#*} != 5 ]]
then
  echo "Provide a parameters file and the number of CPUs to use!"
  echo "EG. ./RunPeaks.sh parameters.txt 320"
  exit 1
fi

if [[ $1 == /* ]]; then ParamsFile=$1; else ParamsFile=$(pwd)/$1; fi



StartNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${ParamsFile} )
EndNr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${ParamsFile} )
# RingNr=$( awk '$1 ~ /^RingNumbers/ { print $2 }' ${ParamsFile} )
# ImTransOpt=$( awk '$1 ~ /^ImTransOpt/ { print $2 " " $3 " " $4 " " $5 " " $6 " " $7 " " $8 " " $9 " "}' ${ParamsFile} )
PFDIR=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/
# PATH=~wilde/swift/rev/swift-0.94.1/bin:$PATH
PATH=~wilde/swift/rev/swift-0.95-RC6/bin:$PATH
# PATH=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/SWIFT/swift-0.95-RC7/bin:$PATH
# swift -sites.file ${PFDIR}sites$2.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}trial.swift
#swift -sites.file ${PFDIR}sites$2.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaksMult.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Peaks:"
swift -sites.file ${PFDIR}sites$2.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaksMultPeaksOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}
echo "Process Peaks"
swift -sites.file ${PFDIR}sites$2.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}RunPeaksMultProcessOnly.swift -paramsfile=$4 -ringfile=$3 -fstm=$5 -startnr=${StartNr} -endnr=${EndNr}

