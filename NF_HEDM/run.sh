#!/bin/bash -eu

#PATH=~wilde/swift/rev/swift-0.94.1/bin:$PATH
PATH=~wilde/swift/rev/swift-0.95-RC6/bin:$PATH
BINfolder=$6
cp $1 $4
swift -sites.file ${BINfolder}/sites${5}.xml -tc.file ${BINfolder}/tc -config ${BINfolder}/cf ${BINfolder}/FitOrientation.swift -startnr=$2 -endnr=$3 -paramfile=$1 -logdir=$4
