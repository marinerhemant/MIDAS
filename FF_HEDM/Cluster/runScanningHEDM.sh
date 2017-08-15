#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
${BINFOLDER}/GetHKLList $1

#${PFDIR}/MergeMultipleScans.py $1
#${PFDIR}/MakeMeshGridScanning.py

nrelements=$( head -n 1 grid.txt )
GrainsFN=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${1} )
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites orthrosall \
 ${PFDIR}/processScanningHEDM.swift -ParamsFile=$1 -GrainsFile=${GrainsFN} \
 -nrelements=${nrelements}

${BINDIR}/ProcessGrainsScanningHEDM $1 ${nrelements}
