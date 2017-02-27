#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv
fldr=$( pwd )

${PFDIR}/SHMOperators.sh

nNODES=${1}
export nNODES
MACHINE_NAME=$3
echo "MACHINE NAME is ${MACHINE_NAME}"

mkdir -p Output
mkdir -p Results
mkdir -p logs
${SWIFTDIR}/swift -config ${PFDIR}/sites.conf -sites ${MACHINE_NAME} ${PFDIR}/IndexRefine.swift \
 -folder=${fldr}
${BINFOLDER}/ProcessGrains $2
ls -lh
