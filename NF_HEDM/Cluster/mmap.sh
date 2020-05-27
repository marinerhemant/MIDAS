#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF

fn=$1
direct=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${fn} )
pushd ${direct}
${BINFOLDER}/MMapImageInfo ${fn}
sleep 5
tar -cvzf binsNF_${2}.tar.gz SpotsInfo.bin DiffractionSpots.bin Key.bin OrientMat.bin
mkdir -p ${HOME}/swiftwork/bins/
cp binsNF_${2}.tar.gz ${HOME}/swiftwork/bins
if [[ ${2} == *"local"* ]]; then
	cp SpotsInfo.bin DiffractionSpots.bin Key.bin OrientMat.bin /dev/shm
fi
popd
