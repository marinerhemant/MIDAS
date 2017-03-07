#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF

fn=$1
direct=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${fn} )
${BINFOLDER}/MMapImageInfo ${fn}
pushd ${direct}
tar -cvzf binsNF.tar.gz SpotsInfo.bin DiffractionSpots.bin Key.bin OrientMat.bin
mkdir -p ${HOME}/swiftwork/bins/
cp binsNF.tar.gz ${HOME}/swiftwork/bins
popd
