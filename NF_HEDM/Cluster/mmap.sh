#!/bin/bash

source ${HOME}/.MIDAS/pathsNF

fn=$1
direct=$2
${BINFOLDER}/MMapImageInfo ${fn}
pushd ${direct}
tar -cvzf binsNF.tar.gz SpotsInfo.bin DiffractionSpots.bin Key.bin OrientMat.bin
mkdir -p ${HOME}/swiftwork/bins/
cp binsNF.tar.gz ${HOME}/swiftwork/bins
popd
