#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

outfolder=$1
pushd ${outfolder}
${PFDIR}/SHM.sh
if [ $2 = "hydra" ]; then
	tar -cvzf bins_${4}.tar.gz Spots.bin Data.bin nData.bin ExtraInfo.bin BigDetectorMask.bin
else
	tar -cvzf bins_${4}.tar.gz Spots.bin Data.bin nData.bin ExtraInfo.bin
fi
mkdir -p ${HOME}/swiftwork/bins/
cp bins_${4}.tar.gz ${HOME}/swiftwork/bins
mkdir -p Output
mkdir -p Results
popd
cp ${outfolder}/SpotsToIndex.csv $3
