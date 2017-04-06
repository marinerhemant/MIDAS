#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

outfolder=$1
pushd ${outfolder}
${PFDIR}/SHMOperators.sh
mkdir -p Output
mkdir -p Results
popd
cp ${outfolder}/SpotsToIndex.csv $3
