#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
tar -cvzf bins.tar.gz Spots.bin Data.bin nData.bin ExtraInfo.bin
mkdir -p ${HOME}/swiftwork/bins/
cp bins.tar.gz ${HOME}/swiftwork/bins
