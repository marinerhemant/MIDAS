#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
tar -cvzf bins_${MACHINE_NAME}.tar.gz Spots.bin ExtraInfo.bin
cp *.bin /dev/shm
mkdir -p ${HOME}/swiftwork/bins/
cp bins_${MACHINE_NAME}.tar.gz ${HOME}/swiftwork/bins
