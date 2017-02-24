#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

# Trial option, create a bins.tar.gz, copy it over to /dev/shm of head node.
tar -cvzf bins.tar.gz Spots.bin Data.bin nData.bin ExtraInfo.bin
mkdir -p ${HOME}/swiftwork/bins/
cp bins.tar.gz ${HOME}/swiftwork/bins
