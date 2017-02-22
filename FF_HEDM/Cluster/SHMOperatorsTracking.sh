#!/bin/bash -eu
source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

# Trial option, create a bins.tar.gz, copy it over to /dev/shm of head node.
cp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
tar -cvzf bins.tar.gz Spots.bin ExtraInfo.bin
mkdir -p ${HOME}/swiftwork/bins/
cp bins.tar.gz ${HOME}/swiftwork/bins
