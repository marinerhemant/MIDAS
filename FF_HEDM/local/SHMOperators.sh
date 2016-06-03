#!/bin/bash -eu
source ${HOME}/.bashrc

${PFDIR}/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

cp Spots.bin /dev/shm/
cp Data.bin /dev/shm/
cp nData.bin /dev/shm/
cp ExtraInfo.bin /dev/shm/
