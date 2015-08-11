#!/bin/bash -eu

/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

cp Spots.bin /dev/shm/
scp Spots.bin pup0100:/dev/shm
scp Spots.bin pup0101:/dev/shm
scp Spots.bin pup0102:/dev/shm
scp Spots.bin pup0103:/dev/shm
scp Spots.bin pup0104:/dev/shm
scp Spots.bin pup0105:/dev/shm
cp Data.bin /dev/shm/
scp Data.bin pup0100:/dev/shm
scp Data.bin pup0101:/dev/shm
scp Data.bin pup0102:/dev/shm
scp Data.bin pup0103:/dev/shm
scp Data.bin pup0104:/dev/shm
scp Data.bin pup0105:/dev/shm
cp nData.bin /dev/shm/
scp nData.bin pup0100:/dev/shm
scp nData.bin pup0101:/dev/shm
scp nData.bin pup0102:/dev/shm
scp nData.bin pup0103:/dev/shm
scp nData.bin pup0104:/dev/shm
scp nData.bin pup0105:/dev/shm
cp ExtraInfo.bin /dev/shm/
scp ExtraInfo.bin pup0100:/dev/shm
scp ExtraInfo.bin pup0101:/dev/shm
scp ExtraInfo.bin pup0102:/dev/shm
scp ExtraInfo.bin pup0103:/dev/shm
scp ExtraInfo.bin pup0104:/dev/shm
scp ExtraInfo.bin pup0105:/dev/shm

