#!/bin/bash -eu
source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

if [[ $1 == 128 ]]
then
  cp Spots.bin /dev/shm/
  scp Spots.bin puppy20:/dev/shm
  scp Spots.bin puppy21:/dev/shm
  scp Spots.bin puppy22:/dev/shm
  scp Spots.bin puppy37:/dev/shm
  scp Spots.bin puppy39:/dev/shm
  scp Spots.bin puppy41:/dev/shm
  scp Spots.bin puppy43:/dev/shm
  scp Spots.bin puppy44:/dev/shm
  cp Data.bin /dev/shm/
  scp Data.bin puppy20:/dev/shm
  scp Data.bin puppy21:/dev/shm
  scp Data.bin puppy22:/dev/shm
  scp Data.bin puppy37:/dev/shm
  scp Data.bin puppy39:/dev/shm
  scp Data.bin puppy41:/dev/shm
  scp Data.bin puppy43:/dev/shm
  scp Data.bin puppy44:/dev/shm
  cp nData.bin /dev/shm/
  scp nData.bin puppy20:/dev/shm
  scp nData.bin puppy21:/dev/shm
  scp nData.bin puppy22:/dev/shm
  scp nData.bin puppy37:/dev/shm
  scp nData.bin puppy39:/dev/shm
  scp nData.bin puppy41:/dev/shm
  scp nData.bin puppy43:/dev/shm
  scp nData.bin puppy44:/dev/shm
  cp ExtraInfo.bin /dev/shm/
  scp ExtraInfo.bin puppy20:/dev/shm
  scp ExtraInfo.bin puppy21:/dev/shm
  scp ExtraInfo.bin puppy22:/dev/shm
  scp ExtraInfo.bin puppy37:/dev/shm
  scp ExtraInfo.bin puppy39:/dev/shm
  scp ExtraInfo.bin puppy41:/dev/shm
  scp ExtraInfo.bin puppy43:/dev/shm
  scp ExtraInfo.bin puppy44:/dev/shm
  exit
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
