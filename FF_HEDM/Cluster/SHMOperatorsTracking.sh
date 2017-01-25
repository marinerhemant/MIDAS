#!/bin/bash -eu
source ${HOME}/.MIDAS/paths
${PFDIR}/SHM.sh
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

scp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
# Trial option, create a bins.tar.gz, copy it over to /dev/shm of head node.
tar -cvf bins.tar.gz Spots.bin ExtraInfo.bin
cp bins.tar.gz /dev/shm

return

if [[ $1 == 128 ]]
then
  scp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
  scp Spots.bin ExtraInfo.bin puppy21:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy22:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy37:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy39:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy41:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy43:/dev/shm
  scp Spots.bin ExtraInfo.bin puppy44:/dev/shm
elif [[ $1 == 64 ]]
then
	scp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
	scp Spots.bin ExtraInfo.bin pup0100:/dev/shm
elif [[ $1 == 320 ]]
then
	scp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
	scp Spots.bin ExtraInfo.bin pup0101:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0102:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0103:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0104:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0105:/dev/shm
elif [[ $1 == 384 ]]
then
	scp Spots.bin Data.bin nData.bin ExtraInfo.bin /dev/shm
	scp Spots.bin ExtraInfo.bin pup0100:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0101:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0102:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0103:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0104:/dev/shm
	scp Spots.bin ExtraInfo.bin pup0105:/dev/shm
fi
