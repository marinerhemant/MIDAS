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

#if [ ${MACHINE_NAME} = "lcrc" ]
#then
	#cp Spots.bin ExtraInfo.bin /dev/shm
	#cd /dev/shm
	#hostsToCopy=$( awk '$1 ~ /^192.168/ { print $1 }' /etc/hosts)
	#echo ${hostsToCopy} | tr " " "\n" > ${HOME}/scphosts.txt
	#pscp.pssh -v -h ${HOME}/scphosts.txt Spots.bin /dev/shm/Spots.bin
	#pscp.pssh -v -h ${HOME}/scphosts.txt ExtraInfo.bin /dev/shm/ExtraInfo.bin
#fi
