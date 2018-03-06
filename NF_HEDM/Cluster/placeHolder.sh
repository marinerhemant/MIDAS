#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
outfolder=$2
cd $outfolder
nrfiles=$1
echo $( pwd )
echo $nrfiles
nfiles=$( ls output | grep ImageProcessing | wc -l )
while [[ $nfiles != $nrfiles ]];
do
	sleep 1
	nfiles=$( ls output | grep ImageProcessing | wc -l )
	echo $nfiles
done
