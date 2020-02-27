#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
outfolder=$3
cd $outfolder
nrfiles=$1
nfiles=$( ls | grep ImageProcessing_$2 | wc -l )
while [[ $nfiles != $nrfiles ]];
do
	sleep 1
	nfiles=$( ls | grep ImageProcessing_$2 | wc -l )
	echo $nfiles $nfiles
done
