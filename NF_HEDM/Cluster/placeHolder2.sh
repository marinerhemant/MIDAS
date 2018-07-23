#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
outfolder=$2
cd $outfolder
nrfiles=$1
nfiles=$( ls | grep LayersCompleted | wc -l )
while [[ $nfiles != $nrfiles ]];
do
	sleep 1
	nfiles=$( ls | grep LayersCompleted | wc -l )
	echo $nfiles
done
