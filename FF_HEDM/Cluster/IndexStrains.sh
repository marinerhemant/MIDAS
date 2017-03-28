#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
if [[ ${#*} == 2 ]]; then
	cd $2
	nlines=$( wc -l < SpotsToIndex.csv )
	echo $nlines
	if [[ $1 < $nlines ]]
	then
		echo $1
		id=$( head -n $1 SpotsToIndex.csv | tail -1 )
		echo $id
		${BINFOLDER}/IndexerLinuxArgsShm paramstest.txt $id
		${BINFOLDER}/FitPosOrStrains paramstest.txt $id
	fi
else
	${BINFOLDER}/IndexerLinuxArgsShm paramstest.txt $1
	${BINFOLDER}/FitPosOrStrains paramstest.txt $1
fi
# This was the idea with 2000 jobs
#~ echo "Chunk:"
#~ echo $1
#~ python ${PFDIR}/IndexStrains.py $1 $2
