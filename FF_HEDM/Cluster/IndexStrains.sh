#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
echo "Chunk:"
echo $1
python ${PFDIR}/IndexStrains.py $1 $2
