#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths
rm -rf /dev/shm/*.bin
python ${PFDIR}/runPeaks.py ${BINFOLDER} $1 $2 $3

