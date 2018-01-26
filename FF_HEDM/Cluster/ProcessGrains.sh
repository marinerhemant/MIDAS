#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

cd $1
${BINFOLDER}/ProcessGrains $2
ls -lh
rm -fv /dev/shm/*.bin
