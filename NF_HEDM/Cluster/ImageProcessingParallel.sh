#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.MIDAS/LIBTIFF/lib/
export LD_LIBRARY_PATH

${BINFOLDER}/ImageProcessingLibTiff $1 $2 $3
