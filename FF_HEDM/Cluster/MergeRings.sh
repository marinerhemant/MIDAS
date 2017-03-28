#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
paramfile=$1 # always relative path
CHART=/
flr=${paramfile%$CHART*}
cd $flr
${BINFOLDER}/MergeMultipleRings $1
