#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths
echo "Spot ID:"
echo $1
#~ set +e
${BINFOLDER}/FitPosOrStrains paramstest.txt $1
echo $1
