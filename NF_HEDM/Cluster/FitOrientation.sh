#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF
TOP_PARAM_FILE=$1
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
Micf=$(awk '$1 ~ /^MicFileBinary/ { print $2 }' ${TOP_PARAM_FILE})
MICFN=${DataDirectory}/${Micf}
echo $MICFN
echo $Micf

${BINFOLDER}/FitOrientation ${TOP_PARAM_FILE} $2 ${MICFN}
