#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

LOCAL_DIR=$( pwd )/Cluster
CHART=/
NF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${NF_MIDAS_DIR}/bin

##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/pathsNF
echo "BINFOLDER=${BINFOLDER}/" > ${configfile}
echo "PFDIR=${LOCAL_DIR}/" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift/bin/" >> ${configfile}
ln -s ${LOCAL_DIR}/runSingleLayer.sh ${configdir}/MIDAS_V4_NearField_SingleLayer
ln -s ${LOCAL_DIR}/runMultipleLayers.sh ${configdir}/MIDAS_V4_NearField_MultipleLayers
ln -s ${LOCAL_DIR}/runNFParameters.sh ${configdir}/MIDAS_V4_NearField_Parameters

echo "Congratulations, you can now use MIDAS to run NeField analysis"
echo "Go to ${HOME}/.MIDAS folder, there is MIDAS_V4_NearField.....sh files for running analysis"
