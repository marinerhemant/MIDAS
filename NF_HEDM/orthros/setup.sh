#!/bin/bash

LOCAL_DIR=$( pwd )/orthros
CHART=/
NF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${NF_MIDAS_DIR}/bin

##### SETUP tc.data
echo cluster fitorientation  ${LOCAL_DIR}/FitOrientation.sh > ${LOCAL_DIR}/tc.data
echo cluster runmedian  ${LOCAL_DIR}/MedianImage.sh > ${LOCAL_DIR}/tc.data
echo cluster runimageprocessing  ${LOCAL_DIR}/ImageProcessing.sh > ${LOCAL_DIR}/tc.data
echo cluster runconvertfiles ${LOCAL_DIR}/ConvertFiles.sh > ${LOCAL_DIR}/tc.data
echo cluster runmedianparallel  ${LOCAL_DIR}/MedianImageParallel.sh > ${LOCAL_DIR}/tc.data
echo cluster runimageprocessingparallel  ${LOCAL_DIR}/ImageProcessingParallel.sh > ${LOCAL_DIR}/tc.data

##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/pathsNF
echo "BINFOLDER=${BINFOLDER}/" > ${configfile}
echo "PFDIR=${LOCAL_DIR}/" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift-0.95-RC6/bin/" >> ${configfile}
ln -s ${LOCAL_DIR}/runSingleLayer.sh ${configdir}/MIDAS_V3_NearFieldSingleLayer.sh
ln -s ${LOCAL_DIR}/runNFParameters.sh ${configdir}/MIDAS_V3_NearFieldParameters.sh

echo "Congratulations, you can now use MIDAS to run NeField analysis"
echo "Go to ${HOME}/.MIDAS folder, there is MIDAS_V3_NearField.....sh files for running analysis"
