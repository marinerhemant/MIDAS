#!/bin/bash

LOCAL_DIR=$( pwd )/Cluster
CHART=/
FF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${FF_MIDAS_DIR}/bin

##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/paths
echo "BINFOLDER=${BINFOLDER}" > ${configfile}
echo "PFDIR=${LOCAL_DIR}" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift/bin" >> ${configfile}
echo "MACHINE_NAME=${1}" >> ${configfile}
ln -s ${LOCAL_DIR}/RealtimeAnalysisMultRingsPS.sh ${configdir}/MIDAS_V4_FarField_Layers
ln -s ${LOCAL_DIR}/TrackGrains.sh ${configdir}/MIDAS_V4_FarField_TrackGrains
ln -s ${BINFOLDER}/Calibrant ${configdir}/MIDAS_V4_FarField_Calibration
ln -s ${BINFOLDER}/FitWedge ${configdir}/MIDAS_V4_FarField_Wedge
ln -s ${BINFOLDER}/FitTiltX ${configdir}/MIDAS_V4_FarField_TiltX
ln -s ${BINFOLDER}/ProcessGrains ${configdir}/MIDAS_V4_FarField_ProcessGrains
ln -s ${BINFOLDER}/FitGrain ${configdir}/MIDAS_V4_FarField_FitGrain

echo "Congratulations, you can now use MIDAS to run FarField analysis"
echo "Go to ${HOME}/.MIDAS folder, there are MIDAS_V3... files for running analysis"
