#!/bin/bash

LOCAL_DIR=$( pwd )/Orthros
CHART=/
FF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${FF_MIDAS_DIR}/bin

##### SETUP tc.data
echo cluster indexstrains ${LOCAL_DIR}/IndexStrains.sh > ${LOCAL_DIR}/tc.data
echo cluster strainsrefine ${LOCAL_DIR}/StrainsRefine.sh >> ${LOCAL_DIR}/tc.data
echo cluster peaks ${LOCAL_DIR}/Peaks.sh >> ${LOCAL_DIR}/tc.data
echo cluster processPeaks ${LOCAL_DIR}/ProcessPeaks.sh >> ${LOCAL_DIR}/tc.data
echo localhost processPeaksFullImg ${LOCAL_DIR}/ProcessPeaksFullImg.sh >> ${LOCAL_DIR}/tc.data

##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/paths
echo "BINFOLDER=${BINFOLDER}" > ${configfile}
echo "PFDIR=${LOCAL_DIR}" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift-0.95-RC6/bin" >> ${configfile}
ln -s ${LOCAL_DIR}/RealtimeAnalysisV2MultRingsPS.sh ${configdir}/MIDAS_V3_FarFieldLayers
ln -s ${LOCAL_DIR}/RealtimeAnalysisV3GrainTracking.sh ${configdir}/MIDAS_V3_FarFieldGrainTracking
ln -s ${LOCAL_DIR}/TrackGrains.sh ${configdir}/MIDAS_V3_FarFieldTrackGrains
ln -s ${BINFOLDER}/Calibrant ${configdir}/MIDAS_V3_FarField_Calibration
ln -s ${BINFOLDER}/FitWedge ${configdir}/MIDAS_V3_FarField_Wedge
ln -s ${BINFOLDER}/FitTiltX ${configdir}/MIDAS_V3_FarField_TiltX

echo "Congratulations, you can now use MIDAS to run FarField analysis"
echo "Go to ${HOME}/.MIDAS folder, there are MIDAS_V3... files for running analysis"
