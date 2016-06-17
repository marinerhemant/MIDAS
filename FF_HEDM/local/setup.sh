#!/bin/bash

LOCAL_DIR=$( pwd )/local
CHART=/
FF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${FF_MIDAS_DIR}/bin

##### SETUP tc.data
echo localhost indexstrains ${LOCAL_DIR}/IndexStrains.sh > ${LOCAL_DIR}/tc.data
echo localhost peaks ${LOCAL_DIR}/Peaks.sh >> ${LOCAL_DIR}/tc.data
echo localhost processPeaks ${LOCAL_DIR}/ProcessPeaks.sh >> ${LOCAL_DIR}/tc.data
echo localhost processPeaksFullImg ${LOCAL_DIR}/ProcessPeaksFullImg.sh >> ${LOCAL_DIR}/tc.data

##### Setup sites file
sitesStr='   <profile namespace="karajan" key="initialScore">10000</profile>'
nCPUs=$(cat /proc/cpuinfo | grep processor | wc -l)
isIntel=$(cat /proc/cpuinfo | grep Intel | wc -l)
if [[ $isIntel -eq 0 ]];
then
	(( nCPUs -- ))
else
	nCPUs=$(( nCPUs/2 ))
	(( nCPUs -- ))
fi
(( nCPUs -- ))
if [[ ${nCPUs} -lt 10 ]];
then
	cpustr="0.0${nCPUs}"
else
	cpustr="0.${nCPUs}"
fi

sitesStr2="   <profile namespace=\"karajan\" key=\"jobThrottle\">${cpustr}</profile>"
echo $sitesStr > fileTemp.tmp
echo $sitesStr2 >> fileTemp.tmp
sed '7r fileTemp.tmp' < ${LOCAL_DIR}/sitesTemplate.xml > ${LOCAL_DIR}/sites.xml


##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/paths
echo "BINFOLDER=${BINFOLDER}" > ${configfile}
echo "PFDIR=${LOCAL_DIR}" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift-0.95-RC6/bin" >> ${configfile}
ln -s ${LOCAL_DIR}/RealtimeAnalysisV2MultRingsPS.sh ${configdir}/MIDAS_V3_FarFieldLayers
ln -s ${LOCAL_DIR}/RealtimeAnalysisV3GrainTracking.sh ${configdir}/MIDAS_V3_FarFieldGrainTracking
ln -s ${BINFOLDER}/Calibrant ${configdir}/MIDAS_V3_FarField_Calibration
ln -s ${BINFOLDER}/FitWedge ${configdir}/MIDAS_V3_FarField_Wedge
ln -s ${BINFOLDER}/FitTiltX ${configdir}/MIDAS_V3_FarField_TiltX
chmod 700 ${LOCAL_DIR}/RealtimeAnalysisV2MultRingsPS.sh 

echo "Congratulations, you can now use MIDAS to run FarField analysis"
echo "Go to ${HOME}/.MIDAS folder, there are MIDAS_V3... files for running analysis"
