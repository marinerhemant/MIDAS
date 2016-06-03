#!/bin/bash

source ${HOME}/.bashrc

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
cat ${HOME}/.bashrc | grep -q BINFOLDER
if [[ $? == 1 ]];
then
	echo "BINFOLDER=${BINFOLDER}" >> ${HOME}/.bashrc;
	echo "PFDIR=${LOCAL_DIR}" >> ${HOME}/.bashrc;
	echo "alias FFSingleLayer=${LOCAL_DIR}/RealtimeAnalysisV2MultRingsPS.sh" >> ${HOME}/.bashrc;
	chmod 700 ${LOCAL_DIR}/RealtimeAnalysisV2MultRingsPS.sh 
fi

echo "Congratulations, you can now use FFSingleLayer to run analysis"
echo "First type:"
echo "source ~/.bashrc"
echo "Or restart terminal"
echo "and check what happens when you type"
echo "which swift"
