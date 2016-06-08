#!/bin/bash

source ${HOME}/.bashrc

LOCAL_DIR=$( pwd )/local
CHART=/
FF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${FF_MIDAS_DIR}/bin

##### SETUP tc.data
echo cluster indexstrains ${LOCAL_DIR}/IndexStrains.sh > ${LOCAL_DIR}/tc.data
echo cluster peaks ${LOCAL_DIR}/Peaks.sh >> ${LOCAL_DIR}/tc.data
echo cluster processPeaks ${LOCAL_DIR}/ProcessPeaks.sh >> ${LOCAL_DIR}/tc.data
echo localhost processPeaksFullImg ${LOCAL_DIR}/ProcessPeaksFullImg.sh >> ${LOCAL_DIR}/tc.data

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
