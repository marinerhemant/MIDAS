#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

cmdname=$(basename $0)

if [[ ${#*} != 6 ]]
then
  echo "Provide ParametersFile StartLayerNr EndLayerNr Number of NODEs to use MachineName and EmailAddress!"
  echo "EG. ${cmdname} Parameters.txt 1 1 6 orhtros(or orthrosextra) hsharma@anl.gov"
  echo "The parameter file should have a parameter called OldStateFolder which is the seed folder used in the previous state."
  echo "MinNrSpots must be 1!!!!!"
  exit 1
fi

# Create OldFolder for a layer, submit a single layer job for RealTimeAnalysisV3GrainTracking.sh

STARTLAYERNR=$2
ENDLAYERNR=$3
if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
OldStateFolder=$( awk '$1 ~ /^OldStateFolder/ { print $2 }' ${TOP_PARAM_FILE} )

for ((LAYERNR=$STARTLAYERNR; LAYERNR<=$ENDLAYERNR; LAYERNR++))
do
	cd ${OldStateFolder}
	OldFolder=$( find -type d -name "*Layer${LAYERNR}_*" )
	OldFolder=${OldFolder:1}
	OldFolder=${OldStateFolder}/${OldFolder}
	PSThisLayer=${TOP_PARAM_FILE}.Layer${LAYERNR}.txt
	cp ${TOP_PARAM_FILE} ${PSThisLayer}
	echo OldFolder ${OldFolder} >> ${PSThisLayer}
	${PFDIR}/RealtimeAnalysisGrainTracking.sh ${PSThisLayer} ${LAYERNR} ${LAYERNR} $4 $5
done

EmailAdd=$6
echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
