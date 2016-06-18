#!/bin/bash

if [[ ${#*} != 4 ]]
then
  echo "Provide ParametersFile StartLayerNr EndLayerNr and the number of CPUs to use!"
  echo "EG. $0 Parameters.txt 1 1 320"
  echo "The parameter file should have a parameter called OldStateFolder which is the seed folder used in the previous state."
  exit 1
fi

# Create OldFolder for a layer, submit a single layer job for RealTimeAnalysisV3GrainTracking.sh

STARTLAYERNR=$2
ENDLAYERNR=$3
if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
OldStateFolder=$( awk '$1 ~ /^OldStateFolder/ { print $2 }' ${TOP_PARAM_FILE} )

for ((LAYERNR=$STARTLAYERNR; LAYERNR<=$ENDLAYERNR; LAYERNR++))
do
	OldFolder=$( ${OldStateFolder}/*Layer${LayerNr}_* )
	echo $OldFolder
	PSThisLayer=${TOP_PARAM_FILE}.Layer${LAYERNR}.txt
	echo ${OldFolder} >> ${PSThisLayer}
	echo ${HOME}/.MIDAS/MIDAS_V3_FarFieldGrainTracking PSThisLayer ${LAYERNR} ${LAYERNR} $4
done
