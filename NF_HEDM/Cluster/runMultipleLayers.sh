#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF
cmdname=$(basename $0)

if [[ ${#*} != 7 ]];
then
  echo "Usage: ${cmdname} top_parameter_file(full_path) FFSeedOrientations ProcessImages startLayerNr endLayerNr nNODEs EmailAddress"
  echo "Eg. ${cmdname} ParametersFile.txt 1(or 0) 1(or 0) 1 5 6 hsharma@anl.gov"
  echo "This will run from layer 1 to 5."
  echo "This will produce the output in the run folder."
  echo "FFSeedOrientations is when either Orientations exist already (0) or when you provide a FF Orientation file (1)."
  echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
  echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
  echo "At least the parameters file should be in the folder from where the command is executed."
  echo "The following parameters should NOT be in the ParametersFile.txt file:"
  echo "RawStartNr, GlobalPosition, MicFileBinary, MicFileText, ReducedFileName, GrainsFile, SeedOrientations"
  echo "The following parameters must be present:"  
  echo "OrigFileName, OverallStartNr, GlobalPositionFirstLayer, LayerThickness, WFImages, nDistances, NrFilesPerDistance, DataDirectory"
  echo "If WF images were taken, it is assumed there were 10 WF images."  
  echo "If using FF seeding, the grains files should be in the same folder and named: GrainsLayer{LAYERNR}.csv"
  echo "If no seeding is used, a single Orientations.txt file should be present."
  echo "At successfull completion, it will send an email to EmailAddress."
  exit 1
fi

PARAMFILE=$1
NCPUS=$6
FFSEEDORIENTATIONS=$2
PROCESSIMAGES=$3
STARTLAYERNR=$4
ENDLAYERNR=$5
STEM=$( awk '$1 ~ /^OrigFileName/ { print $2 }' ${PARAMFILE} )
CHART=/
FOLDER=${STEM%$CHART*}
FILENAME=${STEM#*$CHART}
OVERALLSTARTNR=$( awk '$1 ~ /^OverallStartNr/ { print $2 }' ${PARAMFILE} )
STARTGLOBALPOS=$( awk '$1 ~ /^GlobalPositionFirstLayer/ { print $2 }' ${PARAMFILE} )
LAYERTHICKNESS=$( awk '$1 ~ /^LayerThickness/ { print $2 }' ${PARAMFILE} )
PFSTEM=${PARAMFILE%.*}
WFIMAGES=$( awk '$1 ~ /^WFImages/ { print $2 }' ${PARAMFILE} )
NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${PARAMFILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${PARAMFILE} )
DATADIRECTORY=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${PARAMFILE} )

for ((LAYERNR=${STARTLAYERNR}; LAYERNR<=${ENDLAYERNR}; LAYERNR++))
do
    THISPARAMFILE=${PFSTEM}Layer${LAYERNR}.txt
    cp ${PARAMFILE} ${THISPARAMFILE}
    if [[ $WFIMAGES -eq 1 ]]; then
        INCREASEDFILES=$(($NRFILESPERDISTANCE+10))
    else
        INCREASEDFILES=$NRFILESPERDISTANCE
    fi
    STARTFILENRTHISLAYER=$(($(($(($LAYERNR-1))*$(($NDISTANCES))*$(($INCREASEDFILES))))+$OVERALLSTARTNR))
    GLOBALPOSITIONTHISLAYER=$(($(($(($LAYERNR-1))*$(($LAYERTHICKNESS))))+$STARTGLOBALPOS))
    echo "RawStartNr ${STARTFILENRTHISLAYER}" >> ${THISPARAMFILE}
    echo "GlobalPosition ${GLOBALPOSITIONTHISLAYER}" >> ${THISPARAMFILE}
    echo "MicFileBinary MicrostructureBinary_Layer${LAYERNR}.mic" >> ${THISPARAMFILE}
    echo "MicFileText MicrostructureText_Layer${LAYERNR}.mic" >> ${THISPARAMFILE}
    echo "ReducedFileName ${FOLDER}_Layer${LAYERNR}_Reduced/${FILENAME}" >> ${THISPARAMFILE}
    mkdir -p ${FOLDER}_Layer${LAYERNR}_Reduced
    if [[ ${FFSEEDORIENTATIONS} -eq 1 ]]; then
        echo "GrainsFile ${DATADIRECTORY}/GrainsLayer${LAYERNR}.csv" >> ${THISPARAMFILE}
        echo "SeedOrientations ${DATADIRECTORY}/Orientations_Layer${LAYERNR}.txt" >> ${THISPARAMFILE}
    else
        echo "SeedOrientations ${DATADIRECTORY}/Orientations.txt" >> ${THISPARAMFILE}
    fi
    ${PFDIR}/runSingleLayer.sh ${THISPARAMFILE} ${NCPUS} ${FFSEEDORIENTATIONS} ${PROCESSIMAGES} $7
done

EmailAdd=$7
echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
