#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/pathsNF
cmdname=$(basename $0)

if [[ ${#*} != 8 ]];
then
	echo "Usage: ${cmdname} top_parameter_file(just the name, no directory) FFSeedOrientations ProcessImages startLayerNr endLayerNr nNODEs MachineName EmailAddress"
	echo "Eg. ${cmdname} ParametersFile.txt 1(or 0) 1(or 0) 1 5 6 orthros hsharma@anl.gov"
	echo "This will run from layer 1 to 5."
	echo "Run from the directory with the parameter file."
	echo "This will produce the output in the run folder."
	echo "FFSeedOrientations is when either Orientations exist already (0) or when you provide a FF Orientation file (1)."
	echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
	echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
	echo "At least the parameters file should be in the folder from where the command is executed."
	echo "The following parameters should NOT be in the ParametersFile.txt file:"
	echo "RawStartNr, GlobalPosition, MicFileBinary, MicFileText, ReducedFileName, GrainsFile, SeedOrientations, DataDirectory, FullSeedFile"
	echo "The following parameters must be present:"
	echo "OrigFileName, OverallStartNr, GlobalPositionFirstLayer, LayerThickness, WFImages, nDistances, NrFilesPerDistance, TopDataDirectory"
	echo "If WF images were taken, it is assumed there were 10 WF images."
	echo "If using FF seeding, the grains files should be in the same folder and named: GrainsLayer{LAYERNR}.csv"
	echo "If using FF seeding, include a file called OrientationsAll.txt with all possible orientations if you want to try all orientations for low confidence points."
	echo "If using FF seeding, please add line to your analysis: MinConfidence to redo the analysis using all orientations and minconfidence."
	echo "If no seeding is used, a single OrientationsAll.txt file should be present."
	echo "At successfull completion, it will send an email to EmailAddress."
	echo "**********NOTE: For local runs, nNodes should be nCPUs.**********"
	exit 1
fi

PARAMFILE=$1
EmailAdd=$8
nNODES=$6
export nNODES
MACHINE_NAME=$7
if [[ ${MACHINE_NAME} == *"edison"* ]]; then
	echo "We are in NERSC EDISON"
	hn=$( hostname )
	hn=${hn: -2}
	hn=${hn#0}
	hn=$(( hn+20 ))
	intHN=128.55.203.${hn}
	export intHN
	echo "IP address of login node: $intHN"
elif [[ ${MACHINE_NAME} == *"cori"* ]]; then
	echo "We are in NERSC CORI"
	hn=$( hostname )
	hn=${hn: -2}
	hn=${hn#0}
	hn=$(( hn+30 ))
	intHN=128.55.224.${hn}
	export intHN
	echo "IP address of login node: $intHN"
else
	intHN=10.10.10.100
	export intHN
fi
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
if [[ $WFIMAGES -eq 1 ]]; then
	INCREASEDFILES=$(($NRFILESPERDISTANCE+10))
else
	INCREASEDFILES=$NRFILESPERDISTANCE
fi

TOPDATADIRECTORY=$( awk '$1 ~ /^TopDataDirectory/ { print $2 }' ${PARAMFILE} )
extOrig=$( awk '$1 ~ /^extOrig/ { print $2 }' ${PARAMFILE} )
tmpfn=${TOPDATADIRECTORY}/fns.txt
AllOrientationFile=$( awk '$1 ~ /^FullSeedFile/ { print $2 }' ${PARAMFILE} )

for ((LAYERNR=${STARTLAYERNR}; LAYERNR<=${ENDLAYERNR}; LAYERNR++))
do
	NEWFOLDER=${TOPDATADIRECTORY}/${FOLDER}Layer${LAYERNR}/
	mkdir -p ${NEWFOLDER}/${FOLDER}
	THISPARAMFILE=${PFSTEM}Layer${LAYERNR}.txt
	cp ${PARAMFILE} ${NEWFOLDER}/${THISPARAMFILE}
    if [[ -n $( awk '$1 ~ /^TomoImage/ { print }' ${PARAMFILE} ) ]];
    then
        TomoFN=$( awk '$1 ~ /^TomoImage/ { print $2 }' ${PARAMFILE} )
        cp ${TomoFN} ${NEWFOLDER}
    fi
	cd ${NEWFOLDER}
    if [[ $WFIMAGES -eq 1 ]]; then
        INCREASEDFILES=$(($NRFILESPERDISTANCE+10))
    else
        INCREASEDFILES=$NRFILESPERDISTANCE
    fi
    STARTFILENRTHISLAYER=$(($(($(($LAYERNR-1))*$(($NDISTANCES))*$(($INCREASEDFILES))))+$OVERALLSTARTNR))
    ENDFILENRTHISLAYER=$(($STARTFILENRTHISLAYER+$(($(($NDISTANCES))*$(($INCREASEDFILES))))-1))
    printf -v startnr "%06d" ${STARTFILENRTHISLAYER}
    printf -v endnr "%06d" ${ENDFILENRTHISLAYER}
    for ((FILENR=${STARTFILENRTHISLAYER}; FILENR<=${ENDFILENRTHISLAYER}; FILENR++))
    do
		printf -v filenrformat "%06d" ${FILENR}
		fn=${TOPDATADIRECTORY}/${STEM}_${filenrformat}.${extOrig}
		if [ -f ${fn} ]; then
			mv ${fn} ${NEWFOLDER}/${FOLDER}/
		fi
	done
    DATADIRECTORY=${NEWFOLDER}
    GLOBALPOSITIONTHISLAYER=$(($(($(($LAYERNR-1))*$(($LAYERTHICKNESS))))+$STARTGLOBALPOS))
    echo "RawStartNr ${STARTFILENRTHISLAYER}" >> ${THISPARAMFILE}
    echo "GlobalPosition ${GLOBALPOSITIONTHISLAYER}" >> ${THISPARAMFILE}
    echo "MicFileBinary MicrostructureBinary_Layer${LAYERNR}.mic" >> ${THISPARAMFILE}
    echo "MicFileText MicrostructureText_Layer${LAYERNR}.mic" >> ${THISPARAMFILE}
    echo "ReducedFileName ${FOLDER}_Layer${LAYERNR}_Reduced/${FILENAME}" >> ${THISPARAMFILE}
    echo "DataDirectory ${DATADIRECTORY}" >> ${THISPARAMFILE}
    mkdir -p ${FOLDER}_Layer${LAYERNR}_Reduced
    if [[ ${FFSEEDORIENTATIONS} -eq 1 ]]; then
		mv ${TOPDATADIRECTORY}/GrainsLayer${LAYERNR}.csv ${DATADIRECTORY}/GrainsLayer${LAYERNR}.csv
		if [ -f ${TOPDATADIRECTORY}/OrientationsAll.txt ]; then
			cp ${TOPDATADIRECTORY}/OrientationsAll.txt ${DATADIRECTORY}/OrientationsAll.txt
			echo "FullSeedFile ${DATADIRECTORY}/OrientationsAll.txt" >> ${THISPARAMFILE}
        fi
        echo "GrainsFile ${DATADIRECTORY}/GrainsLayer${LAYERNR}.csv" >> ${THISPARAMFILE}
        echo "SeedOrientations ${DATADIRECTORY}/Orientations_Layer${LAYERNR}.txt" >> ${THISPARAMFILE}
    else
		cp ${TOPDATADIRECTORY}/OrientationsAll.txt ${DATADIRECTORY}/Orientations.txt
        echo "SeedOrientations ${DATADIRECTORY}/Orientations.txt" >> ${THISPARAMFILE}
    fi
    ${PFDIR}/SingleLayerSwift.sh ${DATADIRECTORY}/${THISPARAMFILE} ${FFSEEDORIENTATIONS} ${PROCESSIMAGES} ${nNODES} ${MACHINE_NAME} ${EmailAdd}
	cd ${TOPDATADIRECTORY}
done


echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
