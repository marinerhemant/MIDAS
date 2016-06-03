#!/bin/bash -eu

echo "FF analysis code for Multiple Layers and Multiple rings:"
echo "Version: 2, 2014/11/10, in case of problems contact hsharma@anl.gov"

if [[ ${#*} != 4 ]]
then
  echo "Provide ParametersFile StartLayerNr EndLayerNr and the number of CPUs to use!"
  echo "EG. RealtimeFFMultipleLayers parameters.txt 1 1 320"
  echo "the source parameter file should not have ring numbers and layer numbers in it."
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
STARTLAYERNR=$2
ENDLAYERNR=$3
NCPUS=$4
STARTFNRFIRSTLAYER=$( awk '$1 ~ /^StartFileNrFirstLayer/ { print $2 } ' ${TOP_PARAM_FILE} )

BINFolder=/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/FF_HEDM/

RingNrs=$( awk '$1 ~ /^RingThresh/ { print $2 }' ${TOP_PARAM_FILE} )
SGNum=$( awk '$1 ~ /^SpaceGroup/ { print $2 }' ${TOP_PARAM_FILE} )
Thresholds=($( awk '$1 ~ /^RingThresh/ { print $3 }' ${TOP_PARAM_FILE} ))
FinalRingToIndex=$( awk '$1 ~ /^OverAllRingToIndex/ { print $2 }' ${TOP_PARAM_FILE} )
SeedFolder=$( awk '$1 ~ /^SeedFolder/ { print $2 }' ${TOP_PARAM_FILE} )
NrFilesPerLayer=$( awk '$1 ~ /^NrFilesPerSweep/ { print $2 }' ${TOP_PARAM_FILE} )
MargABC=$( awk '$1 ~ /^MargABC/ { print $2 }' ${TOP_PARAM_FILE} )
MargABG=$( awk '$1 ~ /^MargABG/ { print $2 }' ${TOP_PARAM_FILE} )
Twins=$( awk '$1 ~ /^Twins/ { print $2 }' ${TOP_PARAM_FILE} )
SNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${TOP_PARAM_FILE} )
ENr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${TOP_PARAM_FILE} )
MinNrSpots=$( awk '$1 ~ /^MinNrSpots/ { print $2 }' ${TOP_PARAM_FILE} )
echo "Ring to be used for seed points: $FinalRingToIndex"
CHART=/
flr=${TOP_PARAM_FILE%$CHART*}
fstm=$( echo ${TOP_PARAM_FILE} | awk -F"/" '{print $NF}' )
FileStem=$( awk '$1 ~ /^FileStem/ {print $2 }' $TOP_PARAM_FILE)
for ((LAYERNR=$STARTLAYERNR; LAYERNR<=$ENDLAYERNR; LAYERNR++))
do 
   i=0
   echo "Processing peaks for layer: $LAYERNR"
   StartFileNr=$((${STARTFNRFIRSTLAYER}+$(($NrFilesPerLayer*$((${LAYERNR}-1))))))
   ${BINFolder}/bin/GetHKLList ${TOP_PARAM_FILE}
   echo "StartFileNumber $StartFileNr"
   echo "Creating overall parameter file for this Layer:"
   PFName=${flr}/Layer${LAYERNR}_MultiRing_${fstm}
   echo "$PFName"
   cp ${TOP_PARAM_FILE} ${PFName}
   echo "LayerNr $LAYERNR" >> ${PFName}
   echo "RingToIndex $FinalRingToIndex" >> $PFName
   echo "Folder $SeedFolder" >> $PFName
   OutFldr=${SeedFolder}/Layer${LAYERNR}
   cd ${SeedFolder}
   ide=$( date +%T )
   LayerDir=Layer${LAYERNR}
   if [ -d "$LayerDir" ]; then
       mv Layer${LAYERNR} /data/tomo1/__Can_be_deleted/${ide}
   fi
   mkdir -p ${OutFldr}
   cp hkls.csv ${OutFldr}
   for RINGNR in $RingNrs
   do
        ThisParamFileName=${flr}/Layer${LAYERNR}_Ring${RINGNR}_${fstm}
        echo "ParameterFile used: ${ThisParamFileName}"
        cp ${TOP_PARAM_FILE} ${ThisParamFileName}
        echo "Ring Number: $RINGNR, Threshold: ${Thresholds[$i]}"
        Fldr=${SeedFolder}/Ring${RINGNR}
        mkdir -p $Fldr
        cp hkls.csv $Fldr
        echo "Folder $Fldr" >> ${ThisParamFileName}
        echo "RingToIndex $RINGNR" >> ${ThisParamFileName}
        echo "RingNumbers $RINGNR" >> ${ThisParamFileName}
        echo "LowerBoundThreshold ${Thresholds[$i]}" >> ${ThisParamFileName}
        echo "LayerNr $LAYERNR" >> ${ThisParamFileName}
        echo "StartFileNr $StartFileNr" >> ${ThisParamFileName}
        echo "Output will be saved in $Fldr"
        echo "RingNumbers $RINGNR" >> $PFName
        ${BINFolder}./RunPeaks.sh $ThisParamFileName ${NCPUS}
        cp ${Fldr}/PeakSearch/${FileStem}_${LAYERNR}/paramstest.txt ${OutFldr}/paramstest_RingNr${RINGNR}.txt
        cp ${Fldr}/PeakSearch/${FileStem}_${LAYERNR}/Radius_StartNr_${SNr}_EndNr_${ENr}_RingNr_${RINGNR}.csv ${OutFldr}/.
        RingX=$RINGNR
        mv ${ThisParamFileName} ${OutFldr}
        i=`expr $i + 1`
   done
   ${BINFolder}/bin/MergeMultipleRings ${PFName}
   cd ${SeedFolder}
   mv SpotsToIndex* ${OutFldr}
   mv InputAll* ${OutFldr}
   mv ${PFName} ${OutFldr}
   cp hkls.csv ${OutFldr}
   cd ${OutFldr}
   cp paramstest_RingNr${RingX}.txt paramstest.txt
   sed -i '/^OutputFolder/d' paramstest.txt
   sed -i '/^ResultFolder/d' paramstest.txt
   sed -i '/^RingRadii/d' paramstest.txt
   sed -i '/^RingNumbers/d' paramstest.txt
   echo "OutputFolder ${OutFldr}/Output" >> paramstest.txt
   echo "ResultFolder ${OutFldr}/Results" >> paramstest.txt
   echo "MargABC ${MargABC}" >> paramstest.txt
   echo "MargABG ${MargABG}" >> paramstest.txt
   for RINGs in $RingNrs
   do
        echo "RingNumbers ${RINGs}" >> paramstest.txt
        paramstest_this_ring=paramstest_RingNr${RINGs}.txt
        rad=$( awk '$1 ~ /^RingRadii/ {print $2 }' $paramstest_this_ring)
        echo "RingRadii $rad" >> paramstest.txt
   done
   ${BINFolder}/SHMOperators.sh
   ${BINFolder}/IndexRefine.sh ${NCPUS} ${Twins} ${MinNrSpots} ${SGNum}
done

### RUN BY "/clhome/TOMO1/PeaksAnalysisHemant/FF_HEDM/FitDetectorSrc/MergeMultipleRings step2_RunAnalysis_MultiRing.txt
### then copy paramstest.txt from ${Folder}/Ring${RingToIndex}/... to ${Folder}
### In the new copied paramstest.txt, now copy RingRadii from paramstest.txt for each ring, and paste them in order. Also, add RingNumbers accordingly
### In paramstest.txt change the OutputFolder and ResultFolder to ${Folder}/Output & ${Folder}/Results
### To run: cd ${Folder}
###         MultiRingIndexer 320

