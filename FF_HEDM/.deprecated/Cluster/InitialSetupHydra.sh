#!/bin/bash -e

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

paramfile=$1 # always relative path
layernr=$2
DETECTORNR=$3
echo $DETECTORNR
python ${PFDIR}/checkFiles.py ${paramfile} ${layernr} ${DETECTORNR}
if [[ $? != 0 ]]; then
	exit 1
fi
outfolder=$4
seedfolder=$( pwd )
startfilenrfirstlayer=$( awk '$1 ~ /^StartFileNrFirstLayer/ { print $2 } ' ${paramfile} )
nrfilesperlayer=$( awk '$1 ~ /^NrFilesPerSweep/ { print $2 } ' ${paramfile} )
startfilenr=$((${startfilenrfirstlayer}+$((${nrfilesperlayer}*$((${layernr}-1))))))
mkdir -p ${outfolder}
cp ${seedfolder}/BigDetectorMask.bin ${outfolder}/.
cd ${outfolder}
cp ${seedfolder}/${paramfile} ${outfolder}
outfolder=${outfolder}/Detector${DETECTORNR}/
echo $outfolder
mkdir -p ${outfolder}
cd ${outfolder}
cp ${seedfolder}/${paramfile} ${outfolder}
python ${PFDIR}/prepareFilesHydra.py ${paramfile} ${DETECTORNR}
${BINFOLDER}/GetHKLList ${paramfile}
pfname=${outfolder}/Layer${layernr}_MultiRing_${paramfile}
cp ${paramfile} ${pfname}
finalringtoindex=$( awk '$1 ~ /^OverAllRingToIndex/ { print $2 }' ${paramfile} )
echo "LayerNr ${layernr}" >> ${pfname}
echo "RingToIndex ${finalringtoindex}" >> ${pfname}
echo "Folder ${outfolder}" >> ${pfname}
ringnrsfile=${outfolder}/Layer${layernr}_RingInfo.txt
rm -rf ${ringnrsfile}
paramfnstem=${outfolder}/Layer${layernr}_Ring
NewType=$( awk '$1 ~ /^NewType/ { print $2 }' ${paramfile} )
Thresholds=($( awk '$1 ~ /^RingThresh/ { print $3 }' ${paramfile} ))
i=0
SNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${paramfile} )
ENr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${paramfile} )
RingNrs=$( awk '$1 ~ /^RingThresh/ { print $2 }' ${paramfile} )
fStem=$( awk '$1 ~ /^FileStem/ { print $2 }' ${paramfile} )
echo "${pfname}" >> ${outfolder}/PFNames.txt
for RINGNR in ${RingNrs}
do
	ThisParamFileName=${outfolder}/Layer${layernr}_Ring${RINGNR}_${paramfile}
	cp ${paramfile} ${ThisParamFileName}
	Fldr=${outfolder}/Ring${RINGNR}
	mkdir -p $Fldr/Temp
	mkdir -p $Fldr/PeakSearch/${fStem}_${layernr}
	cp hkls.csv $Fldr
	echo "Folder $Fldr" >> ${ThisParamFileName}
	echo "RingToIndex $RINGNR" >> ${ThisParamFileName}
	echo "RingNumbers $RINGNR" >> ${ThisParamFileName}
	echo "LowerBoundThreshold ${Thresholds[$i]}" >> ${ThisParamFileName}
	echo "LayerNr ${layernr}" >> ${ThisParamFileName}
	echo "StartFileNr $startfilenr" >> ${ThisParamFileName}
	echo "RingNumbers $RINGNR" >> $pfname
	echo $RINGNR >> ${ringnrsfile}
	i=$((i+1))
	echo "${ThisParamFileName}" >> ${outfolder}/ParamFileNames.txt
done
cp ${ringnrsfile} ${seedfolder}/RingInfo.txt
