#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
source ${HOME}/.MIDAS/paths

outfolder=$1
cd ${outfolder}
pfname=$2
MargABC=$( awk '$1 ~ /^MargABC/ { print $2 }' ${pfname} )
MargABG=$( awk '$1 ~ /^MargABG/ { print $2 }' ${pfname} )
FileStem=$( awk '$1 ~ /^FileStem/ {print $2 }' ${pfname} )
layernr=$( awk '$1 ~ /^LayerNr/ { print $2 }' ${pfname} )
RingNrs=$( awk '$1 ~ /^RingThresh/ { print $2 }' ${pfname} )
SNr=$( awk '$1 ~ /^StartNr/ { print $2 }' ${pfname} )
ENr=$( awk '$1 ~ /^EndNr/ { print $2 }' ${pfname} )
SGNum=$( awk '$1 ~ /^SpaceGroup/ { print $2 }' ${pfname} )
for Ring in ${RingNrs}
do
	cp ${outfolder}/PeakSearch/${FileStem}_${layernr}/paramstest.txt ${outfolder}/paramstest_RingNr${Ring}.txt
	cp ${outfolder}/PeakSearch/${FileStem}_${layernr}/Radius_StartNr_${SNr}_EndNr_${ENr}_RingNr_${Ring}.csv ${outfolder}/
	cp ${outfolder}/paramstest_RingNr${Ring}.txt ${outfolder}/paramstest.txt
done
sed -i '/^OutputFolder/d' paramstest.txt
sed -i '/^ResultFolder/d' paramstest.txt
sed -i '/^RingRadii/d' paramstest.txt
sed -i '/^RingNumbers/d' paramstest.txt
echo "OutputFolder ${outfolder}/Output" >> paramstest.txt
echo "ResultFolder ${outfolder}/Results" >> paramstest.txt
echo "MargABC ${MargABC}" >> paramstest.txt
echo "MargABG ${MargABG}" >> paramstest.txt
for RINGs in ${RingNrs}
do
	echo "RingNumbers ${RINGs}" >> paramstest.txt
	paramstest_this_ring=paramstest_RingNr${RINGs}.txt
	rad=$( awk '$1 ~ /^RingRadii/ {print $2 }' ${paramstest_this_ring} )
	echo "RingRadii $rad" >> paramstest.txt
done
echo "SpotIDs To Index:"
wc -l SpotsToIndex.csv
echo "SpaceGroup $SGNum" >> paramstest.txt
cp SpotsToIndex.csv SpotsToIndexIn.csv
cat SpotsToIndex.csv |sort|uniq|less > SpotsToIndexUnq.csv
mv SpotsToIndexUnq.csv SpotsToIndex.csv
${PFDIR}/SHMOperators.sh
mkdir -p Output
mkdir -p Results
