#!/bin/bash

InFS=$1
OutFS=$2
FileNr=$3
LayerNr=$4
printf -v PadFileNr "%06d" $FileNr
InFN=${InFS}_${PadFileNr}.bin${LayerNr}
OutFN=${OutFS}_${PadFileNr}.bin${LayerNr}
echo $InFN
echo $OutFN

/clhome/TOMO1/PeaksAnalysisHemant/HEDM_V2/NF_HEDM/bin/ConvertBinFiles $InFN $OutFN
