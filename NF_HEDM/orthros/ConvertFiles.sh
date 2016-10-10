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

source ${HOME}/.MIDAS/pathsNF

${BINFOLDER}/ConvertBinFiles $InFN $OutFN
