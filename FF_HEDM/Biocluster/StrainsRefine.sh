#!/bin/bash
source ${HOME}/.MIDAS/paths
echo "Spot ID:"
echo $1
${BINFOLDER}/FitPosOrStrains paramstest.txt $1
