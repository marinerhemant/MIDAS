#!/bin/bash
source ${HOME}/.MIDAS/paths
echo "Chunk:"
echo $1
python ${PFDIR}/IndexStrains.py $1
