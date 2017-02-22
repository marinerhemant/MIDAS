#!/bin/bash
source ${HOME}/.MIDAS/paths
echo "Chunk:"
echo $1
${PFDIR}/IndexStrains.py $1
