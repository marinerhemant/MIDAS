#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

if [[ ${#*} != 5 ]];
then
  echo "Usage: $0 NFPath FFPath StartLayerNr EndLayerNr FF-NF-Offset "
  echo "Eg. $0 /data/tomo1/NF /data/tomo1/FF 1 10 5"
  echo "This will copy Grains.csv files from FF folder(s) to NF folder(s) with appropriate (re-)naming."
  echo "FF-NF-Offset is non-zero when the layers in NF and FF don't match."
  echo "In case offset is non-zero, the StartLayerNr and EndLayerNr correspond to the NF layer numbers."
  echo "Offset is NF_LayerNr = FF_LayerNr + Offset"
  exit 1
fi

for ((LAYERNR=$3; LAYERNR<=$4; LAYERNR++))
do
    FFLAYERNR=$(($((LAYERNR))-$(($5))))
    cp -v $2/*Layer${FFLAYERNR}*/Grains.csv $1/GrainsLayer${LAYERNR}.csv
done
