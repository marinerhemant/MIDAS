#!/bin/bash

source ${HOME}/.MIDAS/pathsNF
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.MIDAS/LIBTIFF/lib/
export $LD_LIBRARY_PATH

${BINFOLDER}/ImageProcessingLibTiff $1 $2 $3
