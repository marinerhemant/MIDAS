#!/bin/bash

dirThis=${HOME}/.MIDAS
rm -rf $dirThis
mkdir -p $dirThis
cd $dirThis
echo $(pwd)
wget https://db.tt/62ZtqZvA
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xvf swift-0.95-RC6.tar.gz
tar -xvf nlopt-2.4.2.tar.gz
cd nlopt-2.4.2
./configure --prefix=${dirThis}/NLOPT
make all
make install
