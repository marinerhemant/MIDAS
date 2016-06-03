#!/bin/bash
source ${HOME}/.bashrc

dirThis=$( pwd )
wget http://swiftlang.org/packages/swift-0.95-RC6.tar.gz
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xvf swift-0.95-RC6.tar.gz
tar -xvf nlopt-2.4.2.tar.gz
cd nlopt-2.4.2
./configure --prefix=${dirThis}/NLOPT
make all
make install

echo $PATH | grep -q swift
if [[ $? == 1 ]];
then
	echo export PATH=${dirThis}/NLOPT:${dirThis}/swift-0.95-RC6/bin:${PATH} >> ${HOME}/.bashrc;
	echo "Placed paths at"
	echo ${dirThis}/NLOPT:${dirThis}/swift-0.95-RC6/bin:${PATH}
fi
