#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

###### Provide a machine name as argument to check for package

dirThis=${HOME}/.MIDAS
mkdir -p ${dirThis}
package=${HOME}/opt/MIDAS/FF_HEDM/Cluster/Packages/${1}.tar.gz
echo ${package}
if [ -f ${package} ]
then
	echo "Package found for deployment. ${package}"
	cp ${package} ${dirThis}/
	cd ${dirThis}
	tar -xf ${1}.tar.gz
fi

if [ ! -d ${dirThis}/NLOPT ]; then # NLOPT INSTALL
	mkdir -p $dirThis
	cd $dirThis
	echo $(pwd)
	wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
	tar -xvf nlopt-2.4.2.tar.gz
	cd nlopt-2.4.2
	./configure --prefix=${dirThis}/NLOPT
	make all
	make install
fi

if [ ! -d ${dirThis}/swift ]; then # SWIFT
	cd $dirThis
	echo $(pwd)
	wget -O swift.tar.gz https://dl.dropboxusercontent.com/u/19201865/swift-0.96.2.tar.gz #https://db.tt/62ZtqZvA
	tar -xvzf swift.tar.gz
	mv swift-0.96.2 swift
fi

if [ ! -d ${dirThis}/LIBTIFF ]; then
	cd $dirThis
	echo $(pwd)
	wget -O libtiff.tar.gz http://download.osgeo.org/libtiff/tiff-4.0.6.tar.gz
	tar -xvf libtiff.tar.gz
	cd tiff-4.0.6
	./configure --prefix=${dirThis}/LIBTIFF
	make all
	make install
fi

cd ${HOME}/opt/MIDAS/FF_HEDM
./GetNetCDF.sh
