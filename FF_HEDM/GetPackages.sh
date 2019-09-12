#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

###### Provide a machine name as argument to check for package

dirThis=${HOME}/.MIDAS
mkdir -p ${dirThis}

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
	wget -O swift.tar.gz https://www.dropbox.com/s/rhcav2jxplemuma/swift-0.96.2.tar.gz?dl=0
	tar -xvzf swift.tar.gz
	mv swift-0.96.2 swift
fi

if [ ! -d ${dirThis}/LIBTIFF ]; then # TIFF
	cd $dirThis
	echo $(pwd)
	wget -O libtiff.tar.gz http://download.osgeo.org/libtiff/tiff-4.0.6.tar.gz
	tar -xvf libtiff.tar.gz
	cd tiff-4.0.6
	./configure --prefix=${dirThis}/LIBTIFF
	make all
	make install
fi
if [ ! -d ${dirThis}/ZLIB ]; then # ZLIB
	cd $dirThis
	echo $(pwd)
	wget -O zlib.tar.gz https://www.dropbox.com/s/v44annn9lvx84e2/zlib-1.2.11.tar.gz?dl=0
	tar -xvzf zlib.tar.gz
	cd zlib-1.2.11
	./configure --prefix=${dirThis}/ZLIB
	make install
fi

if [ ! -d ${dirThis}/HDF5 ]; then # HDF5
	cd $dirThis
	echo $(pwd)
	wget -O hdf5.tar.gz https://www.dropbox.com/s/bv4b36qhsilprzf/hdf5-1.8.13.tar.gz?dl=0
	tar -xvzf hdf5.tar.gz
	cd hdf5-1.8.13
	./configure --prefix=${dirThis}/HDF5 --with-zlib=${dirThis}/ZLIB #--enable-shared
	make install
fi

if [ ! -d ${dirThis}/jre1.8.0_181 ]; then # java
	cd $dirThis
	wget -O jre8.tar.gz https://www.dropbox.com/s/1pawwgh9k1xpdgg/jre-8u181-linux-x64.tar.gz?dl=0
	tar -xvzf jre8.tar.gz
fi

if [ ! -d ${dirThis}/FFTW ]; then # fftw
	cd $dirThis
	wget -O fftw.tar.gz https://www.dropbox.com/s/dug1mpsr10rvlqi/fftw-3.3.8.tar.gz?dl=0
	tar -xvzf fftw.tar.gz
	cd fftw-3.3.8
	./configure --prefix=${dirThis}/FFTW --enable-float --disable-fortran
	make install
fi
