#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

dirThis=${HOME}/.MIDAS

ZDIR=~/.MIDAS/zlib
H5DIR=~/.MIDAS/hdf5
CURLLIBDIR=~/.MIDAS/curl
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${H5DIR}/lib:${CURLLIBDIR}/lib
NCDIR=~/.MIDAS/netcdf

if [ ! -d ${dirThis}/zlib/include ]; then #ZLIB
	mkdir -p $dirThis
	cd ${dirThis}
	echo $(pwd)
	wget http://zlib.net/zlib-1.2.11.tar.gz -O zlib-1.2.11.tar.gz
	tar -xvf zlib-1.2.11.tar.gz
	cd zlib-1.2.11
	./configure --prefix=${ZDIR}
	make install
fi

if [ ! -d ${dirThis}/hdf5/include ]; then #HDF5
	cd ${dirThis}
	echo $(pwd)
	wget -O hdf5-1.8.13.tar.gz https://www.dropbox.com/s/bv4b36qhsilprzf/hdf5-1.8.13.tar.gz?dl=0 # https://dl.dropboxusercontent.com/u/19201865/hdf5-1.8.13.tar.gz -O hdf5-1.8.13.tar.gz
	tar -xvf hdf5-1.8.13.tar.gz
	cd hdf5-1.8.13
	./configure --with-zlib=${ZDIR} --prefix=${H5DIR}
	make install
fi

if [ ! -d ${dirThis}/curl/include ]; then #HDF5
	cd ${dirThis}
	echo $(pwd)
	wget -O curl-7.49.1.tar.gz https://www.dropbox.com/s/yu2tg26djnuuwqb/curl-7.49.1.tar.gz?dl=0 # https://dl.dropboxusercontent.com/u/19201865/curl-7.49.1.tar.gz -O curl-7.49.1.tar.gz
	tar -xvf curl-7.49.1.tar.gz
	cd curl-7.49.1
	./configure --prefix=${CURLLIBDIR}
	make install
fi

if [ ! -d ${dirThis}/netcdf/include ]; then #HDF5
	cd ${dirThis}
	echo $(pwd)
	wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.4.0.tar.gz
	tar -xvf netcdf-4.4.0.tar.gz
	cd netcdf-4.4.0
	CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR}
	make install
fi
