#!/bin/bash

cd ~/.MIDAS
rm -rf netcdf* zlib* hdf5*
wget http://zlib.net/zlib-1.2.8.tar.gz -O zlib-1.2.8.tar.gz
tar -xvf zlib-1.2.8.tar.gz
cd zlib-1.2.8
ZDIR=~/.MIDAS/zlib
./configure --prefix=${ZDIR}
make install
cd ~/.MIDAS
wget https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.13/src/hdf5-1.8.13.tar.gz -O hdf5-1.8.13.tar.gz
tar -xvf hdf5-1.8.13.tar.gz
cd hdf5-1.8.13
H5DIR=~/.MIDAS/hdf5
./configure --with-zlib=${ZDIR} --prefix=${H5DIR}
make install
cd ~/.MIDAS
wget https://curl.haxx.se/download/curl-7.49.1.tar.gz -O curl-7.49.1.tar.gz
tar -xvf curl-7.49.1.tar.gz
cd curl-7.49.1
CURLLIBDIR=~/.MIDAS/curl
./configure --prefix=${CURLLIBDIR}
make install
cd ~/.MIDAS
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.4.0.tar.gz
tar -xvf netcdf-4.4.0.tar.gz
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${H5DIR}/lib:${CURLLIBDIR}/lib
NCDIR=~/.MIDAS/netcdf
cd netcdf-4.4.0
CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR}
make install
