#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

dirThis=${HOME}/.MIDAS
mkdir -p ${dirThis}

if [ ! -d ${dirThis}/NLOPT ]; then # NLOPT INSTALL
	cd $dirThis
	echo $(pwd)
	wget -O nlopt.tar.gz https://www.dropbox.com/scl/fi/ux4ccf23z7rotkgbqbrmk/nlopt-2.4.2.tar.gz?rlkey=afq6l6yyu9fnw1hpe62l4gwqq&dl=0 
	wait
	tar -xzf nlopt.tar.gz
	cd nlopt-2.4.2
	./configure --prefix=${dirThis}/NLOPT
	make -j8 install
fi

# if [ ! -d ${dirThis}/NLOPTShared ]; then # NLOPT SHARED INSTALL
# 	cd $dirThis
# 	echo $(pwd)
# 	rm -rf nlopt-2.4.2
# 	tar -xzf nlopt.tar.gz
# 	cd nlopt-2.4.2
# 	./configure --prefix=${dirThis}/NLOPTShared --enable-shared
# 	make -j8 install
# fi

if [ ! -d ${dirThis}/swift ]; then # SWIFT
	cd $dirThis
	echo $(pwd)
	wget -O swift.tar.gz https://www.dropbox.com/s/rhcav2jxplemuma/swift-0.96.2.tar.gz?dl=0
	tar -xzf swift.tar.gz
	mv swift-0.96.2 swift
fi

if [ ! -d ${dirThis}/LIBTIFF ]; then # TIFF
	cd $dirThis
	echo $(pwd)
	wget -O libtiff.tar.gz https://www.dropbox.com/scl/fi/tk3axrjtjgxmjj9hzsk13/tiff-4.6.0.tar.gz?rlkey=judqzxze5g4sg0bviyul8kqvp&dl=0   
	wait
	tar -xzf libtiff.tar.gz
	cd tiff-4.6.0
	./configure --prefix=${dirThis}/LIBTIFF --enable-shared
	make -j8 install
fi

if [ ! -d ${dirThis}/ZLIB ]; then # ZLIB
	cd $dirThis
	echo $(pwd)
	wget -O zlib.tar.gz https://www.dropbox.com/scl/fi/vrgb8i9755eojx5eh1a2r/zlib-1.3.1.tar.gz?rlkey=9jwwjc1aqmflin5b75r2lc2yw&dl=0 
	wait
	tar -xzf zlib.tar.gz
	cd zlib-1.3.1
	./configure --prefix=${dirThis}/ZLIB
	make -j8 install
fi

if [ ! -d ${dirThis}/HDF5 ]; then # HDF5
	cd $dirThis
	echo $(pwd)
	wget -O hdf5.tar.gz https://www.dropbox.com/scl/fi/ib4wkq1s9jhm0oi9n6r7c/hdf5-1.14.2.tar.gz?rlkey=eq20hs7juecpwcn1vuumssjuf&dl=0
	wait
	tar -xzf hdf5.tar.gz
	cd hdf5-1.14.2
	./configure --prefix=${dirThis}/HDF5 --with-zlib=${dirThis}/ZLIB
	make -j8 install
fi

# if [ ! -d ${dirThis}/jre1.8.0_181 ]; then # java
# 	cd $dirThis
# 	wget -O jre8.tar.gz https://www.dropbox.com/s/1pawwgh9k1xpdgg/jre-8u181-linux-x64.tar.gz?dl=0
# 	tar -xzf jre8.tar.gz
# fi

if [ ! -d ${dirThis}/FFTW ]; then # fftw
	cd $dirThis
	wget -O fftw.tar.gz https://www.dropbox.com/scl/fi/yugsuwobadxt5gvfsdz46/fftw-3.3.10.tar.gz?rlkey=cfo1rwazrr4gbm2k043np8skj&dl=0 
	wait
	tar -xzf fftw.tar.gz
	cd fftw-3.3.10
	./configure --prefix=${dirThis}/FFTW --enable-float --disable-fortran --enable-sse --enable-sse2 --enable-avx --enable-avx2 --enable-avx-128-fma --enable-generic-simd128 --enable-generic-simd256 #--enable-avx512
	make -j8 install
fi

if [ ! -d ${dirThis}/BLOSC1 ]; then # blosc1
	cd $dirThis
	git clone https://github.com/Blosc/c-blosc
	cd c-blosc
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=${dirThis}/BLOSC1 ..
	cmake --build . -j 8
	cmake --build . -j 8 --target install
fi

if [ ! -d ${dirThis}/BLOSC ]; then # blosc
	cd $dirThis
	git clone https://github.com/Blosc/c-blosc2
	cd c-blosc2
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=${dirThis}/BLOSC -DBUILD_BENCHMARKS=OFF ..
	cmake --build . -j 8
	cmake --build . -j 8 --target install
fi

if [ ! -d ${dirThis}/LIBZIP ]; then # libzip
	cd $dirThis
	wget -O libzip.tar.gz https://www.dropbox.com/scl/fi/2mo9gzxi8ms3pp10pu6ad/libzip-1.10.1.tar.gz?rlkey=w7ph5tzczb2tfjatul31bs6x4&dl=0
	wait
	tar -xzf libzip.tar.gz
	cd libzip-1.10.1
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=${dirThis}/LIBZIP ..
	cmake --build . -j 8
	cmake --build . -j 8 --target install
fi

