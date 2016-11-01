#!/bin/bash
dirThis=${HOME}/.MIDAS

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

if [ ! -d ${dirThis}/swift-0.95-RC6 ]; then # SWIFT
	cd $dirThis
	echo $(pwd)
	wget -O swift-0.95-RC6.tar.gz https://db.tt/62ZtqZvA
	tar -xvf swift-0.95-RC6.tar.gz
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
