
#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

CC=gcc
NCC=nvcc
NCFLAGS=-O3 -G -g -w -arch sm_52 -lm -rdc=true # -res-usage
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w
NLOPT=""
ifneq ($(NLOPT),"")
	CFLAGSNLOPT=-I$(NLOPT)/include -L$(NLOPT)/lib -lnlopt
else
	CFLAGSNLOPT=-I$${HOME}/.MIDAS/NLOPT/include -L$${HOME}/.MIDAS/NLOPT/lib -lnlopt
endif
TIFF=""
ifneq ($(TIFF),"")
	CFLAGSTIFF=-I$(TIFF)/include -L$(TIFF)/lib -ltiff
else
	CFLAGSTIFF=-I$${HOME}/.MIDAS/LIBTIFF/include -L$${HOME}/.MIDAS/LIBTIFF/lib -ltiff
endif
SRCDIR=src/
BINDIR=bin/

all: help bindircheck makediffrspots makehexgrid medianimage imageprocessing \
	fitorientation fitorientationparameters fitorientationparametersmultipoint \
	genseedorientations simulatediffrspots processnfmicremote \
	mmapimageinfo hkls parsemic fitorientationsinglepoint

#local:
local: predeplocal all runsetup

# Need to try orthrosextra
orthros: predeport all runsetup

# Works
biocluster: predepbio all runsetup

# Works
lcrc_cloud: predeplcrc_cloud all runsetup

# Works?
lcrc: predeplcrc all runsetup

# Works
nersc: predepnersc all runsetup

# Works
purdue: predeppurdue all runsetup

# LANL
lanl: predeplanl all runsetup

stampede: predepstampede all runsetup

notchpeak: predepnotchpeak all runsetup

runsetup:
	Cluster/setup.sh

predeplocal:
	./GetPackages.sh local
predeport:
	./GetPackages.sh orthros
predepbio:
	./GetPackages.sh bio
predeplcrc:
	./GetPackages.sh lcrc
predeplcrc_cloud:
	./GetPackages.sh lcrc_cloud
predepnersc:
	./GetPackages.sh nersc
predeppurdue:
	./GetPackages.sh purdue
predeplanl:
	./GetPackages.sh lanl
predepstampede:
	./GetPackages.sh stampede
predepnotchpeak:
	./GetPackages.sh notchpeak

bindircheck:
	mkdir -p $(BINDIR)

help:
	@echo
	@echo ..............................................................................................
	@echo ....................The make file to compile every C code for FarField HEDM...................
	@echo ..............................\"make all\" to compile all codes.................................
	@echo ...........To run, give the path for NLOPT, or give nothing if it is installed by default.....
	@echo ..............eg. on orthros: NLOPT=\"/clhome/TOMO1/PeaksAnalysisHemant/NF_HEDM/NLOPT/\"........
	@echo ....................................\"make clean\" to clean.....................................
	@echo ...............................Do the same for TIFF if needed.................................
	@echo .....................eg. TIFF=\"/clhome/TOMO1/PeaksAnalysisHemant/Libtiff\".....................
	@echo ..............................................................................................
	@echo
	@echo $(TIFF) $(NLOPT)

makehexgrid: $(SRCDIR)MakeHexGrid.c
	$(CC) $(SRCDIR)MakeHexGrid.c -o $(BINDIR)MakeHexGrid $(CFLAGS)

makediffrspots: $(SRCDIR)MakeDiffrSpots.c
	$(CC) $(SRCDIR)MakeDiffrSpots.c -o $(BINDIR)MakeDiffrSpots $(CFLAGS)

makediffrspotsff2nf: $(SRCDIR)MakeDiffrSpotsFF2NF.c
	$(CC) $(SRCDIR)MakeDiffrSpotsFF2NF.c -o $(BINDIR)MakeDiffrSpotsFF2NF $(CFLAGS)

medianimage: $(SRCDIR)MedianImageLibTiff.c
	$(CC) $(SRCDIR)MedianImageLibTiff.c -o $(BINDIR)MedianImageLibTiff $(CFLAGS) $(CFLAGSTIFF)

imageprocessing: $(SRCDIR)ImageProcessingLibTiff.c
	$(CC) $(SRCDIR)ImageProcessingLibTiff.c -o $(BINDIR)ImageProcessingLibTiff $(CFLAGS) $(CFLAGSTIFF)

fitorientation: $(SRCDIR)FitOrientation.c
	$(CC) $(SRCDIR)FitOrientation.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientation $(CFLAGS) $(CFLAGSNLOPT)

fitorientationsinglepoint: $(SRCDIR)FitOrientationSinglePoint.c
	$(CC) $(SRCDIR)FitOrientationSinglePoint.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientationSinglePoint $(CFLAGS) $(CFLAGSNLOPT)

fitorientationff2nf: $(SRCDIR)FitOrientationFF2NF.c
	$(CC) $(SRCDIR)FitOrientationFF2NF.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientationFF2NF $(CFLAGS) $(CFLAGSNLOPT)

fitorientationparameters: $(SRCDIR)FitOrientationParameters.c
	$(CC) $(SRCDIR)FitOrientationParameters.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientationParameters $(CFLAGS) $(CFLAGSNLOPT)

fitorientationparametersmultipoint: $(SRCDIR)FitOrientationParametersMultiPoint.c
	$(CC) $(SRCDIR)FitOrientationParametersMultiPoint.c $(SRCDIR)CalcDiffractionSpots.c \
	$(SRCDIR)SharedFuncsFit.c -o $(BINDIR)FitOrientationParametersMultiPoint $(CFLAGS) $(CFLAGSNLOPT)

genseedorientations: $(SRCDIR)GenSeedOrientationsFF2NFHEDM.c
	$(CC) $(SRCDIR)GenSeedOrientationsFF2NFHEDM.c -o $(BINDIR)GenSeedOrientationsFF2NFHEDM $(CFLAGS)

simulatediffrspots: $(SRCDIR)SimulateDiffractionSpots.c
	$(CC) $(SRCDIR)SimulateDiffractionSpots.c -o $(BINDIR)SimulateDiffractionSpots $(CFLAGS)

convertbinfiles: $(SRCDIR)Convert-Bin-L2N.c
	$(CC) $(SRCDIR)Convert-Bin-L2N.c -o $(BINDIR)ConvertBinFiles $(CFLAGS)

processnfmicremote: $(SRCDIR)ProcessNFMicRemote.c
	$(CC) $(SRCDIR)ProcessNFMicRemote.c -o $(BINDIR)ProcessNFMicRemote $(CFLAGS)

mmapimageinfo: $(SRCDIR)MMapImageInfo.c
	$(CC) $(SRCDIR)MMapImageInfo.c -o $(BINDIR)MMapImageInfo $(CFLAGS)

hkls: $(SRCDIR)GetHKLList.c
	$(CC) $(SRCDIR)GetHKLList.c $(SRCDIR)sgclib.c $(SRCDIR)sgfind.c $(SRCDIR)sghkl.c $(SRCDIR)sgsi.c \
	$(SRCDIR)sgio.c -o $(BINDIR)GetHKLList $(CFLAGS)

parsemic: $(SRCDIR)ParseMic.c
	$(CC) $(SRCDIR)ParseMic.c -o $(BINDIR)ParseMic $(CFLAGS)

clean:
	rm -rf $(BINDIR)
	mkdir $(BINDIR)
