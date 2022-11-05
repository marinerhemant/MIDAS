
#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

CC=gcc
NCC=nvcc
H5CC=$${HOME}/.MIDAS/HDF5/bin/h5cc
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
CFLAGSHDF=-I$${HOME}/.MIDAS/ZLIB/include -L$${HOME}/.MIDAS/ZLIB/lib -lz -I$${HOME}/.MIDAS/HDF5/include -L$${HOME}/.MIDAS/HDF5/lib -lhdf5 -lhdf5_hl
SRCDIR=src/
BINDIR=bin/

all: help bindircheck makediffrspots makehexgrid medianimage imageprocessing \
	fitorientation fitorientationparameters fitorientationparametersmultipoint \
	genseedorientations simulatediffrspots processnfmicremote gridfiltertomo\
	mmapimageinfo hkls parsemic fitorientationsinglepoint parsedeconv nfgrainscalc \
	comparenf simulatenf imageprocessingomp fitorientationomp

#local:
local: predep all runsetup

# Need to try orthrosextra
orthros: predep all runsetup

# Works
biocluster: predep all runsetup

# Works
lcrc_cloud: predep all runsetup

# Works?
lcrc: predep all runsetup

# Works
nersc: predep all runsetup

# Works
purdue: predep all runsetup

# LANL
lanl: predep all runsetup

stampede: predep all runsetup

notchpeak: predep all runsetup

upitt: predep all runsetup

runsetup:
	Cluster/setup.sh

predep:
	./GetPackages.sh

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

comparelayers: $(SRCDIR)compareLayers.c
	$(CC) $(SRCDIR)compareLayers.c -o $(BINDIR)compareLayers $(CFLAGS)

makehexgrid: $(SRCDIR)MakeHexGrid.c
	$(CC) $(SRCDIR)MakeHexGrid.c -o $(BINDIR)MakeHexGrid $(CFLAGS)

makediffrspots: $(SRCDIR)MakeDiffrSpots.c
	$(CC) $(SRCDIR)MakeDiffrSpots.c -o $(BINDIR)MakeDiffrSpots $(CFLAGS)

makediffrspotsff2nf: $(SRCDIR)MakeDiffrSpotsFF2NF.c
	$(CC) $(SRCDIR)MakeDiffrSpotsFF2NF.c -o $(BINDIR)MakeDiffrSpotsFF2NF $(CFLAGS)

medianimage: $(SRCDIR)MedianImageLibTiff.c
	$(CC) $(SRCDIR)MedianImageLibTiff.c -o $(BINDIR)MedianImageLibTiff $(CFLAGS) $(CFLAGSTIFF)

medianimagehdf: $(SRCDIR)MedianImageHDF.c
	$(H5CC) $(SRCDIR)MedianImageHDF.c -o $(BINDIR)MedianImageHDF $(CFLAGS) $(CFLAGSHDF) -fopenmp

imageprocessinghdf: $(SRCDIR)ImageProcessingHDF.c
	$(H5CC) $(SRCDIR)ImageProcessingHDF.c -o $(BINDIR)ImageProcessingHDF $(CFLAGS) $(CFLAGSHDF) -fopenmp

imageprocessing: $(SRCDIR)ImageProcessingLibTiff.c
	$(CC) $(SRCDIR)ImageProcessingLibTiff.c -o $(BINDIR)ImageProcessingLibTiff $(CFLAGS) $(CFLAGSTIFF)

imageprocessingomp: $(SRCDIR)ImageProcessingLibTiffOMP.c
	$(CC) $(SRCDIR)ImageProcessingLibTiffOMP.c -o $(BINDIR)ImageProcessingLibTiffOMP -fopenmp $(CFLAGS) $(CFLAGSTIFF)

parsedeconv: $(SRCDIR)ParseDeconvOutput.c
	$(CC) $(SRCDIR)ParseDeconvOutput.c -o $(BINDIR)ParseDeconvOutput $(CFLAGS) $(CFLAGSTIFF)

gridfiltertomo: $(SRCDIR)filterGridfromTomo.c
	$(CC) $(SRCDIR)filterGridfromTomo.c -o $(BINDIR)filterGridfromTomo $(CFLAGS)

fitorientation: $(SRCDIR)FitOrientation.c
	$(CC) $(SRCDIR)FitOrientation.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientation $(CFLAGS) $(CFLAGSNLOPT)

comparenf: $(SRCDIR)compareNF.c
	$(CC) $(SRCDIR)compareNF.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)compareNF $(CFLAGS) $(CFLAGSNLOPT)

simulatenf: $(SRCDIR)simulateNF.c
	$(CC) $(SRCDIR)simulateNF.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)simulateNF $(CFLAGS) #$(CFLAGSNLOPT)

fitorientationomp: $(SRCDIR)FitOrientationOMP.c
	$(CC) $(SRCDIR)FitOrientationOMP.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientationOMP -fopenmp $(CFLAGS) $(CFLAGSNLOPT)

fitorientationsinglepoint: $(SRCDIR)FitOrientationSinglePoint.c
	$(CC) $(SRCDIR)FitOrientationSinglePoint.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c \
	-o $(BINDIR)FitOrientationSinglePoint $(CFLAGS) $(CFLAGSNLOPT)

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

nfgrainscalc: $(SRCDIR)NFGrainsCalc.c
	$(CC) $(SRCDIR)NFGrainsCalc.c -shared -Wl,-soname,NFGrainsCalc -o $(BINDIR)NFGrainsCalc.so -g -fPIC -ldl -lm -fgnu89-inline -O3 -w

mainnfgrainscalc: $(SRCDIR)NFGrainsCalc.c
	$(CC) $(SRCDIR)NFGrainsCalc.c -o $(BINDIR)NFGrainsCalcMain -g -fPIC -ldl -lm -fgnu89-inline -O3 -w

clean:
	rm -rf $(BINDIR)
	mkdir $(BINDIR)
