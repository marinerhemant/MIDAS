#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
CC=gcc
NCC=nvcc
MPICC=LD_LIBRARY_PATH=$${HOME}/.MIDAS/MPICH/lib:$LD_LIBRARY_PATH $${HOME}/.MIDAS/MPICH/bin/mpicc
NCFLAGS=-O3 -G -g -w -arch sm_52 -lm -rdc=true # -res-usage
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall
MPICCFLAGS=-I$${HOME}/.MIDAS/MPICH/include -L$${HOME}/.MIDAS/MPICH/lib -fopenmp -lmpi -fPIC -g -ldl -lm -fgnu89-inline -O3 -w
CFLAGSNETCDF=-I$${HOME}/.MIDAS/netcdf/include -L$${HOME}/.MIDAS/netcdf/lib -lnetcdf -I$${HOME}/.MIDAS/hdf5/include -L$${HOME}/.MIDAS/hdf5/lib -lhdf5_hl -lhdf5 -I$${HOME}/.MIDAS/zlib/include -L$${HOME}/.MIDAS/zlib/lib -lz
NLOPT=""
ifneq ($(NLOPT),"")
	CFLAGSNLOPT=-I$(NLOPT)/include -L$(NLOPT)/lib -lnlopt
else
	CFLAGSNLOPT=-I$${HOME}/.MIDAS/NLOPT/include -L$${HOME}/.MIDAS/NLOPT/lib -lnlopt
	CFLAGSNLOPTSO=-I$${HOME}/.MIDAS/NLOPTShared/include -L$${HOME}/.MIDAS/NLOPTShared/lib -lnlopt
endif
TIFF=""
ifneq ($(TIFF),"")
	CFLAGSTIFF=-I$(TIFF)/include -L$(TIFF)/lib -ltiff
else
	CFLAGSTIFF=-I$${HOME}/.MIDAS/LIBTIFF/include -L$${HOME}/.MIDAS/LIBTIFF/lib -ltiff
endif
SRCDIR=src/
BINDIR=bin/

all: help bindircheck calibrant imagemax \
	fittiltbclsdsample fitposorstrains peaksfitting \
	mergeoverlaps calcradius findsaturatedpx genmediandark fitgrain tiff2ge \
	mergerings fittiltx fitwedge hkls indexer bindata processgrains graintracking\
	mapmultdetectors matchgrains detectormapper mergemultiplescans \
	fitposorstrainsscanning indexscanning processgrainsscanning mapbnd fitscanninggrain \
	fitgrainhydra forwardsimulation integrator

local: predep all runsetup

orthros: predep all runsetup

biocluster: predep all runsetup

lcrc_cloud: predep all runsetup

nersc: predep all runsetup

purdue: predep all runsetup

lanl: predep all runsetup

lcrc: predep all runsetup

stampede: predep all runsetup

notchpeak: predep all runsetup

upitt: predep all runsetup

runsetup:
	Cluster/setup.sh

predep:
	./GetPackages.sh

cuda: help indexrefinecuda peaks mergeoverlapsgpu fittiltbclsdsample \
	calcradius mergerings processgrains hkls bindata calibrant findsaturatedpx fittiltx fitwedge

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
	@echo ..............................................................................................
	@echo
	@echo $(NLOPT)

scanningfcns: $(SRCDIR)ScanningFunctions.c
	$(CC) $(SRCDIR)ScanningFunctions.c $(SRCDIR)CalcStrains.c -o $(BINDIR)ScanningFunctions $(CFLAGS) $(CFLAGSNLOPT) -fopenmp

simulatescanning: $(SRCDIR)SimulateScanning.c
	$(CC) $(SRCDIR)SimulateScanning.c -o $(BINDIR)SimulateScanning $(CFLAGS) $(CFLAGSNLOPT) -fopenmp

imagemax: $(SRCDIR)imageMax.c
	$(CC) $(SRCDIR)imageMax.c -shared -Wl,-soname,imageMax -o $(BINDIR)imageMax.so -fPIC -ldl -lm -fgnu89-inline -O3 -w

calibrant: $(SRCDIR)Calibrant.c
	$(CC) $(SRCDIR)Calibrant.c $(SRCDIR)CalcPeakProfile.c -o $(BINDIR)Calibrant $(CFLAGS) $(CFLAGSTIFF) $(CFLAGSNLOPT)

fittiltbclsdsample: $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c
	$(CC) $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c -o $(BINDIR)FitTiltBCLsdSample $(CFLAGS) $(CFLAGSNLOPT)

forwardsimulation: $(SRCDIR)ForwardSimulation.c
	$(CC) $(SRCDIR)ForwardSimulation.c -o $(BINDIR)ForwardSimulation $(CFLAGS)

fitposorstrains: $(SRCDIR)FitPosOrStrains.c
	$(CC) $(SRCDIR)FitPosOrStrains.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitPosOrStrains $(CFLAGS) \
	$(CFLAGSNLOPT)

fitposorstrainsomp: $(SRCDIR)FitPosOrStrainsOMP.c
	$(CC) $(SRCDIR)FitPosOrStrainsOMP.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitPosOrStrainsOMP -fopenmp $(CFLAGS) \
	$(CFLAGSNLOPT)

fitposorstrainsscanning: $(SRCDIR)FitPosOrStrainsScanningHEDM.c
	$(CC) $(SRCDIR)FitPosOrStrainsScanningHEDM.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitPosOrStrainsScanningHEDM $(CFLAGS) \
	$(CFLAGSNLOPT)

ff_mpiomp: $(SRCDIR)MIDAS_FF_MPIOMP.c
	$(MPICC) $(SRCDIR)MIDAS_FF_MPIOMP.c $(SRCDIR)sharedFunctions.c -o $(BINDIR)MIDAS_FF_MPIOMP $(MPICCFLAGS) $(CFLAGSNLOPT) $(CFLAGSTIFF)

peaksfitting: $(SRCDIR)PeaksFittingPerFile.c
	$(CC) $(SRCDIR)PeaksFittingPerFile.c -o $(BINDIR)PeaksFittingPerFile $(CFLAGS) $(CFLAGSNLOPT)

peaksfittingomp: $(SRCDIR)PeaksFittingOMP.c
	$(CC) $(SRCDIR)PeaksFittingOMP.c -o $(BINDIR)PeaksFittingOMP -fopenmp $(CFLAGSNLOPT) $(CFLAGS)

mapmultdetectors: $(SRCDIR)MapMultipleDetectors.c
	$(CC) $(SRCDIR)MapMultipleDetectors.c -o $(BINDIR)MapMultipleDetectors $(CFLAGS)

mergeoverlaps: $(SRCDIR)MergeOverlappingPeaks.c
	$(CC) $(SRCDIR)MergeOverlappingPeaks.c -o $(BINDIR)MergeOverlappingPeaks $(CFLAGS)

mergeoverlapsall: $(SRCDIR)MergeOverlappingPeaksAll.c
	$(CC) $(SRCDIR)MergeOverlappingPeaksAll.c -o $(BINDIR)MergeOverlappingPeaksAll $(CFLAGS)

calcradius: $(SRCDIR)CalcRadius.c
	$(CC) $(SRCDIR)CalcRadius.c -o $(BINDIR)CalcRadius $(CFLAGS)

findsaturatedpx: $(SRCDIR)FindSaturatedPixels.c
	$(CC) $(SRCDIR)FindSaturatedPixels.c -o $(BINDIR)FindSaturatedPixels $(CFLAGS)

graintracking: $(SRCDIR)GrainTracking.c
	$(CC) $(SRCDIR)GrainTracking.c -o $(BINDIR)GrainTracking $(CFLAGS)

mergerings: $(SRCDIR)MergeMultipleRings.c
	$(CC) $(SRCDIR)MergeMultipleRings.c -o $(BINDIR)MergeMultipleRings $(CFLAGS)

genmediandark: $(SRCDIR)GenMedianDark.c
	$(CC) $(SRCDIR)GenMedianDark.c -o $(BINDIR)GenMedianDark $(CFLAGS)

fitgrain: $(SRCDIR)FitGrain.c
	$(CC) $(SRCDIR)FitGrain.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitGrain $(CFLAGS) $(CFLAGSNLOPT)

fitgrainhydra: $(SRCDIR)FitGrainHydra.c
	$(CC) $(SRCDIR)FitGrainHydra.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitGrainHydra $(CFLAGS) $(CFLAGSNLOPT)

fitscanninggrain: $(SRCDIR)FitScanningGrain.c
	$(CC) $(SRCDIR)FitScanningGrain.c -o $(BINDIR)FitScanningGrain $(CFLAGS)

fittiltx: $(SRCDIR)FitTiltX.c
	$(CC) $(SRCDIR)FitTiltX.c -o $(BINDIR)FitTiltX  $(CFLAGS) $(CFLAGSNLOPT)

fitwedge: $(SRCDIR)FitWedge.c
	$(CC) $(SRCDIR)FitWedge.c -o $(BINDIR)FitWedge  $(CFLAGS) $(CFLAGSNLOPT)

hkls: $(SRCDIR)GetHKLList.c
	$(CC) $(SRCDIR)GetHKLList.c $(SRCDIR)sgclib.c $(SRCDIR)sgfind.c $(SRCDIR)sghkl.c $(SRCDIR)sgsi.c \
	$(SRCDIR)sgio.c -o $(BINDIR)GetHKLList $(CFLAGS)

indexer: $(SRCDIR)IndexerLinuxArgsOptimizedShm.c
	$(CC) $(SRCDIR)IndexerLinuxArgsOptimizedShm.c -o $(BINDIR)IndexerLinuxArgsShm $(CFLAGS)

indexeromp: $(SRCDIR)IndexerOMP.c
	$(CC) $(SRCDIR)IndexerOMP.c -o $(BINDIR)IndexerOMP -fopenmp $(CFLAGS)

indexscanning: $(SRCDIR)IndexScanningHEDM.c
	$(CC) $(SRCDIR)IndexScanningHEDM.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)IndexScanningHEDM $(CFLAGS)

bindata: $(SRCDIR)SaveBinData.c
	$(CC) $(SRCDIR)SaveBinData.c -o $(BINDIR)SaveBinData $(CFLAGS)

mergemultiplescans: $(SRCDIR)MergeMultipleScans.c
	$(CC) $(SRCDIR)MergeMultipleScans.c -o $(BINDIR)MergeMultipleScans $(CFLAGS)

processgrains: $(SRCDIR)ProcessGrains.c
	$(CC) $(SRCDIR)ProcessGrains.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcStrains.c -o \
	$(BINDIR)ProcessGrains $(CFLAGS) $(CFLAGSNLOPT)

processgrainsscanning: $(SRCDIR)ProcessGrainsScanningHEDM.c
	$(CC) $(SRCDIR)ProcessGrainsScanningHEDM.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcStrains.c -o \
	$(BINDIR)ProcessGrainsScanningHEDM $(CFLAGS) $(CFLAGSNLOPT)

matchgrains: $(SRCDIR)MatchGrains.c
	$(CC) $(SRCDIR)MatchGrains.c $(SRCDIR)GetMisorientation.c -o $(BINDIR)MatchGrains $(CFLAGS)

detectormapper: $(SRCDIR)DetectorMapper.c
	$(CC) $(SRCDIR)DetectorMapper.c -o $(BINDIR)DetectorMapper $(CFLAGS)

integrator: $(SRCDIR)Integrator.c
	$(CC) $(SRCDIR)Integrator.c -o $(BINDIR)Integrator $(CFLAGS) $(CFLAGSTIFF)

indexercuda: $(SRCDIR)IndexerRefineNLOPT.cu
	$(NCC) $(SRCDIR)IndexerRefineNLOPT.cu -o $(BINDIR)Indexer $(NCFLAGS)

tiff2ge: $(SRCDIR)ConvTiffToGE.c
	$(CC) $(SRCDIR)ConvTiffToGE.c -o $(BINDIR)ConfTiffToGE $(CFLAGS)

mapbnd: $(SRCDIR)MapBND.c
	$(CC) $(SRCDIR)MapBND.c -o $(BINDIR)MapBND $(CFLAGS)

indexrefinecuda: $(SRCDIR)IndexerRefine.cu
	$(NCC) $(SRCDIR)IndexerRefine.cu -o $(BINDIR)Indexer $(NCFLAGS)

peaks: $(SRCDIR)PeaksNLMD.cu
	$(NCC) $(SRCDIR)PeaksNLMD.cu -o $(BINDIR)Peaks $(NCFLAGS)

mergeoverlapsgpu: $(SRCDIR)MergeOverlappingPeaksGPU.c
	$(CC) $(SRCDIR)MergeOverlappingPeaksGPU.c -o $(BINDIR)MergeOverlaps $(CFLAGS)

clean:
	rm -rf $(BINDIR)
	mkdir $(BINDIR)
	rm -rf $$HOME/.MIDAS
