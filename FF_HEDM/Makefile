CC=gcc
NCC=nvcc
NCFLAGS=-G -g -w -arch sm_52 -lm -rdc=true # -res-usage
CFLAGS=-fPIC -ldl -lm -finline-functions -O3 -w
NLOPT=""
ifneq ($(NLOPT),"")
	CFLAGSNLOPT=-I$(NLOPT)/include -L$(NLOPT)/lib -lnlopt
else
	CFLAGSNLOPT=-lnlopt
endif
SRCDIR=src/
BINDIR=bin/

all: help bindircheck calibrant fittiltbclsdsample fitposorstrains peaksfitting \
	mergeoverlaps calcradius findsaturatedpx\
	mergerings fittiltx fitwedge hkls indexer bindata processgrains indexercuda peakscuda \
	fitposorstrainscuda mergeoverlapsgpu

cuda: help indexercuda peakscuda fitposorstrainscuda mergeoverlapsgpu fittiltbclsdsample \
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

calibrant: $(SRCDIR)Calibrant.c
	$(CC) $(SRCDIR)Calibrant.c $(SRCDIR)CalcPeakProfile.c -o $(BINDIR)Calibrant $(CFLAGS) $(CFLAGSNLOPT) 

fittiltbclsdsample: $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c
	$(CC) $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c -o $(BINDIR)FitTiltBCLsdSample $(CFLAGS) $(CFLAGSNLOPT) 

fitposorstrains: $(SRCDIR)FitPosOrStrains.c
	$(CC) $(SRCDIR)FitPosOrStrains.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitPosOrStrains $(CFLAGS) \
	$(CFLAGSNLOPT) 

peaksfitting: $(SRCDIR)PeaksFittingPerFile.c
	$(CC) $(SRCDIR)PeaksFittingPerFile.c -o $(BINDIR)PeaksFittingPerFile $(CFLAGS) $(CFLAGSNLOPT) 

mergeoverlaps: $(SRCDIR)MergeOverlappingPeaks.c
	$(CC) $(SRCDIR)MergeOverlappingPeaks.c -o $(BINDIR)MergeOverlappingPeaks $(CFLAGS)

calcradius: $(SRCDIR)CalcRadius.c
	$(CC) $(SRCDIR)CalcRadius.c -o $(BINDIR)CalcRadius $(CFLAGS)

findsaturatedpx: $(SRCDIR)FindSaturatedPixels.c
	$(CC) $(SRCDIR)FindSaturatedPixels.c -o $(BINDIR)FindSaturatedPixels $(CFLAGS)

mergerings: $(SRCDIR)MergeMultipleRings.c
	$(CC) $(SRCDIR)MergeMultipleRings.c -o $(BINDIR)MergeMultipleRings $(CFLAGS)

fittiltx: $(SRCDIR)FitTiltX.c
	$(CC) $(SRCDIR)FitTiltX.c -o $(BINDIR)FitTiltX  $(CFLAGS) $(CFLAGSNLOPT) 

fitwedge: $(SRCDIR)FitWedge.c
	$(CC) $(SRCDIR)FitWedge.c -o $(BINDIR)FitWedge  $(CFLAGS) $(CFLAGSNLOPT) 

hkls: $(SRCDIR)GetHKLList.c
	$(CC) $(SRCDIR)GetHKLList.c $(SRCDIR)sgclib.c $(SRCDIR)sgfind.c $(SRCDIR)sghkl.c $(SRCDIR)sgsi.c \
	$(SRCDIR)sgio.c -o $(BINDIR)GetHKLList $(CFLAGS)

indexer: $(SRCDIR)IndexerLinuxArgsOptimizedShm.c
	$(CC) $(SRCDIR)IndexerLinuxArgsOptimizedShm.c -o $(BINDIR)IndexerLinuxArgsShm $(CFLAGS)

bindata: $(SRCDIR)SaveBinData.c
	$(CC) $(SRCDIR)SaveBinData.c -o $(BINDIR)SaveBinData $(CFLAGS)

processgrains: $(SRCDIR)ProcessGrains.c
	$(CC) $(SRCDIR)ProcessGrains.c $(SRCDIR)GetMisorientation.c $(SRCDIR)CalcStrains.c -o \
	$(BINDIR)ProcessGrains $(CFLAGS) $(CFLAGSNLOPT) 

indexercuda: $(SRCDIR)Indexer.cu
	$(NCC) $(SRCDIR)Indexer.cu -o $(BINDIR)Indexer $(NCFLAGS)

indexrefinecuda: $(SRCDIR)IndexerRefine.cu
	$(NCC) $(SRCDIR)IndexerRefine.cu -o $(BINDIR)IndexRefine $(NCFLAGS)

peakscuda: $(SRCDIR)Peaks.cu
	$(NCC) $(SRCDIR)Peaks.cu -o $(BINDIR)Peaks $(NCFLAGS)

mergeoverlapsgpu: $(SRCDIR)MergeOverlappingPeaksGPU.c
	$(CC) $(SRCDIR)MergeOverlappingPeaksGPU.c -o $(BINDIR)MergeOverlaps $(CFLAGS)

fitposorstrainscuda: $(SRCDIR)FitPosOrStrains.cu
	$(NCC) $(SRCDIR)FitPosOrStrains.cu -o $(BINDIR)RefineGrains $(NCFLAGS)

clean:
	rm -rf $(BINDIR)
	mkdir $(BINDIR)
