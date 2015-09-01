# The make file to compile everything
# To run, give the path for NLOPT, or give nothing if it is installed by default.
# eg. on orthros: NLOPT="/clhome/TOMO1/PeaksAnalysisHemant/NF_HEDM/NLOPT/"

CC=gcc
CFLAGS=-fPIC -ldl -lm -finline-functions -O3
NLOPT=""
ifneq ($(NLOPT),"")
	CFLAGSNLOPT=-I$(NLOPT)/include -L$(NLOPT)/lib -lnlopt
else
	CFLAGSNLOPT=-lnlopt
endif
SRCDIR=src/
BINDIR=bin/

all: calibrant fittiltbclsdsample

calibrant: $(SRCDIR)Calibrant.c
	$(CC) $(CFLAGS) $(SRCDIR)Calibrant.c $(SRCDIR)CalcPeakProfile.c -o $(BINDIR)Calibrant

fittiltbclsdsample: $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c
	$(CC) $(CFLAGS) $(CFLAGSNLOPT) $(SRCDIR)FitTiltBCLsdSampleOmegaCorrection.c -o $(BINDIR)FitTiltBCLsdSampleOmegaCorrection

fitposorstrains: $(SRCDIR)FitPosOrStrains.c
	$(CC) $(CFLAGS) $(CFLAGSNLOPT) $(SRCDIR)FitPosOrStrains.c $(SRCDIR)CalcDiffractionSpots.c -o $(BINDIR)FitPosOrStrains

peaksfitting: $(SRCDIR)PeaksFittingPerFile.c
	$(CC) $(CFLAGS) $(CFLAGSNLOPT) $(SRCDIR)PeaksFittingPerFile.c -o $(BINDIR)PeaksFittingPerFile


gcc -g -o Calibrant Calibrant.c CalcPeakProfile.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o FitTiltBCLsdSample FitTiltBCLsdSample.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o FitPosOrStrains FitPosOrStrains.c CalcDiffractionSpots.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o FitTiltBCLsdSampleOmegaCorrection FitTiltBCLsdSampleOmegaCorrection.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o PeaksFittingPerFile PeaksFittingPerFile.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o MergeOverlappingPeaks MergeOverlappingPeaks.c -fPIC -ldl -lm
gcc -g -o CalcRadius CalcRadius.c -fPIC -ldl -lm
gcc -g -o FindSaturatedPixels FindSaturatedPixels.c -fPIC -ldl -lm
gcc -g -o MergeMultipleRings MergeMultipleRings.c -finline-functions -fPIC -ldl -lm
gcc -g -o FitTiltX FitTiltX.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o FitWedge FitWedge.c -finline-functions -fPIC -ldl -lm -lnlopt
gcc -g -o GetHKLList GetHKLList.c sgclib.c sgfind.c sghkl.c sgsi.c sgio.c -ldl -lm -O3
gcc -g -o IndexerLinuxArgsShm IndexerLinuxArgsOptimizedShm.c -finline-functions -fPIC -ldl -lm
gcc -g -o SaveBinData SaveBinData.c -ldl -lm -O3 -fPIC
gcc -g -o ProcessGrains ProcessGrains.c GetMisorientation.c CalcStrains.c -finline-functions -fPIC -ldl -lm -lnlopt
