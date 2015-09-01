# The make file to compile everything
# To run, give the path for NLOPT, or give nothing if it is installed by default.
# eg. on orthros: NLOPT="/clhome/TOMO1/PeaksAnalysisHemant/NF_HEDM/NLOPT/"

CC=gcc
NCC=nvcc
NCFLAGS=-w -arch sm_52
CFLAGS=-fPIC -ldl -lm -finline-functions -O3 -w
NLOPT=""
ifneq ($(NLOPT),"")
	CFLAGSNLOPT=-I$(NLOPT)/include -L$(NLOPT)/lib -lnlopt
else
	CFLAGSNLOPT=-lnlopt
endif
TIFF=-ltiff
SRCDIR=src/
BINDIR=bin/

all: makediffrspots makediffrspotsff2nf makehexgrid medianimage imageprocessing fitorientation fitorientationff2nf fitorientationparameters fitorientationparametersmultipoint \
	genseedorientations simulatediffrspots simulatediffrspotslatc convertbinfiles processnfmicremote mmapimageinfo hkls parsemic

makehexgrid: $(SRCDIR)MakeHexGrid.c
	$(CC) $(SRCDIR)MakeHexGrid.c -o $(BINDIR)MakeHexGrid $(CFLAGS)

makediffrspots: $(SRCDIR)MakeDiffrSpots.c
	$(CC) $(SRCDIR)MakeDiffrSpots.c -o $(BINDIR)MakeDiffrSpots $(CFLAGS)

makediffrspotsff2nf: $(SRCDIR)MakeDiffrSpotsFF2NF.c
	$(CC) $(SRCDIR)MakeDiffrSpotsFF2NF.c -o $(BINDIR)MakeDiffrSpotsFF2NF $(CFLAGS)

medianimage: $(SRCDIR)MedianImageLibTiff.c
	$(CC) $(SRCDIR)MedianImageLibTiff.c -o $(BINDIR)MedianImageLibTiff $(CFLAGS) $(TIFF)

imageprocessing: $(SRCDIR)ImageProcessingLibTiff.c
	$(CC) $(SRCDIR)ImageProcessingLibTiff.c -o $(BINDIR)ImageProcessingLibTiff $(CFLAGS) $(TIFF)

fitorientation: $(SRCDIR)FitOrientation.c
	$(CC) $(SRCDIR)FitOrientation.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c -o $(BINDIR)FitOrientation $(CFLAGS) $(CFLAGSNLOPT)

fitorientationff2nf: $(SRCDIR)FitOrientationFF2NF.c
	$(CC) $(SRCDIR)FitOrientationFF2NF.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c -o $(BINDIR)FitOrientationFF2NF $(CFLAGS) $(CFLAGSNLOPT)

fitorientationparameters: $(SRCDIR)FitOrientationParameters.c
	$(CC) $(SRCDIR)FitOrientationParameters.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c -o $(BINDIR)FitOrientationParameters $(CFLAGS) $(CFLAGSNLOPT)

fitorientationparametersmultipoint: $(SRCDIR)FitOrientationParametersMultiPoint.c
	$(CC) $(SRCDIR)FitOrientationParametersMultiPoint.c $(SRCDIR)CalcDiffractionSpots.c $(SRCDIR)SharedFuncsFit.c -o $(BINDIR)FitOrientationParametersMultiPoint $(CFLAGS) $(CFLAGSNLOPT)

genseedorientations: $(SRCDIR)GenSeedOrientationsFF2NFHEDM.c
	$(CC) $(SRCDIR)GenSeedOrientationsFF2NFHEDM.c -o $(BINDIR)GenSeedOrientationsFF2NFHEDM $(CFLAGS)

simulatediffrspots: $(SRCDIR)SimulateDiffractionSpots.c
	$(CC) $(SRCDIR)SimulateDiffractionSpots.c -o $(BINDIR)SimulateDiffractionSpots $(CFLAGS)

simulatediffrspotslatc: $(SRCDIR)SimulateDiffractionSpotsLatticeParameter.c
	$(CC) $(SRCDIR)SimulateDiffractionSpotsLatticeParameter.c -o $(BINDIR)SimulateDiffractionSpotsLatticeParameter $(CFLAGS)

convertbinfiles: $(SRCDIR)Convert-Bin-L2N.c
	$(CC) $(SRCDIR)Convert-Bin-L2N.c -o $(BINDIR)ConvertBinFiles $(CFLAGS)

processnfmicremote: $(SRCDIR)ProcessNFMicRemote.c
	$(CC) $(SRCDIR)ProcessNFMicRemote.c -o $(BINDIR)ProcessNFMicRemote $(CFLAGS)

mmapimageinfo: $(SRCDIR)MMapImageInfo.c
	$(CC) $(SRCDIR)MMapImageInfo.c -o $(BINDIR)MMapImageInfo $(CFLAGS)

hkls: $(SRCDIR)GetHKLList.c
	$(CC) $(SRCDIR)GetHKLList.c $(SRCDIR)sgclib.c $(SRCDIR)sgfind.c $(SRCDIR)sghkl.c $(SRCDIR)sgsi.c $(SRCDIR)sgio.c -o $(BINDIR)GetHKLList $(CFLAGS)

parsemic: $(SRCDIR)ParseMic.c
	$(CC) $(SRCDIR)ParseMic.c -o $(BINDIR)ParseMic $(CFLAGS)

clean:
	rm -r $(BINDIR)
	mkdir $(BINDIR)
