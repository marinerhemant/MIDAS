CC=gcc
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall
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

all: bindircheck detectormapper integratorpeakfitomp

bindircheck:
	mkdir -p $(BINDIR)

detectormapper: $(SRCDIR)DetectorMapper.c
	$(CC) $(SRCDIR)DetectorMapper.c -o $(BINDIR)DetectorMapper $(CFLAGS)

integratorpeakfitomp: $(SRCDIR)IntegratorPeakFitOMP.c
	$(CC) $(SRCDIR)IntegratorPeakFitOMP.c -o $(BINDIR)IntegratorPeakFitOMP -fopenmp $(CFLAGS) $(CFLAGSNLOPT) $(CFLAGSTIFF)

peakfit:$(SRCDIR)PeakFit.c
	$(CC) $(SRCDIR)PeakFit.c -o $(BINDIR)PeakFit -fopenmp $(CFLAGS) $(CFLAGSNLOPT) $(CFLAGSTIFF)
