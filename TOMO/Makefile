CC=gcc
CFLAGS=-fPIC -ldl -lm -fgnu89-inline -O3 -w -Wall
CDFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall
SRCDIR=src/
BINDIR=bin/
CFLAGSFFT=-I$${HOME}/.MIDAS/FFTW/include -L$${HOME}/.MIDAS/FFTW/lib -lfftw3f

all: bindircheck tomo tomodebug

bindircheck:
	mkdir -p $(BINDIR)

tomo: $(SRCDIR)/tomo_init.c
	$(CC) $(SRCDIR)/tomo_init.c $(SRCDIR)/tomo_gridrec.c $(SRCDIR)/tomo_utils.c -o $(BINDIR)/MIDAS_TOMO $(CFLAGSFFT) $(CFLAGS) -DPI=M_PI -fopenmp

tomodebug: $(SRCDIR)/tomo_init.c
	$(CC) $(SRCDIR)/tomo_init.c $(SRCDIR)/tomo_gridrec.c $(SRCDIR)/tomo_utils.c -o $(BINDIR)/MIDAS_TOMO_DEBUG $(CFLAGSFFT) $(CDFLAGS) -DPI=M_PI -fopenmp
