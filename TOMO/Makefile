CC=gcc
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w
SRCDIR=src/
BINDIR=bin/
CFLAGSFFT=-I$${HOME}/.MIDAS/FFTW/include -L$${HOME}/.MIDAS/FFTW/lib -lfftw3f

all: bindircheck tomo_init

bindircheck:
	mkdir -p $(BINDIR)

tomo: $(SRCDIR)/tomo_init.c
	$(CC) $(SRCDIR)/tomo_init.c $(SRCDIR)/tomo_gridrec.c $(SRCDIR)/tomo_utils.c -o $(BINDIR)/tomo $(CFLAGSFFT) $(CFLAGS) -DPI=M_PI
