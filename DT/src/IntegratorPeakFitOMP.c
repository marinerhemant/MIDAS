//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Integrator.c
//
// Hemant Sharma
// Dt: 2017/07/26
//
// TODO: Add option to give QbinSize instead of RbinSize

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <nlopt.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdint.h>
#include <tiffio.h>
#include <libgen.h>

typedef double pixelvalue;

//~ #define PRINTOPT
#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
#define rad2deg 57.2957795130823
#define maxNFits 300
#define NrValsFitOutput 9

static inline double atand(double x){return rad2deg*(atan(x));}

static void
check (int test, const char * message, ...)
{
    if (test) {
        va_list args;
        va_start (args, message);
        vfprintf (stderr, message, args);
        va_end (args);
        fprintf (stderr, "\n");
        exit (EXIT_FAILURE);
    }
}

static inline
double**
allocMatrix(int nrows, int ncols)
{
    double** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

struct data {
	int y;
	int z;
	double frac;
};

struct data *pxList;
int *nPxList;

int ReadBins(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "Map.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    int sizelen = 2*(int)sizeof(int) + (int)sizeof(double);
    //~ printf("Map size in bytes: %lld, each element size: %d, total elements: %lld. \n",(long long int)size,sizelen,(long long int)(size/sizelen));
    pxList = mmap (0, size, PROT_READ, MAP_SHARED, fd, 0);
    check (pxList == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));

    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "nMap.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    nPxList = mmap (0, size2, PROT_READ, MAP_SHARED, fd2, 0);
    //~ printf("nMap size in bytes: %lld, each element size: %d, total elements: %lld. \n",(long long int)size2,2*(int)sizeof(int),2*(long long int)(size2/sizeof(int)));
    fflush(stdout);
    check (nPxList == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));
	return 1;
}

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

static inline void Transposer (double *x, int n1, int n2, double *y)
{
	int i,j;
	for (i=0;i<n1;i++){
		for (j=0;j<n2;j++){
			y[(i*n2)+j] = x[(j*n1)+i];
		}
	}
}

static inline
void
REtaMapper(
	double Rmin,
	double EtaMin,
	int nEtaBins,
	int nRBins,
	double EtaBinSize,
	double RBinSize,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh)
{
	int i, j, k, l;
	for (i=0;i<nEtaBins;i++){
		EtaBinsLow[i] = EtaBinSize*i      + EtaMin;
		EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
	}
	for (i=0;i<nRBins;i++){
		RBinsLow[i] =  RBinSize * i     + Rmin;
		RBinsHigh[i] = RBinSize * (i+1) + Rmin;
	}
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *ImageIn, pixelvalue *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	int i,j,k,l,m;
	if (NrTransOpt == 0 || (NrTransOpt==1 && TransOpt[0]==0)){
		memcpy(ImageOut,ImageIn,NrPixelsY*NrPixelsZ*sizeof(*ImageIn)); // Nothing to do
		return;
	}
    for (i=0;i<NrTransOpt;i++){
		if (TransOpt[i] == 1){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[l*NrPixelsY+(NrPixelsY-k-1)]; // Invert Y
				}
			}
		}else if (TransOpt[i] == 2){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[(NrPixelsZ-l-1)*NrPixelsY+k]; // Invert Z
				}
			}
		}
	}
}

int fileReader (FILE *f,char fn[], int dType, int NrPixels, double *returnArr)
{
	int i;
	if (dType == 1){ // Binary with uint16
		uint16_t *readData;
		readData = calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 2){ // Binary with double
		double *readData;
		readData = calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 3){ // Binary with float
		float *readData;
		readData = calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 4){ // Binary with uint32
		uint32_t *readData;
		readData = calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 5){ // Binary with int32
		int32_t *readData;
		readData = calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 6){ // TIFF with uint32 format
		TIFFErrorHandler oldhandler;
		oldhandler = TIFFSetWarningHandler(NULL);
		printf("%s\n",fn);
		TIFF* tif = TIFFOpen(fn, "r");
		TIFFSetWarningHandler(oldhandler);
		if (tif){
			uint32 imagelength;
			tsize_t scanline;
			TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imagelength);
			scanline = TIFFScanlineSize(tif);
			tdata_t buf;
			buf = _TIFFmalloc(scanline);
			uint32_t *datar;
			int rnr;
			for (rnr=0;rnr<imagelength;rnr++){
				TIFFReadScanline(tif,buf,rnr,1);
				datar = (uint32_t*)buf;
				for (i=0;i<scanline/sizeof(uint32_t);i++){
					returnArr[rnr*(scanline/sizeof(uint32_t)) + i] = (double) datar[i];
				}
			}
			_TIFFfree(buf);
		}
		return 0;
	} else if (dType == 7){ // TIFF with uint8 format
		TIFFErrorHandler oldhandler;
		oldhandler = TIFFSetWarningHandler(NULL);
		printf("%s\n",fn);
		TIFF* tif = TIFFOpen(fn, "r");
		TIFFSetWarningHandler(oldhandler);
		if (tif){
			uint32 imagelength;
			tsize_t scanline;
			TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imagelength);
			scanline = TIFFScanlineSize(tif);
			tdata_t buf;
			buf = _TIFFmalloc(scanline);
			uint8_t *datar;
			int rnr;
			for (rnr=0;rnr<imagelength;rnr++){
				TIFFReadScanline(tif,buf,rnr,1);
				datar = (uint8_t*)buf;
				for (i=0;i<scanline/sizeof(uint8_t);i++){
					if (datar[i] == 1){
						returnArr[rnr*(scanline/sizeof(uint8_t)) + i] = 1;
					}
				}
			}
			_TIFFfree(buf);
		}
		return 0;
	} else {
		return 127;
	}
}

struct my_profile_func_data{
	int NrPtsForFit;
	double *Rs;
	double *PeakShape;
};

static
double problem_function_profile(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_profile_func_data *f_data = (struct my_profile_func_data *) f_data_trial;
	int NrPtsForFit = f_data->NrPtsForFit;
	double *Rs, *PeakShape;
	Rs = &(f_data->Rs[0]);
	PeakShape = &(f_data->PeakShape[0]);
	double Rcen, Mu, Sigma, FWHM, Imax, BG;
	Rcen = x[0];
	Mu = x[1];
	FWHM = x[2];
	Sigma = FWHM / (2*sqrt(2*log(2)));
	Imax = x[3];
	BG = x[4];
	double TotalDifferenceIntensity=0,CalcIntensity;
	int i,j,k;
	double L, G;
	for (i=0;i<NrPtsForFit;i++){
		L = FWHM/((2*M_PI)*((Rs[i]-Rcen)*(Rs[i]-Rcen) + (FWHM/2)*(FWHM/2)));
		G = exp((-0.5)*(Rs[i]-Rcen)*(Rs[i]-Rcen)/(Sigma*Sigma))/(Sigma*sqrt(2*M_PI));
		CalcIntensity = BG + Imax*((Mu*G)+((1-Mu)*L));
		TotalDifferenceIntensity += (CalcIntensity - PeakShape[i])*(CalcIntensity - PeakShape[i]);
	}
#ifdef PRINTOPT
	printf("Peak profiler intensity difference: %f\n",TotalDifferenceIntensity);
#endif
	return TotalDifferenceIntensity;
}

static
double CalcIntegratedIntensity(
	const double *x,
	void* f_data_trial)
{
	struct my_profile_func_data *f_data = (struct my_profile_func_data *) f_data_trial;
	int NrPtsForFit = f_data->NrPtsForFit;
	double *Rs;
	Rs = &(f_data->Rs[0]);
	double Rcen, Mu, Sigma, FWHM, Imax, BG;
	Rcen = x[0];
	Mu = x[1];
	FWHM = x[2];
	Sigma = FWHM / (2*sqrt(2*log(2)));
	Imax = x[3];
	BG = x[4];
	double TotalIntensity=0;
	int i,j,k;
	double L, G;
	for (i=0;i<NrPtsForFit;i++){
		L = FWHM/((2*M_PI)*((Rs[i]-Rcen)*(Rs[i]-Rcen) + (FWHM/2)*(FWHM/2)));
		G = exp((-0.5)*(Rs[i]-Rcen)*(Rs[i]-Rcen)/(Sigma*Sigma))/(Sigma*sqrt(2*M_PI));
		TotalIntensity += BG + Imax*((Mu*G)+((1-Mu)*L));
	}
#ifdef PRINTOPT
	printf("Peak fit intensity value: %lf\n",TotalIntensity);
#endif
	return TotalIntensity;
}

void FitPeakShape(int NrPtsForFit, double Rs[NrPtsForFit], double PeakShape[NrPtsForFit],
				double *Rfit, double Rstep, double Rmean)
{
	unsigned n = 5;
	double x[n],xl[n],xu[n];
	struct my_profile_func_data f_data;
	f_data.NrPtsForFit = NrPtsForFit;
	f_data.Rs = &Rs[0];
	f_data.PeakShape = &PeakShape[0];
	double BG0 = (PeakShape[0]+PeakShape[NrPtsForFit-1])/2;
	if (BG0 < 0) BG0=0;
	double MaxI=-100000, TotInt = 0;
	int i;
	for (i=0;i<NrPtsForFit;i++){
		if (PeakShape[i] > MaxI){
			TotInt += PeakShape[i] - BG0;
			MaxI=PeakShape[i];
		}
	}
	x[0] = Rmean; xl[0] = Rs[0];    xu[0] = Rs[NrPtsForFit-1];
	x[1] = 0.5;   xl[1] = 0;        xu[1] = 1;
	x[2] = 1; xl[2] = 0.01;  xu[2] = 500; // This is now FWHM
	x[3] = MaxI;  xl[3] = MaxI/100; xu[3] = MaxI*3;
	x[4] = BG0;   xl[4] = 0;        xu[4] = BG0*3;
	struct my_profile_func_data *f_datat;
	f_datat = &f_data;
	void* trp = (struct my_profile_func_data *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function_profile, trp);
	double minf,MeanDiff;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
	MeanDiff = sqrt(minf)/(NrPtsForFit);
	Rfit[0] = x[0];
	Rfit[1] = x[1];
	Rfit[2] = x[2];
	Rfit[3] = x[3];
	Rfit[4] = x[4];
	Rfit[5] = BG0;
	Rfit[6] = MeanDiff;
	Rfit[7] = CalcIntegratedIntensity(x,trp); // Calculate integrated intensity
	Rfit[8] = TotInt; // Total intensity after removing background
}


int mainFunc(char *ParamFN, char *darkFN, char *imageFN, double *retValArr)
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    //~ system("cp Map.bin nMap.bin /dev/shm");
	double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
	int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
	int nEtaBins, nRBins;
    FILE *paramFile;
	char aline[4096], dummy[4096], *str;
	paramFile = fopen(ParamFN,"r");
	int HeadSize = 8192;
    int NrTransOpt=0;
    long long int GapIntensity=0, BadPxIntensity=0;
    int TransOpt[10];
    int makeMap = 0;
    size_t mapMaskSize = 0;
	int *mapMask;
	int dType = 1;
	char GapFN[4096], BadPxFN[4096], outputFolder[4096];
	int sumImages=0, separateFolder=0,newOutput=0, binOutput = 0;
	double radiiToFit[maxNFits][6], etasToFit[maxNFits][4];
	size_t nRadFits = 0, nEtaFits = 0;
	while (fgets(aline,4096,paramFile) != NULL){
		str = "GapFile ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, GapFN);
			makeMap = 2;
		}
		str = "OutFolder ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, outputFolder);
			separateFolder = 1;
		}
		str = "BadPxFile ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, BadPxFN);
			makeMap = 2;
		}
		str = "RadiusToFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf %lf", dummy, &radiiToFit[nRadFits][0], &radiiToFit[nRadFits][1]);
			radiiToFit[nRadFits][2] = radiiToFit[nRadFits][0] - radiiToFit[nRadFits][1];
			radiiToFit[nRadFits][3] = radiiToFit[nRadFits][0] + radiiToFit[nRadFits][1];
			nRadFits++;
		}
		str = "EtaToFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf %lf", dummy, &etasToFit[nEtaFits][0], &etasToFit[nEtaFits][1]);
			etasToFit[nEtaFits][2] = etasToFit[nEtaFits][0] - etasToFit[nEtaFits][1];
			etasToFit[nEtaFits][3] = etasToFit[nEtaFits][0] + etasToFit[nEtaFits][1];
			nEtaFits++;
		}
		str = "EtaBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaBinSize);
		}
		str = "NewOutput ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &newOutput);
		}
		str = "BinOutput ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &binOutput);
		}
		str = "RBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RBinSize);
		}
		str = "DataType ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &dType);
		}
		str = "HeadSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &HeadSize);
		}
		str = "RMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMax);
		}
		str = "RMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMin);
		}
		str = "EtaMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMax);
		}
		str = "EtaMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMin);
		}
		str = "Lsd ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Lsd);
		}
		str = "px ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &px);
		}
		str = "NrPixelsY ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
		}
		str = "NrPixelsZ ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "Normalize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &Normalize);
		}
		str = "NrPixels ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "GapIntensity ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %lld", dummy, &GapIntensity);
            makeMap = 1;
            continue;
        }
		str = "BadPxIntensity ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %lld", dummy, &BadPxIntensity);
            makeMap = 1;
            continue;
        }
        str = "ImTransOpt ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "SumImages ";
        if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &sumImages);
			continue;
        }
	}
	if (separateFolder == 0){
		sprintf(outputFolder,".");
		separateFolder = 1;
	}

	// Let's make an array to store the intensities for fitting peaks
	double *peakIntensities, *peakVals, *AreaMapPixels;
	size_t nFits = nEtaFits * nRadFits;
	size_t iRadFit, iEtaFit, nEls, nElsTot=0;
	for (iRadFit=0;iRadFit<nRadFits;iRadFit++){
		nEls = (int)(ceil(radiiToFit[iRadFit][1]*2 / RBinSize)) + 2;
		radiiToFit[iRadFit][4] = nElsTot;
		radiiToFit[iRadFit][5] = nEls;
		nElsTot += nEls;
	}
	peakIntensities = calloc(nElsTot*nEtaFits,sizeof(*peakIntensities));
	AreaMapPixels = calloc(NrPixelsY*NrPixelsZ,sizeof(*AreaMapPixels));

	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int)ceil((EtaMax - EtaMin)/EtaBinSize);
	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = malloc(nRBins*sizeof(*RBinsHigh));
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);

	int i,j,k,l;
	//~ printf("NrTransOpt: %d\n",NrTransOpt);
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 2){printf("TransformationOptions can only be 0, 1, 2.\nExiting.\n");return 0;}
        //~ printf("TransformationOptions: %d ",TransOpt[i]);
        //~ if (TransOpt[i] == 0) printf("No change.\n");
        //~ else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        //~ else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
    }
	double *Image;
	pixelvalue *ImageIn;
	pixelvalue *DarkIn;
	pixelvalue *ImageInT;
	pixelvalue *DarkInT;
	double *AverageDark;
	DarkIn = malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkIn));
	DarkInT = malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkInT));
	AverageDark = calloc(NrPixelsY*NrPixelsZ,sizeof(*AverageDark));
	ImageIn = malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageIn));
	ImageInT = malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageInT));
	Image = malloc(NrPixelsY*NrPixelsZ*sizeof(*Image));
	size_t pxSize;
	if (dType == 1){ // Uint16
		pxSize = sizeof(uint16_t);
	} else if (dType == 2){ // Double
		pxSize = sizeof(double);
	} else if (dType == 3){ // Float
		pxSize = sizeof(float);
	} else if (dType == 4){ // Uint32
		pxSize = sizeof(uint32_t);
	} else if (dType == 5){ // Int32
		pxSize = sizeof(int32_t);
	} else if (dType == 6){ // Tiff Uint32
		pxSize = sizeof(uint32_t);
		HeadSize = 0;
	} else if (dType == 7){ // Tiff Uint8
		pxSize = sizeof(uint8_t);
		HeadSize = 0;
	}
	size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
	int nFrames;
	size_t sz;
	int Skip = HeadSize;
	FILE *fp, *fd;
	int nrdone = 0;
	fd = fopen(darkFN,"rb");
	fseek(fd,0L,SEEK_END);
	sz = ftell(fd);
	rewind(fd);
	nFrames = sz / (SizeFile);
	//~ printf("Reading dark file:      %s, nFrames: %d, skipping first %d bytes.\n",darkFN,nFrames,Skip);
	fseek(fd,Skip,SEEK_SET);
	for (i=0;i<nFrames;i++){
		int retCode = fileReader(fd,darkFN,dType,NrPixelsY*NrPixelsZ,DarkInT);
		DoImageTransformations(NrTransOpt,TransOpt,DarkInT,DarkIn,NrPixelsY,NrPixelsZ);
		if (makeMap == 1){
			mapMaskSize = NrPixelsY;
			mapMaskSize *= NrPixelsZ;
			mapMaskSize /= 32;
			mapMaskSize ++;
			mapMask = calloc(mapMaskSize,sizeof(*mapMask));
			for (j=0;j<NrPixelsY*NrPixelsZ;j++){
				if (DarkIn[j] == (pixelvalue) GapIntensity || DarkIn[j] == (pixelvalue) BadPxIntensity){
					SetBit(mapMask,j);
					nrdone++;
				}
			}
			//~ printf("Nr mask pixels: %d\n",nrdone);
			makeMap = 0;
		}
		for(j=0;j<NrPixelsY*NrPixelsZ;j++) AverageDark[j] += (double)DarkIn[j]/nFrames;
	}
	//~ printf("Dark file read\n");
	if (makeMap == 2){
		mapMaskSize = NrPixelsY;
		mapMaskSize *= NrPixelsZ;
		mapMaskSize /= 32;
		mapMaskSize ++;
		mapMask = calloc(mapMaskSize,sizeof(*mapMask));
		double *mapper;
		mapper = calloc(NrPixelsY*NrPixelsZ,sizeof(*mapper));
		double *mapperOut;
		mapperOut = calloc(NrPixelsY*NrPixelsZ,sizeof(*mapperOut));
		fileReader(fd,GapFN,7,NrPixelsY*NrPixelsZ,mapper);
		DoImageTransformations(NrTransOpt,TransOpt,mapper,mapperOut,NrPixelsY,NrPixelsZ);
		for (i=0;i<NrPixelsY*NrPixelsZ;i++){
			if (mapperOut[i] != 0){
				SetBit(mapMask,i);
				mapperOut[i] = 0;
				nrdone++;
			}
		}
		fileReader(fd,BadPxFN,7,NrPixelsY*NrPixelsZ,mapper);
		DoImageTransformations(NrTransOpt,TransOpt,mapper,mapperOut,NrPixelsY,NrPixelsZ);
		for (i=0;i<NrPixelsY*NrPixelsZ;i++){
			if (mapperOut[i] != 0){
				SetBit(mapMask,i);
				mapperOut[i] = 0;
				nrdone++;
			}
		}
		//~ printf("Nr mask pixels: %d\n",nrdone);
	}
	fp = fopen(imageFN,"rb");
	fseek(fp,0L,SEEK_END);
	sz = ftell(fp);
	rewind(fp);
	fseek(fp,Skip,SEEK_SET);
	nFrames = (sz-Skip) / SizeFile;
	//~ printf("Number of eta bins: %d, number of R bins: %d. Number of frames in the file: %d\n",nEtaBins,nRBins,(int)nFrames);
	long long int Pos;
	int nPixels, dataPos;
	struct data ThisVal;
	char outfn[4096];
	char outfn2[4096];
	FILE *out,*out2;
	char outFN1d[4096];
	char dmyt[10000];
	FILE *out1d;
	double Intensity, totArea, ThisInt;
	size_t testPos;
	double RMean, EtaMean;
	double Int1d;
	int n1ds;
	double *sumMatrix;
	if (sumImages == 1){
		sumMatrix = calloc(nEtaBins*nRBins*5,sizeof(*sumMatrix));
	}
	double *outArr, *outThisArr, *out1dArr;
	char *outext;
	outext = ".csv";
	size_t bigArrSize = nEtaBins*nRBins;
	double *IntArrPerFrame;
	IntArrPerFrame = calloc(bigArrSize,sizeof(*IntArrPerFrame));
	FILE *out3, *outPeakFit;
	int etaFitNr, radFitNr, radBinFitNr,nEtaF;
	size_t PeakIntPos, peakPos, nElsThis, peakValArrPos;
	double *RPosArr, *RFitThis, *IntensityThis, *ResultArr;
	RPosArr = calloc(nElsTot,sizeof(*RPosArr));
	ResultArr = malloc(NrValsFitOutput*sizeof(*ResultArr));
	peakVals = calloc(nFits*NrValsFitOutput*nFrames,sizeof(*peakVals));
	if (newOutput == 1){
		char outfnAll[4096];
		char outfnFit[4096];
		char fn3[4096];
		sprintf(fn3,"%s",imageFN);
		char *bname3;
		bname3 = basename(fn3);
		sprintf(outfnAll,"%s/%s_integrated.bin",outputFolder,bname3,outext);
		sprintf(outfnFit,"%s/%s_PeakFits.bin",outputFolder,bname3,outext);
		//~ printf("%s\n",outfnAll);
		out3 = fopen(outfnAll,"wb");
		outPeakFit = fopen(outfnFit,"wb");
	}
	printf("Processing file %s\n",imageFN);
	for (i=0;i<nFrames;i++){
		int rcode = fileReader(fp,imageFN,dType,NrPixelsY*NrPixelsZ,ImageInT);
		DoImageTransformations(NrTransOpt,TransOpt,ImageInT,ImageIn,NrPixelsY,NrPixelsZ);
		for (j=0;j<NrPixelsY*NrPixelsZ;j++){
			Image[j] = (double)ImageIn[j] - AverageDark[j];
		}
		if (newOutput == 0){
			if (separateFolder == 0){
				sprintf(outfn,"%s_integrated_framenr_%d%s",imageFN,i,outext);
				sprintf(outFN1d,"%s_integrated_framenr_%d.1d%s",imageFN,i,outext);
			} else {
				char fn2[4096];
				sprintf(fn2,"%s",imageFN);
				char *bname;
				bname = basename(fn2);
				sprintf(outfn,"%s/%s_integrated_framenr_%d%s",outputFolder,bname,i,outext);
				sprintf(outFN1d,"%s/%s_integrated_framenr_%d.1d%s",outputFolder,bname,i,outext);
			}
			out = fopen(outfn,"w");
			out1d = fopen(outFN1d,"w");
			fprintf(out1d,"%%nRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tIntensity(counts)\n",nRBins);
			fprintf(out,"%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tEta(degrees)\tIntensity(counts)\tBinArea\n",nEtaBins,nRBins);
		}
		if (i==0 && newOutput==1){
			if (separateFolder==0) sprintf(outfn2,"%s.REtaAreaMap.csv",imageFN);
			else{
				char fn2[4096];
				sprintf(fn2,"%s",imageFN);
				char *bnname;
				bnname = basename(fn2);
				sprintf(outfn2,"%s/%s.REtaAreaMap.csv",outputFolder,bnname);
			}
			out2 = fopen(outfn2,"w");
			fprintf(out2,"%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tEta(degrees)\tBinArea\n",nEtaBins,nRBins);
		}
		memset(IntArrPerFrame,0,bigArrSize);
		memset(peakIntensities,0,nElsTot*nEtaFits);
		for (j=0;j<nRBins;j++){
			RMean = (RBinsLow[j]+RBinsHigh[j])/2;
			Int1d = 0;
			n1ds = 0;
			radFitNr = -1;
			for (k=0;k<nRadFits;k++){
				if (RBinsHigh[j] >= radiiToFit[k][2] && RBinsLow[j] <= radiiToFit[k][3]){
					radFitNr = k;
					radBinFitNr = (int) ((RBinsHigh[j]-radiiToFit[k][2])/RBinSize);
					RPosArr[(int)radiiToFit[k][4]+ radBinFitNr] = RMean;
					if (radBinFitNr >= (int)radiiToFit[k][5]){
						printf("Something went wrong in fitting calculation, exiting %d %lf.\n",radBinFitNr,radiiToFit[k][5]);
						return 1;
					}
				}
			}
			for (k=0;k<nEtaBins;k++){
				etaFitNr = -1;
				if (radFitNr > -1){
					for (nEtaF=0;nEtaF<nEtaFits;nEtaF++){
						if (EtaBinsHigh[k] >= etasToFit[nEtaF][2] && EtaBinsLow[k] <= etasToFit[nEtaF][3]){
							etaFitNr = nEtaF;
						}
					}
				}
				Pos = j*nEtaBins + k;
				nPixels = nPxList[2*Pos + 0];
				dataPos = nPxList[2*Pos + 1];
				Intensity = 0;
				totArea = 0;
				for (l=0;l<nPixels;l++){
					ThisVal = pxList[dataPos + l];
					testPos = ThisVal.z;
					testPos *= NrPixelsY;
					testPos += ThisVal.y;
					if (mapMaskSize!=0){
						if (TestBit(mapMask,testPos)){
							continue;
						}
					}
					if (i==0){
						AreaMapPixels[testPos] += ThisVal.frac;
					}
					ThisInt = Image[testPos]; // The data is arranged as y(fast) and then z(slow)
					Intensity += ThisInt*ThisVal.frac;
					totArea += ThisVal.frac;
				}
				if (Intensity != 0){
					if (Normalize == 1){
						Intensity /= totArea;
					}
				}
				if (etaFitNr > -1){
					PeakIntPos = nElsTot;
					PeakIntPos *= etaFitNr;
					PeakIntPos += (int)radiiToFit[radFitNr][4];
					PeakIntPos += radBinFitNr;
					peakIntensities[PeakIntPos] += Intensity;
				}
				EtaMean = (EtaBinsLow[k]+EtaBinsHigh[k])/2;
				Int1d += Intensity;
				n1ds ++;
				if (newOutput == 0){
					fprintf(out,"%lf\t%lf\t%lf\t%lf\t%lf\n",RMean,atand(RMean*px/Lsd),EtaMean,Intensity,totArea);
				}else{
					if (i==0){
						fprintf(out2,"%lf\t%lf\t%lf\t%lf\n",RMean,atand(RMean*px/Lsd),EtaMean,totArea);
					}
					IntArrPerFrame[j*nEtaBins+k] = Intensity;
				}
				if (sumImages==1){
					if (i==0){
						sumMatrix[j*nEtaBins*5+k*5+0] = RMean;
						sumMatrix[j*nEtaBins*5+k*5+1] = atand(RMean*px/Lsd);
						sumMatrix[j*nEtaBins*5+k*5+2] = EtaMean;
						sumMatrix[j*nEtaBins*5+k*5+4] = totArea;
					}
					sumMatrix[j*nEtaBins*5+k*5+3] += Intensity;
				}
			}
			Int1d /= n1ds;
			if (newOutput == 0) fprintf(out1d,"%lf\t%lf\t%lf\n",RMean,atand(RMean*px/Lsd),Int1d);
		}
		// Do peak fitting here
		for (j=0;j<nEtaFits;j++){
			for (k=0;k<nRadFits;k++){
				RMean = radiiToFit[k][0];
				peakPos = j;
				peakPos *= nElsTot;
				peakPos += (int)radiiToFit[k][4];
				RFitThis = &RPosArr[(int)radiiToFit[k][4]];
				IntensityThis = &peakIntensities[peakPos];
				nElsThis = (int)radiiToFit[k][5];
				FitPeakShape(nElsThis,RFitThis,IntensityThis,ResultArr,RBinSize,RMean);
				for (l=0;l<NrValsFitOutput;l++){
					// Arranged as each parameter, then each fit, then each frame: size of each parameter: nFits*nFrames, size of each fit: nFrames
					peakValArrPos = l;
					peakValArrPos *= nEtaFits;
					peakValArrPos *= nRadFits;
					peakValArrPos *= nFrames;
					peakValArrPos += k*nEtaFits*nFrames;
					peakValArrPos += j*nFrames;
					peakValArrPos += i;
					peakVals[peakValArrPos] = ResultArr[l];
				}
			}
		}
		if (newOutput == 1){
			fwrite(IntArrPerFrame,bigArrSize*sizeof(*IntArrPerFrame),1,out3);
			if (i==0){
				FILE *areamap;
				char areamapfn[4096];
				sprintf(areamapfn,"%s/AreaFractionPixels.bin",outputFolder);
				areamap = fopen(areamapfn,"wb");
				fwrite(AreaMapPixels,NrPixelsY*NrPixelsZ*sizeof(double),1,areamap);
				fclose(areamap);
				fclose(out2);
			}
		} else{
			fclose(out);
			fclose(out1d);
		}
	}
	if (newOutput == 1){
		for (i=0;i<nFits*NrValsFitOutput;i++){
			for (j=0;j<nFrames;j++){
				retValArr[i*nFrames+j] = peakVals[i*nFrames+j];
			}
		}
		fwrite(peakVals,nFits*NrValsFitOutput*nFrames*sizeof(*peakVals),1,outPeakFit);
		fclose(out3);
		fclose(outPeakFit);
	}
	if (sumImages == 1){
		FILE *sumFile;
		char sumFN[4096];
		if (separateFolder == 0){
			sprintf(sumFN,"%s_sum%s",imageFN,outext);
		} else {
			char fn2[4096];
			sprintf(fn2,"%s",imageFN);
			char *bname;
			bname = basename(fn2);
			sprintf(sumFN,"%s/%s_sum%s",outputFolder,bname,outext);
		}
		sumFile = fopen(sumFN,"w");
		if (newOutput == 0){
			fprintf(sumFile,"%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tEta(degrees)\tIntensity(counts)\tBinArea\n");
			for (i=0;i<nRBins*nEtaBins;i++){
				for (k=0;k<5;k++)
					fprintf(sumFile,"%lf\t",sumMatrix[i*5+k]);
				fprintf(sumFile,"\n");
			}
		} else {
			fprintf(sumFile,"%%Intensity(counts)\n");
			for (i=0;i<nRBins*nEtaBins;i++){
				fprintf(sumFile,"%lf\n",sumMatrix[i*5+3]);
			}
		}
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	//~ printf("Total time elapsed:\t%f s.\n",diftotal);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 10){
		printf("Usage: ./IntegratorPeakFitOMP ParamFN FileStem(fullpath until _before digits) StartNr EndNr Padding ext darkFN nFrames numProcs \n");
		return 1;
	}
	char *pfn = argv[1];
	char *FileStem = argv[2];
	int startNr = atoi(argv[3]);
	int endNr = atoi(argv[4]);
	int pad = atoi(argv[5]);
	char *ext = argv[6];
	char *darkFN = argv[7];
	int nFrames = atoi(argv[8]);
	int numProcs = atoi(argv[9]);
	int frameNr;

	// Get Number of Sinos
	double radiiToFit[maxNFits][6], etasToFit[maxNFits][4];
	size_t nRadFits = 0, nEtaFits = 0;
	FILE *paramFile;
	char aline[4096], dummy[4096], *str, outputFolder[4096];
	int separateFolder = 0;
	paramFile = fopen(pfn,"r");
	while (fgets(aline,4096,paramFile) != NULL){
		str = "RadiusToFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf %lf", dummy, &radiiToFit[nRadFits][0], &radiiToFit[nRadFits][1]);
			radiiToFit[nRadFits][2] = radiiToFit[nRadFits][0] - radiiToFit[nRadFits][1];
			radiiToFit[nRadFits][3] = radiiToFit[nRadFits][0] + radiiToFit[nRadFits][1];
			nRadFits++;
		}
		str = "EtaToFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf %lf", dummy, &etasToFit[nEtaFits][0], &etasToFit[nEtaFits][1]);
			etasToFit[nEtaFits][2] = etasToFit[nEtaFits][0] - etasToFit[nEtaFits][1];
			etasToFit[nEtaFits][3] = etasToFit[nEtaFits][0] + etasToFit[nEtaFits][1];
			nEtaFits++;
		}
		str = "OutFolder ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, outputFolder);
			separateFolder = 1;
		}
	}
	if (separateFolder == 0){
		sprintf(outputFolder,".");
		separateFolder = 1;
	}

	size_t totalNrSinos = nRadFits*nEtaFits*NrValsFitOutput;
	int nFiles = (endNr - startNr + 1);
	size_t arrSize = totalNrSinos;
	arrSize *= nFrames;
	arrSize *= nFiles;
	double *SinoArr;
	SinoArr = calloc(arrSize,sizeof(*SinoArr));

	int rc = ReadBins();
	#pragma omp parallel for num_threads(numProcs) private(frameNr) schedule(dynamic)
	for (frameNr=startNr;frameNr<=endNr;frameNr++)
	{
		char FN[4096];
		sprintf(FN,"%s_%0*d%s",FileStem,pad,frameNr,ext);
		double *thisArr;
		size_t loc = frameNr - startNr;
		loc *= totalNrSinos;
		loc *= nFrames;
		thisArr = &SinoArr[loc];
		int rt = mainFunc(pfn,darkFN,FN,thisArr);
	}
	double *SinoArrArranged;
	SinoArrArranged = malloc(arrSize*sizeof(*SinoArrArranged));
	size_t iVal, iRad, iEta, iFrame, iFile, posArranged, posArr;
	for (iVal=0;iVal<NrValsFitOutput;iVal++){
		for (iRad=0;iRad<nRadFits;iRad++){
			for (iEta=0;iEta<nEtaFits;iEta++){
				for (iFrame=0;iFrame<nFrames;iFrame++){
					for (iFile=0;iFile<nFiles;iFile++){
						posArranged  =  iVal*nRadFits*nEtaFits*nFrames*nFiles;
						posArranged +=           iRad*nEtaFits*nFrames*nFiles;
						posArranged +=                    iEta*nFrames*nFiles;
						posArranged +=                          iFrame*nFiles;
						posArranged +=                                  iFile;
						posArr  = iFile*NrValsFitOutput*nRadFits*nEtaFits*nFrames;
						posArr +=                  iVal*nRadFits*nEtaFits*nFrames;
						posArr +=                           iRad*nEtaFits*nFrames;
						posArr +=                                    iEta*nFrames;
						posArr +=                                          iFrame;
						SinoArrArranged[posArranged] = SinoArr[posArr];
					}
				}
			}
		}
	}
	// What's needed: separateFolder. Then save the sinos with the appropriate fileStemBaseName
	char SinoBaseName[4096],*fStemBaseName, outFN[4096];
	char *valTypes[NrValsFitOutput];
	valTypes[0] = "RMEAN";
	valTypes[1] = "MixFactor";
	valTypes[2] = "FWHM";
	valTypes[3] = "MaxInt";
	valTypes[4] = "BGFit";
	valTypes[5] = "BGSimple";
	valTypes[6] = "MeanError";
	valTypes[7] = "FitIntegratedIntensity";
	valTypes[8] = "TotalIntensityBackgroundCorr";
	FILE *outFile;
	fStemBaseName = basename(FileStem);
	sprintf(SinoBaseName,"%s/%s",outputFolder,fStemBaseName);
	size_t sinoSize = nFrames*nFiles*sizeof(double);
	for (iVal=0;iVal<NrValsFitOutput;iVal++){
		for (iRad=0;iRad<nRadFits;iRad++){
			for (iEta=0;iEta<nEtaFits;iEta++){
				sprintf(outFN,"%s_%s_RadRange_%d_EtaRange_%d.bin",SinoBaseName,valTypes[iVal],iRad,iEta);
				outFile = fopen(outFN,"wb");
				fwrite(SinoArrArranged,sinoSize,1,outFile);
				fclose(outFile);
			}
		}
	}
}
