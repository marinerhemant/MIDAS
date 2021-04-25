//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Integrator.c
//
// Hemant Sharma
// Dt: 2017/07/26
//
// TODO: 1. Transpose is not working right now.


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
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

#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
#define rad2deg 57.2957795130823

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
    const char * file_name = "/dev/shm/Map.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    int sizelen = 2*(int)sizeof(int) + (int)sizeof(double);
    printf("Map size in bytes: %lld, each element size: %d, total elements: %lld. \n",(long long int)size,sizelen,(long long int)(size/sizelen));
    pxList = mmap (0, size, PROT_READ, MAP_SHARED, fd, 0);
    check (pxList == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));

    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "/dev/shm/nMap.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    nPxList = mmap (0, size2, PROT_READ, MAP_SHARED, fd2, 0);
    printf("nMap size in bytes: %lld, each element size: %d, total elements: %lld. \n",(long long int)size2,2*(int)sizeof(int),2*(long long int)(size2/sizeof(int)));
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

int main(int argc, char **argv)
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    if (argc < 3){
		printf("Usage: ./Integrator ParamFN ImageName (optional)DarkName\n"
		"Optional:\n\tDark file: dark correction with average of all dark frames"
		".\n");
		return(1);
	}
    system("cp Map.bin nMap.bin /dev/shm");
	int rc = ReadBins();
	double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
	int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
	int nEtaBins, nRBins;
    char *ParamFN;
    FILE *paramFile;
    ParamFN = argv[1];
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
	int sumImages=0, separateFolder=0;
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
		str = "EtaBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaBinSize);
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
			sumImages=1;
			continue;
        }
	}
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
	printf("NrTransOpt: %d\n",NrTransOpt);
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 2){printf("TransformationOptions can only be 0, 1, 2.\nExiting.\n");return 0;}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
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
	char *darkFN;
	int nrdone = 0;
	if (argc > 3){
		darkFN = argv[3];
		fd = fopen(darkFN,"rb");
		fseek(fd,0L,SEEK_END);
		sz = ftell(fd);
		rewind(fd);
		nFrames = sz / (SizeFile);
		printf("Reading dark file:      %s, nFrames: %d, skipping first %d bytes.\n",darkFN,nFrames,Skip);
		fseek(fd,Skip,SEEK_SET);
		for (i=0;i<nFrames;i++){
			rc = fileReader(fd,darkFN,dType,NrPixelsY*NrPixelsZ,DarkInT);
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
				printf("Nr mask pixels: %d\n",nrdone);
				makeMap = 0;
			}
			for(j=0;j<NrPixelsY*NrPixelsZ;j++) AverageDark[j] += (double)DarkIn[j]/nFrames;
		}
		printf("Dark file read\n");
	}
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
		printf("Nr mask pixels: %d\n",nrdone);
	}
	char *imageFN;
	imageFN = argv[2];
	fp = fopen(imageFN,"rb");
	fseek(fp,0L,SEEK_END);
	sz = ftell(fp);
	rewind(fp);
	fseek(fp,Skip,SEEK_SET);
	nFrames = sz / SizeFile;
	printf("Number of eta bins: %d, number of R bins: %d. Number of frames in the file: %d\n",nEtaBins,nRBins,(int)nFrames);
	long long int Pos;
	int nPixels, dataPos;
	struct data ThisVal;
	char outfn[4096];
	FILE *out;
	char outFN1d[4096];
	FILE *out1d;
	double Intensity, totArea, ThisInt;
	size_t testPos;
	double RMean, EtaMean;
	double RM1d,Int1d;
	int n1ds;
	double *sumMatrix;
	if (sumImages == 1){
		sumMatrix = calloc(nEtaBins*nRBins*5,sizeof(*sumMatrix));
	}
	for (i=0;i<nFrames;i++){
		printf("Processing frame number: %d of %d of file %s.\n",i+1,nFrames,imageFN);
		rc = fileReader(fp,imageFN,dType,NrPixelsY*NrPixelsZ,ImageInT);
		DoImageTransformations(NrTransOpt,TransOpt,ImageInT,ImageIn,NrPixelsY,NrPixelsZ);
		for (j=0;j<NrPixelsY*NrPixelsZ;j++){
			Image[j] = (double)ImageIn[j] - AverageDark[j];
		}
		if (separateFolder == 0){
			sprintf(outfn,"%s_integrated_framenr_%d.csv",imageFN,i);
			sprintf(outFN1d,"%s_integrated_framenr_%d.1d.csv",imageFN,i);
		} else {
			char fn2[4096];
			sprintf(fn2,"%s",imageFN);
			char *bname;
			bname = basename(fn2);
			sprintf(outfn,"%s/%s_integrated_framenr_%d.csv",outputFolder,bname,i);
			sprintf(outFN1d,"%s/%s_integrated_framenr_%d.1d.csv",outputFolder,bname,i);
		}
		out = fopen(outfn,"w");
		fprintf(out,"%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tEta(degrees)\tIntensity(counts)\tBinArea\n",nEtaBins,nRBins);
		out1d = fopen(outFN1d,"w");
		fprintf(out1d,"%%nRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tIntensity(counts)\n",nRBins);
		for (j=0;j<nRBins;j++){
			RMean = (RBinsLow[j]+RBinsHigh[j])/2;
			Int1d = 0;
			n1ds = 0;
			for (k=0;k<nEtaBins;k++){
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
					ThisInt = Image[testPos]; // The data is arranged as y(fast) and then z(slow)
					//~ ThisInt = *(Image+testPos); // The data is arranged as y(fast) and then z(slow)
					Intensity += ThisInt*ThisVal.frac;
					totArea += ThisVal.frac;
				}
				if (Intensity != 0){
					if (Normalize == 1){
						Intensity /= totArea;
					}
				}
				EtaMean = (EtaBinsLow[k]+EtaBinsHigh[k])/2;
				Int1d += Intensity;
				n1ds ++;
				fprintf(out,"%lf\t%lf\t%lf\t%lf\t%lf\n",RMean,atand(RMean*px/Lsd),EtaMean,Intensity,totArea);
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
			RM1d = RMean;
			Int1d /= n1ds;
			fprintf(out1d,"%lf\t%lf\t%lf\n",RM1d,atand(RM1d*px/Lsd),Int1d);
		}
		fclose(out);
		fclose(out1d);
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
	}
	if (sumImages == 1){
		FILE *sumFile;
		char sumFN[4096];
		if (separateFolder == 0){
			sprintf(sumFN,"%s_sum.csv",imageFN);
		} else {
			char fn2[4096];
			sprintf(fn2,"%s",imageFN);
			char *bname;
			bname = basename(fn2);
			sprintf(sumFN,"%s/%s_sum.csv",outputFolder,bname);
		}
		sumFile = fopen(sumFN,"w");
		fprintf(sumFile,"%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tEta(degrees)\tIntensity(counts)\tBinArea\n");
		for (i=0;i<nRBins*nEtaBins;i++){
			for (k=0;k<4;k++)
				fprintf(sumFile,"%lf\t",sumMatrix[i*5+k]);
			fprintf(sumFile,"\n");
		}
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
	return 0;
}
