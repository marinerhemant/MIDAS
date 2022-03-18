//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <tiffio.h>
#include <omp.h>

typedef uint16_t pixelvalue;
pixelvalue quick_select(pixelvalue a[], int n) ;

#define PIX_SWAP(a,b) { pixelvalue temp=(a);(a)=(b);(b)=temp; }
pixelvalue quick_select(pixelvalue a[], int n)
{
	int low, high ;
	int median;
	int middle, ll, hh;
	low = 0 ; high = n-1 ; median = (low + high) / 2;
	for (;;) {
		if (high <= low)
			return a[median] ;
		if (high == low + 1) {
			if (a[low] > a[high])
				PIX_SWAP(a[low], a[high]) ;
			return a[median] ;
		}
		middle = (low + high) / 2;
		if (a[middle] > a[high])    PIX_SWAP(a[middle], a[high]) ;
		if (a[low] > a[high])       PIX_SWAP(a[low], a[high]) ;
		if (a[middle] > a[low])     PIX_SWAP(a[middle], a[low]) ;
		PIX_SWAP(a[middle], a[low+1]) ;
		ll = low + 1;
		hh = high;
		for (;;) {
			do ll++; while (a[low] > a[ll]) ;
			do hh--; while (a[hh]  > a[low]) ;
			if (hh < ll)
			break;
			PIX_SWAP(a[ll], a[hh]) ;
		}
		PIX_SWAP(a[low], a[hh]) ;
		if (hh <= median)
			low = ll;
		if (hh >= median)
			high = hh - 1;
	}
}
#undef PIX_SWAP

pixelvalue**
allocMatrixInt(int nrows, int ncols)
{
	pixelvalue** arr;
	int i;
	arr = malloc(nrows * sizeof(*arr));
	for ( i = 0 ; i < nrows ; i++) {
		arr[i] = malloc(ncols * sizeof(*arr[i]));
	}
	return arr;
}

pixelvalue***
allocMatrix3Int(int nrows, int ncols, int nmats)
{
	pixelvalue*** arr;
	int i,j;
	arr = malloc(nrows * sizeof(*arr));
	for ( i = 0 ; i < nrows ; i++) {
		arr[i] = malloc(ncols * sizeof(*arr[i]));
		for (j=0;j<ncols;j++){
			arr[i][j] = malloc(nmats * sizeof(*arr[i][j]));
		}
	}
	return arr;
}

void
FreeMemMatrixInt(pixelvalue **mat,int nrows)
{
	int r;
	for ( r = 0 ; r < nrows ; r++) {
		free(mat[r]);
	}
	free(mat);
}

void
FreeMemMatrix3Int(pixelvalue ***mat,int nrows, int ncols)
{
	int r,c;
	for ( r = 0 ; r < nrows ; r++) {
		for (c=0;c<ncols;c++){
			free(mat[r][c]);
		}
		free(mat[r]);
	}
	free(mat);
}


int CalcMedian(char fn[1000],
	int LayerNr,
	int StartNr,
	int NrPixels,
	int NrFilesPerLayer,
	char ext[1024],
	char extReduced[1024],
	int numProcs)
{
	int i,j,k;
	pixelvalue **AllIntensities, *MedianArray;
	AllIntensities = allocMatrixInt(NrPixels*NrPixels,NrFilesPerLayer);
	char MedianFileName[1024],MaxIntFileName[1024],MaxIntMedianCorrFileName[1024];
	sprintf(MedianFileName,"%s_Median_Background_Distance_%d.%s",fn,LayerNr-1,extReduced);
	sprintf(MaxIntFileName,"%s_MaximumIntensity_Distance_%d.%s",fn,LayerNr-1,extReduced);
	int fld = 0;
	sprintf(MaxIntMedianCorrFileName,"%s_MaximumIntensityMedianCorrected_Distance_%d.%s",fn,LayerNr-1,extReduced);
	printf("Reading files.\n");
	#pragma omp parallel for num_threads(numProcs) private(j) schedule(dynamic)
	for (j=0;j<NrFilesPerLayer;j++){
		printf("File Number: %d out of %d\n",j,NrFilesPerLayer);
		TIFFErrorHandler oldhandler;
		oldhandler = TIFFSetWarningHandler(NULL);
		int FileNr = ((LayerNr - 1) * NrFilesPerLayer) + StartNr + j;
		char FileName[1024];
		sprintf(FileName,"%s_%06d.%s",fn,FileNr,ext);
		TIFF* tif = TIFFOpen(FileName, "r");
		if (tif == NULL){
			printf("%s not found.\n",FileName);
			fld = 1;
			continue;
		}
		TIFFSetWarningHandler(oldhandler);
		int roil;
		if (tif) {
			tdata_t buf;
			buf = _TIFFmalloc(TIFFScanlineSize(tif));
			pixelvalue *datar;
			for (roil=0; roil < NrPixels; roil++){
				TIFFReadScanline(tif, buf, roil, 1);
				datar=(uint16*)buf;
				for (i=0;i<NrPixels;i++){
					AllIntensities[roil*NrPixels+i][j] = datar[i];
				}
			}
			_TIFFfree(buf);
		}
		TIFFClose(tif);
	}
	if (fld == 0) return 0;
	printf("Calculating median.\n");
	MedianArray = malloc(NrPixels*NrPixels*sizeof(*MedianArray));
	pixelvalue *MaxIntArr, *MaxIntMedianArr;
	MaxIntArr = malloc(NrPixels*NrPixels*sizeof(*MaxIntArr));
	MaxIntMedianArr = malloc(NrPixels*NrPixels*sizeof(*MaxIntMedianArr));
	#pragma omp parallel for num_threads(numProcs) private(k) schedule(dynamic)
	for (k=0;k<NrPixels;k++){
		printf("Pixel Nr %d of %d\n",k,NrPixels);
		pixelvalue SubArr[NrFilesPerLayer];
		int tempVal;
		int t;
		for (t=0;t<NrPixels;t++){
			int it;
			it = k*NrPixels + t;
			MaxIntArr[it] = 0;
			MaxIntMedianArr[it] = 0;
			for (j=0;j<NrFilesPerLayer;j++){
				SubArr[j] = AllIntensities[it][j];
				if (AllIntensities[it][j] > MaxIntArr[it]){
					MaxIntArr[it] = AllIntensities[it][j];
				}
			}
			MedianArray[it] = quick_select(SubArr,NrFilesPerLayer);
			tempVal =  (MaxIntArr[it] - MedianArray[it]);
			MaxIntMedianArr[it] = (pixelvalue) (tempVal > 0 ? tempVal : 0);
		}
	}
	int SizeOutFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	int fb = open(MedianFileName, O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
	pwrite(fb,MedianArray,SizeOutFile,0);
	int fMaxInt = open(MaxIntFileName,O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
	pwrite(fMaxInt,MaxIntArr,SizeOutFile,0);
	int fMaxIntMedian = open(MaxIntMedianCorrFileName,O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
	pwrite(fMaxIntMedian,MaxIntMedianArr,SizeOutFile,0);
	printf("Median calculated.\n");
	free(MedianArray);
	free(MaxIntArr);
	free(MaxIntMedianArr);
	FreeMemMatrixInt(AllIntensities,NrPixels);
	return 1;
}


static void
usage(void)
{
    printf("MedianImage: usage: ./MedianImage <ParametersFile> <LayerNr> <numProcs>\n");
}

int
main(int argc, char *argv[])
{
	if (argc != 4)
	{
		usage();
		return 1;
	}
	double start_time = omp_get_wtime();
	// Read params file.
	char *ParamFN;
	FILE *fileParam;
	ParamFN = argv[1];
	char aline[1000];
	char fn2[1000],fn[1000], direct[1000], ext[1000], extReduced[1000];
	fileParam = fopen(ParamFN,"r");
	char *str, dummy[1000];
	int LowNr,nLayers,StartNr,NrFilesPerLayer,NrPixels,WFImages=0;
	nLayers = atoi(argv[2]);
	while (fgets(aline,1000,fileParam)!=NULL){
		str = "RawStartNr ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d", dummy, &StartNr);
			continue;
		}
		str = "DataDirectory ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %s", dummy, direct);
			continue;
		}
		str = "NrPixels ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d", dummy, &NrPixels);
			continue;
		}
		str = "WFImages ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d", dummy, &WFImages);
			continue;
		}
		str = "NrFilesPerDistance ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d", dummy, &NrFilesPerLayer);
			continue;
		}
		str = "OrigFileName ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %s", dummy, fn2);
			continue;
		}
		str = "extOrig ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %s", dummy, ext);
			continue;
		}
		str = "extReduced ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %s", dummy, extReduced);
			continue;
		}
	}
	int numProcs = atoi(argv[3]);
	StartNr = StartNr + (nLayers-1)*WFImages;
	sprintf(fn,"%s/%s",direct,fn2);
	fclose(fileParam);
	int ReturnCode;
	ReturnCode = CalcMedian(fn, nLayers,StartNr,NrPixels,NrFilesPerLayer,ext,extReduced,numProcs);
	if (ReturnCode == 0){
		printf("Median Calculation failed. Exiting.\n");
		return 1;
	}
	double time = omp_get_wtime() - start_time;
	printf("Finished, time elapsed: %lf seconds.\n",time);
	return 0;
}
