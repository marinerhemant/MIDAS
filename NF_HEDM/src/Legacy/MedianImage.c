//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include "diplib.h"
#include "dipio.h"
#include "dipio_tiff.h"

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
	int BlnketSubtraction)
{
	time_t timer;
    char buffer[25];
    struct tm* tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
    puts(buffer);
	int i,j,k,FileNr;
	char FileName[1024];
	DIP_FN_DECLARE("CalcMedian");
	DIPXJ(dip_Initialise());
    DIPXJ(dipio_Initialise());
	pixelvalue **AllIntensities, *MedianArray, *DifferenceImage;
	AllIntensities = allocMatrixInt(NrPixels*NrPixels,NrFilesPerLayer);
	char MedianFileName[1024];
	sprintf(MedianFileName,"%s_Median_Background_Distance_%d.%s",fn,LayerNr-1,extReduced);
	for (j=0;j<NrFilesPerLayer;j++){
		FileNr = ((LayerNr - 1) * NrFilesPerLayer) + StartNr + j;
		sprintf(FileName,"%s_%06d.%s",fn,FileNr,ext);
	    dip_Resources rg = 0;
	    dip_Image im;
	    dip_String filename;
	    DIPXJ( dip_ResourcesNew (&rg, 0) );
	    DIPXJ( dip_ImageNew (&im, rg) );
	    DIPXJ( dip_StringNew (&filename, 1024, FileName, rg) );
	    DIPXJ( dipio_ImageRead (im, filename, 0, DIP_FALSE, 0) );
	    {
			//printf("Reading file: %s ",FileName);
			dip_IntegerArray dimensions;
			dip_ImageType type;
			dip_DataType dataType;
			dip_int ii;
			DIPXJ( dip_ImageGetDimensions (im, &dimensions, rg) );
			DIPXJ( dip_ImageGetType (im, &type) );
			DIPXJ( dip_ImageGetDataType (im, &dataType) );
			/*printf ("Dimensions: [%d", (int)dimensions->array[0]);
			for (ii = 1; ii < dimensions->size; ii++)
				printf (", %d", (int)dimensions->array[ii]);
			printf ("] ");
			printf ("Image type: %s Data type: %s\n",type == DIP_IMTP_SCALAR ? "Scalar,OK" : "Wrong Format, please check!!!",
				dataType == DIP_DT_UINT16 ? "uint16,OK" : "Wrong Format, please check!!!");*/
	    }
		dip_ImageArray out;
		dip_VoidPointerArray odp;
		dip_uint16* ptr;
		dip_ImageArrayNew( &out, 1, rg );
		out->array[ 0 ] = im;
		dip_ImageGetData( 0, 0, 0, out, &odp, 0, 0, rg );
		ptr = (dip_uint16*)odp->array[ 0 ];
		dip_IntegerArray stride;
		dip_int x, y;
		dip_ImageGetStride( im, &stride, rg );
		for (i=0;i<NrPixels;i++){
			for (k=0;k<NrPixels;k++){
				AllIntensities[k+(NrPixels*i)][j] = *(ptr + k * stride->array[0] + i * stride->array[1]);
			}
		}
	  	dip_error:
	    DIPXC( dip_ResourcesFree (&rg) );
	}
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
    puts(buffer);
	printf("Calculating median.\n");
	MedianArray = malloc(NrPixels*NrPixels*sizeof(*MedianArray));
	pixelvalue SubArr[NrFilesPerLayer];
	for (i=0;i<NrPixels*NrPixels;i++){
		for (j=0;j<NrFilesPerLayer;j++){
			SubArr[j] = AllIntensities[i][j];
		}
		MedianArray[i] = quick_select(SubArr,NrFilesPerLayer);
	}
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
    puts(buffer);
	int SizeOutFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	FILE *fb;
	fb = fopen(MedianFileName,"wb");
	fwrite(MedianArray,SizeOutFile,1,fb);
	fclose(fb);
	printf("Median calculated.\nSaving corrected files.\n");
	int x;
	FILE *ft;
	char DifferenceFileName[1024];
	for (i=0;i<NrFilesPerLayer;i++){
	    DifferenceImage = malloc(NrPixels*NrPixels*sizeof(*DifferenceImage));
		for (j=0;j<NrPixels*NrPixels;j++){
			x = (int)AllIntensities[j][i] - ((int)MedianArray[j] + (int)BlnketSubtraction);
			DifferenceImage[j] = (pixelvalue)(x > 0 ? x : 0);
		}
		FileNr = ((LayerNr - 1) * NrFilesPerLayer) + StartNr + i;
		sprintf(DifferenceFileName,"%s_%06d.%s",fn,FileNr,extReduced);
		//printf("Saving file: %s\n",DifferenceFileName);
		ft = fopen(DifferenceFileName,"wb");
		fwrite(DifferenceImage,SizeOutFile,1,ft);
		fclose(ft);
		free(DifferenceImage);
	}
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
    puts(buffer);
    DIPXC(dipio_Exit());
    DIPXC(dip_Exit());
	free(MedianArray);
	FreeMemMatrixInt(AllIntensities,NrPixels*NrPixels);
	return 1;
}


static void
usage(void)
{
    printf("MedianImage: usage: ./MedianImage <ParametersFile> <LayerNr>\n");
}

int
main(int argc, char *argv[])
{
    if (argc < 3)
    {
        usage();
        return 1;
    }

    clock_t start, end;
    double diftotal;
    start = clock();
    
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
	char fn2[1000],fn[1000], direct[1000], ext[1000], extReduced[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    int LowNr,nLayers,StartNr,NrFilesPerLayer,NrPixels,BlnketSubtraction;
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
        str = "NrFilesPerDistance ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrFilesPerLayer);
            continue;
        }
        str = "BlanketSubtraction ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &BlnketSubtraction);
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
    sprintf(fn,"%s/%s",direct,fn2);
    fclose(fileParam);
	int ReturnCode;
	ReturnCode = CalcMedian(fn, nLayers,StartNr,NrPixels,NrFilesPerLayer,ext,extReduced,BlnketSubtraction);
	if (ReturnCode == 0){
		printf("Median Calculation failed. Exiting.");
		return 0;
	}
    end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed in computing median for layer %d: %f [s]\n",nLayers,diftotal);
    return 0;
}
