//
// GenMedianDark.c
//
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>

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

static inline
pixelvalue**
allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
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

static inline
void
FreeMemMatrixPx(pixelvalue **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}


int main(int argc, char *argv[]){
	clock_t start, end;
	if (argc != 2){
		printf("Not enough arguments, exiting.\n");
		return 1;
	}
    double diftotal;
    start = clock();
	char *inFN, *outFN;
	inFN = argv[1];
	outFN = argv[2];
	int nrPixels = 2048;
	FILE *fileIn, *fileOut;
	fileIn = fopen(inFN,"rb");
	fileOut = fopen(outFN,"wb");
	size_t sz;
	int *skipContent;
	skipContent = malloc(8192);
	fread(skipContent,8192,1,fileIn);
	pixelvalue **median, **image, *subArr;
	// Median : nrPixels * nrPixels (1d), Image : nrPixels * nrPixels * nFrames (1d)
	// subArr : nFrames (1d)
	median = malloc(nrPixels*nrPixels*sizeof(*median));
	fseek(fileIn,0L,SEEK_END);
	sz = ftell(inFN) - 8192;
	rewind(fileIn);
	image = malloc(sz);
	int nFrames = sz/(8*1024*1024);
	fseek(fileIn,8192,SEEK_SET);
	fread(image,sz,1,fileIn);
	subArr = malloc(nFrames*sizeof(*subArr));
	int i,j,k;
	for (i=0;i<nrPixels*nrPixels;i++){
		for (j=0;j<nFrames;j++){ // Fill subarr
			subArr[j] = image[j*nrPixels*nrPixels + i];
		}
		// Calc Median
		median[i] = quick_select(subArr,nFrames);
	}
	fwrite(skipContent,8192,1,fileOut);
	fwrite(median,nrPixels*nrPixels*sizeof(pixelvalue),1,fileOut);
}
