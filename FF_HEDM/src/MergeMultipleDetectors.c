//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
// MergeMultipleDetectors.c
//
// Created by Hemant Sharma on 2016/04/01
//
//

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h> 
#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/types.h>

#define MAX_SPOTS_FILE 2000000
#define MAX_SPOTS_TOTAL 5000000

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

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline
int**
allocMatrixInt(int nrows, int ncols)
{
    int** arr;
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
FreeMemMatrixInt(int **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

int main (int argc, char *argv[])
{
	clock_t start, end;
	start = clock();
	double diftotal;
	char *topfolder;
	topfolder = argv[1];
	char *ParamFN;
	ParamFN = argv[2];
	FILE *fileParam;
	
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n",diftotal);
	return 0;
}
