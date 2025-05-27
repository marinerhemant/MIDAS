//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  ProcessNFMicRemote.c
//  
//
//  Created by Hemant Sharma on 2014/08/09.
//
//
//  

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
// #include <byteswap.h>

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
double endian_double (const double in)
{
static unsigned char uch[8]; //8 bit char assumed, 64 bit double assumed
static unsigned char *cp;

cp = (unsigned char *) (&in);
uch[7] = cp[0];
uch[6] = cp[1];
uch[5] = cp[2];
uch[4] = cp[3];
uch[3] = cp[4];
uch[2] = cp[5];
uch[1] = cp[6];
uch[0] = cp[7];
return (*(double *)(uch));
}

int main (int argc, char *argv[])
{
	FILE *fgrid=fopen("grid.txt","r");
	char line[1024];
	int rc;
	char *rc2;
	rc2 = malloc(1050*sizeof(rc2));
    rc2 = fgets(line,1000,fgrid);
    int TotNrSpots=0;
    sscanf(line,"%d",&TotNrSpots);
	double **GridPoints;
	GridPoints = allocMatrix(TotNrSpots,2);
	int counter = 0;
	char dummy[1024];
	while (fgets(line,1000,fgrid) != NULL){
		sscanf(line,"%s %s %lf %lf %s",dummy,dummy,&GridPoints[counter][0],&GridPoints[counter][1],dummy);
		counter++;
	}
	fclose(fgrid);
	FILE *fmic=fopen("microstructure.mic","r");
	double *MicContents;
	fseek(fmic,0L,SEEK_END);
	int sz;
	sz = ftell(fmic) / (5*sizeof(double));
	MicContents = malloc(sz*5*sizeof(*MicContents));
	fseek(fmic,0L,SEEK_SET);
	rc = fread(MicContents,sz*5*sizeof(double),1,fmic);
	int i,j;
	int rown;
	double **OutMicContents;
	OutMicContents = allocMatrix(sz,9);
	counter = 0;
	for (i=0;i<sz;i++){
		rown = i+1;
		if (rown > TotNrSpots){
			printf("Grid point number was greater than the total number of grid points. Exiting.\n");
			return 0;
		}
		if (MicContents[i*5+4] == 0) continue;
		OutMicContents[counter][0] = (double) rown;
		OutMicContents[counter][1] = 0;
		OutMicContents[counter][2] = 0;
		OutMicContents[counter][3] = GridPoints[rown-1][0];
		OutMicContents[counter][4] = GridPoints[rown-1][1];
		OutMicContents[counter][5] = endian_double(MicContents[i*5 + 1]);
		OutMicContents[counter][6] = endian_double(MicContents[i*5 + 2]);
		OutMicContents[counter][7] = endian_double(MicContents[i*5 + 3]);
		OutMicContents[counter][8] = endian_double(MicContents[i*5 + 4]);
		counter++;
	}
	int LayerNr = atoi(argv[1]);
	char outfn[1024];
	sprintf(outfn,"Microstructure_Layer%d.mic",LayerNr);
	FILE *fmicout = fopen(outfn,"w");
	for (i=0;i<counter;i++){
		for (j=0;j<9;j++){
			fprintf(fmicout,"%lf ",OutMicContents[i][j]);
		}
		fprintf(fmicout,"\n");
	}
	FreeMemMatrix(OutMicContents,sz);
	FreeMemMatrix(GridPoints,TotNrSpots);
	free(MicContents);
}
