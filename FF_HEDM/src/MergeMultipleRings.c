//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
// MergeMultipleRings.c
//
// Created by Hemant Sharma on 2014/07/28
//
//

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h> 
#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/types.h>

#define MAX_SPOTS_FILE 12000000
#define MAX_SPOTS_TOTAL 25000000

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
	clock_t start, end, start0, end0;
    start0 = clock();
    int i,j,k;
    double diftotal;
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[2000];
    char aline2[2000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000],folder[1024],Folder[1024],FileStem[1024],fs[1024];
    int LayerNr;
    int LowNr;
    int RingNumbers[50], nRings=0, RingToIndex, rnr;
    char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	if (hklf == NULL){
		printf("Could not read the hkl file. Exiting.\n");
		return 1;
	}
	fgets(aline2,2000,hklf);
	while (fgets(aline,2000,fileParam)!=NULL){
		//printf("%s\n",aline);
		str = "LayerNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &LayerNr);
            continue;
        }
		str = "Folder ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Folder);
            continue;
        }
		str = "FileStem ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fs);
            continue;
        }
		str = "RingNumbers ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingNumbers[nRings]);
            nRings++;
            continue;
        }
		str = "RingToIndex ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingToIndex);
            continue;
        }
	}
	double dspacing[nRings], ds;
	while (fgets(aline2,2000,hklf)!=NULL){
		sscanf(aline2,"%s %s %s %lf %d %s %s %s %s %s %s", dummy, dummy, 
			dummy, &ds, &rnr, dummy, dummy, dummy, dummy, dummy, dummy);
		//printf("%d %lf %d\n",rnr, ds,nRings);
		for (i=0;i<nRings;i++){
			//printf("%d\n",RingNumbers[i]);
			if (RingNumbers[i] == rnr){
				dspacing[i] = ds;
			}
		}
	}
    char fnInputAll[1024], fnExtraAll[1024],fnSpIDs[1024],fnidhsh[1024];
    FILE *inp, *ext;
    FILE *sp, *idhsh;
    double **Input, **Extra;
    int *SpIDs,**SpotsTemp;
    Input = allocMatrix(MAX_SPOTS_TOTAL,8);
    Extra = allocMatrix(MAX_SPOTS_TOTAL,14);
    SpotsTemp = allocMatrixInt(MAX_SPOTS_FILE,2);
    SpIDs = malloc(MAX_SPOTS_FILE*sizeof(*SpIDs));
    sprintf(FileStem,"%s_%d",fs,LayerNr);
    int counterTotal=0;
    double dumf;
    int startcntr=0;
    int cntr;
    int counterIDs=0,IDTemp;
    int startIDNr[nRings], endIDNr[nRings];
    sprintf(fnidhsh,"%s/IDRings.csv",Folder);
    idhsh = fopen(fnidhsh,"w");
    fprintf(idhsh,"RingNumber OriginalID NewID(RingsMerge)\n");
    for (i=0;i<nRings;i++){
	    sprintf(fnInputAll,"%s/Ring%d/PeakSearch/%s/InputAll.csv",Folder,RingNumbers[i],FileStem);
	    sprintf(fnExtraAll,"%s/Ring%d/PeakSearch/%s/InputAllExtraInfoFittingAll.csv",Folder,RingNumbers[i],FileStem);
	    inp = fopen(fnInputAll,"r");
	    ext = fopen(fnExtraAll,"r");
	    cntr = 0;
	    if (inp == NULL){
	        printf("Input file %s did not exist.\n",fnInputAll);
	        continue;
	    }
		fgets(aline,2000,inp);
		counterTotal = startcntr;
	    while (fgets(aline,2000,inp)!=NULL){
			sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf",&Input[counterTotal][0],&Input[counterTotal][1]
				,&Input[counterTotal][2],&Input[counterTotal][3],&dumf
				,&Input[counterTotal][5],&Input[counterTotal][6],&Input[counterTotal][7]);
				SpotsTemp[cntr][1] = counterTotal+1;
				SpotsTemp[cntr][0] = (int)dumf;
				Input[counterTotal][4] = counterTotal+1;
				counterTotal++;
				cntr++;
		}
		printf("RingNr: %d TotalSpots: %d SpotsThisRing: %d\n",RingNumbers[i],counterTotal,counterTotal-startcntr);
		counterTotal = startcntr;
		fgets(aline,2000,ext);
		while(fgets(aline,2000,ext)!=NULL){
			sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&Extra[counterTotal][0],
				&Extra[counterTotal][1],&Extra[counterTotal][2],&Extra[counterTotal][3],&dumf,
				&Extra[counterTotal][5],&Extra[counterTotal][6],&Extra[counterTotal][7],&Extra[counterTotal][8],
				&Extra[counterTotal][9],&Extra[counterTotal][10],&Extra[counterTotal][11],&Extra[counterTotal][12],
				&Extra[counterTotal][13]);
			Extra[counterTotal][4] = counterTotal+1;
			fprintf(idhsh,"%d %d %d\n",RingNumbers[i],(int)dumf,counterTotal+1);
			counterTotal++;
		}
		startIDNr[i] = startcntr + 1;
		endIDNr[i] = counterTotal;
		startcntr = counterTotal;
		fclose(inp);
		fclose(ext);
	    if (RingNumbers[i] == RingToIndex){
		    sprintf(fnSpIDs,"%s/Ring%d/PeakSearch/%s/SpotsToIndex.bin",Folder,RingNumbers[i],FileStem);
		    sp = fopen(fnSpIDs,"rb");
		    int *SpTemp;
		    SpTemp = malloc(cntr*sizeof(int));
		    fread(SpTemp,cntr*sizeof(int),1,sp);
		    for (k=0;k<cntr;k++){
				if (SpTemp[k] == 0) continue;
				for (j=0;j<cntr;j++){
					if (SpotsTemp[j][0] == SpTemp[k]){
						SpIDs[counterIDs] = SpotsTemp[j][1];
						counterIDs++;
						break;
					}
				}
			}
		}
	}
	fclose(idhsh);
	//Write files
	FILE *inpout, *extout, *idout, *idshashout;
	char fninpout[1024], fnextout[1024], fnidout[1024], fnidshash[1024];
	sprintf(fninpout,"%s/InputAll.csv",Folder);
    sprintf(fnextout,"%s/InputAllExtraInfoFittingAll.csv",Folder);
    sprintf(fnidout,"%s/SpotsToIndex.csv",Folder);
    sprintf(fnidshash,"%s/IDsHash.csv",Folder);
    idshashout = fopen(fnidshash,"w");
    inpout = fopen(fninpout,"w");
    extout = fopen(fnextout,"w");
    if (extout == NULL){
        printf("Could not open file for writing.\n");
        return 1;
    }
    if (idshashout == NULL){
		printf("Could not open hash file for writing.\n");
        return 1;
	}
    for (i=0;i<nRings;i++){
		fprintf(idshashout,"%d %d %d %2.6lf\n",RingNumbers[i],startIDNr[i],endIDNr[i],dspacing[i]);
	}
    idout = fopen(fnidout,"w");
    fprintf(extout,"YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) ZOrig(NoWedgeCorr) YOrig(DetCor) ZOrig(DetCor) OmegaOrig(DetCor)\n");
    fprintf(inpout,"YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta\n");
	for (i=0;i<counterIDs;i++){
		fprintf(idout,"%d\n",SpIDs[i]);
	}
	for (i=0;i<counterTotal;i++){
		for (j=0;j<8;j++){
			fprintf(inpout,"%12.5f ",Input[i][j]);
		}
		for (j=0;j<14;j++){
			fprintf(extout,"%12.5f ",Extra[i][j]);
		}
		fprintf(inpout,"\n");
		fprintf(extout,"\n");
	}
    FreeMemMatrixInt(SpotsTemp,MAX_SPOTS_FILE);
    FreeMemMatrix(Input,MAX_SPOTS_TOTAL);
    FreeMemMatrix(Extra,MAX_SPOTS_TOTAL);
    free(SpIDs);
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
	return 0;
}
