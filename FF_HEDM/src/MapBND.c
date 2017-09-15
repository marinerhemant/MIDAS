//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// MapBND.c
//
// Created by Hemant Sharma on 2017/09/15
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

#define MaxNSpots 1000000

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

int main(int argc, char* argv[]){
	if (argc < 4){
		printf("Usage: ./MapBND ps.txt fltfn bndfn\n");
		return 1;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
	char *paramFN;
	paramFN = argv[1];
	FILE *paramFile;
	paramFile = fopen(paramFN,"r");
	char *fltFN, *bndFN;
	fltFN = argv[2];
	bndFN = argv[3];
	FILE *fltFile, *bndFile;
	fltFile = fopen(fltFN,"r");
	bndFile = fopen(bndFN,"rb");
	char *str, aline[4096];
	char dummy[4096];
	double OmegaStep, startOmega;
	while(fgets(aline,4096,paramFile)!=NULL){
		if (StartsWith(aline,"OmegaStep ")){
			sscanf(aline, "%s %lf", dummy, &OmegaStep);
		} else if (StartsWith(aline,"OmegaFirstFile ")){
			sscanf(aline, "%s %lf", dummy, &startOmega);
		}
	}
	startOmega += OmegaStep/2; // Trick for DIGIGrain
	int minS, maxS, minF, maxF, imaxS, imaxF;
	double minO, maxO;
	float temp3;
	uint16_t temp1;
	uint32_t temp2;
	fread(&temp3,sizeof(temp3),1,bndFile);
	fread(&temp2,sizeof(temp2),1,bndFile);
	int nSpots = (int) temp2;
	if (nSpots < 1) return 0; // If no spots were there.
	uint32_t *outMatr;
	outMatr = malloc(11*nSpots*sizeof(*outMatr));
	fgets(aline,4096,fltFile);
	int i, j;
	int pos;
	printf("nSpots in BND file: %d\n",nSpots);
	for (i=0;i<nSpots;i++){
		fgets(aline,4096,fltFile);
		fread(&temp2,sizeof(temp2),1,bndFile);
		fread(&temp1,sizeof(temp1),1,bndFile);
		pos = ftell(bndFile);
		sscanf(aline, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
			" %d %d %s %d %d %d %d %lf %lf",dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,&imaxS,&imaxF,dummy,&minS,&maxS,&minF,&maxF,
			&minO,&maxO);
		outMatr[i*11+0]  = (uint32_t)  pos; // StartPos
		pos = maxS - minS + 1;
		pos *= (maxF - minF + 1);
		pos *= (int)((maxO-minO)/OmegaStep) + 1;
		outMatr[i*11+1]  = (uint32_t)  pos; // Bounding Box size
		outMatr[i*11+2]  = (uint32_t)temp1; // nrPixels
		outMatr[i*11+3]  = (uint32_t)imaxS;
		outMatr[i*11+4]  = (uint32_t)imaxF;
		outMatr[i*11+5]  = (uint32_t) minS;
		outMatr[i*11+6]  = (uint32_t) maxS;
		outMatr[i*11+7]  = (uint32_t) minF;
		outMatr[i*11+8]  = (uint32_t) maxF;
		outMatr[i*11+9]  = (uint32_t)((minO-startOmega)/OmegaStep); // MinFrameNr
		outMatr[i*11+10] = (uint32_t)((maxO-startOmega)/OmegaStep); // MaxFrameNr
		for (j=0;j<11;j++) printf("%d ",(int)outMatr[i*11+j]);
		printf("\n");
	}
	FILE *outFN;
	outFN = fopen("bndMap.bin","wb");
	fwrite(outMatr,11*nSpots*sizeof(*outMatr),1,outFN);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed for mapping BND file: %f s.\n",diftotal);
    return 0;
}
