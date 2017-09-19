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

#define nColsOutMatr 10

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
	double minO, maxO, imaxO;
	float temp3;
	uint16_t temp1;
	uint32_t temp2;
	fread(&temp3,sizeof(temp3),1,bndFile);
	fread(&temp2,sizeof(temp2),1,bndFile);
	int nSpots = (int) temp2;
	if (nSpots < 1) return 0; // If no spots were there.
	uint32_t *outMatr;
	outMatr = malloc(nColsOutMatr*nSpots*sizeof(*outMatr));
	fgets(aline,4096,fltFile);
	int i, j;
	int pos, nrY, nrZ, nrOme;
	printf("nSpots in BND file: %d\n",nSpots);
	int skipUnit = sizeof(uint16_t)*2 + sizeof(float)*2;
	for (i=0;i<nSpots;i++){
		fgets(aline,4096,fltFile);
		fread(&temp2,sizeof(temp2),1,bndFile);
		fread(&temp1,sizeof(temp1),1,bndFile);
		pos = ftell(bndFile);
		fseek(bndFile,skipUnit*temp1,SEEK_CUR);
		sscanf(aline, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
			" %d %d %lf %d %d %d %d %lf %lf",dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,&imaxS,&imaxF,&imaxO,&minS,&maxS,&minF,&maxF,
			&minO,&maxO);
		outMatr[i*nColsOutMatr+0]  = (uint32_t)  pos; // StartPos
		nrY   = maxS - minS + 1; // nrY
		nrZ   = (maxF - minF + 1); // nrZ
		nrOme = (int)((maxO-minO)/fabs(OmegaStep)) + 1; // nrOmega
		outMatr[i*nColsOutMatr+1]  = (uint32_t)  nrY*nrZ*nrOme; // Bounding Box size
		outMatr[i*nColsOutMatr+2]  = (uint32_t)temp1; // nrPixels
		outMatr[i*nColsOutMatr+3]  = (uint32_t) minS; // Edge Y
		outMatr[i*nColsOutMatr+4]  = (uint32_t) minF; // Edge Z
		outMatr[i*nColsOutMatr+5]  = (uint32_t) ((minO-startOmega)/OmegaStep); // Edge FrameNr
		outMatr[i*nColsOutMatr+6]  = (uint32_t) nrY; // nrY
		outMatr[i*nColsOutMatr+7]  = (uint32_t) nrZ; // nrZ
		outMatr[i*nColsOutMatr+8]  = (uint32_t) nrOme; // nFrames
		outMatr[i*nColsOutMatr+9]  = (uint32_t) (imaxS - minS + nrY*(imaxF - minF) + nrY*nrZ*((int)((imaxO-minO)/OmegaStep))); // maximaPos w.r.t. edge of bounding box
		for (j=0;j<10;j++) printf("%d ",outMatr[i*nColsOutMatr+j]);
	}
	FILE *outFN;
	outFN = fopen("bndMap.bin","wb");
	fwrite(outMatr,nColsOutMatr*nSpots*sizeof(*outMatr),1,outFN);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed for mapping BND file: %f s.\n",diftotal);
    return 0;
}
