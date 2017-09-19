//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// FitScanningGrain.c
//
// Created by Hemant Sharma on 2017/09/15
//
//

/* Binary file structure:
 *	|	stageposition 	- float32
 *	|	nSpots 			- uint32
 *	|	|	spotID		- uint32
 * 	|	|	nPixels		- uint16
 * 	|	|	|	yPx		- uint16
 * 	|	|	|	zPx		- uint16
 * 	|	|	|	Omega	- float32
 * 	|	|	|	Int		- float32
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <unistd.h>

#define nColsBndMap 10
#define maxNPoints 100000
#define maxNSpots  1000000
#define maxNLayers 5000

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

int main(int argc, char *argv[]){
	if (argc < 3){
		printf("Usage: ./FitScanningGrain ps.txt GrainNr");
		return 1;
	}
	clock_t start, end;
	double diftotal;
	start = clock();
	char *ParamFN;
	ParamFN = argv[1];
	int GrainNr = atoi(argv[2]);
	FILE *paramFile;
	paramFile = fopen(ParamFN,"r");
	char aline[4096], dummy[4096], outdirpath[4096], cwd[4096];
	while (fgets(aline,4096,paramFile)!=NULL){
		if (StartsWith(aline,"OutDirPath ")){
			sscanf(aline, "%s %s",dummy,outdirpath);
		}
	}
	fclose(paramFile);
	getcwd(cwd,4096);
	
	// Read mapFile.csv to get sizes
	char mapFN[4096];
	sprintf(mapFN,"mapFile.csv.%d",GrainNr);
	FILE *mapFile;
	mapFile = fopen(mapFN,"r");
	fgets(aline,4096,mapFile);
	int totPos, totUniqSpots, totAllSpots;
	sscanf(aline,"%d %d %d",&totPos,&totUniqSpots,&totAllSpots);
	int *mapArr, mapctr = 0, startposhere = 0;
	mapArr = malloc(2*totPos*sizeof(*mapArr));
	while (fgets(aline,4096,mapFile)!=NULL){
		sscanf(aline,"%d",&mapArr[mapctr*2+0]);
		mapArr[mapctr*2+1] = startposhere;
		startposhere += mapArr[mapctr*2+0];
		mapctr++;
	}
	printf("%d %d %d %d\n",totPos,mapctr,totUniqSpots,totAllSpots);
	
	//Read SpotMatch.csv to get spotIDs
	char spotMatchFN[4096];
	sprintf(spotMatchFN,"SpotMatch.csv.%d",GrainNr);
	FILE *spotMatchFile;
	spotMatchFile = fopen(spotMatchFN,"r");
	int *spotMatchArr, spotmatchctr=0, maxID=0;
	spotMatchArr = malloc(totAllSpots*sizeof(*spotMatchArr));
	while (fgets(aline,4096,spotMatchFile)!=NULL){
		sscanf(aline,"%d",&spotMatchArr[spotmatchctr]);
		if (maxID < spotMatchArr[spotmatchctr]) maxID = spotMatchArr[spotmatchctr];
		spotmatchctr++;
	}
	printf("%d\n",spotmatchctr);
	
	// Read GrainList.csv & SpotMatch.csv
	char grainFN[4096];
	sprintf(grainFN,"GrainList.csv.%d",GrainNr);
	FILE *grainFile;
	grainFile = fopen(grainFN,"r");
	double *grainProps;
	grainProps = malloc(totPos*9*sizeof(*grainProps));
	double *grainPos;
	grainPos = malloc(totPos*3*sizeof(*grainPos));
	int nPoints=0;
	while (fgets(aline,4096,grainFile)!=NULL){
		sscanf(aline,"%s %s %s %s %s %s %s %s %s %s %lf %lf %s %lf %lf "
			"%lf %lf %lf %lf %s %s %s %s %s %s %s %s %s %s %s %s %s %s "
			"%s %s %s %s %s %s %s %s %s %s %s %lf %lf %lf",dummy,dummy,
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,&grainPos[nPoints*3],
			&grainPos[nPoints*3+1],dummy,&grainProps[nPoints*9+0],&grainProps[nPoints*9+1],
			&grainProps[nPoints*9+2],&grainProps[nPoints*9+3],&grainProps[nPoints*9+4],
			&grainProps[nPoints*9+5],dummy,dummy,dummy,dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,&grainProps[nPoints*9+6],
			&grainProps[nPoints*9+7],&grainProps[nPoints*9+8]);
		grainPos[nPoints*3+2] = 0;
		nPoints++;
	}
	printf("nPoints for this Grain: %d\n",nPoints);

	// Read SpotList
	char spotsFN[4096];
	sprintf(spotsFN,"SpotList.csv.%d",GrainNr);
	FILE *spotFile;
	spotFile = fopen(spotsFN,"r");
	int *MinMaxLayers, *spotIDInfo, nSpotIDs=0, maxLayerNr, minLayerNr;
	MinMaxLayers = malloc(maxNLayers*2*sizeof(*MinMaxLayers));
	spotIDInfo = malloc(totUniqSpots*(nColsBndMap+2)*sizeof(*spotIDInfo));
	int i, j, k, l, LayerNr;
	double *LayerPosInfo;
	LayerPosInfo = malloc(2*maxNLayers*sizeof(*LayerPosInfo));
	int nLayers = 0, currentLayer = 0, *spotIDMap;
	spotIDMap = malloc(maxID*sizeof(*spotIDMap));
	LayerPosInfo[0] = -1;
	for (i=0;i<maxNLayers;i++){
		MinMaxLayers[2*i+0] =  maxNSpots;
		MinMaxLayers[2*i+1] = -maxNSpots;
	}
	while (fgets(aline,4096,spotFile)!=NULL){
		sscanf(aline,"%d %d %d",&spotIDInfo[nSpotIDs*(nColsBndMap+2)+0],
						&spotIDInfo[nSpotIDs*(nColsBndMap+2)+1],&LayerNr);
		// Store nSpotIDs position corresponding to the new ID so that we can match.
		spotIDMap[spotIDInfo[nSpotIDs*(nColsBndMap+2)+1]] = nSpotIDs;
		if (MinMaxLayers[LayerNr*2+0] > nSpotIDs) MinMaxLayers[LayerNr*2+0] = nSpotIDs;
		if (MinMaxLayers[LayerNr*2+1] < nSpotIDs) MinMaxLayers[LayerNr*2+1] = nSpotIDs;
		maxLayerNr = LayerNr;
		if (nSpotIDs == 0) minLayerNr = LayerNr;
		if (LayerNr != currentLayer){
			LayerPosInfo[2*nLayers+0] = (double)LayerNr;
			currentLayer = LayerNr;
			nLayers++;
		}
		nSpotIDs++; 
	}
	printf("%d\n",nSpotIDs);

	// Read BndMap
	int SkipBlock = nColsBndMap*sizeof(uint32_t);
	int skip;
	int minRowNr, maxRowNr, currSpotID;
	char bndFN[4096];
	FILE *bndFile;
	uint32_t *bndReadData;
	bndReadData = malloc(SkipBlock);
	for (i=1;i<=maxLayerNr;i++){
		if (MinMaxLayers[i*2+0] == maxNSpots) continue;
		minRowNr = MinMaxLayers[i*2+0];
		maxRowNr = MinMaxLayers[i*2+1];
		sprintf(bndFN,"%s/%s/Layer%d/bndMap.bin",cwd,outdirpath,i);
		bndFile = fopen(bndFN,"rb");
		for (j=minRowNr;j<=maxRowNr;j++){
			currSpotID = spotIDInfo[j*(nColsBndMap+2)+0];
			skip = (currSpotID-1)*SkipBlock;
			fseek(bndFile,skip,SEEK_SET);
			fread(bndReadData,SkipBlock,1,bndFile);
			for (k=0;k<nColsBndMap;k++){
				spotIDInfo[j*(nColsBndMap+2)+2+k] = (int)bndReadData[k];
			}
		}
		fclose(bndFile);
	}
	
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n",diftotal);
	return 0;
}
