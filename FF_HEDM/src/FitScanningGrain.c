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
	char aline[4096], dummy[4096], outdirpath[4096], cwd[4096], 
		positionsFN[4096], bndStem[4096], bndExt[4096];
	double OmegaStep, startOmega;
	double Lsd, OmegaRanges[50][2], BoxSizes[50][4], MinEta, Wavelength, LatC[6];
	int RingNumbers[500], nRings=0, nOmeRanges=0, nBoxSizes=0;
	while (fgets(aline,4096,paramFile)!=NULL){
		if (StartsWith(aline,"OutDirPath ")){
			sscanf(aline, "%s %s",dummy,outdirpath);
		} else if (StartsWith(aline,"PositionsFile ")){
			sscanf(aline, "%s %s",dummy,positionsFN);
		} else if (StartsWith(aline,"BinStem ")){
			sscanf(aline, "%s %s",dummy,bndStem);
		} else if (StartsWith(aline,"BinExt ")){
			sscanf(aline, "%s %s",dummy,bndExt);
		} else if (StartsWith(aline,"OmegaStep ")){
			sscanf(aline, "%s %lf", dummy, &OmegaStep);
		} else if (StartsWith(aline,"OmegaFirstFile ")){
			sscanf(aline, "%s %lf", dummy, &startOmega);
		} else if (StartsWith(aline,"Lsd ")){
			sscanf(aline,"%s %lf",dummy,&Lsd);
		} else if (StartsWith(aline,"Distance ")){
			sscanf(aline,"%s %lf",dummy,&Lsd);
		} else if (StartsWith(aline,"RingThresh ")){
			sscanf(aline,"%s %d",dummy, &RingNumbers[nRings]);
			nRings++;
		} else if (StartsWith(aline,"OmegaRange ")){
			sscanf(aline,"%s %lf %lf", dummy, 
				&OmegaRanges[nOmeRanges][0], &OmegaRanges[nOmeRanges][1]);
            nOmeRanges++;
		} else if (StartsWith(aline,"BoxSize ")){
			sscanf(aline,"%s %lf %lf %lf %lf", dummy, 
				&BoxSizes[nBoxSizes][0], &BoxSizes[nBoxSizes][1],
				&BoxSizes[nBoxSizes][2], &BoxSizes[nBoxSizes][3]);
            nBoxSizes++;
		} else if (StartsWith(aline,"MinEta ")){
			sscanf(aline,"%s %lf",dummy,&MinEta);
		} else if (StartsWith(aline,"Wavelength ")){
			sscanf(aline,"%s %lf",dummy,&Wavelength);
		} else if (StartsWith(aline,"LatticeConstant ")){
			sscanf(aline,"%s %lf %lf %lf %lf %lf %lf",dummy,&LatC[0],&LatC[1],&LatC[2],
				&LatC[3],&LatC[4],&LatC[5]);
		}
	}
	fclose(paramFile);
	getcwd(cwd,4096);
	startOmega += OmegaStep/2; // Trick for DIGIGrain
	
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
	maxID++; // The array should be one bigger than the maxID
	
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
	int *minMaxLayers, *spotIDInfo, nSpotIDs=0, maxLayerNr, minLayerNr;
	minMaxLayers = malloc(maxNLayers*2*sizeof(*minMaxLayers));
	spotIDInfo = malloc(totUniqSpots*(nColsBndMap+2)*sizeof(*spotIDInfo));
	int i, j, k, l, LayerNr;
	double *layerPosInfo;
	layerPosInfo = malloc(3*maxNLayers*sizeof(*layerPosInfo));
	int *spotIDMap;
	spotIDMap = malloc(maxID*sizeof(*spotIDMap));
	for (i=0;i<maxNLayers;i++){
		layerPosInfo[3*i+0] = 0;
		minMaxLayers[2*i+0] =  maxNSpots;
		minMaxLayers[2*i+1] = -maxNSpots;
	}
	while (fgets(aline,4096,spotFile)!=NULL){
		sscanf(aline,"%d %d %d",&spotIDInfo[nSpotIDs*(nColsBndMap+2)+0],
						&spotIDInfo[nSpotIDs*(nColsBndMap+2)+1],&LayerNr);
		// Store nSpotIDs position corresponding to the new ID so that we can match.
		spotIDMap[spotIDInfo[nSpotIDs*(nColsBndMap+2)+1]] = nSpotIDs;
		if (minMaxLayers[LayerNr*2+0] > nSpotIDs) minMaxLayers[LayerNr*2+0] = nSpotIDs;
		if (minMaxLayers[LayerNr*2+1] < nSpotIDs) minMaxLayers[LayerNr*2+1] = nSpotIDs;
		maxLayerNr = LayerNr;
		if (nSpotIDs == 0) minLayerNr = LayerNr;
		layerPosInfo[3*LayerNr+0] = (double)LayerNr;
		nSpotIDs++; 
	}

	// Read positions file to get file nrs
	FILE *positionFile;
	positionFile = fopen(positionsFN,"r");
	fgets(aline,4096,positionFile);
	int tempctr=1;
	while(fgets(aline,4096,positionFile)!=NULL){
		sscanf(aline,"%lf %s %lf",&layerPosInfo[3*tempctr+1],dummy,&layerPosInfo[3*tempctr+2]);
		tempctr++;
	}

	// Allocate spotInfoArr
	float **spotInfoArr;
	spotInfoArr = malloc(nSpotIDs*sizeof(*spotInfoArr));
	
	// Read BndMap and binFiles
	int SkipBlock = nColsBndMap*sizeof(int);
	int skip;
	int minRowNr, maxRowNr, currSpotID, bndfnr;
	char bndFN[4096], binFN[4096];
	FILE *bndFile, *binFile;
	int *bndReadData;
	bndReadData = malloc(SkipBlock);
	uint16_t ypx, zpx;
	float ome, intensity;
	int minY, minZ, nrY, nrZ, minFrameNr, currentFrameNr;
	int idNr = 0, posToStore;
	long long int totNrPx = 0;
	for (i=1;i<=maxLayerNr;i++){
		if (minMaxLayers[i*2+0] == maxNSpots) continue;
		minRowNr = minMaxLayers[i*2+0];
		maxRowNr = minMaxLayers[i*2+1];
		bndfnr = (int)layerPosInfo[3*i+2];
		sprintf(bndFN,"%s/%s/Layer%d/bndMap.bin",cwd,outdirpath,i);
		sprintf(binFN,"%s/%s%06d%s",cwd,bndStem,bndfnr,bndExt);
		bndFile = fopen(bndFN,"rb");
		binFile = fopen(binFN,"rb");
		for (j=minRowNr;j<=maxRowNr;j++){
			currSpotID = spotIDInfo[j*(nColsBndMap+2)+0];
			skip = (currSpotID-1)*SkipBlock;
			fseek(bndFile,skip,SEEK_SET);
			fread(bndReadData,SkipBlock,1,bndFile);
			for (k=0;k<nColsBndMap;k++){
				spotIDInfo[j*(nColsBndMap+2)+2+k] = bndReadData[k];
			}
			// allocate a new arr
			spotInfoArr[idNr] = calloc(spotIDInfo[j*(nColsBndMap+2)+2+1],sizeof(*spotInfoArr[idNr]));
			minY = spotIDInfo[j*(nColsBndMap+2)+2+3];
			minZ = spotIDInfo[j*(nColsBndMap+2)+2+4];
			minFrameNr = spotIDInfo[j*(nColsBndMap+2)+2+5];
			nrY = spotIDInfo[j*(nColsBndMap+2)+2+6];
			nrZ = spotIDInfo[j*(nColsBndMap+2)+2+7];
			// open bnd file, read data into arr
			fseek(binFile,(int)bndReadData[0],SEEK_SET);
			// We are at the beginning of the data it looks like y, z, ome, intensity
			totNrPx += spotIDInfo[j*(nColsBndMap+2)+2+1];
			for (k=0;k<spotIDInfo[j*(nColsBndMap+2)+2+2];k++){
				fread(&ypx,sizeof(uint16_t),1,binFile);
				fread(&zpx,sizeof(uint16_t),1,binFile);
				fread(&ome,sizeof(float),1,binFile);
				fread(&intensity,sizeof(float),1,binFile);
				currentFrameNr = (int)((ome-startOmega)/OmegaStep);
				// Calculate position:
				posToStore = ((int)ypx - minY) + nrY*((int)zpx - minZ) + nrY*nrZ*(currentFrameNr - minFrameNr);
				spotInfoArr[idNr][posToStore] = intensity;
			}
			idNr++;
		}
		fclose(bndFile);
	}
	printf("%lld\n",totNrPx);	
	// Call simulation function: provide nSpots, spotInfoArr 
	
	
	// Free arrays
	free(bndReadData);
	for (i=0;i<nSpotIDs;i++) free(spotInfoArr[i]);
	free(minMaxLayers);
	free(mapArr);
	free(layerPosInfo);
	free(grainPos);
	free(spotIDMap);
	free(grainProps);
	free(spotIDInfo);
	free(spotMatchArr);
	free(spotInfoArr);
	
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n",diftotal);
	return 0;
}
