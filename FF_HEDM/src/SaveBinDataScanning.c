//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// SaveBinData.c
//
// Created by Hemant Sharma on 2014/11/07
// WE NEED TO UPDATE PARAMSTEST.TXT with RingToIndex
//
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define N_COL_OBSSPOTS 10      // This is one less number of columns
#define MAX_N_SPOTS 100000000   // max nr of observed spots that can be stored
#define MAX_N_RINGS 500       // max nr of rings that can be stored (applies to the arrays ringttheta, ringhkl, etc)

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

static inline void FreeMemMatrix(double **mat,int nrows)
{
	int r;
	for ( r = 0 ; r < nrows ; r++) {
		free(mat[r]);
	}
	free(mat);
}

void
CalcDistanceIdealRing(double **ObsSpotsLab, int nspots, double RingRadii[]  )
{
    int i;
    for (i = 0 ; i < nspots ; ++i)
    {
       double y = ObsSpotsLab[i][0];
       double z = ObsSpotsLab[i][1];
       double rad = sqrt(y*y + z*z);
       int ringno = (int) ObsSpotsLab[i][5];
       ObsSpotsLab[i][15] = rad - RingRadii[ringno];
    }

}

struct InpData{
	double Values[15];
};

static int cmpfunc(const void *a, const void *b){
	struct InpData *ia = (struct InpData *)a;
	struct InpData *ib = (struct InpData *)b;
	if (ia->Values[5] < ib->Values[5]) return -1;
	else if (ib->Values[5] < ia->Values[5]) return 1;
	else{
		if (ia->Values[2] < ib->Values[2]) return -1;
		else if (ib->Values[2] < ia->Values[2]) return 1;
		else{
			if (ia->Values[6] < ib->Values[6]) return -1;
			else if (ib->Values[6] < ia->Values[6]) return 1;
		}
	}
}

int main(int argc, char* argv[]){
	clock_t start, end;
	if (argc != 2) {
		printf("Usage: SaveBinDataScanning nScans\n");
		return 1;
	}
	int nScans = atoi(argv[1]);
    double diftotal;
    start = clock();
    char *ParamFN = "paramstest.txt", dummy[1024], *str;
	char aline[4096];
	int LowNr;
    FILE *fileParam;
	fileParam = fopen(ParamFN,"r");
	int NrOfRings = 0, NoRingNumbers = 0, RingNumbers[MAX_N_RINGS];
	double omemargin0, etamargin0, rotationstep, RingRadii[MAX_N_RINGS],
			RingRadiiUser[MAX_N_RINGS], etabinsize, omebinsize;
	int nosaveall = 0;
	while (fgets(aline,4096,fileParam)!=NULL){
        str = "NoSaveAll ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nosaveall);
            continue;
        }
        str = "MarginOme ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &omemargin0);
            continue;
        }
        str = "MarginEta ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &etamargin0);
            continue;
        }
        str = "EtaBinSize ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &etabinsize);
			continue;
		}
        str = "StepsizeOrient ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &rotationstep);
			continue;
		}
        str = "OmeBinSize ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &omebinsize);
			continue;
		}
		str = "RingRadii ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &RingRadiiUser[NrOfRings]);
			NrOfRings++;
			continue;
		}
		str = "RingNumbers ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %d", dummy, &RingNumbers[NoRingNumbers]);
			NoRingNumbers++;
			continue;
		}
	}


	int i,j,k,t;

	int scanNr;
	size_t nSpots = 0;
	int countr=0;
	char AllSpotsFN[4096];
	FILE *AllSpotsFile;
	char *rc;
	struct InpData *MyData;
	MyData = malloc(MAX_N_SPOTS*sizeof(*MyData));
	for (scanNr = 0;scanNr <nScans;scanNr++){
		sprintf(AllSpotsFN,"InputAllExtraInfoFittingAll%d.csv",scanNr);
		AllSpotsFile = fopen(AllSpotsFN,"r");
		rc = fgets(aline,4096,AllSpotsFile);
		while (fgets(aline,4096,AllSpotsFile) != NULL){
			sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
				&MyData[nSpots].Values[0],&MyData[nSpots].Values[1],
				&MyData[nSpots].Values[2],&MyData[nSpots].Values[3],&MyData[nSpots].Values[4],
				&MyData[nSpots].Values[5],&MyData[nSpots].Values[6],&MyData[nSpots].Values[7],
				&MyData[nSpots].Values[8],&MyData[nSpots].Values[9],&MyData[nSpots].Values[10],
				&MyData[nSpots].Values[11],&MyData[nSpots].Values[12],&MyData[nSpots].Values[13]);
			MyData[nSpots].Values[14] = scanNr;
			if (fabs(MyData[nSpots].Values[3]) > 0.0001)
				nSpots++;
		}
		fclose(AllSpotsFile);
	}
	realloc(MyData,nSpots*sizeof(*MyData));
	printf("Number of Spots: %d\n",nSpots);

	// Now sort the spots, depending on the ring numbers, then omega value, then on the eta-value.
	qsort(MyData,nSpots,sizeof(struct InpData),cmpfunc);
	printf("Data sorted.\n");
	double **ObsSpots, **IDMat;
	ObsSpots = allocMatrix(nSpots,16);
	IDMat = allocMatrix(nSpots,3);
	for (i=0;i<nSpots;i++){
		for (j=0;j<15;j++){
			ObsSpots[i][j] = MyData[i].Values[j];
		}
		IDMat[i][0] = i+1;
		IDMat[i][1] = ObsSpots[i][4];
		IDMat[i][2] = ObsSpots[i][14];
		ObsSpots[i][4] = i+1;
	}
	free(MyData);

	for(i=0;i<MAX_N_RINGS;i++){
		RingRadii[i]=0;
	}
	for(i=0;i<NrOfRings;i++){
		RingRadii[RingNumbers[i]] = RingRadiiUser[i];
	}
	CalcDistanceIdealRing(ObsSpots,nSpots,RingRadii);
	// Make SpotsMatrix
	double *SpotsMat;
	SpotsMat = malloc(nSpots*10*sizeof(*SpotsMat));
	for (i=0;i<nSpots;i++){
		for (j=0;j<8;j++){
			SpotsMat[i*10+j] = ObsSpots[i][j];
		}
		SpotsMat[i*10+8] = ObsSpots[i][15];
		SpotsMat[i*10+9] = ObsSpots[i][14];
	}
	// Make ExtraInfoSpotMatrix
	double *ExtraMat;
	ExtraMat = malloc(nSpots*14*sizeof(*ExtraMat));
	for (i=0;i<nSpots;i++){
		for (j=0;j<14;j++){
			ExtraMat[i*14+j] = ObsSpots[i][j];
		}
	}
	char *SpotsFN = "Spots.bin";
	char *ExtraFN = "ExtraInfo.bin";
	char *IDMatFN = "IDsMergedScanning.csv";
	FILE *SpotsFile = fopen(SpotsFN,"wb");
	fwrite(SpotsMat,nSpots*10*sizeof(*SpotsMat),1,SpotsFile);
	fclose(SpotsFile);
	FILE *ExtraFile = fopen(ExtraFN,"wb");
	fwrite(ExtraMat,nSpots*14*sizeof(*ExtraMat),1,ExtraFile);
	fclose(ExtraFile);
	FILE *IDMatFile = fopen(IDMatFN,"w");
	fprintf(IDMatFile,"NewID,OrigID,ScanNr\n");
	for (i=0;i<nSpots;i++) fprintf(IDMatFile,"%d,%d,%d\n",(int)IDMat[i][0],(int)IDMat[i][1],(int)IDMat[i][2]);
	fclose(IDMatFile);
	free(SpotsMat);
	FreeMemMatrix(IDMat,nSpots);
	if (nosaveall == 1){
		return 0;
	}
	printf("Files written. Now generating map.\n");

	// data will save id, scanNr for each spot, everything else remains the same. Twice the size now.
	size_t ****data;
	size_t ***ndata;
	size_t ***maxndata;
	int n_ring_bins;
	int n_eta_bins;
	int n_ome_bins;
	size_t *newarray;
	size_t *oldarray;
	int iEta, iOme, iEta0, iOme0;
	size_t rowno;
	double EtaBinSize = etabinsize;
	double OmeBinSize = omebinsize;
	int HighestRingNo = 0;
	for (i = 0 ; i < MAX_N_RINGS ; i++ ) {
	  if ( RingRadii[i] != 0) HighestRingNo = i;
	}
	n_ring_bins = HighestRingNo;
	n_eta_bins = ceil(360.0 / etabinsize);
	n_ome_bins = ceil(360.0 / omebinsize);
	printf("nRings: %zu, nEtas: %zu, nOmes: %zu\n",n_ring_bins,n_eta_bins,n_ome_bins);
	printf("Total bins: %zu\n",n_ring_bins*n_eta_bins*n_ome_bins);
	int i1, i2, i3;
	data = malloc(n_ring_bins * sizeof(data));
	if (data == NULL ) {
		printf("Memory error: memory full?\n");
		return 1;
	}
	for (i1 = 0 ; i1 < n_ring_bins ; i1++) {
		data[i1] = malloc(n_eta_bins * sizeof(data[i1]));
		if (data[i1] == NULL ) {
			printf("Memory error: memory full?\n");
			return 1;
		}
		for (i2 = 0 ; i2 < n_eta_bins ; i2++) {
			data[i1][i2] = malloc(n_ome_bins * sizeof(data[i1][i2]));
			if (data[i1][i2] == NULL ) {
				printf("Memory error: memory full?\n");
				return 1;
			}
			for (i3 = 0 ; i3 < n_ome_bins ; i3++) {
				data[i1][i2][i3] = NULL;
			}
		}
	}
	ndata = malloc(n_ring_bins * sizeof(ndata));
	if (ndata == NULL ) {
		printf("Memory error: memory full?\n");
		return 1;
	}
	for (i1 = 0 ; i1 < n_ring_bins ; i1++) {
		ndata[i1] = malloc(n_eta_bins * sizeof(ndata[i1]));
		if (ndata[i1] == NULL ) {
			printf("Memory error: memory full?\n");
			return 1;
		}
		for (i2 = 0 ; i2 < n_eta_bins ; i2++) {
			ndata[i1][i2] = malloc(n_ome_bins * sizeof(ndata[i1][i2]));
			if (ndata[i1][i2] == NULL ) {
				printf("Memory error: memory full?\n");
				return 1;
			}
			for (i3 = 0 ; i3 < n_ome_bins ; i3++) {
				ndata[i1][i2][i3] = 0;
			}
		}
	}
	maxndata = malloc(n_ring_bins * sizeof(maxndata));
	if (maxndata == NULL ) {
		printf("Memory error: memory full?\n");
		return 1;
	}
	for (i1 = 0 ; i1 < n_ring_bins ; i1++) {
		maxndata[i1] = malloc(n_eta_bins * sizeof(maxndata[i1]));
		if (maxndata[i1] == NULL ) {
			printf("Memory error: memory full?\n");
			return 1;
		}
		for (i2 = 0 ; i2 < n_eta_bins ; i2++) {
			maxndata[i1][i2] = malloc(n_ome_bins * sizeof(maxndata[i1][i2]));
			if (maxndata[i1][i2] == NULL ) {
				printf("Memory error: memory full?\n");
				return 1;
			}
			for (i3 = 0 ; i3 < n_ome_bins ; i3++) {
				maxndata[i1][i2][i3] = 0;
			}
		}
	}
	size_t TotNumberOfBins = 0;
	for (rowno = 0 ; rowno < nSpots ; rowno++ ) {
		int ringnr = (int) ObsSpots[rowno][5];
		int scanno = (int) ObsSpots[rowno][14];
		double eta = ObsSpots[rowno][6];
		double omega = ObsSpots[rowno][2];
		int iRing = ringnr-1;
		if ( (iRing < 0) || (iRing > n_ring_bins-1) ) continue;
		if ( RingRadii[ringnr] == 0 ) continue;
		double omemargin = omemargin0 + ( 0.5 * rotationstep / fabs(sin(eta * deg2rad)));
		double omemin = 180 + omega - omemargin;
		double omemax = 180 + omega + omemargin;
		int iOmeMin = floor(omemin / omebinsize);
		int iOmeMax = floor(omemax / omebinsize);
		double etamargin = rad2deg * atan(etamargin0/RingRadii[ringnr]) + 0.5 * rotationstep;
		double etamin = 180 + eta - etamargin;
		double etamax = 180 + eta + etamargin;
		int iEtaMin = floor(etamin / etabinsize);
		int iEtaMax = floor(etamax / etabinsize);
		for ( iEta0 = iEtaMin ; iEta0 <= iEtaMax ; iEta0++) {
			iEta = iEta0 % n_eta_bins;
			if ( iEta < 0 ) iEta = iEta + n_eta_bins;
			for ( iOme0 = iOmeMin ; iOme0 <= iOmeMax ; iOme0++) {
				iOme = iOme0 % n_ome_bins;
				if ( iOme < 0 ) iOme = iOme + n_ome_bins;
				size_t iSpot = ndata[iRing][iEta][iOme];
				size_t maxnspot = maxndata[iRing][iEta][iOme];
				if ( iSpot >= maxnspot ) {
					maxnspot = maxnspot + 2;
					oldarray = data[iRing][iEta][iOme];
					newarray = realloc(oldarray, maxnspot*2 * sizeof(*newarray) );
					if ( newarray == NULL ) {
						printf("Memory error: memory full?\n");
						return 1;
					}
					data[iRing][iEta][iOme] = newarray;
					maxndata[iRing][iEta][iOme] = maxnspot;
				}
				data[iRing][iEta][iOme][iSpot*2+0] = rowno; // Put row number
				data[iRing][iEta][iOme][iSpot*2+1] = scanno; // Put scanNr
				(ndata[iRing][iEta][iOme])++;
				TotNumberOfBins++;
			}
		}
	}
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed in making DataArray: %fs Nbins: %lld. Now converting to a 1d array.\n",diftotal,TotNumberOfBins);
	size_t LengthNDataStore = n_ring_bins*n_eta_bins*n_ome_bins;
	size_t *nDataStore, *DataStore;
	nDataStore = malloc(LengthNDataStore*2*sizeof(*nDataStore));
	DataStore = malloc(TotNumberOfBins*2*sizeof(*DataStore));
	size_t localCounter = 0, localNDataVal;
	size_t Pos, tnr;
	for (i=0;i<n_ring_bins;i++){
		for (j=0;j<n_eta_bins;j++){
			for (k=0;k<n_ome_bins;k++){
				localNDataVal = ndata[i][j][k];
				Pos = i*n_eta_bins;
				Pos *= n_ome_bins;
				Pos += j*n_ome_bins;
				Pos += k;
				nDataStore[(Pos*2)+0] = localNDataVal;
				nDataStore[(Pos*2)+1] = localCounter;
				for (tnr=0;tnr<localNDataVal*2;tnr++){
					DataStore[localCounter*2+tnr] = data[i][j][k][tnr];
				}
				localCounter += localNDataVal;
			}
		}
	}
	char *DataFN = "Data.bin";
	char *nDataFN = "nData.bin";
	FILE *DataFile = fopen(DataFN,"wb");
	fwrite(DataStore,TotNumberOfBins*2*sizeof(*DataStore),1,DataFile);
	FILE *nDataFile = fopen(nDataFN,"wb");
	fwrite(nDataStore,LengthNDataStore*2*sizeof(*nDataStore),1,nDataFile);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("nData: %zu, Data: %zu\n",LengthNDataStore*2,TotNumberOfBins*2);
    printf("Total Time elapsed: %f s.\n",diftotal);
    return 0;
}
