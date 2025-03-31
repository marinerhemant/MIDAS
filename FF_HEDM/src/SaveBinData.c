//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// SaveBinData.c
//
// Created by Hemant Sharma on 2014/11/07
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

#define N_COL_OBSSPOTS 9      // This is one less number of columns
#define MAX_N_SPOTS 6000000   // max nr of observed spots that can be stored
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
       ObsSpotsLab[i][8] = rad - RingRadii[ringno];
    }

}

int main(int arc, char* argv[]){
	clock_t start, end;
    double diftotal;
    start = clock();
	double **ObsSpots;
	ObsSpots = allocMatrix(MAX_N_SPOTS,N_COL_OBSSPOTS);
	char *ObsSpotsFN = "InputAll.csv";
	FILE *ObsSpotsFile = fopen(ObsSpotsFN,"r");
	char aline[4096];
	char *rc = fgets(aline,4096,ObsSpotsFile);
	int nSpots = 0;
	while (fgets(aline,4096,ObsSpotsFile) != NULL){
		sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf",&ObsSpots[nSpots][0],&ObsSpots[nSpots][1]
			,&ObsSpots[nSpots][2],&ObsSpots[nSpots][3],&ObsSpots[nSpots][4]
			,&ObsSpots[nSpots][5],&ObsSpots[nSpots][6],&ObsSpots[nSpots][7]);
		nSpots++;
	}
	char *AllSpotsFN = "InputAllExtraInfoFittingAll.csv";
	FILE *AllSpotsFile = fopen(AllSpotsFN,"r");
	char *rc2 = fgets(aline,4096,AllSpotsFile);
	int countr=0;
	double **AllSpots;
	AllSpots = allocMatrix(nSpots,14);
	while (fgets(aline,4096,AllSpotsFile) != NULL){
		sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&AllSpots[countr][0],&AllSpots[countr][1],&AllSpots[countr][2],
		&AllSpots[countr][3],&AllSpots[countr][4],&AllSpots[countr][5],&AllSpots[countr][6],&AllSpots[countr][7],&AllSpots[countr][8],
		&AllSpots[countr][9],&AllSpots[countr][10],&AllSpots[countr][11],&AllSpots[countr][12],&AllSpots[countr][13]);
		countr++;
	}
	if (nSpots != countr){
		printf("AllSpots from InputAll and InputAllExtraInfo files don't match. Do something. Exiting\n");
		return 1;
	}
    char *ParamFN = "paramstest.txt", dummy[1024], *str;
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
	printf("%lf %lf %lf %lf %lf\n",omemargin0,etamargin0,etabinsize,rotationstep,omebinsize);


	int i,j,k,t;

	for(i=0;i<MAX_N_RINGS;i++){
		RingRadii[i]=0;
	}
	for(i=0;i<NrOfRings;i++){
		RingRadii[RingNumbers[i]] = RingRadiiUser[i];
	}
	CalcDistanceIdealRing(ObsSpots,nSpots,RingRadii);
	// Make SpotsMatrix
	double *SpotsMat;
	SpotsMat = malloc(nSpots*9*sizeof(*SpotsMat));
	for (i=0;i<nSpots;i++){
		for (j=0;j<9;j++){
			SpotsMat[i*9+j] = ObsSpots[i][j];
		}
	}
	// Make ExtraInfoSpotMatrix
	double *ExtraMat;
	ExtraMat = malloc(nSpots*14*sizeof(*ExtraMat));
	for (i=0;i<nSpots;i++){
		for (j=0;j<14;j++){
			ExtraMat[i*14+j] = AllSpots[i][j];
		}
	}
	char *SpotsFN = "Spots.bin";
	char *ExtraFN = "ExtraInfo.bin";
	FILE *SpotsFile = fopen(SpotsFN,"wb");
	fwrite(SpotsMat,nSpots*9*sizeof(*SpotsMat),1,SpotsFile);
	FILE *ExtraFile = fopen(ExtraFN,"wb");
	fwrite(ExtraMat,nSpots*14*sizeof(*ExtraMat),1,ExtraFile);
	if (nosaveall == 1){
		return 0;
	}

	// Only continue if wanted to save all.
	int ****data;
	int ***ndata;
	int ***maxndata;
	int n_ring_bins;
	int n_eta_bins;
	int n_ome_bins;
	int *newarray;
	int *oldarray;
	int iEta, iOme, iEta0, iOme0;
	int rowno;
	double EtaBinSize = etabinsize;
	double OmeBinSize = omebinsize;
	int HighestRingNo = 0;
	for (i = 0 ; i < MAX_N_RINGS ; i++ ) {
	  if ( RingRadii[i] != 0) HighestRingNo = i;
	}
	n_ring_bins = HighestRingNo;
	n_eta_bins = ceil(360.0 / etabinsize);
	n_ome_bins = ceil(360.0 / omebinsize);
	printf("nRings: %d, nEtas: %d, nOmes: %d\n",n_ring_bins,n_eta_bins,n_ome_bins);
	printf("Total bins: %d\n",n_ring_bins*n_eta_bins*n_ome_bins);
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
	long long int TotNumberOfBins = 0;
	for (rowno = 0 ; rowno < nSpots ; rowno++ ) {
		int ringnr = (int) ObsSpots[rowno][5];
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
				int iSpot = ndata[iRing][iEta][iOme];
				int maxnspot = maxndata[iRing][iEta][iOme];
				if ( iSpot >= maxnspot ) {
					maxnspot = maxnspot + 2;
					oldarray = data[iRing][iEta][iOme];
					newarray = realloc(oldarray, maxnspot * sizeof(*newarray) );
					if ( newarray == NULL ) {
						printf("Memory error: memory full?\n");
						return 1;
					}
					data[iRing][iEta][iOme] = newarray;
					maxndata[iRing][iEta][iOme] = maxnspot;
				}
				data[iRing][iEta][iOme][iSpot] = rowno; // Put row number
				(ndata[iRing][iEta][iOme])++;
				TotNumberOfBins++;
			}
		}
	}
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed in making DataArray: %f s.\n",diftotal);
	long long int LengthNDataStore = n_ring_bins*n_eta_bins*n_ome_bins;
	int *nDataStore, *DataStore;
	nDataStore = malloc(LengthNDataStore*2*sizeof(*nDataStore));
	DataStore = malloc(TotNumberOfBins*sizeof(*DataStore));
	int localCounter = 0, localNDataVal;
	long long int Pos;
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
				for (t=0;t<localNDataVal;t++){
					DataStore[localCounter+t] = data[i][j][k][t];
				}
				localCounter += localNDataVal;
			}
		}
	}

	char *DataFN = "Data.bin";
	char *nDataFN = "nData.bin";
	FILE *DataFile = fopen(DataFN,"wb");
	fwrite(DataStore,TotNumberOfBins*sizeof(*DataStore),1,DataFile);
	FILE *nDataFile = fopen(nDataFN,"wb");
	fwrite(nDataStore,LengthNDataStore*2*sizeof(*nDataStore),1,nDataFile);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Total Time elapsed: %f s.\n",diftotal);
    return 0;
}
