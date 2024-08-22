//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "nf_headers.h"

#define RealType double
#define float32_t float
#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define ClearBit(A,k) (A[(k/32)] &= ~(1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;
int SpaceGrp;

double**
allocMatrixF(int nrows, int ncols)
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

int**
allocMatrixIntF(int nrows, int ncols)
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

int
main(int argc, char *argv[])
{
	if (argc != 4){
		printf("Usage:\n simulateNF params.txt InputMicFN OutputFN\n");
		return 1;
	}

    clock_t start, end;
    double diftotal;
    start = clock();

    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
	char cmmd[4096];
	sprintf(cmmd,"~/opt/MIDAS/NF_HEDM/bin/GetHKLList %s",ParamFN);
	system(cmmd);
    char *MicFN = argv[2];
    char *outputFN = argv[3];
    char aline[4096];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[4096];
    int LowNr,nLayers;
    double tx,ty,tz;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "nDistances ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nLayers);
            break;
        }
    }
    rewind(fileParam);
    double Lsd[nLayers],ybc[nLayers],zbc[nLayers],ExcludePoleAngle,
		LatticeConstant[6],doubledummy,
		MaxRingRad,MaxTtheta;
    double px, OmegaStart,OmegaStep;
	char fn[1000];
	char fn2[1000];
	char direct[1000];
	char gridfn[1000];
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
    int cntr=0,countr=0,conter=0,StartNr,EndNr,intdummy,SpaceGroup, RingsToUse[100],nRingsToUse=0;
    int NoOfOmegaRanges=0;
    int nSaves = 1;
    int gridfnfound = 0;
    Wedge = 0;
    int MinMiso = 0;
    int skipBin = 0;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd[cntr]);
            cntr++;
            continue;
        }
        str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SpaceGroup);
            continue;
        }
        str = "SaveReducedOutput ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
			skipBin = 1;
            continue;
        }
        str = "MaxRingRad ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxRingRad);
            continue;
        }
        str = "StartNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &StartNr);
            continue;
        }
        str = "EndNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &EndNr);
            continue;
        }
        str = "ExcludePoleAngle ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ExcludePoleAngle);
            continue;
        }
        str = "LatticeParameter ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf %lf %lf", dummy,
				&LatticeConstant[0],&LatticeConstant[1],
				&LatticeConstant[2],&LatticeConstant[3],
				&LatticeConstant[4],&LatticeConstant[5]);
            continue;
        }
        str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
        str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ty);
            continue;
        }
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &ybc[conter], &zbc[conter]);
            conter++;
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tz);
            continue;
        }
        str = "OmegaStart ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStart);
            continue;
        }
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "Wedge ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wedge);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
		str = "RingsToUse ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingsToUse[nRingsToUse]);
            nRingsToUse++;
            continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &OmegaRanges[NoOfOmegaRanges][0],&OmegaRanges[NoOfOmegaRanges][1]);
            NoOfOmegaRanges++;
            continue;
        }
        str = "BoxSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf", dummy, &BoxSizes[countr][0], &BoxSizes[countr][1], &BoxSizes[countr][2], &BoxSizes[countr][3]);
            countr++;
            continue;
        }
    }
    int i,j,k,l,m,nrFiles,nrPixels;
    for (i=0;i<NoOfOmegaRanges;i++){
		OmegaRang[i][0] = OmegaRanges[i][0];
		OmegaRang[i][1] = OmegaRanges[i][1];
	}
    nOmeRang = NoOfOmegaRanges;
    fclose(fileParam);
    MaxTtheta = rad2deg*atan(MaxRingRad/Lsd[0]);
    uint16_t *ObsSpotsInfo;
    uint16_t *binArr;
    nrFiles = EndNr - StartNr + 1;
    nrPixels = 2048*2048;
    long long int SizeObsSpots;
    SizeObsSpots = (nLayers);
    SizeObsSpots*=nrPixels;
    SizeObsSpots*=nrFiles;
    printf("SizeSimulation: %lld bytes\n",SizeObsSpots*2);
    ObsSpotsInfo = calloc(SizeObsSpots,sizeof(*ObsSpotsInfo));
    binArr = calloc(SizeObsSpots,sizeof(*binArr)); // This is assuming we have quarter of data with signal, not unreasonable.
    if (ObsSpotsInfo == NULL || binArr == NULL){
		printf("Could not allocate arrays! Ran out of RAM?");
		return 1;
	}

	double RotMatTilts[3][3];
	RotationTilts(tx,ty,tz,RotMatTilts);
	double MatIn[3],P0[nLayers][3],P0T[3];
	double xs,ys,edgeLen,gs,ud,eulThis[3],origConf,XG[3],YG[3],dy1,dy2;
	MatIn[0]=0;
	MatIn[1]=0;
	MatIn[2]=0;
	for (i=0;i<nLayers;i++){
		MatIn[0] = -Lsd[i];
		MatrixMultF(RotMatTilts,MatIn,P0T);
		for (j=0;j<3;j++){
			P0[i][j] = P0T[j];
		}
	}
	int n_hkls = 0;
	double hkls[5000][4];
	double Thetas[5000];
	char hklfn[1024];
	sprintf(hklfn,"hkls.csv");
	FILE *hklf = fopen(hklfn,"r");
	fgets(aline,1000,hklf);
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s",dummy,dummy,dummy,
			dummy,&hkls[n_hkls][3],&hkls[n_hkls][0],&hkls[n_hkls][1],
			&hkls[n_hkls][2],&Thetas[n_hkls],dummy,dummy);
		n_hkls++;
	}
	if (nRingsToUse > 0){
		double hklTemps[n_hkls][4],thetaTemps[n_hkls];
		int totalHKLs=0;
		for (i=0;i<nRingsToUse;i++){
			for (j=0;j<n_hkls;j++){
				if ((int)hkls[j][3] == RingsToUse[i]){
					hklTemps[totalHKLs][0] = hkls[j][0];
					hklTemps[totalHKLs][1] = hkls[j][1];
					hklTemps[totalHKLs][2] = hkls[j][2];
					hklTemps[totalHKLs][3] = hkls[j][3];
					thetaTemps[totalHKLs] = Thetas[j];
					totalHKLs++;
				}
			}
		}
		for (i=0;i<totalHKLs;i++){
			hkls[i][0] = hklTemps[i][0];
			hkls[i][1] = hklTemps[i][1];
			hkls[i][2] = hklTemps[i][2];
			hkls[i][3] = hklTemps[i][3];
			Thetas[i] = thetaTemps[i];
		}
		n_hkls = totalHKLs;
	}
	double OMIn[3][3], FracCalc;
	FILE *InpMicF;
	InpMicF = fopen(MicFN,"r");
	printf("Reading from the mic: %s\n",MicFN);
	if (InpMicF == NULL) return 1;
	char outFN[4096];
	sprintf(outFN,"%s.bin",outputFN);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	int lineNr=0;
	double *TheorSpots;
	TheorSpots = malloc(MAX_N_SPOTS*3*sizeof(*TheorSpots));
	if (TheorSpots == NULL){
		printf("Could not allocate memory\n");
		return 1;
	}
	int voxNr=0;
	FILE *spF;
	spF = fopen("SimulatedSpots.csv","w");
    char *headOutThis = "VoxRowNr\tDistanceNr\tFrameNr\tHorPx\tVerPx\tOmegaRaw\tYRaw\tZRaw";
	fprintf(spF,"%s\n",headOutThis);
	while (fgets(aline,4096,InpMicF)!= NULL){
		sscanf(aline,"%s %s %s %lf %lf %lf %lf %lf %lf %lf %lf %s",dummy,dummy,dummy,&xs,&ys,&edgeLen,&ud,&eulThis[0],&eulThis[1],&eulThis[2],&origConf,dummy);
		gs = edgeLen/2;
		dy1 = edgeLen/sqrt(3);
		dy2 = -edgeLen/(2*sqrt(3));
		if (ud < 0){
			dy1 *= -1;
			dy2 *= -1;
		}
		int NrPixelsGrid=2*(ceil((gs*2)/px))*(ceil((gs*2)/px));
		if (gs*2 < px)
			NrPixelsGrid = 1;
		XG[0] = xs;
		XG[1] = xs-gs;
		XG[2] = xs+gs;
		YG[0] = ys+dy1;
		YG[1] = ys+dy2;
		YG[2] = ys+dy2;
		Euler2OrientMat(eulThis,OMIn);
        // printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",OMIn[0][0],OMIn[0][1],OMIn[0][2],OMIn[1][0],OMIn[1][1],OMIn[1][2],OMIn[2][0],OMIn[2][1],OMIn[2][2]);
		SimulateAccOrient(nrFiles,nLayers,ExcludePoleAngle,Lsd,SizeObsSpots,XG,YG,
			RotMatTilts,OmegaStart,OmegaStep,px,ybc,zbc,gs,hkls,n_hkls,Thetas,
			OmegaRanges,NoOfOmegaRanges,BoxSizes,P0,NrPixelsGrid,ObsSpotsInfo,
			OMIn,TheorSpots,voxNr,spF);
		voxNr++;
	}
	printf("Writing output file\n");
	FILE *OutputF;
	if (skipBin == 0){
		OutputF = fopen(outputFN,"wb");
		if (OutputF == NULL){
			printf("Could not write output file\n");
			return 1;
		}
		char dummychar[8192];
		fwrite(dummychar,8192,1,OutputF);
		fwrite(ObsSpotsInfo,SizeObsSpots*sizeof(*ObsSpotsInfo),1,OutputF);
		fclose(OutputF);
	}
	printf("Done with full file\n");
	size_t idxpos,tmpcntr,nrF=0;
	int *bitArr;
	bitArr = calloc(SizeObsSpots/32,sizeof(*bitArr));
	// Sequential read.
	idxpos = 0;
	for (l=0;l<nLayers;l++){
		for (k=0;k<nrFiles;k++){
			for (j=0;j<2048;j++){
				for (i=0;i<2048;i++){
					if (ObsSpotsInfo[idxpos] != 0){
						for (m=0;m<ObsSpotsInfo[idxpos];m++){
							binArr[nrF*5+0] = j;
							binArr[nrF*5+1] = i;
							binArr[nrF*5+2] = k;
							binArr[nrF*5+3] = l;
							binArr[nrF*5+4] = ObsSpotsInfo[idxpos];
							nrF++;
						}
						SetBit(bitArr,idxpos);
					}
					idxpos ++;
				}
			}
		}
	}
	printf("Total number of illuminated pixels: %zu\n",nrF);
	OutputF = fopen(outFN,"wb");
	if (OutputF == NULL){
		printf("Could not write output file\n");
		return 1;
	}
	fwrite(binArr,nrF*5*sizeof(*binArr),1,OutputF);
	fclose(OutputF);
	FILE *outputSpotsInfo = fopen("SpotsInfo.bin","wb");
	if (outputSpotsInfo == NULL){
		printf("Could not write output file\n");
		return 1;
	}
	fwrite(bitArr,SizeObsSpots*sizeof(*bitArr)/32,1,outputSpotsInfo);
	fclose(outputSpotsInfo);
	return 0;
}
