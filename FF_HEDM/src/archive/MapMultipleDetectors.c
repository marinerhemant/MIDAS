//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
//
// MapMultipleDetectors.c
//
// Image is saved: y first then z.
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
#include <stdbool.h>
#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>

#define MAX_LINE_LENGTH 4096
#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define ClearBit(A,k) (A[(k/32)] &= ~(1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

static inline
void
MatrixMultF33(
    double m[3][3],
    double n[3][3],
    double res[3][3])
{
    int r;
    for (r=0; r<3; r++) {
        res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
        res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
        res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
    }
}

static inline
void
MatrixMult(
           double m[3][3],
           double  v[3],
           double r[3])
{
    int i;
    for (i=0; i<3; i++) {
        r[i] = m[i][0]*v[0] +
        m[i][1]*v[1] +
        m[i][2]*v[2];
    }
}

static inline 
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline
void YZ4mREta(double R, double Eta, double *Y, double *Z){
	*Y = -R*sin(Eta*deg2rad);
	*Z = R*cos(Eta*deg2rad);
}

int main (int argc, char *argv[]){
	clock_t start, end;
	if (argc != 2){
		printf("Usage: ./MapMultipleDetectors Parameters.txt\n");
		return 1;
	}
	double diftotal;
	start = clock();
	char *ParamFN;
	FILE *fileParam;
	ParamFN = argv[1];
	char aline[MAX_LINE_LENGTH];
	char *str, dummy[MAX_LINE_LENGTH];
	int LowNr, nDetectors;
	fileParam = fopen(ParamFN,"r");
	while(fgets(aline,MAX_LINE_LENGTH,fileParam)!=NULL){
		str = "NumDetectors ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr == 0){
			sscanf(aline,"%s %d",dummy,&nDetectors);
			continue;
		}
	}
	rewind(fileParam);
	double DetParams[nDetectors][10]; // Lsd[1], BC[2], ts[3], ps[3], RhoD[1], total 10
	int BigDetSize, NrPixels, BorderToExclude;
	double px;
	int counter=0;
	while(fgets(aline,MAX_LINE_LENGTH,fileParam)!=NULL){
		str = "DetParams ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr == 0){
			sscanf(aline,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",dummy,
				&DetParams[counter][0],&DetParams[counter][1],&DetParams[counter][2],
				&DetParams[counter][3],&DetParams[counter][4],&DetParams[counter][5],
				&DetParams[counter][6],&DetParams[counter][7],&DetParams[counter][8],
				&DetParams[counter][9]);
			counter++;
			continue;
		}
		str = "NrPixels ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr == 0){
			sscanf(aline,"%s %d",dummy, &NrPixels);
			continue;
		}
		str = "BigDetSize ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr == 0){
			sscanf(aline,"%s %d",dummy, &BigDetSize);
			continue;
		}
		str = "BorderToExclude ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr == 0){
			sscanf(aline,"%s %d",dummy, &BorderToExclude);
			continue;
		}
		str = "px ";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf", dummy, &px);
			continue;
		}
	}

	// Initiate BigDetector array
	int *BigDetector;
	uint16_t *BigDetU;
	BigDetU = malloc(BigDetSize*BigDetSize*sizeof(*BigDetU));
	memset(BigDetU,0,BigDetSize*BigDetSize*sizeof(*BigDetU));
	long long int totNrPixels;
	totNrPixels = BigDetSize;
	totNrPixels *= BigDetSize;
	totNrPixels /= 32;
	totNrPixels ++;
	BigDetector = malloc(totNrPixels*sizeof(*BigDetector));
	memset(BigDetector,0,totNrPixels*sizeof(*BigDetector));

	// Run for each detector.
	int i,j,k;
	double ybc, zbc, tx, ty, tz, p0, p1, p2, LsdMean=0, RhoD, n0=2, n1=4,
		   n2=2, Yc, Zc, txr, tyr, tzr, Rad, Eta, RNorm, DistortFunc, Rcorr,
		   EtaT, Lsd, Y, Z, yCorr, zCorr;
	long long int idx;
	int YCInt, ZCInt;
	for (i=0;i<nDetectors;i++){
		LsdMean += DetParams[i][0]/nDetectors;
	}
	printf("Lsd to use: %lf microns.\n",LsdMean);
	fflush(stdout);
	for (j=0;j<nDetectors;j++){
		// Get bc, lsd, ts, ps, RhoD
		Lsd = DetParams[j][0];
		ybc = DetParams[j][1];
		zbc = DetParams[j][2];
		tx = DetParams[j][3];
		ty = DetParams[j][4];
		tz = DetParams[j][5];
		p0 = DetParams[j][6];
		p1 = DetParams[j][7];
		p2 = DetParams[j][8];
		RhoD = DetParams[j][9];
		printf("Detector %d of %d.\ntx:%lf ty:%lf tz:%lf Lsd:%lf p0:%lf p1:%lf p2:%lf RhoD:%lf\n",j,nDetectors-1,tx,ty,tz,Lsd,p0,p1,p2,RhoD);
		fflush(stdout);
		txr = deg2rad*tx;
		tyr = deg2rad*ty;
		tzr = deg2rad*tz;
		double Rx[3][3] =  {{1,0,0},{0,cos(txr),-sin(txr)},{0,sin(txr),cos(txr)}};
		double Ry[3][3] = {{cos(tyr),0,sin(tyr)},{0,1,0},{-sin(tyr),0,cos(tyr)}};
		double Rz[3][3] = {{cos(tzr),-sin(tzr),0},{sin(tzr),cos(tzr),0},{0,0,1}};
		double TRint[3][3], TRs[3][3];
		MatrixMultF33(Ry,Rz,TRint);
		MatrixMultF33(Rx,TRint,TRs);
		for (i=(NrPixels*BorderToExclude)+BorderToExclude;i<(NrPixels*(NrPixels-BorderToExclude))-BorderToExclude;i++){
			// Remove Border Pixels
			if (i%NrPixels <  BorderToExclude ||
				i%NrPixels >= NrPixels - BorderToExclude)
				continue;
			// Convert to Y and Z
			Y = (double) (i/NrPixels);
			Z = (double) (i%NrPixels);
			Yc = -(Y-ybc)*px;
			Zc =  (Z-zbc)*px;
			double ABC[3] = {0,Yc,Zc};
			double ABCPr[3];
			MatrixMult(TRs,ABC,ABCPr);
			double XYZ[3] = {Lsd+ABCPr[0],ABCPr[1],ABCPr[2]};
			Rad = (LsdMean/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
			Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
			RNorm = Rad/RhoD;
			EtaT = 90 - Eta;
			DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT)))) + (p2*(pow(RNorm,n2))) + 1;
			Rcorr = Rad * DistortFunc;
			YZ4mREta(Rcorr,Eta,&yCorr,&zCorr);
			YCInt = (int)floor((BigDetSize/2) - (-yCorr/px));
			ZCInt = (int)floor(((zCorr/px + (BigDetSize/2))));
			idx = (long long int)(YCInt + BigDetSize*ZCInt);
			SetBit(BigDetector,idx);
			BigDetU[idx] = 1;
		}
	}

	// Find single pixels which were by mistake 0.
	int nNeighbors = 0;
	for (i=BigDetSize+1;i<(BigDetSize*(BigDetSize-1))-1;i++){
		if (i%BigDetSize == 0 || i%BigDetSize == (BigDetSize-1)) continue; // Don't do it for border pixels.
		// Take neighbors
		if (BigDetU[i]==0){
			nNeighbors = 0;
			if (BigDetU[i-1-BigDetSize] == 1) nNeighbors++;
			if (BigDetU[i-BigDetSize] == 1) nNeighbors++;
			if (BigDetU[i+1-BigDetSize] == 1) nNeighbors++;
			if (BigDetU[i-1] == 1) nNeighbors++;
			if (BigDetU[i+1] == 1) nNeighbors++;
			if (BigDetU[i-1+BigDetSize] == 1) nNeighbors++;
			if (BigDetU[i+BigDetSize] == 1) nNeighbors++;
			if (BigDetU[i+1+BigDetSize] == 1) nNeighbors++;
			if (nNeighbors > 4){
				BigDetU[i] = 1;
				SetBit(BigDetector,i);
			}
		}
	}
	
	for (i=0;i<1000;i++) for (j=0;j<100;j++) BigDetU[i+BigDetSize*(500+j)] = 1500; // i will be col(y), j will be row(z)

	char FN[MAX_LINE_LENGTH];
	sprintf(FN,"BigDetectorMask.bin");
	FILE *File;
	File = fopen(FN,"wb");
	fwrite(BigDetector,totNrPixels*sizeof(*BigDetector),1,File);

	char FN2[MAX_LINE_LENGTH];
	sprintf(FN2,"BigDetectorMaskEdgeSize%dx%dUnsigned16Bit.bin",BigDetSize,BigDetSize);
	FILE *File2;
	File2 = fopen(FN2,"wb");
	fwrite(BigDetU,BigDetSize*BigDetSize*sizeof(*BigDetU),1,File2);

	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %lf s.\n",diftotal);
	return 0;
}
