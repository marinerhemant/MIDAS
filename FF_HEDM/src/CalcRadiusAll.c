//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
//
// CalcRadius.c
//
//
// Created by Hemant Sharma on 2014/07/15
//
//
// Need to update SpotIDs when assigned to multiple rings.

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
#include <sys/types.h>
#include <errno.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y)   sqrt((x)*(x) + (y)*(y))
#define MAXNRINGS 500

#define MAX_N_SPOTS 2000000

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
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

int main(int argc, char *argv[]){
	if (argc != 2){
		printf("Usage:\n %s params.txt\n",argv[0]);
		return 1;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    //~ int RingNr = atoi(argv[2]);
    char aline[1000], *str, dummy[1000];
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Could not read file %s\n",ParamFN);
		return 1;
	}
    int LowNr = 1;
    char Folder[1024], FileStem[1024], fs[1024];
    int StartNr, EndNr, LayerNr;
    double Ycen, Zcen, OmegaStep, OmegaFirstFile, Lsd, px, Wavelength, LatticeConstant,Rsample,Hbeam;
    int CellStruct, TopLayer=0, RingNrs[MAXNRINGS],nRings=0;
    double PowderIntIn = 0;
    int DiscModel = 0;
    double DiscArea = 0, Vsample = 0, width;
    while (fgets(aline,1000,fileParam)!=NULL){
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
        str = "RingThresh ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingNrs[nRings]);
            nRings++;
            continue;
        }
        str = "LayerNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &LayerNr);
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
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &Ycen, &Zcen);
            continue;
        }
        //~ str = "PowderIntensity ";
        //~ LowNr = strncmp(aline,str,strlen(str));
        //~ if (LowNr==0){
            //~ sscanf(aline,"%s %lf", dummy, &PowderIntIn);
            //~ continue;
        //~ }
        str = "Width ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &width);
            continue;
        }
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
        str = "LatticeConstant ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &LatticeConstant);
            continue;
        }
        str = "Rsample ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Rsample);
            continue;
        }
        str = "Hbeam ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Hbeam);
            continue;
        }
        str = "Vsample ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Vsample);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd);
            continue;
        }
        str = "DiscModel ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &DiscModel);
            continue;
        }
        str = "DiscArea ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &DiscArea);
            continue;
        }
	}
	sprintf(FileStem,"%s_%d",fs,LayerNr);
	fclose(fileParam);
	char InputFile[2048];
    sprintf(InputFile,"%s/Result_StartNr_%d_EndNr_%d.csv",Folder,StartNr,EndNr);
	FILE *Infile;
	Infile = fopen(InputFile,"r");
    if (Infile == NULL){
		printf("Could not read file %s\n",InputFile);
		return 1;
	}
	fgets(aline,1000,Infile);
	int counter = 0, RingNr;
	double **SpotsMat;
	SpotsMat = allocMatrix(MAX_N_SPOTS,16);
	double PowderInt[nRings];
	int i;
	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	int mhkl[nRings];
	for (i=0;i<nRings;i++){
		mhkl[i] = 0;
		PowderInt[i] = 0;
	}
	int RN;
	fgets(aline,1000,hklf);
	double RingRads[nRings], rrd;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %s %s %s %s %s %lf",dummy, dummy, dummy, dummy,
				&RN, dummy, dummy, dummy, dummy, dummy, &rrd);
		for (i=0;i<nRings;i++){
			RingNr = RingNrs[i];
			if (RN == RingNr){
				RingRads[i] = rrd;
				mhkl[i]++;
				break;
			}
		}
	}
    char header[2048] = "SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px)"
					" IMax MinOme(degrees) MaxOme(degress) Radius(px) Theta(degrees) Eta(degrees) "
					" DeltaOmega NImgs RingNr GrainVolume GrainRadius PowderIntensity SigmaR SigmaEta NrPx NrPxTot\n";
	char OutFile[2048];
	sprintf(OutFile,"%s/Radius_StartNr_%d_EndNr_%d.csv",Folder,StartNr,EndNr);
	FILE *outfile;
	outfile = fopen(OutFile,"w");
	fprintf(outfile,"%s",header);
	double **Sigmas;
	Sigmas = allocMatrix(MAX_N_SPOTS,2);
	double **NrPx;
	NrPx = allocMatrix(MAX_N_SPOTS,2);
	double MinOme=100000, MaxOme=-100000;
	int thisRings[nRings][2];
	double tempArr[13],dummyDouble;
	while (fgets(aline,1000,Infile)!=NULL){
		sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&dummyDouble,&tempArr[0],&tempArr[1],&tempArr[2],&tempArr[3],
			&tempArr[4],&tempArr[5],&tempArr[6],&tempArr[7],&tempArr[8],&tempArr[9],&tempArr[10],&tempArr[11],&tempArr[12]);
		rrd = tempArr[11]*px;
		for (i=0;i<nRings;i++){
			if (fabs(rrd-RingRads[i]) < width){
				SpotsMat[counter][0] = counter+1;
				SpotsMat[counter][1] = tempArr[0];
				SpotsMat[counter][2] = tempArr[1];
				SpotsMat[counter][3] = tempArr[2];
				SpotsMat[counter][4] = tempArr[3];
				SpotsMat[counter][5] = tempArr[4];
				SpotsMat[counter][6] = tempArr[5];
				SpotsMat[counter][7] = tempArr[6];
				Sigmas[counter][0] = tempArr[7];
				Sigmas[counter][1] = tempArr[8];
				NrPx[counter][0] = tempArr[9];
				NrPx[counter][1] = tempArr[10];
				SpotsMat[counter][8] = tempArr[11];
				SpotsMat[counter][10] = tempArr[12];
				//~ if (SpotsMat[counter][2] < MinOme) MinOme = SpotsMat[counter][2];
				//~ if (SpotsMat[counter][2] > MaxOme) MaxOme = SpotsMat[counter][2];
				PowderInt[i] += SpotsMat[counter][1];
				SpotsMat[counter][9] = 0.5*(atand(SpotsMat[counter][8]*px/Lsd));
				SpotsMat[counter][11] = fabs(OmegaStep) + SpotsMat[counter][7] - SpotsMat[counter][6];
				SpotsMat[counter][12] = SpotsMat[counter][11]/fabs(OmegaStep);
				SpotsMat[counter][13] = RingNrs[i];
				if (TopLayer == 1 && fabs(SpotsMat[counter][10]) < 90)
				{}
				else {
					counter++;
				}
			}
		}
	}
	for (i=0;i<nRings;i++){
		PowderInt[i] /= (EndNr-StartNr+1);
	}
	double Vgauge = Hbeam * M_PI * Rsample * Rsample;
	if (Vsample != 0){
		Vgauge = Vsample;
	}
	if (DiscModel == 1){
		Vgauge = DiscArea;
	}
	int j,ctr;
	double deltaTheta;
	for (i=0;i<counter;i++){
		RingNr = SpotsMat[i][13];
		for (j=0;j<nRings;j++){
			if (RingNrs[j] == RingNr){
				ctr = j;
			}
		}
		deltaTheta = deg2rad*(asind(((sind(SpotsMat[i][9]))*(cosd(SpotsMat[i][11])))+((cosd(SpotsMat[i][9]))
						   *(fabs(sind(SpotsMat[i][10])))*(sind(SpotsMat[i][11])))) - SpotsMat[i][9]);
		SpotsMat[i][14] = 0.5*((double)mhkl[ctr])*deltaTheta*cosd(SpotsMat[i][9])*Vgauge*SpotsMat[i][1]/(SpotsMat[i][12]*PowderInt[ctr]);
		//~ totVol += SpotsMat[i][14];
		SpotsMat[i][15] = cbrt(3*SpotsMat[i][14]/(4*M_PI));
		if (DiscModel == 1){
			SpotsMat[i][15] = sqrt(SpotsMat[i][14]/M_PI);
		}
		for (j=0;j<16;j++){
			fprintf(outfile,"%f ",SpotsMat[i][j]);
		}
		fprintf(outfile,"%f %f %f %f %f\n",PowderInt[ctr],Sigmas[i][0],Sigmas[i][1],NrPx[i][0],NrPx[i][1]);
	}
	FreeMemMatrix(SpotsMat,MAX_N_SPOTS);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
