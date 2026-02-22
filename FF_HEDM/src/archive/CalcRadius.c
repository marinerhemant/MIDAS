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

#define MAX_N_SPOTS 1000000

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
	if (argc != 3){
		printf("Usage:\n CalcGrains params.txt ringNr\n");
		return 1;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    int RingNr = atoi(argv[2]);
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
    int CellStruct, TopLayer=0;
    double PowderIntIn = 0;
    int DiscModel = 0;
    double DiscArea = 0, Vsample = 0;
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
        str = "PowderIntensity ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &PowderIntIn);
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
    sprintf(InputFile,"%s/PeakSearch/%s/Result_StartNr_%d_EndNr_%d_RingNr_%d.csv",Folder,FileStem,StartNr,EndNr,RingNr);
	FILE *Infile;
	Infile = fopen(InputFile,"r");
    if (Infile == NULL){
		printf("Could not read file %s\n",InputFile);
		return 1;
	}
	fgets(aline,1000,Infile);
	int counter = 0;
	double **SpotsMat;
	SpotsMat = allocMatrix(MAX_N_SPOTS,16);
	double PowderInt = 0;
	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	int mhkl = 0;
	int RN;
	fgets(aline,1000,hklf);
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %s %s %s %s %s %s",dummy, dummy, dummy, dummy,
				&RN, dummy, dummy, dummy, dummy, dummy, dummy);
		if (RN == RingNr){
			mhkl++;
		}
	}
	printf("mhkl: %d\n",mhkl);
    char header[2048] = "SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px)"
					" IMax MinOme(degrees) MaxOme(degress) Radius(px) Theta(degrees) Eta(degrees) "
					" DeltaOmega NImgs RingNr GrainVolume GrainRadius PowderIntensity SigmaR SigmaEta NrPx NrPxTot\n";
	char OutFile[2048];
	sprintf(OutFile,"%s/PeakSearch/%s/Radius_StartNr_%d_EndNr_%d_RingNr_%d.csv",Folder,FileStem,StartNr,EndNr,RingNr);
	FILE *outfile;
	outfile = fopen(OutFile,"w");
	fprintf(outfile,"%s",header);
	double totVol=0;
	double **Sigmas;
	Sigmas = allocMatrix(MAX_N_SPOTS,2);
	double **NrPx;
	NrPx = allocMatrix(MAX_N_SPOTS,2);
	double MinOme=100000, MaxOme=-100000;
	while (fgets(aline,1000,Infile)!=NULL){
		sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&SpotsMat[counter][0],&SpotsMat[counter][1],
				&SpotsMat[counter][2],&SpotsMat[counter][3],&SpotsMat[counter][4],&SpotsMat[counter][5],
				&SpotsMat[counter][6],&SpotsMat[counter][7],&Sigmas[counter][0],&Sigmas[counter][1],&NrPx[counter][0],&NrPx[counter][1]);
		PowderInt += SpotsMat[counter][1];
		if (SpotsMat[counter][2] < MinOme) MinOme = SpotsMat[counter][2];
		if (SpotsMat[counter][2] > MaxOme) MaxOme = SpotsMat[counter][2];
		SpotsMat[counter][8] = CalcNorm2(SpotsMat[counter][3]-Ycen,SpotsMat[counter][4]-Zcen);
		SpotsMat[counter][9] = 0.5*(atand(SpotsMat[counter][8]*px/Lsd));
		SpotsMat[counter][10] = CalcEtaAngle(-(SpotsMat[counter][3]-Ycen),(SpotsMat[counter][4]-Zcen));
		SpotsMat[counter][11] = fabs(OmegaStep) + SpotsMat[counter][7] - SpotsMat[counter][6];
		SpotsMat[counter][12] = SpotsMat[counter][11]/fabs(OmegaStep);
		SpotsMat[counter][13] = RingNr;
		if (TopLayer == 1 && fabs(SpotsMat[counter][10]) < 90) {}
		else {
			counter++;
		}
	}
	double omrang = MaxOme - MinOme;
	PowderInt /= (EndNr-StartNr+1);
	if (PowderIntIn > 0){
		PowderInt = PowderIntIn;
	}
	double Vgauge = Hbeam * M_PI * Rsample * Rsample;
	if (Vsample != 0){
		Vgauge = Vsample;
	}
	if (DiscModel == 1){
		Vgauge = DiscArea;
	}
	int i,j;
	double deltaTheta;
	for (i=0;i<counter;i++){
		deltaTheta = deg2rad*(asind(((sind(SpotsMat[i][9]))*(cosd(SpotsMat[i][11])))+((cosd(SpotsMat[i][9]))
						   *(fabs(sind(SpotsMat[i][10])))*(sind(SpotsMat[i][11])))) - SpotsMat[i][9]);
		SpotsMat[i][14] = 0.5*((double)mhkl)*deltaTheta*cosd(SpotsMat[i][9])*Vgauge*SpotsMat[i][1]/(SpotsMat[i][12]*PowderInt);
		totVol += SpotsMat[i][14];
		SpotsMat[i][15] = cbrt(3*SpotsMat[i][14]/(4*M_PI));
		if (DiscModel == 1){
			SpotsMat[i][15] = sqrt(SpotsMat[i][14]/M_PI);
		}
		for (j=0;j<16;j++){
			fprintf(outfile,"%f ",SpotsMat[i][j]);
		}
		fprintf(outfile,"%f %f %f %f %f\n",PowderInt,Sigmas[i][0],Sigmas[i][1],NrPx[i][0],NrPx[i][1]);
	}
	totVol /= (mhkl*2*omrang/360.0);
	FreeMemMatrix(SpotsMat,MAX_N_SPOTS);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
