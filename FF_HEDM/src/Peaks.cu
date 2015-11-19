//
//  Peaks.cu
//
//
//  Created by Hemant Sharma on 2015/07/04.
//

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "nldrmd.cuh"

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAX_LINE_LENGTH 10240
#define MAX_N_RINGS 5000
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
typedef uint16_t pixelvalue;

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

int main(int argc, char *argv[]){ // Arguments: parameter file name
	if (argc != 2){
		printf("Not enough arguments, exiting. Use as:\n\t\t%s %s\n",argv[0],argv[1]);
		return 1;
	}
	//Read params file
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char line[MAX_LINE_LENGTH];
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Parameter file: %s could not be read. Exiting\n",argv[1]);
		return 1;
	}
	char *str;
	int cmpres, StartFileNr, NrFilesPerSweep, NumDarkBegin=0, NumDarkEnd=0,
		ColBeamCurrent, NrOfRings=0, RingNumbers[MAX_N_HKLS];
	double OmegaOffset = 0, bc=0, RingSizeThreshold[MAX_N_HKLS][4];
	char dummy[MAX_LINE_LENGTH], ParFilePath[MAX_LINE_LENGTH], FileStem[MAX_LINE_LENGTH];
	while (fgets(line, MAX_LINE_LENGTH, fileParam) != NULL) {
		str = "ParFilePath ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, ParFilePath);
			continue;
		}
		str = "RingThresh ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d %lf", dummy, RingNumbers[NrOfRings], 
				RingSizeThreshols[NrOfRings][1]);
			NrOfRings++;
			continue;
		}
		str = "FileStem ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, FileStem);
			continue;
		}
		str = "ParFileColBeamCurrent ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &ColBeamCurrent);
			continue;
		}
		str = "StartFileNr ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &StartFileNr);
			continue;
		}
		str = "NrFilesPerSweep ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NrFilesPerSweep);
			continue;
		}
		str = "NumDarkBegin ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkBegin);
			continue;
		}
		str = "NumDarkEnd ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkEnd);
			continue;
		}
		str = "OmegaOffset ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &OmegaOffset);
			continue;
		}
		str = "BeamCurrent ";
		cmpres = strncmp(line,str,strlen(str));
		if (cmpres==0){
			sscanf(line,"%s %lf", dummy, &bc);
			continue;
		}
	}
	FILE *ParFile;
	ParFile = fopen(ParFilePath,"r");
	if (ParFile == NULL){
		printf("ParFile could not be read");
		return 1;
	}
	int i, j;
	int NrFramesPerFile[NrFilesPerSweep],CurrFileNrOffset;
	for (i=0;i<NrFilesPerSweep;i++)
		NrFramesPerFile[i] = -(NumDarkBegin+NumDarkEnd);
	char *token, *saveptr,aline[MAX_LINE_LENGTH];
	int OmegaSign, goodLine, omegafound;
	double Omegas[NrFilesPerSweep][300],BeamCurrents[NrFilesPerSweep][300],
			maxBC=0;
	while (fgets(line, MAX_LINE_LENGTH, ParFile) != NULL) {
		strncpy(aline,line,strlen(line));
		goodLine = 0;
		for (str = aline; ; str=NULL){
			token = strtok_r(str, " ", &saveptr);
			if (token == NULL) break;
			if (!strncmp(token,FileStem,strlen(FileStem))){
				token = strtok_r(str, " ", &saveptr);
				token = strtok_r(str, " ", &saveptr);
				CurrFileNrOffset = atoi(token)-StartFileNr;
				if (CurrFileNrOffset >=0 && CurrFileNrOffset < NrFilesPerSweep){
					NrFramesPerFile[CurrFileNrOffset]++;
					goodLine = 1;
				}
			}
		}
		if (NrFramesPerFile[CurrFileNrOffset] < -NumDarkBegin + 1) continue;
		if (goodLine){
			strncpy(aline,line,strlen(line));
			omegafound = 0;
			for (i=1, str = aline; ; i++, str = NULL){
				token = strtok_r(str, " ", &saveptr);
				if (token == NULL) break;
				if (!strncmp(token,"ramsrot",strlen("ramsrot"))){
					omegafound = 1;
					OmegaSign = 1;
				} else if (!strncmp(token,"aero",strlen("aero"))){
					omegafound = 1;
					OmegaSign = -1;
				} else if (!strncmp(token,"preci",strlen("preci"))){
					omegafound = 1;
					OmegaSign = 1;
				}
				if (omegafound){
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					i+=3;
					Omegas[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] 
								= atof(token) * OmegaSign + OmegaOffset;
					omegafound = 0;
				}
				if (i == ColBeamCurrent){
					BeamCurrents[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] = atof(token);
					maxBC = (maxBC > atof(token)) ? maxBC : atof(token);
				}
			}
		}
	}
	for (i=0;i<NrFilesPerSweep;i++){
		printf("%d \n", NrFramesPerFile[i]);
		for (j=0;j<NrFramesPerFile[i];j++){
			printf("%lf %lf\t",Omegas[i][j],BeamCurrents[i][j]);
		}
		printf("\n");
	}
	bc = (bc > maxBC) ? bc : maxBC;
	printf("%lf\n",bc);
	
	// Read hkls.csv
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(line,1000,hklf);
	int Rnr;
	double RRd;
	while (fgets(line,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %s %s %s %s %s %lf", dummy, dummy, 
			dummy, dummy, &Rnr, dummy, dummmy, dummy, dummy ,dummy, &RRd);
		for (i=0;i<NrOfRings;i++){
			if (Rnr == RingNumbers[i]){
				RingSizeThreshold[i][0] = RRd;
			}
		}
	}

}
