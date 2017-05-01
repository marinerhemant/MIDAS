//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//	ParseMic.c - To convert from binary to txt file
//
//
// Hemant Sharma 2014/11/18

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

int main (int argc, char *argv[]){
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    int LowNr,PhaseNr,NumPhases;
    double GlobalPosition;
	char inputfile[1024];
	char outputfile[1024];
    int nSaves = 1;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "PhaseNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &PhaseNr);
            continue;
		}
		str = "NumPhases ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NumPhases);
            continue;
		}
		str = "GlobalPosition ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &GlobalPosition);
            continue;
		}
		str = "MicFileBinary ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, inputfile);
            continue;
		}
		str = "MicFileText ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, outputfile);
            continue;
		}
		str = "SaveNSolutions ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nSaves);
            continue;
        }
	}

	FILE *inp=fopen(inputfile,"rb");
	int sz;
	fseek(inp,0L,SEEK_END);
	sz = ftell(inp);
	rewind(inp);
	double *MicContents;
	MicContents = malloc(sz);
	fread(MicContents,sz,1,inp);
	int NrRows = sz/(sizeof(double)*11);
	printf("NrRows: %d\n",NrRows);
	FILE *out=fopen(outputfile,"w");
	int i,j;
	fprintf(out,"%%TriEdgeSize %lf\n",MicContents[5]);
	fprintf(out,"%%NumPhases %d\n",NumPhases);
	fprintf(out,"%%GlobalPosition %lf\n",GlobalPosition);
	fprintf(out,"%%RowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\tUpDown\tEul1\tEul2\tEul3\tConfidence\tPhaseNr\n");
	for (i=0;i<NrRows;i++){
		for (j=0;j<11;j++){
			//if (j==5) continue; // Skip writing useless info in cols 0,1,2,5 // Write out everything now
			fprintf(out,"%lf\t",MicContents[i*11+j]);
		}
		fprintf(out,"%d\n",PhaseNr);
	}
	// All matches now
	char inputfile2[4096],outputfile2[4096];
	sprintf(outputfile2,"%s.AllMatches",outputfile);
	sprintf(inputfile2,"%s.AllMatches",inputfile);
	FILE *inp2 = fopen(inputfile2,"rb");
	int sz2;
	fseek(inp2,0L,SEEK_END);
	sz2 = ftell(inp2);
	rewind(inp2);
	double *AllMatchesMicContents;
	AllMatchesMicContents = malloc(sz2);
	fread(AllMatchesMicContents,sz2,1,inp2);
	int nCols = 7+4*nSaves;
	int NrRows2 = sz2 / (sizeof(double) * nCols);
	FILE *out2 = fopen(outputfile2,"w");
	fprintf(out2,"%%TriEdgeSize %lf\n",MicContents[5]);
	fprintf(out2,"%%NumPhases %d\n",NumPhases);
	fprintf(out2,"%%GlobalPosition %lf\n",GlobalPosition);
	fprintf(out2,"%%RowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\tUpDown\tEul1\tEul2\tEul3\tConfidence\t...\t...\t...\t...\t...\t...\tPhaseNr\n");
	for (i=0;i<NrRows2;i++){
		for (j=0;j<nCols;j++){
			fprintf(out2,"%lf\t",AllMatchesMicContents[i*nCols+j]);
		}
		fprintf(out2,"%d\n",PhaseNr);
	}
	return(0);
}
