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
	char *inputfile = argv[1];
	char *outputfile = argv[2];
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
	for (i=0;i<NrRows;i++){
		for (j=0;j<11;j++){
			fprintf(out,"%f ",MicContents[i*11+j]);
		}
		fprintf(out,"\n");
	}
}
