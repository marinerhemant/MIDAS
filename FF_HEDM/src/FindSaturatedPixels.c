//
//
// FindSaturatedPixels.c
//
//
// Created by Hemant Sharma on 2014/08/06
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
#include <malloc.h>

typedef uint16_t pixelvalue;

int main (int argc, char *argv[])
{
	if (argc != 9){
		printf("Usage: FindSaturatedPixels <folder> <filestem> <padding> <startnr> <endnr> <ext> <darkname> <saturationIntensity>\n");
		return 0;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
    char *folder, *filestem, *darkName, *ext;
    int startNr, endNr;
    folder = argv[1];
    filestem = argv[2];
    int Padding = atoi(argv[3]);
    startNr = atoi(argv[4]);
    endNr = atoi(argv[5]);
    ext = argv[6];
    darkName = argv[7];
    int satInt = atoi(argv[8]);
    int i,j,k,nrPixels;
    int nrSaturatedPixels;
    char filename[2048];
    FILE *fileread;
    int sz=0, sz_previous,nFramesDark,nFrames;
    pixelvalue *Image = NULL;
    pixelvalue *Dark;
    int NrSatPxDark=0,NrPixels;
    char darkfilename[2048];
    sprintf(darkfilename,"%s/%s",folder,darkName);
    fileread = fopen(darkfilename,"r");
    if (fileread == NULL){
		printf("Could not read the dark file: %s. Exiting.\n",darkfilename);
		return 1;
	}
	fseek(fileread,0L,SEEK_END);
	sz_previous = sz;
	sz = ftell(fileread) - 8192;
	NrPixels = sz/sizeof(pixelvalue);
	nFramesDark = sz / (8*1024*1024);
	fseek(fileread,8192,SEEK_SET);
	if (sz > sz_previous){
		Image = realloc(Image,sz);
	}
    fread(Image,sz,1,fileread);
    fclose(fileread);
    for (j=0;j<NrPixels;j++){
		NrSatPxDark += Image[j] >= satInt;
	}
	NrSatPxDark /= nFramesDark;
	printf("Number of saturated pixels per dark frame = %d.\n",NrSatPxDark);
    for (i=startNr;i<=endNr;i++){
		nrSaturatedPixels = 0 - NrSatPxDark;
		if (Padding == 2){sprintf(filename,"%s/%s_%02d%s",folder,filestem,i,ext);}
		else if (Padding == 3){sprintf(filename,"%s/%s_%03d%s",folder,filestem,i,ext);}
		else if (Padding == 4){sprintf(filename,"%s/%s_%04d%s",folder,filestem,i,ext);}
		else if (Padding == 5){sprintf(filename,"%s/%s_%05d%s",folder,filestem,i,ext);}
		else if (Padding == 6){sprintf(filename,"%s/%s_%06d%s",folder,filestem,i,ext);}
		else if (Padding == 7){sprintf(filename,"%s/%s_%07d%s",folder,filestem,i,ext);}
		else if (Padding == 8){sprintf(filename,"%s/%s_%08d%s",folder,filestem,i,ext);}
		else if (Padding == 9){sprintf(filename,"%s/%s_%09d%s",folder,filestem,i,ext);}
		fileread = fopen(filename,"r");
		if (fileread == NULL){
			printf("Could not read the input file: %s. Exiting.\n",filename);
			return 1;
		}
		fseek(fileread,0L,SEEK_END);
		sz_previous = sz;
		sz = ftell(fileread) - 8192;
		nFrames = sz/(8*1024*1024);
		fseek(fileread,8192,SEEK_SET);
		if (sz > sz_previous){
			Image = realloc(Image,sz);
		}
		printf("Reading file: %s\n",filename);
		fread(Image,sz,1,fileread);
		fclose(fileread);
		NrPixels = sz/sizeof(pixelvalue);
		printf("Number of saturated pixels more than the dark image in file: \n%s\n",filename);
		for (j=0;j<NrPixels;j++) {
			if ((j+1) % (2048*2048) == 0){
				printf("Frame %03d of %d %d.\n",
							((j+1)/(2048*2048)),nFrames,nrSaturatedPixels);
				nrSaturatedPixels = 0 - NrSatPxDark;
			}
			nrSaturatedPixels += Image[j] >= satInt;
		}
	}
	free(Image);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
