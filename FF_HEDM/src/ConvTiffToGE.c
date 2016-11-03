//
// ConvTiffToGE.c
// Since Perkin Elmer saves output as a binary format, libtiff is not required.
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

typedef uint16_t pixelvalue;

int main(int argc, char *argv[]){
	clock_t start, end;
	if (argc != 6){
		printf("Usage: ./ConvTiffToGE InFileStem Padding StartNr EndNr InExt.\nRight now works for padding 5 only.\nWill always save .ge3 files as output.\n");
		return 1;
	}
    double diftotal;
    int startNr, endNr, Padding;
    startNr = atoi(argv[2]);
    endNr = atoi(argv[3]);
    Padding = atoi(argv[4]);
    start = clock();
	char *inFStem, *inExt;
	inFStem = argv[1];
	inExt = argv[5];
	char FileName[4096], OutFileName[4096];
	int i,j,k;
	int nrPixels = 2048;
	long long int nElements = nrPixels*nrPixels;
	float maxVal = 0;
	for (i=startNr;i<=endNr;i++){
		if (Padding == 5){
			sprintf(FileName,"%s_%05d.%s",inFStem,i,inExt);
			sprintf(OutFileName,"%s_%05d.ge3",inFStem,i);
		}else return 1;
		FILE *fileIn, *fileOut;
		fileIn = fopen(FileName,"rb");
		fileOut = fopen(OutFileName,"wb");
		size_t sz;
		int *skipContent;
		skipContent = calloc(8192,sizeof(int));
		pixelvalue *outimage;
		float *inimage;
		outimage = malloc(nrPixels*nrPixels*sizeof(*outimage));
		inimage =  malloc(nrPixels*nrPixels*sizeof(*inimage));
		printf("Read file %s.\n",FileName);
		fflush(stdout);
		fread(inimage,nElements*sizeof(float),1,fileIn);
		for (j=0;j<nElements;j++){
			if (inimage[i] > maxVal){
				maxVal = inimage[i];
			}
			if (inimage[i] < 0) inimage[i] = 0;
		}
		for (j=0;j<nElements;j++){
			outimage[i] = (pixelvalue) (inimage[i]*14000/maxVal);
		}
		fwrite(skipContent,8192,1,fileOut);
		fwrite(outimage,nrPixels*nrPixels*sizeof(pixelvalue),1,fileOut);
	}
}
