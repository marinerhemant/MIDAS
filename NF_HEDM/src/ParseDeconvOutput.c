//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <tiffio.h>

#define float32_t float
#define MAX_N_OVERLAPS 55000
typedef uint16_t pixelvalue;

int**
allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
    }
    return arr;
}
void
FreeMemMatrixInt(int **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static void
usage(void)
{
    printf("ParseDeconvOutput: usage: ./ParseDeconvOutput <inputFile>\n");
}

int
main(int argc, char *argv[])
{
	if (argc != 4)
    {
        usage();
        return 1;
    }
    clock_t start, end;
    double diftotal;
    start = clock();
    int i,j,k;
    
    char *inpFN;
    inpFN = argv[1];
    char *OutFN;
	OutFN = argv[2];
	FILE *fk;
	fk = fopen(OutFN,"wb");
	int NrPixels = atoi(argv[3]);
	int TotPixelsInt = 0;
    
    float32_t *Image;
    Image = malloc(NrPixels*NrPixels*sizeof(*Image)); // Original image.
    int SizeFile = sizeof(float32_t) * NrPixels * NrPixels;
    // Use LibTiff to read file
    TIFFErrorHandler oldhandler;
	oldhandler = TIFFSetWarningHandler(NULL);
	TIFF* tif = TIFFOpen(inpFN, "r");
	TIFFSetWarningHandler(oldhandler);
	int interInt;
	if (tif) {
		printf("Read file %s\n",inpFN);
		tdata_t buf;
		buf = _TIFFmalloc(TIFFScanlineSize(tif));
		float32_t *datar;
		int rnr;
		for (rnr=0;rnr<NrPixels;rnr++){
			TIFFReadScanline(tif, buf, rnr, 1);
			datar=(float32_t*)buf;
			for (i=0;i<NrPixels;i++){
				Image[rnr*NrPixels+i] = datar[i];
				if (datar[i] != 0){
					TotPixelsInt++;
				}
			}
		}
		_TIFFfree(buf);
	}
	TIFFClose(tif);
	if (TotPixelsInt > 0){
		TotPixelsInt--;
	}else{
		TotPixelsInt = 1;
		Image[2045] = 1;
	}
	printf("Total number of pixels with intensity: %d\n",TotPixelsInt);
	
	pixelvalue *ys, *zs, *peakID;
	float32_t *intensity;
	ys = malloc(TotPixelsInt*2*sizeof(*ys));
	zs = malloc(TotPixelsInt*2*sizeof(*zs));
	peakID = malloc(TotPixelsInt*2*sizeof(*peakID));
	intensity = malloc(TotPixelsInt*2*sizeof(*intensity));
	int PeaksFilledCounter=0;
	int RowNr, ColNr;
	for (i=0;i<NrPixels*NrPixels;i++){
		if (Image[i]!=0){
			peakID[PeaksFilledCounter] = Image[i];
			intensity[PeaksFilledCounter] = Image[i];
			RowNr = i/NrPixels;
			ColNr = i%NrPixels;
			ys[PeaksFilledCounter] = NrPixels - 1 - ColNr;
			zs[PeaksFilledCounter] = NrPixels - 1 - RowNr;
			PeaksFilledCounter++;
		}
	}
	free(Image);
	// Write the result file.
	printf("Now writing file: %s.txt .\n",OutFN);
	FILE *ft;
	char OutFNt[4096];
	sprintf(OutFNt,"%s.txt",OutFN);
	ft = fopen(OutFNt,"w");
	fprintf(ft,"YPos\tZPos\tpeakID\tIntensity\n");
	for (i=0;i<TotPixelsInt;i++){
		fprintf(ft,"%d\t%d\t%d\t%lf\n",(int)ys[i],(int)zs[i],
			(int)peakID[i],(double)intensity[i]);
	}
	fclose(ft);
	float32_t dummy1 = 1;
	uint32_t dummy2 = 1;
	pixelvalue dummy3 = 1;
	char dummy4 = 'x';
	uint32_t DataSize16 = TotPixelsInt*2 + dummy3;
	uint32_t DataSize32 = TotPixelsInt*4 + dummy3;
	fwrite(&dummy1,sizeof(float32_t),1,fk);
	//Start writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk); //uBlockHeader
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //BlockType
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //DataFormat
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NumChildren
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NameSize
	fwrite(&DataSize16,sizeof(uint32_t),1,fk); //DataSize
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //ChunkNumber
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //TotalChunks
	fwrite(&dummy4,sizeof(char)*dummy3,1,fk); //BlockName
	//Finish writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk);
	fwrite(&dummy2,sizeof(uint32_t),1,fk);
	fwrite(&dummy2,sizeof(uint32_t),1,fk);
	fwrite(&dummy2,sizeof(uint32_t),1,fk);
	fwrite(&dummy2,sizeof(uint32_t),1,fk);
	//Start writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk); //uBlockHeader
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //BlockType
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //DataFormat
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NumChildren
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NameSize
	fwrite(&DataSize16,sizeof(uint32_t),1,fk); //DataSize
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //ChunkNumber
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //TotalChunks
	fwrite(&dummy4,sizeof(char)*dummy3,1,fk); //BlockName
	//Finish writing header.
	//Now write y positions.
	fwrite(ys,TotPixelsInt*sizeof(pixelvalue),1,fk);
	//Start writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk); //uBlockHeader
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //BlockType
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //DataFormat
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NumChildren
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NameSize
	fwrite(&DataSize16,sizeof(uint32_t),1,fk); //DataSize
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //ChunkNumber
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //TotalChunks
	fwrite(&dummy4,sizeof(char)*dummy3,1,fk); //BlockName
	//Finish writing header.
	//Now write z positions.
	fwrite(zs,TotPixelsInt*sizeof(pixelvalue),1,fk);
	//Start writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk); //uBlockHeader
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //BlockType
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //DataFormat
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NumChildren
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NameSize
	fwrite(&DataSize32,sizeof(uint32_t),1,fk); //DataSize
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //ChunkNumber
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //TotalChunks
	fwrite(&dummy4,sizeof(char)*dummy3,1,fk); //BlockName
	//Finish writing header.
	//Now write intensities.
	fwrite(intensity,TotPixelsInt*sizeof(float32_t),1,fk);
	//Start writing header.
	fwrite(&dummy2,sizeof(uint32_t),1,fk); //uBlockHeader
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //BlockType
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //DataFormat
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NumChildren
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //NameSize
	fwrite(&DataSize16,sizeof(uint32_t),1,fk); //DataSize
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //ChunkNumber
	fwrite(&dummy3,sizeof(pixelvalue),1,fk); //TotalChunks
	fwrite(&dummy4,sizeof(char)*dummy3,1,fk); //BlockName
	//Finish writing header.
	//Now write PeakIDs.
	fwrite(peakID,TotPixelsInt*sizeof(pixelvalue),1,fk);
	printf("File written, now closing.\n");
	fclose(fk);
	printf("File writing finished.\n");
    end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed for %s:%f [s]\n",inpFN,diftotal);
    return 0;
}
