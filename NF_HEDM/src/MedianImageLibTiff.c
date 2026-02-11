//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>

typedef uint16_t pixelvalue;
pixelvalue quick_select(pixelvalue a[], int n);

#define PIX_SWAP(a, b)                                                         \
  {                                                                            \
    pixelvalue temp = (a);                                                     \
    (a) = (b);                                                                 \
    (b) = temp;                                                                \
  }
pixelvalue quick_select(pixelvalue a[], int n) {
  int low, high;
  int median;
  int middle, ll, hh;
  low = 0;
  high = n - 1;
  median = (low + high) / 2;
  for (;;) {
    if (high <= low)
      return a[median];
    if (high == low + 1) {
      if (a[low] > a[high])
        PIX_SWAP(a[low], a[high]);
      return a[median];
    }
    middle = (low + high) / 2;
    if (a[middle] > a[high])
      PIX_SWAP(a[middle], a[high]);
    if (a[low] > a[high])
      PIX_SWAP(a[low], a[high]);
    if (a[middle] > a[low])
      PIX_SWAP(a[middle], a[low]);
    PIX_SWAP(a[middle], a[low + 1]);
    ll = low + 1;
    hh = high;
    for (;;) {
      do
        ll++;
      while (a[low] > a[ll]);
      do
        hh--;
      while (a[hh] > a[low]);
      if (hh < ll)
        break;
      PIX_SWAP(a[ll], a[hh]);
    }
    PIX_SWAP(a[low], a[hh]);
    if (hh <= median)
      low = ll;
    if (hh >= median)
      high = hh - 1;
  }
}
#undef PIX_SWAP

pixelvalue **allocMatrixInt(int nrows, int ncols) {
  pixelvalue **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
  }
  return arr;
}

pixelvalue ***allocMatrix3Int(int nrows, int ncols, int nmats) {
  pixelvalue ***arr;
  int i, j;
  arr = malloc(nrows * sizeof(*arr));
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    for (j = 0; j < ncols; j++) {
      arr[i][j] = malloc(nmats * sizeof(*arr[i][j]));
    }
  }
  return arr;
}

void FreeMemMatrixInt(pixelvalue **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

void FreeMemMatrix3Int(pixelvalue ***mat, int nrows, int ncols) {
  int r, c;
  for (r = 0; r < nrows; r++) {
    for (c = 0; c < ncols; c++) {
      free(mat[r][c]);
    }
    free(mat[r]);
  }
  free(mat);
}

int CalcMedian(char fn[1000], char outFN[1000], int LayerNr, int StartNr,
               int NrPixels, int NrFilesPerLayer, char ext[1024],
               char extReduced[1024]) {
  time_t timer;
  char buffer[25];
  struct tm *tm_info;
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);
  int i, j, k, FileNr;
  char FileName[1024];
  pixelvalue **AllIntensities, *MedianArray;
  AllIntensities = allocMatrixInt(NrPixels * NrPixels, NrFilesPerLayer);
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);
  char MedianFileName[1024], MaxIntFileName[1024],
      MaxIntMedianCorrFileName[1024];
  sprintf(MedianFileName, "%s_Median_Background_Distance_%d.%s", outFN,
          LayerNr - 1, extReduced);
  sprintf(MaxIntFileName, "%s_MaximumIntensity_Distance_%d.%s", outFN,
          LayerNr - 1, extReduced);
  sprintf(MaxIntMedianCorrFileName,
          "%s_MaximumIntensityMedianCorrected_Distance_%d.%s", outFN,
          LayerNr - 1, extReduced);
  int roil;
  for (j = 0; j < NrFilesPerLayer; j++) {
    TIFFErrorHandler oldhandler;
    oldhandler = TIFFSetWarningHandler(NULL);
    FileNr = ((LayerNr - 1) * NrFilesPerLayer) + StartNr + j;
    sprintf(FileName, "%s_%06d.%s", fn, FileNr, ext);
    TIFF *tif = TIFFOpen(FileName, "r");
    printf("Opening file: %s\n", FileName);
    fflush(stdout);
    if (tif == NULL) {
      printf("%s not found.\n", FileName);
      return 0;
    }
    TIFFSetWarningHandler(oldhandler);
    if (tif) {
      tdata_t buf;
      buf = _TIFFmalloc(TIFFScanlineSize(tif));
      pixelvalue *datar;
      for (roil = 0; roil < NrPixels; roil++) {
        TIFFReadScanline(tif, buf, roil, 1);
        datar = (uint16 *)buf;
        for (i = 0; i < NrPixels; i++) {
          AllIntensities[roil * NrPixels + i][j] = datar[i];
        }
      }
      _TIFFfree(buf);
    }
    TIFFClose(tif);
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
    puts(buffer);
  }
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);
  printf("Calculating median.\n");
  MedianArray = malloc(NrPixels * NrPixels * sizeof(*MedianArray));
  pixelvalue *MaxIntArr, *MaxIntMedianArr;
  MaxIntArr = malloc(NrPixels * NrPixels * sizeof(*MaxIntArr));
  MaxIntMedianArr = malloc(NrPixels * NrPixels * sizeof(*MaxIntMedianArr));
  pixelvalue SubArr[NrFilesPerLayer];
  int tempVal;
  for (i = 0; i < NrPixels * NrPixels; i++) {
    MaxIntArr[i] = 0;
    MaxIntMedianArr[i] = 0;
    for (j = 0; j < NrFilesPerLayer; j++) {
      SubArr[j] = AllIntensities[i][j];
      if (AllIntensities[i][j] > MaxIntArr[i]) {
        MaxIntArr[i] = AllIntensities[i][j];
      }
    }
    MedianArray[i] = quick_select(SubArr, NrFilesPerLayer);
    tempVal = (MaxIntArr[i] - MedianArray[i]);
    MaxIntMedianArr[i] = (pixelvalue)(tempVal > 0 ? tempVal : 0);
  }
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);
  int SizeOutFile = sizeof(pixelvalue) * NrPixels * NrPixels;
  int fb = open(MedianFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  pwrite(fb, MedianArray, SizeOutFile, 0);
  int fMaxInt = open(MaxIntFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  pwrite(fMaxInt, MaxIntArr, SizeOutFile, 0);
  int fMaxIntMedian =
      open(MaxIntMedianCorrFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  pwrite(fMaxIntMedian, MaxIntMedianArr, SizeOutFile, 0);
  printf("Median calculated.\n");
  free(MedianArray);
  free(MaxIntArr);
  free(MaxIntMedianArr);
  FreeMemMatrixInt(AllIntensities, NrPixels);
  return 1;
}

static void usage(void) {
  printf("MedianImage: usage: ./MedianImage <ParametersFile> <LayerNr>\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    usage();
    return 1;
  }

  clock_t start, end;
  double diftotal;
  start = clock();

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  char fn2[1000], fn[1000], outFN[1000], direct[1000], ext[1000],
      extReduced[1000], ReducedFileName[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  int LowNr, nLayers, StartNr, NrFilesPerLayer, NrPixels, WFImages = 0;
  nLayers = atoi(argv[2]);
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "RawStartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixels);
      continue;
    }
    str = "WFImages ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WFImages);
      continue;
    }
    str = "NrFilesPerDistance ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrFilesPerLayer);
      continue;
    }
    str = "OrigFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
      continue;
    }
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, ReducedFileName);
      continue;
    }
    str = "extOrig ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, ext);
      continue;
    }
    str = "extReduced ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, extReduced);
      continue;
    }
  }
  StartNr = StartNr + (nLayers - 1) * WFImages;
  sprintf(fn, "%s/%s", direct, fn2);
  sprintf(outFN, "%s/%s", direct, ReducedFileName);
  fclose(fileParam);

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("        MedianImageLibTiff: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:        %s\n", direct);
  printf("  OrigFileName:         %s\n", fn2);
  printf("  ReducedFileName:      %s\n", ReducedFileName);
  printf("  Full input path:      %s\n", fn);
  printf("  Full output path:     %s\n", outFN);
  printf("\n--- Image Parameters ---\n");
  printf("  NrPixels:             %d\n", NrPixels);
  printf("  NrFilesPerDistance:   %d\n", NrFilesPerLayer);
  printf("  WFImages:             %d\n", WFImages);
  printf("\n--- File Extensions ---\n");
  printf("  extOrig:              %s\n", ext);
  printf("  extReduced:           %s\n", extReduced);
  printf("\n--- Scan Parameters ---\n");
  printf("  RawStartNr:           %d\n", StartNr);
  printf("  LayerNr (argv[2]):    %d\n", nLayers);
  printf(
      "================================================================\n\n");

  int ReturnCode;
  ReturnCode = CalcMedian(fn, outFN, nLayers, StartNr, NrPixels,
                          NrFilesPerLayer, ext, extReduced);
  if (ReturnCode == 0) {
    printf("Median Calculation failed. Exiting.");
    return 0;
  }
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed in computing median for layer %d: %f [s]\n", nLayers,
         diftotal);
  return 0;
}
