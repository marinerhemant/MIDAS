//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
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

#define PIX_SWAP(a, b)                                                         \
  {                                                                            \
    pixelvalue temp = (a);                                                     \
    (a) = (b);                                                                 \
    (b) = temp;                                                                \
  }

static pixelvalue quick_select(pixelvalue a[], int n) {
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

int CalcMedian(char fn[1000], char outFN[1000], int LayerNr, int StartNr,
               int NrPixels, int NrFilesPerLayer, char ext[1024],
               char extReduced[1024], int numProcs) {
  time_t timer;
  char buffer[25];
  struct tm *tm_info;
  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);

  size_t nPixelsTotal = (size_t)NrPixels * NrPixels;

  // Single contiguous allocation instead of millions of tiny mallocs
  pixelvalue *AllIntensities =
      malloc(nPixelsTotal * NrFilesPerLayer * sizeof(pixelvalue));
  if (AllIntensities == NULL) {
    printf("Could not allocate %.2f GB for intensity data.\n",
           (double)(nPixelsTotal * NrFilesPerLayer * sizeof(pixelvalue)) /
               (1024.0 * 1024.0 * 1024.0));
    return 0;
  }

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

  // Read TIFF files — parallelized across files
  int badRead = 0;
  TIFFErrorHandler oldhandler =
      TIFFSetWarningHandler(NULL); // Suppress warnings globally
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int j = 0; j < NrFilesPerLayer; j++) {
    if (badRead)
      continue;
    char FileName[1024];
    int FileNr = ((LayerNr - 1) * NrFilesPerLayer) + StartNr + j;
    sprintf(FileName, "%s_%06d.%s", fn, FileNr, ext);

    TIFF *tif = TIFFOpen(FileName, "r");

    if (tif == NULL) {
      printf("%s not found.\n", FileName);
      badRead = 1;
      continue;
    }
    printf("Reading file: %s\n", FileName);
    fflush(stdout);

    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
    for (int roil = 0; roil < NrPixels; roil++) {
      TIFFReadScanline(tif, buf, roil, 1);
      pixelvalue *datar = (pixelvalue *)buf;
      for (int i = 0; i < NrPixels; i++) {
        AllIntensities[(size_t)(roil * NrPixels + i) * NrFilesPerLayer + j] =
            datar[i];
      }
    }
    _TIFFfree(buf);
    TIFFClose(tif);
  }
  TIFFSetWarningHandler(oldhandler); // Restore handler
  if (badRead) {
    free(AllIntensities);
    return 0;
  }

  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);
  printf("Calculating median with %d threads.\n", numProcs);

  pixelvalue *MedianArray = malloc(nPixelsTotal * sizeof(pixelvalue));
  pixelvalue *MaxIntArr = malloc(nPixelsTotal * sizeof(pixelvalue));
  pixelvalue *MaxIntMedianArr = malloc(nPixelsTotal * sizeof(pixelvalue));

  // Median + max computation — parallelized across pixels
#pragma omp parallel num_threads(numProcs)
  {
    pixelvalue *SubArr = malloc(NrFilesPerLayer * sizeof(pixelvalue));
#pragma omp for schedule(dynamic, 1024)
    for (size_t i = 0; i < nPixelsTotal; i++) {
      pixelvalue maxVal = 0;
      pixelvalue *src = &AllIntensities[i * NrFilesPerLayer];
      for (int j = 0; j < NrFilesPerLayer; j++) {
        SubArr[j] = src[j];
        if (src[j] > maxVal)
          maxVal = src[j];
      }
      MaxIntArr[i] = maxVal;
      MedianArray[i] = quick_select(SubArr, NrFilesPerLayer);
      int tempVal = (int)maxVal - (int)MedianArray[i];
      MaxIntMedianArr[i] = (pixelvalue)(tempVal > 0 ? tempVal : 0);
    }
    free(SubArr);
  }

  free(AllIntensities);

  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 25, "%Y:%m:%d:%H:%M:%S", tm_info);
  puts(buffer);

  size_t SizeOutFile = sizeof(pixelvalue) * nPixelsTotal;
  int fb =
      open(MedianFileName, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  pwrite(fb, MedianArray, SizeOutFile, 0);
  close(fb);
  int fMaxInt =
      open(MaxIntFileName, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  pwrite(fMaxInt, MaxIntArr, SizeOutFile, 0);
  close(fMaxInt);
  int fMaxIntMedian = open(MaxIntMedianCorrFileName,
                           O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  pwrite(fMaxIntMedian, MaxIntMedianArr, SizeOutFile, 0);
  close(fMaxIntMedian);

  printf("Median calculated.\n");
  free(MedianArray);
  free(MaxIntArr);
  free(MaxIntMedianArr);
  return 1;
}

static void usage(void) {
  printf("MedianImage: usage: ./MedianImage <ParametersFile> <LayerNr> "
         "[nCPUs]\n");
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    usage();
    return 1;
  }

  double start, end;
  double diftotal;
  start = omp_get_wtime();

  int numProcs = 1;
  if (argc == 4) {
    numProcs = atoi(argv[3]);
    if (numProcs < 1)
      numProcs = 1;
  }

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  char fn2[1000], fn[1000], outFN[1000], direct[1000], outputDir[1000],
      ext[1000], extReduced[1000], ReducedFileName[1000];
  outputDir[0] = '\0';
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
    str = "OutputDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, outputDir);
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
  if (outputDir[0] == '\0')
    strcpy(outputDir, direct);
  sprintf(fn, "%s/%s", direct, fn2);
  sprintf(outFN, "%s/%s", outputDir, ReducedFileName);
  fclose(fileParam);

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("        MedianImageLibTiff: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:        %s\n", direct);
  printf("  OutputDirectory:      %s\n", outputDir);
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
  printf("  nCPUs:                %d\n", numProcs);
  printf(
      "================================================================\n\n");

  int ReturnCode;
  ReturnCode = CalcMedian(fn, outFN, nLayers, StartNr, NrPixels,
                          NrFilesPerLayer, ext, extReduced, numProcs);
  if (ReturnCode == 0) {
    printf("Median Calculation failed. Exiting.");
    return 0;
  }
  end = omp_get_wtime();
  diftotal = end - start;
  printf("Time elapsed in computing median for layer %d: %f [s]\n", nLayers,
         diftotal);
  return 0;
}
