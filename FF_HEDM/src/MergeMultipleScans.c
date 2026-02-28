//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// MergeMultipleScans.c
//
// Created by Hemant Sharma on 2017/08/07
//
//

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static inline int StartsWith(const char *a, const char *b) {
  if (strncmp(a, b, strlen(b)) == 0)
    return 1;
  return 0;
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double diftotal;
  start = clock();
  char *paramFN;
  paramFN = argv[1];
  FILE *paramFile;
  paramFile = fopen(paramFN, "r");
  char *str, aline[4096];
  int nLayers, nRings = 0;
  char outdirpath[4096], dummy[4096], cwd[4096];
  getcwd(cwd, 4096);
  while (fgets(aline, 4096, paramFile) != NULL) {
    if (StartsWith(aline, "nLayers ")) {
      sscanf(aline, "%s %d", dummy, &nLayers);
    } else if (StartsWith(aline, "OutDirPath ")) {
      sscanf(aline, "%s %s", dummy, outdirpath);
    } else if (StartsWith(aline, "RingThresh ")) {
      nRings++;
    }
  }
  fclose(paramFile);
  int i, j, k;
  char outpath[4096];
  char IDsHashFN[4096];
  FILE *IDsHashFile;
  int ringNr, startID, endID;
  int readMatr[nRings][3];
  int outMatr[nLayers * nRings][5], spotIDsTillNow = 0, IDsThisLayer;
  for (i = 0; i < nLayers * nRings; i++) {
    for (j = 0; j < 5; j++) {
      outMatr[i][j] = 0;
    }
  }
  double dSpacing;
  int toSkip;
  char extraFN[4096], outFN[4096];
  FILE *extraFile, *outFile;
  sprintf(outFN, "%s/%s/ExtraInfo.bin", cwd, outdirpath);
  outFile = fopen(outFN, "wb");
  double *ReadMatr;
  ReadMatr = malloc(nRings * 200000);
  for (i = 1; i <= nLayers; i++) {
    sprintf(outpath, "%s/%s/Layer%d/", cwd, outdirpath, i);
    sprintf(IDsHashFN, "%s/IDsHash.csv", outpath);
    printf("Processing layer %d out of %d TotalIDsTillNow: %d\n", i, nLayers,
           spotIDsTillNow);
    IDsHashFile = fopen(IDsHashFN, "r");
    toSkip = 0;
    j = 0;
    while (fgets(aline, 4096, IDsHashFile) != NULL) {
      sscanf(aline, "%d %d %d %lf", &ringNr, &startID, &endID, &dSpacing);
      readMatr[j][0] = ringNr;
      readMatr[j][1] = startID;
      readMatr[j][2] = endID;
      j++;
      if (endID < startID) {
        toSkip = 1;
        break;
      }
    }
    fclose(IDsHashFile);
    if (toSkip == 0) {
      IDsThisLayer = readMatr[nRings - 1][2];
      for (j = 0; j < nRings; j++) {
        outMatr[nRings * (i - 1) + j][0] = i;
        outMatr[nRings * (i - 1) + j][1] = readMatr[j][0];
        outMatr[nRings * (i - 1) + j][2] = spotIDsTillNow + readMatr[j][1];
        outMatr[nRings * (i - 1) + j][3] = spotIDsTillNow + readMatr[j][2];
        outMatr[nRings * (i - 1) + j][4] =
            spotIDsTillNow; // Starting ID, we will subtract this from the total
                            // number of IDs
      }
      fflush(stdout);
      sprintf(extraFN, "%s/ExtraInfo.bin", outpath);
      extraFile = fopen(extraFN, "rb");
      size_t rc =
          fread(ReadMatr, IDsThisLayer * 16 * sizeof(double), 1, extraFile);
      fclose(extraFile);
      for (j = 0; j < IDsThisLayer; j++) {
        ReadMatr[j * 16 + 4] = (double)(spotIDsTillNow + 1 + j);
      }
      fwrite(ReadMatr, IDsThisLayer * 16 * sizeof(double), 1, outFile);
      spotIDsTillNow += IDsThisLayer;
    }
  }
  fclose(outFile);
  sprintf(outFN, "%s/%s/IDsHash.csv", cwd, outdirpath);
  outFile = fopen(outFN, "w");
  for (i = 0; i < nLayers * nRings; i++) {
    for (j = 0; j < 5; j++) {
      fprintf(outFile, "%d\t", outMatr[i][j]);
    }
    fprintf(outFile, "\n");
  }
}
