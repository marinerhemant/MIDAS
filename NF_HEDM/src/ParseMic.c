//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//	ParseMic.c - To convert from binary to txt file
//
//
// Hemant Sharma 2014/11/18

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CalcNorm2(a, b, c, d) sqrt((a - b) * (a - b) + (c - d) * (c - d))

int main(int argc, char *argv[]) {
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  int LowNr, PhaseNr, NumPhases;
  double GlobalPosition;
  char inputfile[1024];
  char outputfile[1024];
  int nSaves = 1;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "PhaseNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &PhaseNr);
      continue;
    }
    str = "NumPhases ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NumPhases);
      continue;
    }
    str = "GlobalPosition ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &GlobalPosition);
      continue;
    }
    str = "MicFileBinary ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, inputfile);
      continue;
    }
    str = "MicFileText ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, outputfile);
      continue;
    }
    str = "SaveNSolutions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nSaves);
      continue;
    }
  }

  FILE *inp = fopen(inputfile, "rb");
  size_t sz;
  fseek(inp, 0L, SEEK_END);
  sz = ftell(inp);
  rewind(inp);
  double *MicContents;
  MicContents = malloc(sz);
  fread(MicContents, sz, 1, inp);
  int NrRows = sz / (sizeof(double) * 11);
  printf("NrRows: %d\n", NrRows);
  if (NrRows == 0)
    return 0;
  FILE *out = fopen(outputfile, "w");
  char outfilebin[4096];
  sprintf(outfilebin, "%s.map", outputfile);
  FILE *outmap = fopen(outfilebin, "w");
  size_t i;
  long long int j, k;
  fprintf(out, "%%TriEdgeSize %lf\n", MicContents[5]);
  fprintf(out, "%%NumPhases %d\n", NumPhases);
  fprintf(out, "%%GlobalPosition %lf\n", GlobalPosition);
  fprintf(out, "%%"
               "OrientationRowNr\tOrientationID\tRunTime\tX\tY\tTriEdgeSize\tUp"
               "Down\tEul1\tEul2\tEul3\tConfidence\tPhaseNr\n");
  double minXRange = 1e10, minYRange = 1e10, maxXRange = -1e10,
         maxYRange = -1e10;
  for (i = 0; i < NrRows; i++) {
    if (MicContents[i * 11 + 10] == 0)
      continue;
    // Get min max extent
    if (MicContents[i * 11 + 3] > maxXRange)
      maxXRange = MicContents[i * 11 + 3];
    if (MicContents[i * 11 + 3] < minXRange)
      minXRange = MicContents[i * 11 + 3];
    if (MicContents[i * 11 + 4] > maxYRange)
      maxYRange = MicContents[i * 11 + 4];
    if (MicContents[i * 11 + 4] < minYRange)
      minYRange = MicContents[i * 11 + 4];
    for (j = 0; j < 11; j++) {
      fprintf(out, "%lf\t", MicContents[i * 11 + j]);
    }
    fprintf(out, "%d\n", PhaseNr);
  }
  minXRange -= MicContents[5] + 25;
  maxXRange += MicContents[5] + 25;
  minYRange -= MicContents[5] + 25;
  maxYRange += MicContents[5] + 25;
  printf("Ranges for map: %lf %lf %lf %lf\n", minXRange, maxXRange, minYRange,
         maxYRange);
  size_t xSizeMap = (size_t)(ceil(maxXRange) - floor(minXRange) + 1);
  size_t ySizeMap = (size_t)(ceil(maxYRange) - floor(minYRange) + 1);
  size_t size_map = xSizeMap * ySizeMap;
  printf("Size of map: %zu %zu %zu\n", xSizeMap, ySizeMap, size_map);
  double *map;
  map = malloc((size_map * 7 + 4) * sizeof(*map));
  for (i = 0; i < size_map * 7 + 4; i++)
    map[i] = -15;
  double edge_size = MicContents[5];
  double *lengthMat;
  int *RowNrMat;
  lengthMat = malloc(size_map * sizeof(*lengthMat));
  RowNrMat = malloc(size_map * sizeof(*RowNrMat));
  for (i = 0; i < size_map; i++)
    RowNrMat[i] = -1;
  int intX, intY, posX, posY;
  long long int posThis;
  double diffLen;
  for (i = 0; i < NrRows; i++) {
    if (MicContents[i * 11 + 10] == 0)
      continue;
    intX = (int)MicContents[i * 11 + 3];
    intY = (int)MicContents[i * 11 + 4];
    for (j = -(edge_size + 5); j <= edge_size + 5; j++) {
      posX = -minXRange + (intX + j);
      for (k = -(edge_size + 5); k <= edge_size + 5; k++) {
        posY = -minYRange + (intY + k);
        posThis = posY * xSizeMap + posX;
        //~ printf("%zu %zu %d %d %d %d %lld %zu\n",xSizeMap, ySizeMap, intX,
        // intY, posX, posY,posThis,size_map);
        diffLen = CalcNorm2(MicContents[i * 11 + 3], intX + j,
                            MicContents[i * 11 + 4], intY + k);
        if (RowNrMat[posThis] == -1) {
          RowNrMat[posThis] = i;
          lengthMat[posThis] = diffLen;
        } else {
          if (diffLen < lengthMat[posThis]) {
            RowNrMat[posThis] = i;
            lengthMat[posThis] = diffLen;
          }
        }
      }
    }
  }
  int thisRowNr;
  map[0] = xSizeMap;
  map[1] = ySizeMap;
  map[2] = minXRange;
  map[3] = minYRange;
  for (i = 0; i < size_map; i++) {
    if (RowNrMat[i] != -1) {
      thisRowNr = RowNrMat[i];
      map[4 + i + size_map * 0] =
          MicContents[thisRowNr * 11 + 10];                        // Confidence
      map[4 + i + size_map * 1] = MicContents[thisRowNr * 11 + 7]; // Eul1
      map[4 + i + size_map * 2] = MicContents[thisRowNr * 11 + 8]; // Eul2
      map[4 + i + size_map * 3] = MicContents[thisRowNr * 11 + 9]; // Eul3
      map[4 + i + size_map * 4] =
          MicContents[thisRowNr * 11 + 0];      // OrientationRowNr (grainID)
      map[4 + i + size_map * 5] = PhaseNr;      // PhaseNr
      map[4 + i + size_map * 6] = lengthMat[i]; // Distt from HEDM Voxel
    }
  }
  // Write out the map.
  fwrite(map, sizeof(*map) * (size_map * 7 + 4), 1, outmap);
  // All matches now
  char inputfile2[4096], outputfile2[4096];
  sprintf(outputfile2, "%s.AllMatches", outputfile);
  sprintf(inputfile2, "%s.AllMatches", inputfile);
  FILE *inp2 = fopen(inputfile2, "rb");
  if (inp2 == NULL) {
    printf("AllMatched file not found. Exiting\n");
    return 0;
  }
  size_t sz2;
  fseek(inp2, 0L, SEEK_END);
  sz2 = ftell(inp2);
  rewind(inp2);
  double *AllMatchesMicContents;
  AllMatchesMicContents = malloc(sz2);
  fread(AllMatchesMicContents, sz2, 1, inp2);
  size_t nCols = 7 + 4 * nSaves;
  size_t NrRows2 = sz2 / (sizeof(double) * nCols);
  FILE *out2 = fopen(outputfile2, "w");
  fprintf(out2, "%%TriEdgeSize %lf\n", MicContents[5]);
  fprintf(out2, "%%NumPhases %d\n", NumPhases);
  fprintf(out2, "%%GlobalPosition %lf\n", GlobalPosition);
  fprintf(out2,
          "%%"
          "OrientationRowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\tUpDown\tEul"
          "1\tEul2\tEul3\tConfidence\t...\t...\t...\t...\t...\t...\tPhaseNr\n");
  for (i = 0; i < NrRows2; i++) {
    if (AllMatchesMicContents[i * nCols + 10] == 0)
      continue;
    for (j = 0; j < nCols; j++) {
      fprintf(out2, "%lf\t", AllMatchesMicContents[i * nCols + j]);
    }
    fprintf(out2, "%d\n", PhaseNr);
  }
  return (0);
}
