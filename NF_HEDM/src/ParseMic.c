
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//	ParseMic.c - To convert from binary to txt file
//
//
// Hemant Sharma 2014/11/18
//

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CalcNorm2(a, b, c, d) sqrt((a - b) * (a - b) + (c - d) * (c - d))

typedef struct {
  int PhaseNr;
  int NumPhases;
  double GlobalPosition;
  char inputfile[1024];
  char outputfile[1024];
  int nSaves;
} MicParams;

static void usage(void) {
  printf("Usage: ./ParseMic <ParametersFile>\n");
  printf("Parses a binary .mic file and converts it to text format, generating "
         "a map.\n");
}

static int ReadParameters(const char *ParamFN, MicParams *params) {
  FILE *fileParam = fopen(ParamFN, "r");
  if (fileParam == NULL) {
    fprintf(stderr, "Error: Could not open parameter file %s\n", ParamFN);
    return 0;
  }

  char aline[1000];
  char *str, dummy[1000];
  int LowNr;

  // Defaults
  params->nSaves = 1;

  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "PhaseNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %d", &params->PhaseNr);
      continue;
    }
    str = "NumPhases ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %d", &params->NumPhases);
      continue;
    }
    str = "GlobalPosition ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %lf", &params->GlobalPosition);
      continue;
    }
    str = "MicFileBinary ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %s", params->inputfile);
      continue;
    }
    str = "MicFileText ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %s", params->outputfile);
      continue;
    }
    str = "SaveNSolutions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %d", &params->nSaves);
      continue;
    }
  }
  fclose(fileParam);
  return 1;
}

static double *ReadMicBinary(const char *inputfile, size_t *sz) {
  FILE *inp = fopen(inputfile, "rb");
  if (inp == NULL) {
    fprintf(stderr, "Error: Could not open binary mic file %s\n", inputfile);
    return NULL;
  }
  fseek(inp, 0L, SEEK_END);
  *sz = ftell(inp);
  rewind(inp);

  double *MicContents = malloc(*sz);
  if (MicContents == NULL) {
    fprintf(stderr,
            "Error: Memory allocation failed for MicContents (%zu bytes)\n",
            *sz);
    fclose(inp);
    return NULL;
  }

  if (fread(MicContents, *sz, 1, inp) != 1) {
    fprintf(stderr, "Error: Failed to read MicContents\n");
    free(MicContents);
    fclose(inp);
    return NULL;
  }
  fclose(inp);
  return MicContents;
}

static void WriteMicText(const char *outputfile, double *MicContents,
                         int NrRows, MicParams *params) {
  FILE *out = fopen(outputfile, "w");
  if (out == NULL) {
    fprintf(stderr, "Error: Could not open output file %s\n", outputfile);
    return;
  }

  // Buffered I/O for text writing
  char buf[1048576];
  setvbuf(out, buf, _IOFBF, sizeof(buf));

  fprintf(out, "%%TriEdgeSize %lf\n", MicContents[5]);
  fprintf(out, "%%NumPhases %d\n", params->NumPhases);
  fprintf(out, "%%GlobalPosition %lf\n", params->GlobalPosition);
  fprintf(out, "%%"
               "OrientationRowNr\tOrientationID\tRunTime\tX\tY\tTriEdgeSize\tUp"
               "Down\tEul1\tEul2\tEul3\tConfidence\tPhaseNr\n");

  for (int i = 0; i < NrRows; i++) {
    if (MicContents[i * 11 + 10] == 0)
      continue;
    for (int j = 0; j < 11; j++) {
      fprintf(out, "%lf\t", MicContents[i * 11 + j]);
    }
    fprintf(out, "%d\n", params->PhaseNr);
  }
  fclose(out);
}

static void GenerateMap(const char *outputfile, double *MicContents, int NrRows,
                        MicParams *params) {
  double minXRange = 1e10, minYRange = 1e10, maxXRange = -1e10,
         maxYRange = -1e10;

  for (int i = 0; i < NrRows; i++) {
    if (MicContents[i * 11 + 10] == 0)
      continue;
    if (MicContents[i * 11 + 3] > maxXRange)
      maxXRange = MicContents[i * 11 + 3];
    if (MicContents[i * 11 + 3] < minXRange)
      minXRange = MicContents[i * 11 + 3];
    if (MicContents[i * 11 + 4] > maxYRange)
      maxYRange = MicContents[i * 11 + 4];
    if (MicContents[i * 11 + 4] < minYRange)
      minYRange = MicContents[i * 11 + 4];
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

  double *map = malloc((size_map * 7 + 4) * sizeof(*map));
  double *lengthMat = malloc(size_map * sizeof(*lengthMat));
  int *RowNrMat = malloc(size_map * sizeof(*RowNrMat));

  if (!map || !lengthMat || !RowNrMat) {
    fprintf(stderr, "Error: Memory allocation failed for map generation.\n");
    free(map);
    free(lengthMat);
    free(RowNrMat);
    return;
  }

  for (size_t i = 0; i < size_map * 7 + 4; i++)
    map[i] = -15;
  for (size_t i = 0; i < size_map; i++)
    RowNrMat[i] = -1;

  double edge_size = MicContents[5];
  int intX, intY, posX, posY;
  long long int posThis;
  double diffLen;

  for (int i = 0; i < NrRows; i++) {
    if (MicContents[i * 11 + 10] == 0)
      continue;

    intX = (int)MicContents[i * 11 + 3];
    intY = (int)MicContents[i * 11 + 4];

    for (int j = -(edge_size + 5); j <= edge_size + 5; j++) {
      posX = -minXRange + (intX + j);
      for (int k = -(edge_size + 5); k <= edge_size + 5; k++) {
        posY = -minYRange + (intY + k);
        posThis = posY * xSizeMap + posX;

        diffLen = CalcNorm2(MicContents[i * 11 + 3], intX + j,
                            MicContents[i * 11 + 4], intY + k);

        if (RowNrMat[posThis] == -1 || diffLen < lengthMat[posThis]) {
          RowNrMat[posThis] = i;
          lengthMat[posThis] = diffLen;
        }
      }
    }
  }

  map[0] = xSizeMap;
  map[1] = ySizeMap;
  map[2] = minXRange;
  map[3] = minYRange;

  for (size_t i = 0; i < size_map; i++) {
    if (RowNrMat[i] != -1) {
      int thisRowNr = RowNrMat[i];
      map[4 + i + size_map * 0] =
          MicContents[thisRowNr * 11 + 10];                        // Confidence
      map[4 + i + size_map * 1] = MicContents[thisRowNr * 11 + 7]; // Eul1
      map[4 + i + size_map * 2] = MicContents[thisRowNr * 11 + 8]; // Eul2
      map[4 + i + size_map * 3] = MicContents[thisRowNr * 11 + 9]; // Eul3
      map[4 + i + size_map * 4] =
          MicContents[thisRowNr * 11 + 0];         // OrientationRowNr
      map[4 + i + size_map * 5] = params->PhaseNr; // PhaseNr
      map[4 + i + size_map * 6] = lengthMat[i];    // Distt from HEDM Voxel
    }
  }

  char outfilebin[4096];
  sprintf(outfilebin, "%s.map", outputfile);
  FILE *outmap = fopen(outfilebin, "w");
  if (outmap) {
    fwrite(map, sizeof(*map) * (size_map * 7 + 4), 1, outmap);
    fclose(outmap);
  } else {
    fprintf(stderr, "Error: Could not open map output file %s\n", outfilebin);
  }

  free(map);
  free(lengthMat);
  free(RowNrMat);
}

static void ProcessAllMatches(MicParams *params, double *MicContents) {
  char inputfile2[4096], outputfile2[4096];
  sprintf(outputfile2, "%s.AllMatches", params->outputfile);
  sprintf(inputfile2, "%s.AllMatches", params->inputfile);

  FILE *inp2 = fopen(inputfile2, "rb");
  if (inp2 == NULL) {
    printf("AllMatched file not found. Skipping.\n");
    return;
  }

  size_t sz2;
  fseek(inp2, 0L, SEEK_END);
  sz2 = ftell(inp2);
  rewind(inp2);

  double *AllMatchesMicContents = malloc(sz2);
  if (AllMatchesMicContents == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for AllMatches.\n");
    fclose(inp2);
    return;
  }

  if (fread(AllMatchesMicContents, sz2, 1, inp2) != 1) {
    fprintf(stderr, "Error: Failed to read AllMatches.\n");
    free(AllMatchesMicContents);
    fclose(inp2);
    return;
  }
  fclose(inp2);

  size_t nCols = 7 + 4 * params->nSaves;
  size_t NrRows2 = sz2 / (sizeof(double) * nCols);

  FILE *out2 = fopen(outputfile2, "w");
  if (out2 == NULL) {
    fprintf(stderr, "Error: Could not open AllMatches output file %s\n",
            outputfile2);
    free(AllMatchesMicContents);
    return;
  }

  // Buffering
  char buf[1048576];
  setvbuf(out2, buf, _IOFBF, sizeof(buf));

  fprintf(out2, "%%TriEdgeSize %lf\n", MicContents[5]);
  fprintf(out2, "%%NumPhases %d\n", params->NumPhases);
  fprintf(out2, "%%GlobalPosition %lf\n", params->GlobalPosition);
  fprintf(out2,
          "%%"
          "OrientationRowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\tUpDown\tEul"
          "1\tEul2\tEul3\tConfidence\t...\t...\t...\t...\t...\t...\tPhaseNr\n");

  for (int i = 0; i < NrRows2; i++) {
    if (AllMatchesMicContents[i * nCols + 10] == 0)
      continue;
    for (int j = 0; j < nCols; j++) {
      fprintf(out2, "%lf\t", AllMatchesMicContents[i * nCols + j]);
    }
    fprintf(out2, "%d\n", params->PhaseNr);
  }

  fclose(out2);
  free(AllMatchesMicContents);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    usage();
    return 1;
  }

  MicParams params;
  if (!ReadParameters(argv[1], &params)) {
    return 1;
  }

  size_t sz;
  double *MicContents = ReadMicBinary(params.inputfile, &sz);
  if (MicContents == NULL) {
    return 1;
  }

  int NrRows = sz / (sizeof(double) * 11);
  printf("NrRows: %d\n", NrRows);

  if (NrRows > 0) {
    WriteMicText(params.outputfile, MicContents, NrRows, &params);
    GenerateMap(params.outputfile, MicContents, NrRows, &params);
    ProcessAllMatches(&params, MicContents);
  }

  free(MicContents);
  return 0;
}
