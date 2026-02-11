//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define float32_t float
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

int Flag = 0;

struct Theader {
  uint32_t uBlockHeader;
  uint16_t BlockType;
  uint16_t DataFormat;
  uint16_t NumChildren;
  uint16_t NameSize;
  char BlockName[4096];
  uint32_t DataSize;
  uint16_t ChunkNumber;
  uint16_t TotalChunks;
};

void ReadHeader(FILE *fp, struct Theader *head) {
  size_t sz;
  sz = fread(&head->uBlockHeader, sizeof(uint32_t), 1, fp);
  sz = fread(&head->BlockType, sizeof(uint16_t), 1, fp);
  sz = fread(&head->DataFormat, sizeof(uint16_t), 1, fp);
  sz = fread(&head->NumChildren, sizeof(uint16_t), 1, fp);
  sz = fread(&head->NameSize, sizeof(uint16_t), 1, fp);
  sz = fread(&head->DataSize, sizeof(uint32_t), 1, fp);
  sz = fread(&head->ChunkNumber, sizeof(uint16_t), 1, fp);
  sz = fread(&head->TotalChunks, sizeof(uint16_t), 1, fp);
  sz = fread(&head->BlockName, (sizeof(char) * (head->NameSize)), 1, fp);
}

static inline void realloc_buffers(int nElements, int nElements_previous,
                                   uint16_t **ys, uint16_t **zs,
                                   uint16_t **peakID, float32_t **intensity) {
  if (nElements > nElements_previous) {
    *ys = realloc(*ys, nElements * sizeof(**ys));
    *zs = realloc(*zs, nElements * sizeof(**zs));
    *peakID = realloc(*peakID, nElements * sizeof(**peakID));
    *intensity = realloc(*intensity, nElements * sizeof(**intensity));
  }
}

int checkFOPEN(FILE *f, char *fn) {
  if (f == NULL) {
    printf("Could not open %s\n", fn);
    return 1;
  } else
    return 0;
}

int ReadBinFiles(char FileStem[1000], char *ext, int StartNr, int EndNr,
                 int *ObsSpotsMat, int nLayers, long long int ObsSpotsSize) {
  int i, j, k, nElements = 0, nElements_previous, nCheck, ythis, zthis,
               NrOfFiles;
  long long NrOfPixels;
  long long int BinNr;
  long long int TempCntr;
  float32_t dummy;
  uint32_t dummy2;
  FILE *fp;
  char FileName[1024];
  struct Theader Header1;
  uint16_t *ys = NULL, *zs = NULL, *peakID = NULL;
  float32_t *intensity = NULL;
  int counter = 0;
  NrOfFiles = EndNr - StartNr + 1;
  NrOfPixels = 2048 * 2048;
  long long int kT;
  kT = nLayers;
  kT *= NrOfPixels;
  kT *= NrOfFiles;
  for (k = 0; k < nLayers; k++) {
    for (i = StartNr; i <= EndNr; i++) {
      sprintf(FileName, "%s_%06d.%s%d", FileStem, i, ext, k);
      fp = fopen(FileName, "r");
      if (checkFOPEN(fp, FileName))
        return 0;
      size_t sz;
      sz = fread(&dummy, sizeof(float32_t), 1, fp);
      ReadHeader(fp, &Header1);
      sz = fread(&dummy2, sizeof(uint32_t), 1, fp);
      sz = fread(&dummy2, sizeof(uint32_t), 1, fp);
      sz = fread(&dummy2, sizeof(uint32_t), 1, fp);
      sz = fread(&dummy2, sizeof(uint32_t), 1, fp);
      sz = fread(&dummy2, sizeof(uint32_t), 1, fp);
      ReadHeader(fp, &Header1);
      nElements_previous = nElements;
      nElements = (Header1.DataSize - Header1.NameSize) / 2;
      realloc_buffers(nElements, nElements_previous, &ys, &zs, &peakID,
                      &intensity);
      sz = fread(ys, sizeof(uint16_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 2;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      sz = fread(zs, sizeof(uint16_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 4;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      sz = fread(intensity, sizeof(float32_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 2;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      sz = fread(peakID, sizeof(uint16_t) * nElements, 1, fp);
      for (j = 0; j < nElements; j++) {
        ythis = (int)ys[j];
        zthis = (int)zs[j];
        if (Flag == 1)
          zthis = 2048 - zthis;
        BinNr = k;
        BinNr *= NrOfFiles;
        BinNr *= NrOfPixels;
        TempCntr = counter;
        TempCntr *= NrOfPixels;
        BinNr += TempCntr;
        BinNr += (ythis * (2048));
        BinNr += zthis;
        if (BinNr < 0 || BinNr >= kT) {
          printf("%lld %d %d %d %lld %d %d %d %d %d %d %d %d\n", BinNr, k,
                 NrOfFiles, NrOfPixels, TempCntr, ythis, zthis, nElements, j,
                 (int)ys[j], (int)zs[j], i);
          printf("Something was wrong with the Binary Files. Distance %d and "
                 "FileNr %d contained %d for y and %d for z position. Please "
                 "check, exiting.\n",
                 k, i, ythis, zthis);
          return 0;
        }
        SetBit(ObsSpotsMat, BinNr);
      }
      fclose(fp);
      counter += 1;
    }
    counter = 0;
  }
  return 1;
  free(ys);
  free(zs);
  free(peakID);
  free(intensity);
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double diftotal;
  start = clock();

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000], line[1000];
  fileParam = fopen(ParamFN, "r");
  if (checkFOPEN(fileParam, ParamFN))
    return 1;
  char *str, dummy[1000];
  int LowNr, nLayers;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "nDistances ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nLayers);
      break;
    }
  }
  rewind(fileParam);
  char fn[1000];
  char fn2[1000];
  char direct[1000];
  int StartNr, EndNr, skipBin = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
    str = "StartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "EndNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &EndNr);
      continue;
    }
    str = "Ice9Input ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      Flag = 1;
      continue;
    }
    str = "SkipImageBinning ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      skipBin = 1;
      continue;
    }
  }

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("          MMapImageInfo: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:      %s\n", direct);
  printf("  ReducedFileName:    %s\n", fn2);
  printf("\n--- Scan Parameters ---\n");
  printf("  nDistances (nLayers): %d\n", nLayers);
  printf("  StartNr:              %d\n", StartNr);
  printf("  EndNr:                %d\n", EndNr);
  printf("\n--- Flags ---\n");
  printf("  Ice9Input:            %s\n", Flag ? "YES" : "NO");
  printf("  SkipImageBinning:     %s\n", skipBin ? "YES" : "NO");
  printf(
      "================================================================\n\n");

  fclose(fileParam);
  // Read bin files
  char fnG[1000];
  sprintf(fnG, "%s/grid.txt", direct);
  char fnDS[1000];
  char fnKey[1000];
  char fnOr[1000];
  sprintf(fnDS, "%s/DiffractionSpots.txt", direct);
  sprintf(fnKey, "%s/Key.txt", direct);
  sprintf(fnOr, "%s/OrientMat.txt", direct);
  sprintf(fn, "%s/%s", direct, fn2);
  int i, j, m, nrFiles, nrPixels;
  char *ext = "bin";
  int *ObsSpotsInfo;
  int ReadCode;
  nrFiles = EndNr - StartNr + 1;
  nrPixels = 2048 * 2048;
  long long int SizeObsSpots;
  SizeObsSpots = (nLayers);
  SizeObsSpots *= nrPixels;
  SizeObsSpots *= nrFiles;
  SizeObsSpots /= 32;
  ObsSpotsInfo = malloc(SizeObsSpots * sizeof(*ObsSpotsInfo));
  memset(ObsSpotsInfo, 0, SizeObsSpots * sizeof(*ObsSpotsInfo));
  if (ObsSpotsInfo == NULL) {
    printf("Could not allocate ObsSpotsInfo.\n");
    return 0;
  }
  if (skipBin == 0)
    ReadCode = ReadBinFiles(fn, ext, StartNr, EndNr, ObsSpotsInfo, nLayers,
                            SizeObsSpots);
  else
    ReadCode = 1;
  if (ReadCode == 0) {
    printf("Reading bin files did not go well. Please check.\n");
    return 1;
  }
  printf("Read Orientations\n");
  clock_t startthis;
  startthis = clock();
  FILE *fd, *fk, *fo;
  int NrOrientations, TotalDiffrSpots;
  fd = fopen(fnDS, "r");
  fk = fopen(fnKey, "r");
  fo = fopen(fnOr, "r");
  if (checkFOPEN(fd, fnDS))
    return 1;
  if (checkFOPEN(fk, fnKey))
    return 1;
  if (checkFOPEN(fo, fnOr))
    return 1;
  char *rx;
  rx = fgets(line, 1000, fk);
  sscanf(line, "%d", &NrOrientations);
  int *NrSpots;
  NrSpots = malloc(NrOrientations * 2 * sizeof(*NrSpots));
  TotalDiffrSpots = 0;
  for (i = 0; i < NrOrientations; i++) {
    rx = fgets(line, 1000, fk);
    sscanf(line, "%d", &NrSpots[2 * i]);
    TotalDiffrSpots += NrSpots[2 * i];
    NrSpots[2 * i + 1] = TotalDiffrSpots - NrSpots[2 * i];
  }
  fclose(fk);
  double *SpotsMat;
  SpotsMat = malloc(TotalDiffrSpots * 3 * sizeof(*SpotsMat));
  for (i = 0; i < TotalDiffrSpots; i++) {
    rx = fgets(line, 1000, fd);
    sscanf(line, "%lf %lf %lf", &SpotsMat[3 * i], &SpotsMat[3 * i + 1],
           &SpotsMat[3 * i + 2]);
  }
  fclose(fd);
  double *OrientationMatrix;
  OrientationMatrix = malloc(NrOrientations * 9 * sizeof(*OrientationMatrix));
  printf("ObsSpotsInfo: %llu mb, NrSpots: %lu mb, SpotsMat: %lu mb and "
         "Orientation Matrix: %lu mb\n",
         SizeObsSpots * sizeof(*ObsSpotsInfo) / (1024 * 1024),
         NrOrientations * 2 * sizeof(*NrSpots) / (1024 * 1024),
         TotalDiffrSpots * 3 * sizeof(*SpotsMat) / (1024 * 1024),
         NrOrientations * 9 * sizeof(*OrientationMatrix) / (1024 * 1024));
  for (i = 0; i < NrOrientations; i++) {
    rx = fgets(line, 1000, fo);
    sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &OrientationMatrix[9 * i], &OrientationMatrix[9 * i + 1],
           &OrientationMatrix[9 * i + 2], &OrientationMatrix[9 * i + 3],
           &OrientationMatrix[9 * i + 4], &OrientationMatrix[9 * i + 5],
           &OrientationMatrix[9 * i + 6], &OrientationMatrix[9 * i + 7],
           &OrientationMatrix[9 * i + 8]);
  }
  fclose(fo);
  char SI[1024];
  sprintf(SI, "%s/SpotsInfo.bin", direct);
  char DS[1024];
  sprintf(DS, "%s/DiffractionSpots.bin", direct);
  char KEY[1024];
  sprintf(KEY, "%s/Key.bin", direct);
  char OM[1024];
  sprintf(OM, "%s/OrientMat.bin", direct);
  FILE *fSI, *fDS, *fKEY, *fOM;
  if (skipBin == 0)
    fSI = fopen(SI, "wb");
  fDS = fopen(DS, "wb");
  fKEY = fopen(KEY, "wb");
  fOM = fopen(OM, "wb");
  if (skipBin == 0)
    if (checkFOPEN(fSI, SI))
      return 1;
  if (checkFOPEN(fDS, DS))
    return 1;
  if (checkFOPEN(fKEY, KEY))
    return 1;
  if (checkFOPEN(fOM, OM))
    return 1;
  if (skipBin == 0)
    fwrite(ObsSpotsInfo, SizeObsSpots * sizeof(*ObsSpotsInfo), 1, fSI);
  fwrite(SpotsMat, TotalDiffrSpots * 3 * sizeof(*SpotsMat), 1, fDS);
  fwrite(NrSpots, NrOrientations * 2 * sizeof(*NrSpots), 1, fKEY);
  fwrite(OrientationMatrix, NrOrientations * 9 * sizeof(*OrientationMatrix), 1,
         fOM);
  // fcloseall();
}
