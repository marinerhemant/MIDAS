//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  MergeOverlappingPeaks.c
//
//
//  Created by Hemant Sharma on 2024/02/27.
//
//
//

#include "ZarrReader.h"
#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <zip.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#include "MIDAS_Limits.h"
#define MAXNHKLS MAX_N_HKLS
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define nOverlapsMaxPerImage 200000

// Lightweight linked-list node for tracking constituent peaks (just IDs)
typedef struct ConstituentNode {
  int frameNr;
  int peakID; // SpotID from the _PS.csv file
  struct ConstituentNode *next;
} ConstituentNode;

#define N_PS_COLS 26

int UseMaximaPositions;

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void YZ4mREta(int NrElements, double *R, double *Eta, double *Y,
                            double *Z) {
  int i;
  for (i = 0; i < NrElements; i++) {
    Y[i] = -R[i] * sin(Eta[i] * deg2rad);
    Z[i] = R[i] * cos(Eta[i] * deg2rad);
  }
}

static inline int **allocMatrixInt(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline double **allocMatrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }

struct InputData {
  double SpotID;
  double IntegratedIntensity;
  double Omega;
  double YCen;
  double ZCen;
  double IMax;
  double Radius;
  double Eta;
  double SigmaR;
  double SigmaEta;
  double NrPx;
  double NrPxTot;
  double nPeaks;
  double maxY;
  double maxZ;
  double diffY;
  double diffZ;
  double rawIMax;
  double returnCode;
  double retVal;
  double BG;
  double SigmaGR;
  double SigmaLR;
  double SigmaGEta;
  double SigmaLEta;
  double MU;
};

static int cmpfunc(const void *a, const void *b) {
  struct InputData *ia = (struct InputData *)a;
  struct InputData *ib = (struct InputData *)b;
  return (int)(1000.f * ia->Eta - 1000.f * ib->Eta);
}

static inline int CheckDirectoryCreation(char Folder[1024],
                                         char FileStem[1024]) {
  int e;
  struct stat sb;
  char totOutDir[1024];
  sprintf(totOutDir, "%s/PeakSearch/", Folder);
  e = stat(totOutDir, &sb);
  if (e != 0 && errno == ENOENT) {
    printf("Output directory did not exist, creating %s\n", totOutDir);
    e = mkdir(totOutDir, S_IRWXU);
    if (e != 0) {
      printf("Could not make the directory. Exiting\n");
      return 0;
    }
  }
  sprintf(totOutDir, "%s/PeakSearch/%s", Folder, FileStem);
  e = stat(totOutDir, &sb);
  if (e != 0 && errno == ENOENT) {
    printf("Output directory did not exist, creating %s\n", totOutDir);
    e = mkdir(totOutDir, S_IRWXU);
    if (e != 0) {
      printf("Could not make the directory. Exiting\n");
      return 0;
    }
  }
  return 1;
}

static inline int ReadSortFiles(char OutFolderName[1024], char FileStem[1024],
                                int FileNr, int Padding,
                                double **SortedMatrix) {
  char aline[1000], dummy[1000];
  char InFile[1024];
  sprintf(InFile, "%s/%s_%0*d_PS.csv", OutFolderName, FileStem, Padding,
          FileNr);
  FILE *infileread;
  infileread = fopen(InFile, "r");
  if (infileread == NULL)
    printf("Could not read the input file %s\n", InFile);
  struct InputData *MyData;
  MyData = malloc(nOverlapsMaxPerImage * sizeof(*MyData));
  int counter = 0;
  fgets(aline, 1000, infileread);
  double maxY, maxZ;
  while (fgets(aline, 1000, infileread) != NULL) {
    if (counter >= nOverlapsMaxPerImage) {
      printf("Warning: Maximum number of peaks reached (%d). Skipping rest of "
             "file %s\n",
             nOverlapsMaxPerImage, InFile);
      break;
    }
    sscanf(aline,
           "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
           "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &(MyData[counter].SpotID), &(MyData[counter].IntegratedIntensity),
           &(MyData[counter].Omega), &(MyData[counter].YCen),
           &(MyData[counter].ZCen), &(MyData[counter].IMax),
           &(MyData[counter].Radius), &(MyData[counter].Eta),
           &(MyData[counter].SigmaR), &(MyData[counter].SigmaEta),
           &(MyData[counter].NrPx), &(MyData[counter].NrPxTot),
           &(MyData[counter].nPeaks), &(MyData[counter].maxY),
           &(MyData[counter].maxZ), &(MyData[counter].diffY),
           &(MyData[counter].diffZ), &(MyData[counter].rawIMax),
           &(MyData[counter].returnCode), &(MyData[counter].retVal),
           &(MyData[counter].BG), &(MyData[counter].SigmaGR),
           &(MyData[counter].SigmaLR), &(MyData[counter].SigmaGEta),
           &(MyData[counter].SigmaLEta), &(MyData[counter].MU));
    maxY = MyData[counter].maxY;
    maxZ = MyData[counter].maxZ;
    if (UseMaximaPositions == 1) {
      MyData[counter].YCen = maxY;
      MyData[counter].ZCen = maxZ;
    }
    counter++;
  }
  fclose(infileread);
  qsort(MyData, counter, sizeof(struct InputData), cmpfunc);
  int i, j, counter2 = 0;
  for (i = 0; i < counter; i++) {
    if (MyData[i].IntegratedIntensity < 1) {
      continue;
    }
    SortedMatrix[counter2][0] = MyData[i].SpotID;
    SortedMatrix[counter2][1] = MyData[i].IntegratedIntensity;
    SortedMatrix[counter2][2] = MyData[i].Omega;
    SortedMatrix[counter2][3] = MyData[i].YCen;
    SortedMatrix[counter2][4] = MyData[i].ZCen;
    SortedMatrix[counter2][5] = MyData[i].IMax;
    SortedMatrix[counter2][6] = MyData[i].Radius;
    SortedMatrix[counter2][7] = MyData[i].Eta;
    SortedMatrix[counter2][8] = MyData[i].SigmaR;
    SortedMatrix[counter2][9] = MyData[i].SigmaEta;
    SortedMatrix[counter2][10] = MyData[i].NrPx;
    SortedMatrix[counter2][11] = MyData[i].NrPxTot;
    SortedMatrix[counter2][12] = MyData[i].nPeaks;
    SortedMatrix[counter2][13] = MyData[i].maxY;
    SortedMatrix[counter2][14] = MyData[i].maxZ;
    SortedMatrix[counter2][15] = MyData[i].diffY;
    SortedMatrix[counter2][16] = MyData[i].diffZ;
    SortedMatrix[counter2][17] = MyData[i].rawIMax;
    SortedMatrix[counter2][18] = MyData[i].returnCode;
    SortedMatrix[counter2][19] = MyData[i].retVal;
    SortedMatrix[counter2][20] = MyData[i].BG;
    SortedMatrix[counter2][21] = MyData[i].SigmaGR;
    SortedMatrix[counter2][22] = MyData[i].SigmaLR;
    SortedMatrix[counter2][23] = MyData[i].SigmaGEta;
    SortedMatrix[counter2][24] = MyData[i].SigmaLEta;
    SortedMatrix[counter2][25] = MyData[i].MU;
    counter2++;
  }
  free(MyData);
  return counter2;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage:\n MergeOverlappingPeaks ZarrZip (optional)ResultFolder\n");
    return 1;
  }
  clock_t start, end;
  double diftotal;
  start = clock();
  char *DataFN = argv[1];
  blosc2_init();
  // Read zarr config
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(DataFN, 0, &errorp);
  if (arch == NULL) {
    fprintf(stderr, "ERROR: Could not open zip archive '%s' (error code: %d)\n",
            DataFN, errorp);
    return 1;
  }
  struct zip_stat *finfo = NULL;
  finfo = calloc(16384, sizeof(int));
  zip_stat_init(finfo);
  int count = 0;
  char *Folder = NULL, FileStem[1024], *TmpFolder;
  sprintf(FileStem, "%s", basename(DataFN));
  int StartNr = 1, EndNr, Padding = 6;
  TmpFolder = "Temp";
  double MarginOmegaOverlap = sqrt(4);
  UseMaximaPositions = 0;
  int skipFrame = 0;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      ReadZarrString(arch, count, &Folder, 4096);
    }
    if (strstr(finfo->name, "exchange/data/.zarray") != NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &EndNr);
      } else {
        free(s);
        return 1;
      }
      free(s);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OverlapLength/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &MarginOmegaOverlap, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/UseMaximaPositions/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &UseMaximaPositions, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SkipFrame/0") != NULL) {
      ReadZarrChunk(arch, count, &skipFrame, sizeof(int));
    }
    count++;
  }
  printf(
      "Read Parameters from Zarr:\n\tFolder: %s\n\tEndNr: %d\n\tOverlapLength: "
      "%lf\n\tUseMaximaPositions: %d\n\tSkipFrame: %d\n",
      Folder, EndNr, MarginOmegaOverlap, UseMaximaPositions, skipFrame);
  // printf("%lf\n",MarginOmegaOverlap);
  if (argc == 3) {
    if (Folder != NULL)
      free(Folder);
    Folder = argv[2];
  }
  EndNr = EndNr - skipFrame; // This ensures we don't over-read.
                             // Read params file.
  int i, j, k;
  char OutFolderName[1024];
  char OutFileName[1024];
  sprintf(OutFolderName, "%s/%s", Folder, TmpFolder);
  char header[1024] =
      "SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px)"
      " IMax MinOme(degrees) MaxOme(degress) SigmaR SigmaEta NrPx NrPxTot "
      "Radius(px) Eta(degrees)\n";

  // Read first file
  fflush(stdout);
  int FileNr = StartNr;
  int nSpots, nSpotsNew;
  double **NewIDs, **CurrentIDs, **TempIDs;
  NewIDs = allocMatrix(nOverlapsMaxPerImage, N_PS_COLS);
  CurrentIDs = allocMatrix(nOverlapsMaxPerImage, 16);
  TempIDs = allocMatrix(nOverlapsMaxPerImage, 16);

  // Lightweight constituent tracking: per-slot linked lists of (FrameNr,
  // PeakID)
  ConstituentNode **constituents =
      calloc(nOverlapsMaxPerImage, sizeof(ConstituentNode *));
  ConstituentNode **tmpConstituents =
      calloc(nOverlapsMaxPerImage, sizeof(ConstituentNode *));

  char MergeMapFileName[2048];
  sprintf(MergeMapFileName, "%s/MergeMap.csv", Folder);
  FILE *MergeMapFile = fopen(MergeMapFileName, "w");
  fprintf(MergeMapFile, "%%MergedSpotID\tFrameNr\tPeakID\n");

  nSpots = ReadSortFiles(OutFolderName, FileStem, FileNr, Padding, NewIDs);
  for (i = 0; i < nSpots; i++) {
    CurrentIDs[i][0] = NewIDs[i][0];                // SpotID
    CurrentIDs[i][1] = NewIDs[i][1];                // IntegratedIntensity
    CurrentIDs[i][2] = NewIDs[i][2] * NewIDs[i][1]; // Omega*IntegratedIntensity
    CurrentIDs[i][3] = NewIDs[i][3] * NewIDs[i][1]; // YCen*IntegratedIntensity
    CurrentIDs[i][4] = NewIDs[i][4] * NewIDs[i][1]; // ZCen*IntegratedIntensity
    CurrentIDs[i][5] = NewIDs[i][5];                // IMax
    CurrentIDs[i][6] = NewIDs[i][6];                // Radius
    CurrentIDs[i][7] = NewIDs[i][7];                // Eta
    CurrentIDs[i][8] = NewIDs[i][3];                // YCen
    CurrentIDs[i][9] = NewIDs[i][4];                // ZCen
    CurrentIDs[i][10] = NewIDs[i][2];               // MinOmega
    CurrentIDs[i][11] = NewIDs[i][2];               // MaxOmega
    CurrentIDs[i][12] = NewIDs[i][8];               // SigmaR
    CurrentIDs[i][13] = NewIDs[i][9];               // SigmaEta
    CurrentIDs[i][14] = NewIDs[i][10];              // NrPx
    CurrentIDs[i][15] = NewIDs[i][11];              // NrPxTot
    // Record first constituent peak
    ConstituentNode *node = malloc(sizeof(ConstituentNode));
    node->frameNr = FileNr;
    node->peakID = (int)NewIDs[i][0];
    node->next = NULL;
    constituents[i] = node;
  }
  sprintf(OutFileName, "%s/Result_StartNr_%d_EndNr_%d.csv", Folder, StartNr,
          EndNr);
  FILE *OutFile;
  OutFile = fopen(OutFileName, "w");
  fprintf(OutFile, "%s", header);
  double diffLen, yThis, zThis, minLen, yFwd, zFwd, diffLenFwd;
  int *TempIDsCurrent, *TempIDsNew, BestID, IDFound;
  TempIDsCurrent = malloc(nOverlapsMaxPerImage * sizeof(*TempIDsCurrent));
  TempIDsNew = malloc(nOverlapsMaxPerImage * sizeof(*TempIDsNew));
  memset(TempIDsCurrent, 0, nOverlapsMaxPerImage * sizeof(*TempIDsCurrent));
  memset(TempIDsNew, 0, nOverlapsMaxPerImage * sizeof(*TempIDsNew));
  int SpotIDNr = 1, counter;
  if (StartNr == EndNr) { // If there is only one file.
    for (i = 0; i < nSpots; i++) {
      fprintf(
          OutFile, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
          SpotIDNr, NewIDs[i][1], NewIDs[i][2], NewIDs[i][3], NewIDs[i][4],
          NewIDs[i][5], NewIDs[i][2], NewIDs[i][2], NewIDs[i][8], NewIDs[i][9],
          NewIDs[i][10], NewIDs[i][11], NewIDs[i][6], NewIDs[i][7]);
      // Write MergeMap entry (single frame, single constituent)
      fprintf(MergeMapFile, "%d\t%d\t%d\n", SpotIDNr, FileNr,
              (int)NewIDs[i][0]);
      SpotIDNr++;
    }
  } else { // If there are multiple files:
    for (FileNr = (StartNr + 1); FileNr <= EndNr; FileNr++) {
      nSpotsNew =
          ReadSortFiles(OutFolderName, FileStem, FileNr, Padding, NewIDs);
      fflush(stdout);
      for (i = 0; i < nSpots; i++) {
        minLen = 10000000;
        IDFound = 0;
        yThis = CurrentIDs[i][8];
        zThis = CurrentIDs[i][9];
        for (j = 0; j < nSpotsNew;
             j++) { // Try to find the smallest difference in Y,Z.
          if (TempIDsNew[j] != 1) {
            diffLen = CalcNorm2(NewIDs[j][3] - yThis, NewIDs[j][4] - zThis);
            if (diffLen < MarginOmegaOverlap && diffLen < minLen) {
              minLen = diffLen;
              BestID = j;
              IDFound = 1;
            }
          }
        }
        if (IDFound == 1) { // If a candidate for overlapping has been detected,
                            // check if it is the best pair.
          yFwd = NewIDs[BestID][3];
          zFwd = NewIDs[BestID][4];
          for (k = 0; k < nSpots; k++) {
            if (k != i && TempIDsCurrent[k] != 1) {
              diffLenFwd =
                  CalcNorm2(CurrentIDs[k][8] - yFwd, CurrentIDs[k][9] - zFwd);
              if (diffLenFwd < minLen) {
                IDFound = 0;
                break;
              }
            }
          }
        }
        if (IDFound == 1) { // If the best pair for overlapping was found,
                            // update current IDs.
          TempIDsCurrent[i] = 1;
          TempIDsNew[BestID] = 1;
          CurrentIDs[i][1] += NewIDs[BestID][1];
          CurrentIDs[i][2] += (NewIDs[BestID][2] * NewIDs[BestID][1]);
          CurrentIDs[i][3] += (NewIDs[BestID][3] * NewIDs[BestID][1]);
          CurrentIDs[i][4] += (NewIDs[BestID][4] * NewIDs[BestID][1]);
          if (CurrentIDs[i][5] < NewIDs[BestID][5]) {
            CurrentIDs[i][5] = NewIDs[BestID][5]; // IMax update
          }
          CurrentIDs[i][6] = NewIDs[BestID][6];
          CurrentIDs[i][7] = NewIDs[BestID][7];
          CurrentIDs[i][8] = NewIDs[BestID][3]; // Ycen
          CurrentIDs[i][9] = NewIDs[BestID][4]; // ZCen
          if (CurrentIDs[i][10] > NewIDs[BestID][2]) {
            CurrentIDs[i][10] = NewIDs[BestID][2]; // MinOme
          }
          if (CurrentIDs[i][11] < NewIDs[BestID][2]) {
            CurrentIDs[i][11] = NewIDs[BestID][2]; // MaxOme
          }
          if (CurrentIDs[i][12] < NewIDs[BestID][8]) {
            CurrentIDs[i][12] = NewIDs[BestID][8]; // SigmaR
          }
          if (CurrentIDs[i][13] < NewIDs[BestID][9]) {
            CurrentIDs[i][13] = NewIDs[BestID][9]; // SigmaEta
          }
          CurrentIDs[i][14] += NewIDs[BestID][10]; // NrPx
          CurrentIDs[i][15] += NewIDs[BestID][11]; // NrPxTot
          // Record constituent peak (prepend to linked list)
          ConstituentNode *node = malloc(sizeof(ConstituentNode));
          node->frameNr = FileNr;
          node->peakID = (int)NewIDs[BestID][0];
          node->next = constituents[i];
          constituents[i] = node;
        }
      }
      // Write all the spots not overlapping to the output file.
      for (i = 0; i < nSpots; i++) {
        if (TempIDsCurrent[i] == 0) { // Spot was not overlapping.
          fprintf(OutFile,
                  "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                  SpotIDNr, CurrentIDs[i][1],
                  (CurrentIDs[i][2] / CurrentIDs[i][1]),
                  (CurrentIDs[i][3] / CurrentIDs[i][1]),
                  (CurrentIDs[i][4] / CurrentIDs[i][1]), CurrentIDs[i][5],
                  CurrentIDs[i][10], CurrentIDs[i][11], CurrentIDs[i][12],
                  CurrentIDs[i][13], CurrentIDs[i][14], CurrentIDs[i][15],
                  CurrentIDs[i][6], CurrentIDs[i][7]);
          // Write MergeMap entries for this finalized spot
          {
            ConstituentNode *cn = constituents[i];
            while (cn) {
              fprintf(MergeMapFile, "%d\t%d\t%d\n", SpotIDNr, cn->frameNr,
                      cn->peakID);
              cn = cn->next;
            }
            // Free the list
            cn = constituents[i];
            while (cn) {
              ConstituentNode *tmp = cn->next;
              free(cn);
              cn = tmp;
            }
            constituents[i] = NULL;
          }
          SpotIDNr++;
        }
      }
      // Reset everything for the next file.
      counter = 0;
      for (i = 0; i < nSpots; i++) {
        if (TempIDsCurrent[i] == 1) { // Spot was overlapping.
          if (counter >= nOverlapsMaxPerImage) {
            printf("Warning: Maximum number of merged peaks reached (%d). "
                   "Truncating.\n",
                   nOverlapsMaxPerImage);
            break;
          }
          for (j = 0; j < 16; j++) {
            TempIDs[counter][j] = CurrentIDs[i][j];
          }
          // Move constituent list to tmp
          tmpConstituents[counter] = constituents[i];
          constituents[i] = NULL;
          counter++;
        }
      }
      for (i = 0; i < nSpotsNew; i++) {
        if (counter >= nOverlapsMaxPerImage) {
          printf("Warning: Maximum number of merged peaks reached (%d). "
                 "Truncating.\n",
                 nOverlapsMaxPerImage);
          break;
        }
        if (TempIDsNew[i] == 0) {             // Spot was not overlapping.
          TempIDs[counter][0] = NewIDs[i][0]; // SpotID
          TempIDs[counter][1] = NewIDs[i][1]; // IntegratedIntensity
          TempIDs[counter][2] =
              NewIDs[i][2] * NewIDs[i][1]; // Omega*IntegratedIntensity
          TempIDs[counter][3] =
              NewIDs[i][3] * NewIDs[i][1]; // YCen*IntegratedIntensity
          TempIDs[counter][4] =
              NewIDs[i][4] * NewIDs[i][1];      // ZCen*IntegratedIntensity
          TempIDs[counter][5] = NewIDs[i][5];   // IMax
          TempIDs[counter][6] = NewIDs[i][6];   // Radius
          TempIDs[counter][7] = NewIDs[i][7];   // Eta
          TempIDs[counter][8] = NewIDs[i][3];   // YCen
          TempIDs[counter][9] = NewIDs[i][4];   // ZCen
          TempIDs[counter][10] = NewIDs[i][2];  // MinOmega
          TempIDs[counter][11] = NewIDs[i][2];  // MaxOmega
          TempIDs[counter][12] = NewIDs[i][8];  // SigmaR
          TempIDs[counter][13] = NewIDs[i][9];  // SigmaEta
          TempIDs[counter][14] = NewIDs[i][10]; // NrPx
          TempIDs[counter][15] = NewIDs[i][11]; // NrPxTot
          // New spot: single constituent
          ConstituentNode *node = malloc(sizeof(ConstituentNode));
          node->frameNr = FileNr;
          node->peakID = (int)NewIDs[i][0];
          node->next = NULL;
          tmpConstituents[counter] = node;
          counter++;
        }
      }
      if (counter != nSpotsNew) {
        printf("Number of spots mismatch. Please have a look.\n");
      }
      for (i = 0; i < nSpots; i++) {
        for (j = 0; j < 16; j++) {
          CurrentIDs[i][j] = 0;
        }
      }
      for (i = 0; i < nSpotsNew; i++) {
        for (j = 0; j < 16; j++) {
          CurrentIDs[i][j] = TempIDs[i][j];
        }
        // Move constituent list back
        constituents[i] = tmpConstituents[i];
        tmpConstituents[i] = NULL;
      }
      nSpots = nSpotsNew;
      memset(TempIDsCurrent, 0, nOverlapsMaxPerImage * sizeof(*TempIDsCurrent));
      memset(TempIDsNew, 0, nOverlapsMaxPerImage * sizeof(*TempIDsNew));
    }
  }
  for (i = 0; i < nSpots; i++) {
    fprintf(OutFile, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            SpotIDNr, CurrentIDs[i][1], (CurrentIDs[i][2] / CurrentIDs[i][1]),
            (CurrentIDs[i][3] / CurrentIDs[i][1]),
            (CurrentIDs[i][4] / CurrentIDs[i][1]), CurrentIDs[i][5],
            CurrentIDs[i][10], CurrentIDs[i][11], CurrentIDs[i][12],
            CurrentIDs[i][13], CurrentIDs[i][14], CurrentIDs[i][15],
            CurrentIDs[i][6], CurrentIDs[i][7]);
    // Write final MergeMap entries
    {
      ConstituentNode *cn = constituents[i];
      while (cn) {
        fprintf(MergeMapFile, "%d\t%d\t%d\n", SpotIDNr, cn->frameNr,
                cn->peakID);
        ConstituentNode *tmp = cn->next;
        free(cn);
        cn = tmp;
      }
      constituents[i] = NULL;
    }
    SpotIDNr++;
  }
  printf("Total spots: %d\n", SpotIDNr - 1);
  fclose(MergeMapFile);
  printf("MergeMap written to: %s\n", MergeMapFileName);
  FreeMemMatrix(NewIDs, nOverlapsMaxPerImage);
  FreeMemMatrix(CurrentIDs, nOverlapsMaxPerImage);
  FreeMemMatrix(TempIDs, nOverlapsMaxPerImage);
  free(TempIDsCurrent);
  free(TempIDsNew);
  free(constituents);
  free(tmpConstituents);
  free(finfo);
  if (argc < 3)
    free(Folder);
  zip_close(arch);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
