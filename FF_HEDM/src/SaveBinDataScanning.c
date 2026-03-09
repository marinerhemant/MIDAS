//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// SaveBinDataScanning.c
//
// Created by Hemant Sharma on 2014/11/07
// WE NEED TO UPDATE PARAMSTEST.TXT with RingToIndex
//
//

#include "midas_version.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define N_COL_OBSSPOTS 10             // This is one less number of columns
#define INITIAL_SPOT_CAPACITY 1000000 // start with 1M spots, grow dynamically
#define MAX_N_RINGS                                                            \
  500 // max nr of rings that can be stored (applies to the arrays ringttheta,
      // ringhkl, etc)

void CalcDistanceIdealRing(double *ObsSpotsLab, int nspots, int nCols,
                           double RingRadii[]) {
  int i;
  for (i = 0; i < nspots; ++i) {
    double y = ObsSpotsLab[i * nCols + 0];
    double z = ObsSpotsLab[i * nCols + 1];
    double rad = sqrt(y * y + z * z);
    int ringno = (int)ObsSpotsLab[i * nCols + 5];
    if (ringno < 0 || ringno >= MAX_N_RINGS) {
      fprintf(stderr,
              "ERROR: CalcDistanceIdealRing: spot %d has invalid "
              "ringno=%d (must be 0..%d). Skipping.\n",
              i, ringno, MAX_N_RINGS - 1);
      ObsSpotsLab[i * nCols + 17] = 0;
      continue;
    }
    ObsSpotsLab[i * nCols + 17] = rad - RingRadii[ringno];
  }
}

struct InpData {
  double Values[17];
};

static int cmpfunc(const void *a, const void *b) {
  struct InpData *ia = (struct InpData *)a;
  struct InpData *ib = (struct InpData *)b;
  if (ia->Values[5] < ib->Values[5])
    return -1;
  else if (ib->Values[5] < ia->Values[5])
    return 1;
  else {
    if (ia->Values[2] < ib->Values[2])
      return -1;
    else if (ib->Values[2] < ia->Values[2])
      return 1;
    else {
      if (ia->Values[6] < ib->Values[6])
        return -1;
      else if (ib->Values[6] < ia->Values[6])
        return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  clock_t start, end;
  if (argc != 2) {
    fprintf(stderr, "ERROR: Usage: SaveBinDataScanning nScans\n");
    printf("Usage: SaveBinDataScanning nScans\n");
    return 1;
  }
  int nScans = atoi(argv[1]);
  if (nScans <= 0) {
    fprintf(stderr, "ERROR: nScans=%d is invalid (must be > 0)\n", nScans);
    return 1;
  }
  printf("SaveBinDataScanning: nScans=%d\n", nScans);
  fprintf(stderr, "INFO: SaveBinDataScanning starting with nScans=%d\n",
          nScans);

  double diftotal;
  start = clock();
  char *ParamFN = "paramstest.txt", dummy[1024], *str;
  char aline[4096];
  int LowNr;
  FILE *fileParam;
  fileParam = fopen(ParamFN, "r");
  if (fileParam == NULL) {
    fprintf(stderr,
            "ERROR: Could not open parameter file '%s': %s (errno=%d)\n",
            ParamFN, strerror(errno), errno);
    printf("Could not open %s. Exiting.\n", ParamFN);
    return 1;
  }
  fprintf(stderr, "INFO: Opened parameter file '%s'\n", ParamFN);

  int NrOfRings = 0, NoRingNumbers = 0, RingNumbers[MAX_N_RINGS];
  double omemargin0 = -1, etamargin0 = -1, rotationstep = -1,
         RingRadii[MAX_N_RINGS], RingRadiiUser[MAX_N_RINGS], etabinsize = -1,
         omebinsize = -1;
  int nosaveall = 0;

  // Track which required params were found
  int found_marginome = 0, found_margineta = 0, found_etabinsize = 0,
      found_omebinsize = 0, found_stepsize = 0;

  while (fgets(aline, 4096, fileParam) != NULL) {
    str = "NoSaveAll ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nosaveall);
      continue;
    }
    str = "MarginOme ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &omemargin0);
      found_marginome = 1;
      continue;
    }
    str = "MarginEta ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &etamargin0);
      found_margineta = 1;
      continue;
    }
    str = "EtaBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &etabinsize);
      found_etabinsize = 1;
      continue;
    }
    str = "StepsizeOrient ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &rotationstep);
      found_stepsize = 1;
      continue;
    }
    str = "OmeBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &omebinsize);
      found_omebinsize = 1;
      continue;
    }
    str = "RingRadii ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      if (NrOfRings >= MAX_N_RINGS) {
        fprintf(stderr, "ERROR: Too many RingRadii entries (max=%d)\n",
                MAX_N_RINGS);
        return 1;
      }
      sscanf(aline, "%s %lf", dummy, &RingRadiiUser[NrOfRings]);
      NrOfRings++;
      continue;
    }
    str = "RingNumbers ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      if (NoRingNumbers >= MAX_N_RINGS) {
        fprintf(stderr, "ERROR: Too many RingNumbers entries (max=%d)\n",
                MAX_N_RINGS);
        return 1;
      }
      sscanf(aline, "%s %d", dummy, &RingNumbers[NoRingNumbers]);
      NoRingNumbers++;
      continue;
    }
  }
  fclose(fileParam);

  // ---- Validate required parameters ----
  fprintf(stderr, "INFO: Parameter summary:\n");
  fprintf(stderr, "INFO:   NrOfRings=%d, NoRingNumbers=%d\n", NrOfRings,
          NoRingNumbers);
  fprintf(stderr, "INFO:   MarginOme=%.4f (found=%d)\n", omemargin0,
          found_marginome);
  fprintf(stderr, "INFO:   MarginEta=%.4f (found=%d)\n", etamargin0,
          found_margineta);
  fprintf(stderr, "INFO:   EtaBinSize=%.4f (found=%d)\n", etabinsize,
          found_etabinsize);
  fprintf(stderr, "INFO:   OmeBinSize=%.4f (found=%d)\n", omebinsize,
          found_omebinsize);
  fprintf(stderr, "INFO:   StepsizeOrient=%.6f (found=%d)\n", rotationstep,
          found_stepsize);
  fprintf(stderr, "INFO:   NoSaveAll=%d\n", nosaveall);

  int param_error = 0;
  if (!found_marginome) {
    fprintf(stderr, "ERROR: Required parameter 'MarginOme' not found in %s\n",
            ParamFN);
    param_error = 1;
  }
  if (!found_margineta) {
    fprintf(stderr, "ERROR: Required parameter 'MarginEta' not found in %s\n",
            ParamFN);
    param_error = 1;
  }
  if (!found_etabinsize) {
    fprintf(stderr, "ERROR: Required parameter 'EtaBinSize' not found in %s\n",
            ParamFN);
    param_error = 1;
  }
  if (!found_omebinsize) {
    fprintf(stderr, "ERROR: Required parameter 'OmeBinSize' not found in %s\n",
            ParamFN);
    param_error = 1;
  }
  if (!found_stepsize) {
    fprintf(stderr,
            "ERROR: Required parameter 'StepsizeOrient' not found in %s\n",
            ParamFN);
    param_error = 1;
  }
  if (NrOfRings == 0) {
    fprintf(stderr, "ERROR: No 'RingRadii' entries found in %s\n", ParamFN);
    param_error = 1;
  }
  if (NoRingNumbers == 0) {
    fprintf(stderr, "ERROR: No 'RingNumbers' entries found in %s\n", ParamFN);
    param_error = 1;
  }
  if (NrOfRings != NoRingNumbers) {
    fprintf(stderr,
            "ERROR: NrOfRings(%d) != NoRingNumbers(%d) — mismatch in %s\n",
            NrOfRings, NoRingNumbers, ParamFN);
    param_error = 1;
  }
  if (param_error) {
    fprintf(stderr, "ERROR: Missing required parameters. Cannot proceed.\n");
    return 1;
  }

  // Validate parameter values
  if (etabinsize <= 0) {
    fprintf(stderr, "ERROR: EtaBinSize=%.4f is invalid (must be > 0)\n",
            etabinsize);
    return 1;
  }
  if (omebinsize <= 0) {
    fprintf(stderr, "ERROR: OmeBinSize=%.4f is invalid (must be > 0)\n",
            omebinsize);
    return 1;
  }

  // Log ring info
  for (int r = 0; r < NrOfRings; r++) {
    fprintf(stderr, "INFO:   Ring[%d]: number=%d, radius=%.4f\n", r,
            RingNumbers[r], RingRadiiUser[r]);
  }

  int i, j, k;

  int scanNr;
  size_t nSpots = 0;
  size_t spotCapacity = INITIAL_SPOT_CAPACITY;

  char AllSpotsFN[4096];
  FILE *AllSpotsFile;
  char *rc;
  struct InpData *MyData;
  MyData = malloc(spotCapacity * sizeof(*MyData));
  if (MyData == NULL) {
    fprintf(stderr, "ERROR: Could not allocate MyData array (%zu bytes)\n",
            spotCapacity * sizeof(*MyData));
    printf("Memory error: could not allocate MyData.\n");
    return 1;
  }
  fprintf(stderr, "INFO: Allocated MyData array for %zu spots (%.1f MB)\n",
          spotCapacity,
          (double)(spotCapacity * sizeof(*MyData)) / (1024.0 * 1024.0));

  int nFilesFound = 0;
  int nFilesNotFound = 0;
  for (scanNr = 0; scanNr < nScans; scanNr++) {
    sprintf(AllSpotsFN, "InputAllExtraInfoFittingAll%d.csv", scanNr);
    AllSpotsFile = fopen(AllSpotsFN, "r");
    if (AllSpotsFile == NULL) {
      fprintf(stderr,
              "WARNING: Could not open '%s': %s (errno=%d). Skipping.\n",
              AllSpotsFN, strerror(errno), errno);
      printf("Could not open %s. Skipping.\n", AllSpotsFN);
      nFilesNotFound++;
      continue;
    }
    nFilesFound++;
    setvbuf(AllSpotsFile, NULL, _IOFBF, 1 << 20); // 1 MB read buffer
    rc = fgets(aline, 4096, AllSpotsFile);        // skip header
    if (rc == NULL) {
      fprintf(stderr, "WARNING: File '%s' is empty or has no header line.\n",
              AllSpotsFN);
      fclose(AllSpotsFile);
      continue;
    }
    size_t nSpotsThisScan = 0;
    size_t nLinesRead = 0;
    size_t nParseErrors = 0;
    while (fgets(aline, 4096, AllSpotsFile) != NULL) {
      nLinesRead++;
      double dummy0, dummy1;
      int nParsed = sscanf(
          aline,
          "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
          "%lf %lf",
          &MyData[nSpots].Values[0], &MyData[nSpots].Values[1],
          &MyData[nSpots].Values[2], &MyData[nSpots].Values[3],
          &MyData[nSpots].Values[4], &MyData[nSpots].Values[5],
          &MyData[nSpots].Values[6], &MyData[nSpots].Values[7],
          &MyData[nSpots].Values[8], &MyData[nSpots].Values[9],
          &MyData[nSpots].Values[10], &MyData[nSpots].Values[11],
          &MyData[nSpots].Values[12], &MyData[nSpots].Values[13], &dummy0,
          &dummy1, &MyData[nSpots].Values[14], &MyData[nSpots].Values[15]);

      if (nParsed != 18) {
        // Check if this is a zero-filled placeholder row (all values ~0
        // except spotID at index 4). These are empty entries from the
        // pipeline and would be filtered by the Values[3] check anyway.
        int is_zero_row = (nParsed >= 6);
        for (int zi = 0; zi < nParsed && zi < 16 && is_zero_row; zi++) {
          if (zi == 4)
            continue; // skip spotID
          double val = (zi < 14)    ? MyData[nSpots].Values[zi]
                       : (zi == 14) ? dummy0
                                    : dummy1;
          if (fabs(val) > 0.001)
            is_zero_row = 0;
        }
        if (!is_zero_row) {
          nParseErrors++;
          if (nParseErrors <= 5) {
            fprintf(stderr,
                    "WARNING: Parse error in '%s' line %zu: expected 18 "
                    "fields, got %d. Line: %.80s...\n",
                    AllSpotsFN, nLinesRead + 1, nParsed, aline);
          }
        }
        continue;
      }

      // Validate ring number bounds
      int ringno = (int)MyData[nSpots].Values[5];
      if (ringno < 0 || ringno >= MAX_N_RINGS) {
        fprintf(stderr,
                "WARNING: Spot in '%s' line %zu has invalid ringno=%d "
                "(must be 0..%d). Skipping.\n",
                AllSpotsFN, nLinesRead + 1, ringno, MAX_N_RINGS - 1);
        continue;
      }

      MyData[nSpots].Values[16] = scanNr;
      if (fabs(MyData[nSpots].Values[3]) > 0.0001) {
        nSpots++;
        nSpotsThisScan++;
      }

      // Grow array if full
      if (nSpots >= spotCapacity) {
        size_t newCap = spotCapacity * 2;
        struct InpData *grown = realloc(MyData, newCap * sizeof(*MyData));
        if (grown == NULL) {
          fprintf(stderr,
                  "ERROR: Could not grow MyData from %zu to %zu spots "
                  "(%.1f GB requested). Processing with %zu spots.\n",
                  spotCapacity, newCap,
                  (double)(newCap * sizeof(*MyData)) /
                      (1024.0 * 1024.0 * 1024.0),
                  nSpots);
          break;
        }
        MyData = grown;
        spotCapacity = newCap;
        fprintf(stderr, "INFO: Grew MyData to %zu spots (%.1f MB)\n",
                spotCapacity,
                (double)(spotCapacity * sizeof(*MyData)) / (1024.0 * 1024.0));
      }
    }
    fclose(AllSpotsFile);
    fprintf(stderr,
            "INFO: Scan %d/%d: read %zu data lines from '%s', "
            "%zu valid spots (total so far: %zu)",
            scanNr + 1, nScans, nLinesRead, AllSpotsFN, nSpotsThisScan, nSpots);
    if (nParseErrors > 0) {
      fprintf(stderr, ", %zu parse errors", nParseErrors);
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "INFO: Input files: %d found, %d not found (out of %d)\n",
          nFilesFound, nFilesNotFound, nScans);

  if (nFilesFound == 0) {
    fprintf(stderr,
            "ERROR: No input files found! Expected "
            "InputAllExtraInfoFittingAll0.csv .. "
            "InputAllExtraInfoFittingAll%d.csv in current directory.\n",
            nScans - 1);
    // List current directory for debugging
    fprintf(stderr,
            "ERROR: Current working directory contents (first check):\n");
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      fprintf(stderr, "ERROR: CWD = %s\n", cwd);
    }
    free(MyData);
    return 1;
  }

  if (nSpots == 0) {
    fprintf(stderr,
            "ERROR: No valid spots found across %d scans (%d files). "
            "All spots may have been filtered (Values[3] <= 0.0001) or "
            "files may be empty. Cannot create Spots.bin.\n",
            nScans, nFilesFound);
    free(MyData);
    return 1;
  }

  // Shrink allocation to actual size (fix: capture return value)
  struct InpData *tmp = realloc(MyData, nSpots * sizeof(*MyData));
  if (tmp != NULL) {
    MyData = tmp;
  }
  printf("Number of Spots: %zu\n", nSpots);
  fprintf(stderr, "INFO: Total valid spots: %zu\n", nSpots);

  // Now sort the spots, depending on the ring numbers, then omega value, then
  // on the eta-value.
  qsort(MyData, nSpots, sizeof(struct InpData), cmpfunc);
  printf("Data sorted.\n");
  fprintf(stderr, "INFO: Spots sorted by ring/omega/eta\n");

  // Use contiguous 1D arrays instead of row-pointer matrices
  int nObsCols = 18;
  int nIDCols = 3;
  double *ObsSpots = malloc(nSpots * nObsCols * sizeof(double));
  double *IDMat = malloc(nSpots * nIDCols * sizeof(double));
  if (ObsSpots == NULL || IDMat == NULL) {
    fprintf(stderr,
            "ERROR: Could not allocate ObsSpots (%zu bytes) or IDMat (%zu "
            "bytes)\n",
            nSpots * nObsCols * sizeof(double),
            nSpots * nIDCols * sizeof(double));
    printf("Memory error: could not allocate ObsSpots/IDMat.\n");
    return 1;
  }
  for (i = 0; i < (int)nSpots; i++) {
    // Copy 17 values via memcpy for speed
    memcpy(&ObsSpots[i * nObsCols], MyData[i].Values, 17 * sizeof(double));
    ObsSpots[i * nObsCols + 17] = 0; // will be filled by CalcDistanceIdealRing
    IDMat[i * nIDCols + 0] = i + 1;
    IDMat[i * nIDCols + 1] = ObsSpots[i * nObsCols + 4];
    IDMat[i * nIDCols + 2] = ObsSpots[i * nObsCols + 16];
    ObsSpots[i * nObsCols + 4] = i + 1;
  }
  free(MyData);

  for (i = 0; i < MAX_N_RINGS; i++) {
    RingRadii[i] = 0;
  }
  for (i = 0; i < NrOfRings; i++) {
    if (RingNumbers[i] < 0 || RingNumbers[i] >= MAX_N_RINGS) {
      fprintf(stderr, "ERROR: RingNumbers[%d]=%d is out of range (0..%d)\n", i,
              RingNumbers[i], MAX_N_RINGS - 1);
      return 1;
    }
    RingRadii[RingNumbers[i]] = RingRadiiUser[i];
  }
  CalcDistanceIdealRing(ObsSpots, nSpots, nObsCols, RingRadii);

  // Make SpotsMatrix
  double *SpotsMat;
  SpotsMat = malloc(nSpots * 10 * sizeof(*SpotsMat));
  if (SpotsMat == NULL) {
    fprintf(stderr, "ERROR: Could not allocate SpotsMat (%zu bytes)\n",
            nSpots * 10 * sizeof(*SpotsMat));
    printf("Memory error: could not allocate SpotsMat.\n");
    return 1;
  }
  for (i = 0; i < (int)nSpots; i++) {
    for (j = 0; j < 8; j++) {
      SpotsMat[i * 10 + j] = ObsSpots[i * nObsCols + j];
    }
    SpotsMat[i * 10 + 8] = ObsSpots[i * nObsCols + 17];
    SpotsMat[i * 10 + 9] = ObsSpots[i * nObsCols + 16];
  }
  // Make ExtraInfoSpotMatrix
  double *ExtraMat;
  ExtraMat = malloc(nSpots * 16 * sizeof(*ExtraMat));
  if (ExtraMat == NULL) {
    fprintf(stderr, "ERROR: Could not allocate ExtraMat (%zu bytes)\n",
            nSpots * 16 * sizeof(*ExtraMat));
    printf("Memory error: could not allocate ExtraMat.\n");
    return 1;
  }
  for (i = 0; i < (int)nSpots; i++) {
    memcpy(&ExtraMat[i * 16], &ObsSpots[i * nObsCols], 16 * sizeof(double));
  }

  // ---- Write output files ----
  char *SpotsFN = "Spots.bin";
  char *ExtraFN = "ExtraInfo.bin";
  char *IDMatFN = "IDsMergedScanning.csv";

  FILE *SpotsFile = fopen(SpotsFN, "wb");
  if (SpotsFile == NULL) {
    fprintf(stderr, "ERROR: Could not open '%s' for writing: %s (errno=%d)\n",
            SpotsFN, strerror(errno), errno);
    return 1;
  }
  size_t nWritten =
      fwrite(SpotsMat, nSpots * 10 * sizeof(*SpotsMat), 1, SpotsFile);
  if (nWritten != 1) {
    fprintf(stderr,
            "ERROR: fwrite to '%s' failed: wrote %zu of 1 blocks "
            "(errno=%d: %s)\n",
            SpotsFN, nWritten, errno, strerror(errno));
    fclose(SpotsFile);
    return 1;
  }
  if (fclose(SpotsFile) != 0) {
    fprintf(stderr, "ERROR: fclose '%s' failed: %s (errno=%d)\n", SpotsFN,
            strerror(errno), errno);
    return 1;
  }
  fprintf(stderr, "INFO: Wrote %s (%zu spots, %zu bytes)\n", SpotsFN, nSpots,
          nSpots * 10 * sizeof(*SpotsMat));

  FILE *ExtraFile = fopen(ExtraFN, "wb");
  if (ExtraFile == NULL) {
    fprintf(stderr, "ERROR: Could not open '%s' for writing: %s (errno=%d)\n",
            ExtraFN, strerror(errno), errno);
    return 1;
  }
  nWritten = fwrite(ExtraMat, nSpots * 16 * sizeof(*ExtraMat), 1, ExtraFile);
  if (nWritten != 1) {
    fprintf(stderr,
            "ERROR: fwrite to '%s' failed: wrote %zu of 1 blocks "
            "(errno=%d: %s)\n",
            ExtraFN, nWritten, errno, strerror(errno));
    fclose(ExtraFile);
    return 1;
  }
  if (fclose(ExtraFile) != 0) {
    fprintf(stderr, "ERROR: fclose '%s' failed: %s (errno=%d)\n", ExtraFN,
            strerror(errno), errno);
    return 1;
  }
  fprintf(stderr, "INFO: Wrote %s (%zu spots, %zu bytes)\n", ExtraFN, nSpots,
          nSpots * 16 * sizeof(*ExtraMat));
  free(ExtraMat);

  FILE *IDMatFile = fopen(IDMatFN, "w");
  if (IDMatFile == NULL) {
    fprintf(stderr, "ERROR: Could not open '%s' for writing: %s (errno=%d)\n",
            IDMatFN, strerror(errno), errno);
    return 1;
  }
  setvbuf(IDMatFile, NULL, _IOFBF, 1 << 20); // 1 MB write buffer
  fprintf(IDMatFile, "NewID,OrigID,ScanNr\n");
  for (i = 0; i < (int)nSpots; i++)
    fprintf(IDMatFile, "%d,%d,%d\n", (int)IDMat[i * nIDCols + 0],
            (int)IDMat[i * nIDCols + 1], (int)IDMat[i * nIDCols + 2]);
  if (fclose(IDMatFile) != 0) {
    fprintf(stderr, "ERROR: fclose '%s' failed: %s (errno=%d)\n", IDMatFN,
            strerror(errno), errno);
    return 1;
  }
  fprintf(stderr, "INFO: Wrote %s (%zu entries)\n", IDMatFN, nSpots);

  free(SpotsMat);
  free(IDMat);
  if (nosaveall == 1) {
    free(ObsSpots);
    fprintf(stderr, "INFO: NoSaveAll=1, skipping Data.bin/nData.bin. Done.\n");
    return 0;
  }
  printf("Files written. Now generating map.\n");
  fprintf(stderr,
          "INFO: Files written. Now generating Data.bin/nData.bin map.\n");

  // data will save id, scanNr for each spot, everything else remains the same.
  // Twice the size now.
  size_t ***data;
  size_t **ndata;
  size_t **maxndata;
  int n_ring_bins;
  int n_eta_bins;
  int n_ome_bins;
  size_t *newarray;
  size_t *oldarray;
  int iEta, iOme, iEta0, iOme0;
  size_t rowno;

  int HighestRingNo = 0;
  for (i = 0; i < MAX_N_RINGS; i++) {
    if (RingRadii[i] != 0)
      HighestRingNo = i;
  }
  n_ring_bins = HighestRingNo;
  n_eta_bins = ceil(360.0 / etabinsize);
  n_ome_bins = ceil(360.0 / omebinsize);
  printf("nRings: %d, nEtas: %d, nOmes: %d\n", n_ring_bins, n_eta_bins,
         n_ome_bins);
  printf("Total bins: %d\n", n_ring_bins * n_eta_bins * n_ome_bins);
  fprintf(stderr,
          "INFO: Map dimensions: nRings=%d, nEtas=%d, nOmes=%d, "
          "totalBins=%d\n",
          n_ring_bins, n_eta_bins, n_ome_bins,
          n_ring_bins * n_eta_bins * n_ome_bins);

  if (n_ring_bins <= 0 || n_eta_bins <= 0 || n_ome_bins <= 0) {
    fprintf(stderr,
            "ERROR: Invalid bin dimensions: nRings=%d, nEtas=%d, "
            "nOmes=%d. Check EtaBinSize, OmeBinSize, and ring radii.\n",
            n_ring_bins, n_eta_bins, n_ome_bins);
    free(ObsSpots);
    return 1;
  }

  char *DataFN = "Data.bin";
  char *nDataFN = "nData.bin";
  FILE *DataFile = fopen(DataFN, "wb");
  if (DataFile == NULL) {
    fprintf(stderr, "ERROR: Could not open '%s' for writing: %s (errno=%d)\n",
            DataFN, strerror(errno), errno);
    free(ObsSpots);
    return 1;
  }
  FILE *nDataFile = fopen(nDataFN, "wb");
  if (nDataFile == NULL) {
    fprintf(stderr, "ERROR: Could not open '%s' for writing: %s (errno=%d)\n",
            nDataFN, strerror(errno), errno);
    fclose(DataFile);
    free(ObsSpots);
    return 1;
  }
  setvbuf(DataFile, NULL, _IOFBF, 1 << 20);  // 1 MB write buffer
  setvbuf(nDataFile, NULL, _IOFBF, 1 << 20); // 1 MB write buffer
  size_t TotNumberOfBins = 0;
  size_t globalCounter = 0;

  // Process Ring by Ring
  for (int iRing = 0; iRing < n_ring_bins; iRing++) {
    // Allocation for ONE ring (2D arrays)
    data = malloc(n_eta_bins * sizeof(size_t **));    // Corrected sizeof
    ndata = malloc(n_eta_bins * sizeof(size_t *));    // Corrected sizeof
    maxndata = malloc(n_eta_bins * sizeof(size_t *)); // Corrected sizeof

    if (!data || !ndata || !maxndata) {
      fprintf(stderr,
              "ERROR: Memory allocation failed for ring %d (n_eta_bins=%d)\n",
              iRing, n_eta_bins);
      printf("Memory error: memory full during allocation for ring %d\n",
             iRing);
      fclose(DataFile);
      fclose(nDataFile);
      free(ObsSpots);
      return 1;
    }

    for (j = 0; j < n_eta_bins; j++) {
      data[j] = malloc(n_ome_bins * sizeof(size_t *));   // Corrected sizeof
      ndata[j] = malloc(n_ome_bins * sizeof(size_t));    // Corrected sizeof
      maxndata[j] = malloc(n_ome_bins * sizeof(size_t)); // Corrected sizeof

      if (!data[j] || !ndata[j] || !maxndata[j]) {
        fprintf(stderr,
                "ERROR: Memory allocation failed for ring %d, eta bin %d "
                "(n_ome_bins=%d)\n",
                iRing, j, n_ome_bins);
        printf("Memory error: memory full during allocation for ring %d, eta "
               "bin %d\n",
               iRing, j);
        fclose(DataFile);
        fclose(nDataFile);
        free(ObsSpots);
        return 1;
      }

      for (k = 0; k < n_ome_bins; k++) {
        data[j][k] = NULL;
        ndata[j][k] = 0;
        maxndata[j][k] = 0;
      }
    }

    // Bin spots for THIS ring
    size_t nSpotsThisRing = 0;
    for (rowno = 0; rowno < nSpots; rowno++) {
      int ringnr = (int)ObsSpots[rowno * nObsCols + 5];
      if ((ringnr - 1) != iRing)
        continue;
      if (ringnr < 0 || ringnr >= MAX_N_RINGS || RingRadii[ringnr] == 0)
        continue;

      int scanno = (int)ObsSpots[rowno * nObsCols + 16];
      double eta = ObsSpots[rowno * nObsCols + 6];
      double omega = ObsSpots[rowno * nObsCols + 2];

      double omemargin =
          omemargin0 + (0.5 * rotationstep / fabs(sin(eta * deg2rad)));
      double omemin = 180 + omega - omemargin;
      double omemax = 180 + omega + omemargin;
      int iOmeMin = floor(omemin / omebinsize);
      int iOmeMax = floor(omemax / omebinsize);
      double etamargin =
          rad2deg * atan(etamargin0 / RingRadii[ringnr]) + 0.5 * rotationstep;
      double etamin = 180 + eta - etamargin;
      double etamax = 180 + eta + etamargin;
      int iEtaMin = floor(etamin / etabinsize);
      int iEtaMax = floor(etamax / etabinsize);

      nSpotsThisRing++;
      for (iEta0 = iEtaMin; iEta0 <= iEtaMax; iEta0++) {
        iEta = iEta0 % n_eta_bins;
        if (iEta < 0)
          iEta = iEta + n_eta_bins;
        for (iOme0 = iOmeMin; iOme0 <= iOmeMax; iOme0++) {
          iOme = iOme0 % n_ome_bins;
          if (iOme < 0)
            iOme = iOme + n_ome_bins;

          size_t iSpot = ndata[iEta][iOme];
          size_t maxnspot = maxndata[iEta][iOme];
          if (iSpot >= maxnspot) {
            // Geometric growth: reduces realloc calls dramatically
            maxnspot = (maxnspot == 0) ? 8 : maxnspot * 2;
            oldarray = data[iEta][iOme];
            newarray = realloc(oldarray, maxnspot * 2 * sizeof(*newarray));
            if (newarray == NULL) {
              fprintf(stderr,
                      "ERROR: realloc failed for ring %d, eta=%d, ome=%d "
                      "(requested %zu bytes): %s\n",
                      iRing, iEta, iOme, maxnspot * 2 * sizeof(*newarray),
                      strerror(errno));
              printf("Memory error: memory full?\n");
              fclose(DataFile);
              fclose(nDataFile);
              free(ObsSpots);
              return 1;
            }
            data[iEta][iOme] = newarray;
            maxndata[iEta][iOme] = maxnspot;
          }
          data[iEta][iOme][iSpot * 2 + 0] = rowno;
          data[iEta][iOme][iSpot * 2 + 1] = scanno;
          (ndata[iEta][iOme])++;
          TotNumberOfBins++;
        }
      }
    }

    fprintf(stderr, "INFO: Ring %d/%d: %zu spots binned\n", iRing + 1,
            n_ring_bins, nSpotsThisRing);

    // Write to File immediately
    int write_error = 0;
    for (j = 0; j < n_eta_bins; j++) {
      for (k = 0; k < n_ome_bins; k++) {
        size_t localNDataVal = ndata[j][k];
        size_t nDataInfo[2];
        nDataInfo[0] = localNDataVal;
        nDataInfo[1] = globalCounter;
        size_t w = fwrite(nDataInfo, sizeof(size_t), 2, nDataFile);
        if (w != 2 && !write_error) {
          fprintf(stderr,
                  "ERROR: fwrite to nData.bin failed at ring=%d, "
                  "eta=%d, ome=%d: wrote %zu/2 (errno=%d: %s)\n",
                  iRing, j, k, w, errno, strerror(errno));
          write_error = 1;
        }

        if (localNDataVal > 0) {
          w = fwrite(data[j][k], sizeof(size_t), localNDataVal * 2, DataFile);
          if (w != localNDataVal * 2 && !write_error) {
            fprintf(stderr,
                    "ERROR: fwrite to Data.bin failed at ring=%d, "
                    "eta=%d, ome=%d: wrote %zu/%zu (errno=%d: %s)\n",
                    iRing, j, k, w, localNDataVal * 2, errno, strerror(errno));
            write_error = 1;
          }
          globalCounter += localNDataVal;
          free(data[j][k]); // Free the spot array
        }
      }
      free(data[j]);
      free(ndata[j]);
      free(maxndata[j]);
    }
    free(data);
    free(ndata);
    free(maxndata);

    if (write_error) {
      fprintf(stderr,
              "ERROR: Write errors occurred during ring %d. Aborting.\n",
              iRing);
      fclose(DataFile);
      fclose(nDataFile);
      free(ObsSpots);
      return 1;
    }
  }

  if (fclose(DataFile) != 0) {
    fprintf(stderr, "ERROR: fclose 'Data.bin' failed: %s (errno=%d)\n",
            strerror(errno), errno);
    return 1;
  }
  if (fclose(nDataFile) != 0) {
    fprintf(stderr, "ERROR: fclose 'nData.bin' failed: %s (errno=%d)\n",
            strerror(errno), errno);
    return 1;
  }
  free(ObsSpots);

  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("nData: %zu, Data: %zu\n",
         (size_t)n_ring_bins * n_eta_bins * n_ome_bins * 2,
         TotNumberOfBins * 2);
  printf("Total Time elapsed: %f s.\n", diftotal);
  fprintf(stderr,
          "INFO: SaveBinDataScanning completed successfully. "
          "nSpots=%zu, nData=%zu, Data=%zu, Time=%.2fs\n",
          nSpots, (size_t)n_ring_bins * n_eta_bins * n_ome_bins * 2,
          TotNumberOfBins * 2, diftotal);

  // Verify output files exist and have size
  struct stat st;
  const char *outFiles[] = {"Spots.bin", "ExtraInfo.bin",
                            "IDsMergedScanning.csv", "Data.bin", "nData.bin"};
  for (int f = 0; f < 5; f++) {
    if (stat(outFiles[f], &st) == 0) {
      fprintf(stderr, "INFO: Output file '%s': %lld bytes\n", outFiles[f],
              (long long)st.st_size);
      if (st.st_size == 0) {
        fprintf(stderr, "WARNING: Output file '%s' is 0 bytes!\n", outFiles[f]);
      }
    } else {
      fprintf(stderr, "ERROR: Output file '%s' not found after writing!\n",
              outFiles[f]);
    }
  }

  return 0;
}
