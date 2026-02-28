//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// SaveBinData.c
//
// Created by Hemant Sharma on 2014/11/07
//
//

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define N_COL_OBSSPOTS 9 // This is one less number of columns
#define MAX_N_RINGS                                                            \
  500 // max nr of rings that can be stored (applies to the arrays ringttheta,
      // ringhkl, etc)

// Count lines in a CSV file (excluding header)
static int countCSVLines(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (f == NULL)
    return -1;
  setvbuf(f, NULL, _IOFBF, 1 << 20);
  char buf[4096];
  int count = 0;
  // Skip header
  if (fgets(buf, 4096, f) == NULL) {
    fclose(f);
    return 0;
  }
  while (fgets(buf, 4096, f) != NULL) {
    count++;
  }
  fclose(f);
  return count;
}

void CalcDistanceIdealRing(double *ObsSpotsLab, int nspots, int nCols,
                           double RingRadii[]) {
  int i;
  for (i = 0; i < nspots; ++i) {
    double y = ObsSpotsLab[i * nCols + 0];
    double z = ObsSpotsLab[i * nCols + 1];
    double rad = sqrt(y * y + z * z);
    int ringno = (int)ObsSpotsLab[i * nCols + 5];
    ObsSpotsLab[i * nCols + 8] = rad - RingRadii[ringno];
  }
}

int main(int arc, char *argv[]) {
  clock_t start, end;
  double diftotal;
  start = clock();

  // Count lines first to avoid massive upfront allocation
  int nSpots = countCSVLines("InputAll.csv");
  if (nSpots < 0) {
    printf("Could not open InputAll.csv. Exiting.\n");
    return 1;
  }
  if (nSpots == 0) {
    printf("No spots found in InputAll.csv. Exiting.\n");
    return 1;
  }
  printf("Counted %d spots in InputAll.csv.\n", nSpots);

  // Allocate contiguous 1D arrays instead of row-pointer matrices
  double *ObsSpots = malloc((size_t)nSpots * N_COL_OBSSPOTS * sizeof(double));
  if (ObsSpots == NULL) {
    printf("Memory error: could not allocate ObsSpots.\n");
    return 1;
  }

  char *ObsSpotsFN = "InputAll.csv";
  FILE *ObsSpotsFile = fopen(ObsSpotsFN, "r");
  setvbuf(ObsSpotsFile, NULL, _IOFBF, 1 << 20); // 1 MB read buffer
  char aline[4096];
  char *rc = fgets(aline, 4096, ObsSpotsFile); // skip header
  int spotIdx = 0;
  while (fgets(aline, 4096, ObsSpotsFile) != NULL) {
    sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf",
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 0],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 1],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 2],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 3],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 4],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 5],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 6],
           &ObsSpots[spotIdx * N_COL_OBSSPOTS + 7]);
    spotIdx++;
  }
  fclose(ObsSpotsFile);

  char *AllSpotsFN = "InputAllExtraInfoFittingAll.csv";
  FILE *AllSpotsFile = fopen(AllSpotsFN, "r");
  if (AllSpotsFile == NULL) {
    printf("Could not open %s. Exiting.\n", AllSpotsFN);
    return 1;
  }
  setvbuf(AllSpotsFile, NULL, _IOFBF, 1 << 20); // 1 MB read buffer
  char *rc2 = fgets(aline, 4096, AllSpotsFile); // skip header
  int countr = 0;
  double *AllSpots = malloc((size_t)nSpots * 16 * sizeof(double));
  if (AllSpots == NULL) {
    printf("Memory error: could not allocate AllSpots.\n");
    return 1;
  }
  while (fgets(aline, 4096, AllSpotsFile) != NULL) {
    double dummy0, dummy1;
    sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &AllSpots[countr * 16 + 0], &AllSpots[countr * 16 + 1],
           &AllSpots[countr * 16 + 2], &AllSpots[countr * 16 + 3],
           &AllSpots[countr * 16 + 4], &AllSpots[countr * 16 + 5],
           &AllSpots[countr * 16 + 6], &AllSpots[countr * 16 + 7],
           &AllSpots[countr * 16 + 8], &AllSpots[countr * 16 + 9],
           &AllSpots[countr * 16 + 10], &AllSpots[countr * 16 + 11],
           &AllSpots[countr * 16 + 12], &AllSpots[countr * 16 + 13],
           &dummy0, &dummy1,
           &AllSpots[countr * 16 + 14], &AllSpots[countr * 16 + 15]);
    countr++;
  }
  fclose(AllSpotsFile);
  if (nSpots != countr) {
    printf("AllSpots from InputAll and InputAllExtraInfo files don't match. Do "
           "something. Exiting\n");
    return 1;
  }
  char *ParamFN = "paramstest.txt", dummy[1024], *str;
  int LowNr;
  FILE *fileParam;
  fileParam = fopen(ParamFN, "r");
  if (fileParam == NULL) {
    printf("Could not open %s. Exiting.\n", ParamFN);
    return 1;
  }
  int NrOfRings = 0, NoRingNumbers = 0, RingNumbers[MAX_N_RINGS];
  double omemargin0, etamargin0, rotationstep, RingRadii[MAX_N_RINGS],
      RingRadiiUser[MAX_N_RINGS], etabinsize, omebinsize;
  int nosaveall = 0;
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
      continue;
    }
    str = "MarginEta ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &etamargin0);
      continue;
    }
    str = "EtaBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &etabinsize);
      continue;
    }
    str = "StepsizeOrient ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &rotationstep);
      continue;
    }
    str = "StepSizeOrient ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &rotationstep);
      continue;
    }
    str = "OmeBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &omebinsize);
      continue;
    }
    str = "RingRadii ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &RingRadiiUser[NrOfRings]);
      NrOfRings++;
      continue;
    }
    str = "RingNumbers ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingNumbers[NoRingNumbers]);
      NoRingNumbers++;
      continue;
    }
  }
  fclose(fileParam);
  printf("Read Parameters:\n\tNoSaveAll: %d\n\tMarginOme: %lf\n\tMarginEta: "
         "%lf\n\tEtaBinSize: %lf\n\tStepSizeOrient: %lf\n\tOmeBinSize: "
         "%lf\n\tNrOfRings: %d\n\tNoRingNumbers: %d\n",
         nosaveall, omemargin0, etamargin0, etabinsize, rotationstep,
         omebinsize, NrOfRings, NoRingNumbers);
  for (int iter = 0; iter < NrOfRings; iter++)
    printf("\tRingRadiiUser[%d]: %lf\n", iter, RingRadiiUser[iter]);
  for (int iter = 0; iter < NoRingNumbers; iter++)
    printf("\tRingNumbers[%d]: %d\n", iter, RingNumbers[iter]);

  int i, j, k, t;

  for (i = 0; i < MAX_N_RINGS; i++) {
    RingRadii[i] = 0;
  }
  for (i = 0; i < NrOfRings; i++) {
    RingRadii[RingNumbers[i]] = RingRadiiUser[i];
  }
  CalcDistanceIdealRing(ObsSpots, nSpots, N_COL_OBSSPOTS, RingRadii);
  // Make SpotsMatrix
  double *SpotsMat;
  SpotsMat = malloc(nSpots * 9 * sizeof(*SpotsMat));
  if (SpotsMat == NULL) {
    printf("Memory error: could not allocate SpotsMat.\n");
    return 1;
  }
  for (i = 0; i < nSpots; i++) {
    memcpy(&SpotsMat[i * 9], &ObsSpots[i * N_COL_OBSSPOTS], 9 * sizeof(double));
  }
  // Make ExtraInfoSpotMatrix
  double *ExtraMat;
  ExtraMat = malloc(nSpots * 16 * sizeof(*ExtraMat));
  if (ExtraMat == NULL) {
    printf("Memory error: could not allocate ExtraMat.\n");
    return 1;
  }
  for (i = 0; i < nSpots; i++) {
    memcpy(&ExtraMat[i * 16], &AllSpots[i * 16], 16 * sizeof(double));
  }
  char *SpotsFN = "Spots.bin";
  char *ExtraFN = "ExtraInfo.bin";
  FILE *SpotsFile = fopen(SpotsFN, "wb");
  fwrite(SpotsMat, nSpots * 9 * sizeof(*SpotsMat), 1, SpotsFile);
  fclose(SpotsFile);
  FILE *ExtraFile = fopen(ExtraFN, "wb");
  fwrite(ExtraMat, nSpots * 16 * sizeof(*ExtraMat), 1, ExtraFile);
  fclose(ExtraFile);
  free(ExtraMat);
  free(AllSpots);
  if (nosaveall == 1) {
    free(ObsSpots);
    free(SpotsMat);
    return 0;
  }

  // Only continue if wanted to save all.
  int n_ring_bins;
  int n_eta_bins;
  int n_ome_bins;
  int iEta, iOme, iEta0, iOme0;
  int rowno;
  double EtaBinSize = etabinsize;
  double OmeBinSize = omebinsize;
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

  // Flat arrays for per-ring processing (same ring-by-ring pattern as Scanning)
  int **data_ring;    // [eta*ome] -> dynamic int array
  int *ndata_ring;    // [eta*ome] -> count
  int *maxndata_ring; // [eta*ome] -> capacity
  long long int ring_eo = (long long int)n_eta_bins * n_ome_bins;

  char *DataFN = "Data.bin";
  char *nDataFN = "nData.bin";
  FILE *DataFile = fopen(DataFN, "wb");
  FILE *nDataFile = fopen(nDataFN, "wb");
  setvbuf(DataFile, NULL, _IOFBF, 1 << 20);
  setvbuf(nDataFile, NULL, _IOFBF, 1 << 20);
  long long int TotNumberOfBins = 0;
  int localCounter = 0;

  for (i = 0; i < n_ring_bins; i++) {
    // Allocate flat arrays for this ring
    data_ring = malloc(ring_eo * sizeof(int *));
    ndata_ring = calloc(ring_eo, sizeof(int));
    maxndata_ring = calloc(ring_eo, sizeof(int));
    if (!data_ring || !ndata_ring || !maxndata_ring) {
      printf("Memory error: memory full during ring %d allocation.\n", i);
      return 1;
    }
    for (long long int idx = 0; idx < ring_eo; idx++) {
      data_ring[idx] = NULL;
    }

    // Bin spots for this ring
    for (rowno = 0; rowno < nSpots; rowno++) {
      int ringnr = (int)ObsSpots[rowno * N_COL_OBSSPOTS + 5];
      double eta = ObsSpots[rowno * N_COL_OBSSPOTS + 6];
      double omega = ObsSpots[rowno * N_COL_OBSSPOTS + 2];
      int iRing = ringnr - 1;
      if (iRing != i)
        continue;
      if (RingRadii[ringnr] == 0)
        continue;
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
      for (iEta0 = iEtaMin; iEta0 <= iEtaMax; iEta0++) {
        iEta = iEta0 % n_eta_bins;
        if (iEta < 0)
          iEta = iEta + n_eta_bins;
        for (iOme0 = iOmeMin; iOme0 <= iOmeMax; iOme0++) {
          iOme = iOme0 % n_ome_bins;
          if (iOme < 0)
            iOme = iOme + n_ome_bins;
          long long int binIdx = (long long int)iEta * n_ome_bins + iOme;
          int iSpot = ndata_ring[binIdx];
          int maxnspot = maxndata_ring[binIdx];
          if (iSpot >= maxnspot) {
            // Geometric growth
            maxnspot = (maxnspot == 0) ? 8 : maxnspot * 2;
            int *newarray =
                realloc(data_ring[binIdx], maxnspot * sizeof(*newarray));
            if (newarray == NULL) {
              printf("Memory error: memory full?\n");
              return 1;
            }
            data_ring[binIdx] = newarray;
            maxndata_ring[binIdx] = maxnspot;
          }
          data_ring[binIdx][iSpot] = rowno;
          ndata_ring[binIdx]++;
          TotNumberOfBins++;
        }
      }
    }

    // Write ring data and free immediately
    for (j = 0; j < n_eta_bins; j++) {
      for (k = 0; k < n_ome_bins; k++) {
        long long int binIdx = (long long int)j * n_ome_bins + k;
        int localNDataVal = ndata_ring[binIdx];
        int nDataStore2[2];
        nDataStore2[0] = localNDataVal;
        nDataStore2[1] = localCounter;
        fwrite(nDataStore2, sizeof(int), 2, nDataFile);
        if (localNDataVal > 0) {
          fwrite(data_ring[binIdx], sizeof(int), localNDataVal, DataFile);
          localCounter += localNDataVal;
          free(data_ring[binIdx]);
        }
      }
    }
    free(data_ring);
    free(ndata_ring);
    free(maxndata_ring);
  }

  fclose(DataFile);
  fclose(nDataFile);
  free(ObsSpots);
  free(SpotsMat);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total Time elapsed: %f s.\n", diftotal);
  return 0;
}
