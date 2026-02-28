//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// SaveBinData.c
//
// Created by Hemant Sharma on 2014/11/07
// WE NEED TO UPDATE PARAMSTEST.TXT with RingToIndex
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

#define N_COL_OBSSPOTS 10     // This is one less number of columns
#define MAX_N_SPOTS 100000000 // max nr of observed spots that can be stored
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
  clock_t start, end;
  if (argc != 2) {
    printf("Usage: SaveBinDataScanning nScans\n");
    return 1;
  }
  int nScans = atoi(argv[1]);
  double diftotal;
  start = clock();
  char *ParamFN = "paramstest.txt", dummy[1024], *str;
  char aline[4096];
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

  int i, j, k, t;

  int scanNr;
  size_t nSpots = 0;
  int countr = 0;
  char AllSpotsFN[4096];
  FILE *AllSpotsFile;
  char *rc;
  struct InpData *MyData;
  MyData = malloc(MAX_N_SPOTS * sizeof(*MyData));
  if (MyData == NULL) {
    printf("Memory error: could not allocate MyData.\n");
    return 1;
  }
  for (scanNr = 0; scanNr < nScans; scanNr++) {
    sprintf(AllSpotsFN, "InputAllExtraInfoFittingAll%d.csv", scanNr);
    AllSpotsFile = fopen(AllSpotsFN, "r");
    if (AllSpotsFile == NULL) {
      printf("Could not open %s. Skipping.\n", AllSpotsFN);
      continue;
    }
    setvbuf(AllSpotsFile, NULL, _IOFBF, 1 << 20); // 1 MB read buffer
    rc = fgets(aline, 4096, AllSpotsFile);
    while (fgets(aline, 4096, AllSpotsFile) != NULL) {
      double dummy0, dummy1;
      sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
             &MyData[nSpots].Values[0], &MyData[nSpots].Values[1],
             &MyData[nSpots].Values[2], &MyData[nSpots].Values[3],
             &MyData[nSpots].Values[4], &MyData[nSpots].Values[5],
             &MyData[nSpots].Values[6], &MyData[nSpots].Values[7],
             &MyData[nSpots].Values[8], &MyData[nSpots].Values[9],
             &MyData[nSpots].Values[10], &MyData[nSpots].Values[11],
             &MyData[nSpots].Values[12], &MyData[nSpots].Values[13],
             &dummy0, &dummy1,
             &MyData[nSpots].Values[14], &MyData[nSpots].Values[15]);
      MyData[nSpots].Values[16] = scanNr;
      if (fabs(MyData[nSpots].Values[3]) > 0.0001)
        nSpots++;
    }
    fclose(AllSpotsFile);
  }
  // Shrink allocation to actual size (fix: capture return value)
  struct InpData *tmp = realloc(MyData, nSpots * sizeof(*MyData));
  if (tmp != NULL) {
    MyData = tmp;
  }
  printf("Number of Spots: %zu\n", nSpots);

  // Now sort the spots, depending on the ring numbers, then omega value, then
  // on the eta-value.
  qsort(MyData, nSpots, sizeof(struct InpData), cmpfunc);
  printf("Data sorted.\n");

  // Use contiguous 1D arrays instead of row-pointer matrices
  int nObsCols = 18;
  int nIDCols = 3;
  double *ObsSpots = malloc(nSpots * nObsCols * sizeof(double));
  double *IDMat = malloc(nSpots * nIDCols * sizeof(double));
  if (ObsSpots == NULL || IDMat == NULL) {
    printf("Memory error: could not allocate ObsSpots/IDMat.\n");
    return 1;
  }
  for (i = 0; i < (int)nSpots; i++) {
    // Copy 15 values via memcpy for speed
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
    RingRadii[RingNumbers[i]] = RingRadiiUser[i];
  }
  CalcDistanceIdealRing(ObsSpots, nSpots, nObsCols, RingRadii);

  // Make SpotsMatrix
  double *SpotsMat;
  SpotsMat = malloc(nSpots * 10 * sizeof(*SpotsMat));
  if (SpotsMat == NULL) {
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
    printf("Memory error: could not allocate ExtraMat.\n");
    return 1;
  }
  for (i = 0; i < (int)nSpots; i++) {
    memcpy(&ExtraMat[i * 16], &ObsSpots[i * nObsCols], 16 * sizeof(double));
  }
  char *SpotsFN = "Spots.bin";
  char *ExtraFN = "ExtraInfo.bin";
  char *IDMatFN = "IDsMergedScanning.csv";
  FILE *SpotsFile = fopen(SpotsFN, "wb");
  fwrite(SpotsMat, nSpots * 10 * sizeof(*SpotsMat), 1, SpotsFile);
  fclose(SpotsFile);
  FILE *ExtraFile = fopen(ExtraFN, "wb");
  fwrite(ExtraMat, nSpots * 16 * sizeof(*ExtraMat), 1, ExtraFile);
  fclose(ExtraFile);
  free(ExtraMat);
  FILE *IDMatFile = fopen(IDMatFN, "w");
  setvbuf(IDMatFile, NULL, _IOFBF, 1 << 20); // 1 MB write buffer
  fprintf(IDMatFile, "NewID,OrigID,ScanNr\n");
  for (i = 0; i < (int)nSpots; i++)
    fprintf(IDMatFile, "%d,%d,%d\n", (int)IDMat[i * nIDCols + 0],
            (int)IDMat[i * nIDCols + 1], (int)IDMat[i * nIDCols + 2]);
  fclose(IDMatFile);
  free(SpotsMat);
  free(IDMat);
  if (nosaveall == 1) {
    free(ObsSpots);
    return 0;
  }
  printf("Files written. Now generating map.\n");

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

  char *DataFN = "Data.bin";
  char *nDataFN = "nData.bin";
  FILE *DataFile = fopen(DataFN, "wb");
  FILE *nDataFile = fopen(nDataFN, "wb");
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
      printf("Memory error: memory full during allocation for ring %d\n",
             iRing);
      return 1;
    }

    for (j = 0; j < n_eta_bins; j++) {
      data[j] = malloc(n_ome_bins * sizeof(size_t *));   // Corrected sizeof
      ndata[j] = malloc(n_ome_bins * sizeof(size_t));    // Corrected sizeof
      maxndata[j] = malloc(n_ome_bins * sizeof(size_t)); // Corrected sizeof

      if (!data[j] || !ndata[j] || !maxndata[j]) {
        printf("Memory error: memory full during allocation for ring %d, eta "
               "bin %d\n",
               iRing, j);
        return 1;
      }

      for (k = 0; k < n_ome_bins; k++) {
        data[j][k] = NULL;
        ndata[j][k] = 0;
        maxndata[j][k] = 0;
      }
    }

    // Bin spots for THIS ring
    for (rowno = 0; rowno < nSpots; rowno++) {
      int ringnr = (int)ObsSpots[rowno * nObsCols + 5];
      if ((ringnr - 1) != iRing)
        continue;
      if (RingRadii[ringnr] == 0)
        continue;

      int scanno = (int)ObsSpots[rowno * nObsCols + 14];
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
              printf("Memory error: memory full?\n");
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

    // Write to File immediately
    for (j = 0; j < n_eta_bins; j++) {
      for (k = 0; k < n_ome_bins; k++) {
        size_t localNDataVal = ndata[j][k];
        size_t nDataInfo[2];
        nDataInfo[0] = localNDataVal;
        nDataInfo[1] = globalCounter;
        fwrite(nDataInfo, sizeof(size_t), 2, nDataFile);

        if (localNDataVal > 0) {
          fwrite(data[j][k], sizeof(size_t), localNDataVal * 2, DataFile);
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
  }

  fclose(DataFile);
  fclose(nDataFile);
  free(ObsSpots);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("nData: %zu, Data: %zu\n",
         (size_t)n_ring_bins * n_eta_bins * n_ome_bins * 2,
         TotNumberOfBins * 2);
  printf("Total Time elapsed: %f s.\n", diftotal);
  return 0;
}
