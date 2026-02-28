#include <ctype.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define MAX_N_SPOTS 100000000

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\tMerging Scans in scanning in PF-HEDM.\n\n");
  int returncode;
  if (argc != 6) {
    printf("Supply %s nScans, nMerges, pxTol, omeTol, nCPUs as arguments.\n",
           argv[0]);
    return 1;
  }
  int nScans = atoi(argv[1]);
  int nMerges = atoi(argv[2]);
  double tolPx = atof(argv[3]);
  double tolOme = atof(argv[4]);
  int numProcs = atoi(argv[5]);
  int nFinScans = (int)floor((double)nScans / (double)nMerges);
  double *positions, *positionsNew;
  positions = calloc(nScans, sizeof(*positions));
  positionsNew = calloc(nFinScans, sizeof(*positionsNew));
  FILE *posF = fopen("original_positions.csv", "r");
  int iter;
  char aline[2048];
  for (iter = 0; iter < nScans; iter++) {
    fgets(aline, 2048, posF);
    sscanf(aline, "%lf", &positions[iter]);
  }
  fclose(posF);

  int finScanNr;
#pragma omp parallel for num_threads(numProcs) private(finScanNr)              \
    schedule(dynamic)
  for (finScanNr = 0; finScanNr < nFinScans; finScanNr++) {
    int startScanNr = finScanNr * nMerges;
    double thisPosition = positions[startScanNr];
    double *thisSpots, *allSpots;
    thisSpots = calloc(MAX_N_SPOTS * 16, sizeof(*thisSpots));
    allSpots = calloc(MAX_N_SPOTS * 16, sizeof(*allSpots));
    // Read the first fileNr
    char thisFN[2048], thisLine[2048], headThis[2048];
    sprintf(thisFN, "original_InputAllExtraInfoFittingAll%d.csv", startScanNr);
    FILE *thisF;
    thisF = fopen(thisFN, "r");
    fgets(thisLine, 2048, thisF);
    sprintf(headThis, "%s", thisLine);
    int *lastScansSpots, *thisScansSpots;
    lastScansSpots = calloc(MAX_N_SPOTS, sizeof(*lastScansSpots));
    thisScansSpots = calloc(MAX_N_SPOTS, sizeof(*thisScansSpots));
    int nSpotsLastScan;
    size_t nAll = 0;
    while (fgets(thisLine, 2048, thisF) != NULL) {
      double dummy0, dummy1;
      sscanf(thisLine,
             "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
             "%lf %lf",
             &allSpots[nAll * 16 + 0], &allSpots[nAll * 16 + 1],
             &allSpots[nAll * 16 + 2], &allSpots[nAll * 16 + 3],
             &allSpots[nAll * 16 + 4], &allSpots[nAll * 16 + 5],
             &allSpots[nAll * 16 + 6], &allSpots[nAll * 16 + 7],
             &allSpots[nAll * 16 + 8], &allSpots[nAll * 16 + 9],
             &allSpots[nAll * 16 + 10], &allSpots[nAll * 16 + 11],
             &allSpots[nAll * 16 + 12], &allSpots[nAll * 16 + 13], &dummy0,
             &dummy1, &allSpots[nAll * 16 + 14], &allSpots[nAll * 16 + 15]);
      if (allSpots[nAll * 16 + 3] < 0.01)
        continue;
      lastScansSpots[nAll] = nAll;
      nAll++;
    }
    nSpotsLastScan = nAll;
    fclose(thisF);
    int scanNr, thisScanNr;
    int i, j, k, l, found;
    double origWeight, newWeight;
    for (scanNr = 1; scanNr < nMerges; scanNr++) {
      printf("ScanNr: %d, nAll: %zu\n", scanNr, nAll);
      thisScanNr = startScanNr + scanNr;
      thisPosition += positions[thisScanNr];
      int nThis = 0;
      sprintf(thisFN, "original_InputAllExtraInfoFittingAll%d.csv", thisScanNr);
      thisF = fopen(thisFN, "r");
      fgets(thisLine, 2048, thisF);
      while (fgets(thisLine, 2048, thisF) != NULL) {
        double dummy0, dummy1;
        sscanf(thisLine,
               "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
               "%lf %lf %lf",
               &thisSpots[nThis * 16 + 0], &thisSpots[nThis * 16 + 1],
               &thisSpots[nThis * 16 + 2], &thisSpots[nThis * 16 + 3],
               &thisSpots[nThis * 16 + 4], &thisSpots[nThis * 16 + 5],
               &thisSpots[nThis * 16 + 6], &thisSpots[nThis * 16 + 7],
               &thisSpots[nThis * 16 + 8], &thisSpots[nThis * 16 + 9],
               &thisSpots[nThis * 16 + 10], &thisSpots[nThis * 16 + 11],
               &thisSpots[nThis * 16 + 12], &thisSpots[nThis * 16 + 13],
               &dummy0, &dummy1, &thisSpots[nThis * 16 + 14],
               &thisSpots[nThis * 16 + 15]);
        if (thisSpots[nThis * 16 + 3] < 0.01)
          continue;
        nThis++;
      }
      fclose(thisF);
      // Go through each spot in both files, if found a match add weighted
      // values, if not, add to the original array
      for (i = 0; i < nThis; i++) {
        found = 0;
        for (l = 0; l < nSpotsLastScan; l++) {
          j = lastScansSpots[l];
          if (fabs(thisSpots[i * 16 + 5] - allSpots[j * 16 + 5]) < 0.01) {
            if (fabs(thisSpots[i * 16 + 0] - allSpots[j * 16 + 0]) < tolPx) {
              if (fabs(thisSpots[i * 16 + 1] - allSpots[j * 16 + 1]) < tolPx) {
                if (fabs(thisSpots[i * 16 + 2] - allSpots[j * 16 + 2]) <
                    tolOme) {
                  found = 1;
                  origWeight = allSpots[j * 16 + 3];
                  newWeight = thisSpots[i * 16 + 3];
                  thisScansSpots[i] = j;
                  for (k = 0; k < 16; k++) {
                    allSpots[j * 16 + k] = (allSpots[j * 16 + k] * origWeight +
                                            thisSpots[i * 16 + k] * newWeight) /
                                           (origWeight + newWeight);
                  }
                }
              }
            }
          }
        }
        if (found == 0) {
          thisScansSpots[i] = nAll;
          for (j = 0; j < 16; j++)
            allSpots[nAll * 16 + j] = thisSpots[i * 16 + j];
          nAll++;
        }
      }
      for (i = 0; i < nSpotsLastScan; i++)
        lastScansSpots[i] = thisScansSpots[i];
      nSpotsLastScan = nThis;
    }
    for (i = 0; i < nAll; i++)
      allSpots[i * 16 + 4] = i + 1;
    sprintf(thisFN, "InputAllExtraInfoFittingAll%d.csv", finScanNr);
    thisF = fopen(thisFN, "w");
    fprintf(thisF, "%s", headThis);
    for (i = 0; i < nAll; i++) {
      for (j = 0; j < 16; j++) {
        fprintf(thisF, "%lf ", allSpots[i * 16 + j]);
      }
      fprintf(thisF, "\n");
    }
    thisPosition /= nMerges;
    positionsNew[finScanNr] = thisPosition;
  }
  posF = fopen("positions.csv", "w");
  for (iter = 0; iter < nFinScans; iter++) {
    fprintf(posF, "%lf\n", positionsNew[iter]);
  }
  fclose(posF);
}
