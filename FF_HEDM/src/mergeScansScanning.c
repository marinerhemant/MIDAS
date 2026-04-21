#include "midas_version.h"
#include <ctype.h>
#include <fcntl.h>
#include <libgen.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* Comparator for argsort of positions (ascending).
 * sort_perm[i] will hold the file index of the i-th spatially ordered scan. */
static const double *_merge_positions = NULL;
static int cmp_merge_argsort(const void *a, const void *b) {
  double da = _merge_positions[*(const int *)a];
  double db = _merge_positions[*(const int *)b];
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
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

  /* Compute spatial sort permutation: sort_perm[spatialIdx] = fileIdx.
   * This ensures we merge spatially adjacent scans regardless of file order. */
  int *sort_perm = malloc(nScans * sizeof(*sort_perm));
  for (iter = 0; iter < nScans; iter++)
    sort_perm[iter] = iter;
  _merge_positions = positions;
  qsort(sort_perm, nScans, sizeof(int), cmp_merge_argsort);
  _merge_positions = NULL;

  printf("Spatial sort permutation (first 10):\n");
  for (iter = 0; iter < nScans && iter < 10; iter++)
    printf("  spatial %d -> file %d (pos %.4f)\n", iter, sort_perm[iter],
           positions[sort_perm[iter]]);
  if (nScans > 10)
    printf("  ...\n");

  int finScanNr;
#pragma omp parallel for num_threads(numProcs) private(finScanNr)              \
    schedule(dynamic)
  for (finScanNr = 0; finScanNr < nFinScans; finScanNr++) {
    /* The first file in this merge group is the spatially-ordered scan. */
    int spatialStart = finScanNr * nMerges;
    int firstFileIdx = sort_perm[spatialStart];
    double thisPosition = positions[firstFileIdx];
    char thisFN[2048], thisLine[2048], headThis[2048];
    FILE *thisF;
    /* Pre-scan every input file in this merge group to size buffers exactly.
       Old code allocated 100M*16 doubles (~12.8 GB) per thread as a safety
       cap; that wastes VM and still overflows on extreme inputs. */
    size_t totalLines = 0;
    size_t maxLines = 0;
    int ms;
    for (ms = 0; ms < nMerges; ms++) {
      int fileIdx = sort_perm[spatialStart + ms];
      sprintf(thisFN, "original_InputAllExtraInfoFittingAll%d.csv", fileIdx);
      FILE *cf = fopen(thisFN, "r");
      if (cf == NULL) {
        printf("Could not open %s. Exiting.\n", thisFN);
        exit(EXIT_FAILURE);
      }
      fgets(thisLine, 2048, cf); /* skip header */
      size_t nRows = 0;
      while (fgets(thisLine, 2048, cf) != NULL)
        nRows++;
      fclose(cf);
      totalLines += nRows;
      if (nRows > maxLines)
        maxLines = nRows;
    }
    double *thisSpots, *allSpots;
    thisSpots = calloc(maxLines * 16, sizeof(*thisSpots));
    allSpots = calloc(totalLines * 16, sizeof(*allSpots));
    int *lastScansSpots, *thisScansSpots;
    lastScansSpots = calloc(maxLines, sizeof(*lastScansSpots));
    thisScansSpots = calloc(maxLines, sizeof(*thisScansSpots));
    if (thisSpots == NULL || allSpots == NULL || lastScansSpots == NULL ||
        thisScansSpots == NULL) {
      printf("Allocation failed in mergeScansScanning (finScanNr=%d). Exiting.\n",
             finScanNr);
      exit(EXIT_FAILURE);
    }
    // Read the first fileNr
    sprintf(thisFN, "original_InputAllExtraInfoFittingAll%d.csv", firstFileIdx);
    thisF = fopen(thisFN, "r");
    fgets(thisLine, 2048, thisF);
    sprintf(headThis, "%s", thisLine);
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
      thisScanNr = sort_perm[spatialStart + scanNr];
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
    fclose(thisF);
    thisPosition /= nMerges;
    positionsNew[finScanNr] = thisPosition;
    free(thisSpots);
    free(allSpots);
    free(lastScansSpots);
    free(thisScansSpots);
  }
  /* Write merged positions in spatial order (ascending). */
  posF = fopen("positions.csv", "w");
  for (iter = 0; iter < nFinScans; iter++) {
    fprintf(posF, "%lf\n", positionsNew[iter]);
  }
  fclose(posF);
  free(sort_perm);
}
