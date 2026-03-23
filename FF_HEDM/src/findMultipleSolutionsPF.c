#include <ctype.h>
#include <fcntl.h>
#include <libgen.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "MIDAS_Limits.h"
#include "IndexerConsolidatedIO.h"
#include "midas_version.h"
#define BUFFER_SIZE MAX_BUFFER_SIZE

// Function prototypes
void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                  int nScans, double minConf,
                  const ConsolidatedReader *valsReader,
                  const ConsolidatedReader *keysReader);
void writeSpotsToIndex(const char *folderName, const char *originalFolder,
                       int nScans);
inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);
inline double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);

int main(int argc, char *argv[]) {
	printf("Version: %s\n", MIDAS_VERSION_STRING);
  double start_time = omp_get_wtime();
  printf("\n\n\t\tFinding Multiple Solutions in PF-HEDM.\n\n");

  if (argc < 6) {
    printf("Usage: %s foldername sgNum maxAngle nScans nCPUs [minConf]\n\n"
           "The indexing results must be in folderName/Output\n",
           argv[0]);
    return 1;
  }

  char folderName[BUFFER_SIZE];
  sprintf(folderName, "%s/Output/", argv[1]);
  int sgNr = atoi(argv[2]);
  double maxAng = atof(argv[3]);
  int nScans = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  double minConf = 0.0;
  if (argc > 6) {
    minConf = atof(argv[6]);
    printf("Minimum confidence: %lf\n", minConf);
  }

  /* Load consolidated files */
  char consolValsFN[BUFFER_SIZE], consolKeysFN[BUFFER_SIZE];
  sprintf(consolValsFN, "%s/IndexBest_all.bin", folderName);
  sprintf(consolKeysFN, "%s/IndexKey_all.bin", folderName);
  ConsolidatedReader valsReader, keysReader;
  if (ConsolidatedReader_open(&valsReader, consolValsFN) != 0) {
    fprintf(stderr, "Failed to open %s\n", consolValsFN);
    return 1;
  }
  if (ConsolidatedReader_open(&keysReader, consolKeysFN) != 0) {
    fprintf(stderr, "Failed to open %s\n", consolKeysFN);
    return 1;
  }
  printf("Loaded consolidated indexer files (%d voxels)\n", valsReader.nVoxels);

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
    processVoxel(voxNr, folderName, sgNr, maxAng, nScans, minConf,
                 &valsReader, &keysReader);
  }

  ConsolidatedReader_close(&valsReader);
  ConsolidatedReader_close(&keysReader);

  writeSpotsToIndex(folderName, argv[1], nScans);

  printf("Execution completed in %.2f seconds.\n",
         omp_get_wtime() - start_time);
  return 0;
}

void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                  int nScans, double minConf,
                  const ConsolidatedReader *valsReader,
                  const ConsolidatedReader *keysReader) {
  /* Read from consolidated files */
  int nIDs = keysReader->nSolutions[voxNr];
  if (nIDs <= 0) return;

  const double *valsData = ConsolidatedReader_getVals(valsReader, voxNr);
  const size_t *keysData = ConsolidatedReader_getKeys(keysReader, voxNr);
  if (!valsData || !keysData) return;

  size_t *keys = calloc(nIDs * 4, sizeof(*keys));
  for (int i = 0; i < nIDs; i++) {
    keys[i * 4 + 0] = keysData[i * CONSOLIDATED_KEY_COLS + 0];
    keys[i * 4 + 1] = keysData[i * CONSOLIDATED_KEY_COLS + 1];
    keys[i * 4 + 2] = 0;
    keys[i * 4 + 3] = 0;
  }

  double *OMArr = calloc(nIDs * 9, sizeof(double));
  double *confIAArr = calloc(nIDs * 2, sizeof(double));
  if (!OMArr || !confIAArr) {
    free(keys);
    free(OMArr);
    free(confIAArr);
    return;
  }

  for (int i = 0; i < nIDs; i++) {
    const double *row = &valsData[i * CONSOLIDATED_VALS_COLS];
    confIAArr[i * 2 + 0] = row[15] / row[14];
    confIAArr[i * 2 + 1] = row[1];
    for (int j = 0; j < 9; j++)
      OMArr[i * 9 + j] = row[2 + j];
  }

  bool *markArr = calloc(nIDs, sizeof(*markArr));
  size_t *uniqueArr = calloc(nIDs * 4, sizeof(*uniqueArr));
  if (!markArr || !uniqueArr) {
    free(keys);
    free(OMArr);
    free(confIAArr);
    free(markArr);
    free(uniqueArr);
    return;
  }

  int nUniques = 0;
  double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3], ang;
  int bRN;
  double bCon, bIA, conIn, iaIn;
  for (int i = 0; i < nIDs; i++) {
    if (markArr[i] == true)
      continue;
    if (confIAArr[i * 2 + 0] < minConf) {
      markArr[i] = true;
      continue;
    }
    for (int j = 0; j < 9; j++)
      OMThis[j] = OMArr[i * 9 + j];
    OrientMat2Quat(OMThis, Quat1);
    bCon = confIAArr[i * 2 + 0];
    bIA = confIAArr[i * 2 + 1];
    bRN = i;
    for (int j = i + 1; j < nIDs; j++) {
      if (markArr[j] == true)
        continue;
      for (int k = 0; k < 9; k++)
        OMInside[k] = OMArr[j * 9 + k];
      OrientMat2Quat(OMInside, Quat2);
      GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
      conIn = confIAArr[j * 2 + 0];
      if (conIn < minConf) {
        markArr[j] = true;
        continue;
      }
      iaIn = confIAArr[j * 2 + 1];
      if (ang < maxAng) {
        if (bCon < conIn) {
          bCon = conIn;
          bIA = iaIn;
          bRN = j;
        } else if (bCon == conIn) {
          if (bIA > iaIn) {
            bCon = conIn;
            bIA = iaIn;
            bRN = j;
          }
        }
        markArr[j] = true;
      }
    }
    for (int j = 0; j < 4; j++)
      uniqueArr[nUniques * 4 + j] = keys[bRN * 4 + j];
    nUniques++;
  }

  char outKeyFN[BUFFER_SIZE];
  sprintf(outKeyFN, "%s/UniqueIndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);
  FILE *outKeyF = fopen(outKeyFN, "w");
  if (outKeyF) {
    for (int i = 0; i < nUniques; i++) {
      fprintf(outKeyF, "%zu %zu %zu %zu\n", uniqueArr[i * 4 + 0],
              uniqueArr[i * 4 + 1], uniqueArr[i * 4 + 2], uniqueArr[i * 4 + 3]);
    }
    fclose(outKeyF);
  }

  free(markArr);
  free(keys);
  free(OMArr);
  free(confIAArr);
  free(uniqueArr);
}

void writeSpotsToIndex(const char *folderName, const char *originalFolder,
                       int nScans) {
  char outFN[BUFFER_SIZE];
  sprintf(outFN, "%s/SpotsToIndex.csv", originalFolder);
  FILE *outF = fopen(outFN, "w");
  if (!outF)
    return;

  for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
    char outKeyFN[BUFFER_SIZE], aline[BUFFER_SIZE];
    sprintf(outKeyFN, "%s/UniqueIndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);
    FILE *outKeyF = fopen(outKeyFN, "r");
    if (!outKeyF)
      continue;

    while (fgets(aline, BUFFER_SIZE, outKeyF) != NULL) {
      fprintf(outF, "%d %s", voxNr, aline);
    }
    fclose(outKeyF);
  }
  fclose(outF);
}