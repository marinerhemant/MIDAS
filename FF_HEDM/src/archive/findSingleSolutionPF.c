#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MAX_N_SOLUTIONS_PER_VOX 1000000
#define MAX_N_SPOTS_PER_GRAIN 5000
#define MAX_N_SPOTS_TOTAL 100000000
#define INVALID_VOX ((size_t)-1)

inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);
inline double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);

static void check(int test, const char *message, ...) {
  if (test) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}

struct InputData {
  double omega;
  double eta;
  int ringNr;
  int mergedID;
  int scanNr;
  int grainNr;
  int spotNr;
};

struct SinoSortData {
  double *intensities;
  double angle;
};

static int cmpfunc(const void *a, const void *b) {
  struct SinoSortData *ia = (struct SinoSortData *)a;
  struct SinoSortData *ib = (struct SinoSortData *)b;
  if (ia->angle >= ib->angle)
    return 1;
  else
    return -1;
}

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tFinding Single Solution in PF-HEDM.\n\n");
  int returncode;
  if (argc < 8 || argc > 10) {
    printf("Usage: %s foldername sgNum maxAngle nScans nCPUs tolOme tolEta "
           "[normalizeSino] [absTransform]\n"
           "  normalizeSino:  0=off, 1=on (default: 1)\n"
           "  absTransform:   0=off, 1=on (default: 1)\n"
           "    When on, applies exp(-I/Imax) so absorption CT codes\n"
           "    can recover diffraction intensity via -ln().\n"
           "\nThe indexing results need to be in folderName/Output\n",
           argv[0]);
    return 1;
  }
  char folderName[2048];
  sprintf(folderName, "%s/Output/", argv[1]);
  int sgNr = atoi(argv[2]);
  double maxAng = atof(argv[3]);
  int nScans = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  double tolOme = atof(argv[6]);
  double tolEta = atof(argv[7]);
  int normalizeSino = (argc >= 9) ? atoi(argv[8]) : 1;
  int absTransform = (argc >= 10) ? atoi(argv[9]) : 1;
  size_t *allKeyArr, *uniqueKeyArr;
  allKeyArr = calloc(nScans * nScans * 4, sizeof(*allKeyArr));
  uniqueKeyArr = calloc(nScans * nScans * 5, sizeof(*uniqueKeyArr));
  double *allOrientationsArr;
  allOrientationsArr =
      calloc(nScans * nScans * 10, sizeof(*allOrientationsArr));
  int voxNr;
#pragma omp parallel for num_threads(numProcs) private(voxNr) schedule(dynamic)
  for (voxNr = 0; voxNr < nScans * nScans; voxNr++) {
    char outKeyFN[2048];
    sprintf(outKeyFN, "%s/UniqueIndexSingleKey.bin", folderName);
    int ib = open(outKeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    FILE *valsF, *keyF;
    char valsFN[2048], keyFN[2048];
    sprintf(valsFN, "%s/IndexBest_voxNr_%0*d.bin", folderName, 6, voxNr);
    sprintf(keyFN, "%s/IndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);
    valsF = fopen(valsFN, "rb");
    keyF = fopen(keyFN, "r");
    if (keyF == NULL || valsF == NULL) {
      if (keyF == NULL)
        printf("Could not open key file %s. Behavior undefined.\n", keyFN);
      if (valsF == NULL)
        printf("Could not open vals file %s. Behavior undefined.\n", valsFN);
      allKeyArr[voxNr * 4 + 0] = INVALID_VOX;
      size_t outarr[5] = {0, 0, 0, 0, 0};
      if (pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr) <
          0)
        perror("pwrite");
      close(ib);
    } else {
      fseek(keyF, 0L, SEEK_END);
      size_t szt = ftell(keyF);
      rewind(keyF);
      if (szt == 0) {
        allKeyArr[voxNr * 4 + 0] = INVALID_VOX;
        fclose(keyF);
        fclose(valsF);
        size_t outarr[5] = {0, 0, 0, 0, 0};
        if (pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr) <
            0)
          perror("pwrite");
        close(ib);
      } else {
        size_t *keys;
        keys = calloc(MAX_N_SOLUTIONS_PER_VOX * 4, sizeof(*keys));
        char aline[2048];
        int nIDs = 0;
        while (fgets(aline, 1000, keyF) != NULL) {
          sscanf(aline, "%zu %zu %zu %zu", &keys[nIDs * 4 + 0],
                 &keys[nIDs * 4 + 1], &keys[nIDs * 4 + 2], &keys[nIDs * 4 + 3]);
          nIDs++;
        }
        keys = realloc(keys, nIDs * 4 * sizeof(*keys));
        fclose(keyF);
        double *OMArr, *tmpArr, *confIAArr;
        OMArr = calloc(nIDs * 9, sizeof(double));
        confIAArr = calloc(nIDs * 2, sizeof(double));
        tmpArr = calloc(nIDs * 16, sizeof(double));
        fread(tmpArr, nIDs * 16 * sizeof(double), 1, valsF);
        fclose(valsF);
        int i, j, k;
        for (i = 0; i < nIDs; i++) {
          confIAArr[i * 2 + 0] = tmpArr[i * 16 + 15] / tmpArr[i * 16 + 14];
          confIAArr[i * 2 + 1] = tmpArr[i * 16 + 1];
          for (k = 0; k < 9; k++)
            OMArr[i * 9 + k] = tmpArr[i * 16 + 2 + k];
        }
        bool *markArr;
        markArr = malloc(nIDs * sizeof(*markArr));
        for (i = 0; i < nIDs; i++)
          markArr[i] = false;
        int bestRow = -1;
        double bestConf = -1, bestIA = 100;
        for (i = 0; i < nIDs; i++) {
          if (markArr[i] == true)
            continue;
          if (confIAArr[i * 2 + 0] < bestConf)
            continue;
          if (confIAArr[i * 2 + 0] == bestConf && confIAArr[i * 2 + 1] > bestIA)
            continue;
          bestConf = confIAArr[i * 2 + 0];
          bestIA = confIAArr[i * 2 + 1];
          bestRow = i;
        }
        if (bestRow == -1) {
          allKeyArr[voxNr * 4 + 0] = INVALID_VOX;
          free(confIAArr);
          free(keys);
          free(markArr);
          free(tmpArr);
          free(OMArr);
          size_t outarr[5] = {0, 0, 0, 0, 0};
          if (pwrite(ib, outarr, 5 * sizeof(size_t),
                     5 * sizeof(size_t) * voxNr) < 0)
            perror("pwrite");
          close(ib);
        } else {
          for (i = 0; i < nIDs; i++)
            markArr[i] = false;
          double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Angle, Axis[3],
              ang;
          size_t *uniqueArrThis;
          uniqueArrThis = calloc(nIDs * 4, sizeof(*uniqueArrThis));
          double *uniqueOrientArrThis;
          uniqueOrientArrThis = calloc(nIDs * 9, sizeof(*uniqueOrientArrThis));
          int nUniquesThis = 0;
          int bRN;
          double bCon, bIA, conIn, iaIn;
          for (i = 0; i < nIDs; i++) {
            if (markArr[i] == true)
              continue;
            for (j = 0; j < 9; j++)
              OMThis[j] = OMArr[i * 9 + j];
            OrientMat2Quat(OMThis, Quat1);
            bCon = confIAArr[i * 2 + 0];
            bIA = confIAArr[i * 2 + 1];
            bRN = i;
            for (j = i + 1; j < nIDs; j++) {
              if (markArr[j] == true)
                continue;
              for (k = 0; k < 9; k++)
                OMInside[k] = OMArr[j * 9 + k];
              OrientMat2Quat(OMInside, Quat2);
              GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
              conIn = confIAArr[j * 2 + 0];
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
            for (j = 0; j < 4; j++)
              uniqueArrThis[nUniquesThis * 4 + j] = keys[bRN * 4 + j];
            for (j = 0; j < 9; j++)
              uniqueOrientArrThis[nUniquesThis * 9 + j] = OMArr[bRN * 9 + j];
            nUniquesThis++;
          }
          FILE *outKeyF;
          size_t outarr[5] = {voxNr, keys[bestRow * 4 + 0],
                              keys[bestRow * 4 + 1], keys[bestRow * 4 + 2],
                              keys[bestRow * 4 + 3]};
          if (pwrite(ib, outarr, 5 * sizeof(size_t),
                     5 * sizeof(size_t) * voxNr) < 0)
            perror("pwrite");
          close(ib);
          sprintf(outKeyFN, "%s/UniqueIndexKeyOrientAll_voxNr_%0*d.txt",
                  folderName, 6, voxNr);
          outKeyF = fopen(outKeyFN, "w");
          for (i = 0; i < nUniquesThis; i++) {
            for (j = 0; j < 4; j++)
              fprintf(outKeyF, "%zu ", uniqueArrThis[i * 4 + j]);
            for (j = 0; j < 9; j++)
              fprintf(outKeyF, "%lf ", uniqueOrientArrThis[i * 9 + j]);
            fprintf(outKeyF, "\n");
          }
          fclose(outKeyF);
          for (i = 0; i < 4; i++)
            allKeyArr[voxNr * 4 + i] = keys[bestRow * 4 + i];
          for (i = 0; i < 9; i++)
            allOrientationsArr[voxNr * 10 + i] = tmpArr[bestRow * 16 + 2 + i];
          allOrientationsArr[voxNr * 10 + 9] = confIAArr[bestRow * 2 + 0];
          free(uniqueArrThis);
          free(uniqueOrientArrThis);
          free(OMArr);
          free(tmpArr);
          free(markArr);
          free(keys);
          free(confIAArr);
        }
      }
    }
  }

  int nUniques = 0;
  int i, j, k;
  bool *markArr2;
  markArr2 = malloc(nScans * nScans * sizeof(*markArr2));
  for (i = 0; i < nScans * nScans; i++) {
    if (allKeyArr[i * 4 + 0] == INVALID_VOX)
      markArr2[i] = true;
    else
      markArr2[i] = false;
  }
  double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3], ang2, fracInside,
      bestFrac;
  int bestOrientationRowNr;
  double *uniqueOrientArr;
  uniqueOrientArr = calloc(nScans * nScans * 9, sizeof(*uniqueOrientArr));
  for (i = 0; i < nScans * nScans; i++) {
    if (markArr2[i] == true)
      continue;
    for (j = 0; j < 9; j++)
      OMThis[j] = allOrientationsArr[i * 10 + j];
    bestFrac = allOrientationsArr[i * 10 + 9];
    OrientMat2Quat(OMThis, Quat1);
    bestOrientationRowNr = i;
    int counter = 1;
    for (j = i + 1; j < nScans * nScans; j++) {
      if (markArr2[j] == true)
        continue;
      fracInside = allOrientationsArr[j * 10 + 9];
      for (k = 0; k < 9; k++)
        OMInside[k] = allOrientationsArr[j * 10 + k];
      OrientMat2Quat(OMInside, Quat2);
      GetMisOrientation(Quat1, Quat2, Axis, &ang2, sgNr);
      if (ang2 < maxAng) {
        if (bestFrac < fracInside) {
          bestFrac = fracInside;
          bestOrientationRowNr = j;
        }
        markArr2[j] = true;
        counter++;
      }
    }
    // printf("%d %lf\n",counter,bestFrac);
    uniqueKeyArr[nUniques * 5 + 0] = bestOrientationRowNr;
    for (j = 0; j < 4; j++)
      uniqueKeyArr[nUniques * 5 + 1 + j] =
          allKeyArr[bestOrientationRowNr * 4 + j];
    for (j = 0; j < 9; j++)
      uniqueOrientArr[nUniques * 9 + j] =
          allOrientationsArr[bestOrientationRowNr * 10 + j];
    nUniques++;
  }
  free(markArr2);
  free(allKeyArr);
  free(allOrientationsArr);
  char *originalFolder = argv[1];
  FILE *uniqueOrientationsF;
  char uniqueOrientsFN[2048];
  sprintf(uniqueOrientsFN, "%s/UniqueOrientations.csv", originalFolder);
  uniqueOrientationsF = fopen(uniqueOrientsFN, "w");
  for (i = 0; i < nUniques; i++) {
    for (j = 0; j < 5; j++)
      fprintf(uniqueOrientationsF, "%zu ", uniqueKeyArr[i * 5 + j]);
    for (j = 0; j < 9; j++)
      fprintf(uniqueOrientationsF, "%lf ", uniqueOrientArr[i * 9 + j]);
    fprintf(uniqueOrientationsF, "\n");
  }
  fclose(uniqueOrientationsF);
  printf("Number of unique orientations: %d\n", nUniques);

  double *AllSpots;
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/Spots.bin", originalFolder);
  char cmmd[4096];
  sprintf(cmmd, "cp %s /dev/shm/", filename);
  int sys_rc = system(cmmd);
  if (sys_rc != 0) {
    fprintf(stderr, "Failed to copy %s to /dev/shm (rc=%d)\n", filename,
            sys_rc);
    return 1;
  }
  sprintf(filename, "/dev/shm/Spots.bin");
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  AllSpots = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(AllSpots == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  size_t nSpotsAll;
  nSpotsAll = size;
  nSpotsAll /= sizeof(double);
  nSpotsAll /= 10;
  printf("nSpotsAll: %zu\n", nSpotsAll);

  struct InputData *allSpotIDs, *allSpotIDsT;
  // double *allSpots;
  allSpotIDsT = calloc(MAX_N_SPOTS_PER_GRAIN * nUniques, sizeof(*allSpotIDsT));
  size_t nAllSpots = 0, thisVoxNr;
  int maxNHKLs = -1, *nrHKLsFilled;
  nrHKLsFilled = calloc(nUniques, sizeof(*nrHKLsFilled));
  size_t startPos, nSpots;
  bool *IsNotUniqueSpot;
  IsNotUniqueSpot =
      calloc(MAX_N_SPOTS_PER_GRAIN * nUniques, sizeof(*IsNotUniqueSpot));
  for (i = 0; i < MAX_N_SPOTS_PER_GRAIN * nUniques; i++)
    IsNotUniqueSpot[i] = false;
  for (i = 0; i < nUniques; i++) {
    thisVoxNr = uniqueKeyArr[i * 5 + 0];
    nSpots = uniqueKeyArr[i * 5 + 2];
    startPos = uniqueKeyArr[i * 5 + 4];
    char IDsFNThis[2048];
    sprintf(IDsFNThis, "%s/IndexBest_IDs_voxNr_%0*d.bin", folderName, 6,
            (int)thisVoxNr);
    FILE *IDF;
    IDF = fopen(IDsFNThis, "rb");
    if (IDF == NULL) {
      fprintf(stderr, "Could not open IDs file %s: %s\n", IDsFNThis,
              strerror(errno));
      continue;
    }
    fseek(IDF, startPos, SEEK_SET);
    int *IDArrThis;
    IDArrThis = malloc(nSpots * sizeof(*IDArrThis));
    fread(IDArrThis, nSpots * sizeof(int), 1, IDF);
    fclose(IDF);
    // printf("graiNr: %d nSpots: %d ",i,nSpots);
    int count2 = 0;
    for (j = 0; j < nSpots; j++) {
      if (IDArrThis[j] > nSpotsAll) {
        printf("IDArrThis[j] %d > nSpotsAll %zu This is a bug. Exiting.\n",
               IDArrThis[j], nSpotsAll);
        return 1;
      }
      if (AllSpots[10 * (IDArrThis[j] - 1) + 4] != (double)IDArrThis[j]) {
        printf("Data is not aligned. Please check. Exiting.\n");
        return 1;
      }
      allSpotIDsT[nAllSpots + j].mergedID = IDArrThis[j];
      allSpotIDsT[nAllSpots + j].omega = AllSpots[10 * (IDArrThis[j] - 1) + 2];
      allSpotIDsT[nAllSpots + j].eta = AllSpots[10 * (IDArrThis[j] - 1) + 6];
      allSpotIDsT[nAllSpots + j].ringNr =
          (int)AllSpots[10 * (IDArrThis[j] - 1) + 5];
      allSpotIDsT[nAllSpots + j].grainNr = i;
      allSpotIDsT[nAllSpots + j].spotNr = j;
      int count = 0;
      for (k = 0; k < nAllSpots + j; k++) {
        if (allSpotIDsT[k].ringNr == allSpotIDsT[nAllSpots + j].ringNr) {
          if (fabs(allSpotIDsT[nAllSpots + j].omega - allSpotIDsT[k].omega) <
              tolOme) {
            if (fabs(allSpotIDsT[nAllSpots + j].eta - allSpotIDsT[k].eta) <
                tolEta) {
              IsNotUniqueSpot[k] = true;
              IsNotUniqueSpot[nAllSpots + j] = true;
              count++;
            }
          }
        }
      }
      if (count == 0)
        count2++;
      // printf("%d ",count);
    }
    // printf("nUniqueSpots: %d\n",count2);
    free(IDArrThis);
    nAllSpots += nSpots;
  }
  allSpotIDsT = realloc(allSpotIDsT, nAllSpots * sizeof(*allSpotIDsT));
  printf("nAllSpotsGrains: %zu ", nAllSpots);
  allSpotIDs = calloc(nAllSpots, sizeof(*allSpotIDs));
  size_t nDone = 0;
  for (i = 0; i < nAllSpots; i++) {
    if (IsNotUniqueSpot[i] == false) {
      allSpotIDs[nDone].mergedID = allSpotIDsT[i].mergedID;
      allSpotIDs[nDone].omega = allSpotIDsT[i].omega;
      allSpotIDs[nDone].eta = allSpotIDsT[i].eta;
      allSpotIDs[nDone].ringNr = allSpotIDsT[i].ringNr;
      allSpotIDs[nDone].grainNr = allSpotIDsT[i].grainNr;
      allSpotIDs[nDone].spotNr = nrHKLsFilled[allSpotIDs[nDone].grainNr];
      nrHKLsFilled[allSpotIDs[nDone].grainNr]++;
      nDone++;
    }
  }
  for (i = 0; i < nUniques; i++) {
    if (nrHKLsFilled[i] > maxNHKLs)
      maxNHKLs = nrHKLsFilled[i];
  }
  // print nrHKLsFilled contents
  for (i = 0; i < nUniques; i++) {
    if (nrHKLsFilled[i] > 0) {
      printf("%d ", nrHKLsFilled[i]);
    }
  }
  printf("\n");
  nAllSpots = nDone;
  allSpotIDs = realloc(allSpotIDs, nAllSpots * sizeof(*allSpotIDs));
  printf("nAllSpotsGrainsUnique: %zu\n", nAllSpots);
  free(uniqueKeyArr);
  free(IsNotUniqueSpot);
  char fnUniqueSpots[2048];
  sprintf(fnUniqueSpots, "%s/UniqueOrientationSpots.csv", originalFolder);
  FILE *fUniqueSpots;
  fUniqueSpots = fopen(fnUniqueSpots, "w");
  fprintf(fUniqueSpots, "ID GrainNr SpotNr RingNr Omega Eta\n");
  for (i = 0; i < nAllSpots; i++)
    fprintf(fUniqueSpots, "%d %d %d %d %lf %lf\n", allSpotIDs[i].mergedID,
            allSpotIDs[i].grainNr, allSpotIDs[i].spotNr, allSpotIDs[i].ringNr,
            allSpotIDs[i].omega, allSpotIDs[i].eta);
  fclose(fUniqueSpots);

  double *sinoArr, *omeArr, *allOmeArr, *maxIntArr;
  size_t szSino = nUniques;
  szSino *= maxNHKLs;
  szSino *= nScans;
  sinoArr = calloc(szSino, sizeof(*sinoArr));
  allOmeArr = calloc(szSino, sizeof(*allOmeArr));
  if (sinoArr == NULL) {
    printf("Could not allocate sinoArr with %zu bytes. Exiting.",
           szSino * sizeof(double));
    return 1;
  }
  for (i = 0; i < szSino; i++)
    sinoArr[i] = 0;
  omeArr = calloc(nUniques * maxNHKLs, sizeof(*omeArr));
  maxIntArr = calloc(nUniques * maxNHKLs, sizeof(*maxIntArr));
  for (i = 0; i < nUniques * maxNHKLs; i++)
    omeArr[i] = -10000.0;

#pragma omp parallel for num_threads(numProcs) private(i) schedule(dynamic)
  for (i = 0; i < nScans; i++) {
    int iterAllSps;
    for (iterAllSps = 0; iterAllSps < nSpotsAll; iterAllSps++) {
      if ((int)AllSpots[10 * iterAllSps + 9] != i)
        continue;
      int iterAllSpIDs;
      for (iterAllSpIDs = 0; iterAllSpIDs < nAllSpots; iterAllSpIDs++) {
        if ((int)AllSpots[10 * iterAllSps + 5] ==
            allSpotIDs[iterAllSpIDs].ringNr) {
          if (fabs(AllSpots[10 * iterAllSps + 2] -
                   allSpotIDs[iterAllSpIDs].omega) < tolOme) {
            if (fabs(AllSpots[10 * iterAllSps + 6] -
                     allSpotIDs[iterAllSpIDs].eta) < tolEta) {
              size_t locThis;
              locThis = ((size_t)allSpotIDs[iterAllSpIDs].grainNr) * maxNHKLs *
                        nScans;
              locThis += ((size_t)allSpotIDs[iterAllSpIDs].spotNr) * nScans;
              locThis += i;
              sinoArr[locThis] = AllSpots[10 * iterAllSps + 3];
              allOmeArr[locThis] = AllSpots[10 * iterAllSps + 2];
              locThis = ((size_t)allSpotIDs[iterAllSpIDs].grainNr) * maxNHKLs +
                        ((size_t)allSpotIDs[iterAllSpIDs].spotNr);
              if (maxIntArr[locThis] < AllSpots[10 * iterAllSps + 3])
                maxIntArr[locThis] = AllSpots[10 * iterAllSps + 3];
            }
          }
        }
      }
    }
  }
  free(allSpotIDs);
  free(allSpotIDsT);
  munmap(AllSpots, size);
  close(fd);
  unlink("/dev/shm/Spots.bin");

  // --- Compute omeArr (average angle per HKL) from raw data ---
  double AvAngle, maxIntThis;
  int nAngles;
  size_t locThis;
  size_t sinoSize = (size_t)nUniques * maxNHKLs * nScans;
  for (i = 0; i < nUniques; i++) {
    for (j = 0; j < maxNHKLs; j++) {
      AvAngle = 0;
      nAngles = 0;
      for (k = 0; k < nScans; k++) {
        locThis = i * maxNHKLs * nScans + j * nScans + k;
        if (sinoArr[locThis] > 0) {
          AvAngle += allOmeArr[locThis];
          nAngles++;
        }
      }
      if (nAngles > 0)
        omeArr[i * maxNHKLs + j] = AvAngle / nAngles;
    }
  }

  // --- Make a raw copy before any transforms ---
  double *rawSinoArr = malloc(sinoSize * sizeof(double));
  memcpy(rawSinoArr, sinoArr, sinoSize * sizeof(double));

  // --- Apply user-requested transforms to sinoArr ---
  for (i = 0; i < nUniques; i++) {
    for (j = 0; j < maxNHKLs; j++) {
      maxIntThis = maxIntArr[i * maxNHKLs + j];
      for (k = 0; k < nScans; k++) {
        locThis = i * maxNHKLs * nScans + j * nScans + k;
        if (sinoArr[locThis] > 0) {
          if (normalizeSino && maxIntThis > 0)
            sinoArr[locThis] /= maxIntThis;
          if (absTransform) {
            sinoArr[locThis] *= -1;
            sinoArr[locThis] = exp(sinoArr[locThis]);
          }
        }
      }
    }
  }

  // --- Sort all sinograms by omega ---
  // Helper: sort a single grain's HKL rows by omega in the given array
  for (i = 0; i < nUniques; i++) {
    struct SinoSortData *st;
    st = malloc(maxNHKLs * sizeof(*st));
    int nSpThis = 0;
    for (j = 0; j < maxNHKLs; j++) {
      if (omeArr[i * maxNHKLs + j] == -10000.0)
        break;
      st[j].angle = omeArr[i * maxNHKLs + j];
      st[j].intensities = calloc(nScans, sizeof(double));
      for (k = 0; k < nScans; k++)
        st[j].intensities[k] = sinoArr[i * maxNHKLs * nScans + j * nScans + k];
      nSpThis++;
    }
    qsort(st, nSpThis, sizeof(struct SinoSortData), cmpfunc);
    for (j = 0; j < nSpThis; j++) {
      omeArr[i * maxNHKLs + j] = st[j].angle;
      for (k = 0; k < nScans; k++)
        sinoArr[i * maxNHKLs * nScans + j * nScans + k] = st[j].intensities[k];
      free(st[j].intensities);
    }
    free(st);
  }

  // --- Write main output files (user-requested combination) ---
  char sinoFN[2048], omeFN[2048], HKLsFN[2048];
  sprintf(sinoFN, "%s/sinos_%d_%d_%d.bin", originalFolder, nUniques, maxNHKLs,
          nScans);
  sprintf(omeFN, "%s/omegas_%d_%d.bin", originalFolder, nUniques, maxNHKLs);
  sprintf(HKLsFN, "%s/nrHKLs_%d.bin", originalFolder, nUniques);
  FILE *sinoF, *omeF, *HKLsF;
  sinoF = fopen(sinoFN, "wb");
  omeF = fopen(omeFN, "wb");
  HKLsF = fopen(HKLsFN, "wb");
  fwrite(sinoArr, sinoSize * sizeof(double), 1, sinoF);
  fwrite(omeArr, nUniques * maxNHKLs * sizeof(*omeArr), 1, omeF);
  fwrite(nrHKLsFilled, nUniques * sizeof(*nrHKLsFilled), 1, HKLsF);
  fclose(sinoF);
  fclose(omeF);
  fclose(HKLsF);

  // --- Generate and save all 4 combinations for visualization ---
  // Combination labels: raw, norm, abs, normabs
  const char *comboNames[4] = {"raw", "norm", "abs", "normabs"};
  int comboNorm[4] = {0, 1, 0, 1};
  int comboAbs[4] = {0, 0, 1, 1};

  for (int combo = 0; combo < 4; combo++) {
    double *comboArr = malloc(sinoSize * sizeof(double));
    memcpy(comboArr, rawSinoArr, sinoSize * sizeof(double));

    // Apply this combo's transforms
    for (i = 0; i < nUniques; i++) {
      for (j = 0; j < maxNHKLs; j++) {
        maxIntThis = maxIntArr[i * maxNHKLs + j];
        for (k = 0; k < nScans; k++) {
          locThis = i * maxNHKLs * nScans + j * nScans + k;
          if (comboArr[locThis] > 0) {
            if (comboNorm[combo] && maxIntThis > 0)
              comboArr[locThis] /= maxIntThis;
            if (comboAbs[combo]) {
              comboArr[locThis] *= -1;
              comboArr[locThis] = exp(comboArr[locThis]);
            }
          }
        }
      }
    }

    // Sort by omega (using the already-sorted omeArr order)
    // Re-sort using same approach, using the raw omeArr copy
    double *comboOmeArr = calloc(nUniques * maxNHKLs, sizeof(double));
    memcpy(comboOmeArr, omeArr, nUniques * maxNHKLs * sizeof(double));

    for (i = 0; i < nUniques; i++) {
      struct SinoSortData *st = malloc(maxNHKLs * sizeof(*st));
      int nSpThis = 0;
      for (j = 0; j < maxNHKLs; j++) {
        if (comboOmeArr[i * maxNHKLs + j] == -10000.0)
          break;
        st[j].angle = comboOmeArr[i * maxNHKLs + j];
        st[j].intensities = calloc(nScans, sizeof(double));
        for (k = 0; k < nScans; k++)
          st[j].intensities[k] =
              comboArr[i * maxNHKLs * nScans + j * nScans + k];
        nSpThis++;
      }
      qsort(st, nSpThis, sizeof(struct SinoSortData), cmpfunc);
      for (j = 0; j < nSpThis; j++) {
        for (k = 0; k < nScans; k++)
          comboArr[i * maxNHKLs * nScans + j * nScans + k] =
              st[j].intensities[k];
        free(st[j].intensities);
      }
      free(st);
    }
    free(comboOmeArr);

    char comboFN[2048];
    sprintf(comboFN, "%s/sinos_%s_%d_%d_%d.bin", originalFolder,
            comboNames[combo], nUniques, maxNHKLs, nScans);
    FILE *comboF = fopen(comboFN, "wb");
    fwrite(comboArr, sinoSize * sizeof(double), 1, comboF);
    fclose(comboF);
    free(comboArr);
  }
  printf("Saved all 4 sinogram combinations.\n");

  free(rawSinoArr);
  free(sinoArr);
  free(omeArr);
  free(allOmeArr);
  free(maxIntArr);
  free(nrHKLsFilled);
  free(uniqueOrientArr);
}