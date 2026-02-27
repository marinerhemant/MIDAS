//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  ProcessGrains.c
//
//
//  Created by Hemant Sharma on 2014/06/24.
//
//  New Features (2014/11/06):
//  - Twins were implemented in a previous version.
//  - Single file reading is implemented now.
//  New Features (2014/11/19):
//  - Strains!!
//

#include "ZarrReader.h"
#include <blosc2.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <zip.h>

#include "MIDAS_Limits.h"
#define NR_MAX_IDS_PER_GRAIN 5000 // Nr spots per grain max.
#define IAColNr 20 // 20 for Internal Angle, 18 for position, 19 for omega

#define EPS 1E-12
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

inline double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = psi;
  Euler[1] = phi;
  Euler[2] = theta;
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

inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);

static inline int FindInternalAnglesTwins(int nrIDs, int *IDs, int *IDsPerGrain,
                                          int *NrIDsPerID, bool *IDsChecked,
                                          double **OPs, double *ID_IA_Mat,
                                          int counter, int Pos, int StartingID,
                                          double *Radiuses, int SGNr) {
  int i, j, k, ThisID, ThisID2;
  bool AreTwins = false;
  ID_IA_Mat[(counter * 4)] = (double)StartingID;
  ID_IA_Mat[(counter * 4) + 1] = (double)Pos;
  ID_IA_Mat[(counter * 4) + 2] = OPs[Pos][IAColNr];
  ID_IA_Mat[(counter * 4) + 3] = Radiuses[Pos];
  IDsChecked[Pos] = true;
  counter++;
  double Angle, Axis[3], q1[4], q2[4], ang;
  double OR1[9], OR2[9];
  for (i = 0; i < 9; i++) {
    OR1[i] = OPs[Pos][i];
  }
  OrientMat2Quat(OR1, q1);
  for (i = 0; i < NrIDsPerID[Pos]; i++) {
    ThisID = IDsPerGrain[(Pos * NR_MAX_IDS_PER_GRAIN) + i];
    for (j = 0; j < nrIDs; j++) {
      ThisID2 = IDs[j];
      if (ThisID == ThisID2 && IDsChecked[j] == false) {
        for (k = 0; k < 9; k++) {
          OR2[k] = OPs[j][k];
        }
        OrientMat2Quat(OR2, q2);
        Angle = GetMisOrientation(q1, q2, Axis, &ang, SGNr);
        AreTwins = (fabs(ang - 60) < 0.4) &&
                   (fabs(Axis[0]) - fabs(Axis[1])) < 0.01 &&
                   (fabs(Axis[2]) - fabs(Axis[1])) < 0.01;
        if (fabs(ang) < 0.4 || AreTwins) {
          counter = FindInternalAnglesTwins(nrIDs, IDs, IDsPerGrain, NrIDsPerID,
                                            IDsChecked, OPs, ID_IA_Mat, counter,
                                            j, ThisID, Radiuses, SGNr);
          break;
        }
      }
    }
  }
  int counte = counter;
  return counte;
}

static inline int FindInternalAngles(int nrIDs, int *IDs, int *IDsPerGrain,
                                     int *NrIDsPerID, bool *IDsChecked,
                                     double **OPs, double *ID_IA_Mat,
                                     int counter, int Pos, int StartingID,
                                     double *Radiuses, int SGNr) {
  int i, j, k, ThisID, ThisID2;
  ID_IA_Mat[(counter * 4)] = (double)StartingID;
  ID_IA_Mat[(counter * 4) + 1] = (double)Pos;
  ID_IA_Mat[(counter * 4) + 2] = OPs[Pos][IAColNr];
  ID_IA_Mat[(counter * 4) + 3] = Radiuses[Pos];
  IDsChecked[Pos] = true;
  counter++;
  double Angle, Axis[3], q1[4], q2[4], ang;
  double OR1[9], OR2[9];
  for (i = 0; i < 9; i++) {
    OR1[i] = OPs[Pos][i];
  }
  OrientMat2Quat(OR1, q1);
  size_t posSize = Pos;
  posSize *= NR_MAX_IDS_PER_GRAIN;
  for (i = 0; i < NrIDsPerID[Pos]; i++) {
    ThisID = IDsPerGrain[(posSize) + i];
    for (j = 0; j < nrIDs; j++) {
      ThisID2 = IDs[j];
      if (ThisID == ThisID2 && IDsChecked[j] == false) {
        for (k = 0; k < 9; k++) {
          OR2[k] = OPs[j][k];
        }
        OrientMat2Quat(OR2, q2);
        Angle = GetMisOrientation(q1, q2, Axis, &ang, SGNr);
        if (fabs(ang) < 0.4) {
          counter = FindInternalAngles(nrIDs, IDs, IDsPerGrain, NrIDsPerID,
                                       IDsChecked, OPs, ID_IA_Mat, counter, j,
                                       ThisID, Radiuses, SGNr);
          break;
        }
      }
    }
  }
  int counte = counter;
  return counte;
}

static inline void QuatToOrientMat(double Quat[4], double OrientMat[9]) {
  double Q1_2, Q2_2, Q3_2, Q12, Q03, Q13, Q02, Q23, Q01;
  Q1_2 = Quat[1] * Quat[1];
  Q2_2 = Quat[2] * Quat[2];
  Q3_2 = Quat[3] * Quat[3];
  Q12 = Quat[1] * Quat[2];
  Q03 = Quat[0] * Quat[3];
  Q13 = Quat[1] * Quat[3];
  Q02 = Quat[0] * Quat[2];
  Q23 = Quat[2] * Quat[3];
  Q01 = Quat[0] * Quat[1];
  OrientMat[0] = 1 - 2 * (Q2_2 + Q3_2);
  OrientMat[1] = 2 * (Q12 - Q03);
  OrientMat[2] = 2 * (Q13 + Q02);
  OrientMat[3] = 2 * (Q12 + Q03);
  OrientMat[4] = 1 - 2 * (Q1_2 + Q3_2);
  OrientMat[5] = 2 * (Q23 - Q01);
  OrientMat[6] = 2 * (Q13 - Q02);
  OrientMat[7] = 2 * (Q23 + Q01);
  OrientMat[8] = 1 - 2 * (Q1_2 + Q2_2);
}

inline void CalcStrainTensorFableBeaudoin(double LatCin[6],
                                          double LatticeParameterFit[6],
                                          double Orient[3][3],
                                          double StrainTensorSample[3][3]);

inline int StrainTensorKenesei(int nspots, double **SpotsInfo, double Distance,
                               double wavelength,
                               double StrainTensorSample[3][3], int **IDHash,
                               double *dspacings, int nRings,
                               int startSpotMatrix, double **SpotMatrix,
                               double *RetVal, double StrainTensorInput[3][3]);

// Simple hash set for IDsDone duplicate checking (O(1) lookup vs O(n) linear
// scan)
#define HASH_TABLE_SIZE 65536 // power of 2
typedef struct HashNode {
  int key;
  struct HashNode *next;
} HashNode;

typedef struct {
  HashNode **buckets;
  int count;
} HashSet;

static HashSet *hashset_create(void) {
  HashSet *hs = malloc(sizeof(HashSet));
  if (hs == NULL)
    return NULL;
  hs->buckets = calloc(HASH_TABLE_SIZE, sizeof(HashNode *));
  if (hs->buckets == NULL) {
    free(hs);
    return NULL;
  }
  hs->count = 0;
  return hs;
}

static int hashset_contains(HashSet *hs, int key) {
  unsigned int idx = ((unsigned int)key * 2654435761U) & (HASH_TABLE_SIZE - 1);
  HashNode *node = hs->buckets[idx];
  while (node) {
    if (node->key == key)
      return 1;
    node = node->next;
  }
  return 0;
}

static int hashset_insert(HashSet *hs, int key) {
  unsigned int idx = ((unsigned int)key * 2654435761U) & (HASH_TABLE_SIZE - 1);
  HashNode *node = hs->buckets[idx];
  while (node) {
    if (node->key == key)
      return 0;
    node = node->next;
  }
  HashNode *newNode = malloc(sizeof(HashNode));
  if (newNode == NULL)
    return -1;
  newNode->key = key;
  newNode->next = hs->buckets[idx];
  hs->buckets[idx] = newNode;
  hs->count++;
  return 1;
}

static void hashset_destroy(HashSet *hs) {
  if (hs == NULL)
    return;
  for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    HashNode *node = hs->buckets[i];
    while (node) {
      HashNode *next = node->next;
      free(node);
      node = next;
    }
  }
  free(hs->buckets);
  free(hs);
}

// Count lines in a CSV file (excluding header)
static int countCSVLines(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (f == NULL)
    return -1;
  setvbuf(f, NULL, _IOFBF, 1 << 20);
  char buf[4096];
  int count = 0;
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf(
        "Usage: ProcessGrainsZarr ZarrZip (optionally)TrackGrains (0 or 1)\n");
    return 1;
  }
  clock_t start, end;
  double diftotal;
  start = clock();
  char line[5024];

  char aline[1000];
  char *str, dummy[1000];
  int Twin = 0, MinNrSpots = 1, SGNr = 225;
  double Distance, wavelength, LatCin[6];
  double BeamThickness = 0, GlobalPosition = 0;
  int NumPhases = 1, PhaseNr = 1;

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

  // Single-pass Zarr parameter scan with early-exit tracking
  int count = 0;
  int paramsFound = 0;
  const int totalParams = 10; // number of parameters to find
  while ((zip_stat_index(arch, count, 0, finfo)) == 0 &&
         paramsFound < totalParams) {
    const char *name = finfo->name;
    if (strstr(name, "analysis/process/analysis_parameters/SpaceGroup/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &SGNr, sizeof(int));
      paramsFound++;
    } else if (strstr(name, "analysis/process/analysis_parameters/Twins/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &Twin, sizeof(int));
      paramsFound++;
    } else if (strstr(name,
                      "analysis/process/analysis_parameters/MinNrSpots/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &MinNrSpots, sizeof(int));
      paramsFound++;
    } else if (strstr(name,
                      "analysis/process/analysis_parameters/NumPhases/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &NumPhases, sizeof(int));
      paramsFound++;
    } else if (strstr(name, "analysis/process/analysis_parameters/PhaseNr/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &PhaseNr, sizeof(int));
      paramsFound++;
    } else if (strstr(
                   name,
                   "analysis/process/analysis_parameters/GlobalPosition/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &GlobalPosition, sizeof(double));
      paramsFound++;
    } else if (strstr(name,
                      "analysis/process/analysis_parameters/BeamThickness/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &BeamThickness, sizeof(double));
      paramsFound++;
    } else if (strstr(
                   name,
                   "analysis/process/analysis_parameters/LatticeParameter/0") !=
               NULL) {
      ReadZarrChunk(arch, count, LatCin, 6 * sizeof(double));
      paramsFound++;
    } else if (strstr(name, "analysis/process/analysis_parameters/Lsd/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &Distance, sizeof(double));
      paramsFound++;
    } else if (strstr(name,
                      "analysis/process/analysis_parameters/Wavelength/0") !=
               NULL) {
      ReadZarrChunk(arch, count, &wavelength, sizeof(double));
      paramsFound++;
    }
    count++;
  }

  int i, j, k, ThisID, counter;
  int *IDs;
  IDs = calloc(MAX_N_IDS, sizeof(*IDs)); // calloc instead of malloc+loop
  if (IDs == NULL) {
    printf("Memory error: could not allocate IDs.\n");
    return 1;
  }
  int nrIDs = 0;
  char IDsFileName[1024];
  FILE *IDsFile;
  sprintf(IDsFileName, "SpotsToIndex.csv");
  printf("Reading IDs file: %s\n", IDsFileName);
  IDsFile = fopen(IDsFileName, "r");
  if (IDsFile == NULL) {
    printf("Could not open spots file.\n");
    return 1;
  }
  while (fgets(line, 5024, IDsFile) != NULL) {
    sscanf(line, "%d", &IDs[nrIDs]);
    if (IDs[nrIDs] < 0)
      continue;
    nrIDs++;
  }
  if (nrIDs == 0) {
    printf("No ID was found in SpotsToIndex.csv file. Please check your "
           "parameters file.\n");
    return 1;
  }
  fclose(IDsFile);
  printf("Total of %d IDs will be sorted into grains now.\n", nrIDs);
  bool *IDsToKeep;
  IDsToKeep = malloc(nrIDs * sizeof(*IDsToKeep));
  double *Radiuses;
  Radiuses = calloc(nrIDs, sizeof(*Radiuses)); // calloc for zero-init
  double *OPThis, **OPs;
  OPThis = calloc(27, sizeof(*OPThis)); // calloc for zero-init
  OPs = allocMatrix(nrIDs, 23);
  int *IDsPerGrain, *NrIDsPerID;
  NrIDsPerID = calloc(nrIDs, sizeof(*NrIDsPerID)); // calloc for zero-init
  size_t sizeMat = NR_MAX_IDS_PER_GRAIN;
  sizeMat *= nrIDs;
  sizeMat *= sizeof(*IDsPerGrain);
  IDsPerGrain = calloc(sizeMat / sizeof(*IDsPerGrain),
                       sizeof(*IDsPerGrain)); // calloc replaces malloc+loop
  if (IDsPerGrain == NULL) {
    printf("Memory error: could not allocate IDsPerGrain.\n");
    return 1;
  }
  for (i = 0; i < nrIDs; i++) {
    IDsToKeep[i] = false;
    for (j = 0; j < 23; j++) {
      OPs[i][j] = 0;
    }
  }
  FILE *fileKey = fopen("Results/Key.bin", "r");
  FILE *fileOPFit = fopen("Results/OrientPosFit.bin", "r");
  FILE *fileProcessKey = fopen("Results/ProcessKey.bin", "r");
  if (fileKey == NULL) {
    printf("Key file was not found. This means nothing was indexed in the "
           "previous step.\nTypically this means parameters were not correct. "
           "Please check.\nExiting.\n");
    return 1;
  }
  if (fileOPFit == NULL) {
    printf("OrientPos file was not found. This means nothing was indexed in "
           "the previous step.\nTypically this means parameters were not "
           "correct. Please check.\nExiting.\n");
    return 1;
  }
  if (fileProcessKey == NULL) {
    printf("ProcessKey file was not found. This means nothing was indexed in "
           "the previous step.\nTypically this means parameters were not "
           "correct. Please check.\nExiting.\n");
    return 1;
  }

  // Bulk read ProcessKey.bin
  size_t readProcess;
  readProcess = fread(IDsPerGrain, NR_MAX_IDS_PER_GRAIN * nrIDs * sizeof(int),
                      1, fileProcessKey);
  fclose(fileProcessKey);

  // Bulk read Key.bin
  int *keyBuf = malloc(nrIDs * 2 * sizeof(int));
  if (keyBuf == NULL) {
    printf("Memory error: could not allocate keyBuf.\n");
    return 1;
  }
  size_t readKey = fread(keyBuf, 2 * sizeof(int), nrIDs, fileKey);
  fclose(fileKey);

  // Bulk read OrientPosFit.bin
  double *opBuf = malloc(nrIDs * 27 * sizeof(double));
  if (opBuf == NULL) {
    printf("Memory error: could not allocate opBuf.\n");
    return 1;
  }
  size_t readOP = fread(opBuf, 27 * sizeof(double), nrIDs, fileOPFit);
  fclose(fileOPFit);

  // Process bulk-read data
  for (i = 0; i < nrIDs; i++) {
    IDsToKeep[i] = true;
    if (keyBuf[i * 2] == 0) {
      IDsToKeep[i] = false;
    }
    NrIDsPerID[i] = keyBuf[i * 2 + 1];
    counter = 0;
    for (j = 0; j < 27; j++) {
      if (j == 0 || j == 10 || j == 14 || j == 21) {
        continue;
      }
      OPs[i][counter] = opBuf[i * 27 + j];
      counter++;
    }
    Radiuses[i] = opBuf[i * 27 + 25];
  }
  free(keyBuf);
  free(opBuf);

  int StartingID, ThisID1, ThisID2;
  int nGrainPositions = 0, BestGrainPos, bestGrainID;
  int *GrainPositions, *nGrainsMatched;
  GrainPositions = malloc(nrIDs * sizeof(*GrainPositions));
  nGrainsMatched = malloc(nrIDs * sizeof(*nGrainsMatched));
  double minIA, maxRadThis;
  printf("Read all grain files.\n");
  bool *IDsChecked;
  IDsChecked = malloc(nrIDs * sizeof(*IDsChecked));
  for (i = 0; i < nrIDs; i++)
    IDsChecked[i] = false;
  for (i = 0; i < nrIDs; i++) {
    GrainPositions[i] = 0;
    nGrainsMatched[i] = 0;
    if (IDsToKeep[i] == false) {
      IDsChecked[i] = true;
    }
  }
  double *ID_IA_MAT;
  double ang, Angle, Axis[3], DiffPos, OR1[9], q1[4], OR2[9], q2[4], q3[4];
  int counte, counten, totcount = 0;
  ID_IA_MAT = calloc(MAX_ID_IA_MAT * 4, sizeof(*ID_IA_MAT));
  FILE *fIDs = fopen("GrainIDsKey.csv", "w");
  setvbuf(fIDs, NULL, _IOFBF, 1 << 20); // Buffered output
  int trackGrains = 0;
  if (argc == 3)
    trackGrains = atoi(argv[2]);
  for (i = 0; i < nrIDs; i++) {
    if (i % 1000 == 0)
      printf("Processed %d of %d IDs.\n", i, nrIDs);
    if (IDsChecked[i] == false) {
      counte = 0;
      StartingID = IDs[i];
      maxRadThis = Radiuses[i];
      minIA = OPs[i][IAColNr];
      BestGrainPos = i;
      if (trackGrains == 0) {
        if (Twin == 0) {
          counten = FindInternalAngles(nrIDs, IDs, IDsPerGrain, NrIDsPerID,
                                       IDsChecked, OPs, ID_IA_MAT, counte, i,
                                       StartingID, Radiuses, SGNr);
        } else {
          counten = FindInternalAnglesTwins(nrIDs, IDs, IDsPerGrain, NrIDsPerID,
                                            IDsChecked, OPs, ID_IA_MAT, counte,
                                            i, StartingID, Radiuses, SGNr);
        }
      } else {
        counten = 0;
        ID_IA_MAT[(counten * 4)] = (double)StartingID;
        ID_IA_MAT[(counten * 4) + 1] = (double)i;
        ID_IA_MAT[(counten * 4) + 2] = OPs[i][IAColNr];
        ID_IA_MAT[(counten * 4) + 3] = Radiuses[i];
        counten = 1;
      }
      totcount += counten;
      nGrainsMatched[i] = counten;
      if (counten < MinNrSpots) {
        continue;
      }
      for (j = 0; j < counten; j++) {
        if (ID_IA_MAT[(j * 4) + 2] < minIA) {
          minIA = ID_IA_MAT[(j * 4) + 2];
          BestGrainPos = (int)ID_IA_MAT[(j * 4) + 1];
          bestGrainID = (int)ID_IA_MAT[(j * 4)];
          maxRadThis = ID_IA_MAT[(j * 4) + 3];
        }
      }
      fprintf(fIDs, "%d %d ", bestGrainID, BestGrainPos);
      for (j = 0; j < counten; j++) {
        if ((int)ID_IA_MAT[(j * 4) + 1] == BestGrainPos)
          continue;
        fprintf(fIDs, "%d %d ", (int)ID_IA_MAT[(j * 4)],
                (int)ID_IA_MAT[(j * 4) + 1]);
      }
      fprintf(fIDs, "\n");
      GrainPositions[nGrainPositions] = BestGrainPos;
      Radiuses[BestGrainPos] = maxRadThis;
      nGrainPositions++;
    }
  }
  fclose(fIDs);

  // Write out - use hash set for O(1) duplicate check
  int nGrains = 0;
  HashSet *idsDoneSet = hashset_create();
  if (idsDoneSet == NULL) {
    printf("Memory error: could not create hash set.\n");
    return 1;
  }
  int DoneAlready = 0;
  double StrainTensorSampleKen[3][3];
  double StrainTensorSampleFab[3][3];
  double *dummySampleInfo;
  dummySampleInfo =
      malloc(22 * NR_MAX_IDS_PER_GRAIN * sizeof(*dummySampleInfo));
  double LatticeParameterFit[6], Orient[3][3];
  double **SpotsInfo;
  SpotsInfo = allocMatrix(NR_MAX_IDS_PER_GRAIN, 8);
  int nspots, rown;

  // mmap FitBest.bin for better OS caching
  int fullInfoFile = open("Output/FitBest.bin", O_RDONLY);
  double *fitBestMap = NULL;
  size_t fitBestSize = 0;
  if (fullInfoFile >= 0) {
    struct stat sb;
    if (fstat(fullInfoFile, &sb) == 0 && sb.st_size > 0) {
      fitBestSize = sb.st_size;
      fitBestMap = mmap(0, fitBestSize, PROT_READ, MAP_SHARED, fullInfoFile, 0);
      if (fitBestMap == MAP_FAILED) {
        fitBestMap = NULL;
        printf(
            "Warning: mmap failed for FitBest.bin, falling back to pread.\n");
      }
    }
  }

  size_t OffSt;
  size_t ReadSize;
  double MultR = 1000000.0;
  double BeamCenter = 0, FullVol = 0, VNorm;
  int rown2;
  int **IDHash;
  IDHash = allocMatrixInt(NR_MAX_IDS_PER_GRAIN, 3);
  double *dspacings;
  dspacings = malloc(NR_MAX_IDS_PER_GRAIN * sizeof(*dspacings));
  int nRings = 0;
  char *hashfn = "IDsHash.csv";
  FILE *hashfile = fopen(hashfn, "r");
  int MakeHash = 0;
  if (hashfile != NULL) {
    while (fgets(aline, 2000, hashfile) != NULL) {
      sscanf(aline, "%d %d %d %lf", &IDHash[nRings][0], &IDHash[nRings][1],
             &IDHash[nRings][2], &dspacings[nRings]);
      nRings++;
    }
    fclose(hashfile);
  } else {
    MakeHash = 1;
  }

  // Count-then-allocate for InputMatrix
  int nInputLines = countCSVLines("InputAllExtraInfoFittingAll.csv");
  if (nInputLines < 0) {
    printf("Could not open InputAllExtraInfoFittingAll.csv. Exiting.\n");
    return 1;
  }

  double **SpotMatrix, **InputMatrix;
  SpotMatrix = allocMatrix(NR_MAX_IDS_PER_GRAIN, 12);
  InputMatrix = allocMatrix(nInputLines, 10); // Sized to actual count
  int counterSpotMatrix = 0;
  char *inputallfn = "InputAllExtraInfoFittingAll.csv";
  FILE *inpfile = fopen(inputallfn, "r");
  if (inpfile == NULL) {
    printf("Could not open %s. Exiting.\n", inputallfn);
    return 1;
  }
  setvbuf(inpfile, NULL, _IOFBF, 1 << 20); // Buffered read
  int counterIF = 0;
  fgets(aline, 2000, inpfile);
  int currentRing;
  while (fgets(aline, 2000, inpfile) != NULL) {
    sscanf(aline, "%lf %lf %lf %s %lf %lf %lf %lf %s %s %s %lf %lf %lf",
           &InputMatrix[counterIF][6], &InputMatrix[counterIF][7],
           &InputMatrix[counterIF][0], dummy, &InputMatrix[counterIF][1],
           &InputMatrix[counterIF][5], &InputMatrix[counterIF][4],
           &InputMatrix[counterIF][8], dummy, dummy, dummy,
           &InputMatrix[counterIF][2], &InputMatrix[counterIF][3],
           &InputMatrix[counterIF][9]);
    if ((int)InputMatrix[counterIF][1] != counterIF + 1) {
      printf("IDs dont match.\nExiting\n");
      return (1);
    }
    if (MakeHash == 1) {
      if (counterIF == 0) {
        IDHash[nRings][0] = (int)InputMatrix[counterIF][5];
        IDHash[nRings][1] = counterIF + 1;
        currentRing = (int)InputMatrix[counterIF][5];
        nRings++;
      } else {
        if ((int)InputMatrix[counterIF][5] != currentRing) {
          IDHash[nRings][0] = (int)InputMatrix[counterIF][5];
          IDHash[nRings][1] = counterIF + 1;
          IDHash[nRings - 1][2] = counterIF;
          currentRing = (int)InputMatrix[counterIF][5];
          nRings++;
        }
      }
    }
    counterIF++;
  }
  fclose(inpfile);
  IDHash[nRings - 1][2] = counterIF;
  if (MakeHash == 1) {
    FILE *hklf = fopen("hkls.csv", "r");
    char aline2[2048];
    double ds;
    int rnr;
    while (fgets(aline2, 2000, hklf) != NULL) {
      sscanf(aline2, "%s %s %s %lf %d %s %s %s %s %s %s", dummy, dummy, dummy,
             &ds, &rnr, dummy, dummy, dummy, dummy, dummy, dummy);
      for (i = 0; i < nRings; i++) {
        if (IDHash[i][0] == rnr) {
          dspacings[i] = ds;
        }
      }
    }
    fclose(hklf);
  }
  for (j = 0; j < NR_MAX_IDS_PER_GRAIN; j++)
    for (k = 0; k < 12; k++)
      SpotMatrix[j][k] = 0;
  int rowSpotID, startSpotMatrix;
  double RetVal, Eul[3];
  FILE *spotsfile = fopen("SpotMatrix.csv", "w");
  setvbuf(spotsfile, NULL, _IOFBF, 1 << 20); // Buffered output
  fprintf(spotsfile, "%%"
                     "GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
                     "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n");
  double **FinalMatrix;
  FinalMatrix = allocMatrix(nGrainPositions, 47);
  for (i = 0; i < nGrainPositions; i++) {
    rown = GrainPositions[i];
    if (rown >= nrIDs) {
      printf("Something is wrong. Please check.\n");
      return 1;
    }
    if (trackGrains == 0) {
      // O(1) hash set check instead of O(n) linear scan
      if (hashset_contains(idsDoneSet, IDs[rown])) {
        continue;
      }
      hashset_insert(idsDoneSet, IDs[rown]);

      for (k = 0; k < 9; k++) {
        OR1[k] = OPs[rown][k];
      }
      OrientMat2Quat(OR1, q1);
      for (j = i + 1; j < nGrainPositions; j++) {
        rown2 = GrainPositions[j];
        if (rown2 >= nrIDs) {
          printf("Something is wrong. Please check.\n");
          return 1;
        }
        // O(1) hash set check
        if (hashset_contains(idsDoneSet, IDs[rown2])) {
          continue;
        }
        for (k = 0; k < 9; k++) {
          OR2[k] = OPs[rown2][k];
        }
        OrientMat2Quat(OR2, q2);
        Angle = GetMisOrientation(q1, q2, Axis, &ang, SGNr);
        DiffPos = sqrt((OPs[rown][9] - OPs[rown2][9]) *
                           (OPs[rown][9] - OPs[rown2][9]) +
                       (OPs[rown][10] - OPs[rown2][10]) *
                           (OPs[rown][10] - OPs[rown2][10]) +
                       (OPs[rown][11] - OPs[rown2][11]) *
                           (OPs[rown][11] - OPs[rown2][11]));
        if (ang < 0.1 && DiffPos < 5) {
          hashset_insert(idsDoneSet, IDs[rown2]);
        }
      }
    }
    if (OPs[rown][22] < 0.05) {
      continue;
    }

    nspots = NrIDsPerID[rown];
    OffSt = rown;
    OffSt *= 22;
    OffSt *= NR_MAX_IDS_PER_GRAIN;

    // Use mmap if available, otherwise fall back to pread
    if (fitBestMap != NULL) {
      size_t offsetDoubles = (size_t)rown * 22 * NR_MAX_IDS_PER_GRAIN;
      memcpy(dummySampleInfo, &fitBestMap[offsetDoubles],
             22 * nspots * sizeof(double));
    } else {
      OffSt *= sizeof(double);
      ReadSize = 22 * nspots * sizeof(double);
      int rc = pread(fullInfoFile, dummySampleInfo, ReadSize, OffSt);
    }

    counterSpotMatrix = 0;
    startSpotMatrix = counterSpotMatrix;
    double GrainIDThis = (double)IDs[rown];
    for (j = 0; j < nspots; j++) {
      SpotsInfo[j][0] = dummySampleInfo[j * 22 + 4];
      SpotsInfo[j][1] = dummySampleInfo[j * 22 + 5];
      SpotsInfo[j][2] = dummySampleInfo[j * 22 + 6];
      SpotsInfo[j][3] = dummySampleInfo[j * 22 + 1];
      SpotsInfo[j][4] = dummySampleInfo[j * 22 + 2];
      SpotsInfo[j][5] = dummySampleInfo[j * 22 + 7];
      SpotsInfo[j][6] = dummySampleInfo[j * 22 + 8];
      SpotsInfo[j][7] = dummySampleInfo[j * 22 + 0]; // SpotID
      rowSpotID = (int)dummySampleInfo[j * 22 + 0] - 1;
      if (rowSpotID >= counterIF) {
        printf("Looking at the wrong info. Please check.\n");
        return 1;
      }
      SpotMatrix[counterSpotMatrix][0] = GrainIDThis;                 // GrainID
      SpotMatrix[counterSpotMatrix][1] = dummySampleInfo[j * 22 + 0]; // SpotID
      SpotMatrix[counterSpotMatrix][2] = InputMatrix[rowSpotID][0];   // Omega
      SpotMatrix[counterSpotMatrix][3] = InputMatrix[rowSpotID][2];   // YRaw
      SpotMatrix[counterSpotMatrix][4] = InputMatrix[rowSpotID][3];   // ZRaw
      SpotMatrix[counterSpotMatrix][5] = InputMatrix[rowSpotID][9];   // OmeRaw
      SpotMatrix[counterSpotMatrix][6] = InputMatrix[rowSpotID][4];   // Eta
      SpotMatrix[counterSpotMatrix][7] = InputMatrix[rowSpotID][5];   // RingNr
      SpotMatrix[counterSpotMatrix][8] = InputMatrix[rowSpotID][6];   // YLab
      SpotMatrix[counterSpotMatrix][9] = InputMatrix[rowSpotID][7];   // ZLab
      SpotMatrix[counterSpotMatrix][10] =
          InputMatrix[rowSpotID][8] / 2.0; // Theta
      counterSpotMatrix++;
    }
    LatticeParameterFit[0] = OPs[rown][12];
    LatticeParameterFit[1] = OPs[rown][13];
    LatticeParameterFit[2] = OPs[rown][14];
    LatticeParameterFit[3] = OPs[rown][15];
    LatticeParameterFit[4] = OPs[rown][16];
    LatticeParameterFit[5] = OPs[rown][17];
    Orient[0][0] = OPs[rown][0];
    Orient[0][1] = OPs[rown][1];
    Orient[0][2] = OPs[rown][2];
    Orient[1][0] = OPs[rown][3];
    Orient[1][1] = OPs[rown][4];
    Orient[1][2] = OPs[rown][5];
    Orient[2][0] = OPs[rown][6];
    Orient[2][1] = OPs[rown][7];
    Orient[2][2] = OPs[rown][8];
    CalcStrainTensorFableBeaudoin(LatCin, LatticeParameterFit, Orient,
                                  StrainTensorSampleFab);
    int retval = StrainTensorKenesei(nspots, SpotsInfo, Distance, wavelength,
                                     StrainTensorSampleKen, IDHash, dspacings,
                                     nRings, startSpotMatrix, SpotMatrix,
                                     &RetVal, StrainTensorSampleFab);
    for (j = 0; j < counterSpotMatrix; j++) {
      for (k = 0; k < 2; k++)
        fprintf(spotsfile, "%d\t", (int)SpotMatrix[j][k]);
      for (k = 2; k < 7; k++)
        fprintf(spotsfile, "%lf\t", SpotMatrix[j][k]);
      fprintf(spotsfile, "%d\t", (int)SpotMatrix[j][7]);
      for (k = 8; k < 12; k++)
        fprintf(spotsfile, "%lf\t", SpotMatrix[j][k]);
      fprintf(spotsfile, "\n");
    }
    if (retval == 0) {
      printf("Did not read correct hash table for IDs. Exiting\n");
      return 1;
    }
    FinalMatrix[nGrains][0] = GrainIDThis;
    for (j = 0; j < 21; j++) {
      FinalMatrix[nGrains][j + 1] = OPs[rown][j];
    }
    FinalMatrix[nGrains][22] = Radiuses[rown];
    FinalMatrix[nGrains][23] = OPs[rown][22];
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        FinalMatrix[nGrains][24 + 3 * j + k] =
            MultR * StrainTensorSampleFab[j][k];
        FinalMatrix[nGrains][33 + 3 * j + k] =
            MultR * StrainTensorSampleKen[j][k];
      }
    }
    FinalMatrix[nGrains][42] = MultR * RetVal;
    FinalMatrix[nGrains][43] = (double)PhaseNr;
    OrientMat2Euler(Orient, Eul);
    FinalMatrix[nGrains][44] = Eul[0];
    FinalMatrix[nGrains][45] = Eul[1];
    FinalMatrix[nGrains][46] = Eul[2];
    VNorm = FinalMatrix[nGrains][22] * FinalMatrix[nGrains][22] *
            FinalMatrix[nGrains][22];
    BeamCenter += (FinalMatrix[nGrains][12]) * (VNorm);
    FullVol += VNorm;
    nGrains++;
  }
  printf("Number of grains: %d.\n", nGrains);
  BeamCenter /= FullVol;
  // Write file
  fclose(spotsfile);
  char GrainsFileName[1024];
  sprintf(GrainsFileName, "Grains.csv");
  FILE *GrainsFile;
  GrainsFile = fopen(GrainsFileName, "w");
  if (GrainsFile == NULL) {
    printf("Could not write to Grains.csv. Please check.\n");
    return 1;
  }
  setvbuf(GrainsFile, NULL, _IOFBF, 1 << 20); // Buffered output
  fprintf(GrainsFile, "%%NumGrains %d\n", nGrains);
  fprintf(GrainsFile, "%%BeamCenter %f\n", BeamCenter);
  fprintf(GrainsFile, "%%BeamThickness %f\n", BeamThickness);
  fprintf(GrainsFile, "%%GlobalPosition %f\n", GlobalPosition);
  fprintf(GrainsFile, "%%NumPhases %d\n", NumPhases);
  fprintf(GrainsFile, "%%PhaseInfo\n%%\tSpaceGroup:%d\n", SGNr);
  fprintf(GrainsFile, "%%\tLattice Parameter: %lf %lf %lf %lf %lf %lf\n",
          LatCin[0], LatCin[1], LatCin[2], LatCin[3], LatCin[4], LatCin[5]);
  fprintf(
      GrainsFile,
      "%%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\ta\tb"
      "\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfi"
      "dence\t");
  fprintf(GrainsFile, "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\t"
                      "eFab32\teFab33\t");
  fprintf(GrainsFile,
          "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen"
          "33\tRMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n");
  for (i = 0; i < nGrains; i++) {
    fprintf(GrainsFile, "%d\t", (int)FinalMatrix[i][0]);
    for (j = 1; j < 47; j++) {
      fprintf(GrainsFile, "%lf\t", FinalMatrix[i][j]);
    }
    fprintf(GrainsFile, "\n");
  }
  fclose(GrainsFile);

  // Cleanup
  if (fitBestMap != NULL) {
    munmap(fitBestMap, fitBestSize);
  }
  if (fullInfoFile >= 0) {
    close(fullInfoFile);
  }
  hashset_destroy(idsDoneSet);

  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
