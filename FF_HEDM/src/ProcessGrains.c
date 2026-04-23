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
#include "midas_version.h"
#include <blosc2.h>
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <zip.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "GetMisorientation.h"
#include "MIDAS_Limits.h"
#include "MIDAS_ParamParser.h"
#define NR_MAX_IDS_PER_GRAIN 5000 // Nr spots per grain max.
#define IAColNr 20 // 20 for Internal Angle, 18 for position, 19 for omega

#define EPS 1E-12
#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)

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

static inline int FindInternalAnglesTwins(int nrIDs, int *IDs, int *IDsPerGrain,
                                          int *NrIDsPerID, bool *IDsChecked,
                                          double **OPs, double *ID_IA_Mat,
                                          int counter, int Pos, int StartingID,
                                          double *Radiuses, int SGNr,
                                          const int *pos_by_id, int maxID) {
  int i, k, ThisID;
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
    int j;
    if (pos_by_id != NULL && ThisID >= 0 && ThisID <= maxID) {
      j = pos_by_id[ThisID];
      if (j < 0 || IDsChecked[j])
        continue;
    } else {
      for (j = 0; j < nrIDs; j++) {
        if (IDs[j] == ThisID && !IDsChecked[j])
          break;
      }
      if (j == nrIDs)
        continue;
    }
    for (k = 0; k < 9; k++) {
      OR2[k] = OPs[j][k];
    }
    OrientMat2Quat(OR2, q2);
    Angle = GetMisOrientation(q1, q2, Axis, &ang, SGNr);
    AreTwins = (fabs(ang - 60.0 * deg2rad) < 0.4 * deg2rad) &&
               fabs(fabs(Axis[0]) - fabs(Axis[1])) < 0.01 &&
               fabs(fabs(Axis[2]) - fabs(Axis[1])) < 0.01;
    if (fabs(ang) < 0.4 * deg2rad || AreTwins) {
      // Atomically claim j before recursing; another thread may have
      // grabbed it between our racy read above and now.
      if (!__atomic_test_and_set(&IDsChecked[j], __ATOMIC_ACQ_REL)) {
        counter = FindInternalAnglesTwins(
            nrIDs, IDs, IDsPerGrain, NrIDsPerID, IDsChecked, OPs, ID_IA_Mat,
            counter, j, ThisID, Radiuses, SGNr, pos_by_id, maxID);
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
                                     double *Radiuses, int SGNr,
                                     const int *pos_by_id, int maxID) {
  int i, k, ThisID;
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
    int j;
    if (pos_by_id != NULL && ThisID >= 0 && ThisID <= maxID) {
      j = pos_by_id[ThisID];
      if (j < 0 || IDsChecked[j])
        continue;
    } else {
      for (j = 0; j < nrIDs; j++) {
        if (IDs[j] == ThisID && !IDsChecked[j])
          break;
      }
      if (j == nrIDs)
        continue;
    }
    for (k = 0; k < 9; k++) {
      OR2[k] = OPs[j][k];
    }
    OrientMat2Quat(OR2, q2);
    Angle = GetMisOrientation(q1, q2, Axis, &ang, SGNr);
    if (fabs(ang) < 0.4 * deg2rad) {
      // Atomically claim j before recursing; another thread may have
      // grabbed it between our racy read above and now.
      if (!__atomic_test_and_set(&IDsChecked[j], __ATOMIC_ACQ_REL)) {
        counter = FindInternalAngles(nrIDs, IDs, IDsPerGrain, NrIDsPerID,
                                     IDsChecked, OPs, ID_IA_Mat, counter, j,
                                     ThisID, Radiuses, SGNr, pos_by_id, maxID);
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
  // Line-buffer stdout so progress messages appear immediately when piped
  // or captured (default is fully-buffered when stdout is not a TTY).
  setvbuf(stdout, NULL, _IOLBF, 0);
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 2) {
    printf("Usage: ProcessGrains ZarrZip [TrackGrains(0|1)] [nCPUs]\n");
    printf("       ProcessGrains -paramFN params.txt [-trackGrains 0|1] "
           "[-nCPUs N]\n");
    return 1;
  }
  // Wall-clock timing (clock() reports summed CPU time across OpenMP
  // threads, which overstates elapsed time by ~nThreads on parallel runs).
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  double diftotal;
  char line[5024];

  char aline[1000];
  char *str, dummy[1000];
  int Twin = 0, MinNrSpots = 1, SGNr = 225;
  double Distance = 0, wavelength = 0, LatCin[6];
  double BeamThickness = 0, GlobalPosition = 0;
  int NumPhases = 1, PhaseNr = 1;
  int nCPUs = 1;
  int trackGrains = 0;

  // ── Mode detection ──────────────────────────────────────────
  int useParamFile = 0;
  char *paramFN = NULL;

  if (argc >= 2 && strcmp(argv[1], "-paramFN") == 0) {
    useParamFile = 1;
    for (int ai = 1; ai < argc; ai++) {
      if (strcmp(argv[ai], "-paramFN") == 0 && ai + 1 < argc)
        paramFN = argv[++ai];
      else if (strcmp(argv[ai], "-trackGrains") == 0 && ai + 1 < argc)
        trackGrains = atoi(argv[++ai]);
      else if (strcmp(argv[ai], "-nCPUs") == 0 && ai + 1 < argc)
        nCPUs = atoi(argv[++ai]);
    }
    if (!paramFN) {
      fprintf(stderr, "ERROR: -paramFN requires a filename argument.\n");
      return 1;
    }
  }

  blosc2_init();

  if (useParamFile) {
    // ── PARAM-FILE MODE ─────────────────────────────────────
    MIDASConfig cfg;
    if (midas_parse_params(paramFN, &cfg) != 0) {
      printf("Could not parse %s. Exiting.\n", paramFN);
      return 1;
    }
    Twin = cfg.Twins;
    wavelength = cfg.Wavelength;
    BeamThickness = cfg.BeamThickness;
    GlobalPosition = cfg.GlobalPosition;
    NumPhases = cfg.NumPhases;
    PhaseNr = cfg.PhaseNr;
    MinNrSpots = cfg.MinNrSpots;
    SGNr = cfg.SpaceGroup;
    Distance = cfg.Lsd;
    memcpy(LatCin, cfg.LatticeConstant, sizeof(LatCin));
    printf("Read Parameters from %s:\n\tTwins: %d\n\tWavelength: "
           "%lf\n\tBeamThickness: "
           "%lf\n\tGlobalPosition: %lf\n\tNumPhases: %d\n\tPhaseNr: "
           "%d\n\tMinNrSpots: %d\n\tSpaceGroup: %d\n\tDistance: %lf\n",
           paramFN, Twin, wavelength, BeamThickness, GlobalPosition, NumPhases,
           PhaseNr, MinNrSpots, SGNr, Distance);
    printf("\tLatticeConstant: %lf %lf %lf %lf %lf %lf\n", LatCin[0], LatCin[1],
           LatCin[2], LatCin[3], LatCin[4], LatCin[5]);
  } else {
    // ── ZARR MODE ───────────────────────────────────────────
    char *DataFN = argv[1];
    if (argc >= 3) {
      // Second positional arg: trackGrains or nCPUs
      int val2 = atoi(argv[2]);
      if (val2 == 0 || val2 == 1)
        trackGrains = val2;
      else
        nCPUs = val2;
    }
    if (argc >= 4)
      nCPUs = atoi(argv[3]);

    int errorp = 0;
    zip_t *arch = NULL;
    arch = zip_open(DataFN, 0, &errorp);
    if (arch == NULL) {
      fprintf(stderr,
              "ERROR: Could not open zip archive '%s' (error code: %d)\n",
              DataFN, errorp);
      return 1;
    }
    struct zip_stat *finfo = NULL;
    finfo = calloc(16384, sizeof(int));
    zip_stat_init(finfo);

    // Single-pass Zarr parameter scan with early-exit tracking
    int count = 0;
    int paramsFound = 0;
    const int totalParams = 10;
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
      } else if (strstr(name,
                        "analysis/process/analysis_parameters/PhaseNr/0") !=
                 NULL) {
        ReadZarrChunk(arch, count, &PhaseNr, sizeof(int));
        paramsFound++;
      } else if (strstr(
                     name,
                     "analysis/process/analysis_parameters/GlobalPosition/0") !=
                 NULL) {
        ReadZarrChunk(arch, count, &GlobalPosition, sizeof(double));
        paramsFound++;
      } else if (strstr(
                     name,
                     "analysis/process/analysis_parameters/BeamThickness/0") !=
                 NULL) {
        ReadZarrChunk(arch, count, &BeamThickness, sizeof(double));
        paramsFound++;
      } else if (strstr(name, "analysis/process/analysis_parameters/"
                              "LatticeParameter/0") != NULL) {
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
    free(finfo);
    zip_close(arch);
  }

#ifdef _OPENMP
  omp_set_num_threads(nCPUs);
  printf("Running with %d OpenMP threads.\n", nCPUs);
#endif

  int i, j, k, ThisID, counter;
  int *IDs;
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
  /* Pre-count lines so IDs can be sized exactly (prevents overflow past
     the old MAX_N_IDS cap when SpotsToIndex.csv is large). */
  size_t nLinesIDs = 0;
  while (fgets(line, 5024, IDsFile) != NULL)
    nLinesIDs++;
  rewind(IDsFile);
  IDs = calloc(nLinesIDs > 0 ? nLinesIDs : 1, sizeof(*IDs));
  if (IDs == NULL) {
    printf("Memory error: could not allocate IDs (%zu entries).\n", nLinesIDs);
    fclose(IDsFile);
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

  // Build id→position lookup so FindInternalAngles avoids an O(nrIDs)
  // linear scan per child-ID match (dominant cost for 250k+ ID datasets).
  // Keep "first-wins" semantics to match the original linear scan when
  // SpotsToIndex.csv contains duplicate IDs.
  int pos_maxID = 0;
  for (int ii = 0; ii < nrIDs; ii++) {
    if (IDs[ii] > pos_maxID)
      pos_maxID = IDs[ii];
  }
  int *pos_by_id = NULL;
  if (pos_maxID >= 0) {
    size_t mapBytes = ((size_t)pos_maxID + 1) * sizeof(int);
    if (mapBytes <= (size_t)500 * 1024 * 1024) {
      pos_by_id = malloc(mapBytes);
    }
    if (pos_by_id != NULL) {
      for (size_t k = 0; k <= (size_t)pos_maxID; k++)
        pos_by_id[k] = -1;
      for (int ii = 0; ii < nrIDs; ii++) {
        int id = IDs[ii];
        if (id >= 0 && id <= pos_maxID && pos_by_id[id] == -1)
          pos_by_id[id] = ii;
      }
      printf("Built id→position lookup (maxID=%d, %.1f MB).\n", pos_maxID,
             (double)mapBytes / (1024.0 * 1024.0));
    } else {
      printf("pos_by_id map too large (maxID=%d); using linear scan "
             "fallback.\n",
             pos_maxID);
    }
  }
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
  // Kept for compatibility with legacy local decls in subsequent passes.
  double ang, Angle, Axis[3], DiffPos, OR1[9], q1[4], OR2[9], q2[4], q3[4];
  int counte, counten, totcount = 0;
  (void)counte;
  (void)counten;
  (void)StartingID;
  (void)BestGrainPos;
  (void)bestGrainID;
  (void)minIA;
  (void)maxRadThis;
  FILE *fIDs = fopen("GrainIDsKey.csv", "w");
  if (fIDs == NULL) {
    printf("Could not open GrainIDsKey.csv for writing.\n");
    return 1;
  }
  setvbuf(fIDs, NULL, _IOFBF, 1 << 20);

  // Per-thread cluster buffer — an upper bound on cluster size is nrIDs,
  // so size accordingly. 4 doubles per entry (StartingID, Pos, IA, Radius).
  size_t perThreadMatEntries = (size_t)nrIDs;
  if (perThreadMatEntries < 1024)
    perThreadMatEntries = 1024;
  size_t perThreadMatBytes = perThreadMatEntries * 4 * sizeof(double);

  int progressCounter = 0;

  // Stage 1: cluster IDs into grains (parallelized). Each outer i is a
  // potential cluster root; atomic test-and-set on IDsChecked[i] picks
  // exactly one thread per cluster. Children are also claimed atomically
  // inside FindInternalAngles before recursion.
#pragma omp parallel reduction(+ : totcount)
  {
    double *local_ID_IA_MAT = malloc(perThreadMatBytes);
    if (local_ID_IA_MAT == NULL) {
      fprintf(stderr, "Stage 1 per-thread ID_IA_MAT alloc failed.\n");
      exit(1);
    }

#pragma omp for schedule(dynamic, 32)
    for (int ii = 0; ii < nrIDs; ii++) {
      // Atomically claim ii. test_and_set returns previous value; false
      // → we claimed, true → someone else (or the IDsToKeep-false init).
      if (__atomic_test_and_set(&IDsChecked[ii], __ATOMIC_ACQ_REL))
        continue;

      int done;
#pragma omp atomic capture
      done = ++progressCounter;
      if (done % 1000 == 0 || done == 1) {
        printf("Processed %d of %d IDs.\n", done, nrIDs);
      }

      int StartingID_l = IDs[ii];
      double maxRadThis_l = Radiuses[ii];
      double minIA_l = OPs[ii][IAColNr];
      int BestGrainPos_l = ii;
      int bestGrainID_l = StartingID_l;
      int counten_l;

      if (trackGrains == 0) {
        if (Twin == 0) {
          counten_l = FindInternalAngles(
              nrIDs, IDs, IDsPerGrain, NrIDsPerID, IDsChecked, OPs,
              local_ID_IA_MAT, 0, ii, StartingID_l, Radiuses, SGNr, pos_by_id,
              pos_maxID);
        } else {
          counten_l = FindInternalAnglesTwins(
              nrIDs, IDs, IDsPerGrain, NrIDsPerID, IDsChecked, OPs,
              local_ID_IA_MAT, 0, ii, StartingID_l, Radiuses, SGNr, pos_by_id,
              pos_maxID);
        }
      } else {
        local_ID_IA_MAT[0] = (double)StartingID_l;
        local_ID_IA_MAT[1] = (double)ii;
        local_ID_IA_MAT[2] = OPs[ii][IAColNr];
        local_ID_IA_MAT[3] = Radiuses[ii];
        counten_l = 1;
      }

      totcount += counten_l;
      nGrainsMatched[ii] = counten_l;
      if (counten_l < MinNrSpots)
        continue;

      for (int jj = 0; jj < counten_l; jj++) {
        if (local_ID_IA_MAT[(jj * 4) + 2] < minIA_l) {
          minIA_l = local_ID_IA_MAT[(jj * 4) + 2];
          BestGrainPos_l = (int)local_ID_IA_MAT[(jj * 4) + 1];
          bestGrainID_l = (int)local_ID_IA_MAT[(jj * 4)];
          maxRadThis_l = local_ID_IA_MAT[(jj * 4) + 3];
        }
      }

      // Serialized emit of cluster record + GrainPositions push.
#pragma omp critical(stage1_emit)
      {
        fprintf(fIDs, "%d %d ", bestGrainID_l, BestGrainPos_l);
        for (int jj = 0; jj < counten_l; jj++) {
          if ((int)local_ID_IA_MAT[(jj * 4) + 1] == BestGrainPos_l)
            continue;
          fprintf(fIDs, "%d %d ", (int)local_ID_IA_MAT[(jj * 4)],
                  (int)local_ID_IA_MAT[(jj * 4) + 1]);
        }
        fprintf(fIDs, "\n");
        GrainPositions[nGrainPositions] = BestGrainPos_l;
        Radiuses[BestGrainPos_l] = maxRadThis_l;
        nGrainPositions++;
      }
    }

    free(local_ID_IA_MAT);
  }
  fclose(fIDs);

  // ── Dedup + strain computation (two passes, parallelized). ───────
  int nGrains = 0;
  bool *isDup =
      calloc(nGrainPositions > 0 ? nGrainPositions : 1, sizeof(*isDup));
  if (isDup == NULL) {
    printf("Memory error: could not allocate isDup.\n");
    return 1;
  }

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

  double MultR = 1000000.0;
  double BeamCenter = 0, FullVol = 0;
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

  double **InputMatrix;
  InputMatrix = allocMatrix(nInputLines, 10);
  char *inputallfn = "InputAllExtraInfoFittingAll.csv";
  FILE *inpfile = fopen(inputallfn, "r");
  if (inpfile == NULL) {
    printf("Could not open %s. Exiting.\n", inputallfn);
    return 1;
  }
  setvbuf(inpfile, NULL, _IOFBF, 1 << 20);
  int counterIF = 0;
  fgets(aline, 2000, inpfile);
  int currentRing = 0;
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

  // ── Pass A ── Dedup: serial outer i, parallel inner j. isDup[j]=true
  // writes are idempotent, so races between threads are benign. Preserves
  // the original greedy-merge order (outer serial).
  if (trackGrains == 0) {
    for (int ii = 0; ii < nGrainPositions; ii++) {
      if (isDup[ii])
        continue;
      int rown_i = GrainPositions[ii];
      if (rown_i >= nrIDs) {
        printf("Something is wrong. Please check.\n");
        return 1;
      }
      double OR1_s[9], q1_s[4];
      for (int kk = 0; kk < 9; kk++)
        OR1_s[kk] = OPs[rown_i][kk];
      OrientMat2Quat(OR1_s, q1_s);
      double x_i = OPs[rown_i][9], y_i = OPs[rown_i][10], z_i = OPs[rown_i][11];
#pragma omp parallel for schedule(static)
      for (int jj = ii + 1; jj < nGrainPositions; jj++) {
        if (isDup[jj])
          continue;
        int rown_j = GrainPositions[jj];
        if (rown_j >= nrIDs)
          continue; // error path; keptList check below will still catch it
        double OR2_l[9], q2_l[4], Axis_l[3], ang_l;
        for (int kk = 0; kk < 9; kk++)
          OR2_l[kk] = OPs[rown_j][kk];
        OrientMat2Quat(OR2_l, q2_l);
        (void)GetMisOrientation(q1_s, q2_l, Axis_l, &ang_l, SGNr);
        double dx = x_i - OPs[rown_j][9];
        double dy = y_i - OPs[rown_j][10];
        double dz = z_i - OPs[rown_j][11];
        double DiffPos_l = sqrt(dx * dx + dy * dy + dz * dz);
        if (ang_l < 0.1 * deg2rad && DiffPos_l < 5) {
#pragma omp atomic write
          isDup[jj] = true;
        }
      }
    }
  }

  // Build kept list (dedup + confidence filter).
  int *keptList = malloc((nGrainPositions > 0 ? nGrainPositions : 1) *
                         sizeof(*keptList));
  if (keptList == NULL) {
    printf("Memory error: could not allocate keptList.\n");
    return 1;
  }
  int nKept = 0;
  for (int ii = 0; ii < nGrainPositions; ii++) {
    int ri = GrainPositions[ii];
    if (ri >= nrIDs) {
      printf("Something is wrong. Please check.\n");
      return 1;
    }
    if (trackGrains == 0 && isDup[ii])
      continue;
    if (OPs[ri][22] < 0.05)
      continue;
    keptList[nKept++] = ii;
  }

  double **FinalMatrix;
  FinalMatrix = allocMatrix(nKept > 0 ? nKept : 1, 47);

  // Per-iteration buffers for SpotMatrix.csv — keeps output byte-identical
  // to the serial version (rows emitted in keptList order).
  char **perIterBuf = calloc(nKept > 0 ? nKept : 1, sizeof(*perIterBuf));
  size_t *perIterLen = calloc(nKept > 0 ? nKept : 1, sizeof(*perIterLen));
  if (perIterBuf == NULL || perIterLen == NULL) {
    printf("Memory error: could not allocate per-iteration SpotMatrix "
           "buffers.\n");
    return 1;
  }

  double beamCenterAcc = 0, fullVolAcc = 0;

  // ── Pass B ── Strain per kept grain, parallel over kk.
#pragma omp parallel reduction(+ : beamCenterAcc, fullVolAcc)
  {
    double *dummySampleInfo_l =
        malloc(22UL * NR_MAX_IDS_PER_GRAIN * sizeof(*dummySampleInfo_l));
    double **SpotsInfo_l = allocMatrix(NR_MAX_IDS_PER_GRAIN, 8);
    double **SpotMatrix_l = allocMatrix(NR_MAX_IDS_PER_GRAIN, 12);
    if (dummySampleInfo_l == NULL || SpotsInfo_l == NULL ||
        SpotMatrix_l == NULL) {
      fprintf(stderr, "Per-thread allocation failed in strain loop.\n");
      exit(1);
    }
    for (int jj = 0; jj < NR_MAX_IDS_PER_GRAIN; jj++)
      for (int kk = 0; kk < 12; kk++)
        SpotMatrix_l[jj][kk] = 0;

#pragma omp for schedule(dynamic, 8)
    for (int kk = 0; kk < nKept; kk++) {
      int ii = keptList[kk];
      int rown_l = GrainPositions[ii];
      int nspots_l = NrIDsPerID[rown_l];

      if (fitBestMap != NULL) {
        size_t offsetDoubles = (size_t)rown_l * 22 * NR_MAX_IDS_PER_GRAIN;
        memcpy(dummySampleInfo_l, &fitBestMap[offsetDoubles],
               22 * nspots_l * sizeof(double));
      } else {
        size_t offst_l =
            (size_t)rown_l * 22 * NR_MAX_IDS_PER_GRAIN * sizeof(double);
        size_t rs_l = 22 * nspots_l * sizeof(double);
        ssize_t rc = pread(fullInfoFile, dummySampleInfo_l, rs_l, offst_l);
        (void)rc;
      }

      int counterSpotMatrix_l = 0;
      int startSpotMatrix_l = 0;
      double GrainIDThis = (double)IDs[rown_l];
      for (int jj = 0; jj < nspots_l; jj++) {
        SpotsInfo_l[jj][0] = dummySampleInfo_l[jj * 22 + 4];
        SpotsInfo_l[jj][1] = dummySampleInfo_l[jj * 22 + 5];
        SpotsInfo_l[jj][2] = dummySampleInfo_l[jj * 22 + 6];
        SpotsInfo_l[jj][3] = dummySampleInfo_l[jj * 22 + 1];
        SpotsInfo_l[jj][4] = dummySampleInfo_l[jj * 22 + 2];
        SpotsInfo_l[jj][5] = dummySampleInfo_l[jj * 22 + 7];
        SpotsInfo_l[jj][6] = dummySampleInfo_l[jj * 22 + 8];
        SpotsInfo_l[jj][7] = dummySampleInfo_l[jj * 22 + 0];
        int rowSpotID = (int)dummySampleInfo_l[jj * 22 + 0] - 1;
        if (rowSpotID >= counterIF) {
          fprintf(stderr, "Looking at the wrong info. Please check.\n");
          exit(1);
        }
        SpotMatrix_l[counterSpotMatrix_l][0] = GrainIDThis;
        SpotMatrix_l[counterSpotMatrix_l][1] = dummySampleInfo_l[jj * 22 + 0];
        SpotMatrix_l[counterSpotMatrix_l][2] = InputMatrix[rowSpotID][0];
        SpotMatrix_l[counterSpotMatrix_l][3] = InputMatrix[rowSpotID][2];
        SpotMatrix_l[counterSpotMatrix_l][4] = InputMatrix[rowSpotID][3];
        SpotMatrix_l[counterSpotMatrix_l][5] = InputMatrix[rowSpotID][9];
        SpotMatrix_l[counterSpotMatrix_l][6] = InputMatrix[rowSpotID][4];
        SpotMatrix_l[counterSpotMatrix_l][7] = InputMatrix[rowSpotID][5];
        SpotMatrix_l[counterSpotMatrix_l][8] = InputMatrix[rowSpotID][6];
        SpotMatrix_l[counterSpotMatrix_l][9] = InputMatrix[rowSpotID][7];
        SpotMatrix_l[counterSpotMatrix_l][10] =
            InputMatrix[rowSpotID][8] / 2.0;
        counterSpotMatrix_l++;
      }

      double LatticeParameterFit_l[6];
      double Orient_l[3][3];
      double StrainTensorSampleFab_l[3][3];
      double StrainTensorSampleKen_l[3][3];
      double RetVal_l = 0;
      double Eul_l[3];
      LatticeParameterFit_l[0] = OPs[rown_l][12];
      LatticeParameterFit_l[1] = OPs[rown_l][13];
      LatticeParameterFit_l[2] = OPs[rown_l][14];
      LatticeParameterFit_l[3] = OPs[rown_l][15];
      LatticeParameterFit_l[4] = OPs[rown_l][16];
      LatticeParameterFit_l[5] = OPs[rown_l][17];
      Orient_l[0][0] = OPs[rown_l][0];
      Orient_l[0][1] = OPs[rown_l][1];
      Orient_l[0][2] = OPs[rown_l][2];
      Orient_l[1][0] = OPs[rown_l][3];
      Orient_l[1][1] = OPs[rown_l][4];
      Orient_l[1][2] = OPs[rown_l][5];
      Orient_l[2][0] = OPs[rown_l][6];
      Orient_l[2][1] = OPs[rown_l][7];
      Orient_l[2][2] = OPs[rown_l][8];
      CalcStrainTensorFableBeaudoin(LatCin, LatticeParameterFit_l, Orient_l,
                                    StrainTensorSampleFab_l);
      int retval = StrainTensorKenesei(
          nspots_l, SpotsInfo_l, Distance, wavelength, StrainTensorSampleKen_l,
          IDHash, dspacings, nRings, startSpotMatrix_l, SpotMatrix_l, &RetVal_l,
          StrainTensorSampleFab_l);
      if (retval == 0) {
        fprintf(stderr, "Did not read correct hash table for IDs. Exiting.\n");
        exit(1);
      }

      // Render SpotMatrix rows into a per-iteration buffer. ~256 B/row is
      // a safe upper bound given the format string.
      size_t need = (size_t)counterSpotMatrix_l * 256 + 16;
      char *buf = malloc(need);
      if (buf == NULL) {
        fprintf(stderr, "Per-iteration SpotMatrix buffer alloc failed.\n");
        exit(1);
      }
      size_t off = 0;
      for (int jj = 0; jj < counterSpotMatrix_l; jj++) {
        int n = snprintf(buf + off, need - off,
                         "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%d\t%lf\t%lf\t%lf\t"
                         "%lf\t\n",
                         (int)SpotMatrix_l[jj][0], (int)SpotMatrix_l[jj][1],
                         SpotMatrix_l[jj][2], SpotMatrix_l[jj][3],
                         SpotMatrix_l[jj][4], SpotMatrix_l[jj][5],
                         SpotMatrix_l[jj][6], (int)SpotMatrix_l[jj][7],
                         SpotMatrix_l[jj][8], SpotMatrix_l[jj][9],
                         SpotMatrix_l[jj][10], SpotMatrix_l[jj][11]);
        if (n < 0 || (size_t)n >= need - off) {
          fprintf(stderr, "snprintf overflow in SpotMatrix render.\n");
          exit(1);
        }
        off += (size_t)n;
      }
      perIterBuf[kk] = buf;
      perIterLen[kk] = off;

      FinalMatrix[kk][0] = GrainIDThis;
      for (int jj = 0; jj < 21; jj++) {
        FinalMatrix[kk][jj + 1] = OPs[rown_l][jj];
      }
      FinalMatrix[kk][22] = Radiuses[rown_l];
      FinalMatrix[kk][23] = OPs[rown_l][22];
      for (int jj = 0; jj < 3; jj++) {
        for (int kka = 0; kka < 3; kka++) {
          FinalMatrix[kk][24 + 3 * jj + kka] =
              MultR * StrainTensorSampleFab_l[jj][kka];
          FinalMatrix[kk][33 + 3 * jj + kka] =
              MultR * StrainTensorSampleKen_l[jj][kka];
        }
      }
      FinalMatrix[kk][42] = MultR * RetVal_l;
      FinalMatrix[kk][43] = (double)PhaseNr;
      OrientMat2Euler(Orient_l, Eul_l);
      FinalMatrix[kk][44] = Eul_l[0];
      FinalMatrix[kk][45] = Eul_l[1];
      FinalMatrix[kk][46] = Eul_l[2];
      double VNorm_l =
          FinalMatrix[kk][22] * FinalMatrix[kk][22] * FinalMatrix[kk][22];
      beamCenterAcc += FinalMatrix[kk][12] * VNorm_l;
      fullVolAcc += VNorm_l;
    }

    free(dummySampleInfo_l);
    FreeMemMatrix(SpotsInfo_l, NR_MAX_IDS_PER_GRAIN);
    FreeMemMatrix(SpotMatrix_l, NR_MAX_IDS_PER_GRAIN);
  }

  nGrains = nKept;
  BeamCenter = (fullVolAcc > 0) ? (beamCenterAcc / fullVolAcc) : 0;
  FullVol = fullVolAcc;

  // Emit SpotMatrix.csv (ordered by kk, matches serial baseline).
  FILE *spotsfile = fopen("SpotMatrix.csv", "w");
  if (spotsfile == NULL) {
    printf("Could not write to SpotMatrix.csv. Please check.\n");
    return 1;
  }
  setvbuf(spotsfile, NULL, _IOFBF, 1 << 20);
  fprintf(spotsfile, "%%"
                     "GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
                     "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n");
  for (int kk = 0; kk < nKept; kk++) {
    if (perIterBuf[kk] != NULL && perIterLen[kk] > 0) {
      fwrite(perIterBuf[kk], 1, perIterLen[kk], spotsfile);
    }
    free(perIterBuf[kk]);
  }
  free(perIterBuf);
  free(perIterLen);
  fclose(spotsfile);

  printf("Number of grains: %d.\n", nGrains);

  // Write Grains.csv
  char GrainsFileName[1024];
  sprintf(GrainsFileName, "Grains.csv");
  FILE *GrainsFile;
  GrainsFile = fopen(GrainsFileName, "w");
  if (GrainsFile == NULL) {
    printf("Could not write to Grains.csv. Please check.\n");
    return 1;
  }
  setvbuf(GrainsFile, NULL, _IOFBF, 1 << 20);
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
  for (int ii = 0; ii < nGrains; ii++) {
    fprintf(GrainsFile, "%d\t", (int)FinalMatrix[ii][0]);
    for (int jj = 1; jj < 47; jj++) {
      fprintf(GrainsFile, "%lf\t", FinalMatrix[ii][jj]);
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
  free(isDup);
  free(keptList);

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  diftotal = (double)(ts_end.tv_sec - ts_start.tv_sec) +
             (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
