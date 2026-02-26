#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "MIDAS_Limits.h"
#define BUFFER_SIZE MAX_BUFFER_SIZE

/* Spots.bin column layout:
   [0] yCen_det  [1] zCen_det  [2] omega   [3] intensity  [4] spotID
   [5] ringNr    [6] eta       [7] theta    [8] dspacing   [9] scanNr */
#define SPOTS_COLS 10
#define KEY_COLS 4 /* grainID, nSpots, startRowNr, spotListStartPos */
#define OM_COLS 9
#define MAX_N_SPOTS_PER_GRAIN 5000
#define SPOT_META_COLS 4 /* eta, 2theta, yCen, zCen */

/* ---- Structures ---- */

/* Per-voxel: orientation + key for global grouping */
typedef struct {
  double OM[9];
  double confidence;
  size_t key[KEY_COLS]; /* grainID, nSpots, startRowNr, spotListStartPos */
  int voxNr;
} VoxelOrientation;

/* Unique grain across all voxels */
typedef struct {
  double OM[9];
  int representativeVoxNr; /* voxel that has the best confidence */
  size_t key[KEY_COLS];
  /* reverse mapping: which voxels have this grain and their keys */
  int *voxNrs;                 /* array of voxel numbers */
  size_t (*voxKeys)[KEY_COLS]; /* array of keys, one per voxel */
  int nVoxels;
  int capVoxels;
} UniqueGrain;

/* For sorting sinogram rows by omega */
typedef struct {
  double angle;
  double *intensities;
  int *spotIDs;
  double *meta; /* SPOT_META_COLS per scan */
  int origIdx;
} SinoSortData;

/* ---- Function prototypes ---- */
void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                  int nScans, double minConf, VoxelOrientation *voxOrients,
                  int *nOrients);
void writeSpotsToIndex(const char *folderName, const char *originalFolder,
                       int nScans);
void usage(const char *prog);

inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);
inline double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);

/* ---- Utility: mmap a file for reading ---- */
static double *mmap_file_read(const char *path, size_t *sizeOut) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Error: cannot open %s: %s\n", path, strerror(errno));
    return NULL;
  }
  struct stat st;
  if (fstat(fd, &st) < 0 || st.st_size == 0) {
    close(fd);
    fprintf(stderr, "Error: cannot stat or empty file %s\n", path);
    return NULL;
  }
  *sizeOut = (size_t)st.st_size;
  double *p = mmap(NULL, *sizeOut, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    fprintf(stderr, "Error: mmap %s failed: %s\n", path, strerror(errno));
    return NULL;
  }
  return p;
}

/* ---- qsort comparator for sinogram rows by angle ---- */
static int cmp_sino_angle(const void *a, const void *b) {
  double da = ((const SinoSortData *)a)->angle;
  double db = ((const SinoSortData *)b)->angle;
  return (da >= db) ? 1 : -1;
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tFinding Multiple Solutions in PF-HEDM.\n\n");

  if (argc < 6) {
    usage(argv[0]);
    return 1;
  }

  char folderName[BUFFER_SIZE];
  sprintf(folderName, "%s/Output/", argv[1]);
  int sgNr = atoi(argv[2]);
  double maxAng = atof(argv[3]);
  int nScans = atoi(argv[4]);
  int numProcs = atoi(argv[5]);

  /* Optional args: minConf (old mode) or sinogram args (new mode).
     Detect mode: if argc >= 10, sinogram mode.
     Old: prog folder sg maxAng nScans nCPUs [minConf]
     New: prog folder sg maxAng nScans nCPUs tolOme tolEta paramFile
          [normalizeSino] [absTransform] [minConf]              */
  double minConf = 0.0;
  int generateSinograms = 0;
  double tolOme = 1.0, tolEta = 1.0;
  char paramFile[BUFFER_SIZE] = "";
  int normalizeSino = 1, absTransform = 1;

  if (argc >= 9) {
    /* Sinogram generation mode */
    generateSinograms = 1;
    tolOme = atof(argv[6]);
    tolEta = atof(argv[7]);
    strncpy(paramFile, argv[8], BUFFER_SIZE - 1);
    paramFile[BUFFER_SIZE - 1] = '\0';
    if (argc >= 10)
      normalizeSino = atoi(argv[9]);
    if (argc >= 11)
      absTransform = atoi(argv[10]);
    if (argc >= 12)
      minConf = atof(argv[11]);
    printf("Sinogram generation mode enabled.\n");
    printf("  tolOme=%.3f  tolEta=%.3f  normalizeSino=%d  absTransform=%d\n",
           tolOme, tolEta, normalizeSino, absTransform);
  } else if (argc > 6) {
    minConf = atof(argv[6]);
  }
  if (minConf > 0)
    printf("Minimum confidence: %lf\n", minConf);

  /* -- Phase 1: process each voxel (find per-voxel unique orientations) -- */
  int totalVox = nScans * nScans;

  /* Allocate per-voxel orientation storage:
     Each voxel may produce up to some number of unique orientations;
     we store the best one per voxel for global grouping. */
  VoxelOrientation *voxOrients = calloc(totalVox, sizeof(*voxOrients));
  int *nOrientsPerVox = calloc(totalVox, sizeof(*nOrientsPerVox));
  if (!voxOrients || !nOrientsPerVox) {
    fprintf(stderr, "Error: memory allocation failed.\n");
    return 1;
  }

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int voxNr = 0; voxNr < totalVox; voxNr++) {
    processVoxel(voxNr, folderName, sgNr, maxAng, nScans, minConf, voxOrients,
                 nOrientsPerVox);
  }

  writeSpotsToIndex(folderName, argv[1], nScans);

  /* -- Phase 2: sinogram generation (if enabled) -- */
  if (generateSinograms) {
    printf("Phase 2: Finding unique orientations across all voxels...\n");

    /* ---- 2a. Find global unique orientations ---- */
    bool *markArr = calloc(totalVox, sizeof(*markArr));
    if (!markArr) {
      fprintf(stderr, "Error allocating mark array.\n");
      free(voxOrients);
      free(nOrientsPerVox);
      return 1;
    }
    /* Mark voxels with no valid orientation */
    for (int i = 0; i < totalVox; i++) {
      markArr[i] = (nOrientsPerVox[i] == 0);
    }

    /* Allocate unique grains array (upper bound: totalVox) */
    int maxGrains = 0;
    for (int i = 0; i < totalVox; i++)
      if (!markArr[i])
        maxGrains++;

    UniqueGrain *grains =
        calloc(maxGrains > 0 ? maxGrains : 1, sizeof(*grains));
    int nGrains = 0;

    for (int i = 0; i < totalVox; i++) {
      if (markArr[i])
        continue;
      double Quat1[4], Quat2[4], Axis[3], ang;
      OrientMat2Quat(voxOrients[i].OM, Quat1);

      double bestConf = voxOrients[i].confidence;
      int bestIdx = i;

      for (int j = i + 1; j < totalVox; j++) {
        if (markArr[j])
          continue;
        OrientMat2Quat(voxOrients[j].OM, Quat2);
        GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
        if (ang < maxAng) {
          if (voxOrients[j].confidence > bestConf) {
            bestConf = voxOrients[j].confidence;
            bestIdx = j;
          }
          markArr[j] = true;
        }
      }

      /* Create a new unique grain */
      UniqueGrain *g = &grains[nGrains];
      memcpy(g->OM, voxOrients[bestIdx].OM, 9 * sizeof(double));
      g->representativeVoxNr = voxOrients[bestIdx].voxNr;
      memcpy(g->key, voxOrients[bestIdx].key, KEY_COLS * sizeof(size_t));
      g->nVoxels = 0;
      g->capVoxels = 64;
      g->voxNrs = malloc(g->capVoxels * sizeof(int));
      g->voxKeys = malloc(g->capVoxels * sizeof(*g->voxKeys));
      nGrains++;
    }
    free(markArr);

    printf("Found %d unique grains across all voxels.\n", nGrains);

    /* ---- 2b. Build reverse index: for each grain, which voxels have it ----
     */
    printf("Building voxel-to-grain mapping...\n");
    for (int voxNr = 0; voxNr < totalVox; voxNr++) {
      if (nOrientsPerVox[voxNr] == 0)
        continue;

      /* Read this voxel's UniqueIndexKey file to get ALL its orientations */
      char ukFN[BUFFER_SIZE];
      sprintf(ukFN, "%s/UniqueIndexKey_voxNr_%06d.txt", folderName, voxNr);
      FILE *ukF = fopen(ukFN, "r");
      if (!ukF)
        continue;

      char aline[BUFFER_SIZE];
      while (fgets(aline, BUFFER_SIZE, ukF)) {
        size_t vKey[KEY_COLS];
        if (sscanf(aline, "%zu %zu %zu %zu", &vKey[0], &vKey[1], &vKey[2],
                   &vKey[3]) != 4)
          continue;

        /* Read the orientation for this key from IndexBest */
        char valsFN[BUFFER_SIZE];
        sprintf(valsFN, "%s/IndexBest_voxNr_%06d.bin", folderName, voxNr);
        FILE *valsF = fopen(valsFN, "rb");
        if (!valsF)
          continue;

        /* Seek to the correct row: startRowNr * 16 doubles */
        long offset = (long)vKey[2] * 16 * sizeof(double);
        if (fseek(valsF, offset, SEEK_SET) != 0) {
          fclose(valsF);
          continue;
        }
        double tmpRow[16];
        if (fread(tmpRow, sizeof(double), 16, valsF) != 16) {
          fclose(valsF);
          continue;
        }
        fclose(valsF);

        double OMVox[9];
        for (int k = 0; k < 9; k++)
          OMVox[k] = tmpRow[2 + k];
        double QuatV[4];
        OrientMat2Quat(OMVox, QuatV);

        /* Find which grain this orientation belongs to */
        for (int gIdx = 0; gIdx < nGrains; gIdx++) {
          double QuatG[4], Axis[3], ang;
          OrientMat2Quat(grains[gIdx].OM, QuatG);
          GetMisOrientation(QuatV, QuatG, Axis, &ang, sgNr);
          if (ang < maxAng) {
            UniqueGrain *g = &grains[gIdx];
            if (g->nVoxels >= g->capVoxels) {
              g->capVoxels *= 2;
              g->voxNrs = realloc(g->voxNrs, g->capVoxels * sizeof(int));
              g->voxKeys =
                  realloc(g->voxKeys, g->capVoxels * sizeof(*g->voxKeys));
            }
            g->voxNrs[g->nVoxels] = voxNr;
            memcpy(g->voxKeys[g->nVoxels], vKey, KEY_COLS * sizeof(size_t));
            g->nVoxels++;
            break; /* assigned to first matching grain */
          }
        }
      }
      fclose(ukF);
    }

    /* Print grain stats */
    for (int i = 0; i < nGrains; i++) {
      printf("  Grain %d: %d voxels claimed it\n", i, grains[i].nVoxels);
    }

    /* ---- 2c. Save UniqueOrientations.csv ---- */
    {
      char uoFN[BUFFER_SIZE];
      sprintf(uoFN, "%s/UniqueOrientations.csv", argv[1]);
      FILE *uoF = fopen(uoFN, "w");
      if (uoF) {
        fprintf(uoF, "# GrainID RowNr nSpots StartRowNr ListStartPos "
                     "OM1 OM2 OM3 OM4 OM5 OM6 OM7 OM8 OM9\n");
        for (int i = 0; i < nGrains; i++) {
          fprintf(uoF, "%d %d %zu %zu %zu ", i, grains[i].representativeVoxNr,
                  grains[i].key[1], grains[i].key[2], grains[i].key[3]);
          for (int j = 0; j < 9; j++)
            fprintf(uoF, "%lf ", grains[i].OM[j]);
          fprintf(uoF, "\n");
        }
        fclose(uoF);
        printf("Saved UniqueOrientations.csv\n");
      }
    }

    /* ---- 2d. MMap Spots.bin ---- */
    char spotsFN[BUFFER_SIZE];
    sprintf(spotsFN, "%s/Spots.bin", argv[1]);
    size_t spotsSize;
    double *allSpots = mmap_file_read(spotsFN, &spotsSize);
    if (!allSpots) {
      fprintf(stderr, "Error: could not mmap Spots.bin\n");
      goto cleanup_grains;
    }
    size_t nSpotsAll = spotsSize / (SPOTS_COLS * sizeof(double));
    printf("Spots.bin: %zu spots\n", nSpotsAll);

    /* ---- 2e. Collect spots from indexing results per grain ---- */
    /* For each grain, collect all spot IDs from all its voxels,
       group them into HKL slots, and fill the sinogram. */

    /* First pass: count max HKLs per grain and collect spot data */
    /* We'll build per-grain spot lists, then figure out HKL slots. */

    /* Simple approach: for each grain, collect (spotID, scanNr, intensity,
       omega, eta, ringNr) from its voxels' IndexBest_IDs files.
       Then group by (ringNr, omega) to get HKL slots.
       Fill sinogram[grain][hklSlot][scanNr] = intensity. */

    typedef struct {
      int spotID;
      int scanNr;
      double intensity;
      double omega;
      double eta;
      double yCen, zCen, theta;
      int ringNr;
    } CollectedSpot;

    int *nHKLsPerGrain = calloc(nGrains, sizeof(int));
    int maxNHKLs = 0;

    /* Per-grain spot collections */
    CollectedSpot **grainSpots = calloc(nGrains, sizeof(*grainSpots));
    int *grainSpotCounts = calloc(nGrains, sizeof(int));
    int *grainSpotCaps = calloc(nGrains, sizeof(int));

    for (int gIdx = 0; gIdx < nGrains; gIdx++) {
      grainSpotCaps[gIdx] = 256;
      grainSpots[gIdx] = malloc(grainSpotCaps[gIdx] * sizeof(CollectedSpot));
      grainSpotCounts[gIdx] = 0;
    }

    printf("Collecting spots from indexing results...\n");
    for (int gIdx = 0; gIdx < nGrains; gIdx++) {
      UniqueGrain *g = &grains[gIdx];
      bool *seenSpot =
          calloc(nSpotsAll + 1, sizeof(bool)); /* dedup by spotID */

      for (int vi = 0; vi < g->nVoxels; vi++) {
        int voxNr = g->voxNrs[vi];
        size_t nSpots = g->voxKeys[vi][1];   /* nSpots from key */
        size_t startPos = g->voxKeys[vi][3]; /* spotListStartPos */

        /* Read spot IDs from IndexBest_IDs */
        char idsFN[BUFFER_SIZE];
        sprintf(idsFN, "%s/IndexBest_IDs_voxNr_%06d.bin", folderName, voxNr);
        FILE *idsF = fopen(idsFN, "rb");
        if (!idsF)
          continue;

        if (fseek(idsF, (long)startPos, SEEK_SET) != 0) {
          fclose(idsF);
          continue;
        }

        int *idArr = malloc(nSpots * sizeof(int));
        if (!idArr) {
          fclose(idsF);
          continue;
        }

        size_t nRead = fread(idArr, sizeof(int), nSpots, idsF);
        fclose(idsF);

        for (size_t si = 0; si < nRead; si++) {
          int sid = idArr[si];
          if (sid < 1 || sid > (int)nSpotsAll)
            continue;
          if (seenSpot[sid])
            continue;
          seenSpot[sid] = true;

          size_t idx = (size_t)(sid - 1);
          /* Verify alignment */
          if ((int)allSpots[SPOTS_COLS * idx + 4] != sid)
            continue;

          /* Grow array if needed */
          if (grainSpotCounts[gIdx] >= grainSpotCaps[gIdx]) {
            grainSpotCaps[gIdx] *= 2;
            grainSpots[gIdx] = realloc(
                grainSpots[gIdx], grainSpotCaps[gIdx] * sizeof(CollectedSpot));
          }

          CollectedSpot *cs = &grainSpots[gIdx][grainSpotCounts[gIdx]];
          cs->spotID = sid;
          cs->scanNr = (int)allSpots[SPOTS_COLS * idx + 9];
          cs->intensity = allSpots[SPOTS_COLS * idx + 3];
          cs->omega = allSpots[SPOTS_COLS * idx + 2];
          cs->eta = allSpots[SPOTS_COLS * idx + 6];
          cs->ringNr = (int)allSpots[SPOTS_COLS * idx + 5];
          cs->yCen = allSpots[SPOTS_COLS * idx + 0];
          cs->zCen = allSpots[SPOTS_COLS * idx + 1];
          cs->theta = allSpots[SPOTS_COLS * idx + 7];
          grainSpotCounts[gIdx]++;
        }
        free(idArr);
      }
      free(seenSpot);
    }

    /* ---- 2f. Group collected spots into HKL slots per grain ---- */
    /* An HKL slot = unique (ringNr, omega within tolOme).
       We assign a slot index to each collected spot. */
    printf("Grouping spots into HKL slots...\n");

    /* For each grain, the slots are determined by grouping spots with
       same ringNr and omega within tolOme. Spots in the same slot but
       different scans fill different columns of the sinogram row. */
    int **spotSlot = calloc(nGrains, sizeof(int *));

    for (int gIdx = 0; gIdx < nGrains; gIdx++) {
      int ncs = grainSpotCounts[gIdx];
      spotSlot[gIdx] = calloc(ncs, sizeof(int));
      for (int i = 0; i < ncs; i++)
        spotSlot[gIdx][i] = -1;

      int nextSlot = 0;

      for (int i = 0; i < ncs; i++) {
        if (spotSlot[gIdx][i] >= 0)
          continue; /* already assigned */
        spotSlot[gIdx][i] = nextSlot;

        /* Find other spots in the same HKL slot */
        for (int j = i + 1; j < ncs; j++) {
          if (spotSlot[gIdx][j] >= 0)
            continue;
          if (grainSpots[gIdx][j].ringNr == grainSpots[gIdx][i].ringNr &&
              fabs(grainSpots[gIdx][j].omega - grainSpots[gIdx][i].omega) <
                  tolOme &&
              fabs(grainSpots[gIdx][j].eta - grainSpots[gIdx][i].eta) <
                  tolEta) {
            spotSlot[gIdx][j] = nextSlot;
          }
        }
        nextSlot++;
      }

      nHKLsPerGrain[gIdx] = nextSlot;
      if (nextSlot > maxNHKLs)
        maxNHKLs = nextSlot;

      printf("  Grain %d: %d collected spots → %d HKL slots\n", gIdx, ncs,
             nextSlot);
    }

    /* ---- 2g. Fill sinogram arrays ---- */
    printf("Filling sinograms (nGrains=%d, maxNHKLs=%d, nScans=%d)...\n",
           nGrains, maxNHKLs, nScans);

    size_t szSino = (size_t)nGrains * maxNHKLs * nScans;
    double *sinoArr = calloc(szSino, sizeof(double));
    int *spotIDArr = calloc(szSino, sizeof(int));
    double *spotMetaArr = calloc(szSino * SPOT_META_COLS, sizeof(double));
    double *omeArr = calloc((size_t)nGrains * maxNHKLs, sizeof(double));
    double *maxIntArr = calloc((size_t)nGrains * maxNHKLs, sizeof(double));
    double *sumOmeArr = calloc((size_t)nGrains * maxNHKLs, sizeof(double));
    int *countOmeArr = calloc((size_t)nGrains * maxNHKLs, sizeof(int));

    if (!sinoArr || !spotIDArr || !spotMetaArr || !omeArr || !maxIntArr ||
        !sumOmeArr || !countOmeArr) {
      fprintf(stderr, "Error: failed to allocate sinogram arrays.\n");
      goto cleanup_spots;
    }

    /* Initialize */
    for (size_t i = 0; i < szSino; i++)
      spotIDArr[i] = -1;
    for (size_t i = 0; i < szSino * SPOT_META_COLS; i++)
      spotMetaArr[i] = NAN;
    for (size_t i = 0; i < (size_t)nGrains * maxNHKLs; i++)
      omeArr[i] = -10000.0;

    /* Fill */
    for (int gIdx = 0; gIdx < nGrains; gIdx++) {
      for (int si = 0; si < grainSpotCounts[gIdx]; si++) {
        CollectedSpot *cs = &grainSpots[gIdx][si];
        int slot = spotSlot[gIdx][si];
        if (slot < 0 || slot >= maxNHKLs)
          continue;
        if (cs->scanNr < 0 || cs->scanNr >= nScans)
          continue;

        size_t locSino = (size_t)gIdx * maxNHKLs * nScans +
                         (size_t)slot * nScans + cs->scanNr;
        size_t locOme = (size_t)gIdx * maxNHKLs + slot;

        /* Keep best intensity per cell */
        if (cs->intensity > sinoArr[locSino]) {
          sinoArr[locSino] = cs->intensity;
          spotIDArr[locSino] = cs->spotID;
          spotMetaArr[locSino * SPOT_META_COLS + 0] = cs->eta;
          spotMetaArr[locSino * SPOT_META_COLS + 1] = cs->theta * 2.0;
          spotMetaArr[locSino * SPOT_META_COLS + 2] = cs->yCen;
          spotMetaArr[locSino * SPOT_META_COLS + 3] = cs->zCen;
        }
        if (cs->intensity > maxIntArr[locOme])
          maxIntArr[locOme] = cs->intensity;
        if (cs->intensity > 0) {
          sumOmeArr[locOme] += cs->omega;
          countOmeArr[locOme]++;
        }
      }
    }

    /* Compute average omegas */
    for (size_t i = 0; i < (size_t)nGrains * maxNHKLs; i++) {
      if (countOmeArr[i] > 0)
        omeArr[i] = sumOmeArr[i] / countOmeArr[i];
    }

    /* Print fill stats */
    for (int g = 0; g < nGrains; g++) {
      int nSp = nHKLsPerGrain[g];
      int filled = 0;
      for (int s = 0; s < nSp; s++)
        for (int sc = 0; sc < nScans; sc++) {
          size_t loc = (size_t)g * maxNHKLs * nScans + s * nScans + sc;
          if (sinoArr[loc] > 0)
            filled++;
        }
      printf("  Grain %d: %d/%d cells filled (%.1f%%)\n", g, filled,
             nSp * nScans,
             (nSp * nScans > 0) ? 100.0 * filled / (nSp * nScans) : 0.0);
    }

    /* ---- 2h. Sort spots by omega angle per grain ---- */
    for (int gIdx = 0; gIdx < nGrains; gIdx++) {
      SinoSortData *sortData = malloc(maxNHKLs * sizeof(*sortData));
      int **sortSpotIDs = malloc(maxNHKLs * sizeof(int *));
      double **sortMeta = malloc(maxNHKLs * sizeof(double *));
      if (!sortData || !sortSpotIDs || !sortMeta) {
        free(sortData);
        free(sortSpotIDs);
        free(sortMeta);
        continue;
      }

      int nValid = 0;
      for (int s = 0; s < maxNHKLs; s++) {
        if (omeArr[gIdx * maxNHKLs + s] > -9999.0) {
          sortData[nValid].angle = omeArr[gIdx * maxNHKLs + s];
          sortData[nValid].intensities = calloc(nScans, sizeof(double));
          sortSpotIDs[nValid] = calloc(nScans, sizeof(int));
          sortMeta[nValid] = calloc(nScans * SPOT_META_COLS, sizeof(double));
          sortData[nValid].origIdx = nValid;

          for (int sc = 0; sc < nScans; sc++) {
            size_t loc = (size_t)gIdx * maxNHKLs * nScans + s * nScans + sc;
            sortData[nValid].intensities[sc] = sinoArr[loc];
            sortSpotIDs[nValid][sc] = spotIDArr[loc];
            for (int m = 0; m < SPOT_META_COLS; m++)
              sortMeta[nValid][sc * SPOT_META_COLS + m] =
                  spotMetaArr[loc * SPOT_META_COLS + m];
          }
          nValid++;
        }
      }

      qsort(sortData, nValid, sizeof(SinoSortData), cmp_sino_angle);

      /* Copy sorted data back */
      for (int s = 0; s < nValid; s++) {
        omeArr[gIdx * maxNHKLs + s] = sortData[s].angle;
        for (int sc = 0; sc < nScans; sc++) {
          size_t loc = (size_t)gIdx * maxNHKLs * nScans + s * nScans + sc;
          sinoArr[loc] = sortData[s].intensities[sc];
          int oi = sortData[s].origIdx;
          spotIDArr[loc] = sortSpotIDs[oi][sc];
          for (int m = 0; m < SPOT_META_COLS; m++)
            spotMetaArr[loc * SPOT_META_COLS + m] =
                sortMeta[oi][sc * SPOT_META_COLS + m];
        }
        free(sortData[s].intensities);
        free(sortSpotIDs[s]);
        free(sortMeta[s]);
      }
      /* Clear remaining slots */
      for (int s = nValid; s < maxNHKLs; s++) {
        omeArr[gIdx * maxNHKLs + s] = -10000.0;
        for (int sc = 0; sc < nScans; sc++) {
          size_t loc = (size_t)gIdx * maxNHKLs * nScans + s * nScans + sc;
          sinoArr[loc] = 0;
          spotIDArr[loc] = -1;
          for (int m = 0; m < SPOT_META_COLS; m++)
            spotMetaArr[loc * SPOT_META_COLS + m] = NAN;
        }
      }
      free(sortData);
      free(sortSpotIDs);
      free(sortMeta);
    }

    /* ---- 2i. Save raw copy, then apply transforms & save all 4 combos ---- */
    double *rawSinoArr = malloc(szSino * sizeof(double));
    if (rawSinoArr)
      memcpy(rawSinoArr, sinoArr, szSino * sizeof(double));

    /* Apply user-requested transforms to sinoArr */
    for (int g = 0; g < nGrains; g++) {
      for (int s = 0; s < maxNHKLs; s++) {
        double maxI = maxIntArr[g * maxNHKLs + s];
        for (int sc = 0; sc < nScans; sc++) {
          size_t loc = (size_t)g * maxNHKLs * nScans + s * nScans + sc;
          if (sinoArr[loc] > 0) {
            if (normalizeSino && maxI > 0)
              sinoArr[loc] /= maxI;
            if (absTransform)
              sinoArr[loc] = exp(-sinoArr[loc]);
          }
        }
      }
    }

    /* Save main sinogram, omegas, nrHKLs */
    char sinoFN[BUFFER_SIZE], omeFN[BUFFER_SIZE], hklFN[BUFFER_SIZE];
    sprintf(sinoFN, "%s/sinos_%d_%d_%d.bin", argv[1], nGrains, maxNHKLs,
            nScans);
    sprintf(omeFN, "%s/omegas_%d_%d.bin", argv[1], nGrains, maxNHKLs);
    sprintf(hklFN, "%s/nrHKLs_%d.bin", argv[1], nGrains);

    FILE *sinoF = fopen(sinoFN, "wb");
    FILE *omeF = fopen(omeFN, "wb");
    FILE *hklF = fopen(hklFN, "wb");
    if (sinoF && omeF && hklF) {
      fwrite(sinoArr, sizeof(double), szSino, sinoF);
      fwrite(omeArr, sizeof(double), (size_t)nGrains * maxNHKLs, omeF);
      fwrite(nHKLsPerGrain, sizeof(int), nGrains, hklF);
      printf("Saved sinogram data:\n  %s\n  %s\n  %s\n", sinoFN, omeFN, hklFN);
    } else {
      fprintf(stderr, "Error: could not open sinogram output files.\n");
    }
    if (sinoF)
      fclose(sinoF);
    if (omeF)
      fclose(omeF);
    if (hklF)
      fclose(hklF);

    /* Save all 4 combinations (raw, norm, abs, normabs) */
    const char *comboNames[4] = {"raw", "norm", "abs", "normabs"};
    int comboNorm[4] = {0, 1, 0, 1};
    int comboAbs[4] = {0, 0, 1, 1};

    for (int combo = 0; combo < 4; combo++) {
      double *cArr = malloc(szSino * sizeof(double));
      if (!cArr)
        continue;
      if (rawSinoArr)
        memcpy(cArr, rawSinoArr, szSino * sizeof(double));
      else
        memcpy(cArr, sinoArr, szSino * sizeof(double));

      for (int g = 0; g < nGrains; g++) {
        for (int s = 0; s < maxNHKLs; s++) {
          double maxI = maxIntArr[g * maxNHKLs + s];
          for (int sc = 0; sc < nScans; sc++) {
            size_t loc = (size_t)g * maxNHKLs * nScans + s * nScans + sc;
            if (cArr[loc] > 0) {
              if (comboNorm[combo] && maxI > 0)
                cArr[loc] /= maxI;
              if (comboAbs[combo])
                cArr[loc] = exp(-cArr[loc]);
            }
          }
        }
      }

      char cFN[BUFFER_SIZE];
      sprintf(cFN, "%s/sinos_%s_%d_%d_%d.bin", argv[1], comboNames[combo],
              nGrains, maxNHKLs, nScans);
      FILE *cF = fopen(cFN, "wb");
      if (cF) {
        fwrite(cArr, sizeof(double), szSino, cF);
        fclose(cF);
      }
      free(cArr);
    }
    printf("Saved all 4 sinogram combinations.\n");
    free(rawSinoArr);

    /* Save spotID mapping */
    {
      char fn[BUFFER_SIZE];
      sprintf(fn, "%s/spotMapping_%d_%d_%d.bin", argv[1], nGrains, maxNHKLs,
              nScans);
      FILE *f = fopen(fn, "wb");
      if (f) {
        fwrite(spotIDArr, sizeof(int), szSino, f);
        fclose(f);
      }
    }
    /* Save spot metadata */
    {
      char fn[BUFFER_SIZE];
      sprintf(fn, "%s/spotMeta_%d_%d_%d.bin", argv[1], nGrains, maxNHKLs,
              nScans);
      FILE *f = fopen(fn, "wb");
      if (f) {
        fwrite(spotMetaArr, sizeof(double), szSino * SPOT_META_COLS, f);
        fclose(f);
      }
    }

  cleanup_spots:
    for (int i = 0; i < nGrains; i++) {
      free(grainSpots[i]);
      free(spotSlot[i]);
    }
    free(grainSpots);
    free(grainSpotCounts);
    free(grainSpotCaps);
    free(spotSlot);
    free(nHKLsPerGrain);
    free(sinoArr);
    free(spotIDArr);
    free(spotMetaArr);
    free(omeArr);
    free(maxIntArr);
    free(sumOmeArr);
    free(countOmeArr);

    munmap(allSpots, spotsSize);

  cleanup_grains:
    for (int i = 0; i < nGrains; i++) {
      free(grains[i].voxNrs);
      free(grains[i].voxKeys);
    }
    free(grains);
  }

  free(voxOrients);
  free(nOrientsPerVox);

  printf("Execution completed in %.2f seconds.\n",
         omp_get_wtime() - start_time);
  return 0;
}

/* ================================================================
 * processVoxel — same logic as before, plus stores best orientation
 *                in shared arrays for global grouping
 * ================================================================ */
void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                  int nScans, double minConf, VoxelOrientation *voxOrients,
                  int *nOrients) {
  char valsFN[BUFFER_SIZE], keyFN[BUFFER_SIZE];
  sprintf(valsFN, "%s/IndexBest_voxNr_%0*d.bin", folderName, 6, voxNr);
  sprintf(keyFN, "%s/IndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);

  FILE *valsF = fopen(valsFN, "rb");
  FILE *keyF = fopen(keyFN, "r");
  if (!valsF || !keyF) {
    if (valsF)
      fclose(valsF);
    if (keyF)
      fclose(keyF);
    nOrients[voxNr] = 0;
    return;
  }

  size_t *keys = calloc(MAX_N_SOLUTIONS_PER_VOX * 4, sizeof(*keys));
  if (!keys) {
    fclose(valsF);
    fclose(keyF);
    nOrients[voxNr] = 0;
    return;
  }

  char aline[BUFFER_SIZE];
  int nIDs = 0;
  while (fgets(aline, BUFFER_SIZE, keyF) != NULL) {
    if (nIDs >= MAX_N_SOLUTIONS_PER_VOX) {
      fprintf(stderr,
              "Warning: voxel %d exceeds MAX_N_SOLUTIONS_PER_VOX (%d), "
              "truncating.\n",
              voxNr, MAX_N_SOLUTIONS_PER_VOX);
      break;
    }
    sscanf(aline, "%zu %zu %zu %zu", &keys[nIDs * 4 + 0], &keys[nIDs * 4 + 1],
           &keys[nIDs * 4 + 2], &keys[nIDs * 4 + 3]);
    nIDs++;
  }
  fclose(keyF);

  if (nIDs == 0) {
    free(keys);
    fclose(valsF);
    nOrients[voxNr] = 0;
    return;
  }

  keys = realloc(keys, nIDs * 4 * sizeof(*keys));
  double *OMArr = calloc(nIDs * 9, sizeof(double));
  double *confIAArr = calloc(nIDs * 2, sizeof(double));
  double *tmpArr = calloc(nIDs * 16, sizeof(double));
  if (!OMArr || !confIAArr || !tmpArr) {
    free(keys);
    free(OMArr);
    free(confIAArr);
    free(tmpArr);
    fclose(valsF);
    nOrients[voxNr] = 0;
    return;
  }

  if (fread(tmpArr, nIDs * 16 * sizeof(double), 1, valsF) != 1) {
    fprintf(stderr, "Warning: incomplete read from %s\n", valsFN);
  }
  fclose(valsF);

  for (int i = 0; i < nIDs; i++) {
    confIAArr[i * 2 + 0] = tmpArr[i * 16 + 15] / tmpArr[i * 16 + 14];
    confIAArr[i * 2 + 1] = tmpArr[i * 16 + 1];
    for (int j = 0; j < 9; j++)
      OMArr[i * 9 + j] = tmpArr[i * 16 + 2 + j];
  }
  free(tmpArr);

  bool *markArr = calloc(nIDs, sizeof(*markArr));
  size_t *uniqueArr = calloc(nIDs * 4, sizeof(*uniqueArr));
  if (!markArr || !uniqueArr) {
    free(keys);
    free(OMArr);
    free(confIAArr);
    free(markArr);
    free(uniqueArr);
    nOrients[voxNr] = 0;
    return;
  }

  int nUniques = 0;
  double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3], ang;
  int bRN;
  double bCon, bIA, conIn, iaIn;

  /* Track best overall orientation for global grouping */
  int bestOverallRow = -1;
  double bestOverallConf = -1;

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

    /* Track best orientation for this voxel (for global grouping) */
    if (bCon > bestOverallConf) {
      bestOverallConf = bCon;
      bestOverallRow = bRN;
    }
    nUniques++;
  }

  /* Write per-voxel UniqueIndexKey (same as before) */
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

  /* Store best orientation in shared array for global grouping */
  if (bestOverallRow >= 0) {
    voxOrients[voxNr].voxNr = voxNr;
    voxOrients[voxNr].confidence = bestOverallConf;
    memcpy(voxOrients[voxNr].OM, &OMArr[bestOverallRow * 9],
           9 * sizeof(double));
    memcpy(voxOrients[voxNr].key, &keys[bestOverallRow * 4],
           KEY_COLS * sizeof(size_t));
    nOrients[voxNr] = nUniques;
  } else {
    nOrients[voxNr] = 0;
  }

  free(markArr);
  free(keys);
  free(OMArr);
  free(confIAArr);
  free(uniqueArr);
}

/* ================================================================
 * writeSpotsToIndex — unchanged
 * ================================================================ */
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

/* ================================================================
 * usage
 * ================================================================ */
void usage(const char *prog) {
  printf(
      "Usage:\n"
      "  Mode 1 (no sinograms):\n"
      "    %s foldername sgNum maxAngle nScans nCPUs [minConf]\n"
      "\n"
      "  Mode 2 (with sinogram generation from indexing results):\n"
      "    %s foldername sgNum maxAngle nScans nCPUs tolOme tolEta paramFile\n"
      "       [normalizeSino] [absTransform] [minConf]\n"
      "\n"
      "  The indexing results must be in folderName/Output\n"
      "\n"
      "  Mode 2 generates sinograms using spots directly matched by the\n"
      "  indexer at each voxel, rather than tolerance-matching against all\n"
      "  spots. This produces cleaner sinograms with less noise.\n",
      prog, prog);
}