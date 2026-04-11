//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "midas_paths.h"
#include "nf_headers.h"
#include <blosc.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <zip.h>
#include "midas_version.h"

#define RealType double
#define float32_t float
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

/* --- System RAM detection for auto-batch sizing --- */
#if defined(__APPLE__)
#include <sys/sysctl.h>
static size_t get_system_ram(void) {
  int64_t ram;
  size_t len = sizeof(ram);
  if (sysctlbyname("hw.memsize", &ram, &len, NULL, 0) == 0)
    return (size_t)ram;
  return 0;
}
#elif defined(__linux__)
#include <sys/sysinfo.h>
static size_t get_system_ram(void) {
  struct sysinfo si;
  if (sysinfo(&si) == 0)
    return (size_t)si.totalram * si.mem_unit;
  return 0;
}
#else
static size_t get_system_ram(void) { return 0; }
#endif

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;
int SpaceGrp;

static inline int writeStrZip(char outstr[8192], char zarrfn[8192],
                              zip_t *zipper) {
  zip_source_t *sourc0 =
      zip_source_buffer(zipper, (const void *)outstr, strlen(outstr), 0);
  zip_int64_t rcv0 =
      zip_file_add(zipper, zarrfn, sourc0, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
  if (rcv0 < 0) {
    printf("Could not add the file %s to zip. Exiting.\n", zarrfn);
    return 1;
  }
  int rc = zip_set_file_compression(zipper, rcv0, ZIP_CM_STORE, 0);
  if (rc != 0) {
    printf(
        "Could not change compression type of the file %s to zip. Exiting.\n",
        zarrfn);
    return 1;
  }
  return 0;
}

double **allocMatrixF(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      for (int j = 0; j < i; j++)
        free(arr[j]);
      free(arr);
      return NULL;
    }
  }
  return arr;
}

int **allocMatrixIntF(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      for (int j = 0; j < i; j++)
        free(arr[j]);
      free(arr);
      return NULL;
    }
  }
  return arr;
}

int main(int argc, char *argv[]) {
	printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 4 || argc > 5) {
    printf("Usage:\n simulateNF params.txt InputMicFN OutputFN [nCPUs]\n");
    return 1;
  }

  int nCPUs = 1;
  if (argc == 5) {
    nCPUs = atoi(argv[4]);
    if (nCPUs < 1)
      nCPUs = 1;
  }
  omp_set_num_threads(nCPUs);
  printf("Using %d CPU threads\n", nCPUs);

  double start_time, end_time;
  double diftotal;
  start_time = omp_get_wtime();

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  run_midas_binary("GetHKLListNF", ParamFN);
  char *MicFN = argv[2];
  char *outputFN = argv[3];
  char aline[4096];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[4096];
  int LowNr, nLayers;
  double tx, ty, tz;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "nDistances ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nLayers);
      break;
    }
  }
  rewind(fileParam);
  double Lsd[nLayers], ybc[nLayers], zbc[nLayers], ExcludePoleAngle,
      LatticeConstant[6], doubledummy, MaxRingRad, MaxTtheta;
  double px, OmegaStart, OmegaStep;
  char fn[1000];
  char fn2[1000];
  char direct[1000];
  char gridfn[1000];
  double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
  int cntr = 0, countr = 0, conter = 0, StartNr, EndNr, intdummy, SpaceGroup,
      RingsToUse[100], nRingsToUse = 0;
  int NoOfOmegaRanges = 0;
  int nSaves = 1;
  int gridfnfound = 0;
  Wedge = 0;
  int MinMiso = 0;
  int skipBin = 0;
  int WriteImage = 1;
  int SimulationBatches = 0; /* 0 = auto-detect */
  int NrPixelsY = 2048, NrPixelsZ = 2048;
  double MinConfidence = -1.0;
  int OnlySpotsInfo = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd[cntr]);
      cntr++;
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SpaceGroup);
      continue;
    }
    str = "SaveReducedOutput ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &skipBin);
      continue;
    }
    str = "MaxRingRad ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }
    str = "StartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "EndNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &EndNr);
      continue;
    }
    str = "ExcludePoleAngle ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &ExcludePoleAngle);
      continue;
    }
    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeConstant[0],
             &LatticeConstant[1], &LatticeConstant[2], &LatticeConstant[3],
             &LatticeConstant[4], &LatticeConstant[5]);
      continue;
    }
    str = "LatticeConstant ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeConstant[0],
             &LatticeConstant[1], &LatticeConstant[2], &LatticeConstant[3],
             &LatticeConstant[4], &LatticeConstant[5]);
      continue;
    }
    str = "tx ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tx);
      continue;
    }
    str = "ty ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &ty);
      continue;
    }
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &ybc[conter], &zbc[conter]);
      conter++;
      continue;
    }
    str = "tz ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tz);
      continue;
    }
    str = "OmegaStart ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStart);
      continue;
    }
    str = "OmegaStep ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStep);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "Wedge ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wedge);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "RingsToUse ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRingsToUse]);
      nRingsToUse++;
      continue;
    }
    str = "OmegaRange ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &OmegaRanges[NoOfOmegaRanges][0],
             &OmegaRanges[NoOfOmegaRanges][1]);
      NoOfOmegaRanges++;
      continue;
    }
    str = "BoxSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf", dummy, &BoxSizes[countr][0],
             &BoxSizes[countr][1], &BoxSizes[countr][2], &BoxSizes[countr][3]);
      countr++;
      continue;
    }
    str = "WriteImage ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &WriteImage);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      NrPixelsZ = NrPixelsY;
      continue;
    }
    str = "NrPixelsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      continue;
    }
    str = "NrPixelsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      continue;
    }
    str = "MinConfidence ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MinConfidence);
      continue;
    }
    str = "OnlySpotsInfo ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &OnlySpotsInfo);
      continue;
    }
    str = "SimulationBatches ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SimulationBatches);
      continue;
    }
  }
  int i, j, k, l, m, nrFiles, nrPixels;
  for (i = 0; i < NoOfOmegaRanges; i++) {
    OmegaRang[i][0] = OmegaRanges[i][0];
    OmegaRang[i][1] = OmegaRanges[i][1];
  }
  nOmeRang = NoOfOmegaRanges;
  fclose(fileParam);
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd[0]);
  nrFiles = EndNr - StartNr + 1;
  nrPixels = NrPixelsY * NrPixelsZ;
  long long int SizeObsSpots;
  SizeObsSpots = (long long int)nLayers * nrPixels * nrFiles;
  printf("SizeSimulation: %lld elements (%lld bytes)\n", SizeObsSpots,
         SizeObsSpots * (long long int)sizeof(uint16_t));

  /* --- Auto-detect nBatches based on system RAM --- */
  int nBatches = SimulationBatches;
  if (nBatches == 0) {
    size_t sysRAM = get_system_ram();
    size_t imageBytes =
        (size_t)nLayers * nrPixels * nrFiles * sizeof(uint16_t);
    if (sysRAM > 0) {
      size_t availRAM = (size_t)(sysRAM * 0.7); /* use 70% of RAM */
      if (imageBytes <= availRAM) {
        nBatches = 1;
      } else {
        size_t bytesPerFrame = (size_t)nLayers * nrPixels * sizeof(uint16_t);
        size_t framesPerBatch = availRAM / bytesPerFrame;
        if (framesPerBatch < 1)
          framesPerBatch = 1;
        nBatches = (nrFiles + (int)framesPerBatch - 1) / (int)framesPerBatch;
      }
    } else {
      nBatches = 1; /* Can't detect RAM, fall back to single pass */
    }
    printf("Auto-detected SimulationBatches: %d (system RAM: %.1f GB, image "
           "size: %.1f GB)\n",
           nBatches, get_system_ram() / (1024.0 * 1024.0 * 1024.0),
           (double)imageBytes / (1024.0 * 1024.0 * 1024.0));
  } else if (nBatches < 0) {
    nBatches = 1;
  } else {
    printf("SimulationBatches: %d\n", nBatches);
  }

  double RotMatTilts[3][3];
  RotationTilts(tx, ty, tz, RotMatTilts);
  double MatIn[3], P0[nLayers][3], P0T[3];
  double xs, ys, edgeLen, gs, ud, eulThis[3], origConf, XG[3], YG[3], dy1, dy2;
  MatIn[0] = 0;
  MatIn[1] = 0;
  MatIn[2] = 0;
  for (i = 0; i < nLayers; i++) {
    MatIn[0] = -Lsd[i];
    MatrixMultF(RotMatTilts, MatIn, P0T);
    for (j = 0; j < 3; j++) {
      P0[i][j] = P0T[j];
    }
  }
  int n_hkls = 0;
  double hkls[5000][4];
  double Thetas[5000];
  char hklfn[1024];
  sprintf(hklfn, "hkls.csv");
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s", dummy, dummy, dummy,
           dummy, &hkls[n_hkls][3], &hkls[n_hkls][0], &hkls[n_hkls][1],
           &hkls[n_hkls][2], &Thetas[n_hkls], dummy, dummy);
    n_hkls++;
  }
  if (nRingsToUse > 0) {
    double hklTemps[n_hkls][4], thetaTemps[n_hkls];
    int totalHKLs = 0;
    for (i = 0; i < nRingsToUse; i++) {
      for (j = 0; j < n_hkls; j++) {
        if ((int)hkls[j][3] == RingsToUse[i]) {
          hklTemps[totalHKLs][0] = hkls[j][0];
          hklTemps[totalHKLs][1] = hkls[j][1];
          hklTemps[totalHKLs][2] = hkls[j][2];
          hklTemps[totalHKLs][3] = hkls[j][3];
          thetaTemps[totalHKLs] = Thetas[j];
          totalHKLs++;
        }
      }
    }
    for (i = 0; i < totalHKLs; i++) {
      hkls[i][0] = hklTemps[i][0];
      hkls[i][1] = hklTemps[i][1];
      hkls[i][2] = hklTemps[i][2];
      hkls[i][3] = hklTemps[i][3];
      Thetas[i] = thetaTemps[i];
    }
    n_hkls = totalHKLs;
  }
  // Precompute Gs for CalcDiffractionSpots optimization
  double Gs[n_hkls];
  for (i = 0; i < n_hkls; i++) {
    double len = sqrt(hkls[i][0] * hkls[i][0] + hkls[i][1] * hkls[i][1] +
                      hkls[i][2] * hkls[i][2]);
    Gs[i] = sin(Thetas[i] * deg2rad) * len;
  }
  double OMIn[3][3], FracCalc;
  FILE *InpMicF;
  InpMicF = fopen(MicFN, "r");
  printf("Reading from the mic: %s\n", MicFN);
  if (InpMicF == NULL)
    return 1;
  char outFN[4096];
  sprintf(outFN, "%s.bin", outputFN);
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  int lineNr = 0;
  int voxNr = 0;
  FILE *spF = NULL;
  if (!OnlySpotsInfo) {
    spF = fopen("SimulatedSpots.csv", "w");
    char *headOutThis =
        "VoxRowNr\tDistanceNr\tFrameNr\tHorPx\tVerPx\tOmegaRaw\tYRaw\tZRaw";
    fprintf(spF, "%s\n", headOutThis);
  }

  // Pre-read all voxels from mic file into arrays
  int maxVoxels = 1000000; // Initial capacity
  double *allXS = malloc(maxVoxels * sizeof(double));
  double *allYS = malloc(maxVoxels * sizeof(double));
  double *allEdgeLen = malloc(maxVoxels * sizeof(double));
  double *allUD = malloc(maxVoxels * sizeof(double));
  double *allEul = malloc(maxVoxels * 3 * sizeof(double));
  int nVoxels = 0;

  while (fgets(aline, 4096, InpMicF) != NULL) {
    if (nVoxels >= maxVoxels) {
      maxVoxels *= 2;
      allXS = realloc(allXS, maxVoxels * sizeof(double));
      allYS = realloc(allYS, maxVoxels * sizeof(double));
      allEdgeLen = realloc(allEdgeLen, maxVoxels * sizeof(double));
      allUD = realloc(allUD, maxVoxels * sizeof(double));
      allEul = realloc(allEul, maxVoxels * 3 * sizeof(double));
    }
    double origConf_tmp;
    sscanf(aline, "%s %s %s %lf %lf %lf %lf %lf %lf %lf %lf %s", dummy, dummy,
           dummy, &allXS[nVoxels], &allYS[nVoxels], &allEdgeLen[nVoxels],
           &allUD[nVoxels], &allEul[nVoxels * 3 + 0], &allEul[nVoxels * 3 + 1],
           &allEul[nVoxels * 3 + 2], &origConf_tmp, dummy);
    if (MinConfidence > 0 && origConf_tmp < MinConfidence) {
      continue;
    }
    nVoxels++;
  }
  fclose(InpMicF);
  if (MinConfidence > 0) {
    printf("Read %d voxels from mic file (filtered by MinConfidence %.4f)\n",
           nVoxels, MinConfidence);
  } else {
    printf("Read %d voxels from mic file\n", nVoxels);
  }

  /* ================================================================
   * SINGLE-PASS FAST PATH (nBatches == 1): original monolithic code
   * ================================================================ */
  uint16_t *ObsSpotsInfo = NULL;
  uint16_t *binArr = NULL;
  NFHitBuffer *hitBufs = NULL;
  NFPixelHit *allHits = NULL;
  int totalHits = 0;

  if (nBatches == 1) {
    ObsSpotsInfo = calloc(SizeObsSpots, sizeof(*ObsSpotsInfo));
    if (!OnlySpotsInfo)
      binArr = calloc(SizeObsSpots, sizeof(*binArr));
    if (ObsSpotsInfo == NULL || (!OnlySpotsInfo && binArr == NULL)) {
      printf("Could not allocate arrays! Ran out of RAM?\n");
      return 1;
    }

// Parallel loop over voxels — original path
#pragma omp parallel for schedule(dynamic)                                     \
    private(xs, ys, edgeLen, gs, ud, dy1, dy2)
    for (voxNr = 0; voxNr < nVoxels; voxNr++) {
      xs = allXS[voxNr];
      ys = allYS[voxNr];
      edgeLen = allEdgeLen[voxNr];
      ud = allUD[voxNr];
      double eulThis_local[3];
      eulThis_local[0] = allEul[voxNr * 3 + 0];
      eulThis_local[1] = allEul[voxNr * 3 + 1];
      eulThis_local[2] = allEul[voxNr * 3 + 2];
      gs = edgeLen / 2;
      dy1 = edgeLen / sqrt(3);
      dy2 = -edgeLen / (2 * sqrt(3));
      if (ud < 0) {
        dy1 *= -1;
        dy2 *= -1;
      }
      int NrPixelsGrid = 2 * (ceil((gs * 2) / px)) * (ceil((gs * 2) / px));
      if (gs * 2 < px)
        NrPixelsGrid = 1;
      double XG_local[3], YG_local[3];
      XG_local[0] = xs;
      XG_local[1] = xs - gs;
      XG_local[2] = xs + gs;
      YG_local[0] = ys + dy1;
      YG_local[1] = ys + dy2;
      YG_local[2] = ys + dy2;
      double OMIn_local[3][3];
      Euler2OrientMat(eulThis_local, OMIn_local);
      int **InPixels_local;
      InPixels_local = allocMatrixIntF(NrPixelsGrid, 2);
      double *TheorSpots_local;
      TheorSpots_local = malloc(MAX_N_SPOTS * 3 * sizeof(*TheorSpots_local));
      SimulateAccOrient(nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots,
                        XG_local, YG_local, RotMatTilts, OmegaStart, OmegaStep,
                        px, ybc, zbc, gs, hkls, n_hkls, Thetas, OmegaRanges,
                        NoOfOmegaRanges, BoxSizes, P0, NrPixelsGrid,
                        ObsSpotsInfo, OMIn_local, TheorSpots_local, voxNr, spF,
                        InPixels_local, Gs, NrPixelsY, NrPixelsZ);
      FreeMemMatrixInt(InPixels_local, NrPixelsGrid);
      free(TheorSpots_local);
    }
  } else {
    /* ================================================================
     * BATCHED PATH: spot-first — compute all hits once, replay later
     * ================================================================ */
    printf("Using spot-first batched mode (%d batches)\n", nBatches);

    /* Per-thread hit buffers */
    hitBufs = calloc(nCPUs, sizeof(NFHitBuffer));
    for (i = 0; i < nCPUs; i++) {
      hitBufs[i].capacity = 4096;
      hitBufs[i].hits = malloc(hitBufs[i].capacity * sizeof(NFPixelHit));
      hitBufs[i].count = 0;
    }

    /* Phase 1: compute all voxels, buffer hits */
    double phase1_start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)                                     \
    private(xs, ys, edgeLen, gs, ud, dy1, dy2)
    for (voxNr = 0; voxNr < nVoxels; voxNr++) {
      int tid = omp_get_thread_num();
      xs = allXS[voxNr];
      ys = allYS[voxNr];
      edgeLen = allEdgeLen[voxNr];
      ud = allUD[voxNr];
      double eulThis_local[3];
      eulThis_local[0] = allEul[voxNr * 3 + 0];
      eulThis_local[1] = allEul[voxNr * 3 + 1];
      eulThis_local[2] = allEul[voxNr * 3 + 2];
      gs = edgeLen / 2;
      dy1 = edgeLen / sqrt(3);
      dy2 = -edgeLen / (2 * sqrt(3));
      if (ud < 0) {
        dy1 *= -1;
        dy2 *= -1;
      }
      int NrPixelsGrid = 2 * (ceil((gs * 2) / px)) * (ceil((gs * 2) / px));
      if (gs * 2 < px)
        NrPixelsGrid = 1;
      double XG_local[3], YG_local[3];
      XG_local[0] = xs;
      XG_local[1] = xs - gs;
      XG_local[2] = xs + gs;
      YG_local[0] = ys + dy1;
      YG_local[1] = ys + dy2;
      YG_local[2] = ys + dy2;
      double OMIn_local[3][3];
      Euler2OrientMat(eulThis_local, OMIn_local);
      int **InPixels_local = allocMatrixIntF(NrPixelsGrid, 2);
      double *TheorSpots_local =
          malloc(MAX_N_SPOTS * 3 * sizeof(*TheorSpots_local));
      SimulateAccOrientBuffered(
          nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots, XG_local,
          YG_local, RotMatTilts, OmegaStart, OmegaStep, px, ybc, zbc, gs, hkls,
          n_hkls, Thetas, OmegaRanges, NoOfOmegaRanges, BoxSizes, P0,
          NrPixelsGrid, OMIn_local, TheorSpots_local, voxNr, spF,
          InPixels_local, Gs, NrPixelsY, NrPixelsZ, &hitBufs[tid]);
      FreeMemMatrixInt(InPixels_local, NrPixelsGrid);
      free(TheorSpots_local);
    }
    printf("Phase 1 (compute spots): %.2f sec\n",
           omp_get_wtime() - phase1_start);

    /* Merge per-thread buffers */
    totalHits = 0;
    for (i = 0; i < nCPUs; i++)
      totalHits += hitBufs[i].count;
    printf("Total pixel hits: %d (%.2f MB)\n", totalHits,
           (double)totalHits * sizeof(NFPixelHit) / (1024.0 * 1024.0));
    allHits = malloc((size_t)totalHits * sizeof(NFPixelHit));
    int offset = 0;
    for (i = 0; i < nCPUs; i++) {
      memcpy(allHits + offset, hitBufs[i].hits,
             hitBufs[i].count * sizeof(NFPixelHit));
      offset += hitBufs[i].count;
      free(hitBufs[i].hits);
    }
    free(hitBufs);
    hitBufs = NULL;

    /* Phase 2: replay hits into frame batches to build ObsSpotsInfo
       Then scan each batch to update bitArr (and optionally binArr) */
    int framesPerBatch = (nrFiles + nBatches - 1) / nBatches;
    printf("Replaying %d hits into %d batches (%d frames/batch)...\n",
           totalHits, nBatches, framesPerBatch);

    /* We'll build a temporary ObsSpotsInfo per batch, then
       just set the bitArr directly from the hits. Actually, we can
       skip the intermediate array entirely and just set bits directly
       from the hits — no batch array needed! */
  }
  free(allXS);
  free(allYS);
  free(allEdgeLen);
  free(allUD);
  free(allEul);
  printf("Writing output file\n");

  size_t nrF = 0;
  int *bitArr;
  bitArr = calloc((SizeObsSpots + 31) / 32, sizeof(*bitArr));

  if (nBatches == 1) {
    /* --- Single-pass post-processing (original code) --- */
    FILE *OutputF;
    if (skipBin == 0 && !OnlySpotsInfo) {
      OutputF = fopen(outputFN, "wb");
      if (OutputF == NULL) {
        printf("Could not write output file\n");
        return 1;
      }
      char dummychar[8192];
      memset(dummychar, 0, sizeof(dummychar));
      fwrite(dummychar, 8192, 1, OutputF);
      fwrite(ObsSpotsInfo, SizeObsSpots * sizeof(*ObsSpotsInfo), 1, OutputF);
      fclose(OutputF);
    }
    printf("Done with full file\n");
    size_t idxpos;
    if (OnlySpotsInfo) {
      idxpos = 0;
      for (l = 0; l < nLayers; l++) {
        for (k = 0; k < nrFiles; k++) {
          for (j = 0; j < NrPixelsZ; j++) {
            for (i = 0; i < NrPixelsY; i++) {
              if (ObsSpotsInfo[idxpos] != 0) {
                SetBit(bitArr, idxpos);
                nrF++;
              }
              idxpos++;
            }
          }
        }
      }
      printf("Total number of illuminated pixels: %zu\n", nrF);
    } else {
      size_t binArrCapacity = SizeObsSpots;
      idxpos = 0;
      for (l = 0; l < nLayers; l++) {
        for (k = 0; k < nrFiles; k++) {
          for (j = 0; j < NrPixelsZ; j++) {
            for (i = 0; i < NrPixelsY; i++) {
              if (ObsSpotsInfo[idxpos] != 0) {
                for (m = 0; m < ObsSpotsInfo[idxpos]; m++) {
                  if ((nrF + 1) * 5 > binArrCapacity) {
                    binArrCapacity *= 2;
                    binArr = realloc(binArr, binArrCapacity * sizeof(*binArr));
                    if (binArr == NULL) {
                      printf("Could not realloc binArr! Ran out of RAM?\n");
                      return 1;
                    }
                  }
                  binArr[nrF * 5 + 0] = j;
                  binArr[nrF * 5 + 1] = i;
                  binArr[nrF * 5 + 2] = k;
                  binArr[nrF * 5 + 3] = l;
                  binArr[nrF * 5 + 4] = ObsSpotsInfo[idxpos];
                  nrF++;
                }
                SetBit(bitArr, idxpos);
              }
              idxpos++;
            }
          }
        }
      }
      printf("Total number of illuminated pixels: %zu\n", nrF);
      FILE *OutputF2 = fopen(outFN, "wb");
      if (OutputF2 == NULL) {
        printf("Could not write output file\n");
        return 1;
      }
      fwrite(binArr, nrF * 5 * sizeof(*binArr), 1, OutputF2);
      fclose(OutputF2);
    }
    free(ObsSpotsInfo);
    ObsSpotsInfo = NULL;
  } else {
    /* --- Batched post-processing: set bits directly from merged hits --- */
    double phase2_start = omp_get_wtime();
    for (int h = 0; h < totalHits; h++) {
      NFPixelHit *hit = &allHits[h];
      long long int BinNr = (long long int)hit->layer * nrFiles;
      BinNr *= NrPixelsY;
      BinNr *= NrPixelsZ;
      long long int TempCntr = (long long int)hit->omeBin;
      TempCntr *= NrPixelsY;
      TempCntr *= NrPixelsZ;
      BinNr += TempCntr;
      TempCntr = (long long int)NrPixelsZ;
      TempCntr *= hit->multY;
      BinNr += TempCntr;
      BinNr += hit->multZ;
      if (BinNr >= 0 && BinNr < SizeObsSpots) {
        SetBit(bitArr, BinNr);
        nrF++;
      }
    }
    printf("Phase 2 (replay hits): %.2f sec, %zu illuminated pixels\n",
           omp_get_wtime() - phase2_start, nrF);
    free(allHits);
    allHits = NULL;
  }
  FILE *outputSpotsInfo = fopen("SpotsInfo.bin", "wb");
  if (outputSpotsInfo == NULL) {
    printf("Could not write output file\n");
    return 1;
  }
  fwrite(bitArr, (SizeObsSpots + 31) / 32 * sizeof(*bitArr), 1,
         outputSpotsInfo);
  fclose(outputSpotsInfo);

  // =========================================================================
  // Optional: Write Zarr/ZIP output
  // =========================================================================
  if (WriteImage && !OnlySpotsInfo) {
    printf("Writing Zarr/ZIP output...\n");
    blosc_init();
    blosc_set_nthreads(4);
    blosc_set_compressor("zstd");

    int errorp;
    char outZipFN[4096];
    sprintf(outZipFN, "%s.zip", outputFN);
    zip_t *zipper = zip_open(outZipFN, ZIP_CREATE | ZIP_TRUNCATE, &errorp);
    if (zipper == NULL) {
      printf("Could not open the zip file %s for writing. Exiting.\n",
             outZipFN);
      return 1;
    }

    // Write Zarr v2 metadata
    char outstr0[8192];
    sprintf(outstr0, "{\n    \"zarr_format\": 2\n}");
    char zarrfn0[8192];
    sprintf(zarrfn0, ".zgroup");
    int rcv0 = writeStrZip(outstr0, zarrfn0, zipper);
    if (rcv0 != 0)
      return 1;

    char zarrfn1[8192];
    sprintf(zarrfn1, "exchange/.zgroup");
    rcv0 = writeStrZip(outstr0, zarrfn1, zipper);
    if (rcv0 != 0)
      return 1;

    // .zarray: shape [nLayers, nrFiles, NrPixelsZ, NrPixelsY], chunks [1, 1,
    // NrPixelsZ, NrPixelsY]
    char outstr2[8192];
    sprintf(outstr2,
            "{\n    \"chunks\": [\n        1,\n        1,\n        %d,\n"
            "        %d\n    ],\n"
            "    \"compressor\": {\n        \"blocksize\": 0,\n"
            "        \"clevel\": 3,\n"
            "        \"cname\": \"zstd\",\n        \"id\": \"blosc\",\n"
            "        \"shuffle\": 2\n    },\n"
            "    \"dtype\": \"<u2\",\n    \"fill_value\": 0,\n"
            "    \"filters\": null,\n"
            "    \"order\": \"C\",\n    \"shape\": [\n        %d,\n"
            "        %d,\n        %d,\n        %d\n    ],\n"
            "    \"zarr_format\": 2\n}",
            NrPixelsZ, NrPixelsY, nLayers, nrFiles, NrPixelsZ, NrPixelsY);
    char zarrfn2[8192];
    sprintf(zarrfn2, "exchange/data/.zarray");
    rcv0 = writeStrZip(outstr2, zarrfn2, zipper);
    if (rcv0 != 0)
      return 1;

    // .zattrs
    char outstr3[8192];
    sprintf(outstr3,
            "{\n    \"_ARRAY_DIMENSIONS\": [\n        %d,\n        %d,\n"
            "        %d,\n        %d\n    ]\n}",
            nLayers, nrFiles, NrPixelsZ, NrPixelsY);
    char zarrfn3[8192];
    sprintf(zarrfn3, "exchange/data/.zattrs");
    rcv0 = writeStrZip(outstr3, zarrfn3, zipper);
    if (rcv0 != 0)
      return 1;

    // Write compressed chunks: one chunk per (layer, frame)
    size_t frameSize = (size_t)NrPixelsY * NrPixelsZ;
    size_t frameSizeBytes = frameSize * sizeof(uint16_t);
    // Temporary compression buffer (reused each iteration, then
    // data is copied to a per-chunk buffer for libzip ownership)
    uint16_t *compTmp = calloc(frameSize, sizeof(uint16_t));
    if (compTmp == NULL) {
      printf("Could not allocate compression buffer. Exiting.\n");
      return 1;
    }

    for (l = 0; l < nLayers; l++) {
      for (k = 0; k < nrFiles; k++) {
        // Extract this frame from ObsSpotsInfo
        // ObsSpotsInfo layout: [layer][frame][row][col]
        size_t offset = (size_t)l * nrFiles * frameSize + (size_t)k * frameSize;
        uint16_t *frameData = &ObsSpotsInfo[offset];

        // Compress with blosc into temporary buffer
        int compressedSize =
            blosc_compress(3, 2, sizeof(uint16_t), frameSizeBytes, frameData,
                           compTmp, frameSizeBytes);
        if (compressedSize <= 0) {
          printf("Blosc compression failed for layer %d frame %d. Exiting.\n",
                 l, k);
          free(compTmp);
          return 1;
        }

        // Allocate a per-chunk buffer for zip (libzip takes ownership via
        // freefunc=1 and will free() it after zip_close)
        void *chunkData = malloc(compressedSize);
        if (chunkData == NULL) {
          printf("Could not allocate chunk buffer. Exiting.\n");
          free(compTmp);
          return 1;
        }
        memcpy(chunkData, compTmp, compressedSize);

        // Write to zip
        char chunkfn[8192];
        sprintf(chunkfn, "exchange/data/%d.%d.0.0", l, k);
        zip_source_t *source =
            zip_source_buffer(zipper, chunkData, compressedSize, 1);
        zip_int64_t rct = zip_file_add(zipper, chunkfn, source,
                                       ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
        if (rct < 0) {
          printf("Could not add %s to zip. Exiting.\n", chunkfn);
          free(compTmp);
          return 1;
        }
        int rc = zip_set_file_compression(zipper, rct, ZIP_CM_STORE, 0);
        if (rc != 0) {
          printf("Could not set compression for %s. Exiting.\n", chunkfn);
          free(compTmp);
          return 1;
        }
      }
    }
    free(compTmp);
    int zc = zip_close(zipper);
    if (zc != 0) {
      printf("Error closing zip file. Exiting.\n");
      return 1;
    }
    blosc_destroy();
    printf("Zarr/ZIP output written to: %s\n", outZipFN);
  }

  free(ObsSpotsInfo);
  if (!OnlySpotsInfo)
    free(binArr);
  free(bitArr);
  end_time = omp_get_wtime();
  diftotal = end_time - start_time;
  printf("Total time elapsed: %f [s]\n", diftotal);
  return 0;
}