//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "nf_headers.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
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

#define RealType double
#define float32_t float
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;
int SpaceGrp;

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
      return NULL;
    }
  }
  return arr;
}

struct my_func_data {
  int NrOfFiles;
  int nLayers;
  double ExcludePoleAngle;
  long long int SizeObsSpots;
  double *XGrain;
  double *YGrain;
  double OmegaStart;
  double OmegaStep;
  double px;
  double gs;
  double (*hkls)[4];
  int n_hkls;
  double *Thetas;
  int NoOfOmegaRanges;
  int NrPixelsGrid;
  double (*OmegaRanges)[2];
  double (*BoxSizes)[4];
  int *ObsSpotsInfo;
  double *Lsd;
  double (*RotMatTilts)[3];
  double *ybc;
  double *zbc;
  int **InPixels;
  double *TheorSpots;
  double *P0Flat;
  double *Gs;
  int NrPixelsY;
  int NrPixelsZ;
};

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  int i, j;
  const int NrOfFiles = f_data->NrOfFiles;
  const int nLayers = f_data->nLayers;
  const double ExcludePoleAngle = f_data->ExcludePoleAngle;
  const long long int SizeObsSpots = f_data->SizeObsSpots;
  const double OmegaStart = f_data->OmegaStart;
  const double OmegaStep = f_data->OmegaStep;
  const double px = f_data->px;
  const double gs = f_data->gs;
  const int NoOfOmegaRanges = f_data->NoOfOmegaRanges;
  const int NrPixelsGrid = f_data->NrPixelsGrid;
  int n_hkls = f_data->n_hkls;
  int *ObsSpotsInfo = f_data->ObsSpotsInfo;
  double *Lsd = f_data->Lsd;
  double *ybc = f_data->ybc;
  double *zbc = f_data->zbc;
  double *XGrain = f_data->XGrain;
  double *YGrain = f_data->YGrain;
  double (*P0)[3] = (double (*)[3])f_data->P0Flat;

  double (*OmegaRanges)[2] = f_data->OmegaRanges;
  double (*BoxSizes)[4] = f_data->BoxSizes;
  double (*RotMatTilts)[3] = f_data->RotMatTilts;
  double (*hkls)[4] = f_data->hkls;
  double *Thetas = f_data->Thetas;
  double *TheorSpots = f_data->TheorSpots;
  double *Gs = f_data->Gs;

  double OrientMatIn[3][3], FracOverlap, x2[3];
  x2[0] = x[0] * rad2deg;
  x2[1] = x[1] * rad2deg;
  x2[2] = x[2] * rad2deg;
  Euler2OrientMat(x2, OrientMatIn);
  CalcOverlapAccOrient(
      NrOfFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots, XGrain, YGrain,
      RotMatTilts, OmegaStart, OmegaStep, px, ybc, zbc, gs, hkls, n_hkls,
      Thetas, OmegaRanges, NoOfOmegaRanges, BoxSizes, P0, NrPixelsGrid,
      ObsSpotsInfo, OrientMatIn, &FracOverlap, TheorSpots, f_data->InPixels, Gs,
      f_data->NrPixelsY, f_data->NrPixelsZ);
  return (1 - FracOverlap);
}

void FitOrientation(
    const int NrOfFiles, const int nLayers, const double ExcludePoleAngle,
    double Lsd[nLayers], const long long int SizeObsSpots,
    const double XGrain[3], const double YGrain[3], double RotMatTilts[3][3],
    const double OmegaStart, const double OmegaStep, const double px,
    double ybc[nLayers], double zbc[nLayers], const double gs,
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4], double P0[nLayers][3],
    const int NrPixelsGrid, int *ObsSpotsInfo, double EulerIn[3], double tol,
    double *EulerOutA, double *EulerOutB, double *EulerOutC,
    double *ResultFracOverlap, double hkls[5000][4], double Thetas[5000],
    int n_hkls, double *Gs, int *out_nevals, int *out_retcode, int NrPixelsY,
    int NrPixelsZ) {
  unsigned n;
  long int i, j;
  n = 3;
  double x[n], xl[n], xu[n];
  for (i = 0; i < n; i++) {
    x[i] = EulerIn[i];
    xl[i] = x[i] - (tol * M_PI / 180);
    xu[i] = x[i] + (tol * M_PI / 180);
  }
  struct my_func_data f_data;
  f_data.NrOfFiles = NrOfFiles;
  f_data.nLayers = nLayers;
  f_data.n_hkls = n_hkls;
  f_data.hkls = hkls;
  f_data.Thetas = Thetas;
  f_data.ExcludePoleAngle = ExcludePoleAngle;
  f_data.SizeObsSpots = SizeObsSpots;

  // Allocate P0Flat for use in problem_function
  f_data.P0Flat = malloc(nLayers * 3 * sizeof(double));
  for (i = 0; i < nLayers; i++) {
    // We do not strictly need f_data.P0 to point to valid memory for
    // problem_function anymore since we use P0Flat. However, f_data.P0 was used
    // to COPY from the input P0 argument. Let's copy directly from input P0 to
    // P0Flat.
    f_data.P0Flat[i * 3 + 0] = P0[i][0];
    f_data.P0Flat[i * 3 + 1] = P0[i][1];
    f_data.P0Flat[i * 3 + 2] = P0[i][2];
  }

  // Cast away const for struct assignment, we promise not to modify these in
  // problem_function
  f_data.XGrain = (double *)XGrain;
  f_data.YGrain = (double *)YGrain;
  f_data.RotMatTilts = RotMatTilts;
  f_data.OmegaRanges = OmegaRanges;
  f_data.BoxSizes = BoxSizes;

  f_data.ObsSpotsInfo = &ObsSpotsInfo[0];
  f_data.Lsd = &Lsd[0];
  f_data.ybc = &ybc[0];
  f_data.zbc = &zbc[0];
  f_data.OmegaStart = OmegaStart;
  f_data.OmegaStep = OmegaStep;
  f_data.px = px;
  f_data.gs = gs;
  f_data.NoOfOmegaRanges = NoOfOmegaRanges;
  f_data.NrPixelsGrid = NrPixelsGrid;
  f_data.InPixels = allocMatrixIntF(NrPixelsGrid, 2);
  f_data.TheorSpots = malloc(MAX_N_SPOTS * 3 * sizeof(double));
  f_data.Gs = Gs;
  f_data.NrPixelsY = NrPixelsY;
  f_data.NrPixelsZ = NrPixelsZ;
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;
  double tole = 1e-5;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function, trp);
  nlopt_set_maxeval(opt, 5000);
  nlopt_set_maxtime(opt, 30);
  nlopt_set_ftol_rel(opt, 1e-5);
  nlopt_set_xtol_rel(opt, 1e-5);
  double minf = 1;
  int retcode = nlopt_optimize(opt, x, &minf);
  *out_nevals = (int)nlopt_get_numevals(opt);
  *out_retcode = retcode;
  nlopt_destroy(opt);
  // f_data.P0 was allocated with malloc for the pointer array only, but we
  // didn't alloc rows
  free(f_data.P0Flat);

  free(f_data.TheorSpots);
  FreeMemMatrixInt(f_data.InPixels, NrPixelsGrid);
  *ResultFracOverlap = minf;
  *EulerOutA = x[0];
  *EulerOutB = x[1];
  *EulerOutC = x[2];
}

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

static inline void QuatToOrientMat(double Quat[4], double OrientMat[3][3]) {
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
  OrientMat[0][0] = 1 - 2 * (Q2_2 + Q3_2);
  OrientMat[0][1] = 2 * (Q12 - Q03);
  OrientMat[0][2] = 2 * (Q13 + Q02);
  OrientMat[1][0] = 2 * (Q12 + Q03);
  OrientMat[1][1] = 1 - 2 * (Q1_2 + Q3_2);
  OrientMat[1][2] = 2 * (Q23 - Q01);
  OrientMat[2][0] = 2 * (Q13 - Q02);
  OrientMat[2][1] = 2 * (Q23 + Q01);
  OrientMat[2][2] = 1 - 2 * (Q1_2 + Q2_2);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage:\n FitOrientation params.txt blockNr nBlocks nCPUs\n");
    return 1;
  }
  double start_time = omp_get_wtime();
  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
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
      LatticeConstant[6], minFracOverlap, doubledummy, MaxRingRad, MaxTtheta;
  double px, OmegaStart, OmegaStep, tol;
  char fn[1000], MicFN[1000];
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
  int NrPixelsY = 2048, NrPixelsZ = 2048;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
      continue;
    }
    str = "MicFileBinary ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, MicFN);
      continue;
    }
    str = "GridFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, gridfn);
      gridfnfound = 1;
      continue;
    }
    str = "SaveNSolutions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nSaves);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
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
    str = "OrientTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tol);
      continue;
    }
    str = "MinFracAccept ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &minFracOverlap);
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
    str = "Ice9Input ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      Flag = 1;
      continue;
    }
    str = "NearestMisorientation ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MinMiso);
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
  }
  int it, jt, mt, nrFiles, nrPixels;
  for (it = 0; it < NoOfOmegaRanges; it++) {
    OmegaRang[it][0] = OmegaRanges[it][0];
    OmegaRang[it][1] = OmegaRanges[it][1];
  }
  nOmeRang = NoOfOmegaRanges;
  fclose(fileParam);
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd[0]);
  // Read bin files
  char fnG[1000];
  if (gridfnfound == 1)
    sprintf(fnG, "%s/%s", direct, gridfn);
  else
    sprintf(fnG, "%s/grid.txt", direct);
  char fnDS[1000];
  char fnKey[1000];
  char fnOr[1000];
  sprintf(fnDS, "%s/DiffractionSpots.txt", direct);
  sprintf(fnKey, "%s/Key.txt", direct);
  sprintf(fnOr, "%s/OrientMat.txt", direct);
  char *ext = "bin";
  int *ObsSpotsInfo;
  nrFiles = EndNr - StartNr + 1;
  nrPixels = NrPixelsY * NrPixelsZ;
  long long int SizeObsSpots;
  SizeObsSpots = (nLayers);
  SizeObsSpots *= nrPixels;
  SizeObsSpots *= nrFiles;
  SizeObsSpots /= 32;
  // printf("%lld\n",SizeObsSpots*32);

  // Read spots info
  char cmmd[4096];
  sprintf(cmmd, "cp SpotsInfo.bin /dev/shm/");
  system(cmmd);
  char *file_name = "/dev/shm/SpotsInfo.bin";
  int descp;
  struct stat s;
  int status;
  size_t size;
  int rc;
  descp = open(file_name, O_RDONLY);
  check(descp < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(descp, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  ObsSpotsInfo = mmap(0, size, PROT_READ, MAP_SHARED, descp, 0);
  check(ObsSpotsInfo == MAP_FAILED, "mmap %s failed: %s", file_name,
        strerror(errno));

  // Read DiffractionSpots
  double *SpotsMat;
  sprintf(cmmd, "cp DiffractionSpots.bin /dev/shm/");
  system(cmmd);
  char *spfn = "/dev/shm/DiffractionSpots.bin";
  int spf;
  struct stat s2;
  int status2;
  size_t size2;
  int rc2;
  spf = open(spfn, O_RDONLY);
  check(spf < 0, "open %s failed: %s", spfn, strerror(errno));
  status2 = fstat(spf, &s2);
  check(status2 < 0, "stat %s failed: %s", spfn, strerror(errno));
  size2 = s2.st_size;
  SpotsMat = mmap(0, size2, PROT_READ, MAP_SHARED, spf, 0);
  check(SpotsMat == MAP_FAILED, "mmap %s failed: %s", spfn, strerror(errno));

  // Read OrientationMatrix
  double *OrientationMatrix;
  sprintf(cmmd, "cp OrientMat.bin /dev/shm/");
  system(cmmd);
  char *omfn = "/dev/shm/OrientMat.bin";
  int omf;
  struct stat s3;
  int status3;
  size_t size3;
  int rc3;
  omf = open(omfn, O_RDONLY);
  check(omf < 0, "open %s failed: %s", omfn, strerror(errno));
  status3 = fstat(omf, &s3);
  check(status3 < 0, "stat %s failed: %s", omfn, strerror(errno));
  size3 = s3.st_size;
  OrientationMatrix = mmap(0, size3, PROT_READ, MAP_SHARED, omf, 0);
  check(OrientationMatrix == MAP_FAILED, "mmap %s failed: %s", omfn,
        strerror(errno));

  // Read Key
  char line[1024];
  clock_t startthis;
  startthis = clock();
  FILE *fk;
  int NrOrientations, TotalDiffrSpots;
  fk = fopen(fnKey, "r");
  fgets(line, 1000, fk);
  sscanf(line, "%d", &NrOrientations);
  int **NrSpots;
  NrSpots = allocMatrixIntF(NrOrientations, 2);
  TotalDiffrSpots = 0;
  for (it = 0; it < NrOrientations; it++) {
    fgets(line, 1000, fk);
    sscanf(line, "%d", &NrSpots[it][0]);
    TotalDiffrSpots += NrSpots[it][0];
    NrSpots[it][1] = TotalDiffrSpots - NrSpots[it][0];
  }
  fclose(fk);

  // Read position.
  FILE *fp;
  fp = fopen(fnG, "r");
  fgets(line, 1000, fp);
  int TotalNrSpots = 0;
  sscanf(line, "%d", &TotalNrSpots);
  int nBlocks = atoi(argv[3]);
  int blockNr = atoi(argv[2]);
  int numProcs = atoi(argv[4]);
  int rown;
  //~ int multF = 1 + (int)floor((TotalNrSpots/2000));
  //~ int startRowNr = multF*(atoi(argv[2])-1) + 1;
  //~ int endRowNr = multF*(atoi(argv[2]));
  int startRowNr, endRowNr;
  startRowNr = (int)(ceil((double)TotalNrSpots / (double)nBlocks)) * blockNr;
  int tmp = (int)(ceil((double)TotalNrSpots / (double)nBlocks)) * (blockNr + 1);
  endRowNr = tmp < (TotalNrSpots - 1) ? tmp : (TotalNrSpots - 1);
  int nrows = endRowNr - startRowNr + 1;
  int counter = 0;
  while (counter < startRowNr - 1) {
    fgets(line, 1000, fp);
    counter += 1;
  }
  char **lines_raw; // Renamed to avoid conflict with parsed_lines
  lines_raw = malloc(nrows * sizeof(*lines_raw));
  printf("%d %d %d\n", startRowNr, endRowNr, nrows);
  //~ lines = malloc(nrows*sizeof(*lines));
  //~ lines[0] = malloc(nrows*1000*sizeof(**lines));
  //~ for (it=1;it<nrows;it++) lines[it] = lines[0] + it*1000;
  for (it = 0; it < nrows; it++) {
    fgets(line, 1000, fp);
    lines_raw[it] = strndup(line, 1000);
  }
  //~ for (it=0;it<nrows;it++) fgets(lines[it],1000,fp);
  fclose(fp);

  int n_hkls = 0, iter_i, iter_j;
  double hkls[5000][4];
  double Thetas[5000];
  char aliner[1000];
  char hklfn[1024];
  sprintf(hklfn, "%s/hkls.csv", direct);
  FILE *hklf = fopen(hklfn, "r");
  fgets(aliner, 1000, hklf);
  while (fgets(aliner, 1000, hklf) != NULL) {
    sscanf(aliner, "%s %s %s %s %lf %lf %lf %lf %lf %s %s", dummy, dummy, dummy,
           dummy, &hkls[n_hkls][3], &hkls[n_hkls][0], &hkls[n_hkls][1],
           &hkls[n_hkls][2], &Thetas[n_hkls], dummy, dummy);
    n_hkls++;
  }
  fclose(hklf);
  if (nRingsToUse > 0) {
    double hklTemps[n_hkls][4], thetaTemps[n_hkls];
    int totalHKLs = 0;
    for (iter_i = 0; iter_i < nRingsToUse; iter_i++) {
      for (iter_j = 0; iter_j < n_hkls; iter_j++) {
        if ((int)hkls[iter_j][3] == RingsToUse[iter_i]) {
          hklTemps[totalHKLs][0] = hkls[iter_j][0];
          hklTemps[totalHKLs][1] = hkls[iter_j][1];
          hklTemps[totalHKLs][2] = hkls[iter_j][2];
          hklTemps[totalHKLs][3] = hkls[iter_j][3];
          thetaTemps[totalHKLs] = Thetas[iter_j];
          totalHKLs++;
        }
      }
    }
    for (iter_i = 0; iter_i < totalHKLs; iter_i++) {
      hkls[iter_i][0] = hklTemps[iter_i][0];
      hkls[iter_i][1] = hklTemps[iter_i][1];
      hkls[iter_i][2] = hklTemps[iter_i][2];
      hkls[iter_i][3] = hklTemps[iter_i][3];
      Thetas[iter_i] = thetaTemps[iter_i];
    }
    n_hkls = totalHKLs;
  }

  // Precompute Gs for CalcDiffractionSpots optimization
  double *Gs;
  Gs = malloc(n_hkls * sizeof(double));
  int i;
  for (i = 0; i < n_hkls; i++) {
    double Ghkl[3];
    Ghkl[0] = hkls[i][0];
    Ghkl[1] = hkls[i][1];
    Ghkl[2] = hkls[i][2];
    double len =
        sqrt(Ghkl[0] * Ghkl[0] + Ghkl[1] * Ghkl[1] + Ghkl[2] * Ghkl[2]);
    Gs[i] =
        sin(Thetas[i] * M_PI / 180.0) * len; // Assuming deg2rad is M_PI/180.0
  }

  // Parse input lines for sscanf hoisting
  struct ParsedLine {
    double y1, y2, xs, ys, gs;
    int valid;
  };
  struct ParsedLine *parsed_lines =
      malloc((endRowNr - startRowNr + 1) * sizeof(struct ParsedLine));
  for (rown = startRowNr; rown <= endRowNr; rown++) {
    int idx = rown - startRowNr;
    struct ParsedLine *pl = &parsed_lines[idx];
    if (rown > TotalNrSpots) {
      pl->valid = 0;
    } else {
      int nparsed = sscanf(lines_raw[idx], "%lf %lf %lf %lf %lf", &pl->y1,
                           &pl->y2, &pl->xs, &pl->ys, &pl->gs);
      if (nparsed != 5) {
        pl->valid = 0;
      } else {
        pl->valid = 1;
      }
    }
  }
  // Free raw lines after parsing
  for (it = 0; it < nrows; it++) {
    free(lines_raw[it]);
  }
  free(lines_raw);

  double RotMatTilts[3][3];
  RotationTilts(tx, ty, tz, RotMatTilts);
  double *OrientMatrixAll;
  OrientMatrixAll =
      calloc(MAX_POINTS_GRID_GOOD * 10 * numProcs, sizeof(*OrientMatrixAll));
  double *ThrSpsAll;
  ThrSpsAll = calloc(numProcs * MAX_N_SPOTS * 3, sizeof(*ThrSpsAll));
  printf("Number of individual diffracting planes: %d\n", n_hkls);

  //~ double *OutResultAll, *ResultMatrAll;
  //~ size_t numJobs = endRowNr - startRowNr + 1;
  //~ OutResultAll = calloc(numJobs*11,sizeof(*OutResultAll));
  //~ ResultMatrAll = calloc(numJobs*(7+(nSaves*4)),sizeof(*ResultMatrAll));

  // Open files for writing (Create/Open once)
  int result = open(MicFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (result <= 0) {
    printf("Could not open output file %s.\n", MicFN);
    exit(1);
  }
  char outfn2[4096];
  sprintf(outfn2, "%s.AllMatches", MicFN);
  int result2 = open(outfn2, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (result2 <= 0) {
    printf("Could not successfully open output file for all matches %s.\n",
           outfn2);
    exit(1);
  }

// DO OMP HERE??????
#pragma omp parallel for num_threads(numProcs) private(rown) schedule(dynamic)
  for (rown = startRowNr; rown <= endRowNr; rown++) {
    //~ clock_t start, end;
    //~ double diftotal;
    //~ start = clock();
    int procNum = omp_get_thread_num();
    int i, j, k, m;
    int idx = rown - startRowNr;
    // Use pre-parsed data
    if (!parsed_lines[idx].valid) {
      printf("Error: Grid point number greater than total number of grid "
             "points or parse error.\n");
      continue;
    }
    double y1 = parsed_lines[idx].y1;
    double y2 = parsed_lines[idx].y2;
    double xs = parsed_lines[idx].xs;
    double ys = parsed_lines[idx].ys;
    double gs = parsed_lines[idx].gs;

    // Alloc array and clear it.
    double XY[3][3];
    int UD;
    if (y1 > y2) {
      UD = -1;
      XY[0][0] = xs;
      XY[0][1] = ys - y1;
      XY[1][0] = xs - gs;
      XY[1][1] = ys + y2;
      XY[2][0] = xs + gs;
      XY[2][1] = ys + y2;
    } else {
      UD = 1;
      XY[0][0] = xs;
      XY[0][1] = ys + y2;
      XY[1][0] = xs - gs;
      XY[1][1] = ys - y1;
      XY[2][0] = xs + gs;
      XY[2][1] = ys - y1;
    }
    double GridSize = 2 * gs;
    // Go through each orientation and compare with observed spots.
    int NrPixelsGrid = 2 * (ceil((gs * 2) / px)) * (ceil((gs * 2) / px));
    int NrSpotsThis, StartingRowNr;
    double FracOverT;
    double OrientationMatThis[9], OrientationMatThisUnNorm[9];
    int OrientationGoodID = 0;
    double MatIn[3], P0[nLayers][3], P0T[3];
    double OrientMatIn[3][3], XG[3], YG[3];
    double *ThrSps;
    ThrSps = &ThrSpsAll[procNum * MAX_N_SPOTS * 3];
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
    for (j = 0; j < 3; j++) {
      XG[j] = XY[j][0];
      YG[j] = XY[j][1];
    }
    double *OrientMatrix;
    OrientMatrix = &OrientMatrixAll[MAX_POINTS_GRID_GOOD * 10 * procNum];
    //~ printf("Checking orientation grid.\n");
    // Hoisted InPixels allocation
    int **InPixels;
    InPixels = allocMatrixIntF(NrPixelsGrid, 2);
    for (i = 0; i < NrOrientations; i++) {
      NrSpotsThis = NrSpots[i][0];
      StartingRowNr = NrSpots[i][1];
      m = 0;
      for (m = 0; m < 9; m++) {
        OrientationMatThisUnNorm[m] = OrientationMatrix[i * 9 + m];
        if (OrientationMatThisUnNorm[m] == -0.0) {
          OrientationMatThisUnNorm[m] = 0;
        }
      }
      m = 0;
      NormalizeMat(OrientationMatThisUnNorm, OrientationMatThis);
      for (j = StartingRowNr; j < (StartingRowNr + NrSpotsThis); j++) {
        ThrSps[m * 3 + 0] = SpotsMat[j * 3 + 0];
        ThrSps[m * 3 + 1] = SpotsMat[j * 3 + 1];
        ThrSps[m * 3 + 2] = SpotsMat[j * 3 + 2];
        m++;
      }
      Convert9To3x3(OrientationMatThis, OrientMatIn);
      // Hoisted InPixels allocation to before loop
      CalcFracOverlap(nrFiles, nLayers, NrSpotsThis, ThrSps, OmegaStart,
                      OmegaStep, XG, YG, Lsd, SizeObsSpots, RotMatTilts, px,
                      ybc, zbc, gs, P0, NrPixelsGrid, ObsSpotsInfo, OrientMatIn,
                      &FracOverT, InPixels, NrPixelsY, NrPixelsZ);
      if (FracOverT >= minFracOverlap) {
        for (j = 0; j < 9; j++) {
          OrientMatrix[OrientationGoodID * 10 + j] = OrientationMatThis[j];
        }
        OrientMatrix[OrientationGoodID * 10 + 9] = (double)
            i; // Store the row nr in OrientationsList for grainID determination
        OrientationGoodID++;
      }
    }
    FreeMemMatrixInt(InPixels, NrPixelsGrid);
    // printf("Finished checking orientation grid for point %d. Now fitting %d"
    //        " orientations.\n",
    //        rown, OrientationGoodID);
    fflush(stdout);
    double tFitStart = omp_get_wtime();
    int totalNloptEvals = 0;
    double BestFrac, BestEuler[3];
    double ResultMatr[7 + (nSaves * 4)], QuatIn[4], QuatOut[4];
    double bestRowNr = 0;
    if (OrientationGoodID > 0) {
      double Fractions, EulerIn[3], OrientIn[3][3], FracOut, EulerOutA,
          EulerOutB, EulerOutC, OMTemp[9];
      BestFrac = -1;
      ResultMatr[0] = (double)atoi(argv[2]);
      ResultMatr[1] = (double)OrientationGoodID;
      ResultMatr[2] = 0;
      ResultMatr[3] = xs;
      ResultMatr[4] = ys;
      ResultMatr[5] = GridSize;
      ResultMatr[6] = (double)UD;
      int nFilled = 0;
      int t;
      for (i = 0; i < nSaves; i++) {
        ResultMatr[7 + i * 4] = 0;
        ResultMatr[7 + i * 4 + 1] = 0;
        ResultMatr[7 + i * 4 + 2] = 0;
        ResultMatr[7 + i * 4 + 3] = 0;
      }
      int firstSol = 0, UpdSol = 0;
      for (i = 0; i < OrientationGoodID; i++) {
        for (j = 0; j < 9; j++)
          OMTemp[j] = OrientMatrix[i * 10 + j];
        Convert9To3x3(OMTemp, OrientIn);
        OrientMat2Euler(OrientIn, EulerIn);
        // Note: FitOrientation internally calls optimization which calls
        // problem_function. problem_function in SharedFuncsFit.c needs to
        // handle InPixels if called there. Wait, FitOrientation calls nlopt
        // which calls problem_function. problem_function needs to allocate
        // InPixels internally or be passed it? Let's check problem_function in
        // SharedFuncsFit.c first.
        int fitNevals = 0, fitRetcode = 0;
        FitOrientation(nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots,
                       XG, YG, RotMatTilts, OmegaStart, OmegaStep, px, ybc, zbc,
                       gs, OmegaRanges, nOmeRang, BoxSizes, P0, NrPixelsGrid,
                       ObsSpotsInfo, EulerIn, tol, &EulerOutA, &EulerOutB,
                       &EulerOutC, &FracOut, hkls, Thetas, n_hkls, Gs,
                       &fitNevals, &fitRetcode, NrPixelsY, NrPixelsZ);
        totalNloptEvals += fitNevals;
        // if (i > 0 && i % 100 == 0) {
        //   printf("  Point %d: fitted %d/%d orientations, elapsed %.1fs, "
        //          "last nlopt evals=%d ret=%d\n",
        //          rown, i, OrientationGoodID, omp_get_wtime() - tFitStart,
        //          fitNevals, fitRetcode);
        //   fflush(stdout);
        // }
        Fractions = 1 - FracOut;
        if (Fractions >= BestFrac) {
          bestRowNr = OrientMatrix[i * 10 + 9]; // Save best RowNr
          BestFrac = Fractions;
          BestEuler[0] = EulerOutA;
          BestEuler[1] = EulerOutB;
          BestEuler[2] = EulerOutC;
          if (1 - BestFrac < 0.0001 && nSaves == 1)
            break;
        }
        if (nSaves > 1) {
          if (firstSol == 0) { // Put initial first solution in!!!
            ResultMatr[7 + 0] = EulerOutA;
            ResultMatr[7 + 1] = EulerOutB;
            ResultMatr[7 + 2] = EulerOutC;
            ResultMatr[7 + 3] = Fractions;
            firstSol = 1;
          } else {
            for (j = 0; j < nFilled;
                 j++) { // ResultMatr format: 7 initial common things, then 4
                        // values for each match, Angle, Angle, Angle and
                        // Confidence
              if (Fractions >= ResultMatr[7 + j * 4 + 3]) { // Put this solution
                                                            // in the array.
                for (m = nFilled - 1; m >= j; m--) { // Move everything upto j
                  if (m == nSaves - 1)
                    continue; // Worst match, trash
                  for (t = 0; t < 4; t++) {
                    ResultMatr[7 + (m + 1) * 4 + t] =
                        ResultMatr[7 + (m) * 4 + t];
                  }
                }
                ResultMatr[7 + j * 4] = EulerOutA;
                ResultMatr[7 + j * 4 + 1] = EulerOutB;
                ResultMatr[7 + j * 4 + 2] = EulerOutC;
                ResultMatr[7 + j * 4 + 3] = Fractions;
              }
            }
          }
          if (nFilled < nSaves)
            nFilled++;
        }
      }
    } else {
      printf("No good ID found.\n");
      continue;
    }
    double tFitElapsed = omp_get_wtime() - tFitStart;
    // printf("Point %d: fitting done in %.2fs, %d total nlopt evals, "
    //        "bestFrac=%.4f\n",
    //        rown, tFitElapsed, totalNloptEvals, BestFrac);
    // fflush(stdout);
    double outresult[11] = {bestRowNr,    (double)OrientationGoodID,
                            tFitElapsed,  xs,
                            ys,           GridSize,
                            (double)UD,   BestEuler[0],
                            BestEuler[1], BestEuler[2],
                            BestFrac};
    int SizeWritten = 11 * sizeof(double);
    size_t OffsetHere = (rown);
    OffsetHere *= SizeWritten;
    int SizeWritten2 = (7 + (nSaves * 4)) * sizeof(double);
    size_t OffsetThis = (rown);
    OffsetThis *= SizeWritten2;

    // Write files (Thread-safe pwrite)
    int rc4 = pwrite(result, outresult, SizeWritten, OffsetHere);
    if (rc4 < 0) {
#pragma omp critical
      printf("Could not write to output file %zu %d %d %d.\n", OffsetHere, rown,
             startRowNr, endRowNr);
    } else {
// printf is thread-safe but output might be interleaved.
// Using critical for clean output if desired, or removing for speed.
// Keeping critical for now as user expects feedback.
#pragma omp critical
      {
        printf("%zu %d ", OffsetHere, rown);
        for (i = 0; i < 11; i++) {
          printf("%.5f ", outresult[i]);
        }
        printf("\n");
        fflush(stdout);
      }
    }
    int rc5 = pwrite(result2, ResultMatr, SizeWritten2, OffsetThis);
    if (rc5 < 0) {
#pragma omp critical
      printf("Could not write all matches %zu %d %d %d.\n", OffsetThis, rown,
             startRowNr, endRowNr);
    }

    //~ printf("Time elapsed in comparing diffraction spots: %f
    //[s]\n",diftotal); ~ for (i=0;i<MAX_POINTS_GRID_GOOD*10;i++)
    // OrientMatrix[i] = 0; // Maybe not needed.
  }

  // Close files after loop
  free(parsed_lines);
  free(Gs);
  close(result);
  close(result2);
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return 0;
}