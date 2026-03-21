//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
#include "MIDAS_Math.h"
#include "midas_version.h"
#include "nf_headers.h"
#include "nf_gpu.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdarg.h>
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
  int debugTrace;  // >0 = print first N evals
  int evalCount;
};

// Global debug flag: when >0, next FitOrientation call enables trace
static int g_debugNextFit = 0;

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
  // Debug trace: print evals when trace is enabled
  if (f_data->debugTrace > 0 && f_data->evalCount < 10) {
    printf("CPU-NM eval %d: euler_rad=(%.9f,%.9f,%.9f) euler_deg=(%.6f,%.6f,%.6f) frac=%.6f obj=%.6f\n",
           f_data->evalCount, x[0], x[1], x[2], x2[0], x2[1], x2[2], FracOverlap, 1 - FracOverlap);
    f_data->evalCount++;
  }
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
  f_data.debugTrace = g_debugNextFit;
  f_data.evalCount = 0;
  if (g_debugNextFit) g_debugNextFit = 0;  // one-shot
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;
  double tole = 1e-5;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double minf = 1;
  int retcode = run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  minf = config.min_function_val;
  // Note: nlopt_get_numevals cannot be extracted cleanly without exposing opt,
  // so we just set a dummy value or what we asked for. Actually,
  // max_evaluations is what it took at most.
  *out_nevals = config.max_evaluations;
  *out_retcode = retcode;
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

// check() is now provided by MIDAS_Limits.h (via nf_headers.h)

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
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc != 5) {
    printf("Usage:\n FitOrientation params.txt blockNr nBlocks nCPUs\n");
    return 1;
  }
  double start_time = omp_get_wtime();
  // Verbosity: set MIDAS_VERBOSE=1 for per-voxel output
  int verbose = 0;
  {
    const char *v = getenv("MIDAS_VERBOSE");
    if (v && atoi(v) > 0) verbose = 1;
  }
  // Screen-only mode: set MIDAS_SCREEN_ONLY=1 to skip Phase 2 fitting
  int screen_only = 0;
  {
    const char *v = getenv("MIDAS_SCREEN_ONLY");
    if (v && atoi(v) > 0) screen_only = 1;
  }
  if (screen_only) printf("*** SCREEN-ONLY MODE: Phase 2 fitting will be skipped ***\n");
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
  char direct[1000] = ".";
  char outputDir[1000];
  outputDir[0] = '\0';
  char gridfn[1000];
  double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
  int cntr = 0, countr = 0, conter = 0, StartNr, EndNr, intdummy, SpaceGroup,
      RingsToUse[100], nRingsToUse = 0;
  int NoOfOmegaRanges = 0;
  int nSaves = 1;
  int gridfnfound = 0;
  Wedge = 0;
  int MinMiso = 0;
  double MinMisoNSaves = 1.0; // degrees, default
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
    str = "OutputDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, outputDir);
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
    str = "MinMisoNSaves ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MinMisoNSaves);
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

  // Format and print the parsed parameters for user diagnostic review
  printf("\n======================================================\n");
  printf("  FitOrientationOMP Parameter Summary\n");
  printf("======================================================\n");
  printf("  DataDirectory   : %s\n", direct);
  printf("  OutputDirectory : %s\n",
         outputDir[0] ? outputDir : "(same as DataDirectory)");
  printf("  MicFileBinary   : %s\n", MicFN);
  printf("  ReducedFileName : %s\n", fn2);
  printf("  GridFileName    : %s\n", gridfnfound ? gridfn : "grid.txt");
  printf("  nLayers         : %d\n", nLayers);
  int i_print;
  for (i_print = 0; i_print < nLayers; i_print++) {
    printf("    Layer %d      : Lsd=%.4f, BC=(%.4f, %.4f)\n", i_print,
           Lsd[i_print], ybc[i_print], zbc[i_print]);
  }
  printf("  StartNr-EndNr   : %d - %d\n", StartNr, EndNr);
  printf("  OmegaRange      : %.2f (step: %.2f)\n", OmegaStart, OmegaStep);
  printf("  Pixels (YxZ)    : %d x %d (px: %.4f)\n", NrPixelsY, NrPixelsZ, px);
  printf("  SpaceGroup      : %d\n", SpaceGroup);
  printf("  LatticeConstant : %.4f %.4f %.4f %.4f %.4f %.4f\n",
         LatticeConstant[0], LatticeConstant[1], LatticeConstant[2],
         LatticeConstant[3], LatticeConstant[4], LatticeConstant[5]);
  printf("  MaxRingRad      : %.4f\n", MaxRingRad);
  printf("  Wavelength      : %.4f\n", Wavelength);
  printf("  Wedge           : %.4f\n", Wedge);
  printf("  ExcludePoleAngle: %.4f\n", ExcludePoleAngle);
  printf("  Rot. Tilts      : tx:%.3f, ty:%.3f, tz:%.3f\n", tx, ty, tz);
  printf("  Optimization    : OrientTol:%.3f, MinFrac:%.3f, nSaves:%d\n", tol,
         minFracOverlap, nSaves);
  printf("  MinMisoNSaves   : %.3f deg\n", MinMisoNSaves);
  printf("  Ice9Input       : %s\n", Flag ? "Yes" : "No");
  printf("  NearestMiso     : %d\n", MinMiso);
  printf("  OmegaRanges (%d) : ", NoOfOmegaRanges);
  for (i_print = 0; i_print < NoOfOmegaRanges; i_print++)
    printf("[%.2f, %.2f] ", OmegaRanges[i_print][0], OmegaRanges[i_print][1]);
  printf("\n");
  printf("  BoxSizes (%d)    : ", countr);
  for (i_print = 0; i_print < countr; i_print++)
    printf("[%.2f, %.2f, %.2f, %.2f] ", BoxSizes[i_print][0],
           BoxSizes[i_print][1], BoxSizes[i_print][2], BoxSizes[i_print][3]);
  printf("\n");
  if (nRingsToUse > 0) {
    printf("  RingsToUse (%d)  : ", nRingsToUse);
    for (i_print = 0; i_print < nRingsToUse; i_print++)
      printf("%d ", RingsToUse[i_print]);
    printf("\n");
  }
  printf("======================================================\n\n");
  fflush(stdout);

  int it, jt, mt, nrFiles, nrPixels;
  for (it = 0; it < NoOfOmegaRanges; it++) {
    OmegaRang[it][0] = OmegaRanges[it][0];
    OmegaRang[it][1] = OmegaRanges[it][1];
  }
  nOmeRang = NoOfOmegaRanges;
  fclose(fileParam);
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd[0]);
  // Read bin files
  if (outputDir[0] == '\0')
    strcpy(outputDir, direct);
  char fnG[1000];
  if (gridfnfound == 1)
    sprintf(fnG, "%s/%s", outputDir, gridfn);
  else
    sprintf(fnG, "%s/grid.txt", outputDir);
  char fnKey[1000];
  sprintf(fnKey, "%s/Key.bin", outputDir);
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

  // Read spots info (mmap directly from DataDirectory, OS page cache keeps it
  // fast)
  char file_name[1024];
  sprintf(file_name, "%s/SpotsInfo.bin", outputDir);
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
  char spfn[1024];
  sprintf(spfn, "%s/DiffractionSpots.bin", outputDir);
  int spf;
  struct stat s2;
  int status2;
  spf = open(spfn, O_RDONLY);
  check(spf < 0, "open %s failed: %s", spfn, strerror(errno));
  status2 = fstat(spf, &s2);
  check(status2 < 0, "stat %s failed: %s", spfn, strerror(errno));
  size_t size2 = s2.st_size;
  SpotsMat = mmap(0, size2, PROT_READ, MAP_SHARED, spf, 0);
  check(SpotsMat == MAP_FAILED, "mmap %s failed: %s", spfn, strerror(errno));

  // Read OrientationMatrix
  double *OrientationMatrix;
  char omfn[1024];
  sprintf(omfn, "%s/OrientMat.bin", outputDir);
  int omf;
  struct stat s3;
  int status3;
  omf = open(omfn, O_RDONLY);
  check(omf < 0, "open %s failed: %s", omfn, strerror(errno));
  status3 = fstat(omf, &s3);
  check(status3 < 0, "stat %s failed: %s", omfn, strerror(errno));
  size_t size3 = s3.st_size;
  OrientationMatrix = mmap(0, size3, PROT_READ, MAP_SHARED, omf, 0);
  check(OrientationMatrix == MAP_FAILED, "mmap %s failed: %s", omfn,
        strerror(errno));

  // Read Key.bin
  char line[1024];
  clock_t startthis;
  startthis = clock();
  int NrOrientations, TotalDiffrSpots;
  int keyfd = open(fnKey, O_RDONLY);
  check(keyfd < 0, "open %s failed: %s", fnKey, strerror(errno));
  struct stat keyStat;
  fstat(keyfd, &keyStat);
  int *KeyData = mmap(0, keyStat.st_size, PROT_READ, MAP_SHARED, keyfd, 0);
  check(KeyData == MAP_FAILED, "mmap %s failed: %s", fnKey, strerror(errno));
  NrOrientations = keyStat.st_size / (2 * sizeof(int));
  int **NrSpots;
  NrSpots = allocMatrixIntF(NrOrientations, 2);
  TotalDiffrSpots = 0;
  for (it = 0; it < NrOrientations; it++) {
    NrSpots[it][0] = KeyData[2 * it];
    NrSpots[it][1] = KeyData[2 * it + 1];
    TotalDiffrSpots += NrSpots[it][0];
  }

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
  sprintf(hklfn, "%s/hkls.csv", outputDir);
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

  // Precompute crystal symmetries for misorientation uniqueness check
  double Sym[24][4];
  int NrSymmetries = MakeSymmetries(SpaceGroup, Sym);

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

#ifdef ENABLE_CUDA
  // ══════════════════════════════════════════════════════════════
  //  GPU PATH: Phase 1 (screening) on GPU, Phase 2 (fitting) on CPU
  // ══════════════════════════════════════════════════════════════
  {
    printf("\n=== GPU-ACCELERATED PATH ===\n");
    double gpu_t0 = omp_get_wtime();

    // Initialise GPU context
    NFGPUContext *gpuCtx = nf_gpu_init(0, NrPixelsY, NrPixelsZ, nrFiles, nLayers);
    if (!gpuCtx) {
      fprintf(stderr, "NF GPU: init failed, falling back to CPU path\n");
      goto cpu_fallback;
    }

    // Upload ObsSpotsInfo (reorganized into per-frame slabs)
    if (nf_gpu_upload_obs_spots(gpuCtx, ObsSpotsInfo, SizeObsSpots) != 0) {
      fprintf(stderr, "NF GPU: obs spots upload failed\n");
      nf_gpu_destroy(gpuCtx);
      goto cpu_fallback;
    }

    // Upload orientations (precompute spot metadata)
    if (nf_gpu_upload_orientations(gpuCtx,
            OrientationMatrix, NrOrientations,
            SpotsMat, (const int *)KeyData,
            (const double (*)[4])hkls, Thetas, n_hkls, Gs,
            Lsd, ybc, zbc, px, parsed_lines[0].gs,
            OmegaStart, OmegaStep, RotMatTilts,
            ExcludePoleAngle,
            (const double (*)[2])OmegaRanges, NoOfOmegaRanges,
            (const double (*)[4])BoxSizes,
            Wedge, Wavelength) != 0) {
      fprintf(stderr, "NF GPU: orientation upload failed\n");
      nf_gpu_destroy(gpuCtx);
      goto cpu_fallback;
    }

    // Prepare ALL voxels for GPU screening (not just current block).
    // The GPU replaces the multi-process fan-out — it screens every voxel in
    // one kernel launch.
    int nVoxels = TotalNrSpots;
    double *allXG = (double *)malloc(nVoxels * 3 * sizeof(double));
    double *allYG = (double *)malloc(nVoxels * 3 * sizeof(double));

    // Voxel data for GPU Phase 2 fitting (same fields as parsed_lines)
    struct ParsedLine *gpu_parsed =
        (struct ParsedLine *)malloc(nVoxels * sizeof(struct ParsedLine));

    // Re-read grid file from the top to get ALL voxels
    FILE *fp_gpu = fopen(fnG, "r");
    char gpu_line[1000];
    fgets(gpu_line, sizeof(gpu_line), fp_gpu); // skip header (TotalNrSpots)
    for (int v = 0; v < nVoxels; v++) {
      struct ParsedLine *gp = &gpu_parsed[v];
      if (!fgets(gpu_line, sizeof(gpu_line), fp_gpu) ||
          sscanf(gpu_line, "%lf %lf %lf %lf %lf",
                 &gp->y1, &gp->y2, &gp->xs, &gp->ys, &gp->gs) != 5) {
        gp->valid = 0;
        for (int k = 0; k < 3; k++) {
          allXG[v * 3 + k] = 0;
          allYG[v * 3 + k] = 0;
        }
        continue;
      }
      gp->valid = 1;
      double XY[3][3];
      if (gp->y1 > gp->y2) {
        XY[0][0] = gp->xs;         XY[0][1] = gp->ys - gp->y1;
        XY[1][0] = gp->xs - gp->gs; XY[1][1] = gp->ys + gp->y2;
        XY[2][0] = gp->xs + gp->gs; XY[2][1] = gp->ys + gp->y2;
      } else {
        XY[0][0] = gp->xs;         XY[0][1] = gp->ys + gp->y2;
        XY[1][0] = gp->xs - gp->gs; XY[1][1] = gp->ys - gp->y1;
        XY[2][0] = gp->xs + gp->gs; XY[2][1] = gp->ys - gp->y1;
      }
      for (int k = 0; k < 3; k++) {
        allXG[v * 3 + k] = XY[k][0];
        allYG[v * 3 + k] = XY[k][1];
      }
    }
    fclose(fp_gpu);

    // Run GPU screening
    NFGPUWinner *gpuWinners = NULL;
    int nGpuWinners = 0;
    if (nf_gpu_screen(gpuCtx, allXG, allYG, nVoxels,
                      minFracOverlap, &gpuWinners, &nGpuWinners) != 0) {
      fprintf(stderr, "NF GPU: screening failed\n");
      free(allXG); free(allYG); free(gpu_parsed);
      nf_gpu_destroy(gpuCtx);
      goto cpu_fallback;
    }

    double gpu_screen_time = omp_get_wtime() - gpu_t0;
    printf("NF GPU: Phase 1 screening: %d winners in %.2f s\n",
           nGpuWinners, gpu_screen_time);

    // ── Phase 2: CPU fitting using GPU winners ──
    // Build per-voxel winner index
    // winnerStart[v] = first index into gpuWinners for voxel v
    // winnerCount[v] = number of winners for voxel v
    int *winnerCount = (int *)calloc(nVoxels, sizeof(int));
    for (int w = 0; w < nGpuWinners; w++) {
      int v = gpuWinners[w].voxelIdx;
      if (v >= 0 && v < nVoxels)
        winnerCount[v]++;
    }
    int *winnerStart = (int *)malloc(nVoxels * sizeof(int));
    int *winnerPos   = (int *)calloc(nVoxels, sizeof(int));
    int cumulative = 0;
    for (int v = 0; v < nVoxels; v++) {
      winnerStart[v] = cumulative;
      cumulative += winnerCount[v];
    }
    // Build sorted index: winnerIdx[winnerStart[v] + k] = index into gpuWinners
    int *winnerIdx = (int *)malloc(nGpuWinners * sizeof(int));
    for (int w = 0; w < nGpuWinners; w++) {
      int v = gpuWinners[w].voxelIdx;
      if (v >= 0 && v < nVoxels) {
        winnerIdx[winnerStart[v] + winnerPos[v]] = w;
        winnerPos[v]++;
      }
    }
    free(winnerPos);

    // Skip Phase 2 if screen-only mode
    if (screen_only) {
      double gpu_screen_time_only = omp_get_wtime() - gpu_t0;
      printf("NF GPU: screen-only mode — skipping Phase 2 fitting\n");
      printf("NF GPU: Phase 1 screening: %d winners in %.2f s\n",
             nGpuWinners, gpu_screen_time_only);

      // Dump GPU screening results for diagnostics
      // Sort winners by (voxelIdx, orientIdx) for easy diffing
      for (int i = 0; i < nGpuWinners; i++) {
        for (int j = i + 1; j < nGpuWinners; j++) {
          if (gpuWinners[j].voxelIdx < gpuWinners[i].voxelIdx ||
              (gpuWinners[j].voxelIdx == gpuWinners[i].voxelIdx &&
               gpuWinners[j].orientIdx < gpuWinners[i].orientIdx)) {
            NFGPUWinner tmp = gpuWinners[i];
            gpuWinners[i] = gpuWinners[j];
            gpuWinners[j] = tmp;
          }
        }
      }
      FILE *fp_diag = fopen("screen_gpu.csv", "w");
      if (fp_diag) {
        fprintf(fp_diag, "voxelIdx,orientIdx,fracOverlap\n");
        for (int i = 0; i < nGpuWinners; i++) {
          fprintf(fp_diag, "%d,%d,%.6f\n",
                  gpuWinners[i].voxelIdx,
                  gpuWinners[i].orientIdx,
                  gpuWinners[i].fracOverlap);
        }
        fclose(fp_diag);
        printf("NF GPU: diagnostic dump written to screen_gpu.csv (%d records)\n",
               nGpuWinners);
      }

      printf("=== END GPU PATH (screen-only) ===\n");
      free(winnerCount);
      free(winnerStart);
      free(winnerIdx);
      free(allXG);
      free(allYG);
      free(gpu_parsed);
      if (gpuWinners) free(gpuWinners);
      nf_gpu_destroy(gpuCtx);
      goto skip_cpu_path;
    }
    // ── GPU Phase 2 fitting (optional, enabled by MIDAS_GPU_FIT=1) ──
    int gpu_fit = 0;
    {
      const char *env = getenv("MIDAS_GPU_FIT");
      if (env && atoi(env)) gpu_fit = 1;
    }

    if (gpu_fit) {
      printf("NF GPU: Phase 2 (GPU fitting) starting, %d winners...\n", nGpuWinners);
      double gpu_fit_t0 = omp_get_wtime();

      // Upload HKL data for on-device spot computation
      int rc = nf_gpu_upload_hkls(gpuCtx, hkls, Gs, n_hkls,
                                  ExcludePoleAngle, OmegaStart, OmegaStep,
                                  OmegaRanges, nOmeRang, BoxSizes);
      if (rc != 0) {
        fprintf(stderr, "NF GPU: failed to upload HKL data, falling back to CPU\n");
        gpu_fit = 0;
        goto gpu_fit_fallback;
      }

      // Run GPU fitting
      NFGPUFitResult *gpuFitResults = NULL;
      int nGpuFitResults = 0;
      rc = nf_gpu_fit(gpuCtx, gpuWinners, nGpuWinners,
                      allXG, allYG, tol,
                      OrientationMatrix, &gpuFitResults, &nGpuFitResults);
      if (rc != 0) {
        fprintf(stderr, "NF GPU: fitting kernel failed, falling back to CPU\n");
        gpu_fit = 0;
        goto gpu_fit_fallback;
      }

      double gpu_fit_time = omp_get_wtime() - gpu_fit_t0;
      printf("NF GPU: Phase 2 (GPU fitting) complete: %d results in %.2f s\n",
             nGpuFitResults, gpu_fit_time);

      // Post-process: find best per voxel and write output files
      // Build per-voxel best result from GPU fit results
      for (int vIdx = 0; vIdx < nVoxels; vIdx++) {
        if (!gpu_parsed[vIdx].valid) continue;
        double xs = gpu_parsed[vIdx].xs;
        double ys = gpu_parsed[vIdx].ys;
        double gs_v = gpu_parsed[vIdx].gs;
        double y1 = gpu_parsed[vIdx].y1;
        double y2 = gpu_parsed[vIdx].y2;
        int UD = (y1 > y2) ? -1 : 1;
        double GridSize_v = 2 * gs_v;

        double BestFrac = -1, BestEuler[3] = {0};
        double bestRowNr_v = 0;
        int nWinnersThisVoxel = winnerCount[vIdx];

        // Scan GPU fit results for this voxel
        for (int r = 0; r < nGpuFitResults; r++) {
          if (gpuFitResults[r].voxelIdx != vIdx) continue;
          double frac = gpuFitResults[r].fracOverlap;
          if (frac >= BestFrac) {
            BestFrac = frac;
            BestEuler[0] = gpuFitResults[r].eulerA * deg2rad;
            BestEuler[1] = gpuFitResults[r].eulerB * deg2rad;
            BestEuler[2] = gpuFitResults[r].eulerC * deg2rad;
            // Find matching winner to get orientIdx
            for (int w = 0; w < nWinnersThisVoxel; w++) {
              int gpuIdx = winnerIdx[winnerStart[vIdx] + w];
              if (gpuWinners[gpuIdx].orientIdx >= 0) {
                bestRowNr_v = (double)gpuWinners[gpuIdx].orientIdx;
              }
            }
          }
        }

        double outresult[11] = {bestRowNr_v, (double)nWinnersThisVoxel,
                                0.0, xs, ys, GridSize_v,
                                (double)UD, BestEuler[0], BestEuler[1],
                                BestEuler[2], BestFrac};
        int SizeWritten = 11 * sizeof(double);
        size_t OffsetHere = (size_t)vIdx * SizeWritten;

        double ResultMatr_v[7 + (nSaves * 4)];
        ResultMatr_v[0] = (double)atoi(argv[2]);
        ResultMatr_v[1] = (double)nWinnersThisVoxel;
        ResultMatr_v[2] = 0;
        ResultMatr_v[3] = xs;
        ResultMatr_v[4] = ys;
        ResultMatr_v[5] = GridSize_v;
        ResultMatr_v[6] = (double)UD;
        for (int i = 0; i < nSaves; i++) {
          ResultMatr_v[7 + i * 4] = 0;
          ResultMatr_v[7 + i * 4 + 1] = 0;
          ResultMatr_v[7 + i * 4 + 2] = 0;
          ResultMatr_v[7 + i * 4 + 3] = 0;
        }
        if (BestFrac >= 0) {
          ResultMatr_v[7 + 0] = BestEuler[0];
          ResultMatr_v[7 + 1] = BestEuler[1];
          ResultMatr_v[7 + 2] = BestEuler[2];
          ResultMatr_v[7 + 3] = BestFrac;
        }
        int SizeWritten2 = (7 + (nSaves * 4)) * sizeof(double);
        size_t OffsetThis = (size_t)vIdx * SizeWritten2;
        pwrite(result, outresult, SizeWritten, OffsetHere);
        pwrite(result2, ResultMatr_v, SizeWritten2, OffsetThis);
      }

      if (gpuFitResults) free(gpuFitResults);

      // Cleanup and skip CPU path
      free(winnerCount);
      free(winnerStart);
      free(winnerIdx);
      free(allXG);
      free(allYG);
      free(gpu_parsed);
      if (gpuWinners) free(gpuWinners);
      nf_gpu_destroy(gpuCtx);
      goto skip_cpu_path;
    }

gpu_fit_fallback:

    printf("NF GPU: Phase 2 (CPU fitting) starting, %d voxels with winners...\n",
           nVoxels);
    double gpu_fit_t0 = omp_get_wtime();

    // OMP parallel over voxels (same structure as CPU path)
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int vIdx = 0; vIdx < nVoxels; vIdx++) {
      int rown_gpu = vIdx;  // Direct index into TotalNrSpots
      if (!gpu_parsed[vIdx].valid) continue;
      int nWinnersThisVoxel = winnerCount[vIdx];

      double y1 = gpu_parsed[vIdx].y1;
      double y2 = gpu_parsed[vIdx].y2;
      double xs = gpu_parsed[vIdx].xs;
      double ys = gpu_parsed[vIdx].ys;
      double gs_v = gpu_parsed[vIdx].gs;

      double XY[3][3];
      int UD;
      if (y1 > y2) {
        UD = -1;
        XY[0][0] = xs;      XY[0][1] = ys - y1;
        XY[1][0] = xs - gs_v; XY[1][1] = ys + y2;
        XY[2][0] = xs + gs_v; XY[2][1] = ys + y2;
      } else {
        UD = 1;
        XY[0][0] = xs;      XY[0][1] = ys + y2;
        XY[1][0] = xs - gs_v; XY[1][1] = ys - y1;
        XY[2][0] = xs + gs_v; XY[2][1] = ys - y1;
      }
      double GridSize_v = 2 * gs_v;
      int NrPixelsGrid_v = 2 * (int)(ceil((gs_v * 2) / px)) * (int)(ceil((gs_v * 2) / px));

      double MatIn_v[3], P0_v[nLayers][3], P0T_v[3], XG_v[3], YG_v[3];
      MatIn_v[0] = 0; MatIn_v[1] = 0; MatIn_v[2] = 0;
      for (int i = 0; i < nLayers; i++) {
        MatIn_v[0] = -Lsd[i];
        MatrixMultF(RotMatTilts, MatIn_v, P0T_v);
        for (int j = 0; j < 3; j++) P0_v[i][j] = P0T_v[j];
      }
      for (int j = 0; j < 3; j++) {
        XG_v[j] = XY[j][0];
        YG_v[j] = XY[j][1];
      }

      // Setup result tracking (same as CPU path)
      double BestFrac = -1, BestEuler[3] = {0};
      double ResultMatr_v[7 + (nSaves * 4)];
      double bestRowNr_v = 0;
      int OrientationGoodID_v = nWinnersThisVoxel;

      ResultMatr_v[0] = (double)atoi(argv[2]);
      ResultMatr_v[1] = (double)nWinnersThisVoxel;
      ResultMatr_v[2] = 0;
      ResultMatr_v[3] = xs;
      ResultMatr_v[4] = ys;
      ResultMatr_v[5] = GridSize_v;
      ResultMatr_v[6] = (double)UD;
      int nFilled = 0;
      for (int i = 0; i < nSaves; i++) {
        ResultMatr_v[7 + i * 4] = 0;
        ResultMatr_v[7 + i * 4 + 1] = 0;
        ResultMatr_v[7 + i * 4 + 2] = 0;
        ResultMatr_v[7 + i * 4 + 3] = 0;
      }

      if (nWinnersThisVoxel > 0) {
        double tFitStart = omp_get_wtime();
        int totalNloptEvals = 0;

        for (int wi = 0; wi < nWinnersThisVoxel; wi++) {
          int gpuIdx = winnerIdx[winnerStart[vIdx] + wi];
          int oriIdx = gpuWinners[gpuIdx].orientIdx;

          // Get orientation matrix, normalize, convert to Euler
          double OMTemp[9], OrientIn[3][3], EulerIn[3];
          for (int j = 0; j < 9; j++) {
            OMTemp[j] = OrientationMatrix[oriIdx * 9 + j];
            if (OMTemp[j] == -0.0) OMTemp[j] = 0;
          }
          double OMNorm[9];
          NormalizeMat(OMTemp, OMNorm);
          Convert9To3x3(OMNorm, OrientIn);
          OrientMat2Euler(OrientIn, EulerIn);

          int fitNevals = 0, fitRetcode = 0;
          double EulerOutA, EulerOutB, EulerOutC, FracOut;

          // Debug: activate trace for first voxel's first winner
          static int cpuDebugDone = 0;
          if (!cpuDebugDone && wi == 0) {
            int voxBase = gpuWinners[gpuIdx].voxelIdx;
            printf("CPU FIT debug: voxIdx=%d oriIdx=%d euler_deg=(%.6f,%.6f,%.6f) XG=(%.4f,%.4f,%.4f)\n",
                   voxBase, oriIdx, EulerIn[0]*57.2957795130823, EulerIn[1]*57.2957795130823, EulerIn[2]*57.2957795130823,
                   XG_v[0], XG_v[1], XG_v[2]);
            g_debugNextFit = 1;  // enable trace for next FitOrientation call
          }

          FitOrientation(nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots,
                         XG_v, YG_v, RotMatTilts, OmegaStart, OmegaStep, px,
                         ybc, zbc, gs_v, OmegaRanges, nOmeRang, BoxSizes,
                         P0_v, NrPixelsGrid_v, ObsSpotsInfo, EulerIn, tol,
                         &EulerOutA, &EulerOutB, &EulerOutC, &FracOut,
                         hkls, Thetas, n_hkls, Gs,
                         &fitNevals, &fitRetcode, NrPixelsY, NrPixelsZ);
          if (!cpuDebugDone && wi == 0) cpuDebugDone = 1;
          totalNloptEvals += fitNevals;

          double Fractions = 1 - FracOut;
          if (Fractions >= BestFrac) {
            bestRowNr_v = (double)oriIdx;
            BestFrac = Fractions;
            BestEuler[0] = EulerOutA;
            BestEuler[1] = EulerOutB;
            BestEuler[2] = EulerOutC;
            if (1 - BestFrac < 0.0001 && nSaves == 1)
              break;
          }
          if (nSaves > 1) {
            // Uniqueness check + sorted insertion (same as CPU path)
            double candEul[3] = {EulerOutA * rad2deg, EulerOutB * rad2deg,
                                 EulerOutC * rad2deg};
            double candOM[3][3], candOM9[9], candQuat[4];
            Euler2OrientMat(candEul, candOM);
            for (int q = 0; q < 3; q++)
              for (int r = 0; r < 3; r++)
                candOM9[q * 3 + r] = candOM[q][r];
            OrientMat2Quat(candOM9, candQuat);

            int isUnique = 1;
            for (int s = 0; s < nFilled; s++) {
              double existEul[3] = {ResultMatr_v[7 + s * 4 + 0] * rad2deg,
                                    ResultMatr_v[7 + s * 4 + 1] * rad2deg,
                                    ResultMatr_v[7 + s * 4 + 2] * rad2deg};
              double existOM[3][3], existOM9[9], existQuat[4];
              Euler2OrientMat(existEul, existOM);
              for (int q = 0; q < 3; q++)
                for (int r = 0; r < 3; r++)
                  existOM9[q * 3 + r] = existOM[q][r];
              OrientMat2Quat(existOM9, existQuat);
              double misoAngle;
              GetMisOrientationAngle(candQuat, existQuat, &misoAngle,
                                     NrSymmetries, Sym);
              if (misoAngle < MinMisoNSaves) {
                isUnique = 0;
                break;
              }
            }
            if (!isUnique) continue;

            int inserted = 0;
            for (int j = 0; j < nFilled; j++) {
              if (Fractions >= ResultMatr_v[7 + j * 4 + 3]) {
                for (int m = nFilled - 1; m >= j; m--) {
                  if (m == nSaves - 1) continue;
                  for (int t = 0; t < 4; t++)
                    ResultMatr_v[7 + (m + 1) * 4 + t] = ResultMatr_v[7 + m * 4 + t];
                }
                ResultMatr_v[7 + j * 4] = EulerOutA;
                ResultMatr_v[7 + j * 4 + 1] = EulerOutB;
                ResultMatr_v[7 + j * 4 + 2] = EulerOutC;
                ResultMatr_v[7 + j * 4 + 3] = Fractions;
                inserted = 1;
                if (nFilled < nSaves) nFilled++;
                break;
              }
            }
            if (!inserted && nFilled < nSaves) {
              ResultMatr_v[7 + nFilled * 4] = EulerOutA;
              ResultMatr_v[7 + nFilled * 4 + 1] = EulerOutB;
              ResultMatr_v[7 + nFilled * 4 + 2] = EulerOutC;
              ResultMatr_v[7 + nFilled * 4 + 3] = Fractions;
              nFilled++;
            }
          }
        }

        double tFitElapsed = omp_get_wtime() - tFitStart;
        double outresult[11] = {bestRowNr_v, (double)nWinnersThisVoxel,
                                tFitElapsed, xs, ys, GridSize_v,
                                (double)UD, BestEuler[0], BestEuler[1],
                                BestEuler[2], BestFrac};
        int SizeWritten = 11 * sizeof(double);
        size_t OffsetHere = (size_t)rown_gpu * SizeWritten;
        int SizeWritten2 = (7 + (nSaves * 4)) * sizeof(double);
        size_t OffsetThis = (size_t)rown_gpu * SizeWritten2;

        pwrite(result, outresult, SizeWritten, OffsetHere);
        pwrite(result2, ResultMatr_v, SizeWritten2, OffsetThis);

        if (verbose) {
#pragma omp critical
          {
            printf("%zu %d ", OffsetHere, rown_gpu);
            for (int i = 0; i < 11; i++) printf("%.5f ", outresult[i]);
            printf("\n");
            fflush(stdout);
          }
        }
      } else {
        // No winners — write empty result
        double outresult[11] = {0, 0, 0, xs, ys, GridSize_v, (double)UD,
                                0, 0, 0, 0};
        int SizeWritten = 11 * sizeof(double);
        size_t OffsetHere = (size_t)rown_gpu * SizeWritten;
        int SizeWritten2 = (7 + (nSaves * 4)) * sizeof(double);
        size_t OffsetThis = (size_t)rown_gpu * SizeWritten2;
        ResultMatr_v[2] = 0;
        pwrite(result, outresult, SizeWritten, OffsetHere);
        pwrite(result2, ResultMatr_v, SizeWritten2, OffsetThis);
      }
    }

    double gpu_fit_time = omp_get_wtime() - gpu_fit_t0;
    printf("NF GPU: Phase 2 fitting: %.2f s\n", gpu_fit_time);

    free(winnerCount);
    free(winnerStart);
    free(winnerIdx);
    free(allXG);
    free(allYG);
    free(gpu_parsed);
    if (gpuWinners) free(gpuWinners);
    nf_gpu_destroy(gpuCtx);

    double gpu_total = omp_get_wtime() - gpu_t0;
    printf("NF GPU: total time: %.2f s (screen %.2f + fit %.2f)\n",
           gpu_total, gpu_screen_time, gpu_fit_time);
    printf("=== END GPU PATH ===\n\n");

    // Skip CPU path
    goto skip_cpu_path;
  }
cpu_fallback:
  printf("Running CPU path...\n");
#endif /* ENABLE_CUDA */

  double cpu_wall_t0 = omp_get_wtime();
  double cpu_screen_accum = 0.0;  // accumulated Phase 1 time across threads
  double cpu_fit_accum = 0.0;     // accumulated Phase 2 time across threads
  int cpu_total_winners = 0;

  // Diagnostic collection for screen_cpu.csv
  int cpu_diag_cap = 100000;
  int cpu_diag_count = 0;
  int *cpu_diag_vox = (int *)malloc(cpu_diag_cap * sizeof(int));
  int *cpu_diag_ori = (int *)malloc(cpu_diag_cap * sizeof(int));
  double *cpu_diag_frac = (double *)malloc(cpu_diag_cap * sizeof(double));

#pragma omp parallel for num_threads(numProcs) private(rown) schedule(dynamic) \
    reduction(+:cpu_screen_accum, cpu_fit_accum, cpu_total_winners)
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
    // Hoisted InPixels allocation
    int **InPixels;
    InPixels = allocMatrixIntF(NrPixelsGrid, 2);
    double tScreenStart = omp_get_wtime();
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

        // Collect for diagnostic dump
        if (screen_only) {
          #pragma omp critical
          {
            if (cpu_diag_count < cpu_diag_cap) {
              cpu_diag_vox[cpu_diag_count] = rown;
              cpu_diag_ori[cpu_diag_count] = i;
              cpu_diag_frac[cpu_diag_count] = FracOverT;
              cpu_diag_count++;
            }
          }
        }

        if (OrientationGoodID >= MAX_POINTS_GRID_GOOD)
          break;
      }
    }
    FreeMemMatrixInt(InPixels, NrPixelsGrid);
    double tScreenEnd = omp_get_wtime();
    cpu_screen_accum += (tScreenEnd - tScreenStart);
    cpu_total_winners += OrientationGoodID;
    fflush(stdout);
    // Skip Phase 2 if screen-only mode
    if (screen_only) continue;
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
      int firstSol = 0;
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
          // Convert candidate Euler angles to quaternion for miso check
          double candEul[3] = {EulerOutA * rad2deg, EulerOutB * rad2deg,
                               EulerOutC * rad2deg};
          double candOM[3][3], candOM9[9], candQuat[4];
          Euler2OrientMat(candEul, candOM);
          for (int q = 0; q < 3; q++)
            for (int r = 0; r < 3; r++)
              candOM9[q * 3 + r] = candOM[q][r];
          OrientMat2Quat(candOM9, candQuat);

          // Check uniqueness against existing saved solutions
          int isUnique = 1;
          for (int s = 0; s < nFilled; s++) {
            double existEul[3] = {ResultMatr[7 + s * 4 + 0] * rad2deg,
                                  ResultMatr[7 + s * 4 + 1] * rad2deg,
                                  ResultMatr[7 + s * 4 + 2] * rad2deg};
            double existOM[3][3], existOM9[9], existQuat[4];
            Euler2OrientMat(existEul, existOM);
            for (int q = 0; q < 3; q++)
              for (int r = 0; r < 3; r++)
                existOM9[q * 3 + r] = existOM[q][r];
            OrientMat2Quat(existOM9, existQuat);
            double misoAngle;
            GetMisOrientationAngle(candQuat, existQuat, &misoAngle,
                                   NrSymmetries, Sym);
            if (misoAngle < MinMisoNSaves) {
              isUnique = 0;
              break;
            }
          }
          if (!isUnique)
            continue; // skip duplicate orientation

          int inserted = 0;
          // Walk sorted array; insert where Fractions >= existing entry
          for (j = 0; j < nFilled; j++) {
            if (Fractions >= ResultMatr[7 + j * 4 + 3]) {
              // Shift entries [j..nFilled-1] down by one (last falls off if full)
              for (m = nFilled - 1; m >= j; m--) {
                if (m == nSaves - 1)
                  continue; // last slot, discard
                for (t = 0; t < 4; t++) {
                  ResultMatr[7 + (m + 1) * 4 + t] =
                      ResultMatr[7 + (m) * 4 + t];
                }
              }
              ResultMatr[7 + j * 4] = EulerOutA;
              ResultMatr[7 + j * 4 + 1] = EulerOutB;
              ResultMatr[7 + j * 4 + 2] = EulerOutC;
              ResultMatr[7 + j * 4 + 3] = Fractions;
              inserted = 1;
              if (nFilled < nSaves)
                nFilled++;
              break;
            }
          }
          // Worse than all existing, but room remains — append at end
          if (!inserted && nFilled < nSaves) {
            ResultMatr[7 + nFilled * 4] = EulerOutA;
            ResultMatr[7 + nFilled * 4 + 1] = EulerOutB;
            ResultMatr[7 + nFilled * 4 + 2] = EulerOutC;
            ResultMatr[7 + nFilled * 4 + 3] = Fractions;
            nFilled++;
          }
        }
      }
    } else {
      if (verbose) printf("No good ID found.\n");
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
      if (verbose) {
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
    }
    int rc5 = pwrite(result2, ResultMatr, SizeWritten2, OffsetThis);
    if (rc5 < 0) {
#pragma omp critical
      printf("Could not write all matches %zu %d %d %d.\n", OffsetThis, rown,
             startRowNr, endRowNr);
    }

    double tFitEnd = omp_get_wtime();
    cpu_fit_accum += (tFitEnd - tFitStart);
  }

  double cpu_wall_elapsed = omp_get_wtime() - cpu_wall_t0;

  // Dump CPU screening results for diagnostics (sort by voxelIdx, orientIdx)
  if (screen_only && cpu_diag_count > 0) {
    for (int i = 0; i < cpu_diag_count; i++) {
      for (int j = i + 1; j < cpu_diag_count; j++) {
        if (cpu_diag_vox[j] < cpu_diag_vox[i] ||
            (cpu_diag_vox[j] == cpu_diag_vox[i] &&
             cpu_diag_ori[j] < cpu_diag_ori[i])) {
          int tv = cpu_diag_vox[i]; cpu_diag_vox[i] = cpu_diag_vox[j]; cpu_diag_vox[j] = tv;
          int to = cpu_diag_ori[i]; cpu_diag_ori[i] = cpu_diag_ori[j]; cpu_diag_ori[j] = to;
          double tf = cpu_diag_frac[i]; cpu_diag_frac[i] = cpu_diag_frac[j]; cpu_diag_frac[j] = tf;
        }
      }
    }
    FILE *fp_diag = fopen("screen_cpu.csv", "w");
    if (fp_diag) {
      fprintf(fp_diag, "voxelIdx,orientIdx,fracOverlap\n");
      for (int i = 0; i < cpu_diag_count; i++) {
        fprintf(fp_diag, "%d,%d,%.6f\n",
                cpu_diag_vox[i], cpu_diag_ori[i], cpu_diag_frac[i]);
      }
      fclose(fp_diag);
      printf("NF CPU: diagnostic dump written to screen_cpu.csv (%d records)\n",
             cpu_diag_count);
    }
  }
  free(cpu_diag_vox);
  free(cpu_diag_ori);
  free(cpu_diag_frac);

  printf("\n=== CPU PATH TIMING ===\n");
  printf("NF CPU: Phase 1 screening: %.2f s (sum of per-thread times)\n", cpu_screen_accum);
  printf("NF CPU: Phase 2 fitting:   %.2f s (sum of per-thread times)\n", cpu_fit_accum);
  printf("NF CPU: wall time:         %.2f s (%d voxels, %d orientations, %d threads)\n",
         cpu_wall_elapsed, endRowNr - startRowNr + 1, NrOrientations, numProcs);
  printf("NF CPU: total winners:     %d (avg %.1f/voxel)\n",
         cpu_total_winners,
         (endRowNr - startRowNr + 1) > 0 ? (double)cpu_total_winners / (endRowNr - startRowNr + 1) : 0);
  printf("=== END CPU PATH ===\n");

#ifdef ENABLE_CUDA
skip_cpu_path:
#endif
  // Close files and clean up mmap'd regions
  free(parsed_lines);
  free(Gs);
  close(result);
  close(result2);
  munmap(ObsSpotsInfo, size);
  close(descp);
  munmap(SpotsMat, size2);
  close(spf);
  munmap(OrientationMatrix, size3);
  close(omf);
  munmap(KeyData, keyStat.st_size);
  close(keyfd);
  free(OrientMatrixAll);
  free(ThrSpsAll);
  FreeMemMatrixInt(NrSpots, NrOrientations);
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return 0;
}