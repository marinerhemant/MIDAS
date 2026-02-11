//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "nf_headers.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
  double XGrain[3];
  double YGrain[3];
  double OmegaStart;
  double OmegaStep;
  double px;
  double gs;
  double hkls[5000][4];
  int n_hkls;
  double Thetas[5000];
  int NoOfOmegaRanges;
  int NrPixelsGrid;
  double OmegaRanges[MAX_N_OMEGA_RANGES][2];
  double BoxSizes[MAX_N_OMEGA_RANGES][4];
  double **P0;
  int *ObsSpotsInfo;
};

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  int i, j, count = 1;
  const int NrOfFiles = f_data->NrOfFiles;
  const int nLayers = f_data->nLayers;
  const double ExcludePoleAngle = f_data->ExcludePoleAngle;
  const long long int SizeObsSpots = f_data->SizeObsSpots;
  double XGrain[3];
  double YGrain[3];
  const double OmegaStart = f_data->OmegaStart;
  const double OmegaStep = f_data->OmegaStep;
  const double px = f_data->px;
  const double gs = f_data->gs;
  const int NoOfOmegaRanges = f_data->NoOfOmegaRanges;
  const int NrPixelsGrid = f_data->NrPixelsGrid;
  double P0[nLayers][3];
  double OmegaRanges[MAX_N_OMEGA_RANGES][2];
  double BoxSizes[MAX_N_OMEGA_RANGES][4];
  double hkls[5000][4];
  int n_hkls = f_data->n_hkls;
  double Thetas[5000];
  for (i = 0; i < 5000; i++) {
    hkls[i][0] = f_data->hkls[i][0];
    hkls[i][1] = f_data->hkls[i][1];
    hkls[i][2] = f_data->hkls[i][2];
    hkls[i][3] = f_data->hkls[i][3];
    Thetas[i] = f_data->Thetas[i];
  }
  int *ObsSpotsInfo;
  ObsSpotsInfo = &(f_data->ObsSpotsInfo[0]);
  for (i = 0; i < 3; i++) {
    XGrain[i] = f_data->XGrain[i];
    YGrain[i] = f_data->YGrain[i];
    for (j = 0; j < nLayers; j++) {
      P0[j][i] = f_data->P0[j][i];
    }
  }
  for (i = 0; i < MAX_N_OMEGA_RANGES; i++) {
    for (j = 0; j < 2; j++) {
      OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
    }
    for (j = 0; j < 4; j++) {
      BoxSizes[i][j] = f_data->BoxSizes[i][j];
    }
  }
  double OrientMatIn[3][3], FracOverlap, EulIn[3];
  EulIn[0] = x[0];
  EulIn[1] = x[1];
  EulIn[2] = x[2];
  Euler2OrientMat(EulIn, OrientMatIn);
  double Lsd[nLayers], ybc[nLayers], zbc[nLayers], tx, ty, tz,
      RotMatTilts[3][3];
  tx = x[3];
  ty = x[4];
  tz = x[5];
  RotationTilts(tx, ty, tz, RotMatTilts);
  Lsd[0] = x[6];
  for (i = 1; i < nLayers; i++) {
    Lsd[i] = Lsd[i - 1] + x[6 + i];
  }
  for (i = 0; i < nLayers; i++) {
    ybc[i] = x[6 + nLayers + i];
    zbc[i] = x[6 + nLayers + nLayers + i];
  }
  double *TheorSpots;
  TheorSpots = malloc(MAX_N_SPOTS * 3 * sizeof(*TheorSpots));
  CalcOverlapAccOrient(NrOfFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots,
                       XGrain, YGrain, RotMatTilts, OmegaStart, OmegaStep, px,
                       ybc, zbc, gs, hkls, n_hkls, Thetas, OmegaRanges,
                       NoOfOmegaRanges, BoxSizes, P0, NrPixelsGrid,
                       ObsSpotsInfo, OrientMatIn, &FracOverlap, TheorSpots);
  free(TheorSpots);
  return (1 - FracOverlap);
}

void FitOrientation(
    const int NrOfFiles, const int nLayers, const double ExcludePoleAngle,
    double Lsd[nLayers], const long long int SizeObsSpots,
    const double XGrain[3], const double YGrain[3], double TiltsOrig[3],
    const double OmegaStart, const double OmegaStep, const double px,
    double ybc[nLayers], double zbc[nLayers], const double gs,
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4], double P0[nLayers][3],
    const int NrPixelsGrid, int *ObsSpotsInfo, double EulerIn[3], double tol,
    double *EulerOutA, double *EulerOutB, double *EulerOutC,
    double *ResultFracOverlap, double hkls[5000][4], double Thetas[5000],
    int n_hkls, double *LsdFit, double *TiltsFit, double **BCsFit,
    double tolLsd, double tolLsdRel, double tolTilts, double tolBCsa,
    double tolBCsb) {
  unsigned n;
  long int i, j;
  n = 6 + (nLayers * 3);
  double x[n], xl[n], xu[n];
  for (i = 0; i < n; i++) {
    x[i] = 0;
    xl[i] = 0;
    xu[i] = 0;
  }
  for (i = 0; i < 3; i++) {
    x[i] = EulerIn[i];
    xl[i] = x[i] - (tol * M_PI / 180);
    xu[i] = x[i] + (tol * M_PI / 180);
  }
  int count = 0;
  for (i = 3; i < 6; i++) {
    x[i] = TiltsOrig[count];
    xl[i] = x[i] - tolTilts;
    xu[i] = x[i] + tolTilts;
    count++;
  }
  count = 0;
  x[6] = Lsd[0];
  xl[6] = x[6] - tolLsd;
  xu[6] = x[6] + tolLsd;
  count++;
  for (i = 7; i < nLayers + 6; i++) {
    x[i] = Lsd[count] - Lsd[count - 1];
    xl[i] = x[i] - tolLsdRel;
    xu[i] = x[i] + tolLsdRel;
    count++;
  }
  count = 0;
  for (i = 6 + nLayers; i < 6 + (nLayers * 2); i++) {
    x[i] = ybc[count];
    x[i + nLayers] = zbc[count];
    xl[i] = x[i] - tolBCsa;
    xl[i + nLayers] = x[i + nLayers] - tolBCsb;
    xu[i] = x[i] + tolBCsa;
    xu[i + nLayers] = x[i + nLayers] + tolBCsb;
    count++;
  }
  struct my_func_data f_data;
  f_data.NrOfFiles = NrOfFiles;
  f_data.nLayers = nLayers;
  f_data.n_hkls = n_hkls;
  for (i = 0; i < 5000; i++) {
    f_data.hkls[i][0] = hkls[i][0];
    f_data.hkls[i][1] = hkls[i][1];
    f_data.hkls[i][2] = hkls[i][2];
    f_data.hkls[i][3] = hkls[i][3];
    f_data.Thetas[i] = Thetas[i];
  }
  f_data.ExcludePoleAngle = ExcludePoleAngle;
  f_data.SizeObsSpots = SizeObsSpots;
  f_data.P0 = allocMatrixF(nLayers, 3);
  for (i = 0; i < 3; i++) {
    f_data.XGrain[i] = XGrain[i];
    f_data.YGrain[i] = YGrain[i];
    for (j = 0; j < nLayers; j++) {
      f_data.P0[j][i] = P0[j][i];
    }
  }
  for (i = 0; i < MAX_N_OMEGA_RANGES; i++) {
    for (j = 0; j < 2; j++) {
      f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
    }
    for (j = 0; j < 4; j++) {
      f_data.BoxSizes[i][j] = BoxSizes[i][j];
    }
  }
  f_data.ObsSpotsInfo = &ObsSpotsInfo[0];
  f_data.OmegaStart = OmegaStart;
  f_data.OmegaStep = OmegaStep;
  f_data.px = px;
  f_data.gs = gs;
  f_data.NoOfOmegaRanges = NoOfOmegaRanges;
  f_data.NrPixelsGrid = NrPixelsGrid;
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;
  double tole = 1e-3;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function, trp);
  double minf = 1;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  *ResultFracOverlap = minf;
  *EulerOutA = x[0];
  *EulerOutB = x[1];
  *EulerOutC = x[2];
  TiltsFit[0] = x[3];
  TiltsFit[1] = x[4];
  TiltsFit[2] = x[5];
  LsdFit[0] = x[6];
  for (i = 1; i < nLayers; i++) {
    LsdFit[i] = LsdFit[i - 1] + x[6 + i];
  }
  for (i = 0; i < nLayers; i++) {
    BCsFit[i][0] = x[6 + nLayers + i];
    BCsFit[i][1] = x[6 + nLayers + nLayers + i];
  }
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double diftotal;
  start = clock();

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
  double px, OmegaStart, OmegaStep, tol, lsdtol, tiltstol, bctola, bctolb,
      lsdtolrel;
  char fn[1000];
  char fn2[1000];
  char direct[1000];
  double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
  int cntr = 0, countr = 0, conter = 0, StartNr, EndNr, intdummy, SpaceGroup,
      RingsToUse[100], nRingsToUse = 0;
  int NoOfOmegaRanges = 0;
  Wedge = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
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
    str = "RingsToUse ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRingsToUse]);
      nRingsToUse++;
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
    str = "LsdTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &lsdtol);
      continue;
    }
    str = "LsdRelativeTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &lsdtolrel);
      continue;
    }
    str = "BCTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &bctola, &bctolb);
      continue;
    }
    str = "TiltsTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tiltstol);
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
  }
  int i, j, m, nrFiles, nrPixels;
  for (i = 0; i < NoOfOmegaRanges; i++) {
    OmegaRang[i][0] = OmegaRanges[i][0];
    OmegaRang[i][1] = OmegaRanges[i][1];
  }
  nOmeRang = NoOfOmegaRanges;
  fclose(fileParam);

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("       FitOrientationParameters: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:      %s\n", direct);
  printf("  ReducedFileName:    %s\n", fn2);
  printf("\n--- Geometry ---\n");
  printf("  nDistances (nLayers): %d\n", nLayers);
  printf("  SpaceGroup:           %d\n", SpaceGroup);
  printf("  Wavelength:           %f\n", Wavelength);
  printf("  Wedge:                %f\n", Wedge);
  printf("  px (pixel size):      %f\n", px);
  printf("  MaxRingRad:           %f\n", MaxRingRad);
  printf("\n--- Detector Distances (Lsd) ---\n");
  for (i = 0; i < nLayers; i++) {
    printf("  Lsd[%d]:               %f\n", i, Lsd[i]);
  }
  printf("\n--- Beam Centers (BC) ---\n");
  for (i = 0; i < nLayers; i++) {
    printf("  BC[%d]:                %f  %f\n", i, ybc[i], zbc[i]);
  }
  printf("\n--- Tilts ---\n");
  printf("  tx:                   %f\n", tx);
  printf("  ty:                   %f\n", ty);
  printf("  tz:                   %f\n", tz);
  printf("\n--- Scan Parameters ---\n");
  printf("  OmegaStart:           %f\n", OmegaStart);
  printf("  OmegaStep:            %f\n", OmegaStep);
  printf("  StartNr:              %d\n", StartNr);
  printf("  EndNr:                %d\n", EndNr);
  printf("\n--- Lattice Parameters ---\n");
  printf("  a=%f  b=%f  c=%f\n", LatticeConstant[0], LatticeConstant[1],
         LatticeConstant[2]);
  printf("  alpha=%f  beta=%f  gamma=%f\n", LatticeConstant[3],
         LatticeConstant[4], LatticeConstant[5]);
  printf("\n--- Fitting Tolerances ---\n");
  printf("  OrientTol:            %f\n", tol);
  printf("  LsdTol:               %f\n", lsdtol);
  printf("  LsdRelativeTol:       %f\n", lsdtolrel);
  printf("  TiltsTol:             %f\n", tiltstol);
  printf("  BCTol:                %f  %f\n", bctola, bctolb);
  printf("  MinFracAccept:        %f\n", minFracOverlap);
  printf("  ExcludePoleAngle:     %f\n", ExcludePoleAngle);
  printf("\n--- Omega Ranges (%d) ---\n", NoOfOmegaRanges);
  for (i = 0; i < NoOfOmegaRanges; i++) {
    printf("  OmegaRange[%d]:       %f  %f\n", i, OmegaRanges[i][0],
           OmegaRanges[i][1]);
  }
  printf("\n--- Box Sizes ---\n");
  for (i = 0; i < countr; i++) {
    printf("  BoxSize[%d]:          %f  %f  %f  %f\n", i, BoxSizes[i][0],
           BoxSizes[i][1], BoxSizes[i][2], BoxSizes[i][3]);
  }
  printf("\n--- Rings To Use (%d) ---\n", nRingsToUse);
  for (i = 0; i < nRingsToUse; i++) {
    printf("  Ring: %d\n", RingsToUse[i]);
  }
  printf("\n--- Flags ---\n");
  printf("  Ice9Input:            %s\n", Flag ? "YES" : "NO");
  printf(
      "================================================================\n\n");

  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd[0]);
  int x = 0;
  // Read bin files
  char fnG[1000];
  sprintf(fnG, "%s/grid.txt", direct);
  char fnDS[1000];
  char fnKey[1000];
  char fnOr[1000];
  sprintf(fnDS, "%s/DiffractionSpots.txt", direct);
  sprintf(fnKey, "%s/Key.txt", direct);
  sprintf(fnOr, "%s/OrientMat.txt", direct);
  sprintf(fn, "%s/%s", direct, fn2);
  char *ext = "bin";
  int *ObsSpotsInfo;
  int ReadCode;
  nrFiles = EndNr - StartNr + 1;
  nrPixels = 2048 * 2048;
  long long int SizeObsSpots, iT;
  SizeObsSpots = (nLayers);
  SizeObsSpots *= nrPixels;
  SizeObsSpots *= nrFiles;
  SizeObsSpots /= 32;
  ObsSpotsInfo = malloc(SizeObsSpots * sizeof(*ObsSpotsInfo));
  for (iT = 0; iT < SizeObsSpots; iT++) {
    ObsSpotsInfo[i] = 0;
  }
  memset(ObsSpotsInfo, 0, SizeObsSpots * sizeof(*ObsSpotsInfo));
  printf("Size of spot info: %llu mb\n",
         SizeObsSpots * sizeof(int) / (1024 * 1024));
  if (ObsSpotsInfo == NULL) {
    printf("Could not allocate ObsSpotsInfo.\n");
    return 0;
  }
  ReadCode = ReadBinFiles(fn, ext, StartNr, EndNr, ObsSpotsInfo, nLayers,
                          SizeObsSpots);
  if (ReadCode == 0) {
    printf("Reading bin files did not go well. Please check.\n");
    return 0;
  }
  // Read position.
  int rown = atoi(argv[2]);
  FILE *fp;
  fp = fopen(fnG, "r");
  char line[1024];
  fgets(line, 1000, fp);
  int TotalNrSpots = 0;
  sscanf(line, "%d", &TotalNrSpots);
  if (rown > TotalNrSpots) {
    printf(
        "Error: Grid point number greater than total number of grid points.\n");
    return 0;
  }
  int counter = 0;
  double y1, y2, xs, ys, gs;
  double **XY;
  XY = allocMatrixF(3, 3);
  while (counter < rown) {
    fgets(line, 1000, fp);
    counter += 1;
  }
  sscanf(line, "%lf %lf %lf %lf %lf", &y1, &y2, &xs, &ys, &gs);
  printf("Processing line: %s", line);
  fclose(fp);
  XY[0][0] = xs;
  XY[1][0] = xs - gs;
  XY[2][0] = xs + gs;
  if (y1 > y2) {
    XY[0][1] = ys - y1;
    XY[1][1] = ys + y2;
    XY[2][1] = ys + y2;
  } else {
    XY[0][1] = ys + y2;
    XY[1][1] = ys - y1;
    XY[2][1] = ys - y1;
  }
  double GridSize = 2 * gs;

  // Read Orientations
  clock_t startthis;
  startthis = clock();
  FILE *fd, *fk, *fo;
  int NrOrientations, TotalDiffrSpots;
  fd = fopen(fnDS, "r");
  fk = fopen(fnKey, "r");
  fo = fopen(fnOr, "r");
  fgets(line, 1000, fk);
  sscanf(line, "%d", &NrOrientations);
  int **NrSpots;
  NrSpots = allocMatrixIntF(NrOrientations, 2);
  TotalDiffrSpots = 0;
  for (i = 0; i < NrOrientations; i++) {
    fgets(line, 1000, fk);
    sscanf(line, "%d", &NrSpots[i][0]);
    TotalDiffrSpots += NrSpots[i][0];
    NrSpots[i][1] = TotalDiffrSpots - NrSpots[i][0];
  }
  double **SpotsMat;
  SpotsMat = allocMatrixF(TotalDiffrSpots, 3);
  printf("Diffraction spots: %lu mb\n",
         (TotalDiffrSpots * (3 * sizeof(double) + sizeof(*ObsSpotsInfo))) /
             (1024 * 1024));
  for (i = 0; i < TotalDiffrSpots; i++) {
    fgets(line, 1000, fd);
    sscanf(line, "%lf %lf %lf", &SpotsMat[i][0], &SpotsMat[i][1],
           &SpotsMat[i][2]);
  }
  double **OrientationMatrix;
  OrientationMatrix = allocMatrixF(NrOrientations, 9);
  for (i = 0; i < NrOrientations; i++) {
    fgets(line, 1000, fo);
    sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &OrientationMatrix[i][0], &OrientationMatrix[i][1],
           &OrientationMatrix[i][2], &OrientationMatrix[i][3],
           &OrientationMatrix[i][4], &OrientationMatrix[i][5],
           &OrientationMatrix[i][6], &OrientationMatrix[i][7],
           &OrientationMatrix[i][8]);
  }
  printf("NrOrientations: %lu mb\n",
         NrOrientations * 10 * sizeof(double) / (1024 * 1024));
  // Go through each orientation and compare with observed spots.
  clock_t startthis2;
  startthis2 = clock();
  int NrPixelsGrid = 2 * (ceil((gs * 2) / px)) * (ceil((gs * 2) / px));
  int NrSpotsThis, StartingRowNr;
  double FracOverT;
  double RotMatTilts[3][3], OrientationMatThis[9], OrientationMatThisUnNorm[9];
  RotationTilts(tx, ty, tz, RotMatTilts);
  double **OrientMatrix;
  OrientMatrix = allocMatrixF(MAX_POINTS_GRID_GOOD, 10);
  int OrientationGoodID = 0;
  double MatIn[3], P0[nLayers][3], P0T[3];
  double OrientMatIn[3][3], XG[3], YG[3];
  double *ThrSps;
  ThrSps = malloc(MAX_N_SPOTS * 3 * sizeof(*ThrSps));
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
  for (i = 0; i < NrOrientations; i++) {
    NrSpotsThis = NrSpots[i][0];
    StartingRowNr = NrSpots[i][1];
    m = 0;
    for (m = 0; m < 9; m++) {
      OrientationMatThisUnNorm[m] = OrientationMatrix[i][m];
      fflush(stdout);
      if (OrientationMatThisUnNorm[m] == -0.0) {
        OrientationMatThisUnNorm[m] = 0;
      }
    }
    m = 0;
    NormalizeMat(OrientationMatThisUnNorm, OrientationMatThis);
    for (j = StartingRowNr; j < (StartingRowNr + NrSpotsThis); j++) {
      ThrSps[m * 3 + 0] = SpotsMat[j][0];
      ThrSps[m * 3 + 1] = SpotsMat[j][1];
      ThrSps[m * 3 + 2] = SpotsMat[j][2];
      m++;
    }
    Convert9To3x3(OrientationMatThis, OrientMatIn);
    fflush(stdout);
    CalcFracOverlap(nrFiles, nLayers, NrSpotsThis, ThrSps, OmegaStart,
                    OmegaStep, XG, YG, Lsd, SizeObsSpots, RotMatTilts, px, ybc,
                    zbc, gs, P0, NrPixelsGrid, ObsSpotsInfo, OrientMatIn,
                    &FracOverT);
    if (FracOverT >= minFracOverlap) {
      for (j = 0; j < 9; j++) {
        OrientMatrix[OrientationGoodID][j] = OrientationMatThis[j];
      }
      OrientMatrix[OrientationGoodID][9] = FracOverT;
      OrientationGoodID++;
    }
  }
  if (OrientationGoodID > 0) {
    double EulerIn[3], OrientIn[3][3], FracOut, EulerOutA, EulerOutB, EulerOutC,
        BestFrac, BestEuler[3], OMTemp[9];
    double *LsdFit, *TiltsFit, **BCsFit;
    double TiltsOrig[3];
    TiltsOrig[0] = tx;
    TiltsOrig[1] = ty;
    TiltsOrig[2] = tz;
    LsdFit = malloc(nLayers * sizeof(*LsdFit));
    TiltsFit = malloc(nLayers + sizeof(*TiltsFit));
    BCsFit = allocMatrixF(nLayers, 2);
    int n_hkls = 0;
    double hkls[5000][4];
    double Thetas[5000];
    double OMBest[3][3], EulBest[3];
    char *hklfn = "hkls.csv";
    FILE *hklf = fopen(hklfn, "r");
    fgets(aline, 1000, hklf);
    while (fgets(aline, 1000, hklf) != NULL) {
      sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s", dummy, dummy,
             dummy, dummy, &hkls[n_hkls][3], &hkls[n_hkls][0], &hkls[n_hkls][1],
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
    printf("Number of individual diffracting planes: %d\n", n_hkls);
    BestFrac = -1;
    for (i = 0; i < OrientationGoodID; i++) {
      for (j = 0; j < 9; j++) {
        OMTemp[j] = OrientMatrix[i][j];
      }
      Convert9To3x3(OMTemp, OrientIn);
      OrientMat2Euler(OrientIn, EulerIn);
      FitOrientation(nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots, XG,
                     YG, TiltsOrig, OmegaStart, OmegaStep, px, ybc, zbc, gs,
                     OmegaRanges, NoOfOmegaRanges, BoxSizes, P0, NrPixelsGrid,
                     ObsSpotsInfo, EulerIn, tol, &EulerOutA, &EulerOutB,
                     &EulerOutC, &FracOut, hkls, Thetas, n_hkls, LsdFit,
                     TiltsFit, BCsFit, lsdtol, lsdtolrel, tiltstol, bctola,
                     bctolb);
      if ((1 - FracOut) > BestFrac) {
        BestFrac = 1 - FracOut;
        printf("\nBest fraction till now: %f, Orientation number: %d of %d.\n",
               BestFrac, i + 1, OrientationGoodID);
        printf("Euler angles: %f %f %f, ConfidenceIndex: %f, Before fit: "
               "%f\nTilts: %f %f %f\n",
               EulerOutA, EulerOutB, EulerOutC, 1 - FracOut, OrientMatrix[i][9],
               TiltsFit[0], TiltsFit[1], TiltsFit[2]);
        printf("Orientation Matrix:\n");
        EulBest[0] = EulerOutA;
        EulBest[1] = EulerOutB;
        EulBest[2] = EulerOutC;
        Euler2OrientMat(EulBest, OMBest);
        for (j = 0; j < 3; j++) {
          for (m = 0; m < 3; m++) {
            printf("%f ", OMBest[j][m]);
          }
        }
        printf("\n");
        for (j = 0; j < nLayers; j++) {
          printf("Layer Nr: %d, Lsd: %f, BCs: %f %f\n", j, LsdFit[j],
                 BCsFit[j][0], BCsFit[j][1]);
        }
        for (j = 0; j < nLayers; j++) {
          printf("Lsd %f\n", LsdFit[j]);
        }
        for (j = 0; j < nLayers; j++) {
          printf("BC %f %f\n", BCsFit[j][0], BCsFit[j][1]);
        }
        printf("tx %f\nty %f\ntz %f\n", TiltsFit[0], TiltsFit[1], TiltsFit[2]);
        fflush(stdout);
        if (1 - BestFrac < 0.0001)
          break;
      }
    }
  }
  // Free memory
  FreeMemMatrix(SpotsMat, TotalDiffrSpots);
  FreeMemMatrixInt(NrSpots, NrOrientations);
  free(ObsSpotsInfo);
  FreeMemMatrix(XY, 3);
  FreeMemMatrix(OrientationMatrix, NrOrientations);
  FreeMemMatrix(OrientMatrix, MAX_POINTS_GRID_GOOD);
  end = clock();
  diftotal = ((double)(startthis - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed in reading bin files: %f [s]\n", diftotal);
  diftotal = ((double)(startthis2 - startthis)) / CLOCKS_PER_SEC;
  printf("Time elapsed in reading orientations: %f [s]\n", diftotal);
  diftotal = ((double)(end - startthis2)) / CLOCKS_PER_SEC;
  printf("Time elapsed in comparing diffraction spots: %f [s]\n", diftotal);
  return 0;
}