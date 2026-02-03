//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  Calibrant.c
//
//
//  Created by Hemant Sharma on 2014/06/18.
//
//
//  Important: row major, starting with y's and going up. Bottom right is 0,0.
//  TODO: Implement proper transformations for rectangular detectors.

#include "FileReader.h"
#include "Panel.h"
#include "midas_paths.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static Panel *panels = NULL;
static int nPanels = 0;

// #define PRINTOPT
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef double pixelvalue;
long long int NrCalls;
long long int NrCallsProfiler;
int NrPixelsGlobal = 2048;
#define OBJ_FUNC_SCALE 1
#define EPS 1E-12

int numProcs;
int skipFrame = 0;

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
extern size_t mapMaskSize;
extern int *mapMask;

size_t mapMaskSize = 0;
int *mapMask;

inline void CalcPeakProfile(int **Indices, int *NrEachIndexBin, int idx,
                            double *Average, double Rmi, double Rma,
                            double EtaMi, double EtaMa, double ybc, double zbc,
                            double px, int NrPixelsY, double *ReturnValue);

inline void CalcPeakProfileParallel(int *Indices, int NrEachIndexBin, int idx,
                                    double *Average, double Rmi, double Rma,
                                    double EtaMi, double EtaMa, double ybc,
                                    double zbc, double px, int NrPixelsY,
                                    double *ReturnValue);

static inline pixelvalue **allocMatrixPX(int nrows, int ncols) {
  pixelvalue **arr;
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

static inline void FreeMemMatrixPx(pixelvalue **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
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

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline double R4mTtheta(double Ttheta, double Lsd) {
  return Lsd * tan(deg2rad * Ttheta);
}

static inline double Ttheta4mR(double R, double Lsd) {
  return rad2deg * atan(R / Lsd);
}

static inline void YZ4mREta(int NrElements, double *R, double *Eta, double *Y,
                            double *Z) {
  int i;
  for (i = 0; i < NrElements; i++) {
    Y[i] = -R[i] * sin(Eta[i] * deg2rad);
    Z[i] = R[i] * cos(Eta[i] * deg2rad);
  }
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void Car2Pol(int n_hkls, int nEtaBins, int y, int z, double ybc,
                           double zbc, double px, double *R, double *Eta,
                           double Rmins[n_hkls], double Rmaxs[n_hkls],
                           double EtaBinsLow[nEtaBins],
                           double EtaBinsHigh[nEtaBins], int nIndices,
                           int *NrEachIndexbin, int **Indices, double tx,
                           double ty, double tz, double p0, double p1,
                           double p2, double p3, double RhoD, double Lsd) {
  int i, j, k, l, counter = 0, ctr = 0;
  for (i = 0; i < nIndices; i++)
    NrEachIndexbin[i] = 0;
  double txr, tyr, tzr;
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  double Yc, Zc, n0 = 2, n1 = 4, n2 = 2;
  double ABC[3], ABCPr[3], XYZ[3];
  double Rt, Rad, EtaS, EtaST, RNorm, DistortFunc, EtaT;
  for (i = 0; i < z; i++) {
    for (j = 0; j < y; j++) {
      Yc = (-j + ybc) * px;
      Zc = (i - zbc) * px;
      ABC[0] = 0;
      ABC[1] = Yc;
      ABC[2] = Zc;
      EtaST = CalcEtaAngle(ABC[1], ABC[2]);
      MatrixMult(TRs, ABC, ABCPr);
      XYZ[0] = Lsd + ABCPr[0];
      XYZ[1] = ABCPr[1];
      XYZ[2] = ABCPr[2];
      Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
      EtaS = CalcEtaAngle(XYZ[1], XYZ[2]);
      RNorm = Rad / RhoD;
      EtaT = 90 - EtaS;
      // Reset Eta so that calculations later on are still good.
      EtaS = EtaST;
      DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                    (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                    (p2 * (pow(RNorm, n2))) + 1;
      Rt = Rad * DistortFunc;
      R[counter] = Rt;
      Eta[counter] = EtaS;
      for (k = 0; k < n_hkls; k++) {
        if (Rt >= (Rmins[k] - px) && Rt <= (Rmaxs[k] + px)) {
          for (l = 0; l < nEtaBins; l++) {
            if (EtaS >= (EtaBinsLow[l] - px / R[counter]) &&
                EtaS <= (EtaBinsHigh[l] + px / R[counter])) {
              Indices[(nEtaBins * k) + l][NrEachIndexbin[(nEtaBins * k) + l]] =
                  (i * NrPixelsGlobal) + j;
              NrEachIndexbin[(nEtaBins * k) + l] += 1;
              ctr++;
              break;
            }
          }
          break;
        }
      }
      counter++;
    }
  }
}

static inline void CalcWeightedMean(int nIndices, int *NrEachIndexBin,
                                    int **Indices, double *Average, double *R,
                                    double *Eta, double *RMean,
                                    double *EtaMean) {
  int i, j, k;
  double TotIntensities[nIndices];
  for (i = 0; i < nIndices; i++) {
    TotIntensities[i] = 0;
    EtaMean[i] = 0;
    RMean[i] = 0;
  }
  for (i = 0; i < nIndices; i++) {
    for (j = 0; j < NrEachIndexBin[i]; j++) {
      TotIntensities[i] += Average[Indices[i][j]];
    }
  }
  for (i = 0; i < nIndices; i++) {
    for (j = 0; j < NrEachIndexBin[i]; j++) {
      RMean[i] +=
          (Average[Indices[i][j]] * R[Indices[i][j]]) / TotIntensities[i];
      EtaMean[i] +=
          (Average[Indices[i][j]] * Eta[Indices[i][j]]) / TotIntensities[i];
    }
  }
}

struct my_profile_func_data {
  int NrPtsForFit;
  double *Rs;
  double *PeakShape;
};

static double problem_function_profile(unsigned n, const double *x,
                                       double *grad, void *f_data_trial) {
  struct my_profile_func_data *f_data =
      (struct my_profile_func_data *)f_data_trial;
  int NrPtsForFit = f_data->NrPtsForFit;
  double *Rs, *PeakShape;
  Rs = &(f_data->Rs[0]);
  PeakShape = &(f_data->PeakShape[0]);
  double Rcen, Mu, SigmaG, SigmaL, Imax, BG;
  Rcen = x[0];
  Mu = x[1];
  SigmaG = x[2];
  SigmaL = x[3];
  Imax = x[4];
  BG = x[5];
  double TotalDifferenceIntensity = 0, CalcIntensity;
  int i, j, k;
  double L, G;
  for (i = 0; i < NrPtsForFit; i++) {
    L = (1 / (((Rs[i] - Rcen) * (Rs[i] - Rcen) / (SigmaL * SigmaL)) + (1)));
    G = (exp((-0.5) * (Rs[i] - Rcen) * (Rs[i] - Rcen) / (SigmaG * SigmaG)));
    CalcIntensity = BG + Imax * ((Mu * L) + ((1 - Mu) * G));
    TotalDifferenceIntensity +=
        (CalcIntensity - PeakShape[i]) * (CalcIntensity - PeakShape[i]);
  }
#pragma omp critical
  {
    NrCallsProfiler++;
  }
#ifdef PRINTOPT
  printf("Peak profiler intensity difference: %f\n", TotalDifferenceIntensity);
#endif
  return TotalDifferenceIntensity;
}

void FitPeakShape(int NrPtsForFit, double Rs[NrPtsForFit],
                  double PeakShape[NrPtsForFit], double *Rfit, double Rstep,
                  double Rmean) {
  unsigned n = 6;
  double x[n], xl[n], xu[n];
  struct my_profile_func_data f_data;
  f_data.NrPtsForFit = NrPtsForFit;
  f_data.Rs = &Rs[0];
  f_data.PeakShape = &PeakShape[0];
  double BG0 = (PeakShape[0] + PeakShape[NrPtsForFit - 1]) / 2;
  if (BG0 < 0)
    BG0 = 0;
  double MaxI = -100000;
  int i;
  for (i = 0; i < NrPtsForFit; i++) {
    if (PeakShape[i] > MaxI) {
      MaxI = PeakShape[i];
    }
  }
  x[0] = Rmean;
  xl[0] = Rs[0];
  xu[0] = Rs[NrPtsForFit - 1];
  x[1] = 0.5;
  xl[1] = 0;
  xu[1] = 1;
  x[2] = Rstep;
  xl[2] = Rstep / 2;
  xu[2] = Rstep * NrPtsForFit / 2;
  x[3] = Rstep;
  xl[3] = Rstep / 2;
  xu[3] = Rstep * NrPtsForFit / 2;
  x[4] = MaxI;
  xl[4] = MaxI / 100;
  xu[4] = MaxI * 1.5;
  x[5] = BG0;
  xl[5] = 0;
  xu[5] = BG0 * 1.5;
  struct my_profile_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_profile_func_data *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_profile, trp);
  double minf, MeanDiff;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  MeanDiff = sqrt(minf) / (NrPtsForFit);
  *Rfit = x[0];
}

void CalcFittedMean(int nIndices, int *NrEachIndexBin, int **Indices,
                    double *Average, double *R, double *Eta, double *RMean,
                    double *EtaMean, int NrPtsForFit, double *IdealRmins,
                    double *IdealRmaxs, int nBinsPerRing, double ybc,
                    double zbc, double px, int NrPixels,
                    double EtaBinsLow[nBinsPerRing],
                    double EtaBinsHigh[nBinsPerRing]) {
  int idxThis;
#pragma omp parallel for num_threads(numProcs) private(idxThis)                \
    schedule(dynamic)
  for (idxThis = 0; idxThis < nIndices; idxThis++) {
    int j, k, BinNr;
    double *PeakShape, Rmin, Rmax, Rstep, Rthis, *Rs;
    PeakShape = calloc(NrPtsForFit, sizeof(*PeakShape));
    Rs = calloc(NrPtsForFit, sizeof(*Rs));
    double Rfit = 0;
    int *IndicesThis;
    int NrIndicesThis = NrEachIndexBin[idxThis];
    int **Idxs;
    Idxs = allocMatrixInt(1, NrPtsForFit);
    double Etas[NrPtsForFit];
    double EtaMi, EtaMa, Rmi, Rma;
    double RetVal;
    for (j = 0; j < NrPtsForFit; j++)
      Idxs[0][j] = j;
    double AllZero;
    double ytr, ztr;
    double EtaTempThis, RTempThis;
    // If no pixel inside the detector, ignore this bin
    if (NrIndicesThis == 0) {
      Rfit = 0;
      RMean[idxThis] = Rfit;
      continue;
    }
    IndicesThis = malloc(NrIndicesThis * sizeof(*IndicesThis));
    for (j = 0; j < NrIndicesThis; j++)
      IndicesThis[j] = Indices[idxThis][j];
    Rmin = IdealRmins[idxThis];
    Rmax = IdealRmaxs[idxThis];
    AllZero = 1;
    Rstep = (Rmax - Rmin) / NrPtsForFit;
    BinNr = idxThis % nBinsPerRing;
    EtaMi = -180 + BinNr * (360 / nBinsPerRing);
    EtaMa = -180 + (BinNr + 1) * (360 / nBinsPerRing);
    // Find if either etamin or etamax result in outside the detector, then
    // ignore this bin
    ytr = ybc - (-Rmax * sin(EtaMa * deg2rad)) / px;
    ztr = zbc + (Rmax * cos(EtaMa * deg2rad)) / px;
    if (((int)ytr > NrPixels - 3) || ((int)ytr < 3)) {
      Rfit = 0;
      RMean[idxThis] = Rfit;
      continue;
    }
    if (((int)ztr > NrPixels - 3) || ((int)ztr < 3)) {
      Rfit = 0;
      RMean[idxThis] = Rfit;
      continue;
    }
    ytr = ybc - (-Rmax * sin(EtaMi * deg2rad)) / px;
    ztr = zbc + (Rmax * cos(EtaMi * deg2rad)) / px;
    if (((int)ytr > NrPixels - 3) || ((int)ytr < 3)) {
      Rfit = 0;
      RMean[idxThis] = Rfit;
      continue;
    }
    if (((int)ztr > NrPixels - 3) || ((int)ztr < 3)) {
      Rfit = 0;
      RMean[idxThis] = Rfit;
      continue;
    }
    EtaMean[idxThis] = (EtaMi + EtaMa) / 2;
    for (j = 0; j < NrPtsForFit; j++) {
      PeakShape[j] = 0;
      Rs[j] = (Rmin + (j * Rstep) + Rstep / 2);
      Rmi = Rs[j] - Rstep / 2;
      Rma = Rs[j] + Rstep / 2;
      CalcPeakProfileParallel(IndicesThis, NrIndicesThis, idxThis, Average, Rmi,
                              Rma, EtaMi, EtaMa, ybc, zbc, px, NrPixels,
                              &RetVal);
      // printf("%lf ",RetVal);
      PeakShape[j] = RetVal;
      if (RetVal != 0) {
        AllZero = 0;
      }
    }
    // printf("\n");
    for (j = 0; j < NrPtsForFit; j++) {
      Etas[j] = EtaMean[idxThis];
    }
    double *Rm, *Etam;
    int *NrPts;
    NrPts = malloc(sizeof(*NrPts));
    Rm = malloc(sizeof(*Rm));
    Etam = malloc(sizeof(*Etam));
    NrPts[0] = NrPtsForFit;
    if (AllZero != 1) {
      CalcWeightedMean(1, NrPts, Idxs, PeakShape, Rs, Etas, Rm, Etam);
      double Rmean = Rm[0];
      FitPeakShape(NrPtsForFit, Rs, PeakShape, &Rfit, Rstep, Rmean);
    } else {
      Rfit = 0;
    }
    // printf("\t\t%lf %lf %lf %lf %lf\n",Rmin,Rmax,Rfit,ytr,ztr);
    RMean[idxThis] = Rfit;
    free(NrPts);
    free(Rm);
    free(Etam);
    free(IndicesThis);
    free(PeakShape);
    free(Rs);
    FreeMemMatrixInt(Idxs, 1);
  }
}

struct my_func_data {
  int nIndices;
  double *YMean;
  double *ZMean;
  double *IdealTtheta;
  double MaxRad;
  double px;
  double tx;
  int fixPanel; // Added fixPanel
};

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  int MaxRad = f_data->MaxRad;
  int nIndices = f_data->nIndices;
  double *YMean, *ZMean, *IdealTtheta, px;
  YMean = &(f_data->YMean[0]);
  ZMean = &(f_data->ZMean[0]);
  IdealTtheta = &(f_data->IdealTtheta[0]);
  px = f_data->px;
  double Lsd, ybc, zbc, tx, ty, tz, p0, p1, p2, p3, txr, tyr, tzr;
  Lsd = x[0];
  ybc = x[1];
  zbc = x[2];
  tx = f_data->tx;
  ty = x[3];
  tz = x[4];
  p0 = x[5];
  p1 = x[6];
  p2 = x[7];
  p3 = x[8];
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  int i;
  double TotalDiff = 0;

#pragma omp parallel for reduction(+ : TotalDiff)
  for (i = 0; i < nIndices; i++) {
    double n0 = 2, n1 = 4, n2 = 2, Yc, Zc;
    double Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, EtaT;
    double dY = 0, dZ = 0;
    int pIdx =
        GetPanelIndex((double)YMean[i], (double)ZMean[i], nPanels, panels);
    if (pIdx == -1) {
      continue;
    }

    dY = 0;
    dZ = 0;
    if (n > 9) {
      if (pIdx != f_data->fixPanel) {
        int logicalIndex = (pIdx < f_data->fixPanel) ? pIdx : pIdx - 1;
        int xIdx = 9 + logicalIndex * 2;
        dY = x[xIdx];
        dZ = x[xIdx + 1];
      }
    }
    Yc = -(YMean[i] + dY - ybc) * px;
    Zc = (ZMean[i] + dZ - zbc) * px;
    double ABC[3] = {0, Yc, Zc};
    double ABCPr[3];
    MatrixMult(TRs, ABC, ABCPr);
    double XYZ[3] = {Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
    Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
    Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
    RNorm = Rad / MaxRad;
    EtaT = 90 - Eta;
    DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                  (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                  (p2 * (pow(RNorm, n2))) + 1;
    Rcorr = Rad * DistortFunc;
    RIdeal = Lsd * tan(deg2rad * IdealTtheta[i]);
    Rcorr = Rad * DistortFunc;
    RIdeal = Lsd * tan(deg2rad * IdealTtheta[i]);
    double Diff = fabs(1 - (Rcorr / RIdeal));
    TotalDiff += Diff;
  }
  TotalDiff *= OBJ_FUNC_SCALE;
  NrCalls++;
  // printf("Mean strain: %0.40f\n", TotalDiff / (OBJ_FUNC_SCALE * nIndices));
#ifdef PRINTOPT
  printf("Mean strain: %0.40f\n", TotalDiff / (OBJ_FUNC_SCALE * nIndices));
#endif
  return TotalDiff;
}

void FitTiltBCLsd(int nIndices, double *YMean, double *ZMean,
                  double *IdealTtheta, double Lsd, double MaxRad, double ybc,
                  double zbc, double tx, double tyin, double tzin, double p0in,
                  double p1in, double p2in, double p3in, double *ty, double *tz,
                  double *LsdFit, double *ybcFit, double *zbcFit, double *p0,
                  double *p1, double *p2, double *p3, double *MeanDiff,
                  double tolTilts, double tolLsd, double tolBC, double tolP,
                  double tolP0, double tolP1, double tolP2, double tolP3,
                  double tolShifts, double px, double outlierFactor,
                  int minIndices, int fixPanel) {
  // Look at the possibility of including translations for each of the small
  // panels on a multi-panel detector in the optimization.... Also change
  // CorrectTiltSpatialDistortion to include translations!!!
  unsigned n = 9;
  if (tolShifts > EPS && nPanels > 1) {
    // Instead of n=9, we need n=9+(nPanels-1)*2
    // We anchor Panel 0 to (0,0) to avoid degeneracy with ybc/zbc
    n += (nPanels - 1) * 2;
  }
  struct my_func_data f_data;
  f_data.nIndices = nIndices;
  f_data.YMean = &YMean[0];
  f_data.ZMean = &ZMean[0];
  f_data.IdealTtheta = &IdealTtheta[0];
  f_data.MaxRad = MaxRad;
  f_data.px = px;
  f_data.tx = tx;
  f_data.fixPanel = fixPanel;
  double x[n], xl[n], xu[n];
  x[0] = Lsd;
  xl[0] = Lsd - tolLsd;
  xu[0] = Lsd + tolLsd;
  x[1] = ybc;
  xl[1] = ybc - tolBC;
  xu[1] = ybc + tolBC;
  x[2] = zbc;
  xl[2] = zbc - tolBC;
  xu[2] = zbc + tolBC;
  x[3] = tyin;
  xl[3] = tyin - tolTilts;
  xu[3] = tyin + tolTilts;
  x[4] = tzin;
  xl[4] = tzin - tolTilts;
  xu[4] = tzin + tolTilts;
  x[5] = p0in;
  xl[5] = p0in - tolP0;
  xu[5] = p0in + tolP0;
  x[6] = p1in;
  xl[6] = p1in - tolP1;
  xu[6] = p1in + tolP1;
  x[7] = p2in;
  xl[7] = p2in - tolP2;
  xu[7] = p2in + tolP2;
  x[8] = p3in;
  xl[8] = p3in - tolP3;
  xu[8] = p3in + tolP3;

  if (tolShifts > EPS && nPanels > 1) {
    int panelCounts[nPanels];
    memset(panelCounts, 0, nPanels * sizeof(int));
    for (int i = 0; i < nIndices; i++) {
      int pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
      if (pIdx >= 0 && pIdx < nPanels) {
        panelCounts[pIdx]++;
      }
    }

    for (int i = 0; i < nPanels; i++) {
      if (i == fixPanel)
        continue;
      // We map Panel i to x indices
      // int logicalIndex = (i < fixPanel) ? i : i - 1; // Not needed with
      // direct pointer, but kept for logic if derived Since we are iterating
      // and filling linear buffer x, we can just track p_idx.
    }

    if (nPanels > 1) {
      int p_idx = 9;
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel)
          continue;

        x[p_idx] = panels[i].dY;

        if (panelCounts[i] < minIndices) {
          xl[p_idx] = 0;
          xu[p_idx] = 0;
        } else {
          xl[p_idx] = x[p_idx] - tolShifts;
          xu[p_idx] = x[p_idx] + tolShifts;
        }
        p_idx++;

        x[p_idx] = panels[i].dZ;

        if (panelCounts[i] < minIndices) {
          xl[p_idx] = 0;
          xu[p_idx] = 0;
        } else {
          xl[p_idx] = x[p_idx] - tolShifts;
          xu[p_idx] = x[p_idx] + tolShifts;
        }
        p_idx++;
      }
    }
    // free(panelCounts); // VLA - no free needed
    struct my_func_data *f_datat;
    f_datat = &f_data;
    void *trp = (struct my_func_data *)f_datat;
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
    nlopt_set_lower_bounds(opt, xl);
    nlopt_set_upper_bounds(opt, xu);
    nlopt_set_min_objective(opt, problem_function, trp);
    double minf;
    nlopt_optimize(opt, x, &minf);
    nlopt_destroy(opt);
    *MeanDiff = minf / (OBJ_FUNC_SCALE * nIndices);

    // 1. Update output parameters with optimized values
    *LsdFit = x[0];
    *ybcFit = x[1];
    *zbcFit = x[2];
    *ty = x[3];
    *tz = x[4];
    *p0 = x[5];
    *p1 = x[6];
    *p2 = x[7];
    *p3 = x[8];

    // 2. Update panel shifts if applicable
    // 2. Update panel shifts if applicable
    if (nPanels > 0) {
      // Reset fixed panel
      if (fixPanel >= 0 && fixPanel < nPanels) {
        panels[fixPanel].dY = 0;
        panels[fixPanel].dZ = 0;
      }

      // Update others
      if (tolShifts > EPS && nPanels > 1) {
        int xIdx = 9;
        for (int i = 0; i < nPanels; i++) {
          if (i == fixPanel)
            continue;
          panels[i].dY = x[xIdx++];
          panels[i].dZ = x[xIdx++];
        }
      } else {
        // Reset others if optimization didn't run for them
        for (int i = 0; i < nPanels; i++) {
          if (i == fixPanel)
            continue;
          panels[i].dY = 0;
          panels[i].dZ = 0;
        }
      }
    }

    // 3. Calculate deviations and outlier rejection (using updated params)
    {
      double txr, tyr, tzr;
      txr = deg2rad * tx; // tx is argument
      tyr = deg2rad * (*ty);
      tzr = deg2rad * (*tz);
      double Rx[3][3] = {
          {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
      double Ry[3][3] = {
          {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
      double Rz[3][3] = {
          {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
      double TRint[3][3], TRs[3][3];
      MatrixMultF33(Ry, Rz, TRint);
      MatrixMultF33(Rx, TRint, TRs);

      double *tempDiffs = malloc(nIndices * sizeof(double));
      for (int k = 0; k < nIndices; k++)
        tempDiffs[k] = -1.0;

      double totalSum = 0;
      int validCount = 0;

      int i;
      double n0 = 2, n1 = 4, n2 = 2;
      double Yc, Zc, Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, Diff, EtaT;
      double LsdV = *LsdFit;
      double ybcV = *ybcFit;
      double zbcV = *zbcFit;
      double p0V = *p0;
      double p1V = *p1;
      double p2V = *p2;
      double p3V = *p3;

      for (i = 0; i < nIndices; i++) {
        double dY = 0, dZ = 0;
        int pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
        if (pIdx == -1)
          continue; // Skip invalid points

        if (pIdx >= 0) {
          dY = panels[pIdx].dY;
          dZ = panels[pIdx].dZ;
        }
        Yc = -(YMean[i] + dY - ybcV) * px;
        Zc = (ZMean[i] + dZ - zbcV) * px;
        double ABC[3] = {0, Yc, Zc};
        double ABCPr[3];
        MatrixMult(TRs, ABC, ABCPr);
        double XYZ[3] = {LsdV + ABCPr[0], ABCPr[1], ABCPr[2]};
        Rad = (LsdV / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
        Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
        RNorm = Rad / MaxRad;
        EtaT = 90 - Eta;
        DistortFunc =
            (p0V * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
            (p1V * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3V)))) +
            (p2V * (pow(RNorm, n2))) + 1;
        Rcorr = Rad * DistortFunc;
        RIdeal = LsdV * tan(deg2rad * IdealTtheta[i]);
        Diff = fabs(1 - (Rcorr / RIdeal));
        tempDiffs[i] = Diff;
        totalSum += Diff;
        validCount++;
      }

      double currentMean = (validCount > 0) ? (totalSum / validCount) : 0;

      if (outlierFactor > 0) {
        double threshold = outlierFactor * currentMean;
        double cleanSum = 0;
        int cleanCount = 0;

        for (i = 0; i < nIndices; i++) {
          if (tempDiffs[i] < 0)
            continue;
          if (tempDiffs[i] <= threshold) {
            cleanSum += tempDiffs[i];
            cleanCount++;
          }
        }

        if (cleanCount > 0) {
          *MeanDiff = cleanSum / cleanCount;
          printf(
              "Outlier rejection (Factor %.2f): Excluded %d / %d points. Mean "
              "Strain: %.8f -> %.8f\n",
              outlierFactor, validCount - cleanCount, validCount, currentMean,
              *MeanDiff);
        } else {
          *MeanDiff = currentMean;
        }
      } else {
        // No outlier rejection, but update *MeanDiff to use the correct
        // denominator (validCount)
        *MeanDiff = currentMean;
      }

      free(tempDiffs);
    }
  }
}

static inline void CorrectTiltSpatialDistortion(
    int nIndices, double MaxRad, double *YMean, double *ZMean,
    double *IdealTtheta, double px, double Lsd, double ybc, double zbc,
    double tx, double ty, double tz, double p0, double p1, double p2, double p3,
    double *Etas, double *Diffs, double *RadOuts, double *StdDiff,
    double outlierFactor, int *IsOutlier) {
  double txr, tyr, tzr;
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  int i, j, k;
  double n0 = 2, n1 = 4, n2 = 2, Yc, Zc;
  double Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, EtaT, Diff, MeanDiff = 0;
  int nValidPoints = 0;
  for (i = 0; i < nIndices; i++) {
    double dY = 0, dZ = 0;
    int pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
    if (pIdx == -1) {
      Diffs[i] = -1.0; // Mark as invalid
      continue;
    }
    nValidPoints++;

    if (pIdx >= 0) {
      dY = panels[pIdx].dY;
      dZ = panels[pIdx].dZ;
    }
    Yc = -(YMean[i] + dY - ybc) * px;
    Zc = (ZMean[i] + dZ - zbc) * px;
    double ABC[3] = {0, Yc, Zc};
    double ABCPr[3];
    MatrixMult(TRs, ABC, ABCPr);
    double XYZ[3] = {Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
    Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
    Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
    RNorm = Rad / MaxRad;
    EtaT = 90 - Eta;
    DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                  (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                  (p2 * (pow(RNorm, n2))) + 1;
    Rcorr = Rad * DistortFunc;
    RIdeal = Lsd * tan(deg2rad * IdealTtheta[i]);
    Diff = fabs(1 - (Rcorr / RIdeal));
    Etas[i] = Eta;
    Diffs[i] = Diff;
    MeanDiff += Diff;
    RadOuts[i] = Rcorr;
    // printf("%lf %lf %lf %lf %lf %lf
    // %lf\n",Rad,Lsd,XYZ[0],XYZ[1],XYZ[2],YMean[i],ZMean[i]);
  }
  if (nValidPoints > 0) {
    MeanDiff /= nValidPoints;
  } else {
    MeanDiff = 0;
  }

  // Filter outliers for StdDiff calculation
  double *validDiffs = malloc(nIndices * sizeof(double));
  int validCount = 0;
  double threshold = (outlierFactor > 0)
                         ? (outlierFactor * MeanDiff)
                         : 1e9; // 1e9 effectively no filter if 0

  if (outlierFactor > 0) {
    double newSum = 0;
    for (i = 0; i < nIndices; i++) {
      if (Diffs[i] < 0) {
        if (IsOutlier)
          IsOutlier[i] = 1;
        continue; // Skip invalid
      }
      if (Diffs[i] <= threshold) {
        if (IsOutlier)
          IsOutlier[i] = 0;
        validDiffs[validCount] = Diffs[i];
        newSum += Diffs[i];
        validCount++;
      } else {
        if (IsOutlier)
          IsOutlier[i] = 1;
      }
    }
    if (validCount > 0) {
      double originalMean = MeanDiff;
      MeanDiff = newSum / validCount;
      printf("StdDev Outlier Rejection (Factor %.2f): Excluded %d / %d "
             "points. "
             "Mean Strain: %.8f -> %.8f\n",
             outlierFactor, nValidPoints - validCount, nValidPoints,
             originalMean, MeanDiff);
    }
  } else {
    for (i = 0; i < nIndices; i++) {
      if (Diffs[i] >= 0) {
        if (IsOutlier)
          IsOutlier[i] = 0;
        validDiffs[validCount] = Diffs[i];
        validCount++;
      } else {
        if (IsOutlier)
          IsOutlier[i] = 1;
      }
    }
  }

  double StdDiff2 = 0;
  for (i = 0; i < validCount; i++) {
    StdDiff2 += (validDiffs[i] - MeanDiff) * (validDiffs[i] - MeanDiff);
  }
  *StdDiff = sqrt(StdDiff2 / validCount);
  free(validDiffs);
}

static inline void DoImageTransformations(int NrTransOpt, int TransOpt[10],
                                          pixelvalue *Image, int NrPixels) {
  int i, j, k, l, m;
  pixelvalue **ImageTemp1, **ImageTemp2;
  ImageTemp1 = allocMatrixPX(NrPixels, NrPixels);
  ImageTemp2 = allocMatrixPX(NrPixels, NrPixels);
  if (NrTransOpt == 0) {
    return;
  }
  for (k = 0; k < NrPixels; k++) {
    for (l = 0; l < NrPixels; l++) {
      ImageTemp1[k][l] = Image[(NrPixels * k) + l];
    }
  }
  for (k = 0; k < NrTransOpt; k++) {
    if (TransOpt[k] == 1) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[l][NrPixels - m - 1]; // Inverting Y.
    } else if (TransOpt[k] == 2) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[NrPixels - l - 1][m]; // Inverting Z.
    } else if (TransOpt[k] == 3) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[m][l];
    } else if (TransOpt[k] == 0) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[l][m];
    }
    for (l = 0; l < NrPixels; l++)
      for (m = 0; m < NrPixels; m++)
        ImageTemp1[l][m] = ImageTemp2[l][m];
  }
  for (k = 0; k < NrPixels; k++) {
    for (l = 0; l < NrPixels; l++) {
      Image[(NrPixels * k) + l] = ImageTemp2[k][l];
    }
  }
  FreeMemMatrixPx(ImageTemp1, NrPixels);
  FreeMemMatrixPx(ImageTemp2, NrPixels);
}

static inline void MakeSquare(int NrPixels, int NrPixelsY, int NrPixelsZ,
                              pixelvalue *InImage, pixelvalue *OutImage) {
  int i, j, k;
  if (NrPixelsY == NrPixelsZ) {
    memcpy(OutImage, InImage, NrPixels * NrPixels * sizeof(*InImage));
  } else {
    if (NrPixelsY > NrPixelsZ) { // Filling along the slow direction // easy
      memcpy(OutImage, InImage, NrPixelsY * NrPixelsZ * sizeof(*InImage));
    } else {
      for (i = 0; i < NrPixelsZ; i++) {
        memcpy(OutImage + i * NrPixelsZ, InImage + i * NrPixelsY,
               NrPixelsY * sizeof(*InImage));
      }
    }
  }
}

// The fileReader function is being removed/replaced as part of the update.
// The original fileReader function ended with:
// } else {
//   return 127;
// }
// } // End of fileReader function
// This section is being removed.

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: CalibrantOMP ps.txt nCPUs\n");
    return 1;
  }
  double start, end, start0, end0;
  start0 = omp_get_wtime();
  double diftotal;
  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  numProcs = atoi(argv[2]);
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  char fn[1024], folder[1024], Ext[1024], Dark[1024];
  int StartNr, EndNr, LowNr;
  int SpaceGroup, FitWeightMean = 0;
  double LatticeConstant[6], Wavelength, MaxRingRad, Lsd, MaxTtheta, TthetaTol,
      ybc, zbc, EtaBinSize, px, Width;
  double tx = 0, tolTilts, tolLsd, tolBC, tolP, tolP0 = 0, tolP1 = 0, tolP2 = 0,
         tolP3 = 0, tyin = 0, tzin = 0, p0in = 0, p1in = 0, p2in = 0, p3in = 0,
         padY = 0, padZ = 0;
  double tolShifts = 1.0;
  double outlierFactor = 0.0;
  int MinIndicesForFit = 1;
  int FixPanelID = 0;
  int Padding = 6, NrPixelsY, NrPixelsZ, NrPixels;
  int NrTransOpt = 0, RBinWidth = 4;
  long long int GapIntensity = 0, BadPxIntensity = 0;
  int TransOpt[10], nRingsExclude = 0, RingsExclude[50];
  int makeMap = 0;
  int HeadSize = 8192;
  int dType = 1;
  char GapFN[4096], BadPxFN[4096];
  char darkDatasetName[4096], dataDatasetName[4096];
  // Parameter defaults
  int NPanelsY = 0;
  int NPanelsZ = 0;
  int PanelSizeY = 0;
  int PanelSizeZ = 0;
  int *PanelGapsY = NULL;
  int *PanelGapsZ = NULL;
  char PanelShiftsFile[1024];
  PanelShiftsFile[0] = '\0';
  sprintf(darkDatasetName, "exchange/dark");
  sprintf(dataDatasetName, "exchange/data");
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "FileStem ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn);
      continue;
    }
    str = "darkDataset ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, darkDatasetName);
      continue;
    }
    str = "dataDataset ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, dataDatasetName);
      continue;
    }
    str = "Folder ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, folder);
      continue;
    }
    str = "GapFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, GapFN);
      makeMap = 2;
      continue;
    }
    str = "BadPxFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, BadPxFN);
      makeMap = 2;
      continue;
    }
    str = "DataType ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &dType);
      continue;
    }
    str = "RBinDivisions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RBinWidth);
      continue;
    }
    str = "SkipFrame ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &skipFrame);
      continue;
    }
    str = "GapIntensity ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lld", dummy, &GapIntensity);
      makeMap = 1;
      continue;
    }
    str = "BadPxIntensity ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lld", dummy, &BadPxIntensity);
      makeMap = 1;
      continue;
    }
    str = "Ext ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, Ext);
      continue;
    }
    str = "Dark ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, Dark);
      continue;
    }
    str = "Padding ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &Padding);
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
    str = "ImTransOpt ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &TransOpt[NrTransOpt]);
      NrTransOpt++;
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SpaceGroup);
      continue;
    }
    str = "NPanelsY ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &NPanelsY);
      continue;
    }
    str = "NPanelsZ ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &NPanelsZ);
      continue;
    }
    str = "PanelSizeY ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PanelSizeY);
      continue;
    }
    str = "PanelSizeZ ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PanelSizeZ);
      continue;
    }
    str = "PanelShiftsFile ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %s", dummy, PanelShiftsFile);
      continue;
    }

    str = "PanelGapsY ";
    if (!strncmp(aline, str, strlen(str))) {
      char *ptr = aline + strlen(str);
      if (NPanelsY > 1) {
        PanelGapsY = (int *)malloc((NPanelsY - 1) * sizeof(int));
        for (int k = 0; k < NPanelsY - 1; k++) {
          PanelGapsY[k] = strtol(ptr, &ptr, 10);
        }
      }
      continue;
    }
    str = "PanelGapsZ ";
    if (!strncmp(aline, str, strlen(str))) {
      char *ptr = aline + strlen(str);
      if (NPanelsZ > 1) {
        PanelGapsZ = (int *)malloc((NPanelsZ - 1) * sizeof(int));
        for (int k = 0; k < NPanelsZ - 1; k++) {
          PanelGapsZ[k] = strtol(ptr, &ptr, 10);
        }
      }
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
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "RhoD ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "ty ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tyin);
      continue;
    }
    str = "tz ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tzin);
      continue;
    }
    str = "p0 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p0in);
      continue;
    }
    str = "p1 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p1in);
      continue;
    }
    str = "p2 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p2in);
      continue;
    }
    str = "p3 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p3in);
      continue;
    }
    str = "Width ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Width);
      continue;
    }
    str = "EtaBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &EtaBinSize);
      continue;
    }
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &ybc, &zbc);
      continue;
    }
    str = "tolTilts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTilts);
      continue;
    }
    str = "tolBC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolBC);
      continue;
    }
    str = "tolLsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolLsd);
      continue;
    }
    str = "tolP ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP);
      continue;
    }
    str = "tolP0 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP0);
      continue;
    }
    str = "tolP1 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP1);
      continue;
    }
    str = "tolP2 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP2);
      continue;
    }
    str = "tolP3 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP3);
      continue;
    }
    str = "tolShifts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolShifts);
      continue;
    }
    str = "MultFactor ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &outlierFactor);
      continue;
    }
    str = "MinIndicesForFit ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MinIndicesForFit);
      continue;
    }
    str = "FixPanelID ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &FixPanelID);
      continue;
    }
    str = "tx ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tx);
      continue;
    }
    str = "FitOrWeightedMean ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &FitWeightMean);
      continue;
    }
    str = "RingsToExclude ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsExclude[nRingsExclude]);
      nRingsExclude++;
      continue;
    }
    str = "HeadSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &HeadSize);
    }
  }

  // Generate Panels
  if (NPanelsY > 0 && NPanelsZ > 0) {
    if (GeneratePanels(NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ, PanelGapsY,
                       PanelGapsZ, &panels, &nPanels) != 0) {
      fprintf(stderr, "Fast generation failed.\n");
      return 1;
    }
    printf("Generated %d panels.\n", nPanels);
  }
  if (tolP0 == 0)
    tolP0 = tolP;
  if (tolP1 == 0)
    tolP1 = tolP;
  if (tolP2 == 0)
    tolP2 = tolP;
  if (tolP3 == 0)
    tolP3 = 45;
  if (NrPixelsY > NrPixelsZ) {
    NrPixels = NrPixelsY;
    NrPixelsGlobal = NrPixelsY;
  } else {
    NrPixels = NrPixelsZ;
    NrPixelsGlobal = NrPixelsZ;
  }
  int i, j, k;
  printf("NrTransOpt: %d\n", NrTransOpt);
  for (i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] < 0 || TransOpt[i] > 3) {
      printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
      return 0;
    }
    printf("TransformationOptions: %d ", TransOpt[i]);
    if (TransOpt[i] == 0)
      printf("No change.\n");
    else if (TransOpt[i] == 1)
      printf("Flip Left Right.\n");
    else if (TransOpt[i] == 2)
      printf("Flip Top Bottom.\n");
    else
      printf("Transpose.\n");
  }
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  double Thetas[100];
  int RingIDs[100];
  for (i = 0; i < 100; i++)
    Thetas[i] = 0;
  int n_hkls = 0;

  run_midas_binary("GetHKLList", ParamFN);
  // char cmmd[4096];
  // sprintf(cmmd,"~/opt/MIDAS/FF_HEDM/bin/GetHKLList %s",ParamFN);
  // system(cmmd);
  // Read hkls.csv
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  int tRnr, Exclude, LastRingDone = 0;
  double theta;
  printf("Thetas: ");
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%s %s %s %s %d %s %s %s %lf %s %s", dummy, dummy, dummy,
           dummy, &tRnr, dummy, dummy, dummy, &theta, dummy, dummy);
    if (theta * 2 > MaxTtheta)
      break;
    Exclude = 0;
    for (i = 0; i < nRingsExclude; i++) {
      if (tRnr == RingsExclude[i]) {
        Exclude = 1;
      }
    }
    if (Exclude == 0 && tRnr > LastRingDone) {
      Thetas[n_hkls] = theta;
      RingIDs[n_hkls] = tRnr;
      LastRingDone = tRnr;
      printf("%lf ", theta);
      n_hkls++;
    }
  }
  printf("\n");

  printf("Number of planes being considered: %d.\n", n_hkls);
  printf("The following rings will be excluded:");
  for (i = 0; i < nRingsExclude; i++) {
    printf(" %d", RingsExclude[i]);
  }

  TthetaTol = Ttheta4mR((MaxRingRad + Width), Lsd) -
              Ttheta4mR((MaxRingRad - Width), Lsd);
  printf("\n2Theta Tolerance: %f \n", TthetaTol);
  pixelvalue *DarkFile;
  pixelvalue *DarkFile2;
  double *AverageDark;
  size_t pxSize;
  if (dType == 1) { // Uint16
    pxSize = sizeof(uint16_t);
  } else if (dType == 2) { // Double
    pxSize = sizeof(double);
  } else if (dType == 3) { // Float
    pxSize = sizeof(float);
  } else if (dType == 4) { // Uint32
    pxSize = sizeof(uint32_t);
  } else if (dType == 5) { // Int32
    pxSize = sizeof(int32_t);
  } else if (dType == 6) { // Tiff Uint32
    pxSize = sizeof(uint32_t);
    HeadSize = 0;
  } else if (dType == 7) { // Tiff Uint8
    pxSize = sizeof(uint8_t);
    HeadSize = 0;
  } else if (dType == 8) { // HDF Unit16
    pxSize = sizeof(uint16_t);
    HeadSize = 0;
  } else if (dType == 9) { // Tiff Unit16
    pxSize = sizeof(uint16_t);
    HeadSize = 0;
  }
  size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
  size_t sz;
  char FileName[1024];
  size_t Skip;
  FILE *fp, *fd;
  int nFrames, TotFrames = 0;
  double *Average;
  pixelvalue *Image;
  pixelvalue *Image2;
  DarkFile = malloc(NrPixelsY * NrPixelsZ * sizeof(*DarkFile));    // Raw.
  Image = malloc(NrPixelsY * NrPixelsZ * sizeof(*Image));          // Raw.
  DarkFile2 = calloc(NrPixels * NrPixels, sizeof(*DarkFile2));     // Squared.
  Image2 = calloc(NrPixels * NrPixels, sizeof(*Image2));           // Squared.
  AverageDark = calloc(NrPixels * NrPixels, sizeof(*AverageDark)); // Squared.
  Average = calloc(NrPixels * NrPixels, sizeof(*Average));         // Squared.
  fd = fopen(Dark, "rb");

  int rc;
  char *dname;
  if (fd == NULL && dType != 8) {
    printf("Dark file %s could not be read. Making an empty array for dark.\n",
           Dark);
    for (j = 0; j < (NrPixels * NrPixels); j++)
      AverageDark[j] = 0;
  } else {
    if (dType != 8) {
      dname = "/";
      fseek(fd, 0L, SEEK_END);
      sz = ftell(fd);
      sz -= HeadSize;
      rewind(fd);
      nFrames = sz / (SizeFile);
      Skip = HeadSize;
      printf("Reading dark file:      %s, nFrames: %d, skipping first %ld "
             "bytes.\n",
             Dark, nFrames, Skip);
      fseek(fd, Skip, SEEK_SET);
      for (i = 0; i < nFrames; i++) {
        if (dType == 6 || dType == 7 || dType == 9) {
          rc = ReadTiffFrame(Dark, dType, NrPixelsY * NrPixelsZ, DarkFile, i);
        } else {
          rc = ReadBinaryFrame(fd, dType, NrPixelsY * NrPixelsZ, DarkFile);
        }
        MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, DarkFile, DarkFile2);
        DoImageTransformations(NrTransOpt, TransOpt, DarkFile2, NrPixels);
        if (makeMap == 1) {
          size_t badPxCounter = 0;
          mapMaskSize = NrPixels;
          mapMaskSize *= NrPixels;
          mapMaskSize /= 32;
          mapMaskSize++;
          mapMask = calloc(mapMaskSize, sizeof(*mapMask));
          for (j = 0; j < NrPixels * NrPixels; j++) {
            if (DarkFile2[j] == (pixelvalue)GapIntensity ||
                DarkFile2[j] == (pixelvalue)BadPxIntensity) {
              badPxCounter++;
              SetBit(mapMask, j);
            }
          }
          makeMap = 0;
          printf("%lld\n", (long long int)badPxCounter);
        }
        for (j = 0; j < (NrPixels * NrPixels); j++)
          AverageDark[j] += DarkFile2[j];
      }
      printf("Dark file read.\n");
      for (j = 0; j < (NrPixels * NrPixels); j++)
        AverageDark[j] = AverageDark[j] / nFrames;
      fclose(fd);
    }
  }
  if (makeMap == 2) {
    mapMaskSize = NrPixels;
    mapMaskSize *= NrPixels;
    mapMaskSize /= 32;
    mapMaskSize++;
    mapMask = calloc(mapMaskSize, sizeof(*mapMask));
    double *mapper;
    mapper = calloc(NrPixelsY * NrPixelsZ, sizeof(*mapper));
    double *mapperSquare;
    mapperSquare = calloc(NrPixels * NrPixels, sizeof(*mapperSquare));
    ReadTiffFrame(GapFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, mapper, mapperSquare);
    DoImageTransformations(NrTransOpt, TransOpt, mapperSquare, NrPixels);
    for (i = 0; i < NrPixels * NrPixels; i++) {
      if (mapperSquare[i] == 1) {
        SetBit(mapMask, i);
        mapperSquare[i] = 0;
      }
    }
    ReadTiffFrame(BadPxFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, mapper, mapperSquare);
    DoImageTransformations(NrTransOpt, TransOpt, mapperSquare, NrPixels);
    for (i = 0; i < NrPixels * NrPixels; i++) {
      if (mapperSquare[i] == 1) {
        SetBit(mapMask, i);
        mapperSquare[i] = 0;
      }
    }
  }
  int a;
  double means[11];
  for (a = 0; a < 11; a++)
    means[a] = 0;
  for (a = StartNr; a <= EndNr; a++) {
    start = omp_get_wtime();
    sprintf(FileName, "%s/%s_%0*d%s", folder, fn, Padding, a, Ext);
    if (dType != 8) {
      fp = fopen(FileName, "rb");
      if (fp == NULL) {
        printf("File %s could not be read. Continuing to next one.\n",
               FileName);
        continue;
      }
      fseek(fp, 0L, SEEK_END);
      sz = ftell(fp);
      sz = sz - HeadSize;
      nFrames = sz / (SizeFile);
      Skip = HeadSize;
      printf("Reading calibrant file: %s, nFrames: %d %d %d, skipping first "
             "%ld bytes.\n",
             FileName, nFrames, (int)sz, (int)SizeFile, Skip);
      rewind(fp);
      fseek(fp, Skip, SEEK_SET);
      for (j = 0; j < nFrames; j++) {
        if (dType == 6 || dType == 7 || dType == 9) {
          rc = ReadTiffFrame(FileName, dType, NrPixelsY * NrPixelsZ, Image, j);
        } else {
          rc = ReadBinaryFrame(fp, dType, NrPixelsY * NrPixelsZ, Image);
        }
        MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, Image, Image2);
        DoImageTransformations(NrTransOpt, TransOpt, Image2, NrPixels);
        for (k = 0; k < (NrPixels * NrPixels); k++) {
          Average[k] +=
              (double)(Image2[k]) - AverageDark[k]; // In reality this is sum
        }
      }
      TotFrames += nFrames;
      fclose(fp);
    } else {
      printf("Reading HDF5.\n");
      printf("%s\n", FileName);
      // sprintf(dname,"%s",darkDatasetName);
      // printf("%s\n",dname);
      // return 1;
      dname = "exchange/dark";
      rc = SumHDF5Frames(FileName, dname, NrPixelsY * NrPixelsZ, DarkFile,
                         skipFrame);
      MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, DarkFile, DarkFile2);
      DoImageTransformations(NrTransOpt, TransOpt, DarkFile2, NrPixels);
      if (makeMap == 1) {
        size_t badPxCounter = 0;
        mapMaskSize = NrPixels;
        mapMaskSize *= NrPixels;
        mapMaskSize /= 32;
        mapMaskSize++;
        mapMask = calloc(mapMaskSize, sizeof(*mapMask));
        for (j = 0; j < NrPixels * NrPixels; j++) {
          if (DarkFile2[j] == (pixelvalue)GapIntensity ||
              DarkFile2[j] == (pixelvalue)BadPxIntensity) {
            badPxCounter++;
            SetBit(mapMask, j);
          }
        }
        makeMap = 0;
        printf("%lld\n", (long long int)badPxCounter);
      }
      printf("Dark file read.\n");
      for (j = 0; j < (NrPixels * NrPixels); j++)
        AverageDark[j] = DarkFile2[j];
      // sprintf(dname,"%s",dataDatasetName);
      dname = "exchange/data";
      rc = SumHDF5Frames(FileName, dname, NrPixelsY * NrPixelsZ, Image,
                         skipFrame);
      MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, Image, Image2);
      DoImageTransformations(NrTransOpt, TransOpt, Image2, NrPixels);
      for (j = 0; j < (NrPixels * NrPixels); j++)
        Average[j] = Image2[j] - AverageDark[j];
      //~ uint16_t *outMatrix;
      //~ outMatrix = calloc(NrPixels*NrPixels,sizeof(uint16_t));
      //~ for (j=0;j<(NrPixels*NrPixels);j++)outMatrix[j] = (uint16_t)
      // Average[j]; ~ FILE *fnew; ~ fnew = fopen("Data_000001.ge3","wb"); ~
      // fwrite(outMatrix,sizeof(uint16_t)*NrPixels*NrPixels,1,fnew); ~
      // fclose(fnew); ~ return;
    }
    double IdealTthetas[n_hkls], TthetaMins[n_hkls], TthetaMaxs[n_hkls];
    for (i = 0; i < n_hkls; i++) {
      IdealTthetas[i] = 2 * Thetas[i];
      TthetaMins[i] = IdealTthetas[i] - TthetaTol;
      TthetaMaxs[i] = IdealTthetas[i] + TthetaTol;
    }
    double IdealRs[n_hkls], Rmins[n_hkls], Rmaxs[n_hkls];
    for (i = 0; i < n_hkls; i++) {
      IdealRs[i] = R4mTtheta(IdealTthetas[i], Lsd);
      Rmins[i] = R4mTtheta(TthetaMins[i], Lsd);
      Rmaxs[i] = R4mTtheta(TthetaMaxs[i], Lsd);
    }
    int nEtaBins;
    nEtaBins = (int)ceil(359.99 / EtaBinSize);
    printf("Number of eta bins: %d.\n", nEtaBins);
    double EtaBinsLow[nEtaBins], EtaBinsHigh[nEtaBins];
    for (i = 0; i < nEtaBins; i++) {
      EtaBinsLow[i] = EtaBinSize * i - 179.995;
      EtaBinsHigh[i] = EtaBinSize * (i + 1) - 179.995;
    }
    double *R, *Eta;
    R = malloc(NrPixels * NrPixels * sizeof(*R));
    Eta = malloc(NrPixels * NrPixels * sizeof(*Eta));
    int **Indices, nIndices;
    nIndices = nEtaBins * n_hkls;
    int *NrEachIndexBin, *etaBinNr;
    NrEachIndexBin = malloc(nIndices * sizeof(*NrEachIndexBin));
    Indices = allocMatrixInt(nIndices, 20000);
    Car2Pol(n_hkls, nEtaBins, NrPixels, NrPixels, ybc, zbc, px, R, Eta, Rmins,
            Rmaxs, EtaBinsLow, EtaBinsHigh, nIndices, NrEachIndexBin, Indices,
            tx, tyin, tzin, p0in, p1in, p2in, p3in, MaxRingRad, Lsd);
    double *RMean, *EtaMean, *IdealR, *IdealTtheta, *IdealRmins, *IdealRmaxs;
    IdealR = malloc(nIndices * sizeof(*IdealR));
    IdealRmins = malloc(nIndices * sizeof(*IdealRmins));
    IdealRmaxs = malloc(nIndices * sizeof(*IdealRmaxs));
    IdealTtheta = malloc(nIndices * sizeof(*IdealTtheta));
    int *RingNumbers;
    RingNumbers = malloc(nIndices * sizeof(*RingNumbers));
    RMean = malloc(nIndices * sizeof(*RMean));
    EtaMean = malloc(nIndices * sizeof(*EtaMean));
    int NrPtsForFit;
    NrPtsForFit = (int)((floor)((Rmaxs[0] - Rmins[0]) / px)) * RBinWidth;
    for (i = 0; i < nIndices; i++) {
      IdealR[i] = IdealRs[(int)(floor(i / nEtaBins))];
      IdealRmins[i] = Rmins[(int)(floor(i / nEtaBins))];
      IdealRmaxs[i] = Rmaxs[(int)(floor(i / nEtaBins))];
      IdealTtheta[i] = rad2deg * atan(IdealR[i] / Lsd);
      RingNumbers[i] = RingIDs[(int)(floor(i / nEtaBins))];
    }
    NrCallsProfiler = 0;
    if (FitWeightMean == 1) {
      CalcWeightedMean(nIndices, NrEachIndexBin, Indices, Average, R, Eta,
                       RMean, EtaMean);
    } else {
      CalcFittedMean(nIndices, NrEachIndexBin, Indices, Average, R, Eta, RMean,
                     EtaMean, NrPtsForFit, IdealRmins, IdealRmaxs, nEtaBins,
                     ybc, zbc, px, NrPixels, EtaBinsLow, EtaBinsHigh);
    }
    // Find the RMean, which are 0 and update accordingly.
    int countr = 0;
    double *RMean2, *EtaMean2, *IdealTtheta2;
    int *RingNumbers2;
    RMean2 = malloc(nIndices * sizeof(*RMean2));
    EtaMean2 = malloc(nIndices * sizeof(*EtaMean2));
    IdealTtheta2 = malloc(nIndices * sizeof(*IdealTtheta2));
    RingNumbers2 = malloc(nIndices * sizeof(*RingNumbers2));
    for (i = 0; i < nIndices; i++) {
      if (RMean[i] != 0) {
        RMean2[countr] = RMean[i];
        EtaMean2[countr] = EtaMean[i];
        EtaMean2[countr] = EtaMean[i];
        IdealTtheta2[countr] = IdealTtheta[i];
        RingNumbers2[countr] = RingNumbers[i];
        // printf("%lf %lf %lf
        // %lf\n",RMean[i],IdealR[i],EtaMean[i],IdealTtheta[i]);
        countr++;
      }
    }
    printf("Out of %d slices, %d were in the detector\n", nIndices, countr);
    nIndices = countr;
    free(RMean);
    free(EtaMean);
    free(IdealTtheta);
    free(RingNumbers);
    RMean = RMean2;
    EtaMean = EtaMean2;
    IdealTtheta = IdealTtheta2;
    RingNumbers = RingNumbers2;
    end = omp_get_wtime();
    diftotal = end - start;
    if (FitWeightMean != 1) {
      printf("Number of calls to profiler function: %lld\n", NrCallsProfiler);
      printf("Time elapsed in fitting peak profiles:\t%f s.\n", diftotal);
    } else
      printf("Time elapsed in finding peak positions:\t%f s.\n", diftotal);
    double *YMean, *ZMean;
    YMean = malloc(nIndices * sizeof(*YMean));
    ZMean = malloc(nIndices * sizeof(*ZMean));
    YZ4mREta(nIndices, RMean, EtaMean, YMean, ZMean);
    double ty, tz, LsdFit, ybcFit, zbcFit, p0, p1, p2, p3, MeanDiff, *Yc, *Zc,
        *EtaIns, *RadIns, *DiffIns, StdDiff;
    Yc = malloc(nIndices * sizeof(*Yc));
    Zc = malloc(nIndices * sizeof(*Zc));
    EtaIns = malloc(nIndices * sizeof(*EtaIns));
    RadIns = malloc(nIndices * sizeof(*RadIns));
    DiffIns = malloc(nIndices * sizeof(*DiffIns));
    for (i = 0; i < nIndices; i++) {
      Yc[i] = (ybc - (YMean[i] / px));
      Zc[i] = (zbc + (ZMean[i] / px));
    }
    CorrectTiltSpatialDistortion(nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px,
                                 Lsd, ybc, zbc, tx, tyin, tzin, p0in, p1in,
                                 p2in, p3in, EtaIns, DiffIns, RadIns, &StdDiff,
                                 outlierFactor, NULL);
    NrCalls = 0;
    NrCalls = 0;
    // Count and print indices per panel
    if (nPanels > 0) {
      int *panelCounts = calloc(nPanels, sizeof(int));
      for (int i = 0; i < nIndices; i++) {
        int pIdx = GetPanelIndex(Yc[i], Zc[i], nPanels, panels);
        if (pIdx >= 0) {
          panelCounts[pIdx]++;
        }
      }
      printf("\n******************* Indices per Panel (Visual Layout: Z^ Y>) "
             "*******************\n");
      printf("                        Anchored Panel ID: %d \n", FixPanelID);
      double charAspect = 0.5;         // Width / Height
      double textWidthPerPanel = 14.0; // "|  12 (12345) " is 14 chars
      double visualWidthPoints = NPanelsY * textWidthPerPanel * charAspect;
      double targetHeightPoints =
          visualWidthPoints * ((double)NrPixelsZ / (double)NrPixelsY);
      // Reduce vertical spacing factor significantly (0.15 factor)
      int linesPerRow = (int)(targetHeightPoints / NPanelsZ * 0.15 + 0.5);
      if (linesPerRow < 1)
        linesPerRow = 1;

      for (int z = NPanelsZ - 1; z >= 0; z--) {
        for (int l = 0; l < linesPerRow; l++) {
          if (l == linesPerRow / 2)
            printf("Z=%-2d | ", z);
          else
            printf("     | ");

          if (l == linesPerRow / 2) {
            for (int y = 0; y < NPanelsY; y++) {
              int pIdx = y * NPanelsZ + z;
              if (pIdx < nPanels) {
                printf("| %3d (%5d) ", pIdx, panelCounts[pIdx]);
              } else {
                printf("|             ");
              }
            }
            printf("|"); // Closing pipe
          }
          printf("\n");
        }
      }
      printf("       ");
      for (int y = 0; y < NPanelsY; y++) {
        // Data block is 14 chars: "| %3d (%5d) "
        // We want Y label centered: 5 spaces + "Y=%-2d" (4) + 5 spaces = 14
        printf("     Y=%-2d     ", y);
      }
      printf("\n");
      printf("*****************************************************************"
             "****"
             "***********\n\n");
      free(panelCounts);
    }
    FitTiltBCLsd(nIndices, Yc, Zc, IdealTtheta, Lsd, MaxRingRad, ybc, zbc, tx,
                 tyin, tzin, p0in, p1in, p2in, p3in, &ty, &tz, &LsdFit, &ybcFit,
                 &zbcFit, &p0, &p1, &p2, &p3, &MeanDiff, tolTilts, tolLsd,
                 tolBC, tolP, tolP0, tolP1, tolP2, tolP3, tolShifts, px,
                 outlierFactor, MinIndicesForFit, FixPanelID);
    printf("Number of function calls: %lld\n", NrCalls);
    printf("Lsd %0.12f\nBC %0.12f %0.12f\nty %0.12f\ntz %0.12f\np0 %0.12f\np1 "
           "%0.12f\np2 %0.12f\np3 %0.12f\nMeanStrain %0.12lf\n",
           LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3, MeanDiff);
    double *Etas, *Diffs, *RadOuts;
    Etas = malloc(nIndices * sizeof(*Etas));
    Diffs = malloc(nIndices * sizeof(*Diffs));
    RadOuts = malloc(nIndices * sizeof(*RadOuts));
    int *IsOutlier = calloc(nIndices, sizeof(int));
    CorrectTiltSpatialDistortion(nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px,
                                 LsdFit, ybcFit, zbcFit, tx, ty, tz, p0, p1, p2,
                                 p3, Etas, Diffs, RadOuts, &StdDiff,
                                 outlierFactor, IsOutlier);
    printf("StdStrain %0.12lf\n", StdDiff);
    means[0] += LsdFit;
    means[1] += ybcFit;
    means[2] += zbcFit;
    means[3] += ty;
    means[4] += tz;
    means[5] += p0;
    means[6] += p1;
    means[7] += p2;
    means[8] += p3;
    means[9] += MeanDiff;
    means[10] += StdDiff;
    FILE *Out;
    char OutFileName[1024];
    sprintf(OutFileName, "%s/%s_%0*d%s.%s", folder, fn, Padding, a, Ext,
            "corr.csv");
    Out = fopen(OutFileName, "w");
    fprintf(Out, "%%Eta Strain RadFit EtaCalc DiffCalc RadCalc Ideal2Theta "
                 "Outlier YRawCorr ZRawCorr\n");
    for (i = 0; i < nIndices; i++) {
      if (Diffs[i] < 0)
        continue;
      double dY = 0, dZ = 0;
      int pIdx = GetPanelIndex(Yc[i], Zc[i], nPanels, panels);
      if (pIdx >= 0) {
        dY = panels[pIdx].dY;
        dZ = panels[pIdx].dZ;
      }
      double YRawCorr = Yc[i] + dY;
      double ZRawCorr = Zc[i] + dZ;
      fprintf(Out, "%f %10.8f %10.8f %f %10.8f %10.8f %f %d %f %f %d\n",
              Etas[i], Diffs[i], RadOuts[i], EtaIns[i], DiffIns[i], RadIns[i],
              IdealTtheta[i], IsOutlier[i], YRawCorr, ZRawCorr, RingNumbers[i]);
    }
    fclose(Out);
    FreeMemMatrixInt(Indices, nIndices);
    free(R);
    free(Eta);
    free(NrEachIndexBin);
    free(IdealR);
    free(IdealRmins);
    free(IdealRmaxs);
    free(IdealTtheta);
    free(RMean);
    free(EtaMean);
    free(YMean);
    free(ZMean);
    free(Diffs);
    free(Etas);
    free(IsOutlier);
    free(RingNumbers);
    end = omp_get_wtime();
    diftotal = end - start;
    printf("Time elapsed for this file:\t%f s.\n", diftotal);
  }
  end0 = omp_get_wtime();
  diftotal = end0 - start0;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  printf("*******************Mean Values*******************\n");
  for (a = 0; a < 11; a++)
    means[a] /= (EndNr - StartNr + 1);
  printf("Lsd %0.12f\nBC %0.12f %0.12f\nty %0.12f\ntz %0.12f\np0 %0.12f\np1 "
         "%0.12f\np2 %0.12f\np3 %0.12f\nMeanStrain %0.12lf\nStdStrain  "
         "%0.12lf\n",
         means[0], means[1], means[2], means[3], means[4], means[5], means[6],
         means[7], means[8], means[9], means[10]);
  printf("*******************Copy to par*******************\n");
  free(DarkFile);
  free(AverageDark);
  free(Average);
  free(Image);

  // Save Panel Shifts
  if (PanelShiftsFile[0] != '\0' && nPanels > 0) {
    SavePanelShifts(PanelShiftsFile, nPanels, panels);
    printf("Saved panel shifts to %s\n", PanelShiftsFile);
  }
  return 0;
}
