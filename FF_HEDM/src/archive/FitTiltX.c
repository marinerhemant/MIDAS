//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitTiltX.c
//
//
//  Created by Hemant Sharma on 2014/07/01
//

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

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
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

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

struct my_func_data {
  double Ycen;
  double Zcen;
  double px;
  double Lsd;
  double ty;
  double tz;
  double p0;
  double p1;
  double p2;
  double MaxRad;
  double Ys[4];
  double Zs[4];
};

static inline int sign(double x) {
  if (x < 0)
    return -1;
  else
    return 1;
}

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  double Error;
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  double Ycen = f_data->Ycen;
  double Zcen = f_data->Zcen;
  double Lsd = f_data->Lsd;
  double p0 = f_data->p0;
  double p1 = f_data->p1;
  double p2 = f_data->p2;
  double MaxRad = f_data->MaxRad;
  double px = f_data->px;
  double tx = x[0];
  double ty = f_data->ty;
  double tz = f_data->tz;
  double Ys[4], Zs[4];
  int i;
  for (i = 0; i < 4; i++) {
    Ys[i] = f_data->Ys[i];
    Zs[i] = f_data->Zs[i];
  }
  double txr = deg2rad * tx;
  double tyr = deg2rad * ty;
  double tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  double n0 = 2, n1 = 4, n2 = 2, Yc, Zc, Yt, Zt;
  double Etas[4], Eta, RNorm, Rad, EtaT, DistortFunc, Rcorr, YCorrected[i],
      ZCorrected[i];
  for (i = 0; i < 4; i++) {
    Yc = -(Ys[i] - Ycen) * px;
    Zc = (Zs[i] - Zcen) * px;
    double ABC[3] = {0, Yc, Zc};
    double ABCPr[3];
    MatrixMult(TRs, ABC, ABCPr);
    double XYZ[3] = {Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
    Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
    Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
    RNorm = Rad / MaxRad;
    EtaT = 90 - Eta;
    DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                  (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT)))) +
                  (p2 * (pow(RNorm, n2))) + 1;
    Rcorr = Rad * DistortFunc;
    YCorrected[i] = -Rcorr * sin(deg2rad * Eta);
    ZCorrected[i] = Rcorr * cos(deg2rad * Eta);
    Etas[i] = Eta;
  }
  int Case;
  if (Etas[0] > 0)
    Case = 1;
  else
    Case = 2;
  double ZDiff1, ZDiff2, ZDiff;
  if (Case == 1) {
    ZDiff1 = (ZCorrected[0] + ZCorrected[1]) / 2;
    ZDiff2 = (ZCorrected[2] + ZCorrected[3]) / 2;
    ZDiff = (ZDiff1 + ZDiff2) / 2;
  } else {
    ZDiff1 = (ZCorrected[1] + ZCorrected[2]) / 2;
    ZDiff2 = (ZCorrected[3] + ZCorrected[0]) / 2;
    ZDiff = (ZDiff1 + ZDiff2) / 2;
  }
  for (i = 0; i < 4; i++) {
    ZCorrected[i] = ZCorrected[i] - ZDiff;
    Etas[i] = CalcEtaAngle(YCorrected[i], ZCorrected[i]);
  }
  Error = 0;
  if (Etas[0] > 0)
    Case = 1;
  else
    Case = 2;
  if (Case == 1) {
    Error = fabs(Etas[0] + Etas[3]) + fabs(Etas[1] + Etas[2]);
  } else {
    Error = fabs(Etas[0] + Etas[1]) + fabs(Etas[2] + Etas[3]);
  }
  return Error;
}

void FitTX(double Lsd, double Ycen, double Zcen, double p0, double p1,
           double p2, double MaxRad, double ty, double tz, double px,
           double Ys[4], double Zs[4], double txIn, double *TxFit) {
  struct my_func_data f_data;
  f_data.Lsd = Lsd;
  f_data.Ycen = Ycen;
  f_data.Zcen = Zcen;
  f_data.p0 = p0;
  f_data.p1 = p1;
  f_data.p2 = p2;
  f_data.MaxRad = MaxRad;
  f_data.ty = ty;
  f_data.tz = tz;
  f_data.px = px;
  int i;
  for (i = 0; i < 4; i++) {
    f_data.Ys[i] = Ys[i];
    f_data.Zs[i] = Zs[i];
  }
  unsigned n = 1;
  double xl[n], xu[n], x[n];
  xl[0] = txIn - 5;
  xu[0] = txIn + 5;
  x[0] = txIn;
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function, trp);
  nlopt_set_maxeval(opt, 5000);
  nlopt_set_maxtime(opt, 30);
  nlopt_set_ftol_rel(opt, 1e-5);
  nlopt_set_xtol_rel(opt, 1e-5);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  *TxFit = x[0];
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
  int LowNr;
  double Ycen, Zcen, px, Lsd, ty, tz, p0, p1, p2, MaxRad, txIn, Ys[4], Zs[4],
      Y[4], Z[4], Omegas[4];
  int cntr = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &Ycen, &Zcen);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "tx ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &txIn);
      continue;
    }
    str = "ty ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &ty);
      continue;
    }
    str = "tz ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tz);
      continue;
    }
    str = "p0 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p0);
      continue;
    }
    str = "p1 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p1);
      continue;
    }
    str = "p2 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p2);
      continue;
    }
    str = "RhoD ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRad);
      continue;
    }
    str = "SpotsTX ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf", dummy, &Y[cntr], &Z[cntr], &Omegas[cntr]);
      cntr++;
      continue;
    }
  }
  fclose(fileParam);
  int i;
  double MinOme = 10000, MaxOme = -1000, Mid1 = 10000, Mid2 = -10000;
  int First, Second, Third, Fourth;
  for (i = 0; i < 4; i++) {
    if (Omegas[i] < MinOme) {
      MinOme = Omegas[i];
      First = i;
    }
    if (Omegas[i] > MaxOme) {
      MaxOme = Omegas[i];
      Fourth = i;
    }
  }
  for (i = 0; i < 4; i++) {
    if (i != First && i != Fourth) {
      if (Omegas[i] < Mid1) {
        Mid1 = Omegas[i];
        Second = i;
      }
      if (Omegas[i] > Mid2) {
        Mid2 = Omegas[i];
        Third = i;
      }
    }
  }
  Ys[0] = Y[First];
  Zs[0] = Z[First];
  Ys[1] = Y[Second];
  Zs[1] = Z[Second];
  Ys[2] = Y[Third];
  Zs[2] = Z[Third];
  Ys[3] = Y[Fourth];
  Zs[3] = Z[Fourth];
  for (i = 0; i < 4; i++) {
    printf("%f %f\n", Ys[i], Zs[i]);
  }
  double TxFit;
  FitTX(Lsd, Ycen, Zcen, p0, p1, p2, MaxRad, ty, tz, px, Ys, Zs, txIn, &TxFit);
  printf("Refined TiltX: %10.30f\n", TxFit);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
