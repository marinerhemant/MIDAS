//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitWedge.c
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
  double Lsd;
  double Ys;
  double Zs;
  double Ome1;
  double Ome2;
  double Wavelength;
};

void YsZsCalc(double Lsd, double Ycen, double Zcen, double p0, double p1,
              double p2, double MaxRad, double tx, double ty, double tz,
              double px, double Ys, double Zs, double *YOut, double *ZOut) {
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
  double Eta, RNorm, Rad, EtaT, DistortFunc, Rcorr;
  Yc = -(Ys - Ycen) * px;
  Zc = (Zs - Zcen) * px;
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
  *YOut = -Rcorr * sin(deg2rad * Eta);
  *ZOut = Rcorr * cos(deg2rad * Eta);
}

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  double Error;
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  double Lsd = f_data->Lsd;
  double Omega1 = f_data->Ome1;
  double Omega2 = f_data->Ome2;
  double ysi = f_data->Ys;
  double zsi = f_data->Zs;
  double wl = f_data->Wavelength;
  double wedge = x[0];
  double CosOme = cosd(Omega1);
  double SinOme = sind(Omega1);
  double eta = rad2deg * (atan2(-ysi, zsi));
  double Ring_radius = sqrt((ysi * ysi) + (zsi * zsi));
  double tth = atand(Ring_radius / Lsd);
  double theta = tth / 2;
  double SinTheta = sind(theta);
  double CosTheta = cosd(theta);
  double ds = 2 * SinTheta / wl;
  double CosW = cosd(wedge);
  double SinW = sind(wedge);
  double SinEta = sind(eta);
  double CosEta = cosd(eta);
  double k1 = -ds * SinTheta;
  double k2 = -ds * CosTheta * SinEta;
  double k3 = ds * CosTheta * CosEta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW);
  double k3f = (k3 * CosW) - (k1 * SinW);
  double k2f = k2;
  double G1a = k1f * CosOme + k2f * SinOme;
  double G2a = k2f * CosOme - k1f * SinOme;
  double G3a = k3f;
  double normGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / normGa;
  double g2 = G2a * ds / normGa;
  double g3 = G3a * ds / normGa;
  g1 = -g1;
  g2 = -g2;
  g3 = -g3;
  double Length_G = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(Length_G * Length_G) * (wl / 2);
  double A = (k1i + (g3 * SinW)) / (CosW);
  double a_Sin = (g1 * g1) + (g2 * g2);
  double b_Sin = 2 * A * g2;
  double c_Sin = (A * A) - (g1 * g1);
  double a_Cos = a_Sin;
  double b_Cos = -2 * A * g1;
  double c_Cos = (A * A) - (g2 * g2);
  double Par_Sin = (b_Sin * b_Sin) - (4 * a_Sin * c_Sin);
  double Par_Cos = (b_Cos * b_Cos) - (4 * a_Cos * c_Cos);
  double P_check_Sin = 0;
  double P_check_Cos = 0;
  double P_Sin, P_Cos;
  if (Par_Sin >= 0) {
    P_Sin = sqrt(Par_Sin);
  } else {
    P_Sin = 0;
    P_check_Sin = 1;
  }
  if (Par_Cos >= 0) {
    P_Cos = sqrt(Par_Cos);
  } else {
    P_Cos = 0;
    P_check_Cos = 1;
  }

  double Sin_Omega1 = ((-b_Sin) - (P_Sin)) / (2 * a_Sin);
  double Sin_Omega2 = ((-b_Sin) + (P_Sin)) / (2 * a_Sin);
  double Cos_Omega1 = ((-b_Cos) - (P_Cos)) / (2 * a_Cos);
  double Cos_Omega2 = ((-b_Cos) + (P_Cos)) / (2 * a_Cos);

  if (Sin_Omega1 < -1)
    Sin_Omega1 = 0;
  else if (Sin_Omega1 > 1)
    Sin_Omega1 = 0;
  else if (Sin_Omega2 > 1)
    Sin_Omega2 = 0;
  else if (Sin_Omega2 < -1)
    Sin_Omega2 = 0;
  if (Cos_Omega1 < -1)
    Cos_Omega1 = 0;
  else if (Cos_Omega1 > 1)
    Cos_Omega1 = 0;
  else if (Cos_Omega2 > 1)
    Cos_Omega2 = 0;
  else if (Cos_Omega2 < -1)
    Cos_Omega2 = 0;
  if (P_check_Sin == 1) {
    Sin_Omega1 = 0;
    Sin_Omega2 = 0;
  }
  if (P_check_Cos == 1) {
    Cos_Omega1 = 0;
    Cos_Omega2 = 0;
  }
  double Option_1 =
      fabs((Sin_Omega1 * Sin_Omega1) + (Cos_Omega1 * Cos_Omega1) - 1);
  double Option_2 =
      fabs((Sin_Omega1 * Sin_Omega1) + (Cos_Omega2 * Cos_Omega2) - 1);
  double Omega_1, Omega_2;
  if (Option_1 < Option_2) {
    Omega_1 = rad2deg * (atan2(Sin_Omega1, Cos_Omega1));
    Omega_2 = rad2deg * (atan2(Sin_Omega2, Cos_Omega2));
  } else {
    Omega_1 = rad2deg * (atan2(Sin_Omega1, Cos_Omega2));
    Omega_2 = rad2deg * (atan2(Sin_Omega2, Cos_Omega1));
  }
  double Omega_diff1 = fabs(Omega_1 - Omega2);
  double Omega_diff2 = fabs(Omega_2 - Omega2);
  double OmegaMin = 10000;
  if (Omega_diff1 < OmegaMin) {
    OmegaMin = Omega_diff1;
  }
  if (Omega_diff2 < OmegaMin) {
    OmegaMin = Omega_diff2;
  }
  Error = OmegaMin;
  printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", Omega1, Omega_1,
         Omega_2, fabs(Omega1 - Omega_1), fabs(Omega1 - Omega_2), wedge, G1a,
         G2a, G3a, g1, g2, g3, Length_G, ds, normGa);
  printf("%2.30f\n", Error);
  return Error;
}

void FitWedge(double Lsd, double Ycen, double Zcen, double p0, double p1,
              double p2, double MaxRad, double tx, double ty, double tz,
              double px, double Ys, double Zs, double MinOme, double MaxOme,
              double WedgeIn, double *WedgeFit, double Wavelength) {
  struct my_func_data f_data;
  f_data.Lsd = Lsd;
  f_data.Ome1 = MinOme;
  f_data.Ome2 = MaxOme;
  f_data.Wavelength = Wavelength;
  double Y, Z;
  YsZsCalc(Lsd, Ycen, Zcen, p0, p1, p2, MaxRad, tx, ty, tz, px, Ys, Zs, &Y, &Z);
  f_data.Ys = Y;
  f_data.Zs = Z;
  unsigned n = 1;
  double xl[n], xu[n], x[n];
  xl[0] = WedgeIn - 2;
  xu[0] = WedgeIn + 2;
  x[0] = WedgeIn;
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
  *WedgeFit = x[0];
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
  double Ycen, Zcen, px, Lsd, ty, tz, p0, p1, p2, MaxRad, tx, Ys[2];
  double Wavelength, Zs[2], Y[2], Z[2], Omegas[2], Wedge;
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
      sscanf(aline, "%s %lf", dummy, &tx);
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
    str = "Wedge ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wedge);
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
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "SpotsWedge ";
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
  int First, Second;
  for (i = 0; i < 2; i++) {
    if (Omegas[i] < MinOme) {
      MinOme = Omegas[i];
      First = i;
    }
    if (Omegas[i] > MaxOme) {
      MaxOme = Omegas[i];
      Second = i;
    }
  }
  // print max and min
  printf("Max: %f\n", MaxOme);
  printf("Min: %f\n", MinOme);
  Ys[0] = Y[First];
  Zs[0] = Z[First];
  Ys[1] = Y[Second];
  Zs[1] = Z[Second];
  double WedgeFit;
  FitWedge(Lsd, Ycen, Zcen, p0, p1, p2, MaxRad, tx, ty, tz, px, Ys[0], Zs[0],
           MinOme, MaxOme, Wedge, &WedgeFit, Wavelength);
  printf("Refined Wedge: %10.30f\n", WedgeFit);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
