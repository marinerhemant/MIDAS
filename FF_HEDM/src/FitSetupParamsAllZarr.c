//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitTiltBCLsdSample.c
//
//
//  Created by Hemant Sharma on 2024/02/27.
//
//
//  Important: row major, starting with y's and going up. Bottom right is 0,0.

#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <zip.h>

//~ #define PRINTOPT
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef uint16_t pixelvalue;
int NrCalls;
#define MultFactor 1
#define MaxNSpots 2000000

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

static inline void CalcWeightedMean(int nIndices, int *NrEachIndexBin,
                                    int **Indices, double *Average, double *R,
                                    double *Eta, double *RMean,
                                    double *EtaMean) {
  int i, j, k;
  double TotIntensities[nIndices];
  for (i = 0; i < nIndices; i++)
    TotIntensities[i] = 0;
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

struct my_func_data {
  int nIndices;
  double *YMean;
  double *ZMean;
  double *IdealTtheta;
  double MaxRad;
  double px;
  double tx;
  double p0;
  double p1;
  double p2;
  double p3;
};

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  double MaxRad = f_data->MaxRad;
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
  p0 = f_data->p0;
  p1 = f_data->p1;
  p2 = f_data->p2;
  p3 = f_data->p3;
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
  double Rad, Eta, RNorm, DistortFunc, Rcorr, Theta, Diff, IdealTheta,
      TotalDiff = 0, RIdeal, EtaT;
  double MeanErrorEtas[72];
  int nMeanErrorEtas[72];
  for (i = 0; i < 72; i++)
    MeanErrorEtas[i] = 0;
  for (i = 0; i < 72; i++)
    nMeanErrorEtas[i] = 0;
  int idx;
  for (i = 0; i < nIndices; i++) {
    Yc = -(YMean[i] - ybc) * px;
    Zc = (ZMean[i] - zbc) * px;
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
    //~ #ifdef PRINTOPT
    //~ printf("%lf %lf %lf\n",Lsd,IdealTtheta[i],Diff);
    //~ #endif
    //~ TotalDiff+=Diff;
    idx = (Eta + 180) / 5;
    MeanErrorEtas[idx] += Diff;
    nMeanErrorEtas[idx]++;
  }
  TotalDiff = 0;
  for (i = 0; i < 72; i++) {
    if (nMeanErrorEtas[i] != 0) {
      MeanErrorEtas[i] /= (double)nMeanErrorEtas[i];
      TotalDiff += MeanErrorEtas[i];
    }
  }
  TotalDiff *= MultFactor;
  NrCalls++;
#ifdef PRINTOPT
  printf("Mean Strain: %0.40f ty: %lf tz: %lf bc: %lf %lf Lsd: %lf\n",
         TotalDiff / (MultFactor * nIndices), ty, tz, ybc, zbc, Lsd);
#endif
  return TotalDiff;
}

void FitTiltBCLsd(int nIndices, double *YMean, double *ZMean,
                  double *IdealTtheta, double Lsd, double MaxRad, double ybc,
                  double zbc, double tx, double tyIn, double tzIn, double *ty,
                  double *tz, double *LsdFit, double *ybcFit, double *zbcFit,
                  double p0, double p1, double p2, double p3, double *MeanDiff,
                  double tolTilts, double tolLsd, double tolBC, double px) {
  unsigned n = 5;
  struct my_func_data f_data;
  f_data.nIndices = nIndices;
  f_data.YMean = &YMean[0];
  f_data.ZMean = &ZMean[0];
  f_data.IdealTtheta = &IdealTtheta[0];
  f_data.MaxRad = MaxRad;
  f_data.px = px;
  f_data.tx = tx;
  f_data.p0 = p0;
  f_data.p1 = p1;
  f_data.p2 = p2;
  f_data.p3 = p3;
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
  x[3] = tyIn;
  xl[3] = tyIn - tolTilts;
  xu[3] = tyIn + tolTilts;
  x[4] = tzIn;
  xl[4] = tzIn - tolTilts;
  xu[4] = tzIn + tolTilts;
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
  *MeanDiff = minf / (MultFactor * nIndices);
  *LsdFit = x[0];
  *ybcFit = x[1];
  *zbcFit = x[2];
  *ty = x[3];
  *tz = x[4];
}

static inline void
CorrectTiltSpatialDistortion(int nIndices, double MaxRad, double *YMean,
                             double *ZMean, double px, double Lsd, double ybc,
                             double zbc, double tx, double ty, double tz,
                             double p0, double p1, double p2, double p3,
                             double *YCorrected, double *ZCorrected) {
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
  double Rad, Eta, RNorm, DistortFunc, Rcorr, EtaT;
  for (i = 0; i < nIndices; i++) {
    Yc = -(YMean[i] - ybc) * px;
    Zc = (ZMean[i] - zbc) * px;
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
    YCorrected[i] = -Rcorr * sin(deg2rad * Eta);
    ZCorrected[i] = Rcorr * cos(deg2rad * Eta);
  }
}

static inline void CorrectWedge(double yc, double zc, double Lsd,
                                double OmegaIni, double wl, double wedge,
                                double *ysOut, double *zsOut, double *OmegaOut,
                                double *EtaOut, double *TthetaOut) {
  double ysi = yc, zsi = zc;
  double CosOme = cos(deg2rad * OmegaIni), SinOme = sin(deg2rad * OmegaIni);
  double eta = CalcEtaAngle(ysi, zsi);
  double RingRadius = sqrt((ysi * ysi) + (zsi * zsi));
  double tth = rad2deg * atan(RingRadius / Lsd);
  double theta = tth / 2;
  double SinTheta = sin(deg2rad * theta);
  double CosTheta = cos(deg2rad * theta);
  double ds = 2 * SinTheta / wl;
  double CosW = cos(deg2rad * wedge);
  double SinW = sin(deg2rad * wedge);
  double SinEta = sin(deg2rad * eta);
  double CosEta = cos(deg2rad * eta);
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
  double k2f = k2;
  double k3f = (k3 * CosW) - (k1 * SinW);
  double G1a = (k1f * CosOme) + (k2f * SinOme);
  double G2a = (k2f * CosOme) - (k1f * SinOme);
  double G3a = k3f;
  double LenGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / LenGa;
  double g2 = G2a * ds / LenGa;
  double g3 = G3a * ds / LenGa;
  SinW = 0;
  CosW = 1;
  double LenG = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(LenG * LenG * wl) / 2;
  tth = 2 * rad2deg * asin(wl * LenG / 2);
  RingRadius = Lsd * tan(deg2rad * tth);
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
  if (Par_Sin >= 0)
    P_Sin = sqrt(Par_Sin);
  else {
    P_Sin = 0;
    P_check_Sin = 1;
  }
  if (Par_Cos >= 0)
    P_Cos = sqrt(Par_Cos);
  else {
    P_Cos = 0;
    P_check_Cos = 1;
  }
  double SinOmega1 = (-b_Sin - P_Sin) / (2 * a_Sin);
  double SinOmega2 = (-b_Sin + P_Sin) / (2 * a_Sin);
  double CosOmega1 = (-b_Cos - P_Cos) / (2 * a_Cos);
  double CosOmega2 = (-b_Cos + P_Cos) / (2 * a_Cos);
  if (SinOmega1 < -1)
    SinOmega1 = -1;
  else if (SinOmega1 > 1)
    SinOmega1 = 1;
  else if (SinOmega2 < -1)
    SinOmega2 = -1;
  else if (SinOmega2 > 1)
    SinOmega2 = 1;
  if (CosOmega1 < -1)
    CosOmega1 = -1;
  else if (CosOmega1 > 1)
    CosOmega1 = 1;
  else if (CosOmega2 < -1)
    CosOmega2 = -1;
  else if (CosOmega2 > 1)
    CosOmega2 = 1;
  if (P_check_Sin == 1) {
    SinOmega1 = 0;
    SinOmega2 = 0;
  }
  if (P_check_Cos == 1) {
    CosOmega1 = 0;
    CosOmega2 = 0;
  }
  double Option1 = fabs((SinOmega1 * SinOmega1) + (CosOmega1 * CosOmega1) - 1);
  double Option2 = fabs((SinOmega1 * SinOmega1) + (CosOmega2 * CosOmega2) - 1);
  double Omega1, Omega2;
  if (Option1 < Option2) {
    Omega1 = rad2deg * atan2(SinOmega1, CosOmega1);
    Omega2 = rad2deg * atan2(SinOmega2, CosOmega2);
  } else {
    Omega1 = rad2deg * atan2(SinOmega1, CosOmega2);
    Omega2 = rad2deg * atan2(SinOmega2, CosOmega1);
  }
  double OmeDiff1 = fabs(Omega1 - OmegaIni);
  double OmeDiff2 = fabs(Omega2 - OmegaIni);
  if (fabs(OmeDiff1 - 360) < 0.1) {
    OmeDiff1 = 0;
    Omega1 *= -1;
  }
  if (fabs(OmeDiff2 - 360) < 0.1) {
    OmeDiff2 = 0;
    Omega2 *= -1;
  }
  double Omega;
  if (OmeDiff1 < OmeDiff2)
    Omega = Omega1;
  else
    Omega = Omega2;
  double SinOmega = sin(deg2rad * Omega);
  double CosOmega = cos(deg2rad * Omega);
  double Fact = (g1 * CosOmega) - (g2 * SinOmega);
  double k2N = (g1 * SinOmega) + (g2 * CosOmega);
  double k3N = (SinW * Fact) + (g3 * CosW);
  double Eta = CalcEtaAngle(k2, k3);
  double Sin_Eta = sin(deg2rad * Eta);
  double Cos_Eta = cos(deg2rad * Eta);
  *ysOut = -RingRadius * Sin_Eta;
  *zsOut = RingRadius * Cos_Eta;
  *OmegaOut = Omega;
  *EtaOut = Eta;
  *TthetaOut = rad2deg * atan(RingRadius / Lsd);
}

struct SpotsData {
  double SpotID;
  double Omega;
  double Y;
  double Z;
  double RingNr;
  double Radius;
  double IntInt;
};

static int cmpfunc(const void *a, const void *b) {
  struct SpotsData *ia = (struct SpotsData *)a;
  struct SpotsData *ib = (struct SpotsData *)b;
  return (int)(1000.f * ia->Omega - 1000.f * ib->Omega);
}

static inline void SortSpots(int nIndices, double **SpotsInfo) {
  struct SpotsData *MyData;
  MyData = malloc(nIndices * sizeof(*MyData));
  int i, j, k;
  for (i = 0; i < nIndices; i++) {
    MyData[i].SpotID = SpotsInfo[i][0];
    MyData[i].Omega = SpotsInfo[i][1];
    MyData[i].Y = SpotsInfo[i][2];
    MyData[i].Z = SpotsInfo[i][3];
    MyData[i].RingNr = SpotsInfo[i][4];
    MyData[i].Radius = SpotsInfo[i][5];
    MyData[i].IntInt = SpotsInfo[i][6];
  }
  qsort(MyData, nIndices, sizeof(struct SpotsData), cmpfunc);
  for (i = 0; i < nIndices; i++) {
    SpotsInfo[i][0] = MyData[i].SpotID;
    SpotsInfo[i][1] = MyData[i].Omega;
    SpotsInfo[i][2] = MyData[i].Y;
    SpotsInfo[i][3] = MyData[i].Z;
    SpotsInfo[i][4] = MyData[i].RingNr;
    SpotsInfo[i][5] = MyData[i].Radius;
    SpotsInfo[i][6] = MyData[i].IntInt;
  }
  free(MyData);
}

static inline int CheckDirectoryCreation(char Folder[1024]) {
  int e;
  struct stat sb;
  char totOutDir[1024];
  sprintf(totOutDir, "%s/", Folder);
  e = stat(totOutDir, &sb);
  if (e != 0 && errno == ENOENT) {
    printf("Output directory did not exist, creating %s\n", totOutDir);
    e = mkdir(totOutDir, S_IRWXU);
    if (e != 0) {
      printf("Could not make the directory. Exiting\n");
      return 0;
    }
  }
  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s ZarrZip (optional)ResultFolder\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  clock_t start, end;
  double diftotal;
  start = clock();
  char *DataFN = argv[1];
  blosc2_init();
  // Read zarr config
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(DataFN, 0, &errorp);
  if (errorp != NULL)
    return 1;
  struct zip_stat *finfo = NULL;
  finfo = calloc(16384, sizeof(int));
  zip_stat_init(finfo);
  zip_file_t *fd = NULL;
  int count = 0;
  char *data = NULL;
  char *s = NULL;
  char *arr;
  int32_t dsize;
  char *resultFolder;

  char aline[1000];
  char *str, dummy[1000];
  char folder[1024], *Folder = NULL, *spotsfn, outfolder[1024],
                     resultfolder[1024], *idfn;
  int StartNr = 1, EndNr, NrPixels, NrPixelsZ, NrPixelsY, LayerNr;
  double LatticeConstant[6], Wavelength, MaxRingRad, Lsd, MaxTtheta, TthetaTol,
      ybc, zbc, px, tyIn, tzIn, BeamSize = 0;
  double tx, tolTilts = 1, tolLsd = 5000, tolBC = 1, p0, p1, p2, p3, RhoD,
             wedge, MinEta;
  int cs = 0, DoFit = 0, RingToIndex;
  double Rsample, Hbeam, MinMatchesToAcceptFrac = 0, MinOmeSpotIDsToIndex,
                         MaxOmeSpotIDsToIndex, Width = -1, WidthOrig;
  int UseFriedelPairs = 1;
  double t_int = 1, t_gap = 0;
  int TopLayer = 0;
  int maxNFrames = 100000, SGnum = 225;
  spotsfn = "InputAll.csv";
  idfn = "SpotsToIndex.csv";
  double StepSizePos = 5, StepSizeOrient = 0.2, MarginRadius = 500,
         MarginRadial = 500, OmeBinSize = 0.1, EtaBinSize = 0.1,
         MarginEta = 500, MarginOme = 0.5, OmegaStep, MargABC = 2.0,
         MargABG = 2.0;
  int skipFrame = 0;
  int locOmegaRanges, nOmegaRanges = 0;
  int locBoxSizes, nBoxSizes = 0;
  int locRingThresh;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/RingThresh/0.0") != NULL) {
      locRingThresh = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/RingThresh/.zarray") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &cs);
      } else
        return 1;
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = 4096;
      resultFolder = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, resultFolder, dsize);
      resultFolder[dsize] = '\0';
      free(arr);
      // free(data); // Bug fix: decompresses into resultFolder, not data
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p3/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      p3 = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(
            finfo->name,
            "analysis/process/analysis_parameters/MinMatchesToAcceptFrac/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MinMatchesToAcceptFrac = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p2/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      p2 = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p1/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      p1 = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p0/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      p0 = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Wedge/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      wedge = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tz/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tzIn = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/ty/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tyIn = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tx/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tx = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SpaceGroup/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      SGnum = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tolLsd/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tolLsd = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tolBC/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tolBC = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/tolTilts/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      tolTilts = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BeamThickness/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      BeamSize = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaxOmeSpotIDsToIndex/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MaxOmeSpotIDsToIndex = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MinOmeSpotIDsToIndex/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MinOmeSpotIDsToIndex = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/MinEta/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MinEta = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RhoD/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      RhoD = *(double *)&data[0];
      MaxRingRad = RhoD;
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaxRingRad/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MaxRingRad = *(double *)&data[0];
      RhoD = MaxRingRad;
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/LatticeParameter/0") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      int32_t dsize = 6 * sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(s, data, dsize);
      int iter;
      for (iter = 0; iter < 6; iter++)
        LatticeConstant[iter] = *(double *)&data[iter * sizeof(double)];
      free(data);
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OverallRingToIndex/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      RingToIndex = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/DoFit/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      DoFit = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/UseFriedelPairs/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      UseFriedelPairs = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/EtaBinSize/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      EtaBinSize = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OmeBinSize/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      OmeBinSize = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/MargABG/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MargABG = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/MargABC/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MargABC = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MarginOme/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MarginOme = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MarginEta/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MarginEta = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MarginRadial/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MarginRadial = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MarginRadius/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      MarginRadius = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/StepSizeOrient/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      StepSizeOrient = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tGap/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      t_gap = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tInt/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      t_int = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/StepSizePos/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      StepSizePos = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaxNFrames/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      maxNFrames = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BoxSizes/0.0") != NULL) {
      locBoxSizes = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BoxSizes/.zarray") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &nBoxSizes);
      } else
        return 1;
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OmegaRanges/0.0") !=
        NULL) {
      locOmegaRanges = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OmegaRanges/.zarray") !=
        NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &nOmegaRanges);
      } else
        return 1;
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      Lsd = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/Wavelength/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      Wavelength = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Hbeam/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      Hbeam = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Rsample/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      Rsample = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSize/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      px = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/step/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      OmegaStep = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/WidthTthPx/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      Width = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Width/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      WidthOrig = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/YCen/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      ybc = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/ZCen/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(double);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      zbc = *(double *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = 4096;
      Folder = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, Folder, dsize);
      Folder[dsize] = '\0';
      free(arr);
      // free(data); // Bug fix: decompresses into Folder, not data
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "exchange/data/.zarray") != NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
               &EndNr, &NrPixelsZ, &NrPixelsY);
      } else
        return 1;
      free(s);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/LayerNr/0") !=
        NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      LayerNr = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SkipFrame/0") != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      dsize = sizeof(int);
      data = (char *)malloc((size_t)dsize);
      dsize = blosc1_decompress(arr, data, dsize);
      skipFrame = *(int *)&data[0];
      free(arr);
      free(data);
      zip_fclose(fd);
    }
    count++;
  }
  if (Width == -1)
    Width = WidthOrig;
  if (NrPixelsY != NrPixelsZ) {
    if (NrPixelsY > NrPixelsZ) {
      NrPixels = NrPixelsY;
    } else {
      NrPixels = NrPixelsZ;
    }
  } else {
    NrPixels = NrPixelsY;
  }
  if (argc == 3)
    Folder = argv[2];
  if (argc == 3)
    resultFolder = argv[2];
  int i, j, k;
  int RingNumbers[cs];
  zip_stat_index(arch, locRingThresh, 0, finfo);
  s = calloc(finfo->size + 1, sizeof(char));
  fd = zip_fopen_index(arch, locRingThresh, 0);
  zip_fread(fd, s, finfo->size);
  dsize = cs * 2 * sizeof(double);
  data = (char *)malloc((size_t)dsize);
  dsize = blosc1_decompress(s, data, dsize);
  int iter;
  for (iter = 0; iter < cs; iter++) {
    RingNumbers[iter] = (int)*(double *)&data[(iter * 2 + 0) * sizeof(double)];
  }
  free(s);
  free(data);
  zip_fclose(fd);
  EndNr -= skipFrame; // This ensures we don't over-read.
  double BoxSizes[nBoxSizes][4];
  double OmegaRanges[nOmegaRanges][2];
  zip_stat_index(arch, locBoxSizes, 0, finfo);
  s = calloc(finfo->size + 1, sizeof(char));
  fd = zip_fopen_index(arch, locBoxSizes, 0);
  zip_fread(fd, s, finfo->size);
  dsize = nBoxSizes * 4 * sizeof(double);
  data = (char *)malloc((size_t)dsize);
  dsize = blosc1_decompress(s, data, dsize);
  for (iter = 0; iter < nBoxSizes; iter++) {
    BoxSizes[iter][0] = *(double *)&data[(iter * 4 + 0) * sizeof(double)];
    BoxSizes[iter][1] = *(double *)&data[(iter * 4 + 1) * sizeof(double)];
    BoxSizes[iter][2] = *(double *)&data[(iter * 4 + 2) * sizeof(double)];
    BoxSizes[iter][3] = *(double *)&data[(iter * 4 + 3) * sizeof(double)];
  }
  free(s);
  free(data);
  zip_fclose(fd);

  zip_stat_index(arch, locOmegaRanges, 0, finfo);
  s = calloc(finfo->size + 1, sizeof(char));
  fd = zip_fopen_index(arch, locOmegaRanges, 0);
  zip_fread(fd, s, finfo->size);
  dsize = nOmegaRanges * 2 * sizeof(double);
  data = (char *)malloc((size_t)dsize);
  dsize = blosc1_decompress(s, data, dsize);
  for (iter = 0; iter < nOmegaRanges; iter++) {
    OmegaRanges[iter][0] = *(double *)&data[(iter * 2 + 0) * sizeof(double)];
    OmegaRanges[iter][1] = *(double *)&data[(iter * 2 + 1) * sizeof(double)];
  }
  free(s);
  free(data);
  zip_fclose(fd);

  sprintf(folder, "%s", Folder);
  int nOmeRanges = nOmegaRanges;
  if (nOmeRanges != nBoxSizes) {
    printf(
        "Number of omega ranges and number of box sizes don't match. Exiting!");
    return 1;
  }
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  double **SpotsInfo;
  SpotsInfo = allocMatrix(MaxNSpots, 7);
  char FileName[1024];
  if (TopLayer == 1) {
    for (i = 0; i < nBoxSizes; i++) {
      BoxSizes[i][3] = 0;
    }
  }
  FILE *fp;
  // sprintf(FileName,"%s/%s",folder,fn);
  sprintf(folder, "%s/", Folder);
  sprintf(FileName, "%s/Radius_StartNr_%d_EndNr_%d.csv", Folder, StartNr,
          EndNr);
  char line[5024];
  char hklfn[2048];
  sprintf(hklfn, "%s/hkls.csv", resultFolder);
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  int Rnr;
  double tht;
  int PlaneNumbers[cs], donePlanes[cs];
  double Thetas[cs];
  double RingRadsIdeal[cs], ds[cs], rrdideal, dsthis;
  int n_hkls = cs, nhkls = 0;
  for (i = 0; i < cs; i++)
    donePlanes[i] = 0;
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%s %s %s %lf %d %s %s %s %lf %s %lf", dummy, dummy, dummy,
           &dsthis, &Rnr, dummy, dummy, dummy, &tht, dummy, &rrdideal);
    if (tht > MaxTtheta / 2)
      break;
    for (i = 0; i < cs; i++) {
      if (Rnr == RingNumbers[i] && donePlanes[i] == 0) {
        donePlanes[i] = 1;
        Thetas[nhkls] = tht;
        PlaneNumbers[nhkls] = Rnr;
        RingRadsIdeal[nhkls] = rrdideal;
        ds[nhkls] = dsthis;
        nhkls++;
        break;
      }
    }
  }
  TthetaTol = Ttheta4mR((MaxRingRad + Width), Lsd) -
              Ttheta4mR((MaxRingRad - Width), Lsd);
  double IdealTthetas[n_hkls], TthetaMins[n_hkls], TthetaMaxs[n_hkls];
  for (i = 0; i < n_hkls; i++) {
    IdealTthetas[i] = 2 * Thetas[i];
    TthetaMins[i] = IdealTthetas[i] - TthetaTol;
    TthetaMaxs[i] = IdealTthetas[i] + TthetaTol;
  }
  int counter = 0;
  double nFramesThis;
  int nSpotsEachRing[n_hkls];
  for (i = 0; i < n_hkls; i++)
    nSpotsEachRing[i] = 0;
  fp = fopen(FileName, "r");
  printf("Reading file: %s.\n", FileName);
  fgets(line, 5000, fp);
  while (fgets(line, 5000, fp) != NULL) {
    sscanf(line,
           "%lf %lf %lf %lf %lf %s %s %s %s %s %s %s %lf %lf %s %lf %s %s %s",
           &SpotsInfo[counter][0], &SpotsInfo[counter][6],
           &SpotsInfo[counter][1], &SpotsInfo[counter][2],
           &SpotsInfo[counter][3], dummy, dummy, dummy, dummy, dummy, dummy,
           dummy, &nFramesThis, &SpotsInfo[counter][4], dummy,
           &SpotsInfo[counter][5], dummy, dummy, dummy);
    for (i = 0; i < n_hkls; i++)
      if ((int)SpotsInfo[counter][4] == PlaneNumbers[i])
        nSpotsEachRing[i]++;
    if ((int)nFramesThis > maxNFrames)
      continue; // Overwrite the spot if nFrames is greater than maxNFrames
    counter++;
  }
  printf("Number spots per ring: ");
  for (i = 0; i < n_hkls; i++)
    printf("%d ", nSpotsEachRing[i]);
  printf("\n");
  int nIndices = counter;
  printf("Number of planes being considered: %d.\nNumber of spots: %d.\n",
         n_hkls, nIndices);
  for (i = 0; i < nIndices; i++) {
    // Omega correction
    SpotsInfo[i][1] = SpotsInfo[i][1] -
                      (t_gap / (t_gap + t_int)) * OmegaStep *
                          (1.0 - fabs(2 * SpotsInfo[i][3] - (double)NrPixels) /
                                     (double)NrPixels);
    if (SpotsInfo[i][1] < -180)
      SpotsInfo[i][1] += 360;
    if (SpotsInfo[i][1] > 180)
      SpotsInfo[i][1] -= 360;
  }
  // Sort spots per ring
  char fnidhsh[1024];
  FILE *idhsh;
  sprintf(fnidhsh, "%s/IDRings.csv", folder);
  idhsh = fopen(fnidhsh, "w");
  fprintf(idhsh, "RingNumber OriginalID NewID(RingsMerge)\n");
  FILE *idshashout;
  char fnidshash[1024];
  sprintf(fnidshash, "%s/IDsHash.csv", folder);
  idshashout = fopen(fnidshash, "w");
  int nSpotsThis, nctr = 0, colN, startrowN = 0;
  double **spotsall;
  spotsall = allocMatrix(nIndices, 7);
  for (i = 0; i < n_hkls; i++) {
    double **spotsTemp;
    nSpotsThis = nSpotsEachRing[i];
    spotsTemp = allocMatrix(nSpotsThis, 7);
    nctr = 0;
    for (j = 0; j < nIndices; j++) {
      if (SpotsInfo[j][4] == PlaneNumbers[i]) {
        for (colN = 0; colN < 7; colN++)
          spotsTemp[nctr][colN] = SpotsInfo[j][colN];
        nctr++;
      }
    }
    SortSpots(nSpotsThis, spotsTemp);
    for (j = 0; j < nSpotsThis; j++) {
      // printf("%d %d %d %d %d %d
      // %d\n",i,n_hkls,nIndices,nSpotsThis,j,startrowN,j+startrowN);
      spotsall[j + startrowN][0] = j + startrowN + 1;
      for (colN = 1; colN < 7; colN++) {
        spotsall[j + startrowN][colN] = spotsTemp[j][colN];
      }
      fprintf(idhsh, "%d %d %d\n", PlaneNumbers[i], (int)spotsTemp[j][0],
              (int)spotsall[j + startrowN][0]);
    }
    fprintf(idshashout, "%d %d %d %lf\n", PlaneNumbers[i], startrowN + 1,
            startrowN + nSpotsThis + 1, ds[i]);
    startrowN += nSpotsThis;
    FreeMemMatrix(spotsTemp, nSpotsThis);
  }
  fclose(idhsh);
  fclose(idshashout);
  for (i = 0; i < nIndices; i++)
    for (j = 0; j < 7; j++)
      SpotsInfo[i][j] = spotsall[i][j];
  FreeMemMatrix(spotsall, nIndices);
  double *Ys, *Zs, *IdealTtheta, omegaCorrTemp;
  Ys = malloc(nIndices * sizeof(*Ys));
  Zs = malloc(nIndices * sizeof(*Zs));
  IdealTtheta = malloc(nIndices * sizeof(*IdealTtheta));
  for (i = 0; i < nIndices; i++) {
    Ys[i] = SpotsInfo[i][2];
    Zs[i] = SpotsInfo[i][3];
    for (j = 0; j < n_hkls; j++) {
      if (PlaneNumbers[j] == (int)SpotsInfo[i][4]) {
        IdealTtheta[i] = IdealTthetas[j];
        break;
      }
    }
  }
  double ty, tz, LsdFit, ybcFit, zbcFit, MeanDiff;
  if (DoFit == 1) {
    printf("Fitting parameters.\n");
    FitTiltBCLsd(nIndices, Ys, Zs, IdealTtheta, Lsd, RhoD, ybc, zbc, tx, tyIn,
                 tzIn, &ty, &tz, &LsdFit, &ybcFit, &zbcFit, p0, p1, p2, p3,
                 &MeanDiff, tolTilts, tolLsd, tolBC, px);
    printf("Number of function calls: %d\n", NrCalls);
    printf("LsdFit:\t\t%0.12f\nYBCFit:\t\t%0.12f\nZBCFit:\t\t%0.12f\ntyFit:"
           "\t\t%0.12f\ntzFit:\t\t%0.12f\nMeanStrain:\t%0.12lf\n",
           LsdFit, ybcFit, zbcFit, ty, tz, MeanDiff);
  } else {
    printf("Fitting not used. Using intial values for final results.\n");
    LsdFit = Lsd;
    ty = tyIn;
    tz = tzIn;
    ybcFit = ybc;
    zbcFit = zbc;
  }
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  double *YCorrected, *ZCorrected;
  YCorrected = malloc(nIndices * sizeof(*YCorrected));
  ZCorrected = malloc(nIndices * sizeof(*ZCorrected));
  CorrectTiltSpatialDistortion(nIndices, RhoD, Ys, Zs, px, LsdFit, ybcFit,
                               zbcFit, tx, ty, tz, p0, p1, p2, p3, YCorrected,
                               ZCorrected);
  double *YCorrWedge, *ZCorrWedge, *OmegaCorrWedge, *EtaCorrWedge,
      *TthetaCorrWedge, YCorrWedgeT, ZCorrWedgeT, OmegaCorrWedgeT,
      EtaCorrWedgeT, TthetaCorrWedgeT;
  YCorrWedge = malloc(nIndices * sizeof(*YCorrWedge));
  ZCorrWedge = malloc(nIndices * sizeof(*ZCorrWedge));
  OmegaCorrWedge = malloc(nIndices * sizeof(*OmegaCorrWedge));
  EtaCorrWedge = malloc(nIndices * sizeof(*EtaCorrWedge));
  TthetaCorrWedge = malloc(nIndices * sizeof(*TthetaCorrWedge));
  for (i = 0; i < nIndices; i++) {
    CorrectWedge(YCorrected[i], ZCorrected[i], LsdFit, SpotsInfo[i][1],
                 Wavelength, wedge, &YCorrWedgeT, &ZCorrWedgeT,
                 &OmegaCorrWedgeT, &EtaCorrWedgeT, &TthetaCorrWedgeT);
    YCorrWedge[i] = YCorrWedgeT;
    ZCorrWedge[i] = ZCorrWedgeT;
    OmegaCorrWedge[i] = OmegaCorrWedgeT;
    //~ if (fabs(OmegaCorrWedgeT-SpotsInfo[i][1])>0.1) printf("%lf %lf %lf %lf
    //%lf
    //%lf\n",YCorrected[i],YCorrWedgeT,ZCorrected[i],ZCorrWedgeT,SpotsInfo[i][1],OmegaCorrWedgeT);
    EtaCorrWedge[i] = EtaCorrWedgeT;
    TthetaCorrWedge[i] = TthetaCorrWedgeT;
  }
  // Useful arrays till now:
  // SpotsInfo,YCorrected,ZCorrected,YCorrWedge,ZCorrWedge,OmegaCorrWedge,EtaCorrWedge
  int NumberSpotsToKeep = 0;
  int *goodRows;
  goodRows = calloc(nIndices, sizeof(*goodRows));
  int KeepSpot, nSpotIDsToIndex = 0, *SpotIDsToIndex;
  SpotIDsToIndex = malloc(nIndices * sizeof(*SpotIDsToIndex));
  int UniqueRingNumbers[200], nrUniqueRingNumbers = 0, RingNumberThis,
                              RingNumberPresent = 0, nRejects = 0;
  printf("Indexing filters: RingNr: %d MinOme: %lf MaxOme: %lf \n", RingToIndex,
         MinOmeSpotIDsToIndex, MaxOmeSpotIDsToIndex);
  for (i = 0; i < nIndices; i++) {
    if (((EtaCorrWedge[i] > (-180 + MinEta)) && (EtaCorrWedge[i] < -MinEta)) ||
        ((EtaCorrWedge[i] > MinEta) && (EtaCorrWedge[i] < (180 - MinEta)))) {
      KeepSpot = 0;
      for (j = 0; j < nOmeRanges; j++) {
        if ((OmegaCorrWedge[i] >= OmegaRanges[j][0]) &&
            (OmegaCorrWedge[i] <= OmegaRanges[j][1]) &&
            (YCorrWedge[i] > BoxSizes[j][0]) &&
            (YCorrWedge[i] < BoxSizes[j][1]) &&
            (ZCorrWedge[i] > BoxSizes[j][2]) &&
            (ZCorrWedge[i] < BoxSizes[j][3])) {
          KeepSpot = 1;
          break;
        }
      }
      if (KeepSpot == 1) {
        goodRows[i] = 1;
        NumberSpotsToKeep++;
        RingNumberThis = (int)(SpotsInfo[i][4]);
        RingNumberPresent = 0;
        if (RingNumberThis == RingToIndex &&
            OmegaCorrWedge[i] >= MinOmeSpotIDsToIndex &&
            OmegaCorrWedge[i] <= MaxOmeSpotIDsToIndex) {
          SpotIDsToIndex[nSpotIDsToIndex] = SpotsInfo[i][0];
          nSpotIDsToIndex++;
        }
        for (j = 0; j < nrUniqueRingNumbers; j++) {
          if (RingNumberThis == UniqueRingNumbers[j]) {
            RingNumberPresent = 1;
            break;
          }
        }
        if (RingNumberPresent == 0) {
          UniqueRingNumbers[nrUniqueRingNumbers] = RingNumberThis;
          nrUniqueRingNumbers++;
        }
      } else {
        nRejects++;
      }
    } else {
      nRejects++;
    }
  }
  printf("nRejects: %d, nIndices: %d, Spots to keep: %d, SpotIDsToIndex: %d\n",
         nRejects, nIndices, NumberSpotsToKeep, nSpotIDsToIndex);
  FILE *IndexAll, *IndexAllNoHeader, *ExtraInfo, *IDs, *PF;
  char fnIndexAll[2048], fnIndexAllNoHeader[2048], fnExtraInfo[2048],
      fnSpIds[1024], parfn[1024];
  sprintf(parfn, "%s/paramstest.txt", folder);
  sprintf(fnIndexAll, "%s/InputAll.csv", folder);
  sprintf(outfolder, "%s/Output", folder);
  sprintf(resultfolder, "%s/Results", folder);
  int e = CheckDirectoryCreation(outfolder);
  if (e == 0) {
    return 1;
  }
  e = CheckDirectoryCreation(resultfolder);
  if (e == 0) {
    return 1;
  }

  sprintf(fnIndexAllNoHeader, "%s/InputAllNoHeader.csv", folder);
  sprintf(fnExtraInfo, "%s/InputAllExtraInfoFittingAll.csv", folder);
  sprintf(fnSpIds, "%s/%s", folder, idfn);
  IDs = fopen(fnSpIds, "w");
  for (i = 0; i < nSpotIDsToIndex; i++) {
    fprintf(IDs, "%d\n", SpotIDsToIndex[i]);
  }
  fclose(IDs);
  IndexAll = fopen(fnIndexAll, "w");
  IndexAllNoHeader = fopen(fnIndexAllNoHeader, "w");
  ExtraInfo = fopen(fnExtraInfo, "w");
  fprintf(IndexAll,
          "%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta\n");
  fprintf(ExtraInfo, "%YLab ZLab Omega GrainRadius SpotID RingNumber Eta "
                     "Ttheta OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) "
                     "ZOrig(NoWedgeCorr) YOrig(DetCor) ZOrig(DetCor) "
                     "OmegaOrig(DetCor) IntegratedIntensity(count)\n");
  for (i = 0; i < nIndices; i++) {
    if (goodRows[i] == 1) {
      fprintf(IndexAll,
              "%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",
              YCorrWedge[i], ZCorrWedge[i], OmegaCorrWedge[i], SpotsInfo[i][5],
              SpotsInfo[i][0], SpotsInfo[i][4], EtaCorrWedge[i],
              TthetaCorrWedge[i]);
      fprintf(IndexAllNoHeader,
              "%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",
              YCorrWedge[i], ZCorrWedge[i], OmegaCorrWedge[i], SpotsInfo[i][5],
              SpotsInfo[i][0], SpotsInfo[i][4], EtaCorrWedge[i],
              TthetaCorrWedge[i]);
      fprintf(ExtraInfo,
              "%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f "
              "%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",
              YCorrWedge[i], ZCorrWedge[i], OmegaCorrWedge[i], SpotsInfo[i][5],
              SpotsInfo[i][0], SpotsInfo[i][4], EtaCorrWedge[i],
              TthetaCorrWedge[i], SpotsInfo[i][1], YCorrected[i], ZCorrected[i],
              SpotsInfo[i][2], SpotsInfo[i][3], SpotsInfo[i][1],
              SpotsInfo[i][6]);
    } else {
      fprintf(IndexAll,
              "0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000\n",
              SpotsInfo[i][0]);
      fprintf(IndexAllNoHeader,
              "0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000\n",
              SpotsInfo[i][0]);
      fprintf(ExtraInfo,
              "0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000 0.000 "
              "0.000 0.000 0.0000 0.000 0.000 0.000\n",
              SpotsInfo[i][0]);
    }
  }

  fclose(IndexAll);
  fclose(IndexAllNoHeader);
  fclose(ExtraInfo);
  PF = fopen(parfn, "w");
  //~ fprintf(PF,"LatticeConstant %f;\n",LatticeConstant[0]);
  fprintf(PF, "LatticeParameter %f %f %f %f %f %f;\n", LatticeConstant[0],
          LatticeConstant[1], LatticeConstant[2], LatticeConstant[3],
          LatticeConstant[4], LatticeConstant[5]);
  //~ fprintf(PF,"CellStruct %d;\n",2);
  fprintf(PF, "MaxRingRad %f;\n", MaxRingRad);
  fprintf(PF, "SpaceGroup %d;\n", SGnum);
  fprintf(PF, "Wavelength %f;\n", Wavelength);
  fprintf(PF, "Distance %f;\n", LsdFit);
  fprintf(PF, "Rsample %f;\n", Rsample);
  fprintf(PF, "Hbeam %f;\n", Hbeam);
  fprintf(PF, "px %f;\n", px);
  fprintf(PF, "BeamSize %f;\n", BeamSize);
  fprintf(PF, "StepsizePos %f;\n", StepSizePos);
  fprintf(PF, "StepsizeOrient %f;\n", StepSizeOrient);
  fprintf(PF, "MarginRadius %f;\n", MarginRadius);
  fprintf(PF, "OmeBinSize %f;\n", OmeBinSize);
  fprintf(PF, "EtaBinSize %f;\n", EtaBinSize);
  fprintf(PF, "ExcludePoleAngle %f;\n", MinEta);
  for (i = 0; i < nrUniqueRingNumbers; i++) {
    fprintf(PF, "RingNumbers %d;\n", UniqueRingNumbers[i]);
  }
  for (i = 0; i < nrUniqueRingNumbers; i++) {
    fprintf(PF, "RingRadii %f;\n", RingRadsIdeal[i]);
  }
  fprintf(PF, "UseFriedelPairs %d;\n", UseFriedelPairs);
  fprintf(PF, "Wedge %f;\n", wedge);
  for (i = 0; i < nOmeRanges; i++) {
    fprintf(PF, "OmegaRange %f %f;\n", OmegaRanges[i][0], OmegaRanges[i][1]);
  }
  for (i = 0; i < nOmeRanges; i++) {
    fprintf(PF, "BoxSize %f %f %f %f;\n", BoxSizes[i][0], BoxSizes[i][1],
            BoxSizes[i][2], BoxSizes[i][3]);
  }
  fprintf(PF, "MarginEta %f;\n", MarginEta);
  fprintf(PF, "MarginOme %f;\n", MarginOme);
  fprintf(PF, "MargABC %f;\n", MargABC);
  fprintf(PF, "MargABG %f;\n", MargABG);
  fprintf(PF, "MarginRadial %f;\n", MarginRadial);
  fprintf(PF, "MinMatchesToAcceptFrac %f;\n", MinMatchesToAcceptFrac);
  fprintf(PF, "SpotsFileName %s\n", spotsfn);
  fprintf(PF, "RefinementFileName %s\n", "InputAllExtraInfoFittingAll.csv");
  fprintf(PF, "OutputFolder %s\n", outfolder);
  fprintf(PF, "ResultFolder %s\n", resultfolder);
  fprintf(PF, "IDsFileName %s\n", idfn);
  fprintf(PF, "LsdFit %f\n", LsdFit);
  fprintf(PF, "YBCFit %f\n", ybcFit);
  fprintf(PF, "ZBCFit %f\n", zbcFit);
  fprintf(PF, "tyFit %f\n", ty);
  fprintf(PF, "tzFit %f\n", tz);
  fprintf(PF, "p0 %f\n", p0);
  fprintf(PF, "p1 %f\n", p1);
  fprintf(PF, "p2 %f\n", p2);
  fprintf(PF, "p3 %f\n", p3);
  fclose(PF);
  FreeMemMatrix(SpotsInfo, MaxNSpots);
  free(Ys);
  free(Zs);
  free(IdealTtheta);
  free(YCorrected);
  free(ZCorrected);
  free(YCorrWedge);
  free(ZCorrWedge);
  free(OmegaCorrWedge);
  free(EtaCorrWedge);
  free(TthetaCorrWedge);
  free(SpotIDsToIndex);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
