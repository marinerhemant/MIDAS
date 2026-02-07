//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitPosOrStrains.c
//
//
//  Created by Hemant Sharma on 2014/06/20.
//
//
//  Output Array Contents:
//  0: ID
//  y z ome g0 g1 g2
//  1-6  : Observed spots corrected for grain position etc
//  7-12 : Simulated spots according to deformed lattice
//  13-15: Observed spots not corrected for grain position y z ome
//  16-18: Ome y z corrected without wedge till tilts spacial distortion
//  19-21: IA, LenDiff, OmeDiff

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
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
#include <unistd.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MaxNSpots 6000000
#define MaxNSpotsBest 5000 // MUST BE EQUAL TO MaxNHKLS
#define MaxNHKLS 5000
#define EPS 1E-12
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define MAXNOMEGARANGES 2000

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAXNOMEGARANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAXNOMEGARANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots);

// For detector mapping!
extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;
extern double DetParams[4][10];

static int GlobalDebugFlag = 0;

int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;
double DetParams[4][10];

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

static inline void Convert9To3x3(double MatIn[9], double MatOut[3][3]) {
  int i, j, k = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      MatOut[i][j] = MatIn[k];
      k++;
    }
  }
}
static inline void Convert3x3To9(double MatIn[3][3], double MatOut[9]) {
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      MatOut[(i * 3) + j] = MatIn[i][j];
}
static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  double determinant;
  determinant = m[0][0] * ((m[1][1] * m[2][2]) - (m[2][1] * m[1][2])) -
                m[0][1] * (m[1][0] * m[2][2] - m[2][0] * m[1][2]) +
                m[0][2] * (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
  //~ printf("Determinant: %lf\n",determinant);

  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = rad2deg * psi;
  Euler[1] = rad2deg * phi;
  Euler[2] = rad2deg * theta;
}

static inline void Euler2OrientMat(double Euler[3], double m_out[3][3]) {
  double psi, phi, theta, cps, cph, cth, sps, sph, sth;
  psi = Euler[0];
  phi = Euler[1];
  theta = Euler[2];
  cps = cosd(psi);
  cph = cosd(phi);
  cth = cosd(theta);
  sps = sind(psi);
  sph = sind(phi);
  sth = sind(theta);
  m_out[0][0] = cth * cps - sth * cph * sps;
  m_out[0][1] = -cth * cph * sps - sth * cps;
  m_out[0][2] = sph * sps;
  m_out[1][0] = cth * sps + sth * cph * cps;
  m_out[1][1] = cth * cph * cps - sth * sps;
  m_out[1][2] = -sph * cps;
  m_out[2][0] = sth * sph;
  m_out[2][1] = cth * sph;
  m_out[2][2] = cph;
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void CorrectHKLsLatC(double LatC[6], double **hklsIn, int nhkls,
                                   double Lsd, double Wavelength,
                                   double **hkls) {
  double a = LatC[0], b = LatC[1], c = LatC[2], alpha = LatC[3], beta = LatC[4],
         gamma = LatC[5];
  int hklnr;
  double SinA = sind(alpha), SinB = sind(beta), SinG = sind(gamma),
         CosA = cosd(alpha), CosB = cosd(beta), CosG = cosd(gamma);
  double GammaPr = acosd((CosA * CosB - CosG) / (SinA * SinB)),
         BetaPr = acosd((CosG * CosA - CosB) / (SinG * SinA)),
         SinBetaPr = sind(BetaPr);
  double Vol = (a * (b * (c * (SinA * (SinBetaPr * (SinG)))))),
         APr = b * c * SinA / Vol, BPr = c * a * SinB / Vol,
         CPr = a * b * SinG / Vol;
  double B[3][3];
  B[0][0] = APr;
  B[0][1] = (BPr * cosd(GammaPr)), B[0][2] = (CPr * cosd(BetaPr)), B[1][0] = 0,
  B[1][1] = (BPr * sind(GammaPr)), B[1][2] = (-CPr * SinBetaPr * CosA),
  B[2][0] = 0, B[2][1] = 0, B[2][2] = (CPr * SinBetaPr * SinA);
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    double ginit[3];
    ginit[0] = hklsIn[hklnr][0];
    ginit[1] = hklsIn[hklnr][1];
    ginit[2] = hklsIn[hklnr][2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hkls[hklnr][0] = GCart[0];
    hkls[hklnr][1] = GCart[1];
    hkls[hklnr][2] = GCart[2];
    hkls[hklnr][3] = Ds;
    double Theta = (asind((Wavelength) / (2 * Ds)));
    hkls[hklnr][4] = Theta;
    double Rad = Lsd * (tand(2 * Theta));
    hkls[hklnr][5] = Rad;
    hkls[hklnr][6] = hklsIn[hklnr][6];
  }
}

static inline void DisplacementInTheSpot(double a, double b, double c,
                                         double xi, double yi, double zi,
                                         double omega, double wedge, double chi,
                                         double *Displ_y, double *Displ_z) {
  double sinOme = sind(omega), cosOme = cosd(omega), AcosOme = a * cosOme,
         BsinOme = b * sinOme;
  double XNoW = AcosOme - BsinOme, YNoW = (a * sinOme) + (b * cosOme), ZNoW = c;
  double WedgeRad = deg2rad * wedge, CosW = cos(WedgeRad), SinW = sin(WedgeRad),
         XW = XNoW * CosW - ZNoW * SinW, YW = YNoW;
  double ZW = (XNoW * SinW) + (ZNoW * CosW), ChiRad = deg2rad * chi,
         CosC = cos(ChiRad), SinC = sin(ChiRad), XC = XW;
  double YC = (CosC * YW) - (SinC * ZW), ZC = (SinC * YW) + (CosC * ZW);
  double IK[3], NormIK;
  IK[0] = xi - XC;
  IK[1] = yi - YC;
  IK[2] = zi - ZC;
  NormIK = sqrt((IK[0] * IK[0]) + (IK[1] * IK[1]) + (IK[2] * IK[2]));
  IK[0] = IK[0] / NormIK;
  IK[1] = IK[1] / NormIK;
  IK[2] = IK[2] / NormIK;
  *Displ_y = YC - ((XC * IK[1]) / (IK[0]));
  *Displ_z = ZC - ((XC * IK[2]) / (IK[0]));
}

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void CorrectForOme(double yc, double zc, double Lsd,
                                 double OmegaIni, double wl, double wedge,
                                 double *ysOut, double *zsOut,
                                 double *OmegaOut) {
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
    SinOmega1 = 0;
  else if (SinOmega1 > 1)
    SinOmega1 = 0;
  else if (SinOmega2 < -1)
    SinOmega2 = 0;
  else if (SinOmega2 > 1)
    SinOmega2 = 0;
  if (CosOmega1 < -1)
    CosOmega1 = 0;
  else if (CosOmega1 > 1)
    CosOmega1 = 0;
  else if (CosOmega2 < -1)
    CosOmega2 = 0;
  else if (CosOmega2 > 1)
    CosOmega2 = 0;
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
  double Omega;
  if (OmeDiff1 < OmeDiff2)
    Omega = Omega1;
  else
    Omega = Omega2;
  double SinOmega = sin(deg2rad * Omega);
  double CosOmega = cos(deg2rad * Omega);
  double Fact = (g1 * CosOmega) - (g2 * SinOmega);
  double Eta = CalcEtaAngle(k2, k3);
  double Sin_Eta = sin(deg2rad * Eta);
  double Cos_Eta = cos(deg2rad * Eta);
  *ysOut = -RingRadius * Sin_Eta;
  *zsOut = RingRadius * Cos_Eta;
  *OmegaOut = Omega;
}

static inline void SpotToGv(double xi, double yi, double zi, double Omega,
                            double theta, double *g1, double *g2, double *g3) {
  double CosOme = cosd(Omega), SinOme = sind(Omega), eta = CalcEtaAngle(yi, zi),
         TanEta = tand(-eta), SinTheta = sind(theta);
  double CosTheta = cosd(theta), CosW = 1, SinW = 0,
         k3 = SinTheta * (1 + xi) / ((yi * TanEta) + zi), k2 = TanEta * k3,
         k1 = -SinTheta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW), k3f = (k3 * CosW) - (k1 * SinW),
         k2f = k2;
  *g1 = (k1f * CosOme) + (k2f * SinOme);
  *g2 = (k2f * CosOme) - (k1f * SinOme);
  *g3 = k3f;
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

static inline void CorrectTiltSpatialDistortion(
    int nIndices, double MaxRad, double yDet, double zDet, double px,
    double Lsd, double ybc, double zbc, double tx, double ty, double tz,
    double p0, double p1, double p2, double *yt, double *zt) {
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
    Yc = -(yDet - ybc) * px;
    Zc = (zDet - zbc) * px;
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
    *yt = -Rcorr * sin(deg2rad * Eta);
    *zt = Rcorr * cos(deg2rad * Eta);
  }
}

static inline void
CalcAngleErrors(int nspots, int nhkls, int nOmegaRanges, double x[12],
                double **spotsYZO, double **hklsIn, double Lsd,
                double Wavelength, double OmegaRange[MAXNOMEGARANGES][2],
                double BoxSize[MAXNOMEGARANGES][4], double MinEta, double wedge,
                double chi, double **SpotsComp, double **SpList, double *Error,
                int *nSpotsComp, int notIniRun) {
  int i, j;
  //~ for (i=0;i<12;i++) printf("%lf ",x[i]); printf("\n");
  int nrMatchedIndexer = nspots;
  double **MatchDiff;
  MatchDiff = allocMatrix(nrMatchedIndexer, 3);
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[6 + i];
  double **hkls;
  hkls = allocMatrix(nhkls, 7);
  CorrectHKLsLatC(LatC, hklsIn, nhkls, Lsd, Wavelength, hkls);
  double OrientMatrix[3][3], EulerIn[3];
  EulerIn[0] = x[3];
  EulerIn[1] = x[4];
  EulerIn[2] = x[5];
  Euler2OrientMat(EulerIn, OrientMatrix);
  int nTspots, nrSp;
  double **TheorSpots;
  TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  // TheorSpots are calculated according to LsdMean in case of Hydra
  CalcDiffractionSpots(Lsd, MinEta, OmegaRange, nOmegaRanges, hkls, nhkls,
                       BoxSize, &nTspots, OrientMatrix, TheorSpots);
  if (notIniRun == 1)
    printf(
        "DEBUG OMP: CalcAngleErrors Entry. nspots=%d, nhkls=%d, nTspots=%d\n",
        nspots, nhkls, nTspots);
  double **SpotsYZOGCorr;
  SpotsYZOGCorr = allocMatrix(nrMatchedIndexer, 7);
  double DisplY, DisplZ, ys, zs, Omega, Radius, Theta, lenK, yt, zt;
  for (nrSp = 0; nrSp < nrMatchedIndexer; nrSp++) {
    DisplacementInTheSpot(x[0], x[1], x[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    SpotsYZOGCorr[nrSp][0] = ys;
    SpotsYZOGCorr[nrSp][1] = zs;
    SpotsYZOGCorr[nrSp][2] = Omega;
    lenK = sqrt((Lsd * Lsd) + (ys * ys) + (zs * zs));
    Radius = sqrt((ys * ys) + (zs * zs));
    Theta = 0.5 * atand(Radius / Lsd);
    double g1, g2, g3;
    SpotToGv(Lsd / lenK, ys / lenK, zs / lenK, Omega, Theta, &g1, &g2, &g3);
    SpotsYZOGCorr[nrSp][3] = g1;
    SpotsYZOGCorr[nrSp][4] = g2;
    SpotsYZOGCorr[nrSp][5] = g3;
    SpotsYZOGCorr[nrSp][6] = spotsYZO[nrSp][7];
    //~ printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf
    //%lf\n",ys,zs,Omega,spotsYZO[nrSp][0],spotsYZO[nrSp][1],spotsYZO[nrSp][2],spotsYZO[nrSp][3],spotsYZO[nrSp][4],spotsYZO[nrSp][5],spotsYZO[nrSp][6],spotsYZO[nrSp][7]);
  }
  double **TheorSpotsYZWE;
  TheorSpotsYZWE = allocMatrix(nTspots, 9);
  for (i = 0; i < nTspots; i++) {
    for (j = 0; j < 9; j++) {
      TheorSpotsYZWE[i][j] = TheorSpots[i][j];
    }
  }
  int sp, nTheorSpotsYZWER, nMatched = 0, RowBest = 0;
  double GObs[3], GTheors[3], NormGObs, NormGTheors, DotGs, **TheorSpotsYZWER,
      Numers, Denoms, *Angles, minAngle;
  double diffLenM, diffOmeM;
  TheorSpotsYZWER = allocMatrix(MaxNSpotsBest, 9);
  Angles = malloc(MaxNSpotsBest * sizeof(*Angles));
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    if (notIniRun == 1 && sp < 5) {
      printf(
          "DEBUG OMP: Inspecting Obs Spot %d. Ring=%f, Omega=%f, Y=%f, Z=%f\n",
          sp, SpotsYZOGCorr[sp][6], SpotsYZOGCorr[sp][2], SpotsYZOGCorr[sp][0],
          SpotsYZOGCorr[sp][1]);
    }
    nTheorSpotsYZWER = 0;
    GObs[0] = SpotsYZOGCorr[sp][3];
    GObs[1] = SpotsYZOGCorr[sp][4];
    GObs[2] = SpotsYZOGCorr[sp][5];
    NormGObs = CalcNorm3(GObs[0], GObs[1], GObs[2]);
    for (i = 0; i < nTspots; i++) {
      /*
      if (notIniRun == 1 && sp < 5 &&
          ((int)TheorSpotsYZWE[i][7] == (int)SpotsYZOGCorr[sp][6])) {
        printf("DEBUG OMP:   Ring Match with Theor Spot %d. TheorOmega=%f, "
               "Diff=%f. Accept if < 5.0\n",
               i, TheorSpotsYZWE[i][2],
               fabs(SpotsYZOGCorr[sp][2] - TheorSpotsYZWE[i][2]));
      }
      */
      if (((int)TheorSpotsYZWE[i][7] == (int)SpotsYZOGCorr[sp][6]) &&
          (fabs(SpotsYZOGCorr[sp][2] - TheorSpotsYZWE[i][2]) < 5.0)) {
        for (j = 0; j < 9; j++) {
          TheorSpotsYZWER[nTheorSpotsYZWER][j] = TheorSpotsYZWE[i][j];
        }
        GTheors[0] = TheorSpotsYZWE[i][3];
        GTheors[1] = TheorSpotsYZWE[i][4];
        GTheors[2] = TheorSpotsYZWE[i][5];
        DotGs = ((GTheors[0] * GObs[0]) + (GTheors[1] * GObs[1]) +
                 (GTheors[2] * GObs[2]));
        NormGTheors = CalcNorm3(GTheors[0], GTheors[1], GTheors[2]);
        Numers = DotGs;
        Denoms = NormGObs * NormGTheors;
        double ratio = Numers / Denoms;
        if (ratio > 1.0)
          ratio = 1.0;
        if (ratio < -1.0)
          ratio = -1.0;
        Angles[nTheorSpotsYZWER] = fabs(acosd(ratio));
        nTheorSpotsYZWER++;
      }
    }
    if (nTheorSpotsYZWER == 0) {
      continue;
    }
    minAngle = 1000000;
    for (i = 0; i < nTheorSpotsYZWER; i++) {
      if (Angles[i] < minAngle) {
        minAngle = Angles[i];
        RowBest = i;
      }
    }
    diffLenM = CalcNorm2((SpotsYZOGCorr[sp][0] - TheorSpotsYZWER[RowBest][0]),
                         (SpotsYZOGCorr[sp][1] - TheorSpotsYZWER[RowBest][1]));
    diffOmeM = fabs(SpotsYZOGCorr[sp][2] - TheorSpotsYZWER[RowBest][2]);
    //~ printf("%lf\n",minAngle);
    if (notIniRun == 1 && sp < 5 && nTheorSpotsYZWER > 0) {
      printf("DEBUG OMP: Spot %d best minAngle=%f (Limit 1.0)\n", sp, minAngle);
    }
    if (minAngle < 1) {
      MatchDiff[nMatched][0] = minAngle;
      MatchDiff[nMatched][1] = diffLenM;
      MatchDiff[nMatched][2] = diffOmeM;
      SpotsComp[nMatched][0] = spotsYZO[sp][3];
      for (i = 0; i < 6; i++) {
        SpotsComp[nMatched][i + 1] = SpotsYZOGCorr[sp][i];
        SpotsComp[nMatched][i + 7] = TheorSpotsYZWER[RowBest][i];
      }
      SpotsComp[nMatched][13] = spotsYZO[sp][0];
      SpotsComp[nMatched][14] = spotsYZO[sp][1];
      SpotsComp[nMatched][15] = spotsYZO[sp][2];
      SpotsComp[nMatched][16] = spotsYZO[sp][4];
      SpotsComp[nMatched][17] = spotsYZO[sp][5];
      SpotsComp[nMatched][18] = spotsYZO[sp][6];
      SpotsComp[nMatched][19] = minAngle;
      SpotsComp[nMatched][20] = diffLenM;
      SpotsComp[nMatched][21] = diffOmeM;
      for (i = 0; i < 8; i++) {
        SpList[nMatched][i] = spotsYZO[sp][i];
      }
      SpList[nMatched][8] = TheorSpotsYZWER[RowBest][8];
      nMatched++;
    }
  }
  *nSpotsComp = nMatched;
  Error[0] = 0;
  Error[1] = 0;
  Error[2] = 0;
  for (i = 0; i < nMatched; i++) {
    Error[0] += fabs(MatchDiff[i][1] / nMatched);
    Error[1] += fabs(MatchDiff[i][2] / nMatched);
    Error[2] += fabs(MatchDiff[i][0] / nMatched);
  }
  // if (notIniRun == 1) {
  if (notIniRun == 1)
    printf("DEBUG OMP: nMatched=%d, Errors=%f %f %f, Pos=%f %f %f, Orient=%f "
           "%f %f\n",
           nMatched, Error[0], Error[1], Error[2], x[0], x[1], x[2], x[3], x[4],
           x[5]);
  // }
  FreeMemMatrix(MatchDiff, nrMatchedIndexer);
  FreeMemMatrix(hkls, nhkls);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(SpotsYZOGCorr, nrMatchedIndexer);
  FreeMemMatrix(TheorSpotsYZWE, nTspots);
  FreeMemMatrix(TheorSpotsYZWER, MaxNSpotsBest);
  free(Angles);
}

static inline void ConcatPosEulLatc(double *Ini, double Pos0[3],
                                    double Euler0[3], double LatCin[6]) {
  int i;
  for (i = 0; i < 3; i++)
    Ini[i] = Pos0[i];
  for (i = 0; i < 3; i++)
    Ini[i + 3] = Euler0[i];
  for (i = 0; i < 6; i++)
    Ini[i + 6] = LatCin[i];
}

struct OptimizeScratch {
  double **hkls;           // [nhkls][7]
  double **TheorSpots;     // [MaxNSpotsBest][9]
  double **SpotsYZOGCorr;  // [nSpotsComp][3]
  double **TheorSpotsYZWE; // [MaxNSpotsBest][3]
  double **hklsIn2;        // [nhkls][7] - Added to hoist the hkls copy buffer
  double **spotsYZO;       // [nSpotsComp][9] - Added to hoist spots copy buffer
  double *Angles;          // [MaxNSpotsBest] - Added for FitErrorsOrientStrains
};

struct data_FitPosIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  struct OptimizeScratch *scratch;
};

struct data_FitOrientIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Pos[3];
  struct OptimizeScratch *scratch;
};

struct data_FitStrainIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Pos[3];
  double Orient[3];
  struct OptimizeScratch *scratch;
};

struct data_FitPos {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Orient[3];
  double Strains[6];
  struct OptimizeScratch *scratch;
  int nTspots;
};

static inline double
FitErrorsPosSec(double x[3], int nSpotsComp, double spotsYZOIn[nSpotsComp][9],
                int nTspots, double Lsd, double Wavelength, double wedge,
                double chi, struct OptimizeScratch *scratch,
                struct data_FitPos
                    *f_data // Pass f_data to access pre-computed TheorSpotsYZWE
) {
  int i, i1, j, k;
  double X = x[0];
  double Y = x[1];
  double Z = x[2];
  double diff, min_diff;

  double **spotsYZO = scratch->spotsYZO;
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 9; j++)
      spotsYZO[i][j] = spotsYZOIn[i][j];

  double DisplY, DisplZ, ys, zs, Omega;
  for (i = 0; i < nSpotsComp; i++) {
    DisplacementInTheSpot(X, Y, Z, Lsd, spotsYZO[i][5], spotsYZO[i][6],
                          spotsYZO[i][4], wedge, chi, &DisplY, &DisplZ);
    double yt = spotsYZO[i][5] - DisplY;
    double zt = spotsYZO[i][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[i][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    scratch->SpotsYZOGCorr[i][0] = ys;
    scratch->SpotsYZOGCorr[i][1] = zs;
    scratch->SpotsYZOGCorr[i][2] = spotsYZO[i][8];
  }

  // Use pre-computed TheorSpotsYZWE from scratch
  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;

  double PosObs[2], PosTheor[2], Spnr;
  double Error = 0;
  for (i = 0; i < nSpotsComp; i++) {
    PosObs[0] = SpotsYZOGCorr[i][0];
    PosObs[1] = SpotsYZOGCorr[i][1];
    Spnr = SpotsYZOGCorr[i][2];
    for (k = 0; k < nTspots; k++) {
      if ((int)TheorSpotsYZWE[k][2] == (int)Spnr) {
        PosTheor[0] = TheorSpotsYZWE[k][0];
        PosTheor[1] = TheorSpotsYZWE[k][1];
        Error +=
            CalcNorm2((PosObs[0] - PosTheor[0]), (PosObs[1] - PosTheor[1]));
        break;
      }
    }
  }
  return Error;
}

static inline double
FitErrorsPosT(double x[12], int nSpotsComp, double spotsYZOIn[nSpotsComp][9],
              int nhkls, double hklsIn[nhkls][7], double Lsd, double Wavelength,
              int nOmeRanges, double OmegaRanges[nOmeRanges][2],
              double BoxSizes[nOmeRanges][4], double MinEta, double wedge,
              double chi, struct OptimizeScratch *scratch) {
  int i, j;
  int nrMatchedIndexer = nSpotsComp;
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[6 + i];
  double **hklsIn2 = scratch->hklsIn2;
  // hklsIn2 = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hklsIn[i][j];
  double **spotsYZO = scratch->spotsYZO;
  // spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = spotsYZOIn[i][j];
    }
  }
  double **hkls = scratch->hkls;
  // hkls = allocMatrix(nhkls, 7);
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hkls);
  double OrientMatrix[3][3], EulerIn[3];
  EulerIn[0] = x[3];
  EulerIn[1] = x[4];
  EulerIn[2] = x[5];
  Euler2OrientMat(EulerIn, OrientMatrix);
  int nTspots, nrSp;
  double **TheorSpots = scratch->TheorSpots;
  // TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);
  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;
  // SpotsYZOGCorr = allocMatrix(nrMatchedIndexer, 3);
  double DisplY, DisplZ, ys, zs, Omega, Radius, Theta, lenK, yt, zt;
  for (nrSp = 0; nrSp < nrMatchedIndexer; nrSp++) {
    DisplacementInTheSpot(x[0], x[1], x[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    SpotsYZOGCorr[nrSp][0] = ys;
    SpotsYZOGCorr[nrSp][1] = zs;
    SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
  }
  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  // TheorSpotsYZWE = allocMatrix(nTspots, 3);
  for (i = 0; i < nTspots; i++) {
    TheorSpotsYZWE[i][0] = TheorSpots[i][0];
    TheorSpotsYZWE[i][1] = TheorSpots[i][1];
    TheorSpotsYZWE[i][2] = TheorSpots[i][8];
  }
  int sp;
  double PosObs[2], PosTheor[2], Spnr;
  double Error = 0;
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    PosObs[0] = SpotsYZOGCorr[sp][0];
    PosObs[1] = SpotsYZOGCorr[sp][1];
    Spnr = SpotsYZOGCorr[sp][2];
    for (i = 0; i < nTspots; i++) {
      if ((int)TheorSpotsYZWE[i][2] == (int)Spnr) {
        PosTheor[0] = TheorSpotsYZWE[i][0];
        PosTheor[1] = TheorSpotsYZWE[i][1];
        Error +=
            CalcNorm2((PosObs[0] - PosTheor[0]), (PosObs[1] - PosTheor[1]));
        break;
      }
    }
  }
  /*
  FreeMemMatrix(hklsIn2, nhkls);
  FreeMemMatrix(spotsYZO, nSpotsComp);
  FreeMemMatrix(hkls, nhkls);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(SpotsYZOGCorr, nrMatchedIndexer);
  FreeMemMatrix(TheorSpotsYZWE, nTspots);
  */
  return Error;
}

static inline double FitErrorsOrientStrains(
    double x[9], int nSpotsComp, double spotsYZOIn[nSpotsComp][9], int nhkls,
    double hklsIn[nhkls][7], double Lsd, double Wavelength, int nOmeRanges,
    double OmegaRanges[nOmeRanges][2], double BoxSizes[nOmeRanges][4],
    double MinEta, double wedge, double chi, double Pos[3],
    struct OptimizeScratch *scratch) {
  int i, j;
  int nrMatchedIndexer = nSpotsComp;
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[3 + i];
  double **hklsIn2 = scratch->hklsIn2;
  // hklsIn2 = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hklsIn[i][j];
  double **hkls = scratch->hkls;
  // hkls = allocMatrix(nhkls, 7);
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hkls);
  double **spotsYZO = scratch->spotsYZO;
  // spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = spotsYZOIn[i][j];
    }
  }
  double OrientMatrix[3][3], EulerIn[3];
  EulerIn[0] = x[0];
  EulerIn[1] = x[1];
  EulerIn[2] = x[2];
  Euler2OrientMat(EulerIn, OrientMatrix);
  int nTspots, nrSp;
  double **TheorSpots = scratch->TheorSpots;
  // TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);
  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;
  // SpotsYZOGCorr = allocMatrix(nrMatchedIndexer, 4);
  double DisplY, DisplZ, ys, zs, Omega, Radius, Theta, lenK, yt, zt;
  for (nrSp = 0; nrSp < nrMatchedIndexer; nrSp++) {
    DisplacementInTheSpot(Pos[0], Pos[1], Pos[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    lenK = sqrt((Lsd * Lsd) + (ys * ys) + (zs * zs));
    Radius = sqrt((ys * ys) + (zs * zs));
    Theta = 0.5 * atand(Radius / Lsd);
    double g1, g2, g3;
    SpotToGv(Lsd / lenK, ys / lenK, zs / lenK, Omega, Theta, &g1, &g2, &g3);
    SpotsYZOGCorr[nrSp][0] = g1;
    SpotsYZOGCorr[nrSp][1] = g2;
    SpotsYZOGCorr[nrSp][2] = g3;
    SpotsYZOGCorr[nrSp][3] = spotsYZO[nrSp][8];
  }
  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  // TheorSpotsYZWE = allocMatrix(nTspots, 4);
  for (i = 0; i < nTspots; i++) {
    for (j = 0; j < 3; j++) {
      TheorSpotsYZWE[i][j] = TheorSpots[i][j + 3];
    }
    TheorSpotsYZWE[i][3] = TheorSpots[i][8];
  }
  int sp, nTheorSpotsYZWER, Spnr;
  double GObs[3], GTheors[3], NormGObs, NormGTheors, DotGs, Numers, Denoms;
  double *Angles = scratch->Angles;
  // Angles = malloc(20 * sizeof(*Angles));
  double minAngle, Error = 0;
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    nTheorSpotsYZWER = 0;
    GObs[0] = SpotsYZOGCorr[sp][0];
    GObs[1] = SpotsYZOGCorr[sp][1];
    GObs[2] = SpotsYZOGCorr[sp][2];
    NormGObs = CalcNorm3(GObs[0], GObs[1], GObs[2]);
    Spnr = SpotsYZOGCorr[sp][3];
    for (i = 0; i < nTspots; i++) {
      if ((int)TheorSpotsYZWE[i][3] == (int)Spnr) {
        GTheors[0] = TheorSpotsYZWE[i][0];
        GTheors[1] = TheorSpotsYZWE[i][1];
        GTheors[2] = TheorSpotsYZWE[i][2];
        DotGs = ((GTheors[0] * GObs[0]) + (GTheors[1] * GObs[1]) +
                 (GTheors[2] * GObs[2]));
        NormGTheors = CalcNorm3(GTheors[0], GTheors[1], GTheors[2]);
        Numers = DotGs;
        Denoms = NormGObs * NormGTheors;
        double ratio = Numers / Denoms;
        if (ratio > 1.0)
          ratio = 1.0;
        if (ratio < -1.0)
          ratio = -1.0;
        Angles[nTheorSpotsYZWER] = fabs(acosd(ratio));
        nTheorSpotsYZWER++;
      }
    }
    if (nTheorSpotsYZWER == 0)
      continue;
    minAngle = 1000000;
    for (i = 0; i < nTheorSpotsYZWER; i++) {
      if (Angles[i] < minAngle) {
        minAngle = Angles[i];
      }
    }
    if (minAngle > 4)
      continue;
    Error += minAngle;
  }
  /*
  FreeMemMatrix(hklsIn2, nhkls);
  FreeMemMatrix(hkls, nhkls);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(SpotsYZOGCorr, nrMatchedIndexer);
  FreeMemMatrix(TheorSpotsYZWE, nTspots);
  free(Angles);
  */
  return Error;
}

static inline double FitErrorsStrains(
    double x[6], int nSpotsComp, double spotsYZOIn[nSpotsComp][9], int nhkls,
    double hklsIn[nhkls][7], double Lsd, double Wavelength, int nOmeRanges,
    double OmegaRanges[nOmeRanges][2], double BoxSizes[nOmeRanges][4],
    double MinEta, double wedge, double chi, double Pos[3], double Orient[3],
    struct OptimizeScratch *scratch) {
  int i, j;
  int nrMatchedIndexer = nSpotsComp;
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[i];
  double **hklsIn2 = scratch->hklsIn2;
  // hklsIn2 = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hklsIn[i][j];
  double **spotsYZO = scratch->spotsYZO;
  // spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = spotsYZOIn[i][j];
    }
  }
  double **hkls = scratch->hkls;
  // hkls = allocMatrix(nhkls, 7);
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hkls);
  double OrientMatrix[3][3];
  Euler2OrientMat(Orient, OrientMatrix);
  int nTspots, nrSp;
  double **TheorSpots = scratch->TheorSpots;
  // TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);
  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;
  // SpotsYZOGCorr = allocMatrix(nrMatchedIndexer, 3);
  double DisplY, DisplZ, ys, zs, Omega, Radius, Theta, lenK, yt, zt;
  for (nrSp = 0; nrSp < nrMatchedIndexer; nrSp++) {
    DisplacementInTheSpot(Pos[0], Pos[1], Pos[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    SpotsYZOGCorr[nrSp][0] = ys;
    SpotsYZOGCorr[nrSp][1] = zs;
    SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
  }
  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  // TheorSpotsYZWE = allocMatrix(nTspots, 3);
  for (i = 0; i < nTspots; i++) {
    TheorSpotsYZWE[i][0] = TheorSpots[i][0];
    TheorSpotsYZWE[i][1] = TheorSpots[i][1];
    TheorSpotsYZWE[i][2] = TheorSpots[i][8];
  }
  int sp;
  double PosObs[2], PosTheor[2], Spnr;
  double Error = 0;
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    PosObs[0] = SpotsYZOGCorr[sp][0];
    PosObs[1] = SpotsYZOGCorr[sp][1];
    Spnr = SpotsYZOGCorr[sp][2];
    for (i = 0; i < nTspots; i++) {
      if ((int)TheorSpotsYZWE[i][2] == (int)Spnr) {
        PosTheor[0] = TheorSpotsYZWE[i][0];
        PosTheor[1] = TheorSpotsYZWE[i][1];
        Error +=
            CalcNorm2((PosObs[0] - PosTheor[0]), (PosObs[1] - PosTheor[1]));
        break;
      }
    }
  }
  /*
  FreeMemMatrix(hklsIn2, nhkls);
  FreeMemMatrix(spotsYZO, nSpotsComp);
  FreeMemMatrix(hkls, nhkls);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(SpotsYZOGCorr, nrMatchedIndexer);
  FreeMemMatrix(TheorSpotsYZWE, nTspots);
  */
  return Error;
}

static double problem_function_PosIni(unsigned n, const double *x, double *grad,
                                      void *f_data_trial) {
  int i, j;
  struct data_FitPosIni *f_data = (struct data_FitPosIni *)f_data_trial;
  int nSpotsComp = f_data->nSpotsComp;
  double spotsYZO[nSpotsComp][9];
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = f_data->spotsYZO[i][j];
    }
  }
  int nhkls = f_data->nhkls;
  double hkls[nhkls][7];
  for (i = 0; i < nhkls; i++) {
    for (j = 0; j < 7; j++) {
      hkls[i][j] = f_data->hkls[i][j];
    }
  }
  double Lsd = f_data->Lsd;
  double Wavelength = f_data->Wavelength;
  int nOmeRanges = f_data->nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      BoxSizes[i][j] = f_data->BoxSizes[i][j];
  }
  double MinEta = f_data->MinEta;
  double wedge = f_data->wedge;
  double chi = f_data->chi;
  double XIn[n];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsPosT(XIn, nSpotsComp, spotsYZO, nhkls, hkls, Lsd, Wavelength,
                       nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                       f_data->scratch);
}

static double problem_function_OrientIni(unsigned n, const double *x,
                                         double *grad, void *f_data_trial) {
  int i, j;
  struct data_FitOrientIni *f_data = (struct data_FitOrientIni *)f_data_trial;
  int nSpotsComp = f_data->nSpotsComp;
  double spotsYZO[nSpotsComp][9];
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = f_data->spotsYZO[i][j];
    }
  }
  int nhkls = f_data->nhkls;
  double hkls[nhkls][7];
  for (i = 0; i < nhkls; i++) {
    for (j = 0; j < 7; j++) {
      hkls[i][j] = f_data->hkls[i][j];
    }
  }
  double Lsd = f_data->Lsd;
  double Wavelength = f_data->Wavelength;
  int nOmeRanges = f_data->nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      BoxSizes[i][j] = f_data->BoxSizes[i][j];
  }
  double MinEta = f_data->MinEta;
  double wedge = f_data->wedge;
  double chi = f_data->chi;
  double Pos[3];
  for (i = 0; i < 3; i++)
    Pos[i] = f_data->Pos[i];
  double XIn[n];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsOrientStrains(XIn, nSpotsComp, spotsYZO, nhkls, hkls, Lsd,
                                Wavelength, nOmeRanges, OmegaRanges, BoxSizes,
                                MinEta, wedge, chi, Pos, f_data->scratch);
}

static double problem_function_StrainIni(unsigned n, const double *x,
                                         double *grad, void *f_data_trial) {
  int i, j;
  struct data_FitStrainIni *f_data = (struct data_FitStrainIni *)f_data_trial;
  int nSpotsComp = f_data->nSpotsComp;
  double spotsYZO[nSpotsComp][9];
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = f_data->spotsYZO[i][j];
    }
  }
  int nhkls = f_data->nhkls;
  double hkls[nhkls][7];
  for (i = 0; i < nhkls; i++) {
    for (j = 0; j < 7; j++) {
      hkls[i][j] = f_data->hkls[i][j];
    }
  }
  double Lsd = f_data->Lsd;
  double Wavelength = f_data->Wavelength;
  int nOmeRanges = f_data->nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      BoxSizes[i][j] = f_data->BoxSizes[i][j];
  }
  double MinEta = f_data->MinEta;
  double wedge = f_data->wedge;
  double chi = f_data->chi;
  double Pos[3];
  for (i = 0; i < 3; i++)
    Pos[i] = f_data->Pos[i];
  double Orient[3];
  for (i = 0; i < 3; i++)
    Orient[i] = f_data->Orient[i];
  double XIn[n];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsStrains(XIn, nSpotsComp, spotsYZO, nhkls, hkls, Lsd,
                          Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta,
                          wedge, chi, Pos, Orient, f_data->scratch);
}

static double problem_function_Pos(unsigned n, const double *x, double *grad,
                                   void *f_data_trial) {
  int i, j;
  struct data_FitPos *f_data = (struct data_FitPos *)f_data_trial;
  int nSpotsComp = f_data->nSpotsComp;
  double spotsYZO[nSpotsComp][9];
  for (i = 0; i < nSpotsComp; i++) {
    for (j = 0; j < 9; j++) {
      spotsYZO[i][j] = f_data->spotsYZO[i][j];
    }
  }
  int nhkls = f_data->nhkls;
  double hkls[nhkls][7];
  for (i = 0; i < nhkls; i++) {
    for (j = 0; j < 7; j++) {
      hkls[i][j] = f_data->hkls[i][j];
    }
  }
  double Lsd = f_data->Lsd;
  double Wavelength = f_data->Wavelength;
  int nOmeRanges = f_data->nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      BoxSizes[i][j] = f_data->BoxSizes[i][j];
  }
  double MinEta = f_data->MinEta;
  double wedge = f_data->wedge;
  double chi = f_data->chi;
  double XIn[n];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsPosSec(XIn, nSpotsComp, spotsYZO, f_data->nTspots, Lsd,
                         Wavelength, wedge, chi, f_data->scratch, f_data);
}

void FitPositionIni(double X0[12], int nSpotsComp, double **spotsYZO, int nhkls,
                    double **hkls, double Lsd, double Wavelength,
                    int nOmeRanges, double OmegaRanges[MAXNOMEGARANGES][2],
                    double BoxSizes[MAXNOMEGARANGES][4], double MinEta,
                    double wedge, double chi, double *XFit, double lb[12],
                    double ub[12]) {
  unsigned n = 12;
  double x[n], xl[n], xu[n];
  int i, j;
  struct data_FitPosIni f_data;
  f_data.nSpotsComp = nSpotsComp;
  f_data.spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 9; j++)
      f_data.spotsYZO[i][j] = spotsYZO[i][j];
  f_data.nhkls = nhkls;
  f_data.hkls = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++) {
      f_data.hkls[i][j] = hkls[i][j];
    }
  f_data.Lsd = Lsd;
  f_data.Wavelength = Wavelength;
  f_data.nOmeRanges = nOmeRanges;
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      f_data.BoxSizes[i][j] = BoxSizes[i][j];
  }
  f_data.MinEta = MinEta;
  f_data.wedge = wedge;
  f_data.chi = chi;
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 4);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 3);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 9);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));

  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }
  struct data_FitPosIni *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitPosIni *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_PosIni, trp);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_PosIni, trp);
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  if (GlobalDebugFlag)
    printf("DEBUG OMP FitPositionIni: %10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  free(f_data.scratch->Angles);
  free(f_data.scratch);
}

void FitOrientIni(double X0[9], int nSpotsComp, double **spotsYZO, int nhkls,
                  double **hkls, double Lsd, double Wavelength, int nOmeRanges,
                  double OmegaRanges[MAXNOMEGARANGES][2],
                  double BoxSizes[MAXNOMEGARANGES][4], double MinEta,
                  double wedge, double chi, double *XFit, double lb[9],
                  double ub[9], double Pos[3]) {
  unsigned n = 9;
  double x[n], xl[n], xu[n];
  int i, j;
  struct data_FitOrientIni f_data;
  f_data.nSpotsComp = nSpotsComp;
  f_data.spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 9; j++)
      f_data.spotsYZO[i][j] = spotsYZO[i][j];
  f_data.nhkls = nhkls;
  f_data.hkls = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      f_data.hkls[i][j] = hkls[i][j];
  f_data.Lsd = Lsd;
  f_data.Wavelength = Wavelength;
  f_data.nOmeRanges = nOmeRanges;
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      f_data.BoxSizes[i][j] = BoxSizes[i][j];
  }
  f_data.MinEta = MinEta;
  f_data.wedge = wedge;
  f_data.chi = chi;
  for (i = 0; i < 3; i++)
    f_data.Pos[i] = Pos[i];
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 4);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 4);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 9);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));

  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }
  struct data_FitOrientIni *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitOrientIni *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_OrientIni, trp);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_OrientIni, trp);
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  if (GlobalDebugFlag)
    printf("DEBUG OMP FitOrientIni: %10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  free(f_data.scratch->Angles);
  free(f_data.scratch);
}

void FitStrainIni(double X0[6], int nSpotsComp, double **spotsYZO, int nhkls,
                  double **hkls, double Lsd, double Wavelength, int nOmeRanges,
                  double OmegaRanges[MAXNOMEGARANGES][2],
                  double BoxSizes[MAXNOMEGARANGES][4], double MinEta,
                  double wedge, double chi, double *XFit, double lb[6],
                  double ub[6], double Pos[3], double Orient[3]) {
  unsigned n = 6;
  double x[n], xl[n], xu[n];
  int i, j;
  struct data_FitStrainIni f_data;
  f_data.nSpotsComp = nSpotsComp;
  f_data.spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 9; j++)
      f_data.spotsYZO[i][j] = spotsYZO[i][j];
  f_data.nhkls = nhkls;
  f_data.hkls = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      f_data.hkls[i][j] = hkls[i][j];
  f_data.Lsd = Lsd;
  f_data.Wavelength = Wavelength;
  f_data.nOmeRanges = nOmeRanges;
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      f_data.BoxSizes[i][j] = BoxSizes[i][j];
  }
  f_data.MinEta = MinEta;
  f_data.wedge = wedge;
  f_data.chi = chi;
  for (i = 0; i < 3; i++)
    f_data.Pos[i] = Pos[i];
  for (i = 0; i < 3; i++)
    f_data.Orient[i] = Orient[i];
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 4);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 3);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 9);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));

  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }
  struct data_FitStrainIni *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitStrainIni *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_StrainIni, trp);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_StrainIni, trp);
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  if (GlobalDebugFlag)
    printf("DEBUG OMP FitStrainIni: %10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  free(f_data.scratch->Angles);
  free(f_data.scratch);
}

void FitPosSec(double X0[3], int nSpotsComp, double **spotsYZO, int nhkls,
               double **hkls, double Lsd, double Wavelength, int nOmeRanges,
               double OmegaRanges[MAXNOMEGARANGES][2],
               double BoxSizes[MAXNOMEGARANGES][4], double MinEta, double wedge,
               double chi, double *XFit, double lb[3], double ub[3],
               double Orient[3], double Strains[6]) {
  unsigned n = 3;
  double x[n], xl[n], xu[n];
  int i, j;
  struct data_FitPos f_data;
  f_data.nSpotsComp = nSpotsComp;
  f_data.spotsYZO = allocMatrix(nSpotsComp, 9);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 9; j++)
      f_data.spotsYZO[i][j] = spotsYZO[i][j];
  f_data.nhkls = nhkls;
  f_data.hkls = allocMatrix(nhkls, 7);
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      f_data.hkls[i][j] = hkls[i][j];
  f_data.Lsd = Lsd;
  f_data.Wavelength = Wavelength;
  f_data.nOmeRanges = nOmeRanges;
  for (i = 0; i < nOmeRanges; i++) {
    for (j = 0; j < 2; j++)
      f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
    for (j = 0; j < 4; j++)
      f_data.BoxSizes[i][j] = BoxSizes[i][j];
  }
  f_data.MinEta = MinEta;
  f_data.wedge = wedge;
  f_data.chi = chi;
  for (i = 0; i < 3; i++)
    f_data.Orient[i] = Orient[i];
  for (i = 0; i < 6; i++)
    f_data.Strains[i] = Strains[i];
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 4);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 3);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 9);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));

  // Pre-calculate invariant data
  double **hklsIn2 = f_data.scratch->hklsIn2;
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hkls[i][j];
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = Strains[i];
  double **hklsScratch = f_data.scratch->hkls;
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hklsScratch);
  double OrientMatrix[3][3];
  Euler2OrientMat(Orient, OrientMatrix);
  int nTspots;
  double **TheorSpots = f_data.scratch->TheorSpots;
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hklsScratch, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);
  f_data.nTspots = nTspots;
  double **TheorSpotsYZWE = f_data.scratch->TheorSpotsYZWE;
  for (i = 0; i < nTspots; i++) {
    TheorSpotsYZWE[i][0] = TheorSpots[i][0];
    TheorSpotsYZWE[i][1] = TheorSpots[i][1];
    TheorSpotsYZWE[i][2] = TheorSpots[i][8];
  }

  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }
  struct data_FitPos *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitPos *)f_datat;
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_Pos, trp);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function_Pos, trp);
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  if (GlobalDebugFlag)
    printf("DEBUG OMP FitPosSec: %10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  free(f_data.scratch->Angles);
  free(f_data.scratch);
}

long long int ReadBigDet(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/BigDetectorMask.bin", cwd);
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  BigDetector = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(BigDetector == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  return (long long int)size;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "Supply a parameter file, blockNr, nBlocks, nSpotsToIndex, numProcs as "
        "arguments: ie %s param.txt nBlocks blockNr nSpotsToIndex numProcs\n\n",
        argv[0]);
    exit(EXIT_FAILURE);
  }
  double start_time = omp_get_wtime();
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000], outfolder[1000], spotsfilename[1000],
      inputfilename[1000];
  int LowNr;
  double Wavelength, Lsd;
  double LatCinT[6];
  double wedge, MinEta, OmegaRanges[MAXNOMEGARANGES][2],
      BoxSizes[MAXNOMEGARANGES][4], MaxRingRad;
  int RingNumbers[200], cs = 0, cs2 = 0, nOmeRanges = 0, nBoxSizes = 0,
                        CellStruct;
  double Rsample, Hbeam, RingRadii[200], MargABC = 0.3, MargABG = 0.3;
  char OutputFolder[1024], ResultFolder[1024];
  int DiscModel = 0, TopLayer = 0, TakeGrainMax = 0;
  int isGrainsInput = 0;
  char GrainsFileName[4096];
  int cntrdet = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCinT[0],
             &LatCinT[1], &LatCinT[2], &LatCinT[3], &LatCinT[4], &LatCinT[5]);
      continue;
    }
    str = "LatticeConstant ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCinT[0],
             &LatCinT[1], &LatCinT[2], &LatCinT[3], &LatCinT[4], &LatCinT[5]);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &pixelsize);
      continue;
    }
    str = "DetParams ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", dummy,
             &DetParams[cntrdet][0], &DetParams[cntrdet][1],
             &DetParams[cntrdet][2], &DetParams[cntrdet][3],
             &DetParams[cntrdet][4], &DetParams[cntrdet][5],
             &DetParams[cntrdet][6], &DetParams[cntrdet][7],
             &DetParams[cntrdet][8], &DetParams[cntrdet][9]);
      cntrdet++;
      continue;
    }
    str = "BigDetSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &BigDetSize);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "Distance ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "MaxRingRad ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }
    str = "ExcludePoleAngle ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MinEta);
      continue;
    }
    str = "MinEta ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MinEta);
      continue;
    }
    str = "TopLayer ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &TopLayer);
      continue;
    }
    str = "Hbeam ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Hbeam);
      continue;
    }
    str = "Rsample ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Rsample);
      continue;
    }
    str = "Wedge ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &wedge);
      continue;
    }
    str = "RingNumbers ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingNumbers[cs]);
      cs++;
      continue;
    }
    str = "RingRadii ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &RingRadii[cs2]);
      cs2++;
      continue;
    }
    str = "OmegaRange ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &OmegaRanges[nOmeRanges][0],
             &OmegaRanges[nOmeRanges][1]);
      nOmeRanges++;
      continue;
    }
    str = "BoxSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf", dummy, &BoxSizes[nBoxSizes][0],
             &BoxSizes[nBoxSizes][1], &BoxSizes[nBoxSizes][2],
             &BoxSizes[nBoxSizes][3]);
      nBoxSizes++;
      continue;
    }
    str = "OutputFolder ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, OutputFolder);
      continue;
    }
    str = "GrainsFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      isGrainsInput = 1;
      sscanf(aline, "%s %s", dummy, GrainsFileName);
      continue;
    }
    str = "ResultFolder ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, ResultFolder);
      continue;
    }
    str = "RefinementFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, inputfilename);
      continue;
    }
    str = "TakeGrainMax ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &TakeGrainMax);
      continue;
    }
    str = "MargABC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MargABC);
      continue;
    }
    str = "MargABG ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MargABG);
      continue;
    }
    str = "DebugMode ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &GlobalDebugFlag);
      continue;
    }
  }
  fclose(fileParam);
  double *AllSpots;
  int fd;
  struct stat s;
  int status;
  size_t size;
  size_t size2;
  char tmpstr[2048];
  sprintf(tmpstr, "%s", ResultFolder);
  char filename[2048], *cwd = dirname(tmpstr);
  sprintf(filename, "%s/ExtraInfo.bin", cwd);
  // char cmmd[4096];
  // sprintf(cmmd,"cp %s /dev/shm/",filename);
  // system(cmmd);
  // sprintf(filename,"/dev/shm/ExtraInfo.bin");
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  AllSpots = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(AllSpots == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  int nSpots = (int)size / (14 * sizeof(double));
  if (BigDetSize != 0) {
    long long int size2 = ReadBigDet(cwd);
    totNrPixelsBigDetector = BigDetSize;
    totNrPixelsBigDetector *= BigDetSize;
    totNrPixelsBigDetector /= 32;
    totNrPixelsBigDetector++;
  }
  int numProcs = atoi(argv[5]);
  double **hkls;
  hkls = allocMatrix(MaxNHKLS, 7);
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  if (hklf == NULL) {
    printf("Could not read the hkl file. Exiting.\n");
    return 1;
  }
  fgets(aline, 1000, hklf);
  int h, kt, l, Rnr, nhkls = 0;
  double ds, tht;
  int iter;
  double MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %lf %d %s %s %s %lf %s %s", &h, &kt, &l, &ds, &Rnr,
           dummy, dummy, dummy, &tht, dummy, dummy);
    if (tht > MaxTtheta / 2)
      break;
    for (iter = 0; iter < cs; iter++) {
      if (Rnr == RingNumbers[iter]) {
        hkls[nhkls][0] = h;
        hkls[nhkls][1] = kt;
        hkls[nhkls][2] = l;
        hkls[nhkls][3] = ds;
        hkls[nhkls][4] = tht;
        hkls[nhkls][5] = RingRadii[iter];
        hkls[nhkls][6] = RingNumbers[iter];
        nhkls++;
      }
    }
  }
  fclose(hklf);
  if (nOmeRanges != nBoxSizes) {
    printf("Number of omega ranges and number of box sizes don't match. "
           "Exiting!\n");
    return 1;
  }
  double MargOme = 0.01, MargPos = Rsample, MargPos2 = Rsample / 2,
         MargOme2 = 2, chi = 0;
  char *h1 = "SpotID,YObsCorrPos,ZObsCorrPos,OmegaObsCorrPos,G1Obs,G2Obs,G3Obs,"
             "YExp,ZExp,OmegaExp,G1Exp,G2Exp,G3Exp,";
  char *h2 = "YObsCorrWedge,ZObsCorrWedge,OmegaObsCorrWedge,OmegaObs,YObs,ZObs,"
             "InternalAngle,DiffLen,DiffOmega\n";
  char header[2048];
  sprintf(header, "%s%s", h1, h2);

  if (isGrainsInput == 0) {
    printf("Running full file!\n");
    //////////////////////////// OPENMP
    int *SptIDs;
    int nSptIDs;
    int nBlocks = atoi(argv[3]);
    int blockNr = atoi(argv[2]);
    int nSpotsToIndex = atoi(argv[4]);
    int startRowNr;
    int endRowNr;
    startRowNr = (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * blockNr;
    int tmp =
        (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * (blockNr + 1);
    endRowNr = tmp < (nSpotsToIndex - 1) ? tmp : (nSpotsToIndex - 1);
    nSptIDs = endRowNr - startRowNr + 1;
    SptIDs = malloc(nSptIDs * sizeof(*SptIDs));
    // Read spotIDs
    FILE *spotsFile = fopen("SpotsToIndex.csv", "r");
    int it;
    for (it = 0; it < startRowNr; it++) {
      fgets(aline, 1000, spotsFile);
    }
    for (it = 0; it < nSptIDs; it++) {
      fgets(aline, 1000, spotsFile);
      sscanf(aline, "%d", &SptIDs[it]);
    }
    fclose(spotsFile);
    int thisRowNr;
#pragma omp parallel for num_threads(numProcs) private(thisRowNr)              \
    schedule(dynamic)
    for (thisRowNr = 0; thisRowNr < nSptIDs; thisRowNr++) {
      int nrSpIds = 1;
      char OutFN[1024], OrigOutFN[1024];
      double OrientsOrig[nrSpIds][10], PositionsOrig[nrSpIds][4],
          ErrorsOrig[nrSpIds][4], OrientsFit[nrSpIds][10],
          PositionsFit[nrSpIds][4], StrainsFit[nrSpIds][7],
          ErrorsFin[nrSpIds][4];
      int i, j, k;
      int SpId = SptIDs[thisRowNr];
      double LatCin[6];
      char *SpFN = "SpotsToIndex.csv";
      FILE *SpFile = fopen(SpFN, "r");
      if (SpFile == NULL) {
        printf("Could not read the SpotsToIndex.csv file while trying %d. "
               "Exiting.\n",
               SpId);
        continue;
      }
      int rowNr = 0;
      int ThisID;
      int count = 0;
      char line[5024];
      while (fgets(line, 5000, SpFile) != NULL) {
        sscanf(line, "%d", &ThisID);
        if (ThisID == SpId) {
          rowNr = count;
          break;
        }
        count++;
      }
      fclose(SpFile);
      for (i = 0; i < 6; i++)
        LatCin[i] = LatCinT[i];

      int nSpID = 0;
      char InpFN[2048], InpFN2[2048];
      sprintf(InpFN, "%s/IndexBest.bin", OutputFolder);
      sprintf(InpFN2, "%s/IndexBestFull.bin", OutputFolder);
      int inpF = open(InpFN, O_RDONLY);
      if (inpF == -1) {
        printf("Nothing was found during indexing, nothing to do.\n");
        continue;
      }
      size_t offst1 = rowNr;
      offst1 *= 15 * sizeof(double);
      double *locArr;
      locArr = malloc(15 * sizeof(*locArr));
      int rcA = pread(inpF, locArr, 15 * sizeof(double), offst1);
      close(inpF);
      if (locArr[14] == 0) {
        printf("Good result not found. Skipping this rowNr: %d\n", rowNr);
        char KeyFN[1024];
        sprintf(KeyFN, "%s/Key.bin", ResultFolder);
        int SizeKeyFile = 2 * sizeof(int);
        size_t OffStKeyFile = SizeKeyFile;
        OffStKeyFile *= rowNr;
        int KeyInfo[2] = {0, 0};
#pragma omp critical
        {
          int resultKeyFN = open(KeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
          if (resultKeyFN <= 0) {
            printf("Could not open output file.\n");
          }
          int rc = pwrite(resultKeyFN, KeyInfo, SizeKeyFile, OffStKeyFile);
          if (rc < 0) {
            printf("Could not write to output file.\n");
          }
          close(resultKeyFN);
        }
        continue;
      }
      double Orient0[9], Pos0[3], IA0, Euler0[3], Orient0_3[3][3], NrExpected,
          NrObserved, meanRadius = 0, thisRadius, completeness,
                      MaxRadTot = -100;
      IA0 = locArr[0];
      for (i = 0; i < 9; i++)
        Orient0[i] = locArr[i + 1];
      for (i = 0; i < 3; i++)
        Pos0[i] = locArr[i + 10];
      NrExpected = locArr[13];
      NrObserved = locArr[14];
      free(locArr);
      completeness = NrObserved / NrExpected;
      char SpotsCompFN[2048];
      int nSpotsBest = (int)NrObserved, *spotIDS, nSpotsRad = 0;
      double *locArr2;
      spotIDS = malloc(nSpotsBest * sizeof(*spotIDS));
      locArr2 = malloc((int)NrObserved * 2 * sizeof(*locArr2));
      size_t offst2 = rowNr;
      offst2 *= MaxNHKLS;
      offst2 *= 2 * sizeof(double);
      int inpF2 = open(InpFN2, O_RDONLY);
      int rcB =
          pread(inpF2, locArr2, (int)NrObserved * 2 * sizeof(double), offst2);
      close(inpF2);
      for (i = 0; i < NrObserved; i++) {
        spotIDS[i] = (int)locArr2[i * 2 + 0];
        thisRadius = locArr2[i * 2 + 1];
        meanRadius += thisRadius;
        nSpotsRad++;
        if (TakeGrainMax == 1) {
          if (thisRadius > MaxRadTot) {
            MaxRadTot = thisRadius;
          }
        }
      }
      free(locArr2);
      meanRadius /= nSpotsRad;
      if (TakeGrainMax == 1) {
        meanRadius = MaxRadTot;
      }

      double a = LatCin[0], b = LatCin[1], c = LatCin[2], alph = LatCin[3],
             bet = LatCin[4], gamm = LatCin[5];
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          Orient0_3[i][j] = Orient0[i * 3 + j];
      OrientMat2Euler(Orient0_3, Euler0);
      Euler2OrientMat(Euler0, Orient0_3);
      Convert3x3To9(Orient0_3, Orient0);
      OrientMat2Euler(Orient0_3, Euler0);
      double **spotsYZO;
      spotsYZO = allocMatrix(nSpotsBest, 9);
      int nSpotsYZO = nSpotsBest;
      // Idea: spotID always starts from 1 and is increasing in number, so
      // spotIDS[i] should correspond to AllSpots[(spotIDS[i]-1)*14+...], this
      // should reduce execution time.
      size_t spotPosAllSpots;
      for (i = 0; i < nSpotsBest; i++) {
        spotPosAllSpots = (int)spotIDS[i] - 1;
        // check if spotPosAllSpots + 1 > nSpots, then print and error and
        // continue to next spot
        if (spotPosAllSpots >= nSpots || spotPosAllSpots < 0) {
          printf("\n\nData mismatch! Original: %d, Looked for "
                 "spotNr %d, nSpots %d, skipping this spot, GrID: %d.\n\n\n",
                 (int)spotIDS[i], (int)(spotPosAllSpots + 1), (int)nSpots,
                 (int)thisRowNr);
          fflush(stdout);
          continue;
        }
        if (spotPosAllSpots + 1 != (size_t)AllSpots[spotPosAllSpots * 14 + 4]) {
          printf("\n\nData mismatch! Original: %d, Looked for %d, found %d, "
                 "skipping this spot, GrID: %d.\n\n\n",
                 (int)spotIDS[i], (int)(spotPosAllSpots + 1),
                 (int)AllSpots[spotPosAllSpots * 14 + 4], (int)thisRowNr);
          fflush(stdout);
          continue;
        }
        spotsYZO[i][0] = AllSpots[spotPosAllSpots * 14 + 0];
        spotsYZO[i][1] = AllSpots[spotPosAllSpots * 14 + 1];
        spotsYZO[i][2] = AllSpots[spotPosAllSpots * 14 + 2];
        spotsYZO[i][3] = AllSpots[spotPosAllSpots * 14 + 4];
        spotsYZO[i][4] = AllSpots[spotPosAllSpots * 14 + 8];
        spotsYZO[i][5] = AllSpots[spotPosAllSpots * 14 + 9];
        spotsYZO[i][6] = AllSpots[spotPosAllSpots * 14 + 10];
        spotsYZO[i][7] = AllSpots[spotPosAllSpots * 14 + 5];
      }
      double *Ini;
      Ini = malloc(12 * sizeof(*Ini));
      double **SpotsComp, **Splist, *ErrorIni;
      SpotsComp = allocMatrix(MaxNSpotsBest, 22);
      Splist = allocMatrix(MaxNSpotsBest, 9);
      ErrorIni = malloc(3 * sizeof(*ErrorIni));
      int nSpotsComp;
      ConcatPosEulLatc(Ini, Pos0, Euler0, LatCin);
      ConcatPosEulLatc(Ini, Pos0, Euler0, LatCin);
      if (GlobalDebugFlag) {
        printf("DEBUG OMP: Calling initial CalcAngleErrors. nSpotsYZO=%d, "
               "nhkls=%d\n",
               nSpotsYZO, nhkls);
      }
      CalcAngleErrors(nSpotsYZO, nhkls, nOmeRanges, Ini, spotsYZO, hkls, Lsd,
                      Wavelength, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                      SpotsComp, Splist, ErrorIni, &nSpotsComp,
                      GlobalDebugFlag);
      double **spotsYZONew;
      spotsYZONew = allocMatrix(nSpotsComp, 9);
      for (i = 0; i < nSpotsComp; i++) {
        for (j = 0; j < 9; j++) {
          spotsYZONew[i][j] = Splist[i][j];
        }
      }
      OrientsOrig[nSpID][0] = (double)SpId;
      for (i = 0; i < 9; i++)
        OrientsOrig[nSpID][i + 1] = Orient0[i];
      PositionsOrig[nSpID][0] = (double)SpId;
      for (i = 0; i < 3; i++)
        PositionsOrig[nSpID][i + 1] = Pos0[i];
      ErrorsOrig[nSpID][0] = (double)SpId;
      for (i = 0; i < 3; i++)
        ErrorsOrig[nSpID][i + 1] = ErrorIni[i];
      double Inp[12];
      for (i = 0; i < 12; i++)
        Inp[i] = Ini[i];
      double X[3];
      for (i = 0; i < 3; i++)
        X[i] = Pos0[i];
      double X0[12];
      for (i = 0; i < 12; i++)
        X0[i] = Inp[i];
      double XLow[3], XHigh[3], EulerLow[3], EulerHigh[3];
      for (i = 0; i < 3; i++) {
        XLow[i] = X[i] - MargPos2;
        XHigh[i] = X[i] + MargPos2;
        EulerLow[i] = Euler0[i] - MargOme;
        EulerHigh[i] = Euler0[i] + MargOme;
      }
      if (XLow[0] < -Rsample)
        XLow[0] = -Rsample;
      if (XLow[1] < -Rsample)
        XLow[1] = -Rsample;
      if (XLow[2] < -Hbeam / 2)
        XLow[2] = -Hbeam / 2;
      if (XHigh[0] > Rsample)
        XHigh[0] = Rsample;
      if (XHigh[1] > Rsample)
        XHigh[1] = Rsample;
      if (XHigh[2] > Hbeam / 2)
        XHigh[2] = Hbeam / 2;
      double lb[12], ub[12];
      for (i = 0; i < 3; i++) {
        lb[i] = XLow[i];
        ub[i] = XHigh[i];
      }
      for (i = 3; i < 6; i++) {
        lb[i] = EulerLow[i - 3];
        ub[i] = EulerHigh[i - 3];
      }
      lb[6] = a * (1 - (MargABC / 100));
      lb[7] = b * (1 - (MargABC / 100));
      lb[8] = c * (1 - (MargABC / 100));
      lb[9] = alph * (1 - (MargABG / 100));
      lb[10] = bet * (1 - (MargABG / 100));
      lb[11] = gamm * (1 - (MargABG / 100));
      ub[6] = a * (1 + (MargABC / 100));
      ub[7] = b * (1 + (MargABC / 100));
      ub[8] = c * (1 + (MargABC / 100));
      ub[9] = alph * (1 + (MargABG / 100));
      ub[10] = bet * (1 + (MargABG / 100));
      ub[11] = gamm * (1 + (MargABG / 100));
      double *XFit;
      XFit = malloc(12 * sizeof(*XFit));
      double *ErrorInt1;
      ErrorInt1 = malloc(3 * sizeof(*ErrorInt1));
      FitPositionIni(X0, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                     nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                     XFit, lb, ub);
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, XFit, spotsYZONew, hkls,
                      Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta, wedge,
                      chi, SpotsComp, Splist, ErrorInt1, &nSpotsComp,
                      GlobalDebugFlag);
      for (i = 0; i < 3; i++)
        XFit[i + 3] = Euler0[i];
      for (i = 0; i < 6; i++)
        XFit[i + 6] = LatCin[i];
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, XFit, spotsYZONew, hkls,
                      Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta, wedge,
                      chi, SpotsComp, Splist, ErrorInt1, &nSpotsComp,
                      GlobalDebugFlag);
      for (i = 0; i < nSpotsComp; i++)
        for (j = 0; j < 9; j++)
          spotsYZONew[i][j] = Splist[i][j];
      double X0_2[9];
      X0_2[0] = Euler0[0];
      X0_2[1] = Euler0[1];
      X0_2[2] = Euler0[2];
      for (i = 0; i < 6; i++)
        X0_2[i + 3] = LatCin[i];
      double lb2[9], ub2[9];
      for (i = 0; i < 3; i++) {
        EulerLow[i] = Euler0[i] - MargOme2;
        EulerHigh[i] = Euler0[i] + MargOme2;
      }
      for (i = 0; i < 3; i++) {
        lb2[i] = EulerLow[i];
        ub2[i] = EulerHigh[i];
      }
      lb2[3] = a * (1 - (MargABC / 100));
      lb2[4] = b * (1 - (MargABC / 100));
      lb2[5] = c * (1 - (MargABC / 100));
      lb2[6] = alph * (1 - (MargABG / 100));
      lb2[7] = bet * (1 - (MargABG / 100));
      lb2[8] = gamm * (1 - (MargABG / 100));
      ub2[3] = a * (1 + (MargABC / 100));
      ub2[4] = b * (1 + (MargABC / 100));
      ub2[5] = c * (1 + (MargABC / 100));
      ub2[6] = alph * (1 + (MargABG / 100));
      ub2[7] = bet * (1 + (MargABG / 100));
      ub2[8] = gamm * (1 + (MargABG / 100));
      double *XFit2;
      XFit2 = malloc(9 * sizeof(*XFit2));
      double PosFitOrientIn[3];
      for (i = 0; i < 3; i++)
        PosFitOrientIn[i] = XFit[i];
      FitOrientIni(X0_2, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                   nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit2,
                   lb2, ub2, PosFitOrientIn);
      double UseXFit[12];
      for (i = 0; i < 3; i++)
        UseXFit[i] = XFit[i];
      for (i = 0; i < 3; i++)
        UseXFit[i + 3] = XFit2[i];
      for (i = 0; i < 6; i++)
        UseXFit[i + 6] = LatCin[i];
      double *ErrorInt2;
      ErrorInt2 = malloc(3 * sizeof(*ErrorInt2));
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFit, spotsYZONew, hkls,
                      Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta, wedge,
                      chi, SpotsComp, Splist, ErrorInt2, &nSpotsComp,
                      GlobalDebugFlag);
      for (i = 0; i < nSpotsComp; i++)
        for (j = 0; j < 9; j++)
          spotsYZONew[i][j] = Splist[i][j];
      double X0_3[6];
      for (i = 0; i < 6; i++)
        X0_3[i] = LatCin[i];
      double lb3[6], ub3[6];
      lb3[0] = a * (1 - (MargABC / 100));
      lb3[1] = b * (1 - (MargABC / 100));
      lb3[2] = c * (1 - (MargABC / 100));
      lb3[3] = alph * (1 - (MargABG / 100));
      lb3[4] = bet * (1 - (MargABG / 100));
      lb3[5] = gamm * (1 - (MargABG / 100));
      ub3[0] = a * (1 + (MargABC / 100));
      ub3[1] = b * (1 + (MargABC / 100));
      ub3[2] = c * (1 + (MargABC / 100));
      ub3[3] = alph * (1 + (MargABG / 100));
      ub3[4] = bet * (1 + (MargABG / 100));
      ub3[5] = gamm * (1 + (MargABG / 100));
      double OrientFitIn[3];
      for (i = 0; i < 3; i++)
        OrientFitIn[i] = XFit2[i];
      double *XFit3;
      XFit3 = malloc(6 * sizeof(*XFit3));
      FitStrainIni(X0_3, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                   nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit3,
                   lb3, ub3, PosFitOrientIn, OrientFitIn);
      double UseXFit2[12];
      for (i = 0; i < 3; i++)
        UseXFit2[i] = XFit[i];
      for (i = 0; i < 3; i++)
        UseXFit2[i + 3] = XFit2[i];
      for (i = 0; i < 6; i++)
        UseXFit2[i + 6] = XFit3[i];
      double *ErrorInt3;
      ErrorInt3 = malloc(3 * sizeof(*ErrorInt3));
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFit2, spotsYZONew,
                      hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta,
                      wedge, chi, SpotsComp, Splist, ErrorInt3, &nSpotsComp,
                      GlobalDebugFlag);
      for (i = 0; i < nSpotsComp; i++)
        for (j = 0; j < 9; j++)
          spotsYZONew[i][j] = Splist[i][j];
      double X0_4[3];
      for (i = 0; i < 3; i++)
        X0_4[i] = XFit[i];
      double XLow2[3], XHigh2[3];
      for (i = 0; i < 3; i++) {
        XLow2[i] = X0_4[i] - MargPos2;
        XHigh2[i] = X0_4[i] + MargPos2;
      }
      if (XLow2[0] < -Rsample)
        XLow2[0] = -Rsample;
      if (XLow2[1] < -Rsample)
        XLow2[1] = -Rsample;
      if (XLow2[2] < -Hbeam / 2)
        XLow2[2] = -Hbeam / 2;
      if (XHigh2[0] > Rsample)
        XHigh2[0] = Rsample;
      if (XHigh2[1] > Rsample)
        XHigh2[1] = Rsample;
      if (XHigh2[2] > Hbeam / 2)
        XHigh2[2] = Hbeam / 2;
      double lb4[3], ub4[3];
      for (i = 0; i < 3; i++) {
        lb4[i] = XLow2[i];
        ub4[i] = XHigh2[i];
      }
      double StrainsFitIn[6];
      for (i = 0; i < 6; i++)
        StrainsFitIn[i] = XFit3[i];
      double *XFit4;
      XFit4 = malloc(3 * sizeof(*XFit4));
      FitPosSec(X0_4, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit4,
                lb4, ub4, OrientFitIn, StrainsFitIn);
      double FinalResult[12];
      for (i = 0; i < 3; i++)
        FinalResult[i] = XFit4[i];
      for (i = 0; i < 3; i++)
        FinalResult[i + 3] = XFit2[i];
      for (i = 0; i < 6; i++)
        FinalResult[i + 6] = XFit3[i];
      double *ErrorFin;
      ErrorFin = malloc(3 * sizeof(*ErrorFin));
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, FinalResult, spotsYZONew,
                      hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta,
                      wedge, chi, SpotsComp, Splist, ErrorFin, &nSpotsComp,
                      GlobalDebugFlag);
      printf("SpotID %6d, %6d out of %6d, Errors: %7.2f %6.4f %6.4f, ", SpId,
             thisRowNr, nSptIDs, ErrorFin[0], ErrorFin[1], ErrorFin[2]);
      for (i = 0; i < nSpotsComp; i++)
        for (j = 0; j < 9; j++)
          spotsYZONew[i][j] = Splist[i][j];
      printf("Fitvals: Pos: %7.2f %7.2f %7.2f, Orient: %7.2f %7.2f %7.2f, "
             "LatC: %6.4f %6.4f %6.4f %7.3f %7.3f %7.3f\n",
             FinalResult[0], FinalResult[1], FinalResult[2], FinalResult[3],
             FinalResult[4], FinalResult[5], FinalResult[6], FinalResult[7],
             FinalResult[8], FinalResult[9], FinalResult[10], FinalResult[11]);
      double OF[3][3], OrientFit[9], EulerFit[3], PositionFit[3],
          LatticeParameterFit[6];
      for (i = 0; i < 3; i++)
        EulerFit[i] = FinalResult[i + 3];
      for (i = 0; i < 3; i++)
        PositionFit[i] = FinalResult[i];
      for (i = 0; i < 6; i++)
        LatticeParameterFit[i] = FinalResult[i + 6];
      Euler2OrientMat(EulerFit, OF);
      Convert3x3To9(OF, OrientFit);
      OrientsFit[nSpID][0] = SpId;
      PositionsFit[nSpID][0] = SpId;
      ErrorsFin[nSpID][0] = SpId;
      StrainsFit[nSpID][0] = SpId;
      for (i = 0; i < 9; i++) {
        OrientsFit[nSpID][i + 1] = OrientFit[i];
      }
      for (i = 0; i < 3; i++)
        PositionsFit[nSpID][i + 1] = PositionFit[i];
      for (i = 0; i < 6; i++)
        StrainsFit[nSpID][i + 1] = LatticeParameterFit[i];
      for (i = 0; i < 3; i++)
        ErrorsFin[nSpID][i + 1] = ErrorFin[i];

      // Start Writing: SpotsCompFN, OutFN, Key, ProcessGrainsFile
      char KeyFN[1024];
      sprintf(KeyFN, "%s/Key.bin", ResultFolder);
      int SizeKeyFile = 2 * sizeof(int);
      size_t OffStKeyFile = SizeKeyFile;
      OffStKeyFile *= rowNr;
      int KeyInfo[2] = {SpId, nSpotsComp};
      char ProcessGrainsFN[1024];
      sprintf(ProcessGrainsFN, "%s/ProcessKey.bin", ResultFolder);
      int SizeProcessFile = nSpotsComp * sizeof(int);
      size_t OffStProcessFile = MaxNHKLS;
      OffStProcessFile *= sizeof(int);
      OffStProcessFile *= rowNr;
      int ProcessInfo[nSpotsComp];
      for (i = 0; i < nSpotsComp; i++) {
        ProcessInfo[i] = SpotsComp[i][0];
      }
      sprintf(OutFN, "%s/OrientPosFit.bin", ResultFolder);
      int SizeOutFile = 27 * sizeof(double);
      size_t OffStSizeOutFile = SizeOutFile;
      OffStSizeOutFile *= rowNr;
      double OutMatr[27];
      for (i = 0; i < 10; i++) {
        OutMatr[i] = OrientsFit[nSpID][i];
      }
      for (i = 0; i < 4; i++) {
        OutMatr[i + 10] = PositionsFit[nSpID][i];
      }
      for (i = 0; i < 7; i++) {
        OutMatr[i + 14] = StrainsFit[nSpID][i];
      }
      for (i = 0; i < 4; i++) {
        OutMatr[i + 21] = ErrorsFin[nSpID][i];
      }
      OutMatr[25] = meanRadius;
      OutMatr[26] = completeness;
      sprintf(SpotsCompFN, "%s/FitBest.bin", OutputFolder);
      int SizeSpotsFile = 22 * sizeof(double) * nSpotsComp;
      size_t OffStSpotsFile = 22;
      OffStSpotsFile *= sizeof(double);
      OffStSpotsFile *= MaxNHKLS;
      OffStSpotsFile *= rowNr;
      double SpotsCompFNContents[nSpotsComp][22];
      for (i = 0; i < nSpotsComp; i++) {
        for (j = 0; j < 22; j++) {
          SpotsCompFNContents[i][j] = SpotsComp[i][j];
        }
      }
#pragma omp critical
      {
        int resultKeyFN = open(KeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultKeyFN <= 0) {
          printf("Could not open output file. %s\n", KeyFN);
        }
        int rcKey = pwrite(resultKeyFN, KeyInfo, SizeKeyFile, OffStKeyFile);
        if (rcKey < 0) {
          printf("Could not write to output file.\n");
          rcKey = close(resultKeyFN);
        }
        rcKey = close(resultKeyFN);
        int ProcessKeyFN =
            open(ProcessGrainsFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (ProcessKeyFN <= 0) {
          printf("Could not open output file. %s\n", ProcessGrainsFN);
        }
        int rcProcess = pwrite(ProcessKeyFN, ProcessInfo, SizeProcessFile,
                               OffStProcessFile);
        if (rcProcess < 0) {
          printf("Could not write to output file.\n");
        }
        rcProcess = close(ProcessKeyFN);
        int resultOutFN = open(OutFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultOutFN <= 0) {
          printf("Could not open output file. %s\n", OutFN);
        }
        int rcOut = pwrite(resultOutFN, OutMatr, SizeOutFile, OffStSizeOutFile);
        if (rcOut < 0) {
          printf("Could not write to output file.\n");
        }
        rcOut = close(resultOutFN);
        int resultSpotsCompFN =
            open(SpotsCompFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultSpotsCompFN <= 0) {
          printf("Could not open output file. %s\n", SpotsCompFN);
        }
        int rcSpots = pwrite(resultSpotsCompFN, SpotsCompFNContents,
                             SizeSpotsFile, OffStSpotsFile);
        if (rcSpots < 0) {
          printf("Could not write to output file.\n");
        }
        rcSpots = close(resultSpotsCompFN);
      }
      free(spotIDS);
      FreeMemMatrix(spotsYZO, nSpotsBest);
      free(Ini);
      FreeMemMatrix(SpotsComp, MaxNSpotsBest);
      FreeMemMatrix(Splist, MaxNSpotsBest);
      free(ErrorIni);
      FreeMemMatrix(spotsYZONew, nSpotsComp);
      free(XFit);
      free(ErrorInt1);
      free(XFit2);
      free(ErrorInt2);
      free(XFit3);
      free(ErrorInt3);
      free(XFit4);
      free(ErrorFin);
    }

    FreeMemMatrix(hkls, MaxNHKLS);
    free(SptIDs);
    double time = omp_get_wtime() - start_time;
    printf("Finished, time elapsed: %lf seconds.\n", time);
  } else {
    // We have the GrainsFile input, so just read the bin file, get the info and
    // fit strain only! Read SpotsToIndex.csv
    printf("We are tracking grains!\n");
    int *SptIDs;
    int nSptIDs = 0;
    FILE *spotsFile = fopen("SpotsToIndex.csv", "r");
    if (spotsFile == NULL) {
      printf("Could not open SpotsToIndex.csv. Exiting.\n");
      exit(EXIT_FAILURE);
    }
    // Count the number of lines in the file
    char line[1000];
    while (fgets(line, sizeof(line), spotsFile)) {
      nSptIDs++;
    }
    rewind(spotsFile);

    // Allocate memory for SptIDs and read all spot IDs
    SptIDs = malloc(nSptIDs * sizeof(*SptIDs));
    int it = 0;
    while (fgets(line, sizeof(line), spotsFile)) {
      sscanf(line, "%d", &SptIDs[it]);
      it++;
    }
    fclose(spotsFile);
// do openmp on each of these lines
#pragma omp parallel for num_threads(numProcs) private(it) schedule(dynamic)
    for (it = 0; it < nSptIDs; it++) {
      int i, j, k;
      int SpId = SptIDs[it];
      if (SpId == -1) {
        printf("Good result not found. Skipping this rowNr: %d\n", it);
        char KeyFN[1024];
        sprintf(KeyFN, "%s/Key.bin", ResultFolder);
        int SizeKeyFile = 2 * sizeof(int);
        size_t OffStKeyFile = SizeKeyFile;
        OffStKeyFile *= it;
        int KeyInfo[2] = {0, 0};
#pragma omp critical
        {
          int resultKeyFN = open(KeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
          if (resultKeyFN <= 0) {
            printf("Could not open output file.\n");
          }
          int rc = pwrite(resultKeyFN, KeyInfo, SizeKeyFile, OffStKeyFile);
          if (rc < 0) {
            printf("Could not write to output file.\n");
          }
          close(resultKeyFN);
        }
        continue;
      }
      char InpFN[2048], InpFN2[2048];
      sprintf(InpFN, "%s/IndexBest.bin", OutputFolder);
      sprintf(InpFN2, "%s/IndexBestFull.bin", OutputFolder);
      int inpF = open(InpFN, O_RDONLY);
      if (inpF == -1) {
        printf("Nothing was found during indexing, nothing to do.\n");
        continue;
      }
      size_t offst1 = it;
      offst1 *= 15 * sizeof(double);
      double *locArr;
      locArr = malloc(15 * sizeof(*locArr));
      int rcA = pread(inpF, locArr, 15 * sizeof(double), offst1);
      close(inpF);
      if (locArr[14] == 0) {
        printf("Good result not found. Skipping this rowNr: %d\n", it);
        char KeyFN[1024];
        sprintf(KeyFN, "%s/Key.bin", ResultFolder);
        int SizeKeyFile = 2 * sizeof(int);
        size_t OffStKeyFile = SizeKeyFile;
        OffStKeyFile *= it;
        int KeyInfo[2] = {0, 0};
#pragma omp critical
        {
          int resultKeyFN = open(KeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
          if (resultKeyFN <= 0) {
            printf("Could not open output file.\n");
          }
          int rc = pwrite(resultKeyFN, KeyInfo, SizeKeyFile, OffStKeyFile);
          if (rc < 0) {
            printf("Could not write to output file.\n");
          }
          close(resultKeyFN);
        }
        continue;
      }
      double Orient0[9], Pos0[3], IA0, Euler0[3], Orient0_3[3][3], NrExpected,
          NrObserved, completeness;
      IA0 = locArr[0];
      for (i = 0; i < 9; i++)
        Orient0[i] = locArr[i + 1];
      for (i = 0; i < 3; i++)
        Pos0[i] = locArr[i + 10];
      NrExpected = locArr[13];
      NrObserved = locArr[14];
      free(locArr);
      completeness = NrObserved / NrExpected;
      int nSpotsBest = (int)NrObserved, *spotIDS, nSpotsRad = 0;
      double *locArr2;
      spotIDS = malloc(nSpotsBest * sizeof(*spotIDS));
      locArr2 = malloc((int)NrObserved * 2 * sizeof(*locArr2));
      size_t offst2 = it;
      offst2 *= MaxNHKLS;
      offst2 *= 2 * sizeof(double);
      int inpF2 = open(InpFN2, O_RDONLY);
      int rcB =
          pread(inpF2, locArr2, (int)NrObserved * 2 * sizeof(double), offst2);
      close(inpF2);
      double thisRadius, meanRadius = 0, MaxRadTot = -100;
      for (i = 0; i < NrObserved; i++) {
        spotIDS[i] = (int)locArr2[i * 2 + 0];
        thisRadius = locArr2[i * 2 + 1];
        meanRadius += thisRadius;
        nSpotsRad++;
        if (TakeGrainMax == 1) {
          if (thisRadius > MaxRadTot) {
            MaxRadTot = thisRadius;
          }
        }
      }
      free(locArr2);
      meanRadius /= nSpotsRad;
      if (TakeGrainMax == 1) {
        meanRadius = MaxRadTot;
      }
      double LatCin[6];
      for (i = 0; i < 6; i++)
        LatCin[i] = LatCinT[i];
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          Orient0_3[i][j] = Orient0[i * 3 + j];
      OrientMat2Euler(Orient0_3, Euler0);
      Euler2OrientMat(Euler0, Orient0_3);
      Convert3x3To9(Orient0_3, Orient0);
      OrientMat2Euler(Orient0_3, Euler0);
      double **spotsYZO;
      spotsYZO = allocMatrix(nSpotsBest, 8);
      int nSpotsYZO = nSpotsBest;
      size_t spotPosAllSpots;
      for (i = 0; i < nSpotsBest; i++) {
        spotPosAllSpots = (int)spotIDS[i] - 1;
        if (spotPosAllSpots + 1 != (size_t)AllSpots[spotPosAllSpots * 14 + 4]) {
          printf("Data mismatch! Behavior undefined. Original: %d, Looked for "
                 "%zu, found %zu\n",
                 (int)spotIDS[i], spotPosAllSpots + 1,
                 (size_t)AllSpots[spotPosAllSpots * 14 + 4]);
        }
        spotsYZO[i][0] = AllSpots[spotPosAllSpots * 14 + 0];
        spotsYZO[i][1] = AllSpots[spotPosAllSpots * 14 + 1];
        spotsYZO[i][2] = AllSpots[spotPosAllSpots * 14 + 2];
        spotsYZO[i][3] = AllSpots[spotPosAllSpots * 14 + 4];
        spotsYZO[i][4] = AllSpots[spotPosAllSpots * 14 + 8];
        spotsYZO[i][5] = AllSpots[spotPosAllSpots * 14 + 9];
        spotsYZO[i][6] = AllSpots[spotPosAllSpots * 14 + 10];
        spotsYZO[i][7] = AllSpots[spotPosAllSpots * 14 + 5];
      }
      double *Ini;
      Ini = malloc(12 * sizeof(*Ini));
      double **SpotsComp, **Splist, *ErrorIni;
      SpotsComp = allocMatrix(MaxNSpotsBest, 22);
      Splist = allocMatrix(MaxNSpotsBest, 9);
      ErrorIni = malloc(3 * sizeof(*ErrorIni));
      int nSpotsComp;
      ConcatPosEulLatc(Ini, Pos0, Euler0, LatCin);
      CalcAngleErrors(nSpotsYZO, nhkls, nOmeRanges, Ini, spotsYZO, hkls, Lsd,
                      Wavelength, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                      SpotsComp, Splist, ErrorIni, &nSpotsComp, 0);
      double **spotsYZONew;
      spotsYZONew = allocMatrix(nSpotsComp, 9);
      for (i = 0; i < nSpotsComp; i++) {
        for (j = 0; j < 9; j++) {
          spotsYZONew[i][j] = Splist[i][j];
        }
      }
      double a = LatCin[0], b = LatCin[1], c = LatCin[2], alph = LatCin[3],
             bet = LatCin[4], gamm = LatCin[5];
      double X0_3[6];
      for (i = 0; i < 6; i++)
        X0_3[i] = LatCin[i];
      double lb3[6], ub3[6];
      lb3[0] = a * (1 - (MargABC / 100));
      lb3[1] = b * (1 - (MargABC / 100));
      lb3[2] = c * (1 - (MargABC / 100));
      lb3[3] = alph * (1 - (MargABG / 100));
      lb3[4] = bet * (1 - (MargABG / 100));
      lb3[5] = gamm * (1 - (MargABG / 100));
      ub3[0] = a * (1 + (MargABC / 100));
      ub3[1] = b * (1 + (MargABC / 100));
      ub3[2] = c * (1 + (MargABC / 100));
      ub3[3] = alph * (1 + (MargABG / 100));
      ub3[4] = bet * (1 + (MargABG / 100));
      ub3[5] = gamm * (1 + (MargABG / 100));
      double *XFit3;
      XFit3 = malloc(6 * sizeof(*XFit3));
      FitStrainIni(X0_3, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                   nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit3,
                   lb3, ub3, Pos0, Euler0);
      double *ErrorFin;
      ErrorFin = malloc(3 * sizeof(*ErrorFin));
      double UseXFit2[12];
      for (i = 0; i < 3; i++)
        UseXFit2[i] = Pos0[i];
      for (i = 0; i < 3; i++)
        UseXFit2[i + 3] = Euler0[i];
      for (i = 0; i < 6; i++)
        UseXFit2[i + 6] = XFit3[i];
      CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFit2, spotsYZONew,
                      hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta,
                      wedge, chi, SpotsComp, Splist, ErrorFin, &nSpotsComp, 1);
      for (i = 0; i < nSpotsComp; i++)
        for (j = 0; j < 9; j++)
          spotsYZONew[i][j] = Splist[i][j];
      double FinalResult[12];
      for (i = 0; i < 3; i++)
        FinalResult[i] = Pos0[i];
      for (i = 0; i < 3; i++)
        FinalResult[i + 3] = Euler0[i];
      for (i = 0; i < 6; i++)
        FinalResult[i + 6] = XFit3[i];
      printf("SpotID %6d, %6d out of %6d, Errors: %7.2f %6.4f %6.4f, ", SpId,
             it, nSptIDs, ErrorFin[0], ErrorFin[1], ErrorFin[2]);
      printf("Fitvals: Pos: %7.2f %7.2f %7.2f, Orient: %7.2f %7.2f %7.2f, "
             "LatC: %6.4f %6.4f %6.4f %7.3f %7.3f %7.3f\n",
             FinalResult[0], FinalResult[1], FinalResult[2], FinalResult[3],
             FinalResult[4], FinalResult[5], FinalResult[6], FinalResult[7],
             FinalResult[8], FinalResult[9], FinalResult[10], FinalResult[11]);
      double OF[3][3], OrientFit[9], EulerFit[3], PositionFit[3],
          LatticeParameterFit[6];
      for (i = 0; i < 3; i++)
        EulerFit[i] = FinalResult[i + 3];
      for (i = 0; i < 3; i++)
        PositionFit[i] = FinalResult[i];
      for (i = 0; i < 6; i++)
        LatticeParameterFit[i] = FinalResult[i + 6];
      Euler2OrientMat(EulerFit, OF);
      Convert3x3To9(OF, OrientFit);
      // Start Writing: SpotsCompFN, OutFN, Key, ProcessGrainsFile
      char KeyFN[1024];
      sprintf(KeyFN, "%s/Key.bin", ResultFolder);
      int SizeKeyFile = 2 * sizeof(int);
      size_t OffStKeyFile = SizeKeyFile;
      OffStKeyFile *= it;
      int KeyInfo[2] = {SpId, nSpotsComp};
      char ProcessGrainsFN[1024];
      sprintf(ProcessGrainsFN, "%s/ProcessKey.bin", ResultFolder);
      int SizeProcessFile = nSpotsComp * sizeof(int);
      size_t OffStProcessFile = MaxNHKLS;
      OffStProcessFile *= sizeof(int);
      OffStProcessFile *= it;
      int ProcessInfo[nSpotsComp];
      for (i = 0; i < nSpotsComp; i++) {
        ProcessInfo[i] = SpotsComp[i][0];
      }
      char OutFN[1024];
      sprintf(OutFN, "%s/OrientPosFit.bin", ResultFolder);
      int SizeOutFile = 27 * sizeof(double);
      size_t OffStSizeOutFile = SizeOutFile;
      OffStSizeOutFile *= it;
      double OutMatr[27];
      OutMatr[0] = SpId;
      OutMatr[10] = SpId;
      OutMatr[14] = SpId;
      OutMatr[21] = SpId;
      for (i = 0; i < 9; i++) {
        OutMatr[i + 1] = OrientFit[i];
      }
      for (i = 0; i < 3; i++) {
        OutMatr[i + 11] = PositionFit[i];
      }
      for (i = 0; i < 6; i++) {
        OutMatr[i + 15] = LatticeParameterFit[i];
      }
      for (i = 0; i < 3; i++) {
        OutMatr[i + 22] = ErrorFin[i];
      }
      OutMatr[25] = meanRadius;
      OutMatr[26] = completeness;
      char SpotsCompFN[2048];
      sprintf(SpotsCompFN, "%s/FitBest.bin", OutputFolder);
      int SizeSpotsFile = 22 * sizeof(double) * nSpotsComp;
      size_t OffStSpotsFile = 22;
      OffStSpotsFile *= sizeof(double);
      OffStSpotsFile *= MaxNHKLS;
      OffStSpotsFile *= it;
      double SpotsCompFNContents[nSpotsComp][22];
      for (i = 0; i < nSpotsComp; i++) {
        for (j = 0; j < 22; j++) {
          SpotsCompFNContents[i][j] = SpotsComp[i][j];
        }
      }
#pragma omp critical
      {
        int resultKeyFN = open(KeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultKeyFN <= 0) {
          printf("Could not open output file. %s\n", KeyFN);
        }
        int rcKey = pwrite(resultKeyFN, KeyInfo, SizeKeyFile, OffStKeyFile);
        if (rcKey < 0) {
          printf("Could not write to output file.\n");
          rcKey = close(resultKeyFN);
        }
        rcKey = close(resultKeyFN);
        int ProcessKeyFN =
            open(ProcessGrainsFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (ProcessKeyFN <= 0) {
          printf("Could not open output file. %s\n", ProcessGrainsFN);
        }
        int rcProcess = pwrite(ProcessKeyFN, ProcessInfo, SizeProcessFile,
                               OffStProcessFile);
        if (rcProcess < 0) {
          printf("Could not write to output file.\n");
        }
        rcProcess = close(ProcessKeyFN);
        int resultOutFN = open(OutFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultOutFN <= 0) {
          printf("Could not open output file. %s\n", OutFN);
        }
        int rcOut = pwrite(resultOutFN, OutMatr, SizeOutFile, OffStSizeOutFile);
        if (rcOut < 0) {
          printf("Could not write to output file.\n");
        }
        rcOut = close(resultOutFN);
        int resultSpotsCompFN =
            open(SpotsCompFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        if (resultSpotsCompFN <= 0) {
          printf("Could not open output file. %s\n", SpotsCompFN);
        }
        int rcSpots = pwrite(resultSpotsCompFN, SpotsCompFNContents,
                             SizeSpotsFile, OffStSpotsFile);
        if (rcSpots < 0) {
          printf("Could not write to output file.\n");
        }
        rcSpots = close(resultSpotsCompFN);
      }
      // Free any arrays that were malloc'ed
      free(Ini);
      FreeMemMatrix(spotsYZO, nSpotsBest);
      FreeMemMatrix(SpotsComp, MaxNSpotsBest);
      FreeMemMatrix(Splist, MaxNSpotsBest);
      free(ErrorIni);
      FreeMemMatrix(spotsYZONew, nSpotsComp);
      free(XFit3);
      free(spotIDS);
      free(ErrorFin);
    }
  }
  return 0;
}