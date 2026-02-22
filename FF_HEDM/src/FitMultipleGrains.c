//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
//
// FitGrain.c
//
//
// Created by Hemant sharma on 2016/10/24
// This code is to be used in cases where BC(y) has moved.
// Will NOT optimize: p0, p1, p2, RhoD, Lsd
// Will optimize: yBC,tx,ty,tz,zBC,wedge , x,y,z, orient, a,b,c,alpha,beta,gamma
// Things to read in: SpotMatrix.csv, Grains.csv, Params.txt
// Things to read from Params.txt: tx, ty, tz, Lsd, p0, p1, p2,
//				RhoD, BC, Wedge, NrPixels, px,
//				Wavelength, OmegaRange, BoxSize
//				MinEta, Hbeam, Rsample, RingNumbers (will
// provide cs), 				RingRadii,

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
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-12
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define MAX_LINE_LENGTH 4096
#define MaxNSpotsBest 1000

// For detector mapping!
extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;

int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;
#define MAX_N_OMEGA_RANGES 2000

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAX_N_OMEGA_RANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAX_N_OMEGA_RANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots);

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

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
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
    hkls[hklnr][6] = hklsIn[hklnr][3];
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

static inline double CalcAngleErrors(int nspots, int nhkls, int nOmegaRanges,
                                     double x[12], double **spotsYZO,
                                     double **hklsIn, double Lsd,
                                     double Wavelength,
                                     double OmegaRange[2000][2],
                                     double BoxSize[2000][4], double MinEta,
                                     double wedge, double chi, double *Error) {
  int i, j;
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
  CalcDiffractionSpots(Lsd, MinEta, OmegaRange, nOmegaRanges, hkls, nhkls,
                       BoxSize, &nTspots, OrientMatrix, TheorSpots);
  double **SpotsYZOGCorr;
  SpotsYZOGCorr = allocMatrix(nrMatchedIndexer, 7);
  double DisplY, DisplZ, ys, zs, Omega, Radius, Theta, lenK;
  for (nrSp = 0; nrSp < nrMatchedIndexer; nrSp++) {
    DisplacementInTheSpot(x[0], x[1], x[2], Lsd, spotsYZO[nrSp][2],
                          spotsYZO[nrSp][3], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    CorrectForOme(spotsYZO[nrSp][2] - DisplY, spotsYZO[nrSp][3] - DisplZ, Lsd,
                  spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs, &Omega);
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
    SpotsYZOGCorr[nrSp][6] = spotsYZO[nrSp][1];
  }
  double **TheorSpotsYZWE;
  TheorSpotsYZWE = allocMatrix(nTspots, 9);
  for (i = 0; i < nTspots; i++) {
    for (j = 0; j < 9; j++) {
      TheorSpotsYZWE[i][j] = TheorSpots[i][j];
    }
  }
  int sp, nTheorSpotsYZWER, nMatched = 0, RowBest = 0;
  double GObs[3], GTheors[3], NormGObs, NormGTheors, DotGs, Numers, Denoms,
      minAngle;
  double diffLenM, diffOmeM;
  // Angles = malloc(MaxNSpotsBest * sizeof(*Angles)); // Moved inside parallel
  // region
  nMatched = 0;

#pragma omp parallel private(sp, i, j, nTheorSpotsYZWER, GObs, GTheors,        \
                                 NormGObs, NormGTheors, DotGs, Numers, Denoms, \
                                 minAngle, RowBest, diffLenM, diffOmeM)
  {
    double *Angles_Th = malloc(MaxNSpotsBest * sizeof(double));
    double **TheorSpotsYZWER_Th = allocMatrix(MaxNSpotsBest, 9);

#pragma omp for
    for (sp = 0; sp < nrMatchedIndexer; sp++) {
      nTheorSpotsYZWER = 0;
      GObs[0] = SpotsYZOGCorr[sp][3];
      GObs[1] = SpotsYZOGCorr[sp][4];
      GObs[2] = SpotsYZOGCorr[sp][5];
      NormGObs = CalcNorm3(GObs[0], GObs[1], GObs[2]);
      for (i = 0; i < nTspots; i++) {
        if (((int)TheorSpotsYZWE[i][7] == (int)SpotsYZOGCorr[sp][6]) &&
            (fabs(SpotsYZOGCorr[sp][2] - TheorSpotsYZWE[i][2]) < 3.0)) {
          for (j = 0; j < 9; j++) {
            TheorSpotsYZWER_Th[nTheorSpotsYZWER][j] = TheorSpotsYZWE[i][j];
          }
          GTheors[0] = TheorSpotsYZWE[i][3];
          GTheors[1] = TheorSpotsYZWE[i][4];
          GTheors[2] = TheorSpotsYZWE[i][5];
          DotGs = ((GTheors[0] * GObs[0]) + (GTheors[1] * GObs[1]) +
                   (GTheors[2] * GObs[2]));
          NormGTheors = CalcNorm3(GTheors[0], GTheors[1], GTheors[2]);
          Numers = DotGs;
          Denoms = NormGObs * NormGTheors;
          Angles_Th[nTheorSpotsYZWER] = fabs(acosd(Numers / Denoms));
          nTheorSpotsYZWER++;
        }
      }
      if (nTheorSpotsYZWER == 0)
        continue;
      minAngle = 1000000;
      for (i = 0; i < nTheorSpotsYZWER; i++) {
        if (Angles_Th[i] < minAngle) {
          minAngle = Angles_Th[i];
          RowBest = i;
        }
      }
      diffLenM =
          CalcNorm2((SpotsYZOGCorr[sp][0] - TheorSpotsYZWER_Th[RowBest][0]),
                    (SpotsYZOGCorr[sp][1] - TheorSpotsYZWER_Th[RowBest][1]));
      diffOmeM = fabs(SpotsYZOGCorr[sp][2] - TheorSpotsYZWER_Th[RowBest][2]);
      if (minAngle < 1) {
        int idx;
#pragma omp atomic capture
        idx = nMatched++;

        if (idx < nrMatchedIndexer) {
          MatchDiff[idx][0] = minAngle;
          MatchDiff[idx][1] = diffLenM;
          MatchDiff[idx][2] = diffOmeM;
        }
      }
    }
    free(Angles_Th);
    FreeMemMatrix(TheorSpotsYZWER_Th, MaxNSpotsBest);
  }
  Error[0] = 0;
  Error[1] = 0;
  Error[2] = 0;
  for (i = 0; i < nMatched; i++) {
    Error[0] += fabs(MatchDiff[i][1] / nMatched); // Len
    Error[1] += fabs(MatchDiff[i][2] / nMatched); // Ome
    Error[2] += fabs(MatchDiff[i][0] / nMatched); // Angle
  }
  extern int nIter;
  FreeMemMatrix(MatchDiff, nrMatchedIndexer);
  FreeMemMatrix(hkls, nhkls);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(SpotsYZOGCorr, nrMatchedIndexer);
  FreeMemMatrix(TheorSpotsYZWE, nTspots);
  // FreeMemMatrix(TheorSpotsYZWER, MaxNSpotsBest); // Removed as it is now
  // thread-local free(Angles); // Removed as it is now thread-local
  return Error[0];
}

static inline void CorrectTiltSpatialDistortion(
    int nIndices, double MaxRad, double **SpotInfoAll, double px, double Lsd,
    double ybc, double zbc, double tx, double ty, double tz, double p0,
    double p1, double p2, double p3, double **SpotInfoCorr) {
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
    Yc = -(SpotInfoAll[i][2] - ybc) * px;
    Zc = (SpotInfoAll[i][3] - zbc) * px;
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
    SpotInfoCorr[i][0] = SpotInfoAll[i][0];
    SpotInfoCorr[i][1] = SpotInfoAll[i][1];
    SpotInfoCorr[i][4] = SpotInfoAll[i][4];
    SpotInfoCorr[i][2] = -Rcorr * sin(deg2rad * Eta);
    SpotInfoCorr[i][3] = Rcorr * cos(deg2rad * Eta);
  }
}

#include "MIDAS_Math.h"
#include "Panel.h"

struct GrainData {
  int NrPixels;
  int nOmeRanges;
  int nRings;
  int nSpots;
  int nhkls;
  double RhoD;
  double Lsd;
  double px;
  double Wavelength;
  double MinEta;
  double OmegaRanges[2000][2];
  double BoxSizes[2000][4];
  double **SpotInfoAll;
  double **hkls;
  double *Error;
  // Parameters specific to this grain
  double Ini[12];
  double OptP[10];
  double tol[22];
};

struct data {
  int nGrains;
  struct GrainData *grains;

  int nPanels;
  Panel *panels;
  int fixPanel; // Added fixPanel
};

// Helper to calculate errors for reporting
static void calc_grain_errors(int nGrains, int nPanels, Panel *panels,
                              struct GrainData *grains, const double *x,
                              double *errors) {
  // Unpack global
  int global_offset = nGrains * 12;
  double tx = x[global_offset + 0];
  double ty = x[global_offset + 1];
  double tz = x[global_offset + 2];
  double ybc = x[global_offset + 3];
  double zbc = x[global_offset + 4];
  double Wedge = x[global_offset + 5];
  double p0 = x[global_offset + 6];
  double p1 = x[global_offset + 7];
  double p2 = x[global_offset + 8];
  double p3 = x[global_offset + 9];

  for (int g = 0; g < nGrains; g++) {
    struct GrainData *g_data = &grains[g];
    int grain_offset = g * 12;
    double Inp[12];
    for (int i = 0; i < 12; i++)
      Inp[i] = x[grain_offset + i];

    double **SpotInfoCorr = allocMatrix(g_data->nSpots, 5);
    double **SpotInfoShifted = allocMatrix(g_data->nSpots, 5);

    for (int i = 0; i < g_data->nSpots; i++) {
      SpotInfoShifted[i][0] = g_data->SpotInfoAll[i][0];
      SpotInfoShifted[i][1] = g_data->SpotInfoAll[i][1];
      SpotInfoShifted[i][4] = g_data->SpotInfoAll[i][4];
      double y_raw = g_data->SpotInfoAll[i][2];
      double z_raw = g_data->SpotInfoAll[i][3];
      double dy = 0, dz = 0;
      if (nPanels > 0) {
        int pIdx = GetPanelIndex(y_raw, z_raw, nPanels, panels);
        if (pIdx >= 0) {
          dy = panels[pIdx].dY;
          dz = panels[pIdx].dZ;
        }
      }
      SpotInfoShifted[i][2] = y_raw + dy;
      SpotInfoShifted[i][3] = z_raw + dz;
    }

    CorrectTiltSpatialDistortion(g_data->nSpots, g_data->RhoD, SpotInfoShifted,
                                 g_data->px, g_data->Lsd, ybc, zbc, tx, ty, tz,
                                 p0, p1, p2, p3, SpotInfoCorr);
    FreeMemMatrix(SpotInfoShifted, g_data->nSpots);

    double gError[3];
    CalcAngleErrors(g_data->nSpots, g_data->nhkls, g_data->nOmeRanges, Inp,
                    SpotInfoCorr, g_data->hkls, g_data->Lsd, g_data->Wavelength,
                    g_data->OmegaRanges, g_data->BoxSizes, g_data->MinEta,
                    Wedge, 0.0, gError);

    FreeMemMatrix(SpotInfoCorr, g_data->nSpots);

    errors[g * 3 + 0] = gError[0]; // Len
    errors[g * 3 + 1] = gError[1]; // Ome
    errors[g * 3 + 2] = gError[2]; // Ang
  }
}

int nIter = 0;

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct data *f_data = (struct data *)f_data_trial;
  int nGrains = f_data->nGrains;
  int nPanels = f_data->nPanels;
  Panel *panels = f_data->panels;

  // Unpack Global Parameters
  // Order: [Grain1(12) ... GrainN(12)] [Global(10)] [Panels(2*NP)]
  int global_offset = nGrains * 12;
  double tx = x[global_offset + 0];
  double ty = x[global_offset + 1];
  double tz = x[global_offset + 2];
  double ybc = x[global_offset + 3];
  double zbc = x[global_offset + 4];
  double Wedge = x[global_offset + 5];
  double p0 = x[global_offset + 6];
  double p1 = x[global_offset + 7];
  double p2 = x[global_offset + 8];
  double p3 = x[global_offset + 9];

  // Panels are at the end of x
  if (nPanels > 1) {
    int p_idx = global_offset + 10;
    for (int i = 0; i < nPanels; i++) {
      if (i == f_data->fixPanel) {
        panels[i].dY = 0;
        panels[i].dZ = 0;
      } else {
        panels[i].dY = x[p_idx++];
        panels[i].dZ = x[p_idx++];
      }
    }
  }

  double totalError = 0;
  double totalLen = 0;
  double totalOme = 0;
  double totalAng = 0;

  // Per Grain Loop
  for (int g = 0; g < nGrains; g++) {
    struct GrainData *g_data = &f_data->grains[g];
    int grain_offset = g * 12;

    // Grain Parameters: Pos(3), Euler(3), Strain(6)
    double Inp[12];
    for (int i = 0; i < 12; i++)
      Inp[i] = x[grain_offset + i];

    // Alloc temporary structures
    double **SpotInfoCorr = allocMatrix(g_data->nSpots, 5);
    double **SpotInfoShifted = allocMatrix(g_data->nSpots, 5);

    for (int i = 0; i < g_data->nSpots; i++) {
      SpotInfoShifted[i][0] = g_data->SpotInfoAll[i][0];
      SpotInfoShifted[i][1] = g_data->SpotInfoAll[i][1];
      SpotInfoShifted[i][4] = g_data->SpotInfoAll[i][4];
      double y_raw = g_data->SpotInfoAll[i][2];
      double z_raw = g_data->SpotInfoAll[i][3];
      double dy = 0, dz = 0;
      if (nPanels > 0) {
        int pIdx = GetPanelIndex(y_raw, z_raw, nPanels, panels);
        if (pIdx >= 0) {
          dy = panels[pIdx].dY;
          dz = panels[pIdx].dZ;
        }
      }
      SpotInfoShifted[i][2] = y_raw + dy;
      SpotInfoShifted[i][3] = z_raw + dz;
    }

    CorrectTiltSpatialDistortion(g_data->nSpots, g_data->RhoD, SpotInfoShifted,
                                 g_data->px, g_data->Lsd, ybc, zbc, tx, ty, tz,
                                 p0, p1, p2, p3, SpotInfoCorr);
    FreeMemMatrix(SpotInfoShifted, g_data->nSpots);

    double gError[3];
    double err = CalcAngleErrors(
        g_data->nSpots, g_data->nhkls, g_data->nOmeRanges, Inp, SpotInfoCorr,
        g_data->hkls, g_data->Lsd, g_data->Wavelength, g_data->OmegaRanges,
        g_data->BoxSizes, g_data->MinEta, Wedge, 0.0, gError);

    FreeMemMatrix(SpotInfoCorr, g_data->nSpots);
    totalError += err;
    totalLen += gError[0];
    totalOme += gError[1];
    totalAng += gError[2];
  }

  int nIterLocal = nIter;
  if (nIterLocal % 500 == 0) {
    printf("Iter %d Total: L=%.5lf O=%.5lf A=%.5lf (PerG: L=%.5lf O=%.5lf "
           "A=%.5lf)\n",
           nIterLocal, totalLen, totalOme, totalAng, totalLen / nGrains,
           totalOme / nGrains, totalAng / nGrains);
    fflush(stdout);
  }
  nIter++;
  return totalError;
}

void FitMultipleGrains(struct GrainData *grains, int nGrains, double OptP[10],
                       double NonOptP[5], double tols[22], double *Out,
                       Panel *panels, int nPanels, double tolShifts,
                       int *spotsPerPanel, int fixPanel) {
  unsigned n = nGrains * 12 + 10; // Grains(12 each) + Global(10)
  if (nPanels > 1) {
    n += (nPanels - 1) * 2;
  }

  double *x = malloc(n * sizeof(double));
  double *xl = malloc(n * sizeof(double));
  double *xu = malloc(n * sizeof(double));

  struct data f_data;
  f_data.nGrains = nGrains;
  f_data.grains = grains;
  f_data.nPanels = nPanels;
  f_data.panels = panels;
  f_data.fixPanel = fixPanel;

  struct data *f_datat;
  f_datat = &f_data;
  void *trp = (struct data *)f_datat;

  // 1. Set Grain Parameters (0 to nGrains*12)
  for (int g = 0; g < nGrains; g++) {
    int offset = g * 12;
    double *xi = &x[offset];
    double *xli = &xl[offset];
    double *xui = &xu[offset];

    struct GrainData *gd = &grains[g];

    for (int i = 0; i < 12; i++)
      xi[i] = gd->Ini[i];

    // Tols: First 6 are Pos/Orient, Next 6 are Strain
    // In tols array: 0-5 (Pos/Orient), 6-11 (Strain), 12-21 (Global)

    for (int i = 0; i < 6; i++) { // Pos and Orient
      xli[i] = xi[i] - tols[i];
      xui[i] = xi[i] + tols[i];
    }
    for (int i = 6; i < 12; i++) {      // Strains
      xli[i] = xi[i] - (tols[i] / 100); // tols[6..11]
      xui[i] = xi[i] + (tols[i] / 100);
    }
  }

  // 2. Set Global Parameters
  int global_offset = nGrains * 12;
  for (int i = 0; i < 10; i++) {
    x[global_offset + i] = OptP[i];
    // Tols for global params are at indices 12-21 in 'tols' array
    xl[global_offset + i] = x[global_offset + i] - tols[12 + i];
    xu[global_offset + i] = x[global_offset + i] + tols[12 + i];
  }

  // 3. Set Panel Parameters
  if (nPanels > 1) {
    int p_idx = global_offset + 10; // Start of panel parameters
    for (int i = 0; i < nPanels; i++) {
      if (i == fixPanel) { // Skip the fixed panel
        continue;
      }

      x[p_idx] = panels[i].dY;
      xl[p_idx] = x[p_idx] - tolShifts;
      xu[p_idx] = x[p_idx] + tolShifts;
      if (spotsPerPanel != NULL && spotsPerPanel[i] < 1) {
        xl[p_idx] = x[p_idx]; // Lock to current value if no spots
        xu[p_idx] = x[p_idx];
      }
      p_idx++;

      x[p_idx] = panels[i].dZ;
      xl[p_idx] = x[p_idx] - tolShifts;
      xu[p_idx] = x[p_idx] + tolShifts;
      if (spotsPerPanel != NULL && spotsPerPanel[i] < 1) {
        xl[p_idx] = x[p_idx]; // Lock to current value if no spots
        xu[p_idx] = x[p_idx];
      }
      p_idx++;
    }
  }

  // Calculate Initial Errors
  double *IniErrs = malloc(nGrains * 3 * sizeof(double));
  calc_grain_errors(nGrains, nPanels, panels, grains, x,
                    IniErrs); // x has initial values here

  NLoptConfig config = {0};
  config.objective_function = problem_function;
  config.obj_data = trp;
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  // Note: nlopt_set_maxtime(opt, 30); is not in config, I will add it to
  // MIDAS_Math
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  double minf = config.min_function_val;

  // Calculate Final Errors
  double *FinErrs = malloc(nGrains * 3 * sizeof(double));
  calc_grain_errors(nGrains, nPanels, panels, grains, x, FinErrs);

  // Unpack Output
  // Global Params
  printf("\n========================================\n");
  printf("        Global Optimization Results       \n");
  printf("========================================\n");
  double *gOpt = &x[global_offset];
  printf("Lsd %.12f (Fixed)\n", NonOptP[1]);
  printf("BC %.12f %.12f\n", gOpt[3], gOpt[4]);
  printf("tx %.12f\n", gOpt[0]);
  printf("ty %.12f\n", gOpt[1]);
  printf("tz %.12f\n", gOpt[2]);
  printf("Wedge %.12f\n", gOpt[5]);
  printf("p0 %.12f\n", gOpt[6]);
  printf("p1 %.12f\n", gOpt[7]);
  printf("p2 %.12f\n", gOpt[8]);
  printf("p3 %.12f\n", gOpt[9]);

  printf("\n-------------------------------------------------------------------"
         "----------------------------------------------\n");
  printf(" Grain | Initial Error (L, O, A) | Final Error (L, O, "
         "A) | Position / Orientation / LatticeParameter\n");
  printf("---------------------------------------------------------------------"
         "--------------------------------------------\n");
  for (int g = 0; g < nGrains; g++) {
    double *xi = &x[g * 12];
    printf(" %5d | %.4f %.4f %.4f | %.4f %.4f %.4f | Pos: %.2f %.2f %.2f | "
           "Orientation: %.2f %.2f %.2f | LatticeParameter: %.5f %.5f %.5f "
           "%.5f %.5f %.5f\n",
           g, IniErrs[g * 3], IniErrs[g * 3 + 1], IniErrs[g * 3 + 2],
           FinErrs[g * 3], FinErrs[g * 3 + 1], FinErrs[g * 3 + 2], xi[0], xi[1],
           xi[2], xi[3], xi[4], xi[5], xi[6], xi[7], xi[8], xi[9], xi[10],
           xi[11]);
  }
  printf("---------------------------------------------------------------------"
         "--------------------------------------------\n");

  free(IniErrs);
  free(FinErrs);

  for (int i = 0; i < n; i++)
    Out[i] = x[i];

  if (nPanels > 1) {
    int p_idx = global_offset + 10;
    for (int i = 0; i < nPanels; i++) {
      if (i == fixPanel) {
        panels[i].dY = 0;
        panels[i].dZ = 0;
      } else {
        panels[i].dY = x[p_idx++];
        panels[i].dZ = x[p_idx++];
      }
    }
  }

  free(x);
  free(xl);
  free(xu);
}

// Sort helper
struct GrainInfo {
  int ID;
  double FitError;
};

int compareGrains(const void *a, const void *b) {
  struct GrainInfo *ga = (struct GrainInfo *)a;
  struct GrainInfo *gb = (struct GrainInfo *)b;
  if (ga->FitError < gb->FitError)
    return -1;
  if (ga->FitError > gb->FitError)
    return 1;
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 4 || argc > 5) {
    printf("Usage: FitMultipleGrains Folder Parameters.txt nGrains [nCPUs]\n");
    return 0;
  }
  int nGrainsReq = atoi(argv[3]);
  int nCPUs = 10;
  if (argc == 5) {
    nCPUs = atoi(argv[4]);
  }
  omp_set_num_threads(nCPUs);
  clock_t start, end;
  double diftotal;
  start = clock();
  char aline[MAX_LINE_LENGTH];
  int LowNr;
  int GrainID = atoi(argv[3]);
  FILE *fileParam;
  char *ParamFN;
  ParamFN = argv[2];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[MAX_LINE_LENGTH];
  double tx, ty, tz, Lsd, p0, p1, p2, p3, RhoD, yBC, zBC, wedge, px, a, b, c,
      alpha, beta, gamma, OmegaRanges[2000][2], BoxSizes[2000][4], MaxRingRad,
      MaxTtheta, Wavelength, MinEta, Hbeam, Rsample;
  int NrPixels, nOmeRanges = 0, nBoxSizes = 0, cs = 0, RingNumbers[200],
                cs2 = 0;
  double tolShifts = 1.0;
  double tolBC = 1.0, tolTilts = 1.0, tolP0 = 1E-3, tolP1 = 1E-3, tolP2 = 1E-3,
         tolP3 = 45.0, tolTiltX, tolTiltY, tolTiltZ;
  int FixPanelID = 0;

  // Panel parameters
  int nPanelsY = 0;
  int nPanelsZ = 0;
  int panelSizeY = 0;
  int panelSizeZ = 0;
  int nGapsY = 0;
  int nGapsZ = 0;
  int panelGapsY[10]; // Assuming max 10 gaps for now
  int panelGapsZ[10];
  char panelShiftsFile[MAX_LINE_LENGTH] = "";
  while (fgets(aline, MAX_LINE_LENGTH, fileParam) != NULL) {
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
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
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
    str = "p3 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p3);
      continue;
    }
    str = "RhoD ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &RhoD);
      continue;
    }
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &yBC, &zBC);
      continue;
    }
    str = "Wedge ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &wedge);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "ExcludePoleAngle ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MinEta);
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
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixels);
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
    str = "RingThresh ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingNumbers[cs]);
      cs++;
      continue;
    }
    str = "MaxRingRad ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }

    // Panel parameter parsing
    str = "NPanelsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nPanelsY);
      continue;
    }
    str = "NPanelsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nPanelsZ);
      continue;
    }
    str = "PanelSizeY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &panelSizeY);
      continue;
    }
    str = "PanelSizeZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &panelSizeZ);
      continue;
    }
    str = "PanelGapsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      char *ptr = aline + strlen(str); // Skip "PanelGapsY "
      int gapVal;
      int gapIdx = 0;
      while (sscanf(ptr, "%d", &gapVal) == 1) {
        panelGapsY[gapIdx++] = gapVal;
        // Advance ptr to next number
        while (*ptr && *ptr != ' ')
          ptr++; // current number
        while (*ptr && *ptr == ' ')
          ptr++; // whitespace
        if (gapIdx >= 10)
          break; // Safety
      }
      continue;
    }
    str = "PanelGapsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      char *ptr = aline + strlen(str); // Skip "PanelGapsZ "
      int gapVal;
      int gapIdx = 0;
      while (sscanf(ptr, "%d", &gapVal) == 1) {
        panelGapsZ[gapIdx++] = gapVal;
        // Advance ptr to next number
        while (*ptr && *ptr != ' ')
          ptr++;
        while (*ptr && *ptr == ' ')
          ptr++;
        if (gapIdx >= 10)
          break;
      }
      continue;
    }
    str = "PanelShiftsFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, panelShiftsFile);
      continue;
    }
    str = "tolShifts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolShifts);
      continue;
    }
    str = "tolBC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolBC);
      continue;
    }
    str = "tolTilts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTilts);
      tolTiltX = tolTilts;
      tolTiltY = tolTilts;
      tolTiltZ = tolTilts;
      continue;
    }
    str = "tolTiltX ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTiltX);
      continue;
    }
    str = "tolTiltY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTiltY);
      continue;
    }
    str = "tolTiltZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTiltZ);
      continue;
    }
    str = "tolP ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP0);
      tolP1 = tolP0;
      tolP2 = tolP0;
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
    str = "FixPanelID ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &FixPanelID);
      continue;
    }
  }

  // Generate Panels
  Panel *panels = NULL;
  int nPanels = 0;
  if (nPanelsY > 0 && nPanelsZ > 0) {
    int ret = GeneratePanels(nPanelsY, nPanelsZ, panelSizeY, panelSizeZ,
                             panelGapsY, panelGapsZ, &panels, &nPanels);
    if (ret != 0) {
      printf("Error generating panels.\n");
      return 1;
    }

    if (strlen(panelShiftsFile) > 0) {
      char fullPath[MAX_LINE_LENGTH];
      sprintf(fullPath, "%s/%s", argv[1], panelShiftsFile);
      if (LoadPanelShifts(fullPath, nPanels, panels) != 0) {
        printf(
            "Warning: Could not load panel shifts from %s. Using 0 shifts.\n",
            fullPath);
      } else {
        printf("Loaded panel shifts from %s\n", fullPath);
      }
    }
  }

  // Read Grains.csv to find best nGrains
  FILE *GrainsF;
  char fnGrains[MAX_LINE_LENGTH];
  char *folder = argv[1];
  int i; // Declared here
  sprintf(fnGrains, "%s/Grains.csv", folder);
  GrainsF = fopen(fnGrains, "r");
  if (GrainsF == NULL) {
    printf("Error opening %s\n", fnGrains);
    return 1;
  }

  // First pass: Read all grains and sort
  struct GrainInfo *allGrains =
      malloc(sizeof(struct GrainInfo) * 100000); // Plenty space
  int nTotalGrains = 0;

  // Re-read file line by line
  while (fgets(aline, MAX_LINE_LENGTH, GrainsF) != NULL) {
    int id;
    double orient[9], pos[3], lat[6], error;
    // We need to read column 21 (0-indexed).
    // Current sscanf reads 19 values (0-18).
    // ID (0), Orient (1-9), Pos (10-12), Lat (13-18).
    // Column 19, 20 are unknown. 21 is FitError.
    // So we need to skip 2 values after Lat.
    double d1, d2;
    int nRead = sscanf(aline,
                       "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                       "%lf %lf %lf %lf %lf %lf %lf %lf",
                       &id, &orient[0], &orient[1], &orient[2], &orient[3],
                       &orient[4], &orient[5], &orient[6], &orient[7],
                       &orient[8], &pos[0], &pos[1], &pos[2], &lat[0], &lat[1],
                       &lat[2], &lat[3], &lat[4], &lat[5], &d1, &d2, &error);

    if (nRead >= 22) {
      allGrains[nTotalGrains].ID = id;
      allGrains[nTotalGrains].FitError = error;
      nTotalGrains++;
    }
  }
  fclose(GrainsF);

  qsort(allGrains, nTotalGrains, sizeof(struct GrainInfo), compareGrains);

  int nGrains = (nTotalGrains < nGrainsReq) ? nTotalGrains : nGrainsReq;
  printf("Selected top %d grains from %d total.\n", nGrains, nTotalGrains);
  for (i = 0; i < nGrains; i++) {
    printf("Grain %d: ID %d, Error %f\n", i, allGrains[i].ID,
           allGrains[i].FitError);
  }

  // Allocate GrainData structures
  struct GrainData *grainDataArray = malloc(sizeof(struct GrainData) * nGrains);

  // Loop over selected grains to load details
  for (int g = 0; g < nGrains; g++) {
    int GrainID = allGrains[g].ID;
    struct GrainData *gd = &grainDataArray[g];

    // Common params copy
    gd->NrPixels = NrPixels;
    gd->nOmeRanges = nOmeRanges;
    gd->nSpots = 0; // Will count later
    // ... fill others ...
    gd->RhoD = RhoD;
    gd->Lsd = Lsd;
    gd->px = px;
    gd->Wavelength = Wavelength;
    gd->MinEta = MinEta;
    for (i = 0; i < nOmeRanges; i++) {
      gd->OmegaRanges[i][0] = OmegaRanges[i][0];
      gd->OmegaRanges[i][1] = OmegaRanges[i][1];
    }
    for (i = 0; i < nBoxSizes; i++) {
      gd->BoxSizes[i][0] = BoxSizes[i][0];
      gd->BoxSizes[i][1] = BoxSizes[i][1];
      gd->BoxSizes[i][2] = BoxSizes[i][2];
      gd->BoxSizes[i][3] = BoxSizes[i][3];
    }

    // Reload Grains.csv to get params for THIS grain
    // This is inefficient but robust.
    GrainsF = fopen(fnGrains, "r");
    double Orient[9], Pos[3], LatC[6], Euler[3];
    while (fgets(aline, MAX_LINE_LENGTH, GrainsF) != NULL) {
      int id_lines;
      sscanf(aline,
             "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
             "%lf %lf %lf",
             &id_lines, &Orient[0], &Orient[1], &Orient[2], &Orient[3],
             &Orient[4], &Orient[5], &Orient[6], &Orient[7], &Orient[8],
             &Pos[0], &Pos[1], &Pos[2], &LatC[0], &LatC[1], &LatC[2], &LatC[3],
             &LatC[4], &LatC[5]);
      if (id_lines == GrainID)
        break;
    }
    fclose(GrainsF);

    double Orient33[3][3];
    int j; // Declared j
    for (i = 0; i < 3; i++)
      for (j = 0; j < 3; j++)
        Orient33[i][j] = Orient[i * 3 + j];
    OrientMat2Euler(Orient33, Euler);

    for (i = 0; i < 3; i++)
      gd->Ini[i] = Pos[i];
    for (i = 0; i < 3; i++)
      gd->Ini[i + 3] = Euler[i];
    for (i = 0; i < 6; i++)
      gd->Ini[i + 6] = LatC[i];

    // Load Spots
    gd->SpotInfoAll = allocMatrix(MaxNSpotsBest, 5);
    char fnSpotMatrix[MAX_LINE_LENGTH];
    sprintf(fnSpotMatrix, "%s/SpotMatrix.csv", folder);
    FILE *SpotMF = fopen(fnSpotMatrix, "r");
    double YZOme[3];
    int Rnr, nS = 0, SpID, id_spot;
    fgets(aline, MAX_LINE_LENGTH, SpotMF); // Skip header
    while (fgets(aline, MAX_LINE_LENGTH, SpotMF) != NULL) {
      sscanf(aline, "%d %d %s %lf %lf %lf %s %d", &id_spot, &SpID, dummy,
             &YZOme[0], &YZOme[1], &YZOme[2], dummy, &Rnr);
      if (id_spot == GrainID) {
        gd->SpotInfoAll[nS][0] = (double)SpID;
        gd->SpotInfoAll[nS][1] = (double)Rnr;
        gd->SpotInfoAll[nS][2] = YZOme[0];
        gd->SpotInfoAll[nS][3] = YZOme[1];
        gd->SpotInfoAll[nS][4] = YZOme[2];

        // Undo panel shifts
        if (nPanels > 0) {
          int pIdx = GetPanelIndex(YZOme[0], YZOme[1], nPanels, panels);
          if (pIdx >= 0) {
            gd->SpotInfoAll[nS][2] -= panels[pIdx].dY;
            gd->SpotInfoAll[nS][3] -= panels[pIdx].dZ;
          }
        }
        nS++;
        if (nS >= MaxNSpotsBest)
          break;
      }
    }
    fclose(SpotMF);
    gd->nSpots = nS;
    printf("Loaded %d spots for Grain %d\n", nS, GrainID);

    // Load HKLs
    gd->hkls = allocMatrix(MaxNSpotsBest, 4);
    gd->nhkls = 0;
    char hklfn[2048];
    sprintf(hklfn, "%s/hkls.csv", folder);
    FILE *hklf = fopen(hklfn, "r");
    fgets(aline, MAX_LINE_LENGTH, hklf);
    int h, kt, l, RNr;
    while (fgets(aline, MAX_LINE_LENGTH, hklf) != NULL) {
      sscanf(aline, "%d %d %d %s %d", &h, &kt, &l, dummy, &RNr);
      for (i = 0; i < cs; i++) { // cs is nRings
        if (RNr == RingNumbers[i]) {
          if (gd->nhkls < MaxNSpotsBest) {
            gd->hkls[gd->nhkls][0] = h;
            gd->hkls[gd->nhkls][1] = kt;
            gd->hkls[gd->nhkls][2] = l;
            gd->hkls[gd->nhkls][3] = RingNumbers[i];
            gd->nhkls++;
          }
        }
      }
    }
    fclose(hklf);
    gd->nRings = cs;
  }

  // Calculate spot counts per panel (aggregated)
  int *spotsPerPanel = NULL;
  if (nPanels > 0) {
    spotsPerPanel = calloc(nPanels, sizeof(int));
    for (int g = 0; g < nGrains; g++) {
      struct GrainData *gd = &grainDataArray[g];
      for (i = 0; i < gd->nSpots; i++) {
        // Using shifted back coords approx original panel location
        int pIdx = GetPanelIndex(gd->SpotInfoAll[i][2], gd->SpotInfoAll[i][3],
                                 nPanels, panels);
        if (pIdx >= 0 && pIdx < nPanels)
          spotsPerPanel[pIdx]++;
      }
    }
    printf("\n******************* Indices per Panel (Visual Layout: Z^ Y>) "
           "*******************\n");
    printf("                        Anchored Panel ID: %d \n", FixPanelID);
    double charAspect = 0.5;         // Width / Height
    double textWidthPerPanel = 14.0; // "|  12 (12345) " is 14 chars
    double visualWidthPoints = nPanelsY * textWidthPerPanel * charAspect;
    double targetHeightPoints =
        visualWidthPoints; // Assuming aspect ratio 1:1 if NrPixels is single
                           // value
    if (NrPixels > 0)
      targetHeightPoints = visualWidthPoints * 1.0;

    // Reduce vertical spacing factor significantly (0.15 factor)
    int linesPerRow = (int)(targetHeightPoints / nPanelsZ * 0.15 + 0.5);
    if (linesPerRow < 1)
      linesPerRow = 1;

    for (int z = nPanelsZ - 1; z >= 0; z--) {
      for (int l = 0; l < linesPerRow; l++) {
        if (l == linesPerRow / 2)
          printf("Z=%-2d | ", z);
        else
          printf("     | ");

        if (l == linesPerRow / 2) {
          for (int y = 0; y < nPanelsY; y++) {
            int pIdx = y * nPanelsZ + z;
            if (pIdx < nPanels) {
              printf("| %3d (%5d) ", pIdx, spotsPerPanel[pIdx]);
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
    for (int y = 0; y < nPanelsY; y++) {
      // Data block is 14 chars: "| %3d (%5d) "
      // We want Y label centered: 5 spaces + "Y=%-2d" (4) + 5 spaces = 14
      printf("     Y=%-2d     ", y);
    }
    printf("\n");
    printf("*******************************************************************"
           "*************\n\n");
  }

  // Group Setup parameters
  // Non Optimized: NonOptP: double 5
  // Optimized OptP[10]
  double NonOptP[5] = {RhoD, Lsd, px, Wavelength, MinEta};
  // int NonOptPInt[5] = {NrPixels, nOmeRanges, nRings, nSpots, nhkls}; //
  // Removed, unused and caused errors
  double OptP[10] = {tx, ty, tz, yBC, zBC, wedge, p0, p1, p2, p3};
  double tols[22] = {
      250,
      250,
      250,
      deg2rad * 0.0005,
      deg2rad * 0.0005,
      deg2rad * 0.0005,
      1,
      1,
      1,
      1,
      1,
      1,
      tolTiltX,
      tolTiltY,
      tolTiltZ,
      tolBC,
      tolBC,
      0.00001,
      tolP0,
      tolP1,
      tolP2,
      tolP3,
  }; // 250 microns for position, 0.0005 degrees for orient, 1 % for
     // latticeParameter, 1 degree for tilts, 1 pixel for yBC,
     // 1 pixel for zBC, 0.00001 degree for wedge, 1E-3 for p0,p1,p2, 45 for
     // p3

  // Now call a function with all the info which will optimize parameters
  // Arguments: Ini(12), OptP(6), NonOptP, RingNumbers,  SpotInfoAll,
  // OmegaRanges,
  //			  BoxSizes, hkls
  // CalcAngleErrors would need Y,Z,Ome before wedge correction.
  // Everything till CorrectTiltSpatialDistortion function in FitTiltBCLsd
  // Perform Optimization
  // Prepare Output
  double *Out; // Declared Out
  Out = malloc(((nGrains * 12 + 10) + nPanels * 2) *
               sizeof(double)); // Fixed size calc too just in case
  FitMultipleGrains(grainDataArray, nGrains, OptP, NonOptP, tols, Out, panels,
                    nPanels, tolShifts, spotsPerPanel, FixPanelID);

  if (nPanels > 1) {
    char fullPath[MAX_LINE_LENGTH];
    sprintf(fullPath, "%s/PanelShiftsOptimizedMulti.txt", folder);
    SavePanelShifts(fullPath, nPanels, panels);
    printf("Saved optimized panel shifts to %s\n", fullPath);
  }

  if (spotsPerPanel != NULL)
    free(spotsPerPanel);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
}