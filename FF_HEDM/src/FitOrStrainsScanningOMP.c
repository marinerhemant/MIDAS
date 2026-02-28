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

#include "MIDAS_Math.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MaxNSpots 60000000
#define MaxNSpotsBest 5000
#define MaxNHKLS 5000
#define EPS 1E-12
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#include "MIDAS_Limits.h"
#define MAXNOMEGARANGES MAX_N_OMEGA_RANGES

// Detector mapping globals
int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;
double DetParams[4][10];
#define MAX_N_OMEGA_RANGES 2000
double WeightMask = 1.0;
double WeightFitRMSE = 1.0;

// Dynamic spot reassignment: bin data structures (same layout as
// IndexerScanningOMP)
static double
    *ObsSpotsLab;     // mmap of Spots.bin (10 doubles per spot for scanning)
static int *BinData;  // mmap of Data.bin (spot row indices + scan numbers)
static int *nBinData; // mmap of nData.bin (count, offset pairs)
static double gEtaBinSize = 2.0;
static double gOmeBinSize = 2.0;
static int g_n_ring_bins = 0;
static int g_n_eta_bins = 0;
static int g_n_ome_bins = 0;
static int gNSpotsBin = 0; // total spots (from Spots.bin)

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAX_N_OMEGA_RANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAX_N_OMEGA_RANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots);

// check() is now provided by MIDAS_Limits.h

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
      for (int k = 0; k < i; k++)
        free(arr[k]);
      free(arr);
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
      for (int k = 0; k < i; k++)
        free(arr[k]);
      free(arr);
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
    MatrixMultF(B, ginit, GCart);
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

static inline double CalcEtaAngleLocal(double y, double z) {
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
  double eta = CalcEtaAngleLocal(ysi, zsi);
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
  double Eta = CalcEtaAngleLocal(k2, k3);
  double Sin_Eta = sin(deg2rad * Eta);
  double Cos_Eta = cos(deg2rad * Eta);
  *ysOut = -RingRadius * Sin_Eta;
  *zsOut = RingRadius * Cos_Eta;
  *OmegaOut = Omega;
}

static inline void SpotToGv(double xi, double yi, double zi, double Omega,
                            double theta, double *g1, double *g2, double *g3) {
  double CosOme = cosd(Omega), SinOme = sind(Omega),
         eta = CalcEtaAngleLocal(yi, zi), TanEta = tand(-eta),
         SinTheta = sind(theta);
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
    nTheorSpotsYZWER = 0;
    GObs[0] = SpotsYZOGCorr[sp][3];
    GObs[1] = SpotsYZOGCorr[sp][4];
    GObs[2] = SpotsYZOGCorr[sp][5];
    NormGObs = CalcNorm3(GObs[0], GObs[1], GObs[2]);
    for (i = 0; i < nTspots; i++) {
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
        Angles[nTheorSpotsYZWER] = fabs(acosd(Numers / Denoms));
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
    if (minAngle < 1) {
      double maskTouched = spotsYZO[sp][8];
      double FitRMSE = spotsYZO[sp][9];
      double weight = 1.0;
      if (maskTouched > 0.5)
        weight *= WeightMask;
      weight *= exp(-FitRMSE * WeightFitRMSE);
      MatchDiff[nMatched][0] = minAngle * weight;
      MatchDiff[nMatched][1] = diffLenM * weight;
      MatchDiff[nMatched][2] = diffOmeM * weight;
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
      SpotsComp[nMatched][22] = SpotsYZOGCorr[sp][6];
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
  double **hkls;
  double **hklsIn2;
  double **spotsYZO;
  double **TheorSpots;
  double **SpotsYZOGCorr;
  double **TheorSpotsYZWE;
  double *Angles;
  int nTspots;
  int *SpotLookup;
  int MaxSpnr;
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

static inline double FitErrorsOrientStrains(
    double x[9], int nSpotsComp, double **spotsYZO, int nhkls, double **hklsIn,
    double Lsd, double Wavelength, int nOmeRanges,
    double OmegaRanges[MAXNOMEGARANGES][2], double BoxSizes[MAXNOMEGARANGES][4],
    double MinEta, double wedge, double chi, double Pos[3],
    struct OptimizeScratch *scratch) {
  int i, j;
  int nrMatchedIndexer = nSpotsComp;
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[3 + i];

  // Use scratch memory
  double **hklsIn2 = scratch->hklsIn2;
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hklsIn[i][j];

  double **hkls = scratch->hkls;
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hkls);

  double OrientMatrix[3][3], EulerIn[3];
  EulerIn[0] = x[0];
  EulerIn[1] = x[1];
  EulerIn[2] = x[2];
  Euler2OrientMat(EulerIn, OrientMatrix);

  int nTspots = 0, nrSp;
  double **TheorSpots = scratch->TheorSpots;
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);

  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;
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
    SpotsYZOGCorr[nrSp][4] = CalcNorm3(g1, g2, g3);
  }

  int *SpotLookup = scratch->SpotLookup;
  int MaxSpnr = scratch->MaxSpnr;
  for (i = 0; i <= MaxSpnr; i++)
    SpotLookup[i] = -1;

  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  for (i = 0; i < nTspots; i++) {
    for (j = 0; j < 3; j++) {
      TheorSpotsYZWE[i][j] = TheorSpots[i][j + 3];
    }
    TheorSpotsYZWE[i][3] = TheorSpots[i][8];
    TheorSpotsYZWE[i][4] = CalcNorm3(TheorSpotsYZWE[i][0], TheorSpotsYZWE[i][1],
                                     TheorSpotsYZWE[i][2]);
    int spnr = (int)TheorSpots[i][8];
    if (spnr >= 0 && spnr <= MaxSpnr && SpotLookup[spnr] == -1) {
      SpotLookup[spnr] = i;
    }
  }

  int sp, nTheorSpotsYZWER;
  double GObs[3], GTheors[3], NormGObs, NormGTheors, DotGs, Numers, Denoms,
      *Angles, minAngle, Error = 0;

  Angles = scratch->Angles;
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    nTheorSpotsYZWER = 0;
    GObs[0] = SpotsYZOGCorr[sp][0];
    GObs[1] = SpotsYZOGCorr[sp][1];
    GObs[2] = SpotsYZOGCorr[sp][2];
    NormGObs = SpotsYZOGCorr[sp][4];
    int Spnr = (int)SpotsYZOGCorr[sp][3];

    if (Spnr >= 0 && Spnr <= MaxSpnr) {
      int t_idx_start = SpotLookup[Spnr];
      if (t_idx_start != -1) {
        for (i = t_idx_start; i < nTspots; i++) {
          if ((int)TheorSpotsYZWE[i][3] != Spnr)
            break;
          GTheors[0] = TheorSpotsYZWE[i][0];
          GTheors[1] = TheorSpotsYZWE[i][1];
          GTheors[2] = TheorSpotsYZWE[i][2];
          DotGs = ((GTheors[0] * GObs[0]) + (GTheors[1] * GObs[1]) +
                   (GTheors[2] * GObs[2]));
          NormGTheors = TheorSpotsYZWE[i][4];
          Numers = DotGs;
          Denoms = NormGObs * NormGTheors;
          double ratio = Numers / Denoms;
          if (ratio > 1.0)
            ratio = 1.0;
          if (ratio < -1.0)
            ratio = -1.0;
          if (ratio >= 0.9975640502598242) {
            Angles[nTheorSpotsYZWER] = fabs(acosd(ratio));
            nTheorSpotsYZWER++;
          }
        }
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
  return Error;
}

static inline double
FitErrorsStrains(double x[6], int nSpotsComp, double **spotsYZO, int nhkls,
                 double **hklsIn, double Lsd, double Wavelength, int nOmeRanges,
                 double OmegaRanges[MAXNOMEGARANGES][2],
                 double BoxSizes[MAXNOMEGARANGES][4], double MinEta,
                 double wedge, double chi, double Pos[3], double EulerIn[3],
                 struct OptimizeScratch *scratch) {
  int i, j;
  int nrMatchedIndexer = nSpotsComp;
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[i];

  // Use scratch memory
  double **hklsIn2 = scratch->hklsIn2;
  for (i = 0; i < nhkls; i++)
    for (j = 0; j < 7; j++)
      hklsIn2[i][j] = hklsIn[i][j];

  double **hkls = scratch->hkls;
  CorrectHKLsLatC(LatC, hklsIn2, nhkls, Lsd, Wavelength, hkls);

  double OrientMatrix[3][3];
  Euler2OrientMat(EulerIn, OrientMatrix);

  int nTspots = 0, nrSp;
  double **TheorSpots = scratch->TheorSpots;
  CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                       BoxSizes, &nTspots, OrientMatrix, TheorSpots);

  double **SpotsYZOGCorr = scratch->SpotsYZOGCorr;
  int *SpotLookup = scratch->SpotLookup;
  int MaxSpnr = scratch->MaxSpnr;
  for (i = 0; i <= MaxSpnr; i++)
    SpotLookup[i] = -1;

  double **TheorSpotsYZWE = scratch->TheorSpotsYZWE;
  for (i = 0; i < nTspots; i++) {
    TheorSpotsYZWE[i][0] = TheorSpots[i][0];
    TheorSpotsYZWE[i][1] = TheorSpots[i][1];
    TheorSpotsYZWE[i][2] = TheorSpots[i][8];
    int spnr = (int)TheorSpots[i][8];
    if (spnr >= 0 && spnr <= MaxSpnr && SpotLookup[spnr] == -1) {
      SpotLookup[spnr] = i;
    }
  }

  int sp;
  double PosObs[2], PosTheor[2];
  double Error = 0;
  for (sp = 0; sp < nrMatchedIndexer; sp++) {
    PosObs[0] = SpotsYZOGCorr[sp][0];
    PosObs[1] = SpotsYZOGCorr[sp][1];
    int Spnr = (int)SpotsYZOGCorr[sp][2];
    if (Spnr >= 0 && Spnr <= MaxSpnr) {
      int t_idx = SpotLookup[Spnr];
      if (t_idx != -1) {
        PosTheor[0] = TheorSpotsYZWE[t_idx][0];
        PosTheor[1] = TheorSpotsYZWE[t_idx][1];
        Error +=
            CalcNorm2((PosObs[0] - PosTheor[0]), (PosObs[1] - PosTheor[1]));
      }
    }
  }
  return Error;
}

static double problem_function_OrientIni(unsigned n, const double *x,
                                         double *grad, void *f_data_trial) {
  int i;
  struct data_FitOrientIni *f_data = (struct data_FitOrientIni *)f_data_trial;
  if (n > 24)
    return 1e30; // Max fitting degrees of freedom
  double XIn[24];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsOrientStrains(
      XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
      f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges,
      f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi, f_data->Pos,
      f_data->scratch);
}

static double problem_function_StrainIni(unsigned n, const double *x,
                                         double *grad, void *f_data_trial) {
  int i;
  struct data_FitStrainIni *f_data = (struct data_FitStrainIni *)f_data_trial;
  if (n > 24)
    return 1e30; // Max fitting degrees of freedom
  double XIn[24];
  for (i = 0; i < n; i++)
    XIn[i] = x[i];
  return FitErrorsStrains(
      XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
      f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges,
      f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi, f_data->Pos,
      f_data->Orient, f_data->scratch);
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
  f_data.spotsYZO = allocMatrix(nSpotsComp, 11);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 11; j++)
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
  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }

  // Allocate scratch memory
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 11);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 5);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 5);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));
  int maxSpnr = 0;
  for (int sp = 0; sp < nSpotsComp; sp++) {
    if ((int)spotsYZO[sp][8] > maxSpnr)
      maxSpnr = (int)spotsYZO[sp][8];
  }
  f_data.scratch->MaxSpnr = maxSpnr;
  f_data.scratch->SpotLookup = malloc((maxSpnr + 1) * sizeof(int));

  // Precompute SpotsYZOGCorr (constant for FitOrientIni since Pos is constant)
  double DisplY, DisplZ, ys, zs, Omega, yt, zt;
  for (int nrSp = 0; nrSp < nSpotsComp; nrSp++) {
    DisplacementInTheSpot(Pos[0], Pos[1], Pos[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    f_data.scratch->SpotsYZOGCorr[nrSp][0] = ys;
    f_data.scratch->SpotsYZOGCorr[nrSp][1] = zs;
    f_data.scratch->SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
  }

  struct data_FitOrientIni *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitOrientIni *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function_OrientIni;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double minf = 0, prev_minf = 1e30;
  for (int run = 0; run < 2; run++) {
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
    minf = config.min_function_val;
    if (fabs(prev_minf - minf) < 1e-4)
      break;
    prev_minf = minf;
  }
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);

  // Free scratch memory
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
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
  f_data.spotsYZO = allocMatrix(nSpotsComp, 11);
  for (i = 0; i < nSpotsComp; i++)
    for (j = 0; j < 11; j++)
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
  for (i = 0; i < n; i++) {
    x[i] = X0[i];
    xl[i] = lb[i];
    xu[i] = ub[i];
  }

  // Allocate scratch memory
  f_data.scratch = malloc(sizeof(struct OptimizeScratch));
  f_data.scratch->hkls = allocMatrix(nhkls, 7);
  f_data.scratch->hklsIn2 = allocMatrix(nhkls, 7);
  f_data.scratch->spotsYZO = allocMatrix(nSpotsComp, 11);
  f_data.scratch->TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  f_data.scratch->SpotsYZOGCorr = allocMatrix(nSpotsComp, 5);
  f_data.scratch->TheorSpotsYZWE = allocMatrix(MaxNSpotsBest, 5);
  f_data.scratch->Angles = malloc(MaxNSpotsBest * sizeof(double));
  int maxSpnr = 0;
  for (int sp = 0; sp < nSpotsComp; sp++) {
    if ((int)spotsYZO[sp][8] > maxSpnr)
      maxSpnr = (int)spotsYZO[sp][8];
  }
  f_data.scratch->MaxSpnr = maxSpnr;
  f_data.scratch->SpotLookup = malloc((maxSpnr + 1) * sizeof(int));

  // Precompute SpotsYZOGCorr (constant for FitStrainIni since Pos is constant)
  double DisplY, DisplZ, ys, zs, Omega, yt, zt, Radius, Theta, lenK, g1, g2, g3;
  for (int nrSp = 0; nrSp < nSpotsComp; nrSp++) {
    DisplacementInTheSpot(Pos[0], Pos[1], Pos[2], Lsd, spotsYZO[nrSp][5],
                          spotsYZO[nrSp][6], spotsYZO[nrSp][4], wedge, chi,
                          &DisplY, &DisplZ);
    yt = spotsYZO[nrSp][5] - DisplY;
    zt = spotsYZO[nrSp][6] - DisplZ;
    CorrectForOme(yt, zt, Lsd, spotsYZO[nrSp][4], Wavelength, wedge, &ys, &zs,
                  &Omega);
    Radius = sqrt(ys * ys + zs * zs);
    Theta = atan2(zs, ys);
    lenK = sqrt((Radius * Radius) + (Lsd * Lsd));
    g1 = (-Lsd / lenK) + 1;
    g2 = -ys / lenK;
    g3 = -zs / lenK;
    f_data.scratch->SpotsYZOGCorr[nrSp][0] = g1;
    f_data.scratch->SpotsYZOGCorr[nrSp][1] = g2;
    f_data.scratch->SpotsYZOGCorr[nrSp][2] = g3;
    f_data.scratch->SpotsYZOGCorr[nrSp][3] = spotsYZO[nrSp][8];
  }
  struct data_FitStrainIni *f_datat;
  f_datat = &f_data;
  void *trp = (struct data_FitStrainIni *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function_StrainIni;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double minf = 0, prev_minf = 1e30;
  for (int run = 0; run < 2; run++) {
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
    minf = config.min_function_val;
    if (fabs(prev_minf - minf) < 1e-4)
      break;
    prev_minf = minf;
  }
  //~ for (i=0;i<n;i++) printf("%f ",x[i]);
  //~ printf("%10.30f \n", minf);
  for (i = 0; i < n; i++)
    XFit[i] = x[i];
  FreeMemMatrix(f_data.spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.hkls, nhkls);

  // Free scratch memory
  FreeMemMatrix(f_data.scratch->hkls, nhkls);
  FreeMemMatrix(f_data.scratch->hklsIn2, nhkls);
  FreeMemMatrix(f_data.scratch->spotsYZO, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpots, MaxNSpotsBest);
  FreeMemMatrix(f_data.scratch->SpotsYZOGCorr, nSpotsComp);
  FreeMemMatrix(f_data.scratch->TheorSpotsYZWE, MaxNSpotsBest);
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

struct data_StrainFit {
  int nspots;
  double A[5000][6];
  double B[5000];
};

static double problem_function_strain(unsigned n, const double *x, double *grad,
                                      void *f_data_trial) {
  struct data_StrainFit *f_data = (struct data_StrainFit *)f_data_trial;
  int nspots = f_data->nspots;
  int i, j;
  double TotDiff = 0, InterDiff, Diff;
  for (i = 0; i < nspots; i++) {
    InterDiff = 0;
    for (j = 0; j < 6; j++) {
      InterDiff += f_data->A[i][j] * x[j];
    }
    Diff = InterDiff - f_data->B[i];
    TotDiff += Diff * Diff;
  }
  //~ printf("%lf\n",TotDiff);
  return 1000000 * TotDiff;
}

inline int StrainTensorKenesei(int nspots, double **SpotsInfo, double Distance,
                               double wavelength, int nhkls, double **hkls,
                               double StrainTensorSample[3][3],
                               double **SpotMatrix, double *RetVal) {
  int i, j;
  struct data_StrainFit mydata;
  double gobs[3], lenGobs;
  mydata.nspots = nspots;
  //~ int id;
  int ringNr;
  double ds0 = 0, dsObs;
  for (i = 0; i < nspots; i++) {
    lenGobs = CalcNorm3(SpotsInfo[i][4], SpotsInfo[i][5], SpotsInfo[i][6]);
    dsObs =
        wavelength /
        (2 *
         sin(atan((CalcNorm2(SpotsInfo[i][1], SpotsInfo[i][2])) / Distance) /
             2));
    gobs[0] = SpotsInfo[i][4] / lenGobs;
    gobs[1] = SpotsInfo[i][5] / lenGobs;
    gobs[2] = SpotsInfo[i][6] / lenGobs;
    ringNr = (int)SpotsInfo[i][22];
    for (j = 0; j < nhkls; j++) {
      if ((int)hkls[j][6] == ringNr) {
        ds0 = hkls[j][3];
        break;
      }
    }
    if (ds0 == 0) {
      printf("This ringNr was not detected: %d. Something is wrong. Please "
             "contact hsharma@anl.gov to investigate.\n",
             ringNr);
      return 0;
    }
    mydata.B[i] = (dsObs - ds0) / ds0;
    mydata.A[i][0] = gobs[0] * gobs[0];
    mydata.A[i][1] = gobs[1] * gobs[1];
    mydata.A[i][2] = gobs[2] * gobs[2];
    mydata.A[i][3] = gobs[0] * gobs[1] * 2;
    mydata.A[i][4] = gobs[0] * gobs[2] * 2;
    mydata.A[i][5] = gobs[1] * gobs[2] * 2;
    ds0 = 0;
  }
  int n = 6;
  double x[n], xl[n], xu[n];
  for (i = 0; i < n; i++) {
    x[i] = 0.0;
    xl[i] = -0.01;
    xu[i] = 0.01;
  }
  struct data_StrainFit *f_datat;
  f_datat = &mydata;
  void *trp = (struct data_StrainFit *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function_strain;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  double minf = config.min_function_val;
  *RetVal = problem_function_strain(n, x, NULL, trp) / (nspots * 1000000);
  *RetVal = sqrt(*RetVal);
  StrainTensorSample[0][0] = x[0];
  StrainTensorSample[0][1] = x[3];
  StrainTensorSample[0][2] = x[4];
  StrainTensorSample[1][0] = x[3];
  StrainTensorSample[1][1] = x[1];
  StrainTensorSample[1][2] = x[5];
  StrainTensorSample[2][0] = x[4];
  StrainTensorSample[2][1] = x[5];
  StrainTensorSample[2][2] = x[2];
  for (i = 0; i < nspots; i++) {
    for (j = 0; j < 23; j++) {
      SpotMatrix[i][j] = SpotsInfo[i][j];
    }
    SpotMatrix[i][23] =
        (mydata.B[i] -
         (mydata.A[i][0] * x[0] + // No fabs, to get feeling of sign.
          mydata.A[i][1] * x[1] + mydata.A[i][2] * x[2] +
          mydata.A[i][3] * x[3] + mydata.A[i][4] * x[4] +
          mydata.A[i][5] * x[5])) *
        1000000;
  }
  return 1;
}

static int ReassignSpotsFromBins(
    double x[12], int nhkls, double **hklsIn, double Lsd, double Wavelength,
    int nOmegaRanges, double OmegaRange[MAXNOMEGARANGES][2],
    double BoxSize[MAXNOMEGARANGES][4], double MinEta, double wedge, double chi,
    double **spotsYZO, int maxSpots, double *AllSpotsPtr, int totalNSpots) {
  if (ObsSpotsLab == NULL || BinData == NULL || nBinData == NULL)
    return 0;
  int i, j;
  // Compute theoretical spots for current orientation
  double LatC[6];
  for (i = 0; i < 6; i++)
    LatC[i] = x[6 + i];
  double **hkls;
  hkls = allocMatrix(MaxNSpotsBest, 7);
  CorrectHKLsLatC(LatC, hklsIn, nhkls, Lsd, Wavelength, hkls);
  double OrientMatrix[3][3], EulerIn[3];
  EulerIn[0] = x[3];
  EulerIn[1] = x[4];
  EulerIn[2] = x[5];
  Euler2OrientMat(EulerIn, OrientMatrix);
  int nTspots;
  double **TheorSpots;
  TheorSpots = allocMatrix(MaxNSpotsBest, 9);
  CalcDiffractionSpots(Lsd, MinEta, OmegaRange, nOmegaRanges, hkls, nhkls,
                       BoxSize, &nTspots, OrientMatrix, TheorSpots);
  // For each theoretical spot, search the bin structure for the best observed
  // match
  int nMatched = 0;
  int usedSpotIDs[MaxNSpotsBest];
  for (i = 0; i < MaxNSpotsBest; i++)
    usedSpotIDs[i] = -1;
  for (int sp = 0; sp < nTspots && nMatched < maxSpots; sp++) {
    int RingNr = (int)TheorSpots[sp][7];
    double theorOmega = TheorSpots[sp][2];
    double theorEta = CalcEtaAngleLocal(-TheorSpots[sp][0], TheorSpots[sp][1]);
    int iRing = RingNr - 1;
    if (iRing < 0 || iRing >= g_n_ring_bins)
      continue;
    int iEta = (int)floor((180.0 + theorEta) / gEtaBinSize);
    int iOme = (int)floor((180.0 + theorOmega) / gOmeBinSize);
    if (iEta < 0)
      iEta = 0;
    if (iEta >= g_n_eta_bins)
      iEta = g_n_eta_bins - 1;
    if (iOme < 0)
      iOme = 0;
    if (iOme >= g_n_ome_bins)
      iOme = g_n_ome_bins - 1;
    long long int Pos = (long long int)iRing * g_n_eta_bins * g_n_ome_bins +
                        iEta * g_n_ome_bins + iOme;
    int nInBin = nBinData[Pos * 2];
    int DataPos = nBinData[Pos * 2 + 1];
    if (nInBin == 0)
      continue;
    // Compute theoretical g-vector for angular comparison
    double g1t = TheorSpots[sp][3], g2t = TheorSpots[sp][4],
           g3t = TheorSpots[sp][5];
    double normGt = CalcNorm3(g1t, g2t, g3t);
    double bestDiffOme = 1e9;
    int bestRow = -1;
    for (int iSpot = 0; iSpot < nInBin; iSpot++) {
      // Data.bin for scanning stores (rowno, scanno) pairs
      int spotRow = BinData[(DataPos + iSpot) * 2];
      if (spotRow < 0 || spotRow >= gNSpotsBin)
        continue;
      // Check if already assigned to a previous theoretical spot
      int alreadyUsed = 0;
      for (int u = 0; u < nMatched; u++) {
        if (usedSpotIDs[u] == spotRow) {
          alreadyUsed = 1;
          break;
        }
      }
      if (alreadyUsed)
        continue;
      double obsOmega = ObsSpotsLab[spotRow * 10 + 2];
      double diffOme = fabs(theorOmega - obsOmega);
      if (diffOme < 5.0 && diffOme < bestDiffOme) {
        bestDiffOme = diffOme;
        bestRow = spotRow;
      }
    }
    if (bestRow >= 0) {
      usedSpotIDs[nMatched] = bestRow;
      // Populate spotsYZO from AllSpots (ExtraInfo.bin, 16 doubles/row)
      size_t spos = (size_t)bestRow;
      if (spos < (size_t)totalNSpots) {
        spotsYZO[nMatched][0] = AllSpotsPtr[spos * 16 + 0];  // YLab
        spotsYZO[nMatched][1] = AllSpotsPtr[spos * 16 + 1];  // ZLab
        spotsYZO[nMatched][2] = AllSpotsPtr[spos * 16 + 2];  // Omega
        spotsYZO[nMatched][3] = AllSpotsPtr[spos * 16 + 4];  // SpotID
        spotsYZO[nMatched][4] = AllSpotsPtr[spos * 16 + 8];  // OmegaIni
        spotsYZO[nMatched][5] = AllSpotsPtr[spos * 16 + 9];  // YOrig
        spotsYZO[nMatched][6] = AllSpotsPtr[spos * 16 + 10]; // ZOrig
        spotsYZO[nMatched][7] = AllSpotsPtr[spos * 16 + 5];  // RingNr
        spotsYZO[nMatched][8] = AllSpotsPtr[spos * 16 + 14]; // maskTouched
        spotsYZO[nMatched][9] = AllSpotsPtr[spos * 16 + 15]; // FitRMSE
        spotsYZO[nMatched][10] = 0;
        nMatched++;
      }
    }
  }
  FreeMemMatrix(hkls, MaxNSpotsBest);
  FreeMemMatrix(TheorSpots, MaxNSpotsBest);
  return nMatched;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Supply a parameter file, blockNr, nBlocks, nSpotsToIndex, numProcs "
           "as arguments.\n");
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
  int RingNumbers[200], cs = 0, cs2 = 0, nOmeRanges = 0, nBoxSizes = 0;
  double Rsample, Hbeam, RingRadii[200], MargABC = 0.3, MargABG = 0.3;
  char OutputFolder[1024], ResultFolder[1024];
  int TopLayer = 0, TakeGrainMax = 0;
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
    str = "EtaBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &gEtaBinSize);
      continue;
    }
    str = "OmeBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &gOmeBinSize);
      continue;
    }
  }
  fclose(fileParam);
  double *AllSpots;
  int fd;
  struct stat s;
  int status;
  size_t size;
  char tmpstr[2048];
  sprintf(tmpstr, "%s", ResultFolder);
  char filename[2048], *cwd = dirname(tmpstr);
  sprintf(filename, "%s/ExtraInfo.bin", cwd);
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  AllSpots = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(AllSpots == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  int nSpots = (int)size / (16 * sizeof(double));
  // Mmap Spots.bin, Data.bin and nData.bin for dynamic spot reassignment
  {
    char binFN[2048];
    struct stat bs;
    // Spots.bin (10 doubles per spot for scanning)
    sprintf(binFN, "%s/Spots.bin", cwd);
    int fdSpots = open(binFN, O_RDONLY);
    if (fdSpots >= 0) {
      fstat(fdSpots, &bs);
      ObsSpotsLab = mmap(0, bs.st_size, PROT_READ, MAP_SHARED, fdSpots, 0);
      if (ObsSpotsLab == MAP_FAILED)
        ObsSpotsLab = NULL;
      gNSpotsBin = (int)(bs.st_size / (10 * sizeof(double)));
      printf("Spots.bin mapped: %d spots\n", gNSpotsBin);
    } else {
      ObsSpotsLab = NULL;
      printf("Warning: Spots.bin not found, dynamic reassignment disabled.\n");
    }
    // Data.bin
    sprintf(binFN, "%s/Data.bin", cwd);
    int fdData = open(binFN, O_RDONLY);
    if (fdData >= 0) {
      fstat(fdData, &bs);
      BinData = mmap(0, bs.st_size, PROT_READ, MAP_SHARED, fdData, 0);
      if (BinData == MAP_FAILED)
        BinData = NULL;
      printf("Data.bin mapped: %lld bytes\n", (long long int)bs.st_size);
    } else {
      BinData = NULL;
    }
    // nData.bin
    sprintf(binFN, "%s/nData.bin", cwd);
    int fdNData = open(binFN, O_RDONLY);
    if (fdNData >= 0) {
      fstat(fdNData, &bs);
      nBinData = mmap(0, bs.st_size, PROT_READ, MAP_SHARED, fdNData, 0);
      if (nBinData == MAP_FAILED)
        nBinData = NULL;
      printf("nData.bin mapped: %lld bytes\n", (long long int)bs.st_size);
    } else {
      nBinData = NULL;
    }
    // Compute bin dimensions
    g_n_eta_bins = (int)ceil(360.0 / gEtaBinSize);
    g_n_ome_bins = (int)ceil(360.0 / gOmeBinSize);
    int highRing = 0;
    for (int ri = 0; ri < cs; ri++) {
      if (RingNumbers[ri] > highRing)
        highRing = RingNumbers[ri];
    }
    g_n_ring_bins = highRing;
    printf("Bin dims: rings=%d, eta=%d, ome=%d (EtaBinSize=%.1f, "
           "OmeBinSize=%.1f)\n",
           g_n_ring_bins, g_n_eta_bins, g_n_ome_bins, gEtaBinSize, gOmeBinSize);
  }
  if (BigDetSize != 0) {
    long long int size2 = ReadBigDet(cwd);
    totNrPixelsBigDetector = BigDetSize;
    totNrPixelsBigDetector *= BigDetSize;
    totNrPixelsBigDetector /= 32;
    totNrPixelsBigDetector++;
  }

  //////////////////////////// OPENMP
  size_t *infoArr;
  int nSptIDs;
  int nBlocks = atoi(argv[3]);
  int blockNr = atoi(argv[2]);
  int nSpotsToIndex = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  int startRowNr;
  int endRowNr;
  startRowNr = (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * blockNr;
  int tmp =
      (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * (blockNr + 1);
  endRowNr = tmp < (nSpotsToIndex - 1) ? tmp : (nSpotsToIndex - 1);
  printf("%d %d %d\n", startRowNr, tmp, endRowNr);
  nSptIDs = endRowNr - startRowNr + 1;
  infoArr = malloc(nSptIDs * 5 * sizeof(*infoArr));
  // Read spotIDs
  int it;
  FILE *spotsFile = fopen("SpotsToIndex.csv", "r");
  if (spotsFile == NULL) {
    printf("Could not open SpotsToIndex.csv. Exiting.\n");
    return 1;
  }
  for (it = 0; it < startRowNr - 1; it++)
    fgets(aline, 1000, spotsFile); // skip first startRowNr-1 lines
  for (it = 0; it < nSptIDs; it++) {
    fgets(aline, 1000, spotsFile);
    sscanf(aline, "%zu %zu %zu %zu %zu", &infoArr[it * 5 + 0],
           &infoArr[it * 5 + 1], &infoArr[it * 5 + 2], &infoArr[it * 5 + 3],
           &infoArr[it * 5 + 4]);
  }
  fclose(spotsFile);
  double **hkls;
  hkls = allocMatrix(MaxNHKLS, 7);
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  if (hklf == NULL) {
    printf("Could not read the hkl file. Exiting.\n");
    return 0;
  }
  fgets(aline, 1000, hklf);
  int h, kt, l, Rnr, nhkls = 0;
  double ds, tht;
  double MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %lf %d %s %s %s %lf %s %s", &h, &kt, &l, &ds, &Rnr,
           dummy, dummy, dummy, &tht, dummy, dummy);
    if (tht > MaxTtheta / 2)
      break;
    for (it = 0; it < cs; it++) {
      if (Rnr == RingNumbers[it]) {
        hkls[nhkls][0] = h;
        hkls[nhkls][1] = kt;
        hkls[nhkls][2] = l;
        hkls[nhkls][3] = ds;
        hkls[nhkls][4] = tht;
        hkls[nhkls][5] = RingRadii[it];
        hkls[nhkls][6] = RingNumbers[it];
        nhkls++;
      }
    }
  }
  fclose(hklf);
  if (nOmeRanges != nBoxSizes) {
    printf("Number of omega ranges and number of box sizes don't match. "
           "Exiting!\n");
    return 0;
  }
  double MargOme2 = 2, chi = 0;
  int thisRowNr;
#pragma omp parallel for num_threads(numProcs) private(thisRowNr)              \
    schedule(dynamic)
  for (thisRowNr = 0; thisRowNr < nSptIDs; thisRowNr++) {
    char *h1 = "SpotID\tYObsCorrPos\tZObsCorrPos\tOmegaObsCorrPos\tG1Obs\tG2Obs"
               "\tG3Obs\tYExp\tZExp\tOmegaExp\tG1Exp\tG2Exp\tG3Exp\t";
    char *h2 = "YObsCorrWedge\tZObsCorrWedge\tOmegaObsCorrWedge\tOmegaObs\tYObs"
               "\tZObs\tInternalAngle\tDiffLen\tDiffOmega\tRingNr\n";
    char header[2048];
    sprintf(header, "%s%s", h1, h2);
    int i, j;
    double LatCin[6];
    for (i = 0; i < 6; i++)
      LatCin[i] = LatCinT[i];
    int SpId = infoArr[thisRowNr * 5 + 1];
    int voxNr = (int)infoArr[thisRowNr * 5 + 0];
    int nSpotsBest = (int)infoArr[thisRowNr * 5 + 2];
    int *spotIDS = malloc(nSpotsBest * sizeof(*spotIDS));
    size_t locOffsetVals = infoArr[thisRowNr * 5 + 3],
           locOffsetIDs = infoArr[thisRowNr * 5 + 4];
    char fnVals[2048], fnIDs[2048];
    sprintf(fnVals, "Output/IndexBest_voxNr_%0*d.bin", 6, voxNr);
    sprintf(fnIDs, "Output/IndexBest_IDs_voxNr_%0*d.bin", 6, voxNr);
    FILE *fVals = fopen(fnVals, "rb");
    FILE *fIDs = fopen(fnIDs, "rb");
    if (!fVals || !fIDs) {
      printf("Warning: could not open %s or %s, skipping voxel %d\n", fnVals,
             fnIDs, voxNr);
      if (fVals)
        fclose(fVals);
      if (fIDs)
        fclose(fIDs);
      free(spotIDS);
      continue;
    }
    fseek(fVals, locOffsetVals, SEEK_SET);
    double tmpArr[16];
    if (fread(tmpArr, 16 * sizeof(double), 1, fVals) != 1) {
      fprintf(stderr, "Warning: incomplete read from %s\n", fnVals);
    }
    fclose(fVals);
    fseek(fIDs, locOffsetIDs, SEEK_SET);
    if (fread(spotIDS, nSpotsBest * sizeof(int), 1, fIDs) != 1) {
      fprintf(stderr, "Warning: incomplete read from %s\n", fnIDs);
    }
    fclose(fIDs);

    // Extract initial values from binary record
    double Orient0[9], Pos0[3], Euler0[3], Orient0_3[3][3];
    double NrExpected = tmpArr[14], NrObserved = tmpArr[15];
    double completeness = NrObserved / NrExpected;
    double meanRadius = 1;
    for (i = 0; i < 9; i++)
      Orient0[i] = tmpArr[i + 2];
    for (i = 0; i < 3; i++)
      Pos0[i] = tmpArr[i + 11];

    double a = LatCin[0], b = LatCin[1], c = LatCin[2], alph = LatCin[3],
           bet = LatCin[4], gamm = LatCin[5];
    for (i = 0; i < 3; i++)
      for (j = 0; j < 3; j++)
        Orient0_3[i][j] = Orient0[i * 3 + j];
    OrientMat2Euler(Orient0_3, Euler0);
    Euler2OrientMat(Euler0, Orient0_3);
    Convert3x3To9(Orient0_3, Orient0);

    // Populate observed spots from mmap'd AllSpots
    double **spotsYZO = allocMatrix(nSpotsBest, 11);
    size_t spotPosAllSpots;
    for (i = 0; i < nSpotsBest; i++) {
      spotPosAllSpots = (int)spotIDS[i] - 1;
      spotsYZO[i][0] = AllSpots[spotPosAllSpots * 16 + 0];
      spotsYZO[i][1] = AllSpots[spotPosAllSpots * 16 + 1];
      spotsYZO[i][2] = AllSpots[spotPosAllSpots * 16 + 2];
      spotsYZO[i][3] = AllSpots[spotPosAllSpots * 16 + 4];
      spotsYZO[i][4] = AllSpots[spotPosAllSpots * 16 + 8];
      spotsYZO[i][5] = AllSpots[spotPosAllSpots * 16 + 9];
      spotsYZO[i][6] = AllSpots[spotPosAllSpots * 16 + 10];
      spotsYZO[i][7] = AllSpots[spotPosAllSpots * 16 + 5];
      spotsYZO[i][8] = AllSpots[spotPosAllSpots * 16 + 14];
      spotsYZO[i][9] = AllSpots[spotPosAllSpots * 16 + 15];
    }

    // Initial CalcAngleErrors: match observed to theoretical spots
    double Ini[12];
    ConcatPosEulLatc(Ini, Pos0, Euler0, LatCin);
    double **SpotsComp = allocMatrix(MaxNSpotsBest, 23);
    double **Splist = allocMatrix(MaxNSpotsBest, 11);
    double Error[3];
    int nSpotsComp;
    CalcAngleErrors(nSpotsBest, nhkls, nOmeRanges, Ini, spotsYZO, hkls, Lsd,
                    Wavelength, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                    SpotsComp, Splist, Error, &nSpotsComp, 0);

    double **spotsYZONew = allocMatrix(nSpotsComp, 11);
    for (i = 0; i < nSpotsComp; i++)
      for (j = 0; j < 11; j++)
        spotsYZONew[i][j] = Splist[i][j];

    // Re-evaluate matched spots with notIniRun=1
    CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, Ini, spotsYZONew, hkls, Lsd,
                    Wavelength, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                    SpotsComp, Splist, Error, &nSpotsComp, 1);
    for (i = 0; i < nSpotsComp; i++)
      for (j = 0; j < 11; j++)
        spotsYZONew[i][j] = Splist[i][j];

    // Step 1: Fit orientation (Euler angles + lattice parameters)
    double X0_2[9];
    for (i = 0; i < 3; i++)
      X0_2[i] = Euler0[i];
    for (i = 0; i < 6; i++)
      X0_2[i + 3] = LatCin[i];
    double EulerLow[3], EulerHigh[3];
    for (i = 0; i < 3; i++) {
      EulerLow[i] = Euler0[i] - MargOme2;
      EulerHigh[i] = Euler0[i] + MargOme2;
    }
    double lb2[9], ub2[9];
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
    double *XFit2 = malloc(9 * sizeof(*XFit2));
    FitOrientIni(X0_2, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                 nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit2,
                 lb2, ub2, Pos0);

    // Re-evaluate after orientation fit
    double UseXFit[12];
    for (i = 0; i < 3; i++)
      UseXFit[i] = Pos0[i];
    for (i = 0; i < 3; i++)
      UseXFit[i + 3] = XFit2[i];
    for (i = 0; i < 6; i++)
      UseXFit[i + 6] = LatCin[i];
    CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFit, spotsYZONew, hkls,
                    Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                    SpotsComp, Splist, Error, &nSpotsComp, 1);
    for (i = 0; i < nSpotsComp; i++)
      for (j = 0; j < 11; j++)
        spotsYZONew[i][j] = Splist[i][j];
    // === Dynamic Reassignment after Orient Fit (Pass 2) ===
    {
      int nReassigned = ReassignSpotsFromBins(
          UseXFit, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges,
          BoxSizes, MinEta, wedge, chi, spotsYZONew, MaxNSpotsBest, AllSpots,
          nSpots);
      if (nReassigned > 0) {
        nSpotsComp = nReassigned;
        FitOrientIni(X0_2, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd,
                     Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta,
                     wedge, chi, XFit2, lb2, ub2, Pos0);
        for (i = 0; i < 3; i++)
          UseXFit[i] = Pos0[i];
        for (i = 0; i < 3; i++)
          UseXFit[i + 3] = XFit2[i];
        for (i = 0; i < 6; i++)
          UseXFit[i + 6] = LatCin[i];
        CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFit, spotsYZONew,
                        hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta,
                        wedge, chi, SpotsComp, Splist, Error, &nSpotsComp, 1);
        for (i = 0; i < nSpotsComp; i++)
          for (j = 0; j < 11; j++)
            spotsYZONew[i][j] = Splist[i][j];
      }
    }

    // Step 2: Fit strains (lattice parameters)
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
    double *XFit3 = malloc(6 * sizeof(*XFit3));
    FitStrainIni(X0_3, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd, Wavelength,
                 nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, XFit3,
                 lb3, ub3, Pos0, OrientFitIn);

    // Build final result: [Pos0, fitted Orient, fitted LatC]
    double FinalResult[12];
    for (i = 0; i < 3; i++)
      FinalResult[i] = Pos0[i];
    for (i = 0; i < 3; i++)
      FinalResult[i + 3] = XFit2[i];
    for (i = 0; i < 6; i++)
      FinalResult[i + 6] = XFit3[i];
    double ErrorFin[3];
    CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, FinalResult, spotsYZONew,
                    hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta, wedge,
                    chi, SpotsComp, Splist, ErrorFin, &nSpotsComp, 1);
    printf("VoxNr %6d, SpotID %6d, %6d out of %6d, Errors: %7.2f %6.4f %6.4f, ",
           voxNr, SpId, thisRowNr, nSptIDs, ErrorFin[0], ErrorFin[1],
           ErrorFin[2]);
    for (i = 0; i < nSpotsComp; i++)
      for (j = 0; j < 11; j++)
        spotsYZONew[i][j] = Splist[i][j];
    // === Dynamic Reassignment after Strain Fit (Pass 2) ===
    {
      double UseXFitS[12];
      for (i = 0; i < 3; i++)
        UseXFitS[i] = Pos0[i];
      for (i = 0; i < 3; i++)
        UseXFitS[i + 3] = XFit2[i];
      for (i = 0; i < 6; i++)
        UseXFitS[i + 6] = XFit3[i];
      int nReassigned = ReassignSpotsFromBins(
          UseXFitS, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges,
          BoxSizes, MinEta, wedge, chi, spotsYZONew, MaxNSpotsBest, AllSpots,
          nSpots);
      if (nReassigned > 0) {
        nSpotsComp = nReassigned;
        FitStrainIni(X0_3, nSpotsComp, spotsYZONew, nhkls, hkls, Lsd,
                     Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta,
                     wedge, chi, XFit3, lb3, ub3, Pos0, OrientFitIn);
        for (i = 0; i < 3; i++)
          UseXFitS[i] = Pos0[i];
        for (i = 0; i < 3; i++)
          UseXFitS[i + 3] = XFit2[i];
        for (i = 0; i < 6; i++)
          UseXFitS[i + 6] = XFit3[i];
        CalcAngleErrors(nSpotsComp, nhkls, nOmeRanges, UseXFitS, spotsYZONew,
                        hkls, Lsd, Wavelength, OmegaRanges, BoxSizes, MinEta,
                        wedge, chi, SpotsComp, Splist, ErrorFin, &nSpotsComp,
                        1);
        for (i = 0; i < nSpotsComp; i++)
          for (j = 0; j < 11; j++)
            spotsYZONew[i][j] = Splist[i][j];
      }
    }
    printf("Fitvals: Pos: %7.2f %7.2f %7.2f, Orient: %7.2f %7.2f %7.2f, LatC: "
           "%6.4f %6.4f %6.4f %7.3f %7.3f %7.3f\n",
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
    double OrientsFit[10], PositionsFit[4], StrainsFit[7], ErrorsFin[4];
    OrientsFit[0] = SpId;
    PositionsFit[0] = SpId;
    ErrorsFin[0] = SpId;
    StrainsFit[0] = SpId;
    for (i = 0; i < 9; i++) {
      OrientsFit[i + 1] = OrientFit[i];
    }
    for (i = 0; i < 3; i++)
      PositionsFit[i + 1] = PositionFit[i];
    for (i = 0; i < 6; i++)
      StrainsFit[i + 1] = LatticeParameterFit[i];
    for (i = 0; i < 3; i++)
      ErrorsFin[i + 1] = ErrorFin[i];

    // Calculate Strains here.
    double StrainTensorSample[3][3], **SpotsOut;
    SpotsOut = allocMatrix(nSpotsComp, 24);
    double RetVal;
    StrainTensorKenesei(nSpotsComp, SpotsComp, Lsd, Wavelength, nhkls, hkls,
                        StrainTensorSample, SpotsOut, &RetVal);

    // Start Writing FitBest+FNs[thisRowNr]
    // What to write: Orientation, Position, LatticeParameter, Errors
    char outFN[2048];
    sprintf(outFN, "%s/FitBest_%0*d_%0*d.csv", ResultFolder, 6, voxNr, 9, SpId);
    double OutMatr[27];
    for (i = 0; i < 10; i++) {
      OutMatr[i] = OrientsFit[i];
    }
    for (i = 0; i < 4; i++) {
      OutMatr[i + 10] = PositionsFit[i];
    }
    for (i = 0; i < 7; i++) {
      OutMatr[i + 14] = StrainsFit[i];
    }
    for (i = 0; i < 4; i++) {
      OutMatr[i + 21] = ErrorsFin[i];
    }
    OutMatr[25] = meanRadius;
    OutMatr[26] = completeness;
#pragma omp critical
    {
      FILE *outF = fopen(outFN, "w");
      if (outF == NULL) {
        printf("Could not open output file for writing: %s\n", outFN);
      } else {
        fprintf(outF,
                "SpotID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tSpotID\tx"
                "\ty\tz\tSpotID\ta\tb\tc\talpha\tbeta\tgamma\tSpotID\tPosErr\tO"
                "meErr\tInternalAngle\tRadius\tCompleteness\tE11\tE12\tE13\tE21"
                "\tE22\tE23\tE31\tE32\tE33\tEul1\tEul2\tEul3\n");
        for (i = 0; i < 27; i++)
          fprintf(outF, "%lf\t", OutMatr[i]);
        for (i = 0; i < 3; i++)
          for (j = 0; j < 3; j++)
            fprintf(outF, "%lf\t", StrainTensorSample[i][j] * 1000000);
        for (i = 0; i < 3; i++)
          fprintf(outF, "%lf\t", EulerFit[i]);
        fprintf(outF, "\n");
        fprintf(outF, "%s", header);
        for (i = 0; i < nSpotsComp; i++) {
          for (j = 0; j < 23; j++) {
            fprintf(outF, "%lf\t", SpotsOut[i][j]);
          }
          fprintf(outF, "\n");
        }
      }
      fclose(outF);
    }
    free(spotIDS);
    FreeMemMatrix(spotsYZO, nSpotsBest);
    FreeMemMatrix(SpotsOut, nSpotsComp);
    FreeMemMatrix(SpotsComp, MaxNSpotsBest);
    FreeMemMatrix(Splist, MaxNSpotsBest);
    FreeMemMatrix(spotsYZONew, nSpotsComp);
    free(XFit2);
    free(XFit3);
  }

  FreeMemMatrix(hkls, MaxNHKLS);
  free(infoArr);
  munmap(AllSpots, size);
  close(fd);
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return 0;
}
