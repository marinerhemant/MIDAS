//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  ForwardSimulationCompressed.c
//
//
//  Created by Hemant Sharma on 2024/05/15.
//  Full Multi-Scan forward simulation for FF.
//  2025/09/05:
// 		1. Added relative intensity of peaks from a cif type of file.
//		2. Added the energy resolution imlementation/
//		3. Bilinear interpolation for accurate intensity distribution.
//		4. Openmp!!!
//
//

#include "midas_paths.h"
#include <blosc.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zip.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define RealType double
#define MAX_N_HKLS 5000
#define EPS 0.00001
#define MAX_NR_POINTS 20000000
#define MAX_OUTPUT_INTENSITY 15000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int n_hkls = 0;
double hkls[MAX_N_HKLS][4];
int mHKLs[MAX_N_HKLS];

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

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void MatrixMultF(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }

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

static inline void RotateAroundZ(double v1[3], double alpha, double v2[3]) {
  double cosa = cos(alpha * deg2rad);
  double sina = sin(alpha * deg2rad);

  double mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};

  MatrixMultF(mat, v1, v2);
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

static inline void CalcEtaAngle(double y, double z, double *alpha) {
  *alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    *alpha = -*alpha;
}

static inline void CalcSpotPosition(double RingRadius, double eta, double *yl,
                                    double *zl) {
  double etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

static inline void CalcOmega(double x, double y, double z, double theta,
                             double omegas[4], double etas[4], int *nsol) {
  *nsol = 0;
  double ome;
  double len = sqrt(x * x + y * y + z * z);
  double v = sin(theta * deg2rad) * len;

  double almostzero = 1e-4;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      double cosome1 = -v / x;
      if (fabs(cosome1 <= 1)) {
        ome = acos(cosome1) * rad2deg;
        omegas[*nsol] = ome;
        *nsol = *nsol + 1;
        omegas[*nsol] = -ome;
        *nsol = *nsol + 1;
      }
    }
  } else {
    double y2 = y * y;
    double a = 1 + ((x * x) / y2);
    double b = (2 * v * x) / y2;
    double c = ((v * v) / y2) - 1;
    double discr = b * b - 4 * a * c;

    double ome1a;
    double ome1b;
    double ome2a;
    double ome2b;
    double cosome1;
    double cosome2;

    double eqa, eqb, diffa, diffb;

    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1) {
        ome1a = acos(cosome1);
        ome1b = -ome1a;
        eqa = -x * cos(ome1a) + y * sin(ome1a);
        diffa = fabs(eqa - v);
        eqb = -x * cos(ome1b) + y * sin(ome1b);
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome1a * rad2deg;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome1b * rad2deg;
          *nsol = *nsol + 1;
        }
      }

      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2);
        ome2b = -ome2a;

        eqa = -x * cos(ome2a) + y * sin(ome2a);
        diffa = fabs(eqa - v);
        eqb = -x * cos(ome2b) + y * sin(ome2b);
        diffb = fabs(eqb - v);

        if (diffa < diffb) {
          omegas[*nsol] = ome2a * rad2deg;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome2b * rad2deg;
          *nsol = *nsol + 1;
        }
      }
    }
  }
  double gw[3];
  double gv[3] = {x, y, z};
  double eta;
  int indexOme;
  for (indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

static inline void FreeMemMatrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

void RotationTilts(double tx, double ty, double tz, double RotMatOut[3][3]) {
  tx = deg2rad * tx;
  ty = deg2rad * ty;
  tz = deg2rad * tz;
  double r1[3][3];
  double r2[3][3];
  double r3[3][3];
  double r1r2[3][3];
  r1[0][0] = cos(tz);
  r1[0][1] = -sin(tz);
  r1[0][2] = 0;
  r1[1][0] = sin(tz);
  r1[1][1] = cos(tz);
  r1[1][2] = 0;
  r1[2][0] = 0;
  r1[2][1] = 0;
  r1[2][2] = 1;
  r2[0][0] = cos(ty);
  r2[0][1] = 0;
  r2[0][2] = sin(ty);
  r2[1][0] = 0;
  r2[1][1] = 1;
  r2[1][2] = 0;
  r2[2][0] = -sin(ty);
  r2[2][1] = 0;
  r2[2][2] = cos(ty);
  r3[0][0] = 1;
  r3[0][1] = 0;
  r3[0][2] = 0;
  r3[1][0] = 0;
  r3[1][1] = cos(tx);
  r3[1][2] = -sin(tx);
  r3[2][0] = 0;
  r3[2][1] = sin(tx);
  r3[2][2] = cos(tx);
  MatrixMultF33(r1, r2, r1r2);
  MatrixMultF33(r1r2, r3, RotMatOut);
}

static inline void DisplacementInTheSpot(double a, double b, double c,
                                         double xi, double yi, double zi,
                                         double omega, double *Displ_y,
                                         double *Displ_z) {
  double sinOme = sind(omega), cosOme = cosd(omega), AcosOme = a * cosOme,
         BsinOme = b * sinOme;
  double XNoW = AcosOme - BsinOme, YNoW = (a * sinOme) + (b * cosOme), ZNoW = c;
  double XW = XNoW, YW = YNoW;
  double ZW = ZNoW, XC = XW;
  double YC = YW, ZC = ZW;
  double IK[3], NormIK;
  IK[0] = xi - XC;
  IK[1] = yi - YC;
  IK[2] = zi - ZC;
  NormIK = sqrt((IK[0] * IK[0]) + (IK[1] * IK[1]) + (IK[2] * IK[2]));
  IK[0] = IK[0] / NormIK;
  IK[1] = IK[1] / NormIK;
  IK[2] = IK[2] / NormIK;
  if (fabs(IK[0]) > EPS) {
    *Displ_y = YC - ((XC * IK[1]) / (IK[0]));
    *Displ_z = ZC - ((XC * IK[2]) / (IK[0]));
  }
}

static inline void CalcDiffrSpots_Furnace(double **hklIns,
                                          double OrientMatrix[3][3],
                                          double distance, double wavelength,
                                          double **spots, int *nspots) {
  int i, OmegaRangeNo;
  double theta;
  int KeepSpot;
  double Ghkl[3];
  int indexhkl;
  double Gc[3];
  double omegas[4];
  double etas[4];
  double yl;
  double zl;
  double Gw[3], OmeMat[3][3];
  int nspotsPlane;
  int spotnr = 0;
  int spotid = 0;
  double RingRadius, ds, GLen, NormG;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hklIns[indexhkl][0];
    Ghkl[1] = hklIns[indexhkl][1];
    Ghkl[2] = hklIns[indexhkl][2];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = hklIns[indexhkl][3];
    RingRadius = distance * tan(2 * deg2rad * theta);
    ds = 2 * sind(theta) / wavelength;
    GLen = 2 * M_PI / ds;
    CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
    for (i = 0; i < nspotsPlane; i++) {
      if (isnan(etas[i]))
        continue;
      if (isnan(omegas[i]))
        continue;
      spots[spotnr][0] = RingRadius;
      spots[spotnr][1] = etas[i];
      spots[spotnr][2] = omegas[i];
      spots[spotnr][3] = theta;
      spots[spotnr][4] = hklIns[indexhkl][4]; // RingNr
      spotnr++;
      spotid++;
    }
  }
  *nspots = spotnr;
}

void Euler2OrientMat(double Euler[3], double m_out[9]) {
  double psi, phi, theta, cps, cph, cth, sps, sph, sth;
  psi = Euler[0];
  phi = Euler[1];
  theta = Euler[2];
  cps = cos(psi);
  cph = cos(phi);
  cth = cos(theta);
  sps = sin(psi);
  sph = sin(phi);
  sth = sin(theta);
  m_out[0] = cth * cps - sth * cph * sps;
  m_out[1] = -cth * cph * sps - sth * cps;
  m_out[2] = sph * sps;
  m_out[3] = cth * sps + sth * cph * cps;
  m_out[4] = cth * cph * cps - sth * sps;
  m_out[5] = -sph * cps;
  m_out[6] = sth * sph;
  m_out[7] = cth * sph;
  m_out[8] = cph;
}

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
  Euler[0] = psi;
  Euler[1] = phi;
  Euler[2] = theta;
}

static inline void CorrectHKLsLatC(double LatC[6], double Wavelength,
                                   double **hklsOut) {
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
  for (hklnr = 0; hklnr < n_hkls; hklnr++) {
    double ginit[3];
    ginit[0] = hkls[hklnr][0];
    ginit[1] = hkls[hklnr][1];
    ginit[2] = hkls[hklnr][2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hklsOut[hklnr][0] = GCart[0];
    hklsOut[hklnr][1] = GCart[1];
    hklsOut[hklnr][2] = GCart[2];
    hklsOut[hklnr][3] = (asind((Wavelength) / (2 * Ds))); // Theta
    hklsOut[hklnr][4] = hkls[hklnr][3];
  }
}

inline void MatInv(double A[3][3], double AInv[3][3]) {
  double a = A[0][0];
  double b = A[0][1];
  double c = A[0][2];
  double d = A[1][0];
  double e = A[1][1];
  double f = A[1][2];
  double g = A[2][0];
  double h = A[2][1];
  double i = A[2][2];
  double DetA =
      (a * (e * i - f * h)) - (b * (i * d - f * g)) + (c * (d * h - e * g));
  AInv[0][0] = (e * i - f * h) / DetA;
  AInv[0][1] = -(b * i - c * h) / DetA;
  AInv[0][2] = (b * f - c * e) / DetA;
  AInv[1][0] = -(d * i - f * g) / DetA;
  AInv[1][1] = (a * i - c * g) / DetA;
  AInv[1][2] = -(a * f - c * d) / DetA;
  AInv[2][0] = (d * h - e * g) / DetA;
  AInv[2][1] = -(a * h - b * g) / DetA;
  AInv[2][2] = (a * e - b * d) / DetA;
}

static inline void CorrectHKLsLatCEpsilon(double LatC[6], double eps[6],
                                          double Wavelength, double **hklsOut) {
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
  //~ printf("%lf\n",Vol);
  double B0[3][3];
  B0[0][0] = APr;
  B0[0][1] = (BPr * cosd(GammaPr)), B0[0][2] = (CPr * cosd(BetaPr)),
  B0[1][0] = 0, B0[1][1] = (BPr * sind(GammaPr)),
  B0[1][2] = (-CPr * SinBetaPr * CosA), B0[2][0] = 0, B0[2][1] = 0,
  B0[2][2] = (CPr * SinBetaPr * SinA);
  double B[3][3], Binv[3][3];
  Binv[0][0] = (eps[0] + 1) / B0[0][0];
  Binv[1][1] = (eps[3] + 1) / B0[1][1];
  Binv[2][2] = (eps[5] + 1) / B0[2][2];
  Binv[0][1] = (2 * eps[1] - B0[0][1] * Binv[1][1]) / B0[0][0];
  Binv[1][2] = (2 * eps[4] - B0[1][2] * Binv[2][2]) / B0[1][1];
  Binv[0][2] =
      (2 * eps[2] - B0[0][1] * Binv[1][2] - B0[0][2] * Binv[2][2]) / B0[0][0];
  Binv[1][0] = Binv[0][1];
  Binv[2][1] = Binv[1][2];
  Binv[2][0] = Binv[0][2];
  int i, j;
  MatInv(Binv, B);
  //~ for (i=0;i<3;i++) {for (j=0;j<3;j++) {printf("%lf %lf %lf
  //",B0[i][j],Binv[i][j],B[i][j]); }}printf("\n");
  for (hklnr = 0; hklnr < n_hkls; hklnr++) {
    double ginit[3];
    ginit[0] = hkls[hklnr][0];
    ginit[1] = hkls[hklnr][1];
    ginit[2] = hkls[hklnr][2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hklsOut[hklnr][0] = GCart[0];
    hklsOut[hklnr][1] = GCart[1];
    hklsOut[hklnr][2] = GCart[2];
    hklsOut[hklnr][3] = (asind((Wavelength) / (2 * Ds))); // Theta
    hklsOut[hklnr][4] = hkls[hklnr][3];
    //~ printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf
    //%lf\n",Wavelength,Ds,hkls[hklnr][0],hkls[hklnr][1],hkls[hklnr][2],hkls[hklnr][3],hklsOut[hklnr][0],hklsOut[hklnr][1],hklsOut[hklnr][2],hklsOut[hklnr][3],hklsOut[hklnr][4]);
  }
}

void DisplacementSpots(RealType a, RealType b, RealType Lsd, RealType yi,
                       RealType zi, RealType omega, RealType *Displ_y,
                       RealType *Displ_z) {
  RealType OmegaRad = deg2rad * omega;
  RealType sinOme = sin(OmegaRad);
  RealType cosOme = cos(OmegaRad);
  RealType xa = a * cosOme - b * sinOme;
  RealType ya = a * sinOme + b * cosOme;
  RealType t = 1 - (xa / Lsd);
  *Displ_y = ya + (yi * t);
  *Displ_z = t * zi;
}

static inline double CorrectWedge(double eta, double theta, double wl,
                                  double wedge) {
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
  double G1a = (k1f);
  double G2a = (k2f);
  double G3a = k3f;
  double LenGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / LenGa;
  double g2 = G2a * ds / LenGa;
  double g3 = G3a * ds / LenGa;
  SinW = 0;
  CosW = 1;
  double LenG = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(LenG * LenG * wl) / 2;
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
  if (fabs(fabs(Omega1) - 180.0) < EPS)
    Omega1 = 0;
  if (fabs(fabs(Omega2) - 180.0) < EPS)
    Omega2 = 0;
  double OmeDiff1 = fabs(Omega1);
  double OmeDiff2 = fabs(Omega2);
  double Omega;
  if (OmeDiff1 < OmeDiff2)
    Omega = Omega1;
  else
    Omega = Omega2;
  return Omega;
}

static inline void
CorrectTiltSpatialDistortion(double px, double Lsd, double ybc, double zbc,
                             double tx, double ty, double tz, double RhoD,
                             double p0, double p1, double p2, int NrPixels,
                             double *yDispl, double *zDispl) {
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
  double Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, EtaT, Diff, MeanDiff;
  double ABC[3], ABCPr[3], XYZ[3], YCorr, ZCorr, yDiff, zDiff;
  int yTrans, zTrans;
  long long int idx;
  for (i = 0; i < NrPixels; i++) {
    for (j = 0; j < NrPixels; j++) {
      Yc = -(i - ybc) * px;
      Zc = (j - zbc) * px;
      ABC[0] = 0;
      ABC[1] = Yc;
      ABC[2] = Zc;
      MatrixMult(TRs, ABC, ABCPr);
      XYZ[0] = Lsd + ABCPr[0];
      XYZ[1] = ABCPr[1];
      XYZ[2] = ABCPr[2];
      Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
      CalcEtaAngle(XYZ[1], XYZ[2], &Eta);
      RNorm = Rad / RhoD;
      EtaT = 90 - Eta;
      DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                    (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT)))) +
                    (p2 * (pow(RNorm, n2))) + 1;
      Rcorr = Rad * DistortFunc;
      YCorr = -Rcorr * sin(Eta * deg2rad);
      ZCorr = Rcorr * cos(Eta * deg2rad);
      yDiff = Yc - YCorr;
      zDiff = Zc - ZCorr;
      yTrans = (int)(-YCorr / px + ybc);
      zTrans = (int)(ZCorr / px + zbc);
      if (yTrans < 0 || yTrans >= NrPixels || zTrans < 0 || zTrans >= NrPixels)
        continue;
      idx = yTrans + NrPixels * zTrans;
      if (idx < 0)
        continue;
      //~ printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %lld %d %d\n",
      //~ Rad,Rcorr,Eta,Yc,Zc,YCorr,ZCorr,yDiff,zDiff,yTrans,zTrans,idx,i,j);
      //~ fflush(stdout);
      yDispl[idx] = yDiff;
      zDispl[idx] = zDiff;
    }
  }
}

static inline int writeStrZip(char outstr[8192], char zarrfn[8192],
                              zip_t *zipper) {
  zip_source_t *sourc0 =
      zip_source_buffer(zipper, (const void *)outstr, strlen(outstr), 0);
  zip_int64_t rcv0 =
      zip_file_add(zipper, zarrfn, sourc0, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
  if (rcv0 < 0) {
    printf("Could not add the file %s to zip. Exiting.\n", zarrfn);
    return 1;
  }
  int rc = zip_set_file_compression(zipper, rcv0, ZIP_CM_STORE, 0);
  if (rc != 0) {
    printf(
        "Could not change compression type of the file %s to zip. Exiting.\n",
        zarrfn);
    return 1;
  }
  return 0;
}

static inline void usage(void) {
  printf("Make diffraction spots: usage: ./ForwardSimulationCompressed "
         "<ParameterFile> nCPUs\n"
         "If you want to do a multi-scan simulation, there should be "
         "positions.csv file in the current folder.\n"
         "Positions in micron.\n");
}

static long ReadInputData(char *InFileName, int isBin, double **InputInfo,
                          double *LatC, double minConfidence,
                          int UpdatedOrientations, int LoadNr,
                          int *dataType_out, double *maxVol_out) {
  long nrPoints = 0;
  int dataType = 0;
  double maxVol = 0;
  FILE *inpF;
  char aline[4096];
  char dummy[4096];
  char strLine[4096];
  int i, j, nrSkip;
  double EulerThis[3], OrientThis[9], OrientTemp[3][3];

  if (isBin) {
    dataType = 3;
    inpF = fopen(InFileName, "rb");
    if (inpF == NULL)
      return 0;
    size_t sz;
    fseek(inpF, 0L, SEEK_END);
    sz = ftell(inpF);
    rewind(inpF);
    nrPoints = sz / (18 * sizeof(double));
    printf("Number of elements: %ld, Size: %lu\n", nrPoints, (unsigned long)sz);
    double *holdArr;
    holdArr = calloc(nrPoints * 18, sizeof(double));
    if (holdArr == NULL) {
      fclose(inpF);
      return 0;
    }
    fread(holdArr, sz, 1, inpF);
    fclose(inpF);
    for (i = 0; i < nrPoints; i++) {
      OrientTemp[0][0] = holdArr[i * 18 + 3];
      OrientTemp[0][1] = holdArr[i * 18 + 4];
      OrientTemp[0][2] = holdArr[i * 18 + 5];
      OrientTemp[1][0] = holdArr[i * 18 + 6];
      OrientTemp[1][1] = holdArr[i * 18 + 7];
      OrientTemp[1][2] = holdArr[i * 18 + 8];
      OrientTemp[2][0] = holdArr[i * 18 + 9];
      OrientTemp[2][1] = holdArr[i * 18 + 10];
      OrientTemp[2][2] = holdArr[i * 18 + 11];
      OrientMat2Euler(OrientTemp, EulerThis);
      Euler2OrientMat(EulerThis, OrientThis);
      InputInfo[i][0] = OrientThis[0];
      InputInfo[i][1] = OrientThis[1];
      InputInfo[i][2] = OrientThis[2];
      InputInfo[i][3] = OrientThis[3];
      InputInfo[i][4] = OrientThis[4];
      InputInfo[i][5] = OrientThis[5];
      InputInfo[i][6] = OrientThis[6];
      InputInfo[i][7] = OrientThis[7];
      InputInfo[i][8] = OrientThis[8];
      InputInfo[i][9] = holdArr[i * 18 + 0];
      InputInfo[i][10] = holdArr[i * 18 + 1];
      InputInfo[i][11] = holdArr[i * 18 + 2];
      InputInfo[i][12] = holdArr[i * 18 + 12];
      InputInfo[i][13] = holdArr[i * 18 + 13];
      InputInfo[i][14] = holdArr[i * 18 + 14];
      InputInfo[i][15] = holdArr[i * 18 + 15];
      InputInfo[i][16] = holdArr[i * 18 + 16];
      InputInfo[i][17] = holdArr[i * 18 + 17];
    }
    free(holdArr);
  } else {
    inpF = fopen(InFileName, "r");
    if (inpF == NULL)
      return 0;
    fgets(aline, 4096, inpF);
    if (strncmp(aline, "%NumGrains ", strlen("%NumGrains ")) == 0) { // Grains
      dataType = 0;
      printf(
          "We will write a separate Grains.csv file matching the GrainIDs.\n");
      long NrOrientations;
      sscanf(aline, "%s %ld", dummy, &NrOrientations);
      char *GrainOutFN = "GrainsGen.csv";
      FILE *foutGrains;
      foutGrains = fopen(GrainOutFN, "w");
      fprintf(foutGrains, "%s", aline);
      NrOrientations += 2; // Increment matching original logic (lines
                           // 1064-1065) or just skip?
      // "NrOrientations++; NrOrientations++;" was in original code.
      // But actually we just write the next few lines.

      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);
      fgets(aline, 4096, inpF);
      fprintf(foutGrains, "%s", aline);

      while (fgets(aline, 4096, inpF) != NULL) {
        sscanf(aline,
               "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
               "%lf %lf %lf",
               dummy, &InputInfo[nrPoints][0], &InputInfo[nrPoints][1],
               &InputInfo[nrPoints][2], &InputInfo[nrPoints][3],
               &InputInfo[nrPoints][4], &InputInfo[nrPoints][5],
               &InputInfo[nrPoints][6], &InputInfo[nrPoints][7],
               &InputInfo[nrPoints][8], &InputInfo[nrPoints][9],
               &InputInfo[nrPoints][10], &InputInfo[nrPoints][11],
               &InputInfo[nrPoints][12], &InputInfo[nrPoints][13],
               &InputInfo[nrPoints][14], &InputInfo[nrPoints][15],
               &InputInfo[nrPoints][16], &InputInfo[nrPoints][17]);
        fprintf(foutGrains, "%ld\t", nrPoints + 1);
        for (i = 0; i < 18; i++)
          fprintf(foutGrains, "%lf\t", InputInfo[nrPoints][i]);
        fprintf(foutGrains, "\n");
        nrPoints++;
      }
      fclose(foutGrains);
    } else if (strncmp(aline, "# SpotID", strlen("# SpotID")) == 0) { // SpotID
      dataType = 0;
      while (fgets(aline, 4096, inpF) != NULL) {
        sscanf(aline,
               "%s,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%s,%lf,%lf,%lf,%s,%lf,%"
               "lf,%lf,%lf,%lf,%lf",
               dummy, &InputInfo[nrPoints][0], &InputInfo[nrPoints][1],
               &InputInfo[nrPoints][2], &InputInfo[nrPoints][3],
               &InputInfo[nrPoints][4], &InputInfo[nrPoints][5],
               &InputInfo[nrPoints][6], &InputInfo[nrPoints][7],
               &InputInfo[nrPoints][8], dummy, &InputInfo[nrPoints][9],
               &InputInfo[nrPoints][10], &InputInfo[nrPoints][11], dummy,
               &InputInfo[nrPoints][12], &InputInfo[nrPoints][13],
               &InputInfo[nrPoints][14], &InputInfo[nrPoints][15],
               &InputInfo[nrPoints][16], &InputInfo[nrPoints][17]);
        nrPoints++;
      }
    } else if (strncmp(aline, "#EBSD", strlen("#EBSD")) == 0) { // EBSD
      dataType = 1;
      printf("We are reading an EBSD file.\n");
      fgets(aline, 4096, inpF);
      while (fgets(aline, 4096, inpF) != NULL) {
        for (char *p = aline; *p; ++p) {
          if (*p == ',' || *p == '\t')
            *p = ' ';
        }
        int itemsRead =
            sscanf(aline, "%lf %lf %lf %lf %lf %lf", &InputInfo[nrPoints][9],
                   &InputInfo[nrPoints][10], &InputInfo[nrPoints][11],
                   &EulerThis[0], &EulerThis[1], &EulerThis[2]);
        if (itemsRead != 6) {
          continue;
        }
        EulerThis[0] *= deg2rad;
        EulerThis[1] *= deg2rad;
        EulerThis[2] *= deg2rad;
        Euler2OrientMat(EulerThis, OrientThis);
        for (i = 0; i < 9; i++) {
          InputInfo[nrPoints][i] = OrientThis[i];
        }
        for (i = 0; i < 6; i++) {
          InputInfo[nrPoints][i + 12] = LatC[i];
        }
        nrPoints++;
      }
      printf("NrPoints: %ld\n", nrPoints);
    } else if (strncmp(aline, "%TriEdgeSize ", strlen("%TriEdgeSize ")) ==
               0) { // TriEdge
      dataType = 1;
      // NrOrientations = 2000000; ?? Unused
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      double zThis;
      sscanf(aline, "%s %lf", dummy, &zThis);
      fgets(aline, 4096, inpF);
      double thisConfidence;
      while (fgets(aline, 4096, inpF) != NULL) {
        sscanf(aline, "%s %s %s %lf %lf %s %s %lf %lf %lf %lf %s", dummy, dummy,
               dummy, &InputInfo[nrPoints][9], &InputInfo[nrPoints][10], dummy,
               dummy, &EulerThis[0], &EulerThis[1], &EulerThis[2],
               &thisConfidence, dummy);
        if (thisConfidence <= minConfidence)
          continue;
        InputInfo[nrPoints][11] = zThis;
        Euler2OrientMat(EulerThis, OrientThis);
        for (i = 0; i < 9; i++) {
          InputInfo[nrPoints][i] = OrientThis[i];
        }
        for (i = 0; i < 6; i++) {
          InputInfo[nrPoints][i + 12] = LatC[i];
        }
        nrPoints++;
      }
    } else if (strncmp(aline, "# vtk DataFile ", strlen("# vtk DataFile ")) ==
               0) { // VTK
      dataType = 2;
      long long int totalPoints, totalElements;
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      sscanf(aline, "%s %lld", dummy, &totalPoints);
      for (i = 0; i < totalPoints; i++)
        fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      sscanf(aline, "%s %lld", dummy, &totalElements);
      for (i = 0; i < totalElements + 1; i++)
        fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements + 5; i++)
        fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements; i++) {
        fgets(aline, 4096, inpF);
        sscanf(aline, "%lf", &InputInfo[i][20]);
      }
      for (i = 0; i < 3; i++)
        fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements; i++) {
        fgets(aline, 4096, inpF);
        sscanf(aline, "%lf", &InputInfo[i][18]);
        if (maxVol < InputInfo[i][18])
          maxVol = InputInfo[i][18];
      }
      for (i = 0; i < 3; i++)
        fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements; i++) {
        fgets(aline, 4096, inpF);
        sscanf(aline, "%lf %lf %lf", &InputInfo[i][9], &InputInfo[i][10],
               &InputInfo[i][11]);
      }
      for (i = 0; i < 3; i++)
        fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements; i++) {
        fgets(aline, 4096, inpF);
        if (UpdatedOrientations == 0) {
          sscanf(aline, "%lf %lf %lf", &EulerThis[0], &EulerThis[1],
                 &EulerThis[2]);
          Euler2OrientMat(EulerThis, OrientThis);
          for (j = 0; j < 9; j++) {
            InputInfo[i][j] = OrientThis[j];
          }
        }
      }
      for (i = 0; i < 3; i++)
        fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      for (i = 0; i < totalElements; i++) {
        fgets(aline, 4096, inpF);
        sscanf(aline, "%lf", &InputInfo[i][19]);
      }
      for (i = 0; i < 2; i++)
        fgets(aline, 4096, inpF);
      fgets(aline, 4096, inpF);
      sscanf(aline, "%s %s", dummy, strLine);
      while (strncmp(strLine, "Plastic", strlen("Plastic")) == 0) {
        for (i = 0; i < totalElements + 3; i++)
          fgets(aline, 4096, inpF);
        fgets(aline, 4096, inpF);
        sscanf(aline, "%s %s", dummy, strLine);
      }
      fgets(aline, 4096, inpF);
      if (LoadNr == 0) {
        for (i = 0; i < totalElements; i++)
          for (j = 12; j < 18; j++)
            InputInfo[i][j] = 0;
      } else if (LoadNr == 1) {
        for (i = 0; i < totalElements; i++) {
          fgets(aline, 4096, inpF);
          sscanf(aline, "%lf %lf %lf %lf %lf %lf", &InputInfo[i][12],
                 &InputInfo[i][15], &InputInfo[i][17], &InputInfo[i][13],
                 &InputInfo[i][14], &InputInfo[i][16]);
        }
      } else if (LoadNr > 1) {
        nrSkip = (LoadNr - 1) * (totalElements + 4);
        for (i = 0; i < nrSkip; i++)
          fgets(aline, 4096, inpF);
        for (i = 0; i < totalElements; i++) {
          fgets(aline, 4096, inpF);
          sscanf(aline, "%lf %lf %lf %lf %lf %lf", &InputInfo[i][12],
                 &InputInfo[i][15], &InputInfo[i][17], &InputInfo[i][13],
                 &InputInfo[i][14], &InputInfo[i][16]);
        }
      }
      for (i = 0; i < 3; i++)
        fgets(aline, 4096, inpF);
      sscanf(aline, "%s %s", dummy, strLine);
      if (UpdatedOrientations == 1) {
        while (strncmp(strLine, "Orientation", strlen("Orientation")) != 0) {
          for (i = 0; i < totalElements + 4; i++)
            fgets(aline, 4096, inpF);
          sscanf(aline, "%s %s", dummy, strLine);
        }
        fgets(aline, 4096, inpF);
        if (LoadNr == 1) {
          for (i = 0; i < totalElements; i++) {
            fgets(aline, 4096, inpF);
            sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &OrientTemp[0][0], &OrientTemp[1][0], &OrientTemp[2][0],
                   &OrientTemp[0][1], &OrientTemp[1][1], &OrientTemp[2][1],
                   &OrientTemp[0][2], &OrientTemp[1][2], &OrientTemp[2][2]);
            OrientMat2Euler(OrientTemp, EulerThis);
            Euler2OrientMat(EulerThis, OrientThis);
            for (j = 0; j < 9; j++) {
              InputInfo[i][j] = OrientThis[j];
            }
          }
        } else {
          nrSkip = (LoadNr - 1) * (totalElements + 4);
          for (i = 0; i < nrSkip; i++)
            fgets(aline, 4096, inpF);
          for (i = 0; i < totalElements; i++) {
            fgets(aline, 4096, inpF);
            sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &OrientTemp[0][0], &OrientTemp[1][0], &OrientTemp[2][0],
                   &OrientTemp[0][1], &OrientTemp[1][1], &OrientTemp[2][1],
                   &OrientTemp[0][2], &OrientTemp[1][2], &OrientTemp[2][2]);
            OrientMat2Euler(OrientTemp, EulerThis);
            Euler2OrientMat(EulerThis, OrientThis);
            for (j = 0; j < 9; j++) {
              InputInfo[i][j] = OrientThis[j];
            }
          }
        }
      }
      FILE *fl;
      fl = fopen("Orientations.txt", "w");
      int nrElements = 0;
      for (i = 0; i < totalElements; i++) {
        if ((int)InputInfo[i][20] == 1) {
          nrElements++;
          for (j = 0; j < 9; j++)
            fprintf(fl, "%f ", InputInfo[i][j]);
          fprintf(fl, "\n");
        }
      }
      printf("%d\n", nrElements);
      fclose(fl);
      nrPoints = totalElements;
    }
    fclose(inpF);
  }

  *dataType_out = dataType;
  *maxVol_out = maxVol;
  return nrPoints;
}

int main(int argc, char *argv[]) {

  if (argc != 3) {
    usage();
    return 0;
  }
  clock_t start0, end;
  double diftotal;
  start0 = clock();
  int i, j, k, t, p;

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  int nCPUs = atoi(argv[2]);
  run_midas_binary("GetHKLList", ParamFN);
  // char cmD[4096];
  // sprintf(cmD,"~/opt/MIDAS/FF_HEDM/bin/GetHKLList %s",ParamFN);
  // system(cmD);
  int LowNr;
  char *str, dummy[4096], aline[4096];
  fileParam = fopen(ParamFN, "r");
  char InFileName[4096], OutFileName[4096];
  int Padding = 6, NrPixels;
  double Lsd, tx, ty, tz, yBC, zBC, OmegaStep, OmegaStart, OmegaEnd, px;
  int RingsToUse[500], nRings = 0;
  double LatC[6], Wavelength, Wedge = 0, p0, p1, p2, RhoD, GaussWidth = 1,
                              PeakIntensity = 2000;
  int writeSpots, isBin = 0;
  int LoadNr = 0, UpdatedOrientations = 1, nfOutput = 0, geOutput = 1;
  double minConfidence = 0;
  int nScans = 1;
  double *positions, beamSize = -1;
  double eResolution = 0.0; // Default to monochromatic (zero spread)
  char IntensitiesFile[4096] = "";
  double RingNrIntensity[500];
  int num_lambda_samples = 1;
  while (fgets(aline, 4096, fileParam) != NULL) {
    str = "IntensitiesFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, IntensitiesFile);
      continue;
    }
    str = "RingsToUse ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRings]);
      nRings++;
      continue;
    }
    str = "RingThresh ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRings]);
      nRings++;
      continue;
    }
    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatC[0], &LatC[1],
             &LatC[2], &LatC[3], &LatC[4], &LatC[5]);
      continue;
    }
    str = "LatticeConstant ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatC[0], &LatC[1],
             &LatC[2], &LatC[3], &LatC[4], &LatC[5]);
      continue;
    }
    str = "InFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, InFileName);
      continue;
    }
    str = "OutFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, OutFileName);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "MinConfidence ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &minConfidence);
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
    str = "RhoD ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &RhoD);
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
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &yBC, &zBC);
      continue;
    }
    str = "OmegaStep ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStep);
      continue;
    }
    str = "OmegaStart ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStart);
      continue;
    }
    str = "OmegaEnd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaEnd);
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
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixels);
      continue;
    }
    str = "nScans ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nScans);
      continue;
    }
    str = "BeamSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &beamSize);
      continue;
    }
    str = "WriteSpots ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &writeSpots);
      continue;
    }
    str = "LoadNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &LoadNr);
      continue;
    }
    str = "UpdatedOrientations ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &UpdatedOrientations);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "GaussWidth ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &GaussWidth);
      continue;
    }
    str = "PeakIntensity ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &PeakIntensity);
      continue;
    }
    str = "IsBinary ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &isBin);
      continue;
    }
    str = "NFOutput ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nfOutput);
      continue;
    }
    str = "GEOutput ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &geOutput);
      continue;
    }
    str = "EResolution ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %d", dummy, &eResolution, &num_lambda_samples);
      continue;
    }
  }
  if (OmegaStep == 0.0) {
    fprintf(stderr, "Error: OmegaStep cannot be zero. Aborting.\n");
    return 1;
  }
  if (GaussWidth <= 0.0) {
    fprintf(stderr, "Error: GaussWidth must be a positive number. Aborting.\n");
    return 1;
  }
  if (NrPixels <= 0) {
    fprintf(stderr, "Error: NrPixels must be a positive number. Aborting.\n");
    return 1;
  }

  int scanNr;
  positions = malloc((nScans + 1) * sizeof(*positions));
  positions[0] = 0;
  if (nScans > 1) {
    printf("Multi-scan simulation requested with %d scans.\nReading "
           "positions.csv, no error checking from here on!!!\n",
           nScans);
    FILE *positionsF = fopen("positions.csv", "r");
    for (scanNr = 0; scanNr < nScans; scanNr++) {
      if (fgets(aline, sizeof(aline), positionsF) != NULL) {
        if (sscanf(aline, "%lf", &positions[scanNr]) == 1) {
          printf("Position: %d value: %lf\n", scanNr, positions[scanNr]);
        } else {
          fprintf(stderr, "Warning: Could not parse a number from line %d: %s",
                  scanNr + 1, aline);
        }
      } else {
        fprintf(stderr, "Warning: Could not read line %d from positions.csv.\n",
                scanNr + 1);
        break;
      }
    }
  }
  printf("Output will be saved to: %s_scanNr_XYZ.zip, XYZ is the scanNr\n",
         OutFileName);

  char inpFN[4096];
  sprintf(inpFN, "%s", InFileName);
  // Preallocate Arrays
  long NrOrientations, nrPoints = 0;
  double EulerThis[3], zThis, OrientThis[9], ElasticStrainThis[6],
      OrientTemp[3][3];
  double **InputInfo;
  int dataType;
  double maxVol = 0;
  char strLine[4096];
  int nrSkip;
  FILE *inpF;
  InputInfo = allocMatrix(MAX_NR_POINTS, 21);
  nrPoints = ReadInputData(inpFN, isBin, InputInfo, LatC, minConfidence,
                           UpdatedOrientations, LoadNr, &dataType, &maxVol);
  if (nrPoints == 0)
    return 1;
  printf("Read file., total number of orientations: %d\n", nrPoints);
  if (writeSpots == 1)
    printf("Will generate a SpotMatrixGen.csv file.\n");
  else
    printf("Will not generate a SpotMatrixGen.csv file.\n");
  // Read hkls file.
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  int thisRingNr;
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%lf %lf %lf %s %lf", &hkls[n_hkls][0], &hkls[n_hkls][1],
           &hkls[n_hkls][2], dummy, &hkls[n_hkls][3]);
    n_hkls++;
    if (n_hkls >= MAX_N_HKLS) {
      printf("MAX_N_HKLS exceeded. Please check! Aborting.\n");
      return 1;
    }
  }
  for (i = 0; i < n_hkls; i++)
    mHKLs[i] = 0;
  if (nRings > 0) {
    double hklTemps[n_hkls][4];
    int totalHKLs = 0;
    for (i = 0; i < nRings; i++) {
      for (j = 0; j < n_hkls; j++) {
        if ((int)hkls[j][3] == RingsToUse[i]) {
          hklTemps[totalHKLs][0] = hkls[j][0];
          hklTemps[totalHKLs][1] = hkls[j][1];
          hklTemps[totalHKLs][2] = hkls[j][2];
          hklTemps[totalHKLs][3] = hkls[j][3];
          mHKLs[(int)hkls[j][3]]++;
          totalHKLs++;
        }
      }
    }
    for (i = 0; i < totalHKLs; i++) {
      hkls[i][0] = hklTemps[i][0];
      hkls[i][1] = hklTemps[i][1];
      hkls[i][2] = hklTemps[i][2];
      hkls[i][3] = hklTemps[i][3];
    }
    n_hkls = totalHKLs;
  }
  //~ for (i=0;i<n_hkls;i++) printf("%lf ",hkls[i][3]); printf("\n");
  printf("Number of planes: %d\n", n_hkls);

  // If an IntensitiesFile is provided, read it and populate the RingNrIntensity
  // array.
  if (strcmp(IntensitiesFile, "") != 0) {
    FILE *intensities_file = fopen(IntensitiesFile, "r");
    if (intensities_file == NULL) {
      fprintf(stderr,
              "Error: Could not open the specified IntensitiesFile: %s\n",
              IntensitiesFile);
      return 1; // Exit if the file cannot be opened
    }

    // Initialize array with zeros
    for (i = 0; i < 500; i++) {
      RingNrIntensity[i] = 0.0;
    }

    char line_buffer[1024];
    double h_val, k_val, l_val, spot_int, d_spacing;
    int ring_nr;

    // Read and discard the header line
    if (fgets(line_buffer, sizeof(line_buffer), intensities_file) == NULL) {
      fprintf(stderr, "Error: Could not read header from IntensitiesFile: %s\n",
              IntensitiesFile);
      fclose(intensities_file);
      return 1;
    }

    // Read the file line by line
    while (fgets(line_buffer, sizeof(line_buffer), intensities_file) != NULL) {
      if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %d", &h_val, &k_val, &l_val,
                 &spot_int, &d_spacing, &ring_nr) == 6) {
        if (ring_nr >= 0 && ring_nr < 500) {
          // Sum intensities for hkls belonging to the same ring
          RingNrIntensity[ring_nr] += spot_int;
        }
      }
    }
    fclose(intensities_file);

    // Normalize the intensities so the maximum value is 1.0
    double max_intensity = 0.0;
    for (i = 0; i < 500; i++) {
      if (RingNrIntensity[i] > max_intensity) {
        max_intensity = RingNrIntensity[i];
      }
    }
    if (max_intensity > 0) {
      for (i = 0; i < 500; i++) {
        RingNrIntensity[i] /= max_intensity;
      }
    }
    printf("Successfully read and processed intensities from %s\n",
           IntensitiesFile);
  } else {
    // If no file is provided, default all ring intensities to 1.0
    // so that it does not affect the final intensity calculation.
    for (i = 0; i < 500; i++) {
      RingNrIntensity[i] = 1.0;
    }
  }

  // Allocate image array.
  double maxInt;
  double *ImageArr;
  size_t ImageArrSize;
  ImageArrSize = NrPixels;
  ImageArrSize *= NrPixels;
  ImageArrSize *= ceil(fabs((OmegaEnd - OmegaStart) / OmegaStep));
  ImageArr = calloc(ImageArrSize, sizeof(*ImageArr));
  printf("Number of elements in ImageArray: %zu, number of frames: %zu\n",
         ImageArrSize, (size_t)ceil(fabs((OmegaEnd - OmegaStart) / OmegaStep)));
  if (ImageArr == NULL) {
    printf("Could not allocate enough memory for image array. Exiting.\n");
    return 1;
  }
  // Make distortion tilt map for each pixel within the hkl range
  printf("Making distortion map.\n");
  double *yDispl, *zDispl;
  yDispl = malloc(NrPixels * NrPixels *
                  sizeof(*yDispl)); // Arrangement: y+ z*NrPixels
  zDispl = malloc(NrPixels * NrPixels * sizeof(*zDispl));
  for (i = 0; i < NrPixels * NrPixels; i++) {
    yDispl[i] =
        -32100.0; // Set to a special number to check if it was set or not.
  }
  CorrectTiltSpatialDistortion(px, Lsd, yBC, zBC, tx, ty, tz, RhoD, p0, p1, p2,
                               NrPixels, yDispl, zDispl);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("Distortion map done in %lf sec.\n", diftotal);

  // Make GaussMask for blurring - REMOVED (Analytic Integration)
  // int nrPxMask = 1 + 8*((int)ceil(GaussWidth));
  // ...

  char spotMatrFN[4096];
  sprintf(spotMatrFN, "SpotMatrixGen.csv");
  FILE *spotsfile = fopen(spotMatrFN, "w");
  fprintf(spotsfile,
          "%%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta"
          "\tRingNr\tYLab\tZLab\tTheta\tStrainError\tScanNr\tRingRad\n");
  double spotMatr[12];
  double **TheorSpots;
  int nTspots, voxNr, spotNr;
  int nRowsPerGrain = 2 * n_hkls;
  TheorSpots = allocMatrix(nRowsPerGrain, 7);
  double OM[3][3], LatCThis[6], **hklsOut, **hklsTemp, EpsThis[6];
  double RotMatTilts[3][3];
  RotationTilts(tx, ty, tz, RotMatTilts);
  double MatIn[3], P0[3], P0T[3];
  MatIn[0] = 0;
  MatIn[1] = 0;
  MatIn[2] = 0;
  MatIn[0] = -Lsd;
  MatrixMultF(RotMatTilts, MatIn, P0T);
  for (j = 0; j < 3; j++) {
    P0[j] = P0T[j];
  }
  double OmeDiff, yTemp, zTemp, yThis, omeThis, etaThis;
  double Info[5], DisplY, DisplZ, yDet, zDet, DisplY2, DisplZ2, xGrain, yGrain,
      xyz[3], P1[3], ABC[3], outxyz[3], YZSpotsT[2];
  double newY = 0, yOffset = 0;
  int yTrans, zTrans;
  int idxNrY, idxNrZ;
  long long int idx;
  size_t omeBin, yBin, zBin, imageBin, centIdx;
  long long int currentPos;
  size_t displ;
  hklsOut = allocMatrix(n_hkls, 5);
  hklsTemp = allocMatrix(n_hkls, 5);
  printf("Total number of orientations: %d\n", nrPoints);
  double omegaLarge = OmegaStart > OmegaEnd ? OmegaStart : OmegaEnd;
  double omegaSmall = OmegaStart < OmegaEnd ? OmegaStart : OmegaEnd;
  printf("Omega range for simulation: min: %lf, max: %lf\n", omegaSmall,
         omegaLarge);
  // Look at OpenMP?
  for (scanNr = 0; scanNr < nScans; scanNr++) {
    if (scanNr > 0)
      memset(ImageArr, 0, ImageArrSize * sizeof(*ImageArr));
    yOffset = positions[scanNr];
    printf("yPosition: %lf\n", yOffset);
#pragma omp parallel for num_threads(nCPUs)                                    \
    private(voxNr, i, j, LatCThis, EpsThis, OM, EulerThis, nTspots, spotNr,    \
                Info, OmeDiff, omeThis, newY, yTemp, zTemp, yThis, zThis,      \
                DisplY2, DisplZ2, yTrans, zTrans, idx, DisplY, DisplZ, yDet,   \
                zDet, etaThis, spotMatr, omeBin, yBin, zBin, imageBin,         \
                centIdx, idxNrY, idxNrZ, displ, currentPos)
    for (voxNr = 0; voxNr < nrPoints; voxNr++) {
      // First calculate new hkls
      if (dataType < 2) {
        for (i = 0; i < 6; i++)
          LatCThis[i] = InputInfo[voxNr][i + 12];
        CorrectHKLsLatC(LatCThis, Wavelength, hklsOut);
      } else if (dataType == 2) {
        if (InputInfo[voxNr][i + 19] == 0)
          continue;
        for (i = 0; i < 6; i++)
          EpsThis[i] = InputInfo[voxNr][i + 12];
        CorrectHKLsLatCEpsilon(LatC, EpsThis, Wavelength, hklsOut);
      } else if (dataType == 3) { // binary file
        for (i = 0; i < 6; i++)
          EpsThis[i] = InputInfo[voxNr][i + 12];
        CorrectHKLsLatCEpsilon(LatC, EpsThis, Wavelength, hklsOut);
      }
      // Get the Orientation Matrix
      for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
          OM[i][j] = InputInfo[voxNr][i * 3 + j];
        }
      }
      OrientMat2Euler(OM, EulerThis);
      double OMT[9];
      Euler2OrientMat(EulerThis, OMT);
      for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
          OM[i][j] = OMT[i * 3 + j];
        }
      }
      // Calculate the spots now.
      CalcDiffrSpots_Furnace(hklsOut, OM, Lsd, Wavelength, TheorSpots,
                             &nTspots);
      // For each spot, calculate displacement, calculate tilt and wedge effect.
      for (spotNr = 0; spotNr < nTspots; spotNr++) {
        double sub_peak_intensity = PeakIntensity / num_lambda_samples;
        double original_theta_rad = deg2rad * TheorSpots[spotNr][3];
        double d_spacing = Wavelength / (2 * sin(original_theta_rad));

        for (int lambda_i = 0; lambda_i < num_lambda_samples; lambda_i++) {
          // Calculate the specific wavelength for this sample
          double sample_lambda = Wavelength;
          if (eResolution > 0 && num_lambda_samples > 1) {
            double delta_lambda = Wavelength * eResolution;
            sample_lambda =
                (Wavelength - delta_lambda / 2.0) +
                lambda_i * (delta_lambda / (num_lambda_samples - 1));
          }
          // Recalculate theta for this new wavelength
          // We use the invariant d-spacing calculated outside
          double sin_sample_theta = sample_lambda / (2 * d_spacing);
          // If the sampled wavelength is too far off, the reflection is
          // impossible. Skip it.
          if (sin_sample_theta > 1.0 || sin_sample_theta < -1.0) {
            continue;
          }
          double sample_theta = asind(sin_sample_theta);
          // Recalculate all parameters that depend on theta and create a
          // temporary spot
          double sample_ring_radius = Lsd * tan(2 * deg2rad * sample_theta);
          double TempInfo[5];
          for (i = 0; i < 5; i++)
            TempInfo[i] = TheorSpots[spotNr][i];
          TempInfo[0] = sample_ring_radius;
          TempInfo[3] = sample_theta;
          // Run the original simulation logic for this single sub-spot
          OmeDiff =
              CorrectWedge(TempInfo[1], TempInfo[3], sample_lambda, Wedge);
          omeThis = TempInfo[2] - OmeDiff;
          if ((omeThis >= omegaLarge) || (omeThis <= omegaSmall))
            continue;
          if (nScans > 1 || beamSize > 0) {
            newY = InputInfo[voxNr][9] * sin(deg2rad * omeThis) +
                   InputInfo[voxNr][10] * cos(deg2rad * omeThis);
            if (fabs(newY - yOffset) > beamSize / 2) {
              continue;
            }
          }
          yTemp = -TempInfo[0] * sin(TempInfo[1] * deg2rad);
          zTemp = TempInfo[0] * cos(TempInfo[1] * deg2rad);
          DisplacementInTheSpot(InputInfo[voxNr][9], InputInfo[voxNr][10],
                                InputInfo[voxNr][11], Lsd, yTemp, zTemp,
                                omeThis, &DisplY2, &DisplZ2);
          yThis = yTemp + DisplY2 + yOffset;
          zThis = zTemp + DisplZ2;
          yTrans = (int)(-yThis / px + yBC);
          zTrans = (int)(zThis / px + zBC);
          idx = yTrans + NrPixels * zTrans;
          if (idx < 1 || idx >= NrPixels * NrPixels)
            continue;
          DisplY = yDispl[idx];
          DisplZ = zDispl[idx];
          if (DisplY == -32100) {
            if (idx - NrPixels < 0 || idx + NrPixels > NrPixels * NrPixels - 1)
              continue;
            if (yDispl[idx - 1] != -32100.0) {
              DisplY = yDispl[idx - 1];
              DisplZ = zDispl[idx - 1];
            } else if (yDispl[idx + 1] != -32100.0) {
              DisplY = yDispl[idx + 1];
              DisplZ = zDispl[idx + 1];
            } else if (yDispl[idx - NrPixels] != -32100.0) {
              DisplY = yDispl[idx - NrPixels];
              DisplZ = zDispl[idx - NrPixels];
            } else if (yDispl[idx + NrPixels] != -32100.0) {
              DisplY = yDispl[idx + NrPixels];
              DisplZ = zDispl[idx + NrPixels];
            } else {
              continue;
            }
          }
          yTemp = yThis + DisplY;
          zTemp = zThis + DisplZ;
          yDet = yBC - yTemp / px + 1;
          zDet = zBC + zTemp / px + 1.5;
          if (yDet < 0 || yDet >= NrPixels || zDet < 0 || zDet >= NrPixels)
            continue;
          // ========================================================================
          // NEW: Analytic Gaussian Integration
          // ========================================================================

          double sigma = GaussWidth;
          double twoSigmaSq = 2.0 * sigma * sigma;
          int extent = (int)ceil(4.0 * sigma);
          if (extent > 49)
            extent = 49; // Clamp to buffer size (buffer is 100)

          int y_base = (int)round(yDet);
          int z_base = (int)round(zDet);

          omeBin = (size_t)floor(-(OmegaStart - omeThis) / OmegaStep);
          size_t frameOffset = omeBin * NrPixels * NrPixels;

          // Precompute 1D Gaussian weights
          double WeightsY[100];
          double WeightsZ[100];

          for (int dy = -extent; dy <= extent; dy++) {
            int y_curr = y_base + dy;
            if (y_curr < 0 || y_curr >= NrPixels) {
              WeightsY[dy + extent] = 0.0;
              continue;
            }
            double distY = (double)y_curr - yDet;
            WeightsY[dy + extent] = exp(-(distY * distY) / twoSigmaSq);
          }

          for (int dz = -extent; dz <= extent; dz++) {
            int z_curr = z_base + dz;
            if (z_curr < 0 || z_curr >= NrPixels) {
              WeightsZ[dz + extent] = 0.0;
              continue;
            }
            double distZ = (double)z_curr - zDet;
            WeightsZ[dz + extent] = exp(-(distZ * distZ) / twoSigmaSq);
          }

          // Convolve using precomputed weights
          for (int dy = -extent; dy <= extent; dy++) {
            double wy = WeightsY[dy + extent];
            if (wy == 0.0)
              continue;

            int y_curr = y_base + dy;

            for (int dz = -extent; dz <= extent; dz++) {
              double wz = WeightsZ[dz + extent];
              if (wz == 0.0)
                continue;

              int z_curr = z_base + dz;

              // Gaussian weight
              double weight = wy * wz;

              // Check relative intensity from rings
              int currentRingNr = (int)TempInfo[4];
              double relativeIntensity = 1.0;
              if (currentRingNr >= 0 && currentRingNr < 500) {
                relativeIntensity = RingNrIntensity[currentRingNr];
              }

              // Total intensity contribution
              double intensity_to_add =
                  weight * sub_peak_intensity * relativeIntensity;

              long long int currentPos =
                  (long long int)(frameOffset + y_curr * NrPixels + z_curr);
              if (currentPos >= 0 && currentPos < ImageArrSize) {
#pragma omp atomic
                ImageArr[currentPos] += intensity_to_add;
              }
            }
          }
        }
        if (writeSpots == 1) {
#pragma omp critical
          {
            fprintf(spotsfile,
                    "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%d\t%lf\t%lf\t%lf\t%lf\t%"
                    "d\t%lf\t%d\n",
                    (int)spotMatr[0], (int)spotMatr[1], spotMatr[2],
                    spotMatr[3], spotMatr[4], spotMatr[5], spotMatr[6],
                    (int)spotMatr[7], spotMatr[8], spotMatr[9], spotMatr[10],
                    spotMatr[11], scanNr, sqrt(yThis * yThis + zThis * zThis),
                    omeBin);
          }
        }
      }
    }
    maxInt = 0.0;
    for (size_t i = 0; i < ImageArrSize; ++i) {
      if (ImageArr[i] > maxInt) {
        maxInt = ImageArr[i];
      }
    }
    printf("Maximum intensity: %lf, scanNr: %d\n", maxInt, scanNr);
    if (geOutput == 1) {
      blosc_init();
      blosc_set_nthreads(10);
      int nFrames = (int)ceil(fabs((OmegaEnd - OmegaStart) / OmegaStep));
      int frameNr;
      uint16_t *data_out, *outArr;
      outArr = calloc(NrPixels * NrPixels, sizeof(*outArr));
      size_t loc = 0;
      int compressedSize;
      char outfn[4096];
      int errorp;
      char outZipFN[4096];
      sprintf(outZipFN, "%s_scanNr_%d.zip", OutFileName, scanNr);
      printf("Diffraction spots done, now writing zip file: %s\n", outZipFN);
      zip_t *zipper = zip_open(outZipFN, ZIP_CREATE | ZIP_TRUNCATE, &errorp);
      if (zipper == NULL) {
        printf("Could not open the zip file %s for writing. Exiting.\n",
               outZipFN);
        return 1;
      }
      int rc;
      char outstr0[8192];
      sprintf(outstr0, "{\n    \"zarr_format\": 2\n}");
      char zarrfn0[8192];
      sprintf(zarrfn0, ".zgroup");
      int rcv0 = writeStrZip(outstr0, zarrfn0, zipper);
      if (rcv0 != 0) {
        printf("Did not succeed to add %s to zip. Exiting.\n", zarrfn0);
        return 1;
      }
      char zarrfn1[8192];
      sprintf(zarrfn1, "exchange/.zgroup");
      rcv0 = writeStrZip(outstr0, zarrfn1, zipper);
      if (rcv0 != 0) {
        printf("Did not succeed to add %s to zip. Exiting.\n", zarrfn1);
        return 1;
      }
      char outstr2[8192];
      sprintf(outstr2,
              "{\n    \"chunks\": [\n        1,\n        %d,\n        %d\n"
              "    ],\n    \"compressor\": {\n        \"blocksize\": 0,\n      "
              "  \"clevel\": 3,\n"
              "\n        \"cname\": \"zstd\",\n        \"id\": \"blosc\",\n    "
              "    \"shuffle\": 2\n"
              "    },\n    \"dtype\": \"<u2\",\n    \"fill_value\": 0,\n    "
              "\"filters\": null,\n"
              "    \"order\": \"C\",\n    \"shape\": [\n        %d,\n        "
              "%d,\n        %d\n    ],\n"
              "    \"zarr_format\": 2\n}",
              NrPixels, NrPixels, nFrames, NrPixels, NrPixels);
      char zarrfn2[8192];
      sprintf(zarrfn2, "exchange/data/.zarray");
      rcv0 = writeStrZip(outstr2, zarrfn2, zipper);
      if (rcv0 != 0) {
        printf("Did not succeed to add %s to zip. Exiting.\n", zarrfn2);
        return 1;
      }
      char outstr3[8192];
      sprintf(outstr3,
              "{\n    \"_ARRAY_DIMENSIONS\": [\n        %d,\n        %d,\n     "
              "   %d\n    ]\n}",
              nFrames, NrPixels, NrPixels);
      char zarrfn3[8192];
      sprintf(zarrfn3, "exchange/data/.zattrs");
      rcv0 = writeStrZip(outstr3, zarrfn3, zipper);
      if (rcv0 != 0) {
        printf("Did not succeed to add %s to zip. Exiting.\n", zarrfn3);
        return 1;
      }
      for (frameNr = 0; frameNr < nFrames; frameNr++) {
        int locCounter = 0;
        for (i = 0; i < NrPixels * NrPixels; i++) {
          if (maxInt > 0) {
            outArr[i] =
                (uint16_t)(ImageArr[loc] * MAX_OUTPUT_INTENSITY / maxInt);
          } else {
            outArr[i] = 0; // If maxInt is 0, the whole image is black.
          }
          if (outArr[i] > 0)
            locCounter++;
          loc++;
        }
        data_out = calloc(NrPixels * NrPixels, sizeof(*data_out));
        blosc_set_compressor("zstd");
        compressedSize = blosc_compress(
            3, 2, 2, NrPixels * NrPixels * sizeof(uint16_t), outArr, data_out,
            NrPixels * NrPixels * sizeof(uint16_t));
        sprintf(outfn, "exchange/data/%d.0.0", frameNr);
        // printf("%zu %s %d %d\n",loc,outfn,compressedSize,locCounter);
        zip_error_t *errp;
        const void *dataT;
        dataT = (const void *)data_out;
        zip_source_t *source =
            zip_source_buffer(zipper, dataT, compressedSize, 1);
        zip_int64_t rct = zip_file_add(zipper, outfn, source,
                                       ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
        if (rct < 0) {
          printf("Could not add the file %s to zip. Exiting.\n", outfn);
          return 1;
        }
        rc = zip_set_file_compression(zipper, rct, ZIP_CM_STORE, 0);
        if (rc != 0) {
          printf("Could not change compression type of the file %s to zip. "
                 "Exiting.\n",
                 outfn);
          return 1;
        }
      }
      int zc = zip_close(zipper);
      free(outArr);
    }
  }
  fclose(spotsfile);
  // =========================================================================
  // FINAL MEMORY CLEANUP
  // =========================================================================
  printf("Cleaning up allocated memory...\n");
  FreeMemMatrix(InputInfo, MAX_NR_POINTS);
  FreeMemMatrix(TheorSpots, nRowsPerGrain);
  FreeMemMatrix(hklsOut, n_hkls);
  FreeMemMatrix(hklsTemp, n_hkls);
  free(positions);
  free(ImageArr);
  free(yDispl);
  free(zDispl);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("Time elapsed in making diffraction spots: %f [s]\n", diftotal);
}
