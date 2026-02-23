//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  CalcDiffractionSpots.c
//
//
//  Created by Hemant Sharma on 12/3/13.
//
//

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "MIDAS_Math.h"

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define RealType double
#include "MIDAS_Limits.h"
#define EPS 0.000000001
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

// For detector mapping!
extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;

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
  OrientMat[0][1] = 2 * (Q12 + Q03);
  OrientMat[0][2] = 2 * (Q13 - Q02);
  OrientMat[1][0] = 2 * (Q12 - Q03);
  OrientMat[1][1] = 1 - 2 * (Q1_2 + Q3_2);
  OrientMat[1][2] = 2 * (Q23 + Q01);
  OrientMat[2][0] = 2 * (Q13 + Q02);
  OrientMat[2][1] = 2 * (Q23 - Q01);
  OrientMat[2][2] = 1 - 2 * (Q1_2 + Q2_2);
}

static inline void CalcSpotPosition(RealType RingRadius, RealType eta,
                                    RealType *yl, RealType *zl) {
  RealType etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

static inline void CalcOmega(RealType x, RealType y, RealType z, RealType theta,
                             RealType omegas[4], RealType etas[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  RealType len = sqrt(x * x + y * y + z * z);
  RealType v = sin(theta * deg2rad) * len;

  RealType almostzero = 1e-4;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * rad2deg;
        omegas[*nsol] = ome;
        *nsol = *nsol + 1;
        omegas[*nsol] = -ome;
        *nsol = *nsol + 1;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType a = 1 + ((x * x) / y2);
    RealType b = (2 * v * x) / y2;
    RealType c = ((v * v) / y2) - 1;
    RealType discr = b * b - 4 * a * c;

    RealType ome1a;
    RealType ome1b;
    RealType ome2a;
    RealType ome2b;
    RealType cosome1;
    RealType cosome2;

    RealType eqa, eqb, diffa, diffb;

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
  RealType gw[3];
  RealType gv[3] = {x, y, z};
  RealType eta;
  int indexOme;
  for (indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

static inline void CalcDiffrSpots_Furnace(
    RealType OrientMatrix[3][3], RealType distance,
    RealType OmegaRange[MAX_N_OMEGA_RANGES][2],
    RealType BoxSizes[MAX_N_OMEGA_RANGES][4], int NOmegaRanges, double **hkls,
    int n_hkls, RealType ExcludePoleAngle, RealType **spots, int *nspots) {
  int i, OmegaRangeNo;
  RealType theta;
  int KeepSpot = 0;
  double Ghkl[3];
  int indexhkl;
  RealType Gc[3];
  RealType omegas[4];
  RealType etas[4];
  RealType yl;
  RealType zl;
  int nspotsPlane;
  int spotnr = 0;
  int YCInt, ZCInt;
  long long int idx;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    double RingNr = hkls[indexhkl][6];
    RealType RingRadius = hkls[indexhkl][5];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = hkls[indexhkl][4];
    CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
    double GCr[3],
        NGc = sqrt((Gc[0] * Gc[0]) + (Gc[1] * Gc[1]) + (Gc[2] * Gc[2])),
        Ds = hkls[indexhkl][3];
    GCr[0] = Ds * Gc[0] / NGc;
    GCr[1] = Ds * Gc[1] / NGc;
    GCr[2] = Ds * Gc[2] / NGc;
    double nrhkls = (double)indexhkl * 2 + 1;
    for (i = 0; i < nspotsPlane; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      if (isnan(Omega) || isnan(Eta))
        continue;
      RealType EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle))
        continue;
      CalcSpotPosition(RingRadius, etas[i], &(yl), &(zl));
      for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
        KeepSpot = 0;
        if ((Omega > OmegaRange[OmegaRangeNo][0]) &&
            (Omega < OmegaRange[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) &&
            (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) &&
            (zl < BoxSizes[OmegaRangeNo][3])) {
          KeepSpot = 1;
          break;
        }
      }
      // Check if there is bigDetector, check if within the mask
      if (BigDetSize != 0) {
        YCInt = (int)floor((BigDetSize / 2) - (int)(-yl / pixelsize));
        ZCInt = (int)floor((((int)(zl / pixelsize) + (BigDetSize / 2))));
        idx = (long long int)(YCInt + BigDetSize * ZCInt);
        if (!TestBit(BigDetector, idx)) {
          KeepSpot = 0;
        }
      }
      if (KeepSpot == 1) {
        spots[spotnr][0] = yl;
        spots[spotnr][1] = zl;
        spots[spotnr][2] = omegas[i];
        spots[spotnr][3] = GCr[0];
        spots[spotnr][4] = GCr[1];
        spots[spotnr][5] = GCr[2];
        spots[spotnr][6] = distance;
        spots[spotnr][7] = RingNr;
        spots[spotnr][8] = nrhkls;
        nrhkls += 1;
        spotnr++;
      }
    }
  }
  *nspots = spotnr;
}

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAX_N_OMEGA_RANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAX_N_OMEGA_RANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots) {
  *nTspots = 0;
  int nTsps;
  if (TheorSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  CalcDiffrSpots_Furnace(OrientMatr, Distance, OmegaRanges, BoxSizes,
                         NoOfOmegaRanges, hkls, n_hkls, ExcludePoleAngle,
                         TheorSpots, &nTsps);
  *nTspots = nTsps;
}
