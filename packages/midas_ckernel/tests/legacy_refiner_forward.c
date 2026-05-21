/* legacy_refiner_forward.c — FROZEN copy of the original refiner forward model
 * (CalcDiffrSpots_Furnace / CalcDiffractionSpots from FF_HEDM/src), kept ONLY
 * as the parity reference for tests/parity_test.c.
 *
 * The shipped refiner (midas_fit_grain/c_src/CalcDiffractionSpots.c) is now a
 * thin adapter over the shared forward, so it can no longer serve as an
 * independent reference. This file preserves the pre-unification math verbatim
 * (function renamed ref_refiner_CalcDiffractionSpots) so the harness keeps
 * proving the shared forward reproduces it. DO NOT wire into production.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MIDAS_Limits.h"

#define LRF_deg2rad (M_PI / 180.0)
#define LRF_rad2deg (180.0 / M_PI)
#define LRF_TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;

/* MIDAS_Math primitives (linked from MIDAS_Math.c). */
void MatrixMultF(double m[3][3], double v[3], double r[3]);
void RotateAroundZ(double v1[3], double alpha, double v2[3]);
void CalcEtaAngle(double y, double z, double *alpha);

static void lrf_CalcSpotPosition(double RingRadius, double eta, double *yl,
                                 double *zl) {
  double etaRad = LRF_deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

static void lrf_CalcOmega(double x, double y, double z, double theta,
                          double omegas[4], double etas[4], int *nsol) {
  *nsol = 0;
  double ome;
  double len = sqrt(x * x + y * y + z * z);
  double v = sin(theta * LRF_deg2rad) * len;
  double almostzero = 1e-12;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      double cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * LRF_rad2deg;
        omegas[*nsol] = ome; *nsol = *nsol + 1;
        omegas[*nsol] = -ome; *nsol = *nsol + 1;
      }
    }
  } else {
    double y2 = y * y;
    double a = 1 + ((x * x) / y2);
    double b = (2 * v * x) / y2;
    double c = ((v * v) / y2) - 1;
    double discr = b * b - 4 * a * c;
    double ome1a, ome1b, ome2a, ome2b, cosome1, cosome2, eqa, eqb, diffa, diffb;
    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1) {
        ome1a = acos(cosome1); ome1b = -ome1a;
        eqa = -x * cos(ome1a) + y * sin(ome1a); diffa = fabs(eqa - v);
        eqb = -x * cos(ome1b) + y * sin(ome1b); diffb = fabs(eqb - v);
        if (diffa < diffb) { omegas[*nsol] = ome1a * LRF_rad2deg; *nsol = *nsol + 1; }
        else { omegas[*nsol] = ome1b * LRF_rad2deg; *nsol = *nsol + 1; }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2); ome2b = -ome2a;
        eqa = -x * cos(ome2a) + y * sin(ome2a); diffa = fabs(eqa - v);
        eqb = -x * cos(ome2b) + y * sin(ome2b); diffb = fabs(eqb - v);
        if (diffa < diffb) { omegas[*nsol] = ome2a * LRF_rad2deg; *nsol = *nsol + 1; }
        else { omegas[*nsol] = ome2b * LRF_rad2deg; *nsol = *nsol + 1; }
      }
    }
  }
  double gw[3], gv[3] = {x, y, z}, eta;
  for (int indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

int ref_refiner_CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                                     double OmegaRanges[MAX_N_OMEGA_RANGES][2],
                                     int NoOfOmegaRanges, double **hkls,
                                     int n_hkls,
                                     double BoxSizes[MAX_N_OMEGA_RANGES][4],
                                     int *nTspots, double OrientMatr[3][3],
                                     double **spots) {
  int i, OmegaRangeNo, indexhkl, KeepSpot = 0, nspotsPlane, spotnr = 0;
  int YCInt, ZCInt;
  long long int idx;
  double theta, Ghkl[3], Gc[3], omegas[4], etas[4], yl, zl;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0]; Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    double RingNr = hkls[indexhkl][6];
    double RingRadius = hkls[indexhkl][5];
    MatrixMultF(OrientMatr, Ghkl, Gc);
    theta = hkls[indexhkl][4];
    lrf_CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
    double GCr[3],
        NGc = sqrt((Gc[0] * Gc[0]) + (Gc[1] * Gc[1]) + (Gc[2] * Gc[2])),
        Ds = hkls[indexhkl][3];
    GCr[0] = Ds * Gc[0] / NGc; GCr[1] = Ds * Gc[1] / NGc;
    GCr[2] = Ds * Gc[2] / NGc;
    double nrhkls = (double)indexhkl * 2 + 1;
    for (i = 0; i < nspotsPlane; i++) {
      double Omega = omegas[i], Eta = etas[i];
      if (isnan(Omega) || isnan(Eta)) continue;
      double EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle))
        continue;
      lrf_CalcSpotPosition(RingRadius, etas[i], &yl, &zl);
      for (OmegaRangeNo = 0; OmegaRangeNo < NoOfOmegaRanges; OmegaRangeNo++) {
        KeepSpot = 0;
        if ((Omega > OmegaRanges[OmegaRangeNo][0]) &&
            (Omega < OmegaRanges[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) &&
            (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) &&
            (zl < BoxSizes[OmegaRangeNo][3])) { KeepSpot = 1; break; }
      }
      if (BigDetSize != 0) {
        YCInt = (int)floor((BigDetSize / 2) - (int)(-yl / pixelsize));
        ZCInt = (int)floor((((int)(zl / pixelsize) + (BigDetSize / 2))));
        idx = (long long int)(YCInt + BigDetSize * ZCInt);
        if (!LRF_TestBit(BigDetector, idx)) KeepSpot = 0;
      }
      if (KeepSpot == 1) {
        spots[spotnr][0] = yl; spots[spotnr][1] = zl; spots[spotnr][2] = omegas[i];
        spots[spotnr][3] = GCr[0]; spots[spotnr][4] = GCr[1]; spots[spotnr][5] = GCr[2];
        spots[spotnr][6] = Distance; spots[spotnr][7] = RingNr; spots[spotnr][8] = nrhkls;
        nrhkls += 1; spotnr++;
      }
    }
  }
  *nTspots = spotnr;
  return 0;
}
