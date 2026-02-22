//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// SO function to set-up arrays and evaluate a single function value.
//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <signal.h>
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

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-10
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))

int numProcs;
int nIters;
struct timespec start;
struct timespec end;
double dx[4] = {-0.5, +0.5, +0.5, -0.5};
double dy[4] = {-0.5, -0.5, +0.5, +0.5};
int FitType;

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

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

static inline void
CalcEtaAngle(double y, double z,
             double *alpha) { // No return but a pointer is updated.
  *alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    *alpha = -*alpha;
}

static inline double CalcEta(double y, double z) { // Returns the eta
  double alpha;
  alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void CorrectHKLsLatC(double LatC[6], double *hklsIn, int nhkls,
                                   double Lsd, double Wavelength,
                                   double *hkls) {
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
    ginit[0] = hklsIn[hklnr * 4 + 0];
    ginit[1] = hklsIn[hklnr * 4 + 1];
    ginit[2] = hklsIn[hklnr * 4 + 2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hkls[hklnr * 5 + 0] = GCart[0];
    hkls[hklnr * 5 + 1] = GCart[1];
    hkls[hklnr * 5 + 2] = GCart[2];
    hkls[hklnr * 5 + 3] = asind((Wavelength) / (2 * Ds)); // Theta
    hkls[hklnr * 5 + 4] = hklsIn[hklnr * 4 + 3];          // RingNr
  }
}

static inline void Euler2OrientMat(double Euler[3], double m_out[3][3]) {
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

static inline void RotateAroundZ(double v1[3], double alpha, double v2[3]) {
  double cosa = cos(alpha * deg2rad);
  double sina = sin(alpha * deg2rad);
  double mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};
  MatrixMult(mat, v1, v2);
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
      if (fabs(cosome1) <= 1.0) {
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

// Function to calculate the fraction of a voxel in a beam profile.
// Omega[degrees] Assuming a gaussian beam profile.
static inline double IntensityFraction(double voxLen, double beamPosition,
                                       double beamFWHM, double voxelPosition[3],
                                       double Omega) {
  double xy[4][2], xyPr[4][2], minY = 1e6, maxY = -1e6, startY, endY, yStep,
                               intX, volFr = 0, sigma, thisPos, delX;
  int inSide = 0, nrYs = 500, i, j, splCase = 0;
  double omePr, etaPr, eta;
  sigma = beamFWHM / (2 * sqrt(2 * log(2)));
  // Convert from Omega to Eta (look in the computation notebook, 08/19/19 pg.
  // 34 for calculation)
  if (Omega < 0)
    omePr = 360 + Omega;
  else
    omePr = Omega;
  if (abs(omePr) < 1e-5)
    splCase = 1;
  else if (abs(omePr - 90) < 1e-5)
    splCase = 1;
  else if (abs(omePr - 180) < 1e-5)
    splCase = 1;
  else if (abs(omePr - 270) < 1e-5)
    splCase = 1;
  else if (abs(omePr - 360) < 1e-5)
    splCase = 1;
  else {
    if (omePr < 90)
      etaPr = 90 - omePr;
    else if (omePr < 180)
      etaPr = 180 - omePr;
    else if (omePr < 270)
      etaPr = 270 - omePr;
    else
      etaPr = 360 - omePr;
    if (etaPr < 45)
      eta = 90 - etaPr;
    else
      eta = etaPr;
  }
  // What we need: minY, maxY
  for (i = 0; i < 4; i++) {
    xy[i][0] = voxelPosition[0] + dx[i] * voxLen;
    xy[i][1] = voxelPosition[1] + dy[i] * voxLen;
    xyPr[i][1] = xy[i][0] * sind(Omega) + xy[i][1] * cosd(Omega);
    if (xyPr[i][1] < minY) {
      minY = xyPr[i][1];
    }
    if (xyPr[i][1] > maxY) {
      maxY = xyPr[i][1];
    }
  }
  if (maxY >= beamPosition - 2 * beamFWHM &&
      minY <= beamPosition + 2 * beamFWHM)
    inSide = 1;
  if (inSide == 1) {
    startY = (minY > beamPosition - 2 * beamFWHM) ? minY
                                                  : beamPosition - 2 * beamFWHM;
    endY = (maxY < beamPosition + 2 * beamFWHM) ? maxY
                                                : beamPosition + 2 * beamFWHM;
    yStep = (endY - startY) / ((double)nrYs);
    for (i = 0; i <= nrYs; i++) {
      if (splCase == 1)
        delX = 1;
      else {
        thisPos = i * yStep;
        if (thisPos < voxLen * cosd(eta))
          delX = thisPos * (tand(eta) + (1 / tand(eta)));
        else if (maxY - minY - thisPos < voxLen * cosd(eta))
          delX = (maxY - minY - thisPos) * (tand(eta) + (1 / tand(eta)));
        else
          delX = voxLen * (sind(eta) + (cosd(eta) / tand(eta)));
      }
      thisPos = startY + i * yStep;
      intX = yStep *
             exp(-((thisPos - beamPosition) * (thisPos - beamPosition)) /
                 (2 * sigma * sigma)) /
             (sigma * sqrt(2 * M_PI));
      volFr += intX * delX;
    }
  }
  return volFr;
}

static inline void SpotToGv(double xi, double yi, double zi, double Omega,
                            double theta, double *g1, double *g2, double *g3) {
  double CosOme = cosd(Omega), SinOme = sind(Omega);
  double eta;
  CalcEtaAngle(yi, zi, &eta);
  double TanEta = tand(-eta), SinTheta = sind(theta);
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

// Function to calculate y,z,ome of diffraction spots, given euler angles
// (radians), position and lattice parameter. Ideal hkls need to be provided,
// with 4 columns: h,k,l,ringNr Output for comparisonType 0:
// g1,g2,g3,eta,omega,y,z,2theta,nrhkls
static inline long CalcDiffractionSpots(double Lsd, double Wavelength,
                                        double position[3], double LatC[6],
                                        double EulerAngles[3], int nhkls,
                                        double *hklsIn, double *spotPos,
                                        int comparisonType) {
  double *hkls; // We need h,k,l,theta,ringNr
  hkls = calloc(nhkls * 5, sizeof(*hkls));
  CorrectHKLsLatC(LatC, hklsIn, nhkls, Lsd, Wavelength, hkls);
  double Gc[3], Ghkl[3], cosOme, sinOme, yspot, zspot;
  double omegas[4], etas[4], lenK, xs, ys, zs, th, g1, g2, g3;
  double yspots, zspots, xGr = position[0], yGr = position[1],
                         zGr = position[2], xRot, yRot, yPrr, zPrr;
  double OM[3][3], theta, RingRadius, omega, eta, etanew, nrhkls;
  int hklnr, nspotsPlane, i;
  Euler2OrientMat(EulerAngles, OM);
  int spotNr = 0;
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    Ghkl[0] = hkls[hklnr * 5 + 0];
    Ghkl[1] = hkls[hklnr * 5 + 1];
    Ghkl[2] = hkls[hklnr * 5 + 2];
    MatrixMult(OM, Ghkl, Gc);
    theta = hkls[hklnr * 5 + 3];
    CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
    nrhkls = (double)hklnr * 2 + 1;
    for (i = 0; i < nspotsPlane; i++) {
      omega = omegas[i];
      eta = etas[i];
      if (isnan(omega) || isnan(eta))
        continue;
      cosOme = cosd(omega);
      sinOme = sind(omega);
      xRot = xGr * cosOme - yGr * sinOme;
      yRot = xGr * sinOme + yGr * cosOme;
      RingRadius = tand(2 * theta) * (Lsd + xRot);
      yPrr = -(sind(eta) * RingRadius);
      zPrr = cosd(eta) * RingRadius;
      yspot = yPrr + yRot;
      zspot = zPrr + zGr;
      RingRadius = sqrt(yspot * yspot + zspot * zspot);
      CalcEtaAngle(yspot, zspot, &etanew);
      th = atand(RingRadius / Lsd) / 2;
      // Depending on comparisonType:
      // 1. Gvector:
      switch (comparisonType) {
      case 1:
        xs = Lsd;
        ys = yspot;
        zs = zspot;
        lenK = CalcNorm3(xs, ys, zs);
        SpotToGv(xs / lenK, ys / lenK, zs / lenK, omega, th, &g1, &g2, &g3);
        spotPos[spotNr * 9 + 0] = g1;
        spotPos[spotNr * 9 + 1] = g2;
        spotPos[spotNr * 9 + 2] = g3;
        spotPos[spotNr * 9 + 3] = etanew;
        spotPos[spotNr * 9 + 4] = omega;
        spotPos[spotNr * 9 + 5] = hkls[hklnr * 5 + 4]; // ringNr
        spotPos[spotNr * 9 + 6] = nrhkls;
        spotPos[spotNr * 9 + 7] = ys;
        spotPos[spotNr * 9 + 8] = zs;
        break;
      case 3:
        spotPos[spotNr * 4 + 0] = yspot;
        spotPos[spotNr * 4 + 1] = zspot;
        spotPos[spotNr * 4 + 2] = omega;
        spotPos[spotNr * 4 + 3] = nrhkls;
        break;
      default:
        spotNr = -1;
        nrhkls = -1;
        break;
      }
      nrhkls++;
      spotNr++;
    }
  }
  free(hkls);
  return spotNr;
}

// An assumption here that fraction of voxel in the beam does not change during
// optimization
static inline void PopulateMatrices(
    double omegaStep, double px, int nVoxels, double *voxelList,
    double voxelLen, double beamFWHM, int nBeamPositions, double *beamPositions,
    double omeTol, int nRings, double Euler[3], double LatC[6], int nhkls,
    double *hkls, double Lsd, double Wavelength, long totalNrSpots,
    long *AllIDsInfo, double *AllSpotsInfo, int maxNPos, long *FLUT,
    double *Fthis, double *spotInfoMat, double *filteredSpotInfo) {
  double *spotInfo, pos0[3] = {0, 0, 0};
  long nSpots;
  spotInfo = calloc(nhkls * 2 * 9, sizeof(*spotInfo));
  nSpots = CalcDiffractionSpots(Lsd, Wavelength, pos0, LatC, Euler, nhkls, hkls,
                                spotInfo, 1);
// Let's do the next part in parallel
#pragma omp parallel num_threads(numProcs)
  {
    long i, voxelNr, spotNr, positionNr, startRowNr, endRowNr, bestHKLNr,
        bestRow, idxPos;
    double etaTol = 1.0, voxelFraction; // EtaTol is assumed to be 1.
    double thisOmega, thisEta, omeObs, etaObs, ys, zs, lenK, IA, bestAngle,
        gSim[3], gObs[3], thisBeamPosition, thisPos[3];
    int ringNr;
    long startSpotNr, endSpotNr;
    long nrSpotsThread = (long)ceil((double)nSpots / (double)numProcs);
    long procNr = omp_get_thread_num();
    startSpotNr = procNr * nrSpotsThread;
    endSpotNr = (startSpotNr + nrSpotsThread > nSpots)
                    ? nSpots
                    : startSpotNr + nrSpotsThread;
    for (spotNr = startSpotNr; spotNr < endSpotNr; spotNr++) {
      thisOmega = spotInfo[spotNr * 9 + 4];
      thisEta = spotInfo[spotNr * 9 + 3];
      ringNr = (int)spotInfo[spotNr * 9 + 5];
      gSim[0] = spotInfo[spotNr * 9 + 0];
      gSim[1] = spotInfo[spotNr * 9 + 1];
      gSim[2] = spotInfo[spotNr * 9 + 2];
      bestHKLNr = (long)spotInfo[spotNr * 9 + 6];
      for (positionNr = 0; positionNr < nBeamPositions; positionNr++) {
        thisBeamPosition = beamPositions[positionNr];
        startRowNr = AllIDsInfo[(positionNr * nRings + ringNr) * 2 + 0];
        endRowNr = AllIDsInfo[(positionNr * nRings + ringNr) * 2 + 1];
        if (startRowNr == 0)
          continue;
        bestAngle = 1e10;
        for (i = startRowNr; i <= endRowNr; i++) {
          omeObs = AllSpotsInfo[14 * (i - 1) + 2];
          etaObs = AllSpotsInfo[14 * (i - 1) + 6];
          if (fabs(thisOmega - omeObs) < omeTol &&
              fabs(thisEta - etaObs) < etaTol) {
            ys = AllSpotsInfo[14 * (i - 1) + 0];
            zs = AllSpotsInfo[14 * (i - 1) + 1];
            lenK = CalcNorm3(Lsd, ys, zs);
            SpotToGv(Lsd / lenK, ys / lenK, zs / lenK, omeObs,
                     AllSpotsInfo[14 * (i - 1) + 7] / 2, &gObs[0], &gObs[1],
                     &gObs[2]);
            IA = fabs(acosd(
                (gSim[0] * gObs[0] + gSim[1] * gObs[1] + gSim[2] * gObs[2]) /
                (CalcNorm3(gSim[0], gSim[1], gSim[2]) *
                 CalcNorm3(gObs[0], gObs[1], gObs[2]))));
            if (IA < bestAngle) {
              bestAngle = IA;
              bestRow = i;
            }
          }
        }
        if (bestAngle < 1) {
          for (voxelNr = 0; voxelNr < nVoxels; voxelNr++) {
            thisPos[0] = voxelList[voxelNr * 2 + 0];
            thisPos[1] = voxelList[voxelNr * 2 + 1];
            thisPos[2] = 0;
            voxelFraction = IntensityFraction(voxelLen, thisBeamPosition,
                                              beamFWHM, thisPos, thisOmega);
            if (voxelFraction > 0) { // We now know that this voxel is in the
                                     // beam, let's populate our matrix
              idxPos = voxelNr;
              idxPos *= nhkls + 2;
              idxPos *= 2;
              idxPos *= maxNPos;
              idxPos += bestHKLNr * maxNPos;
#pragma omp critical
              {
                while (FLUT[idxPos] >=
                       0) { // Check if we already filled this up, wait until we
                            // find the one not filled up
                  idxPos++;
                }
                FLUT[idxPos] = bestRow - 1;
                idxPos *= 5;
                Fthis[idxPos + 0] = spotInfo[spotNr * 9 + 7];
                Fthis[idxPos + 1] = spotInfo[spotNr * 9 + 8];
                Fthis[idxPos + 2] = spotInfo[spotNr * 9 + 4];
                Fthis[idxPos + 3] = voxelFraction;
                Fthis[idxPos + 4] = positionNr;
                // We also need to fill in spotInfoMat and filteredSpotInfo
                if (filteredSpotInfo[(bestRow - 1) * 4 + 3] == 0) {
                  filteredSpotInfo[(bestRow - 1) * 4 + 0] =
                      AllSpotsInfo[14 * (bestRow - 1) + 0];
                  filteredSpotInfo[(bestRow - 1) * 4 + 1] =
                      AllSpotsInfo[14 * (bestRow - 1) + 1];
                  filteredSpotInfo[(bestRow - 1) * 4 + 2] =
                      AllSpotsInfo[14 * (bestRow - 1) + 2];
                  filteredSpotInfo[(bestRow - 1) * 4 + 3] = 1;
                }
                spotInfoMat[(bestRow - 1) * 4 + 0] +=
                    spotInfo[spotNr * 9 + 7] * voxelFraction;
                spotInfoMat[(bestRow - 1) * 4 + 1] +=
                    spotInfo[spotNr * 9 + 8] * voxelFraction;
                spotInfoMat[(bestRow - 1) * 4 + 2] +=
                    spotInfo[spotNr * 9 + 4] * voxelFraction;
                spotInfoMat[(bestRow - 1) * 4 + 3] += voxelFraction;
              }
            }
          }
        }
      }
    }
  }
  long i;
  for (i = 0; i < totalNrSpots; i++) {
    if (filteredSpotInfo[i * 4 + 3] == 0)
      continue;
    spotInfoMat[i * 4 + 0] /= spotInfoMat[i * 4 + 3];
    spotInfoMat[i * 4 + 1] /= spotInfoMat[i * 4 + 3];
    spotInfoMat[i * 4 + 2] /= spotInfoMat[i * 4 + 3];
  }
  free(spotInfo);
}

static inline double CalcDifferences(double omegaStep, double px,
                                     long totalNrSpots, double *spotInfoMat,
                                     double *filteredSpotInfo,
                                     double *differencesMat) {
  long i;
  double diff = 0;
  double normParams[4], EtaObs, EtaSim;
  normParams[0] = 0.1 * px;
  normParams[1] = 0.1 * px;
  for (i = 0; i < totalNrSpots; i++) {
    if (filteredSpotInfo[i * 4 + 3] == 0)
      continue;
    EtaSim = CalcEta(spotInfoMat[i * 4 + 0], spotInfoMat[i * 4 + 1]);
    normParams[2] = omegaStep * 0.5 * (1 + 1 / sind(EtaSim));
    if (FitType == 0) {
      differencesMat[i] =
          CalcNorm3((spotInfoMat[i * 4 + 0] - filteredSpotInfo[i * 4 + 0]) /
                        normParams[0],
                    (spotInfoMat[i * 4 + 1] - filteredSpotInfo[i * 4 + 1]) /
                        normParams[1],
                    (spotInfoMat[i * 4 + 2] - filteredSpotInfo[i * 4 + 2]) /
                        normParams[2]);
    } else if (FitType == 1) { // Only Orientation Fit, we will use eta and
                               // omega difference
      normParams[3] = atand(
          sqrt(0.02) * px /
          CalcNorm2(filteredSpotInfo[i * 4 + 0], filteredSpotInfo[i * 4 + 1]));
      EtaObs =
          CalcEta(filteredSpotInfo[i * 4 + 0], filteredSpotInfo[i * 4 + 1]);
      differencesMat[i] =
          CalcNorm2((spotInfoMat[i * 4 + 2] - filteredSpotInfo[i * 4 + 2]) /
                        normParams[2],
                    (EtaSim - EtaObs));
    } else if (FitType == 2) { // Only lattice parameter fit, we will use
                               // difference in 2theta or ring radius
      normParams[3] = sqrt(0.02) * px;
      differencesMat[i] =
          CalcNorm2(filteredSpotInfo[i * 4 + 0], filteredSpotInfo[i * 4 + 1]) /
          normParams[3];
    }
    diff += differencesMat[i];
  }
  return diff;
}

static inline void
UpdSpotPosOneVox(double omegaStep, double px, double voxelLen, double beamFWHM,
                 int nBeamPositions, double *beamPositions, double omeTol,
                 double *EulLatC, int nRings, int nhkls, double *hkls,
                 double Lsd, double Wavelength, double voxelPos[3],
                 double *arrUpd, long *FLUTThis, long maxNPos,
                 double *spotInfoMat, double *spotInfo, double *AllSpotsInfo,
                 long *AllIDsInfo) {
  long i, nSpots, positionNr, bestHKLNr, idxPos, spotNr, spotRowNr, posNr,
      startRowNr, endRowNr, tmpPos;
  double LatCThis[6], EulerThis[3], thisBeamPos;
  double gSim[3], gObs[3], IA, bestAngle, omeObs, ys, zs, lenK;
  int ringNr;
  for (i = 0; i < 3; i++)
    EulerThis[i] = EulLatC[i];
  for (i = 0; i < 6; i++)
    LatCThis[i] = EulLatC[i + 3];
  nSpots = CalcDiffractionSpots(Lsd, Wavelength, voxelPos, LatCThis, EulerThis,
                                nhkls, hkls, spotInfo, 3);
  for (spotNr = 0; spotNr < nSpots; spotNr++) {
    bestHKLNr = (long)spotInfo[spotNr * 4 + 3];
    for (i = 0; i < maxNPos; i++) {
      if (FLUTThis[bestHKLNr * maxNPos + i] >= 0) {
        idxPos = (bestHKLNr * maxNPos + i) * 5;
        positionNr = (long)arrUpd[idxPos + 4];
        thisBeamPos =
            beamPositions[positionNr]; // Include here update voxelFraction with
                                       // 4x voxelFraction!!
        spotRowNr = FLUTThis[bestHKLNr * maxNPos + i];
#pragma omp critical
        {
          spotInfoMat[spotRowNr * 4 + 0] +=
              ((spotInfo[spotNr * 4 + 0] - arrUpd[idxPos + 0]) *
               arrUpd[idxPos + 3]) /
              spotInfoMat[spotRowNr * 4 + 3];
          spotInfoMat[spotRowNr * 4 + 1] +=
              ((spotInfo[spotNr * 4 + 1] - arrUpd[idxPos + 1]) *
               arrUpd[idxPos + 3]) /
              spotInfoMat[spotRowNr * 4 + 3];
          spotInfoMat[spotRowNr * 4 + 2] +=
              ((spotInfo[spotNr * 4 + 2] - arrUpd[idxPos + 2]) *
               arrUpd[idxPos + 3]) /
              spotInfoMat[spotRowNr * 4 + 3];
        }
        arrUpd[idxPos + 0] = spotInfo[spotNr * 4 + 0];
        arrUpd[idxPos + 1] = spotInfo[spotNr * 4 + 1];
        arrUpd[idxPos + 2] = spotInfo[spotNr * 4 + 2];
        arrUpd[idxPos + 4] = positionNr;
      } else {
        break;
      }
    }
  }
}

static inline double
UpdateArraysThis(double omegaStep, double px, int nVoxels, double *voxelList,
                 double voxelLen, double *x, double beamFWHM,
                 int nBeamPositions, double *beamPositions, double omeTol,
                 int nRings, int nhkls, double *hkls, double Lsd,
                 double Wavelength, double *Fthis, long *FLUT, long maxNPos,
                 long totalNrSpots, double *spotInfoMat, double *AllSpotsInfo,
                 long *AllIDsInfo, double *filteredSpotInfo,
                 double *spotInfoAll, double *differencesMat) {
#pragma omp parallel num_threads(numProcs)
  {
    long voxelNr;
    long procNr = omp_get_thread_num();
    long nrVoxelsThread = (long)ceil((double)nVoxels / (double)numProcs);
    long startVoxNr = procNr * nrVoxelsThread;
    long endVoxNr = (startVoxNr + nrVoxelsThread > nVoxels)
                        ? nVoxels
                        : startVoxNr + nrVoxelsThread;
    double *spotInfo;
    long spotInfoPos;
    spotInfoPos = procNr;
    spotInfoPos *= nhkls * 2;
    spotInfoPos *= 4;
    spotInfo = &spotInfoAll[spotInfoPos];
    for (voxelNr = startVoxNr; voxelNr < endVoxNr; voxelNr++) {
      double thisParams[9];
      thisParams[0] = x[voxelNr * 9 + 0];
      thisParams[1] = x[voxelNr * 9 + 1];
      thisParams[2] = x[voxelNr * 9 + 2];
      thisParams[3] = x[voxelNr * 9 + 3];
      thisParams[4] = x[voxelNr * 9 + 4];
      thisParams[5] = x[voxelNr * 9 + 5];
      thisParams[6] = x[voxelNr * 9 + 6];
      thisParams[7] = x[voxelNr * 9 + 7];
      thisParams[8] = x[voxelNr * 9 + 8];
      double voxelPos[3];
      voxelPos[0] = voxelList[voxelNr * 2 + 0];
      voxelPos[1] = voxelList[voxelNr * 2 + 1];
      voxelPos[2] = 0;
      long posFthis, posFLUT, *FLUTVoxel;
      posFthis = voxelNr;
      posFthis *= nhkls + 2;
      posFthis *= 2;
      posFthis *= maxNPos;
      posFLUT = posFthis;
      posFthis *= 5;
      double *FthisVoxel;
      FthisVoxel = &Fthis[posFthis];
      FLUTVoxel = &FLUT[posFLUT];
      UpdSpotPosOneVox(omegaStep, px, voxelLen, beamFWHM, nBeamPositions,
                       beamPositions, omeTol, thisParams, nRings, nhkls, hkls,
                       Lsd, Wavelength, voxelPos, FthisVoxel, FLUTVoxel,
                       maxNPos, spotInfoMat, spotInfo, AllSpotsInfo,
                       AllIDsInfo);
    }
  }
  double diffFThis = CalcDifferences(omegaStep, px, totalNrSpots, spotInfoMat,
                                     filteredSpotInfo, differencesMat);
  //~ printf("Error now: %.12lf\n",diffFThis);
  fflush(stdout);
  return diffFThis;
}

struct FITTING_PARAMS {
  double omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength;
  double *voxelList, *beamPositions, *hkls, *Fthis, *spotInfoMat, *AllSpotsInfo,
      *filteredSpotInfo, *spotInfoAll, *differencesMat;
  int nBeamPositions, nhkls, nRings, nConn;
  int *Connections;
  long *FLUT, *AllIDsInfo;
  long maxNPos, totalNrSpots;
};

static double problem_function(unsigned n, const double *x,
                               void *f_data_trial) {
  time_t current_time;
  char *c_time_string;
  current_time = time(NULL);
  c_time_string = ctime(&current_time);
  //~ printf("Current time is %s", c_time_string);
  int nVoxels = n / 9;
  struct FITTING_PARAMS *f_data = (struct FITTING_PARAMS *)f_data_trial;
  double omegaStep = f_data->omegaStep, px = f_data->px,
         voxelLen = f_data->voxelLen, beamFWHM = f_data->beamFWHM,
         omeTol = f_data->omeTol;
  double Lsd = f_data->Lsd, Wavelength = f_data->Wavelength;
  double *voxelList = &(f_data->voxelList[0]),
         *beamPositions = &(f_data->beamPositions[0]),
         *hkls = &(f_data->hkls[0]);
  double *Fthis = &(f_data->Fthis[0]), *spotInfoMat = &(f_data->spotInfoMat[0]),
         *filteredSpotInfo = &(f_data->filteredSpotInfo[0]);
  double *spotInfoAll = &(f_data->spotInfoAll[0]);
  double *AllSpotsInfo = &(f_data->AllSpotsInfo[0]),
         *differencesMat = &(f_data->differencesMat[0]);
  int nBeamPositions = f_data->nBeamPositions, nhkls = f_data->nhkls,
      nRings = f_data->nRings;
  long *AllIDsInfo = &(f_data->AllIDsInfo[0]), *FLUT = &(f_data->FLUT[0]);
  long maxNPos = f_data->maxNPos, totalNrSpots = f_data->totalNrSpots;
  double err;
  err = UpdateArraysThis(
      omegaStep, px, nVoxels, voxelList, voxelLen, x, beamFWHM, nBeamPositions,
      beamPositions, omeTol, nRings, nhkls, hkls, Lsd, Wavelength, Fthis, FLUT,
      maxNPos, totalNrSpots, spotInfoMat, AllSpotsInfo, AllIDsInfo,
      filteredSpotInfo, spotInfoAll, differencesMat);
  return err;
}

// TODO: Make a connectivity function, implement constraints.
// Maximum difference in euler angles: 0.1 degrees, max change in lattice
// parameter 0.0001 fraction

static int conn(double *voxelList, double voxelLen, int nVoxels,
                int *Connections) {
  // How to find connections (8-connected): we have a list of voxels (with x,y
  // positions) and we have the voxel length
  int i, j;
  int nConn = 0;
  double px1[2], px2[2];
  for (i = 0; i < nVoxels; i++) {
    px1[0] = voxelList[i * 2 + 0];
    px1[1] = voxelList[i * 2 + 1];
    for (j = i + 1; j < nVoxels; j++) {
      px2[0] = voxelList[j * 2 + 0];
      px2[1] = voxelList[j * 2 + 1];
      if (CalcNorm2(px1[0] - px2[0], px1[0] - px2[0]) <
          sqrt(2) * voxelLen +
              1) { // we use 1 micron extra for rounding off errors
        Connections[nConn * 2 + 0] = i;
        Connections[nConn * 2 + 1] = j;
        nConn++;
      }
    }
  }
  return nConn;
}

// nlopt_add_inequality_mconstraint(opt,unsigned m,nlopt_mfunc c,void*
// c_data,const double* tol); we need to make a function void c(unsigned m,
// double *result, unsigned n, const double* x, double* grad, void* f_data); ci
// <=0 is the constraint, grad of constraint (m*n in size) is always 0 (since
// constraints are linear in this case) m is the number of connectivity
// parameters, c_data is the orientation/latticeparameter variation allowed, tol
// is a pointer to m values <1e-8 (not 0 here for stopping criteria) f_data will
// still be supplied but is not used Our constraint will be orient difference
// between neighbors < a, latticeparameter difference between neighbors < b a
// and b will be supplied in the c_data pointer

void populate_arrays(char *paramFN) {
  FILE *fileParam;
  //~ printf("%s\n",paramFN);
  fileParam = fopen(paramFN, "r");
  double omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength;
  int nScans, rings[500], nRings = 0;
  int GrainNr = 1;
  char aline[4096], dummy[4096];
  char positionsFN[4096];
  char grainFN[4096];
  char idsfn[4096];
  char hklfn[4096];
  char spotInfoFN[4096];
  sprintf(hklfn, "hkls.csv");
  sprintf(grainFN, "Grains.csv");
  sprintf(idsfn, "IDsHash.csv");
  sprintf(spotInfoFN, "ExtraInfo.bin");
  char voxelsFN[4096];
  double EulTol = 3 * deg2rad;
  double ABCTol = 3;
  double ABGTol = 3;
  double LatCin[6];
  int optType = 0, fitT = 0;
  while (fgets(aline, 4096, fileParam) != NULL) {
    if (strncmp(aline, "HKLFile", strlen("HKLFile")) == 0) {
      sscanf(aline, "%s %s", dummy, hklfn);
    }
    if (strncmp(aline, "OptType", strlen("OptType")) == 0) {
      sscanf(aline, "%s %d", dummy, &optType);
    }
    if (strncmp(aline, "FitBoth", strlen("FitBoth")) == 0) {
      sscanf(aline, "%s %d", dummy, &fitT);
    }
    if (strncmp(aline, "GrainsFile", strlen("GrainsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, grainFN);
    }
    if (strncmp(aline, "IDsFile", strlen("IDsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, idsfn);
    }
    if (strncmp(aline, "SpotsFile", strlen("SpotsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, spotInfoFN);
    }
    if (strncmp(aline, "OmegaStep", strlen("OmegaStep")) == 0) {
      sscanf(aline, "%s %lf", dummy, &omegaStep);
      omegaStep = fabs(omegaStep);
    }
    if (strncmp(aline, "LatticeConstant", strlen("LatticeConstant")) == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
    }
    if (strncmp(aline, "LatticeParameter", strlen("LatticeParameter")) == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
    }
    if (strncmp(aline, "px", strlen("px")) == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
    }
    if (strncmp(aline, "VoxelLength", strlen("VoxelLength")) == 0) {
      sscanf(aline, "%s %lf", dummy, &voxelLen);
    }
    if (strncmp(aline, "BeamFWHM", strlen("BeamFWHM")) == 0) {
      sscanf(aline, "%s %lf", dummy, &beamFWHM);
    }
    if (strncmp(aline, "OmegaTol", strlen("OmegaTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &omeTol);
    }
    if (strncmp(aline, "Lsd", strlen("Lsd")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    }
    if (strncmp(aline, "Wavelength", strlen("Wavelength")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
    }
    if (strncmp(aline, "OrientTol", strlen("OrientTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &EulTol);
      EulTol *= deg2rad;
    }
    if (strncmp(aline, "ABCTol", strlen("ABCTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &ABCTol);
    }
    if (strncmp(aline, "ABGTol", strlen("ABGTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &ABGTol);
    }
    if (strncmp(aline, "nLayers", strlen("nLayers")) == 0) {
      sscanf(aline, "%s %d", dummy, &nScans);
    }
    if (strncmp(aline, "GrainNr", strlen("GrainNr")) == 0) {
      sscanf(aline, "%s %d", dummy, &GrainNr);
    }
    if (strncmp(aline, "NProcs", strlen("NProcs")) == 0) {
      sscanf(aline, "%s %d", dummy, &numProcs);
    }
    if (strncmp(aline, "PositionsFile", strlen("PositionsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, positionsFN);
    }
    if (strncmp(aline, "VoxelsFile", strlen("VoxelsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, voxelsFN);
    }
    if (strncmp(aline, "RingThresh", strlen("RingThresh")) == 0) {
      sscanf(aline, "%s %d", dummy, &rings[nRings]);
      nRings++;
    }
  }
  fclose(fileParam);
  if ((ABCTol < 0.00001 || ABGTol < 0.00001) && fitT == 0) {
    FitType = 1;
  } else if ((EulTol < 0.00001) && fitT == 0) {
    FitType = 2;
  }

  long i, j, k;
  FILE *positionsFile;
  positionsFile = fopen(positionsFN, "r");
  int nBeamPositions = nScans;
  double *beamPositions;
  beamPositions = calloc(nBeamPositions, sizeof(*beamPositions));
  fgets(aline, 4096, positionsFile);
  for (i = 0; i < nBeamPositions; i++) {
    fgets(aline, 4096, positionsFile);
    sscanf(aline, "%lf", &beamPositions[i]);
    beamPositions[i] *= -1000;
  }
  fclose(positionsFile);

  FILE *hklf;
  hklf = fopen(hklfn, "r");
  double ht, kt, lt, ringT;
  double *hklTs;
  hklTs = calloc(500 * 4, sizeof(*hklTs));
  int nhkls = 0;
  fgets(aline, 4096, hklf);
  while (fgets(aline, 4096, hklf) != NULL) {
    sscanf(aline, "%lf %lf %lf %s %lf", &ht, &kt, &lt, dummy, &ringT);
    for (i = 0; i < nRings; i++) {
      if ((int)ringT == rings[i]) {
        hklTs[nhkls * 4 + 0] = ht;
        hklTs[nhkls * 4 + 1] = kt;
        hklTs[nhkls * 4 + 2] = lt;
        hklTs[nhkls * 4 + 3] = ringT;
        nhkls++;
      }
    }
  }
  fclose(hklf);
  double *hkls;
  hkls = calloc(nhkls * 4, sizeof(*hkls));
  for (i = 0; i < nhkls * 4; i++)
    hkls[i] = hklTs[i];
  nRings = (int)hkls[nhkls * 4 - 1];
  free(hklTs);

  FILE *voxelsFile;
  voxelsFile = fopen(voxelsFN, "r");
  double *voxelsT;
  voxelsT = calloc(nBeamPositions * nBeamPositions, sizeof(*voxelsT));
  int nVoxels = 0;
  while (fgets(aline, 4096, voxelsFile) != NULL) {
    sscanf(aline, "%lf,%lf", &voxelsT[nVoxels * 2 + 0],
           &voxelsT[nVoxels * 2 + 1]);
    nVoxels++;
  }
  double *voxelList;
  voxelList = calloc(nVoxels * 2, sizeof(*voxelList));
  for (i = 0; i < nVoxels * 2; i++)
    voxelList[i] = voxelsT[i];
  free(voxelsT);

  char cpCommand[4096];
  sprintf(cpCommand, "cp %s /dev/shm/ExtraInfo.bin", spotInfoFN);
  system(cpCommand);
  const char *filename = "/dev/shm/ExtraInfo.bin";
  int rc;
  double *AllSpotsInfo;
  struct stat s;
  size_t size;
  int fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  int status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  AllSpotsInfo = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(AllSpotsInfo == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  long totalNrSpots = size / (14 * sizeof(double));

  long *AllIDsInfo;
  AllIDsInfo = calloc(nBeamPositions * nRings * 2, sizeof(*AllIDsInfo));
  FILE *idsfile;
  idsfile = fopen(idsfn, "r");
  int positionNr, startNr, endNr, ringNr;
  fgets(aline, 4096, idsfile);
  while (fgets(aline, 4096, idsfile) != NULL) {
    sscanf(aline, "%d %d %d %d", &positionNr, &ringNr, &startNr, &endNr);
    if (positionNr == 0)
      continue;
    AllIDsInfo[((positionNr - 1) * nRings + ringNr) * 2 + 0] = startNr;
    AllIDsInfo[((positionNr - 1) * nRings + ringNr) * 2 + 1] = endNr;
  }
  fclose(idsfile);

  FILE *grainsFile;
  grainsFile = fopen(grainFN, "r");
  char line[20000];
  for (i = 0; i < (9 + GrainNr); i++)
    fgets(line, 20000, grainsFile);
  double OM[3][3], LatC[6];
  sscanf(
      line,
      "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %lf %lf %lf %lf %lf %lf",
      dummy, &OM[0][0], &OM[0][1], &OM[0][2], &OM[1][0], &OM[1][1], &OM[1][2],
      &OM[2][0], &OM[2][1], &OM[2][2], dummy, dummy, dummy, &LatC[0], &LatC[1],
      &LatC[2], &LatC[3], &LatC[4], &LatC[5]);
  double Eul[3];
  OrientMat2Euler(OM, Eul);

  int n = nVoxels * 9;
  double *x;
  x = calloc(n, sizeof(*x));
  for (i = 0; i < nVoxels; i++) {
    for (j = 0; j < 3; j++)
      x[i * 9 + j] = Eul[j];
    for (j = 0; j < 6; j++)
      x[i * 9 + 3 + j] = LatCin[j];
  }
  FILE *xF;
  xF = fopen("/dev/shm/x.bin", "w");
  fwrite(x, n * sizeof(*x), 1, xF);
  fclose(xF);

  int maxNPos = 2 * (2 + ceil(4 * beamFWHM / voxelLen));
  size_t dataArrSize;
  double *Fthis;
  dataArrSize = nVoxels;
  dataArrSize *= nhkls + 2;
  dataArrSize *= 2;
  dataArrSize *= maxNPos;
  dataArrSize *= 5;
  Fthis = calloc(dataArrSize, sizeof(*Fthis));

  size_t sizeSpotInfoMat;
  sizeSpotInfoMat = 4;
  sizeSpotInfoMat *= totalNrSpots;
  double *spotInfoMat, *filteredSpotInfo, *differencesMat;
  spotInfoMat = calloc(sizeSpotInfoMat, sizeof(*spotInfoMat));
  filteredSpotInfo = calloc(sizeSpotInfoMat, sizeof(*filteredSpotInfo));
  differencesMat = calloc(totalNrSpots, sizeof(*differencesMat));

  size_t sizeFLUT;
  sizeFLUT = nVoxels;
  sizeFLUT *= nhkls + 2;
  sizeFLUT *= 2;
  sizeFLUT *= maxNPos;
  long *FLUT;
  FLUT = calloc(sizeFLUT, sizeof(*FLUT));
  for (i = 0; i < sizeFLUT; i++)
    FLUT[i] = -1;

  double *spotInfoAll;
  long lenSpotInfoAll;
  lenSpotInfoAll = numProcs;
  lenSpotInfoAll *= nhkls * 2;
  lenSpotInfoAll *= 4;
  spotInfoAll = calloc(lenSpotInfoAll, sizeof(*spotInfoAll));

  // Make connections
  /*int maxNConnections = nVoxels*8;
  int *Connections;
  Connections = calloc(2*maxNConnections,sizeof(*Connections));
  int nConn = conn(voxelList, voxelLen, nVoxels, Connections);
  */
  int nConn = 2 * 8 * nVoxels;

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
  time_t current_time;
  char *c_time_string;
  current_time = time(NULL);
  c_time_string = ctime(&current_time);
  //~ printf("Current time is %s", c_time_string);
  printf("Populating matrices.\n");
  PopulateMatrices(omegaStep, px, nVoxels, voxelList, voxelLen, beamFWHM,
                   nBeamPositions, beamPositions, omeTol, nRings, Eul, LatCin,
                   nhkls, hkls, Lsd, Wavelength, totalNrSpots, AllIDsInfo,
                   AllSpotsInfo, maxNPos, FLUT, Fthis, spotInfoMat,
                   filteredSpotInfo);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
  size_t saveParams[8] = {nVoxels,     nhkls,    nConn,        maxNPos,
                          dataArrSize, sizeFLUT, totalNrSpots, sizeSpotInfoMat};
  FILE *paramsT = fopen("/dev/shm/scanningParams.txt", "w");
  fprintf(paramsT, "%s\n", paramFN);
  for (i = 0; i < 8; i++)
    fprintf(paramsT, "%zu\n", saveParams[i]);
  fprintf(paramsT, "\n");
  fclose(paramsT);
  //~ FILE *connF = fopen("/dev/shm/Connections.bin","wb");
  //~ fwrite(Connections,nConn*2*sizeof(*Connections),1,connF);
  //~ fclose(connF);
  FILE *fthisF = fopen("/dev/shm/Fthis.bin", "wb");
  fwrite(Fthis, dataArrSize * sizeof(*Fthis), 1, fthisF);
  fclose(fthisF);
  FILE *flutF = fopen("/dev/shm/FLUT.bin", "wb");
  fwrite(FLUT, sizeFLUT * sizeof(*FLUT), 1, flutF);
  fclose(flutF);
  FILE *spF = fopen("/dev/shm/spotInfoMat.bin", "wb");
  fwrite(spotInfoMat, sizeSpotInfoMat * sizeof(*spotInfoMat), 1, spF);
  fclose(spF);
  FILE *filtF = fopen("/dev/shm/filteredSpotInfo.bin", "wb");
  fwrite(filteredSpotInfo, sizeSpotInfoMat * sizeof(*filteredSpotInfo), 1,
         filtF);
  fclose(filtF);
  // What needs to be saved here:
  // 1. nVoxels, nhkls, nConn, maxNPos, dataArrSize, sizeFLUT, totalNrSpots.
  // 2. Arrays: Connections, Fthis, FLUT, spotInfoMat, filteredSpotInfo.
  double t_ns = (double)(end.tv_sec - start.tv_sec) * 1.0e9 +
                (double)(end.tv_nsec - start.tv_nsec);
}

double evaluateF(double *x) {
  // What needs to be read here:
  // 1. nVoxels, nhkls, nConn, maxNPos, dataArrSize, sizeFLUT, totalNrSpots.
  // 2. Arrays: Connections, Fthis, FLUT, spotInfoMat, filteredSpotInfo,
  // 3. Empty arrays, can be initialized in each call: differencesMat
  // 4. Arrays that can be read directly: voxelList, beamPositions, hkls,
  // AllIDsInfo, AllSpotsInfo
  // 5. Fthis is updated each time. need to read it for rw access.
  FILE *paramsT = fopen("/dev/shm/scanningParams.txt", "r");
  char paramFN[4096], aline[4096], dummy[4096];
  size_t nVoxels, nhkls, nConn, maxNPos, dataArrSize, sizeFLUT, totalNrSpots,
      sizeSpotInfoMat;
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%s", paramFN);
  //~ printf("%s\n",paramFN);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &nVoxels);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &nhkls);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &nConn);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &maxNPos);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &dataArrSize);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &sizeFLUT);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &totalNrSpots);
  fgets(aline, 4096, paramsT);
  sscanf(aline, "%zu", &sizeSpotInfoMat);
  fclose(paramsT);
  int n = nVoxels * 9;
  FILE *fileParam;
  fileParam = fopen(paramFN, "r");
  double omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength;
  int nScans, rings[500], nRings = 0, maxNEvals = 1000;
  int GrainNr = 1;
  char positionsFN[4096];
  char grainFN[4096];
  char idsfn[4096];
  char hklfn[4096];
  char spotInfoFN[4096];
  sprintf(hklfn, "hkls.csv");
  sprintf(grainFN, "Grains.csv");
  sprintf(idsfn, "IDsHash.csv");
  sprintf(spotInfoFN, "ExtraInfo.bin");
  char voxelsFN[4096];
  double EulTol = 3 * deg2rad;
  double ABCTol = 3;
  double ABGTol = 3;
  double LatCin[6];
  int optType = 0, fitT = 0;
  while (fgets(aline, 4096, fileParam) != NULL) {
    if (strncmp(aline, "HKLFile", strlen("HKLFile")) == 0) {
      sscanf(aline, "%s %s", dummy, hklfn);
    }
    if (strncmp(aline, "OptType", strlen("OptType")) == 0) {
      sscanf(aline, "%s %d", dummy, &optType);
    }
    if (strncmp(aline, "FitBoth", strlen("FitBoth")) == 0) {
      sscanf(aline, "%s %d", dummy, &fitT);
    }
    if (strncmp(aline, "GrainsFile", strlen("GrainsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, grainFN);
    }
    if (strncmp(aline, "IDsFile", strlen("IDsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, idsfn);
    }
    if (strncmp(aline, "SpotsFile", strlen("SpotsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, spotInfoFN);
    }
    if (strncmp(aline, "OmegaStep", strlen("OmegaStep")) == 0) {
      sscanf(aline, "%s %lf", dummy, &omegaStep);
      omegaStep = fabs(omegaStep);
    }
    if (strncmp(aline, "LatticeConstant", strlen("LatticeConstant")) == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
    }
    if (strncmp(aline, "LatticeParameter", strlen("LatticeParameter")) == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
    }
    if (strncmp(aline, "px", strlen("px")) == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
    }
    if (strncmp(aline, "VoxelLength", strlen("VoxelLength")) == 0) {
      sscanf(aline, "%s %lf", dummy, &voxelLen);
    }
    if (strncmp(aline, "BeamFWHM", strlen("BeamFWHM")) == 0) {
      sscanf(aline, "%s %lf", dummy, &beamFWHM);
    }
    if (strncmp(aline, "OmegaTol", strlen("OmegaTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &omeTol);
    }
    if (strncmp(aline, "Lsd", strlen("Lsd")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    }
    if (strncmp(aline, "Wavelength", strlen("Wavelength")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
    }
    if (strncmp(aline, "OrientTol", strlen("OrientTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &EulTol);
      EulTol *= deg2rad;
    }
    if (strncmp(aline, "ABCTol", strlen("ABCTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &ABCTol);
    }
    if (strncmp(aline, "ABGTol", strlen("ABGTol")) == 0) {
      sscanf(aline, "%s %lf", dummy, &ABGTol);
    }
    if (strncmp(aline, "nLayers", strlen("nLayers")) == 0) {
      sscanf(aline, "%s %d", dummy, &nScans);
    }
    if (strncmp(aline, "GrainNr", strlen("GrainNr")) == 0) {
      sscanf(aline, "%s %d", dummy, &GrainNr);
    }
    if (strncmp(aline, "NProcs", strlen("NProcs")) == 0) {
      sscanf(aline, "%s %d", dummy, &numProcs);
    }
    if (strncmp(aline, "PositionsFile", strlen("PositionsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, positionsFN);
    }
    if (strncmp(aline, "VoxelsFile", strlen("VoxelsFile")) == 0) {
      sscanf(aline, "%s %s", dummy, voxelsFN);
    }
    if (strncmp(aline, "RingThresh", strlen("RingThresh")) == 0) {
      sscanf(aline, "%s %d", dummy, &rings[nRings]);
      nRings++;
    }
  }
  fclose(fileParam);
  if ((ABCTol < 0.00001 || ABGTol < 0.00001) && fitT == 0) {
    FitType = 1;
  } else if ((EulTol < 0.00001) && fitT == 0) {
    FitType = 2;
  }
  long i, j, k;
  // read positions file
  FILE *positionsFile;
  positionsFile = fopen(positionsFN, "r");
  int nBeamPositions = nScans;
  double *beamPositions;
  beamPositions = calloc(nBeamPositions, sizeof(*beamPositions));
  fgets(aline, 4096, positionsFile);
  for (i = 0; i < nBeamPositions; i++) {
    fgets(aline, 4096, positionsFile);
    sscanf(aline, "%lf", &beamPositions[i]);
    beamPositions[i] *= -1000;
  }
  fclose(positionsFile);
  // read hkls file
  FILE *hklf;
  hklf = fopen(hklfn, "r");
  double ht, kt, lt, ringT;
  double *hklTs;
  hklTs = calloc(500 * 4, sizeof(*hklTs));
  nhkls = 0;
  fgets(aline, 4096, hklf);
  while (fgets(aline, 4096, hklf) != NULL) {
    sscanf(aline, "%lf %lf %lf %s %lf", &ht, &kt, &lt, dummy, &ringT);
    for (i = 0; i < nRings; i++) {
      if ((int)ringT == rings[i]) {
        hklTs[nhkls * 4 + 0] = ht;
        hklTs[nhkls * 4 + 1] = kt;
        hklTs[nhkls * 4 + 2] = lt;
        hklTs[nhkls * 4 + 3] = ringT;
        nhkls++;
      }
    }
  }
  fclose(hklf);
  double *hkls;
  hkls = calloc(nhkls * 4, sizeof(*hkls));
  for (i = 0; i < nhkls * 4; i++)
    hkls[i] = hklTs[i];
  nRings = (int)hkls[nhkls * 4 - 1];
  free(hklTs);

  FILE *voxelsFile;
  voxelsFile = fopen(voxelsFN, "r");
  double *voxelsT;
  voxelsT = calloc(nBeamPositions * nBeamPositions, sizeof(*voxelsT));
  nVoxels = 0;
  while (fgets(aline, 4096, voxelsFile) != NULL) {
    sscanf(aline, "%lf,%lf", &voxelsT[nVoxels * 2 + 0],
           &voxelsT[nVoxels * 2 + 1]);
    nVoxels++;
  }
  double *voxelList;
  voxelList = calloc(nVoxels * 2, sizeof(*voxelList));
  for (i = 0; i < nVoxels * 2; i++)
    voxelList[i] = voxelsT[i];
  free(voxelsT);
  long *AllIDsInfo;
  AllIDsInfo = calloc(nBeamPositions * nRings * 2, sizeof(*AllIDsInfo));
  FILE *idsfile;
  idsfile = fopen(idsfn, "r");
  int positionNr, startNr, endNr, ringNr;
  fgets(aline, 4096, idsfile);
  while (fgets(aline, 4096, idsfile) != NULL) {
    sscanf(aline, "%d %d %d %d", &positionNr, &ringNr, &startNr, &endNr);
    if (positionNr == 0)
      continue;
    AllIDsInfo[((positionNr - 1) * nRings + ringNr) * 2 + 0] = startNr;
    AllIDsInfo[((positionNr - 1) * nRings + ringNr) * 2 + 1] = endNr;
  }
  fclose(idsfile);
  double *spotInfoAll;
  long lenSpotInfoAll;
  lenSpotInfoAll = numProcs;
  lenSpotInfoAll *= nhkls * 2;
  lenSpotInfoAll *= 4;
  spotInfoAll = calloc(lenSpotInfoAll, sizeof(*spotInfoAll));
  double *differencesMat;
  differencesMat = calloc(totalNrSpots, sizeof(*differencesMat));

  // mmap Connections, Fthis, FLUT, spotInfoMat, filteredSpotInfo for read.
  // Read AllSpotsInfo
  const char *filename = "/dev/shm/ExtraInfo.bin";
  int rc;
  double *AllSpotsInfo;
  struct stat s;
  size_t size;
  int fd = open(filename, O_RDWR);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  int status = fstat(fd, &s);
  size = s.st_size;
  AllSpotsInfo = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check(AllSpotsInfo == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  close(fd);
  // Read x
  // x will be passed as an argument.
  //~ double *x;
  //~ fd = open("/dev/shm/x.bin",O_RDWR);
  //~ check(fd < 0, "open %s failed: %s", "/dev/shm/x.bin", strerror(errno));
  //~ status = fstat(fd,&s);
  //~ size = s.st_size;
  //~ x = mmap(0,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  //~ check (x == MAP_FAILED,"mmap %s failed: %s", "/dev/shm/x.bin",
  //strerror(errno)); ~ close(fd);
  // Read Fthis
  double *Fthis;
  fd = open("/dev/shm/Fthis.bin", O_RDWR);
  check(fd < 0, "open %s failed: %s", "/dev/shm/Fthis.bin", strerror(errno));
  status = fstat(fd, &s);
  size = s.st_size;
  Fthis = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check(Fthis == MAP_FAILED, "mmap %s failed: %s", "/dev/shm/Fthis.bin",
        strerror(errno));
  close(fd);
  // Read Fthis
  long *FLUT;
  fd = open("/dev/shm/FLUT.bin", O_RDWR);
  check(fd < 0, "open %s failed: %s", "/dev/shm/FLUT.bin", strerror(errno));
  status = fstat(fd, &s);
  size = s.st_size;
  FLUT = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check(FLUT == MAP_FAILED, "mmap %s failed: %s", "/dev/shm/FLUT.bin",
        strerror(errno));
  close(fd);
  // Read spotInfoMat
  double *spotInfoMat;
  fd = open("/dev/shm/spotInfoMat.bin", O_RDWR);
  check(fd < 0, "open %s failed: %s", "/dev/shm/spotInfoMat.bin",
        strerror(errno));
  status = fstat(fd, &s);
  size = s.st_size;
  spotInfoMat = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check(spotInfoMat == MAP_FAILED, "mmap %s failed: %s",
        "/dev/shm/spotInfoMat.bin", strerror(errno));
  close(fd);
  // Read filteredSpotInfo
  double *filteredSpotInfo;
  fd = open("/dev/shm/filteredSpotInfo.bin", O_RDWR);
  check(fd < 0, "open %s failed: %s", "/dev/shm/filteredSpotInfo.bin",
        strerror(errno));
  status = fstat(fd, &s);
  size = s.st_size;
  filteredSpotInfo = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check(filteredSpotInfo == MAP_FAILED, "mmap %s failed: %s",
        "/dev/shm/filteredSpotInfo.bin", strerror(errno));
  close(fd);
  // Make struct to pass to problem function
  struct FITTING_PARAMS f_data;
  f_data.FLUT = &FLUT[0];
  f_data.Fthis = &Fthis[0];
  f_data.Lsd = Lsd;
  f_data.Wavelength = Wavelength;
  f_data.beamFWHM = beamFWHM;
  f_data.beamPositions = &beamPositions[0];
  f_data.filteredSpotInfo = &filteredSpotInfo[0];
  f_data.hkls = &hkls[0];
  f_data.maxNPos = maxNPos;
  f_data.nBeamPositions = nBeamPositions;
  f_data.nhkls = nhkls;
  f_data.nRings = nRings;
  f_data.omeTol = omeTol;
  f_data.omegaStep = omegaStep;
  f_data.px = px;
  f_data.spotInfoMat = &spotInfoMat[0];
  f_data.totalNrSpots = totalNrSpots;
  f_data.voxelLen = voxelLen;
  f_data.voxelList = &voxelList[0];
  f_data.spotInfoAll = &spotInfoAll[0];
  f_data.AllSpotsInfo = &AllSpotsInfo[0];
  f_data.AllIDsInfo = &AllIDsInfo[0];
  f_data.differencesMat = &differencesMat[0];
  //~ f_data.Connections = &Connections[0];
  f_data.nConn = nConn;
  struct FITTING_PARAMS *f_datat;
  f_datat = &f_data;
  void *trp = (struct FITTING_PARAMS *)f_datat;
  double function_val = problem_function(n, x, trp);
  time_t current_time;
  char *c_time_string;
  current_time = time(NULL);
  c_time_string = ctime(&current_time);
  //~ printf("Current time is %s", c_time_string);
  //~ printf("%lf\n",function_val);
  return function_val;
}
