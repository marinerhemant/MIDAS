//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
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
#define Calc2Theta(y, z, Lsd) atan(CalcNorm2(y, z) / Lsd) * rad2deg

int numProcs;
int nIters;
struct timespec start;
struct timespec end;
double dx[4] = {-0.5, +0.5, +0.5, -0.5};
double dy[4] = {-0.5, -0.5, +0.5, +0.5};
nlopt_opt opt;

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

void sigintHandler(int sig_num) {
  printf("\n Optimization terminated using Ctrl+C, we will exit gracefully and "
         "will write out last result. \n");
  fflush(stdout);
  nlopt_force_stop(opt);
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
  int inSide = 0, nrYs = 200, i, j, splCase = 0;
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
  if (maxY >= beamPosition - beamFWHM && minY <= beamPosition + beamFWHM)
    inSide = 1;
  if (inSide == 1) {
    startY = (minY > beamPosition - beamFWHM) ? minY : beamPosition - beamFWHM;
    endY = (maxY < beamPosition + beamFWHM) ? maxY : beamPosition + beamFWHM;
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

int main(int argc, char *argv[]) {
  //~ double Eul[3] = {283.440257,6.519380,44.496020}; // In degrees
  if (argc != 2) {
    printf("Usage: ./ScanningFunctions ParameterFile\n"
           "Eg.	   ./ScanningFunctions  params.txt\n");
    return;
  }
  char *paramFN;
  paramFN = argv[1];
  FILE *fileParam;
  fileParam = fopen(paramFN, "r");
  double omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength;
  int nScans, rings[500], nRings = 0, maxNEvals = 1000;
  int GrainNr = 1;
  char aline[4096], dummy[4096];
  char positionsFN[4096], outFN[4096];
  char voxelsFN[4096];
  double EulTol = 3 * deg2rad;
  double ABCTol = 3;
  double ABGTol = 3;
  double LatCin[6];
  double Eul[3];
  while (fgets(aline, 4096, fileParam) != NULL) {
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
    if (strncmp(aline, "Lsd", strlen("Lsd")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    }
    if (strncmp(aline, "Wavelength", strlen("Wavelength")) == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
    }
    if (strncmp(aline, "EulerAngle", strlen("EulerAngle")) == 0) {
      sscanf(aline, "%s %lf %lf %lf", dummy, &Eul[0], &Eul[1], &Eul[2]);
    }
    if (strncmp(aline, "nLayers", strlen("nLayers")) ==
        0) { // This is actually the number of scans.
      sscanf(aline, "%s %d", dummy, &nScans);
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

  char hklfn[4096];
  sprintf(hklfn, "hkls.csv");
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

  // What we need to return:
  //		SpotsList: 14*nSpots [0:y,1:z,2:omega,6:eta,7:2theta] Binary
  //format 		IDsHash [for upto maxRingNr for each spotNr, positionNr, ringNr,
  //startRowNr+1, endRowNr+1] csv format
  // What we need: voxelList, beamPositions, LatticeParameter, Wavelength, HKLs,
  // ringsToUse, nRings,

  // Read voxelList
  FILE *voxelsFile;
  voxelsFile = fopen(voxelsFN, "r");
  double *voxelsT;
  voxelsT = calloc(nBeamPositions * nBeamPositions, sizeof(*voxelsT));
  int nVoxels = 0;
  double COM[2] = {0, 0};
  while (fgets(aline, 4096, voxelsFile) != NULL) {
    sscanf(aline, "%lf,%lf", &voxelsT[nVoxels * 2 + 0],
           &voxelsT[nVoxels * 2 + 1]);
    COM[0] += voxelsT[nVoxels * 2 + 0];
    COM[1] += voxelsT[nVoxels * 2 + 1];
    nVoxels++;
  }
  COM[0] /= nVoxels;
  COM[1] /= nVoxels;
  double *voxelList;
  voxelList = calloc(nVoxels * 2, sizeof(*voxelList));
  for (i = 0; i < nVoxels * 2; i++)
    voxelList[i] = voxelsT[i];
  free(voxelsT);

  double *SpotInfo;
  long sizeSpotInfo = nBeamPositions;
  sizeSpotInfo *= nhkls + 2;
  sizeSpotInfo *= 2;
  sizeSpotInfo *= 14;
  SpotInfo = calloc(sizeSpotInfo, sizeof(*SpotInfo));
  double *localSpotInfo;
  localSpotInfo = calloc(nhkls * 2 * 9, sizeof(*localSpotInfo));
  long voxelNr, nSpots, spotNr, positionNr, rowNr, bestHKLNr;
  double thisPosition[3], rotatedPosition[3], LatC[6], Euler[3];
  double thisOmega, thisEta, thisBeamPosition, voxelFraction;
  for (voxelNr = 0; voxelNr < nVoxels; voxelNr++) {
    printf("%ld\n", voxelNr);
    fflush(stdout);
    // Let's simulate the dataset without any pertubation
    LatC[0] = LatCin[0];
    LatC[1] = LatCin[1];
    LatC[2] = LatCin[2];
    LatC[3] = LatCin[3];
    LatC[4] = LatCin[4];
    LatC[5] = LatCin[5];
    Euler[0] = Eul[0];
    Euler[1] = Eul[1];
    Euler[2] = Eul[2];
    thisPosition[0] = voxelList[voxelNr * 2 + 0];
    thisPosition[1] = voxelList[voxelNr * 2 + 1];
    thisPosition[2] = 0;
    nSpots = CalcDiffractionSpots(Lsd, Wavelength, thisPosition, LatC, Euler,
                                  nhkls, hkls, localSpotInfo, 1);
    for (spotNr = 0; spotNr < nSpots; spotNr++) {
      thisOmega = localSpotInfo[spotNr * 9 + 4];
      bestHKLNr = (long)localSpotInfo[spotNr * 9 + 6];
      RotateAroundZ(thisPosition, thisOmega, rotatedPosition);
      for (positionNr = 0; positionNr < nBeamPositions; positionNr++) {
        thisBeamPosition = beamPositions[positionNr];
        voxelFraction = IntensityFraction(voxelLen, thisBeamPosition, beamFWHM,
                                          thisPosition, thisOmega);
        if (voxelFraction > 0) {
          rowNr = positionNr * (nhkls + 2) * 2 * 14 + bestHKLNr * 14;
          SpotInfo[rowNr + 0] +=
              localSpotInfo[spotNr * 9 + 7] * voxelFraction; // y
          SpotInfo[rowNr + 1] +=
              localSpotInfo[spotNr * 9 + 8] * voxelFraction; // z
          SpotInfo[rowNr + 2] +=
              localSpotInfo[spotNr * 9 + 4] * voxelFraction; // omega
          SpotInfo[rowNr + 3] += voxelFraction;              // voxelFraction
          thisEta = CalcEta(localSpotInfo[spotNr * 9 + 7],
                            localSpotInfo[spotNr * 9 + 8]);
          SpotInfo[rowNr + 4] = localSpotInfo[spotNr * 9 + 5]; // ringNr
          SpotInfo[rowNr + 6] += thisEta * voxelFraction;      // Eta
          SpotInfo[rowNr + 7] +=
              Calc2Theta(localSpotInfo[spotNr * 9 + 7],
                         localSpotInfo[spotNr * 9 + 8], Lsd) *
              voxelFraction; // 2Theta
        }
      }
    }
  }
  for (spotNr = 0; spotNr < sizeSpotInfo / 14; spotNr++) {
    if (SpotInfo[spotNr * 14 + 3] > 0) {
      SpotInfo[spotNr * 14 + 0] /= SpotInfo[spotNr * 14 + 3];
      SpotInfo[spotNr * 14 + 1] /= SpotInfo[spotNr * 14 + 3];
      SpotInfo[spotNr * 14 + 2] /= SpotInfo[spotNr * 14 + 3];
      SpotInfo[spotNr * 14 + 6] /= SpotInfo[spotNr * 14 + 3];
      SpotInfo[spotNr * 14 + 7] /= SpotInfo[spotNr * 14 + 3];
    }
  }
  int *IDsInfo, thisRingNr;
  IDsInfo = calloc(5 * nRings * nBeamPositions, sizeof(*IDsInfo));
  double *outSpots;
  outSpots = calloc(sizeSpotInfo, sizeof(*outSpots));
  long nrFilled = 0, thishklnr, spotPosNr, idsrownr;
  for (positionNr = 0; positionNr < nBeamPositions; positionNr++) {
    // hklnr = (int)(besthkl-1)/2
    for (bestHKLNr = 0; bestHKLNr < (nhkls + 2) * 2; bestHKLNr++) {
      thishklnr = (long)(bestHKLNr - 1) / 2; // Use hkls to read ringNr
      thisRingNr = (int)hkls[thishklnr * 4 + 3];
      spotPosNr = positionNr;
      spotPosNr *= (nhkls + 2) * 2 * 14;
      spotPosNr += bestHKLNr * 14;
      if (SpotInfo[spotPosNr + 3] > 0) {
        outSpots[nrFilled * 14 + 0] = SpotInfo[spotPosNr + 0];
        outSpots[nrFilled * 14 + 1] = SpotInfo[spotPosNr + 1];
        outSpots[nrFilled * 14 + 2] = SpotInfo[spotPosNr + 2];
        outSpots[nrFilled * 14 + 3] = SpotInfo[spotPosNr + 3];
        outSpots[nrFilled * 14 + 4] = SpotInfo[spotPosNr + 4];
        outSpots[nrFilled * 14 + 6] = SpotInfo[spotPosNr + 6];
        outSpots[nrFilled * 14 + 7] = SpotInfo[spotPosNr + 7];
        nrFilled++;
        idsrownr = positionNr * nRings * 5 + thisRingNr * 5;
        IDsInfo[idsrownr + 0] = positionNr + 1;
        IDsInfo[idsrownr + 1] = thisRingNr;
        if (IDsInfo[idsrownr + 2] == 0)
          IDsInfo[idsrownr + 2] = nrFilled;
        IDsInfo[idsrownr + 3] = nrFilled;
      }
    }
  }
  FILE *fb, *fid;
  fb = fopen("SimExtraInfo.bin", "wb");
  fwrite(outSpots, nrFilled * 14 * sizeof(double), 1, fb);
  printf("Total spots: %ld\n", nrFilled);
  fclose(fb);
  fid = fopen("SimIDsHash.csv", "w");
  for (i = 0; i < nRings * nBeamPositions; i++)
    fprintf(fid, "%d %d %d %d %d\n", IDsInfo[i * 5 + 0], IDsInfo[i * 5 + 1],
            IDsInfo[i * 5 + 2], IDsInfo[i * 5 + 3], IDsInfo[i * 5 + 4]);
  fclose(fid);
}
