//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
//
// GetMisorientation.c
//
// Crystallographic orientation math: misorientation angles, Euler/quaternion/
// matrix conversions, symmetry reduction, and OpenMP batch operations.
//
// All angles in RADIANS unless noted otherwise in parameter docs.
// Quaternion convention: (w, x, y, z) with w = cos(theta/2).
//
// Hemant Sharma

#include "GetMisorientation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPS 1e-9
#define ANGLE_TOL 1e-4

// ── Internal helpers (not exported) ──────────────────────────

static inline double clamp_acos(double x) {
  if (x > 1.0) x = 1.0;
  if (x < -1.0) x = -1.0;
  return acos(x);
}

static inline double sin_cos_to_angle(double s, double c) {
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

// ══════════════════════════════════════════════════════════════
//  Symmetry quaternion tables
//
//  All quaternions are unit norm.  Verified via group closure tests.
//  Convention: (w, x, y, z) with w = cos(theta/2).
// ══════════════════════════════════════════════════════════════

// Triclinic (1 operator)
static const double TricSym[1][4] = {
    {1.0, 0.0, 0.0, 0.0}};

// Monoclinic — default b-axis unique (180 deg about Y).
// Callers should use ConfigureMonoclinicSym() with lattice angles
// for correct axis selection.
static double MonoSym[2][4] = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0}};   // 180 deg about Y (b-unique, standard ITA)

// Orthorhombic — point group 222 (FIXED: was {1,1,0,0})
static const double OrtSym[4][4] = {
    {1.0, 0.0, 0.0, 0.0},    // E
    {0.0, 1.0, 0.0, 0.0},    // 180 deg about X
    {0.0, 0.0, 1.0, 0.0},    // 180 deg about Y
    {0.0, 0.0, 0.0, 1.0}};   // 180 deg about Z

// Tetragonal HIGH — Laue 4/mmm, point group 422 (SG 89-142)
static const double TetSym[8][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.70711, 0.00000, 0.00000, 0.70711},   // 90 deg about Z
    {0.00000, 0.00000, 0.00000, 1.00000},   // 180 deg about Z
    {0.70711, 0.00000, 0.00000,-0.70711},   // 270 deg about Z
    {0.00000, 1.00000, 0.00000, 0.00000},   // 180 deg about X
    {0.00000, 0.00000, 1.00000, 0.00000},   // 180 deg about Y
    {0.00000, 0.70711, 0.70711, 0.00000},   // 180 deg about [110]
    {0.00000,-0.70711, 0.70711, 0.00000}};  // 180 deg about [-110]

// Tetragonal LOW — Laue 4/m, point group 4 (SG 75-88)
static const double TetSymLow[4][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.70711, 0.00000, 0.00000, 0.70711},   // 90 deg about Z
    {0.00000, 0.00000, 0.00000, 1.00000},   // 180 deg about Z
    {0.70711, 0.00000, 0.00000,-0.70711}};  // 270 deg about Z

// Trigonal HIGH Type 1 — Laue -3m1 (SG 150,152,154,155,156,158,160,161,164,165,166,167)
// 2-fold axes on Y-axis and at +/-30 deg from X in xy-plane
static const double TrigSym[6][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.00000, 0.86603,-0.50000, 0.00000},   // 180 deg about axis at -30 deg
    {0.50000, 0.00000, 0.00000, 0.86603},   // 120 deg about Z
    {0.00000, 0.00000, 1.00000, 0.00000},   // 180 deg about Y
    {0.50000, 0.00000, 0.00000,-0.86603},   // 240 deg about Z
    {0.00000, 0.86603, 0.50000, 0.00000}};  // 180 deg about axis at +30 deg

// Trigonal HIGH Type 2 — Laue -31m (SG 149,151,153,157,159,162,163)
// 2-fold axes on X-axis and at +/-60 deg from X in xy-plane
static const double TrigSym2[6][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.50000, 0.00000, 0.00000, 0.86603},   // 120 deg about Z
    {0.50000, 0.00000, 0.00000,-0.86603},   // 240 deg about Z
    {0.00000, 0.50000,-0.86603, 0.00000},   // 180 deg about axis at -60 deg
    {0.00000, 1.00000, 0.00000, 0.00000},   // 180 deg about X
    {0.00000, 0.50000, 0.86603, 0.00000}};  // 180 deg about axis at +60 deg

// Trigonal LOW — Laue -3 (SG 143-148)
static const double TrigSymLow[3][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.50000, 0.00000, 0.00000, 0.86603},   // 120 deg about Z
    {0.50000, 0.00000, 0.00000,-0.86603}};  // 240 deg about Z

// Hexagonal HIGH — Laue 6/mmm, point group 622 (SG 177-194)
static const double HexSym[12][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.86603, 0.00000, 0.00000, 0.50000},   // 60 deg about Z
    {0.50000, 0.00000, 0.00000, 0.86603},   // 120 deg about Z
    {0.00000, 0.00000, 0.00000, 1.00000},   // 180 deg about Z
    {0.50000, 0.00000, 0.00000,-0.86603},   // 240 deg about Z
    {0.86603, 0.00000, 0.00000,-0.50000},   // 300 deg about Z
    {0.00000, 1.00000, 0.00000, 0.00000},   // 180 deg about X
    {0.00000, 0.86603, 0.50000, 0.00000},   // 180 deg about axis at +30 deg
    {0.00000, 0.50000, 0.86603, 0.00000},   // 180 deg about axis at +60 deg
    {0.00000, 0.00000, 1.00000, 0.00000},   // 180 deg about Y
    {0.00000,-0.50000, 0.86603, 0.00000},   // 180 deg about axis at +120 deg
    {0.00000,-0.86603, 0.50000, 0.00000}};  // 180 deg about axis at +150 deg

// Hexagonal LOW — Laue 6/m (SG 168-176)
static const double HexSymLow[6][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.86603, 0.00000, 0.00000, 0.50000},   // 60 deg about Z
    {0.50000, 0.00000, 0.00000, 0.86603},   // 120 deg about Z
    {0.00000, 0.00000, 0.00000, 1.00000},   // 180 deg about Z
    {0.50000, 0.00000, 0.00000,-0.86603},   // 240 deg about Z
    {0.86603, 0.00000, 0.00000,-0.50000}};  // 300 deg about Z

// Cubic HIGH — Laue m-3m, point group 432 (SG 207-230)
static const double CubSym[24][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.70711, 0.70711, 0.00000, 0.00000},
    {0.00000, 1.00000, 0.00000, 0.00000},
    {0.70711,-0.70711, 0.00000, 0.00000},
    {0.70711, 0.00000, 0.70711, 0.00000},
    {0.00000, 0.00000, 1.00000, 0.00000},
    {0.70711, 0.00000,-0.70711, 0.00000},
    {0.70711, 0.00000, 0.00000, 0.70711},
    {0.00000, 0.00000, 0.00000, 1.00000},
    {0.70711, 0.00000, 0.00000,-0.70711},
    {0.50000, 0.50000, 0.50000, 0.50000},
    {0.50000,-0.50000,-0.50000,-0.50000},
    {0.50000,-0.50000, 0.50000, 0.50000},
    {0.50000, 0.50000,-0.50000,-0.50000},
    {0.50000, 0.50000,-0.50000, 0.50000},
    {0.50000,-0.50000, 0.50000,-0.50000},
    {0.50000,-0.50000,-0.50000, 0.50000},
    {0.50000, 0.50000, 0.50000,-0.50000},
    {0.00000, 0.70711, 0.70711, 0.00000},
    {0.00000,-0.70711, 0.70711, 0.00000},
    {0.00000, 0.70711, 0.00000, 0.70711},
    {0.00000, 0.70711, 0.00000,-0.70711},
    {0.00000, 0.00000, 0.70711, 0.70711},
    {0.00000, 0.00000, 0.70711,-0.70711}};

// Cubic LOW — Laue m-3, point group 23 (SG 195-206)
static const double CubSymLow[12][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},   // E
    {0.00000, 1.00000, 0.00000, 0.00000},   // 180 deg about X
    {0.00000, 0.00000, 1.00000, 0.00000},   // 180 deg about Y
    {0.00000, 0.00000, 0.00000, 1.00000},   // 180 deg about Z
    {0.50000, 0.50000, 0.50000, 0.50000},   // 120 deg about [111]
    {0.50000,-0.50000,-0.50000,-0.50000},   // 240 deg about [111]
    {0.50000,-0.50000, 0.50000, 0.50000},   // 120 deg about [-111]
    {0.50000, 0.50000,-0.50000,-0.50000},   // 240 deg about [-111]
    {0.50000, 0.50000,-0.50000, 0.50000},   // 120 deg about [1-11]
    {0.50000,-0.50000, 0.50000,-0.50000},   // 240 deg about [1-11]
    {0.50000,-0.50000,-0.50000, 0.50000},   // 120 deg about [-1-11]
    {0.50000, 0.50000, 0.50000,-0.50000}};  // 240 deg about [-1-11]


// ══════════════════════════════════════════════════════════════
//  Quaternion utilities
// ══════════════════════════════════════════════════════════════

void normalizeQuat(double quat[4]) {
  double norm = sqrt(quat[0]*quat[0] + quat[1]*quat[1] +
                     quat[2]*quat[2] + quat[3]*quat[3]);
  if (norm < 1e-15) {
    quat[0] = 1.0; quat[1] = 0.0; quat[2] = 0.0; quat[3] = 0.0;
    return;
  }
  quat[0] /= norm; quat[1] /= norm; quat[2] /= norm; quat[3] /= norm;
}

void QuaternionProduct(const double q[4], const double r[4], double Q[4]) {
  Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3];
  Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3];
  Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1];
  Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2];
  if (Q[0] < 0) {
    Q[0] = -Q[0]; Q[1] = -Q[1]; Q[2] = -Q[2]; Q[3] = -Q[3];
  }
  normalizeQuat(Q);
}


// ══════════════════════════════════════════════════════════════
//  Symmetry operations
// ══════════════════════════════════════════════════════════════

void ConfigureMonoclinicSym(double alpha, double beta, double gamma,
                            double Sym[2][4]) {
  Sym[0][0] = 1.0; Sym[0][1] = 0.0; Sym[0][2] = 0.0; Sym[0][3] = 0.0;

  if (fabs(alpha - 90.0) > ANGLE_TOL &&
      fabs(beta  - 90.0) < ANGLE_TOL &&
      fabs(gamma - 90.0) < ANGLE_TOL) {
    // a-axis unique (alpha != 90): 180 deg about X
    Sym[1][0] = 0.0; Sym[1][1] = 1.0; Sym[1][2] = 0.0; Sym[1][3] = 0.0;
  } else if (fabs(beta  - 90.0) > ANGLE_TOL &&
             fabs(alpha - 90.0) < ANGLE_TOL &&
             fabs(gamma - 90.0) < ANGLE_TOL) {
    // b-axis unique (beta != 90): 180 deg about Y — standard ITA setting
    Sym[1][0] = 0.0; Sym[1][1] = 0.0; Sym[1][2] = 1.0; Sym[1][3] = 0.0;
  } else if (fabs(gamma - 90.0) > ANGLE_TOL &&
             fabs(alpha - 90.0) < ANGLE_TOL &&
             fabs(beta  - 90.0) < ANGLE_TOL) {
    // c-axis unique (gamma != 90): 180 deg about Z
    Sym[1][0] = 0.0; Sym[1][1] = 0.0; Sym[1][2] = 0.0; Sym[1][3] = 1.0;
  } else {
    // Default to b-axis unique.  Only warn if caller provided non-trivial
    // angles (suppress for the MakeSymmetries default of 90/90/90).
    int allRight = (fabs(alpha - 90.0) < ANGLE_TOL &&
                    fabs(beta  - 90.0) < ANGLE_TOL &&
                    fabs(gamma - 90.0) < ANGLE_TOL);
    if (!allRight)
      fprintf(stderr, "WARNING: Cannot determine monoclinic unique axis from "
              "angles (%.2f, %.2f, %.2f). Defaulting to b-unique.\n",
              alpha, beta, gamma);
    Sym[1][0] = 0.0; Sym[1][1] = 0.0; Sym[1][2] = 1.0; Sym[1][3] = 0.0;
  }
}

// Trigonal type 2 space groups (-31m convention)
static const int TrigType2SGs[] = {149, 151, 153, 157, 159, 162, 163};
static const int NTrigType2 = 7;

static int isTrigonalType2(int SGNr) {
  for (int k = 0; k < NTrigType2; k++)
    if (SGNr == TrigType2SGs[k]) return 1;
  return 0;
}

int MakeSymmetriesWithLattice(int SGNr, double alpha, double beta, double gamma,
                              double Sym[24][4]) {
  int i, j, NrSymmetries = 0;
  const double (*src)[4] = NULL;

  if (SGNr <= 2) {
    NrSymmetries = 1;
    src = TricSym;
  } else if (SGNr <= 15) {
    NrSymmetries = 2;
    ConfigureMonoclinicSym(alpha, beta, gamma, Sym);
    return NrSymmetries;  // already written by ConfigureMonoclinicSym
  } else if (SGNr <= 74) {
    NrSymmetries = 4;
    src = OrtSym;
  } else if (SGNr <= 88) {
    NrSymmetries = 4;
    src = TetSymLow;
  } else if (SGNr <= 142) {
    NrSymmetries = 8;
    src = TetSym;
  } else if (SGNr <= 148) {
    NrSymmetries = 3;
    src = TrigSymLow;
  } else if (SGNr <= 167) {
    NrSymmetries = 6;
    src = isTrigonalType2(SGNr) ? TrigSym2 : TrigSym;
  } else if (SGNr <= 176) {
    NrSymmetries = 6;
    src = HexSymLow;
  } else if (SGNr <= 194) {
    NrSymmetries = 12;
    src = HexSym;
  } else if (SGNr <= 206) {
    NrSymmetries = 12;
    src = CubSymLow;
  } else if (SGNr <= 230) {
    NrSymmetries = 24;
    src = CubSym;
  } else {
    fprintf(stderr, "ERROR: Invalid space group number %d\n", SGNr);
    NrSymmetries = 0;
    return 0;
  }

  for (i = 0; i < NrSymmetries; i++)
    for (j = 0; j < 4; j++)
      Sym[i][j] = src[i][j];

  return NrSymmetries;
}

int MakeSymmetries(int SGNr, double Sym[24][4]) {
  // Default: all angles 90 deg.  For monoclinic this triggers b-unique default.
  return MakeSymmetriesWithLattice(SGNr, 90.0, 90.0, 90.0, Sym);
}


// ══════════════════════════════════════════════════════════════
//  Fundamental region reduction
// ══════════════════════════════════════════════════════════════

void BringDownToFundamentalRegionSym(const double QuatIn[4], double QuatOut[4],
                                     int NrSymmetries,
                                     const double Sym[24][4]) {
  int i, maxCosRowNr = 0;
  double qps[24][4], q2[4], qt[4], maxCos = -10000.0;
  double qIn[4] = {QuatIn[0], QuatIn[1], QuatIn[2], QuatIn[3]};
  normalizeQuat(qIn);
  for (i = 0; i < NrSymmetries; i++) {
    q2[0] = Sym[i][0]; q2[1] = Sym[i][1];
    q2[2] = Sym[i][2]; q2[3] = Sym[i][3];
    QuaternionProduct(qIn, q2, qt);
    qps[i][0] = qt[0]; qps[i][1] = qt[1];
    qps[i][2] = qt[2]; qps[i][3] = qt[3];
    if (maxCos < qt[0]) {
      maxCos = qt[0];
      maxCosRowNr = i;
    }
  }
  QuatOut[0] = qps[maxCosRowNr][0]; QuatOut[1] = qps[maxCosRowNr][1];
  QuatOut[2] = qps[maxCosRowNr][2]; QuatOut[3] = qps[maxCosRowNr][3];
  normalizeQuat(QuatOut);
}

void BringDownToFundamentalRegion(const double QuatIn[4], double QuatOut[4],
                                  int SGNr) {
  double Sym[24][4];
  int NrSymmetries = MakeSymmetries(SGNr, Sym);
  BringDownToFundamentalRegionSym(QuatIn, QuatOut, NrSymmetries, Sym);
}


// ══════════════════════════════════════════════════════════════
//  Misorientation
// ══════════════════════════════════════════════════════════════

double GetMisOrientation(const double quat1[4], const double quat2[4],
                         double axis[3], double *Angle, int SGNr) {
  double q1[4] = {quat1[0], quat1[1], quat1[2], quat1[3]};
  double q2[4] = {quat2[0], quat2[1], quat2[2], quat2[3]};
  double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];

  normalizeQuat(q1);
  normalizeQuat(q2);
  BringDownToFundamentalRegion(q1, q1FR, SGNr);
  BringDownToFundamentalRegion(q2, q2FR, SGNr);

  q1Inv[0] = -q1FR[0]; q1Inv[1] = q1FR[1];
  q1Inv[2] = q1FR[2];  q1Inv[3] = q1FR[3];
  QuaternionProduct(q1Inv, q2FR, QP);
  BringDownToFundamentalRegion(QP, MisV, SGNr);

  if (MisV[0] > 1.0) MisV[0] = 1.0;
  double angle = 2.0 * clamp_acos(MisV[0]);  // radians

  if (fabs(MisV[0] - 1.0) < 1e-10) {
    axis[0] = 1.0; axis[1] = 0.0; axis[2] = 0.0;
  } else {
    double s = sqrt(1.0 - MisV[0] * MisV[0]);
    axis[0] = MisV[1] / s;
    axis[1] = MisV[2] / s;
    axis[2] = MisV[3] / s;
  }
  *Angle = angle;
  return angle;
}

double GetMisOrientationAngle(const double quat1[4], const double quat2[4],
                              double *Angle, int NrSymmetries,
                              const double Sym[24][4]) {
  double q1[4] = {quat1[0], quat1[1], quat1[2], quat1[3]};
  double q2[4] = {quat2[0], quat2[1], quat2[2], quat2[3]};
  double q1FR[4], q2FR[4], QP[4], MisV[4];

  normalizeQuat(q1);
  normalizeQuat(q2);
  BringDownToFundamentalRegionSym(q1, q1FR, NrSymmetries, Sym);
  BringDownToFundamentalRegionSym(q2, q2FR, NrSymmetries, Sym);

  q1FR[0] = -q1FR[0];
  QuaternionProduct(q1FR, q2FR, QP);
  BringDownToFundamentalRegionSym(QP, MisV, NrSymmetries, Sym);

  if (MisV[0] > 1.0) MisV[0] = 1.0;
  double angle = 2.0 * clamp_acos(MisV[0]);  // radians
  *Angle = angle;
  return angle;
}


// ══════════════════════════════════════════════════════════════
//  Orientation conversions
// ══════════════════════════════════════════════════════════════

void OrientMat2Quat(const double OrientMat[9], double Quat[4]) {
  double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
  if (trace > 0) {
    double s = 0.5 / sqrt(trace + 1.0);
    Quat[0] = 0.25 / s;
    Quat[1] = (OrientMat[7] - OrientMat[5]) * s;
    Quat[2] = (OrientMat[2] - OrientMat[6]) * s;
    Quat[3] = (OrientMat[3] - OrientMat[1]) * s;
  } else {
    if (OrientMat[0] > OrientMat[4] && OrientMat[0] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[0] - OrientMat[4] - OrientMat[8]);
      Quat[0] = (OrientMat[7] - OrientMat[5]) / s;
      Quat[1] = 0.25 * s;
      Quat[2] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[3] = (OrientMat[2] + OrientMat[6]) / s;
    } else if (OrientMat[4] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[4] - OrientMat[0] - OrientMat[8]);
      Quat[0] = (OrientMat[2] - OrientMat[6]) / s;
      Quat[1] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[2] = 0.25 * s;
      Quat[3] = (OrientMat[5] + OrientMat[7]) / s;
    } else {
      double s = 2.0 * sqrt(1.0 + OrientMat[8] - OrientMat[0] - OrientMat[4]);
      Quat[0] = (OrientMat[3] - OrientMat[1]) / s;
      Quat[1] = (OrientMat[2] + OrientMat[6]) / s;
      Quat[2] = (OrientMat[5] + OrientMat[7]) / s;
      Quat[3] = 0.25 * s;
    }
  }
  if (Quat[0] < 0) {
    Quat[0] = -Quat[0]; Quat[1] = -Quat[1];
    Quat[2] = -Quat[2]; Quat[3] = -Quat[3];
  }
  normalizeQuat(Quat);
}

void OrientMat2Euler(const double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = clamp_acos(m[2][2]);
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

// Euler angles in RADIANS → 3x3 orientation matrix.
void Euler2OrientMat(const double Euler[3], double m_out[3][3]) {
  double psi = Euler[0], phi = Euler[1], theta = Euler[2];
  double cps = cos(psi), cph = cos(phi), cth = cos(theta);
  double sps = sin(psi), sph = sin(phi), sth = sin(theta);
  m_out[0][0] = cth * cps - sth * cph * sps;
  m_out[0][1] = -cth * cph * sps - sth * cps;
  m_out[0][2] = sph * sps;
  m_out[1][0] = cth * sps + sth * cph * cps;
  m_out[1][1] = cth * cph * cps - sth * sps;
  m_out[1][2] = -sph * cps;
  m_out[2][0] = sth * sph;
  m_out[2][1] = cth * sph;
  m_out[2][2] = cph;
  OrthogonalizeOrientMat(m_out);
}

// Euler angles in RADIANS → flat 9-element row-major matrix.
void Euler2OrientMat9(const double Euler[3], double m_out[9]) {
  double m33[3][3];
  Euler2OrientMat(Euler, m33);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      m_out[i * 3 + j] = m33[i][j];
}

void QuatToOrientMat33(const double Quat[4], double R[3][3]) {
  double q0 = Quat[0], q1 = Quat[1], q2 = Quat[2], q3 = Quat[3];
  double q11 = q1*q1, q22 = q2*q2, q33 = q3*q3;
  double q01 = q0*q1, q02 = q0*q2, q03 = q0*q3;
  double q12 = q1*q2, q13 = q1*q3, q23 = q2*q3;
  R[0][0] = 1 - 2*(q22 + q33);
  R[0][1] = 2*(q12 - q03);
  R[0][2] = 2*(q13 + q02);
  R[1][0] = 2*(q12 + q03);
  R[1][1] = 1 - 2*(q11 + q33);
  R[1][2] = 2*(q23 - q01);
  R[2][0] = 2*(q13 - q02);
  R[2][1] = 2*(q23 + q01);
  R[2][2] = 1 - 2*(q11 + q22);
}

void OrthogonalizeOrientMat(double R[3][3]) {
  double flat[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      flat[i * 3 + j] = R[i][j];
  double q[4];
  OrientMat2Quat(flat, q);
  QuatToOrientMat33(q, R);
}


// ══════════════════════════════════════════════════════════════
//  Batch functions (OpenMP-parallel)
// ══════════════════════════════════════════════════════════════

void GetMisOrientationAngleBatch(int n, const double *quats1,
                                 const double *quats2, double *angles_out,
                                 int NrSymmetries, const double Sym[24][4]) {
  #pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; i++) {
    double q1[4], q2[4], angle;
    memcpy(q1, &quats1[i * 4], 4 * sizeof(double));
    memcpy(q2, &quats2[i * 4], 4 * sizeof(double));
    GetMisOrientationAngle(q1, q2, &angle, NrSymmetries, Sym);
    angles_out[i] = angle;
  }
}

void GetMisOrientationAngleOMBatch(int n, const double *OMs1,
                                   const double *OMs2, double *angles_out,
                                   int SGNr) {
  double Sym[24][4];
  int NrSymmetries = MakeSymmetries(SGNr, Sym);
  #pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; i++) {
    double q1[4], q2[4], angle;
    OrientMat2Quat(&OMs1[i * 9], q1);
    OrientMat2Quat(&OMs2[i * 9], q2);
    GetMisOrientationAngle(q1, q2, &angle, NrSymmetries, Sym);
    angles_out[i] = angle;
  }
}

void Euler2OrientMatBatch(int n, const double *eulers, double *OMs_out) {
  #pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; i++) {
    Euler2OrientMat9(&eulers[i * 3], &OMs_out[i * 9]);
  }
}

void OrientMat2QuatBatch(int n, const double *OMs, double *quats_out) {
  #pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; i++) {
    OrientMat2Quat(&OMs[i * 9], &quats_out[i * 4]);
  }
}
