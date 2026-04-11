//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// GetMisorientation.h — Canonical header for crystallographic orientation math.
//
// All angles in RADIANS unless noted otherwise.
// Quaternion convention: (w, x, y, z) with w = cos(theta/2).
//

#ifndef GETMISORIENTATION_H
#define GETMISORIENTATION_H

#include <math.h>

#define MIDAS_DEG2RAD (M_PI / 180.0)
#define MIDAS_RAD2DEG (180.0 / M_PI)

// ── Quaternion utilities ─────────────────────────────────────

void normalizeQuat(double quat[4]);

// Quaternion Hamilton product: Q = q * r.  Forces Q[0] >= 0.
void QuaternionProduct(const double q[4], const double r[4], double Q[4]);

// ── Symmetry operations ──────────────────────────────────────

// Determine symmetry operators from space group number alone.
// For monoclinic (SG 3-15), defaults to b-axis unique.
// Returns number of symmetry operators.
int MakeSymmetries(int SGNr, double Sym[24][4]);

// Full version: uses lattice angles (degrees) to resolve monoclinic unique axis.
// For non-monoclinic SGNr, angles are ignored.
int MakeSymmetriesWithLattice(int SGNr, double alpha, double beta, double gamma,
                              double Sym[24][4]);

// Configure monoclinic symmetry operator based on lattice angles (degrees).
void ConfigureMonoclinicSym(double alpha, double beta, double gamma,
                            double Sym[2][4]);

// ── Fundamental region reduction ─────────────────────────────

void BringDownToFundamentalRegionSym(const double QuatIn[4], double QuatOut[4],
                                     int NrSymmetries, const double Sym[24][4]);

void BringDownToFundamentalRegion(const double QuatIn[4], double QuatOut[4],
                                  int SGNr);

// ── Misorientation ───────────────────────────────────────────

// Returns misorientation angle in RADIANS.  Also writes to *Angle.
// Does NOT modify input quaternion arrays.
double GetMisOrientation(const double quat1[4], const double quat2[4],
                         double axis[3], double *Angle, int SGNr);

// Returns misorientation angle in RADIANS.  Also writes to *Angle.
// Does NOT modify input quaternion arrays.
double GetMisOrientationAngle(const double quat1[4], const double quat2[4],
                              double *Angle, int NrSymmetries,
                              const double Sym[24][4]);

// ── Orientation conversions ──────────────────────────────────

// Flat 9-element row-major orientation matrix → quaternion (w,x,y,z).
void OrientMat2Quat(const double OrientMat[9], double Quat[4]);

// 3x3 orientation matrix → Euler angles (psi, phi, theta) in RADIANS.
void OrientMat2Euler(const double m[3][3], double Euler[3]);

// Euler angles (psi, phi, theta) in RADIANS → 3x3 orientation matrix.
void Euler2OrientMat(const double Euler[3], double m_out[3][3]);

// Euler angles (psi, phi, theta) in RADIANS → flat 9-element row-major matrix.
void Euler2OrientMat9(const double Euler[3], double m_out[9]);

// Quaternion (w,x,y,z) → 3x3 rotation matrix.
void QuatToOrientMat33(const double Quat[4], double R[3][3]);

// Project a 3x3 matrix onto SO(3) via quaternion normalization roundtrip.
void OrthogonalizeOrientMat(double R[3][3]);

// ── Batch functions (OpenMP-parallel) ────────────────────────

// Batch misorientation angles for n pairs of quaternions.
// quats1, quats2: n*4 contiguous arrays.  angles_out: n-element output (radians).
void GetMisOrientationAngleBatch(int n, const double *quats1, const double *quats2,
                                 double *angles_out, int NrSymmetries,
                                 const double Sym[24][4]);

// Batch misorientation from orientation matrices.
// OMs1, OMs2: n*9 contiguous arrays.  angles_out: n-element output (radians).
void GetMisOrientationAngleOMBatch(int n, const double *OMs1, const double *OMs2,
                                   double *angles_out, int SGNr);

// Batch Euler (radians) → flat-9 orientation matrices.
void Euler2OrientMatBatch(int n, const double *eulers, double *OMs_out);

// Batch flat-9 orientation matrices → quaternions.
void OrientMat2QuatBatch(int n, const double *OMs, double *quats_out);

#endif // GETMISORIENTATION_H
