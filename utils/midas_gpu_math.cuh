//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// midas_gpu_math.cuh — Shared GPU device functions for MIDAS
//
// Header-only library of crystallographic math used by both NF-HEDM
// (FitOrientationGPU.cu) and FF-HEDM (IndexerGPU.cu).
// All functions use RealType (default: double) and are __device__ __forceinline__.
//
// Usage:
//   #define RealType float   // or double (default)
//   #include "midas_gpu_math.cuh"
//

#ifndef MIDAS_GPU_MATH_CUH
#define MIDAS_GPU_MATH_CUH

#include <cuda_runtime.h>
#include <math.h>

// Default to double precision if not defined by the including .cu file
#ifndef RealType
#define RealType double
#endif

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

#ifndef MIDAS_GPU_DEG2RAD
#define MIDAS_GPU_DEG2RAD 0.0174532925199433
#endif
#ifndef MIDAS_GPU_RAD2DEG
#define MIDAS_GPU_RAD2DEG 57.2957795130823
#endif
#ifndef MIDAS_GPU_EPS
#define MIDAS_GPU_EPS 1e-9
#endif

// ─────────────────────────────────────────────────────────────
// Trig convenience
// ─────────────────────────────────────────────────────────────

__device__ __forceinline__ RealType midas_sindf(RealType deg) {
  return sin(MIDAS_GPU_DEG2RAD * deg);
}

__device__ __forceinline__ RealType midas_cosdf(RealType deg) {
  return cos(MIDAS_GPU_DEG2RAD * deg);
}

__device__ __forceinline__ RealType midas_asindf(RealType x) {
  return MIDAS_GPU_RAD2DEG * asin(fmax((RealType)-1.0, fmin((RealType)1.0, x)));
}

__device__ __forceinline__ RealType midas_acosdf(RealType x) {
  return MIDAS_GPU_RAD2DEG * acos(fmax((RealType)-1.0, fmin((RealType)1.0, x)));
}

// ─────────────────────────────────────────────────────────────
// Vector / Matrix operations
// ─────────────────────────────────────────────────────────────

/// 3×3 matrix × 3-vector: r = m · v
__device__ __forceinline__
void midas_MatrixMultF(const RealType m[3][3], const RealType v[3], RealType r[3]) {
  RealType t[3];
  for (int i = 0; i < 3; i++)
    t[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
  r[0] = t[0]; r[1] = t[1]; r[2] = t[2];
}

/// 3×3 matrix × 3×3 matrix: res = m · n
__device__ __forceinline__
void midas_MatrixMultF33(const RealType m[3][3], const RealType n[3][3],
                         RealType res[3][3]) {
  RealType t[3][3];
  for (int r = 0; r < 3; r++) {
    t[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
    t[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
    t[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
  }
  for (int r = 0; r < 3; r++) {
    res[r][0] = t[r][0]; res[r][1] = t[r][1]; res[r][2] = t[r][2];
  }
}

/// Dot product of two 3-vectors
__device__ __forceinline__
RealType midas_dot3(const RealType a[3], const RealType b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Length of a 3-vector
__device__ __forceinline__
RealType midas_len3(RealType x, RealType y, RealType z) {
  return sqrt(x*x + y*y + z*z);
}

/// Cross product: c = a × b
__device__ __forceinline__
void midas_cross3(const RealType a[3], const RealType b[3], RealType c[3]) {
  c[0] = a[1]*b[2] - b[1]*a[2];
  c[1] = a[2]*b[0] - b[2]*a[0];
  c[2] = a[0]*b[1] - b[0]*a[1];
}

// ─────────────────────────────────────────────────────────────
// Rotation helpers
// ─────────────────────────────────────────────────────────────

/// Rotate 3-vector around Z axis by alpha degrees: v2 = Rz(alpha) · v1
__device__ __forceinline__
void midas_RotateAroundZ(const RealType v1[3], RealType alpha, RealType v2[3]) {
  RealType cosa = cos(MIDAS_GPU_DEG2RAD * alpha);
  RealType sina = sin(MIDAS_GPU_DEG2RAD * alpha);
  v2[0] = cosa * v1[0] - sina * v1[1];
  v2[1] = sina * v1[0] + cosa * v1[1];
  v2[2] = v1[2];
}

/// Compute azimuthal angle eta (degrees) from detector (y, z) coordinates.
/// Convention: eta = acos(z/r) with sign flip for y>0.
__device__ __forceinline__
void midas_CalcEtaAngle(RealType y, RealType z, RealType *eta) {
  RealType denom = sqrt(y*y + z*z);
  if (denom < MIDAS_GPU_EPS) {
    *eta = (RealType)0.0;
    return;
  }
  RealType cos_val = fmax((RealType)-1.0, fmin((RealType)1.0, z / denom));
  *eta = MIDAS_GPU_RAD2DEG * acos(cos_val);
  if (y > (RealType)0.0) *eta = -(*eta);
}

/// Axis-angle → 3×3 rotation matrix. angle is in degrees.
__device__ __forceinline__
void midas_AxisAngle2RotMatrix(const RealType axis[3], RealType angle,
                               RealType R[3][3]) {
  RealType norm_sq = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2];
  if (norm_sq < MIDAS_GPU_EPS) {
    // Zero axis → identity
    R[0][0]=1; R[0][1]=0; R[0][2]=0;
    R[1][0]=0; R[1][1]=1; R[1][2]=0;
    R[2][0]=0; R[2][1]=0; R[2][2]=1;
    return;
  }
  RealType inv_len = (RealType)1.0 / sqrt(norm_sq);
  RealType u = axis[0]*inv_len, v = axis[1]*inv_len, w = axis[2]*inv_len;
  RealType rad = MIDAS_GPU_DEG2RAD * angle;
  RealType c = cos(rad), s = sin(rad), omc = (RealType)1.0 - c;

  R[0][0] =    c + u*u*omc;   R[0][1] = -w*s + u*v*omc; R[0][2] =  v*s + u*w*omc;
  R[1][0] =  w*s + v*u*omc;   R[1][1] =    c + v*v*omc; R[1][2] = -u*s + v*w*omc;
  R[2][0] = -v*s + w*u*omc;   R[2][1] =  u*s + w*v*omc; R[2][2] =    c + w*w*omc;
}

// ─────────────────────────────────────────────────────────────
// Diffraction geometry
// ─────────────────────────────────────────────────────────────

/// Compute detector position from ring radius and eta angle.
__device__ __forceinline__
void midas_CalcSpotPosition(RealType ringRadius, RealType eta,
                            RealType *yl, RealType *zl) {
  RealType etaRad = MIDAS_GPU_DEG2RAD * eta;
  *yl = -(sin(etaRad) * ringRadius);
  *zl =   cos(etaRad) * ringRadius;
}

/// Compute omega solutions and corresponding eta values for a G-vector.
/// theta is in degrees. Up to 4 solutions are returned.
__device__ __forceinline__
void midas_CalcOmega(RealType x, RealType y, RealType z, RealType theta,
                     RealType omegas[4], RealType etas[4], int *nsol) {
  *nsol = 0;
  RealType len = sqrt(x*x + y*y + z*z);
  if (len < MIDAS_GPU_EPS) return;

  RealType v = sin(MIDAS_GPU_DEG2RAD * theta) * len;
  const RealType almost_zero = (RealType)1e-4;

  if (fabs(y) < almost_zero) {
    if (fabs(x) > almost_zero) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= (RealType)1.0) {
        RealType ome = acos(cosome1) * MIDAS_GPU_RAD2DEG;
        if (*nsol < 4) omegas[(*nsol)++] = ome;
        if (*nsol < 4) omegas[(*nsol)++] = -ome;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType a = (RealType)1.0 + (x*x) / y2;
    RealType b = ((RealType)2.0 * v * x) / y2;
    RealType c = (v*v) / y2 - (RealType)1.0;
    RealType discr = b*b - (RealType)4.0*a*c;

    if (discr >= (RealType)0.0 && fabs(a) > MIDAS_GPU_EPS) {
      RealType sqrt_d = sqrt(discr);
      RealType two_a = (RealType)2.0 * a;

      // Solution 1
      RealType cosome1 = (-b + sqrt_d) / two_a;
      if (fabs(cosome1) <= (RealType)1.0) {
        RealType ome_a = acos(cosome1), ome_b = -ome_a;
        RealType eq_a = -x*cos(ome_a) + y*sin(ome_a);
        RealType eq_b = -x*cos(ome_b) + y*sin(ome_b);
        if (*nsol < 4)
          omegas[(*nsol)++] = (fabs(eq_a-v) < fabs(eq_b-v) ? ome_a : ome_b)
                              * MIDAS_GPU_RAD2DEG;
      }

      // Solution 2
      RealType cosome2 = (-b - sqrt_d) / two_a;
      if (fabs(cosome2) <= (RealType)1.0) {
        RealType ome_a = acos(cosome2), ome_b = -ome_a;
        RealType eq_a = -x*cos(ome_a) + y*sin(ome_a);
        RealType eq_b = -x*cos(ome_b) + y*sin(ome_b);
        if (*nsol < 4)
          omegas[(*nsol)++] = (fabs(eq_a-v) < fabs(eq_b-v) ? ome_a : ome_b)
                              * MIDAS_GPU_RAD2DEG;
      }
    }
  }

  // Compute eta for each omega solution
  RealType gv[3] = {x, y, z};
  for (int i = 0; i < *nsol; i++) {
    RealType gw[3];
    midas_RotateAroundZ(gv, omegas[i], gw);
    midas_CalcEtaAngle(gw[1], gw[2], &etas[i]);
  }
}

/// Displacement of a diffraction spot due to sample position.
/// Used in FF-HEDM Indexer (COM-based displacement).
/// (a,b,c) = sample position in lab frame, (xi,yi,zi) = spot direction,
/// omega = rotation angle in degrees.
__device__ __forceinline__
void midas_displacement_spot_COM(RealType a, RealType b, RealType c,
                                 RealType xi, RealType yi, RealType zi,
                                 RealType omega,
                                 RealType *Displ_y, RealType *Displ_z) {
  RealType inv_len = (RealType)1.0 / sqrt(xi*xi + yi*yi + zi*zi);
  xi *= inv_len; yi *= inv_len; zi *= inv_len;
  RealType sinOme = sin(MIDAS_GPU_DEG2RAD * omega);
  RealType cosOme = cos(MIDAS_GPU_DEG2RAD * omega);
  RealType t = (a * cosOme - b * sinOme) / xi;
  *Displ_y = (a * sinOme + b * cosOme) - t * yi;
  *Displ_z = c - t * zi;
}

/// Convert detector spot to G-vector (reciprocal space).
__device__ __forceinline__
void midas_spot_to_gv(RealType xi, RealType yi, RealType zi, RealType omega,
                      RealType *g1, RealType *g2, RealType *g3) {
  RealType len = sqrt(xi*xi + yi*yi + zi*zi);
  if (len < MIDAS_GPU_EPS) {
    *g1 = 0; *g2 = 0; *g3 = 0;
    return;
  }
  RealType xn = xi/len, yn = yi/len, zn = zi/len;
  RealType g1r = (RealType)-1.0 + xn;
  RealType g2r = yn;
  RealType cosOme = cos(-omega * MIDAS_GPU_DEG2RAD);
  RealType sinOme = sin(-omega * MIDAS_GPU_DEG2RAD);
  *g1 = g1r * cosOme - g2r * sinOme;
  *g2 = g1r * sinOme + g2r * cosOme;
  *g3 = zn;
}

/// Convert detector spot to G-vector, corrected for sample position.
__device__ __forceinline__
void midas_spot_to_gv_pos(RealType xi, RealType yi, RealType zi, RealType omega,
                          RealType cx, RealType cy, RealType cz,
                          RealType *g1, RealType *g2, RealType *g3) {
  RealType v[3] = {cx, cy, cz}, vr[3];
  midas_RotateAroundZ(v, omega, vr);
  midas_spot_to_gv(xi - vr[0], yi - vr[1], zi - vr[2], omega, g1, g2, g3);
}

/// Internal angle between two 3-vectors (degrees).
__device__ __forceinline__
void midas_CalcInternalAngle(RealType x1, RealType y1, RealType z1,
                             RealType x2, RealType y2, RealType z2,
                             RealType *ia) {
  RealType l1 = midas_len3(x1, y1, z1);
  RealType l2 = midas_len3(x2, y2, z2);
  if (l1 < MIDAS_GPU_EPS || l2 < MIDAS_GPU_EPS) {
    *ia = (RealType)0.0;
    return;
  }
  RealType v1[3] = {x1,y1,z1}, v2[3] = {x2,y2,z2};
  double tmp = (double)midas_dot3(v1, v2) / ((double)l1 * (double)l2);
  if (tmp > 1.0) tmp = 1.0; if (tmp < -1.0) tmp = -1.0;
  *ia = (RealType)(MIDAS_GPU_RAD2DEG * acos(tmp));
}

/// Euler angles (degrees) → 3x3 orientation matrix.
__device__ __forceinline__
void midas_Euler2OrientMat(RealType psi, RealType phi, RealType theta,
                           RealType m[3][3]) {
  RealType cps = midas_cosdf(psi),  sps = midas_sindf(psi);
  RealType cph = midas_cosdf(phi),  sph = midas_sindf(phi);
  RealType cth = midas_cosdf(theta), sth = midas_sindf(theta);
  m[0][0] = cth*cps - sth*cph*sps;    m[0][1] = -cth*cph*sps - sth*cps;  m[0][2] =  sph*sps;
  m[1][0] = cth*sps + sth*cph*cps;    m[1][1] =  cth*cph*cps - sth*sps;  m[1][2] = -sph*cps;
  m[2][0] = sth*sph;                  m[2][1] =  cth*sph;                m[2][2] =  cph;
}

/// Make a unit vector from (x,y,z).
__device__ __forceinline__
void midas_MakeUnitLength(RealType x, RealType y, RealType z,
                          RealType *xu, RealType *yu, RealType *zu) {
  RealType len = sqrt(x*x + y*y + z*z);
  if (len < MIDAS_GPU_EPS) {
    *xu = 0; *yu = 0; *zu = 0;
    return;
  }
  RealType inv = (RealType)1.0 / len;
  *xu = x * inv; *yu = y * inv; *zu = z * inv;
}

#endif // MIDAS_GPU_MATH_CUH
