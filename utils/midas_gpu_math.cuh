//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// midas_gpu_math.cuh — Shared GPU device functions for MIDAS
//
// Header-only library of crystallographic math used by both NF-HEDM
// (FitOrientationGPU.cu) and FF-HEDM (IndexerGPU.cu).
// All functions use float32 and are __device__ __forceinline__.
//
// Usage:
//   #include "midas_gpu_math.cuh"   // from any .cu file
//

#ifndef MIDAS_GPU_MATH_CUH
#define MIDAS_GPU_MATH_CUH

#include <cuda_runtime.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

#ifndef MIDAS_GPU_DEG2RAD
#define MIDAS_GPU_DEG2RAD 0.0174532925199433f
#endif
#ifndef MIDAS_GPU_RAD2DEG
#define MIDAS_GPU_RAD2DEG 57.2957795130823f
#endif
#ifndef MIDAS_GPU_EPS
#define MIDAS_GPU_EPS 1e-9f
#endif

// ─────────────────────────────────────────────────────────────
// Trig convenience
// ─────────────────────────────────────────────────────────────

__device__ __forceinline__ float midas_sindf(float deg) {
  return sinf(MIDAS_GPU_DEG2RAD * deg);
}

__device__ __forceinline__ float midas_cosdf(float deg) {
  return cosf(MIDAS_GPU_DEG2RAD * deg);
}

__device__ __forceinline__ float midas_asindf(float x) {
  return MIDAS_GPU_RAD2DEG * asinf(fmaxf(-1.0f, fminf(1.0f, x)));
}

__device__ __forceinline__ float midas_acosdf(float x) {
  return MIDAS_GPU_RAD2DEG * acosf(fmaxf(-1.0f, fminf(1.0f, x)));
}

// ─────────────────────────────────────────────────────────────
// Vector / Matrix operations
// ─────────────────────────────────────────────────────────────

/// 3×3 matrix × 3-vector: r = m · v
__device__ __forceinline__
void midas_MatrixMultF(const float m[3][3], const float v[3], float r[3]) {
  float t[3];
  for (int i = 0; i < 3; i++)
    t[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
  r[0] = t[0]; r[1] = t[1]; r[2] = t[2];
}

/// 3×3 matrix × 3×3 matrix: res = m · n
__device__ __forceinline__
void midas_MatrixMultF33(const float m[3][3], const float n[3][3],
                         float res[3][3]) {
  float t[3][3];
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
float midas_dot3(const float a[3], const float b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Length of a 3-vector
__device__ __forceinline__
float midas_len3(float x, float y, float z) {
  return sqrtf(x*x + y*y + z*z);
}

/// Cross product: c = a × b
__device__ __forceinline__
void midas_cross3(const float a[3], const float b[3], float c[3]) {
  c[0] = a[1]*b[2] - b[1]*a[2];
  c[1] = a[2]*b[0] - b[2]*a[0];
  c[2] = a[0]*b[1] - b[0]*a[1];
}

// ─────────────────────────────────────────────────────────────
// Rotation helpers
// ─────────────────────────────────────────────────────────────

/// Rotate 3-vector around Z axis by alpha degrees: v2 = Rz(alpha) · v1
__device__ __forceinline__
void midas_RotateAroundZ(const float v1[3], float alpha, float v2[3]) {
  float cosa = cosf(MIDAS_GPU_DEG2RAD * alpha);
  float sina = sinf(MIDAS_GPU_DEG2RAD * alpha);
  v2[0] = cosa * v1[0] - sina * v1[1];
  v2[1] = sina * v1[0] + cosa * v1[1];
  v2[2] = v1[2];
}

/// Compute azimuthal angle eta (degrees) from detector (y, z) coordinates.
/// Convention: eta = acos(z/r) with sign flip for y>0.
__device__ __forceinline__
void midas_CalcEtaAngle(float y, float z, float *eta) {
  float denom = sqrtf(y*y + z*z);
  if (denom < MIDAS_GPU_EPS) {
    *eta = 0.0f;
    return;
  }
  float cos_val = fmaxf(-1.0f, fminf(1.0f, z / denom));
  *eta = MIDAS_GPU_RAD2DEG * acosf(cos_val);
  if (y > 0.0f) *eta = -(*eta);
}

/// Axis-angle → 3×3 rotation matrix. angle is in degrees.
__device__ __forceinline__
void midas_AxisAngle2RotMatrix(const float axis[3], float angle,
                               float R[3][3]) {
  float norm_sq = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2];
  if (norm_sq < MIDAS_GPU_EPS) {
    // Zero axis → identity
    R[0][0]=1; R[0][1]=0; R[0][2]=0;
    R[1][0]=0; R[1][1]=1; R[1][2]=0;
    R[2][0]=0; R[2][1]=0; R[2][2]=1;
    return;
  }
  float inv_len = 1.0f / sqrtf(norm_sq);
  float u = axis[0]*inv_len, v = axis[1]*inv_len, w = axis[2]*inv_len;
  float rad = MIDAS_GPU_DEG2RAD * angle;
  float c = cosf(rad), s = sinf(rad), omc = 1.0f - c;

  R[0][0] =    c + u*u*omc;   R[0][1] = -w*s + u*v*omc; R[0][2] =  v*s + u*w*omc;
  R[1][0] =  w*s + v*u*omc;   R[1][1] =    c + v*v*omc; R[1][2] = -u*s + v*w*omc;
  R[2][0] = -v*s + w*u*omc;   R[2][1] =  u*s + w*v*omc; R[2][2] =    c + w*w*omc;
}

// ─────────────────────────────────────────────────────────────
// Diffraction geometry
// ─────────────────────────────────────────────────────────────

/// Compute detector position from ring radius and eta angle.
__device__ __forceinline__
void midas_CalcSpotPosition(float ringRadius, float eta,
                            float *yl, float *zl) {
  float etaRad = MIDAS_GPU_DEG2RAD * eta;
  *yl = -(sinf(etaRad) * ringRadius);
  *zl =   cosf(etaRad) * ringRadius;
}

/// Compute omega solutions and corresponding eta values for a G-vector.
/// theta is in degrees. Up to 4 solutions are returned.
__device__ __forceinline__
void midas_CalcOmega(float x, float y, float z, float theta,
                     float omegas[4], float etas[4], int *nsol) {
  *nsol = 0;
  float len = sqrtf(x*x + y*y + z*z);
  if (len < MIDAS_GPU_EPS) return;

  float v = sinf(MIDAS_GPU_DEG2RAD * theta) * len;
  const float almost_zero = 1e-4f;

  if (fabsf(y) < almost_zero) {
    if (fabsf(x) > almost_zero) {
      float cosome1 = -v / x;
      if (fabsf(cosome1) <= 1.0f) {
        float ome = acosf(cosome1) * MIDAS_GPU_RAD2DEG;
        if (*nsol < 4) omegas[(*nsol)++] = ome;
        if (*nsol < 4) omegas[(*nsol)++] = -ome;
      }
    }
  } else {
    float y2 = y * y;
    float a = 1.0f + (x*x) / y2;
    float b = (2.0f * v * x) / y2;
    float c = (v*v) / y2 - 1.0f;
    float discr = b*b - 4.0f*a*c;

    if (discr >= 0.0f && fabsf(a) > MIDAS_GPU_EPS) {
      float sqrt_d = sqrtf(discr);
      float two_a = 2.0f * a;

      // Solution 1
      float cosome1 = (-b + sqrt_d) / two_a;
      if (fabsf(cosome1) <= 1.0f) {
        float ome_a = acosf(cosome1), ome_b = -ome_a;
        float eq_a = -x*cosf(ome_a) + y*sinf(ome_a);
        float eq_b = -x*cosf(ome_b) + y*sinf(ome_b);
        if (*nsol < 4)
          omegas[(*nsol)++] = (fabsf(eq_a-v) < fabsf(eq_b-v) ? ome_a : ome_b)
                              * MIDAS_GPU_RAD2DEG;
      }

      // Solution 2
      float cosome2 = (-b - sqrt_d) / two_a;
      if (fabsf(cosome2) <= 1.0f) {
        float ome_a = acosf(cosome2), ome_b = -ome_a;
        float eq_a = -x*cosf(ome_a) + y*sinf(ome_a);
        float eq_b = -x*cosf(ome_b) + y*sinf(ome_b);
        if (*nsol < 4)
          omegas[(*nsol)++] = (fabsf(eq_a-v) < fabsf(eq_b-v) ? ome_a : ome_b)
                              * MIDAS_GPU_RAD2DEG;
      }
    }
  }

  // Compute eta for each omega solution
  float gv[3] = {x, y, z};
  for (int i = 0; i < *nsol; i++) {
    float gw[3];
    midas_RotateAroundZ(gv, omegas[i], gw);
    midas_CalcEtaAngle(gw[1], gw[2], &etas[i]);
  }
}

/// Displacement of a diffraction spot due to sample position.
/// Used in FF-HEDM Indexer (COM-based displacement).
/// (a,b,c) = sample position in lab frame, (xi,yi,zi) = spot direction,
/// omega = rotation angle in degrees.
__device__ __forceinline__
void midas_displacement_spot_COM(float a, float b, float c,
                                 float xi, float yi, float zi,
                                 float omega,
                                 float *Displ_y, float *Displ_z) {
  float inv_len = 1.0f / sqrtf(xi*xi + yi*yi + zi*zi);
  xi *= inv_len; yi *= inv_len; zi *= inv_len;
  float sinOme = sinf(MIDAS_GPU_DEG2RAD * omega);
  float cosOme = cosf(MIDAS_GPU_DEG2RAD * omega);
  float t = (a * cosOme - b * sinOme) / xi;
  *Displ_y = (a * sinOme + b * cosOme) - t * yi;
  *Displ_z = c - t * zi;
}

/// Convert detector spot to G-vector (reciprocal space).
__device__ __forceinline__
void midas_spot_to_gv(float xi, float yi, float zi, float omega,
                      float *g1, float *g2, float *g3) {
  float len = sqrtf(xi*xi + yi*yi + zi*zi);
  if (len < MIDAS_GPU_EPS) {
    *g1 = 0; *g2 = 0; *g3 = 0;
    return;
  }
  float xn = xi/len, yn = yi/len, zn = zi/len;
  float g1r = -1.0f + xn;
  float g2r = yn;
  float cosOme = cosf(-omega * MIDAS_GPU_DEG2RAD);
  float sinOme = sinf(-omega * MIDAS_GPU_DEG2RAD);
  *g1 = g1r * cosOme - g2r * sinOme;
  *g2 = g1r * sinOme + g2r * cosOme;
  *g3 = zn;
}

/// Convert detector spot to G-vector, corrected for sample position.
__device__ __forceinline__
void midas_spot_to_gv_pos(float xi, float yi, float zi, float omega,
                          float cx, float cy, float cz,
                          float *g1, float *g2, float *g3) {
  float v[3] = {cx, cy, cz}, vr[3];
  midas_RotateAroundZ(v, omega, vr);
  midas_spot_to_gv(xi - vr[0], yi - vr[1], zi - vr[2], omega, g1, g2, g3);
}

/// Internal angle between two 3-vectors (degrees).
__device__ __forceinline__
void midas_CalcInternalAngle(float x1, float y1, float z1,
                             float x2, float y2, float z2,
                             float *ia) {
  float l1 = midas_len3(x1, y1, z1);
  float l2 = midas_len3(x2, y2, z2);
  if (l1 < MIDAS_GPU_EPS || l2 < MIDAS_GPU_EPS) {
    *ia = 0.0f;
    return;
  }
  float v1[3] = {x1,y1,z1}, v2[3] = {x2,y2,z2};
  double tmp = (double)midas_dot3(v1, v2) / ((double)l1 * (double)l2);
  if (tmp > 1.0) tmp = 1.0; if (tmp < -1.0) tmp = -1.0;
  *ia = (float)(MIDAS_GPU_RAD2DEG * acos(tmp));
}

/// Euler angles (degrees) → 3x3 orientation matrix.
__device__ __forceinline__
void midas_Euler2OrientMat(float psi, float phi, float theta,
                           float m[3][3]) {
  float cps = midas_cosdf(psi),  sps = midas_sindf(psi);
  float cph = midas_cosdf(phi),  sph = midas_sindf(phi);
  float cth = midas_cosdf(theta), sth = midas_sindf(theta);
  m[0][0] = cth*cps - sth*cph*sps;    m[0][1] = -cth*cph*sps - sth*cps;  m[0][2] =  sph*sps;
  m[1][0] = cth*sps + sth*cph*cps;    m[1][1] =  cth*cph*cps - sth*sps;  m[1][2] = -sph*cps;
  m[2][0] = sth*sph;                  m[2][1] =  cth*sph;                m[2][2] =  cph;
}

/// Make a unit vector from (x,y,z).
__device__ __forceinline__
void midas_MakeUnitLength(float x, float y, float z,
                          float *xu, float *yu, float *zu) {
  float len = sqrtf(x*x + y*y + z*z);
  if (len < MIDAS_GPU_EPS) {
    *xu = 0; *yu = 0; *zu = 0;
    return;
  }
  float inv = 1.0f / len;
  *xu = x * inv; *yu = y * inv; *zu = z * inv;
}

#endif // MIDAS_GPU_MATH_CUH
