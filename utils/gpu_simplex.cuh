/*
 * gpu_simplex.cuh — Header-only GPU batch Nelder-Mead simplex optimizer
 *
 * Copyright (c) 2014, UChicago Argonne, LLC. See LICENSE file.
 *
 * Usage:
 *   1. Define a functor with __device__ float operator()(const float *x, int ndim)
 *   2. Call gpu_simplex_batch_kernel<NDIM, Functor><<<grid, block>>>(...);
 *
 * Each thread runs one independent NM optimization.
 * Template on NDIM for compile-time unrolling (typical: 3, 6, 12).
 */

#ifndef GPU_SIMPLEX_CUH
#define GPU_SIMPLEX_CUH

#include <cuda_runtime.h>
#include <float.h>

// Maximum dimension supported (for static arrays)
#define GPU_SIMPLEX_MAX_DIM 12

/// Nelder-Mead parameters (standard values)
struct NMParams {
  float alpha;  // reflection:  1.0
  float gamma;  // expansion:   2.0
  float rho;    // contraction:  0.5
  float sigma;  // shrink:       0.5
};

__device__ static inline NMParams nm_default_params() {
  NMParams p;
  p.alpha = 1.0f;
  p.gamma = 2.0f;
  p.rho   = 0.5f;
  p.sigma = 0.5f;
  return p;
}

/// Clamp x[i] to [lo[i], hi[i]]
template <int NDIM>
__device__ static inline void nm_clamp(float *x, const float *lo, const float *hi) {
  #pragma unroll
  for (int i = 0; i < NDIM; i++) {
    if (x[i] < lo[i]) x[i] = lo[i];
    if (x[i] > hi[i]) x[i] = hi[i];
  }
}

/// Find indices of best, worst, second-worst vertices
template <int NDIM>
__device__ static inline void nm_find_indices(
    const float fvals[NDIM + 1], int *best, int *worst, int *second_worst) {
  *best = 0; *worst = 0; *second_worst = 0;
  for (int i = 1; i <= NDIM; i++) {
    if (fvals[i] < fvals[*best]) *best = i;
    if (fvals[i] > fvals[*worst]) *worst = i;
  }
  // Find second worst
  *second_worst = *best;  // initialize to best so any other is >= it
  for (int i = 0; i <= NDIM; i++) {
    if (i == *worst) continue;
    if (fvals[i] > fvals[*second_worst]) *second_worst = i;
  }
}

/// Compute centroid of all vertices except 'exclude'
template <int NDIM>
__device__ static inline void nm_centroid(
    const float simplex[GPU_SIMPLEX_MAX_DIM + 1][GPU_SIMPLEX_MAX_DIM],
    int exclude, float centroid[GPU_SIMPLEX_MAX_DIM]) {
  #pragma unroll
  for (int j = 0; j < NDIM; j++) centroid[j] = 0.0f;
  for (int i = 0; i <= NDIM; i++) {
    if (i == exclude) continue;
    #pragma unroll
    for (int j = 0; j < NDIM; j++) centroid[j] += simplex[i][j];
  }
  float inv = 1.0f / (float)NDIM;
  #pragma unroll
  for (int j = 0; j < NDIM; j++) centroid[j] *= inv;
}

/**
 * GPU Nelder-Mead simplex optimization — runs ONE optimization per thread.
 *
 * Template parameters:
 *   NDIM    — dimension of optimization (e.g., 3 for Euler angles)
 *   ObjFunc — functor type with: __device__ float operator()(const float *x, int ndim)
 *
 * @param nJobs       Number of independent optimization jobs
 * @param startPoints [nJobs × NDIM] initial guesses
 * @param lowerBounds [NDIM] lower bounds
 * @param upperBounds [NDIM] upper bounds
 * @param results     [nJobs × NDIM] optimized parameters (output)
 * @param fvals_out   [nJobs] final objective values (output)
 * @param objFunc     Objective function functor
 * @param tol         Convergence tolerance (on function value range)
 * @param maxIter     Maximum iterations per job
 * @param initStep    Initial simplex step size (fraction of bound range)
 */
template <int NDIM, typename ObjFunc>
__global__ void gpu_simplex_batch_kernel(
    int nJobs,
    const float * __restrict__ startPoints,
    const float * __restrict__ lowerBounds,
    const float * __restrict__ upperBounds,
    float *results,
    float *fvals_out,
    ObjFunc objFunc,
    float tol,
    int maxIter,
    float initStep) {

  int jobIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (jobIdx >= nJobs) return;

  NMParams nm = nm_default_params();

  // Load bounds into registers
  float lo[NDIM], hi[NDIM];
  #pragma unroll
  for (int i = 0; i < NDIM; i++) {
    lo[i] = lowerBounds[i];
    hi[i] = upperBounds[i];
  }

  // Initialize simplex: vertex 0 = start point, vertex i = start + step along dim i
  float simplex[GPU_SIMPLEX_MAX_DIM + 1][GPU_SIMPLEX_MAX_DIM];
  float fvals[GPU_SIMPLEX_MAX_DIM + 1];

  const float *x0 = &startPoints[jobIdx * NDIM];

  // Vertex 0 = starting point
  #pragma unroll
  for (int j = 0; j < NDIM; j++) simplex[0][j] = x0[j];
  nm_clamp<NDIM>(simplex[0], lo, hi);
  fvals[0] = objFunc(simplex[0], NDIM);

  // Vertices 1..NDIM: perturb along each dimension
  for (int i = 1; i <= NDIM; i++) {
    #pragma unroll
    for (int j = 0; j < NDIM; j++) simplex[i][j] = simplex[0][j];
    float step = initStep * (hi[i-1] - lo[i-1]);
    if (step < 1e-8f) step = 1e-4f;
    simplex[i][i-1] += step;
    nm_clamp<NDIM>(simplex[i], lo, hi);
    fvals[i] = objFunc(simplex[i], NDIM);
  }

  // Main NM loop
  float trial[GPU_SIMPLEX_MAX_DIM];
  float centroid[GPU_SIMPLEX_MAX_DIM];

  for (int iter = 0; iter < maxIter; iter++) {
    int best, worst, second_worst;
    nm_find_indices<NDIM>(fvals, &best, &worst, &second_worst);

    // Check convergence: range of function values
    float frange = fvals[worst] - fvals[best];
    if (frange < tol) break;

    // Centroid of all vertices except worst
    nm_centroid<NDIM>(simplex, worst, centroid);

    // --- Reflection ---
    #pragma unroll
    for (int j = 0; j < NDIM; j++)
      trial[j] = centroid[j] + nm.alpha * (centroid[j] - simplex[worst][j]);
    nm_clamp<NDIM>(trial, lo, hi);
    float f_reflect = objFunc(trial, NDIM);

    if (f_reflect < fvals[second_worst] && f_reflect >= fvals[best]) {
      // Accept reflection
      #pragma unroll
      for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial[j];
      fvals[worst] = f_reflect;
      continue;
    }

    if (f_reflect < fvals[best]) {
      // --- Expansion ---
      float trial_e[GPU_SIMPLEX_MAX_DIM];
      #pragma unroll
      for (int j = 0; j < NDIM; j++)
        trial_e[j] = centroid[j] + nm.gamma * (trial[j] - centroid[j]);
      nm_clamp<NDIM>(trial_e, lo, hi);
      float f_expand = objFunc(trial_e, NDIM);

      if (f_expand < f_reflect) {
        #pragma unroll
        for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial_e[j];
        fvals[worst] = f_expand;
      } else {
        #pragma unroll
        for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial[j];
        fvals[worst] = f_reflect;
      }
      continue;
    }

    // --- Contraction ---
    #pragma unroll
    for (int j = 0; j < NDIM; j++)
      trial[j] = centroid[j] + nm.rho * (simplex[worst][j] - centroid[j]);
    nm_clamp<NDIM>(trial, lo, hi);
    float f_contract = objFunc(trial, NDIM);

    if (f_contract < fvals[worst]) {
      #pragma unroll
      for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial[j];
      fvals[worst] = f_contract;
      continue;
    }

    // --- Shrink ---
    for (int i = 0; i <= NDIM; i++) {
      if (i == best) continue;
      #pragma unroll
      for (int j = 0; j < NDIM; j++)
        simplex[i][j] = simplex[best][j] + nm.sigma * (simplex[i][j] - simplex[best][j]);
      nm_clamp<NDIM>(simplex[i], lo, hi);
      fvals[i] = objFunc(simplex[i], NDIM);
    }
  }

  // Output best vertex
  int best, worst, second_worst;
  nm_find_indices<NDIM>(fvals, &best, &worst, &second_worst);

  #pragma unroll
  for (int j = 0; j < NDIM; j++)
    results[jobIdx * NDIM + j] = simplex[best][j];
  fvals_out[jobIdx] = fvals[best];
}

#endif // GPU_SIMPLEX_CUH
