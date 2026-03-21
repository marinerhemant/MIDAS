/*
 * gpu_simplex.cuh — Header-only GPU batch Nelder-Mead simplex optimizer
 *
 * Copyright (c) 2014, UChicago Argonne, LLC. See LICENSE file.
 *
 * Algorithm matches NLOPT's nldrmd.c implementation exactly:
 * - Reflection α=1, Expansion γ=2, Contraction β=0.5, Shrink δ=0.5
 * - Inside AND outside contraction (based on fh vs fr)
 * - Contraction accepted when fc < fr AND fc < fh
 * - Shrink: all vertices contract toward best
 * - Initial simplex: direction reversal near bounds
 * - Convergence: frange < tol
 *
 * Usage:
 *   1. Define a functor with __device__ float operator()(const float *x, int ndim)
 *   2. Call from kernel with nm_optimize<NDIM>(...)
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

/// Nelder-Mead parameters (matching NLOPT: ALPHA=1, BETA=0.5, GAMMA=2, DELTA=0.5)
struct NMParams {
  float alpha;  // reflection:  1.0
  float gamma;  // expansion:   2.0
  float rho;    // contraction:  0.5 (NLOPT calls this BETA)
  float sigma;  // shrink:       0.5 (NLOPT calls this DELTA)
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

/// Reflect: xnew = c + scale * (c - xold), then clamp to bounds. (matches NLOPT reflectpt)
template <int NDIM>
__device__ static inline void nm_reflect(
    float *xnew, const float *c, float scale, const float *xold,
    const float *lo, const float *hi) {
  #pragma unroll
  for (int j = 0; j < NDIM; j++) {
    xnew[j] = c[j] + scale * (c[j] - xold[j]);
  }
  nm_clamp<NDIM>(xnew, lo, hi);
}

/**
 * GPU Nelder-Mead simplex optimization — runs ONE optimization per thread.
 * Algorithm matches NLOPT's nldrmd_minimize_ exactly.
 *
 * Template parameters:
 *   NDIM    — dimension of optimization (e.g., 3 for Euler angles)
 *   ObjFunc — functor type with: __device__ float operator()(const float *x, int ndim)
 *
 * @param nJobs       Number of independent optimization jobs
 * @param startPoints [nJobs × NDIM] initial guesses
 * @param lowerBounds [nJobs × NDIM] per-job lower bounds
 * @param upperBounds [nJobs × NDIM] per-job upper bounds
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

  // Load per-job bounds into registers
  float lo[NDIM], hi[NDIM];
  #pragma unroll
  for (int i = 0; i < NDIM; i++) {
    lo[i] = lowerBounds[jobIdx * NDIM + i];
    hi[i] = upperBounds[jobIdx * NDIM + i];
  }

  // Initialize simplex: vertex 0 = start point
  float simplex[GPU_SIMPLEX_MAX_DIM + 1][GPU_SIMPLEX_MAX_DIM];
  float fvals[GPU_SIMPLEX_MAX_DIM + 1];

  const float *x0 = &startPoints[jobIdx * NDIM];

  // Vertex 0 = starting point
  #pragma unroll
  for (int j = 0; j < NDIM; j++) simplex[0][j] = x0[j];
  nm_clamp<NDIM>(simplex[0], lo, hi);
  fvals[0] = objFunc(simplex[0], NDIM);

  // Vertices 1..NDIM: perturb along each dimension (matching NLOPT initial simplex)
  for (int i = 1; i <= NDIM; i++) {
    #pragma unroll
    for (int j = 0; j < NDIM; j++) simplex[i][j] = simplex[0][j];
    float step = initStep * (hi[i-1] - lo[i-1]);
    if (step < 1e-8f) step = 1e-4f;
    float newval = simplex[0][i-1] + step;
    // NLOPT-style direction reversal near bounds
    if (newval > hi[i-1]) {
      if (hi[i-1] - simplex[0][i-1] > fabsf(step) * 0.1f)
        newval = hi[i-1];
      else
        newval = simplex[0][i-1] - fabsf(step);
    }
    if (newval < lo[i-1]) {
      if (simplex[0][i-1] - lo[i-1] > fabsf(step) * 0.1f)
        newval = lo[i-1];
      else {
        newval = simplex[0][i-1] + fabsf(step);
        if (newval > hi[i-1])
          newval = 0.5f * ((hi[i-1] - simplex[0][i-1] > simplex[0][i-1] - lo[i-1]
                            ? hi[i-1] : lo[i-1]) + simplex[0][i-1]);
      }
    }
    simplex[i][i-1] = newval;
    nm_clamp<NDIM>(simplex[i], lo, hi);
    fvals[i] = objFunc(simplex[i], NDIM);
  }

  // Main NM loop — matches NLOPT's nldrmd algorithm
  float trial[GPU_SIMPLEX_MAX_DIM];
  float centroid[GPU_SIMPLEX_MAX_DIM];

  for (int iter = 0; iter < maxIter; iter++) {
    int best, worst, second_worst;
    nm_find_indices<NDIM>(fvals, &best, &worst, &second_worst);

    float fl = fvals[best];
    float fh = fvals[worst];

    // Check convergence: range of function values
    float frange = fh - fl;
    if (frange < tol) break;

    // Centroid of all vertices except worst
    nm_centroid<NDIM>(simplex, worst, centroid);

    // --- Reflection: xcur = c + alpha * (c - xh) ---
    nm_reflect<NDIM>(trial, centroid, nm.alpha, simplex[worst], lo, hi);
    float fr = objFunc(trial, NDIM);

    if (fr < fl) {
      // --- Expansion: try c + gamma * (c - xh) ---
      float trial_e[GPU_SIMPLEX_MAX_DIM];
      nm_reflect<NDIM>(trial_e, centroid, nm.gamma, simplex[worst], lo, hi);
      float fe = objFunc(trial_e, NDIM);

      if (fe < fr) {
        // Expansion is better — replace worst with expanded
        #pragma unroll
        for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial_e[j];
        fvals[worst] = fe;
      } else {
        // Expansion didn't improve — use reflection
        #pragma unroll
        for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial[j];
        fvals[worst] = fr;
      }
      continue;
    }

    if (fr < fvals[second_worst]) {
      // --- Accept reflection ---
      #pragma unroll
      for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial[j];
      fvals[worst] = fr;
      continue;
    }

    // --- Contraction (NLOPT-style: inside or outside based on fh vs fr) ---
    // If fh <= fr: inside contraction  → scale = -beta (toward centroid from worst)
    // If fh >  fr: outside contraction → scale = +beta (toward reflected from centroid)
    float cscale = (fh <= fr) ? -nm.rho : nm.rho;
    float trial_c[GPU_SIMPLEX_MAX_DIM];
    nm_reflect<NDIM>(trial_c, centroid, cscale, simplex[worst], lo, hi);
    float fc = objFunc(trial_c, NDIM);

    if (fc < fr && fc < fh) {
      // Successful contraction — accept
      #pragma unroll
      for (int j = 0; j < NDIM; j++) simplex[worst][j] = trial_c[j];
      fvals[worst] = fc;
      continue;
    }

    // --- Shrink: all vertices toward best ---
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
