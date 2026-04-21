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
 * Precision controlled by RealType (define before including):
 *   #define RealType float   // for NF-HEDM
 *   #define RealType double  // for FF-HEDM
 *
 * Usage:
 *   1. Define a functor with __device__ RealType operator()(const RealType *x, int ndim)
 *   2. Call gpu_simplex_batch_kernel<NDIM>(...) as a kernel, OR
 *   3. Call nm_optimize<NDIM>(...) from within another kernel (device-callable)
 *
 * Each thread runs one independent NM optimization.
 * Template on NDIM for compile-time unrolling (typical: 3, 6, 12).
 */

#ifndef GPU_SIMPLEX_CUH
#define GPU_SIMPLEX_CUH

#include <cuda_runtime.h>
#include <float.h>

// Default RealType to double if not defined by the including .cu file
#ifndef RealType
#define RealType double
#endif

// Maximum dimension supported (for static arrays)
#define GPU_SIMPLEX_MAX_DIM 12

/// Nelder-Mead parameters (matching NLOPT: ALPHA=1, BETA=0.5, GAMMA=2, DELTA=0.5)
struct NMParams {
  RealType alpha;  // reflection:  1.0
  RealType gamma;  // expansion:   2.0
  RealType rho;    // contraction:  0.5 (NLOPT calls this BETA)
  RealType sigma;  // shrink:       0.5 (NLOPT calls this DELTA)
};

__device__ static inline NMParams nm_default_params() {
  NMParams p;
  p.alpha = (RealType)1.0;
  p.gamma = (RealType)2.0;
  p.rho   = (RealType)0.5;
  p.sigma = (RealType)0.5;
  return p;
}

/// Clamp x[i] to [lo[i], hi[i]]
template <int NDIM>
__device__ static inline void nm_clamp(RealType *x, const RealType *lo, const RealType *hi) {
  #pragma unroll
  for (int i = 0; i < NDIM; i++) {
    if (x[i] < lo[i]) x[i] = lo[i];
    if (x[i] > hi[i]) x[i] = hi[i];
  }
}

/// Find indices of best, worst, second-worst vertices
template <int NDIM>
__device__ static inline void nm_find_indices(
    const RealType fvals[NDIM + 1], int *best, int *worst, int *second_worst) {
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
    const RealType simplex[GPU_SIMPLEX_MAX_DIM + 1][GPU_SIMPLEX_MAX_DIM],
    int exclude, RealType centroid[GPU_SIMPLEX_MAX_DIM]) {
  #pragma unroll
  for (int j = 0; j < NDIM; j++) centroid[j] = (RealType)0.0;
  for (int i = 0; i <= NDIM; i++) {
    if (i == exclude) continue;
    #pragma unroll
    for (int j = 0; j < NDIM; j++) centroid[j] += simplex[i][j];
  }
  RealType inv = (RealType)1.0 / (RealType)NDIM;
  #pragma unroll
  for (int j = 0; j < NDIM; j++) centroid[j] *= inv;
}

/// Reflect: xnew = c + scale * (c - xold), then clamp to bounds. (matches NLOPT reflectpt)
template <int NDIM>
__device__ static inline void nm_reflect(
    RealType *xnew, const RealType *c, RealType scale, const RealType *xold,
    const RealType *lo, const RealType *hi) {
  #pragma unroll
  for (int j = 0; j < NDIM; j++) {
    xnew[j] = c[j] + scale * (c[j] - xold[j]);
  }
  nm_clamp<NDIM>(xnew, lo, hi);
}

/**
 * Device-callable Nelder-Mead simplex optimization — runs ONE optimization inline.
 * Can be called from within another kernel (no kernel launch overhead).
 * Algorithm matches NLOPT's nldrmd_minimize_ exactly.
 *
 * Template parameters:
 *   NDIM    — dimension of optimization (e.g., 3, 6, 9, 12)
 *   ObjFunc — functor type with: __device__ RealType operator()(const RealType *x, int ndim)
 *
 * @param x0       [NDIM] initial guess (input)
 * @param lo       [NDIM] lower bounds
 * @param hi       [NDIM] upper bounds
 * @param result   [NDIM] optimized parameters (output)
 * @param objFunc  Objective function functor
 * @param tol      Convergence tolerance (on function value range)
 * @param maxIter  Maximum iterations
 * @param initStep Initial simplex step size (fraction of bound range)
 * @return         Final objective value at best vertex
 */
template <int NDIM, typename ObjFunc>
__device__ RealType nm_optimize(
    const RealType *x0,
    const RealType *lo,
    const RealType *hi,
    RealType *result,
    ObjFunc &objFunc,
    RealType tol,
    int maxIter,
    RealType initStep) {

  NMParams nm = nm_default_params();

  // Initialize simplex: vertex 0 = start point
  RealType simplex[GPU_SIMPLEX_MAX_DIM + 1][GPU_SIMPLEX_MAX_DIM];
  RealType fvals[GPU_SIMPLEX_MAX_DIM + 1];

  // Vertex 0 = starting point
  #pragma unroll
  for (int j = 0; j < NDIM; j++) simplex[0][j] = x0[j];
  nm_clamp<NDIM>(simplex[0], lo, hi);
  fvals[0] = objFunc(simplex[0], NDIM);

  // Vertices 1..NDIM: perturb along each dimension (matching NLOPT initial simplex)
  for (int i = 1; i <= NDIM; i++) {
    #pragma unroll
    for (int j = 0; j < NDIM; j++) simplex[i][j] = simplex[0][j];
    RealType step = initStep * (hi[i-1] - lo[i-1]);
    if (step < (RealType)1e-8) step = (RealType)1e-4;
    RealType newval = simplex[0][i-1] + step;
    // NLOPT-style direction reversal near bounds
    if (newval > hi[i-1]) {
      if (hi[i-1] - simplex[0][i-1] > fabs(step) * (RealType)0.1)
        newval = hi[i-1];
      else
        newval = simplex[0][i-1] - fabs(step);
    }
    if (newval < lo[i-1]) {
      if (simplex[0][i-1] - lo[i-1] > fabs(step) * (RealType)0.1)
        newval = lo[i-1];
      else {
        newval = simplex[0][i-1] + fabs(step);
        if (newval > hi[i-1])
          newval = (RealType)0.5 * ((hi[i-1] - simplex[0][i-1] > simplex[0][i-1] - lo[i-1]
                            ? hi[i-1] : lo[i-1]) + simplex[0][i-1]);
      }
    }
    simplex[i][i-1] = newval;
    nm_clamp<NDIM>(simplex[i], lo, hi);
    fvals[i] = objFunc(simplex[i], NDIM);
  }

  // Main NM loop — matches NLOPT's nldrmd algorithm
  RealType trial[GPU_SIMPLEX_MAX_DIM];
  RealType centroid[GPU_SIMPLEX_MAX_DIM];

  for (int iter = 0; iter < maxIter; iter++) {
    int best, worst, second_worst;
    nm_find_indices<NDIM>(fvals, &best, &worst, &second_worst);

    RealType fl = fvals[best];
    RealType fh = fvals[worst];

    // Check convergence: range of function values
    RealType frange = fh - fl;
    if (frange < tol) break;

    // Centroid of all vertices except worst
    nm_centroid<NDIM>(simplex, worst, centroid);

    // --- Reflection: xcur = c + alpha * (c - xh) ---
    nm_reflect<NDIM>(trial, centroid, nm.alpha, simplex[worst], lo, hi);
    RealType fr = objFunc(trial, NDIM);

    if (fr < fl) {
      // --- Expansion: try c + gamma * (c - xh) ---
      RealType trial_e[GPU_SIMPLEX_MAX_DIM];
      nm_reflect<NDIM>(trial_e, centroid, nm.gamma, simplex[worst], lo, hi);
      RealType fe = objFunc(trial_e, NDIM);

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
    RealType cscale = (fh <= fr) ? -nm.rho : nm.rho;
    RealType trial_c[GPU_SIMPLEX_MAX_DIM];
    nm_reflect<NDIM>(trial_c, centroid, cscale, simplex[worst], lo, hi);
    RealType fc = objFunc(trial_c, NDIM);

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
    result[j] = simplex[best][j];
  return fvals[best];
}

/**
 * GPU Nelder-Mead simplex optimization — batch kernel version.
 * Each thread runs one independent optimization from startPoints arrays.
 *
 * Template parameters:
 *   NDIM    — dimension of optimization (e.g., 3 for Euler angles)
 *   ObjFunc — functor type with: __device__ RealType operator()(const RealType *x, int ndim)
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
    const RealType * __restrict__ startPoints,
    const RealType * __restrict__ lowerBounds,
    const RealType * __restrict__ upperBounds,
    RealType *results,
    RealType *fvals_out,
    ObjFunc objFunc,
    RealType tol,
    int maxIter,
    RealType initStep) {

  int jobIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (jobIdx >= nJobs) return;

  // Load per-job start and bounds
  RealType x0[NDIM], lo[NDIM], hi[NDIM], result[NDIM];
  #pragma unroll
  for (int i = 0; i < NDIM; i++) {
    x0[i] = startPoints[jobIdx * NDIM + i];
    lo[i] = lowerBounds[jobIdx * NDIM + i];
    hi[i] = upperBounds[jobIdx * NDIM + i];
  }

  RealType fval = nm_optimize<NDIM>(x0, lo, hi, result, objFunc, tol, maxIter, initStep);

  // Write output
  #pragma unroll
  for (int j = 0; j < NDIM; j++)
    results[jobIdx * NDIM + j] = result[j];
  fvals_out[jobIdx] = fval;
}

#endif // GPU_SIMPLEX_CUH
