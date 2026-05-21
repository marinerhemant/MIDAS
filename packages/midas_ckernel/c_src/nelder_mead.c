/* nelder_mead.c — bounded Nelder–Mead simplex (NLopt-free).
 *
 * Standard simplex with reflection/expansion/contraction/shrink and
 * box-bound projection (clamp). Termination on relative function spread
 * (ftol_rel), relative simplex size (xtol_rel), max evaluations, wall-clock
 * budget, or stopval — matching the NLopt LN_NELDERMEAD stopping contract the
 * MIDAS refiners configure.
 *
 * Validated against NLopt's LN_NELDERMEAD on standard test functions (sphere,
 * Rosenbrock, Beale) — see nm_vs_nlopt_test.c.
 */
#include "nelder_mead.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Standard Nelder–Mead coefficients. */
#define NM_ALPHA 1.0 /* reflection */
#define NM_GAMMA 2.0 /* expansion  */
#define NM_RHO   0.5 /* contraction */
#define NM_SIGMA 0.5 /* shrink     */

static double clampd(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/* Project a point into the box [lb, ub] (NULL bound => unbounded). */
static void project(double *x, unsigned n, const double *lb, const double *ub) {
  for (unsigned i = 0; i < n; ++i) {
    double lo = lb ? lb[i] : -DBL_MAX;
    double hi = ub ? ub[i] : DBL_MAX;
    x[i] = clampd(x[i], lo, hi);
  }
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int run_nlopt_optimization(midas_nm_algorithm algo, NLoptConfig *cfg) {
  (void)algo; /* only Nelder–Mead is implemented */
  if (!cfg || cfg->dimension == 0 || !cfg->objective_function ||
      !cfg->initial_guess)
    return MIDAS_NM_INVALID_ARGS;

  const unsigned n = cfg->dimension;
  const unsigned np1 = n + 1;
  const double *lb = cfg->lower_bounds;
  const double *ub = cfg->upper_bounds;
  const int maxeval = cfg->max_evaluations > 0 ? cfg->max_evaluations : 100000;
  const double maxtime = cfg->max_time_seconds;
  const double ftol = cfg->ftol_rel > 0.0 ? cfg->ftol_rel : 1e-6;
  const double xtol = cfg->xtol_rel > 0.0 ? cfg->xtol_rel : 1e-6;
  const double t0 = now_seconds();

  /* Allocate simplex: np1 vertices × n coords, plus per-vertex f and scratch. */
  double *V = (double *)malloc((size_t)np1 * n * sizeof(double));
  double *F = (double *)malloc((size_t)np1 * sizeof(double));
  double *cen = (double *)malloc((size_t)n * sizeof(double));
  double *xr = (double *)malloc((size_t)n * sizeof(double));
  double *xe = (double *)malloc((size_t)n * sizeof(double));
  int *ord = (int *)malloc((size_t)np1 * sizeof(int));
  if (!V || !F || !cen || !xr || !xe || !ord) {
    free(V); free(F); free(cen); free(xr); free(xe); free(ord);
    return MIDAS_NM_FAILURE;
  }

#define VTX(i) (V + (size_t)(i) * n)
  int nev = 0;
  int rc = MIDAS_NM_SUCCESS;

  /* Vertex 0 = projected initial guess. */
  for (unsigned j = 0; j < n; ++j) VTX(0)[j] = cfg->initial_guess[j];
  project(VTX(0), n, lb, ub);

  /* Vertices 1..n: offset along each axis by an initial step. Default step
   * mirrors NLopt: 10% of the finite bound range, else ~5% of |x| (or a small
   * absolute floor). Step toward the interior if the vertex sits on a bound. */
  for (unsigned i = 1; i <= n; ++i) {
    unsigned ax = i - 1;
    for (unsigned j = 0; j < n; ++j) VTX(i)[j] = VTX(0)[j];
    double step;
    if (cfg->step_sizes) {
      step = cfg->step_sizes[ax];
    } else if (lb && ub) {
      step = 0.1 * (ub[ax] - lb[ax]);
    } else {
      double x0 = VTX(0)[ax];
      step = (x0 != 0.0) ? 0.05 * fabs(x0) : 0.00025;
    }
    if (step == 0.0) step = 0.00025;
    double cand = VTX(0)[ax] + step;
    if (ub && cand > ub[ax]) cand = VTX(0)[ax] - step; /* reflect inward */
    VTX(i)[ax] = cand;
    project(VTX(i), n, lb, ub);
  }

  /* Evaluate all vertices. */
  for (unsigned i = 0; i < np1; ++i) {
    F[i] = cfg->objective_function(n, VTX(i), NULL, cfg->obj_data);
    ord[i] = (int)i;
    ++nev;
  }

  /* insertion-sort indices by F ascending */
#define SORT_ORD()                                                            \
  do {                                                                        \
    for (unsigned a = 1; a < np1; ++a) {                                      \
      int key = ord[a];                                                       \
      double fk = F[key];                                                     \
      int b = (int)a - 1;                                                     \
      while (b >= 0 && F[ord[b]] > fk) { ord[b + 1] = ord[b]; --b; }          \
      ord[b + 1] = key;                                                       \
    }                                                                         \
  } while (0)

  SORT_ORD();

  while (1) {
    int best = ord[0], worst = ord[n], second = ord[n - 1];
    double fb = F[best], fw = F[worst];

    /* Convergence is gated on the simplex actually being SMALL (xtol). We do
     * NOT stop on function-spread alone: a simplex can sit on a level set
     * (fspread==0) while still large in x — stopping there returns a non-
     * minimum. Requiring a small simplex lets normal contraction break the
     * level-set symmetry and converge. ftol is reported only as the reason
     * code when both are satisfied. */
    double xsize = 0.0, xscale = 0.0;
    for (unsigned j = 0; j < n; ++j) {
      double bj = VTX(best)[j];
      xscale += fabs(bj);
      for (unsigned i = 0; i < np1; ++i) {
        double d = fabs(VTX((unsigned)ord[i])[j] - bj);
        if (d > xsize) xsize = d;
      }
    }
    if (xsize <= xtol * (xscale / n + 1.0)) {
      double fspread = fabs(fw - fb);
      rc = (fspread <= ftol * (fabs(fb) + fabs(fw)) * 0.5 + 1e-300)
               ? MIDAS_NM_FTOL_REACHED : MIDAS_NM_XTOL_REACHED;
      break;
    }

    if (cfg->has_stopval && fb <= cfg->stopval) { rc = MIDAS_NM_SUCCESS; break; }
    if (nev >= maxeval) { rc = MIDAS_NM_MAXEVAL; break; }
    if (maxtime > 0.0 && (now_seconds() - t0) >= maxtime) {
      rc = MIDAS_NM_MAXTIME; break;
    }

    /* Centroid of all but the worst. */
    for (unsigned j = 0; j < n; ++j) {
      double s = 0.0;
      for (unsigned i = 0; i < n; ++i) s += VTX((unsigned)ord[i])[j];
      cen[j] = s / n;
    }

    /* Reflection. */
    for (unsigned j = 0; j < n; ++j)
      xr[j] = cen[j] + NM_ALPHA * (cen[j] - VTX(worst)[j]);
    project(xr, n, lb, ub);
    double fr = cfg->objective_function(n, xr, NULL, cfg->obj_data); ++nev;

    if (fr < F[best]) {
      /* Expansion. */
      for (unsigned j = 0; j < n; ++j)
        xe[j] = cen[j] + NM_GAMMA * (xr[j] - cen[j]);
      project(xe, n, lb, ub);
      double fe = cfg->objective_function(n, xe, NULL, cfg->obj_data); ++nev;
      if (fe < fr) {
        memcpy(VTX(worst), xe, n * sizeof(double)); F[worst] = fe;
      } else {
        memcpy(VTX(worst), xr, n * sizeof(double)); F[worst] = fr;
      }
    } else if (fr < F[second]) {
      memcpy(VTX(worst), xr, n * sizeof(double)); F[worst] = fr;
    } else {
      /* Contraction (outside if fr<fw, else inside). */
      int outside = (fr < F[worst]);
      for (unsigned j = 0; j < n; ++j) {
        double base = outside ? xr[j] : VTX(worst)[j];
        xe[j] = cen[j] + NM_RHO * (base - cen[j]);
      }
      project(xe, n, lb, ub);
      double fc = cfg->objective_function(n, xe, NULL, cfg->obj_data); ++nev;
      double fcmp = outside ? fr : F[worst];
      if (fc < fcmp) {
        memcpy(VTX(worst), xe, n * sizeof(double)); F[worst] = fc;
      } else {
        /* Shrink toward best. */
        for (unsigned i = 0; i < np1; ++i) {
          if ((int)i == best) continue;
          for (unsigned j = 0; j < n; ++j)
            VTX(i)[j] = VTX(best)[j] + NM_SIGMA * (VTX(i)[j] - VTX(best)[j]);
          project(VTX(i), n, lb, ub);
          F[i] = cfg->objective_function(n, VTX(i), NULL, cfg->obj_data); ++nev;
        }
      }
    }
    SORT_ORD();
  }

  /* Write back the best vertex. */
  int best = ord[0];
  for (unsigned j = 0; j < n; ++j) cfg->initial_guess[j] = VTX(best)[j];
  cfg->min_function_val = F[best];

  free(V); free(F); free(cen); free(xr); free(xe); free(ord);
  return rc;
#undef VTX
#undef SORT_ORD
}
