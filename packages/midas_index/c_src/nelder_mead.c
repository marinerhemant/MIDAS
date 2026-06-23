/* nelder_mead.c — bounded Nelder–Mead simplex, an EXACT reimplementation of
 * NLopt's LN_NELDERMEAD (nldrmd.c, Steven G. Johnson, MIT-licensed), so the
 * dependency-free MIDAS refiner reproduces the legacy NLopt-based refiners.
 *
 * Mirrors, in behavior, NLopt 2.x src/algs/neldermead/nldrmd.c:
 *   - Richardson & Kuester (1973) bound handling: reflected points are PINNED
 *     to [lb,ub]; a reflection that coincides (close(): 1e-13 rel) with the
 *     centroid or the reflected vertex TERMINATES (XTOL) instead of clamping
 *     and wandering along the bound face.
 *   - initial simplex from xstep with NLopt's >ub / <lb fallback (the *0.1
 *     "too close to bound, go the other way" rule).
 *   - default initial step = NLopt nlopt_set_default_initial_step() heuristic
 *     ((ub-lb)*0.25, (ub-x)*0.75, (x-lb)*0.75, …) when step_sizes is NULL —
 *     exactly what the legacy refiners get (they pass no explicit step).
 *   - centroid excludes the worst vertex; ftol test (relstop) then x test
 *     (L1: Σ maxradius < xtol_rel · Σ|c|) per NLopt nlopt_stop_{ftol,x}.
 *   - reflect(α=1)/expand(γ=2)/contract(β=0.5, accept iff fc<fr && fc<fh)/
 *     shrink(δ=0.5)+restart.
 *   - the returned x is the BEST point ever evaluated (NLopt CHECK_EVAL), not
 *     necessarily a final simplex vertex.
 *
 * The red-black tree NLopt uses is only an O(log n) accelerator for min/max/
 * 2nd-max; with the tiny n of the refiner we scan instead. fp64 throughout.
 */
#include "nelder_mead.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* NLopt heuristic strategy constants (nldrmd.c:35). */
#define NM_ALPHA 1.0 /* reflection */
#define NM_BETA  0.5 /* contraction */
#define NM_GAMMA 2.0 /* expansion  */
#define NM_DELTA 0.5 /* shrink     */

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* NLopt close(): a and b approximately equal to fp precision (nldrmd.c:47). */
static int nm_close(double a, double b) {
  return (fabs(a - b) <= 1e-13 * (fabs(a) + fabs(b)));
}

/* NLopt relstop() (stop.c:81): converged if |new-old| < abstol, or
 * < reltol·(|new|+|old|)/2, or (reltol>0 && new==old). */
static int relstop(double vold, double vnew, double reltol, double abstol) {
  return (fabs(vnew - vold) < abstol ||
          fabs(vnew - vold) < reltol * (fabs(vnew) + fabs(vold)) * 0.5 ||
          (reltol > 0.0 && vnew == vold));
}

/* NLopt reflectpt() (nldrmd.c:63): xnew = c + scale*(c - xold), pinned to
 * [lb,ub]. Returns 0 if xnew coincides with c or xold (→ caller terminates),
 * 1 otherwise. */
static int reflectpt(unsigned n, double *xnew, const double *c, double scale,
                     const double *xold, const double *lb, const double *ub) {
  int equalc = 1, equalold = 1;
  for (unsigned i = 0; i < n; ++i) {
    double newx = c[i] + scale * (c[i] - xold[i]);
    if (lb && newx < lb[i]) newx = lb[i];
    if (ub && newx > ub[i]) newx = ub[i];
    equalc = equalc && nm_close(newx, c[i]);
    equalold = equalold && nm_close(newx, xold[i]);
    xnew[i] = newx;
  }
  return !(equalc || equalold);
}

/* NLopt nlopt_set_default_initial_step() heuristic (options.c:912). lb/ub may
 * be NULL (=> ±inf, i.e. that bound is "not finite"). */
static void default_initial_step(unsigned n, const double *x, const double *lb,
                                 const double *ub, double *dx) {
  for (unsigned i = 0; i < n; ++i) {
    double step = DBL_MAX; /* stand-in for HUGE_VAL */
    int ubf = (ub != NULL);
    int lbf = (lb != NULL);
    if (ubf && lbf && (ub[i] - lb[i]) * 0.25 < step && ub[i] > lb[i])
      step = (ub[i] - lb[i]) * 0.25;
    if (ubf && ub[i] - x[i] < step && ub[i] > x[i])
      step = (ub[i] - x[i]) * 0.75;
    if (lbf && x[i] - lb[i] < step && x[i] > lb[i])
      step = (x[i] - lb[i]) * 0.75;
    if (step == DBL_MAX) {
      if (ubf && fabs(ub[i] - x[i]) < fabs(step)) step = (ub[i] - x[i]) * 1.1;
      if (lbf && fabs(x[i] - lb[i]) < fabs(step)) step = (x[i] - lb[i]) * 1.1;
    }
    if (step == DBL_MAX || fabs(step) < 1e-300) step = x[i];
    if (step == DBL_MAX || step == 0.0) step = 1.0;
    dx[i] = step;
  }
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
  const double ftol_rel = cfg->ftol_rel > 0.0 ? cfg->ftol_rel : 0.0;
  const double xtol_rel = cfg->xtol_rel > 0.0 ? cfg->xtol_rel : 0.0;
  const double minf_max = cfg->has_stopval ? cfg->stopval : -DBL_MAX;
  const double t0 = now_seconds();
  const double ninv = 1.0 / (double)n;

  midas_obj_func f = cfg->objective_function;
  void *fdata = cfg->obj_data;

  double *pts = (double *)malloc((size_t)np1 * n * sizeof(double));
  double *fval = (double *)malloc((size_t)np1 * sizeof(double));
  double *cen = (double *)malloc((size_t)n * sizeof(double));
  double *xcur = (double *)malloc((size_t)n * sizeof(double));
  double *xstep = (double *)malloc((size_t)n * sizeof(double));
  double *xbest = (double *)malloc((size_t)n * sizeof(double));
  if (!pts || !fval || !cen || !xcur || !xstep || !xbest) {
    free(pts); free(fval); free(cen); free(xcur); free(xstep); free(xbest);
    return MIDAS_NM_FAILURE;
  }
#define VTX(i) (pts + (size_t)(i) * n)

  int nev = 0;
  int rc = MIDAS_NM_XTOL_REACHED;
  double fbest;

  if (cfg->step_sizes)
    for (unsigned j = 0; j < n; ++j) xstep[j] = cfg->step_sizes[j];
  else
    default_initial_step(n, cfg->initial_guess, lb, ub, xstep);

  /* CHECK_EVAL: update best-ever (xbest,fbest); honor stopval/maxeval/maxtime. */
#define CHECK_EVAL(xc, fc)                                                    \
  do {                                                                        \
    ++nev;                                                                    \
    if ((fc) <= fbest) {                                                      \
      fbest = (fc);                                                           \
      memcpy(xbest, (xc), n * sizeof(double));                                \
      if (fbest < minf_max) { rc = MIDAS_NM_SUCCESS; goto done; }             \
    }                                                                         \
    if (nev >= maxeval) { rc = MIDAS_NM_MAXEVAL; goto done; }                 \
    if (maxtime > 0.0 && (now_seconds() - t0) >= maxtime) {                   \
      rc = MIDAS_NM_MAXTIME; goto done;                                       \
    }                                                                         \
  } while (0)

  /* vertex 0 = initial guess (evaluated once). */
  memcpy(VTX(0), cfg->initial_guess, n * sizeof(double));
  fval[0] = f(n, VTX(0), NULL, fdata);
  fbest = fval[0];
  memcpy(xbest, VTX(0), n * sizeof(double));
  ++nev;
  if (fbest < minf_max) { rc = MIDAS_NM_SUCCESS; goto done; }

  /* vertices 1..n: offset along each axis by xstep, NLopt bound fallback. */
  for (unsigned i = 1; i <= n; ++i) {
    unsigned ax = i - 1;
    double x0 = cfg->initial_guess[ax];
    memcpy(VTX(i), cfg->initial_guess, n * sizeof(double));
    double p = x0 + xstep[ax];
    if (ub && p > ub[ax]) {
      if (ub[ax] - x0 > fabs(xstep[ax]) * 0.1) p = ub[ax];
      else p = x0 - fabs(xstep[ax]);
    }
    if (lb && p < lb[ax]) {
      if (x0 - lb[ax] > fabs(xstep[ax]) * 0.1) {
        p = lb[ax];
      } else {
        p = x0 + fabs(xstep[ax]);
        if (ub && p > ub[ax])
          p = 0.5 * (((ub[ax] - x0 > x0 - lb[ax]) ? ub[ax] : lb[ax]) + x0);
      }
    }
    VTX(i)[ax] = p;
    if (nm_close(p, x0)) { rc = MIDAS_NM_FAILURE; goto done; }
    fval[i] = f(n, VTX(i), NULL, fdata);
    CHECK_EVAL(VTX(i), fval[i]);
  }

restart:
  while (1) {
    int low = 0, high = 0;
    for (unsigned i = 1; i < np1; ++i) {
      if (fval[i] < fval[low]) low = (int)i;
      if (fval[i] > fval[high]) high = (int)i;
    }
    int second = -1;
    for (unsigned i = 0; i < np1; ++i) {
      if ((int)i == high) continue;
      if (second < 0 || fval[i] > fval[second]) second = (int)i;
    }
    double fl = fval[low], fh = fval[high];

    if (relstop(fl, fh, ftol_rel, 0.0)) { rc = MIDAS_NM_FTOL_REACHED; goto done; }

    /* centroid of all but the highest. */
    memset(cen, 0, n * sizeof(double));
    for (unsigned i = 0; i < np1; ++i) {
      if ((int)i == high) continue;
      for (unsigned j = 0; j < n; ++j) cen[j] += VTX(i)[j];
    }
    for (unsigned j = 0; j < n; ++j) cen[j] *= ninv;

    /* x test: Σ maxradius < xtol_rel · Σ|c| (NLopt L1, unit weights). */
    for (unsigned j = 0; j < n; ++j) xcur[j] = 0.0;
    for (unsigned i = 0; i < np1; ++i)
      for (unsigned j = 0; j < n; ++j) {
        double dxj = fabs(VTX(i)[j] - cen[j]);
        if (dxj > xcur[j]) xcur[j] = dxj;
      }
    {
      double dnorm = 0.0, cnorm = 0.0;
      for (unsigned j = 0; j < n; ++j) { dnorm += xcur[j]; cnorm += fabs(cen[j]); }
      if (xtol_rel > 0.0 && dnorm < xtol_rel * cnorm) {
        rc = MIDAS_NM_XTOL_REACHED; goto done;
      }
    }

    /* reflection: xcur = c + α(c - xh). */
    if (!reflectpt(n, xcur, cen, NM_ALPHA, VTX(high), lb, ub)) {
      rc = MIDAS_NM_XTOL_REACHED; goto done;
    }
    double fr = f(n, xcur, NULL, fdata);
    CHECK_EVAL(xcur, fr);

    if (fr < fl) {
      /* expansion: reflect xh through c with γ, into VTX(high). */
      if (!reflectpt(n, VTX(high), cen, NM_GAMMA, VTX(high), lb, ub)) {
        rc = MIDAS_NM_XTOL_REACHED; goto done;
      }
      fh = f(n, VTX(high), NULL, fdata);
      CHECK_EVAL(VTX(high), fh);
      if (fh >= fr) { /* expansion didn't improve → keep reflection */
        fh = fr;
        memcpy(VTX(high), xcur, n * sizeof(double));
      }
      fval[high] = fh;
    } else if (second >= 0 && fr < fval[second]) {
      memcpy(VTX(high), xcur, n * sizeof(double));
      fval[high] = fr;
    } else {
      /* contraction: -β inside (reflection no better than worst) else +β. */
      double scale = (fh <= fr) ? -NM_BETA : NM_BETA;
      if (!reflectpt(n, xcur, cen, scale, VTX(high), lb, ub)) {
        rc = MIDAS_NM_XTOL_REACHED; goto done;
      }
      double fc = f(n, xcur, NULL, fdata);
      CHECK_EVAL(xcur, fc);
      if (fc < fr && fc < fh) {
        memcpy(VTX(high), xcur, n * sizeof(double));
        fval[high] = fc;
      } else {
        /* shrink all but the lowest toward the lowest, then restart. */
        for (unsigned i = 0; i < np1; ++i) {
          if ((int)i == low) continue;
          if (!reflectpt(n, VTX(i), VTX(low), -NM_DELTA, VTX(i), lb, ub)) {
            rc = MIDAS_NM_XTOL_REACHED; goto done;
          }
          fval[i] = f(n, VTX(i), NULL, fdata);
          CHECK_EVAL(VTX(i), fval[i]);
        }
        goto restart;
      }
    }
  }

done:
  /* NLopt returns the best point ever evaluated. */
  memcpy(cfg->initial_guess, xbest, n * sizeof(double));
  cfg->min_function_val = fbest;
  free(pts); free(fval); free(cen); free(xcur); free(xstep); free(xbest);
  return rc;
#undef VTX
#undef CHECK_EVAL
}
