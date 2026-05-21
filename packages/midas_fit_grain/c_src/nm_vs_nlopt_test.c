/* nm_vs_nlopt_test.c — validate the vendored Nelder–Mead against NLopt's
 * LN_NELDERMEAD on standard test functions. Build (where NLopt is available):
 *
 *   gcc -O2 -DHAVE_NLOPT nm_vs_nlopt_test.c nelder_mead.c -lnlopt -lm -o nmtest
 *   ./nmtest
 *
 * Without NLopt, build with just nelder_mead.c to sanity-check the vendored
 * optimizer reaches the known minima:
 *   gcc -O2 nm_vs_nlopt_test.c nelder_mead.c -lm -o nmtest
 */
#include "nelder_mead.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#ifdef HAVE_NLOPT
#include <nlopt.h>
#endif

static int g_calls;

static double sphere(unsigned n, const double *x, double *g, void *d) {
  (void)g; (void)d; ++g_calls;
  double s = 0; for (unsigned i = 0; i < n; ++i) s += (x[i]-1.0)*(x[i]-1.0);
  return s;
}
static double rosen(unsigned n, const double *x, double *g, void *d) {
  (void)g; (void)d; (void)n; ++g_calls;
  double a = 1.0 - x[0], b = x[1] - x[0]*x[0];
  return a*a + 100.0*b*b;
}
static double beale(unsigned n, const double *x, double *g, void *d) {
  (void)g; (void)d; (void)n; ++g_calls;
  double X = x[0], Y = x[1];
  double t1 = 1.5 - X + X*Y;
  double t2 = 2.25 - X + X*Y*Y;
  double t3 = 2.625 - X + X*Y*Y*Y;
  return t1*t1 + t2*t2 + t3*t3;
}

typedef struct { const char *name; midas_obj_func f; double x0[4]; double sol[4];
                 unsigned n; double lb[4]; double ub[4]; int bounded; } Case;

static void run_case(const Case *c) {
  /* vendored */
  double xv[4]; memcpy(xv, c->x0, c->n*sizeof(double));
  NLoptConfig cfg; memset(&cfg, 0, sizeof(cfg));
  cfg.objective_function = c->f; cfg.dimension = c->n; cfg.initial_guess = xv;
  cfg.ftol_rel = 1e-8; cfg.xtol_rel = 1e-8; cfg.max_evaluations = 5000;
  cfg.max_time_seconds = 30;
  if (c->bounded) { cfg.lower_bounds = (double*)c->lb; cfg.upper_bounds = (double*)c->ub; }
  g_calls = 0; run_nlopt_optimization(MIDAS_LN_NELDERMEAD, &cfg);
  int vcalls = g_calls;
  double verr = 0; for (unsigned i=0;i<c->n;++i) verr += fabs(xv[i]-c->sol[i]);

  printf("%-10s vendored: f=%.3e |x-x*|=%.3e (%d evals)", c->name,
         cfg.min_function_val, verr, vcalls);
#ifdef HAVE_NLOPT
  double xn[4]; memcpy(xn, c->x0, c->n*sizeof(double)); double fn=0;
  nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, c->n);
  if (c->bounded){ nlopt_set_lower_bounds(opt,c->lb); nlopt_set_upper_bounds(opt,c->ub);}
  nlopt_set_min_objective(opt, c->f, NULL);
  nlopt_set_ftol_rel(opt,1e-8); nlopt_set_xtol_rel(opt,1e-8);
  nlopt_set_maxeval(opt,5000); nlopt_set_maxtime(opt,30);
  g_calls = 0; nlopt_optimize(opt, xn, &fn); int ncalls=g_calls;
  double nerr=0; for(unsigned i=0;i<c->n;++i) nerr += fabs(xn[i]-c->sol[i]);
  double dx=0; for(unsigned i=0;i<c->n;++i) dx += fabs(xn[i]-xv[i]);
  printf("  | nlopt: f=%.3e |x-x*|=%.3e (%d evals) | |x_v-x_n|=%.3e",
         fn, nerr, ncalls, dx);
  nlopt_destroy(opt);
#endif
  printf("\n");
}

int main(void) {
  Case cases[] = {
    {"sphere2", sphere, {-3,4}, {1,1}, 2, {-10,-10}, {10,10}, 1},
    {"rosen",   rosen,  {-1.2,1}, {1,1}, 2, {-5,-5}, {5,5}, 1},
    {"beale",   beale,  {0,0}, {3,0.5}, 2, {-4.5,-4.5}, {4.5,4.5}, 1},
    {"sphere2u",sphere, {-3,4}, {1,1}, 2, {0,0}, {0,0}, 0},
    {"rosenu",  rosen,  {-1.2,1}, {1,1}, 2, {0,0}, {0,0}, 0},
  };
  for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) run_case(&cases[i]);
  return 0;
}
