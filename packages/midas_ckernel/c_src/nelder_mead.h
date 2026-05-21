/* nelder_mead.h — self-contained, NLopt-free Nelder–Mead simplex optimizer.
 *
 * Drop-in replacement for the single NLopt entry point the MIDAS refiners use
 * (`run_nlopt_optimization` with `NLOPT_LN_NELDERMEAD`), so the unified C
 * refiner (FitUnified.c) builds with no external NLopt dependency — matching
 * the pip-installable, dependency-free convention of midas-index's c_src.
 *
 * The objective signature is byte-compatible with NLopt's `nlopt_func`
 * (gradient pointer is always NULL for the derivative-free simplex), so the
 * refiner objective functions need no edits.
 */
#ifndef MIDAS_NELDER_MEAD_H
#define MIDAS_NELDER_MEAD_H

#include <math.h>
#include <stddef.h>

/* ---- NLopt-compatible objective + result codes (subset we use) ---- */
typedef double (*midas_obj_func)(unsigned n, const double *x, double *grad,
                                 void *data);

/* Result codes mirror NLopt's sign convention: >0 success, <0 error. */
enum {
  MIDAS_NM_SUCCESS = 1,        /* generic success */
  MIDAS_NM_FTOL_REACHED = 3,   /* ftol_rel satisfied */
  MIDAS_NM_XTOL_REACHED = 4,   /* xtol_rel satisfied */
  MIDAS_NM_MAXEVAL = 5,        /* hit max function evaluations */
  MIDAS_NM_MAXTIME = 6,        /* hit wall-clock budget */
  MIDAS_NM_FAILURE = -1,
  MIDAS_NM_INVALID_ARGS = -2,
};

/* Optimizer configuration. Field names/semantics intentionally mirror the
 * legacy NLoptConfig so call sites are unchanged when the vendored header
 * replaces the NLopt one. `initial_guess` is IN/OUT: on return it holds the
 * best point found. */
typedef struct {
  midas_obj_func objective_function;
  void *obj_data;
  unsigned int dimension;
  double *lower_bounds;        /* length `dimension`, or NULL (= -inf) */
  double *upper_bounds;        /* length `dimension`, or NULL (= +inf) */
  double *initial_guess;       /* length `dimension`, IN/OUT */
  double *step_sizes;          /* initial simplex step, or NULL (auto) */
  double min_function_val;     /* OUT: best objective value */
  double ftol_rel;
  double xtol_rel;
  int max_evaluations;         /* <=0 => default cap */
  double max_time_seconds;     /* <=0 => no time limit */
  unsigned int population;     /* ignored by NM (kept for ABI parity) */
  double stopval;
  int has_stopval;
} NLoptConfig;

/* Algorithm selector kept as an int for source compatibility; only the
 * Nelder–Mead path is implemented (the only algo the refiners request). */
typedef int midas_nm_algorithm;
#define MIDAS_LN_NELDERMEAD 0

/* Run the bounded Nelder–Mead simplex. Writes the optimum into
 * config->initial_guess and config->min_function_val. Returns a result code
 * (>0 success, <0 error). `algo` is accepted for signature parity and ignored
 * (always Nelder–Mead). */
int run_nlopt_optimization(midas_nm_algorithm algo, NLoptConfig *config);

#endif /* MIDAS_NELDER_MEAD_H */
