#ifndef MIDAS_MATH_H
#define MIDAS_MATH_H

#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  nlopt_func objective_function;
  void *obj_data;
  unsigned int dimension;
  double *lower_bounds;
  double *upper_bounds;
  double *initial_guess;
  double *step_sizes; // Array of length 'dimension', or NULL to skip
  double min_function_val;
  double ftol_rel;
  double xtol_rel;
  int max_evaluations;
  double max_time_seconds;
  unsigned int population;
  double stopval;
  int has_stopval;
} NLoptConfig;

/**
 * run_nlopt_optimization
 * ----------------------
 * Wrapper around NLopt setup parameters to deduplicate configuration.
 * Returns the final objective value found by the optimizer upon success.
 * 'config->min_function_val' will be updated with the output.
 * Returns NLopt result code (< 0 indicates error).
 */
int run_nlopt_optimization(nlopt_algorithm algo, NLoptConfig *config);

#endif // MIDAS_MATH_H
