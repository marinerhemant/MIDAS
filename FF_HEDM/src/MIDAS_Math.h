#ifndef MIDAS_MATH_H
#define MIDAS_MATH_H

#include <math.h>
#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- Shared constants ---- */
#ifndef MIDAS_DEG2RAD
#define MIDAS_DEG2RAD 0.0174532925199433
#endif
#ifndef MIDAS_RAD2DEG
#define MIDAS_RAD2DEG 57.2957795130823
#endif

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

/**
 * MatrixMultF – multiply a 3×3 matrix by a 3-vector: r = m · v
 */
void MatrixMultF(double m[3][3], double v[3], double r[3]);

/**
 * RotateAroundZ – rotate 3-vector v1 by angle alpha (degrees) around Z axis.
 */
void RotateAroundZ(double v1[3], double alpha, double v2[3]);

/**
 * MatrixMultF33 – multiply two 3×3 matrices: res = m · n
 */
void MatrixMultF33(double m[3][3], double n[3][3], double res[3][3]);

/**
 * MatrixMult – multiply a 3×3 double matrix by a 3-element int vector: r = m ·
 * v
 */
void MatrixMult(double m[3][3], int v[3], double r[3]);

/**
 * CalcEtaAngle – compute azimuthal angle eta (degrees) from y,z detector
 * coords.
 */
void CalcEtaAngle(double y, double z, double *alpha);

/**
 * sind/cosd – sin/cos taking degrees.
 */
double sind(double x);
double cosd(double x);

/**
 * DisplacementInTheSpot – compute detector displacement for a voxel
 * at position (a,b,c) given diffraction geometry.
 * For no wedge/chi correction, pass wedge=0 and chi=0.
 */
void DisplacementInTheSpot(double a, double b, double c, double xi, double yi,
                           double zi, double omega, double wedge, double chi,
                           double *Displ_y, double *Displ_z);

#endif // MIDAS_MATH_H
