/* MIDAS_Math.h — vendored copy for the midas-index pip build.
 *
 * Canonical source: FF_HEDM/src/MIDAS_Math.h
 * Vendored 2026-05-19. Contains ONLY the linear-algebra primitives that
 * IndexerUnified.c needs (no NLopt dependency). Re-sync if the canonical
 * file changes in ways that affect MatrixMultF / RotateAroundZ /
 * MatrixMultF33 / CalcEtaAngle / DisplacementInTheSpot.
 *
 * Why vendored: the full FF_HEDM/src/MIDAS_Math.h pulls <nlopt.h> for an
 * unrelated optimisation wrapper. Forcing pip users to install libnlopt-dev
 * just to compile the unified indexer is a poor UX. This stripped copy
 * is functionally identical for the primitives the indexer uses.
 */
#ifndef MIDAS_MATH_H
#define MIDAS_MATH_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MIDAS_DEG2RAD
#define MIDAS_DEG2RAD (M_PI / 180.0)
#endif
#ifndef MIDAS_RAD2DEG
#define MIDAS_RAD2DEG (180.0 / M_PI)
#endif

void MatrixMultF(double m[3][3], double v[3], double r[3]);
void RotateAroundZ(double v1[3], double alpha, double v2[3]);
void MatrixMultF33(double m[3][3], double n[3][3], double res[3][3]);
void MatrixMult(double m[3][3], int v[3], double r[3]);
void CalcEtaAngle(double y, double z, double *alpha);
double sind(double x);
double cosd(double x);
void DisplacementInTheSpot(double a, double b, double c, double xi, double yi,
                           double zi, double omega, double wedge, double chi,
                           double *Displ_y, double *Displ_z);

#endif /* MIDAS_MATH_H */
