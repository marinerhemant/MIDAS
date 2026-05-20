/* MIDAS_Math.c — vendored copy for the midas-index pip build.
 *
 * Canonical source: FF_HEDM/src/MIDAS_Math.c
 * Vendored 2026-05-19. Contains ONLY the linear-algebra primitives needed
 * by IndexerUnified.c (no NLopt wrapper). Re-sync if the canonical file
 * changes in ways that affect MatrixMultF / RotateAroundZ / MatrixMultF33 /
 * MatrixMult / CalcEtaAngle / sind / cosd / DisplacementInTheSpot.
 */
#include "MIDAS_Math.h"

void MatrixMultF(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

void RotateAroundZ(double v1[3], double alpha, double v2[3]) {
  double cosa = cos(alpha * MIDAS_DEG2RAD);
  double sina = sin(alpha * MIDAS_DEG2RAD);
  double mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};
  MatrixMultF(mat, v1, v2);
}

void MatrixMultF33(double m[3][3], double n[3][3], double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

void MatrixMult(double m[3][3], int v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

void CalcEtaAngle(double y, double z, double *alpha) {
  *alpha = MIDAS_RAD2DEG * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    *alpha = -*alpha;
}

double sind(double x) { return sin(MIDAS_DEG2RAD * x); }
double cosd(double x) { return cos(MIDAS_DEG2RAD * x); }

void DisplacementInTheSpot(double a, double b, double c, double xi, double yi,
                           double zi, double omega, double wedge, double chi,
                           double *Displ_y, double *Displ_z) {
  double sinOme = sind(omega), cosOme = cosd(omega), AcosOme = a * cosOme,
         BsinOme = b * sinOme;
  double XNoW = AcosOme - BsinOme, YNoW = (a * sinOme) + (b * cosOme), ZNoW = c;
  double WedgeRad = MIDAS_DEG2RAD * wedge, CosW = cos(WedgeRad),
         SinW = sin(WedgeRad), XW = XNoW * CosW - ZNoW * SinW, YW = YNoW;
  double ZW = (XNoW * SinW) + (ZNoW * CosW), ChiRad = MIDAS_DEG2RAD * chi,
         CosC = cos(ChiRad), SinC = sin(ChiRad), XC = XW;
  double YC = (CosC * YW) - (SinC * ZW), ZC = (SinC * YW) + (CosC * ZW);
  double IK[3], NormIK;
  IK[0] = xi - XC;
  IK[1] = yi - YC;
  IK[2] = zi - ZC;
  NormIK = sqrt((IK[0] * IK[0]) + (IK[1] * IK[1]) + (IK[2] * IK[2]));
  IK[0] = IK[0] / NormIK;
  IK[1] = IK[1] / NormIK;
  IK[2] = IK[2] / NormIK;
  if (fabs(IK[0]) > 1e-12) {
    *Displ_y = YC - ((XC * IK[1]) / (IK[0]));
    *Displ_z = ZC - ((XC * IK[2]) / (IK[0]));
  }
}
