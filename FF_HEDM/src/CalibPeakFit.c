// CalibPeakFit.c — Standalone pseudo-Voigt peak fitting extracted from
// CalibrationCore.c for use in IntegratorZarrOMP.
//
// Same intensity-weighted pseudo-Voigt fitting as CI's calib_fit_peak_shape.
// No M-step dependencies.

#include "CalibPeakFit.h"
#include <math.h>
#include <string.h>
#include <nlopt.h>

typedef struct {
  int NrPtsForFit;
  double *Rs;
  double *PeakShape;
} CpfProfileData;

// NLopt objective for singlet pseudo-Voigt profile.
// x[0]=Rcen, x[1]=Mu, x[2]=Gamma(FWHM), x[3]=Imax, x[4]=BG
static double cpf_pv_obj(unsigned n, const double *x,
                         double *grad, void *data) {
  CpfProfileData *f = (CpfProfileData *)data;
  int N = f->NrPtsForFit;
  double *Rs = f->Rs;
  double *PS = f->PeakShape;
  double Rcen = x[0], Mu = x[1], Gamma = x[2], Imax = x[3], BG = x[4];
  double C0 = 4.0 * log(2.0);
  double G2 = Gamma * Gamma, G3 = G2 * Gamma;
  double invG2 = 1.0 / G2;
  double Total = 0;

  if (grad) memset(grad, 0, n * sizeof(double));

  for (int i = 0; i < N; i++) {
    double dr = Rs[i] - Rcen;
    double dr2 = dr * dr;
    double denom = 1.0 + 4.0 * dr2 * invG2;
    double L = 1.0 / denom;
    double G = exp(-C0 * dr2 * invG2);
    double CalcI = BG + Imax * (Mu * L + (1.0 - Mu) * G);
    double diff = CalcI - PS[i];
    Total += diff * diff;

    if (grad) {
      double c = 2.0 * diff;
      double dLdc = 8.0 * dr / (G2 * denom * denom);
      double dGdc = G * 2.0 * C0 * dr * invG2;
      grad[0] += c * Imax * (Mu * dLdc + (1.0 - Mu) * dGdc);
      grad[1] += c * Imax * (L - G);
      double dLdGam = 8.0 * dr2 / (G3 * denom * denom);
      double dGdGam = G * 2.0 * C0 * dr2 / G3;
      grad[2] += c * Imax * (Mu * dLdGam + (1.0 - Mu) * dGdGam);
      grad[3] += c * (Mu * L + (1.0 - Mu) * G);
      grad[4] += c;
    }
  }
  return Total;
}

void cpf_fit_peak(int NrPtsForFit, double *Rs, double *PeakShape,
                  double *Rfit, double *fitSNR, double Rstep, double Rmean) {
  double x[5], xl[5], xu[5];
  CpfProfileData f_data;
  f_data.NrPtsForFit = NrPtsForFit;
  f_data.Rs = Rs;
  f_data.PeakShape = PeakShape;

  double BG0 = (PeakShape[0] + PeakShape[NrPtsForFit - 1]) / 2;
  if (BG0 < 0) BG0 = 0;
  double MaxI = -1e20;
  for (int i = 0; i < NrPtsForFit; i++)
    if (PeakShape[i] > MaxI) MaxI = PeakShape[i];

  x[0] = Rmean;  xl[0] = Rs[0];  xu[0] = Rs[NrPtsForFit - 1];
  x[1] = 0.5;    xl[1] = 0;      xu[1] = 1;
  double GammaGuess = Rstep * 3;
  x[2] = GammaGuess;  xl[2] = Rstep / 2;  xu[2] = Rstep * NrPtsForFit / 2;
  x[3] = MaxI - BG0;  xl[3] = (MaxI - BG0) / 100;  xu[3] = MaxI * 1.5;
  x[4] = BG0;         xl[4] = 0;  xu[4] = (BG0 > 0) ? BG0 * 1.5 : MaxI * 0.5;

  double x_init[5];
  memcpy(x_init, x, 5 * sizeof(double));

  nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 5);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, cpf_pv_obj, &f_data);
  nlopt_set_xtol_rel(opt, 1e-8);
  nlopt_set_ftol_rel(opt, 1e-8);
  nlopt_set_maxeval(opt, 10000);
  double minf;
  int rc = nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);

  if (rc < 0) {
    memcpy(x, x_init, 5 * sizeof(double));
    opt = nlopt_create(NLOPT_LN_NELDERMEAD, 5);
    nlopt_set_lower_bounds(opt, xl);
    nlopt_set_upper_bounds(opt, xu);
    nlopt_set_min_objective(opt, cpf_pv_obj, &f_data);
    nlopt_set_xtol_rel(opt, 1e-8);
    nlopt_set_ftol_rel(opt, 1e-8);
    nlopt_set_maxeval(opt, 10000);
    nlopt_optimize(opt, x, &minf);
    nlopt_destroy(opt);
  }

  *Rfit = x[0];
  double rmsResid = sqrt(minf / NrPtsForFit);
  *fitSNR = (rmsResid > 0) ? x[3] / rmsResid : 1.0;
}
