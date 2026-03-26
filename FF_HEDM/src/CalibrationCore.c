//
// CalibrationCore.c — Shared calibration optimization functions
//
// See CalibrationCore.h for API documentation.
// Geometry computations delegate to dg_pixel_to_REta() from DetectorGeometry.h.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "CalibrationCore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1E-12

// ── Per-evaluation trace file ─────────────────────────────────────

static FILE *calib_trace_fp = NULL;

void calib_set_trace_file(const char *filename) {
  calib_close_trace_file();
  if (filename == NULL) return;
  calib_trace_fp = fopen(filename, "w");
  if (calib_trace_fp) {
    fprintf(calib_trace_fp,
            "Eval,Objective,MeanStrain_ue,Lsd,ybc,zbc,ty,tz,"
            "p0,p1,p2,p3,p4,p5,p6\n");
    fflush(calib_trace_fp);
  }
}

void calib_close_trace_file(void) {
  if (calib_trace_fp) {
    fclose(calib_trace_fp);
    calib_trace_fp = NULL;
  }
}

// ── qsort comparator ──────────────────────────────────────────────

int calib_cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

// ── Intensity-weighted centroid ────────────────────────────────────

void calib_weighted_mean(int nIndices, int *NrEachIndexBin, int **Indices,
                         double *Average, double *R, double *Eta,
                         double *RMean, double *EtaMean) {
  int i;
  for (i = 0; i < nIndices; i++) {
    int j;
    int NrIndicesThis = NrEachIndexBin[i];
    if (NrIndicesThis == 0) {
      RMean[i] = 0;
      EtaMean[i] = 0;
      continue;
    }
    double TotIntensity = 0;
    double totRIntensity = 0;
    double totEtaIntensity = 0;
    for (j = 0; j < NrIndicesThis; j++) {
      int idx = Indices[i][j];
      double intensity = Average[idx];
      totRIntensity += R[idx] * intensity;
      totEtaIntensity += Eta[idx] * intensity;
      TotIntensity += intensity;
    }
    if (TotIntensity > 0) {
      RMean[i] = totRIntensity / TotIntensity;
      EtaMean[i] = totEtaIntensity / TotIntensity;
    } else {
      RMean[i] = 0;
      EtaMean[i] = 0;
    }
  }
}

// ── Peak profile fitting: pseudo-Voigt singlet ─────────────────────

// NLopt objective for singlet pseudo-Voigt profile.
// x[0]=Rcen, x[1]=Mu, x[2]=Gamma(FWHM), x[3]=Imax, x[4]=BG
static double calib_pv_singlet_obj(unsigned n, const double *x,
                                   double *grad, void *f_data_trial) {
  struct calib_profile_data *f = (struct calib_profile_data *)f_data_trial;
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

void calib_fit_peak_shape(int NrPtsForFit, double *Rs, double *PeakShape,
                          double *Rfit, double *fitSNR,
                          double Rstep, double Rmean) {
  unsigned n = 5;
  double x[5], xl[5], xu[5];
  struct calib_profile_data f_data;
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

  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = calib_pv_singlet_obj;
  config.obj_data = &f_data;
  config.initial_guess = x;
  config.max_evaluations = 10000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-8;
  config.xtol_rel = 1e-8;

  double x_init[5];
  memcpy(x_init, x, n * sizeof(double));
  int rc = run_nlopt_optimization(NLOPT_LD_LBFGS, &config);
  if (rc < 0) {
    memcpy(x, x_init, n * sizeof(double));
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  }
  *Rfit = x[0];
  double rmsResid = sqrt(config.min_function_val / NrPtsForFit);
  *fitSNR = (rmsResid > 0) ? x[3] / rmsResid : 1.0;
}

// ── Peak profile fitting: pseudo-Voigt doublet ─────────────────────

// NLopt objective for doublet pseudo-Voigt profile.
// x[0]=Rcen1, x[1]=Rcen2, x[2]=Mu, x[3]=Gamma1, x[4]=Imax1,
// x[5]=Gamma2, x[6]=Imax2, x[7]=BG
static double calib_pv_doublet_obj(unsigned n, const double *x,
                                   double *grad, void *f_data_trial) {
  struct calib_profile_data *f = (struct calib_profile_data *)f_data_trial;
  int N = f->NrPtsForFit;
  double *Rs = f->Rs;
  double *PS = f->PeakShape;
  double Rcen1 = x[0], Rcen2 = x[1], Mu = x[2];
  double Gamma1 = x[3], Imax1 = x[4];
  double Gamma2 = x[5], Imax2 = x[6];
  double BG = x[7];
  double C0 = 4.0 * log(2.0);
  double G1_2 = Gamma1 * Gamma1, G1_3 = G1_2 * Gamma1;
  double G2_2 = Gamma2 * Gamma2, G2_3 = G2_2 * Gamma2;
  double invG1_2 = 1.0 / G1_2, invG2_2 = 1.0 / G2_2;
  double Total = 0;

  if (grad) memset(grad, 0, n * sizeof(double));

  for (int i = 0; i < N; i++) {
    double dr1 = Rs[i] - Rcen1, dr2 = Rs[i] - Rcen2;
    double dr1_2 = dr1 * dr1, dr2_2 = dr2 * dr2;
    double den1 = 1.0 + 4.0 * dr1_2 * invG1_2;
    double den2 = 1.0 + 4.0 * dr2_2 * invG2_2;
    double L1 = 1.0 / den1, Gau1 = exp(-C0 * dr1_2 * invG1_2);
    double L2 = 1.0 / den2, Gau2 = exp(-C0 * dr2_2 * invG2_2);
    double CalcI = BG + Imax1 * (Mu * L1 + (1 - Mu) * Gau1)
                      + Imax2 * (Mu * L2 + (1 - Mu) * Gau2);
    double diff = CalcI - PS[i];
    Total += diff * diff;

    if (grad) {
      double c = 2.0 * diff;
      double dL1dc = 8.0 * dr1 / (G1_2 * den1 * den1);
      double dG1dc = Gau1 * 2.0 * C0 * dr1 * invG1_2;
      grad[0] += c * Imax1 * (Mu * dL1dc + (1.0 - Mu) * dG1dc);
      double dL1dG = 8.0 * dr1_2 / (G1_3 * den1 * den1);
      double dG1dG = Gau1 * 2.0 * C0 * dr1_2 / G1_3;
      grad[3] += c * Imax1 * (Mu * dL1dG + (1.0 - Mu) * dG1dG);
      grad[4] += c * (Mu * L1 + (1.0 - Mu) * Gau1);
      double dL2dc = 8.0 * dr2 / (G2_2 * den2 * den2);
      double dG2dc = Gau2 * 2.0 * C0 * dr2 * invG2_2;
      grad[1] += c * Imax2 * (Mu * dL2dc + (1.0 - Mu) * dG2dc);
      double dL2dG = 8.0 * dr2_2 / (G2_3 * den2 * den2);
      double dG2dG = Gau2 * 2.0 * C0 * dr2_2 / G2_3;
      grad[5] += c * Imax2 * (Mu * dL2dG + (1.0 - Mu) * dG2dG);
      grad[6] += c * (Mu * L2 + (1.0 - Mu) * Gau2);
      grad[2] += c * (Imax1 * (L1 - Gau1) + Imax2 * (L2 - Gau2));
      grad[7] += c;
    }
  }
  return Total;
}

void calib_fit_doublet_peak_shape(int NrPtsForFit, double *Rs,
                                  double *PeakShape,
                                  double *Rfit1, double *Rfit2,
                                  double *fitSNR1, double *fitSNR2,
                                  double Rstep, double Rmean1,
                                  double Rmean2, double Rmid) {
  unsigned n = 8;
  double x[8], xl[8], xu[8];
  struct calib_profile_data f_data;
  f_data.NrPtsForFit = NrPtsForFit;
  f_data.Rs = Rs;
  f_data.PeakShape = PeakShape;
  double BG0 = (PeakShape[0] + PeakShape[NrPtsForFit - 1]) / 2;
  if (BG0 < 0) BG0 = 0;
  double MaxI = -1e20;
  for (int i = 0; i < NrPtsForFit; i++)
    if (PeakShape[i] > MaxI) MaxI = PeakShape[i];
  double GammaGuess = Rstep * 3;
  double ImaxGuess = MaxI - BG0;
  if (ImaxGuess < 1e-6) ImaxGuess = 1e-6;

  x[0] = Rmean1;  xl[0] = Rs[0];  xu[0] = Rmid;
  x[1] = Rmean2;  xl[1] = Rmid;   xu[1] = Rs[NrPtsForFit - 1];
  x[2] = 0.5;     xl[2] = 0;      xu[2] = 1;
  x[3] = GammaGuess;  xl[3] = Rstep / 2;  xu[3] = Rstep * NrPtsForFit / 2;
  x[4] = ImaxGuess * 0.6;  xl[4] = ImaxGuess / 100;  xu[4] = MaxI * 1.5;
  x[5] = GammaGuess;  xl[5] = Rstep / 2;  xu[5] = Rstep * NrPtsForFit / 2;
  x[6] = ImaxGuess * 0.6;  xl[6] = ImaxGuess / 100;  xu[6] = MaxI * 1.5;
  x[7] = BG0;  xl[7] = 0;  xu[7] = (BG0 > 0) ? BG0 * 2.0 : 1.0;

  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = calib_pv_doublet_obj;
  config.obj_data = &f_data;
  config.initial_guess = x;
  config.max_evaluations = 10000;
  config.max_time_seconds = 60;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double x_init[8];
  memcpy(x_init, x, n * sizeof(double));
  int rc = run_nlopt_optimization(NLOPT_LD_LBFGS, &config);
  if (rc < 0) {
    memcpy(x, x_init, n * sizeof(double));
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  }
  *Rfit1 = x[0];
  *Rfit2 = x[1];
  double rmsResid = sqrt(config.min_function_val / NrPtsForFit);
  if (rmsResid > 0) {
    *fitSNR1 = x[4] / rmsResid;
    *fitSNR2 = x[6] / rmsResid;
  } else {
    *fitSNR1 = 1.0;
    *fitSNR2 = 1.0;
  }
}

// ── Geometry optimization objective ────────────────────────────────
//
// Delegates pixel→(R,η) to dg_pixel_to_REta from DetectorGeometry.h.
// This is the single source of truth for the tilt+distortion model.

double calib_problem_function(unsigned n, const double *x, double *grad,
                              void *f_data_trial) {
  struct calib_opt_data *f = (struct calib_opt_data *)f_data_trial;
  int nIndices = f->nIndices;
  double *YMean = f->YMean;
  double *ZMean = f->ZMean;
  double *IdealTtheta = f->IdealTtheta;
  double MaxRad = f->MaxRad;
  double px = f->px;
  double tx = f->tx;
  double Lsd = x[0], ybc = x[1], zbc = x[2];
  double ty = x[3], tz = x[4];
  double p0 = x[5], p1 = x[6], p2 = x[7], p3 = x[8];
  double p4 = x[9], p5 = x[10], p6 = x[11];
  double parallax = 0;
  if (f->fitParallax)
    parallax = x[12];
  double wavelength = 0;
  if (f->fitWavelength)
    wavelength = x[f->nBase - 1];

  // Build tilt matrix using dg_ canonical function
  double TRs[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs);

  int doTrim = (f->trimFraction < 1.0 - 1e-9 && f->trimScratch != NULL);
  double TotalDiff = 0;
  int i;

#pragma omp parallel for num_threads(numProcs) reduction(+ : TotalDiff)
  for (i = 0; i < nIndices; i++) {
    double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
    int pIdx = -1;
    if (nPanels > 0) {
      pIdx = GetPanelIndex((double)YMean[i], (double)ZMean[i], nPanels, panels);
      if (pIdx == -1) {
        if (doTrim) f->trimScratch[i] = -1.0;
        continue;
      }
    }
    if (f->skipBin && f->skipBin[i]) {
      if (doTrim) f->trimScratch[i] = -1.0;
      continue;
    }

    // Extract optimizer-provided panel shifts
    int stride = (f->tolRotation > 1e-12) ? 3 : 2;
    if (f->perPanelLsd) stride++;
    if (f->perPanelDistort) stride++;
    if (n > (unsigned)f->nBase && pIdx != f->fixPanel) {
      int logicalIndex = (pIdx < f->fixPanel) ? pIdx : pIdx - 1;
      int xIdx = f->nBase + logicalIndex * stride;
      dY = x[xIdx];
      dZ = x[xIdx + 1];
      int off = 2;
      if (f->tolRotation > 1e-12) dTheta = x[xIdx + off++];
      if (f->perPanelLsd)         dLsd = x[xIdx + off++];
      if (f->perPanelDistort)     dP2 = x[xIdx + off++];
    }

    // Apply optimizer panel shifts to peak position
    double rawY = YMean[i] + dY;
    double rawZ = ZMean[i] + dZ;
    if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
      double cY = panels[pIdx].centerY;
      double cZ = panels[pIdx].centerZ;
      double cosT = cos(DG_DEG2RAD * dTheta);
      double sinT = sin(DG_DEG2RAD * dTheta);
      double dy = rawY - cY, dz = rawZ - cZ;
      rawY = cY + dy * cosT - dz * sinT;
      rawZ = cZ + dy * sinT + dz * cosT;
    }

    // Delegate geometry to dg_pixel_to_REta (single source of truth)
    // Note: dg_pixel_to_REta expects pixel coords with ybc/zbc,
    // and it internally computes Y_phys = (-rawY + ybc) * px
    double R_px, Eta;
    dg_pixel_to_REta(rawY, rawZ, ybc, zbc, TRs, Lsd, MaxRad,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_px, &Eta, NULL);

    // Ideal R in pixels: Lsd * tan(2θ) / px
    double thisTtheta;
    if (f->fitWavelength) {
      double d = f->PointDSpacing[i];
      thisTtheta = 2.0 * asin(wavelength / (2.0 * d)) / DG_DEG2RAD;
    } else {
      thisTtheta = IdealTtheta[i];
    }
    double RIdeal_px = Lsd * tan(DG_DEG2RAD * thisTtheta) / px;

    double Diff = 1.0 - R_px / RIdeal_px;

    double w = (f->Weights != NULL) ? f->Weights[i] : 1.0;
    if (f->weightByRadius)
      w *= (R_px * px) / MaxRad; // R in µm / MaxRad
    if (f->snrWeights != NULL)
      w *= f->snrWeights[i];

    double val = (f->useL2 ? Diff * Diff : fabs(Diff)) * w;
    if (doTrim) {
      f->trimScratch[i] = val;
    } else {
      TotalDiff += val;
    }
  }

  // Trimmed mean: sort residuals and sum only bottom fraction
  if (doTrim) {
    int nValid = 0;
    for (i = 0; i < nIndices; i++) {
      if (f->trimScratch[i] >= 0.0) {
        f->trimScratch[nValid++] = f->trimScratch[i];
      }
    }
    qsort(f->trimScratch, nValid, sizeof(double), calib_cmp_double);
    int trimCount = (int)(nValid * f->trimFraction);
    if (trimCount < 1) trimCount = 1;
    TotalDiff = 0;
    for (i = 0; i < trimCount; i++) {
      TotalDiff += f->trimScratch[i];
    }
  }

  NrCalls++;

  // Per-evaluation trace
  if (calib_trace_fp) {
    double meanStrain_ue = (nIndices > 0)
        ? (TotalDiff / nIndices) * 1e6
        : 0.0;
    fprintf(calib_trace_fp,
            "%lld,%.10e,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,"
            "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
            NrCalls, TotalDiff, meanStrain_ue,
            Lsd, ybc, zbc, ty, tz,
            p0, p1, p2, p3, p4, p5, p6);
    fflush(calib_trace_fp);
  }

  return TotalDiff;
}

// ── Main optimizer: fit Lsd, BC, tilts, distortion, panel shifts ───

void calib_fit_tilt_bc_lsd(
    int nIndices, double *YMean, double *ZMean,
    double *IdealTtheta, double Lsd, double MaxRad, double ybc,
    double zbc, double tx, double tyin, double tzin, double p0in,
    double p1in, double p2in, double p3in, double *ty, double *tz,
    double *LsdFit, double *ybcFit, double *zbcFit, double *p0,
    double *p1, double *p2, double *p3, double *MeanDiff,
    double tolTilts, double tolLsd, double tolBC, double tolP,
    double tolP0, double tolP1, double tolP2, double tolP3,
    double tolShifts, double tolRotation, double px,
    double outlierFactor, int minIndices, int fixPanel,
    double *Weights, double p4in, double tolP4, int PerPanelLsd,
    double tolLsdPanel, int PerPanelDistortion, double tolP2Panel,
    int WeightByRadius, double *snrWeights, double *p4Out,
    double p5in, double tolP5, double *p5Out,
    double p6in, double tolP6, double *p6Out,
    int verbose, int L2Objective, double *initParams,
    Panel *initPanels, int fitWavelength, double wavelengthIn,
    double tolWavelength, double *PointDSpacing,
    double *wavelengthOut,
    double parallaxIn, double tolParallax,
    double *parallaxOut,
    double trimmedMeanFraction,
    const int *skipBin) {

  int fitParallax = (tolParallax > EPS) ? 1 : 0;
  int nBase = 12; // Lsd, ybc, zbc, ty, tz, p0-p3, p4, p5, p6
  if (fitParallax) nBase++;
  if (fitWavelength) nBase++;
  unsigned n = nBase;
  if (tolShifts > EPS && nPanels > 1) {
    int stride = (tolRotation > EPS) ? 3 : 2;
    if (PerPanelLsd) stride++;
    if (PerPanelDistortion) stride++;
    n += (nPanels - 1) * stride;
  }

  struct calib_opt_data f_data;
  f_data.nIndices = nIndices;
  f_data.YMean = YMean;
  f_data.ZMean = ZMean;
  f_data.IdealTtheta = IdealTtheta;
  f_data.MaxRad = MaxRad;
  f_data.px = px;
  f_data.tx = tx;
  f_data.fixPanel = fixPanel;
  f_data.tolRotation = tolRotation;
  f_data.Weights = Weights;
  f_data.nBase = nBase;
  f_data.perPanelLsd = PerPanelLsd;
  f_data.perPanelDistort = PerPanelDistortion;
  f_data.weightByRadius = WeightByRadius;
  f_data.snrWeights = snrWeights;
  f_data.useL2 = L2Objective;
  f_data.fitWavelength = fitWavelength;
  f_data.PointDSpacing = PointDSpacing;
  f_data.fitParallax = fitParallax;
  f_data.trimFraction = trimmedMeanFraction;
  f_data.trimScratch = NULL;
  if (trimmedMeanFraction < 1.0 - 1e-9)
    f_data.trimScratch = malloc(nIndices * sizeof(double));
  f_data.skipBin = skipBin;

  double x[n], xl[n], xu[n];
  double bLsd  = (initParams ? initParams[0]  : Lsd);
  double bybc  = (initParams ? initParams[1]  : ybc);
  double bzbc  = (initParams ? initParams[2]  : zbc);
  double btyin = (initParams ? initParams[3]  : tyin);
  double btzin = (initParams ? initParams[4]  : tzin);
  double bp0   = (initParams ? initParams[5]  : p0in);
  double bp1   = (initParams ? initParams[6]  : p1in);
  double bp2   = (initParams ? initParams[7]  : p2in);
  double bp3   = (initParams ? initParams[8]  : p3in);
  double bp4   = (initParams ? initParams[9]  : p4in);
  double bp5   = (initParams ? initParams[10] : p5in);
  double bp6   = (initParams ? initParams[11] : p6in);

  x[0]  = Lsd;   xl[0]  = bLsd  - tolLsd;   xu[0]  = bLsd  + tolLsd;
  x[1]  = ybc;   xl[1]  = bybc  - tolBC;    xu[1]  = bybc  + tolBC;
  x[2]  = zbc;   xl[2]  = bzbc  - tolBC;    xu[2]  = bzbc  + tolBC;
  x[3]  = tyin;  xl[3]  = btyin - tolTilts; xu[3]  = btyin + tolTilts;
  x[4]  = tzin;  xl[4]  = btzin - tolTilts; xu[4]  = btzin + tolTilts;
  x[5]  = p0in;  xl[5]  = bp0   - tolP0;   xu[5]  = bp0   + tolP0;
  x[6]  = p1in;  xl[6]  = bp1   - tolP1;   xu[6]  = bp1   + tolP1;
  x[7]  = p2in;  xl[7]  = bp2   - tolP2;   xu[7]  = bp2   + tolP2;
  x[8]  = p3in;  xl[8]  = bp3   - tolP3;   xu[8]  = bp3   + tolP3;
  x[9]  = p4in;  xl[9]  = bp4   - tolP4;   xu[9]  = bp4   + tolP4;
  x[10] = p5in;  xl[10] = bp5   - tolP5;   xu[10] = bp5   + tolP5;
  x[11] = p6in;  xl[11] = bp6   - tolP6;   xu[11] = bp6   + tolP6;

  if (fitParallax) {
    double bpar = (initParams ? initParams[12] : parallaxIn);
    x[12]  = parallaxIn;
    xl[12] = bpar - tolParallax;
    xu[12] = bpar + tolParallax;
  }
  if (fitWavelength) {
    int wlIdx = nBase - 1;
    double bwl = (initParams ? initParams[wlIdx] : wavelengthIn);
    x[wlIdx]  = wavelengthIn;
    xl[wlIdx] = bwl - tolWavelength;
    xu[wlIdx] = bwl + tolWavelength;
  }

  // Panel shift bounds
  if (tolShifts > EPS && nPanels > 1) {
    int panelCounts[nPanels];
    memset(panelCounts, 0, nPanels * sizeof(int));
    for (int i = 0; i < nIndices; i++) {
      int pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
      if (pIdx >= 0 && pIdx < nPanels)
        panelCounts[pIdx]++;
    }

    int p_idx = nBase;
    for (int i = 0; i < nPanels; i++) {
      if (i == fixPanel) continue;

      // dY
      x[p_idx] = panels[i].dY;
      if (panelCounts[i] < minIndices) {
        x[p_idx] = 0; xl[p_idx] = 0; xu[p_idx] = 0;
      } else {
        double anchor = (initPanels ? initPanels[i].dY : x[p_idx]);
        xl[p_idx] = anchor - tolShifts;
        xu[p_idx] = anchor + tolShifts;
      }
      p_idx++;

      // dZ
      x[p_idx] = panels[i].dZ;
      if (panelCounts[i] < minIndices) {
        x[p_idx] = 0; xl[p_idx] = 0; xu[p_idx] = 0;
      } else {
        double anchor = (initPanels ? initPanels[i].dZ : x[p_idx]);
        xl[p_idx] = anchor - tolShifts;
        xu[p_idx] = anchor + tolShifts;
      }
      p_idx++;

      // dTheta
      if (tolRotation > EPS) {
        x[p_idx] = panels[i].dTheta;
        if (panelCounts[i] < minIndices) {
          x[p_idx] = 0; xl[p_idx] = 0; xu[p_idx] = 0;
        } else {
          double anchor = (initPanels ? initPanels[i].dTheta : x[p_idx]);
          xl[p_idx] = anchor - tolRotation;
          xu[p_idx] = anchor + tolRotation;
        }
        p_idx++;
      }

      // dLsd
      if (PerPanelLsd) {
        x[p_idx] = panels[i].dLsd;
        if (panelCounts[i] < minIndices) {
          x[p_idx] = 0; xl[p_idx] = 0; xu[p_idx] = 0;
        } else {
          double anchor = (initPanels ? initPanels[i].dLsd : x[p_idx]);
          xl[p_idx] = anchor - tolLsdPanel;
          xu[p_idx] = anchor + tolLsdPanel;
        }
        p_idx++;
      }

      // dP2
      if (PerPanelDistortion) {
        x[p_idx] = panels[i].dP2;
        if (panelCounts[i] < minIndices) {
          x[p_idx] = 0; xl[p_idx] = 0; xu[p_idx] = 0;
        } else {
          double anchor = (initPanels ? initPanels[i].dP2 : x[p_idx]);
          xl[p_idx] = anchor - tolP2Panel;
          xu[p_idx] = anchor + tolP2Panel;
        }
        p_idx++;
      }
    }
  }

  // Run NLopt optimization
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = calib_problem_function;
  config.obj_data = &f_data;
  config.initial_guess = x;
  config.max_evaluations = 50000;
  config.max_time_seconds = 120;
  config.ftol_rel = 1e-10;
  config.xtol_rel = 1e-10;

  if (n > 20)
    run_nlopt_optimization(NLOPT_LN_SBPLX, &config);
  else
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);

  *MeanDiff = config.min_function_val / nIndices;

  if (f_data.trimScratch) {
    free(f_data.trimScratch);
    f_data.trimScratch = NULL;
  }

  // Extract optimized parameters
  *LsdFit = x[0];
  *ybcFit = x[1];
  *zbcFit = x[2];
  *ty = x[3];
  *tz = x[4];
  *p0 = x[5];
  *p1 = x[6];
  *p2 = x[7];
  *p3 = x[8];
  if (nBase > 9 && p4Out) *p4Out = x[9];
  if (p5Out) *p5Out = x[10];
  if (p6Out) *p6Out = x[11];
  if (fitParallax && parallaxOut) *parallaxOut = x[12];
  if (fitWavelength && wavelengthOut) *wavelengthOut = x[nBase - 1];

  // Update panel shifts
  if (nPanels > 0) {
    if (fixPanel >= 0 && fixPanel < nPanels) {
      panels[fixPanel].dY = 0;
      panels[fixPanel].dZ = 0;
      panels[fixPanel].dTheta = 0;
    }
    if (tolShifts > EPS && nPanels > 1) {
      int xIdx = nBase;
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel) continue;
        panels[i].dY = x[xIdx++];
        panels[i].dZ = x[xIdx++];
        if (tolRotation > EPS) panels[i].dTheta = x[xIdx++];
        if (PerPanelLsd) panels[i].dLsd = x[xIdx++];
        if (PerPanelDistortion) panels[i].dP2 = x[xIdx++];
      }
    } else {
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel) continue;
        panels[i].dY = 0;
        panels[i].dZ = 0;
        panels[i].dTheta = 0;
      }
    }
  }

  // Post-optimization outlier rejection (using dg_pixel_to_REta)
  {
    double TRs[3][3];
    dg_build_tilt_matrix(tx, *ty, *tz, TRs);

    double *tempDiffs = malloc(nIndices * sizeof(double));
    for (int k = 0; k < nIndices; k++) tempDiffs[k] = -1.0;

    double totalSum = 0;
    int validCount = 0;

    for (int i = 0; i < nIndices; i++) {
      double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
      int pIdx = -1;
      if (nPanels > 0) {
        pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
        if (pIdx == -1) continue;
      }
      if (pIdx >= 0) {
        dY = panels[pIdx].dY;
        dZ = panels[pIdx].dZ;
        dTheta = panels[pIdx].dTheta;
        dLsd = panels[pIdx].dLsd;
        dP2 = panels[pIdx].dP2;
      }

      double rawY = YMean[i] + dY, rawZ = ZMean[i] + dZ;
      if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
        double cY = panels[pIdx].centerY;
        double cZ = panels[pIdx].centerZ;
        double cosT = cos(DG_DEG2RAD * dTheta);
        double sinT = sin(DG_DEG2RAD * dTheta);
        double dy = rawY - cY, dz = rawZ - cZ;
        rawY = cY + dy * cosT - dz * sinT;
        rawZ = cZ + dy * sinT + dz * cosT;
      }

      double R_px, Eta;
      dg_pixel_to_REta(rawY, rawZ, *ybcFit, *zbcFit, TRs, *LsdFit, MaxRad,
                       *p0, *p1, *p2, *p3,
                       (p4Out ? *p4Out : p4in), (p5Out ? *p5Out : p5in),
                       (p6Out ? *p6Out : p6in),
                       px, dLsd, dP2,
                       (fitParallax && parallaxOut) ? *parallaxOut : 0,
                       &R_px, &Eta, NULL);

      double RIdeal_px = (*LsdFit + dLsd) * tan(DG_DEG2RAD * IdealTtheta[i]) / px;
      // Re-project to global Lsd plane (dg_pixel_to_REta already does this)
      double Diff = fabs(1.0 - R_px / RIdeal_px);
      tempDiffs[i] = Diff;
      totalSum += Diff;
      validCount++;
    }

    double currentMean = (validCount > 0) ? (totalSum / validCount) : 0;

    if (outlierFactor > 0) {
      double threshold = outlierFactor * currentMean;
      double cleanSum = 0;
      int cleanCount = 0;
      for (int i = 0; i < nIndices; i++) {
        if (tempDiffs[i] < 0) continue;
        if (tempDiffs[i] <= threshold) {
          cleanSum += tempDiffs[i];
          cleanCount++;
        }
      }
      if (cleanCount > 0) {
        *MeanDiff = cleanSum / cleanCount;
        if (verbose)
          printf("Outlier rejection (Factor %.2f): Excluded %d / %d points. "
                 "Mean Strain: %.8f -> %.8f\n",
                 outlierFactor, validCount - cleanCount, validCount,
                 currentMean, *MeanDiff);
      } else {
        *MeanDiff = currentMean;
      }
    } else {
      *MeanDiff = currentMean;
    }

    free(tempDiffs);
  }
}

// ── Residual evaluator ─────────────────────────────────────────────
//
// Computes per-bin strain residuals using dg_pixel_to_REta.

void calib_correct_tilt_distortion(
    int nIndices, double MaxRad, double *YMean, double *ZMean,
    double *IdealTtheta, double px, double Lsd, double ybc, double zbc,
    double tx, double ty, double tz, double p0, double p1, double p2,
    double p3, double *Etas, double *Diffs, double *RadOuts,
    double *StdDiff, double outlierFactor, int *IsOutlier,
    double p4, double p5, double p6, int OutlierIterations,
    int verbose, double *MeanDiffOut, double parallax,
    const int *skipBin) {

  double TRs[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs);

  int i;
  double MeanDiff = 0;
  int nValidPoints = 0;

  for (i = 0; i < nIndices; i++) {
    double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
    int pIdx = -1;
    if (nPanels > 0) {
      pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
      if (pIdx == -1) {
        Diffs[i] = -1.0;
        continue;
      }
    }

    if (pIdx >= 0) {
      dY = panels[pIdx].dY;
      dZ = panels[pIdx].dZ;
      dTheta = panels[pIdx].dTheta;
      dLsd = panels[pIdx].dLsd;
      dP2 = panels[pIdx].dP2;
    }

    double rawY = YMean[i] + dY, rawZ = ZMean[i] + dZ;
    if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
      double cY = panels[pIdx].centerY;
      double cZ = panels[pIdx].centerZ;
      double cosT = cos(DG_DEG2RAD * dTheta);
      double sinT = sin(DG_DEG2RAD * dTheta);
      double dy = rawY - cY, dz = rawZ - cZ;
      rawY = cY + dy * cosT - dz * sinT;
      rawZ = cZ + dy * sinT + dz * cosT;
    }

    // Canonical geometry via dg_pixel_to_REta
    double R_px, Eta;
    dg_pixel_to_REta(rawY, rawZ, ybc, zbc, TRs, Lsd, MaxRad,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_px, &Eta, NULL);

    double RIdeal_px = (Lsd + dLsd) * tan(DG_DEG2RAD * IdealTtheta[i]) / px;
    // dg_pixel_to_REta already re-projects R to global Lsd plane,
    // so RIdeal should also be in global plane
    RIdeal_px = Lsd * tan(DG_DEG2RAD * IdealTtheta[i]) / px;

    double Diff = 1.0 - R_px / RIdeal_px;  // signed strain
    Etas[i] = Eta;
    Diffs[i] = Diff;  // signed: positive = R_fitted > R_ideal
    RadOuts[i] = R_px * px; // convert R to microns for output

    if (skipBin && skipBin[i])
      continue;
    nValidPoints++;
    MeanDiff += fabs(Diff);
  }

  if (nValidPoints > 0) {
    MeanDiff /= nValidPoints;
  } else {
    MeanDiff = 0;
  }

  // Iterative sigma-clipped outlier rejection
  double *validDiffs = malloc(nIndices * sizeof(double));
  int validCount = 0;

  if (outlierFactor > 0) {
    for (i = 0; i < nIndices; i++) {
      if (IsOutlier)
        IsOutlier[i] = (skipBin && skipBin[i]) ? 1 : 0;
    }

    int nIter = (OutlierIterations > 0) ? OutlierIterations : 1;
    for (int iter = 0; iter < nIter; iter++) {
      double threshold = outlierFactor * MeanDiff;
      double newSum = 0;
      validCount = 0;

      for (i = 0; i < nIndices; i++) {
        if (skipBin && skipBin[i]) continue;
        double absDiff = fabs(Diffs[i]);
        if (absDiff <= threshold) {
          if (IsOutlier) IsOutlier[i] = 0;
          validDiffs[validCount] = absDiff;
          newSum += absDiff;
          validCount++;
        } else {
          if (IsOutlier) IsOutlier[i] = 1;
        }
      }
      if (validCount > 0) {
        double prevMean = MeanDiff;
        MeanDiff = newSum / validCount;
        if (iter == nIter - 1 || fabs(MeanDiff - prevMean) < 1e-10) {
          if (verbose)
            printf("Outlier Rejection (Factor %.2f, iter %d/%d): Excluded "
                   "%d / %d. Mean Strain: %.8f -> %.8f\n",
                   outlierFactor, iter + 1, nIter,
                   nValidPoints - validCount, nValidPoints,
                   prevMean, MeanDiff);
          if (fabs(MeanDiff - prevMean) < 1e-10) break;
        }
      }
    }
  } else {
    for (i = 0; i < nIndices; i++) {
      if (!(skipBin && skipBin[i])) {
        if (IsOutlier) IsOutlier[i] = 0;
        validDiffs[validCount] = fabs(Diffs[i]);
        validCount++;
      } else {
        if (IsOutlier) IsOutlier[i] = 1;
      }
    }
  }

  double StdDiff2 = 0;
  for (i = 0; i < validCount; i++) {
    StdDiff2 += (validDiffs[i] - MeanDiff) * (validDiffs[i] - MeanDiff);
  }
  *StdDiff = (validCount > 0) ? sqrt(StdDiff2 / validCount) : 0;
  if (MeanDiffOut)
    *MeanDiffOut = MeanDiff;
  free(validDiffs);
}
