//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFit.c — Shared GSAS-II pseudo-Voigt peak fitting module
//
// Area-normalized pseudo-Voigt with Thompson-Cox-Hastings (TCH) mixing.
// Uses NLOPT (L-BFGS with Nelder-Mead fallback).

#include "PeakFit.h"
#include <nlopt.h>

// =========================================================================
// Thompson-Cox-Hastings: derive total FWHM (deg) and mixing eta
// from GSAS-II parameters. Direct port of Python _tch_eta_fwhm.
// =========================================================================
void pf_tch_eta_fwhm(double sig_centideg2, double gam_centideg,
                     double *out_fwhm_deg, double *out_eta) {
  double fg =
      sqrt(fmax(8.0 * log(2.0) * fmax(sig_centideg2, 1e-12), 0.0)) / 100.0;
  double fl = fmax(gam_centideg, 1e-6) / 100.0;

  double fg2 = fg * fg, fg3 = fg2 * fg, fg4 = fg3 * fg, fg5 = fg4 * fg;
  double fl2 = fl * fl, fl3 = fl2 * fl, fl4 = fl3 * fl, fl5 = fl4 * fl;

  double FWHM = pow(fg5 + 2.69269 * fg4 * fl + 2.42843 * fg3 * fl2 +
                        4.47163 * fg2 * fl3 + 0.07842 * fg * fl4 + fl5,
                    0.2);
  if (FWHM < 1e-15)
    FWHM = 1e-15;

  double ratio = fl / FWHM;
  double eta = 1.36603 * ratio - 0.47719 * ratio * ratio +
               0.11116 * ratio * ratio * ratio;
  if (eta < 0.0)
    eta = 0.0;
  if (eta > 1.0)
    eta = 1.0;

  *out_fwhm_deg = FWHM;
  *out_eta = eta;
}

// =========================================================================
// Area-normalized pseudo-Voigt model evaluation
//
// params layout: [Area0, Center0, sig0, gam0, Area1, ...., bg0, bg1]
//                (4 doubles per peak, then 2 background params)
// =========================================================================
void pf_calculate_model(int n_peaks, const double *params, int n_points,
                        const double *x_values, double x_lo, double x_hi,
                        double *out_model_curve) {
  // Background: Chebyshev T0 + T1
  double bg0 = params[n_peaks * 4];
  double bg1 = params[n_peaks * 4 + 1];
  double x_range = fmax(x_hi - x_lo, 1e-12);

  for (int i = 0; i < n_points; ++i) {
    double x_norm = 2.0 * (x_values[i] - x_lo) / x_range - 1.0;
    out_model_curve[i] = bg0 + bg1 * x_norm;
  }

  for (int pN = 0; pN < n_peaks; ++pN) {
    double area = params[pN * 4 + 0];
    double center = params[pN * 4 + 1];
    double sig = fmax(params[pN * 4 + 2], 1e-12);
    double gam = fmax(params[pN * 4 + 3], 1e-6);

    double FWHM, eta;
    pf_tch_eta_fwhm(sig, gam, &FWHM, &eta);
    if (FWHM < 1e-15)
      FWHM = 1e-15;

    // Gaussian component (area-normalized)
    double sigma_g = FWHM / (2.0 * sqrt(2.0 * log(2.0)));
    double G_norm = 1.0 / (sigma_g * sqrt(2.0 * M_PI));

    // Lorentzian component (area-normalized)
    double half_fwhm = FWHM / 2.0;
    double L_norm = half_fwhm / M_PI;

    for (int i = 0; i < n_points; ++i) {
      double dx = x_values[i] - center;
      double dx2 = dx * dx;
      double G = G_norm * exp(-0.5 * dx2 / (sigma_g * sigma_g));
      double L = L_norm / (dx2 + half_fwhm * half_fwhm);
      out_model_curve[i] += area * (eta * L + (1.0 - eta) * G);
    }
  }
}

// =========================================================================
// NLOPT objective: sum of squared residuals
// =========================================================================
double pf_objective(unsigned n, const double *x, double *grad, void *fdat) {
  const PF_DataFit *d = (const PF_DataFit *)fdat;
  const int Np = d->nrBins;
  const double *xs = d->R;
  const double *Is = d->Int;
  const int nP = (n - 2) / 4; // 4 per peak + 2 BG
  const double x_lo = d->x_lo;
  const double x_hi = d->x_hi;

  double *calculated = (double *)malloc(Np * sizeof(double));
  if (!calculated)
    return INFINITY;
  pf_calculate_model(nP, x, Np, xs, x_lo, x_hi, calculated);

  double total_sq_error = 0.0;
  for (int i = 0; i < Np; ++i) {
    double residual = calculated[i] - Is[i];
    total_sq_error += residual * residual;
  }

  if (grad) {
    memset(grad, 0, n * sizeof(double));

    // Numerical gradients (safer than analytical for TCH mixing)
    double eps = 1e-6;
    double *calc_plus = (double *)malloc(Np * sizeof(double));
    double *x_pert = (double *)malloc(n * sizeof(double));
    memcpy(x_pert, x, n * sizeof(double));

    for (unsigned p = 0; p < n; ++p) {
      double h = fmax(fabs(x[p]) * eps, eps);
      x_pert[p] = x[p] + h;
      pf_calculate_model(nP, x_pert, Np, xs, x_lo, x_hi, calc_plus);
      double err_plus = 0.0;
      for (int i = 0; i < Np; ++i) {
        double r = calc_plus[i] - Is[i];
        err_plus += r * r;
      }
      grad[p] = (err_plus - total_sq_error) / h;
      x_pert[p] = x[p]; // restore
    }

    free(calc_plus);
    free(x_pert);
  }

  free(calculated);
  return total_sq_error;
}

// =========================================================================
// Savitzky-Golay smoothing
// =========================================================================
void pf_smoothData(const double *in, double *out, int N, int W) {
  if (W < 3 || W % 2 == 0) {
    memcpy(out, in, N * sizeof(double));
    return;
  }
  int H = W / 2;
  double *coeffs = (double *)malloc(W * sizeof(double));
  if (!coeffs) {
    memcpy(out, in, N * sizeof(double));
    return;
  }
  double norm = 0.0;

  switch (W) {
  case 5:
    norm = 35.0;
    coeffs[0] = -3;
    coeffs[1] = 12;
    coeffs[2] = 17;
    coeffs[3] = 12;
    coeffs[4] = -3;
    break;
  case 7:
    norm = 21.0;
    coeffs[0] = -2;
    coeffs[1] = 3;
    coeffs[2] = 6;
    coeffs[3] = 7;
    coeffs[4] = 6;
    coeffs[5] = 3;
    coeffs[6] = -2;
    break;
  case 9:
    norm = 231.0;
    coeffs[0] = -21;
    coeffs[1] = 14;
    coeffs[2] = 39;
    coeffs[3] = 54;
    coeffs[4] = 59;
    coeffs[5] = 54;
    coeffs[6] = 39;
    coeffs[7] = 14;
    coeffs[8] = -21;
    break;
  default:
    fprintf(stderr, "pf_smoothData: Unsupported window %d, no smoothing.\n", W);
    memcpy(out, in, N * sizeof(double));
    free(coeffs);
    return;
  }

  for (int i = 0; i < W; ++i)
    coeffs[i] /= norm;

  for (int i = 0; i < N; ++i) {
    if (i < H || i >= N - H) {
      out[i] = in[i];
    } else {
      double val = 0.0;
      for (int j = 0; j < W; ++j)
        val += coeffs[j] * in[i - H + j];
      out[i] = val;
    }
  }
  free(coeffs);
}

// =========================================================================
// Initial parameter estimation
// =========================================================================
double pf_estimate_initial_params(const double *intensity_data, int n_points,
                                  int peak_idx_local, double *out_bg_guess,
                                  double *out_amp_guess) {
  int bg_width = (int)fmin(5, n_points / 4);
  if (bg_width < 1)
    bg_width = 1;
  double bg_sum = 0.0;
  for (int i = 0; i < bg_width; ++i) {
    bg_sum += intensity_data[i];
    bg_sum += intensity_data[n_points - 1 - i];
  }
  *out_bg_guess = bg_sum / (2.0 * bg_width);

  *out_amp_guess = intensity_data[peak_idx_local] - *out_bg_guess;
  if (*out_amp_guess <= 0)
    *out_amp_guess = intensity_data[peak_idx_local];

  double half_max = *out_bg_guess + (*out_amp_guess / 2.0);

  int left_idx = peak_idx_local;
  while (left_idx > 0 && intensity_data[left_idx] > half_max)
    left_idx--;

  int right_idx = peak_idx_local;
  while (right_idx < n_points - 1 && intensity_data[right_idx] > half_max)
    right_idx++;

  double fwhm = (double)(right_idx - left_idx);
  return (fwhm > 1.0) ? fwhm : 2.0;
}

// =========================================================================
// Sort helper
// =========================================================================
int pf_comparePeaksByIndex(const void *a, const void *b) {
  PF_Peak *peakA = (PF_Peak *)a;
  PF_Peak *peakB = (PF_Peak *)b;
  return (peakA->index - peakB->index);
}

// =========================================================================
// Top-level fitting function (GSAS-II area-normalized)
//
// Input:
//   x[]            - x-axis values (any unit: pixels, 2θ degrees, etc.)
//   intensity[]    - measured 1D profile
//   peakLocations  - target x positions for peaks
//
// Output per peak (PF_PARAMS_PER_PEAK = 7 doubles):
//   [0] area        - integrated intensity
//   [1] center      - fitted peak center (same units as x)
//   [2] sig         - Gaussian variance (centideg² if x in degrees)
//   [3] gam         - Lorentzian FWHM (centideg if x in degrees)
//   [4] FWHM        - total FWHM (same units as x)
//   [5] eta         - pseudo-Voigt mixing (0=Gaussian, 1=Lorentzian)
//   [6] chi_sq      - goodness-of-fit (sum sq resid / dof)
// =========================================================================
int fitPeaks(const double *x, const double *intensity, int nBins,
             const double *peakLocations, int nPeaks, double xBinSize,
             int fitROIPadding, double *outFitParams) {
  if (nPeaks <= 0 || nBins <= 0)
    return 0;

  static int pf_version_printed = 0;
  if (!pf_version_printed) {
    printf("  [PeakFit GSAS-II] area-normalized, nPeaks=%d, xBinSize=%.4f\n",
           nPeaks, xBinSize);
    pf_version_printed = 1;
  }

  // Zero output
  memset(outFitParams, 0, nPeaks * PF_PARAMS_PER_PEAK * sizeof(double));

  // Step 1: Identify peak bin indices
  PF_Peak *pks = (PF_Peak *)malloc(nPeaks * sizeof(PF_Peak));
  int validCount = 0;
  for (int p = 0; p < nPeaks; ++p) {
    int bestBin = -1;
    double minDiff = 1e10;
    for (int r = 0; r < nBins; ++r) {
      double diff = fabs(x[r] - peakLocations[p]);
      if (diff < minDiff) {
        minDiff = diff;
        bestBin = r;
      }
    }
    if (bestBin != -1 && minDiff < xBinSize * 2.0) {
      pks[validCount].index = bestBin;
      pks[validCount].radius = x[bestBin];
      pks[validCount].intensity = intensity[bestBin];
      validCount++;
    } else if (p < 3) {
      // printf("  [PeakFit] Peak %d: loc=%.4f, bestBin=%d, minDiff=%.6f => "
      //        "REJECTED\n",
      //        p, peakLocations[p], bestBin, minDiff);
    }
  }

  if (validCount == 0) {
    free(pks);
    return 0;
  }

  // Step 2: Sort peaks and create fit jobs (merge overlapping ROIs)
  qsort(pks, validCount, sizeof(PF_Peak), pf_comparePeaksByIndex);

  PF_FitJob *fitJobs = (PF_FitJob *)malloc(validCount * sizeof(PF_FitJob));
  int *job_result_indices = (int *)calloc(validCount, sizeof(int));
  int numJobs = 0;

  int roi_half = fitROIPadding;
  fitJobs[0].startIndex = (int)fmax(0, pks[0].index - roi_half);
  fitJobs[0].endIndex = (int)fmin(nBins - 1, pks[0].index + roi_half);
  fitJobs[0].numPeaks = 1;
  fitJobs[0].peaks = &pks[0];
  job_result_indices[0] = 0;
  numJobs = 1;

  for (int i = 1; i < validCount; ++i) {
    int current_roi_start = (int)fmax(0, pks[i].index - roi_half);
    if (current_roi_start <= fitJobs[numJobs - 1].endIndex) {
      fitJobs[numJobs - 1].endIndex =
          (int)fmin(nBins - 1, pks[i].index + roi_half);
      fitJobs[numJobs - 1].numPeaks++;
    } else {
      job_result_indices[numJobs] =
          job_result_indices[numJobs - 1] + fitJobs[numJobs - 1].numPeaks;
      fitJobs[numJobs].startIndex = current_roi_start;
      fitJobs[numJobs].endIndex = (int)fmin(nBins - 1, pks[i].index + roi_half);
      fitJobs[numJobs].numPeaks = 1;
      fitJobs[numJobs].peaks = &pks[i];
      numJobs++;
    }
  }

  // Step 3: Fit each job
  int total_successful = 0;

#pragma omp parallel for reduction(+ : total_successful)
  for (int i = 0; i < numJobs; ++i) {
    PF_FitJob *job = &fitJobs[i];
    int nJobPeaks = job->numPeaks;
    int nFitParams = nJobPeaks * 4 + 2; // 4 per peak + 2 BG (bg0, bg1)

    double *fitParams = (double *)malloc(nFitParams * sizeof(double));
    double *lowerBounds = (double *)malloc(nFitParams * sizeof(double));
    double *upperBounds = (double *)malloc(nFitParams * sizeof(double));

    int roiLen = job->endIndex - job->startIndex + 1;
    double roi_max = -1e30, roi_min = 1e30;
    for (int rr = 0; rr < roiLen; rr++) {
      double v = intensity[job->startIndex + rr];
      if (v > roi_max)
        roi_max = v;
      if (v < roi_min)
        roi_min = v;
    }
    double data_range = fmax(roi_max - roi_min, 1.0);

    // x bounds for Chebyshev normalization
    double x_lo = x[job->startIndex];
    double x_hi = x[job->endIndex];

    double primary_bg_guess = 0.0;
    for (int p = 0; p < nJobPeaks; ++p) {
      PF_Peak *peak = &(job->peaks[p]);
      int p_idx_local = peak->index - job->startIndex;
      double bg_g, amp_g;
      double fwhm_bins = pf_estimate_initial_params(
          &intensity[job->startIndex], roiLen, p_idx_local, &bg_g, &amp_g);
      double fwhm_x = fwhm_bins * xBinSize; // FWHM in x-units

      if (p == 0)
        primary_bg_guess = bg_g;

      int b = p * 4;

      // Initial sig (centideg²): convert from FWHM
      // FWHM_G = sqrt(8 ln2 * sig_centideg2) / 100
      // => sig_centideg2 = (FWHM * 100)² / (8 ln2)
      // Start with half the FWHM as Gaussian contribution
      double fwhm_g_init = fwhm_x * 0.5;
      double sig_init =
          (fwhm_g_init * 100.0) * (fwhm_g_init * 100.0) / (8.0 * log(2.0));
      if (sig_init < 1e-6)
        sig_init = 1.0;

      // Initial gam (centideg): Lorentzian FWHM contribution
      double gam_init = fwhm_x * 0.5 * 100.0; // half the FWHM as Lorentzian
      if (gam_init < 0.01)
        gam_init = 1.0;

      // Initial area: amp * FWHM * factor (area ≈ Imax * FWHM * π/2 for
      // Lorentzian)
      double area_init = amp_g * fwhm_x * 1.5;
      if (area_init <= 0)
        area_init = data_range * xBinSize;

      // Center: same as peak position
      double center_init = peak->radius;

      // Bounds
      double center_range = fmax(fwhm_x, (double)roiLen * xBinSize * 0.5);
      double area_hi = data_range * (double)roiLen * xBinSize * 2.0;
      if (area_hi < 1.0)
        area_hi = 1.0;
      // sig bounds: very small to very large (centideg²)
      double sig_lo = 1e-6;
      double sig_hi = (xBinSize * 200.0 * 100.0) * (xBinSize * 200.0 * 100.0) /
                      (8.0 * log(2.0));
      // gam bounds (centideg)
      double gam_lo = 0.01;
      double gam_hi = xBinSize * 200.0 * 100.0;

      fitParams[b + 0] = fmax(fmin(area_init, area_hi * 0.9), 0.01);
      fitParams[b + 1] = center_init;
      fitParams[b + 2] = fmax(fmin(sig_init, sig_hi * 0.5), sig_lo * 2.0);
      fitParams[b + 3] = fmax(fmin(gam_init, gam_hi * 0.5), gam_lo * 2.0);

      lowerBounds[b + 0] = 0;
      lowerBounds[b + 1] = center_init - center_range;
      lowerBounds[b + 2] = sig_lo;
      lowerBounds[b + 3] = gam_lo;

      upperBounds[b + 0] = area_hi;
      upperBounds[b + 1] = center_init + center_range;
      upperBounds[b + 2] = sig_hi;
      upperBounds[b + 3] = gam_hi;

      // Clamp initial values strictly within bounds
      for (int bb = b; bb < b + 4; bb++) {
        if (fitParams[bb] <= lowerBounds[bb])
          fitParams[bb] =
              lowerBounds[bb] + (upperBounds[bb] - lowerBounds[bb]) * 0.01;
        if (fitParams[bb] >= upperBounds[bb])
          fitParams[bb] =
              upperBounds[bb] - (upperBounds[bb] - lowerBounds[bb]) * 0.01;
      }
    }

    // Background params
    fitParams[nJobPeaks * 4] = primary_bg_guess;
    lowerBounds[nJobPeaks * 4] = fmin(roi_min - data_range * 0.5, -1.0);
    upperBounds[nJobPeaks * 4] = roi_max + data_range;
    if (fitParams[nJobPeaks * 4] <= lowerBounds[nJobPeaks * 4])
      fitParams[nJobPeaks * 4] = lowerBounds[nJobPeaks * 4] + 0.01;
    if (fitParams[nJobPeaks * 4] >= upperBounds[nJobPeaks * 4])
      fitParams[nJobPeaks * 4] = upperBounds[nJobPeaks * 4] - 0.01;

    fitParams[nJobPeaks * 4 + 1] = 0.0;
    lowerBounds[nJobPeaks * 4 + 1] = -data_range;
    upperBounds[nJobPeaks * 4 + 1] = data_range;

    PF_DataFit fitData;
    fitData.nrBins = roiLen;
    fitData.R = &x[job->startIndex];
    fitData.Int = &intensity[job->startIndex];
    fitData.x_lo = x_lo;
    fitData.x_hi = x_hi;

    // Primary: L-BFGS
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, nFitParams);
    nlopt_set_lower_bounds(opt, lowerBounds);
    nlopt_set_upper_bounds(opt, upperBounds);
    nlopt_set_min_objective(opt, pf_objective, &fitData);
    nlopt_set_xtol_rel(opt, 1e-5);
    nlopt_set_maxeval(opt, 500 * nFitParams);

    double minObj;
    int rc = nlopt_optimize(opt, fitParams, &minObj);
    nlopt_destroy(opt);

    // Fallback: Nelder-Mead
    if (rc < 0) {
      opt = nlopt_create(NLOPT_LN_NELDERMEAD, nFitParams);
      nlopt_set_lower_bounds(opt, lowerBounds);
      nlopt_set_upper_bounds(opt, upperBounds);
      nlopt_set_min_objective(opt, pf_objective, &fitData);
      nlopt_set_maxeval(opt, 5000);
      nlopt_set_maxtime(opt, 30);
      nlopt_set_ftol_rel(opt, 1e-5);
      nlopt_set_xtol_rel(opt, 1e-5);
      rc = nlopt_optimize(opt, fitParams, &minObj);
      nlopt_destroy(opt);
    }

    if (rc >= 0) {
      double chi_sq = minObj / fmax((double)(roiLen - nFitParams), 1.0);
      int result_start = job_result_indices[i];

      for (int p = 0; p < nJobPeaks; ++p) {
        int out_base = (result_start + p) * PF_PARAMS_PER_PEAK;
        int in_base = p * 4;

        double area = fitParams[in_base + 0];
        double center = fitParams[in_base + 1];
        double sig = fitParams[in_base + 2];
        double gam = fitParams[in_base + 3];

        double FWHM, eta;
        pf_tch_eta_fwhm(sig, gam, &FWHM, &eta);

        outFitParams[out_base + 0] = area;   // area
        outFitParams[out_base + 1] = center; // center
        outFitParams[out_base + 2] = sig;    // sig (centideg²)
        outFitParams[out_base + 3] = gam;    // gam (centideg)
        outFitParams[out_base + 4] = FWHM;   // FWHM (x-units)
        outFitParams[out_base + 5] = eta;    // eta mixing
        outFitParams[out_base + 6] = chi_sq; // chi²/dof
      }
      total_successful += nJobPeaks;
    } else if (i < 2) {
      // printf("  [PeakFit] Job %d NLOPT FAILED: rc=%d, nBins=%d, nPeaks=%d\n",
      // i,
      //        rc, roiLen, nJobPeaks);
    }

    free(fitParams);
    free(lowerBounds);
    free(upperBounds);
  }

  free(fitJobs);
  free(job_result_indices);
  free(pks);
  return total_successful;
}

// =========================================================================
// SNIP background (Morháč LLS transform)
// =========================================================================
void pf_snip_background(const double *intensity, double *background, int nBins,
                        int nIter) {
  if (nBins <= 0 || nIter <= 0) {
    if (nBins > 0)
      memset(background, 0, nBins * sizeof(double));
    return;
  }

  // LLS forward: v = log(log(sqrt(y+1)+1)+1)
  double *v = (double *)malloc(nBins * sizeof(double));
  if (!v) {
    memset(background, 0, nBins * sizeof(double));
    return;
  }
  for (int i = 0; i < nBins; i++) {
    double y = fmax(intensity[i], 0.0);
    v[i] = log(log(sqrt(y + 1.0) + 1.0) + 1.0);
  }

  // Iterative clipping with decreasing window
  double *tmp = (double *)malloc(nBins * sizeof(double));
  if (!tmp) {
    memcpy(background, intensity, nBins * sizeof(double));
    free(v);
    return;
  }
  for (int p = nIter; p >= 1; p--) {
    memcpy(tmp, v, nBins * sizeof(double));
    for (int i = p; i < nBins - p; i++) {
      double avg = (tmp[i - p] + tmp[i + p]) / 2.0;
      if (v[i] > avg)
        v[i] = avg;
    }
  }

  // LLS inverse: y = (exp(exp(v)-1)-1)^2 - 1
  for (int i = 0; i < nBins; i++) {
    double w = (exp(exp(v[i]) - 1.0) - 1.0);
    background[i] = fmax(w * w - 1.0, 0.0);
  }

  free(v);
  free(tmp);
}

// =========================================================================
// Automatic peak detection
// =========================================================================

// Internal: compare helpers for sorting peak candidates
typedef struct {
  int index;
  double prominence;
  double value;
} PF_PeakCandidate;

static int pf_cmp_prominence_desc(const void *a, const void *b) {
  double pa = ((const PF_PeakCandidate *)a)->prominence;
  double pb = ((const PF_PeakCandidate *)b)->prominence;
  if (pb > pa)
    return 1;
  if (pb < pa)
    return -1;
  return 0;
}

static int pf_cmp_index_asc(const void *a, const void *b) {
  return ((const PF_PeakCandidate *)a)->index -
         ((const PF_PeakCandidate *)b)->index;
}

int pf_detect_peaks(const double *x, const double *corrected, int nBins,
                    int maxPeaks, double *outPeakLocations,
                    double minSeparation) {
  if (nBins < 10 || maxPeaks <= 0)
    return 0;

  // Step 1: Smooth the signal (Savitzky-Golay, window=7)
  double *smooth = (double *)malloc(nBins * sizeof(double));
  if (!smooth)
    return 0;
  pf_smoothData(corrected, smooth, nBins, 7);

  // Step 2: Compute negative 2nd derivative via finite differences
  double *neg_d2 = (double *)calloc(nBins, sizeof(double));
  if (!neg_d2) {
    free(smooth);
    return 0;
  }
  for (int i = 1; i < nBins - 1; i++) {
    neg_d2[i] = -(smooth[i + 1] - 2.0 * smooth[i] + smooth[i - 1]);
  }

  // Step 3: Find local maxima of neg_d2 with minimum separation
  double binWidth = (nBins > 1) ? fabs(x[1] - x[0]) : 1.0;
  int minSepBins = (int)fmax(3, minSeparation / binWidth);

  int maxCandidates = nBins / 2;
  PF_PeakCandidate *cands =
      (PF_PeakCandidate *)malloc(maxCandidates * sizeof(PF_PeakCandidate));
  if (!cands) {
    free(smooth);
    free(neg_d2);
    return 0;
  }
  int nCands = 0;
  for (int i = 2; i < nBins - 2; i++) {
    if (neg_d2[i] > 0 && neg_d2[i] >= neg_d2[i - 1] &&
        neg_d2[i] >= neg_d2[i + 1] && corrected[i] > 0) {
      double left_min = neg_d2[i], right_min = neg_d2[i];
      for (int j = i - 1; j >= 0 && j >= i - minSepBins; j--) {
        if (neg_d2[j] < left_min)
          left_min = neg_d2[j];
      }
      for (int j = i + 1; j < nBins && j <= i + minSepBins; j++) {
        if (neg_d2[j] < right_min)
          right_min = neg_d2[j];
      }
      double prom = neg_d2[i] - fmax(left_min, right_min);
      if (prom < 0)
        prom = neg_d2[i];
      if (nCands < maxCandidates) {
        cands[nCands].index = i;
        cands[nCands].prominence = prom;
        cands[nCands].value = corrected[i];
        nCands++;
      }
    }
  }

  // Step 4: Sort by prominence (descending), greedy pick with separation
  qsort(cands, nCands, sizeof(PF_PeakCandidate), pf_cmp_prominence_desc);

  PF_PeakCandidate *selected =
      (PF_PeakCandidate *)malloc(maxPeaks * sizeof(PF_PeakCandidate));
  if (!selected) {
    free(cands);
    free(smooth);
    free(neg_d2);
    return 0;
  }
  int nSelected = 0;
  for (int i = 0; i < nCands && nSelected < maxPeaks; i++) {
    int too_close = 0;
    for (int j = 0; j < nSelected; j++) {
      if (abs(cands[i].index - selected[j].index) < minSepBins) {
        too_close = 1;
        break;
      }
    }
    if (!too_close) {
      selected[nSelected++] = cands[i];
    }
  }

  // Step 5: Sort selected by position, write x-values
  qsort(selected, nSelected, sizeof(PF_PeakCandidate), pf_cmp_index_asc);
  for (int i = 0; i < nSelected; i++) {
    outPeakLocations[i] = x[selected[i].index];
  }

  free(selected);
  free(cands);
  free(smooth);
  free(neg_d2);
  return nSelected;
}

// =========================================================================
// Top-level auto-detect + fit (GSAS-II output)
// =========================================================================
int fitPeaksAutoDetect(const double *x, const double *intensity, int nBins,
                       int maxPeaks, double xBinSize, int fitROIPadding,
                       double *outFitParams, int snipIter) {
  if (nBins <= 0 || maxPeaks <= 0)
    return 0;

  // Step 1: SNIP background subtraction
  double *bg = (double *)malloc(nBins * sizeof(double));
  double *corrected = (double *)malloc(nBins * sizeof(double));
  if (!bg || !corrected) {
    free(bg);
    free(corrected);
    return 0;
  }
  pf_snip_background(intensity, bg, nBins, snipIter);
  for (int i = 0; i < nBins; i++) {
    corrected[i] = fmax(intensity[i] - bg[i], 0.0);
  }

  // Step 2: Auto-detect peaks (over-select by 3×)
  int overSelect = maxPeaks * 3;
  if (overSelect > nBins / 2)
    overSelect = nBins / 2;
  double *peakLocs = (double *)malloc(overSelect * sizeof(double));
  if (!peakLocs) {
    free(bg);
    free(corrected);
    return 0;
  }
  double minSep = xBinSize * 3.0;
  int nDetected =
      pf_detect_peaks(x, corrected, nBins, overSelect, peakLocs, minSep);

  if (nDetected <= 0) {
    free(bg);
    free(corrected);
    free(peakLocs);
    return 0;
  }

  static int autodetect_printed = 0;
  if (!autodetect_printed) {
    printf("  [PeakFit AutoDetect] SNIP(%d iter) found %d peak candidates\n",
           snipIter, nDetected);
    autodetect_printed = 1;
  }

  // Step 3: Fit using GSAS-II profile (on background-subtracted data)
  int fitBufSize = nDetected * PF_PARAMS_PER_PEAK;
  double *rawFit = (double *)calloc(fitBufSize, sizeof(double));
  if (!rawFit) {
    free(bg);
    free(corrected);
    free(peakLocs);
    return 0;
  }

  int nFitted = fitPeaks(x, corrected, nBins, peakLocs, nDetected, xBinSize,
                         fitROIPadding, rawFit);

  free(bg);
  free(corrected);
  free(peakLocs);

  if (nFitted <= 0) {
    free(rawFit);
    return 0;
  }

  // Step 4: Quality filtering
  typedef struct {
    int origIdx;
    double area;
    double fwhm;
  } FitEntry;

  FitEntry *entries = (FitEntry *)malloc(nDetected * sizeof(FitEntry));
  int nValid = 0;
  for (int i = 0; i < nDetected; i++) {
    int base = i * PF_PARAMS_PER_PEAK;
    double area = fabs(rawFit[base + 0]); // area
    double fwhm = rawFit[base + 4];       // FWHM
    if (area > 0 && fwhm > 0) {
      entries[nValid].origIdx = i;
      entries[nValid].area = area;
      entries[nValid].fwhm = fwhm;
      nValid++;
    }
  }

  // Adaptive FWHM rejection: 5× median
  if (nValid > 2) {
    double *fwhms = (double *)malloc(nValid * sizeof(double));
    for (int i = 0; i < nValid; i++)
      fwhms[i] = entries[i].fwhm;
    // Bubble sort (small N)
    for (int i = 0; i < nValid - 1; i++)
      for (int j = i + 1; j < nValid; j++)
        if (fwhms[j] < fwhms[i]) {
          double t = fwhms[i];
          fwhms[i] = fwhms[j];
          fwhms[j] = t;
        }
    double medFwhm = fwhms[nValid / 2];
    double maxFwhm = fmax(5.0 * medFwhm, xBinSize * 2.0);
    free(fwhms);

    int dst = 0;
    for (int i = 0; i < nValid; i++) {
      if (entries[i].fwhm <= maxFwhm)
        entries[dst++] = entries[i];
    }
    nValid = dst;
  }

  // Select top maxPeaks by area (descending), then sort by position
  if (nValid > maxPeaks) {
    for (int i = 0; i < nValid - 1; i++)
      for (int j = i + 1; j < nValid; j++)
        if (entries[j].area > entries[i].area) {
          FitEntry t = entries[i];
          entries[i] = entries[j];
          entries[j] = t;
        }
    nValid = maxPeaks;
  }

  // Sort by center position (ascending)
  for (int i = 0; i < nValid - 1; i++)
    for (int j = i + 1; j < nValid; j++)
      if (entries[j].origIdx < entries[i].origIdx) {
        FitEntry t = entries[i];
        entries[i] = entries[j];
        entries[j] = t;
      }

  // Step 5: Copy to output buffer
  memset(outFitParams, 0, maxPeaks * PF_PARAMS_PER_PEAK * sizeof(double));
  for (int i = 0; i < nValid; i++) {
    int src = entries[i].origIdx * PF_PARAMS_PER_PEAK;
    int dst = i * PF_PARAMS_PER_PEAK;
    memcpy(&outFitParams[dst], &rawFit[src],
           PF_PARAMS_PER_PEAK * sizeof(double));
  }

  free(entries);
  free(rawFit);
  return nValid;
}

// =========================================================================
// Height-normalized pseudo-Voigt (pV) — singlet
// Ported from CalibrationCore.c / CalibPeakFit.c
// x[0]=Rcen, x[1]=Mu, x[2]=Gamma(FWHM), x[3]=Imax, x[4]=BG
// =========================================================================

typedef struct {
  int N;
  double *Rs;
  double *Ints;
} PF_PV_Data;

static double pf_pv_singlet_obj(unsigned n, const double *x,
                                double *grad, void *data) {
  PF_PV_Data *f = (PF_PV_Data *)data;
  int N = f->N;
  double *Rs = f->Rs;
  double *PS = f->Ints;
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

static void pf_fit_singlet_pv(int nPts, double *Rs, double *Ints,
                               double *Rfit, double *fitSNR,
                               double Rstep, double Rmean) {
  double x[5], xl[5], xu[5];
  PF_PV_Data f_data;
  f_data.N = nPts;
  f_data.Rs = Rs;
  f_data.Ints = Ints;

  double BG0 = (Ints[0] + Ints[nPts - 1]) / 2;
  if (BG0 < 0) BG0 = 0;
  double MaxI = -1e20;
  for (int i = 0; i < nPts; i++)
    if (Ints[i] > MaxI) MaxI = Ints[i];

  x[0] = Rmean;  xl[0] = Rs[0];  xu[0] = Rs[nPts - 1];
  x[1] = 0.5;    xl[1] = 0;      xu[1] = 1;
  double GammaGuess = Rstep * 3;
  x[2] = GammaGuess;  xl[2] = Rstep / 2;  xu[2] = Rstep * nPts / 2;
  x[3] = MaxI - BG0;  xl[3] = (MaxI - BG0) / 100;  xu[3] = MaxI * 1.5;
  x[4] = BG0;         xl[4] = 0;  xu[4] = (BG0 > 0) ? BG0 * 1.5 : MaxI * 0.5;

  double x_init[5];
  memcpy(x_init, x, 5 * sizeof(double));
  double minf;

  nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 5);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, pf_pv_singlet_obj, &f_data);
  nlopt_set_xtol_rel(opt, 1e-8);
  nlopt_set_ftol_rel(opt, 1e-8);
  nlopt_set_maxeval(opt, 10000);
  int rc = nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);

  if (rc < 0) {
    memcpy(x, x_init, 5 * sizeof(double));
    opt = nlopt_create(NLOPT_LN_NELDERMEAD, 5);
    nlopt_set_lower_bounds(opt, xl);
    nlopt_set_upper_bounds(opt, xu);
    nlopt_set_min_objective(opt, pf_pv_singlet_obj, &f_data);
    nlopt_set_xtol_rel(opt, 1e-8);
    nlopt_set_ftol_rel(opt, 1e-8);
    nlopt_set_maxeval(opt, 10000);
    nlopt_optimize(opt, x, &minf);
    nlopt_destroy(opt);
  }

  *Rfit = x[0];
  double rmsResid = sqrt(minf / nPts);
  *fitSNR = (rmsResid > 0) ? x[3] / rmsResid : 1.0;
}

// =========================================================================
// Height-normalized pseudo-Voigt (pV) — doublet
// x[0]=Rcen1, x[1]=Rcen2, x[2]=Mu, x[3]=Gamma1, x[4]=Imax1,
// x[5]=Gamma2, x[6]=Imax2, x[7]=BG
// =========================================================================

static double pf_pv_doublet_obj(unsigned n, const double *x,
                                double *grad, void *data) {
  PF_PV_Data *f = (PF_PV_Data *)data;
  int N = f->N;
  double *Rs = f->Rs;
  double *PS = f->Ints;
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

static void pf_fit_doublet_pv(int nPts, double *Rs, double *Ints,
                               double *Rfit1, double *Rfit2,
                               double *fitSNR1, double *fitSNR2,
                               double Rstep, double Rmean1,
                               double Rmean2, double Rmid) {
  double x[8], xl[8], xu[8];
  PF_PV_Data f_data;
  f_data.N = nPts;
  f_data.Rs = Rs;
  f_data.Ints = Ints;

  double BG0 = (Ints[0] + Ints[nPts - 1]) / 2;
  if (BG0 < 0) BG0 = 0;
  double MaxI = -1e20;
  for (int i = 0; i < nPts; i++)
    if (Ints[i] > MaxI) MaxI = Ints[i];
  double GammaGuess = Rstep * 3;
  double ImaxGuess = MaxI - BG0;
  if (ImaxGuess < 1e-6) ImaxGuess = 1e-6;

  x[0] = Rmean1;  xl[0] = Rs[0];  xu[0] = Rmid;
  x[1] = Rmean2;  xl[1] = Rmid;   xu[1] = Rs[nPts - 1];
  x[2] = 0.5;     xl[2] = 0;      xu[2] = 1;
  x[3] = GammaGuess;    xl[3] = Rstep / 2;           xu[3] = Rstep * nPts / 2;
  x[4] = ImaxGuess / 2; xl[4] = ImaxGuess / 100;     xu[4] = MaxI * 1.5;
  x[5] = GammaGuess;    xl[5] = Rstep / 2;           xu[5] = Rstep * nPts / 2;
  x[6] = ImaxGuess / 2; xl[6] = ImaxGuess / 100;     xu[6] = MaxI * 1.5;
  x[7] = BG0;           xl[7] = 0;                   xu[7] = (BG0 > 0) ? BG0 * 1.5 : MaxI * 0.5;

  double x_init[8];
  memcpy(x_init, x, 8 * sizeof(double));
  double minf;

  nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 8);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, pf_pv_doublet_obj, &f_data);
  nlopt_set_xtol_rel(opt, 1e-8);
  nlopt_set_ftol_rel(opt, 1e-8);
  nlopt_set_maxeval(opt, 10000);
  int rc = nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);

  if (rc < 0) {
    memcpy(x, x_init, 8 * sizeof(double));
    opt = nlopt_create(NLOPT_LN_NELDERMEAD, 8);
    nlopt_set_lower_bounds(opt, xl);
    nlopt_set_upper_bounds(opt, xu);
    nlopt_set_min_objective(opt, pf_pv_doublet_obj, &f_data);
    nlopt_set_xtol_rel(opt, 1e-8);
    nlopt_set_ftol_rel(opt, 1e-8);
    nlopt_set_maxeval(opt, 10000);
    nlopt_optimize(opt, x, &minf);
    nlopt_destroy(opt);
  }

  *Rfit1 = x[0];
  *Rfit2 = x[1];
  double rmsResid = sqrt(minf / nPts);
  *fitSNR1 = (rmsResid > 0) ? x[4] / rmsResid : 1.0;
  *fitSNR2 = (rmsResid > 0) ? x[6] / rmsResid : 1.0;
}

// =========================================================================
// TCH singlet wrapper: calls fitPeaks() with 1 peak, extracts (Rfit, SNR)
// =========================================================================

static void pf_fit_singlet_tch(int nPts, double *Rs, double *Ints,
                                double *Rfit, double *fitSNR,
                                double Rstep, double Rmean) {
  double peakLoc = Rmean;
  double fitResult[PF_PARAMS_PER_PEAK];
  memset(fitResult, 0, sizeof(fitResult));

  int nFitted = fitPeaks(Rs, Ints, nPts, &peakLoc, 1, Rstep,
                         nPts / 2, fitResult);

  if (nFitted > 0 && fitResult[1] > 0) {
    *Rfit = fitResult[1];   // center
    // SNR from chi²: amplitude / sqrt(chi²)
    double chi_sq = fitResult[6];
    double amplitude = fitResult[0];  // area
    double fwhm = fitResult[4];
    // Convert area to approximate peak height: height ≈ area / (fwhm * ~1.06)
    double height = (fwhm > 0) ? amplitude / (fwhm * 1.064) : amplitude;
    *fitSNR = (chi_sq > 0) ? height / sqrt(chi_sq) : 1.0;
  } else {
    // TCH failed — fall back to pV
    pf_fit_singlet_pv(nPts, Rs, Ints, Rfit, fitSNR, Rstep, Rmean);
  }
}

// =========================================================================
// TCH doublet wrapper: calls fitPeaks() with 2 peaks
// =========================================================================

static void pf_fit_doublet_tch(int nPts, double *Rs, double *Ints,
                                double *Rfit1, double *Rfit2,
                                double *fitSNR1, double *fitSNR2,
                                double Rstep, double Rmean1,
                                double Rmean2, double Rmid) {
  double peakLocs[2] = {Rmean1, Rmean2};
  double fitResults[2 * PF_PARAMS_PER_PEAK];
  memset(fitResults, 0, sizeof(fitResults));

  int nFitted = fitPeaks(Rs, Ints, nPts, peakLocs, 2, Rstep,
                         nPts / 2, fitResults);

  if (nFitted >= 2 && fitResults[1] > 0 && fitResults[PF_PARAMS_PER_PEAK + 1] > 0) {
    *Rfit1 = fitResults[1];
    *Rfit2 = fitResults[PF_PARAMS_PER_PEAK + 1];
    double chi1 = fitResults[6], chi2 = fitResults[PF_PARAMS_PER_PEAK + 6];
    double a1 = fitResults[0], a2 = fitResults[PF_PARAMS_PER_PEAK + 0];
    double f1 = fitResults[4], f2 = fitResults[PF_PARAMS_PER_PEAK + 4];
    double h1 = (f1 > 0) ? a1 / (f1 * 1.064) : a1;
    double h2 = (f2 > 0) ? a2 / (f2 * 1.064) : a2;
    *fitSNR1 = (chi1 > 0) ? h1 / sqrt(chi1) : 1.0;
    *fitSNR2 = (chi2 > 0) ? h2 / sqrt(chi2) : 1.0;
  } else {
    // TCH failed — fall back to pV
    pf_fit_doublet_pv(nPts, Rs, Ints, Rfit1, Rfit2, fitSNR1, fitSNR2,
                      Rstep, Rmean1, Rmean2, Rmid);
  }
}

// =========================================================================
// Unified dispatch: pf_fit_single_peak / pf_fit_doublet_peak
// =========================================================================

void pf_fit_single_peak(int mode, int nPts, double *Rs, double *Ints,
                        double *Rfit, double *fitSNR,
                        double Rstep, double Rmean) {
  if (mode == PF_MODE_TCH)
    pf_fit_singlet_tch(nPts, Rs, Ints, Rfit, fitSNR, Rstep, Rmean);
  else
    pf_fit_singlet_pv(nPts, Rs, Ints, Rfit, fitSNR, Rstep, Rmean);
}

void pf_fit_doublet_peak(int mode, int nPts, double *Rs, double *Ints,
                         double *Rfit1, double *Rfit2,
                         double *fitSNR1, double *fitSNR2,
                         double Rstep, double Rmean1,
                         double Rmean2, double Rmid) {
  if (mode == PF_MODE_TCH)
    pf_fit_doublet_tch(nPts, Rs, Ints, Rfit1, Rfit2, fitSNR1, fitSNR2,
                       Rstep, Rmean1, Rmean2, Rmid);
  else
    pf_fit_doublet_pv(nPts, Rs, Ints, Rfit1, Rfit2, fitSNR1, fitSNR2,
                      Rstep, Rmean1, Rmean2, Rmid);
}
