//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFit.c - Shared pseudo-Voigt peak fitting module
//
// Extracted from IntegratorFitPeaksGPUStream.cu for CPU-based peak fitting
// in IntegratorZarrOMP and other integrators.

#include "PeakFit.h"
#include <nlopt.h>

// =========================================================================
// Model evaluation + analytical area
// =========================================================================
void pf_calculate_model_and_area(int n_peaks, const double *params,
                                 int n_points, const double *R_values,
                                 double *out_model_curve,
                                 double *out_peak_areas) {
  const double bg = params[n_peaks * 4];
  const double C0 = 4.0 * log(2.0);

  for (int i = 0; i < n_points; ++i) {
    out_model_curve[i] = bg;
  }

  for (int pN = 0; pN < n_peaks; ++pN) {
    double Imax = params[pN * 4 + 0];
    double m = fmax(0.0, fmin(1.0, params[pN * 4 + 1]));
    double c = params[pN * 4 + 2];
    double G = fmax(1e-9, params[pN * 4 + 3]);
    double invG2 = 1.0 / (G * G);

    for (int i = 0; i < n_points; ++i) {
      double diff = R_values[i] - c;
      double diff_sq = diff * diff;
      double L = 1.0 / (1.0 + 4.0 * diff_sq * invG2);
      double Gaus = exp(-C0 * diff_sq * invG2);
      out_model_curve[i] += Imax * (m * L + (1.0 - m) * Gaus);
    }

    if (out_peak_areas != NULL) {
      out_peak_areas[pN] =
          Imax * G / 2.0 * (m * M_PI + (1.0 - m) * sqrt(M_PI / log(2.0)));
    }
  }
}

// =========================================================================
// NLOPT objective: sum of squared residuals with analytical gradients
// =========================================================================
double pf_problem_function_global_bg(unsigned n, const double *x, double *grad,
                                     void *fdat) {
  const PF_DataFit *d = (const PF_DataFit *)fdat;
  const int Np = d->nrBins;
  const double *Rs = d->R;
  const double *Is = d->Int;
  const int nP = (n - 1) / 4;
  const double C0 = 4.0 * log(2.0);

  double *calculated_I = (double *)malloc(Np * sizeof(double));
  if (!calculated_I)
    return INFINITY;
  pf_calculate_model_and_area(nP, x, Np, Rs, calculated_I, NULL);

  double total_sq_error = 0.0;
  for (int i = 0; i < Np; ++i) {
    double residual = calculated_I[i] - Is[i];
    total_sq_error += residual * residual;
  }

  if (grad) {
    memset(grad, 0, n * sizeof(double));
    for (int pN = 0; pN < nP; ++pN) {
      double Imax = x[pN * 4 + 0], m = fmax(0., fmin(1., x[pN * 4 + 1])),
             c = x[pN * 4 + 2], G = fmax(1e-9, x[pN * 4 + 3]);
      double *gA = &grad[pN * 4 + 0], *gm = &grad[pN * 4 + 1],
             *gc = &grad[pN * 4 + 2], *gG = &grad[pN * 4 + 3];
      double G2 = G * G, G3 = G2 * G;
      double invG2 = 1.0 / G2;

      for (int i = 0; i < Np; ++i) {
        double diff = Rs[i] - c, diff_sq = diff * diff;
        double denom = 1.0 + 4.0 * diff_sq * invG2;
        double L = 1.0 / denom;
        double Gaus = exp(-C0 * diff_sq * invG2);
        double residual = calculated_I[i] - Is[i];
        double common = 2.0 * residual;

        *gA += common * (m * L + (1. - m) * Gaus);
        *gm += common * Imax * (L - Gaus);
        double dLdc = 8.0 * diff / (G2 * denom * denom);
        double dGausdc = Gaus * 2.0 * C0 * diff * invG2;
        *gc += common * Imax * (m * dLdc + (1. - m) * dGausdc);
        double dLdG = 8.0 * diff_sq / (G3 * denom * denom);
        double dGausdG = Gaus * 2.0 * C0 * diff_sq / G3;
        *gG += common * Imax * (m * dLdG + (1. - m) * dGausdG);
      }
    }
    for (int i = 0; i < Np; ++i) {
      grad[n - 1] += 2.0 * (calculated_I[i] - Is[i]);
    }
  }

  free(calculated_I);
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
// Top-level fitting function
// =========================================================================
int fitPeaksForLineout(const double *R, const double *intensity, int nRBins,
                       const double *peakLocations, int nPeaks, double RBinSize,
                       int fitROIPadding, double *outFitParams) {
  if (nPeaks <= 0 || nRBins <= 0)
    return 0;

  // Zero output
  memset(outFitParams, 0, nPeaks * PF_PARAMS_PER_PEAK * sizeof(double));

  // Step 1: Identify peak bin indices
  PF_Peak *pks = (PF_Peak *)malloc(nPeaks * sizeof(PF_Peak));
  int validCount = 0;
  for (int p = 0; p < nPeaks; ++p) {
    int bestBin = -1;
    double minDiff = 1e10;
    for (int r = 0; r < nRBins; ++r) {
      double diff = fabs(R[r] - peakLocations[p]);
      if (diff < minDiff) {
        minDiff = diff;
        bestBin = r;
      }
    }
    if (bestBin != -1 && minDiff < RBinSize * 2.0) {
      pks[validCount].index = bestBin;
      pks[validCount].radius = R[bestBin];
      pks[validCount].intensity = intensity[bestBin];
      validCount++;
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
  fitJobs[0].endIndex = (int)fmin(nRBins - 1, pks[0].index + roi_half);
  fitJobs[0].numPeaks = 1;
  fitJobs[0].peaks = &pks[0];
  job_result_indices[0] = 0;
  numJobs = 1;

  for (int i = 1; i < validCount; ++i) {
    int current_roi_start = (int)fmax(0, pks[i].index - roi_half);
    if (current_roi_start <= fitJobs[numJobs - 1].endIndex) {
      // Merge into current job
      fitJobs[numJobs - 1].endIndex =
          (int)fmin(nRBins - 1, pks[i].index + roi_half);
      fitJobs[numJobs - 1].numPeaks++;
    } else {
      // New job
      job_result_indices[numJobs] =
          job_result_indices[numJobs - 1] + fitJobs[numJobs - 1].numPeaks;
      fitJobs[numJobs].startIndex = current_roi_start;
      fitJobs[numJobs].endIndex =
          (int)fmin(nRBins - 1, pks[i].index + roi_half);
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
    int nFitParams = nJobPeaks * 4 + 1; // 4 per peak + 1 background

    double *fitParams = (double *)malloc(nFitParams * sizeof(double));
    double *lowerBounds = (double *)malloc(nFitParams * sizeof(double));
    double *upperBounds = (double *)malloc(nFitParams * sizeof(double));

    // Estimate initial parameters
    double primary_amp_guess = 1.0, primary_bg_guess = 0.0;
    for (int p = 0; p < nJobPeaks; ++p) {
      PF_Peak *peak = &(job->peaks[p]);
      int p_idx_local = peak->index - job->startIndex;
      double bg_g, amp_g;
      double fwhm = pf_estimate_initial_params(
          &intensity[job->startIndex], job->endIndex - job->startIndex + 1,
          p_idx_local, &bg_g, &amp_g);
      double sigma_g = fwhm * RBinSize / 2.355;
      if (sigma_g < RBinSize * 0.5)
        sigma_g = RBinSize * 2.0;
      double gamma_g = sigma_g * 2.355; // Convert sigma to FWHM

      if (p == 0) {
        primary_amp_guess = amp_g;
        primary_bg_guess = bg_g;
      }

      int b = p * 4;
      fitParams[b + 0] = amp_g;
      fitParams[b + 1] = 0.5;          // Mix
      fitParams[b + 2] = peak->radius; // Center
      fitParams[b + 3] = gamma_g;      // FWHM (Gamma)
      lowerBounds[b + 0] = 0;
      lowerBounds[b + 1] = 0;
      lowerBounds[b + 2] = peak->radius - fwhm * RBinSize;
      lowerBounds[b + 3] = RBinSize * 0.5;
      upperBounds[b + 0] = amp_g * 3.0;
      upperBounds[b + 1] = 1.0;
      upperBounds[b + 2] = peak->radius + fwhm * RBinSize;
      upperBounds[b + 3] = (R[job->endIndex] - R[job->startIndex]);
    }
    fitParams[nFitParams - 1] = primary_bg_guess;
    lowerBounds[nFitParams - 1] = -fabs(primary_amp_guess);
    upperBounds[nFitParams - 1] = fabs(primary_amp_guess);

    PF_DataFit fitData;
    fitData.nrBins = job->endIndex - job->startIndex + 1;
    fitData.R = &R[job->startIndex];
    fitData.Int = &intensity[job->startIndex];

    // Primary: L-BFGS
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, nFitParams);
    nlopt_set_lower_bounds(opt, lowerBounds);
    nlopt_set_upper_bounds(opt, upperBounds);
    nlopt_set_min_objective(opt, pf_problem_function_global_bg, &fitData);
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
      nlopt_set_min_objective(opt, pf_problem_function_global_bg, &fitData);
      nlopt_set_maxeval(opt, 5000);
      nlopt_set_maxtime(opt, 30);
      nlopt_set_ftol_rel(opt, 1e-5);
      nlopt_set_xtol_rel(opt, 1e-5);
      rc = nlopt_optimize(opt, fitParams, &minObj);
      nlopt_destroy(opt);
    }

    if (rc >= 0) {
      double dof = (double)(fitData.nrBins - nFitParams);
      double gof = (dof > 0) ? minObj / dof : -1.0;
      int result_start = job_result_indices[i];
      double localBG = fitParams[nFitParams - 1];

      for (int p = 0; p < nJobPeaks; ++p) {
        int out_base = (result_start + p) * PF_PARAMS_PER_PEAK;
        int in_base = p * 4;
        outFitParams[out_base + 0] = fitParams[in_base + 0]; // Imax
        outFitParams[out_base + 1] = localBG;
        outFitParams[out_base + 2] = fitParams[in_base + 1]; // Mix
        outFitParams[out_base + 3] = fitParams[in_base + 2]; // Center
        // Gamma → Gaussian-equiv sigma
        outFitParams[out_base + 4] =
            fitParams[in_base + 3] / (2.0 * sqrt(2.0 * log(2.0)));
        outFitParams[out_base + 5] = gof;
        // Analytical pV area
        {
          double pImax = fitParams[in_base + 0];
          double pMix = fmax(0.0, fmin(1.0, fitParams[in_base + 1]));
          double pGamma = fmax(1e-9, fitParams[in_base + 3]);
          outFitParams[out_base + 6] =
              pImax * pGamma / 2.0 *
              (pMix * M_PI + (1.0 - pMix) * sqrt(M_PI / log(2.0)));
        }
      }
      total_successful += nJobPeaks;
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
