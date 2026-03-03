//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFit.h - Shared pseudo-Voigt peak fitting module
//
// Extracted from IntegratorFitPeaksGPUStream.cu for reuse in
// IntegratorZarrOMP.c and other CPU-based integrators.
//
// Uses NLOPT (L-BFGS with Nelder-Mead fallback) to fit height-normalized
// pseudo-Voigt profiles to 1D integrated lineouts.

#ifndef PEAKFIT_H
#define PEAKFIT_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_PEAK_LOCATIONS_PF 200

// --- Data Structures ---

// Data passed to the NLOPT objective function
typedef struct {
  int nrBins;
  const double *R;   // Radial positions (X-axis)
  const double *Int; // Intensity values  (Y-axis)
} PF_DataFit;

// Detected / specified peak candidate
typedef struct {
  int index;        // Index in the R/Int array
  double radius;    // R value at the peak
  double intensity; // Intensity value at the peak
} PF_Peak;

// A fitting job: a contiguous region with one or more peaks
typedef struct {
  int startIndex;
  int endIndex;
  int numPeaks;
  PF_Peak *peaks; // Pointer to the first peak in this job
} PF_FitJob;

// Per-peak output (7 doubles, matches GPU binary format):
//   [0] Imax        - peak height
//   [1] Background  - local background
//   [2] Mix         - Lorentzian mixing fraction (0=Gauss, 1=Lorentz)
//   [3] Center      - fitted R center
//   [4] Sigma       - Gaussian-equivalent sigma
//   [5] GoF         - goodness-of-fit (chi^2 / dof)
//   [6] Area        - analytical peak area
#define PF_PARAMS_PER_PEAK 7

// --- Public API ---

// Top-level convenience function: fit all peaks in a 1D lineout.
//
// Parameters:
//   R              - R bin centers             [nRBins]
//   intensity      - eta-summed 1D lineout     [nRBins]
//   nRBins         - number of R bins
//   peakLocations  - target R positions        [nPeaks]
//   nPeaks         - number of peaks to fit
//   RBinSize       - radial bin width
//   fitROIPadding  - bins of padding around each peak ROI
//   outFitParams   - output buffer             [nPeaks * PF_PARAMS_PER_PEAK]
//
// Returns: number of successfully fitted peaks (may be < nPeaks)
int fitPeaksForLineout(const double *R, const double *intensity, int nRBins,
                       const double *peakLocations, int nPeaks, double RBinSize,
                       int fitROIPadding, double *outFitParams);

// Height-normalized pseudo-Voigt model evaluation + analytical areas.
//
// Model: I(R) = BG + sum_j Imax_j * (m_j * L(R-c_j) + (1-m_j) * G(R-c_j))
//   L(x) = 1 / (1 + 4*(x/Gamma)^2)
//   G(x) = exp(-4*ln2*(x/Gamma)^2)
//
// params layout: [Imax0, Mix0, Center0, Gamma0, ..., BG]
void pf_calculate_model_and_area(int n_peaks, const double *params,
                                 int n_points, const double *R_values,
                                 double *out_model_curve,
                                 double *out_peak_areas);

// NLOPT objective function (sum of squared residuals) with analytical
// gradients.
double pf_problem_function_global_bg(unsigned n, const double *x, double *grad,
                                     void *fdat);

// Estimate initial parameters (FWHM, background, amplitude) for a peak.
// Returns estimated FWHM in bin units.
double pf_estimate_initial_params(const double *intensity_data, int n_points,
                                  int peak_idx_local, double *out_bg_guess,
                                  double *out_amp_guess);

// Savitzky-Golay smoothing filter (window sizes 5, 7, or 9).
void pf_smoothData(const double *in, double *out, int N, int W);

// Sort helper for peaks
int pf_comparePeaksByIndex(const void *a, const void *b);

#ifdef __cplusplus
}
#endif

#endif // PEAKFIT_H
