//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFit.h — Shared GSAS-II pseudo-Voigt peak fitting module
//
// Uses NLOPT (L-BFGS with Nelder-Mead fallback) to fit area-normalized
// pseudo-Voigt profiles with Thompson-Cox-Hastings (TCH) mixing.
//
// Parameters per peak match GSAS-II conventions:
//   Area        : integrated intensity (area under peak above background)
//   Center      : peak position (same units as input x-axis)
//   sig         : Gaussian variance (centideg² when x is in degrees)
//   gam         : Lorentzian FWHM (centideg when x is in degrees)
//
// Background: 2-parameter Chebyshev (bg0 + bg1 * x_norm)

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

// Peak fitting mode enum
#define PF_MODE_PV  0  // Height-normalized pseudo-Voigt (CI legacy)
#define PF_MODE_TCH 1  // GSAS-II area-normalized TCH pseudo-Voigt

// Per-peak output: 7 doubles
//   [0] area          - integrated intensity (area above background)
//   [1] center        - fitted peak center (same units as input x)
//   [2] sig           - Gaussian variance (centideg²)
//   [3] gam           - Lorentzian FWHM (centideg)
//   [4] FWHM          - total FWHM (same units as x)
//   [5] eta           - pseudo-Voigt mixing parameter (0=Gaussian,
//   1=Lorentzian) [6] chi_sq        - goodness-of-fit (chi² / dof)
#define PF_PARAMS_PER_PEAK 7

// --- Data Structures ---

// Data passed to the NLOPT objective function
typedef struct {
  int nrBins;
  const double *R;   // X-axis values (e.g. 2θ degrees)
  const double *Int; // Intensity values (Y-axis)
  double x_lo, x_hi; // Window bounds for Chebyshev normalization
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

// --- TCH Mixing ---

// Thompson-Cox-Hastings: compute total FWHM and mixing η
// from Gaussian variance (centideg²) and Lorentzian FWHM (centideg).
// Returns FWHM in degrees, η in [0, 1].
void pf_tch_eta_fwhm(double sig_centideg2, double gam_centideg,
                     double *out_fwhm_deg, double *out_eta);

// --- Model Evaluation ---

// Area-normalized pseudo-Voigt with TCH mixing and Chebyshev background.
//
// params layout: [Area0, Center0, sig0, gam0, ..., bg0, bg1]
//                (4 doubles per peak, then 2 background params)
void pf_calculate_model(int n_peaks, const double *params, int n_points,
                        const double *x_values, double x_lo, double x_hi,
                        double *out_model_curve);

// --- NLOPT Objective ---

// Sum of squared residuals with analytical gradients.
double pf_objective(unsigned n, const double *x, double *grad, void *fdat);

// --- Peak Fitting ---

// Fit all specified peaks in a 1D profile using area-normalized GSAS-II
// pseudo-Voigt profiles. Input x-axis can be any unit (pixels, 2θ degrees).
//
// Parameters:
//   x              - x-axis values                [nBins]
//   intensity      - 1D profile                   [nBins]
//   nBins          - number of data points
//   peakLocations  - target x positions            [nPeaks]
//   nPeaks         - number of peaks to fit
//   xBinSize       - bin width (same units as x)
//   fitROIPadding  - bins of padding around each peak ROI
//   outFitParams   - output buffer                [nPeaks * PF_PARAMS_PER_PEAK]
//
// Returns: number of successfully fitted peaks (may be < nPeaks)
int fitPeaks(const double *x, const double *intensity, int nBins,
             const double *peakLocations, int nPeaks, double xBinSize,
             int fitROIPadding, double *outFitParams);

// --- Utility ---

// Estimate initial parameters (FWHM, background, amplitude) for a peak.
// Returns estimated FWHM in bin units.
double pf_estimate_initial_params(const double *intensity_data, int n_points,
                                  int peak_idx_local, double *out_bg_guess,
                                  double *out_amp_guess);

// Savitzky-Golay smoothing filter (window sizes 5, 7, or 9).
void pf_smoothData(const double *in, double *out, int N, int W);

// Sort helper for peaks
int pf_comparePeaksByIndex(const void *a, const void *b);

// --- SNIP Background ---

// SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
// background estimation (Morháč LLS transform).
void pf_snip_background(const double *intensity, double *background, int nBins,
                        int nIter);

// --- Auto-Detect ---

// Automatic peak detection: smooths, computes negative 2nd derivative,
// finds local maxima, ranks by prominence, returns top `maxPeaks`.
// outPeakLocations receives x-values (not indices).
// Returns: number of peaks found (<= maxPeaks).
int pf_detect_peaks(const double *x, const double *corrected, int nBins,
                    int maxPeaks, double *outPeakLocations,
                    double minSeparation);

// Top-level auto-detect + fit: SNIP → detect → fit → quality filter.
// Returns: number of peaks that pass quality filtering (<= maxPeaks).
int fitPeaksAutoDetect(const double *x, const double *intensity, int nBins,
                       int maxPeaks, double xBinSize, int fitROIPadding,
                       double *outFitParams, int snipIter);

// --- Unified Peak Fitting API (switchable mode) ---

// Fit a single peak to a 1D radial profile (mode-dispatched).
// mode: PF_MODE_PV (height-normalized pV) or PF_MODE_TCH (GSAS-II TCH)
// Returns fitted center in *Rfit, signal-to-noise in *fitSNR,
// and FWHM in *fitFWHM (NULL-safe: pass NULL if not needed).
void pf_fit_single_peak(int mode, int nPts, double *Rs, double *Ints,
                        double *Rfit, double *fitSNR, double *fitFWHM,
                        double Rstep, double Rmean);

// Fit a doublet (two overlapping peaks) in a 1D radial profile.
// mode: PF_MODE_PV or PF_MODE_TCH
// Peak1 constrained to [Rs[0], Rmid], Peak2 to [Rmid, Rs[end]].
// fitFWHM1/fitFWHM2 are NULL-safe.
void pf_fit_doublet_peak(int mode, int nPts, double *Rs, double *Ints,
                         double *Rfit1, double *Rfit2,
                         double *fitSNR1, double *fitSNR2,
                         double *fitFWHM1, double *fitFWHM2,
                         double Rstep, double Rmean1,
                         double Rmean2, double Rmid);

#ifdef __cplusplus
}
#endif

#endif // PEAKFIT_H
