//
// CalibrationCore.h — Shared calibration optimization functions
//
// Extracted from CalibrantPanelShiftsOMP.c to share across calibrant
// executables.  Geometry computations delegate to dg_pixel_to_REta()
// from DetectorGeometry.h (single source of truth).
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef CALIBRATION_CORE_H
#define CALIBRATION_CORE_H

#include "Panel.h"
#include "DetectorGeometry.h"
#include "MIDAS_Math.h"
#include <omp.h>

// ── Globals (defined in each executable, extern'd here) ────────────

extern Panel *panels;
extern int nPanels;
extern int numProcs;
extern long long int NrCalls;

// ── Per-evaluation M-step tracing ──────────────────────────────────

// Open a CSV trace file for per-evaluation logging.  The header is
// written automatically.  Pass NULL to disable tracing.
void calib_set_trace_file(const char *filename);

// Close the trace file (safe to call even if no file is open).
void calib_close_trace_file(void);

// ── Coordinate conversion helpers ──────────────────────────────────

static inline double calib_R4mTtheta(double Ttheta_deg, double Lsd) {
  return Lsd * tan(Ttheta_deg * DG_DEG2RAD);
}

static inline double calib_Ttheta4mR(double R, double Lsd) {
  return atan(R / Lsd) * DG_RAD2DEG;
}

// Convert arrays of (R, Eta) → (Y, Z) in microns (centered at origin).
static inline void calib_YZ4mREta(int n, const double *R, const double *Eta,
                                   double *Y, double *Z) {
  for (int i = 0; i < n; i++) {
    Y[i] = -R[i] * sin(Eta[i] * DG_DEG2RAD);
    Z[i] =  R[i] * cos(Eta[i] * DG_DEG2RAD);
  }
}

// ── Peak profile fitting ───────────────────────────────────────────

// Data for pseudo-Voigt peak fitting NLopt objectives.
struct calib_profile_data {
  int NrPtsForFit;
  double *Rs;
  double *PeakShape;
};

// Fit a pseudo-Voigt singlet profile to a 1D radial intensity slice.
// Returns fitted peak center in *Rfit and fit SNR in *fitSNR.
void calib_fit_peak_shape(int NrPtsForFit, double *Rs, double *PeakShape,
                          double *Rfit, double *fitSNR,
                          double Rstep, double Rmean);

// Fit a pseudo-Voigt doublet profile to a 1D radial intensity slice.
// Peak1 constrained to [Rs[0], Rmid], Peak2 to [Rmid, Rs[end]].
void calib_fit_doublet_peak_shape(int NrPtsForFit, double *Rs,
                                  double *PeakShape,
                                  double *Rfit1, double *Rfit2,
                                  double *fitSNR1, double *fitSNR2,
                                  double Rstep, double Rmean1,
                                  double Rmean2, double Rmid);

// Intensity-weighted centroid over binned pixel data.
void calib_weighted_mean(int nIndices, int *NrEachIndexBin, int **Indices,
                         double *Average, double *R, double *Eta,
                         double *RMean, double *EtaMean);

// ── Geometry optimization ──────────────────────────────────────────

// NLopt data struct for the geometry optimization objective.
struct calib_opt_data {
  int nIndices;
  double *YMean;        // fitted peak Y positions (pixel coords)
  double *ZMean;        // fitted peak Z positions (pixel coords)
  double *IdealTtheta;  // theoretical 2θ per point (degrees)
  double MaxRad;        // RhoD: max ring radius for distortion normalization
  double px;            // pixel size (µm)
  double tx;            // tilt X (fixed, degrees)
  int fixPanel;         // anchored panel index
  double tolRotation;   // panel rotation tolerance
  double *Weights;      // per-point ring normalization weight (NULL=uniform)
  int nBase;            // base parameter count (always ≥ 10)
  int perPanelLsd;      // flag: per-panel ΔLsd
  int perPanelDistort;  // flag: per-panel Δp2
  double *snrWeights;   // per-point SNR weight (NULL=uniform)
  int weightByRadius;   // flag: weight by R/RhoD
  int useL2;            // flag: squared objective
  int fitWavelength;    // flag: wavelength is a fitting parameter
  double *PointDSpacing;// per-point d-spacing (when fitWavelength=1)
  int fitParallax;      // flag: parallax is a fitting parameter
  double trimFraction;  // trimmed mean fraction (1.0 = all)
  double *trimScratch;  // pre-allocated scratch for trim sort (size nIndices)
  const int *skipBin;   // per-point skip mask (NULL = no skipping)
};

// NLopt geometry optimization objective.
// Computes sum of |1 - R_fitted/R_ideal| (or squared) over all points.
// Delegates pixel→(R,η) to dg_pixel_to_REta.
double calib_problem_function(unsigned n, const double *x, double *grad,
                              void *f_data_trial);

// Main optimizer: fit Lsd, BC, tilts, distortion, panel shifts.
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
    const int *skipBin);

// Compute per-bin strain residuals using fitted geometry.
// Delegates pixel→(R,η) to dg_pixel_to_REta.
void calib_correct_tilt_distortion(
    int nIndices, double MaxRad, double *YMean, double *ZMean,
    double *IdealTtheta, double px, double Lsd, double ybc, double zbc,
    double tx, double ty, double tz, double p0, double p1, double p2,
    double p3, double *Etas, double *Diffs, double *RadOuts,
    double *StdDiff, double outlierFactor, int *IsOutlier,
    double p4, double p5, double p6, int OutlierIterations,
    int verbose, double *MeanDiffOut, double parallax,
    const int *skipBin);

// qsort comparator for doubles.
int calib_cmp_double(const void *a, const void *b);

#endif /* CALIBRATION_CORE_H */
