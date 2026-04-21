// CalibPeakFit.h — Standalone CI pseudo-Voigt peak fitting
#ifndef CALIB_PEAK_FIT_H
#define CALIB_PEAK_FIT_H

// Fit a single pseudo-Voigt peak to a 1D radial profile.
// Same algorithm as CalibrationCore.c calib_fit_peak_shape.
// Args: NrPtsForFit, Rs (x-axis), PeakShape (y-axis),
//       Rfit (fitted center), fitSNR (signal-to-noise),
//       Rstep (x bin size), Rmean (initial center guess).
void cpf_fit_peak(int NrPtsForFit, double *Rs, double *PeakShape,
                  double *Rfit, double *fitSNR, double Rstep, double Rmean);

#endif
