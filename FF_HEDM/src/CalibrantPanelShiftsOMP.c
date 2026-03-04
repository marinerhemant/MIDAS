//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  Calibrant.c
//
//
//  Created by Hemant Sharma on 2014/06/18.
//
//
//  Important: row major, starting with y's and going up. Bottom right is 0,0.
//  TODO: Implement proper transformations for rectangular detectors.

#include "DetectorGeometry.h"
#include "FileReader.h"
#include "MIDAS_Math.h"
#include "Panel.h"
#include "midas_paths.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static Panel *panels = NULL;
static int nPanels = 0;

// #define PRINTOPT
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef double pixelvalue;
long long int NrCalls;
long long int NrCallsProfiler;
int NrPixelsGlobal = 2048;
#define OBJ_FUNC_SCALE 1
#define EPS 1E-12

int numProcs;
int skipFrame = 0;

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
extern size_t mapMaskSize;
extern int *mapMask;

size_t mapMaskSize = 0;
int *mapMask;

inline void CalcPeakProfile(int **Indices, int *NrEachIndexBin, int idx,
                            double *Average, double Rmi, double Rma,
                            double EtaMi, double EtaMa, double ybc, double zbc,
                            double px, int NrPixelsY, double *ReturnValue);

inline void CalcPeakProfileParallel(int *Indices, int NrEachIndexBin, int idx,
                                    double *Average, double Rmi, double Rma,
                                    double EtaMi, double EtaMa, double ybc,
                                    double zbc, double px, int NrPixelsY,
                                    double *ReturnValue);

static inline pixelvalue **allocMatrixPX(int nrows, int ncols) {
  pixelvalue **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrixPx(pixelvalue **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline int **allocMatrixInt(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

// dg_calc_eta_angle replaced by dg_calc_eta_angle from DetectorGeometry.h

static inline double R4mTtheta(double Ttheta, double Lsd) {
  return Lsd * tan(deg2rad * Ttheta);
}

static inline double Ttheta4mR(double R, double Lsd) {
  return rad2deg * atan(R / Lsd);
}

static inline void YZ4mREta(int NrElements, double *R, double *Eta, double *Y,
                            double *Z) {
  int i;
  for (i = 0; i < NrElements; i++) {
    Y[i] = -R[i] * sin(Eta[i] * deg2rad);
    Z[i] = R[i] * cos(Eta[i] * deg2rad);
  }
}

static inline void
Car2Pol(int n_hkls, int nEtaBins, int y, int z, double ybc, double zbc,
        double px, double *R, double *Eta, double Rmins[n_hkls],
        double Rmaxs[n_hkls], double EtaBinsLow[nEtaBins],
        double EtaBinsHigh[nEtaBins], int nIndices, int *NrEachIndexbin,
        int **Indices, double tx, double ty, double tz, double p0, double p1,
        double p2, double p3, double RhoD, double Lsd, double p4) {
  int i, j, k, l, counter = 0, ctr = 0;
  for (i = 0; i < nIndices; i++)
    NrEachIndexbin[i] = 0;
  double TRs[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs);
  double Rt_px, EtaS;
  for (i = 0; i < z; i++) {
    for (j = 0; j < y; j++) {
      long long int pixIdx = (long long int)i * y + j;
      // Skip masked pixels
      if (mapMask != NULL && (mapMask[pixIdx / 32] & (1 << (pixIdx % 32)))) {
        R[counter] = 0;
        Eta[counter] = 0;
        counter++;
        continue;
      }
      // Apply panel corrections (dY, dZ, dTheta, dLsd, dP2)
      double pdY = 0, pdZ = 0;
      double dLsd = 0, dP2 = 0;
      int pIdx = GetPanelIndex((double)j, (double)i, nPanels, panels);
      if (pIdx >= 0) {
        ApplyPanelCorrection((double)j, (double)i, &panels[pIdx], &pdY, &pdZ);
        pdY -= (double)j;
        pdZ -= (double)i;
        dLsd = panels[pIdx].dLsd;
        dP2 = panels[pIdx].dP2;
      }
      double ypr = (double)j + pdY;
      double zpr = (double)i + pdZ;
      dg_pixel_to_REta(ypr, zpr, ybc, zbc, TRs, Lsd, RhoD, p0, p1, p2, p3, p4,
                       px, dLsd, dP2, &Rt_px, &EtaS);
      double Rt = Rt_px * px; // convert from pixels to microns
      R[counter] = Rt;
      Eta[counter] = EtaS;
      for (k = 0; k < n_hkls; k++) {
        if (Rt >= (Rmins[k] - px) && Rt <= (Rmaxs[k] + px)) {
          for (l = 0; l < nEtaBins; l++) {
            if (EtaS >= (EtaBinsLow[l] - px / R[counter]) &&
                EtaS <= (EtaBinsHigh[l] + px / R[counter])) {
              Indices[(nEtaBins * k) + l][NrEachIndexbin[(nEtaBins * k) + l]] =
                  (i * NrPixelsGlobal) + j;
              NrEachIndexbin[(nEtaBins * k) + l] += 1;
              ctr++;
              break;
            }
          }
          break;
        }
      }
      counter++;
    }
  }
}

static inline void CalcWeightedMean(int nIndices, int *NrEachIndexBin,
                                    int **Indices, double *Average, double *R,
                                    double *Eta, double *RMean,
                                    double *EtaMean) {
  int i, j, k;
  if (nIndices > 2048) {
    printf("nIndices > 2048\\n");
    return;
  }
  double TotIntensities[2048];
  for (i = 0; i < nIndices; i++) {
    TotIntensities[i] = 0;
    EtaMean[i] = 0;
    RMean[i] = 0;
  }
  for (i = 0; i < nIndices; i++) {
    for (j = 0; j < NrEachIndexBin[i]; j++) {
      TotIntensities[i] += Average[Indices[i][j]];
    }
  }
  for (i = 0; i < nIndices; i++) {
    for (j = 0; j < NrEachIndexBin[i]; j++) {
      RMean[i] +=
          (Average[Indices[i][j]] * R[Indices[i][j]]) / TotIntensities[i];
      EtaMean[i] +=
          (Average[Indices[i][j]] * Eta[Indices[i][j]]) / TotIntensities[i];
    }
  }
}

struct my_profile_func_data {
  int NrPtsForFit;
  double *Rs;
  double *PeakShape;
};

// Height-normalized Pseudo-Voigt singlet objective with analytical gradients.
// Parameters: x[0]=Rcen, x[1]=Mu, x[2]=Gamma(FWHM), x[3]=Imax, x[4]=BG
// L(x) = 1 / (1 + 4*(x/Gamma)^2)         [peaks at 1]
// G(x) = exp(-4*ln2*(x/Gamma)^2)          [peaks at 1]
// I(R) = BG + Imax * (Mu*L + (1-Mu)*G)
static double problem_function_profile(unsigned n, const double *x,
                                       double *grad, void *f_data_trial) {
  struct my_profile_func_data *f_data =
      (struct my_profile_func_data *)f_data_trial;
  int NrPtsForFit = f_data->NrPtsForFit;
  double *Rs = &(f_data->Rs[0]);
  double *PeakShape = &(f_data->PeakShape[0]);
  double Rcen = x[0], Mu = x[1], Gamma = x[2], Imax = x[3], BG = x[4];
  double C0 = 4.0 * log(2.0);
  double Gamma2 = Gamma * Gamma, Gamma3 = Gamma2 * Gamma;
  double invGamma2 = 1.0 / Gamma2;
  double TotalDifferenceIntensity = 0;

  if (grad)
    memset(grad, 0, n * sizeof(double));

  for (int i = 0; i < NrPtsForFit; i++) {
    double dr = Rs[i] - Rcen;
    double dr2 = dr * dr;
    double denom = 1.0 + 4.0 * dr2 * invGamma2;
    double L = 1.0 / denom;
    double G = exp(-C0 * dr2 * invGamma2);
    double CalcIntensity = BG + Imax * (Mu * L + (1.0 - Mu) * G);
    double diff = CalcIntensity - PeakShape[i];
    TotalDifferenceIntensity += diff * diff;

    if (grad) {
      double common = 2.0 * diff;
      // dI/dRcen: dr = R - Rcen, d(dr)/dRcen = -1
      double dLdc = 8.0 * dr / (Gamma2 * denom * denom);
      double dGdc = G * 2.0 * C0 * dr * invGamma2;
      grad[0] += common * Imax * (Mu * dLdc + (1.0 - Mu) * dGdc);
      // dI/dMu
      grad[1] += common * Imax * (L - G);
      // dI/dGamma
      double dLdG = 8.0 * dr2 / (Gamma3 * denom * denom);
      double dGdGamma = G * 2.0 * C0 * dr2 / Gamma3;
      grad[2] += common * Imax * (Mu * dLdG + (1.0 - Mu) * dGdGamma);
      // dI/dImax
      grad[3] += common * (Mu * L + (1.0 - Mu) * G);
      // dI/dBG
      grad[4] += common;
    }
  }
  __atomic_add_fetch(&NrCallsProfiler, 1, __ATOMIC_RELAXED);
#ifdef PRINTOPT
  printf("Peak profiler intensity difference: %f\n", TotalDifferenceIntensity);
#endif
  return TotalDifferenceIntensity;
}

void FitPeakShape(int NrPtsForFit, double Rs[NrPtsForFit],
                  double PeakShape[NrPtsForFit], double *Rfit, double *fitSNR,
                  double Rstep, double Rmean) {
  unsigned n = 5;
  double x[5], xl[5], xu[5];
  struct my_profile_func_data f_data;
  f_data.NrPtsForFit = NrPtsForFit;
  f_data.Rs = &Rs[0];
  f_data.PeakShape = &PeakShape[0];
  double BG0 = (PeakShape[0] + PeakShape[NrPtsForFit - 1]) / 2;
  if (BG0 < 0)
    BG0 = 0;
  double MaxI = -100000;
  int i;
  for (i = 0; i < NrPtsForFit; i++) {
    if (PeakShape[i] > MaxI) {
      MaxI = PeakShape[i];
    }
  }
  // x[0] = Rcen
  x[0] = Rmean;
  xl[0] = Rs[0];
  xu[0] = Rs[NrPtsForFit - 1];
  // x[1] = Mu (mixing: 0=pure Gaussian, 1=pure Lorentzian)
  x[1] = 0.5;
  xl[1] = 0;
  xu[1] = 1;
  // x[2] = Gamma (FWHM, shared by Gaussian and Lorentzian)
  double GammaGuess = Rstep * 3;
  x[2] = GammaGuess;
  xl[2] = Rstep / 2;
  xu[2] = Rstep * NrPtsForFit / 2;
  // x[3] = Imax (peak height above background)
  x[3] = MaxI - BG0;
  xl[3] = (MaxI - BG0) / 100;
  xu[3] = MaxI * 1.5;
  // x[4] = BG
  x[4] = BG0;
  xl[4] = 0;
  xu[4] = (BG0 > 0) ? BG0 * 1.5 : MaxI * 0.5;
  struct my_profile_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_profile_func_data *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function_profile;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double minf, MeanDiff;
  double x_init[5];
  memcpy(x_init, x, n * sizeof(double));
  int rc = run_nlopt_optimization(NLOPT_LD_LBFGS, &config);
  if (rc < 0) {
    // L-BFGS failed — restore initial guesses and retry with Nelder-Mead
    memcpy(x, x_init, n * sizeof(double));
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  }
  minf = config.min_function_val;
  MeanDiff = sqrt(minf) / (NrPtsForFit);
  *Rfit = x[0];
  // SNR = Imax / rms residual
  double rmsResid = sqrt(minf / NrPtsForFit);
  if (rmsResid > 0)
    *fitSNR = x[3] / rmsResid;
  else
    *fitSNR = 1.0;
}

// Height-normalized Pseudo-Voigt doublet objective with analytical gradients.
// Two peaks sharing Mu and BG, each with independent Gamma and Imax.
// x[0]=Rcen1, x[1]=Rcen2, x[2]=Mu
// x[3]=Gamma1, x[4]=Imax1
// x[5]=Gamma2, x[6]=Imax2
// x[7]=BG
static double problem_function_doublet_profile(unsigned n, const double *x,
                                               double *grad,
                                               void *f_data_trial) {
  struct my_profile_func_data *f_data =
      (struct my_profile_func_data *)f_data_trial;
  int NrPtsForFit = f_data->NrPtsForFit;
  double *Rs = &(f_data->Rs[0]);
  double *PeakShape = &(f_data->PeakShape[0]);
  double Rcen1 = x[0], Rcen2 = x[1], Mu = x[2];
  double Gamma1 = x[3], Imax1 = x[4];
  double Gamma2 = x[5], Imax2 = x[6];
  double BG = x[7];
  double C0 = 4.0 * log(2.0);
  double G1_2 = Gamma1 * Gamma1, G1_3 = G1_2 * Gamma1;
  double G2_2 = Gamma2 * Gamma2, G2_3 = G2_2 * Gamma2;
  double invG1_2 = 1.0 / G1_2;
  double invG2_2 = 1.0 / G2_2;
  double TotalDiff = 0;

  if (grad)
    memset(grad, 0, n * sizeof(double));

  for (int i = 0; i < NrPtsForFit; i++) {
    double dr1 = Rs[i] - Rcen1;
    double dr2 = Rs[i] - Rcen2;
    double dr1_2 = dr1 * dr1;
    double dr2_2 = dr2 * dr2;
    double den1 = 1.0 + 4.0 * dr1_2 * invG1_2;
    double den2 = 1.0 + 4.0 * dr2_2 * invG2_2;
    double L1 = 1.0 / den1;
    double Gau1 = exp(-C0 * dr1_2 * invG1_2);
    double L2 = 1.0 / den2;
    double Gau2 = exp(-C0 * dr2_2 * invG2_2);
    double CalcIntensity = BG + Imax1 * (Mu * L1 + (1 - Mu) * Gau1) +
                           Imax2 * (Mu * L2 + (1 - Mu) * Gau2);
    double diff = CalcIntensity - PeakShape[i];
    TotalDiff += diff * diff;

    if (grad) {
      double common = 2.0 * diff;
      // Peak 1 gradients
      double dL1dc = 8.0 * dr1 / (G1_2 * den1 * den1);
      double dG1dc = Gau1 * 2.0 * C0 * dr1 * invG1_2;
      grad[0] +=
          common * Imax1 * (Mu * dL1dc + (1.0 - Mu) * dG1dc); // dI/dRcen1
      double dL1dG = 8.0 * dr1_2 / (G1_3 * den1 * den1);
      double dG1dG = Gau1 * 2.0 * C0 * dr1_2 / G1_3;
      grad[3] +=
          common * Imax1 * (Mu * dL1dG + (1.0 - Mu) * dG1dG); // dI/dGamma1
      grad[4] += common * (Mu * L1 + (1.0 - Mu) * Gau1);      // dI/dImax1
      // Peak 2 gradients
      double dL2dc = 8.0 * dr2 / (G2_2 * den2 * den2);
      double dG2dc = Gau2 * 2.0 * C0 * dr2 * invG2_2;
      grad[1] +=
          common * Imax2 * (Mu * dL2dc + (1.0 - Mu) * dG2dc); // dI/dRcen2
      double dL2dG = 8.0 * dr2_2 / (G2_3 * den2 * den2);
      double dG2dG = Gau2 * 2.0 * C0 * dr2_2 / G2_3;
      grad[5] +=
          common * Imax2 * (Mu * dL2dG + (1.0 - Mu) * dG2dG); // dI/dGamma2
      grad[6] += common * (Mu * L2 + (1.0 - Mu) * Gau2);      // dI/dImax2
      // Shared: dI/dMu (contributions from both peaks)
      grad[2] += common * (Imax1 * (L1 - Gau1) + Imax2 * (L2 - Gau2));
      // dI/dBG
      grad[7] += common;
    }
  }
  __atomic_add_fetch(&NrCallsProfiler, 1, __ATOMIC_RELAXED);
  return TotalDiff;
}

void FitDoubletPeakShape(int NrPtsForFit, double *Rs, double *PeakShape,
                         double *Rfit1, double *Rfit2, double *fitSNR1,
                         double *fitSNR2, double Rstep, double Rmean1,
                         double Rmean2) {
  unsigned n = 8;
  double x[8], xl[8], xu[8];
  struct my_profile_func_data f_data;
  f_data.NrPtsForFit = NrPtsForFit;
  f_data.Rs = Rs;
  f_data.PeakShape = PeakShape;
  double BG0 = (PeakShape[0] + PeakShape[NrPtsForFit - 1]) / 2;
  if (BG0 < 0)
    BG0 = 0;
  double MaxI = -100000;
  int i;
  for (i = 0; i < NrPtsForFit; i++) {
    if (PeakShape[i] > MaxI)
      MaxI = PeakShape[i];
  }
  double GammaGuess = Rstep * 3;
  double ImaxGuess = MaxI - BG0;
  if (ImaxGuess < 1e-6)
    ImaxGuess = 1e-6;
  // Rcen1
  x[0] = Rmean1;
  xl[0] = Rs[0];
  xu[0] = Rs[NrPtsForFit - 1];
  // Rcen2
  x[1] = Rmean2;
  xl[1] = Rs[0];
  xu[1] = Rs[NrPtsForFit - 1];
  // Mu (shared)
  x[2] = 0.5;
  xl[2] = 0;
  xu[2] = 1;
  // Gamma1 (FWHM of peak 1)
  x[3] = GammaGuess;
  xl[3] = Rstep / 2;
  xu[3] = Rstep * NrPtsForFit / 4;
  // Imax1
  x[4] = ImaxGuess * 0.6;
  xl[4] = ImaxGuess / 100;
  xu[4] = MaxI * 1.5;
  // Gamma2 (FWHM of peak 2)
  x[5] = GammaGuess;
  xl[5] = Rstep / 2;
  xu[5] = Rstep * NrPtsForFit / 4;
  // Imax2
  x[6] = ImaxGuess * 0.6;
  xl[6] = ImaxGuess / 100;
  xu[6] = MaxI * 1.5;
  // BG (shared)
  x[7] = BG0;
  xl[7] = 0;
  xu[7] = (BG0 > 0) ? BG0 * 2.0 : 1.0;

  struct my_profile_func_data *f_datat = &f_data;
  void *trp = (struct my_profile_func_data *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function_doublet_profile;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 10000;
  config.max_time_seconds = 60;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double x_init[8];
  memcpy(x_init, x, n * sizeof(double));
  int rc = run_nlopt_optimization(NLOPT_LD_LBFGS, &config);
  if (rc < 0) {
    // L-BFGS failed — restore initial guesses and retry with Nelder-Mead
    memcpy(x, x_init, n * sizeof(double));
    run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  }
  *Rfit1 = x[0];
  *Rfit2 = x[1];
  // SNR for each peak: Imax / rms_residual
  double rmsResid = sqrt(config.min_function_val / NrPtsForFit);
  if (rmsResid > 0) {
    *fitSNR1 = x[4] / rmsResid;
    *fitSNR2 = x[6] / rmsResid;
  } else {
    *fitSNR1 = 1.0;
    *fitSNR2 = 1.0;
  }
}

void CalcFittedMean(int nIndices, int *NrEachIndexBin, int **Indices,
                    double *Average, double *R, double *Eta, double *RMean,
                    double *EtaMean, double *FitSNR, int NrPtsForFit,
                    double *IdealRmins, double *IdealRmaxs, int nBinsPerRing,
                    double ybc, double zbc, double px, int NrPixelsY,
                    int NrPixelsZ, double EtaBinsLow[nBinsPerRing],
                    double EtaBinsHigh[nBinsPerRing], int *doubletFlag,
                    int *doubletPartner) {
  int NrPixels =
      NrPixelsY > NrPixelsZ ? NrPixelsY : NrPixelsZ; // square image stride
  int idxThis;
#pragma omp parallel for num_threads(numProcs) private(idxThis)                \
    schedule(dynamic)
  for (idxThis = 0; idxThis < nIndices; idxThis++) {
    int ringIdx = idxThis / nBinsPerRing;
    int etaBin = idxThis % nBinsPerRing;

    // Doublet secondary ring: skip — the primary ring thread writes our RMean
    if (doubletFlag != NULL && doubletFlag[ringIdx] == 2) {
      continue;
    }

    int j, k, BinNr;
    double *PeakShape, Rmin, Rmax, Rstep, Rthis, *Rs;
    double Rfit = 0;
    int *IndicesThis;
    int NrIndicesThis = NrEachIndexBin[idxThis];
    int **Idxs;
    double EtaMi, EtaMa, Rmi, Rma;
    double RetVal;
    double AllZero;
    double ytr, ztr;

    // Check for doublet primary: merge indices from both rings
    int isDoublet = (doubletFlag != NULL && doubletFlag[ringIdx] == 1);
    int partnerIdx = -1;
    int NrIndicesPartner = 0;
    int *IndicesPartner = NULL;
    if (isDoublet) {
      partnerIdx = doubletPartner[ringIdx] * nBinsPerRing + etaBin;
      NrIndicesPartner = NrEachIndexBin[partnerIdx];
    }

    // If no pixels for this bin (or both bins in doublet case), skip
    if (NrIndicesThis == 0 && (!isDoublet || NrIndicesPartner == 0)) {
      RMean[idxThis] = 0;
      FitSNR[idxThis] = 0;
      if (isDoublet) {
        RMean[partnerIdx] = 0;
        FitSNR[partnerIdx] = 0;
      }
      continue;
    }

    // For doublet: use merged Rmin/Rmax window
    if (isDoublet) {
      double RminPartner = IdealRmins[partnerIdx];
      double RmaxPartner = IdealRmaxs[partnerIdx];
      Rmin = (IdealRmins[idxThis] < RminPartner) ? IdealRmins[idxThis]
                                                 : RminPartner;
      Rmax = (IdealRmaxs[idxThis] > RmaxPartner) ? IdealRmaxs[idxThis]
                                                 : RmaxPartner;
    } else {
      Rmin = IdealRmins[idxThis];
      Rmax = IdealRmaxs[idxThis];
    }

    // Compute NrPtsForFit for this bin based on the window size
    int NrPtsLocal = (int)((Rmax - Rmin) / px) * 4;
    if (NrPtsLocal < NrPtsForFit)
      NrPtsLocal = NrPtsForFit;

    PeakShape = calloc(NrPtsLocal, sizeof(*PeakShape));
    Rs = calloc(NrPtsLocal, sizeof(*Rs));
    Idxs = allocMatrixInt(1, NrPtsLocal);
    double *Etas = malloc(NrPtsLocal * sizeof(double));
    for (j = 0; j < NrPtsLocal; j++)
      Idxs[0][j] = j;

    AllZero = 1;
    Rstep = (Rmax - Rmin) / NrPtsLocal;
    BinNr = etaBin;
    EtaMi = -180 + BinNr * (360.0 / nBinsPerRing);
    EtaMa = -180 + (BinNr + 1) * (360.0 / nBinsPerRing);

    // Detector boundary check using the (possibly merged) Rmax
    ytr = ybc - (-Rmax * sin(EtaMa * deg2rad)) / px;
    ztr = zbc + (Rmax * cos(EtaMa * deg2rad)) / px;
    if (((int)ytr > NrPixelsY - 3) || ((int)ytr < 3) ||
        ((int)ztr > NrPixelsZ - 3) || ((int)ztr < 3)) {
      RMean[idxThis] = 0;
      FitSNR[idxThis] = 0;
      if (isDoublet) {
        RMean[partnerIdx] = 0;
        FitSNR[partnerIdx] = 0;
      }
      goto cleanup;
    }
    ytr = ybc - (-Rmax * sin(EtaMi * deg2rad)) / px;
    ztr = zbc + (Rmax * cos(EtaMi * deg2rad)) / px;
    if (((int)ytr > NrPixelsY - 3) || ((int)ytr < 3) ||
        ((int)ztr > NrPixelsZ - 3) || ((int)ztr < 3)) {
      RMean[idxThis] = 0;
      FitSNR[idxThis] = 0;
      if (isDoublet) {
        RMean[partnerIdx] = 0;
        FitSNR[partnerIdx] = 0;
      }
      goto cleanup;
    }
    EtaMean[idxThis] = (EtaMi + EtaMa) / 2;
    if (isDoublet)
      EtaMean[partnerIdx] = EtaMean[idxThis];

    // Merge indices from both rings for the profile calculation
    {
      int totalIndices = NrIndicesThis + NrIndicesPartner;
      int actualIndices = (totalIndices > 0) ? totalIndices : 1;
      IndicesThis = malloc(actualIndices * sizeof(*IndicesThis));
      for (j = 0; j < NrIndicesThis; j++)
        IndicesThis[j] = Indices[idxThis][j];
      if (isDoublet) {
        for (j = 0; j < NrIndicesPartner; j++)
          IndicesThis[NrIndicesThis + j] = Indices[partnerIdx][j];
      }
      int NrIndicesMerged = totalIndices;

      // Build the radial profile over the (merged) window
      for (j = 0; j < NrPtsLocal; j++) {
        PeakShape[j] = 0;
        Rs[j] = Rmin + (j * Rstep) + Rstep / 2;
        Rmi = Rs[j] - Rstep / 2;
        Rma = Rs[j] + Rstep / 2;
        CalcPeakProfileParallel(IndicesThis, NrIndicesMerged, idxThis, Average,
                                Rmi, Rma, EtaMi, EtaMa, ybc, zbc, px, NrPixels,
                                &RetVal);
        PeakShape[j] = RetVal;
        if (RetVal != 0)
          AllZero = 0;
      }

      for (j = 0; j < NrPtsLocal; j++)
        Etas[j] = EtaMean[idxThis];

      if (AllZero != 1) {
        if (isDoublet) {
          // For doublet: compute weighted means in each ring's sub-window
          // as initial guesses, then fit the doublet
          double Rmean1 = 0, Rmean2 = 0;
          double W1 = 0, W2 = 0;
          double Rmid = (IdealRmaxs[idxThis] + IdealRmins[partnerIdx]) / 2;
          for (j = 0; j < NrPtsLocal; j++) {
            if (Rs[j] < Rmid) {
              Rmean1 += Rs[j] * PeakShape[j];
              W1 += PeakShape[j];
            } else {
              Rmean2 += Rs[j] * PeakShape[j];
              W2 += PeakShape[j];
            }
          }
          if (W1 > 0)
            Rmean1 /= W1;
          else
            Rmean1 = (IdealRmins[idxThis] + IdealRmaxs[idxThis]) / 2;
          if (W2 > 0)
            Rmean2 /= W2;
          else
            Rmean2 = (IdealRmins[partnerIdx] + IdealRmaxs[partnerIdx]) / 2;

          double Rfit1 = 0, Rfit2 = 0;
          double snr1 = 0, snr2 = 0;
          FitDoubletPeakShape(NrPtsLocal, Rs, PeakShape, &Rfit1, &Rfit2, &snr1,
                              &snr2, Rstep, Rmean1, Rmean2);
          RMean[idxThis] = Rfit1;
          RMean[partnerIdx] = Rfit2;
          FitSNR[idxThis] = snr1;
          FitSNR[partnerIdx] = snr2;
        } else {
          // Singlet: existing path
          double *Rm = malloc(sizeof(*Rm));
          double *Etam = malloc(sizeof(*Etam));
          int *NrPts = malloc(sizeof(*NrPts));
          NrPts[0] = NrPtsLocal;
          CalcWeightedMean(1, NrPts, Idxs, PeakShape, Rs, Etas, Rm, Etam);
          double Rmean = Rm[0];
          double snr = 0;
          FitPeakShape(NrPtsLocal, Rs, PeakShape, &Rfit, &snr, Rstep, Rmean);
          RMean[idxThis] = Rfit;
          FitSNR[idxThis] = snr;
          free(NrPts);
          free(Rm);
          free(Etam);
        }
      } else {
        RMean[idxThis] = 0;
        FitSNR[idxThis] = 0;
        if (isDoublet) {
          RMean[partnerIdx] = 0;
          FitSNR[partnerIdx] = 0;
        }
      }
      free(IndicesThis);
    }

  cleanup:
    free(PeakShape);
    free(Rs);
    free(Etas);
    FreeMemMatrixInt(Idxs, 1);
  }
}

struct my_func_data {
  int nIndices;
  double *YMean;
  double *ZMean;
  double *IdealTtheta;
  double MaxRad;
  double px;
  double tx;
  int fixPanel;
  double tolRotation;
  double *Weights;     // per-point weight for ring normalization (NULL=uniform)
  int nBase;           // always 10 (p4 permanently enabled)
  int perPanelLsd;     // flag
  int perPanelDistort; // flag
  double *snrWeights;  // per-point SNR-based weight (NULL=uniform)
  int weightByRadius;  // flag for Feature 4
  int useL2;           // flag: 1 = squared objective
};

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  int MaxRad = f_data->MaxRad;
  int nIndices = f_data->nIndices;
  double *YMean, *ZMean, *IdealTtheta, px;
  YMean = &(f_data->YMean[0]);
  ZMean = &(f_data->ZMean[0]);
  IdealTtheta = &(f_data->IdealTtheta[0]);
  px = f_data->px;
  double Lsd, ybc, zbc, tx, ty, tz, p0, p1, p2, p3, txr, tyr, tzr;
  Lsd = x[0];
  ybc = x[1];
  zbc = x[2];
  tx = f_data->tx;
  ty = x[3];
  tz = x[4];
  p0 = x[5];
  p1 = x[6];
  p2 = x[7];
  p3 = x[8];
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  int i;
  double TotalDiff = 0;

#pragma omp parallel for num_threads(numProcs) reduction(+ : TotalDiff)
  for (i = 0; i < nIndices; i++) {
    double n0 = 2, n1 = 4, n2 = 2, Yc, Zc;
    double Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, EtaT;
    double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
    int pIdx = -1;
    if (nPanels > 0) {
      pIdx = GetPanelIndex((double)YMean[i], (double)ZMean[i], nPanels, panels);
      if (pIdx == -1) {
        continue;
      }
    }

    dY = 0;
    dZ = 0;
    dTheta = 0;
    int stride = (f_data->tolRotation > 1e-12) ? 3 : 2;
    if (f_data->perPanelLsd)
      stride++;
    if (f_data->perPanelDistort)
      stride++;
    if (n > f_data->nBase) {
      if (pIdx != f_data->fixPanel) {
        int logicalIndex = (pIdx < f_data->fixPanel) ? pIdx : pIdx - 1;
        int xIdx = f_data->nBase + logicalIndex * stride;
        dY = x[xIdx];
        dZ = x[xIdx + 1];
        int off = 2;
        if (f_data->tolRotation > 1e-12)
          dTheta = x[xIdx + off++];
        if (f_data->perPanelLsd)
          dLsd = x[xIdx + off++];
        if (f_data->perPanelDistort)
          dP2 = x[xIdx + off++];
      }
    }
    double rawY = YMean[i], rawZ = ZMean[i];
    if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
      double cY = panels[pIdx].centerY;
      double cZ = panels[pIdx].centerZ;
      double cosT = cos(deg2rad * dTheta);
      double sinT = sin(deg2rad * dTheta);
      double dy = rawY - cY, dz = rawZ - cZ;
      rawY = cY + dy * cosT - dz * sinT;
      rawZ = cZ + dy * sinT + dz * cosT;
    }
    Yc = -(rawY + dY - ybc) * px;
    Zc = (rawZ + dZ - zbc) * px;
    double ABC[3] = {0, Yc, Zc};
    double ABCPr[3];
    MatrixMultF(TRs, ABC, ABCPr);
    double panelLsd = Lsd + dLsd;
    double XYZ[3] = {panelLsd + ABCPr[0], ABCPr[1], ABCPr[2]};
    Rad = (panelLsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
    Eta = dg_calc_eta_angle(XYZ[1], XYZ[2]);
    RNorm = Rad / MaxRad;
    EtaT = 90 - Eta;
    double panelP2 = p2 + dP2;
    DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                  (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                  (panelP2 * (pow(RNorm, n2)));
    if (f_data->nBase > 9) {
      double p4 = x[9];
      DistortFunc += p4 * pow(RNorm, 6.0);
    }
    DistortFunc += 1;
    Rcorr = Rad * DistortFunc;
    RIdeal = panelLsd * tan(deg2rad * IdealTtheta[i]);
    double Diff = 1 - (Rcorr / RIdeal);
    double w = (f_data->Weights != NULL) ? f_data->Weights[i] : 1.0;
    if (f_data->weightByRadius)
      w *= RNorm;
    if (f_data->snrWeights != NULL)
      w *= f_data->snrWeights[i];
    TotalDiff += (f_data->useL2 ? Diff * Diff : fabs(Diff)) * w;
  }
  TotalDiff *= OBJ_FUNC_SCALE;
  NrCalls++;
  // printf("Mean strain: %0.40f\n", TotalDiff / (OBJ_FUNC_SCALE * nIndices));
#ifdef PRINTOPT
  printf("Mean strain: %0.40f\n", TotalDiff / (OBJ_FUNC_SCALE * nIndices));
#endif
  return TotalDiff;
}

void FitTiltBCLsd(int nIndices, double *YMean, double *ZMean,
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
                  int verbose, int L2Objective) {
  int nBase = 10;
  unsigned n = nBase;
  if (tolShifts > EPS && nPanels > 1) {
    int stride = (tolRotation > EPS) ? 3 : 2;
    if (PerPanelLsd)
      stride++;
    if (PerPanelDistortion)
      stride++;
    n += (nPanels - 1) * stride;
  }
  struct my_func_data f_data;
  f_data.nIndices = nIndices;
  f_data.YMean = &YMean[0];
  f_data.ZMean = &ZMean[0];
  f_data.IdealTtheta = &IdealTtheta[0];
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
  double x[n], xl[n], xu[n];
  x[0] = Lsd;
  xl[0] = Lsd - tolLsd;
  xu[0] = Lsd + tolLsd;
  x[1] = ybc;
  xl[1] = ybc - tolBC;
  xu[1] = ybc + tolBC;
  x[2] = zbc;
  xl[2] = zbc - tolBC;
  xu[2] = zbc + tolBC;
  x[3] = tyin;
  xl[3] = tyin - tolTilts;
  xu[3] = tyin + tolTilts;
  x[4] = tzin;
  xl[4] = tzin - tolTilts;
  xu[4] = tzin + tolTilts;
  x[5] = p0in;
  xl[5] = p0in - tolP0;
  xu[5] = p0in + tolP0;
  x[6] = p1in;
  xl[6] = p1in - tolP1;
  xu[6] = p1in + tolP1;
  x[7] = p2in;
  xl[7] = p2in - tolP2;
  xu[7] = p2in + tolP2;
  x[8] = p3in;
  xl[8] = p3in - tolP3;
  xu[8] = p3in + tolP3;
  x[9] = p4in;
  xl[9] = p4in - tolP4;
  xu[9] = p4in + tolP4;

  if (tolShifts > EPS && nPanels > 1) {
    int panelCounts[nPanels];
    memset(panelCounts, 0, nPanels * sizeof(int));
    for (int i = 0; i < nIndices; i++) {
      int pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
      if (pIdx >= 0 && pIdx < nPanels) {
        panelCounts[pIdx]++;
      }
    }

    for (int i = 0; i < nPanels; i++) {
      if (i == fixPanel)
        continue;
      // We map Panel i to x indices
      // int logicalIndex = (i < fixPanel) ? i : i - 1; // Not needed with
      // direct pointer, but kept for logic if derived Since we are iterating
      // and filling linear buffer x, we can just track p_idx.
    }

    if (nPanels > 1) {
      int p_idx = nBase;
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel)
          continue;

        x[p_idx] = panels[i].dY;

        if (panelCounts[i] < minIndices) {
          x[p_idx] = 0;
          xl[p_idx] = 0;
          xu[p_idx] = 0;
        } else {
          xl[p_idx] = x[p_idx] - tolShifts;
          xu[p_idx] = x[p_idx] + tolShifts;
        }
        p_idx++;

        x[p_idx] = panels[i].dZ;

        if (panelCounts[i] < minIndices) {
          x[p_idx] = 0;
          xl[p_idx] = 0;
          xu[p_idx] = 0;
        } else {
          xl[p_idx] = x[p_idx] - tolShifts;
          xu[p_idx] = x[p_idx] + tolShifts;
        }
        p_idx++;

        if (tolRotation > EPS) {
          x[p_idx] = panels[i].dTheta;
          if (panelCounts[i] < minIndices) {
            x[p_idx] = 0;
            xl[p_idx] = 0;
            xu[p_idx] = 0;
          } else {
            xl[p_idx] = x[p_idx] - tolRotation;
            xu[p_idx] = x[p_idx] + tolRotation;
          }
          p_idx++;
        }

        // Per-panel dLsd
        if (PerPanelLsd) {
          x[p_idx] = panels[i].dLsd;
          if (panelCounts[i] < minIndices) {
            x[p_idx] = 0;
            xl[p_idx] = 0;
            xu[p_idx] = 0;
          } else {
            xl[p_idx] = x[p_idx] - tolLsdPanel;
            xu[p_idx] = x[p_idx] + tolLsdPanel;
          }
          p_idx++;
        }

        // Per-panel dP2
        if (PerPanelDistortion) {
          x[p_idx] = panels[i].dP2;
          if (panelCounts[i] < minIndices) {
            x[p_idx] = 0;
            xl[p_idx] = 0;
            xu[p_idx] = 0;
          } else {
            xl[p_idx] = x[p_idx] - tolP2Panel;
            xu[p_idx] = x[p_idx] + tolP2Panel;
          }
          p_idx++;
        }
      }
    }
  }
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = problem_function;
  config.obj_data = trp;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  double minf;
  run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  minf = config.min_function_val;
  *MeanDiff = minf / (OBJ_FUNC_SCALE * nIndices);

  // 1. Update output parameters with optimized values
  *LsdFit = x[0];
  *ybcFit = x[1];
  *zbcFit = x[2];
  *ty = x[3];
  *tz = x[4];
  *p0 = x[5];
  *p1 = x[6];
  *p2 = x[7];
  *p3 = x[8];
  if (nBase > 9 && p4Out)
    *p4Out = x[9];

  // 2. Update panel shifts if applicable
  if (nPanels > 0) {
    // Reset fixed panel
    if (fixPanel >= 0 && fixPanel < nPanels) {
      panels[fixPanel].dY = 0;
      panels[fixPanel].dZ = 0;
      panels[fixPanel].dTheta = 0;
    }

    // Update others
    if (tolShifts > EPS && nPanels > 1) {
      int xIdx = nBase;
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel)
          continue;
        panels[i].dY = x[xIdx++];
        panels[i].dZ = x[xIdx++];
        if (tolRotation > EPS)
          panels[i].dTheta = x[xIdx++];
        if (PerPanelLsd)
          panels[i].dLsd = x[xIdx++];
        if (PerPanelDistortion)
          panels[i].dP2 = x[xIdx++];
      }
    } else {
      // Reset others if optimization didn't run for them
      for (int i = 0; i < nPanels; i++) {
        if (i == fixPanel)
          continue;
        panels[i].dY = 0;
        panels[i].dZ = 0;
        panels[i].dTheta = 0;
      }
    }
  }

  // 3. Calculate deviations and outlier rejection (using updated params)
  {
    double txr, tyr, tzr;
    txr = deg2rad * tx; // tx is argument
    tyr = deg2rad * (*ty);
    tzr = deg2rad * (*tz);
    double Rx[3][3] = {
        {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
    double Ry[3][3] = {
        {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
    double Rz[3][3] = {
        {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
    double TRint[3][3], TRs[3][3];
    MatrixMultF33(Ry, Rz, TRint);
    MatrixMultF33(Rx, TRint, TRs);

    double *tempDiffs = malloc(nIndices * sizeof(double));
    for (int k = 0; k < nIndices; k++)
      tempDiffs[k] = -1.0;

    double totalSum = 0;
    int validCount = 0;

    int i;
    double n0 = 2, n1 = 4, n2 = 2;
    double Yc, Zc, Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, Diff, EtaT;
    double LsdV = *LsdFit;
    double ybcV = *ybcFit;
    double zbcV = *zbcFit;
    double p0V = *p0;
    double p1V = *p1;
    double p2V = *p2;
    double p3V = *p3;

    for (i = 0; i < nIndices; i++) {
      double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
      int pIdx = -1;
      if (nPanels > 0) {
        pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
        if (pIdx == -1)
          continue; // Skip invalid points
      }

      if (pIdx >= 0) {
        dY = panels[pIdx].dY;
        dZ = panels[pIdx].dZ;
        dTheta = panels[pIdx].dTheta;
        dLsd = panels[pIdx].dLsd;
        dP2 = panels[pIdx].dP2;
      }
      double rawY = YMean[i], rawZ = ZMean[i];
      if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
        double cY = panels[pIdx].centerY;
        double cZ = panels[pIdx].centerZ;
        double cosT = cos(deg2rad * dTheta);
        double sinT = sin(deg2rad * dTheta);
        double dy = rawY - cY, dz = rawZ - cZ;
        rawY = cY + dy * cosT - dz * sinT;
        rawZ = cZ + dy * sinT + dz * cosT;
      }
      Yc = -(rawY + dY - ybcV) * px;
      Zc = (rawZ + dZ - zbcV) * px;
      double ABC[3] = {0, Yc, Zc};
      double ABCPr[3];
      MatrixMultF(TRs, ABC, ABCPr);
      double panelLsd = LsdV + dLsd;
      double XYZ[3] = {panelLsd + ABCPr[0], ABCPr[1], ABCPr[2]};
      Rad = (panelLsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
      Eta = dg_calc_eta_angle(XYZ[1], XYZ[2]);
      RNorm = Rad / MaxRad;
      EtaT = 90 - Eta;
      double panelP2 = p2V + dP2;
      DistortFunc =
          (p0V * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
          (p1V * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3V)))) +
          (panelP2 * (pow(RNorm, n2)));
      if (nBase > 9 && p4Out)
        DistortFunc += (*p4Out) * pow(RNorm, 6.0);
      DistortFunc += 1;
      Rcorr = Rad * DistortFunc;
      RIdeal = panelLsd * tan(deg2rad * IdealTtheta[i]);
      Diff = fabs(1 - (Rcorr / RIdeal));
      tempDiffs[i] = Diff;
      totalSum += Diff;
      validCount++;
    }

    double currentMean = (validCount > 0) ? (totalSum / validCount) : 0;

    if (outlierFactor > 0) {
      double threshold = outlierFactor * currentMean;
      double cleanSum = 0;
      int cleanCount = 0;

      for (i = 0; i < nIndices; i++) {
        if (tempDiffs[i] < 0)
          continue;
        if (tempDiffs[i] <= threshold) {
          cleanSum += tempDiffs[i];
          cleanCount++;
        }
      }

      if (cleanCount > 0) {
        *MeanDiff = cleanSum / cleanCount;
        if (verbose)
          printf(
              "Outlier rejection (Factor %.2f): Excluded %d / %d points. Mean "
              "Strain: %.8f -> %.8f\n",
              outlierFactor, validCount - cleanCount, validCount, currentMean,
              *MeanDiff);
      } else {
        *MeanDiff = currentMean;
      }
    } else {
      // No outlier rejection, but update *MeanDiff to use the correct
      // denominator (validCount)
      *MeanDiff = currentMean;
    }

    free(tempDiffs);
  }
}

static inline void CorrectTiltSpatialDistortion(
    int nIndices, double MaxRad, double *YMean, double *ZMean,
    double *IdealTtheta, double px, double Lsd, double ybc, double zbc,
    double tx, double ty, double tz, double p0, double p1, double p2, double p3,
    double *Etas, double *Diffs, double *RadOuts, double *StdDiff,
    double outlierFactor, int *IsOutlier, double p4, int OutlierIterations,
    int verbose, double *MeanDiffOut) {
  double txr, tyr, tzr;
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  int i, j, k;
  double n0 = 2, n1 = 4, n2 = 2, Yc, Zc;
  double Rad, Eta, RNorm, DistortFunc, Rcorr, RIdeal, EtaT, Diff, MeanDiff = 0;
  int nValidPoints = 0;
  for (i = 0; i < nIndices; i++) {
    double dY = 0, dZ = 0, dTheta = 0, dLsd = 0, dP2 = 0;
    int pIdx = -1;
    if (nPanels > 0) {
      pIdx = GetPanelIndex(YMean[i], ZMean[i], nPanels, panels);
      if (pIdx == -1) {
        Diffs[i] = -1.0; // Mark as invalid
        continue;
      }
    }
    nValidPoints++;

    if (pIdx >= 0) {
      dY = panels[pIdx].dY;
      dZ = panels[pIdx].dZ;
      dTheta = panels[pIdx].dTheta;
      dLsd = panels[pIdx].dLsd;
      dP2 = panels[pIdx].dP2;
    }
    double rawY = YMean[i], rawZ = ZMean[i];
    if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
      double cY = panels[pIdx].centerY;
      double cZ = panels[pIdx].centerZ;
      double cosT = cos(deg2rad * dTheta);
      double sinT = sin(deg2rad * dTheta);
      double dy = rawY - cY, dz = rawZ - cZ;
      rawY = cY + dy * cosT - dz * sinT;
      rawZ = cZ + dy * sinT + dz * cosT;
    }
    Yc = -(rawY + dY - ybc) * px;
    Zc = (rawZ + dZ - zbc) * px;
    double ABC[3] = {0, Yc, Zc};
    double ABCPr[3];
    MatrixMultF(TRs, ABC, ABCPr);
    double panelLsd = Lsd + dLsd;
    double XYZ[3] = {panelLsd + ABCPr[0], ABCPr[1], ABCPr[2]};
    Rad = (panelLsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
    Eta = dg_calc_eta_angle(XYZ[1], XYZ[2]);
    RNorm = Rad / MaxRad;
    EtaT = 90 - Eta;
    double panelP2 = p2 + dP2;
    DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                  (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                  (panelP2 * (pow(RNorm, n2)));
    DistortFunc += p4 * pow(RNorm, 6.0);
    DistortFunc += 1;
    Rcorr = Rad * DistortFunc;
    RIdeal = panelLsd * tan(deg2rad * IdealTtheta[i]);
    Diff = fabs(1 - (Rcorr / RIdeal));
    Etas[i] = Eta;
    Diffs[i] = Diff;
    MeanDiff += Diff;
    RadOuts[i] = Rcorr;
  }
  if (nValidPoints > 0) {
    MeanDiff /= nValidPoints;
  } else {
    MeanDiff = 0;
  }

  // Filter outliers with iterative sigma-clipping
  double *validDiffs = malloc(nIndices * sizeof(double));
  int validCount = 0;

  if (outlierFactor > 0) {
    // Initialize: all valid points are inliers
    for (i = 0; i < nIndices; i++) {
      if (IsOutlier)
        IsOutlier[i] = (Diffs[i] < 0) ? 1 : 0;
    }

    int nIter = (OutlierIterations > 0) ? OutlierIterations : 1;
    for (int iter = 0; iter < nIter; iter++) {
      double threshold = outlierFactor * MeanDiff;
      double newSum = 0;
      validCount = 0;

      for (i = 0; i < nIndices; i++) {
        if (Diffs[i] < 0)
          continue;
        if (Diffs[i] <= threshold) {
          if (IsOutlier)
            IsOutlier[i] = 0;
          validDiffs[validCount] = Diffs[i];
          newSum += Diffs[i];
          validCount++;
        } else {
          if (IsOutlier)
            IsOutlier[i] = 1;
        }
      }
      if (validCount > 0) {
        double prevMean = MeanDiff;
        MeanDiff = newSum / validCount;
        if (iter == nIter - 1 || fabs(MeanDiff - prevMean) < 1e-10) {
          if (verbose)
            printf("Outlier Rejection (Factor %.2f, iter %d/%d): Excluded %d / "
                   "%d. Mean Strain: %.8f -> %.8f\n",
                   outlierFactor, iter + 1, nIter, nValidPoints - validCount,
                   nValidPoints, prevMean, MeanDiff);
          if (fabs(MeanDiff - prevMean) < 1e-10)
            break;
        }
      }
    }
  } else {
    for (i = 0; i < nIndices; i++) {
      if (Diffs[i] >= 0) {
        if (IsOutlier)
          IsOutlier[i] = 0;
        validDiffs[validCount] = Diffs[i];
        validCount++;
      } else {
        if (IsOutlier)
          IsOutlier[i] = 1;
      }
    }
  }

  double StdDiff2 = 0;
  for (i = 0; i < validCount; i++) {
    StdDiff2 += (validDiffs[i] - MeanDiff) * (validDiffs[i] - MeanDiff);
  }
  *StdDiff = sqrt(StdDiff2 / validCount);
  if (MeanDiffOut)
    *MeanDiffOut = MeanDiff;
  free(validDiffs);
}

static inline void DoImageTransformations(int NrTransOpt, int TransOpt[10],
                                          pixelvalue *Image, int NrPixels) {
  int i, j, k, l, m;
  pixelvalue **ImageTemp1, **ImageTemp2;
  ImageTemp1 = allocMatrixPX(NrPixels, NrPixels);
  ImageTemp2 = allocMatrixPX(NrPixels, NrPixels);
  if (NrTransOpt == 0) {
    return;
  }
  for (k = 0; k < NrPixels; k++) {
    for (l = 0; l < NrPixels; l++) {
      ImageTemp1[k][l] = Image[(NrPixels * k) + l];
    }
  }
  for (k = 0; k < NrTransOpt; k++) {
    if (TransOpt[k] == 1) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[l][NrPixels - m - 1]; // Inverting Y.
    } else if (TransOpt[k] == 2) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[NrPixels - l - 1][m]; // Inverting Z.
    } else if (TransOpt[k] == 3) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[m][l];
    } else if (TransOpt[k] == 0) {
      for (l = 0; l < NrPixels; l++)
        for (m = 0; m < NrPixels; m++)
          ImageTemp2[l][m] = ImageTemp1[l][m];
    }
    for (l = 0; l < NrPixels; l++)
      for (m = 0; m < NrPixels; m++)
        ImageTemp1[l][m] = ImageTemp2[l][m];
  }
  for (k = 0; k < NrPixels; k++) {
    for (l = 0; l < NrPixels; l++) {
      Image[(NrPixels * k) + l] = ImageTemp2[k][l];
    }
  }
  FreeMemMatrixPx(ImageTemp1, NrPixels);
  FreeMemMatrixPx(ImageTemp2, NrPixels);
}

static inline void MakeSquare(int NrPixels, int NrPixelsY, int NrPixelsZ,
                              pixelvalue *InImage, pixelvalue *OutImage) {
  int i, j, k;
  if (NrPixelsY == NrPixelsZ) {
    memcpy(OutImage, InImage, NrPixels * NrPixels * sizeof(*InImage));
  } else {
    if (NrPixelsY > NrPixelsZ) { // Filling along the slow direction // easy
      memcpy(OutImage, InImage, NrPixelsY * NrPixelsZ * sizeof(*InImage));
    } else {
      for (i = 0; i < NrPixelsZ; i++) {
        memcpy(OutImage + i * NrPixelsZ, InImage + i * NrPixelsY,
               NrPixelsY * sizeof(*InImage));
      }
    }
  }
}

// The fileReader function is being removed/replaced as part of the update.
// The original fileReader function ended with:
// } else {
//   return 127;
// }
// } // End of fileReader function
// This section is being removed.

int main(int argc, char *argv[]) {
  setvbuf(stdout, NULL, _IOLBF, 0); // line-buffer stdout for piped output
  if (argc != 3) {
    printf("Usage: CalibrantOMP ps.txt nCPUs\n");
    return 1;
  }
  double start, end, start0, end0;
  start0 = omp_get_wtime();
  double diftotal;
  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  numProcs = atoi(argv[2]);
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  char fn[1024], folder[1024], Ext[1024], Dark[1024];
  int StartNr, EndNr, LowNr;
  int SpaceGroup, FitWeightMean = 0;
  double LatticeConstant[6], Wavelength, MaxRingRad, Lsd, MaxTtheta, TthetaTol,
      ybc, zbc, EtaBinSize, px, Width;
  double tx = 0, tolTilts, tolLsd, tolBC, tolP, tolP0 = 0, tolP1 = 0, tolP2 = 0,
         tolP3 = 45, tyin = 0, tzin = 0, p0in = 0, p1in = 0, p2in = 0, p3in = 0,
         padY = 0, padZ = 0;
  double tolShifts = 1.0;
  double tolRotation = 0.0;
  double outlierFactor = 0.0;
  int MinIndicesForFit = 1;
  int FixPanelID = 0;
  int nIterations = 1;
  double DoubletSeparation = 0;
  int NormalizeRingWeights = 0;
  int OutlierIterations = 1;
  int WeightByRadius = 0;
  int WeightByFitSNR = 0;
  int L2Objective = 0;
  int PerPanelLsd = 0;
  int PerPanelDistortion = 0;
  double tolP4 = 0, p4in = 0;
  int tolP4Set = 0;
  double tolLsdPanel = 100;
  double tolP2Panel = 0.0001;
  int Padding = 6, NrPixelsY, NrPixelsZ, NrPixels;
  int NrTransOpt = 0, RBinWidth = 4;
  long long int GapIntensity = 0, BadPxIntensity = 0;
  int TransOpt[10], nRingsExclude = 0, RingsExclude[50];
  int makeMap = 0;
  int HeadSize = 8192;
  int dType = 1;
  char GapFN[4096], BadPxFN[4096], MaskFN[4096];
  char darkDatasetName[4096], dataDatasetName[4096];
  // Parameter defaults
  int NPanelsY = 0;
  int NPanelsZ = 0;
  int PanelSizeY = 0;
  int PanelSizeZ = 0;
  int *PanelGapsY = NULL;
  int *PanelGapsZ = NULL;
  char PanelShiftsFile[1024];
  PanelShiftsFile[0] = '\0';
  MaskFN[0] = '\0';
  sprintf(darkDatasetName, "exchange/dark");
  sprintf(dataDatasetName, "exchange/data");
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "FileStem ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn);
      continue;
    }
    str = "darkDataset ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, darkDatasetName);
      continue;
    }
    str = "darkLoc ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, darkDatasetName);
      continue;
    }
    str = "dataDataset ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, dataDatasetName);
      continue;
    }
    str = "dataLoc ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, dataDatasetName);
      continue;
    }
    str = "Folder ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, folder);
      continue;
    }
    str = "MaskFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, MaskFN);
      makeMap = 3;
      continue;
    }
    str = "GapFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, GapFN);
      makeMap = 2;
      continue;
    }
    str = "BadPxFile ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, BadPxFN);
      makeMap = 2;
      continue;
    }
    str = "DataType ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &dType);
      continue;
    }
    str = "RBinDivisions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RBinWidth);
      continue;
    }
    str = "SkipFrame ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &skipFrame);
      continue;
    }
    str = "GapIntensity ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lld", dummy, &GapIntensity);
      makeMap = 1;
      continue;
    }
    str = "BadPxIntensity ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lld", dummy, &BadPxIntensity);
      makeMap = 1;
      continue;
    }
    str = "Ext ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, Ext);
      continue;
    }
    str = "Dark ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, Dark);
      continue;
    }
    str = "Padding ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &Padding);
      continue;
    }
    str = "StartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "EndNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &EndNr);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      NrPixelsZ = NrPixelsY;
      continue;
    }
    str = "NrPixelsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      continue;
    }
    str = "NrPixelsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      continue;
    }
    str = "ImTransOpt ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &TransOpt[NrTransOpt]);
      NrTransOpt++;
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SpaceGroup);
      continue;
    }
    str = "NPanelsY ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &NPanelsY);
      continue;
    }
    str = "NPanelsZ ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &NPanelsZ);
      continue;
    }
    str = "PanelSizeY ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PanelSizeY);
      continue;
    }
    str = "PanelSizeZ ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PanelSizeZ);
      continue;
    }
    str = "PanelShiftsFile ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %s", dummy, PanelShiftsFile);
      continue;
    }

    str = "PanelGapsY ";
    if (!strncmp(aline, str, strlen(str))) {
      char *ptr = aline + strlen(str);
      if (NPanelsY > 1) {
        PanelGapsY = (int *)malloc((NPanelsY - 1) * sizeof(int));
        for (int k = 0; k < NPanelsY - 1; k++) {
          PanelGapsY[k] = strtol(ptr, &ptr, 10);
        }
      }
      continue;
    }
    str = "PanelGapsZ ";
    if (!strncmp(aline, str, strlen(str))) {
      char *ptr = aline + strlen(str);
      if (NPanelsZ > 1) {
        PanelGapsZ = (int *)malloc((NPanelsZ - 1) * sizeof(int));
        for (int k = 0; k < NPanelsZ - 1; k++) {
          PanelGapsZ[k] = strtol(ptr, &ptr, 10);
        }
      }
      continue;
    }

    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeConstant[0],
             &LatticeConstant[1], &LatticeConstant[2], &LatticeConstant[3],
             &LatticeConstant[4], &LatticeConstant[5]);
      continue;
    }
    str = "LatticeConstant ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeConstant[0],
             &LatticeConstant[1], &LatticeConstant[2], &LatticeConstant[3],
             &LatticeConstant[4], &LatticeConstant[5]);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "RhoD ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "ty ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tyin);
      continue;
    }
    str = "tz ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tzin);
      continue;
    }
    str = "p0 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p0in);
      continue;
    }
    str = "p1 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p1in);
      continue;
    }
    str = "p2 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p2in);
      continue;
    }
    str = "p3 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &p3in);
      continue;
    }
    str = "Width ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Width);
      continue;
    }
    str = "EtaBinSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &EtaBinSize);
      continue;
    }
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &ybc, &zbc);
      continue;
    }
    str = "tolTilts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolTilts);
      continue;
    }
    str = "tolBC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolBC);
      continue;
    }
    str = "tolLsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolLsd);
      continue;
    }
    str = "tolP ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP);
      continue;
    }
    str = "tolP0 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP0);
      continue;
    }
    str = "tolP1 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP1);
      continue;
    }
    str = "tolP2 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP2);
      continue;
    }
    str = "tolP3 ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolP3);
      continue;
    }
    str = "tolShifts ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolShifts);
      continue;
    }
    str = "tolRotation ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tolRotation);
      continue;
    }
    str = "MultFactor ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &outlierFactor);
      continue;
    }
    str = "MinIndicesForFit ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MinIndicesForFit);
      continue;
    }
    str = "FixPanelID ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &FixPanelID);
      continue;
    }
    str = "tx ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tx);
      continue;
    }
    str = "FitOrWeightedMean ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &FitWeightMean);
      continue;
    }
    str = "RingsToExclude ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsExclude[nRingsExclude]);
      nRingsExclude++;
      continue;
    }
    str = "HeadSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &HeadSize);
      continue;
    }
    str = "nIterations ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &nIterations);
      continue;
    }
    str = "DoubletSeparation ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %lf", dummy, &DoubletSeparation);
      continue;
    }
    str = "NormalizeRingWeights ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &NormalizeRingWeights);
      continue;
    }
    str = "OutlierIterations ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &OutlierIterations);
      continue;
    }
    str = "WeightByRadius ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &WeightByRadius);
      continue;
    }
    str = "WeightByFitSNR ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &WeightByFitSNR);
      continue;
    }
    str = "L2Objective ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &L2Objective);
      continue;
    }
    str = "DistortionOrder ";
    if (!strncmp(aline, str, strlen(str))) {
      continue; // DistortionOrder ignored; p4 is always enabled
    }
    str = "PerPanelLsd ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PerPanelLsd);
      continue;
    }
    str = "PerPanelDistortion ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %d", dummy, &PerPanelDistortion);
      continue;
    }
    str = "tolP4 ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %lf", dummy, &tolP4);
      tolP4Set = 1;
      continue;
    }
    str = "tolLsdPanel ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %lf", dummy, &tolLsdPanel);
      continue;
    }
    str = "tolP2Panel ";
    if (!strncmp(aline, str, strlen(str))) {
      sscanf(aline, "%s %lf", dummy, &tolP2Panel);
      continue;
    }
  }

  // Generate Panels
  if (NPanelsY > 0 && NPanelsZ > 0) {
    if (GeneratePanels(NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ, PanelGapsY,
                       PanelGapsZ, &panels, &nPanels) != 0) {
      fprintf(stderr, "Fast generation failed.\n");
      return 1;
    }
    printf("Generated %d panels.\n", nPanels);
  }
  if (tolP0 == 0)
    tolP0 = tolP;
  if (tolP1 == 0)
    tolP1 = tolP;
  if (tolP2 == 0)
    tolP2 = tolP;
  if (!tolP4Set && tolP4 == 0)
    tolP4 = tolP;
  if (NrPixelsY > NrPixelsZ) {
    NrPixels = NrPixelsY;
    NrPixelsGlobal = NrPixelsY;
  } else {
    NrPixels = NrPixelsZ;
    NrPixelsGlobal = NrPixelsZ;
  }
  int i, j, k;

  // Print parameter summary
  printf("\n");
  printf("╔══════════════════════════════════════════════════════════════╗\n");
  printf("║           CalibrantPanelShiftsOMP — Parameter Summary       ║\n");
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  FILE I/O                                                   ║\n");
  printf("║    FileStem:       %-40s ║\n", fn);
  printf("║    Folder:         %-40s ║\n", folder);
  printf("║    Ext:            %-40s ║\n", Ext);
  printf("║    Dark:           %-40s ║\n", Dark);
  printf("║    StartNr:        %-40d ║\n", StartNr);
  printf("║    EndNr:          %-40d ║\n", EndNr);
  printf("║    Padding:        %-40d ║\n", Padding);
  printf("║    HeadSize:       %-40d ║\n", HeadSize);
  printf("║    DataType:       %-40d ║\n", dType);
  printf("║    SkipFrame:      %-40d ║\n", skipFrame);
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  DETECTOR GEOMETRY                                          ║\n");
  printf("║    NrPixelsY:      %-40d ║\n", NrPixelsY);
  printf("║    NrPixelsZ:      %-40d ║\n", NrPixelsZ);
  printf("║    NrPixels(max):  %-40d ║\n", NrPixels);
  printf("║    PixelSize:      %-40.6f ║\n", px);
  printf("║    Lsd:            %-40.2f ║\n", Lsd);
  printf("║    BC:             %-20.4f %-19.4f ║\n", ybc, zbc);
  printf("║    tx (fixed):     %-40.6f ║\n", tx);
  printf("║    ty (initial):   %-40.6f ║\n", tyin);
  printf("║    tz (initial):   %-40.6f ║\n", tzin);
  printf("║    p0 (initial):   %-40.8f ║\n", p0in);
  printf("║    p1 (initial):   %-40.8f ║\n", p1in);
  printf("║    p2 (initial):   %-40.8f ║\n", p2in);
  printf("║    p3 (initial):   %-40.8f ║\n", p3in);
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  CRYSTALLOGRAPHY                                            ║\n");
  printf("║    SpaceGroup:     %-40d ║\n", SpaceGroup);
  printf("║    LatticeConst:   %-8.4f %-8.4f %-8.4f %-5.2f %-5.2f %-5.2f  ║\n",
         LatticeConstant[0], LatticeConstant[1], LatticeConstant[2],
         LatticeConstant[3], LatticeConstant[4], LatticeConstant[5]);
  printf("║    Wavelength:     %-40.8f ║\n", Wavelength);
  printf("║    RhoD:           %-40.2f ║\n", MaxRingRad);
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  MASKING                                                    ║\n");
  printf("║    BadPxIntensity: %-40lld ║\n", BadPxIntensity);
  printf("║    GapIntensity:   %-40lld ║\n", GapIntensity);
  if (MaskFN[0] != '\0')
    printf("║    MaskFile:       %-40s ║\n", MaskFN);
  printf("║    MakeMap mode:   %-40d ║\n", makeMap);
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  OPTIMIZATION TOLERANCES                                    ║\n");
  printf("║    tolTilts:       %-40.6f ║\n", tolTilts);
  printf("║    tolBC:          %-40.6f ║\n", tolBC);
  printf("║    tolLsd:         %-40.2f ║\n", tolLsd);
  printf("║    tolP0:          %-40.8f ║\n", tolP0);
  printf("║    tolP1:          %-40.8f ║\n", tolP1);
  printf("║    tolP2:          %-40.8f ║\n", tolP2);
  printf("║    tolP3:          %-40.8f ║\n", tolP3);
  printf("║    tolShifts:      %-40.6f ║\n", tolShifts);
  printf("║    tolRotation:    %-40.6f ║\n", tolRotation);
  printf("╠══════════════════════════════════════════════════════════════╣\n");
  printf("║  CALIBRATION CONTROL                                        ║\n");
  printf("║    Width:          %-40.2f ║\n", Width);
  printf("║    EtaBinSize:     %-40.4f ║\n", EtaBinSize);
  printf("║    RBinDivisions:  %-40d ║\n", RBinWidth);
  printf("║    MultFactor:     %-40.4f ║\n", outlierFactor);
  printf("║    MinIndices:     %-40d ║\n", MinIndicesForFit);
  printf("║    FitOrWMean:     %-40d ║\n", FitWeightMean);
  printf("║    nIterations:    %-40d ║\n", nIterations);
  if (DoubletSeparation > 0)
    printf("║    DoubletSep(px): %-40.1f ║\n", DoubletSeparation);
  if (NormalizeRingWeights)
    printf("║    RingWeightNorm: %-40s ║\n", "ON");
  if (WeightByRadius)
    printf("║    WeightByRadius: %-40s ║\n", "ON");
  if (WeightByFitSNR)
    printf("║    WeightByFitSNR: %-40s ║\n", "ON");
  if (L2Objective)
    printf("║    L2Objective:   %-40s ║\n", "ON (squared strain)");
  if (OutlierIterations > 1)
    printf("║    OutlierIters:   %-40d ║\n", OutlierIterations);
  if (PerPanelLsd)
    printf("║    PerPanelLsd:    %-40s (tol=%.1f) ║\n", "ON", tolLsdPanel);
  if (PerPanelDistortion)
    printf("║    PerPanelP2:     %-40s (tol=%.6f) ║\n", "ON", tolP2Panel);
  if (nRingsExclude > 0) {
    printf("║    RingsExclude:   ");
    int printed = 0;
    for (i = 0; i < nRingsExclude; i++) {
      printed += printf("%d ", RingsExclude[i]);
    }
    for (int pad = printed; pad < 41; pad++)
      printf(" ");
    printf("║\n");
  }
  if (nPanels > 0) {
    printf(
        "╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  MULTI-PANEL                                                ║\n");
    printf("║    NPanelsY:       %-40d ║\n", NPanelsY);
    printf("║    NPanelsZ:       %-40d ║\n", NPanelsZ);
    printf("║    PanelSizeY:     %-40d ║\n", PanelSizeY);
    printf("║    PanelSizeZ:     %-40d ║\n", PanelSizeZ);
    printf("║    FixPanelID:     %-40d ║\n", FixPanelID);
    printf("║    Panels:         %-40d ║\n", nPanels);
    if (PanelShiftsFile[0] != '\0')
      printf("║    ShiftsFile:     %-40s ║\n", PanelShiftsFile);
  }
  printf(
      "╚══════════════════════════════════════════════════════════════╝\n\n");
  for (i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] < 0 || TransOpt[i] > 3) {
      printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
      return 0;
    }
    printf("TransformationOptions: %d ", TransOpt[i]);
    if (TransOpt[i] == 0)
      printf("No change.\n");
    else if (TransOpt[i] == 1)
      printf("Flip Left Right.\n");
    else if (TransOpt[i] == 2)
      printf("Flip Top Bottom.\n");
    else
      printf("Transpose.\n");
  }
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  double Thetas[100];
  int RingIDs[100];
  for (i = 0; i < 100; i++)
    Thetas[i] = 0;
  int n_hkls = 0;

  run_midas_binary("GetHKLList", ParamFN);
  // char cmmd[4096];
  // sprintf(cmmd,"~/opt/MIDAS/FF_HEDM/bin/GetHKLList %s",ParamFN);
  // system(cmmd);
  // Read hkls.csv
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  int tRnr, Exclude, LastRingDone = 0;
  double theta;
  printf("Ring Info:\n");
  printf("  %5s  %10s  %10s\n", "RingNr", "Theta(deg)", "Radius(px)");
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%s %s %s %s %d %s %s %s %lf %s %s", dummy, dummy, dummy,
           dummy, &tRnr, dummy, dummy, dummy, &theta, dummy, dummy);
    if (theta * 2 > MaxTtheta)
      break;
    Exclude = 0;
    for (i = 0; i < nRingsExclude; i++) {
      if (tRnr == RingsExclude[i]) {
        Exclude = 1;
      }
    }
    if (Exclude == 0 && tRnr > LastRingDone) {
      Thetas[n_hkls] = theta;
      RingIDs[n_hkls] = tRnr;
      LastRingDone = tRnr;
      double ringRadPx = Lsd * tan(deg2rad * 2 * theta) / px;
      printf("  %5d  %10.4f  %10.1f\n", tRnr, theta, ringRadPx);
      n_hkls++;
    }
  }
  printf("\n");

  printf("Number of planes being considered: %d.\n", n_hkls);
  printf("The following rings will be excluded:");
  for (i = 0; i < nRingsExclude; i++) {
    printf(" %d", RingsExclude[i]);
  }

  TthetaTol = Ttheta4mR((MaxRingRad + Width), Lsd) -
              Ttheta4mR((MaxRingRad - Width), Lsd);
  printf("\n2Theta Tolerance: %f \n", TthetaTol);
  pixelvalue *DarkFile;
  pixelvalue *DarkFile2;
  double *AverageDark;
  size_t pxSize;
  if (dType == 1) { // Uint16
    pxSize = sizeof(uint16_t);
  } else if (dType == 2) { // Double
    pxSize = sizeof(double);
  } else if (dType == 3) { // Float
    pxSize = sizeof(float);
  } else if (dType == 4) { // Uint32
    pxSize = sizeof(uint32_t);
  } else if (dType == 5) { // Int32
    pxSize = sizeof(int32_t);
  } else if (dType == 6) { // Tiff Uint32
    pxSize = sizeof(uint32_t);
    HeadSize = 0;
  } else if (dType == 7) { // Tiff Uint8
    pxSize = sizeof(uint8_t);
    HeadSize = 0;
  } else if (dType == 8) { // HDF Unit16
    pxSize = sizeof(uint16_t);
    HeadSize = 0;
  } else if (dType == 9) { // Tiff Unit16
    pxSize = sizeof(uint16_t);
    HeadSize = 0;
  }
  size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
  size_t sz;
  char FileName[1024];
  size_t Skip;
  FILE *fp, *fd;
  int nFrames, TotFrames = 0;
  double *Average;
  pixelvalue *Image;
  pixelvalue *Image2;
  DarkFile = malloc(NrPixelsY * NrPixelsZ * sizeof(*DarkFile));    // Raw.
  Image = malloc(NrPixelsY * NrPixelsZ * sizeof(*Image));          // Raw.
  DarkFile2 = calloc(NrPixels * NrPixels, sizeof(*DarkFile2));     // Squared.
  Image2 = calloc(NrPixels * NrPixels, sizeof(*Image2));           // Squared.
  AverageDark = calloc(NrPixels * NrPixels, sizeof(*AverageDark)); // Squared.
  Average = calloc(NrPixels * NrPixels, sizeof(*Average));         // Squared.
  fd = fopen(Dark, "rb");

  int rc;
  int darkInitialized = 0;
  char *dname;
  if (fd == NULL && dType != 8) {
    printf("Dark file %s could not be read. Making an empty array for dark.\n",
           Dark);
    for (j = 0; j < (NrPixels * NrPixels); j++)
      AverageDark[j] = 0;
    darkInitialized = 1;
  } else {
    if (dType != 8) {
      dname = "/";
      fseek(fd, 0L, SEEK_END);
      sz = ftell(fd);
      sz -= HeadSize;
      rewind(fd);
      nFrames = sz / (SizeFile);
      Skip = HeadSize;
      printf("Reading dark file:      %s, nFrames: %d, skipping first %ld "
             "bytes.\n",
             Dark, nFrames, Skip);
      fseek(fd, Skip, SEEK_SET);
      for (i = 0; i < nFrames; i++) {
        if (dType == 6 || dType == 7 || dType == 9) {
          rc = ReadTiffFrame(Dark, dType, NrPixelsY * NrPixelsZ, DarkFile, i);
        } else {
          rc = ReadBinaryFrame(fd, dType, NrPixelsY * NrPixelsZ, DarkFile);
        }
        MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, DarkFile, DarkFile2);
        DoImageTransformations(NrTransOpt, TransOpt, DarkFile2, NrPixels);
        if (makeMap == 1) {
          size_t badPxCounter = 0;
          mapMaskSize = NrPixels;
          mapMaskSize *= NrPixels;
          mapMaskSize /= 32;
          mapMaskSize++;
          mapMask = calloc(mapMaskSize, sizeof(*mapMask));
          for (j = 0; j < NrPixels * NrPixels; j++) {
            if (DarkFile2[j] == (pixelvalue)GapIntensity ||
                DarkFile2[j] == (pixelvalue)BadPxIntensity) {
              badPxCounter++;
              SetBit(mapMask, j);
            }
          }
          makeMap = 0;
          printf("%lld\n", (long long int)badPxCounter);
        }
        for (j = 0; j < (NrPixels * NrPixels); j++)
          AverageDark[j] += DarkFile2[j];
      }
      printf("Dark file read.\n");
      for (j = 0; j < (NrPixels * NrPixels); j++)
        AverageDark[j] = AverageDark[j] / nFrames;
      fclose(fd);
      darkInitialized = 1;
    } else {
      // read dark from dark hdf5
      printf("Reading dark from dark HDF5.\n");
      printf("Dark file: %s\n", Dark);
      dname = darkDatasetName;
      printf("Dark dataset name: %s\n", dname);
      rc = SumHDF5Frames(Dark, dname, NrPixelsY * NrPixelsZ, DarkFile,
                         skipFrame);
      MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, DarkFile, DarkFile2);
      DoImageTransformations(NrTransOpt, TransOpt, DarkFile2, NrPixels);
      if (makeMap == 1) {
        size_t badPxCounter = 0;
        mapMaskSize = NrPixels;
        mapMaskSize *= NrPixels;
        mapMaskSize /= 32;
        mapMaskSize++;
        mapMask = calloc(mapMaskSize, sizeof(*mapMask));
        for (j = 0; j < NrPixels * NrPixels; j++) {
          if (DarkFile2[j] == (pixelvalue)GapIntensity ||
              DarkFile2[j] == (pixelvalue)BadPxIntensity) {
            badPxCounter++;
            SetBit(mapMask, j);
          }
        }
        makeMap = 0;
        printf("%lld\n", (long long int)badPxCounter);
      }
      printf("Dark file read from HDF5.\n");
      for (j = 0; j < (NrPixels * NrPixels); j++)
        AverageDark[j] = DarkFile2[j];
      darkInitialized = 1;
    }
  }
  printf("makeMap: %d\n", makeMap);
  if (makeMap == 2) {
    mapMaskSize = NrPixels;
    mapMaskSize *= NrPixels;
    mapMaskSize /= 32;
    mapMaskSize++;
    mapMask = calloc(mapMaskSize, sizeof(*mapMask));
    double *mapper;
    mapper = calloc(NrPixelsY * NrPixelsZ, sizeof(*mapper));
    double *mapperSquare;
    mapperSquare = calloc(NrPixels * NrPixels, sizeof(*mapperSquare));
    ReadTiffFrame(GapFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, mapper, mapperSquare);
    DoImageTransformations(NrTransOpt, TransOpt, mapperSquare, NrPixels);
    for (i = 0; i < NrPixels * NrPixels; i++) {
      if (mapperSquare[i] == 1) {
        SetBit(mapMask, i);
        mapperSquare[i] = 0;
      }
    }
    ReadTiffFrame(BadPxFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, mapper, mapperSquare);
    DoImageTransformations(NrTransOpt, TransOpt, mapperSquare, NrPixels);
    for (i = 0; i < NrPixels * NrPixels; i++) {
      if (mapperSquare[i] == 1) {
        SetBit(mapMask, i);
        mapperSquare[i] = 0;
      }
    }
  } else if (makeMap == 3) {
    mapMaskSize = NrPixels;
    mapMaskSize *= NrPixels;
    mapMaskSize /= 32;
    mapMaskSize++;
    mapMask = calloc(mapMaskSize, sizeof(*mapMask));
    double *mapper;
    mapper = calloc(NrPixelsY * NrPixelsZ, sizeof(*mapper));
    double *mapperSquare;
    mapperSquare = calloc(NrPixels * NrPixels, sizeof(*mapperSquare));
    ReadTiffFrame(MaskFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, mapper, mapperSquare);
    DoImageTransformations(NrTransOpt, TransOpt, mapperSquare, NrPixels);
    for (i = 0; i < NrPixels * NrPixels; i++) {
      if (mapperSquare[i] == 1) {
        SetBit(mapMask, i);
        mapperSquare[i] = 0;
      }
    }
    printf("Mask file read: %s\n", MaskFN);
  }
  int a;
  double means[12];
  for (a = 0; a < 11; a++)
    means[a] = 0;
  means[11] = 0;
  double medStrain = 0, q25Strain = 0, q75Strain = 0, minStrain = 0,
         maxStrain = 0;
  // Radius residual statistics (signed and absolute)
  double medRadRes = 0, q25RadRes = 0, q75RadRes = 0;
  double minRadRes = 0, maxRadRes = 0, meanRadRes = 0, stdRadRes = 0;
  double meanAbsRadRes = 0, maxAbsRadRes = 0;
  int nValidRadRes = 0;
  // 2theta residual statistics
  double med2thRes = 0, q252thRes = 0, q752thRes = 0;
  double min2thRes = 0, max2thRes = 0, mean2thRes = 0, std2thRes = 0;
  double meanAbs2thRes = 0, maxAbs2thRes = 0;
  int nValid2thRes = 0;
// Per-ring radius residual accumulators
#define MAX_RINGS_STAT 200
  double ringMeanRadRes[MAX_RINGS_STAT] = {0};
  double ringMeanAbsRadRes[MAX_RINGS_STAT] = {0};
  double ringMaxAbsRadRes[MAX_RINGS_STAT] = {0};
  double ringIdealR[MAX_RINGS_STAT] = {0};
  int ringCountRadRes[MAX_RINGS_STAT] = {0};
  // Per-ring 2theta residual accumulators
  double ringMean2thRes[MAX_RINGS_STAT] = {0};
  double ringMeanAbs2thRes[MAX_RINGS_STAT] = {0};
  double ringMax2thAbs[MAX_RINGS_STAT] = {0};
  double ringIdeal2th[MAX_RINGS_STAT] = {0};
  int ringCount2thRes[MAX_RINGS_STAT] = {0};
  // Lattice parameter residual statistics
  double medLatRes = 0, q25LatRes = 0, q75LatRes = 0;
  double minLatRes = 0, maxLatRes = 0, meanLatRes = 0, stdLatRes = 0;
  double meanAbsLatRes = 0, maxAbsLatRes = 0;
  int nValidLatRes = 0;
  double IdealA = LatticeConstant[0]; // ideal lattice parameter 'a'
  // Per-ring lattice parameter residual accumulators
  double ringMeanLatRes[MAX_RINGS_STAT] = {0};
  double ringMeanAbsLatRes[MAX_RINGS_STAT] = {0};
  double ringMaxAbsLat[MAX_RINGS_STAT] = {0};
  int ringCountLatRes[MAX_RINGS_STAT] = {0};
  int nValid = 0;
  for (a = StartNr; a <= EndNr; a++) {
    start = omp_get_wtime();
    sprintf(FileName, "%s/%s_%0*d%s", folder, fn, Padding, a, Ext);
    if (dType != 8) {
      fp = fopen(FileName, "rb");
      if (fp == NULL) {
        printf("File %s could not be read. Continuing to next one.\n",
               FileName);
        continue;
      }
      fseek(fp, 0L, SEEK_END);
      sz = ftell(fp);
      sz = sz - HeadSize;
      nFrames = sz / (SizeFile);
      Skip = HeadSize;
      printf("Reading calibrant file: %s, nFrames: %d %d %d, skipping first "
             "%ld bytes.\n",
             FileName, nFrames, (int)sz, (int)SizeFile, Skip);
      rewind(fp);
      fseek(fp, Skip, SEEK_SET);
      for (j = 0; j < nFrames; j++) {
        if (dType == 6 || dType == 7 || dType == 9) {
          rc = ReadTiffFrame(FileName, dType, NrPixelsY * NrPixelsZ, Image, j);
        } else {
          rc = ReadBinaryFrame(fp, dType, NrPixelsY * NrPixelsZ, Image);
        }
        MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, Image, Image2);
        DoImageTransformations(NrTransOpt, TransOpt, Image2, NrPixels);
        for (k = 0; k < (NrPixels * NrPixels); k++) {
          Average[k] +=
              (double)(Image2[k]) - AverageDark[k]; // In reality this is sum
        }
      }
      TotFrames += nFrames;
      fclose(fp);
    } else {
      printf("Reading HDF5.\n");
      printf("Data file: %s\n", FileName);
      // check if dark was not already initialized, the read dark.
      if (darkInitialized == 0) {
        printf("Reading dark from Data HDF5.\n");
        dname = darkDatasetName;
        printf("%s\n", dname);
        rc = SumHDF5Frames(FileName, dname, NrPixelsY * NrPixelsZ, DarkFile,
                           skipFrame);
        MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, DarkFile, DarkFile2);
        DoImageTransformations(NrTransOpt, TransOpt, DarkFile2, NrPixels);
        if (makeMap == 1) {
          size_t badPxCounter = 0;
          mapMaskSize = NrPixels;
          mapMaskSize *= NrPixels;
          mapMaskSize /= 32;
          mapMaskSize++;
          mapMask = calloc(mapMaskSize, sizeof(*mapMask));
          for (j = 0; j < NrPixels * NrPixels; j++) {
            if (DarkFile2[j] == (pixelvalue)GapIntensity ||
                DarkFile2[j] == (pixelvalue)BadPxIntensity) {
              badPxCounter++;
              SetBit(mapMask, j);
            }
          }
          makeMap = 0;
          printf("%lld\n", (long long int)badPxCounter);
        }
        printf("Dark file read.\n");
        for (j = 0; j < (NrPixels * NrPixels); j++)
          AverageDark[j] = DarkFile2[j];
        darkInitialized = 1;
      }
      // sprintf(dname,"%s",dataDatasetName);
      dname = dataDatasetName;
      printf("Data dataset name: %s, data file: %s\n", dname, FileName);
      rc = SumHDF5Frames(FileName, dname, NrPixelsY * NrPixelsZ, Image,
                         skipFrame);
      MakeSquare(NrPixels, NrPixelsY, NrPixelsZ, Image, Image2);
      DoImageTransformations(NrTransOpt, TransOpt, Image2, NrPixels);
      for (j = 0; j < (NrPixels * NrPixels); j++)
        Average[j] = Image2[j] - AverageDark[j];
      //~ uint16_t *outMatrix;
      //~ outMatrix = calloc(NrPixels*NrPixels,sizeof(uint16_t));
      //~ for (j=0;j<(NrPixels*NrPixels);j++)outMatrix[j] = (uint16_t)
      // Average[j]; ~ FILE *fnew; ~ fnew = fopen("Data_000001.ge3","wb"); ~
      // fwrite(outMatrix,sizeof(uint16_t)*NrPixels*NrPixels,1,fnew); ~
      // fclose(fnew); ~ return;
    }
    double IdealTthetas[n_hkls], TthetaMins[n_hkls], TthetaMaxs[n_hkls];
    for (i = 0; i < n_hkls; i++) {
      IdealTthetas[i] = 2 * Thetas[i];
      TthetaMins[i] = IdealTthetas[i] - TthetaTol;
      TthetaMaxs[i] = IdealTthetas[i] + TthetaTol;
    }
    int nEtaBins;
    nEtaBins = (int)ceil(359.99 / EtaBinSize);
    printf("Number of eta bins: %d.\n", nEtaBins);
    double EtaBinsLow[nEtaBins], EtaBinsHigh[nEtaBins];
    for (i = 0; i < nEtaBins; i++) {
      EtaBinsLow[i] = EtaBinSize * i - 179.995;
      EtaBinsHigh[i] = EtaBinSize * (i + 1) - 179.995;
    }

    // Variables that persist across iterations (declared here, populated
    // inside)
    double ty, tz, LsdFit, ybcFit, zbcFit, p0, p1, p2, p3, MeanDiff, StdDiff;
    double *Yc = NULL, *Zc = NULL, *EtaIns = NULL, *RadIns = NULL,
           *DiffIns = NULL;
    double *RMean = NULL, *EtaMean = NULL, *YMean = NULL, *ZMean = NULL;
    double *IdealTtheta = NULL;
    int *RingNumbers = NULL;
    int nIndices = nEtaBins * n_hkls;
    int nIndicesFinal = 0;

    // Best-iteration tracking
    double bestMeanDiff = 1e30;
    int bestIter = -1;
    double bestLsd, bestYbc, bestZbc, bestTy, bestTz;
    double bestP0, bestP1, bestP2, bestP3, bestP4 = 0;
    int stagnantCount = 0;
    double prevIterMeanDiff = -1;
    Panel *bestPanels = NULL;
    if (nPanels > 0)
      bestPanels = malloc(nPanels * sizeof(Panel));

    int perturbedFlag =
        0; // set by perturbation logic; triggers re-bin next iter
    double *FitSNR = NULL; // persists across iterations for snrWeights

    for (int iter = 0; iter < nIterations; iter++) {
      if (nIterations > 1) {
        char iterLabel[64];
        snprintf(iterLabel, sizeof(iterLabel), "Iteration %d / %d", iter + 1,
                 nIterations);
        int boxW = 44;
        printf("\n╔");
        for (int b = 0; b < boxW; b++)
          printf("═");
        printf("╗\n║  %-*s║\n╚", boxW - 2, iterLabel);
        for (int b = 0; b < boxW; b++)
          printf("═");
        printf("╝\n");
      }

      // --- Option B: skip initial bin+fit on iter>0 unless perturbed ---
      int didInitBin = 0;
      double *R = NULL, *Eta = NULL, *IdealR = NULL;
      double *IdealRmins = NULL, *IdealRmaxs = NULL;
      int *NrEachIndexBin = NULL;
      int **Indices = NULL;
      int nIndicesOrig = 0;

      if (iter == 0 || perturbedFlag) {
        didInitBin = 1;
        perturbedFlag = 0;

        // Recompute Lsd-dependent radii
        double IdealRs[n_hkls], Rmins[n_hkls], Rmaxs[n_hkls];
        for (i = 0; i < n_hkls; i++) {
          IdealRs[i] = R4mTtheta(IdealTthetas[i], Lsd);
          Rmins[i] = R4mTtheta(TthetaMins[i], Lsd);
          Rmaxs[i] = R4mTtheta(TthetaMaxs[i], Lsd);
        }

        // Detect doublet ring pairs
        int doubletFlag[n_hkls], doubletPartner[n_hkls];
        for (i = 0; i < n_hkls; i++) {
          doubletFlag[i] = 0;
          doubletPartner[i] = -1;
        }
        if (DoubletSeparation > 0) {
          double sepThresh = DoubletSeparation * px;
          for (i = 0; i < n_hkls - 1; i++) {
            if (doubletFlag[i] != 0)
              continue;
            if (fabs(IdealRs[i + 1] - IdealRs[i]) < sepThresh) {
              doubletFlag[i] = 1;
              doubletFlag[i + 1] = 2;
              doubletPartner[i] = i + 1;
              doubletPartner[i + 1] = i;
              if (iter == 0)
                printf("Doublet detected: ring %d (R=%.2f) <-> ring %d "
                       "(R=%.2f), separation %.1f px\n",
                       RingIDs[i], IdealRs[i], RingIDs[i + 1], IdealRs[i + 1],
                       fabs(IdealRs[i + 1] - IdealRs[i]) / px);
            }
          }
        }

        // Allocate per-iteration arrays
        double *R = malloc(NrPixels * NrPixels * sizeof(*R));
        double *Eta = malloc(NrPixels * NrPixels * sizeof(*Eta));
        nIndices = nEtaBins * n_hkls;
        int *NrEachIndexBin = malloc(nIndices * sizeof(*NrEachIndexBin));
        int **Indices = allocMatrixInt(nIndices, 20000);

        // Bin pixels into rings using current parameters
        Car2Pol(n_hkls, nEtaBins, NrPixels, NrPixels, ybc, zbc, px, R, Eta,
                Rmins, Rmaxs, EtaBinsLow, EtaBinsHigh, nIndices, NrEachIndexBin,
                Indices, tx, tyin, tzin, p0in, p1in, p2in, p3in, MaxRingRad,
                Lsd, p4in);

        double *IdealR = malloc(nIndices * sizeof(*IdealR));
        double *IdealRmins = malloc(nIndices * sizeof(*IdealRmins));
        double *IdealRmaxs = malloc(nIndices * sizeof(*IdealRmaxs));
        IdealTtheta = malloc(nIndices * sizeof(*IdealTtheta));
        RingNumbers = malloc(nIndices * sizeof(*RingNumbers));
        RMean = malloc(nIndices * sizeof(*RMean));
        EtaMean = malloc(nIndices * sizeof(*EtaMean));
        int NrPtsForFit;
        NrPtsForFit = (int)((floor)((Rmaxs[0] - Rmins[0]) / px)) * RBinWidth;
        for (i = 0; i < nIndices; i++) {
          IdealR[i] = IdealRs[(int)(floor((float)i / nEtaBins))];
          IdealRmins[i] = Rmins[(int)(floor((float)i / nEtaBins))];
          IdealRmaxs[i] = Rmaxs[(int)(floor((float)i / nEtaBins))];
          IdealTtheta[i] = rad2deg * atan(IdealR[i] / Lsd);
          RingNumbers[i] = RingIDs[(int)(floor((float)i / nEtaBins))];
        }

        // Find peak positions
        NrCallsProfiler = 0;
        if (FitSNR) {
          free(FitSNR);
          FitSNR = NULL;
        } // free previous iter's
        FitSNR = calloc(nIndices, sizeof(*FitSNR));
        if (FitWeightMean == 1) {
          CalcWeightedMean(nIndices, NrEachIndexBin, Indices, Average, R, Eta,
                           RMean, EtaMean);
        } else {
          CalcFittedMean(nIndices, NrEachIndexBin, Indices, Average, R, Eta,
                         RMean, EtaMean, FitSNR, NrPtsForFit, IdealRmins,
                         IdealRmaxs, nEtaBins, ybc, zbc, px, NrPixelsY,
                         NrPixelsZ, EtaBinsLow, EtaBinsHigh, doubletFlag,
                         doubletPartner);
        }

        // Compact: remove zero-RMean entries
        int countr = 0;
        double *RMean2 = malloc(nIndices * sizeof(*RMean2));
        double *EtaMean2 = malloc(nIndices * sizeof(*EtaMean2));
        double *IdealTtheta2 = malloc(nIndices * sizeof(*IdealTtheta2));
        int *RingNumbers2 = malloc(nIndices * sizeof(*RingNumbers2));
        double *FitSNR2 = malloc(nIndices * sizeof(*FitSNR2));
        for (i = 0; i < nIndices; i++) {
          if (RMean[i] != 0) {
            RMean2[countr] = RMean[i];
            EtaMean2[countr] = EtaMean[i];
            IdealTtheta2[countr] = IdealTtheta[i];
            RingNumbers2[countr] = RingNumbers[i];
            FitSNR2[countr] = FitSNR[i];
            countr++;
          }
        }
        if (iter == 0)
          printf("Out of %d slices, %d were in the detector\n", nIndices,
                 countr);
        int nIndicesOrig = nIndices; // save for freeing Indices
        nIndices = countr;
        free(RMean);
        free(EtaMean);
        free(IdealTtheta);
        free(RingNumbers);
        if (didInitBin)
          free(FitSNR);
        RMean = RMean2;
        EtaMean = EtaMean2;
        IdealTtheta = IdealTtheta2;
        RingNumbers = RingNumbers2;
        FitSNR = FitSNR2;

        end = omp_get_wtime();
        diftotal = end - start;
        if (iter == 0) {
          if (FitWeightMean != 1) {
            printf("Number of calls to profiler function: %lld\n",
                   NrCallsProfiler);
            printf("Time elapsed in fitting peak profiles:\t%f s.\n", diftotal);
          } else
            printf("Time elapsed in finding peak positions:\t%f s.\n",
                   diftotal);
        }

        // Convert to detector coords
        YMean = malloc(nIndices * sizeof(*YMean));
        ZMean = malloc(nIndices * sizeof(*ZMean));
        YZ4mREta(nIndices, RMean, EtaMean, YMean, ZMean);
        Yc = malloc(nIndices * sizeof(*Yc));
        Zc = malloc(nIndices * sizeof(*Zc));
        EtaIns = malloc(nIndices * sizeof(*EtaIns));
        RadIns = malloc(nIndices * sizeof(*RadIns));
        DiffIns = malloc(nIndices * sizeof(*DiffIns));
        for (i = 0; i < nIndices; i++) {
          Yc[i] = (ybc - (YMean[i] / px));
          Zc[i] = (zbc + (ZMean[i] / px));
        }
        CorrectTiltSpatialDistortion(nIndices, MaxRingRad, Yc, Zc, IdealTtheta,
                                     px, Lsd, ybc, zbc, tx, tyin, tzin, p0in,
                                     p1in, p2in, p3in, EtaIns, DiffIns, RadIns,
                                     &StdDiff, outlierFactor, NULL, p4in,
                                     OutlierIterations, iter == 0, NULL);
        NrCalls = 0;

      } // end if (iter == 0 || perturbedFlag) — initial bin+fit

      // Count and print indices per panel (only on iter 0)
      if (nPanels > 0 && iter == 0) {
        int *panelCounts = calloc(nPanels, sizeof(int));
        for (int ii = 0; ii < nIndices; ii++) {
          int pIdx = GetPanelIndex(Yc[ii], Zc[ii], nPanels, panels);
          if (pIdx >= 0) {
            panelCounts[pIdx]++;
          }
        }
        printf("\n******************* Indices per Panel (Visual Layout: Z^ Y>) "
               "*******************\n");
        printf("                        Anchored Panel ID: %d \n", FixPanelID);
        double charAspect = 0.5;
        double textWidthPerPanel = 14.0;
        double visualWidthPoints = NPanelsY * textWidthPerPanel * charAspect;
        double targetHeightPoints =
            visualWidthPoints * ((double)NrPixelsZ / (double)NrPixelsY);
        int linesPerRow = (int)(targetHeightPoints / NPanelsZ * 0.15 + 0.5);
        if (linesPerRow < 1)
          linesPerRow = 1;
        for (int z = NPanelsZ - 1; z >= 0; z--) {
          for (int l = 0; l < linesPerRow; l++) {
            if (l == linesPerRow / 2)
              printf("Z=%-2d | ", z);
            else
              printf("     | ");
            if (l == linesPerRow / 2) {
              for (int y = 0; y < NPanelsY; y++) {
                int pIdx = y * NPanelsZ + z;
                if (pIdx < nPanels) {
                  printf("| %3d (%5d) ", pIdx, panelCounts[pIdx]);
                } else {
                  printf("|             ");
                }
              }
              printf("|");
            }
            printf("\n");
          }
        }
        printf("       ");
        for (int y = 0; y < NPanelsY; y++) {
          printf("     Y=%-2d     ", y);
        }
        printf("\n");
        printf("*************************************************************"
               "****"
               "****"
               "***********\n\n");
        free(panelCounts);
      }

      // Compute per-ring weights (works with either fresh or carried-forward
      // RingNumbers)
      double *RingWeights = NULL;
      if (NormalizeRingWeights) {
        RingWeights = malloc(nIndices * sizeof(*RingWeights));
        int ringCounts[200] = {0};
        for (i = 0; i < nIndices; i++) {
          if (RingNumbers[i] >= 0 && RingNumbers[i] < 200)
            ringCounts[RingNumbers[i]]++;
        }
        for (i = 0; i < nIndices; i++) {
          int rn = RingNumbers[i];
          if (rn >= 0 && rn < 200 && ringCounts[rn] > 0)
            RingWeights[i] = 1.0 / ((double)ringCounts[rn]);
          else
            RingWeights[i] = 1.0;
        }
      }
      // snrWeights: compute from FitSNR (available from initial bin or previous
      // re-fit)
      double *snrWeights = NULL;
      if (WeightByFitSNR && FitWeightMean != 1 && FitSNR != NULL) {
        snrWeights = malloc(nIndices * sizeof(*snrWeights));
        double *snrSorted = malloc(nIndices * sizeof(*snrSorted));
        memcpy(snrSorted, FitSNR, nIndices * sizeof(*snrSorted));
        for (int a = 1; a < nIndices; a++) {
          double key = snrSorted[a];
          int b = a - 1;
          while (b >= 0 && snrSorted[b] > key) {
            snrSorted[b + 1] = snrSorted[b];
            b--;
          }
          snrSorted[b + 1] = key;
        }
        double medianSNR = snrSorted[nIndices / 2];
        if (medianSNR < 1e-12)
          medianSNR = 1.0;
        free(snrSorted);
        for (i = 0; i < nIndices; i++) {
          double w = FitSNR[i] / medianSNR;
          snrWeights[i] = (w < 1.0) ? w : 1.0;
        }
      }

      // Optimize
      double p4 = p4in;
      FitTiltBCLsd(nIndices, Yc, Zc, IdealTtheta, Lsd, MaxRingRad, ybc, zbc, tx,
                   tyin, tzin, p0in, p1in, p2in, p3in, &ty, &tz, &LsdFit,
                   &ybcFit, &zbcFit, &p0, &p1, &p2, &p3, &MeanDiff, tolTilts,
                   tolLsd, tolBC, tolP, tolP0, tolP1, tolP2, tolP3, tolShifts,
                   tolRotation, px, outlierFactor, MinIndicesForFit, FixPanelID,
                   RingWeights, p4in, tolP4, PerPanelLsd, tolLsdPanel,
                   PerPanelDistortion, tolP2Panel, WeightByRadius, snrWeights,
                   &p4, iter == 0, L2Objective);
      if (iter == 0) {
        printf("Number of function calls: %lld\n", NrCalls);
        printf("Lsd        %0.12f\n"
               "BC         %0.12f %0.12f\n"
               "ty         %0.12f\n"
               "tz         %0.12f\n"
               "p0         %0.12f\n"
               "p1         %0.12f\n"
               "p2         %0.12f\n"
               "p3         %0.12f\n",
               LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3);
        printf("p4         %0.12f\n", p4);
      }
      printf("           *** microstrain (pre-refit) ***\n");
      printf("MeanStrain %0.6lf\nStdStrain  %0.6lf\n", MeanDiff * 1e6,
             StdDiff * 1e6);

      // --- PER-ITERATION RE-FIT with optimized parameters ---
      {
        // Free pre-optimization peak arrays
        free(YMean);
        free(ZMean);
        free(Yc);
        free(Zc);
        free(EtaIns);
        free(RadIns);
        free(DiffIns);
        free(RMean);
        free(EtaMean);
        free(IdealTtheta);
        free(RingNumbers);
        YMean = ZMean = Yc = Zc = EtaIns = RadIns = DiffIns = NULL;
        RMean = EtaMean = IdealTtheta = NULL;
        RingNumbers = NULL;

        // Recompute radii for optimized Lsd
        double IdealRs_rf[n_hkls], Rmins_rf[n_hkls], Rmaxs_rf[n_hkls];
        for (i = 0; i < n_hkls; i++) {
          IdealRs_rf[i] = R4mTtheta(IdealTthetas[i], LsdFit);
          Rmins_rf[i] = R4mTtheta(TthetaMins[i], LsdFit);
          Rmaxs_rf[i] = R4mTtheta(TthetaMaxs[i], LsdFit);
        }
        // Doublets for re-fit
        int dbf_rf[n_hkls], dbp_rf[n_hkls];
        for (i = 0; i < n_hkls; i++) {
          dbf_rf[i] = 0;
          dbp_rf[i] = -1;
        }
        if (DoubletSeparation > 0) {
          double sepT = DoubletSeparation * px;
          for (i = 0; i < n_hkls - 1; i++) {
            if (dbf_rf[i] != 0)
              continue;
            if (fabs(IdealRs_rf[i + 1] - IdealRs_rf[i]) < sepT) {
              dbf_rf[i] = 1;
              dbf_rf[i + 1] = 2;
              dbp_rf[i] = i + 1;
              dbp_rf[i + 1] = i;
            }
          }
        }

        int nIdx_rf = nEtaBins * n_hkls;
        double *R_rf = malloc(NrPixels * NrPixels * sizeof(*R_rf));
        double *Eta_rf = malloc(NrPixels * NrPixels * sizeof(*Eta_rf));
        int *NrBin_rf = malloc(nIdx_rf * sizeof(*NrBin_rf));
        int **Idx_rf = allocMatrixInt(nIdx_rf, 20000);

        Car2Pol(n_hkls, nEtaBins, NrPixels, NrPixels, ybcFit, zbcFit, px, R_rf,
                Eta_rf, Rmins_rf, Rmaxs_rf, EtaBinsLow, EtaBinsHigh, nIdx_rf,
                NrBin_rf, Idx_rf, tx, ty, tz, p0, p1, p2, p3, MaxRingRad,
                LsdFit, p4);

        double *IRmins_rf = malloc(nIdx_rf * sizeof(*IRmins_rf));
        double *IRmaxs_rf = malloc(nIdx_rf * sizeof(*IRmaxs_rf));
        IdealTtheta = malloc(nIdx_rf * sizeof(*IdealTtheta));
        RingNumbers = malloc(nIdx_rf * sizeof(*RingNumbers));
        RMean = malloc(nIdx_rf * sizeof(*RMean));
        EtaMean = malloc(nIdx_rf * sizeof(*EtaMean));
        int NPF_rf =
            (int)((floor)((Rmaxs_rf[0] - Rmins_rf[0]) / px)) * RBinWidth;
        for (i = 0; i < nIdx_rf; i++) {
          int ri = (int)(floor((float)i / nEtaBins));
          IRmins_rf[i] = Rmins_rf[ri];
          IRmaxs_rf[i] = Rmaxs_rf[ri];
          IdealTtheta[i] = rad2deg * atan(IdealRs_rf[ri] / LsdFit);
          RingNumbers[i] = RingIDs[ri];
        }

        double *FitSNR_rf = calloc(nIdx_rf, sizeof(*FitSNR_rf));
        if (FitWeightMean == 1) {
          CalcWeightedMean(nIdx_rf, NrBin_rf, Idx_rf, Average, R_rf, Eta_rf,
                           RMean, EtaMean);
        } else {
          CalcFittedMean(nIdx_rf, NrBin_rf, Idx_rf, Average, R_rf, Eta_rf,
                         RMean, EtaMean, FitSNR_rf, NPF_rf, IRmins_rf,
                         IRmaxs_rf, nEtaBins, ybcFit, zbcFit, px, NrPixelsY,
                         NrPixelsZ, EtaBinsLow, EtaBinsHigh, dbf_rf, dbp_rf);
        }
        // Don't free FitSNR_rf — compact it and save for next iter snrWeights

        // Compact
        int cnt_rf = 0;
        double *RM2 = malloc(nIdx_rf * sizeof(*RM2));
        double *EM2 = malloc(nIdx_rf * sizeof(*EM2));
        double *IT2 = malloc(nIdx_rf * sizeof(*IT2));
        int *RN2 = malloc(nIdx_rf * sizeof(*RN2));
        double *FS2 = malloc(nIdx_rf * sizeof(*FS2)); // compacted FitSNR
        for (i = 0; i < nIdx_rf; i++) {
          if (RMean[i] != 0) {
            RM2[cnt_rf] = RMean[i];
            EM2[cnt_rf] = EtaMean[i];
            IT2[cnt_rf] = IdealTtheta[i];
            RN2[cnt_rf] = RingNumbers[i];
            FS2[cnt_rf] = FitSNR_rf[i];
            cnt_rf++;
          }
        }
        free(RMean);
        free(EtaMean);
        free(IdealTtheta);
        free(RingNumbers);
        free(FitSNR_rf);
        if (FitSNR)
          free(FitSNR);
        RMean = RM2;
        EtaMean = EM2;
        IdealTtheta = IT2;
        RingNumbers = RN2;
        FitSNR = FS2; // persist for next iter's snrWeights
        nIndices = cnt_rf;
        printf("Re-fit: %d valid slices (optimized geometry).\n", cnt_rf);

        // Recompute Yc, Zc
        YMean = malloc(nIndices * sizeof(*YMean));
        ZMean = malloc(nIndices * sizeof(*ZMean));
        YZ4mREta(nIndices, RMean, EtaMean, YMean, ZMean);
        Yc = malloc(nIndices * sizeof(*Yc));
        Zc = malloc(nIndices * sizeof(*Zc));
        EtaIns = malloc(nIndices * sizeof(*EtaIns));
        RadIns = malloc(nIndices * sizeof(*RadIns));
        DiffIns = malloc(nIndices * sizeof(*DiffIns));
        for (i = 0; i < nIndices; i++) {
          Yc[i] = ybcFit - (YMean[i] / px);
          Zc[i] = zbcFit + (ZMean[i] / px);
        }
        CorrectTiltSpatialDistortion(
            nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
            zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
            &StdDiff, outlierFactor, NULL, p4, OutlierIterations, 0, &MeanDiff);
        printf("           *** microstrain (re-fit) ***\n");
        printf("MeanStrain %0.6lf\nStdStrain  %0.6lf\n", MeanDiff * 1e6,
               StdDiff * 1e6);
        printf("Lsd %0.4f  BC %0.4f %0.4f  ty %0.6f  tz %0.6f\n", LsdFit,
               ybcFit, zbcFit, ty, tz);
        printf("p0 %0.9f  p1 %0.9f  p2 %0.9f  p3 %0.6f  p4 %0.9f\n", p0, p1, p2,
               p3, p4);

        // Free temp pixel arrays
        free(R_rf);
        free(Eta_rf);
        free(NrBin_rf);
        FreeMemMatrixInt(Idx_rf, nIdx_rf);
        free(IRmins_rf);
        free(IRmaxs_rf);
      }
      // --- END PER-ITERATION RE-FIT ---

      // Feed outputs back as inputs for next iteration
      p4in = p4;
      Lsd = LsdFit;
      ybc = ybcFit;
      zbc = zbcFit;
      tyin = ty;
      tzin = tz;
      p0in = p0;
      p1in = p1;
      p2in = p2;
      p3in = p3;

      // Track best iteration
      if (MeanDiff < bestMeanDiff) {
        bestMeanDiff = MeanDiff;
        bestIter = iter;
        bestLsd = LsdFit;
        bestYbc = ybcFit;
        bestZbc = zbcFit;
        bestTy = ty;
        bestTz = tz;
        bestP0 = p0;
        bestP1 = p1;
        bestP2 = p2;
        bestP3 = p3;
        bestP4 = p4;
        if (bestPanels && nPanels > 0)
          memcpy(bestPanels, panels, nPanels * sizeof(Panel));
      }

      // Stagnation: detect when optimizer is truly stuck (identical results)
      if (iter > 0 && fabs(MeanDiff - prevIterMeanDiff) < 1e-12) {
        stagnantCount++;
      } else {
        stagnantCount = 0;
      }
      prevIterMeanDiff = MeanDiff;

      // Perturbation: if stagnant for 3+ iterations, kick parameters
      if (stagnantCount >= 3 && iter < nIterations - 1) {
        // Restore from best before perturbing
        Lsd = bestLsd;
        ybc = bestYbc;
        zbc = bestZbc;
        tyin = bestTy;
        tzin = bestTz;
        p0in = bestP0;
        p1in = bestP1;
        p2in = bestP2;
        p3in = bestP3;
        p4in = bestP4;
        if (bestPanels && nPanels > 0)
          memcpy(panels, bestPanels, nPanels * sizeof(Panel));

        // Seed from iteration for reproducibility
        srand(42 + iter);
        // Random perturbation in [-1, +1] * fraction * tolerance
        double pertFrac = 0.5;
#define PERT(val, tol)                                                         \
  ((val) + pertFrac * (tol) * (2.0 * rand() / RAND_MAX - 1.0))
        Lsd = PERT(Lsd, tolLsd);
        ybc = PERT(ybc, tolBC);
        zbc = PERT(zbc, tolBC);
        tyin = PERT(tyin, tolTilts);
        tzin = PERT(tzin, tolTilts);
        p0in = PERT(p0in, tolP);
        p1in = PERT(p1in, tolP);
        p2in = PERT(p2in, tolP);
        p3in = PERT(p3in, tolP3);
        p4in = PERT(p4in, tolP4);
#undef PERT
        printf("  [Perturbation applied after %d identical iterations, "
               "restarting from best iter %d]\n",
               stagnantCount, bestIter + 1);
        stagnantCount = 0;
        perturbedFlag = 1; // next iteration will re-bin
      }

      // Save final nIndices for post-loop processing
      nIndicesFinal = nIndices;

      // Free per-iteration arrays (except those needed after final iteration)
      if (didInitBin) {
        FreeMemMatrixInt(Indices, nIndicesOrig);
        free(R);
        free(Eta);
        free(NrEachIndexBin);
        free(IdealR);
        free(IdealRmins);
        free(IdealRmaxs);
      }
      if (RingWeights)
        free(RingWeights);
      if (snrWeights)
        free(snrWeights);

      // On non-final iterations: free peak arrays only if next iter will
      // re-bin (either via perturbation or always on iter 0 → 1 which the
      // re-fit handles)
      if (iter < nIterations - 1 && perturbedFlag) {
        free(RMean);
        free(EtaMean);
        free(YMean);
        free(ZMean);
        free(IdealTtheta);
        free(RingNumbers);
        free(Yc);
        free(Zc);
        free(EtaIns);
        free(RadIns);
        free(DiffIns);
        RMean = EtaMean = YMean = ZMean = NULL;
        IdealTtheta = NULL;
        RingNumbers = NULL;
        Yc = Zc = EtaIns = RadIns = DiffIns = NULL;
      }
    } // end iteration loop

    // Restore best iteration if it wasn't the last one
    if (nIterations > 1 && bestIter >= 0) {
      if (bestIter != nIterations - 1) {
        printf("\n*** Restoring best result from iteration %d/%d "
               "(MeanStrain %.12f vs last %.12f) ***\n",
               bestIter + 1, nIterations, bestMeanDiff, MeanDiff);
        LsdFit = bestLsd;
        ybcFit = bestYbc;
        zbcFit = bestZbc;
        ty = bestTy;
        tz = bestTz;
        p0 = bestP0;
        p1 = bestP1;
        p2 = bestP2;
        p3 = bestP3;
        p4in = bestP4;
        Lsd = bestLsd;
        ybc = bestYbc;
        zbc = bestZbc;
        tyin = bestTy;
        tzin = bestTz;
        p0in = bestP0;
        p1in = bestP1;
        p2in = bestP2;
        p3in = bestP3;
        MeanDiff = bestMeanDiff;
        if (bestPanels && nPanels > 0)
          memcpy(panels, bestPanels, nPanels * sizeof(Panel));
      } else {
        printf("\n*** Best result is from final iteration %d/%d "
               "(MeanStrain %.12f) ***\n",
               bestIter + 1, nIterations, bestMeanDiff);
      }
    }
    if (bestPanels)
      free(bestPanels);

    // --- POST-LOOP RE-FIT (only if best iteration != final) ---
    // When bestIter == last, the per-iteration re-fit already produced
    // correct arrays. When restoring from an earlier best, we must re-fit.
    if (nIterations > 1 && bestIter >= 0 && bestIter != nIterations - 1) {
      printf("\n*** Post-loop re-fit with restored best parameters ***\n");
      free(RMean);
      free(EtaMean);
      free(YMean);
      free(ZMean);
      free(IdealTtheta);
      free(RingNumbers);
      free(Yc);
      free(Zc);
      free(EtaIns);
      free(RadIns);
      free(DiffIns);
      YMean = ZMean = Yc = Zc = EtaIns = RadIns = DiffIns = NULL;
      RMean = EtaMean = IdealTtheta = NULL;
      RingNumbers = NULL;

      double IdealRs_pl[n_hkls], Rmins_pl[n_hkls], Rmaxs_pl[n_hkls];
      for (i = 0; i < n_hkls; i++) {
        IdealRs_pl[i] = R4mTtheta(IdealTthetas[i], LsdFit);
        Rmins_pl[i] = R4mTtheta(TthetaMins[i], LsdFit);
        Rmaxs_pl[i] = R4mTtheta(TthetaMaxs[i], LsdFit);
      }
      int dbf_pl[n_hkls], dbp_pl[n_hkls];
      for (i = 0; i < n_hkls; i++) {
        dbf_pl[i] = 0;
        dbp_pl[i] = -1;
      }
      if (DoubletSeparation > 0) {
        double sepT = DoubletSeparation * px;
        for (i = 0; i < n_hkls - 1; i++) {
          if (dbf_pl[i] != 0)
            continue;
          if (fabs(IdealRs_pl[i + 1] - IdealRs_pl[i]) < sepT) {
            dbf_pl[i] = 1;
            dbf_pl[i + 1] = 2;
            dbp_pl[i] = i + 1;
            dbp_pl[i + 1] = i;
          }
        }
      }
      int nIdx_pl = nEtaBins * n_hkls;
      double *R_pl = malloc(NrPixels * NrPixels * sizeof(*R_pl));
      double *Eta_pl = malloc(NrPixels * NrPixels * sizeof(*Eta_pl));
      int *NrBin_pl = malloc(nIdx_pl * sizeof(*NrBin_pl));
      int **Idx_pl = allocMatrixInt(nIdx_pl, 20000);
      Car2Pol(n_hkls, nEtaBins, NrPixels, NrPixels, ybcFit, zbcFit, px, R_pl,
              Eta_pl, Rmins_pl, Rmaxs_pl, EtaBinsLow, EtaBinsHigh, nIdx_pl,
              NrBin_pl, Idx_pl, tx, ty, tz, p0, p1, p2, p3, MaxRingRad, LsdFit,
              p4in);
      double *IRmins_pl = malloc(nIdx_pl * sizeof(*IRmins_pl));
      double *IRmaxs_pl = malloc(nIdx_pl * sizeof(*IRmaxs_pl));
      IdealTtheta = malloc(nIdx_pl * sizeof(*IdealTtheta));
      RingNumbers = malloc(nIdx_pl * sizeof(*RingNumbers));
      RMean = malloc(nIdx_pl * sizeof(*RMean));
      EtaMean = malloc(nIdx_pl * sizeof(*EtaMean));
      int NPF_pl = (int)((floor)((Rmaxs_pl[0] - Rmins_pl[0]) / px)) * RBinWidth;
      for (i = 0; i < nIdx_pl; i++) {
        int ri = (int)(floor((float)i / nEtaBins));
        IRmins_pl[i] = Rmins_pl[ri];
        IRmaxs_pl[i] = Rmaxs_pl[ri];
        IdealTtheta[i] = rad2deg * atan(IdealRs_pl[ri] / LsdFit);
        RingNumbers[i] = RingIDs[ri];
      }
      double *FitSNR_pl = calloc(nIdx_pl, sizeof(*FitSNR_pl));
      if (FitWeightMean == 1) {
        CalcWeightedMean(nIdx_pl, NrBin_pl, Idx_pl, Average, R_pl, Eta_pl,
                         RMean, EtaMean);
      } else {
        CalcFittedMean(nIdx_pl, NrBin_pl, Idx_pl, Average, R_pl, Eta_pl, RMean,
                       EtaMean, FitSNR_pl, NPF_pl, IRmins_pl, IRmaxs_pl,
                       nEtaBins, ybcFit, zbcFit, px, NrPixelsY, NrPixelsZ,
                       EtaBinsLow, EtaBinsHigh, dbf_pl, dbp_pl);
      }
      free(FitSNR_pl);
      int cnt_pl = 0;
      double *RM2 = malloc(nIdx_pl * sizeof(*RM2));
      double *EM2 = malloc(nIdx_pl * sizeof(*EM2));
      double *IT2 = malloc(nIdx_pl * sizeof(*IT2));
      int *RN2 = malloc(nIdx_pl * sizeof(*RN2));
      for (i = 0; i < nIdx_pl; i++) {
        if (RMean[i] != 0) {
          RM2[cnt_pl] = RMean[i];
          EM2[cnt_pl] = EtaMean[i];
          IT2[cnt_pl] = IdealTtheta[i];
          RN2[cnt_pl] = RingNumbers[i];
          cnt_pl++;
        }
      }
      free(RMean);
      free(EtaMean);
      free(IdealTtheta);
      free(RingNumbers);
      RMean = RM2;
      EtaMean = EM2;
      IdealTtheta = IT2;
      RingNumbers = RN2;
      nIndices = cnt_pl;
      nIndicesFinal = nIndices;
      printf("Post-loop re-fit: %d valid slices.\n", cnt_pl);
      YMean = malloc(nIndices * sizeof(*YMean));
      ZMean = malloc(nIndices * sizeof(*ZMean));
      YZ4mREta(nIndices, RMean, EtaMean, YMean, ZMean);
      Yc = malloc(nIndices * sizeof(*Yc));
      Zc = malloc(nIndices * sizeof(*Zc));
      EtaIns = malloc(nIndices * sizeof(*EtaIns));
      RadIns = malloc(nIndices * sizeof(*RadIns));
      DiffIns = malloc(nIndices * sizeof(*DiffIns));
      for (i = 0; i < nIndices; i++) {
        Yc[i] = ybcFit - (YMean[i] / px);
        Zc[i] = zbcFit + (ZMean[i] / px);
      }
      CorrectTiltSpatialDistortion(
          nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit, zbcFit,
          tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns, &StdDiff,
          outlierFactor, NULL, p4in, OutlierIterations, 0, &MeanDiff);
      printf("Post-loop re-fit MeanStrain %0.6lf\n", MeanDiff * 1e6);
      free(R_pl);
      free(Eta_pl);
      free(NrBin_pl);
      FreeMemMatrixInt(Idx_pl, nIdx_pl);
      free(IRmins_pl);
      free(IRmaxs_pl);
    }
    // Reassign nIndices for post-loop code
    nIndices = nIndicesFinal;

    double *Etas, *Diffs, *RadOuts;
    Etas = malloc(nIndices * sizeof(*Etas));
    Diffs = malloc(nIndices * sizeof(*Diffs));
    RadOuts = malloc(nIndices * sizeof(*RadOuts));
    int *IsOutlier = calloc(nIndices, sizeof(int));
    CorrectTiltSpatialDistortion(
        nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit, zbcFit,
        tx, ty, tz, p0, p1, p2, p3, Etas, Diffs, RadOuts, &StdDiff,
        outlierFactor, IsOutlier, p4in, OutlierIterations, 1, &MeanDiff);
    printf("StdStrain %0.12lf\n", StdDiff);
    // Compute strain statistics from valid (non-outlier) diffs
    nValid = 0;
    for (i = 0; i < nIndices; i++)
      if (Diffs[i] >= 0 && !IsOutlier[i])
        nValid++;
    medStrain = 0;
    q25Strain = 0;
    q75Strain = 0;
    minStrain = 0;
    maxStrain = 0;
    if (nValid > 0) {
      double *sortedDiffs = malloc(nValid * sizeof(double));
      int idx = 0;
      for (i = 0; i < nIndices; i++)
        if (Diffs[i] >= 0 && !IsOutlier[i])
          sortedDiffs[idx++] = Diffs[i];
      for (int ii = 1; ii < nValid; ii++) {
        double key = sortedDiffs[ii];
        int jj = ii - 1;
        while (jj >= 0 && sortedDiffs[jj] > key) {
          sortedDiffs[jj + 1] = sortedDiffs[jj];
          jj--;
        }
        sortedDiffs[jj + 1] = key;
      }
      medStrain = sortedDiffs[nValid / 2];
      q25Strain = sortedDiffs[nValid / 4];
      q75Strain = sortedDiffs[3 * nValid / 4];
      minStrain = sortedDiffs[0];
      maxStrain = sortedDiffs[nValid - 1];
      free(sortedDiffs);
    }

    // Per-panel strain summary
    if (nPanels > 0) {
      double *panelSum = calloc(nPanels, sizeof(double));
      double *panelSumSq = calloc(nPanels, sizeof(double));
      int *panelCount = calloc(nPanels, sizeof(int));
      for (i = 0; i < nIndices; i++) {
        if (Diffs[i] < 0 || IsOutlier[i])
          continue;
        int pIdx = GetPanelIndex(Yc[i], Zc[i], nPanels, panels);
        if (pIdx >= 0 && pIdx < nPanels) {
          panelSum[pIdx] += Diffs[i];
          panelSumSq[pIdx] += Diffs[i] * Diffs[i];
          panelCount[pIdx]++;
        }
      }
      // Compute per-panel mean strain (x10000)
      double *panelMean = calloc(nPanels, sizeof(double));
      for (int p = 0; p < nPanels; p++)
        panelMean[p] =
            (panelCount[p] > 0) ? (panelSum[p] / panelCount[p] * 1e6) : -1.0;

      printf("\n*** Per-Panel Microstrain (Z^ Y>) ***\n");
      for (int z = NPanelsZ - 1; z >= 0; z--) {
        printf("Z%-1d |", z);
        for (int y = 0; y < NPanelsY; y++) {
          int pIdx = y * NPanelsZ + z;
          if (pIdx < nPanels && panelMean[pIdx] >= 0) {
            printf(" %2d:%5.1f", pIdx, panelMean[pIdx]);
          } else if (pIdx < nPanels) {
            printf(" %2d: --- ", pIdx);
          }
          printf(" |");
        }
        printf("\n");
      }
      printf("   ");
      for (int y = 0; y < NPanelsY; y++)
        printf("    Y=%-2d   ", y);
      printf("\n***********************************\n");

      // Save strain TIFF (each pixel = its panel's mean strain x10000)
      if (PanelShiftsFile[0] != '\0') {
        char strainTiffFN[2048];
        snprintf(strainTiffFN, sizeof(strainTiffFN), "%s.strain.tif",
                 PanelShiftsFile);
        int imgW = NrPixelsY, imgH = NrPixelsZ;
        int nPx = imgW * imgH;
        float *strainImg = (float *)calloc(nPx, sizeof(float));
        if (strainImg) {
          for (int row = 0; row < imgH; row++) {
            for (int col = 0; col < imgW; col++) {
              int pIdx =
                  GetPanelIndex((double)col, (double)row, nPanels, panels);
              if (pIdx >= 0 && panelMean[pIdx] >= 0)
                strainImg[row * imgW + col] = (float)panelMean[pIdx];
              else
                strainImg[row * imgW + col] = -1.0f;
            }
          }
          FILE *tf = fopen(strainTiffFN, "wb");
          if (tf) {
            unsigned char hdr[8] = {'I', 'I', 42, 0, 8, 0, 0, 0};
            fwrite(hdr, 1, 8, tf);
            unsigned short nTags = 10;
            fwrite(&nTags, 2, 1, tf);
#define WRITE_TAG_S(tag, type, count, value)                                   \
  {                                                                            \
    unsigned short t = tag, tp = type;                                         \
    unsigned int c = count, v = value;                                         \
    fwrite(&t, 2, 1, tf);                                                      \
    fwrite(&tp, 2, 1, tf);                                                     \
    fwrite(&c, 4, 1, tf);                                                      \
    fwrite(&v, 4, 1, tf);                                                      \
  }
            unsigned int dataBytes = nPx * 4;
            unsigned int stripOff = 8 + 2 + 10 * 12 + 4;
            WRITE_TAG_S(256, 3, 1, imgW);
            WRITE_TAG_S(257, 3, 1, imgH);
            WRITE_TAG_S(258, 3, 1, 32);
            WRITE_TAG_S(259, 3, 1, 1);
            WRITE_TAG_S(262, 3, 1, 1);
            WRITE_TAG_S(273, 4, 1, stripOff);
            WRITE_TAG_S(278, 4, 1, imgH);
            WRITE_TAG_S(279, 4, 1, dataBytes);
            WRITE_TAG_S(339, 3, 1, 3);
            WRITE_TAG_S(277, 3, 1, 1);
#undef WRITE_TAG_S
            unsigned int nextIFD = 0;
            fwrite(&nextIFD, 4, 1, tf);
            fwrite(strainImg, sizeof(float), nPx, tf);
            fclose(tf);
            printf("Saved per-panel strain map to %s\n", strainTiffFN);
          }
          free(strainImg);
        }
      }

      free(panelSum);
      free(panelSumSq);
      free(panelCount);
      free(panelMean);
    }

    means[0] += LsdFit;
    means[1] += ybcFit;
    means[2] += zbcFit;
    means[3] += ty;
    means[4] += tz;
    means[5] += p0;
    means[6] += p1;
    means[7] += p2;
    means[8] += p3;
    means[9] += MeanDiff;
    means[10] += StdDiff;
    means[11] += p4in;
    FILE *Out;
    char OutFileName[1024];
    sprintf(OutFileName, "%s_%0*d%s.%s", fn, Padding, a, Ext, "corr.csv");
    Out = fopen(OutFileName, "w");
    if (!Out) {
      printf("Error: cannot open output file %s\n", OutFileName);
      perror("fopen");
      return 1;
    }
    fprintf(Out, "%%Eta Strain RadFit EtaCalc DiffCalc RadCalc Ideal2Theta "
                 "Outlier YRawCorr ZRawCorr RingNr RadGlobal IdealR Fit2Theta "
                 "IdealA FitA\n");
    // Build tilt rotation matrix for RadGlobal computation
    double txrG = deg2rad * tx;
    double tyrG = deg2rad * ty;
    double tzrG = deg2rad * tz;
    double RxG[3][3] = {
        {1, 0, 0}, {0, cos(txrG), -sin(txrG)}, {0, sin(txrG), cos(txrG)}};
    double RyG[3][3] = {
        {cos(tyrG), 0, sin(tyrG)}, {0, 1, 0}, {-sin(tyrG), 0, cos(tyrG)}};
    double RzG[3][3] = {
        {cos(tzrG), -sin(tzrG), 0}, {sin(tzrG), cos(tzrG), 0}, {0, 0, 1}};
    double TRintG[3][3], TRsG[3][3];
    MatrixMultF33(RyG, RzG, TRintG);
    MatrixMultF33(RxG, TRintG, TRsG);
    // Allocate array for radius residuals (for sorting/percentiles)
    double *radResArr = malloc(nIndices * sizeof(double));
    int nRadRes = 0;
    double sumRadRes = 0, sumAbsRadRes = 0, sumRadResSq = 0;
    double localMaxAbsRadRes = 0;
    // 2theta residual accumulators for this file
    double *tthResArr = malloc(nIndices * sizeof(double));
    int nTthRes = 0;
    double sumTthRes = 0, sumAbsTthRes = 0, sumTthResSq = 0;
    double localMaxAbsTthRes = 0;
    // Lattice parameter residual accumulators for this file
    double *latResArr = malloc(nIndices * sizeof(double));
    int nLatRes = 0;
    double sumLatRes = 0, sumAbsLatRes = 0, sumLatResSq = 0;
    double localMaxAbsLatRes = 0;
    // Per-ring accumulators for this file
    double localRingSum[MAX_RINGS_STAT] = {0};
    double localRingAbsSum[MAX_RINGS_STAT] = {0};
    double localRingMax[MAX_RINGS_STAT] = {0};
    double localRingIdealR[MAX_RINGS_STAT] = {0};
    int localRingCount[MAX_RINGS_STAT] = {0};
    // Per-ring 2theta accumulators for this file
    double localRing2thSum[MAX_RINGS_STAT] = {0};
    double localRing2thAbsSum[MAX_RINGS_STAT] = {0};
    double localRing2thMax[MAX_RINGS_STAT] = {0};
    double localRingIdeal2th[MAX_RINGS_STAT] = {0};
    int localRing2thCount[MAX_RINGS_STAT] = {0};
    // Per-ring lattice param accumulators for this file
    double localRingLatSum[MAX_RINGS_STAT] = {0};
    double localRingLatAbsSum[MAX_RINGS_STAT] = {0};
    double localRingLatMax[MAX_RINGS_STAT] = {0};
    int localRingLatCount[MAX_RINGS_STAT] = {0};
    // Per-panel radius residual accumulators
    double *panelRadResSum = NULL;
    double *panelRadResAbsSum = NULL;
    int *panelRadResCount = NULL;
    if (nPanels > 0) {
      panelRadResSum = calloc(nPanels, sizeof(double));
      panelRadResAbsSum = calloc(nPanels, sizeof(double));
      panelRadResCount = calloc(nPanels, sizeof(int));
    }
    for (i = 0; i < nIndices; i++) {
      if (Diffs[i] < 0)
        continue;
      double dY = 0, dZ = 0, dTheta = 0;
      int pIdx = GetPanelIndex(Yc[i], Zc[i], nPanels, panels);
      if (pIdx >= 0) {
        dY = panels[pIdx].dY;
        dZ = panels[pIdx].dZ;
        dTheta = panels[pIdx].dTheta;
      }
      double rawY = Yc[i], rawZ = Zc[i];
      if (pIdx >= 0 && fabs(dTheta) > 1e-12) {
        double cY = panels[pIdx].centerY;
        double cZ = panels[pIdx].centerZ;
        double cosT = cos(deg2rad * dTheta);
        double sinT = sin(deg2rad * dTheta);
        double dy = rawY - cY, dz = rawZ - cZ;
        rawY = cY + dy * cosT - dz * sinT;
        rawZ = cZ + dy * sinT + dz * cosT;
      }
      double YRawCorr = rawY + dY;
      double ZRawCorr = rawZ + dZ;
      // Compute RadGlobal: corrected radius using global Lsd only (no dLsd)
      double YcG = -(rawY + dY - ybcFit) * px;
      double ZcG = (rawZ + dZ - zbcFit) * px;
      double ABCG[3] = {0, YcG, ZcG};
      double ABCPrG[3];
      MatrixMultF(TRsG, ABCG, ABCPrG);
      double XYZG[3] = {LsdFit + ABCPrG[0], ABCPrG[1], ABCPrG[2]};
      double RadG =
          (LsdFit / XYZG[0]) * sqrt(XYZG[1] * XYZG[1] + XYZG[2] * XYZG[2]);
      double EtaG = dg_calc_eta_angle(XYZG[1], XYZG[2]);
      double RNormG = RadG / MaxRingRad;
      double EtaTG = 90 - EtaG;
      double DistortG =
          (p0 * pow(RNormG, 2.0) * cos(deg2rad * (2 * EtaTG))) +
          (p1 * pow(RNormG, 4.0) * cos(deg2rad * (4 * EtaTG + p3))) +
          (p2 * pow(RNormG, 2.0));
      DistortG += p4in * pow(RNormG, 6.0);
      DistortG += 1;
      double RadGlobal = RadG * DistortG;
      double IdealR = LsdFit * tan(deg2rad * IdealTtheta[i]);
      double Fit2Theta = atan(RadGlobal / LsdFit) / deg2rad;
      // Compute fit lattice parameter: a_fit = a_ideal * sin(ideal_theta) /
      // sin(fit_theta)
      double sinIdeal = sin(deg2rad * IdealTtheta[i] / 2.0);
      double sinFit = sin(deg2rad * Fit2Theta / 2.0);
      double FitA = (sinFit > 1e-15) ? IdealA * sinIdeal / sinFit : IdealA;
      fprintf(
          Out,
          "%f %10.8f %10.8f %f %10.8f %10.8f %f %d %f %f %d %10.8f %10.8f %f "
          "%10.8f %10.8f\n",
          Etas[i], Diffs[i], RadOuts[i], EtaIns[i], DiffIns[i], RadIns[i],
          IdealTtheta[i], IsOutlier[i], YRawCorr, ZRawCorr, RingNumbers[i],
          RadGlobal, IdealR, Fit2Theta, IdealA, FitA);
      // Accumulate radius residual statistics (non-outlier only)
      if (!IsOutlier[i]) {
        double dR = RadGlobal - IdealR; // signed residual in microns
        radResArr[nRadRes++] = dR;
        sumRadRes += dR;
        sumAbsRadRes += fabs(dR);
        sumRadResSq += dR * dR;
        if (fabs(dR) > localMaxAbsRadRes)
          localMaxAbsRadRes = fabs(dR);
        // Per-ring accumulation
        int rn = RingNumbers[i];
        if (rn >= 0 && rn < MAX_RINGS_STAT) {
          localRingSum[rn] += dR;
          localRingAbsSum[rn] += fabs(dR);
          if (fabs(dR) > localRingMax[rn])
            localRingMax[rn] = fabs(dR);
          localRingIdealR[rn] = IdealR; // same for all points on this ring
          localRingCount[rn]++;
        }
        // Per-panel accumulation
        if (pIdx >= 0 && pIdx < nPanels) {
          panelRadResSum[pIdx] += dR;
          panelRadResAbsSum[pIdx] += fabs(dR);
          panelRadResCount[pIdx]++;
        }
        // 2theta residual accumulation
        double d2th = Fit2Theta - IdealTtheta[i]; // degrees
        tthResArr[nTthRes++] = d2th;
        sumTthRes += d2th;
        sumAbsTthRes += fabs(d2th);
        sumTthResSq += d2th * d2th;
        if (fabs(d2th) > localMaxAbsTthRes)
          localMaxAbsTthRes = fabs(d2th);
        // Per-ring 2theta accumulation
        if (rn >= 0 && rn < MAX_RINGS_STAT) {
          localRing2thSum[rn] += d2th;
          localRing2thAbsSum[rn] += fabs(d2th);
          if (fabs(d2th) > localRing2thMax[rn])
            localRing2thMax[rn] = fabs(d2th);
          localRingIdeal2th[rn] = IdealTtheta[i];
          localRing2thCount[rn]++;
        }
        // Lattice parameter residual accumulation
        double dLat = FitA - IdealA; // Angstroms
        latResArr[nLatRes++] = dLat;
        sumLatRes += dLat;
        sumAbsLatRes += fabs(dLat);
        sumLatResSq += dLat * dLat;
        if (fabs(dLat) > localMaxAbsLatRes)
          localMaxAbsLatRes = fabs(dLat);
        // Per-ring lattice param accumulation
        if (rn >= 0 && rn < MAX_RINGS_STAT) {
          localRingLatSum[rn] += dLat;
          localRingLatAbsSum[rn] += fabs(dLat);
          if (fabs(dLat) > localRingLatMax[rn])
            localRingLatMax[rn] = fabs(dLat);
          localRingLatCount[rn]++;
        }
      }
    }
    fclose(Out);

    // --- Compute radius residual statistics ---
    nValidRadRes = nRadRes;
    if (nRadRes > 0) {
      meanRadRes = sumRadRes / nRadRes;
      meanAbsRadRes = sumAbsRadRes / nRadRes;
      maxAbsRadRes = localMaxAbsRadRes;
      stdRadRes = sqrt(sumRadResSq / nRadRes - meanRadRes * meanRadRes);
      // Sort signed residuals for percentiles
      for (int ii = 1; ii < nRadRes; ii++) {
        double key = radResArr[ii];
        int jj = ii - 1;
        while (jj >= 0 && radResArr[jj] > key) {
          radResArr[jj + 1] = radResArr[jj];
          jj--;
        }
        radResArr[jj + 1] = key;
      }
      medRadRes = radResArr[nRadRes / 2];
      q25RadRes = radResArr[nRadRes / 4];
      q75RadRes = radResArr[3 * nRadRes / 4];
      minRadRes = radResArr[0];
      maxRadRes = radResArr[nRadRes - 1];
    }
    free(radResArr);

    // --- Compute 2theta residual statistics ---
    nValid2thRes = nTthRes;
    if (nTthRes > 0) {
      mean2thRes = sumTthRes / nTthRes;
      meanAbs2thRes = sumAbsTthRes / nTthRes;
      maxAbs2thRes = localMaxAbsTthRes;
      std2thRes = sqrt(sumTthResSq / nTthRes - mean2thRes * mean2thRes);
      for (int ii = 1; ii < nTthRes; ii++) {
        double key = tthResArr[ii];
        int jj = ii - 1;
        while (jj >= 0 && tthResArr[jj] > key) {
          tthResArr[jj + 1] = tthResArr[jj];
          jj--;
        }
        tthResArr[jj + 1] = key;
      }
      med2thRes = tthResArr[nTthRes / 2];
      q252thRes = tthResArr[nTthRes / 4];
      q752thRes = tthResArr[3 * nTthRes / 4];
      min2thRes = tthResArr[0];
      max2thRes = tthResArr[nTthRes - 1];
    }
    free(tthResArr);

    // Copy per-ring stats for final summary (last file wins, like strain
    // stats)
    for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
      ringMeanRadRes[rr] =
          (localRingCount[rr] > 0) ? localRingSum[rr] / localRingCount[rr] : 0;
      ringMeanAbsRadRes[rr] = (localRingCount[rr] > 0)
                                  ? localRingAbsSum[rr] / localRingCount[rr]
                                  : 0;
      ringMaxAbsRadRes[rr] = localRingMax[rr];
      ringIdealR[rr] = localRingIdealR[rr];
      ringCountRadRes[rr] = localRingCount[rr];
    }
    // Copy per-ring 2theta stats
    for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
      ringMean2thRes[rr] = (localRing2thCount[rr] > 0)
                               ? localRing2thSum[rr] / localRing2thCount[rr]
                               : 0;
      ringMeanAbs2thRes[rr] =
          (localRing2thCount[rr] > 0)
              ? localRing2thAbsSum[rr] / localRing2thCount[rr]
              : 0;
      ringMax2thAbs[rr] = localRing2thMax[rr];
      ringIdeal2th[rr] = localRingIdeal2th[rr];
      ringCount2thRes[rr] = localRing2thCount[rr];
    }

    // --- Compute lattice parameter residual statistics ---
    nValidLatRes = nLatRes;
    if (nLatRes > 0) {
      meanLatRes = sumLatRes / nLatRes;
      meanAbsLatRes = sumAbsLatRes / nLatRes;
      maxAbsLatRes = localMaxAbsLatRes;
      stdLatRes = sqrt(sumLatResSq / nLatRes - meanLatRes * meanLatRes);
      for (int ii = 1; ii < nLatRes; ii++) {
        double key = latResArr[ii];
        int jj = ii - 1;
        while (jj >= 0 && latResArr[jj] > key) {
          latResArr[jj + 1] = latResArr[jj];
          jj--;
        }
        latResArr[jj + 1] = key;
      }
      medLatRes = latResArr[nLatRes / 2];
      q25LatRes = latResArr[nLatRes / 4];
      q75LatRes = latResArr[3 * nLatRes / 4];
      minLatRes = latResArr[0];
      maxLatRes = latResArr[nLatRes - 1];
    }
    free(latResArr);
    // Copy per-ring lattice param stats
    for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
      ringMeanLatRes[rr] = (localRingLatCount[rr] > 0)
                               ? localRingLatSum[rr] / localRingLatCount[rr]
                               : 0;
      ringMeanAbsLatRes[rr] =
          (localRingLatCount[rr] > 0)
              ? localRingLatAbsSum[rr] / localRingLatCount[rr]
              : 0;
      ringMaxAbsLat[rr] = localRingLatMax[rr];
      ringCountLatRes[rr] = localRingLatCount[rr];
    }

    // Print per-panel radius residual grid (if panels defined)
    if (nPanels > 0 && panelRadResCount) {
      printf("*** Per-Panel Mean|\u0394R| (\u03bcm) (Z^ Y>) ***\n");
      for (int z = NPanelsZ - 1; z >= 0; z--) {
        printf("Z%-1d |", z);
        for (int y = 0; y < NPanelsY; y++) {
          int pIdx = y * NPanelsZ + z;
          if (pIdx < nPanels && panelRadResCount[pIdx] > 0) {
            double pmean = panelRadResAbsSum[pIdx] / panelRadResCount[pIdx];
            printf(" %2d:%6.2f", pIdx, pmean);
          } else if (pIdx < nPanels) {
            printf(" %2d:  --- ", pIdx);
          }
          printf(" |");
        }
        printf("\n");
      }
      printf("   ");
      for (int y = 0; y < NPanelsY; y++)
        printf("    Y=%-2d   ", y);
      printf("\n***********************************\n\n");
    }
    if (panelRadResSum)
      free(panelRadResSum);
    if (panelRadResAbsSum)
      free(panelRadResAbsSum);
    if (panelRadResCount)
      free(panelRadResCount);

    // Free arrays kept from the final iteration
    // (R, Eta, Indices, NrEachIndexBin, IdealR, IdealRmins, IdealRmaxs
    //  were already freed inside the iteration loop)
    if (IdealTtheta)
      free(IdealTtheta);
    if (RMean)
      free(RMean);
    if (EtaMean)
      free(EtaMean);
    if (YMean)
      free(YMean);
    if (ZMean)
      free(ZMean);
    if (RingNumbers)
      free(RingNumbers);
    if (Yc)
      free(Yc);
    if (Zc)
      free(Zc);
    if (EtaIns)
      free(EtaIns);
    if (RadIns)
      free(RadIns);
    if (DiffIns)
      free(DiffIns);
    if (FitSNR)
      free(FitSNR);
    free(Diffs);
    free(Etas);
    free(IsOutlier);
    end = omp_get_wtime();
    diftotal = end - start;
    printf("Time elapsed for this file:\t%f s.\n", diftotal);
  }
  end0 = omp_get_wtime();
  diftotal = end0 - start0;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  printf("NPoints    %d\n", nValid);

  // --- Aggregate radius residual statistics ---
  printf("           *** radius residual (\u03bcm) ***\n");
  printf("  Mean\u0394R      %+12.4f   (signed bias)\n", meanRadRes);
  printf("  Mean|\u0394R|    %12.4f   (absolute)\n", meanAbsRadRes);
  printf("  Std\u0394R       %12.4f\n", stdRadRes);
  printf("  Max|\u0394R|     %12.4f\n", maxAbsRadRes);
  printf("  Median\u0394R   %+12.4f\n", medRadRes);
  printf("  Q25\u0394R      %+12.4f\n", q25RadRes);
  printf("  Q75\u0394R      %+12.4f\n", q75RadRes);
  printf("  Min\u0394R      %+12.4f\n", minRadRes);
  printf("  Max\u0394R      %+12.4f\n", maxRadRes);
  printf("  NPoints    %12d\n", nValidRadRes);
  // Per-ring summary in final output
  printf("\n           *** per-ring radius residual (\u03bcm) ***\n");
  printf("  Ring    IdealR(\u03bcm)  NPoints   Mean|\u0394R|    Max|\u0394R|   "
         "  Mean\u0394R\n");
  printf(
      "  "
      "------------------------------------------------------------------\n");
  for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
    if (ringCountRadRes[rr] > 0) {
      printf("  %4d  %11.2f  %6d  %10.4f  %10.4f  %+10.4f\n", rr,
             ringIdealR[rr], ringCountRadRes[rr], ringMeanAbsRadRes[rr],
             ringMaxAbsRadRes[rr], ringMeanRadRes[rr]);
    }
  }
  printf(
      "  "
      "------------------------------------------------------------------\n");

  // --- Aggregate 2theta residual statistics ---
  printf("           *** 2theta residual (deg) ***\n");
  printf("  Mean\u0394(2\u03b8)    %+12.6f   (signed bias)\n", mean2thRes);
  printf("  Mean|\u0394(2\u03b8)|  %12.6f   (absolute)\n", meanAbs2thRes);
  printf("  Std\u0394(2\u03b8)     %12.6f\n", std2thRes);
  printf("  Max|\u0394(2\u03b8)|   %12.6f\n", maxAbs2thRes);
  printf("  Median\u0394(2\u03b8) %+12.6f\n", med2thRes);
  printf("  Q25\u0394(2\u03b8)    %+12.6f\n", q252thRes);
  printf("  Q75\u0394(2\u03b8)    %+12.6f\n", q752thRes);
  printf("  Min\u0394(2\u03b8)    %+12.6f\n", min2thRes);
  printf("  Max\u0394(2\u03b8)    %+12.6f\n", max2thRes);
  printf("  NPoints    %12d\n", nValid2thRes);
  // Per-ring 2theta summary
  printf("\n           *** per-ring 2theta residual (deg) ***\n");
  printf("  Ring  Ideal2\u03b8(deg)  NPoints  Mean|\u0394(2\u03b8)|   "
         "Max|\u0394(2\u03b8)|  Mean\u0394(2\u03b8)\n");
  printf(
      "  "
      "------------------------------------------------------------------\n");
  for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
    if (ringCount2thRes[rr] > 0) {
      printf("  %4d  %11.4f  %6d  %12.6f %12.6f %+12.6f\n", rr,
             ringIdeal2th[rr], ringCount2thRes[rr], ringMeanAbs2thRes[rr],
             ringMax2thAbs[rr], ringMean2thRes[rr]);
    }
  }
  printf(
      "  "
      "------------------------------------------------------------------\n");

  // --- Aggregate lattice parameter residual statistics ---
  printf("           *** lattice parameter residual (\u00c5) ***\n");
  printf("  IdealA      %12.8f\n", IdealA);
  printf("  Mean\u0394a     %+12.8f   (signed bias)\n", meanLatRes);
  printf("  Mean|\u0394a|    %12.8f   (absolute)\n", meanAbsLatRes);
  printf("  Std\u0394a       %12.8f\n", stdLatRes);
  printf("  Max|\u0394a|     %12.8f\n", maxAbsLatRes);
  printf("  Median\u0394a   %+12.8f\n", medLatRes);
  printf("  Q25\u0394a      %+12.8f\n", q25LatRes);
  printf("  Q75\u0394a      %+12.8f\n", q75LatRes);
  printf("  Min\u0394a      %+12.8f\n", minLatRes);
  printf("  Max\u0394a      %+12.8f\n", maxLatRes);
  printf("  NPoints    %12d\n", nValidLatRes);
  // Per-ring lattice parameter summary
  printf("\n           *** per-ring lattice parameter residual (\u00c5) ***\n");
  printf("  Ring  NPoints    Mean|\u0394a|       Max|\u0394a|        "
         "Mean\u0394a\n");
  printf(
      "  "
      "------------------------------------------------------------------\n");
  for (int rr = 0; rr < MAX_RINGS_STAT; rr++) {
    if (ringCountLatRes[rr] > 0) {
      printf("  %4d  %6d  %12.8f  %12.8f  %+12.8f\n", rr, ringCountLatRes[rr],
             ringMeanAbsLatRes[rr], ringMaxAbsLat[rr], ringMeanLatRes[rr]);
    }
  }
  printf(
      "  "
      "------------------------------------------------------------------\n");
  printf("*******************Copy to par*******************\n");
  free(DarkFile);
  free(AverageDark);
  free(Average);
  free(Image);

  // Save Panel Shifts
  if (PanelShiftsFile[0] != '\0' && nPanels > 0) {
    SavePanelShifts(PanelShiftsFile, nPanels, panels);
    printf("Saved panel shifts to %s\n", PanelShiftsFile);

    // Print panel shift statistics (excluding fixPanel and panels with no
    // points)
    {
      double *dYarr = malloc(nPanels * sizeof(double));
      double *dZarr = malloc(nPanels * sizeof(double));
      double *dTarr = malloc(nPanels * sizeof(double));
      int nValid = 0;
      for (int i = 0; i < nPanels; i++) {
        if (i == FixPanelID)
          continue;
        if (fabs(panels[i].dY) < 1e-15 && fabs(panels[i].dZ) < 1e-15 &&
            fabs(panels[i].dTheta) < 1e-15)
          continue;
        dYarr[nValid] = panels[i].dY;
        dZarr[nValid] = panels[i].dZ;
        dTarr[nValid] = panels[i].dTheta;
        nValid++;
      }
      if (nValid > 0) {
        // Sort for median (simple insertion sort — nPanels is small)
        for (int i = 1; i < nValid; i++) {
          double ky = dYarr[i], kz = dZarr[i], kt = dTarr[i];
          int j = i - 1;
          while (j >= 0 && dYarr[j] > ky) {
            dYarr[j + 1] = dYarr[j];
            j--;
          }
          dYarr[j + 1] = ky;
          // Sort dZ separately
          j = i - 1;
          kz = dZarr[i];
          while (j >= 0 && dZarr[j] > kz) {
            dZarr[j + 1] = dZarr[j];
            j--;
          }
          dZarr[j + 1] = kz;
          // Sort dTheta separately
          j = i - 1;
          kt = dTarr[i];
          while (j >= 0 && dTarr[j] > kt) {
            dTarr[j + 1] = dTarr[j];
            j--;
          }
          dTarr[j + 1] = kt;
        }
        double medY = (nValid % 2)
                          ? dYarr[nValid / 2]
                          : (dYarr[nValid / 2 - 1] + dYarr[nValid / 2]) / 2.0;
        double medZ = (nValid % 2)
                          ? dZarr[nValid / 2]
                          : (dZarr[nValid / 2 - 1] + dZarr[nValid / 2]) / 2.0;
        double medT = (nValid % 2)
                          ? dTarr[nValid / 2]
                          : (dTarr[nValid / 2 - 1] + dTarr[nValid / 2]) / 2.0;
        // Mean, min, max, std
        double sumY = 0, sumZ = 0, sumT = 0, sumY2 = 0, sumZ2 = 0, sumT2 = 0;
        double minY = dYarr[0], maxY = dYarr[nValid - 1];
        double minZ = dZarr[0], maxZ = dZarr[nValid - 1];
        double minT = dTarr[0], maxT = dTarr[nValid - 1];
        for (int i = 0; i < nValid; i++) {
          sumY += dYarr[i];
          sumY2 += dYarr[i] * dYarr[i];
          sumZ += dZarr[i];
          sumZ2 += dZarr[i] * dZarr[i];
          sumT += dTarr[i];
          sumT2 += dTarr[i] * dTarr[i];
        }
        double meanY = sumY / nValid, meanZ = sumZ / nValid,
               meanT = sumT / nValid;
        double stdY = sqrt(sumY2 / nValid - meanY * meanY);
        double stdZ = sqrt(sumZ2 / nValid - meanZ * meanZ);
        double stdT = sqrt(sumT2 / nValid - meanT * meanT);
        printf("\n*** Panel Shift Statistics (%d optimized panels, excluding "
               "fixPanel=%d) ***\n",
               nValid, FixPanelID);
        printf("         %12s %12s %12s %12s %12s\n", "Mean", "Median", "Min",
               "Max", "Std");
        printf("  dY:    %12.4f %12.4f %12.4f %12.4f %12.4f\n", meanY, medY,
               minY, maxY, stdY);
        printf("  dZ:    %12.4f %12.4f %12.4f %12.4f %12.4f\n", meanZ, medZ,
               minZ, maxZ, stdZ);
        printf("  dTheta:%12.6f %12.6f %12.6f %12.6f %12.6f\n", meanT, medT,
               minT, maxT, stdT);
        printf("***************************************************************"
               "**********\n");
      } else {
        printf("\n*** No optimized panels to report statistics for. ***\n");
      }
      free(dYarr);
      free(dZarr);
      free(dTarr);
    }

    // Write per-pixel shift magnitude TIFF
    {
      char shiftsTiffFN[2048];
      snprintf(shiftsTiffFN, sizeof(shiftsTiffFN), "%s.shifts.tif",
               PanelShiftsFile);

      int imgW = NrPixelsY; // columns
      int imgH = NrPixelsZ; // rows
      int nPixels = imgW * imgH;
      float *shiftImg = (float *)calloc(nPixels, sizeof(float));
      if (shiftImg) {
        for (int row = 0; row < imgH; row++) {
          for (int col = 0; col < imgW; col++) {
            int pIdx = GetPanelIndex((double)col, (double)row, nPanels, panels);
            float mag = 0.0f;
            if (pIdx >= 0) {
              double dY = panels[pIdx].dY;
              double dZ = panels[pIdx].dZ;
              double dT = panels[pIdx].dTheta;
              // Shift from rotation at this pixel (max displacement at panel
              // edge)
              double cY = panels[pIdx].centerY;
              double cZ = panels[pIdx].centerZ;
              double dy = col - cY, dz = row - cZ;
              double cosT = cos(deg2rad * dT);
              double sinT = sin(deg2rad * dT);
              double rotDY = (cY + dy * cosT - dz * sinT) -
                             col; // rotation-induced Y shift
              double rotDZ = (cZ + dy * sinT + dz * cosT) -
                             row; // rotation-induced Z shift
              double totalDY = dY + rotDY;
              double totalDZ = dZ + rotDZ;
              mag = (float)sqrt(totalDY * totalDY + totalDZ * totalDZ);
            } else {
              mag = -1.0f; // gap pixel
            }
            shiftImg[row * imgW + col] = mag;
          }
        }

        // Write minimal uncompressed float32 TIFF (little-endian)
        FILE *tf = fopen(shiftsTiffFN, "wb");
        if (tf) {
          // TIFF Header
          unsigned char hdr[8] = {'I', 'I', 42, 0,
                                  8,   0,   0,  0}; // LE, IFD at offset 8
          fwrite(hdr, 1, 8, tf);

          // IFD with 10 entries
          unsigned short nTags = 10;
          fwrite(&nTags, 2, 1, tf);

// Helper: write one IFD entry (12 bytes)
#define WRITE_TAG(tag, type, count, value)                                     \
  {                                                                            \
    unsigned short t = tag, tp = type;                                         \
    unsigned int c = count, v = value;                                         \
    fwrite(&t, 2, 1, tf);                                                      \
    fwrite(&tp, 2, 1, tf);                                                     \
    fwrite(&c, 4, 1, tf);                                                      \
    fwrite(&v, 4, 1, tf);                                                      \
  }

          unsigned int dataBytes = nPixels * 4;
          unsigned int stripOffset =
              8 + 2 + 10 * 12 + 4; // header + nTags + entries + next_ifd

          WRITE_TAG(256, 3, 1, imgW);        // ImageWidth
          WRITE_TAG(257, 3, 1, imgH);        // ImageLength
          WRITE_TAG(258, 3, 1, 32);          // BitsPerSample
          WRITE_TAG(259, 3, 1, 1);           // Compression (none)
          WRITE_TAG(262, 3, 1, 1);           // PhotometricInterp (MinIsBlack)
          WRITE_TAG(273, 4, 1, stripOffset); // StripOffsets
          WRITE_TAG(278, 4, 1, imgH);        // RowsPerStrip (all rows)
          WRITE_TAG(279, 4, 1, dataBytes);   // StripByteCounts
          WRITE_TAG(339, 3, 1, 3);           // SampleFormat (IEEEFP)
          WRITE_TAG(277, 3, 1, 1);           // SamplesPerPixel

#undef WRITE_TAG

          unsigned int nextIFD = 0;
          fwrite(&nextIFD, 4, 1, tf);

          // Pixel data
          fwrite(shiftImg, sizeof(float), nPixels, tf);
          fclose(tf);
          printf("Saved shift map to %s (%d x %d, float32)\n", shiftsTiffFN,
                 imgW, imgH);
        }
        free(shiftImg);
      }
    }
  }
  // Print final fitted parameters (moved to end for easy copy/paste)
  printf("\n*******************Mean Values*******************\n");
  for (a = 0; a < 12; a++)
    means[a] /= (EndNr - StartNr + 1);
  printf("Lsd        %0.12f\n"
         "BC         %0.12f %0.12f\n"
         "ty         %0.12f\n"
         "tz         %0.12f\n"
         "p0         %0.12f\n"
         "p1         %0.12f\n"
         "p2         %0.12f\n"
         "p3         %0.12f\n",
         means[0], means[1], means[2], means[3], means[4], means[5], means[6],
         means[7], means[8]);
  printf("p4         %0.12f\n", means[11]);
  printf("           *** microstrain ***\n");
  printf("MeanStrain %0.6lf\nStdStrain  %0.6lf\n", means[9] * 1e6,
         means[10] * 1e6);
  printf("MedianStr  %0.6lf\n", medStrain * 1e6);
  printf("Q25        %0.6lf\n", q25Strain * 1e6);
  printf("Q75        %0.6lf\n", q75Strain * 1e6);
  printf("MinStrain  %0.6lf\n", minStrain * 1e6);
  printf("MaxStrain  %0.6lf\n", maxStrain * 1e6);
  printf("NPoints    %d\n", nValid);
  return 0;
}