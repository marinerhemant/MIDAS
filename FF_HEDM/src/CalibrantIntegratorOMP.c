//
// CalibrantIntegratorOMP.c — Calibrant fitting using Green's theorem mapping
//
// Uses mapper_build_map (sub-pixel area-weighted mapping) + integration_apply_map
// for the E-step, replacing the simple pixel-by-pixel Car2Pol binning from
// CalibrantPanelShiftsOMP. M-step uses CalibrationCore shared functions.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "CalibrationCore.h"
#include "FileReader.h"
#include "IntegrationCore.h"
#include "MIDAS_ParamParser.h"
#include "MapperCore.h"
#include "Panel.h"
#include "midas_paths.h"
#include "midas_version.h"
#include "PeakFit.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Globals required by CalibrationCore (extern'd in CalibrationCore.h)
Panel *panels = NULL;
int nPanels = 0;
int numProcs = 1;
long long int NrCalls = 0;

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef double pixelvalue;

static int skipFrame = 0;

// ── Parameter parsing ─────────────────────────────────────────────

static int parse_parameters(const char *filename, MIDASConfig *cfg) {
  if (midas_parse_params(filename, cfg) != 0)
    return -1;
  skipFrame = cfg->SkipFrame;
  if (cfg->NPanelsY > 0 && cfg->NPanelsZ > 0) {
    if (GeneratePanels(cfg->NPanelsY, cfg->NPanelsZ, cfg->PanelSizeY,
                       cfg->PanelSizeZ, cfg->PanelGapsY, cfg->PanelGapsZ,
                       &panels, &nPanels) != 0) {
      fprintf(stderr, "Panel generation failed.\n");
      return -1;
    }
    printf("Generated %d panels.\n", nPanels);
  }
  if (cfg->PanelShiftsFile[0] != '\0')
    LoadPanelShifts(cfg->PanelShiftsFile, nPanels, panels);
  if (cfg->FitParallax && cfg->tolParallax < 1e-12)
    cfg->tolParallax = 200.0;
  return 0;
}

// ── E-step: mapper → integrate → peak fit ─────────────────────────
// Returns number of valid bins (nIndices).
// Outputs: YMean, ZMean, IdealTtheta, RingNumbers, FitSNR, skipBin
// Caller must free all output arrays.

static int run_estep(
    double *Average, double *AverageDark,
    int NrPixelsY, int NrPixelsZ, int NrPixels,
    double px, double Lsd, double ybc, double zbc,
    double tx, double ty, double tz,
    double p0, double p1, double p2, double p3, double p4, double p5,
    double MaxRingRad, double EtaBinSize, int RBinWidth,
    double parallax, int n_hkls,
    double *Thetas, double *DSpacings, int *RingIDs,
    int NrTransOpt, const int TransOpt[10],
    double *mask, double DoubletSeparation,
    double Wavelength, double Width,
    int peakFitMode,
    // outputs:
    double **out_RMean, double **out_EtaMean,
    double **out_YMean, double **out_ZMean,
    double **out_IdealTtheta, double **out_PointDSpacing,
    int **out_RingNumbers, double **out_FitSNR,
    int **out_skipBin)
{

  // Build R bin edges: one contiguous set covering all rings
  // with fine RBinWidth subdivision within each ring region.
  double globalRMin = 1e20, globalRMax = -1e20;
  double *ringRLo = malloc(n_hkls * sizeof(double));
  double *ringRHi = malloc(n_hkls * sizeof(double));
  int *ringRBinStart = malloc(n_hkls * sizeof(int)); // index into global R bins
  int *ringNRBins = malloc(n_hkls * sizeof(int));

  double halfW_px = Width / px; // ring half-width in pixels
  for (int r = 0; r < n_hkls; r++) {
    double IdealR = Lsd * tan(deg2rad * 2.0 * Thetas[r]) / px; // pixels
    ringRLo[r] = IdealR - halfW_px;
    ringRHi[r] = IdealR + halfW_px;
    if (ringRLo[r] < 0) ringRLo[r] = 0;
    if (ringRLo[r] < globalRMin) globalRMin = ringRLo[r];
    if (ringRHi[r] > globalRMax) globalRMax = ringRHi[r];
  }

  // Fine R bins: subdivide each pixel by RBinWidth
  double rBinSize = 1.0 / RBinWidth; // pixels per bin
  int totalRBins = 0;
  for (int r = 0; r < n_hkls; r++) {
    ringRBinStart[r] = totalRBins;
    ringNRBins[r] = (int)ceil((ringRHi[r] - ringRLo[r]) / rBinSize);
    if (ringNRBins[r] < 3) ringNRBins[r] = 3;
    totalRBins += ringNRBins[r];
  }

  // Build global R bin edge arrays
  double *RBinsLow = malloc(totalRBins * sizeof(double));
  double *RBinsHigh = malloc(totalRBins * sizeof(double));
  for (int r = 0; r < n_hkls; r++) {
    for (int b = 0; b < ringNRBins[r]; b++) {
      int idx = ringRBinStart[r] + b;
      RBinsLow[idx] = ringRLo[r] + b * rBinSize;
      RBinsHigh[idx] = ringRLo[r] + (b + 1) * rBinSize;
    }
  }

  // Eta bins: full 360° coverage
  int nEtaBins = (int)(360.0 / EtaBinSize);
  if (nEtaBins < 1) nEtaBins = 1;
  double *EtaBinsLow = malloc(nEtaBins * sizeof(double));
  double *EtaBinsHigh = malloc(nEtaBins * sizeof(double));
  for (int e = 0; e < nEtaBins; e++) {
    EtaBinsLow[e] = -180.0 + e * EtaBinSize;
    EtaBinsHigh[e] = -180.0 + (e + 1) * EtaBinSize;
  }

  // Allocate map structures
  long long nTotalBins = (long long)totalRBins * nEtaBins;
  struct MapPixelData ***pxList = malloc(totalRBins * sizeof(*pxList));
  int **nPxList = malloc(totalRBins * sizeof(*nPxList));
  int **maxnPx = malloc(totalRBins * sizeof(*maxnPx));
  int *binMaskFlag = calloc(nTotalBins, sizeof(int));
  for (int r = 0; r < totalRBins; r++) {
    pxList[r] = calloc(nEtaBins, sizeof(*pxList[r]));
    nPxList[r] = calloc(nEtaBins, sizeof(*nPxList[r]));
    maxnPx[r] = calloc(nEtaBins, sizeof(*maxnPx[r]));
    for (int e = 0; e < nEtaBins; e++) {
      maxnPx[r][e] = 2;
      pxList[r][e] = malloc(2 * sizeof(struct MapPixelData));
    }
  }

  // Initialize bin locks
  omp_lock_t binLocks[MAPPER_N_BIN_LOCKS];
  for (int i = 0; i < MAPPER_N_BIN_LOCKS; i++)
    omp_init_lock(&binLocks[i]);

  // Zero distortion maps (not used in calibrant)
  double *distortionMapY = calloc((size_t)NrPixelsY * NrPixelsZ, sizeof(double));
  double *distortionMapZ = calloc((size_t)NrPixelsY * NrPixelsZ, sizeof(double));

  printf("Building map: %d R bins x %d Eta bins = %lld total bins\n",
         totalRBins, nEtaBins, nTotalBins);

  // Build the map
  long long nEntries = mapper_build_map(
      tx, ty, tz, NrPixelsY, NrPixelsZ, px, px, ybc, zbc, Lsd, MaxRingRad,
      p0, p1, p2, p3, p4, p5,
      EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh,
      totalRBins, nEtaBins,
      pxList, nPxList, maxnPx,
      mask, binMaskFlag,
      NrTransOpt, TransOpt,
      binLocks, 1, 360.0, // SubPixelLevel=1: no sub-pixel splitting
      parallax, 0, 0, 0.0, // no solid angle/polarization correction
      distortionMapY, distortionMapZ,
      panels, nPanels);

  printf("Map built: %lld pixel-bin entries\n", nEntries);

  // Integrate: map × image → profiles
  double *profiles = calloc(nTotalBins, sizeof(double));
  double *norm = calloc(nTotalBins, sizeof(double));

  integration_apply_map(pxList, nPxList, totalRBins, nEtaBins,
                        Average, NrPixelsY, NrPixelsZ,
                        AverageDark, profiles, norm);

  // Normalize profiles by area weight
  for (long long b = 0; b < nTotalBins; b++) {
    if (norm[b] > 1e-10)
      profiles[b] /= norm[b];
    else
      profiles[b] = 0;
  }


  // Peak fitting: for each ring × eta, extract 1D radial profile and fit
  int maxBins = n_hkls * nEtaBins;
  double *RMean = calloc(maxBins, sizeof(double));
  double *EtaMean = calloc(maxBins, sizeof(double));
  double *IdealTtheta = malloc(maxBins * sizeof(double));
  double *PointDSpacing = malloc(maxBins * sizeof(double));
  int *RingNumbers = malloc(maxBins * sizeof(int));
  double *FitSNR = calloc(maxBins, sizeof(double));
  int *skipBinOut = calloc(maxBins, sizeof(int));

  // Detect doublets
  int *dbFlag = calloc(n_hkls, sizeof(int));
  int *dbPair = malloc(n_hkls * sizeof(int));
  for (int r = 0; r < n_hkls; r++) dbPair[r] = -1;
  if (DoubletSeparation > 0) {
    double sepPx = DoubletSeparation;
    for (int r = 0; r < n_hkls - 1; r++) {
      if (dbFlag[r] != 0) continue;
      double R1 = Lsd * tan(deg2rad * 2.0 * Thetas[r]) / px;
      double R2 = Lsd * tan(deg2rad * 2.0 * Thetas[r+1]) / px;
      if (fabs(R2 - R1) < sepPx) {
        dbFlag[r] = 1; dbFlag[r+1] = 2;
        dbPair[r] = r+1; dbPair[r+1] = r;
      }
    }
  }

  int nValid = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:nValid)
  for (int re = 0; re < n_hkls * nEtaBins; re++) {
    int r = re / nEtaBins;
    int e = re % nEtaBins;
    int binIdx = r * nEtaBins + e;

    // Check if any R bin in this ring×eta is mask-contaminated
    int masked = 0;
    for (int rb = 0; rb < ringNRBins[r]; rb++) {
      long long fPos = (long long)(ringRBinStart[r] + rb) * nEtaBins + e;
      if (binMaskFlag[fPos]) { masked = 1; break; }
    }
    if (masked) {
      skipBinOut[binIdx] = 1;
      continue;
    }

    // Extract 1D radial profile for this ring×eta
    int nPts = ringNRBins[r];
    double *Rs = malloc(nPts * sizeof(double));
    double *PeakShape = malloc(nPts * sizeof(double));
    int hasData = 0;
    for (int rb = 0; rb < nPts; rb++) {
      int globalRIdx = ringRBinStart[r] + rb;
      Rs[rb] = (RBinsLow[globalRIdx] + RBinsHigh[globalRIdx]) * 0.5;
      long long fPos = (long long)globalRIdx * nEtaBins + e;
      PeakShape[rb] = profiles[fPos];
      if (PeakShape[rb] > 0) hasData = 1;
    }

    if (!hasData || nPts < 5) {
      free(Rs); free(PeakShape);
      continue;
    }

    double IdealR = Lsd * tan(deg2rad * 2.0 * Thetas[r]) / px;
    double Rfit, snr;
    double Rstep = rBinSize;

    if (dbFlag[r] == 0) {
      // Singlet fit
      pf_fit_single_peak(peakFitMode, nPts, Rs, PeakShape, &Rfit, &snr, Rstep, IdealR);
    } else if (dbFlag[r] == 1) {
      // Doublet: this is the inner ring
      int pairR = dbPair[r];
      double IdealR2 = Lsd * tan(deg2rad * 2.0 * Thetas[pairR]) / px;
      double Rmid = (IdealR + IdealR2) * 0.5;
      double Rfit2, snr2;
      pf_fit_doublet_peak(peakFitMode, nPts, Rs, PeakShape,
                           &Rfit, &Rfit2, &snr, &snr2,
                           Rstep, IdealR, IdealR2, Rmid);
    } else {
      // dbFlag[r] == 2: outer ring of doublet — fit independently as singlet
      pf_fit_single_peak(peakFitMode, nPts, Rs, PeakShape, &Rfit, &snr, Rstep, IdealR);
    }

    if (snr < 1.0) {
      free(Rs); free(PeakShape);
      continue;
    }

    // Convert (R_fit, eta_center) → (Y, Z) in microns
    double etaCenter = (EtaBinsLow[e] + EtaBinsHigh[e]) * 0.5;
    double Y_um, Z_um;
    Y_um = -Rfit * px * sin(etaCenter * deg2rad);
    Z_um =  Rfit * px * cos(etaCenter * deg2rad);

    // DEBUG: print E-step peak fit details for ring 1
    if (r == 0 && (e % 30 == 0 || e == 0)) {
      double strain_dbg = fabs(1.0 - Rfit / IdealR);
      double Y_px = ybc + Rfit * sin(etaCenter * deg2rad);
      double Z_px = zbc + Rfit * cos(etaCenter * deg2rad);
      printf("  DEBUG E-step Ring1 eta=%7.1f: Rfit=%10.6f IdealR=%10.6f dR=%+.6f strain=%8.1f ue SNR=%6.1f Y_px=%9.3f Z_px=%9.3f nPts=%d\n",
             etaCenter, Rfit, IdealR, Rfit-IdealR, strain_dbg*1e6, snr, Y_px, Z_px, nPts);
    }

    // Store results
    RMean[binIdx] = Rfit * px; // microns
    EtaMean[binIdx] = etaCenter;
    IdealTtheta[binIdx] = 2.0 * Thetas[r];
    PointDSpacing[binIdx] = DSpacings[r];
    RingNumbers[binIdx] = RingIDs[r];
    FitSNR[binIdx] = snr;
    nValid++;

    free(Rs);
    free(PeakShape);
  }

  printf("E-step: %d valid bins from %d total\n", nValid, maxBins);

  // Compact: remove empty bins
  double *cRM = malloc(nValid * sizeof(double));
  double *cEM = malloc(nValid * sizeof(double));
  double *cYM = malloc(nValid * sizeof(double));
  double *cZM = malloc(nValid * sizeof(double));
  double *cIT = malloc(nValid * sizeof(double));
  double *cPD = malloc(nValid * sizeof(double));
  int *cRN = malloc(nValid * sizeof(int));
  double *cFS = malloc(nValid * sizeof(double));
  int *cSB = calloc(nValid, sizeof(int));

  // Build tilt matrix for the current E-step geometry (needed for inversion)
  double TRs_estep[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs_estep);
  double MaxRingRad_local = MaxRingRad; // local copy for dg_pixel_to_REta

  int cnt = 0;
  for (int i = 0; i < maxBins; i++) {
    if (RMean[i] != 0) {
      cRM[cnt] = RMean[i];
      cEM[cnt] = EtaMean[i];
      cIT[cnt] = IdealTtheta[i];
      cPD[cnt] = PointDSpacing[i];
      cRN[cnt] = RingNumbers[i];
      cFS[cnt] = FitSNR[i];
      // Convert (R, Eta) → raw pixel coords via numerical inversion
      // This correctly inverts the tilt/distortion correction,
      // unlike the old flat-detector polar formula.
      double R_px = cRM[cnt] / px;
      double Y_inv, Z_inv;
      dg_invert_REta_to_pixel(R_px, cEM[cnt],
                               ybc, zbc, TRs_estep, Lsd, MaxRingRad_local,
                               p0, p1, p2, p3, p4, p5,
                               px, 0, 0, parallax,
                               &Y_inv, &Z_inv);
      cYM[cnt] = Y_inv;
      cZM[cnt] = Z_inv;
      if (skipBinOut[i]) cSB[cnt] = 1;
      cnt++;
    }
  }

  // Cleanup map
  for (int r = 0; r < totalRBins; r++) {
    for (int e = 0; e < nEtaBins; e++)
      free(pxList[r][e]);
    free(pxList[r]);
    free(nPxList[r]);
    free(maxnPx[r]);
  }
  free(pxList); free(nPxList); free(maxnPx);
  free(binMaskFlag); free(profiles); free(norm);
  free(RBinsLow); free(RBinsHigh);
  free(EtaBinsLow); free(EtaBinsHigh);
  free(ringRLo); free(ringRHi);
  free(ringRBinStart); free(ringNRBins);
  free(distortionMapY); free(distortionMapZ);
  free(RMean); free(EtaMean);
  free(IdealTtheta); free(PointDSpacing);
  free(RingNumbers); free(FitSNR); free(skipBinOut);
  free(dbFlag); free(dbPair);
  for (int i = 0; i < MAPPER_N_BIN_LOCKS; i++)
    omp_destroy_lock(&binLocks[i]);

  *out_RMean = cRM;
  *out_EtaMean = cEM;
  *out_YMean = cYM;
  *out_ZMean = cZM;
  *out_IdealTtheta = cIT;
  *out_PointDSpacing = cPD;
  *out_RingNumbers = cRN;
  *out_FitSNR = cFS;
  *out_skipBin = cSB;
  return cnt;
}

// ── main ──────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  printf("CalibrantIntegratorOMP Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 3) {
    printf("Usage: CalibrantIntegratorOMP ps.txt nCPUs\n");
    return 1;
  }

  double start0 = omp_get_wtime();
  char *ParamFN = argv[1];
  numProcs = atoi(argv[2]);

  MIDASConfig cfg;
  if (parse_parameters(ParamFN, &cfg) != 0) return 1;

  // Unpack config
  int NrPixelsY = cfg.NrPixelsY, NrPixelsZ = cfg.NrPixelsZ;
  int NrPixels = cfg.NrPixels;
  double px = cfg.px, Lsd = cfg.Lsd, ybc = cfg.ybc, zbc = cfg.zbc;
  double tx = cfg.tx, tyin = cfg.ty, tzin = cfg.tz;
  double p0in = cfg.p0, p1in = cfg.p1, p2in = cfg.p2, p3in = cfg.p3;
  double p4in = cfg.p4, p5in = cfg.p5;
  double MaxRingRad = cfg.RhoD, Wavelength = cfg.Wavelength;
  double EtaBinSize = cfg.EtaBinSize;
  int RBinWidth = cfg.RBinWidth;
  double tolTilts = cfg.tolTilts, tolLsd = cfg.tolLsd, tolBC = cfg.tolBC;
  double tolP0 = cfg.tolP0, tolP1 = cfg.tolP1, tolP2 = cfg.tolP2;
  double tolP3 = cfg.tolP3, tolP4 = cfg.tolP4, tolP5 = cfg.tolP5;
  double tolShifts = cfg.tolShifts, tolRotation = cfg.tolRotation;
  double outlierFactor = cfg.outlierFactor;
  int MinIndicesForFit = cfg.MinIndicesForFit, FixPanelID = cfg.FixPanelID;
  int nIterations = cfg.nIterations;
  double DoubletSeparation = cfg.DoubletSeparation;
  int NormalizeRingWeights = cfg.NormalizeRingWeights;
  int OutlierIterations = cfg.OutlierIterations;
  int RemoveOutliersBetweenIters = cfg.RemoveOutliersBetweenIters;
  int ReFitPeaks = cfg.ReFitPeaks;
  double TrimmedMeanFraction = cfg.TrimmedMeanFraction;
  int WeightByRadius = cfg.WeightByRadius, WeightByFitSNR = cfg.WeightByFitSNR;
  int L2Objective = cfg.L2Objective;
  int PerPanelLsd = cfg.PerPanelLsd, PerPanelDistortion = cfg.PerPanelDistortion;
  double tolLsdPanel = cfg.tolLsdPanel, tolP2Panel = cfg.tolP2Panel;
  int FitWavelength = cfg.FitWavelength;
  double tolWavelength = cfg.tolWavelength;
  int FitParallax = cfg.FitParallax;
  double parallaxIn = cfg.parallaxIn, tolParallax = cfg.tolParallax;
  int NrTransOpt = cfg.NrTransOpt;
  int TransOpt[10];
  memcpy(TransOpt, cfg.TransOpt, sizeof(TransOpt));
  int nRingsExclude = cfg.nRingsExclude, RingsExclude[50];
  memcpy(RingsExclude, cfg.RingsExclude, sizeof(RingsExclude));
  int MaxRingNumber = cfg.MaxRingNumber;
  int dType = cfg.DataType, HeadSize = cfg.HeadSize, Padding = cfg.Padding;
  long long int GapIntensity = cfg.GapIntensity, BadPxIntensity = cfg.BadPxIntensity;
  int NPanelsY = cfg.NPanelsY, NPanelsZ = cfg.NPanelsZ;
  double LatticeConstant[6];
  memcpy(LatticeConstant, cfg.LatticeConstant, sizeof(LatticeConstant));
  double IdealA = LatticeConstant[0]; // ideal lattice parameter 'a'
  char PanelShiftsFile[1024];
  strcpy(PanelShiftsFile, cfg.PanelShiftsFile);
  int peakFitMode = cfg.PeakFitMode;

  printf("\n=== CalibrantIntegratorOMP ===\n");
  printf("Detector: %dx%d, px=%.4f, Lsd=%.2f\n", NrPixelsY, NrPixelsZ, px, Lsd);
  printf("Iterations: %d, RBinWidth: %d, EtaBinSize: %.2f\n",
         nIterations, RBinWidth, EtaBinSize);

  // Generate HKL list
  double Thetas[100], DSpacings[100];
  int RingIDs[100];
  int n_hkls = 0;
  double MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);

  run_midas_binary("GetHKLList", ParamFN);
  FILE *hklf = fopen("hkls.csv", "r");
  if (!hklf) { fprintf(stderr, "Cannot open hkls.csv\n"); return 1; }
  char aline[1000], dummy[1000];
  fgets(aline, 1000, hklf);
  int LastRingDone = 0;
  while (fgets(aline, 1000, hklf) != NULL) {
    int tRnr;
    double theta;
    sscanf(aline, "%s %s %s %s %d %s %s %s %lf %s %s",
           dummy, dummy, dummy, dummy, &tRnr, dummy, dummy, dummy,
           &theta, dummy, dummy);
    if (theta * 2 > MaxTtheta) break;
    if (MaxRingNumber > 0 && tRnr > MaxRingNumber) break;
    int exclude = 0;
    for (int i = 0; i < nRingsExclude; i++)
      if (tRnr == RingsExclude[i]) exclude = 1;
    if (!exclude && tRnr > LastRingDone) {
      Thetas[n_hkls] = theta;
      DSpacings[n_hkls] = Wavelength / (2.0 * sin(theta * deg2rad));
      RingIDs[n_hkls] = tRnr;
      LastRingDone = tRnr;
      printf("  Ring %d: 2θ=%.4f° R=%.1f px\n", tRnr, 2*theta,
             Lsd * tan(deg2rad * 2 * theta) / px);
      n_hkls++;
    }
  }
  fclose(hklf);
  printf("%d rings selected.\n\n", n_hkls);

  // Load dark + image in ORIGINAL dimensions (NrPixelsY × NrPixelsZ).
  // mapper_build_map works on the original pixel grid and handles transforms
  // internally via inverse_transform_pixel, so no squaring/transforming here.
  size_t nOrigPx = (size_t)NrPixelsY * NrPixelsZ;
  pixelvalue *DarkRaw = malloc(nOrigPx * sizeof(*DarkRaw));
  double *AverageDark = calloc(nOrigPx, sizeof(double));
  double *Average = calloc(nOrigPx, sizeof(double));

  // Read dark
  FILE *fd = fopen(cfg.Dark, "rb");
  if (fd == NULL && dType != 8) {
    printf("Dark file not found — using zeros.\n");
  } else {
    int rc;
    if (dType == 8) {
      rc = SumHDF5Frames(cfg.Dark, cfg.darkDatasetName,
                         NrPixelsY * NrPixelsZ, DarkRaw, skipFrame);
      for (size_t j = 0; j < nOrigPx; j++)
        AverageDark[j] = DarkRaw[j];
    } else {
      fseek(fd, 0L, SEEK_END);
      size_t sz = ftell(fd) - HeadSize;
      rewind(fd);
      size_t pxSz = sizeof(double);
      if (dType == 1) pxSz = sizeof(uint16_t);
      else if (dType == 3) pxSz = sizeof(float);
      else if (dType == 4 || dType == 6) pxSz = sizeof(uint32_t);
      int nFrames = sz / (pxSz * NrPixelsY * NrPixelsZ);
      if (nFrames == 0) nFrames = 1;
      fseek(fd, HeadSize, SEEK_SET);
      for (int i = 0; i < nFrames; i++) {
        if (dType == 6 || dType == 7 || dType == 9)
          rc = ReadTiffFrame(cfg.Dark, dType, NrPixelsY * NrPixelsZ, DarkRaw, i);
        else
          rc = ReadBinaryFrame(fd, dType, NrPixelsY * NrPixelsZ, DarkRaw);
        for (size_t j = 0; j < nOrigPx; j++)
          AverageDark[j] += DarkRaw[j];
      }
      for (size_t j = 0; j < nOrigPx; j++)
        AverageDark[j] /= nFrames;
      fclose(fd);
    }
  }

  // Build mask in ORIGINAL dimensions
  // Only apply sentinel masking when GapIntensity/BadPxIntensity are explicitly
  // set (non-zero), since 0 is a valid pixel value and the default.
  double *mask = calloc(nOrigPx, sizeof(double));
  if (GapIntensity != 0 || BadPxIntensity != 0) {
    for (size_t j = 0; j < nOrigPx; j++) {
      if ((GapIntensity != 0 && AverageDark[j] == (double)GapIntensity) ||
          (BadPxIntensity != 0 && AverageDark[j] == (double)BadPxIntensity))
        mask[j] = 1.0;
    }
  }
  if (cfg.MaskFN[0] != '\0') {
    double *maskRaw = calloc(nOrigPx, sizeof(double));
    ReadTiffFrame(cfg.MaskFN, 7, NrPixelsY * NrPixelsZ, maskRaw, 0);
    for (size_t j = 0; j < nOrigPx; j++)
      if (maskRaw[j] == 1.0) mask[j] = 1.0;
    free(maskRaw);
    printf("Mask file loaded: %s\n", cfg.MaskFN);
  }

  // Read and average data frames (original dimensions)
  pixelvalue *ImageRaw = malloc(nOrigPx * sizeof(*ImageRaw));
  int TotFrames = 0;
  for (int fnr = cfg.StartNr; fnr <= cfg.EndNr; fnr++) {
    char FileName[4096];
    snprintf(FileName, sizeof(FileName), "%s/%s_%0*d%s",
             cfg.Folder, cfg.FileStem, Padding, fnr, cfg.Ext);
    int rc;
    if (dType == 8) {
      rc = SumHDF5Frames(FileName, cfg.dataDatasetName,
                         NrPixelsY * NrPixelsZ, ImageRaw, skipFrame);
    } else if (dType == 6 || dType == 7 || dType == 9) {
      rc = ReadTiffFrame(FileName, dType, NrPixelsY * NrPixelsZ, ImageRaw, 0);
    } else {
      FILE *fp = fopen(FileName, "rb");
      if (!fp) { printf("Cannot open %s\n", FileName); continue; }
      fseek(fp, HeadSize, SEEK_SET);
      rc = ReadBinaryFrame(fp, dType, NrPixelsY * NrPixelsZ, ImageRaw);
      fclose(fp);
    }
    for (size_t j = 0; j < nOrigPx; j++)
      Average[j] += ImageRaw[j];
    TotFrames++;
  }
  if (TotFrames > 1)
    for (size_t j = 0; j < nOrigPx; j++)
      Average[j] /= TotFrames;
  printf("Read %d frames.\n\n", TotFrames);

  free(DarkRaw); free(ImageRaw);

  // ── Iteration loop ────────────────────────────────────────────────
  double LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3;
  double MeanDiff = 1e20, StdDiff = 0;
  int nIndices = 0;
  double *RMean = NULL, *EtaMean = NULL, *YMean = NULL, *ZMean = NULL;
  double *IdealTtheta = NULL, *PointDSpacing = NULL, *FitSNR = NULL;
  int *RingNumbers = NULL, *skipBin = NULL;
  double *EtaIns = NULL, *DiffIns = NULL, *RadIns = NULL;
  double *Yc = NULL, *Zc = NULL;

  // Best-iteration tracking
  double bestMeanDiff = 1e20;
  int bestIter = -1;
  double bestLsd, bestYbc, bestZbc, bestTy, bestTz;
  double bestP0, bestP1, bestP2, bestP3, bestP4, bestP5, bestParallax;
  Panel *bestPanels = (nPanels > 0) ? malloc(nPanels * sizeof(Panel)) : NULL;

  // Guard state
  double prevIterMeanDiff = 1e20, prevPrevIterMeanDiff = 1e20;
  int stagnantCount = 0, oscillationCount = 0;
  int postPerturbGrace = 0;

  // Initial parameters (for bound anchoring)
  double initParams[12];
  initParams[0] = Lsd;   initParams[1] = ybc;  initParams[2] = zbc;
  initParams[3] = tyin;  initParams[4] = tzin;
  initParams[5] = p0in;  initParams[6] = p1in; initParams[7] = p2in;
  initParams[8] = p3in;  initParams[9] = p4in; initParams[10] = p5in;
  initParams[11] = parallaxIn;
  Panel *initPanels = NULL;
  if (nPanels > 0) {
    initPanels = malloc(nPanels * sizeof(Panel));
    memcpy(initPanels, panels, nPanels * sizeof(Panel));
  }

  // Build rawFN for output file naming: FileStem_NNNNNN.ext
  char rawFN[4096];
  snprintf(rawFN, sizeof(rawFN), "%s_%0*d.%s",
           cfg.FileStem, Padding, cfg.StartNr, cfg.Ext);

  // Convergence CSV (in cwd)
  char convHistFN[4096];
  snprintf(convHistFN, sizeof(convHistFN), "%s.convergence_history.csv", rawFN);
  FILE *convHistFP = fopen(convHistFN, "w");
  if (convHistFP)
    fprintf(convHistFP, "Iter,MeanStrain_ppm,StdStrain_ppm,Lsd,ybc,zbc,"
                        "ty,tz,p0,p1,p2,p3,p4,p5\n");

  // nIterations=0: evaluate-only mode — run E-step once, skip optimization
  if (nIterations == 0) {
    printf("\n*** nIterations=0: evaluate-only mode ***\n");
    nIndices = run_estep(
        Average, AverageDark, NrPixelsY, NrPixelsZ, NrPixels,
        px, Lsd, ybc, zbc, tx, tyin, tzin,
        p0in, p1in, p2in, p3in, p4in, p5in,
        MaxRingRad, EtaBinSize, RBinWidth,
        parallaxIn, n_hkls, Thetas, DSpacings, RingIDs,
        NrTransOpt, TransOpt, mask, DoubletSeparation, Wavelength,
        cfg.Width, peakFitMode,
        &RMean, &EtaMean, &YMean, &ZMean,
        &IdealTtheta, &PointDSpacing, &RingNumbers, &FitSNR, &skipBin);
    if (nIndices >= 3) {
      Yc = malloc(nIndices * sizeof(double));
      Zc = malloc(nIndices * sizeof(double));
      EtaIns = malloc(nIndices * sizeof(double));
      DiffIns = malloc(nIndices * sizeof(double));
      RadIns = malloc(nIndices * sizeof(double));
      memcpy(Yc, YMean, nIndices * sizeof(double));
      memcpy(Zc, ZMean, nIndices * sizeof(double));
      // Set Fit params to input params (no optimization)
      LsdFit = Lsd; ybcFit = ybc; zbcFit = zbc;
      ty = tyin; tz = tzin;
      p0 = p0in; p1 = p1in; p2 = p2in; p3 = p3in;
    } else {
      fprintf(stderr, "E-step produced only %d bins.\n", nIndices);
    }
  }

  for (int iter = 0; iter < nIterations; iter++) {
    // Always rebuild the map with current best geometry for true E-M convergence.
    // Previously, ReFitPeaks=0 caused the map to be built only at iter==0,
    // so the M-step optimized geometry against stale pixel positions.
    int needRebuild = 1;
    printf("\n──── Iteration %d/%d ────\n", iter + 1, nIterations);

    if (needRebuild) {
      // Free previous E-step arrays
      free(RMean); free(EtaMean); free(YMean); free(ZMean);
      free(IdealTtheta); free(PointDSpacing); free(FitSNR);
      free(RingNumbers); free(skipBin);
      free(EtaIns); free(DiffIns); free(RadIns);
      free(Yc); free(Zc);
      RMean = EtaMean = YMean = ZMean = NULL;
      IdealTtheta = PointDSpacing = FitSNR = NULL;
      EtaIns = DiffIns = RadIns = Yc = Zc = NULL;
      RingNumbers = NULL; skipBin = NULL;

      // For iter > 0, use the M-step's converged geometry to build the map.
      // For iter == 0, use the original input params.
      double estep_Lsd = (iter > 0) ? LsdFit : Lsd;
      double estep_ybc = (iter > 0) ? ybcFit : ybc;
      double estep_zbc = (iter > 0) ? zbcFit : zbc;
      double estep_ty  = (iter > 0) ? ty     : tyin;
      double estep_tz  = (iter > 0) ? tz     : tzin;
      double estep_p0  = (iter > 0) ? p0     : p0in;
      double estep_p1  = (iter > 0) ? p1     : p1in;
      double estep_p2  = (iter > 0) ? p2     : p2in;
      double estep_p3  = (iter > 0) ? p3     : p3in;
      // p4in, p5in, parallaxIn are updated in place by the M-step

      // Run E-step with current best geometry
      nIndices = run_estep(
          Average, AverageDark, NrPixelsY, NrPixelsZ, NrPixels,
          px, estep_Lsd, estep_ybc, estep_zbc, tx, estep_ty, estep_tz,
          estep_p0, estep_p1, estep_p2, estep_p3, p4in, p5in,
          MaxRingRad, EtaBinSize, RBinWidth,
          parallaxIn, n_hkls, Thetas, DSpacings, RingIDs,
          NrTransOpt, TransOpt, mask, DoubletSeparation, Wavelength,
          cfg.Width, peakFitMode,
          &RMean, &EtaMean, &YMean, &ZMean,
          &IdealTtheta, &PointDSpacing, &RingNumbers, &FitSNR, &skipBin);

      if (nIndices < 3) {
        fprintf(stderr, "E-step produced only %d bins — aborting.\n", nIndices);
        break;
      }

      // Allocate per-bin output arrays
      Yc = malloc(nIndices * sizeof(double));
      Zc = malloc(nIndices * sizeof(double));
      EtaIns = malloc(nIndices * sizeof(double));
      DiffIns = malloc(nIndices * sizeof(double));
      RadIns = malloc(nIndices * sizeof(double));
      memcpy(Yc, YMean, nIndices * sizeof(double));
      memcpy(Zc, ZMean, nIndices * sizeof(double));
    }

    // Ring normalization weights
    double *RingWeights = NULL;
    if (NormalizeRingWeights) {
      RingWeights = calloc(nIndices, sizeof(double));
      int ringCounts[100] = {0};
      for (int i = 0; i < nIndices; i++) {
        for (int r = 0; r < n_hkls; r++) {
          if (RingNumbers[i] == RingIDs[r]) { ringCounts[r]++; break; }
        }
      }
      for (int i = 0; i < nIndices; i++) {
        for (int r = 0; r < n_hkls; r++) {
          if (RingNumbers[i] == RingIDs[r] && ringCounts[r] > 0) {
            RingWeights[i] = 1.0 / ringCounts[r];
            break;
          }
        }
      }
    }

    // SNR weights
    double *snrWeights = NULL;
    if (WeightByFitSNR && FitSNR != NULL) {
      snrWeights = malloc(nIndices * sizeof(double));
      double *snrSorted = malloc(nIndices * sizeof(double));
      memcpy(snrSorted, FitSNR, nIndices * sizeof(double));
      qsort(snrSorted, nIndices, sizeof(double), calib_cmp_double);
      double medSNR = snrSorted[nIndices / 2];
      if (medSNR < 1e-12) medSNR = 1.0;
      free(snrSorted);
      for (int i = 0; i < nIndices; i++) {
        double w = FitSNR[i] / medSNR;
        snrWeights[i] = (w < 1.0) ? w : 1.0;
      }
    }

    // M-step: optimize geometry
    double p4 = p4in, p5 = p5in;
    double wavelengthFit = Wavelength, parallaxFit = parallaxIn;

    // DEBUG M-step: print input params
    printf("  DEBUG M-step iter %d INPUT:  Lsd=%.3f BC=(%.6f,%.6f) ty=%.6f tz=%.6f p0=%.2e p1=%.2e p2=%.2e p3=%.2e\n",
           iter+1, Lsd, ybc, zbc, tyin, tzin, p0in, p1in, p2in, p3in);

    calib_fit_tilt_bc_lsd(
        nIndices, Yc, Zc, IdealTtheta, Lsd, MaxRingRad, ybc, zbc, tx,
        tyin, tzin, p0in, p1in, p2in, p3in, &ty, &tz, &LsdFit,
        &ybcFit, &zbcFit, &p0, &p1, &p2, &p3, &MeanDiff, tolTilts,
        tolLsd, tolBC, 0, tolP0, tolP1, tolP2, tolP3, tolShifts,
        tolRotation, px, outlierFactor, MinIndicesForFit, FixPanelID,
        RingWeights, p4in, tolP4, PerPanelLsd, tolLsdPanel,
        PerPanelDistortion, tolP2Panel, WeightByRadius, snrWeights,
        &p4, p5in, tolP5, &p5,
        iter == 0, L2Objective, initParams, initPanels,
        FitWavelength, Wavelength, tolWavelength, PointDSpacing,
        &wavelengthFit,
        parallaxIn, tolParallax, &parallaxFit,
        TrimmedMeanFraction, skipBin);

    // DEBUG M-step: print output params
    printf("  DEBUG M-step iter %d OUTPUT: Lsd=%.3f BC=(%.6f,%.6f) ty=%.6f tz=%.6f p0=%.2e p1=%.2e p2=%.2e p3=%.2e MeanDiff=%.6f\n",
           iter+1, LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3, MeanDiff*1e6);

    if (FitParallax) parallaxIn = parallaxFit;
    if (FitWavelength) Wavelength = wavelengthFit;

    // Evaluate residuals
    int *iterOutlier = NULL;
    if (RemoveOutliersBetweenIters && outlierFactor > 0 && iter < nIterations - 1)
      iterOutlier = calloc(nIndices, sizeof(int));
    calib_correct_tilt_distortion(
        nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
        zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
        &StdDiff, outlierFactor, iterOutlier, p4, p5, OutlierIterations,
        0, &MeanDiff, parallaxIn, skipBin);

    printf("Iter %2d/%d  MeanStrain %8.3f  StdStrain %8.3f  nBins=%d\n",
           iter + 1, nIterations, MeanDiff * 1e6, StdDiff * 1e6, nIndices);

    // Per-ring diagnostic summary
    if (iter == 0 || iter == nIterations - 1) {
      printf("  Ring  NPoints   Mean(ΔR) µε  Med(ΔR) µε  Mean|ΔR| µε  Med|ΔR| µε  MeanSNR\n");
      printf("  ----  -------   -----------  ----------  -----------  ----------  -------\n");
      for (int ri = 0; ri < n_hkls; ri++) {
        // Collect per-ring deltaR values
        int cnt = 0;
        for (int bi = 0; bi < nIndices; bi++)
          if (RingNumbers[bi] == RingIDs[ri] && !skipBin[bi]) cnt++;
        if (cnt == 0) continue;
        double *drs = (double *)malloc(cnt * sizeof(double));
        double *adrs = (double *)malloc(cnt * sizeof(double));
        double sumSNR = 0;
        int k = 0;
        for (int bi = 0; bi < nIndices; bi++) {
          if (RingNumbers[bi] == RingIDs[ri] && !skipBin[bi]) {
            drs[k] = DiffIns[bi];
            adrs[k] = fabs(DiffIns[bi]);
            sumSNR += FitSNR[bi];
            k++;
          }
        }
        // Mean
        double meanDR = 0, meanADR = 0;
        for (int j = 0; j < cnt; j++) { meanDR += drs[j]; meanADR += adrs[j]; }
        meanDR /= cnt; meanADR /= cnt;
        // Median (sort copies)
        qsort(drs, cnt, sizeof(double), calib_cmp_double);
        qsort(adrs, cnt, sizeof(double), calib_cmp_double);
        double medDR = (cnt % 2) ? drs[cnt/2] : (drs[cnt/2-1] + drs[cnt/2]) / 2;
        double medADR = (cnt % 2) ? adrs[cnt/2] : (adrs[cnt/2-1] + adrs[cnt/2]) / 2;
        printf("  %4d  %7d   %11.3f  %10.3f  %11.3f  %10.3f  %7.1f\n",
               RingIDs[ri], cnt,
               meanDR * 1e6, medDR * 1e6,
               meanADR * 1e6, medADR * 1e6,
               sumSNR / cnt);
        free(drs); free(adrs);
      }
    }

    if (convHistFP) {
      fprintf(convHistFP,
              "%d,%.6e,%.6e,%.6f,%.6f,%.6f,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
              iter + 1, MeanDiff * 1e6, StdDiff * 1e6,
              LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3, p4, p5);
      fflush(convHistFP);
    }

    // Outlier removal between iterations
    if (iterOutlier) {
      int nNew = 0;
      for (int i = 0; i < nIndices; i++) {
        if (iterOutlier[i] && !skipBin[i]) { skipBin[i] = 1; nNew++; }
      }
      if (nNew > 0)
        printf("  Removed %d outliers between iterations.\n", nNew);
      free(iterOutlier);
    }

    // Feed outputs back as inputs
    p4in = p4; Lsd = LsdFit; ybc = ybcFit; zbc = zbcFit;
    tyin = ty; tzin = tz;
    p0in = p0; p1in = p1; p2in = p2; p3in = p3;

    // Track best iteration
    if (MeanDiff < bestMeanDiff) {
      bestMeanDiff = MeanDiff; bestIter = iter;
      bestLsd = LsdFit; bestYbc = ybcFit; bestZbc = zbcFit;
      bestTy = ty; bestTz = tz;
      bestP0 = p0; bestP1 = p1; bestP2 = p2; bestP3 = p3;
      bestP4 = p4; bestP5 = p5; bestParallax = parallaxIn;
      if (bestPanels && nPanels > 0)
        memcpy(bestPanels, panels, nPanels * sizeof(Panel));
    }

    // Guards
    if (iter > 0 && fabs(MeanDiff - prevIterMeanDiff) < 1e-12)
      stagnantCount++;
    else
      stagnantCount = 0;
    if (iter > 1 && fabs(MeanDiff - prevPrevIterMeanDiff) < 1e-9)
      oscillationCount++;
    else
      oscillationCount = 0;
    prevPrevIterMeanDiff = prevIterMeanDiff;
    prevIterMeanDiff = MeanDiff;

    // Divergence guard
    if (postPerturbGrace > 0) {
      postPerturbGrace--;
    } else if (iter > 0 && bestMeanDiff > 0 && MeanDiff > 1.5 * bestMeanDiff) {
      printf("  [Divergence guard] Reverting to best iter %d\n", bestIter + 1);
      LsdFit = bestLsd; ybcFit = bestYbc; zbcFit = bestZbc;
      ty = bestTy; tz = bestTz;
      p0 = bestP0; p1 = bestP1; p2 = bestP2; p3 = bestP3;
      p4in = bestP4; p5in = bestP5; parallaxIn = bestParallax;
      Lsd = bestLsd; ybc = bestYbc; zbc = bestZbc;
      tyin = bestTy; tzin = bestTz;
      p0in = bestP0; p1in = bestP1; p2in = bestP2; p3in = bestP3;
      MeanDiff = bestMeanDiff;
      if (bestPanels && nPanels > 0)
        memcpy(panels, bestPanels, nPanels * sizeof(Panel));
      break;
    }

    // Stagnation perturbation
    if (stagnantCount >= 5 && iter < nIterations - 1) {
      printf("  [Stagnation] Perturbing from best iter %d\n", bestIter + 1);
      Lsd = bestLsd; ybc = bestYbc; zbc = bestZbc;
      tyin = bestTy; tzin = bestTz;
      p0in = bestP0; p1in = bestP1; p2in = bestP2; p3in = bestP3;
      p4in = bestP4; p5in = bestP5; parallaxIn = bestParallax;
      if (bestPanels && nPanels > 0)
        memcpy(panels, bestPanels, nPanels * sizeof(Panel));
      unsigned long long lcg = 42ULL + iter;
#define LCG_N(s) ((s)=(s)*6364136223846793005ULL+1442695040888963407ULL)
#define LCG_D(s) ((double)(LCG_N(s)>>33)/(double)(1ULL<<31))
#define PERT(v,t,lo,hi) fmin(fmax((v)+0.05*(t)*(2.0*LCG_D(lcg)-1.0),(lo)),(hi))
      Lsd = PERT(Lsd,tolLsd,initParams[0]-tolLsd,initParams[0]+tolLsd);
      ybc = PERT(ybc,tolBC,initParams[1]-tolBC,initParams[1]+tolBC);
      zbc = PERT(zbc,tolBC,initParams[2]-tolBC,initParams[2]+tolBC);
      tyin = PERT(tyin,tolTilts,initParams[3]-tolTilts,initParams[3]+tolTilts);
      tzin = PERT(tzin,tolTilts,initParams[4]-tolTilts,initParams[4]+tolTilts);
#undef PERT
#undef LCG_N
#undef LCG_D
      stagnantCount = 0;
      postPerturbGrace = 3;
    }

    // Oscillation perturbation (same pattern)
    if (oscillationCount >= 3 && iter < nIterations - 1) {
      printf("  [Oscillation] Perturbing from best iter %d\n", bestIter + 1);
      Lsd = bestLsd; ybc = bestYbc; zbc = bestZbc;
      tyin = bestTy; tzin = bestTz;
      p0in = bestP0; p1in = bestP1; p2in = bestP2; p3in = bestP3;
      p4in = bestP4; p5in = bestP5; parallaxIn = bestParallax;
      if (bestPanels && nPanels > 0)
        memcpy(panels, bestPanels, nPanels * sizeof(Panel));
      unsigned long long lcg = 137ULL + iter;
#define LCG_N(s) ((s)=(s)*6364136223846793005ULL+1442695040888963407ULL)
#define LCG_D(s) ((double)(LCG_N(s)>>33)/(double)(1ULL<<31))
#define PERT(v,t,lo,hi) fmin(fmax((v)+0.08*(t)*(2.0*LCG_D(lcg)-1.0),(lo)),(hi))
      Lsd = PERT(Lsd,tolLsd,initParams[0]-tolLsd,initParams[0]+tolLsd);
      ybc = PERT(ybc,tolBC,initParams[1]-tolBC,initParams[1]+tolBC);
      zbc = PERT(zbc,tolBC,initParams[2]-tolBC,initParams[2]+tolBC);
#undef PERT
#undef LCG_N
#undef LCG_D
      oscillationCount = 0;
      postPerturbGrace = 3;
    }

    free(RingWeights);
    free(snrWeights);
  } // end iteration loop

  // Restore best iteration
  if (nIterations > 1 && bestIter >= 0 && bestIter != nIterations - 1) {
    printf("\n*** Restoring best iter %d (MeanStrain %.3f) ***\n",
           bestIter + 1, bestMeanDiff * 1e6);
    LsdFit = bestLsd; ybcFit = bestYbc; zbcFit = bestZbc;
    ty = bestTy; tz = bestTz;
    p0 = bestP0; p1 = bestP1; p2 = bestP2; p3 = bestP3;
    p4in = bestP4; p5in = bestP5; parallaxIn = bestParallax;
    MeanDiff = bestMeanDiff;
    if (bestPanels && nPanels > 0)
      memcpy(panels, bestPanels, nPanels * sizeof(Panel));
  }

  // ── Verification E-step: rebuild map with converged params ──
  // The iteration loop builds maps with the ORIGINAL input tilts,
  // and the M-step only re-projects the fixed pixel positions.
  // This final E-step rebuilds everything with the converged geometry
  // to verify the reported strain is consistent.
  if (nIterations > 0 && Yc) {
    printf("\n──── Verification E-step (fresh map with converged params) ────\n");
    // Free previous E-step arrays
    free(RMean); free(EtaMean); free(YMean); free(ZMean);
    free(IdealTtheta); free(PointDSpacing); free(FitSNR);
    free(RingNumbers); free(skipBin);
    free(EtaIns); free(DiffIns); free(RadIns);
    free(Yc); free(Zc);
    RMean = EtaMean = YMean = ZMean = NULL;
    IdealTtheta = PointDSpacing = FitSNR = NULL;
    EtaIns = DiffIns = RadIns = Yc = Zc = NULL;
    RingNumbers = NULL; skipBin = NULL;

    // Run E-step with CONVERGED geometry
    int verifyN = run_estep(
        Average, AverageDark, NrPixelsY, NrPixelsZ, NrPixels,
        px, LsdFit, ybcFit, zbcFit, tx, ty, tz,
        p0, p1, p2, p3, p4in, p5in,
        MaxRingRad, EtaBinSize, RBinWidth,
        parallaxIn, n_hkls, Thetas, DSpacings, RingIDs,
        NrTransOpt, TransOpt, mask, DoubletSeparation, Wavelength,
        cfg.Width, peakFitMode,
        &RMean, &EtaMean, &YMean, &ZMean,
        &IdealTtheta, &PointDSpacing, &RingNumbers, &FitSNR, &skipBin);

    if (verifyN >= 3) {
      nIndices = verifyN;
      Yc = malloc(nIndices * sizeof(double));
      Zc = malloc(nIndices * sizeof(double));
      EtaIns = malloc(nIndices * sizeof(double));
      DiffIns = malloc(nIndices * sizeof(double));
      RadIns = malloc(nIndices * sizeof(double));
      memcpy(Yc, YMean, nIndices * sizeof(double));
      memcpy(Zc, ZMean, nIndices * sizeof(double));

      // Evaluate strain at converged params (no optimization)
      double verifyMean = 0, verifyStd = 0;
      int *verifyOutlier = calloc(nIndices, sizeof(int));
      calib_correct_tilt_distortion(
          nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
          zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
          &verifyStd, outlierFactor, verifyOutlier, p4in, p5in, OutlierIterations,
          1, &verifyMean, parallaxIn, skipBin);
      printf("  Verification E-step: %d bins, MeanStrain=%.6f µε, StdStrain=%.6f µε\n",
             nIndices, verifyMean * 1e6, verifyStd * 1e6);
      printf("  (Compare to M-step reported: MeanStrain=%.6f µε)\n",
             MeanDiff * 1e6);
      // Update to verification values
      MeanDiff = verifyMean;
      StdDiff = verifyStd;
      free(verifyOutlier);
    } else {
      printf("  Verification E-step: only %d bins (too few)\n", verifyN);
    }
  }

  if (Yc && Zc && IdealTtheta) {
    int *IsOutlier = calloc(nIndices, sizeof(int));
    calib_correct_tilt_distortion(
        nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
        zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
        &StdDiff, outlierFactor, IsOutlier, p4in, p5in, OutlierIterations,
        1, &MeanDiff, parallaxIn, skipBin);
    if (skipBin) {
      for (int i = 0; i < nIndices; i++)
        if (skipBin[i]) IsOutlier[i] = 1;
    }
    printf("\nMean Values\n");
    printf("Lsd %f\n", LsdFit);
    printf("BC %f %f\n", ybcFit, zbcFit);
    printf("ty %f\n", ty);
    printf("tz %f\n", tz);
    printf("p0 %e\n", p0);
    printf("p1 %e\n", p1);
    printf("p2 %e\n", p2);
    printf("p3 %e\n", p3);
    if (p4in != 0.0) printf("p4 %e\n", p4in);
    if (p5in != 0.0) printf("p5 %e\n", p5in);
    printf("RhoD %f\n", MaxRingRad);
    printf("MeanStrain %f\n", MeanDiff * 1e6);
    printf("StdStrain %f\n", StdDiff * 1e6);

    printf("\nFinal MeanStrain: %.6f µε  StdStrain: %.6f µε\n",
           MeanDiff * 1e6, StdDiff * 1e6);

    // Write corr.csv (in cwd, named after input file)
    char corrFN[4096];
    snprintf(corrFN, sizeof(corrFN), "%s.corr.csv", rawFN);
    FILE *corrFP = fopen(corrFN, "w");
    if (corrFP) {
      fprintf(corrFP, "Lsd,ybcFit,zbcFit,ty,tz,p0,p1,p2,p3,MeanStrain,StdStrain,"
                       "tx,p4,p5,Wavelength,Parallax\n");
      fprintf(corrFP, "%.10f,%.10f,%.10f,%.10f,%.10f,%.12e,%.12e,%.12e,%.12e,"
                       "%.12e,%.12e,%.10f,%.12e,%.12e,%.10f,%.10f\n",
              LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3,
              MeanDiff, StdDiff, tx, p4in, p5in, Wavelength, parallaxIn);
      // Per-bin data — 16 columns matching CalibrantPanelShiftsOMP format
      fprintf(corrFP, "\n%%Eta Strain RadFit EtaCalc DiffCalc RadCalc "
                       "Ideal2Theta Outlier YRawCorr ZRawCorr RingNr "
                       "RadGlobal IdealR Fit2Theta IdealA FitA\n");
      for (int i = 0; i < nIndices; i++) {
        double eta = EtaIns[i];
        double strain = DiffIns[i];  // signed: (1 - R_fitted/R_ideal)
        double radFit = RadIns[i];
        double ideal2theta = IdealTtheta[i];
        double idealR = LsdFit * tan(deg2rad * ideal2theta);
        double radCalc = idealR;
        double fit2theta = rad2deg * atan(radFit / LsdFit);
        double sinIdeal = sin(deg2rad * ideal2theta / 2.0);
        double sinFit = sin(deg2rad * fit2theta / 2.0);
        double fitA = (sinFit > 1e-15) ? IdealA * sinIdeal / sinFit : IdealA;
        fprintf(corrFP, "%.6f %.10e %.6f %.6f %.10e %.6f "
                         "%.6f %d %.6f %.6f %d "
                         "%.6f %.6f %.6f %.10f %.10f\n",
                eta, strain, radFit, eta, fabs(strain), radCalc,
                ideal2theta, IsOutlier[i], Yc[i], Zc[i], RingNumbers[i],
                radFit, idealR, fit2theta, IdealA, fitA);
      }
      fclose(corrFP);
      printf("Results written to: %s\n", corrFN);
    }

    // Panel shifts output
    if (nPanels > 0 && PanelShiftsFile[0] != '\0') {
      SavePanelShifts(PanelShiftsFile, nPanels, panels);
      printf("Panel shifts saved to: %s\n", PanelShiftsFile);
    }
    free(IsOutlier);
  }

  if (convHistFP) {
    fclose(convHistFP);
    printf("Convergence history: %s\n", convHistFN);
  }

  // Cleanup
  free(RMean); free(EtaMean); free(YMean); free(ZMean);
  free(IdealTtheta); free(PointDSpacing); free(FitSNR);
  free(RingNumbers); free(skipBin);
  free(EtaIns); free(DiffIns); free(RadIns);
  free(Yc); free(Zc);
  free(Average); free(AverageDark);
  free(mask);
  free(bestPanels); free(initPanels);

  double end0 = omp_get_wtime();
  printf("\nTotal time: %.2f s\n", end0 - start0);
  return 0;
}
