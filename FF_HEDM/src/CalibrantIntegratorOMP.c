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
#include "CalibrantEstep.h"
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

// Calibration context — replaces global state
static CalibContext ctx = { .panels = NULL, .nPanels = 0, .numProcs = 1, .NrCalls = 0 };
// Convenience aliases (point into ctx)
#define panels    ctx.panels
#define nPanels   ctx.nPanels
#define numProcs  ctx.numProcs

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

// ── E-step sub-functions ──────────────────────────────────────────
//
// The E-step pipeline: build_ring_bins → build_and_integrate →
// fit_peaks → compact_and_invert.  Each sub-function has clear
// input/output, owns its local temporaries, and returns 0 on success.

// ── Free functions for E-step structs ─────────────────────────────

void free_bin_edges(BinEdges *b) {
  if (!b) return;
  free(b->RBinsLow);   free(b->RBinsHigh);
  free(b->EtaBinsLow); free(b->EtaBinsHigh);
  free(b->ringRBinStart); free(b->ringNRBins);
  free(b->ringRLo);    free(b->ringRHi);
  memset(b, 0, sizeof(*b));
}

void free_profile_data(ProfileData *p) {
  if (!p) return;
  free(p->profiles);
  free(p->norm);
  free(p->binMaskFlag);
  memset(p, 0, sizeof(*p));
}

void free_peak_data(PeakData *p) {
  if (!p) return;
  free(p->RMean);     free(p->EtaMean);
  free(p->IdealTtheta); free(p->PointDSpacing);
  free(p->RingNumbers); free(p->FitSNR);
  free(p->FitFWHM);
  free(p->skipBin);
  memset(p, 0, sizeof(*p));
}

void estep_free_result(EstepResult *r) {
  if (!r) return;
  free(r->RMean);     free(r->EtaMean);
  free(r->YMean);     free(r->ZMean);
  free(r->IdealTtheta); free(r->PointDSpacing);
  free(r->RingNumbers); free(r->FitSNR);
  free(r->FitFWHM);
  free(r->skipBin);
  memset(r, 0, sizeof(*r));
}

// ── 1. Build per-ring R bin edges ─────────────────────────────────

static int estep_build_ring_bins(const EstepGeometry *g,
                                  const RingTable *rings,
                                  BinEdges *out)
{
  int n = rings->n_hkls;
  out->n_hkls = n;

  out->ringRLo     = malloc(n * sizeof(double));
  out->ringRHi     = malloc(n * sizeof(double));
  out->ringRBinStart = malloc(n * sizeof(int));
  out->ringNRBins  = malloc(n * sizeof(int));
  if (!out->ringRLo || !out->ringRHi || !out->ringRBinStart || !out->ringNRBins)
    return -1;

  double halfW_px = g->Width / g->px;
  for (int r = 0; r < n; r++) {
    double IdealR = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r]) / g->px;
    out->ringRLo[r] = IdealR - halfW_px;
    out->ringRHi[r] = IdealR + halfW_px;
    if (out->ringRLo[r] < 0) out->ringRLo[r] = 0;
  }

  // Fine R bins: subdivide each pixel by RBinWidth
  out->rBinSize = 1.0 / g->RBinWidth;
  out->totalRBins = 0;
  for (int r = 0; r < n; r++) {
    out->ringRBinStart[r] = out->totalRBins;
    out->ringNRBins[r] = (int)ceil((out->ringRHi[r] - out->ringRLo[r]) / out->rBinSize);
    if (out->ringNRBins[r] < 3) out->ringNRBins[r] = 3;
    out->totalRBins += out->ringNRBins[r];
  }

  // Build global R bin edge arrays
  out->RBinsLow  = malloc(out->totalRBins * sizeof(double));
  out->RBinsHigh = malloc(out->totalRBins * sizeof(double));
  if (!out->RBinsLow || !out->RBinsHigh) return -1;

  for (int r = 0; r < n; r++) {
    for (int b = 0; b < out->ringNRBins[r]; b++) {
      int idx = out->ringRBinStart[r] + b;
      out->RBinsLow[idx]  = out->ringRLo[r] + b * out->rBinSize;
      out->RBinsHigh[idx] = out->ringRLo[r] + (b + 1) * out->rBinSize;
    }
  }

  // Eta bins: full 360° coverage
  out->nEtaBins = (int)(360.0 / g->EtaBinSize);
  if (out->nEtaBins < 1) out->nEtaBins = 1;
  out->EtaBinsLow  = malloc(out->nEtaBins * sizeof(double));
  out->EtaBinsHigh = malloc(out->nEtaBins * sizeof(double));
  if (!out->EtaBinsLow || !out->EtaBinsHigh) return -1;

  for (int e = 0; e < out->nEtaBins; e++) {
    out->EtaBinsLow[e]  = -180.0 + e * g->EtaBinSize;
    out->EtaBinsHigh[e] = -180.0 + (e + 1) * g->EtaBinSize;
  }

  return 0;
}

// ── 2. Build map, integrate, normalize ────────────────────────────

static int estep_build_and_integrate(
    const EstepGeometry *g, const BinEdges *bins, const RingTable *rings,
    const double *image, const double *dark, const double *mask,
    const Panel *pan, int nPan,
    ProfileData *out)
{
  int totalRBins = bins->totalRBins;
  int nEtaBins = bins->nEtaBins;
  long long nTotalBins = (long long)totalRBins * nEtaBins;

  // Allocate map structures
  struct MapPixelData ***pxList = malloc(totalRBins * sizeof(*pxList));
  int **nPxList = malloc(totalRBins * sizeof(*nPxList));
  int **maxnPx  = malloc(totalRBins * sizeof(*maxnPx));
  int *binMaskFlag = calloc(nTotalBins, sizeof(int));
  if (!pxList || !nPxList || !maxnPx || !binMaskFlag) return -1;

  for (int r = 0; r < totalRBins; r++) {
    pxList[r]  = calloc(nEtaBins, sizeof(*pxList[r]));
    nPxList[r] = calloc(nEtaBins, sizeof(*nPxList[r]));
    maxnPx[r]  = calloc(nEtaBins, sizeof(*maxnPx[r]));
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
  size_t nPx = (size_t)g->NrPixelsY * g->NrPixelsZ;
  double *distortionMapY = calloc(nPx, sizeof(double));
  double *distortionMapZ = calloc(nPx, sizeof(double));

  printf("Building map: %d R bins x %d Eta bins = %lld total bins\n",
         totalRBins, nEtaBins, nTotalBins);

  // Build the map
  long long nEntries = mapper_build_map(
      g->tx, g->ty, g->tz, g->NrPixelsY, g->NrPixelsZ, g->px, g->px,
      g->ybc, g->zbc, g->Lsd, g->MaxRingRad,
      g->p0, g->p1, g->p2, g->p3, g->p4, g->p5, g->p6, g->p7, g->p8, g->p9, g->p10,
      g->p11, g->p12, g->p13, g->p14,
      bins->EtaBinsLow, bins->EtaBinsHigh, bins->RBinsLow, bins->RBinsHigh,
      totalRBins, nEtaBins,
      pxList, nPxList, maxnPx,
      (double *)mask, binMaskFlag,
      g->NrTransOpt, g->TransOpt,
      binLocks, g->SubPixelLevel, g->SubPixelCardinalWidth,
      g->parallax, 0, 0, 0.0,
      distortionMapY, distortionMapZ,
      (Panel *)pan, nPan,
      g->residualCorr.map ? &g->residualCorr : NULL);

  printf("Map built: %lld pixel-bin entries\n", nEntries);

  // Integrate: map × image → profiles
  double *profiles = calloc(nTotalBins, sizeof(double));
  double *norm     = calloc(nTotalBins, sizeof(double));

  integration_apply_map(pxList, nPxList, totalRBins, nEtaBins,
                        (double *)image, g->NrPixelsY, g->NrPixelsZ,
                        (double *)dark, g->ybc, g->zbc, g->GradientCorrection,
                        profiles, norm);

  // Normalize profiles by area weight
  for (long long b = 0; b < nTotalBins; b++) {
    if (norm[b] > 1e-10)
      profiles[b] /= norm[b];
    else
      profiles[b] = 0;
  }

  // Save profiles for diagnostics (ci_profiles.csv)
  {
    FILE *pf = fopen("ci_profiles.csv", "w");
    if (pf) {
      fprintf(pf, "Ring,RBin,EtaBin,RCenter,EtaCenter,Intensity,Norm\n");
      for (int r = 0; r < rings->n_hkls; r++) {
        for (int rb = 0; rb < bins->ringNRBins[r]; rb++) {
          int globalRIdx = bins->ringRBinStart[r] + rb;
          double rCenter = (bins->RBinsLow[globalRIdx] + bins->RBinsHigh[globalRIdx]) * 0.5;
          for (int e = 0; e < nEtaBins; e++) {
            long long fPos = (long long)globalRIdx * nEtaBins + e;
            double etaCenter = (bins->EtaBinsLow[e] + bins->EtaBinsHigh[e]) * 0.5;
            if (profiles[fPos] != 0 || norm[fPos] > 0)
              fprintf(pf, "%d,%d,%d,%.6f,%.4f,%.6f,%.6f\n",
                      rings->RingIDs[r], rb, e, rCenter, etaCenter,
                      profiles[fPos], norm[fPos]);
          }
        }
      }
      fclose(pf);
      printf("Saved ci_profiles.csv (%d rings, %d R bins, %d Eta bins)\n",
             rings->n_hkls, totalRBins, nEtaBins);
    }
  }

  // Cleanup map structures (profiles and binMaskFlag survive via output)
  for (int r = 0; r < totalRBins; r++) {
    for (int e = 0; e < nEtaBins; e++)
      free(pxList[r][e]);
    free(pxList[r]);
    free(nPxList[r]);
    free(maxnPx[r]);
  }
  free(pxList); free(nPxList); free(maxnPx);
  free(distortionMapY); free(distortionMapZ);
  for (int i = 0; i < MAPPER_N_BIN_LOCKS; i++)
    omp_destroy_lock(&binLocks[i]);

  out->nTotalBins = nTotalBins;
  out->nEntries = nEntries;
  out->profiles = profiles;
  out->norm = norm;
  out->binMaskFlag = binMaskFlag;
  return 0;
}

// ── 3. Parallel peak fitting ──────────────────────────────────────

static int estep_fit_peaks(
    const EstepGeometry *g, const RingTable *rings,
    const BinEdges *bins, const ProfileData *prof,
    PeakData *out)
{
  int n_hkls = rings->n_hkls;
  int nEtaBins = bins->nEtaBins;
  int maxBins = n_hkls * nEtaBins;

  out->maxBins = maxBins;
  out->RMean         = calloc(maxBins, sizeof(double));
  out->EtaMean       = calloc(maxBins, sizeof(double));
  out->IdealTtheta   = malloc(maxBins * sizeof(double));
  out->PointDSpacing = malloc(maxBins * sizeof(double));
  out->RingNumbers   = malloc(maxBins * sizeof(int));
  out->FitSNR        = calloc(maxBins, sizeof(double));
  out->FitFWHM       = calloc(maxBins, sizeof(double));
  out->skipBin       = calloc(maxBins, sizeof(int));
  if (!out->RMean || !out->EtaMean || !out->IdealTtheta ||
      !out->PointDSpacing || !out->RingNumbers || !out->FitSNR ||
      !out->FitFWHM || !out->skipBin)
    return -1;

  // Detect doublets
  int *dbFlag = calloc(n_hkls, sizeof(int));
  int *dbPair = malloc(n_hkls * sizeof(int));
  for (int r = 0; r < n_hkls; r++) dbPair[r] = -1;
  if (g->DoubletSeparation > 0) {
    double sepPx = g->DoubletSeparation;
    for (int r = 0; r < n_hkls - 1; r++) {
      if (dbFlag[r] != 0) continue;
      double R1 = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r]) / g->px;
      double R2 = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r+1]) / g->px;
      if (fabs(R2 - R1) < sepPx) {
        dbFlag[r] = 1; dbFlag[r+1] = 2;
        dbPair[r] = r+1; dbPair[r+1] = r;
      }
    }
  }

  // Adaptive eta: compute per-ring merge factors
  int *etaMerge = malloc(n_hkls * sizeof(int));  // bins to average per ring
  if (g->AdaptiveEtaBins) {
    double maxR = 0;
    for (int r = 0; r < n_hkls; r++) {
      double idealR = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r]) / g->px;
      if (idealR > maxR) maxR = idealR;
    }
    for (int r = 0; r < n_hkls; r++) {
      double idealR = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r]) / g->px;
      int merge = (int)(maxR / idealR + 0.5);
      if (merge < 1) merge = 1;
      // Clamp so merged bin doesn't exceed ~10 degrees
      int maxMerge = (int)(10.0 / g->EtaBinSize);
      if (maxMerge < 1) maxMerge = 1;
      if (merge > maxMerge) merge = maxMerge;
      etaMerge[r] = merge;
    }
  } else {
    for (int r = 0; r < n_hkls; r++) etaMerge[r] = 1;
  }

  int nValid = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:nValid)
  for (int re = 0; re < maxBins; re++) {
    int r = re / nEtaBins;
    int e = re % nEtaBins;
    int binIdx = r * nEtaBins + e;

    // Adaptive: only process every nMerge-th eta bin (center of merged window)
    int nMerge = etaMerge[r];
    if (nMerge > 1 && (e % nMerge) != nMerge / 2) continue;

    // For each sub-bin in the merge window, check if it is mask-clean.
    // A sub-bin is clean if NONE of its R bins are mask-contaminated.
    int nCleanEta = 0;
    int *etaClean = NULL;
    if (nMerge > 1) {
      etaClean = malloc(nMerge * sizeof(int));
      for (int me = 0; me < nMerge; me++) {
        int ee = e - nMerge / 2 + me;
        if (ee < 0) ee += nEtaBins;
        if (ee >= nEtaBins) ee -= nEtaBins;
        int subMasked = 0;
        for (int rb = 0; rb < bins->ringNRBins[r]; rb++) {
          long long fPos = (long long)(bins->ringRBinStart[r] + rb) * nEtaBins + ee;
          if (prof->binMaskFlag[fPos]) { subMasked = 1; break; }
        }
        etaClean[me] = !subMasked;
        if (!subMasked) nCleanEta++;
      }
      // Need at least half the window to be clean
      if (nCleanEta < (nMerge + 1) / 2) {
        free(etaClean);
        out->skipBin[binIdx] = 1;
        continue;
      }
    } else {
      // nMerge == 1: original single-bin mask check
      int masked = 0;
      for (int rb = 0; rb < bins->ringNRBins[r]; rb++) {
        long long fPos = (long long)(bins->ringRBinStart[r] + rb) * nEtaBins + e;
        if (prof->binMaskFlag[fPos]) { masked = 1; break; }
      }
      if (masked) {
        out->skipBin[binIdx] = 1;
        continue;
      }
    }

    // Extract 1D radial profile: average across clean eta bins in the window
    int nPts = bins->ringNRBins[r];
    double *Rs = malloc(nPts * sizeof(double));
    double *PeakShape = calloc(nPts, sizeof(double));
    int hasData = 0;
    for (int rb = 0; rb < nPts; rb++) {
      int globalRIdx = bins->ringRBinStart[r] + rb;
      Rs[rb] = (bins->RBinsLow[globalRIdx] + bins->RBinsHigh[globalRIdx]) * 0.5;
      int nContrib = 0;
      for (int me = 0; me < nMerge; me++) {
        // Skip masked sub-bins
        if (etaClean && !etaClean[me]) continue;
        int ee = e - nMerge / 2 + me;
        if (ee < 0) ee += nEtaBins;
        if (ee >= nEtaBins) ee -= nEtaBins;
        long long fPos = (long long)globalRIdx * nEtaBins + ee;
        double val = prof->profiles[fPos];
        if (val > 0) { PeakShape[rb] += val; nContrib++; }
      }
      if (nContrib > 0) {
        PeakShape[rb] /= nContrib;
        hasData = 1;
      }
    }
    free(etaClean);

    if (!hasData || nPts < 5) {
      free(Rs); free(PeakShape);
      continue;
    }

    double IdealR = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[r]) / g->px;
    double Rfit, snr, fwhm = 0;
    double Rstep = bins->rBinSize;

    if (dbFlag[r] == 0) {
      pf_fit_single_peak(g->peakFitMode, nPts, Rs, PeakShape, &Rfit, &snr, &fwhm, Rstep, IdealR);
    } else if (dbFlag[r] == 1) {
      int pairR = dbPair[r];
      double IdealR2 = g->Lsd * tan(deg2rad * 2.0 * rings->Thetas[pairR]) / g->px;
      double Rmid = (IdealR + IdealR2) * 0.5;
      double Rfit2, snr2, fwhm2;
      pf_fit_doublet_peak(g->peakFitMode, nPts, Rs, PeakShape,
                           &Rfit, &Rfit2, &snr, &snr2,
                           &fwhm, &fwhm2,
                           Rstep, IdealR, IdealR2, Rmid);
    } else {
      pf_fit_single_peak(g->peakFitMode, nPts, Rs, PeakShape, &Rfit, &snr, &fwhm, Rstep, IdealR);
    }

    if (snr < 1.0) {
      free(Rs); free(PeakShape);
      continue;
    }

    double etaCenter = (bins->EtaBinsLow[e] + bins->EtaBinsHigh[e]) * 0.5;

    // DEBUG: print E-step peak fit details for ring 1
    if (r == 0 && (e % 30 == 0 || e == 0)) {
      double strain_dbg = fabs(1.0 - Rfit / IdealR);
      double Y_px = g->ybc + Rfit * sin(etaCenter * deg2rad);
      double Z_px = g->zbc + Rfit * cos(etaCenter * deg2rad);
      printf("  DEBUG E-step Ring1 eta=%7.1f: Rfit=%10.6f IdealR=%10.6f dR=%+.6f strain=%8.1f ue SNR=%6.1f Y_px=%9.3f Z_px=%9.3f nPts=%d\n",
             etaCenter, Rfit, IdealR, Rfit-IdealR, strain_dbg*1e6, snr, Y_px, Z_px, nPts);
    }

    // Store results
    out->RMean[binIdx]         = Rfit * g->px; // microns
    out->EtaMean[binIdx]       = etaCenter;
    out->IdealTtheta[binIdx]   = 2.0 * rings->Thetas[r];
    out->PointDSpacing[binIdx] = rings->DSpacings[r];
    out->RingNumbers[binIdx]   = rings->RingIDs[r];
    out->FitSNR[binIdx]        = snr;
    out->FitFWHM[binIdx]       = fwhm;
    nValid++;

    free(Rs);
    free(PeakShape);
  }

  free(dbFlag); free(dbPair); free(etaMerge);
  out->nValid = nValid;
  printf("E-step: %d valid bins from %d total\n", nValid, maxBins);
  return 0;
}

// ── 4. Compact valid bins + invert to raw pixel coords ────────────

static int estep_compact_and_invert(
    const EstepGeometry *g, const PeakData *peaks,
    const BinEdges *bins,
    const Panel *pan, int nPan,
    EstepResult *out)
{
  int nV = peaks->nValid;
  int nEtaBins = bins->nEtaBins;

  out->nValid        = nV;
  out->RMean         = malloc(nV * sizeof(double));
  out->EtaMean       = malloc(nV * sizeof(double));
  out->YMean         = malloc(nV * sizeof(double));
  out->ZMean         = malloc(nV * sizeof(double));
  out->IdealTtheta   = malloc(nV * sizeof(double));
  out->PointDSpacing = malloc(nV * sizeof(double));
  out->RingNumbers   = malloc(nV * sizeof(int));
  out->FitSNR        = malloc(nV * sizeof(double));
  out->FitFWHM       = malloc(nV * sizeof(double));
  out->skipBin       = calloc(nV, sizeof(int));
  if (!out->RMean || !out->EtaMean || !out->YMean || !out->ZMean ||
      !out->IdealTtheta || !out->PointDSpacing || !out->RingNumbers ||
      !out->FitSNR || !out->FitFWHM || !out->skipBin)
    return -1;

  // Build tilt matrix for the current E-step geometry
  double TRs_estep[3][3];
  dg_build_tilt_matrix(g->tx, g->ty, g->tz, TRs_estep);

  int cnt = 0;
  for (int i = 0; i < peaks->maxBins; i++) {
    if (peaks->RMean[i] == 0) continue;

    out->RMean[cnt]         = peaks->RMean[i];
    out->EtaMean[cnt]       = peaks->EtaMean[i];
    out->IdealTtheta[cnt]   = peaks->IdealTtheta[i];
    out->PointDSpacing[cnt] = peaks->PointDSpacing[i];
    out->RingNumbers[cnt]   = peaks->RingNumbers[i];
    out->FitSNR[cnt]        = peaks->FitSNR[i];
    out->FitFWHM[cnt]       = peaks->FitFWHM[i];

    // Convert (R, Eta) → raw pixel coords via panel-aware inversion
    double R_px = out->RMean[cnt] / g->px;
    const Panel *binPanel = NULL;
    if (nPan > 0) {
      double approxY = g->ybc + R_px * sin(out->EtaMean[cnt] * deg2rad);
      double approxZ = g->zbc + R_px * cos(out->EtaMean[cnt] * deg2rad);
      int pIdx = GetPanelIndex(approxY, approxZ, nPan, pan);
      if (pIdx >= 0)
        binPanel = &pan[pIdx];
    }

    double Y_inv, Z_inv;
    dg_invert_REta_to_pixel_panel_corr(R_px, out->EtaMean[cnt],
                             g->ybc, g->zbc, TRs_estep, g->Lsd, g->MaxRingRad,
                             g->p0, g->p1, g->p2, g->p3, g->p4, g->p5, g->p6,
                             g->p7, g->p8, g->p9, g->p10,
                             g->p11, g->p12, g->p13, g->p14,
                             g->px, g->parallax,
                             g->residualCorr.map ? &g->residualCorr : NULL,
                             binPanel,
                             &Y_inv, &Z_inv);
    out->YMean[cnt] = Y_inv;
    out->ZMean[cnt] = Z_inv;
    if (peaks->skipBin[i]) out->skipBin[cnt] = 1;
    cnt++;
  }

  out->nValid = cnt;
  return 0;
}

// ── Top-level E-step orchestrator ─────────────────────────────────
// ── Ring diagnostics helper ──────────────────────────────────────────────────
// Prints per-ring statistics (mean/median signed+abs strain, SNR) and
// optionally appends to a CSV file.

static void emit_ring_diagnostics(
    int iter, int n_hkls, const int *RingIDs,
    int nIndices, const int *RingNumbers, const double *DiffIns,
    const double *FitSNR, const int *skipBin,
    double outlierFactor, int OutlierIterations,
    const char *csvPath)
{
  // Compute outlier threshold (same sigma-clip as M-step)
  double outlierThreshold = 0;
  if (outlierFactor > 0) {
    double rawMean = 0;
    int rawCount = 0;
    for (int bi = 0; bi < nIndices; bi++) {
      if (skipBin && skipBin[bi]) continue;
      rawMean += fabs(DiffIns[bi]);
      rawCount++;
    }
    if (rawCount > 0) rawMean /= rawCount;
    int nClipIter = (OutlierIterations > 0) ? OutlierIterations : 1;
    double clipMean = rawMean;
    for (int ci = 0; ci < nClipIter; ci++) {
      double thresh = outlierFactor * clipMean;
      double newSum = 0; int newCnt = 0;
      for (int bi = 0; bi < nIndices; bi++) {
        if (skipBin && skipBin[bi]) continue;
        if (fabs(DiffIns[bi]) <= thresh) { newSum += fabs(DiffIns[bi]); newCnt++; }
      }
      if (newCnt > 0) clipMean = newSum / newCnt;
    }
    outlierThreshold = outlierFactor * clipMean;
  }

  // Open CSV for append if requested
  FILE *csvFP = NULL;
  if (csvPath && csvPath[0] != '\0') {
    int isNew = 0;
    FILE *testFP = fopen(csvPath, "r");
    if (!testFP) isNew = 1; else fclose(testFP);
    csvFP = fopen(csvPath, "a");
    if (csvFP && isNew)
      fprintf(csvFP, "Iter,Ring,NPoints,MeanDR_ppm,MedDR_ppm,MeanAbsDR_ppm,MedAbsDR_ppm,MeanSNR\n");
  }

  printf("  Ring  NPoints   Mean(ΔR) µε  Med(ΔR) µε  Mean|ΔR| µε  Med|ΔR| µε  MeanSNR\n");
  printf("  ----  -------   -----------  ----------  -----------  ----------  -------\n");
  for (int ri = 0; ri < n_hkls; ri++) {
    int cnt = 0;
    for (int bi = 0; bi < nIndices; bi++) {
      if (RingNumbers[bi] == RingIDs[ri] && !(skipBin && skipBin[bi])) {
        if (outlierThreshold > 0 && fabs(DiffIns[bi]) > outlierThreshold) continue;
        cnt++;
      }
    }
    if (cnt == 0) continue;
    double *drs = malloc(cnt * sizeof(double));
    double *adrs = malloc(cnt * sizeof(double));
    double sumSNR = 0;
    int k = 0;
    for (int bi = 0; bi < nIndices; bi++) {
      if (RingNumbers[bi] == RingIDs[ri] && !(skipBin && skipBin[bi])) {
        if (outlierThreshold > 0 && fabs(DiffIns[bi]) > outlierThreshold) continue;
        drs[k] = DiffIns[bi];
        adrs[k] = fabs(DiffIns[bi]);
        sumSNR += FitSNR[bi];
        k++;
      }
    }
    double meanDR = 0, meanADR = 0;
    for (int j = 0; j < cnt; j++) { meanDR += drs[j]; meanADR += adrs[j]; }
    meanDR /= cnt; meanADR /= cnt;
    qsort(drs, cnt, sizeof(double), calib_cmp_double);
    qsort(adrs, cnt, sizeof(double), calib_cmp_double);
    double medDR = (cnt % 2) ? drs[cnt/2] : (drs[cnt/2-1] + drs[cnt/2]) / 2;
    double medADR = (cnt % 2) ? adrs[cnt/2] : (adrs[cnt/2-1] + adrs[cnt/2]) / 2;
    printf("  %4d  %7d   %11.3f  %10.3f  %11.3f  %10.3f  %7.1f\n",
           RingIDs[ri], cnt,
           meanDR * 1e6, medDR * 1e6, meanADR * 1e6, medADR * 1e6,
           sumSNR / cnt);
    if (csvFP)
      fprintf(csvFP, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.1f\n",
              iter, RingIDs[ri], cnt,
              meanDR * 1e6, medDR * 1e6, meanADR * 1e6, medADR * 1e6,
              sumSNR / cnt);
    free(drs); free(adrs);
  }
  if (csvFP) { fflush(csvFP); fclose(csvFP); }
}

// ── Checkpoint write/load ────────────────────────────────────────────────────

static void write_checkpoint(const char *rawFN, int iter,
    double Lsd, double ybc, double zbc, double ty, double tz,
    double p0, double p1, double p2, double p3,
    double p4, double p5, double p6, double p7, double p8, double p9, double p10,
    double p11, double p12, double p13, double p14,
    double parallax, double MeanDiff)
{
  char fn[4096];
  snprintf(fn, sizeof(fn), "%s.checkpoint.txt", rawFN);
  FILE *fp = fopen(fn, "w");
  if (!fp) return;
  fprintf(fp, "Iteration %d\n", iter);
  fprintf(fp, "Lsd %.12f\n", Lsd);
  fprintf(fp, "ybc %.12f\n", ybc);
  fprintf(fp, "zbc %.12f\n", zbc);
  fprintf(fp, "ty %.12f\n", ty);
  fprintf(fp, "tz %.12f\n", tz);
  fprintf(fp, "p0 %.12e\n", p0);
  fprintf(fp, "p1 %.12e\n", p1);
  fprintf(fp, "p2 %.12e\n", p2);
  fprintf(fp, "p3 %.12e\n", p3);
  fprintf(fp, "p4 %.12e\n", p4);
  fprintf(fp, "p5 %.12e\n", p5);
  fprintf(fp, "p6 %.12e\n", p6);
  fprintf(fp, "p7 %.12e\n", p7);
  fprintf(fp, "p8 %.12e\n", p8);
  fprintf(fp, "p9 %.12e\n", p9);
  fprintf(fp, "p10 %.12e\n", p10);
  fprintf(fp, "p11 %.12e\n", p11);
  fprintf(fp, "p12 %.12e\n", p12);
  fprintf(fp, "p13 %.12e\n", p13);
  fprintf(fp, "p14 %.12e\n", p14);
  fprintf(fp, "Parallax %.12e\n", parallax);
  fprintf(fp, "MeanStrain %.12e\n", MeanDiff);
  fclose(fp);
}

static int load_checkpoint(const char *rawFN, int *iter,
    double *Lsd, double *ybc, double *zbc, double *ty, double *tz,
    double *p0, double *p1, double *p2, double *p3,
    double *p4, double *p5, double *p6, double *p7, double *p8, double *p9, double *p10,
    double *p11, double *p12, double *p13, double *p14,
    double *parallax, double *MeanDiff)
{
  char fn[4096];
  snprintf(fn, sizeof(fn), "%s.checkpoint.txt", rawFN);
  FILE *fp = fopen(fn, "r");
  if (!fp) return -1;
  char aline[256];
  while (fgets(aline, sizeof(aline), fp)) {
    sscanf(aline, "Iteration %d", iter);
    sscanf(aline, "Lsd %lf", Lsd);
    sscanf(aline, "ybc %lf", ybc);
    sscanf(aline, "zbc %lf", zbc);
    sscanf(aline, "ty %lf", ty);
    sscanf(aline, "tz %lf", tz);
    sscanf(aline, "p0 %lf", p0);
    sscanf(aline, "p1 %lf", p1);
    sscanf(aline, "p2 %lf", p2);
    sscanf(aline, "p3 %lf", p3);
    sscanf(aline, "p4 %lf", p4);
    sscanf(aline, "p5 %lf", p5);
    sscanf(aline, "p6 %lf", p6);
    sscanf(aline, "p7 %lf", p7);
    sscanf(aline, "p8 %lf", p8);
    sscanf(aline, "p9 %lf", p9);
    sscanf(aline, "p10 %lf", p10);
    sscanf(aline, "p11 %lf", p11);
    sscanf(aline, "p12 %lf", p12);
    sscanf(aline, "p13 %lf", p13);
    sscanf(aline, "p14 %lf", p14);
    sscanf(aline, "Parallax %lf", parallax);
    sscanf(aline, "MeanStrain %lf", MeanDiff);
  }
  fclose(fp);
  return 0;
}

// ── E-step orchestrator ──────────────────────────────────────────────────────
// Returns number of valid bins (nIndices).
// Caller must free result via estep_free_result().

static int run_estep(
    const EstepGeometry *geom, const RingTable *rings,
    const double *image, const double *dark, const double *mask,
    const Panel *pan, int nPan,
    EstepResult *result)
{
  BinEdges     bins  = {0};
  ProfileData  prof  = {0};
  PeakData     peaks = {0};
  int rc = 0;

  if ((rc = estep_build_ring_bins(geom, rings, &bins)) != 0) {
    fprintf(stderr, "estep_build_ring_bins failed\n");
    goto cleanup;
  }

  if ((rc = estep_build_and_integrate(geom, &bins, rings, image, dark, mask,
                                       pan, nPan, &prof)) != 0) {
    fprintf(stderr, "estep_build_and_integrate failed\n");
    goto cleanup;
  }

  if ((rc = estep_fit_peaks(geom, rings, &bins, &prof, &peaks)) != 0) {
    fprintf(stderr, "estep_fit_peaks failed\n");
    goto cleanup;
  }

  if (peaks.nValid < 3) {
    fprintf(stderr, "E-step: only %d valid peaks (need ≥ 3)\n", peaks.nValid);
    rc = -1;
    goto cleanup;
  }

  if ((rc = estep_compact_and_invert(geom, &peaks, &bins, pan, nPan, result)) != 0) {
    fprintf(stderr, "estep_compact_and_invert failed\n");
    goto cleanup;
  }

cleanup:
  free_bin_edges(&bins);
  free_profile_data(&prof);
  free_peak_data(&peaks);
  return (rc == 0) ? result->nValid : rc;
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
  double p4in = cfg.p4, p5in = cfg.p5, p6in = cfg.p6;
  double p7in = cfg.p7, p8in = cfg.p8, p9in = cfg.p9, p10in = cfg.p10;
  double p11in = cfg.p11, p12in = cfg.p12, p13in = cfg.p13, p14in = cfg.p14;
  double MaxRingRad = cfg.RhoD, Wavelength = cfg.Wavelength;
  double EtaBinSize = cfg.EtaBinSize;
  int RBinWidth = cfg.RBinWidth;
  double tolTilts = cfg.tolTilts, tolLsd = cfg.tolLsd, tolBC = cfg.tolBC;
  double tolP0 = cfg.tolP0, tolP1 = cfg.tolP1, tolP2 = cfg.tolP2;
  double tolP3 = cfg.tolP3, tolP4 = cfg.tolP4, tolP5 = cfg.tolP5, tolP6 = cfg.tolP6;
  double tolP7 = cfg.tolP7, tolP8 = cfg.tolP8, tolP9 = cfg.tolP9, tolP10 = cfg.tolP10;
  double tolP11 = cfg.tolP11, tolP12 = cfg.tolP12, tolP13 = cfg.tolP13, tolP14 = cfg.tolP14;
  double tolShifts = cfg.tolShifts, tolRotation = cfg.tolRotation;
  double outlierFactor = cfg.outlierFactor;
  int MinIndicesForFit = cfg.MinIndicesForFit, FixPanelID = cfg.FixPanelID;
  int nIterations = cfg.nIterations;
  int iterOffset = cfg.iterOffset;   // iteration numbering offset for multi-stage runs
  double DoubletSeparation = cfg.DoubletSeparation;
  int NormalizeRingWeights = cfg.NormalizeRingWeights;
  int OutlierIterations = cfg.OutlierIterations;
  int RemoveOutliersBetweenIters = cfg.RemoveOutliersBetweenIters;
  int ReFitPeaks = cfg.ReFitPeaks;
  double TrimmedMeanFraction = cfg.TrimmedMeanFraction;
  int WeightByRadius = cfg.WeightByRadius, WeightByFitSNR = cfg.WeightByFitSNR;
  int WeightByPositionUncertainty = cfg.WeightByPositionUncertainty;
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
  double ConvergenceThresholdPPM = cfg.ConvergenceThresholdPPM;
  int SkipVerification = cfg.SkipVerification;
  int ResumeFromCheckpoint = cfg.ResumeFromCheckpoint;
  int GradientCorrection = cfg.GradientCorrection;
  int convergenceCount = 0;  // consecutive iters below threshold

  printf("\n=== CalibrantIntegratorOMP ===\n");
  printf("Detector: %dx%d, px=%.4f, Lsd=%.2f\n", NrPixelsY, NrPixelsZ, px, Lsd);
  printf("Iterations: %d, RBinWidth: %d, EtaBinSize: %.2f\n",
         nIterations, RBinWidth, EtaBinSize);

  // Generate HKL list
  int n_hkls = 0;
  double MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);

  run_midas_binary("GetHKLList", ParamFN);
  FILE *hklf = fopen("hkls.csv", "r");
  if (!hklf) { fprintf(stderr, "Cannot open hkls.csv\n"); return 1; }

  // Pre-count data lines to allocate dynamic arrays
  char aline[1024];
  fgets(aline, sizeof(aline), hklf);  // header
  int hklCapacity = 0;
  while (fgets(aline, sizeof(aline), hklf) != NULL)
    hklCapacity++;
  if (hklCapacity == 0) {
    fprintf(stderr, "hkls.csv has no data lines\n");
    fclose(hklf);
    return 1;
  }
  rewind(hklf);

  double *Thetas   = malloc(hklCapacity * sizeof(double));
  double *DSpacings = malloc(hklCapacity * sizeof(double));
  int    *RingIDs   = malloc(hklCapacity * sizeof(int));
  if (!Thetas || !DSpacings || !RingIDs) {
    fprintf(stderr, "Failed to allocate HKL arrays (%d entries)\n", hklCapacity);
    fclose(hklf);
    return 1;
  }

  // Parse header to find column indices for 'RingNr' and 'Theta'
  fgets(aline, sizeof(aline), hklf);  // re-read header
  int colRingNr = -1, colTheta = -1;
  {
    char hdrCopy[1024];
    strncpy(hdrCopy, aline, sizeof(hdrCopy));
    hdrCopy[sizeof(hdrCopy) - 1] = '\0';
    char *tok = strtok(hdrCopy, " \t\r\n");
    int col = 0;
    while (tok) {
      if (strcmp(tok, "RingNr") == 0)    colRingNr = col;
      else if (strcmp(tok, "Theta") == 0) colTheta = col;
      tok = strtok(NULL, " \t\r\n");
      col++;
    }
  }
  if (colRingNr < 0 || colTheta < 0) {
    fprintf(stderr, "hkls.csv header missing required columns: "
            "RingNr(%s) Theta(%s)\n",
            colRingNr < 0 ? "MISSING" : "ok",
            colTheta < 0 ? "MISSING" : "ok");
    fclose(hklf);
    free(Thetas); free(DSpacings); free(RingIDs);
    return 1;
  }
  int maxCol = (colRingNr > colTheta) ? colRingNr : colTheta;

  int LastRingDone = 0;
  while (fgets(aline, sizeof(aline), hklf) != NULL) {
    // Tokenize the line and extract columns by index
    char lineCopy[1024];
    strncpy(lineCopy, aline, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';
    char *tokens[32];
    int nTok = 0;
    char *tok = strtok(lineCopy, " \t\r\n");
    while (tok && nTok < 32) {
      tokens[nTok++] = tok;
      tok = strtok(NULL, " \t\r\n");
    }
    if (nTok <= maxCol) continue;  // malformed line

    int tRnr = atoi(tokens[colRingNr]);
    double theta = atof(tokens[colTheta]);

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
  printf("%d rings selected (capacity %d).\n\n", n_hkls, hklCapacity);

  // Build RingTable for the factored E-step
  RingTable rings = { .n_hkls = n_hkls, .Thetas = Thetas,
                      .DSpacings = DSpacings, .RingIDs = RingIDs };

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

  // Load residual correction map (binary file: NrPixelsY*NrPixelsZ doubles)
  DGResidualCorr residualCorr = {NULL, 0, 0};
  if (cfg.ResidualCorrMapFN[0] != '\0') {
    FILE *rcf = fopen(cfg.ResidualCorrMapFN, "rb");
    if (rcf == NULL) {
      fprintf(stderr, "Error: cannot open residual correction map %s\n",
              cfg.ResidualCorrMapFN);
      return 1;
    }
    fseek(rcf, 0L, SEEK_END);
    long rcSize = ftell(rcf);
    rewind(rcf);
    size_t expectedSize = (size_t)NrPixelsY * NrPixelsZ * sizeof(double);
    if ((size_t)rcSize != expectedSize) {
      fprintf(stderr, "Error: residual correction map size mismatch: "
              "got %ld, expected %zu (%dx%d doubles)\n",
              rcSize, expectedSize, NrPixelsY, NrPixelsZ);
      fclose(rcf);
      return 1;
    }
    double *corrMap = malloc(expectedSize);
    if (corrMap == NULL) {
      fprintf(stderr, "Error: failed to allocate residual correction map\n");
      fclose(rcf);
      return 1;
    }
    fread(corrMap, expectedSize, 1, rcf);
    fclose(rcf);
    residualCorr.map = corrMap;
    residualCorr.NrPixelsY = NrPixelsY;
    residualCorr.NrPixelsZ = NrPixelsZ;
    printf("Loaded residual correction map: %s (%dx%d)\n",
           cfg.ResidualCorrMapFN, NrPixelsY, NrPixelsZ);
  }

  // ── Iteration loop ────────────────────────────────────────────────
  double LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3;
  double MeanDiff = 1e20, StdDiff = 0;
  int nIndices = 0;
  // E-step result: managed via EstepResult struct
  EstepResult eResult = {0};
  // Aliases for downstream M-step code (point into eResult arrays)
  double *RMean = NULL, *EtaMean = NULL, *YMean = NULL, *ZMean = NULL;
  double *IdealTtheta = NULL, *PointDSpacing = NULL, *FitSNR = NULL, *FitFWHM = NULL;
  int *RingNumbers = NULL, *skipBin = NULL;
  double *EtaIns = NULL, *DiffIns = NULL, *RadIns = NULL;
  double *Yc = NULL, *Zc = NULL;

  // Best-iteration tracking
  double bestMeanDiff = 1e20;
  int bestIter = -1;
  double bestLsd, bestYbc, bestZbc, bestTy, bestTz;
  double bestP0, bestP1, bestP2, bestP3, bestP4, bestP5, bestP6, bestP7, bestP8, bestP9, bestP10;
  double bestP11, bestP12, bestP13, bestP14, bestParallax;
  Panel *bestPanels = (nPanels > 0) ? malloc(nPanels * sizeof(Panel)) : NULL;

  // Guard state
  double prevIterMeanDiff = 1e20, prevPrevIterMeanDiff = 1e20;
  int stagnantCount = 0, oscillationCount = 0;
  int postPerturbGrace = 0;

  // Initial parameters (for bound anchoring)
  // Layout must match CalibrationCore.c expectations:
  //   0-4: Lsd, ybc, zbc, ty, tz
  //   5-8: p0, p1, p2, p3
  //   9-11: p4, p5, p6
  //   12-15: p7, p8, p9, p10
  //   16-19: p11, p12, p13, p14
  //   20: parallax
  double initParams[21];
  initParams[0] = Lsd;   initParams[1] = ybc;  initParams[2] = zbc;
  initParams[3] = tyin;  initParams[4] = tzin;
  initParams[5] = p0in;  initParams[6] = p1in; initParams[7] = p2in;
  initParams[8] = p3in;  initParams[9] = p4in; initParams[10] = p5in;
  initParams[11] = p6in;
  initParams[12] = p7in; initParams[13] = p8in;
  initParams[14] = p9in; initParams[15] = p10in;
  initParams[16] = p11in; initParams[17] = p12in;
  initParams[18] = p13in; initParams[19] = p14in;
  initParams[20] = parallaxIn;
  Panel *initPanels = NULL;
  if (nPanels > 0) {
    initPanels = malloc(nPanels * sizeof(Panel));
    memcpy(initPanels, panels, nPanels * sizeof(Panel));
  }

  // Build rawFN for output file naming: FileStem_NNNNNN.ext
  char rawFN[4096];
  snprintf(rawFN, sizeof(rawFN), "%s_%0*d.%s",
           cfg.FileStem, Padding, cfg.StartNr, cfg.Ext);

  // Checkpoint resume
  if (ResumeFromCheckpoint && nIterations > 0) {
    int ckptIter = 0;
    double ckptMean = 0;
    if (load_checkpoint(rawFN, &ckptIter, &Lsd, &ybc, &zbc, &tyin, &tzin,
                        &p0in, &p1in, &p2in, &p3in, &p4in, &p5in, &p6in,
                        &p7in, &p8in, &p9in, &p10in,
                        &p11in, &p12in, &p13in, &p14in,
                        &parallaxIn, &ckptMean) == 0) {
      iterOffset += ckptIter;
      printf("\n*** Resuming from checkpoint (iter %d, MeanStrain=%.3f ppm) ***\n",
             ckptIter, ckptMean * 1e6);
      // Update initParams for bound anchoring
      initParams[0] = Lsd; initParams[1] = ybc; initParams[2] = zbc;
      initParams[3] = tyin; initParams[4] = tzin;
      initParams[5] = p0in; initParams[6] = p1in; initParams[7] = p2in;
      initParams[8] = p3in; initParams[9] = p4in; initParams[10] = p5in;
      initParams[11] = p6in; initParams[12] = p7in; initParams[13] = p8in;
      initParams[14] = p9in; initParams[15] = p10in;
      initParams[16] = p11in; initParams[17] = p12in;
      initParams[18] = p13in; initParams[19] = p14in;
      initParams[20] = parallaxIn;
    } else {
      printf("  Checkpoint file not found — starting from scratch.\n");
    }
  }

  // Convergence CSV (in cwd)
  char convHistFN[4096];
  snprintf(convHistFN, sizeof(convHistFN), "%s.convergence_history.csv", rawFN);
  FILE *convHistFP = fopen(convHistFN, iterOffset > 0 ? "a" : "w");
  if (convHistFP && iterOffset == 0)
    fprintf(convHistFP, "Iter,MeanStrain_ppm,StdStrain_ppm,Lsd,ybc,zbc,"
                        "ty,tz,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14\n");

  // nIterations=0: evaluate-only mode — run E-step once, skip optimization
  if (nIterations == 0) {
    printf("\n*** nIterations=0: evaluate-only mode ***\n");
    EstepGeometry eg0 = {
      .Lsd = Lsd, .ybc = ybc, .zbc = zbc, .tx = tx, .ty = tyin, .tz = tzin,
      .p0 = p0in, .p1 = p1in, .p2 = p2in, .p3 = p3in,
      .p4 = p4in, .p5 = p5in, .p6 = p6in,
      .p7 = p7in, .p8 = p8in, .p9 = p9in, .p10 = p10in,
      .p11 = p11in, .p12 = p12in, .p13 = p13in, .p14 = p14in,
      .px = px, .MaxRingRad = MaxRingRad, .parallax = parallaxIn,
      .EtaBinSize = EtaBinSize, .RBinWidth = RBinWidth,
      .NrPixelsY = NrPixelsY, .NrPixelsZ = NrPixelsZ, .NrPixels = NrPixels,
      .NrTransOpt = NrTransOpt, .SubPixelLevel = cfg.SubPixelLevel,
      .SubPixelCardinalWidth = cfg.SubPixelCardinalWidth,
      .GradientCorrection = GradientCorrection,
      .peakFitMode = peakFitMode, .DoubletSeparation = DoubletSeparation,
      .Wavelength = Wavelength, .Width = cfg.Width,
      .AdaptiveEtaBins = cfg.AdaptiveEtaBins,
      .residualCorr = residualCorr
    };
    memcpy(eg0.TransOpt, TransOpt, sizeof(TransOpt));
    estep_free_result(&eResult);
    nIndices = run_estep(&eg0, &rings, Average, AverageDark, mask,
                         panels, nPanels, &eResult);
    // Alias result arrays for downstream code
    RMean = eResult.RMean; EtaMean = eResult.EtaMean;
    YMean = eResult.YMean; ZMean = eResult.ZMean;
    IdealTtheta = eResult.IdealTtheta; PointDSpacing = eResult.PointDSpacing;
    RingNumbers = eResult.RingNumbers; FitSNR = eResult.FitSNR; FitFWHM = eResult.FitFWHM;
    skipBin = eResult.skipBin;
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
    printf("\n──── Iteration %d/%d ────\n", iter + 1, nIterations);

    // Free previous E-step result + per-iter arrays
    estep_free_result(&eResult);
    RMean = EtaMean = YMean = ZMean = NULL;
    IdealTtheta = PointDSpacing = FitSNR = FitFWHM = NULL;
    RingNumbers = NULL; skipBin = NULL;
    free(EtaIns); free(DiffIns); free(RadIns);
    free(Yc); free(Zc);
    EtaIns = DiffIns = RadIns = Yc = Zc = NULL;

    // Build EstepGeometry with current best params
    EstepGeometry eg = {
      .Lsd = (iter > 0) ? LsdFit : Lsd,
      .ybc = (iter > 0) ? ybcFit : ybc,
      .zbc = (iter > 0) ? zbcFit : zbc,
      .tx = tx,
      .ty  = (iter > 0) ? ty     : tyin,
      .tz  = (iter > 0) ? tz     : tzin,
      .p0  = (iter > 0) ? p0     : p0in,
      .p1  = (iter > 0) ? p1     : p1in,
      .p2  = (iter > 0) ? p2     : p2in,
      .p3  = (iter > 0) ? p3     : p3in,
      .p4 = p4in, .p5 = p5in, .p6 = p6in,
      .p7 = p7in, .p8 = p8in, .p9 = p9in, .p10 = p10in,
      .p11 = p11in, .p12 = p12in, .p13 = p13in, .p14 = p14in,
      .px = px, .MaxRingRad = MaxRingRad, .parallax = parallaxIn,
      .EtaBinSize = EtaBinSize, .RBinWidth = RBinWidth,
      .NrPixelsY = NrPixelsY, .NrPixelsZ = NrPixelsZ, .NrPixels = NrPixels,
      .NrTransOpt = NrTransOpt, .SubPixelLevel = cfg.SubPixelLevel,
      .SubPixelCardinalWidth = cfg.SubPixelCardinalWidth,
      .GradientCorrection = GradientCorrection,
      .peakFitMode = peakFitMode, .DoubletSeparation = DoubletSeparation,
      .Wavelength = Wavelength, .Width = cfg.Width,
      .AdaptiveEtaBins = cfg.AdaptiveEtaBins,
      .residualCorr = residualCorr
    };
    memcpy(eg.TransOpt, TransOpt, sizeof(TransOpt));

    // Run E-step with current best geometry
    nIndices = run_estep(&eg, &rings, Average, AverageDark, mask,
                         panels, nPanels, &eResult);

    if (nIndices < 3) {
      fprintf(stderr, "E-step produced only %d bins — aborting.\n", nIndices);
      break;
    }

    // Alias result arrays for downstream M-step code
    RMean = eResult.RMean; EtaMean = eResult.EtaMean;
    YMean = eResult.YMean; ZMean = eResult.ZMean;
    IdealTtheta = eResult.IdealTtheta; PointDSpacing = eResult.PointDSpacing;
    RingNumbers = eResult.RingNumbers; FitSNR = eResult.FitSNR; FitFWHM = eResult.FitFWHM;
    skipBin = eResult.skipBin;

    // Allocate per-bin output arrays
    Yc = malloc(nIndices * sizeof(double));
    Zc = malloc(nIndices * sizeof(double));
    EtaIns = malloc(nIndices * sizeof(double));
    DiffIns = malloc(nIndices * sizeof(double));
    RadIns = malloc(nIndices * sizeof(double));
    memcpy(Yc, YMean, nIndices * sizeof(double));
    memcpy(Zc, ZMean, nIndices * sizeof(double));

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

    // Position-uncertainty weights: w ~ (SNR / FWHM)^2
    // Replaces SNR weights when enabled, since it already incorporates SNR.
    double *posWeights = NULL;
    if (WeightByPositionUncertainty && FitFWHM != NULL && FitSNR != NULL) {
      posWeights = malloc(nIndices * sizeof(double));
      double maxPosW = 0;
      for (int i = 0; i < nIndices; i++) {
        double fw = FitFWHM[i];
        if (fw < 1e-6) fw = 1.0;
        double snrVal = FitSNR[i];
        double rawW = snrVal / fw;
        posWeights[i] = rawW * rawW;
        if (posWeights[i] > maxPosW) maxPosW = posWeights[i];
      }
      if (maxPosW > 0) {
        for (int i = 0; i < nIndices; i++)
          posWeights[i] /= maxPosW;
      }
    }
    double *effectiveSNRWeights = posWeights ? posWeights : snrWeights;

    // M-step: optimize geometry
    double p4 = p4in, p5 = p5in, p6 = p6in, p7 = p7in, p8 = p8in, p9 = p9in, p10 = p10in;
    double p11 = p11in, p12 = p12in, p13 = p13in, p14 = p14in;
    double wavelengthFit = Wavelength, parallaxFit = parallaxIn;

    // Open per-evaluation trace file for this iteration
    char traceFN[4096];
    snprintf(traceFN, sizeof(traceFN), "%s.m_step_trace_iter%d.csv",
             rawFN, iter + 1 + iterOffset);
    calib_set_trace_file(traceFN);

    int mstepRC = calib_fit_tilt_bc_lsd(
        &ctx,
        nIndices, Yc, Zc, IdealTtheta, Lsd, MaxRingRad, ybc, zbc, tx,
        tyin, tzin, p0in, p1in, p2in, p3in, &ty, &tz, &LsdFit,
        &ybcFit, &zbcFit, &p0, &p1, &p2, &p3, &MeanDiff, tolTilts,
        tolLsd, tolBC, 0, tolP0, tolP1, tolP2, tolP3, tolShifts,
        tolRotation, px, outlierFactor, MinIndicesForFit, FixPanelID,
        RingWeights, p4in, tolP4, PerPanelLsd, tolLsdPanel,
        PerPanelDistortion, tolP2Panel, WeightByRadius, effectiveSNRWeights,
        &p4, p5in, tolP5, &p5, p6in, tolP6, &p6,
        p7in, tolP7, &p7, p8in, tolP8, &p8, p9in, tolP9, &p9, p10in, tolP10, &p10,
        p11in, tolP11, &p11, p12in, tolP12, &p12, p13in, tolP13, &p13, p14in, tolP14, &p14,
        iter == 0, L2Objective, initParams, initPanels,
        FitWavelength, Wavelength, tolWavelength, PointDSpacing,
        &wavelengthFit,
        parallaxIn, tolParallax, &parallaxFit,
        TrimmedMeanFraction, skipBin, &residualCorr);

    // Close per-evaluation trace file
    calib_close_trace_file();

    // Log M-step convergence status
    if (mstepRC == 1)
      printf("  M-step: hit eval limit (may need more iterations)\n");
    else if (mstepRC < 0)
      printf("  M-step: NLopt error (rc=%d)\n", mstepRC);

    if (FitParallax) parallaxIn = parallaxFit;
    if (FitWavelength) Wavelength = wavelengthFit;

    // Evaluate residuals
    int *iterOutlier = NULL;
    if (RemoveOutliersBetweenIters && outlierFactor > 0 && iter < nIterations - 1)
      iterOutlier = calloc(nIndices, sizeof(int));
    calib_correct_tilt_distortion(
        &ctx,
        nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
        zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
        &StdDiff, outlierFactor, iterOutlier, p4, p5, p6, p7, p8, p9, p10,
        p11, p12, p13, p14, OutlierIterations,
        0, &MeanDiff, parallaxIn, skipBin, &residualCorr);

    printf("Iter %2d/%d  MeanStrain %8.3f  StdStrain %8.3f  nBins=%d\n",
           iter + 1, nIterations, MeanDiff * 1e6, StdDiff * 1e6, nIndices);

    // Per-ring diagnostic summary (first + last iter, or every iter if CSV enabled)
    if (iter == 0 || iter == nIterations - 1 || cfg.RingDiagnosticsCSV[0] != '\0') {
      emit_ring_diagnostics(
          iter + 1 + iterOffset, n_hkls, RingIDs,
          nIndices, RingNumbers, DiffIns, FitSNR, skipBin,
          outlierFactor, OutlierIterations,
          cfg.RingDiagnosticsCSV);
    }

    if (convHistFP) {
      fprintf(convHistFP,
              "%d,%.6e,%.6e,%.6f,%.6f,%.6f,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
              iter + 1 + iterOffset, MeanDiff * 1e6, StdDiff * 1e6,
              LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
              p11, p12, p13, p14);
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
    p4in = p4; p5in = p5; p6in = p6;
    p7in = p7; p8in = p8; p9in = p9; p10in = p10;
    p11in = p11; p12in = p12; p13in = p13; p14in = p14;
    Lsd = LsdFit; ybc = ybcFit; zbc = zbcFit;
    tyin = ty; tzin = tz;
    p0in = p0; p1in = p1; p2in = p2; p3in = p3;

    // Write checkpoint
    write_checkpoint(rawFN, iter + 1 + iterOffset,
                     LsdFit, ybcFit, zbcFit, ty, tz,
                     p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                     p11, p12, p13, p14, parallaxIn, MeanDiff);

    // Track best iteration
    if (MeanDiff < bestMeanDiff) {
      bestMeanDiff = MeanDiff; bestIter = iter;
      bestLsd = LsdFit; bestYbc = ybcFit; bestZbc = zbcFit;
      bestTy = ty; bestTz = tz;
      bestP0 = p0; bestP1 = p1; bestP2 = p2; bestP3 = p3;
      bestP4 = p4; bestP5 = p5; bestP6 = p6;
      bestP7 = p7; bestP8 = p8; bestP9 = p9; bestP10 = p10;
      bestP11 = p11; bestP12 = p12; bestP13 = p13; bestP14 = p14;
      bestParallax = parallaxIn;
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

    // Convergence early-stop
    double convThresh = ConvergenceThresholdPPM * 1e-6;
    if (convThresh > 0 && iter > 0) {
      double relChange = fabs(MeanDiff - prevPrevIterMeanDiff) / fmax(MeanDiff, 1e-12);
      if (relChange < convThresh)
        convergenceCount++;
      else
        convergenceCount = 0;
      if (convergenceCount >= 3) {
        printf("  [Early stop] Converged: ΔMeanStrain < %.1f ppm for 3 consecutive iters\n",
               ConvergenceThresholdPPM);
        break;
      }
    }

    // Divergence guard
    if (postPerturbGrace > 0) {
      postPerturbGrace--;
    } else if (iter > 0 && bestMeanDiff > 0 && MeanDiff > 1.5 * bestMeanDiff) {
      printf("  [Divergence guard] Reverting to best iter %d\n", bestIter + 1);
      LsdFit = bestLsd; ybcFit = bestYbc; zbcFit = bestZbc;
      ty = bestTy; tz = bestTz;
      p0in = bestP0; p1in = bestP1; p2in = bestP2; p3in = bestP3;
      p4in = bestP4; p5in = bestP5; p6in = bestP6; 
      p7in = bestP7; p8in = bestP8; p9in = bestP9; p10in = bestP10;
      p11in = bestP11; p12in = bestP12; p13in = bestP13; p14in = bestP14;
      parallaxIn = bestParallax;
      Lsd = bestLsd; ybc = bestYbc; zbc = bestZbc;
      tyin = bestTy; tzin = bestTz;
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
      p4in = bestP4; p5in = bestP5; p6in = bestP6;
      p7in = bestP7; p8in = bestP8; p9in = bestP9; p10in = bestP10;
      p11in = bestP11; p12in = bestP12; p13in = bestP13; p14in = bestP14;
      parallaxIn = bestParallax;
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
      p4in = bestP4; p5in = bestP5; p6in = bestP6;
      p7in = bestP7; p8in = bestP8; p9in = bestP9; p10in = bestP10;
      p11in = bestP11; p12in = bestP12; p13in = bestP13; p14in = bestP14;
      parallaxIn = bestParallax;
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
    free(posWeights);
  } // end iteration loop

  // Restore best iteration
  if (nIterations > 1 && bestIter >= 0 && bestIter != nIterations - 1) {
    printf("\n*** Restoring best iter %d (MeanStrain %.3f) ***\n",
           bestIter + 1, bestMeanDiff * 1e6);
    LsdFit = bestLsd; ybcFit = bestYbc; zbcFit = bestZbc;
    ty = bestTy; tz = bestTz;
    p0 = bestP0; p1 = bestP1; p2 = bestP2; p3 = bestP3;
    p4in = bestP4; p5in = bestP5; p6in = bestP6; 
    p7in = bestP7; p8in = bestP8; p9in = bestP9; p10in = bestP10;
    p11in = bestP11; p12in = bestP12; p13in = bestP13; p14in = bestP14;
    parallaxIn = bestParallax;
    MeanDiff = bestMeanDiff;
    if (bestPanels && nPanels > 0)
      memcpy(panels, bestPanels, nPanels * sizeof(Panel));
  }

  // ── Verification E-step: rebuild map with converged params ──
  // The iteration loop builds maps with the ORIGINAL input tilts,
  // and the M-step only re-projects the fixed pixel positions.
  // This final E-step rebuilds everything with the converged geometry
  // to verify the reported strain is consistent.
  if (nIterations > 0 && Yc && !SkipVerification) {
    printf("\n──── Verification E-step (fresh map with converged params) ────\n");
    // Free previous E-step result + per-iter arrays
    estep_free_result(&eResult);
    RMean = EtaMean = YMean = ZMean = NULL;
    IdealTtheta = PointDSpacing = FitSNR = FitFWHM = NULL;
    RingNumbers = NULL; skipBin = NULL;
    free(EtaIns); free(DiffIns); free(RadIns);
    free(Yc); free(Zc);
    EtaIns = DiffIns = RadIns = Yc = Zc = NULL;

    // Run E-step with CONVERGED geometry
    EstepGeometry egv = {
      .Lsd = LsdFit, .ybc = ybcFit, .zbc = zbcFit, .tx = tx, .ty = ty, .tz = tz,
      .p0 = p0, .p1 = p1, .p2 = p2, .p3 = p3,
      .p4 = p4in, .p5 = p5in, .p6 = p6in,
      .p7 = p7in, .p8 = p8in, .p9 = p9in, .p10 = p10in,
      .p11 = p11in, .p12 = p12in, .p13 = p13in, .p14 = p14in,
      .px = px, .MaxRingRad = MaxRingRad, .parallax = parallaxIn,
      .EtaBinSize = EtaBinSize, .RBinWidth = RBinWidth,
      .NrPixelsY = NrPixelsY, .NrPixelsZ = NrPixelsZ, .NrPixels = NrPixels,
      .NrTransOpt = NrTransOpt, .SubPixelLevel = cfg.SubPixelLevel,
      .SubPixelCardinalWidth = cfg.SubPixelCardinalWidth,
      .GradientCorrection = GradientCorrection,
      .peakFitMode = peakFitMode, .DoubletSeparation = DoubletSeparation,
      .Wavelength = Wavelength, .Width = cfg.Width,
      .AdaptiveEtaBins = cfg.AdaptiveEtaBins,
      .residualCorr = residualCorr
    };
    memcpy(egv.TransOpt, TransOpt, sizeof(TransOpt));
    int verifyN = run_estep(&egv, &rings, Average, AverageDark, mask,
                            panels, nPanels, &eResult);
    // Alias result arrays
    RMean = eResult.RMean; EtaMean = eResult.EtaMean;
    YMean = eResult.YMean; ZMean = eResult.ZMean;
    IdealTtheta = eResult.IdealTtheta; PointDSpacing = eResult.PointDSpacing;
    RingNumbers = eResult.RingNumbers; FitSNR = eResult.FitSNR; FitFWHM = eResult.FitFWHM;
    skipBin = eResult.skipBin;

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
          &ctx,
          nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
          zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
          &verifyStd, outlierFactor, verifyOutlier, p4in, p5in, p6in, p7in, p8in, p9in, p10in,
          p11in, p12in, p13in, p14in, OutlierIterations,
          1, &verifyMean, parallaxIn, skipBin, &residualCorr);
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
        &ctx,
        nIndices, MaxRingRad, Yc, Zc, IdealTtheta, px, LsdFit, ybcFit,
        zbcFit, tx, ty, tz, p0, p1, p2, p3, EtaIns, DiffIns, RadIns,
        &StdDiff, outlierFactor, IsOutlier, p4in, p5in, p6in, p7in, p8in, p9in, p10in,
        p11in, p12in, p13in, p14in, OutlierIterations,
        1, &MeanDiff, parallaxIn, skipBin, &residualCorr);
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
    if (p6in != 0.0) printf("p6 %e\n", p6in);
    if (p7in != 0.0) printf("p7 %e\n", p7in);
    if (p8in != 0.0) printf("p8 %e\n", p8in);
    if (p9in != 0.0) printf("p9 %e\n", p9in);
    if (p10in != 0.0) printf("p10 %e\n", p10in);
    if (p11in != 0.0) printf("p11 %e\n", p11in);
    if (p12in != 0.0) printf("p12 %e\n", p12in);
    if (p13in != 0.0) printf("p13 %e\n", p13in);
    if (p14in != 0.0) printf("p14 %e\n", p14in);
    if (FitParallax || parallaxIn != 0.0) printf("Parallax %e\n", parallaxIn);
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
                       "tx,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,Wavelength,Parallax\n");
      fprintf(corrFP, "%.10f,%.10f,%.10f,%.10f,%.10f,%.12e,%.12e,%.12e,%.12e,"
                       "%.12e,%.12e,%.10f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,"
                       "%.12e,%.12e,%.12e,%.12e,%.10f,%.10f\n",
              LsdFit, ybcFit, zbcFit, ty, tz, p0, p1, p2, p3,
              MeanDiff, StdDiff, tx, p4in, p5in, p6in, p7in, p8in, p9in, p10in,
              p11in, p12in, p13in, p14in, Wavelength, parallaxIn);
      // Per-bin data — 16 columns matching CalibrantPanelShiftsOMP format
      fprintf(corrFP, "\n%%Eta Strain RadFit EtaCalc DiffCalc RadCalc "
                       "Ideal2Theta Outlier YRawCorr ZRawCorr RingNr "
                       "RadGlobal IdealR Fit2Theta IdealA FitA DeltaR DeltaA\n");
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
        double deltaR = radFit - idealR;  // microns
        double deltaA = fitA - IdealA;    // angstroms
        fprintf(corrFP, "%.6f %.10e %.6f %.6f %.10e %.6f "
                         "%.6f %d %.6f %.6f %d "
                         "%.6f %.6f %.6f %.10f %.10f %.6f %.10f\n",
                eta, strain, radFit, eta, fabs(strain), radCalc,
                ideal2theta, IsOutlier[i], Yc[i], Zc[i], RingNumbers[i],
                radFit, idealR, fit2theta, IdealA, fitA, deltaR, deltaA);
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
  estep_free_result(&eResult);
  free(EtaIns); free(DiffIns); free(RadIns);
  free(Yc); free(Zc);
  free(Average); free(AverageDark);
  free(mask);
  free(Thetas); free(DSpacings); free(RingIDs);
  free(bestPanels); free(initPanels);

  double end0 = omp_get_wtime();
  printf("\nTotal time: %.2f s\n", end0 - start0);
  return 0;
}
