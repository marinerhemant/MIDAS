//
// CalibrantEstep.h — Data structures for CalibrantIntegratorOMP E-step
//
// Defines input/output structs that replace the 33+ function parameters
// of run_estep, enabling factoring into testable sub-functions.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef CALIBRANT_ESTEP_H
#define CALIBRANT_ESTEP_H

#include <stdlib.h>
#include <string.h>

// ── Input geometry for E-step map building ────────────────────────

typedef struct {
    double Lsd, ybc, zbc, tx, ty, tz;
    double p0, p1, p2, p3, p4, p5, p6;
    double px, MaxRingRad, parallax;
    double EtaBinSize;
    int RBinWidth;
    int NrPixelsY, NrPixelsZ, NrPixels;
    int NrTransOpt;
    int TransOpt[10];
    int SubPixelLevel;
    double SubPixelCardinalWidth;
    int peakFitMode;
    double DoubletSeparation;
    double Wavelength;
    double Width;
} EstepGeometry;

// ── Ring info (from hkls.csv) ─────────────────────────────────────

typedef struct {
    int n_hkls;
    double *Thetas;
    double *DSpacings;
    int *RingIDs;
} RingTable;

// ── Per-ring R bin edge layout ────────────────────────────────────

typedef struct {
    int n_hkls;         // number of rings
    int totalRBins;     // total R bins across all rings
    int nEtaBins;       // number of eta bins
    double rBinSize;    // size of each R sub-bin (pixels)
    double *RBinsLow;   // [totalRBins] lower edge
    double *RBinsHigh;  // [totalRBins] upper edge
    double *EtaBinsLow; // [nEtaBins] lower edge
    double *EtaBinsHigh;// [nEtaBins] upper edge
    int *ringRBinStart; // [n_hkls] index into global R bins
    int *ringNRBins;    // [n_hkls] count of R bins per ring
    double *ringRLo;    // [n_hkls] lower R bound per ring
    double *ringRHi;    // [n_hkls] upper R bound per ring
} BinEdges;

// ── Map + integrated profiles ─────────────────────────────────────

typedef struct {
    long long nTotalBins;
    long long nEntries;         // total pixel-bin entries
    double *profiles;           // [nTotalBins] normalized intensity
    double *norm;               // [nTotalBins] area weight
    int *binMaskFlag;           // [nTotalBins] mask contamination flag
} ProfileData;

// ── Peak fitting results (before compaction) ──────────────────────

typedef struct {
    int maxBins;        // n_hkls * nEtaBins
    int nValid;         // number of valid (non-zero) bins
    double *RMean;      // [maxBins] fitted R in microns (0 = invalid)
    double *EtaMean;    // [maxBins] eta center
    double *IdealTtheta;// [maxBins] ideal 2-theta
    double *PointDSpacing; // [maxBins] d-spacing
    int *RingNumbers;   // [maxBins] ring ID
    double *FitSNR;     // [maxBins] fit signal-to-noise
    int *skipBin;       // [maxBins] skip flag
} PeakData;

// ── Output of E-step (compacted per-bin data) ─────────────────────

typedef struct {
    int nValid;
    double *RMean, *EtaMean;
    double *YMean, *ZMean;          // raw pixel coords for M-step
    double *IdealTtheta, *PointDSpacing;
    int *RingNumbers;
    double *FitSNR;
    int *skipBin;
} EstepResult;

// ── Lifecycle ─────────────────────────────────────────────────────

void free_bin_edges(BinEdges *b);
void free_profile_data(ProfileData *p);
void free_peak_data(PeakData *p);
void estep_free_result(EstepResult *r);

#endif /* CALIBRANT_ESTEP_H */
