//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// MIDAS_ParamParser.h — Centralized parameter file parsing for MIDAS executables
//
// Provides:
//   1. key_match()       — whitespace-tolerant key matching
//   2. param_*() helpers — type-safe single-line parsers
//   3. MIDASConfig       — shared config struct for common parameters
//   4. midas_parse_params() — full parser function
//

#ifndef MIDAS_PARAM_PARSER_H
#define MIDAS_PARAM_PARSER_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MIDAS_Limits.h"

// ─── Key matching ────────────────────────────────────────────────────────────

// Returns 1 if `line` starts with `key` followed by whitespace (space, tab, etc.)
static inline int key_match(const char *line, const char *key) {
  size_t klen = strlen(key);
  return strncmp(line, key, klen) == 0 && isspace((unsigned char)line[klen]);
}

// Returns 1 if line should be skipped (blank, or comment starting with #)
static inline int param_skip_line(const char *line) {
  while (*line && isspace((unsigned char)*line)) line++;
  return (*line == '\0' || *line == '#');
}

// ─── Type-safe single-line parsers ──────────────────────────────────────────
// Each returns 1 if key matched (and *out is set), 0 otherwise.

static inline int param_double(const char *line, const char *key, double *out) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  return sscanf(line, "%s %lf", dummy, out) == 2;
}

static inline int param_int(const char *line, const char *key, int *out) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  return sscanf(line, "%s %d", dummy, out) == 2;
}

static inline int param_lld(const char *line, const char *key, long long int *out) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  return sscanf(line, "%s %lld", dummy, out) == 2;
}

static inline int param_str(const char *line, const char *key,
                            char *out, size_t maxlen) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  (void)maxlen; // sscanf will write to out; caller is responsible for sizing
  return sscanf(line, "%s %s", dummy, out) == 2;
}

static inline int param_double2(const char *line, const char *key,
                                double *a, double *b) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  return sscanf(line, "%s %lf %lf", dummy, a, b) == 3;
}

static inline int param_double6(const char *line, const char *key,
                                double out[6]) {
  if (!key_match(line, key)) return 0;
  char dummy[MAX_LINE_LENGTH];
  return sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy,
                &out[0], &out[1], &out[2], &out[3], &out[4], &out[5]) == 7;
}

// ─── Shared Config Struct ───────────────────────────────────────────────────
// Common parameters used by most MIDAS C executables.
// Each executable reads what it needs; unused fields stay at zero/default.
// Calibration-specific fields (from CalibrantPanelShiftsOMP) are included
// so that CalibConfig can eventually be replaced by this struct.

typedef struct {
  // ── File I/O ──
  char FileStem[MAX_LINE_LENGTH];
  char Folder[MAX_LINE_LENGTH];
  char Ext[MAX_LINE_LENGTH];
  char Dark[MAX_LINE_LENGTH];
  int  StartNr, EndNr;
  int  Padding;
  int  HeadSize;
  int  DataType;
  int  SkipFrame;

  // ── HDF5/Zarr dataset names ──
  char darkDatasetName[MAX_LINE_LENGTH];
  char dataDatasetName[MAX_LINE_LENGTH];

  // ── Output / Input paths ──
  char OutputFolder[MAX_LINE_LENGTH];
  char ResultFolder[MAX_LINE_LENGTH];
  char InputFileName[MAX_LINE_LENGTH];   // RefinementFileName, InFileName
  char GrainsFile[MAX_LINE_LENGTH];
  char SeedOrientations[MAX_LINE_LENGTH];
  char DataDirectory[MAX_LINE_LENGTH];
  char GridFileName[MAX_LINE_LENGTH];

  // ── Detector Geometry ──
  int    NrPixelsY, NrPixelsZ, NrPixels;
  double px;
  double Lsd, ybc, zbc;
  double tx, ty, tz;
  double p0, p1, p2, p3, p4, p5;
  double Wedge;
  double RhoD;

  // ── Multi-detector geometry ──
  double DetParams[10][10];
  int    nDetParams;
  int    BigDetSize;

  // ── Crystallography ──
  int    SpaceGroup;
  double LatticeConstant[6];
  double Wavelength;

  // ── Optimization Tolerances ──
  double tolTilts, tolLsd, tolBC, tolP;
  double tolP0, tolP1, tolP2, tolP3, tolP4, tolP5;
  int    tolP0Set, tolP1Set, tolP2Set, tolP3Set, tolP4Set, tolP5Set;
  double tolShifts, tolRotation;
  double tolLsdPanel, tolP2Panel;
  double tolWavelength;
  double tolParallax;

  // ── Image Transforms ──
  int NrTransOpt;
  int TransOpt[10];

  // ── Masking ──
  long long int GapIntensity, BadPxIntensity;
  int    makeMap;
  char   MaskFN[MAX_LINE_LENGTH];
  char   GapFN[MAX_LINE_LENGTH];
  char   BadPxFN[MAX_LINE_LENGTH];

  // ── Omega / Box ranges ──
  int    nOmeRanges;
  double OmegaRanges[MAX_N_OMEGA_RANGES][2];
  int    nBoxSizes;
  double BoxSizes[MAX_N_OMEGA_RANGES][4];
  double OmegaStart, OmegaEnd, OmegaStep;

  // ── Ring selection ──
  int nRingsExclude;
  int RingsExclude[MAX_N_RINGS];
  int nRingThresh;
  int RingThresh[MAX_N_RINGS];
  int MaxRingNumber;
  int nRingNumbers;
  int RingNumbers[MAX_N_RINGS];
  int nRingRadii;
  double RingRadii[MAX_N_RINGS];

  // ── Calibration control ──
  double Width, EtaBinSize;
  int    RBinWidth;
  double outlierFactor;       // MultFactor
  int    MinIndicesForFit;
  int    FitWeightMean;       // FitOrWeightedMean
  int    nIterations;
  double DoubletSeparation;
  int    NormalizeRingWeights;
  int    OutlierIterations;
  int    RemoveOutliersBetweenIters;
  int    ReFitPeaks;
  double TrimmedMeanFraction;
  int    WeightByRadius;
  int    WeightByFitSNR;
  int    L2Objective;
  int    FitWavelength;
  int    FitParallax;
  double parallaxIn;
  int    PerPanelLsd;
  int    PerPanelDistortion;
  int    GradientCorrection;

  // ── FitPos/Strain control ──
  int    TopLayer;
  int    TakeGrainMax;
  int    LocalMaximaOnly;
  double MargABC, MargABG;
  int    DebugMode;
  double OmeBinSize;
  double WeightMask, WeightFitRMSE;
  int    DoDynamicReassignment;

  // ── NF/Sample parameters ──
  double MinEta;
  double Hbeam, Rsample;
  double BeamSize, BeamThickness;

  // ── NF image processing ──
  int    Deblur;
  int    DoLoGFilter;
  int    GaussFiltRadius;
  double GaussWidth;
  int    LoGMaskRadius;
  int    MedFiltRadius;
  int    BlanketSubtraction;
  int    SkipImageBinning;

  // ── NF orientation/grid ──
  double StepSizeOrient;
  int    NrOrientations;
  double EResolution;
  int    nDistances;
  int    NrFilesPerDistance;
  double LsdMean;
  double OrientTol;
  int    Twins;
  double GBAngle;
  double MarginOme, MarginEta;
  double MinConfidence;
  double MinFracAccept;

  // ── ProcessGrains / Forward ──
  int    nScans;
  int    PhaseNr;
  int    NumPhases;
  int    MinNrSpots;
  double GlobalPosition;
  char   OutDirPath[MAX_LINE_LENGTH];
  int    NoSaveAll;
  double GridSize;
  double EdgeLength;
  int    GridPoints;
  int    WriteLegacyBin;
  int    SimulationBatches;

  // ── Panel config ──
  int  NPanelsY, NPanelsZ;
  int  PanelSizeY, PanelSizeZ;
  int  PanelGapsY[10], PanelGapsZ[10];
  char PanelShiftsFile[MAX_LINE_LENGTH];
  int  FixPanelID;

  // ── Lineout parameters (integrator) ──
  double lineoutRBinSize;
  double lineoutRMin;
  double lineoutRMax;

  // ── ForwardSimulation ──
  // Note: InFileName is stored in InputFileName above (shared with RefinementFileName)
  char   OutFileName[MAX_LINE_LENGTH];
  char   IntensitiesFile[MAX_LINE_LENGTH];
  char   MaskFile[MAX_LINE_LENGTH];
  int    WriteSpots;
  int    WriteImage;
  int    IsBinary;
  int    LoadNr;
  int    UpdatedOrientations;
  int    NFOutput;
  double PeakIntensity;
  double MaxOutputIntensity;
  int    RingsToUse[500];
  int    nRingsToUse;
  int    num_lambda_samples;
  int    useMask;

} MIDASConfig;

// Parse a standard MIDAS parameter file into a MIDASConfig struct.
// Returns 0 on success, -1 on file-open error.
// Unrecognized keys are silently skipped.
int midas_parse_params(const char *filename, MIDASConfig *cfg);

// Set sensible defaults for a MIDASConfig struct (called by midas_parse_params,
// but available for manual use too).
void midas_config_defaults(MIDASConfig *cfg);

// Apply tolerance fallbacks (tolP0..tolP5 fall back to tolP if not explicitly set).
void midas_apply_tol_defaults(MIDASConfig *cfg);

#endif // MIDAS_PARAM_PARSER_H
