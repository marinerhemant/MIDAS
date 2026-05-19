/*
 * Copyright (c) 2014, UChicago Argonne, LLC
 * See LICENSE file.
 *
 * IndexerUnified.c
 *
 * Unified MIDAS indexer (FF + PF) — supersedes IndexerOMP.c and
 * IndexerScanningOMP.c. Implements the approved plan in
 *   packages/midas_index/dev/implementation_plan_unified_c_indexer.md
 *
 * Two algorithmic modes share a single forward+match kernel:
 *   PF mode (nScans > 1): voxel grid loop, position fixed at voxel center,
 *     per-spot scan-position filter inside CompareSpots, multi-solution
 *     consolidated output per voxel.
 *   FF mode (nScans == 1): one "voxel" per spot-to-index, per-seed Friedel-pair
 *     plane-normal generation, adaptive 1D position-grid search with stride
 *     `nDelta`, FF-only RefRad/MarginRad gate, single best solution per spot
 *     written through the same consolidated accumulator path.
 *
 * Auto-detect: PF iff `positions.csv` exists in `dirname(OutputFolder)` AND
 * nWork > 1.
 *
 * IMPORTANT: The legacy sources IndexerOMP.c and IndexerScanningOMP.c stay
 * in tree (built as separate targets) and serve as the parity ground truth.
 *
 * Author: Hemant Sharma / AI Assistant. Functions copied verbatim from the
 * two legacy binaries are annotated inline.
 */

#include "midas_version.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "MIDAS_Math.h"
#include "IndexerConsolidatedIO.h"
#include "GetMisorientation.h"

/* ----------------------------------------------------------------------------
 * Compile-time switches.
 * --------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
 * check() — same as both legacy files. MIDAS_CHECK_DEFINED guard required so
 * MIDAS_Math.h / MIDAS_Limits.h consumers don't redefine it.
 * --------------------------------------------------------------------------*/
#ifndef MIDAS_CHECK_DEFINED
#define MIDAS_CHECK_DEFINED
static inline void check(int test, const char *message, ...) {
  if (test) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}
#endif

#define RealType double

/* conversions */
#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)

/* Plan ruling #1: take the upper bound of each constant. */
#define MAX_N_SPOTS 100000000
#define MAX_N_STEPS 1000
#define MAX_N_OR 36000
#define MAX_N_MATCHES 1
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_OMEGARANGES 2000
#define MAX_MIC_ROWS 50000000
#define MAX_N_DETS 32            /* Phase 9: pinwheel/hydra panel cap */
#define MAX_ETA_ARCS_PER_DET 256 /* up to N arcs per panel (eta-coverage list) */
#define N_COL_THEORSPOTS 16   /* PF width: includes sinOme/cosOme cols 14,15 */
#define N_COL_OBSSPOTS 10     /* PF width: includes ScanNr col 9 */
#define N_COL_GRAINSPOTS 17
#define N_COL_GRAINMATCHES 16

/* Math macros (verbatim from both legacy sources). */
#define crossProduct(a, b, c)                                                  \
  (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];                                  \
  (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];                                  \
  (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

#define dot(v, q) ((v)[0] * (q)[0] + (v)[1] * (q)[1] + (v)[2] * (q)[2])

#define CalcLength(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

/* ----------------------------------------------------------------------------
 * TParams: union of FF and PF parameter structs (plan ruling #1 + #13).
 * --------------------------------------------------------------------------*/
struct TParams {
  int RingNumbers[MAX_N_RINGS];
  int SpaceGroupNum;
  RealType LatticeConstant;
  RealType Wavelength;
  RealType Distance;
  RealType Rsample;
  RealType Hbeam;
  RealType StepsizePos;
  RealType StepsizeOrient;
  int NrOfRings;
  RealType RingRadii[MAX_N_RINGS];
  RealType RingRadiiUser[MAX_N_RINGS];
  RealType MarginOme;
  RealType MarginEta;
  RealType MarginRad;
  RealType MarginRadial;
  RealType EtaBinSize;
  RealType OmeBinSize;
  RealType InvEtaBinSize;        /* PF hot-loop optimization */
  RealType InvOmeBinSize;
  RealType ExcludePoleAngle;
  RealType MinMatchesToAcceptFrac;
  RealType BoxSizes[MAX_N_OMEGARANGES][4];
  RealType OmegaRanges[MAX_N_OMEGARANGES][2];
  char OutputFolder[4096];
  char SpotsFileName[4096];
  char IDsFileName[4096];
  char MicFN[4096];              /* PF seeded path */
  char GrainsFN[4096];           /* PF grains-seeded path */
  char GrainsFileName[4096];     /* FF GrainsFile path (sets isGrainsInput) */
  int isGrainsInput;             /* FF GrainsFile semantics */
  int NoOfOmegaRanges;
  int UseFriedelPairs;
  int RingsToReject[MAX_N_RINGS];
  int nRingsToRejectCalc;
  int RingToIndex;               /* PF seed-ring restriction */
  RealType ScanPosTol;           /* PF override for BeamSize/2 */
  /* Soft beam attribution (Phase 8). SoftAttrMode:
   *   0 = none / hard window (legacy scan_pos_tol_um filter, default)
   *   1 = top_hat   (trapezoidal: 1 inside SoftAttrFwhm/2, linear ramp
   *                  over SoftAttrFalloff past edge)
   *   2 = gaussian  (peak-1 Gaussian, σ = FWHM/(2√(2 ln 2)),
   *                  zero past SoftAttrTruncate when >0)
   * Per-match weight is written to the IndexBest_weights_all.bin
   * sidecar, 1:1 with IndexBest_IDs_all.bin entries. Mode 0 emits
   * all-1.0 weights — file is always produced. */
  int SoftAttrMode;
  RealType SoftAttrFwhm;
  RealType SoftAttrFalloff;
  RealType SoftAttrTruncate;
  /* Multi-detector / pinwheel scaffolding (Phase 9). For single-detector
   * runs nDetParams stays 0 and the indexer falls back to global geometry
   * (RingRadii[], Distance) — preserves PF / FF parity bit-identically.
   *
   * Algorithmic per-panel forward simulation + per-panel η-coverage
   * gating in CalcDiffrSpots / CompareSpots is NOT YET implemented:
   * see "Phase 9 TODO" markers in those functions for the plug-in
   * points. Parsing + Spots_det.bin loading are wired so a multi-det
   * paramstest no longer warn-skips, and the data structure is ready
   * for the algorithm port when a multi-det fixture lands. */
  int nDetParams;                                /* # active panels */
  int DetIDs[MAX_N_DETS];                        /* per-slot det_id */
  RealType DetLsd[MAX_N_DETS];
  RealType DetYbc[MAX_N_DETS];
  RealType DetZbc[MAX_N_DETS];
  RealType DetTx[MAX_N_DETS];
  RealType DetTy[MAX_N_DETS];
  RealType DetTz[MAX_N_DETS];
  RealType DetDistortion[MAX_N_DETS][11];
  /* Per-(panel, ring) radius. 0 ⇒ unmapped (use global RingRadii). */
  RealType RingRadiiPerDet[MAX_N_DETS][MAX_N_RINGS];
  int hasRingRadiiPerDet;
  /* η-coverage arcs per panel: list of (ring_nr, eta_lo_deg, eta_hi_deg). */
  int EtaArcRing[MAX_N_DETS][MAX_ETA_ARCS_PER_DET];
  RealType EtaArcLo[MAX_N_DETS][MAX_ETA_ARCS_PER_DET];
  RealType EtaArcHi[MAX_N_DETS][MAX_ETA_ARCS_PER_DET];
  int nEtaArcs[MAX_N_DETS];
  int hasEtaCoverage;
};

/* ----------------------------------------------------------------------------
 * Module-level globals (kept compatible with the legacy structure).
 * --------------------------------------------------------------------------*/
RealType *ObsSpotsLab;
size_t n_spots = 0;             /* PF type (size_t) — superset */
/* Phase 9: per-spot DetID side-car (Spots_det.bin). NULL ⇒ single-det run
 * or file absent (treated as all det_id=1, matching midas-transforms's
 * default emit). When non-NULL, length == n_spots, dtype int32. */
int32_t *SpotsDetID = NULL;
size_t SpotsDetID_size = 0;

/* hkls in PF format (10 cols: H K L Rnr Ds tht RRd sin(tht) v vSq). */
double hkls[MAX_N_HKLS][10];
int n_hkls = 0;
int HKLints[MAX_N_HKLS][4];
double ABCABG[6];
double RingHKL[MAX_N_RINGS][3];
RealType RingTtheta[MAX_N_RINGS]; /* used by FF Friedel-pair generation */

double pixelsize;
double BeamSize = 0.0;
int numScans = 1;

/* int64 bins always (plan ruling #10). */
size_t *data;
size_t *ndata;
int SGNum;
double *grid = NULL;
double *ypos = NULL;

int n_ring_bins;
int n_eta_bins;
int n_ome_bins;

RealType EtaBinSize = 0;
RealType OmeBinSize = 0;

/* Etamargin table (per-ring). Computed once in main. */
RealType etamargins[MAX_N_RINGS];

/* ----------------------------------------------------------------------------
 * IndexerScratch — per-thread workspace, PF-style.
 * --------------------------------------------------------------------------*/
struct IndexerScratch {
  RealType **GrainMatches;
  RealType **GrainMatchesT;
  RealType **AllGrainSpots;
  RealType **AllGrainSpotsT;
  RealType **GrainSpots;
  RealType **TheorSpots;
  RealType *IAgrainspots;
  /* OrMat heap-allocated per-thread (plan risk-register item). MAX_N_OR=36000
   * × [3][3] doubles = 2.6 MB per thread — too large for stack. */
  RealType (*OrMat)[3][3];
};

/* ----------------------------------------------------------------------------
 * Forward declarations.
 * --------------------------------------------------------------------------*/
static int ReadParams(char FileName[], struct TParams *Params);
static int ReadBins(char *cwd);
static int ReadSpots(char *cwd);

/* ----------------------------------------------------------------------------
 * Comparators / helpers.
 * --------------------------------------------------------------------------*/

/* cmp_double_asc — from PF (line 146). */
static int cmp_double_asc(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

/* FindInMatrix — identical FF (line 138) / PF (line 168). */
static void FindInMatrix(RealType *aMatrixp, int nrows, int ncols,
                         int SearchColumn, RealType aVal, int *idx) {
  int r, LIndex;
  *idx = -1;
  for (r = 0; r < nrows; r++) {
    LIndex = (r * ncols) + SearchColumn;
    if (aMatrixp[LIndex] == aVal) {
      *idx = r;
      break;
    }
  }
}

/* allocMatrix — PF style (contiguous, single block). FF "allocMatrix" was
 * row-wise; PF's contiguous form is the better choice (plan §2 alloc). */
static RealType **allocMatrix(int nrows, int ncols) {
  RealType **arr = (RealType **)malloc(nrows * sizeof(*arr));
  if (arr == NULL) return NULL;
  RealType *block = (RealType *)calloc((size_t)nrows * ncols, sizeof(RealType));
  if (block == NULL) {
    free(arr);
    return NULL;
  }
  for (int i = 0; i < nrows; i++) arr[i] = &block[i * ncols];
  return arr;
}

static void FreeMemMatrix(RealType **mat, int nrows) {
  (void)nrows;
  if (mat != NULL) {
    if (mat[0] != NULL) free(mat[0]);
    free(mat);
  }
}

static RealType minR(RealType a, RealType b) { return (a < b ? a : b); }
static RealType maxR(RealType a, RealType b) { return (a > b ? a : b); }

/* CalcInternalAngle — identical FF (253) / PF (218). */
static void CalcInternalAngle(RealType x1, RealType y1, RealType z1, RealType x2,
                              RealType y2, RealType z2, RealType *ia) {
  RealType v1[3] = {x1, y1, z1};
  RealType v2[3] = {x2, y2, z2};
  RealType l1 = CalcLength(x1, y1, z1);
  RealType l2 = CalcLength(x2, y2, z2);
  RealType tmp = dot(v1, v2) / (l1 * l2);
  if (tmp > 1) tmp = 1;
  if (tmp < -1) tmp = -1;
  *ia = rad2deg * acos(tmp);
}

/* CalcSpotPosition — identical FF (275) / PF (240). */
static void CalcSpotPosition(RealType RingRadius, RealType eta, RealType *yl,
                             RealType *zl) {
  RealType etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

/* CalcOmega — PF signature (line 247): takes precomputed v, vSq AND outputs
 * cosOmes[]/sinOmes[]. FF callers (Friedel-pair helpers) wrap with their own
 * theta input via CalcOmegaFF (below). */
static void CalcOmega(RealType x, RealType y, RealType z, RealType v,
                      RealType vSq, RealType omegas[4], RealType etas[4],
                      RealType cosOmes[4], RealType sinOmes[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  RealType almostzero = 1e-12;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * rad2deg;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        omegas[*nsol] = ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = sinome1;
        *nsol = *nsol + 1;
        omegas[*nsol] = -ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = -sinome1;
        *nsol = *nsol + 1;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType inv_y2 = 1.0 / y2;
    RealType a = 1 + ((x * x) * inv_y2);
    RealType b = (2 * v * x) * inv_y2;
    RealType c = (vSq * inv_y2) - 1;
    RealType discr = b * b - 4 * a * c;
    RealType ome1a;
    RealType ome1b;
    RealType ome2a;
    RealType ome2b;
    RealType cosome1;
    RealType cosome2;
    RealType eqa, eqb, diffa, diffb;
    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1.0) {
        ome1a = acos(cosome1);
        ome1b = -ome1a;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        eqa = -x * cosome1 + y * sinome1;
        diffa = fabs(eqa - v);
        eqb = -x * cosome1 - y * sinome1;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome1a * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = sinome1;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome1b * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = -sinome1;
          *nsol = *nsol + 1;
        }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2);
        ome2b = -ome2a;
        RealType sinome2 = sqrt(1 - cosome2 * cosome2);
        eqa = -x * cosome2 + y * sinome2;
        diffa = fabs(eqa - v);
        eqb = -x * cosome2 - y * sinome2;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome2a * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = sinome2;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome2b * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = -sinome2;
          *nsol = *nsol + 1;
        }
      }
    }
  }
  RealType gw[3];
  RealType gv[3] = {x, y, z};
  RealType eta;
  int indexOme;
  for (indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

/* CalcOmegaFF — FF-style entry point used inside FF-only Friedel-pair helpers
 * (GenerateIdealSpotsFriedelMixed). Computes (v, vSq) from theta-degrees and
 * forwards to the unified CalcOmega. Output etas/cosOmes/sinOmes still
 * populated. Discards cosOmes/sinOmes (caller doesn't need them). */
static void CalcOmegaFF(RealType x, RealType y, RealType z, RealType theta,
                        RealType omegas[4], RealType etas[4], int *nsol) {
  RealType len = sqrt(x * x + y * y + z * z);
  RealType v = sin(theta * deg2rad) * len;
  RealType vSq = v * v;
  RealType cosOmes[4], sinOmes[4];
  CalcOmega(x, y, z, v, vSq, omegas, etas, cosOmes, sinOmes, nsol);
}

/* AxisAngle2RotMatrix — identical FF (565) / PF (524). */
static void AxisAngle2RotMatrix(RealType axis[3], RealType angle,
                                RealType R[3][3]) {
  if ((axis[0] == 0) && (axis[1] == 0) && (axis[2] == 0)) {
    R[0][0] = 1; R[1][0] = 0; R[2][0] = 0;
    R[0][1] = 0; R[1][1] = 1; R[2][1] = 0;
    R[0][2] = 0; R[1][2] = 0; R[2][2] = 1;
    return;
  }
  RealType lenInv =
      1 / sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  RealType u = axis[0] * lenInv;
  RealType v = axis[1] * lenInv;
  RealType w = axis[2] * lenInv;
  RealType angleRad = deg2rad * angle;
  RealType rcos = cos(angleRad);
  RealType rsin = sin(angleRad);
  R[0][0] = rcos + u * u * (1 - rcos);
  R[1][0] = w * rsin + v * u * (1 - rcos);
  R[2][0] = -v * rsin + w * u * (1 - rcos);
  R[0][1] = -w * rsin + u * v * (1 - rcos);
  R[1][1] = rcos + v * v * (1 - rcos);
  R[2][1] = u * rsin + w * v * (1 - rcos);
  R[0][2] = v * rsin + u * w * (1 - rcos);
  R[1][2] = -u * rsin + v * w * (1 - rcos);
  R[2][2] = rcos + w * w * (1 - rcos);
}

/* CalcRotationAngle — identical FF (597) / PF (556). */
static double CalcRotationAngle(int RingNr) {
  int habs = 0, kabs = 0, labs = 0;
  int i;
  for (i = 0; i < MAX_N_HKLS; i++) {
    if (HKLints[i][3] == RingNr) {
      habs = abs(HKLints[i][0]);
      kabs = abs(HKLints[i][1]);
      labs = abs(HKLints[i][2]);
      break;
    }
  }
  int nzeros = 0;
  if (habs == 0) nzeros++;
  if (kabs == 0) nzeros++;
  if (labs == 0) nzeros++;
  if (nzeros == 3) return 0;
  if (SGNum == 1 || SGNum == 2) {
    return 360;
  } else if (SGNum >= 3 && SGNum <= 15) {
    if (nzeros != 2) return 360;
    else if (ABCABG[3] == 90 && ABCABG[4] == 90 && labs != 0) return 180;
    else if (ABCABG[3] == 90 && ABCABG[5] == 90 && habs != 0) return 180;
    else if (ABCABG[3] == 90 && ABCABG[5] == 90 && kabs != 0) return 180;
    else return 360;
  } else if (SGNum >= 16 && SGNum <= 74) {
    if (nzeros != 2) return 360;
    else return 180;
  } else if (SGNum >= 75 && SGNum <= 142) {
    if (nzeros == 0) return 360;
    else if (nzeros == 1 && labs == 0 && habs == kabs) return 180;
    else if (nzeros == 2) {
      if (labs == 0) return 180;
      else return 90;
    } else return 360;
  } else if (SGNum >= 143 && SGNum <= 167) {
    if (nzeros == 0) return 360;
    else if (nzeros == 2 && labs != 0) return 120;
    else return 360;
  } else if (SGNum >= 168 && SGNum <= 194) {
    if (nzeros == 2 && labs != 0) return 60;
    else return 360;
  } else if (SGNum >= 195 && SGNum <= 230) {
    if (nzeros == 2) return 90;
    else if (nzeros == 1) {
      if (habs == kabs || kabs == labs || habs == labs) return 180;
    } else if (habs == kabs && kabs == labs) return 120;
    else return 360;
  } else
    return 0;
  return 0;
}

/* GenerateCandidateOrientationsF — PF 3D OrMat layout (line 633).
 * Same math as FF; FF used flat OrMat[or*9 + row*3 + col]. */
static int GenerateCandidateOrientationsF(double hkl[3], RealType hklnormal[3],
                                          RealType stepsize,
                                          RealType OrMat[][3][3], int *nOrient,
                                          int RingNr) {
  RealType v[3];
  RealType MaxAngle = 0;
  crossProduct(v, hkl, hklnormal);
  RealType hkllen = sqrt(hkl[0] * hkl[0] + hkl[1] * hkl[1] + hkl[2] * hkl[2]);
  RealType hklnormallen =
      sqrt(hklnormal[0] * hklnormal[0] + hklnormal[1] * hklnormal[1] +
           hklnormal[2] * hklnormal[2]);
  RealType dotpr = dot(hkl, hklnormal);
  RealType angled = rad2deg * acos(dotpr / (hkllen * hklnormallen));
  RealType RotMat[3][3];
  RealType RotMat2[3][3];
  RealType RotMat3[3][3];
  AxisAngle2RotMatrix(v, angled, RotMat);
  MaxAngle = CalcRotationAngle(RingNr);
  RealType nsteps = (MaxAngle / stepsize);
  int nstepsi = (int)nsteps;
  int or_;
  int row, col;
  RealType angle2;
  for (or_ = 0; or_ < nstepsi; or_++) {
    angle2 = or_ * stepsize;
    AxisAngle2RotMatrix(hklnormal, angle2, RotMat2);
    MatrixMultF33(RotMat2, RotMat, RotMat3);
    for (row = 0; row < 3; row++) {
      for (col = 0; col < 3; col++) {
        OrMat[or_][row][col] = RotMat3[row][col];
      }
    }
  }
  *nOrient = nstepsi;
  return 0;
}

/* displacement_spot_needed_COM — PF signature (precomputed sin/cosOme).
 * FF version computed sin/cos inline; PF avoids the redundant trig. */
static void displacement_spot_needed_COM(RealType a, RealType b, RealType c,
                                         RealType xi, RealType yi, RealType zi,
                                         RealType sinOme, RealType cosOme,
                                         RealType *Displ_y, RealType *Displ_z) {
  RealType lenInv = 1 / sqrt(xi * xi + yi * yi + zi * zi);
  xi = xi * lenInv;
  yi = yi * lenInv;
  zi = zi * lenInv;
  RealType t = (a * cosOme - b * sinOme) / xi;
  *Displ_y = ((a * sinOme) + (b * cosOme)) - (t * yi);
  *Displ_z = c - t * zi;
}

/* FF-only inline helper: takes degrees, computes sin/cos and dispatches.
 * Used by FF inner position-search loop (omega comes from the seed spot,
 * not the theor-spot's precomputed sin/cos which is the per-theor-spot
 * dose). */
static void displacement_spot_needed_COM_omega(RealType a, RealType b,
                                               RealType c, RealType xi,
                                               RealType yi, RealType zi,
                                               RealType omega,
                                               RealType *Displ_y,
                                               RealType *Displ_z) {
  RealType OmegaRad = deg2rad * omega;
  RealType sinOme = sin(OmegaRad);
  RealType cosOme = cos(OmegaRad);
  displacement_spot_needed_COM(a, b, c, xi, yi, zi, sinOme, cosOme, Displ_y,
                                Displ_z);
}

/* spot_to_gv — identical FF (726) / PF (685). */
static void spot_to_gv(RealType xi, RealType yi, RealType zi, RealType Omega,
                       RealType *g1, RealType *g2, RealType *g3) {
  RealType len = sqrt(xi * xi + yi * yi + zi * zi);
  if (len == 0) {
    *g1 = 0; *g2 = 0; *g3 = 0;
    printf("len o!\n");
    return;
  }
  RealType xn = xi / len;
  RealType yn = yi / len;
  RealType zn = zi / len;
  RealType g1r = (-1 + xn);
  RealType g2r = yn;
  RealType CosOme = cos(-Omega * deg2rad);
  RealType SinOme = sin(-Omega * deg2rad);
  *g1 = g1r * CosOme - g2r * SinOme;
  *g2 = g1r * SinOme + g2r * CosOme;
  *g3 = zn;
}

/* spot_to_gv_pos — identical FF (748) / PF (707). */
static void spot_to_gv_pos(RealType xi, RealType yi, RealType zi,
                           RealType Omega, RealType cx, RealType cy,
                           RealType cz, RealType *g1, RealType *g2,
                           RealType *g3) {
  RealType v[3] = {cx, cy, cz};
  RealType vr[3];
  RotateAroundZ(v, Omega, vr);
  xi = xi - vr[0];
  yi = yi - vr[1];
  zi = zi - vr[2];
  spot_to_gv(xi, yi, zi, Omega, g1, g2, g3);
}

/* AddUnique — identical FF (1081) / PF (721). */
static int AddUnique(int *arr, int *n, int val) {
  int i;
  for (i = 0; i < *n; ++i)
    if (arr[i] == val) return 0;
  arr[*n] = val;
  (*n)++;
  return 1;
}

/* MakeUnitLength — identical FF (1093) / PF (733). */
static void MakeUnitLength(RealType x, RealType y, RealType z, RealType *xu,
                           RealType *yu, RealType *zu) {
  RealType len = CalcLength(x, y, z);
  if (len == 0) {
    *xu = 0; *yu = 0; *zu = 0;
    return;
  }
  *xu = x / len;
  *yu = y / len;
  *zu = z / len;
}

/* MakeFullFileName — identical FF (1711) / PF (1064). */
static void MakeFullFileName(char *fullFileName, char *aPath, char *aFileName) {
  if (aPath[0] == '\0') {
    strcpy(fullFileName, aFileName);
  } else {
    strcpy(fullFileName, aPath);
    strcat(fullFileName, "/");
    strcat(fullFileName, aFileName);
  }
}

/* CalcAvgIA — identical FF (1548) / PF (1008). */
static RealType CalcAvgIA(RealType *Arr, int n) {
  RealType total = 0;
  int nnum = 0;
  int i;
  for (i = 0; i < n; i++) {
    if (Arr[i] == 999) continue;
    total = total + fabs(Arr[i]);
    nnum++;
  }
  if (nnum == 0) return 0;
  else return total / nnum;
}

/* CalcIA — PF signature (with scratch buffer, no debug). */
static void CalcIA(RealType **GrainMatches, int ngrains,
                   RealType **AllGrainSpots, RealType distance,
                   struct IndexerScratch *scratch) {
  RealType *IAgrainspots = scratch->IAgrainspots;
  int r, g;
  RealType g1x, g1y, g1z;
  RealType x1, y1, z1, w1, x2, y2, z2, w2, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
  int nspots;
  int rt = 0;
  for (g = 0; g < ngrains; g++) {
    nspots = GrainMatches[g][12];
    for (r = 0; r < nspots; r++) {
      if (AllGrainSpots[rt][0] < 0) {
        AllGrainSpots[rt][16] = 999;
        IAgrainspots[r] = AllGrainSpots[rt][16];
        rt++;
        continue;
      }
      x1 = distance;
      x2 = distance;
      y1 = AllGrainSpots[rt][2];
      y2 = AllGrainSpots[rt][3];
      z1 = AllGrainSpots[rt][5];
      z2 = AllGrainSpots[rt][6];
      w1 = AllGrainSpots[rt][8];
      w2 = AllGrainSpots[rt][9];
      g1x = GrainMatches[g][9];
      g1y = GrainMatches[g][10];
      g1z = GrainMatches[g][11];
      spot_to_gv_pos(x1, y1, z1, w1, g1x, g1y, g1z, &gv1x, &gv1y, &gv1z);
      spot_to_gv_pos(x2, y2, z2, w2, g1x, g1y, g1z, &gv2x, &gv2y, &gv2z);
      CalcInternalAngle(gv1x, gv1y, gv1z, gv2x, gv2y, gv2z,
                        &AllGrainSpots[rt][16]);
      IAgrainspots[r] = AllGrainSpots[rt][16];
      rt++;
    }
    GrainMatches[g][15] = CalcAvgIA(IAgrainspots, nspots);
  }
}

/* ----------------------------------------------------------------------------
 * CalcDiffrSpots — unified forward simulator.
 * Uses PF body (sin/cosOme written into cols 14, 15) plus FF's RingsToReject
 * fraction-tally semantics (plan ruling #3, ruling #4).
 *
 * TODO (Phase 9b — multi-detector algorithm port): when
 * `Params.nDetParams > 0`, loop over panels [0, nDetParams):
 *   1. Use `Params.DetLsd[k]` instead of `distance`.
 *   2. Use `Params.RingRadiiPerDet[k]` instead of `RingRadii` (with
 *      fallback to the global table for unmapped rings — see
 *      ParamsTest.panel_ring_radius() in the Python ref).
 *   3. For each `KeepSpot`, also gate by the panel's η-coverage arcs
 *      (Params.EtaArcRing[k] / EtaArcLo[k] / EtaArcHi[k]).
 *   4. Tag the emitted spot with its panel id (e.g. spots[*][16] —
 *      requires N_COL_THEORSPOTS = 17 and the matching reads in
 *      CompareSpots / DoIndexing_*).
 * Single-panel path stays the default (this function as-is) when
 * nDetParams == 0 — preserves current parity bit-for-bit.
 *
 * Scaffolding (Phase 9a, this commit): params are PARSED and stored in
 * Params.DetIDs[] / DetLsd[] / RingRadiiPerDet[] / EtaArc*[] but NOT
 * consumed by the algorithm. A multi-det paramstest runs as single-det
 * (uses global geometry); the warning suppression is the only
 * observable change.
 * --------------------------------------------------------------------------*/
static void CalcDiffrSpots(RealType OrientMatrix[3][3], RealType LatticeConstant,
                           RealType Wavelength, RealType distance,
                           RealType RingRadii[],
                           RealType OmegaRange[][2], RealType BoxSizes[][4],
                           int NOmegaRanges, RealType ExcludePoleAngle,
                           RealType **spots, int *nspots,
                           const int ringsToReject[], int nRingsToReject,
                           int *nSpotsFracCalc) {
  (void)LatticeConstant;
  (void)Wavelength;
  int i, OmegaRangeNo;
  RealType theta;
  int KeepSpot;
  double Ghkl[3];
  int indexhkl;
  RealType Gc[3];
  RealType omegas[4];
  RealType cosOmes[4];
  RealType sinOmes[4];
  RealType etas[4];
  RealType yl;
  RealType zl;
  int nspotsPlane;
  int spotnr = 0;
  int spotid = 0;
  int OrientID = 0;
  int ringnr = 0;
  int nSpotsForFracCalc = 0;
  int SpotToFrac, ringFracCntr;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    ringnr = (int)(hkls[indexhkl][3]);
    SpotToFrac = 1;
    for (ringFracCntr = 0; ringFracCntr < nRingsToReject; ringFracCntr++) {
      if (ringnr == ringsToReject[ringFracCntr]) {
        SpotToFrac = 0;
        break;
      }
    }
    RealType RingRadius = RingRadii[ringnr];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = hkls[indexhkl][5];
    /* PF CalcOmega: precomputed v=hkls[*][8], vSq=hkls[*][9]. */
    CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl][8], hkls[indexhkl][9], omegas,
              etas, cosOmes, sinOmes, &nspotsPlane);
    for (i = 0; i < nspotsPlane; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      RealType EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle))
        continue;
      CalcSpotPosition(RingRadius, etas[i], &(yl), &(zl));
      for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
        KeepSpot = 0;
        if ((Omega > OmegaRange[OmegaRangeNo][0]) &&
            (Omega < OmegaRange[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) &&
            (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) &&
            (zl < BoxSizes[OmegaRangeNo][3])) {
          KeepSpot = 1;
          break;
        }
      }
      if (KeepSpot) {
        spots[spotnr][0] = OrientID;
        spots[spotnr][1] = spotid;
        spots[spotnr][2] = indexhkl;
        spots[spotnr][3] = distance;
        spots[spotnr][4] = yl;
        spots[spotnr][5] = zl;
        spots[spotnr][6] = omegas[i];
        spots[spotnr][7] = etas[i];
        spots[spotnr][8] = theta;
        spots[spotnr][9] = ringnr;
        spots[spotnr][14] = sinOmes[i];
        spots[spotnr][15] = cosOmes[i];
        spotnr++;
        spotid++;
        if (SpotToFrac) nSpotsForFracCalc++;
      }
    }
  }
  *nspots = spotnr;
  *nSpotsFracCalc = nSpotsForFracCalc;
}

/* ----------------------------------------------------------------------------
 * CompareSpots — UNIFIED.
 *
 * TODO (Phase 9b — multi-detector algorithm port): when
 * `Params->nDetParams > 0` and the global `SpotsDetID` array is loaded,
 * gate each obs candidate row by panel: skip iff
 * `SpotsDetID[spotRow] != TheorSpots[sp][16]` (the theor spot's emitting
 * panel, tagged in CalcDiffrSpots). Per-panel MarginRad also lookup
 * Params->RingRadiiPerDet[k][RingNr] instead of the global RefRad.
 * Single-panel path stays the default — current behavior preserved
 * when nDetParams == 0.
 *
 * Plan §2.3 verbatim body. doRefRadFilter gated on nScans==1 (ruling #14).
 * doScanFilter gated on nScans>1. omemargin computed inline per ruling #15
 * using true float eta (no integer table lookup; sub-degree omemargin
 * variation accepted by parity tolerance gate).
 *
 * RingsToReject (ruling #4):
 *   - skipRadialFilter relaxes the FF RefRad/MarginRad check for those rings
 *   - matched spots on rejected rings don't contribute to *nMatchesFracCalc
 *
 * TODO (Phase 8 — soft attribution): when nScans>1 add a soft_attr_fn(dy,tol)
 * weight ∈ [0,1] returned per-match. Output to sidecar
 * IndexBest_weights_all.bin, NOT through the 16-col record (which stays
 * unchanged).
 *
 * TODO (Phase 9 — multi-detector): per-panel gating happens in CalcDiffrSpots;
 * CompareSpots itself is unchanged.
 * --------------------------------------------------------------------------*/
static void CompareSpots(RealType **TheorSpots, int nTspots, RealType RefRad,
                         RealType MarginRad, RealType MarginRadial,
                         const RealType etamargins_[],
                         RealType MarginOme, RealType StepsizeOrient,
                         int nScans_, RealType xThis, RealType yThis,
                         const struct TParams *Params, int *nMatch,
                         RealType **GrainSpots,
                         int *nMatchesFracCalc,
                         const int ringsToReject[], int nRingsToReject) {
  int nMatched = 0;
  int nNonMatched = 0;
  *nMatchesFracCalc = 0;
  int doRefRadFilter = (nScans_ == 1);
  int doScanFilter = (nScans_ > 1);
  RealType scanTol = (Params->ScanPosTol > 0) ? Params->ScanPosTol
                                              : (BeamSize / 2);
  RealType diffOmeInit =
      doRefRadFilter ? 100000.0 : (MarginOme + 1e-5);

  /* Soft beam attribution (Phase 8). Mode 0 (default) keeps the legacy
   * binary scanTol filter and emits weight=1.0 per match — PF parity vs
   * legacy IndexerScanningOMP preserved bit-exact. Modes 1/2 replace the
   * binary window with a wider mode-specific candidate window AND
   * compute a per-match weight from |dy| of the winning spot. */
  int softMode = Params->SoftAttrMode;
  RealType softFwhm = Params->SoftAttrFwhm;
  RealType softFalloff = Params->SoftAttrFalloff;
  RealType softTrunc = Params->SoftAttrTruncate;
  RealType softTopHatInner = (softMode == 1) ? 0.5 * softFwhm : 0.0;
  RealType softTopHatOuter = (softMode == 1) ? softTopHatInner + softFalloff
                                             : 0.0;
  RealType softSigma = (softMode == 2 && softFwhm > 0)
                       ? softFwhm / (2.0 * sqrt(2.0 * log(2.0))) : 0.0;
  RealType softTwoSigmaSq = (softSigma > 0) ? 2.0 * softSigma * softSigma : 0.0;
  RealType softWindow;
  if (softMode == 1) softWindow = softTopHatOuter;
  else if (softMode == 2) softWindow = (softTrunc > 0) ? softTrunc : 1e30;
  else softWindow = scanTol;

  for (int sp = 0; sp < nTspots; sp++) {
    int RingNr = (int)TheorSpots[sp][9];
    int skipRadialFilter = 0;
    for (int i = 0; i < nRingsToReject; i++) {
      if (RingNr == ringsToReject[i]) {
        skipRadialFilter = 1;
        break;
      }
    }
    int iEta = (int)floor((180 + TheorSpots[sp][12]) * Params->InvEtaBinSize);
    int iOme = (int)floor((180 + TheorSpots[sp][6]) * Params->InvOmeBinSize);
    RealType etamargin = etamargins_[RingNr];

    /* omemargin — true float eta (plan ruling #15). The legacy
     * omemargins[181] integer-eta lookup is gone; we keep the same
     * MarginOme + 0.5*StepsizeOrient/sin(|eta|) formula but evaluated at
     * the actual theor eta. Guard against sin(0)→0 with a 1e-9 floor. */
    RealType etaAbs = fabs(TheorSpots[sp][12]);
    RealType sinAbs = fabs(sin(etaAbs * deg2rad));
    if (sinAbs < 1e-9) sinAbs = 1e-9;
    RealType omemargin = MarginOme + 0.5 * StepsizeOrient / sinAbs;

    RealType yRot =
        doScanFilter
            ? (xThis * TheorSpots[sp][14] + yThis * TheorSpots[sp][15])
            : 0.0;

    int MatchFound = 0;
    RealType diffOmeBest = diffOmeInit;
    size_t spotRowBest = 0;
    RealType dyBest = 0.0;  /* |yRot − ypos[scannrobs]| of the winning match;
                             * meaningful only when MatchFound && doScanFilter */

    size_t iRing = (size_t)(RingNr - 1);
    size_t Pos = iRing;
    Pos *= n_eta_bins;
    Pos *= n_ome_bins;
    Pos += (size_t)iEta * n_ome_bins;
    Pos += (size_t)iOme;
    size_t nspotsBin = ndata[Pos * 2];
    size_t DataPos = ndata[Pos * 2 + 1];

    for (size_t iSpot = 0; iSpot < nspotsBin; iSpot++) {
      size_t spotRow = data[(DataPos + iSpot) * 2 + 0];
      size_t scannrobs = data[(DataPos + iSpot) * 2 + 1];

      RealType dy = 0.0;
      if (doScanFilter) {
        RealType ySpot = ypos[scannrobs];
        dy = fabs(yRot - ySpot);
        if (softMode == 0) {
          if (!(dy < scanTol)) continue;
        } else {
          if (!(dy < softWindow)) continue;
        }
      }

      if (doRefRadFilter && !skipRadialFilter) {
        if (!(fabs(RefRad - ObsSpotsLab[spotRow * 10 + 3]) < MarginRad))
          continue;
      }
      if (!(fabs(TheorSpots[sp][13] - ObsSpotsLab[spotRow * 10 + 8]) <
            MarginRadial))
        continue;
      if (!(fabs(TheorSpots[sp][12] - ObsSpotsLab[spotRow * 10 + 6]) <
            etamargin))
        continue;
      RealType diffOme =
          fabs(TheorSpots[sp][6] - ObsSpotsLab[spotRow * 10 + 2]);
      /* FF parity: legacy IndexerOMP does NOT pre-filter by omemargin —
       * the smallest-diffOme tiebreak (diffOmeBest init = 100000) is the
       * only omega gate. PF legacy uses diffOmeBest init = MarginOme + 1e-5,
       * which is the implicit omemargin pre-filter. So the unified omemargin
       * gate only applies in PF mode (where diffOmeInit already encodes it).
       * Skip the redundant pre-filter; rely on diffOmeBest tracking. */
      if (diffOme < diffOmeBest) {
        diffOmeBest = diffOme;
        spotRowBest = spotRow;
        dyBest = dy;
        MatchFound = 1;
      }
    }

    if (MatchFound == 1) {
      /* Soft-attribution weight from |dy| of the winning match. */
      RealType weight = 1.0;
      if (doScanFilter && softMode == 1) {
        if (dyBest <= softTopHatInner) {
          weight = 1.0;
        } else if (dyBest >= softTopHatOuter || softFalloff <= 0) {
          weight = (dyBest < softTopHatOuter) ? 1.0 : 0.0;
        } else {
          weight = (softTopHatOuter - dyBest)
                   / (softTopHatOuter - softTopHatInner);
        }
      } else if (doScanFilter && softMode == 2) {
        if (softTwoSigmaSq > 0) {
          weight = exp(-(dyBest * dyBest) / softTwoSigmaSq);
        }
        if (softTrunc > 0 && dyBest > softTrunc) weight = 0.0;
      }
      GrainSpots[nMatched][0] = nMatched;
      GrainSpots[nMatched][1] = 999.0;
      GrainSpots[nMatched][2] = TheorSpots[sp][10];
      GrainSpots[nMatched][3] = ObsSpotsLab[spotRowBest * 10 + 0];
      GrainSpots[nMatched][4] =
          ObsSpotsLab[spotRowBest * 10 + 0] - TheorSpots[sp][10];
      GrainSpots[nMatched][5] = TheorSpots[sp][11];
      GrainSpots[nMatched][6] = ObsSpotsLab[spotRowBest * 10 + 1];
      GrainSpots[nMatched][7] =
          ObsSpotsLab[spotRowBest * 10 + 1] - TheorSpots[sp][11];
      GrainSpots[nMatched][8] = TheorSpots[sp][6];
      GrainSpots[nMatched][9] = ObsSpotsLab[spotRowBest * 10 + 2];
      GrainSpots[nMatched][10] =
          ObsSpotsLab[spotRowBest * 10 + 2] - TheorSpots[sp][6];
      GrainSpots[nMatched][11] = RefRad;
      GrainSpots[nMatched][12] = ObsSpotsLab[spotRowBest * 10 + 3];
      GrainSpots[nMatched][13] = ObsSpotsLab[spotRowBest * 10 + 3] - RefRad;
      GrainSpots[nMatched][14] = ObsSpotsLab[spotRowBest * 10 + 4];
      GrainSpots[nMatched][15] = weight;  /* Phase 8: soft-attribution weight */
      nMatched++;
      (*nMatchesFracCalc)++;
      for (int i = 0; i < nRingsToReject; i++) {
        if (RingNr == ringsToReject[i]) {
          (*nMatchesFracCalc)--;
          break;
        }
      }
    } else {
      nNonMatched++;
      int idx = nTspots - nNonMatched;
      GrainSpots[idx][0] = -nNonMatched;
      GrainSpots[idx][1] = 999.0;
      GrainSpots[idx][2] = TheorSpots[sp][10];
      GrainSpots[idx][3] = 0;
      GrainSpots[idx][4] = 0;
      GrainSpots[idx][5] = TheorSpots[sp][11];
      GrainSpots[idx][6] = 0;
      GrainSpots[idx][7] = 0;
      GrainSpots[idx][8] = TheorSpots[sp][6];
      GrainSpots[idx][9] = 0;
      GrainSpots[idx][10] = 0;
      GrainSpots[idx][11] = 0;
      GrainSpots[idx][12] = 0;
      GrainSpots[idx][13] = 0;
      GrainSpots[idx][14] = 0;
    }
  }
  *nMatch = nMatched;
}

/* ----------------------------------------------------------------------------
 * FF-only helpers: Friedel-pair generation + position-grid search.
 * All copied verbatim from IndexerOMP.c with the following adjustments only:
 *   • ObsSpotsLab stride changed from 9 → 10 (per plan ruling #1, #10).
 *   • CalcOmega calls go through CalcOmegaFF (uses theta-degrees signature).
 * --------------------------------------------------------------------------*/

/* GetBin — FF (line 115). Used by GenerateIdealSpotsFriedelMixed only.
 * Returns malloc'd spotRows array; caller frees. */
static int GetBin(int ringno, RealType eta, RealType omega, int **spotRows,
                  int *nspotRows) {
  int iRing = ringno - 1;
  int iEta = (int)floor((180 + eta) / EtaBinSize);
  int iOme = (int)floor((180 + omega) / OmeBinSize);
  size_t Pos = (size_t)iRing * n_eta_bins * n_ome_bins +
               (size_t)iEta * n_ome_bins + (size_t)iOme;
  size_t nspots = ndata[Pos * 2];
  size_t DataPos = ndata[Pos * 2 + 1];
  *spotRows = (int *)malloc(nspots * sizeof(int));
  if (*spotRows == NULL) {
    printf("Memory error: could not allocate memory for spotRows matrix.\n");
    return 1;
  }
  /* Unified bin layout: int64 pair (spotRow, scannrobs). FF inputs have
   * scannrobs=0 throughout. We only need spotRow here. */
  for (size_t iSpot = 0; iSpot < nspots; iSpot++) {
    (*spotRows)[iSpot] = (int)data[(DataPos + iSpot) * 2 + 0];
  }
  *nspotRows = (int)nspots;
  return 0;
}

/* FriedelEtaCalculation — FF (line 762) verbatim. */
static void FriedelEtaCalculation(RealType ys, RealType zs, RealType ttheta,
                                  RealType eta, RealType Ring_rad,
                                  RealType Rsample, RealType Hbeam,
                                  RealType *EtaMinFr, RealType *EtaMaxFr) {
  RealType quadr_coeff2 = 0;
  RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0,
                                   y0_min_z0, y0_max = 0, y0_min = 0,
                                   z0_min = 0, z0_max = 0;
  if (eta > 90) eta_Hbeam = 180 - eta;
  else if (eta < -90) eta_Hbeam = 180 - fabs(eta);
  else eta_Hbeam = 90 - fabs(eta);
  Hbeam =
      Hbeam +
      2 * (Rsample * tan(ttheta * deg2rad)) * (sin(eta_Hbeam * deg2rad));
  RealType eta_pole = 1 + rad2deg * acos(1 - (Hbeam / Ring_rad));
  RealType eta_equator = 1 + rad2deg * acos(1 - (Rsample / Ring_rad));
  if ((eta >= eta_pole) && (eta <= (90 - eta_equator))) {
    quadr_coeff = 1; coeff_y0 = -1; coeff_z0 = 1;
  } else if ((eta >= (90 + eta_equator)) && (eta <= (180 - eta_pole))) {
    quadr_coeff = 2; coeff_y0 = -1; coeff_z0 = -1;
  } else if ((eta >= (-90 + eta_equator)) && (eta <= -eta_pole)) {
    quadr_coeff = 2; coeff_y0 = 1; coeff_z0 = 1;
  } else if ((eta >= (-180 + eta_pole)) && (eta <= (-90 - eta_equator))) {
    quadr_coeff = 1; coeff_y0 = 1; coeff_z0 = -1;
  } else
    quadr_coeff = 0;
  RealType y0_max_Rsample = ys + Rsample;
  RealType y0_min_Rsample = ys - Rsample;
  RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
  RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
    y0_min_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
  } else if (quadr_coeff == 2) {
    y0_max_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
    y0_min_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
  }
  if (quadr_coeff > 0) {
    y0_max = minR(y0_max_Rsample, y0_max_z0);
    y0_min = maxR(y0_min_Rsample, y0_min_z0);
  } else {
    if ((eta > -eta_pole) && (eta < eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = 1;
    } else if (eta < (-180 + eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = -1;
    } else if (eta > (180 - eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = -1;
    } else if ((eta > (90 - eta_equator)) && (eta < (90 + eta_equator))) {
      quadr_coeff2 = 1; z0_max = z0_max_Hbeam; z0_min = z0_min_Hbeam;
      coeff_y0 = -1;
    } else if ((eta > (-90 - eta_equator)) && (eta < (-90 + eta_equator))) {
      quadr_coeff2 = 1; z0_max = z0_max_Hbeam; z0_min = z0_min_Hbeam;
      coeff_y0 = 1;
    }
  }
  if (quadr_coeff2 == 0) {
    z0_min = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y0_min * y0_min));
    z0_max = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y0_max * y0_max));
  } else {
    y0_min = coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min * z0_min));
    y0_max = coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max * z0_max));
  }
  RealType dYMin = ys - y0_min;
  RealType dYMax = ys - y0_max;
  RealType dZMin = zs - z0_min;
  RealType dZMax = zs - z0_max;
  RealType YMinFrIdeal = y0_min;
  RealType YMaxFrIdeal = y0_max;
  RealType ZMinFrIdeal = -z0_min;
  RealType ZMaxFrIdeal = -z0_max;
  RealType YMinFr = YMinFrIdeal - dYMin;
  RealType YMaxFr = YMaxFrIdeal - dYMax;
  RealType ZMinFr = ZMinFrIdeal + dZMin;
  RealType ZMaxFr = ZMaxFrIdeal + dZMax;
  RealType Eta1, Eta2;
  CalcEtaAngle((YMinFr + ys), (ZMinFr - zs), &Eta1);
  CalcEtaAngle((YMaxFr + ys), (ZMaxFr - zs), &Eta2);
  *EtaMinFr = minR(Eta1, Eta2);
  *EtaMaxFr = maxR(Eta1, Eta2);
}

/* GenerateIdealSpots — FF (line 867) verbatim. */
static void GenerateIdealSpots(RealType ys, RealType zs, RealType ttheta,
                               RealType eta, RealType Ring_rad,
                               RealType Rsample, RealType Hbeam,
                               RealType step_size, RealType y0_vector[],
                               RealType z0_vector[], int *NoOfSteps) {
  int quadr_coeff2 = 0;
  RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0,
                                   y0_min_z0, y0_max = 0, y0_min = 0,
                                   z0_min = 0, z0_max = 0;
  RealType y01, z01, y02, z02, y_diff, z_diff, length;
  int nsteps;
  if (eta > 90) eta_Hbeam = 180 - eta;
  else if (eta < -90) eta_Hbeam = 180 - fabs(eta);
  else eta_Hbeam = 90 - fabs(eta);
  Hbeam = Hbeam +
          2 * (Rsample * tan(ttheta * deg2rad)) * (sin(eta_Hbeam * deg2rad));
  RealType eta_pole = 1 + rad2deg * acos(1 - (Hbeam / Ring_rad));
  RealType eta_equator = 1 + rad2deg * acos(1 - (Rsample / Ring_rad));
  if ((eta >= eta_pole) && (eta <= (90 - eta_equator))) {
    quadr_coeff = 1; coeff_y0 = -1; coeff_z0 = 1;
  } else if ((eta >= (90 + eta_equator)) && (eta <= (180 - eta_pole))) {
    quadr_coeff = 2; coeff_y0 = -1; coeff_z0 = -1;
  } else if ((eta >= (-90 + eta_equator)) && (eta <= -eta_pole)) {
    quadr_coeff = 2; coeff_y0 = 1; coeff_z0 = 1;
  } else if ((eta >= (-180 + eta_pole)) && (eta <= (-90 - eta_equator))) {
    quadr_coeff = 1; coeff_y0 = 1; coeff_z0 = -1;
  } else
    quadr_coeff = 0;
  RealType y0_max_Rsample = ys + Rsample;
  RealType y0_min_Rsample = ys - Rsample;
  RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
  RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
    y0_min_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
  } else if (quadr_coeff == 2) {
    y0_max_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
    y0_min_z0 = coeff_y0 *
                sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
  }
  if (quadr_coeff > 0) {
    y0_max = minR(y0_max_Rsample, y0_max_z0);
    y0_min = maxR(y0_min_Rsample, y0_min_z0);
  } else {
    if ((eta > -eta_pole) && (eta < eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = 1;
    } else if (eta < (-180 + eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = -1;
    } else if (eta > (180 - eta_pole)) {
      y0_max = y0_max_Rsample; y0_min = y0_min_Rsample; coeff_z0 = -1;
    } else if ((eta > (90 - eta_equator)) && (eta < (90 + eta_equator))) {
      quadr_coeff2 = 1; z0_max = z0_max_Hbeam; z0_min = z0_min_Hbeam;
      coeff_y0 = -1;
    } else if ((eta > (-90 - eta_equator)) && (eta < (-90 + eta_equator))) {
      quadr_coeff2 = 1; z0_max = z0_max_Hbeam; z0_min = z0_min_Hbeam;
      coeff_y0 = 1;
    }
  }
  if (quadr_coeff2 == 0) {
    y01 = y0_min;
    z01 = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y01 * y01));
    y02 = y0_max;
    z02 = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y02 * y02));
    y_diff = y01 - y02; z_diff = z01 - z02;
    length = sqrt(y_diff * y_diff + z_diff * z_diff);
    nsteps = ceil(length / step_size);
  } else {
    z01 = z0_min;
    y01 = coeff_y0 * sqrt((Ring_rad * Ring_rad) - ((z01 * z01)));
    z02 = z0_max;
    y02 = coeff_y0 * sqrt((Ring_rad * Ring_rad) - ((z02 * z02)));
    y_diff = y01 - y02; z_diff = z01 - z02;
    length = sqrt(y_diff * y_diff + z_diff * z_diff);
    nsteps = ceil(length / step_size);
  }
  if ((nsteps % 2) == 0) nsteps = nsteps + 1;
  if (nsteps == 1) {
    if (quadr_coeff2 == 0) {
      y0_vector[0] = (y0_max + y0_min) / 2;
      z0_vector[0] = coeff_z0 * sqrt((Ring_rad * Ring_rad) -
                                     (y0_vector[0] * y0_vector[0]));
    } else {
      z0_vector[0] = (z0_max + z0_min) / 2;
      y0_vector[0] = coeff_y0 * sqrt((Ring_rad * Ring_rad) -
                                     (z0_vector[0] * z0_vector[0]));
    }
  } else {
    int i;
    RealType stepsizeY = (y0_max - y0_min) / (nsteps - 1);
    RealType stepsizeZ = (z0_max - z0_min) / (nsteps - 1);
    if (quadr_coeff2 == 0) {
      for (i = 0; i < nsteps; i++) {
        y0_vector[i] = y0_min + i * stepsizeY;
        z0_vector[i] = coeff_z0 * sqrt((Ring_rad * Ring_rad) -
                                       (y0_vector[i] * y0_vector[i]));
      }
    } else {
      for (i = 0; i < nsteps; i++) {
        z0_vector[i] = z0_min + i * stepsizeZ;
        y0_vector[i] = coeff_y0 * sqrt((Ring_rad * Ring_rad) -
                                       (z0_vector[i] * z0_vector[i]));
      }
    }
  }
  *NoOfSteps = nsteps;
}

/* calc_n_max_min — FF (line 1001) verbatim. */
static void calc_n_max_min(RealType xi, RealType yi, RealType ys, RealType y0,
                           RealType R_sample, int step_size, int *n_max,
                           int *n_min) {
  RealType dy = ys - y0;
  RealType a = xi * xi + yi * yi;
  RealType b = 2 * yi * dy;
  RealType c = dy * dy - R_sample * R_sample;
  RealType D = b * b - 4 * a * c;
  RealType P = sqrt(D);
  RealType lambda_max = (-b + P) / (2 * a) + 20;
  *n_max = (int)((lambda_max * xi) / (step_size));
  *n_min = -*n_max;
}

/* spot_to_unrotated_coordinates — FF (line 1014) verbatim. */
static void spot_to_unrotated_coordinates(RealType xi, RealType yi, RealType zi,
                                          RealType ys, RealType zs, RealType y0,
                                          RealType z0,
                                          RealType step_size_in_x, int n,
                                          RealType omega, RealType *a,
                                          RealType *b, RealType *c) {
  RealType lambda = (step_size_in_x) * (n / xi);
  RealType x1 = lambda * xi;
  RealType y1 = ys - y0 + lambda * yi;
  RealType z1 = zs - z0 + lambda * zi;
  RealType cosOme = cos(omega * deg2rad);
  RealType sinOme = sin(omega * deg2rad);
  *a = (x1 * cosOme) + (y1 * sinOme);
  *b = (y1 * cosOme) - (x1 * sinOme);
  *c = z1;
}

/* GenerateIdealSpotsFriedel — FF (line 1030) verbatim (10-col stride). */
static void GenerateIdealSpotsFriedel(RealType ys, RealType zs, RealType ttheta,
                                      RealType eta, RealType omega, int ringno,
                                      RealType Ring_rad, RealType Rsample,
                                      RealType Hbeam, RealType OmeTol,
                                      RealType RadiusTol,
                                      RealType y0_vector[],
                                      RealType z0_vector[], int *NoOfSteps) {
  RealType EtaF;
  RealType OmeF;
  RealType EtaMinF, EtaMaxF, etaIdealF;
  RealType IdealYPos, IdealZPos;
  *NoOfSteps = 0;
  if (omega < 0) OmeF = omega + 180;
  else OmeF = omega - 180;
  if (eta < 0) EtaF = -180 - eta;
  else EtaF = 180 - eta;
  (void)EtaF; /* mirrors legacy: computed but unused below */
  size_t r;
  int rno_obs;
  RealType ome_obs;
  for (r = 0; r < n_spots; r++) {
    rno_obs = (int)round(ObsSpotsLab[r * 10 + 5]);
    ome_obs = ObsSpotsLab[r * 10 + 2];
    if (rno_obs != ringno) continue;
    if (fabs(ome_obs - OmeF) > OmeTol) continue;
    RealType yf = ObsSpotsLab[r * 10 + 0];
    RealType zf = ObsSpotsLab[r * 10 + 1];
    RealType EtaTransf;
    CalcEtaAngle(yf + ys, zf - zs, &EtaTransf);
    RealType radius = sqrt((yf + ys) * (yf + ys) + (zf - zs) * (zf - zs));
    if (fabs(radius - 2 * Ring_rad) > RadiusTol) continue;
    FriedelEtaCalculation(ys, zs, ttheta, eta, Ring_rad, Rsample, Hbeam,
                          &EtaMinF, &EtaMaxF);
    if ((EtaTransf < EtaMinF) || (EtaTransf > EtaMaxF)) continue;
    RealType ZPositionAccZ = zs - ((zf + zs) / 2);
    RealType YPositionAccY = ys - ((-yf + ys) / 2);
    CalcEtaAngle(YPositionAccY, ZPositionAccZ, &etaIdealF);
    CalcSpotPosition(Ring_rad, etaIdealF, &IdealYPos, &IdealZPos);
    y0_vector[*NoOfSteps] = IdealYPos;
    z0_vector[*NoOfSteps] = IdealZPos;
    (*NoOfSteps)++;
  }
}

/* GenerateIdealSpotsFriedelMixed — FF (line 1107) verbatim (10-col stride
 * + CalcOmegaFF). */
static void GenerateIdealSpotsFriedelMixed(
    RealType ys, RealType zs, RealType Ttheta, RealType Eta, RealType Omega,
    int RingNr, RealType Ring_rad, RealType Lsd, RealType Rsample,
    RealType Hbeam, RealType StepSizePos, RealType OmeTol, RealType RadialTol,
    RealType EtaTol, RealType spots_y[], RealType spots_z[], int *NoOfSteps) {
  const int MinEtaReject = 10;
  RealType omegasFP[4];
  RealType etasFP[4];
  int nsol;
  int nFPCandidates;
  RealType theta = Ttheta / 2;
  RealType SinMinEtaReject = sin(MinEtaReject * deg2rad);
  RealType y0_vector[2000];
  RealType z0_vector[2000];
  RealType G1, G2, G3;
  int SpOnRing, NoOfSpots;
  int FPCandidatesUnique[2000];
  RealType FPCandidates[2000][3];
  RealType xi, yi, zi;
  RealType y0, z0;
  RealType YFP1, ZFP1;
  int nMax, nMin;
  RealType EtaTolDeg;
  EtaTolDeg = rad2deg * atan(EtaTol / Ring_rad);
  *NoOfSteps = 0;
  nFPCandidates = 0;
  if (fabs(sin(Eta * deg2rad)) < SinMinEtaReject) return;
  GenerateIdealSpots(ys, zs, Ttheta, Eta, Ring_rad, Rsample, Hbeam, StepSizePos,
                     y0_vector, z0_vector, &NoOfSpots);
  for (SpOnRing = 0; SpOnRing < NoOfSpots; ++SpOnRing) {
    y0 = y0_vector[SpOnRing];
    z0 = z0_vector[SpOnRing];
    MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi);
    spot_to_gv(Lsd, y0, z0, Omega, &G1, &G2, &G3);
    CalcOmegaFF(-G1, -G2, -G3, theta, omegasFP, etasFP, &nsol);
    if (nsol <= 1) {
      printf("no omega solutions. skipping plane.\n");
      continue;
    }
    RealType OmegaFP, EtaFP, diff0, diff1;
    diff0 = fabs(omegasFP[0] - Omega);
    if (diff0 > 180) diff0 = 360 - diff0;
    diff1 = fabs(omegasFP[1] - Omega);
    if (diff1 > 180) diff1 = 360 - diff1;
    if (diff0 < diff1) {
      OmegaFP = omegasFP[0]; EtaFP = etasFP[0];
    } else {
      OmegaFP = omegasFP[1]; EtaFP = etasFP[1];
    }
    CalcSpotPosition(Ring_rad, EtaFP, &YFP1, &ZFP1);
    calc_n_max_min(xi, yi, ys, y0, Rsample, StepSizePos, &nMax, &nMin);
    RealType a, b, c, YFP, ZFP, RadialPosFP, EtaFPCorr;
    int n;
    for (n = nMin; n <= nMax; ++n) {
      spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0, StepSizePos, n,
                                    Omega, &a, &b, &c);
      if (fabs(c) > Hbeam / 2) continue;
      RealType Dy, Dz;
      /* legacy used FF displacement_spot_needed_COM(omega); we have the
       * omega-degrees variant. */
      displacement_spot_needed_COM_omega(a, b, c, Lsd, YFP1, ZFP1, OmegaFP,
                                          &Dy, &Dz);
      YFP = YFP1 + Dy;
      ZFP = ZFP1 + Dz;
      RadialPosFP = sqrt(YFP * YFP + ZFP * ZFP) - Ring_rad;
      CalcEtaAngle(YFP, ZFP, &EtaFPCorr);
      int *spotRows;
      int nspotRows, iSpot, spotRow;
      GetBin(RingNr, EtaFPCorr, OmegaFP, &spotRows, &nspotRows);
      RealType diffPos2, dy, dz;
      for (iSpot = 0; iSpot < nspotRows; iSpot++) {
        spotRow = spotRows[iSpot];
        if ((fabs(RadialPosFP - ObsSpotsLab[spotRow * 10 + 8]) < RadialTol) &&
            (fabs(OmegaFP - ObsSpotsLab[spotRow * 10 + 2]) < OmeTol) &&
            (fabs(EtaFPCorr - ObsSpotsLab[spotRow * 10 + 6]) < EtaTolDeg)) {
          dy = (YFP - ObsSpotsLab[spotRow * 10 + 0]);
          dz = (ZFP - ObsSpotsLab[spotRow * 10 + 1]);
          diffPos2 = dy * dy + dz * dz;
          int i;
          int idx = nFPCandidates;
          for (i = 0; i < nFPCandidates; ++i) {
            if (FPCandidates[i][0] == ObsSpotsLab[spotRow * 10 + 4]) {
              if (diffPos2 < FPCandidates[i][2]) {
                idx = i;
              } else {
                idx = -1;
              }
              break;
            }
          }
          if (idx >= 0) {
            FPCandidates[idx][0] = ObsSpotsLab[spotRow * 10 + 4];
            FPCandidates[idx][1] = SpOnRing;
            FPCandidates[idx][2] = diffPos2;
            if (idx == nFPCandidates) nFPCandidates++;
          }
        }
      }
      free(spotRows);
    }
  }
  int i;
  int nFPCandidatesUniq = 0;
  for (i = 0; i < nFPCandidates; ++i) {
    AddUnique(FPCandidatesUnique, &nFPCandidatesUniq,
              (int)FPCandidates[i][1]);
  }
  int iFP;
  for (iFP = 0; iFP < nFPCandidatesUniq; ++iFP) {
    spots_y[iFP] = y0_vector[FPCandidatesUnique[iFP]];
    spots_z[iFP] = z0_vector[FPCandidatesUnique[iFP]];
  }
  *NoOfSteps = nFPCandidatesUniq;
}

/* ----------------------------------------------------------------------------
 * DoIndexing_PF — voxel-mode kernel. Body adapted from PF DoIndexing
 * (IndexerScanningOMP.c lines 1152-1332).
 * --------------------------------------------------------------------------*/
static int DoIndexing_PF(int SpotID, int voxNr, double xThis, double yThis,
                         double zThis, struct TParams Params, int SpotRowNo,
                         VoxelAccumulator *acc,
                         struct IndexerScratch *scratch) {
  (void)voxNr;
  int i;
  RealType ga = xThis, gb = yThis, gc = zThis;
  RealType y0 = ObsSpotsLab[SpotRowNo * 10 + 0];
  RealType z0 = ObsSpotsLab[SpotRowNo * 10 + 1];
  RealType omega = ObsSpotsLab[SpotRowNo * 10 + 2];
  int ringnr = (int)ObsSpotsLab[SpotRowNo * 10 + 5];
  RealType xi, yi, zi, g1, g2, g3, hklnormal[3], hkl[3];
  MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi);
  spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
  hklnormal[0] = g1;
  hklnormal[1] = g2;
  hklnormal[2] = g3;
  int nOrient, or_ = 0, orDelta = 1;
  RealType RefRad = ObsSpotsLab[SpotRowNo * 10 + 3];
  hkl[0] = RingHKL[ringnr][0];
  hkl[1] = RingHKL[ringnr][1];
  hkl[2] = RingHKL[ringnr][2];
  GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient,
                                 scratch->OrMat, &nOrient, ringnr);
  RealType **TheorSpots = scratch->TheorSpots;
  RealType **GrainSpots = scratch->GrainSpots;
  RealType **GrainMatches = scratch->GrainMatches;
  RealType **AllGrainSpots = scratch->AllGrainSpots;
  RealType **GrainMatchesT = scratch->GrainMatchesT;
  RealType **AllGrainSpotsT = scratch->AllGrainSpotsT;
  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  int nTspots, nTspotsFracCalc, nMatchesFracCalc;
  int bestnMatchesIsp = -1, bestnTspotsIsp = -1;
  int nMatches, bestMatchFound = 0;
  int r;
  int rownr = 0;
  int sp;
  RealType MinInternalAngle = 1000;
  RealType Displ_y, Displ_z;
  RealType fracMatches = 0;
  RealType bestConfidence = 0;
  RealType FracThis;

  while (or_ < nOrient) {
    CalcDiffrSpots(scratch->OrMat[or_], Params.LatticeConstant,
                   Params.Wavelength, Params.Distance, Params.RingRadii,
                   Params.OmegaRanges, Params.BoxSizes, Params.NoOfOmegaRanges,
                   Params.ExcludePoleAngle, TheorSpots, &nTspots,
                   Params.RingsToReject, Params.nRingsToRejectCalc,
                   &nTspotsFracCalc);
    /* Use raw nTspots for accept threshold (matches legacy PF line 1213). */
    for (sp = 0; sp < nTspots; sp++) {
      displacement_spot_needed_COM(ga, gb, gc, TheorSpots[sp][3],
                                   TheorSpots[sp][4], TheorSpots[sp][5],
                                   TheorSpots[sp][14], TheorSpots[sp][15],
                                   &Displ_y, &Displ_z);
      TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
      TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
      CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12]);
      TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                                TheorSpots[sp][11] * TheorSpots[sp][11]) -
                           Params.RingRadii[(int)TheorSpots[sp][9]];
    }
    CompareSpots(TheorSpots, nTspots, RefRad, Params.MarginRad,
                 Params.MarginRadial, etamargins, Params.MarginOme,
                 Params.StepsizeOrient, numScans, xThis, yThis, &Params,
                 &nMatches, GrainSpots, &nMatchesFracCalc,
                 Params.RingsToReject, Params.nRingsToRejectCalc);
    /* Use FracCalc denominators when RingsToReject is active (ruling #4 +
     * N5 resolution). When nRingsToReject==0, FracCalc denominators equal
     * raw counts → PF bit-identity vs legacy IndexerScanningOMP preserved. */
    int nMatchesAccept = (Params.nRingsToRejectCalc > 0) ? nMatchesFracCalc : nMatches;
    int nTspotsAccept  = (Params.nRingsToRejectCalc > 0) ? nTspotsFracCalc  : nTspots;
    FracThis = (nTspotsAccept > 0)
               ? (double)nMatchesAccept / (double)nTspotsAccept : 0.0;
    if (FracThis > Params.MinMatchesToAcceptFrac) {
      if (FracThis >= bestConfidence) {
        RealType prevBestConfidence = bestConfidence;
        bestConfidence = FracThis;
        bestMatchFound = 1;
        for (i = 0; i < 9; i++)
          GrainMatchesT[0][i] = scratch->OrMat[or_][i / 3][i % 3];
        GrainMatchesT[0][9] = ga;
        GrainMatchesT[0][10] = gb;
        GrainMatchesT[0][11] = gc;
        GrainMatchesT[0][12] = nTspots;
        GrainMatchesT[0][13] = nMatches;
        GrainMatchesT[0][14] = 1;
        for (r = 0; r < nTspots; r++) {
          /* Phase 8: copy 16 cols so the per-match weight written by
           * CompareSpots at col 15 carries through. AllGrainSpots[*][16]
           * is the IA scratch slot, written by CalcIA below. */
          memcpy(AllGrainSpotsT[r], GrainSpots[r], 16 * sizeof(RealType));
        }
        CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance, scratch);
        if (FracThis == prevBestConfidence &&
            GrainMatchesT[0][15] > MinInternalAngle) {
          /* same conf, worse IA → revert */
          bestConfidence = prevBestConfidence;
        } else {
          MinInternalAngle = GrainMatchesT[0][15];
          /* Store with the same denominator policy used for FracThis above
           * (raw when no rejection, FracCalc otherwise). The final accept
           * gate below recomputes fracMatches from these. */
          bestnMatchesIsp = nMatchesAccept;
          bestnTspotsIsp = nTspotsAccept;
          rownr = nTspots;
          memcpy(GrainMatches[0], GrainMatchesT[0],
                 N_COL_GRAINMATCHES * sizeof(RealType));
          for (r = 0; r < nTspots; r++)
            memcpy(AllGrainSpots[r], AllGrainSpotsT[r],
                   N_COL_GRAINSPOTS * sizeof(RealType));
          for (r = nTspots; r < nRowsOutput; r++)
            memset(AllGrainSpots[r], 0, N_COL_GRAINSPOTS * sizeof(RealType));
        }
      }
    }
    or_ += orDelta;
  }

  if (bestnMatchesIsp < 0) return 0;
  fracMatches = (RealType)bestnMatchesIsp / (RealType)bestnTspotsIsp;
  if ((fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 ||
       (int)bestnMatchesIsp == -1 || bestMatchFound == 0) ||
      fracMatches < Params.MinMatchesToAcceptFrac) {
    return 0;
  }
  double outArr[16] = {
      (double)SpotID,      GrainMatches[0][15], GrainMatches[0][0],
      GrainMatches[0][1],  GrainMatches[0][2],  GrainMatches[0][3],
      GrainMatches[0][4],  GrainMatches[0][5],  GrainMatches[0][6],
      GrainMatches[0][7],  GrainMatches[0][8],  GrainMatches[0][9],
      GrainMatches[0][10], GrainMatches[0][11], GrainMatches[0][12],
      GrainMatches[0][13]};
  int matchedNrSpots = (int)GrainMatches[0][13];
  int *outArr2 = (int *)malloc(matchedNrSpots * sizeof(int));
  double *outWeights = (double *)malloc(matchedNrSpots * sizeof(double));
  for (i = 0; i < matchedNrSpots; i++) {
    outArr2[i] = (int)AllGrainSpots[i][14];
    outWeights[i] = AllGrainSpots[i][15];  /* Phase 8: soft-attr weight */
  }
  size_t keyArr[4] = {(size_t)SpotID, (size_t)matchedNrSpots, 0, 0};
  VoxelAccum_addSolution(acc, outArr, keyArr, outArr2, outWeights, matchedNrSpots);
  free(outArr2);
  free(outWeights);
  (void)rownr;
  return 0;
}

/* ----------------------------------------------------------------------------
 * DoIndexing_Seeded — mic/grains-seeded path. Adapted from PF
 * DoIndexingSingle (lines 1074-1150). Used for both `MicFile` and PF-style
 * `GrainsFile`. Always writes through the consolidated accumulator path.
 * --------------------------------------------------------------------------*/
static int DoIndexing_Seeded(int voxNr, int grainIdx, double OM[3][3],
                             double xThis, double yThis,
                             struct TParams Params, VoxelAccumulator *acc,
                             struct IndexerScratch *scratch) {
  (void)voxNr; (void)grainIdx;
  RealType ga = xThis, gb = yThis, gc = 0;
  RealType **TheorSpots = scratch->TheorSpots;
  RealType **GrainSpots = scratch->GrainSpots;
  RealType **GrainMatches = scratch->GrainMatches;
  RealType **AllGrainSpots = scratch->AllGrainSpots;

  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  int nTspots, nTspotsFracCalc, nMatchesFracCalc;
  int nMatches;
  int r, i, SpotID;
  int sp;
  RealType Displ_y, Displ_z;
  RealType FracThis;
  RealType RefRad = -1;  /* legacy PF DoIndexingSingle convention */
  CalcDiffrSpots(OM, Params.LatticeConstant, Params.Wavelength,
                 Params.Distance, Params.RingRadii, Params.OmegaRanges,
                 Params.BoxSizes, Params.NoOfOmegaRanges,
                 Params.ExcludePoleAngle, TheorSpots, &nTspots,
                 Params.RingsToReject, Params.nRingsToRejectCalc,
                 &nTspotsFracCalc);
  for (sp = 0; sp < nTspots; sp++) {
    displacement_spot_needed_COM(ga, gb, gc, TheorSpots[sp][3],
                                 TheorSpots[sp][4], TheorSpots[sp][5],
                                 TheorSpots[sp][14], TheorSpots[sp][15],
                                 &Displ_y, &Displ_z);
    TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
    TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
    CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12]);
    TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                              TheorSpots[sp][11] * TheorSpots[sp][11]) -
                         Params.RingRadii[(int)TheorSpots[sp][9]];
  }
  CompareSpots(TheorSpots, nTspots, RefRad, Params.MarginRad,
               Params.MarginRadial, etamargins, Params.MarginOme,
               Params.StepsizeOrient, numScans, xThis, yThis, &Params,
               &nMatches, GrainSpots, &nMatchesFracCalc,
               Params.RingsToReject, Params.nRingsToRejectCalc);
  /* N5: use FracCalc denominators when RingsToReject is active. */
  int nMatchesAccept = (Params.nRingsToRejectCalc > 0) ? nMatchesFracCalc : nMatches;
  int nTspotsAccept  = (Params.nRingsToRejectCalc > 0) ? nTspotsFracCalc  : nTspots;
  FracThis = (nTspotsAccept > 0)
             ? (double)nMatchesAccept / (double)nTspotsAccept : 0.0;
  if (FracThis <= Params.MinMatchesToAcceptFrac) return 0;
  for (i = 0; i < 9; i++)
    GrainMatches[0][i] = OM[i / 3][i % 3];
  GrainMatches[0][9] = ga;
  GrainMatches[0][10] = gb;
  GrainMatches[0][11] = gc;
  GrainMatches[0][12] = nTspots;
  GrainMatches[0][13] = nMatches;
  GrainMatches[0][14] = 1;
  for (r = 0; r < nTspots; r++) {
    /* Phase 8: 16-col copy preserves the per-match weight from CompareSpots. */
    memcpy(AllGrainSpots[r], GrainSpots[r], 16 * sizeof(RealType));
  }
  for (r = nTspots; r < nRowsOutput; r++)
    memset(AllGrainSpots[r], 0, N_COL_GRAINSPOTS * sizeof(RealType));
  CalcIA(GrainMatches, 1, AllGrainSpots, Params.Distance, scratch);
  SpotID = (int)AllGrainSpots[0][14];
  double outArr[16] = {
      (double)SpotID,      GrainMatches[0][15], GrainMatches[0][0],
      GrainMatches[0][1],  GrainMatches[0][2],  GrainMatches[0][3],
      GrainMatches[0][4],  GrainMatches[0][5],  GrainMatches[0][6],
      GrainMatches[0][7],  GrainMatches[0][8],  GrainMatches[0][9],
      GrainMatches[0][10], GrainMatches[0][11], GrainMatches[0][12],
      GrainMatches[0][13]};
  int *outArr2 = (int *)malloc(nMatches * sizeof(int));
  double *outWeights = (double *)malloc(nMatches * sizeof(double));
  for (i = 0; i < nMatches; i++) {
    outArr2[i] = (int)AllGrainSpots[i][14];
    outWeights[i] = AllGrainSpots[i][15];  /* Phase 8 */
  }
  size_t keyArr[4] = {(size_t)SpotID, (size_t)nMatches, 0, 0};
  VoxelAccum_addSolution(acc, outArr, keyArr, outArr2, outWeights, nMatches);
  free(outArr2);
  free(outWeights);
  return 0;
}

/* ----------------------------------------------------------------------------
 * DoIndexing_FF — FF spot-mode kernel. Wraps the shared per-orientation body
 * in the 3D (isp, or, n) search Friedel-pair generation + adaptive position
 * stride from FF DoIndexing (IndexerOMP.c lines 1721-1987).
 *
 * Writes 0 or 1 solution per "voxel" (which represents one spot-to-index)
 * through the consolidated accumulator path (plan ruling #8).
 * --------------------------------------------------------------------------*/
static int DoIndexing_FF(int SpotID, int SpotRowNo, struct TParams Params,
                         VoxelAccumulator *acc,
                         struct IndexerScratch *scratch) {
  RealType HalfBeam = Params.Hbeam / 2;
  RealType MinMatchesToAccept;
  RealType ga, gb, gc;
  int nTspots, nTspotsFracCalc, nMatchesFracCalc;
  int bestnMatchesIsp, bestnMatchesRot, bestnMatchesPos;
  int bestnTspotsIsp, bestnTspotsRot, bestnTspotsPos;
  int nOrient;
  RealType hklnormal[3];
  RealType Displ_y;
  RealType Displ_z;
  int or_;
  int sp;
  int nMatches;
  int r, c, i, j;
  RealType y0_vector[MAX_N_STEPS];
  RealType z0_vector[MAX_N_STEPS];
  int nPlaneNormals;
  double hkl[3];
  RealType g1, g2, g3;
  int isp;
  RealType xi, yi, zi;
  int n_max, n_min, n;
  RealType y0, z0;
  int orDelta, ispDelta, nDelta;
  RealType fracMatches;
  int usingFriedelPair;

  RealType **AllGrainSpots = scratch->AllGrainSpots;
  RealType **AllGrainSpotsT = scratch->AllGrainSpotsT;
  RealType **GrainMatchesT = scratch->GrainMatchesT;
  RealType **GrainMatches = scratch->GrainMatches;
  RealType **GrainSpots = scratch->GrainSpots;
  RealType **TheorSpots = scratch->TheorSpots;
  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;

  RealType ys = ObsSpotsLab[SpotRowNo * 10 + 0];
  RealType zs = ObsSpotsLab[SpotRowNo * 10 + 1];
  RealType omega = ObsSpotsLab[SpotRowNo * 10 + 2];
  RealType RefRad = ObsSpotsLab[SpotRowNo * 10 + 3];
  RealType eta = ObsSpotsLab[SpotRowNo * 10 + 6];
  int ringnr = (int)ObsSpotsLab[SpotRowNo * 10 + 5];
  hkl[0] = RingHKL[ringnr][0];
  hkl[1] = RingHKL[ringnr][1];
  hkl[2] = RingHKL[ringnr][2];

  nPlaneNormals = 0;
  usingFriedelPair = 0;
  if (Params.UseFriedelPairs == 1) {
    usingFriedelPair = 1;
    GenerateIdealSpotsFriedel(ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
                              Params.RingRadii[ringnr], Params.Rsample,
                              Params.Hbeam, Params.MarginOme,
                              Params.MarginRadial, y0_vector, z0_vector,
                              &nPlaneNormals);
    if (nPlaneNormals == 0) {
      GenerateIdealSpotsFriedelMixed(
          ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
          Params.RingRadii[ringnr], Params.Distance, Params.Rsample,
          Params.Hbeam, Params.StepsizePos, Params.MarginOme,
          Params.MarginRadial, Params.MarginEta, y0_vector, z0_vector,
          &nPlaneNormals);
    }
  }
  if (nPlaneNormals == 0) {
    if (usingFriedelPair == 1) {
      /* legacy FF returned 1 and dropped the seed entirely. */
      return 1;
    }
    usingFriedelPair = 0;
    GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                       Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam,
                       Params.StepsizePos, y0_vector, z0_vector,
                       &nPlaneNormals);
  }
  bestnMatchesIsp = -1;
  bestnTspotsIsp = 0;
  isp = 0;
  RealType bestFracTillNow = -1;
  int bestMatchFound = 0;
  int rownr = 0;
  RealType MinInternalAngle = 1000;
  while (isp < nPlaneNormals) {
    y0 = y0_vector[isp];
    z0 = z0_vector[isp];
    MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi);
    spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
    hklnormal[0] = g1;
    hklnormal[1] = g2;
    hklnormal[2] = g3;
    GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient,
                                   scratch->OrMat, &nOrient, ringnr);
    bestnMatchesRot = -1;
    bestnTspotsRot = 0;
    or_ = 0;
    orDelta = 1;
    while (or_ < nOrient) {
      CalcDiffrSpots(scratch->OrMat[or_], Params.LatticeConstant,
                     Params.Wavelength, Params.Distance, Params.RingRadii,
                     Params.OmegaRanges, Params.BoxSizes,
                     Params.NoOfOmegaRanges, Params.ExcludePoleAngle,
                     TheorSpots, &nTspots, Params.RingsToReject,
                     Params.nRingsToRejectCalc, &nTspotsFracCalc);
      MinMatchesToAccept = nTspotsFracCalc * Params.MinMatchesToAcceptFrac;
      bestnMatchesPos = -1;
      bestnTspotsPos = 0;
      calc_n_max_min(xi, yi, ys, y0, Params.Rsample, Params.StepsizePos, &n_max,
                     &n_min);
      n = n_min;
      while (n <= n_max) {
        spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0,
                                      Params.StepsizePos, n, omega, &ga, &gb,
                                      &gc);
        if (fabs(gc) > HalfBeam) {
          n++;
          continue;
        }
        for (sp = 0; sp < nTspots; sp++) {
          /* FF parity: recompute sin/cos from omega via the _omega wrapper,
           * matching IndexerOMP exactly. Cached cols 14/15 (PF style) would
           * differ from sin(omega*deg2rad) by ~1 ULP because of the
           * acos(cosome)→omega→sin(omega) round-trip, which CalcIA amplifies
           * to ~1e-7 in avgIA. */
          displacement_spot_needed_COM_omega(
              ga, gb, gc, TheorSpots[sp][3], TheorSpots[sp][4],
              TheorSpots[sp][5], TheorSpots[sp][6], &Displ_y, &Displ_z);
          TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
          TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
          CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11],
                       &TheorSpots[sp][12]);
          TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                                    TheorSpots[sp][11] * TheorSpots[sp][11]) -
                               Params.RingRadii[(int)TheorSpots[sp][9]];
        }
        CompareSpots(TheorSpots, nTspots, RefRad, Params.MarginRad,
                     Params.MarginRadial, etamargins, Params.MarginOme,
                     Params.StepsizeOrient, /*nScans=*/1, 0.0, 0.0, &Params,
                     &nMatches, GrainSpots, &nMatchesFracCalc,
                     Params.RingsToReject, Params.nRingsToRejectCalc);
        if (nMatchesFracCalc > bestnMatchesPos) {
          bestnMatchesPos = nMatchesFracCalc;
          bestnTspotsPos = nTspotsFracCalc;
        }
        double fracMatchesThis;
        if (nTspotsFracCalc > 0)
          fracMatchesThis = (RealType)nMatchesFracCalc / (RealType)nTspotsFracCalc;
        else
          fracMatchesThis = 0.0;
        if (nMatchesFracCalc >= MinMatchesToAccept &&
            fracMatchesThis >= bestFracTillNow) {
          bestMatchFound = 1;
          for (i = 0; i < 9; i++)
            GrainMatchesT[0][i] = scratch->OrMat[or_][i / 3][i % 3];
          GrainMatchesT[0][9] = ga;
          GrainMatchesT[0][10] = gb;
          GrainMatchesT[0][11] = gc;
          GrainMatchesT[0][12] = (double)nTspots;
          GrainMatchesT[0][13] = (double)nMatches;
          GrainMatchesT[0][14] = 1;
          for (r = 0; r < nTspots; r++) {
            /* Phase 8: copy 16 cols so the per-match weight propagates. */
            for (c = 0; c < 16; c++) AllGrainSpotsT[r][c] = GrainSpots[r][c];
          }
          CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance, scratch);
          if (fracMatchesThis > bestFracTillNow ||
              (fracMatchesThis == bestFracTillNow &&
               GrainMatchesT[0][15] < MinInternalAngle)) {
            bestFracTillNow = fracMatchesThis;
            MinInternalAngle = GrainMatchesT[0][15];
            rownr = nTspots;
            for (i = 0; i < 16; i++) GrainMatches[0][i] = GrainMatchesT[0][i];
            for (r = 0; r < nTspots; r++)
              for (c = 0; c < 17; c++)
                AllGrainSpots[r][c] = AllGrainSpotsT[r][c];
            for (r = nTspots; r < nRowsOutput; r++)
              for (c = 0; c < 17; c++) AllGrainSpots[r][c] = 0;
          }
        }
        /* Adaptive stride (legacy FF lines 1940-1947). FP-sensitive — copy
         * verbatim. */
        nDelta = 1;
        if (nTspotsFracCalc != 0) {
          fracMatches = (RealType)nMatchesFracCalc / nTspotsFracCalc;
          if (fracMatches < 0.5) {
            nDelta = 5 - (int)round(fracMatches * (5 - 1) / 0.5);
          }
        }
        n = n + nDelta;
      }
      if (bestnMatchesPos > bestnMatchesRot) {
        bestnMatchesRot = bestnMatchesPos;
        bestnTspotsRot = bestnTspotsPos;
      }
      or_ = or_ + orDelta;
    }
    if (bestnMatchesRot > bestnMatchesIsp) {
      bestnMatchesIsp = bestnMatchesRot;
      bestnTspotsIsp = bestnTspotsRot;
    }
    ispDelta = 1;
    /* (Adaptive isp stride disabled in legacy FF too — see comment in
     *  IndexerOMP.c lines 1959-1965.) */
    isp = isp + ispDelta;
  }
  fracMatches = bestFracTillNow;
  if (fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 ||
      (int)bestnMatchesIsp == -1 || bestMatchFound == 0) {
    return 1;
  }
  /* Write 0-or-1 solution per spot through the consolidated accumulator. The
   * 16-double payload mirrors PF DoIndexing (lines 1317-1323):
   *   outArr = [SpotID, avgIA, OM[0..8], ga, gb, gc, nTspots, nMatches]
   */
  double outArr[16] = {
      (double)SpotID,      GrainMatches[0][15], GrainMatches[0][0],
      GrainMatches[0][1],  GrainMatches[0][2],  GrainMatches[0][3],
      GrainMatches[0][4],  GrainMatches[0][5],  GrainMatches[0][6],
      GrainMatches[0][7],  GrainMatches[0][8],  GrainMatches[0][9],
      GrainMatches[0][10], GrainMatches[0][11], GrainMatches[0][12],
      GrainMatches[0][13]};
  int matchedNrSpots = (int)GrainMatches[0][13];
  int *outArr2 = (int *)malloc(matchedNrSpots * sizeof(int));
  double *outWeights = (double *)malloc(matchedNrSpots * sizeof(double));
  /* Matched spot IDs are stored in AllGrainSpots[i][14] for the matched rows
   * [0, nMatches); soft-attribution weights at AllGrainSpots[i][15]. */
  for (i = 0; i < matchedNrSpots; i++) {
    outArr2[i] = (int)AllGrainSpots[i][14];
    outWeights[i] = AllGrainSpots[i][15];  /* Phase 8 */
  }
  size_t keyArr[4] = {(size_t)SpotID, (size_t)matchedNrSpots, 0, 0};
  VoxelAccum_addSolution(acc, outArr, keyArr, outArr2, outWeights, matchedNrSpots);
  free(outArr2);
  free(outWeights);
  (void)j;
  return 0;
}

/* ----------------------------------------------------------------------------
 * ReadParams — unified, all aliases (plan ruling #13).
 *
 * BigDet entirely dropped (plan ruling #5).
 * --------------------------------------------------------------------------*/
static int ReadParams(char FileName[], struct TParams *Params) {
#define MAX_LINE_LENGTH 4096
  FILE *fp;
  char line[MAX_LINE_LENGTH];
  char dummy[MAX_LINE_LENGTH];
  char *str;
  int NrOfBoxSizes = 0;
  int cmpres;
  int NoRingNumbers = 0;
  Params->NrOfRings = 0;
  Params->NoOfOmegaRanges = 0;
  Params->isGrainsInput = 0;
  Params->nRingsToRejectCalc = 0;
  Params->ScanPosTol = 0.0;
  Params->RingToIndex = 0;
  Params->UseFriedelPairs = 0;
  Params->SoftAttrMode = 0;        /* legacy hard window by default */
  Params->SoftAttrFwhm = 0.0;
  Params->SoftAttrFalloff = 0.0;
  Params->SoftAttrTruncate = 0.0;
  /* Phase 9: multi-detector scaffolding init. */
  Params->nDetParams = 0;
  Params->hasRingRadiiPerDet = 0;
  Params->hasEtaCoverage = 0;
  for (int k = 0; k < MAX_N_DETS; k++) {
    Params->DetIDs[k] = -1;
    Params->nEtaArcs[k] = 0;
    for (int r = 0; r < MAX_N_RINGS; r++) {
      Params->RingRadiiPerDet[k][r] = 0.0;
    }
  }
  Params->MarginEta = 0;
  Params->MarginRad = 0;
  sprintf(Params->MicFN, "0");
  sprintf(Params->GrainsFN, "0");
  Params->GrainsFileName[0] = '\0';
  Params->OutputFolder[0] = '\0';
  Params->SpotsFileName[0] = '\0';
  Params->IDsFileName[0] = '\0';

  fp = fopen(FileName, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s.\n", FileName);
    return 1;
  }
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
    str = "RingNumbers ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      if (NoRingNumbers >= MAX_N_RINGS) {
        fprintf(stderr, "Error: more than %d RingNumbers entries.\n",
                MAX_N_RINGS);
        fclose(fp);
        return 1;
      }
      sscanf(line, "%s %d", dummy, &(Params->RingNumbers[NoRingNumbers]));
      NoRingNumbers++;
      continue;
    }
    str = "RingsToExcludeFraction ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy,
             &(Params->RingsToReject[Params->nRingsToRejectCalc]));
      Params->nRingsToRejectCalc++;
      continue;
    }
    str = "RingToIndex ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->RingToIndex));
      continue;
    }
    str = "px ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &pixelsize);
      continue;
    }
    str = "BeamSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &BeamSize);
      /* Match legacy PF: small relaxation. */
      BeamSize += 0.1;
      continue;
    }
    str = "SpaceGroup ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->SpaceGroupNum));
      SGNum = Params->SpaceGroupNum;
      continue;
    }
    str = "LatticeParameter ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->LatticeConstant));
      sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &ABCABG[0], &ABCABG[1],
             &ABCABG[2], &ABCABG[3], &ABCABG[4], &ABCABG[5]);
      continue;
    }
    str = "LatticeConstant ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->LatticeConstant));
      sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &ABCABG[0], &ABCABG[1],
             &ABCABG[2], &ABCABG[3], &ABCABG[4], &ABCABG[5]);
      continue;
    }
    str = "Wavelength ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Wavelength));
      continue;
    }
    str = "Distance ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Distance));
      continue;
    }
    str = "Lsd ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Distance));
      continue;
    }
    str = "Rsample ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Rsample));
      continue;
    }
    str = "Hbeam ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Hbeam));
      continue;
    }
    str = "StepsizePos ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->StepsizePos));
      continue;
    }
    str = "StepsizeOrient ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->StepsizeOrient));
      continue;
    }
    str = "StepSizeOrient ";   /* CamelCase alias */
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->StepsizeOrient));
      continue;
    }
    str = "MarginOme ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginOme));
      continue;
    }
    str = "MarginRadius ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginRad));
      continue;
    }
    str = "MarginRad ";        /* alias of MarginRadius */
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginRad));
      continue;
    }
    str = "MarginRadial ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginRadial));
      continue;
    }
    str = "EtaBinSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->EtaBinSize));
      continue;
    }
    str = "OmeBinSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->OmeBinSize));
      continue;
    }
    str = "MinMatchesToAcceptFrac ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MinMatchesToAcceptFrac));
      continue;
    }
    str = "Completeness ";     /* alias */
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MinMatchesToAcceptFrac));
      continue;
    }
    str = "ExcludePoleAngle ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->ExcludePoleAngle));
      continue;
    }
    str = "MinEta ";           /* alias */
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->ExcludePoleAngle));
      continue;
    }
    str = "RingRadii ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy,
             &(Params->RingRadiiUser[Params->NrOfRings]));
      Params->NrOfRings = Params->NrOfRings + 1;
      continue;
    }
    str = "OmegaRange ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf %lf", dummy,
             &(Params->OmegaRanges[Params->NoOfOmegaRanges][0]),
             &(Params->OmegaRanges[Params->NoOfOmegaRanges][1]));
      (Params->NoOfOmegaRanges)++;
      continue;
    }
    str = "BoxSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf %lf %lf %lf", dummy,
             &(Params->BoxSizes[NrOfBoxSizes][0]),
             &(Params->BoxSizes[NrOfBoxSizes][1]),
             &(Params->BoxSizes[NrOfBoxSizes][2]),
             &(Params->BoxSizes[NrOfBoxSizes][3]));
      NrOfBoxSizes++;
      continue;
    }
    str = "SpotsFileName ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->SpotsFileName);
      continue;
    }
    str = "GrainsFile ";
    /* FF semantics: setting GrainsFile triggers seeded-from-grains path
     * (isGrainsInput=1). PF also accepts via Params.GrainsFN — keep both
     * populated. */
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      Params->isGrainsInput = 1;
      sscanf(line, "%s %s", dummy, Params->GrainsFileName);
      sscanf(line, "%s %s", dummy, Params->GrainsFN);
      continue;
    }
    str = "IDsFileName ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->IDsFileName);
      continue;
    }
    str = "MicFile ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->MicFN);
      continue;
    }
    str = "MarginEta ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginEta));
      continue;
    }
    str = "UseFriedelPairs ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->UseFriedelPairs));
      continue;
    }
    str = "ScanPosTol ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->ScanPosTol));
      continue;
    }
    str = "SoftAttrMode ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      char modeStr[64] = {0};
      sscanf(line, "%s %s", dummy, modeStr);
      if (strncmp(modeStr, "none", 4) == 0 ||
          strncmp(modeStr, "hard", 4) == 0 ||
          strncmp(modeStr, "0", 1) == 0) {
        Params->SoftAttrMode = 0;
      } else if (strncmp(modeStr, "top_hat", 7) == 0 ||
                 strncmp(modeStr, "tophat", 6) == 0 ||
                 strncmp(modeStr, "1", 1) == 0) {
        Params->SoftAttrMode = 1;
      } else if (strncmp(modeStr, "gaussian", 8) == 0 ||
                 strncmp(modeStr, "2", 1) == 0) {
        Params->SoftAttrMode = 2;
      } else {
        fprintf(stderr,
                "Warning: unknown SoftAttrMode %s; defaulting to none.\n",
                modeStr);
        Params->SoftAttrMode = 0;
      }
      continue;
    }
    str = "SoftAttrFwhm ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->SoftAttrFwhm));
      continue;
    }
    str = "SoftAttrFalloff ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->SoftAttrFalloff));
      continue;
    }
    str = "SoftAttrTruncate ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->SoftAttrTruncate));
      continue;
    }
    str = "OutputFolder ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->OutputFolder);
      continue;
    }
    /* ---- Phase 9: multi-detector params ---- */
    str = "DetParams ";
    if (strncmp(line, str, strlen(str)) == 0) {
      int det_id = -1;
      double lsd = 0, y_bc = 0, z_bc = 0, tx = 0, ty = 0, tz = 0;
      double pdist[11] = {0};
      int got = sscanf(
          line,
          "%s %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
          dummy, &det_id, &lsd, &y_bc, &z_bc, &tx, &ty, &tz,
          &pdist[0], &pdist[1], &pdist[2], &pdist[3], &pdist[4],
          &pdist[5], &pdist[6], &pdist[7], &pdist[8], &pdist[9], &pdist[10]);
      if (got >= 8 && Params->nDetParams < MAX_N_DETS) {
        int slot = Params->nDetParams;
        Params->DetIDs[slot] = det_id;
        Params->DetLsd[slot] = lsd;
        Params->DetYbc[slot] = y_bc;
        Params->DetZbc[slot] = z_bc;
        Params->DetTx[slot] = tx;
        Params->DetTy[slot] = ty;
        Params->DetTz[slot] = tz;
        for (int k = 0; k < 11; k++) Params->DetDistortion[slot][k] = pdist[k];
        Params->nDetParams++;
      } else if (Params->nDetParams >= MAX_N_DETS) {
        fprintf(stderr,
                "Warning: DetParams row dropped — MAX_N_DETS=%d exceeded.\n",
                MAX_N_DETS);
      }
      continue;
    }
    if (strncmp(line, "RingRadii_Det", strlen("RingRadii_Det")) == 0) {
      int det_id = -1, ring_nr = -1;
      double radius = 0;
      if (sscanf(line, "RingRadii_Det%d %d %lf", &det_id, &ring_nr, &radius) == 3
          && ring_nr >= 0 && ring_nr < MAX_N_RINGS) {
        int slot = -1;
        for (int k = 0; k < Params->nDetParams; k++) {
          if (Params->DetIDs[k] == det_id) { slot = k; break; }
        }
        /* Allow RingRadii_Det without a preceding DetParams row by
         * auto-registering an empty panel slot. */
        if (slot == -1 && Params->nDetParams < MAX_N_DETS) {
          slot = Params->nDetParams;
          Params->DetIDs[slot] = det_id;
          Params->nDetParams++;
        }
        if (slot >= 0) {
          Params->RingRadiiPerDet[slot][ring_nr] = radius;
          Params->hasRingRadiiPerDet = 1;
        }
      }
      continue;
    }
    if (strncmp(line, "EtaCoverage_Det", strlen("EtaCoverage_Det")) == 0) {
      int det_id = -1, ring_nr = -1;
      double eta_lo = 0, eta_hi = 0;
      if (sscanf(line, "EtaCoverage_Det%d %d %lf %lf",
                 &det_id, &ring_nr, &eta_lo, &eta_hi) == 4) {
        int slot = -1;
        for (int k = 0; k < Params->nDetParams; k++) {
          if (Params->DetIDs[k] == det_id) { slot = k; break; }
        }
        if (slot == -1 && Params->nDetParams < MAX_N_DETS) {
          slot = Params->nDetParams;
          Params->DetIDs[slot] = det_id;
          Params->nDetParams++;
        }
        if (slot >= 0 && Params->nEtaArcs[slot] < MAX_ETA_ARCS_PER_DET) {
          int i = Params->nEtaArcs[slot];
          Params->EtaArcRing[slot][i] = ring_nr;
          Params->EtaArcLo[slot][i] = eta_lo;
          Params->EtaArcHi[slot][i] = eta_hi;
          Params->nEtaArcs[slot]++;
          Params->hasEtaCoverage = 1;
        }
      }
      continue;
    }
    /* Empty / unknown line — skip (BigDetSize dropped per plan ruling #5). */
    str = "";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) continue;
    printf("Warning: skipping line in parameters file:\n");
    printf("%s\n", line);
  }
  fclose(fp);
  int i;
  for (i = 0; i < MAX_N_RINGS; i++) Params->RingRadii[i] = 0;
  for (i = 0; i < Params->NrOfRings; i++)
    Params->RingRadii[Params->RingNumbers[i]] = Params->RingRadiiUser[i];
  return 0;
}

/* ----------------------------------------------------------------------------
 * ReadBins — int64 always (plan ruling #10). Spots.bin = 10 cols float64.
 * --------------------------------------------------------------------------*/
static int ReadBins(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char file_name[2048];
  sprintf(file_name, "%s/Data.bin", cwd);
  fd = open(file_name, O_RDONLY);
  check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  data = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(data == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  int fd2;
  struct stat s2;
  int status2;
  char file_name2[2048];
  sprintf(file_name2, "%s/nData.bin", cwd);
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  ndata = mmap(0, size2, PROT_READ, MAP_SHARED, fd2, 0);
  check(ndata == MAP_FAILED, "mmap %s failed: %s", file_name2,
        strerror(errno));
  printf("DataSize: %lld %d Nelems: %lld \n", (long long int)size,
         (int)sizeof(*data), (long long int)(size / sizeof(*data)));
  printf("nDataSize: %lld %d Nelems: %lld \n", (long long int)size2,
         (int)sizeof(*ndata), (long long int)(size2 / sizeof(*ndata)));
  fflush(stdout);
  return 1;
}

/* ReadSpots — returns nrows (PF stride = 10 doubles). */
static int ReadSpots(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/Spots.bin", cwd);
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  ObsSpotsLab = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(ObsSpotsLab == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  size_t nrsps = size / sizeof(double) / 10;
  return (int)nrsps;
}

/* ----------------------------------------------------------------------------
 * Phase 9 scaffolding: load Spots_det.bin if present. Sets the global
 * SpotsDetID array to a malloc'd int32 buffer; caller is responsible
 * for free() at shutdown (or just let the process exit reclaim it).
 * Returns 1 if loaded, 0 if file absent. Length must match n_spots.
 * --------------------------------------------------------------------------*/
static int ReadSpotsDet(char *cwd, size_t expected_n_spots) {
  char filename[2048];
  snprintf(filename, sizeof(filename), "%s/Spots_det.bin", cwd);
  if (access(filename, F_OK) != 0) {
    return 0;  /* single-detector fallback */
  }
  struct stat s;
  int fd = open(filename, O_RDONLY);
  if (fd < 0) return 0;
  if (fstat(fd, &s) < 0) { close(fd); return 0; }
  size_t n_entries = (size_t)s.st_size / sizeof(int32_t);
  if (n_entries != expected_n_spots) {
    fprintf(stderr,
            "Warning: Spots_det.bin has %zu entries but n_spots=%zu; "
            "ignoring multi-detector side-car.\n",
            n_entries, expected_n_spots);
    close(fd);
    return 0;
  }
  SpotsDetID = (int32_t *)malloc(s.st_size);
  if (!SpotsDetID) { close(fd); return 0; }
  ssize_t r = read(fd, SpotsDetID, s.st_size);
  close(fd);
  if (r != s.st_size) { free(SpotsDetID); SpotsDetID = NULL; return 0; }
  SpotsDetID_size = n_entries;
  printf("Spots_det.bin loaded: %zu entries.\n", n_entries);
  return 1;
}

/* ----------------------------------------------------------------------------
 * usage() — printed on CLI misuse. Required by MIDAS C-style.
 * --------------------------------------------------------------------------*/
static void usage(const char *progname) {
  fprintf(stderr,
          "\nUsage: %s paramstest.txt blockNr nBlocks nWork numProcs\n"
          "\n"
          "  paramstest.txt   path to parameter file (.../OutputFolder/...)\n"
          "  blockNr          0-based block index for sharding\n"
          "  nBlocks          total number of blocks\n"
          "  nWork            FF mode: nSpotsToIndex (legacy IndexerOMP\n"
          "                   semantics); PF mode: numScans (legacy\n"
          "                   IndexerScanningOMP semantics).\n"
          "  numProcs         OpenMP thread count\n"
          "\n"
          "Auto-detect: PF iff `positions.csv` exists in\n"
          "  dirname(OutputFolder) AND nWork > 1.\n",
          progname);
}

/* ----------------------------------------------------------------------------
 * main — driver loop. PF parallel-over-voxels; FF parallel-over-spots
 * (treated as nVoxels = nSpotsToIndex with one voxel per spot). Both paths
 * use the same OpenMP shell with thread-local IndexerScratch and write
 * through VoxelAccumulator + WriteConsolidatedFiles.
 * --------------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tmidas_indexer (%s)\nContact hsharma@anl.gov in case of "
         "questions about the MIDAS project.\n\n",
         MIDAS_VERSION_STRING);
  if (argc < 6) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  char *ParamFN = argv[1];
  int blockNr = atoi(argv[2]);
  int nBlocks = atoi(argv[3]);
  int nWork = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  printf("Reading parameters from file: %s.\n", ParamFN);

  struct TParams Params;
  int returncode = ReadParams(ParamFN, &Params);
  if (returncode != 0) {
    printf("Error reading params file %s\n", ParamFN);
    exit(EXIT_FAILURE);
  }

  /* Pre-compute per-ring etamargin (used in all modes). */
  for (int i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] == 0) etamargins[i] = 0;
    else
      etamargins[i] = rad2deg * atan(Params.MarginEta / Params.RingRadii[i]) +
                      0.5 * Params.StepsizeOrient;
  }

  /* Resolve the dataset directory (where Spots.bin/Data.bin/positions.csv
   * live). Note: dirname() may modify its argument; copy first. */
  char outputFolderTmp[4096];
  strcpy(outputFolderTmp, Params.OutputFolder);
  char *cwdstr = dirname(outputFolderTmp);

  /* Mode auto-detect: PF iff positions.csv has >1 row. FF if it has 1 row
   * (with "0.000000" — the convention for the unified format) or is absent.
   * In PF mode, numScans = number of rows in positions.csv (NOT nWork).
   * In FF mode, nWork = nSpotsToIndex (legacy IndexerOMP arg4 semantics). */
  char posfn[4096];
  snprintf(posfn, sizeof(posfn), "%s/positions.csv", cwdstr);
  int hasPositions = (access(posfn, F_OK) == 0);
  int nPositions = 0;
  if (hasPositions) {
    FILE *pf = fopen(posfn, "r");
    if (pf) {
      char tmpline[1024];
      while (fgets(tmpline, sizeof(tmpline), pf) != NULL) {
        /* Skip blank lines */
        char *p = tmpline;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\0') continue;
        nPositions++;
      }
      fclose(pf);
    }
  }
  int isPF = (nPositions > 1);
  numScans = isPF ? nPositions : 1;
  if (isPF && nWork != nPositions) {
    fprintf(stderr,
            "WARNING: PF mode auto-detected with nPositions=%d, but argv "
            "nWork=%d. Using nPositions; argv ignored for PF.\n",
            nPositions, nWork);
  }

  /* hkls.csv — 11 cols, populate the PF 10-col layout. */
  const char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  if (!hklf) {
    fprintf(stderr, "Could not open hkls.csv\n");
    exit(EXIT_FAILURE);
  }
  char aline[5024], dummy_h[1024];
  if (fgets(aline, 1000, hklf) == NULL) {
    fprintf(stderr, "hkls.csv empty\n");
    fclose(hklf);
    exit(EXIT_FAILURE);
  }
  int Rnr;
  int hi, ki, li;
  double hc, kc, lc, RRd, Ds, tht, tth;
  while (fgets(aline, 1000, hklf) != NULL) {
    /* Accept FF 11-col (numeric ringRad) format. */
    if (sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %lf %lf", &hi, &ki, &li,
               &Ds, &Rnr, &hc, &kc, &lc, &tht, &tth, &RRd) < 11) {
      /* Fall back to PF format with a string in col 10 (legacy variant). */
      if (sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf", &hi, &ki, &li,
                 &Ds, &Rnr, &hc, &kc, &lc, &tht, dummy_h, &RRd) < 11) {
        continue;
      }
      tth = 2.0 * tht;
    }
    RingHKL[Rnr][0] = hc;
    RingHKL[Rnr][1] = kc;
    RingHKL[Rnr][2] = lc;
    RingTtheta[Rnr] = tth;
    for (int i = 0; i < Params.NrOfRings; i++) {
      if (Rnr == Params.RingNumbers[i]) {
        HKLints[n_hkls][0] = hi;
        HKLints[n_hkls][1] = ki;
        HKLints[n_hkls][2] = li;
        HKLints[n_hkls][3] = Rnr;
        hkls[n_hkls][0] = hc;
        hkls[n_hkls][1] = kc;
        hkls[n_hkls][2] = lc;
        hkls[n_hkls][3] = (double)Rnr;
        hkls[n_hkls][4] = Ds;
        hkls[n_hkls][5] = tht;
        hkls[n_hkls][6] = RRd;
        hkls[n_hkls][7] = sin(tht * deg2rad);
        RealType len = sqrt(hc * hc + kc * kc + lc * lc);
        RealType v = hkls[n_hkls][7] * len;
        hkls[n_hkls][8] = v;
        hkls[n_hkls][9] = v * v;
        n_hkls++;
      }
    }
    if (n_hkls >= MAX_N_HKLS) {
      fprintf(stderr, "Error: too many HKLs (max %d).\n", MAX_N_HKLS);
      fclose(hklf);
      exit(EXIT_FAILURE);
    }
  }
  fclose(hklf);
  printf("No of hkls: %d\n", n_hkls);

  n_spots = (size_t)ReadSpots(cwdstr);
  printf("nSpots = %d\n", (int)n_spots);
  ReadBins(cwdstr);
  /* Phase 9: load per-spot DetID side-car if multi-detector. Single-det
   * runs (Params.nDetParams == 0) skip the load — SpotsDetID stays NULL
   * and code paths treat every spot as belonging to a single global
   * panel (current behavior). */
  if (Params.nDetParams > 0) {
    ReadSpotsDet(cwdstr, n_spots);
    printf("Multi-detector mode: %d panels parsed.\n", Params.nDetParams);
  }

  int HighestRingNo = 0;
  for (int i = 0; i < MAX_N_RINGS; i++)
    if (Params.RingRadii[i] != 0) HighestRingNo = i;
  n_ring_bins = HighestRingNo;
  n_eta_bins = (int)ceil(360.0 / Params.EtaBinSize);
  n_ome_bins = (int)ceil(360.0 / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize;
  OmeBinSize = Params.OmeBinSize;
  Params.InvEtaBinSize = 1.0 / EtaBinSize;
  Params.InvOmeBinSize = 1.0 / OmeBinSize;
  printf("Binning: ring=%d eta=%d ome=%d total=%lld\n", n_ring_bins,
         n_eta_bins, n_ome_bins,
         (long long)n_ring_bins * n_eta_bins * n_ome_bins);

  /* positions.csv — PF needs it; FF synthesizes a single 0.0. */
  if (isPF) {
    FILE *positionsF = fopen(posfn, "r");
    if (!positionsF) {
      fprintf(stderr, "positions.csv missing in PF mode.\n");
      exit(EXIT_FAILURE);
    }
    ypos = (double *)malloc(numScans * sizeof(double));
    for (int i = 0; i < numScans; i++) {
      if (fgets(aline, 1000, positionsF) == NULL) {
        fprintf(stderr, "positions.csv has fewer rows than nScans=%d\n",
                numScans);
        exit(EXIT_FAILURE);
      }
      sscanf(aline, "%lf", &ypos[i]);
    }
    fclose(positionsF);
  } else {
    ypos = (double *)malloc(sizeof(double));
    ypos[0] = 0.0;
  }

  /* Mode-specific setup. */
  int nVoxels = 0;
  int startVoxel = 0, endVoxel = 0;

  /* PF: voxel grid = sorted positions × sorted positions. */
  if (isPF) {
    nVoxels = numScans * numScans;
    grid = (double *)malloc(nVoxels * 2 * sizeof(double));
    double *ypos_sorted = (double *)malloc(numScans * sizeof(double));
    memcpy(ypos_sorted, ypos, numScans * sizeof(double));
    qsort(ypos_sorted, numScans, sizeof(double), cmp_double_asc);
    for (int i = 0; i < numScans; i++) {
      for (int j = 0; j < numScans; j++) {
        grid[(i * numScans + j) * 2 + 0] = ypos_sorted[i];
        grid[(i * numScans + j) * 2 + 1] = ypos_sorted[j];
      }
    }
    free(ypos_sorted);
    startVoxel = (int)(ceil((double)nVoxels / (double)nBlocks)) * blockNr;
    int tmp = (int)(ceil((double)nVoxels / (double)nBlocks)) * (blockNr + 1);
    endVoxel = tmp < nVoxels ? tmp : nVoxels;
  } else {
    /* FF: nVoxels = nSpotsToIndex; one synthetic voxel per spot. */
    nVoxels = nWork;
    startVoxel = (int)(ceil((double)nVoxels / (double)nBlocks)) * blockNr;
    int tmp = (int)(ceil((double)nVoxels / (double)nBlocks)) * (blockNr + 1);
    /* Legacy FF used (endRowNr - startRowNr + 1) semantics — but we want
     * pure half-open intervals for the consolidated path. Cap at nVoxels. */
    endVoxel = tmp < nVoxels ? tmp : nVoxels;
  }
  int nLocal = endVoxel - startVoxel;
  if (nLocal < 0) nLocal = 0;
  printf("Mode: %s | numScans=%d nVoxels=%d blockNr=%d nBlocks=%d "
         "[startVoxel,endVoxel)=[%d,%d) numProcs=%d\n",
         isPF ? "PF" : "FF", numScans, nVoxels, blockNr, nBlocks, startVoxel,
         endVoxel, numProcs);

  /* ---- PF-only: load optional MicFile / GrainsFile and pre-compute
   *      per-spot sin/cosOme for the seed scan filter. ---- */
  int hasMic = 0, nrMic = 0;
  double *mic = NULL;
  if (isPF && strncmp(Params.MicFN, "0", strlen("0"))) {
    hasMic = 1;
    mic = (double *)calloc(MAX_MIC_ROWS * 5, sizeof(double));
    FILE *micF = fopen(Params.MicFN, "r");
    if (!micF) {
      fprintf(stderr, "MicFile %s missing.\n", Params.MicFN);
      exit(EXIT_FAILURE);
    }
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    while (fgets(aline, 5000, micF) != NULL) {
      sscanf(aline, "%s %s %s %lf %lf %s %s %lf %lf %lf %s %s", dummy_h,
             dummy_h, dummy_h, &mic[nrMic * 5 + 0], &mic[nrMic * 5 + 1],
             dummy_h, dummy_h, &mic[nrMic * 5 + 2], &mic[nrMic * 5 + 3],
             &mic[nrMic * 5 + 4], dummy_h, dummy_h);
      nrMic++;
    }
    fclose(micF);
    mic = (double *)realloc(mic, nrMic * 5 * sizeof(double));
  }

  int hasGrainsPF = 0, nrGrainsPF = 0;
  double *grainsOM_PF = NULL;
  /* PF GrainsFile semantics: per-voxel seed each provided OM (DoIndexing_Seeded
   * loop). Only activate this PF-style path when we're in PF mode AND the
   * legacy isGrainsInput flag is set. (FF mode's GrainsFile path is handled
   * below via DoIndexing_Seeded on a FF-style flat orientation+position list.)
   */
  if (isPF && Params.isGrainsInput) {
    hasGrainsPF = 1;
    grainsOM_PF = (double *)calloc(MAX_MIC_ROWS * 9, sizeof(double));
    FILE *grainsF = fopen(Params.GrainsFN, "r");
    if (!grainsF) {
      fprintf(stderr, "GrainsFile %s missing (PF).\n", Params.GrainsFN);
      exit(EXIT_FAILURE);
    }
    while (fgets(aline, 5000, grainsF) != NULL) {
      if (aline[0] == '%') continue;
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf", dummy_h,
             &grainsOM_PF[nrGrainsPF * 9 + 0],
             &grainsOM_PF[nrGrainsPF * 9 + 1],
             &grainsOM_PF[nrGrainsPF * 9 + 2],
             &grainsOM_PF[nrGrainsPF * 9 + 3],
             &grainsOM_PF[nrGrainsPF * 9 + 4],
             &grainsOM_PF[nrGrainsPF * 9 + 5],
             &grainsOM_PF[nrGrainsPF * 9 + 6],
             &grainsOM_PF[nrGrainsPF * 9 + 7],
             &grainsOM_PF[nrGrainsPF * 9 + 8]);
      nrGrainsPF++;
    }
    fclose(grainsF);
    grainsOM_PF = (double *)realloc(grainsOM_PF, nrGrainsPF * 9 * sizeof(double));
    printf("Read %d grains from %s (PF seeded)\n", nrGrainsPF, Params.GrainsFN);
  }

  /* PF seed scan: per-spot sin/cosOme + RingToIndex row range. */
  double *spotSinOme = NULL;
  double *spotCosOme = NULL;
  size_t startRowNrSp = MAX_N_SPOTS, endRowNrSp = 0;
  if (isPF && !hasMic && !hasGrainsPF) {
    int RingToIndex = Params.RingToIndex;
    for (size_t i = 0; i < n_spots; i++) {
      if ((int)ObsSpotsLab[i * 10 + 5] == RingToIndex && startRowNrSp > i)
        startRowNrSp = i;
      if ((int)ObsSpotsLab[i * 10 + 5] == RingToIndex && endRowNrSp < i)
        endRowNrSp = i;
    }
    spotSinOme = (double *)malloc(n_spots * sizeof(double));
    spotCosOme = (double *)malloc(n_spots * sizeof(double));
    for (size_t i = 0; i < n_spots; i++) {
      double omeRad = deg2rad * ObsSpotsLab[i * 10 + 2];
      spotSinOme[i] = sin(omeRad);
      spotCosOme[i] = cos(omeRad);
    }
  }

  /* ---- FF-only: load SpotsToIndex.csv or, when isGrainsInput, the seeded
   *      Grains.csv (FF semantics — separate orientations + positions arrays).
   *      Then map "voxel v in [startVoxel, endVoxel)" to a SpotID + obs row. */
  int *FF_SpotIDs = NULL;
  int *FF_SpotRowNos = NULL;
  RealType **FF_orients = NULL;
  RealType **FF_positions = NULL;
  int FF_nGrains = 0;
  if (!isPF) {
    if (Params.isGrainsInput) {
      FILE *inpF = fopen(Params.GrainsFileName, "r");
      if (!inpF) {
        fprintf(stderr, "FF GrainsFile %s missing.\n", Params.GrainsFileName);
        exit(EXIT_FAILURE);
      }
      fgets(aline, 4096, inpF);
      sscanf(aline, "%s %d", dummy_h, &FF_nGrains);
      FF_orients = allocMatrix(FF_nGrains, 9);
      FF_positions = allocMatrix(FF_nGrains, 3);
      for (int i = 0; i < 8; i++) fgets(aline, 4096, inpF);
      int grainNr = 0;
      double tmpD;
      while (fgets(aline, 4096, inpF) != NULL && grainNr < FF_nGrains) {
        sscanf(aline,
               "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %s "
               "%s %s %s %s %s %lf",
               dummy_h, &FF_orients[grainNr][0], &FF_orients[grainNr][1],
               &FF_orients[grainNr][2], &FF_orients[grainNr][3],
               &FF_orients[grainNr][4], &FF_orients[grainNr][5],
               &FF_orients[grainNr][6], &FF_orients[grainNr][7],
               &FF_orients[grainNr][8], &FF_positions[grainNr][0],
               &FF_positions[grainNr][1], &FF_positions[grainNr][2], dummy_h,
               dummy_h, dummy_h, dummy_h, dummy_h, dummy_h, dummy_h, dummy_h,
               dummy_h, &tmpD);
        grainNr++;
      }
      fclose(inpF);
      /* In FF GrainsFile mode, "voxel v" indexes a (orient, position) pair. */
      nVoxels = FF_nGrains;
      startVoxel = (int)(ceil((double)nVoxels / (double)nBlocks)) * blockNr;
      int tmp = (int)(ceil((double)nVoxels / (double)nBlocks)) * (blockNr + 1);
      endVoxel = tmp < nVoxels ? tmp : nVoxels;
      nLocal = endVoxel - startVoxel;
      printf("FF seeded: %d grains, voxels [%d,%d).\n", FF_nGrains, startVoxel,
             endVoxel);
    } else {
      FILE *spotsFile = fopen("SpotsToIndex.csv", "r");
      if (!spotsFile) {
        fprintf(stderr, "SpotsToIndex.csv missing.\n");
        exit(EXIT_FAILURE);
      }
      FF_SpotIDs = (int *)malloc(nVoxels * sizeof(int));
      for (int i = 0; i < nVoxels; i++) FF_SpotIDs[i] = -1;
      for (int i = 0; i < nVoxels; i++) {
        if (fgets(aline, 1000, spotsFile) == NULL) break;
        sscanf(aline, "%d", &FF_SpotIDs[i]);
      }
      fclose(spotsFile);
      /* Pre-resolve SpotID -> SpotRowNo once (O(N·nVoxels) is acceptable for
       * typical FF nVoxels and is what the legacy main loop does per call).
       * Note: FindInMatrix scans ObsSpotsLab[*,4] linearly. */
      FF_SpotRowNos = (int *)malloc(nVoxels * sizeof(int));
      for (int i = 0; i < nVoxels; i++) {
        if (FF_SpotIDs[i] < 0) {
          FF_SpotRowNos[i] = -1;
          continue;
        }
        int idx;
        FindInMatrix(&ObsSpotsLab[0], (int)n_spots, N_COL_OBSSPOTS, 4,
                     (RealType)FF_SpotIDs[i], &idx);
        FF_SpotRowNos[i] = idx;
      }
    }
  }

  /* Allocate per-voxel accumulators for this block. */
  VoxelAccumulator *accs =
      (VoxelAccumulator *)calloc(nLocal > 0 ? nLocal : 1,
                                  sizeof(VoxelAccumulator));
  for (int vi = 0; vi < nLocal; vi++) VoxelAccum_init(&accs[vi]);

  int thisRowNr;
#pragma omp parallel num_threads(numProcs) private(thisRowNr)
  {
    /* Thread-local IndexerScratch. Heap-allocated OrMat per plan risk note. */
    struct IndexerScratch scratch;
    int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
    int nRowsPerGrain = 2 * n_hkls;
    scratch.GrainMatches = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
    scratch.GrainMatchesT = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
    scratch.AllGrainSpots = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
    scratch.AllGrainSpotsT = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
    scratch.GrainSpots = allocMatrix(nRowsPerGrain, N_COL_GRAINSPOTS);
    scratch.TheorSpots = allocMatrix(nRowsPerGrain, N_COL_THEORSPOTS);
    scratch.IAgrainspots =
        (RealType *)malloc(nRowsOutput * sizeof(RealType));
    scratch.OrMat =
        (RealType (*)[3][3])malloc(MAX_N_OR * sizeof(*scratch.OrMat));
    if (!scratch.GrainMatches || !scratch.GrainMatchesT ||
        !scratch.AllGrainSpots || !scratch.AllGrainSpotsT ||
        !scratch.GrainSpots || !scratch.TheorSpots || !scratch.IAgrainspots ||
        !scratch.OrMat) {
      fprintf(stderr, "Thread-local scratch alloc failed.\n");
      exit(EXIT_FAILURE);
    }

    /* (Outer #pragma must enclose the for loop; schedule(dynamic) matches PF
     * legacy.) */
#pragma omp for schedule(dynamic)
    for (thisRowNr = startVoxel; thisRowNr < endVoxel; thisRowNr++) {
      VoxelAccumulator *acc = &accs[thisRowNr - startVoxel];
      if (isPF) {
        double xThis = grid[thisRowNr * 2 + 0];
        double yThis = grid[thisRowNr * 2 + 1];
        if (hasMic) {
          int bestRow = -1;
          double bestLen = 1e30, lenThis;
          for (int iter = 0; iter < nrMic; iter++) {
            lenThis = sqrt((xThis - mic[iter * 5 + 0]) *
                                (xThis - mic[iter * 5 + 0]) +
                            (yThis - mic[iter * 5 + 1]) *
                                (yThis - mic[iter * 5 + 1]));
            if (lenThis < bestLen) {
              bestLen = lenThis;
              bestRow = iter;
            }
          }
          if (bestRow != -1) {
            double eulerThis[3] = {mic[bestRow * 5 + 2],
                                     mic[bestRow * 5 + 3],
                                     mic[bestRow * 5 + 4]};
            double OMThis[3][3];
            Euler2OrientMat(eulerThis, OMThis);
            DoIndexing_Seeded(thisRowNr, 0, OMThis, xThis, yThis, Params, acc,
                              &scratch);
          }
        } else if (hasGrainsPF) {
          double OMThis[3][3];
          for (int iter = 0; iter < nrGrainsPF; iter++) {
            OMThis[0][0] = grainsOM_PF[iter * 9 + 0];
            OMThis[0][1] = grainsOM_PF[iter * 9 + 1];
            OMThis[0][2] = grainsOM_PF[iter * 9 + 2];
            OMThis[1][0] = grainsOM_PF[iter * 9 + 3];
            OMThis[1][1] = grainsOM_PF[iter * 9 + 4];
            OMThis[1][2] = grainsOM_PF[iter * 9 + 5];
            OMThis[2][0] = grainsOM_PF[iter * 9 + 6];
            OMThis[2][1] = grainsOM_PF[iter * 9 + 7];
            OMThis[2][2] = grainsOM_PF[iter * 9 + 8];
            DoIndexing_Seeded(thisRowNr, iter, OMThis, xThis, yThis, Params,
                              acc, &scratch);
          }
        } else {
          /* spot-driven seed mode (PF default). */
          RealType seedTol =
              (Params.ScanPosTol > 0) ? Params.ScanPosTol : (BeamSize / 2);
          for (size_t idnr = startRowNrSp; idnr <= endRowNrSp; idnr++) {
            int thisID = (int)ObsSpotsLab[idnr * 10 + 4];
            double newY = xThis * spotSinOme[idnr] + yThis * spotCosOme[idnr];
            if (fabs(newY - ypos[(int)ObsSpotsLab[idnr * 10 + 9]]) <=
                seedTol) {
              DoIndexing_PF(thisID, thisRowNr, xThis, yThis, 0, Params,
                            (int)idnr, acc, &scratch);
            }
          }
        }
      } else {
        /* FF mode. thisRowNr indexes a spot-to-index (or seeded grain row). */
        if (Params.isGrainsInput) {
          /* FF GrainsFile path: per-grain seeded — fixed OM + position. */
          int g = thisRowNr;
          double OMThis[3][3];
          for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
              OMThis[ii][jj] = FF_orients[g][ii * 3 + jj];
          DoIndexing_Seeded(g, g, OMThis, FF_positions[g][0],
                            FF_positions[g][1], Params, acc, &scratch);
        } else {
          int spotID = FF_SpotIDs[thisRowNr];
          int spotRowNo = FF_SpotRowNos[thisRowNr];
          if (spotID == -1 || spotRowNo < 0) continue;
          DoIndexing_FF(spotID, spotRowNo, Params, acc, &scratch);
        }
      }
    }
    FreeMemMatrix(scratch.GrainMatches, MAX_N_MATCHES);
    FreeMemMatrix(scratch.GrainMatchesT, MAX_N_MATCHES);
    FreeMemMatrix(scratch.AllGrainSpots, nRowsOutput);
    FreeMemMatrix(scratch.AllGrainSpotsT, nRowsOutput);
    FreeMemMatrix(scratch.GrainSpots, nRowsPerGrain);
    FreeMemMatrix(scratch.TheorSpots, nRowsPerGrain);
    free(scratch.IAgrainspots);
    free(scratch.OrMat);
  }

  /* Always write consolidated output (plan ruling #8). */
  printf("Writing consolidated output files...\n");
  WriteConsolidatedFiles(accs, nVoxels, startVoxel, endVoxel,
                         Params.OutputFolder);

  for (int vi = 0; vi < nLocal; vi++) VoxelAccum_free(&accs[vi]);
  free(accs);
  if (spotSinOme) free(spotSinOme);
  if (spotCosOme) free(spotCosOme);
  if (mic) free(mic);
  if (grainsOM_PF) free(grainsOM_PF);
  if (FF_SpotIDs) free(FF_SpotIDs);
  if (FF_SpotRowNos) free(FF_SpotRowNos);
  if (FF_orients) FreeMemMatrix(FF_orients, FF_nGrains);
  if (FF_positions) FreeMemMatrix(FF_positions, FF_nGrains);
  if (grid) free(grid);
  if (ypos) free(ypos);

  double time = omp_get_wtime() - start_time;
  printf("Finished. Mode=%s nVoxels=%d block=[%d,%d). Time: %lfs.\n",
         isPF ? "PF" : "FF", nVoxels, startVoxel, endVoxel, time);
  return 0;
}
