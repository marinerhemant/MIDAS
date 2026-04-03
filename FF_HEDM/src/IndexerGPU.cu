//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated FF-HEDM Indexer for MIDAS.
//
// Architecture (Option B — flattened evaluation):
//   CPU pre-computation (OMP): For each spotID, generate candidate
//   orientations and position steps. Flatten all (spotID, orientation,
//   position) tuples into a 1D array.
//   GPU kernel: Each thread processes one tuple — compute theoretical
//   diffraction spots, compare with observed spots via binned lookup,
//   produce a match score.
//   GPU reduction: Per-spotID atomicMax to find the best match.
//   CPU post-processing: Write IndexBest.bin / IndexBestFull.bin results.
//
// Supports both unseeded (DoIndexing) and seeded (DoIndexingSeed) modes.
//

#include <ctype.h>
#include <cuda_runtime.h>
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define RealType float
#include "midas_gpu_math.cuh"
#include "midas_version.h"

// ─────────────────────────────────────────────────────────────
// CUDA error checking
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ─────────────────────────────────────────────────────────────
// Constants and types
// ─────────────────────────────────────────────────────────────
#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)

#define MAX_N_SPOTS 100000000
#define MAX_N_OR 7200
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_STEPS 2000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS 14
#define N_COL_OBSSPOTS 9
#define N_COL_GRAINSPOTS 17
#define N_COL_GRAINMATCHES 16
#define EPS 1e-9f

#define crossProduct(a, b, c)                                                  \
  (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];                                  \
  (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];                                  \
  (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];
#define dot(v, q) ((v)[0] * (q)[0] + (v)[1] * (q)[1] + (v)[2] * (q)[2])
#define CalcLength(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

static void check_host(int test, const char *msg, ...) {
  if (test) {
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}
#define check(test, ...) check_host(test, __VA_ARGS__)

// ─────────────────────────────────────────────────────────────
// Global host data (same layout as IndexerOMP.c)
// ─────────────────────────────────────────────────────────────
static double *ObsSpotsLab_d = NULL; // mmap'd as double (file format)
static double *ObsSpotsLab = NULL;   // converted to double for CPU/GPU use
static int *data = NULL;
static int *ndata = NULL;
static size_t n_spots = 0;
static int n_hkls = 0;
static RealType hkls[MAX_N_HKLS][7];
static int HKLints[MAX_N_HKLS][4];
static RealType RingHKL[MAX_N_RINGS][3];
static RealType RingTtheta[MAX_N_RINGS];
static int n_ring_bins = 0;
static int n_eta_bins = 0;
static int n_ome_bins = 0;
static RealType EtaBinSize = 0;
static RealType OmeBinSize = 0;
static int BigDetSize = 0;
static int *BigDetector = NULL;
static long long int totNrPixelsBigDetector = 0;
static double pixelsize = 0; // sscanf %lf writes here
static double ABCABG[6];
static int SGNum = 0;

// ─────────────────────────────────────────────────────────────
// Parameters structure (same as IndexerOMP)
// ─────────────────────────────────────────────────────────────
struct TParams {
  int RingNumbers[MAX_N_RINGS];
  int SpaceGroupNum;
  double LatticeConstant; // Must be double: ReadParams uses sscanf %lf
  double Wavelength;
  double Distance;
  double Rsample;
  double Hbeam;
  double StepsizePos;
  double StepsizeOrient;
  int NrOfRings;
  double RingRadii[MAX_N_RINGS];
  double RingRadiiUser[MAX_N_RINGS];
  double MarginOme;
  double MarginEta;
  double MarginRad;
  double MarginRadial;
  double EtaBinSize;
  double OmeBinSize;
  double ExcludePoleAngle;
  double MinMatchesToAcceptFrac;
  double BoxSizes[MAX_N_OMEGARANGES][4];
  double OmegaRanges[MAX_N_OMEGARANGES][2];
  char OutputFolder[4096];
  int NoOfOmegaRanges;
  int isGrainsInput;
  char GrainsFileName[4096];
  char SpotsFileName[4096];
  char IDsFileName[4096];
  int UseFriedelPairs;
  int RingsToReject[MAX_N_RINGS];
  int nRingsToRejectCalc;
  int IndexBestFD;
  int IndexBestFullFD;
};

// ─────────────────────────────────────────────────────────────
// A flattened evaluation tuple (CPU → GPU)
// ─────────────────────────────────────────────────────────────
struct EvalTuple {
  int spotIdx;       // Index back to spotID array (for reduction)
  double OrMat[9];   // Orientation matrix (flattened 3x3)
  double ga, gb, gc; // Sample position in lab frame
  double RefRad;     // Reference radial position of the spot
};

// Per-spotID best result (GPU output, one per spotID)
struct SpotResult {
  unsigned long long atomicKey; // Packed (frac, -IA) for 64-bit atomicCAS —
                                // MUST be first for alignment
  double bestFrac;              // Best fraction of matches (confidence)
  double bestIA;                // Best internal angle
  double bestOrMat[9];          // Best orientation matrix
  double bestPos[3];            // Best position (ga, gb, gc)
  int nTspots;                  // Number of theoretical spots for best match
  int nMatches;                 // Number of matches for best match
};

// ─────────────────────────────────────────────────────────────
// GPU constant memory for geometry/margins
// ─────────────────────────────────────────────────────────────
__device__ RealType c_RingRadii[MAX_N_RINGS];
__device__ RealType c_OmegaRanges[MAX_N_OMEGARANGES][2];
__device__ RealType c_BoxSizes[MAX_N_OMEGARANGES][4];
__device__ RealType c_omemargins[181];
__device__ RealType c_etamargins[MAX_N_RINGS];
__device__ int c_ringsToReject[MAX_N_RINGS];

// Scalar constants in constant memory
struct GPUParams {
  RealType Distance;
  RealType Wavelength;
  RealType ExcludePoleAngle;
  RealType MarginRad;
  RealType MarginRadial;
  RealType MarginOme;
  int NoOfOmegaRanges;
  int nRingsToRejectCalc;
  int n_ring_bins;
  int n_eta_bins;
  int n_ome_bins;
  RealType EtaBinSize;
  RealType OmeBinSize;
};
__constant__ GPUParams c_params;

// ─────────────────────────────────────────────────────────────
// Device: CalcDiffrSpots_Furnace — compute theoretical spots
// from an orientation matrix against HKL table
// ─────────────────────────────────────────────────────────────
__device__ int
gpu_CalcDiffrSpots(const RealType OrMat[3][3],
                   const RealType *d_hkls_flat, // [n_hkls × 7]
                   int n_hkls_d,
                   RealType *spots_out, // flat: [max_spots × N_COL_THEORSPOTS]
                   int max_spots, const int *d_ringsToReject,
                   int nRingsToRejectCalc, int *nTspotsFracCalc) {
  int spotnr = 0;
  int nFracCalc = 0;

  for (int ih = 0; ih < n_hkls_d && spotnr < max_spots; ih++) {
    RealType Ghkl[3] = {d_hkls_flat[ih * 7 + 0], d_hkls_flat[ih * 7 + 1],
                        d_hkls_flat[ih * 7 + 2]};
    int ringnr = (int)d_hkls_flat[ih * 7 + 3];
    if (ringnr < 0 || ringnr >= MAX_N_RINGS)
      continue;
    RealType RingRadius = c_RingRadii[ringnr];
    if (RingRadius < EPS)
      continue;
    RealType theta = d_hkls_flat[ih * 7 + 5];

    RealType Gc[3];
    RealType OrM[3][3];
    for (int r = 0; r < 3; r++)
      for (int cc = 0; cc < 3; cc++)
        OrM[r][cc] = OrMat[r][cc];
    midas_MatrixMultF(OrM, Ghkl, Gc);

    RealType omegas[4], etas[4];
    int nsol;
    midas_CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nsol);

    for (int i = 0; i < nsol && spotnr < max_spots; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      RealType EtaAbs = fabs(Eta);

      if (EtaAbs < c_params.ExcludePoleAngle ||
          (180.0 - EtaAbs) < c_params.ExcludePoleAngle)
        continue;

      RealType yl, zl;
      midas_CalcSpotPosition(RingRadius, Eta, &yl, &zl);

      int keep = 0;
      for (int orn = 0; orn < c_params.NoOfOmegaRanges; orn++) {
        if (Omega > c_OmegaRanges[orn][0] && Omega < c_OmegaRanges[orn][1] &&
            yl > c_BoxSizes[orn][0] && yl < c_BoxSizes[orn][1] &&
            zl > c_BoxSizes[orn][2] && zl < c_BoxSizes[orn][3]) {
          keep = 1;
          break;
        }
      }
      if (!keep)
        continue;

      RealType *sp = &spots_out[spotnr * N_COL_THEORSPOTS];
      sp[0] = 0;
      sp[1] = (RealType)spotnr;
      sp[2] = (RealType)ih;
      sp[3] = c_params.Distance;
      sp[4] = yl;
      sp[5] = zl;
      sp[6] = Omega;
      sp[7] = Eta;
      sp[8] = theta;
      sp[9] = (RealType)ringnr;

      // Check if ring is excluded from fraction calc
      int rejected = 0;
      for (int rr = 0; rr < nRingsToRejectCalc; rr++) {
        if (ringnr == d_ringsToReject[rr]) {
          rejected = 1;
          break;
        }
      }
      if (!rejected)
        nFracCalc++;
      spotnr++;
    }
  }
  *nTspotsFracCalc = nFracCalc;
  return spotnr;
}

// ─────────────────────────────────────────────────────────────
// Device: CompareSpots — match theoretical vs observed via bins
// ─────────────────────────────────────────────────────────────
__device__ int
gpu_CompareSpots(RealType *spots, // flat: [nTspots × N_COL_THEORSPOTS]
                 int nTspots,
                 const RealType *d_ObsSpotsLab, // [n_spots × 9]
                 RealType RefRad, const int *d_data, const int *d_ndata,
                 const int *d_ringsToReject, int nRingsToRejectCalc,
                 int *nMatchesFracCalc, RealType ga, RealType gb,
                 RealType gc,     // grain position for IA
                 RealType *avgIA) // output: average internal angle
{
  int nMatched = 0;
  int nMatchedFrac = 0;
  RealType iaSum = 0.0;
  int iaCount = 0;
  RealType Distance = c_params.Distance;

  for (int sp = 0; sp < nTspots; sp++) {
    RealType *s = &spots[sp * N_COL_THEORSPOTS];
    int RingNr = (int)s[9];
    if (RingNr <= 0 || RingNr >= MAX_N_RINGS)
      continue;

    RealType theorEta = s[12];
    RealType theorOme = s[6];
    RealType theorRadDiff = s[13];
    RealType theorY = s[10];
    RealType theorZ = s[11];

    int iRing = RingNr - 1;
    int iEta = (int)floor((180.0 + theorEta) / c_params.EtaBinSize);
    int iOme = (int)floor((180.0 + theorOme) / c_params.OmeBinSize);
    iEta = max(0, min(c_params.n_eta_bins - 1, iEta));
    iOme = max(0, min(c_params.n_ome_bins - 1, iOme));

    size_t Pos = (size_t)iRing;
    Pos *= (size_t)c_params.n_eta_bins;
    Pos += (size_t)iEta;
    Pos *= (size_t)c_params.n_ome_bins;
    Pos += (size_t)iOme;

    int nInBin = d_ndata[Pos * 2 + 0];
    int DataPos = d_ndata[Pos * 2 + 1];

    RealType etamargin = c_etamargins[RingNr];

    // Check if this ring is excluded from radial filter
    int skipRadialFilter = 0;
    for (int rr = 0; rr < nRingsToRejectCalc; rr++) {
      if (RingNr == d_ringsToReject[rr]) {
        skipRadialFilter = 1;
        break;
      }
    }

    int matchFound = 0;
    int bestSpotRow = -1;
    RealType diffOmeBest = 100000.0; // match CPU: find closest, no threshold

    for (int is = 0; is < nInBin; is++) {
      int spotRow = d_data[DataPos + is];
      int base = spotRow * N_COL_OBSSPOTS;

      // Filter 1: radial difference (MarginRadial)
      RealType obsRadDiff = d_ObsSpotsLab[base + 8];
      if (fabs(theorRadDiff - obsRadDiff) >= c_params.MarginRadial)
        continue;

      // Filter 2: RefRad check (MarginRad) — skip if ring is excluded
      if (!skipRadialFilter) {
        RealType obsRefRad = d_ObsSpotsLab[base + 3];
        if (fabs(RefRad - obsRefRad) >= c_params.MarginRad)
          continue;
      }

      // Filter 3: eta margin
      RealType obsEta = d_ObsSpotsLab[base + 6];
      if (fabs(theorEta - obsEta) >= etamargin)
        continue;

      // Find closest omega match (no threshold)
      RealType obsOme = d_ObsSpotsLab[base + 2];
      RealType diffOme = fabs(theorOme - obsOme);
      if (diffOme < diffOmeBest) {
        diffOmeBest = diffOme;
        bestSpotRow = spotRow;
        matchFound = 1;
      }
    }

    if (matchFound) {
      nMatched++;
      int rejected = 0;
      for (int rr = 0; rr < nRingsToRejectCalc; rr++) {
        if (RingNr == d_ringsToReject[rr]) {
          rejected = 1;
          break;
        }
      }
      if (!rejected)
        nMatchedFrac++;

      // Compute internal angle for this matched spot — ALL in RealType
      int base = bestSpotRow * N_COL_OBSSPOTS;
      RealType obsY_d = (RealType)d_ObsSpotsLab[base + 0];
      RealType obsZ_d = (RealType)d_ObsSpotsLab[base + 1];
      RealType obsOme_d = (RealType)d_ObsSpotsLab[base + 2];
      RealType tY_d = (RealType)theorY, tZ_d = (RealType)theorZ,
               tO_d = (RealType)theorOme;
      RealType ga_d = (RealType)ga, gb_d = (RealType)gb, gc_d = (RealType)gc,
               Dist_d = (RealType)Distance;
      // gv1 = spot_to_gv_pos(Dist, theorY, theorZ, theorOme, ga, gb, gc)
      RealType ca1 = cos(deg2rad * tO_d), sa1 = sin(deg2rad * tO_d);
      RealType vr1x = ca1 * ga_d - sa1 * gb_d, vr1y = sa1 * ga_d + ca1 * gb_d,
               vr1z = gc_d;
      RealType xi1 = Dist_d - vr1x, yi1 = tY_d - vr1y, zi1 = tZ_d - vr1z;
      RealType l1 = sqrt(xi1 * xi1 + yi1 * yi1 + zi1 * zi1);
      RealType xn1 = xi1 / l1, yn1 = yi1 / l1, zn1 = zi1 / l1;
      RealType g1r1 = -1.0 + xn1, g2r1 = yn1;
      RealType co1 = cos(-tO_d * deg2rad), so1 = sin(-tO_d * deg2rad);
      RealType gv1x_d = g1r1 * co1 - g2r1 * so1,
               gv1y_d = g1r1 * so1 + g2r1 * co1, gv1z_d = zn1;
      // gv2 = spot_to_gv_pos(Dist, obsY, obsZ, obsOme, ga, gb, gc)
      RealType ca2 = cos(deg2rad * obsOme_d), sa2 = sin(deg2rad * obsOme_d);
      RealType vr2x = ca2 * ga_d - sa2 * gb_d, vr2y = sa2 * ga_d + ca2 * gb_d,
               vr2z = gc_d;
      RealType xi2 = Dist_d - vr2x, yi2 = obsY_d - vr2y, zi2 = obsZ_d - vr2z;
      RealType l2 = sqrt(xi2 * xi2 + yi2 * yi2 + zi2 * zi2);
      RealType xn2 = xi2 / l2, yn2 = yi2 / l2, zn2 = zi2 / l2;
      RealType g1r2 = -1.0 + xn2, g2r2 = yn2;
      RealType co2 = cos(-obsOme_d * deg2rad), so2 = sin(-obsOme_d * deg2rad);
      RealType gv2x_d = g1r2 * co2 - g2r2 * so2,
               gv2y_d = g1r2 * so2 + g2r2 * co2, gv2z_d = zn2;
      // CalcInternalAngle in RealType
      RealType la = sqrt(gv1x_d * gv1x_d + gv1y_d * gv1y_d + gv1z_d * gv1z_d);
      RealType lb = sqrt(gv2x_d * gv2x_d + gv2y_d * gv2y_d + gv2z_d * gv2z_d);
      RealType dp =
          (gv1x_d * gv2x_d + gv1y_d * gv2y_d + gv1z_d * gv2z_d) / (la * lb);
      if (dp > 1.0)
        dp = 1.0;
      if (dp < -1.0)
        dp = -1.0;
      RealType ia = (RealType)(rad2deg * acos(dp));
      if (ia < 999.0) {
        iaSum += ia;
        iaCount++;
      }
    }
  }
  *nMatchesFracCalc = nMatchedFrac;
  *avgIA = (iaCount > 0) ? (iaSum / (RealType)iaCount) : 999.0;
  return nMatched;
}

// ─────────────────────────────────────────────────────────────
// Main evaluation kernel: one thread = one (orientation, position) tuple
// ─────────────────────────────────────────────────────────────
__global__ void indexer_eval_kernel(
    const EvalTuple *tuples, int nTuples, const RealType *d_hkls_flat,
    int n_hkls_d, const RealType *d_ObsSpotsLab, const int *d_data,
    const int *d_ndata, const int *d_ringsToReject, int nRingsToRejectCalc,
    SpotResult *d_results,    // [nSpotIDs] — per-spotID best
    RealType *d_theorScratch, // [batchSize × maxTheorSpots × N_COL_THEORSPOTS]
    int maxTheorSpots) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nTuples)
    return;

  EvalTuple t = tuples[tid];
  int spotIdx = t.spotIdx;

  // Build orientation matrix
  RealType OrMat[3][3];
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      OrMat[r][c] = t.OrMat[r * 3 + c];

  // Per-thread scratch slice in global memory
  RealType *TheorSpots =
      &d_theorScratch[(long long)tid * maxTheorSpots * N_COL_THEORSPOTS];

  // 1. Compute theoretical diffraction spots
  int nTspots, nTspotsFracCalc;
  nTspots = gpu_CalcDiffrSpots(OrMat, d_hkls_flat, n_hkls_d, TheorSpots,
                               maxTheorSpots, d_ringsToReject,
                               nRingsToRejectCalc, &nTspotsFracCalc);
  if (nTspots == 0 || nTspotsFracCalc == 0)
    return;

  // 2. Apply displacement for this position
  for (int sp = 0; sp < nTspots; sp++) {
    RealType *s = &TheorSpots[sp * N_COL_THEORSPOTS];
    RealType Displ_y, Displ_z;
    midas_displacement_spot_COM(t.ga, t.gb, t.gc, s[3], s[4], s[5], s[6],
                                &Displ_y, &Displ_z);
    s[10] = s[4] + Displ_y;
    s[11] = s[5] + Displ_z;
    midas_CalcEtaAngle(s[10], s[11], &s[12]);
    int rn = (int)s[9];
    s[13] = sqrt(s[10] * s[10] + s[11] * s[11]) - c_RingRadii[rn];
  }

  // 3. Compare with observed spots + compute IA
  int nMatchesFracCalc;
  RealType avgIA;
  int nMatches =
      gpu_CompareSpots(TheorSpots, nTspots, d_ObsSpotsLab, t.RefRad, d_data,
                       d_ndata, d_ringsToReject, nRingsToRejectCalc,
                       &nMatchesFracCalc, t.ga, t.gb, t.gc, &avgIA);

  RealType fracMatches = (RealType)nMatchesFracCalc / (RealType)nTspotsFracCalc;

  // 4. Atomic best-match update per spotID using 64-bit packed key:
  //    upper 32 bits = frac (maximize), lower 32 bits = -IA (minimize IA)
  //    __float_as_int requires float; cast for ordering only, actual values
  //    stored as RealType. For IA: we want lower IA → higher key. Since avgIA >
  //    0, __float_as_int is monotonically increasing for positive floats.
  //    Bitwise complement (~) inverts the ordering: lower IA → higher bits.
  //    NOTE: __float_as_int(-avgIA) is WRONG because negative float bit
  //    patterns have reversed ordering.
  unsigned int fracBits = (unsigned int)__float_as_int((float)fracMatches);
  unsigned int iaBits = ~(unsigned int)__float_as_int((float)avgIA);
  unsigned long long newKey =
      ((unsigned long long)fracBits << 32) | (unsigned long long)iaBits;

  SpotResult *res = &d_results[spotIdx];
  unsigned long long *keyAddr = &res->atomicKey;

  while (true) {
    unsigned long long oldKey = *keyAddr;
    if (newKey <= oldKey)
      break;

    unsigned long long prev = atomicCAS(keyAddr, oldKey, newKey);
    if (prev == oldKey) {
      // Won the race — copy all result fields
      res->bestFrac = fracMatches;
      res->bestIA = avgIA;
      for (int i = 0; i < 9; i++)
        res->bestOrMat[i] = t.OrMat[i];
      res->bestPos[0] = t.ga;
      res->bestPos[1] = t.gb;
      res->bestPos[2] = t.gc;
      res->nTspots = nTspots;
      res->nMatches = nMatches;
      break;
    }
  }
}

// ═════════════════════════════════════════════════════════════
//  HOST-SIDE CPU FUNCTIONS
// ═════════════════════════════════════════════════════════════

// ─── CPU math helpers (same as IndexerOMP) ──────────────────

static inline void h_CalcEtaAngle(double y, double z, double *alpha) {
  double denom = sqrt(y * y + z * z);
  if (denom < EPS) {
    *alpha = 0;
    return;
  }
  *alpha = rad2deg * acos(fmax(-1.0, fmin(1.0, z / denom)));
  if (y > 0)
    *alpha = -(*alpha);
}

// h_CalcSpotPosition removed — unused on host side

// h_MatrixMultF removed — unused on host side

static inline void h_MatrixMultF33(double m[3][3], double n[3][3],
                                   double res[3][3]) {
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      res[r][c] = m[r][0] * n[0][c] + m[r][1] * n[1][c] + m[r][2] * n[2][c];
}

static inline void h_AxisAngle2RotMatrix(double axis[3], double angle,
                                         double R[3][3]) {
  double n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (n2 < EPS) {
    R[0][0] = 1;
    R[0][1] = 0;
    R[0][2] = 0;
    R[1][0] = 0;
    R[1][1] = 1;
    R[1][2] = 0;
    R[2][0] = 0;
    R[2][1] = 0;
    R[2][2] = 1;
    return;
  }
  double inv = 1.0 / sqrt(n2);
  double u = axis[0] * inv, v = axis[1] * inv, w = axis[2] * inv;
  double rad = deg2rad * angle, co = cos(rad), si = sin(rad), omc = 1 - co;
  R[0][0] = co + u * u * omc;
  R[0][1] = -w * si + u * v * omc;
  R[0][2] = v * si + u * w * omc;
  R[1][0] = w * si + v * u * omc;
  R[1][1] = co + v * v * omc;
  R[1][2] = -u * si + v * w * omc;
  R[2][0] = -v * si + w * u * omc;
  R[2][1] = u * si + w * v * omc;
  R[2][2] = co + w * w * omc;
}

// h_CalcOmega removed — unused on host side

static double h_CalcRotationAngle(int RingNr) {
  int habs = 0, kabs = 0, labs = 0, i;
  for (i = 0; i < n_hkls; i++)
    if (HKLints[i][3] == RingNr) {
      habs = abs(HKLints[i][0]);
      kabs = abs(HKLints[i][1]);
      labs = abs(HKLints[i][2]);
      break;
    }
  int nz = 0;
  if (!habs)
    nz++;
  if (!kabs)
    nz++;
  if (!labs)
    nz++;
  if (nz == 3)
    return 0;
  if (SGNum <= 2)
    return 360;
  if (SGNum <= 15) {
    if (nz != 2)
      return 360;
    if (ABCABG[3] == 90 && ABCABG[4] == 90 && labs)
      return 180;
    if (ABCABG[3] == 90 && ABCABG[5] == 90 && habs)
      return 180;
    if (ABCABG[3] == 90 && ABCABG[5] == 90 && kabs)
      return 180;
    return 360;
  }
  if (SGNum <= 74) {
    if (nz != 2)
      return 360;
    return 180;
  }
  if (SGNum <= 142) {
    if (!nz)
      return 360;
    if (nz == 1 && !labs && habs == kabs)
      return 180;
    if (nz == 2)
      return labs ? 90 : 180;
    return 360;
  }
  if (SGNum <= 167) {
    if (!nz)
      return 360;
    if (nz == 2 && labs)
      return 120;
    return 360;
  }
  if (SGNum <= 194) {
    if (nz == 2 && labs)
      return 60;
    return 360;
  }
  if (SGNum <= 230) {
    if (nz == 2)
      return 90;
    if (nz == 1 && (habs == kabs || kabs == labs || habs == labs))
      return 180;
    if (!nz && habs == kabs && kabs == labs)
      return 120;
    return 360;
  }
  return 0;
}

static int h_GenerateCandidateOrientations(double hkl[3], double hklnormal[3],
                                           double stepsize, double *OrMat,
                                           int *nOrient, int RingNr) {
  double v[3];
  double hkl_f[3] = {(double)hkl[0], (double)hkl[1], (double)hkl[2]};
  crossProduct(v, hkl_f, hklnormal);
  double hl = CalcLength(hkl_f[0], hkl_f[1], hkl_f[2]);
  double nl = CalcLength(hklnormal[0], hklnormal[1], hklnormal[2]);
  double dp = dot(hkl_f, hklnormal);
  double angled = rad2deg * acos(fmax(-1.0, fmin(1.0, dp / (hl * nl))));
  double RM[3][3], RM2[3][3], RM3[3][3];
  h_AxisAngle2RotMatrix(v, angled, RM);
  double MaxAngle = h_CalcRotationAngle(RingNr);
  int nsteps = (int)(MaxAngle / stepsize);
  for (int o = 0; o < nsteps; o++) {
    h_AxisAngle2RotMatrix(hklnormal, o * stepsize, RM2);
    h_MatrixMultF33(RM2, RM, RM3);
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        OrMat[o * 9 + r * 3 + c] = RM3[r][c];
  }
  *nOrient = nsteps;
  return 0;
}

static inline void h_MakeUnitLength(double d, double y, double z, double *xu,
                                    double *yu, double *zu) {
  double l = sqrt(d * d + y * y + z * z);
  if (l < EPS) {
    *xu = *yu = *zu = 0;
    return;
  }
  double inv = 1.0 / l;
  *xu = d * inv;
  *yu = y * inv;
  *zu = z * inv;
}

static inline void h_spot_to_gv(double xi, double yi, double zi, double omega,
                                double *g1, double *g2, double *g3) {
  double l = sqrt(xi * xi + yi * yi + zi * zi);
  if (l < EPS) {
    *g1 = *g2 = *g3 = 0;
    return;
  }
  double xn = xi / l, yn = yi / l;
  double g1r = -1 + xn, g2r = yn;
  double co = cos(-omega * deg2rad), so = sin(-omega * deg2rad);
  *g1 = g1r * co - g2r * so;
  *g2 = g1r * so + g2r * co;
  *g3 = zi / l;
}

static inline void h_calc_n_max_min(double xi, double yi, double ys, double y0,
                                    double Rsamp, int step, int *nmax,
                                    int *nmin) {
  double dy = ys - y0, a = xi * xi + yi * yi, b = 2 * yi * dy,
         c = dy * dy - Rsamp * Rsamp;
  double D = b * b - 4 * a * c, P = sqrt(fabs(D));
  double lmax = (-b + P) / (2 * a) + 20;
  *nmax = (int)((lmax * xi) / step);
  *nmin = -(*nmax);
}

static inline void h_spot_to_unrotated(double xi, double yi, double zi,
                                       double ys, double zs, double y0,
                                       double z0, double ss, int n,
                                       double omega, double *a, double *b,
                                       double *c) {
  double lam = ss * (n / xi);
  double x1 = lam * xi, y1 = ys - y0 + lam * yi, z1 = zs - z0 + lam * zi;
  double co = cos(omega * deg2rad), so = sin(omega * deg2rad);
  *a = x1 * co + y1 * so;
  *b = y1 * co - x1 * so;
  *c = z1;
}

// ─── Data I/O (same as IndexerOMP) ─────────────────────────

static size_t ReadBigDet(char *cwd) {
  char fn[2048];
  sprintf(fn, "%s/BigDetectorMask.bin", cwd);
  int fd = open(fn, O_RDONLY);
  check(fd < 0, "open %s: %s", fn, strerror(errno));
  struct stat s;
  fstat(fd, &s);
  BigDetector = (int *)mmap(0, s.st_size, PROT_READ, MAP_SHARED, fd, 0);
  check(BigDetector == MAP_FAILED, "mmap %s: %s", fn, strerror(errno));
  return s.st_size;
}

static int ReadParams(char *fn, struct TParams *P) {
  FILE *fp = fopen(fn, "r");
  if (!fp) {
    printf("Cannot open %s\n", fn);
    return 1;
  }
  char line[4096], dummy[4096];
  int NoRingNumbers = 0, NrOfBoxSizes = 0;
  P->NrOfRings = 0;
  P->NoOfOmegaRanges = 0;
  P->isGrainsInput = 0;
  P->nRingsToRejectCalc = 0;
  totNrPixelsBigDetector = 0;
  while (fgets(line, 4096, fp)) {
    if (!strncmp(line, "RingNumbers ", 12)) {
      sscanf(line, "%s %d", dummy, &P->RingNumbers[NoRingNumbers++]);
      continue;
    }
    if (!strncmp(line, "RingsToExcludeFraction ", 23)) {
      sscanf(line, "%s %d", dummy, &P->RingsToReject[P->nRingsToRejectCalc++]);
      continue;
    }
    if (!strncmp(line, "BigDetSize ", 11)) {
      sscanf(line, "%s %d", dummy, &BigDetSize);
      totNrPixelsBigDetector = (long long)BigDetSize * BigDetSize / 32 + 1;
      continue;
    }
    if (!strncmp(line, "px ", 3)) {
      sscanf(line, "%s %lf", dummy, &pixelsize);
      continue;
    }
    if (!strncmp(line, "SpaceGroup ", 11)) {
      sscanf(line, "%s %d", dummy, &P->SpaceGroupNum);
      SGNum = P->SpaceGroupNum;
      continue;
    }
    if (!strncmp(line, "LatticeParameter ", 17) ||
        !strncmp(line, "LatticeConstant ", 16)) {
      sscanf(line, "%s %lf", dummy, &P->LatticeConstant);
      sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &ABCABG[0], &ABCABG[1],
             &ABCABG[2], &ABCABG[3], &ABCABG[4], &ABCABG[5]);
      continue;
    }
    if (!strncmp(line, "Wavelength ", 11)) {
      sscanf(line, "%s %lf", dummy, &P->Wavelength);
      continue;
    }
    if (!strncmp(line, "Distance ", 9) || !strncmp(line, "Lsd ", 4)) {
      sscanf(line, "%s %lf", dummy, &P->Distance);
      continue;
    }
    if (!strncmp(line, "Rsample ", 8)) {
      sscanf(line, "%s %lf", dummy, &P->Rsample);
      continue;
    }
    if (!strncmp(line, "Hbeam ", 6)) {
      sscanf(line, "%s %lf", dummy, &P->Hbeam);
      continue;
    }
    if (!strncmp(line, "StepsizePos ", 12)) {
      sscanf(line, "%s %lf", dummy, &P->StepsizePos);
      continue;
    }
    if (!strncmp(line, "StepsizeOrient ", 15) ||
        !strncmp(line, "StepSizeOrient ", 15)) {
      sscanf(line, "%s %lf", dummy, &P->StepsizeOrient);
      continue;
    }
    if (!strncmp(line, "MarginOme ", 10)) {
      sscanf(line, "%s %lf", dummy, &P->MarginOme);
      continue;
    }
    if (!strncmp(line, "MarginRadius ", 13)) {
      sscanf(line, "%s %lf", dummy, &P->MarginRad);
      continue;
    }
    if (!strncmp(line, "MarginRadial ", 13)) {
      sscanf(line, "%s %lf", dummy, &P->MarginRadial);
      continue;
    }
    if (!strncmp(line, "EtaBinSize ", 11)) {
      sscanf(line, "%s %lf", dummy, &P->EtaBinSize);
      continue;
    }
    if (!strncmp(line, "OmeBinSize ", 11)) {
      sscanf(line, "%s %lf", dummy, &P->OmeBinSize);
      continue;
    }
    if (!strncmp(line, "MinMatchesToAcceptFrac ", 22) ||
        !strncmp(line, "Completeness ", 13)) {
      sscanf(line, "%s %lf", dummy, &P->MinMatchesToAcceptFrac);
      continue;
    }
    if (!strncmp(line, "ExcludePoleAngle ", 17) ||
        !strncmp(line, "MinEta ", 7)) {
      sscanf(line, "%s %lf", dummy, &P->ExcludePoleAngle);
      continue;
    }
    if (!strncmp(line, "RingRadii ", 10)) {
      sscanf(line, "%s %lf", dummy, &P->RingRadiiUser[P->NrOfRings]);
      P->NrOfRings++;
      continue;
    }
    if (!strncmp(line, "OmegaRange ", 11)) {
      sscanf(line, "%s %lf %lf", dummy, &P->OmegaRanges[P->NoOfOmegaRanges][0],
             &P->OmegaRanges[P->NoOfOmegaRanges][1]);
      P->NoOfOmegaRanges++;
      continue;
    }
    if (!strncmp(line, "BoxSize ", 8)) {
      sscanf(line, "%s %lf %lf %lf %lf", dummy, &P->BoxSizes[NrOfBoxSizes][0],
             &P->BoxSizes[NrOfBoxSizes][1], &P->BoxSizes[NrOfBoxSizes][2],
             &P->BoxSizes[NrOfBoxSizes][3]);
      NrOfBoxSizes++;
      continue;
    }
    if (!strncmp(line, "SpotsFileName ", 14)) {
      sscanf(line, "%s %s", dummy, P->SpotsFileName);
      continue;
    }
    if (!strncmp(line, "GrainsFile ", 11)) {
      P->isGrainsInput = 1;
      sscanf(line, "%s %s", dummy, P->GrainsFileName);
      continue;
    }
    if (!strncmp(line, "IDsFileName ", 12)) {
      sscanf(line, "%s %s", dummy, P->IDsFileName);
      continue;
    }
    if (!strncmp(line, "MarginEta ", 10)) {
      sscanf(line, "%s %lf", dummy, &P->MarginEta);
      continue;
    }
    if (!strncmp(line, "UseFriedelPairs ", 16)) {
      sscanf(line, "%s %d", dummy, &P->UseFriedelPairs);
      continue;
    }
    if (!strncmp(line, "OutputFolder ", 13)) {
      sscanf(line, "%s %s", dummy, P->OutputFolder);
      continue;
    }
  }
  fclose(fp);
  if (totNrPixelsBigDetector)
    ReadBigDet(dirname(P->OutputFolder));
  for (int i = 0; i < MAX_N_RINGS; i++)
    P->RingRadii[i] = 0;
  for (int i = 0; i < P->NrOfRings; i++)
    P->RingRadii[P->RingNumbers[i]] = P->RingRadiiUser[i];
  return 0;
}

static int ReadSpots(char *cwd) {
  char fn[2048];
  sprintf(fn, "%s/Spots.bin", cwd);
  int fd = open(fn, O_RDONLY);
  check(fd < 0, "open %s: %s", fn, strerror(errno));
  struct stat s;
  fstat(fd, &s);
  ObsSpotsLab_d = (double *)mmap(0, s.st_size, PROT_READ, MAP_SHARED, fd, 0);
  check(ObsSpotsLab_d == MAP_FAILED, "mmap %s: %s", fn, strerror(errno));
  int nsp = (int)(s.st_size / (N_COL_OBSSPOTS * sizeof(double)));
  ObsSpotsLab = ObsSpotsLab_d; // Both are double now, no conversion needed
  printf("Spots.bin: %d spots (double precision)\n", nsp);
  return nsp;
}

static void ReadBins(char *cwd) {
  char fn1[2048];
  sprintf(fn1, "%s/Data.bin", cwd);
  int fd1 = open(fn1, O_RDONLY);
  check(fd1 < 0, "open %s: %s", fn1, strerror(errno));
  struct stat s1;
  fstat(fd1, &s1);
  data = (int *)mmap(0, s1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
  check(data == MAP_FAILED, "mmap %s", fn1);

  char fn2[2048];
  sprintf(fn2, "%s/nData.bin", cwd);
  int fd2 = open(fn2, O_RDONLY);
  check(fd2 < 0, "open %s: %s", fn2, strerror(errno));
  struct stat s2;
  fstat(fd2, &s2);
  ndata = (int *)mmap(0, s2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
  check(ndata == MAP_FAILED, "mmap %s", fn2);
  printf("Data.bin read. nData.bin read.\n");
}

// ─── Generate ideal spot positions ──────────────────────────

// Forward declaration (needed by h_GenerateIdealSpotsFriedelMixed)
static void h_GenerateIdealSpots(double ys, double zs, double ttheta,
                                 double eta, double Ring_rad, double Rsample,
                                 double Hbeam, double step_size, double y0v[],
                                 double z0v[], int *nSteps);

static inline void h_CalcSpotPosition(double RingRadius, double eta, double *yl,
                                      double *zl) {
  double etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

static inline void h_RotateAroundZ(double v1[3], double alpha, double v2[3]) {
  double cosa = cos(alpha * deg2rad);
  double sina = sin(alpha * deg2rad);
  v2[0] = cosa * v1[0] - sina * v1[1];
  v2[1] = sina * v1[0] + cosa * v1[1];
  v2[2] = v1[2];
}

static void h_CalcOmega(double x, double y, double z, double theta,
                        double omegas[4], double etas[4], int *nsol) {
  *nsol = 0;
  double len = sqrt(x * x + y * y + z * z);
  double v = sin(theta * deg2rad) * len;
  double almostzero = 1e-12;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      double cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        double ome = acos(cosome1) * rad2deg;
        omegas[(*nsol)++] = ome;
        omegas[(*nsol)++] = -ome;
      }
    }
  } else {
    double y2 = y * y;
    double a = 1 + (x * x) / y2;
    double b = (2 * v * x) / y2;
    double c = (v * v) / y2 - 1;
    double discr = b * b - 4 * a * c;
    if (discr >= 0) {
      double cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1.0) {
        double ome1a = acos(cosome1);
        double ome1b = -ome1a;
        double eqa = -x * cos(ome1a) + y * sin(ome1a);
        double eqb = -x * cos(ome1b) + y * sin(ome1b);
        omegas[(*nsol)++] =
            (fabs(eqa - v) < fabs(eqb - v) ? ome1a : ome1b) * rad2deg;
      }
      double cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1.0) {
        double ome2a = acos(cosome2);
        double ome2b = -ome2a;
        double eqa = -x * cos(ome2a) + y * sin(ome2a);
        double eqb = -x * cos(ome2b) + y * sin(ome2b);
        omegas[(*nsol)++] =
            (fabs(eqa - v) < fabs(eqb - v) ? ome2a : ome2b) * rad2deg;
      }
    }
  }
  double gw[3], gv[3] = {x, y, z};
  for (int i = 0; i < *nsol; i++) {
    h_RotateAroundZ(gv, omegas[i], gw);
    h_CalcEtaAngle(gw[1], gw[2], &etas[i]);
  }
}

static inline void h_displacement_spot_needed_COM(double a, double b, double c,
                                                  double xi, double yi,
                                                  double zi, double omega,
                                                  double *Displ_y,
                                                  double *Displ_z) {
  double lenInv = 1.0 / sqrt(xi * xi + yi * yi + zi * zi);
  xi *= lenInv;
  yi *= lenInv;
  zi *= lenInv;
  double OmegaRad = deg2rad * omega;
  double sinOme = sin(OmegaRad), cosOme = cos(OmegaRad);
  double t = (a * cosOme - b * sinOme) / xi;
  *Displ_y = (a * sinOme + b * cosOme) - t * yi;
  *Displ_z = c - t * zi;
}

static void h_FriedelEtaCalculation(double ys, double zs, double ttheta,
                                    double eta, double Ring_rad, double Rsample,
                                    double Hbeam, double *EtaMinFr,
                                    double *EtaMaxFr) {
  double quadr_coeff2 = 0;
  double eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0 = 0,
                                 y0_min_z0 = 0;
  double y0_max = 0, y0_min = 0, z0_min = 0, z0_max = 0;
  if (eta > 90)
    eta_Hbeam = 180 - eta;
  else if (eta < -90)
    eta_Hbeam = 180 - fabs(eta);
  else
    eta_Hbeam = 90 - fabs(eta);
  Hbeam =
      Hbeam + 2 * (Rsample * tanf(ttheta * deg2rad)) * sin(eta_Hbeam * deg2rad);
  double eta_pole = 1 + rad2deg * acos(1 - Hbeam / Ring_rad);
  double eta_equator = 1 + rad2deg * acos(1 - Rsample / Ring_rad);
  if (eta >= eta_pole && eta <= (90 - eta_equator)) {
    quadr_coeff = 1;
    coeff_y0 = -1;
    coeff_z0 = 1;
  } else if (eta >= (90 + eta_equator) && eta <= (180 - eta_pole)) {
    quadr_coeff = 2;
    coeff_y0 = -1;
    coeff_z0 = -1;
  } else if (eta >= (-90 + eta_equator) && eta <= -eta_pole) {
    quadr_coeff = 2;
    coeff_y0 = 1;
    coeff_z0 = 1;
  } else if (eta >= (-180 + eta_pole) && eta <= (-90 - eta_equator)) {
    quadr_coeff = 1;
    coeff_y0 = 1;
    coeff_z0 = -1;
  } else
    quadr_coeff = 0;
  double y0_max_R = ys + Rsample, y0_min_R = ys - Rsample;
  double z0_max_H = zs + 0.5 * Hbeam, z0_min_H = zs - 0.5 * Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_max_H * z0_max_H);
    y0_min_z0 = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_min_H * z0_min_H);
  } else if (quadr_coeff == 2) {
    y0_max_z0 = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_min_H * z0_min_H);
    y0_min_z0 = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_max_H * z0_max_H);
  }
  if (quadr_coeff > 0) {
    y0_max = fmin(y0_max_R, y0_max_z0);
    y0_min = fmax(y0_min_R, y0_min_z0);
  } else {
    if (eta > -eta_pole && eta < eta_pole) {
      y0_max = y0_max_R;
      y0_min = y0_min_R;
      coeff_z0 = 1;
    } else if (eta < (-180 + eta_pole)) {
      y0_max = y0_max_R;
      y0_min = y0_min_R;
      coeff_z0 = -1;
    } else if (eta > (180 - eta_pole)) {
      y0_max = y0_max_R;
      y0_min = y0_min_R;
      coeff_z0 = -1;
    } else if (eta > (90 - eta_equator) && eta < (90 + eta_equator)) {
      quadr_coeff2 = 1;
      z0_max = z0_max_H;
      z0_min = z0_min_H;
      coeff_y0 = -1;
    } else if (eta > (-90 - eta_equator) && eta < (-90 + eta_equator)) {
      quadr_coeff2 = 1;
      z0_max = z0_max_H;
      z0_min = z0_min_H;
      coeff_y0 = 1;
    }
  }
  if (quadr_coeff2 == 0) {
    z0_min = coeff_z0 * sqrt(Ring_rad * Ring_rad - y0_min * y0_min);
    z0_max = coeff_z0 * sqrt(Ring_rad * Ring_rad - y0_max * y0_max);
  } else {
    y0_min = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_min * z0_min);
    y0_max = coeff_y0 * sqrt(Ring_rad * Ring_rad - z0_max * z0_max);
  }
  double dYMin = ys - y0_min, dYMax = ys - y0_max, dZMin = zs - z0_min,
         dZMax = zs - z0_max;
  double YMinFr = y0_min - dYMin, YMaxFr = y0_max - dYMax;
  double ZMinFr = -z0_min + dZMin, ZMaxFr = -z0_max + dZMax;
  double Eta1, Eta2;
  h_CalcEtaAngle(YMinFr + ys, ZMinFr - zs, &Eta1);
  h_CalcEtaAngle(YMaxFr + ys, ZMaxFr - zs, &Eta2);
  *EtaMinFr = fmin(Eta1, Eta2);
  *EtaMaxFr = fmax(Eta1, Eta2);
}

static void h_GenerateIdealSpotsFriedel(double ys, double zs, double ttheta,
                                        double eta, double omega, int ringno,
                                        double Ring_rad, double Rsample,
                                        double Hbeam, double OmeTol,
                                        double RadiusTol, double y0v[],
                                        double z0v[], int *nSteps) {
  *nSteps = 0;
  double OmeF = (omega < 0) ? omega + 180 : omega - 180;
  double EtaF = (eta < 0) ? -180 - eta : 180 - eta;
  for (int r = 0; r < (int)n_spots; r++) {
    int rno_obs = (int)roundf(ObsSpotsLab[r * N_COL_OBSSPOTS + 5]);
    double ome_obs = ObsSpotsLab[r * N_COL_OBSSPOTS + 2];
    if (rno_obs != ringno)
      continue;
    if (fabs(ome_obs - OmeF) > OmeTol)
      continue;
    double yf = ObsSpotsLab[r * N_COL_OBSSPOTS + 0];
    double zf = ObsSpotsLab[r * N_COL_OBSSPOTS + 1];
    double EtaTransf;
    h_CalcEtaAngle(yf + ys, zf - zs, &EtaTransf);
    double radius = sqrt((yf + ys) * (yf + ys) + (zf - zs) * (zf - zs));
    if (fabs(radius - 2 * Ring_rad) > RadiusTol)
      continue;
    double EtaMinF, EtaMaxF;
    h_FriedelEtaCalculation(ys, zs, ttheta, eta, Ring_rad, Rsample, Hbeam,
                            &EtaMinF, &EtaMaxF);
    if (EtaTransf < EtaMinF || EtaTransf > EtaMaxF)
      continue;
    double ZPosAccZ = zs - (zf + zs) / 2;
    double YPosAccY = ys - (-yf + ys) / 2;
    double etaIdealF;
    h_CalcEtaAngle(YPosAccY, ZPosAccZ, &etaIdealF);
    double IdealYPos, IdealZPos;
    h_CalcSpotPosition(Ring_rad, etaIdealF, &IdealYPos, &IdealZPos);
    y0v[*nSteps] = IdealYPos;
    z0v[*nSteps] = IdealZPos;
    (*nSteps)++;
  }
}

static int h_AddUnique(int *arr, int *n, int val) {
  for (int i = 0; i < *n; i++)
    if (arr[i] == val)
      return 0;
  arr[*n] = val;
  (*n)++;
  return 1;
}

static void h_GenerateIdealSpotsFriedelMixed(
    double ys, double zs, double Ttheta, double Eta, double Omega, int RingNr,
    double Ring_rad, double Lsd, double Rsample, double Hbeam,
    double StepSizePos, double OmeTol, double RadialTol, double EtaBinSz,
    double OmeBinSz, double EtaTol, double spots_y[], double spots_z[],
    int *nSteps) {
  const int MinEtaReject = 10;
  double theta = Ttheta / 2;
  double SinMinEtaReject = sin(MinEtaReject * deg2rad);
  *nSteps = 0;
  if (fabs(sin(Eta * deg2rad)) < SinMinEtaReject)
    return;

  double y0_vector[2000], z0_vector[2000];
  int NoOfSpots;
  h_GenerateIdealSpots(ys, zs, Ttheta, Eta, Ring_rad, Rsample, Hbeam,
                       StepSizePos, y0_vector, z0_vector, &NoOfSpots);

  double FPCandidates[2000][3];
  int FPCandidatesUnique[2000];
  int nFPCandidates = 0;
  double EtaTolDeg = rad2deg * atan(EtaTol / Ring_rad);

  for (int SpOnRing = 0; SpOnRing < NoOfSpots; SpOnRing++) {
    double y0 = y0_vector[SpOnRing], z0 = z0_vector[SpOnRing];
    double xi, yi, zi;
    h_MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi);
    double G1, G2, G3;
    h_spot_to_gv(xi, yi, zi, Omega, &G1, &G2, &G3);
    double omegasFP[4], etasFP[4];
    int nsol;
    h_CalcOmega(-G1, -G2, -G3, theta, omegasFP, etasFP, &nsol);
    if (nsol <= 1)
      continue;
    double diff0 = fabs(omegasFP[0] - Omega);
    if (diff0 > 180)
      diff0 = 360 - diff0;
    double diff1 = fabs(omegasFP[1] - Omega);
    if (diff1 > 180)
      diff1 = 360 - diff1;
    double OmegaFP, EtaFP;
    if (diff0 < diff1) {
      OmegaFP = omegasFP[0];
      EtaFP = etasFP[0];
    } else {
      OmegaFP = omegasFP[1];
      EtaFP = etasFP[1];
    }
    double YFP1, ZFP1;
    h_CalcSpotPosition(Ring_rad, EtaFP, &YFP1, &ZFP1);
    int nMax, nMin;
    h_calc_n_max_min(xi, yi, ys, y0, Rsample, (int)StepSizePos, &nMax, &nMin);
    for (int n = nMin; n <= nMax; n++) {
      double a, b, c;
      h_spot_to_unrotated(xi, yi, zi, ys, zs, y0, z0, StepSizePos, n, Omega, &a,
                          &b, &c);
      if (fabs(c) > Hbeam / 2)
        continue;
      double Dy, Dz;
      h_displacement_spot_needed_COM(a, b, c, Lsd, YFP1, ZFP1, OmegaFP, &Dy,
                                     &Dz);
      double YFP = YFP1 + Dy, ZFP = ZFP1 + Dz;
      double RadialPosFP = sqrt(YFP * YFP + ZFP * ZFP) - Ring_rad;
      double EtaFPCorr;
      h_CalcEtaAngle(YFP, ZFP, &EtaFPCorr);
      // Bin lookup (inline GetBin)
      int iRing = RingNr - 1;
      int iEta = (int)floor((180.0 + EtaFPCorr) / EtaBinSz);
      int iOme = (int)floor((180.0 + OmegaFP) / OmeBinSz);
      if (iEta < 0)
        iEta = 0;
      if (iEta >= n_eta_bins)
        iEta = n_eta_bins - 1;
      if (iOme < 0)
        iOme = 0;
      if (iOme >= n_ome_bins)
        iOme = n_ome_bins - 1;
      size_t Pos = (size_t)iRing * n_eta_bins * n_ome_bins +
                   (size_t)iEta * n_ome_bins + iOme;
      int nInBin = ndata[(int)(Pos * 2 + 0)],
          DataPos_ = ndata[(int)(Pos * 2 + 1)];
      for (int iSpot = 0; iSpot < nInBin; iSpot++) {
        int spotRow = data[DataPos_ + iSpot];
        int base = spotRow * N_COL_OBSSPOTS;
        if (fabs(RadialPosFP - ObsSpotsLab[base + 8]) >= RadialTol)
          continue;
        if (fabs(OmegaFP - ObsSpotsLab[base + 2]) >= OmeTol)
          continue;
        if (fabs(EtaFPCorr - ObsSpotsLab[base + 6]) >= EtaTolDeg)
          continue;
        double dy = YFP - ObsSpotsLab[base + 0],
               dz = ZFP - ObsSpotsLab[base + 1];
        double diffPos2 = dy * dy + dz * dz;
        int idx = nFPCandidates;
        for (int i = 0; i < nFPCandidates; i++) {
          if (FPCandidates[i][0] == ObsSpotsLab[base + 4]) {
            idx = (diffPos2 < FPCandidates[i][2]) ? i : -1;
            break;
          }
        }
        if (idx >= 0) {
          FPCandidates[idx][0] = ObsSpotsLab[base + 4];
          FPCandidates[idx][1] = (double)SpOnRing;
          FPCandidates[idx][2] = diffPos2;
          if (idx == nFPCandidates)
            nFPCandidates++;
        }
      }
    }
  }
  int nFPCandidatesUniq = 0;
  for (int i = 0; i < nFPCandidates; i++)
    h_AddUnique(FPCandidatesUnique, &nFPCandidatesUniq,
                (int)FPCandidates[i][1]);
  for (int i = 0; i < nFPCandidatesUniq; i++) {
    spots_y[i] = y0_vector[FPCandidatesUnique[i]];
    spots_z[i] = z0_vector[FPCandidatesUnique[i]];
  }
  *nSteps = nFPCandidatesUniq;
}

static void h_GenerateIdealSpots(double ys, double zs, double ttheta,
                                 double eta, double Ring_rad, double Rsample,
                                 double Hbeam, double step_size, double y0v[],
                                 double z0v[], int *nSteps) {
  // Simplified version: generate y0 along the illuminated arc
  int qc2 = 0;
  double eh, qc, cy = 0, cz = 0, ymax_z0, ymin_z0, ymax = 0, ymin = 0, zmin = 0,
                 zmax = 0;
  if (eta > 90)
    eh = 180 - eta;
  else if (eta < -90)
    eh = 180 - fabs(eta);
  else
    eh = 90 - fabs(eta);
  Hbeam += 2 * (Rsample * tan(ttheta * deg2rad)) * sin(eh * deg2rad);
  double epole = 1 + rad2deg * acos(1 - Hbeam / Ring_rad);
  double eeq = 1 + rad2deg * acos(1 - Rsample / Ring_rad);
  if (eta >= epole && eta <= (90 - eeq)) {
    qc = 1;
    cy = -1;
    cz = 1;
  } else if (eta >= (90 + eeq) && eta <= (180 - epole)) {
    qc = 2;
    cy = -1;
    cz = -1;
  } else if (eta >= (-90 + eeq) && eta <= -epole) {
    qc = 2;
    cy = 1;
    cz = 1;
  } else if (eta >= (-180 + epole) && eta <= (-90 - eeq)) {
    qc = 1;
    cy = 1;
    cz = -1;
  } else
    qc = 0;
  double ymaxR = ys + Rsample, yminR = ys - Rsample;
  double zmaxH = zs + 0.5 * Hbeam, zminH = zs - 0.5 * Hbeam;
  if (qc == 1) {
    ymax_z0 = cy * sqrt(Ring_rad * Ring_rad - zmaxH * zmaxH);
    ymin_z0 = cy * sqrt(Ring_rad * Ring_rad - zminH * zminH);
  } else if (qc == 2) {
    ymax_z0 = cy * sqrt(Ring_rad * Ring_rad - zminH * zminH);
    ymin_z0 = cy * sqrt(Ring_rad * Ring_rad - zmaxH * zmaxH);
  }
  if (qc > 0) {
    ymax = fmin(ymaxR, ymax_z0);
    ymin = fmax(yminR, ymin_z0);
  } else {
    if (eta > -epole && eta < epole) {
      ymax = ymaxR;
      ymin = yminR;
      cz = 1;
    } else if (eta < (-180 + epole)) {
      ymax = ymaxR;
      ymin = yminR;
      cz = -1;
    } else if (eta > (180 - epole)) {
      ymax = ymaxR;
      ymin = yminR;
      cz = -1;
    } else if (eta > (90 - eeq) && eta < (90 + eeq)) {
      qc2 = 1;
      zmax = zmaxH;
      zmin = zminH;
      cy = -1;
    } else if (eta > (-90 - eeq) && eta < (-90 + eeq)) {
      qc2 = 1;
      zmax = zmaxH;
      zmin = zminH;
      cy = 1;
    }
  }
  double y1, z1, y2, z2;
  int ns;
  if (!qc2) {
    y1 = ymin;
    z1 = cz * sqrt(Ring_rad * Ring_rad - y1 * y1);
    y2 = ymax;
    z2 = cz * sqrt(Ring_rad * Ring_rad - y2 * y2);
  } else {
    z1 = zmin;
    y1 = cy * sqrt(Ring_rad * Ring_rad - z1 * z1);
    z2 = zmax;
    y2 = cy * sqrt(Ring_rad * Ring_rad - z2 * z2);
  }
  double yd = y1 - y2, zd = z1 - z2;
  double len = sqrt(yd * yd + zd * zd);
  ns = (int)ceil(len / step_size);
  if (ns % 2 == 0)
    ns++;
  if (ns < 1)
    ns = 1;
  if (ns == 1) {
    if (!qc2) {
      y0v[0] = (ymax + ymin) / 2;
      z0v[0] = cz * sqrt(Ring_rad * Ring_rad - y0v[0] * y0v[0]);
    } else {
      z0v[0] = (zmax + zmin) / 2;
      y0v[0] = cy * sqrt(Ring_rad * Ring_rad - z0v[0] * z0v[0]);
    }
  } else {
    double sy = (ymax - ymin) / (ns - 1), sz = (zmax - zmin) / (ns - 1);
    for (int i = 0; i < ns; i++) {
      if (!qc2) {
        y0v[i] = ymin + i * sy;
        z0v[i] = cz * sqrt(Ring_rad * Ring_rad - y0v[i] * y0v[i]);
      } else {
        z0v[i] = zmin + i * sz;
        y0v[i] = cy * sqrt(Ring_rad * Ring_rad - z0v[i] * z0v[i]);
      }
    }
  }
  *nSteps = ns;
}

// ═════════════════════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tIndexerGPU (%s)\n"
         "GPU-accelerated FF-HEDM Indexer.\n"
         "Contact hsharma@anl.gov for questions.\n\n",
         MIDAS_VERSION_STRING);

  if (argc != 6) {
    printf("Usage: %s param.txt blockNr nBlocks nSpotsToIndex numProcs\n",
           argv[0]);
    exit(EXIT_FAILURE);
  }

  // 1. Read parameters
  struct TParams Params;
  if (ReadParams(argv[1], &Params)) {
    printf("Error reading params\n");
    exit(1);
  }
  printf("SpaceGroup: %d, Distance: %f, NrOfRings: %d\n", Params.SpaceGroupNum,
         Params.Distance, Params.NrOfRings);

  // 2. Read HKLs
  FILE *hklf = fopen("hkls.csv", "r");
  if (!hklf) {
    printf("Cannot open hkls.csv\n");
    exit(1);
  }
  char aline[4096];
  (void)fgets(aline, 1000, hklf);
  int hi, ki, li, Rnr;
  double hc, kc, lc, RRd, Ds, tht, tth;
  while (fgets(aline, 1000, hklf)) {
    sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %lf %lf", &hi, &ki, &li, &Ds,
           &Rnr, &hc, &kc, &lc, &tht, &tth, &RRd);
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
        n_hkls++;
        break;
      }
    }
  }
  fclose(hklf);
  printf("HKLs: %d\n", n_hkls);

  // 3. Read observed spots and bins
  char tmpstr[4096];
  snprintf(tmpstr, sizeof(tmpstr), "%s", Params.OutputFolder);
  char *cwdstr = dirname(tmpstr);
  n_spots = ReadSpots(cwdstr);
  printf("Spots: %zu\n", n_spots);
  ReadBins(cwdstr);

  int HighestRingNo = 0;
  for (int i = 0; i < MAX_N_RINGS; i++)
    if (Params.RingRadii[i])
      HighestRingNo = i;
  n_ring_bins = HighestRingNo;
  n_eta_bins = (int)ceilf(360.0 / Params.EtaBinSize);
  n_ome_bins = (int)ceilf(360.0 / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize;
  OmeBinSize = Params.OmeBinSize;
  printf("Bins: ring=%d eta=%d ome=%d\n", n_ring_bins, n_eta_bins, n_ome_bins);

  // 4. Precompute margins
  double omemargins[181], etamargins[MAX_N_RINGS];
  for (int i = 1; i < 180; i++)
    omemargins[i] = Params.MarginOme +
                    (0.5 * Params.StepsizeOrient / fabs(sin(i * deg2rad)));
  omemargins[0] = omemargins[1];
  omemargins[180] = omemargins[1];
  for (int i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] == 0)
      etamargins[i] = 0;
    else
      etamargins[i] = rad2deg * atanf(Params.MarginEta / Params.RingRadii[i]) +
                      0.5 * Params.StepsizeOrient;
  }

  // 5. Open output files
  int blockNr = atoi(argv[2]);
  int nSpotsToIndex = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  int openFlags = O_CREAT | O_WRONLY | (blockNr == 0 ? O_TRUNC : 0);
  char outFN[4096];
  sprintf(outFN, "%s/IndexBest.bin", Params.OutputFolder);
  Params.IndexBestFD = open(outFN, openFlags, S_IRUSR | S_IWUSR);
  sprintf(outFN, "%s/IndexBestFull.bin", Params.OutputFolder);
  Params.IndexBestFullFD = open(outFN, openFlags, S_IRUSR | S_IWUSR);
  if (blockNr == 0) {
    (void)ftruncate(Params.IndexBestFD, (off_t)nSpotsToIndex * 15 * sizeof(double));
    (void)ftruncate(Params.IndexBestFullFD,
              (off_t)nSpotsToIndex * MAX_N_HKLS * 2 * sizeof(double));
  }

  // 6. Read spot IDs
  int nBlocks = atoi(argv[3]);
  int startRow = (int)(ceilf((double)nSpotsToIndex / nBlocks)) * blockNr;
  int tmp = (int)(ceilf((double)nSpotsToIndex / nBlocks)) * (blockNr + 1);
  int endRow = tmp < (nSpotsToIndex - 1) ? tmp : (nSpotsToIndex - 1);
  int nSpotIDs = endRow - startRow + 1;
  int *SpotIDs = (int *)malloc(nSpotIDs * sizeof(int));
  for (int i = 0; i < nSpotIDs; i++)
    SpotIDs[i] = -1;

  FILE *sf = fopen("SpotsToIndex.csv", "r");
  if (!sf) {
    printf("Cannot open SpotsToIndex.csv\n");
    exit(1);
  }
  for (int i = 0; i < startRow; i++)
    (void)fgets(aline, 1000, sf);
  for (int i = 0; i < nSpotIDs; i++) {
    if (fgets(aline, 1000, sf))
      sscanf(aline, "%d", &SpotIDs[i]);
  }
  fclose(sf);
  printf("SpotIDs to index: %d\n", nSpotIDs);

  // ═══════════════════════════════════════════════════════════
  //  GPU SETUP
  // ═══════════════════════════════════════════════════════════
  int deviceId = 0;
  CUDA_CHECK(cudaSetDevice(deviceId));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  printf("GPU: %s (%.0f MB)\n", prop.name, prop.totalGlobalMem / 1048576.0);

  // Upload constant memory (convert double→float)
  {
    RealType f_RingRadii[MAX_N_RINGS];
    for (int i = 0; i < MAX_N_RINGS; i++)
      f_RingRadii[i] = (RealType)Params.RingRadii[i];
    CUDA_CHECK(
        cudaMemcpyToSymbol(c_RingRadii, f_RingRadii, sizeof(f_RingRadii)));

    RealType f_OmegaRanges[MAX_N_OMEGARANGES][2];
    for (int i = 0; i < MAX_N_OMEGARANGES; i++) {
      f_OmegaRanges[i][0] = (RealType)Params.OmegaRanges[i][0];
      f_OmegaRanges[i][1] = (RealType)Params.OmegaRanges[i][1];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_OmegaRanges, f_OmegaRanges,
                                  sizeof(f_OmegaRanges)));

    RealType f_BoxSizes[MAX_N_OMEGARANGES][4];
    for (int i = 0; i < MAX_N_OMEGARANGES; i++)
      for (int j = 0; j < 4; j++)
        f_BoxSizes[i][j] = (RealType)Params.BoxSizes[i][j];
    CUDA_CHECK(cudaMemcpyToSymbol(c_BoxSizes, f_BoxSizes, sizeof(f_BoxSizes)));

    RealType f_omemargins[181];
    for (int i = 0; i < 181; i++)
      f_omemargins[i] = (RealType)omemargins[i];
    CUDA_CHECK(
        cudaMemcpyToSymbol(c_omemargins, f_omemargins, sizeof(f_omemargins)));

    RealType f_etamargins[MAX_N_RINGS];
    for (int i = 0; i < MAX_N_RINGS; i++)
      f_etamargins[i] = (RealType)etamargins[i];
    CUDA_CHECK(
        cudaMemcpyToSymbol(c_etamargins, f_etamargins, sizeof(f_etamargins)));
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_ringsToReject, Params.RingsToReject,
                                sizeof(Params.RingsToReject)));

  GPUParams gp;
  gp.Distance = (RealType)Params.Distance;
  gp.Wavelength = (RealType)Params.Wavelength;
  gp.ExcludePoleAngle = (RealType)Params.ExcludePoleAngle;
  gp.MarginRad = (RealType)Params.MarginRad;
  gp.MarginRadial = (RealType)Params.MarginRadial;
  gp.MarginOme = (RealType)Params.MarginOme;
  gp.NoOfOmegaRanges = Params.NoOfOmegaRanges;
  gp.nRingsToRejectCalc = Params.nRingsToRejectCalc;
  gp.n_ring_bins = n_ring_bins;
  gp.n_eta_bins = n_eta_bins;
  gp.n_ome_bins = n_ome_bins;
  gp.EtaBinSize = (RealType)EtaBinSize;
  gp.OmeBinSize = (RealType)OmeBinSize;
  CUDA_CHECK(cudaMemcpyToSymbol(c_params, &gp, sizeof(GPUParams)));

  // Helper: convert double array to float
  auto d2f = [](const double *src, size_t n) -> RealType * {
    RealType *dst = (RealType *)malloc(n * sizeof(RealType));
    for (size_t i = 0; i < n; i++)
      dst[i] = (RealType)src[i];
    return dst;
  };

  // Upload persistent data to GPU
  RealType *d_hkls_flat;
  size_t nHkls7 = (size_t)n_hkls * 7;
  CUDA_CHECK(cudaMalloc(&d_hkls_flat, nHkls7 * sizeof(RealType)));
  CUDA_CHECK(cudaMemcpy(d_hkls_flat, hkls, nHkls7 * sizeof(RealType),
                        cudaMemcpyHostToDevice));

  RealType *d_ObsSpotsLab;
  size_t nObsElems = (size_t)n_spots * N_COL_OBSSPOTS;
  RealType *f_obs = d2f(ObsSpotsLab, nObsElems);
  CUDA_CHECK(cudaMalloc(&d_ObsSpotsLab, nObsElems * sizeof(RealType)));
  CUDA_CHECK(cudaMemcpy(d_ObsSpotsLab, f_obs, nObsElems * sizeof(RealType),
                        cudaMemcpyHostToDevice));
  free(f_obs);

  int *d_data, *d_ndata;
  // Determine data sizes from file sizes
  {
    char fn1[2048];
    sprintf(fn1, "%s/Data.bin", cwdstr);
    struct stat s1;
    stat(fn1, &s1);
    CUDA_CHECK(cudaMalloc(&d_data, s1.st_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, s1.st_size, cudaMemcpyHostToDevice));
    char fn2[2048];
    sprintf(fn2, "%s/nData.bin", cwdstr);
    struct stat s2;
    stat(fn2, &s2);
    CUDA_CHECK(cudaMalloc(&d_ndata, s2.st_size));
    CUDA_CHECK(cudaMemcpy(d_ndata, ndata, s2.st_size, cudaMemcpyHostToDevice));
  }

  int *d_ringsToReject;
  CUDA_CHECK(cudaMalloc(&d_ringsToReject, MAX_N_RINGS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ringsToReject, Params.RingsToReject,
                        MAX_N_RINGS * sizeof(int), cudaMemcpyHostToDevice));

  // Results array (one per spotID)
  SpotResult *d_results;
  CUDA_CHECK(cudaMalloc(&d_results, nSpotIDs * sizeof(SpotResult)));

  SpotResult *h_results = (SpotResult *)calloc(nSpotIDs, sizeof(SpotResult));
  for (int i = 0; i < nSpotIDs; i++) {
    h_results[i].atomicKey = 0ULL;
    h_results[i].bestFrac = -1.0;
    h_results[i].bestIA = 999.0;
  }
  CUDA_CHECK(cudaMemcpy(d_results, h_results, nSpotIDs * sizeof(SpotResult),
                        cudaMemcpyHostToDevice));

  printf("GPU data uploaded. Starting tuple generation...\n");

  // ═══════════════════════════════════════════════════════════
  //  CPU: Generate all evaluation tuples (OMP parallel)
  // ═══════════════════════════════════════════════════════════
  double tuple_start = omp_get_wtime();

  // Phase 1: Count tuples per spotID (parallel)
  int *tupleCounts = (int *)calloc(nSpotIDs, sizeof(int));

  // Pre-allocate per-thread OrTmp buffers (MAX_N_OR*9 floats = 1.3 MB each)
  double **OrTmp_all = (double **)malloc(numProcs * sizeof(double *));
  for (int t = 0; t < numProcs; t++)
    OrTmp_all[t] = (double *)malloc(MAX_N_OR * 9 * sizeof(double));

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int si = 0; si < nSpotIDs; si++) {
    if (SpotIDs[si] == -1)
      continue;
    int spotID = SpotIDs[si];
    // Find spot row
    int SpotRowNo = -1;
    for (int r = 0; r < (int)n_spots; r++) {
      if ((int)ObsSpotsLab[r * N_COL_OBSSPOTS + 4] == spotID) {
        SpotRowNo = r;
        break;
      }
    }
    if (SpotRowNo < 0)
      continue;

    double ys = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 0];
    double zs = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 1];
    double omega = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 2];
    double eta = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 6];
    int ringnr = (int)ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 5];

    // Generate plane normals — match CPU's Friedel pair logic
    double y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
    int nPN = 0;
    int usingFriedelPair = 0;
    if (Params.UseFriedelPairs == 1) {
      usingFriedelPair = 1;
      h_GenerateIdealSpotsFriedel(
          ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
          Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam,
          Params.MarginOme, Params.MarginRadial, y0v, z0v, &nPN);
      if (nPN == 0) {
        h_GenerateIdealSpotsFriedelMixed(
            ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
            Params.RingRadii[ringnr], Params.Distance, Params.Rsample,
            Params.Hbeam, Params.StepsizePos, Params.MarginOme,
            Params.MarginRadial, Params.EtaBinSize, Params.OmeBinSize,
            Params.MarginEta, y0v, z0v, &nPN);
      }
    }
    if (nPN == 0) {
      if (usingFriedelPair == 1)
        continue; // CPU skips spots with no Friedel match
      h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                           Params.RingRadii[ringnr], Params.Rsample,
                           Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);
    }

    int count = 0;
    for (int isp = 0; isp < nPN; isp++) {
      double xi, yi, zi;
      h_MakeUnitLength(Params.Distance, y0v[isp], z0v[isp], &xi, &yi, &zi);
      double g1, g2, g3;
      h_spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
      double hn[3] = {g1, g2, g3};
      double hkl_d[3] = {RingHKL[ringnr][0], RingHKL[ringnr][1],
                         RingHKL[ringnr][2]};
      int nOr;
      double *OrTmp = OrTmp_all[omp_get_thread_num()];
      h_GenerateCandidateOrientations(hkl_d, hn, Params.StepsizeOrient, OrTmp,
                                      &nOr, ringnr);

      for (int or_ = 0; or_ < nOr; or_++) {
        int nmax, nmin;
        h_calc_n_max_min(xi, yi, ys, y0v[isp], Params.Rsample,
                         Params.StepsizePos, &nmax, &nmin);
        for (int n = nmin; n <= nmax; n++) {
          double ga, gb, gc;
          h_spot_to_unrotated(xi, yi, zi, ys, zs, y0v[isp], z0v[isp],
                              Params.StepsizePos, n, omega, &ga, &gb, &gc);
          if (fabs(gc) > Params.Hbeam / 2)
            continue;
          count++;
        }
      }
    }
    tupleCounts[si] = count;
  }

  // Prefix sum for offsets
  long long totalTuples = 0;
  int *tupleOffsets = (int *)malloc(nSpotIDs * sizeof(int));
  for (int i = 0; i < nSpotIDs; i++) {
    tupleOffsets[i] = (int)totalTuples;
    totalTuples += tupleCounts[i];
  }

  printf(
      "Total evaluation tuples: %lld (%.1f avg per spot). Count time: %.2fs\n",
      totalTuples, (double)totalTuples / nSpotIDs,
      omp_get_wtime() - tuple_start);

  int nGood = 0;

  if (totalTuples == 0) {
    printf("No tuples to evaluate. Exiting.\n");
    goto cleanup;
  }

  // Phase 2: Fill tuples (parallel)
  {
    EvalTuple *h_tuples = (EvalTuple *)malloc(totalTuples * sizeof(EvalTuple));
    if (!h_tuples) {
      printf("Cannot allocate %lld tuples (%.1f GB)\n", totalTuples,
             totalTuples * sizeof(EvalTuple) / 1e9);
      exit(1);
    }

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int si = 0; si < nSpotIDs; si++) {
      if (SpotIDs[si] == -1 || tupleCounts[si] == 0)
        continue;
      int spotID = SpotIDs[si];
      int SpotRowNo = -1;
      for (int r = 0; r < (int)n_spots; r++) {
        if ((int)ObsSpotsLab[r * N_COL_OBSSPOTS + 4] == spotID) {
          SpotRowNo = r;
          break;
        }
      }
      if (SpotRowNo < 0)
        continue;

      double ys = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 0];
      double zs = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 1];
      double omega = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 2];
      double RefRad = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 3];
      double eta = ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 6];
      int ringnr = (int)ObsSpotsLab[SpotRowNo * N_COL_OBSSPOTS + 5];

      double y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
      int nPN = 0;
      int usingFriedelPair = 0;
      if (Params.UseFriedelPairs == 1) {
        usingFriedelPair = 1;
        h_GenerateIdealSpotsFriedel(
            ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
            Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam,
            Params.MarginOme, Params.MarginRadial, y0v, z0v, &nPN);
        if (nPN == 0) {
          h_GenerateIdealSpotsFriedelMixed(
              ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
              Params.RingRadii[ringnr], Params.Distance, Params.Rsample,
              Params.Hbeam, Params.StepsizePos, Params.MarginOme,
              Params.MarginRadial, Params.EtaBinSize, Params.OmeBinSize,
              Params.MarginEta, y0v, z0v, &nPN);
        }
      }
      if (nPN == 0) {
        if (usingFriedelPair == 1)
          continue;
        h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                             Params.RingRadii[ringnr], Params.Rsample,
                             Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);
      }
      int idx = tupleOffsets[si];
      for (int isp = 0; isp < nPN; isp++) {
        double xi, yi, zi;
        h_MakeUnitLength(Params.Distance, y0v[isp], z0v[isp], &xi, &yi, &zi);
        double g1, g2, g3;
        h_spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
        double hn[3] = {g1, g2, g3};
        double hkl_d[3] = {RingHKL[ringnr][0], RingHKL[ringnr][1],
                           RingHKL[ringnr][2]};
        int nOr;
        double *OrTmp = OrTmp_all[omp_get_thread_num()];
        h_GenerateCandidateOrientations(hkl_d, hn, Params.StepsizeOrient, OrTmp,
                                        &nOr, ringnr);

        for (int or_ = 0; or_ < nOr; or_++) {
          int nmax, nmin;
          h_calc_n_max_min(xi, yi, ys, y0v[isp], Params.Rsample,
                           Params.StepsizePos, &nmax, &nmin);
          for (int n = nmin; n <= nmax; n++) {
            double ga, gb, gc;
            h_spot_to_unrotated(xi, yi, zi, ys, zs, y0v[isp], z0v[isp],
                                Params.StepsizePos, n, omega, &ga, &gb, &gc);
            if (fabs(gc) > Params.Hbeam / 2)
              continue;
            EvalTuple *t = &h_tuples[idx++];
            t->spotIdx = si;
            for (int k = 0; k < 9; k++)
              t->OrMat[k] = OrTmp[or_ * 9 + k];
            t->ga = ga;
            t->gb = gb;
            t->gc = gc;
            t->RefRad = RefRad;
          }
        }
      }
    }

    printf("Tuples filled. Uploading to GPU...\n");

    // ═════════════════════════════════════════════════════════
    //  GPU DISPATCH: Upload tuples and launch kernel
    // ═════════════════════════════════════════════════════════

    // Compute dynamic scratch size: each HKL can produce at most 2 omega
    // solutions
    int maxTheorSpots = n_hkls * 2;
    size_t scratchPerThread =
        (size_t)maxTheorSpots * N_COL_THEORSPOTS * sizeof(RealType);

    // Process in batches — account for both tuples AND scratch memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    size_t bytesPerThread = sizeof(EvalTuple) + scratchPerThread;
    size_t maxBatch = (freeMem / 2) / bytesPerThread;
    if (maxBatch < 1)
      maxBatch = 1;
    int nBatches = (int)((totalTuples + maxBatch - 1) / maxBatch);
    if (nBatches < 1)
      nBatches = 1;

    long long batchSize = (totalTuples + nBatches - 1) / nBatches;

    printf("GPU free: %.0f MB, tupleBytes: %.0f MB, scratchPerThread: %zu B, "
           "maxTheorSpots: %d, batches: %d, batchSize: %lld\n",
           freeMem / 1e6, totalTuples * sizeof(EvalTuple) / 1e6,
           scratchPerThread, maxTheorSpots, nBatches, batchSize);

    EvalTuple *d_tuples;
    CUDA_CHECK(cudaMalloc(&d_tuples, batchSize * sizeof(EvalTuple)));

    RealType *d_theorScratch;
    CUDA_CHECK(cudaMalloc(&d_theorScratch, batchSize * scratchPerThread));

    int blockSize = 256;

    for (int b = 0; b < nBatches; b++) {
      long long bStart = b * batchSize;
      long long bEnd = bStart + batchSize;
      if (bEnd > totalTuples)
        bEnd = totalTuples;
      long long bN = bEnd - bStart;

      CUDA_CHECK(cudaMemcpy(d_tuples, h_tuples + bStart, bN * sizeof(EvalTuple),
                            cudaMemcpyHostToDevice));

      int nBlks = (int)((bN + blockSize - 1) / blockSize);
      printf("  Batch %d/%d: %lld tuples, %d blocks\n", b + 1, nBatches, bN,
             nBlks);

      indexer_eval_kernel<<<nBlks, blockSize>>>(
          d_tuples, (int)bN, d_hkls_flat, n_hkls, d_ObsSpotsLab, d_data,
          d_ndata, d_ringsToReject, Params.nRingsToRejectCalc, d_results,
          d_theorScratch, maxTheorSpots);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_tuples));
    CUDA_CHECK(cudaFree(d_theorScratch));

    free(h_tuples);
  }

  // ═══════════════════════════════════════════════════════════
  //  Download results and write output
  // ═══════════════════════════════════════════════════════════
  CUDA_CHECK(cudaMemcpy(h_results, d_results, nSpotIDs * sizeof(SpotResult),
                        cudaMemcpyDeviceToHost));

  for (int si = 0; si < nSpotIDs; si++) {
    if (h_results[si].bestFrac <= 0)
      continue;
    nGood++;

    // Write IndexBest.bin (same format as IndexerOMP)
    double res[15];
    res[0] =
        (double)h_results[si]
            .bestIA; // IA — matches CPU format (confidence = res[14]/res[13])
    for (int k = 0; k < 9; k++)
      res[k + 1] = (double)h_results[si].bestOrMat[k];
    res[10] = (double)h_results[si].bestPos[0];
    res[11] = (double)h_results[si].bestPos[1];
    res[12] = (double)h_results[si].bestPos[2];
    res[13] = (double)h_results[si].nTspots;  // nExp
    res[14] = (double)h_results[si].nMatches; // nObs

    size_t offset = (size_t)(si + startRow) * 15 * sizeof(double);
    (void)pwrite(Params.IndexBestFD, res, 15 * sizeof(double), offset);

    // ── Write IndexBestFull.bin (re-match spots on CPU) ──
    // Use the best orientation to compute theoretical spots, match against
    // observed using bin structure, write interleaved [spotID, distance].
    {
      double OrM[3][3];
      for (int r2 = 0; r2 < 3; r2++)
        for (int cc = 0; cc < 3; cc++)
          OrM[r2][cc] = h_results[si].bestOrMat[r2 * 3 + cc];

      // Compute theoretical spots inline (mirrors gpu_CalcDiffrSpots)
      double theorBuf[MAX_N_HKLS * 9];
      int nTspots = 0;
      for (int ih = 0; ih < n_hkls && nTspots < MAX_N_HKLS; ih++) {
        double theta = hkls[ih][5];
        double RingRadius = hkls[ih][6];
        int RingNr = (int)hkls[ih][3];
        double gHat[3] = {hkls[ih][0] * OrM[0][0] + hkls[ih][1] * OrM[0][1] +
                              hkls[ih][2] * OrM[0][2],
                          hkls[ih][0] * OrM[1][0] + hkls[ih][1] * OrM[1][1] +
                              hkls[ih][2] * OrM[1][2],
                          hkls[ih][0] * OrM[2][0] + hkls[ih][1] * OrM[2][1] +
                              hkls[ih][2] * OrM[2][2]};
        double omegas[4], etas[4];
        int nOme;
        h_CalcOmega(gHat[0], gHat[1], gHat[2], theta, omegas, etas, &nOme);
        for (int om = 0; om < nOme; om++) {
          double ome = omegas[om];
          int inRange = 0;
          for (int or2 = 0; or2 < Params.NoOfOmegaRanges; or2++) {
            if (ome >= Params.OmegaRanges[or2][0] &&
                ome <= Params.OmegaRanges[or2][1]) {
              inRange = 1;
              break;
            }
          }
          if (!inRange)
            continue;
          double eta = etas[om];
          if (fabs(eta) < Params.ExcludePoleAngle)
            continue;
          double yl, zl;
          h_CalcSpotPosition(RingRadius, eta, &yl, &zl);
          double *ts = &theorBuf[nTspots * 9];
          ts[0] = yl;
          ts[1] = zl;
          ts[2] = ome;
          ts[7] = RingNr;
          nTspots++;
          if (nTspots >= MAX_N_HKLS)
            break;
        }
      }

      // Match theoretical spots against observed using bin structure
      double *outArr = (double *)calloc(MAX_N_HKLS * 2, sizeof(double));
      int nMatched = 0;
      for (int sp = 0; sp < nTspots && nMatched < MAX_N_HKLS; sp++) {
        double theorYl = theorBuf[sp * 9 + 0], theorZl = theorBuf[sp * 9 + 1];
        double theorOmega = theorBuf[sp * 9 + 2];
        int RingNr = (int)theorBuf[sp * 9 + 7];
        // Use h_CalcEtaAngle (same as bin construction uses)
        double theorEta;
        h_CalcEtaAngle(theorYl, theorZl, &theorEta);

        int iRing = RingNr - 1;
        if (iRing < 0 || iRing >= n_ring_bins)
          continue;
        int iEta = (int)floor((180.0 + theorEta) / EtaBinSize);
        int iOme = (int)floor((180.0 + theorOmega) / OmeBinSize);
        if (iEta < 0)
          iEta = 0;
        if (iEta >= n_eta_bins)
          iEta = n_eta_bins - 1;
        if (iOme < 0)
          iOme = 0;
        if (iOme >= n_ome_bins)
          iOme = n_ome_bins - 1;

        long long int Pos2 = (long long int)iRing * n_eta_bins * n_ome_bins +
                             iEta * n_ome_bins + iOme;
        int nInBin = ndata[Pos2 * 2];
        int DataPos = ndata[Pos2 * 2 + 1];
        if (nInBin == 0)
          continue;

        double bestDiffOme = 1e9;
        int bestRow = -1;
        for (int iSpot = 0; iSpot < nInBin; iSpot++) {
          int spotRow = data[DataPos + iSpot];
          if (spotRow < 0 || spotRow >= (int)n_spots)
            continue;
          double obsOme = ObsSpotsLab_d[spotRow * 9 + 2];
          double diffOme = fabs(theorOmega - obsOme);
          if (diffOme < 5.0 && diffOme < bestDiffOme) {
            bestDiffOme = diffOme;
            bestRow = spotRow;
          }
        }
        if (bestRow >= 0) {
          outArr[nMatched * 2 + 0] = (double)(bestRow + 1); // 1-indexed spotID
          double obsY = ObsSpotsLab_d[bestRow * 9 + 0];
          double obsZ = ObsSpotsLab_d[bestRow * 9 + 1];
          outArr[nMatched * 2 + 1] = sqrt(obsY * obsY + obsZ * obsZ);
          nMatched++;
        }
      }

      size_t offset2 =
          (size_t)(si + startRow) * MAX_N_HKLS * 2 * sizeof(double);
      (void)pwrite(Params.IndexBestFullFD, outArr, MAX_N_HKLS * 2 * sizeof(double),
             offset2);
      printf("  IndexBestFull: grain %d, nTspots=%d, nMatched=%d, offset=%zu\n",
             si, nTspots, nMatched, offset2);
      free(outArr);
    }

    printf("IDNr: %d, ID: %d, Confidence: %.4f, nExp: %d, nObs: %d\n", si,
           SpotIDs[si], h_results[si].bestFrac, h_results[si].nTspots,
           h_results[si].nMatches);
  }

  printf("\nGPU Indexer complete: %d/%d spots found matches.\n", nGood,
         nSpotIDs);

cleanup:
  // Free GPU memory
  CUDA_CHECK(cudaFree(d_hkls_flat));
  CUDA_CHECK(cudaFree(d_ObsSpotsLab));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ndata));
  CUDA_CHECK(cudaFree(d_ringsToReject));
  CUDA_CHECK(cudaFree(d_results));

  free(h_results);
  free(tupleCounts);
  free(tupleOffsets);
  free(SpotIDs);
  for (int t = 0; t < numProcs; t++)
    free(OrTmp_all[t]);
  free(OrTmp_all);
  // ObsSpotsLab points to mmap'd memory (ObsSpotsLab_d), do not free

  close(Params.IndexBestFD);
  close(Params.IndexBestFullFD);

  double elapsed = omp_get_wtime() - start_time;
  printf("Total time: %.2f seconds.\n", elapsed);
  return 0;
}
