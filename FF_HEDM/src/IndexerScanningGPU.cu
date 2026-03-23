//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated FF-HEDM Scanning-Mode Indexer for MIDAS.
//
// Architecture: Same flattened-evaluation kernel as IndexerGPU.cu,
// but the outer loop is over voxels in a scanning grid instead of
// spotIDs. Supports 3 modes:
//   1. Spot-driven (default): for each voxel, iterate spots passing
//      beam proximity filter, generate candidate orientations.
//   2. MicFile-seeded: use nearest microstructure entry's orientation.
//   3. GrainsFile-seeded: try each grain's orientation.
// Writes output in consolidated format (IndexerConsolidatedIO.h).
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
#include "IndexerConsolidatedIO.h"
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
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define MAX_N_SPOTS 100000000
#define MAX_N_OR 7200
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_STEPS 2000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS 14
#define N_COL_OBSSPOTS 10
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
static size_t *data = NULL;
static size_t *ndata = NULL;
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
  int voxelIdx;      // Which voxel this tuple belongs to (for multi-voxel batching)
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
                 const RealType *d_ObsSpotsLab, // [n_spots × N_COL_OBSSPOTS]
                 RealType RefRad, const size_t *d_data, const size_t *d_ndata,
                 const int *d_ringsToReject, int nRingsToRejectCalc,
                 int *nMatchesFracCalc, RealType ga, RealType gb,
                 RealType gc,     // grain position for IA
                 RealType *avgIA, // output: average internal angle
                 const double *d_ypos, double BeamSize) // beam proximity
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

    size_t nInBin = d_ndata[Pos * 2 + 0];
    size_t DataPos = d_ndata[Pos * 2 + 1];

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
      int spotRow = (int)d_data[(DataPos + is) * 2 + 0];
      int scannrobs = (int)d_data[(DataPos + is) * 2 + 1];
      int base = spotRow * N_COL_OBSSPOTS;

      // Filter 0: beam proximity check (matching OMP)
      if (d_ypos != nullptr && BeamSize > 0) {
        RealType theorOmeRad = spots[sp * N_COL_THEORSPOTS + 6] * deg2rad;
        RealType yRot = ga * sin(theorOmeRad) + gb * cos(theorOmeRad);
        if (fabs(yRot - d_ypos[scannrobs]) >= BeamSize / 2.0)
          continue;
      }

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
    int n_hkls_d, const RealType *d_ObsSpotsLab, const size_t *d_data,
    const size_t *d_ndata, const int *d_ringsToReject, int nRingsToRejectCalc,
    SpotResult *d_results,    // [nResultSlots] — per-spotID best
    RealType *d_theorScratch, // [batchSize × maxTheorSpots × N_COL_THEORSPOTS]
    int maxTheorSpots,
    const double *d_ypos, double BeamSize,
    int maxSpotsPerVoxel) {
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
                       &nMatchesFracCalc, t.ga, t.gb, t.gc, &avgIA,
                       d_ypos, BeamSize);

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

  SpotResult *res = &d_results[t.voxelIdx * maxSpotsPerVoxel + spotIdx];
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


// ─────────────────────────────────────────────────────────────
// Device constant memory for on-GPU orientation generation
// ─────────────────────────────────────────────────────────────
__device__ int d_HKLints[MAX_N_HKLS][4];
__device__ double d_RingHKL[MAX_N_RINGS][3];
__device__ double d_RingTtheta[MAX_N_RINGS];
__device__ int d_SGNum;
__device__ double d_ABCABG[6];

// ─────────────────────────────────────────────────────────────
// Device: pure math helpers for orientation generation
// ─────────────────────────────────────────────────────────────
__device__ static inline void d_MatrixMultF33(double m[3][3], double n[3][3], double res[3][3]) {
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      res[r][c] = m[r][0] * n[0][c] + m[r][1] * n[1][c] + m[r][2] * n[2][c];
}

__device__ static inline void d_AxisAngle2RotMatrix(double axis[3], double angle, double R[3][3]) {
  double n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (n2 < EPS) {
    R[0][0] = 1; R[0][1] = 0; R[0][2] = 0;
    R[1][0] = 0; R[1][1] = 1; R[1][2] = 0;
    R[2][0] = 0; R[2][1] = 0; R[2][2] = 1;
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

__device__ static double d_CalcRotationAngle(int RingNr, int n_hkls_d) {
  int habs = 0, kabs = 0, labs = 0;
  for (int i = 0; i < n_hkls_d; i++)
    if (d_HKLints[i][3] == RingNr) {
      habs = abs(d_HKLints[i][0]);
      kabs = abs(d_HKLints[i][1]);
      labs = abs(d_HKLints[i][2]);
      break;
    }
  int nz = 0;
  if (!habs) nz++;
  if (!kabs) nz++;
  if (!labs) nz++;
  if (nz == 3) return 0;
  int sg = d_SGNum;
  if (sg <= 2) return 360;
  if (sg <= 15) {
    if (nz != 2) return 360;
    if (d_ABCABG[3] == 90 && d_ABCABG[4] == 90 && labs) return 180;
    if (d_ABCABG[3] == 90 && d_ABCABG[5] == 90 && habs) return 180;
    if (d_ABCABG[3] == 90 && d_ABCABG[5] == 90 && kabs) return 180;
    return 360;
  }
  if (sg <= 74) { if (nz != 2) return 360; return 180; }
  if (sg <= 142) {
    if (!nz) return 360;
    if (nz == 1 && !labs && habs == kabs) return 180;
    if (nz == 2) return labs ? 90 : 180;
    return 360;
  }
  if (sg <= 167) {
    if (!nz) return 360;
    if (nz == 2 && labs) return 120;
    return 360;
  }
  if (sg <= 194) { if (nz == 2 && labs) return 60; return 360; }
  if (sg <= 230) {
    if (nz == 2) return 90;
    if (nz == 1 && (habs == kabs || kabs == labs || habs == labs)) return 180;
    if (!nz && habs == kabs && kabs == labs) return 120;
    return 360;
  }
  return 0;
}

__device__ static void d_GenerateIdealSpots(double ys, double zs, double ttheta,
                                 double eta, double Ring_rad, double Rsample,
                                 double Hbeam, double step_size, double y0v[],
                                 double z0v[], int *nSteps) {
  int qc2 = 0;
  double eh, qc, cy = 0, cz = 0, ymax_z0 = 0, ymin_z0 = 0, ymax = 0, ymin = 0, zmin = 0, zmax = 0;
  if (eta > 90) eh = 180 - eta;
  else if (eta < -90) eh = 180 - fabs(eta);
  else eh = 90 - fabs(eta);
  Hbeam += 2 * (Rsample * tan(ttheta * deg2rad)) * sin(eh * deg2rad);
  double epole = 1 + rad2deg * acos(1 - Hbeam / Ring_rad);
  double eeq = 1 + rad2deg * acos(1 - Rsample / Ring_rad);
  if (eta >= epole && eta <= (90 - eeq)) { qc = 1; cy = -1; cz = 1; }
  else if (eta >= (90 + eeq) && eta <= (180 - epole)) { qc = 2; cy = -1; cz = -1; }
  else if (eta >= (-90 + eeq) && eta <= -epole) { qc = 2; cy = 1; cz = 1; }
  else if (eta >= (-180 + epole) && eta <= (-90 - eeq)) { qc = 1; cy = 1; cz = -1; }
  else qc = 0;
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
    if (eta > -epole && eta < epole) { ymax = ymaxR; ymin = yminR; cz = 1; }
    else if (eta < (-180 + epole)) { ymax = ymaxR; ymin = yminR; cz = -1; }
    else if (eta > (180 - epole)) { ymax = ymaxR; ymin = yminR; cz = -1; }
    else if (eta > (90 - eeq) && eta < (90 + eeq)) { qc2 = 1; zmax = zmaxH; zmin = zminH; cy = -1; }
    else if (eta > (-90 - eeq) && eta < (-90 + eeq)) { qc2 = 1; zmax = zmaxH; zmin = zminH; cy = 1; }
  }
  double y1, z1, y2, z2;
  int ns;
  if (!qc2) {
    y1 = ymin; z1 = cz * sqrt(Ring_rad * Ring_rad - y1 * y1);
    y2 = ymax; z2 = cz * sqrt(Ring_rad * Ring_rad - y2 * y2);
  } else {
    z1 = zmin; y1 = cy * sqrt(Ring_rad * Ring_rad - z1 * z1);
    z2 = zmax; y2 = cy * sqrt(Ring_rad * Ring_rad - z2 * z2);
  }
  double yd = y1 - y2, zd = z1 - z2;
  double len = sqrt(yd * yd + zd * zd);
  ns = (int)ceil(len / step_size);
  if (ns % 2 == 0) ns++;
  if (ns < 1) ns = 1;
  if (ns > MAX_N_STEPS) ns = MAX_N_STEPS;
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

__device__ static inline void d_MakeUnitLength(double d, double y, double z,
                                                double *xu, double *yu, double *zu) {
  double l = sqrt(d * d + y * y + z * z);
  if (l < EPS) { *xu = *yu = *zu = 0; return; }
  double inv = 1.0 / l;
  *xu = d * inv; *yu = y * inv; *zu = z * inv;
}

__device__ static inline void d_spot_to_gv(double xi, double yi, double zi, double omega,
                                            double *g1, double *g2, double *g3) {
  double l = sqrt(xi * xi + yi * yi + zi * zi);
  if (l < EPS) { *g1 = *g2 = *g3 = 0; return; }
  double xn = xi / l, yn = yi / l;
  double g1r = -1 + xn, g2r = yn;
  double co = cos(-omega * deg2rad), so = sin(-omega * deg2rad);
  *g1 = g1r * co - g2r * so;
  *g2 = g1r * so + g2r * co;
  *g3 = zi / l;
}

// ─────────────────────────────────────────────────────────────
// New on-GPU kernel: each thread = one (spot, voxel) pair
// Generates orientations internally and evaluates + atomicCAS
// ─────────────────────────────────────────────────────────────
__global__ void indexer_spotdriven_kernel(
    const RealType *d_ObsSpotsLab, int nSpots,
    const RealType *d_hkls_flat, int n_hkls_d,
    const size_t *d_data, const size_t *d_ndata,
    const int *d_ringsToReject, int nRingsToRejectCalc,
    SpotResult *d_results,     // [nVoxels × nSpots]
    RealType *d_theorScratch,  // [totalThreads × maxTheorSpots × N_COL_THEORSPOTS]
    int maxTheorSpots,
    const double *d_ypos, double BeamSize,
    const double *d_spotSinOme, const double *d_spotCosOme,
    const double *d_grid, int startRowNr,
    int startRowNrSp, int endRowNrSp,
    int nVoxels, int nSpotsRange,
    double StepsizeOrient, double StepsizePos,
    double Distance, double Rsample, double Hbeam)
{
  // 2D grid: x = spot index within range, y = voxel index
  int spotLocalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int voxelIdx = blockIdx.y;

  if (spotLocalIdx >= nSpotsRange || voxelIdx >= nVoxels)
    return;

  int idnr = startRowNrSp + spotLocalIdx;

  int thisRowNr = startRowNr + voxelIdx;
  double xThis = d_grid[thisRowNr * 2 + 0];
  double yThis = d_grid[thisRowNr * 2 + 1];

  // Beam proximity check
  double newY = xThis * d_spotSinOme[idnr] + yThis * d_spotCosOme[idnr];
  if (fabs(newY - d_ypos[(int)d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 9]]) > BeamSize / 2)
    return;

  // Compute seedIdx: count how many spots before this one pass beam check
  int seedIdx = 0;
  for (int j = startRowNrSp; j < idnr; j++) {
    double nY = xThis * d_spotSinOme[j] + yThis * d_spotCosOme[j];
    if (fabs(nY - d_ypos[(int)d_ObsSpotsLab[j * N_COL_OBSSPOTS + 9]]) <= BeamSize / 2)
      seedIdx++;
  }

  // Read spot data
  double ys = d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 0];
  double zs = d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 1];
  double omega = d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 2];
  double eta = d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 6];
  double RefRad = d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 3];
  int ringnr = (int)d_ObsSpotsLab[idnr * N_COL_OBSSPOTS + 5];

  // Generate ideal spots
  double y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
  int nPN = 0;
  d_GenerateIdealSpots(ys, zs, d_RingTtheta[ringnr], eta,
                       c_RingRadii[ringnr], Rsample, Hbeam, StepsizePos,
                       y0v, z0v, &nPN);

  // Per-thread scratch for theoretical spots
  long long globalTid = (long long)voxelIdx * nSpotsRange + spotLocalIdx;
  RealType *TheorSpots = &d_theorScratch[globalTid * maxTheorSpots * N_COL_THEORSPOTS];

  for (int isp = 0; isp < nPN; isp++) {
    double xi, yi, zi;
    d_MakeUnitLength(Distance, y0v[isp], z0v[isp], &xi, &yi, &zi);
    double g1, g2, g3;
    d_spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);

    double hn[3] = {g1, g2, g3};
    double hkl_d[3] = {d_RingHKL[ringnr][0], d_RingHKL[ringnr][1], d_RingHKL[ringnr][2]};

    // Generate candidate orientations
    double v[3];
    crossProduct(v, hkl_d, hn);
    double hl = CalcLength(hkl_d[0], hkl_d[1], hkl_d[2]);
    double nl = CalcLength(hn[0], hn[1], hn[2]);
    double dp = dot(hkl_d, hn);
    double angled = rad2deg * acos(fmax(-1.0, fmin(1.0, dp / (hl * nl))));
    double RM[3][3];
    d_AxisAngle2RotMatrix(v, angled, RM);
    double MaxAngle = d_CalcRotationAngle(ringnr, n_hkls_d);
    int nsteps = (int)(MaxAngle / StepsizeOrient);

    for (int o = 0; o < nsteps; o++) {
      double RM2[3][3], RM3[3][3];
      d_AxisAngle2RotMatrix(hn, o * StepsizeOrient, RM2);
      d_MatrixMultF33(RM2, RM, RM3);

      // Build OrMat for eval
      RealType OrMat[3][3];
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
          OrMat[r][c] = (RealType)RM3[r][c];

      // CalcDiffrSpots
      int nTspotsFracCalc;
      int nTspots = gpu_CalcDiffrSpots(OrMat, d_hkls_flat, n_hkls_d, TheorSpots,
                                       maxTheorSpots, d_ringsToReject,
                                       nRingsToRejectCalc, &nTspotsFracCalc);
      if (nTspots == 0 || nTspotsFracCalc == 0) continue;

      // Apply displacement
      for (int sp = 0; sp < nTspots; sp++) {
        RealType *s = &TheorSpots[sp * N_COL_THEORSPOTS];
        RealType Displ_y, Displ_z;
        midas_displacement_spot_COM(xThis, yThis, 0, s[3], s[4], s[5], s[6],
                                    &Displ_y, &Displ_z);
        s[10] = s[4] + Displ_y;
        s[11] = s[5] + Displ_z;
        midas_CalcEtaAngle(s[10], s[11], &s[12]);
        int rn = (int)s[9];
        s[13] = sqrt(s[10] * s[10] + s[11] * s[11]) - c_RingRadii[rn];
      }

      // CompareSpots
      int nMatchesFracCalc;
      RealType avgIA;
      int nMatches = gpu_CompareSpots(TheorSpots, nTspots, d_ObsSpotsLab, RefRad,
                                      d_data, d_ndata, d_ringsToReject, nRingsToRejectCalc,
                                      &nMatchesFracCalc, xThis, yThis, 0, &avgIA,
                                      d_ypos, BeamSize);

      RealType fracMatches = (RealType)nMatchesFracCalc / (RealType)nTspotsFracCalc;

      // Atomic best update per (voxel, spot)
      unsigned int fracBits = (unsigned int)__float_as_int((float)fracMatches);
      unsigned int iaBits = ~(unsigned int)__float_as_int((float)avgIA);
      unsigned long long newKey = ((unsigned long long)fracBits << 32) | (unsigned long long)iaBits;

      SpotResult *res = &d_results[voxelIdx * nSpotsRange + seedIdx];
      unsigned long long *keyAddr = &res->atomicKey;

      while (true) {
        unsigned long long oldKey = *keyAddr;
        if (newKey <= oldKey) break;
        unsigned long long prev = atomicCAS(keyAddr, oldKey, newKey);
        if (prev == oldKey) {
          res->bestFrac = fracMatches;
          res->bestIA = avgIA;
          for (int i = 0; i < 9; i++) res->bestOrMat[i] = OrMat[i / 3][i % 3];
          res->bestPos[0] = xThis;
          res->bestPos[1] = yThis;
          res->bestPos[2] = 0;
          res->nTspots = nTspots;
          res->nMatches = nMatches;
          break;
        }
      }
    } // end orientation loop
  } // end ideal spot loop
}


// ═════════════════════════════════════════════════════════════
//  HOST-SIDE CPU FUNCTIONS
// ═════════════════════════════════════════════════════════════

// ─── CPU math helpers (same as IndexerOMP) ──────────────────

// h_CalcEtaAngle removed — unused on host side

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
  data = (size_t *)mmap(0, s1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
  check(data == MAP_FAILED, "mmap %s", fn1);

  char fn2[2048];
  sprintf(fn2, "%s/nData.bin", cwd);
  int fd2 = open(fn2, O_RDONLY);
  check(fd2 < 0, "open %s: %s", fn2, strerror(errno));
  struct stat s2;
  fstat(fd2, &s2);
  ndata = (size_t *)mmap(0, s2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
  check(ndata == MAP_FAILED, "mmap %s", fn2);
  printf("Data.bin read. nData.bin read.\n");
}

// ─── Generate ideal spot positions ──────────────────────────


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


// ═════════════════════════════════════════════════════════════
//  SCANNING-MODE MAIN — processes voxels in a scanning grid
// ═════════════════════════════════════════════════════════════

// Scanning-specific additions to TParams
static double BeamSize = 0;
static int numScans = 0;
static double *ypos = NULL;     // scan positions from positions.csv
static int nYpos = 0;
static double *spotSinOme = NULL;
static double *spotCosOme = NULL;

// Grid and mic data
static double *grid = NULL;
static double *mic = NULL;
static int nrMic = 0;
static int hasMic = 0;
static int hasGrains = 0;
static double *grainsOM = NULL;
static int nrGrains = 0;
static double startEuler[3] = {0};
static char MicFN[4096] = {0};

static void ReadScanParams(char *fn) {
  FILE *fp = fopen(fn, "r");
  if (!fp) return;
  char line[4096], dummy[4096];
  while (fgets(line, 4096, fp)) {
    if (!strncmp(line, "BeamSize ", 9)) {
      sscanf(line, "%s %lf", dummy, &BeamSize);
      continue;
    }
    if (!strncmp(line, "NumScans ", 9)) {
      sscanf(line, "%s %d", dummy, &numScans);
      continue;
    }
    if (!strncmp(line, "MicFile ", 8)) {
      sscanf(line, "%s %s", dummy, MicFN);
      hasMic = 1;
      continue;
    }
    if (!strncmp(line, "StartEuler ", 11)) {
      sscanf(line, "%s %lf %lf %lf", dummy, &startEuler[0], &startEuler[1], &startEuler[2]);
      continue;
    }
  }
  fclose(fp);
}

static void LoadPositions(const char *fn) {
  FILE *f = fopen(fn, "r");
  if (!f) { printf("Cannot open %s\n", fn); return; }
  char line[4096];
  int cap = 1024;
  ypos = (double *)malloc(cap * sizeof(double));
  nYpos = 0;
  while (fgets(line, 4096, f)) {
    if (nYpos >= cap) { cap *= 2; ypos = (double *)realloc(ypos, cap * sizeof(double)); }
    double y;
    sscanf(line, "%lf", &y);
    ypos[nYpos++] = y;
  }
  fclose(f);
  printf("positions.csv: %d scan positions\n", nYpos);
}



static void LoadMic(const char *fn) {
  FILE *f = fopen(fn, "r");
  if (!f) { hasMic = 0; return; }
  char line[4096];
  // skip 4 header lines
  for (int i = 0; i < 4; i++) (void)fgets(line, 4096, f);
  int cap = 1024;
  mic = (double *)malloc(cap * 5 * sizeof(double));
  nrMic = 0;
  while (fgets(line, 4096, f)) {
    if (nrMic >= cap) { cap *= 2; mic = (double *)realloc(mic, cap * 5 * sizeof(double)); }
    double vals[10];
    sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &vals[0], &vals[1], &vals[2], &vals[3], &vals[4],
           &vals[5], &vals[6], &vals[7], &vals[8], &vals[9]);
    mic[nrMic * 5 + 0] = vals[3]; // x
    mic[nrMic * 5 + 1] = vals[4]; // y
    mic[nrMic * 5 + 2] = vals[7]; // euler1
    mic[nrMic * 5 + 3] = vals[8]; // euler2
    mic[nrMic * 5 + 4] = vals[9]; // euler3
    nrMic++;
  }
  fclose(f);
  printf("Mic file: %d entries\n", nrMic);
}

static void LoadGrains(const char *fn) {
  FILE *f = fopen(fn, "r");
  if (!f) return;
  char line[4096];
  // Skip header lines starting with %
  while (fgets(line, 4096, f) && line[0] == '%');
  int cap = 256;
  grainsOM = (double *)malloc(cap * 9 * sizeof(double));
  nrGrains = 0;
  do {
    if (nrGrains >= cap) { cap *= 2; grainsOM = (double *)realloc(grainsOM, cap * 9 * sizeof(double)); }
    double id, vals[21];
    if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &id, &vals[0], &vals[1], &vals[2], &vals[3], &vals[4],
               &vals[5], &vals[6], &vals[7], &vals[8], &vals[9],
               &vals[10], &vals[11], &vals[12], &vals[13], &vals[14],
               &vals[15], &vals[16], &vals[17], &vals[18], &vals[19]) >= 10) {
      for (int j = 0; j < 9; j++)
        grainsOM[nrGrains * 9 + j] = vals[j];
      nrGrains++;
    }
  } while (fgets(line, 4096, f));
  fclose(f);
  hasGrains = 1;
  printf("Grains file: %d grains\n", nrGrains);
}

// Helper: Euler angles to orientation matrix (same as CPU IndexerScanningOMP)
static void Euler2OrientMat(double Euler[3], double OrientMat[3][3]) {
  double psi = Euler[0] * deg2rad, phi = Euler[1] * deg2rad, theta = Euler[2] * deg2rad;
  double cps = cos(psi), cph = cos(phi), cth = cos(theta);
  double sps = sin(psi), sph = sin(phi), sth = sin(theta);
  OrientMat[0][0] = cth * cps - sth * cph * sps;
  OrientMat[0][1] = -cth * sps - sth * cph * cps;
  OrientMat[0][2] = sth * sph;
  OrientMat[1][0] = sth * cps + cth * cph * sps;
  OrientMat[1][1] = -sth * sps + cth * cph * cps;
  OrientMat[1][2] = -cth * sph;
  OrientMat[2][0] = sph * sps;
  OrientMat[2][1] = sph * cps;
  OrientMat[2][2] = cph;
}

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("IndexerScanningGPU %s\n", MIDAS_VERSION);

  if (argc < 6) {
    printf("Usage: %s paramtest.txt blockNr nBlocks numScans numProcs\n", argv[0]);
    return 1;
  }

  // 1. Read parameters
  struct TParams Params;
  memset(&Params, 0, sizeof(Params));
  ReadParams(argv[1], &Params);
  ReadScanParams(argv[1]);

  int blockNr = atoi(argv[2]);
  int nBlocks = atoi(argv[3]);
  numScans = atoi(argv[4]);
  int numProcs = atoi(argv[5]);

  int nVoxels = numScans * numScans;
  int startRowNr = (int)(ceil((double)nVoxels / (double)nBlocks)) * blockNr;
  int tmp = (int)(ceil((double)nVoxels / (double)nBlocks)) * (blockNr + 1);
  int endRowNr = tmp < nVoxels ? tmp : nVoxels;
  nVoxels = endRowNr - startRowNr;

  printf("numScans=%d, totalVoxels=%d, block=%d/%d, rows=%d-%d (%d voxels), procs=%d\n",
         numScans, numScans * numScans, blockNr, nBlocks, startRowNr, endRowNr, nVoxels, numProcs);

  // 2. Read HKLs
  char hklfn[4096], aline[4096];
  sprintf(hklfn, "hkls.csv");
  FILE *hklf = fopen(hklfn, "r");
  check(!hklf, "Cannot open %s", hklfn);
  (void)fgets(aline, 4096, hklf);
  while (fgets(aline, 4096, hklf)) {
    double h, k, l, rn, ds, th, rr;
    sscanf(aline, "%lf %lf %lf %lf %lf %lf %lf", &h, &k, &l, &rn, &ds, &th, &rr);
    hkls[n_hkls][0] = (RealType)h;
    hkls[n_hkls][1] = (RealType)k;
    hkls[n_hkls][2] = (RealType)l;
    hkls[n_hkls][3] = (RealType)rn;
    hkls[n_hkls][4] = (RealType)ds;
    hkls[n_hkls][5] = (RealType)th;
    hkls[n_hkls][6] = (RealType)rr;
    RingTtheta[(int)rn] = 2 * (RealType)th;
    RingHKL[(int)rn][0] = (RealType)h;
    RingHKL[(int)rn][1] = (RealType)k;
    RingHKL[(int)rn][2] = (RealType)l;
    HKLints[n_hkls][0] = (int)h;
    HKLints[n_hkls][1] = (int)k;
    HKLints[n_hkls][2] = (int)l;
    HKLints[n_hkls][3] = (int)rn;
    n_hkls++;
  }
  fclose(hklf);
  printf("HKLs: %d\n", n_hkls);

  // 3. Read observed spots and bins
  char tmpstr[4096];
  sprintf(tmpstr, "%s", Params.OutputFolder);
  char *cwdstr = dirname(tmpstr);
  n_spots = ReadSpots(cwdstr);
  printf("Spots: %zu\n", n_spots);
  ReadBins(cwdstr);

  int HighestRingNo = 0;
  for (int i = 0; i < MAX_N_RINGS; i++)
    if (Params.RingRadii[i]) HighestRingNo = i;
  n_ring_bins = HighestRingNo;
  n_eta_bins = (int)ceilf(360.0 / Params.EtaBinSize);
  n_ome_bins = (int)ceilf(360.0 / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize;
  OmeBinSize = Params.OmeBinSize;
  printf("Bins: ring=%d eta=%d ome=%d\n", n_ring_bins, n_eta_bins, n_ome_bins);

  // 4. Scanning-specific loading
  LoadPositions("positions.csv");
  // Build grid from positions (matching OMP behavior)
  {
    int totalVox = numScans * numScans;
    grid = (double *)malloc(totalVox * 2 * sizeof(double));
    for (int i = 0; i < numScans; i++)
      for (int j = 0; j < numScans; j++) {
        grid[(i * numScans + j) * 2 + 0] = ypos[i];
        grid[(i * numScans + j) * 2 + 1] = ypos[j];
      }
    printf("Grid: %d voxels (from positions.csv)\n", totalVox);
  }
  if (hasMic && strlen(MicFN) > 0) LoadMic(MicFN);
  if (Params.isGrainsInput) LoadGrains(Params.GrainsFileName);

  // Pre-compute sin/cos of omega for beam proximity filter
  spotSinOme = (double *)malloc(n_spots * sizeof(double));
  spotCosOme = (double *)malloc(n_spots * sizeof(double));
  for (size_t i = 0; i < n_spots; i++) {
    double omeRad = ObsSpotsLab[i * N_COL_OBSSPOTS + 2] * deg2rad;
    spotSinOme[i] = sin(omeRad);
    spotCosOme[i] = cos(omeRad);
  }
  // Compute spot range for RingToIndex (matching OMP behavior)
  int RingToIndex = Params.RingNumbers[0];
  size_t startRowNrSp = n_spots, endRowNrSp = 0;
  for (size_t i = 0; i < n_spots; i++) {
    if ((int)ObsSpotsLab[i * N_COL_OBSSPOTS + 5] == RingToIndex && startRowNrSp > i)
      startRowNrSp = i;
    if ((int)ObsSpotsLab[i * N_COL_OBSSPOTS + 5] == RingToIndex && endRowNrSp < i)
      endRowNrSp = i;
  }
  printf("Spot-driven range: RingToIndex=%d, startRowNrSp=%zu, endRowNrSp=%zu\n",
         RingToIndex, startRowNrSp, endRowNrSp);

  // 5. Precompute margins
  double omemargins[181], etamargins[MAX_N_RINGS];
  for (int i = 1; i < 180; i++)
    omemargins[i] = Params.MarginOme + (0.5 * Params.StepsizeOrient / fabs(sin(i * deg2rad)));
  omemargins[0] = omemargins[1];
  omemargins[180] = omemargins[1];
  for (int i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] == 0) etamargins[i] = 0;
    else etamargins[i] = rad2deg * atanf(Params.MarginEta / Params.RingRadii[i]) + 0.5 * Params.StepsizeOrient;
  }

  // ═══════════════════════════════════════════════════════════
  //  GPU SETUP
  // ═══════════════════════════════════════════════════════════
  int deviceId = 0;
  CUDA_CHECK(cudaSetDevice(deviceId));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  printf("GPU: %s (%.0f MB)\n", prop.name, prop.totalGlobalMem / 1048576.0);

  // Upload constant memory
  {
    RealType f_RingRadii[MAX_N_RINGS];
    for (int i = 0; i < MAX_N_RINGS; i++) f_RingRadii[i] = (RealType)Params.RingRadii[i];
    CUDA_CHECK(cudaMemcpyToSymbol(c_RingRadii, f_RingRadii, sizeof(f_RingRadii)));
    RealType f_OmegaRanges[MAX_N_OMEGARANGES][2];
    for (int i = 0; i < MAX_N_OMEGARANGES; i++) {
      f_OmegaRanges[i][0] = (RealType)Params.OmegaRanges[i][0];
      f_OmegaRanges[i][1] = (RealType)Params.OmegaRanges[i][1];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_OmegaRanges, f_OmegaRanges, sizeof(f_OmegaRanges)));
    RealType f_BoxSizes[MAX_N_OMEGARANGES][4];
    for (int i = 0; i < MAX_N_OMEGARANGES; i++)
      for (int j = 0; j < 4; j++)
        f_BoxSizes[i][j] = (RealType)Params.BoxSizes[i][j];
    CUDA_CHECK(cudaMemcpyToSymbol(c_BoxSizes, f_BoxSizes, sizeof(f_BoxSizes)));
    RealType f_omemargins[181];
    for (int i = 0; i < 181; i++) f_omemargins[i] = (RealType)omemargins[i];
    CUDA_CHECK(cudaMemcpyToSymbol(c_omemargins, f_omemargins, sizeof(f_omemargins)));
    RealType f_etamargins[MAX_N_RINGS];
    for (int i = 0; i < MAX_N_RINGS; i++) f_etamargins[i] = (RealType)etamargins[i];
    CUDA_CHECK(cudaMemcpyToSymbol(c_etamargins, f_etamargins, sizeof(f_etamargins)));
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_ringsToReject, Params.RingsToReject, sizeof(Params.RingsToReject)));

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

  // Upload data for on-GPU orientation generation
  CUDA_CHECK(cudaMemcpyToSymbol(d_HKLints, HKLints, sizeof(HKLints)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_RingHKL, RingHKL, sizeof(RingHKL)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_RingTtheta, RingTtheta, sizeof(RingTtheta)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_SGNum, &SGNum, sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_ABCABG, ABCABG, sizeof(ABCABG)));

  // Upload persistent data to GPU
  RealType *d_hkls_flat;
  CUDA_CHECK(cudaMalloc(&d_hkls_flat, (size_t)n_hkls * 7 * sizeof(RealType)));
  CUDA_CHECK(cudaMemcpy(d_hkls_flat, hkls, (size_t)n_hkls * 7 * sizeof(RealType), cudaMemcpyHostToDevice));

  size_t *d_data, *d_ndata;
  {
    char fn1[2048], fn2[2048];
    sprintf(fn1, "%s/Data.bin", cwdstr);
    sprintf(fn2, "%s/nData.bin", cwdstr);
    struct stat s1, s2;
    stat(fn1, &s1);
    stat(fn2, &s2);
    CUDA_CHECK(cudaMalloc(&d_data, s1.st_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, s1.st_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ndata, s2.st_size));
    CUDA_CHECK(cudaMemcpy(d_ndata, ndata, s2.st_size, cudaMemcpyHostToDevice));
  }

  // Upload ypos for beam proximity checks in kernel
  double *d_ypos;
  CUDA_CHECK(cudaMalloc(&d_ypos, nYpos * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_ypos, ypos, nYpos * sizeof(double), cudaMemcpyHostToDevice));
  double BeamSize = Params.Hbeam;

  size_t nObsElems = (size_t)n_spots * N_COL_OBSSPOTS;
  RealType *d_ObsSpotsLab;
  CUDA_CHECK(cudaMalloc(&d_ObsSpotsLab, nObsElems * sizeof(RealType)));
  {
    RealType *h_obs = (RealType *)malloc(nObsElems * sizeof(RealType));
    for (size_t i = 0; i < nObsElems; i++)
      h_obs[i] = (RealType)ObsSpotsLab[i];
    CUDA_CHECK(
        cudaMemcpy(d_ObsSpotsLab, h_obs, nObsElems * sizeof(RealType), cudaMemcpyHostToDevice));
    free(h_obs);
  }
  printf("GPU obs spots uploaded (%zu elements)\n", nObsElems);

  int *d_ringsToReject;
  CUDA_CHECK(cudaMalloc(&d_ringsToReject, MAX_N_RINGS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ringsToReject, Params.RingsToReject, MAX_N_RINGS * sizeof(int), cudaMemcpyHostToDevice));

  int maxTheorSpots = n_hkls * 2;
  size_t scratchPerThread = (size_t)maxTheorSpots * N_COL_THEORSPOTS * sizeof(RealType);

  // Pre-allocate OrTmp buffers for CPU tuple generation
  double **OrTmp_all = (double **)malloc(numProcs * sizeof(double *));
  for (int t = 0; t < numProcs; t++)
    OrTmp_all[t] = (double *)malloc(MAX_N_OR * 9 * sizeof(double));

  // ═══════════════════════════════════════════════════════════
  //  VOXEL LOOP — process each voxel on GPU
  // ═══════════════════════════════════════════════════════════
  printf("Starting scanning indexer for %d voxels...\n", nVoxels); fflush(stdout);

  // Allocate per-voxel accumulators for consolidated output
  VoxelAccumulator *accs = (VoxelAccumulator *)calloc(nVoxels, sizeof(VoxelAccumulator));
  for (int vi = 0; vi < nVoxels; vi++)
    VoxelAccum_init(&accs[vi]);


  // GPU memory for results — reused across voxels
  // For seeded modes, results array = 1 slot per seed
  // For spot-driven, results array = 1 slot per spotID
  int maxResultSlots = (hasMic || hasGrains) ? (hasGrains ? nrGrains : 1) : (int)n_spots;

  SpotResult *d_results;
  CUDA_CHECK(cudaMalloc(&d_results, maxResultSlots * sizeof(SpotResult)));


  // Determine max batch size from GPU memory
  size_t freeMem, totalMem;
  CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
  size_t bytesPerThread = sizeof(EvalTuple) + scratchPerThread;
  size_t maxBatchGPU = (freeMem / 3) / bytesPerThread;
  if (maxBatchGPU < 1024) maxBatchGPU = 1024;


  EvalTuple *d_tuples;
  CUDA_CHECK(cudaMalloc(&d_tuples, maxBatchGPU * sizeof(EvalTuple)));
  RealType *d_theorScratch;
  CUDA_CHECK(cudaMalloc(&d_theorScratch, maxBatchGPU * scratchPerThread));


  int blockSize = 256;

  for (int thisRowNr = startRowNr; thisRowNr < endRowNr; thisRowNr++) {
    int vi = thisRowNr - startRowNr;
    VoxelAccumulator *acc = &accs[vi];
    double xThis = grid[thisRowNr * 2 + 0];
    double yThis = grid[thisRowNr * 2 + 1];


    // ─── MODE 1: MicFile-seeded ────────────────────────────
    if (hasMic == 1) {
      int bestRow = -1;
      double bestLen = 1e10;
      for (int iter = 0; iter < nrMic; iter++) {
        double lenThis = sqrt((xThis - mic[iter * 5 + 0]) * (xThis - mic[iter * 5 + 0]) +
                              (yThis - mic[iter * 5 + 1]) * (yThis - mic[iter * 5 + 1]));
        if (lenThis < bestLen) { bestLen = lenThis; bestRow = iter; }
      }
      if (bestRow < 0) continue;

      double eulerThis[3], OMThis[3][3];
      eulerThis[0] = mic[bestRow * 5 + 2] * rad2deg;
      eulerThis[1] = mic[bestRow * 5 + 3] * rad2deg;
      eulerThis[2] = mic[bestRow * 5 + 4] * rad2deg;
      Euler2OrientMat(eulerThis, OMThis);

      // Generate single tuple for this seeded orientation at the voxel position
      EvalTuple t;
      t.spotIdx = 0;
      t.voxelIdx = 0;
      for (int k = 0; k < 9; k++) t.OrMat[k] = OMThis[k / 3][k % 3];
      t.ga = xThis; t.gb = yThis; t.gc = 0; t.RefRad = 0;

      // Reset results
      SpotResult h_res;
      memset(&h_res, 0, sizeof(h_res));
      h_res.bestFrac = -1.0; h_res.bestIA = 999.0;
      CUDA_CHECK(cudaMemcpy(d_results, &h_res, sizeof(SpotResult), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_tuples, &t, sizeof(EvalTuple), cudaMemcpyHostToDevice));

      indexer_eval_kernel<<<1, 1>>>(d_tuples, 1, d_hkls_flat, n_hkls,
                                     d_ObsSpotsLab, d_data, d_ndata,
                                     d_ringsToReject, Params.nRingsToRejectCalc,
                                     d_results, d_theorScratch, maxTheorSpots,
                                     d_ypos, BeamSize, 1);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(&h_res, d_results, sizeof(SpotResult), cudaMemcpyDeviceToHost));

      if (h_res.bestFrac > 0) {
        double outArr[16] = {0, h_res.bestIA,
          h_res.bestOrMat[0], h_res.bestOrMat[1], h_res.bestOrMat[2],
          h_res.bestOrMat[3], h_res.bestOrMat[4], h_res.bestOrMat[5],
          h_res.bestOrMat[6], h_res.bestOrMat[7], h_res.bestOrMat[8],
          h_res.bestPos[0], h_res.bestPos[1], h_res.bestPos[2],
          (double)h_res.nTspots, (double)h_res.nMatches};
        size_t keyArr[4] = {0, (size_t)h_res.nMatches, 0, 0};
        VoxelAccum_addSolution(acc, outArr, keyArr, NULL, 0);
      }

    // ─── MODE 2: GrainsFile-seeded ─────────────────────────
    } else if (hasGrains == 1) {
      // Generate one tuple per grain orientation
      int nSeeds = nrGrains;
      SpotResult *h_res = (SpotResult *)calloc(nSeeds, sizeof(SpotResult));
      for (int i = 0; i < nSeeds; i++) { h_res[i].bestFrac = -1.0; h_res[i].bestIA = 999.0; }
      CUDA_CHECK(cudaMemcpy(d_results, h_res, nSeeds * sizeof(SpotResult), cudaMemcpyHostToDevice));

      EvalTuple *h_tuples = (EvalTuple *)malloc(nSeeds * sizeof(EvalTuple));
      for (int g = 0; g < nSeeds; g++) {
        h_tuples[g].spotIdx = g;
        h_tuples[g].voxelIdx = 0;
        for (int k = 0; k < 9; k++) h_tuples[g].OrMat[k] = grainsOM[g * 9 + k];
        h_tuples[g].ga = xThis; h_tuples[g].gb = yThis;
        h_tuples[g].gc = 0; h_tuples[g].RefRad = 0;
      }
      CUDA_CHECK(cudaMemcpy(d_tuples, h_tuples, nSeeds * sizeof(EvalTuple), cudaMemcpyHostToDevice));

      int nBlks = (nSeeds + blockSize - 1) / blockSize;
      indexer_eval_kernel<<<nBlks, blockSize>>>(d_tuples, nSeeds, d_hkls_flat, n_hkls,
                                                d_ObsSpotsLab, d_data, d_ndata,
                                                d_ringsToReject, Params.nRingsToRejectCalc,
                                                d_results, d_theorScratch, maxTheorSpots,
                                                d_ypos, BeamSize, nSeeds);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_res, d_results, nSeeds * sizeof(SpotResult), cudaMemcpyDeviceToHost));

      // Collect best result across all grains
      int bestG = -1; double bestFr = -1;
      for (int g = 0; g < nSeeds; g++) {
        if (h_res[g].bestFrac > bestFr) { bestFr = h_res[g].bestFrac; bestG = g; }
      }
      if (bestG >= 0 && bestFr > 0) {
        SpotResult *r = &h_res[bestG];
        double outArr[16] = {0, r->bestIA,
          r->bestOrMat[0], r->bestOrMat[1], r->bestOrMat[2],
          r->bestOrMat[3], r->bestOrMat[4], r->bestOrMat[5],
          r->bestOrMat[6], r->bestOrMat[7], r->bestOrMat[8],
          r->bestPos[0], r->bestPos[1], r->bestPos[2],
          (double)r->nTspots, (double)r->nMatches};
        size_t keyArr[4] = {0, (size_t)r->nMatches, 0, 0};
        VoxelAccum_addSolution(acc, outArr, keyArr, NULL, 0);
      }
      free(h_tuples); free(h_res);

    // ─── MODE 3: Spot-driven (default) ─────────────────────
    } else {
      // Mode 3 handled below in multi-voxel batch
    }

    if (thisRowNr % 50 == 0 && (hasMic || hasGrains))
      printf("Voxel %d/%d done (%.1fs)\n", thisRowNr - startRowNr, nVoxels, omp_get_wtime() - start_time);
  }

  // ═══════════════════════════════════════════════════════════
  //  MODE 3: On-GPU spot-driven indexing (no tuple materialization)
  // ═══════════════════════════════════════════════════════════
  if (!hasMic && !hasGrains) {
    double tSpotDrivenStart = omp_get_wtime();

    // Free Mode 1/2 GPU buffers — not needed for spot-driven
    CUDA_CHECK(cudaFree(d_tuples)); d_tuples = NULL;
    CUDA_CHECK(cudaFree(d_theorScratch)); d_theorScratch = NULL;
    CUDA_CHECK(cudaFree(d_results)); d_results = NULL;

    int nSpotsRange = endRowNrSp - startRowNrSp + 1;
    printf("Spot-driven mode (on-GPU): %d voxels × %d spots = %d threads\n",
           nVoxels, nSpotsRange, nVoxels * nSpotsRange);

    // Upload spotSinOme, spotCosOme, grid to GPU
    double *d_spotSinOme, *d_spotCosOme, *d_gridGPU;
    CUDA_CHECK(cudaMalloc(&d_spotSinOme, n_spots * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spotCosOme, n_spots * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gridGPU, totalVox * 2 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_spotSinOme, spotSinOme, n_spots * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spotCosOme, spotCosOme, n_spots * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gridGPU, grid, totalVox * 2 * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate results: nVoxels × nSpotsRange
    int totalResultSlots = nVoxels * nSpotsRange;
    SpotResult *h_allResults = (SpotResult *)calloc(totalResultSlots, sizeof(SpotResult));
    for (int i = 0; i < totalResultSlots; i++) {
      h_allResults[i].bestFrac = -1.0;
      h_allResults[i].bestIA = 999.0;
      h_allResults[i].atomicKey = 0;
    }
    SpotResult *d_spotResults;
    CUDA_CHECK(cudaMalloc(&d_spotResults, totalResultSlots * sizeof(SpotResult)));
    CUDA_CHECK(cudaMemcpy(d_spotResults, h_allResults, totalResultSlots * sizeof(SpotResult), cudaMemcpyHostToDevice));

    // Allocate per-thread scratch for theoretical spots
    // Total threads = nVoxels × nSpotsRange, but we can't allocate scratch for ALL
    // Use batched approach: process voxels in groups sized by GPU memory
    size_t freeMem2, totalMem2;
    CUDA_CHECK(cudaMemGetInfo(&freeMem2, &totalMem2));
    size_t scratchPerThread2 = (size_t)maxTheorSpots * N_COL_THEORSPOTS * sizeof(RealType);
    size_t availForScratch = (freeMem2 > 512*1024*1024) ? (freeMem2 - 512*1024*1024) : (256*1024*1024);
    int maxThreadsGPU = (int)(availForScratch / scratchPerThread2);
    int voxelsPerBatch = maxThreadsGPU / nSpotsRange;
    if (voxelsPerBatch < 1) voxelsPerBatch = 1;
    if (voxelsPerBatch > nVoxels) voxelsPerBatch = nVoxels;

    int batchThreads = voxelsPerBatch * nSpotsRange;
    RealType *d_batchScratch;
    CUDA_CHECK(cudaMalloc(&d_batchScratch, (size_t)batchThreads * scratchPerThread2));

    int nBatches = (nVoxels + voxelsPerBatch - 1) / voxelsPerBatch;
    printf("  GPU batch: %d voxels/batch, %d batches, scratch=%.0f MB\n",
           voxelsPerBatch, nBatches,
           (double)batchThreads * scratchPerThread2 / 1048576.0);

    // Launch kernel in voxel batches
    dim3 block(256, 1);
    for (int bi = 0; bi < nBatches; bi++) {
      int vStart = bi * voxelsPerBatch;
      int vEnd = vStart + voxelsPerBatch;
      if (vEnd > nVoxels) vEnd = nVoxels;
      int vN = vEnd - vStart;

      dim3 grid2((nSpotsRange + block.x - 1) / block.x, vN);

      // Adjust results pointer for this batch
      SpotResult *d_batchResults = d_spotResults + vStart * nSpotsRange;

      indexer_spotdriven_kernel<<<grid2, block>>>(
          d_ObsSpotsLab, (int)n_spots,
          d_hkls_flat, n_hkls,
          d_data, d_ndata,
          d_ringsToReject, Params.nRingsToRejectCalc,
          d_batchResults,
          d_batchScratch, maxTheorSpots,
          d_ypos, BeamSize,
          d_spotSinOme, d_spotCosOme,
          d_gridGPU, startRowNr + vStart,
          startRowNrSp, endRowNrSp,
          vN, nSpotsRange,
          Params.StepsizeOrient, Params.StepsizePos,
          Params.Distance, Params.Rsample, Params.Hbeam);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      if (nBatches > 1)
        printf("  Batch %d/%d: %d voxels, %.3f s\n",
               bi + 1, nBatches, vN, omp_get_wtime() - tSpotDrivenStart);
    }

    double tKernelEnd = omp_get_wtime();
    printf("  Kernel execution: %.3f s\n", tKernelEnd - tSpotDrivenStart);

    // Download results
    CUDA_CHECK(cudaMemcpy(h_allResults, d_spotResults, totalResultSlots * sizeof(SpotResult), cudaMemcpyDeviceToHost));

    // Distribute to accumulators
    // Need to map seedIdx back: for each voxel, count passing spots and map
    for (int vi = 0; vi < nVoxels; vi++) {
      VoxelAccumulator *acc = &accs[vi];
      for (int si = 0; si < nSpotsRange; si++) {
        SpotResult *r = &h_allResults[vi * nSpotsRange + si];
        if (r->bestFrac <= 0) continue;
        double outArr[16] = {(double)si, r->bestIA,
          r->bestOrMat[0], r->bestOrMat[1], r->bestOrMat[2],
          r->bestOrMat[3], r->bestOrMat[4], r->bestOrMat[5],
          r->bestOrMat[6], r->bestOrMat[7], r->bestOrMat[8],
          r->bestPos[0], r->bestPos[1], r->bestPos[2],
          (double)r->nTspots, (double)r->nMatches};
        size_t keyArr[4] = {(size_t)si, (size_t)r->nMatches, 0, 0};
        VoxelAccum_addSolution(acc, outArr, keyArr, NULL, 0);
      }
    }

    printf("  Total spot-driven time: %.3f s\n", omp_get_wtime() - tSpotDrivenStart);

    // Cleanup
    free(h_allResults);
    CUDA_CHECK(cudaFree(d_spotResults));
    CUDA_CHECK(cudaFree(d_batchScratch));
    CUDA_CHECK(cudaFree(d_spotSinOme));
    CUDA_CHECK(cudaFree(d_spotCosOme));
    CUDA_CHECK(cudaFree(d_gridGPU));
  }


  // ═══════════════════════════════════════════════════════════
  //  Write consolidated output
  // ═══════════════════════════════════════════════════════════
  printf("Writing consolidated output files...\n");
  WriteConsolidatedFiles(accs, endRowNr - startRowNr, startRowNr, endRowNr, Params.OutputFolder);

  // Cleanup
  for (int vi = 0; vi < nVoxels; vi++) VoxelAccum_free(&accs[vi]);
  free(accs);

  CUDA_CHECK(cudaFree(d_hkls_flat));
  CUDA_CHECK(cudaFree(d_ObsSpotsLab));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ypos));
  CUDA_CHECK(cudaFree(d_ndata));
  CUDA_CHECK(cudaFree(d_ringsToReject));
  if (d_results) CUDA_CHECK(cudaFree(d_results));
  if (d_tuples) CUDA_CHECK(cudaFree(d_tuples));
  if (d_theorScratch) CUDA_CHECK(cudaFree(d_theorScratch));

  for (int t = 0; t < numProcs; t++) free(OrTmp_all[t]);
  free(OrTmp_all);
  free(spotSinOme);
  free(spotCosOme);

  double elapsed = omp_get_wtime() - start_time;
  printf("IndexerScanningGPU finished: %.2f seconds.\n", elapsed);
  return 0;
}
