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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/types.h>
#include <omp.h>
#include <libgen.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "midas_gpu_math.cuh"
#include "midas_version.h"

// ─────────────────────────────────────────────────────────────
// CUDA error checking
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", \
              __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// ─────────────────────────────────────────────────────────────
// Constants and types
// ─────────────────────────────────────────────────────────────
#define RealType float
#define deg2rad 0.0174532925199433f
#define rad2deg 57.2957795130823f

#define MAX_N_SPOTS    100000000
#define MAX_N_OR       7200
#define MAX_N_RINGS    500
#define MAX_N_HKLS     5000
#define MAX_N_STEPS    2000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS  14
#define N_COL_OBSSPOTS    9
#define N_COL_GRAINSPOTS  17
#define N_COL_GRAINMATCHES 16
#define EPS 1e-9f

#define crossProduct(a,b,c) \
  (a)[0] = (b)[1]*(c)[2] - (c)[1]*(b)[2]; \
  (a)[1] = (b)[2]*(c)[0] - (c)[2]*(b)[0]; \
  (a)[2] = (b)[0]*(c)[1] - (c)[0]*(b)[1];
#define dot(v,q) ((v)[0]*(q)[0]+(v)[1]*(q)[1]+(v)[2]*(q)[2])
#define CalcLength(x,y,z) sqrtf((x)*(x)+(y)*(y)+(z)*(z))
#define TestBit(A,k) (A[(k/32)] & (1 << (k%32)))

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
static double *ObsSpotsLab_d = NULL;  // mmap'd as double (file format)
static float  *ObsSpotsLab = NULL;    // converted to float for CPU/GPU use
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
static RealType pixelsize = 0;
static double ABCABG[6];
static int SGNum = 0;

// ─────────────────────────────────────────────────────────────
// Parameters structure (same as IndexerOMP)
// ─────────────────────────────────────────────────────────────
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
  RealType ExcludePoleAngle;
  RealType MinMatchesToAcceptFrac;
  RealType BoxSizes[MAX_N_OMEGARANGES][4];
  RealType OmegaRanges[MAX_N_OMEGARANGES][2];
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
  int spotIdx;      // Index back to spotID array (for reduction)
  float OrMat[9];   // Orientation matrix (flattened 3x3)
  float ga, gb, gc; // Sample position in lab frame
  float RefRad;     // Reference radial position of the spot
};

// Per-spotID best result (GPU output, one per spotID)
struct SpotResult {
  unsigned long long atomicKey;  // Packed (frac, -IA) for 64-bit atomicCAS — MUST be first for alignment
  float bestFrac;      // Best fraction of matches (confidence)
  float bestIA;        // Best internal angle
  float bestOrMat[9];  // Best orientation matrix
  float bestPos[3];    // Best position (ga, gb, gc)
  int   nTspots;       // Number of theoretical spots for best match
  int   nMatches;      // Number of matches for best match
};

// ─────────────────────────────────────────────────────────────
// GPU constant memory for geometry/margins
// ─────────────────────────────────────────────────────────────
__constant__ float c_RingRadii[MAX_N_RINGS];
__constant__ float c_OmegaRanges[MAX_N_OMEGARANGES][2];
__constant__ float c_BoxSizes[MAX_N_OMEGARANGES][4];
__constant__ float c_omemargins[181];
__constant__ float c_etamargins[MAX_N_RINGS];
__constant__ int   c_ringsToReject[MAX_N_RINGS];

// Scalar constants in constant memory
struct GPUParams {
  float Distance;
  float Wavelength;
  float ExcludePoleAngle;
  float MarginRad;
  float MarginRadial;
  float MarginOme;
  int   NoOfOmegaRanges;
  int   nRingsToRejectCalc;
  int   n_ring_bins;
  int   n_eta_bins;
  int   n_ome_bins;
  float EtaBinSize;
  float OmeBinSize;
};
__constant__ GPUParams c_params;

// ─────────────────────────────────────────────────────────────
// Device: CalcDiffrSpots_Furnace — compute theoretical spots
// from an orientation matrix against HKL table
// ─────────────────────────────────────────────────────────────
__device__
int gpu_CalcDiffrSpots(
    const float OrMat[3][3],
    const float *d_hkls_flat, // [n_hkls × 7]
    int n_hkls_d,
    float *spots_out,         // flat: [max_spots × N_COL_THEORSPOTS]
    int max_spots,
    const int *d_ringsToReject,
    int nRingsToRejectCalc,
    int *nTspotsFracCalc)
{
  int spotnr = 0;
  int nFracCalc = 0;

  for (int ih = 0; ih < n_hkls_d && spotnr < max_spots; ih++) {
    float Ghkl[3] = {
      d_hkls_flat[ih*7+0],
      d_hkls_flat[ih*7+1],
      d_hkls_flat[ih*7+2]
    };
    int ringnr = (int)d_hkls_flat[ih*7+3];
    if (ringnr < 0 || ringnr >= MAX_N_RINGS) continue;
    float RingRadius = c_RingRadii[ringnr];
    if (RingRadius < EPS) continue;
    float theta = d_hkls_flat[ih*7+5];

    float Gc[3];
    float OrM[3][3];
    for (int r=0; r<3; r++) for (int cc=0; cc<3; cc++) OrM[r][cc] = OrMat[r][cc];
    midas_MatrixMultF(OrM, Ghkl, Gc);

    float omegas[4], etas[4];
    int nsol;
    midas_CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nsol);

    for (int i = 0; i < nsol && spotnr < max_spots; i++) {
      float Omega = omegas[i];
      float Eta = etas[i];
      float EtaAbs = fabsf(Eta);

      if (EtaAbs < c_params.ExcludePoleAngle ||
          (180.0f - EtaAbs) < c_params.ExcludePoleAngle) continue;

      float yl, zl;
      midas_CalcSpotPosition(RingRadius, Eta, &yl, &zl);

      int keep = 0;
      for (int orn = 0; orn < c_params.NoOfOmegaRanges; orn++) {
        if (Omega > c_OmegaRanges[orn][0] && Omega < c_OmegaRanges[orn][1] &&
            yl > c_BoxSizes[orn][0] && yl < c_BoxSizes[orn][1] &&
            zl > c_BoxSizes[orn][2] && zl < c_BoxSizes[orn][3]) {
          keep = 1; break;
        }
      }
      if (!keep) continue;

      float *sp = &spots_out[spotnr * N_COL_THEORSPOTS];
      sp[0] = 0;
      sp[1] = (float)spotnr;
      sp[2] = (float)ih;
      sp[3] = c_params.Distance;
      sp[4] = yl;
      sp[5] = zl;
      sp[6] = Omega;
      sp[7] = Eta;
      sp[8] = theta;
      sp[9] = (float)ringnr;

      // Check if ring is excluded from fraction calc
      int rejected = 0;
      for (int rr = 0; rr < nRingsToRejectCalc; rr++) {
        if (ringnr == d_ringsToReject[rr]) { rejected = 1; break; }
      }
      if (!rejected) nFracCalc++;
      spotnr++;
    }
  }
  *nTspotsFracCalc = nFracCalc;
  return spotnr;
}

// ─────────────────────────────────────────────────────────────
// Device: CompareSpots — match theoretical vs observed via bins
// ─────────────────────────────────────────────────────────────
__device__
int gpu_CompareSpots(
    float *spots,             // flat: [nTspots × N_COL_THEORSPOTS]
    int nTspots,
    const float *d_ObsSpotsLab, // [n_spots × 9]
    float RefRad,
    const int *d_data,
    const int *d_ndata,
    const int *d_ringsToReject,
    int nRingsToRejectCalc,
    int *nMatchesFracCalc,
    float ga, float gb, float gc,  // grain position for IA
    float *avgIA,                   // output: average internal angle
    int debugSpot)                  // if 1, print per-spot IA trace
{
  int nMatched = 0;
  int nMatchedFrac = 0;
  float iaSum = 0.0f;
  int iaCount = 0;
  float Distance = c_params.Distance;

  for (int sp = 0; sp < nTspots; sp++) {
    float *s = &spots[sp * N_COL_THEORSPOTS];
    int RingNr = (int)s[9];
    if (RingNr <= 0 || RingNr >= MAX_N_RINGS) continue;

    float theorEta = s[12];
    float theorOme = s[6];
    float theorRadDiff = s[13];
    float theorY = s[10];
    float theorZ = s[11];

    int iRing = RingNr - 1;
    int iEta = (int)floorf((180.0f + theorEta) / c_params.EtaBinSize);
    int iOme = (int)floorf((180.0f + theorOme) / c_params.OmeBinSize);
    iEta = max(0, min(c_params.n_eta_bins - 1, iEta));
    iOme = max(0, min(c_params.n_ome_bins - 1, iOme));

    size_t Pos = (size_t)iRing;
    Pos *= (size_t)c_params.n_eta_bins;
    Pos += (size_t)iEta;
    Pos *= (size_t)c_params.n_ome_bins;
    Pos += (size_t)iOme;

    int nInBin = d_ndata[Pos * 2 + 0];
    int DataPos = d_ndata[Pos * 2 + 1];

    float etamargin = c_etamargins[RingNr];

    // Check if this ring is excluded from radial filter
    int skipRadialFilter = 0;
    for (int rr = 0; rr < nRingsToRejectCalc; rr++) {
      if (RingNr == d_ringsToReject[rr]) { skipRadialFilter = 1; break; }
    }

    int matchFound = 0;
    int bestSpotRow = -1;
    float diffOmeBest = 100000.0f;  // match CPU: find closest, no threshold

    for (int is = 0; is < nInBin; is++) {
      int spotRow = d_data[DataPos + is];
      int base = spotRow * N_COL_OBSSPOTS;

      // Filter 1: radial difference (MarginRadial)
      float obsRadDiff = d_ObsSpotsLab[base + 8];
      if (fabsf(theorRadDiff - obsRadDiff) >= c_params.MarginRadial) continue;

      // Filter 2: RefRad check (MarginRad) — skip if ring is excluded
      if (!skipRadialFilter) {
        float obsRefRad = d_ObsSpotsLab[base + 3];
        if (fabsf(RefRad - obsRefRad) >= c_params.MarginRad) continue;
      }

      // Filter 3: eta margin
      float obsEta = d_ObsSpotsLab[base + 6];
      if (fabsf(theorEta - obsEta) >= etamargin) continue;

      // Find closest omega match (no threshold)
      float obsOme = d_ObsSpotsLab[base + 2];
      float diffOme = fabsf(theorOme - obsOme);
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
        if (RingNr == d_ringsToReject[rr]) { rejected = 1; break; }
      }
      if (!rejected) nMatchedFrac++;

      // Compute internal angle for this matched spot
      int base = bestSpotRow * N_COL_OBSSPOTS;
      float obsY = d_ObsSpotsLab[base + 0];
      float obsZ = d_ObsSpotsLab[base + 1];
      float obsOme = d_ObsSpotsLab[base + 2];
      float gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
      midas_spot_to_gv_pos(Distance, theorY, theorZ, theorOme, ga, gb, gc, &gv1x, &gv1y, &gv1z);
      midas_spot_to_gv_pos(Distance, obsY,   obsZ,   obsOme,   ga, gb, gc, &gv2x, &gv2y, &gv2z);
      float ia;
      midas_CalcInternalAngle(gv1x, gv1y, gv1z, gv2x, gv2y, gv2z, &ia);
      if (debugSpot) {
        printf("[GPU IA sp=%d] row=%d ring=%d theorY=%.6f theorZ=%.6f theorOme=%.6f obsY=%.6f obsZ=%.6f obsOme=%.6f\n",
               sp, bestSpotRow, RingNr, theorY, theorZ, theorOme, obsY, obsZ, obsOme);
        printf("  dist=%.2f ga=%.6f gb=%.6f gc=%.6f gv1=(%.8f,%.8f,%.8f) gv2=(%.8f,%.8f,%.8f) ia=%.8f\n",
               Distance, ga, gb, gc, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z, ia);
      }
      if (ia < 999.0f) {
        iaSum += ia;
        iaCount++;
      }
    }
  }
  *nMatchesFracCalc = nMatchedFrac;
  *avgIA = (iaCount > 0) ? (iaSum / (float)iaCount) : 999.0f;
  if (debugSpot) {
    printf("[GPU IA SUMMARY] nMatched=%d iaSum=%.8f iaCount=%d avgIA=%.8f\n",
           nMatched, iaSum, iaCount, *avgIA);
  }
  return nMatched;
}

// ─────────────────────────────────────────────────────────────
// Main evaluation kernel: one thread = one (orientation, position) tuple
// ─────────────────────────────────────────────────────────────
__global__ void indexer_eval_kernel(
    const EvalTuple *tuples,
    int nTuples,
    const float *d_hkls_flat,
    int n_hkls_d,
    const float *d_ObsSpotsLab,
    const int *d_data,
    const int *d_ndata,
    const int *d_ringsToReject,
    int nRingsToRejectCalc,
    SpotResult *d_results,    // [nSpotIDs] — per-spotID best
    float *d_theorScratch,    // [batchSize × maxTheorSpots × N_COL_THEORSPOTS]
    int maxTheorSpots
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nTuples) return;

  EvalTuple t = tuples[tid];
  int spotIdx = t.spotIdx;

  // Build orientation matrix
  float OrMat[3][3];
  for (int r=0; r<3; r++)
    for (int c=0; c<3; c++)
      OrMat[r][c] = t.OrMat[r*3+c];

  // Per-thread scratch slice in global memory
  float *TheorSpots = &d_theorScratch[(long long)tid * maxTheorSpots * N_COL_THEORSPOTS];

  // 1. Compute theoretical diffraction spots
  int nTspots, nTspotsFracCalc;
  nTspots = gpu_CalcDiffrSpots(OrMat, d_hkls_flat, n_hkls_d,
                               TheorSpots, maxTheorSpots,
                               d_ringsToReject, nRingsToRejectCalc,
                               &nTspotsFracCalc);
  if (nTspots == 0 || nTspotsFracCalc == 0) return;

  // 2. Apply displacement for this position
  for (int sp = 0; sp < nTspots; sp++) {
    float *s = &TheorSpots[sp * N_COL_THEORSPOTS];
    float Displ_y, Displ_z;
    midas_displacement_spot_COM(
      t.ga, t.gb, t.gc,
      s[3], s[4], s[5], s[6], &Displ_y, &Displ_z);
    s[10] = s[4] + Displ_y;
    s[11] = s[5] + Displ_z;
    midas_CalcEtaAngle(s[10], s[11], &s[12]);
    int rn = (int)s[9];
    s[13] = sqrtf(s[10]*s[10] + s[11]*s[11]) - c_RingRadii[rn];
  }

  // 3. Compare with observed spots + compute IA
  int nMatchesFracCalc;
  float avgIA;
  int debugSpot = (spotIdx == 0) ? 1 : 0;
  int nMatches = gpu_CompareSpots(
    TheorSpots, nTspots, d_ObsSpotsLab, t.RefRad,
    d_data, d_ndata,
    d_ringsToReject, nRingsToRejectCalc,
    &nMatchesFracCalc,
    t.ga, t.gb, t.gc, &avgIA, debugSpot);

  float fracMatches = (float)nMatchesFracCalc / (float)nTspotsFracCalc;

  // 4. Atomic best-match update per spotID using 64-bit packed key:
  //    upper 32 bits = frac (maximize), lower 32 bits = -IA (minimize IA)
  //    For positive floats, __float_as_int preserves ordering.
  unsigned int fracBits = (unsigned int)__float_as_int(fracMatches);
  unsigned int iaBits   = (unsigned int)__float_as_int(-avgIA);  // negate: smaller IA → larger -IA
  unsigned long long newKey = ((unsigned long long)fracBits << 32) | (unsigned long long)iaBits;

  SpotResult *res = &d_results[spotIdx];
  unsigned long long *keyAddr = &res->atomicKey;

  while (true) {
    unsigned long long oldKey = *keyAddr;
    if (newKey <= oldKey) break;

    unsigned long long prev = atomicCAS(keyAddr, oldKey, newKey);
    if (prev == oldKey) {
      // Won the race — copy all result fields
      res->bestFrac = fracMatches;
      res->bestIA = avgIA;
      for (int i = 0; i < 9; i++) res->bestOrMat[i] = t.OrMat[i];
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

static inline void h_CalcEtaAngle(float y, float z, float *alpha) {
  float denom = sqrtf(y*y + z*z);
  if (denom < EPS) { *alpha = 0; return; }
  *alpha = rad2deg * acosf(fmaxf(-1.0f, fminf(1.0f, z/denom)));
  if (y > 0) *alpha = -(*alpha);
}

// h_CalcSpotPosition removed — unused on host side

// h_MatrixMultF removed — unused on host side

static inline void h_MatrixMultF33(float m[3][3], float n[3][3], float res[3][3]) {
  for (int r=0;r<3;r++) for(int c=0;c<3;c++)
    res[r][c] = m[r][0]*n[0][c] + m[r][1]*n[1][c] + m[r][2]*n[2][c];
}

static inline void h_AxisAngle2RotMatrix(float axis[3], float angle, float R[3][3]) {
  float n2 = axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2];
  if (n2 < EPS) { R[0][0]=1;R[0][1]=0;R[0][2]=0;R[1][0]=0;R[1][1]=1;R[1][2]=0;R[2][0]=0;R[2][1]=0;R[2][2]=1; return; }
  float inv=1.0f/sqrtf(n2); float u=axis[0]*inv,v=axis[1]*inv,w=axis[2]*inv;
  float rad=deg2rad*angle, co=cosf(rad), si=sinf(rad), omc=1-co;
  R[0][0]=co+u*u*omc; R[0][1]=-w*si+u*v*omc; R[0][2]=v*si+u*w*omc;
  R[1][0]=w*si+v*u*omc; R[1][1]=co+v*v*omc; R[1][2]=-u*si+v*w*omc;
  R[2][0]=-v*si+w*u*omc; R[2][1]=u*si+w*v*omc; R[2][2]=co+w*w*omc;
}

// h_CalcOmega removed — unused on host side

static float h_CalcRotationAngle(int RingNr) {
  int habs=0,kabs=0,labs=0,i;
  for (i=0;i<n_hkls;i++) if(HKLints[i][3]==RingNr){habs=abs(HKLints[i][0]);kabs=abs(HKLints[i][1]);labs=abs(HKLints[i][2]);break;}
  int nz=0; if(!habs)nz++; if(!kabs)nz++; if(!labs)nz++;
  if(nz==3)return 0;
  if(SGNum<=2) return 360;
  if(SGNum<=15){if(nz!=2)return 360; if(ABCABG[3]==90&&ABCABG[4]==90&&labs)return 180; if(ABCABG[3]==90&&ABCABG[5]==90&&habs)return 180; if(ABCABG[3]==90&&ABCABG[5]==90&&kabs)return 180; return 360;}
  if(SGNum<=74){if(nz!=2)return 360; return 180;}
  if(SGNum<=142){if(!nz)return 360; if(nz==1&&!labs&&habs==kabs)return 180; if(nz==2)return labs?90:180; return 360;}
  if(SGNum<=167){if(!nz)return 360; if(nz==2&&labs)return 120; return 360;}
  if(SGNum<=194){if(nz==2&&labs)return 60; return 360;}
  if(SGNum<=230){if(nz==2)return 90; if(nz==1&&(habs==kabs||kabs==labs||habs==labs))return 180; if(!nz&&habs==kabs&&kabs==labs)return 120; return 360;}
  return 0;
}

static int h_GenerateCandidateOrientations(double hkl[3], float hklnormal[3],
    float stepsize, float *OrMat, int *nOrient, int RingNr) {
  float v[3]; float hkl_f[3]={(float)hkl[0],(float)hkl[1],(float)hkl[2]};
  crossProduct(v, hkl_f, hklnormal);
  float hl=CalcLength(hkl_f[0],hkl_f[1],hkl_f[2]);
  float nl=CalcLength(hklnormal[0],hklnormal[1],hklnormal[2]);
  float dp=dot(hkl_f,hklnormal);
  float angled=rad2deg*acosf(fmaxf(-1.0f,fminf(1.0f,dp/(hl*nl))));
  float RM[3][3],RM2[3][3],RM3[3][3];
  h_AxisAngle2RotMatrix(v, angled, RM);
  float MaxAngle = h_CalcRotationAngle(RingNr);
  int nsteps = (int)(MaxAngle/stepsize);
  for (int o=0;o<nsteps;o++) {
    h_AxisAngle2RotMatrix(hklnormal, o*stepsize, RM2);
    h_MatrixMultF33(RM2, RM, RM3);
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) OrMat[o*9+r*3+c]=RM3[r][c];
  }
  *nOrient = nsteps;
  return 0;
}

static inline void h_MakeUnitLength(float d, float y, float z,
                                    float *xu, float *yu, float *zu) {
  float l=sqrtf(d*d+y*y+z*z); if(l<EPS){*xu=*yu=*zu=0;return;}
  float inv=1.0f/l; *xu=d*inv; *yu=y*inv; *zu=z*inv;
}

static inline void h_spot_to_gv(float xi,float yi,float zi,float omega,
                                float *g1,float *g2,float *g3){
  float l=sqrtf(xi*xi+yi*yi+zi*zi); if(l<EPS){*g1=*g2=*g3=0;return;}
  float xn=xi/l,yn=yi/l; float g1r=-1+xn,g2r=yn;
  float co=cosf(-omega*deg2rad),so=sinf(-omega*deg2rad);
  *g1=g1r*co-g2r*so; *g2=g1r*so+g2r*co; *g3=zi/l;
}

static inline void h_calc_n_max_min(float xi,float yi,float ys,float y0,
    float Rsamp,int step,int *nmax,int *nmin){
  float dy=ys-y0, a=xi*xi+yi*yi, b=2*yi*dy, c=dy*dy-Rsamp*Rsamp;
  float D=b*b-4*a*c, P=sqrtf(fabsf(D));
  float lmax=(-b+P)/(2*a)+20;
  *nmax=(int)((lmax*xi)/step); *nmin=-(*nmax);
}

static inline void h_spot_to_unrotated(float xi,float yi,float zi,
    float ys,float zs,float y0,float z0,float ss,int n,float omega,
    float *a,float *b,float *c){
  float lam=ss*(n/xi); float x1=lam*xi, y1=ys-y0+lam*yi, z1=zs-z0+lam*zi;
  float co=cosf(omega*deg2rad),so=sinf(omega*deg2rad);
  *a=x1*co+y1*so; *b=y1*co-x1*so; *c=z1;
}

// ─── Data I/O (same as IndexerOMP) ─────────────────────────

static size_t ReadBigDet(char *cwd) {
  char fn[2048]; sprintf(fn,"%s/BigDetectorMask.bin",cwd);
  int fd=open(fn,O_RDONLY); check(fd<0,"open %s: %s",fn,strerror(errno));
  struct stat s; fstat(fd,&s);
  BigDetector=(int*)mmap(0,s.st_size,PROT_READ,MAP_SHARED,fd,0);
  check(BigDetector==MAP_FAILED,"mmap %s: %s",fn,strerror(errno));
  return s.st_size;
}

static int ReadParams(char *fn, struct TParams *P) {
  FILE *fp=fopen(fn,"r"); if(!fp){printf("Cannot open %s\n",fn);return 1;}
  char line[4096],dummy[4096]; int NoRingNumbers=0, NrOfBoxSizes=0;
  P->NrOfRings=0; P->NoOfOmegaRanges=0; P->isGrainsInput=0; P->nRingsToRejectCalc=0;
  totNrPixelsBigDetector=0;
  while(fgets(line,4096,fp)){
    if(!strncmp(line,"RingNumbers ",12)){sscanf(line,"%s %d",dummy,&P->RingNumbers[NoRingNumbers++]);continue;}
    if(!strncmp(line,"RingsToExcludeFraction ",23)){sscanf(line,"%s %d",dummy,&P->RingsToReject[P->nRingsToRejectCalc++]);continue;}
    if(!strncmp(line,"BigDetSize ",11)){sscanf(line,"%s %d",dummy,&BigDetSize);totNrPixelsBigDetector=(long long)BigDetSize*BigDetSize/32+1;continue;}
    if(!strncmp(line,"px ",3)){sscanf(line,"%s %f",dummy,&pixelsize);continue;}
    if(!strncmp(line,"SpaceGroup ",11)){sscanf(line,"%s %d",dummy,&P->SpaceGroupNum);SGNum=P->SpaceGroupNum;continue;}
    if(!strncmp(line,"LatticeParameter ",17)||!strncmp(line,"LatticeConstant ",16)){
      sscanf(line,"%s %f",dummy,&P->LatticeConstant);
      sscanf(line,"%s %lf %lf %lf %lf %lf %lf",dummy,&ABCABG[0],&ABCABG[1],&ABCABG[2],&ABCABG[3],&ABCABG[4],&ABCABG[5]);continue;}
    if(!strncmp(line,"Wavelength ",11)){sscanf(line,"%s %f",dummy,&P->Wavelength);continue;}
    if(!strncmp(line,"Distance ",9)||!strncmp(line,"Lsd ",4)){sscanf(line,"%s %f",dummy,&P->Distance);continue;}
    if(!strncmp(line,"Rsample ",8)){sscanf(line,"%s %f",dummy,&P->Rsample);continue;}
    if(!strncmp(line,"Hbeam ",6)){sscanf(line,"%s %f",dummy,&P->Hbeam);continue;}
    if(!strncmp(line,"StepsizePos ",12)){sscanf(line,"%s %f",dummy,&P->StepsizePos);continue;}
    if(!strncmp(line,"StepsizeOrient ",15)||!strncmp(line,"StepSizeOrient ",15)){sscanf(line,"%s %f",dummy,&P->StepsizeOrient);continue;}
    if(!strncmp(line,"MarginOme ",10)){sscanf(line,"%s %f",dummy,&P->MarginOme);continue;}
    if(!strncmp(line,"MarginRadius ",13)){sscanf(line,"%s %f",dummy,&P->MarginRad);continue;}
    if(!strncmp(line,"MarginRadial ",13)){sscanf(line,"%s %f",dummy,&P->MarginRadial);continue;}
    if(!strncmp(line,"EtaBinSize ",11)){sscanf(line,"%s %f",dummy,&P->EtaBinSize);continue;}
    if(!strncmp(line,"OmeBinSize ",11)){sscanf(line,"%s %f",dummy,&P->OmeBinSize);continue;}
    if(!strncmp(line,"MinMatchesToAcceptFrac ",22)||!strncmp(line,"Completeness ",13)){sscanf(line,"%s %f",dummy,&P->MinMatchesToAcceptFrac);continue;}
    if(!strncmp(line,"ExcludePoleAngle ",17)||!strncmp(line,"MinEta ",7)){sscanf(line,"%s %f",dummy,&P->ExcludePoleAngle);continue;}
    if(!strncmp(line,"RingRadii ",10)){sscanf(line,"%s %f",dummy,&P->RingRadiiUser[P->NrOfRings]);P->NrOfRings++;continue;}
    if(!strncmp(line,"OmegaRange ",11)){sscanf(line,"%s %f %f",dummy,&P->OmegaRanges[P->NoOfOmegaRanges][0],&P->OmegaRanges[P->NoOfOmegaRanges][1]);P->NoOfOmegaRanges++;continue;}
    if(!strncmp(line,"BoxSize ",8)){sscanf(line,"%s %f %f %f %f",dummy,&P->BoxSizes[NrOfBoxSizes][0],&P->BoxSizes[NrOfBoxSizes][1],&P->BoxSizes[NrOfBoxSizes][2],&P->BoxSizes[NrOfBoxSizes][3]);NrOfBoxSizes++;continue;}
    if(!strncmp(line,"SpotsFileName ",14)){sscanf(line,"%s %s",dummy,P->SpotsFileName);continue;}
    if(!strncmp(line,"GrainsFile ",11)){P->isGrainsInput=1;sscanf(line,"%s %s",dummy,P->GrainsFileName);continue;}
    if(!strncmp(line,"IDsFileName ",12)){sscanf(line,"%s %s",dummy,P->IDsFileName);continue;}
    if(!strncmp(line,"MarginEta ",10)){sscanf(line,"%s %f",dummy,&P->MarginEta);continue;}
    if(!strncmp(line,"UseFriedelPairs ",16)){sscanf(line,"%s %d",dummy,&P->UseFriedelPairs);continue;}
    if(!strncmp(line,"OutputFolder ",13)){sscanf(line,"%s %s",dummy,P->OutputFolder);continue;}
  }
  fclose(fp);
  if(totNrPixelsBigDetector) ReadBigDet(dirname(P->OutputFolder));
  for(int i=0;i<MAX_N_RINGS;i++) P->RingRadii[i]=0;
  for(int i=0;i<P->NrOfRings;i++) P->RingRadii[P->RingNumbers[i]]=P->RingRadiiUser[i];
  return 0;
}

static int ReadSpots(char *cwd) {
  char fn[2048]; sprintf(fn,"%s/Spots.bin",cwd);
  int fd=open(fn,O_RDONLY); check(fd<0,"open %s: %s",fn,strerror(errno));
  struct stat s; fstat(fd,&s);
  ObsSpotsLab_d=(double*)mmap(0,s.st_size,PROT_READ,MAP_SHARED,fd,0);
  check(ObsSpotsLab_d==MAP_FAILED,"mmap %s: %s",fn,strerror(errno));
  int nsp = (int)(s.st_size/(N_COL_OBSSPOTS*sizeof(double)));
  // Convert double → float for CPU lookups and GPU upload
  ObsSpotsLab = (float*)malloc(nsp * N_COL_OBSSPOTS * sizeof(float));
  check(!ObsSpotsLab, "malloc ObsSpotsLab float conversion");
  for (int i = 0; i < nsp * N_COL_OBSSPOTS; i++)
    ObsSpotsLab[i] = (float)ObsSpotsLab_d[i];
  printf("Spots.bin: %d spots (converted double→float)\n", nsp);
  return nsp;
}

static void ReadBins(char *cwd) {
  char fn1[2048]; sprintf(fn1,"%s/Data.bin",cwd);
  int fd1=open(fn1,O_RDONLY); check(fd1<0,"open %s: %s",fn1,strerror(errno));
  struct stat s1; fstat(fd1,&s1);
  data=(int*)mmap(0,s1.st_size,PROT_READ,MAP_SHARED,fd1,0);
  check(data==MAP_FAILED,"mmap %s",fn1);

  char fn2[2048]; sprintf(fn2,"%s/nData.bin",cwd);
  int fd2=open(fn2,O_RDONLY); check(fd2<0,"open %s: %s",fn2,strerror(errno));
  struct stat s2; fstat(fd2,&s2);
  ndata=(int*)mmap(0,s2.st_size,PROT_READ,MAP_SHARED,fd2,0);
  check(ndata==MAP_FAILED,"mmap %s",fn2);
  printf("Data.bin read. nData.bin read.\n");
}

// ─── Generate ideal spot positions (simplified Friedel-free) ──
static void h_GenerateIdealSpots(float ys, float zs, float ttheta, float eta,
    float Ring_rad, float Rsample, float Hbeam, float step_size,
    float y0v[], float z0v[], int *nSteps) {
  // Simplified version: generate y0 along the illuminated arc
  int qc2=0; float eh,qc,cy=0,cz=0,ymax_z0,ymin_z0,ymax=0,ymin=0,zmin=0,zmax=0;
  if(eta>90) eh=180-eta; else if(eta<-90) eh=180-fabsf(eta); else eh=90-fabsf(eta);
  Hbeam += 2*(Rsample*tanf(ttheta*deg2rad))*sinf(eh*deg2rad);
  float epole=1+rad2deg*acosf(1-Hbeam/Ring_rad);
  float eeq=1+rad2deg*acosf(1-Rsample/Ring_rad);
  if(eta>=epole&&eta<=(90-eeq)){qc=1;cy=-1;cz=1;}
  else if(eta>=(90+eeq)&&eta<=(180-epole)){qc=2;cy=-1;cz=-1;}
  else if(eta>=(-90+eeq)&&eta<=-epole){qc=2;cy=1;cz=1;}
  else if(eta>=(-180+epole)&&eta<=(-90-eeq)){qc=1;cy=1;cz=-1;}
  else qc=0;
  float ymaxR=ys+Rsample, yminR=ys-Rsample;
  float zmaxH=zs+0.5f*Hbeam, zminH=zs-0.5f*Hbeam;
  if(qc==1){ymax_z0=cy*sqrtf(Ring_rad*Ring_rad-zmaxH*zmaxH);ymin_z0=cy*sqrtf(Ring_rad*Ring_rad-zminH*zminH);}
  else if(qc==2){ymax_z0=cy*sqrtf(Ring_rad*Ring_rad-zminH*zminH);ymin_z0=cy*sqrtf(Ring_rad*Ring_rad-zmaxH*zmaxH);}
  if(qc>0){ymax=fminf(ymaxR,ymax_z0);ymin=fmaxf(yminR,ymin_z0);}
  else {
    if(eta>-epole&&eta<epole){ymax=ymaxR;ymin=yminR;cz=1;}
    else if(eta<(-180+epole)){ymax=ymaxR;ymin=yminR;cz=-1;}
    else if(eta>(180-epole)){ymax=ymaxR;ymin=yminR;cz=-1;}
    else if(eta>(90-eeq)&&eta<(90+eeq)){qc2=1;zmax=zmaxH;zmin=zminH;cy=-1;}
    else if(eta>(-90-eeq)&&eta<(-90+eeq)){qc2=1;zmax=zmaxH;zmin=zminH;cy=1;}
  }
  float y1,z1,y2,z2; int ns;
  if(!qc2){y1=ymin;z1=cz*sqrtf(Ring_rad*Ring_rad-y1*y1);y2=ymax;z2=cz*sqrtf(Ring_rad*Ring_rad-y2*y2);}
  else{z1=zmin;y1=cy*sqrtf(Ring_rad*Ring_rad-z1*z1);z2=zmax;y2=cy*sqrtf(Ring_rad*Ring_rad-z2*z2);}
  float yd=y1-y2,zd=z1-z2; float len=sqrtf(yd*yd+zd*zd);
  ns=(int)ceilf(len/step_size); if(ns%2==0) ns++; if(ns<1) ns=1;
  if(ns==1){
    if(!qc2){y0v[0]=(ymax+ymin)/2;z0v[0]=cz*sqrtf(Ring_rad*Ring_rad-y0v[0]*y0v[0]);}
    else{z0v[0]=(zmax+zmin)/2;y0v[0]=cy*sqrtf(Ring_rad*Ring_rad-z0v[0]*z0v[0]);}
  } else {
    float sy=(ymax-ymin)/(ns-1), sz=(zmax-zmin)/(ns-1);
    for(int i=0;i<ns;i++){
      if(!qc2){y0v[i]=ymin+i*sy;z0v[i]=cz*sqrtf(Ring_rad*Ring_rad-y0v[i]*y0v[i]);}
      else{z0v[i]=zmin+i*sz;y0v[i]=cy*sqrtf(Ring_rad*Ring_rad-z0v[i]*z0v[i]);}
    }
  }
  *nSteps=ns;
}

// ═════════════════════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tIndexerGPU (%s)\n"
         "GPU-accelerated FF-HEDM Indexer.\n"
         "Contact hsharma@anl.gov for questions.\n\n", MIDAS_VERSION_STRING);

  if (argc != 6) {
    printf("Usage: %s param.txt blockNr nBlocks nSpotsToIndex numProcs\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // 1. Read parameters
  struct TParams Params;
  if (ReadParams(argv[1], &Params)) { printf("Error reading params\n"); exit(1); }
  printf("SpaceGroup: %d, Distance: %f, NrOfRings: %d\n",
         Params.SpaceGroupNum, Params.Distance, Params.NrOfRings);

  // 2. Read HKLs
  FILE *hklf = fopen("hkls.csv","r");
  if (!hklf) { printf("Cannot open hkls.csv\n"); exit(1); }
  char aline[4096];
  fgets(aline,1000,hklf);
  int hi,ki,li,Rnr; float hc,kc,lc,RRd,Ds,tht,tth;
  while (fgets(aline,1000,hklf)) {
    sscanf(aline,"%d %d %d %f %d %f %f %f %f %f %f",
           &hi,&ki,&li,&Ds,&Rnr,&hc,&kc,&lc,&tht,&tth,&RRd);
    RingHKL[Rnr][0]=hc; RingHKL[Rnr][1]=kc; RingHKL[Rnr][2]=lc;
    RingTtheta[Rnr]=tth;
    for (int i=0;i<Params.NrOfRings;i++) {
      if (Rnr==Params.RingNumbers[i]) {
        HKLints[n_hkls][0]=hi; HKLints[n_hkls][1]=ki;
        HKLints[n_hkls][2]=li; HKLints[n_hkls][3]=Rnr;
        hkls[n_hkls][0]=hc; hkls[n_hkls][1]=kc; hkls[n_hkls][2]=lc;
        hkls[n_hkls][3]=(float)Rnr; hkls[n_hkls][4]=Ds;
        hkls[n_hkls][5]=tht; hkls[n_hkls][6]=RRd;
        n_hkls++; break;
      }
    }
  }
  fclose(hklf);
  printf("HKLs: %d\n", n_hkls);

  // 3. Read observed spots and bins
  char tmpstr[2048]; sprintf(tmpstr,"%s",Params.OutputFolder);
  char *cwdstr = dirname(tmpstr);
  n_spots = ReadSpots(cwdstr);
  printf("Spots: %zu\n", n_spots);
  ReadBins(cwdstr);

  int HighestRingNo = 0;
  for (int i=0;i<MAX_N_RINGS;i++) if(Params.RingRadii[i]) HighestRingNo=i;
  n_ring_bins = HighestRingNo;
  n_eta_bins = (int)ceilf(360.0f / Params.EtaBinSize);
  n_ome_bins = (int)ceilf(360.0f / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize; OmeBinSize = Params.OmeBinSize;
  printf("Bins: ring=%d eta=%d ome=%d\n", n_ring_bins, n_eta_bins, n_ome_bins);

  // 4. Precompute margins
  float omemargins[181], etamargins[MAX_N_RINGS];
  for (int i=1;i<180;i++) omemargins[i]=Params.MarginOme+(0.5f*Params.StepsizeOrient/fabsf(sinf(i*deg2rad)));
  omemargins[0]=omemargins[1]; omemargins[180]=omemargins[1];
  for (int i=0;i<MAX_N_RINGS;i++) {
    if (Params.RingRadii[i]==0) etamargins[i]=0;
    else etamargins[i]=rad2deg*atanf(Params.MarginEta/Params.RingRadii[i])+0.5f*Params.StepsizeOrient;
  }

  // 5. Open output files
  int blockNr = atoi(argv[2]);
  int nSpotsToIndex = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  int openFlags = O_CREAT|O_WRONLY|(blockNr==0?O_TRUNC:0);
  char outFN[4096];
  sprintf(outFN,"%s/IndexBest.bin",Params.OutputFolder);
  Params.IndexBestFD = open(outFN, openFlags, S_IRUSR|S_IWUSR);
  sprintf(outFN,"%s/IndexBestFull.bin",Params.OutputFolder);
  Params.IndexBestFullFD = open(outFN, openFlags, S_IRUSR|S_IWUSR);
  if (blockNr==0) {
    ftruncate(Params.IndexBestFD, (off_t)nSpotsToIndex*15*sizeof(double));
    ftruncate(Params.IndexBestFullFD, (off_t)nSpotsToIndex*MAX_N_HKLS*2*sizeof(double));
  }

  // 6. Read spot IDs
  int nBlocks = atoi(argv[3]);
  int startRow = (int)(ceilf((float)nSpotsToIndex/nBlocks))*blockNr;
  int tmp = (int)(ceilf((float)nSpotsToIndex/nBlocks))*(blockNr+1);
  int endRow = tmp < (nSpotsToIndex-1) ? tmp : (nSpotsToIndex-1);
  int nSpotIDs = endRow - startRow + 1;
  int *SpotIDs = (int*)malloc(nSpotIDs*sizeof(int));
  for(int i=0;i<nSpotIDs;i++) SpotIDs[i]=-1;

  FILE *sf = fopen("SpotsToIndex.csv","r");
  if (!sf) { printf("Cannot open SpotsToIndex.csv\n"); exit(1); }
  for (int i=0;i<startRow;i++) fgets(aline,1000,sf);
  for (int i=0;i<nSpotIDs;i++) { if(fgets(aline,1000,sf)) sscanf(aline,"%d",&SpotIDs[i]); }
  fclose(sf);
  printf("SpotIDs to index: %d\n", nSpotIDs);

  // ═══════════════════════════════════════════════════════════
  //  GPU SETUP
  // ═══════════════════════════════════════════════════════════
  int deviceId = 0;
  CUDA_CHECK(cudaSetDevice(deviceId));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  printf("GPU: %s (%.0f MB)\n", prop.name, prop.totalGlobalMem/1048576.0);

  // Upload constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(c_RingRadii, Params.RingRadii, sizeof(Params.RingRadii)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_OmegaRanges, Params.OmegaRanges, sizeof(Params.OmegaRanges)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_BoxSizes, Params.BoxSizes, sizeof(Params.BoxSizes)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_omemargins, omemargins, sizeof(omemargins)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_etamargins, etamargins, sizeof(etamargins)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_ringsToReject, Params.RingsToReject, sizeof(Params.RingsToReject)));

  GPUParams gp;
  gp.Distance = Params.Distance;
  gp.Wavelength = Params.Wavelength;
  gp.ExcludePoleAngle = Params.ExcludePoleAngle;
  gp.MarginRad = Params.MarginRad;
  gp.MarginRadial = Params.MarginRadial;
  gp.MarginOme = Params.MarginOme;
  gp.NoOfOmegaRanges = Params.NoOfOmegaRanges;
  gp.nRingsToRejectCalc = Params.nRingsToRejectCalc;
  gp.n_ring_bins = n_ring_bins;
  gp.n_eta_bins = n_eta_bins;
  gp.n_ome_bins = n_ome_bins;
  gp.EtaBinSize = EtaBinSize;
  gp.OmeBinSize = OmeBinSize;
  CUDA_CHECK(cudaMemcpyToSymbol(c_params, &gp, sizeof(GPUParams)));

  // Upload persistent data to GPU
  float *d_hkls_flat;
  CUDA_CHECK(cudaMalloc(&d_hkls_flat, n_hkls*7*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_hkls_flat, hkls, n_hkls*7*sizeof(float), cudaMemcpyHostToDevice));

  float *d_ObsSpotsLab;
  CUDA_CHECK(cudaMalloc(&d_ObsSpotsLab, n_spots*N_COL_OBSSPOTS*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_ObsSpotsLab, ObsSpotsLab, n_spots*N_COL_OBSSPOTS*sizeof(float), cudaMemcpyHostToDevice));

  int *d_data, *d_ndata;
  // Determine data sizes from file sizes
  {
    char fn1[2048]; sprintf(fn1,"%s/Data.bin",cwdstr);
    struct stat s1; stat(fn1,&s1);
    CUDA_CHECK(cudaMalloc(&d_data, s1.st_size));
    CUDA_CHECK(cudaMemcpy(d_data, data, s1.st_size, cudaMemcpyHostToDevice));
    char fn2[2048]; sprintf(fn2,"%s/nData.bin",cwdstr);
    struct stat s2; stat(fn2,&s2);
    CUDA_CHECK(cudaMalloc(&d_ndata, s2.st_size));
    CUDA_CHECK(cudaMemcpy(d_ndata, ndata, s2.st_size, cudaMemcpyHostToDevice));
  }

  int *d_ringsToReject;
  CUDA_CHECK(cudaMalloc(&d_ringsToReject, MAX_N_RINGS*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ringsToReject, Params.RingsToReject, MAX_N_RINGS*sizeof(int), cudaMemcpyHostToDevice));

  // Results array (one per spotID)
  SpotResult *d_results;
  CUDA_CHECK(cudaMalloc(&d_results, nSpotIDs*sizeof(SpotResult)));

  SpotResult *h_results = (SpotResult*)calloc(nSpotIDs, sizeof(SpotResult));
  for (int i=0;i<nSpotIDs;i++) {
    h_results[i].atomicKey = 0ULL;
    h_results[i].bestFrac = -1.0f;
    h_results[i].bestIA = 999.0f;
  }
  CUDA_CHECK(cudaMemcpy(d_results, h_results, nSpotIDs*sizeof(SpotResult), cudaMemcpyHostToDevice));

  printf("GPU data uploaded. Starting tuple generation...\n");

  // ═══════════════════════════════════════════════════════════
  //  CPU: Generate all evaluation tuples (OMP parallel)
  // ═══════════════════════════════════════════════════════════
  double tuple_start = omp_get_wtime();

  // Phase 1: Count tuples per spotID (parallel)
  int *tupleCounts = (int*)calloc(nSpotIDs, sizeof(int));

  // Pre-allocate per-thread OrTmp buffers (MAX_N_OR*9 floats = 1.3 MB each)
  float **OrTmp_all = (float**)malloc(numProcs * sizeof(float*));
  for (int t = 0; t < numProcs; t++)
    OrTmp_all[t] = (float*)malloc(MAX_N_OR * 9 * sizeof(float));

  #pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int si = 0; si < nSpotIDs; si++) {
    if (SpotIDs[si] == -1) continue;
    int spotID = SpotIDs[si];
    // Find spot row
    int SpotRowNo = -1;
    for (int r=0; r<(int)n_spots; r++) {
      if ((int)ObsSpotsLab[r*N_COL_OBSSPOTS+4] == spotID) { SpotRowNo=r; break; }
    }
    if (SpotRowNo < 0) continue;

    float ys = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+0];
    float zs = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+1];
    float omega = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+2];
    float eta = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+6];
    int ringnr = (int)ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+5];

    // Generate plane normals
    float y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
    int nPN;
    h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                         Params.RingRadii[ringnr], Params.Rsample,
                         Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);

    int count = 0;
    for (int isp=0; isp<nPN; isp++) {
      float xi,yi,zi;
      h_MakeUnitLength(Params.Distance, y0v[isp], z0v[isp], &xi, &yi, &zi);
      float g1,g2,g3;
      h_spot_to_gv(xi,yi,zi,omega,&g1,&g2,&g3);
      float hn[3]={g1,g2,g3};
      double hkl_d[3]={RingHKL[ringnr][0],RingHKL[ringnr][1],RingHKL[ringnr][2]};
      int nOr;
      float *OrTmp = OrTmp_all[omp_get_thread_num()];
      h_GenerateCandidateOrientations(hkl_d, hn, Params.StepsizeOrient, OrTmp, &nOr, ringnr);

      for (int or_=0; or_<nOr; or_++) {
        int nmax,nmin;
        h_calc_n_max_min(xi,yi,ys,y0v[isp],Params.Rsample,Params.StepsizePos,&nmax,&nmin);
        for (int n=nmin; n<=nmax; n++) {
          float ga,gb,gc;
          h_spot_to_unrotated(xi,yi,zi,ys,zs,y0v[isp],z0v[isp],
                              Params.StepsizePos,n,omega,&ga,&gb,&gc);
          if (fabsf(gc) > Params.Hbeam/2) continue;
          count++;
        }
      }
    }
    tupleCounts[si] = count;
  }

  // Prefix sum for offsets
  long long totalTuples = 0;
  int *tupleOffsets = (int*)malloc(nSpotIDs*sizeof(int));
  for (int i=0;i<nSpotIDs;i++) { tupleOffsets[i]=(int)totalTuples; totalTuples+=tupleCounts[i]; }

  printf("Total evaluation tuples: %lld (%.1f avg per spot). Count time: %.2fs\n",
         totalTuples, (double)totalTuples/nSpotIDs, omp_get_wtime()-tuple_start);

  int nGood = 0;

  if (totalTuples == 0) {
    printf("No tuples to evaluate. Exiting.\n");
    goto cleanup;
  }

  // Phase 2: Fill tuples (parallel)
  {
    EvalTuple *h_tuples = (EvalTuple*)malloc(totalTuples * sizeof(EvalTuple));
    if (!h_tuples) { printf("Cannot allocate %lld tuples (%.1f GB)\n",
                            totalTuples, totalTuples*sizeof(EvalTuple)/1e9); exit(1); }

    #pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int si = 0; si < nSpotIDs; si++) {
      if (SpotIDs[si] == -1 || tupleCounts[si] == 0) continue;
      int spotID = SpotIDs[si];
      int SpotRowNo = -1;
      for (int r=0; r<(int)n_spots; r++) {
        if ((int)ObsSpotsLab[r*N_COL_OBSSPOTS+4] == spotID) { SpotRowNo=r; break; }
      }
      if (SpotRowNo < 0) continue;

      float ys = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+0];
      float zs = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+1];
      float omega = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+2];
      float RefRad = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+3];
      float eta = ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+6];
      int ringnr = (int)ObsSpotsLab[SpotRowNo*N_COL_OBSSPOTS+5];

      float y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
      int nPN;
      h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                           Params.RingRadii[ringnr], Params.Rsample,
                           Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);
      int idx = tupleOffsets[si];
      for (int isp=0; isp<nPN; isp++) {
        float xi,yi,zi;
        h_MakeUnitLength(Params.Distance, y0v[isp], z0v[isp], &xi, &yi, &zi);
        float g1,g2,g3;
        h_spot_to_gv(xi,yi,zi,omega,&g1,&g2,&g3);
        float hn[3]={g1,g2,g3};
        double hkl_d[3]={RingHKL[ringnr][0],RingHKL[ringnr][1],RingHKL[ringnr][2]};
        int nOr; float *OrTmp = OrTmp_all[omp_get_thread_num()];
        h_GenerateCandidateOrientations(hkl_d, hn, Params.StepsizeOrient, OrTmp, &nOr, ringnr);

        for (int or_=0; or_<nOr; or_++) {
          int nmax,nmin;
          h_calc_n_max_min(xi,yi,ys,y0v[isp],Params.Rsample,Params.StepsizePos,&nmax,&nmin);
          for (int n=nmin; n<=nmax; n++) {
            float ga,gb,gc;
            h_spot_to_unrotated(xi,yi,zi,ys,zs,y0v[isp],z0v[isp],
                                Params.StepsizePos,n,omega,&ga,&gb,&gc);
            if (fabsf(gc) > Params.Hbeam/2) continue;
            EvalTuple *t = &h_tuples[idx++];
            t->spotIdx = si;
            for(int k=0;k<9;k++) t->OrMat[k] = OrTmp[or_*9+k];
            t->ga = ga; t->gb = gb; t->gc = gc;
            t->RefRad = RefRad;
          }
        }
      }
    }

    printf("Tuples filled. Uploading to GPU...\n");

    // ═════════════════════════════════════════════════════════
    //  GPU DISPATCH: Upload tuples and launch kernel
    // ═════════════════════════════════════════════════════════

    // Compute dynamic scratch size: each HKL can produce at most 2 omega solutions
    int maxTheorSpots = n_hkls * 2;
    size_t scratchPerThread = (size_t)maxTheorSpots * N_COL_THEORSPOTS * sizeof(float);

    // Process in batches — account for both tuples AND scratch memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    size_t bytesPerThread = sizeof(EvalTuple) + scratchPerThread;
    size_t maxBatch = (freeMem / 2) / bytesPerThread;
    if (maxBatch < 1) maxBatch = 1;
    int nBatches = (int)((totalTuples + maxBatch - 1) / maxBatch);
    if (nBatches < 1) nBatches = 1;

    long long batchSize = (totalTuples + nBatches - 1) / nBatches;

    printf("GPU free: %.0f MB, tupleBytes: %.0f MB, scratchPerThread: %zu B, maxTheorSpots: %d, batches: %d, batchSize: %lld\n",
           freeMem/1e6, totalTuples*sizeof(EvalTuple)/1e6, scratchPerThread, maxTheorSpots, nBatches, batchSize);

    EvalTuple *d_tuples;
    CUDA_CHECK(cudaMalloc(&d_tuples, batchSize * sizeof(EvalTuple)));

    float *d_theorScratch;
    CUDA_CHECK(cudaMalloc(&d_theorScratch, batchSize * scratchPerThread));

    int blockSize = 256;

    for (int b=0; b<nBatches; b++) {
      long long bStart = b * batchSize;
      long long bEnd = bStart + batchSize;
      if (bEnd > totalTuples) bEnd = totalTuples;
      long long bN = bEnd - bStart;

      CUDA_CHECK(cudaMemcpy(d_tuples, h_tuples + bStart,
                            bN * sizeof(EvalTuple), cudaMemcpyHostToDevice));

      int nBlks = (int)((bN + blockSize - 1) / blockSize);
      printf("  Batch %d/%d: %lld tuples, %d blocks\n", b+1, nBatches, bN, nBlks);

      indexer_eval_kernel<<<nBlks, blockSize>>>(
        d_tuples, (int)bN,
        d_hkls_flat, n_hkls,
        d_ObsSpotsLab, d_data, d_ndata,
        d_ringsToReject, Params.nRingsToRejectCalc,
        d_results, d_theorScratch, maxTheorSpots);

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
  CUDA_CHECK(cudaMemcpy(h_results, d_results, nSpotIDs*sizeof(SpotResult), cudaMemcpyDeviceToHost));

  for (int si=0; si<nSpotIDs; si++) {
    if (h_results[si].bestFrac <= 0) continue;
    nGood++;

    // Write IndexBest.bin (same format as IndexerOMP)
    double res[15];
    res[0] = (double)h_results[si].bestIA; // IA — matches CPU format (confidence = res[14]/res[13])
    for (int k=0;k<9;k++) res[k+1] = (double)h_results[si].bestOrMat[k];
    res[10]=(double)h_results[si].bestPos[0];
    res[11]=(double)h_results[si].bestPos[1];
    res[12]=(double)h_results[si].bestPos[2];
    res[13]=(double)h_results[si].nTspots;   // nExp
    res[14]=(double)h_results[si].nMatches;  // nObs

    size_t offset = (size_t)(si + startRow) * 15 * sizeof(double);
    pwrite(Params.IndexBestFD, res, 15*sizeof(double), offset);

    printf("IDNr: %d, ID: %d, Confidence: %.4f, nExp: %d, nObs: %d\n",
           si, SpotIDs[si], h_results[si].bestFrac,
           h_results[si].nTspots, h_results[si].nMatches);
  }

  printf("\nGPU Indexer complete: %d/%d spots found matches.\n", nGood, nSpotIDs);

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
  for (int t = 0; t < numProcs; t++) free(OrTmp_all[t]);
  free(OrTmp_all);
  if (ObsSpotsLab) free(ObsSpotsLab);

  close(Params.IndexBestFD);
  close(Params.IndexBestFullFD);

  double elapsed = omp_get_wtime() - start_time;
  printf("Total time: %.2f seconds.\n", elapsed);
  return 0;
}
