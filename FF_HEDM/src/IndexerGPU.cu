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
    float *avgIA)                   // output: average internal angle
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
      if (ia < 999.0f) {
        iaSum += ia;
        iaCount++;
      }
    }
  }
  *nMatchesFracCalc = nMatchedFrac;
  *avgIA = (iaCount > 0) ? (iaSum / (float)iaCount) : 999.0f;
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
  int nMatches = gpu_CompareSpots(
    TheorSpots, nTspots, d_ObsSpotsLab, t.RefRad,
    d_data, d_ndata,
    d_ringsToReject, nRingsToRejectCalc,
    &nMatchesFracCalc,
    t.ga, t.gb, t.gc, &avgIA);

  float fracMatches = (float)nMatchesFracCalc / (float)nTspotsFracCalc;

  if (spotIdx == 0) {
    printf("[GPU CAND] pos=(%.2f,%.2f,%.2f) conf=%.6f nObs=%d nExp=%d IA=%.6f\n",
           t.ga, t.gb, t.gc, fracMatches, nMatchesFracCalc, nTspotsFracCalc, avgIA);
  }

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

// ─── Generate ideal spot positions ──────────────────────────

// Forward declaration (needed by h_GenerateIdealSpotsFriedelMixed)
static void h_GenerateIdealSpots(float ys, float zs, float ttheta, float eta,
    float Ring_rad, float Rsample, float Hbeam, float step_size,
    float y0v[], float z0v[], int *nSteps);

static inline void h_CalcSpotPosition(float RingRadius, float eta, float *yl, float *zl) {
  float etaRad = deg2rad * eta;
  *yl = -(sinf(etaRad) * RingRadius);
  *zl = cosf(etaRad) * RingRadius;
}

static inline void h_RotateAroundZ(float v1[3], float alpha, float v2[3]) {
  float cosa = cosf(alpha * deg2rad);
  float sina = sinf(alpha * deg2rad);
  v2[0] = cosa*v1[0] - sina*v1[1];
  v2[1] = sina*v1[0] + cosa*v1[1];
  v2[2] = v1[2];
}

static void h_CalcOmega(float x, float y, float z, float theta,
    float omegas[4], float etas[4], int *nsol) {
  *nsol = 0;
  float len = sqrtf(x*x + y*y + z*z);
  float v = sinf(theta * deg2rad) * len;
  float almostzero = 1e-4f;
  if (fabsf(y) < almostzero) {
    if (x != 0) {
      float cosome1 = -v / x;
      if (fabsf(cosome1) <= 1.0f) {
        float ome = acosf(cosome1) * rad2deg;
        omegas[(*nsol)++] = ome;
        omegas[(*nsol)++] = -ome;
      }
    }
  } else {
    float y2 = y*y;
    float a = 1 + (x*x)/y2;
    float b = (2*v*x)/y2;
    float c = (v*v)/y2 - 1;
    float discr = b*b - 4*a*c;
    if (discr >= 0) {
      float cosome1 = (-b + sqrtf(discr)) / (2*a);
      if (fabsf(cosome1) <= 1.0f) {
        float ome1a = acosf(cosome1);
        float ome1b = -ome1a;
        float eqa = -x*cosf(ome1a) + y*sinf(ome1a);
        float eqb = -x*cosf(ome1b) + y*sinf(ome1b);
        omegas[(*nsol)++] = (fabsf(eqa-v) < fabsf(eqb-v) ? ome1a : ome1b) * rad2deg;
      }
      float cosome2 = (-b - sqrtf(discr)) / (2*a);
      if (fabsf(cosome2) <= 1.0f) {
        float ome2a = acosf(cosome2);
        float ome2b = -ome2a;
        float eqa = -x*cosf(ome2a) + y*sinf(ome2a);
        float eqb = -x*cosf(ome2b) + y*sinf(ome2b);
        omegas[(*nsol)++] = (fabsf(eqa-v) < fabsf(eqb-v) ? ome2a : ome2b) * rad2deg;
      }
    }
  }
  float gw[3], gv[3] = {x, y, z};
  for (int i = 0; i < *nsol; i++) {
    h_RotateAroundZ(gv, omegas[i], gw);
    h_CalcEtaAngle(gw[1], gw[2], &etas[i]);
  }
}

static inline void h_displacement_spot_needed_COM(float a, float b, float c,
    float xi, float yi, float zi, float omega, float *Displ_y, float *Displ_z) {
  float lenInv = 1.0f / sqrtf(xi*xi + yi*yi + zi*zi);
  xi *= lenInv; yi *= lenInv; zi *= lenInv;
  float OmegaRad = deg2rad * omega;
  float sinOme = sinf(OmegaRad), cosOme = cosf(OmegaRad);
  float t = (a*cosOme - b*sinOme) / xi;
  *Displ_y = (a*sinOme + b*cosOme) - t*yi;
  *Displ_z = c - t*zi;
}

static void h_FriedelEtaCalculation(float ys, float zs, float ttheta,
    float eta, float Ring_rad, float Rsample, float Hbeam,
    float *EtaMinFr, float *EtaMaxFr) {
  float quadr_coeff2 = 0;
  float eta_Hbeam, quadr_coeff, coeff_y0=0, coeff_z0=0, y0_max_z0=0, y0_min_z0=0;
  float y0_max=0, y0_min=0, z0_min=0, z0_max=0;
  if (eta > 90)       eta_Hbeam = 180 - eta;
  else if (eta < -90) eta_Hbeam = 180 - fabsf(eta);
  else                eta_Hbeam = 90 - fabsf(eta);
  Hbeam = Hbeam + 2*(Rsample*tanf(ttheta*deg2rad))*sinf(eta_Hbeam*deg2rad);
  float eta_pole = 1 + rad2deg*acosf(1 - Hbeam/Ring_rad);
  float eta_equator = 1 + rad2deg*acosf(1 - Rsample/Ring_rad);
  if (eta>=eta_pole && eta<=(90-eta_equator))           { quadr_coeff=1; coeff_y0=-1; coeff_z0=1; }
  else if (eta>=(90+eta_equator) && eta<=(180-eta_pole)) { quadr_coeff=2; coeff_y0=-1; coeff_z0=-1; }
  else if (eta>=(-90+eta_equator) && eta<=-eta_pole)     { quadr_coeff=2; coeff_y0=1;  coeff_z0=1; }
  else if (eta>=(-180+eta_pole) && eta<=(-90-eta_equator)) { quadr_coeff=1; coeff_y0=1; coeff_z0=-1; }
  else quadr_coeff = 0;
  float y0_max_R = ys + Rsample, y0_min_R = ys - Rsample;
  float z0_max_H = zs + 0.5f*Hbeam, z0_min_H = zs - 0.5f*Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_max_H*z0_max_H);
    y0_min_z0 = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_min_H*z0_min_H);
  } else if (quadr_coeff == 2) {
    y0_max_z0 = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_min_H*z0_min_H);
    y0_min_z0 = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_max_H*z0_max_H);
  }
  if (quadr_coeff > 0) {
    y0_max = fminf(y0_max_R, y0_max_z0);
    y0_min = fmaxf(y0_min_R, y0_min_z0);
  } else {
    if (eta > -eta_pole && eta < eta_pole)              { y0_max=y0_max_R; y0_min=y0_min_R; coeff_z0=1; }
    else if (eta < (-180+eta_pole))                     { y0_max=y0_max_R; y0_min=y0_min_R; coeff_z0=-1; }
    else if (eta > (180-eta_pole))                      { y0_max=y0_max_R; y0_min=y0_min_R; coeff_z0=-1; }
    else if (eta>(90-eta_equator) && eta<(90+eta_equator))   { quadr_coeff2=1; z0_max=z0_max_H; z0_min=z0_min_H; coeff_y0=-1; }
    else if (eta>(-90-eta_equator) && eta<(-90+eta_equator)) { quadr_coeff2=1; z0_max=z0_max_H; z0_min=z0_min_H; coeff_y0=1; }
  }
  if (quadr_coeff2 == 0) {
    z0_min = coeff_z0*sqrtf(Ring_rad*Ring_rad - y0_min*y0_min);
    z0_max = coeff_z0*sqrtf(Ring_rad*Ring_rad - y0_max*y0_max);
  } else {
    y0_min = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_min*z0_min);
    y0_max = coeff_y0*sqrtf(Ring_rad*Ring_rad - z0_max*z0_max);
  }
  float dYMin = ys-y0_min, dYMax = ys-y0_max, dZMin = zs-z0_min, dZMax = zs-z0_max;
  float YMinFr = y0_min - dYMin, YMaxFr = y0_max - dYMax;
  float ZMinFr = -z0_min + dZMin, ZMaxFr = -z0_max + dZMax;
  float Eta1, Eta2;
  h_CalcEtaAngle(YMinFr+ys, ZMinFr-zs, &Eta1);
  h_CalcEtaAngle(YMaxFr+ys, ZMaxFr-zs, &Eta2);
  *EtaMinFr = fminf(Eta1, Eta2);
  *EtaMaxFr = fmaxf(Eta1, Eta2);
}

static void h_GenerateIdealSpotsFriedel(float ys, float zs, float ttheta,
    float eta, float omega, int ringno, float Ring_rad, float Rsample,
    float Hbeam, float OmeTol, float RadiusTol,
    float y0v[], float z0v[], int *nSteps) {
  *nSteps = 0;
  float OmeF = (omega < 0) ? omega + 180 : omega - 180;
  float EtaF = (eta < 0) ? -180 - eta : 180 - eta;
  for (int r = 0; r < (int)n_spots; r++) {
    int rno_obs = (int)roundf(ObsSpotsLab[r*N_COL_OBSSPOTS+5]);
    float ome_obs = ObsSpotsLab[r*N_COL_OBSSPOTS+2];
    if (rno_obs != ringno) continue;
    if (fabsf(ome_obs - OmeF) > OmeTol) continue;
    float yf = ObsSpotsLab[r*N_COL_OBSSPOTS+0];
    float zf = ObsSpotsLab[r*N_COL_OBSSPOTS+1];
    float EtaTransf;
    h_CalcEtaAngle(yf+ys, zf-zs, &EtaTransf);
    float radius = sqrtf((yf+ys)*(yf+ys) + (zf-zs)*(zf-zs));
    if (fabsf(radius - 2*Ring_rad) > RadiusTol) continue;
    float EtaMinF, EtaMaxF;
    h_FriedelEtaCalculation(ys, zs, ttheta, eta, Ring_rad, Rsample, Hbeam, &EtaMinF, &EtaMaxF);
    if (EtaTransf < EtaMinF || EtaTransf > EtaMaxF) continue;
    float ZPosAccZ = zs - (zf+zs)/2;
    float YPosAccY = ys - (-yf+ys)/2;
    float etaIdealF;
    h_CalcEtaAngle(YPosAccY, ZPosAccZ, &etaIdealF);
    float IdealYPos, IdealZPos;
    h_CalcSpotPosition(Ring_rad, etaIdealF, &IdealYPos, &IdealZPos);
    y0v[*nSteps] = IdealYPos;
    z0v[*nSteps] = IdealZPos;
    (*nSteps)++;
  }
}

static int h_AddUnique(int *arr, int *n, int val) {
  for (int i = 0; i < *n; i++) if (arr[i] == val) return 0;
  arr[*n] = val;
  (*n)++;
  return 1;
}

static void h_GenerateIdealSpotsFriedelMixed(
    float ys, float zs, float Ttheta, float Eta, float Omega,
    int RingNr, float Ring_rad, float Lsd, float Rsample,
    float Hbeam, float StepSizePos, float OmeTol, float RadialTol,
    float EtaBinSz, float OmeBinSz,
    float EtaTol, float spots_y[], float spots_z[], int *nSteps) {
  const int MinEtaReject = 10;
  float theta = Ttheta / 2;
  float SinMinEtaReject = sinf(MinEtaReject * deg2rad);
  *nSteps = 0;
  if (fabsf(sinf(Eta * deg2rad)) < SinMinEtaReject) return;

  float y0_vector[2000], z0_vector[2000];
  int NoOfSpots;
  h_GenerateIdealSpots(ys, zs, Ttheta, Eta, Ring_rad, Rsample, Hbeam,
                       StepSizePos, y0_vector, z0_vector, &NoOfSpots);

  float FPCandidates[2000][3];
  int FPCandidatesUnique[2000];
  int nFPCandidates = 0;
  float EtaTolDeg = rad2deg * atanf(EtaTol / Ring_rad);

  for (int SpOnRing = 0; SpOnRing < NoOfSpots; SpOnRing++) {
    float y0 = y0_vector[SpOnRing], z0 = z0_vector[SpOnRing];
    float xi, yi, zi;
    h_MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi);
    float G1, G2, G3;
    h_spot_to_gv(xi, yi, zi, Omega, &G1, &G2, &G3);
    float omegasFP[4], etasFP[4]; int nsol;
    h_CalcOmega(-G1, -G2, -G3, theta, omegasFP, etasFP, &nsol);
    if (nsol <= 1) continue;
    float diff0 = fabsf(omegasFP[0] - Omega);
    if (diff0 > 180) diff0 = 360 - diff0;
    float diff1 = fabsf(omegasFP[1] - Omega);
    if (diff1 > 180) diff1 = 360 - diff1;
    float OmegaFP, EtaFP;
    if (diff0 < diff1) { OmegaFP = omegasFP[0]; EtaFP = etasFP[0]; }
    else               { OmegaFP = omegasFP[1]; EtaFP = etasFP[1]; }
    float YFP1, ZFP1;
    h_CalcSpotPosition(Ring_rad, EtaFP, &YFP1, &ZFP1);
    int nMax, nMin;
    h_calc_n_max_min(xi, yi, ys, y0, Rsample, (int)StepSizePos, &nMax, &nMin);
    for (int n = nMin; n <= nMax; n++) {
      float a, b, c;
      h_spot_to_unrotated(xi, yi, zi, ys, zs, y0, z0, StepSizePos, n, Omega, &a, &b, &c);
      if (fabsf(c) > Hbeam/2) continue;
      float Dy, Dz;
      h_displacement_spot_needed_COM(a, b, c, Lsd, YFP1, ZFP1, OmegaFP, &Dy, &Dz);
      float YFP = YFP1 + Dy, ZFP = ZFP1 + Dz;
      float RadialPosFP = sqrtf(YFP*YFP + ZFP*ZFP) - Ring_rad;
      float EtaFPCorr;
      h_CalcEtaAngle(YFP, ZFP, &EtaFPCorr);
      // Bin lookup (inline GetBin)
      int iRing = RingNr - 1;
      int iEta = (int)floorf((180.0f + EtaFPCorr) / EtaBinSz);
      int iOme = (int)floorf((180.0f + OmegaFP) / OmeBinSz);
      if (iEta<0) iEta=0; if (iEta>=n_eta_bins) iEta=n_eta_bins-1;
      if (iOme<0) iOme=0; if (iOme>=n_ome_bins) iOme=n_ome_bins-1;
      size_t Pos = (size_t)iRing*n_eta_bins*n_ome_bins + (size_t)iEta*n_ome_bins + iOme;
      int nInBin = ndata[(int)(Pos*2+0)], DataPos_ = ndata[(int)(Pos*2+1)];
      for (int iSpot = 0; iSpot < nInBin; iSpot++) {
        int spotRow = data[DataPos_ + iSpot];
        int base = spotRow * N_COL_OBSSPOTS;
        if (fabsf(RadialPosFP - ObsSpotsLab[base+8]) >= RadialTol) continue;
        if (fabsf(OmegaFP - ObsSpotsLab[base+2]) >= OmeTol) continue;
        if (fabsf(EtaFPCorr - ObsSpotsLab[base+6]) >= EtaTolDeg) continue;
        float dy = YFP - ObsSpotsLab[base+0], dz = ZFP - ObsSpotsLab[base+1];
        float diffPos2 = dy*dy + dz*dz;
        int idx = nFPCandidates;
        for (int i = 0; i < nFPCandidates; i++) {
          if (FPCandidates[i][0] == ObsSpotsLab[base+4]) {
            idx = (diffPos2 < FPCandidates[i][2]) ? i : -1;
            break;
          }
        }
        if (idx >= 0) {
          FPCandidates[idx][0] = ObsSpotsLab[base+4];
          FPCandidates[idx][1] = (float)SpOnRing;
          FPCandidates[idx][2] = diffPos2;
          if (idx == nFPCandidates) nFPCandidates++;
        }
      }
    }
  }
  int nFPCandidatesUniq = 0;
  for (int i = 0; i < nFPCandidates; i++)
    h_AddUnique(FPCandidatesUnique, &nFPCandidatesUniq, (int)FPCandidates[i][1]);
  for (int i = 0; i < nFPCandidatesUniq; i++) {
    spots_y[i] = y0_vector[FPCandidatesUnique[i]];
    spots_z[i] = z0_vector[FPCandidatesUnique[i]];
  }
  *nSteps = nFPCandidatesUniq;
}

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

    // Generate plane normals — match CPU's Friedel pair logic
    float y0v[MAX_N_STEPS], z0v[MAX_N_STEPS];
    int nPN = 0;
    int usingFriedelPair = 0;
    if (Params.UseFriedelPairs == 1) {
      usingFriedelPair = 1;
      h_GenerateIdealSpotsFriedel(ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
                                  Params.RingRadii[ringnr], Params.Rsample,
                                  Params.Hbeam, Params.MarginOme,
                                  Params.MarginRadial, y0v, z0v, &nPN);
      if (nPN == 0) {
        h_GenerateIdealSpotsFriedelMixed(ys, zs, RingTtheta[ringnr], eta, omega,
                                         ringnr, Params.RingRadii[ringnr],
                                         Params.Distance, Params.Rsample,
                                         Params.Hbeam, Params.StepsizePos,
                                         Params.MarginOme, Params.MarginRadial,
                                         Params.EtaBinSize, Params.OmeBinSize,
                                         Params.MarginEta, y0v, z0v, &nPN);
      }
    }
    if (nPN == 0) {
      if (usingFriedelPair == 1) continue;  // CPU skips spots with no Friedel match
      h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                           Params.RingRadii[ringnr], Params.Rsample,
                           Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);
    }

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
      int nPN = 0;
      int usingFriedelPair = 0;
      if (Params.UseFriedelPairs == 1) {
        usingFriedelPair = 1;
        h_GenerateIdealSpotsFriedel(ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
                                    Params.RingRadii[ringnr], Params.Rsample,
                                    Params.Hbeam, Params.MarginOme,
                                    Params.MarginRadial, y0v, z0v, &nPN);
        if (nPN == 0) {
          h_GenerateIdealSpotsFriedelMixed(ys, zs, RingTtheta[ringnr], eta, omega,
                                           ringnr, Params.RingRadii[ringnr],
                                           Params.Distance, Params.Rsample,
                                           Params.Hbeam, Params.StepsizePos,
                                           Params.MarginOme, Params.MarginRadial,
                                           Params.EtaBinSize, Params.OmeBinSize,
                                           Params.MarginEta, y0v, z0v, &nPN);
        }
      }
      if (nPN == 0) {
        if (usingFriedelPair == 1) continue;
        h_GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                             Params.RingRadii[ringnr], Params.Rsample,
                             Params.Hbeam, Params.StepsizePos, y0v, z0v, &nPN);
      }
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

    // ═══════════════════════════════════════════════════════════
    //  Verify: for spot 0, recompute IA in double precision on host
    //  to compare with GPU float IA
    // ═══════════════════════════════════════════════════════════
    if (nSpotIDs > 0 && tupleCounts[0] > 0) {
      // Download results to get GPU kernel's IA values
      CUDA_CHECK(cudaMemcpy(h_results, d_results, nSpotIDs*sizeof(SpotResult), cudaMemcpyDeviceToHost));

      long long t0start = tupleOffsets[0];
      long long t0end = t0start + tupleCounts[0];
      printf("\n[IA VERIFY] Spot 0: %d tuples, recomputing IA in double precision on host\n", tupleCounts[0]);

      for (long long ti = t0start; ti < t0end; ti++) {
        EvalTuple *t = &h_tuples[ti];
        if (t->spotIdx != 0) continue;
        double OrMat[3][3];
        for (int r=0;r<3;r++) for(int c=0;c<3;c++) OrMat[r][c]=(double)t->OrMat[r*3+c];
        double ga=(double)t->ga, gb=(double)t->gb, gc=(double)t->gc;
        double Dist=(double)Params.Distance;

        // CalcDiffrSpots in double
        int nTsp=0, nTspFrac=0;
        // Allocate local theor spots
        double theorY[MAX_N_HKLS*2], theorZ[MAX_N_HKLS*2], theorOme[MAX_N_HKLS*2];
        int theorRing[MAX_N_HKLS*2];
        int theorMatched[MAX_N_HKLS*2]; // 1 if matched
        int matchedRow[MAX_N_HKLS*2];

        for (int ih=0; ih<n_hkls; ih++) {
          double Ghkl[3]={hkls[ih][0],hkls[ih][1],hkls[ih][2]};
          int ringnr=(int)hkls[ih][3];
          if(ringnr<0||ringnr>=MAX_N_RINGS) continue;
          double RR=Params.RingRadii[ringnr];
          if(RR<1e-9) continue;
          // Check reject
          int reject=0;
          for(int rr=0;rr<Params.nRingsToRejectCalc;rr++) if(Params.RingsToReject[rr]==ringnr){reject=1;break;}
          if(reject) continue;
          double theta=hkls[ih][5];
          double Gc[3]; for(int r=0;r<3;r++) Gc[r]=OrMat[r][0]*Ghkl[0]+OrMat[r][1]*Ghkl[1]+OrMat[r][2]*Ghkl[2];
          double len=sqrt(Gc[0]*Gc[0]+Gc[1]*Gc[1]+Gc[2]*Gc[2]);
          if(len<1e-9) continue;
          double v_=sin(deg2rad*theta)*len;
          double omegas[4],etas[4]; int nsol=0;
          if(fabs(Gc[1])<1e-4){
            if(fabs(Gc[0])>1e-4){double co=-v_/Gc[0]; if(fabs(co)<=1.0){double om=acos(co)*rad2deg; omegas[nsol++]=om; omegas[nsol++]=-om;}}
          } else {
            double y2=Gc[1]*Gc[1],a=1.0+(Gc[0]*Gc[0])/y2,b=(2.0*v_*Gc[0])/y2,cc=(v_*v_)/y2-1.0;
            double D=b*b-4.0*a*cc;
            if(D>=0&&fabs(a)>1e-9){
              double sd=sqrt(D),ta=2.0*a;
              double co1=(-b+sd)/ta; if(fabs(co1)<=1.0){double oa=acos(co1),ob=-oa;double ea=-Gc[0]*cos(oa)+Gc[1]*sin(oa),eb=-Gc[0]*cos(ob)+Gc[1]*sin(ob);omegas[nsol++]=(fabs(ea-v_)<fabs(eb-v_)?oa:ob)*rad2deg;}
              double co2=(-b-sd)/ta; if(fabs(co2)<=1.0){double oa=acos(co2),ob=-oa;double ea=-Gc[0]*cos(oa)+Gc[1]*sin(oa),eb=-Gc[0]*cos(ob)+Gc[1]*sin(ob);omegas[nsol++]=(fabs(ea-v_)<fabs(eb-v_)?oa:ob)*rad2deg;}
            }
          }
          for(int i=0;i<nsol;i++){double cosA=cos(deg2rad*omegas[i]),sinA=sin(deg2rad*omegas[i]);double gw2=sinA*Gc[0]+cosA*Gc[1],gw3=Gc[2];double den=sqrt(gw2*gw2+gw3*gw3);if(den<1e-9){etas[i]=0;continue;}double cv=gw3/den;if(cv>1)cv=1;if(cv<-1)cv=-1;etas[i]=rad2deg*acos(cv);if(gw2>0)etas[i]=-etas[i];}
          for(int i=0;i<nsol;i++){
            double EtaAbs=fabs(etas[i]);
            if(EtaAbs<Params.ExcludePoleAngle||(180.0-EtaAbs)<Params.ExcludePoleAngle) continue;
            double eR=deg2rad*etas[i], yl=-(sin(eR)*RR), zl=cos(eR)*RR;
            int keep=0;
            for(int orn=0;orn<Params.NoOfOmegaRanges;orn++){if(omegas[i]>Params.OmegaRanges[orn][0]&&omegas[i]<Params.OmegaRanges[orn][1]&&yl>Params.BoxSizes[orn][0]&&yl<Params.BoxSizes[orn][1]&&zl>Params.BoxSizes[orn][2]&&zl<Params.BoxSizes[orn][3]){keep=1;break;}}
            if(!keep) continue;
            nTspFrac++;
            // Displacement
            double xi0=Dist,yi0=yl,zi0=zl,ome0=omegas[i];
            double lI=1.0/sqrt(xi0*xi0+yi0*yi0+zi0*zi0);double xn=xi0*lI,yn=yi0*lI,zn=zi0*lI;
            double sO=sin(deg2rad*ome0),cO=cos(deg2rad*ome0);
            double tv=(ga*cO-gb*sO)/xn;
            double tY=yi0+(ga*sO+gb*cO)-tv*yn, tZ=zi0+gc-tv*zn;
            double tOme=ome0;
            // Eta after displacement
            double tDen=sqrt(tY*tY+tZ*tZ);
            double tEta=0;if(tDen>1e-9){double cv2=tZ/tDen;if(cv2>1)cv2=1;if(cv2<-1)cv2=-1;tEta=rad2deg*acos(cv2);if(tY>0)tEta=-tEta;}
            double tRadDiff=tDen-RR;
            // Bin lookup + match
            int iRing=ringnr-1;
            int iEta=(int)floor((180.0+tEta)/Params.EtaBinSize);
            int iOme=(int)floor((180.0+tOme)/Params.OmeBinSize);
            if(iEta<0)iEta=0;if(iEta>=n_eta_bins)iEta=n_eta_bins-1;
            if(iOme<0)iOme=0;if(iOme>=n_ome_bins)iOme=n_ome_bins-1;
            size_t Pos=(size_t)iRing*n_eta_bins*n_ome_bins+(size_t)iEta*n_ome_bins+iOme;
            int nInBin=ndata[(int)(Pos*2+0)],DataPos_=ndata[(int)(Pos*2+1)];
            double etam=etamargins[ringnr];
            int bestRow=-1; double diffOmeBest=1e9;
            for(int is=0;is<nInBin;is++){
              int sr=data[DataPos_+is];int base=sr*N_COL_OBSSPOTS;
              double obsRadDiff=(double)ObsSpotsLab[base+8];
              if(fabs(tRadDiff-obsRadDiff)>=Params.MarginRadial) continue;
              double obsEta=(double)ObsSpotsLab[base+6];
              if(fabs(tEta-obsEta)>=etam)continue;
              double obsOme=(double)ObsSpotsLab[base+2];
              double dO=fabs(tOme-obsOme);
              if(dO<diffOmeBest){diffOmeBest=dO;bestRow=sr;}
            }
            theorY[nTsp]=tY; theorZ[nTsp]=tZ; theorOme[nTsp]=tOme;
            theorRing[nTsp]=ringnr;
            theorMatched[nTsp]=(bestRow>=0)?1:0;
            matchedRow[nTsp]=bestRow;
            nTsp++;
          }
        }
        // Compute IA in double, with per-step trace for first matched spot
        double iaSum=0; int iaCount=0, nMatchD=0;
        int traceDone=0;
        for(int sp=0;sp<nTsp;sp++){
          if(!theorMatched[sp]) continue;
          nMatchD++;
          int br=matchedRow[sp];
          double obsY=(double)ObsSpotsLab[br*N_COL_OBSSPOTS+0];
          double obsZ=(double)ObsSpotsLab[br*N_COL_OBSSPOTS+1];
          double obsOme=(double)ObsSpotsLab[br*N_COL_OBSSPOTS+2];
          // gv1 = spot_to_gv_pos(Dist, theorY, theorZ, theorOme, ga, gb, gc)
          double v1[3]={ga,gb,gc},vr1[3];{double ca=cos(deg2rad*theorOme[sp]),sa=sin(deg2rad*theorOme[sp]);vr1[0]=ca*v1[0]-sa*v1[1];vr1[1]=sa*v1[0]+ca*v1[1];vr1[2]=v1[2];}
          double xi1=Dist-vr1[0],yi1=theorY[sp]-vr1[1],zi1=theorZ[sp]-vr1[2];
          double l1=sqrt(xi1*xi1+yi1*yi1+zi1*zi1);double xn1=xi1/l1,yn1=yi1/l1,zn1=zi1/l1;
          double g1r=-1+xn1,g2r=yn1;double co1=cos(-theorOme[sp]*deg2rad),so1=sin(-theorOme[sp]*deg2rad);
          double gv1x=g1r*co1-g2r*so1,gv1y=g1r*so1+g2r*co1,gv1z=zn1;
          // gv2
          double vr2[3];{double ca=cos(deg2rad*obsOme),sa=sin(deg2rad*obsOme);vr2[0]=ca*v1[0]-sa*v1[1];vr2[1]=sa*v1[0]+ca*v1[1];vr2[2]=v1[2];}
          double xi2=Dist-vr2[0],yi2=obsY-vr2[1],zi2=obsZ-vr2[2];
          double l2=sqrt(xi2*xi2+yi2*yi2+zi2*zi2);double xn2=xi2/l2,yn2=yi2/l2,zn2=zi2/l2;
          double g1r2=-1+xn2,g2r2=yn2;double co2=cos(-obsOme*deg2rad),so2=sin(-obsOme*deg2rad);
          double gv2x=g1r2*co2-g2r2*so2,gv2y=g1r2*so2+g2r2*co2,gv2z=zn2;
          double la=sqrt(gv1x*gv1x+gv1y*gv1y+gv1z*gv1z),lb=sqrt(gv2x*gv2x+gv2y*gv2y+gv2z*gv2z);
          double dp=(gv1x*gv2x+gv1y*gv2y+gv1z*gv2z)/(la*lb);
          if(dp>1)dp=1;if(dp<-1)dp=-1;
          double ia=rad2deg*acos(dp);

          // Per-step trace for first matched spot of this candidate
          if (!traceDone) {
            // Now do the same in float for comparison
            float obsY_f=ObsSpotsLab[br*N_COL_OBSSPOTS+0];
            float obsZ_f=ObsSpotsLab[br*N_COL_OBSSPOTS+1];
            float obsOme_f=ObsSpotsLab[br*N_COL_OBSSPOTS+2];
            float tY_f=(float)theorY[sp],tZ_f=(float)theorZ[sp],tO_f=(float)theorOme[sp];
            float ga_f=(float)ga,gb_f=(float)gb,gc_f=(float)gc,Dist_f=(float)Dist;
            // Step 1: RotateAroundZ
            float v1f[3]={ga_f,gb_f,gc_f},vr1f[3];
            {float ca=cosf(deg2rad*tO_f),sa=sinf(deg2rad*tO_f);vr1f[0]=ca*v1f[0]-sa*v1f[1];vr1f[1]=sa*v1f[0]+ca*v1f[1];vr1f[2]=v1f[2];}
            // Step 2: subtract
            float xi1f=Dist_f-vr1f[0],yi1f=tY_f-vr1f[1],zi1f=tZ_f-vr1f[2];
            // Step 3: normalize
            float l1f=sqrtf(xi1f*xi1f+yi1f*yi1f+zi1f*zi1f);
            float xn1f=xi1f/l1f,yn1f=yi1f/l1f,zn1f=zi1f/l1f;
            // Step 4: g1r,g2r
            float g1rf=-1+xn1f,g2rf=yn1f;
            // Step 5: rotate by -omega
            float co1f=cosf(-tO_f*deg2rad),so1f=sinf(-tO_f*deg2rad);
            float gv1xf=g1rf*co1f-g2rf*so1f,gv1yf=g1rf*so1f+g2rf*co1f,gv1zf=zn1f;
            // Same for obs
            float vr2f[3];{float ca=cosf(deg2rad*obsOme_f),sa=sinf(deg2rad*obsOme_f);vr2f[0]=ca*v1f[0]-sa*v1f[1];vr2f[1]=sa*v1f[0]+ca*v1f[1];vr2f[2]=v1f[2];}
            float xi2f=Dist_f-vr2f[0],yi2f=obsY_f-vr2f[1],zi2f=obsZ_f-vr2f[2];
            float l2f=sqrtf(xi2f*xi2f+yi2f*yi2f+zi2f*zi2f);
            float xn2f=xi2f/l2f,yn2f=yi2f/l2f,zn2f=zi2f/l2f;
            float g1r2f=-1+xn2f,g2r2f=yn2f;
            float co2f=cosf(-obsOme_f*deg2rad),so2f=sinf(-obsOme_f*deg2rad);
            float gv2xf=g1r2f*co2f-g2r2f*so2f,gv2yf=g1r2f*so2f+g2r2f*co2f,gv2zf=zn2f;
            float laf=sqrtf(gv1xf*gv1xf+gv1yf*gv1yf+gv1zf*gv1zf);
            float lbf=sqrtf(gv2xf*gv2xf+gv2yf*gv2yf+gv2zf*gv2zf);
            float dpf=(gv1xf*gv2xf+gv1yf*gv2yf+gv1zf*gv2zf)/(laf*lbf);
            if(dpf>1)dpf=1;if(dpf<-1)dpf=-1;
            float iaf=rad2deg*acosf(dpf);

            printf("[IA TRACE] pos=(%.2f,%.2f,%.2f) spot=%d obsRow=%d theorOme=%.6f obsOme=%.6f\n",
                   t->ga,t->gb,t->gc, sp, br, theorOme[sp], obsOme);
            printf("  [STEP1 RotZ]  vr_dbl=(%.10f,%.10f,%.10f)  vr_flt=(%.10f,%.10f,%.10f)\n",
                   vr1[0],vr1[1],vr1[2], (double)vr1f[0],(double)vr1f[1],(double)vr1f[2]);
            printf("  [STEP2 sub]   xi_dbl=%.10f yi_dbl=%.10f zi_dbl=%.10f\n", xi1, yi1, zi1);
            printf("                xi_flt=%.10f yi_flt=%.10f zi_flt=%.10f\n", (double)xi1f,(double)yi1f,(double)zi1f);
            printf("  [STEP3 norm]  xn_dbl=%.12f yn_dbl=%.12f zn_dbl=%.12f  len=%.10f\n", xn1,yn1,zn1,l1);
            printf("                xn_flt=%.12f yn_flt=%.12f zn_flt=%.12f  len=%.10f\n", (double)xn1f,(double)yn1f,(double)zn1f,(double)l1f);
            printf("  [STEP4 g1g2]  g1r_dbl=%.15e g2r_dbl=%.15e\n", g1r, g2r);
            printf("                g1r_flt=%.15e g2r_flt=%.15e\n", (double)g1rf, (double)g2rf);
            printf("  [STEP5 gv1]   gv_dbl=(%.12f,%.12f,%.12f)\n", gv1x,gv1y,gv1z);
            printf("                gv_flt=(%.12f,%.12f,%.12f)\n", (double)gv1xf,(double)gv1yf,(double)gv1zf);
            printf("  [STEP5 gv2]   gv_dbl=(%.12f,%.12f,%.12f)\n", gv2x,gv2y,gv2z);
            printf("                gv_flt=(%.12f,%.12f,%.12f)\n", (double)gv2xf,(double)gv2yf,(double)gv2zf);
            printf("  [STEP6 dot]   dp_dbl=%.15e  dp_flt=%.15e\n", dp, (double)dpf);
            printf("  [STEP7 ia]    ia_dbl=%.10f  ia_flt=%.10f\n", ia, (double)iaf);
            traceDone = 1;
          }

          iaSum+=fabs(ia); iaCount++;
        }
        double avgIA_double = (iaCount>0) ? iaSum/iaCount : 999.0;

        // Now recompute using float (same as GPU kernel would)
        float iaSum_f=0; int iaCount_f=0;
        for(int sp=0;sp<nTsp;sp++){
          if(!theorMatched[sp]) continue;
          int br=matchedRow[sp];
          float obsY=ObsSpotsLab[br*N_COL_OBSSPOTS+0];
          float obsZ=ObsSpotsLab[br*N_COL_OBSSPOTS+1];
          float obsOme=ObsSpotsLab[br*N_COL_OBSSPOTS+2];
          float tY_f=(float)theorY[sp],tZ_f=(float)theorZ[sp],tO_f=(float)theorOme[sp];
          float ga_f=(float)ga,gb_f=(float)gb,gc_f=(float)gc,Dist_f=(float)Dist;
          // gv1 inline float
          float v1f[3]={ga_f,gb_f,gc_f},vr1f[3];{float ca=cosf(deg2rad*tO_f),sa=sinf(deg2rad*tO_f);vr1f[0]=ca*v1f[0]-sa*v1f[1];vr1f[1]=sa*v1f[0]+ca*v1f[1];vr1f[2]=v1f[2];}
          float xi1f=Dist_f-vr1f[0],yi1f=tY_f-vr1f[1],zi1f=tZ_f-vr1f[2];
          float l1f=sqrtf(xi1f*xi1f+yi1f*yi1f+zi1f*zi1f);float xn1f=xi1f/l1f,yn1f=yi1f/l1f,zn1f=zi1f/l1f;
          float g1rf=-1+xn1f,g2rf=yn1f;float co1f=cosf(-tO_f*deg2rad),so1f=sinf(-tO_f*deg2rad);
          float gv1x_f=g1rf*co1f-g2rf*so1f,gv1y_f=g1rf*so1f+g2rf*co1f,gv1z_f=zn1f;
          // gv2 inline float
          float vr2f[3];{float ca=cosf(deg2rad*obsOme),sa=sinf(deg2rad*obsOme);vr2f[0]=ca*v1f[0]-sa*v1f[1];vr2f[1]=sa*v1f[0]+ca*v1f[1];vr2f[2]=v1f[2];}
          float xi2f=Dist_f-vr2f[0],yi2f=obsY-vr2f[1],zi2f=obsZ-vr2f[2];
          float l2f=sqrtf(xi2f*xi2f+yi2f*yi2f+zi2f*zi2f);float xn2f=xi2f/l2f,yn2f=yi2f/l2f,zn2f=zi2f/l2f;
          float g1r2f=-1+xn2f,g2r2f=yn2f;float co2f=cosf(-obsOme*deg2rad),so2f=sinf(-obsOme*deg2rad);
          float gv2x_f=g1r2f*co2f-g2r2f*so2f,gv2y_f=g1r2f*so2f+g2r2f*co2f,gv2z_f=zn2f;
          float la_f=sqrtf(gv1x_f*gv1x_f+gv1y_f*gv1y_f+gv1z_f*gv1z_f);
          float lb_f=sqrtf(gv2x_f*gv2x_f+gv2y_f*gv2y_f+gv2z_f*gv2z_f);
          float dp_f=(gv1x_f*gv2x_f+gv1y_f*gv2y_f+gv1z_f*gv2z_f)/(la_f*lb_f);
          if(dp_f>1)dp_f=1;if(dp_f<-1)dp_f=-1;
          float ia_f=rad2deg*acosf(dp_f);
          iaSum_f+=fabsf(ia_f); iaCount_f++;
        }
        float avgIA_float = (iaCount_f>0) ? iaSum_f/iaCount_f : 999.0f;

        printf("[IA VERIFY] pos=(%.2f,%.2f,%.2f) nObs=%d nExp=%d IA_float=%.6f IA_double=%.6f\n",
               t->ga, t->gb, t->gc, nMatchD, nTspFrac, avgIA_float, avgIA_double);
      }
    }

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
