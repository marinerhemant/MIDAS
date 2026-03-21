//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated NF-HEDM orientation screening for MIDAS.
//
// Architecture:
//   Phase 1 (screening): Each GPU thread handles one (voxel, orientation) pair.
//   It iterates over all theoretical spots for the orientation, displaces each
//   for the voxel position, rasterizes the triangle footprint using
//   CalcPixels2-equivalent logic, and checks each pixel against ALL layers in
//   ObsSpotsInfo.  Produces FracOverlap per pair; winners above threshold are
//   atomically appended to an output buffer.
//
//   Phase 2 (fitting) remains on CPU — the GPU returns a sparse list of
//   (voxel, orientation, FracOverlap) winners that the existing NLopt
//   Nelder-Mead code can refine.
//

#include "nf_gpu.h"

#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Forward-declare only what we need from NF-HEDM.
// We cannot include nf_headers.h directly because it uses C99 VLAs
// (e.g., double Lsd[nLayers]) which are illegal in C++ / CUDA.
extern "C" {

#define RealType double
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

// NF-HEDM uses MAX_N_SPOTS=5000 and MAX_N_OMEGA_RANGES=20 for
// CalcDiffractionSpots (these are array-size hints only, decay to pointers).
#define NF_MAX_N_SPOTS 5000
#define NF_MAX_N_OMEGA_RANGES 20

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[NF_MAX_N_OMEGA_RANGES][2],
                         int NoOfOmegaRanges,
                         double hkls[NF_MAX_N_SPOTS][4], int n_hkls,
                         double Thetas[NF_MAX_N_SPOTS],
                         double BoxSizes[NF_MAX_N_OMEGA_RANGES][4],
                         int *nTspots,
                         double OrientMatr[3][3], double *TheorSpots,
                         double *Gs);
}

// ─────────────────────────────────────────────────────────────
// Error-checking macros
// ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "NF GPU CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_NULL(call)                                                  \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "NF GPU CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                         \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

static double nf_gpu_timer_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ─────────────────────────────────────────────────────────────
// GPU-side data structures
// ─────────────────────────────────────────────────────────────

/// Per-spot metadata, precomputed once per orientation (voxel-independent).
struct GPUSpot {
  float y;        ///< Detector-space Y of theoretical spot
  float z;        ///< Detector-space Z of theoretical spot
  float sinOme;   ///< sin(omega) — precomputed
  float cosOme;   ///< cos(omega) — precomputed
  float refYpx;   ///< Reference pixel Y (undisplaced, voxel-independent)
  float refZpx;   ///< Reference pixel Z (undisplaced, voxel-independent)
  int   omeBin;   ///< Omega frame index
  int   valid;    ///< 1 if spot is in range, 0 if out-of-bounds
};

/// Per-orientation header: number of spots and offset into spot array.
struct GPUOrientHeader {
  int nSpots;
  int spotOffset;
};

/// Comparison function for sorting GPUSpots by omeBin (L2 locality optimization)
static int nf_gpu_spot_cmp_omebin(const void *a, const void *b) {
  int oa = ((const GPUSpot *)a)->omeBin;
  int ob = ((const GPUSpot *)b)->omeBin;
  return (oa > ob) - (oa < ob);
}

// ─────────────────────────────────────────────────────────────
// GPU context
// ─────────────────────────────────────────────────────────────

struct NFGPUContext {
  int deviceId;
  int nrPixelsY, nrPixelsZ;
  int nrFiles;        // Number of omega frames
  int nLayers;

  // Flat ObsSpotsInfo on device (same layout as CPU)
  // Bit = layer*nrFiles*nPxY*nPxZ + omeBin*nPxY*nPxZ + pixY*nPxZ + pixZ
  uint32_t *d_obsFlat;
  long long obsFlatSize;  // Number of bits
  int obsUploaded;

  // Precomputed spot data on device
  GPUSpot *d_spots;
  GPUOrientHeader *d_orientHeaders;
  int nOrientations;
  int totalSpots;
  int orientsUploaded;

  // Geometry constants (uploaded once)
  float *d_Lsd;   // [nLayers]
  float *d_ybc;   // [nLayers]
  float *d_zbc;   // [nLayers]
  float px, gs;
  float rotMatTilts[9]; // Flattened 3x3
  float *d_P0;    // [nLayers * 3]

  // Screening output
  NFGPUWinner *d_winners;
  int *d_nWinners;
  int maxWinners;

  // Phase 2 fitting: HKL data in global memory (too large for __constant__)
  float *d_hkls;    // [n_hkls * 4]
  float *d_Gs;      // [n_hkls]
  int n_hkls;
  int hklsUploaded;

  cudaStream_t stream;
};

// ═════════════════════════════════════════════════════════════
//  GPU HELPER: Test bit in flat ObsSpotsInfo (same as CPU TestBit)
// ═════════════════════════════════════════════════════════════

__device__ static inline int gpu_test_bit(const uint32_t *obs, long long bitIdx) {
  int word = (int)(bitIdx / 32);
  int bit  = (int)(bitIdx % 32);
  return (obs[word] >> bit) & 1;
}

// ═════════════════════════════════════════════════════════════
//  Constant memory for geometry (broadcast to all threads)
// ═════════════════════════════════════════════════════════════
#define MAX_LAYERS 8
__constant__ float c_Lsd[MAX_LAYERS];
__constant__ float c_ybc[MAX_LAYERS];
__constant__ float c_zbc[MAX_LAYERS];
__constant__ float c_P0[MAX_LAYERS * 3];
__constant__ float c_RM[9];

// ═════════════════════════════════════════════════════════════
//  GPU HELPER: Precompute reference pixel positions (voxel-independent)
//  Must use the SAME GPU FMA arithmetic as the screening kernel.
// ═════════════════════════════════════════════════════════════

__global__ void compute_ref_positions_kernel(
    GPUSpot *spots, int totalSpots,
    float px, float ybc0, float zbc0) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= totalSpots) return;

  float ythis = spots[tid].y;
  float zthis = spots[tid].z;

  // Same arithmetic as in screen_pairs_kernel:
  float refP1x = c_RM[0*3+1] * ythis + c_RM[0*3+2] * zthis;
  float refP1y = c_RM[1*3+1] * ythis + c_RM[1*3+2] * zthis;
  float refP1z = c_RM[2*3+1] * ythis + c_RM[2*3+2] * zthis;

  float P0_0[3];
  P0_0[0] = c_P0[0]; P0_0[1] = c_P0[1]; P0_0[2] = c_P0[2];

  float refABCx = refP1x - P0_0[0];
  float refABCy = refP1y - P0_0[1];
  float refABCz = refP1z - P0_0[2];

  float refOutY = P0_0[1] - (refABCy * P0_0[0]) / refABCx;
  float refOutZ = P0_0[2] - (refABCz * P0_0[0]) / refABCx;

  spots[tid].refYpx = refOutY / px + ybc0;
  spots[tid].refZpx = refOutZ / px + zbc0;
}

// ═════════════════════════════════════════════════════════════
//  GPU HELPER: DisplacementSpotsPrecomp (same as CPU)
// ═════════════════════════════════════════════════════════════

__device__ static inline void gpu_displacement(
    float XGT, float YGT, float Lsd,
    float ythis, float zthis, float sinOme, float cosOme,
    float *Displ_Y, float *Displ_Z) {
  float xa = XGT * cosOme - YGT * sinOme;
  float ya = XGT * sinOme + YGT * cosOme;
  float t = 1.0f - (xa / Lsd);
  *Displ_Y = ya + ythis * t;
  *Displ_Z = t * zthis;
}



// ═════════════════════════════════════════════════════════════
//  MAIN SCREENING KERNEL: one thread = one (voxel, orientation) pair
//  Exact same logic as original, but reads geometry from __constant__.
// ═════════════════════════════════════════════════════════════

__global__ __launch_bounds__(128, 8)
void screen_pairs_kernel(
    const uint32_t * __restrict__ obsFlat,
    const GPUSpot * __restrict__ spots,
    const GPUOrientHeader * __restrict__ headers,
    const float * __restrict__ voxXG,
    const float * __restrict__ voxYG,
    int nVoxels,
    int nOrientations,
    int nLayers,
    int nrFiles,
    int nrPixelsY, int nrPixelsZ,
    int frameBatchSize,
    float px, float gs,
    float minFracOverlap,
    NFGPUWinner *winners,
    int *nWinners,
    int maxWinners,
    int diagonal) {

  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int oriIdx, voxIdx;
  if (diagonal) {
    // 1:1 mode: each thread evaluates one candidate (oriIdx == voxIdx == tid)
    if (tid >= nVoxels) return;
    oriIdx = (int)tid;
    voxIdx = (int)tid;
  } else {
    // 2D grid mode: tid maps to (oriIdx, voxIdx) pair
    long long totalPairs = (long long)nVoxels * nOrientations;
    if (tid >= totalPairs) return;
    oriIdx = (int)(tid / nVoxels);
    voxIdx = (int)(tid % nVoxels);
  }

  float RM[9];
  #pragma unroll
  for (int i = 0; i < 9; i++) RM[i] = c_RM[i];

  float XG[3], YG[3];
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    XG[k] = __ldg(&voxXG[voxIdx * 3 + k]);
    YG[k] = __ldg(&voxYG[voxIdx * 3 + k]);
  }

  float Lsd0 = c_Lsd[0];
  float ybc0 = c_ybc[0];
  float zbc0 = c_zbc[0];
  float P0_0[3] = { c_P0[0], c_P0[1], c_P0[2] };

  GPUOrientHeader hdr = headers[oriIdx];

  int OverlapPixels = 0;
  int TotalPixels = 0;

  // Frame-synchronized spot processing: iterate over omega frame batches.
  // All threads on the SM process the same frame range simultaneously,
  // so the bitfield data for these frames stays L2-resident.
  // Spots are sorted by omeBin, so binary search finds the sub-range.
  const GPUSpot *mySpots = &spots[hdr.spotOffset];
  int nSpots = hdr.nSpots;

  for (int fbStart = 0; fbStart < nrFiles; fbStart += frameBatchSize) {
    int fbEnd = fbStart + frameBatchSize;
    if (fbEnd > nrFiles) fbEnd = nrFiles;

    // Binary search: first spot with omeBin >= fbStart
    int sLo;
    {
      int lo = 0, hi = nSpots;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (mySpots[mid].omeBin < fbStart) lo = mid + 1;
        else hi = mid;
      }
      sLo = lo;
    }
    // Binary search: first spot with omeBin >= fbEnd
    int sHi;
    {
      int lo = sLo, hi = nSpots;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (mySpots[mid].omeBin < fbEnd) lo = mid + 1;
        else hi = mid;
      }
      sHi = lo;
    }

    for (int s = sLo; s < sHi; s++) {
    GPUSpot spot = mySpots[s];
    if (!spot.valid) continue;

    float ythis = spot.y;
    float zthis = spot.z;
    int omeBin = spot.omeBin;
    float sinOme = spot.sinOme;
    float cosOme = spot.cosOme;

    float YZSpotsT[3][2];
    int oob = 0;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      float Displ_Y, Displ_Z;
      gpu_displacement(XG[k], YG[k], Lsd0, ythis, zthis, sinOme, cosOme,
                       &Displ_Y, &Displ_Z);

      float P1x = RM[0*3+1] * Displ_Y + RM[0*3+2] * Displ_Z;
      float P1y = RM[1*3+1] * Displ_Y + RM[1*3+2] * Displ_Z;
      float P1z = RM[2*3+1] * Displ_Y + RM[2*3+2] * Displ_Z;

      float ABCx = P1x - P0_0[0];
      float ABCy = P1y - P0_0[1];
      float ABCz = P1z - P0_0[2];

      float outY = P0_0[1] - (ABCy * P0_0[0]) / ABCx;
      float outZ = P0_0[2] - (ABCz * P0_0[0]) / ABCx;

      YZSpotsT[k][0] = outY / px + ybc0;
      YZSpotsT[k][1] = outZ / px + zbc0;

      if (YZSpotsT[k][0] > nrPixelsY || YZSpotsT[k][0] < 0 ||
          YZSpotsT[k][1] > nrPixelsZ || YZSpotsT[k][1] < 0) {
        oob = 1;
        break;
      }
    }
    if (oob) continue;

    // Reference pixel position: precomputed on CPU (voxel-independent)
    float refYpx = spot.refYpx;
    float refZpx = spot.refZpx;

    float YZSpots[3][2];
    #pragma unroll
    for (int l = 0; l < 3; l++) {
      YZSpots[l][0] = YZSpotsT[l][0] - refYpx;
      YZSpots[l][1] = YZSpotsT[l][1] - refZpx;
    }

    // Precompute per-layer base positions once per spot (no pixel dependency)
    // Same arithmetic as original: floorf(((refYpx - ybc0) * px * (Lsd[l]/Lsd0)) / px + ybc[l])
    int layerBaseY[MAX_LAYERS], layerBaseZ[MAX_LAYERS];
    long long layerBinBase[MAX_LAYERS];
    for (int layer = 0; layer < nLayers; layer++) {
      layerBaseY[layer] = (int)floorf(((refYpx - ybc0) * px * (c_Lsd[layer] / Lsd0)) / px
                                      + c_ybc[layer]);
      layerBaseZ[layer] = (int)floorf(((refZpx - zbc0) * px * (c_Lsd[layer] / Lsd0)) / px
                                      + c_zbc[layer]);
      layerBinBase[layer] = (long long)layer * nrFiles * nrPixelsY * nrPixelsZ
                          + (long long)omeBin * nrPixelsY * nrPixelsZ;
    }

    // One-pass: rasterize + check bitfield inline (no local arrays needed)
    if (gs * 2.0f > px) {
      float edges[3][2];
      float minY = 1e9f, maxY = -1e9f, minZ = 1e9f, maxZ = -1e9f;
      for (int i = 0; i < 3; i++) {
        edges[i][0] = roundf(YZSpots[i][0]);
        edges[i][1] = roundf(YZSpots[i][1]);
        if (edges[i][0] < minY) minY = edges[i][0];
        if (edges[i][0] > maxY) maxY = edges[i][0];
        if (edges[i][1] < minZ) minZ = edges[i][1];
        if (edges[i][1] > maxZ) maxZ = edges[i][1];
      }

      for (int pz = (int)minZ; pz <= (int)maxZ; pz++) {
        for (int py = (int)minY; py <= (int)maxY; py++) {
          int w0 = ((int)edges[1][1]-(int)edges[2][1]) * (py-(int)edges[1][0])
                 + ((int)edges[2][0]-(int)edges[1][0]) * (pz-(int)edges[1][1]);
          int w1 = ((int)edges[2][1]-(int)edges[0][1]) * (py-(int)edges[2][0])
                 + ((int)edges[0][0]-(int)edges[2][0]) * (pz-(int)edges[2][1]);
          int w2 = ((int)edges[0][1]-(int)edges[1][1]) * (py-(int)edges[0][0])
                 + ((int)edges[1][0]-(int)edges[0][0]) * (pz-(int)edges[0][1]);

          int inside = (w0 >= 0 && w1 >= 0 && w2 >= 0);

          if (!inside) {
            for (int e = 0; e < 3 && !inside; e++) {
              int i0 = e, i1 = (e + 1) % 3;
              float dx = edges[i1][0] - edges[i0][0];
              float dz = edges[i1][1] - edges[i0][1];
              float num = dx * (pz - (int)edges[i0][1])
                        - dz * (py - (int)edges[i0][0]);
              float den = dz * dz + dx * dx;
              if (den > 0 && (num * num) / den < 0.9801f)
                inside = 1;
            }
          }

          if (inside) {
            int allFound = 1;
            int pixOob = 0;

            for (int layer = 0; layer < nLayers; layer++) {
              int MultY = layerBaseY[layer] + py;
              int MultZ = layerBaseZ[layer] + pz;

              if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
                pixOob = 1;
                break;
              }

              long long binNr = layerBinBase[layer]
                              + (long long)MultY * nrPixelsZ
                              + MultZ;

              if (!gpu_test_bit(obsFlat, binNr)) {
                allFound = 0;
              }
            }

            if (!pixOob) {
              if (allFound) OverlapPixels++;
              TotalPixels++;
            }
          }
        }
      }
    } else {
      int py = (int)roundf((YZSpots[0][0] + YZSpots[1][0] + YZSpots[2][0]) / 3.0f);
      int pz = (int)roundf((YZSpots[0][1] + YZSpots[1][1] + YZSpots[2][1]) / 3.0f);

      int allFound = 1;
      int pixOob = 0;
      for (int layer = 0; layer < nLayers; layer++) {
        int MultY = layerBaseY[layer] + py;
        int MultZ = layerBaseZ[layer] + pz;

        if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
          pixOob = 1;
          break;
        }

        long long binNr = layerBinBase[layer]
                        + (long long)MultY * nrPixelsZ
                        + MultZ;

        if (!gpu_test_bit(obsFlat, binNr)) {
          allFound = 0;
        }
      }

      if (!pixOob) {
        if (allFound) OverlapPixels++;
        TotalPixels++;
      }
    }  // end if/else (multi-pixel vs single-pixel)
    }  // end spot loop (for s)
  }  // end frame batch loop (for fbStart)

  if (TotalPixels > 0) {
    float fracOverlap = (float)OverlapPixels / (float)TotalPixels;
    if (fracOverlap >= minFracOverlap) {
      int pos = atomicAdd(nWinners, 1);
      if (pos < maxWinners) {
        winners[pos].voxelIdx = voxIdx;
        winners[pos].orientIdx = oriIdx;
        winners[pos].fracOverlap = fracOverlap;
      }
    }
  }
}


// ═════════════════════════════════════════════════════════════
//  PHASE 2: GPU NM FITTING — Device functions
// ═════════════════════════════════════════════════════════════

#include "../../utils/gpu_simplex.cuh"

// Maximum HKLs supported on GPU (matches CPU MAX_N_HKLS)
#define GPU_MAX_HKLS 5000
#define GPU_MAX_SPOTS_PER_ORIENT 500
#define GPU_MAX_LAYERS 8

// Note: HKL data (hkls, Gs) stored in global device memory (NFGPUContext)
// because arrays are too large for __constant__ (64KB limit).
__constant__ float c_omegaStart;
__constant__ float c_omegaStep;
__constant__ float c_excludePoleAngle;
__constant__ float c_omegaRanges[20][2];
__constant__ float c_boxSizes[20][4];
__constant__ int   c_nOmeRanges;

/// Device: Euler angles (degrees) → 3×3 orientation matrix
__device__ static inline void gpu_euler2orient(const float euler[3], float m[3][3]) {
  float psi = euler[0] * (float)deg2rad;
  float phi = euler[1] * (float)deg2rad;
  float theta = euler[2] * (float)deg2rad;
  float cps = cosf(psi), sps = sinf(psi);
  float cph = cosf(phi), sph = sinf(phi);
  float cth = cosf(theta), sth = sinf(theta);
  m[0][0] = cth * cps - sth * cph * sps;
  m[0][1] = -cth * cph * sps - sth * cps;
  m[0][2] = sph * sps;
  m[1][0] = cth * sps + sth * cph * cps;
  m[1][1] = cth * cph * cps - sth * sps;
  m[1][2] = -sph * cps;
  m[2][0] = sth * sph;
  m[2][1] = cth * sph;
  m[2][2] = cph;
}

/// Device: Compute omega solutions for a reciprocal lattice vector.
/// CalcOmega from CalcDiffractionSpots.c — up to 2 solutions.
__device__ static inline int gpu_calc_omega(
    float gx, float gy, float gz, float v,
    float omegas[4], float etas[4]) {
  int nsol = 0;
  float almostzero = 1e-4f;

  if (fabsf(gy) < almostzero) {
    if (gx != 0.0f) {
      float cosome1 = -v / gx;
      if (fabsf(cosome1) <= 1.0f) {
        float ome = acosf(cosome1) * (float)rad2deg;
        omegas[nsol++] = ome;
        omegas[nsol++] = -ome;
      }
    }
  } else {
    float y2 = gy * gy;
    float a = 1.0f + (gx * gx) / y2;
    float b = (2.0f * v * gx) / y2;
    float c = (v * v) / y2 - 1.0f;
    float discr = b * b - 4.0f * a * c;
    if (discr >= 0.0f) {
      float sd = sqrtf(discr);
      float cosome1 = (-b + sd) / (2.0f * a);
      if (fabsf(cosome1) <= 1.0f) {
        float ome1a = acosf(cosome1);
        float ome1b = -ome1a;
        float eqa = -gx * cosf(ome1a) + gy * sinf(ome1a);
        float eqb = -gx * cosf(ome1b) + gy * sinf(ome1b);
        omegas[nsol++] = (fabsf(eqa - v) < fabsf(eqb - v))
                       ? ome1a * (float)rad2deg
                       : ome1b * (float)rad2deg;
      }
      float cosome2 = (-b - sd) / (2.0f * a);
      if (fabsf(cosome2) <= 1.0f) {
        float ome2a = acosf(cosome2);
        float ome2b = -ome2a;
        float eqa = -gx * cosf(ome2a) + gy * sinf(ome2a);
        float eqb = -gx * cosf(ome2b) + gy * sinf(ome2b);
        omegas[nsol++] = (fabsf(eqa - v) < fabsf(eqb - v))
                       ? ome2a * (float)rad2deg
                       : ome2b * (float)rad2deg;
      }
    }
  }

  // Compute eta for each solution
  for (int i = 0; i < nsol; i++) {
    float omeRad = omegas[i] * (float)deg2rad;
    float cosO = cosf(omeRad), sinO = sinf(omeRad);
    // gw1 = gx * cosO + gy * sinO; (not needed for eta calc)
    float gw2 = -gx * sinO + gy * cosO;
    float gw3 = gz;
    // CalcEtaAngle
    float r = sqrtf(gw2 * gw2 + gw3 * gw3);
    float eta = (r > 1e-10f) ? (float)rad2deg * acosf(gw3 / r) : 0.0f;
    if (gw2 > 0.0f) eta = -eta;
    etas[i] = eta;
  }
  return nsol;
}

/// Device: Compute fracOverlap for a trial orientation.
/// This is the GPU version of CalcOverlapAccOrient:
///   Euler → OrientMat → CalcDiffrSpots → check bitfield
__device__ static float gpu_calc_frac_overlap(
    const float euler_deg[3],
    const float XG[3], const float YG[3],
    const uint32_t *obsFlat,
    float Lsd0, float ybc0, float zbc0,
    const float RM[9], const float P0_0[3],
    int nLayers, int nrFiles, int nrPixelsY, int nrPixelsZ,
    float px, float gs,
    const float *d_hkls, const float *d_Gs, int n_hkls,
    int debugPrint = 0) {

  // 1. Euler → orientation matrix
  float orient[3][3];
  gpu_euler2orient(euler_deg, orient);

  int OverlapPixels = 0, TotalPixels = 0;
  int nValidSpots = 0;

  if (debugPrint) {
    printf("  gpu_frac: euler=(%.2f,%.2f,%.2f) Lsd0=%.2f ybc0=%.2f zbc0=%.2f P0=(%.2f,%.2f,%.2f) nLayers=%d nHkls=%d px=%.3f gs=%.3f\n",
           euler_deg[0], euler_deg[1], euler_deg[2],
           Lsd0, ybc0, zbc0, P0_0[0], P0_0[1], P0_0[2], nLayers, n_hkls, px, gs);
  }

  // 2. For each HKL, compute spots
  for (int ih = 0; ih < n_hkls; ih++) {
    float Ghkl[3] = { d_hkls[ih*4+0], d_hkls[ih*4+1], d_hkls[ih*4+2] };
    float Gc[3]; // orient * Ghkl
    Gc[0] = orient[0][0]*Ghkl[0] + orient[0][1]*Ghkl[1] + orient[0][2]*Ghkl[2];
    Gc[1] = orient[1][0]*Ghkl[0] + orient[1][1]*Ghkl[1] + orient[1][2]*Ghkl[2];
    Gc[2] = orient[2][0]*Ghkl[0] + orient[2][1]*Ghkl[1] + orient[2][2]*Ghkl[2];

    float v = d_Gs[ih];
    float omegas[4], etas[4];
    int nsol = gpu_calc_omega(Gc[0], Gc[1], Gc[2], v, omegas, etas);

    for (int isol = 0; isol < nsol; isol++) {
      float Omega = omegas[isol];
      float Eta = etas[isol];
      float EtaAbs = fabsf(Eta);

      // Pole angle exclusion
      if (EtaAbs < c_excludePoleAngle ||
          (180.0f - EtaAbs) < c_excludePoleAngle) continue;

      // Ring radius and spot position
      float Lsd_dist = c_Lsd[0];  // Use Lsd[0] for ring radius
      // Thetas already baked into Gs, need ringrad from theta
      // RingRadius = distance * tan(2 * theta)
      // But theta = Thetas[ih] which is asin(wavelength/(2*d))
      // We need to recompute or pass ringrad. Simpler: position from eta + ringrad.
      float ringRad = Lsd_dist * tanf(2.0f * acosf(-v / sqrtf(Gc[0]*Gc[0] + Gc[1]*Gc[1] + Gc[2]*Gc[2])));
      // Actually, theta is already in Thetas. We can approximate:
      // We passed Gs[ih] = sin(theta)*|Ghkl| to CalcOmega. But we need
      // RingRadius = Lsd * tan(2*theta). Since v = sin(theta)*|G|:
      float lenG = sqrtf(Gc[0]*Gc[0] + Gc[1]*Gc[1] + Gc[2]*Gc[2]);
      float sinTheta = v / lenG;
      if (fabsf(sinTheta) > 1.0f) continue;
      float theta = asinf(sinTheta);
      float tan2th = tanf(2.0f * theta);
      float RingRadius = Lsd_dist * tan2th;

      // Spot position (yl, zl) from eta + ring radius
      float etaRad = Eta * (float)deg2rad;
      float yl = -sinf(etaRad) * RingRadius;
      float zl = cosf(etaRad) * RingRadius;

      // Omega range check
      int keepSpot = 0;
      for (int r = 0; r < c_nOmeRanges; r++) {
        if (Omega > c_omegaRanges[r][0] && Omega < c_omegaRanges[r][1] &&
            yl > c_boxSizes[r][0] && yl < c_boxSizes[r][1] &&
            zl > c_boxSizes[r][2] && zl < c_boxSizes[r][3]) {
          keepSpot = 1;
          break;
        }
      }
      if (!keepSpot) continue;

      // OmeBin
      int omeBin = (int)floorf((-c_omegaStart + Omega) / c_omegaStep);
      if (omeBin < 0 || omeBin >= nrFiles) continue;

      nValidSpots++;

      float sinOme = sinf(Omega * (float)deg2rad);
      float cosOme = cosf(Omega * (float)deg2rad);

      // 3. Displacement + projection (same as screening kernel)
      float YZSpotsT[3][2];
      int oob = 0;
      for (int k = 0; k < 3; k++) {
        float Displ_Y, Displ_Z;
        gpu_displacement(XG[k], YG[k], Lsd0, yl, zl, sinOme, cosOme,
                         &Displ_Y, &Displ_Z);
        float P1x = RM[0*3+1] * Displ_Y + RM[0*3+2] * Displ_Z;
        float P1y = RM[1*3+1] * Displ_Y + RM[1*3+2] * Displ_Z;
        float P1z = RM[2*3+1] * Displ_Y + RM[2*3+2] * Displ_Z;
        float ABCx = P1x - P0_0[0];
        float ABCy = P1y - P0_0[1];
        float ABCz = P1z - P0_0[2];
        float outY = P0_0[1] - (ABCy * P0_0[0]) / ABCx;
        float outZ = P0_0[2] - (ABCz * P0_0[0]) / ABCx;
        YZSpotsT[k][0] = outY / px + ybc0;
        YZSpotsT[k][1] = outZ / px + zbc0;
        if (YZSpotsT[k][0] > nrPixelsY || YZSpotsT[k][0] < 0 ||
            YZSpotsT[k][1] > nrPixelsZ || YZSpotsT[k][1] < 0) {
          oob = 1; break;
        }
      }
      if (oob) continue;

      // Reference pixel (undisplaced)
      float refP1x = RM[0*3+1] * yl + RM[0*3+2] * zl;
      float refP1y = RM[1*3+1] * yl + RM[1*3+2] * zl;
      float refP1z = RM[2*3+1] * yl + RM[2*3+2] * zl;
      float refABCx = refP1x - P0_0[0];
      float refABCy = refP1y - P0_0[1];
      float refABCz = refP1z - P0_0[2];
      float refOutY = P0_0[1] - (refABCy * P0_0[0]) / refABCx;
      float refOutZ = P0_0[2] - (refABCz * P0_0[0]) / refABCx;
      float refYpx = refOutY / px + ybc0;
      float refZpx = refOutZ / px + zbc0;

      float YZSpots[3][2];
      for (int l = 0; l < 3; l++) {
        YZSpots[l][0] = YZSpotsT[l][0] - refYpx;
        YZSpots[l][1] = YZSpotsT[l][1] - refZpx;
      }

      // Layer base positions
      int layerBaseY[GPU_MAX_LAYERS], layerBaseZ[GPU_MAX_LAYERS];
      long long layerBinBase[GPU_MAX_LAYERS];
      for (int layer = 0; layer < nLayers; layer++) {
        layerBaseY[layer] = (int)floorf(((refYpx - ybc0) * px * (c_Lsd[layer]/Lsd0))/px + c_ybc[layer]);
        layerBaseZ[layer] = (int)floorf(((refZpx - zbc0) * px * (c_Lsd[layer]/Lsd0))/px + c_zbc[layer]);
        layerBinBase[layer] = (long long)layer * nrFiles * nrPixelsY * nrPixelsZ
                            + (long long)omeBin * nrPixelsY * nrPixelsZ;
      }

      // 4. Single-pixel check (gs*2 <= px in typical NF)
      if (gs * 2.0f <= px) {
        int py = (int)roundf((YZSpots[0][0]+YZSpots[1][0]+YZSpots[2][0])/3.0f);
        int pz = (int)roundf((YZSpots[0][1]+YZSpots[1][1]+YZSpots[2][1])/3.0f);
        int allFound = 1;
        int pixOob = 0;

        // Debug print for first spot
        if (debugPrint && nValidSpots <= 3) {
          printf("  spot %d: yl=%.2f zl=%.2f ome=%.2f omeBin=%d refYpx=%.2f refZpx=%.2f py=%d pz=%d\n",
                 nValidSpots, yl, zl, Omega, omeBin, refYpx, refZpx, py, pz);
        }

        for (int layer = 0; layer < nLayers; layer++) {
          int MultY = layerBaseY[layer] + py;
          int MultZ = layerBaseZ[layer] + pz;
          if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
            pixOob = 1; break;
          }
          long long binNr = layerBinBase[layer] + (long long)MultY*nrPixelsZ + MultZ;
          int bitVal = gpu_test_bit(obsFlat, binNr) ? 1 : 0;

          if (debugPrint && nValidSpots <= 3) {
            printf("    layer %d: MultY=%d MultZ=%d binNr=%lld bit=%d\n",
                   layer, MultY, MultZ, binNr, bitVal);
          }

          if (!bitVal) { allFound = 0; break; }
        }
        if (!pixOob) {
          if (allFound) OverlapPixels++;
          TotalPixels++;
        }
      } else {
        // Multi-pixel: rasterize triangle (same as screening kernel)
        float edges[3][2];
        float minY = 1e9f, maxY = -1e9f, minZ = 1e9f, maxZ = -1e9f;
        for (int i = 0; i < 3; i++) {
          edges[i][0] = roundf(YZSpots[i][0]);
          edges[i][1] = roundf(YZSpots[i][1]);
          if (edges[i][0] < minY) minY = edges[i][0];
          if (edges[i][0] > maxY) maxY = edges[i][0];
          if (edges[i][1] < minZ) minZ = edges[i][1];
          if (edges[i][1] > maxZ) maxZ = edges[i][1];
        }
        for (int pz = (int)minZ; pz <= (int)maxZ; pz++) {
          for (int py = (int)minY; py <= (int)maxY; py++) {
            int w0 = ((int)edges[1][1]-(int)edges[2][1])*(py-(int)edges[1][0])
                   + ((int)edges[2][0]-(int)edges[1][0])*(pz-(int)edges[1][1]);
            int w1 = ((int)edges[2][1]-(int)edges[0][1])*(py-(int)edges[2][0])
                   + ((int)edges[0][0]-(int)edges[2][0])*(pz-(int)edges[2][1]);
            int w2 = ((int)edges[0][1]-(int)edges[1][1])*(py-(int)edges[0][0])
                   + ((int)edges[1][0]-(int)edges[0][0])*(pz-(int)edges[0][1]);
            int inside = (w0 >= 0 && w1 >= 0 && w2 >= 0);
            if (!inside) {
              for (int e = 0; e < 3 && !inside; e++) {
                int i0 = e, i1 = (e+1)%3;
                float dx = edges[i1][0] - edges[i0][0];
                float dz = edges[i1][1] - edges[i0][1];
                float num = dx*(pz-(int)edges[i0][1]) - dz*(py-(int)edges[i0][0]);
                float den = dz*dz + dx*dx;
                if (den > 0 && (num*num)/den < 0.9801f) inside = 1;
              }
            }
            if (inside) {
              int allFound = 1, pixOob = 0;
              for (int layer = 0; layer < nLayers; layer++) {
                int MultY = layerBaseY[layer] + py;
                int MultZ = layerBaseZ[layer] + pz;
                if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
                  pixOob = 1; break;
                }
                long long binNr = layerBinBase[layer] + (long long)MultY*nrPixelsZ + MultZ;
                if (!gpu_test_bit(obsFlat, binNr)) { allFound = 0; break; }
              }
              if (!pixOob) {
                if (allFound) OverlapPixels++;
                TotalPixels++;
              }
            }
          }
        }
      }
    }
  }

  if (debugPrint) {
    printf("  gpu_frac: nValidSpots=%d OverlapPixels=%d TotalPixels=%d frac=%.6f\n",
           nValidSpots, OverlapPixels, TotalPixels,
           TotalPixels > 0 ? (float)OverlapPixels / (float)TotalPixels : 0.0f);
  }
  return (TotalPixels > 0) ? (float)OverlapPixels / (float)TotalPixels : 0.0f;
}

/// GPU NM fitting functor: evaluates 1 - fracOverlap for trial Euler angles.
struct NFFitObjective {
  const uint32_t *obsFlat;
  const float *XG;    // [3] for this voxel
  const float *YG;    // [3] for this voxel
  float Lsd0, ybc0, zbc0;
  float RM[9];
  float P0_0[3];
  int nLayers, nrFiles, nrPixelsY, nrPixelsZ;
  float px, gs;
  const float *d_hkls;  // [n_hkls * 4] in global memory
  const float *d_Gs;    // [n_hkls] in global memory
  int n_hkls;
  mutable int evalCount;  // for debug printing
  int debugJob;           // 1 = print diagnostics for this job

  __device__ float operator()(const float *x, int ndim) const {
    float euler_deg[3] = { x[0] * (float)rad2deg,
                           x[1] * (float)rad2deg,
                           x[2] * (float)rad2deg };
    int doPrint = (debugJob && evalCount == 0) ? 1 : 0;
    const_cast<NFFitObjective*>(this)->evalCount++;
    float frac = gpu_calc_frac_overlap(
        euler_deg, XG, YG, obsFlat,
        Lsd0, ybc0, zbc0, RM, P0_0,
        nLayers, nrFiles, nrPixelsY, nrPixelsZ, px, gs,
        d_hkls, d_Gs, n_hkls, doPrint);
    return 1.0f - frac;
  }
};

/// Phase 2 fitting kernel: one thread per winner, runs NM simplex.
__global__ void nm_fit_kernel(
    int nJobs,
    const float *startEulers,   // [nJobs * 3] initial Euler angles (radians)
    const float *lowerBounds,   // [nJobs * 3] per-job lower bounds (radians)
    const float *upperBounds,   // [nJobs * 3] per-job upper bounds (radians)
    const float *voxXG,         // [nJobs * 3] per-job voxel XG
    const float *voxYG,         // [nJobs * 3] per-job voxel YG
    const uint32_t *obsFlat,
    float Lsd0, float ybc0, float zbc0,
    const float *RM,            // [9] rotation matrix
    const float *P0_0,          // [3] P0 layer 0
    int nLayers, int nrFiles, int nrPixelsY, int nrPixelsZ,
    float px, float gs,
    float tol, int maxIter, float initStep,
    const float *d_hkls,        // [n_hkls * 4] HKL data
    const float *d_Gs,          // [n_hkls] G magnitudes
    int n_hkls,
    float *results,             // [nJobs * 3] output Euler angles
    float *fvals_out) {         // [nJobs] output fracOverlap

  int jobIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (jobIdx >= nJobs) return;

  // Build per-thread objective functor
  NFFitObjective obj;
  obj.obsFlat = obsFlat;
  obj.XG = &voxXG[jobIdx * 3];
  obj.YG = &voxYG[jobIdx * 3];
  obj.Lsd0 = Lsd0;
  obj.ybc0 = ybc0;
  obj.zbc0 = zbc0;
  for (int i = 0; i < 9; i++) obj.RM[i] = RM[i];
  for (int i = 0; i < 3; i++) obj.P0_0[i] = P0_0[i];
  obj.nLayers = nLayers;
  obj.nrFiles = nrFiles;
  obj.nrPixelsY = nrPixelsY;
  obj.nrPixelsZ = nrPixelsZ;
  obj.px = px;
  obj.gs = gs;
  obj.d_hkls = d_hkls;
  obj.d_Gs = d_Gs;
  obj.n_hkls = n_hkls;
  obj.evalCount = 0;
  obj.debugJob = (jobIdx == 0) ? 1 : 0;  // debug only first job

  // Load per-job bounds
  float lo[3], hi[3];
  for (int i = 0; i < 3; i++) {
    lo[i] = lowerBounds[jobIdx * 3 + i];
    hi[i] = upperBounds[jobIdx * 3 + i];
  }

  // Initialize simplex
  NMParams nm = nm_default_params();
  float simplex[4][3];  // NDIM+1 × NDIM (3+1 × 3)
  float fvals[4];

  const float *x0 = &startEulers[jobIdx * 3];
  for (int j = 0; j < 3; j++) simplex[0][j] = x0[j];
  for (int j = 0; j < 3; j++) {
    if (simplex[0][j] < lo[j]) simplex[0][j] = lo[j];
    if (simplex[0][j] > hi[j]) simplex[0][j] = hi[j];
  }
  fvals[0] = obj(simplex[0], 3);

  // Debug print for first 5 jobs
  if (jobIdx < 5) {
    printf("GPU-NM job %d: start=(%.6f,%.6f,%.6f) deg, bounds=[%.6f±%.6f], fval0=%.6f (frac=%.6f)\n",
           jobIdx,
           x0[0]*(float)rad2deg, x0[1]*(float)rad2deg, x0[2]*(float)rad2deg,
           x0[0]*(float)rad2deg, (hi[0]-lo[0])*0.5f*(float)rad2deg,
           fvals[0], 1.0f - fvals[0]);
  }

  for (int i = 1; i <= 3; i++) {
    for (int j = 0; j < 3; j++) simplex[i][j] = simplex[0][j];
    float step = initStep * (hi[i-1] - lo[i-1]);
    if (step < 1e-8f) step = 1e-4f;
    simplex[i][i-1] += step;
    if (simplex[i][i-1] > hi[i-1]) simplex[i][i-1] = hi[i-1];
    fvals[i] = obj(simplex[i], 3);
  }

  // Main NM loop (inlined for NDIM=3)
  float trial[3], centroid[3];

  for (int iter = 0; iter < maxIter; iter++) {
    // Find best, worst, second_worst
    int best = 0, worst = 0, sw = 0;
    for (int i = 1; i <= 3; i++) {
      if (fvals[i] < fvals[best]) best = i;
      if (fvals[i] > fvals[worst]) worst = i;
    }
    sw = best;
    for (int i = 0; i <= 3; i++) {
      if (i == worst) continue;
      if (fvals[i] > fvals[sw]) sw = i;
    }

    if (fvals[worst] - fvals[best] < tol) break;

    // Centroid (exclude worst)
    for (int j = 0; j < 3; j++) centroid[j] = 0;
    for (int i = 0; i <= 3; i++) {
      if (i == worst) continue;
      for (int j = 0; j < 3; j++) centroid[j] += simplex[i][j];
    }
    for (int j = 0; j < 3; j++) centroid[j] /= 3.0f;

    // Reflection
    for (int j = 0; j < 3; j++) {
      trial[j] = centroid[j] + nm.alpha * (centroid[j] - simplex[worst][j]);
      if (trial[j] < lo[j]) trial[j] = lo[j];
      if (trial[j] > hi[j]) trial[j] = hi[j];
    }
    float f_r = obj(trial, 3);

    if (f_r < fvals[sw] && f_r >= fvals[best]) {
      for (int j = 0; j < 3; j++) simplex[worst][j] = trial[j];
      fvals[worst] = f_r;
      continue;
    }

    if (f_r < fvals[best]) {
      // Expansion
      float trial_e[3];
      for (int j = 0; j < 3; j++) {
        trial_e[j] = centroid[j] + nm.gamma * (trial[j] - centroid[j]);
        if (trial_e[j] < lo[j]) trial_e[j] = lo[j];
        if (trial_e[j] > hi[j]) trial_e[j] = hi[j];
      }
      float f_e = obj(trial_e, 3);
      if (f_e < f_r) {
        for (int j = 0; j < 3; j++) simplex[worst][j] = trial_e[j];
        fvals[worst] = f_e;
      } else {
        for (int j = 0; j < 3; j++) simplex[worst][j] = trial[j];
        fvals[worst] = f_r;
      }
      continue;
    }

    // Contraction
    for (int j = 0; j < 3; j++) {
      trial[j] = centroid[j] + nm.rho * (simplex[worst][j] - centroid[j]);
      if (trial[j] < lo[j]) trial[j] = lo[j];
      if (trial[j] > hi[j]) trial[j] = hi[j];
    }
    float f_c = obj(trial, 3);
    if (f_c < fvals[worst]) {
      for (int j = 0; j < 3; j++) simplex[worst][j] = trial[j];
      fvals[worst] = f_c;
      continue;
    }

    // Shrink
    for (int i = 0; i <= 3; i++) {
      if (i == best) continue;
      for (int j = 0; j < 3; j++) {
        simplex[i][j] = simplex[best][j] + nm.sigma * (simplex[i][j] - simplex[best][j]);
        if (simplex[i][j] < lo[j]) simplex[i][j] = lo[j];
        if (simplex[i][j] > hi[j]) simplex[i][j] = hi[j];
      }
      fvals[i] = obj(simplex[i], 3);
    }
  }

  // Output best
  int best = 0;
  for (int i = 1; i <= 3; i++)
    if (fvals[i] < fvals[best]) best = i;

  for (int j = 0; j < 3; j++)
    results[jobIdx * 3 + j] = simplex[best][j];
  fvals_out[jobIdx] = fvals[best];  // raw NM objective (= 1 - fracOverlap)
}


// ═════════════════════════════════════════════════════════════
//  CONTEXT LIFECYCLE
// ═════════════════════════════════════════════════════════════

extern "C" void nf_gpu_print_info(int deviceId) {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
    printf("NF GPU: could not query device %d\n", deviceId);
    return;
  }
  printf("NF GPU: %s, SM %d.%d, %.1f GB, %d SMs, smem %zu KB/SM\n",
         prop.name, prop.major, prop.minor,
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
         prop.multiProcessorCount,
         prop.sharedMemPerMultiprocessor / 1024);
}

extern "C" size_t nf_gpu_get_free_memory(void) {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return free_bytes;
}

extern "C" NFGPUContext *nf_gpu_init(int deviceId,
                                     int nrPixelsY, int nrPixelsZ,
                                     int nrFiles, int nLayers) {
  double tInit0 = nf_gpu_timer_sec();
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    fprintf(stderr, "NF GPU: failed to set device %d: %s\n",
            deviceId, cudaGetErrorString(err));
    return NULL;
  }

  nf_gpu_print_info(deviceId);

  NFGPUContext *ctx = (NFGPUContext *)calloc(1, sizeof(NFGPUContext));
  if (!ctx) return NULL;

  ctx->deviceId = deviceId;
  ctx->nrPixelsY = nrPixelsY;
  ctx->nrPixelsZ = nrPixelsZ;
  ctx->nrFiles = nrFiles;
  ctx->nLayers = nLayers;
  ctx->obsUploaded = 0;
  ctx->orientsUploaded = 0;

  // Allocate per-layer geometry arrays
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_Lsd, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_ybc, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_zbc, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_P0, nLayers * 3 * sizeof(float)));

  CUDA_CHECK_NULL(cudaStreamCreate(&ctx->stream));

  double dtInit = nf_gpu_timer_sec() - tInit0;
  long long totalBits = (long long)nLayers * nrFiles * nrPixelsY * nrPixelsZ;
  printf("NF GPU: context initialised — %dx%d detector, %d frames, %d layers (%.2f s)\n",
         nrPixelsY, nrPixelsZ, nrFiles, nLayers, dtInit);
  printf("NF GPU: ObsSpotsInfo flat size = %.1f MB (%lld bits)\n",
         (totalBits / 8.0) / (1024.0 * 1024.0), totalBits);

  return ctx;
}

extern "C" void nf_gpu_destroy(NFGPUContext *ctx) {
  if (!ctx) return;
  cudaSetDevice(ctx->deviceId);

  if (ctx->d_obsFlat)       cudaFree(ctx->d_obsFlat);
  if (ctx->d_spots)         cudaFree(ctx->d_spots);
  if (ctx->d_orientHeaders) cudaFree(ctx->d_orientHeaders);
  if (ctx->d_Lsd)           cudaFree(ctx->d_Lsd);
  if (ctx->d_ybc)           cudaFree(ctx->d_ybc);
  if (ctx->d_zbc)           cudaFree(ctx->d_zbc);
  if (ctx->d_P0)            cudaFree(ctx->d_P0);
  if (ctx->d_winners)       cudaFree(ctx->d_winners);
  if (ctx->d_nWinners)      cudaFree(ctx->d_nWinners);
  if (ctx->d_hkls)          cudaFree(ctx->d_hkls);
  if (ctx->d_Gs)            cudaFree(ctx->d_Gs);

  if (ctx->stream) cudaStreamDestroy(ctx->stream);

  free(ctx);
  printf("NF GPU: context destroyed\n");
}

// ═════════════════════════════════════════════════════════════
//  DATA UPLOAD
// ═════════════════════════════════════════════════════════════

extern "C" int nf_gpu_upload_obs_spots(NFGPUContext *ctx,
                                        const int *h_obsSpots,
                                        long long sizeObsSpots) {
  if (!ctx) return -1;
  cudaSetDevice(ctx->deviceId);
  double t0 = nf_gpu_timer_sec();

  // sizeObsSpots = number of int words (CPU already divided totalBits by 32).
  // Total bits = sizeObsSpots * 32.
  ctx->obsFlatSize = sizeObsSpots * 32;  // Store total number of BITS
  size_t nWords = (size_t)(sizeObsSpots + 1);  // +1 for safety (same as CPU mmap)
  size_t flatBytes = nWords * sizeof(uint32_t);

  printf("NF GPU: uploading ObsSpotsInfo: %lld words = %zu bytes (%.1f MB)\n",
         sizeObsSpots, flatBytes, flatBytes / (1024.0 * 1024.0));

  CUDA_CHECK(cudaMalloc(&ctx->d_obsFlat, flatBytes));
  CUDA_CHECK(cudaMemcpy(ctx->d_obsFlat, h_obsSpots, flatBytes,
                        cudaMemcpyHostToDevice));

  ctx->obsUploaded = 1;
  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: ObsSpotsInfo uploaded (%.1f MB, %.2f s)\n",
         flatBytes / (1024.0 * 1024.0), dt);


  return 0;
}

extern "C" int nf_gpu_upload_orientations(NFGPUContext *ctx,
                                           const double *h_orientationMatrix,
                                           int nOrientations,
                                           const double *h_spotsMat,
                                           const int *h_nrSpots,
                                           const double hkls[][4],
                                           const double *thetas,
                                           int n_hkls,
                                           const double *Gs,
                                           const double *Lsd,
                                           const double *ybc,
                                           const double *zbc,
                                           double px, double gs,
                                           double omegaStart, double omegaStep,
                                           const double rotMatTilts[3][3],
                                           double excludePoleAngle,
                                           const double omegaRanges[][2],
                                           int nOmeRanges,
                                           const double boxSizes[][4],
                                           double wedge, double wavelength) {
  if (!ctx) return -1;
  cudaSetDevice(ctx->deviceId);
  double t0 = nf_gpu_timer_sec();

  // Store geometry constants
  ctx->px = (float)px;
  ctx->gs = (float)gs;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      ctx->rotMatTilts[i*3+j] = (float)rotMatTilts[i][j];

  // Upload per-layer data
  float *h_Lsd_f = (float *)malloc(ctx->nLayers * sizeof(float));
  float *h_ybc_f = (float *)malloc(ctx->nLayers * sizeof(float));
  float *h_zbc_f = (float *)malloc(ctx->nLayers * sizeof(float));
  float *h_P0_f  = (float *)malloc(ctx->nLayers * 3 * sizeof(float));

  for (int i = 0; i < ctx->nLayers; i++) {
    h_Lsd_f[i] = (float)Lsd[i];
    h_ybc_f[i] = (float)ybc[i];
    h_zbc_f[i] = (float)zbc[i];

    // Compute P0 = RotMatTilts * [-Lsd, 0, 0]
    h_P0_f[i*3+0] = (float)(rotMatTilts[0][0] * (-Lsd[i]));
    h_P0_f[i*3+1] = (float)(rotMatTilts[1][0] * (-Lsd[i]));
    h_P0_f[i*3+2] = (float)(rotMatTilts[2][0] * (-Lsd[i]));
  }

  CUDA_CHECK(cudaMemcpy(ctx->d_Lsd, h_Lsd_f, ctx->nLayers * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_ybc, h_ybc_f, ctx->nLayers * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_zbc, h_zbc_f, ctx->nLayers * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_P0, h_P0_f, ctx->nLayers * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));

  free(h_Lsd_f);
  free(h_ybc_f);
  free(h_zbc_f);
  free(h_P0_f);

  // Build GPUSpot array directly from precomputed SpotsMat + Key.bin
  // h_nrSpots layout: [nSpots_0, startRow_0, nSpots_1, startRow_1, ...]
  // h_spotsMat layout: [y_0, z_0, omega_0, y_1, z_1, omega_1, ...]
  ctx->nOrientations = nOrientations;

  // Compute total spots and headers from Key.bin (no CalcDiffractionSpots needed!)
  int totalSpots = 0;
  GPUOrientHeader *h_headers = (GPUOrientHeader *)malloc(
      nOrientations * sizeof(GPUOrientHeader));

  for (int i = 0; i < nOrientations; i++) {
    int nSpotsThis = h_nrSpots[i * 2];
    h_headers[i].nSpots = nSpotsThis;
    h_headers[i].spotOffset = totalSpots;
    totalSpots += nSpotsThis;
  }

  ctx->totalSpots = totalSpots;
  printf("NF GPU: %d orientations, %d total spots (avg %.1f/orient)\n",
         nOrientations, totalSpots,
         nOrientations > 0 ? (float)totalSpots / nOrientations : 0.0f);

  // Fill GPUSpot array from SpotsMat — parallelized with OMP
  GPUSpot *h_spots = (GPUSpot *)malloc(totalSpots * sizeof(GPUSpot));

  #pragma omp parallel for schedule(dynamic, 1000)
  for (int i = 0; i < nOrientations; i++) {
    int nSpotsThis = h_nrSpots[i * 2];
    int startRow   = h_nrSpots[i * 2 + 1];
    int offset     = h_headers[i].spotOffset;

    for (int s = 0; s < nSpotsThis; s++) {
      int srcIdx = startRow + s;
      float ythis = (float)h_spotsMat[srcIdx * 3 + 0];
      float zthis = (float)h_spotsMat[srcIdx * 3 + 1];
      float omegaThis = (float)h_spotsMat[srcIdx * 3 + 2];

      int omeBin = (int)floor((-omegaStart + omegaThis) / omegaStep);
      int outOfBounds = (omeBin < 0 || omeBin >= ctx->nrFiles) ? 1 : 0;

      float omegaRad = (float)(omegaThis * M_PI / 180.0);
      h_spots[offset + s].y = ythis;
      h_spots[offset + s].z = zthis;
      h_spots[offset + s].sinOme = sinf(omegaRad);
      h_spots[offset + s].cosOme = cosf(omegaRad);
      h_spots[offset + s].refYpx = 0.0f;  // computed on GPU below
      h_spots[offset + s].refZpx = 0.0f;
      h_spots[offset + s].omeBin = omeBin;
      h_spots[offset + s].valid = outOfBounds ? 0 : 1;
    }

    // Sort this orientation's spots by omeBin for L2 locality
    // (adjacent spots access same omega frame → same 512KB bitfield region)
    qsort(&h_spots[offset], nSpotsThis, sizeof(GPUSpot),
          nf_gpu_spot_cmp_omebin);
  }

  // Upload to device
  CUDA_CHECK(cudaMalloc(&ctx->d_spots, totalSpots * sizeof(GPUSpot)));
  CUDA_CHECK(cudaMemcpy(ctx->d_spots, h_spots, totalSpots * sizeof(GPUSpot),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&ctx->d_orientHeaders,
                        nOrientations * sizeof(GPUOrientHeader)));
  CUDA_CHECK(cudaMemcpy(ctx->d_orientHeaders, h_headers,
                        nOrientations * sizeof(GPUOrientHeader),
                        cudaMemcpyHostToDevice));

  free(h_spots);
  free(h_headers);

  // Compute refYpx/refZpx ON GPU to match screening kernel's FMA arithmetic
  {
    // Upload RM and P0 (layer 0) to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_RM, ctx->rotMatTilts, 9 * sizeof(float)));
    float h_P0_0[3];
    h_P0_0[0] = (float)(rotMatTilts[0][0] * (-Lsd[0]));
    h_P0_0[1] = (float)(rotMatTilts[1][0] * (-Lsd[0]));
    h_P0_0[2] = (float)(rotMatTilts[2][0] * (-Lsd[0]));
    CUDA_CHECK(cudaMemcpyToSymbol(c_P0, h_P0_0, 3 * sizeof(float)));

    int bs = 256;
    int gs = (totalSpots + bs - 1) / bs;
    compute_ref_positions_kernel<<<gs, bs>>>(ctx->d_spots, totalSpots,
                                             ctx->px, (float)ybc[0], (float)zbc[0]);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  ctx->orientsUploaded = 1;
  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: orientations loaded and uploaded (%.2f s)\n", dt);

  return 0;
}

// ═════════════════════════════════════════════════════════════
//  PHASE 1: GPU SCREENING
// ═════════════════════════════════════════════════════════════

extern "C" int nf_gpu_screen(NFGPUContext *ctx,
                              const double *h_XGrains,
                              const double *h_YGrains,
                              int nVoxels,
                              double minFracOverlap,
                              NFGPUWinner **h_winners,
                              int *nWinners) {
  if (!ctx || !ctx->obsUploaded || !ctx->orientsUploaded) {
    fprintf(stderr, "NF GPU: data not uploaded before screening\n");
    return -1;
  }
  cudaSetDevice(ctx->deviceId);
  double t0 = nf_gpu_timer_sec();

  int nOrient = ctx->nOrientations;
  long long totalPairs = (long long)nVoxels * nOrient;

  printf("NF GPU: screening %d voxels × %d orientations = %lld pairs\n",
         nVoxels, nOrient, totalPairs);

  // Upload voxel data (convert double → float)
  float *h_XG = (float *)malloc(nVoxels * 3 * sizeof(float));
  float *h_YG = (float *)malloc(nVoxels * 3 * sizeof(float));
  for (int i = 0; i < nVoxels * 3; i++) {
    h_XG[i] = (float)h_XGrains[i];
    h_YG[i] = (float)h_YGrains[i];
  }

  float *d_XG, *d_YG;
  CUDA_CHECK(cudaMalloc(&d_XG, nVoxels * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_YG, nVoxels * 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_XG, h_XG, nVoxels * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_YG, h_YG, nVoxels * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));
  free(h_XG);
  free(h_YG);

  // Upload geometry to constant memory
  {
    float h_Lsd[MAX_LAYERS], h_ybc[MAX_LAYERS], h_zbc[MAX_LAYERS];
    float h_P0[MAX_LAYERS * 3];

    // Read from device arrays back to host (they were uploaded in upload_orientations)
    CUDA_CHECK(cudaMemcpy(h_Lsd, ctx->d_Lsd, ctx->nLayers * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ybc, ctx->d_ybc, ctx->nLayers * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_zbc, ctx->d_zbc, ctx->nLayers * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P0, ctx->d_P0, ctx->nLayers * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Lsd, h_Lsd, ctx->nLayers * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ybc, h_ybc, ctx->nLayers * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_zbc, h_zbc, ctx->nLayers * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_P0, h_P0, ctx->nLayers * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_RM, ctx->rotMatTilts, 9 * sizeof(float)));
  }

  // Allocate winner buffer
  ctx->maxWinners = (int)fmin((double)totalPairs, 10000000.0) + 10000;
  CUDA_CHECK(cudaMalloc(&ctx->d_winners,
                        ctx->maxWinners * sizeof(NFGPUWinner)));
  CUDA_CHECK(cudaMalloc(&ctx->d_nWinners, sizeof(int)));
  CUDA_CHECK(cudaMemset(ctx->d_nWinners, 0, sizeof(int)));

  // Auto-tune block size for maximum occupancy
  int blockSize, minGridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     screen_pairs_kernel, 0, 0);
  printf("NF GPU: auto-tuned blockSize=%d (minGrid=%d)\n", blockSize, minGridSize);

  // Frame batching disabled — overhead of binary search per batch outweighs L2 benefit
  // (A6000: 12 frames/batch = 120 iterations → too much loop overhead)
  int frameBatchSize = ctx->nrFiles;  // single batch = original simple loop
  double tKernel0 = nf_gpu_timer_sec();

  if (ctx->nLayers > 1) {
    // ─── TWO-PASS FUNNEL ───
    // Pass 1: screen with single layer (layer 0) — produces a superset of winners.
    // Pass 2: re-evaluate only pass-1 winners with ALL layers.
    // This halves the bitfield working set and gives ~2× kernel speedup for pass 1,
    // and pass 2 processes only O(1000) pairs instead of O(10^9).

    // --- Pass 1: single-layer screening ---
    int gridSize1 = (int)((totalPairs + blockSize - 1) / blockSize);

    // Upload layer-0-only geometry to constant memory
    {
      float h_Lsd1[1], h_ybc1[1], h_zbc1[1], h_P0_1[3];
      CUDA_CHECK(cudaMemcpy(h_Lsd1, ctx->d_Lsd, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_ybc1, ctx->d_ybc, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_zbc1, ctx->d_zbc, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_P0_1, ctx->d_P0, 3 * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpyToSymbol(c_Lsd, h_Lsd1, sizeof(float)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_ybc, h_ybc1, sizeof(float)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_zbc, h_zbc1, sizeof(float)));
      CUDA_CHECK(cudaMemcpyToSymbol(c_P0, h_P0_1, 3 * sizeof(float)));
    }

    screen_pairs_kernel<<<gridSize1, blockSize, 0, ctx->stream>>>(
        ctx->d_obsFlat,
        ctx->d_spots,
        ctx->d_orientHeaders,
        d_XG, d_YG,
        nVoxels, nOrient,
        1,  // nLayers = 1 for pass 1
        ctx->nrFiles,
        ctx->nrPixelsY, ctx->nrPixelsZ,
        frameBatchSize,
        ctx->px, ctx->gs,
        (float)(minFracOverlap * 0.9),  // slightly lower threshold for pass 1
        // (single-layer TotalPixels can increase when layer-1 OOB pixels
        //  are in-bounds on layer 0, lowering fracOverlap slightly)
        ctx->d_winners, ctx->d_nWinners, ctx->maxWinners,
        0);  // not diagonal

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    double dtPass1 = nf_gpu_timer_sec() - tKernel0;

    // Download pass-1 winners
    int nPass1 = 0;
    CUDA_CHECK(cudaMemcpy(&nPass1, ctx->d_nWinners, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (nPass1 > ctx->maxWinners) nPass1 = ctx->maxWinners;

    printf("NF GPU: pass 1 (layer 0 only) — %d candidates in %.2f s\n",
           nPass1, dtPass1);

    if (nPass1 > 0) {
      NFGPUWinner *h_pass1 = (NFGPUWinner *)malloc(nPass1 * sizeof(NFGPUWinner));
      CUDA_CHECK(cudaMemcpy(h_pass1, ctx->d_winners,
                            nPass1 * sizeof(NFGPUWinner),
                            cudaMemcpyDeviceToHost));

      // --- Pass 2: re-evaluate candidates with ALL layers ---
      // Build per-candidate voxel XG/YG arrays
      float *h_XG2 = (float *)malloc(nPass1 * 3 * sizeof(float));
      float *h_YG2 = (float *)malloc(nPass1 * 3 * sizeof(float));
      GPUOrientHeader *h_hdr2 = (GPUOrientHeader *)malloc(nPass1 * sizeof(GPUOrientHeader));

      // Batch-download all orient headers once (avoid N individual cudaMemcpy calls)
      GPUOrientHeader *h_allHeaders = (GPUOrientHeader *)malloc(
          ctx->nOrientations * sizeof(GPUOrientHeader));
      CUDA_CHECK(cudaMemcpy(h_allHeaders, ctx->d_orientHeaders,
                            ctx->nOrientations * sizeof(GPUOrientHeader),
                            cudaMemcpyDeviceToHost));

      // We need each pass-1 winner to be its own (voxel=0, orient=i) pair.
      // Re-pack: for each candidate, copy its voxel triangle and orient header.
      for (int i = 0; i < nPass1; i++) {
        int vI = h_pass1[i].voxelIdx;
        int oI = h_pass1[i].orientIdx;
        for (int k = 0; k < 3; k++) {
          h_XG2[i * 3 + k] = (float)h_XGrains[vI * 3 + k];
          h_YG2[i * 3 + k] = (float)h_YGrains[vI * 3 + k];
        }
        h_hdr2[i] = h_allHeaders[oI];
      }
      free(h_allHeaders);

      // Upload pass-2 data
      float *d_XG2, *d_YG2;
      GPUOrientHeader *d_hdr2;
      CUDA_CHECK(cudaMalloc(&d_XG2, nPass1 * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_YG2, nPass1 * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_hdr2, nPass1 * sizeof(GPUOrientHeader)));
      CUDA_CHECK(cudaMemcpy(d_XG2, h_XG2, nPass1 * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_YG2, h_YG2, nPass1 * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_hdr2, h_hdr2, nPass1 * sizeof(GPUOrientHeader),
                            cudaMemcpyHostToDevice));

      // Re-upload full geometry to constant memory
      {
        float h_Lsd[MAX_LAYERS], h_ybc[MAX_LAYERS], h_zbc[MAX_LAYERS];
        float h_P0[MAX_LAYERS * 3];
        CUDA_CHECK(cudaMemcpy(h_Lsd, ctx->d_Lsd, ctx->nLayers * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ybc, ctx->d_ybc, ctx->nLayers * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_zbc, ctx->d_zbc, ctx->nLayers * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_P0, ctx->d_P0, ctx->nLayers * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyToSymbol(c_Lsd, h_Lsd, ctx->nLayers * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_ybc, h_ybc, ctx->nLayers * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_zbc, h_zbc, ctx->nLayers * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_P0, h_P0, ctx->nLayers * 3 * sizeof(float)));
      }

      // --- Pass 2: linear 1:1 evaluation ---
      // Each of the nPass1 candidates gets its own thread.
      // diagonal=1 → thread i evaluates (voxel=i, orient=i) directly.
      // No N² grid, no off-diagonal waste, no buffer overflow.
      CUDA_CHECK(cudaMemset(ctx->d_nWinners, 0, sizeof(int)));

      // Ensure winner buffer can hold nPass1 results
      if (nPass1 > ctx->maxWinners) {
        cudaFree(ctx->d_winners);
        CUDA_CHECK(cudaMalloc(&ctx->d_winners,
                              (nPass1 + 1000) * sizeof(NFGPUWinner)));
        ctx->maxWinners = nPass1 + 1000;
      }

      int gridSize2 = (nPass1 + blockSize - 1) / blockSize;
      screen_pairs_kernel<<<gridSize2, blockSize, 0, ctx->stream>>>(
          ctx->d_obsFlat,
          ctx->d_spots,
          d_hdr2,
          d_XG2, d_YG2,
          nPass1,  // nVoxels = nPass1 (used as total count in diagonal mode)
          nPass1,  // nOrientations (unused in diagonal mode)
          ctx->nLayers, ctx->nrFiles,
          ctx->nrPixelsY, ctx->nrPixelsZ,
          frameBatchSize,
          ctx->px, ctx->gs,
          (float)minFracOverlap,
          ctx->d_winners, ctx->d_nWinners, ctx->maxWinners,
          1);  // diagonal=1: 1:1 candidate evaluation

      CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
      double dtPass2 = nf_gpu_timer_sec() - tKernel0 - dtPass1;
      printf("NF GPU: pass 2 (all %d layers) — verified %d candidates in %.2f s\n",
             ctx->nLayers, nPass1, dtPass2);

      // Download pass-2 winners — already 1:1, just remap to original indices
      int nPass2 = 0;
      CUDA_CHECK(cudaMemcpy(&nPass2, ctx->d_nWinners, sizeof(int),
                            cudaMemcpyDeviceToHost));
      if (nPass2 > ctx->maxWinners) nPass2 = ctx->maxWinners;

      NFGPUWinner *h_pass2Raw = (NFGPUWinner *)malloc(nPass2 * sizeof(NFGPUWinner));
      CUDA_CHECK(cudaMemcpy(h_pass2Raw, ctx->d_winners,
                            nPass2 * sizeof(NFGPUWinner),
                            cudaMemcpyDeviceToHost));

      // Remap to original indices (pass2 voxelIdx/orientIdx are pass1 indices)
      NFGPUWinner *h_pass2 = (NFGPUWinner *)malloc(nPass2 * sizeof(NFGPUWinner));
      for (int i = 0; i < nPass2; i++) {
        int p1Idx = h_pass2Raw[i].voxelIdx;  // == orientIdx in diagonal mode
        h_pass2[i].voxelIdx = h_pass1[p1Idx].voxelIdx;
        h_pass2[i].orientIdx = h_pass1[p1Idx].orientIdx;
        h_pass2[i].fracOverlap = h_pass2Raw[i].fracOverlap;
      }

      *h_winners = h_pass2;
      *nWinners = nPass2;
      free(h_pass2Raw);

      cudaFree(d_XG2); cudaFree(d_YG2); cudaFree(d_hdr2);
      free(h_XG2); free(h_YG2); free(h_hdr2); free(h_pass1);

    } else {
      // No pass-1 winners → no pass-2 needed
      *h_winners = (NFGPUWinner *)malloc(1);
      *nWinners = 0;
    }

    double dtKernel = nf_gpu_timer_sec() - tKernel0;
    printf("NF GPU: screening kernel completed in %.2f s\n", dtKernel);

  } else {
    // ─── Single layer: no funnel needed, direct screening ───
    int gridSize = (int)((totalPairs + blockSize - 1) / blockSize);

    screen_pairs_kernel<<<gridSize, blockSize, 0, ctx->stream>>>(
        ctx->d_obsFlat,
        ctx->d_spots,
        ctx->d_orientHeaders,
        d_XG, d_YG,
        nVoxels, nOrient,
        ctx->nLayers, ctx->nrFiles,
        ctx->nrPixelsY, ctx->nrPixelsZ,
        frameBatchSize,
        ctx->px, ctx->gs,
        (float)minFracOverlap,
        ctx->d_winners, ctx->d_nWinners, ctx->maxWinners,
        0);  // not diagonal

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    double dtKernel = nf_gpu_timer_sec() - tKernel0;
    printf("NF GPU: screening kernel completed in %.2f s\n", dtKernel);

    // Download winner count
    int h_nW = 0;
    CUDA_CHECK(cudaMemcpy(&h_nW, ctx->d_nWinners, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_nW > ctx->maxWinners) h_nW = ctx->maxWinners;

    *h_winners = (NFGPUWinner *)malloc(h_nW * sizeof(NFGPUWinner));
    CUDA_CHECK(cudaMemcpy(*h_winners, ctx->d_winners,
                          h_nW * sizeof(NFGPUWinner),
                          cudaMemcpyDeviceToHost));
    *nWinners = h_nW;
  }

  // Cleanup
  cudaFree(d_XG);
  cudaFree(d_YG);
  cudaFree(ctx->d_winners);   ctx->d_winners = NULL;
  cudaFree(ctx->d_nWinners);  ctx->d_nWinners = NULL;

  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: screening complete — %d winners from %lld pairs "
         "(%.4f%% pass rate, %.2f s total)\n",
         *nWinners, totalPairs, 100.0 * *nWinners / totalPairs, dt);

  return 0;
}


// ═════════════════════════════════════════════════════════════
//  PHASE 2: GPU NM FITTING — Host functions
// ═════════════════════════════════════════════════════════════

extern "C" int nf_gpu_upload_hkls(NFGPUContext *ctx,
                                   const double hkls[][4],
                                   const double *Gs,
                                   int n_hkls,
                                   double excludePoleAngle,
                                   double omegaStart, double omegaStep,
                                   const double omegaRanges[][2],
                                   int nOmeRanges,
                                   const double boxSizes[][4]) {
  if (n_hkls > GPU_MAX_HKLS) {
    fprintf(stderr, "NF GPU: n_hkls=%d exceeds GPU_MAX_HKLS=%d\n", n_hkls, GPU_MAX_HKLS);
    return -1;
  }

  // Convert HKLs to float and upload to global device memory
  float *h_hkls = (float*)malloc(n_hkls * 4 * sizeof(float));
  float *h_Gs = (float*)malloc(n_hkls * sizeof(float));
  for (int i = 0; i < n_hkls; i++) {
    h_hkls[i*4+0] = (float)hkls[i][0];
    h_hkls[i*4+1] = (float)hkls[i][1];
    h_hkls[i*4+2] = (float)hkls[i][2];
    h_hkls[i*4+3] = (float)hkls[i][3];
    h_Gs[i] = (float)Gs[i];
  }

  // Free old device buffers if re-uploading
  if (ctx->d_hkls) cudaFree(ctx->d_hkls);
  if (ctx->d_Gs)   cudaFree(ctx->d_Gs);

  CUDA_CHECK(cudaMalloc(&ctx->d_hkls, n_hkls * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ctx->d_Gs, n_hkls * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(ctx->d_hkls, h_hkls, n_hkls * 4 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_Gs, h_Gs, n_hkls * sizeof(float), cudaMemcpyHostToDevice));
  ctx->n_hkls = n_hkls;
  ctx->hklsUploaded = 1;
  free(h_hkls);
  free(h_Gs);

  float h_excl = (float)excludePoleAngle;
  float h_omeStart = (float)omegaStart;
  float h_omeStep = (float)omegaStep;
  CUDA_CHECK(cudaMemcpyToSymbol(c_excludePoleAngle, &h_excl, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_omegaStart, &h_omeStart, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_omegaStep, &h_omeStep, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_nOmeRanges, &nOmeRanges, sizeof(int)));

  float h_omeRanges[20][2], h_boxSizes[20][4];
  for (int i = 0; i < nOmeRanges; i++) {
    h_omeRanges[i][0] = (float)omegaRanges[i][0];
    h_omeRanges[i][1] = (float)omegaRanges[i][1];
    h_boxSizes[i][0] = (float)boxSizes[i][0];
    h_boxSizes[i][1] = (float)boxSizes[i][1];
    h_boxSizes[i][2] = (float)boxSizes[i][2];
    h_boxSizes[i][3] = (float)boxSizes[i][3];
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_omegaRanges, h_omeRanges, nOmeRanges * 2 * sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_boxSizes, h_boxSizes, nOmeRanges * 4 * sizeof(float)));

  printf("NF GPU: HKL data uploaded (%d reflections, %.1f KB global)\n",
         n_hkls, (n_hkls * 4 + n_hkls) * sizeof(float) / 1024.0);
  return 0;
}

/// Helper: convert orientation matrix (row-major double[9]) to Euler angles (degrees)
static void orientMat9ToEuler(const double *m, double euler[3]) {
  // Same as OrientMat2Euler in FitPosOrStrainsOMP.c
  double phi, psi, theta, sph;
  if (fabs(m[8] - 1.0) < 1e-12) {
    phi = 0;
  } else {
    phi = acos(m[8]);
  }
  sph = sin(phi);
  if (fabs(sph) < 1e-12) {
    psi = 0.0;
    if (fabs(m[8] - 1.0) < 1e-12) {
      theta = atan2(m[3], m[0]);
    } else {
      theta = atan2(-m[3], m[0]);
    }
  } else {
    psi = atan2(m[2] / sph, -m[5] / sph);
    theta = atan2(m[6] / sph, m[7] / sph);
  }
  euler[0] = psi * rad2deg;
  euler[1] = phi * rad2deg;
  euler[2] = theta * rad2deg;
}

extern "C" int nf_gpu_fit(NFGPUContext *ctx,
                           const NFGPUWinner *winners,
                           int nWinners,
                           const double *h_XGrains,
                           const double *h_YGrains,
                           double eulerTol,
                           const double *h_orientMatrix,
                           NFGPUFitResult **fitResults,
                           int *nFitResults) {
  if (nWinners == 0) {
    *fitResults = NULL;
    *nFitResults = 0;
    return 0;
  }

  double t0 = nf_gpu_timer_sec();

  // Prepare per-job arrays on host
  int nJobs = nWinners;
  float *h_startEulers = (float*)malloc(nJobs * 3 * sizeof(float));
  float *h_voxXG = (float*)malloc(nJobs * 3 * sizeof(float));
  float *h_voxYG = (float*)malloc(nJobs * 3 * sizeof(float));

  for (int j = 0; j < nJobs; j++) {
    int oriIdx = winners[j].orientIdx;
    int voxIdx = winners[j].voxelIdx;

    // Normalize orient matrix (same as CPU NormalizeMat: divide by cbrt(det))
    // This ensures det=1 before extracting Euler angles via acos(m[8]).
    double OMRaw[9], OMNorm[9];
    for (int k = 0; k < 9; k++) {
      OMRaw[k] = h_orientMatrix[oriIdx * 9 + k];
      if (OMRaw[k] == -0.0) OMRaw[k] = 0.0;
    }
    double det = OMRaw[0] * (OMRaw[4]*OMRaw[8] - OMRaw[5]*OMRaw[7])
               - OMRaw[1] * (OMRaw[3]*OMRaw[8] - OMRaw[5]*OMRaw[6])
               + OMRaw[2] * (OMRaw[3]*OMRaw[7] - OMRaw[4]*OMRaw[6]);
    double scale = cbrt(det);
    for (int k = 0; k < 9; k++) OMNorm[k] = OMRaw[k] / scale;

    // Convert normalized orient matrix to Euler angles for initial guess
    double euler[3];
    orientMat9ToEuler(OMNorm, euler);

    // NM works in radians
    h_startEulers[j * 3 + 0] = (float)(euler[0] * deg2rad);
    h_startEulers[j * 3 + 1] = (float)(euler[1] * deg2rad);
    h_startEulers[j * 3 + 2] = (float)(euler[2] * deg2rad);

    // Voxel corners
    for (int k = 0; k < 3; k++) {
      h_voxXG[j * 3 + k] = (float)h_XGrains[voxIdx * 3 + k];
      h_voxYG[j * 3 + k] = (float)h_YGrains[voxIdx * 3 + k];
    }

    // Debug: orient matrix round-trip for first job
    if (j == 0) {
      printf("NF GPU FIT DEBUG job 0: oriIdx=%d voxIdx=%d det=%.9f scale=%.9f\n", oriIdx, voxIdx, det, scale);
      printf("  OMRaw:  [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n",
             OMRaw[0],OMRaw[1],OMRaw[2],OMRaw[3],OMRaw[4],OMRaw[5],OMRaw[6],OMRaw[7],OMRaw[8]);
      printf("  OMNorm: [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n",
             OMNorm[0],OMNorm[1],OMNorm[2],OMNorm[3],OMNorm[4],OMNorm[5],OMNorm[6],OMNorm[7],OMNorm[8]);
      printf("  Euler(deg): psi=%.6f phi=%.6f theta=%.6f\n", euler[0], euler[1], euler[2]);
      // Reconstruct orient matrix from Euler to check round-trip (host-side)
      double psi_r=euler[0]*deg2rad, phi_r=euler[1]*deg2rad, theta_r=euler[2]*deg2rad;
      double cps=cos(psi_r), sps=sin(psi_r), cph=cos(phi_r), sph=sin(phi_r), cth=cos(theta_r), sth=sin(theta_r);
      double recon[9] = {
        cth*cps - sth*cph*sps, -cth*cph*sps - sth*cps, sph*sps,
        cth*sps + sth*cph*cps,  cth*cph*cps - sth*sps, -sph*cps,
        sth*sph,                cth*sph,                 cph
      };
      printf("  Recon:  [%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f]\n",
             recon[0],recon[1],recon[2],recon[3],recon[4],recon[5],recon[6],recon[7],recon[8]);
      printf("  XG=(%.4f,%.4f,%.4f) YG=(%.4f,%.4f,%.4f)\n",
             h_voxXG[j*3+0], h_voxXG[j*3+1], h_voxXG[j*3+2],
             h_voxYG[j*3+0], h_voxYG[j*3+1], h_voxYG[j*3+2]);

      // Print precomputed screening spots for this orientation
      // These are the known-good spot positions that the screening kernel used
      GPUOrientHeader h_hdr;
      cudaMemcpy(&h_hdr, &ctx->d_orientHeaders[oriIdx], sizeof(GPUOrientHeader), cudaMemcpyDeviceToHost);
      int nShow = (h_hdr.nSpots < 5) ? h_hdr.nSpots : 5;
      GPUSpot h_dbgSpots[5];
      cudaMemcpy(h_dbgSpots, &ctx->d_spots[h_hdr.spotOffset], nShow * sizeof(GPUSpot), cudaMemcpyDeviceToHost);
      printf("  Screening spots for orient %d (first %d of %d):\n", oriIdx, nShow, h_hdr.nSpots);
      for (int s = 0; s < nShow; s++) {
        if (!h_dbgSpots[s].valid) continue;
        float ome = atan2f(h_dbgSpots[s].sinOme, h_dbgSpots[s].cosOme) * (float)rad2deg;
        printf("    spot[%d]: y=%.2f z=%.2f ome=%.2f omeBin=%d refYpx=%.2f refZpx=%.2f\n",
               s, h_dbgSpots[s].y, h_dbgSpots[s].z, ome, h_dbgSpots[s].omeBin,
               h_dbgSpots[s].refYpx, h_dbgSpots[s].refZpx);
      }
    }
  }

  // Per-job bounds: euler ± eulerTol (matching CPU NLOPT setup)
  // eulerTol is in degrees, convert to radians for the bounds
  float tolRad = (float)(eulerTol * M_PI / 180.0);  // typically 2° → 0.0349 rad
  float *h_lb = (float*)malloc(nJobs * 3 * sizeof(float));
  float *h_ub = (float*)malloc(nJobs * 3 * sizeof(float));
  for (int j = 0; j < nJobs; j++) {
    for (int d = 0; d < 3; d++) {
      h_lb[j * 3 + d] = h_startEulers[j * 3 + d] - tolRad;
      h_ub[j * 3 + d] = h_startEulers[j * 3 + d] + tolRad;
    }
  }

  // Upload to device
  float *d_startEulers, *d_voxXG, *d_voxYG;
  float *d_lb, *d_ub, *d_results, *d_fvals;
  float *d_RM, *d_P0;
  CUDA_CHECK(cudaMalloc(&d_startEulers, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_voxXG, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_voxYG, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lb, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ub, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_results, nJobs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fvals, nJobs * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_RM, 9 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_P0, 3 * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_startEulers, h_startEulers, nJobs * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_voxXG, h_voxXG, nJobs * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_voxYG, h_voxYG, nJobs * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lb, h_lb, nJobs * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ub, h_ub, nJobs * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_RM, ctx->rotMatTilts, 9 * sizeof(float), cudaMemcpyHostToDevice));

  // P0 layer 0
  float h_P0[3];
  CUDA_CHECK(cudaMemcpy(h_P0, ctx->d_P0, 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(d_P0, h_P0, 3 * sizeof(float), cudaMemcpyHostToDevice));

  // Lsd0, ybc0, zbc0 from constant memory
  float h_Lsd0, h_ybc0, h_zbc0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_Lsd0, c_Lsd, sizeof(float)));
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_ybc0, c_ybc, sizeof(float)));
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_zbc0, c_zbc, sizeof(float)));

  // Launch fitting kernel — match CPU NM parameters
  int blockSize = 64;  // Each thread does heavy work, smaller blocks
  int gridSize = (nJobs + blockSize - 1) / blockSize;
  float ftol = 1e-5f;    // matches CPU ftol_rel
  int maxIter = 1000;     // CPU uses 5000 evals; each NM iter uses ~4 evals
  float initStep = 0.25f; // 25% of bound range = 0.25 * 2*tolRad ≈ 0.5°

  printf("NF GPU: Phase 2 fitting — %d jobs, blockSize=%d\n", nJobs, blockSize);

  nm_fit_kernel<<<gridSize, blockSize, 0, ctx->stream>>>(
      nJobs, d_startEulers, d_lb, d_ub,
      d_voxXG, d_voxYG,
      ctx->d_obsFlat,
      h_Lsd0, h_ybc0, h_zbc0,
      d_RM, d_P0,
      ctx->nLayers, ctx->nrFiles, ctx->nrPixelsY, ctx->nrPixelsZ,
      ctx->px, ctx->gs,
      ftol, maxIter, initStep,
      ctx->d_hkls, ctx->d_Gs, ctx->n_hkls,
      d_results, d_fvals);

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
  double dtKernel = nf_gpu_timer_sec() - t0;

  // Download results
  float *h_results = (float*)malloc(nJobs * 3 * sizeof(float));
  float *h_fvals = (float*)malloc(nJobs * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_results, d_results, nJobs * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fvals, d_fvals, nJobs * sizeof(float), cudaMemcpyDeviceToHost));

  // Build output
  NFGPUFitResult *results = (NFGPUFitResult*)malloc(nJobs * sizeof(NFGPUFitResult));
  for (int j = 0; j < nJobs; j++) {
    results[j].voxelIdx = winners[j].voxelIdx;
    results[j].eulerA = h_results[j * 3 + 0] * (float)rad2deg;
    results[j].eulerB = h_results[j * 3 + 1] * (float)rad2deg;
    results[j].eulerC = h_results[j * 3 + 2] * (float)rad2deg;
    results[j].fracOverlap = 1.0f - h_fvals[j];  // NM minimizes 1-frac
  }

  *fitResults = results;
  *nFitResults = nJobs;

  printf("NF GPU: Phase 2 fitting complete — %d results in %.2f s\n", nJobs, dtKernel);

  // Cleanup
  free(h_startEulers); free(h_voxXG); free(h_voxYG); free(h_lb); free(h_ub);
  free(h_results); free(h_fvals);
  cudaFree(d_startEulers); cudaFree(d_voxXG); cudaFree(d_voxYG);
  cudaFree(d_lb); cudaFree(d_ub);
  cudaFree(d_results); cudaFree(d_fvals);
  cudaFree(d_RM); cudaFree(d_P0);

  return 0;
}

#endif /* ENABLE_CUDA */
