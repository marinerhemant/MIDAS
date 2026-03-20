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
  int   omeBin;   ///< Omega frame index
  int   valid;    ///< 1 if spot is in range, 0 if out-of-bounds
};

/// Per-orientation header: number of spots and offset into spot array.
struct GPUOrientHeader {
  int nSpots;
  int spotOffset;
};

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
//
//  Matches CPU CalcFracOverlap exactly:
//  1. For each spot: compute 3-corner displacement -> 3 pixel positions
//  2. Subtract reference (undisplaced spot) to get relative triangle
//  3. Rasterize triangle (CalcPixels2-equivalent)
//  4. For each rasterized pixel: check ALL layers with scaling
//  5. Accumulate OverlapPixels / TotalPixels
//  6. If FracOverlap >= threshold: atomically append winner
// ═════════════════════════════════════════════════════════════
//  Constant memory for geometry (broadcast to all threads)
// ═════════════════════════════════════════════════════════════
#define MAX_LAYERS 8
__constant__ float c_Lsd[MAX_LAYERS];
__constant__ float c_ybc[MAX_LAYERS];
__constant__ float c_zbc[MAX_LAYERS];
__constant__ float c_P0[MAX_LAYERS * 3];
__constant__ float c_RM[9];         // rotMatTilts flattened
__constant__ float c_layerScaleY[MAX_LAYERS]; // precomputed (refY - ybc0) * Lsd[l]/Lsd0 factor
__constant__ float c_layerScaleZ[MAX_LAYERS]; // same for Z

// ═════════════════════════════════════════════════════════════
//  MAIN SCREENING KERNEL: one thread = one (voxel, orientation) pair
//
//  Optimizations applied:
//  1. Inline pixel processing (no local arrays)
//  2. __ldg() for read-only global memory
//  3. Constant memory for geometry
//  4. Early exit when remaining spots can't reach threshold
//  5. Precomputed layer scaling
//  6. Break on first layer miss
// ═════════════════════════════════════════════════════════════

__global__ void screen_pairs_kernel(
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
    float px, float gs,
    float minFracOverlap,
    NFGPUWinner *winners,
    int *nWinners,
    int maxWinners) {

  // Linearised thread ID → (oriIdx, voxIdx)
  // Adjacent threads process different voxels with the SAME orientation
  // → same GPUSpot data → excellent L1/L2 cache reuse
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long totalPairs = (long long)nVoxels * nOrientations;
  if (tid >= totalPairs) return;

  int oriIdx = tid / nVoxels;
  int voxIdx = tid % nVoxels;

  // Load rotation matrix from constant memory into registers
  float RM[9];
  #pragma unroll
  for (int i = 0; i < 9; i++) RM[i] = c_RM[i];

  // Load voxel corners (different per thread — from global via __ldg)
  float XG[3], YG[3];
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    XG[k] = __ldg(&voxXG[voxIdx * 3 + k]);
    YG[k] = __ldg(&voxYG[voxIdx * 3 + k]);
  }

  // Load layer-0 geometry from constant memory
  float Lsd0 = c_Lsd[0];
  float ybc0 = c_ybc[0];
  float zbc0 = c_zbc[0];
  float P0_0[3] = { c_P0[0], c_P0[1], c_P0[2] };

  GPUOrientHeader hdr = headers[oriIdx];

  int OverlapPixels = 0;
  int TotalPixels = 0;
  int totalSpotsThisOri = hdr.nSpots;
  int spotsProcessed = 0;

  // Iterate over all spots for this orientation
  for (int s = 0; s < totalSpotsThisOri; s++) {
    // Early termination: if remaining spots can't reach threshold
    // Max possible additional overlap = remaining spots * ~5 pixels each (generous)
    // If (OverlapPixels + remainingMax) / (TotalPixels + remainingMax) < threshold, exit
    if (TotalPixels > 0 && spotsProcessed > 20) {
      int remaining = totalSpotsThisOri - s;
      int maxRemaining = remaining * 5;  // ~5 pixels per spot max
      // If even perfect overlap on all remaining can't reach threshold
      if ((float)(OverlapPixels + maxRemaining) / (float)(TotalPixels + maxRemaining)
          < minFracOverlap) {
        break;
      }
    }

    GPUSpot spot = spots[hdr.spotOffset + s];
    if (!spot.valid) continue;
    spotsProcessed++;

    float ythis = spot.y;
    float zthis = spot.z;
    int omeBin = spot.omeBin;
    float sinOme = spot.sinOme;
    float cosOme = spot.cosOme;

    // Compute displaced pixel positions for 3 triangle corners
    float YZSpotsT[3][2];  // Absolute pixel coords
    int oob = 0;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      float Displ_Y, Displ_Z;
      gpu_displacement(XG[k], YG[k], Lsd0, ythis, zthis, sinOme, cosOme,
                       &Displ_Y, &Displ_Z);

      // RotMatTilts × [0, Displ_Y, Displ_Z] = P1
      float P1x = RM[0*3+1] * Displ_Y + RM[0*3+2] * Displ_Z;
      float P1y = RM[1*3+1] * Displ_Y + RM[1*3+2] * Displ_Z;
      float P1z = RM[2*3+1] * Displ_Y + RM[2*3+2] * Displ_Z;

      float ABCx = P1x - P0_0[0];
      float ABCy = P1y - P0_0[1];
      float ABCz = P1z - P0_0[2];

      float invABCx = 1.0f / ABCx;
      float outY = P0_0[1] - ABCy * P0_0[0] * invABCx;
      float outZ = P0_0[2] - ABCz * P0_0[0] * invABCx;

      YZSpotsT[k][0] = outY / px + ybc0;
      YZSpotsT[k][1] = outZ / px + zbc0;

      if (YZSpotsT[k][0] > nrPixelsY || YZSpotsT[k][0] < 0 ||
          YZSpotsT[k][1] > nrPixelsZ || YZSpotsT[k][1] < 0) {
        oob = 1;
        break;
      }
    }
    if (oob) continue;

    // Compute undisplaced reference
    float refP1x = RM[0*3+1] * ythis + RM[0*3+2] * zthis;
    float refP1y = RM[1*3+1] * ythis + RM[1*3+2] * zthis;
    float refP1z = RM[2*3+1] * ythis + RM[2*3+2] * zthis;

    float refABCx = refP1x - P0_0[0];
    float refInvABCx = 1.0f / refABCx;
    float refOutY = P0_0[1] - (refP1y - P0_0[1]) * P0_0[0] * refInvABCx;
    float refOutZ = P0_0[2] - (refP1z - P0_0[2]) * P0_0[0] * refInvABCx;
    float refYpx = refOutY / px + ybc0;
    float refZpx = refOutZ / px + zbc0;

    // Precompute per-layer base positions for this spot
    float layerBaseY[MAX_LAYERS], layerBaseZ[MAX_LAYERS];
    for (int l = 0; l < nLayers; l++) {
      float scale = c_Lsd[l] / Lsd0;
      layerBaseY[l] = floorf((refYpx - ybc0) * scale + c_ybc[l]);
      layerBaseZ[l] = floorf((refZpx - zbc0) * scale + c_zbc[l]);
    }

    // Relative triangle: subtract reference
    float YZSpots[3][2];
    #pragma unroll
    for (int l = 0; l < 3; l++) {
      YZSpots[l][0] = YZSpotsT[l][0] - refYpx;
      YZSpots[l][1] = YZSpotsT[l][1] - refZpx;
    }

    // Inline pixel processing: rasterize + check bitfield in one pass
    // No local arrays needed
    if (gs * 2.0f > px) {
      // Multi-pixel: edge-function rasterization
      int e0y = (int)roundf(YZSpots[0][0]), e0z = (int)roundf(YZSpots[0][1]);
      int e1y = (int)roundf(YZSpots[1][0]), e1z = (int)roundf(YZSpots[1][1]);
      int e2y = (int)roundf(YZSpots[2][0]), e2z = (int)roundf(YZSpots[2][1]);

      int minY = min(e0y, min(e1y, e2y));
      int maxY = max(e0y, max(e1y, e2y));
      int minZ = min(e0z, min(e1z, e2z));
      int maxZ = max(e0z, max(e1z, e2z));

      // For distance test
      float edgeYs[3] = {(float)e0y, (float)e1y, (float)e2y};
      float edgeZs[3] = {(float)e0z, (float)e1z, (float)e2z};

      for (int pz = minZ; pz <= maxZ; pz++) {
        for (int py = minY; py <= maxY; py++) {
          // Edge function test
          int w0_ = (e1z - e2z) * (py - e1y) + (e2y - e1y) * (pz - e1z);
          int w1_ = (e2z - e0z) * (py - e2y) + (e0y - e2y) * (pz - e2z);
          int w2_ = (e0z - e1z) * (py - e0y) + (e1y - e0y) * (pz - e0z);

          int inside = (w0_ >= 0 && w1_ >= 0 && w2_ >= 0);

          if (!inside) {
            // Distance-to-edge test
            for (int e = 0; e < 3 && !inside; e++) {
              int i1 = (e + 1) % 3;
              float dx = edgeYs[i1] - edgeYs[e];
              float dz = edgeZs[i1] - edgeZs[e];
              float num = dx * (pz - edgeZs[e]) - dz * (py - edgeYs[e]);
              float den = dz * dz + dx * dx;
              if (den > 0 && (num * num) / den < 0.9801f)
                inside = 1;
            }
          }

          if (inside) {
            // Immediately check bitfield (inline — no storage needed)
            int allFound = 1;
            int pixOob = 0;

            for (int layer = 0; layer < nLayers; layer++) {
              int MultY = (int)layerBaseY[layer] + py;
              int MultZ = (int)layerBaseZ[layer] + pz;

              if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
                pixOob = 1;
                break;
              }

              long long binNr = (long long)layer * nrFiles * nrPixelsY * nrPixelsZ
                              + (long long)omeBin * nrPixelsY * nrPixelsZ
                              + (long long)MultY * nrPixelsZ
                              + MultZ;

              if (!gpu_test_bit(obsFlat, binNr)) {
                allFound = 0;
                break;  // No need to check remaining layers
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
      // Single pixel: centroid
      int py = (int)roundf((YZSpots[0][0] + YZSpots[1][0] + YZSpots[2][0]) / 3.0f);
      int pz = (int)roundf((YZSpots[0][1] + YZSpots[1][1] + YZSpots[2][1]) / 3.0f);

      int allFound = 1;
      int pixOob = 0;
      for (int layer = 0; layer < nLayers; layer++) {
        int MultY = (int)layerBaseY[layer] + py;
        int MultZ = (int)layerBaseZ[layer] + pz;

        if (MultY >= nrPixelsY || MultY < 0 || MultZ >= nrPixelsZ || MultZ < 0) {
          pixOob = 1;
          break;
        }

        long long binNr = (long long)layer * nrFiles * nrPixelsY * nrPixelsZ
                        + (long long)omeBin * nrPixelsY * nrPixelsZ
                        + (long long)MultY * nrPixelsZ
                        + MultZ;

        if (!gpu_test_bit(obsFlat, binNr)) {
          allFound = 0;
          break;
        }
      }

      if (!pixOob) {
        if (allFound) OverlapPixels++;
        TotalPixels++;
      }
    }
  }

  // Compute FracOverlap and check threshold
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

  long long totalBits = (long long)nLayers * nrFiles * nrPixelsY * nrPixelsZ;
  printf("NF GPU: context initialised — %dx%d detector, %d frames, %d layers\n",
         nrPixelsY, nrPixelsZ, nrFiles, nLayers);
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
      h_spots[offset + s].omeBin = omeBin;
      h_spots[offset + s].valid = outOfBounds ? 0 : 1;
    }
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

  // Launch per-pair kernel
  int blockSize = 256;
  int gridSize = (int)((totalPairs + blockSize - 1) / blockSize);

  double tKernel0 = nf_gpu_timer_sec();

  screen_pairs_kernel<<<gridSize, blockSize, 0, ctx->stream>>>(
      ctx->d_obsFlat,
      ctx->d_spots,
      ctx->d_orientHeaders,
      d_XG, d_YG,
      nVoxels, nOrient,
      ctx->nLayers, ctx->nrFiles,
      ctx->nrPixelsY, ctx->nrPixelsZ,
      ctx->px, ctx->gs,
      (float)minFracOverlap,
      ctx->d_winners, ctx->d_nWinners, ctx->maxWinners);

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
  double dtKernel = nf_gpu_timer_sec() - tKernel0;
  printf("NF GPU: screening kernel completed in %.2f s\n", dtKernel);

  // Download winner count
  int h_nW = 0;
  CUDA_CHECK(cudaMemcpy(&h_nW, ctx->d_nWinners, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (h_nW > ctx->maxWinners) {
    printf("NF GPU: WARNING: winner buffer overflow (%d > %d), "
           "results truncated\n", h_nW, ctx->maxWinners);
    h_nW = ctx->maxWinners;
  }

  // Download winners
  *h_winners = (NFGPUWinner *)malloc(h_nW * sizeof(NFGPUWinner));
  CUDA_CHECK(cudaMemcpy(*h_winners, ctx->d_winners,
                        h_nW * sizeof(NFGPUWinner),
                        cudaMemcpyDeviceToHost));
  *nWinners = h_nW;

  // Cleanup
  cudaFree(d_XG);
  cudaFree(d_YG);
  cudaFree(ctx->d_winners);   ctx->d_winners = NULL;
  cudaFree(ctx->d_nWinners);  ctx->d_nWinners = NULL;

  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: screening complete — %d winners from %lld pairs "
         "(%.4f%% pass rate, %.2f s total)\n",
         h_nW, totalPairs, 100.0 * h_nW / totalPairs, dt);

  return 0;
}

#endif /* ENABLE_CUDA */
