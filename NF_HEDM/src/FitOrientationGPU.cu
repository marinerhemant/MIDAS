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
    int maxWinners) {

  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long totalPairs = (long long)nVoxels * nOrientations;
  if (tid >= totalPairs) return;

  int oriIdx = tid / nVoxels;
  int voxIdx = tid % nVoxels;

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

  // Compute frame batch size from L2 cache size
  // Each omega frame = nrPixelsY * nrPixelsZ bits = (nrPixelsY * nrPixelsZ / 8) bytes
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, ctx->deviceId);
  int l2CacheBytes = prop.l2CacheSize;  // L2 cache size in bytes
  int bytesPerFrame = (ctx->nrPixelsY * ctx->nrPixelsZ) / 8;  // bits→bytes
  int frameBatchSize = l2CacheBytes / bytesPerFrame;
  if (frameBatchSize < 1) frameBatchSize = 1;
  if (frameBatchSize > ctx->nrFiles) frameBatchSize = ctx->nrFiles;
  printf("NF GPU: L2=%d MB, frameBatchSize=%d (of %d frames)\n",
         l2CacheBytes / (1024*1024), frameBatchSize, ctx->nrFiles);
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
        ctx->d_winners, ctx->d_nWinners, ctx->maxWinners);

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

      // Reset winner count and launch pass 2
      // Diagonal approach: nVoxels=nPass1, nOrientations=nPass1
      // Candidate i is processed by the diagonal thread where oriIdx=i AND voxIdx=i
      // For 541 candidates → 541^2=293K threads — trivially fast on GPU
      CUDA_CHECK(cudaMemset(ctx->d_nWinners, 0, sizeof(int)));

      long long pass2Pairs = (long long)nPass1 * nPass1;
      int gridSize2 = (int)((pass2Pairs + blockSize - 1) / blockSize);
      screen_pairs_kernel<<<gridSize2, blockSize, 0, ctx->stream>>>(
          ctx->d_obsFlat,
          ctx->d_spots,
          d_hdr2,
          d_XG2, d_YG2,
          nPass1,  // nVoxels = nPass1
          nPass1,  // nOrientations = nPass1
          ctx->nLayers, ctx->nrFiles,
          ctx->nrPixelsY, ctx->nrPixelsZ,
          frameBatchSize,
          ctx->px, ctx->gs,
          (float)minFracOverlap,
          ctx->d_winners, ctx->d_nWinners, ctx->maxWinners);

      CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
      double dtPass2 = nf_gpu_timer_sec() - tKernel0 - dtPass1;
      printf("NF GPU: pass 2 (all %d layers) — verified %d candidates in %.2f s\n",
             ctx->nLayers, nPass1, dtPass2);

      // Download pass-2 winners and filter to diagonal only (oriIdx == voxIdx)
      int nPass2Raw = 0;
      CUDA_CHECK(cudaMemcpy(&nPass2Raw, ctx->d_nWinners, sizeof(int),
                            cudaMemcpyDeviceToHost));
      if (nPass2Raw > ctx->maxWinners) nPass2Raw = ctx->maxWinners;

      NFGPUWinner *h_pass2Raw = (NFGPUWinner *)malloc(nPass2Raw * sizeof(NFGPUWinner));
      CUDA_CHECK(cudaMemcpy(h_pass2Raw, ctx->d_winners,
                            nPass2Raw * sizeof(NFGPUWinner),
                            cudaMemcpyDeviceToHost));

      // Filter: keep only diagonal results and remap to original indices
      NFGPUWinner *h_pass2 = (NFGPUWinner *)malloc(nPass2Raw * sizeof(NFGPUWinner));
      int nPass2 = 0;
      for (int i = 0; i < nPass2Raw; i++) {
        if (h_pass2Raw[i].voxelIdx == h_pass2Raw[i].orientIdx) {
          int p1Idx = h_pass2Raw[i].orientIdx;  // index into h_pass1
          h_pass2[nPass2].voxelIdx = h_pass1[p1Idx].voxelIdx;
          h_pass2[nPass2].orientIdx = h_pass1[p1Idx].orientIdx;
          h_pass2[nPass2].fracOverlap = h_pass2Raw[i].fracOverlap;
          nPass2++;
        }
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
        ctx->d_winners, ctx->d_nWinners, ctx->maxWinners);

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

#endif /* ENABLE_CUDA */
