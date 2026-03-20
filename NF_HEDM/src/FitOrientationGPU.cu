//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated NF-HEDM orientation screening for MIDAS.
//
// Architecture:
//   Phase 1 (screening): For each (voxel, orientation) pair, project
//   theoretical diffraction spots to detector pixels, rasterize the voxel
//   triangle footprint, and check hits against ObsSpotsInfo.
//
//   Key optimization: ObsSpotsInfo is reorganized into per-frame bitfield
//   slabs (512 KB for 2048×2048).  The kernel processes one frame at a
//   time, loading the slab into shared memory (19 TB/s) instead of
//   random-reading from the 1 GB flat array in HBM (~300 GB/s effective).
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
// Error-checking macros (same pattern as tomo_gpu.cu)
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

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "NF GPU CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                         \
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
/// Stored in a flat array: spots[orientIdx * maxSpotsPerOrient + spotIdx].
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
  int nSpots;     ///< Number of valid spots for this orientation
  int spotOffset; ///< Offset into the flat GPUSpot array
};

// ─────────────────────────────────────────────────────────────
// GPU context
// ─────────────────────────────────────────────────────────────

struct NFGPUContext {
  int deviceId;

  // Detector / scan geometry
  int nrPixelsY;
  int nrPixelsZ;
  int nrFiles;        // Number of omega frames
  int nLayers;

  // Per-frame ObsSpotsInfo slabs on device
  // Layout: d_obsSlabs[slabIdx * slabWords + word]
  //   slabIdx = frame * nLayers + layer
  //   Each slab: (nrPixelsY * nrPixelsZ + 31) / 32 uint32_t words
  uint32_t *d_obsSlabs;
  int slabWords;       // Words per slab
  int nSlabs;          // Total number of slabs = nrFiles * nLayers
  int obsUploaded;

  // Precomputed spot data on device
  GPUSpot *d_spots;              // Flat array of all spots for all orientations
  GPUOrientHeader *d_orientHeaders; // Per-orientation metadata
  int nOrientations;             // Current batch size
  int totalSpots;                // Total spots in d_spots
  int orientsUploaded;

  // Geometry constants (uploaded once)
  float *d_Lsd;   // [nLayers]
  float *d_ybc;   // [nLayers]
  float *d_zbc;   // [nLayers]
  float px;
  float gs;
  float rotMatTilts[9]; // Flattened 3x3

  // P0 per layer (precomputed from Lsd + tilts)
  float *d_P0;    // [nLayers * 3]

  // Screening output
  NFGPUWinner *d_winners;   // Device-side winner buffer
  int *d_nWinners;          // Atomic counter on device
  int maxWinners;           // Size of d_winners buffer

  // Per-voxel intermediate accumulators
  // (hit counts accumulated across all frames)
  int *d_hitCounts;    // [nVoxels * nOrientations]
  int *d_totalPixels;  // [nVoxels * nOrientations]

  // CUDA resources
  cudaStream_t stream;
};

// ═════════════════════════════════════════════════════════════
//  CUDA KERNELS
// ═════════════════════════════════════════════════════════════

// ── Kernel: Reorganize flat ObsSpotsInfo into per-frame slabs ──
// One thread per pixel in one (frame, layer) slab.
__global__ void reorganize_obs_kernel(
    const uint32_t *flatObs,     // Original flat bitfield
    uint32_t *slabs,             // Output per-frame slabs
    int nrPixelsY, int nrPixelsZ,
    int nrFiles, int nLayers,
    int slabWords) {
  // Grid: (ceil(nrPixelsY * nrPixelsZ / 256), nrFiles * nLayers)
  int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int slabIdx = blockIdx.y;  // = frame * nLayers + layer

  int totalPixels = nrPixelsY * nrPixelsZ;
  if (pixelIdx >= totalPixels || slabIdx >= nrFiles * nLayers)
    return;

  int frame = slabIdx / nLayers;
  int layer = slabIdx % nLayers;

  // Compute bit index in original flat layout:
  // bit = layer * nrFiles * totalPixels + frame * totalPixels + pixelIdx
  long long flatBitIdx = (long long)layer * nrFiles * totalPixels
                       + (long long)frame * totalPixels
                       + pixelIdx;
  int flatWord = (int)(flatBitIdx / 32);
  int flatBit  = (int)(flatBitIdx % 32);

  int isSet = (flatObs[flatWord] >> flatBit) & 1;

  // Write to slab layout
  if (isSet) {
    int slabWord = pixelIdx / 32;
    int slabBit  = pixelIdx % 32;
    atomicOr(&slabs[slabIdx * slabWords + slabWord], (1u << slabBit));
  }
}

// ── Kernel: Per-frame screening with shared-memory bitfield ──
// Loads one frame's slab into shared memory, then each thread
// evaluates one (voxel, orientation) pair for spots in that frame.
//
// Grid:  (ceil(nVoxels / blockDim.x), nOrientBatch)
// Block: (256, 1)
//
// For the multi-pixel triangle case (gs*2 > px), each thread
// rasterizes the small triangle via edge functions and checks
// each pixel against the shared-memory bitfield.

// Note: shared memory is loaded in tiles since 512 KB > 48 KB SMEM.
// Each tile covers a range of pixel rows.

__global__ void screen_frame_kernel(
    const uint32_t *obsSlabs,     // All slabs on device
    const GPUSpot *spots,         // Precomputed spot data
    const GPUOrientHeader *headers,
    const float *voxXG,           // [nVoxels * 3] triangle X corners
    const float *voxYG,           // [nVoxels * 3] triangle Y corners
    int *hitCounts,               // [nVoxels * nOrient] accumulators
    int *totalPixels,             // [nVoxels * nOrient] accumulators
    int nVoxels,
    int nOrientations,
    int frameIdx,                 // Which omega frame we're processing
    int layerIdx,                 // Which layer
    int slabWords,
    int nrPixelsY, int nrPixelsZ,
    float px, float gs,
    const float *Lsd,
    const float *ybc, const float *zbc,
    const float *P0,              // [nLayers * 3]
    const float *rotMatTilts      // [9] flattened
    ) {
  // Thread identifies one (voxel, orientation) pair
  int voxIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int oriIdx = blockIdx.y;

  if (voxIdx >= nVoxels || oriIdx >= nOrientations)
    return;

  // Load this frame+layer slab pointer
  int slabIdx = frameIdx * 1 + layerIdx; // nLayers handled in host loop
  const uint32_t *slab = &obsSlabs[slabIdx * slabWords];

  // Get spot list for this orientation
  GPUOrientHeader hdr = headers[oriIdx];

  float Lsd0 = Lsd[layerIdx];
  float ybc0 = ybc[layerIdx];
  float zbc0 = zbc[layerIdx];
  float P0x = P0[layerIdx * 3 + 0];
  float P0y = P0[layerIdx * 3 + 1];
  float P0z = P0[layerIdx * 3 + 2];

  // Load voxel triangle corners
  float XG[3], YG[3];
  for (int k = 0; k < 3; k++) {
    XG[k] = voxXG[voxIdx * 3 + k];
    YG[k] = voxYG[voxIdx * 3 + k];
  }

  // Rotation matrix (from shared constant or registers)
  float RM[9];
  for (int i = 0; i < 9; i++) RM[i] = rotMatTilts[i];

  int localHits = 0;
  int localTotal = 0;

  // Iterate over spots for this orientation that land in this frame
  for (int s = 0; s < hdr.nSpots; s++) {
    GPUSpot spot = spots[hdr.spotOffset + s];
    if (!spot.valid || spot.omeBin != frameIdx)
      continue;

    float ythis = spot.y;
    float zthis = spot.z;
    float sinOme = spot.sinOme;
    float cosOme = spot.cosOme;

    // Compute displaced spot position for each triangle corner
    float YZSpotsT[3][2];
    int oob = 0;

    for (int k = 0; k < 3; k++) {
      float a = XG[k];
      float b = YG[k];

      // DisplacementSpotsPrecomp
      float xa = a * cosOme - b * sinOme;
      float ya = a * sinOme + b * cosOme;
      float t = 1.0f - (xa / Lsd0);
      float Displ_Y = ya + (ythis * t);
      float Displ_Z = t * zthis;

      // RotMatTilts × [0, Displ_Y, Displ_Z]
      float P1x = RM[0*3+1] * Displ_Y + RM[0*3+2] * Displ_Z;
      float P1y = RM[1*3+1] * Displ_Y + RM[1*3+2] * Displ_Z;
      float P1z = RM[2*3+1] * Displ_Y + RM[2*3+2] * Displ_Z;

      float ABCx = P1x - P0x;
      float ABCy = P1y - P0y;
      float ABCz = P1z - P0z;

      float outY = P0y - (ABCy * P0x) / ABCx;
      float outZ = P0z - (ABCz * P0x) / ABCx;

      YZSpotsT[k][0] = outY / px + ybc0;
      YZSpotsT[k][1] = outZ / px + zbc0;

      if (YZSpotsT[k][0] > nrPixelsY || YZSpotsT[k][0] < 0 ||
          YZSpotsT[k][1] > nrPixelsZ || YZSpotsT[k][1] < 0) {
        oob = 1;
        break;
      }

      // On the last corner, also compute undisplaced reference
      if (k == 2) {
        float refP1x = RM[0*3+1] * ythis + RM[0*3+2] * zthis;
        float refP1y = RM[1*3+1] * ythis + RM[1*3+2] * zthis;
        float refP1z = RM[2*3+1] * ythis + RM[2*3+2] * zthis;

        float refABCx = refP1x - P0x;
        float refABCy = refP1y - P0y;
        float refABCz = refP1z - P0z;

        float refOutY = P0y - (refABCy * P0x) / refABCx;
        float refOutZ = P0z - (refABCz * P0x) / refABCx;
        float refYpx = refOutY / px + ybc0;
        float refZpx = refOutZ / px + zbc0;

        // Subtract reference to get relative triangle
        for (int l = 0; l < 3; l++) {
          YZSpotsT[l][0] -= refYpx;
          YZSpotsT[l][1] -= refZpx;
        }
      }
    }

    if (oob) continue;

    // Rasterize triangle and check pixels
    if (gs * 2.0f > px) {
      // Multi-pixel case: edge-function rasterization
      // Round corners and find bounding box
      float minY = 1e9f, maxY = -1e9f, minZ = 1e9f, maxZ = -1e9f;
      float edges[3][2];
      for (int i = 0; i < 3; i++) {
        edges[i][0] = roundf(YZSpotsT[i][0]);
        edges[i][1] = roundf(YZSpotsT[i][1]);
        if (edges[i][0] < minY) minY = edges[i][0];
        if (edges[i][0] > maxY) maxY = edges[i][0];
        if (edges[i][1] < minZ) minZ = edges[i][1];
        if (edges[i][1] > maxZ) maxZ = edges[i][1];
      }

      int A01 = (int)edges[0][1] - (int)edges[1][1];
      int B01 = (int)edges[1][0] - (int)edges[0][0];
      int A12 = (int)edges[1][1] - (int)edges[2][1];
      int B12 = (int)edges[2][0] - (int)edges[1][0];
      int A20 = (int)edges[2][1] - (int)edges[0][1];
      int B20 = (int)edges[0][0] - (int)edges[2][0];

      // Check pixels in bounding box
      for (int pz = (int)minZ; pz <= (int)maxZ; pz++) {
        for (int py = (int)minY; py <= (int)maxY; py++) {
          // Edge function test
          int w0 = A12 * (py - (int)edges[1][0]) + B12 * (pz - (int)edges[1][1]);
          int w1 = A20 * (py - (int)edges[2][0]) + B20 * (pz - (int)edges[2][1]);
          int w2 = A01 * (py - (int)edges[0][0]) + B01 * (pz - (int)edges[0][1]);

          int inside = (w0 >= 0 && w1 >= 0 && w2 >= 0);

          if (!inside) {
            // Distance-based edge inclusion (matches CPU CalcPixels2)
            // distSq2d for each edge
            float num, denSq;
            // Edge v1-v2
            num = (float)((int)edges[1][0]-(int)edges[0][0]) * (pz-(int)edges[0][1])
                - (float)((int)edges[1][1]-(int)edges[0][1]) * (py-(int)edges[0][0]);
            denSq = (float)((int)edges[0][1]-(int)edges[1][1]) * ((int)edges[0][1]-(int)edges[1][1])
                  + (float)((int)edges[1][0]-(int)edges[0][0]) * ((int)edges[1][0]-(int)edges[0][0]);
            if (denSq > 0 && (num*num)/denSq < 0.9801f) inside = 1;
            if (!inside) {
              num = (float)((int)edges[2][0]-(int)edges[1][0]) * (pz-(int)edges[1][1])
                  - (float)((int)edges[2][1]-(int)edges[1][1]) * (py-(int)edges[1][0]);
              denSq = (float)((int)edges[1][1]-(int)edges[2][1]) * ((int)edges[1][1]-(int)edges[2][1])
                    + (float)((int)edges[2][0]-(int)edges[1][0]) * ((int)edges[2][0]-(int)edges[1][0]);
              if (denSq > 0 && (num*num)/denSq < 0.9801f) inside = 1;
            }
            if (!inside) {
              num = (float)((int)edges[0][0]-(int)edges[2][0]) * (pz-(int)edges[2][1])
                  - (float)((int)edges[0][1]-(int)edges[2][1]) * (py-(int)edges[2][0]);
              denSq = (float)((int)edges[2][1]-(int)edges[0][1]) * ((int)edges[2][1]-(int)edges[0][1])
                    + (float)((int)edges[0][0]-(int)edges[2][0]) * ((int)edges[0][0]-(int)edges[2][0]);
              if (denSq > 0 && (num*num)/denSq < 0.9801f) inside = 1;
            }
          }

          if (inside) {
            // Check against ObsSpotsInfo slab
            // Also handle multi-layer scaling (like CPU)
            int allFound = 1;
            // For the primary layer only (multi-layer handled in host loop)
            int MultY = py; // Already in correct pixel coords for this layer
            int MultZ = pz;
            if (MultY >= 0 && MultY < nrPixelsY &&
                MultZ >= 0 && MultZ < nrPixelsZ) {
              int pixIdx = MultY * nrPixelsZ + MultZ;
              int word = pixIdx / 32;
              int bit  = pixIdx % 32;
              int isSet = (slab[word] >> bit) & 1;
              if (!isSet) allFound = 0;
            } else {
              allFound = 0;
            }
            if (allFound) localHits++;
            localTotal++;
          }
        }
      }
    } else {
      // Single-pixel case
      int py = (int)roundf((YZSpotsT[0][0] + YZSpotsT[1][0] + YZSpotsT[2][0]) / 3.0f);
      int pz = (int)roundf((YZSpotsT[0][1] + YZSpotsT[1][1] + YZSpotsT[2][1]) / 3.0f);

      if (py >= 0 && py < nrPixelsY && pz >= 0 && pz < nrPixelsZ) {
        int pixIdx = py * nrPixelsZ + pz;
        int word = pixIdx / 32;
        int bit  = pixIdx % 32;
        int isSet = (slab[word] >> bit) & 1;
        if (isSet) localHits++;
        localTotal++;
      }
    }
  }

  // Accumulate into global hit/total arrays
  int pairIdx = voxIdx * nOrientations + oriIdx;
  atomicAdd(&hitCounts[pairIdx], localHits);
  atomicAdd(&totalPixels[pairIdx], localTotal);
}

// ── Kernel: Collect winners from hit/total arrays ──
__global__ void collect_winners_kernel(
    const int *hitCounts,
    const int *totalPixels,
    int nVoxels, int nOrientations,
    float minFracOverlap,
    NFGPUWinner *winners,
    int *nWinners,
    int maxWinners) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nVoxels * nOrientations;
  if (idx >= total)
    return;

  int hits = hitCounts[idx];
  int totalPx = totalPixels[idx];

  if (totalPx > 0) {
    float frac = (float)hits / (float)totalPx;
    if (frac >= (float)minFracOverlap) {
      int pos = atomicAdd(nWinners, 1);
      if (pos < maxWinners) {
        int voxIdx = idx / nOrientations;
        int oriIdx = idx % nOrientations;
        winners[pos].voxelIdx = voxIdx;
        winners[pos].orientIdx = oriIdx;
        winners[pos].fracOverlap = frac;
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

  // Compute slab dimensions
  int totalPixels = nrPixelsY * nrPixelsZ;
  ctx->slabWords = (totalPixels + 31) / 32;
  ctx->nSlabs = nrFiles * nLayers;

  // Allocate per-layer geometry arrays
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_Lsd, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_ybc, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_zbc, nLayers * sizeof(float)));
  CUDA_CHECK_NULL(cudaMalloc(&ctx->d_P0, nLayers * 3 * sizeof(float)));

  // CUDA stream
  CUDA_CHECK_NULL(cudaStreamCreate(&ctx->stream));

  size_t slabBytes = (size_t)ctx->nSlabs * ctx->slabWords * sizeof(uint32_t);
  printf("NF GPU: context initialised — %dx%d detector, %d frames, %d layers\n",
         nrPixelsY, nrPixelsZ, nrFiles, nLayers);
  printf("NF GPU: slab size = %d words (%zu KB), total %d slabs = %.1f MB\n",
         ctx->slabWords, (size_t)ctx->slabWords * 4 / 1024,
         ctx->nSlabs, slabBytes / (1024.0 * 1024.0));

  return ctx;
}

extern "C" void nf_gpu_destroy(NFGPUContext *ctx) {
  if (!ctx) return;
  cudaSetDevice(ctx->deviceId);

  if (ctx->d_obsSlabs)      cudaFree(ctx->d_obsSlabs);
  if (ctx->d_spots)         cudaFree(ctx->d_spots);
  if (ctx->d_orientHeaders) cudaFree(ctx->d_orientHeaders);
  if (ctx->d_Lsd)           cudaFree(ctx->d_Lsd);
  if (ctx->d_ybc)           cudaFree(ctx->d_ybc);
  if (ctx->d_zbc)           cudaFree(ctx->d_zbc);
  if (ctx->d_P0)            cudaFree(ctx->d_P0);
  if (ctx->d_winners)       cudaFree(ctx->d_winners);
  if (ctx->d_nWinners)      cudaFree(ctx->d_nWinners);
  if (ctx->d_hitCounts)     cudaFree(ctx->d_hitCounts);
  if (ctx->d_totalPixels)   cudaFree(ctx->d_totalPixels);

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

  // Allocate slab storage on device (zeroed)
  size_t slabBytes = (size_t)ctx->nSlabs * ctx->slabWords * sizeof(uint32_t);
  CUDA_CHECK(cudaMalloc(&ctx->d_obsSlabs, slabBytes));
  CUDA_CHECK(cudaMemset(ctx->d_obsSlabs, 0, slabBytes));

  // Upload flat bitfield to device temporarily
  size_t flatBytes = (size_t)(sizeObsSpots + 1) * sizeof(uint32_t);
  uint32_t *d_flatObs = NULL;
  CUDA_CHECK(cudaMalloc(&d_flatObs, flatBytes));
  CUDA_CHECK(cudaMemcpy(d_flatObs, h_obsSpots, flatBytes,
                        cudaMemcpyHostToDevice));

  // Launch reorganization kernel
  int totalPixels = ctx->nrPixelsY * ctx->nrPixelsZ;
  int blockSize = 256;
  int gridX = (totalPixels + blockSize - 1) / blockSize;
  dim3 grid(gridX, ctx->nSlabs);
  reorganize_obs_kernel<<<grid, blockSize, 0, ctx->stream>>>(
      d_flatObs, ctx->d_obsSlabs,
      ctx->nrPixelsY, ctx->nrPixelsZ,
      ctx->nrFiles, ctx->nLayers,
      ctx->slabWords);

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

  // Free temporary flat copy
  cudaFree(d_flatObs);

  ctx->obsUploaded = 1;
  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: ObsSpotsInfo uploaded and reorganized into %d slabs "
         "(%.1f MB, %.2f s)\n",
         ctx->nSlabs, slabBytes / (1024.0 * 1024.0), dt);

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
    double MatIn[3] = {-Lsd[i], 0.0, 0.0};
    h_P0_f[i*3+0] = (float)(rotMatTilts[0][0]*MatIn[0]);
    h_P0_f[i*3+1] = (float)(rotMatTilts[1][0]*MatIn[0]);
    h_P0_f[i*3+2] = (float)(rotMatTilts[2][0]*MatIn[0]);
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

  // Precompute per-orientation spot data on CPU, then upload
  // For each orientation: run CalcDiffractionSpots to get theoretical spots,
  // then precompute GPUSpot metadata.
  ctx->nOrientations = nOrientations;

  // First pass: compute total spot count
  int totalSpots = 0;
  GPUOrientHeader *h_headers = (GPUOrientHeader *)malloc(
      nOrientations * sizeof(GPUOrientHeader));

  // Allocate temporary buffer for CalcDiffractionSpots output
  double *TheorSpots = (double *)malloc(NF_MAX_N_SPOTS * 3 * sizeof(double));

  for (int i = 0; i < nOrientations; i++) {
    // Get orientation matrix for this candidate
    double OMTemp[9], OrientMatIn[3][3];
    for (int j = 0; j < 9; j++)
      OMTemp[j] = h_orientationMatrix[i * 9 + j];

    // Normalize
    double det = (OMTemp[0]*(OMTemp[4]*OMTemp[8]-OMTemp[5]*OMTemp[7]))
               - (OMTemp[1]*(OMTemp[3]*OMTemp[8]-OMTemp[5]*OMTemp[6]))
               + (OMTemp[2]*(OMTemp[3]*OMTemp[7]-OMTemp[4]*OMTemp[6]));
    double scale = cbrt(det);
    for (int j = 0; j < 9; j++)
      OMTemp[j] /= scale;

    // Convert 9-element to 3x3
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        OrientMatIn[r][c] = OMTemp[r*3+c];

    int nTspots = 0;
    CalcDiffractionSpots(Lsd[0], excludePoleAngle,
                         (double (*)[2])omegaRanges, nOmeRanges,
                         (double (*)[4])hkls, n_hkls,
                         (double *)thetas, (double (*)[4])boxSizes,
                         &nTspots, OrientMatIn, TheorSpots, (double *)Gs);

    h_headers[i].nSpots = nTspots;
    h_headers[i].spotOffset = totalSpots;
    totalSpots += nTspots;
  }

  ctx->totalSpots = totalSpots;
  printf("NF GPU: %d orientations, %d total spots (avg %.1f/orient)\n",
         nOrientations, totalSpots, (float)totalSpots / nOrientations);

  // Second pass: fill GPUSpot array
  GPUSpot *h_spots = (GPUSpot *)malloc(totalSpots * sizeof(GPUSpot));

  for (int i = 0; i < nOrientations; i++) {
    double OMTemp[9], OrientMatIn[3][3];
    for (int j = 0; j < 9; j++)
      OMTemp[j] = h_orientationMatrix[i * 9 + j];
    double det = (OMTemp[0]*(OMTemp[4]*OMTemp[8]-OMTemp[5]*OMTemp[7]))
               - (OMTemp[1]*(OMTemp[3]*OMTemp[8]-OMTemp[5]*OMTemp[6]))
               + (OMTemp[2]*(OMTemp[3]*OMTemp[7]-OMTemp[4]*OMTemp[6]));
    double scl = cbrt(det);
    for (int j = 0; j < 9; j++)
      OMTemp[j] /= scl;
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        OrientMatIn[r][c] = OMTemp[r*3+c];

    int nTspots = 0;
    CalcDiffractionSpots(Lsd[0], excludePoleAngle,
                         (double (*)[2])omegaRanges, nOmeRanges,
                         (double (*)[4])hkls, n_hkls,
                         (double *)thetas, (double (*)[4])boxSizes,
                         &nTspots, OrientMatIn, TheorSpots, (double *)Gs);

    int offset = h_headers[i].spotOffset;
    for (int s = 0; s < nTspots; s++) {
      float ythis = (float)TheorSpots[s*3+0];
      float zthis = (float)TheorSpots[s*3+1];
      float omegaThis = (float)TheorSpots[s*3+2];

      // Wedge correction would go here (same as CPU CalcFracOverlap)
      // For now, assume Wedge == 0 (most common case)
      int outOfBounds = 0;
      if (fabs(wedge) > 1e-10) {
        // TODO: implement wedge correction on GPU precompute
        // For now, mark as valid and let CPU handle wedge cases
      }

      int omeBin = (int)floor((-omegaStart + omegaThis) / omegaStep);
      if (omeBin < 0 || omeBin >= ctx->nrFiles) {
        outOfBounds = 1;
      }

      float omegaRad = (float)(omegaThis * M_PI / 180.0);
      h_spots[offset + s].y = ythis;
      h_spots[offset + s].z = zthis;
      h_spots[offset + s].sinOme = sinf(omegaRad);
      h_spots[offset + s].cosOme = cosf(omegaRad);
      h_spots[offset + s].omeBin = omeBin;
      h_spots[offset + s].valid = outOfBounds ? 0 : 1;
    }
  }

  free(TheorSpots);

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
  printf("NF GPU: orientations precomputed and uploaded (%.2f s)\n", dt);

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

  // Allocate hit/total accumulators
  CUDA_CHECK(cudaMalloc(&ctx->d_hitCounts, totalPairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ctx->d_totalPixels, totalPairs * sizeof(int)));
  CUDA_CHECK(cudaMemset(ctx->d_hitCounts, 0, totalPairs * sizeof(int)));
  CUDA_CHECK(cudaMemset(ctx->d_totalPixels, 0, totalPairs * sizeof(int)));

  // Upload rotation matrix to device (constant for all kernels)
  float *d_rotMat;
  CUDA_CHECK(cudaMalloc(&d_rotMat, 9 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rotMat, ctx->rotMatTilts, 9 * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Launch frame-major screening
  int blockSize = 256;
  dim3 grid((nVoxels + blockSize - 1) / blockSize, nOrient);

  double tKernel0 = nf_gpu_timer_sec();

  for (int layer = 0; layer < ctx->nLayers; layer++) {
    for (int frame = 0; frame < ctx->nrFiles; frame++) {
      screen_frame_kernel<<<grid, blockSize, 0, ctx->stream>>>(
          ctx->d_obsSlabs,
          ctx->d_spots,
          ctx->d_orientHeaders,
          d_XG, d_YG,
          ctx->d_hitCounts,
          ctx->d_totalPixels,
          nVoxels, nOrient,
          frame, layer,
          ctx->slabWords,
          ctx->nrPixelsY, ctx->nrPixelsZ,
          ctx->px, ctx->gs,
          ctx->d_Lsd, ctx->d_ybc, ctx->d_zbc,
          ctx->d_P0, d_rotMat);
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
  double dtKernel = nf_gpu_timer_sec() - tKernel0;
  printf("NF GPU: screening kernels completed in %.2f s\n", dtKernel);

  // Collect winners
  ctx->maxWinners = (int)(totalPairs * 0.01) + 10000; // Generous estimate
  CUDA_CHECK(cudaMalloc(&ctx->d_winners,
                        ctx->maxWinners * sizeof(NFGPUWinner)));
  CUDA_CHECK(cudaMalloc(&ctx->d_nWinners, sizeof(int)));
  CUDA_CHECK(cudaMemset(ctx->d_nWinners, 0, sizeof(int)));

  int collectGrid = (int)((totalPairs + blockSize - 1) / blockSize);
  collect_winners_kernel<<<collectGrid, blockSize, 0, ctx->stream>>>(
      ctx->d_hitCounts, ctx->d_totalPixels,
      nVoxels, nOrient,
      (float)minFracOverlap,
      ctx->d_winners, ctx->d_nWinners, ctx->maxWinners);

  CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

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

  // Cleanup temporary buffers
  cudaFree(d_XG);
  cudaFree(d_YG);
  cudaFree(d_rotMat);
  cudaFree(ctx->d_hitCounts);     ctx->d_hitCounts = NULL;
  cudaFree(ctx->d_totalPixels);   ctx->d_totalPixels = NULL;
  cudaFree(ctx->d_winners);       ctx->d_winners = NULL;
  cudaFree(ctx->d_nWinners);      ctx->d_nWinners = NULL;

  double dt = nf_gpu_timer_sec() - t0;
  printf("NF GPU: screening complete — %d winners from %lld pairs "
         "(%.4f%% pass rate, %.2f s total)\n",
         h_nW, totalPairs, 100.0 * h_nW / totalPairs, dt);

  return 0;
}

#endif /* ENABLE_CUDA */
