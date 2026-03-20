//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated orientation screening for MIDAS NF-HEDM.
// This header provides the public C API for the CUDA implementation.
// All functions are guarded by ENABLE_CUDA — on CPU-only builds,
// including this header is safe and the functions simply won't exist.
//

#ifndef NF_GPU_H
#define NF_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef ENABLE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// Opaque GPU context (defined in FitOrientationGPU.cu)
typedef struct NFGPUContext NFGPUContext;

// ──────────────────────────────────────────────────────────────
// Lifecycle
// ──────────────────────────────────────────────────────────────

/// Initialise NF-HEDM GPU context.
///
/// @param deviceId       CUDA device ordinal (0 for the first GPU).
/// @param nrPixelsY      Detector Y dimension (pixels).
/// @param nrPixelsZ      Detector Z dimension (pixels).
/// @param nrFiles        Number of omega frames.
/// @param nLayers        Number of detector distances.
/// @return Opaque context handle, or NULL on error.
NFGPUContext *nf_gpu_init(int deviceId,
                          int nrPixelsY, int nrPixelsZ,
                          int nrFiles, int nLayers);

/// Free all GPU resources.
void nf_gpu_destroy(NFGPUContext *ctx);

// ──────────────────────────────────────────────────────────────
// Data Upload
// ──────────────────────────────────────────────────────────────

/// Upload the ObsSpotsInfo bitfield, reorganizing it from the CPU's
/// flat layout [layer][frame][pixel] into per-frame slabs for
/// shared-memory-friendly access on the GPU.
///
/// CPU layout (flat): bit index = layer * NrFiles * NrPixY * NrPixZ
///                              + frame * NrPixY * NrPixZ
///                              + y * NrPixZ + z
///
/// GPU layout (per-frame): slab[frame * nLayers + layer] is a
///     uint32_t bitfield array of size (NrPixY * NrPixZ + 31) / 32,
///     where bit (y * NrPixZ + z) encodes the pixel observation.
///
/// @param ctx             GPU context.
/// @param h_obsSpots      Host pointer to the flat ObsSpotsInfo bitfield
///                        (as produced by the existing CPU pipeline).
/// @param sizeObsSpots    Total size in uint32_t words of h_obsSpots.
/// @return 0 on success, non-zero on error.
int nf_gpu_upload_obs_spots(NFGPUContext *ctx,
                            const int *h_obsSpots,
                            long long sizeObsSpots);

/// Upload precomputed per-orientation spot data.
///
/// For each orientation, its theoretical diffraction spots are voxel-
/// independent (y, z, omega positions), computed once by CalcDiffractionSpots.
///
/// This function processes orientation matrices + SpotsMat data, precomputing
/// GPU-friendly per-spot metadata including sinOme, cosOme, OmeBin,
/// and the base pixel projection (at sample center).
///
/// @param ctx                 GPU context.
/// @param h_orientationMatrix Host OrientationMatrix.bin data [nOrient * 9].
/// @param nOrientations       Number of candidate orientations in this batch.
/// @param h_spotsMat          Host DiffractionSpots.bin data [totalSpots * 3].
/// @param h_nrSpots           Per-orientation (NrSpots, StartingRowNr) pairs [nOrient * 2].
/// @param hkls                The HKL list [n_hkls][4].
/// @param thetas              Bragg angles [n_hkls].
/// @param n_hkls              Number of HKL reflections.
/// @param Gs                  Precomputed G-vector magnitudes [n_hkls].
/// @param Lsd                 Sample-to-detector distances [nLayers].
/// @param ybc                 Beam center Y [nLayers].
/// @param zbc                 Beam center Z [nLayers].
/// @param px                  Pixel size (microns).
/// @param gs                  Grid size (microns).
/// @param omegaStart          Omega start angle (degrees).
/// @param omegaStep           Omega step angle (degrees).
/// @param rotMatTilts         Detector tilt rotation matrix [3][3].
/// @param excludePoleAngle    Pole angle exclusion (degrees).
/// @param omegaRanges         Omega range limits [nOmeRanges][2].
/// @param nOmeRanges          Number of omega ranges.
/// @param boxSizes            Box sizes [nOmeRanges][4].
/// @param wedge               Wedge angle (degrees).
/// @param wavelength          X-ray wavelength.
/// @return 0 on success.
int nf_gpu_upload_orientations(NFGPUContext *ctx,
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
                               double wedge, double wavelength);

// ──────────────────────────────────────────────────────────────
// Phase 1: GPU Screening
// ──────────────────────────────────────────────────────────────

/// GPU screening result for a single (voxel, orientation) pair
/// that passed the minFracOverlap threshold.
typedef struct {
    int voxelIdx;      ///< Index into the voxel grid
    int orientIdx;     ///< Index into the orientation list
    float fracOverlap; ///< Computed FracOverlap value
} NFGPUWinner;

/// Run GPU Phase 1 screening: for each (voxel, orientation) pair,
/// compute FracOverlap and collect winners above threshold.
///
/// @param ctx              GPU context with data already uploaded.
/// @param h_XGrains        Voxel triangle X-corners [nVoxels * 3].
/// @param h_YGrains        Voxel triangle Y-corners [nVoxels * 3].
/// @param nVoxels          Number of voxels to evaluate.
/// @param minFracOverlap   Minimum FracOverlap threshold (e.g., 0.04).
/// @param h_winners        Output: array of NFGPUWinner structs.
///                         Caller must free with free().
/// @param nWinners         Output: number of winners found.
/// @return 0 on success.
int nf_gpu_screen(NFGPUContext *ctx,
                  const double *h_XGrains,
                  const double *h_YGrains,
                  int nVoxels,
                  double minFracOverlap,
                  NFGPUWinner **h_winners,
                  int *nWinners);

// ──────────────────────────────────────────────────────────────
// Phase 2: GPU Nelder-Mead Fitting
// ──────────────────────────────────────────────────────────────

/// GPU fitting result for a single winner.
typedef struct {
    int voxelIdx;
    float eulerA, eulerB, eulerC;  ///< Refined Euler angles (degrees)
    float fracOverlap;             ///< Refined FracOverlap
} NFGPUFitResult;

/// Upload HKL data to GPU constant memory for Phase 2 fitting.
/// Must be called before nf_gpu_fit.
int nf_gpu_upload_hkls(NFGPUContext *ctx,
                       const double hkls[][4],
                       const double *Gs,
                       int n_hkls,
                       double excludePoleAngle,
                       double omegaStart, double omegaStep,
                       const double omegaRanges[][2],
                       int nOmeRanges,
                       const double boxSizes[][4]);

/// Run GPU Phase 2 NM fitting for Phase 1 winners.
///
/// @param ctx              GPU context with Phase 1 data uploaded.
/// @param winners          Phase 1 winners (from nf_gpu_screen).
/// @param nWinners         Number of winners.
/// @param h_XGrains        Voxel triangle X-corners [nVoxels * 3].
/// @param h_YGrains        Voxel triangle Y-corners [nVoxels * 3].
/// @param eulerTol         Euler angle tolerance for NM bounds (radians).
/// @param h_orientMatrix   Orientation matrices [nOrient * 9] (for initial guess).
/// @param fitResults       Output: array of NFGPUFitResult. Caller must free().
/// @param nFitResults      Output: number of results.
/// @return 0 on success.
int nf_gpu_fit(NFGPUContext *ctx,
               const NFGPUWinner *winners,
               int nWinners,
               const double *h_XGrains,
               const double *h_YGrains,
               double eulerTol,
               const double *h_orientMatrix,
               NFGPUFitResult **fitResults,
               int *nFitResults);

// ──────────────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────────────

/// Print GPU device info.
void nf_gpu_print_info(int deviceId);

/// Query free GPU memory in bytes.
size_t nf_gpu_get_free_memory(void);

#ifdef __cplusplus
}
#endif

#endif /* ENABLE_CUDA */

#endif /* NF_GPU_H */
