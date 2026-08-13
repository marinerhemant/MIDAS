//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated tomographic reconstruction for MIDAS TOMO.
// This header provides the public C API for the CUDA implementation.
// All functions are guarded by ENABLE_CUDA — on CPU-only builds,
// including this header is safe and the functions simply won't exist.
//

#ifndef TOMO_GPU_H
#define TOMO_GPU_H

#ifdef ENABLE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: This header is always included from tomo_heads.h (after all struct
// definitions), so no forward declarations of GLOBAL_CONFIG_OPTS etc. are
// needed.  The only opaque type is TomoGPUContext, defined in tomo_gpu.cu.
typedef struct TomoGPUContext TomoGPUContext;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Initialise GPU context.  Call once before any reconstruction.
///
/// @param deviceId   CUDA device ordinal (0 for the first GPU).
/// @param sinogram_x_dim  Width of the padded sinogram (param->sinogram_x_dim).
/// @param theta_list_size Number of projection angles.
/// @param theta_list      Pointer to the angle table (host, float[theta_list_size]).
/// @param filter_type     Filter enum (FILTER_NONE, FILTER_HANN, …).
/// @param useFftwBridge   If non-zero, 1D and 2D FFTs are performed on the CPU
///                        (via FFTW) and data is transferred GPU↔CPU around
///                        each FFT call.  This guarantees byte-identical output
///                        to the CPU-only code path.
///                        If zero, cuFFT is used (faster, but output differs
///                        from FFTW by ~1e-6 per pixel).
/// @return Opaque context handle, or NULL on error.
TomoGPUContext *tomo_gpu_init(int deviceId,
                              unsigned long sinogram_x_dim,
                              int theta_list_size,
                              const float *theta_list,
                              int filter_type,
                              int useFftwBridge);

/// Free all GPU resources.
void tomo_gpu_destroy(TomoGPUContext *ctx);

/// Upload pre-computed tables from CPU gridrecParams to GPU.
/// Must be called once after initGridRec() on the CPU side, before any
/// tomo_gpu_reconstruct() calls.  Copies SINE, COSE, wtbl, winv, and
/// filphase arrays from host to device, ensuring exact numeric parity.
///
/// @param ctx   Opaque GPU context.
/// @param SINE  Host array [theta_list_size].
/// @param COSE  Host array [theta_list_size].
/// @param wtbl  Host array [ltbl+1].
/// @param winv  Host array [M0].
/// @param filphase  Host array of complex {r,i} pairs [(pdim/2+1)*2 floats].
/// @param ltbl  PSWF table length (512).
/// @param M0    Output image size.
/// @param pdim  Padded FFT dimension.
/// @return 0 on success.
int tomo_gpu_upload_tables(TomoGPUContext *ctx,
                           const float *SINE, const float *COSE,
                           const float *wtbl, const float *winv,
                           const float *filphase,
                           long ltbl, long M0, long pdim);

// ---------------------------------------------------------------------------
// Reconstruction (replaces CPU reconstruct() call)
// ---------------------------------------------------------------------------

/// GPU-accelerated reconstruction of a sinogram pair (like the CPU's
/// reconstruct()).
///
/// The caller is responsible for:
///   1. Setting up sinograms_boundary_padding and reconstructions_boundary_padding
///      (identical to the CPU path — reconCentering + setSinoAndReconBuffers).
///   2. Calling this function instead of reconstruct(&param).
///   3. The function copies sinogram data to the GPU, runs phase1+phase2+phase3,
///      and copies the reconstruction back.
///
/// @param ctx        Opaque GPU context returned by tomo_gpu_init().
/// @param sinogram1  Host pointer to padded sinogram 1
///                   (float[theta_list_size * sinogram_x_dim]).
/// @param sinogram2  Host pointer to padded sinogram 2.
/// @param reconstruction1  Host output buffer for reconstruction 1
///                         (float[M0 * sinogram_x_dim]).
/// @param reconstruction2  Host output buffer for reconstruction 2.
/// @param M          Grid size (power of 2, from gridrecParams.M).
/// @param M0         Output image size (from gridrecParams.M0).
/// @param M02        Half of (M0-1), from gridrecParams.M02.
/// @param pdim       Padded FFT dimension (from gridrecParams.pdim).
/// @return 0 on success, non-zero on error.
int tomo_gpu_reconstruct(TomoGPUContext *ctx,
                         const float *sinogram1,
                         const float *sinogram2,
                         float *reconstruction1,
                         float *reconstruction2,
                         long M, long M0, long M02, long pdim);

/// Batched GPU reconstruction of multiple sinogram pairs at once.
/// Processes n_pairs slice-pairs simultaneously on the GPU for maximum
/// throughput.  Each element of the pointer arrays corresponds to one pair.
///
/// @param ctx        GPU context.
/// @param n_pairs    Number of slice-pairs to process.
/// @param sinogram1s  Array of n_pairs host sinogram1 pointers.
/// @param sinogram2s  Array of n_pairs host sinogram2 pointers.
/// @param recon1s     Array of n_pairs host reconstruction1 output pointers.
/// @param recon2s     Array of n_pairs host reconstruction2 output pointers.
/// @return 0 on success.
int tomo_gpu_reconstruct_batch(TomoGPUContext *ctx,
                               int n_pairs,
                               const float **sinogram1s,
                               const float **sinogram2s,
                               float **recon1s,
                               float **recon2s,
                               long M, long M0, long M02, long pdim);

/// Batched GPU reconstruction from RAW sinograms (no CPU preprocessing needed).
/// Performs Pad + shift + boundary-replicate on GPU, then reconstruct, then
/// extract compact reconstructions on GPU before D2H.
///
/// @param raw_sino1s  Array of n_pairs host raw sinogram1 pointers [det_xdim × n_angles each]
/// @param raw_sino2s  Array of n_pairs host raw sinogram2 pointers
/// @param compact_recon1s  Output: compact recon [recon_xdim × recon_xdim each]
/// @param compact_recon2s  Output: compact recon
/// @param det_xdim       Raw detector width
/// @param adjusted_xdim  Padded detector width
/// @param sino_xdim      Sinogram X dimension (may differ from det_xdim)
/// @param recon_xdim     Reconstruction width
/// @param pad_front      Front padding size for Pad()
/// @param shift          Rotation-axis shift
/// @param doLog          Apply -log transform
/// @param auto_centering Apply auto-centering shift to output
int tomo_gpu_reconstruct_batch_raw(TomoGPUContext *ctx,
                                   int n_pairs,
                                   const float **raw_sino1s,
                                   const float **raw_sino2s,
                                   float **compact_recon1s,
                                   float **compact_recon2s,
                                   long M, long M0, long M02, long pdim,
                                   int det_xdim, int adjusted_xdim,
                                   int sino_xdim, int recon_xdim,
                                   int pad_front, float shift,
                                   int doLog, int auto_centering);
// ---------------------------------------------------------------------------
// Preprocessing kernels (optional — can also be done on CPU)
// ---------------------------------------------------------------------------

/// GPU-accelerated sinogram preprocessing.
/// Performs: Normalize → LogProj → reconCentering → RingCorrection
/// in a single GPU pipeline, avoiding round-trips.
///
/// @param ctx        GPU context.
/// @param short_sinogram  Raw uint16 sinogram (host, [theta_list_size * det_xdim]).
/// @param dark_field      Dark field average (host, float[det_xdim]).
/// @param white_field     Two white fields (host, float[2 * det_xdim]).
/// @param norm_sino_out   Output normalised sinogram (host, float[theta_list_size * adjusted_xdim]).
/// @param det_xdim        Raw detector width.
/// @param adjusted_xdim   Padded detector width.
/// @param theta_list_size Number of projection angles.
/// @param doLog           Apply -log transform.
/// @param shift           Rotation-axis shift.
/// @param ring_coeff      Ring removal coefficient (0 to disable).
/// @return 0 on success.
int tomo_gpu_preprocess(TomoGPUContext *ctx,
                        const unsigned short *short_sinogram,
                        const float *dark_field,
                        const float *white_field,
                        float *norm_sino_out,
                        int det_xdim,
                        int adjusted_xdim,
                        int theta_list_size,
                        int doLog,
                        float shift,
                        float ring_coeff);

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Print GPU device info to stdout.
void tomo_gpu_print_info(int deviceId);

/// Query free GPU memory in bytes.
size_t tomo_gpu_get_free_memory(void);

/// Allocate pinned (page-locked) host memory for async DMA.
/// @param ptr  Pointer to receive the allocation.
/// @param size Bytes to allocate.
/// @return 0 on success.
int tomo_gpu_pinned_alloc(void **ptr, size_t size);

/// Free pinned host memory.
void tomo_gpu_pinned_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* ENABLE_CUDA */

#endif /* TOMO_GPU_H */
