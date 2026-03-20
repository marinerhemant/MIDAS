//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// GPU-accelerated tomographic reconstruction for MIDAS TOMO.
// Direct port of the CPU gridrec algorithm (phase1/phase2/phase3)
// for byte-level parity (FFTW-bridge mode) or maximum speed (cuFFT mode).
//

#include "tomo_gpu.h"

#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double gpu_timer_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Include the CPU header for constants and struct definitions
extern "C" {
#include "tomo_heads.h"
}

// ─────────────────────────────────────────────────────────────
// Error-checking macros
// ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_VOID(call)                                                  \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult res = (call);                                                  \
    if (res != CUFFT_SUCCESS) {                                                \
      fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__,        \
              (int)res);                                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// ─────────────────────────────────────────────────────────────
// GPU context
// ─────────────────────────────────────────────────────────────

struct TomoGPUContext {
  int deviceId;

  // Problem dimensions
  unsigned long sinogram_x_dim; // = adjusted_xdim * 2 (padded, complex-doubled)
  int theta_list_size;
  int filter_type;
  int useFftwBridge;

  // Pre-computed tables on device (uploaded from CPU's initGridRec)
  float *d_SINE;     // [theta_list_size]
  float *d_COSE;     // [theta_list_size]
  float *d_wtbl;     // [ltbl+1]
  float *d_winv;     // [M0]
  float *d_filphase; // [(pdim/2+1)*2] — complex as float pairs
  int tables_uploaded;

  // Gridrec parameters
  long pdim, M, M0, M02, ltbl;
  float L, scale, sampl;
  int flag;

  // Device buffers
  float *d_sino1; // [theta_list_size * sinogram_x_dim]
  float *d_sino2;
  cufftComplex *d_cproj; // [pdim] — 1D FFT work buffer (FFTW-bridge only)
  cufftComplex
      *d_cproj_batch; // [theta_list_size * pdim] — batched 1D FFT buffer
  cufftComplex *d_H;  // [(M+1)*(M+1)] — Fourier grid
  float *d_recon1;    // [M0 * sinogram_x_dim] (stride = sinogram_x_dim)
  float *d_recon2;

  // cuFFT plans
  cufftHandle plan1d;       // single 1D FFT (FFTW-bridge fallback)
  cufftHandle plan1d_batch; // batched 1D FFT: all angles at once
  cufftHandle plan2d;
  int cufft_plans_created;

  // FFTW-bridge pinned host buffer
  float *h_fft_bridge;
  size_t h_fft_bridge_size;

  // 3-stream pipeline
  cudaStream_t stream_h2d;
  cudaStream_t stream_compute;
  cudaStream_t stream_d2h;
  cudaEvent_t event_h2d_done;
  cudaEvent_t event_compute_done;
  cudaEvent_t event_d2h_done;

  // Multi-pair batch buffers (lazy-allocated on first batch call)
  int batch_max_pairs; // max pairs these buffers can hold
  float *d_sino1_multi;
  float *d_sino2_multi;
  cufftComplex *d_cproj_multi;
  cufftComplex *d_H_multi;
  float *d_recon1_multi;
  float *d_recon2_multi;
  cufftHandle plan1d_multi;
  cufftHandle plan2d_multi;
  int batch_plans_created;

  // Raw-sinogram batch buffers (lazy-allocated)
  int raw_batch_max_pairs;
  int raw_det_xdim;
  int raw_recon_xdim;
  float *d_raw_sino1_multi; // [n_pairs * det_xdim * n_angles]
  float *d_raw_sino2_multi;
  float *d_compact_recon1_multi; // [n_pairs * recon_xdim * recon_xdim]
  float *d_compact_recon2_multi;
};

// ═════════════════════════════════════════════════════════════
//  CUDA KERNELS
// ═════════════════════════════════════════════════════════════

// ── Load one sinogram row into cproj as complex (FFTW-bridge fallback) ──
__global__ void load_sinogram_row_kernel(
    cufftComplex *cproj, // output [pdim]
    const float *sino1,  // [theta_list_size * sinogram_x_dim]
    const float *sino2, int angle_idx, unsigned long sinogram_x_dim,
    long pdim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= pdim)
    return;

  if (j < (int)sinogram_x_dim) {
    long row_off = (long)angle_idx * sinogram_x_dim;
    cproj[j].x = sino1[row_off + j];
    cproj[j].y = sino2[row_off + j];
  } else {
    cproj[j].x = 0.0f;
    cproj[j].y = 0.0f;
  }
}

// ── Batched: load ALL sinogram rows into cproj_batch ──
// Grid: (ceil(pdim/256), theta_list_size), Block: (256, 1)
__global__ void load_all_sinogram_rows_kernel(
    cufftComplex *cproj_batch, // output [theta_list_size * pdim]
    const float *sino1, const float *sino2, unsigned long sinogram_x_dim,
    long pdim, int n_angles) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int angle = blockIdx.y;
  if (j >= pdim || angle >= n_angles)
    return;

  long out_idx = (long)angle * pdim + j;
  if (j < (int)sinogram_x_dim) {
    long row_off = (long)angle * sinogram_x_dim;
    cproj_batch[out_idx].x = sino1[row_off + j];
    cproj_batch[out_idx].y = sino2[row_off + j];
  } else {
    cproj_batch[out_idx].x = 0.0f;
    cproj_batch[out_idx].y = 0.0f;
  }
}

// ── Apply filter × phase to FFT output + scatter to H grid ──
// One thread per frequency bin j ∈ [1, pdim/2).
//
// CPU four1 interface:
//   four1((float *)param->cproj + 1, param->pdim, 1, param)
//   This copies from data+1 = (float*)cproj+2, i.e. starting at cproj[1],
//   forward pdim complex values → FFT → copy back.
//   After FFT: CPU reads cproj[j+1] for positive freq, cproj[(pdim-j)+1] for
//   mirror.
//
// Our GPU cuFFT operates on d_cproj[0..pdim-1] (0-indexed).
// cuFFT inverse of 0-indexed data ≡ FFTW backward of 1-indexed data shifted.
// Net mapping: CPU cproj[j+1] → GPU d_cproj[j], CPU cproj[(pdim-j)+1] → GPU
// d_cproj[pdim-j]. ── Shared scatter logic (used by both single and batched
// kernels) ──
__device__ void
scatter_to_H(cufftComplex *H, const float *filphase,
             const cufftComplex *cproj_row, // [pdim], post-FFT for one angle
             const float *wtbl, int angle_idx,
             int j, // frequency bin
             long pdim, long M, long ltbl, float L, float scale, float tblspcg,
             float cosE, float sinE) {
  float L2 = L / 2.0f;
  long M2 = M >> 1;

  float fr = filphase[j * 2];
  float fi = filphase[j * 2 + 1];

  float cr = cproj_row[j].x;
  float ci = cproj_row[j].y;
  float d1r = fr * cr - fi * ci;
  float d1i = fr * ci + fi * cr;

  float mr = cproj_row[pdim - j].x;
  float mi = cproj_row[pdim - j].y;
  float d2r = fr * mr + fi * mi;
  float d2i = -fi * mr + fr * mi;

  float rtmp = scale * j;
  float U = rtmp * cosE + M2;
  float V = rtmp * sinE + M2;

  long iul = (long)(ceilf(U - L2));
  long iuh = (long)(floorf(U + L2));
  long ivl = (long)(ceilf(V - L2));
  long ivh = (long)(floorf(V + L2));

  if (iul < 1)
    iul = 1;
  if (iuh >= M)
    iuh = M - 1;
  if (ivl < 1)
    ivl = 1;
  if (ivh >= M)
    ivh = M - 1;

  float work[16];
  int k = 0;
  for (long iv = ivl; iv <= ivh; iv++, k++) {
    float absV = fabsf(V - iv) * tblspcg;
#ifndef INTERP
    work[k] = wtbl[(int)(absV + 0.5f)];
#else
    int idx = (int)absV;
    work[k] = wtbl[idx] + (absV - idx) * (wtbl[idx + 1] - wtbl[idx]);
#endif
  }

  for (long iu = iul; iu <= iuh; iu++) {
    float absU = fabsf(U - iu) * tblspcg;
#ifndef INTERP
    float convU = wtbl[(int)(absU + 0.5f)];
#else
    int idx = (int)absU;
    float convU = wtbl[idx] + (absU - idx) * (wtbl[idx + 1] - wtbl[idx]);
#endif
    k = 0;
    for (long iv = ivl; iv <= ivh; iv++, k++) {
      float convolv = convU * work[k];
      long idx1 = iu * M + iv + 1;
      atomicAdd(&H[idx1].x, convolv * d1r);
      atomicAdd(&H[idx1].y, convolv * d1i);
      long idx2 = (M - iu) * M + (M - iv) + 1;
      atomicAdd(&H[idx2].x, convolv * d2r);
      atomicAdd(&H[idx2].y, convolv * d2i);
    }
  }
}

// ── Single-angle scatter (FFTW-bridge fallback) ──
__global__ void filter_and_scatter_kernel(
    cufftComplex *H, const cufftComplex *cproj, const float *filphase,
    const float *COSE, const float *SINE, const float *wtbl, int angle_idx,
    long pdim, long M, long ltbl, float L, float scale, float tblspcg) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  long pdim2 = pdim >> 1;
  if (j < 1 || j >= pdim2)
    return;

  scatter_to_H(H, filphase, cproj, wtbl, angle_idx, j, pdim, M, ltbl, L, scale,
               tblspcg, COSE[angle_idx], SINE[angle_idx]);
}

// ── Batched: scatter ALL angles at once ──
// Grid: (ceil(pdim2/256), n_angles), Block: (256, 1)
__global__ void filter_and_scatter_all_kernel(
    cufftComplex *H,
    const cufftComplex *cproj_batch, // [n_angles * pdim], post-FFT
    const float *filphase, const float *COSE, const float *SINE,
    const float *wtbl, long pdim, long M, long ltbl, float L, float scale,
    float tblspcg, int n_angles) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int angle = blockIdx.y;
  long pdim2 = pdim >> 1;
  if (j < 1 || j >= pdim2 || angle >= n_angles)
    return;

  const cufftComplex *cproj_row = cproj_batch + (long)angle * pdim;
  scatter_to_H(H, filphase, cproj_row, wtbl, angle, j, pdim, M, ltbl, L, scale,
               tblspcg, COSE[angle], SINE[angle]);
}

// ── Phase 3: PSWF correction + extraction ──
// CPU phase3 maps M×M Fourier grid to M0×M0 output using wrap-around
// indexing and PSWF correction.  Output S1/S2 have stride sinogram_x_dim
// (NOT M0!), matching setSinoAndReconBuffers.
//
// CPU loop structure:
//   j=0: iu starts at M-M02, goes to M-1   (M02+1 values, but first block is
//   M02 values since ufin=M is exclusive... wait) Actually: ustart=M-M02,
//   ufin=M → iu goes M-M02..M-1 → that's M02 iterations Then ustart=0,
//   ufin=M02+1 → iu goes 0..M02 → that's M02+1 iterations Total = M02 + M02 + 1
//   = 2*M02 + 1 = M0 ✓
//
// Mapping: output pixel (tj, tk) → grid (iu, iv):
//   tj ∈ [0, M02): iu = (M-M02) + tj
//   tj ∈ [M02, M0): iu = tj - M02
//   Same for tk → iv
__global__ void pswf_extract_kernel(
    const cufftComplex *H,
    float *S1, // [M0 * sinogram_x_dim]  (stride = sinogram_x_dim)
    float *S2,
    const float *winv, // [M0]
    long M, long M0, long M02,
    unsigned long sinogram_x_dim) // stride for output
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;
  if (tj >= M0 || tk >= M0)
    return;

  float corrn = winv[tj] * winv[tk];

  // Wrap-around: first M02 pixels come from end of grid, rest from start
  long iu, iv;
  if (tj < M02) {
    iu = (M - M02) + tj; // tj=0 → iu=M-M02, tj=M02-1 → iu=M-1
  } else {
    iu = tj - M02; // tj=M02 → iu=0, tj=M0-1 → iu=M02
  }
  if (tk < M02) {
    iv = (M - M02) + tk;
  } else {
    iv = tk - M02;
  }

  // H[iu*M + iv + 1] — same indexing as phase1 scatter
  long h_idx = iu * M + iv + 1;
  // Output uses sinogram_x_dim as stride (matching CPU's S1[j][k] =
  // recon1[j*sinogram_x_dim + k])
  long out_idx = (long)tj * sinogram_x_dim + tk;
  S1[out_idx] = corrn * H[h_idx].x;
  S2[out_idx] = corrn * H[h_idx].y;
}

// ═════════════════════════════════════════════════════════════
//  CONTEXT LIFECYCLE
// ═════════════════════════════════════════════════════════════

extern "C" void tomo_gpu_print_info(int deviceId) {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
    printf("TOMO GPU: could not query device %d\n", deviceId);
    return;
  }
  printf("TOMO GPU: %s, SM %d.%d, %.1f GB, %d SMs\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
         prop.multiProcessorCount);
}

extern "C" TomoGPUContext *
tomo_gpu_init(int deviceId, unsigned long sinogram_x_dim, int theta_list_size,
              const float *theta_list, int filter_type, int useFftwBridge) {
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess) {
    fprintf(stderr, "TOMO GPU: failed to set device %d: %s\n", deviceId,
            cudaGetErrorString(err));
    return NULL;
  }

  tomo_gpu_print_info(deviceId);

  TomoGPUContext *ctx = (TomoGPUContext *)calloc(1, sizeof(TomoGPUContext));
  if (!ctx)
    return NULL;

  ctx->deviceId = deviceId;
  ctx->sinogram_x_dim = sinogram_x_dim;
  ctx->theta_list_size = theta_list_size;
  ctx->filter_type = filter_type;
  ctx->useFftwBridge = useFftwBridge;
  ctx->tables_uploaded = 0;

  // ── Compute gridrec parameters (same as CPU initGridRec) ──
  long itmp;
  ctx->pdim = 1;
  itmp = sinogram_x_dim - 1;
  while (itmp) {
    ctx->pdim <<= 1;
    itmp >>= 1;
  }

  ctx->sampl = 1.0f;
  float D0 = 1.0f * sinogram_x_dim;
  float D1 = ctx->sampl * D0;
  ctx->M = 1;
  itmp = (long)(D1 / 1.0f - 1);
  while (itmp) {
    ctx->M <<= 1;
    itmp >>= 1;
  }

  ctx->M02 = (long)(floor(ctx->M / 2.0f / ctx->sampl - 0.5f));
  ctx->M0 = 2 * ctx->M02 + 1;
  ctx->sampl = (float)ctx->M / ctx->M0;
  D1 = ctx->sampl * D0;

  float C = 6.0f;
  ctx->L = 2 * C * ctx->sampl / (float)PI;
  ctx->scale = D1 / ctx->pdim;
  ctx->ltbl = 512;
  ctx->flag = 0;

  // ── Create CUDA streams ──
  CUDA_CHECK_VOID(cudaStreamCreate(&ctx->stream_h2d));
  CUDA_CHECK_VOID(cudaStreamCreate(&ctx->stream_compute));
  CUDA_CHECK_VOID(cudaStreamCreate(&ctx->stream_d2h));

  CUDA_CHECK_VOID(cudaEventCreate(&ctx->event_h2d_done));
  CUDA_CHECK_VOID(cudaEventCreate(&ctx->event_compute_done));
  CUDA_CHECK_VOID(cudaEventCreate(&ctx->event_d2h_done));

  // ── Allocate device tables (filled later by upload_tables) ──
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_SINE, theta_list_size * sizeof(float)));
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_COSE, theta_list_size * sizeof(float)));
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_wtbl, (ctx->ltbl + 1) * sizeof(float)));
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_winv, ctx->M0 * sizeof(float)));
  CUDA_CHECK_VOID(
      cudaMalloc(&ctx->d_filphase, (ctx->pdim / 2 + 1) * 2 * sizeof(float)));

  // ── Allocate device work buffers ──
  // Sinogram: flat array [theta_list_size * sinogram_x_dim]
  size_t sino_size = (size_t)theta_list_size * sinogram_x_dim * sizeof(float);
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_sino1, sino_size));
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_sino2, sino_size));

  // H grid: (M+1)*(M+1) complex values (1-indexed access pattern)
  size_t H_elems = (size_t)(ctx->M + 1) * (ctx->M + 1);
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_H, H_elems * sizeof(cufftComplex)));

  // 1D FFT work buffer (single — used by FFTW-bridge fallback)
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_cproj, ctx->pdim * sizeof(cufftComplex)));

  // Batched 1D FFT buffer: all angles at once
  size_t cproj_batch_size =
      (size_t)theta_list_size * ctx->pdim * sizeof(cufftComplex);
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_cproj_batch, cproj_batch_size));

  // Reconstruction output: M0 rows × sinogram_x_dim cols (stride =
  // sinogram_x_dim)
  size_t recon_size = (size_t)ctx->M0 * sinogram_x_dim * sizeof(float);
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_recon1, recon_size));
  CUDA_CHECK_VOID(cudaMalloc(&ctx->d_recon2, recon_size));

  // ── cuFFT plans ──
  if (!useFftwBridge) {
    // Single 1D plan (kept for FFTW-bridge fallback)
    cufftResult r1 = cufftPlan1d(&ctx->plan1d, (int)ctx->pdim, CUFFT_C2C, 1);
    // Batched 1D plan: all angles in one call
    int pdim_int = (int)ctx->pdim;
    cufftResult r1b =
        cufftPlanMany(&ctx->plan1d_batch, 1, &pdim_int, // rank, n
                      NULL, 1, pdim_int, // inembed, istride, idist
                      NULL, 1, pdim_int, // onembed, ostride, odist
                      CUFFT_C2C, theta_list_size);
    cufftResult r2 =
        cufftPlan2d(&ctx->plan2d, (int)ctx->M, (int)ctx->M, CUFFT_C2C);
    if (r1 != CUFFT_SUCCESS || r1b != CUFFT_SUCCESS || r2 != CUFFT_SUCCESS) {
      fprintf(stderr,
              "TOMO GPU: cuFFT plan creation failed (r1=%d, r1b=%d, r2=%d)\n",
              (int)r1, (int)r1b, (int)r2);
      tomo_gpu_destroy(ctx);
      return NULL;
    }
    cufftSetStream(ctx->plan1d, ctx->stream_compute);
    cufftSetStream(ctx->plan1d_batch, ctx->stream_compute);
    cufftSetStream(ctx->plan2d, ctx->stream_compute);
    ctx->cufft_plans_created = 1;
  }

  // ── Pinned host buffer for FFTW bridge ──
  if (useFftwBridge) {
    size_t bridge_size = H_elems * sizeof(cufftComplex);
    if (ctx->pdim * sizeof(cufftComplex) > bridge_size)
      bridge_size = ctx->pdim * sizeof(cufftComplex);
    CUDA_CHECK_VOID(cudaMallocHost(&ctx->h_fft_bridge, bridge_size));
    ctx->h_fft_bridge_size = bridge_size;
  }

  printf("TOMO GPU: context initialised — sinogram_x_dim=%lu, M=%ld, M0=%ld, "
         "pdim=%ld, L=%.1f, %s mode, 3 streams\n",
         sinogram_x_dim, ctx->M, ctx->M0, ctx->pdim, ctx->L,
         useFftwBridge ? "FFTW-bridge" : "cuFFT");

  return ctx;
}

extern "C" void tomo_gpu_destroy(TomoGPUContext *ctx) {
  if (!ctx)
    return;
  cudaSetDevice(ctx->deviceId);

  if (ctx->d_SINE)
    cudaFree(ctx->d_SINE);
  if (ctx->d_COSE)
    cudaFree(ctx->d_COSE);
  if (ctx->d_wtbl)
    cudaFree(ctx->d_wtbl);
  if (ctx->d_winv)
    cudaFree(ctx->d_winv);
  if (ctx->d_filphase)
    cudaFree(ctx->d_filphase);
  if (ctx->d_sino1)
    cudaFree(ctx->d_sino1);
  if (ctx->d_sino2)
    cudaFree(ctx->d_sino2);
  if (ctx->d_H)
    cudaFree(ctx->d_H);
  if (ctx->d_cproj)
    cudaFree(ctx->d_cproj);
  if (ctx->d_cproj_batch)
    cudaFree(ctx->d_cproj_batch);
  if (ctx->d_recon1)
    cudaFree(ctx->d_recon1);
  if (ctx->d_recon2)
    cudaFree(ctx->d_recon2);

  if (ctx->cufft_plans_created) {
    cufftDestroy(ctx->plan1d);
    cufftDestroy(ctx->plan1d_batch);
    cufftDestroy(ctx->plan2d);
  }
  // Multi-pair batch buffers
  if (ctx->batch_max_pairs > 0) {
    cudaFree(ctx->d_sino1_multi);
    cudaFree(ctx->d_sino2_multi);
    cudaFree(ctx->d_cproj_multi);
    cudaFree(ctx->d_H_multi);
    cudaFree(ctx->d_recon1_multi);
    cudaFree(ctx->d_recon2_multi);
    if (ctx->batch_plans_created) {
      cufftDestroy(ctx->plan1d_multi);
      cufftDestroy(ctx->plan2d_multi);
    }
  }
  if (ctx->h_fft_bridge)
    cudaFreeHost(ctx->h_fft_bridge);

  cudaStreamDestroy(ctx->stream_h2d);
  cudaStreamDestroy(ctx->stream_compute);
  cudaStreamDestroy(ctx->stream_d2h);
  cudaEventDestroy(ctx->event_h2d_done);
  cudaEventDestroy(ctx->event_compute_done);
  cudaEventDestroy(ctx->event_d2h_done);

  free(ctx);
  printf("TOMO GPU: context destroyed\n");
}

// ═════════════════════════════════════════════════════════════
//  MEMORY QUERY (for dynamic batch sizing)
// ═════════════════════════════════════════════════════════════

extern "C" size_t tomo_gpu_get_free_memory(void) {
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return free_bytes;
}

// ═════════════════════════════════════════════════════════════
//  PINNED MEMORY WRAPPERS (for use from C code)
// ═════════════════════════════════════════════════════════════

extern "C" int tomo_gpu_pinned_alloc(void **ptr, size_t size) {
  cudaError_t err = cudaMallocHost(ptr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "TOMO GPU: cudaMallocHost failed (%zu bytes): %s\n", size,
            cudaGetErrorString(err));
    return -1;
  }
  return 0;
}

extern "C" void tomo_gpu_pinned_free(void *ptr) {
  if (ptr)
    cudaFreeHost(ptr);
}

// ═════════════════════════════════════════════════════════════
//  TABLE UPLOAD (from CPU initGridRec → GPU)
// ═════════════════════════════════════════════════════════════

extern "C" int tomo_gpu_upload_tables(TomoGPUContext *ctx, const float *SINE,
                                      const float *COSE, const float *wtbl,
                                      const float *winv, const float *filphase,
                                      long ltbl, long M0, long pdim) {
  if (!ctx)
    return -1;
  cudaSetDevice(ctx->deviceId);

  CUDA_CHECK(cudaMemcpy(ctx->d_SINE, SINE, ctx->theta_list_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_COSE, COSE, ctx->theta_list_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_wtbl, wtbl, (ltbl + 1) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_winv, winv, M0 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // filphase is complex* on CPU = float pairs {r, i}
  // Size is (pdim/2 + 1) complex values = (pdim/2 + 1) * 2 floats
  CUDA_CHECK(cudaMemcpy(ctx->d_filphase, filphase,
                        (pdim / 2 + 1) * 2 * sizeof(float),
                        cudaMemcpyHostToDevice));

  ctx->tables_uploaded = 1;
  printf("TOMO GPU: tables uploaded (ltbl=%ld, M0=%ld, pdim=%ld)\n", ltbl, M0,
         pdim);
  return 0;
}

// ═════════════════════════════════════════════════════════════
//  GPU RECONSTRUCTION
// ═════════════════════════════════════════════════════════════

extern "C" int tomo_gpu_reconstruct(TomoGPUContext *ctx, const float *sinogram1,
                                    const float *sinogram2,
                                    float *reconstruction1,
                                    float *reconstruction2, long M, long M0,
                                    long M02, long pdim) {
  if (!ctx)
    return -1;
  if (!ctx->tables_uploaded) {
    fprintf(
        stderr,
        "TOMO GPU: tables not uploaded — call tomo_gpu_upload_tables first\n");
    return -1;
  }
  if (!sinogram1 || !sinogram2 || !reconstruction1 || !reconstruction2) {
    fprintf(stderr, "TOMO GPU: NULL buffer pointer\n");
    return -1;
  }
  cudaSetDevice(ctx->deviceId);

  long pdim2 = pdim >> 1;
  size_t sino_bytes =
      (size_t)ctx->theta_list_size * ctx->sinogram_x_dim * sizeof(float);
  size_t H_elems = (size_t)(M + 1) * (M + 1);
  size_t recon_bytes = (size_t)M0 * ctx->sinogram_x_dim * sizeof(float);
  float tblspcg = 2.0f * ctx->ltbl / ctx->L;

  // ════════════════════════════════════════════════════════
  // Step 1: H2D — copy sinograms to GPU
  // ════════════════════════════════════════════════════════
  CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino1, sinogram1, sino_bytes,
                             cudaMemcpyHostToDevice, ctx->stream_h2d));
  CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino2, sinogram2, sino_bytes,
                             cudaMemcpyHostToDevice, ctx->stream_h2d));
  CUDA_CHECK(cudaEventRecord(ctx->event_h2d_done, ctx->stream_h2d));

  // ════════════════════════════════════════════════════════
  // Step 2: Compute — phase1 + phase2 + phase3
  // ════════════════════════════════════════════════════════
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_compute, ctx->event_h2d_done, 0));

  // Clear H grid and recon buffers
  CUDA_CHECK(cudaMemsetAsync(ctx->d_H, 0, H_elems * sizeof(cufftComplex),
                             ctx->stream_compute));
  CUDA_CHECK(
      cudaMemsetAsync(ctx->d_recon1, 0, recon_bytes, ctx->stream_compute));
  CUDA_CHECK(
      cudaMemsetAsync(ctx->d_recon2, 0, recon_bytes, ctx->stream_compute));

  // ─── Phase 1: 1D FFT + filter + PSWF scatter ───
  int blockSize = 256;
  int gridSize_pdim = ((int)pdim + blockSize - 1) / blockSize;
  int gridSize_scatter = ((int)pdim2 + blockSize - 1) / blockSize;
  int n_angles = ctx->theta_list_size;

  if (ctx->useFftwBridge) {
    // FFTW-bridge: must stay serial (CPU FFT per angle)
    for (int n = 0; n < n_angles; n++) {
      load_sinogram_row_kernel<<<gridSize_pdim, blockSize, 0,
                                 ctx->stream_compute>>>(
          ctx->d_cproj, ctx->d_sino1, ctx->d_sino2, n, ctx->sinogram_x_dim,
          pdim);

      CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
      CUDA_CHECK(cudaMemcpy(ctx->h_fft_bridge, ctx->d_cproj,
                            pdim * sizeof(cufftComplex),
                            cudaMemcpyDeviceToHost));
      {
        fftwf_complex *fft_buf = (fftwf_complex *)ctx->h_fft_bridge;
        fftwf_plan plan = fftwf_plan_dft_1d((int)pdim, fft_buf, fft_buf,
                                            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
      }
      CUDA_CHECK(cudaMemcpy(ctx->d_cproj, ctx->h_fft_bridge,
                            pdim * sizeof(cufftComplex),
                            cudaMemcpyHostToDevice));

      filter_and_scatter_kernel<<<gridSize_scatter, blockSize, 0,
                                  ctx->stream_compute>>>(
          ctx->d_H, ctx->d_cproj, ctx->d_filphase, ctx->d_COSE, ctx->d_SINE,
          ctx->d_wtbl, n, pdim, M, ctx->ltbl, ctx->L, ctx->scale, tblspcg);
    }
  } else {
    // Batched cuFFT path: 3 operations instead of 3×n_angles

    // 1a. Load ALL sinogram rows into cproj_batch
    dim3 loadGrid(gridSize_pdim, n_angles);
    load_all_sinogram_rows_kernel<<<loadGrid, blockSize, 0,
                                    ctx->stream_compute>>>(
        ctx->d_cproj_batch, ctx->d_sino1, ctx->d_sino2, ctx->sinogram_x_dim,
        pdim, n_angles);

    // 1b. Batched 1D backward FFT (all angles at once)
    CUFFT_CHECK(cufftExecC2C(ctx->plan1d_batch, ctx->d_cproj_batch,
                             ctx->d_cproj_batch, CUFFT_INVERSE));

    // 1c. Batched filter+scatter (all angles at once)
    dim3 scatterGrid(gridSize_scatter, n_angles);
    filter_and_scatter_all_kernel<<<scatterGrid, blockSize, 0,
                                    ctx->stream_compute>>>(
        ctx->d_H, ctx->d_cproj_batch, ctx->d_filphase, ctx->d_COSE, ctx->d_SINE,
        ctx->d_wtbl, pdim, M, ctx->ltbl, ctx->L, ctx->scale, tblspcg, n_angles);
  }

  // ─── Phase 2: 2D forward FFT of H grid ───
  if (ctx->useFftwBridge) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
    size_t fft2d_bytes = (size_t)M * M * sizeof(cufftComplex);
    CUDA_CHECK(cudaMemcpy(ctx->h_fft_bridge, ctx->d_H + 1, fft2d_bytes,
                          cudaMemcpyDeviceToHost));
    {
      fftwf_complex *fft_buf = (fftwf_complex *)ctx->h_fft_bridge;
      fftwf_plan plan = fftwf_plan_dft_2d((int)M, (int)M, fft_buf, fft_buf,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_H + 1, ctx->h_fft_bridge, fft2d_bytes,
                          cudaMemcpyHostToDevice));
  } else {
    CUFFT_CHECK(
        cufftExecC2C(ctx->plan2d, ctx->d_H + 1, ctx->d_H + 1, CUFFT_FORWARD));
  }

  // ─── Phase 3: PSWF extraction ───
  dim3 block3(16, 16);
  dim3 grid3(((int)M0 + 15) / 16, ((int)M0 + 15) / 16);
  pswf_extract_kernel<<<grid3, block3, 0, ctx->stream_compute>>>(
      ctx->d_H, ctx->d_recon1, ctx->d_recon2, ctx->d_winv, M, M0, M02,
      ctx->sinogram_x_dim);

  CUDA_CHECK(cudaEventRecord(ctx->event_compute_done, ctx->stream_compute));

  // ════════════════════════════════════════════════════════
  // Step 3: D2H — copy reconstructions back
  // ════════════════════════════════════════════════════════
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_d2h, ctx->event_compute_done, 0));
  CUDA_CHECK(cudaMemcpyAsync(reconstruction1, ctx->d_recon1, recon_bytes,
                             cudaMemcpyDeviceToHost, ctx->stream_d2h));
  CUDA_CHECK(cudaMemcpyAsync(reconstruction2, ctx->d_recon2, recon_bytes,
                             cudaMemcpyDeviceToHost, ctx->stream_d2h));
  CUDA_CHECK(cudaEventRecord(ctx->event_d2h_done, ctx->stream_d2h));

  // Synchronize — caller expects results to be ready
  CUDA_CHECK(cudaStreamSynchronize(ctx->stream_d2h));

  return 0;
}

// ═════════════════════════════════════════════════════════════
//  GPU PREPROCESSING KERNELS
// ═════════════════════════════════════════════════════════════

// Fused Pad + shift + boundary-replicate kernel
// Input:  raw sinogram (det_xdim × n_angles) per slice — compact detector
// layout Output: boundary-padded sinogram (sinogram_x_dim × n_angles) per slice
//         ready for the cuFFT load kernel
//
// Each output pixel (angle, col) in the sinogram_x_dim-wide row:
//   1. The output has stride sinogram_x_dim (= adjusted_xdim * 2)
//   2. The adjusted_xdim center portion lives at [adjusted_xdim/2 ..
//   adjusted_xdim/2 + adjusted_xdim)
//   3. Within that center: sub-pixel shift interpolation from padded source
//   4. Left/right edges: boundary replication (constant extension)
//
// Parameters that are CONSTANT across all sinograms (computed once on host):
//   det_xdim, adjusted_xdim, sinogram_x_dim, pad_front, shift_int, w0, w1
__global__ void pad_shift_boundary_kernel(
    float *out,            // output: [sinogram_x_dim × n_angles], pre-processed
    const float *raw_sino, // input:  [det_xdim × n_angles], raw from detector
    int det_xdim,          // detector width (e.g. 2048)
    int adjusted_xdim,     // padded width (e.g. 2048 for pow2, or next_pow2)
    int sinogram_x_dim,    // output stride = adjusted_xdim * 2 (e.g. 4096)
    int n_angles,          // number of projection angles
    int pad_front,         // front padding size for Pad()
    int sino_xdim,         // original sinogram_xdim (may differ from det_xdim)
    int shift_int,         // integer part of shift
    float w0, float w1,    // interpolation weights (1-frac, frac)
    int doLog)             // apply -log transform?
{
  int col = blockIdx.x * blockDim.x +
            threadIdx.x;  // output column [0..sinogram_x_dim)
  int angle = blockIdx.y; // angle index
  if (col >= sinogram_x_dim || angle >= n_angles)
    return;

  long out_idx = (long)angle * sinogram_x_dim + col;

  // Step 1: determine position within the adjusted_xdim center band
  int half = adjusted_xdim / 2;
  // The center band in the output is [half .. half + adjusted_xdim)
  // Boundary replication: left of half = value at half, right of
  // half+adjusted_xdim = value at half+adjusted_xdim-1

  // First compute the shifted+padded sinogram value at position 'col_center'
  // where col_center is the column index within the adjusted_xdim center band
  int col_center = col - half; // position relative to center band start

  float val;
  if (col_center < 0) {
    // Left boundary replication region — will be filled after we know the edge
    // value We need the value at col_center=0 (the leftmost center pixel)
    col_center = 0;
  } else if (col_center >= adjusted_xdim) {
    // Right boundary replication region
    col_center = adjusted_xdim - 1;
  }

  // Step 2: Apply sub-pixel shift interpolation
  // col_center is now in [0..adjusted_xdim), which maps to the
  // "sino_calc_buffer" space The shift interpolation: dst[k] =
  // src[k-shift_int]*w0 + src[k-shift_int-1]*w1 where src is the padded
  // sinogram (after Pad)
  int nkk = col_center - shift_int;

  // Step 3: Get the Pad'd value at position nkk (and nkk-1) in the padded
  // sinogram Pad maps: padded[col] = raw[col - pad_front] with edge replication
  auto getPadded = [&](int padded_col) -> float {
    float v;
    if (padded_col < pad_front) {
      // Left pad: replicate first pixel
      v = raw_sino[(long)angle * det_xdim + 0];
    } else if (padded_col >= pad_front + sino_xdim) {
      // Right pad: replicate last pixel
      v = raw_sino[(long)angle * det_xdim + (sino_xdim - 1)];
    } else {
      // Center: direct mapping
      v = raw_sino[(long)angle * det_xdim + (padded_col - pad_front)];
    }
    if (doLog && v > 0.0f)
      v = -logf(v);
    return v;
  };

  float fInterpPixel = 0.0f;
  float fInterpWeight = 0.0f;
  if (nkk >= 0 && nkk < adjusted_xdim) {
    fInterpPixel += getPadded(nkk) * w0;
    fInterpWeight += w0;
  }
  if (nkk - 1 >= 0 && nkk - 1 < adjusted_xdim) {
    fInterpPixel += getPadded(nkk - 1) * w1;
    fInterpWeight += w1;
  }
  val = (fInterpWeight < 1e-5f) ? 0.0f : fInterpPixel / fInterpWeight;

  out[out_idx] = val;
}

// Extract compact reconstruction from boundary-padded GPU output
// Input:  recon_boundary_padding [M0 × sinogram_x_dim] on GPU
// Output: compact recon [recon_xdim × recon_xdim] for direct D2H
__global__ void extract_recon_kernel(
    float *compact_out,    // output: [recon_xdim × recon_xdim]
    const float *recon_bp, // input:  [M0 × sinogram_x_dim] boundary-padded
    int recon_xdim,        // reconstruction width (e.g. 2048)
    int sinogram_x_dim,    // padded stride (e.g. 4096)
    int M0,                // reconstruction grid dim (e.g. 4095)
    int shift_round,       // round(shift) for auto-centering
    int auto_centering)    // whether to apply shift correction
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= recon_xdim || row >= recon_xdim)
    return;

  // getRecons extraction: source at (row + xdim/2, col + xdim/2) in
  // boundary-padded buffer
  int half = recon_xdim / 2;
  int src_row = row + half;
  int src_col = col + half;
  float val = recon_bp[(long)src_row * sinogram_x_dim + src_col];

  // Auto-centering shift
  if (auto_centering) {
    int src_col_shifted;
    if (shift_round >= 0) {
      // shifted_recon[j*xdim + k] = recon_buffer[j*xdim + k + shift_round]
      src_col_shifted = col + shift_round;
      if (src_col_shifted >= recon_xdim)
        val = 0.0f;
      else {
        int sc = src_col_shifted + half;
        val = recon_bp[(long)src_row * sinogram_x_dim + sc];
      }
    } else {
      int abs_shift = -shift_round;
      if (col < abs_shift)
        val = 0.0f;
      else {
        int sc = (col - abs_shift) + half;
        val = recon_bp[(long)src_row * sinogram_x_dim + sc];
      }
    }
  }

  compact_out[(long)row * recon_xdim + col] = val;
}

// ═════════════════════════════════════════════════════════════
//  MULTI-PAIR BATCHED GPU RECONSTRUCTION
// ═════════════════════════════════════════════════════════════

// Multi-pair load kernel: 3D grid (pdim_blocks, n_angles, n_pairs)
__global__ void load_all_sinogram_rows_multi_kernel(
    cufftComplex *cproj_multi, // [n_pairs * n_angles * pdim]
    const float *sino1_flat,   // [n_pairs * n_angles * sinogram_x_dim]
    const float *sino2_flat, unsigned long sinogram_x_dim, long pdim,
    int n_angles, int n_pairs) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int angle = blockIdx.y;
  int pair = blockIdx.z;
  if (j >= pdim || angle >= n_angles || pair >= n_pairs)
    return;

  long sino_stride = (long)n_angles * sinogram_x_dim;
  long cproj_stride = (long)n_angles * pdim;
  long out_idx = (long)pair * cproj_stride + (long)angle * pdim + j;

  if (j < (int)sinogram_x_dim) {
    long in_idx = (long)pair * sino_stride + (long)angle * sinogram_x_dim + j;
    cproj_multi[out_idx].x = sino1_flat[in_idx];
    cproj_multi[out_idx].y = sino2_flat[in_idx];
  } else {
    cproj_multi[out_idx].x = 0.0f;
    cproj_multi[out_idx].y = 0.0f;
  }
}

// Multi-pair scatter kernel: 3D grid (freq_blocks, n_angles, n_pairs)
__global__ void filter_and_scatter_multi_kernel(
    cufftComplex *H_multi,           // [n_pairs * H_stride]
    const cufftComplex *cproj_multi, // [n_pairs * n_angles * pdim]
    const float *filphase, const float *COSE, const float *SINE,
    const float *wtbl, long pdim, long M, long ltbl, float L, float scale,
    float tblspcg, int n_angles, int n_pairs,
    long H_stride) // (M+1)*(M+1)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int angle = blockIdx.y;
  int pair = blockIdx.z;
  long pdim2 = pdim >> 1;
  if (j < 1 || j >= pdim2 || angle >= n_angles || pair >= n_pairs)
    return;

  cufftComplex *H = H_multi + (long)pair * H_stride;
  const cufftComplex *cproj_row =
      cproj_multi + (long)pair * n_angles * pdim + (long)angle * pdim;

  scatter_to_H(H, filphase, cproj_row, wtbl, angle, j, pdim, M, ltbl, L, scale,
               tblspcg, COSE[angle], SINE[angle]);
}

// Multi-pair extract kernel: 3D grid (M0_blocks, M0_blocks, n_pairs)
__global__ void
pswf_extract_multi_kernel(const cufftComplex *H_multi, // [n_pairs * H_stride]
                          float *recon1_flat, // [n_pairs * M0 * sinogram_x_dim]
                          float *recon2_flat, const float *winv, long M,
                          long M0, long M02, unsigned long sinogram_x_dim,
                          int n_pairs, long H_stride,
                          long recon_stride) // M0 * sinogram_x_dim floats
{
  int tj = blockIdx.x * blockDim.x + threadIdx.x;
  int tk = blockIdx.y * blockDim.y + threadIdx.y;
  int pair = blockIdx.z;
  if (tj >= M0 || tk >= M0 || pair >= n_pairs)
    return;

  float corrn = winv[tj] * winv[tk];

  long iu = (tj < M02) ? (M - M02) + tj : tj - M02;
  long iv = (tk < M02) ? (M - M02) + tk : tk - M02;

  long h_idx = (long)pair * H_stride + iu * M + iv + 1;
  long out_idx = (long)pair * recon_stride + (long)tj * sinogram_x_dim + tk;
  recon1_flat[out_idx] = corrn * H_multi[h_idx].x;
  recon2_flat[out_idx] = corrn * H_multi[h_idx].y;
}

extern "C" int tomo_gpu_reconstruct_batch(TomoGPUContext *ctx, int n_pairs,
                                          const float **sinogram1s,
                                          const float **sinogram2s,
                                          float **recon1s, float **recon2s,
                                          long M, long M0, long M02,
                                          long pdim) {
  if (!ctx || n_pairs <= 0)
    return -1;
  if (!ctx->tables_uploaded)
    return -1;

  // Fall back to single-pair for n_pairs == 1 or FFTW-bridge mode
  if (n_pairs == 1 || ctx->useFftwBridge) {
    for (int i = 0; i < n_pairs; i++) {
      int rc = tomo_gpu_reconstruct(ctx, sinogram1s[i], sinogram2s[i],
                                    recon1s[i], recon2s[i], M, M0, M02, pdim);
      if (rc != 0)
        return rc;
    }
    return 0;
  }

  cudaSetDevice(ctx->deviceId);

  double t0 = gpu_timer_sec();
  double t_alloc = 0, t_plan = 0, t_h2d, t_compute, t_d2h;

  long pdim2 = pdim >> 1;
  int n_angles = ctx->theta_list_size;
  size_t sino_per_pair = (size_t)n_angles * ctx->sinogram_x_dim;
  size_t sino_pair_bytes = sino_per_pair * sizeof(float);
  size_t H_stride = (size_t)(M + 1) * (M + 1);
  size_t recon_per_pair = (size_t)M0 * ctx->sinogram_x_dim;
  size_t recon_pair_bytes = recon_per_pair * sizeof(float);
  float tblspcg = 2.0f * ctx->ltbl / ctx->L;
  size_t cproj_per_pair = (size_t)n_angles * pdim;

  // ── Lazy-allocate batch GPU buffers (once, reused across all batches) ──
  if (ctx->batch_max_pairs < n_pairs) {
    // Free old buffers if resizing
    if (ctx->batch_max_pairs > 0) {
      cudaFree(ctx->d_sino1_multi);
      cudaFree(ctx->d_sino2_multi);
      cudaFree(ctx->d_cproj_multi);
      cudaFree(ctx->d_H_multi);
      cudaFree(ctx->d_recon1_multi);
      cudaFree(ctx->d_recon2_multi);
      if (ctx->batch_plans_created) {
        cufftDestroy(ctx->plan1d_multi);
        cufftDestroy(ctx->plan2d_multi);
      }
    }

    double ta = gpu_timer_sec();
    size_t total_gpu_mb =
        ((size_t)n_pairs *
         (2 * sino_pair_bytes + cproj_per_pair * sizeof(cufftComplex) +
          H_stride * sizeof(cufftComplex) + 2 * recon_pair_bytes)) >>
        20;

    CUDA_CHECK(
        cudaMalloc(&ctx->d_sino1_multi, (size_t)n_pairs * sino_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_sino2_multi, (size_t)n_pairs * sino_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_cproj_multi,
                   (size_t)n_pairs * cproj_per_pair * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&ctx->d_H_multi,
                          (size_t)n_pairs * H_stride * sizeof(cufftComplex)));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_recon1_multi, (size_t)n_pairs * recon_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_recon2_multi, (size_t)n_pairs * recon_pair_bytes));
    t_alloc = gpu_timer_sec() - ta;

    double tp = gpu_timer_sec();
    int pdim_int = (int)pdim;
    cufftResult r1 =
        cufftPlanMany(&ctx->plan1d_multi, 1, &pdim_int, NULL, 1, pdim_int, NULL,
                      1, pdim_int, CUFFT_C2C, n_pairs * n_angles);
    int M_int = (int)M;
    int n2d[2] = {M_int, M_int};
    int inembed[2] = {M_int, M_int};
    cufftResult r2 =
        cufftPlanMany(&ctx->plan2d_multi, 2, n2d, inembed, 1, (int)H_stride,
                      inembed, 1, (int)H_stride, CUFFT_C2C, n_pairs);
    if (r1 != CUFFT_SUCCESS || r2 != CUFFT_SUCCESS) {
      fprintf(stderr,
              "TOMO GPU: batch cuFFT plan creation failed (r1=%d, r2=%d)\n",
              (int)r1, (int)r2);
      return -1;
    }
    cufftSetStream(ctx->plan1d_multi, ctx->stream_compute);
    cufftSetStream(ctx->plan2d_multi, ctx->stream_compute);
    ctx->batch_plans_created = 1;
    t_plan = gpu_timer_sec() - tp;

    ctx->batch_max_pairs = n_pairs;
    fprintf(stderr,
            "TOMO GPU: batch buffers allocated (%zu MB, %d pairs): "
            "alloc=%.3fs plan=%.3fs\n",
            total_gpu_mb, n_pairs, t_alloc, t_plan);
  }

  // ── H2D: copy all N sinogram pairs to GPU ──
  double th = gpu_timer_sec();
  for (int i = 0; i < n_pairs; i++) {
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino1_multi + i * sino_per_pair,
                               sinogram1s[i], sino_pair_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino2_multi + i * sino_per_pair,
                               sinogram2s[i], sino_pair_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
  }
  CUDA_CHECK(cudaEventRecord(ctx->event_h2d_done, ctx->stream_h2d));
  // No CPU sync needed — stream_compute waits via event
  t_h2d = gpu_timer_sec() - th;

  // ── Compute ──
  double tc = gpu_timer_sec();
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_compute, ctx->event_h2d_done, 0));

  // Clear all H grids and recon buffers
  CUDA_CHECK(cudaMemsetAsync(ctx->d_H_multi, 0,
                             (size_t)n_pairs * H_stride * sizeof(cufftComplex),
                             ctx->stream_compute));
  CUDA_CHECK(cudaMemsetAsync(ctx->d_recon1_multi, 0,
                             (size_t)n_pairs * recon_pair_bytes,
                             ctx->stream_compute));
  CUDA_CHECK(cudaMemsetAsync(ctx->d_recon2_multi, 0,
                             (size_t)n_pairs * recon_pair_bytes,
                             ctx->stream_compute));

  // Phase 1: load + FFT + scatter (all pairs × all angles)
  {
    int blockSize = 256;
    int gridSize_pdim = ((int)pdim + blockSize - 1) / blockSize;
    int gridSize_scatter = ((int)pdim2 + blockSize - 1) / blockSize;

    dim3 loadGrid(gridSize_pdim, n_angles, n_pairs);
    load_all_sinogram_rows_multi_kernel<<<loadGrid, blockSize, 0,
                                          ctx->stream_compute>>>(
        ctx->d_cproj_multi, ctx->d_sino1_multi, ctx->d_sino2_multi,
        ctx->sinogram_x_dim, pdim, n_angles, n_pairs);

    CUFFT_CHECK(cufftExecC2C(ctx->plan1d_multi, ctx->d_cproj_multi,
                             ctx->d_cproj_multi, CUFFT_INVERSE));

    dim3 scatterGrid(gridSize_scatter, n_angles, n_pairs);
    filter_and_scatter_multi_kernel<<<scatterGrid, blockSize, 0,
                                      ctx->stream_compute>>>(
        ctx->d_H_multi, ctx->d_cproj_multi, ctx->d_filphase, ctx->d_COSE,
        ctx->d_SINE, ctx->d_wtbl, pdim, M, ctx->ltbl, ctx->L, ctx->scale,
        tblspcg, n_angles, n_pairs, (long)H_stride);
  }

  // Phase 2: batched 2D FFT
  CUFFT_CHECK(cufftExecC2C(ctx->plan2d_multi, ctx->d_H_multi + 1,
                           ctx->d_H_multi + 1, CUFFT_FORWARD));

  // Phase 3: batched extraction
  {
    dim3 block3(16, 16);
    dim3 grid3(((int)M0 + 15) / 16, ((int)M0 + 15) / 16, n_pairs);
    pswf_extract_multi_kernel<<<grid3, block3, 0, ctx->stream_compute>>>(
        ctx->d_H_multi, ctx->d_recon1_multi, ctx->d_recon2_multi, ctx->d_winv,
        M, M0, M02, ctx->sinogram_x_dim, n_pairs, (long)H_stride,
        (long)recon_per_pair);
  }

  CUDA_CHECK(cudaEventRecord(ctx->event_compute_done, ctx->stream_compute));
  // No CPU sync needed — stream_d2h waits via event
  t_compute = gpu_timer_sec() - tc;

  // ── D2H: copy all results back ──
  double td = gpu_timer_sec();
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_d2h, ctx->event_compute_done, 0));
  for (int i = 0; i < n_pairs; i++) {
    CUDA_CHECK(cudaMemcpyAsync(
        recon1s[i], ctx->d_recon1_multi + i * recon_per_pair, recon_pair_bytes,
        cudaMemcpyDeviceToHost, ctx->stream_d2h));
    CUDA_CHECK(cudaMemcpyAsync(
        recon2s[i], ctx->d_recon2_multi + i * recon_per_pair, recon_pair_bytes,
        cudaMemcpyDeviceToHost, ctx->stream_d2h));
  }
  CUDA_CHECK(cudaStreamSynchronize(ctx->stream_d2h));
  t_d2h = gpu_timer_sec() - td;

  double t_total = gpu_timer_sec() - t0;
  fprintf(stderr,
          "TOMO GPU batch(%d pairs): "
          "H2D=%.3fs compute=%.3fs D2H=%.3fs total=%.3fs\n",
          n_pairs, t_h2d, t_compute, t_d2h, t_total);

  return 0;
}

// ═════════════════════════════════════════════════════════════
//  GPU PREPROCESSING (stub)
// ═════════════════════════════════════════════════════════════

extern "C" int tomo_gpu_preprocess(TomoGPUContext *ctx,
                                   const unsigned short *short_sinogram,
                                   const float *dark_field,
                                   const float *white_field,
                                   float *norm_sino_out, int det_xdim,
                                   int adjusted_xdim, int theta_list_size,
                                   int doLog, float shift, float ring_coeff) {
  if (!ctx)
    return -1;
  fprintf(stderr, "TOMO GPU: preprocess not yet implemented, use CPU path\n");
  return -1;
}

// ═════════════════════════════════════════════════════════════
//  RAW-SINOGRAM BATCH RECONSTRUCTION
// ═════════════════════════════════════════════════════════════

extern "C" int tomo_gpu_reconstruct_batch_raw(
    TomoGPUContext *ctx, int n_pairs, const float **raw_sino1s,
    const float **raw_sino2s, float **compact_recon1s, float **compact_recon2s,
    long M, long M0, long M02, long pdim, int det_xdim, int adjusted_xdim,
    int sino_xdim, int recon_xdim, int pad_front, float shift, int doLog,
    int auto_centering) {
  if (!ctx || n_pairs <= 0)
    return -1;
  cudaSetDevice(ctx->deviceId);

  double t0 = gpu_timer_sec();
  int n_angles = ctx->theta_list_size;
  int sinogram_x_dim = (int)ctx->sinogram_x_dim;
  size_t raw_sino_per_slice = (size_t)det_xdim * n_angles;
  size_t raw_sino_bytes = raw_sino_per_slice * sizeof(float);
  size_t compact_recon_per_slice = (size_t)recon_xdim * recon_xdim;
  size_t compact_recon_bytes = compact_recon_per_slice * sizeof(float);
  long pdim2 = pdim >> 1;
  size_t sino_per_pair = (size_t)n_angles * sinogram_x_dim;
  size_t H_stride = (size_t)(M + 1) * (M + 1);
  size_t recon_per_pair = (size_t)M0 * sinogram_x_dim;
  float tblspcg = 2.0f * ctx->ltbl / ctx->L;
  size_t cproj_per_pair = (size_t)n_angles * pdim;

  // Precompute shift parameters (constant for all sinograms)
  int shift_int = (int)floorf(shift);
  float frac = shift - (float)shift_int;
  float w0 = 1.0f - frac;
  float w1 = frac;
  int shift_round = (int)roundf(shift);

  // ── Lazy-allocate raw batch GPU buffers ──
  if (ctx->raw_batch_max_pairs < n_pairs || ctx->raw_det_xdim != det_xdim ||
      ctx->raw_recon_xdim != recon_xdim) {
    // Free old
    if (ctx->raw_batch_max_pairs > 0) {
      cudaFree(ctx->d_raw_sino1_multi);
      cudaFree(ctx->d_raw_sino2_multi);
      cudaFree(ctx->d_compact_recon1_multi);
      cudaFree(ctx->d_compact_recon2_multi);
    }
    CUDA_CHECK(
        cudaMalloc(&ctx->d_raw_sino1_multi, (size_t)n_pairs * raw_sino_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_raw_sino2_multi, (size_t)n_pairs * raw_sino_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_compact_recon1_multi,
                          (size_t)n_pairs * compact_recon_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_compact_recon2_multi,
                          (size_t)n_pairs * compact_recon_bytes));
    ctx->raw_batch_max_pairs = n_pairs;
    ctx->raw_det_xdim = det_xdim;
    ctx->raw_recon_xdim = recon_xdim;
  }

  // ── Ensure cuFFT-format batch buffers exist (reuse from regular batch) ──
  size_t sino_pair_bytes = sino_per_pair * sizeof(float);
  size_t recon_pair_bytes = recon_per_pair * sizeof(float);
  if (ctx->batch_max_pairs < n_pairs) {
    if (ctx->batch_max_pairs > 0) {
      cudaFree(ctx->d_sino1_multi);
      cudaFree(ctx->d_sino2_multi);
      cudaFree(ctx->d_cproj_multi);
      cudaFree(ctx->d_H_multi);
      cudaFree(ctx->d_recon1_multi);
      cudaFree(ctx->d_recon2_multi);
      if (ctx->batch_plans_created) {
        cufftDestroy(ctx->plan1d_multi);
        cufftDestroy(ctx->plan2d_multi);
      }
    }
    CUDA_CHECK(
        cudaMalloc(&ctx->d_sino1_multi, (size_t)n_pairs * sino_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_sino2_multi, (size_t)n_pairs * sino_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_cproj_multi,
                   (size_t)n_pairs * cproj_per_pair * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&ctx->d_H_multi,
                          (size_t)n_pairs * H_stride * sizeof(cufftComplex)));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_recon1_multi, (size_t)n_pairs * recon_pair_bytes));
    CUDA_CHECK(
        cudaMalloc(&ctx->d_recon2_multi, (size_t)n_pairs * recon_pair_bytes));

    int pdim_int = (int)pdim;
    cufftPlanMany(&ctx->plan1d_multi, 1, &pdim_int, NULL, 1, pdim_int, NULL, 1,
                  pdim_int, CUFFT_C2C, n_pairs * n_angles);
    int M_int = (int)M;
    int n2d[2] = {M_int, M_int};
    int inembed[2] = {M_int, M_int};
    cufftPlanMany(&ctx->plan2d_multi, 2, n2d, inembed, 1, (int)H_stride,
                  inembed, 1, (int)H_stride, CUFFT_C2C, n_pairs);
    cufftSetStream(ctx->plan1d_multi, ctx->stream_compute);
    cufftSetStream(ctx->plan2d_multi, ctx->stream_compute);
    ctx->batch_plans_created = 1;
    ctx->batch_max_pairs = n_pairs;
  }

  // ── H2D: upload raw sinograms (smaller than boundary-padded) ──
  double th = gpu_timer_sec();
  for (int i = 0; i < n_pairs; i++) {
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_raw_sino1_multi + i * raw_sino_per_slice,
                               raw_sino1s[i], raw_sino_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_raw_sino2_multi + i * raw_sino_per_slice,
                               raw_sino2s[i], raw_sino_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
  }
  CUDA_CHECK(cudaEventRecord(ctx->event_h2d_done, ctx->stream_h2d));
  double t_h2d = gpu_timer_sec() - th;

  // ── GPU preprocessing: pad + shift + boundary-replicate ──
  double tp = gpu_timer_sec();
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_compute, ctx->event_h2d_done, 0));
  {
    int blockSize = 256;
    int gridX = (sinogram_x_dim + blockSize - 1) / blockSize;
    // Process all slices (2 per pair)
    for (int i = 0; i < n_pairs; i++) {
      dim3 grid(gridX, n_angles);
      // Slice 1
      pad_shift_boundary_kernel<<<grid, blockSize, 0, ctx->stream_compute>>>(
          ctx->d_sino1_multi + i * sino_per_pair,
          ctx->d_raw_sino1_multi + i * raw_sino_per_slice, det_xdim,
          adjusted_xdim, sinogram_x_dim, n_angles, pad_front, sino_xdim,
          shift_int, w0, w1, doLog);
      // Slice 2
      pad_shift_boundary_kernel<<<grid, blockSize, 0, ctx->stream_compute>>>(
          ctx->d_sino2_multi + i * sino_per_pair,
          ctx->d_raw_sino2_multi + i * raw_sino_per_slice, det_xdim,
          adjusted_xdim, sinogram_x_dim, n_angles, pad_front, sino_xdim,
          shift_int, w0, w1, doLog);
    }
  }
  double t_preproc = gpu_timer_sec() - tp;

  // ── Compute: same pipeline as regular batch ──
  double tc = gpu_timer_sec();
  CUDA_CHECK(cudaMemsetAsync(ctx->d_H_multi, 0,
                             (size_t)n_pairs * H_stride * sizeof(cufftComplex),
                             ctx->stream_compute));
  CUDA_CHECK(cudaMemsetAsync(ctx->d_recon1_multi, 0,
                             (size_t)n_pairs * recon_pair_bytes,
                             ctx->stream_compute));
  CUDA_CHECK(cudaMemsetAsync(ctx->d_recon2_multi, 0,
                             (size_t)n_pairs * recon_pair_bytes,
                             ctx->stream_compute));

  // Phase 1: load + FFT + scatter
  {
    int blockSize = 256;
    int gridSize_pdim = ((int)pdim + blockSize - 1) / blockSize;
    int gridSize_scatter = ((int)pdim2 + blockSize - 1) / blockSize;

    dim3 loadGrid(gridSize_pdim, n_angles, n_pairs);
    load_all_sinogram_rows_multi_kernel<<<loadGrid, blockSize, 0,
                                          ctx->stream_compute>>>(
        ctx->d_cproj_multi, ctx->d_sino1_multi, ctx->d_sino2_multi,
        ctx->sinogram_x_dim, pdim, n_angles, n_pairs);

    CUFFT_CHECK(cufftExecC2C(ctx->plan1d_multi, ctx->d_cproj_multi,
                             ctx->d_cproj_multi, CUFFT_INVERSE));

    dim3 scatterGrid(gridSize_scatter, n_angles, n_pairs);
    filter_and_scatter_multi_kernel<<<scatterGrid, blockSize, 0,
                                      ctx->stream_compute>>>(
        ctx->d_H_multi, ctx->d_cproj_multi, ctx->d_filphase, ctx->d_COSE,
        ctx->d_SINE, ctx->d_wtbl, pdim, M, ctx->ltbl, ctx->L, ctx->scale,
        tblspcg, n_angles, n_pairs, (long)H_stride);
  }

  // Phase 2: batched 2D FFT
  CUFFT_CHECK(cufftExecC2C(ctx->plan2d_multi, ctx->d_H_multi + 1,
                           ctx->d_H_multi + 1, CUFFT_FORWARD));

  // Phase 3: PSWF extraction to recon buffers
  {
    dim3 block3(16, 16);
    dim3 grid3(((int)M0 + 15) / 16, ((int)M0 + 15) / 16, n_pairs);
    pswf_extract_multi_kernel<<<grid3, block3, 0, ctx->stream_compute>>>(
        ctx->d_H_multi, ctx->d_recon1_multi, ctx->d_recon2_multi, ctx->d_winv,
        M, M0, M02, ctx->sinogram_x_dim, n_pairs, (long)H_stride,
        (long)recon_per_pair);
  }

  // Phase 4: extract compact reconstructions on GPU
  {
    dim3 block4(16, 16);
    dim3 grid4((recon_xdim + 15) / 16, (recon_xdim + 15) / 16);
    for (int i = 0; i < n_pairs; i++) {
      extract_recon_kernel<<<grid4, block4, 0, ctx->stream_compute>>>(
          ctx->d_compact_recon1_multi + i * compact_recon_per_slice,
          ctx->d_recon1_multi + i * recon_per_pair, recon_xdim, sinogram_x_dim,
          (int)M0, shift_round, auto_centering);
      extract_recon_kernel<<<grid4, block4, 0, ctx->stream_compute>>>(
          ctx->d_compact_recon2_multi + i * compact_recon_per_slice,
          ctx->d_recon2_multi + i * recon_per_pair, recon_xdim, sinogram_x_dim,
          (int)M0, shift_round, auto_centering);
    }
  }

  CUDA_CHECK(cudaEventRecord(ctx->event_compute_done, ctx->stream_compute));
  double t_compute = gpu_timer_sec() - tc;

  // ── D2H: download compact reconstructions (much smaller) ──
  double td = gpu_timer_sec();
  CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_d2h, ctx->event_compute_done, 0));
  for (int i = 0; i < n_pairs; i++) {
    CUDA_CHECK(cudaMemcpyAsync(
        compact_recon1s[i],
        ctx->d_compact_recon1_multi + i * compact_recon_per_slice,
        compact_recon_bytes, cudaMemcpyDeviceToHost, ctx->stream_d2h));
    CUDA_CHECK(cudaMemcpyAsync(
        compact_recon2s[i],
        ctx->d_compact_recon2_multi + i * compact_recon_per_slice,
        compact_recon_bytes, cudaMemcpyDeviceToHost, ctx->stream_d2h));
  }
  CUDA_CHECK(cudaStreamSynchronize(ctx->stream_d2h));
  double t_d2h = gpu_timer_sec() - td;

  double t_total = gpu_timer_sec() - t0;
  fprintf(stderr,
          "TOMO GPU raw-batch(%d pairs): "
          "H2D=%.3fs preproc=%.3fs compute=%.3fs D2H=%.3fs total=%.3fs\n",
          n_pairs, t_h2d, t_preproc, t_compute - t_preproc, t_d2h, t_total);

  return 0;
}

#endif /* ENABLE_CUDA */
