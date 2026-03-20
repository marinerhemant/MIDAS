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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include the CPU header for constants and struct definitions
extern "C" {
#include "tomo_heads.h"
}

// ─────────────────────────────────────────────────────────────
// Error-checking macros
// ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        return -1;                                                     \
    }                                                                  \
} while(0)

#define CUDA_CHECK_VOID(call) do {                                     \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
    }                                                                  \
} while(0)

#define CUFFT_CHECK(call) do {                                         \
    cufftResult res = (call);                                          \
    if (res != CUFFT_SUCCESS) {                                        \
        fprintf(stderr, "cuFFT error at %s:%d: %d\n",                  \
                __FILE__, __LINE__, (int)res);                         \
        return -1;                                                     \
    }                                                                  \
} while(0)

// ─────────────────────────────────────────────────────────────
// GPU context
// ─────────────────────────────────────────────────────────────

struct TomoGPUContext {
    int deviceId;

    // Problem dimensions
    unsigned long sinogram_x_dim;  // = adjusted_xdim * 2 (padded, complex-doubled)
    int theta_list_size;
    int filter_type;
    int useFftwBridge;

    // Pre-computed tables on device (uploaded from CPU's initGridRec)
    float *d_SINE;          // [theta_list_size]
    float *d_COSE;          // [theta_list_size]
    float *d_wtbl;          // [ltbl+1]
    float *d_winv;          // [M0]
    float *d_filphase;      // [(pdim/2+1)*2] — complex as float pairs
    int tables_uploaded;

    // Gridrec parameters
    long pdim, M, M0, M02, ltbl;
    float L, scale, sampl;
    int flag;

    // Device buffers
    float *d_sino1;              // [theta_list_size * sinogram_x_dim]
    float *d_sino2;
    cufftComplex *d_cproj;       // [pdim] — 1D FFT work buffer
    cufftComplex *d_H;           // [(M+1)*(M+1)] — Fourier grid
    float *d_recon1;             // [M0 * sinogram_x_dim] (stride = sinogram_x_dim)
    float *d_recon2;

    // cuFFT plans
    cufftHandle plan1d;
    cufftHandle plan2d;
    int cufft_plans_created;

    // FFTW-bridge pinned host buffer
    float *h_fft_bridge;
    size_t h_fft_bridge_size;

    // 3-stream pipeline
    cudaStream_t stream_h2d;
    cudaStream_t stream_compute;
    cudaStream_t stream_d2h;
    cudaEvent_t  event_h2d_done;
    cudaEvent_t  event_compute_done;
    cudaEvent_t  event_d2h_done;
};


// ═════════════════════════════════════════════════════════════
//  CUDA KERNELS
// ═════════════════════════════════════════════════════════════

// ── Load one sinogram row into cproj as complex ──
// CPU phase1 does:
//   cproj[j].r = G1[n][j-1],  cproj[j].i = G2[n][j-1]   for j=1..sinogram_x_dim
//   cproj[j] = 0                                          for j=sinogram_x_dim+1..pdim-1
// Then four1((float*)cproj + 1, pdim, ...) which does FFT on cproj[1..pdim].
//
// For GPU: we use 0-indexed cproj[0..pdim-1] and pass it directly to cuFFT.
// So we load: cproj[j].x = G1[n][j] for j=0..sinogram_x_dim-1, cproj[j]=0 otherwise.
//
// G1[n] = sinogram1[n * sinogram_x_dim]  (flat array, stride = sinogram_x_dim)
__global__ void load_sinogram_row_kernel(
    cufftComplex *cproj,            // output [pdim]
    const float *sino1,             // [theta_list_size * sinogram_x_dim]
    const float *sino2,
    int angle_idx,
    unsigned long sinogram_x_dim,
    long pdim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= pdim) return;

    if (j < (int)sinogram_x_dim) {
        long row_off = (long)angle_idx * sinogram_x_dim;
        cproj[j].x = sino1[row_off + j];
        cproj[j].y = sino2[row_off + j];
    } else {
        cproj[j].x = 0.0f;
        cproj[j].y = 0.0f;
    }
}

// ── Apply filter × phase to FFT output + scatter to H grid ──
// One thread per frequency bin j ∈ [1, pdim/2).
//
// CPU four1 interface:
//   four1((float *)param->cproj + 1, param->pdim, 1, param)
//   This copies from data+1 = (float*)cproj+2, i.e. starting at cproj[1],
//   forward pdim complex values → FFT → copy back.
//   After FFT: CPU reads cproj[j+1] for positive freq, cproj[(pdim-j)+1] for mirror.
//
// Our GPU cuFFT operates on d_cproj[0..pdim-1] (0-indexed).
// cuFFT inverse of 0-indexed data ≡ FFTW backward of 1-indexed data shifted.
// Net mapping: CPU cproj[j+1] → GPU d_cproj[j], CPU cproj[(pdim-j)+1] → GPU d_cproj[pdim-j].
__global__ void filter_and_scatter_kernel(
    cufftComplex *H,                // [(M+1)*(M+1)]
    const cufftComplex *cproj,      // [pdim], post-FFT
    const float *filphase,          // [(pdim/2+1)*2], interleaved r,i
    const float *COSE,
    const float *SINE,
    const float *wtbl,              // [ltbl+1]
    int angle_idx,
    long pdim, long M, long ltbl,
    float L, float scale, float tblspcg)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    long pdim2 = pdim >> 1;
    if (j < 1 || j >= pdim2) return;

    float L2 = L / 2.0f;
    long M2 = M >> 1;

    // Read filter × phase for this frequency bin
    float fr = filphase[j * 2];
    float fi = filphase[j * 2 + 1];

    // Apply filter to positive frequency: Cdata1 = filphase[j] * cproj[j]
    float cr = cproj[j].x;
    float ci = cproj[j].y;
    float d1r = fr * cr - fi * ci;
    float d1i = fr * ci + fi * cr;

    // Mirror frequency: Cdata2 = conj(filphase[j]) * cproj[pdim - j]
    float mr = cproj[pdim - j].x;
    float mi = cproj[pdim - j].y;
    float d2r =  fr * mr + fi * mi;   // conj: fi → -fi
    float d2i = -fi * mr + fr * mi;   // = fr*mi - fi*mr

    // Compute grid position
    float cosE = COSE[angle_idx];
    float sinE = SINE[angle_idx];
    float rtmp = scale * j;
    float U = rtmp * cosE + M2;
    float V = rtmp * sinE + M2;

    long iul = (long)(ceilf(U - L2));
    long iuh = (long)(floorf(U + L2));
    long ivl = (long)(ceilf(V - L2));
    long ivh = (long)(floorf(V + L2));

    if (iul < 1) iul = 1;
    if (iuh >= M) iuh = M - 1;
    if (ivl < 1) ivl = 1;
    if (ivh >= M) ivh = M - 1;

    // Pre-compute V-direction weights (L ≈ 3.8, so ≤ 5 entries)
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

    // Scatter to H grid
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

            // H[iu*M + iv + 1] — the +1 accounts for 1-indexed H array
            long idx1 = iu * M + iv + 1;
            atomicAdd(&H[idx1].x, convolv * d1r);
            atomicAdd(&H[idx1].y, convolv * d1i);

            // Mirror: H[(M-iu)*M + (M-iv) + 1]
            long idx2 = (M - iu) * M + (M - iv) + 1;
            atomicAdd(&H[idx2].x, convolv * d2r);
            atomicAdd(&H[idx2].y, convolv * d2i);
        }
    }
}

// ── Phase 3: PSWF correction + extraction ──
// CPU phase3 maps M×M Fourier grid to M0×M0 output using wrap-around
// indexing and PSWF correction.  Output S1/S2 have stride sinogram_x_dim
// (NOT M0!), matching setSinoAndReconBuffers.
//
// CPU loop structure:
//   j=0: iu starts at M-M02, goes to M-1   (M02+1 values, but first block is M02 values since ufin=M is exclusive... wait)
//   Actually: ustart=M-M02, ufin=M → iu goes M-M02..M-1 → that's M02 iterations
//   Then ustart=0, ufin=M02+1 → iu goes 0..M02 → that's M02+1 iterations
//   Total = M02 + M02 + 1 = 2*M02 + 1 = M0 ✓
//
// Mapping: output pixel (tj, tk) → grid (iu, iv):
//   tj ∈ [0, M02): iu = (M-M02) + tj
//   tj ∈ [M02, M0): iu = tj - M02
//   Same for tk → iv
__global__ void pswf_extract_kernel(
    const cufftComplex *H,
    float *S1,                      // [M0 * sinogram_x_dim]  (stride = sinogram_x_dim)
    float *S2,
    const float *winv,              // [M0]
    long M, long M0, long M02,
    unsigned long sinogram_x_dim)   // stride for output
{
    int tj = blockIdx.x * blockDim.x + threadIdx.x;
    int tk = blockIdx.y * blockDim.y + threadIdx.y;
    if (tj >= M0 || tk >= M0) return;

    float corrn = winv[tj] * winv[tk];

    // Wrap-around: first M02 pixels come from end of grid, rest from start
    long iu, iv;
    if (tj < M02) {
        iu = (M - M02) + tj;       // tj=0 → iu=M-M02, tj=M02-1 → iu=M-1
    } else {
        iu = tj - M02;             // tj=M02 → iu=0, tj=M0-1 → iu=M02
    }
    if (tk < M02) {
        iv = (M - M02) + tk;
    } else {
        iv = tk - M02;
    }

    // H[iu*M + iv + 1] — same indexing as phase1 scatter
    long h_idx = iu * M + iv + 1;
    // Output uses sinogram_x_dim as stride (matching CPU's S1[j][k] = recon1[j*sinogram_x_dim + k])
    long out_idx = (long)tj * sinogram_x_dim + tk;
    S1[out_idx] = corrn * H[h_idx].x;
    S2[out_idx] = corrn * H[h_idx].y;
}


// ═════════════════════════════════════════════════════════════
//  CONTEXT LIFECYCLE
// ═════════════════════════════════════════════════════════════

extern "C"
void tomo_gpu_print_info(int deviceId) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
        printf("TOMO GPU: could not query device %d\n", deviceId);
        return;
    }
    printf("TOMO GPU: %s, SM %d.%d, %.1f GB, %d SMs\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.multiProcessorCount);
}

extern "C"
TomoGPUContext *tomo_gpu_init(int deviceId,
                              unsigned long sinogram_x_dim,
                              int theta_list_size,
                              const float *theta_list,
                              int filter_type,
                              int useFftwBridge) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "TOMO GPU: failed to set device %d: %s\n",
                deviceId, cudaGetErrorString(err));
        return NULL;
    }

    tomo_gpu_print_info(deviceId);

    TomoGPUContext *ctx = (TomoGPUContext *)calloc(1, sizeof(TomoGPUContext));
    if (!ctx) return NULL;

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
    while (itmp) { ctx->pdim <<= 1; itmp >>= 1; }

    ctx->sampl = 1.0f;
    float D0 = 1.0f * sinogram_x_dim;
    float D1 = ctx->sampl * D0;
    ctx->M = 1;
    itmp = (long)(D1 / 1.0f - 1);
    while (itmp) { ctx->M <<= 1; itmp >>= 1; }

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
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_filphase,
                               (ctx->pdim / 2 + 1) * 2 * sizeof(float)));

    // ── Allocate device work buffers ──
    // Sinogram: flat array [theta_list_size * sinogram_x_dim]
    size_t sino_size = (size_t)theta_list_size * sinogram_x_dim * sizeof(float);
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_sino1, sino_size));
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_sino2, sino_size));

    // H grid: (M+1)*(M+1) complex values (1-indexed access pattern)
    size_t H_elems = (size_t)(ctx->M + 1) * (ctx->M + 1);
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_H, H_elems * sizeof(cufftComplex)));

    // 1D FFT work buffer
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_cproj, ctx->pdim * sizeof(cufftComplex)));

    // Reconstruction output: M0 rows × sinogram_x_dim cols (stride = sinogram_x_dim)
    // This matches CPU: S1[j] = &reconstruction1[j * sinogram_x_dim]
    size_t recon_size = (size_t)ctx->M0 * sinogram_x_dim * sizeof(float);
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_recon1, recon_size));
    CUDA_CHECK_VOID(cudaMalloc(&ctx->d_recon2, recon_size));

    // ── cuFFT plans ──
    if (!useFftwBridge) {
        cufftResult r1 = cufftPlan1d(&ctx->plan1d, (int)ctx->pdim, CUFFT_C2C, 1);
        cufftResult r2 = cufftPlan2d(&ctx->plan2d, (int)ctx->M, (int)ctx->M,
                                     CUFFT_C2C);
        if (r1 != CUFFT_SUCCESS || r2 != CUFFT_SUCCESS) {
            fprintf(stderr, "TOMO GPU: cuFFT plan creation failed\n");
            tomo_gpu_destroy(ctx);
            return NULL;
        }
        cufftSetStream(ctx->plan1d, ctx->stream_compute);
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

extern "C"
void tomo_gpu_destroy(TomoGPUContext *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->deviceId);

    if (ctx->d_SINE)     cudaFree(ctx->d_SINE);
    if (ctx->d_COSE)     cudaFree(ctx->d_COSE);
    if (ctx->d_wtbl)     cudaFree(ctx->d_wtbl);
    if (ctx->d_winv)     cudaFree(ctx->d_winv);
    if (ctx->d_filphase) cudaFree(ctx->d_filphase);
    if (ctx->d_sino1)    cudaFree(ctx->d_sino1);
    if (ctx->d_sino2)    cudaFree(ctx->d_sino2);
    if (ctx->d_H)        cudaFree(ctx->d_H);
    if (ctx->d_cproj)    cudaFree(ctx->d_cproj);
    if (ctx->d_recon1)   cudaFree(ctx->d_recon1);
    if (ctx->d_recon2)   cudaFree(ctx->d_recon2);

    if (ctx->cufft_plans_created) {
        cufftDestroy(ctx->plan1d);
        cufftDestroy(ctx->plan2d);
    }
    if (ctx->h_fft_bridge) cudaFreeHost(ctx->h_fft_bridge);

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
//  TABLE UPLOAD (from CPU initGridRec → GPU)
// ═════════════════════════════════════════════════════════════

extern "C"
int tomo_gpu_upload_tables(TomoGPUContext *ctx,
                           const float *SINE, const float *COSE,
                           const float *wtbl, const float *winv,
                           const float *filphase,
                           long ltbl, long M0, long pdim) {
    if (!ctx) return -1;
    cudaSetDevice(ctx->deviceId);

    CUDA_CHECK(cudaMemcpy(ctx->d_SINE, SINE,
                          ctx->theta_list_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_COSE, COSE,
                          ctx->theta_list_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_wtbl, wtbl,
                          (ltbl + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_winv, winv,
                          M0 * sizeof(float),
                          cudaMemcpyHostToDevice));
    // filphase is complex* on CPU = float pairs {r, i}
    // Size is (pdim/2 + 1) complex values = (pdim/2 + 1) * 2 floats
    CUDA_CHECK(cudaMemcpy(ctx->d_filphase, filphase,
                          (pdim / 2 + 1) * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    ctx->tables_uploaded = 1;
    printf("TOMO GPU: tables uploaded (ltbl=%ld, M0=%ld, pdim=%ld)\n",
           ltbl, M0, pdim);
    return 0;
}


// ═════════════════════════════════════════════════════════════
//  GPU RECONSTRUCTION
// ═════════════════════════════════════════════════════════════

extern "C"
int tomo_gpu_reconstruct(TomoGPUContext *ctx,
                         const float *sinogram1,
                         const float *sinogram2,
                         float *reconstruction1,
                         float *reconstruction2,
                         long M, long M0, long M02, long pdim) {
    if (!ctx) { fprintf(stderr, "TOMO GPU: ctx is NULL\n"); return -1; }
    if (!ctx->tables_uploaded) {
        fprintf(stderr, "TOMO GPU: tables not uploaded yet\n");
        return -1;
    }
    cudaSetDevice(ctx->deviceId);

    fprintf(stderr, "TOMO GPU reconstruct: sinogram1=%p sinogram2=%p recon1=%p recon2=%p\n",
            (void*)sinogram1, (void*)sinogram2,
            (void*)reconstruction1, (void*)reconstruction2);
    fprintf(stderr, "TOMO GPU reconstruct: M=%ld M0=%ld M02=%ld pdim=%ld "
            "sinogram_x_dim=%lu theta=%d\n",
            M, M0, M02, pdim, ctx->sinogram_x_dim, ctx->theta_list_size);

    if (!sinogram1 || !sinogram2 || !reconstruction1 || !reconstruction2) {
        fprintf(stderr, "TOMO GPU: NULL buffer pointer!\n");
        return -1;
    }

    long pdim2 = pdim >> 1;
    size_t sino_bytes = (size_t)ctx->theta_list_size *
                        ctx->sinogram_x_dim * sizeof(float);
    size_t H_elems = (size_t)(M + 1) * (M + 1);
    size_t recon_bytes = (size_t)M0 * ctx->sinogram_x_dim * sizeof(float);

    fprintf(stderr, "TOMO GPU: sino_bytes=%zu H_elems=%zu recon_bytes=%zu\n",
            sino_bytes, H_elems, recon_bytes);

    float tblspcg = 2.0f * ctx->ltbl / ctx->L;

    // ════════════════════════════════════════════════════════
    // Step 1: H2D — copy sinograms to GPU
    // ════════════════════════════════════════════════════════
    fprintf(stderr, "TOMO GPU: H2D sino1...\n"); fflush(stderr);
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino1, sinogram1, sino_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
    fprintf(stderr, "TOMO GPU: H2D sino2...\n"); fflush(stderr);
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_sino2, sinogram2, sino_bytes,
                               cudaMemcpyHostToDevice, ctx->stream_h2d));
    CUDA_CHECK(cudaEventRecord(ctx->event_h2d_done, ctx->stream_h2d));

    // Sync to ensure H2D completes before we proceed (debug)
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_h2d));
    fprintf(stderr, "TOMO GPU: H2D done\n"); fflush(stderr);

    // ════════════════════════════════════════════════════════
    // Step 2: Compute — phase1 + phase2 + phase3
    // ════════════════════════════════════════════════════════
    CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_compute,
                                   ctx->event_h2d_done, 0));

    // Clear H grid
    CUDA_CHECK(cudaMemsetAsync(ctx->d_H, 0, H_elems * sizeof(cufftComplex),
                               ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_recon1, 0, recon_bytes,
                               ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_recon2, 0, recon_bytes,
                               ctx->stream_compute));

    // ─── Phase 1: per-angle 1D FFT + filter + PSWF scatter ───
    int blockSize = 256;
    int gridSize_pdim = ((int)pdim + blockSize - 1) / blockSize;
    int gridSize_scatter = ((int)pdim2 + blockSize - 1) / blockSize;

    fprintf(stderr, "TOMO GPU: Phase 1 starting (%d angles)...\n",
            ctx->theta_list_size); fflush(stderr);

    for (int n = 0; n < ctx->theta_list_size; n++) {
        // 1a. Load sinogram row n into cproj
        load_sinogram_row_kernel<<<gridSize_pdim, blockSize, 0,
                                   ctx->stream_compute>>>(
            ctx->d_cproj, ctx->d_sino1, ctx->d_sino2,
            n, ctx->sinogram_x_dim, pdim);

        // Check for launch error
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "TOMO GPU: load_sinogram_row launch error at angle %d: %s\n",
                        n, cudaGetErrorString(kerr));
                return -1;
            }
        }

        // 1b. 1D backward FFT of cproj
        if (ctx->useFftwBridge) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
            CUDA_CHECK(cudaMemcpy(ctx->h_fft_bridge, ctx->d_cproj,
                                  pdim * sizeof(cufftComplex),
                                  cudaMemcpyDeviceToHost));
            {
                fftwf_complex *fft_buf = (fftwf_complex *)ctx->h_fft_bridge;
                fftwf_plan plan = fftwf_plan_dft_1d(
                    (int)pdim, fft_buf, fft_buf,
                    FFTW_BACKWARD, FFTW_ESTIMATE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            CUDA_CHECK(cudaMemcpy(ctx->d_cproj, ctx->h_fft_bridge,
                                  pdim * sizeof(cufftComplex),
                                  cudaMemcpyHostToDevice));
        } else {
            cufftResult fft_res = cufftExecC2C(ctx->plan1d, ctx->d_cproj,
                                               ctx->d_cproj, CUFFT_INVERSE);
            if (fft_res != CUFFT_SUCCESS) {
                fprintf(stderr, "TOMO GPU: cuFFT 1D error at angle %d: %d\n",
                        n, (int)fft_res);
                return -1;
            }
        }

        // 1c. Apply filter+phase and scatter to H grid
        filter_and_scatter_kernel<<<gridSize_scatter, blockSize, 0,
                                    ctx->stream_compute>>>(
            ctx->d_H,
            ctx->d_cproj,
            ctx->d_filphase,
            ctx->d_COSE, ctx->d_SINE,
            ctx->d_wtbl,
            n,
            pdim, M, ctx->ltbl,
            ctx->L, ctx->scale, tblspcg);

        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "TOMO GPU: filter_and_scatter launch error at angle %d: %s\n",
                        n, cudaGetErrorString(kerr));
                return -1;
            }
        }

        // Print progress every 100 angles
        if (n == 0 || (n + 1) % 500 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
            fprintf(stderr, "TOMO GPU: Phase 1 angle %d/%d done\n",
                    n + 1, ctx->theta_list_size);
            fflush(stderr);
        }
    }

    // Sync phase 1
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
    fprintf(stderr, "TOMO GPU: Phase 1 complete\n"); fflush(stderr);

    // ─── Phase 2: 2D forward FFT of H grid ───
    fprintf(stderr, "TOMO GPU: Phase 2 (2D FFT)...\n"); fflush(stderr);
    if (ctx->useFftwBridge) {
        size_t fft2d_bytes = (size_t)M * M * sizeof(cufftComplex);
        CUDA_CHECK(cudaMemcpy(ctx->h_fft_bridge, ctx->d_H + 1,
                              fft2d_bytes, cudaMemcpyDeviceToHost));
        {
            fftwf_complex *fft_buf = (fftwf_complex *)ctx->h_fft_bridge;
            fftwf_plan plan = fftwf_plan_dft_2d(
                (int)M, (int)M, fft_buf, fft_buf,
                FFTW_FORWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        CUDA_CHECK(cudaMemcpy(ctx->d_H + 1, ctx->h_fft_bridge,
                              fft2d_bytes, cudaMemcpyHostToDevice));
    } else {
        CUFFT_CHECK(cufftExecC2C(ctx->plan2d, ctx->d_H + 1,
                                 ctx->d_H + 1, CUFFT_FORWARD));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
    }
    fprintf(stderr, "TOMO GPU: Phase 2 complete\n"); fflush(stderr);

    // ─── Phase 3: PSWF extraction ───
    fprintf(stderr, "TOMO GPU: Phase 3 (extract M0=%ld, stride=%lu)...\n",
            M0, ctx->sinogram_x_dim); fflush(stderr);
    dim3 block3(16, 16);
    dim3 grid3(((int)M0 + 15) / 16, ((int)M0 + 15) / 16);
    pswf_extract_kernel<<<grid3, block3, 0, ctx->stream_compute>>>(
        ctx->d_H, ctx->d_recon1, ctx->d_recon2,
        ctx->d_winv,
        M, M0, M02, ctx->sinogram_x_dim);

    {
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "TOMO GPU: pswf_extract launch error: %s\n",
                    cudaGetErrorString(kerr));
            return -1;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
    fprintf(stderr, "TOMO GPU: Phase 3 complete\n"); fflush(stderr);

    CUDA_CHECK(cudaEventRecord(ctx->event_compute_done,
                               ctx->stream_compute));

    // ════════════════════════════════════════════════════════
    // Step 3: D2H — copy reconstructions back
    // ════════════════════════════════════════════════════════
    fprintf(stderr, "TOMO GPU: D2H recon (recon_bytes=%zu)...\n",
            recon_bytes); fflush(stderr);
    CUDA_CHECK(cudaStreamWaitEvent(ctx->stream_d2h,
                                   ctx->event_compute_done, 0));

    CUDA_CHECK(cudaMemcpyAsync(reconstruction1, ctx->d_recon1, recon_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream_d2h));
    CUDA_CHECK(cudaMemcpyAsync(reconstruction2, ctx->d_recon2, recon_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream_d2h));
    CUDA_CHECK(cudaEventRecord(ctx->event_d2h_done, ctx->stream_d2h));

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_d2h));
    fprintf(stderr, "TOMO GPU: D2H complete, reconstruct done\n"); fflush(stderr);

    return 0;
}


// ═════════════════════════════════════════════════════════════
//  GPU PREPROCESSING (stub)
// ═════════════════════════════════════════════════════════════

extern "C"
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
                        float ring_coeff) {
    if (!ctx) return -1;
    // Preprocessing remains on CPU for now — it's I/O-bound and not
    // the bottleneck.  The GPU acceleration targets the reconstruction
    // (phase1/phase2/phase3) which dominates compute time.
    fprintf(stderr, "TOMO GPU: preprocess not yet implemented, use CPU path\n");
    return -1;
}

#endif /* ENABLE_CUDA */
