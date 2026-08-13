//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef tomo_headsH
#define tomo_headsH

/* Pulls in FFTW when it was found at build time, and otherwise supplies the
 * fftwf_* types/stubs so this header's gridrecParams still compiles. Must
 * come before anything that names fftwf_complex or fftwf_plan. */
#ifdef MIDAS_TOMO_HAVE_FFTW
#include <fftw3.h>
#endif
#include "midas_fft.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef uint
typedef unsigned int uint; // for compatibility with C++ code
#endif

//--------------------------------------------------------------------------------------------------------------------------
// Parameters for gridrec
#define max(A, B) ((A) > (B) ? (A) : (B))
#define min(A, B) ((A) < (B) ? (A) : (B))
#define abs(A) ((A) > 0 ? (A) : -(A))
#define Cmult(A, B, C)                                                         \
  {                                                                            \
    (A).r = (B).r * (C).r - (B).i * (C).i;                                     \
    (A).i = (B).r * (C).i + (B).i * (C).r;                                     \
  }
#define TOLERANCE 0.1
#define LTBL_DEF 512
#define NO_PSWFS 5
#define FILTER_NONE 0
#define FILTER_SHEPP_LOGAN 1
#define FILTER_HANN 2
#define FILTER_HAMMING 3
#define FILTER_RAMP 4
#define MAX_N_THETAS 36000

/* Error codes recorded in gridrecParams.error (see also
 * midas_tomo_error_message()). Values are stable: Python reports them. */
#define MIDAS_TOMO_OK 0
#define MIDAS_TOMO_ERR_PSWF 2         /* prolate parameter C not in database */
#define MIDAS_TOMO_ERR_ANGLE_GEOM 3   /* illegal angle geometry indicator */
#define MIDAS_TOMO_ERR_LEGENDRE 4     /* legendre() argument outside [-1, 1] */

const char *midas_tomo_error_message(int code);

/* Run a reconstruction described by a parameter file.
 *
 * The library entry point: this is what the CLI binary and the Python ctypes
 * wrapper both call. Returns 0 on success, non-zero on failure, and never
 * calls exit(), so it is safe to invoke in-process.
 *
 *   paramFileName    parameter file (absolute paths inside it)
 *   requestedProcs   OpenMP thread count; clamped down to what RAM allows
 *   useGPU           use the CUDA path (ignored in a non-CUDA build)
 *   useFftwBridge    GPU only: compute FFTs with CPU FFTW for exact CPU parity
 *   useDeterministic plan with FFTW_ESTIMATE; reproducible, no wisdom file
 */
int midas_tomo_run(const char *paramFileName, int requestedProcs, int useGPU,
                   int useFftwBridge, int useDeterministic);

/* As midas_tomo_run(), but reads the sinograms from caller memory instead of
 * from dataFileName. areSinos mode only. `sinos` must hold
 * n_slices * theta_list_size * det_xdim floats, laid out slice-major, and must
 * stay alive for the duration of the call. The parameter file still supplies
 * everything else, and still names an output file. Pass sinos = NULL to get
 * exactly midas_tomo_run(). */
int midas_tomo_run_sinos(const char *paramFileName, int requestedProcs,
                         int useGPU, int useFftwBridge, int useDeterministic,
                         const float *sinos, size_t sinoBytes);

/* Fully in-memory: sinograms in, reconstruction out, no data files at all.
 * `out` must hold n_cleanup * n_shifts * n_slices * X * X floats and stay
 * alive for the call. Pass NULL for either buffer to keep that side on disk.
 * Not supported with useGPU or saveReconSeparate; both are rejected. */
int midas_tomo_run_arrays(const char *paramFileName, int requestedProcs,
                          int useGPU, int useFftwBridge, int useDeterministic,
                          const float *sinos, size_t sinoBytes,
                          float *out, size_t outBytes);

/* As midas_tomo_run(), choosing the FFT backend explicitly. */
int midas_tomo_run_engine(const char *paramFileName, int requestedProcs,
                          int useGPU, int useFftwBridge, int useDeterministic,
                          int fftEngine);

/* Everything at once: in-memory I/O and an explicit FFT backend. The other
 * entry points are thin wrappers over this. */
int midas_tomo_run_full(const char *paramFileName, int requestedProcs,
                        int useGPU, int useFftwBridge, int useDeterministic,
                        const float *sinos, size_t sinoBytes,
                        float *out, size_t outBytes, int fftEngine);

typedef struct PSWF_STRUCT {
  float C, lmbda;
  int nt;
  float coefs[15];
} pswf_struct;

typedef struct {
  float r, i;
} complex;

typedef struct {
  long pdim, M, M0, M02, ltbl, imgsiz;
  float sampl, scale, L, X0, Y0, *SINE, *COSE, *wtbl, *dwtbl, *work, *winv,
      **G1, **G2, **S1, **S2, *sinogram1, *sinogram2, *reconstruction1,
      *reconstruction2, *theta_list;
  complex *cproj, *filphase, *H;
  pswf_struct pswf_db[NO_PSWFS];
  int flag, theta_list_size, filter_type, n_prev, nx_prev, ny_prev, setPlan;
  unsigned long sinogram_x_dim;
  fftwf_complex *in_1d, *out_1d;
  fftwf_plan backward_plan_1d, forward_plan_2d;
  fftwf_complex *in_2d, *out_2d;
  char *wisdom_string;
  long sizeMatrices;
  /* Non-zero after a fatal condition inside the gridrec kernel. These used
   * to call exit(2), which is survivable in a standalone binary but kills the
   * host process when the engine is called in-process (ctypes). The kernel
   * now records the code here, returns a safe value, and the caller checks.
   * See MIDAS_TOMO_ERR_* below. */
  int error;
  /* Which FFT backend to use: MIDAS_FFT_FFTW or MIDAS_FFT_POCKET. Forced to
   * pocketfft in a build without FFTW. */
  int fft_engine;
  /* When set, plan with FFTW_ESTIMATE instead of FFTW_MEASURE: the plan
   * becomes a deterministic function of transform size and FFTW build rather
   * than of runtime timings, and no wisdom file is read or written. Costs
   * some speed; buys run-to-run and machine-to-machine reproducibility. */
  int deterministic;
} gridrecParams;

// Functions
inline float Cnvlvnt(float X, gridrecParams *param);
void phase1(gridrecParams *param);
void phase2(gridrecParams *param);
void phase3(gridrecParams *param);
void trig_su(int geom, int n_ang, gridrecParams *param);
void filphase_su(long pd, float center, gridrecParams *param);
void pswf_su(pswf_struct *pswf, long ltbl, long linv, float *wtbl, float *dwtbl,
             float *winv, gridrecParams *param);
float legendre(int n, float *coefs, float x, gridrecParams *param);
void get_pswf(float C, pswf_struct **P, gridrecParams *param);
void setSinoAndReconBuffers(int number, float *sinogram_address,
                            float *reconstruction_address,
                            gridrecParams *param);
float filterData(float x, gridrecParams *param);
float shlo(float x);
float hann(float x);
float hamm(float x);
float ramp(float x);
void reconstruct(gridrecParams *param);
void initGridRec(gridrecParams *param);
void getGridRecFourSizes(gridrecParams *param);
//--------------------------------------------------------------------------------------------------------------------------
// FFTW
void four1(float data[], unsigned long nn, int isign, gridrecParams *param);
void fourn(float data[], unsigned long nn[], int ndim, int isign,
           gridrecParams *param);
void initFFTMemoryStructures(gridrecParams *param);
void destroyFFTMemoryStructures(gridrecParams *param);

//--------------------------------------------------------------------------------------------------------------------------
// ConfigurationParameters
typedef struct {
  uint det_xdim, det_ydim, *slices_to_process;
  bool are_sinos, auto_centering, use_ring_removal;
  float start_angle, end_angle, angle_interval, *shift_values,
      ring_removal_coeff, *theta_list, start_shift, end_shift, shift_interval;
  char DataFileName[4096], ReconFileName[4096], SliceFileName[4096],
      thetaFileName[4096], HDF5FileName[4096], ImageDatasetName[4096],
      DarkDatasetName[4096], WhiteDatasetName[4096];
  int sinogram_xdim, sinogram_ydim, reconstruction_xdim, reconstruction_ydim,
      theta_list_size, n_shifts, n_slices, filter, debug;
  int sinogram_adjusted_xdim, reconstruction_size, sinogram_adjusted_size;
  char *wisdom_string;
  int saveReconSeparate;
  int powerIncrement;
  int doLogProj;
  int deterministic; /* --deterministic: FFTW_ESTIMATE, no wisdom. */
  int fft_engine;    /* --fft-engine: MIDAS_FFT_FFTW | MIDAS_FFT_POCKET */
  int use_hdf5;
  int doStripeRemoval;
  float stripeSnr;
  int stripeLaSize;
  int stripeSmSize;
  /* Cleanup parameter sweep.
   * When stripeConfigFile is non-empty AND doStripeRemoval is on, the engine
   * sweeps over n_cleanup_configs (snr, la, sm) triplets and writes a 5-D
   * output cube of shape (n_cleanup_configs, n_shifts, n_slices, X, Y).
   * Defaults: n_cleanup_configs = 1 (legacy single-config behavior).
   * A row of (0, 0, 0) in the config file is treated as the no-cleanup
   * baseline (stripe removal is skipped for that config). */
  int n_cleanup_configs;
  float *cleanup_snr_values;
  int *cleanup_la_values;
  int *cleanup_sm_values;
  char stripeConfigFile[4096];
  long sizeMatrices;
  /* Caller-owned sinogram buffer, used INSTEAD of opening DataFileName when
   * non-NULL (areSinos mode only). This is what lets Python hand the engine a
   * numpy array directly rather than staging it to disk and having the engine
   * read it back. Set by midas_tomo_run_sinos() BEFORE setGlobalOpts(), which
   * must not clobber it. */
  const float *sino_in_ptr;
  size_t sino_in_bytes;
  /* Caller-owned output buffer, used INSTEAD of writing reconFileName when
   * non-NULL. Same (cleanup, shift, slice, Y, X) layout the file would have,
   * so Python gets the cube without the file or its filename-encoded shape.
   * Requires saveReconSeparate == 0 and the CPU path. */
  float *recon_out_ptr;
  size_t recon_out_bytes;
} GLOBAL_CONFIG_OPTS;

typedef struct {
  int sinogram_adjusted_xdim, reconstruction_size, sinogram_adjusted_size;
  float *sino_calc_buffer, *recon_calc_buffer, *shifted_recon,
      *shifted_sinogram, *sinograms_boundary_padding,
      *reconstructions_boundary_padding, *mean_vect, *low_pass_sino_lines_data,
      *mean_sino_line_data, shift;
} LOCAL_CONFIG_OPTS;

typedef struct {
  unsigned short int *short_sinogram;
  float *norm_sino, *init_sinogram, *white_field_sino, *dark_field_sino_ave;
  long sizeMatrices;
} SINO_READ_OPTS;

//--------------------------------------------------------------------------------------------------------------------------
// Initiate Config Opts Structs
int setGlobalOpts(char inputFile[], GLOBAL_CONFIG_OPTS *recon_info_record);
void setSinoSize(LOCAL_CONFIG_OPTS *information,
                 const GLOBAL_CONFIG_OPTS *recon_info_record);
void freeSinoBuffers(LOCAL_CONFIG_OPTS *information);
void setReadStructSize(GLOBAL_CONFIG_OPTS *recon_info_record);
void memsets(LOCAL_CONFIG_OPTS *information,
             const GLOBAL_CONFIG_OPTS *recon_info_record);
void setGridRecPSWF(gridrecParams *param);

//--------------------------------------------------------------------------------------------------------------------------
// ReadData
int readSino(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
             SINO_READ_OPTS *readStruct);
int readRaw(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
            SINO_READ_OPTS *readStruct, int fd);
int readRawHDF5(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
                SINO_READ_OPTS *readStruct);

//--------------------------------------------------------------------------------------------------------------------------
// Corrections
void RingCorrectionSingle(float *data, float ring_coeff,
                          LOCAL_CONFIG_OPTS *information,
                          const GLOBAL_CONFIG_OPTS *recon_info_record);
void LogSinogram(float *data, int xdim, int ydim);
void LogProj(float *data, int xdim, int ydim);
void Normalize(SINO_READ_OPTS *readStruct,
               const GLOBAL_CONFIG_OPTS *recon_info_record);
void Pad(SINO_READ_OPTS *readStruct,
         const GLOBAL_CONFIG_OPTS *recon_info_record);

// Stripe removal (Vo et al. 2018)
void cleanup_sinogram_stripes(float *sinogram, int nrow, int ncol, float snr,
                              int la_size, int sm_size, int dim);
void cleanup_sinogram_filtering(float *sinogram, int nrow, int ncol,
                                float sigma, int sm_size, int dim);
void cleanup_sinogram_fitting(float *sinogram, int nrow, int ncol,
                              int order, float sigma_x, float sigma_y);

//--------------------------------------------------------------------------------------------------------------------------
// Processing code
void reconCentering(LOCAL_CONFIG_OPTS *information,
                    const GLOBAL_CONFIG_OPTS *recon_info_record, size_t offt,
                    int doLog);
void getRecons(LOCAL_CONFIG_OPTS *information,
               const GLOBAL_CONFIG_OPTS *recon_info_record,
               gridrecParams *param, size_t offsetRecons);
int writeRecon(int sliceNr, int slicePos, LOCAL_CONFIG_OPTS *information,
               const GLOBAL_CONFIG_OPTS *recon_info_record, int shiftNr,
               int cleanupNr, int fd);
int createPlanFile(GLOBAL_CONFIG_OPTS *recon_info_record);

// GPU-accelerated reconstruction (only available when built with CUDA)
#ifdef ENABLE_CUDA
#include "tomo_gpu.h"
#endif

#endif
