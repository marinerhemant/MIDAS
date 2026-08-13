//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "midas_version.h"
#include "tomo_heads.h"
#include <ctype.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

// GPU pthread compute — file-scope for C compatibility
#ifdef ENABLE_CUDA
typedef struct {
  void *ctx;  // TomoGPUContext*
  int count;
  const float **sino1, **sino2;
  float **recon1, **recon2;
  long M, M0, M02, pdim;
  int det_xdim, adjusted_xdim, sino_xdim, recon_xdim;
  int pad_front, doLog, auto_centering;
  float shift;
} GPURawWork;
static void *gpu_raw_compute_func(void *arg) {
  GPURawWork *w = (GPURawWork *)arg;
  tomo_gpu_reconstruct_batch_raw((TomoGPUContext *)w->ctx, w->count,
      w->sino1, w->sino2, w->recon1, w->recon2,
      w->M, w->M0, w->M02, w->pdim,
      w->det_xdim, w->adjusted_xdim, w->sino_xdim, w->recon_xdim,
      w->pad_front, w->shift, w->doLog, w->auto_centering);
  return NULL;
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
// #include <sys/sysinfo.h>

/*
 * The data can be one of two types:
 * 							sinogram already with
 * float data type, directly give to reconstruct code with some additional
 * centering etc. dark, whites (2) and then raw images. Using number of angles,
 * we know how many images are there. The scaling with white should be
 * proportional to the distance from a white and appropriate dark value.
 */

// TODO:
//	1. Check size of arrays needed and then allocate number of threads
// accordingly.
//	2. If nSlices*nShifts is not a multiple of nThreads, correctly calculate
// the number of
//	3. Safe malloc to check NULL ptrs.

void usage() {
  printf(
      "MIDAS-TOMO Code to do tomo recon using Gridrec. Based on tomompi "
      "implementation from Brian Tiemann, APS. Maintained by Hemant Sharma, "
      "APS (hsharma@anl.gov).\nUsage is: \n"
      "tomo ParamsFile.txt numberOfParallelJobs [--gpu] [--fftw-bridge]\n"
      "  --gpu          Use GPU-accelerated reconstruction (requires CUDA build)\n"
      "  --fftw-bridge  GPU mode: use CPU FFTW for FFTs (byte-identical to CPU; slower)\n"
      "  --fft-engine=<fftw|pocketfft>  Choose the FFT backend.\n"
      "                 DEFAULT: pocketfft. It is BSD-licensed and vendored (so\n"
      "                 no external FFT library is needed), measurably faster at\n"
      "                 typical sizes, and reproducible run to run.\n"
      "                 Use fftw to reproduce historical runs bit-for-bit: the\n"
      "                 two agree to ~3e-7 relative but not to the bit, and the\n"
      "                 FFTW planner is itself not reproducible between runs.\n"
      "  --deterministic  Plan FFTs with FFTW_ESTIMATE instead of FFTW_MEASURE.\n"
      "                 Makes repeated runs of THIS binary reproducible and writes\n"
      "                 no fftwf_wisdom_*.txt, at some speed cost. The default\n"
      "                 FFTW_MEASURE planner chooses by timing, so two identical\n"
      "                 default runs can differ in the low-order bits.\n"
      "                 SCOPE: measured reproducibility is within one build and\n"
      "                 one process context. Output still differs at the ~1e-7\n"
      "                 relative level between the binary and the shared library,\n"
      "                 and across FFTW builds -- a different plan means a\n"
      "                 different order of floating-point operations.\n"
      "Params file must have the following parameters:\n"
      "Input file is a text file name with a data link: sino data is a "
      "!!!single!!! binary file with darks, whites and tomo data in that "
      "order.\n"
      "* The rest of the file consists of the parameters required.\n"
      "* Parameters to be supplied:\n"
      "	* saveReconSeparate (optional): [int] 0 if want to save in recon in "
      "single file, 1 if want to save in individual files.\n"
      "	* dataFileName: [char*] name of the file with the raw data or sino "
      "data\n"
      "	* reconFileName: [char*] Name of the file for saving the "
      "reconstruction\n"
      "	* areSinos: If the input is a sinogram instead of raw (cleaned) images "
      "[0 or 1]\n"
      "	* The data can be one of two types: \n"
      "	* 							sinogram "
      "already with float data type, directly give to reconstruct code with "
      "some additional centering etc. \n"
      "	* 							dark[float], "
      "whites (2,floats) and then raw images[shorts]. Using number of angles, "
      "we know how many images are there.\n"
      "	*							The scaling "
      "with white should be proportional to the distance from a white and "
      "appropriate dark value.\n"
      "	*							Alternatively, "
      "HDF5 input can be used by providing:\n"
      "	*							HDF5FileName: "
      "[char*] Path to HDF5 file\n"
      "	*							"
      "ImageDatasetName: [char*] Path to image dataset inside HDF5\n"
      "	*							"
      "DarkDatasetName: [char*] Path to dark dataset inside HDF5\n"
      "	*							"
      "WhiteDatasetName: [char*] Path to white dataset inside HDF5\n"
      "	*								- "
      "Darks: Averaged if > 1 frame\n"
      "	*								- "
      "Whites: If > 2 frames, first N/2 "
      "averaged for White1, rest for White2\n"
      "	* Stripe Removal (Vo et al. 2018):\n"
      "	*								"
      "doStripeRemoval: [int] 0 = off, 1 = on (default 0)\n"
      "	*								"
      "stripeSnr: [float] SNR for stripe detection (default 3.0)\n"
      "	*								"
      "stripeLaSize: [int] Median filter window for large stripes (default 61, "
      "odd)\n"
      "	*								"
      "stripeSmSize: [int] Median filter window for small stripes (default 21, "
      "odd)\n"
      "	* detXdim - [uint]\n"
      "	* detYdim - [uint]\n"
      "	* Thetas can either be given as a range:\n"
      "	* 	thetaRange: startAngle endAngle angleInterval - [floats]\n"
      "	* or a File:\n"
      "	* 	thetaFileName [char*] with each line having an angle value "
      "[float].\n"
      "	* filter - [int] set to * 0: default\n"
      "							* 1: Shepp / Logan\n"
      "							* 2: Hann\n"
      "							* 3: Hamming\n"
      "							* 4: Ramp\n"
      "	* shiftValues: start_shift end_shift shift_interval [floats] In case "
      "of 1 shift, give start_shift=end_shift, shift_interval doesn't matter.\n"
      "	*					ENSURE TO GIVE A RANGE WITH "
      "EVEN NUMBER OF SHIFTS\n"
      "	* ringRemovalCoefficient - If given, will do ringRemoval, otherwise "
      "comment or remove line [float] default 1.0\n"
      "   * doLog - If 1, will take Log of intensities to calculate "
      "transmission, otherwise will use intensities directly. [int] default "
      "1.\n"
      "	* slicesToProcess - -1 for all or FileName. ENSURE TO GIVE EVEN NUMBER "
      "OF SLICES\n"
      "	* ExtraPad - 0 if half padding, 1 if one-half padding\n"
      "	* AutoCentering - 0 if don't want reconstruction shifted in one "
      "direction (rotation axis in center of recon)\n"
      "	* 				- 1 if want shift (rotation axis is "
      "offset) [default]\n"
      "Output file: float with reconstruction_xdim*reconstruction_xdim size\n"
      "OutputFileName: "
      "{recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_"
      "reconstruction_xdim_float_4byte.bin\n"
      "The code will generate two text files: fftwf_wisdom_{1,2}d.txt.\n"
      "These files are ways to speed up the fft calculation.\n"
      "First run on a dataset generates these files which can be used to speed "
      "up subsequent runs.\n");
}

/* Library entry point.
 *
 * This is main()'s former body verbatim: the only changes are where the
 * inputs come from (arguments instead of argv) and that argv parsing moved
 * to the thin main() at the end of this file. Nothing in the numerics path
 * was touched, which is what lets the bitwise parity gate against a pre-fork
 * build keep passing across this refactor.
 *
 * Returns 0 on success, non-zero on failure. It does NOT call exit() -- see
 * divergence #3 in FORK.txt -- so it is safe to call in-process.
 */
int midas_tomo_run(const char *paramFileName, int requestedProcs, int useGPU,
                   int useFftwBridge, int useDeterministic) {
  return midas_tomo_run_sinos(paramFileName, requestedProcs, useGPU,
                              useFftwBridge, useDeterministic, NULL, 0);
}

int midas_tomo_run_engine(const char *paramFileName, int requestedProcs,
                          int useGPU, int useFftwBridge, int useDeterministic,
                          int fftEngine) {
  return midas_tomo_run_full(paramFileName, requestedProcs, useGPU,
                             useFftwBridge, useDeterministic, NULL, 0, NULL, 0,
                             fftEngine);
}

int midas_tomo_run_sinos(const char *paramFileName, int requestedProcs,
                         int useGPU, int useFftwBridge, int useDeterministic,
                         const float *sinos, size_t sinoBytes) {
  return midas_tomo_run_arrays(paramFileName, requestedProcs, useGPU,
                               useFftwBridge, useDeterministic, sinos,
                               sinoBytes, NULL, 0);
}

int midas_tomo_run_arrays(const char *paramFileName, int requestedProcs,
                          int useGPU, int useFftwBridge, int useDeterministic,
                          const float *sinos, size_t sinoBytes,
                          float *out, size_t outBytes) {
  return midas_tomo_run_full(paramFileName, requestedProcs, useGPU,
                             useFftwBridge, useDeterministic, sinos, sinoBytes,
                             out, outBytes,
                             MIDAS_FFT_POCKET);
}

int midas_tomo_run_full(const char *paramFileName, int requestedProcs,
                        int useGPU, int useFftwBridge, int useDeterministic,
                        const float *sinos, size_t sinoBytes,
                        float *out, size_t outBytes, int fftEngine) {
  if (out != NULL && useGPU) {
    fprintf(stderr, "ERROR: in-memory output is not supported on the GPU path "
                    "yet; the CUDA writer goes through its own mmap'd file.\n");
    return 1;
  }
#ifdef ENABLE_CUDA
  if (useGPU)
    printf("GPU reconstruction enabled%s.\n",
           useFftwBridge ? " (FFTW-bridge mode)" : "");
#else
  if (useGPU) {
    printf("Warning: --gpu requested but this binary was built without CUDA. "
           "Using CPU path.\n");
    useGPU = 0;
  }
#endif
  GLOBAL_CONFIG_OPTS recon_info_record;
  /* Zero first: setGlobalOpts() initialises the fields it knows about, but a
   * whole-struct zero is cheap insurance against the next field that gets
   * added and forgotten -- which has already happened twice here. */
  memset(&recon_info_record, 0, sizeof(recon_info_record));
  recon_info_record.sizeMatrices = 0;
  /* Set BEFORE setGlobalOpts(), which consults it: with a caller buffer the
   * parameter file need not name (or even have) a dataFileName. */
  recon_info_record.sino_in_ptr = sinos;
  recon_info_record.sino_in_bytes = sinoBytes;
  recon_info_record.recon_out_ptr = out;
  recon_info_record.recon_out_bytes = outBytes;
  if (fftEngine == MIDAS_FFT_FFTW && !midas_fftw_available()) {
    fprintf(stderr, "ERROR: the FFTW engine was requested but this build has "
                    "no FFTW. Rebuild with FFTW, or use pocketfft.\n");
    return 1;
  }
  if (fftEngine != MIDAS_FFT_FFTW && fftEngine != MIDAS_FFT_POCKET) {
    fprintf(stderr, "ERROR: unknown FFT engine id %d.\n", fftEngine);
    return 1;
  }
  recon_info_record.fft_engine = fftEngine;
  printf("FFT engine: %s\n", midas_fft_engine_name(fftEngine));
  char *fileName;
  fileName = (char *)paramFileName;
  int RC;
  RC = setGlobalOpts(fileName, &recon_info_record);
  recon_info_record.deterministic = useDeterministic;
  if (out != NULL && recon_info_record.saveReconSeparate == 1) {
    fprintf(stderr, "ERROR: in-memory output requires saveReconSeparate 0 "
                    "(saveReconSeparate 1 writes one file per slice).\n");
    return 1;
  }
  setReadStructSize(&recon_info_record);
  gridrecParams pm;
  pm.error = MIDAS_TOMO_OK;
  pm.fft_engine = fftEngine;
  pm.deterministic = useDeterministic;
  pm.sinogram_x_dim = recon_info_record.sinogram_adjusted_xdim * 2;
  getGridRecFourSizes(&pm);
  int fftw1d_size = (int)pm.pdim;
  int fftw2d_size = (int)pm.M;
  char plan2DFN[4096];
  sprintf(plan2DFN, "fftwf_wisdom_2d_%d.txt", (int)fftw2d_size);
  char plan1DFN[4096];
  sprintf(plan1DFN, "fftwf_wisdom_1d_%d.txt", (int)fftw1d_size);
  if (RC != 0) {
    printf("Parameter file could not be read. Exiting.\n");
    return 1;
  }
  // Get FFT Plan
  //
  // createPlanFile() must run in EVERY mode, deterministic included: besides
  // building the wisdom it performs a trial reconstruction to measure
  // recon_info_record.sizeMatrices, which the job-splitting arithmetic below
  // divides by. Skipping it leaves sizeMatrices at 0 and the process dies with
  // SIGFPE on that integer division. What deterministic mode changes is only
  // the planner flag and the wisdom file I/O, both handled inside gridrec.
  if (recon_info_record.fft_engine == MIDAS_FFT_POCKET) {
    /* pocketfft has no timing-based planner and no wisdom cache, so it is
     * already reproducible run to run -- measured bitwise identical across
     * fresh runs, which the FFTW default is not. --deterministic is therefore
     * a no-op here rather than an error. */
    if (useDeterministic)
      printf("Note: --deterministic is redundant with the pocketfft engine, "
             "which has no timing-based planner.\n");
    if (createPlanFile(&recon_info_record) != 0) {
      fprintf(stderr, "ERROR: could not read the input data.\n");
      return 1;
    }
  } else if (useDeterministic) {
    printf("Deterministic mode: planning with FFTW_ESTIMATE, no wisdom file.\n");
    if (createPlanFile(&recon_info_record) != 0) {
      fprintf(stderr, "ERROR: could not read the input data.\n");
      return 1;
    }
  } else if (access(plan2DFN, F_OK) == -1) {
    printf("FFT plan file did not exist, creating one %s.\n",
           plan2DFN); // Check if sizes are okay.
    if (createPlanFile(&recon_info_record) != 0) {
      fprintf(stderr, "ERROR: could not read the input data.\n");
      return 1;
    }
  } else if (access(plan1DFN, F_OK) == -1) {
    printf("FFT plan file did not exist, creating one %s.\n", plan2DFN);
    if (createPlanFile(&recon_info_record) != 0) {
      fprintf(stderr, "ERROR: could not read the input data.\n");
      return 1;
    }
  } else {
    printf("Reading wisdom file %s.\n", plan2DFN);
    if (createPlanFile(&recon_info_record) != 0) {
      fprintf(stderr, "ERROR: could not read the input data.\n");
      return 1;
    }
  }
  // Detect available RAM (platform-specific)
  long long int avRAM;
#ifdef __APPLE__
  {
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
      avRAM = (long long int)memsize; // already in bytes
    } else {
      avRAM = 8LL * 1024 * 1024 * 1024; // fallback: assume 8 GB
      printf("Warning: could not determine RAM, assuming 8 GB.\n");
    }
  }
#else
  {
    FILE *memf = fopen("/proc/meminfo", "r");
    if (memf == NULL) {
      avRAM = 8LL * 1024 * 1024 * 1024; // fallback: assume 8 GB
      printf("Warning: could not open /proc/meminfo, assuming 8 GB RAM.\n");
    } else {
      char aline[4096], dummy[4096];
      avRAM = 8LL * 1024 * 1024 * 1024; // default fallback
      while (fgets(aline, 4096, memf) != NULL) {
        if (strncmp(aline, "MemAvailable:", strlen("MemAvailable:")) == 0) {
          sscanf(aline, "%s %lld", dummy, &avRAM);
          avRAM *= 1024; // MemAvailable is in kB, convert to bytes
          break;
        }
      }
      fclose(memf);
    }
  }
#endif
  recon_info_record.sizeMatrices *= 2;
  long long int maxNProcs =
      (long long int)avRAM / (long long int)recon_info_record.sizeMatrices;
  int numProcs =
      (requestedProcs < maxNProcs - 2) ? requestedProcs : maxNProcs - 2;
  printf(
      "Memory needed per process: %lld, Total available RAM: %lld, MaxNProcs: "
      "%lld.\nWe can run up to %lld processes.\nWe will use %lld MB RAM.\n",
      (long long int)recon_info_record.sizeMatrices, avRAM, maxNProcs,
      maxNProcs - 2,
      (long long int)numProcs * recon_info_record.sizeMatrices / (1000 * 1000));
  // Check if sizes are okay.
  if (recon_info_record.n_shifts > 1 && recon_info_record.n_shifts % 2 != 0) {
    printf("Number of shifts must be even. Exiting\n");
    return 1;
  }
  /* Cleanup sweep requires the multi-shift inner loop (it pairs shifts in
   * the dual-slice gridrec call). Pass at least 2 shifts so it can pair. */
  if (recon_info_record.n_cleanup_configs > 1 &&
      recon_info_record.n_shifts < 2) {
    fprintf(stderr,
            "Error: stripeConfigFile sweep requires n_shifts >= 2.\n"
            "Use shiftValues like 'X X+0.1 0.1' to produce 2 shifts and\n"
            "ignore the second result, or call run_tomo_cleanup_sweep() in\n"
            "Python which handles this automatically.\n");
    return 1;
  }
  {
    printf("Total number of shifts: %d, total number of  slices: %d.\n",
           recon_info_record.n_shifts, recon_info_record.n_slices);
  }
  if (recon_info_record.n_shifts == 1 && recon_info_record.n_slices % 2 != 0) {
    printf("Number of slices must be even. Exiting\n");
    return 1;
  }
  int rc = fftwf_import_wisdom_from_filename("fftwf_wisdom_1d.txt");

  // ── GPU context initialisation ──
#ifdef ENABLE_CUDA
  TomoGPUContext *gpu_ctx = NULL;
  if (useGPU) {
    gridrecParams pm_gpu;
    pm_gpu.sinogram_x_dim = recon_info_record.sinogram_adjusted_xdim * 2;
    pm_gpu.theta_list = recon_info_record.theta_list;
    pm_gpu.filter_type = recon_info_record.filter;
    pm_gpu.theta_list_size = recon_info_record.theta_list_size;
    pm_gpu.setPlan = 0;
    pm_gpu.wisdom_string = NULL;
    getGridRecFourSizes(&pm_gpu);
    gpu_ctx = tomo_gpu_init(
        0,  /* deviceId */
        pm_gpu.sinogram_x_dim,
        recon_info_record.theta_list_size,
        recon_info_record.theta_list,
        recon_info_record.filter,
        useFftwBridge);
    if (!gpu_ctx) {
      printf("Failed to initialise GPU context. Falling back to CPU.\n");
      useGPU = 0;
    } else {
      // Compute PSWF tables on CPU and upload to GPU
      setGridRecPSWF(&pm_gpu);
      initGridRec(&pm_gpu);
      tomo_gpu_upload_tables(gpu_ctx,
          pm_gpu.SINE, pm_gpu.COSE,
          pm_gpu.wtbl, pm_gpu.winv,
          (const float *)pm_gpu.filphase,
          pm_gpu.ltbl, pm_gpu.M0, pm_gpu.pdim);
      // Free the temporary CPU tables
      if (pm_gpu.cproj) free(pm_gpu.cproj);
      if (pm_gpu.filphase) free(pm_gpu.filphase);
      if (pm_gpu.wtbl) free(pm_gpu.wtbl);
      if (pm_gpu.winv) free(pm_gpu.winv);
      if (pm_gpu.work) free(pm_gpu.work);
      if (pm_gpu.H) free(pm_gpu.H);
      if (pm_gpu.SINE) free(pm_gpu.SINE);
      if (pm_gpu.COSE) free(pm_gpu.COSE);
    }
  }
#endif

  double start_time = omp_get_wtime();
  /* If a cleanup-parameter sweep is active (n_cleanup_configs > 1) we always
   * route through the parallel pre-read multi-shift path below, even if there
   * is only one shift — that path is where the outer cleanup loop lives. */
  if (recon_info_record.n_shifts == 1 &&
      recon_info_record.n_cleanup_configs == 1) {
    printf("Starting processing of all slices with %d threads.\n", numProcs);
    // mmap sinogram input for CPU path (same optimization as GPU)
    float *cpu_sino_mmap = NULL;
    size_t cpu_sino_mmap_len = 0;
    int cpu_sino_is_caller_buffer = 0;
    /* Set by any worker whose slice failed to read. A benign race: every
     * writer stores the same value, and it is only read after the parallel
     * region joins. Previously a failed slice was `continue`d past, leaving
     * uninitialised output for it and still reporting success. */
    int badReadSingle = 0;
    /* Same treatment for write failures: a full output buffer or a failed
     * pwrite used to be `continue`d past, so the run reported success with
     * part of the cube missing. */
    int badWriteSingle = 0;
    if (recon_info_record.are_sinos && recon_info_record.sino_in_ptr != NULL) {
      /* Already in memory -- skip the mmap entirely. Must not be munmap'd. */
      cpu_sino_mmap = (float *)recon_info_record.sino_in_ptr;
      cpu_sino_mmap_len = recon_info_record.sino_in_bytes;
      cpu_sino_is_caller_buffer = 1;
    } else if (recon_info_record.are_sinos) {
      int mfd = open(recon_info_record.DataFileName, O_RDONLY);
      if (mfd >= 0) {
        struct stat st;
        fstat(mfd, &st);
        cpu_sino_mmap_len = st.st_size;
        cpu_sino_mmap = (float *)mmap(NULL, cpu_sino_mmap_len,
                                      PROT_READ, MAP_PRIVATE, mfd, 0);
        close(mfd);
        if (cpu_sino_mmap == MAP_FAILED) cpu_sino_mmap = NULL;
        else madvise(cpu_sino_mmap, cpu_sino_mmap_len, MADV_SEQUENTIAL);
      }
    }
#pragma omp parallel num_threads(numProcs)
    {
      int procNr = omp_get_thread_num();
      int nrSlicesThread = (int)ceil((double)recon_info_record.n_slices /
                                     (2.0 * (double)numProcs));
      int startSliceNr = procNr * nrSlicesThread * 2;
      int endSliceNr = startSliceNr + nrSlicesThread * 2;
      if (endSliceNr > recon_info_record.n_slices)
        endSliceNr = recon_info_record.n_slices;
      // Allocate all the structs and arrays now
      SINO_READ_OPTS readStruct;
      readStruct.norm_sino = (float *)malloc(
          sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
          recon_info_record.theta_list_size);
      LOCAL_CONFIG_OPTS information;
      information.shift = recon_info_record.shift_values[0];
      setSinoSize(&information, &recon_info_record);
      gridrecParams param;
      param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
      param.theta_list = recon_info_record.theta_list;
      param.filter_type = recon_info_record.filter;
      param.theta_list_size = recon_info_record.theta_list_size;
      param.wisdom_string = (char *)malloc(
          sizeof(char) * (strlen(recon_info_record.wisdom_string) + 1));
      param.setPlan = 0;
      param.error = MIDAS_TOMO_OK;
      param.fft_engine = recon_info_record.fft_engine;
      param.deterministic = recon_info_record.deterministic;
      strcpy(param.wisdom_string, recon_info_record.wisdom_string);
      size_t offt, offsetRecons;
      setGridRecPSWF(&param);
      initFFTMemoryStructures(&param);
      initGridRec(&param);
      int numSlice, sliceRowNr, oldSliceNr;
      int input_fd = -1;
      if (!recon_info_record.are_sinos && !recon_info_record.use_hdf5) {
        input_fd = open(recon_info_record.DataFileName, O_RDONLY);
      }
      int output_fd = -1;
      if (recon_info_record.saveReconSeparate == 0) {
        char outFileName[4096];
        sprintf(
            outFileName,
            "%s_NrShifts_%03d_NrSlices_%05d_XDim_%06d_YDim_%06d_float32.bin",
            recon_info_record.ReconFileName, recon_info_record.n_shifts,
            recon_info_record.n_slices, recon_info_record.reconstruction_xdim,
            recon_info_record.reconstruction_xdim);
        if (recon_info_record.recon_out_ptr == NULL)
        output_fd = open(outFileName, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
      }
      int totalPairs = (endSliceNr - startSliceNr) / 2;
#ifdef ENABLE_CUDA
      if (useGPU && gpu_ctx) {
        // ── GPU dispatch: pthread pipeline + dynamic batch sizing ──
        // Pipeline: GPU compute buf[cur] || CPU write buf[prev] + read buf[nxt]
        double gpu_t_read = 0, gpu_t_compute = 0, gpu_t_write = 0;
        struct timespec ts_tmp;
        #define TOMO_WTIME() (clock_gettime(CLOCK_MONOTONIC, &ts_tmp), \
                              ts_tmp.tv_sec + ts_tmp.tv_nsec * 1e-9)
        int gpu_rw_threads = omp_get_max_threads();
        if (!recon_info_record.are_sinos) gpu_rw_threads = 1;

        // ── Dynamic batch size from GPU free memory (75%) ──
        size_t sinogram_x_dim_gpu = (size_t)information.sinogram_adjusted_xdim * 2;
        int n_angles = recon_info_record.theta_list_size;
        size_t per_pair_gpu =
            2 * (size_t)n_angles * sinogram_x_dim_gpu * sizeof(float)   // sino1+sino2
          + (size_t)n_angles * param.pdim * 2 * sizeof(float)           // cproj (complex)
          + (size_t)(param.M + 1) * (param.M + 1) * 2 * sizeof(float)  // H (complex)
          + 2 * (size_t)param.M0 * sinogram_x_dim_gpu * sizeof(float); // recon1+recon2
        size_t gpu_free = tomo_gpu_get_free_memory();
        int gpu_batch_pairs = (int)((gpu_free * 3 / 4) / per_pair_gpu);
        // Cap at 50 pairs: larger batches waste pinned host memory (double-buffered)
        // which evicts page cache and makes file I/O much slower
        if (gpu_batch_pairs > 50) gpu_batch_pairs = 50;
        if (gpu_batch_pairs > totalPairs) gpu_batch_pairs = totalPairs;
        if (gpu_batch_pairs < 1) gpu_batch_pairs = 1;
        int nBatches = (totalPairs + gpu_batch_pairs - 1) / gpu_batch_pairs;
        fprintf(stderr, "TOMO GPU: dynamic batch=%d pairs (%zu MB GPU/pair, "
                "%zu MB free, %d batches)\n",
                gpu_batch_pairs, per_pair_gpu >> 20, gpu_free >> 20, nBatches);

        // ── mmap sinogram input for zero-copy reads (areSinos mode) ──
        float *sino_mmap = NULL;
        size_t sino_mmap_len = 0;
        size_t sino_slice_bytes = (size_t)recon_info_record.det_xdim
            * recon_info_record.theta_list_size * sizeof(float);
        if (recon_info_record.are_sinos) {
          int mfd = open(recon_info_record.DataFileName, O_RDONLY);
          if (mfd >= 0) {
            struct stat st;
            fstat(mfd, &st);
            sino_mmap_len = st.st_size;
            sino_mmap = (float *)mmap(NULL, sino_mmap_len,
                                      PROT_READ, MAP_PRIVATE, mfd, 0);
            close(mfd);
            if (sino_mmap == MAP_FAILED) sino_mmap = NULL;
            else {
              madvise(sino_mmap, sino_mmap_len, MADV_SEQUENTIAL);
              fprintf(stderr, "TOMO GPU: sinogram mmap'd (%zu MB)\n",
                      sino_mmap_len >> 20);
            }
          }
        }

        // ── Double-buffered batch state (raw sinograms + compact recons) ──
        size_t raw_sino_slice = (size_t)recon_info_record.det_xdim * n_angles;
        size_t raw_sino_bytes_per = raw_sino_slice * sizeof(float);
        size_t compact_recon_slice = (size_t)recon_info_record.reconstruction_xdim
            * recon_info_record.reconstruction_xdim;
        size_t compact_recon_bytes_per = compact_recon_slice * sizeof(float);
        int det_xdim_raw = recon_info_record.det_xdim;
        int recon_xdim_out = recon_info_record.reconstruction_xdim;
        int pad_size_raw = recon_info_record.sinogram_adjusted_xdim - recon_info_record.sinogram_xdim;
        int pad_front_raw = pad_size_raw / 2;
        typedef struct {
          const float **sino1;
          const float **sino2;
          float **recon1;
          float **recon2;
          int *oldSliceNr;
          int *sliceNr;
          float **raw_buf;      // pinned: 2 per pair (sino1, sino2)
          float **compact_buf;  // pinned: 2 per pair (recon1, recon2)
          int count;
        } BatchBuf;
        BatchBuf buf[2];
        for (int s = 0; s < 2; s++) {
          buf[s].sino1 = (const float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].sino2 = (const float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].recon1 = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].recon2 = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].oldSliceNr = (int *)malloc(gpu_batch_pairs * sizeof(int));
          buf[s].sliceNr = (int *)malloc(gpu_batch_pairs * sizeof(int));
          buf[s].raw_buf = (float **)malloc(gpu_batch_pairs * 2 * sizeof(float *));
          buf[s].compact_buf = (float **)malloc(gpu_batch_pairs * 2 * sizeof(float *));
          for (int b = 0; b < gpu_batch_pairs; b++) {
            tomo_gpu_pinned_alloc((void **)&buf[s].raw_buf[b*2], raw_sino_bytes_per);
            tomo_gpu_pinned_alloc((void **)&buf[s].raw_buf[b*2+1], raw_sino_bytes_per);
            tomo_gpu_pinned_alloc((void **)&buf[s].compact_buf[b*2], compact_recon_bytes_per);
            tomo_gpu_pinned_alloc((void **)&buf[s].compact_buf[b*2+1], compact_recon_bytes_per);
          }
          buf[s].count = 0;
        }
        size_t pinned_mb = (2 * gpu_batch_pairs * 2 * (raw_sino_bytes_per + compact_recon_bytes_per)) >> 20;
        fprintf(stderr, "TOMO GPU: raw batch buffers allocated (%zu MB pinned)\n", pinned_mb);

        // ── mmap output file for fast writes ──
        float *output_mmap = NULL;
        size_t output_mmap_len = 0;
        if (output_fd >= 0) {
          output_mmap_len = (size_t)recon_info_record.n_slices * compact_recon_bytes_per;
          ftruncate(output_fd, output_mmap_len);
          output_mmap = (float *)mmap(NULL, output_mmap_len, PROT_WRITE,
                                      MAP_SHARED, output_fd, 0);
          if (output_mmap == MAP_FAILED) {
            fprintf(stderr, "TOMO GPU: output mmap failed, falling back to pwrite\n");
            output_mmap = NULL;
          } else {
            fprintf(stderr, "TOMO GPU: output mmap'd (%zu MB)\n", output_mmap_len >> 20);
          }
        }

        // ── Helper: read raw sinograms into buf (OMP-parallel memcpy) ──
        #define GPU_READ_BATCH(sIdx, batchStart) do { \
          int _this = totalPairs - (batchStart); \
          if (_this > gpu_batch_pairs) _this = gpu_batch_pairs; \
          buf[sIdx].count = _this; \
          _Pragma("omp parallel for schedule(dynamic, 1) num_threads(gpu_rw_threads)") \
          for (int b = 0; b < _this; b++) { \
            int pi = (batchStart) + b; \
            int sr = startSliceNr + pi * 2; \
            int sn1 = recon_info_record.slices_to_process[sr]; \
            int sn2 = recon_info_record.slices_to_process[sr + 1]; \
            buf[sIdx].oldSliceNr[b] = sn1; \
            buf[sIdx].sliceNr[b] = sn2; \
            if (sino_mmap) { \
              memcpy(buf[sIdx].raw_buf[b*2], \
                     sino_mmap + (size_t)sn1 * det_xdim_raw * n_angles, raw_sino_bytes_per); \
              memcpy(buf[sIdx].raw_buf[b*2+1], \
                     sino_mmap + (size_t)sn2 * det_xdim_raw * n_angles, raw_sino_bytes_per); \
            } else if (recon_info_record.are_sinos) { \
              /* Sinogram input but no mmap: use pread to read raw data directly */ \
              pread(input_fd, buf[sIdx].raw_buf[b*2], raw_sino_bytes_per, \
                    (off_t)sn1 * raw_sino_bytes_per); \
              pread(input_fd, buf[sIdx].raw_buf[b*2+1], raw_sino_bytes_per, \
                    (off_t)sn2 * raw_sino_bytes_per); \
            } else { \
              /* Raw detector input: readRaw produces norm_sino at adj_xdim width. */ \
              /* Extract center det_xdim pixels (skip padding) for GPU kernel. */ \
              SINO_READ_OPTS _trs; \
              _trs.norm_sino = (float *)malloc(sizeof(float) * recon_info_record.sinogram_adjusted_xdim * n_angles); \
              readRaw(sn1, &recon_info_record, &_trs, input_fd); \
              for (int _a = 0; _a < n_angles; _a++) \
                memcpy(&buf[sIdx].raw_buf[b*2][(size_t)_a * det_xdim_raw], \
                       &_trs.norm_sino[(size_t)_a * recon_info_record.sinogram_adjusted_xdim + pad_front_raw], \
                       det_xdim_raw * sizeof(float)); \
              readRaw(sn2, &recon_info_record, &_trs, input_fd); \
              for (int _a = 0; _a < n_angles; _a++) \
                memcpy(&buf[sIdx].raw_buf[b*2+1][(size_t)_a * det_xdim_raw], \
                       &_trs.norm_sino[(size_t)_a * recon_info_record.sinogram_adjusted_xdim + pad_front_raw], \
                       det_xdim_raw * sizeof(float)); \
              free(_trs.norm_sino); \
            } \
            buf[sIdx].sino1[b] = buf[sIdx].raw_buf[b*2]; \
            buf[sIdx].sino2[b] = buf[sIdx].raw_buf[b*2+1]; \
            buf[sIdx].recon1[b] = buf[sIdx].compact_buf[b*2]; \
            buf[sIdx].recon2[b] = buf[sIdx].compact_buf[b*2+1]; \
            /* Stripe removal on raw sinograms (before GPU upload) */ \
            if (recon_info_record.doStripeRemoval) { \
              cleanup_sinogram_stripes(buf[sIdx].raw_buf[b*2], n_angles, det_xdim_raw, \
                  recon_info_record.stripeSnr, recon_info_record.stripeLaSize, \
                  recon_info_record.stripeSmSize, 1); \
              cleanup_sinogram_stripes(buf[sIdx].raw_buf[b*2+1], n_angles, det_xdim_raw, \
                  recon_info_record.stripeSnr, recon_info_record.stripeLaSize, \
                  recon_info_record.stripeSmSize, 1); \
            } \
          } \
        } while(0)

        // ── Helper: write compact recons to output (OMP-parallel memcpy via mmap) ──
        #define GPU_WRITE_BATCH(sIdx) do { \
          int _cnt = buf[sIdx].count; \
          _Pragma("omp parallel for schedule(dynamic, 1) num_threads(gpu_rw_threads)") \
          for (int b = 0; b < _cnt; b++) { \
            size_t off1 = (size_t)buf[sIdx].oldSliceNr[b] * compact_recon_bytes_per; \
            size_t off2 = (size_t)buf[sIdx].sliceNr[b] * compact_recon_bytes_per; \
            if (output_mmap) { \
              memcpy((char *)output_mmap + off1, buf[sIdx].compact_buf[b*2], compact_recon_bytes_per); \
              memcpy((char *)output_mmap + off2, buf[sIdx].compact_buf[b*2+1], compact_recon_bytes_per); \
            } else { \
              pwrite(output_fd, buf[sIdx].compact_buf[b*2], compact_recon_bytes_per, off1); \
              pwrite(output_fd, buf[sIdx].compact_buf[b*2+1], compact_recon_bytes_per, off2); \
            } \
          } \
        } while(0)

        // ── Pipeline: GPU thread computes while main thread does I/O ──
        // Read first batch into buf[0]
        double tr0 = TOMO_WTIME();
        GPU_READ_BATCH(0, 0);
        gpu_t_read += TOMO_WTIME() - tr0;

        for (int bi = 0; bi < nBatches; bi++) {
          int cur = bi % 2;
          int nxt = 1 - cur;
          int nextStart = (bi + 1) * gpu_batch_pairs;
          int hasNext = (nextStart < totalPairs);
          int hasPrev = (bi > 0);

          // Launch GPU compute in a dedicated pthread
          GPURawWork gw;
          gw.ctx = gpu_ctx;
          gw.count = buf[cur].count;
          gw.sino1 = buf[cur].sino1; gw.sino2 = buf[cur].sino2;
          gw.recon1 = buf[cur].recon1; gw.recon2 = buf[cur].recon2;
          gw.M = param.M; gw.M0 = param.M0;
          gw.M02 = param.M02; gw.pdim = param.pdim;
          gw.det_xdim = det_xdim_raw;
          gw.adjusted_xdim = recon_info_record.sinogram_adjusted_xdim;
          gw.sino_xdim = recon_info_record.sinogram_xdim;
          gw.recon_xdim = recon_xdim_out;
          gw.pad_front = pad_front_raw;
          gw.shift = information.shift;
          gw.doLog = recon_info_record.doLogProj;
          gw.auto_centering = recon_info_record.auto_centering;
          pthread_t gpu_thr;
          double tc0 = TOMO_WTIME();
          pthread_create(&gpu_thr, NULL, gpu_raw_compute_func, &gw);

          // Meanwhile on main thread: write prev results + read next batch
          if (hasPrev) {
            double tw0 = TOMO_WTIME();
            GPU_WRITE_BATCH(nxt);  // nxt == prev buffer (just swapped)
            gpu_t_write += TOMO_WTIME() - tw0;
          }
          if (hasNext) {
            double trN = TOMO_WTIME();
            GPU_READ_BATCH(nxt, nextStart);
            gpu_t_read += TOMO_WTIME() - trN;
          }

          // Wait for GPU compute to finish
          pthread_join(gpu_thr, NULL);
          gpu_t_compute += TOMO_WTIME() - tc0;
        }

        // Write last batch
        {
          int last = (nBatches - 1) % 2;
          double tw0 = TOMO_WTIME();
          GPU_WRITE_BATCH(last);
          gpu_t_write += TOMO_WTIME() - tw0;
        }

        fprintf(stderr, "TOMO GPU dispatch: read=%.3fs compute=%.3fs write=%.3fs total=%.3fs\n",
                gpu_t_read, gpu_t_compute, gpu_t_write,
                gpu_t_read + gpu_t_compute + gpu_t_write);
        #undef GPU_READ_BATCH
        #undef GPU_WRITE_BATCH
        #undef TOMO_WTIME

        // Cleanup
        for (int s = 0; s < 2; s++) {
          for (int b = 0; b < gpu_batch_pairs; b++) {
            tomo_gpu_pinned_free(buf[s].raw_buf[b*2]);
            tomo_gpu_pinned_free(buf[s].raw_buf[b*2+1]);
            tomo_gpu_pinned_free(buf[s].compact_buf[b*2]);
            tomo_gpu_pinned_free(buf[s].compact_buf[b*2+1]);
          }
          free(buf[s].raw_buf); free(buf[s].compact_buf);
          free(buf[s].sino1); free(buf[s].sino2);
          free(buf[s].recon1); free(buf[s].recon2);
          free(buf[s].oldSliceNr); free(buf[s].sliceNr);
        }
        if (sino_mmap) munmap(sino_mmap, sino_mmap_len);
        if (output_mmap) {
          msync(output_mmap, output_mmap_len, MS_SYNC);
          munmap(output_mmap, output_mmap_len);
        }
      } else
#endif
      {
      for (numSlice = 0; numSlice < totalPairs;
           numSlice++) {
        memsets(&information, &recon_info_record);
        int sliceNr;
        sliceRowNr = startSliceNr + numSlice * 2;
        sliceNr = recon_info_record.slices_to_process[sliceRowNr];
        oldSliceNr = sliceNr;
        if (cpu_sino_mmap) {
          readStruct.init_sinogram = cpu_sino_mmap + (size_t)sliceNr * recon_info_record.det_xdim * recon_info_record.theta_list_size;
          Pad(&readStruct, &recon_info_record);
          readStruct.init_sinogram = NULL;
        } else if (recon_info_record.are_sinos) {
          int rc = readSino(sliceNr, &recon_info_record, &readStruct);
          if (rc == 1) {
            badReadSingle = 1;
            continue;
          }
        } else {
          int rc = readRaw(sliceNr, &recon_info_record, &readStruct, input_fd);
          if (rc == 1) {
            badReadSingle = 1;
            continue;
          }
        }
        // Stripe removal on normalized sinogram
        if (recon_info_record.doStripeRemoval) {
          cleanup_sinogram_stripes(
              readStruct.norm_sino, recon_info_record.theta_list_size,
              recon_info_record.sinogram_adjusted_xdim,
              recon_info_record.stripeSnr, recon_info_record.stripeLaSize,
              recon_info_record.stripeSmSize, 1);
        }
        memcpy(information.sino_calc_buffer, readStruct.norm_sino,
               sizeof(float) * information.sinogram_adjusted_xdim *
                   recon_info_record.theta_list_size);
        offt = 0;
        offsetRecons = 0;
        reconCentering(&information, &recon_info_record, offt,
                       recon_info_record.doLogProj);
        setSinoAndReconBuffers(
            1, &information.sinograms_boundary_padding[offt],
            &information.reconstructions_boundary_padding[offsetRecons],
            &param);
        sliceRowNr++;
        sliceNr = recon_info_record.slices_to_process[sliceRowNr];
        if (cpu_sino_mmap) {
          readStruct.init_sinogram = cpu_sino_mmap + (size_t)sliceNr * recon_info_record.det_xdim * recon_info_record.theta_list_size;
          Pad(&readStruct, &recon_info_record);
          readStruct.init_sinogram = NULL;
        } else if (recon_info_record.are_sinos) {
          int rc = readSino(sliceNr, &recon_info_record, &readStruct);
          if (rc == 1) {
            badReadSingle = 1;
            continue;
          }
        } else {
          int rc = readRaw(sliceNr, &recon_info_record, &readStruct, input_fd);
          if (rc == 1) {
            badReadSingle = 1;
            continue;
          }
        }
        // Stripe removal on normalized sinogram (second slice)
        if (recon_info_record.doStripeRemoval) {
          cleanup_sinogram_stripes(
              readStruct.norm_sino, recon_info_record.theta_list_size,
              recon_info_record.sinogram_adjusted_xdim,
              recon_info_record.stripeSnr, recon_info_record.stripeLaSize,
              recon_info_record.stripeSmSize, 1);
        }
        memcpy(information.sino_calc_buffer, readStruct.norm_sino,
               sizeof(float) * information.sinogram_adjusted_xdim *
                   recon_info_record.theta_list_size);
        offt = information.sinogram_adjusted_size * 2;
        offsetRecons = information.reconstruction_size * 4;
        reconCentering(&information, &recon_info_record, offt,
                       recon_info_record.doLogProj);
        setSinoAndReconBuffers(
            2, &information.sinograms_boundary_padding[offt],
            &information.reconstructions_boundary_padding[offsetRecons],
            &param);
        {
          reconstruct(&param);
        }

        getRecons(&information, &recon_info_record, &param, 0);
        int rw = writeRecon(oldSliceNr, sliceRowNr - 1, &information,
                            &recon_info_record, 0, 0, output_fd);
        if (rw == 1) {
          badWriteSingle = 1;
          continue;
        }
        getRecons(&information, &recon_info_record, &param, offsetRecons);
        rw = writeRecon(sliceNr, sliceRowNr, &information, &recon_info_record,
                        0, 0, output_fd);

        if (rw == 1) {
          badWriteSingle = 1;
          continue;
        }
      }
      }
      if (input_fd != -1) {
        close(input_fd);
      }
      if (output_fd != -1) {
        close(output_fd);
      }
#ifdef ENABLE_CUDA
      // GPU path never calls reconstruct(), so FFTW plans were never
      // created.  Only destroy them when the CPU path was used.
      if (!(useGPU && gpu_ctx))
#endif
      destroyFFTMemoryStructures(&param);
    } // end #pragma omp parallel
    /* Never munmap a caller-owned buffer -- that memory belongs to the
     * calling process (a numpy array, in the Python case) and unmapping it
     * would corrupt the caller rather than just leak. */
    if (cpu_sino_mmap && !cpu_sino_is_caller_buffer)
      munmap(cpu_sino_mmap, cpu_sino_mmap_len);
    if (badReadSingle) {
      fprintf(stderr, "ERROR: could not read one or more sinograms.\n");
      return 1;
    }
    if (badWriteSingle) {
      fprintf(stderr, "ERROR: could not write one or more reconstructions.\n");
      return 1;
    }
  } else { // We have multiple shifts, (possibly multiple slices_to_process)
    SINO_READ_OPTS *readStruct;
    readStruct = malloc(recon_info_record.n_slices * sizeof(*readStruct));
    int i;
    for (i = 0; i < recon_info_record.n_slices; i++)
      readStruct[i].norm_sino = (float *)malloc(
          sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
          recon_info_record.theta_list_size);
    // ReadStruct is now ready.
    //~ int nJobs = (numProcs < recon_info_record.n_slices) ? numProcs :
    // recon_info_record.n_slices;
    int nJobs = recon_info_record.n_slices;
    int badRead = 0;
    int badWrite = 0;
    int input_fd = -1;
    if (!recon_info_record.are_sinos && !recon_info_record.use_hdf5) {
      input_fd = open(recon_info_record.DataFileName, O_RDONLY);
    }
    int output_fd = -1;
    if (recon_info_record.saveReconSeparate == 0) {
      char outFileName[4096];
      if (recon_info_record.n_cleanup_configs > 1) {
        sprintf(outFileName,
                "%s_NrCleanup_%03d_NrShifts_%03d_NrSlices_%05d_XDim_%06d_YDim_%06d_float32.bin",
                recon_info_record.ReconFileName,
                recon_info_record.n_cleanup_configs,
                recon_info_record.n_shifts, recon_info_record.n_slices,
                recon_info_record.reconstruction_xdim,
                recon_info_record.reconstruction_xdim);
        /* Companion metadata: list configs in cube order. */
        char cfgFileName[4096];
        sprintf(cfgFileName, "%s_cleanup_configs.txt",
                recon_info_record.ReconFileName);
        FILE *cfgf = fopen(cfgFileName, "w");
        if (cfgf != NULL) {
          fprintf(cfgf, "# cleanup_idx\tsnr\tla_size\tsm_size\n");
          int ci;
          for (ci = 0; ci < recon_info_record.n_cleanup_configs; ci++) {
            fprintf(cfgf, "%d\t%.4f\t%d\t%d\n", ci,
                    recon_info_record.cleanup_snr_values[ci],
                    recon_info_record.cleanup_la_values[ci],
                    recon_info_record.cleanup_sm_values[ci]);
          }
          fclose(cfgf);
        }
      } else {
        sprintf(outFileName,
                "%s_NrShifts_%03d_NrSlices_%05d_XDim_%06d_YDim_%06d_float32.bin",
                recon_info_record.ReconFileName, recon_info_record.n_shifts,
                recon_info_record.n_slices,
                recon_info_record.reconstruction_xdim,
                recon_info_record.reconstruction_xdim);
      }
      if (recon_info_record.recon_out_ptr == NULL)
        output_fd = open(outFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    }
#pragma omp parallel num_threads(nJobs)
    {
      // This will only read up to nJobs slices. TODO: Fix this!!!
      int procNr = omp_get_thread_num();
      int sliceNr;
      sliceNr = recon_info_record.slices_to_process[procNr];
      //~ printf("Reading SliceNr: %d.\n",sliceNr);
      if (recon_info_record.are_sinos) {
        int rc = readSino(sliceNr, &recon_info_record, &readStruct[procNr]);
        if (rc == 1)
          badRead = 1;
      } else {
        int rc =
            readRaw(sliceNr, &recon_info_record, &readStruct[procNr], input_fd);
        if (rc == 1)
          badRead = 1;
      }
    }
    if (input_fd != -1) {
      close(input_fd);
    }
    if (badRead == 1) {
      /* Was `return 0` -- i.e. success -- so a failed read produced a
       * silently empty reconstruction and a zero exit code. Harmless-looking
       * in a CLI where the operator sees the printed message; invisible to a
       * library caller. */
      fprintf(stderr, "ERROR: could not read one or more sinograms.\n");
      return 1;
    }

    /* Master clean copies of each sinogram. Allocated only when sweeping
     * cleanup configs — for the legacy single-config path, sinograms are
     * mutated in-place exactly as before (no extra memory). */
    float **master_sinos = NULL;
    size_t sino_floats = (size_t)recon_info_record.sinogram_adjusted_xdim *
                         (size_t)recon_info_record.theta_list_size;
    if (recon_info_record.n_cleanup_configs > 1) {
      master_sinos = (float **)malloc(
          sizeof(float *) * recon_info_record.n_slices);
      if (master_sinos == NULL) {
        fprintf(stderr, "Out of memory allocating master sino pointer array\n");
        return 1;
      }
      for (i = 0; i < recon_info_record.n_slices; i++) {
        master_sinos[i] = (float *)malloc(sizeof(float) * sino_floats);
        if (master_sinos[i] == NULL) {
          fprintf(stderr, "Out of memory allocating master sino %d\n", i);
          return 1;
        }
        memcpy(master_sinos[i], readStruct[i].norm_sino,
               sizeof(float) * sino_floats);
      }
    }

    /* OUTER LOOP — over cleanup configurations. For a single config this
     * runs once and reproduces the original behavior. */
    int cleanupNr;
    for (cleanupNr = 0; cleanupNr < recon_info_record.n_cleanup_configs;
         cleanupNr++) {
      /* Restore clean sinograms before applying this config's cleanup. */
      if (master_sinos != NULL) {
#pragma omp parallel for schedule(dynamic)
        for (i = 0; i < recon_info_record.n_slices; i++) {
          memcpy(readStruct[i].norm_sino, master_sinos[i],
                 sizeof(float) * sino_floats);
        }
      }

      /* Apply stripe removal — either the sweep's per-config params, or the
       * legacy single-config params. snr <= 0 is the "no cleanup" baseline. */
      int apply_cleanup;
      float use_snr;
      int use_la, use_sm;
      if (recon_info_record.n_cleanup_configs > 1) {
        use_snr = recon_info_record.cleanup_snr_values[cleanupNr];
        use_la = recon_info_record.cleanup_la_values[cleanupNr];
        use_sm = recon_info_record.cleanup_sm_values[cleanupNr];
        apply_cleanup = (use_snr > 0.0f);
      } else {
        use_snr = recon_info_record.stripeSnr;
        use_la = recon_info_record.stripeLaSize;
        use_sm = recon_info_record.stripeSmSize;
        apply_cleanup = recon_info_record.doStripeRemoval ? 1 : 0;
      }
      if (apply_cleanup) {
        if (recon_info_record.n_cleanup_configs > 1) {
          printf("Cleanup config %d/%d: snr=%.2f la=%d sm=%d\n",
                 cleanupNr + 1, recon_info_record.n_cleanup_configs,
                 use_snr, use_la, use_sm);
        }
#pragma omp parallel for schedule(dynamic)
        for (i = 0; i < recon_info_record.n_slices; i++) {
          cleanup_sinogram_stripes(
              readStruct[i].norm_sino, recon_info_record.theta_list_size,
              recon_info_record.sinogram_adjusted_xdim,
              use_snr, use_la, use_sm, 1);
        }
      } else if (recon_info_record.n_cleanup_configs > 1) {
        printf("Cleanup config %d/%d: baseline (no stripe removal)\n",
               cleanupNr + 1, recon_info_record.n_cleanup_configs);
      }

      nJobs = recon_info_record.n_slices * recon_info_record.n_shifts;
      int innerProcs = (nJobs / 2 < numProcs) ? nJobs / 2 : numProcs;
      if (innerProcs < 1) innerProcs = 1;
      int nrSlicesThread = (int)ceil((double)nJobs / (2.0 * (double)innerProcs));
      if (cleanupNr == 0) {
        printf("Number of FFT jobs per thread %d, Number of threads: %d.\n"
               "Starting processing.\n",
               nrSlicesThread, innerProcs);
      }
#pragma omp parallel num_threads(innerProcs)
      {
        int procNr = omp_get_thread_num();
        int startJobNr, endJobNr;
        startJobNr = procNr * nrSlicesThread * 2;
        endJobNr = (startJobNr + nrSlicesThread * 2 < nJobs)
                       ? startJobNr + nrSlicesThread * 2
                       : nJobs;
        LOCAL_CONFIG_OPTS information;
        information.shift = recon_info_record.shift_values[0];
        setSinoSize(&information, &recon_info_record);
        gridrecParams param;
        param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
        param.theta_list = recon_info_record.theta_list;
        param.filter_type = recon_info_record.filter;
        param.theta_list_size = recon_info_record.theta_list_size;
        param.wisdom_string = (char *)malloc(
            sizeof(char) * (strlen(recon_info_record.wisdom_string) + 1));
        param.setPlan = 0;
        param.error = MIDAS_TOMO_OK;
        param.fft_engine = recon_info_record.fft_engine;
        param.deterministic = recon_info_record.deterministic;
        strcpy(param.wisdom_string, recon_info_record.wisdom_string);
        size_t offt, offsetRecons;
        setGridRecPSWF(&param);
        initFFTMemoryStructures(&param);
        initGridRec(&param);
        int jobNr, sliceNr, shiftNr, localSliceNr;
        for (jobNr = 0; jobNr < (endJobNr - startJobNr) / 2; jobNr++) {
          memsets(&information, &recon_info_record);
          sliceNr = (startJobNr + jobNr * 2) / recon_info_record.n_shifts;
          shiftNr = (startJobNr + jobNr * 2) % recon_info_record.n_shifts;
          /* When n_shifts is odd (only legal at n_shifts==1 with sweep)
           * the +1 below would walk off shift_values; guard. */
          int shiftNr2 = (shiftNr + 1 < recon_info_record.n_shifts)
                             ? shiftNr + 1
                             : shiftNr;
          localSliceNr = recon_info_record.slices_to_process[sliceNr];
          information.shift = recon_info_record.shift_values[shiftNr];
          memcpy(information.sino_calc_buffer, readStruct[sliceNr].norm_sino,
                 sizeof(float) * information.sinogram_adjusted_xdim *
                     recon_info_record.theta_list_size);
          offt = 0;
          offsetRecons = 0;
          reconCentering(&information, &recon_info_record, offt,
                         recon_info_record.doLogProj);
          setSinoAndReconBuffers(
              1, &information.sinograms_boundary_padding[offt],
              &information.reconstructions_boundary_padding[offsetRecons],
              &param);
          information.shift = recon_info_record.shift_values[shiftNr2];
          memcpy(information.sino_calc_buffer, readStruct[sliceNr].norm_sino,
                 sizeof(float) * information.sinogram_adjusted_xdim *
                     recon_info_record.theta_list_size);
          offt = information.sinogram_adjusted_size * 2;
          offsetRecons = information.reconstruction_size * 4;
          reconCentering(&information, &recon_info_record, offt,
                         recon_info_record.doLogProj);
          setSinoAndReconBuffers(
              2, &information.sinograms_boundary_padding[offt],
              &information.reconstructions_boundary_padding[offsetRecons],
              &param);
#ifdef ENABLE_CUDA
          if (useGPU && gpu_ctx) {
            tomo_gpu_reconstruct(gpu_ctx,
                param.sinogram1, param.sinogram2,
                param.reconstruction1, param.reconstruction2,
                param.M, param.M0, param.M02, param.pdim);
          } else
#endif
          {
            reconstruct(&param);
          }
          information.shift = recon_info_record.shift_values[shiftNr];
          getRecons(&information, &recon_info_record, &param, 0);
          int rw = writeRecon(localSliceNr, sliceNr, &information,
                              &recon_info_record, shiftNr, cleanupNr,
                              output_fd);
          if (rw == 1) {
            badWrite = 1;
            continue;
          }
          if (shiftNr2 != shiftNr) {
            information.shift = recon_info_record.shift_values[shiftNr2];
            getRecons(&information, &recon_info_record, &param, offsetRecons);
            rw = writeRecon(localSliceNr, sliceNr, &information,
                            &recon_info_record, shiftNr2, cleanupNr,
                            output_fd);
            if (rw == 1) {
              badWrite = 1;
              continue;
            }
          }
        }
        destroyFFTMemoryStructures(&param);
        free(param.wisdom_string);
        freeSinoBuffers(&information);
      }
    } /* end outer cleanup loop */

    if (master_sinos != NULL) {
      for (i = 0; i < recon_info_record.n_slices; i++)
        free(master_sinos[i]);
      free(master_sinos);
    }
    if (badWrite) {
      fprintf(stderr, "ERROR: could not write one or more reconstructions.\n");
      if (output_fd != -1)
        close(output_fd);
      return 1;
    }
    if (output_fd != -1) {
      close(output_fd);
    }
  }
#ifdef ENABLE_CUDA
  if (gpu_ctx) {
    tomo_gpu_destroy(gpu_ctx);
    gpu_ctx = NULL;
  }
#endif

  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return 0;
}

/* Thin CLI wrapper.
 *
 * Deliberately retained rather than deleted once Python moved to calling the
 * library in-process: it is what dev/build_reference_binary.sh compares
 * against, so keeping it is what keeps the bitwise parity gate runnable.
 */
#ifndef MIDAS_TOMO_LIBRARY_BUILD
int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 3) {
    usage();
    return 1;
  }

  int useGPU = 0;
  int useFftwBridge = 0;
  int useDeterministic = 0;
  /* pocketfft is the DEFAULT (2026-08-13). It is BSD-licensed and vendored,
   * measurably faster than FFTW at these sizes, and reproducible run to run,
   * which the FFTW planner is not. Use --fft-engine=fftw to reproduce
   * historical runs bit-for-bit. */
  int fftEngine = MIDAS_FFT_POCKET;
  for (int argi = 3; argi < argc; argi++) {
    if (strcmp(argv[argi], "--gpu") == 0) {
      useGPU = 1;
    } else if (strcmp(argv[argi], "--fftw-bridge") == 0) {
      useFftwBridge = 1;
    } else if (strcmp(argv[argi], "--deterministic") == 0) {
      useDeterministic = 1;
    } else if (strncmp(argv[argi], "--fft-engine=", 13) == 0) {
      const char *want = argv[argi] + 13;
      if (strcmp(want, "pocketfft") == 0) {
        fftEngine = MIDAS_FFT_POCKET;
      } else if (strcmp(want, "fftw") == 0) {
        fftEngine = MIDAS_FFT_FFTW;
      } else {
        fprintf(stderr, "ERROR: unknown --fft-engine=%s (use fftw or pocketfft)\n",
                want);
        return 1;
      }
    } else {
      fprintf(stderr, "Warning: ignoring unrecognised argument '%s'.\n",
              argv[argi]);
    }
  }

  return midas_tomo_run_engine(argv[1], atoi(argv[2]), useGPU, useFftwBridge,
                               useDeterministic, fftEngine);
}
#endif /* MIDAS_TOMO_LIBRARY_BUILD */
