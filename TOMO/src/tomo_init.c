//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "midas_version.h"
#include "tomo_heads.h"
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
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

int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 3) {
    usage();
    return 1;
  }

  // Parse optional GPU flags from argv[3..]
  int useGPU = 0;
  int useFftwBridge = 0;
  for (int argi = 3; argi < argc; argi++) {
    if (strcmp(argv[argi], "--gpu") == 0) {
      useGPU = 1;
    } else if (strcmp(argv[argi], "--fftw-bridge") == 0) {
      useFftwBridge = 1;
    }
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
  recon_info_record.sizeMatrices = 0;
  char *fileName;
  fileName = argv[1];
  int RC;
  RC = setGlobalOpts(fileName, &recon_info_record);
  setReadStructSize(&recon_info_record);
  gridrecParams pm;
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
  if (access(plan2DFN, F_OK) == -1) {
    printf("FFT plan file did not exist, creating one %s.\n",
           plan2DFN); // Check if sizes are okay.
    createPlanFile(&recon_info_record);
  } else if (access(plan1DFN, F_OK) == -1) {
    printf("FFT plan file did not exist, creating one %s.\n", plan2DFN);
    createPlanFile(&recon_info_record);
  } else {
    printf("Reading wisdom file %s.\n", plan2DFN);
    createPlanFile(&recon_info_record);
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
      (atoi(argv[2]) < maxNProcs - 2) ? atoi(argv[2]) : maxNProcs - 2;
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
  } else {
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
  if (recon_info_record.n_shifts == 1) {
    printf("Starting processing of all slices with %d threads.\n", numProcs);
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
        output_fd = open(outFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
      }
      int totalPairs = (endSliceNr - startSliceNr) / 2;
#ifdef ENABLE_CUDA
      if (useGPU && gpu_ctx) {
        // GPU: double-buffered pipeline with OMP-parallel read/write
        int gpu_batch_pairs = 50;
        size_t sino_bp_size = information.sinogram_adjusted_size * 4;
        size_t recon_bp_size = information.reconstruction_size * 8;
        double gpu_t_read = 0, gpu_t_compute = 0, gpu_t_write = 0;
        struct timespec ts_tmp;
        #define TOMO_WTIME() (clock_gettime(CLOCK_MONOTONIC, &ts_tmp), \
                              ts_tmp.tv_sec + ts_tmp.tv_nsec * 1e-9)
        int gpu_read_threads = omp_get_max_threads();
        if (!recon_info_record.are_sinos) gpu_read_threads = 1;

        // Double-buffered batch state: buf[0] and buf[1]
        typedef struct {
          const float **sino1;
          const float **sino2;
          float **recon1;
          float **recon2;
          int *oldSliceNr;
          int *sliceNr;
          size_t *offsetRecons;
          float **sino_bp;
          float **recon_bp;
          int count;  // how many pairs in this batch
        } BatchBuf;
        BatchBuf buf[2];
        for (int s = 0; s < 2; s++) {
          buf[s].sino1 = (const float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].sino2 = (const float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].recon1 = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].recon2 = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].oldSliceNr = (int *)malloc(gpu_batch_pairs * sizeof(int));
          buf[s].sliceNr = (int *)malloc(gpu_batch_pairs * sizeof(int));
          buf[s].offsetRecons = (size_t *)malloc(gpu_batch_pairs * sizeof(size_t));
          buf[s].sino_bp = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          buf[s].recon_bp = (float **)malloc(gpu_batch_pairs * sizeof(float *));
          for (int b = 0; b < gpu_batch_pairs; b++) {
            buf[s].sino_bp[b] = (float *)calloc(sino_bp_size, sizeof(float));
            buf[s].recon_bp[b] = (float *)calloc(recon_bp_size, sizeof(float));
          }
          buf[s].count = 0;
        }

        // ── Helper: read a batch into buf[s] ──
        #define GPU_READ_BATCH(s, batchStart) do { \
          int _this = totalPairs - (batchStart); \
          if (_this > gpu_batch_pairs) _this = gpu_batch_pairs; \
          buf[s].count = _this; \
          _Pragma("omp parallel for schedule(dynamic, 1) num_threads(gpu_read_threads)") \
          for (int b = 0; b < _this; b++) { \
            LOCAL_CONFIG_OPTS ti; \
            ti.sinogram_adjusted_xdim = information.sinogram_adjusted_xdim; \
            ti.sinogram_adjusted_size = information.sinogram_adjusted_size; \
            ti.reconstruction_size = information.reconstruction_size; \
            ti.shift = information.shift; \
            ti.shifted_sinogram = (float *)malloc(sizeof(float) * ti.sinogram_adjusted_size); \
            ti.sinograms_boundary_padding = (float *)calloc(ti.sinogram_adjusted_size * 4, sizeof(float)); \
            ti.sino_calc_buffer = (float *)malloc(sizeof(float) * ti.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            ti.reconstructions_boundary_padding = (float *)calloc(ti.reconstruction_size * 8, sizeof(float)); \
            ti.shifted_recon = (float *)malloc(sizeof(float) * ti.reconstruction_size); \
            ti.recon_calc_buffer = (float *)malloc(sizeof(float) * ti.reconstruction_size * 2); \
            ti.mean_vect = (float *)malloc(sizeof(float) * recon_info_record.sinogram_ydim); \
            ti.mean_sino_line_data = (float *)malloc(sizeof(float) * ti.sinogram_adjusted_xdim); \
            ti.low_pass_sino_lines_data = (float *)malloc(sizeof(float) * ti.sinogram_adjusted_xdim); \
            SINO_READ_OPTS trs; \
            trs.norm_sino = (float *)malloc(sizeof(float) * recon_info_record.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            int pi = (batchStart) + b; \
            memset(buf[s].sino_bp[b], 0, sino_bp_size * sizeof(float)); \
            memset(buf[s].recon_bp[b], 0, recon_bp_size * sizeof(float)); \
            /* slice 1 */ \
            int sr = startSliceNr + pi * 2; \
            int sn = recon_info_record.slices_to_process[sr]; \
            buf[s].oldSliceNr[b] = sn; \
            memset(trs.norm_sino, 0, sizeof(float) * recon_info_record.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            memsets(&ti, &recon_info_record); \
            if (recon_info_record.are_sinos) readSino(sn, &recon_info_record, &trs); \
            else readRaw(sn, &recon_info_record, &trs, input_fd); \
            if (recon_info_record.doStripeRemoval) cleanup_sinogram_stripes(trs.norm_sino, recon_info_record.theta_list_size, recon_info_record.sinogram_adjusted_xdim, recon_info_record.stripeSnr, recon_info_record.stripeLaSize, recon_info_record.stripeSmSize, 1); \
            memcpy(ti.sino_calc_buffer, trs.norm_sino, sizeof(float) * ti.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            reconCentering(&ti, &recon_info_record, 0, recon_info_record.doLogProj); \
            memcpy(buf[s].sino_bp[b], &ti.sinograms_boundary_padding[0], ti.sinogram_adjusted_size * 2 * sizeof(float)); \
            /* slice 2 */ \
            sn = recon_info_record.slices_to_process[sr + 1]; \
            buf[s].sliceNr[b] = sn; \
            memset(trs.norm_sino, 0, sizeof(float) * recon_info_record.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            if (recon_info_record.are_sinos) readSino(sn, &recon_info_record, &trs); \
            else readRaw(sn, &recon_info_record, &trs, input_fd); \
            if (recon_info_record.doStripeRemoval) cleanup_sinogram_stripes(trs.norm_sino, recon_info_record.theta_list_size, recon_info_record.sinogram_adjusted_xdim, recon_info_record.stripeSnr, recon_info_record.stripeLaSize, recon_info_record.stripeSmSize, 1); \
            memcpy(ti.sino_calc_buffer, trs.norm_sino, sizeof(float) * ti.sinogram_adjusted_xdim * recon_info_record.theta_list_size); \
            size_t of2 = ti.sinogram_adjusted_size * 2; \
            size_t or2 = ti.reconstruction_size * 4; \
            buf[s].offsetRecons[b] = or2; \
            reconCentering(&ti, &recon_info_record, of2, recon_info_record.doLogProj); \
            memcpy(&buf[s].sino_bp[b][of2], &ti.sinograms_boundary_padding[of2], ti.sinogram_adjusted_size * 2 * sizeof(float)); \
            buf[s].sino1[b] = &buf[s].sino_bp[b][0]; \
            buf[s].sino2[b] = &buf[s].sino_bp[b][of2]; \
            buf[s].recon1[b] = &buf[s].recon_bp[b][0]; \
            buf[s].recon2[b] = &buf[s].recon_bp[b][or2]; \
            free(ti.shifted_sinogram); free(ti.sinograms_boundary_padding); free(ti.sino_calc_buffer); \
            free(ti.reconstructions_boundary_padding); free(ti.shifted_recon); free(ti.recon_calc_buffer); \
            free(ti.mean_vect); free(ti.mean_sino_line_data); free(ti.low_pass_sino_lines_data); free(trs.norm_sino); \
          } \
        } while(0)

        // ── Helper: write a batch from buf[s] (OMP-parallel) ──
        #define GPU_WRITE_BATCH(s) do { \
          int _cnt = buf[s].count; \
          _Pragma("omp parallel for schedule(dynamic, 1) num_threads(gpu_read_threads)") \
          for (int b = 0; b < _cnt; b++) { \
            LOCAL_CONFIG_OPTS wi; \
            wi.sinogram_adjusted_xdim = information.sinogram_adjusted_xdim; \
            wi.sinogram_adjusted_size = information.sinogram_adjusted_size; \
            wi.reconstruction_size = information.reconstruction_size; \
            wi.shift = information.shift; \
            wi.sinograms_boundary_padding = buf[s].sino_bp[b]; \
            wi.reconstructions_boundary_padding = buf[s].recon_bp[b]; \
            wi.recon_calc_buffer = (float *)malloc(sizeof(float) * wi.reconstruction_size * 2); \
            wi.shifted_recon = (float *)malloc(sizeof(float) * wi.reconstruction_size); \
            gridrecParams wp; \
            setSinoAndReconBuffers(1, &buf[s].sino_bp[b][0], &buf[s].recon_bp[b][0], &wp); \
            setSinoAndReconBuffers(2, &buf[s].sino_bp[b][wi.sinogram_adjusted_size * 2], &buf[s].recon_bp[b][buf[s].offsetRecons[b]], &wp); \
            getRecons(&wi, &recon_info_record, &wp, 0); \
            writeRecon(buf[s].oldSliceNr[b], &wi, &recon_info_record, 0, output_fd); \
            getRecons(&wi, &recon_info_record, &wp, buf[s].offsetRecons[b]); \
            writeRecon(buf[s].sliceNr[b], &wi, &recon_info_record, 0, output_fd); \
            free(wi.recon_calc_buffer); \
            free(wi.shifted_recon); \
          } \
        } while(0)

        // ── Pipeline: read[0] → { compute[cur] | read[nxt] | write[prev] } ──
        int nBatches = (totalPairs + gpu_batch_pairs - 1) / gpu_batch_pairs;
        int cur = 0;
        // Prime: read first batch
        double tr0 = TOMO_WTIME();
        GPU_READ_BATCH(cur, 0);
        gpu_t_read += TOMO_WTIME() - tr0;

        for (int bi = 0; bi < nBatches; bi++) {
          int nxt = 1 - cur;
          int nextBatchStart = (bi + 1) * gpu_batch_pairs;
          int hasNext = (nextBatchStart < totalPairs);
          int hasPrev = (bi > 0);

          // Launch GPU compute for buf[cur] (async — returns quickly)
          double tc0 = TOMO_WTIME();
          tomo_gpu_reconstruct_batch(gpu_ctx,
              buf[cur].count,
              buf[cur].sino1, buf[cur].sino2,
              buf[cur].recon1, buf[cur].recon2,
              param.M, param.M0, param.M02, param.pdim);
          gpu_t_compute += TOMO_WTIME() - tc0;

          // While GPU was computing, we could have overlapped read+write
          // but tomo_gpu_reconstruct_batch is currently synchronous.
          // So we do read+write after compute for now, but they overlap each other.
          #pragma omp parallel sections num_threads(2)
          {
            #pragma omp section
            {
              if (hasNext) {
                double tr = TOMO_WTIME();
                GPU_READ_BATCH(nxt, nextBatchStart);
                gpu_t_read += TOMO_WTIME() - tr;
              }
            }
            #pragma omp section
            {
              if (hasPrev) {
                int prev = 1 - cur;
                double tw = TOMO_WTIME();
                GPU_WRITE_BATCH(prev);
                gpu_t_write += TOMO_WTIME() - tw;
              }
            }
          }

          cur = nxt;
        }
        // Write last batch
        {
          int last = 1 - cur;
          double tw = TOMO_WTIME();
          GPU_WRITE_BATCH(last);
          gpu_t_write += TOMO_WTIME() - tw;
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
            free(buf[s].sino_bp[b]);
            free(buf[s].recon_bp[b]);
          }
          free(buf[s].sino_bp); free(buf[s].recon_bp);
          free(buf[s].sino1); free(buf[s].sino2);
          free(buf[s].recon1); free(buf[s].recon2);
          free(buf[s].oldSliceNr); free(buf[s].sliceNr);
          free(buf[s].offsetRecons);
        }
      } else
#endif
      {
      for (numSlice = 0; numSlice < totalPairs;
           numSlice++) {
        memset(readStruct.norm_sino, 0,
               sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
                   recon_info_record.theta_list_size);
        memsets(&information, &recon_info_record);
        int sliceNr;
        sliceRowNr = startSliceNr + numSlice * 2;
        sliceNr = recon_info_record.slices_to_process[sliceRowNr];
        oldSliceNr = sliceNr;
        if (recon_info_record.are_sinos) {
          int rc = readSino(sliceNr, &recon_info_record, &readStruct);
          if (rc == 1)
            continue;
        } else {
          int rc = readRaw(sliceNr, &recon_info_record, &readStruct, input_fd);
          if (rc == 1)
            continue;
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
        if (recon_info_record.are_sinos) {
          int rc = readSino(sliceNr, &recon_info_record, &readStruct);
          if (rc == 1)
            continue;
        } else {
          int rc = readRaw(sliceNr, &recon_info_record, &readStruct, input_fd);
          if (rc == 1)
            continue;
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
        int rw = writeRecon(oldSliceNr, &information, &recon_info_record, 0,
                            output_fd);
        if (rw == 1)
          continue;
        getRecons(&information, &recon_info_record, &param, offsetRecons);
        rw =
            writeRecon(sliceNr, &information, &recon_info_record, 0, output_fd);

        if (rw == 1)
          continue;
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
    int input_fd = -1;
    if (!recon_info_record.are_sinos && !recon_info_record.use_hdf5) {
      input_fd = open(recon_info_record.DataFileName, O_RDONLY);
    }
    int output_fd = -1;
    if (recon_info_record.saveReconSeparate == 0) {
      char outFileName[4096];
      sprintf(outFileName,
              "%s_NrShifts_%03d_NrSlices_%05d_XDim_%06d_YDim_%06d_float32.bin",
              recon_info_record.ReconFileName, recon_info_record.n_shifts,
              recon_info_record.n_slices, recon_info_record.reconstruction_xdim,
              recon_info_record.reconstruction_xdim);
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
    if (badRead == 1)
      return 0;
    // Apply stripe removal to all pre-read sinograms (parallel — each is
    // independent)
    if (recon_info_record.doStripeRemoval) {
#pragma omp parallel for schedule(dynamic)
      for (i = 0; i < recon_info_record.n_slices; i++) {
        cleanup_sinogram_stripes(
            readStruct[i].norm_sino, recon_info_record.theta_list_size,
            recon_info_record.sinogram_adjusted_xdim,
            recon_info_record.stripeSnr, recon_info_record.stripeLaSize,
            recon_info_record.stripeSmSize, 1);
      }
    }
    nJobs = recon_info_record.n_slices * recon_info_record.n_shifts;
    numProcs = (nJobs / 2 < numProcs) ? nJobs / 2 : numProcs;
    int nrSlicesThread = (int)ceil((double)nJobs / (2.0 * (double)numProcs));
    printf("Number of FFT jobs per thread %d, Number of threads: %d.\nStarting "
           "processing.\n",
           nrSlicesThread, numProcs);
#pragma omp parallel num_threads(numProcs)
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
        information.shift = recon_info_record.shift_values[shiftNr + 1];
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
        int rw = writeRecon(localSliceNr, &information, &recon_info_record,
                            shiftNr, output_fd);
        if (rw == 1)
          continue;
        information.shift = recon_info_record.shift_values[shiftNr + 1];
        getRecons(&information, &recon_info_record, &param, offsetRecons);
        rw = writeRecon(localSliceNr, &information, &recon_info_record,
                        shiftNr + 1, output_fd);
        if (rw == 1)
          continue;
      }
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
