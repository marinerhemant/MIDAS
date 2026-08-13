//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <fcntl.h>
#ifdef MIDAS_TOMO_HAVE_HDF5
#include <hdf5.h>
#endif
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "tomo_heads.h"

void LogProj(float *data, int xdim, int ydim) {
  int i, k;
  for (i = 0; i < ydim; i++) {
    float *row = &data[i * xdim];
    // Find row max
    float max = row[0];
    for (k = 1; k < xdim; k++) {
      if (row[k] > max)
        max = row[k];
    }
    // Single pass: log(max) - log(val) == log(max/val)
    float log_max = logf(max);
    for (k = 0; k < xdim; k++) {
      if (row[k] <= 0.0f)
        row[k] = 1.0f;
      row[k] = log_max - logf(row[k]);
    }
  }
}

void LogSinogram(float *data, int xdim, int ydim) {
  int i, k;
  for (i = 0; i < ydim; i++) {
    for (k = 0; k < xdim; k++) {
      if (data[i * xdim + k] > 0)
        data[i * xdim + k] = -1 * log(data[i * xdim + k]);
      else
        data[i * xdim + k] = 0;
    }
  }
}

void RingCorrectionSingle(float *data, float ring_coeff,
                          LOCAL_CONFIG_OPTS *information,
                          const GLOBAL_CONFIG_OPTS *recon_info_record) {
  int i, j, m;
  float mean_total;
  float tmp;
  int xdim = information->sinogram_adjusted_xdim;
  int ydim = recon_info_record->sinogram_ydim;
  for (m = 0; m < 20; m++) {
    for (i = 0; i < ydim; i++)
      information->mean_vect[i] = 0.0;
    mean_total = 0.0;
    for (i = 0; i < ydim; i++) {
      float *row = &data[i * xdim];
      float row_sum = 0.0f;
      for (j = 0; j < xdim; j++) {
        row_sum += row[j];
      }
      information->mean_vect[i] = row_sum / xdim;
      mean_total += information->mean_vect[i];
    }
    mean_total /= ydim;
    // Check convergence: if all row means are close to the global mean, stop
    float max_delta = 0.0f;
    for (i = 0; i < ydim; i++) {
      float delta = fabsf(information->mean_vect[i] - mean_total);
      if (delta > max_delta)
        max_delta = delta;
    }
    if (max_delta < 1e-6f)
      break;
    for (i = 0; i < ydim; i++) {
      if (information->mean_vect[i] != 0.0f) {
        float ratio = mean_total / information->mean_vect[i];
        float *row = &data[i * xdim];
        for (j = 0; j < xdim; j++) {
          row[j] *= ratio;
        }
      }
    }
    for (i = 0; i < xdim; i++)
      information->mean_sino_line_data[i] = 0.0;
    for (i = 0; i < ydim; i++)
      for (j = 0; j < xdim; j++)
        information->mean_sino_line_data[j] += data[i * xdim + j];
    for (i = 0; i < xdim; i++)
      information->mean_sino_line_data[i] /= ydim;
    for (j = 1; j < xdim - 1; j++) {
      information->low_pass_sino_lines_data[j] =
          (information->mean_sino_line_data[j - 1] +
           information->mean_sino_line_data[j] +
           information->mean_sino_line_data[j + 1]) /
          3.0;
    }
    information->low_pass_sino_lines_data[0] =
        information->mean_sino_line_data[0];
    information->low_pass_sino_lines_data[xdim - 1] =
        information->mean_sino_line_data[xdim - 1];
    for (i = 0; i < ydim; i++) {
      for (j = 0; j < xdim; j++) {
        tmp = information->mean_sino_line_data[j] -
              information->low_pass_sino_lines_data[j];
        if ((data[i * xdim + j] - (tmp * ring_coeff)) > 0.0)
          data[i * xdim + j] -= (tmp * ring_coeff);
        else
          data[i * xdim + j] = 0.0;
      }
    }
  }
}

/* This is the definition:
 * 1 dark(D_x),
 * 2 whites (W1_x, W2_x) and
 * y Images (I_x_y),
 * the intensity should be
 * I'_x_y = (I_x_y-D_x)/(W_x-D_x), where
 * W_x = (1-p)*W1_x + (p)*W2_x and
 * p = y/nr_y
 */
// This function assumes the short_sino is the proper sinogram, white_field_sino
// is two rows of first and last wf image slice, dark_field_sino_ave is a single
// slice. Size of each sino is recon_info_record->sinogram_xdim, output
// norm_sino is information->sinogram_adjusted_xdim (padded)

void Normalize(SINO_READ_OPTS *readStruct,
               const GLOBAL_CONFIG_OPTS *recon_info_record) {
  int pad_size = recon_info_record->sinogram_adjusted_xdim -
                 recon_info_record->sinogram_xdim,
      front_pad_size = pad_size / 2, back_pad_size = pad_size - front_pad_size;
  int adj_xdim = recon_info_record->sinogram_adjusted_xdim;
  int det_xdim = recon_info_record->det_xdim;
  int sino_xdim = recon_info_record->sinogram_xdim;
  int frameNr;
  for (frameNr = 0; frameNr < recon_info_record->sinogram_ydim; frameNr++) {
    float factor = (float)frameNr / (float)recon_info_record->theta_list_size;
    float *out_row = &readStruct->norm_sino[frameNr * adj_xdim];
    // Precompute white/dark for first and last pixel (used in padding)
    float white_first = (1 - factor) * readStruct->white_field_sino[0] +
                        factor * readStruct->white_field_sino[sino_xdim];
    float dark_first = readStruct->dark_field_sino_ave[0];
    float val_first =
        ((float)readStruct->short_sinogram[frameNr * det_xdim] - dark_first) /
        (white_first - dark_first);
    int last_col = sino_xdim - 1;
    float white_last =
        (1 - factor) * readStruct->white_field_sino[last_col] +
        factor * readStruct->white_field_sino[last_col + sino_xdim];
    float dark_last = readStruct->dark_field_sino_ave[last_col];
    float val_last =
        ((float)readStruct->short_sinogram[last_col + frameNr * det_xdim] -
         dark_last) /
        (white_last - dark_last);
    // Front padding — replicate first pixel value
    for (int p = 0; p < front_pad_size; p++) {
      out_row[p] = val_first;
    }
    // Actual pixels — no branching
    for (int p = 0; p < sino_xdim; p++) {
      int colNr = p;
      float white_temp =
          (1 - factor) * (float)readStruct->white_field_sino[colNr] +
          (factor) * (float)readStruct->white_field_sino[colNr + sino_xdim];
      out_row[front_pad_size + p] =
          ((float)readStruct->short_sinogram[colNr + frameNr * det_xdim] -
           readStruct->dark_field_sino_ave[colNr]) /
          (white_temp - readStruct->dark_field_sino_ave[colNr]);
    }
    // Back padding — replicate last pixel value
    for (int p = 0; p < back_pad_size; p++) {
      out_row[front_pad_size + sino_xdim + p] = val_last;
    }
  }
}

void Pad(SINO_READ_OPTS *readStruct,
         const GLOBAL_CONFIG_OPTS *
             recon_info_record) { // Take the sino directly read (init_sinogram)
                                  // and pad it, return norm_sino.
  int pad_size = recon_info_record->sinogram_adjusted_xdim -
                 recon_info_record->sinogram_xdim,
      front_pad_size = pad_size / 2, back_pad_size = pad_size - front_pad_size;
  int colNr, frameNr;
  for (frameNr = 0; frameNr < recon_info_record->sinogram_ydim; frameNr++) {
    for (colNr = 0; colNr < recon_info_record->sinogram_adjusted_xdim;
         colNr++) {
      if (colNr < front_pad_size)
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    ->sinogram_adjusted_xdim] =
            readStruct->init_sinogram[frameNr * recon_info_record->det_xdim];
      else if (colNr >= front_pad_size + recon_info_record->sinogram_xdim)
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    ->sinogram_adjusted_xdim] =
            readStruct->init_sinogram[recon_info_record->sinogram_xdim - 1 +
                                      frameNr * recon_info_record->det_xdim];
      else
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    ->sinogram_adjusted_xdim] =
            readStruct
                ->init_sinogram[colNr + frameNr * recon_info_record->det_xdim -
                                front_pad_size];
    }
  }
}

int setGlobalOpts(char *inputFN, GLOBAL_CONFIG_OPTS *recon_info_record) {
  /* Input file is a text file name with a data link: sino data is a
   * !!!single!!! binary file with darks, whites and tomo data in that order.
   * The rest of the file consists of the parameters required.
   * Parameters to be supplied:
   * dataFileName: [char*] name of the file with the raw data or sino data
   * reconFileName: [char*] Name of the file for saving the reconstruction
   * areSinos: If the input is a sinogram instead of raw (cleaned) images [0 or
   * 1] The data can be one of two types: sinogram already with float data type,
   * directly give to reconstruct code with some additional centering etc.
   * 							dark[float], whites
   * (2,floats) and then raw images[shorts]. Using number of angles, we know how
   * many images are there. The scaling with white should be proportional to the
   * distance from a white and appropriate dark value. detXdim - [uint] detYdim
   * - [uint] Thetas can either be given as a range: thetaRange: startAngle
   * endAngle angleInterval - [floats] or a File: thetaFileName [char*] with
   * each line having an angle value [float]. filter - [int] set to * 0: default
   * 1: Shepp / Logan
   * 2: Hann
   * 3: Hamming
   * 4: Ramp
   * shiftValues: start_shift end_shift shift_interval [floats] In case of 1
   * shift, give start_shift=end_shift, shift_interval doesn't matter
   * ringRemovalCoefficient - If given, will do ringRemoval, otherwise comment
   * or remove line [float] default 1.0 doLog - If 1, will take Log of
   * intensities to calculate transmission, otherwise will use intensities
   * directly. [int] default 1 slicesToProcess - -1 for all or FileName ExtraPad
   * - 0 if half padding, 1 if one-half padding AutoCentering - 0 if don't want
   * reconstruction shifted in one direction (rotation axis in center of recon)
   * 				- 1 if want shift (rotation axis is offset)
   * [default]
   */
  int arbThetas = 0;
  FILE *fileParam;
  fileParam = fopen(inputFN, "r");
  if (fileParam == NULL)
    return 1;
  char dummy[4096], aline[4096], slices[4096];
  int temp;
  recon_info_record->use_ring_removal = 0;
  recon_info_record->debug = 0;
  recon_info_record->powerIncrement = 0;
  recon_info_record->doLogProj = 1;
  recon_info_record->auto_centering = 1;
  recon_info_record->saveReconSeparate = 1;
  recon_info_record->use_hdf5 = 0;
  recon_info_record->doStripeRemoval = 0;
  recon_info_record->stripeSnr = 3.0f;
  recon_info_record->stripeLaSize = 61;
  recon_info_record->stripeSmSize = 21;
  recon_info_record->n_cleanup_configs = 1;
  recon_info_record->cleanup_snr_values = NULL;
  recon_info_record->cleanup_la_values = NULL;
  recon_info_record->cleanup_sm_values = NULL;
  recon_info_record->stripeConfigFile[0] = '\0';
  /* Sentinels for the REQUIRED keys, validated after the parse loop below.
   * Without these these fields keep whatever was on the stack: a parameter
   * file missing detXdim would previously reach the allocator with a garbage
   * size and segfault. Survivable when this was only ever a standalone
   * binary; not survivable now that it is called in-process. */
  recon_info_record->det_xdim = 0;
  recon_info_record->det_ydim = 0;
  recon_info_record->DataFileName[0] = '\0';
  recon_info_record->ReconFileName[0] = '\0';
  recon_info_record->thetaFileName[0] = '\0';
  recon_info_record->are_sinos = 0;
  recon_info_record->filter = 0;
  recon_info_record->start_angle = 0.0f;
  recon_info_record->end_angle = 0.0f;
  recon_info_record->angle_interval = 0.0f;
  recon_info_record->start_shift = 0.0f;
  recon_info_record->end_shift = 0.0f;
  recon_info_record->shift_interval = 1.0f;
  while (fgets(aline, 4096, fileParam) != NULL) {
    if (strncmp(aline, "saveReconSeparate", strlen("saveReconSeparate")) == 0) {
      int val;
      sscanf(aline, "%s %d", dummy, &val);
      if (val == 0)
        recon_info_record->saveReconSeparate = 0;
      else
        recon_info_record->saveReconSeparate = 1;
    }
    if (strncmp(aline, "dataFileName", strlen("dataFileName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->DataFileName);
    }
    if (strncmp(aline, "reconFileName", strlen("reconFileName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->ReconFileName);
    }
    if (strncmp(aline, "areSinos", strlen("areSinos")) == 0) {
      sscanf(aline, "%s %ud", dummy, &recon_info_record->are_sinos);
    }
    if (strncmp(aline, "detXdim", strlen("detXdim")) == 0) {
      sscanf(aline, "%s %ud", dummy, &recon_info_record->det_xdim);
    }
    if (strncmp(aline, "detYdim", strlen("detYdim")) == 0) {
      sscanf(aline, "%s %ud", dummy, &recon_info_record->det_ydim);
    }
    if (strncmp(aline, "filter", strlen("filter")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->filter);
    }
    if (strncmp(aline, "debug", strlen("debug")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->debug);
    }
    if (strncmp(aline, "doLog", strlen("doLog")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->doLogProj);
    }
    if (strncmp(aline, "thetaRange", strlen("thetaRange")) == 0) {
      sscanf(aline, "%s %f %f %f", dummy, &recon_info_record->start_angle,
             &recon_info_record->end_angle, &recon_info_record->angle_interval);
    }
    if (strncmp(aline, "thetaFileName", strlen("thetaFileName")) == 0) {
      arbThetas = 1;
      sscanf(aline, "%s %s", dummy, recon_info_record->thetaFileName);
    }
    if (strncmp(aline, "shiftValues", strlen("shiftValues")) == 0) {
      sscanf(aline, "%s %f %f %f", dummy, &recon_info_record->start_shift,
             &recon_info_record->end_shift, &recon_info_record->shift_interval);
    }
    if (strncmp(aline, "ringRemovalCoeff", strlen("ringRemovalCoeff")) == 0) {
      recon_info_record->use_ring_removal = 1;
      sscanf(aline, "%s %f", dummy, &recon_info_record->ring_removal_coeff);
    }
    if (strncmp(aline, "slicesToProcess", strlen("slicesToProcess")) == 0) {
      sscanf(aline, "%s %s %s", dummy, slices, dummy);
    }
    if (strncmp(aline, "ExtraPad", strlen("ExtraPad")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->powerIncrement);
    }
    if (strncmp(aline, "AutoCentering", strlen("AutoCentering")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->auto_centering);
    }
    if (strncmp(aline, "HDF5FileName", strlen("HDF5FileName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->HDF5FileName);
      recon_info_record->use_hdf5 = 1;
    }
    if (strncmp(aline, "ImageDatasetName", strlen("ImageDatasetName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->ImageDatasetName);
    }
    if (strncmp(aline, "DarkDatasetName", strlen("DarkDatasetName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->DarkDatasetName);
    }
    if (strncmp(aline, "WhiteDatasetName", strlen("WhiteDatasetName")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->WhiteDatasetName);
    }
    if (strncmp(aline, "doStripeRemoval", strlen("doStripeRemoval")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->doStripeRemoval);
    }
    if (strncmp(aline, "stripeSnr", strlen("stripeSnr")) == 0) {
      sscanf(aline, "%s %f", dummy, &recon_info_record->stripeSnr);
    }
    if (strncmp(aline, "stripeLaSize", strlen("stripeLaSize")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->stripeLaSize);
    }
    if (strncmp(aline, "stripeSmSize", strlen("stripeSmSize")) == 0) {
      sscanf(aline, "%s %d", dummy, &recon_info_record->stripeSmSize);
    }
    if (strncmp(aline, "stripeConfigFile", strlen("stripeConfigFile")) == 0) {
      sscanf(aline, "%s %s", dummy, recon_info_record->stripeConfigFile);
    }
  }
  fclose(fileParam);

  /* Required-key validation. Previously setGlobalOpts only failed when the
   * parameter FILE could not be opened, so any missing key surfaced later as
   * a bad allocation or a segfault far from the cause. */
  if (recon_info_record->DataFileName[0] == '\0' &&
      recon_info_record->sino_in_ptr == NULL) {
    fprintf(stderr, "ERROR: parameter file is missing dataFileName.\n");
    return 1;
  }
  if (recon_info_record->ReconFileName[0] == '\0') {
    fprintf(stderr, "ERROR: parameter file is missing reconFileName.\n");
    return 1;
  }
  if (recon_info_record->det_xdim == 0 || recon_info_record->det_ydim == 0) {
    fprintf(stderr,
            "ERROR: parameter file is missing detXdim and/or detYdim "
            "(got %u x %u).\n",
            recon_info_record->det_xdim, recon_info_record->det_ydim);
    return 1;
  }
  if (recon_info_record->thetaFileName[0] == '\0' &&
      recon_info_record->angle_interval == 0.0f) {
    fprintf(stderr, "ERROR: parameter file gives neither thetaFileName nor a "
                    "usable thetaRange.\n");
    return 1;
  }
  if (recon_info_record->sino_in_ptr == NULL) {
    FILE *probe = fopen(recon_info_record->DataFileName, "rb");
    if (probe == NULL) {
      fprintf(stderr, "ERROR: cannot open dataFileName '%s'.\n",
              recon_info_record->DataFileName);
      return 1;
    }
    fclose(probe);
  }


  /* Cleanup parameter sweep: if a config file is provided and stripe removal
   * is enabled, parse it. Each non-comment / non-blank line is
   *   <snr>  <la_size>  <sm_size>
   * A row of (0, 0, 0) means "no cleanup" (baseline). Otherwise sm/la are
   * forced to odd. */
  if (recon_info_record->doStripeRemoval &&
      recon_info_record->stripeConfigFile[0] != '\0') {
    FILE *fcfg = fopen(recon_info_record->stripeConfigFile, "r");
    if (fcfg == NULL) {
      fprintf(stderr, "Could not open stripeConfigFile: %s\n",
              recon_info_record->stripeConfigFile);
      return 1;
    }
    /* First pass: count usable lines. */
    int n_cfg = 0;
    while (fgets(aline, 4096, fcfg) != NULL) {
      char *p = aline;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '#' || *p == '\n' || *p == '\0') continue;
      n_cfg++;
    }
    if (n_cfg < 1) {
      fprintf(stderr, "stripeConfigFile %s has no usable rows\n",
              recon_info_record->stripeConfigFile);
      fclose(fcfg);
      return 1;
    }
    rewind(fcfg);
    recon_info_record->n_cleanup_configs = n_cfg;
    recon_info_record->cleanup_snr_values =
        (float *)malloc(sizeof(float) * n_cfg);
    recon_info_record->cleanup_la_values = (int *)malloc(sizeof(int) * n_cfg);
    recon_info_record->cleanup_sm_values = (int *)malloc(sizeof(int) * n_cfg);
    if (!recon_info_record->cleanup_snr_values ||
        !recon_info_record->cleanup_la_values ||
        !recon_info_record->cleanup_sm_values) {
      fprintf(stderr, "Out of memory parsing stripeConfigFile\n");
      fclose(fcfg);
      return 1;
    }
    int idx = 0;
    while (fgets(aline, 4096, fcfg) != NULL && idx < n_cfg) {
      char *p = aline;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '#' || *p == '\n' || *p == '\0') continue;
      float snr;
      int la, sm;
      if (sscanf(p, "%f %d %d", &snr, &la, &sm) != 3) {
        fprintf(stderr, "Bad line in stripeConfigFile: %s", aline);
        fclose(fcfg);
        return 1;
      }
      if (snr <= 0.0f) {
        /* baseline: snr<=0 means skip cleanup for this config */
        snr = 0.0f;
        la = 0;
        sm = 0;
      } else {
        if (la % 2 == 0) la += 1;
        if (sm % 2 == 0) sm += 1;
      }
      recon_info_record->cleanup_snr_values[idx] = snr;
      recon_info_record->cleanup_la_values[idx] = la;
      recon_info_record->cleanup_sm_values[idx] = sm;
      idx++;
    }
    fclose(fcfg);
  }
  if (arbThetas == 0) {
    recon_info_record->theta_list_size =
        abs((recon_info_record->end_angle - recon_info_record->start_angle) /
            recon_info_record->angle_interval) +
        1;
    recon_info_record->theta_list =
        (float *)malloc(recon_info_record->theta_list_size * sizeof(float));
    int i;
    for (i = 0; i < recon_info_record->theta_list_size; i++) {
      recon_info_record->theta_list[i] = recon_info_record->start_angle +
                                         i * recon_info_record->angle_interval;
    }
  } else {
    recon_info_record->theta_list_size = 0;
    recon_info_record->theta_list =
        (float *)malloc(MAX_N_THETAS * sizeof(float));
    FILE *fileTheta = fopen(recon_info_record->thetaFileName, "r");
    while (fgets(aline, 4096, fileTheta) != NULL) {
      recon_info_record->theta_list[recon_info_record->theta_list_size] =
          atof(aline);
      recon_info_record->theta_list_size++;
    }
    fclose(fileTheta);
  }
  printf("Total number of thetas: %d\n", recon_info_record->theta_list_size);
  recon_info_record->n_shifts = (round)(abs((recon_info_record->end_shift -
                                             recon_info_record->start_shift)) /
                                        recon_info_record->shift_interval) +
                                1;
  recon_info_record->shift_values =
      (float *)malloc(sizeof(float) * (recon_info_record->n_shifts));
  int i;
  for (i = 0; i < recon_info_record->n_shifts; i++) {
    recon_info_record->shift_values[i] =
        recon_info_record->start_shift + i * recon_info_record->shift_interval;
  }
  long val;
  char *endptr;
  val = strtol(slices, &endptr, 10);
  if (endptr == slices) { // filename with slices, doesn't start with an integer
    sprintf(recon_info_record->SliceFileName, "%s", slices);
    FILE *slicesFile = fopen(recon_info_record->SliceFileName, "r");
    recon_info_record->n_slices = 0;
    recon_info_record->slices_to_process =
        (uint *)malloc(sizeof(uint) * recon_info_record->det_ydim);
    printf("We are reading the slices file: %s.\n", slices);
    while (fgets(aline, 4096, slicesFile) != NULL) {
      recon_info_record->slices_to_process[recon_info_record->n_slices] =
          atoi(aline);
      recon_info_record->n_slices++;
    }
    fclose(slicesFile);
  } else {
    if (strncmp(slices, "-1", strlen("-1")) == 0) {
      printf("We are doing all slices. Total number of slices: %d\n",
             recon_info_record->det_ydim);
      recon_info_record->slices_to_process =
          (uint *)malloc(sizeof(uint) * recon_info_record->det_ydim);
      for (i = 0; i < recon_info_record->det_ydim; i++)
        recon_info_record->slices_to_process[i] = i;
      recon_info_record->n_slices = recon_info_record->det_ydim;
    } else {
      printf("We are doing only 1 slice: %s\n", slices);
      recon_info_record->n_slices = 1;
      recon_info_record->slices_to_process = (uint *)malloc(sizeof(uint) * 1);
      recon_info_record->slices_to_process[0] = atoi(slices);
    }
  }
  recon_info_record->sinogram_ydim =
      recon_info_record->theta_list_size; // Equal to number of files
  recon_info_record->sinogram_xdim = recon_info_record->det_xdim;

  // Print configuration summary
  printf("\n");
  printf("========================================================\n");
  printf("          MIDAS TOMO - Configuration Summary\n");
  printf("========================================================\n");
  printf("  Data Input:\n");
  if (recon_info_record->use_hdf5) {
    printf("    HDF5 File       : %s\n", recon_info_record->HDF5FileName);
    printf("    Image Dataset   : %s\n", recon_info_record->ImageDatasetName);
    printf("    Dark Dataset    : %s\n", recon_info_record->DarkDatasetName);
    printf("    White Dataset   : %s\n", recon_info_record->WhiteDatasetName);
  } else {
    printf("    Data File       : %s\n", recon_info_record->DataFileName);
    printf("    Input Type      : %s\n",
           recon_info_record->are_sinos ? "Sinograms" : "Raw Projections");
  }
  printf("    Recon Output    : %s\n", recon_info_record->ReconFileName);
  printf("  Detector:\n");
  printf("    Dimensions      : %u x %u (X x Y)\n", recon_info_record->det_xdim,
         recon_info_record->det_ydim);
  printf("  Angles:\n");
  printf("    Theta Count     : %d\n", recon_info_record->theta_list_size);
  printf("    Range           : %.2f to %.2f\n",
         recon_info_record->theta_list[0],
         recon_info_record->theta_list[recon_info_record->theta_list_size - 1]);
  printf("  Reconstruction:\n");
  printf("    Filter          : %d\n", recon_info_record->filter);
  printf("    Shift Range     : %.2f to %.2f (step %.2f, n=%d)\n",
         recon_info_record->start_shift, recon_info_record->end_shift,
         recon_info_record->shift_interval, recon_info_record->n_shifts);
  printf("    Slices          : %d\n", recon_info_record->n_slices);
  printf("    Auto Centering  : %s\n",
         recon_info_record->auto_centering ? "Yes" : "No");
  printf("    Log Projection  : %s\n",
         recon_info_record->doLogProj ? "Yes" : "No");
  printf("    Extra Padding   : %d\n", recon_info_record->powerIncrement);
  printf("    Save Separate   : %s\n",
         recon_info_record->saveReconSeparate ? "Yes" : "No");
  printf("  Corrections:\n");
  printf("    Ring Removal    : %s",
         recon_info_record->use_ring_removal ? "Yes" : "No");
  if (recon_info_record->use_ring_removal)
    printf(" (coeff = %.4f)", recon_info_record->ring_removal_coeff);
  printf("\n");
  printf("    Stripe Removal  : %s",
         recon_info_record->doStripeRemoval ? "Yes" : "No");
  if (recon_info_record->doStripeRemoval) {
    if (recon_info_record->n_cleanup_configs > 1) {
      printf(" SWEEP over %d configs (file: %s)",
             recon_info_record->n_cleanup_configs,
             recon_info_record->stripeConfigFile);
    } else {
      printf(" (snr=%.1f, la=%d, sm=%d)", recon_info_record->stripeSnr,
             recon_info_record->stripeLaSize, recon_info_record->stripeSmSize);
    }
  }
  printf("\n");
  printf("========================================================\n\n");
  if (recon_info_record->doStripeRemoval &&
      recon_info_record->n_cleanup_configs > 1) {
    int ci;
    printf("  Cleanup configurations:\n");
    printf("    idx     snr    la_size   sm_size\n");
    for (ci = 0; ci < recon_info_record->n_cleanup_configs; ci++) {
      printf("    %3d   %5.2f   %7d   %7d\n", ci,
             recon_info_record->cleanup_snr_values[ci],
             recon_info_record->cleanup_la_values[ci],
             recon_info_record->cleanup_sm_values[ci]);
    }
    printf("\n");
  }

  return 0;
}

void setReadStructSize(GLOBAL_CONFIG_OPTS *recon_info_record) {
  int power, size;
  bool still_smaller;
  still_smaller = true;
  power = 0;
  while (still_smaller) {
    if (recon_info_record->sinogram_xdim > pow(2, power)) {
      power++;
      still_smaller = true;
    } else {
      still_smaller = false;
    }
  }
  if (recon_info_record->sinogram_xdim == pow(2, power)) {
    printf("Sinograms are a power of 2!\n");
  } else {
    printf("Sinograms are not a power of 2.  They will be increased to %d\n",
           (int)pow(2, power));
  }
  if (recon_info_record->powerIncrement == 1) {
    power++;
    printf("Extra padding was requested. Will increase the size of sinograms "
           "by 2 times. The size of reconstruction will be %d\n",
           (int)pow(2, power));
  }
  size = (int)pow(2, power);
  recon_info_record->sinogram_adjusted_xdim = size;
  recon_info_record->sinogram_adjusted_size =
      size * recon_info_record->sinogram_ydim;
  recon_info_record->reconstruction_xdim = size;
  recon_info_record->reconstruction_ydim = size;
  recon_info_record->reconstruction_size =
      recon_info_record->reconstruction_xdim *
      recon_info_record->reconstruction_ydim;
}

void memsets(LOCAL_CONFIG_OPTS *information,
             const GLOBAL_CONFIG_OPTS *recon_info_record) {
  // shifted_sinogram: fully overwritten by reconCentering interpolation loop
  // shifted_recon: getRecons does its own memset when auto_centering
  // Only zero boundary-padding buffers (padding regions not always written)
  memset(information->sinograms_boundary_padding, 0,
         sizeof(float) * information->sinogram_adjusted_size * 2 *
             2); // Hold two sinos
  memset(information->reconstructions_boundary_padding, 0,
         sizeof(float) * information->reconstruction_size * 4 *
             2); // Hold two recons
}

void setSinoSize(LOCAL_CONFIG_OPTS *information,
                 const GLOBAL_CONFIG_OPTS *recon_info_record) {
  information->sinogram_adjusted_xdim =
      recon_info_record->sinogram_adjusted_xdim;
  information->sinogram_adjusted_size =
      recon_info_record->sinogram_adjusted_size;
  information->reconstruction_size = recon_info_record->reconstruction_size;
  //~ printf("shifted_recon: %ld\n",(long)(sizeof
  //(float)*information->reconstruction_size)); ~ printf("shifted_sinogram
  //%ld\n",(long)(sizeof (float)*information->sinogram_adjusted_size)); ~
  // printf("sinograms_boundary_padding
  //%ld\n",(long)(sizeof(float)*information->sinogram_adjusted_size*2)); ~
  // printf("reconstructions_boundary_padding
  //%ld\n",(long)(sizeof(float)*information->reconstruction_size*4)); ~
  // printf("recon_calc_buffer
  //%ld\n",(long)(sizeof(float)*information->reconstruction_size*2));
  information->shifted_recon =
      (float *)malloc(sizeof(float) * information->reconstruction_size);
  information->shifted_sinogram =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_size);
  information->sinograms_boundary_padding =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_size * 2 *
                      2); // Hold two sinos
  information->reconstructions_boundary_padding =
      (float *)malloc(sizeof(float) * information->reconstruction_size * 4 *
                      2); // Hold two recons
  information->recon_calc_buffer =
      (float *)malloc(sizeof(float) * information->reconstruction_size * 2);
  information->sino_calc_buffer =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_xdim *
                      recon_info_record->theta_list_size);
  //~ printf("mean_vect %ld\n",(long)(sizeof
  //(float)*recon_info_record->sinogram_ydim)); ~ printf("mean_sino_line_data
  //%ld\n",(long)(sizeof (float)*information->sinogram_adjusted_xdim)); ~
  // printf("low_pass_sino_lines_data %ld\n",(long)(sizeof(float)
  //*information->sinogram_adjusted_xdim));
  information->mean_vect =
      (float *)malloc(sizeof(float) * recon_info_record->sinogram_ydim);
  information->mean_sino_line_data =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_xdim);
  information->low_pass_sino_lines_data =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_xdim);
}

/* Counterpart to setSinoSize — releases the per-thread buffers it allocated.
 * Safe to call multiple times only if you re-NULL the pointers; in practice
 * we call it exactly once at end of each thread's lifetime. */
void freeSinoBuffers(LOCAL_CONFIG_OPTS *information) {
  if (information->shifted_recon) free(information->shifted_recon);
  if (information->shifted_sinogram) free(information->shifted_sinogram);
  if (information->sinograms_boundary_padding)
    free(information->sinograms_boundary_padding);
  if (information->reconstructions_boundary_padding)
    free(information->reconstructions_boundary_padding);
  if (information->recon_calc_buffer) free(information->recon_calc_buffer);
  if (information->sino_calc_buffer) free(information->sino_calc_buffer);
  if (information->mean_vect) free(information->mean_vect);
  if (information->mean_sino_line_data) free(information->mean_sino_line_data);
  if (information->low_pass_sino_lines_data)
    free(information->low_pass_sino_lines_data);
}

int readSino(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
             SINO_READ_OPTS *readStruct) {
  FILE *dataFile = NULL;
  if (recon_info_record->sino_in_ptr == NULL) {
#pragma omp critical
    {
      dataFile = fopen(recon_info_record->DataFileName, "rb");
    }
    if (dataFile == NULL) {
      printf("SliceNr: %d, Could not read datafile: %s.\n", sliceNr,
             recon_info_record->DataFileName);
      return 1;
    }
  }
  size_t offset = sizeof(float) * sliceNr * recon_info_record->det_xdim *
                  recon_info_record->theta_list_size;
  size_t SizeSino = sizeof(float) * recon_info_record->det_xdim *
                    recon_info_record->theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  readStruct->sizeMatrices +=
      (sizeof(float) * recon_info_record->sinogram_adjusted_xdim *
       recon_info_record->theta_list_size);
  //~ printf("init_sinogram %ld\n",(long)SizeSino);
  //~ printf("norm_sino
  //%ld\n",(long)(sizeof(float)*recon_info_record->sinogram_adjusted_xdim*recon_info_record->theta_list_size));
  readStruct->init_sinogram = (float *)malloc(SizeSino);
  if (readStruct->init_sinogram == NULL) {
    printf("SliceNr: %d, could not allocate %zu bytes for the sinogram.\n",
           sliceNr, SizeSino);
    if (dataFile != NULL)
      fclose(dataFile);
    return 1;
  }
  if (recon_info_record->sino_in_ptr != NULL) {
    /* Caller-supplied buffer: copy this slice straight out of it. No file is
     * touched, which is the point -- staging a large sinogram stack to disk
     * purely to hand it to the engine was the dominant cost. */
    if (offset + SizeSino > recon_info_record->sino_in_bytes) {
      printf("SliceNr: %d, sinogram buffer is too small: need %zu bytes, "
             "have %zu.\n",
             sliceNr, offset + SizeSino, recon_info_record->sino_in_bytes);
      free(readStruct->init_sinogram);
      return 1;
    }
    memcpy(readStruct->init_sinogram,
           (const char *)recon_info_record->sino_in_ptr + offset, SizeSino);
  } else {
#pragma omp critical
    {
      fseek(dataFile, offset, SEEK_SET);
      fread(readStruct->init_sinogram, SizeSino, 1, dataFile);
    }
  }
  if (dataFile != NULL) {
#pragma omp critical
    {
      fclose(dataFile);
    }
  }
  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "init_sinogram_%s", recon_info_record->DataFileName);
    printf("%s", outfn);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->init_sinogram, SizeSino, 1, out);
    fclose(out);
  }
  Pad(readStruct, recon_info_record);
  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "norm_sino_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->norm_sino,
           sizeof(float) * recon_info_record->sinogram_adjusted_xdim *
               recon_info_record->theta_list_size,
           1, out);
    fclose(out);
  }
  free(readStruct->init_sinogram);
  return 0;
}

#ifndef MIDAS_TOMO_HAVE_HDF5
/* Built without HDF5.
 *
 * midas-tomo reads HDF5 in Python (h5py, see midas_tomo/hdf5.py) and hands
 * this engine the staged binary layout, so this code path is unused by the
 * package -- h5py also copes with more layouts, filters and chunkings than
 * the reader below. Keeping the symbol means readRaw() still links; reaching
 * it means a hand-written parameter file asked for HDF5FileName against a
 * build that cannot serve it, which is worth an explicit error rather than a
 * link failure. */
int readRawHDF5(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
                SINO_READ_OPTS *readStruct) {
  (void)sliceNr;
  (void)recon_info_record;
  (void)readStruct;
  fprintf(stderr,
          "ERROR: this MIDAS_TOMO was built without HDF5 support, but the "
          "parameter file specifies HDF5FileName.\n"
          "       Either rebuild with HDF5 available, or read the file in "
          "Python instead:\n"
          "         from midas_tomo.hdf5 import read_exchange\n"
          "         scan = read_exchange('input.h5')\n"
          "         run_tomo(scan.data, scan.dark, scan.whites, ...)\n");
  return 1;
}
#else
int readRawHDF5(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
                SINO_READ_OPTS *readStruct) {
  hid_t file_id;
#pragma omp critical
  {
    file_id =
        H5Fopen(recon_info_record->HDF5FileName, H5F_ACC_RDONLY, H5P_DEFAULT);
  }
  if (file_id < 0) {
    printf("SliceNr: %d, Could not open HDF5 file: %s.\n", sliceNr,
           recon_info_record->HDF5FileName);
    return 1;
  }

  size_t SizeDark, SizeWhite, SizeSino, SizeNormSino;
  hsize_t dims[3];

  // 1. Read Dark
  SizeDark = sizeof(float) * recon_info_record->det_xdim;
  readStruct->sizeMatrices += SizeDark;
  readStruct->dark_field_sino_ave = (float *)malloc(SizeDark);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record->DarkDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: Dark Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record->DarkDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);
      int ndims = H5Sget_simple_extent_ndims(dataspace_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      hsize_t offset[2] = {sliceNr, 0};
      hsize_t count[2] = {1, recon_info_record->det_xdim};

      if (ndims == 3) {
        if (dims[0] > 1) {
          // Multi-frame dark: Average them
          int nFrames = dims[0];
          float *temp_dark =
              (float *)malloc(sizeof(float) * recon_info_record->det_xdim);
          memset(readStruct->dark_field_sino_ave, 0, SizeDark);

          hsize_t count3[3] = {1, 1, recon_info_record->det_xdim};
          hid_t memspace_id = H5Screate_simple(1, &count3[2], NULL);

          for (int i = 0; i < nFrames; i++) {
            hsize_t offset3[3] = {i, sliceNr, 0};
            H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                                count3, NULL);
            H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                    H5P_DEFAULT, temp_dark);

            for (int j = 0; j < recon_info_record->det_xdim; j++) {
              readStruct->dark_field_sino_ave[j] += temp_dark[j];
            }
          }
          for (int j = 0; j < recon_info_record->det_xdim; j++) {
            readStruct->dark_field_sino_ave[j] /= (float)nFrames;
          }
          free(temp_dark);
          H5Sclose(memspace_id);
        } else {
          // Single frame 3D [1, Y, X]
          hsize_t offset3[3] = {0, sliceNr, 0};
          hsize_t count3[3] = {1, 1, recon_info_record->det_xdim};
          H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                              count3, NULL);
          hid_t memspace_id = H5Screate_simple(1, &count3[2], NULL);
          H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                  H5P_DEFAULT, readStruct->dark_field_sino_ave);
          H5Sclose(memspace_id);
        }
      } else {
        // 2D [Y, X]
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
                            NULL);
        hid_t memspace_id = H5Screate_simple(1, &count[1], NULL);
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                H5P_DEFAULT, readStruct->dark_field_sino_ave);
        H5Sclose(memspace_id);
      }
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    } else {
      printf("Could not open Dark dataset: %s\n",
             recon_info_record->DarkDatasetName);
    }
  }

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "dark_field_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->dark_field_sino_ave, SizeDark, 1, out);
    fclose(out);
  }

  // 2. Read Whites
  SizeWhite = sizeof(float) * recon_info_record->det_xdim * 2;
  readStruct->sizeMatrices += SizeWhite;
  readStruct->white_field_sino = (float *)malloc(SizeWhite);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record->WhiteDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: White Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record->WhiteDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);
      int ndims = H5Sget_simple_extent_ndims(dataspace_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      if (ndims == 3) {
        int nFrames = dims[0];
        if (nFrames > 2) {
          // Split Average: First half -> White 1, Second half -> White 2
          int mid = nFrames / 2;
          float *temp_white =
              (float *)malloc(sizeof(float) * recon_info_record->det_xdim);

          // White 1
          memset(readStruct->white_field_sino, 0,
                 sizeof(float) * recon_info_record->det_xdim);
          hsize_t count3[3] = {1, 1, recon_info_record->det_xdim};
          hid_t memspace_id = H5Screate_simple(1, &count3[2], NULL);

          for (int i = 0; i < mid; i++) {
            hsize_t offset3[3] = {i, sliceNr, 0};
            H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                                count3, NULL);
            H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                    H5P_DEFAULT, temp_white);
            for (int j = 0; j < recon_info_record->det_xdim; j++) {
              readStruct->white_field_sino[j] += temp_white[j];
            }
          }
          for (int j = 0; j < recon_info_record->det_xdim; j++) {
            readStruct->white_field_sino[j] /= (float)mid;
          }

          // White 2
          float *white2_ptr =
              (readStruct->white_field_sino) + recon_info_record->det_xdim;
          memset(white2_ptr, 0, sizeof(float) * recon_info_record->det_xdim);

          for (int i = mid; i < nFrames; i++) {
            hsize_t offset3[3] = {i, sliceNr, 0};
            H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                                count3, NULL);
            H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                    H5P_DEFAULT, temp_white);
            for (int j = 0; j < recon_info_record->det_xdim; j++) {
              white2_ptr[j] += temp_white[j];
            }
          }
          int count2 = nFrames - mid;
          for (int j = 0; j < recon_info_record->det_xdim; j++) {
            white2_ptr[j] /= (float)count2;
          }

          free(temp_white);
          H5Sclose(memspace_id);
        } else {
          // 1 or 2 frames
          // Read Frame 0
          hsize_t offset3[3] = {0, sliceNr, 0};
          hsize_t count3[3] = {1, 1, recon_info_record->det_xdim};
          H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                              count3, NULL);
          hid_t memspace_id = H5Screate_simple(1, &count3[2], NULL);
          H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                  H5P_DEFAULT, readStruct->white_field_sino);

          // Read Frame 1 (or 0 if only 1 frame)
          if (dims[0] > 1) {
            offset3[0] = 1;
            H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL,
                                count3, NULL);
            H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                    H5P_DEFAULT,
                    (readStruct->white_field_sino) +
                        recon_info_record->det_xdim);
          } else {
            memcpy((readStruct->white_field_sino) + recon_info_record->det_xdim,
                   readStruct->white_field_sino,
                   sizeof(float) * recon_info_record->det_xdim);
          }
          H5Sclose(memspace_id);
        }

      } else {
        // 2D [Y, X]
        hsize_t offset[2] = {sliceNr, 0};
        hsize_t count[2] = {1, recon_info_record->det_xdim};
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
                            NULL);
        hid_t memspace_id = H5Screate_simple(1, &count[1], NULL);
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                H5P_DEFAULT, readStruct->white_field_sino);
        // Duplicate
        memcpy((readStruct->white_field_sino) + recon_info_record->det_xdim,
               readStruct->white_field_sino,
               sizeof(float) * recon_info_record->det_xdim);
        H5Sclose(memspace_id);
      }
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    } else {
      printf("Could not open White dataset: %s\n",
             recon_info_record->WhiteDatasetName);
    }
  }

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "whites_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->white_field_sino, SizeWhite, 1, out);
    fclose(out);
  }

  // 3. Read Images (Sinogram)
  SizeSino = sizeof(unsigned short int) * recon_info_record->det_xdim *
             recon_info_record->theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  readStruct->short_sinogram = (unsigned short int *)malloc(SizeSino);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record->ImageDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: Image Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record->ImageDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);

      // Expected [nAngles, nY, nX]
      // We want [:, sliceNr, :]
      hsize_t offset3[3] = {0, sliceNr, 0};
      hsize_t count3[3] = {recon_info_record->theta_list_size, 1,
                           recon_info_record->det_xdim};

      // Verify dimensions
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
      if (dims[0] < recon_info_record->theta_list_size) {
        // Adjust count if file has fewer angles (shouldn't happen if setup is
        // correct)
        count3[0] = dims[0];
      }

      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL, count3,
                          NULL);

      // Memory dataspace: continuous buffer [theta_list_size * det_xdim]
      // We map the hyperslab to a 1D buffer in memory.
      // count3 represents the shape selected from file: {N, 1, X}
      // H5Dread will pack this into memory.
      // We can define a memory dataspace to be safe, or H5S_ALL/simple.
      // Let's use simple 1D memory space.
      hsize_t mem_dims[1] = {count3[0] * count3[2]};
      hid_t memspace_id = H5Screate_simple(1, mem_dims, NULL);

      // Use H5T_NATIVE_USHORT to ensure correct conversion
      H5Dread(dataset_id, H5T_NATIVE_USHORT, memspace_id, dataspace_id,
              H5P_DEFAULT, readStruct->short_sinogram);

      H5Sclose(memspace_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    } else {
      printf("Could not open Image dataset: %s\n",
             recon_info_record->ImageDatasetName);
    }

    H5Fclose(file_id);
  }

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "short_sinogram_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->short_sinogram, SizeSino, 1, out);
    fclose(out);
  }

  SizeNormSino = sizeof(float) * recon_info_record->sinogram_adjusted_xdim *
                 recon_info_record->theta_list_size;
  readStruct->sizeMatrices += SizeNormSino;
  Normalize(readStruct, recon_info_record);

#pragma omp critical
  {
    if (recon_info_record->debug > 0) {
      char outfn[4096];
      sprintf(outfn, "norm_sino_%s", recon_info_record->DataFileName);
      FILE *out = fopen(outfn, "ab");
      fwrite(readStruct->norm_sino, SizeNormSino, 1, out);
      fclose(out);
    }
  }

  free(readStruct->short_sinogram);
  free(readStruct->white_field_sino);
  free(readStruct->dark_field_sino_ave);

  return 0;
}
#endif /* MIDAS_TOMO_HAVE_HDF5 */

int readRaw(int sliceNr, const GLOBAL_CONFIG_OPTS *recon_info_record,
            SINO_READ_OPTS *readStruct, int fd) {
  if (recon_info_record->use_hdf5) {
    return readRawHDF5(sliceNr, recon_info_record, readStruct);
  }

  size_t offset, SizeDark, SizeWhite, SizeSino, SizeNormSino;
  // Dark
  SizeDark = sizeof(float) * recon_info_record->det_xdim;
  readStruct->sizeMatrices += SizeDark;
  //~ printf("dark_field_sino_ave %ld\n",(long)SizeDark);
  readStruct->dark_field_sino_ave = (float *)malloc(SizeDark);
  offset = sizeof(float) * sliceNr * recon_info_record->det_xdim;

  pread(fd, readStruct->dark_field_sino_ave, SizeDark, offset);

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "dark_field_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->dark_field_sino_ave, SizeDark, 1, out);
    fclose(out);
  }
  // 2 Whites
  SizeWhite = sizeof(float) * recon_info_record->det_xdim * 2;
  readStruct->sizeMatrices += SizeWhite;
  //~ printf("white_field_sino %ld\n",(long)SizeWhite);
  readStruct->white_field_sino = (float *)malloc(SizeWhite);
  offset =
      sizeof(float) * recon_info_record->det_xdim *
          recon_info_record->det_ydim                          // dark
      + sizeof(float) * recon_info_record->det_xdim * sliceNr; // Partial white

  pread(fd, readStruct->white_field_sino, SizeWhite / 2, offset); // One Row

  offset =
      sizeof(float) * recon_info_record->det_xdim *
          recon_info_record->det_ydim // dark
      + sizeof(float) * recon_info_record->det_xdim *
            recon_info_record->det_ydim                        // One full white
      + sizeof(float) * recon_info_record->det_xdim * sliceNr; // Partial white

  pread(fd, (readStruct->white_field_sino) + recon_info_record->det_xdim,
        SizeWhite / 2, offset); // Second Row

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "whites_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->white_field_sino, SizeWhite, 1, out);
    fclose(out);
  }
  // Sino start
  SizeSino = sizeof(unsigned short int) * recon_info_record->det_xdim *
             recon_info_record->theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  //~ printf("short_sinogram %ld\n",(long)SizeSino);
  readStruct->short_sinogram = (unsigned short int *)malloc(SizeSino);
  offset = sizeof(float) * recon_info_record->det_xdim *
               recon_info_record->det_ydim // dark
           + sizeof(float) * recon_info_record->det_xdim *
                 recon_info_record->det_ydim // One full white
           + sizeof(float) * recon_info_record->det_xdim *
                 recon_info_record->det_ydim; // Second full white

  // Read first row
  size_t start_offset = offset;
  off_t current_offset = start_offset + sizeof(unsigned short int) *
                                            recon_info_record->det_xdim *
                                            sliceNr; // First row

  pread(fd, readStruct->short_sinogram,
        sizeof(unsigned short int) * recon_info_record->det_xdim,
        current_offset);

  int frameNr;
  for (frameNr = 1; frameNr < recon_info_record->sinogram_ydim; frameNr++) {
    // printf("FrameNr: %d\n",frameNr);fflush(stdout);
    current_offset += sizeof(unsigned short int) * recon_info_record->det_xdim *
                      (recon_info_record->det_ydim); // Skip to next angle
    pread(fd,
          (readStruct->short_sinogram) + recon_info_record->det_xdim * frameNr,
          sizeof(unsigned short int) * recon_info_record->det_xdim,
          current_offset); // One row each at the next subsequent place
  }

  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "short_sinogram_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->short_sinogram, SizeSino, 1, out);
    fclose(out);
  }
  SizeNormSino = sizeof(float) * recon_info_record->sinogram_adjusted_xdim *
                 recon_info_record->theta_list_size;
  readStruct->sizeMatrices += SizeNormSino;
  //~ printf("norm_sino %ld\n",(long)SizeNormSino);
  Normalize(readStruct, recon_info_record);

#pragma omp critical
  {
    if (recon_info_record->debug > 0) {
      char outfn[4096];
      sprintf(outfn, "norm_sino_%s", recon_info_record->DataFileName);
      FILE *out = fopen(outfn, "ab");
      fwrite(readStruct->norm_sino, SizeNormSino, 1, out);
      fclose(out);
    }
  }

  free(readStruct->short_sinogram);
  free(readStruct->white_field_sino);
  free(readStruct->dark_field_sino_ave);

  return 0;
}

void reconCentering(LOCAL_CONFIG_OPTS *information,
                    const GLOBAL_CONFIG_OPTS *recon_info_record, size_t offt,
                    int doLog) {
  int j, k;
  if (doLog == 1)
    LogProj(information->sino_calc_buffer, information->sinogram_adjusted_xdim,
            recon_info_record->sinogram_ydim);
  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "logproj_sino_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->sino_calc_buffer,
           sizeof(float) * information->sinogram_adjusted_xdim *
               recon_info_record->sinogram_ydim,
           1, out);
    fclose(out);
  }
  // ***********************This was not the correct size of shifted_recon. We
  // do it properly now.*******************
  // *********************** Not needed to reset shifted_recon, so we don't do
  // it now.******************************* for( j = 0; j <
  // recon_info_record->sinogram_ydim; j++ ){ for( k = 0; k <
  // information->sinogram_adjusted_xdim; k++ ){ information->shifted_recon[j *
  // information->sinogram_adjusted_xdim+ k] = 0.0f;
  // }
  // }
  // Precompute shift decomposition: the fractional part is constant for all
  // pixels
  int xdim = information->sinogram_adjusted_xdim;
  int shift_int = (int)floor(information->shift);
  float frac = information->shift - (float)shift_int;
  float w0 = 1.0f - frac; // weight for pixel at (k - shift_int)
  float w1 = frac;        // weight for pixel at (k - shift_int - 1)
  for (j = 0; j < recon_info_record->sinogram_ydim; j++) {
    float *src = &information->sino_calc_buffer[j * xdim];
    float *dst = &information->shifted_sinogram[j * xdim];
    for (k = 0; k < xdim; k++) {
      int nkk = k - shift_int;
      float fInterpPixel = 0.0f;
      float fInterpWeight = 0.0f;
      if (nkk >= 0 && nkk < xdim) {
        fInterpPixel += src[nkk] * w0;
        fInterpWeight += w0;
      }
      if (nkk - 1 >= 0 && nkk - 1 < xdim) {
        fInterpPixel += src[nkk - 1] * w1;
        fInterpWeight += w1;
      }
      if (fInterpWeight < 1e-5f)
        dst[k] = 0.0f;
      else
        dst[k] = fInterpPixel / fInterpWeight;
    }
  }
  memcpy(&information->sino_calc_buffer[0], information->shifted_sinogram,
         sizeof(float) * information->sinogram_adjusted_size);
  if (recon_info_record->use_ring_removal) {
    RingCorrectionSingle(&information->sino_calc_buffer[0],
                         recon_info_record->ring_removal_coeff, information,
                         recon_info_record);
  }
  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "shifted_sino_%s", recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->sino_calc_buffer,
           sizeof(float) * information->sinogram_adjusted_xdim *
               recon_info_record->sinogram_ydim,
           1, out);
    fclose(out);
  }
  for (j = 0; j < recon_info_record->sinogram_ydim; j++) {
    memcpy(
        &information->sinograms_boundary_padding
             [offt + j * information->sinogram_adjusted_xdim * 2 +
              information->sinogram_adjusted_xdim / 2],
        &information->sino_calc_buffer[j * information->sinogram_adjusted_xdim],
        sizeof(float) * information->sinogram_adjusted_xdim);
    for (k = 0; k < information->sinogram_adjusted_xdim / 2; k++) {
      information->sinograms_boundary_padding
          [offt + j * information->sinogram_adjusted_xdim * 2 + k] =
          information->sinograms_boundary_padding
              [offt + j * information->sinogram_adjusted_xdim * 2 +
               information->sinogram_adjusted_xdim / 2];
    }
    for (k = 0; k < information->sinogram_adjusted_xdim / 2; k++) {
      information->sinograms_boundary_padding
          [offt + j * information->sinogram_adjusted_xdim * 2 +
           information->sinogram_adjusted_xdim / 2 +
           information->sinogram_adjusted_xdim + k] =
          information->sinograms_boundary_padding
              [offt + j * information->sinogram_adjusted_xdim * 2 +
               information->sinogram_adjusted_xdim / 2 +
               information->sinogram_adjusted_xdim - 1];
    }
  }
}

void getRecons(LOCAL_CONFIG_OPTS *information,
               const GLOBAL_CONFIG_OPTS *recon_info_record,
               gridrecParams *param, size_t offsetRecons) {
  int j, k;
  for (j = 0; j < recon_info_record->reconstruction_ydim; j++) {
    memcpy(&information
                ->recon_calc_buffer[j * recon_info_record->reconstruction_xdim],
           &information->reconstructions_boundary_padding
                [offsetRecons +
                 (j + recon_info_record->reconstruction_xdim / 2) *
                     recon_info_record->reconstruction_xdim * 2 +
                 recon_info_record->reconstruction_xdim / 2],
           sizeof(float) * (recon_info_record->reconstruction_xdim));
  }
  if (recon_info_record->debug == 1) {
    char outfn[4096];
    sprintf(outfn, "recon_calc_buffer_before_shift_%s",
            recon_info_record->DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->recon_calc_buffer,
           sizeof(float) * recon_info_record->reconstruction_xdim *
               recon_info_record->reconstruction_ydim,
           1, out);
    fclose(out);
  }
  // ***********************This was not the correct size of shifted_recon. We
  // do it properly now.******************* for( j = 0; j <
  // recon_info_record->sinogram_ydim; j++ ){ for( k = 0; k <
  // recon_info_record->reconstruction_xdim; k++ ){ information->shifted_recon[j
  // * recon_info_record->reconstruction_xdim + k] = 0.0f;
  // }
  // }
  float *recon_buffer;
  if (recon_info_record->auto_centering) {
    memset(information->shifted_recon, 0,
           sizeof(float) * information->reconstruction_size);
    recon_buffer = &information->recon_calc_buffer[0];
    if (information->shift >= 0) {
      for (j = 0; j < recon_info_record->reconstruction_ydim; j++)
        memcpy(
            &information
                 ->shifted_recon[j * recon_info_record->reconstruction_xdim],
            (void *)&recon_buffer[(j * recon_info_record->reconstruction_xdim) +
                                  (int)round(information->shift)],
            sizeof(float) * (recon_info_record->reconstruction_xdim -
                             (int)round(information->shift)));
    } else {
      for (j = 0; j < recon_info_record->reconstruction_ydim; j++)
        memcpy(
            &information
                 ->shifted_recon[(j * recon_info_record->reconstruction_xdim) +
                                 abs((int)round(information->shift))],
            (void *)&recon_buffer[j * recon_info_record->reconstruction_xdim],
            sizeof(float) * (recon_info_record->reconstruction_xdim -
                             abs((int)round(information->shift))));
    }
    memcpy((void *)recon_buffer, information->shifted_recon,
           sizeof(float) * information->reconstruction_size);
  }
}

int writeRecon(int sliceNr, int slicePos, LOCAL_CONFIG_OPTS *information,
               const GLOBAL_CONFIG_OPTS *recon_info_record, int shiftNr,
               int cleanupNr, int fd) {
  /* sliceNr is the actual slice index from slices_to_process[] — used only
   * for per-file naming when saveReconSeparate==1, since users want the
   * physical slice number in those filenames.
   * slicePos is the slice's position WITHIN slices_to_process[] (0..n_slices-1)
   * — used for the binary-offset calculation so a slice subset still packs
   * tightly into a (cleanup, shift, slice) cube without holes. */
  if (recon_info_record->saveReconSeparate == 1) {
    char outFileName[4096];
    if (recon_info_record->n_cleanup_configs > 1) {
      sprintf(outFileName,
              "%s_cleanup_%03d_slice_%05d_shift_%03d_XDim_%06d_YDim_%06d_float32.bin",
              recon_info_record->ReconFileName, cleanupNr, sliceNr, shiftNr,
              recon_info_record->reconstruction_xdim,
              recon_info_record->reconstruction_xdim);
    } else {
      sprintf(outFileName,
              "%s_slice_%05d_shift_%03d_XDim_%06d_YDim_%06d_float32.bin",
              recon_info_record->ReconFileName, sliceNr, shiftNr,
              recon_info_record->reconstruction_xdim,
              recon_info_record->reconstruction_xdim);
    }
    FILE *out = fopen(outFileName, "wb");
    if (out == NULL) {
      printf("Could not open output file.\n");
      return 1;
    }
    fwrite(information->recon_calc_buffer, sizeof(float),
           information->reconstruction_size, out);
    fclose(out);
  } else {
    /* OutputFileName already opened in fd.
     * Layout when n_cleanup_configs > 1: (cleanup, shift, slicePos, Y, X)
     * Layout when n_cleanup_configs == 1: (shift, slicePos, Y, X)
     * For slicesToProcess=-1 (all slices) slicePos == sliceNr so layout is
     * unchanged from the historical behavior. */
    size_t OffsetHere = sizeof(float) * information->reconstruction_size;
    size_t shiftSliceIdx =
        (size_t)shiftNr * (size_t)recon_info_record->n_slices +
        (size_t)slicePos;
    size_t cleanupStride = (size_t)recon_info_record->n_shifts *
                           (size_t)recon_info_record->n_slices;
    OffsetHere *= (size_t)cleanupNr * cleanupStride + shiftSliceIdx;

    size_t nbytes = sizeof(float) * (size_t)information->reconstruction_size;
    if (recon_info_record->recon_out_ptr != NULL) {
      /* Straight into the caller's array, at the same offset the file would
       * have used -- so the in-memory and on-disk cubes are the same layout. */
      if (OffsetHere + nbytes > recon_info_record->recon_out_bytes) {
        printf("Output buffer too small: need %zu bytes, have %zu.\n",
               OffsetHere + nbytes, recon_info_record->recon_out_bytes);
        return 1;
      }
      memcpy((char *)recon_info_record->recon_out_ptr + OffsetHere,
             information->recon_calc_buffer, nbytes);
    } else {
      int rc = pwrite(fd, information->recon_calc_buffer, nbytes, OffsetHere);
      if (rc < 0) {
        printf("Could not write to output file.\n");
        return 1;
      }
    }
  }
  return 0;
}

int createPlanFile(GLOBAL_CONFIG_OPTS *recon_info_record) {
  int sliceNr = recon_info_record->slices_to_process[0];
  SINO_READ_OPTS readStruct;
  readStruct.norm_sino = (float *)malloc(
      sizeof(float) * recon_info_record->sinogram_adjusted_xdim *
      recon_info_record->theta_list_size);
  recon_info_record->sizeMatrices += sizeof(float) *
                                     recon_info_record->sinogram_adjusted_xdim *
                                     recon_info_record->theta_list_size;
  LOCAL_CONFIG_OPTS information;
  GLOBAL_CONFIG_OPTS cpy = *recon_info_record;
  setSinoSize(&information, &cpy);
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.reconstruction_size;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.sinogram_adjusted_size;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.sinogram_adjusted_size * 2 * 2;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.reconstruction_size * 4 * 2;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.reconstruction_size * 2;
  recon_info_record->sizeMatrices += sizeof(float) *
                                     information.sinogram_adjusted_xdim *
                                     recon_info_record->theta_list_size;
  recon_info_record->sizeMatrices +=
      sizeof(float) * recon_info_record->sinogram_ydim;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.sinogram_adjusted_xdim;
  recon_info_record->sizeMatrices +=
      sizeof(float) * information.sinogram_adjusted_xdim;
  gridrecParams param;
  param.sizeMatrices = 0;
  /* Must be initialised, or the planner's `if (param->deterministic)` branch
   * reads uninitialised stack. This routine also runs in deterministic mode
   * (it is what measures sizeMatrices), so propagate rather than hard-code. */
  param.error = MIDAS_TOMO_OK;
  param.deterministic = recon_info_record->deterministic;
  param.fft_engine = recon_info_record->fft_engine;
  /* The branches below test this for NULL; leaving it uninitialised
   * makes that test read stack garbage. */
  param.wisdom_string = NULL;
  param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2;
  param.theta_list = recon_info_record->theta_list;
  param.filter_type = recon_info_record->filter;
  param.theta_list_size = recon_info_record->theta_list_size;
  setGridRecPSWF(&param);
  initFFTMemoryStructures(&param);
  initGridRec(&param);
  recon_info_record->sizeMatrices += param.sizeMatrices;
  param.sizeMatrices = 0;
  readStruct.sizeMatrices = 0;
  information.shift = recon_info_record->shift_values[0];
  int input_fd = -1;
  if (!recon_info_record->are_sinos && !recon_info_record->use_hdf5) {
    input_fd = open(recon_info_record->DataFileName, O_RDONLY);
  }
  /* This trial read exists to size the matrices, but it is also the FIRST
   * time the input is touched -- so it is where a bad input is detected. The
   * return value used to be discarded, which meant a short or missing input
   * was reported to stderr and then reconstructed anyway, with a zero exit
   * code. */
  int readRC;
  if (recon_info_record->are_sinos) {
    readRC = readSino(sliceNr, &cpy, &readStruct);
  } else {
    readRC = readRaw(sliceNr, &cpy, &readStruct, input_fd);
  }
  if (readRC != 0) {
    if (input_fd != -1)
      close(input_fd);
    free(readStruct.norm_sino);
    return 1;
  }
  if (input_fd != -1) {
    close(input_fd);
  }
  recon_info_record->sizeMatrices += readStruct.sizeMatrices;
  param.sizeMatrices = 0;
  memcpy(information.sino_calc_buffer, readStruct.norm_sino,
         sizeof(float) * information.sinogram_adjusted_xdim *
             recon_info_record->theta_list_size);
  reconCentering(&information, &cpy, 0, recon_info_record->doLogProj);
  // Do the same slice twice
  setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[0],
                         &information.reconstructions_boundary_padding[0],
                         &param);
  setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[0],
                         &information.reconstructions_boundary_padding[0],
                         &param);
  recon_info_record->sizeMatrices += readStruct.sizeMatrices;
  param.sizeMatrices = 0;
  param.setPlan = 1;
  reconstruct(&param);
  recon_info_record->wisdom_string =
      (char *)malloc(sizeof(char) * (strlen(param.wisdom_string) + 1));
  recon_info_record->sizeMatrices +=
      sizeof(char) * (strlen(param.wisdom_string) + 1);
  strcpy(recon_info_record->wisdom_string, param.wisdom_string);
  destroyFFTMemoryStructures(&param);
  return 0;
}
