//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <fcntl.h>
#include <hdf5.h>
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
  float mean, max;
  for (i = 0; i < ydim; i++) {
    max = data[i * xdim];
    for (k = 0; k < xdim; k++) {
      if (data[i * xdim + k] > max)
        max = data[i * xdim + k];
    }
    for (k = 0; k < xdim; k++) {
      if (data[i * xdim + k] <= 0.0)
        data[i * xdim + k] = 1.0;
      data[i * xdim + k] = log(max / data[i * xdim + k]);
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
                          GLOBAL_CONFIG_OPTS recon_info_record) {
  int i, j, m;
  float mean_total;
  float tmp;
  for (m = 0; m < 20; m++) {
    for (i = 0; i < recon_info_record.sinogram_ydim; i++)
      information->mean_vect[i] = 0.0;
    mean_total = 0.0;
    for (i = 0; i < recon_info_record.sinogram_ydim; i++) {
      for (j = 0; j < information->sinogram_adjusted_xdim; j++) {
        information->mean_vect[i] +=
            data[i * information->sinogram_adjusted_xdim + j];
      }
      information->mean_vect[i] /= information->sinogram_adjusted_xdim;
      mean_total += information->mean_vect[i];
    }
    mean_total /= recon_info_record.sinogram_ydim;
    for (i = 0; i < recon_info_record.sinogram_ydim; i++) {
      for (j = 0; j < information->sinogram_adjusted_xdim; j++) {
        if (information->mean_vect[i] != 0.0) {
          data[i * information->sinogram_adjusted_xdim + j] =
              data[i * information->sinogram_adjusted_xdim + j] * mean_total /
              information->mean_vect[i];
        }
      }
    }
    for (i = 0; i < information->sinogram_adjusted_xdim; i++)
      information->mean_sino_line_data[i] = 0.0;
    for (i = 0; i < recon_info_record.sinogram_ydim; i++)
      for (j = 0; j < information->sinogram_adjusted_xdim; j++)
        information->mean_sino_line_data[j] +=
            data[i * information->sinogram_adjusted_xdim + j];
    for (i = 0; i < information->sinogram_adjusted_xdim; i++)
      information->mean_sino_line_data[i] /= recon_info_record.sinogram_ydim;
    for (j = 1; j < information->sinogram_adjusted_xdim - 1; j++) {
      information->low_pass_sino_lines_data[j] =
          (information->mean_sino_line_data[j - 1] +
           information->mean_sino_line_data[j] +
           information->mean_sino_line_data[j + 1]) /
          3.0;
    }
    information->low_pass_sino_lines_data[0] =
        information->mean_sino_line_data[0];
    information
        ->low_pass_sino_lines_data[information->sinogram_adjusted_xdim - 1] =
        information
            ->mean_sino_line_data[information->sinogram_adjusted_xdim - 1];
    for (i = 0; i < recon_info_record.sinogram_ydim; i++) {
      for (j = 0; j < information->sinogram_adjusted_xdim; j++) {
        tmp = information->mean_sino_line_data[j] -
              information->low_pass_sino_lines_data[j];
        if ((data[i * information->sinogram_adjusted_xdim + j] -
             (tmp * ring_coeff)) > 0.0)
          data[i * information->sinogram_adjusted_xdim + j] -=
              (tmp * ring_coeff);
        else
          data[i * information->sinogram_adjusted_xdim + j] = 0.0;
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
// slice. Size of each sino is recon_info_record.sinogram_xdim, output norm_sino
// is information->sinogram_adjusted_xdim (padded)

void Normalize(SINO_READ_OPTS *readStruct,
               GLOBAL_CONFIG_OPTS recon_info_record) {
  int pad_size = recon_info_record.sinogram_adjusted_xdim -
                 recon_info_record.sinogram_xdim,
      front_pad_size = pad_size / 2, back_pad_size = pad_size - front_pad_size;
  //~ printf("Pad sizes %d %d\n",front_pad_size,back_pad_size);
  int frameNr, pxNr, colNr;
  float temp_val, white_temp, factor;
  for (frameNr = 0; frameNr < recon_info_record.sinogram_ydim; frameNr++) {
    factor = (float)frameNr / (float)recon_info_record.theta_list_size;
    for (pxNr = 0; pxNr < recon_info_record.sinogram_adjusted_xdim; pxNr++) {
      if (pxNr < front_pad_size) { // front padding
        colNr = 0;                 // first pixel
      } else if (pxNr >= front_pad_size +
                             recon_info_record.sinogram_xdim) { // back padding
        colNr = recon_info_record.sinogram_xdim - 1;            // last pixel
      } else {
        colNr = pxNr - front_pad_size; // actual pixel
      }
      white_temp =
          (1 - factor) * (float)readStruct->white_field_sino[colNr] +
          (factor) *
              (float)readStruct
                  ->white_field_sino[colNr + recon_info_record.sinogram_xdim];
      // printf("colNr frameNr xdim totSize : %d %d %d
      // %d\n",colNr,frameNr,readStruct->sinogram_adjusted_xdim,recon_info_record.det_xdim*recon_info_record.theta_list_size);fflush(stdout);
      temp_val =
          ((float)readStruct
               ->short_sinogram[colNr + frameNr * recon_info_record.det_xdim] -
           readStruct->dark_field_sino_ave[colNr]) /
          (white_temp - readStruct->dark_field_sino_ave[colNr]);
      readStruct->norm_sino[frameNr * recon_info_record.sinogram_adjusted_xdim +
                            pxNr] = temp_val;
    }
  }
}

void Pad(SINO_READ_OPTS *readStruct,
         GLOBAL_CONFIG_OPTS
             recon_info_record) { // Take the sino directly read (init_sinogram)
                                  // and pad it, return norm_sino.
  int pad_size = recon_info_record.sinogram_adjusted_xdim -
                 recon_info_record.sinogram_xdim,
      front_pad_size = pad_size / 2, back_pad_size = pad_size - front_pad_size;
  int colNr, frameNr;
  for (frameNr = 0; frameNr < recon_info_record.sinogram_ydim; frameNr++) {
    for (colNr = 0; colNr < recon_info_record.sinogram_adjusted_xdim; colNr++) {
      if (colNr < front_pad_size)
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    .sinogram_adjusted_xdim] =
            readStruct->init_sinogram[frameNr * recon_info_record.det_xdim];
      else if (colNr >= front_pad_size + recon_info_record.sinogram_xdim)
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    .sinogram_adjusted_xdim] =
            readStruct->init_sinogram[recon_info_record.sinogram_xdim - 1 +
                                      frameNr * recon_info_record.det_xdim];
      else
        readStruct->norm_sino[colNr + frameNr * recon_info_record
                                                    .sinogram_adjusted_xdim] =
            readStruct
                ->init_sinogram[colNr + frameNr * recon_info_record.det_xdim -
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
  }
  fclose(fileParam);
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
             GLOBAL_CONFIG_OPTS recon_info_record) {
  memset(information->shifted_recon, 0,
         sizeof(float) * information->reconstruction_size);
  memset(information->shifted_sinogram, 0,
         sizeof(float) * information->sinogram_adjusted_size);
  memset(information->sinograms_boundary_padding, 0,
         sizeof(float) * information->sinogram_adjusted_size * 2 *
             2); // Hold two sinos
  memset(information->reconstructions_boundary_padding, 0,
         sizeof(float) * information->reconstruction_size * 4 *
             2); // Hold two recons
  memset(information->recon_calc_buffer, 0,
         sizeof(float) * information->reconstruction_size * 2);
  memset(information->sino_calc_buffer, 0,
         sizeof(float) * information->sinogram_adjusted_xdim *
             recon_info_record.theta_list_size);
  memset(information->mean_vect, 0,
         sizeof(float) * recon_info_record.sinogram_ydim);
  memset(information->mean_sino_line_data, 0,
         sizeof(float) * information->sinogram_adjusted_xdim);
  memset(information->low_pass_sino_lines_data, 0,
         sizeof(float) * information->sinogram_adjusted_xdim);
}

void setSinoSize(LOCAL_CONFIG_OPTS *information,
                 GLOBAL_CONFIG_OPTS recon_info_record) {
  information->sinogram_adjusted_xdim =
      recon_info_record.sinogram_adjusted_xdim;
  information->sinogram_adjusted_size =
      recon_info_record.sinogram_adjusted_size;
  information->reconstruction_size = recon_info_record.reconstruction_size;
  //~ printf("shifted_recon: %ld\n",(long)(sizeof
  //(float)*information->reconstruction_size)); ~ printf("shifted_sinogram
  //%ld\n",(long)(sizeof (float)*information->sinogram_adjusted_size)); ~
  //printf("sinograms_boundary_padding
  //%ld\n",(long)(sizeof(float)*information->sinogram_adjusted_size*2)); ~
  //printf("reconstructions_boundary_padding
  //%ld\n",(long)(sizeof(float)*information->reconstruction_size*4)); ~
  //printf("recon_calc_buffer
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
                      recon_info_record.theta_list_size);
  //~ printf("mean_vect %ld\n",(long)(sizeof
  //(float)*recon_info_record.sinogram_ydim)); ~ printf("mean_sino_line_data
  //%ld\n",(long)(sizeof (float)*information->sinogram_adjusted_xdim)); ~
  //printf("low_pass_sino_lines_data %ld\n",(long)(sizeof(float)
  //*information->sinogram_adjusted_xdim));
  information->mean_vect =
      (float *)malloc(sizeof(float) * recon_info_record.sinogram_ydim);
  information->mean_sino_line_data =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_xdim);
  information->low_pass_sino_lines_data =
      (float *)malloc(sizeof(float) * information->sinogram_adjusted_xdim);
}

int readSino(int sliceNr, GLOBAL_CONFIG_OPTS recon_info_record,
             SINO_READ_OPTS *readStruct) {
  FILE *dataFile;
#pragma omp critical
  {
    dataFile = fopen(recon_info_record.DataFileName, "rb");
  }
  if (dataFile == NULL) {
    printf("SliceNr: %d, Could not read datafile: %s.\n", sliceNr,
           recon_info_record.DataFileName);
    return 1;
  }
  size_t offset = sizeof(float) * sliceNr * recon_info_record.det_xdim *
                  recon_info_record.theta_list_size;
  size_t SizeSino = sizeof(float) * recon_info_record.det_xdim *
                    recon_info_record.theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  readStruct->sizeMatrices +=
      (sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
       recon_info_record.theta_list_size);
  //~ printf("init_sinogram %ld\n",(long)SizeSino);
  //~ printf("norm_sino
  //%ld\n",(long)(sizeof(float)*recon_info_record.sinogram_adjusted_xdim*recon_info_record.theta_list_size));
  readStruct->init_sinogram = (float *)malloc(SizeSino);
#pragma omp critical
  {
    fseek(dataFile, offset, SEEK_SET);
    fread(readStruct->init_sinogram, SizeSino, 1, dataFile);
  }
#pragma omp critical
  {
    fclose(dataFile);
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "init_sinogram_%s", recon_info_record.DataFileName);
    printf("%s", outfn);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->init_sinogram, SizeSino, 1, out);
    fclose(out);
  }
  Pad(readStruct, recon_info_record);
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "norm_sino_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->norm_sino,
           sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
               recon_info_record.theta_list_size,
           1, out);
    fclose(out);
  }
  free(readStruct->init_sinogram);
  return 0;
}

int readRawHDF5(int sliceNr, GLOBAL_CONFIG_OPTS recon_info_record,
                SINO_READ_OPTS *readStruct) {
  hid_t file_id;
#pragma omp critical
  {
    file_id =
        H5Fopen(recon_info_record.HDF5FileName, H5F_ACC_RDONLY, H5P_DEFAULT);
  }
  if (file_id < 0) {
    printf("SliceNr: %d, Could not open HDF5 file: %s.\n", sliceNr,
           recon_info_record.HDF5FileName);
    return 1;
  }

  size_t SizeDark, SizeWhite, SizeSino, SizeNormSino;
  hsize_t dims[3];

  // 1. Read Dark
  SizeDark = sizeof(float) * recon_info_record.det_xdim;
  readStruct->sizeMatrices += SizeDark;
  readStruct->dark_field_sino_ave = (float *)malloc(SizeDark);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record.DarkDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: Dark Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record.DarkDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);
      int ndims = H5Sget_simple_extent_ndims(dataspace_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      hsize_t offset[2] = {sliceNr, 0};
      hsize_t count[2] = {1, recon_info_record.det_xdim};

      // Handle 3D Dark if necessary (e.g. [1, Y, X]) -> For now assuming [Y, X]
      // per plan If it is 3D, we assume [1, Y, X], so reading row sliceNr is
      // [0, sliceNr, :]
      if (ndims == 3) {
        hsize_t offset3[3] = {0, sliceNr, 0};
        hsize_t count3[3] = {1, 1, recon_info_record.det_xdim};
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL, count3,
                            NULL);

        hid_t memspace_id =
            H5Screate_simple(1, &count3[2], NULL); // 1D {det_xdim}
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                H5P_DEFAULT, readStruct->dark_field_sino_ave);
        H5Sclose(memspace_id);
      } else {
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
             recon_info_record.DarkDatasetName);
    }
  }

  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "dark_field_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->dark_field_sino_ave, SizeDark, 1, out);
    fclose(out);
  }

  // 2. Read Whites
  SizeWhite = sizeof(float) * recon_info_record.det_xdim * 2;
  readStruct->sizeMatrices += SizeWhite;
  readStruct->white_field_sino = (float *)malloc(SizeWhite);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record.WhiteDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: White Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record.WhiteDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);
      int ndims = H5Sget_simple_extent_ndims(dataspace_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      // Assume [nFrames, Y, X]. If nFrames >= 2, read frame 0 and frame 1.
      // If nFrames < 2, read frame 0 and duplicate.
      // Or if 2D [Y, X], read row sliceNr and duplicate.

      if (ndims == 3) {
        // Read Frame 0
        hsize_t offset3[3] = {0, sliceNr, 0};
        hsize_t count3[3] = {1, 1, recon_info_record.det_xdim};
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset3, NULL, count3,
                            NULL);
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
                  (readStruct->white_field_sino) + recon_info_record.det_xdim);
        } else {
          memcpy((readStruct->white_field_sino) + recon_info_record.det_xdim,
                 readStruct->white_field_sino,
                 sizeof(float) * recon_info_record.det_xdim);
        }
        H5Sclose(memspace_id);

      } else {
        // 2D [Y, X]
        hsize_t offset[2] = {sliceNr, 0};
        hsize_t count[2] = {1, recon_info_record.det_xdim};
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
                            NULL);
        hid_t memspace_id = H5Screate_simple(1, &count[1], NULL);
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
                H5P_DEFAULT, readStruct->white_field_sino);
        // Duplicate
        memcpy((readStruct->white_field_sino) + recon_info_record.det_xdim,
               readStruct->white_field_sino,
               sizeof(float) * recon_info_record.det_xdim);
        H5Sclose(memspace_id);
      }
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    } else {
      printf("Could not open White dataset: %s\n",
             recon_info_record.WhiteDatasetName);
    }
  }

  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "whites_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->white_field_sino, SizeWhite, 1, out);
    fclose(out);
  }

  // 3. Read Images (Sinogram)
  SizeSino = sizeof(unsigned short int) * recon_info_record.det_xdim *
             recon_info_record.theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  readStruct->short_sinogram = (unsigned short int *)malloc(SizeSino);

#pragma omp critical
  {
    hid_t dataset_id =
        H5Dopen2(file_id, recon_info_record.ImageDatasetName, H5P_DEFAULT);
    if (dataset_id >= 0) {
      // Type Check
      hid_t dtype = H5Dget_type(dataset_id);
      H5T_class_t t_class = H5Tget_class(dtype);
      if (t_class != H5T_INTEGER && t_class != H5T_FLOAT) {
        printf("Warning: Image Dataset %s is not Integer or Float. HDF5 "
               "conversion may fail.\n",
               recon_info_record.ImageDatasetName);
      }
      H5Tclose(dtype);

      hid_t dataspace_id = H5Dget_space(dataset_id);

      // Expected [nAngles, nY, nX]
      // We want [:, sliceNr, :]
      hsize_t offset3[3] = {0, sliceNr, 0};
      hsize_t count3[3] = {recon_info_record.theta_list_size, 1,
                           recon_info_record.det_xdim};

      // Verify dimensions
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
      if (dims[0] < recon_info_record.theta_list_size) {
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
             recon_info_record.ImageDatasetName);
    }

    H5Fclose(file_id);
  }

  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "short_sinogram_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->short_sinogram, SizeSino, 1, out);
    fclose(out);
  }

  SizeNormSino = sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
                 recon_info_record.theta_list_size;
  readStruct->sizeMatrices += SizeNormSino;
  Normalize(readStruct, recon_info_record);

#pragma omp critical
  {
    if (recon_info_record.debug > 0) {
      char outfn[4096];
      sprintf(outfn, "norm_sino_%s", recon_info_record.DataFileName);
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

int readRaw(int sliceNr, GLOBAL_CONFIG_OPTS recon_info_record,
            SINO_READ_OPTS *readStruct) {
  if (recon_info_record.use_hdf5) {
    return readRawHDF5(sliceNr, recon_info_record, readStruct);
  }

  FILE *dataFile;
#pragma omp critical
  {
    dataFile = fopen(recon_info_record.DataFileName, "rb");
  }
  if (dataFile == NULL) {
    printf("SliceNr: %d, Could not read datafile: %s.\n", sliceNr,
           recon_info_record.DataFileName);
    return 1;
  }
  size_t offset, SizeDark, SizeWhite, SizeSino, SizeNormSino;
  // Dark
  SizeDark = sizeof(float) * recon_info_record.det_xdim;
  readStruct->sizeMatrices += SizeDark;
  //~ printf("dark_field_sino_ave %ld\n",(long)SizeDark);
  readStruct->dark_field_sino_ave = (float *)malloc(SizeDark);
  offset = sizeof(float) * sliceNr * recon_info_record.det_xdim;
#pragma omp critical
  {
    fseek(dataFile, offset, SEEK_SET);
    fread(readStruct->dark_field_sino_ave, SizeDark, 1, dataFile);
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "dark_field_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->dark_field_sino_ave, SizeDark, 1, out);
    fclose(out);
  }
  // 2 Whites
  SizeWhite = sizeof(float) * recon_info_record.det_xdim * 2;
  readStruct->sizeMatrices += SizeWhite;
  //~ printf("white_field_sino %ld\n",(long)SizeWhite);
  readStruct->white_field_sino = (float *)malloc(SizeWhite);
  offset =
      sizeof(float) * recon_info_record.det_xdim *
          recon_info_record.det_ydim                          // dark
      + sizeof(float) * recon_info_record.det_xdim * sliceNr; // Partial white
#pragma omp critical
  {
    fseek(dataFile, offset, SEEK_SET);
    fread(readStruct->white_field_sino, SizeWhite / 2, 1, dataFile); // One Row
  }
  offset =
      sizeof(float) * recon_info_record.det_xdim *
          recon_info_record.det_ydim // dark
      + sizeof(float) * recon_info_record.det_xdim *
            recon_info_record.det_ydim                        // One full white
      + sizeof(float) * recon_info_record.det_xdim * sliceNr; // Partial white
#pragma omp critical
  {
    fseek(dataFile, offset, SEEK_SET);
    fread((readStruct->white_field_sino) + recon_info_record.det_xdim,
          SizeWhite / 2, 1, dataFile); // Second Row
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "whites_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->white_field_sino, SizeWhite, 1, out);
    fclose(out);
  }
  // Sino start
  SizeSino = sizeof(unsigned short int) * recon_info_record.det_xdim *
             recon_info_record.theta_list_size;
  readStruct->sizeMatrices += SizeSino;
  //~ printf("short_sinogram %ld\n",(long)SizeSino);
  readStruct->short_sinogram = (unsigned short int *)malloc(SizeSino);
  offset = sizeof(float) * recon_info_record.det_xdim *
               recon_info_record.det_ydim // dark
           + sizeof(float) * recon_info_record.det_xdim *
                 recon_info_record.det_ydim // One full white
           + sizeof(float) * recon_info_record.det_xdim *
                 recon_info_record.det_ydim; // Second full white
#pragma omp critical
  {
    fseek(dataFile, offset, SEEK_SET);
    // We are now at the beginning of the image data.
    offset = sizeof(unsigned short int) * recon_info_record.det_xdim * sliceNr;
    fseek(dataFile, offset, SEEK_CUR);
    fread(readStruct->short_sinogram,
          sizeof(unsigned short int) * recon_info_record.det_xdim, 1,
          dataFile); // One row
  }
  int frameNr;
  for (frameNr = 1; frameNr < recon_info_record.sinogram_ydim; frameNr++) {
    // printf("FrameNr: %d\n",frameNr);fflush(stdout);
    offset = sizeof(unsigned short int) * recon_info_record.det_xdim *
             (recon_info_record.det_ydim - 1);
#pragma omp critical
    {
      fseek(dataFile, offset, SEEK_CUR);
      fread((readStruct->short_sinogram) + recon_info_record.det_xdim * frameNr,
            sizeof(unsigned short int) * recon_info_record.det_xdim, 1,
            dataFile); // One row each at the next subsequent place
    }
  }
#pragma omp critical
  {
    fclose(dataFile);
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "short_sinogram_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(readStruct->short_sinogram, SizeSino, 1, out);
    fclose(out);
  }
  SizeNormSino = sizeof(float) * recon_info_record.sinogram_adjusted_xdim *
                 recon_info_record.theta_list_size;
  readStruct->sizeMatrices += SizeNormSino;
  //~ printf("norm_sino %ld\n",(long)SizeNormSino);
  Normalize(readStruct, recon_info_record);
#pragma omp critical
  {
    if (recon_info_record.debug > 0) {
      char outfn[4096];
      sprintf(outfn, "norm_sino_%s", recon_info_record.DataFileName);
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
                    GLOBAL_CONFIG_OPTS recon_info_record, size_t offt,
                    int doLog) {
  int j, k;
  if (doLog == 1)
    LogProj(information->sino_calc_buffer, information->sinogram_adjusted_xdim,
            recon_info_record.sinogram_ydim);
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "logproj_sino_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->sino_calc_buffer,
           sizeof(float) * information->sinogram_adjusted_xdim *
               recon_info_record.sinogram_ydim,
           1, out);
    fclose(out);
  }
  // ***********************This was not the correct size of shifted_recon. We
  // do it properly now.*******************
  // *********************** Not needed to reset shifted_recon, so we don't do
  // it now.******************************* for( j = 0; j <
  // recon_info_record.sinogram_ydim; j++ ){ for( k = 0; k <
  // information->sinogram_adjusted_xdim; k++ ){ information->shifted_recon[j *
  // information->sinogram_adjusted_xdim+ k] = 0.0f;
  // }
  // }
  for (j = 0; j < recon_info_record.sinogram_ydim; j++) {
    for (k = 0; k < information->sinogram_adjusted_xdim; k++) {
      float kk = k - information->shift;
      int nkk = (int)floor(kk);
      float fInterpPixel = 0.0f;
      float fInterpWeight = 0.0f;
      if (nkk >= 0 && nkk < information->sinogram_adjusted_xdim) {
        fInterpPixel +=
            information
                ->sino_calc_buffer[j * information->sinogram_adjusted_xdim +
                                   nkk] *
            (nkk + 1 - kk);
        fInterpWeight = nkk + 1 - kk;
      }
      if (nkk + 1 >= 0 && nkk + 1 < information->sinogram_adjusted_xdim) {
        fInterpPixel +=
            information
                ->sino_calc_buffer[j * information->sinogram_adjusted_xdim +
                                   nkk + 1] *
            (kk - nkk);
        fInterpWeight += kk - nkk;
      }
      if (fInterpWeight < 1e-5)
        fInterpPixel = 0.0f;
      else
        fInterpPixel /= fInterpWeight;
      information
          ->shifted_sinogram[j * information->sinogram_adjusted_xdim + k] =
          fInterpPixel;
    }
  }
  memcpy(&information->sino_calc_buffer[0], information->shifted_sinogram,
         sizeof(float) * information->sinogram_adjusted_size);
  if (recon_info_record.use_ring_removal) {
    RingCorrectionSingle(&information->sino_calc_buffer[0],
                         recon_info_record.ring_removal_coeff, information,
                         recon_info_record);
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "shifted_sino_%s", recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->sino_calc_buffer,
           sizeof(float) * information->sinogram_adjusted_xdim *
               recon_info_record.sinogram_ydim,
           1, out);
    fclose(out);
  }
  for (j = 0; j < recon_info_record.sinogram_ydim; j++) {
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
               GLOBAL_CONFIG_OPTS recon_info_record, gridrecParams *param,
               size_t offsetRecons) {
  int j, k;
  for (j = 0; j < recon_info_record.reconstruction_ydim; j++) {
    if (information->shift >= 0) {
      memcpy(
          &information
               ->recon_calc_buffer[j * recon_info_record.reconstruction_xdim],
          &information->reconstructions_boundary_padding
               [offsetRecons +
                (j + recon_info_record.reconstruction_xdim / 2) *
                    recon_info_record.reconstruction_xdim * 2 +
                recon_info_record.reconstruction_xdim / 2],
          sizeof(float) * (recon_info_record.reconstruction_xdim));
    } else {
      memcpy(
          &information
               ->recon_calc_buffer[j * recon_info_record.reconstruction_xdim],
          &information->reconstructions_boundary_padding
               [offsetRecons +
                (j + recon_info_record.reconstruction_xdim / 2) *
                    recon_info_record.reconstruction_xdim * 2 +
                recon_info_record.reconstruction_xdim / 2],
          sizeof(float) * (recon_info_record.reconstruction_xdim));
    }
  }
  if (recon_info_record.debug == 1) {
    char outfn[4096];
    sprintf(outfn, "recon_calc_buffer_before_shift_%s",
            recon_info_record.DataFileName);
    FILE *out = fopen(outfn, "wb");
    fwrite(information->recon_calc_buffer,
           sizeof(float) * recon_info_record.reconstruction_xdim *
               recon_info_record.reconstruction_ydim,
           1, out);
    fclose(out);
  }
  // ***********************This was not the correct size of shifted_recon. We
  // do it properly now.******************* for( j = 0; j <
  // recon_info_record.sinogram_ydim; j++ ){ for( k = 0; k <
  // recon_info_record.reconstruction_xdim; k++ ){ information->shifted_recon[j
  // * recon_info_record.reconstruction_xdim + k] = 0.0f;
  // }
  // }
  float *recon_buffer;
  if (recon_info_record.auto_centering) {
    memset(information->shifted_recon, 0,
           sizeof(float) * information->reconstruction_size);
    recon_buffer = &information->recon_calc_buffer[0];
    if (information->shift >= 0) {
      for (j = 0; j < recon_info_record.reconstruction_ydim; j++)
        memcpy(
            &information
                 ->shifted_recon[j * recon_info_record.reconstruction_xdim],
            (void *)&recon_buffer[(j * recon_info_record.reconstruction_xdim) +
                                  (int)round(information->shift)],
            sizeof(float) * (recon_info_record.reconstruction_xdim -
                             (int)round(information->shift)));
    } else {
      for (j = 0; j < recon_info_record.reconstruction_ydim; j++)
        memcpy(
            &information
                 ->shifted_recon[(j * recon_info_record.reconstruction_xdim) +
                                 abs((int)round(information->shift))],
            (void *)&recon_buffer[j * recon_info_record.reconstruction_xdim],
            sizeof(float) * (recon_info_record.reconstruction_xdim -
                             abs((int)round(information->shift))));
    }
    memcpy((void *)recon_buffer, information->shifted_recon,
           sizeof(float) * information->reconstruction_size);
  }
}

int writeRecon(int sliceNr, LOCAL_CONFIG_OPTS *information,
               GLOBAL_CONFIG_OPTS recon_info_record, int shiftNr) {
  // The results are in information.recon_calc_buffer
  // Output file: float with reconstruction_xdim*reconstruction_xdim size
  if (recon_info_record.saveReconSeparate == 1) {
    // OutputFileName:
    // {recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_reconstruction_xdim_float32.bin
    char outFileName[4096];
    if (information->shift > -0.0001) {
      sprintf(outFileName, "%s_%05d_%03d_p%06.1f_%d_%d_float32.bin",
              recon_info_record.ReconFileName, sliceNr, shiftNr,
              information->shift, recon_info_record.reconstruction_xdim,
              recon_info_record.reconstruction_xdim);
    } else {
      sprintf(outFileName, "%s_%05d_%03d_m%06.1f_%d_%d_float32.bin",
              recon_info_record.ReconFileName, sliceNr, shiftNr,
              -information->shift, recon_info_record.reconstruction_xdim,
              recon_info_record.reconstruction_xdim);
    }
    FILE *outfile;
#pragma omp critical
    {
      // printf("Saving output to : %s.\n",outFileName);
      outfile = fopen(outFileName, "wb");
    }
    if (outfile == NULL) {
      printf("We could not open the file for writing %s.\n", outFileName);
      return 1;
    }
#pragma omp critical
    {
      fwrite(information->recon_calc_buffer,
             sizeof(float) * information->reconstruction_size, 1, outfile);
      fclose(outfile);
    }
  } else {
    // OutputFileName:
    // {recon_info_record.ReconFileName}_NrSlices_05d_NrShifts_03d_XDim_06d_YDim_06d_float32.bin
    // How to save: For each shiftNr: sliceNr
    char outFileName[4096];
    sprintf(outFileName,
            "%s_NrShifts_%03d_NrSlices_%05d_XDim_%06d_YDim_%06d_float32.bin",
            recon_info_record.ReconFileName, recon_info_record.n_shifts,
            recon_info_record.n_slices, recon_info_record.reconstruction_xdim,
            recon_info_record.reconstruction_xdim);
#pragma omp critical
    {
      int result = open(outFileName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
      if (result <= 0) {
        printf("Could not open output file.\n");
      }
      size_t OffsetHere = sizeof(float) * information->reconstruction_size;
      OffsetHere *= shiftNr * (recon_info_record.n_slices) + sliceNr;
      int rc =
          pwrite(result, information->recon_calc_buffer,
                 sizeof(float) * information->reconstruction_size, OffsetHere);
      if (rc < 0)
        printf("Could not write to output file.\n");
      close(result);
    }
  }
  return 0;
}

void createPlanFile(GLOBAL_CONFIG_OPTS *recon_info_record) {
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
  setSinoSize(&information, cpy);
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
  if (recon_info_record->are_sinos) {
    readSino(sliceNr, cpy, &readStruct);
  } else {
    readRaw(sliceNr, cpy, &readStruct);
  }
  recon_info_record->sizeMatrices += readStruct.sizeMatrices;
  param.sizeMatrices = 0;
  memcpy(information.sino_calc_buffer, readStruct.norm_sino,
         sizeof(float) * information.sinogram_adjusted_xdim *
             recon_info_record->theta_list_size);
  reconCentering(&information, cpy, 0, recon_info_record->doLogProj);
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
}
