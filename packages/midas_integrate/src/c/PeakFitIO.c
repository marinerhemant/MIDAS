//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFitIO.c — Shared HDF5 output for peak fitting results
//

#include "PeakFitIO.h"
#include "PeakFit.h"
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =========================================================================
// Buffer management
// =========================================================================

void pfio_init_buffer(PeakH5Buffer *buf, int initial_capacity) {
  buf->capacity = (initial_capacity > 0) ? initial_capacity : 1024;
  buf->count = 0;
  buf->rows = (PeakH5Row *)malloc(buf->capacity * sizeof(PeakH5Row));
}

void pfio_append_row(PeakH5Buffer *buf, const PeakH5Row *row) {
  if (buf->count >= buf->capacity) {
    buf->capacity *= 2;
    buf->rows =
        (PeakH5Row *)realloc(buf->rows, buf->capacity * sizeof(PeakH5Row));
  }
  buf->rows[buf->count++] = *row;
}

void pfio_free_buffer(PeakH5Buffer *buf) {
  if (buf->rows) {
    free(buf->rows);
    buf->rows = NULL;
  }
  buf->count = 0;
  buf->capacity = 0;
}

// =========================================================================
// Convert GSAS-II fit output to HDF5 row
// =========================================================================

PeakH5Row pfio_make_row(int frame_idx, int eta_idx, int peak_nr, double eta_deg,
                        const double *gsas_params, double px_um, double Lsd_um,
                        double wavelength_A) {
  PeakH5Row row;
  row.frame_idx = frame_idx;
  row.eta_idx = eta_idx;
  row.peak_nr = peak_nr;
  row.eta_deg = eta_deg;

  double center_px = gsas_params[1]; // center in pixel units
  row.area = gsas_params[0];
  row.sig = gsas_params[2];      // centideg²
  row.gam = gsas_params[3];      // centideg
  row.FWHM_deg = gsas_params[4]; // degrees (from TCH)
  row.eta_mix = gsas_params[5];  // mixing parameter
  row.chi_sq = gsas_params[6];   // chi²/dof

  // Convert center_px to 2theta degrees
  double r_um = center_px * px_um;
  row.center_2theta = atan(r_um / Lsd_um) * 180.0 / M_PI;

  // d-spacing via Bragg's law: d = λ / (2 sin(θ))
  double theta_rad = row.center_2theta * M_PI / 360.0; // θ = 2θ/2
  if (theta_rad > 1e-10) {
    row.d_spacing_A = wavelength_A / (2.0 * sin(theta_rad));
  } else {
    row.d_spacing_A = 0.0;
  }

  return row;
}

// =========================================================================
// Helper: create a 1D dataset of int32
// =========================================================================
static void write_int_dataset(hid_t group, const char *name, const int *data,
                              int n) {
  hsize_t dims = (hsize_t)n;
  hid_t space = H5Screate_simple(1, &dims, NULL);
  hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_INT, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

// =========================================================================
// Helper: create a 1D dataset of float64
// =========================================================================
static void write_double_dataset(hid_t group, const char *name,
                                 const double *data, int n) {
  hsize_t dims = (hsize_t)n;
  hid_t space = H5Screate_simple(1, &dims, NULL);
  hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

// =========================================================================
// Write _caked_peaks.h5 in the Python-compatible format
// =========================================================================
void pfio_write_peaks_h5(const char *filename, const PeakH5Buffer *buf,
                         const double *tth_axis, int nRBins,
                         const double *eta_axis, int nEtaBins, int nFrames,
                         const char *source_name) {
  int n = buf->count;

  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    fprintf(stderr, "PeakFitIO: failed to create %s\n", filename);
    return;
  }

  // /metadata group
  hid_t meta =
      H5Gcreate2(file_id, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // zarr_source attribute
  {
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, strlen(source_name) + 1);
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(meta, "zarr_source", atype, aspace, H5P_DEFAULT,
                            H5P_DEFAULT);
    H5Awrite(attr, atype, source_name);
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
  }

  // tth_axis dataset
  write_double_dataset(meta, "tth_axis", tth_axis, nRBins);

  // eta_axis dataset
  write_double_dataset(meta, "eta_axis", eta_axis, nEtaBins);

  // frame_keys dataset (strings "0", "1", "2", ...)
  {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, H5T_VARIABLE);
    hsize_t dims = (hsize_t)nFrames;
    hid_t space = H5Screate_simple(1, &dims, NULL);
    hid_t dset = H5Dcreate2(meta, "frame_keys", str_type, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
    char **frame_strs = (char **)malloc(nFrames * sizeof(char *));
    for (int i = 0; i < nFrames; i++) {
      frame_strs[i] = (char *)malloc(32);
      sprintf(frame_strs[i], "%d", i);
    }
    H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, frame_strs);
    for (int i = 0; i < nFrames; i++)
      free(frame_strs[i]);
    free(frame_strs);
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(str_type);
  }

  H5Gclose(meta);

  if (n == 0) {
    H5Fclose(file_id);
    printf("PeakFitIO: no peaks to write to %s\n", filename);
    return;
  }

  // /peaks group
  hid_t peaks =
      H5Gcreate2(file_id, "/peaks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Extract column arrays
  int *frame_idx_arr = (int *)malloc(n * sizeof(int));
  int *eta_idx_arr = (int *)malloc(n * sizeof(int));
  int *peak_nr_arr = (int *)malloc(n * sizeof(int));
  double *eta_deg_arr = (double *)malloc(n * sizeof(double));
  double *center_arr = (double *)malloc(n * sizeof(double));
  double *area_arr = (double *)malloc(n * sizeof(double));
  double *sig_arr = (double *)malloc(n * sizeof(double));
  double *gam_arr = (double *)malloc(n * sizeof(double));
  double *fwhm_arr = (double *)malloc(n * sizeof(double));
  double *eta_mix_arr = (double *)malloc(n * sizeof(double));
  double *d_arr = (double *)malloc(n * sizeof(double));
  double *chi_arr = (double *)malloc(n * sizeof(double));

  for (int i = 0; i < n; i++) {
    const PeakH5Row *r = &buf->rows[i];
    frame_idx_arr[i] = r->frame_idx;
    eta_idx_arr[i] = r->eta_idx;
    peak_nr_arr[i] = r->peak_nr;
    eta_deg_arr[i] = r->eta_deg;
    center_arr[i] = r->center_2theta;
    area_arr[i] = r->area;
    sig_arr[i] = r->sig;
    gam_arr[i] = r->gam;
    fwhm_arr[i] = r->FWHM_deg;
    eta_mix_arr[i] = r->eta_mix;
    d_arr[i] = r->d_spacing_A;
    chi_arr[i] = r->chi_sq;
  }

  write_int_dataset(peaks, "frame_idx", frame_idx_arr, n);
  write_int_dataset(peaks, "eta_idx", eta_idx_arr, n);
  write_int_dataset(peaks, "peak_nr", peak_nr_arr, n);
  write_double_dataset(peaks, "eta_deg", eta_deg_arr, n);
  write_double_dataset(peaks, "center_2theta", center_arr, n);
  write_double_dataset(peaks, "area", area_arr, n);
  write_double_dataset(peaks, "sig", sig_arr, n);
  write_double_dataset(peaks, "gam", gam_arr, n);
  write_double_dataset(peaks, "FWHM_deg", fwhm_arr, n);
  write_double_dataset(peaks, "eta_mix", eta_mix_arr, n);
  write_double_dataset(peaks, "d_spacing_A", d_arr, n);
  write_double_dataset(peaks, "chi_sq", chi_arr, n);

  // frame_key as variable-length string (matching Python)
  {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, H5T_VARIABLE);
    hsize_t dims = (hsize_t)n;
    hid_t space = H5Screate_simple(1, &dims, NULL);
    hid_t dset = H5Dcreate2(peaks, "frame_key", str_type, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
    char **fk_strs = (char **)malloc(n * sizeof(char *));
    for (int i = 0; i < n; i++) {
      fk_strs[i] = (char *)malloc(32);
      sprintf(fk_strs[i], "%d", frame_idx_arr[i]);
    }
    H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, fk_strs);
    for (int i = 0; i < n; i++)
      free(fk_strs[i]);
    free(fk_strs);
    H5Dclose(dset);
    H5Sclose(space);
    H5Tclose(str_type);
  }

  // Cleanup
  free(frame_idx_arr);
  free(eta_idx_arr);
  free(peak_nr_arr);
  free(eta_deg_arr);
  free(center_arr);
  free(area_arr);
  free(sig_arr);
  free(gam_arr);
  free(fwhm_arr);
  free(eta_mix_arr);
  free(d_arr);
  free(chi_arr);

  H5Gclose(peaks);
  H5Fclose(file_id);

  printf("PeakFitIO: wrote %d peaks to %s\n", n, filename);
}
