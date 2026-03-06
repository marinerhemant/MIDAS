//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// PeakFitIO.h — Shared HDF5 output for peak fitting results
//
// Provides a growable buffer for collecting per-peak GSAS-II fit results
// during integration, and a function to write them as an HDF5 file
// compatible with plot_caked_peaks.py.
//
// THREAD SAFETY: This module uses the single-threaded HDF5 library.
//   - pfio_append_row() is NOT thread-safe: call from one thread only,
//     or protect with a mutex.
//   - pfio_write_peaks_h5() must be called AFTER all parallel work is done.
//   - In IntegratorZarrOMP: all pfio calls are in sequential code (safe).
//   - In IntegratorFitPeaksGPUStream: pfio calls must be serialized
//     (use the writer thread or call after pthread_join).

#ifndef PEAKFITIO_H
#define PEAKFITIO_H

#ifdef __cplusplus
extern "C" {
#endif

// Row structure matching Python _caked_peaks.h5 schema
typedef struct {
  int frame_idx;
  int eta_idx;
  int peak_nr;
  double eta_deg;
  double center_2theta;
  double area;
  double sig; // centideg²
  double gam; // centideg
  double FWHM_deg;
  double eta_mix;
  double d_spacing_A;
  double chi_sq;
} PeakH5Row;

// Growable buffer for collecting peaks during fitting
typedef struct {
  PeakH5Row *rows;
  int count;
  int capacity;
} PeakH5Buffer;

// Initialize buffer with given initial capacity.
void pfio_init_buffer(PeakH5Buffer *buf, int initial_capacity);

// Append a row to the buffer. Grows automatically.
void pfio_append_row(PeakH5Buffer *buf, const PeakH5Row *row);

// Free the buffer.
void pfio_free_buffer(PeakH5Buffer *buf);

// Write _caked_peaks.h5 matching the Python format.
//
// tth_axis[nRBins] - 2theta axis in degrees
// eta_axis[nEtaBins] - eta bin centers in degrees
// nFrames - number of frames processed
// source_name - data source filename
void pfio_write_peaks_h5(const char *filename, const PeakH5Buffer *buf,
                         const double *tth_axis, int nRBins,
                         const double *eta_axis, int nEtaBins, int nFrames,
                         const char *source_name);

// Convert GSAS-II fit output array to PeakH5Row.
// gsas_params points to PF_PARAMS_PER_PEAK doubles:
//   [0]=area, [1]=center, [2]=sig, [3]=gam, [4]=FWHM, [5]=eta, [6]=chi_sq
// center_px and px/Lsd are used to compute 2theta and d-spacing.
PeakH5Row pfio_make_row(int frame_idx, int eta_idx, int peak_nr, double eta_deg,
                        const double *gsas_params, double px_um, double Lsd_um,
                        double wavelength_A);

#ifdef __cplusplus
}
#endif

#endif // PEAKFITIO_H
