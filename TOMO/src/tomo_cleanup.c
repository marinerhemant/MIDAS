//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// Stripe artifact removal for tomographic sinograms.
// Based on algorithms described in:
//   Nghia T. Vo, Robert C. Atwood, and Michael Drakopoulos,
//   "Superior techniques for eliminating ring artifacts in X-ray
//   micro-tomography," Optics Express 26(22), 28396-28412 (2018).
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// qsort comparator: ascending float
// ============================================================================
static int cmp_float_asc(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  if (fa < fb)
    return -1;
  if (fa > fb)
    return 1;
  return 0;
}

// ============================================================================
// qsort comparator: descending float
// ============================================================================
static int cmp_float_desc(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  if (fa > fb)
    return -1;
  if (fa < fb)
    return 1;
  return 0;
}

// ============================================================================
// Entry for paired position-value sorting
// ============================================================================
typedef struct {
  float pos;
  float val;
} SortEntry;

static int cmp_entry_by_val(const void *a, const void *b) {
  float va = ((const SortEntry *)a)->val;
  float vb = ((const SortEntry *)b)->val;
  if (va < vb)
    return -1;
  if (va > vb)
    return 1;
  return 0;
}

static int cmp_entry_by_pos(const void *a, const void *b) {
  float pa = ((const SortEntry *)a)->pos;
  float pb = ((const SortEntry *)b)->pos;
  if (pa < pb)
    return -1;
  if (pa > pb)
    return 1;
  return 0;
}

// ============================================================================
// 1D median filter with reflected boundary conditions.
// Applies a sliding window of 'size' to array 'in' of length 'n'.
// Result is written to 'out'. 'in' and 'out' must not overlap.
// ============================================================================
static void medfilt1(const float *in, float *out, int n, int size) {
  int half = size / 2;
  float *window = (float *)malloc(sizeof(float) * size);
  for (int i = 0; i < n; i++) {
    int count = 0;
    for (int j = i - half; j <= i + half; j++) {
      int idx = j;
      if (idx < 0)
        idx = -idx;
      if (idx >= n)
        idx = 2 * n - idx - 2;
      if (idx < 0)
        idx = 0;
      if (idx >= n)
        idx = n - 1;
      window[count++] = in[idx];
    }
    qsort(window, count, sizeof(float), cmp_float_asc);
    out[i] = window[count / 2];
  }
  free(window);
}

// ============================================================================
// 2D median filter on a row-major [nrow x ncol] array.
// Kernel size is (krow, kcol) with reflected boundaries.
// ============================================================================
static void medfilt2(const float *in, float *out, int nrow, int ncol, int krow,
                     int kcol) {
  int hrow = krow / 2;
  int hcol = kcol / 2;
  int wsize = krow * kcol;
  float *window = (float *)malloc(sizeof(float) * wsize);

  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) {
      int count = 0;
      for (int dr = -hrow; dr <= hrow; dr++) {
        for (int dc = -hcol; dc <= hcol; dc++) {
          int rr = r + dr;
          int cc = c + dc;
          if (rr < 0)
            rr = -rr;
          if (rr >= nrow)
            rr = 2 * nrow - rr - 2;
          if (rr < 0)
            rr = 0;
          if (rr >= nrow)
            rr = nrow - 1;
          if (cc < 0)
            cc = -cc;
          if (cc >= ncol)
            cc = 2 * ncol - cc - 2;
          if (cc < 0)
            cc = 0;
          if (cc >= ncol)
            cc = ncol - 1;
          window[count++] = in[rr * ncol + cc];
        }
      }
      qsort(window, count, sizeof(float), cmp_float_asc);
      out[r * ncol + c] = window[count / 2];
    }
  }
  free(window);
}

// ============================================================================
// Column-wise mean (uniform) filter for a [nrow x ncol] array.
// Each column is smoothed independently with a sliding window of 'size'.
// ============================================================================
static void mean_filter_columns(const float *data, float *out, int nrow,
                                int ncol, int size) {
  int half = size / 2;
  for (int c = 0; c < ncol; c++) {
    // Initial sum for row 0
    float sum = 0.0f;
    int count = 0;
    for (int r = -half; r <= half; r++) {
      int rr = r;
      if (rr < 0)
        rr = -rr;
      if (rr >= nrow)
        rr = 2 * nrow - rr - 2;
      if (rr < 0)
        rr = 0;
      if (rr >= nrow)
        rr = nrow - 1;
      sum += data[rr * ncol + c];
      count++;
    }
    out[0 * ncol + c] = sum / count;

    // Slide the window down through rows
    for (int r = 1; r < nrow; r++) {
      int new_idx = r + half;
      if (new_idx >= nrow)
        new_idx = 2 * nrow - new_idx - 2;
      if (new_idx < 0)
        new_idx = 0;
      if (new_idx >= nrow)
        new_idx = nrow - 1;
      sum += data[new_idx * ncol + c];

      int old_idx = r - half - 1;
      if (old_idx < 0)
        old_idx = -old_idx;
      if (old_idx >= nrow)
        old_idx = 2 * nrow - old_idx - 2;
      if (old_idx < 0)
        old_idx = 0;
      if (old_idx >= nrow)
        old_idx = nrow - 1;
      sum -= data[old_idx * ncol + c];

      out[r * ncol + c] = sum / count;
    }
  }
}

// ============================================================================
// Locate stripe column positions via linear fit + SNR thresholding.
// (Algorithm 4 of Vo et al. 2018)
//
// Input:  col_stats[n]  - per-column statistics array
// Output: stripe_mask[n] - binary mask (1.0 = stripe, 0.0 = clean)
// ============================================================================
static void locate_stripe_columns(const float *col_stats, float *stripe_mask,
                                  int n, float snr) {
  // Sort descending to find outlier ends
  float *sorted_stats = (float *)malloc(sizeof(float) * n);
  memcpy(sorted_stats, col_stats, sizeof(float) * n);
  qsort(sorted_stats, n, sizeof(float), cmp_float_desc);

  // Linear fit on the middle 50% (drop 25% from each tail)
  int ndrop = (int)(0.25f * n);
  int fit_start = ndrop;
  int fit_end = n - ndrop - 1;
  int fit_count = fit_end - fit_start + 1;

  if (fit_count < 2) {
    memset(stripe_mask, 0, sizeof(float) * n);
    free(sorted_stats);
    return;
  }

  // Linear regression: y = slope * x + intercept
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = fit_start; i <= fit_end; i++) {
    double x = (double)i;
    double y = (double)sorted_stats[i];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  double denom = fit_count * sxx - sx * sx;
  double slope = 0, intercept = 0;
  if (fabs(denom) > 1e-12) {
    slope = (fit_count * sxy - sx * sy) / denom;
    intercept = (sy - slope * sx) / fit_count;
  }

  double fitted_tail = intercept + slope * (n - 1);
  double noise_level = fabs(fitted_tail - intercept);
  if (noise_level < 1e-6)
    noise_level = 1e-6;

  double outlier_high = fabs(sorted_stats[0] - intercept) / noise_level;
  double outlier_low = fabs(sorted_stats[n - 1] - fitted_tail) / noise_level;

  memset(stripe_mask, 0, sizeof(float) * n);

  if (outlier_high >= snr) {
    double upper_thresh = intercept + noise_level * snr * 0.5;
    for (int i = 0; i < n; i++) {
      if (col_stats[i] > upper_thresh)
        stripe_mask[i] = 1.0f;
    }
  }
  if (outlier_low >= snr) {
    double lower_thresh = fitted_tail - noise_level * snr * 0.5;
    for (int i = 0; i < n; i++) {
      if (col_stats[i] <= lower_thresh)
        stripe_mask[i] = 1.0f;
    }
  }

  free(sorted_stats);
}

// ============================================================================
// 1D binary dilation on a float mask (0.0/1.0).
// ============================================================================
static void dilate_mask(float *mask, int n, int iterations) {
  float *temp = (float *)malloc(sizeof(float) * n);
  for (int iter = 0; iter < iterations; iter++) {
    memcpy(temp, mask, sizeof(float) * n);
    for (int i = 0; i < n; i++) {
      if (temp[i] > 0.5f)
        continue;
      if (i > 0 && temp[i - 1] > 0.5f) {
        mask[i] = 1.0f;
        continue;
      }
      if (i < n - 1 && temp[i + 1] > 0.5f) {
        mask[i] = 1.0f;
        continue;
      }
    }
  }
  free(temp);
}

// ============================================================================
// Sort each column of a [nrow x ncol] array independently (ascending).
// ============================================================================
static void colwise_sort(const float *sinogram, float *sorted, int nrow,
                         int ncol) {
  float *col = (float *)malloc(sizeof(float) * nrow);
  for (int c = 0; c < ncol; c++) {
    for (int r = 0; r < nrow; r++)
      col[r] = sinogram[r * ncol + c];
    qsort(col, nrow, sizeof(float), cmp_float_asc);
    for (int r = 0; r < nrow; r++)
      sorted[r * ncol + c] = col[r];
  }
  free(col);
}

// ============================================================================
// Correct small-to-medium stripes by value-sorting each column, applying
// a median filter in the sorted domain, then restoring original order.
// (Based on Algorithm 3 of Vo et al. 2018)
//
// sinogram: [nrow x ncol] row-major, modified in-place
// ============================================================================
static void correct_by_sorting(float *sinogram, int nrow, int ncol,
                               int filter_width, int dim) {
  SortEntry *entries = (SortEntry *)malloc(sizeof(SortEntry) * nrow);
  float *smoothed = (float *)malloc(sizeof(float) * nrow);
  float *raw_col = (float *)malloc(sizeof(float) * nrow);

  for (int c = 0; c < ncol; c++) {
    // Build position-value pairs for this column
    for (int r = 0; r < nrow; r++) {
      entries[r].pos = (float)r;
      entries[r].val = sinogram[r * ncol + c];
    }

    // Sort by value
    qsort(entries, nrow, sizeof(SortEntry), cmp_entry_by_val);

    // Extract sorted values
    for (int r = 0; r < nrow; r++)
      raw_col[r] = entries[r].val;

    // Median-filter the sorted profile
    medfilt1(raw_col, smoothed, nrow, filter_width);

    // Replace with filtered values
    for (int r = 0; r < nrow; r++)
      entries[r].val = smoothed[r];

    // Restore original order
    qsort(entries, nrow, sizeof(SortEntry), cmp_entry_by_pos);

    // Write corrected column back
    for (int r = 0; r < nrow; r++)
      sinogram[r * ncol + c] = entries[r].val;
  }

  free(entries);
  free(smoothed);
  free(raw_col);
}

// ============================================================================
// Correct large stripe artifacts by normalizing columns based on their
// sorted-domain statistics, then replacing detected stripe columns with
// median-smoothed values.
// (Based on Algorithm 5 of Vo et al. 2018)
//
// sinogram: [nrow x ncol], modified in-place
// ============================================================================
static void correct_large_artifacts(float *sinogram, int nrow, int ncol,
                                    float snr, int filter_width,
                                    float drop_ratio, int do_normalize) {
  if (drop_ratio < 0.0f)
    drop_ratio = 0.0f;
  if (drop_ratio > 0.8f)
    drop_ratio = 0.8f;

  int ndrop = (int)(0.5f * drop_ratio * nrow);

  // Sort each column
  float *col_sorted = (float *)malloc(sizeof(float) * nrow * ncol);
  colwise_sort(sinogram, col_sorted, nrow, ncol);

  // Median-filter the sorted array with kernel (1, filter_width)
  float *col_smooth = (float *)malloc(sizeof(float) * nrow * ncol);
  medfilt2(col_sorted, col_smooth, nrow, ncol, 1, filter_width);

  // Compute trimmed column means for both sorted and smoothed
  float *mean_sorted = (float *)malloc(sizeof(float) * ncol);
  float *mean_smooth = (float *)malloc(sizeof(float) * ncol);
  int trim_count = nrow - 2 * ndrop;
  if (trim_count < 1)
    trim_count = 1;
  for (int c = 0; c < ncol; c++) {
    float s1 = 0, s2 = 0;
    for (int r = ndrop; r < nrow - ndrop; r++) {
      s1 += col_sorted[r * ncol + c];
      s2 += col_smooth[r * ncol + c];
    }
    mean_sorted[c] = s1 / trim_count;
    mean_smooth[c] = s2 / trim_count;
  }

  // Ratio of means identifies non-uniform columns
  float *col_ratio = (float *)malloc(sizeof(float) * ncol);
  for (int c = 0; c < ncol; c++) {
    if (fabsf(mean_smooth[c]) > 1e-9f)
      col_ratio[c] = mean_sorted[c] / mean_smooth[c];
    else
      col_ratio[c] = 1.0f;
  }

  // Detect stripe columns via SNR thresholding
  float *stripe_mask = (float *)malloc(sizeof(float) * ncol);
  locate_stripe_columns(col_ratio, stripe_mask, ncol, snr);
  dilate_mask(stripe_mask, ncol, 1);

  // Normalize all columns by their ratio
  if (do_normalize) {
    for (int r = 0; r < nrow; r++) {
      for (int c = 0; c < ncol; c++) {
        if (fabsf(col_ratio[c]) > 1e-9f)
          sinogram[r * ncol + c] /= col_ratio[c];
      }
    }
  }

  // For stripe columns: replace with smoothed-sorted values via sort mapping
  SortEntry *entries = (SortEntry *)malloc(sizeof(SortEntry) * nrow);

  for (int c = 0; c < ncol; c++) {
    if (stripe_mask[c] < 0.5f)
      continue;

    for (int r = 0; r < nrow; r++) {
      entries[r].pos = (float)r;
      entries[r].val = sinogram[r * ncol + c];
    }
    qsort(entries, nrow, sizeof(SortEntry), cmp_entry_by_val);

    // Map sorted positions to smoothed column values
    for (int r = 0; r < nrow; r++) {
      entries[r].val = col_smooth[r * ncol + c];
    }

    qsort(entries, nrow, sizeof(SortEntry), cmp_entry_by_pos);

    for (int r = 0; r < nrow; r++) {
      sinogram[r * ncol + c] = entries[r].val;
    }
  }

  free(entries);
  free(col_sorted);
  free(col_smooth);
  free(mean_sorted);
  free(mean_smooth);
  free(col_ratio);
  free(stripe_mask);
}

// ============================================================================
// Correct unresponsive and fluctuating pixel columns by detecting them
// with a fluctuation metric and replacing via linear interpolation from
// neighboring good columns.
// (Based on Algorithm 6 of Vo et al. 2018)
//
// sinogram: [nrow x ncol], modified in-place
// ============================================================================
static void correct_dead_pixels(float *sinogram, int nrow, int ncol, float snr,
                                int filter_width,
                                int apply_residual_correction) {
  // Smooth along projection axis with a mean window of 10
  float *smoothed = (float *)malloc(sizeof(float) * nrow * ncol);
  mean_filter_columns(sinogram, smoothed, nrow, ncol, 10);

  // Fluctuation metric: sum of absolute deviations from smooth
  float *fluctuation = (float *)malloc(sizeof(float) * ncol);
  for (int c = 0; c < ncol; c++) {
    float s = 0;
    for (int r = 0; r < nrow; r++) {
      s += fabsf(sinogram[r * ncol + c] - smoothed[r * ncol + c]);
    }
    fluctuation[c] = s;
  }

  // Median-filter the fluctuation profile as background estimate
  float *fluct_background = (float *)malloc(sizeof(float) * ncol);
  medfilt1(fluctuation, fluct_background, ncol, filter_width);

  // Ratio of fluctuation to background
  float *fluct_ratio = (float *)malloc(sizeof(float) * ncol);
  for (int c = 0; c < ncol; c++) {
    if (fabsf(fluct_background[c]) > 1e-9f)
      fluct_ratio[c] = fluctuation[c] / fluct_background[c];
    else
      fluct_ratio[c] = 1.0f;
  }

  // Detect dead/fluctuating columns
  float *stripe_mask = (float *)malloc(sizeof(float) * ncol);
  locate_stripe_columns(fluct_ratio, stripe_mask, ncol, snr);
  dilate_mask(stripe_mask, ncol, 1);

  // Keep edge columns unmarked (avoid boundary artifacts)
  if (ncol > 2) {
    stripe_mask[0] = 0.0f;
    stripe_mask[1] = 0.0f;
    stripe_mask[ncol - 2] = 0.0f;
    stripe_mask[ncol - 1] = 0.0f;
  }

  // Interpolate bad columns from neighboring good columns
  int *good_cols = (int *)malloc(sizeof(int) * ncol);
  int ngood = 0;
  for (int c = 0; c < ncol; c++) {
    if (stripe_mask[c] < 0.5f)
      good_cols[ngood++] = c;
  }

  if (ngood > 1) {
    for (int c = 0; c < ncol; c++) {
      if (stripe_mask[c] < 0.5f)
        continue;

      // Find bracketing good columns
      int left_idx = -1, right_idx = -1;
      for (int g = 0; g < ngood; g++) {
        if (good_cols[g] <= c)
          left_idx = g;
        if (good_cols[g] >= c && right_idx < 0)
          right_idx = g;
      }

      if (left_idx < 0)
        left_idx = right_idx;
      if (right_idx < 0)
        right_idx = left_idx;
      if (left_idx < 0 || right_idx < 0)
        continue;

      int lc = good_cols[left_idx];
      int rc = good_cols[right_idx];

      if (lc == rc) {
        for (int r = 0; r < nrow; r++)
          sinogram[r * ncol + c] = sinogram[r * ncol + lc];
      } else {
        float t = (float)(c - lc) / (float)(rc - lc);
        for (int r = 0; r < nrow; r++) {
          sinogram[r * ncol + c] = (1.0f - t) * sinogram[r * ncol + lc] +
                                   t * sinogram[r * ncol + rc];
        }
      }
    }
  }

  free(good_cols);

  // Residual correction via large-artifact removal
  if (apply_residual_correction) {
    correct_large_artifacts(sinogram, nrow, ncol, snr, filter_width, 0.1f, 1);
  }

  free(smoothed);
  free(fluctuation);
  free(fluct_background);
  free(fluct_ratio);
  free(stripe_mask);
}

// ============================================================================
// cleanup_sinogram_stripes: Public API.
//
// Removes all types of stripe artifacts from a normalized sinogram by
// combining dead-pixel correction, large-artifact normalization, and
// sorting-based smoothing.
//
// sinogram: [nrow x ncol] row-major float array, modified in-place.
//   nrow = number of projection angles
//   ncol = detector width (horizontal pixels)
// snr:       SNR threshold for stripe detection (default 3)
// la_size:   Median filter window for large artifacts (default 61, odd)
// sm_size:   Median filter window for small artifacts (default 21, odd)
// dim:       Median filter dimension for sorting method (1 or 2)
// ============================================================================
void cleanup_sinogram_stripes(float *sinogram, int nrow, int ncol, float snr,
                              int la_size, int sm_size, int dim) {
  // Ensure odd window sizes
  if (la_size % 2 == 0)
    la_size++;
  if (sm_size % 2 == 0)
    sm_size++;

  // Phase 1: Correct dead/unresponsive columns + large artifacts
  correct_dead_pixels(sinogram, nrow, ncol, snr, la_size, 1);

  // Phase 2: Correct small-to-medium stripes via sorting
  correct_by_sorting(sinogram, nrow, ncol, sm_size, dim);
}
