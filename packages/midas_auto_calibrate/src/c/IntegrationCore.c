//
// IntegrationCore.c — Apply pixel-bin mapping to image → 1D radial profiles
//
// See IntegrationCore.h for API documentation.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "IntegrationCore.h"
#include <string.h>
#include <math.h>

void integration_apply_map(
    struct MapPixelData **const *pxList,
    int *const *nPxList,
    int nRBins, int nEtaBins,
    const double *image, int NrPixelsY, int NrPixelsZ,
    const double *dark,
    double BC_y, double BC_z, int GradientCorrection,
    double *profiles_out,
    double *norm_out)
{
  long long totalBins = (long long)nRBins * nEtaBins;
  memset(profiles_out, 0, totalBins * sizeof(*profiles_out));
  memset(norm_out, 0, totalBins * sizeof(*norm_out));

  // Iterate over all bins and accumulate
  for (int r = 0; r < nRBins; r++) {
    for (int e = 0; e < nEtaBins; e++) {
      int nPx = nPxList[r][e];
      if (nPx == 0) continue;
      const struct MapPixelData *entries = pxList[r][e];
      long long binIdx = (long long)r * nEtaBins + e;
      double sum_weighted = 0.0;
      double sum_area = 0.0;
      for (int p = 0; p < nPx; p++) {
        // Start from sub-pixel center stored by mapper
        double read_y = entries[p].y, read_z = entries[p].z;

        // Radial resampling: shift read position to R-bin center
        if (GradientCorrection && entries[p].deltaR != 0.0f) {
          double dy = entries[p].y - BC_y;
          double dz = entries[p].z - BC_z;
          double R = sqrt(dy * dy + dz * dz);
          if (R > 1.0) {
            read_y -= entries[p].deltaR * dy / R;
            read_z -= entries[p].deltaR * dz / R;
          }
        }

        // Bilinear interpolation at (read_y, read_z)
        int iy = (int)floorf(read_y);
        int iz = (int)floorf(read_z);
        double fy = read_y - iy, fz = read_z - iz;
        if (iy < 0) { iy = 0; fy = 0; }
        if (iy >= NrPixelsY - 1) { iy = NrPixelsY - 2; fy = 1; }
        if (iz < 0) { iz = 0; fz = 0; }
        if (iz >= NrPixelsZ - 1) { iz = NrPixelsZ - 2; fz = 1; }

        double w00 = (1 - fy) * (1 - fz);
        double w10 = fy       * (1 - fz);
        double w01 = (1 - fy) * fz;
        double w11 = fy       * fz;

        size_t idx00 = (size_t)iz * NrPixelsY + iy;
        size_t idx10 = idx00 + 1;
        size_t idx01 = idx00 + NrPixelsY;
        size_t idx11 = idx01 + 1;

        double pixVal = image[idx00] * w00 + image[idx10] * w10 +
                        image[idx01] * w01 + image[idx11] * w11;
        if (dark != NULL) {
          pixVal -= dark[idx00] * w00 + dark[idx10] * w10 +
                    dark[idx01] * w01 + dark[idx11] * w11;
        }
        sum_weighted += pixVal * entries[p].frac;
        sum_area += entries[p].areaWeight;
      }
      profiles_out[binIdx] = sum_weighted;
      norm_out[binIdx] = sum_area;
    }
  }
}
