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
        // Map.bin stores raw (untransformed) pixel coordinates
        int iy = (int)entries[p].y;
        int iz = (int)entries[p].z;
        // Bounds check
        if (iy < 0 || iy >= NrPixelsY || iz < 0 || iz >= NrPixelsZ)
          continue;
        long long pixIdx = (long long)iz * NrPixelsY + iy;
        double pixVal = image[pixIdx];
        if (dark != NULL) {
          pixVal -= dark[pixIdx];
        }
        sum_weighted += pixVal * entries[p].frac;
        sum_area += entries[p].areaWeight;
      }
      profiles_out[binIdx] = sum_weighted;
      norm_out[binIdx] = sum_area;
    }
  }
}
