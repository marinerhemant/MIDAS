//
// IntegrationCore.h — Apply pixel mapping to image → 1D radial profiles
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef INTEGRATION_CORE_H
#define INTEGRATION_CORE_H

#include "MapperCore.h"

// Apply pixel-bin mapping to image data, producing 1D intensity profiles.
//
// For each (rBin, etaBin), sums (image[pixel] - dark[pixel]) * weight
// over all pixels mapped to that bin.  Pixel intensities are read via
// bilinear interpolation at the sub-pixel (y, z) stored in MapPixelData.
//
// When GradientCorrection=1 and deltaR != 0, the read position is shifted
// along the radial direction from (BC_y, BC_z) by -deltaR to sample at
// the R-bin center, suppressing cardinal-angle aliasing.
//
// profiles_out[rBin * nEtaBins + etaBin] = sum of (pixel_value * frac)
// norm_out[rBin * nEtaBins + etaBin]     = sum of areaWeight (for normalization)
//
// All output arrays must be pre-allocated to size nRBins * nEtaBins.
// dark may be NULL (no dark subtraction).
void integration_apply_map(
    struct MapPixelData **const *pxList,
    int *const *nPxList,
    int nRBins, int nEtaBins,
    const double *image, int NrPixelsY, int NrPixelsZ,
    const double *dark,
    double BC_y, double BC_z, int GradientCorrection,
    double *profiles_out,
    double *norm_out);

#endif /* INTEGRATION_CORE_H */
