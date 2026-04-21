//
// ImageUtils.h — Shared image transformation and utility functions
//
// Consolidates DoImageTransformations, MakeSquare, REtaMapper from
// DetectorMapper.c, IntegratorZarrOMP.c, and CalibrantPanelShiftsOMP.c.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

// Unified image transformation.
// Handles rectangular images, all 4 TransOpts:
//   0 = no-op, 1 = FlipLR, 2 = FlipUD, 3 = Transpose (requires NrPixelsY == NrPixelsZ)
// Uses efficient in-place swaps (no temp matrix allocation).
// ImageIn and ImageOut may be the same pointer for in-place operation.
// When ImageIn != ImageOut, ImageIn is first copied to ImageOut.
void midas_image_transform(int NrTransOpt, const int TransOpt[10],
                           double *ImageIn, double *ImageOut,
                           int NrPixelsY, int NrPixelsZ);

// Pad a non-square image into a square buffer.
// NrPixels = max(NrPixelsY, NrPixelsZ). OutImage is pre-zeroed.
// InImage layout: row-major, NrPixelsZ rows × NrPixelsY columns.
void midas_make_square(int NrPixels, int NrPixelsY, int NrPixelsZ,
                       const double *InImage, double *OutImage);

// Build R-bin and Eta-bin edge arrays from start values and bin sizes.
void midas_reta_mapper(double Rmin, double EtaMin, int nEtaBins, int nRBins,
                       double EtaBinSize, double RBinSize,
                       double *EtaBinsLow, double *EtaBinsHigh,
                       double *RBinsLow, double *RBinsHigh);

#endif /* IMAGE_UTILS_H */
