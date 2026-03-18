//
// ImageUtils.c — Shared image transformation and utility functions
//
// See ImageUtils.h for API documentation.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "ImageUtils.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>

// --------------------------------------------------------------------------
// midas_image_transform — unified image transformation
// --------------------------------------------------------------------------
// Uses DetectorMapper's efficient in-place swap approach.
// Supports rectangular images for FlipLR and FlipUD.
// Transpose requires square images (asserted).
void midas_image_transform(int NrTransOpt, const int TransOpt[10],
                           double *ImageIn, double *ImageOut,
                           int NrPixelsY, int NrPixelsZ) {
  // Copy input to output if different buffers
  if (ImageIn != ImageOut) {
    memcpy(ImageOut, ImageIn,
           (size_t)NrPixelsY * NrPixelsZ * sizeof(*ImageIn));
  }
  if (NrTransOpt == 0) {
    return;
  }

  double buffer;
  for (int i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] == 0) {
      // No-op
      continue;
    } else if (TransOpt[i] == 1) {
      // Flip Left-Right (invert Y columns)
      for (int l = 0; l < NrPixelsZ; l++) {
        for (int k = 0; k < NrPixelsY / 2; k++) {
          buffer = ImageOut[l * NrPixelsY + k];
          ImageOut[l * NrPixelsY + k] =
              ImageOut[l * NrPixelsY + (NrPixelsY - k - 1)];
          ImageOut[l * NrPixelsY + (NrPixelsY - k - 1)] = buffer;
        }
      }
    } else if (TransOpt[i] == 2) {
      // Flip Top-Bottom (invert Z rows)
      for (int l = 0; l < NrPixelsZ / 2; l++) {
        for (int k = 0; k < NrPixelsY; k++) {
          buffer = ImageOut[l * NrPixelsY + k];
          ImageOut[l * NrPixelsY + k] =
              ImageOut[(NrPixelsZ - l - 1) * NrPixelsY + k];
          ImageOut[(NrPixelsZ - l - 1) * NrPixelsY + k] = buffer;
        }
      }
    } else if (TransOpt[i] == 3) {
      // Transpose — requires square image
      if (NrPixelsY != NrPixelsZ) {
        fprintf(stderr,
                "midas_image_transform: TransOpt=3 (transpose) requires "
                "NrPixelsY == NrPixelsZ (got %d x %d)\n",
                NrPixelsY, NrPixelsZ);
        return;
      }
      for (int l = 0; l < NrPixelsZ; l++) {
        for (int k = l + 1; k < NrPixelsY; k++) {
          buffer = ImageOut[l * NrPixelsY + k];
          ImageOut[l * NrPixelsY + k] = ImageOut[k * NrPixelsY + l];
          ImageOut[k * NrPixelsY + l] = buffer;
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// midas_make_square — pad non-square image into square buffer
// --------------------------------------------------------------------------
void midas_make_square(int NrPixels, int NrPixelsY, int NrPixelsZ,
                       const double *InImage, double *OutImage) {
  // Always pre-zero the output (padding area must be zero)
  memset(OutImage, 0, (size_t)NrPixels * NrPixels * sizeof(*OutImage));
  if (NrPixelsY == NrPixelsZ) {
    memcpy(OutImage, InImage,
           (size_t)NrPixels * NrPixels * sizeof(*InImage));
  } else if (NrPixelsY > NrPixelsZ) {
    // More columns than rows — copy the first NrPixelsZ rows contiguously
    memcpy(OutImage, InImage,
           (size_t)NrPixelsY * NrPixelsZ * sizeof(*InImage));
  } else {
    // More rows than columns — copy each row with padding
    for (int i = 0; i < NrPixelsZ; i++) {
      memcpy(OutImage + (size_t)i * NrPixels,
             InImage + (size_t)i * NrPixelsY,
             NrPixelsY * sizeof(*InImage));
    }
  }
}

// --------------------------------------------------------------------------
// midas_reta_mapper — build R/Eta bin edge arrays
// --------------------------------------------------------------------------
void midas_reta_mapper(double Rmin, double EtaMin, int nEtaBins, int nRBins,
                       double EtaBinSize, double RBinSize,
                       double *EtaBinsLow, double *EtaBinsHigh,
                       double *RBinsLow, double *RBinsHigh) {
  for (int i = 0; i < nEtaBins; i++) {
    EtaBinsLow[i] = EtaBinSize * i + EtaMin;
    EtaBinsHigh[i] = EtaBinSize * (i + 1) + EtaMin;
  }
  for (int i = 0; i < nRBins; i++) {
    RBinsLow[i] = RBinSize * i + Rmin;
    RBinsHigh[i] = RBinSize * (i + 1) + Rmin;
  }
}
