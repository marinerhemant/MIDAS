//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// MaskUtils.h — Shared mask bitfield utilities for MIDAS executables.
//
// Provides macros and builder functions for creating detector pixel masks.
//
// Mask convention: 1 = masked (bad), 0 = valid pixel.
// Storage: compact bitfield (int array, 32 bits per element).
//

#ifndef MASK_UTILS_H
#define MASK_UTILS_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// --- Bitfield macros ---
// These operate on a compact int[] bitmap: bit k is stored in A[k/32], bit
// (k%32).
#define MaskSetBit(A, k) ((A)[(k) / 32] |= (1 << ((k) % 32)))
#define MaskClearBit(A, k) ((A)[(k) / 32] &= ~(1 << ((k) % 32)))
#define MaskTestBit(A, k) ((A)[(k) / 32] & (1 << ((k) % 32)))

// --- Inline builder functions ---
// All return a calloc'd int* bitfield. Caller must free().
// *outMaskSize receives the number of int elements allocated.

// Allocate a zeroed bitfield for NrPixels x NrPixels.
static inline int *MaskAllocBitfield(int NrPixels, size_t *outMaskSize) {
  size_t sz = (size_t)NrPixels * NrPixels;
  sz = sz / 32 + 1;
  int *mask = calloc(sz, sizeof(*mask));
  if (outMaskSize)
    *outMaskSize = sz;
  return mask;
}

// Build mask from a pre-transformed square image where pixel == 1 means masked.
// The caller is responsible for reading, MakeSquare, and DoImageTransformations
// before calling this function.
static inline int *MaskFromSquareImage(const double *squareImage, int NrPixels,
                                       size_t *outMaskSize) {
  int *mask = MaskAllocBitfield(NrPixels, outMaskSize);
  if (!mask)
    return NULL;

  size_t nMasked = 0;
  size_t totalPx = (size_t)NrPixels * NrPixels;
  for (size_t i = 0; i < totalPx; i++) {
    if (squareImage[i] == 1.0) {
      MaskSetBit(mask, i);
      nMasked++;
    }
  }
  printf("MaskUtils: %zu pixels masked.\n", nMasked);
  return mask;
}

// Build mask from a pre-transformed square image by matching intensity values.
// Marks pixels matching gapIntensity OR badPxIntensity as masked.
static inline int *MaskFromIntensities(const double *squareImage, int NrPixels,
                                       double gapIntensity,
                                       double badPxIntensity,
                                       size_t *outMaskSize) {
  int *mask = MaskAllocBitfield(NrPixels, outMaskSize);
  if (!mask)
    return NULL;

  size_t nMasked = 0;
  size_t totalPx = (size_t)NrPixels * NrPixels;
  for (size_t i = 0; i < totalPx; i++) {
    if (squareImage[i] == gapIntensity || squareImage[i] == badPxIntensity) {
      MaskSetBit(mask, i);
      nMasked++;
    }
  }
  printf("MaskUtils: %zu pixels masked (gap=%.0f, badpx=%.0f).\n", nMasked,
         gapIntensity, badPxIntensity);
  return mask;
}

// OR a second mask TIFF image into an existing mask bitfield.
// Useful for combining gap + bad pixel TIFFs.
static inline void MaskOrSquareImage(int *mask, const double *squareImage,
                                     int NrPixels) {
  size_t totalPx = (size_t)NrPixels * NrPixels;
  for (size_t i = 0; i < totalPx; i++) {
    if (squareImage[i] == 1.0) {
      MaskSetBit(mask, i);
    }
  }
}

#endif // MASK_UTILS_H
