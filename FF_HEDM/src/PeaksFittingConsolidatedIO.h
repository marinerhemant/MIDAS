/**
 * PeaksFittingConsolidatedIO.h - Consolidated file I/O for PeaksFitting
 *
 * Replaces per-frame files ({stem}_{frameNr}_PS.csv and {stem}_{frameNr}_PX.bin)
 * with 2 consolidated binary files:
 *
 *   AllPeaks_PS.bin:
 *     Header: [int32 nFrames][int32 nPeaks[f] x nFrames][int64 offset[f] x nFrames]
 *     Data:   [double x N_PEAK_COLS x nPeaks[f]] concatenated per frame
 *
 *   AllPeaks_PX.bin:
 *     Header: [int32 nFrames][int32 NrPixels][int32 nPeaks[f] x nFrames][int64 offset[f] x nFrames]
 *     Data:   for each frame, for each peak: [int32 nPx][int16 y, int16 z x nPx]
 *
 * Pattern follows IndexerConsolidatedIO.h (VoxelAccumulator).
 *
 * Author: Hemant Sharma / AI Assistant
 */

#ifndef PEAKSFITTING_CONSOLIDATED_IO_H
#define PEAKSFITTING_CONSOLIDATED_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_PEAK_COLS 29

/* Column names (for reference / Python viewer) */
static const char *PEAK_COL_NAMES[N_PEAK_COLS] = {
    "SpotID",
    "IntegratedIntensity",
    "Omega",
    "YCen",
    "ZCen",
    "IMax",
    "Radius",
    "Eta",
    "SigmaR",
    "SigmaEta",
    "NrPixels",
    "NrPxTot",
    "nPeaks",
    "maxY",
    "maxZ",
    "diffY",
    "diffZ",
    "rawIMax",
    "returnCode",
    "retVal",
    "BG",
    "SigmaGR",
    "SigmaLR",
    "SigmaGEta",
    "SigmaLEta",
    "MU",
    "RawSumIntensity",
    "maskTouched",
    "FitRMSE",
};

/**
 * Per-frame accumulator: collects variable-length peak data during processing.
 * Thread-safe: each frame is processed by exactly one thread.
 */
typedef struct {
  double *peakData;   /* N_PEAK_COLS doubles per peak (flat array) */
  int16_t *pixelYZ;   /* Interleaved (y,z) pixel coords, all peaks concatenated */
  int *nPixPerPeak;   /* nPixels for each peak */
  int nPeaks;         /* number of peaks found in this frame */
  int nPixTotal;      /* total pixel coord pairs stored */
  int capPeaks;       /* current capacity for peakData/nPixPerPeak */
  int capPixels;      /* current capacity for pixelYZ (in pairs, i.e. entries/2) */
} FrameAccumulator;

/**
 * Initialize a FrameAccumulator
 */
static inline void FrameAccum_init(FrameAccumulator *acc) {
  acc->nPeaks = 0;
  acc->nPixTotal = 0;
  acc->capPeaks = 64;
  acc->capPixels = 4096;
  acc->peakData =
      (double *)malloc(acc->capPeaks * N_PEAK_COLS * sizeof(double));
  acc->nPixPerPeak = (int *)malloc(acc->capPeaks * sizeof(int));
  acc->pixelYZ = (int16_t *)malloc(acc->capPixels * 2 * sizeof(int16_t));
}

/**
 * Add a peak to the accumulator.
 * @param peakRow   N_PEAK_COLS doubles (same column order as _PS.csv)
 * @param pixY      array of y pixel coordinates for this peak
 * @param pixZ      array of z pixel coordinates for this peak
 * @param nPixels   number of pixel coordinates
 */
static inline void FrameAccum_addPeak(FrameAccumulator *acc,
                                       const double *peakRow,
                                       const int16_t *pixY,
                                       const int16_t *pixZ, int nPixels) {
  /* Grow peak arrays if needed */
  if (acc->nPeaks >= acc->capPeaks) {
    acc->capPeaks *= 2;
    acc->peakData = (double *)realloc(
        acc->peakData, acc->capPeaks * N_PEAK_COLS * sizeof(double));
    acc->nPixPerPeak =
        (int *)realloc(acc->nPixPerPeak, acc->capPeaks * sizeof(int));
  }
  /* Grow pixel arrays if needed */
  while (acc->nPixTotal + nPixels > acc->capPixels) {
    acc->capPixels *= 2;
    acc->pixelYZ = (int16_t *)realloc(acc->pixelYZ,
                                       acc->capPixels * 2 * sizeof(int16_t));
  }

  memcpy(&acc->peakData[acc->nPeaks * N_PEAK_COLS], peakRow,
         N_PEAK_COLS * sizeof(double));
  acc->nPixPerPeak[acc->nPeaks] = nPixels;

  /* Store interleaved y,z */
  for (int i = 0; i < nPixels; i++) {
    acc->pixelYZ[(acc->nPixTotal + i) * 2 + 0] = pixY[i];
    acc->pixelYZ[(acc->nPixTotal + i) * 2 + 1] = pixZ[i];
  }

  acc->nPeaks++;
  acc->nPixTotal += nPixels;
}

/**
 * Free a FrameAccumulator
 */
static inline void FrameAccum_free(FrameAccumulator *acc) {
  free(acc->peakData);
  free(acc->nPixPerPeak);
  free(acc->pixelYZ);
  acc->peakData = NULL;
  acc->nPixPerPeak = NULL;
  acc->pixelYZ = NULL;
}

/**
 * Write consolidated output files from an array of FrameAccumulators.
 *
 * Produces:
 *   {outFolder}/AllPeaks_PS.bin
 *   {outFolder}/AllPeaks_PX.bin
 *
 * For multi-block runs, the header always contains nTotalFrames slots.
 * Only the frames in [startFrame, endFrame) are populated; other slots
 * have nPeaks=0. This allows consumers to always read a single file
 * per dataset regardless of how many blocks were used.
 *
 * @param accs          Array of accumulators (index = frame - startFrame)
 * @param nTotalFrames  Total number of frames in the dataset
 * @param startFrame    First frame processed by this block (0-based)
 * @param endFrame      One past last frame processed (0-based)
 * @param nrPixels      Detector dimension (for PX header)
 * @param outFolder     Output folder path (e.g. ".../Temp")
 */
static inline void WriteConsolidatedPeakFiles(const FrameAccumulator *accs,
                                               int nTotalFrames,
                                               int startFrame, int endFrame,
                                               int nrPixels,
                                               const char *outFolder) {
  char fn[4096];
  FILE *f;
  int nLocal = endFrame - startFrame;

  /* Pre-compute per-frame counts */
  int32_t *nPeaksArr = (int32_t *)calloc(nTotalFrames, sizeof(int32_t));
  for (int i = 0; i < nLocal; i++) {
    nPeaksArr[startFrame + i] = accs[i].nPeaks;
  }

  /* ============================================================
   * AllPeaks_PS.bin
   * Header: int32 nFrames, int32 nPeaks[nFrames], int64 offset[nFrames]
   * Data: double[nPeaks × N_PEAK_COLS] per frame
   * ============================================================ */
  {
    int64_t *offsets = (int64_t *)calloc(nTotalFrames, sizeof(int64_t));
    int64_t headerSize = (int64_t)sizeof(int32_t) +
                         (int64_t)nTotalFrames * sizeof(int32_t) +
                         (int64_t)nTotalFrames * sizeof(int64_t);
    int64_t dataOff = headerSize;
    for (int f2 = 0; f2 < nTotalFrames; f2++) {
      offsets[f2] = dataOff;
      dataOff += (int64_t)nPeaksArr[f2] * N_PEAK_COLS * sizeof(double);
    }

    sprintf(fn, "%s/AllPeaks_PS.bin", outFolder);
    f = fopen(fn, "wb");
    if (f) {
      int32_t nf = (int32_t)nTotalFrames;
      fwrite(&nf, sizeof(int32_t), 1, f);
      fwrite(nPeaksArr, sizeof(int32_t), nTotalFrames, f);
      fwrite(offsets, sizeof(int64_t), nTotalFrames, f);
      for (int v = 0; v < nTotalFrames; v++) {
        if (v >= startFrame && v < endFrame) {
          int li = v - startFrame;
          if (accs[li].nPeaks > 0)
            fwrite(accs[li].peakData, N_PEAK_COLS * sizeof(double),
                   accs[li].nPeaks, f);
        }
      }
      fclose(f);
      printf("Wrote %s (%d frames)\n", fn, nTotalFrames);
    }
    free(offsets);
  }

  /* ============================================================
   * AllPeaks_PX.bin
   * Header: int32 nFrames, int32 NrPixels,
   *         int32 nPeaks[nFrames], int64 offset[nFrames]
   * Data: for each frame, for each peak:
   *         int32 nPixels, (int16 y, int16 z) × nPixels
   * ============================================================ */
  {
    /* Compute pixel data sizes per frame */
    int64_t *offsets = (int64_t *)calloc(nTotalFrames, sizeof(int64_t));
    int64_t headerSize = (int64_t)sizeof(int32_t) * 2 +
                         (int64_t)nTotalFrames * sizeof(int32_t) +
                         (int64_t)nTotalFrames * sizeof(int64_t);
    int64_t dataOff = headerSize;
    for (int f2 = 0; f2 < nTotalFrames; f2++) {
      offsets[f2] = dataOff;
      if (f2 >= startFrame && f2 < endFrame) {
        int li = f2 - startFrame;
        /* Each peak contributes: 4 bytes (nPx) + nPx * 4 bytes (y,z pairs) */
        for (int pk = 0; pk < accs[li].nPeaks; pk++) {
          dataOff += sizeof(int32_t) +
                     (int64_t)accs[li].nPixPerPeak[pk] * 2 * sizeof(int16_t);
        }
      }
    }

    sprintf(fn, "%s/AllPeaks_PX.bin", outFolder);
    f = fopen(fn, "wb");
    if (f) {
      int32_t nf = (int32_t)nTotalFrames;
      int32_t np = (int32_t)nrPixels;
      fwrite(&nf, sizeof(int32_t), 1, f);
      fwrite(&np, sizeof(int32_t), 1, f);
      fwrite(nPeaksArr, sizeof(int32_t), nTotalFrames, f);
      fwrite(offsets, sizeof(int64_t), nTotalFrames, f);

      for (int v = 0; v < nTotalFrames; v++) {
        if (v >= startFrame && v < endFrame) {
          int li = v - startFrame;
          int pixOff = 0;
          for (int pk = 0; pk < accs[li].nPeaks; pk++) {
            int32_t nPx = (int32_t)accs[li].nPixPerPeak[pk];
            fwrite(&nPx, sizeof(int32_t), 1, f);
            fwrite(&accs[li].pixelYZ[pixOff * 2], sizeof(int16_t), nPx * 2, f);
            pixOff += nPx;
          }
        }
      }
      fclose(f);
      printf("Wrote %s (%d frames)\n", fn, nTotalFrames);
    }
    free(offsets);
  }

  free(nPeaksArr);
}

/* ================================================================
 * READER FUNCTIONS (used by MergeOverlappingPeaks and other consumers)
 * ================================================================ */

/**
 * Reader for AllPeaks_PS.bin
 */
typedef struct {
  int32_t nFrames;
  int32_t *nPeaks;    /* per-frame peak counts */
  int64_t *offsets;    /* per-frame byte offsets into file */
  double *data;        /* all peak data loaded into memory */
  int64_t headerSize;
} ConsolidatedPeakReader;

/**
 * Open AllPeaks_PS.bin for reading.
 * Returns 0 on success, -1 on failure.
 */
static inline int ConsolidatedPeakReader_open(ConsolidatedPeakReader *r,
                                               const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) return -1;

  if (fread(&r->nFrames, sizeof(int32_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  r->nPeaks = (int32_t *)malloc(r->nFrames * sizeof(int32_t));
  r->offsets = (int64_t *)malloc(r->nFrames * sizeof(int64_t));
  if (fread(r->nPeaks, sizeof(int32_t), r->nFrames, f) != (size_t)r->nFrames) {
    fclose(f);
    return -1;
  }
  if (fread(r->offsets, sizeof(int64_t), r->nFrames, f) !=
      (size_t)r->nFrames) {
    fclose(f);
    return -1;
  }
  r->headerSize = (int64_t)sizeof(int32_t) +
                  (int64_t)r->nFrames * sizeof(int32_t) +
                  (int64_t)r->nFrames * sizeof(int64_t);

  /* Read all data into memory */
  fseek(f, 0, SEEK_END);
  long totalSize = ftell(f);
  long dataSize = totalSize - r->headerSize;
  if (dataSize > 0) {
    r->data = (double *)malloc(dataSize);
    fseek(f, r->headerSize, SEEK_SET);
    (void)fread(r->data, 1, dataSize, f);
  } else {
    r->data = NULL;
  }
  fclose(f);
  return 0;
}

/**
 * Get pointer to a frame's peak data.
 * @param frameIdx  0-based frame index
 * @return pointer to nPeaks[frameIdx] × N_PEAK_COLS doubles, or NULL
 */
static inline const double *
ConsolidatedPeakReader_getFrame(const ConsolidatedPeakReader *r, int frameIdx) {
  if (frameIdx < 0 || frameIdx >= r->nFrames || r->nPeaks[frameIdx] == 0)
    return NULL;
  int64_t byteOff = r->offsets[frameIdx] - r->headerSize;
  return (const double *)((const char *)r->data + byteOff);
}

static inline void ConsolidatedPeakReader_close(ConsolidatedPeakReader *r) {
  free(r->nPeaks);
  free(r->offsets);
  free(r->data);
  r->nPeaks = NULL;
  r->offsets = NULL;
  r->data = NULL;
}

/**
 * Reader for AllPeaks_PX.bin
 */
typedef struct {
  int32_t nFrames;
  int32_t nrPixels;
  int32_t *nPeaks;
  int64_t *offsets;
  char *data;        /* raw binary data */
  int64_t headerSize;
  int64_t dataSize;
} ConsolidatedPixelReader;

static inline int ConsolidatedPixelReader_open(ConsolidatedPixelReader *r,
                                                const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) return -1;

  if (fread(&r->nFrames, sizeof(int32_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fread(&r->nrPixels, sizeof(int32_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  r->nPeaks = (int32_t *)malloc(r->nFrames * sizeof(int32_t));
  r->offsets = (int64_t *)malloc(r->nFrames * sizeof(int64_t));
  if (fread(r->nPeaks, sizeof(int32_t), r->nFrames, f) != (size_t)r->nFrames ||
      fread(r->offsets, sizeof(int64_t), r->nFrames, f) !=
          (size_t)r->nFrames) {
    fclose(f);
    return -1;
  }
  r->headerSize = (int64_t)sizeof(int32_t) * 2 +
                  (int64_t)r->nFrames * sizeof(int32_t) +
                  (int64_t)r->nFrames * sizeof(int64_t);

  fseek(f, 0, SEEK_END);
  long totalSize = ftell(f);
  r->dataSize = totalSize - r->headerSize;
  if (r->dataSize > 0) {
    r->data = (char *)malloc(r->dataSize);
    fseek(f, r->headerSize, SEEK_SET);
    (void)fread(r->data, 1, r->dataSize, f);
  } else {
    r->data = NULL;
  }
  fclose(f);
  return 0;
}

/**
 * Get pointer to a frame's pixel data block.
 * The block contains: for each peak, int32 nPx, (int16 y, int16 z) × nPx
 * @param frameIdx  0-based frame index
 * @return pointer to raw pixel data block, or NULL
 */
static inline const char *
ConsolidatedPixelReader_getFrame(const ConsolidatedPixelReader *r,
                                  int frameIdx) {
  if (frameIdx < 0 || frameIdx >= r->nFrames || r->nPeaks[frameIdx] == 0)
    return NULL;
  int64_t byteOff = r->offsets[frameIdx] - r->headerSize;
  return r->data + byteOff;
}

static inline void ConsolidatedPixelReader_close(ConsolidatedPixelReader *r) {
  free(r->nPeaks);
  free(r->offsets);
  free(r->data);
  r->nPeaks = NULL;
  r->offsets = NULL;
  r->data = NULL;
}

#endif /* PEAKSFITTING_CONSOLIDATED_IO_H */
