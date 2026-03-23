/**
 * IndexerConsolidatedIO.h - Consolidated file I/O for IndexerScanning
 *
 * Replaces per-voxel files (IndexBest_voxNr_*.bin, IndexKey_voxNr_*.txt,
 * IndexBest_IDs_voxNr_*.bin) with 3 consolidated binaries.
 *
 * Format:
 *   IndexBest_all.bin:
 *     Header: [int32 nVoxels][int32 nSolutions[v] x nVoxels][int64 offset[v] x nVoxels]
 *     Data:   [double x 16 x nSolutions[v]] concatenated per voxel
 *
 *   IndexKey_all.bin:
 *     Header: [int32 nVoxels][int32 nSolutions[v] x nVoxels][int64 offset[v] x nVoxels]
 *     Data:   [size_t x 4 x nSolutions[v]] concatenated per voxel
 *
 *   IndexBest_IDs_all.bin:
 *     Header: [int32 nVoxels][int32 nIDs[v] x nVoxels][int64 offset[v] x nVoxels]
 *     Data:   [int x nIDs[v]] concatenated per voxel
 *
 * All consumers can mmap the file and use offset[v] to jump to any voxel in O(1).
 *
 * Author: Hemant Sharma / AI Assistant
 */

#ifndef INDEXER_CONSOLIDATED_IO_H
#define INDEXER_CONSOLIDATED_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONSOLIDATED_VALS_COLS 16
#define CONSOLIDATED_KEY_COLS 4

/**
 * Per-voxel accumulator: collects variable-length data during indexing.
 * Thread-safe: each voxel is processed by exactly one thread.
 */
typedef struct {
  double *vals;    /* 16 doubles per solution (flat array) */
  size_t *keys;    /* 4 size_t per solution (flat array) */
  int *matchIDs;   /* matched spot IDs (variable per solution, concatenated) */
  int nSolutions;  /* number of solutions found for this voxel */
  int nIDsTotal;   /* total spot IDs stored (sum across all solutions) */
  int capSolutions;/* current capacity for vals/keys arrays */
  int capIDs;      /* current capacity for matchIDs array */
} VoxelAccumulator;

/**
 * Initialize a VoxelAccumulator
 */
static inline void VoxelAccum_init(VoxelAccumulator *acc) {
  acc->nSolutions = 0;
  acc->nIDsTotal = 0;
  acc->capSolutions = 64;
  acc->capIDs = 256;
  acc->vals = (double *)malloc(acc->capSolutions * CONSOLIDATED_VALS_COLS * sizeof(double));
  acc->keys = (size_t *)malloc(acc->capSolutions * CONSOLIDATED_KEY_COLS * sizeof(size_t));
  acc->matchIDs = (int *)malloc(acc->capIDs * sizeof(int));
}

/**
 * Add a solution to the accumulator.
 * @param outArr  16-double record
 * @param keyArr  4-size_t record [SpotID, nMatches, locVals_placeholder, locIDs_placeholder]
 * @param spotIDs array of matched spot IDs
 * @param nIDs    number of matched spot IDs
 */
static inline void VoxelAccum_addSolution(VoxelAccumulator *acc,
                                           const double *outArr,
                                           const size_t *keyArr,
                                           const int *spotIDs, int nIDs) {
  /* Grow vals/keys if needed */
  if (acc->nSolutions >= acc->capSolutions) {
    acc->capSolutions *= 2;
    acc->vals = (double *)realloc(acc->vals, acc->capSolutions * CONSOLIDATED_VALS_COLS * sizeof(double));
    acc->keys = (size_t *)realloc(acc->keys, acc->capSolutions * CONSOLIDATED_KEY_COLS * sizeof(size_t));
  }
  /* Grow IDs if needed */
  while (acc->nIDsTotal + nIDs > acc->capIDs) {
    acc->capIDs *= 2;
    acc->matchIDs = (int *)realloc(acc->matchIDs, acc->capIDs * sizeof(int));
  }
  memcpy(&acc->vals[acc->nSolutions * CONSOLIDATED_VALS_COLS], outArr, CONSOLIDATED_VALS_COLS * sizeof(double));
  /* Store key with solution-local nIDs count in positions [2] and [3] */
  acc->keys[acc->nSolutions * CONSOLIDATED_KEY_COLS + 0] = keyArr[0]; /* SpotID */
  acc->keys[acc->nSolutions * CONSOLIDATED_KEY_COLS + 1] = keyArr[1]; /* nMatches */
  acc->keys[acc->nSolutions * CONSOLIDATED_KEY_COLS + 2] = (size_t)nIDs; /* nIDs for this solution */
  acc->keys[acc->nSolutions * CONSOLIDATED_KEY_COLS + 3] = keyArr[3]; /* reserved */
  memcpy(&acc->matchIDs[acc->nIDsTotal], spotIDs, nIDs * sizeof(int));
  acc->nSolutions++;
  acc->nIDsTotal += nIDs;
}

/**
 * Free a VoxelAccumulator
 */
static inline void VoxelAccum_free(VoxelAccumulator *acc) {
  free(acc->vals);
  free(acc->keys);
  free(acc->matchIDs);
  acc->vals = NULL;
  acc->keys = NULL;
  acc->matchIDs = NULL;
}

/**
 * Write consolidated files from an array of VoxelAccumulators.
 * @param accs      Array of accumulators (index = voxel number relative to startVoxel)
 * @param nVoxels   Total number of voxels in the dataset (nScans*nScans)
 * @param startVoxel First voxel index processed by this block
 * @param endVoxel   One past the last voxel index processed by this block
 * @param outputFolder Path to the Output folder
 */
static inline void WriteConsolidatedFiles(const VoxelAccumulator *accs,
                                           int nVoxels,
                                           int startVoxel, int endVoxel,
                                           const char *outputFolder) {
  char fn[2048];
  FILE *f;
  int32_t nv = nVoxels;
  int nLocal = endVoxel - startVoxel;

  /* Pre-compute counts */
  int32_t *nSolArr = (int32_t *)calloc(nVoxels, sizeof(int32_t));
  int32_t *nIDsArr = (int32_t *)calloc(nVoxels, sizeof(int32_t));
  for (int i = 0; i < nLocal; i++) {
    nSolArr[startVoxel + i] = accs[i].nSolutions;
    nIDsArr[startVoxel + i] = accs[i].nIDsTotal;
  }

  /* Compute byte offsets for each voxel */
  int64_t *valsOff = (int64_t *)calloc(nVoxels, sizeof(int64_t));
  int64_t *keysOff = (int64_t *)calloc(nVoxels, sizeof(int64_t));
  int64_t *idsOff = (int64_t *)calloc(nVoxels, sizeof(int64_t));

  /* Header size = 4 + 4*nVoxels + 8*nVoxels = 4 + 12*nVoxels */
  int64_t headerSize = (int64_t)4 + (int64_t)nVoxels * 4 + (int64_t)nVoxels * 8;
  int64_t valsDataOff = headerSize;
  int64_t keysDataOff = headerSize;
  int64_t idsDataOff = headerSize;

  for (int v = 0; v < nVoxels; v++) {
    valsOff[v] = valsDataOff;
    keysOff[v] = keysDataOff;
    idsOff[v] = idsDataOff;
    valsDataOff += (int64_t)nSolArr[v] * CONSOLIDATED_VALS_COLS * sizeof(double);
    keysDataOff += (int64_t)nSolArr[v] * CONSOLIDATED_KEY_COLS * sizeof(size_t);
    idsDataOff += (int64_t)nIDsArr[v] * sizeof(int);
  }

  /* === IndexBest_all.bin === */
  sprintf(fn, "%s/IndexBest_all.bin", outputFolder);
  f = fopen(fn, "wb");
  if (f) {
    fwrite(&nv, sizeof(int32_t), 1, f);
    fwrite(nSolArr, sizeof(int32_t), nVoxels, f);
    fwrite(valsOff, sizeof(int64_t), nVoxels, f);
    /* Write data: zero-fill for voxels not in this block (multi-block merging is done separately) */
    for (int v = 0; v < nVoxels; v++) {
      if (v >= startVoxel && v < endVoxel) {
        int li = v - startVoxel;
        if (accs[li].nSolutions > 0)
          fwrite(accs[li].vals, CONSOLIDATED_VALS_COLS * sizeof(double), accs[li].nSolutions, f);
      }
      /* else: no data for this voxel in this block, skip */
    }
    fclose(f);
    printf("Wrote %s\n", fn);
  }

  /* === IndexKey_all.bin === */
  sprintf(fn, "%s/IndexKey_all.bin", outputFolder);
  f = fopen(fn, "wb");
  if (f) {
    fwrite(&nv, sizeof(int32_t), 1, f);
    fwrite(nSolArr, sizeof(int32_t), nVoxels, f);
    fwrite(keysOff, sizeof(int64_t), nVoxels, f);
    for (int v = 0; v < nVoxels; v++) {
      if (v >= startVoxel && v < endVoxel) {
        int li = v - startVoxel;
        if (accs[li].nSolutions > 0)
          fwrite(accs[li].keys, CONSOLIDATED_KEY_COLS * sizeof(size_t), accs[li].nSolutions, f);
      }
    }
    fclose(f);
    printf("Wrote %s\n", fn);
  }

  /* === IndexBest_IDs_all.bin === */
  sprintf(fn, "%s/IndexBest_IDs_all.bin", outputFolder);
  f = fopen(fn, "wb");
  if (f) {
    fwrite(&nv, sizeof(int32_t), 1, f);
    fwrite(nIDsArr, sizeof(int32_t), nVoxels, f);
    fwrite(idsOff, sizeof(int64_t), nVoxels, f);
    for (int v = 0; v < nVoxels; v++) {
      if (v >= startVoxel && v < endVoxel) {
        int li = v - startVoxel;
        if (accs[li].nIDsTotal > 0)
          fwrite(accs[li].matchIDs, sizeof(int), accs[li].nIDsTotal, f);
      }
    }
    fclose(f);
    printf("Wrote %s\n", fn);
  }

  free(nSolArr);
  free(nIDsArr);
  free(valsOff);
  free(keysOff);
  free(idsOff);
}

/**
 * Read structure for consolidated voxel data.
 * After loading, use valsData + valsOff[v] / sizeof(double) to get voxel v's data.
 */
typedef struct {
  int32_t nVoxels;
  int32_t *nSolutions;  /* per-voxel solution counts */
  int64_t *offsets;      /* per-voxel byte offsets into data */
  void *data;            /* raw data pointer (mmap'd or malloc'd) */
  size_t dataSize;       /* total data size in bytes */
  size_t headerSize;     /* header size in bytes */
  void *rawMap;          /* full mmap pointer (for munmap) */
  size_t rawMapSize;     /* full mmap size */
} ConsolidatedReader;

/**
 * Open a consolidated file for reading.
 * Maps the header into the reader struct. Data is accessed via offsets.
 */
static inline int ConsolidatedReader_open(ConsolidatedReader *r, const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) return -1;

  /* Read header */
  if (fread(&r->nVoxels, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
  r->nSolutions = (int32_t *)malloc(r->nVoxels * sizeof(int32_t));
  r->offsets = (int64_t *)malloc(r->nVoxels * sizeof(int64_t));
  if (fread(r->nSolutions, sizeof(int32_t), r->nVoxels, f) != (size_t)r->nVoxels) { fclose(f); return -1; }
  if (fread(r->offsets, sizeof(int64_t), r->nVoxels, f) != (size_t)r->nVoxels) { fclose(f); return -1; }
  r->headerSize = (size_t)4 + (size_t)r->nVoxels * 4 + (size_t)r->nVoxels * 8;

  /* Get total file size */
  fseek(f, 0, SEEK_END);
  size_t totalSize = ftell(f);
  r->dataSize = totalSize - r->headerSize;

  /* Read all data into memory */
  r->data = malloc(r->dataSize);
  if (r->data && r->dataSize > 0) {
    fseek(f, r->headerSize, SEEK_SET);
    fread(r->data, 1, r->dataSize, f);
  }
  r->rawMap = NULL;
  r->rawMapSize = 0;
  fclose(f);
  return 0;
}

/**
 * Get pointer to voxel v's data in IndexBest_all.bin.
 * Returns NULL if no solutions for this voxel.
 * @return pointer to first of nSolutions[v] × 16 doubles
 */
static inline const double *ConsolidatedReader_getVals(const ConsolidatedReader *r, int voxNr) {
  if (voxNr < 0 || voxNr >= r->nVoxels || r->nSolutions[voxNr] == 0) return NULL;
  int64_t dataOffset = r->offsets[voxNr] - (int64_t)r->headerSize;
  return (const double *)((const char *)r->data + dataOffset);
}

/**
 * Get pointer to voxel v's data in IndexKey_all.bin.
 * @return pointer to first of nSolutions[v] × 4 size_t
 */
static inline const size_t *ConsolidatedReader_getKeys(const ConsolidatedReader *r, int voxNr) {
  if (voxNr < 0 || voxNr >= r->nVoxels || r->nSolutions[voxNr] == 0) return NULL;
  int64_t dataOffset = r->offsets[voxNr] - (int64_t)r->headerSize;
  return (const size_t *)((const char *)r->data + dataOffset);
}

/**
 * Get pointer to voxel v's IDs in IndexBest_IDs_all.bin.
 * Here nSolutions represents total number of IDs for the voxel.
 * @return pointer to int array
 */
static inline const int *ConsolidatedReader_getIDs(const ConsolidatedReader *r, int voxNr) {
  if (voxNr < 0 || voxNr >= r->nVoxels || r->nSolutions[voxNr] == 0) return NULL;
  int64_t dataOffset = r->offsets[voxNr] - (int64_t)r->headerSize;
  return (const int *)((const char *)r->data + dataOffset);
}

/**
 * Free a ConsolidatedReader
 */
static inline void ConsolidatedReader_close(ConsolidatedReader *r) {
  free(r->nSolutions);
  free(r->offsets);
  free(r->data);
  r->nSolutions = NULL;
  r->offsets = NULL;
  r->data = NULL;
}

#endif /* INDEXER_CONSOLIDATED_IO_H */
