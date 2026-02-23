// ZarrReader.h
// Shared Zarr (ZIP + blosc) chunk-reading utilities for MIDAS.
// Replaces the 6-step inline boilerplate (zip_stat → zip_fopen → zip_fread →
// blosc_decompress → free → zip_fclose) that was copy-pasted ~200+ times
// across 8 FF_HEDM Zarr files.
//
// Created as part of Phase 7: Shared Data Fabric refactoring.

#ifndef ZARR_READER_H
#define ZARR_READER_H

#include <blosc2.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zip.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes (negative = error, non-negative = success)
#define ZR_SUCCESS 0
#define ZR_ERR_STAT -1      // zip_stat_index failed
#define ZR_ERR_OPEN -2      // zip_fopen_index failed
#define ZR_ERR_READ -3      // zip_fread short read or failed
#define ZR_ERR_DECOMP -4    // blosc1_decompress failed
#define ZR_ERR_ALLOC -5     // malloc/calloc failed
#define ZR_ERR_NOT_FOUND -6 // Named entry not found in archive

/**
 * Read a blosc-compressed Zarr chunk by archive index and decompress it
 * into a caller-provided buffer.
 *
 * Covers: single scalars (int, double, float), arrays, and strings.
 * For strings, the decompressed data is NOT null-terminated; the return
 * value gives the decompressed size so the caller can add '\0'.
 *
 * @param arch       Open zip_t archive handle
 * @param entryIndex Index of the entry in the zip archive
 * @param dest       Pre-allocated output buffer for decompressed data
 * @param destSize   Size of dest buffer in bytes
 * @return           Decompressed byte count on success (>= 0),
 *                   negative ZR_ERR_* on failure
 */
int ReadZarrChunk(zip_t *arch, int entryIndex, void *dest, size_t destSize);

/**
 * Read a raw (uncompressed) entry from the zip archive by index.
 * Used for .zarray / .zattrs JSON metadata that is NOT blosc-compressed.
 *
 * Allocates a buffer internally; caller must free(*outBuf) when done.
 * The returned buffer is null-terminated for convenient string use.
 *
 * @param arch       Open zip_t archive handle
 * @param entryIndex Index of the entry in the zip archive
 * @param outBuf     Output: pointer to newly allocated, NUL-terminated buffer
 * @param outSize    Output: number of bytes read (excluding NUL terminator)
 * @return           ZR_SUCCESS or negative ZR_ERR_*
 */
int ReadZarrRaw(zip_t *arch, int entryIndex, char **outBuf, size_t *outSize);

/**
 * Read a blosc-compressed Zarr string chunk by archive index.
 * Convenience wrapper around ReadZarrChunk that allocates the output buffer
 * and null-terminates the result.
 *
 * @param arch       Open zip_t archive handle
 * @param entryIndex Index of the entry in the zip archive
 * @param outStr     Output: pointer to newly allocated, NUL-terminated string
 *                   Caller must free(*outStr) when done.
 * @param maxLen     Maximum expected string length (buffer size to allocate)
 * @return           String length on success (>= 0), negative ZR_ERR_* on
 * failure
 */
int ReadZarrString(zip_t *arch, int entryIndex, char **outStr, size_t maxLen);

#ifdef __cplusplus
}
#endif

#endif // ZARR_READER_H
