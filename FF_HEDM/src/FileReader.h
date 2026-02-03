// FileReader.h
// Common file reading utilities for CalibrantPanelShiftsOMP and Integrator
// Supports Binary, TIFF, and HDF5 formats
// Created as part of refactoring to support dynamic types and code reuse.

#ifndef FILE_READER_H
#define FILE_READER_H

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Library headers
#include <hdf5.h>
#include <hdf5_hl.h>
#include <tiffio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define FR_SUCCESS 0
#define FR_ERR_OPEN 1
#define FR_ERR_MEMORY 2
#define FR_ERR_READ 3
#define FR_ERR_FORMAT 4
#define FR_ERR_HDF5 5

/**
 * Reads a single frame from a raw binary file.
 * logic matches dTypes 1-5 from original code.
 *
 * @param f         Open file pointer (should be seeked to correct position if
 * needed, but typically fread reads sequentially. For random access, caller
 * seeks.) Actually, original code passed open 'f'.
 * @param dType     1=uint16, 2=double, 3=float, 4=uint32, 5=int32
 * @param NrPixels  Total number of pixels to read (framesize)
 * @param returnArr Output buffer (allocated by caller, size NrPixels *
 * sizeof(double))
 * @return          FR_SUCCESS or error code
 */
int ReadBinaryFrame(FILE *f, int dType, size_t NrPixels, double *returnArr);

/**
 * Reads a single frame from a TIFF file.
 * Handles headers and directory navigation internally.
 *
 * @param filename   Path to TIFF file
 * @param dType      6=uint32, 7=uint8 (bitmap), 9=uint16
 * @param NrPixels   Expected number of pixels (for validation)
 * @param returnArr  Output buffer
 * @param frameIndex Frame/Directory index to read (0-based)
 * @return           FR_SUCCESS or error code
 */
int ReadTiffFrame(const char *filename, int dType, size_t NrPixels,
                  double *returnArr, int frameIndex);

/**
 * Reads a single frame from an HDF5 dataset.
 * dynamically detects type and converts to double.
 *
 * @param filename    Path to H5 file
 * @param datasetName Path to dataset within H5 file
 * @param NrPixels    Expected frame size (cols * rows)
 * @param returnArr   Output buffer
 * @param frameIndex  Frame index (row/slice) to read
 * @return            FR_SUCCESS or error code
 */
int ReadHDF5Frame(const char *filename, const char *datasetName,
                  size_t NrPixels, double *returnArr, int frameIndex);

/**
 * Sums a range of frames from an HDF5 dataset.
 * Used by CalibrantPanelShiftsOMP.
 *
 * @param filename    Path to H5 file
 * @param datasetName Path to dataset within H5 file
 * @param NrPixels    Expected frame size
 * @param returnArr   Output buffer (accumulated sum) - caller must
 * initialize/clear if needed? Function will ADD to existing values or
 * overwrite? Original code overwrote `Image` but subtracted Dark. Here we just
 * return the sum. Caller handles clearing. Actually, let's make it defined:
 * This function OVERWRITES returnArr with the sum.
 * @param skipFrames  Number of initial frames to skip
 * @return            FR_SUCCESS or error code
 */
int SumHDF5Frames(const char *filename, const char *datasetName,
                  size_t NrPixels, double *returnArr, int skipFrames);

/**
 * Helper to get HDF5 dataset dimensions.
 *
 * @param filename    Path to H5 file
 * @param datasetName Path to dataset
 * @param dims        Output array of 3 hsize_t (frames, Y, Z) - cast to int for
 * convenience if needed
 * @return            FR_SUCCESS or error code
 */
int GetHDF5Dimensions(const char *filename, const char *datasetName,
                      hsize_t *dims);

#ifdef __cplusplus
}
#endif

#endif // FILE_READER_H
