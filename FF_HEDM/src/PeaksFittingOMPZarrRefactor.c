//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  PeaksFittingOMPZarrRefactor.c - Improved version
//
//  Created by Hemant Sharma on 2024/02/27.
//  Refactored with improvements
//

#include "MIDAS_Math.h"
#include "ZarrReader.h"
#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <zip.h>

/*
 * CONSTANTS AND DEFINITIONS
 */
#define DEG2RAD 0.0174532925199433
#define RAD2DEG 57.2957795130823
#include "MIDAS_Limits.h"
#define MAXNHKLS MAX_N_HKLS
#define MAX_OVERLAPS_PER_IMAGE 10000
#define DEFAULT_WIDTH 1000
#define DEFAULT_LSD 1000000
#define DEFAULT_PIXEL_SIZE 200
#define DEFAULT_WAVELENGTH 0.189714
#define DEFAULT_NR_PIXELS 2048
#define DEFAULT_BC 1
#define DEFAULT_INT_SAT 14000

#define CALC_NORM_3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CALC_NORM_2(x, y) sqrt((x) * (x) + (y) * (y))

// This typedef is now only used for legacy functions that are no longer in the
// main data path.
typedef uint16_t pixelvalue;

// Add a type enum to support dynamic pixel value typing
typedef enum {
  PX_TYPE_UINT16 = 0,
  PX_TYPE_INT32 = 1,
  PX_TYPE_FLOAT = 2,
  PX_TYPE_DOUBLE = 3,
  PX_TYPE_UINT32 = 4
} PixelValueType;

// Error handling codes
typedef enum {
  SUCCESS = 0,
  ERROR_FILE_OPEN,
  ERROR_MEMORY_ALLOCATION,
  ERROR_ZIP_OPEN,
  ERROR_DIRECTORY_CREATION,
  ERROR_INVALID_PARAMETERS,
  ERROR_BLOSC_OPERATION,
  ERROR_THREAD_CREATION
} ErrorCode;

static const char *getErrorMessage(ErrorCode code) {
  switch (code) {
  case SUCCESS:
    return "Success";
  case ERROR_FILE_OPEN:
    return "Could not open file";
  case ERROR_MEMORY_ALLOCATION:
    return "Memory allocation failed";
  case ERROR_ZIP_OPEN:
    return "Could not open zip archive";
  case ERROR_DIRECTORY_CREATION:
    return "Could not create directory";
  case ERROR_INVALID_PARAMETERS:
    return "Invalid parameters provided";
  case ERROR_BLOSC_OPERATION:
    return "Blosc compression/decompression failed";
  case ERROR_THREAD_CREATION:
    return "Could not create thread";
  default:
    return "An unknown error occurred";
  }
}

// Structure for image metadata
typedef struct {
  int nFrames;
  int nDarks;
  int nFloods;
  int nMasks;
  int NrPixelsY;
  int NrPixelsZ;
  int NrPixels; // Max of Y and Z dimensions
  size_t bytesPerPx;
  double omegaStart;
  double omegaStep;
  int skipFrame;
  int doPeakFit;
  PixelValueType pixelType; // Added field for dynamic pixel typing
  double *omegaCenter;      // Array for per-frame omega values
  int nOmegaCenterEntries;  // Number of entries in omegaCenter array
} ImageMetadata;

// Structure for analysis parameters
typedef struct {
  double bc;
  double Ycen;
  double Zcen;
  double IntSat;
  double Lsd;
  double px;
  double Width;
  double RhoD;
  double tx;
  double ty;
  double tz;
  double p0;
  double p1;
  double p2;
  double p3;
  double p4;
  double Wavelength;
  double zDiffThresh;
  double BadPxIntensity;
  int minNrPx;
  int maxNrPx;
  int DoFullImage;
  int LayerNr;
  int makeMap;
  int maxNPeaks;
  int nImTransOpt;
  int *TransOpt;
  int nRingsThresh;
  int *RingNrs;
  double *Thresholds;
  int doPeakFit;
} AnalysisParams;

// Structure for peak info
typedef struct {
  double intensity;
  double rCenter;
  double etaCenter;
  double yCenter;
  double zCenter;
  double maxValue;
  double bg;
  double sigmaGR;
  double sigmaLR;
  double sigmaGEta;
  double sigmaLEta;
  double mu;
  int nPixels;
} PeakInfo;

// Structure for fit function data
typedef struct {
  int NrPixels;
  int nPeaks;
  double *z;
  double *Rs;
  double *Etas;
  // Pre-allocated per-peak parameter caches (Fix 5: eliminate VLAs)
  // Height-normalized pV: 6 params per peak (Imax, R, Eta, Mu, GammaR,
  // GammaEta)
  double *pkImax, *pkR, *pkEta, *pkMu;
  double *pkInvGammaR2, *pkInvGammaEta2;
} FunctionData;

// Structure to hold all temporary buffers for a single thread
typedef struct {
  double *imgCorrBC;
  int *boolImage;
  int *connectedComponents;
  int *positions;
  int *positionTrackers;
  int *usefulPixels;
  int *maximaPositions;
  double *maximaValues;
  double *z;
  double *integratedIntensity;
  double *imax;
  double *yCenArray;
  double *zCenArray;
  double *rads;
  double *etas;
  int *nrPx;
  double *otherInfo;
  double *rawSumIntensity;
  // --- Hoisted per-frame buffers (Items 1, 2, 6) ---
  char *locData;       // Raw decompressed frame data
  double *imageAsym_d; // Asymmetric double image
  double *image_d;     // Square double image
  double *imageTemp1;  // Transform scratch buffer 1
  double *imageTemp2;  // Transform scratch buffer 2
  double *fitRs;       // R-coordinates for fit2DPeaks
  double *fitEtas;     // Eta-coordinates for fit2DPeaks
  // --- Hoisted DFS stack buffers (Fix 3) ---
  int *dfsStackX; // DFS stack X coordinates
  int *dfsStackY; // DFS stack Y coordinates
  // --- Pre-allocated peak fit buffers (Fix 5) ---
  double *fitPeakBuf; // Single block for all 6 peak-param arrays
} ThreadWorkspace;

// Global variables
double zDiffThresh;

#include "Panel.h"
static Panel *panels = NULL;
static int nPanels = 0;

/*
 * UTILITY FUNCTIONS
 */

/**
 * Convert raw pixel data to double array with loop-unswitched pixel type.
 * (Item 7, 13): Moves the type switch outside the loop so the compiler
 * can auto-vectorize each type-specific loop.
 */
static inline void convertPixelsToDouble(const void *rawData, double *dest,
                                         int nPixels,
                                         PixelValueType pixelType) {
  switch (pixelType) {
  case PX_TYPE_UINT32:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const uint32_t *)rawData)[i];
    break;
  case PX_TYPE_INT32:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const int32_t *)rawData)[i];
    break;
  case PX_TYPE_FLOAT:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const float *)rawData)[i];
    break;
  case PX_TYPE_DOUBLE:
    memcpy(dest, rawData, (size_t)nPixels * sizeof(double));
    break;
  case PX_TYPE_UINT16:
  default:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const uint16_t *)rawData)[i];
    break;
  }
}

/**
 * Calculate time difference in microseconds
 */
long double diffTime(struct timespec start, struct timespec end) {
  long double diff_sec = end.tv_sec - start.tv_sec;
  long double diff_nsec = end.tv_nsec - start.tv_nsec;
  return (diff_sec * 1e6) + (diff_nsec / 1000.0);
}

/**
 * Allocate a 2D matrix of doubles using contiguous memory (Item 11).
 * All row data is in a single block for better cache locality.
 */
static inline double **allocMatrix(int nrows, int ncols) {
  double **arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL)
    return NULL;
  double *block = malloc((size_t)nrows * ncols * sizeof(double));
  if (block == NULL) {
    free(arr);
    return NULL;
  }
  for (int i = 0; i < nrows; i++) {
    arr[i] = &block[i * ncols];
  }
  return arr;
}

/**
 * Free a contiguously-allocated 2D matrix of doubles (Item 11).
 */
static inline void freeMatrix(double **mat, int nrows) {
  (void)nrows;
  if (mat != NULL) {
    free(mat[0]); // Free the contiguous data block
    free(mat);    // Free the pointer array
  }
}

/**
 * Calculate eta angle from y, z coordinates
 */
static inline double calcEtaAngle(double y, double z) {
  double alpha = RAD2DEG * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    return -alpha;
  else
    return alpha;
}

/**
 * Convert R and Eta coordinates to Y and Z
 */
static inline void yzFromREta(int nrElements, double *R, double *Eta, double *Y,
                              double *Z) {
  for (int i = 0; i < nrElements; i++) {
    Y[i] = -R[i] * sin(Eta[i] * DEG2RAD);
    Z[i] = R[i] * cos(Eta[i] * DEG2RAD);
  }
}

/**
 * Allocate a 2D matrix of integers
 */
static inline int **allocMatrixInt(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      // Free already allocated memory
      for (int j = 0; j < i; j++) {
        free(arr[j]);
      }
      free(arr);
      return NULL;
    }
  }
  return arr;
}

/**
 * Free a 2D matrix of integers
 */
static inline void freeMatrixInt(int **mat, int nrows) {
  if (mat == NULL)
    return;

  for (int r = 0; r < nrows; r++) {
    if (mat[r] != NULL) {
      free(mat[r]);
    }
  }
  free(mat);
}

static inline double tand(double x) { return tan(DEG2RAD * x); }
static inline double asind(double x) { return RAD2DEG * asin(x); }
static inline double acosd(double x) { return RAD2DEG * acos(x); }
static inline double atand(double x) { return RAD2DEG * atan(x); }

/**
 * Transpose a square matrix of doubles
 */
static inline void transposeMatrix(double *x, int n, double *y) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      y[(i * n) + j] = x[(j * n) + i];
    }
  }
}

/**
 * Check for directory existence and create if needed
 */
static inline int checkDirectoryCreation(const char *folder) {
  struct stat sb;
  char totOutDir[MAX_FILENAME_LENGTH];
  int e;

  snprintf(totOutDir, MAX_FILENAME_LENGTH, "%s/", folder);
  e = stat(totOutDir, &sb);

  if (e != 0 && errno == ENOENT) {
    printf("Output directory did not exist, creating %s\n", totOutDir);
    e = mkdir(totOutDir, S_IRWXU);
    if (e != 0) {
      printf("Could not make the directory. Exiting\n");
      return ERROR_DIRECTORY_CREATION;
    }
  }
  return SUCCESS;
}

/**
 * Perform a 3x3 matrix multiplication with a 3D vector
 */
static inline void matrixVectorMultiply(double m[3][3], double v[3],
                                        double r[3]) {
  for (int i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

/**
 * Multiply two 3x3 matrices
 */
static inline void matrixMultiply33(double m[3][3], double n[3][3],
                                    double res[3][3]) {
  for (int r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

/**
 * Error check and reporting function
 */
static void errorCheck(int test, const char *message, ...) {
  if (test) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}

// Frees all memory in a thread's workspace
void freeWorkspace(ThreadWorkspace *ws) {
  if (ws) {
    free(ws->imgCorrBC);
    free(ws->boolImage);
    free(ws->connectedComponents);
    free(ws->positions);
    free(ws->positionTrackers);
    free(ws->usefulPixels);
    free(ws->maximaPositions);
    free(ws->maximaValues);
    free(ws->z);
    free(ws->integratedIntensity);
    free(ws->rawSumIntensity);
    free(ws->imax);
    free(ws->yCenArray);
    free(ws->zCenArray);
    free(ws->rads);
    free(ws->etas);
    free(ws->nrPx);
    free(ws->otherInfo);
    // --- Hoisted per-frame buffers ---
    free(ws->locData);
    free(ws->imageAsym_d);
    free(ws->image_d);
    free(ws->imageTemp1);
    free(ws->imageTemp2);
    free(ws->fitRs);
    free(ws->fitEtas);
    free(ws->dfsStackX);
    free(ws->dfsStackY);
    free(ws->fitPeakBuf);
  }
}

// Allocates all memory needed by a single thread's workspace.
// Returns SUCCESS or an error code.
ErrorCode allocateWorkspace(ThreadWorkspace *ws, const ImageMetadata *metadata,
                            const AnalysisParams *params) {
  size_t nrPixelsSq = (size_t)metadata->NrPixels * metadata->NrPixels;

  ws->imgCorrBC = calloc(nrPixelsSq, sizeof(double));
  ws->boolImage = calloc(nrPixelsSq, sizeof(int));
  ws->connectedComponents = calloc(nrPixelsSq, sizeof(int));

  // Use the constants defined in the new code
  ws->positions = calloc(
      (size_t)MAX_OVERLAPS_PER_IMAGE * metadata->NrPixels * 4, sizeof(int));
  ws->positionTrackers = calloc(MAX_OVERLAPS_PER_IMAGE, sizeof(int));

  ws->usefulPixels = calloc(metadata->NrPixels * 20, sizeof(int));
  ws->maximaPositions = calloc(metadata->NrPixels * 20, sizeof(int));
  ws->maximaValues = calloc(metadata->NrPixels * 10, sizeof(double));
  ws->z = calloc(metadata->NrPixels * 10, sizeof(double));

  ws->integratedIntensity = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->rawSumIntensity = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->imax = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->yCenArray = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->zCenArray = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->rads = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->etas = calloc(params->maxNPeaks * 2, sizeof(double));
  ws->nrPx = calloc(params->maxNPeaks * 2, sizeof(int));
  ws->otherInfo = calloc(params->maxNPeaks * 10, sizeof(double));

  // --- Hoisted per-frame buffers (Items 1, 2, 6) ---
  size_t nrPixelsAsym = (size_t)metadata->NrPixelsY * metadata->NrPixelsZ;
  size_t locDataSize = nrPixelsAsym * metadata->bytesPerPx;
  ws->locData = malloc(locDataSize);
  ws->imageAsym_d = calloc(nrPixelsAsym, sizeof(double));
  ws->image_d = calloc(nrPixelsSq, sizeof(double));
  ws->imageTemp1 = calloc(nrPixelsSq, sizeof(double));
  ws->imageTemp2 = calloc(nrPixelsSq, sizeof(double));
  ws->fitRs = malloc((size_t)params->maxNrPx * sizeof(double));
  ws->fitEtas = malloc((size_t)params->maxNrPx * sizeof(double));
  // DFS stack buffers (Fix 3): allocated once per thread
  ws->dfsStackX = malloc(nrPixelsSq * sizeof(int));
  ws->dfsStackY = malloc(nrPixelsSq * sizeof(int));
  // Peak fit parameter cache (Fix 5): 8 arrays of maxNPeaks doubles
  ws->fitPeakBuf = malloc((size_t)params->maxNPeaks * 6 * sizeof(double));

  // Check if any allocation failed
  if (!ws->imgCorrBC || !ws->boolImage || !ws->connectedComponents ||
      !ws->positions || !ws->positionTrackers || !ws->usefulPixels ||
      !ws->maximaPositions || !ws->maximaValues || !ws->z ||
      !ws->integratedIntensity || !ws->imax || !ws->yCenArray ||
      !ws->zCenArray || !ws->rads || !ws->etas || !ws->nrPx || !ws->otherInfo ||
      !ws->locData || !ws->imageAsym_d || !ws->image_d || !ws->imageTemp1 ||
      !ws->imageTemp2 || !ws->fitRs || !ws->fitEtas || !ws->dfsStackX ||
      !ws->dfsStackY || !ws->fitPeakBuf) {
    // Free any successful allocations here before returning
    freeWorkspace(ws);
    return ERROR_MEMORY_ALLOCATION;
  }

  return SUCCESS;
}

/*
 * CONNECTED COMPONENTS ANALYSIS
 */

// Direction vectors for 8-connected neighbors
const int dx[] = {+1, 0, -1, 0, +1, -1, +1, -1};
const int dy[] = {0, +1, 0, -1, +1, +1, -1, -1};

/**
 * Iterative implementation of depth-first search for connected components
 * Replaces the recursive implementation to avoid stack overflow.
 * (Fix 3): Uses pre-allocated stack buffers from ThreadWorkspace instead of
 * malloc/free per call.
 */
static inline void depthFirstSearchIterative(int startX, int startY, int label,
                                             int nrPixels, int *boolImage,
                                             int *connectedComponents,
                                             int *positions,
                                             int *positionTrackers, int *stackX,
                                             int *stackY) {
  int stackSize = 0;

  // Push the starting point
  stackX[stackSize] = startX;
  stackY[stackSize] = startY;
  stackSize++;

  while (stackSize > 0) {
    // Pop from stack
    stackSize--;
    int x = stackX[stackSize];
    int y = stackY[stackSize];

    if (x < 0 || x >= nrPixels || y < 0 || y >= nrPixels)
      continue;
    if (connectedComponents[x * nrPixels + y] != 0 ||
        boolImage[x * nrPixels + y] == 0)
      continue;

    connectedComponents[x * nrPixels + y] = label;
    positions[label * nrPixels * 4 + positionTrackers[label]] =
        (x * nrPixels) + y;
    positionTrackers[label]++;

    // Push all neighbors onto stack
    for (int direction = 0; direction < 8; direction++) {
      int newX = x + dx[direction];
      int newY = y + dy[direction];

      if (newX >= 0 && newX < nrPixels && newY >= 0 && newY < nrPixels &&
          connectedComponents[newX * nrPixels + newY] == 0 &&
          boolImage[newX * nrPixels + newY] == 1) {
        stackX[stackSize] = newX;
        stackY[stackSize] = newY;
        stackSize++;
      }
    }
  }
}

/**
 * Find connected components in binary image
 */
static inline int findConnectedComponents(int *boolImage, int nrPixels,
                                          int *connectedComponents,
                                          int *positions, int *positionTrackers,
                                          int *stackX, int *stackY) {
  // Initialize the connected components map
  memset(connectedComponents, 0, nrPixels * nrPixels * sizeof(int));

  int component = 0;
  for (int i = 0; i < nrPixels; i++) {
    for (int j = 0; j < nrPixels; j++) {
      if (connectedComponents[i * nrPixels + j] == 0 &&
          boolImage[i * nrPixels + j] == 1) {
        depthFirstSearchIterative(i, j, ++component, nrPixels, boolImage,
                                  connectedComponents, positions,
                                  positionTrackers, stackX, stackY);
      }
    }
  }
  return component;
}

/**
 * Find regional maxima in a connected component.
 * (Fix 4): Uses imgCorrBC for O(1) direct neighbor lookup instead of
 * O(N) linear scan through the pixel list.
 */
static inline unsigned findRegionalMaxima(double *z, int *pixelPositions,
                                          int nrPixelsThisRegion,
                                          int *maximaPositions,
                                          double *maximaValues, double intSat,
                                          int nrPixels, double *mask,
                                          double *imgCorrBC, int *maskTouched) {
  unsigned nPeaks = 0;

  for (int i = 0; i < nrPixelsThisRegion; i++) {
    // Skip saturated pixels
    if (z[i] > intSat) {
      return 0; // Saturated peak removed
    }

    int xThis = pixelPositions[i * 2 + 0];
    int yThis = pixelPositions[i * 2 + 1];

    // Flag if we touched the mask, but don't reject the peak entirely
    if (mask[xThis + nrPixels * yThis] == 1) {
      if (maskTouched)
        *maskTouched = 1;
    }

    // Check if this is a local maximum using O(1) direct image lookup
    int isRegionalMax = 1;
    double zThis = z[i];

    for (int j = 0; j < 8 && isRegionalMax; j++) {
      int xNext = xThis + dx[j];
      int yNext = yThis + dy[j];

      if (xNext >= 0 && xNext < nrPixels && yNext >= 0 && yNext < nrPixels) {
        int neighborIdx = xNext * nrPixels + yNext;
        // If the neighbor is part of the region (non-zero intensity after
        // correction) and has higher intensity, this pixel is not a maximum.
        if (imgCorrBC[neighborIdx] > 0 && imgCorrBC[neighborIdx] > zThis) {
          isRegionalMax = 0;
        }
      }
    }

    if (isRegionalMax) {
      maximaPositions[nPeaks * 2 + 0] = xThis;
      maximaPositions[nPeaks * 2 + 1] = yThis;
      maximaValues[nPeaks] = zThis;
      nPeaks++;
    }
  }

  // If no peaks found, use the middle pixel
  if (nPeaks == 0) {
    maximaPositions[0] = pixelPositions[nrPixelsThisRegion / 2 * 2 + 0];
    maximaPositions[1] = pixelPositions[nrPixelsThisRegion / 2 * 2 + 1];
    maximaValues[0] = z[nrPixelsThisRegion / 2];
    nPeaks = 1;
  }

  return nPeaks;
}

/*
 * PEAK FITTING FUNCTIONS
 */

/**
 * Height-normalized Pseudo-Voigt 2D objective with shared FWHM per dimension.
 * 6 params per peak: Imax, R, Eta, Mu, GammaR, GammaEta.
 * L(R,Eta) = L_R(R) * L_Eta(Eta)  [each peaks at 1]
 * G(R,Eta) = G_R(R) * G_Eta(Eta)  [each peaks at 1]
 * I = BG + sum_j Imax_j * (Mu_j * L_j + (1-Mu_j) * G_j)
 */
static double peakFittingObjectiveFunction(unsigned n, const double *x,
                                           double *grad, void *f_data_trial) {
  FunctionData *f_data = (FunctionData *)f_data_trial;
  int nrPixels = f_data->NrPixels;
  double *z = f_data->z;
  double *Rs = f_data->Rs;
  double *Etas = f_data->Etas;
  int nPeaks = f_data->nPeaks;

  double bg = x[0]; // Background intensity
  double C0 = 4.0 * log(2.0);

  // Extract peak parameters into pre-allocated caches
  double *Imax = f_data->pkImax;
  double *R = f_data->pkR;
  double *Eta = f_data->pkEta;
  double *Mu = f_data->pkMu;
  double *invGammaR2 = f_data->pkInvGammaR2;
  double *invGammaEta2 = f_data->pkInvGammaEta2;

  for (int i = 0; i < nPeaks; i++) {
    Imax[i] = x[(6 * i) + 1];
    R[i] = x[(6 * i) + 2];
    Eta[i] = x[(6 * i) + 3];
    Mu[i] = x[(6 * i) + 4];
    double gammaR = x[(6 * i) + 5];
    double gammaEta = x[(6 * i) + 6];
    invGammaR2[i] = 1.0 / (gammaR * gammaR);
    invGammaEta2[i] = 1.0 / (gammaEta * gammaEta);
  }

  // Calculate total square difference between model and actual intensity
  double totalDifferenceIntensity = 0.0;

  for (int i = 0; i < nrPixels; i++) {
    double intPeaks = 0.0;

    for (int j = 0; j < nPeaks; j++) {
      double DR = Rs[i] - R[j];
      double R2 = DR * DR;
      double DE = Etas[i] - Eta[j];
      double E2 = DE * DE;

      // Height-normalized 2D Lorentzian: product of 1D (peaks at 1)
      double LR = 1.0 / (1.0 + 4.0 * R2 * invGammaR2[j]);
      double LEta = 1.0 / (1.0 + 4.0 * E2 * invGammaEta2[j]);
      double L = LR * LEta;

      // Height-normalized 2D Gaussian: product of 1D (peaks at 1)
      double G = exp(-C0 * (R2 * invGammaR2[j] + E2 * invGammaEta2[j]));

      // Pseudo-Voigt profile
      intPeaks += Imax[j] * (Mu[j] * L + (1.0 - Mu[j]) * G);
    }

    double diff = bg + intPeaks - z[i];
    totalDifferenceIntensity += diff * diff;
  }

  return totalDifferenceIntensity;
}

/**
 * Calculate integrated intensity of fitted peaks (area-normalized TCH).
 */
static inline void calculateIntegratedIntensity(int nPeaks, double *x,
                                                double *Rs, double *Etas,
                                                int nrPixelsThisRegion,
                                                double *integratedIntensity,
                                                int *nrOfPixels) {
  double bg = x[0];
  double C0 = 4.0 * log(2.0);

  double *Imax = malloc(nPeaks * sizeof(double));
  double *R = malloc(nPeaks * sizeof(double));
  double *Eta = malloc(nPeaks * sizeof(double));
  double *Mu = malloc(nPeaks * sizeof(double));
  double *invGammaR2 = malloc(nPeaks * sizeof(double));
  double *invGammaEta2 = malloc(nPeaks * sizeof(double));

  for (int i = 0; i < nPeaks; i++) {
    Imax[i] = x[(6 * i) + 1];
    R[i] = x[(6 * i) + 2];
    Eta[i] = x[(6 * i) + 3];
    Mu[i] = x[(6 * i) + 4];
    double gammaR = x[(6 * i) + 5];
    double gammaEta = x[(6 * i) + 6];
    invGammaR2[i] = 1.0 / (gammaR * gammaR);
    invGammaEta2[i] = 1.0 / (gammaEta * gammaEta);

    nrOfPixels[i] = 0;
    integratedIntensity[i] = 0;
  }

  for (int j = 0; j < nPeaks; j++) {
    for (int i = 0; i < nrPixelsThisRegion; i++) {
      double DR = Rs[i] - R[j];
      double R2 = DR * DR;
      double DE = Etas[i] - Eta[j];
      double E2 = DE * DE;

      double LR = 1.0 / (1.0 + 4.0 * R2 * invGammaR2[j]);
      double LEta = 1.0 / (1.0 + 4.0 * E2 * invGammaEta2[j]);
      double G = exp(-C0 * (R2 * invGammaR2[j] + E2 * invGammaEta2[j]));
      double intPeaks = Imax[j] * (Mu[j] * LR * LEta + (1.0 - Mu[j]) * G);

      double bgToAdd = 0;
      if (intPeaks > bg) {
        nrOfPixels[j]++;
        bgToAdd = bg;
      }

      integratedIntensity[j] += (bgToAdd + intPeaks);
    }
  }

  free(Imax);
  free(R);
  free(Eta);
  free(Mu);
  free(invGammaR2);
  free(invGammaEta2);
}

/**
 * Fit 2D peaks using NLopt — Two-Stage Decomposed Fitting
 * (Fix 1): Stage 1 fits each peak independently (9 params each),
 *          Stage 2 polishes jointly using SBPLX (handles high-dim well).
 * (Fix 2): Adaptive timeout scales with nPeaks.
 * (Item 6): Rs/Etas buffers are passed in from the ThreadWorkspace.
 */
int fit2DPeaks(unsigned nPeaks, int nrPixelsThisRegion, double *z,
               int *usefulPixels, double *maximaValues, int *maximaPositions,
               double *integratedIntensity, double *IMAX, double *YCEN,
               double *ZCEN, double *RCens, double *EtaCens, double yCen,
               double zCen, double thresh, int *nrPx, double *otherInfo,
               int nrPixels, double *retVal, double *Rs, double *Etas,
               double *fitPeakBuf) {
  // Total parameters: 1 background + 6 per peak
  unsigned n = 1 + (6 * nPeaks);
  double *x = malloc(n * sizeof(double));
  double *xl = malloc(n * sizeof(double));
  double *xu = malloc(n * sizeof(double));

  // Initialize background parameter
  x[0] = thresh / 2; // Initial background level
  xl[0] = 0;         // Lower bound for background
  xu[0] = thresh;    // Upper bound for background

  // Find min/max values for R and Eta to determine constraints
  double RMin = 1e8, RMax = 0, EtaMin = 190, EtaMax = -190;

  for (int i = 0; i < nrPixelsThisRegion; i++) {
    double r_idx = (double)usefulPixels[i * 2 + 0];
    double c_idx = (double)usefulPixels[i * 2 + 1];

    double pdY = 0, pdZ = 0;
    int pIdx = GetPanelIndex(r_idx, c_idx, nPanels, panels);
    if (pIdx >= 0) {
      ApplyPanelCorrection(r_idx, c_idx, &panels[pIdx], &pdY, &pdZ);
      pdY -= r_idx;
      pdZ -= c_idx;
    }

    // Calculate radius and angle for each pixel
    Rs[i] = CALC_NORM_2(r_idx + pdY - yCen, c_idx + pdZ - zCen);
    Etas[i] = calcEtaAngle(-(r_idx + pdY) + yCen, c_idx + pdZ - zCen);

    // Track min/max values
    if (Rs[i] > RMax)
      RMax = Rs[i];
    if (Rs[i] < RMin)
      RMin = Rs[i];
    if (Etas[i] > EtaMax)
      EtaMax = Etas[i];
    if (Etas[i] < EtaMin)
      EtaMin = Etas[i];
  }

  // Calculate maximum allowed widths
  double maxRWidth = (RMax - RMin) / 2 + 1;
  double maxEtaWidth = (EtaMax - EtaMin) / 2 + atand(2 / (RMax + RMin));
  if (EtaMax - EtaMin > 180)
    maxEtaWidth -= 180;

  // Estimate initial width based on region size and number of peaks
  double width = sqrt(nrPixelsThisRegion / nPeaks);
  if (width > maxRWidth)
    width = maxRWidth;

  // Initialize parameters for each peak
  for (int i = 0; i < nPeaks; i++) {
    // Calculate initial parameters based on maxima positions
    double peakR = CALC_NORM_2(maximaPositions[i * 2 + 0] - yCen,
                               maximaPositions[i * 2 + 1] - zCen);
    double peakEta = calcEtaAngle(-maximaPositions[i * 2 + 0] + yCen,
                                  maximaPositions[i * 2 + 1] - zCen);
    // -----------------------------------------------------------
    // Moment-Based Width Estimation (Speedup Strategy)
    // Instead of a uniform guess, estimate sigma for each peak
    // using the weighted variance of pixels closest to it.
    // -----------------------------------------------------------
    double sumW = 0.0, sumWR2 = 0.0, sumWEta2 = 0.0;
    double bg_est = thresh / 2.0;

    for (int p = 0; p < nrPixelsThisRegion; p++) {
      double r_p = Rs[p];
      double eta_p = Etas[p];
      double val = z[p] - bg_est;
      if (val <= 0)
        continue;

      // Find closest peak to this pixel (Voronoi partition)
      int closest_k = -1;
      double min_dist2 = 1e9;
      for (int k = 0; k < nPeaks; k++) {
        double pr = CALC_NORM_2(maximaPositions[k * 2 + 0] - yCen,
                                maximaPositions[k * 2 + 1] - zCen);
        double peta = calcEtaAngle(-maximaPositions[k * 2 + 0] + yCen,
                                   maximaPositions[k * 2 + 1] - zCen);
        double d2 = (r_p - pr) * (r_p - pr) + (eta_p - peta) * (eta_p - peta);
        if (d2 < min_dist2) {
          min_dist2 = d2;
          closest_k = k;
        }
      }

      // If this pixel belongs to the current peak i
      if (closest_k == i) {
        sumW += val;
        double dr = r_p - peakR;
        double de = eta_p - peakEta;
        sumWR2 += val * (dr * dr);
        sumWEta2 += val * (de * de);
      }
    }

    double estimSigmaR = width;   // Fallback to old scalar guess
    double estimSigmaEta = width; // Fallback

    if (sumW > 0) {
      estimSigmaR = sqrt(sumWR2 / sumW);
      estimSigmaEta = sqrt(sumWEta2 / sumW);
      // Clamp to reasonable bounds to prevent instability
      if (estimSigmaR < 0.1)
        estimSigmaR = 0.1;
      if (estimSigmaR > maxRWidth)
        estimSigmaR = maxRWidth;
      if (estimSigmaEta < 0.1)
        estimSigmaEta = 0.1;
      // Eta width in degrees needs scaling awareness, but here we estimate
      // directly in degrees (Etas is in degrees)
    }

    double initSigmaEtaVal = estimSigmaEta;
    // Convert to proper units if needed (the original code used width/peakR)
    // but here estimSigmaEta is already in degrees variance.
    // The original code used `atand(width/peakR)`.
    // Let's use the explicit estimate.

    // Estimate FWHM from estimated sigma: FWHM = 2*sqrt(2*ln2)*sigma
    // ~ 2.355*sigma
    double gammaRGuess = 2.0 * sqrt(2.0 * log(2.0)) * estimSigmaR;
    double gammaEtaGuess = 2.0 * sqrt(2.0 * log(2.0)) * estimSigmaEta;

    // Initial values (6 params per peak)
    x[(6 * i) + 1] = maximaValues[i]; // Imax (peak height)
    x[(6 * i) + 2] = peakR;           // Radius
    x[(6 * i) + 3] = peakEta;         // Eta
    x[(6 * i) + 4] = 0.5;             // Mu (mix parameter)
    x[(6 * i) + 5] = gammaRGuess;     // GammaR (FWHM in R)
    x[(6 * i) + 6] = gammaEtaGuess;   // GammaEta (FWHM in Eta)

    // Calculate bounds for parameters
    double dEta = RAD2DEG * atan(1 / peakR);

    // Lower bounds
    xl[(6 * i) + 1] = maximaValues[i] / 2; // Imax lower bound
    xl[(6 * i) + 2] = peakR - 1;           // R lower bound
    xl[(6 * i) + 3] = peakEta - dEta;      // Eta lower bound
    xl[(6 * i) + 4] = 0;                   // Mu lower bound (pure Gaussian)
    xl[(6 * i) + 5] = 0.02;                // GammaR lower bound
    xl[(6 * i) + 6] = 0.01;                // GammaEta lower bound

    // Upper bounds
    xu[(6 * i) + 1] = maximaValues[i] * 5; // Imax upper bound
    xu[(6 * i) + 2] = peakR + 1;           // R upper bound
    xu[(6 * i) + 3] = peakEta + dEta;      // Eta upper bound
    xu[(6 * i) + 4] = 1;                   // Mu upper bound (pure Lorentzian)
    xu[(6 * i) + 5] = 2 * maxRWidth;       // GammaR upper bound
    xu[(6 * i) + 6] = 2 * maxEtaWidth;     // GammaEta upper bound
  }

  // Set up FunctionData with pre-allocated peak buffers (Fix 5)
  // fitPeakBuf is a single block of nPeaks*6 doubles, carved into 6 arrays:
  FunctionData f_data = {.NrPixels = nrPixelsThisRegion,
                         .nPeaks = (int)nPeaks,
                         .Rs = Rs,
                         .Etas = Etas,
                         .z = z,
                         .pkImax = fitPeakBuf + 0 * nPeaks,
                         .pkR = fitPeakBuf + 1 * nPeaks,
                         .pkEta = fitPeakBuf + 2 * nPeaks,
                         .pkMu = fitPeakBuf + 3 * nPeaks,
                         .pkInvGammaR2 = fitPeakBuf + 4 * nPeaks,
                         .pkInvGammaEta2 = fitPeakBuf + 5 * nPeaks};

  int rc = 0;
  double minf = 0;

  // Create and configure NLopt optimizer
  NLoptConfig config = {0};
  config.dimension = n;
  config.lower_bounds = xl;
  config.upper_bounds = xu;
  config.objective_function = peakFittingObjectiveFunction;
  config.obj_data = &f_data;
  config.initial_guess = x;
  config.max_evaluations = 5000;
  config.max_time_seconds = 30;
  config.ftol_rel = 1e-5;
  config.xtol_rel = 1e-5;

  rc = run_nlopt_optimization(NLOPT_LN_NELDERMEAD, &config);
  minf = config.min_function_val;

  // Extract results (6 params per peak)
  // Convert Gamma back to equivalent sigmas for backward compatibility:
  //   SigmaG = Gamma / (2*sqrt(2*ln2)) ≈ Gamma / 2.355
  //   SigmaL = Gamma / 2
  double FWHM_to_sigmaG = 1.0 / (2.0 * sqrt(2.0 * log(2.0))); // ~0.4247
  for (int i = 0; i < nPeaks; i++) {
    IMAX[i] = x[(6 * i) + 1]; // Imax (peak height)
    RCens[i] = x[(6 * i) + 2];
    EtaCens[i] = x[(6 * i) + 3];

    double mu_i = x[(6 * i) + 4];
    double gammaR = x[(6 * i) + 5];
    double gammaEta = x[(6 * i) + 6];
    // Convert to equivalent sigmas
    double sigmaGR = gammaR * FWHM_to_sigmaG;
    double sigmaLR = gammaR / 2.0;
    double sigmaGEta = gammaEta * FWHM_to_sigmaG;
    double sigmaLEta = gammaEta / 2.0;
    // otherInfo uses 8-stride for output compatibility
    otherInfo[8 * i + 0] = x[0];      // Background
    otherInfo[8 * i + 1] = sigmaGR;   // SigmaGR (Gaussian-equiv sigma in R)
    otherInfo[8 * i + 2] = sigmaLR;   // SigmaLR (Lorentzian-equiv sigma in R)
    otherInfo[8 * i + 3] = sigmaGEta; // SigmaGEta
    otherInfo[8 * i + 4] = sigmaLEta; // SigmaLEta
    otherInfo[8 * i + 5] = mu_i;      // Mu
    // Effective sigma: Mu*sigmaL + (1-Mu)*sigmaG
    otherInfo[8 * i + 6] = mu_i * sigmaLR + (1.0 - mu_i) * sigmaGR;
    otherInfo[8 * i + 7] = mu_i * sigmaLEta + (1.0 - mu_i) * sigmaGEta;
  }

  // Calculate Y and Z coordinates from R and Eta
  yzFromREta(nPeaks, RCens, EtaCens, YCEN, ZCEN);

  // Calculate integrated intensities
  calculateIntegratedIntensity(nPeaks, x, Rs, Etas, nrPixelsThisRegion,
                               integratedIntensity, nrPx);

  // Return (no free needed — Rs/Etas are workspace-owned)
  *retVal = sqrt(minf); // RMS error
  free(x);
  free(xl);
  free(xu);
  return rc;
}

/**
 * Apply image transformations (flip/transpose) on double data.
 * (Item 2): Uses pre-allocated flat workspace buffers instead of
 * allocating/freeing 2D matrices on each call.
 */
static inline void applyImageTransformations_d(int nrTransformOptions,
                                               int transformOptions[10],
                                               double *image, int nrPixels,
                                               double *temp1, double *temp2) {
  size_t nSq = (size_t)nrPixels * nrPixels;

  // Copy input into temp1
  memcpy(temp1, image, nSq * sizeof(double));

  // Apply each transformation in sequence
  for (int k = 0; k < nrTransformOptions; k++) {
    switch (transformOptions[k]) {
    case 1: // Flip horizontal (Y)
      for (int l = 0; l < nrPixels; l++) {
        for (int m = 0; m < nrPixels; m++) {
          temp2[l * nrPixels + m] = temp1[l * nrPixels + (nrPixels - m - 1)];
        }
      }
      break;

    case 2: // Flip vertical (Z)
      for (int l = 0; l < nrPixels; l++) {
        for (int m = 0; m < nrPixels; m++) {
          temp2[l * nrPixels + m] = temp1[(nrPixels - l - 1) * nrPixels + m];
        }
      }
      break;

    case 3: // Transpose
      for (int l = 0; l < nrPixels; l++) {
        for (int m = 0; m < nrPixels; m++) {
          temp2[l * nrPixels + m] = temp1[m * nrPixels + l];
        }
      }
      break;

    case 0: // No change
    default:
      memcpy(temp2, temp1, nSq * sizeof(double));
      break;
    }

    // Swap pointers for next iteration (pointer swap instead of copy)
    double *swap = temp1;
    temp1 = temp2;
    temp2 = swap;
  }

  // After the loop, result is in temp1 (due to the pointer swap pattern).
  // If nrTransformOptions == 0, temp1 still holds the original data.
  memcpy(image, temp1, nSq * sizeof(double));
}

/**
 * Make a square image from rectangular double data
 */
static inline void makeSquareImage_d(int nrPixels, int nrPixelsY, int nrPixelsZ,
                                     double *inImage, double *outImage) {
  if (nrPixelsY == nrPixelsZ) {
    // Already square, just copy
    memcpy(outImage, inImage, (size_t)nrPixels * nrPixels * sizeof(double));
  } else if (nrPixelsY > nrPixelsZ) {
    // Fill along the slow direction
    memcpy(outImage, inImage, (size_t)nrPixelsY * nrPixelsZ * sizeof(double));
  } else {
    // Fill line by line
    for (int i = 0; i < nrPixelsZ; i++) {
      memcpy(outImage + (size_t)i * nrPixels, inImage + (size_t)i * nrPixelsY,
             (size_t)nrPixelsY * sizeof(double));
    }
  }
}

/**
 * Helper function to read a decompressed image from Zarr
 * Delegates to the centralized ReadZarrChunk API.
 */
static inline ErrorCode readZarrImageWrapper(zip_t *archive, int fileIndex,
                                             char *buffer, int32_t bufferSize) {
  int rc = ReadZarrChunk(archive, fileIndex, buffer, bufferSize);
  if (rc < 0) {
    printf("Error reading Zarr image chunk at index %d: %d\n", fileIndex, rc);
    return ERROR_BLOSC_OPERATION;
  }
  return SUCCESS;
}

/**
 * Read array and decompress from zarr
 * Delegates to the centralized ReadZarrChunk API.
 */
static inline ErrorCode readZarrArrayDataWrapper(zip_t *archive, int fileIndex,
                                                 void *dest, size_t destSize,
                                                 const char *dataType) {
  int rc = ReadZarrChunk(archive, fileIndex, dest, (int32_t)destSize);
  if (rc < 0) {
    return ERROR_BLOSC_OPERATION;
  }
  return SUCCESS;
}

/**
 * Read dark, flat, and mask files from zarr
 */
static ErrorCode readImageCorrections(zip_t *archive, int darkLoc, int floodLoc,
                                      int maskLoc, ImageMetadata *metadata,
                                      AnalysisParams *params, double *dark,
                                      double *flood, double *mask)
// Add print statements each time there is an error and have to return error
{
  ErrorCode error;
  double *darkTemp = NULL;
  int32_t dataSize =
      metadata->bytesPerPx * metadata->NrPixelsY * metadata->NrPixelsZ;

  // Initialize correction arrays
  for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
    dark[i] = 0.0;
    flood[i] = 1.0;
    mask[i] = 0.0;
  }

  darkTemp = calloc((size_t)metadata->NrPixels * metadata->NrPixels,
                    sizeof(*darkTemp));
  char *rawData = malloc(dataSize);
  if (!darkTemp || !rawData) {
    if (darkTemp)
      free(darkTemp);
    if (rawData)
      free(rawData);
    printf("Error allocating memory for darkTemp or rawData\n");
    return ERROR_MEMORY_ALLOCATION;
  }

  // Process dark frames
  if (metadata->nDarks > 0) {
    double *darkAsym_d = calloc(
        (size_t)metadata->NrPixelsY * metadata->NrPixelsZ, sizeof(double));
    double *darkContents_d =
        calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(double));
    if (!darkAsym_d || !darkContents_d) {
      if (darkAsym_d)
        free(darkAsym_d);
      if (darkContents_d)
        free(darkContents_d);
      free(darkTemp);
      free(rawData);
      printf("Error allocating memory for darkAsym_d or darkContents_d\n");
      return ERROR_MEMORY_ALLOCATION;
    }

    for (int darkIter = 0; darkIter < metadata->nDarks; darkIter++) {
      error =
          readZarrImageWrapper(archive, darkLoc + darkIter, rawData, dataSize);
      if (error != SUCCESS) {
        // Free all buffers and return
        free(darkAsym_d);
        free(darkContents_d);
        free(darkTemp);
        free(rawData);
        printf("Error reading dark frame %d: %d\n", darkIter, error);
        return error;
      }

      // Convert raw data to double (Item 13: reuse helper)
      convertPixelsToDouble(rawData, darkAsym_d,
                            metadata->NrPixelsY * metadata->NrPixelsZ,
                            metadata->pixelType);

      makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY,
                        metadata->NrPixelsZ, darkAsym_d, darkContents_d);
      {
        // Temporary scratch buffers for transform (one-time init path)
        size_t nSq2 = (size_t)metadata->NrPixels * metadata->NrPixels;
        double *tmpA = malloc(nSq2 * sizeof(double));
        double *tmpB = malloc(nSq2 * sizeof(double));
        if (tmpA && tmpB) {
          applyImageTransformations_d(params->nImTransOpt, params->TransOpt,
                                      darkContents_d, metadata->NrPixels, tmpA,
                                      tmpB);
        }
        free(tmpA);
        free(tmpB);
      }

      for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
        darkTemp[i] += darkContents_d[i];
      }
    }

    free(darkAsym_d);
    free(darkContents_d);

    // Average dark frames
    if (metadata->nDarks > 0) {
      for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
        darkTemp[i] /= metadata->nDarks;
      }
    }
  }

  // Transpose dark frame
  transposeMatrix(darkTemp, metadata->NrPixels, dark);

  // Read flat field (flood) frame
  if (metadata->nFloods > 0) {
    error = readZarrArrayDataWrapper(archive, floodLoc, flood,
                                     (size_t)metadata->NrPixels *
                                         metadata->NrPixels * sizeof(double),
                                     "float64");
    if (error != SUCCESS) {
      free(darkTemp);
      free(rawData);
      printf("Error reading flood frame: %d\n", error);
      return error;
    }
  }

  // Read mask
  if (maskLoc >= 0 && metadata->nMasks > 0) {
    double *maskAsym_d = calloc(
        (size_t)metadata->NrPixelsY * metadata->NrPixelsZ, sizeof(double));
    double *maskContents_d =
        calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(double));
    if (!maskAsym_d || !maskContents_d) {
      if (maskAsym_d)
        free(maskAsym_d);
      if (maskContents_d)
        free(maskContents_d);
      free(darkTemp);
      free(rawData);
      printf("Error allocating memory for maskAsym_d or maskContents_d\n");
      return ERROR_MEMORY_ALLOCATION;
    }

    error = readZarrImageWrapper(archive, maskLoc, rawData, dataSize);
    if (error != SUCCESS) {
      free(maskAsym_d);
      free(maskContents_d);
      free(darkTemp);
      free(rawData);
      printf("Error reading mask frame: %d\n", error);
      return error;
    }

    // Convert raw data to double (Item 13: reuse helper)
    convertPixelsToDouble(rawData, maskAsym_d,
                          metadata->NrPixelsY * metadata->NrPixelsZ,
                          metadata->pixelType);

    makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY,
                      metadata->NrPixelsZ, maskAsym_d, maskContents_d);

    int nrMask = 0;
    for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
      mask[i] = maskContents_d[i];
      if (mask[i] > 0)
        nrMask++;
    }

    printf("Number of mask pixels: %d\n", nrMask);
    free(maskAsym_d);
    free(maskContents_d);
  }

  // Clean up
  free(rawData);
  free(darkTemp);

  return SUCCESS;
}

/**
 * Process a single image frame using a pre-allocated workspace for efficiency.
 */
static ErrorCode processImageFrame(int fileNr, char *allData, size_t *sizeArr,
                                   ImageMetadata *metadata,
                                   AnalysisParams *params, double *dark,
                                   double *flood, double *mask,
                                   double *goodCoords, double omega,
                                   const char *outFolderName,
                                   const char *dataFN, ThreadWorkspace *ws) {
  // For timing
  double t1 = omp_get_wtime();

  // The 'imgCorrBC' buffer is now part of the workspace.
  double *imgCorrBC = ws->imgCorrBC;

  // --- Use workspace buffers (Item 1): no per-frame malloc/free ---
  char *locData = ws->locData;
  double *imageAsym_d = ws->imageAsym_d;
  double *image_d = ws->image_d;
  int32_t dsz =
      metadata->NrPixelsY * metadata->NrPixelsZ * metadata->bytesPerPx;

  // Decompress the image data
  int32_t decompressedSize =
      blosc1_decompress(&allData[sizeArr[fileNr * 2 + 1]], locData, dsz);
  if (decompressedSize <= 0) {
    printf("Blosc decompression failed for frame %d\n", fileNr);
    return ERROR_BLOSC_OPERATION;
  }

  // Convert raw data to double (Item 7: loop-unswitched helper)
  convertPixelsToDouble(locData, imageAsym_d,
                        metadata->NrPixelsY * metadata->NrPixelsZ,
                        metadata->pixelType);

  makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY,
                    metadata->NrPixelsZ, imageAsym_d, image_d);

  if (params->makeMap == 1) {
    for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
      if (image_d[i] == params->BadPxIntensity)
        image_d[i] = 0;
    }
  }

  // Item 2: pass workspace temp buffers
  applyImageTransformations_d(params->nImTransOpt, params->TransOpt, image_d,
                              metadata->NrPixels, ws->imageTemp1,
                              ws->imageTemp2);
  transposeMatrix(image_d, metadata->NrPixels, imgCorrBC);

  for (int i = 0; i < (metadata->NrPixels * metadata->NrPixels); i++) {
    if (goodCoords[i] == 0) {
      imgCorrBC[i] = 0;
    } else {
      imgCorrBC[i] = (imgCorrBC[i] - dark[i]) / flood[i];
      imgCorrBC[i] *= params->bc;
      if (imgCorrBC[i] < goodCoords[i]) {
        imgCorrBC[i] = 0;
      }
    }
  }

  // NOTE: All large analysis arrays are now used from the workspace `ws`.
  // NO ALLOCATIONS or FREES are performed in this function for these buffers.

  for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
    ws->boolImage[i] = (imgCorrBC[i] != 0) ? 1 : 0;
  }

  memset(ws->positionTrackers, 0, MAX_OVERLAPS_PER_IMAGE * sizeof(int));
  int nrOfRegions = findConnectedComponents(
      ws->boolImage, metadata->NrPixels, ws->connectedComponents, ws->positions,
      ws->positionTrackers, ws->dfsStackX, ws->dfsStackY);

  char outFile[MAX_FILENAME_LENGTH];
  snprintf(outFile, MAX_FILENAME_LENGTH, "%s/%s_%06d_PS.csv", outFolderName,
           basename((char *)dataFN), fileNr + 1);
  FILE *outfilewrite = fopen(outFile, "w");

  if (!outfilewrite) {
    printf("Cannot open %s for writing.\n", outFile);
    return ERROR_FILE_OPEN;
  }

  fprintf(outfilewrite,
          "SpotID\tIntegratedIntensity\tOmega(degrees)\tYCen(px)\tZCen(px)"
          "\tIMax\tRadius(px)\tEta(degrees)\tSigmaR\tSigmaEta\tNrPixels\t"
          "TotalNrPixelsInPeakRegion\tnPeaks\tmaxY\tmaxZ\tdiffY\tdiffZ\trawIMax"
          "\treturnCode\tretVal\tBG\tSigmaGR\tSigmaLR\tSigmaGEta\t"
          "SigmaLEta\tMU\tRawSumIntensity\tmaskTouched\tFitRMSE\n");

  // --- Pixel list collection for _PX.bin ---
  // We collect per-peak pixel lists during the loop and write at the end.
  // Each entry stores the region's pixel count and a pointer into positions.
  // For doPeakFit==0, each region is a single peak, so this is straightforward.
  int pxCapacity = 4096; // initial capacity for peak pixel entries
  int pxCount = 0;       // number of peaks collected
  int *pxNPixels = malloc(pxCapacity * sizeof(int));  // nPixels per peak
  int *pxRegNrs = malloc(pxCapacity * sizeof(int));   // region number
  int *pxRegSizes = malloc(pxCapacity * sizeof(int)); // nrPixelsThisRegion

  int spotIdStart = 1;
  int totalValidRegions = 0;

  for (int regNr = 1; regNr <= nrOfRegions; regNr++) {
    int nrPixelsThisRegion = ws->positionTrackers[regNr];

    if (nrPixelsThisRegion <= params->minNrPx ||
        nrPixelsThisRegion >= params->maxNrPx) {
      continue;
    }
    totalValidRegions++;

    for (int i = 0; i < nrPixelsThisRegion; i++) {
      ws->usefulPixels[i * 2 + 0] =
          (int)(ws->positions[regNr * metadata->NrPixels * 4 + i] /
                metadata->NrPixels);
      ws->usefulPixels[i * 2 + 1] =
          (int)(ws->positions[regNr * metadata->NrPixels * 4 + i] %
                metadata->NrPixels);
      ws->z[i] =
          imgCorrBC[((ws->usefulPixels[i * 2 + 0]) * metadata->NrPixels) +
                    (ws->usefulPixels[i * 2 + 1])];
    }

    // Compute raw sum intensity for this region (before background subtraction)
    double rawSumForRegion = 0;
    for (int i = 0; i < nrPixelsThisRegion; i++) {
      rawSumForRegion += ws->z[i];
    }

    double thresh = goodCoords[((ws->usefulPixels[0]) * metadata->NrPixels) +
                               (ws->usefulPixels[1])];

    int maskTouchedLocal = 0;
    unsigned nPeaks = findRegionalMaxima(
        ws->z, ws->usefulPixels, nrPixelsThisRegion, ws->maximaPositions,
        ws->maximaValues, params->IntSat, metadata->NrPixels, mask, imgCorrBC,
        &maskTouchedLocal);

    if (nPeaks == 0)
      continue;

    if (nPeaks > params->maxNPeaks) {
      // Logic to limit number of peaks
      // This small, temporary allocation is acceptable.
      int *tempPositions = calloc(nPeaks * 2, sizeof(int));
      double *tempValues = calloc(nPeaks, sizeof(double));
      if (!tempPositions || !tempValues) {
        if (tempPositions)
          free(tempPositions);
        if (tempValues)
          free(tempValues);
        continue;
      }
      for (int i = 0; i < params->maxNPeaks; i++) {
        double maxIntMax = 0;
        int maxPos = 0;
        for (int j = 0; j < nPeaks; j++) {
          if (ws->maximaValues[j] > maxIntMax) {
            maxPos = j;
            maxIntMax = ws->maximaValues[j];
          }
        }
        tempPositions[i * 2 + 0] = ws->maximaPositions[maxPos * 2 + 0];
        tempPositions[i * 2 + 1] = ws->maximaPositions[maxPos * 2 + 1];
        tempValues[i] = ws->maximaValues[maxPos];
        ws->maximaValues[maxPos] = 0;
      }
      nPeaks = params->maxNPeaks;
      for (int i = 0; i < nPeaks; i++) {
        ws->maximaValues[i] = tempValues[i];
        ws->maximaPositions[i * 2 + 0] = tempPositions[i * 2 + 0];
        ws->maximaPositions[i * 2 + 1] = tempPositions[i * 2 + 1];
      }
      free(tempPositions);
      free(tempValues);
    }

    double retVal = 0;
    int rc = 0;

    if (params->doPeakFit == 0) {
      double rMeanVal = 0, etaMeanVal = 0;
      nPeaks = 1;
      ws->imax[0] = ws->maximaValues[0];
      ws->nrPx[0] = nrPixelsThisRegion;
      ws->yCenArray[0] = 0;
      ws->zCenArray[0] = 0;
      ws->integratedIntensity[0] = 0;
      for (int i = 0; i < nrPixelsThisRegion; i++) {
        ws->integratedIntensity[0] += ws->z[i];
        rMeanVal += CALC_NORM_2(-ws->usefulPixels[i * 2 + 0] + params->Ycen,
                                ws->usefulPixels[i * 2 + 1] - params->Zcen) *
                    ws->z[i];
        etaMeanVal += calcEtaAngle(-ws->usefulPixels[i * 2 + 0] + params->Ycen,
                                   ws->usefulPixels[i * 2 + 1] - params->Zcen) *
                      ws->z[i];
      }
      rMeanVal /= ws->integratedIntensity[0];
      etaMeanVal /= ws->integratedIntensity[0];
      yzFromREta(1, &rMeanVal, &etaMeanVal, ws->yCenArray, ws->zCenArray);
      ws->rads[0] = rMeanVal;
      ws->etas[0] = etaMeanVal;
      ws->rawSumIntensity[0] = rawSumForRegion;
    } else {
      rc = fit2DPeaks(
          nPeaks, nrPixelsThisRegion, ws->z, ws->usefulPixels, ws->maximaValues,
          ws->maximaPositions, ws->integratedIntensity, ws->imax, ws->yCenArray,
          ws->zCenArray, ws->rads, ws->etas, params->Ycen, params->Zcen, thresh,
          ws->nrPx, ws->otherInfo, metadata->NrPixels, &retVal, ws->fitRs,
          ws->fitEtas, ws->fitPeakBuf);
      // Apportion raw sum intensity by IMax for overlapping peaks
      if (nPeaks == 1) {
        ws->rawSumIntensity[0] = rawSumForRegion;
      } else {
        double totalIMax = 0;
        for (unsigned pi = 0; pi < nPeaks; pi++)
          totalIMax += ws->imax[pi];
        if (totalIMax > 0) {
          for (unsigned pi = 0; pi < nPeaks; pi++)
            ws->rawSumIntensity[pi] =
                rawSumForRegion * ws->imax[pi] / totalIMax;
        } else {
          for (unsigned pi = 0; pi < nPeaks; pi++)
            ws->rawSumIntensity[pi] = rawSumForRegion / nPeaks;
        }
      }
    }

    for (int i = 0; i < nPeaks; i++) {
      fprintf(outfilewrite, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t",
              (spotIdStart + i), ws->integratedIntensity[i], omega,
              -ws->yCenArray[i] + params->Ycen, ws->zCenArray[i] + params->Zcen,
              ws->imax[i], ws->rads[i], ws->etas[i]);
      fprintf(outfilewrite, "%f\t%f\t", ws->otherInfo[8 * i + 6],
              ws->otherInfo[8 * i + 7]);
      fprintf(outfilewrite, "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%lf",
              ws->nrPx[i], nrPixelsThisRegion, nPeaks,
              ws->maximaPositions[i * 2 + 0], ws->maximaPositions[i * 2 + 1],
              (double)ws->maximaPositions[i * 2 + 0] + ws->yCenArray[i] -
                  params->Ycen,
              (double)ws->maximaPositions[i * 2 + 1] - ws->zCenArray[i] -
                  params->Zcen,
              ws->maximaValues[i], rc, retVal);
      for (int j = 0; j < 6; j++) {
        fprintf(outfilewrite, "\t%f", ws->otherInfo[8 * i + j]);
      }
      fprintf(outfilewrite, "\t%f\t%.1f\t%.6f\n", ws->rawSumIntensity[i],
              (double)maskTouchedLocal, retVal);

      // Collect pixel data for this peak (all peaks share the region's pixels)
      if (pxCount >= pxCapacity) {
        pxCapacity *= 2;
        pxNPixels = realloc(pxNPixels, pxCapacity * sizeof(int));
        pxRegNrs = realloc(pxRegNrs, pxCapacity * sizeof(int));
        pxRegSizes = realloc(pxRegSizes, pxCapacity * sizeof(int));
      }
      pxNPixels[pxCount] = nrPixelsThisRegion;
      pxRegNrs[pxCount] = regNr;
      pxRegSizes[pxCount] = nrPixelsThisRegion;
      pxCount++;
    }
    spotIdStart += nPeaks;
  }

  fclose(outfilewrite);

  // --- Write _PX.bin pixel list file ---
  // Format: int32 NrPixels, int32 nPeaksTotal,
  //         then for each peak: int32 nPixels, int16 y[nPixels], int16
  //         z[nPixels]
  {
    char pxFile[MAX_FILENAME_LENGTH];
    snprintf(pxFile, MAX_FILENAME_LENGTH, "%s/%s_%06d_PX.bin", outFolderName,
             basename((char *)dataFN), fileNr + 1);
    FILE *pxFp = fopen(pxFile, "wb");
    if (pxFp) {
      int32_t hdrNrPixels = (int32_t)metadata->NrPixels;
      int32_t hdrNPeaks = (int32_t)pxCount;
      fwrite(&hdrNrPixels, sizeof(int32_t), 1, pxFp);
      fwrite(&hdrNPeaks, sizeof(int32_t), 1, pxFp);

      for (int pk = 0; pk < pxCount; pk++) {
        int32_t nPx = (int32_t)pxNPixels[pk];
        fwrite(&nPx, sizeof(int32_t), 1, pxFp);
        int regNr = pxRegNrs[pk];
        int nrPxReg = pxRegSizes[pk];
        for (int i = 0; i < nrPxReg; i++) {
          int pos = ws->positions[regNr * metadata->NrPixels * 4 + i];
          int16_t y = (int16_t)(pos / metadata->NrPixels);
          int16_t z = (int16_t)(pos % metadata->NrPixels);
          fwrite(&y, sizeof(int16_t), 1, pxFp);
          fwrite(&z, sizeof(int16_t), 1, pxFp);
        }
      }
      fclose(pxFp);
    } else {
      printf("Warning: Could not open %s for writing pixel data.\n", pxFile);
    }
  }

  free(pxNPixels);
  free(pxRegNrs);
  free(pxRegSizes);

  double t3 = omp_get_wtime();
  printf("FrameNr: %d, NrOfRegions: %d, Filtered regions: %d, Number of peaks: "
         "%d, Total time: %lf\n",
         fileNr, nrOfRegions, totalValidRegions, spotIdStart - 1, t3 - t1);

  return SUCCESS;
}

/**
 * Extract dimension from Zarr metadata
 */
static ErrorCode getZarrDimension(const char *buffer, int *dimension) {
  char *ptr = strstr(buffer, "shape");
  if (ptr != NULL) {
    char *ptrt = strstr(ptr, "[");
    char *ptr2 = strstr(ptrt, "]");
    int loc = (int)(ptr2 - ptrt);
    char ptr3[MAX_BUFFER_SIZE];
    strncpy(ptr3, ptrt, loc + 1);

    if (1 == sscanf(ptr3, "%*[^0123456789]%d", dimension)) {
      return SUCCESS;
    }
  }
  return ERROR_INVALID_PARAMETERS;
}

/**
 * Read a datatype string from Zarr and determine the pixel value type
 * This is our new function to support dynamic pixel value types
 */
static ErrorCode readZarrDataType(zip_t *archive, PixelValueType *pixelType) {
  // Default to uint16
  *pixelType = PX_TYPE_UINT16;

  // Look for a datatype entry in the Zarr metadata
  int count = 0;
  struct zip_stat fileInfo;
  zip_stat_init(&fileInfo);

  // Try to find measurement/process/scan_parameters/datatype
  while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
    if (strstr(fileInfo.name, "measurement/process/scan_parameters/datatype") !=
        NULL) {
      char *typeName = NULL;
      size_t typeSize;
      int rc_raw = ReadZarrRaw(archive, count + 1, &typeName, &typeSize);

      if (rc_raw < 0) {
        fprintf(stderr, "Error reading Zarr data type\n");
        return ERROR_BLOSC_OPERATION;
      }
      printf("%s\n", typeName);

      // Set the pixel type based on string content
      if (typeName) {
        if (strcasecmp(typeName, "float") == 0 ||
            strcasecmp(typeName, "float32") == 0) {
          *pixelType = PX_TYPE_FLOAT;
          printf("Setting pixel type to float\n");
        } else if (strcasecmp(typeName, "double") == 0 ||
                   strcasecmp(typeName, "float64") == 0) {
          *pixelType = PX_TYPE_DOUBLE;
          printf("Setting pixel type to double\n");
        } else if (strcasecmp(typeName, "int32") == 0) {
          *pixelType = PX_TYPE_INT32;
          printf("Setting pixel type to int32\n");
        } else if (strcasecmp(typeName, "uint32") == 0) {
          *pixelType = PX_TYPE_UINT32;
          printf("Setting pixel type to uint32\n");
        } else {
          // Default to uint16
          *pixelType = PX_TYPE_UINT16;
          printf("Setting pixel type to uint16\n");
        }
        free(typeName);
      }

      return SUCCESS;
    }
    count++;
  }

  // If we didn't find the datatype entry, just use the default
  printf("No datatype specified, using default uint16\n");
  return SUCCESS;
}

static void printAllParameters(const ImageMetadata *metadata,
                               const AnalysisParams *params) {
  printf("\n===========================================================\n");
  printf("            Parameters Read from Zarr Archive            \n");
  printf("===========================================================\n\n");

  // --- Print ImageMetadata Struct ---
  printf("--- Image Metadata ---\n");
  printf("  nFrames            : %d\n", metadata->nFrames);
  printf("  nDarks             : %d\n", metadata->nDarks);
  printf("  nFloods            : %d\n", metadata->nFloods);
  printf("  nMasks             : %d\n", metadata->nMasks);
  printf("  NrPixelsY          : %d\n", metadata->NrPixelsY);
  printf("  NrPixelsZ          : %d\n", metadata->NrPixelsZ);
  printf("  NrPixels (Max)     : %d\n", metadata->NrPixels);
  printf("  bytesPerPx         : %zu\n", metadata->bytesPerPx);
  printf("  omegaStart         : %f\n", metadata->omegaStart);
  printf("  omegaStep          : %f\n", metadata->omegaStep);
  printf("  skipFrame          : %d\n", metadata->skipFrame);
  printf("  doPeakFit (meta)   : %d\n", metadata->doPeakFit);
  printf("  pixelType (enum)   : %d (0=u16, 1=i32, 2=f32, 3=f64, 4=u32)\n",
         metadata->pixelType);
  printf("  nOmegaCenterEntries: %d\n", metadata->nOmegaCenterEntries);
  if (metadata->omegaCenter != NULL && metadata->nOmegaCenterEntries > 0) {
    printf("  omegaCenter array  : [%f, %f, ...]\n", metadata->omegaCenter[0],
           metadata->omegaCenter[1]);
  } else {
    printf("  omegaCenter array  : Not present or empty\n");
  }

  printf("\n--- Analysis Parameters ---\n");
  // --- Print AnalysisParams Struct ---
  // ResultFolder
  printf("  bc (Beam Current)  : %f\n", params->bc);
  printf("  Ycen               : %f\n", params->Ycen);
  printf("  Zcen               : %f\n", params->Zcen);
  printf("  IntSat (Saturation): %f\n", params->IntSat);
  printf("  Lsd (Detector Dist): %f\n", params->Lsd);
  printf("  px (Pixel Size)    : %f\n", params->px);
  printf("  Width              : %f\n", params->Width);
  printf("  RhoD               : %f\n", params->RhoD);
  printf("  tx, ty, tz (tilts) : %f, %f, %f\n", params->tx, params->ty,
         params->tz);
  printf("  p0, p1, p2, p3     : %g, %g, %g, %g\n", params->p0, params->p1,
         params->p2, params->p3);
  printf("  Wavelength         : %f\n", params->Wavelength);
  printf("  zDiffThresh        : %f\n", params->zDiffThresh);
  printf("  BadPxIntensity     : %f\n", params->BadPxIntensity);
  printf("  minNrPx            : %d\n", params->minNrPx);
  printf("  maxNrPx            : %d\n", params->maxNrPx);
  printf("  DoFullImage        : %d\n", params->DoFullImage);
  printf("  LayerNr            : %d\n", params->LayerNr);
  printf("  makeMap            : %d\n", params->makeMap);
  printf("  maxNPeaks          : %d\n", params->maxNPeaks);
  printf("  doPeakFit (params) : %d\n", params->doPeakFit);

  printf("  nImTransOpt        : %d\n", params->nImTransOpt);
  if (params->TransOpt != NULL && params->nImTransOpt > 0) {
    printf("  TransOpt array     : [");
    for (int i = 0; i < params->nImTransOpt; ++i) {
      printf("%d%s", params->TransOpt[i],
             (i == params->nImTransOpt - 1) ? "" : ", ");
    }
    printf("]\n");
  } else {
    printf("  TransOpt array     : Not present or empty\n");
  }

  printf("  nRingsThresh       : %d\n", params->nRingsThresh);
  if (params->RingNrs != NULL && params->Thresholds != NULL &&
      params->nRingsThresh > 0) {
    printf("  Ring Thresholds    : Ring | Threshold\n");
    for (int i = 0; i < params->nRingsThresh; ++i) {
      printf("                     %4d | %f\n", params->RingNrs[i],
             params->Thresholds[i]);
    }
  } else {
    printf("  Ring Thresholds    : Not present or empty\n");
  }
  printf("\n===========================================================\n\n");
}

/**
 * Parse Zarr metadata and extract image parameters
 */
static ErrorCode parseZarrMetadata(const char *dataFile,
                                   ImageMetadata *metadata,
                                   AnalysisParams *params,
                                   char **resultFolder) {
  int errorp = 0;
  zip_t *archive = zip_open(dataFile, 0, &errorp);
  if (!archive) {
    return ERROR_ZIP_OPEN;
  }

  struct zip_stat *fileInfo = calloc(16384, sizeof(struct zip_stat));
  if (!fileInfo) {
    zip_close(archive);
    return ERROR_MEMORY_ALLOCATION;
  }

  // Initialize default values
  metadata->nFrames = 0;
  metadata->nDarks = 0;
  metadata->nFloods = 0;
  metadata->nMasks = 0;
  metadata->NrPixelsY = 0;
  metadata->NrPixelsZ = 0;
  metadata->bytesPerPx = sizeof(uint16_t);
  metadata->omegaStart = 0;
  metadata->omegaStep = 0;
  metadata->skipFrame = 0;
  metadata->doPeakFit = 1;
  metadata->pixelType = PX_TYPE_UINT16; // Default to uint16
  metadata->omegaCenter = NULL;         // Initialize omegaCenter
  metadata->nOmegaCenterEntries = 0;    // Initialize nOmegaCenterEntries

  params->doPeakFit = 1;
  params->bc = DEFAULT_BC;
  params->Ycen = DEFAULT_NR_PIXELS / 2;
  params->Zcen = DEFAULT_NR_PIXELS / 2;
  params->IntSat = DEFAULT_INT_SAT;
  params->Lsd = DEFAULT_LSD;
  params->px = DEFAULT_PIXEL_SIZE;
  params->Width = DEFAULT_WIDTH;
  params->RhoD = DEFAULT_NR_PIXELS * DEFAULT_PIXEL_SIZE;
  params->tx = 0;
  params->ty = 0;
  params->tz = 0;
  params->p0 = 0;
  params->p1 = 0;
  params->p2 = 0;
  params->p3 = 0;
  params->p4 = 0;
  params->Wavelength = DEFAULT_WAVELENGTH;
  params->zDiffThresh = 0;
  params->minNrPx = 1;
  params->maxNrPx = 10000;
  params->DoFullImage = 0;
  params->LayerNr = 1;
  params->makeMap = 0;
  params->maxNPeaks = 400;
  params->BadPxIntensity = 0;
  params->nImTransOpt = 0;
  params->nRingsThresh = 0;
  params->TransOpt = NULL;
  params->RingNrs = NULL;
  params->Thresholds = NULL;

  // Try to read dynamic pixel type - new functionality
  readZarrDataType(archive, &metadata->pixelType);

  // Track locations of various data chunks
  int darkLoc = -1;
  // Panel parameters
  int NPanelsY = 0;
  int NPanelsZ = 0;
  int PanelSizeY = 0;
  int PanelSizeZ = 0;
  int locPanelGapsY = -1;
  int locPanelGapsZ = -1;
  char *PanelShiftsFile = NULL;
  int dataLoc = -1;
  int floodLoc = -1;
  int maskLoc = -1;
  int locImTransOpt = -1;
  int locRingThresh = -1;
  int locOmegaRanges = -1;
  int nOmegaRanges = 0;
  int locOmegaCenterData = -1; // To store zip index of omegaCenter data chunk
  int original_nFrames_for_omega =
      0; // To store nFrames before skipFrame adjustment
  int nGapsY = 0;
  int nGapsZ = 0;

  // Parse all files in the archive
  int count = 0;
  while (zip_stat_index(archive, count, 0, fileInfo) == 0) {
    // Handle main data array metadata
    if (strstr(fileInfo->name, "exchange/data/.zarray") != NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (!buffer) {
        zip_close(archive);
        free(fileInfo);
        return ERROR_MEMORY_ALLOCATION;
      }

      zip_file_t *file = zip_fopen_index(archive, count, 0);
      zip_fread(file, buffer, fileInfo->size);
      zip_fclose(file);

      // Parse shape
      char *ptr = strstr(buffer, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[MAX_BUFFER_SIZE];
        strncpy(ptr3, ptrt, loc + 1);

        if (3 == sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &metadata->nFrames, &metadata->NrPixelsZ,
                        &metadata->NrPixelsY)) {
          printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d\n", metadata->nFrames,
                 metadata->NrPixelsZ, metadata->NrPixelsY);
          original_nFrames_for_omega =
              metadata->nFrames; // Capture original nFrames
        } else {
          free(buffer);
          zip_close(archive);
          free(fileInfo);
          return ERROR_INVALID_PARAMETERS;
        }
      }

      // Parse data type string from .zarray, but use our enum to set bytesPerPx
      // This ensures consistency
      ptr = strstr(buffer, "dtype");
      if (ptr != NULL) {
        switch (metadata->pixelType) {
        case PX_TYPE_FLOAT:
          metadata->bytesPerPx = sizeof(float);
          break;
        case PX_TYPE_DOUBLE:
          metadata->bytesPerPx = sizeof(double);
          break;
        case PX_TYPE_INT32:
          metadata->bytesPerPx = sizeof(int32_t);
          break;
        case PX_TYPE_UINT32:
          metadata->bytesPerPx = sizeof(uint32_t);
          break;
        case PX_TYPE_UINT16:
        default:
          metadata->bytesPerPx = sizeof(uint16_t);
          break;
        }
      }

      printf("BytesPerPx: %zu, nFrames: %d, nrPixelsZ: %d nrPixelsY: %d\n",
             metadata->bytesPerPx, metadata->nFrames, metadata->NrPixelsZ,
             metadata->NrPixelsY);

      free(buffer);
    }

    // Handle dark array metadata
    if (strstr(fileInfo->name, "exchange/dark/.zarray") != NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (!buffer) {
        zip_close(archive);
        free(fileInfo);
        return ERROR_MEMORY_ALLOCATION;
      }
      zip_file_t *file = zip_fopen_index(archive, count, 0);
      zip_fread(file, buffer, fileInfo->size);
      zip_fclose(file);
      char *ptr = strstr(buffer, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[MAX_BUFFER_SIZE];
        strncpy(ptr3, ptrt, loc + 1);
        if (3 != sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &metadata->nDarks, &metadata->NrPixelsZ,
                        &metadata->NrPixelsY)) {
          free(buffer);
          zip_close(archive);
          free(fileInfo);
          return ERROR_INVALID_PARAMETERS;
        }
      }
      free(buffer);
    }

    // Handle flood array metadata
    if (strstr(fileInfo->name, "exchange/flood/.zarray") != NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (!buffer) {
        zip_close(archive);
        free(fileInfo);
        return ERROR_MEMORY_ALLOCATION;
      }
      zip_file_t *file = zip_fopen_index(archive, count, 0);
      zip_fread(file, buffer, fileInfo->size);
      zip_fclose(file);
      char *ptr = strstr(buffer, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[MAX_BUFFER_SIZE];
        strncpy(ptr3, ptrt, loc + 1);
        if (3 != sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &metadata->nFloods, &metadata->NrPixelsZ,
                        &metadata->NrPixelsY)) {
          free(buffer);
          zip_close(archive);
          free(fileInfo);
          return ERROR_INVALID_PARAMETERS;
        }
      }
      free(buffer);
    }

    // Handle mask array metadata
    if (strstr(fileInfo->name, "exchange/mask/.zarray") != NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (!buffer) {
        zip_close(archive);
        free(fileInfo);
        return ERROR_MEMORY_ALLOCATION;
      }
      zip_file_t *file = zip_fopen_index(archive, count, 0);
      zip_fread(file, buffer, fileInfo->size);
      zip_fclose(file);
      char *ptr = strstr(buffer, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[MAX_BUFFER_SIZE];
        strncpy(ptr3, ptrt, loc + 1);
        if (3 != sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &metadata->nMasks, &metadata->NrPixelsZ,
                        &metadata->NrPixelsY)) {
          free(buffer);
          zip_close(archive);
          free(fileInfo);
          return ERROR_INVALID_PARAMETERS;
        }
      }
      free(buffer);
    }

    // Track data locations
    if (strstr(fileInfo->name, "exchange/data/0.0.0") != NULL)
      dataLoc = count;
    if (strstr(fileInfo->name, "exchange/dark/0.0.0") != NULL)
      darkLoc = count;
    if (strstr(fileInfo->name, "exchange/mask/0.0.0") != NULL) {
      printf("Mask is found.\n");
      maskLoc = count;
    }
    if (strstr(fileInfo->name, "exchange/flood/0.0.0") != NULL)
      floodLoc = count;
    if (strcmp(fileInfo->name,
               "measurement/process/scan_parameters/omegaCenter/0") == 0)
      locOmegaCenterData = count;

    // Panel parameters parsing
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/NPanelsY/0") != NULL)
      ReadZarrChunk(archive, count, &NPanelsY, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/NPanelsZ/0") != NULL)
      ReadZarrChunk(archive, count, &NPanelsZ, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelSizeY/0") != NULL)
      ReadZarrChunk(archive, count, &PanelSizeY, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelSizeZ/0") != NULL)
      ReadZarrChunk(archive, count, &PanelSizeZ, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelShiftsFile/0") !=
        NULL)
      ReadZarrString(archive, count, &PanelShiftsFile, 4096);
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelGapsY/0") != NULL)
      locPanelGapsY = count;
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelGapsZ/0") != NULL)
      locPanelGapsZ = count;
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelGapsY/.zarray") !=
        NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (buffer) {
        zip_file_t *f = zip_fopen_index(archive, count, 0);
        zip_fread(f, buffer, fileInfo->size);
        zip_fclose(f);
        getZarrDimension(buffer, &nGapsY);
        free(buffer);
      }
    }
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PanelGapsZ/.zarray") !=
        NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (buffer) {
        zip_file_t *f = zip_fopen_index(archive, count, 0);
        zip_fread(f, buffer, fileInfo->size);
        zip_fclose(f);
        getZarrDimension(buffer, &nGapsZ);
        free(buffer);
      }
    }

    // Read various scalar parameters
    if (strstr(fileInfo->name, "measurement/process/scan_parameters/start/0") !=
        NULL)
      ReadZarrChunk(archive, count, &metadata->omegaStart, sizeof(double));
    if (strstr(fileInfo->name, "measurement/process/scan_parameters/step/0") !=
        NULL)
      ReadZarrChunk(archive, count, &metadata->omegaStep, sizeof(double));
    if (strstr(fileInfo->name,
               "measurement/process/scan_parameters/doPeakFit/0") != NULL) {
      ReadZarrChunk(archive, count, &metadata->doPeakFit, sizeof(int));
      params->doPeakFit = metadata->doPeakFit;
    }
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      ReadZarrString(archive, count, resultFolder, 4096);
      printf("ResultFolder: %s\n", *resultFolder);
    }
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/MaxNPeaks/0") != NULL)
      ReadZarrChunk(archive, count, &params->maxNPeaks, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/SkipFrame/0") != NULL)
      ReadZarrChunk(archive, count, &metadata->skipFrame, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/zDiffThresh/0") != NULL)
      ReadZarrChunk(archive, count, &params->zDiffThresh, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tx/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->tx, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ty/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->ty, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tz/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->tz, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p0/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->p0, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p1/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->p1, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p2/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->p2, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p3/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->p3, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p4/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->p4, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/MinNrPx/0") != NULL)
      ReadZarrChunk(archive, count, &params->minNrPx, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/MaxNrPx/0") != NULL)
      ReadZarrChunk(archive, count, &params->maxNrPx, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/DoFullImage/0") != NULL)
      ReadZarrChunk(archive, count, &params->DoFullImage, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/ReferenceRingCurrent/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->bc, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/YCen/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->Ycen, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ZCen/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->Zcen, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/UpperBoundThreshold/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->IntSat, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/PixelSize/0") != NULL)
      ReadZarrChunk(archive, count, &params->px, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/Width/0") != NULL)
      ReadZarrChunk(archive, count, &params->Width, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/LayerNr/0") != NULL)
      ReadZarrChunk(archive, count, &params->LayerNr, sizeof(int));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/Wavelength/0") != NULL)
      ReadZarrChunk(archive, count, &params->Wavelength, sizeof(double));
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
        NULL)
      ReadZarrChunk(archive, count, &params->Lsd, sizeof(double));
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/BadPxIntensity/0") !=
        NULL) {
      ReadZarrChunk(archive, count, &params->BadPxIntensity, sizeof(double));
      params->makeMap = 1;
    }
    if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RhoD/0") !=
            NULL ||
        strstr(fileInfo->name,
               "analysis/process/analysis_parameters/MaxRingRad/0") != NULL) {
      ReadZarrChunk(archive, count, &params->RhoD, sizeof(double));
    }

    // Track locations for arrays to read later
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/0") != NULL)
      locImTransOpt = count;
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/RingThresh/0.0") != NULL)
      locRingThresh = count;
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/OmegaRanges/0.0") != NULL)
      locOmegaRanges = count;

    // Read array dimensions
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/RingThresh/.zarray") !=
        NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (buffer) {
        zip_file_t *f = zip_fopen_index(archive, count, 0);
        zip_fread(f, buffer, fileInfo->size);
        zip_fclose(f);
        getZarrDimension(buffer, &params->nRingsThresh);
        free(buffer);
      }
    }
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/OmegaRanges/.zarray") !=
        NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (buffer) {
        zip_file_t *f = zip_fopen_index(archive, count, 0);
        zip_fread(f, buffer, fileInfo->size);
        zip_fclose(f);
        getZarrDimension(buffer, &nOmegaRanges);
        free(buffer);
      }
    }
    if (strstr(fileInfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/.zarray") !=
        NULL) {
      char *buffer = calloc(fileInfo->size + 1, sizeof(char));
      if (buffer) {
        zip_file_t *f = zip_fopen_index(archive, count, 0);
        zip_fread(f, buffer, fileInfo->size);
        zip_fclose(f);
        getZarrDimension(buffer, &params->nImTransOpt);
        free(buffer);
      }
    }
    count++;
  }

  // Generate Panels
  // print nPanels, panel gaps, and PanelShifts file
  printf("nPanels: %d %d\n", NPanelsY, NPanelsZ);
  printf("PanelShiftsFile: %s\n", PanelShiftsFile);
  if (NPanelsY > 0 && NPanelsZ > 0) {
    int *PanelGapsY = NULL;
    int *PanelGapsZ = NULL;

    if (locPanelGapsY != -1) {
      int32_t bufSize = nGapsY * sizeof(int);
      PanelGapsY = (int *)malloc((size_t)bufSize);
      ReadZarrChunk(archive, locPanelGapsY, PanelGapsY, bufSize);
    }
    if (locPanelGapsZ != -1) {
      int32_t bufSize = nGapsZ * sizeof(int);
      PanelGapsZ = (int *)malloc((size_t)bufSize);
      ReadZarrChunk(archive, locPanelGapsZ, PanelGapsZ, bufSize);
    }
    printf("PanelGapsY: %d %d\n", nGapsY, nGapsZ);
    for (int i = 0; i < nGapsY; i++)
      printf("PanelGapsY[%d]: %d\n", i, PanelGapsY[i]);
    for (int i = 0; i < nGapsZ; i++)
      printf("PanelGapsZ[%d]: %d\n", i, PanelGapsZ[i]);
    GeneratePanels(NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ, PanelGapsY,
                   PanelGapsZ, &panels, &nPanels);

    if (PanelShiftsFile) {
      if (LoadPanelShifts(PanelShiftsFile, nPanels, panels) == 0) {
        printf("Loaded panel shifts from %s\n", PanelShiftsFile);
      } else {
        printf("Failed to load panel shifts from %s\n", PanelShiftsFile);
      }
      free(PanelShiftsFile);
    }

    if (PanelGapsY)
      free(PanelGapsY);
    if (PanelGapsZ)
      free(PanelGapsZ);
  }

  if (locOmegaCenterData != -1 && original_nFrames_for_omega > 0) {
    metadata->omegaCenter =
        (double *)malloc((size_t)original_nFrames_for_omega * sizeof(double));
    if (metadata->omegaCenter) {
      int rc_oc = ReadZarrChunk(
          archive, locOmegaCenterData, metadata->omegaCenter,
          (int32_t)((size_t)original_nFrames_for_omega * sizeof(double)));
      if (rc_oc >= 0) {
        metadata->nOmegaCenterEntries = original_nFrames_for_omega;
      } else {
        free(metadata->omegaCenter);
        metadata->omegaCenter = NULL;
      }
    }
  }

  // Set NrPixels to max of Y and Z dimensions
  metadata->NrPixels = metadata->NrPixelsY > metadata->NrPixelsZ
                           ? metadata->NrPixelsY
                           : metadata->NrPixelsZ;

  // Read transformation options
  if (params->nImTransOpt > 0 && locImTransOpt != -1) {
    {
      int32_t bufSize = params->nImTransOpt * sizeof(int);
      params->TransOpt = (int *)malloc((size_t)bufSize);
      ReadZarrChunk(archive, locImTransOpt, params->TransOpt, bufSize);
    }
  }

  // Read ring thresholds
  if (params->nRingsThresh > 0 && locRingThresh != -1) {
    params->RingNrs = calloc(params->nRingsThresh, sizeof(int));
    params->Thresholds = calloc(params->nRingsThresh, sizeof(double));
    if (params->RingNrs && params->Thresholds) {
      int32_t rtBufSize = params->nRingsThresh * 2 * sizeof(double);
      double *ringThresholds = (double *)malloc((size_t)rtBufSize);
      if (ringThresholds && ReadZarrChunk(archive, locRingThresh,
                                          ringThresholds, rtBufSize) >= 0) {
        for (int i = 0; i < params->nRingsThresh; i++) {
          params->RingNrs[i] = (int)ringThresholds[i * 2 + 0];
          params->Thresholds[i] = ringThresholds[i * 2 + 1];
        }
      }
      free(ringThresholds);
    }
  }

  zDiffThresh = params->zDiffThresh;
  metadata->nFrames -= metadata->skipFrame;
  metadata->nDarks -= metadata->skipFrame;
  dataLoc += metadata->skipFrame;
  darkLoc += metadata->skipFrame;
  params->Width /= params->px;

  free(fileInfo);
  zip_close(archive);

  return SUCCESS;
}

/**
 * Calculate ring radii based on hkl data
 */
static ErrorCode calculateRingRadii(AnalysisParams *params,
                                    const char *resultFolder,
                                    double **ringRads) {
  *ringRads = calloc(params->nRingsThresh, sizeof(double));
  if (!(*ringRads))
    return ERROR_MEMORY_ALLOCATION;
  if (params->DoFullImage == 1)
    return SUCCESS;

  char hklFileName[MAX_FILENAME_LENGTH];
  snprintf(hklFileName, MAX_FILENAME_LENGTH, "%s/hkls.csv", resultFolder);

  FILE *hklFile = fopen(hklFileName, "r");
  if (!hklFile) {
    free(*ringRads);
    *ringRads = NULL;
    return ERROR_FILE_OPEN;
  }

  char line[MAX_LINE_LENGTH];
  fgets(line, MAX_LINE_LENGTH, hklFile); // Skip header

  int ringNr;
  double ringRad;
  char dummy[MAX_LINE_LENGTH];
  while (fgets(line, MAX_LINE_LENGTH, hklFile)) {
    if (11 == sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", dummy, dummy,
                     dummy, dummy, &ringNr, dummy, dummy, dummy, dummy, dummy,
                     &ringRad)) {
      for (int i = 0; i < params->nRingsThresh; i++) {
        if (ringNr == params->RingNrs[i]) {
          (*ringRads)[i] = ringRad / params->px;
          break;
        }
      }
    }
  }

  fclose(hklFile);
  return SUCCESS;
}

/**
 * Read frame data from Zarr archive
 */
static ErrorCode readFrameData(const char *dataFile, int dataLoc, int nFrames,
                               size_t **sizeArr, char **allData) {
  int errorp = 0;
  zip_t *archive = zip_open(dataFile, 0, &errorp);
  if (!archive)
    return ERROR_ZIP_OPEN;

  *sizeArr = calloc((size_t)nFrames * 2, sizeof(size_t));
  if (!(*sizeArr)) {
    zip_close(archive);
    return ERROR_MEMORY_ALLOCATION;
  }

  size_t totalSize = 0;
  for (int i = 0; i < nFrames; i++) {
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    if (zip_stat_index(archive, dataLoc + i, 0, &fileStat) != 0) {
      free(*sizeArr);
      *sizeArr = NULL;
      zip_close(archive);
      return ERROR_ZIP_OPEN;
    }
    (*sizeArr)[i * 2 + 0] = fileStat.size;
    (*sizeArr)[i * 2 + 1] = totalSize;
    totalSize += fileStat.size;
  }

  *allData = calloc(totalSize + 1, sizeof(char));
  if (!(*allData)) {
    free(*sizeArr);
    *sizeArr = NULL;
    zip_close(archive);
    return ERROR_MEMORY_ALLOCATION;
  }

  for (int i = 0; i < nFrames; i++) {
    zip_file_t *file = zip_fopen_index(archive, dataLoc + i, 0);
    if (!file) {
      free(*sizeArr);
      *sizeArr = NULL;
      free(*allData);
      *allData = NULL;
      zip_close(archive);
      return ERROR_FILE_OPEN;
    }
    zip_fread(file, &(*allData)[(*sizeArr)[i * 2 + 1]], (*sizeArr)[i * 2 + 0]);
    zip_fclose(file);
  }

  zip_close(archive);
  return SUCCESS;
}

/**
 * Main function
 */
int main(int argc, char *argv[]) {
  double startTime = omp_get_wtime();

  if (argc < 5) {
    printf("Usage: %s DataFile blockNr nBlocks numProcs [ResultFolder] "
           "[fitPeaks]\n",
           argv[0]);
    return ERROR_INVALID_PARAMETERS;
  }

  char *dataFile = argv[1];
  int blockNr = atoi(argv[2]);
  int nBlocks = atoi(argv[3]);
  int numProcs = atoi(argv[4]);

  char *resultFolder = NULL;

  blosc2_init();

  ImageMetadata metadata;
  AnalysisParams params;

  ErrorCode error =
      parseZarrMetadata(dataFile, &metadata, &params, &resultFolder);
  if (error != SUCCESS) {
    printf("Error parsing Zarr metadata: %d\n", error);
    if (resultFolder)
      free(resultFolder);
    if (metadata.omegaCenter)
      free(metadata.omegaCenter);
    blosc2_destroy();
    return error;
  }
  printAllParameters(&metadata, &params);

  if (argc > 5) {
    if (resultFolder)
      free(resultFolder);
    resultFolder = strdup(argv[5]);
  }
  if (argc > 6)
    params.doPeakFit = atoi(argv[6]);

  char outFolderName[MAX_FILENAME_LENGTH];
  snprintf(outFolderName, MAX_FILENAME_LENGTH, "%s/Temp", resultFolder);
  checkDirectoryCreation(outFolderName);

  double *ringRads = NULL;
  if (calculateRingRadii(&params, resultFolder, &ringRads) != SUCCESS) {
    if (resultFolder)
      free(resultFolder);
    blosc2_destroy();
    return ERROR_FILE_OPEN;
  }

  // --- Detector geometry and coordinate calculation ---
  double txr = DEG2RAD * params.tx, tyr = DEG2RAD * params.ty,
         tzr = DEG2RAD * params.tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  matrixMultiply33(Ry, Rz, TRint);
  matrixMultiply33(Rx, TRint, TRs);

  double *goodCoords =
      calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
  if (!goodCoords)
    return ERROR_MEMORY_ALLOCATION;

  if (params.DoFullImage == 1) {
    for (int a = 0; a < metadata.NrPixels * metadata.NrPixels; a++)
      goodCoords[a] = params.Thresholds[0];
  } else {
#pragma omp parallel for
    for (int a = 0; a < metadata.NrPixels; a++) {
      for (int b = 0; b < metadata.NrPixels; b++) {
        double pixY = (double)a, pixZ = (double)b;
        double dLsd = 0, dP2 = 0;
        if (nPanels > 0) {
          int pIdx = GetPanelIndex(pixY, pixZ, nPanels, panels);
          if (pIdx >= 0) {
            dLsd = panels[pIdx].dLsd;
            dP2 = panels[pIdx].dP2;
          }
        }
        double panelLsd = params.Lsd + dLsd;
        double panelP2 = params.p2 + dP2;
        double Yc = (-a + params.Ycen) * params.px,
               Zc = (b - params.Zcen) * params.px;
        double ABC[3] = {0, Yc, Zc}, ABCPr[3];
        matrixVectorMultiply(TRs, ABC, ABCPr);
        double XYZ[3] = {panelLsd + ABCPr[0], ABCPr[1], ABCPr[2]};
        double Rad =
            (panelLsd / XYZ[0]) * sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);
        double Eta = calcEtaAngle(XYZ[1], XYZ[2]);
        double RNorm = Rad / params.RhoD;
        double EtaT = 90 - Eta;
        // Item 4: replace pow() with multiplies
        double RNorm2 = RNorm * RNorm;
        double RNorm4 = RNorm2 * RNorm2;
        // Item 5: pre-convert to radians for trig
        double EtaT_rad = EtaT * DEG2RAD;
        double DistortFunc =
            (params.p0 * RNorm2 * cos(2.0 * EtaT_rad)) +
            (params.p1 * RNorm4 * cos(4.0 * EtaT_rad + params.p3 * DEG2RAD)) +
            (panelP2 * RNorm2) + params.p4 * RNorm4 * RNorm2 + 1;
        double Rt = Rad * DistortFunc / params.px;
        Rt = Rt * (params.Lsd / panelLsd); // re-project to global Lsd plane
        for (int r = 0; r < params.nRingsThresh; r++) {
          if (Rt > ringRads[r] - params.Width &&
              Rt < ringRads[r] + params.Width) {
            goodCoords[(a * metadata.NrPixels) + b] = params.Thresholds[r];
          }
        }
      }
    }
  }

  int startFileNr =
      (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * blockNr;
  int endFileNr =
      (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * (blockNr + 1);
  if (endFileNr > metadata.nFrames)
    endFileNr = metadata.nFrames;
  printf("Processing frames %d to %d\n", startFileNr, endFileNr);

  // --- Read Correction and Image Data (Shared by all threads) ---
  int errorp = 0;
  zip_t *archive = zip_open(dataFile, 0, &errorp);
  if (!archive)
    return ERROR_ZIP_OPEN;

  double *dark =
      calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
  double *mask =
      calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
  double *flood =
      calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));

  int darkLoc = -1, floodLoc = -1, maskLoc = -1, count = 0;
  struct zip_stat fileInfo;
  zip_stat_init(&fileInfo);
  while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
    if (strstr(fileInfo.name, "exchange/dark/0.0.0") != NULL)
      darkLoc = count;
    if (strstr(fileInfo.name, "exchange/mask/0.0.0") != NULL)
      maskLoc = count;
    if (strstr(fileInfo.name, "exchange/flood/0.0.0") != NULL)
      floodLoc = count;
    count++;
  }
  darkLoc += metadata.skipFrame;

  error = readImageCorrections(archive, darkLoc, floodLoc, maskLoc, &metadata,
                               &params, dark, flood, mask);
  zip_close(archive);
  if (error != SUCCESS) { /* cleanup and exit */
    printf("Error reading image corrections: %d\n", error);
    return error;
  }

  // Re-find dataLoc as archive was closed and re-opened implicitly in other
  // functions
  int dataLoc = -1;
  count = 0;
  archive = zip_open(dataFile, 0, &errorp);
  if (!archive) {
    fprintf(stderr,
            "ERROR: Could not re-open zip archive '%s' (error code: %d)\n",
            dataFile, errorp);
    return ERROR_ZIP_OPEN;
  }
  zip_stat_init(&fileInfo);
  while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
    if (strstr(fileInfo.name, "exchange/data/0.0.0") != NULL)
      dataLoc = count;
    count++;
  }
  zip_close(archive);
  dataLoc += metadata.skipFrame;

  size_t *sizeArr = NULL;
  char *allData = NULL;
  error =
      readFrameData(dataFile, dataLoc, metadata.nFrames, &sizeArr, &allData);
  if (error != SUCCESS) { /* cleanup and exit */
    return error;
  }

  // --- HIGHLY EFFICIENT PARALLEL PROCESSING LOOP ---
  int nrFilesDone = 0;
#pragma omp parallel num_threads(numProcs) shared(nrFilesDone)
  {
    // 1. Each thread declares its own workspace struct.
    ThreadWorkspace ws;

    // 2. Each thread allocates its workspace MEMORY ONCE.
    ErrorCode alloc_error = allocateWorkspace(&ws, &metadata, &params);
    if (alloc_error != SUCCESS) {
#pragma omp critical
      {
        fprintf(stderr,
                "FATAL: Memory allocation for thread workspace failed.\n");
      }
    } else {
// 3. Begin the parallel work distribution.
#pragma omp for schedule(dynamic)
      for (int fileNr = startFileNr; fileNr < endFileNr; fileNr++) {
        int current_original_frame_idx = fileNr + metadata.skipFrame;
        double omega;
        if (metadata.omegaCenter &&
            current_original_frame_idx < metadata.nOmegaCenterEntries) {
          omega = metadata.omegaCenter[current_original_frame_idx];
        } else {
          omega = metadata.omegaStart +
                  (double)current_original_frame_idx * metadata.omegaStep;
        }

        ErrorCode threadError = processImageFrame(
            fileNr, allData, sizeArr, &metadata, &params, dark, flood, mask,
            goodCoords, omega, outFolderName, dataFile,
            &ws); // Pass workspace pointer

        // Item 10: atomic is cheaper than critical for a simple increment
        if (threadError == SUCCESS) {
#pragma omp atomic
          nrFilesDone++;
        }
      }

      // 4. After its work is done, each thread frees its workspace.
      freeWorkspace(&ws);
    }
  } // --- End of parallel region ---

  // Final Cleanup
  free(dark);
  free(flood);
  free(mask);
  free(goodCoords);
  free(sizeArr);
  free(allData);
  if (ringRads)
    free(ringRads);
  if (params.TransOpt)
    free(params.TransOpt);
  if (params.RingNrs)
    free(params.RingNrs);
  if (params.Thresholds)
    free(params.Thresholds);
  if (resultFolder)
    free(resultFolder);
  if (metadata.omegaCenter)
    free(metadata.omegaCenter);

  blosc2_destroy();

  double totalTime = omp_get_wtime() - startTime;
  printf("Finished, time elapsed: %lf seconds, nrFramesDone: %d.\n", totalTime,
         nrFilesDone);

  return SUCCESS;
}
