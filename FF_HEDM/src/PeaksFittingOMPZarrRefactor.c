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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <nlopt.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>
#include <libgen.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/resource.h>
#include <blosc2.h>
#include <stdlib.h> 
#include <zip.h> 

/*
 * CONSTANTS AND DEFINITIONS
 */
#define DEG2RAD 0.0174532925199433
#define RAD2DEG 57.2957795130823
#define MAXNHKLS 5000
#define MAX_N_RINGS 500
#define MAX_OVERLAPS_PER_IMAGE 10000
#define MAX_FILENAME_LENGTH 2048
#define MAX_BUFFER_SIZE 4096
#define MAX_LINE_LENGTH 1000
#define DEFAULT_WIDTH 1000
#define DEFAULT_LSD 1000000
#define DEFAULT_PIXEL_SIZE 200
#define DEFAULT_WAVELENGTH 0.189714
#define DEFAULT_NR_PIXELS 2048
#define DEFAULT_BC 1
#define DEFAULT_INT_SAT 14000

#define CALC_NORM_3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CALC_NORM_2(x,y) sqrt((x)*(x) + (y)*(y))

// This typedef is now only used for legacy functions that are no longer in the main data path.
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

// Structure for image metadata
typedef struct {
    int nFrames;
    int nDarks;
    int nFloods;
    int nMasks;
    int NrPixelsY;
    int NrPixelsZ;
    int NrPixels;    // Max of Y and Z dimensions
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
    double *z;
    double *Rs;
    double *Etas;
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
} ThreadWorkspace;


// Global variables
double zDiffThresh;

/*
 * UTILITY FUNCTIONS
 */

/**
 * Calculate time difference in microseconds
 */
long double diffTime(struct timespec start, struct timespec end)
{
    long double diff_sec = end.tv_sec - start.tv_sec;
    long double diff_nsec = end.tv_nsec - start.tv_nsec;
    return (diff_sec * 1e6) + (diff_nsec / 1000.0);
}


/**
 * Allocate a 2D matrix of doubles
 */
static inline
double** allocMatrix(int nrows, int ncols)
{
    double** arr;
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
 * Free a 2D matrix of doubles
 */
static inline
void freeMatrix(double **mat, int nrows)
{
    if (mat == NULL) 
        return;
    
    for (int r = 0; r < nrows; r++) {
        if (mat[r] != NULL) {
            free(mat[r]);
        }
    }
    free(mat);
}


/**
 * Calculate eta angle from y, z coordinates
 */
static inline
double calcEtaAngle(double y, double z)
{
    double alpha = RAD2DEG * acos(z / sqrt(y*y + z*z));
    if (y > 0) 
        return -alpha;
    else 
        return alpha;
}

/**
 * Convert R and Eta coordinates to Y and Z
 */
static inline
void yzFromREta(int nrElements, double *R, double *Eta, double *Y, double *Z)
{
    for (int i = 0; i < nrElements; i++) {
        Y[i] = -R[i] * sin(Eta[i] * DEG2RAD);
        Z[i] = R[i] * cos(Eta[i] * DEG2RAD);
    }
}

/**
 * Allocate a 2D matrix of integers
 */
static inline
int** allocMatrixInt(int nrows, int ncols)
{
    int** arr;
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
static inline
void freeMatrixInt(int **mat, int nrows)
{
    if (mat == NULL) 
        return;
    
    for (int r = 0; r < nrows; r++) {
        if (mat[r] != NULL) {
            free(mat[r]);
        }
    }
    free(mat);
}

// Trigonometric functions using degrees
static inline double sind(double x) { return sin(DEG2RAD * x); }
static inline double cosd(double x) { return cos(DEG2RAD * x); }
static inline double tand(double x) { return tan(DEG2RAD * x); }
static inline double asind(double x) { return RAD2DEG * asin(x); }
static inline double acosd(double x) { return RAD2DEG * acos(x); }
static inline double atand(double x) { return RAD2DEG * atan(x); }

/**
 * Transpose a square matrix of doubles
 */
static inline void transposeMatrix(double *x, int n, double *y)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y[(i*n)+j] = x[(j*n)+i];
        }
    }
}

/**
 * Check for directory existence and create if needed
 */
static inline int checkDirectoryCreation(const char *folder)
{
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
static inline
void matrixVectorMultiply(double m[3][3], double v[3], double r[3])
{
    for (int i = 0; i < 3; i++) {
        r[i] = m[i][0] * v[0] +
               m[i][1] * v[1] +
               m[i][2] * v[2];
    }
}

/**
 * Multiply two 3x3 matrices
 */
static inline
void matrixMultiply33(double m[3][3], double n[3][3], double res[3][3])
{
    for (int r = 0; r < 3; r++) {
        res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
        res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
        res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
    }
}

/**
 * Error check and reporting function
 */
static void errorCheck(int test, const char *message, ...)
{
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
        free(ws->imax);
        free(ws->yCenArray);
        free(ws->zCenArray);
        free(ws->rads);
        free(ws->etas);
        free(ws->nrPx);
        free(ws->otherInfo);
    }
}

// Allocates all memory needed by a single thread's workspace.
// Returns SUCCESS or an error code.
ErrorCode allocateWorkspace(ThreadWorkspace *ws, const ImageMetadata *metadata, const AnalysisParams *params) {
    size_t nrPixelsSq = (size_t)metadata->NrPixels * metadata->NrPixels;
    
    ws->imgCorrBC = calloc(nrPixelsSq, sizeof(double));
    ws->boolImage = calloc(nrPixelsSq, sizeof(int));
    ws->connectedComponents = calloc(nrPixelsSq, sizeof(int));
    
    // Use the constants defined in the new code
    ws->positions = calloc((size_t)MAX_OVERLAPS_PER_IMAGE * metadata->NrPixels * 4, sizeof(int));
    ws->positionTrackers = calloc(MAX_OVERLAPS_PER_IMAGE, sizeof(int));
    
    ws->usefulPixels = calloc(metadata->NrPixels * 20, sizeof(int));
    ws->maximaPositions = calloc(metadata->NrPixels * 20, sizeof(int));
    ws->maximaValues = calloc(metadata->NrPixels * 10, sizeof(double));
    ws->z = calloc(metadata->NrPixels * 10, sizeof(double));
    
    ws->integratedIntensity = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->imax = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->yCenArray = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->zCenArray = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->rads = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->etas = calloc(params->maxNPeaks * 2, sizeof(double));
    ws->nrPx = calloc(params->maxNPeaks * 2, sizeof(int));
    ws->otherInfo = calloc(params->maxNPeaks * 10, sizeof(double));
    
    // Check if any allocation failed
    if (!ws->imgCorrBC || !ws->boolImage || !ws->connectedComponents || !ws->positions ||
        !ws->positionTrackers || !ws->usefulPixels || !ws->maximaPositions || !ws->maximaValues ||
        !ws->z || !ws->integratedIntensity || !ws->imax || !ws->yCenArray || !ws->zCenArray ||
        !ws->rads || !ws->etas || !ws->nrPx || !ws->otherInfo) {
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
const int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
const int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

/**
 * Iterative implementation of depth-first search for connected components
 * Replaces the recursive implementation to avoid stack overflow
 */
static inline void depthFirstSearchIterative(
    int startX, int startY, int label, int nrPixels, 
    int *boolImage, int *connectedComponents, int *positions, int *positionTrackers)
{
    // Create a stack for DFS
    typedef struct {
        int x;
        int y;
    } StackNode;
    
    StackNode *stack = malloc(nrPixels * nrPixels * sizeof(StackNode));
    if (!stack) 
        return;
    
    int stackSize = 0;
    
    // Push the starting point
    stack[stackSize].x = startX;
    stack[stackSize].y = startY;
    stackSize++;
    
    while (stackSize > 0) {
        // Pop from stack
        stackSize--;
        int x = stack[stackSize].x;
        int y = stack[stackSize].y;
        
        if (x < 0 || x >= nrPixels || y < 0 || y >= nrPixels) 
            continue;
        if (connectedComponents[x*nrPixels+y] != 0 || boolImage[x*nrPixels+y] == 0) 
            continue;

        connectedComponents[x*nrPixels+y] = label;
        positions[label*nrPixels*4+positionTrackers[label]] = (x*nrPixels) + y;
        positionTrackers[label]++;
        
        // Push all neighbors onto stack
        for (int direction = 0; direction < 8; direction++) {
            int newX = x + dx[direction];
            int newY = y + dy[direction];
            
            if (newX >= 0 && newX < nrPixels && newY >= 0 && newY < nrPixels &&
                connectedComponents[newX*nrPixels+newY] == 0 && boolImage[newX*nrPixels+newY] == 1) {
                stack[stackSize].x = newX;
                stack[stackSize].y = newY;
                stackSize++;
            }
        }
    }
    
    free(stack);
}

/**
 * Find connected components in binary image
 */
static inline int findConnectedComponents(
    int *boolImage, int nrPixels, int *connectedComponents, 
    int *positions, int *positionTrackers)
{
    // Initialize the connected components map
    memset(connectedComponents, 0, nrPixels * nrPixels * sizeof(int));
    
    int component = 0;
    for (int i = 0; i < nrPixels; i++) {
        for (int j = 0; j < nrPixels; j++) {
            if (connectedComponents[i*nrPixels+j] == 0 && boolImage[i*nrPixels+j] == 1) {
                depthFirstSearchIterative(i, j, ++component, nrPixels, boolImage, 
                                        connectedComponents, positions, positionTrackers);
            }
        }
    }
    return component;
}

/**
 * Find regional maxima in a connected component
 */
static inline unsigned findRegionalMaxima(
    double *z, int *pixelPositions, int nrPixelsThisRegion,
    int *maximaPositions, double *maximaValues, double intSat,
    int nrPixels, double *mask)
{
    unsigned nPeaks = 0;
    
    for (int i = 0; i < nrPixelsThisRegion; i++) {
        // Skip saturated pixels
        if (z[i] > intSat) {
            return 0;  // Saturated peak removed
        }
        
        int xThis = pixelPositions[i*2+0];
        int yThis = pixelPositions[i*2+1];
        
        // Skip if we touched the mask
        if (mask[xThis + nrPixels * yThis] == 1) {
            return 0;
        }
        
        // Check if this is a local maximum
        int isRegionalMax = 1;
        double zThis = z[i];
        
        for (int j = 0; j < 8 && isRegionalMax; j++) {
            int xNext = xThis + dx[j];
            int yNext = yThis + dy[j];
            
            for (int k = 0; k < nrPixelsThisRegion; k++) {
                if (xNext == pixelPositions[k*2+0] && 
                    yNext == pixelPositions[k*2+1] && 
                    z[k] > zThis) {
                    isRegionalMax = 0;
                    break;
                }
            }
        }
        
        if (isRegionalMax) {
            maximaPositions[nPeaks*2+0] = xThis;
            maximaPositions[nPeaks*2+1] = yThis;
            maximaValues[nPeaks] = zThis;
            nPeaks++;
        }
    }
    
    // If no peaks found, use the middle pixel
    if (nPeaks == 0) {
        maximaPositions[0] = pixelPositions[nrPixelsThisRegion/2*2+0];
        maximaPositions[1] = pixelPositions[nrPixelsThisRegion/2*2+1];
        maximaValues[0] = z[nrPixelsThisRegion/2];
        nPeaks = 1;
    }
    
    return nPeaks;
}

/*
 * PEAK FITTING FUNCTIONS
 */

/**
 * Objective function for peak fitting
 */
static double peakFittingObjectiveFunction(
    unsigned n, const double *x, double *grad, void *f_data_trial)
{
    FunctionData *f_data = (FunctionData *) f_data_trial;
    int nrPixels = f_data->NrPixels;
    double *z = f_data->z;
    double *Rs = f_data->Rs;
    double *Etas = f_data->Etas;
    
    // Number of peaks is (n-1)/8 because x has 8 parameters per peak plus background
    int nPeaks = (n-1)/8;
    double bg = x[0];  // Background intensity
    
    // Extract peak parameters
    double IMAX[nPeaks], R[nPeaks], Eta[nPeaks], Mu[nPeaks];
    double SigmaGR[nPeaks], SigmaLR[nPeaks], SigmaGEta[nPeaks], SigmaLEta[nPeaks];
    
    for (int i = 0; i < nPeaks; i++) {
        IMAX[i] = x[(8*i)+1];       // Peak intensity
        R[i] = x[(8*i)+2];          // Peak radius
        Eta[i] = x[(8*i)+3];        // Peak eta angle
        Mu[i] = x[(8*i)+4];         // Lorentzian vs Gaussian ratio
        SigmaGR[i] = x[(8*i)+5];    // Gaussian sigma in R
        SigmaLR[i] = x[(8*i)+6];    // Lorentzian sigma in R
        SigmaGEta[i] = x[(8*i)+7];  // Gaussian sigma in Eta
        SigmaLEta[i] = x[(8*i)+8];  // Lorentzian sigma in Eta
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
            
            // Lorentzian component
            double L = 1 / (((R2/((SigmaLR[j])*(SigmaLR[j])))+1) * 
                          ((E2/((SigmaLEta[j])*(SigmaLEta[j])))+1));
            
            // Gaussian component
            double G = exp(-(0.5*(R2/(SigmaGR[j]*SigmaGR[j]))) - 
                         (0.5*(E2/(SigmaGEta[j]*SigmaGEta[j]))));
            
            // Pseudo-Voigt profile (weighted sum of Lorentzian and Gaussian)
            intPeaks += IMAX[j] * ((Mu[j]*L) + ((1-Mu[j])*G));
        }
        
        double calcIntensity = bg + intPeaks;
        totalDifferenceIntensity += pow(calcIntensity - z[i], 2);
    }
    
    return totalDifferenceIntensity;
}

/**
 * Calculate integrated intensity of fitted peaks
 */
static inline void calculateIntegratedIntensity(
    int nPeaks, double *x, double *Rs, double *Etas, 
    int nrPixelsThisRegion, double *integratedIntensity, int *nrOfPixels)
{
    double bg = x[0];
    
    // Extract peak parameters
    double IMAX[nPeaks], R[nPeaks], Eta[nPeaks], Mu[nPeaks];
    double SigmaGR[nPeaks], SigmaLR[nPeaks], SigmaGEta[nPeaks], SigmaLEta[nPeaks];
    
    for (int i = 0; i < nPeaks; i++) {
        IMAX[i] = x[(8*i)+1];
        R[i] = x[(8*i)+2];
        Eta[i] = x[(8*i)+3];
        Mu[i] = x[(8*i)+4];
        SigmaGR[i] = x[(8*i)+5];
        SigmaLR[i] = x[(8*i)+6];
        SigmaGEta[i] = x[(8*i)+7];
        SigmaLEta[i] = x[(8*i)+8];
        
        // Initialize counters
        nrOfPixels[i] = 0;
        integratedIntensity[i] = 0;
    }
    
    // Calculate for each peak
    for (int j = 0; j < nPeaks; j++) {
        for (int i = 0; i < nrPixelsThisRegion; i++) {
            double DR = Rs[i] - R[j];
            double R2 = DR * DR;
            double DE = Etas[i] - Eta[j];
            double E2 = DE * DE;
            
            // Lorentzian component
            double L = 1 / (((R2/((SigmaLR[j])*(SigmaLR[j])))+1) * 
                          ((E2/((SigmaLEta[j])*(SigmaLEta[j])))+1));
            
            // Gaussian component
            double G = exp(-(0.5*(R2/(SigmaGR[j]*SigmaGR[j]))) - 
                         (0.5*(E2/(SigmaGEta[j]*SigmaGEta[j]))));
            
            // Pseudo-Voigt profile
            double intPeaks = IMAX[j] * ((Mu[j]*L) + ((1-Mu[j])*G));
            
            // Add to integrated intensity if above background
            double bgToAdd = 0;
            if (intPeaks > bg) {
                nrOfPixels[j]++;
                bgToAdd = bg;
            }
            
            integratedIntensity[j] += (bgToAdd + intPeaks);
        }
    }
}

/**
 * Fit 2D peaks using NLopt
 */
int fit2DPeaks(
    unsigned nPeaks, int nrPixelsThisRegion, double *z, int *usefulPixels, 
    double *maximaValues, int *maximaPositions, double *integratedIntensity, 
    double *IMAX, double *YCEN, double *ZCEN, double *RCens, double *EtaCens,
    double yCen, double zCen, double thresh, int *nrPx, double *otherInfo,
    int nrPixels, double *retVal)
{
    // Total parameters: 1 background + 8 per peak
    unsigned n = 1 + (8 * nPeaks);
    double x[n], xl[n], xu[n];
    
    // Initialize background parameter
    x[0] = thresh / 2;  // Initial background level
    xl[0] = 0;          // Lower bound for background
    xu[0] = thresh;     // Upper bound for background
    
    // Calculate R and Eta coordinates for pixels in this region
    double *Rs = malloc(nrPixelsThisRegion * sizeof(*Rs));
    double *Etas = malloc(nrPixelsThisRegion * sizeof(*Etas));
    
    if (!Rs || !Etas) {
        if (Rs) free(Rs);
        if (Etas) free(Etas);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Find min/max values for R and Eta to determine constraints
    double RMin = 1e8, RMax = 0, EtaMin = 190, EtaMax = -190;
    
    for (int i = 0; i < nrPixelsThisRegion; i++) {
        // Calculate radius and angle for each pixel
        Rs[i] = CALC_NORM_2(usefulPixels[i*2+0] - yCen, usefulPixels[i*2+1] - zCen);
        Etas[i] = calcEtaAngle(-usefulPixels[i*2+0] + yCen, usefulPixels[i*2+1] - zCen);
        
        // Track min/max values
        if (Rs[i] > RMax) RMax = Rs[i];
        if (Rs[i] < RMin) RMin = Rs[i];
        if (Etas[i] > EtaMax) EtaMax = Etas[i];
        if (Etas[i] < EtaMin) EtaMin = Etas[i];
    }
    
    // Calculate maximum allowed widths
    double maxRWidth = (RMax - RMin) / 2 + 1;
    double maxEtaWidth = (EtaMax - EtaMin) / 2 + atand(2 / (RMax + RMin));
    if (EtaMax - EtaMin > 180) maxEtaWidth -= 180;
    
    // Estimate initial width based on region size and number of peaks
    double width = sqrt(nrPixelsThisRegion / nPeaks);
    if (width > maxRWidth) width = maxRWidth;
    
    // Initialize parameters for each peak
    for (int i = 0; i < nPeaks; i++) {
        // Calculate initial parameters based on maxima positions
        double peakR = CALC_NORM_2(maximaPositions[i*2+0] - yCen, maximaPositions[i*2+1] - zCen);
        double peakEta = calcEtaAngle(-maximaPositions[i*2+0] + yCen, maximaPositions[i*2+1] - zCen);
        double initSigmaEta = width / peakR;
        
        if (atand(initSigmaEta) > maxEtaWidth) {
            initSigmaEta = tand(maxEtaWidth) - 0.0001;
        }
        
        // Initial values
        x[(8*i)+1] = maximaValues[i];              // Imax
        x[(8*i)+2] = peakR;                        // Radius
        x[(8*i)+3] = peakEta;                      // Eta
        x[(8*i)+4] = 0.5;                          // Mu (mix parameter)
        x[(8*i)+5] = width;                        // SigmaGR (Gaussian sigma in R)
        x[(8*i)+6] = width;                        // SigmaLR (Lorentzian sigma in R)
        x[(8*i)+7] = atand(initSigmaEta);          // SigmaGEta (Gaussian sigma in Eta)
        x[(8*i)+8] = atand(initSigmaEta);          // SigmaLEta (Lorentzian sigma in Eta)
        
        // Calculate bounds for parameters
        double dEta = RAD2DEG * atan(1 / peakR);
        
        // Lower bounds
        xl[(8*i)+1] = maximaValues[i] / 2;         // Imax lower bound
        xl[(8*i)+2] = peakR - 1;                   // R lower bound
        xl[(8*i)+3] = peakEta - dEta;              // Eta lower bound
        xl[(8*i)+4] = 0;                           // Mu lower bound (pure Gaussian)
        xl[(8*i)+5] = 0.01;                        // SigmaGR lower bound
        xl[(8*i)+6] = 0.01;                        // SigmaLR lower bound
        xl[(8*i)+7] = 0.005;                       // SigmaGEta lower bound
        xl[(8*i)+8] = 0.005;                       // SigmaLEta lower bound
        
        // Upper bounds
        xu[(8*i)+1] = maximaValues[i] * 5;         // Imax upper bound
        xu[(8*i)+2] = peakR + 1;                   // R upper bound
        xu[(8*i)+3] = peakEta + dEta;              // Eta upper bound
        xu[(8*i)+4] = 1;                           // Mu upper bound (pure Lorentzian)
        xu[(8*i)+5] = 2 * maxRWidth;               // SigmaGR upper bound
        xu[(8*i)+6] = 2 * maxRWidth;               // SigmaLR upper bound
        xu[(8*i)+7] = 2 * maxEtaWidth;             // SigmaGEta upper bound
        xu[(8*i)+8] = 2 * maxEtaWidth;             // SigmaLEta upper bound
    }
    
    // Set up optimization
    FunctionData f_data = {
        .NrPixels = nrPixelsThisRegion,
        .Rs = Rs,
        .Etas = Etas,
        .z = z
    };
    
    // Create and configure NLopt optimizer
    nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
    if (!opt) {
        free(Rs);
        free(Etas);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    nlopt_set_lower_bounds(opt, xl);
    nlopt_set_upper_bounds(opt, xu);
    nlopt_set_maxtime(opt, 45);  // Maximum optimization time in seconds
    nlopt_set_min_objective(opt, peakFittingObjectiveFunction, &f_data);
    
    // Run optimization
    double minf;
    int rc = nlopt_optimize(opt, x, &minf);
    nlopt_destroy(opt);
    
    // Extract results
    for (int i = 0; i < nPeaks; i++) {
        IMAX[i] = x[(8*i)+1];
        RCens[i] = x[(8*i)+2];
        EtaCens[i] = x[(8*i)+3];
        
        // Store additional information
        otherInfo[8*i+0] = x[0];                   // Background
        otherInfo[8*i+1] = x[(8*i)+5];             // SigmaGR
        otherInfo[8*i+2] = x[(8*i)+6];             // SigmaLR
        otherInfo[8*i+3] = x[(8*i)+7];             // SigmaGEta
        otherInfo[8*i+4] = x[(8*i)+8];             // SigmaLEta
        otherInfo[8*i+5] = x[(8*i)+4];             // Mu
        
        // Store maximum sigma values
        otherInfo[8*i+6] = fmax(x[(8*i)+5], x[(8*i)+6]);  // Max sigma in R
        otherInfo[8*i+7] = fmax(x[(8*i)+7], x[(8*i)+8]);  // Max sigma in Eta
    }
    
    // Calculate Y and Z coordinates from R and Eta
    yzFromREta(nPeaks, RCens, EtaCens, YCEN, ZCEN);
    
    // Calculate integrated intensities
    calculateIntegratedIntensity(nPeaks, x, Rs, Etas, nrPixelsThisRegion, integratedIntensity, nrPx);
    
    // Clean up and return
    free(Rs);
    free(Etas);
    *retVal = sqrt(minf);  // RMS error
    return rc;
}

/**
 * Apply image transformations (flip/transpose) on double data
 */
static inline void applyImageTransformations_d(
    int nrTransformOptions, int transformOptions[10], 
    double *image, int nrPixels)
{
    double **imageTemp1 = allocMatrix(nrPixels, nrPixels);
    double **imageTemp2 = allocMatrix(nrPixels, nrPixels);
    
    if (!imageTemp1 || !imageTemp2) {
        if (imageTemp1) freeMatrix(imageTemp1, nrPixels);
        if (imageTemp2) freeMatrix(imageTemp2, nrPixels);
        return;
    }
    
    // Convert 1D array to 2D matrix
    for (int k = 0; k < nrPixels; k++) {
        for (int l = 0; l < nrPixels; l++) {
            imageTemp1[k][l] = image[(nrPixels*k)+l];
        }
    }
    
    // Apply each transformation in sequence
    for (int k = 0; k < nrTransformOptions; k++) {
        switch (transformOptions[k]) {
            case 1: // Flip horizontal (Y)
                for (int l = 0; l < nrPixels; l++) {
                    for (int m = 0; m < nrPixels; m++) {
                        imageTemp2[l][m] = imageTemp1[l][nrPixels-m-1];
                    }
                }
                break;
                
            case 2: // Flip vertical (Z)
                for (int l = 0; l < nrPixels; l++) {
                    for (int m = 0; m < nrPixels; m++) {
                        imageTemp2[l][m] = imageTemp1[nrPixels-l-1][m];
                    }
                }
                break;
                
            case 3: // Transpose
                for (int l = 0; l < nrPixels; l++) {
                    for (int m = 0; m < nrPixels; m++) {
                        imageTemp2[l][m] = imageTemp1[m][l];
                    }
                }
                break;
                
            case 0: // No change
            default:
                for (int l = 0; l < nrPixels; l++) {
                    for (int m = 0; m < nrPixels; m++) {
                        imageTemp2[l][m] = imageTemp1[l][m];
                    }
                }
                break;
        }
        
        // Copy result back to temp1 for next iteration
        for (int l = 0; l < nrPixels; l++) {
            for (int m = 0; m < nrPixels; m++) {
                imageTemp1[l][m] = imageTemp2[l][m];
            }
        }
    }
    
    // Convert back to 1D array
    for (int k = 0; k < nrPixels; k++) {
        for (int l = 0; l < nrPixels; l++) {
            image[(nrPixels*k)+l] = imageTemp2[k][l];
        }
    }
    
    freeMatrix(imageTemp1, nrPixels);
    freeMatrix(imageTemp2, nrPixels);
}

/**
 * Make a square image from rectangular double data
 */
static inline void makeSquareImage_d(
    int nrPixels, int nrPixelsY, int nrPixelsZ, 
    double *inImage, double *outImage)
{
    if (nrPixelsY == nrPixelsZ) {
        // Already square, just copy
        memcpy(outImage, inImage, (size_t)nrPixels * nrPixels * sizeof(double));
    } else if (nrPixelsY > nrPixelsZ) {
        // Fill along the slow direction
        memcpy(outImage, inImage, (size_t)nrPixelsY * nrPixelsZ * sizeof(double));
    } else {
        // Fill line by line
        for (int i = 0; i < nrPixelsZ; i++) {
            memcpy(outImage + (size_t)i * nrPixels, inImage + (size_t)i * nrPixelsY, (size_t)nrPixelsY * sizeof(double));
        }
    }
}

/**
 * Helper function to read a decompressed image from Zarr
 */
static inline ErrorCode readZarrImage(
    zip_t *archive, int fileIndex, char *buffer, int32_t bufferSize)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *compressedData = calloc(fileStat.size + 1, sizeof(char));
    if (!compressedData) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, compressedData, fileStat.size);
    zip_fclose(file);
    
    int32_t decompressedSize = blosc1_decompress(compressedData, buffer, bufferSize);
    free(compressedData);
    
    if (decompressedSize <= 0) {
        return ERROR_BLOSC_OPERATION;
    }
    
    return SUCCESS;
}

/**
 * Read array and decompress from zarr
 */
static inline ErrorCode readZarrArrayData(
    zip_t *archive, int fileIndex, void *dest, size_t destSize, const char *dataType)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *compressedData = calloc(fileStat.size + 1, sizeof(char));
    if (!compressedData) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, compressedData, fileStat.size);
    zip_fclose(file);
    
    int32_t decompressedSize = blosc1_decompress(compressedData, dest, destSize);
    free(compressedData);
    
    if (decompressedSize <= 0) {
        return ERROR_BLOSC_OPERATION;
    }
    
    return SUCCESS;
}

/**
 * Read dark, flat, and mask files from zarr
 */
static ErrorCode readImageCorrections(
    zip_t *archive, int darkLoc, int floodLoc, int maskLoc,
    ImageMetadata *metadata, AnalysisParams *params,
    double *dark, double *flood, double *mask)
{
    ErrorCode error;
    double *darkTemp = NULL;
    int32_t dataSize = metadata->bytesPerPx * metadata->NrPixelsY * metadata->NrPixelsZ;
    
    // Initialize correction arrays
    for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
        dark[i] = 0.0;
        flood[i] = 1.0;
        mask[i] = 0.0;
    }
    
    darkTemp = calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(*darkTemp));
    char *rawData = malloc(dataSize);
    if (!darkTemp || !rawData) {
        if (darkTemp) free(darkTemp);
        if (rawData) free(rawData);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Process dark frames
    if (metadata->nDarks > 0) {
        double *darkAsym_d = calloc((size_t)metadata->NrPixelsY * metadata->NrPixelsZ, sizeof(double));
        double *darkContents_d = calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(double));
        if (!darkAsym_d || !darkContents_d) {
            if (darkAsym_d) free(darkAsym_d);
            if (darkContents_d) free(darkContents_d);
            free(darkTemp);
            free(rawData);
            return ERROR_MEMORY_ALLOCATION;
        }

        for (int darkIter = 0; darkIter < metadata->nDarks; darkIter++) {
            error = readZarrImage(archive, darkLoc + darkIter, rawData, dataSize);
            if (error != SUCCESS) {
                // Free all buffers and return
                free(darkAsym_d);
                free(darkContents_d);
                free(darkTemp);
                free(rawData);
                return error;
            }
            
            // Convert raw data to double based on pixel type
            #pragma omp simd
            for (int i = 0; i < metadata->NrPixelsY * metadata->NrPixelsZ; i++) {
                switch(metadata->pixelType) {
                    case PX_TYPE_UINT32:
                        darkAsym_d[i] = (double)((uint32_t*)rawData)[i];
                        break;
                    case PX_TYPE_UINT16:
                        darkAsym_d[i] = (double)((uint16_t*)rawData)[i];
                        break;
                    case PX_TYPE_FLOAT:
                        darkAsym_d[i] = (double)((float*)rawData)[i];
                        break;
                    // Add other cases as needed
                    default:
                        darkAsym_d[i] = (double)((uint16_t*)rawData)[i];
                }
            }

            makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY, metadata->NrPixelsZ, darkAsym_d, darkContents_d);
            applyImageTransformations_d(params->nImTransOpt, params->TransOpt, darkContents_d, metadata->NrPixels);
            
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
        error = readZarrArrayData(archive, floodLoc, flood, (size_t)metadata->NrPixels * metadata->NrPixels * sizeof(double), "float64");
        if (error != SUCCESS) {
            free(darkTemp);
            free(rawData);
            return error;
        }
    }
    
    // Read mask
    if (maskLoc >= 0 && metadata->nMasks > 0) {
        double *maskAsym_d = calloc((size_t)metadata->NrPixelsY * metadata->NrPixelsZ, sizeof(double));
        double *maskContents_d = calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(double));
        if (!maskAsym_d || !maskContents_d) {
            if(maskAsym_d) free(maskAsym_d);
            if(maskContents_d) free(maskContents_d);
            free(darkTemp);
            free(rawData);
            return ERROR_MEMORY_ALLOCATION;
        }

        error = readZarrImage(archive, maskLoc, rawData, dataSize);
        if (error != SUCCESS) {
            free(maskAsym_d);
            free(maskContents_d);
            free(darkTemp);
            free(rawData);
            return error;
        }
        
        #pragma omp simd
        for (int i = 0; i < metadata->NrPixelsY * metadata->NrPixelsZ; i++) {
             switch(metadata->pixelType) {
                case PX_TYPE_UINT32:
                    maskAsym_d[i] = (double)((uint32_t*)rawData)[i];
                    break;
                case PX_TYPE_UINT16:
                    maskAsym_d[i] = (double)((uint16_t*)rawData)[i];
                    break;
                case PX_TYPE_FLOAT:
                    maskAsym_d[i] = (double)((float*)rawData)[i];
                    break;
                default:
                    maskAsym_d[i] = (double)((uint16_t*)rawData)[i];
            }
        }
        
        makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY, metadata->NrPixelsZ, maskAsym_d, maskContents_d);
        
        int nrMask = 0;
        for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
            mask[i] = maskContents_d[i];
            if (mask[i] > 0) nrMask++;
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
static ErrorCode processImageFrame(
    int fileNr, char *allData, size_t *sizeArr,
    ImageMetadata *metadata, AnalysisParams *params,
    double *dark, double *flood, double *mask,
    double *goodCoords, double omega, const char *outFolderName,
    const char *dataFN, ThreadWorkspace *ws)
{
    // For timing
    double t1 = omp_get_wtime();

    // The 'imgCorrBC' buffer is now part of the workspace.
    double *imgCorrBC = ws->imgCorrBC;
    
    // Allocate buffer for decompressed raw data
    int32_t dsz = metadata->NrPixelsY * metadata->NrPixelsZ * metadata->bytesPerPx;
    char *locData = calloc(dsz, sizeof(char));
    if (!locData) {
        printf("Memory allocation error in processImageFrame for locData\n");
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress the image data
    int32_t decompressedSize = blosc1_decompress(&allData[sizeArr[fileNr*2+1]], locData, dsz);
    if (decompressedSize <= 0) {
        free(locData);
        printf("Blosc decompression failed for frame %d\n", fileNr);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Allocate double-precision buffers
    double *imageAsym_d = calloc((size_t)metadata->NrPixelsY * metadata->NrPixelsZ, sizeof(double));
    double *image_d = calloc((size_t)metadata->NrPixels * metadata->NrPixels, sizeof(double));
    if (!imageAsym_d || !image_d) {
        if(imageAsym_d) free(imageAsym_d);
        if(image_d) free(image_d);
        free(locData);
        printf("Memory allocation error for double buffers in processImageFrame\n");
        return ERROR_MEMORY_ALLOCATION;
    }

    // Convert raw data to double based on its type
    #pragma omp simd
    for (int i = 0; i < metadata->NrPixelsY * metadata->NrPixelsZ; i++) {
        switch(metadata->pixelType) {
            case PX_TYPE_UINT32: imageAsym_d[i] = (double)((uint32_t*)locData)[i]; break;
            case PX_TYPE_UINT16: imageAsym_d[i] = (double)((uint16_t*)locData)[i]; break;
            case PX_TYPE_FLOAT:  imageAsym_d[i] = (double)((float*)locData)[i];   break;
            default:             imageAsym_d[i] = (double)((uint16_t*)locData)[i];
        }
    }
    free(locData); // Raw data no longer needed

    makeSquareImage_d(metadata->NrPixels, metadata->NrPixelsY, metadata->NrPixelsZ, imageAsym_d, image_d);
    free(imageAsym_d);

    if (params->makeMap == 1) {
        for (int i = 0; i < metadata->NrPixels * metadata->NrPixels; i++) {
            if (image_d[i] == params->BadPxIntensity) image_d[i] = 0;
        }
    }
    
    applyImageTransformations_d(params->nImTransOpt, params->TransOpt, image_d, metadata->NrPixels);
    transposeMatrix(image_d, metadata->NrPixels, imgCorrBC);
    free(image_d);

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
    int nrOfRegions = findConnectedComponents(ws->boolImage, metadata->NrPixels, ws->connectedComponents, ws->positions, ws->positionTrackers);
    
    char outFile[MAX_FILENAME_LENGTH];
    snprintf(outFile, MAX_FILENAME_LENGTH, "%s/%s_%06d_PS.csv", outFolderName, basename((char*)dataFN), fileNr+1);
    FILE *outfilewrite = fopen(outFile, "w");
    
    if (!outfilewrite) {
        printf("Cannot open %s for writing.\n", outFile);
        return ERROR_FILE_OPEN;
    }
    
    fprintf(outfilewrite, "SpotID\tIntegratedIntensity\tOmega(degrees)\tYCen(px)\tZCen(px)\tIMax\tRadius(px)\tEta(degrees)\tSigmaR\tSigmaEta\tNrPixels\t"
                         "TotalNrPixelsInPeakRegion\tnPeaks\tmaxY\tmaxZ\tdiffY\tdiffZ\trawIMax\treturnCode\tretVal\tBG\tSigmaGR\tSigmaLR\tSigmaGEta\t"
                         "SigmaLEta\tMU\n");
    
    int spotIdStart = 1;
    int totalValidRegions = 0;
    
    for (int regNr = 1; regNr <= nrOfRegions; regNr++) {
        int nrPixelsThisRegion = ws->positionTrackers[regNr];
        
        if (nrPixelsThisRegion <= params->minNrPx || nrPixelsThisRegion >= params->maxNrPx) {
            continue;
        }
        totalValidRegions++;
        
        for (int i = 0; i < nrPixelsThisRegion; i++) {
            ws->usefulPixels[i*2+0] = (int)(ws->positions[regNr*metadata->NrPixels*4+i] / metadata->NrPixels);
            ws->usefulPixels[i*2+1] = (int)(ws->positions[regNr*metadata->NrPixels*4+i] % metadata->NrPixels);
            ws->z[i] = imgCorrBC[((ws->usefulPixels[i*2+0]) * metadata->NrPixels) + (ws->usefulPixels[i*2+1])];
        }
        
        double thresh = goodCoords[((ws->usefulPixels[0]) * metadata->NrPixels) + (ws->usefulPixels[1])];
        
        unsigned nPeaks = findRegionalMaxima(ws->z, ws->usefulPixels, nrPixelsThisRegion, 
                                            ws->maximaPositions, ws->maximaValues, 
                                            params->IntSat, metadata->NrPixels, mask);
        
        if (nPeaks == 0) continue;
        
        if (nPeaks > params->maxNPeaks) {
            // Logic to limit number of peaks
            // This small, temporary allocation is acceptable.
            int *tempPositions = calloc(nPeaks * 2, sizeof(int));
            double *tempValues = calloc(nPeaks, sizeof(double));
            if (!tempPositions || !tempValues) { if(tempPositions) free(tempPositions); if(tempValues) free(tempValues); continue; }
            for (int i = 0; i < params->maxNPeaks; i++) {
                double maxIntMax = 0; int maxPos = 0;
                for (int j = 0; j < nPeaks; j++) {
                    if (ws->maximaValues[j] > maxIntMax) { maxPos = j; maxIntMax = ws->maximaValues[j]; }
                }
                tempPositions[i*2+0] = ws->maximaPositions[maxPos*2+0];
                tempPositions[i*2+1] = ws->maximaPositions[maxPos*2+1];
                tempValues[i] = ws->maximaValues[maxPos];
                ws->maximaValues[maxPos] = 0;
            }
            nPeaks = params->maxNPeaks;
            for (int i = 0; i < nPeaks; i++) {
                ws->maximaValues[i] = tempValues[i];
                ws->maximaPositions[i*2+0] = tempPositions[i*2+0];
                ws->maximaPositions[i*2+1] = tempPositions[i*2+1];
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
            ws->yCenArray[0] = 0; ws->zCenArray[0] = 0; ws->integratedIntensity[0] = 0;
            for (int i = 0; i < nrPixelsThisRegion; i++) {
                ws->integratedIntensity[0] += ws->z[i];
                rMeanVal += CALC_NORM_2(-ws->usefulPixels[i*2+0] + params->Ycen, ws->usefulPixels[i*2+1] - params->Zcen) * ws->z[i];
                etaMeanVal += calcEtaAngle(-ws->usefulPixels[i*2+0] + params->Ycen, ws->usefulPixels[i*2+1] - params->Zcen) * ws->z[i];
            }
            rMeanVal /= ws->integratedIntensity[0];
            etaMeanVal /= ws->integratedIntensity[0];
            yzFromREta(1, &rMeanVal, &etaMeanVal, ws->yCenArray, ws->zCenArray);
            ws->rads[0] = rMeanVal; 
            ws->etas[0] = etaMeanVal;
        } else {
            rc = fit2DPeaks(nPeaks, nrPixelsThisRegion, ws->z, ws->usefulPixels, ws->maximaValues,
                          ws->maximaPositions, ws->integratedIntensity, ws->imax, ws->yCenArray, ws->zCenArray,
                          ws->rads, ws->etas, params->Ycen, params->Zcen, thresh, ws->nrPx, ws->otherInfo, metadata->NrPixels, &retVal);
        }
        
        for (int i = 0; i < nPeaks; i++) {
            fprintf(outfilewrite, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t",
                   (spotIdStart + i), ws->integratedIntensity[i], omega,
                   -ws->yCenArray[i] + params->Ycen, ws->zCenArray[i] + params->Zcen, ws->imax[i], ws->rads[i], ws->etas[i]);
            fprintf(outfilewrite, "%f\t%f\t", ws->otherInfo[8*i+6], ws->otherInfo[8*i+7]);
            fprintf(outfilewrite, "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%lf",
                   ws->nrPx[i], nrPixelsThisRegion, nPeaks,
                   ws->maximaPositions[i*2+0], ws->maximaPositions[i*2+1],
                   (double)ws->maximaPositions[i*2+0] + ws->yCenArray[i] - params->Ycen,
                   (double)ws->maximaPositions[i*2+1] - ws->zCenArray[i] - params->Zcen,
                   ws->maximaValues[i], rc, retVal);
            for (int j = 0; j < 6; j++) {
                fprintf(outfilewrite, "\t%f", ws->otherInfo[8*i+j]);
            }
            fprintf(outfilewrite, "\n");
        }
        spotIdStart += nPeaks;
    }
    
    fclose(outfilewrite);
    
    double t3 = omp_get_wtime();
    printf("FrameNr: %d, NrOfRegions: %d, Filtered regions: %d, Number of peaks: %d, Total time: %lf\n",
           fileNr, nrOfRegions, totalValidRegions, spotIdStart-1, t3-t1);

    return SUCCESS;
}

/**
 * Read a double value from a Zarr array
 */
static ErrorCode readZarrDouble(
    zip_t *archive, int fileIndex, double *value)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *arr = calloc(fileStat.size + 1, sizeof(char));
    if (!arr) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, arr, fileStat.size);
    zip_fclose(file);
    
    // Allocate buffer for decompressed data
    int32_t dsize = sizeof(double);
    char *data = (char*)malloc((size_t)dsize);
    if (!data) {
        free(arr);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress data
    dsize = blosc1_decompress(arr, data, dsize);
    if (dsize <= 0) {
        free(arr);
        free(data);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Copy value
    *value = *(double *)&data[0];
    
    free(arr);
    free(data);
    
    return SUCCESS;
}

/**
 * Read an integer value from a Zarr array
 */
static ErrorCode readZarrInt(
    zip_t *archive, int fileIndex, int *value)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *arr = calloc(fileStat.size + 1, sizeof(char));
    if (!arr) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, arr, fileStat.size);
    zip_fclose(file);
    
    // Allocate buffer for decompressed data
    int32_t dsize = sizeof(int);
    char *data = (char*)malloc((size_t)dsize);
    if (!data) {
        free(arr);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress data
    dsize = blosc1_decompress(arr, data, dsize);
    if (dsize <= 0) {
        free(arr);
        free(data);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Copy value
    *value = *(int *)&data[0];
    
    free(arr);
    free(data);
    
    return SUCCESS;
}

/**
 * Read a string from a Zarr array
 */
static ErrorCode readZarrString(
    zip_t *archive, int fileIndex, char **value)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *arr = calloc(fileStat.size + 1, sizeof(char));
    if (!arr) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, arr, fileStat.size);
    zip_fclose(file);
    
    // Allocate buffer for decompressed data
    int32_t dsize = MAX_BUFFER_SIZE;
    *value = calloc(dsize, sizeof(char));
    if (!(*value)) {
        free(arr);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress data
    dsize = blosc1_decompress(arr, *value, dsize);
    if (dsize <= 0) {
        free(arr);
        free(*value);
        *value = NULL;
        return ERROR_BLOSC_OPERATION;
    }
    
    // Ensure null termination
    (*value)[dsize] = '\0';
    
    free(arr);
    
    return SUCCESS;
}

/**
 * Read int array from a Zarr array
 */
static ErrorCode readZarrIntArray(
    zip_t *archive, int fileIndex, int *count, int **values)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *arr = calloc(fileStat.size + 1, sizeof(char));
    if (!arr) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, arr, fileStat.size);
    zip_fclose(file);
    
    // Allocate buffer for decompressed data
    int32_t dsize = (*count) * sizeof(int);
    char *data = (char*)malloc((size_t)dsize);
    if (!data) {
        free(arr);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress data
    dsize = blosc1_decompress(arr, data, dsize);
    if (dsize <= 0) {
        free(arr);
        free(data);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Allocate and copy array
    *values = calloc(*count, sizeof(int));
    if (!(*values)) {
        free(arr);
        free(data);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < *count; i++) {
        (*values)[i] = *(int *)&data[i * sizeof(int)];
    }
    
    free(arr);
    free(data);
    
    return SUCCESS;
}

/**
 * Read double array from a Zarr array
 */
static ErrorCode readZarrDoubleArray(
    zip_t *archive, int fileIndex, int count, double **values)
{
    zip_stat_t fileStat;
    zip_stat_init(&fileStat);
    
    if (zip_stat_index(archive, fileIndex, 0, &fileStat) != 0) {
        return ERROR_ZIP_OPEN;
    }
    
    zip_file_t *file = zip_fopen_index(archive, fileIndex, 0);
    if (!file) {
        return ERROR_FILE_OPEN;
    }
    
    char *arr = calloc(fileStat.size + 1, sizeof(char));
    if (!arr) {
        zip_fclose(file);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    zip_fread(file, arr, fileStat.size);
    zip_fclose(file);
    
    // Allocate buffer for decompressed data
    int32_t dsize = count * 2 * sizeof(double);  // 2 values per entry (ring number and threshold)
    char *data = (char*)malloc((size_t)dsize);
    if (!data) {
        free(arr);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress data
    dsize = blosc1_decompress(arr, data, dsize);
    if (dsize <= 0) {
        free(arr);
        free(data);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Allocate and copy array
    *values = calloc(count * 2, sizeof(double));
    if (!(*values)) {
        free(arr);
        free(data);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(*values, data, count * 2 * sizeof(double));
    
    free(arr);
    free(data);
    
    return SUCCESS;
}

/**
 * Extract dimension from Zarr metadata
 */
static ErrorCode getZarrDimension(const char *buffer, int *dimension)
{
    char *ptr = strstr(buffer, "shape");
    if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[MAX_BUFFER_SIZE];
        strncpy(ptr3, ptrt, loc+1);
        
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
static ErrorCode readZarrDataType(
    zip_t *archive, PixelValueType *pixelType)
{
    // Default to uint16
    *pixelType = PX_TYPE_UINT16;
    
    // Look for a datatype entry in the Zarr metadata
    int count = 0;
    struct zip_stat fileInfo;
    zip_stat_init(&fileInfo);
    
    // Try to find measurement/process/scan_parameters/datatype
    while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
        if (strstr(fileInfo.name, "measurement/process/scan_parameters/datatype") != NULL) {
            char *typeName = NULL;
            ErrorCode error = readZarrString(archive, count, &typeName);
            
            if (error != SUCCESS) {
                return error;
            }
            
            // Set the pixel type based on string content
            if (typeName) {
                if (strcasecmp(typeName, "float") == 0 || strcasecmp(typeName, "float32") == 0) {
                    *pixelType = PX_TYPE_FLOAT;
                    printf("Setting pixel type to float\n");
                } else if (strcasecmp(typeName, "double") == 0 || strcasecmp(typeName, "float64") == 0) {
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

/**
 * Parse Zarr metadata and extract image parameters
 */
static ErrorCode parseZarrMetadata(
    const char *dataFile, ImageMetadata *metadata, AnalysisParams *params,
    char **resultFolder)
{
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
    metadata->pixelType = PX_TYPE_UINT16;  // Default to uint16
    metadata->omegaCenter = NULL;       // Initialize omegaCenter
    metadata->nOmegaCenterEntries = 0;  // Initialize nOmegaCenterEntries
    
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
    int dataLoc = -1;
    int floodLoc = -1;
    int maskLoc = -1;
    int locImTransOpt = -1;
    int locRingThresh = -1;
    int locOmegaRanges = -1;
    int nOmegaRanges = 0;
    int locOmegaCenterData = -1;        // To store zip index of omegaCenter data chunk
    int original_nFrames_for_omega = 0; // To store nFrames before skipFrame adjustment
    
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
                strncpy(ptr3, ptrt, loc+1);
                
                if (3 == sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nFrames, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
                    printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d\n", 
                          metadata->nFrames, metadata->NrPixelsZ, metadata->NrPixelsY);
                          original_nFrames_for_omega = metadata->nFrames; // Capture original nFrames
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
                  metadata->bytesPerPx, metadata->nFrames,
                  metadata->NrPixelsZ, metadata->NrPixelsY);
            
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
                strncpy(ptr3, ptrt, loc+1);
                if (3 != sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nDarks, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
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
                strncpy(ptr3, ptrt, loc+1);
                if (3 != sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nFloods, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
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
                strncpy(ptr3, ptrt, loc+1);
                if (3 != sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nMasks, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
                    free(buffer); 
                    zip_close(archive); 
                    free(fileInfo); 
                    return ERROR_INVALID_PARAMETERS;
                }
            }
            free(buffer);
        }
        
        // Track data locations
        if (strstr(fileInfo->name, "exchange/data/0.0.0") != NULL) dataLoc = count;
        if (strstr(fileInfo->name, "exchange/dark/0.0.0") != NULL) darkLoc = count;
        if (strstr(fileInfo->name, "exchange/mask/0.0.0") != NULL) { 
            printf("Mask is found.\n"); 
            maskLoc = count; 
        }
        if (strstr(fileInfo->name, "exchange/flood/0.0.0") != NULL) floodLoc = count;
        if (strcmp(fileInfo->name, "measurement/process/scan_parameters/omegaCenter/0") == 0) locOmegaCenterData = count;

        // Read various scalar parameters
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/start/0") != NULL) readZarrDouble(archive, count, &metadata->omegaStart);
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/step/0") != NULL) readZarrDouble(archive, count, &metadata->omegaStep);
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/doPeakFit/0") != NULL) { 
            readZarrInt(archive, count, &metadata->doPeakFit); 
            params->doPeakFit = metadata->doPeakFit;
        }
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ResultFolder/0") != NULL) readZarrString(archive, count, resultFolder);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxNPeaks/0") != NULL) readZarrInt(archive, count, &params->maxNPeaks);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/SkipFrame/0") != NULL) readZarrInt(archive, count, &metadata->skipFrame);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/zDiffThresh/0") != NULL) readZarrDouble(archive, count, &params->zDiffThresh);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tx/0") != NULL) readZarrDouble(archive, count, &params->tx);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ty/0") != NULL) readZarrDouble(archive, count, &params->ty);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tz/0") != NULL) readZarrDouble(archive, count, &params->tz);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p0/0") != NULL) readZarrDouble(archive, count, &params->p0);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p1/0") != NULL) readZarrDouble(archive, count, &params->p1);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p2/0") != NULL) readZarrDouble(archive, count, &params->p2);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p3/0") != NULL) readZarrDouble(archive, count, &params->p3);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MinNrPx/0") != NULL) readZarrInt(archive, count, &params->minNrPx);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxNrPx/0") != NULL) readZarrInt(archive, count, &params->maxNrPx);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/DoFullImage/0") != NULL) readZarrInt(archive, count, &params->DoFullImage);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ReferenceRingCurrent/0") != NULL) readZarrDouble(archive, count, &params->bc);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/YCen/0") != NULL) readZarrDouble(archive, count, &params->Ycen);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ZCen/0") != NULL) readZarrDouble(archive, count, &params->Zcen);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/UpperBoundThreshold/0") != NULL) readZarrDouble(archive, count, &params->IntSat);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/PixelSize/0") != NULL) readZarrDouble(archive, count, &params->px);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Width/0") != NULL) readZarrDouble(archive, count, &params->Width);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/LayerNr/0") != NULL) readZarrInt(archive, count, &params->LayerNr);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Wavelength/0") != NULL) readZarrDouble(archive, count, &params->Wavelength);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Lsd/0") != NULL) readZarrDouble(archive, count, &params->Lsd);
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/BadPxIntensity/0") != NULL) { 
            readZarrDouble(archive, count, &params->BadPxIntensity); 
            params->makeMap = 1; 
        }
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RhoD/0") != NULL ||
                strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxRingRad/0") != NULL) {
            readZarrDouble(archive, count, &params->RhoD);
        }
        
        // Track locations for arrays to read later
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ImTransOpt/0") != NULL) locImTransOpt = count;
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RingThresh/0.0") != NULL) locRingThresh = count;
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/OmegaRanges/0.0") != NULL) locOmegaRanges = count;
        
        // Read array dimensions
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RingThresh/.zarray") != NULL) {
            char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if(buffer) { 
                zip_file_t *f = zip_fopen_index(archive, count, 0); 
                zip_fread(f, buffer, fileInfo->size); 
                zip_fclose(f); 
                getZarrDimension(buffer, &params->nRingsThresh); 
                free(buffer); 
            }
        }
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/OmegaRanges/.zarray") != NULL) {
            char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if(buffer) { 
                zip_file_t *f = zip_fopen_index(archive, count, 0); 
                zip_fread(f, buffer, fileInfo->size); 
                zip_fclose(f); 
                getZarrDimension(buffer, &nOmegaRanges); 
                free(buffer); 
            }
        }
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ImTransOpt/.zarray") != NULL) {
             char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if(buffer) { 
                zip_file_t *f = zip_fopen_index(archive, count, 0); 
                zip_fread(f, buffer, fileInfo->size); 
                zip_fclose(f); 
                getZarrDimension(buffer, &params->nImTransOpt); 
                free(buffer); 
            }
        }
        count++;
    }
    
    if (locOmegaCenterData != -1 && original_nFrames_for_omega > 0) {
        metadata->omegaCenter = (double*)malloc((size_t)original_nFrames_for_omega * sizeof(double));
        if (metadata->omegaCenter) {
            ErrorCode err_oc = readZarrArrayData(archive, locOmegaCenterData, metadata->omegaCenter, (size_t)original_nFrames_for_omega * sizeof(double), "double_array_omegaCenter");
            if (err_oc == SUCCESS) {
                metadata->nOmegaCenterEntries = original_nFrames_for_omega;
            } else {
                free(metadata->omegaCenter);
                metadata->omegaCenter = NULL;
            }
        }
    }
    
    // Set NrPixels to max of Y and Z dimensions
    metadata->NrPixels = metadata->NrPixelsY > metadata->NrPixelsZ ?  metadata->NrPixelsY : metadata->NrPixelsZ;
    
    // Read transformation options
    if (params->nImTransOpt > 0 && locImTransOpt != -1) {
        readZarrIntArray(archive, locImTransOpt, &params->nImTransOpt, &params->TransOpt);
    }
    
    // Read ring thresholds
    if (params->nRingsThresh > 0 && locRingThresh != -1) {
        params->RingNrs = calloc(params->nRingsThresh, sizeof(int));
        params->Thresholds = calloc(params->nRingsThresh, sizeof(double));
        double *ringThresholds = NULL;
        if (params->RingNrs && params->Thresholds && readZarrDoubleArray(archive, locRingThresh, params->nRingsThresh, &ringThresholds) == SUCCESS) {
            for (int i = 0; i < params->nRingsThresh; i++) {
                params->RingNrs[i] = (int)ringThresholds[i*2+0];
                params->Thresholds[i] = ringThresholds[i*2+1];
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
static ErrorCode calculateRingRadii(
    AnalysisParams *params, const char *resultFolder, double **ringRads)
{
    *ringRads = calloc(params->nRingsThresh, sizeof(double));
    if (!(*ringRads)) return ERROR_MEMORY_ALLOCATION;
    if (params->DoFullImage == 1) return SUCCESS;
    
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
        if (11 == sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", 
                        dummy, dummy, dummy, dummy, &ringNr, dummy, 
                        dummy, dummy, dummy, dummy, &ringRad)) {
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
static ErrorCode readFrameData(
    const char *dataFile, int dataLoc, int nFrames, size_t **sizeArr, char **allData)
{
    int errorp = 0;
    zip_t *archive = zip_open(dataFile, 0, &errorp);
    if (!archive) return ERROR_ZIP_OPEN;
    
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
        (*sizeArr)[i*2+0] = fileStat.size;
        (*sizeArr)[i*2+1] = totalSize;
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
        zip_fread(file, &(*allData)[(*sizeArr)[i*2+1]], (*sizeArr)[i*2+0]);
        zip_fclose(file);
    }
    
    zip_close(archive);
    return SUCCESS;
}

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    double startTime = omp_get_wtime();
    
    if (argc < 5) {
        printf("Usage: %s DataFile blockNr nBlocks numProcs [ResultFolder] [fitPeaks]\n", argv[0]);
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
    
    ErrorCode error = parseZarrMetadata(dataFile, &metadata, &params, &resultFolder);
    if (error != SUCCESS) {
        printf("Error parsing Zarr metadata: %d\n", error);
        if(resultFolder) free(resultFolder);
        if(metadata.omegaCenter) free(metadata.omegaCenter);
        blosc2_destroy();
        return error;
    }

    if (argc > 5) { 
        if(resultFolder) free(resultFolder);
        resultFolder = strdup(argv[5]);
    }
    if (argc > 6) params.doPeakFit = atoi(argv[6]);
    
    char outFolderName[MAX_FILENAME_LENGTH];
    snprintf(outFolderName, MAX_FILENAME_LENGTH, "%s/Temp", resultFolder);
    checkDirectoryCreation(outFolderName);
    
    double *ringRads = NULL;
    if (calculateRingRadii(&params, resultFolder, &ringRads) != SUCCESS) {
        if(resultFolder) free(resultFolder);
        blosc2_destroy();
        return ERROR_FILE_OPEN;
    }
    
    // --- Detector geometry and coordinate calculation ---
    double txr = DEG2RAD * params.tx, tyr = DEG2RAD * params.ty, tzr = DEG2RAD * params.tz;
    double Rx[3][3] = {{1,0,0}, {0,cos(txr),-sin(txr)}, {0,sin(txr),cos(txr)}};
    double Ry[3][3] = {{cos(tyr),0,sin(tyr)}, {0,1,0}, {-sin(tyr),0,cos(tyr)}};
    double Rz[3][3] = {{cos(tzr),-sin(tzr),0}, {sin(tzr),cos(tzr),0}, {0,0,1}};
    double TRint[3][3], TRs[3][3];
    matrixMultiply33(Ry, Rz, TRint);
    matrixMultiply33(Rx, TRint, TRs);
    
    double *goodCoords = calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
    if (!goodCoords) return ERROR_MEMORY_ALLOCATION;
    
    if (params.DoFullImage == 1) {
        for (int a = 0; a < metadata.NrPixels * metadata.NrPixels; a++) goodCoords[a] = params.Thresholds[0];
    } else {
        #pragma omp parallel for
        for (int a = 0; a < metadata.NrPixels; a++) {
            for (int b = 0; b < metadata.NrPixels; b++) {
                double Yc = (-a + params.Ycen) * params.px, Zc = (b - params.Zcen) * params.px;
                double ABC[3] = {0, Yc, Zc}, ABCPr[3];
                matrixVectorMultiply(TRs, ABC, ABCPr);
                double XYZ[3] = {params.Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
                double Rad = (params.Lsd / XYZ[0]) * sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]);
                double Eta = calcEtaAngle(XYZ[1], XYZ[2]);
                double RNorm = Rad / params.RhoD;
                double EtaT = 90 - Eta;
                double DistortFunc = (params.p0 * pow(RNorm, 2) * cosd(2*EtaT)) + 
                                   (params.p1 * pow(RNorm, 4) * cosd(4*EtaT+params.p3)) + 
                                   (params.p2 * pow(RNorm, 2)) + 1;
                double Rt = Rad * DistortFunc / params.px;
                for (int r = 0; r < params.nRingsThresh; r++) {
                    if (Rt > ringRads[r] - params.Width && Rt < ringRads[r] + params.Width) {
                        goodCoords[(a*metadata.NrPixels)+b] = params.Thresholds[r];
                    }
                }
            }
        }
    }
    
    int startFileNr = (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * blockNr;
    int endFileNr = (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * (blockNr+1);
    if (endFileNr > metadata.nFrames) endFileNr = metadata.nFrames;
    printf("Processing frames %d to %d\n", startFileNr, endFileNr);
    
    // --- Read Correction and Image Data (Shared by all threads) ---
    int errorp = 0;
    zip_t *archive = zip_open(dataFile, 0, &errorp);
    if (!archive) return ERROR_ZIP_OPEN;
    
    double *dark = calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
    double *mask = calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
    double *flood = calloc((size_t)metadata.NrPixels * metadata.NrPixels, sizeof(double));
    
    int darkLoc = -1, floodLoc = -1, maskLoc = -1, count = 0;
    struct zip_stat fileInfo;
    zip_stat_init(&fileInfo);
    while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
        if (strstr(fileInfo.name, "exchange/dark/0.0.0") != NULL) darkLoc = count;
        if (strstr(fileInfo.name, "exchange/mask/0.0.0") != NULL) maskLoc = count;
        if (strstr(fileInfo.name, "exchange/flood/0.0.0") != NULL) floodLoc = count;
        count++;
    }
    darkLoc += metadata.skipFrame;
    
    error = readImageCorrections(archive, darkLoc, floodLoc, maskLoc, &metadata, &params, dark, flood, mask);
    zip_close(archive);
    if (error != SUCCESS) { /* cleanup and exit */ return error; }

    // Re-find dataLoc as archive was closed and re-opened implicitly in other functions
    int dataLoc = -1;
    count = 0;
    archive = zip_open(dataFile, 0, &errorp);
    zip_stat_init(&fileInfo);
    while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
        if (strstr(fileInfo.name, "exchange/data/0.0.0") != NULL) dataLoc = count;
        count++;
    }
    zip_close(archive);
    dataLoc += metadata.skipFrame;
    
    size_t *sizeArr = NULL;
    char *allData = NULL;
    error = readFrameData(dataFile, dataLoc, metadata.nFrames, &sizeArr, &allData);
    if (error != SUCCESS) { /* cleanup and exit */ return error; }
    
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
                fprintf(stderr, "FATAL: Memory allocation for thread workspace failed.\n");
            }
        } else {
            // 3. Begin the parallel work distribution.
            #pragma omp for schedule(dynamic)
            for (int fileNr = startFileNr; fileNr < endFileNr; fileNr++) {
                int current_original_frame_idx = fileNr + metadata.skipFrame;
                double omega;
                if (metadata.omegaCenter && current_original_frame_idx < metadata.nOmegaCenterEntries) {
                    omega = metadata.omegaCenter[current_original_frame_idx];
                } else {
                    omega = metadata.omegaStart + (double)current_original_frame_idx * metadata.omegaStep;
                }
                
                ErrorCode threadError = processImageFrame(
                    fileNr, allData, sizeArr, &metadata, &params,
                    dark, flood, mask, goodCoords,
                    omega, outFolderName, dataFile, &ws); // Pass workspace pointer
                
                #pragma omp critical
                if (threadError == SUCCESS) nrFilesDone++;
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
    if(ringRads) free(ringRads);
    if(params.TransOpt) free(params.TransOpt);
    if(params.RingNrs) free(params.RingNrs);
    if(params.Thresholds) free(params.Thresholds);
    if(resultFolder) free(resultFolder);
    if(metadata.omegaCenter) free(metadata.omegaCenter);
    
    blosc2_destroy();
    
    double totalTime = omp_get_wtime() - startTime;
    printf("Finished, time elapsed: %lf seconds, nrFramesDone: %d.\n", totalTime, nrFilesDone);
    
    return SUCCESS;
}