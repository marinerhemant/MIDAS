//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  PeaksFittingOMPZarr.c - Improved version
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

// Default to uint16_t for pixel values, but this can be changed dynamically
typedef uint16_t pixelvalue;

// Add a type enum to support dynamic pixel value typing
typedef enum {
    PX_TYPE_UINT16 = 0,
    PX_TYPE_INT32 = 1,
    PX_TYPE_FLOAT = 2,
    PX_TYPE_DOUBLE = 3
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
 * Allocate a 2D matrix of pixel values
 */
static inline
pixelvalue** allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
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
 * Free a 2D matrix of pixel values
 */
static inline
void freeMatrixPx(pixelvalue **mat, int nrows)
{
    if (mat == NULL) return;
    
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
    if (y > 0) alpha = -alpha;
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
    if (mat == NULL) return;
    
    for (int r = 0; r < nrows; r++) {
        if (mat[r] != NULL) {
            free(mat[r]);
        }
    }
    free(mat);
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
    if (mat == NULL) return;
    
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
 * Transpose a square matrix
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
    if (!stack) return;
    
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
        
        if (x < 0 || x >= nrPixels || y < 0 || y >= nrPixels) continue;
        if (connectedComponents[x*nrPixels+y] != 0 || boolImage[x*nrPixels+y] == 0) continue;
        
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
 * Apply image transformations (flip/transpose)
 */
static inline void applyImageTransformations(
    int nrTransformOptions, int transformOptions[10], 
    pixelvalue *image, int nrPixels)
{
    pixelvalue **imageTemp1 = allocMatrixPX(nrPixels, nrPixels);
    pixelvalue **imageTemp2 = allocMatrixPX(nrPixels, nrPixels);
    
    if (!imageTemp1 || !imageTemp2) {
        if (imageTemp1) freeMatrixPx(imageTemp1, nrPixels);
        if (imageTemp2) freeMatrixPx(imageTemp2, nrPixels);
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
    
    freeMatrixPx(imageTemp1, nrPixels);
    freeMatrixPx(imageTemp2, nrPixels);
}

/**
 * Make a square image from rectangular data
 */
static inline void makeSquareImage(
    int nrPixels, int nrPixelsY, int nrPixelsZ, 
    pixelvalue *inImage, pixelvalue *outImage)
{
    if (nrPixelsY == nrPixelsZ) {
        // Already square, just copy
        memcpy(outImage, inImage, nrPixels * nrPixels * sizeof(*inImage));
    } else if (nrPixelsY > nrPixelsZ) {
        // Fill along the slow direction
        memcpy(outImage, inImage, nrPixelsY * nrPixelsZ * sizeof(*inImage));
    } else {
        // Fill line by line
        for (int i = 0; i < nrPixelsZ; i++) {
            memcpy(outImage + i * nrPixelsZ, inImage + i * nrPixelsY, nrPixelsY * sizeof(*inImage));
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
    int nDarks, int nFloods, int nrPixels, int nrPixelsY, int nrPixelsZ,
    size_t bytesPerPx, int nImTransOpt, int *transOpt,
    double *dark, double *flood, double *mask)
{
    ErrorCode error;
    pixelvalue *darkAsym, *darkContents, *maskTemp, *maskTmp;
    double *darkTemp;
    int32_t dataSize = bytesPerPx * nrPixelsY * nrPixelsZ;
    
    // Initialize correction arrays
    for (int i = 0; i < nrPixels * nrPixels; i++) {
        dark[i] = 0.0;
        flood[i] = 1.0;  // Default to 1.0 for flood (flat field)
        mask[i] = 0.0;   // Default to 0.0 for mask (no masked pixels)
    }
    
    // Allocate temporary arrays
    darkAsym = calloc(nrPixelsY * nrPixelsZ, sizeof(*darkAsym));
    darkContents = calloc(nrPixels * nrPixels, sizeof(*darkContents));
    darkTemp = calloc(nrPixels * nrPixels, sizeof(*darkTemp));
    maskTemp = calloc(nrPixelsY * nrPixelsZ, sizeof(*maskTemp));
    maskTmp = calloc(nrPixels * nrPixels, sizeof(*maskTmp));
    
    if (!darkAsym || !darkContents || !darkTemp || !maskTemp || !maskTmp) {
        if (darkAsym) free(darkAsym);
        if (darkContents) free(darkContents);
        if (darkTemp) free(darkTemp);
        if (maskTemp) free(maskTemp);
        if (maskTmp) free(maskTmp);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Read dark frames
    char *data = malloc(dataSize);
    if (!data) {
        free(darkAsym);
        free(darkContents);
        free(darkTemp);
        free(maskTemp);
        free(maskTmp);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Process dark frames
    for (int darkIter = 0; darkIter < nDarks; darkIter++) {
        // Read dark frame
        error = readZarrImage(archive, darkLoc + darkIter, data, dataSize);
        if (error != SUCCESS) {
            free(data);
            free(darkAsym);
            free(darkContents);
            free(darkTemp);
            free(maskTemp);
            free(maskTmp);
            return error;
        }
        
        // Copy to image array
        memcpy(darkAsym, data, dataSize);
        
        // Make square and apply transformations
        makeSquareImage(nrPixels, nrPixelsY, nrPixelsZ, darkAsym, darkContents);
        applyImageTransformations(nImTransOpt, transOpt, darkContents, nrPixels);
        
        // Accumulate for averaging
        for (int i = 0; i < nrPixels * nrPixels; i++) {
            darkTemp[i] += darkContents[i];
        }
    }
    
    // Average dark frames
    if (nDarks > 0) {
        for (int i = 0; i < nrPixels * nrPixels; i++) {
            darkTemp[i] /= nDarks;
        }
    }
    
    // Transpose dark frame
    transposeMatrix(darkTemp, nrPixels, dark);
    
    // Read flat field (flood) frame
    if (nFloods > 0) {
        double *floodTemp = calloc(nrPixels * nrPixels, sizeof(*floodTemp));
        if (!floodTemp) {
            free(data);
            free(darkAsym);
            free(darkContents);
            free(darkTemp);
            free(maskTemp);
            free(maskTmp);
            return ERROR_MEMORY_ALLOCATION;
        }
        
        error = readZarrArrayData(archive, floodLoc, floodTemp, nrPixels * nrPixels * sizeof(double), "float64");
        if (error != SUCCESS) {
            free(data);
            free(darkAsym);
            free(darkContents);
            free(darkTemp);
            free(maskTemp);
            free(maskTmp);
            free(floodTemp);
            return error;
        }
        
        memcpy(flood, floodTemp, nrPixels * nrPixels * sizeof(double));
        free(floodTemp);
    }
    
    // Read mask
    if (maskLoc >= 0) {
        error = readZarrImage(archive, maskLoc, data, dataSize);
        if (error != SUCCESS) {
            free(data);
            free(darkAsym);
            free(darkContents);
            free(darkTemp);
            free(maskTemp);
            free(maskTmp);
            return error;
        }
        
        memcpy(maskTemp, data, dataSize);
        makeSquareImage(nrPixels, nrPixelsY, nrPixelsZ, maskTemp, maskTmp);
        
        int nrMask = 0;
        for (int i = 0; i < nrPixels * nrPixels; i++) {
            mask[i] = maskTmp[i];
            if (maskTmp[i] > 0) nrMask++;
        }
        
        printf("Number of mask pixels: %d\n", nrMask);
    }
    
    // Clean up
    free(data);
    free(darkAsym);
    free(darkContents);
    free(darkTemp);
    free(maskTemp);
    free(maskTmp);
    
    return SUCCESS;
}

/**
 * Process a single image frame
 */
static ErrorCode processImageFrame(
    int fileNr, char *allData, size_t *sizeArr,
    int nrPixels, int nrPixelsY, int nrPixelsZ, size_t bytesPerPx,
    pixelvalue *image, double *imgCorrBC, double *dark, double *flood, double *mask,
    double *goodCoords, double yCen, double zCen, double intSat, double bc,
    int minNrPx, int maxNrPx, int maxNPeaks, int makeMap, double badPxIntensity,
    int nImTransOpt, int *transOpt, double omega, const char *outFolderName,
    const char *dataFN, int doPeakFit)
{
    // For timing
    double t1 = omp_get_wtime();
    
    // Temporary image arrays
    pixelvalue *imageAsym = calloc(nrPixelsY * nrPixelsZ, sizeof(*imageAsym));
    double *imgCorrBCTemp = calloc(nrPixels * nrPixels, sizeof(*imgCorrBCTemp));
    
    if (!imageAsym || !imgCorrBCTemp) {
        if (imageAsym) free(imageAsym);
        if (imgCorrBCTemp) free(imgCorrBCTemp);
        printf("Memory allocation error in processImageFrame\n");
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Decompress the image data
    int32_t dsz = nrPixelsY * nrPixelsZ * bytesPerPx;
    char *locData = calloc(dsz, sizeof(char));
    if (!locData) {
        free(imageAsym);
        free(imgCorrBCTemp);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    double t2 = omp_get_wtime();
    int32_t decompressedSize = blosc1_decompress(&allData[sizeArr[fileNr*2+1]], locData, dsz);
    
    if (decompressedSize <= 0) {
        free(imageAsym);
        free(imgCorrBCTemp);
        free(locData);
        return ERROR_BLOSC_OPERATION;
    }
    
    // Copy decompressed data to image array
    memcpy(imageAsym, locData, dsz);
    free(locData);
    
    // Make square image from rectangular data
    makeSquareImage(nrPixels, nrPixelsY, nrPixelsZ, imageAsym, image);
    
    // Handle bad pixels if needed
    if (makeMap == 1) {
        int badPxCounter = 0;
        for (int i = 0; i < nrPixels * nrPixels; i++) {
            if (image[i] == (pixelvalue)badPxIntensity) {
                image[i] = 0;
                badPxCounter++;
            }
        }
    }
    
    // Apply transformations (flip/transpose)
    applyImageTransformations(nImTransOpt, transOpt, image, nrPixels);
    
    // Convert to double for processing
    for (int i = 0; i < (nrPixels * nrPixels); i++) {
        imgCorrBCTemp[i] = image[i];
    }
    
    // Transpose for processing
    transposeMatrix(imgCorrBCTemp, nrPixels, imgCorrBC);
    
    // Apply thresholds, dark and flat field corrections
    for (int i = 0; i < (nrPixels * nrPixels); i++) {
        if (goodCoords[i] == 0) {
            imgCorrBC[i] = 0;
        } else {
            // Apply dark and flat field corrections
            imgCorrBC[i] = (imgCorrBC[i] - dark[i]) / flood[i];
            imgCorrBC[i] = imgCorrBC[i] * bc;
            
            // Apply threshold
            if (imgCorrBC[i] < goodCoords[i]) {
                imgCorrBC[i] = 0;
            }
        }
    }
    
    // Arrays for connected components analysis
    int *boolImage = calloc(nrPixels * nrPixels, sizeof(*boolImage));
    int *connectedComponents = calloc(nrPixels * nrPixels, sizeof(*connectedComponents));
    int *positions = calloc(MAX_OVERLAPS_PER_IMAGE * nrPixels * 4, sizeof(*positions));
    int *positionTrackers = calloc(MAX_OVERLAPS_PER_IMAGE, sizeof(*positionTrackers));
    
    if (!boolImage || !connectedComponents || !positions || !positionTrackers) {
        printf("Memory allocation error in processImageFrame\n");
        if (imageAsym) free(imageAsym);
        if (imgCorrBCTemp) free(imgCorrBCTemp);
        if (boolImage) free(boolImage);
        if (connectedComponents) free(connectedComponents);
        if (positions) free(positions);
        if (positionTrackers) free(positionTrackers);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Create binary image for connected components
    for (int i = 0; i < nrPixels * nrPixels; i++) {
        boolImage[i] = (imgCorrBC[i] != 0) ? 1 : 0;
    }
    
    // Find connected components
    memset(positionTrackers, 0, MAX_OVERLAPS_PER_IMAGE * sizeof(*positionTrackers));
    int nrOfRegions = findConnectedComponents(boolImage, nrPixels, connectedComponents, positions, positionTrackers);
    
    // Arrays for peak detection and fitting
    int *usefulPixels = calloc(nrPixels * 20, sizeof(*usefulPixels));
    int *maximaPositions = calloc(nrPixels * 20, sizeof(*maximaPositions));
    double *maximaValues = calloc(nrPixels * 10, sizeof(*maximaValues));
    double *z = calloc(nrPixels * 10, sizeof(*z));
    
    if (!usefulPixels || !maximaPositions || !maximaValues || !z) {
        printf("Memory allocation error in processImageFrame\n");
        if (imageAsym) free(imageAsym);
        if (imgCorrBCTemp) free(imgCorrBCTemp);
        if (boolImage) free(boolImage);
        if (connectedComponents) free(connectedComponents);
        if (positions) free(positions);
        if (positionTrackers) free(positionTrackers);
        if (usefulPixels) free(usefulPixels);
        if (maximaPositions) free(maximaPositions);
        if (maximaValues) free(maximaValues);
        if (z) free(z);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Arrays for peak fitting results
    double *integratedIntensity = calloc(maxNPeaks * 2, sizeof(*integratedIntensity));
    double *imax = calloc(maxNPeaks * 2, sizeof(*imax));
    double *yCenArray = calloc(maxNPeaks * 2, sizeof(*yCenArray));
    double *zCenArray = calloc(maxNPeaks * 2, sizeof(*zCenArray));
    double *rads = calloc(maxNPeaks * 2, sizeof(*rads));
    double *etas = calloc(maxNPeaks * 2, sizeof(*etas));
    int *nrPx = calloc(maxNPeaks * 2, sizeof(*nrPx));
    double *otherInfo = calloc(maxNPeaks * 10, sizeof(*otherInfo));
    
    if (!integratedIntensity || !imax || !yCenArray || !zCenArray || !rads || !etas || !nrPx || !otherInfo) {
        printf("Memory allocation error in processImageFrame\n");
        if (imageAsym) free(imageAsym);
        if (imgCorrBCTemp) free(imgCorrBCTemp);
        if (boolImage) free(boolImage);
        if (connectedComponents) free(connectedComponents);
        if (positions) free(positions);
        if (positionTrackers) free(positionTrackers);
        if (usefulPixels) free(usefulPixels);
        if (maximaPositions) free(maximaPositions);
        if (maximaValues) free(maximaValues);
        if (z) free(z);
        if (integratedIntensity) free(integratedIntensity);
        if (imax) free(imax);
        if (yCenArray) free(yCenArray);
        if (zCenArray) free(zCenArray);
        if (rads) free(rads);
        if (etas) free(etas);
        if (nrPx) free(nrPx);
        if (otherInfo) free(otherInfo);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Create output file
    char outFile[MAX_FILENAME_LENGTH];
    snprintf(outFile, MAX_FILENAME_LENGTH, "%s/%s_%06d_PS.csv", outFolderName, basename((char*)dataFN), fileNr+1);
    FILE *outfilewrite = fopen(outFile, "w");
    
    if (!outfilewrite) {
        printf("Cannot open %s for writing.\n", outFile);
        if (imageAsym) free(imageAsym);
        if (imgCorrBCTemp) free(imgCorrBCTemp);
        if (boolImage) free(boolImage);
        if (connectedComponents) free(connectedComponents);
        if (positions) free(positions);
        if (positionTrackers) free(positionTrackers);
        if (usefulPixels) free(usefulPixels);
        if (maximaPositions) free(maximaPositions);
        if (maximaValues) free(maximaValues);
        if (z) free(z);
        if (integratedIntensity) free(integratedIntensity);
        if (imax) free(imax);
        if (yCenArray) free(yCenArray);
        if (zCenArray) free(zCenArray);
        if (rads) free(rads);
        if (etas) free(etas);
        if (nrPx) free(nrPx);
        if (otherInfo) free(otherInfo);
        return ERROR_FILE_OPEN;
    }
    
    // Write CSV header
    fprintf(outfilewrite, "SpotID\tIntegratedIntensity\tOmega(degrees)\tYCen(px)\tZCen(px)\tIMax\tRadius(px)\tEta(degrees)\tSigmaR\tSigmaEta\tNrPixels\t"
                         "TotalNrPixelsInPeakRegion\tnPeaks\tmaxY\tmaxZ\tdiffY\tdiffZ\trawIMax\treturnCode\tretVal\tBG\tSigmaGR\tSigmaLR\tSigmaGEta\t"
                         "SigmaLEta\tMU\n");
    
    // Process each connected component
    int spotIdStart = 1;
    int totalValidRegions = 0;
    
    for (int regNr = 1; regNr <= nrOfRegions; regNr++) {
        int nrPixelsThisRegion = positionTrackers[regNr];
        
        // Skip regions that are too small or too large
        if (nrPixelsThisRegion <= minNrPx || nrPixelsThisRegion >= maxNrPx) {
            continue;
        }
        
        totalValidRegions++;
        
        // Extract pixel values for this region
        for (int i = 0; i < nrPixelsThisRegion; i++) {
            usefulPixels[i*2+0] = (int)(positions[regNr*nrPixels*4+i] / nrPixels);
            usefulPixels[i*2+1] = (int)(positions[regNr*nrPixels*4+i] % nrPixels);
            z[i] = imgCorrBC[((usefulPixels[i*2+0]) * nrPixels) + (usefulPixels[i*2+1])];
        }
        
        // Get threshold for this region
        double thresh = goodCoords[((usefulPixels[0*2+0]) * nrPixels) + (usefulPixels[0*2+1])];
        
        // Find peaks in this region
        unsigned nPeaks = findRegionalMaxima(z, usefulPixels, nrPixelsThisRegion, 
                                            maximaPositions, maximaValues, 
                                            intSat, nrPixels, mask);
        
        if (nPeaks == 0) {
            // Saturated peak or touched mask
            continue;
        }
        
        // Limit number of peaks if needed
        if (nPeaks > maxNPeaks) {
            // Sort peaks by intensity and keep the strongest ones
            int *tempPositions = calloc(nPeaks * 2, sizeof(int));
            double *tempValues = calloc(nPeaks, sizeof(double));
            
            if (!tempPositions || !tempValues) {
                if (tempPositions) free(tempPositions);
                if (tempValues) free(tempValues);
                printf("Memory allocation error\n");
                continue;
            }
            
            // Find highest peaks
            for (int i = 0; i < maxNPeaks; i++) {
                double maxIntMax = 0;
                int maxPos = 0;
                
                for (int j = 0; j < nPeaks; j++) {
                    if (maximaValues[j] > maxIntMax) {
                        maxPos = j;
                        maxIntMax = maximaValues[j];
                    }
                }
                
                tempPositions[i*2+0] = maximaPositions[maxPos*2+0];
                tempPositions[i*2+1] = maximaPositions[maxPos*2+1];
                tempValues[i] = maximaValues[maxPos];
                maximaValues[maxPos] = 0;  // Mark as used
            }
            
            // Update number of peaks and copy back values
            nPeaks = maxNPeaks;
            for (int i = 0; i < nPeaks; i++) {
                maximaValues[i] = tempValues[i];
                maximaPositions[i*2+0] = tempPositions[i*2+0];
                maximaPositions[i*2+1] = tempPositions[i*2+1];
            }
            
            free(tempPositions);
            free(tempValues);
        }
        
        double retVal = 0;
        int rc = 0;
        
        // Perform peak fitting or center of mass calculation
        if (doPeakFit == 0) {
            // Calculate center of mass if no fitting requested
            double *rMean = calloc(2, sizeof(double));
            double *etaMean = calloc(2, sizeof(double));
            
            if (!rMean || !etaMean) {
                if (rMean) free(rMean);
                if (etaMean) free(etaMean);
                printf("Memory allocation error\n");
                continue;
            }
            
            // Just use the strongest peak
            nPeaks = 1;
            imax[0] = maximaValues[0];
            nrPx[0] = nrPixelsThisRegion;
            yCenArray[0] = 0;
            zCenArray[0] = 0;
            integratedIntensity[0] = 0;
            
            // Calculate weighted center of mass
            for (int i = 0; i < nrPixelsThisRegion; i++) {
                integratedIntensity[0] += z[i];
                rMean[0] += CALC_NORM_2(-usefulPixels[i*2+0] + yCen, usefulPixels[i*2+1] - zCen) * z[i];
                etaMean[0] += calcEtaAngle(-usefulPixels[i*2+0] + yCen, usefulPixels[i*2+1] - zCen) * z[i];
            }
            
            rMean[0] /= integratedIntensity[0];
            etaMean[0] /= integratedIntensity[0];
            
            // Convert R,Eta to Y,Z
            yzFromREta(1, rMean, etaMean, yCenArray, zCenArray);
            rads[0] = rMean[0];
            etas[0] = etaMean[0];
            
            free(rMean);
            free(etaMean);
        } else {
            // Perform 2D peak fitting
            rc = fit2DPeaks(nPeaks, nrPixelsThisRegion, z, usefulPixels, maximaValues,
                          maximaPositions, integratedIntensity, imax, yCenArray, zCenArray,
                          rads, etas, yCen, zCen, thresh, nrPx, otherInfo, nrPixels, &retVal);
        }
        
        // Write results to CSV
        for (int i = 0; i < nPeaks; i++) {
            fprintf(outfilewrite, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t",
                   (spotIdStart + i), integratedIntensity[i], omega,
                   -yCenArray[i] + yCen, zCenArray[i] + zCen, imax[i], rads[i], etas[i]);
            
            // Write sigma values
            for (int j = 0; j < 2; j++) {
                fprintf(outfilewrite, "%f\t", otherInfo[8*i+6+j]);
            }
            
            // Write peak information
            fprintf(outfilewrite, "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%lf",
                   nrPx[i], nrPixelsThisRegion, nPeaks,
                   maximaPositions[i*2+0], maximaPositions[i*2+1],
                   (double)maximaPositions[i*2+0] + yCenArray[i] - yCen,
                   (double)maximaPositions[i*2+1] - zCenArray[i] - zCen,
                   maximaValues[i], rc, retVal);
            
            // Write additional fit parameters
            for (int j = 0; j < 6; j++) {
                fprintf(outfilewrite, "\t%f", otherInfo[8*i+j]);
            }
            
            fprintf(outfilewrite, "\n");
        }
        
        spotIdStart += nPeaks;
    }
    
    free(yCenArray);
    free(zCenArray);
    fclose(outfilewrite);
    
    double t3 = omp_get_wtime();
    printf("FrameNr: %d, NrOfRegions: %d, Filtered regions: %d, Number of peaks: %d, Total time: %lf\n",
           fileNr, nrOfRegions, totalValidRegions, spotIdStart-1, t3-t1);
    
    // Free all allocated memory
    free(imageAsym);
    free(imgCorrBCTemp);
    free(boolImage);
    free(connectedComponents);
    free(positions);
    free(positionTrackers);
    free(usefulPixels);
    free(maximaPositions);
    free(maximaValues);
    free(z);
    free(integratedIntensity);
    free(imax);
    free(rads);
    free(etas);
    free(nrPx);
    free(otherInfo);
    
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
    
    // Try to find measurement/scan/datatype
    while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
        if (strstr(fileInfo.name, "measurement/scan/datatype") != NULL) {
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
    
    struct zip_stat *fileInfo = calloc(16384, sizeof(int));
    if (!fileInfo) {
        zip_close(archive);
        return ERROR_MEMORY_ALLOCATION;
    }
    zip_stat_init(fileInfo);
    
    // Initialize default values
    metadata->nFrames = 0;
    metadata->nDarks = 0;
    metadata->nFloods = 0;
    metadata->nMasks = 0;
    metadata->NrPixelsY = 0;
    metadata->NrPixelsZ = 0;
    metadata->bytesPerPx = 0;
    metadata->omegaStart = 0;
    metadata->omegaStep = 0;
    metadata->skipFrame = 0;
    metadata->doPeakFit = 1;
    metadata->pixelType = PX_TYPE_UINT16;  // Default to uint16
    
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
                } else {
                    free(buffer);
                    zip_close(archive);
                    free(fileInfo);
                    return ERROR_INVALID_PARAMETERS;
                }
            }
            
            // Parse data type
            ptr = strstr(buffer, "dtype");
            if (ptr != NULL) {
                char *ptrt = strstr(ptr, ":");
                char *ptr2 = strstr(ptrt, ",");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[MAX_BUFFER_SIZE];
                strncpy(ptr3, ptrt+3, loc-4);
                
                if (strncmp(ptr3, "<u2", 3) == 0) metadata->bytesPerPx = 2;
                if (strncmp(ptr3, "<u4", 3) == 0) metadata->bytesPerPx = 4;
                
                // Adjust bytes per pixel based on dynamic type if needed
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
            
            // Parse shape
            char *ptr = strstr(buffer, "shape");
            if (ptr != NULL) {
                char *ptrt = strstr(ptr, "[");
                char *ptr2 = strstr(ptrt, "]");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[MAX_BUFFER_SIZE];
                strncpy(ptr3, ptrt, loc+1);
                
                if (3 == sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nDarks, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
                    printf("nDarks: %d nrPixelsZ: %d nrPixelsY: %d\n", 
                          metadata->nDarks, metadata->NrPixelsZ, metadata->NrPixelsY);
                } else {
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
            
            // Parse shape
            char *ptr = strstr(buffer, "shape");
            if (ptr != NULL) {
                char *ptrt = strstr(ptr, "[");
                char *ptr2 = strstr(ptrt, "]");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[MAX_BUFFER_SIZE];
                strncpy(ptr3, ptrt, loc+1);
                
                if (3 == sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nFloods, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
                    printf("nFloods: %d nrPixelsZ: %d nrPixelsY: %d\n", 
                          metadata->nFloods, metadata->NrPixelsZ, metadata->NrPixelsY);
                } else {
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
            
            // Parse shape
            char *ptr = strstr(buffer, "shape");
            if (ptr != NULL) {
                char *ptrt = strstr(ptr, "[");
                char *ptr2 = strstr(ptrt, "]");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[MAX_BUFFER_SIZE];
                strncpy(ptr3, ptrt, loc+1);
                
                if (3 == sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", 
                              &metadata->nMasks, &metadata->NrPixelsZ, &metadata->NrPixelsY)) {
                    printf("nMasks: %d nrPixelsZ: %d nrPixelsY: %d\n", 
                          metadata->nMasks, metadata->NrPixelsZ, metadata->NrPixelsY);
                } else {
                    free(buffer);
                    zip_close(archive);
                    free(fileInfo);
                    return ERROR_INVALID_PARAMETERS;
                }
            }
            
            free(buffer);
        }
        
        // Track data locations
        if (strstr(fileInfo->name, "exchange/data/0.0.0") != NULL) {
            dataLoc = count;
        }
        if (strstr(fileInfo->name, "exchange/dark/0.0.0") != NULL) {
            darkLoc = count;
        }
        if (strstr(fileInfo->name, "exchange/mask/0.0.0") != NULL) {
            printf("Mask is found, we will separate mask and saturated intensity. Please ensure saturated intensity is different from mask pixels\n");
            maskLoc = count;
        }
        if (strstr(fileInfo->name, "exchange/flood/0.0.0") != NULL) {
            floodLoc = count;
        }

        // Read parameters
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/start/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &metadata->omegaStart);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("OmeStart: %lf\n", metadata->omegaStart);
        }
        
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/step/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &metadata->omegaStep);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("OmeStep: %lf\n", metadata->omegaStep);
        }
        
        if (strstr(fileInfo->name, "measurement/process/scan_parameters/doPeakFit/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &metadata->doPeakFit);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            params->doPeakFit = metadata->doPeakFit;
            printf("doPeakFit: %d\n", metadata->doPeakFit);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
            ErrorCode error = readZarrString(archive, count, resultFolder);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("resultFolder: %s\n", *resultFolder);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxNPeaks/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &params->maxNPeaks);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("maxNPeaks: %d\n", params->maxNPeaks);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/SkipFrame/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &metadata->skipFrame);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("skipFrame: %d\n", metadata->skipFrame);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/zDiffThresh/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->zDiffThresh);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("zDiffThresh: %lf\n", params->zDiffThresh);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tx/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->tx);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("tx: %lf\n", params->tx);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ty/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->ty);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("ty: %lf\n", params->ty);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/tz/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->tz);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("tz: %lf\n", params->tz);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p0/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->p0);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("p0: %lf\n", params->p0);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p1/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->p1);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("p1: %lf\n", params->p1);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p2/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->p2);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("p2: %lf\n", params->p2);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/p3/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->p3);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("p3: %lf\n", params->p3);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MinNrPx/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &params->minNrPx);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("minNrPx: %d\n", params->minNrPx);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxNrPx/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &params->maxNrPx);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("maxNrPx: %d\n", params->maxNrPx);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/DoFullImage/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &params->DoFullImage);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("DoFullImage: %d\n", params->DoFullImage);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ReferenceRingCurrent/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->bc);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("bc: %lf\n", params->bc);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/YCen/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->Ycen);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("Ycen: %lf\n", params->Ycen);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ZCen/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->Zcen);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("Zcen: %lf\n", params->Zcen);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/UpperBoundThreshold/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->IntSat);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("IntSat: %lf\n", params->IntSat);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/PixelSize/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->px);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("px: %lf\n", params->px);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Width/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->Width);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("Width: %lf\n", params->Width);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/LayerNr/0") != NULL) {
            ErrorCode error = readZarrInt(archive, count, &params->LayerNr);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("LayerNr: %d\n", params->LayerNr);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Wavelength/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->Wavelength);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("Wavelength: %lf\n", params->Wavelength);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/Lsd/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->Lsd);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("Lsd: %lf\n", params->Lsd);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/BadPxIntensity/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->BadPxIntensity);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            params->makeMap = 1;
            printf("BadPxIntensity: %lf, makeMap: %d\n", params->BadPxIntensity, params->makeMap);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/MaxRingRad/0") != NULL || 
            strstr(fileInfo->name, "analysis/process/analysis_parameters/RhoD/0") != NULL) {
            ErrorCode error = readZarrDouble(archive, count, &params->RhoD);
            if (error != SUCCESS) {
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            printf("RhoD: %lf\n", params->RhoD);
        }
        
        // Track locations for arrays to read later
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ImTransOpt/0") != NULL) {
            locImTransOpt = count;
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RingThresh/0.0") != NULL) {
            locRingThresh = count;
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/OmegaRanges/0.0") != NULL) {
            locOmegaRanges = count;
        }
        
        // Read array dimensions
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/RingThresh/.zarray") != NULL) {
            char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if (!buffer) {
                zip_close(archive);
                free(fileInfo);
                return ERROR_MEMORY_ALLOCATION;
            }
            
            zip_file_t *file = zip_fopen_index(archive, count, 0);
            zip_fread(file, buffer, fileInfo->size);
            zip_fclose(file);
            
            ErrorCode error = getZarrDimension(buffer, &params->nRingsThresh);
            if (error != SUCCESS) {
                free(buffer);
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            
            printf("nRingsThresh: %d\n", params->nRingsThresh);
            free(buffer);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/OmegaRanges/.zarray") != NULL) {
            char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if (!buffer) {
                zip_close(archive);
                free(fileInfo);
                return ERROR_MEMORY_ALLOCATION;
            }
            
            zip_file_t *file = zip_fopen_index(archive, count, 0);
            zip_fread(file, buffer, fileInfo->size);
            zip_fclose(file);
            
            ErrorCode error = getZarrDimension(buffer, &nOmegaRanges);
            if (error != SUCCESS) {
                free(buffer);
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            
            printf("nOmegaRanges: %d\n", nOmegaRanges);
            free(buffer);
        }
        
        if (strstr(fileInfo->name, "analysis/process/analysis_parameters/ImTransOpt/.zarray") != NULL) {
            char *buffer = calloc(fileInfo->size + 1, sizeof(char));
            if (!buffer) {
                zip_close(archive);
                free(fileInfo);
                return ERROR_MEMORY_ALLOCATION;
            }
            
            zip_file_t *file = zip_fopen_index(archive, count, 0);
            zip_fread(file, buffer, fileInfo->size);
            zip_fclose(file);
            
            ErrorCode error = getZarrDimension(buffer, &params->nImTransOpt);
            if (error != SUCCESS) {
                free(buffer);
                zip_close(archive);
                free(fileInfo);
                return error;
            }
            
            printf("nImTransOpt: %d\n", params->nImTransOpt);
            free(buffer);
        }
        
        count++;
    }
    
    // Set NrPixels to max of Y and Z dimensions
    if (metadata->NrPixelsY != metadata->NrPixelsZ) {
        metadata->NrPixels = metadata->NrPixelsY > metadata->NrPixelsZ ? 
                             metadata->NrPixelsY : metadata->NrPixelsZ;
    } else {
        metadata->NrPixels = metadata->NrPixelsY;
    }
    
    // Read transformation options
    if (params->nImTransOpt > 0) {
        params->TransOpt = calloc(params->nImTransOpt, sizeof(int));
        if (!params->TransOpt) {
            zip_close(archive);
            free(fileInfo);
            return ERROR_MEMORY_ALLOCATION;
        }
        
        ErrorCode error = readZarrIntArray(archive, locImTransOpt, &params->nImTransOpt, &params->TransOpt);
        if (error != SUCCESS) {
            zip_close(archive);
            free(fileInfo);
            return error;
        }
        
        for (int i = 0; i < params->nImTransOpt; i++) {
            printf("TransOpt[%d]: %d\n", i, params->TransOpt[i]);
            
            if (params->TransOpt[i] < 0 || params->TransOpt[i] > 3) {
                printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
                zip_close(archive);
                free(fileInfo);
                return ERROR_INVALID_PARAMETERS;
            }
            
            printf("TransformationOptions: %d ", params->TransOpt[i]);
            if (params->TransOpt[i] == 0)
                printf("No change.\n");
            else if (params->TransOpt[i] == 1)
                printf("Flip Left Right.\n");
            else if (params->TransOpt[i] == 2)
                printf("Flip Top Bottom.\n");
            else
                printf("Transpose.\n");
        }
    }
    
    // Read ring thresholds
    if (params->nRingsThresh > 0 && locRingThresh != -1) {
        params->RingNrs = calloc(params->nRingsThresh, sizeof(int));
        params->Thresholds = calloc(params->nRingsThresh, sizeof(double));
        
        if (!params->RingNrs || !params->Thresholds) {
            if (params->RingNrs) free(params->RingNrs);
            if (params->Thresholds) free(params->Thresholds);
            if (params->TransOpt) free(params->TransOpt);
            zip_close(archive);
            free(fileInfo);
            return ERROR_MEMORY_ALLOCATION;
        }
        
        double *ringThresholds = NULL;
        ErrorCode error = readZarrDoubleArray(archive, locRingThresh, params->nRingsThresh, &ringThresholds);
        if (error != SUCCESS) {
            if (params->RingNrs) free(params->RingNrs);
            if (params->Thresholds) free(params->Thresholds);
            if (params->TransOpt) free(params->TransOpt);
            zip_close(archive);
            free(fileInfo);
            return error;
        }
        
        for (int i = 0; i < params->nRingsThresh; i++) {
            params->RingNrs[i] = (int)ringThresholds[i*2+0];
            params->Thresholds[i] = ringThresholds[i*2+1];
            printf("RingNr[%d]: %d, Threshold: %lf\n", i, params->RingNrs[i], params->Thresholds[i]);
        }
        
        free(ringThresholds);
    }
    
    // Update global variable
    zDiffThresh = params->zDiffThresh;
    
    // Adjust frame counts for skipped frames
    metadata->nFrames -= metadata->skipFrame;
    metadata->nDarks -= metadata->skipFrame;
    dataLoc += metadata->skipFrame;
    darkLoc += metadata->skipFrame;
    
    // Clean up
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
    // Allocate memory for ring radii
    *ringRads = calloc(params->nRingsThresh, sizeof(double));
    if (!(*ringRads)) {
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // If using full image, we don't need ring radii
    if (params->DoFullImage == 1) {
        return SUCCESS;
    }
    
    // Open HKL file
    char hklFileName[MAX_FILENAME_LENGTH];
    snprintf(hklFileName, MAX_FILENAME_LENGTH, "%s/hkls.csv", resultFolder);
    
    FILE *hklFile = fopen(hklFileName, "r");
    if (!hklFile) {
        printf("HKL file %s could not be read. Exiting\n", hklFileName);
        free(*ringRads);
        *ringRads = NULL;
        return ERROR_FILE_OPEN;
    }
    
    // Read header line
    char line[MAX_LINE_LENGTH];
    if (!fgets(line, MAX_LINE_LENGTH, hklFile)) {
        fclose(hklFile);
        free(*ringRads);
        *ringRads = NULL;
        return ERROR_FILE_OPEN;
    }
    
    // Read ring data
    int ringNr;
    double ringRad;
    char dummy[MAX_LINE_LENGTH];
    
    while (fgets(line, MAX_LINE_LENGTH, hklFile)) {
        // Parse line to get ring number and radius
        if (11 == sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", 
                        dummy, dummy, dummy, dummy, &ringNr, dummy, 
                        dummy, dummy, dummy, dummy, &ringRad)) {
            
            // Look for matching ring number in our list
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
    if (!archive) {
        return ERROR_ZIP_OPEN;
    }
    
    // Allocate array to hold offsets and sizes
    *sizeArr = calloc(nFrames * 2, sizeof(size_t));
    if (!(*sizeArr)) {
        zip_close(archive);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // First pass to get total size and frame sizes
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
        
        (*sizeArr)[i*2+0] = fileStat.size;     // Size of compressed data
        (*sizeArr)[i*2+1] = totalSize;         // Offset in buffer
        totalSize += fileStat.size;
    }
    
    // Allocate buffer for all compressed data
    *allData = calloc(totalSize + 1, sizeof(char));
    if (!(*allData)) {
        free(*sizeArr);
        *sizeArr = NULL;
        zip_close(archive);
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Read all frame data
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
    
    // Check command line parameters
    if (argc < 5) {
        printf("Usage: %s DataFile blockNr nBlocks numProcs [ResultFolder] [fitPeaks]\n", argv[0]);
        printf("If fitPeaks(0) is provided, MUST provide RESULTFOLDER!!!!!\n");
        return ERROR_INVALID_PARAMETERS;
    }
    
    // Parse command line arguments
    char *dataFile = argv[1];
    int blockNr = atoi(argv[2]);
    int nBlocks = atoi(argv[3]);
    int numProcs = atoi(argv[4]);
    
    // Optional arguments
    char *resultFolder = NULL;
    int doPeakFit = 1;
    
    if (argc > 5) resultFolder = strdup(argv[5]);
    if (argc > 6) doPeakFit = atoi(argv[6]);
    
    // Initialize Blosc compression library
    blosc2_init();
    
    // Parse the Zarr metadata and load parameters
    ImageMetadata metadata;
    AnalysisParams params;
    
    ErrorCode error = parseZarrMetadata(dataFile, &metadata, &params, &resultFolder);
    if (error != SUCCESS) {
        printf("Error parsing Zarr metadata: %d\n", error);
        if (resultFolder) free(resultFolder);
        blosc2_destroy();
        return error;
    }
    
    // Override doPeakFit if specified in command line
    if (argc > 6) {
        params.doPeakFit = doPeakFit;
    }
    
    // Create output directory
    char outFolderName[MAX_FILENAME_LENGTH];
    snprintf(outFolderName, MAX_FILENAME_LENGTH, "%s/Temp", resultFolder);
    printf("Output folder: %s\n", outFolderName);
    
    error = checkDirectoryCreation(outFolderName);
    if (error != SUCCESS) {
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        blosc2_destroy();
        return error;
    }
    
    // Calculate ring radii
    double *ringRads = NULL;
    error = calculateRingRadii(&params, resultFolder, &ringRads);
    if (error != SUCCESS) {
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        blosc2_destroy();
        return error;
    }
    
    // Calculate rotation matrices for detector tilts
    double txr = DEG2RAD * params.tx;
    double tyr = DEG2RAD * params.ty;
    double tzr = DEG2RAD * params.tz;
    
    double Rx[3][3] = {{1,0,0}, {0,cos(txr),-sin(txr)}, {0,sin(txr),cos(txr)}};
    double Ry[3][3] = {{cos(tyr),0,sin(tyr)}, {0,1,0}, {-sin(tyr),0,cos(tyr)}};
    double Rz[3][3] = {{cos(tzr),-sin(tzr),0}, {sin(tzr),cos(tzr),0}, {0,0,1}};
    
    double TRint[3][3], TRs[3][3];
    matrixMultiply33(Ry, Rz, TRint);
    matrixMultiply33(Rx, TRint, TRs);
    
    // Determine the coordinates to process
    double *goodCoords = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*goodCoords));
    if (!goodCoords) {
        if (ringRads) free(ringRads);
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        blosc2_destroy();
        return ERROR_MEMORY_ALLOCATION;
    }
    
    int nrCoords = 0;
    
    if (params.DoFullImage == 1) {
        // Process entire image with first threshold
        for (int a = 0; a < metadata.NrPixels * metadata.NrPixels; a++) {
            goodCoords[a] = params.Thresholds[0];
        }
        nrCoords = metadata.NrPixels * metadata.NrPixels;
    } else {
        // Process only pixels within ring regions
        for (int a = 1; a < metadata.NrPixels; a++) {
            for (int b = 1; b < metadata.NrPixels; b++) {
                // Convert detector coordinates to lab coordinates
                double Yc = (-a + params.Ycen) * params.px;
                double Zc = (b - params.Zcen) * params.px;
                
                double ABC[3] = {0, Yc, Zc};
                double ABCPr[3];
                matrixVectorMultiply(TRs, ABC, ABCPr);
                
                double XYZ[3] = {params.Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
                
                // Calculate radius and angle
                double Rad = (params.Lsd / (XYZ[0])) * sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]);
                double Eta = calcEtaAngle(XYZ[1], XYZ[2]);
                
                // Apply distortion correction
                double RNorm = Rad / params.RhoD;
                double EtaT = 90 - Eta;
                double n0 = 2, n1 = 4, n2 = 2;
                
                double DistortFunc = (params.p0 * pow(RNorm, n0) * cos(DEG2RAD*(2*EtaT))) + 
                                   (params.p1 * pow(RNorm, n1) * cos(DEG2RAD*(4*EtaT+params.p3))) + 
                                   (params.p2 * pow(RNorm, n2)) + 1;
                
                double Rt = Rad * DistortFunc / params.px;
                
                // Check if pixel is within any ring
                for (int thisRingNr = 0; thisRingNr < params.nRingsThresh; thisRingNr++) {
                    if (Rt > ringRads[thisRingNr] - params.Width && 
                        Rt < ringRads[thisRingNr] + params.Width) {
                        goodCoords[((a-1)*metadata.NrPixels)+(b-1)] = params.Thresholds[thisRingNr];
                        nrCoords++;
                    }
                }
            }
        }
    }
    
    printf("Number of coordinates to process: %d\n", nrCoords);
    
    // Calculate job distribution
    int startFileNr = (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * blockNr;
    int endFileNr = (int)(ceil((double)metadata.nFrames / (double)nBlocks)) * (blockNr+1);
    
    if (endFileNr > metadata.nFrames) {
        endFileNr = metadata.nFrames;
    }
    
    int nrJobs = (int)(ceil((double)(endFileNr - startFileNr) / (double)(numProcs)));
    
    printf("StartFileNr: %d EndFileNr: %d numProcs: %d nrJobs/proc: %d blockNr: %d nrBlocks: %d\n",
          startFileNr, endFileNr, numProcs, nrJobs, blockNr, nBlocks);
    
    // Open Zarr archive to read corrections (dark, flat, mask)
    int errorp = 0;
    zip_t *archive = zip_open(dataFile, 0, &errorp);
    if (!archive) {
        if (ringRads) free(ringRads);
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        free(goodCoords);
        blosc2_destroy();
        return ERROR_ZIP_OPEN;
    }
    
    // Allocate memory for correction images
    double *dark = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*dark));
    double *mask = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*mask));
    double *flood = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*flood));
    
    if (!dark || !mask || !flood) {
        if (dark) free(dark);
        if (mask) free(mask);
        if (flood) free(flood);
        zip_close(archive);
        if (ringRads) free(ringRads);
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        free(goodCoords);
        blosc2_destroy();
        return ERROR_MEMORY_ALLOCATION;
    }
    
    // Find data locations in archive
    struct zip_stat fileInfo;
    zip_stat_init(&fileInfo);
    
    int darkLoc = -1, dataLoc = -1, floodLoc = -1, maskLoc = -1;
    int count = 0;
    
    while (zip_stat_index(archive, count, 0, &fileInfo) == 0) {
        if (strstr(fileInfo.name, "exchange/data/0.0.0") != NULL) {
            dataLoc = count;
        }
        if (strstr(fileInfo.name, "exchange/dark/0.0.0") != NULL) {
            darkLoc = count;
        }
        if (strstr(fileInfo.name, "exchange/mask/0.0.0") != NULL) {
            maskLoc = count;
        }
        if (strstr(fileInfo.name, "exchange/flood/0.0.0") != NULL) {
            floodLoc = count;
        }
        count++;
    }
    
    // Apply frame skipping
    dataLoc += metadata.skipFrame;
    darkLoc += metadata.skipFrame;
    
    // Read correction images
    error = readImageCorrections(
        archive, darkLoc, floodLoc, maskLoc,
        metadata.nDarks, metadata.nFloods, metadata.NrPixels,
        metadata.NrPixelsY, metadata.NrPixelsZ, metadata.bytesPerPx,
        params.nImTransOpt, params.TransOpt,
        dark, flood, mask);
    
    zip_close(archive);
    
    if (error != SUCCESS) {
        free(dark);
        free(mask);
        free(flood);
        if (ringRads) free(ringRads);
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        free(goodCoords);
        blosc2_destroy();
        return error;
    }
    printf("%lf\n",dark[0]);
    // Read frame data
    size_t *sizeArr = NULL;
    char *allData = NULL;
    
    error = readFrameData(dataFile, dataLoc, metadata.nFrames, &sizeArr, &allData);
    if (error != SUCCESS) {
        free(dark);
        free(mask);
        free(flood);
        if (ringRads) free(ringRads);
        if (params.TransOpt) free(params.TransOpt);
        if (params.RingNrs) free(params.RingNrs);
        if (params.Thresholds) free(params.Thresholds);
        if (resultFolder) free(resultFolder);
        free(goodCoords);
        blosc2_destroy();
        return error;
    }
    
    // Process image frames in parallel
    int nrFilesDone = 0;
    
    #pragma omp parallel for num_threads(numProcs) shared(nrFilesDone) schedule(dynamic)
    for (int fileNr = startFileNr; fileNr < endFileNr; fileNr++) {
        // Allocate memory for this thread
        pixelvalue *image = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*image));
        double *imgCorrBC = calloc(metadata.NrPixels * metadata.NrPixels, sizeof(*imgCorrBC));
        
        if (!image || !imgCorrBC) {
            if (image) free(image);
            if (imgCorrBC) free(imgCorrBC);
            printf("Memory allocation error for thread\n");
            continue;
        }
        
        // Calculate omega angle for this frame
        double omega = metadata.omegaStart + fileNr * metadata.omegaStep;
        
        // Process the image frame
        ErrorCode threadError = processImageFrame(
            fileNr, allData, sizeArr,
            metadata.NrPixels, metadata.NrPixelsY, metadata.NrPixelsZ, metadata.bytesPerPx,
            image, imgCorrBC, dark, flood, mask, goodCoords,
            params.Ycen, params.Zcen, params.IntSat, params.bc,
            params.minNrPx, params.maxNrPx, params.maxNPeaks, params.makeMap, params.BadPxIntensity,
            params.nImTransOpt, params.TransOpt, omega, outFolderName, dataFile, params.doPeakFit);
        
        // Cleanup thread resources
        free(image);
        free(imgCorrBC);
        
        #pragma omp critical
        {
            if (threadError == SUCCESS) {
                nrFilesDone++;
            }
        }
    }
    
    // Cleanup
    free(dark);
    free(flood);
    free(mask);
    free(goodCoords);
    free(sizeArr);
    free(allData);
    if (ringRads) free(ringRads);
    if (params.TransOpt) free(params.TransOpt);
    if (params.RingNrs) free(params.RingNrs);
    if (params.Thresholds) free(params.Thresholds);
    if (resultFolder) free(resultFolder);
    
    // Clean up Blosc
    blosc2_destroy();
    
    double totalTime = omp_get_wtime() - startTime;
    printf("Finished, time elapsed: %lf seconds, nrFramesDone: %d.\n", totalTime, nrFilesDone);
    
    return SUCCESS;
}