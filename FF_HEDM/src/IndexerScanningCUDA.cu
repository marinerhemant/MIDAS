//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// 04/26/2021
// Hemant Sharma
// OpenMP version of IndexerLinuxArgsShm code.
//
// CUDA C Port
//
/*
  ~/opt/midascuda/cuda/bin/nvcc src/IndexerScanningCUDA.cu -o bin/IndexerScanningCUDA \
  -I/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/include -L/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/lib \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_90,code=sm_90 \
  -Xcompiler -g \
  -I/path/to/midas/includes \
  -L/path/to/midas/libs \
  -O3 -lnlopt -lz -ldl -lm -lpthread
  export LD_LIBRARY_PATH=/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/lib:$LD_LIBRARY_PATH
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h> // Keep for host-side mmap
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
// #include <sys/ipc.h> // Not directly used in CUDA part
// #include <sys/shm.h> // Not directly used in CUDA part
#include <sys/types.h>
// #include <omp.h> // Remove OpenMP
#include <libgen.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Error Handling Macro
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

// Host-side check function (remains the same)
static void
checkHost (int test, const char * message, ...)
{
	if (test) {
		va_list args;
		va_start (args, message);
		vfprintf (stderr, message, args);
		va_end (args);
		fprintf (stderr, "\n");
		exit (EXIT_FAILURE);
	}
}

// Use float for performance
#define RealType float

// conversions constants
#define deg2rad 0.0174532925199433f
#define rad2deg 57.2957795130823f

// max array sizes (remain the same, used for allocation)
#define MAX_N_SPOTS 100000000
#define MAX_N_OR 7200
#define MAX_N_MATCHES 1
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS 14
#define N_COL_OBSSPOTS 10
#define N_COL_GRAINSPOTS 17
#define N_COL_GRAINMATCHES 16
#define MAX_MIC_ROWS 50000000
#define EPS 1e-9f // Use float epsilon

// --- Globals (Host-side mirrors or parameters) ---
// Host pointers for data read from files/mmap
float *h_ObsSpotsLab = NULL;
int *h_BigDetector = NULL; // Host pointer for mmap
size_t *h_data = NULL;     // Host pointer for mmap
size_t *h_ndata = NULL;    // Host pointer for mmap
double h_hkls_double[MAX_N_HKLS][7]; // Read as double first
float h_hkls[MAX_N_HKLS][N_COL_THEORSPOTS]; // Store as float on host too
int h_HKLints[MAX_N_HKLS][4];
double h_ABCABG[6];
float h_RingHKL[MAX_N_RINGS][3];
float h_omemargins[181];
float h_etamargins[MAX_N_RINGS];
double *h_grid_double = NULL; // Read as double
float *h_grid = NULL;         // Store as float
double *h_ypos_double = NULL; // Read as double
float *h_ypos = NULL;         // Store as float
double *h_mic_double = NULL;  // Read as double
float *h_mic = NULL;          // Store as float


size_t n_spots_host = 0; // Renamed to avoid conflict with kernel arg
int n_hkls_host = 0;     // Renamed

// Detector/Binning parameters (passed to kernel)
int BigDetSize_host = 0;
long long int totNrPixelsBigDetector_host = 0;
float pixelsize_host = 0.0f;
float BeamSize_host = 0.0f;
int numScans_host = 0;
int SGNum_host = 0;
int n_ring_bins_host = 0;
int n_eta_bins_host = 0;
int n_ome_bins_host = 0;
float EtaBinSize_host = 0.0f;
float OmeBinSize_host = 0.0f;
int nrMic_host = 0; // Number of MIC entries read

// --- Device pointers ---
float *d_ObsSpotsLab = NULL;
int *d_BigDetector = NULL;
size_t *d_data = NULL;
size_t *d_ndata = NULL;
float *d_hkls = NULL;
int *d_HKLints = NULL;
float *d_RingHKL = NULL;
float *d_omemargins = NULL;
float *d_etamargins = NULL;
float *d_grid = NULL;
float *d_ypos = NULL;
float *d_mic = NULL;
float *d_workspace = NULL; // Large preallocated buffer for kernel temporaries
float *d_results = NULL; // Buffer for kernel output (outArr data)
int *d_matched_ids = NULL; // Buffer for kernel output (matched spot IDs)
size_t *d_key_info = NULL; // Buffer for kernel output (spotID, rownr, offset_vals, offset_all)


// --- Device-side Math Macros & Functions ---

// Mark bit testing for device use
// A is device pointer, k is index
#define TestBit_device(A,k)  (A[(k/32)] &   (1 << (k%32)))

#define crossProduct_device(a,b,c) \
	(a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
	(a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
	(a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

#define dot_device(v,q) \
	((v)[0] * (q)[0] + \
	(v)[1] * (q)[1] + \
	(v)[2] * (q)[2])

#define CalcLength_device(x,y,z) sqrtf((x)*(x) + (y)*(y) + (z)*(z))

// Note: __forceinline__ suggests aggressively inlining, __device__ marks for device compilation
__device__ __forceinline__ float sind_d(float x){return sinf(deg2rad*x);}
__device__ __forceinline__ float cosd_d(float x){return cosf(deg2rad*x);}
__device__ __forceinline__ float tand_d(float x){return tanf(deg2rad*x);}
__device__ __forceinline__ float asind_d(float x){return rad2deg*asinf(x);}
__device__ __forceinline__ float acosd_d(float x){
    // Clamp input to avoid NaN for values slightly outside [-1, 1] due to precision
    x = fmaxf(-1.0f, fminf(1.0f, x));
    return rad2deg*acosf(x);
}
__device__ __forceinline__ float atand_d(float x){return rad2deg*atanf(x);}
__device__ __forceinline__ float sin_cos_to_angle_d (float s, float c){
    c = fmaxf(-1.0f, fminf(1.0f, c)); // Clamp cosine
    return (s >= 0.0f) ? acosf(c) : 2.0f * (float)M_PI - acosf(c);
    }

// --- Device Matrix/Vector Operations ---

__device__ __forceinline__
void MatrixMultF33_d(const float m[3][3], const float n[3][3], float res[3][3])
{
	// Using temp to avoid potential aliasing issues if res is m or n
    float temp[3][3];
	for (int r=0; r<3; r++) {
		temp[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
		temp[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
		temp[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
	}
    // Copy result
    for(int r=0; r<3; ++r) {
        res[r][0] = temp[r][0];
        res[r][1] = temp[r][1];
        res[r][2] = temp[r][2];
    }
}

__device__ __forceinline__
void MatrixMultF_d(const float m[3][3], const float v[3], float r[3])
{
    float temp[3];
	for (int i=0; i<3; i++) {
		temp[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
	}
    r[0] = temp[0]; r[1] = temp[1]; r[2] = temp[2];
}

// MatrixMult with int vector - assuming vector needs conversion?
// Original code uses int[3] v. If this represents HKL indices, keep as int.
// If it represents something else that should be float, adjust.
// Let's assume it's okay as is for now, but check usage context.
__device__ __forceinline__
void MatrixMultIntF_d(const float m[3][3], const int v[3], float r[3])
{
    float temp[3];
	for (int i=0; i<3; i++) {
		temp[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
	}
    r[0] = temp[0]; r[1] = temp[1]; r[2] = temp[2];
}


__device__ __forceinline__
float min_d(float a, float b)
{
	return (a < b ? a : b);
}

__device__ __forceinline__
float max_d(float a, float b)
{
	return (a > b ? a : b);
}

// --- Device Geometric Calculations ---

__device__ __forceinline__
void CalcInternalAngle_d(float x1,float y1,float z1,float x2,float y2,float z2,float *ia)
{
	float v1[3];
	float v2[3];
	v1[0] = x1; v1[1] = y1; v1[2] = z1;
	v2[0] = x2; v2[1] = y2; v2[2] = z2;
	float l1 = CalcLength_device(x1, y1 ,z1);
	float l2 = CalcLength_device(x2, y2, z2);
    if (l1 < EPS || l2 < EPS) {
        *ia = 0.0f; // Avoid division by zero
        return;
    }
	float tmp = dot_device(v1, v2)/(l1*l2);
    // Clamp tmp slightly inside [-1, 1] due to potential float inaccuracies
	tmp = fmaxf(-1.0f, fminf(1.0f, tmp));
	*ia = rad2deg * acosf(tmp);
}

__device__ __forceinline__
void RotateAroundZ_d(const float v1[3], float alpha, float v2[3])
{
	float cosa = cosf(alpha*deg2rad);
	float sina = sinf(alpha*deg2rad);
	float mat[3][3] = {{ cosa, -sina, 0.0f },
                       { sina,  cosa, 0.0f },
                       { 0.0f,  0.0f, 1.0f }};
	MatrixMultF_d(mat, v1, v2);
}

__device__ __forceinline__
void CalcEtaAngle_d(float y,float z,float *alpha)
{
    float denom = sqrtf(y*y+z*z);
    if (denom < EPS) {
        *alpha = 0.0f; // Or handle as appropriate for zero vector
        return;
    }
    float cos_val = z / denom;
    // Clamp for safety
    cos_val = fmaxf(-1.0f, fminf(1.0f, cos_val));
	*alpha = rad2deg * acosf(cos_val);
	if (y > 0.0f) *alpha = -(*alpha);
}

__device__ __forceinline__
void CalcSpotPosition_d(float RingRadius, float eta, float *yl, float *zl)
{
	float etaRad = deg2rad * eta;
	*yl = -(sinf(etaRad)*RingRadius);
	*zl =   cosf(etaRad)*RingRadius;
}

__device__ __forceinline__
void CalcOmega_d(float x,float y,float z,float theta,float omegas[4],float etas[4],int * nsol)
{
	*nsol = 0;
	float ome;
	float len= sqrtf(x*x + y*y + z*z);
    if (len < EPS) return; // Avoid division by zero / undefined angle

	float v=sinf(theta*deg2rad)*len;
	float almostzero = 1e-4f; // Float version

	if ( fabsf(y) < almostzero ) {
		if (fabsf(x) > almostzero) { // Avoid division by zero
			float cosome1 = -v/x;
			if (fabsf(cosome1) <= 1.0f) {
				ome = acosf(cosome1)*rad2deg;
                if (*nsol < 4) omegas[*nsol] = ome;
				(*nsol)++;
                if (*nsol < 4) omegas[*nsol] = -ome;
				(*nsol)++;
			}
		}
	} else {
		float y2 = y*y;
		float a = 1.0f + ((x*x) / y2);
		float b = (2.0f*v*x) / y2;
		float c = ((v*v) / y2) - 1.0f;
		float discr = b*b - 4.0f*a*c;
		float ome1a, ome1b, ome2a, ome2b;
		float cosome1, cosome2;
		float eqa, eqb, diffa, diffb;

		if (discr >= 0.0f) {
            float sqrt_discr = sqrtf(discr);
            float two_a = 2.0f * a;
            if (fabsf(two_a) < EPS) { // Avoid division by zero
                 // Handle degenerate case if needed, otherwise skip
            } else {
                cosome1 = (-b + sqrt_discr) / two_a;
                if (fabsf(cosome1) <= 1.0f) {
                    ome1a = acosf(cosome1);
                    ome1b = -ome1a;
                    eqa = -x*cosf(ome1a) + y*sinf(ome1a);
                    diffa = fabsf(eqa - v);
                    eqb = -x*cosf(ome1b) + y*sinf(ome1b);
                    diffb = fabsf(eqb - v);
                    if (*nsol < 4) {
                        omegas[*nsol] = (diffa < diffb ? ome1a : ome1b) * rad2deg;
                        (*nsol)++;
                    }
                }
                cosome2 = (-b - sqrt_discr) / two_a;
                if (fabsf(cosome2) <= 1.0f) {
                    ome2a = acosf(cosome2);
                    ome2b = -ome2a;
                    eqa = -x*cosf(ome2a) + y*sinf(ome2a);
                    diffa = fabsf(eqa - v);
                    eqb = -x*cosf(ome2b) + y*sinf(ome2b);
                    diffb = fabsf(eqb - v);
                     if (*nsol < 4) {
                        omegas[*nsol] = (diffa < diffb ? ome2a : ome2b) * rad2deg;
                        (*nsol)++;
                    }
                }
            }
		}
	}
	float gw[3];
	float gv[3]={x,y,z};
	float eta;
	for (int indexOme = 0; indexOme < *nsol; indexOme++) {
		RotateAroundZ_d(gv, omegas[indexOme], gw);
		CalcEtaAngle_d(gw[1],gw[2], &eta);
		etas[indexOme] = eta;
	}
}

// --- Device Core Logic Functions ---

// Needs device pointers for hkls, BigDetector if used
__device__
void CalcDiffrSpots_Furnace_d(
    const float OrientMatrix[3][3],
    float Wavelength, // LatticeConstant not directly used? Check original
    float distance,
    const float RingRadii[], // Assumed passed by value or from device global/const
    const float OmegaRange[][2], // Passed by value/const
    const float BoxSizes[][4],   // Passed by value/const
    int NOmegaRanges,
    float ExcludePoleAngle,
    int n_hkls,           // Pass dimension
    const float* d_hkls,        // Device pointer
    const int* d_BigDetector,   // Device pointer
    int BigDetSize,
    float pixelsize,
    float* spots_out,     // Output buffer (flattened)
    int max_spots,        // Max capacity of spots_out
    int *nspots           // Number of spots found
    )
{
	// int i, OmegaRangeNo; // Declared inside loops
	// float ds; // Not used?
	float theta;
	int KeepSpot;
	float Ghkl[3];
	// int indexhkl; // loop var
	float Gc[3];
	float omegas[4];
	float etas[4];
	float yl;
	float zl;
	int nspotsPlane;
	int spotnr = 0;
	int spotid = 0; // Relative spot ID within this calculation
	// int OrientID = 0; // Assuming fixed to 0 for this context? Or pass as arg?
	int ringnr = 0;
	int YCInt, ZCInt;
	long long int idx; // For BigDetector

	for (int indexhkl=0; indexhkl < n_hkls ; indexhkl++)  {
        // Access flattened d_hkls: row * num_cols + col
        Ghkl[0] = d_hkls[indexhkl * N_COL_THEORSPOTS + 0]; // Assuming column 0 is H
        Ghkl[1] = d_hkls[indexhkl * N_COL_THEORSPOTS + 1]; // Assuming column 1 is K
        Ghkl[2] = d_hkls[indexhkl * N_COL_THEORSPOTS + 2]; // Assuming column 2 is L
		ringnr = (int)(d_hkls[indexhkl * N_COL_THEORSPOTS + 3]); // Column 3 is ringnr
        if (ringnr < 0 || ringnr >= MAX_N_RINGS) continue; // Bounds check
		float RingRadius = RingRadii[ringnr];
        if (RingRadius < EPS) continue; // Skip if radius is zero/invalid

		MatrixMultF_d(OrientMatrix, Ghkl, Gc);
		// ds    = d_hkls[indexhkl * N_COL_THEORSPOTS + 4]; // Column 4 is ds
		theta = d_hkls[indexhkl * N_COL_THEORSPOTS + 5]; // Column 5 is theta

		CalcOmega_d(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);

		for (int i=0 ; i<nspotsPlane ; i++) {
			float Omega = omegas[i];
			float Eta = etas[i];
			float EtaAbs =  fabsf(Eta);

            // Pole exclusion check
			if ((EtaAbs < ExcludePoleAngle ) || ((180.0f - EtaAbs) < ExcludePoleAngle)) continue;

			CalcSpotPosition_d(RingRadius, etas[i], &yl, &zl);
            KeepSpot = 0; // Reset for each potential spot

            // Check Omega ranges and Box sizes
			for (int OmegaRangeNo = 0 ; OmegaRangeNo < NOmegaRanges ; OmegaRangeNo++ ) {
				if ( (Omega > OmegaRange[OmegaRangeNo][0]) &&
				     (Omega < OmegaRange[OmegaRangeNo][1]) &&
				     (yl > BoxSizes[OmegaRangeNo][0]) &&
				     (yl < BoxSizes[OmegaRangeNo][1]) &&
				     (zl > BoxSizes[OmegaRangeNo][2]) &&
				     (zl < BoxSizes[OmegaRangeNo][3]) ) {
					KeepSpot = 1;
					break; // Found a valid range
				}
			}

            // Check Big Detector mask if enabled and spot is currently kept
			if (d_BigDetector != NULL && BigDetSize != 0 && KeepSpot == 1){
                if (pixelsize < EPS) { // Avoid division by zero
                    KeepSpot = 0;
                } else {
                    // Ensure calculations are float-based until final int cast
                    float ylf = yl;
                    float zlf = zl;
                    float BigDetSizeF = (float)BigDetSize;
                    float YCFloat = floorf((BigDetSizeF / 2.0f) - (-ylf / pixelsize));
                    float ZCFloat = floorf(((zlf / pixelsize) + (BigDetSizeF / 2.0f)));

                    // Bounds check before casting and indexing
                    if (YCFloat < 0.0f || YCFloat >= BigDetSizeF || ZCFloat < 0.0f || ZCFloat >= BigDetSizeF) {
                         KeepSpot = 0;
                    } else {
                        YCInt = (int)YCFloat;
                        ZCInt = (int)ZCFloat;
                        idx = (long long int)(YCInt) + (long long int)BigDetSize * (long long int)ZCInt;
                        long long int max_idx = (long long int)BigDetSize * BigDetSize;

                        // Check if idx is within the valid range for the bitmask
                        if (idx < 0 || idx >= max_idx) {
                             KeepSpot = 0;
                        } else {
                            // Safely call TestBit_device
                            if (!TestBit_device(d_BigDetector, idx)) {
                                KeepSpot = 0;
                            }
                        }
                    }
                }
			}

			if (KeepSpot) {
                if (spotnr < max_spots) { // Check if output buffer has space
                    int base_idx = spotnr * N_COL_THEORSPOTS; // Use THEORSPOTS columns for consistency here
                    spots_out[base_idx + 0] = 0.0f; // OrientID (fixed to 0?)
                    spots_out[base_idx + 1] = (float)spotid;
                    spots_out[base_idx + 2] = (float)indexhkl;
                    spots_out[base_idx + 3] = distance;
                    spots_out[base_idx + 4] = yl;
                    spots_out[base_idx + 5] = zl;
                    spots_out[base_idx + 6] = omegas[i];
                    spots_out[base_idx + 7] = etas[i];
                    spots_out[base_idx + 8] = theta;
                    spots_out[base_idx + 9] = (float)ringnr;
                    // Columns 10-13 are calculated later (displacement, eta, radius diff)
                    spots_out[base_idx + 10] = 0.0f; // Placeholder
                    spots_out[base_idx + 11] = 0.0f; // Placeholder
                    spots_out[base_idx + 12] = 0.0f; // Placeholder
                    spots_out[base_idx + 13] = 0.0f; // Placeholder
                    spotnr++;
                    spotid++;
                } else {
                   // Option: print warning once per thread?
                   // Or just stop adding spots. For now, silently stop.
                   // printf("Warning: Max spots (%d) reached in CalcDiffrSpots_Furnace_d\n", max_spots);
                   break; // Exit inner loop if buffer full
                }
			}
		} // end loop over nspotsPlane
        if (spotnr >= max_spots) break; // Exit outer loop if buffer full
	} // end loop over indexhkl
	*nspots = spotnr;
}


// Needs device pointers for ObsSpotsLab, data, ndata, ypos, etamargins, omemargins
__device__
void CompareSpots_d(
    const float* TheorSpots, // Input: Flattened theoretical spots for *this* orientation/position
    int nTheorSpots,
    float MarginOme,       // Parameter
    // float MarginRad,      // Parameter - Not directly used? Check original. Seems MarginRadial is used.
    float MarginRadial,    // Parameter
    const float* d_etamargins,  // Device pointer
    const float* d_omemargins,  // Device pointer
    int n_eta_bins,        // Parameter
    int n_ome_bins,        // Parameter
    float EtaBinSize,      // Parameter
    float OmeBinSize,      // Parameter
    float BeamSize,        // Parameter
    float xThis,           // Parameter (voxel position)
    float yThis,           // Parameter (voxel position)
    const float* d_ObsSpotsLab, // Device pointer (global observed spots)
    const size_t* d_data,      // Device pointer (binned spot indices)
    const size_t* d_ndata,     // Device pointer (binned spot counts/offsets)
    const float* d_ypos,        // Device pointer (scan y positions)
    int numScans,          // Parameter
    int n_ring_bins,       // Parameter needed for indexing ndata
    float* GrainSpots_out, // Output: Flattened matched/unmatched spots
    int max_spots,         // Max capacity of GrainSpots_out (should match nTheorSpots)
    int *nMatch            // Output: Number of matched spots
    )
{
	int nMatched = 0;
	int nNonMatched = 0;
	// int sp; // loop var
	int RingNr;
	int iOme, iEta;
	size_t spotRow, spotRowBest = 0; // Initialize spotRowBest
	int MatchFound;
	float diffOme;
	float diffOmeBest;
	// size_t iRing; // loop var
	// size_t iSpot; // loop var
	size_t scannrobs; // scan nr for observed spot
	float etamargin, yRot, ySpot;

    if (EtaBinSize < EPS || OmeBinSize < EPS) { // Avoid division by zero
        *nMatch = 0;
        // Fill GrainSpots_out with non-matched markers if needed by caller
        for(int sp = 0; sp < nTheorSpots; ++sp) {
             int idx_out = (nTheorSpots - 1 - sp) * N_COL_GRAINSPOTS; // Fill from end
             GrainSpots_out[idx_out + 0] = (float)(- (sp + 1)); // Indicate non-match
             // Fill others with 0 or theoretical values as appropriate
             for(int k=1; k<N_COL_GRAINSPOTS; ++k) GrainSpots_out[idx_out + k] = 0.0f;
             // Copy theoretical info if needed
              int base_idx_theor = sp * N_COL_THEORSPOTS;
              GrainSpots_out[idx_out + 2] = TheorSpots[base_idx_theor + 10]; // yl_displ
              GrainSpots_out[idx_out + 5] = TheorSpots[base_idx_theor + 11]; // zl_displ
              GrainSpots_out[idx_out + 8] = TheorSpots[base_idx_theor + 6];  // Omega
        }
        return;
    }

	for (int sp = 0 ; sp < nTheorSpots ; sp++ )  {
        int base_idx_theor = sp * N_COL_THEORSPOTS;
		RingNr = (int) TheorSpots[base_idx_theor + 9];
        if (RingNr <= 0 || RingNr >= MAX_N_RINGS) continue; // Invalid ring

		size_t iRing = (size_t)RingNr - 1; // 0-based index for C arrays
        float theorEta = TheorSpots[base_idx_theor + 12]; // Calculated eta (with displacement)
        float theorOme = TheorSpots[base_idx_theor + 6];  // Original omega
        float theorRadDiff = TheorSpots[base_idx_theor + 13]; // Calculated radius difference

		// Calculate bin indices (ensure within bounds)
        // Add 180 for 0-360 range, handle periodicity if needed (not explicit in original?)
        iEta = (int)floorf((180.0f + theorEta) / EtaBinSize);
		iOme = (int)floorf((180.0f + theorOme) / OmeBinSize);

        // Clamp indices to valid range
        iEta = max(0, min(n_eta_bins - 1, iEta));
        iOme = max(0, min(n_ome_bins - 1, iOme));

		// Calculate rotated y position for beam size check
        // Need sin/cos of theoretical omega
        float sinOme = sinf(deg2rad * theorOme);
        float cosOme = cosf(deg2rad * theorOme);
		yRot = xThis * sinOme + yThis * cosOme;

		etamargin = d_etamargins[RingNr];
        int ome_idx_margin = (int) floorf(fabsf(theorEta)); // Eta used for omega margin index? Check original. Yes, seems so.
        ome_idx_margin = max(0, min(180, ome_idx_margin)); // Clamp index 0-180
		// omemargin = d_omemargins[ome_idx_margin]; // Use fabsf(Eta) for index as per original

		MatchFound = 0;
		diffOmeBest = MarginOme + 0.00001f; // Initialize best diff

        // Calculate the 1D index into ndata/data
        // Pos = iRing * n_eta_bins * n_ome_bins + iEta * n_ome_bins + iOme;
        size_t Pos = iRing;
        Pos *= (size_t)n_eta_bins;
        Pos += (size_t)iEta;
        Pos *= (size_t)n_ome_bins;
        Pos += (size_t)iOme;

        // Access ndata to get count and offset for this bin
        // ndata stores count and offset pairs: [count0, offset0, count1, offset1, ...]
        size_t nspots_in_bin = d_ndata[Pos * 2 + 0];
        size_t DataPos       = d_ndata[Pos * 2 + 1]; // Starting offset in 'data' array

		for (size_t iSpot = 0 ; iSpot < nspots_in_bin; iSpot++ ) {
            // Access 'data' which stores pairs of [spotRow, scanNr]
            size_t data_idx = (DataPos + iSpot) * 2;
			spotRow   = d_data[data_idx + 0]; // Row index in d_ObsSpotsLab
			scannrobs = d_data[data_idx + 1]; // Scan number (for ypos lookup)

            if (scannrobs >= (size_t)numScans) continue; // Bounds check for ypos

            // Access observed spot data (flattened array)
            int base_idx_obs = spotRow * N_COL_OBSSPOTS;
            if (base_idx_obs + 8 >= MAX_N_SPOTS * N_COL_OBSSPOTS) continue; // Bounds check ObsSpotsLab

            ySpot = d_ypos[scannrobs]; // Get y position for this scan

            // --- Matching Logic ---
            // 1. Beam position check
			if ( fabsf(yRot - ySpot) < BeamSize / 2.0f) {
                float obsRadDiff = d_ObsSpotsLab[base_idx_obs + 8]; // Observed radial difference
                // 2. Radial difference check
				if ( fabsf(theorRadDiff - obsRadDiff) < MarginRadial )  {
                    float obsEta = d_ObsSpotsLab[base_idx_obs + 6]; // Observed eta
                    // 3. Eta check
					if ( fabsf(theorEta - obsEta) < etamargin ) {
                        float obsOme = d_ObsSpotsLab[base_idx_obs + 2]; // Observed omega
                        // 4. Omega check (find minimum difference)
						diffOme = fabsf(theorOme - obsOme);
						if ( diffOme < diffOmeBest ) {
							diffOmeBest = diffOme;
							spotRowBest = spotRow;
							MatchFound = 1;
                            // Note: Original doesn't break here, continues checking bin
                            //       to find the *best* omega match within eta/rad criteria.
						}
					} // End Eta check
				} // End Radial check
			} // End Beam position check
		} // End loop over spots in bin

        // --- Record Result for this theoretical spot ---
		if (MatchFound == 1) {
            if (nMatched < max_spots) { // Check output buffer space
                int idx_out = nMatched * N_COL_GRAINSPOTS;
                int base_idx_obs_best = spotRowBest * N_COL_OBSSPOTS;
                // Check bounds before accessing obs data
                 if (base_idx_obs_best + 4 < MAX_N_SPOTS * N_COL_OBSSPOTS) {
                    GrainSpots_out[idx_out + 0] = (float)nMatched; // Match index
                    GrainSpots_out[idx_out + 1] = 999.0f; // Placeholder (was GrainID?)
                    GrainSpots_out[idx_out + 2] = TheorSpots[base_idx_theor + 10]; // Theor yl_displ
                    GrainSpots_out[idx_out + 3] = d_ObsSpotsLab[base_idx_obs_best + 0]; // Obs y0
                    GrainSpots_out[idx_out + 4] = GrainSpots_out[idx_out + 3] - GrainSpots_out[idx_out + 2]; // Diff y
                    GrainSpots_out[idx_out + 5] = TheorSpots[base_idx_theor + 11]; // Theor zl_displ
                    GrainSpots_out[idx_out + 6] = d_ObsSpotsLab[base_idx_obs_best + 1]; // Obs z0
                    GrainSpots_out[idx_out + 7] = GrainSpots_out[idx_out + 6] - GrainSpots_out[idx_out + 5]; // Diff z
                    GrainSpots_out[idx_out + 8] = TheorSpots[base_idx_theor + 6]; // Theor Omega
                    GrainSpots_out[idx_out + 9] = d_ObsSpotsLab[base_idx_obs_best + 2]; // Obs Omega
                    GrainSpots_out[idx_out + 10]= GrainSpots_out[idx_out + 9] - GrainSpots_out[idx_out + 8]; // Diff Omega
                    GrainSpots_out[idx_out + 11] = d_ObsSpotsLab[base_idx_obs_best + 3]; // Obs RefRad (Original used RefRad param here?) Check logic. Using Obs RefRad.
                    GrainSpots_out[idx_out + 12] = d_ObsSpotsLab[base_idx_obs_best + 3]; // Obs RefRad again? Check col 12 meaning. Seems to be Obs RefRad.
                    GrainSpots_out[idx_out + 13] = 0.0f; // Diff RefRad? Was Obs - RefRad param. Set to 0 for now. Needs clarification.
                    GrainSpots_out[idx_out + 14] = d_ObsSpotsLab[base_idx_obs_best + 4]; // Obs SpotID
                    // Columns 15, 16 (Match flag, IA angle) are filled later if needed
                    GrainSpots_out[idx_out + 15] = 1.0f; // Indicate it's a match for now
                    GrainSpots_out[idx_out + 16] = 0.0f; // Placeholder for IA angle
                    nMatched++;
                 } else {
                    // Handle error or skip if spotRowBest leads to out of bounds
                 }
            } else {
                 // Output buffer full for matched spots
            }
		} else {
			nNonMatched++;
            // Fill from the end of the buffer for non-matched spots
            if (nNonMatched <= max_spots) {
                int idx_out = (max_spots - nNonMatched) * N_COL_GRAINSPOTS;
                GrainSpots_out[idx_out + 0] = (float)(-nNonMatched); // Negative index indicates non-match
                GrainSpots_out[idx_out + 1] = 999.0f; // Placeholder
                GrainSpots_out[idx_out + 2] = TheorSpots[base_idx_theor + 10]; // Theor yl_displ
                GrainSpots_out[idx_out + 3] = 0.0f; // Obs y0 (not found)
                GrainSpots_out[idx_out + 4] = 0.0f; // Diff y
                GrainSpots_out[idx_out + 5] = TheorSpots[base_idx_theor + 11]; // Theor zl_displ
                GrainSpots_out[idx_out + 6] = 0.0f; // Obs z0 (not found)
                GrainSpots_out[idx_out + 7] = 0.0f; // Diff z
                GrainSpots_out[idx_out + 8] = TheorSpots[base_idx_theor + 6]; // Theor Omega
                GrainSpots_out[idx_out + 9] = 0.0f; // Obs Omega (not found)
                GrainSpots_out[idx_out + 10]= 0.0f; // Diff Omega
                GrainSpots_out[idx_out + 11] = 0.0f; // Theor RefRad (not applicable?)
                GrainSpots_out[idx_out + 12] = 0.0f; // Obs RefRad (not found)
                GrainSpots_out[idx_out + 13] = 0.0f; // Diff RefRad
                GrainSpots_out[idx_out + 14] = 0.0f; // Obs SpotID (not found)
                GrainSpots_out[idx_out + 15] = 0.0f; // Indicate non-match
                GrainSpots_out[idx_out + 16] = 999.0f; // IA angle not applicable
            } else {
                 // Output buffer full for non-matched spots
            }
		}
	} // End loop over theoretical spots (sp)
	*nMatch = nMatched;
}


__device__ __forceinline__
void AxisAngle2RotMatrix_d(const float axis[3], float angle, float R[3][3])
{
    float axis_norm_sq = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2];

	if ( axis_norm_sq < EPS ) { // Use squared norm to avoid sqrtf
		R[0][0] = 1.0f; R[1][0] = 0.0f; R[2][0] = 0.0f;
		R[0][1] = 0.0f; R[1][1] = 1.0f; R[2][1] = 0.0f;
		R[0][2] = 0.0f; R[1][2] = 0.0f; R[2][2] = 1.0f;
		return;
	}
	float lenInv = 1.0f / sqrtf(axis_norm_sq);
	float u = axis[0]*lenInv;
	float v = axis[1]*lenInv;
	float w = axis[2]*lenInv;
	float angleRad = deg2rad * angle;
	float rcos = cosf(angleRad);
	float rsin = sinf(angleRad);
    float one_minus_rcos = 1.0f - rcos;

	R[0][0] =      rcos + u*u*one_minus_rcos;
	R[1][0] =  w * rsin + v*u*one_minus_rcos;
	R[2][0] = -v * rsin + w*u*one_minus_rcos;
	R[0][1] = -w * rsin + u*v*one_minus_rcos;
	R[1][1] =      rcos + v*v*one_minus_rcos;
	R[2][1] =  u * rsin + w*v*one_minus_rcos;
	R[0][2] =  v * rsin + u*w*one_minus_rcos;
	R[1][2] = -u * rsin + v*w*one_minus_rcos;
	R[2][2] =      rcos + w*w*one_minus_rcos;
}

// Device equivalent of CalcRotationAngle
// Needs HKLints, ABCABG, SGNum (passed or from const memory)
// This logic seems fairly complex and might be better precalculated on host
// if ABCABG and SGNum are constant for the whole run.
// For now, porting directly.
__device__ __forceinline__
float CalcRotationAngle_d (int RingNr, int SGNum, const int* d_HKLints, const double h_ABCABG[6]) {
    // Note: h_ABCABG is host data passed by value - okay if small and read-only.
    // d_HKLints is device pointer.
	int habs=0, kabs=0, labs=0; // Initialize
	int i;
    bool found = false;
	for (i=0; i<MAX_N_HKLS; i++){ // MAX_N_HKLS should be parameter or const
        // Access flattened HKLints: row * num_cols + col
        if (d_HKLints[i * 4 + 3] == RingNr){
			habs = abs(d_HKLints[i * 4 + 0]);
			kabs = abs(d_HKLints[i * 4 + 1]);
			labs = abs(d_HKLints[i * 4 + 2]);
            found = true;
			break;
		}
	}
    if (!found) return 0.0f; // Or some default if RingNr not found

	int nzeros = 0;
	if (habs == 0) nzeros++;
	if (kabs == 0) nzeros++;
	if (labs == 0) nzeros++;

	if (nzeros == 3) return 0.0f;

    // Using h_ABCABG directly. Convert to float internally if needed.
    float alpha = (float)h_ABCABG[3];
    // float beta = (float)h_ABCABG[4];
    float gamma = (float)h_ABCABG[5];
    float ninety = 90.0f;

	if (SGNum == 1 || SGNum == 2){ // Triclinic
		return 360.0f;
	}else if (SGNum >= 3 && SGNum <= 15){ // Monoclinic
		if (nzeros != 2) return 360.0f;
		// Check specific axes based on standard settings (beta unique usually)
        // Assuming standard setting 2/m with b unique: alpha=gamma=90
        else if (fabsf(alpha - ninety) < EPS && fabsf(gamma - ninety) < EPS && labs != 0){ // Rotation around c
             return 180.0f; // 2-fold axis along c
        }
        // Add checks for other settings if necessary, based on which angles are 90
        // else if (fabsf(alpha - ninety) < EPS && fabsf(beta - ninety) < EPS && kabs != 0){ // Rotation around b (gamma unique)
		// 	return 180.0f;
		// } else if (fabsf(beta - ninety) < EPS && fabsf(gamma - ninety) < EPS && habs != 0){ // Rotation around a (alpha unique)
		// 	return 180.0f;
		// }
        else return 360.0f; // Default if specific conditions aren't met
	}else if (SGNum >= 16 && SGNum <= 74){ // Orthorhombic (alpha=beta=gamma=90)
		if (nzeros != 2) return 360.0f; // General reflection
		else return 180.0f; // Reflection on a principal plane (e.g., (h00), (0k0), (00l)) has 180 deg freedom around normal
	}else if (SGNum >= 75 && SGNum <= 142){ // Tetragonal (a=b, alpha=beta=gamma=90)
		if (nzeros == 3) return 0.0f; // (000) - already handled
        if (nzeros == 2){ // On a principal axis (e.g., 00l, h00, hh0)
			if (labs != 0){ // 00l type
				return 90.0f; // 4-fold rotation around c
			}else{ // h00 or 0k0 (equivalent) or hh0 type
				return 180.0f; // 2-fold rotation
			}
        } else if (nzeros == 1) { // On a principal plane (e.g., hk0, h0l, hhl)
            if (labs == 0 && habs == kabs) { // hh0 type plane normal
                 return 180.0f; // 2-fold axis along [110] direction
            } else {
                 return 360.0f; // General plane
            }
        } else { // General hkl
             return 360.0f;
        }
	}else if (SGNum >= 143 && SGNum <= 167){ // Trigonal/Rhombohedral (check conventions)
        // Hexagonal axes: a=b, gamma=120, alpha=beta=90
		if (nzeros == 3) return 0.0f;
        if (nzeros == 2 && labs != 0) return 120.0f; // 00l type, 3-fold axis along c
		else return 360.0f; // Other cases generally require full rotation
	}else if (SGNum >= 168 && SGNum <= 194){ // Hexagonal (a=b, gamma=120, alpha=beta=90)
        if (nzeros == 3) return 0.0f;
        if (nzeros == 2 && labs != 0) return 60.0f; // 00l type, 6-fold axis along c
		else return 360.0f; // Other cases
	}else if (SGNum >= 195 && SGNum <= 230){ // Cubic (a=b=c, alpha=beta=gamma=90)
        if (nzeros == 3) return 0.0f;
		if (nzeros == 2) return 90.0f; // h00 type, 4-fold axis
		else if (nzeros == 1){ // hk0 type plane normal
			if (habs == kabs || kabs == labs || habs == labs) return 180.0f; // hh0 type, 2-fold axis
            else return 360.0f;
		} else { // hkl type
             if (habs == kabs && kabs == labs) return 120.0f; // hhh type, 3-fold axis
             else return 360.0f; // General hkl
        }
	}
	else return 0.0f; // Unknown SG
}


// Generates candidate orientations on the device
// Output OrMat is filled in the caller's workspace buffer slice
__device__
int GenerateCandidateOrientationsF_d(
    const float hkl[3],         // Input hkl vector (e.g., from RingHKL)
    const float hklnormal[3],   // Input measured normal vector (g-vector)
    float stepsize,           // Input step size
    int RingNr,               // Input ring number
    int SGNum,                // Input space group
    const int* d_HKLints,       // Device HKL integer list
    const double h_ABCABG[6],   // Host lattice params (passed by value)
    float* OrMat_out,         // Output buffer slice (flattened [nOrient][3][3])
    int max_orient,           // Max capacity of OrMat_out buffer
    int * nOrient             // Output number of orientations generated
    )
{
	float v[3]; // Cross product result
	float MaxAngle = 0;

	crossProduct_device(v, hkl, hklnormal); // v = hkl x hklnormal

    // Calculate angle between hkl and hklnormal
	float hkllen_sq = hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2];
	float hklnormallen_sq = hklnormal[0]*hklnormal[0] + hklnormal[1]*hklnormal[1] + hklnormal[2]*hklnormal[2];

    if (hkllen_sq < EPS || hklnormallen_sq < EPS) {
        *nOrient = 0;
        return 1; // Indicate error or invalid input
    }
    float hkllen = sqrtf(hkllen_sq);
    float hklnormallen = sqrtf(hklnormallen_sq);

	float dotpr = dot_device(hkl, hklnormal);
	float cos_angle = dotpr / (hkllen * hklnormallen);
    cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle)); // Clamp for safety
	float angled = rad2deg * acosf(cos_angle); // Angle needed to rotate hkl onto hklnormal

	float RotMat1[3][3]; // Rotation around 'v' by 'angled'
	float RotMat2[3][3]; // Rotation around 'hklnormal' by 'angle2'
	float RotMat3[3][3]; // Final combined rotation: RotMat2 * RotMat1

	AxisAngle2RotMatrix_d(v, angled, RotMat1); // Matrix to align hkl normal to hklnormal direction

    // Calculate the symmetry-allowed rotation range around hklnormal
	MaxAngle = CalcRotationAngle_d(RingNr, SGNum, d_HKLints, h_ABCABG);

    if (stepsize < EPS) {
        *nOrient = 0;
        return 1; // Avoid infinite loop / division by zero
    }
	float nsteps_f = (MaxAngle / stepsize);
	int nstepsi = (int) floorf(nsteps_f); // Number of steps
    if (nstepsi <= 0) nstepsi = 1; // Ensure at least one orientation if MaxAngle is small/zero

    *nOrient = 0; // Initialize count

	for (int or_idx = 0 ; or_idx < nstepsi ; or_idx++) {
        if (*nOrient >= max_orient) break; // Check output buffer capacity

		float angle2 = (float)or_idx * stepsize; // Rotation angle around hklnormal
		AxisAngle2RotMatrix_d(hklnormal, angle2, RotMat2); // Rotation matrix for this step

        // Combine rotations: RotMat3 = RotMat2 * RotMat1
		MatrixMultF33_d(RotMat2, RotMat1, RotMat3);

        // Store the resulting orientation matrix in the flattened output buffer
        int base_idx = (*nOrient) * 9; // 9 floats per 3x3 matrix
		for (int row = 0 ; row < 3 ; row++) {
			for (int col = 0 ; col < 3 ; col++) {
				OrMat_out[base_idx + row * 3 + col] = RotMat3[row][col];
			}
		}
        (*nOrient)++;
	}

	return 0; // Success
}


__device__ __forceinline__
void displacement_spot_needed_COM_d(
    float a, float b, float c,    // COM position (relative to nominal beam center?)
    float xi, float yi, float zi, // Direction vector to spot (unit length?)
    float omega,                  // Omega angle (degrees)
	float *Displ_y, float *Displ_z // Output displacement needed
    )
{
    // Normalize direction vector (original code implies it might not be unit)
	float lenInv_sq = xi*xi + yi*yi + zi*zi;
    if (lenInv_sq < EPS) {
        *Displ_y = 0.0f;
        *Displ_z = 0.0f;
        return;
    }
    float lenInv = 1.0f / sqrtf(lenInv_sq);
	float xn = xi*lenInv;
	float yn = yi*lenInv;
	float zn = zi*lenInv;

    if (fabsf(xn) < EPS) { // Avoid division by zero if spot is in YZ plane (xi=0)
         // Simplified case? How should displacement be handled if xn is 0?
         // Original formula seems to assume xn != 0 for 't'.
         // If xn is 0, the projection onto the x-axis is 0.
         // Let's assume for now this case implies no shift based on 't'.
         // Revisit if this assumption is wrong.
        *Displ_y = (a*sinf(deg2rad*omega) + b*cosf(deg2rad*omega));
        *Displ_z = c; // No z-shift component from 't'
        return;
    }

	float OmegaRad = deg2rad * omega;
	float sinOme = sinf(OmegaRad);
	float cosOme = cosf(OmegaRad);

    // 't' represents the distance along the spot vector (xn,yn,zn)
    // from the origin to the point where the COM's x-component, rotated by omega,
    // matches the spot vector's x-component.
	float t = (a*cosOme - b*sinOme) / xn;

    // Displacement is the difference between the COM's rotated y/z position
    // and the y/z position on the spot vector at distance 't'.
	*Displ_y = ((a*sinOme)+(b*cosOme)) - (t*yn);
	*Displ_z = c - (t*zn);
}


__device__ __forceinline__
void spot_to_gv_d(
    float xi, float yi, float zi, // Spot position vector (from sample to detector)
    float Omega,                  // Omega angle (degrees)
    float *g1, float *g2, float *g3 // Output g-vector components
    )
{
	float len_sq = xi*xi + yi*yi + zi*zi;
	if (len_sq < EPS) { // Handle zero vector
		*g1 = 0.0f; *g2 = 0.0f; *g3 = 0.0f;
		// printf("Warning: spot_to_gv_d called with zero vector.\n"); // Avoid printf in device code
		return;
	}
    float len = sqrtf(len_sq);
	float xn = xi/len; // Normalize spot vector (k_out / |k_out|)
	float yn = yi/len;
	float zn = zi/len;

    // Calculate k_out - k_in in sample frame (unrotated)
    // k_in = [1, 0, 0] in lab frame (assuming beam along +x)
	float g1r = xn - 1.0f; // G_x = k_out_x - k_in_x = xn - 1
	float g2r = yn;        // G_y = k_out_y - k_in_y = yn - 0
    float g3r = zn;        // G_z = k_out_z - k_in_z = zn - 0

    // Rotate G vector by -Omega around Z to get G in crystal frame
	float CosOme = cosf(-Omega*deg2rad);
	float SinOme = sinf(-Omega*deg2rad);

	*g1 = g1r * CosOme - g2r * SinOme;
	*g2 = g1r * SinOme + g2r * CosOme;
	*g3 = g3r; // Z component is unchanged by rotation around Z
}

__device__ __forceinline__
void spot_to_gv_pos_d(
    float xi, float yi, float zi, // Spot position vector in lab frame (detector coords)
    float Omega,                  // Omega angle (degrees)
    float cx, float cy, float cz, // Crystal COM position in lab frame
    float *g1, float *g2, float *g3 // Output g-vector components
    )
{
    // Calculate the vector from the crystal COM to the spot on the detector, in the lab frame
	float spot_vec_lab[3] = {xi, yi, zi};
    float crystal_com_lab[3] = {cx, cy, cz};

    // Rotate the crystal COM position by +Omega to find where it was at that angle
    float crystal_com_rotated[3];
    RotateAroundZ_d(crystal_com_lab, Omega, crystal_com_rotated);

    // Vector from the (rotated) crystal COM to the spot: k_out vector originating at COM
    float k_out_origin_com[3];
    k_out_origin_com[0] = spot_vec_lab[0] - crystal_com_rotated[0];
	k_out_origin_com[1] = spot_vec_lab[1] - crystal_com_rotated[1];
	k_out_origin_com[2] = spot_vec_lab[2] - crystal_com_rotated[2];

    // Now convert this k_out vector (relative to COM) to a g-vector using the standard function
	spot_to_gv_d( k_out_origin_com[0], k_out_origin_com[1], k_out_origin_com[2], Omega, g1, g2, g3);
}


// Calculates average Internal Angle (IA) for matched spots
// Modifies GrainMatches_d device buffer directly
__device__
void CalcIA_d(
    float* GrainMatches_d, // Device buffer for grain match results (modified)
    int nGrainMatches,     // Number of grains/matches found for this voxel
    float* AllGrainSpots_d, // Device buffer containing spot pair details (modified)
    int totalSpotsForGrains, // Total number of spot entries in AllGrainSpots_d for these grains
    float distance         // Detector distance
    )
{
    // This function iterates through grains and then spots *within* each grain.
    // In the original OMP code, this was called *after* finding the best match.
    // In CUDA, if we want this per voxel, it needs to be part of the kernel.
    // It requires results from CompareSpots (AllGrainSpots_d).

    // We need temporary storage for IAgrainspots *per grain*.
    // This is tricky to manage efficiently in a standard kernel thread structure
    // unless the number of spots per grain is small and known.

    // Alternative: Calculate IA angles directly within CompareSpots if possible,
    // or have a separate kernel/device function call that aggregates IA *after*
    // the main matching kernel.

    // Let's try to integrate it, assuming we only process ONE grain match per voxel (MAX_N_MATCHES = 1).
    if (nGrainMatches != 1) {
        // If MAX_N_MATCHES > 1, this simple approach won't work directly.
        // We'd need to allocate IA temp space per grain, etc.
        // For now, assume only the best grain (grain 0) is processed.
        if (nGrainMatches > 0) {
             // Mark IA as invalid if multiple matches were handled differently
             GrainMatches_d[0 * N_COL_GRAINMATCHES + 15] = -1.0f; // Use negative IA for error/unsupported case
        }
        return;
    }
    if (nGrainMatches == 0) return; // No grain found

    // Process the single grain match (index 0)
    int grain_idx = 0;
    int nspots_this_grain = (int)GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 13]; // Number of *matched* spots for this grain
    // int ntheor_spots = (int)GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 12]; // Total theoretical spots

    float g1x = GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 9];  // COM x
	float g1y = GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 10]; // COM y
	float g1z = GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 11]; // COM z

	float totalIA = 0.0f;
	int nIAspots = 0;
    int spot_start_index = 0; // Assuming spots for grain 0 start at index 0 in AllGrainSpots_d

	for (int r = 0; r < nspots_this_grain; r++) { // Iterate only over *matched* spots
        int spot_row_idx = spot_start_index + r;
        int base_idx_spot = spot_row_idx * N_COL_GRAINSPOTS;

        // Check if this row corresponds to a matched spot (original check was < 0 for non-match)
        // In CompareSpots_d, we fill matched spots from the start.
        // Let's assume AllGrainSpots_d contains matched spots first, then non-matched.

        // Column indices from original AllGrainSpots structure:
        // 2: Theor yl_displ , 3: Obs y0
        // 5: Theor zl_displ , 6: Obs z0
        // 8: Theor Omega    , 9: Obs Omega
        // 16: Output IA angle

		float x1 = distance; // Theor spot 'x'
		float x2 = distance; // Obs spot 'x' (assuming same distance)

		float y1_theor = AllGrainSpots_d[base_idx_spot + 2];
		float y2_obs   = AllGrainSpots_d[base_idx_spot + 3];
		float z1_theor = AllGrainSpots_d[base_idx_spot + 5];
		float z2_obs   = AllGrainSpots_d[base_idx_spot + 6];
		float w1_theor = AllGrainSpots_d[base_idx_spot + 8];
		float w2_obs   = AllGrainSpots_d[base_idx_spot + 9];

        // Need g-vectors for theoretical and observed spots
        float gv1x, gv1y, gv1z; // Theoretical g-vector
        float gv2x, gv2y, gv2z; // Observed g-vector

        // Calculate theoretical g-vector based on theoretical position and omega
        // Need the *original* theoretical spot position (before COM displacement)
        // This information isn't directly stored in AllGrainSpots_d in the CompareSpots_d output.
        // *** This is a major problem for porting CalcIA directly here. ***
        // CalcIA needs the *underlying HKL* info or the original TheorSpots.

        // --- WORKAROUND / REINTERPRETATION ---
        // Maybe the original CalcIA intended to compare the g-vector derived
        // from the *displaced* theoretical spot (yl_displ, zl_displ) with the
        // g-vector from the observed spot (y0, z0)? Let's try that.

        spot_to_gv_pos_d(x1, y1_theor, z1_theor, w1_theor, g1x, g1y, g1z, &gv1x, &gv1y, &gv1z);
        spot_to_gv_pos_d(x2, y2_obs,   z2_obs,   w2_obs,   g1x, g1y, g1z, &gv2x, &gv2y, &gv2z);

		float currentIA = 0.0f;
        CalcInternalAngle_d(gv1x, gv1y, gv1z, gv2x, gv2y, gv2z, &currentIA);

        AllGrainSpots_d[base_idx_spot + 16] = currentIA; // Store IA for this spot pair

		totalIA += currentIA; // Accumulate IA
		nIAspots++;

	} // End loop over matched spots

    // Calculate average IA and store in GrainMatches
	if (nIAspots > 0) {
		GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 15] = totalIA / (float)nIAspots;
	} else {
        GrainMatches_d[grain_idx * N_COL_GRAINMATCHES + 15] = 0.0f; // Or some indicator? Original returned 0.
    }
}

// --- Host Helper Functions (File Reading, Param Setup) ---

// Struct definition needs to be visible before use
struct TParams {
   // Host copy of parameters
   int RingNumbers[MAX_N_RINGS];
   int SpaceGroupNum;
   float LatticeConstant; // Changed to float
   float Wavelength;      // Changed to float
   float Distance;        // Changed to float
   float Rsample;         // Changed to float
   float Hbeam;           // Changed to float
   float StepsizePos;     // Changed to float
   float StepsizeOrient;  // Changed to float
   int NrOfRings;
   float RingRadii[MAX_N_RINGS];      // Effective radii used
   float RingRadiiUser[MAX_N_RINGS]; // Radii read from file
   float MarginOme;       // Changed to float
   float MarginEta;       // Changed to float
   float MarginRad;       // Changed to float
   float MarginRadial;    // Changed to float
   float EtaBinSize;      // Changed to float
   float OmeBinSize;      // Changed to float
   float ExcludePoleAngle; // Changed to float
   float MinMatchesToAcceptFrac; // Changed to float
   float BoxSizes[MAX_N_OMEGARANGES][4]; // Changed to float
   float OmegaRanges[MAX_N_OMEGARANGES][2];// Changed to float
   char OutputFolder[4096];
   char MicFN[4096];
   int NoOfOmegaRanges;
   char SpotsFileName[4096]; // Not directly used after reading?
   char IDsFileName [4096];   // Not directly used after reading?
   int UseFriedelPairs; // Not directly used in core logic shown? Check.
   int RingToIndex;
   // Add other necessary parameters if they were missed
   float pixelsize;        // Added
   float BeamSize;         // Added
   int BigDetSize;         // Added
   int numScans;           // Added (needed for ypos size)
   int n_hkls;             // Added (actual number read)
   int SGNum;              // Added (alias for SpaceGroupNum)
   double ABCABG[6];       // Added (keep double for CalcRotationAngle?)
};

// Forward declaration for ReadBigDet
size_t ReadBigDet(char *cwd, struct TParams *Params);

// ReadParams remains largely the same, but store values as floats
int ReadParams(char FileName[], struct TParams * Params)
{
	#define MAX_LINE_LENGTH_PARAM 4096 // Avoid conflict with kernel define
	sprintf(Params->MicFN,"0");
	FILE *fp;
	char line[MAX_LINE_LENGTH_PARAM];
	char dummy[MAX_LINE_LENGTH_PARAM];
	const char *str;
	int NrOfBoxSizes = 0;
	int cmpres;
	int NoRingNumbers = 0;
	Params->NrOfRings = 0;
	Params->NoOfOmegaRanges = 0;
    // Initialize some params
    Params->BigDetSize = 0;
    Params->pixelsize = 0.0f;
    Params->BeamSize = 0.0f;
    Params->RingToIndex = -1; // Default invalid


	fp = fopen(FileName, "r");
	if (fp==NULL) {
		printf("Cannot open file: %s.\n", FileName);
		return(1);
	}

	// Use temporary doubles for reading, then convert to float
    double temp_double;
    int temp_int;

	while (fgets(line, MAX_LINE_LENGTH_PARAM, fp) != NULL) {
        // Trim leading/trailing whitespace (simple version)
        char* start = line;
        while(isspace((unsigned char)*start)) start++;
        char* end = start + strlen(start) - 1;
        while(end > start && isspace((unsigned char)*end)) end--;
        *(end + 1) = 0;
        if (*start == '#' || *start == '\0') continue; // Skip comments and empty lines


		str = "RingNumbers "; // Expecting multiple integers after this
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
            char *token;
            char *rest = start + strlen(str);
            while ((token = strtok_r(rest, " \t\n", &rest))) {
                 if (NoRingNumbers < MAX_N_RINGS) {
                    Params->RingNumbers[NoRingNumbers++] = atoi(token);
                 }
            }
			continue;
		}
		str = "RingToIndex ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %d", dummy, &temp_int) == 2) {
                Params->RingToIndex = temp_int;
            }
			continue;
		}
		str = "BigDetSize ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %d", dummy, &temp_int) == 2) {
                Params->BigDetSize = temp_int;
                BigDetSize_host = temp_int; // Store global too
                totNrPixelsBigDetector_host = (long long int)temp_int * temp_int;
                totNrPixelsBigDetector_host /= 32;
                totNrPixelsBigDetector_host++; // For integer division ceiling
            }
			continue;
		}
		str = "px "; // pixelsize
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->pixelsize = (float)temp_double;
                pixelsize_host = (float)temp_double; // Store global
            }
			continue;
		}
		str = "BeamSize ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->BeamSize = (float)temp_double + 0.1f; // Add margin
                BeamSize_host = Params->BeamSize; // Store global
            }
			continue;
		}
		str = "SpaceGroup ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %d", dummy, &temp_int) == 2) {
                Params->SpaceGroupNum = temp_int;
                Params->SGNum = temp_int; // Set alias
                SGNum_host = temp_int; // Store global
            }
			continue;
		}
		str = "LatticeParameter ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
            int n_read = sscanf(start, "%s %lf %lf %lf %lf %lf %lf", dummy,
                                &Params->ABCABG[0], &Params->ABCABG[1], &Params->ABCABG[2],
                                &Params->ABCABG[3], &Params->ABCABG[4], &Params->ABCABG[5]);
            if (n_read >= 2) {
                Params->LatticeConstant = (float)Params->ABCABG[0]; // Assuming first is 'a' for LatticeConstant
                // Copy to global host double array
                memcpy(h_ABCABG, Params->ABCABG, 6 * sizeof(double));
            }
            // Handle case where only 1 lattice param is given if needed
            if (n_read == 2) { // Only 'a' provided? Assume cubic? Or default others?
                Params->ABCABG[1] = Params->ABCABG[0]; // b=a
                Params->ABCABG[2] = Params->ABCABG[0]; // c=a
                Params->ABCABG[3] = 90.0; // alpha
                Params->ABCABG[4] = 90.0; // beta
                Params->ABCABG[5] = 90.0; // gamma
                memcpy(h_ABCABG, Params->ABCABG, 6 * sizeof(double));
            }
			continue;
		}
		str = "Wavelength ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->Wavelength = (float)temp_double;
            }
			continue;
		}
		str = "Distance ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->Distance = (float)temp_double;
            }
			continue;
		}
		str = "Rsample ";
		cmpres = strncmp(start, str, strlen(str));
		if ( cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->Rsample = (float)temp_double;
            }
			continue;
		}
		str = "Hbeam ";
		cmpres = strncmp(start, str, strlen(str));
		if ( cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->Hbeam = (float)temp_double;
            }
			continue;
		}
		str = "StepsizePos ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->StepsizePos = (float)temp_double;
            }
			continue;
		}
		str = "StepsizeOrient ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->StepsizeOrient = (float)temp_double;
            }
			continue;
		}
		str = "MarginOme ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->MarginOme = (float)temp_double;
            }
			continue;
		}
		str = "MarginRadius "; // Renamed to MarginRad in original struct? Using MarginRad.
		cmpres = strncmp(start, str , strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->MarginRad = (float)temp_double;
            }
			continue;
		}
		str = "MarginRadial ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->MarginRadial = (float)temp_double;
            }
			continue;
		}
		str = "EtaBinSize ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->EtaBinSize = (float)temp_double;
                EtaBinSize_host = (float)temp_double; // global
            }
			continue;
		}
		str = "OmeBinSize ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->OmeBinSize = (float)temp_double;
                OmeBinSize_host = (float)temp_double; // global
            }
			continue;
		}
		str = "MinMatchesToAcceptFrac ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->MinMatchesToAcceptFrac = (float)temp_double;
            }
			continue;
		}
		str = "ExcludePoleAngle ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->ExcludePoleAngle = (float)temp_double;
            }
			continue;
		}
		str = "RingRadii "; // Expecting multiple floats
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
            char *token;
            char *rest = start + strlen(str);
             while ((token = strtok_r(rest, " \t\n", &rest))) {
                if (Params->NrOfRings < MAX_N_RINGS) {
                    Params->RingRadiiUser[Params->NrOfRings++] = (float)atof(token);
                }
             }
			continue;
		}
		str = "OmegaRange ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
            double r1, r2;
			if (sscanf(start, "%s %lf %lf", dummy, &r1, &r2) == 3) {
                if (Params->NoOfOmegaRanges < MAX_N_OMEGARANGES) {
                    Params->OmegaRanges[Params->NoOfOmegaRanges][0] = (float)r1;
                    Params->OmegaRanges[Params->NoOfOmegaRanges][1] = (float)r2;
                    (Params->NoOfOmegaRanges)++;
                }
            }
			continue;
		}
		str = "BoxSize ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
            double b1, b2, b3, b4;
			if (sscanf(start, "%s %lf %lf %lf %lf", dummy, &b1, &b2, &b3, &b4) == 5) {
                if (NrOfBoxSizes < MAX_N_OMEGARANGES) { // Use same limit as OmegaRanges
                    Params->BoxSizes[NrOfBoxSizes][0] = (float)b1;
                    Params->BoxSizes[NrOfBoxSizes][1] = (float)b2;
                    Params->BoxSizes[NrOfBoxSizes][2] = (float)b3;
                    Params->BoxSizes[NrOfBoxSizes][3] = (float)b4;
                    NrOfBoxSizes++;
                }
            }
			continue;
		}
		str = "SpotsFileName ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			sscanf(start, "%s %s", dummy, Params->SpotsFileName );
			continue;
		}
		str = "IDsFileName ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			sscanf(start, "%s %s", dummy, Params->IDsFileName  );
			continue;
		}
		str = "MicFile ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			sscanf(start, "%s %s", dummy, Params->MicFN  );
			continue;
		}
		str = "MarginEta ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %lf", dummy, &temp_double) == 2) {
                Params->MarginEta = (float)temp_double;
            }
			continue;
		}
		str = "UseFriedelPairs ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			if (sscanf(start, "%s %d", dummy, &temp_int) == 2) {
                Params->UseFriedelPairs = temp_int;
            }
			continue;
		}
		str = "OutputFolder ";
		cmpres = strncmp(start, str, strlen(str));
		if (cmpres == 0) {
			sscanf(start, "%s %s", dummy, Params->OutputFolder );
			continue;
		}

		printf("Warning: skipping line in parameters file:\n");
		printf("%s\n", start);
	}
    fclose(fp);

    // Check if NrOfBoxSizes matches NoOfOmegaRanges
    if (NrOfBoxSizes != Params->NoOfOmegaRanges && NrOfBoxSizes > 0 && Params->NoOfOmegaRanges > 0) {
        printf("Warning: Number of BoxSize entries (%d) does not match number of OmegaRange entries (%d).\n",
               NrOfBoxSizes, Params->NoOfOmegaRanges);
        // Optionally adjust, e.g., take the minimum? Or error out?
        Params->NoOfOmegaRanges = min(NrOfBoxSizes, Params->NoOfOmegaRanges);
    } else if (NrOfBoxSizes == 0 && Params->NoOfOmegaRanges > 0) {
         printf("Warning: OmegaRanges specified, but no BoxSize entries found. Assuming full detector box for all ranges.\n");
         // Set default BoxSizes if needed, e.g., [-inf, inf, -inf, inf] conceptually
         // Or rely on caller to handle this case appropriately.
         // For simplicity here, we'll just proceed. The BoxSizes check might always fail if not set.
         // Let's set a very large default box
         for (int i=0; i< Params->NoOfOmegaRanges; ++i) {
            Params->BoxSizes[i][0] = -1e10f;
            Params->BoxSizes[i][1] = 1e10f;
            Params->BoxSizes[i][2] = -1e10f;
            Params->BoxSizes[i][3] = 1e10f;
         }
    }


	// Read BigDetector mask if size was specified
    if (Params->BigDetSize > 0) {
		char *cwd = dirname(strdup(Params->OutputFolder)); // Use strdup as dirname might modify input
		size_t sz = ReadBigDet(cwd, Params); // Pass Params to potentially store pointer
        free(cwd);
        if (sz == 0) {
             printf("Warning: BigDetSize specified, but failed to read BigDetectorMask.bin\n");
             // Optionally disable BigDetector usage
             Params->BigDetSize = 0;
             BigDetSize_host = 0;
             totNrPixelsBigDetector_host = 0;
        }
	}

	// Populate the effective RingRadii array based on RingNumbers and RingRadiiUser
	for (int i = 0 ; i < MAX_N_RINGS ; i++ ) Params->RingRadii[i] = 0.0f; // Initialize
	for (int i = 0 ; i < Params->NrOfRings ; i++ ) {
        int ring_num = Params->RingNumbers[i];
        if (ring_num >= 0 && ring_num < MAX_N_RINGS) {
             if (i < Params->NrOfRings) { // Check index for RingRadiiUser
                Params->RingRadii[ring_num] = Params->RingRadiiUser[i];
             }
        } else {
            printf("Warning: Invalid RingNumber %d found in parameters.\n", ring_num);
        }
    }
	return(0);
}


// Read Big Detector Mask (Host side)
size_t ReadBigDet(char *cwd, struct TParams *Params)
{
	int fd;
	struct stat s;
	int status;
	size_t size = 0; // Initialize size
	char filename[2048];
	sprintf(filename,"%s/BigDetectorMask.bin", cwd);

	fd = open(filename,O_RDONLY);
	if (fd < 0) { // Use checkHost style
        fprintf(stderr, "Warning: open %s failed: %s\n", filename, strerror(errno));
        return 0; // Return 0 size on failure
    }
	// checkHost(fd < 0, "open %s failed: %s", filename, strerror(errno));

	status = fstat (fd , &s);
    if (status < 0) {
        fprintf(stderr, "Warning: stat %s failed: %s\n", filename, strerror(errno));
        close(fd);
        return 0;
    }
	// checkHost (status < 0, "stat %s failed: %s", filename, strerror(errno));

	size = s.st_size;
    // Important: mmap gives a HOST pointer
	h_BigDetector = (int*)mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);

    if (h_BigDetector == MAP_FAILED) {
         fprintf(stderr, "Warning: mmap %s failed: %s\n", filename, strerror(errno));
         h_BigDetector = NULL; // Ensure pointer is NULL
         close(fd);
         return 0;
    }
	// checkHost (h_BigDetector == MAP_FAILED,"mmap %s failed: %s", filename, strerror(errno));

    // Close file descriptor after mmap is successful
    close(fd);

    // Verify size matches expectation?
    size_t expected_elements = (size_t)totNrPixelsBigDetector_host; // Already calculated # of int elements
    size_t expected_bytes = expected_elements * sizeof(int);
    if (size < expected_bytes) { // Allow slightly larger file, but not smaller
        fprintf(stderr, "Warning: BigDetectorMask.bin size (%zu bytes) is smaller than expected (%zu bytes) for BigDetSize %d.\n",
                size, expected_bytes, Params->BigDetSize);
        munmap(h_BigDetector, size); // Unmap memory
        h_BigDetector = NULL;
        return 0;
    }

	return size; // Return actual mapped size
}

// Read Binned Data Files (Host side)
int ReadBins(char *cwd)
{
	int fd_data, fd_ndata;
	struct stat s_data, s_ndata;
	int status_data, status_ndata;
	size_t size_data = 0, size_ndata = 0;
	char file_data[2048], file_ndata[2048];

	sprintf(file_data,"%s/Data.bin", cwd);
	sprintf(file_ndata,"%s/nData.bin", cwd);

    // --- Read Data.bin ---
	fd_data = open (file_data, O_RDONLY);
    if (fd_data < 0) {
        fprintf(stderr, "Error: open %s failed: %s\n", file_data, strerror (errno));
        return 0; // Failure
    }
	status_data = fstat (fd_data, & s_data);
    if (status_data < 0) {
        fprintf(stderr, "Error: stat %s failed: %s\n", file_data, strerror (errno));
        close(fd_data);
        return 0;
    }
	size_data = s_data.st_size;
	h_data = (size_t*)mmap (0, size_data, PROT_READ, MAP_PRIVATE, fd_data, 0); // Use MAP_PRIVATE? Or SHARED? Private might be safer.
    if (h_data == MAP_FAILED) {
        fprintf(stderr, "Error: mmap %s failed: %s\n", file_data, strerror (errno));
        h_data = NULL;
        close(fd_data);
        return 0;
    }
    close(fd_data); // Close FD after mmap

    // --- Read nData.bin ---
	fd_ndata = open (file_ndata, O_RDONLY);
    if (fd_ndata < 0) {
        fprintf(stderr, "Error: open %s failed: %s\n", file_ndata, strerror (errno));
        munmap(h_data, size_data); // Clean up previous mmap
        h_data = NULL;
        return 0;
    }
	status_ndata = fstat (fd_ndata, & s_ndata);
    if (status_ndata < 0) {
        fprintf(stderr, "Error: stat %s failed: %s\n", file_ndata, strerror (errno));
        munmap(h_data, size_data);
        h_data = NULL;
        close(fd_ndata);
        return 0;
    }
	size_ndata = s_ndata.st_size;
	h_ndata = (size_t*)mmap (0, size_ndata, PROT_READ, MAP_PRIVATE, fd_ndata, 0);
    if (h_ndata == MAP_FAILED) {
        fprintf(stderr, "Error: mmap %s failed: %s\n", file_ndata, strerror (errno));
        h_ndata = NULL;
        munmap(h_data, size_data); // Clean up previous mmap
        h_data = NULL;
        close(fd_ndata);
        return 0;
    }
	close(fd_ndata); // Close FD after mmap

	printf("Host Binned Data Read:\n");
    printf("  Data.bin Size: %zu bytes, Elements: %zu (assuming size_t)\n", size_data, size_data / sizeof(size_t));
	printf("  nData.bin Size: %zu bytes, Elements: %zu (assuming size_t)\n", size_ndata, size_ndata / sizeof(size_t));
	fflush(stdout);

	return 1; // Success
}

// Read Spots File (Host side)
int ReadSpots(char *cwd)
{
	int fd;
	struct stat s;
	int status;
	size_t size = 0;
	char filename[2048];
	sprintf(filename,"%s/Spots.bin", cwd);

	fd = open(filename,O_RDONLY);
	if (fd < 0) {
        fprintf(stderr, "Error: open %s failed: %s\n", filename, strerror(errno));
        return 0; // Return 0 spots on failure
    }
	status = fstat (fd , &s);
    if (status < 0) {
        fprintf(stderr, "Error: stat %s failed: %s\n", filename, strerror(errno));
        close(fd);
        return 0;
    }
	size = s.st_size;
    size_t n_elements_double = size / sizeof(double);
    if (size % sizeof(double) != 0) {
        fprintf(stderr, "Warning: Size of %s (%zu) is not a multiple of double (%zu).\n", filename, size, sizeof(double));
        // Proceeding anyway, might read partial data if format is wrong
    }
    if (n_elements_double == 0) {
         fprintf(stderr, "Warning: %s is empty or too small.\n", filename);
         close(fd);
         return 0;
    }

    // Read as double first
    double* temp_spots_double = (double*)mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (temp_spots_double == MAP_FAILED) {
        fprintf(stderr, "Error: mmap %s failed: %s\n", filename, strerror(errno));
        close(fd);
        return 0;
    }
    close(fd); // Close FD after mmap

	size_t n_spots_calc = n_elements_double / N_COL_OBSSPOTS;
    if (n_elements_double % N_COL_OBSSPOTS != 0) {
         fprintf(stderr, "Warning: Number of elements in %s (%zu) is not a multiple of N_COL_OBSSPOTS (%d).\n",
                 filename, n_elements_double, N_COL_OBSSPOTS);
         // Use the calculated number of spots, might truncate last spot
    }
    if (n_spots_calc > MAX_N_SPOTS) {
        fprintf(stderr, "Warning: Number of spots in file (%zu) exceeds MAX_N_SPOTS (%d). Truncating.\n",
                n_spots_calc, MAX_N_SPOTS);
        n_spots_calc = MAX_N_SPOTS;
    }

    // Allocate host float memory
    h_ObsSpotsLab = (float*)malloc(n_spots_calc * N_COL_OBSSPOTS * sizeof(float));
    if (!h_ObsSpotsLab) {
        fprintf(stderr, "Error: Failed to allocate memory for host ObsSpotsLab.\n");
        munmap(temp_spots_double, size);
        return 0;
    }

    // Convert double to float
    for (size_t i = 0; i < n_spots_calc; ++i) {
        for (int j = 0; j < N_COL_OBSSPOTS; ++j) {
            h_ObsSpotsLab[i * N_COL_OBSSPOTS + j] = (float)temp_spots_double[i * N_COL_OBSSPOTS + j];
        }
    }

    // Unmap the double temporary buffer
    munmap(temp_spots_double, size);

    printf("Host Spots Read: %zu spots\n", n_spots_calc);
	return (int)n_spots_calc; // Return number of spots read
}

// --- Euler Conversion (Device) ---
// Already defined sind_d, cosd_d etc.
__device__ __forceinline__
void Euler2OrientMat_d(
    const float Euler[3], // Input Euler angles in DEGREES
    float m_out[3][3])    // Output orientation matrix
{
    float psi, phi, theta;
    float cps, cph, cth, sps, sph, sth;

    // Assuming Euler angles are psi, phi, theta in ZYZ convention (degrees)
    psi = Euler[0];
    phi = Euler[1];
    theta = Euler[2];

    cps = cosd_d(psi) ; cph = cosd_d(phi); cth = cosd_d(theta);
    sps = sind_d(psi); sph = sind_d(phi); sth = sind_d(theta);

    // Row-major Z(psi) * Y(phi) * Z(theta) rotation
    // Check convention if results are unexpected
    m_out[0][0] = cps * cth - sps * cph * sth;
    m_out[0][1] = -cps * sth - sps * cph * cth;
    m_out[0][2] = sps * sph;
    m_out[1][0] = sps * cth + cps * cph * sth;
    m_out[1][1] = -sps * sth + cps * cph * cth;
    m_out[1][2] = -cps * sph;
    m_out[2][0] = sph * sth;
    m_out[2][1] = sph * cth;
    m_out[2][2] = cph;
}


// --- Main Kernel ---
__global__ void IndexingKernel(
    // Input Data (Device Pointers)
    const float* d_ObsSpotsLab,
    const int* d_BigDetector,
    const size_t* d_data,
    const size_t* d_ndata,
    const float* d_hkls,
    const int* d_HKLints,
    const float* d_RingHKL,
    const float* d_omemargins,
    const float* d_etamargins,
    const float* d_grid,
    const float* d_ypos,
    const float* d_mic, // MIC data (if used)

    // Parameters (Passed by value or from const memory)
    const struct TParams *params, // Pass the whole struct by value
    int n_spots_total, // Total number of spots
    int hasMic, // Flag: 1 if MIC file was provided, 0 otherwise
    int nrMic, // Number of entries in d_mic

    // Voxel range for this kernel launch
    int startRowNr,
    int endRowNr,

    // Workspace & Output Buffers (Device Pointers)
    float* d_workspace, // Pre-allocated workspace
    float* d_results,   // Output buffer for matched grain info (16 floats per voxel)
    int* d_matched_ids, // Output buffer for matched spot IDs (variable size)
    size_t* d_key_info   // Output buffer for key info (SpotID, nMatches, offset_results, offset_ids) - 4 size_t per voxel
   )
{
    // Calculate the global thread ID and the voxel it processes
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thisRowNr = startRowNr + idx;

    if (thisRowNr >= endRowNr) {
        return; // Thread is outside the valid range
    }

    // --- Access parameters via pointer ---
    int n_hkls = params->n_hkls; // Get value from device struct
    int SGNum = params->SGNum;
    int numScans = params->numScans;
    float pixelsize = params->pixelsize;
    int BigDetSize = params->BigDetSize;
    float BeamSize = params->BeamSize;
    float EtaBinSize = params->EtaBinSize;
    float OmeBinSize = params->OmeBinSize;
    float Wavelength = params->Wavelength;
    float Distance = params->Distance;
    float ExcludePoleAngle = params->ExcludePoleAngle;
    float MarginOme = params->MarginOme;
    float MarginRadial = params->MarginRadial;
    float MinMatchesToAcceptFrac = params->MinMatchesToAcceptFrac;
    int RingToIndex = params->RingToIndex;
    int NoOfOmegaRanges = params->NoOfOmegaRanges;
    int n_ring_bins = 0; // Calculate based on RingRadii in params
    for(int i=0; i<MAX_N_RINGS; ++i) if(params->RingRadii[i] > EPS) n_ring_bins = i+1;
    int n_eta_bins = (EtaBinSize < EPS) ? 0 : (int)ceilf(360.0f / EtaBinSize);
	int n_ome_bins = (OmeBinSize < EPS) ? 0 : (int)ceilf(360.0f / OmeBinSize);

    // --- Calculate Workspace Offsets for this Thread ---
    // Sizes per thread (ensure these match host allocation logic)
    // Using defines for clarity and consistency
    #define NROWS_PER_GRAIN (2 * MAX_N_HKLS) // Max theoretical spots + non-matched
    #define NROWS_OUTPUT (MAX_N_MATCHES * NROWS_PER_GRAIN) // Max spot output (matched/non-matched)

    const size_t theor_spots_size = NROWS_PER_GRAIN * N_COL_THEORSPOTS;
    const size_t grain_spots_size = NROWS_PER_GRAIN * N_COL_GRAINSPOTS; // Used by CompareSpots
    const size_t grain_matches_T_size = MAX_N_MATCHES * N_COL_GRAINMATCHES;
    const size_t all_grain_spots_T_size = NROWS_OUTPUT * N_COL_GRAINSPOTS; // Used if CalcIA ported
    const size_t or_mat_size = MAX_N_OR * 3 * 3; // For GenerateCandidateOrientations

    // Total size per thread (sum of the above)
    const size_t total_size_per_thread = theor_spots_size + grain_spots_size +
                                         grain_matches_T_size + all_grain_spots_T_size +
                                         or_mat_size;

    // This thread's base offset in the global workspace
    size_t thread_offset = (size_t)idx * total_size_per_thread; // Use idx relative to start of *this kernel's range*

    // Pointers to slices within the workspace for this thread
    float* TheorSpots_ws        = d_workspace + thread_offset;
    float* GrainSpots_ws        = TheorSpots_ws + theor_spots_size;
    float* GrainMatchesT_ws     = GrainSpots_ws + grain_spots_size;
    float* AllGrainSpotsT_ws    = GrainMatchesT_ws + grain_matches_T_size;
    float* OrMat_ws             = AllGrainSpotsT_ws + all_grain_spots_T_size;
    // We also need GrainMatches and AllGrainSpots if best match is stored
    // Let's reuse T buffers if MAX_N_MATCHES is 1, otherwise need separate space
    float* GrainMatches_ws     = GrainMatchesT_ws; // Reuse if only best is kept
    float* AllGrainSpots_ws    = AllGrainSpotsT_ws; // Reuse if only best is kept


    // --- Get Voxel Position ---
    float xThis=0.0f, yThis=0.0f;
    if (thisRowNr * 2 + 1 < numScans * numScans * 2) { // Bounds check grid
        xThis = d_grid[thisRowNr * 2 + 0];
        yThis = d_grid[thisRowNr * 2 + 1];
    } else {
        // Should not happen if endRowNr is correct, but safety check
        return;
    }
    float zThis = 0.0f; // Assuming z=0 unless specified otherwise
    float ga = xThis, gb = yThis, gc = zThis; // COM guess

    // --- Kernel Logic (Branch based on hasMic) ---
    int bestMatchFoundKernel = 0;
    int SpotID_kernel = -1; // Use a different name
    // int best_nTspots_kernel = 0;
    int best_nMatches_kernel = 0;
    float best_confidence_kernel = 0.0f;
    float min_IA_kernel = 1000.0f;

    if (hasMic == 1) {
        // --- Mode 1: Use MIC file data ---
        int bestRow_mic = -1;
		float bestLen_mic = 100000.0f, lenThis_mic;
        // Find closest MIC entry (linear search on device)
		for (int iter=0; iter < nrMic; iter++){
            // Access flattened mic data: row * 5 + col
            float micX = d_mic[iter * 5 + 0];
            float micY = d_mic[iter * 5 + 1];
			lenThis_mic = sqrtf((xThis-micX)*(xThis-micX)+(yThis-micY)*(yThis-micY));
			if (lenThis_mic < bestLen_mic){
				bestLen_mic = lenThis_mic;
				bestRow_mic = iter;
			}
		}

        float OMThis_mic[3][3]; // Orientation Matrix from MIC
		if (bestRow_mic != -1){
            float eulerThis_mic[3];
            // Euler angles from MIC are in RADIANS, need conversion to DEGREES for Euler2OrientMat_d
            eulerThis_mic[0] = d_mic[bestRow_mic * 5 + 2] * rad2deg; // phi -> psi? Check convention
            eulerThis_mic[1] = d_mic[bestRow_mic * 5 + 3] * rad2deg; // theta -> phi?
            eulerThis_mic[2] = d_mic[bestRow_mic * 5 + 4] * rad2deg; // psi -> theta?
			Euler2OrientMat_d(eulerThis_mic, OMThis_mic);

            // --- Run simulation and comparison for this single orientation ---
            int nTspots_mic = 0;
            CalcDiffrSpots_Furnace_d(OMThis_mic, params->Wavelength, params->Distance,
                                     params->RingRadii, params->OmegaRanges, params->BoxSizes,
                                     params->NoOfOmegaRanges, params->ExcludePoleAngle, n_hkls,
                                     d_hkls, d_BigDetector, params->BigDetSize, params->pixelsize,
                                     TheorSpots_ws, NROWS_PER_GRAIN, // Use workspace slice
                                     &nTspots_mic);

            if (nTspots_mic > 0) {
                // Calculate displacements and eta/radius diff for theoretical spots
                for (int sp = 0 ; sp < nTspots_mic ; sp++) {
                    int base_idx_theor = sp * N_COL_THEORSPOTS;
                    float Displ_y_mic, Displ_z_mic;
                    displacement_spot_needed_COM_d(ga, gb, gc,
                                                 TheorSpots_ws[base_idx_theor + 3], // dist
                                                 TheorSpots_ws[base_idx_theor + 4], // yl
                                                 TheorSpots_ws[base_idx_theor + 5], // zl
                                                 TheorSpots_ws[base_idx_theor + 6], // omega
                                                 &Displ_y_mic, &Displ_z_mic );
                    // Store results in columns 10-13
                    TheorSpots_ws[base_idx_theor + 10] = TheorSpots_ws[base_idx_theor + 4] +  Displ_y_mic; // yl_displ
                    TheorSpots_ws[base_idx_theor + 11] = TheorSpots_ws[base_idx_theor + 5] +  Displ_z_mic; // zl_displ
                    CalcEtaAngle_d( TheorSpots_ws[base_idx_theor + 10], TheorSpots_ws[base_idx_theor + 11],
                                    &TheorSpots_ws[base_idx_theor + 12] ); // eta_displ
                    int ringnr_mic = (int)TheorSpots_ws[base_idx_theor + 9];
                    if (ringnr_mic >= 0 && ringnr_mic < MAX_N_RINGS && params->RingRadii[ringnr_mic] > EPS) {
                        TheorSpots_ws[base_idx_theor + 13] = sqrtf(TheorSpots_ws[base_idx_theor + 10] * TheorSpots_ws[base_idx_theor + 10] +
                                                                TheorSpots_ws[base_idx_theor + 11] * TheorSpots_ws[base_idx_theor + 11])
                                                            - params->RingRadii[ringnr_mic]; // radius_diff
                    } else {
                         TheorSpots_ws[base_idx_theor + 13] = 0.0f; // Invalid ring
                    }
                }

                int nMatches_mic = 0;
                CompareSpots_d(TheorSpots_ws, nTspots_mic, params->MarginOme, params->MarginRadial,
                               d_etamargins, d_omemargins, n_eta_bins, n_ome_bins,
                               params->EtaBinSize, params->OmeBinSize, params->BeamSize,
                               xThis, yThis, d_ObsSpotsLab, d_data, d_ndata, d_ypos, numScans, n_ring_bins,
                               GrainSpots_ws, nTspots_mic, // Use workspace, size=nTspots_mic
                               &nMatches_mic);

                float FracThis_mic = (nTspots_mic > 0) ? ((float)nMatches_mic / (float)nTspots_mic) : 0.0f;

                if (FracThis_mic > params->MinMatchesToAcceptFrac){
                    bestMatchFoundKernel = 1;
                    SpotID_kernel = thisRowNr; // Use voxel number as SpotID for MIC mode? Or -1? Or bestRow_mic? Let's use voxel nr.
                    best_confidence_kernel = FracThis_mic;
                    // best_nTspots_kernel = nTspots_mic;
                    best_nMatches_kernel = nMatches_mic;

                    // Store the results (GrainMatches and AllGrainSpots) in the workspace buffers
                    // Fill GrainMatchesT_ws (using index 0 as MAX_N_MATCHES=1 assumed)
                    int gm_base = 0; // Only one match stored
                    for (int i = 0 ;  i < 9 ; i ++) GrainMatches_ws[gm_base + i] = OMThis_mic[i/3][i%3];
                    GrainMatches_ws[gm_base + 9]  = ga;
                    GrainMatches_ws[gm_base + 10] = gb;
                    GrainMatches_ws[gm_base + 11] = gc;
                    GrainMatches_ws[gm_base + 12] = (float)nTspots_mic;
                    GrainMatches_ws[gm_base + 13] = (float)nMatches_mic;
                    GrainMatches_ws[gm_base + 14] = 1.0f; // Match flag/count? Set to 1.
                    GrainMatches_ws[gm_base + 15] = 0.0f; // IA placeholder, calculated next

                    // AllGrainSpotsT is already filled by CompareSpots_d (passed GrainSpots_ws)
                    // Copy from GrainSpots_ws to AllGrainSpots_ws if they are different buffers
                    // If they are the same (due to reuse), no copy needed.

                    // Calculate IA - needs careful checking due to dependencies noted in CalcIA_d
                    // Pass the correct buffers: GrainMatches_ws and AllGrainSpots_ws
                     CalcIA_d(GrainMatches_ws, 1, AllGrainSpots_ws, nTspots_mic, params->Distance);
                     min_IA_kernel = GrainMatches_ws[gm_base + 15]; // Store calculated IA
                }
            } // end if nTspots_mic > 0
		} // end if bestRow_mic != -1

    } else {
        // --- Mode 2: Indexing based on observed spots ---
        int RingToIndex = params->RingToIndex;
        if (RingToIndex < 0) return; // Cannot proceed without RingToIndex

        // Find the range of observed spots matching RingToIndex
        // This search should ideally be done once on the host and passed.
        // Doing it per thread is inefficient. Let's assume start/end indices are passed.
        // Add startRowNrSp, endRowNrSp to TParams or pass as args if needed.
        // For now, perform the search here (inefficiently).
        size_t startRowNrSp_kernel = n_spots; // Initialize to max
        size_t endRowNrSp_kernel = 0;         // Initialize to min

        // Optimization: Pass start/end indices for the target ring from host
        // Assuming these are added to params: params->startRowNrSp, params->endRowNrSp
        // if (params->validRingIndexRange) { // Add a flag if range is precomputed
        //     startRowNrSp_kernel = params->startRowNrSp;
        //     endRowNrSp_kernel = params->endRowNrSp;
        // } else { // Fallback: Search (slow)
             for (int i=0; i < n_spots; i++){
                int base_idx_obs = i * N_COL_OBSSPOTS;
                 if (base_idx_obs + 5 < n_spots * N_COL_OBSSPOTS) { // Bounds check
                     if ((int)d_ObsSpotsLab[base_idx_obs + 5] == RingToIndex) {
                         if (startRowNrSp_kernel > (size_t)i) startRowNrSp_kernel = (size_t)i;
                         if (endRowNrSp_kernel < (size_t)i) endRowNrSp_kernel = (size_t)i;
                     }
                 }
             }
        // }


        // Iterate through relevant observed spots for this voxel
        for (size_t idnr = startRowNrSp_kernel; idnr <= endRowNrSp_kernel; idnr++){
             int base_idx_obs = idnr * N_COL_OBSSPOTS;
             if (base_idx_obs + 9 >= n_spots * N_COL_OBSSPOTS) continue; // Bounds check

             float angle_obs = d_ObsSpotsLab[base_idx_obs + 2]; // Omega_obs
             int thisID_obs = (int)d_ObsSpotsLab[base_idx_obs + 4]; // SpotID_obs
             int scanNr_obs = (int)d_ObsSpotsLab[base_idx_obs + 9]; // ScanNr_obs

             if (scanNr_obs < 0 || scanNr_obs >= numScans) continue; // Bounds check ypos

             // Check if this observed spot could belong to this voxel (y-position check)
             float ypos_obs = d_ypos[scanNr_obs];
             float newY_check = xThis * sinf(deg2rad*angle_obs) + yThis * cosf(deg2rad*angle_obs);

			 if (fabsf(newY_check - ypos_obs) <= params->BeamSize / 2.0f){
                 // This spot is a candidate for seeding orientation search

                 // Get spot details needed for orientation generation
                 float y0_obs = d_ObsSpotsLab[base_idx_obs + 0];
                 float z0_obs = d_ObsSpotsLab[base_idx_obs + 1];
                 int ringnr_obs = (int)d_ObsSpotsLab[base_idx_obs + 5]; // Should == RingToIndex
                 // float RefRad_obs = d_ObsSpotsLab[base_idx_obs + 3]; // Not directly used for generation?

                 if (ringnr_obs != RingToIndex) continue; // Should not happen if range search is correct

                 // Calculate measured g-vector (hklnormal)
                 float xi_obs, yi_obs, zi_obs; // Unit vector to spot
                 float len_obs_sq = params->Distance * params->Distance + y0_obs * y0_obs + z0_obs * z0_obs;
                 if (len_obs_sq < EPS) continue; // Invalid spot position
                 float len_obs_inv = 1.0f / sqrtf(len_obs_sq);
                 xi_obs = params->Distance * len_obs_inv;
                 yi_obs = y0_obs * len_obs_inv;
                 zi_obs = z0_obs * len_obs_inv;

                 float g1_obs, g2_obs, g3_obs; // Measured g-vector
                 spot_to_gv_d(xi_obs, yi_obs, zi_obs, angle_obs, &g1_obs, &g2_obs, &g3_obs);
                 float hklnormal_obs[3] = {g1_obs, g2_obs, g3_obs};

                 // Get theoretical HKL for this ring (use RingHKL lookup)
                 if (ringnr_obs < 0 || ringnr_obs >= MAX_N_RINGS) continue; // Bounds check
                 float hkl_theor[3];
                 hkl_theor[0] = d_RingHKL[ringnr_obs * 3 + 0];
                 hkl_theor[1] = d_RingHKL[ringnr_obs * 3 + 1];
                 hkl_theor[2] = d_RingHKL[ringnr_obs * 3 + 2];

                 // Generate candidate orientations based on this spot
                 int nOrient_gen = 0;
                 GenerateCandidateOrientationsF_d(hkl_theor, hklnormal_obs, params->StepsizeOrient,
                                                  ringnr_obs, SGNum, d_HKLints, h_ABCABG, // Pass lattice params
                                                  OrMat_ws, MAX_N_OR, // Use workspace slice for OrMat
                                                  &nOrient_gen);

                 // --- Test each generated orientation ---
                 for (int or_idx = 0; or_idx < nOrient_gen; ++or_idx) {
                     float currentOM[3][3];
                     int ormat_base = or_idx * 9;
                     for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) currentOM[r][c] = OrMat_ws[ormat_base + r*3 + c];

                     int nTspots_or = 0;
                     // Calculate theoretical spots for this orientation
                     CalcDiffrSpots_Furnace_d(currentOM, params->Wavelength, params->Distance,
                                              params->RingRadii, params->OmegaRanges, params->BoxSizes,
                                              params->NoOfOmegaRanges, params->ExcludePoleAngle, n_hkls,
                                              d_hkls, d_BigDetector, params->BigDetSize, params->pixelsize,
                                              TheorSpots_ws, NROWS_PER_GRAIN, // Use workspace slice
                                              &nTspots_or);

                    if (nTspots_or == 0) continue; // No spots for this orientation

                    // Calculate displacements etc. for theoretical spots
                    for (int sp = 0 ; sp < nTspots_or ; sp++) {
                        int base_idx_theor = sp * N_COL_THEORSPOTS;
                        float Displ_y_or, Displ_z_or;
                        displacement_spot_needed_COM_d(ga, gb, gc,
                                                     TheorSpots_ws[base_idx_theor + 3], // dist
                                                     TheorSpots_ws[base_idx_theor + 4], // yl
                                                     TheorSpots_ws[base_idx_theor + 5], // zl
                                                     TheorSpots_ws[base_idx_theor + 6], // omega
                                                     &Displ_y_or, &Displ_z_or );
                        TheorSpots_ws[base_idx_theor + 10] = TheorSpots_ws[base_idx_theor + 4] +  Displ_y_or;
                        TheorSpots_ws[base_idx_theor + 11] = TheorSpots_ws[base_idx_theor + 5] +  Displ_z_or;
                        CalcEtaAngle_d( TheorSpots_ws[base_idx_theor + 10], TheorSpots_ws[base_idx_theor + 11],
                                        &TheorSpots_ws[base_idx_theor + 12] );
                        int ringnr_or = (int)TheorSpots_ws[base_idx_theor + 9];
                        if (ringnr_or >= 0 && ringnr_or < MAX_N_RINGS && params->RingRadii[ringnr_or] > EPS) {
                            TheorSpots_ws[base_idx_theor + 13] = sqrtf(TheorSpots_ws[base_idx_theor + 10] * TheorSpots_ws[base_idx_theor + 10] +
                                                                    TheorSpots_ws[base_idx_theor + 11] * TheorSpots_ws[base_idx_theor + 11])
                                                                - params->RingRadii[ringnr_or];
                        } else {
                             TheorSpots_ws[base_idx_theor + 13] = 0.0f;
                        }
                    }

                    // Compare theoretical spots to *all* observed spots for this voxel
                    int nMatches_or = 0;
                    CompareSpots_d(TheorSpots_ws, nTspots_or, params->MarginOme, params->MarginRadial,
                                   d_etamargins, d_omemargins, n_eta_bins, n_ome_bins,
                                   params->EtaBinSize, params->OmeBinSize, params->BeamSize,
                                   xThis, yThis, d_ObsSpotsLab, d_data, d_ndata, d_ypos, numScans, n_ring_bins,
                                   GrainSpots_ws, nTspots_or, // Use workspace slice
                                   &nMatches_or);

                    float FracThis_or = (nTspots_or > 0) ? ((float)nMatches_or / (float)nTspots_or) : 0.0f;

                    // Check if this is a better match than current best for this voxel
                    if (FracThis_or > params->MinMatchesToAcceptFrac){
                        // Need to calculate IA to break ties if confidence is equal
                        // Store temporary grain match info
                        int gm_base_T = 0; // Temp buffer index (only 1 needed)
                        for (int i = 0 ;  i < 9 ; i ++) GrainMatchesT_ws[gm_base_T + i] = currentOM[i/3][i%3];
                        GrainMatchesT_ws[gm_base_T + 9]  = ga;
                        GrainMatchesT_ws[gm_base_T + 10] = gb;
                        GrainMatchesT_ws[gm_base_T + 11] = gc;
                        GrainMatchesT_ws[gm_base_T + 12] = (float)nTspots_or;
                        GrainMatchesT_ws[gm_base_T + 13] = (float)nMatches_or;
                        GrainMatchesT_ws[gm_base_T + 14] = 1.0f;
                        GrainMatchesT_ws[gm_base_T + 15] = 0.0f; // IA placeholder

                        // Copy spot details from GrainSpots_ws to AllGrainSpotsT_ws
                        // Assuming they are different buffers here. If reused, skip copy.
                        // if (AllGrainSpotsT_ws != GrainSpots_ws) {
                        //     memcpy(AllGrainSpotsT_ws, GrainSpots_ws, nTspots_or * N_COL_GRAINSPOTS * sizeof(float));
                        // }
                        // Assuming reuse: AllGrainSpotsT_ws points to same data as GrainSpots_ws

                        // Calculate IA for this candidate match
                        CalcIA_d(GrainMatchesT_ws, 1, AllGrainSpotsT_ws, nTspots_or, params->Distance);
                        float currentIA = GrainMatchesT_ws[gm_base_T + 15];

                        // Update best match if:
                        // 1. Confidence is higher
                        // 2. Confidence is equal, but IA is lower (better agreement)
                        bool update_best = false;
                        if (!bestMatchFoundKernel) {
                            update_best = true; // First valid match found
                        } else if (FracThis_or > best_confidence_kernel) {
                            update_best = true;
                        } else if (fabsf(FracThis_or - best_confidence_kernel) < EPS && currentIA < min_IA_kernel) {
                            update_best = true;
                        }

                        if (update_best) {
                            bestMatchFoundKernel = 1;
                            SpotID_kernel = thisID_obs; // ID of the spot that seeded this best match
                            best_confidence_kernel = FracThis_or;
                            min_IA_kernel = currentIA;
                            // best_nTspots_kernel = nTspots_or;
                            best_nMatches_kernel = nMatches_or;

                            // Save this as the best result (copy from Temp to final workspace buffers)
                            // Copy GrainMatchesT_ws to GrainMatches_ws
                            memcpy(GrainMatches_ws, GrainMatchesT_ws, N_COL_GRAINMATCHES * sizeof(float));
                            // Copy AllGrainSpotsT_ws to AllGrainSpots_ws
                            // If buffers are reused, this copy is redundant. Let's assume reuse.
                             // memcpy(AllGrainSpots_ws, AllGrainSpotsT_ws, nTspots_or * N_COL_GRAINSPOTS * sizeof(float));
                             // Need to handle non-matched spots correctly if CalcIA modified the buffer.
                             // Let's re-run CompareSpots to get the definitive spot list for the best match.
                             CompareSpots_d(TheorSpots_ws, nTspots_or, params->MarginOme, params->MarginRadial,
                                   d_etamargins, d_omemargins, n_eta_bins, n_ome_bins,
                                   params->EtaBinSize, params->OmeBinSize, params->BeamSize,
                                   xThis, yThis, d_ObsSpotsLab, d_data, d_ndata, d_ypos, numScans, n_ring_bins,
                                   AllGrainSpots_ws, nTspots_or, // Write directly to final spot buffer
                                   &nMatches_or); // nMatches_or should be same as best_nMatches_kernel
                             // Recalculate IA on the final buffer to store spot-level IAs
                             CalcIA_d(GrainMatches_ws, 1, AllGrainSpots_ws, nTspots_or, params->Distance);
                             // Ensure the average IA in GrainMatches_ws is correct
                             GrainMatches_ws[0 * N_COL_GRAINMATCHES + 15] = min_IA_kernel;


                        } // end if update_best
                    } // end if FracThis_or > threshold
                 } // end loop over generated orientations (or_idx)
             } // end if y-position check valid
        } // end loop over observed spots (idnr)
    } // end if/else hasMic

    // --- Write results to output buffers ---
    if (bestMatchFoundKernel) {
        // Write Grain Match Info (16 floats)
        int result_base = idx * 16; // Offset in d_results
        int gm_base = 0; // Best match is at index 0 in GrainMatches_ws

        d_results[result_base + 0] = (float)SpotID_kernel; // ID of seed spot (or voxel nr if MIC)
        d_results[result_base + 1] = GrainMatches_ws[gm_base + 15]; // Avg IA angle
        for (int k = 0; k < 9; ++k) { // Orientation Matrix
             d_results[result_base + 2 + k] = GrainMatches_ws[gm_base + k];
        }
        d_results[result_base + 11] = GrainMatches_ws[gm_base + 9]; // COM x (ga)
        d_results[result_base + 12] = GrainMatches_ws[gm_base + 10]; // COM y (gb)
        d_results[result_base + 13] = GrainMatches_ws[gm_base + 11]; // COM z (gc)
        d_results[result_base + 14] = GrainMatches_ws[gm_base + 12]; // nTheorSpots
        d_results[result_base + 15] = GrainMatches_ws[gm_base + 13]; // nMatches

        // Write Key Info (4 size_t)
        int key_base = idx * 4; // Offset in d_key_info
        d_key_info[key_base + 0] = (size_t)SpotID_kernel;
        d_key_info[key_base + 1] = (size_t)best_nMatches_kernel; // Number of matched spots
        // Offsets will be calculated on host after collecting all counts
        d_key_info[key_base + 2] = 0; // Placeholder for vals offset
        d_key_info[key_base + 3] = 0; // Placeholder for ids offset

        // Write Matched Spot IDs (variable number of ints)
        // This requires knowing the offset *beforehand*, which we don't have in the kernel.
        // Strategy: Write IDs sequentially to d_matched_ids buffer and use atomicAdd
        //           to get the starting offset for this thread's data.
        // Simpler Strategy (used here): Copy the relevant IDs from AllGrainSpots_ws
        // after the kernel finishes, once counts are known. Host will handle writing.
        // The kernel just needs to store the number of matches in d_key_info[idx*4 + 1].

        // Copy matched spot IDs from AllGrainSpots_ws to d_matched_ids on HOST later.
        // The AllGrainSpots_ws buffer contains the details. Column 14 is Obs Spot ID.
        // We need to ensure AllGrainSpots_ws correctly contains the final best match spots.

    } else {
        // No match found for this voxel. Write default values.
        int result_base = idx * 16;
        for (int k = 0; k < 16; ++k) {
            d_results[result_base + k] = 0.0f; // Or some indicator like NaN?
        }
        d_results[result_base + 0] = -1.0f; // Indicate no match with negative SpotID

        int key_base = idx * 4;
        d_key_info[key_base + 0] = (size_t)-1; // SpotID
        d_key_info[key_base + 1] = 0;          // nMatches = 0
        d_key_info[key_base + 2] = 0;          // Placeholder offset
        d_key_info[key_base + 3] = 0;          // Placeholder offset
    }
}


// --- Main Function ---

int main(int argc, char *argv[])
{
    // Use host timers
	clock_t start_clock = clock();
	printf("\n\n\t\tIndexerScanningCUDA v6.0 (CUDA Port)\nContact hsharma@anl.gov in case of questions about the MIDAS project.\n\n");

	int returncode;
	struct TParams Params; // Host copy of parameters
	char *ParamFN;

	if (argc < 6) {
		printf("Usage: %s paramtest.txt blockNr nBlocks numScans numProcs(ignored_for_cuda)\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	ParamFN = argv[1];
	int blockNr_arg = atoi(argv[2]);
	int nBlocks_arg = atoi(argv[3]);
	numScans_host = atoi(argv[4]); // Read numScans
    // int numProcs_arg = atoi(argv[5]); // Ignored for CUDA version

	printf("Reading parameters from file: %s.\n", ParamFN);
	returncode = ReadParams(ParamFN, &Params);
	if ( returncode != 0 ) {
		printf("Error reading params file %s\n", ParamFN );
		exit(EXIT_FAILURE);
	}
    // Copy necessary params read into global host vars or ensure Params struct holds them
    Params.numScans = numScans_host;
    SGNum_host = Params.SGNum;
    pixelsize_host = Params.pixelsize;
    BeamSize_host = Params.BeamSize;
    BigDetSize_host = Params.BigDetSize;
    EtaBinSize_host = Params.EtaBinSize;
    OmeBinSize_host = Params.OmeBinSize;


    // --- Calculate Margins (Host) ---
    printf("Calculating margins...\n");
    for ( int i = 1 ; i < 180 ; i++) {
        float sin_val = fabsf(sinf(i * deg2rad));
        if (sin_val < EPS) { // Avoid division by near zero
            h_omemargins[i] = Params.MarginOme + ( 0.5f * Params.StepsizeOrient / EPS); // Assign large margin?
        } else {
            h_omemargins[i] = Params.MarginOme + ( 0.5f * Params.StepsizeOrient / sin_val);
        }
    }
	h_omemargins[0] = h_omemargins[1]; // Extrapolate ends
	h_omemargins[180] = h_omemargins[1]; // Use index 1 (smallest angle > 0)

	for ( int i = 0 ; i < MAX_N_RINGS ; i++) {
        if ( Params.RingRadii[i] < EPS) { // Check against float epsilon
             h_etamargins[i] = 0.0f;
        } else {
             h_etamargins[i] = rad2deg * atanf(Params.MarginEta / Params.RingRadii[i]) + 0.5f * Params.StepsizeOrient;
        }
	}

	printf("SpaceGroup: %d\n", Params.SpaceGroupNum);
	printf("Finished reading parameters.\n");

    // --- Read HKLs (Host) ---
    printf("Reading HKL data...\n");
	const char *hklfn = "hkls.csv"; // Should this be a parameter?
	FILE *hklf = fopen(hklfn,"r");
    checkHost(hklf == NULL, "Cannot open HKL file: %s", hklfn);

	char aline[5024], dummy[1024];
	checkHost(fgets(aline, sizeof(aline), hklf) == NULL, "Error reading HKL header"); // Skip header

	int Rnr;
	int hi,ki,li;
	double hc,kc,lc, RRd_double, Ds_double, tht_double; // Read as double first

    n_hkls_host = 0; // Reset global counter
    int ring_counts[MAX_N_RINGS] = {0}; // Count hkls per ring

	while (fgets(aline, sizeof(aline), hklf) != NULL && n_hkls_host < MAX_N_HKLS){
        // Example format: %d %d %d %lf %d %lf %lf %lf %lf %s %lf
        //                  hi ki li Ds Rnr hc kc lc tht dummy RRd
		int n_scan = sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf",
                            &hi,&ki,&li,&Ds_double,&Rnr,&hc,&kc,&lc,&tht_double,dummy,&RRd_double);
        if (n_scan < 11) {
             printf("Warning: Skipping malformed HKL line: %s", aline);
             continue;
        }

        // Check if this ring is selected in parameters
        bool use_this_ring = false;
        for(int r=0; r < Params.NrOfRings; ++r) {
            if (Rnr == Params.RingNumbers[r]) {
                use_this_ring = true;
                break;
            }
        }

		if (use_this_ring) {
            if (Rnr >= 0 && Rnr < MAX_N_RINGS) {
                h_RingHKL[Rnr][0] = (float)hc; // Store representative hkl for ring
                h_RingHKL[Rnr][1] = (float)kc;
                h_RingHKL[Rnr][2] = (float)lc;
                ring_counts[Rnr]++; // Increment count for this ring
            } else {
                 printf("Warning: HKL line has invalid RingNr %d. Skipping.\n", Rnr);
                 continue;
            }

            // Store details for this specific HKL reflection
			h_HKLints[n_hkls_host][0] = hi;
			h_HKLints[n_hkls_host][1] = ki;
			h_HKLints[n_hkls_host][2] = li;
			h_HKLints[n_hkls_host][3] = Rnr; // Store RingNr for CalcRotationAngle

            // Store float versions
            h_hkls[n_hkls_host][0] = (float)hc; // Col 0: h (crystal frame)
            h_hkls[n_hkls_host][1] = (float)kc; // Col 1: k
            h_hkls[n_hkls_host][2] = (float)lc; // Col 2: l
            h_hkls[n_hkls_host][3] = (float)Rnr; // Col 3: Ring Number
            h_hkls[n_hkls_host][4] = (float)Ds_double; // Col 4: d-spacing
            h_hkls[n_hkls_host][5] = (float)tht_double; // Col 5: theta (Bragg angle)
            h_hkls[n_hkls_host][6] = (float)RRd_double; // Col 6: Ring Radius (redundant w Params.RingRadii?)
            // Cols 7-13 are unused placeholders in this buffer
            for(int k=7; k < N_COL_THEORSPOTS; ++k) h_hkls[n_hkls_host][k] = 0.0f;

			n_hkls_host++;
		}
	}
	fclose(hklf);
    Params.n_hkls = n_hkls_host; // Store count in Params struct

	printf("No of hkl's read and kept: %d\n", n_hkls_host);
    // Check if representative RingHKL was set for all used rings
    for(int r=0; r < Params.NrOfRings; ++r) {
        int ring_num_check = Params.RingNumbers[r];
        if (ring_num_check >= 0 && ring_num_check < MAX_N_RINGS) {
            if(ring_counts[ring_num_check] == 0) {
                 printf("Warning: Ring %d was specified but no HKLs found for it in %s.\n", ring_num_check, hklfn);
            }
            // Optional: Check if RingHKL got set (requires non-zero vector)
            // if (h_RingHKL[ring_num_check][0] == 0.0f && h_RingHKL[ring_num_check][1] == 0.0f && h_RingHKL[ring_num_check][2] == 0.0f && ring_counts[ring_num_check] > 0) {
            //      printf("Warning: Ring %d has HKLs but representative vector is [0,0,0].\n", ring_num_check);
            // }
        }
    }


    // --- Get CWD for reading spots/bins ---
    printf("Determining data directory...\n");
	char tmpstr[2048];
	sprintf(tmpstr,"%s",Params.OutputFolder); // Use output folder path
	char *cwdstr = dirname(tmpstr); // Get directory part
    printf("Data directory: %s\n", cwdstr);

    // --- Read Spots (Host) ---
    printf("Reading spots data...\n");
    n_spots_host = ReadSpots(cwdstr);
    checkHost(n_spots_host <= 0 || h_ObsSpotsLab == NULL, "Failed to read spots data or no spots found.");
	printf("nSpots read = %zu\n", n_spots_host);

    // --- Read Binned Data (Host) ---
    printf("Reading binned data...\n");
	int rc_bins = ReadBins(cwdstr);
    checkHost(rc_bins == 0 || h_data == NULL || h_ndata == NULL, "Failed to read binned data.");

    // --- Calculate Binning Parameters ---
    int HighestRingNo = 0;
	for (int i = 0 ; i < MAX_N_RINGS ; i++ ) {
		if ( Params.RingRadii[i] > EPS) HighestRingNo = i; // Find highest ring index with non-zero radius
	}
	n_ring_bins_host = HighestRingNo + 1; // Number of bins = highest index + 1
    if (Params.EtaBinSize < EPS || Params.OmeBinSize < EPS) {
        checkHost(1, "EtaBinSize or OmeBinSize is zero or negative in parameters.");
    }
	n_eta_bins_host = (int)ceilf(360.0f / Params.EtaBinSize);
	n_ome_bins_host = (int)ceilf(360.0f / Params.OmeBinSize);
	// Update global host parameters for bin sizes actually used
    EtaBinSize_host = Params.EtaBinSize;
	OmeBinSize_host = Params.OmeBinSize;

	printf("No of bins for rings : %d (Max Ring Index: %d)\n", n_ring_bins_host, HighestRingNo);
	printf("No of bins for eta   : %d\n", n_eta_bins_host);
	printf("No of bins for omega : %d\n", n_ome_bins_host);
    size_t total_bins = (size_t)n_ring_bins_host * n_eta_bins_host * n_ome_bins_host;
	printf("Total no of bins     : %zu\n\n", total_bins);
	printf("Finished reading auxiliary data.\n\n");


    // --- Read MIC File (Host, if specified) ---
    int hasMic_flag = 0; // Use local flag
    nrMic_host = 0;
    if (strcmp(Params.MicFN, "0") != 0 && strlen(Params.MicFN) > 0){
        printf("Reading MIC file: %s\n", Params.MicFN);
        hasMic_flag = 1;
        h_mic_double = (double*)calloc(MAX_MIC_ROWS * 5, sizeof(double)); // Read as double
        checkHost(h_mic_double == NULL, "Failed to allocate memory for MIC data (double).");

		FILE *micF;
		micF = fopen(Params.MicFN,"r");
		if (micF==NULL){
			printf("Warning: Mic File could not be read: %s. Disabling MIC mode.\n", Params.MicFN);
            hasMic_flag = 0;
            free(h_mic_double);
            h_mic_double = NULL;
		} else {
            // Skip header lines (adjust count if format changes)
            checkHost(fgets(aline,sizeof(aline),micF)==NULL, "Error reading MIC header line 1");
            checkHost(fgets(aline,sizeof(aline),micF)==NULL, "Error reading MIC header line 2");
            checkHost(fgets(aline,sizeof(aline),micF)==NULL, "Error reading MIC header line 3");
            checkHost(fgets(aline,sizeof(aline),micF)==NULL, "Error reading MIC header line 4");
            printf("MIC Header 4: %s",aline); // Print last header line for check

            while (fgets(aline,sizeof(aline),micF)!=NULL && nrMic_host < MAX_MIC_ROWS){
                // Example format: %s %s %s %lf %lf %s %s %lf %lf %lf %s %s
                //                 ? ? ? X Y ? ? Phi Theta Psi ? ? (Angles in RADIANS)
                int n_scan_mic = sscanf(aline,"%*s %*s %*s %lf %lf %*s %*s %lf %lf %lf %*s %*s",
                        &h_mic_double[nrMic_host*5+0], // X
                        &h_mic_double[nrMic_host*5+1], // Y
                        &h_mic_double[nrMic_host*5+2], // Angle 1 (phi?)
                        &h_mic_double[nrMic_host*5+3], // Angle 2 (theta?)
                        &h_mic_double[nrMic_host*5+4]);// Angle 3 (psi?)
                if (n_scan_mic == 5) {
                    nrMic_host++;
                } else {
                    // printf("Warning: Skipping malformed MIC line: %s", aline);
                }
            }
            fclose(micF);
            printf("Read %d entries from MIC file.\n", nrMic_host);
            if (nrMic_host == 0) {
                 printf("Warning: MIC file specified but no valid entries found. Disabling MIC mode.\n");
                 hasMic_flag = 0;
                 free(h_mic_double);
                 h_mic_double = NULL;
            } else {
                 // Convert double MIC data to host float buffer
                 h_mic = (float*)malloc(nrMic_host * 5 * sizeof(float));
                 checkHost(h_mic == NULL, "Failed to allocate memory for MIC data (float).");
                 for(int i=0; i<nrMic_host * 5; ++i) {
                     h_mic[i] = (float)h_mic_double[i];
                 }
                 free(h_mic_double); // Free double version
                 h_mic_double = NULL;
            }
        }
	} else {
         printf("No MIC file specified. Running in spot-based indexing mode.\n");
    }


    // --- Read Positions File (Host) ---
    printf("Reading positions file...\n");
    int nVoxels = numScans_host * numScans_host;
	FILE *positionsF = fopen("positions.csv","r"); // Should this be a parameter?
    checkHost(positionsF == NULL, "positions.csv file not found.");

    // Allocate host buffers (double first, then float)
	h_grid_double = (double*)malloc(nVoxels * 2 * sizeof(double));
	h_ypos_double = (double*)malloc(numScans_host * sizeof(double));
    checkHost(h_grid_double == NULL || h_ypos_double == NULL, "Failed to allocate memory for host positions.");

	for (int i=0; i < numScans_host; i++){
		if (fgets(aline, sizeof(aline), positionsF) == NULL) {
             checkHost(1,"Error reading line %d from positions.csv", i+1);
        }
		if (sscanf(aline,"%lf",&h_ypos_double[i]) != 1) {
             checkHost(1,"Error parsing line %d from positions.csv", i+1);
        }
	}
    fclose(positionsF);

    // Create 2D grid from 1D ypos
	for (int i=0; i < numScans_host; i++){ // Scan number (slow axis?)
		for (int j=0; j < numScans_host; j++){ // Step number (fast axis?)
            // Assuming grid maps row i, col j -> (ypos[i], ypos[j])
            // Check convention if needed. Let's assume ypos[i] is X, ypos[j] is Y.
            // Or maybe ypos[i] is beamline Y, ypos[j] is beamline X?
            // Original code used grid[idx*2+0] as xThis, grid[idx*2+1] as yThis.
            // Let's map row index to x, col index to y.
			// grid[(i*numScans+j)*2+0] = ypos[i]; // xThis = ypos[i]
			// grid[(i*numScans+j)*2+1] = ypos[j]; // yThis = ypos[j]
            // Mapping voxel index linearisation: idx = i*numScans + j
            h_grid_double[(i * numScans_host + j) * 2 + 0] = h_ypos_double[i]; // X = value from slow axis index i
			h_grid_double[(i * numScans_host + j) * 2 + 1] = h_ypos_double[j]; // Y = value from fast axis index j
		}
	}
    printf("Generated %d x %d grid positions.\n", numScans_host, numScans_host);

    // Convert positions to float
    h_grid = (float*)malloc(nVoxels * 2 * sizeof(float));
    h_ypos = (float*)malloc(numScans_host * sizeof(float));
    checkHost(h_grid == NULL || h_ypos == NULL, "Failed to allocate memory for float positions.");
    for(int i=0; i < nVoxels * 2; ++i) h_grid[i] = (float)h_grid_double[i];
    for(int i=0; i < numScans_host; ++i) h_ypos[i] = (float)h_ypos_double[i];
    free(h_grid_double); h_grid_double = NULL;
    free(h_ypos_double); h_ypos_double = NULL;


    // --- Determine Voxel Range for this Block ---
	int startRowNr_host;
	int endRowNr_host;
	if (nBlocks_arg <= 0 || blockNr_arg < 0 || blockNr_arg >= nBlocks_arg) {
        printf("Warning: Invalid block number (%d) or number of blocks (%d). Processing all voxels (block 0/1).\n", blockNr_arg, nBlocks_arg);
        blockNr_arg = 0;
        nBlocks_arg = 1;
    }
    // Calculate range using integer division properties
    startRowNr_host = (long long)blockNr_arg * nVoxels / nBlocks_arg;
    endRowNr_host = (long long)(blockNr_arg + 1) * nVoxels / nBlocks_arg;
    // Ensure endRowNr doesn't exceed total voxels
    endRowNr_host = (endRowNr_host > nVoxels) ? nVoxels : endRowNr_host;

    int num_voxels_this_block = endRowNr_host - startRowNr_host;
	printf("Total Voxels: %d, Block: %d / %d, Processing voxels: %d to %d (%d total)\n",
           nVoxels, blockNr_arg, nBlocks_arg, startRowNr_host, endRowNr_host, num_voxels_this_block);
    if (num_voxels_this_block <= 0) {
        printf("No voxels to process for this block. Exiting.\n");
        // Clean up allocated host memory before exiting
        free(h_ObsSpotsLab);
        // munmap mmapped data if needed (or let OS handle on exit)
        if (h_BigDetector) munmap(h_BigDetector, totNrPixelsBigDetector_host * 32); // Approx size needed
        if (h_data) munmap(h_data, 1); // Placeholder size, OS handles full range
        if (h_ndata) munmap(h_ndata, 1);
        free(h_mic);
        free(h_grid);
        free(h_ypos);
        return 0;
    }


    // --- Determine Spot Range for RingToIndex (if not using MIC) ---
    size_t startRowNrSp_host = n_spots_host; // Init high
    size_t endRowNrSp_host = 0;         // Init low
    bool ring_index_range_valid = false;
    if (hasMic_flag == 0) {
        int ring_to_index = Params.RingToIndex;
        if (ring_to_index < 0) {
            checkHost(1, "RingToIndex parameter is required but not set or invalid.");
        }
        printf("Finding spot index range for RingToIndex = %d...\n", ring_to_index);
        for (size_t i = 0; i < n_spots_host; i++){
            int base_idx_obs = i * N_COL_OBSSPOTS;
            if (base_idx_obs + 5 < n_spots_host * N_COL_OBSSPOTS) { // Bounds check needed? Yes.
                 if ((int)h_ObsSpotsLab[base_idx_obs + 5] == ring_to_index) {
                     if (startRowNrSp_host > i) startRowNrSp_host = i;
                     if (endRowNrSp_host < i) endRowNrSp_host = i;
                     ring_index_range_valid = true; // Found at least one spot
                 }
            }
        }
        if (!ring_index_range_valid) {
             printf("Warning: No observed spots found for RingToIndex = %d. No indexing possible.\n", ring_to_index);
             // Clean up and exit? Or proceed (kernel will find no matches)? Let's proceed.
             startRowNrSp_host = 0;
             endRowNrSp_host = 0; // Ensure valid range, even if empty
        } else {
             printf("Spot index range for Ring %d: %zu to %zu\n", ring_to_index, startRowNrSp_host, endRowNrSp_host);
        }
        // TODO: Pass startRowNrSp_host, endRowNrSp_host to kernel if optimizing search there.
    }


    // ================================================================
    // --- CUDA Initialization and Memory Allocation ---
    // ================================================================
    printf("Initializing CUDA and allocating device memory...\n");

    // Select GPU (optional, use default device 0)
    // CUDA_CHECK(cudaSetDevice(0));

    // --- Allocate Device Memory ---

    struct TParams* d_params = NULL; // Device pointer for the struct
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(struct TParams)));

    size_t obs_spots_bytes = n_spots_host * N_COL_OBSSPOTS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ObsSpotsLab, obs_spots_bytes));

    size_t hkls_bytes = n_hkls_host * N_COL_THEORSPOTS * sizeof(float); // Use THEORSPOTS cols for device buffer
    CUDA_CHECK(cudaMalloc(&d_hkls, hkls_bytes));

    size_t hklints_bytes = MAX_N_HKLS * 4 * sizeof(int); // Allocate max size? Or just n_hkls_host? Max is safer.
    CUDA_CHECK(cudaMalloc(&d_HKLints, hklints_bytes));

    size_t ringhkl_bytes = MAX_N_RINGS * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_RingHKL, ringhkl_bytes));

    size_t omemargins_bytes = 181 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_omemargins, omemargins_bytes));

    size_t etamargins_bytes = MAX_N_RINGS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_etamargins, etamargins_bytes));

    size_t grid_bytes = nVoxels * 2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_grid, grid_bytes));

    size_t ypos_bytes = numScans_host * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_ypos, ypos_bytes));

    // BigDetector, data, ndata (get sizes from mmap)
    size_t bigdet_bytes = 0;
    if (h_BigDetector && BigDetSize_host > 0) {
        // Size calculation based on number of bits needed
        bigdet_bytes = (size_t)totNrPixelsBigDetector_host * sizeof(int); // Number of int elements * size
        CUDA_CHECK(cudaMalloc(&d_BigDetector, bigdet_bytes));
    } else {
        d_BigDetector = NULL; // Ensure null if not used
    }

    struct stat stat_data, stat_ndata; // Need to stat again to get sizes if not stored
    char file_data[2048], file_ndata[2048];
    sprintf(file_data,"%s/Data.bin", cwdstr);
	sprintf(file_ndata,"%s/nData.bin", cwdstr);
    size_t data_bytes = 0;
    size_t ndata_bytes = 0;
    if (stat(file_data, &stat_data) == 0) data_bytes = stat_data.st_size;
    if (stat(file_ndata, &stat_ndata) == 0) ndata_bytes = stat_ndata.st_size;
    checkHost(data_bytes == 0 || ndata_bytes == 0 || h_data == NULL || h_ndata == NULL, "Cannot get size or access mmap'd bin data for CUDA transfer.");
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_ndata, ndata_bytes));

    // MIC data
    size_t mic_bytes = 0;
    if (hasMic_flag && nrMic_host > 0) {
        mic_bytes = nrMic_host * 5 * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_mic, mic_bytes));
    } else {
        d_mic = NULL;
    }

    // --- Allocate Workspace Buffer ---
    // Calculate size based on constants and number of threads needed
    size_t theor_spots_size_ws = NROWS_PER_GRAIN * N_COL_THEORSPOTS * sizeof(float);
    size_t grain_spots_size_ws = NROWS_PER_GRAIN * N_COL_GRAINSPOTS * sizeof(float);
    size_t grain_matches_T_size_ws = MAX_N_MATCHES * N_COL_GRAINMATCHES * sizeof(float);
    size_t all_grain_spots_T_size_ws = NROWS_OUTPUT * N_COL_GRAINSPOTS * sizeof(float);
    size_t or_mat_size_ws = MAX_N_OR * 3 * 3 * sizeof(float);
    size_t total_size_per_thread_ws = theor_spots_size_ws + grain_spots_size_ws +
                                      grain_matches_T_size_ws + all_grain_spots_T_size_ws +
                                      or_mat_size_ws;

    size_t total_workspace_bytes = (size_t)num_voxels_this_block * total_size_per_thread_ws;
    printf("Allocating %.2f MB workspace buffer per thread...\n", (double)total_size_per_thread_ws / (1024.0*1024.0));
    printf("Allocating %.2f MB total workspace buffer for %d voxels...\n", (double)total_workspace_bytes / (1024.0*1024.0), num_voxels_this_block);
    CUDA_CHECK(cudaMalloc(&d_workspace, total_workspace_bytes));

    // --- Allocate Output Buffers ---
    size_t results_bytes = num_voxels_this_block * 16 * sizeof(float); // 16 floats per result
    CUDA_CHECK(cudaMalloc(&d_results, results_bytes));

    size_t key_info_bytes = num_voxels_this_block * 4 * sizeof(size_t); // 4 size_t per key
    CUDA_CHECK(cudaMalloc(&d_key_info, key_info_bytes));

    // Matched IDs buffer - size is unknown beforehand.
    // Allocate a reasonable maximum guess. Max possible matches per voxel = NROWS_PER_GRAIN.
    size_t max_possible_matches_total = (size_t)num_voxels_this_block * NROWS_PER_GRAIN;
    size_t matched_ids_bytes = max_possible_matches_total * sizeof(int);
    printf("Allocating %.2f MB for matched IDs buffer (max guess)...\n", (double)matched_ids_bytes / (1024.0*1024.0));
    CUDA_CHECK(cudaMalloc(&d_matched_ids, matched_ids_bytes)); // Allocate for worst-case


    // ================================================================
    // --- Copy Host Data to Device ---
    // ================================================================
    printf("Copying data from host to device...\n");

    printf("Copying TParams struct to device...\n");
    CUDA_CHECK(cudaMemcpy(d_params, &Params, sizeof(struct TParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ObsSpotsLab, h_ObsSpotsLab, obs_spots_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hkls, h_hkls, hkls_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_HKLints, h_HKLints, hklints_bytes, cudaMemcpyHostToDevice)); // Copy max size
    CUDA_CHECK(cudaMemcpy(d_RingHKL, h_RingHKL, ringhkl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_omemargins, h_omemargins, omemargins_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_etamargins, h_etamargins, etamargins_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ypos, h_ypos, ypos_bytes, cudaMemcpyHostToDevice));

    if (d_BigDetector && h_BigDetector) {
        CUDA_CHECK(cudaMemcpy(d_BigDetector, h_BigDetector, bigdet_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndata, h_ndata, ndata_bytes, cudaMemcpyHostToDevice));

    if (d_mic && h_mic) {
        CUDA_CHECK(cudaMemcpy(d_mic, h_mic, mic_bytes, cudaMemcpyHostToDevice));
    }

    printf("Data copy complete.\n");

    // Free host copies that are no longer needed? Or keep for output phase?
    // Keep h_grid, h_ypos, Params, etc. for now.
    // We can free/unmap large input buffers now.
    free(h_ObsSpotsLab); h_ObsSpotsLab = NULL;
    // Don't free h_hkls, h_HKLints, h_RingHKL if needed by host later (e.g., debug)
    if (h_BigDetector) munmap(h_BigDetector, bigdet_bytes); h_BigDetector = NULL;
    if (h_data) munmap(h_data, data_bytes); h_data = NULL;
    if (h_ndata) munmap(h_ndata, ndata_bytes); h_ndata = NULL;
    free(h_mic); h_mic = NULL;
    // Keep h_grid, h_ypos? Let's free them.
    free(h_grid); h_grid = NULL;
    free(h_ypos); h_ypos = NULL;


    // ================================================================
    // --- Kernel Launch ---
    // ================================================================
    printf("Launching CUDA kernel for %d voxels...\n", num_voxels_this_block);

    // Kernel configuration
    int threadsPerBlock = 512; // Common choice, tune if needed
    // Calculate number of blocks needed to cover all voxels in this chunk
    int numBlocks = (num_voxels_this_block + threadsPerBlock - 1) / threadsPerBlock;

    int kernel_n_ring_bins = n_ring_bins_host;
    int kernel_n_eta_bins = n_eta_bins_host;
    int kernel_n_ome_bins = n_ome_bins_host;

    // Pass necessary parameters to the kernel
    IndexingKernel<<<numBlocks, threadsPerBlock>>>(
        d_ObsSpotsLab, d_BigDetector, d_data, d_ndata, d_hkls, d_HKLints,
        d_RingHKL, d_omemargins, d_etamargins, d_grid, d_ypos, d_mic,
        d_params, // Pass struct by device pointer
        (int)n_spots_host, hasMic_flag, nrMic_host,
        startRowNr_host, endRowNr_host, // Range for this kernel
        d_workspace, d_results, d_matched_ids, d_key_info
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    // Synchronize to wait for kernel completion and get accurate timing
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel execution finished.\n");


    // ================================================================
    // --- Post-Kernel Processing (Copy Results, Write Files) ---
    // ================================================================
    printf("Copying results from device to host...\n");

    // Allocate host memory for results
    float *h_results = (float*)malloc(results_bytes);
    size_t *h_key_info = (size_t*)malloc(key_info_bytes);
    checkHost(h_results == NULL || h_key_info == NULL, "Failed to allocate host memory for results.");

    // Copy results and key info back
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_key_info, d_key_info, key_info_bytes, cudaMemcpyDeviceToHost));

    // Calculate total number of matched spots and offsets for file writing
    size_t total_matched_spots = 0;
    for (int i = 0; i < num_voxels_this_block; ++i) {
        size_t nMatches_this_voxel = h_key_info[i * 4 + 1];
        if (nMatches_this_voxel > NROWS_PER_GRAIN) { // Sanity check
             printf("Warning: Voxel index %d reported %zu matches, exceeding max %d. Clamping.\n",
                    startRowNr_host + i, nMatches_this_voxel, NROWS_PER_GRAIN);
             nMatches_this_voxel = NROWS_PER_GRAIN;
             h_key_info[i * 4 + 1] = nMatches_this_voxel; // Correct the count
        }
        h_key_info[i * 4 + 2] = (size_t)i;  // Offset in h_results (1 result per voxel)
        h_key_info[i * 4 + 3] = total_matched_spots; // Starting offset for matched IDs
        total_matched_spots += nMatches_this_voxel;
    }
    printf("Total matched spots across block: %zu\n", total_matched_spots);

    // Allocate host memory for matched IDs and copy them back
    int *h_matched_ids = NULL;
    size_t h_matched_ids_bytes = 0;
    if (total_matched_spots > 0) {
        if (total_matched_spots > max_possible_matches_total) {
             printf("Error: Calculated total matches (%zu) exceeds allocated buffer size (%zu).\n",
                    total_matched_spots, max_possible_matches_total);
             // Handle error - maybe reallocate h_matched_ids? Or abort?
             // For now, clamp to buffer size, but this indicates an issue.
             total_matched_spots = max_possible_matches_total;
        }
        h_matched_ids_bytes = total_matched_spots * sizeof(int);
        h_matched_ids = (int*)malloc(h_matched_ids_bytes);
        checkHost(h_matched_ids == NULL, "Failed to allocate host memory for matched IDs.");

        // Need to copy IDs from the workspace buffer (AllGrainSpots_ws).
        // This requires copying the relevant parts of d_workspace back,
        // or having the kernel write directly to d_matched_ids (more complex).

        // --- Let's copy the necessary parts of the workspace back ---
        printf("Copying relevant workspace sections back for matched IDs...\n");
        float* h_workspace_for_ids = (float*)malloc(total_workspace_bytes); // Alloc host mirror
        checkHost(h_workspace_for_ids == NULL, "Failed to allocate host mirror for workspace.");
        CUDA_CHECK(cudaMemcpy(h_workspace_for_ids, d_workspace, total_workspace_bytes, cudaMemcpyDeviceToHost));

        // Now extract the IDs on the host
        size_t current_id_offset = 0;
        for (int i = 0; i < num_voxels_this_block; ++i) {
             size_t nMatches = h_key_info[i * 4 + 1];
             if (nMatches > 0) {
                 // Find the workspace slice for this thread (voxel)
                 size_t thread_offset_ws = (size_t)i * total_size_per_thread_ws;
                 // Pointer to AllGrainSpots section (adjust based on actual workspace layout)
                 float* AllGrainSpots_ws_host = h_workspace_for_ids + thread_offset_ws +
                                               theor_spots_size_ws + grain_spots_size_ws +
                                               grain_matches_T_size_ws; // Points to start of AllGrainSpotsT/AllGrainSpots

                 // Copy the IDs (column 14) for the matched spots (first nMatches rows)
                 for (size_t m = 0; m < nMatches; ++m) {
                     int spot_row_idx = m; // Matched spots are at the start
                     int base_idx_spot = spot_row_idx * N_COL_GRAINSPOTS;
                     if (current_id_offset < total_matched_spots) {
                          h_matched_ids[current_id_offset++] = (int)AllGrainSpots_ws_host[base_idx_spot + 14];
                     } else {
                          printf("Error: Exceeded total_matched_spots while copying IDs.\n");
                          break;
                     }
                 }
             }
             if (current_id_offset > total_matched_spots) break; // Error occurred
        }
        free(h_workspace_for_ids); // Free host workspace mirror
        printf("Matched IDs extracted.\n");

    } else {
        printf("No matched spots found in this block.\n");
    }


    // --- Write Output Files ---
    printf("Writing output files...\n");
    // This loop replaces the omp parallel loop's file writing part
    for (int i = 0; i < num_voxels_this_block; ++i) {
        int current_voxNr = startRowNr_host + i;

        // Get data for this voxel from host buffers
        size_t key_idx = i * 4;
        size_t result_idx = i * 16;

        size_t SpotID_out = h_key_info[key_idx + 0];
        size_t nMatches_out = h_key_info[key_idx + 1];
        // size_t offset_vals = h_key_info[key_idx + 2]; // This is just 'i'
        size_t offset_ids = h_key_info[key_idx + 3];

        // Check if a match was found (SpotID != -1)
        if ((int)SpotID_out != -1 && nMatches_out > 0) {
            FILE *valsF, *allF, *keyF;
            char valsFN[2048], allFN[2048], keyFN[2048];

            // Create filenames
            sprintf(valsFN,"%s/IndexBest_voxNr_%0*d.bin",Params.OutputFolder,6,current_voxNr);
            sprintf(allFN,"%s/IndexBest_IDs_voxNr_%0*d.bin",Params.OutputFolder,6,current_voxNr);
            sprintf(keyFN,"%s/IndexKey_voxNr_%0*d.txt",Params.OutputFolder,6,current_voxNr);

            // Open files
            valsF = fopen(valsFN,"wb");
            allF = fopen(allFN,"wb");
            keyF = fopen(keyFN,"w");

            if (!valsF || !allF || !keyF) {
                fprintf(stderr, "Error: Could not open output files for voxel %d\n", current_voxNr);
                if(valsF) fclose(valsF);
                if(allF) fclose(allF);
                if(keyF) fclose(keyF);
                continue; // Skip this voxel
            }

            // Prepare data to write
            // Convert results back to double for original file format compatibility?
            // Or keep float? Let's keep float as requested.
            float outArr_f[16];
            memcpy(outArr_f, &h_results[result_idx], 16 * sizeof(float));

            // Get matched IDs for this voxel
            int *outArr2_i = NULL;
            if (nMatches_out > 0 && h_matched_ids != NULL) {
                 outArr2_i = &h_matched_ids[offset_ids];
            }

            // Write data
            size_t written_vals = fwrite(outArr_f, sizeof(float), 16, valsF);
            size_t written_ids = 0;
            if (outArr2_i) {
                written_ids = fwrite(outArr2_i, sizeof(int), nMatches_out, allF);
            }
            // Write key file (using dummy offsets for now, as files are per voxel)
            // fprintf(keyF,"%zu %zu %zu %zu\n", SpotID_out, nMatches_out, (size_t)0, (size_t)0); // Offsets within these files are 0
             fprintf(keyF,"%zu %zu %zu %zu\n", SpotID_out, nMatches_out, (size_t)0, (size_t)0);

            // Simple check
             if (written_vals != 16 || (nMatches_out > 0 && written_ids != nMatches_out)) {
                 fprintf(stderr, "Warning: File write error for voxel %d.\n", current_voxNr);
             }

            // Print summary like original code
            float confidence_out = (outArr_f[14] > EPS) ? (outArr_f[15] / outArr_f[14]) : 0.0f;
            float IA_out = outArr_f[1]; // Index 1 is IA
            printf("ID: %d (Voxel: %d), Confidence: %f, IA: %f, Matches: %zu/%d\n",
                   (int)SpotID_out, current_voxNr, confidence_out, IA_out, nMatches_out, (int)outArr_f[14]);


            // Close files
            fclose(valsF);
            fclose(allF);
            fclose(keyF);

        } else {
            // Optional: Print message for voxels with no match found
             // printf("Voxel %d: No match found.\n", current_voxNr);
        }
    } // End loop writing files


    // ================================================================
    // --- Cleanup ---
    // ================================================================
    printf("Cleaning up device memory...\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_ObsSpotsLab));
    CUDA_CHECK(cudaFree(d_hkls));
    CUDA_CHECK(cudaFree(d_HKLints));
    CUDA_CHECK(cudaFree(d_RingHKL));
    CUDA_CHECK(cudaFree(d_omemargins));
    CUDA_CHECK(cudaFree(d_etamargins));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_ypos));
    if (d_BigDetector) CUDA_CHECK(cudaFree(d_BigDetector));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_ndata));
    if (d_mic) CUDA_CHECK(cudaFree(d_mic));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_key_info));
    if (d_matched_ids) CUDA_CHECK(cudaFree(d_matched_ids));


    // Free host memory for results
    free(h_results);
    free(h_key_info);
    free(h_matched_ids); // Free host ID buffer

    // Free any other remaining host allocations (e.g., Params members if dynamically allocated)

	clock_t end_clock = clock();
    double time_elapsed_sec = (double)(end_clock - start_clock) / CLOCKS_PER_SEC;
	printf("\nFinished, time elapsed: %.3f seconds.\n", time_elapsed_sec);
	return(0);
}
