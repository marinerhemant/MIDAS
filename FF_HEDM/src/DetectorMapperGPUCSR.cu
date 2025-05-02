//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// Optimized for float precision, CSR output, robustness
//
// Example compile command (adjust paths and architecture flags):
/*
~/opt/midascuda/cuda/bin/nvcc src/DetectorMapperGPUCSR.cu -o bin/DetectorMapperGPUCSR \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_90,code=sm_90 \
  -Xcompiler -g \
  -I/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/include \
  -L/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/lib \
  -O3 -lnlopt -lz -ldl -lm -lpthread

*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h> // For bool, true, false

// CUDA specific includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define deg2radf 0.0174532925199433f
#define rad2degf 57.2957795130823f
#define EPS_F 1e-5f      // Epsilon for float comparisons
#define M_PI_F 3.14159265358979323846f // PI for float

// Max bins allowed for __constant__ memory (adjust if necessary)
#define MAX_BINS 4096 // Increase if needed and supported by GPU

// --- CUDA Error Checking Wrapper ---
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort_flag)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort_flag)
        {
             exit(code);
        }
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

// --- Device/Host Helper Functions (Using float) ---

__host__ __device__ static inline
int BETWEEN_F(float val, float min, float max)
{
    volatile float v = val;
    volatile float mn = min;
    volatile float mx = max;
	return ((v - EPS_F <= mx && v + EPS_F >= mn) ? 1 : 0 );
}

__host__ __device__ static inline void MatrixMultF(const float m[3][3], const float v[3], float r[3])
{
    r[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2];
    r[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2];
    r[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2];
}

__host__ __device__ static inline void MatrixMultF33F( const float m[3][3], const float n[3][3], float res[3][3])
{
    float t[3][3];
    int r;
    for(r=0; r<3; r++)
    {
        t[r][0]=m[r][0]*n[0][0]+m[r][1]*n[1][0]+m[r][2]*n[2][0];
        t[r][1]=m[r][0]*n[0][1]+m[r][1]*n[1][1]+m[r][2]*n[2][1];
        t[r][2]=m[r][0]*n[0][2]+m[r][1]*n[1][2]+m[r][2]*n[2][2];
    }
    memcpy(res, t, sizeof(t)); // Use memcpy for block copy
}

__host__ __device__ static inline float CalcEtaAngleF(float y, float z)
{
    return rad2degf * atan2f(-y, z);
}

__host__ __device__ static inline void REta4MYZF( float Y, float Z, float Ycen, float Zcen, const float TRs[3][3], float Lsd, float RhoD, float p0, float p1, float p2, float p3, float n0, float n1, float n2, float px, float RetVals[2])
{
	float Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT, Rt;
	Yc = (-Y + Ycen)*px;
    Zc = ( Z - Zcen)*px;
	ABC[0] = 0.0f;
    ABC[1] = Yc;
    ABC[2] = Zc;
	MatrixMultF(TRs,ABC,ABCPr);
	XYZ[0] = Lsd+ABCPr[0];
    XYZ[1] = ABCPr[1];
    XYZ[2] = ABCPr[2];
    float sq_sum = XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2];
    if (fabsf(XYZ[0]) < EPS_F || sq_sum < 0.0f)
    {
        RetVals[0] = 0.0f;
        RetVals[1] = 0.0f;
        return;
    }
	Rad = (Lsd/(XYZ[0]))*(sqrtf(sq_sum));
    Eta = CalcEtaAngleF(XYZ[1],XYZ[2]);
	RNorm = (RhoD > EPS_F) ? (Rad / RhoD) : 0.0f;
    EtaT = 90.0f - Eta;
	DistortFunc = (p0*(powf(fmaxf(0.0f, RNorm),n0))*(cosf(deg2radf*(2.0f*EtaT)))) +
                  (p1*(powf(fmaxf(0.0f, RNorm),n1))*(cosf(deg2radf*(4.0f*EtaT+p3)))) +
                  (p2*(powf(fmaxf(0.0f, RNorm),n2))) + 1.0f;
	Rt = (px > EPS_F) ? (Rad * DistortFunc / px) : 0.0f;
	RetVals[0] = Eta;
    RetVals[1] = Rt;
}

__host__ __device__ static inline void YZ4mREtaF(float R, float Eta, float YZ[2])
{
    YZ[0] = -R*sinf(Eta*deg2radf);
    YZ[1] = R*cosf(Eta*deg2radf);
}

// --- Constants ---
__constant__ float const_dy[2];
__constant__ float const_dz[2];
__constant__ float const_PosMatrix[4][2];

// --- Structs ---
typedef struct { int r_bin; int eta_bin; int y; int z; float frac; } GpuOutput;
typedef struct { float x; float y; } Point_device_f;
typedef struct { float x; float y; } Point_host_f;
typedef struct { int y; int z; int r_bin; int eta_bin; float frac; size_t pixel_idx; size_t bin_idx; } HostOutputEntry;


// --- Device Polygon/Unique Functions ---
__device__ int cmpfunc_device_f(const Point_device_f *a, const Point_device_f *b, const Point_device_f *center)
{
    float ax = a->x - center->x;
    float ay = a->y - center->y;
    float bx = b->x - center->x;
    float by = b->y - center->y;
    if (ax >= 0.0f && bx < 0.0f) return 1;
    if (ax < 0.0f && bx >= 0.0f) return -1;
    if (fabsf(ax) < EPS_F && fabsf(bx) < EPS_F)
    {
        if (ay >= 0.0f || by >= 0.0f)
        {
            return ay > by ? 1 : -1;
        }
        return by > ay ? 1 : -1;
    }
    float det = ax * by - bx * ay;
    if (fabsf(det) < EPS_F*EPS_F)
    {
        float d1_sq = ax * ax + ay * ay;
        float d2_sq = bx * bx + by * by;
        return d1_sq > d2_sq ? 1 : -1;
    }
    return (det < 0.0f) ? 1 : -1;
}

__device__ void insertionSort_device_f(Point_device_f *points, int n, const Point_device_f *center)
{
    int i, j;
    Point_device_f key;
    for (i = 1; i < n; i++)
    {
        key = points[i];
        j = i - 1;
        while (j >= 0 && cmpfunc_device_f(&points[j], &key, center) > 0)
        {
            points[j + 1] = points[j];
            j = j - 1;
        }
        points[j + 1] = key;
    }
}

__device__ static inline float CalcAreaPolygon_device_f(float Edges[][2], int nEdges)
{
    if (nEdges < 3)
    {
        return 0.0f;
    }
    Point_device_f MyData[50];
    Point_device_f center = {0.0f, 0.0f};
    int i;
	for (i=0; i<nEdges; i++)
    {
        center.x += Edges[i][0];
        center.y += Edges[i][1];
        MyData[i].x = Edges[i][0];
        MyData[i].y = Edges[i][1];
    }
	center.x /= (float)nEdges;
    center.y /= (float)nEdges;
    insertionSort_device_f(MyData, nEdges, &center);
	float Area=0.0f;
    int next;
	for (i=0; i<nEdges; i++)
    {
        next = (i + 1) % nEdges;
        Area += (MyData[i].x * MyData[next].y) - (MyData[next].x * MyData[i].y);
    }
	return fabsf(Area / 2.0f);
}

__device__ bool isEtaInBinF(float etaPoint, float binLow, float binHigh)
{
     float etaCheck = etaPoint;
     if (binHigh - binLow >= 360.0f - 1.0f)
     {
         return true;
     }
     if (binLow < -180.0f + EPS_F && binHigh > 180.0f - EPS_F)
     {
        if (etaPoint > binHigh - 180.0f)
        {
            etaCheck -= 360.0f;
        }
        else if (etaPoint < binLow + 180.0f)
        {
            etaCheck += 360.0f;
        }
     }
     else
     {
         float binCenter = (binLow + binHigh) * 0.5f;
         while (etaCheck < binCenter - 180.0f)
         {
             etaCheck += 360.0f;
         }
         while (etaCheck > binCenter + 180.0f)
         {
             etaCheck -= 360.0f;
         }
     }
     return BETWEEN_F(etaCheck, binLow, binHigh);
}

__device__ bool checkEtaOverlapF(float etaMi, float etaMa, float binLow, float binHigh)
{
    float pMi = etaMi;
    float pMa = etaMa;
    if (pMa < pMi)
    {
        pMa += 360.0f;
    }
    float bLo = binLow;
    float bHi = binHigh;
    int shift;
    for (shift = -1; shift <= 1; ++shift)
    {
        float shifted_bLo = bLo + shift * 360.0f;
        float shifted_bHi = bHi + shift * 360.0f;
        if (fmaxf(pMi, shifted_bLo) <= fminf(pMa, shifted_bHi) + EPS_F)
        {
            return true;
        }
    }
    return false;
}

__device__ static inline int FindUniques_device_f (float EdgesIn[][2], float EdgesOut[][2], int nEdgesIn, float RMin, float RMax, float EtaMin, float EtaMax)
{
	int i,j, k, nEdgesOut=0;
    int duplicate;
    float LenSq, RT, ET;
	for (i=0; i<nEdgesIn; i++)
    {
        duplicate = 0;
		for (j=i+1; j<nEdgesIn; j++)
        {
            LenSq = (EdgesIn[i][0]-EdgesIn[j][0])*(EdgesIn[i][0]-EdgesIn[j][0]) + (EdgesIn[i][1]-EdgesIn[j][1])*(EdgesIn[i][1]-EdgesIn[j][1]);
            if (LenSq < EPS_F*EPS_F)
            {
                duplicate = 1;
                break;
            }
        }
        if (duplicate)
        {
            continue;
        }
		RT = sqrtf(EdgesIn[i][0]*EdgesIn[i][0] + EdgesIn[i][1]*EdgesIn[i][1]);
        ET = CalcEtaAngleF(EdgesIn[i][0],EdgesIn[i][1]);
		if (!BETWEEN_F(RT,RMin,RMax) || !isEtaInBinF(ET, EtaMin, EtaMax))
        {
            duplicate = 1;
        }
		if (duplicate == 0)
        {
            int already_added = 0;
            for(k=0; k < nEdgesOut; ++k)
            {
                LenSq = (EdgesOut[k][0]-EdgesIn[i][0])*(EdgesOut[k][0]-EdgesIn[i][0]) + (EdgesOut[k][1]-EdgesIn[i][1])*(EdgesOut[k][1]-EdgesIn[i][1]);
                if (LenSq < EPS_F*EPS_F)
                {
                    already_added = 1;
                    break;
                }
            }
            if (!already_added)
            {
                EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
                EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
                nEdgesOut++;
                if (nEdgesOut >= 50)
                {
                    break;
                }
            }
		}
	}
	return nEdgesOut;
}

// --- CUDA Kernel ---
__global__ void mapperKernelF(
    float tx, float ty, float tz, const float TRs_arg[3][3],
    int NrPixelsY, int NrPixelsZ, float pxY, float pxZ, float Ycen, float Zcen,
    float Lsd, float RhoD, float p0, float p1, float p2, float p3,
    int nRBins, int nEtaBins,
    const float* __restrict__ etaBinsLow_d,
    const float* __restrict__ etaBinsHigh_d,
    const float* __restrict__ rBinsLow_d,
    const float* __restrict__ rBinsHigh_d,
    const float* __restrict__ distortionMapY_d, const float* __restrict__ distortionMapZ_d,
    GpuOutput* outputBuffer, unsigned int* outputCounter, unsigned int maxOutputSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NrPixelsY || j >= NrPixelsZ)
    {
        return;
    }

    float n0=2.0f, n1=4.0f, n2=2.0f;
    float RetVals[2], RetVals2[2];
    float Y, Z, Eta, Rt;
	float EtaMi, EtaMa, RMi, RMa;
    // These might need adjustment if 500 is not enough for some edge cases
    int RChosen[500];
    int EtaChosen[500];
    int nrRChosen, nrEtaChosen;
    float YZ[2];
    // These might need adjustment if 50 is not enough for some edge cases
    float Edges[50][2];
	float EdgesOut[50][2];
    int nEdges;
    float RMin_bin, RMax_bin, EtaMin_bin, EtaMax_bin;
	float yMin, yMax, zMin, zMax;
    float boxEdge[4][2];
    float Area;
	float RThis, EtaThis;
    float yTemp, zTemp;
    long long int testPos = (long long int)j * NrPixelsY + i;
    float y_center_pix = (float)i;
    float z_center_pix = (float)j;
    float ypr = y_center_pix + distortionMapY_d[testPos];
	float zpr = z_center_pix + distortionMapZ_d[testPos];
    EtaMi = 1800.0f;
    EtaMa = -1800.0f;
    RMi = 1E12f;
    RMa = -1E12f;
    int k, l, m; // Loop variables

    // Calculate pixel corner boundaries in (R, Eta) space
    for (k = 0; k < 2; k++)
    {
        for (l = 0; l < 2; l++)
        {
            Y = ypr + const_dy[k]; // Use constant memory
            Z = zpr + const_dz[l]; // Use constant memory
            REta4MYZF(Y, Z, Ycen, Zcen, TRs_arg, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
            Eta = RetVals[0];
            Rt = RetVals[1];
            if (Eta < EtaMi) EtaMi = Eta;
            if (Eta > EtaMa) EtaMa = Eta;
            if (Rt < RMi) RMi = Rt;
            if (Rt > RMa) RMa = Rt;
        }
    }
    if (RMa < RMi || EtaMa < EtaMi) // Check if pixel maps to a valid range
    {
        return;
    }

    // Get center point in (R, Eta) and transformed (Y, Z) space
    REta4MYZF(ypr, zpr, Ycen, Zcen, TRs_arg, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
    Eta = RetVals[0]; // Center Eta
    Rt = RetVals[1];  // Center R
    YZ4mREtaF(Rt,Eta,RetVals2); // Convert center (R, Eta) back to ideal YZ
    YZ[0] = RetVals2[0];
    YZ[1] = RetVals2[1];

    // Find potentially overlapping R bins
    nrRChosen = 0;
    for (k=0; k<nRBins; k++)
    {
        // Use rBinsLow_d and rBinsHigh_d passed as arguments
        if (RMa >= rBinsLow_d[k] - EPS_F && RMi <= rBinsHigh_d[k] + EPS_F)
        {
            if (nrRChosen < 500) RChosen[nrRChosen++] = k; else break; // Avoid buffer overflow
        }
    }

    // Find potentially overlapping Eta bins
	nrEtaChosen = 0;
	for (k=0; k<nEtaBins; k++)
    {
        // Use etaBinsLow_d and etaBinsHigh_d passed as arguments
        if (checkEtaOverlapF(EtaMi, EtaMa, etaBinsLow_d[k], etaBinsHigh_d[k]))
        {
            if (nrEtaChosen < 500) EtaChosen[nrEtaChosen++] = k; else break; // Avoid buffer overflow
        }
    }

    // Define the bounding box of the pixel in the ideal (Y, Z) space
    yMin = YZ[0] - 0.5f;
    yMax = YZ[0] + 0.5f;
    zMin = YZ[1] - 0.5f;
    zMax = YZ[1] + 0.5f;

    // Iterate over potentially overlapping bins
	for (k=0; k<nrRChosen; k++)
    {
        int r_idx = RChosen[k];
        // Get bin boundaries from kernel arguments
        RMin_bin = rBinsLow_d[r_idx];
        RMax_bin = rBinsHigh_d[r_idx];

		for (l=0; l<nrEtaChosen; l++)
        {
            int eta_idx = EtaChosen[l];
            // Get bin boundaries from kernel arguments
            EtaMin_bin = etaBinsLow_d[eta_idx];
            EtaMax_bin = etaBinsHigh_d[eta_idx];

            // --- Calculate Intersection Polygon ---
            // (This complex intersection logic remains the same, just uses RMin_bin etc.)
            nEdges = 0;
            // 1. Check pixel corners against bin boundaries
            for (m=0; m<4; m++)
            {
                // Use constant memory
                float cornerY = YZ[0] + const_PosMatrix[m][0];
                float cornerZ = YZ[1] + const_PosMatrix[m][1];
                RThis = sqrtf(cornerY*cornerY + cornerZ*cornerZ);
                EtaThis = CalcEtaAngleF(cornerY, cornerZ); // Note: YZ4mREta uses atan2(-y, z), CalcEtaAngle uses atan2(y, z) - ensure consistency if needed
                if (BETWEEN_F(RThis, RMin_bin, RMax_bin) && isEtaInBinF(EtaThis, EtaMin_bin, EtaMax_bin))
                {
                    if (nEdges < 50) // Avoid buffer overflow
                    {
                         Edges[nEdges][0] = cornerY; Edges[nEdges][1] = cornerZ; nEdges++;
                    } else break;
                }
            }
             if (nEdges >= 50) continue; // Skip if already full

            // 2. Check bin corners against pixel boundaries
            YZ4mREtaF(RMin_bin, EtaMin_bin, boxEdge[0]); YZ4mREtaF(RMin_bin, EtaMax_bin, boxEdge[1]); YZ4mREtaF(RMax_bin, EtaMin_bin, boxEdge[2]); YZ4mREtaF(RMax_bin, EtaMax_bin, boxEdge[3]);
            for (m=0; m<4; m++)
            {
                if (BETWEEN_F(boxEdge[m][0], yMin, yMax) && BETWEEN_F(boxEdge[m][1], zMin, zMax))
                {
                    if (nEdges < 50) // Avoid buffer overflow
                    {
                        Edges[nEdges][0] = boxEdge[m][0]; Edges[nEdges][1] = boxEdge[m][1]; nEdges++;
                    } else break;
                }
            }
            if (nEdges >= 50) continue; // Skip if already full

            // 3. Check intersections between pixel edges and bin boundaries (arcs and lines)
            //    (This complex part calculating intersections between box and annulus sector remains the same)
            float RMin_sq = RMin_bin * RMin_bin;
            float RMax_sq = RMax_bin * RMax_bin;
            float yMin_sq = yMin * yMin; // Precompute squares
            float yMax_sq = yMax * yMax;
            float zMin_sq = zMin * zMin;
            float zMax_sq = zMax * zMax;

            // Intersections R=const with y=const
            if (RMin_sq >= yMin_sq - EPS_F) { zTemp = sqrtf(fmaxf(0.0f, RMin_sq - yMin_sq)); if (BETWEEN_F(zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMin; Edges[nEdges][1]= zTemp; nEdges++;} else continue;} if (BETWEEN_F(-zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMin; Edges[nEdges][1]=-zTemp; nEdges++;} else continue;} }
            if (RMin_sq >= yMax_sq - EPS_F) { zTemp = sqrtf(fmaxf(0.0f, RMin_sq - yMax_sq)); if (BETWEEN_F(zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMax; Edges[nEdges][1]= zTemp; nEdges++;} else continue;} if (BETWEEN_F(-zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMax; Edges[nEdges][1]=-zTemp; nEdges++;} else continue;} }
            if (RMax_sq >= yMin_sq - EPS_F) { zTemp = sqrtf(fmaxf(0.0f, RMax_sq - yMin_sq)); if (BETWEEN_F(zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMin; Edges[nEdges][1]= zTemp; nEdges++;} else continue;} if (BETWEEN_F(-zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMin; Edges[nEdges][1]=-zTemp; nEdges++;} else continue;} }
            if (RMax_sq >= yMax_sq - EPS_F) { zTemp = sqrtf(fmaxf(0.0f, RMax_sq - yMax_sq)); if (BETWEEN_F(zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMax; Edges[nEdges][1]= zTemp; nEdges++;} else continue;} if (BETWEEN_F(-zTemp, zMin, zMax)) { if(nEdges<50) {Edges[nEdges][0]=yMax; Edges[nEdges][1]=-zTemp; nEdges++;} else continue;} }
            // Intersections R=const with z=const
            if (RMin_sq >= zMin_sq - EPS_F) { yTemp = sqrtf(fmaxf(0.0f, RMin_sq - zMin_sq)); if (BETWEEN_F(yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]= yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if (BETWEEN_F(-yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]=-yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} }
            if (RMin_sq >= zMax_sq - EPS_F) { yTemp = sqrtf(fmaxf(0.0f, RMin_sq - zMax_sq)); if (BETWEEN_F(yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]= yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} if (BETWEEN_F(-yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]=-yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} }
            if (RMax_sq >= zMin_sq - EPS_F) { yTemp = sqrtf(fmaxf(0.0f, RMax_sq - zMin_sq)); if (BETWEEN_F(yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]= yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if (BETWEEN_F(-yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]=-yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} }
            if (RMax_sq >= zMax_sq - EPS_F) { yTemp = sqrtf(fmaxf(0.0f, RMax_sq - zMax_sq)); if (BETWEEN_F(yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]= yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} if (BETWEEN_F(-yTemp, yMin, yMax)) { if(nEdges<50) {Edges[nEdges][0]=-yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} }

            // Intersections Eta=const with y=const and z=const
            float cosEtaMin = cosf(EtaMin_bin * deg2radf);
            float sinEtaMin = sinf(EtaMin_bin * deg2radf);
            float cosEtaMax = cosf(EtaMax_bin * deg2radf);
            float sinEtaMax = sinf(EtaMax_bin * deg2radf);
            // EtaMin line
            if (fabsf(sinEtaMin) > EPS_F) { // Avoid division by zero
                zTemp = -yMin * cosEtaMin / sinEtaMin; if(BETWEEN_F(zTemp, zMin, zMax)) { RThis = sqrtf(yMin*yMin + zTemp*zTemp); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=zTemp; nEdges++;} else continue;} }
                zTemp = -yMax * cosEtaMin / sinEtaMin; if(BETWEEN_F(zTemp, zMin, zMax)) { RThis = sqrtf(yMax*yMax + zTemp*zTemp); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=zTemp; nEdges++;} else continue;} }
            } else { // Line is z=0 (or close to it)
                 if(BETWEEN_F(0.0f, zMin, zMax) && cosEtaMin > 0) { if(BETWEEN_F(yMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=0.0f; nEdges++;} else continue;} if(BETWEEN_F(yMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=0.0f; nEdges++;} else continue;}}
                 if(BETWEEN_F(0.0f, zMin, zMax) && cosEtaMin < 0) { if(BETWEEN_F(-yMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=0.0f; nEdges++;} else continue;} if(BETWEEN_F(-yMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=0.0f; nEdges++;} else continue;}}
            }
            if (fabsf(cosEtaMin) > EPS_F) { // Avoid division by zero
                yTemp = -zMin * sinEtaMin / cosEtaMin; if(BETWEEN_F(yTemp, yMin, yMax)) { RThis = sqrtf(yTemp*yTemp + zMin*zMin); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} }
                yTemp = -zMax * sinEtaMin / cosEtaMin; if(BETWEEN_F(yTemp, yMin, yMax)) { RThis = sqrtf(yTemp*yTemp + zMax*zMax); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} }
            } else { // Line is y=0 (or close to it)
                 if(BETWEEN_F(0.0f, yMin, yMax) && sinEtaMin < 0) { if(BETWEEN_F(-zMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if(BETWEEN_F(-zMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMax; nEdges++;} else continue;}}
                 if(BETWEEN_F(0.0f, yMin, yMax) && sinEtaMin > 0) { if(BETWEEN_F( zMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if(BETWEEN_F( zMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMax; nEdges++;} else continue;}}
            }
             // EtaMax line (similar logic)
             if (fabsf(sinEtaMax) > EPS_F) {
                 zTemp = -yMin * cosEtaMax / sinEtaMax; if(BETWEEN_F(zTemp, zMin, zMax)) { RThis = sqrtf(yMin*yMin + zTemp*zTemp); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=zTemp; nEdges++;} else continue;} }
                 zTemp = -yMax * cosEtaMax / sinEtaMax; if(BETWEEN_F(zTemp, zMin, zMax)) { RThis = sqrtf(yMax*yMax + zTemp*zTemp); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=zTemp; nEdges++;} else continue;} }
             } else {
                  if(BETWEEN_F(0.0f, zMin, zMax) && cosEtaMax > 0) { if(BETWEEN_F(yMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=0.0f; nEdges++;} else continue;} if(BETWEEN_F(yMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=0.0f; nEdges++;} else continue;}}
                  if(BETWEEN_F(0.0f, zMin, zMax) && cosEtaMax < 0) { if(BETWEEN_F(-yMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMin; Edges[nEdges][1]=0.0f; nEdges++;} else continue;} if(BETWEEN_F(-yMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=yMax; Edges[nEdges][1]=0.0f; nEdges++;} else continue;}}
             }
             if (fabsf(cosEtaMax) > EPS_F) {
                 yTemp = -zMin * sinEtaMax / cosEtaMax; if(BETWEEN_F(yTemp, yMin, yMax)) { RThis = sqrtf(yTemp*yTemp + zMin*zMin); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yTemp; Edges[nEdges][1]=zMin; nEdges++;} else continue;} }
                 yTemp = -zMax * sinEtaMax / cosEtaMax; if(BETWEEN_F(yTemp, yMin, yMax)) { RThis = sqrtf(yTemp*yTemp + zMax*zMax); if(BETWEEN_F(RThis, RMin_bin, RMax_bin)) { if(nEdges<50){Edges[nEdges][0]=yTemp; Edges[nEdges][1]=zMax; nEdges++;} else continue;} }
             } else {
                  if(BETWEEN_F(0.0f, yMin, yMax) && sinEtaMax < 0) { if(BETWEEN_F(-zMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if(BETWEEN_F(-zMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMax; nEdges++;} else continue;}}
                  if(BETWEEN_F(0.0f, yMin, yMax) && sinEtaMax > 0) { if(BETWEEN_F( zMin, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMin; nEdges++;} else continue;} if(BETWEEN_F( zMax, RMin_bin, RMax_bin)){ if(nEdges<50){Edges[nEdges][0]=0.0f; Edges[nEdges][1]=zMax; nEdges++;} else continue;}}
             }
            // --- End Intersection Calculation ---

            if (nEdges < 3) // Need at least 3 vertices for a polygon
            {
                continue;
            }

            // Find unique vertices within the bin boundaries
            nEdges = FindUniques_device_f(Edges, EdgesOut, nEdges, RMin_bin, RMax_bin, EtaMin_bin, EtaMax_bin);
            if (nEdges < 3)
            {
                continue;
            }

            // Calculate area of the resulting intersection polygon
            Area = CalcAreaPolygon_device_f(EdgesOut, nEdges);
            if (Area < EPS_F * EPS_F) // Ignore tiny or zero area overlaps
            {
                continue;
            }

            // Atomically increment output counter and store results
            unsigned int currentIndex = atomicAdd(outputCounter, 1);
            if (currentIndex < maxOutputSize)
            {
                outputBuffer[currentIndex].r_bin = r_idx;
                outputBuffer[currentIndex].eta_bin = eta_idx;
                outputBuffer[currentIndex].y = i; // Original pixel y index
                outputBuffer[currentIndex].z = j; // Original pixel z index
                outputBuffer[currentIndex].frac = Area; // Area of overlap
            }
            // If currentIndex >= maxOutputSize, the buffer is full (data is lost)
		} // end loop eta bins
	} // end loop r bins
}


// --- Host ---

// Host comparison function for qsort (C style)
int compareHostOutput_c(const void* a, const void* b)
{
    const HostOutputEntry* ea = (const HostOutputEntry*)a;
    const HostOutputEntry* eb = (const HostOutputEntry*)b;

    if (ea->pixel_idx < eb->pixel_idx) return -1;
    if (ea->pixel_idx > eb->pixel_idx) return 1;

    // If pixel_idx is the same, compare by bin_idx
    if (ea->bin_idx < eb->bin_idx) return -1;
    if (ea->bin_idx > eb->bin_idx) return 1;

    return 0; // Entries are considered equal for sorting purposes
}

// Host REtaMapper (float)
static inline void REtaMapperF( float Rmin, float EtaMin, int nEtaBins, int nRBins, float EtaBinSize, float RBinSize, float *EtaBinsLow, float *EtaBinsHigh, float *RBinsLow, float *RBinsHigh)
{
    int i;
    for (i=0; i<nEtaBins; i++)
    {
        EtaBinsLow[i] = EtaBinSize*i + EtaMin;
        EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
    }
	for (i=0; i<nRBins; i++)
    {
        RBinsLow[i] = RBinSize * i + Rmin;
        RBinsHigh[i] = RBinSize * (i+1) + Rmin;
    }
}

// Host StartsWith
static inline int StartsWith(const char *a, const char *b)
{
    return (strncmp(a,b,strlen(b)) == 0);
}

// Host image transformation (float) - C version
static inline void DoImageTransformationsF (int NrTransOpt, int TransOpt[10], float *ImageIn, float *ImageOut, int NrPixelsY, int NrPixelsZ)
{
    size_t N = (size_t)NrPixelsY * NrPixelsZ;
    size_t mapSizeF = N * sizeof(float);
    int i, k, l;
	if (NrTransOpt == 0)
    {
        memcpy(ImageOut,ImageIn, mapSizeF);
        return;
    }
    float* TempImage = (float*)malloc(mapSizeF);
    if (!TempImage)
    {
        fprintf(stderr, "Failed to alloc temp image\n");
        exit(1);
    }
    memcpy(TempImage, ImageIn, mapSizeF);
    float* CurrentIn = TempImage;
    float* CurrentOut = ImageOut;
    for (i=0; i<NrTransOpt; i++)
    {
        if (i == NrTransOpt - 1)
        {
             CurrentOut = ImageOut;
        }
        else
        {
            CurrentOut = (CurrentIn == TempImage) ? ImageOut : TempImage;
        }
		if (TransOpt[i] == 1)
        {
            for (l=0; l<NrPixelsZ; l++)
            {
                for (k=0; k<NrPixelsY; k++)
                {
                    CurrentOut[l*NrPixelsY+k] = CurrentIn[l*NrPixelsY+(NrPixelsY-k-1)];
                }
            }
        }
        else if (TransOpt[i] == 2)
        {
            for (l=0; l<NrPixelsZ; l++)
            {
                for (k=0; k<NrPixelsY; k++)
                {
                    CurrentOut[l*NrPixelsY+k] = CurrentIn[(NrPixelsZ-l-1)*NrPixelsY+k];
                }
            }
        }
        else if (TransOpt[i] == 0)
        {
             memcpy(CurrentOut, CurrentIn, mapSizeF);
        }
        else
        {
             fprintf(stderr, "Warning: Ignoring unsupported transform option %d\n", TransOpt[i]);
             memcpy(CurrentOut, CurrentIn, mapSizeF);
        }
        CurrentIn = CurrentOut;
	}
    if (CurrentOut != ImageOut)
    {
         memcpy(ImageOut, CurrentOut, mapSizeF);
    }
    free(TempImage);
}


// --- MAIN ---
int main(int argc, char *argv[])
{
    clock_t start, end, start0, end0;
    start0 = clock();
    float diftotal;
    char *ParamFN;
    FILE *paramFile;

    if (argc != 2)
    {
        printf("Usage: %s <paramfile>\n", argv[0]);
        return 1;
    }
    ParamFN = argv[1];

    // --- Default Parameters (float) ---
	float tx=0.0f, ty=0.0f, tz=0.0f, pxY=200.0f, pxZ=200.0f, yCen=1024.0f, zCen=1024.0f, Lsd=1000000.0f, RhoD=200000.0f,
		p0=0.0f, p1=0.0f, p2=0.0f, p3=0.0f, EtaBinSize=5.0f, RBinSize=0.25f, RMax=1524.0f, RMin=10.0f, EtaMax=180.0f, EtaMin=-180.0f;
	int NrPixelsY=2048, NrPixelsZ=2048;
	char aline[4096], dummy[4096];
    const char *str;
	int distortionFile = 0;
    char distortionFN[4096] = "";
	int NrTransOpt=0;
    int TransOpt[10];

    // --- Parameter Parsing (read as double, cast to float) ---
	paramFile = fopen(ParamFN,"r");
    if (!paramFile)
    {
        perror("Error opening parameter file");
        return 1;
    }
	while (fgets(aline,4096,paramFile) != NULL)
    {
        if (aline[0] == '#' || aline[0] == '\n' || aline[0] == '\r') continue;
        double temp_d;
		str = "tx "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); tx=(float)temp_d; }
		str = "ty "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); ty=(float)temp_d; }
		str = "tz "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); tz=(float)temp_d; }
		str = "pxY "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); pxY=(float)temp_d; }
		str = "pxZ "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); pxZ=(float)temp_d; }
		str = "px "; if (StartsWith(aline,str)) { double tpxY, tpxZ; if (sscanf(aline,"%s %lf %lf", dummy, &tpxY, &tpxZ) < 3) { tpxZ = tpxY; } pxY = (float)tpxY; pxZ = (float)tpxZ; }
		str = "BC "; if (StartsWith(aline,str)) { double tyc, tzc; sscanf(aline,"%s %lf %lf", dummy, &tyc, &tzc); yCen=(float)tyc; zCen=(float)tzc; }
		str = "Lsd "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); Lsd=(float)temp_d; }
		str = "RhoD "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); RhoD=(float)temp_d; }
		str = "p0 "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); p0=(float)temp_d; }
		str = "p1 "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); p1=(float)temp_d; }
		str = "p2 "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); p2=(float)temp_d; }
		str = "p3 "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); p3=(float)temp_d; }
		str = "EtaBinSize "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); EtaBinSize=(float)temp_d; }
		str = "RBinSize "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); RBinSize=(float)temp_d; }
		str = "RMax "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); RMax=(float)temp_d; }
		str = "RMin "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); RMin=(float)temp_d; }
		str = "EtaMax "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); EtaMax=(float)temp_d; }
		str = "EtaMin "; if (StartsWith(aline,str)) { sscanf(aline,"%s %lf", dummy, &temp_d); EtaMin=(float)temp_d; }
		str = "NrPixelsY "; if (StartsWith(aline,str)) { sscanf(aline,"%s %d", dummy, &NrPixelsY); }
		str = "NrPixelsZ "; if (StartsWith(aline,str)) { sscanf(aline,"%s %d", dummy, &NrPixelsZ); }
        str = "NrPixels "; if (StartsWith(aline,str)) { int tny, tnz; if (sscanf(aline,"%s %d %d", dummy, &tny, &tnz) < 3) { tnz = tny; } NrPixelsY = tny; NrPixelsZ = tnz; }
		str = "DistortionFile "; if (StartsWith(aline,str)){ distortionFile = 1; sscanf(aline,"%s %s",dummy, distortionFN); }
        str = "ImTransOpt "; if (StartsWith(aline,str)){ if (NrTransOpt < 10) sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt++]); }
	}
    fclose(paramFile);
    printf("Parameters loaded.\n");
    printf(" tx=%.2f, ty=%.2f, tz=%.2f\n", tx, ty, tz);
    printf(" NrPixelsY=%d, NrPixelsZ=%d\n", NrPixelsY, NrPixelsZ);
    printf(" pxY=%.2f, pxZ=%.2f\n", pxY, pxZ);
    printf(" BC=(%.1f, %.1f)\n", yCen, zCen);
    printf(" Lsd=%.1f, RhoD=%.1f\n", Lsd, RhoD);
    printf(" p0=%.4f, p1=%.4f, p2=%.4f, p3=%.4f\n", p0, p1, p2, p3);
    printf(" R=[%.2f, %.2f], RBinSize=%.4f\n", RMin, RMax, RBinSize);
    printf(" Eta=[%.2f, %.2f], EtaBinSize=%.4f\n", EtaMin, EtaMax, EtaBinSize);
    if (distortionFile) printf(" DistortionFile: %s\n", distortionFN);
    printf(" Transforms(%d):", NrTransOpt); { int i; for(i=0; i<NrTransOpt; ++i) printf(" %d", TransOpt[i]); } printf("\n");


    // --- Allocate Host Distortion Maps (float) ---
    float *distortionMapY_h = NULL, *distortionMapZ_h = NULL;
    size_t mapSize = (size_t)NrPixelsY * NrPixelsZ;
    size_t mapSizeBytes = mapSize * sizeof(float);
    distortionMapY_h = (float*)calloc(mapSize, sizeof(float));
    if (!distortionMapY_h)
    {
        fprintf(stderr, "Failed allocate host distortion map Y\n");
        return 1;
    }
	distortionMapZ_h = (float*)calloc(mapSize, sizeof(float));
    if (!distortionMapZ_h)
    {
        fprintf(stderr, "Failed allocate host distortion map Z\n");
        free(distortionMapY_h);
        return 1;
    }

    // --- Read/Transform Distortion File (double -> float) ---
	if (distortionFile == 1)
    {
        printf("Reading distortion file: %s\n", distortionFN);
		FILE *distortionFileHandle = fopen(distortionFN,"rb");
        if (!distortionFileHandle)
        {
            perror("Error opening distortion file");
            free(distortionMapY_h);
            free(distortionMapZ_h);
            return 1;
        }
		double *distortionMapTempD = (double*)malloc(mapSize * sizeof(double));
        float *distortionMapTempF = (float*)malloc(mapSizeBytes);
        if (!distortionMapTempD || !distortionMapTempF)
        {
            fprintf(stderr, "Failed alloc temp dist buffers\n");
            fclose(distortionFileHandle);
            free(distortionMapY_h);
            free(distortionMapZ_h);
            free(distortionMapTempD);
            free(distortionMapTempF);
            return 1;
        }
        size_t read_count;
        size_t i;

        read_count = fread(distortionMapTempD, sizeof(double), mapSize, distortionFileHandle);
        if (read_count != mapSize)
        {
            fprintf(stderr, "Error reading Y distortion map\n");
            fclose(distortionFileHandle); free(distortionMapTempD); free(distortionMapTempF); free(distortionMapY_h); free(distortionMapZ_h); return 1;
        }
        for(i=0; i<mapSize; ++i)
        {
            distortionMapTempF[i] = (float)distortionMapTempD[i];
        }
		DoImageTransformationsF(NrTransOpt, TransOpt, distortionMapTempF, distortionMapY_h, NrPixelsY, NrPixelsZ);

        read_count = fread(distortionMapTempD, sizeof(double), mapSize, distortionFileHandle);
        if (read_count != mapSize)
        {
             fprintf(stderr, "Error reading Z distortion map\n");
             fclose(distortionFileHandle); free(distortionMapTempD); free(distortionMapTempF); free(distortionMapY_h); free(distortionMapZ_h); return 1;
        }
        for(i=0; i<mapSize; ++i)
        {
            distortionMapTempF[i] = (float)distortionMapTempD[i];
        }
		DoImageTransformationsF(NrTransOpt, TransOpt, distortionMapTempF, distortionMapZ_h, NrPixelsY, NrPixelsZ);
		fclose(distortionFileHandle);
        free(distortionMapTempD);
        free(distortionMapTempF);
		printf("Distortion file processed.\n");
	}
    else
    {
        printf("No distortion file provided, using zero distortion.\n");
    }

    // --- Calculate Binning Info (float) ---
	int nRBins = (RBinSize > EPS_F) ? (int) ceilf((RMax-RMin)/RBinSize) : 1;
	int nEtaBins = (EtaBinSize > EPS_F) ? (int)ceilf((EtaMax - EtaMin)/EtaBinSize) : 1;
    if (nRBins <= 0) nRBins = 1;
    if (nEtaBins <= 0) nEtaBins = 1;
    printf("Mapper bins: %d eta, %d R.\n",nEtaBins,nRBins);
	float *EtaBinsLow_h, *EtaBinsHigh_h, *RBinsLow_h, *RBinsHigh_h;
	EtaBinsLow_h = (float*)malloc(nEtaBins*sizeof(float)); if (!EtaBinsLow_h) { fprintf(stderr,"Malloc failed EtaBinsLow_h\n"); return 1; }
	EtaBinsHigh_h = (float*)malloc(nEtaBins*sizeof(float)); if (!EtaBinsHigh_h) { fprintf(stderr,"Malloc failed EtaBinsHigh_h\n"); free(EtaBinsLow_h); return 1; }
	RBinsLow_h = (float*)malloc(nRBins*sizeof(float)); if (!RBinsLow_h) { fprintf(stderr,"Malloc failed RBinsLow_h\n"); free(EtaBinsLow_h); free(EtaBinsHigh_h); return 1; }
	RBinsHigh_h = (float*)malloc(nRBins*sizeof(float)); if (!RBinsHigh_h) { fprintf(stderr,"Malloc failed RBinsHigh_h\n"); free(EtaBinsLow_h); free(EtaBinsHigh_h); free(RBinsLow_h); return 1; }
	REtaMapperF(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow_h, EtaBinsHigh_h, RBinsLow_h, RBinsHigh_h);

    // --- Allocate GPU Memory (float) ---
    float *distortionMapY_d, *distortionMapZ_d;
    float *etaBinsLow_d, *etaBinsHigh_d, *rBinsLow_d, *rBinsHigh_d;
    GpuOutput* outputBuffer_d;
    unsigned int* outputCounter_d;
    gpuErrchk(cudaMalloc(&distortionMapY_d, mapSizeBytes));
    gpuErrchk(cudaMalloc(&distortionMapZ_d, mapSizeBytes));
    gpuErrchk(cudaMalloc(&outputCounter_d, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&etaBinsLow_d, nEtaBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&etaBinsHigh_d, nEtaBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&rBinsLow_d, nRBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&rBinsHigh_d, nRBins * sizeof(float)));
    unsigned int maxOutputSize = (unsigned int)mapSize * 50; // Heuristic
    unsigned long long requestedBytes = (unsigned long long)maxOutputSize * sizeof(GpuOutput);
    printf("Attempting to allocate %.2f GB for GPU output buffer (%u entries)...\n",
           (double)requestedBytes / (1024.0*1024.0*1024.0), maxOutputSize);
    if (requestedBytes > 4ULL * 1024 * 1024 * 1024 ) { // Example limit: 4GB, adjust as needed
        fprintf(stderr, "Warning: Requested output buffer size is very large (%.2f GB). Clamping or consider smaller bins/detector size.\n", (double)requestedBytes / (1024.0*1024.0*1024.0) );
        // Optionally clamp maxOutputSize here if needed based on available memory
    }
    printf("Allocating GPU output buffer for %u potential entries...\n", maxOutputSize);
    cudaError_t malloc_err = gpuErrchk(cudaMalloc(&outputBuffer_d, maxOutputSize * sizeof(GpuOutput)));
    if (malloc_err != cudaSuccess) {
        fprintf(stderr, "FATAL ERROR: Failed to allocate GPU output buffer (Size: %u, Bytes: %llu). Error: %s\n",
                maxOutputSize, requestedBytes, cudaGetErrorString(malloc_err));
        // ... perform necessary cleanup before exiting ...
        cudaFree(distortionMapY_d); // Example cleanup
        cudaFree(distortionMapZ_d);
        cudaFree(outputCounter_d);
        cudaFree(etaBinsLow_d);
        // ... etc for other allocations ...
        return 1; // Exit
   }
   printf("GPU output buffer allocated successfully.\n");

    // --- Copy Data to GPU (float) ---
    gpuErrchk(cudaMemcpy(distortionMapY_d, distortionMapY_h, mapSizeBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(distortionMapZ_d, distortionMapZ_h, mapSizeBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(outputCounter_d, 0, sizeof(unsigned int)));
    const float h_dy[2] = {-0.5f, +0.5f};
    const float h_dz[2] = {-0.5f, +0.5f};
    const float h_PosMatrix[4][2]={{-0.5f,-0.5f},{-0.5f,0.5f},{0.5f,0.5f},{0.5f,-0.5f}};
    gpuErrchk(cudaMemcpyToSymbol(const_dy, h_dy, 2 * sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(const_dz, h_dz, 2 * sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(const_PosMatrix, h_PosMatrix, 4 * 2 * sizeof(float)));
    gpuErrchk(cudaMemcpy(etaBinsLow_d, EtaBinsLow_h, nEtaBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(etaBinsHigh_d, EtaBinsHigh_h, nEtaBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rBinsLow_d, RBinsLow_h, nRBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rBinsHigh_d, RBinsHigh_h, nRBins * sizeof(float), cudaMemcpyHostToDevice));

    // --- Precompute Rotation Matrix on Host (float) ---
    float TRs[3][3];
    {
        float txr=deg2radf*tx,tyr=deg2radf*ty,tzr=deg2radf*tz;
        float Rx[3][3]={{1,0,0},{0,cosf(txr),-sinf(txr)},{0,sinf(txr),cosf(txr)}};
        float Ry[3][3]={{cosf(tyr),0,sinf(tyr)},{0,1,0},{-sinf(tyr),0,cosf(tyr)}};
        float Rz[3][3]={{cosf(tzr),-sinf(tzr),0},{sinf(tzr),cosf(tzr),0},{0,0,1}};
        float TRint[3][3];
        MatrixMultF33F(Ry,Rz,TRint);
        MatrixMultF33F(Rx,TRint,TRs);
    }

    // --- Launch Kernel ---
    dim3 blockSize={16, 16};
    dim3 gridSize={(NrPixelsY+blockSize.x-1)/blockSize.x, (NrPixelsZ+blockSize.y-1)/blockSize.y};
    printf("Launching kernel (float) grid (%u,%u) block (%u,%u)...\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    start = clock();
    mapperKernelF<<<gridSize, blockSize>>>(
        tx, ty, tz, TRs, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen, Lsd, RhoD,
        p0, p1, p2, p3, nRBins, nEtaBins,
        etaBinsLow_d, etaBinsHigh_d, rBinsLow_d, rBinsHigh_d,
        distortionMapY_d, distortionMapZ_d,
        outputBuffer_d, outputCounter_d, maxOutputSize
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    end = clock();
    printf("GPU Kernel execution time: %.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);

    // --- Copy Results Back ---
    unsigned int h_outputCounter = 0;
    gpuErrchk(cudaMemcpy(&h_outputCounter, outputCounter_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("Kernel generated %u output entries.\n", h_outputCounter);
    GpuOutput* h_outputBuffer = NULL;
    long long TotNrOfBins_gpu = 0;
    if (h_outputCounter > maxOutputSize)
    {
        fprintf(stderr, "Error: GPU counter overflow! Results incomplete.\n");
        h_outputCounter = maxOutputSize; // Cap copy size
    }
    if (h_outputCounter > 0)
    {
        h_outputBuffer = (GpuOutput*)malloc(h_outputCounter * sizeof(GpuOutput));
        if (!h_outputBuffer)
        {
            fprintf(stderr, "Error allocating host output buffer\n"); /* cleanup */ return 1;
        }
        gpuErrchk(cudaMemcpy(h_outputBuffer, outputBuffer_d, h_outputCounter * sizeof(GpuOutput), cudaMemcpyDeviceToHost));
        TotNrOfBins_gpu = h_outputCounter;
    }

    // --- Process GPU Results & Build CSR (C Version) ---
    start = clock();
    printf("Processing GPU results and generating CSR map...\n");
    HostOutputEntry* hostEntries = NULL; // Use C dynamic array
    size_t total_pixels = (size_t)NrPixelsY * NrPixelsZ;
    size_t total_bins = (size_t)nRBins * nEtaBins;

    if (h_outputCounter > 0)
    {
        hostEntries = (HostOutputEntry*)malloc(h_outputCounter * sizeof(HostOutputEntry));
        if (!hostEntries)
        {
            fprintf(stderr, "Error allocating hostEntries\n"); /* cleanup */ return 1;
        }

        unsigned int valid_entries = 0;
        unsigned int idx;
        for (idx = 0; idx < h_outputCounter; ++idx)
        {
            size_t p_idx = (size_t)h_outputBuffer[idx].z * NrPixelsY + h_outputBuffer[idx].y;
            size_t b_idx = (size_t)h_outputBuffer[idx].r_bin * nEtaBins + h_outputBuffer[idx].eta_bin;

            if (p_idx >= total_pixels || b_idx >= total_bins)
            {
                fprintf(stderr, "Warning: Invalid index generated pixel_idx=%zu, bin_idx=%zu\n", p_idx, b_idx);
                continue; // Skip invalid entry
            }
            hostEntries[valid_entries].y = h_outputBuffer[idx].y;
            hostEntries[valid_entries].z = h_outputBuffer[idx].z;
            hostEntries[valid_entries].r_bin = h_outputBuffer[idx].r_bin;
            hostEntries[valid_entries].eta_bin = h_outputBuffer[idx].eta_bin;
            hostEntries[valid_entries].frac = h_outputBuffer[idx].frac;
            hostEntries[valid_entries].pixel_idx = p_idx;
            hostEntries[valid_entries].bin_idx = b_idx;
            valid_entries++;
        }
        // Update count if entries were skipped
        TotNrOfBins_gpu = valid_entries;
        h_outputCounter = valid_entries; // Use the valid count for subsequent operations

        // Sort by pixel index, then bin index using qsort
        qsort(hostEntries, h_outputCounter, sizeof(HostOutputEntry), compareHostOutput_c);
    }

    // Allocate CSR arrays + total area array (C style)
    long long* csr_row_offsets = (long long*)calloc(total_pixels + 1, sizeof(long long));
    int* csr_col_indices = (int*)malloc(TotNrOfBins_gpu * sizeof(int));
    float* csr_values = (float*)malloc(TotNrOfBins_gpu * sizeof(float));
    float* total_area_per_bin = (float*)calloc(total_bins, sizeof(float));
    if (!csr_row_offsets || (TotNrOfBins_gpu > 0 && (!csr_col_indices || !csr_values)) || !total_area_per_bin)
    {
         fprintf(stderr, "Error allocating CSR arrays\n"); /* cleanup */ return 1;
    }

    long long current_csr_idx = 0;
    size_t last_pixel_idx = (size_t)-1;
    unsigned int i;

    for (i = 0; i < h_outputCounter; ++i)
    {
        const HostOutputEntry* entry = &hostEntries[i];
        // hostEntries only contains valid entries now

        csr_col_indices[current_csr_idx] = entry->bin_idx;
        csr_values[current_csr_idx] = entry->frac;
        total_area_per_bin[entry->bin_idx] += entry->frac; // Accumulate area

        // Update row offsets
        if (entry->pixel_idx != last_pixel_idx)
        {
            size_t p_idx;
            for (p_idx = last_pixel_idx + 1; p_idx <= entry->pixel_idx; ++p_idx)
            {
                 if (p_idx <= total_pixels)
                 { // Check bounds strictly
                     csr_row_offsets[p_idx] = current_csr_idx;
                 }
                 else
                 {
                      fprintf(stderr, "Error: Offset index %zu out of bounds (%zu)\n", p_idx, total_pixels);
                      // Handle error, maybe break or exit
                 }
            }
            last_pixel_idx = entry->pixel_idx;
        }
        current_csr_idx++;
    }

    // Fill remaining offsets up to the end
    size_t p_idx;
    for (p_idx = last_pixel_idx + 1; p_idx <= total_pixels; ++p_idx)
    {
        if (p_idx <= total_pixels)
        {
             csr_row_offsets[p_idx] = current_csr_idx;
        }
        else
        {
             fprintf(stderr, "Error: Final offset index %zu out of bounds (%zu)\n", p_idx, total_pixels);
        }
    }

    if (current_csr_idx != TotNrOfBins_gpu)
    {
        fprintf(stderr, "Warning: CSR count mismatch\n");
    }
    end = clock();
    printf("CPU Post-processing & CSR generation time: %.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);

    // --- Write CSR Output Files (C Style) ---
	start = clock();
    printf("Writing CSR output files...\n");
    const char *metaFN = "Map_CSR_Meta.txt";
    const char *offsFN = "Map_CSR_Offsets.bin";
    const char *indsFN = "Map_CSR_Indices.bin";
    const char *valsFN = "Map_CSR_Values.bin";
    const char *areaFN = "Map_CSR_AreaPerBin.bin";
    FILE *metafile = fopen(metaFN, "w");
    if (!metafile) { perror("Error opening CSR meta file"); /* cleanup */ return 1; }
    fprintf(metafile, "NrPixelsY %d\n", NrPixelsY);
    fprintf(metafile, "NrPixelsZ %d\n", NrPixelsZ);
    fprintf(metafile, "nRBins %d\n", nRBins);
    fprintf(metafile, "nEtaBins %d\n", nEtaBins);
    fprintf(metafile, "TotalContributions %lld\n", TotNrOfBins_gpu);
    // Add Rmin, Rmax, etc. if needed by integrator
    fclose(metafile);

    FILE *outfile;
    bool ok = true;
    #define WRITE_BIN(filename, data, count, type) \
        outfile = fopen(filename, "wb"); \
        if (!outfile) { perror("Error opening CSR bin file"); ok = false; } \
        else { \
            size_t written = fwrite(data, sizeof(type), count, outfile); \
            fclose(outfile); \
            if (written != (size_t)(count)) { fprintf(stderr, "Error writing %s\n", filename); ok = false; } \
        }

    WRITE_BIN(offsFN, csr_row_offsets, total_pixels + 1, long long);
    if (TotNrOfBins_gpu > 0)
    { // Avoid writing empty files if no contributions
         WRITE_BIN(indsFN, csr_col_indices, TotNrOfBins_gpu, int);
         WRITE_BIN(valsFN, csr_values, TotNrOfBins_gpu, float);
    }
    else
    { // Create empty files if needed
         outfile=fopen(indsFN,"wb"); if(outfile) fclose(outfile);
         outfile=fopen(valsFN,"wb"); if(outfile) fclose(outfile);
    }
    WRITE_BIN(areaFN, total_area_per_bin, total_bins, float);
    #undef WRITE_BIN

    if (!ok)
    {
        fprintf(stderr, "Error during CSR file writing.\n");
        /* cleanup... */
        return 1;
    }
    end = clock();
    printf("CSR Writing time: %.3f s\n", (float)(end - start) / CLOCKS_PER_SEC);

	// --- Cleanup ---
    printf("Cleaning up resources...\n");
    free(distortionMapY_h);
    free(distortionMapZ_h);
    free(EtaBinsLow_h);
    free(EtaBinsHigh_h);
    free(RBinsLow_h);
    free(RBinsHigh_h);
    if (h_outputBuffer) free(h_outputBuffer);
    if (hostEntries) free(hostEntries); // Free the C dynamic array
    free(csr_row_offsets); // Free CSR arrays
    if (csr_col_indices) free(csr_col_indices);
    if (csr_values) free(csr_values);
    free(total_area_per_bin);
    gpuErrchk(cudaFree(distortionMapY_d));
    gpuErrchk(cudaFree(distortionMapZ_d));
    gpuErrchk(cudaFree(outputBuffer_d));
    gpuErrchk(cudaFree(outputCounter_d));
    gpuErrchk(cudaFree(etaBinsLow_d));
    gpuErrchk(cudaFree(etaBinsHigh_d));
    gpuErrchk(cudaFree(rBinsLow_d));
    gpuErrchk(cudaFree(rBinsHigh_d));

	end0 = clock();
    diftotal = ((float)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%.3f s.\n", diftotal);
    printf("Finished.\n");
    return 0;
}
