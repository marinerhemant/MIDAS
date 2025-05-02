// =========================================================================
// DetectorMapper.c (Modified for CSR Output - v2 with fixes)
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// Purpose: Calculates the mapping between detector pixels and integration bins
//          (R, Eta) considering geometry and distortion, and saves the mapping
//          in Compressed Sparse Row (CSR) format.
// =========================================================================

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <errno.h> // Required for errno
#include <stdarg.h> // <<< Added for va_start, va_end

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-6
double *distortionMapY;
double *distortionMapZ;
int distortionFile;

// --- Error Checking ---
static void check (int test, const char * message, ...) {
    if (test) {
        va_list args;
        va_start (args, message); // Correct usage requires stdarg.h
        fprintf(stderr, "Fatal Error: ");
        vfprintf(stderr, message, args);
        va_end (args); // Correct usage requires stdarg.h
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}


static inline
int BETWEEN(double val, double min, double max)
{
	return ((val-EPS <= max && val+EPS >= min) ? 1 : 0 );
}

static inline
double**
allocMatrix(int nrows, int ncols)
{
    double** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    check(arr == NULL, "allocMatrix: Failed to allocate row pointers");
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        check(arr[i] == NULL, "allocMatrix: Failed to allocate column data for row %d", i);
    }
    return arr;
}

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    if (mat == NULL) return;
    for ( r = 0 ; r < nrows ; r++) {
        if(mat[r] != NULL) free(mat[r]);
    }
    free(mat);
}


static inline double signVal(double x){
	if (x == 0) return 1.0;
	else return x/fabs(x);
}

static inline
void
MatrixMult(
           double m[3][3],
           double  v[3],
           double r[3])
{
    int i;
    for (i=0; i<3; i++) {
        r[i] = m[i][0]*v[0] +
        m[i][1]*v[1] +
        m[i][2]*v[2];
    }
}

static inline
void
MatrixMultF33(
    double m[3][3],
    double n[3][3],
    double res[3][3])
{
    int r;
    for (r=0; r<3; r++) {
        res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0];
        res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1];
        res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2];
    }
}

static inline
double CalcEtaAngle(double y, double z){
    if (fabs(y) < EPS && fabs(z) < EPS) return 0.0; // Avoid atan2(0,0) -> undefined
	double alpha = rad2deg*atan2(-y, z); // Use atan2 for correct quadrant handling
	return alpha;
}

static inline
void
REta4MYZ(
	double Y,
	double Z,
	double Ycen,
	double Zcen,
	double TRs[3][3],
	double Lsd,
	double RhoD,
	double p0,
	double p1,
	double p2,
	double p3,
	double n0,
	double n1,
	double n2,
	double px, // Assume pxY == pxZ for distortion correction simplicity
	double *RetVals) // Expects RetVals[0] = Eta, RetVals[1] = Rt (in pixels)
{
	double Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT, Rt;
	Yc = (-Y + Ycen)*px;
	Zc = ( Z - Zcen)*px;
	ABC[0] = 0;
	ABC[1] = Yc;
	ABC[2] = Zc;
	MatrixMult(TRs,ABC,ABCPr);
	XYZ[0] = Lsd+ABCPr[0];
    // Check for division by zero or near-zero
    if (fabs(XYZ[0]) < EPS) {
        Rt = 1e12; // Effectively infinite R
        Eta = 0.0; // Arbitrary Eta
    } else {
	    Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2])); // Radius in metric units
	    Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
	    // Apply distortion correction (if parameters are non-zero)
        if (RhoD > EPS && (fabs(p0)>EPS || fabs(p1)>EPS || fabs(p2)>EPS)) {
            RNorm = Rad / RhoD;
            EtaT = 90.0 - Eta; // Adjust Eta for distortion function convention if needed
            DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2.0*EtaT))))
                        + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4.0*EtaT+p3))))
                        + (p2*(pow(RNorm,n2))) + 1.0;
            Rad = fmax(0.0, Rad * DistortFunc); // Apply distortion
        }
        Rt = Rad / px; // Convert final metric radius to R in pixels
    }
	RetVals[0] = Eta;
	RetVals[1] = Rt;
}

static inline
void YZ4mREta(double R_pixels, double Eta, double px, double *YZ_metric){
    // Convert R (in pixels) back to metric distance before calculating Y, Z
    double R_metric = R_pixels * px;
	YZ_metric[0] = -R_metric*sin(Eta*deg2rad); // Y coordinate (metric)
	YZ_metric[1] = R_metric*cos(Eta*deg2rad);  // Z coordinate (metric)
}

const double dy[4] = {-0.5, +0.5, +0.5, -0.5}; // Pixel corner y-offsets
const double dz[4] = {-0.5, -0.5, +0.5, +0.5}; // Pixel corner z-offsets

static inline
void
REtaMapper(
	double Rmin,
	double EtaMin,
	int nEtaBins,
	int nRBins,
	double EtaBinSize,
	double RBinSize,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh)
{
	int i;
	for (i=0;i<nEtaBins;i++){
		EtaBinsLow[i] = EtaBinSize*i      + EtaMin;
		EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
	}
	for (i=0;i<nRBins;i++){
		RBinsLow[i] = RBinSize * i      + Rmin;
		RBinsHigh[i] = RBinSize * (i+1) + Rmin;
	}
}

struct Point {
	double x; // Represents Y in the detector plane (metric)
	double y; // Represents Z in the detector plane (metric)
};

struct Point center;

// Comparison function for qsort to sort points counter-clockwise
static int cmpfunc (const void * ia, const void *ib){
	struct Point *a = (struct Point *)ia;
	struct Point *b = (struct Point *)ib;
    double angle_a = atan2(a->y - center.y, a->x - center.x);
    double angle_b = atan2(b->y - center.y, b->x - center.x);
    if (angle_a < angle_b) return -1;
    if (angle_a > angle_b) return 1;
    double dist_sq_a = (a->x - center.x)*(a->x - center.x) + (a->y - center.y)*(a->y - center.y);
    double dist_sq_b = (b->x - center.x)*(b->x - center.x) + (b->y - center.y)*(b->y - center.y);
    if (dist_sq_a < dist_sq_b) return -1;
    if (dist_sq_a > dist_sq_b) return 1;
    return 0;
}

// Calculate area of a polygon using Shoelace formula
static inline
double CalcAreaPolygonShoelace(struct Point *SortedEdges, int nEdges){
	if (nEdges < 3) return 0.0;
	double Area = 0.0;
	int j = nEdges - 1;
	for (int i = 0; i < nEdges; i++) {
		Area += (SortedEdges[j].x + SortedEdges[i].x) * (SortedEdges[j].y - SortedEdges[i].y);
		j = i;
	}
	return fabs(Area / 2.0);
}

// Find unique vertices and sort them counter-clockwise for area calculation
static inline
int SortUniqueVertices(struct Point *EdgesIn, struct Point *EdgesOut, int nEdgesIn) {
    if (nEdgesIn == 0) return 0;
    int nEdgesOut = 0;
    center.x = 0.0;
    center.y = 0.0;
    double unique_threshold_sq = EPS * EPS;
    for (int i = 0; i < nEdgesIn; i++) {
        int duplicate = 0;
        for (int j = 0; j < nEdgesOut; j++) {
            double dx = EdgesIn[i].x - EdgesOut[j].x;
            double dy = EdgesIn[i].y - EdgesOut[j].y;
            if ((dx*dx + dy*dy) < unique_threshold_sq) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            EdgesOut[nEdgesOut] = EdgesIn[i];
            center.x += EdgesIn[i].x;
            center.y += EdgesIn[i].y;
            nEdgesOut++;
        }
    }
    if (nEdgesOut > 0) {
        center.x /= nEdgesOut;
        center.y /= nEdgesOut;
    } else {
        return 0;
    }
    qsort(EdgesOut, nEdgesOut, sizeof(struct Point), cmpfunc);
    return nEdgesOut;
}

// Structure to hold pixel contribution data temporarily
struct data {
	int y;      // Pixel y index
	int z;      // Pixel z index
	double frac; // Fractional area contribution
};

// Function to calculate intersection of two line segments
static int LineSegmentIntersection(struct Point p1, struct Point p2, struct Point p3, struct Point p4, struct Point *intersection) {
    double det = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (fabs(det) < EPS) return 0; // Parallel or collinear
    double t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / det;
    double u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / det;
    if (t >= -EPS && t <= (1.0 + EPS) && u >= -EPS && u <= (1.0 + EPS)) { // Allow small tolerance
        intersection->x = p1.x + t * (p2.x - p1.x);
        intersection->y = p1.y + t * (p2.y - p1.y);
        return 1;
    }
    return 0;
}


// Sutherland-Hodgman polygon clipping algorithm
static int ClipPolygon(struct Point *subjectPolygon, int nSubjectVertices,
                       struct Point *clipPolygon, int nClipVertices,
                       struct Point *resultPolygon, int maxResultVertices) {
    if (nClipVertices < 3 || nSubjectVertices == 0) return 0;

    struct Point tempPolygon[maxResultVertices];
    int nTempVertices = 0;
    int nResultVertices = nSubjectVertices;

    // Initialize resultPolygon with subjectPolygon if they are different buffers
    if (resultPolygon != subjectPolygon) {
        memcpy(resultPolygon, subjectPolygon, nSubjectVertices * sizeof(struct Point));
    }

    for (int i = 0; i < nClipVertices; ++i) {
        struct Point clipP1 = clipPolygon[i];
        struct Point clipP2 = clipPolygon[(i + 1) % nClipVertices];

        nTempVertices = 0;
        if (nResultVertices == 0) break;

        struct Point s = resultPolygon[nResultVertices - 1];

        for (int j = 0; j < nResultVertices; ++j) {
            struct Point e = resultPolygon[j];
            double s_pos = (clipP2.x - clipP1.x) * (s.y - clipP1.y) - (clipP2.y - clipP1.y) * (s.x - clipP1.x);
            double e_pos = (clipP2.x - clipP1.x) * (e.y - clipP1.y) - (clipP2.y - clipP1.y) * (e.x - clipP1.x);
            struct Point intersectionPoint;

            if (s_pos >= -EPS && e_pos >= -EPS) { // Case 1: Both inside
                 if (nTempVertices < maxResultVertices) tempPolygon[nTempVertices++] = e;
                 else { fprintf(stderr,"Warn: Clip buffer overflow (Case 1)\n"); return 0; }
            }
            else if (s_pos >= -EPS && e_pos < -EPS) { // Case 2: S inside, E outside
                if (LineSegmentIntersection(s, e, clipP1, clipP2, &intersectionPoint)) {
                     if (nTempVertices < maxResultVertices) tempPolygon[nTempVertices++] = intersectionPoint;
                     else { fprintf(stderr,"Warn: Clip buffer overflow (Case 2)\n"); return 0; }
                } else { /* Edge case: parallel or intersection fails */ }
            }
            else if (s_pos < -EPS && e_pos >= -EPS) { // Case 3: S outside, E inside
                if (LineSegmentIntersection(s, e, clipP1, clipP2, &intersectionPoint)) {
                    if (nTempVertices < maxResultVertices) tempPolygon[nTempVertices++] = intersectionPoint;
                    else { fprintf(stderr,"Warn: Clip buffer overflow (Case 3a)\n"); return 0; }
                } else { /* Edge case: parallel or intersection fails */ }
                // Output E
                if (nTempVertices < maxResultVertices) tempPolygon[nTempVertices++] = e;
                else { fprintf(stderr,"Warn: Clip buffer overflow (Case 3b)\n"); return 0; }
            }
            // Case 4: Both outside -> No output
            s = e;
        }
        nResultVertices = nTempVertices;
        memcpy(resultPolygon, tempPolygon, nResultVertices * sizeof(struct Point));
    }

    // Final sort and unique check after all clipping
    // Use a separate buffer for SortUniqueVertices output if resultPolygon is needed intact
    // Let's sort in place for simplicity here.
    int nFinalVertices = SortUniqueVertices(resultPolygon, resultPolygon, nResultVertices);

    return nFinalVertices;
}


static inline
long long int
mapperfcn(
	double tx,
	double ty,
	double tz,
	int NrPixelsY,
	int NrPixelsZ,
	double px, // Use single px assuming square pixels for YZ mapping
	double Ycen,
	double Zcen,
	double Lsd,
	double RhoD,
	double p0,
	double p1,
	double p2,
	double p3,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh,
	int nRBins,
	int nEtaBins,
	struct data ***pxList, // Temporary structure to hold contributions per bin
	int **nPxList,         // Temporary count per bin
	int **maxnPx)          // Temporary max size per bin
{
	double txr, tyr, tzr;
	txr = deg2rad*tx;
	tyr = deg2rad*ty;
	tzr = deg2rad*tz;
	double Rx[3][3] = {{1,0,0},{0,cos(txr),-sin(txr)},{0,sin(txr),cos(txr)}};
	double Ry[3][3] = {{cos(tyr),0,sin(tyr)},{0,1,0},{-sin(tyr),0,cos(tyr)}};
	double Rz[3][3] = {{cos(tzr),-sin(tzr),0},{sin(tzr),cos(tzr),0},{0,0,1}};
	double TRint[3][3], TRs[3][3];
	MatrixMultF33(Ry,Rz,TRint);
	MatrixMultF33(Rx,TRint,TRs);
	double n0=2.0, n1=4.0, n2=2.0; // Defaults for distortion exponents
	double RetVals[2]; // Local buffer for REta4MYZ output {Eta, Rt}
    double YZ_temp[2]; // Local buffer for YZ4mREta output {Y, Z} metric

	double Y, Z, Eta, Rt;
	int i,j,k,l,m;
	double EtaMi, EtaMa, RMi, RMa;

	const int MAX_POLY_VERTICES = 16;
    struct Point pixelCornersMetric[4];
    struct Point binCornersMetric[4];
    struct Point clippedPolygon[MAX_POLY_VERTICES];


	struct data *oldarr, *newarr;
	long long int TotNrOfBins = 0;
	long long int skippedAreaCount = 0;

	printf("Starting pixel mapping loop (Y=%d, Z=%d)...\n", NrPixelsY, NrPixelsZ);
    time_t last_print_time = time(NULL);
    long long processed_pixels = 0;


	for (i=0;i<NrPixelsY;i++){
		for (j=0;j<NrPixelsZ;j++){
            processed_pixels++;
            if (processed_pixels % 100000 == 0 || time(NULL) - last_print_time >= 10) { // Print progress
                printf("  Processed %lld / %lld pixels (%.1f%%)\n",
                       processed_pixels, (long long)NrPixelsY * NrPixelsZ,
                       100.0 * (double)processed_pixels / (NrPixelsY * NrPixelsZ));
                last_print_time = time(NULL);
                fflush(stdout);
            }

			EtaMi = 1800;
			EtaMa = -1800;
			RMi = 1E10; // R in pixels
			RMa = -1000;

            // --- Calculate R, Eta range and pixel corners in metric space ---
			long long currentPixelIndex = (long long)j * NrPixelsY + i;
			double y_distorted = (double)i + distortionMapY[currentPixelIndex];
			double z_distorted = (double)j + distortionMapZ[currentPixelIndex];

			for (k = 0; k < 4; k++){ // Iterate over 4 pixel corners
				Y = y_distorted + dy[k]; // Corner Y in pixel units
				Z = z_distorted + dz[k]; // Corner Z in pixel units

				// Get R, Eta including distortion effects for range finding
				REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, px, RetVals);
				Eta = RetVals[0];
				Rt = RetVals[1]; // R in pixels

				if (Eta < EtaMi) EtaMi = Eta;
				if (Eta > EtaMa) EtaMa = Eta;
				if (Rt < RMi) RMi = Rt;
				if (Rt > RMa) RMa = Rt;

                // Get R, Eta *without* distortion for geometric calculations
                double RetValsGeom[2];
                REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, 0,0,0,0, n0, n1, n2, px, RetValsGeom); // p=0 -> no distortion
                // Convert ideal R,Eta to metric YZ for the pixel corner
                YZ4mREta(RetValsGeom[1], RetValsGeom[0], px, YZ_temp);
                pixelCornersMetric[k].x = YZ_temp[0]; // Y metric
                pixelCornersMetric[k].y = YZ_temp[1]; // Z metric
			}
            // Check for angle wrapping (simplified check)
            if (EtaMa - EtaMi > 350.0) {
                 double tempEtaMi = 1800, tempEtaMa = -1800;
                 int wraps = 0;
                 for (k=0; k<4; ++k) {
                     Y = y_distorted + dy[k]; Z = z_distorted + dz[k];
                     REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, px, RetVals);
                     Eta = RetVals[0];
                     if (k>0) { // Check wrap relative to previous corner
                         double prevEta = (k==1)?RetVals[0]:((k==2)?EtaMi:EtaMa); // Hacky way to get a reference
                         if (fabs(Eta - prevEta) > 180.0) wraps++;
                     }
                     if (Eta < 0) Eta += 360.0;
                     if (Eta < tempEtaMi) tempEtaMi = Eta;
                     if (Eta > tempEtaMa) tempEtaMa = Eta;
                 }
                 if (tempEtaMa - tempEtaMi < 350.0 && wraps > 0) { // Use adjusted range if it seems valid
                     EtaMi = tempEtaMi;
                     EtaMa = tempEtaMa;
                 } // else stick with original, might miss bins near wrap
            }


			// --- Find candidate R and Eta bins ---
			int RChosen[500], EtaChosen[500];
            int nrRChosen = 0, nrEtaChosen = 0;
			for (k=0;k<nRBins;k++){
				if (RBinsHigh[k] >= (RMi - EPS) && RBinsLow[k] <= (RMa + EPS)){
                    if (nrRChosen < 500) RChosen[nrRChosen++] = k;
                    else { fprintf(stderr, "Warn: Exceeded RChosen buffer size\n"); break; }
				}
			}
			for (k=0;k<nEtaBins;k++){
                double binEtaLow = EtaBinsLow[k];
                double binEtaHigh = EtaBinsHigh[k];
                // Check overlap (consider simple cases, complex wrapping is tricky)
				if ((binEtaHigh >= (EtaMi - EPS) && binEtaLow <= (EtaMa + EPS)) || // Direct
                    (binEtaHigh >= (EtaMi + 360.0 - EPS) && binEtaLow <= (EtaMa + 360.0 + EPS)) || // Shifted pixel up
                    ((binEtaHigh + 360.0) >= (EtaMi - EPS) && (binEtaLow + 360.0) <= (EtaMa + EPS))    // Shifted bin up
                   )
                {
                     if (nrEtaChosen < 500) EtaChosen[nrEtaChosen++] = k;
                     else { fprintf(stderr, "Warn: Exceeded EtaChosen buffer size\n"); break; }
                }
			}

            if (nrRChosen == 0 || nrEtaChosen == 0) continue;

			// --- Calculate Area Contribution using Polygon Clipping ---
			for (k=0;k<nrRChosen;k++){
				int rBinIdx = RChosen[k];
				double RMinBin = RBinsLow[rBinIdx]; // R in pixels
				double RMaxBin = RBinsHigh[rBinIdx]; // R in pixels

				for (l=0;l<nrEtaChosen;l++){
					int etaBinIdx = EtaChosen[l];
					double EtaMinBin = EtaBinsLow[etaBinIdx];
					double EtaMaxBin = EtaBinsHigh[etaBinIdx];

                    // Define the corners of the R-Eta bin in metric YZ space
                    YZ4mREta(RMinBin, EtaMinBin, px, YZ_temp); binCornersMetric[0].x = YZ_temp[0]; binCornersMetric[0].y = YZ_temp[1]; // BL
                    YZ4mREta(RMaxBin, EtaMinBin, px, YZ_temp); binCornersMetric[1].x = YZ_temp[0]; binCornersMetric[1].y = YZ_temp[1]; // BR
                    YZ4mREta(RMaxBin, EtaMaxBin, px, YZ_temp); binCornersMetric[2].x = YZ_temp[0]; binCornersMetric[2].y = YZ_temp[1]; // TR
                    YZ4mREta(RMinBin, EtaMaxBin, px, YZ_temp); binCornersMetric[3].x = YZ_temp[0]; binCornersMetric[3].y = YZ_temp[1]; // TL

                    // Clip the pixel (subject) against the R-Eta bin (clip)
                    int nClippedVertices = ClipPolygon(pixelCornersMetric, 4,
                                                       binCornersMetric, 4,
                                                       clippedPolygon, MAX_POLY_VERTICES);

                    if (nClippedVertices < 3) continue;

                    // Calculate area of the clipped polygon (metric units)
                    double Area = CalcAreaPolygonShoelace(clippedPolygon, nClippedVertices);

                    // Normalize area by pixel area (metric units) to get fraction
                    double pixelAreaMetric = px * px;
                    double fracArea = (pixelAreaMetric > EPS) ? Area / pixelAreaMetric : 0.0;

					if (fracArea < 1E-9){ // Threshold for area fraction
						skippedAreaCount++;
						continue;
					}
                    if (fracArea > 1.0 + EPS) {
                        // This might happen due to float precision or clipping issues near boundaries
                        // fprintf(stderr, "Warn: Px(%d,%d) Bin(R:%d, E:%d) frac=%.7f > 1. Clamp.\n",
                        //        i, j, rBinIdx, etaBinIdx, fracArea);
                        fracArea = 1.0;
                    }

					// Populate the temporary list structure
					int maxnVal = maxnPx[rBinIdx][etaBinIdx];
					int nVal = nPxList[rBinIdx][etaBinIdx];
					if (nVal >= maxnVal){
						maxnVal = (maxnVal == 0) ? 4 : maxnVal * 2;
						oldarr = pxList[rBinIdx][etaBinIdx];
						newarr = realloc(oldarr, maxnVal*sizeof(*newarr));
						check(newarr == NULL, "Failed to realloc pxList[%d][%d]", rBinIdx, etaBinIdx);
						pxList[rBinIdx][etaBinIdx] = newarr;
						maxnPx[rBinIdx][etaBinIdx] = maxnVal;
					}
					pxList[rBinIdx][etaBinIdx][nVal].y = i;
					pxList[rBinIdx][etaBinIdx][nVal].z = j;
					pxList[rBinIdx][etaBinIdx][nVal].frac = fracArea;
					(nPxList[rBinIdx][etaBinIdx])++;
					TotNrOfBins++;
				} // end eta bins loop
			} // end R bins loop
		} // end pixel z loop
	} // end pixel y loop

    printf("Pixel mapping loop finished. Processed %lld pixels.\n", processed_pixels);
	printf("Total contributions found: %lld\n", TotNrOfBins);
    printf("Contributions skipped due to tiny area: %lld\n", skippedAreaCount);
	return TotNrOfBins;
}

static inline
int StartsWith(const char *a, const char *b)
{
    if (a == NULL || b == NULL) return 0;
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

// Basic image transformation
static inline void DoImageTransformations (int NrTransOpt, const int TransOpt[10], const double *ImageIn, double *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	int i,j,k,l;
    const double* currentIn = ImageIn;
    double* currentOut = ImageOut;
    double* tempBuffer = NULL;
    size_t N = (size_t)NrPixelsY * NrPixelsZ;

    if (NrTransOpt == 0) {
        if (ImageIn != ImageOut) memcpy(ImageOut, ImageIn, N * sizeof(double));
        return;
    }
    if (NrTransOpt % 2 != 0 || ImageIn == ImageOut) {
        tempBuffer = malloc(N * sizeof(double));
        check(tempBuffer == NULL, "Failed to allocate temporary transform buffer");
    }

    for (i=0; i < NrTransOpt; i++) {
        const double* readBuffer = (i == 0) ? ImageIn : ((i % 2 != 0) ? tempBuffer : ImageOut); // Read from previous output
        double* writeBuffer = (i % 2 == 0) ? ((NrTransOpt == 1 || ImageIn == ImageOut || NrTransOpt % 2 != 0) ? tempBuffer : ImageOut) : ImageOut; // Write to target/temp


        if (TransOpt[i] == 1) { // Flip Horizontal (Y)
            for (l = 0; l < NrPixelsZ; ++l) {
                for (k = 0; k < NrPixelsY; ++k) {
                    writeBuffer[l * NrPixelsY + k] = readBuffer[l * NrPixelsY + (NrPixelsY - 1 - k)];
                }
            }
        } else if (TransOpt[i] == 2) { // Flip Vertical (Z)
             for (l = 0; l < NrPixelsZ; ++l) {
                for (k = 0; k < NrPixelsY; ++k) {
                    writeBuffer[l * NrPixelsY + k] = readBuffer[(NrPixelsZ - 1 - l) * NrPixelsY + k];
                }
            }
        } else if (TransOpt[i] == 3) { // Transpose (only if square)
            if (NrPixelsY == NrPixelsZ) {
                for (l = 0; l < NrPixelsZ; ++l) {
                    for (k = 0; k < NrPixelsY; ++k) {
                        writeBuffer[l * NrPixelsY + k] = readBuffer[k * NrPixelsY + l];
                    }
                }
            } else {
                fprintf(stderr, "Warning: Skipping transpose on non-square image (%dx%d)\n", NrPixelsY, NrPixelsZ);
                if (writeBuffer != readBuffer) memcpy(writeBuffer, readBuffer, N * sizeof(double));
            }
        } else { // No-op or unknown
             if (writeBuffer != readBuffer) memcpy(writeBuffer, readBuffer, N * sizeof(double));
        }
    }

    // If odd number of transforms, the result is in tempBuffer, copy to ImageOut
    if (NrTransOpt % 2 != 0 && tempBuffer != NULL) {
         memcpy(ImageOut, tempBuffer, N * sizeof(double));
    }

    if(tempBuffer) free(tempBuffer);
}

int main(int argc, char *argv[])
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    char *ParamFN;
    FILE *paramFile;

    check(argc != 2, "Usage: %s <parameter_file>\n"
		"Parameters needed: tx, ty, tz, px, BC, Lsd, RhoD,"
		"\n\t\t   p0, p1, p2, p3, EtaBinSize, EtaMin,\n\t\t   EtaMax, RBinSize, RMin, RMax,\n\t\t   NrPixelsY, NrPixelsZ, [DistortionFile], [ImTransOpt]\n", argv[0]);
	ParamFN = argv[1];

    // Default values
	double tx=0.0, ty=0.0, tz=0.0, px=200.0, yCen=1024.0, zCen=1024.0, Lsd=1000000.0, RhoD=200000.0,
		p0=0.0, p1=0.0, p2=0.0, p3=0.0, EtaBinSize=5, RBinSize=0.25, RMax=1524.0, RMin=10.0, EtaMax=180.0, EtaMin=-180.0;
	int NrPixelsY=2048, NrPixelsZ=2048;
	char aline[4096], key[1024], val_str[3072];
	distortionFile = 0;
	char distortionFN[4096] = "";
	int NrTransOpt=0;
	int TransOpt[10] = {0};

	paramFile = fopen(ParamFN,"r");
    check(paramFile == NULL, "Failed to open parameter file '%s': %s", ParamFN, strerror(errno));

	printf("Reading parameters from: %s\n", ParamFN);
    while(fgets(aline, sizeof(aline), paramFile)){
        if(aline[0] == '#' || isspace(aline[0]) || strlen(aline) < 3) continue;
        if (sscanf(aline, "%1023s %[^\n]", key, val_str) == 2) {
             if (strcmp(key, "tx") == 0) sscanf(val_str, "%lf", &tx);
             else if (strcmp(key, "ty") == 0) sscanf(val_str, "%lf", &ty);
             else if (strcmp(key, "tz") == 0) sscanf(val_str, "%lf", &tz);
             else if (strcmp(key, "px") == 0) sscanf(val_str, "%lf", &px);
             else if (strcmp(key, "BC") == 0) sscanf(val_str, "%lf %lf", &yCen, &zCen);
             else if (strcmp(key, "Lsd") == 0) sscanf(val_str, "%lf", &Lsd);
             else if (strcmp(key, "RhoD") == 0) sscanf(val_str, "%lf", &RhoD);
             else if (strcmp(key, "p0") == 0) sscanf(val_str, "%lf", &p0);
             else if (strcmp(key, "p1") == 0) sscanf(val_str, "%lf", &p1);
             else if (strcmp(key, "p2") == 0) sscanf(val_str, "%lf", &p2);
             else if (strcmp(key, "p3") == 0) sscanf(val_str, "%lf", &p3);
             else if (strcmp(key, "EtaBinSize") == 0) sscanf(val_str, "%lf", &EtaBinSize);
             else if (strcmp(key, "RBinSize") == 0) sscanf(val_str, "%lf", &RBinSize);
             else if (strcmp(key, "RMax") == 0) sscanf(val_str, "%lf", &RMax);
             else if (strcmp(key, "RMin") == 0) sscanf(val_str, "%lf", &RMin);
             else if (strcmp(key, "EtaMax") == 0) sscanf(val_str, "%lf", &EtaMax);
             else if (strcmp(key, "EtaMin") == 0) sscanf(val_str, "%lf", &EtaMin);
             else if (strcmp(key, "NrPixelsY") == 0) sscanf(val_str, "%d", &NrPixelsY);
             else if (strcmp(key, "NrPixelsZ") == 0) sscanf(val_str, "%d", &NrPixelsZ);
             else if (strcmp(key, "NrPixels") == 0) { sscanf(val_str, "%d", &NrPixelsY); NrPixelsZ = NrPixelsY; }
             else if (strcmp(key, "DistortionFile") == 0) {
                 if (sscanf(val_str, "%s", distortionFN) == 1 && strlen(distortionFN) > 0) {
                    distortionFile = 1;
                 } else {
                    printf("Warn: Empty DistortionFile value ignored.\n");
                 }
             }
             else if (strcmp(key, "ImTransOpt") == 0) {
                 if(NrTransOpt < 10) sscanf(val_str, "%d", &TransOpt[NrTransOpt++]);
                 else printf("Warn: Max 10 ImTransOpt reached.\n");
             }
        }
    }
	fclose(paramFile);

    check(NrPixelsY <= 0 || NrPixelsZ <= 0, "Invalid NrPixelsY/Z: %d x %d", NrPixelsY, NrPixelsZ);
    check(px <= EPS, "Invalid pixel size px: %f", px);
    check(Lsd <= EPS, "Invalid sample-detector distance Lsd: %f", Lsd);
    check(RBinSize <= EPS || EtaBinSize <= EPS, "Invalid R/Eta bin size: %f, %f", RBinSize, EtaBinSize);
    check(RMax <= RMin || EtaMax <= EtaMin, "Invalid R/Eta ranges");

    printf("Parameters Loaded:\n"); // ... (parameter printing) ...
    printf(" Geometry: tx=%.2f ty=%.2f tz=%.2f px=%.3f Lsd=%.1f RhoD=%.1f\n", tx, ty, tz, px, Lsd, RhoD);
    printf(" Center:   Ycen=%.1f Zcen=%.1f\n", yCen, zCen);
    printf(" Detector: %d x %d pixels\n", NrPixelsY, NrPixelsZ);
    printf(" Distortion: p0=%.4f p1=%.4f p2=%.4f p3=%.4f File=%s\n", p0, p1, p2, p3, distortionFile ? distortionFN : "None");
    printf(" Binning: R=[%.2f..%.2f], step=%.3f | Eta=[%.1f..%.1f], step=%.2f\n", RMin, RMax, RBinSize, EtaMin, EtaMax, EtaBinSize);
    printf(" Transforms (%d):", NrTransOpt);
    for(int i=0; i<NrTransOpt; ++i) printf(" %d", TransOpt[i]); printf("\n");


	distortionMapY = calloc((size_t)NrPixelsY*NrPixelsZ,sizeof(double));
	distortionMapZ = calloc((size_t)NrPixelsY*NrPixelsZ,sizeof(double));
    check(distortionMapY == NULL || distortionMapZ == NULL, "Failed to allocate distortion map memory");

	if (distortionFile == 1){
		FILE *distortionFileHandle = fopen(distortionFN,"rb");
        check(distortionFileHandle == NULL, "Failed to open distortion file '%s': %s", distortionFN, strerror(errno));
		double *distortionMapTempY, *distortionMapTempZ;
        size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;
		distortionMapTempY = malloc(totalPixels*sizeof(double));
		distortionMapTempZ = malloc(totalPixels*sizeof(double));
        check(distortionMapTempY == NULL || distortionMapTempZ == NULL, "Failed to allocate temporary distortion buffers");
        printf("Reading Y distortion data...\n");
		size_t readY = fread(distortionMapTempY, sizeof(double), totalPixels, distortionFileHandle);
        check(readY != totalPixels, "Failed to read full Y distortion map (%zu/%zu elements read)", readY, totalPixels);
        printf("Applying transforms to Y distortion map...\n");
		DoImageTransformations(NrTransOpt,TransOpt, distortionMapTempY, distortionMapY, NrPixelsY, NrPixelsZ);
        printf("Reading Z distortion data...\n");
		size_t readZ = fread(distortionMapTempZ, sizeof(double), totalPixels, distortionFileHandle);
        check(readZ != totalPixels, "Failed to read full Z distortion map (%zu/%zu elements read)", readZ, totalPixels);
        printf("Applying transforms to Z distortion map...\n");
		DoImageTransformations(NrTransOpt,TransOpt, distortionMapTempZ, distortionMapZ, NrPixelsY, NrPixelsZ);
		fclose(distortionFileHandle);
        free(distortionMapTempY);
        free(distortionMapTempZ);
		printf("Distortion file %s processed successfully.\n", distortionFN);
	}

    int nEtaBins, nRBins;
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int) ceil((EtaMax - EtaMin)/EtaBinSize);
    check(nRBins <= 0 || nEtaBins <= 0, "Calculated zero or negative bins (R:%d, Eta:%d)", nRBins, nEtaBins);
	printf("Creating mapper: %d Eta bins, %d R bins.\n",nEtaBins,nRBins);

	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = malloc(nRBins*sizeof(*RBinsHigh));
    check(!EtaBinsLow || !EtaBinsHigh || !RBinsLow || !RBinsHigh, "Failed to allocate bin edge arrays");
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);

	struct data ***pxList;
	int **nPxList;
	int **maxnPx;
	pxList = malloc(nRBins * sizeof(*pxList));
	nPxList = malloc(nRBins * sizeof(*nPxList));
	maxnPx = malloc(nRBins * sizeof(*maxnPx));
    check(!pxList || !nPxList || !maxnPx, "Failed to allocate top-level temporary map arrays");

	int i,j;
	for (i=0;i<nRBins;i++){
		pxList[i] = malloc(nEtaBins*sizeof(**pxList));
		nPxList[i] = calloc(nEtaBins, sizeof(**nPxList));
		maxnPx[i] = calloc(nEtaBins, sizeof(**maxnPx));
        check(!pxList[i] || !nPxList[i] || !maxnPx[i], "Failed to allocate temporary map arrays for R bin %d", i);
		for (j=0;j<nEtaBins;j++){
			pxList[i][j] = NULL;
		}
	}

    printf("Running mapper function...\n"); fflush(stdout);
    start = clock();
    long long int TotNrOfBins = mapperfcn(tx, ty, tz, NrPixelsY, NrPixelsZ, px, yCen,
								zCen, Lsd, RhoD, p0, p1, p2, p3, EtaBinsLow,
								EtaBinsHigh, RBinsLow, RBinsHigh, nRBins,
								nEtaBins, pxList, nPxList, maxnPx);
    end = clock();
    printf("Mapper function finished in %.2f seconds.\n", ((double)(end-start))/CLOCKS_PER_SEC);
	printf("Total Number of non-zero contributions (NNZ): %lld\n", TotNrOfBins);
    fflush(stdout);

    if (TotNrOfBins == 0) {
        printf("Warning: No pixel contributions found. Check parameters/geometry. Exiting.\n");
        // Perform cleanup before exiting
        free(EtaBinsLow); free(EtaBinsHigh); free(RBinsLow); free(RBinsHigh);
        free(distortionMapY); free(distortionMapZ);
        for (i=0; i<nRBins; ++i) {
            if (pxList[i]) {
                for (j=0; j<nEtaBins; ++j) free(pxList[i][j]);
                free(pxList[i]);
            }
            if (nPxList[i]) free(nPxList[i]);
            if (maxnPx[i]) free(maxnPx[i]);
        }
        free(pxList); free(nPxList); free(maxnPx);
        return 1;
    }

    printf("Converting to CSR format...\n");
    int num_rows = nRBins * nEtaBins;
    int num_cols = NrPixelsY * NrPixelsZ;
    long long num_nonzeros = TotNrOfBins;

    double *csr_values = malloc(num_nonzeros * sizeof(double));
    int *csr_col_indices = malloc(num_nonzeros * sizeof(int));
    int *csr_row_ptr = malloc((num_rows + 1) * sizeof(int));
    check(!csr_values || !csr_col_indices || !csr_row_ptr, "Failed to allocate CSR arrays");

    long long current_nnz_idx = 0;
    csr_row_ptr[0] = 0;
    for (i=0; i<nRBins; i++){
        for (j=0; j<nEtaBins; j++){
            int row_idx = i * nEtaBins + j;
            int count = nPxList[i][j];
            for (int k=0; k<count; k++){
                int pixel_y = pxList[i][j][k].y;
                int pixel_z = pxList[i][j][k].z;
                double frac = pxList[i][j][k].frac;
                int pixel_col_idx = pixel_z * NrPixelsY + pixel_y;
                check(current_nnz_idx >= num_nonzeros, "CSR index overflow! idx=%lld >= nnz=%lld", current_nnz_idx, num_nonzeros);
                check(pixel_col_idx < 0 || pixel_col_idx >= num_cols, "Invalid pixel column index %d (max %d)", pixel_col_idx, num_cols-1);
                csr_values[current_nnz_idx] = frac;
                csr_col_indices[current_nnz_idx] = pixel_col_idx;
                current_nnz_idx++;
            }
            // Ensure row pointers are integers
            check(current_nnz_idx > INT_MAX, "CSR row pointer exceeds INT_MAX at row %d", row_idx);
            csr_row_ptr[row_idx + 1] = (int)current_nnz_idx;
        }
    }

    if (current_nnz_idx != num_nonzeros) {
         printf("Warning: Final NNZ mismatch! Expected %lld, filled %lld\n", num_nonzeros, current_nnz_idx);
         num_nonzeros = current_nnz_idx; // Adjust count to actual elements filled
    }
    if (csr_row_ptr[num_rows] != num_nonzeros) {
         printf("Warning: Final row pointer mismatch! Expected %lld, got %d\n", num_nonzeros, csr_row_ptr[num_rows]);
         // This usually indicates an error in the loop logic or counting
    }

    printf("Writing CSR files...\n");
    const char *hdr_fn = "MapCSR.hdr";
    const char *val_fn = "MapCSR_values.bin";
    const char *col_fn = "MapCSR_col_indices.bin";
    const char *row_fn = "MapCSR_row_ptr.bin";

    FILE *hdr_file = fopen(hdr_fn, "w");
    check(hdr_file == NULL, "Failed to open header file '%s': %s", hdr_fn, strerror(errno));
    fprintf(hdr_file, "%d\n", num_rows);
    fprintf(hdr_file, "%d\n", num_cols);
    fprintf(hdr_file, "%lld\n", num_nonzeros); // Write the potentially adjusted count
    fclose(hdr_file);
    printf(" - Wrote %s\n", hdr_fn);

    FILE *val_file = fopen(val_fn, "wb");
    check(val_file == NULL, "Failed to open values file '%s': %s", val_fn, strerror(errno));
    size_t written_val = fwrite(csr_values, sizeof(double), num_nonzeros, val_file); // Use adjusted count
    check(written_val != num_nonzeros, "Failed to write full values data (%zu/%lld)", written_val, num_nonzeros);
    fclose(val_file);
    printf(" - Wrote %s (%zu bytes)\n", val_fn, written_val * sizeof(double));

    FILE *col_file = fopen(col_fn, "wb");
    check(col_file == NULL, "Failed to open col_indices file '%s': %s", col_fn, strerror(errno));
    size_t written_col = fwrite(csr_col_indices, sizeof(int), num_nonzeros, col_file); // Use adjusted count
     check(written_col != num_nonzeros, "Failed to write full col_indices data (%zu/%lld)", written_col, num_nonzeros);
    fclose(col_file);
    printf(" - Wrote %s (%zu bytes)\n", col_fn, written_col * sizeof(int));

    FILE *row_file = fopen(row_fn, "wb");
    check(row_file == NULL, "Failed to open row_ptr file '%s': %s", row_fn, strerror(errno));
    size_t written_row = fwrite(csr_row_ptr, sizeof(int), num_rows + 1, row_file);
    check(written_row != (num_rows + 1), "Failed to write full row_ptr data (%zu/%d)", written_row, num_rows + 1);
    fclose(row_file);
    printf(" - Wrote %s (%zu bytes)\n", row_fn, written_row * sizeof(int));

	printf("Cleaning up memory...\n");
    free(csr_values);
    free(csr_col_indices);
    free(csr_row_ptr);

	for (i=0;i<nRBins;i++){
        if (pxList[i]) {
            for (j=0;j<nEtaBins;j++){
                if (pxList[i][j]) free(pxList[i][j]);
            }
            free(pxList[i]);
        }
        if (nPxList[i]) free(nPxList[i]);
        if (maxnPx[i]) free(maxnPx[i]);
	}
	free(pxList);
	free(nPxList);
	free(maxnPx);
    free(EtaBinsLow); free(EtaBinsHigh); free(RBinsLow); free(RBinsHigh);
    free(distortionMapY); free(distortionMapZ);

	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed: %.2f s.\n", diftotal);
    printf("DetectorMapper finished successfully.\n");
    return 0;
}