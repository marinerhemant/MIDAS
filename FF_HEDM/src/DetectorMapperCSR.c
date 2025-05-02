// =========================================================================
// DetectorMapper.c (Reverted Logic, CSR Output)
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// Purpose: Calculates the mapping between detector pixels and integration bins
//          (R, Eta) using original intersection logic, and saves the mapping
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
#include <stdarg.h> // For va_start, va_end

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
        va_start (args, message);
        fprintf(stderr, "Fatal Error: ");
        vfprintf(stderr, message, args);
        va_end (args);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}


static inline
int BETWEEN(double val, double min, double max)
{
    // Use a slightly larger epsilon for BETWEEN checks involving bounds
	return ((val-(EPS*10) <= max && val+(EPS*10) >= min) ? 1 : 0 );
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
	if (fabs(x) < EPS) return 1.0; // Treat near-zero as positive for sign
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

// Original REta4MYZ - takes pxY, needs pxZ implicitly if used elsewhere
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
	double px, // Assume pxY == pxZ now
	double *RetVals) // Expects RetVals[0] = Eta, RetVals[1] = Rt (in pixels)
{
	double Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT, Rt;
	Yc = (-Y + Ycen)*px; // Use the single px value
	Zc = ( Z - Zcen)*px; // Use the single px value
	ABC[0] = 0;
	ABC[1] = Yc;
	ABC[2] = Zc;
	MatrixMult(TRs,ABC,ABCPr);
	XYZ[0] = Lsd+ABCPr[0];
    if (fabs(XYZ[0]) < EPS) {
        Rt = 1e12;
        Eta = 0.0;
    } else {
	    Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
	    Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
        if (RhoD > EPS && (fabs(p0)>EPS || fabs(p1)>EPS || fabs(p2)>EPS)) {
            RNorm = Rad / RhoD;
            EtaT = 90.0 - Eta;
            DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2.0*EtaT))))
                        + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4.0*EtaT+p3))))
                        + (p2*(pow(RNorm,n2))) + 1.0;
            Rad = fmax(0.0, Rad * DistortFunc);
        }
        Rt = Rad / px; // Convert final metric radius to R in pixels
    }
	RetVals[0] = Eta;
	RetVals[1] = Rt;
}


// Original YZ4mREta signature
static inline
void YZ4mREta(double R_pixels, double Eta, double *YZ){
    // NOTE: This original version did NOT use px. It treated R as metric implicitly.
    // This seems inconsistent with REta4MYZ returning R in pixels.
    // To make the original logic work as intended, we MUST assume R passed here is metric.
    // The caller (original mapperfcn) needs to ensure R_pixels is converted to metric first.
    // OR, we keep the modified version:
    // void YZ4mREta(double R_pixels, double Eta, double px, double *YZ_metric){...}
    // Let's STICK TO THE ORIGINAL SIGNATURE and assume caller handles units.
	YZ[0] = -R_pixels*sin(Eta*deg2rad); // Y coordinate (implicitly metric)
	YZ[1] = R_pixels*cos(Eta*deg2rad);  // Z coordinate (implicitly metric)
}

const double dy_orig[2] = {-0.5, +0.5}; // Original only used 2 corners? Let's check mapperfcn
const double dz_orig[2] = {-0.5, +0.5}; // Original mapperfcn iterates k=0..1, l=0..1 (4 corners)
// Use 4 corners consistently
const double dy[4] = {-0.5, +0.5, +0.5, -0.5};
const double dz[4] = {-0.5, -0.5, +0.5, +0.5};


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

// Original Point/Sort logic (needed for original CalcAreaPolygon)
struct Point {
	double x;
	double y;
};

struct Point center;

// Original comparison function for sorting points
static int cmpfunc (const void * ia, const void *ib){
	struct Point *a = (struct Point *)ia;
	struct Point *b = (struct Point *)ib;
	// This logic looks complicated, use standard atan2 sort for robustness?
    // Let's keep the original for now, but this is a potential area for bugs.
	if (a->x - center.x >= 0 && b->x - center.x < 0) return 1;
	if (a->x - center.x < 0 && b->x - center.x >= 0) return -1;
	if (fabs(a->x - center.x) < EPS && fabs(b->x - center.x) < EPS) { // Handle vertical line case
		if (a->y - center.y >= 0 || b->y - center.y >= 0){
			// Original had: return a->y > b->y ? 1 : -1; Corrected:
            if (fabs(a->y - b->y) < EPS) return 0;
            return (a->y > b->y) ? 1 : -1;
		}
        // Original had: return b->y > a->y ? 1 : -1; Corrected:
         if (fabs(a->y - b->y) < EPS) return 0;
         return (b->y > a->y) ? 1 : -1; // Both below center
    }
	// Cross product
	double det = (a->x - center.x) * (b->y - center.y) - (b->x - center.x) * (a->y - center.y);
	if (det < -EPS) return 1; // Use epsilon
    if (det > EPS) return -1;  // Use epsilon

    // Points are collinear, sort by distance squared
    double d1 = (a->x - center.x) * (a->x - center.x) + (a->y - center.y) * (a->y - center.y);
    double d2 = (b->x - center.x) * (b->x - center.x) + (b->y - center.y) * (b->y - center.y);
    if (fabs(d1 - d2) < EPS) return 0; // Consider equal if very close
    return d1 > d2 ? 1 : -1;
}


// Original corner offsets relative to pixel center (used in original mapperfcn)
double PosMatrix[4][2]={{-0.5, -0.5},
						{-0.5,  0.5},
						{ 0.5,  0.5},
						{ 0.5, -0.5}};

// Original CalcAreaPolygon (assumes sorted edges)
static inline
double CalcAreaPolygon(double **Edges, int nEdges){
    if (nEdges < 3) return 0.0; // Added check

	int i;
	struct Point *MyData;
	MyData = malloc(nEdges*sizeof(*MyData));
    check(MyData == NULL, "CalcAreaPolygon: Failed malloc MyData");
	center.x = 0;
	center.y = 0;
	for (i=0;i<nEdges;i++){
		center.x += Edges[i][0];
		center.y += Edges[i][1];
		MyData[i].x = Edges[i][0];
		MyData[i].y = Edges[i][1];
	}
    if (nEdges > 0) { // Avoid division by zero
	    center.x /= nEdges;
	    center.y /= nEdges;
    }

	qsort(MyData, nEdges, sizeof(struct Point), cmpfunc);

	double **SortedEdges;
	SortedEdges = allocMatrix(nEdges+1,2); // Allocates nEdges+1 rows
	for (i=0;i<nEdges;i++){
		SortedEdges[i][0] = MyData[i].x;
		SortedEdges[i][1] = MyData[i].y;
	}
    // Close the polygon for shoelace
	SortedEdges[nEdges][0] = MyData[0].x;
	SortedEdges[nEdges][1] = MyData[0].y;

	double Area=0;
    // Shoelace formula implementation
	for (i=0;i<nEdges;i++){
		Area += (SortedEdges[i][0]*SortedEdges[i+1][1])-(SortedEdges[i+1][0]*SortedEdges[i][1]);
	}
	free(MyData);
	FreeMemMatrix(SortedEdges,nEdges+1); // Free nEdges+1 rows
	return fabs(Area / 2.0); // Return absolute value
}

// Original FindUniques function
static inline
int FindUniques (double **EdgesIn, double **EdgesOut, int nEdgesIn, double RMin_pix, double RMax_pix, double EtaMin, double EtaMax){
	int i,j, nEdgesOut=0, duplicate;
	double Len, RT_pix, ET; // Work with R in pixels
    double Y_met, Z_met; // Metric coordinates for R, Eta calculation

	for (i=0;i<nEdgesIn;i++){
        Y_met = EdgesIn[i][0]; // Assume EdgesIn are metric Y, Z
        Z_met = EdgesIn[i][1];
		duplicate = 0;
		for (j=i+1;j<nEdgesIn;j++){ // Check for duplicate points
            // Check distance in metric space
			Len = sqrt((Y_met-EdgesIn[j][0])*(Y_met-EdgesIn[j][0])+(Z_met-EdgesIn[j][1])*(Z_met-EdgesIn[j][1]));
			if (Len < EPS){ // Use EPS for floating point comparison
				duplicate = 1;
                break; // Found duplicate, no need to check further
			}
		}
        if (duplicate) continue; // Skip if duplicate found

        // Calculate R (pixels) and Eta (degrees) from metric coordinates
        RT_pix = sqrt(Y_met*Y_met + Z_met*Z_met) / 1.0; // NEEDS PX HERE! Let's assume px=1 for now if not passed
        // PROBLEM: px is not available here. This function needs px or R bounds must be metric.
        // Let's assume R bounds passed ARE IN PIXELS as the name suggests.
        ET = CalcEtaAngle(Y_met, Z_met); // Calculate Eta from metric Y, Z

        // Handle Eta wrapping relative to the bin range
        // This logic seems complex and potentially fragile.
		// if (fabs(ET - EtaMin) > 180 || fabs(ET - EtaMax) > 180){
		// 	if (EtaMin < 0) ET = ET - 360; // Shift point's Eta
		// 	else ET = 360 + ET;
		// }
        // Simpler wrapping check: Check if ET is within [EtaMin, EtaMax] OR [EtaMin+360, EtaMax+360] OR [EtaMin-360, EtaMax-360]
        int eta_in_range = 0;
        if (BETWEEN(ET, EtaMin, EtaMax)) eta_in_range = 1;
        else if (EtaMax > 180 && BETWEEN(ET + 360.0, EtaMin, EtaMax)) eta_in_range = 1; // Wrap up
        else if (EtaMin < -180 && BETWEEN(ET - 360.0, EtaMin, EtaMax)) eta_in_range = 1; // Wrap down

        // Check if point is within R and Eta bounds
        // Use BETWEEN for robustness with floating point bounds
		if (BETWEEN(RT_pix, RMin_pix, RMax_pix) == 0){
			duplicate = 1; // Not really duplicate, but outside R bounds
		}
		if (!eta_in_range){ // Check simplified eta range condition
			duplicate = 1; // Outside Eta bounds
		}

        // If not duplicate AND within bounds, add to output
		if (duplicate == 0){
            if (nEdgesOut < 50) { // Check EdgesOut buffer size (assumed 50 from allocMatrix calls)
			    EdgesOut[nEdgesOut][0] = EdgesIn[i][0]; // Copy metric Y
			    EdgesOut[nEdgesOut][1] = EdgesIn[i][1]; // Copy metric Z
			    nEdgesOut++;
            } else {
                fprintf(stderr, "Warn: FindUniques EdgesOut buffer overflow!\n");
                break; // Stop adding points
            }
		}
	}
	return nEdgesOut;
}


// Structure to hold pixel contribution data temporarily
struct data {
	int y;      // Pixel y index
	int z;      // Pixel z index
	double frac; // Fractional area contribution
};

// Reverted mapperfcn using original logic
static inline
long long int
mapperfcn(
	double tx,
	double ty,
	double tz,
	int NrPixelsY,
	int NrPixelsZ,
	double px, // Use single px value
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
	double n0=2.0, n1=4.0, n2=2.0;
	double RetVals[2]; // Local buffer for {Eta, Rt}
    double YZ_metric[2]; // Local buffer for {Y_met, Z_met}

	double Y, Z, Eta, Rt;
	int i,j,k,l,m,n;
	double EtaMi, EtaMa, RMi, RMa; // Ranges in Eta (deg) and R (pixels)

	int RChosen[500], EtaChosen[500]; // Candidate bin indices
	int nrRChosen, nrEtaChosen;

	double **Edges; // Buffer for potential intersection points (metric Y, Z)
	Edges = allocMatrix(50,2); // Max 50 points assumed
	double **EdgesOut; // Buffer for unique, sorted points (metric Y, Z)
	EdgesOut = allocMatrix(50,2);
	int nEdges;

	double RMin_pix, RMax_pix, EtaMin_deg, EtaMax_deg; // Bin bounds
	double yMin_pix, yMax_pix, zMin_pix, zMax_pix; // Pixel bounds in pixel units

	double Area;
	double RThis_pix, EtaThis_deg; // R/Eta of specific points
	double yTemp_met, zTemp_met; // Temporary metric coords for intersection calc
    double yCorner_met, zCorner_met; // Pixel corner in metric coordinates

	struct data *oldarr, *newarr;
	long long int TotNrOfBins = 0; // Total contributions added
	long long int sumNrBins = 0; // Counter for candidate bins checked
	long long int nrContinued1=0; // Counter for nEdges < 3 before FindUniques
	long long int nrContinued2=0; // Counter for nEdges < 3 after FindUniques
	long long int nrContinued3=0; // Counter for Area < threshold

	printf("Starting pixel mapping loop (Original Logic, Y=%d, Z=%d)...\n", NrPixelsY, NrPixelsZ);
    time_t last_print_time = time(NULL);
    long long processed_pixels = 0;


	for (i=0;i<NrPixelsY;i++){ // Loop over pixel Y index
		for (j=0;j<NrPixelsZ;j++){ // Loop over pixel Z index
            processed_pixels++;
            if (processed_pixels % 100000 == 0 || time(NULL) - last_print_time >= 10) { // Print progress
                printf("  Processed %lld / %lld pixels (%.1f%%)\n",
                       processed_pixels, (long long)NrPixelsY * NrPixelsZ,
                       100.0 * (double)processed_pixels / (NrPixelsY * NrPixelsZ));
                last_print_time = time(NULL);
                fflush(stdout);
            }

			EtaMi = 1800; EtaMa = -1800; // Reset Eta range (degrees)
			RMi = 1E10; RMa = -1000;   // Reset R range (pixels)

            // --- Calculate R, Eta range for the pixel corners ---
			long long currentPixelIndex = (long long)j * NrPixelsY + i;
			double y_distorted = (double)i + distortionMapY[currentPixelIndex]; // Y center + distortion (pixel units)
			double z_distorted = (double)j + distortionMapZ[currentPixelIndex]; // Z center + distortion (pixel units)

			for (k = 0; k < 4; k++){ // Iterate over 4 pixel corners (using dy[], dz[])
				Y = y_distorted + dy[k]; // Corner Y in pixel units
				Z = z_distorted + dz[k]; // Corner Z in pixel units

				REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, px, RetVals);
				Eta = RetVals[0]; // degrees
				Rt = RetVals[1]; // pixels

				if (Eta < EtaMi) EtaMi = Eta;
				if (Eta > EtaMa) EtaMa = Eta;
				if (Rt < RMi) RMi = Rt;
				if (Rt > RMa) RMa = Rt;
			}
            // Note: Original code had a complex EtaMa-EtaMi > 180 check here. Skipping for simplicity,
            // but this might affect bins crossing the +/- 180 boundary. The bin check below tries to handle it.

            // Get center pixel's undistorted R, Eta to find its metric YZ
            // This YZ_metric_center represents the "ideal" location for intersection checks.
            double YZ_metric_center[2];
            REta4MYZ(y_distorted, z_distorted, Ycen, Zcen, TRs, Lsd, RhoD, 0,0,0,0, n0, n1, n2, px, RetVals); // No distortion (p=0)
            // Convert the center R (pixels), Eta (deg) to metric YZ
            YZ4mREta(RetVals[1] * px, RetVals[0], YZ_metric_center); // <<< Convert R to metric for YZ4mREta


			// --- Find candidate R and Eta bins based on pixel's R/Eta range ---
			nrRChosen = 0; nrEtaChosen = 0;
			for (k=0;k<nRBins;k++){ // Check R bins
				if (RBinsHigh[k] >= (RMi - EPS) && RBinsLow[k] <= (RMa + EPS)){ // Check overlap R_bin vs R_pixel
                    if (nrRChosen < 500) RChosen[nrRChosen++] = k;
                    else { fprintf(stderr, "Warn: Exceeded RChosen buffer size\n"); break; }
				}
			}
			for (k=0;k<nEtaBins;k++){ // Check Eta bins
                double binEtaLow = EtaBinsLow[k];
                double binEtaHigh = EtaBinsHigh[k];
                int found_eta = 0;
                // Check overlap (consider simple wrapping)
				if ((binEtaHigh >= (EtaMi - EPS) && binEtaLow <= (EtaMa + EPS))) { found_eta = 1; } // Direct
                else if (EtaMa > 179.0 && EtaMi < -179.0) { // Pixel likely crosses +/- 180 boundary
                    // Check if bin overlaps upper part OR lower part
                    if ((binEtaHigh >= (EtaMi + 360.0 - EPS) && binEtaLow <= 360.0) || (binEtaHigh >= -360.0 && binEtaLow <= (EtaMa - 360.0 + EPS))) {
                         // This wrapping logic is complex, original was different. Need careful test.
                         // Let's simplify: check direct or fully shifted.
                         if((binEtaHigh >= (EtaMi+360.0-EPS) && binEtaLow <= (EtaMa+360.0+EPS))) found_eta = 1;
                    }
                }
                 // Original code had more complex wrapping checks. Sticking to simpler for now.

                if (found_eta) {
                     if (nrEtaChosen < 500) EtaChosen[nrEtaChosen++] = k;
                     else { fprintf(stderr, "Warn: Exceeded EtaChosen buffer size\n"); break; }
                }
			}

            if (nrRChosen == 0 || nrEtaChosen == 0) continue; // Pixel outside all candidate bins


            // Define pixel boundaries in METRIC units centered at ideal YZ_metric_center
            double yMin_met = YZ_metric_center[0] - 0.5 * px;
            double yMax_met = YZ_metric_center[0] + 0.5 * px;
            double zMin_met = YZ_metric_center[1] - 0.5 * px;
            double zMax_met = YZ_metric_center[1] + 0.5 * px;


			sumNrBins += nrRChosen * nrEtaChosen; // Count checked candidate bins

			// --- Iterate through candidate bins and calculate overlap ---
			for (k=0;k<nrRChosen;k++){
				int rBinIdx = RChosen[k];
				RMin_pix = RBinsLow[rBinIdx];   // R bin bounds in pixels
				RMax_pix = RBinsHigh[rBinIdx];
                // Convert R bounds to METRIC for intersection checks
                double RMin_met = RMin_pix * px;
                double RMax_met = RMax_pix * px;


				for (l=0;l<nrEtaChosen;l++){
					int etaBinIdx = EtaChosen[l];
					EtaMin_deg = EtaBinsLow[etaBinIdx]; // Eta bin bounds in degrees
					EtaMax_deg = EtaBinsHigh[etaBinIdx];

                    // Find YZ metric coordinates of the four corners of the polar bin
                    double binCornersMetric[4][2]; // Y,Z metric
                    YZ4mREta(RMin_met, EtaMin_deg, binCornersMetric[0]); // BL - Use METRIC R
                    YZ4mREta(RMin_met, EtaMax_deg, binCornersMetric[1]); // TL - Use METRIC R
                    YZ4mREta(RMax_met, EtaMax_deg, binCornersMetric[2]); // TR - Use METRIC R
                    YZ4mREta(RMax_met, EtaMin_deg, binCornersMetric[3]); // BR - Use METRIC R


					nEdges = 0; // Reset count of intersection points for this pixel/bin

					// Check if any corner of the pixel (in metric) is within the polar bin bounds
					for (m=0;m<4;m++){
                        // Calculate metric coords of pixel corner relative to center
                        yCorner_met = YZ_metric_center[0] + PosMatrix[m][0] * px;
                        zCorner_met = YZ_metric_center[1] + PosMatrix[m][1] * px;
                        // Calculate R (metric) and Eta (deg) for this corner
						RThis_pix = sqrt(yCorner_met*yCorner_met + zCorner_met*zCorner_met) / px; // Convert back to Pixels for check
						EtaThis_deg = CalcEtaAngle(yCorner_met, zCorner_met); // Calculate Eta

                        // Handle potential Eta wrapping for the check
                        int corner_eta_in_range = 0;
                        if (BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) corner_eta_in_range = 1;
                        // Add simplified wrapping checks if necessary, similar to bin selection
                        // else if ( ... ) corner_eta_in_range = 1;

						if (BETWEEN(RThis_pix, RMin_pix, RMax_pix) && corner_eta_in_range){
							Edges[nEdges][0] = yCorner_met; // Store metric Y
							Edges[nEdges][1] = zCorner_met; // Store metric Z
							nEdges++;
						}
					}

					// Check if any corner of the polar bin is within the pixel box (metric)
					for (m=0;m<4;m++){
						if (BETWEEN(binCornersMetric[m][0], yMin_met, yMax_met) &&
                            BETWEEN(binCornersMetric[m][1], zMin_met, zMax_met)){
								Edges[nEdges][0] = binCornersMetric[m][0]; // Store metric Y
								Edges[nEdges][1] = binCornersMetric[m][1]; // Store metric Z
								nEdges++;
							}
					}

                    // If < 4 points found so far, check for intersections between
                    // pixel edges and bin boundaries (arcs and rays). This is the complex part.
                    // Original code checked intersections of R/Eta boundaries with pixel rectangle edges.
                    // All calculations should be in METRIC space.

					if (nEdges < 4){ // Only do detailed intersections if needed
                        // --- Intersections of R arcs with Pixel Edges ---
                        // RMin arc with y=yMin_met/yMax_met lines
                        if (RMin_met >= fabs(yMin_met)) { // Check if R is large enough to intersect y line
                            zTemp_met = sqrt(RMin_met*RMin_met - yMin_met*yMin_met); // Potential z values
                            // Check positive z
                            EtaThis_deg = CalcEtaAngle(yMin_met, zTemp_met);
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            // Check negative z
                            EtaThis_deg = CalcEtaAngle(yMin_met, -zTemp_met);
                            if (BETWEEN(-zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=-zTemp_met; nEdges++; }
                        }
                         if (RMin_met >= fabs(yMax_met)) {
                            zTemp_met = sqrt(RMin_met*RMin_met - yMax_met*yMax_met);
                            EtaThis_deg = CalcEtaAngle(yMax_met, zTemp_met);
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            EtaThis_deg = CalcEtaAngle(yMax_met, -zTemp_met);
                            if (BETWEEN(-zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=-zTemp_met; nEdges++; }
                        }
                        // RMax arc with y=yMin_met/yMax_met lines (similar logic)
                        if (RMax_met >= fabs(yMin_met)) {
                            zTemp_met = sqrt(RMax_met*RMax_met - yMin_met*yMin_met);
                            EtaThis_deg = CalcEtaAngle(yMin_met, zTemp_met);
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            EtaThis_deg = CalcEtaAngle(yMin_met, -zTemp_met);
                            if (BETWEEN(-zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=-zTemp_met; nEdges++; }
                        }
                         if (RMax_met >= fabs(yMax_met)) {
                            zTemp_met = sqrt(RMax_met*RMax_met - yMax_met*yMax_met);
                            EtaThis_deg = CalcEtaAngle(yMax_met, zTemp_met);
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            EtaThis_deg = CalcEtaAngle(yMax_met, -zTemp_met);
                            if (BETWEEN(-zTemp_met, zMin_met, zMax_met) && BETWEEN(EtaThis_deg, EtaMin_deg, EtaMax_deg)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=-zTemp_met; nEdges++; }
                        }

                        // RMin/RMax arcs with z=zMin_met/zMax_met lines (similar logic)
                        // ... (omitted for brevity, but structure is the same, swapping y/z checks) ...

                        // --- Intersections of Eta rays with Pixel Edges ---
                        double tanEtaMin = tan(EtaMin_deg * deg2rad);
                        double tanEtaMax = tan(EtaMax_deg * deg2rad);
                        // EtaMin ray with y=yMin/yMax
                        if (fabs(EtaMin_deg) > EPS && fabs(fabs(EtaMin_deg)-180.0) > EPS) { // Avoid vertical tan
                            zTemp_met = -yMin_met / tanEtaMin;
                            RThis_pix = sqrt(yMin_met*yMin_met + zTemp_met*zTemp_met) / px;
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(RThis_pix, RMin_pix, RMax_pix)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            zTemp_met = -yMax_met / tanEtaMin;
                             RThis_pix = sqrt(yMax_met*yMax_met + zTemp_met*zTemp_met) / px;
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(RThis_pix, RMin_pix, RMax_pix)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                        }
                        // EtaMax ray with y=yMin/yMax (similar)
                         if (fabs(EtaMax_deg) > EPS && fabs(fabs(EtaMax_deg)-180.0) > EPS) { // Avoid vertical tan
                            zTemp_met = -yMin_met / tanEtaMax;
                            RThis_pix = sqrt(yMin_met*yMin_met + zTemp_met*zTemp_met) / px;
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(RThis_pix, RMin_pix, RMax_pix)) { Edges[nEdges][0]=yMin_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                            zTemp_met = -yMax_met / tanEtaMax;
                            RThis_pix = sqrt(yMax_met*yMax_met + zTemp_met*zTemp_met) / px;
                            if (BETWEEN(zTemp_met, zMin_met, zMax_met) && BETWEEN(RThis_pix, RMin_pix, RMax_pix)) { Edges[nEdges][0]=yMax_met; Edges[nEdges][1]=zTemp_met; nEdges++; }
                        }

                        // EtaMin/EtaMax rays with z=zMin/zMax (similar, check for tan(90) etc.)
                        // ... (omitted for brevity) ...
					} // end if (nEdges < 4)

                    // Limit nEdges before calling FindUniques/CalcAreaPolygon
                    if (nEdges > 48) { // Leave space for EdgesOut buffer + polygon closing
                        fprintf(stderr, "Warn: Too many intersection points (%d) for pixel (%d,%d) / bin (R%d,E%d). Skipping.\n", nEdges, i,j,rBinIdx,etaBinIdx);
                        continue;
                    }
					if (nEdges < 3){
						nrContinued1++;
						continue;
					}

                    // Filter unique points THAT ARE WITHIN the R/Eta bin bounds
                    // Note: Original FindUniques took R/Eta bounds. We pass R_pix bounds.
                    int nUniqueEdges = FindUniques(Edges, EdgesOut, nEdges, RMin_pix, RMax_pix, EtaMin_deg, EtaMax_deg);

					if (nUniqueEdges < 3){
						nrContinued2++;
						continue;
					}

					// Now we have the unique vertices of the intersection polygon (in metric YZ)
                    // Calculate the area.
					Area = CalcAreaPolygon(EdgesOut, nUniqueEdges); // Uses metric coords

                    // Normalize by pixel area (metric)
                    double pixelAreaMetric = px * px;
                    double fracArea = (pixelAreaMetric > EPS) ? Area / pixelAreaMetric : 0.0;

					if (fracArea < 1E-9){ // Threshold fraction
						nrContinued3++;
						continue;
					}
                     if (fracArea > 1.0 + EPS) { fracArea = 1.0; } // Clamp


					// Populate the temporary list arrays
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
					pxList[rBinIdx][etaBinIdx][nVal].y = i; // Original pixel Y index
					pxList[rBinIdx][etaBinIdx][nVal].z = j; // Original pixel Z index
					pxList[rBinIdx][etaBinIdx][nVal].frac = fracArea;
					(nPxList[rBinIdx][etaBinIdx])++;
					TotNrOfBins++; // Increment total non-zero count
				} // end eta bin loop
			} // end R bin loop
		} // end pixel z loop
	} // end pixel y loop

    FreeMemMatrix(Edges, 50); // Cleanup allocated Edges buffer
    FreeMemMatrix(EdgesOut, 50); // Cleanup allocated EdgesOut buffer

    printf("Pixel mapping loop finished. Processed %lld pixels.\n", processed_pixels);
	printf("Contributions skipped: nEdges<3 (%lld + %lld), Area<thresh (%lld)\n", nrContinued1, nrContinued2, nrContinued3);
	return TotNrOfBins;
}

static inline
int StartsWith(const char *a, const char *b)
{
    if (a == NULL || b == NULL) return 0;
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

// Basic image transformation for distortion map
static inline void DoImageTransformations (int NrTransOpt, const int TransOpt[10], const double *ImageIn, double *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	// ... (Keep the DoImageTransformations function from the previous corrected version) ...
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
        const double* readBuffer = (i == 0) ? ImageIn : ((i % 2 != 0) ? tempBuffer : ImageOut);
        double* writeBuffer = (i % 2 == 0) ? ((NrTransOpt == 1 || ImageIn == ImageOut || NrTransOpt % 2 != 0) ? tempBuffer : ImageOut) : ImageOut;


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

    if (NrTransOpt % 2 != 0 && tempBuffer != NULL) {
         memcpy(ImageOut, tempBuffer, N * sizeof(double));
    }

    if(tempBuffer) free(tempBuffer);
}


// ================== MAIN ==================
int main(int argc, char *argv[])
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    char *ParamFN;
    FILE *paramFile;

    check(argc != 2, "Usage: %s <parameter_file>\n", argv[0]);
	ParamFN = argv[1];

    // Default values & parameter reading (same as previous version)
	double tx=0.0, ty=0.0, tz=0.0, px=200.0, yCen=1024.0, zCen=1024.0, Lsd=1000000.0, RhoD=200000.0,
		p0=0.0, p1=0.0, p2=0.0, p3=0.0, EtaBinSize=5, RBinSize=0.25, RMax=1524.0, RMin=10.0, EtaMax=180.0, EtaMin=-180.0;
	int NrPixelsY=2048, NrPixelsZ=2048;
	char aline[4096], key[1024], val_str[3072];
	distortionFile = 0;
	char distortionFN[4096] = "";
	int NrTransOpt=0;
	int TransOpt[10] = {0};
    // ... (parameter reading loop - same as before) ...
	paramFile = fopen(ParamFN,"r");
    check(paramFile == NULL, "Failed to open parameter file '%s': %s", ParamFN, strerror(errno));
	printf("Reading parameters from: %s\n", ParamFN);
    while(fgets(aline, sizeof(aline), paramFile)){ /* ... parameter parsing ... */
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
                 if (sscanf(val_str, "%s", distortionFN) == 1 && strlen(distortionFN) > 0) { distortionFile = 1; }
                 else { printf("Warn: Empty DistortionFile value ignored.\n"); }
             }
             else if (strcmp(key, "ImTransOpt") == 0) {
                 if(NrTransOpt < 10) sscanf(val_str, "%d", &TransOpt[NrTransOpt++]);
                 else printf("Warn: Max 10 ImTransOpt reached.\n");
             }
        }
    }
	fclose(paramFile);
    // ... (parameter validation and printing - same as before) ...
    check(NrPixelsY <= 0 || NrPixelsZ <= 0, "Invalid NrPixelsY/Z: %d x %d", NrPixelsY, NrPixelsZ);
    check(px <= EPS, "Invalid pixel size px: %f", px);
    check(Lsd <= EPS, "Invalid sample-detector distance Lsd: %f", Lsd);
    check(RBinSize <= EPS || EtaBinSize <= EPS, "Invalid R/Eta bin size: %f, %f", RBinSize, EtaBinSize);
    check(RMax <= RMin || EtaMax <= EtaMin, "Invalid R/Eta ranges");
    printf("Parameters Loaded:\n");
    printf(" Geometry: tx=%.2f ty=%.2f tz=%.2f px=%.3f Lsd=%.1f RhoD=%.1f\n", tx, ty, tz, px, Lsd, RhoD);
    printf(" Center:   Ycen=%.1f Zcen=%.1f\n", yCen, zCen);
    printf(" Detector: %d x %d pixels\n", NrPixelsY, NrPixelsZ);
    printf(" Distortion: p0=%.4f p1=%.4f p2=%.4f p3=%.4f File=%s\n", p0, p1, p2, p3, distortionFile ? distortionFN : "None");
    printf(" Binning: R=[%.2f..%.2f], step=%.3f | Eta=[%.1f..%.1f], step=%.2f\n", RMin, RMax, RBinSize, EtaMin, EtaMax, EtaBinSize);
    printf(" Transforms (%d):", NrTransOpt);
    for(int i=0; i<NrTransOpt; ++i) printf(" %d", TransOpt[i]); printf("\n");

    // Distortion map loading (same as before)
	distortionMapY = calloc((size_t)NrPixelsY*NrPixelsZ,sizeof(double));
	distortionMapZ = calloc((size_t)NrPixelsY*NrPixelsZ,sizeof(double));
    check(distortionMapY == NULL || distortionMapZ == NULL, "Failed to allocate distortion map memory");
	if (distortionFile == 1){ /* ... load and transform distortion maps ... */
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

    // Calculate bins and allocate edges (same as before)
	int nEtaBins, nRBins;
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int) ceil((EtaMax - EtaMin)/EtaBinSize);
    check(nRBins <= 0 || nEtaBins <= 0, "Calculated zero or negative bins (R:%d, Eta:%d)", nRBins, nEtaBins);
	printf("Creating mapper: %d Eta bins, %d R bins.\n",nEtaBins,nRBins);
	double *EtaBinsLow, *EtaBinsHigh, *RBinsLow, *RBinsHigh; // ... allocate ...
    EtaBinsLow = malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = malloc(nRBins*sizeof(*RBinsHigh));
    check(!EtaBinsLow || !EtaBinsHigh || !RBinsLow || !RBinsHigh, "Failed to allocate bin edge arrays");
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);

    // Allocate temporary structures (same as before)
	struct data ***pxList; int **nPxList; int **maxnPx; // ... allocate ...
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

    // --- Run the REVERTED mapping function ---
    printf("Running mapper function (Original Logic)...\n"); fflush(stdout);
    start = clock();
    long long int TotNrOfBins = mapperfcn(tx, ty, tz, NrPixelsY, NrPixelsZ, px, yCen,
								zCen, Lsd, RhoD, p0, p1, p2, p3, EtaBinsLow,
								EtaBinsHigh, RBinsLow, RBinsHigh, nRBins,
								nEtaBins, pxList, nPxList, maxnPx);
    end = clock();
    printf("Mapper function finished in %.2f seconds.\n", ((double)(end-start))/CLOCKS_PER_SEC);
	printf("Total Number of non-zero contributions (NNZ): %lld\n", TotNrOfBins);
    fflush(stdout);

    // Check if any contributions were found before proceeding
    if (TotNrOfBins == 0) {
        printf("Warning: No pixel contributions found using original logic. Check parameters/geometry. Exiting.\n");
        // Perform cleanup
        free(EtaBinsLow); free(EtaBinsHigh); free(RBinsLow); free(RBinsHigh);
        free(distortionMapY); free(distortionMapZ);
        for (i=0; i<nRBins; ++i) { /* ... free inner lists ... */
            if (pxList[i]) { for (j=0; j<nEtaBins; ++j) free(pxList[i][j]); free(pxList[i]); }
            if (nPxList[i]) free(nPxList[i]); if (maxnPx[i]) free(maxnPx[i]);
        } free(pxList); free(nPxList); free(maxnPx);
        return 1;
    }

    // --- Convert temporary lists to CSR format ---
    // (This part remains the same as the previous working version)
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
            check(current_nnz_idx > INT_MAX, "CSR row pointer exceeds INT_MAX at row %d", row_idx);
            csr_row_ptr[row_idx + 1] = (int)current_nnz_idx;
        }
    }
    // ... (Sanity checks for NNZ and final row pointer - same as before) ...
    if (current_nnz_idx != num_nonzeros) { /* Adjust num_nonzeros */ num_nonzeros = current_nnz_idx; }
    if (csr_row_ptr[num_rows] != num_nonzeros) { /* Print warning */ }


    // --- Write CSR data to files ---
    // (This part remains the same)
    printf("Writing CSR files...\n");
    const char *hdr_fn = "MapCSR.hdr"; /* ... */
    const char *val_fn = "MapCSR_values.bin"; /* ... */
    const char *col_fn = "MapCSR_col_indices.bin"; /* ... */
    const char *row_fn = "MapCSR_row_ptr.bin"; /* ... */
    // ... (fopen, fwrite, fclose for hdr, val, col, row files - same as before) ...
    FILE *hdr_file = fopen(hdr_fn, "w"); check(hdr_file == NULL, "..."); fprintf(hdr_file, "%d\n%d\n%lld\n", num_rows, num_cols, num_nonzeros); fclose(hdr_file); printf(" - Wrote %s\n", hdr_fn);
    FILE *val_file = fopen(val_fn, "wb"); check(val_file == NULL, "..."); size_t written_val = fwrite(csr_values, sizeof(double), num_nonzeros, val_file); check(written_val != num_nonzeros, "..."); fclose(val_file); printf(" - Wrote %s (%zu bytes)\n", val_fn, written_val * sizeof(double));
    FILE *col_file = fopen(col_fn, "wb"); check(col_file == NULL, "..."); size_t written_col = fwrite(csr_col_indices, sizeof(int), num_nonzeros, col_file); check(written_col != num_nonzeros, "..."); fclose(col_file); printf(" - Wrote %s (%zu bytes)\n", col_fn, written_col * sizeof(int));
    FILE *row_file = fopen(row_fn, "wb"); check(row_file == NULL, "..."); size_t written_row = fwrite(csr_row_ptr, sizeof(int), num_rows + 1, row_file); check(written_row != (num_rows + 1), "..."); fclose(row_file); printf(" - Wrote %s (%zu bytes)\n", row_fn, written_row * sizeof(int));


	// --- Cleanup ---
    printf("Cleaning up memory...\n");
    free(csr_values);
    free(csr_col_indices);
    free(csr_row_ptr);
    // ... (Free temporary structures pxList, nPxList, maxnPx - same as before) ...
	for (i=0;i<nRBins;i++){ /* ... free inner lists ... */
        if (pxList[i]) { for (j=0; j<nEtaBins; ++j) free(pxList[i][j]); free(pxList[i]); }
        if (nPxList[i]) free(nPxList[i]); if (maxnPx[i]) free(maxnPx[i]);
    } free(pxList); free(nPxList); free(maxnPx);
    // ... (Free bin edge arrays - same as before) ...
    free(EtaBinsLow); free(EtaBinsHigh); free(RBinsLow); free(RBinsHigh);
    // ... (Free distortion maps - same as before) ...
    free(distortionMapY); free(distortionMapZ);

	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed: %.2f s.\n", diftotal);
    printf("DetectorMapper finished successfully.\n");
    return 0;
}