//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  Created by Hemant Sharma on 2017/07/10.
//
//
// TODO: Add option to give QbinSize instead of RbinSize, look at 0,90,180,270

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-6
double *distortionMapY;
double *distortionMapZ;
int distortionFile;

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
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
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
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
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
	double px,
	double *RetVals)
{
	double Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT, Rt;
	Yc = (-Y + Ycen)*px;
	Zc = ( Z - Zcen)*px;
	ABC[0] = 0;
	ABC[1] = Yc;
	ABC[2] = Zc;
	MatrixMult(TRs,ABC,ABCPr);
	XYZ[0] = Lsd+ABCPr[0];
	XYZ[1] = ABCPr[1];
	XYZ[2] = ABCPr[2];
	Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
	Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
	RNorm = Rad/RhoD;
	EtaT = 90 - Eta;
	DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT+p3)))) + (p2*(pow(RNorm,n2))) + 1;
	Rt = Rad * DistortFunc / px; // in pixels
	RetVals[0] = Eta;
	RetVals[1] = Rt;
}

static inline
void YZ4mREta(double R, double Eta, double *YZ){
	YZ[0] = -R*sin(Eta*deg2rad);
	YZ[1] = R*cos(Eta*deg2rad);
}

const double dy[2] = {-0.5, +0.5};
const double dz[2] = {-0.5, +0.5};

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
	int i, j, k, l;
	for (i=0;i<nEtaBins;i++){
		EtaBinsLow[i] = EtaBinSize*i      + EtaMin;
		EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
	}
	for (i=0;i<nRBins;i++){
		RBinsLow[i] = RBinSize * i      + Rmin;
		RBinsHigh[i] = RBinSize * (i+1) + Rmin;
	}
}

static inline
int
nOutside(
	double pos,
	int direction,
	double tempIntercepts[],
	int nIntercepts)
{
	int i;
	int nOD = 0;
	for (i=0;i<nIntercepts;i++){
		if (direction * pos < direction * tempIntercepts[i]){
			nOD ++;
		}
	}
	return nOD;
}

struct Point {
	double x;
	double y;
};

struct Point center;

static int cmpfunc (const void * ia, const void *ib){
	struct Point *a = (struct Point *)ia;
	struct Point *b = (struct Point *)ib;
	if (a->x - center.x >= 0 && b->x - center.x < 0) return 1;
	if (a->x - center.x < 0 && b->x - center.x >= 0) return -1;
	if (a->x - center.x == 0 && b->x - center.x == 0) {
		if (a->y - center.y >= 0 || b->y - center.y >= 0){
			return a->y > b->y ? 1 : -1;
		}
        return b->y > a->y ? 1 : -1;
    }
	double det = (a->x - center.x) * (b->y - center.y) - (b->x - center.x) * (a->y - center.y);
	if (det < 0) return 1;
    if (det > 0) return -1;
    int d1 = (a->x - center.x) * (a->x - center.x) + (a->y - center.y) * (a->y - center.y);
    int d2 = (b->x - center.x) * (b->x - center.x) + (b->y - center.y) * (b->y - center.y);
    return d1 > d2 ? 1 : -1;
}

double PosMatrix[4][2]={{-0.5, -0.5},
						{-0.5,  0.5},
						{ 0.5,  0.5},
						{ 0.5, -0.5}};

static inline
double CalcAreaPolygon(double **Edges, int nEdges){
	int i;
	struct Point *MyData;
	MyData = malloc(nEdges*sizeof(*MyData));
	center.x = 0;
	center.y = 0;
	for (i=0;i<nEdges;i++){
		center.x += Edges[i][0];
		center.y += Edges[i][1];
		MyData[i].x = Edges[i][0];
		MyData[i].y = Edges[i][1];
	}
	center.x /= nEdges;
	center.y /= nEdges;

	qsort(MyData, nEdges, sizeof(struct Point), cmpfunc);
	double **SortedEdges;
	SortedEdges = allocMatrix(nEdges+1,2);
	for (i=0;i<nEdges;i++){
		SortedEdges[i][0] = MyData[i].x;
		SortedEdges[i][1] = MyData[i].y;
	}
	SortedEdges[nEdges][0] = MyData[0].x;
	SortedEdges[nEdges][1] = MyData[0].y;

	double Area=0;
	for (i=0;i<nEdges;i++){
		Area += 0.5*((SortedEdges[i][0]*SortedEdges[i+1][1])-(SortedEdges[i+1][0]*SortedEdges[i][1]));
	}
	free(MyData);
	FreeMemMatrix(SortedEdges,nEdges+1);
	return Area;
}

static inline
int FindUniques (double **EdgesIn, double **EdgesOut, int nEdgesIn, double RMin, double RMax, double EtaMin, double EtaMax){
	int i,j, nEdgesOut=0, duplicate;
	double Len, RT,ET;
	for (i=0;i<nEdgesIn;i++){
		duplicate = 0;
		for (j=i+1;j<nEdgesIn;j++){
			Len = sqrt((EdgesIn[i][0]-EdgesIn[j][0])*(EdgesIn[i][0]-EdgesIn[j][0])+(EdgesIn[i][1]-EdgesIn[j][1])*(EdgesIn[i][1]-EdgesIn[j][1]));
			if (Len ==0){
				duplicate = 1;
			}
		}
		RT = sqrt(EdgesIn[i][0]*EdgesIn[i][0] + EdgesIn[i][1]*EdgesIn[i][1]);
		ET = CalcEtaAngle(EdgesIn[i][0],EdgesIn[i][1]);
		if (fabs(ET - EtaMin) > 180 || fabs(ET - EtaMax) > 180){
			if (EtaMin < 0) ET = ET - 360;
			else ET = 360 + ET;
		}
		if (BETWEEN(RT,RMin,RMax) == 0){
			duplicate = 1;
		}
		if (BETWEEN(ET,EtaMin,EtaMax) == 0){
			duplicate = 1;
		}
		if (duplicate == 0){
			EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
			EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
			nEdgesOut++;
		}
	}
	return nEdgesOut;
}

struct data {
	int y;
	int z;
	double frac;
};

static inline
long long int
mapperfcn(
	double tx,
	double ty,
	double tz,
	int NrPixelsY,
	int NrPixelsZ,
	double pxY,
	double pxZ,
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
	struct data ***pxList,
	int **nPxList,
	int **maxnPx)
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
	double *RetVals, *RetVals2;
	RetVals = malloc(2*sizeof(*RetVals));
	RetVals2 = malloc(2*sizeof(*RetVals2));
	double Y, Z, Eta, Rt;
	int i,j,k,l,m,n;
	double EtaMi, EtaMa, RMi, RMa;
	int RChosen[500], EtaChosen[500];
	int nrRChosen, nrEtaChosen;
	double EtaMiTr, EtaMaTr;
	double YZ[2];
	double **Edges;
	Edges = allocMatrix(50,2);
	double **EdgesOut;
	EdgesOut = allocMatrix(50,2);
	int nEdges;
	double RMin, RMax, EtaMin, EtaMax;
	double yMin, yMax, zMin, zMax;
	double boxEdge[4][2];
	double Area;
	double RThis, EtaThis;
	double yTemp, zTemp, yTempMin, yTempMax, zTempMin, zTempMax;
	int maxnVal, nVal;
	struct data *oldarr, *newarr;
	long long int TotNrOfBins = 0;
	long long int sumNrBins = 0;
	long long int nrContinued=0;
	long long int testPos;
	double ypr,zpr;
	double RT, ET;
	for (i=0;i<NrPixelsY;i++){
		for (j=0;j<NrPixelsZ;j++){
			EtaMi = 1800;
			EtaMa = -1800;
			RMi = 1E8; // In pixels
			RMa = -1000;
			// Calculate RMi, RMa, EtaMi, EtaMa
			testPos = j;
			testPos *= NrPixelsY;
			testPos += i;
			ypr = (double)i + distortionMapY[testPos];
			zpr = (double)j + distortionMapZ[testPos];
			for (k = 0; k < 2; k++){
				for (l = 0; l < 2; l++){
					Y = ypr + dy[k];
					Z = zpr + dz[l];
					REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
					Eta = RetVals[0];
					Rt = RetVals[1]; // in pixels
					if (Eta < EtaMi) EtaMi = Eta;
					if (Eta > EtaMa) EtaMa = Eta;
					if (Rt < RMi) RMi = Rt;
					if (Rt > RMa) RMa = Rt;
				}
			}
			// Get corrected Y, Z for this position.
			REta4MYZ(ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2, pxY, RetVals);
			Eta = RetVals[0];
			Rt = RetVals[1]; // in pixels
			YZ4mREta(Rt,Eta,RetVals2);
			YZ[0] = RetVals2[0]; // Corrected Y position according to R, Eta, center at 0,0
			YZ[1] = RetVals2[1]; // Corrected Z position according to R, Eta, center at 0,0
			// Now check which eta, R ranges should have this pixel
			nrRChosen = 0;
			nrEtaChosen = 0;
			for (k=0;k<nRBins;k++){
				if (  RBinsHigh[k] >=   RMi &&   RBinsLow[k] <=   RMa){
					RChosen[nrRChosen] = k;
					nrRChosen ++;
				}
			}
			for (k=0;k<nEtaBins;k++){ // If Eta is smaller than 0, check for eta, eta+360, if eta is greater than 0, check for eta, eta-360
				// First check if the pixel is a special case
				if (EtaMa - EtaMi > 180){
					EtaMiTr = EtaMa;
					EtaMaTr = 360 + EtaMi;
					EtaMa = EtaMaTr;
					EtaMi = EtaMiTr;
				}
				if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)){
					EtaChosen[nrEtaChosen] = k;
					nrEtaChosen++;
					continue;
				}
				if (EtaMi < 0){
					EtaMi += 360;
					EtaMa += 360;
				} else {
					EtaMi -= 360;
					EtaMa -= 360;
				}
				if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)){
					EtaChosen[nrEtaChosen] = k;
					nrEtaChosen++;
					continue;
				}
			}
			yMin = YZ[0] - 0.5;
			yMax = YZ[0] + 0.5;
			zMin = YZ[1] - 0.5;
			zMax = YZ[1] + 0.5;
			sumNrBins += nrRChosen * nrEtaChosen;
			double totPxArea = 0;
			// printf("%d\n",nrEtaChosen);
			// Line Intercepts ordering: RMin: ymin, ymax, zmin, zmax. RMax: ymin, ymax, zmin, zmax
			//							 EtaMin: ymin, ymax, zmin, zmax. EtaMax: ymin, ymax, zmin, zmax.
			for (k=0;k<nrRChosen;k++){
				RMin = RBinsLow[RChosen[k]];
				RMax = RBinsHigh[RChosen[k]];
				for (l=0;l<nrEtaChosen;l++){
					EtaMin = EtaBinsLow[EtaChosen[l]];
					EtaMax = EtaBinsHigh[EtaChosen[l]];
					// Find YZ of the polar mask.
					YZ4mREta(RMin,EtaMin,RetVals);
					boxEdge[0][0] = RetVals[0];
					boxEdge[0][1] = RetVals[1];
					YZ4mREta(RMin,EtaMax,RetVals);
					boxEdge[1][0] = RetVals[0];
					boxEdge[1][1] = RetVals[1];
					YZ4mREta(RMax,EtaMin,RetVals);
					boxEdge[2][0] = RetVals[0];
					boxEdge[2][1] = RetVals[1];
					YZ4mREta(RMax,EtaMax,RetVals);
					boxEdge[3][0] = RetVals[0];
					boxEdge[3][1] = RetVals[1];
					nEdges = 0;
					// Now check if any edge of the pixel is within the polar mask
					for (m=0;m<4;m++){
						RThis = sqrt((YZ[0]+PosMatrix[m][0])*(YZ[0]+PosMatrix[m][0])+(YZ[1]+PosMatrix[m][1])*(YZ[1]+PosMatrix[m][1]));
						EtaThis = CalcEtaAngle(YZ[0]+PosMatrix[m][0],YZ[1]+PosMatrix[m][1]);
						if (EtaMin < -180 && signVal(EtaThis) != signVal(EtaMin)) EtaThis -= 360;
						if (EtaMax >  180 && signVal(EtaThis) != signVal(EtaMax)) EtaThis += 360;
						if (RThis   >= RMin   && RThis   <= RMax &&
							EtaThis >= EtaMin && EtaThis <= EtaMax){
							Edges[nEdges][0] = YZ[0]+PosMatrix[m][0];
							Edges[nEdges][1] = YZ[1]+PosMatrix[m][1];
							nEdges++;
						}
					}
					for (m=0;m<4;m++){ // Check if any edge of the polar mask is within the pixel edges.
						if (boxEdge[m][0] >= yMin && boxEdge[m][0] <= yMax &&
							boxEdge[m][1] >= zMin && boxEdge[m][1] <= zMax){
								Edges[nEdges][0] = boxEdge[m][0];
								Edges[nEdges][1] = boxEdge[m][1];
								nEdges ++;
							}
					}
					if (nEdges < 4){
						// Now go through Rmin, Rmax, EtaMin, EtaMax and calculate intercepts and check if within the pixel.
						//RMin,Max and yMin,Max
						if (RMin >= yMin) {
							zTemp = signVal(YZ[1])*sqrt(RMin*RMin - yMin*yMin);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMin;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMin >= yMax) {
							zTemp = signVal(YZ[1])*sqrt(RMin*RMin - yMax*yMax);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMax;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMax >= yMin) {
							zTemp = signVal(YZ[1])*sqrt(RMax*RMax - yMin*yMin);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMin;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						if (RMax >= yMax) {
							zTemp = signVal(YZ[1])*sqrt(RMax*RMax - yMax*yMax);
							if (BETWEEN(zTemp,zMin,zMax) == 1){
								Edges[nEdges][0] = yMax;
								Edges[nEdges][1] = zTemp;
								nEdges++;
							}
						}
						//RMin,Max and zMin,Max
						if (RMin >= zMin) {
							yTemp = signVal(YZ[0])*sqrt(RMin*RMin - zMin*zMin);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMin;
								nEdges++;
							}
						}
						if (RMin >= zMax) {
							yTemp = signVal(YZ[0])*sqrt(RMin*RMin - zMax*zMax);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMax;
								nEdges++;
							}
						}
						if (RMax >= zMin) {
							yTemp = signVal(YZ[0])*sqrt(RMax*RMax - zMin*zMin);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMin;
								nEdges++;
							}
						}
						if (RMax >= zMax) {
							yTemp = signVal(YZ[0])*sqrt(RMax*RMax - zMax*zMax);
							if (BETWEEN(yTemp,yMin,yMax) == 1){
								Edges[nEdges][0] = yTemp;
								Edges[nEdges][1] = zMax;
								nEdges++;
							}
						}
						//EtaMin,Max and yMin,Max
						if (fabs(EtaMin) < 1E-5 || fabs(fabs(EtaMin)-180) < 1E-5){
							zTempMin = 0;
							zTempMax = 0;
						}else{
							zTempMin = -yMin/tan(EtaMin*deg2rad);
							zTempMax = -yMax/tan(EtaMin*deg2rad);
						}
						if (BETWEEN(zTempMin,zMin,zMax) == 1){
							Edges[nEdges][0] = yMin;
							Edges[nEdges][1] = zTempMin;
							nEdges++;
						}
						if (BETWEEN(zTempMax,zMin,zMax) == 1){
							Edges[nEdges][0] = yMax;
							Edges[nEdges][1] = zTempMax;
							nEdges++;
						}
						if (fabs(EtaMax) < 1E-5 || fabs(fabs(EtaMax)-180) < 1E-5){
							zTempMin = 0;
							zTempMax = 0;
						}else{
							zTempMin = -yMin/tan(EtaMax*deg2rad);
							zTempMax = -yMax/tan(EtaMax*deg2rad);
						}
						if (BETWEEN(zTempMin,zMin,zMax) == 1){
							Edges[nEdges][0] = yMin;
							Edges[nEdges][1] = zTempMin;
							nEdges++;
						}
						if (BETWEEN(zTempMax,zMin,zMax) == 1){
							Edges[nEdges][0] = yMax;
							Edges[nEdges][1] = zTempMax;
							nEdges++;
						}
						//EtaMin,Max and zMin,Max
						if (fabs(fabs(EtaMin)-90) < 1E-5){
							yTempMin = 0;
							yTempMax = 0;
						}else{
							yTempMin = -zMin*tan(EtaMin*deg2rad);
							yTempMax = -zMax*tan(EtaMin*deg2rad);
						}
						if (BETWEEN(yTempMin,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMin;
							Edges[nEdges][1] = zMin;
							nEdges++;
						}
						if (BETWEEN(yTempMax,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMax;
							Edges[nEdges][1] = zMax;
							nEdges++;
						}
						if (fabs(fabs(EtaMax)-90) < 1E-5){
							yTempMin = 0;
							yTempMax = 0;
						}else{
							yTempMin = -zMin*tan(EtaMax*deg2rad);
							yTempMax = -zMax*tan(EtaMax*deg2rad);
						}
						if (BETWEEN(yTempMin,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMin;
							Edges[nEdges][1] = zMin;
							nEdges++;
						}
						if (BETWEEN(yTempMax,yMin,yMax) == 1){
							Edges[nEdges][0] = yTempMax;
							Edges[nEdges][1] = zMax;
							nEdges++;
						}
					}
					if (nEdges < 3){
						nrContinued++;
						continue;
					}
					nEdges = FindUniques(Edges,EdgesOut,nEdges,RMin,RMax,EtaMin,EtaMax);
					if (nEdges < 3){
						nrContinued++;
						continue;
					}
					// Now we have all the edges, let's calculate the area.
					Area = CalcAreaPolygon(EdgesOut,nEdges);
					if (Area < 1E-5){
						nrContinued++;
						continue;
					}
					// Populate the arrays
					maxnVal = maxnPx[RChosen[k]][EtaChosen[l]];
					nVal = nPxList[RChosen[k]][EtaChosen[l]];
					if (nVal >= maxnVal){
						maxnVal += 2;
						oldarr = pxList[RChosen[k]][EtaChosen[l]];
						newarr = realloc(oldarr, maxnVal*sizeof(*newarr));
						if (newarr == NULL){
							return 0;
						}
						pxList[RChosen[k]][EtaChosen[l]] = newarr;
						maxnPx[RChosen[k]][EtaChosen[l]] = maxnVal;
					}
					pxList[RChosen[k]][EtaChosen[l]][nVal].y = i;
					pxList[RChosen[k]][EtaChosen[l]][nVal].z = j;
					pxList[RChosen[k]][EtaChosen[l]][nVal].frac = Area;
					totPxArea += Area;
					(nPxList[RChosen[k]][EtaChosen[l]])++;
					TotNrOfBins++;
				}
			}
		}
	}
	return TotNrOfBins;
}

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], double *ImageIn, double *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	int i,j,k,l,m;
	if (NrTransOpt == 0){
		memcpy(ImageOut,ImageIn,NrPixelsY*NrPixelsZ*sizeof(*ImageIn)); // Nothing to do
		return;
	}
    for (i=0;i<NrTransOpt;i++){
		if (TransOpt[i] == 1){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[l*NrPixelsY+(NrPixelsY-k-1)]; // Invert Y
				}
			}
		}else if (TransOpt[i] == 2){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[(NrPixelsZ-l-1)*NrPixelsY+k]; // Invert Z
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    char *ParamFN;
    FILE *paramFile;
    ParamFN = argv[1];
    if (argc != 2){
		printf("******************Supply a parameter file as argument.******************\n"
		"Parameters needed: tx, ty, tz, px, BC, Lsd, RhoD,"
		"\n\t\t   p0, p1, p2, EtaBinSize, EtaMin,\n\t\t   EtaMax, RBinSize, RMin, RMax,\n\t\t   NrPixels\n");
		return(1);
	}
	double tx=0.0, ty=0.0, tz=0.0, pxY=200.0, pxZ=200.0, yCen=1024.0, zCen=1024.0, Lsd=1000000.0, RhoD=200000.0,
		p0=0.0, p1=0.0, p2=0.0, p3=0.0, EtaBinSize=5, RBinSize=0.25, RMax=1524.0, RMin=10.0, EtaMax=180.0, EtaMin=-180.0;
	int NrPixelsY=2048, NrPixelsZ=2048;
	char aline[4096], dummy[4096], *str;
	distortionFile = 0;
	char distortionFN[4096];
	int NrTransOpt=0;
	int TransOpt[10];
	paramFile = fopen(ParamFN,"r");
	while (fgets(aline,4096,paramFile) != NULL){
		str = "tx ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &tx);
		}
		str = "ty ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &ty);
		}
		str = "tz ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &tz);
		}
		str = "pxY ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &pxY);
		}
		str = "pxZ ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &pxZ);
		}
		str = "px ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &pxY);
			sscanf(aline,"%s %lf", dummy, &pxZ);
		}
		str = "BC ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf %lf", dummy, &yCen, &zCen);
		}
		str = "Lsd ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Lsd);
		}
		str = "RhoD ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RhoD);
		}
		str = "p0 ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &p0);
		}
		str = "p1 ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &p1);
		}
		str = "p2 ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &p2);
		}
		str = "p3 ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &p3);
		}
		str = "EtaBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaBinSize);
		}
		str = "RBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RBinSize);
		}
		str = "RMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMax);
		}
		str = "RMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMin);
		}
		str = "EtaMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMax);
		}
		str = "EtaMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMin);
		}
		str = "NrPixelsY ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
		}
		str = "NrPixelsZ ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "NrPixels ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "DistortionFile ";
		if (StartsWith(aline,str)==1){
			distortionFile = 1;
			sscanf(aline,"%s %s",dummy, distortionFN);
		}
        str = "ImTransOpt ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
	}
	distortionMapY = calloc(NrPixelsY*NrPixelsZ,sizeof(double));
	distortionMapZ = calloc(NrPixelsY*NrPixelsZ,sizeof(double));
	if (distortionFile == 1){
		FILE *distortionFileHandle = fopen(distortionFN,"rb");
		double *distortionMapTemp;
		distortionMapTemp = malloc(NrPixelsY*NrPixelsZ*sizeof(double));
		fread(distortionMapTemp,NrPixelsY*NrPixelsZ*sizeof(double),1,distortionFileHandle);
		DoImageTransformations(NrTransOpt,TransOpt,distortionMapTemp,distortionMapY,NrPixelsY,NrPixelsZ);
		fread(distortionMapTemp,NrPixelsY*NrPixelsZ*sizeof(double),1,distortionFileHandle);
		DoImageTransformations(NrTransOpt,TransOpt,distortionMapTemp,distortionMapZ,NrPixelsY,NrPixelsZ);
		printf("Distortion file %s was provided and read correctly.\n",distortionFN);
	}
    // Parameters needed: Rmax RMin RBinSize (px) EtaMax EtaMin EtaBinSize (degrees)
	int nEtaBins, nRBins;
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int)ceil((EtaMax - EtaMin)/EtaBinSize);
	printf("Creating a mapper for integration.\nNumber of eta bins: %d, number of R bins: %d.\n",nEtaBins,nRBins);
	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = malloc(nRBins*sizeof(*RBinsHigh));
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);
	// Initialize arrays, need fraction array
	struct data ***pxList;
	int **nPxList;
	int **maxnPx;
	pxList = malloc(nRBins * sizeof(pxList));
	nPxList = malloc(nRBins * sizeof(nPxList));
	maxnPx = malloc(nRBins * sizeof(maxnPx));
	int i,j,k,l;
	for (i=0;i<nEtaBins;i++) printf("%lf %lf \n",EtaBinsHigh[i],EtaBinsLow[i]);
	for (i=0;i<nRBins;i++){
		pxList[i] = malloc(nEtaBins*sizeof(pxList[i]));
		nPxList[i] = malloc(nEtaBins*sizeof(nPxList[i]));
		maxnPx[i] = malloc(nEtaBins*sizeof(maxnPx[i]));
		for (j=0;j<nEtaBins;j++){
			pxList[i][j] = NULL;
			nPxList[i][j] = 0;
			maxnPx[i][j] = 0;
		}
	}
    // Parameters needed: tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen, Lsd, RhoD, p0, p1, p2
    long long int TotNrOfBins = mapperfcn(tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen,
								zCen, Lsd, RhoD, p0, p1, p2, p3, EtaBinsLow,
								EtaBinsHigh, RBinsLow, RBinsHigh, nRBins,
								nEtaBins, pxList, nPxList, maxnPx);
	printf("Total Number of bins %lld\n",TotNrOfBins); fflush(stdout);
	long long int LengthNPxList = nRBins * nEtaBins;
	struct data *pxListStore;
	int *nPxListStore;
	pxListStore = malloc(TotNrOfBins*sizeof(*pxListStore));
	nPxListStore = malloc(LengthNPxList*2*sizeof(*nPxListStore));
	long long int Pos;
	int localNPxVal, localCounter = 0;
	for (i=0;i<nRBins;i++){
		for (j=0;j<nEtaBins;j++){
			localNPxVal = nPxList[i][j];
			Pos = i*nEtaBins;
			Pos += j;
			nPxListStore[(Pos*2)+0] = localNPxVal;
			nPxListStore[(Pos*2)+1] = localCounter;
			for (k=0;k<localNPxVal;k++){
				pxListStore[localCounter+k].y = pxList[i][j][k].y;
				pxListStore[localCounter+k].z = pxList[i][j][k].z;
				pxListStore[localCounter+k].frac = pxList[i][j][k].frac;
			}
			localCounter += localNPxVal;
		}
	}

	// Write out
	char *mapfn = "Map.bin";
	char *nmapfn = "nMap.bin";
	FILE *mapfile = fopen(mapfn,"wb");
	FILE *nmapfile = fopen(nmapfn,"wb");
	fwrite(pxListStore,TotNrOfBins*sizeof(*pxListStore),1,mapfile);
	fwrite(nPxListStore,LengthNPxList*2*sizeof(*nPxListStore),1,nmapfile);

	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
}
