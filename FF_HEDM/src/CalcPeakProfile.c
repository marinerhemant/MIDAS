//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
// CalcPeakProfile.c
//
// Created by Hemant Sharma on 2014/07/26
//

#include <stdio.h>
#include <math.h>
#include <malloc.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
extern size_t mapMaskSize;
extern int *mapMask;

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
FreeMemMatrix(
    double **mat,
    int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}


double PosMatrix[4][2]={{-0.5, -0.5},
						{-0.5,  0.5},
						{ 0.5,  0.5},
						{ 0.5, -0.5}};

#define FindSmaller(Val1,Val2)  Val1 > Val2 ? Val2 : Val1
#define FindLarger(Val1,Val2)  Val1 > Val2 ? Val1 : Val2
#define Len2d(x,y) sqrt(x*x + y*y)

static inline
int FindUniques (double **EdgesIn, double **EdgesOut, int nEdgesIn){
	int i,j, nEdgesOut=0, duplicate;
	double Len;
	for (i=0;i<nEdgesIn;i++){
		duplicate = 0;
		for (j=i+1;j<nEdgesIn;j++){
			Len = Len2d((EdgesIn[i][0]-EdgesIn[j][0]),(EdgesIn[i][1]-EdgesIn[j][1]));
			if (Len ==0){
				duplicate = 1;
			}
		}
		if (duplicate == 0){
			EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
			EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
			nEdgesOut++;
		}
	}
	return nEdgesOut;
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
int CalcNEdges(double **BoxEdges, int *Pos, double **Edges) // Box Edges 
					//should be 5, first should be repeated to the end
{
	int i,j;
	double Px[4],Py[4];
	for (i=0;i<4;i++){
		Px[i] = Pos[0] + PosMatrix[i][0];
		Py[i] = Pos[1] + PosMatrix[i][1];
	}
	int nEdges =0;
	for (i=0;i<4;i++){ // If the pixel is completely inside the box
		if (BoxEdges[i][0] >= Px[0] && BoxEdges[i][0] <= Px[2] &&
			BoxEdges[i][1] >= Py[0] && BoxEdges[i][1] <= Py[2]){
			Edges[nEdges][0] = BoxEdges[i][0];
			Edges[nEdges][1] = BoxEdges[i][1];
			nEdges++;
		}
	}
	if (nEdges == 4){
		return nEdges;
	}
	double XIntersect, YIntersect, XP1, XP2, YP1, YP2, M, YP, XP;
	double SmallX, LargeX, SmallY, LargeY;
	double Intersects[10][2];
	int nIntersects;
	double Temp1,Temp2;
	for (i=0;i<4;i++){
		XP = Px[i];
		YP = Py[i];
		nIntersects = 0;
		Temp1 = 1;
		Temp2 = 1;
		for (j=0;j<4;j++){
			XP1 = BoxEdges[j][0];
			XP2 = BoxEdges[j+1][0];
			YP1 = BoxEdges[j][1];
			YP2 = BoxEdges[j+1][1];
			SmallX = FindSmaller(XP1,XP2);
			LargeX = FindLarger(XP1,XP2);
			SmallY = FindSmaller(YP1,YP2);
			LargeY = FindLarger(YP1,YP2);
			if (fabs(XP1 - XP2) < 1E-5){
				if (YP > SmallY && YP < LargeY){
					Intersects[nIntersects][0] = XP1;
					Intersects[nIntersects][1] = YP;
					nIntersects++;
				}
				continue;
			}
			if (fabs(YP1 - YP2) < 1E-5){
				if (XP > SmallX && XP < LargeX){
					Intersects[nIntersects][0] = XP;
					Intersects[nIntersects][1] = YP1;
					nIntersects++;
				}
				continue;
			}
			M = (YP2 - YP1) / (XP2 - XP1);
			YIntersect = (XP - XP1)*M + YP1;
			XIntersect = (YP - YP1)/M + XP1;
			if (YIntersect > SmallY && YIntersect < LargeY){
				Intersects[nIntersects][0] = XP;
				Intersects[nIntersects][1] = YIntersect;
				nIntersects++;
			}
			if (XIntersect > SmallX && XIntersect < LargeX){
				Intersects[nIntersects][0] = XIntersect;
				Intersects[nIntersects][1] = YP;
				nIntersects++;
				}
		}
		if (nIntersects == 4){
			for (j=0;j<4;j++){
				if (Intersects[j][0]-XP == 0){
					Temp1 *= Intersects[j][1]-YP;
				}else if (Intersects[j][1]-YP == 0){
					Temp2 *= Intersects[j][0]-XP;
				}
			}
			if (Temp1 < 0 && Temp2 < 0){
				Edges[nEdges][0] = XP;
				Edges[nEdges][1] = YP;
				nEdges++;
			}
		}
	}
	for (j=0;j<4;j++){
		XP1 = BoxEdges[j][0];
		XP2 = BoxEdges[j+1][0];
		YP1 = BoxEdges[j][1];
		YP2 = BoxEdges[j+1][1];
		SmallX = FindSmaller(XP1,XP2);
		LargeX = FindLarger(XP1,XP2);
		SmallY = FindSmaller(YP1,YP2);
		LargeY = FindLarger(YP1,YP2);
		
		XIntersect = Px[0];  // Case 1: taking 1 and 2
		if (fabs(XP1 - XP2) < 1E-5){ // Special case, vertical line
			if (fabs(XIntersect - XP1) < 1E-5) {
				if (Py[0] <= LargeY && Py[0] >= SmallY){
					Edges[nEdges][0] = XIntersect;
					Edges[nEdges][1] = Py[0];
					nEdges++;
				}
				if (Py[1] <= LargeY && Py[1] >= SmallY){
					Edges[nEdges][0] = XIntersect;
					Edges[nEdges][1] = Py[1];
					nEdges++;
				}
			}
		} else {
			M = (YP2 - YP1) / (XP2 - XP1);
			YIntersect = (XIntersect - XP1)*M + YP1;
			if ( YIntersect >= Py[0] && YIntersect <= Py[1] 
			  && YIntersect >= SmallY && YIntersect <= LargeY){
				Edges[nEdges][0] = XIntersect;
				Edges[nEdges][1] = YIntersect;
				nEdges++;
			}
		}
		
		XIntersect = Px[2];  // Case 2: taking 3 and 4
		if (fabs(XP1 - XP2) < 1E-5){ // Special case, vertical line
			if (fabs(XIntersect - XP1) < 1E-5) {
				if (Py[2] <= LargeY && Py[2] >= SmallY){
					Edges[nEdges][0] = XIntersect;
					Edges[nEdges][1] = Py[2];
					nEdges++;
				}
				if (Py[3] <= LargeY && Py[3] >= SmallY){
					Edges[nEdges][0] = XIntersect;
					Edges[nEdges][1] = Py[3];
					nEdges ++;
				}
			}
		} else {
			M = (YP2 - YP1) / (XP2 - XP1);
			YIntersect = (XIntersect - XP1)*M + YP1;
			if ( YIntersect >= Py[3] && YIntersect <= Py[2]
			  && YIntersect >= SmallY && YIntersect <= LargeY){
				Edges[nEdges][0] = XIntersect;
				Edges[nEdges][1] = YIntersect;
				nEdges++;
			}
		}
		
		YIntersect = Py[1];  // Case 3: taking 2 and 3
		if (fabs(YP1 - YP2) < 1E-5){ // Special case, horizontal line
			if (fabs(YIntersect - YP1) < 1E-5) {
				if (Px[1] <= LargeX && Px[1] >= SmallX){
					Edges[nEdges][1] = YIntersect;
					Edges[nEdges][0] = Px[1];
					nEdges++;
				}
				if (Px[2] < LargeX && Px[2] > SmallX){
					Edges[nEdges][1] = YIntersect;
					Edges[nEdges][0] = Px[2];
					nEdges ++;
				}
			}
		} else {
			M = (YP2 - YP1) / (XP2 - XP1);
			XIntersect = (YIntersect - YP1)/M + XP1;
			if ( XIntersect >= Px[1] && XIntersect <= Px[2]
			  && XIntersect >= SmallX && XIntersect <= LargeX){
				Edges[nEdges][0] = XIntersect;
				Edges[nEdges][1] = YIntersect;
				nEdges++;
			}
		}
		YIntersect = Py[0];  // Case 4: taking 1 and 4
		if (fabs(YP1 - YP2) < 1E-5){ // Special case, horizontal line
			if (fabs(YIntersect - YP1) < 1E-5) {
				if (Px[0] <= LargeX && Px[0] >= SmallX){
					Edges[nEdges][1] = YIntersect;
					Edges[nEdges][0] = Px[0];
					nEdges++;
				}
				if (Px[3] <= LargeX && Px[3] >= SmallX){
					Edges[nEdges][1] = YIntersect;
					Edges[nEdges][0] = Px[3];
					nEdges ++;
				}
			}
		} else {
			M = (YP2 - YP1) / (XP2 - XP1);
			XIntersect = (YIntersect - YP1)/M + XP1;
			if ( XIntersect > Px[0] && XIntersect < Px[3]
			  && XIntersect >= SmallX && XIntersect <= LargeX){
				Edges[nEdges][0] = XIntersect;
				Edges[nEdges][1] = YIntersect;
				nEdges++;
			}
		}
	}

	return nEdges;
}

static inline
void YZMat4mREta(int NrElements, double *R, double *Eta, double **YZ, double ybc, double zbc, double px){
	int i;
	for (i=0;i<NrElements;i++){
		YZ[i][0] = -R[i]*sin(Eta[i]*deg2rad);
		YZ[i][1] = R[i]*cos(Eta[i]*deg2rad);
		YZ[i][0] = ybc-(YZ[i][0]/px);
		YZ[i][1] = zbc+(YZ[i][1]/px);
	}
}

inline void CalcPeakProfile(int **Indices, int *NrEachIndexBin, int idx,
	double *Average,double Rmi,double Rma,double EtaMi,double EtaMa,
	double ybc,double zbc,double px,int NrPixelsY, double *ReturnValue)
{
	double **BoxEdges,*RMs, *EtaMs, **EdgesIn, **EdgesOut;
	double SumIntensity=0;
	BoxEdges = allocMatrix(5,2);
	EdgesIn = allocMatrix(10,2);
	EdgesOut = allocMatrix(10,2);
	RMs = malloc(5*sizeof(*RMs));
	EtaMs = malloc(5*sizeof(*EtaMs));
	RMs[0] = Rmi; EtaMs[0] = EtaMi;
	RMs[1] = Rma; EtaMs[1] = EtaMi;
	RMs[2] = Rma; EtaMs[2] = EtaMa;
	RMs[3] = Rmi; EtaMs[3] = EtaMa;
	RMs[4] = Rmi; EtaMs[4] = EtaMi;
	YZMat4mREta(5,RMs,EtaMs,BoxEdges,ybc,zbc,px);
	int i;
	int *Pos,nEdges=0;
	Pos = malloc(2*sizeof(*Pos));
	double ThisArea, TotArea=0;
	for (i=0;i<NrEachIndexBin[idx];i++){
		if (mapMaskSize !=0){ // Skip this point if it was on the badPx, gap mask
			if (TestBit(mapMask,Indices[idx][i])){
				//printf("%d %d %d\n",idx,i,Indices[idx][i]);
				continue;
			}
		}
		Pos[0] = Indices[idx][i] % NrPixelsY; // This is Y
		Pos[1] = Indices[idx][i] / NrPixelsY; // This is Z
		nEdges = CalcNEdges(BoxEdges,Pos,EdgesIn);
		if (nEdges == 0){
			continue;
		}
		nEdges = FindUniques(EdgesIn,EdgesOut,nEdges);
		ThisArea = CalcAreaPolygon(EdgesOut,nEdges);
		TotArea += ThisArea;
		//~ printf("%lf %lf %d %d %d\n",TotArea, ThisArea, idx,i,Indices[idx][i]);
		printf("%lf ",Average[Indices[idx][i]]);
		SumIntensity += Average[Indices[idx][i]] * ThisArea;
	}
	SumIntensity /= TotArea;
	printf("%lf %lf ",SumIntensity,TotArea);
	if (TotArea == 0){
		SumIntensity = 0;
	}
	free(Pos);
	FreeMemMatrix(EdgesIn,10);
	FreeMemMatrix(EdgesOut,10);
	free(RMs);
	free(EtaMs);
	FreeMemMatrix(BoxEdges,5);
	*ReturnValue = SumIntensity;
	printf("%lf %lf ",SumIntensity,*ReturnValue);
}
