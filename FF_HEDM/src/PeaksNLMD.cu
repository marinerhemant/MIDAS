//
//  Peaks.cu
//
//
//  Created by Hemant Sharma on 2015/07/04.
//

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAX_LINE_LENGTH 10240
#define MAX_N_RINGS 5000
#define MAX_N_EVALS 3000
#define MAX_N_OVERLAPS 20
#define nOverlapsMaxPerImage 10000
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
#define CHECK(call){														\
	const cudaError_t error = call;											\
	if (error != cudaSuccess){												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(-10*error);													\
	}																		\
}
typedef uint16_t pixelvalue;

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

//BEGIN NLDRMD FUNCTION
__device__ void nelmin ( double fn ( int n_fun, double *x, void *data ), 
  int n, double *start, double *xmin, 
  double *lb, double *ub, double *scratch, double *ynewlo, 
  double reqmin, double *step, int konvge, int kcount, 
  int *icount, int *numres, int *ifault, void *data_t)
{
  double ccoeff = 0.5;
  double del;
  double dn;
  double dnn;
  double ecoeff = 2.0;
  double eps = 0.001;
  int i;
  int ihi;
  int ilo;
  int j;
  int jcount;
  int l;
  int nn;
  double *p;
  double *p2star;
  double *pbar;
  double *pstar;
  double rcoeff = 1.0;
  double rq;
  double x;
  double *y;
  double y2star;
  double ylo;
  double ystar;
  double z;
/*
  Check the input parameters.
*/
  if ( reqmin <= 0.0 )
  {
    *ifault = 1;
    return;
  }

  if ( n < 1 )
  {
    *ifault = 1;
    return;
  }

  if ( konvge < 1 )
  {
    *ifault = 1;
    return;
  }

  p = scratch;
  pstar = p + n*(n+1);
  p2star = pstar + n;
  pbar = p2star + n;
  y = pbar + n;

  *icount = 0;
  *numres = 0;

  jcount = konvge; 
  dn = ( double ) ( n );
  nn = n + 1;
  dnn = ( double ) ( nn );
  del = 1.0;
  rq = reqmin * dn;
/*
  Initial or restarted loop.
*/
  for ( ; ; )
  {
    for ( i = 0; i < n; i++ )
    { 
      p[i+n*n] = start[i];
    }
    y[n] = fn ( n, start, data_t );
    *icount = *icount + 1;

    for ( j = 0; j < n; j++ )
    {
      x = start[j];
      start[j] = start[j] + step[j] * del;
      if (start[j] < lb[j]) start[j] = lb[j]; // Constraints
      if (start[j] > ub[j]) start[j] = ub[j]; // Constraints
      for ( i = 0; i < n; i++ )
      {
        p[i+j*n] = start[i];
      }
      y[j] = fn ( n, start, data_t );
      *icount = *icount + 1;
      start[j] = x;
    }
/*                 
  The simplex construction is complete.
                    
  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
  the vertex of the simplex to be replaced.
*/                
    ylo = y[0];
    ilo = 0;

    for ( i = 1; i < nn; i++ )
    {
      if ( y[i] < ylo )
      {
        ylo = y[i];
        ilo = i;
      }
    }
/*
  Inner loop.
*/
    for ( ; ; )
    {
      if ( kcount <= *icount )
      {
        break;
      }
      *ynewlo = y[0];
      ihi = 0;

      for ( i = 1; i < nn; i++ )
      {
        if ( *ynewlo < y[i] )
        {
          *ynewlo = y[i];
          ihi = i;
        }
      }
/*
  Calculate PBAR, the centroid of the simplex vertices
  excepting the vertex with Y value YNEWLO.
*/
      for ( i = 0; i < n; i++ )
      {
        z = 0.0;
        for ( j = 0; j < nn; j++ )
        { 
          z = z + p[i+j*n];
        }
        z = z - p[i+ihi*n];  
        pbar[i] = z / dn;
      }
/*
  Reflection through the centroid.
*/
      for ( i = 0; i < n; i++ )
      {
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
        if (pstar[i] < lb[i]) pstar[i] = lb[i]; // Constraints
        if (pstar[i] > ub[i]) pstar[i] = ub[i]; // Constraints
      }
      ystar = fn ( n, pstar, data_t );
      *icount = *icount + 1;
/*
  Successful reflection, so extension.
*/
      if ( ystar < ylo )
      {
        for ( i = 0; i < n; i++ )
        {
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
          if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
          if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
        }
        y2star = fn ( n, p2star, data_t );
        *icount = *icount + 1;
/*
  Check extension.
*/
        if ( ystar < y2star )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Retain extension or contraction.
*/
        else
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = p2star[i];
          }
          y[ihi] = y2star;
        }
      }
/*
  No extension.
*/
      else
      {
        l = 0;
        for ( i = 0; i < nn; i++ )
        {
          if ( ystar < y[i] )
          {
            l = l + 1;
          }
        }

        if ( 1 < l )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Contraction on the Y(IHI) side of the centroid.
*/
        else if ( l == 0 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
            if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
            if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
          }
          y2star = fn ( n, p2star, data_t );
          *icount = *icount + 1;
/*
  Contract the whole simplex.
*/
          if ( y[ihi] < y2star )
          {
            for ( j = 0; j < nn; j++ )
            {
              for ( i = 0; i < n; i++ )
              {
                p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
                xmin[i] = p[i+j*n];
                if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
                if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
              }
              y[j] = fn ( n, xmin, data_t );
              *icount = *icount + 1;
            }
            ylo = y[0];
            ilo = 0;

            for ( i = 1; i < nn; i++ )
            {
              if ( y[i] < ylo )
              {
                ylo = y[i];
                ilo = i;
              }
            }
            continue;
          }
/*
  Retain contraction.
*/
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
/*
  Contraction on the reflection side of the centroid.
*/
        else if ( l == 1 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
            if (p2star[i] < lb[i]) p2star[i] = lb[i]; // Constraints
            if (p2star[i] > ub[i]) p2star[i] = ub[i]; // Constraints
          }
          y2star = fn ( n, p2star, data_t );
          *icount = *icount + 1;
/*
  Retain reflection?
*/
          if ( y2star <= ystar )
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = pstar[i];
            }
            y[ihi] = ystar;
          }
        }
      }
/*
  Check if YLO improved.
*/
      if ( y[ihi] < ylo )
      {
        ylo = y[ihi];
        ilo = ihi;
      }
      jcount = jcount - 1;

      if ( 0 < jcount )
      {
        continue;
      }
/*
  Check to see if minimum reached.
*/
      if ( *icount <= kcount )
      {
        jcount = konvge;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + y[i];
        }
        x = z / dnn;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + pow ( y[i] - x, 2 );
        }

        if ( z <= rq )
        {
          break;
        }
      }
    }
/*
  Factorial tests to check that YNEWLO is a local minimum.
*/
    for ( i = 0; i < n; i++ )
    {
      xmin[i] = p[i+ilo*n];
    }
    *ynewlo = y[ilo];

    if ( kcount < *icount )
    {
      *ifault = 2;
      break;
    }

    *ifault = 0;

    for ( i = 0; i < n; i++ )
    {
      del = step[i] * eps;
      xmin[i] = xmin[i] + del;
      if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
      if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
      z = fn ( n, xmin, data_t );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] - del - del;
      if (xmin[i] < lb[i]) xmin[i] = lb[i]; // Constraints
      if (xmin[i] > ub[i]) xmin[i] = ub[i]; // Constraints
      z = fn ( n, xmin, data_t );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] + del;
    }

    if ( *ifault == 0 )
    {
      break;
    }
/*
  Restart the procedure.
*/
    for ( i = 0; i < n; i++ )
    {
      start[i] = xmin[i];
    }
    del = eps;
    *numres = *numres + 1;
  }
  return;
}
//END NLDRMD FUNCTION

static inline pixelvalue** allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
    int i;
    arr = (pixelvalue **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (pixelvalue *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline int** allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = (int **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (int *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}


static inline void FreeMemMatrixPx(pixelvalue **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *Image, int NrPixels)
{
	int i,j,k,l,m;
    pixelvalue **ImageTemp1, **ImageTemp2;
    ImageTemp1 = allocMatrixPX(NrPixels,NrPixels);
    ImageTemp2 = allocMatrixPX(NrPixels,NrPixels);
	for (k=0;k<NrPixels;k++) {
		for (l=0;l<NrPixels;l++) {
			ImageTemp1[k][l] = Image[(NrPixels*k)+l];
		}
	}
	for (k=0;k<NrTransOpt;k++) {
		if (TransOpt[k] == 1){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][NrPixels-m-1]; //Inverting Y.
		} else if (TransOpt[k] == 2){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[NrPixels-l-1][m]; //Inverting Z.
		} else if (TransOpt[k] == 3){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[m][l];
		} else if (TransOpt[k] == 0){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][m];
		}
		for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp1[l][m] = ImageTemp2[l][m];
	}
	for (k=0;k<NrPixels;k++) for (l=0;l<NrPixels;l++) Image[(NrPixels*k)+l] = ImageTemp2[k][l];
	FreeMemMatrixPx(ImageTemp1,NrPixels);
	FreeMemMatrixPx(ImageTemp2,NrPixels);
}

static inline void Transposer (double *x, int n, double *y)
{
	int i,j;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			y[(i*n)+j] = x[(j*n)+i];
		}
	}
}

const int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
const int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

static inline void DepthFirstSearch(int x, int y, int current_label, int NrPixels, int **BoolImage, int **ConnectedComponents,int **Positions, int *PositionTrackers)
{
	if (x < 0 || x == NrPixels) return;
	if (y < 0 || y == NrPixels) return;
	if ((ConnectedComponents[x][y]!=0)||(BoolImage[x][y]==0)) return;
	
	ConnectedComponents[x][y] = current_label;
	Positions[current_label][PositionTrackers[current_label]] = (x*NrPixels) + y;
	PositionTrackers[current_label] += 1;
	int direction;
	for (direction=0;direction<8;++direction){
		DepthFirstSearch(x + dx[direction], y + dy[direction], current_label, NrPixels, BoolImage, ConnectedComponents,Positions,PositionTrackers);
		
	}
}

static inline int FindConnectedComponents(int **BoolImage, int NrPixels, int **ConnectedComponents, int **Positions, int *PositionTrackers){
	int i,j;
	for (i=0;i<NrPixels;i++){
		for (j=0;j<NrPixels;j++){
			ConnectedComponents[i][j] = 0;
		}
	}
	int component = 0;
	for (i=0;i<NrPixels;++i) {
		for (j=0;j<NrPixels;++j) {
			if ((ConnectedComponents[i][j]==0) && (BoolImage[i][j] == 1)){
				DepthFirstSearch(i,j,++component,NrPixels,BoolImage,ConnectedComponents,Positions,PositionTrackers);
			}
		}
	}
	return component;
}

static inline unsigned FindRegionalMaxima(double *z,int **PixelPositions,
		int NrPixelsThisRegion,int **MaximaPositions,double *MaximaValues,
		int *IsSaturated, double IntSat)
{
	unsigned nPeaks = 0;
	int i,j,k,l;
	double zThis, zMatch;
	int xThis, yThis;
	int xNext, yNext;
	int isRegionalMax = 1;
	for (i=0;i<NrPixelsThisRegion;i++){
		isRegionalMax = 1;
		zThis = z[i];
		if (zThis > IntSat) {
			*IsSaturated = 1;
		} else {
			*IsSaturated = 0;
		}
		xThis = PixelPositions[i][0];
		yThis = PixelPositions[i][1];
		for (j=0;j<8;j++){
			xNext = xThis + dx[j];
			yNext = yThis + dy[j];
			for (k=0;k<NrPixelsThisRegion;k++){
				if (xNext == PixelPositions[k][0] && yNext == PixelPositions[k][1] && z[k] > (zThis)){
					isRegionalMax = 0;
				}
			}
		}
		if (isRegionalMax == 1){
			MaximaPositions[nPeaks][0] = xThis;
			MaximaPositions[nPeaks][1] = yThis;
			MaximaValues[nPeaks] = zThis;
			nPeaks++;
		}
	}
	if (nPeaks==0){
        MaximaPositions[nPeaks][0] = PixelPositions[NrPixelsThisRegion/2][0];	
        MaximaPositions[nPeaks][1] = PixelPositions[NrPixelsThisRegion/2][1];
        MaximaValues[nPeaks] = z[NrPixelsThisRegion/2];
        nPeaks=1;
	}
	return nPeaks;
}

struct func_data{
	int NrPixels;
	double *RsEtasZ;
	double *results;
};

__device__ void YZ4mREta(int NrElements, double *R, double *Eta, double *Y, double *Z){
	int i;
	for (i=0;i<NrElements;i++){
		Y[i] = -R[i]*sin(Eta[i]*deg2rad);
		Z[i] = R[i]*cos(Eta[i]*deg2rad);
	}
}

__device__ double CalcEtaAngle(double y, double z){
	double alph;
	alph = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alph = -alph;
	return alph;
}

__global__ void CalcOnePixel(const double *x, double *REtaZ, int nPeaks, int NrPixels, double *result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= NrPixels) return;
	double L, G;
	result[i]=0;
	int j;
	for (j=0;j<nPeaks;j++){
		L = 1/(((((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/((x[(8*j)+6])*(x[(8*j)+6])))+1)*((((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/((x[(8*j)+8])*(x[(8*j)+8])))+1));
		G = exp(-(0.5*(((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/(x[(8*j)+5]*x[(8*j)+5])))-(0.5*(((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/(x[(8*j)+7]*x[(8*j)+7]))));
		result[i] += x[(8*j)+1]*((x[(8*j)+4]*L) + ((1-x[(8*j)+4])*G));
	}
}

__device__ double problem_function(
	int n,
	double *x,
	void* f_data_trial)
{
	struct func_data *f_data = (struct func_data *) f_data_trial;
	int NrPixels = f_data->NrPixels;
	double *REtaZ;
	REtaZ = &(f_data->RsEtasZ[0]);
	int nPeaks = (n-1)/8;
	double TotalDifferenceIntensity=0, CalcIntensity, L, G, IntPeaks, BG = x[0];
	int NrPixelsThisRegion = NrPixels;
	for (int j=0;j<nPeaks;j++){
		IntPeaks = 0;
		for (int i=0;i<NrPixelsThisRegion;i++){
			L = 1/(((((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/((x[(8*j)+6])*(x[(8*j)+6])))+1)*((((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/((x[(8*j)+8])*(x[(8*j)+8])))+1));
			G = exp(-(0.5*(((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/(x[(8*j)+5]*x[(8*j)+5])))-(0.5*(((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/(x[(8*j)+7]*x[(8*j)+7]))));
			IntPeaks += x[(8*j)+1]*((x[(8*j)+4]*L) + ((1-x[(8*j)+4])*G));
		}
		CalcIntensity = BG + IntPeaks;
		TotalDifferenceIntensity += (CalcIntensity - REtaZ[j*3+2])*(CalcIntensity - REtaZ[j+3+2]);
	}
	
	/*double *result;
	result = &(f_data->results[0]);
	//dim3 block (512);
	//dim3 grid ((NrPixels/block.x)+1);
	CalcOnePixel<<<1,NrPixels>>>(x, REtaZ, nPeaks, NrPixels,result);
	cudaDeviceSynchronize();
	long double TotalDifferenceIntensity = 0, CalcIntensity;
	for (int i=0;i<NrPixels;i++){
		CalcIntensity = result[i] + x[0];
		TotalDifferenceIntensity += (CalcIntensity - REtaZ[i*3+2])*(CalcIntensity - REtaZ[i+3+2]);
	}*/
	return TotalDifferenceIntensity;
}

__global__ void Fit2DPeaks (int *PkPx, double *yzInt, double *MaximaInfo, 
	double *ReturnMatrix, int *PosnPeaks, int *PosnPixels, double *ExtraInfo, 
	double *ThreshInfo, double *xDevice, double *xlDevice, double *xuDevice, double *REtaIntDevice,
	double *resultsmat, double *scratch, double *xStepArr, double *xoutDevice){
	int RegNr = blockIdx.x * blockDim.x + threadIdx.x;
	if (RegNr >= (int)ExtraInfo[0]) return;
	int nPeaks = PkPx[RegNr*2];
	int NrPixelsThisRegion = PkPx[RegNr*2+1];
	double Thresh = ThreshInfo[RegNr];
	double *scratchArr;
	int n = 1 + (8*nPeaks);
	double *yzIntThis, *MaximaInfoThis, *ReturnMatrixThis, *resultsThis;
	resultsThis = resultsmat + PosnPixels[RegNr];
	yzIntThis = yzInt+ PosnPixels[RegNr]*3;
	MaximaInfoThis = MaximaInfo + PosnPeaks[RegNr]*3;
	ReturnMatrixThis = ReturnMatrix + PosnPeaks[RegNr]*9;
	double *x,*xl,*xu, *RetaInt, *REtaZ, *xstep, *xout;
	int Posxlu = PosnPeaks[RegNr] * 8 + RegNr;
	int Posreta = PosnPixels[RegNr]*3;
	scratchArr = scratch + ((Posxlu + RegNr)*(Posxlu+RegNr) + 2*Posxlu);
	x =  &xDevice[Posxlu];
	xout = xoutDevice + Posxlu;
	xl = &xlDevice[Posxlu];
	xu = &xuDevice[Posxlu];
	RetaInt = &REtaIntDevice[Posreta];
	REtaZ = RetaInt;
	xstep = xStepArr + Posxlu;
	x[0] = Thresh/2;
	xl[0] = 0;
	xu[0] = Thresh;
	int i;
	for (i=0;i<NrPixelsThisRegion;i++){
		RetaInt[i*3] = CalcNorm2(yzIntThis[i*3]-ExtraInfo[1],yzIntThis[i*3+1]-ExtraInfo[2]);
		RetaInt[i*3+1] = CalcEtaAngle(yzIntThis[i*3]-ExtraInfo[1],yzIntThis[i*3+1]-ExtraInfo[2]);
		RetaInt[i*3+2] = yzIntThis[i*3+2];
	}
	double Width = sqrt((double)NrPixelsThisRegion/(double)nPeaks);
	for (i=0;i<nPeaks;i++){
		x[(8*i)+1] = MaximaInfoThis[i*3]; // Imax
		x[(8*i)+2] = CalcNorm2(MaximaInfoThis[i*3+1]-ExtraInfo[1],MaximaInfoThis[i*3+2]-ExtraInfo[2]); //Radius
		x[(8*i)+3] = CalcEtaAngle(MaximaInfoThis[i*3+1]-ExtraInfo[1],MaximaInfoThis[i*3+2]-ExtraInfo[2]); // Eta
		x[(8*i)+4] = 0.5; // Mu
		x[(8*i)+5] = Width; //SigmaGR
		x[(8*i)+6] = Width; //SigmaLR
		x[(8*i)+7] = rad2deg*atan(Width/x[(8*i)+2]); //SigmaGEta //0.5;
		x[(8*i)+8] = rad2deg*atan(Width/x[(8*i)+2]); //SigmaLEta //0.5;

		double dEta = rad2deg*atan(1/x[(8*i)+2]);
		xl[(8*i)+1] = MaximaInfoThis[i*3]/2;
		xl[(8*i)+2] = x[(8*i)+2] - 1;
		xl[(8*i)+3] = x[(8*i)+3] - dEta;
		xl[(8*i)+4] = 0;
		xl[(8*i)+5] = 0.01;
		xl[(8*i)+6] = 0.01;
		xl[(8*i)+7] = 0.005;
		xl[(8*i)+8] = 0.005;

		xu[(8*i)+1] = MaximaInfoThis[i*3]*2;
		xu[(8*i)+2] = x[(8*i)+2] + 1;
		xu[(8*i)+3] = x[(8*i)+3] + dEta;
		xu[(8*i)+4] = 1;
		xu[(8*i)+5] = 30;
		xu[(8*i)+6] = 30;
		xu[(8*i)+7] = 2;
		xu[(8*i)+8] = 2;
	}
     for (i=0;i<n;i++){
		 xstep[i] = fabs(xu[i]-xl[i])*0.25;
	 }
	struct func_data f_data;
	f_data.NrPixels = NrPixelsThisRegion;
	f_data.RsEtasZ = RetaInt;
	f_data.results = resultsThis;
	struct func_data *f_datat;
	f_datat = &f_data;
	void *trp = (struct func_data *)  f_datat;
	double minf;
    double reqmin = 1e-8;
    int konvge = 10;
    int kcount = MAX_N_EVALS;
    int icount, numres, ifault;
	double IntPeaks, L, G, BGToAdd;
	nelmin(problem_function, n, x, xout, xl, xu, scratchArr, &minf, reqmin, xstep, konvge, kcount, &icount, &numres, &ifault, trp);
	if (ifault !=0) {
	    //printf("%d %d %d %d %d %d\n",RegNr,icount,numres,ifault,nPeaks,NrPixelsThisRegion);
        for (int j=0;j<nPeaks;j++){
            ReturnMatrixThis[j*9+8] = 1;
        }
        return;
	}
	x = xout;
	for (int j=0;j<nPeaks;j++){
		ReturnMatrixThis[j*9] = 0;
		for (i=0;i<NrPixelsThisRegion;i++){
			L = 1/(((((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/((x[(8*j)+6])*(x[(8*j)+6])))+1)*((((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/((x[(8*j)+8])*(x[(8*j)+8])))+1));
			G = exp(-(0.5*(((REtaZ[i*3]-x[(8*j)+2])*(REtaZ[i*3]-x[(8*j)+2]))/(x[(8*j)+5]*x[(8*j)+5])))-(0.5*(((REtaZ[i*3+1]-x[(8*j)+3])*(REtaZ[i*3+1]-x[(8*j)+3]))/(x[(8*j)+7]*x[(8*j)+7]))));
			IntPeaks = x[(8*j)+1]*((x[(8*j)+4]*L) + ((1-x[(8*j)+4])*G));
			BGToAdd = x[0];
			ReturnMatrixThis[j*9] += (BGToAdd + IntPeaks);
		}
		ReturnMatrixThis[j*9+1] = -x[(8*j)+2]*sin(x[(8*j)+3]*deg2rad);
		ReturnMatrixThis[j*9+2] =  x[(8*j)+2]*cos(x[(8*j)+3]*deg2rad);
		ReturnMatrixThis[j*9+3] =  x[8*j+1];
		ReturnMatrixThis[j*9+4] =  x[8*j+2];
		ReturnMatrixThis[j*9+5] =  x[8*j+3];
		ReturnMatrixThis[j*9+6] =  (x[8*j+5]+x[8*j+6])/2;
		ReturnMatrixThis[j*9+7] =  (x[8*j+7]+x[8*j+8])/2;
		ReturnMatrixThis[j*9+8] =  0;
	}
}

void CallFit2DPeaks(int *nPeaksNrPixels, double *yzInt, double *MaximaInfo, 
double *ReturnMatrix, int TotNrRegions, double *YZCen, double *ThreshInfo, 
int *PosMaximaInfoReturnMatrix, int *PosyzInt, int totalPixels, 
int totalPeaks, int blocksize)
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	int *PkPxDevice,*PosMaxInfoRetMatDevice,*PosyzIntDevice;
	cudaMalloc((int **) &PkPxDevice, TotNrRegions*2*sizeof(int));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "PkPxDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(PkPxDevice,nPeaksNrPixels,TotNrRegions*2*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc((int **) &PosMaxInfoRetMatDevice, TotNrRegions*sizeof(int));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "PosMaxInfoRetMatDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(PosMaxInfoRetMatDevice,PosMaximaInfoReturnMatrix,TotNrRegions*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc((int **) &PosyzIntDevice, TotNrRegions*sizeof(int));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "PosyzIntDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(PosyzIntDevice,PosyzInt,TotNrRegions*sizeof(int),cudaMemcpyHostToDevice);
	double ExtraInfo[3] = {(double)TotNrRegions,YZCen[0],YZCen[1]};
	double *yzIntDevice, *MaximaInfoDevice, *ReturnMatrixDevice, *ExtraInfoDevice,
		*ThreshInfoDevice, *xDevice, *xlDevice, *xuDevice, *REtaIntDevice, *resultsmat,
		*scratch, *xStepArr, *xoutDevice;
	cudaMalloc((double **)&yzIntDevice, totalPixels*3*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "yzIntDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(yzIntDevice,yzInt,totalPixels*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&MaximaInfoDevice, totalPeaks*3*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "MaximaInfoDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(MaximaInfoDevice,MaximaInfo,totalPeaks*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&ReturnMatrixDevice, totalPeaks*9*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "ReturnMatrixDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&ThreshInfoDevice, TotNrRegions*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "ThreshInfoDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(ThreshInfoDevice,ThreshInfo,TotNrRegions*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&ExtraInfoDevice, 3*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "ExtraInfoDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMemcpy(ExtraInfoDevice,ExtraInfo,3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMalloc((double **)&xDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "xDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&xlDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "xlDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&xuDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "xuDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&xStepArr,(totalPeaks*8+TotNrRegions)*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "xStepArr Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&xoutDevice,(totalPeaks*8+TotNrRegions)*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "xoutDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	int totN = totalPeaks*8 + TotNrRegions;
	int scratchSpace = (totN+TotNrRegions)*(totN+TotNrRegions) + 2*totN;
	cudaMalloc((double **)&scratch,scratchSpace*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "scratch Free = %zu MB, Total = %zu MB, Scratch size: %zuMB\n", freeMem/(1024*1024), totalMem/(1024*1024),scratchSpace*sizeof(double)/(1024*1024));
	cudaMalloc((double **)&REtaIntDevice,(totalPixels+100)*3*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "REtaIntDevice Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	cudaMalloc((double **)&resultsmat,totalPixels*sizeof(double));
	CHECK(cudaPeekAtLastError());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "resultsmat Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
	int dim = TotNrRegions;
	dim3 block (blocksize);
	dim3 grid ((dim/block.x)+1);
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "block size : %d, grid size: %d Free = %zu MB, Total = %zu MB\n", block.x, grid.x, freeMem/(1024*1024), totalMem/(1024*1024));
    fflush(stdout);
	Fit2DPeaks<<<grid,block>>>(PkPxDevice,yzIntDevice, MaximaInfoDevice, 
		ReturnMatrixDevice, PosMaxInfoRetMatDevice, PosyzIntDevice, 
		ExtraInfoDevice, ThreshInfoDevice, xDevice, xlDevice, xuDevice, 
		REtaIntDevice, resultsmat,scratch, xStepArr, xoutDevice);
	CHECK(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());
	cudaMemcpy(ReturnMatrix,ReturnMatrixDevice,totalPeaks*9*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(PkPxDevice);
	cudaFree(PosMaxInfoRetMatDevice);
	cudaFree(PosyzIntDevice);
	cudaFree(yzIntDevice);
	cudaFree(MaximaInfoDevice);
	cudaFree(ReturnMatrixDevice);
	cudaFree(ExtraInfoDevice);
	cudaFree(ThreshInfoDevice);
	cudaFree(xDevice);
	cudaFree(xlDevice);
	cudaFree(xuDevice);
	cudaFree(xStepArr);
	cudaFree(xoutDevice);
	cudaFree(scratch);
	cudaFree(REtaIntDevice);
	cudaFree(resultsmat);
	CHECK(cudaDeviceSynchronize());
    cudaMemGetInfo(&freeMem, &totalMem);
    fprintf(stderr, "Free = %zu MB, Total = %zu MB\n", freeMem/(1024*1024), totalMem/(1024*1024));
}

double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int main(int argc, char *argv[]){ // Arguments: parameter file name
	if (argc != 2){
		printf("Not enough arguments, exiting. Use as:\n\t\t%s Parameters.txt\n",argv[0]);
		return 1;
	}
	//Read params file
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char line[MAX_LINE_LENGTH];
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Parameter file: %s could not be read. Exiting\n",argv[1]);
		return 1;
	}
	char *str;
	double tstart = cpuSecond();
	int cmpres, StartFileNr, NrFilesPerSweep, NumDarkBegin=0, NumDarkEnd=0,
		ColBeamCurrent, NrOfRings=0, RingNumbers[MAX_N_RINGS], TransOpt[10], 
		NrTransOpt=0, DoFullImage=0, Padding, NrPixels, LayerNr, FrameNumberToDo=-1;
	double OmegaOffset = 0, bc=0, RingSizeThreshold[MAX_N_RINGS][4], px, 
		Width, IntSat, Ycen, Zcen;
	char dummy[MAX_LINE_LENGTH], ParFilePath[MAX_LINE_LENGTH], 
		FileStem[MAX_LINE_LENGTH], RawFolder[MAX_LINE_LENGTH], 
		OutputFolder[MAX_LINE_LENGTH], darkcurrentfilename[MAX_LINE_LENGTH], 
		floodfilename[MAX_LINE_LENGTH], Ext[MAX_LINE_LENGTH];
	while (fgets(line, MAX_LINE_LENGTH, fileParam) != NULL) {
		str = "ParFilePath ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, ParFilePath);
			continue;
		}
		str = "RingThresh ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d %lf", dummy, &RingNumbers[NrOfRings], 
				&RingSizeThreshold[NrOfRings][1]);
			NrOfRings++;
			continue;
		}
		str = "FileStem ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, FileStem);
			continue;
		}
		str = "ParFileColBeamCurrent ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &ColBeamCurrent);
			continue;
		}
		str = "StartFileNr ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &StartFileNr);
			continue;
		}
		str = "NrFilesPerSweep ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NrFilesPerSweep);
			continue;
		}
		str = "NumDarkBegin ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkBegin);
			continue;
		}
		str = "NumDarkEnd ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkEnd);
			continue;
		}
		str = "OmegaOffset ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &OmegaOffset);
			continue;
		}
		str = "BeamCurrent ";
		cmpres = strncmp(line,str,strlen(str));
		if (cmpres==0){
			sscanf(line,"%s %lf", dummy, &bc);
			continue;
		}
        str = "Width ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &Width);
            continue;
        }
        str = "px ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &px);
            continue;
        }
        str = "ImTransOpt ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "DoFullImage ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &DoFullImage);
            continue;
        }
        str = "RawFolder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, RawFolder);
            continue;
        }
        str = "Folder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, OutputFolder);
            continue;
        }
        str = "Dark ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, darkcurrentfilename);
            continue;
        }
        str = "Flood ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, floodfilename);
            continue;
        }
        str = "BC ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf %lf", dummy, &Ycen, &Zcen);
            continue;
        }
        str = "UpperBoundThreshold ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &IntSat);
            continue;
        }
        str = "LayerNr ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "NrPixels ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "Padding ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &Padding);
            continue;
        }
        str = "SingleFrameNumber ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &FrameNumberToDo);
            continue;
        }
        str = "Ext ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, Ext);
            continue;
        }
	}
	printf("Read params file.\n");
	fflush(stdout);
	if (DoFullImage == 1 && FrameNumberToDo == -1){
		printf("For processing the full image you need to provide a single"
			" Frame Number using the FrameNumberToDo parameter in the"
			" param file.\n Exiting\n");
		return (1);
	}
	Width = Width/px;
	FILE *ParFile;
	ParFile = fopen(ParFilePath,"r");
	if (ParFile == NULL){
		printf("ParFile could not be read");
		return 1;
	}
	int i, j, k;
	int NrFramesPerFile[NrFilesPerSweep],CurrFileNrOffset;
	for (i=0;i<NrFilesPerSweep;i++){
		NrFramesPerFile[i] = -(NumDarkBegin+NumDarkEnd);
	}
	char *token, *saveptr;
	int OmegaSign=1, goodLine, omegafound;
	double Omegas[NrFilesPerSweep][300],BeamCurrents[NrFilesPerSweep][300],
			maxBC=0;
	char aline[MAX_LINE_LENGTH];
	int nFramesBC=0;
	while (fgets(aline, MAX_LINE_LENGTH, ParFile) != NULL) {
		strncpy(line,aline,strlen(aline));
		goodLine = 0;
		for (str = line; ; str=NULL){
			token = strtok_r(str, " ", &saveptr);
			if (token == NULL) break;
			if (!strncmp(token,FileStem,strlen(FileStem))){
				token = strtok_r(str, " ", &saveptr);
				token = strtok_r(str, " ", &saveptr);
				CurrFileNrOffset = atoi(token)-StartFileNr;
				if (CurrFileNrOffset >=0 && CurrFileNrOffset < NrFilesPerSweep){
					NrFramesPerFile[CurrFileNrOffset]++;
					goodLine = 1;
				}
			}
		}
		if (NrFramesPerFile[CurrFileNrOffset] < -NumDarkBegin + 1) continue;
		if (goodLine){
			strncpy(line,aline,strlen(aline));
			omegafound = 0;
			for (i=1, str = line; ; i++, str = NULL){
				token = strtok_r(str, " ", &saveptr);
				if (token == NULL) break;
				if (!strncmp(token,"ramsrot",strlen("ramsrot"))){
					omegafound = 1;
					OmegaSign = 1;
				} else if (!strncmp(token,"aero",strlen("aero"))){
					omegafound = 1;
					OmegaSign = -1;
				} else if (!strncmp(token,"preci",strlen("preci"))){
					omegafound = 1;
					OmegaSign = 1;
				}
				if (omegafound){
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					i+=3;
					Omegas[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] 
								= atof(token) * OmegaSign + OmegaOffset;
					omegafound = 0;
				}
				if (i == ColBeamCurrent){
					BeamCurrents[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] = atof(token);
					maxBC = (maxBC > atof(token)) ? maxBC : atof(token);
					nFramesBC++;
				}
			}
		}
	}
	int TotalNrFrames = 0;
	for (i=0;i<NrFilesPerSweep;i++){
		TotalNrFrames += NrFramesPerFile[i];
	}
	bc = (bc > maxBC) ? bc : maxBC;
	// Read hkls.csv
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(line,1000,hklf);
	int Rnr;
	double RRd;
	while (fgets(line,1000,hklf)!=NULL){
		sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", dummy, dummy, 
			dummy, dummy, &Rnr, dummy, dummy, dummy, dummy ,dummy, &RRd);
		for (i=0;i<NrOfRings;i++){
			if (Rnr == RingNumbers[i]){
				RingSizeThreshold[i][0] = RRd/px;
				RingSizeThreshold[i][2] = RRd/px - Width;
				RingSizeThreshold[i][3] = RRd/px + Width;
			}
		}
	}
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 3){
			printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
			return 1;
		}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
        else printf("Transpose.\n");
    }
    int *GoodCoords, *RingInfoImage, TotalGoodPixels=0, ythis, zthis;
    double Rmin, Rmax, Rt;
    GoodCoords = (int*) malloc(NrPixels*NrPixels*sizeof(*GoodCoords));
    RingInfoImage = (int*) malloc(NrPixels*NrPixels*sizeof(*RingInfoImage));
    for (i=0;i<NrPixels*NrPixels;i++){
		GoodCoords[i] = 0;
	}
	for (i=1;i<NrPixels;i++){
		for (j=1;j<NrPixels;j++){
			Rt = sqrt((i-Ycen)*(i-Ycen)+(j-Zcen)*(j-Zcen));
			for (k=0;k<NrOfRings;k++){
				Rmin = RingSizeThreshold[k][2];
				Rmax = RingSizeThreshold[k][3];
				if (Rt > Rmin && Rt < Rmax){
					GoodCoords[((i-1)*NrPixels)+(j-1)] = 1;
					RingInfoImage[((i-1)*NrPixels)+(j-1)] = RingNumbers[k];
					TotalGoodPixels++;
				}
			}
		}
	}
	if (DoFullImage == 1){
		TotalNrFrames = 1;
		for (i=0;i<NrPixels*NrPixels;i++) {
			GoodCoords[i] = 1;
		}
		TotalGoodPixels = NrPixels*NrPixels;
	}
	double *dark,*flood, *darkTemp, *darkTemp2;
	dark = (double *) malloc(NrPixels*NrPixels*NrFilesPerSweep*sizeof(*dark));
	darkTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*darkTemp));
	darkTemp2 = (double *) malloc(NrPixels*NrPixels*sizeof(*darkTemp2));
	flood = (double *) malloc(NrPixels*NrPixels*sizeof(*flood));
	
	// If a darkfile is specified.
	FILE *darkfile=fopen(darkcurrentfilename,"rb");
	int sz, nFrames;
	int SizeFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	long int Skip;
	for (i=0;i<(NrPixels*NrPixels);i++){
		dark[i]=0;
		darkTemp[i]=0;
	}
	pixelvalue *darkcontents;
	darkcontents = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*darkcontents));
	if (darkfile==NULL){
		printf("No dark file was specified, will use %d frames at the beginning of each file for dark calculation.\n", NumDarkBegin);
		for (i=0;i<NrPixels*NrPixels;i++){
			darkTemp[i] = 0;
		}
	}else{
		fseek(darkfile,0L,SEEK_END);
		sz = ftell(darkfile);
		rewind(darkfile);
		nFrames = sz/(8*1024*1024);
		Skip = sz - (nFrames*8*1024*1024);
		fseek(darkfile,Skip,SEEK_SET);
		printf("Reading dark file: %s, nFrames: %d, skipping first %ld bytes.\n",darkcurrentfilename,nFrames,Skip);
		for (i=0;i<nFrames;i++){
			fread(darkcontents,SizeFile,1,darkfile);
			DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
			for (j=0;j<(NrPixels*NrPixels);j++){
				darkTemp[j] += (double) darkcontents[j];
			}
		}
		fclose(darkfile);
		for (i=0;i<(NrPixels*NrPixels);i++){
			darkTemp[i] /= (double) nFrames;
		}
	}
	Transposer(darkTemp,NrPixels,darkTemp2);
	for (i=0;i<NrFilesPerSweep;i++){
		for (j=0;j<NrPixels*NrPixels;j++){
			dark[i*NrPixels*NrPixels + j] = darkTemp2[j];
		}
	}
	char FN[MAX_LINE_LENGTH];
	if (NumDarkBegin != 0){
		for (i=0;i<NrFilesPerSweep;i++){
			for (j=0;j<NrPixels*NrPixels;j++){
				darkTemp[j] = 0;
			}
			if (Padding == 2){sprintf(FN,"%s/%s_%02d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 3){sprintf(FN,"%s/%s_%03d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 4){sprintf(FN,"%s/%s_%04d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 5){sprintf(FN,"%s/%s_%05d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 6){sprintf(FN,"%s/%s_%06d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 7){sprintf(FN,"%s/%s_%07d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 8){sprintf(FN,"%s/%s_%08d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			else if (Padding == 9){sprintf(FN,"%s/%s_%09d%s",RawFolder,FileStem,StartFileNr+i,Ext);}
			FILE *FileTempDark = fopen(FN,"rb");
			fseek(FileTempDark, 0L, SEEK_END);
			sz = ftell(FileTempDark);
			rewind(FileTempDark);
			nFrames = sz/(8*1024*1024);
			Skip = sz - (nFrames*8*1024*1024);
			fseek(FileTempDark,Skip, SEEK_SET);
			for (j=0;j<NumDarkBegin;j++){
				fread(darkcontents,SizeFile,1,FileTempDark);
				DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
				for (k=0;k<NrPixels*NrPixels;k++){
					darkTemp[k] += (double) darkcontents[k];
				}
			}
			fclose(FileTempDark);
			for (j=0;j<NrPixels*NrPixels;j++){
				darkTemp[k] /= NumDarkBegin;
			}
			Transposer(darkTemp,NrPixels,darkTemp2);
			for (j=0;j<NrPixels*NrPixels;j++){
				dark[i*NrPixels*NrPixels + j] = darkTemp2[j];
			}
		}
	}
	free(darkcontents);
	FILE *floodfile=fopen(floodfilename,"rb");
	if (floodfile==NULL){
		printf("Could not read the flood file. Using no flood correction.\n");
		for(i=0;i<(NrPixels*NrPixels);i++){
			flood[i]=1;
		}
	}
	else{
		fread(flood,sizeof(double)*NrPixels*NrPixels, 1, floodfile);
		fclose(floodfile);
	}
	int FrameNr = 0, FramesToSkip, CurrentFileNr, CurrentRingNr;
	double beamcurr, Thresh;
	pixelvalue *Image;
	Image = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*Image));
	double *ImgCorrBCTemp, *ImgCorrBC;
	ImgCorrBC = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBC));
	ImgCorrBCTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBCTemp));
	char outfoldername[MAX_LINE_LENGTH];
	sprintf(outfoldername,"%s/Temp",OutputFolder);
	char extcmd[MAX_LINE_LENGTH];
	sprintf(extcmd,"mkdir -p %s",outfoldername);
	system(extcmd);
	int **BoolImage, **ConnectedComponents, **Positions, *PositionTrackers, NrOfReg;
	BoolImage = allocMatrixInt(NrPixels,NrPixels);
	ConnectedComponents = allocMatrixInt(NrPixels,NrPixels);
	Positions = allocMatrixInt(nOverlapsMaxPerImage,NrPixels*4);
	PositionTrackers = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PositionTrackers));
	int RegNr, IsSaturated;
	char OutFile[MAX_LINE_LENGTH];
	int TotNrRegions=0, NrPixelsThisRegion;
	int *nPeaksNrPixels,*PosyzInt,*PosMaximaInfoReturnMatrix, *RingNumberMatrix;
	nPeaksNrPixels = (int *) malloc(nOverlapsMaxPerImage*2*sizeof(*nPeaksNrPixels));
	RingNumberMatrix = (int *) malloc(nOverlapsMaxPerImage * 200 * sizeof(*RingNumberMatrix));
	double *yzInt, *MaximaInfo, *ReturnMatrix, *ThreshInfo, *YZCen;
	yzInt = (double *) malloc(nOverlapsMaxPerImage*3*NrPixels*sizeof(*yzInt));
	MaximaInfo = (double *) malloc(nOverlapsMaxPerImage*3*100*sizeof(*MaximaInfo));
	ReturnMatrix = (double *) malloc(nOverlapsMaxPerImage*9*100*sizeof(*ReturnMatrix));
	ThreshInfo = (double *) malloc(nOverlapsMaxPerImage*sizeof(*ThreshInfo));
	PosyzInt = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PosyzInt));
	PosMaximaInfoReturnMatrix = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PosMaximaInfoReturnMatrix));
	YZCen = (double *) malloc(2*sizeof(*YZCen));
	YZCen[0] = Ycen;
	YZCen[1] = Zcen;
	int **MaximaPositions, **UsefulPixels, Pos;
	double *MaximaValues, *z, Omega;
	MaximaPositions = allocMatrixInt(NrPixels*10,2);
	MaximaValues = (double*) malloc(NrPixels*10*sizeof(*MaximaValues));
	UsefulPixels = allocMatrixInt(NrPixels*10,2);
	z = (double *) malloc(NrPixels*10*sizeof(*z));
	int counter, counteryzInt, counterMaximaInfoReturnMatrix;
	sprintf(OutFile,"%s/%s_%d_PS.csv",outfoldername,FileStem,LayerNr);
	FILE *outfilewrite;
	outfilewrite = fopen(OutFile,"w");
	fprintf(outfilewrite,"SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta RingNr FrameNr\n");
	int OldCurrentFileNr=StartFileNr-1;
	FILE *ImageFile;
	counter = 0;
	counteryzInt = 0;
	counterMaximaInfoReturnMatrix = 0;
	printf("Starting peaksearch now.\n");
	fflush(stdout);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int nCores = getSPcores(deviceProp);
    printf("Cuda Cores: %d\n",nCores);
    int nJobsLast, nJobsNow=0, resetArrays=1, blocksize = 256, nBad=0;
	while (FrameNr < TotalNrFrames){
		if (TotalNrFrames == 1){ // Look at the next part
			/*FrameNr = FrameNumberToDo;
			for (i=0;i<NrFilesPerSweep;i++){
				if (NrFramesPerFile[i]/FrameNumberToDo > 0){
					FrameNumberToDo -= NrFramesPerFile[i];
				}else{
					CurrentFileNr = StartFileNr + i;
					FramesToSkip = FrameNumberToDo;
					break;
				}
			}*/
		}else{
			CurrentFileNr = StartFileNr;
			FramesToSkip = FrameNr;
			if (FramesToSkip >= (NrFramesPerFile[0])){
				for (i=0;i<NrFilesPerSweep;i++){
					if (FramesToSkip / (NrFramesPerFile[i]) >= 1){
						FramesToSkip -= NrFramesPerFile[i];
						CurrentFileNr++;
					}
				}
			}
		}
		if (OldCurrentFileNr !=CurrentFileNr){
		    if (FrameNr > 0) fclose(ImageFile);
			if (Padding == 2){sprintf(FN,"%s/%s_%02d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 3){sprintf(FN,"%s/%s_%03d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 4){sprintf(FN,"%s/%s_%04d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 5){sprintf(FN,"%s/%s_%05d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 6){sprintf(FN,"%s/%s_%06d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 7){sprintf(FN,"%s/%s_%07d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 8){sprintf(FN,"%s/%s_%08d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			else if (Padding == 9){sprintf(FN,"%s/%s_%09d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
			ImageFile = fopen(FN,"rb");
			if (ImageFile == NULL){
				printf("Could not read the input file. Exiting.\n");
				return 1;
			}
			printf("Read %s file.\n",FN);
			fflush(stdout);
			fseek(ImageFile,0L,SEEK_END);
			sz = ftell(ImageFile);
			rewind(ImageFile);
			Skip = sz - ((NrFramesPerFile[StartFileNr-CurrentFileNr] - NumDarkEnd - FramesToSkip) * 8*1024*1024);
			fseek(ImageFile,Skip,SEEK_SET);
		}
		printf("Now processing file: %s, Frame: %d\n",FN, FramesToSkip);
		fflush(stdout);
		fread(Image,SizeFile,1,ImageFile);
		DoImageTransformations(NrTransOpt,TransOpt,Image,NrPixels);
		beamcurr = BeamCurrents[CurrentFileNr - StartFileNr][FramesToSkip];
		Omega = Omegas[CurrentFileNr - StartFileNr][FramesToSkip];
		printf("Beam current this file: %f, Beam current scaling value: %f\n",beamcurr,bc);
		for (i=0;i<NrPixels*NrPixels;i++)
			ImgCorrBCTemp[i]=(double) Image[i];
		Transposer(ImgCorrBCTemp,NrPixels,ImgCorrBC);
		for (i=0;i<NrPixels*NrPixels;i++){
			ImgCorrBC[i] = (ImgCorrBC[i] - dark[NrPixels*NrPixels*(CurrentFileNr-StartFileNr) + i])/flood[i];
			ImgCorrBC[i] = ImgCorrBC[i]*bc/beamcurr;
			if (GoodCoords[i] == 0){
				ImgCorrBC[i] = 0;
				continue;
			}
			CurrentRingNr = RingInfoImage[i];
			for (j=0;j<NrOfRings;j++){
				if (RingNumbers[j] == CurrentRingNr){
					Pos = j;
				}
			}
			Thresh = RingSizeThreshold[Pos][1];
			if (ImgCorrBC[i] < Thresh){
				ImgCorrBC[i] = 0;
			}
		}
		for (i=0;i<nOverlapsMaxPerImage;i++)
			PositionTrackers[i] = 0;
		for (i=0;i<NrPixels;i++){
			for (j=0;j<NrPixels;j++){
				if (ImgCorrBC[(i*NrPixels)+j] != 0){
					BoolImage[i][j] = 1;
				}else{
					BoolImage[i][j] = 0;
				}
			}
		}
		NrOfReg = FindConnectedComponents(BoolImage,NrPixels,ConnectedComponents,Positions,PositionTrackers);
		if (resetArrays == 1){
			counter = 0;
			counteryzInt = 0;
			counterMaximaInfoReturnMatrix = 0;
			TotNrRegions = 0;
		}
		TotNrRegions += NrOfReg;
		nJobsLast = nJobsNow;
		for (RegNr=1;RegNr<=NrOfReg;RegNr++){
			NrPixelsThisRegion = PositionTrackers[RegNr];
			if (NrPixelsThisRegion == 1){
				TotNrRegions--;
				continue;
			}
			for (i=0;i<NrPixelsThisRegion;i++){
				UsefulPixels[i][0] = (int)(Positions[RegNr][i]/NrPixels);
				UsefulPixels[i][1] = (int)(Positions[RegNr][i]%NrPixels);
				z[i] = ImgCorrBC[((UsefulPixels[i][0])*NrPixels) + (UsefulPixels[i][1])];
			}
			unsigned nPeaks;
			nPeaks = FindRegionalMaxima(z,UsefulPixels,NrPixelsThisRegion,
				MaximaPositions,MaximaValues,&IsSaturated,IntSat);
			if (IsSaturated == 1){
				TotNrRegions--;
				continue;
			}
			nPeaksNrPixels[counter*2] = nPeaks;
			nPeaksNrPixels[counter*2+1] = NrPixelsThisRegion;
			PosMaximaInfoReturnMatrix[counter] = counterMaximaInfoReturnMatrix;
			PosyzInt[counter] = counteryzInt;
			for (i=0;i<NrPixelsThisRegion;i++){
				yzInt[(counteryzInt+i)*3 + 0] = (double)UsefulPixels[i][0];
				yzInt[(counteryzInt+i)*3 + 1] = (double)UsefulPixels[i][1];
				yzInt[(counteryzInt+i)*3 + 2] = z[i];
			}
			for (i=0;i<nPeaks;i++){
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 0] = MaximaValues[i];
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 1] = (double)MaximaPositions[i][0];
				MaximaInfo[(counterMaximaInfoReturnMatrix+i)*3 + 2] = (double)MaximaPositions[i][1];
				RingNumberMatrix[(counterMaximaInfoReturnMatrix+i)*2+0] = RingInfoImage[MaximaPositions[0][0]*NrPixels+MaximaPositions[0][1]];
				RingNumberMatrix[(counterMaximaInfoReturnMatrix+i)*2+1] = FrameNr;
			}
			for (i=0;i<NrOfRings;i++){
				if (RingNumbers[i] == RingNumberMatrix[(counterMaximaInfoReturnMatrix+i)*2+0]){
					Pos = i;
				}
			}
			ThreshInfo[counter] = RingSizeThreshold[Pos][1];
			counteryzInt+= NrPixelsThisRegion;
			counterMaximaInfoReturnMatrix += nPeaks;
			counter++;
		}
		nJobsNow = counterMaximaInfoReturnMatrix;
		nJobsLast = nJobsNow - nJobsLast;
		printf("Time taken till %d frame: %lf seconds.\n",FrameNr, cpuSecond()-tstart);
		printf("Total number of peaks in the images since CUDA run: %d\n",counterMaximaInfoReturnMatrix);
		printf("Total number of useful pixels in the images since CUDA run: %d\n",counteryzInt);
		fflush(stdout);
		resetArrays = 0;
		if (nJobsNow + 2*nJobsLast + blocksize >= nCores || FrameNr == TotalNrFrames-1){
		    printf("Starting CUDA job with %d jobs at %d frameNr.\n",nJobsNow, FrameNr);
			printf("Total number of peaks for CUDA run: %d\n",counterMaximaInfoReturnMatrix);
			printf("Total number of useful pixels for CUDA run: %d\n",counteryzInt);
			// Now send all info to the GPU calling code
			CallFit2DPeaks(nPeaksNrPixels, yzInt, MaximaInfo, ReturnMatrix, 
				TotNrRegions, YZCen, ThreshInfo, PosMaximaInfoReturnMatrix, 
				PosyzInt, counteryzInt, counterMaximaInfoReturnMatrix, blocksize);
			for (i=0;i<counterMaximaInfoReturnMatrix;i++){
			    if (ReturnMatrix[i*9+8] == 0){
				    fprintf(outfilewrite,"%d %f %f %f %f %f %f %f %f %f %d %d\n",i+1,
				    	ReturnMatrix[i*9+0],Omega,ReturnMatrix[i*9+1]+Ycen,
					    ReturnMatrix[i*9+2]+Zcen,ReturnMatrix[i*9+3],
					    ReturnMatrix[i*9+4], ReturnMatrix[i*9+5],
					    ReturnMatrix[i*9+6],ReturnMatrix[i*9+7], RingNumberMatrix[i*2],RingNumberMatrix[i*2+1]);
				}else{
				    nBad++;
                }
			}
			printf("Time taken till %d frame: %lf seconds, bad peaks= %d out of %d peaks.\n",FrameNr, cpuSecond()-tstart, nBad, counterMaximaInfoReturnMatrix);
			resetArrays = 1;
		}
		FrameNr++;
		OldCurrentFileNr = CurrentFileNr;
	}
	fclose(outfilewrite);
	printf("Total time taken: %lf seconds.\n",cpuSecond()-tstart);
}
