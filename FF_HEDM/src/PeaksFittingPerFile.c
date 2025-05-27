//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  PeaksFittingPerFile.c
//
//
//  Created by Hemant Sharma on 2014/07/04.
//
//
// TODO: Rectangular detector, read edf, omp.

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
#include <fcntl.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAXNHKLS 5000
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
typedef uint16_t pixelvalue;

long double diff(struct timespec start, struct timespec end)
{
	long double diff_sec = end.tv_sec - start.tv_sec;
	long double diff_nsec = end.tv_nsec - start.tv_nsec;
	return (diff_sec * 1e6) + (diff_nsec / 1000.0);
}

static inline
pixelvalue**
allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
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
FreeMemMatrixPx(pixelvalue **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline
void YZ4mREta(int NrElements, double *R, double *Eta, double *Y, double *Z){
	int i;
	for (i=0;i<NrElements;i++){
		Y[i] = -R[i]*sin(Eta[i]*deg2rad);
		Z[i] = R[i]*cos(Eta[i]*deg2rad);
	}
}

static inline
int**
allocMatrixInt(int nrows, int ncols)
{
    int** arr;
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
FreeMemMatrixInt(int **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
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

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

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
	double *z;
	double *Rs;
	double *Etas;
};

static
double problem_function(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct func_data *f_data = (struct func_data *) f_data_trial;
	int NrPixels = f_data->NrPixels;
	double *z,*Rs,*Etas;
	z = &(f_data->z[0]);
	Rs = &(f_data->Rs[0]);
	Etas = &(f_data->Etas[0]);
	int nPeaks, i,j,k;
	nPeaks = (n-1)/8;
	double BG = x[0];
	double IMAX[nPeaks], R[nPeaks], Eta[nPeaks], Mu[nPeaks], SigmaGR[nPeaks], SigmaLR[nPeaks], SigmaGEta[nPeaks],SigmaLEta[nPeaks];
	for (i=0;i<nPeaks;i++){
		IMAX[i] = x[(8*i)+1];
		R[i] = x[(8*i)+2];
		Eta[i] = x[(8*i)+3];
		Mu[i] = x[(8*i)+4];
		SigmaGR[i] = x[(8*i)+5];
		SigmaLR[i] = x[(8*i)+6];
		SigmaGEta[i] = x[(8*i)+7];
		SigmaLEta[i] = x[(8*i)+8];
	}
	double TotalDifferenceIntensity = 0, CalcIntensity, IntPeaks;
	double L, G,DR,DE,R2,E2;
	for (i=0;i<NrPixels;i++){
		IntPeaks = 0;
		for (j=0;j<nPeaks;j++){
			DR = Rs[i]-R[j];
			R2 = DR*DR;
			DE = Etas[i]-Eta[j];
			E2 = DE*DE;
			L = 1/(((R2/((SigmaLR[j])*(SigmaLR[j])))+1)*((E2/((SigmaLEta[j])*(SigmaLEta[j])))+1));
			//~ L = 1/(((R2/((SigmaLR[j])*(SigmaLR[j]))))+((E2/((SigmaLEta[j])*(SigmaLEta[j]))))+1);
			G = exp(-(0.5*(R2/(SigmaGR[j]*SigmaGR[j])))-(0.5*(E2/(SigmaGEta[j]*SigmaGEta[j]))));
			IntPeaks += IMAX[j]*((Mu[j]*L) + ((1-Mu[j])*G));
		}
		CalcIntensity = BG + IntPeaks;
		TotalDifferenceIntensity += (CalcIntensity - z[i])*(CalcIntensity - z[i]);
	}
	return TotalDifferenceIntensity;
}

static inline void CalcIntegratedIntensity(int nPeaks,double *x,double *Rs,double *Etas,int NrPixelsThisRegion,double *IntegratedIntensity,int *NrOfPixels){
	double BG = x[0];
	int i,j;
	double IMAX[nPeaks], R[nPeaks], Eta[nPeaks], Mu[nPeaks], SigmaGR[nPeaks], SigmaLR[nPeaks], SigmaGEta[nPeaks],SigmaLEta[nPeaks];
	for (i=0;i<nPeaks;i++){
		IMAX[i] = x[(8*i)+1];
		R[i] = x[(8*i)+2];
		Eta[i] = x[(8*i)+3];
		Mu[i] = x[(8*i)+4];
		SigmaGR[i] = x[(8*i)+5];
		SigmaLR[i] = x[(8*i)+6];
		SigmaGEta[i] = x[(8*i)+7];
		SigmaLEta[i] = x[(8*i)+8];
	}
	double IntPeaks, L, G, BGToAdd,DR,DE,R2,E2;
	for (j=0;j<nPeaks;j++){
		NrOfPixels[j] = 0;
		IntegratedIntensity[j] = 0;
		for (i=0;i<NrPixelsThisRegion;i++){
			DR = Rs[i]-R[j];
			R2 = DR*DR;
			DE = Etas[i]-Eta[j];
			E2 = DE*DE;
			L = 1/(((R2/((SigmaLR[j])*(SigmaLR[j])))+1)*((E2/((SigmaLEta[j])*(SigmaLEta[j])))+1));
			//~ L = 1/(((R2/((SigmaLR[j])*(SigmaLR[j]))))+((E2/((SigmaLEta[j])*(SigmaLEta[j]))))+1);
			G = exp(-(0.5*(R2/(SigmaGR[j]*SigmaGR[j])))-(0.5*(E2/(SigmaGEta[j]*SigmaGEta[j]))));
			IntPeaks = IMAX[j]*((Mu[j]*L) + ((1-Mu[j])*G));
			if (IntPeaks > BG){
				NrOfPixels[j] += 1;
				BGToAdd = BG;
			}else{
				BGToAdd = 0;
			}
			IntegratedIntensity[j] += (BGToAdd + IntPeaks);
		}
	}
}

int Fit2DPeaks(unsigned nPeaks, int NrPixelsThisRegion, double *z, int **UsefulPixels, double *MaximaValues,
				int **MaximaPositions, double *IntegratedIntensity, double *IMAX, double *YCEN, double *ZCEN,
				double *RCens, double *EtaCens,double Ycen, double Zcen, double Thresh, int *NrPx,double *OtherInfo)
{
	unsigned n = 1 + (8*nPeaks);
	double x[n],xl[n],xu[n];
	x[0] = Thresh/2;
	xl[0] = 0;
	xu[0] = Thresh;
	int i,j;
	double *Rs, *Etas;
	Rs = malloc(NrPixelsThisRegion*2*sizeof(*Rs));
	Etas = malloc(NrPixelsThisRegion*2*sizeof(*Etas));
	double RMin=1e8, RMax=0, EtaMin=190, EtaMax=-190;
	for (i=0;i<NrPixelsThisRegion;i++){
		Rs[i] = CalcNorm2(UsefulPixels[i][0]-Ycen,UsefulPixels[i][1]-Zcen);
		Etas[i] = CalcEtaAngle(-UsefulPixels[i][0]+Ycen,UsefulPixels[i][1]-Zcen);
		if (Rs[i] > RMax) RMax = Rs[i];
		if (Rs[i] < RMin) RMin = Rs[i];
		if (Etas[i] > EtaMax) EtaMax = Etas[i];
		if (Etas[i] < EtaMin) EtaMin = Etas[i];
	}
	double MaxEtaWidth, MaxRWidth;
	MaxRWidth = (RMax - RMin)/2 + 1;
	MaxEtaWidth = (EtaMax - EtaMin)/2 + atand(2/(RMax+RMin));
	if (EtaMax - EtaMin > 180) MaxEtaWidth -= 180;
	double Width = sqrt(NrPixelsThisRegion/nPeaks);
	if (Width > MaxRWidth) Width = MaxRWidth;
	double initSigmaEta;
	for (i=0;i<nPeaks;i++){
		x[(8*i)+1] = MaximaValues[i]; // Imax
		x[(8*i)+2] = CalcNorm2(MaximaPositions[i][0]-Ycen,MaximaPositions[i][1]-Zcen); //Radius
		x[(8*i)+3] = CalcEtaAngle(-MaximaPositions[i][0]+Ycen,MaximaPositions[i][1]-Zcen); // Eta
		x[(8*i)+4] = 0.5; // Mu
		x[(8*i)+5] = Width; //SigmaGR
		x[(8*i)+6] = Width; //SigmaLR
		initSigmaEta = Width/x[(8*i)+2];
		if (atand(initSigmaEta) > MaxEtaWidth) initSigmaEta = tand(MaxEtaWidth)-0.0001;
		x[(8*i)+7] = atand(initSigmaEta); //SigmaGEta //0.5;
		x[(8*i)+8] = atand(initSigmaEta); //SigmaLEta //0.5;

		double dEta = rad2deg*atan(1/x[(8*i)+2]);
		xl[(8*i)+1] = MaximaValues[i]/2;
		xl[(8*i)+2] = x[(8*i)+2] - 1;
		xl[(8*i)+3] = x[(8*i)+3] - dEta;
		xl[(8*i)+4] = 0;
		xl[(8*i)+5] = 0.01;
		xl[(8*i)+6] = 0.01;
		xl[(8*i)+7] = 0.005;
		xl[(8*i)+8] = 0.005;

		xu[(8*i)+1] = MaximaValues[i]*2;
		xu[(8*i)+2] = x[(8*i)+2] + 1;
		xu[(8*i)+3] = x[(8*i)+3] + dEta;
		xu[(8*i)+4] = 1;
		xu[(8*i)+5] = 2*MaxRWidth;
		xu[(8*i)+6] = 2*MaxRWidth;
		xu[(8*i)+7] = 2*MaxEtaWidth;
		xu[(8*i)+8] = 2*MaxEtaWidth;

		//~ for (j=0;j<9;j++) printf("Args: %lf %lf %lf\n",x[8*i+j],xl[8*i+j],xu[8*i+j]);
	}
	struct func_data f_data;
	f_data.NrPixels = NrPixelsThisRegion;
	f_data.Rs = &Rs[0];
	f_data.Etas = &Etas[0];
	f_data.z = &z[0];
	struct func_data *f_datat;
	f_datat = &f_data;
	void *trp = (struct func_data *)  f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_maxtime(opt, 300);
	nlopt_set_min_objective(opt, problem_function, trp);
	double minf;
	int rc = nlopt_optimize(opt, x, &minf);
	//~ printf("RC: %d\n",rc);
	nlopt_destroy(opt);
	for (i=0;i<nPeaks;i++){
		IMAX[i] = x[(8*i)+1];
		RCens[i] = x[(8*i)+2];
		EtaCens[i] = x[(8*i)+3];
		if (x[(8*i)+5] > x[(8*i)+6]){
			OtherInfo[2*i] = x[(8*i)+5];
		}else{
			OtherInfo[2*i] = x[(8*i)+6];
		}
		if (x[(8*i)+7] > x[(8*i)+8]){
			OtherInfo[2*i+1] = x[(8*i)+7];
		}else{
			OtherInfo[2*i+1] = x[(8*i)+8];
		}
	}
	YZ4mREta(nPeaks,RCens,EtaCens,YCEN,ZCEN);
	CalcIntegratedIntensity(nPeaks,x,Rs,Etas,NrPixelsThisRegion,IntegratedIntensity,NrPx);
	free(Rs);
	free(Etas);
	return rc;
}

static inline int CheckDirectoryCreation(char Folder[1024])
{
	int e;
    struct stat sb;
	char totOutDir[1024];
	sprintf(totOutDir,"%s/",Folder);
    e = stat(totOutDir,&sb);
    if (e!=0 && errno == ENOENT){
		printf("Output directory did not exist, creating %s\n",totOutDir);
		e = mkdir(totOutDir,S_IRWXU);
		if (e !=0) {printf("Could not make the directory. Exiting\n");return 0;}
	}
	return 1;
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *Image, int NrPixels)
{
	int i,j,k,l,m;
    pixelvalue **ImageTemp1, **ImageTemp2;
    ImageTemp1 = allocMatrixPX(NrPixels,NrPixels);
    ImageTemp2 = allocMatrixPX(NrPixels,NrPixels);
	for (k=0;k<NrPixels;k++) for (l=0;l<NrPixels;l++) ImageTemp1[k][l] = Image[(NrPixels*k)+l];
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

static void
check (int test, const char * message, ...)
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

int main(int argc, char *argv[]){
	clock_t start, end;
	if (argc != 4){
		printf("Usage:\n PeaksFittingPerFile params.txt fileNr ringNr\n");
		return 1;
	}
    double diftotal;
    start = clock();
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fflush(stdout);
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Parameter file could not be read. Exiting\n");
		return 1;
	}
    check (fileParam == NULL,"%s file not found: %s", ParamFN, strerror(errno));
    char *str, dummy[1000], Folder[1024], FileStem[1024], *TmpFolder, darkcurrentfilename[1024], floodfilename[1024], Ext[1024],RawFolder[1024];
    TmpFolder = "Temp";
    int LowNr,FileNr,RingNr;
    FileNr = atoi(argv[2]);
    RingNr = atoi(argv[3]);
    double Thresh, bc=1, Ycen, Zcen, IntSat, OmegaStep, OmegaFirstFile, Lsd, px, Width, Wavelength,MaxRingRad;
    int NrPixels,Padding = 6, StartNr;
    char fs[1024];
    int LayerNr;
    int NrTransOpt=0;
    int TransOpt[10];
    int StartFileNr, NrFilesPerSweep;
    int DoFullImage = 0;
    int FrameNrOmeChange = 1, NrDarkFramesDataFile = 0;
    double OmegaMissing = 0, MisDir;
    double FileOmegaOmeStep[360][2];
    int headSize = 8192;
    int fnr = 0;
    double RhoD, tx, ty, tz, p0, p1, p2;
    double OmegaRanges[2000][2];
    int nOmeRanges = 0;
    long long int BadPxIntensity = 0;
    int minNrPx=1, maxNrPx=10000, makeMap = 0, maxNPeaks=400;
    while (fgets(aline,1000,fileParam)!=NULL){
		//printf("%s",aline);
		fflush(stdout);
		str = "MaxNPeaks ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &maxNPeaks);
            continue;
        }
		str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
		str = "MinNrPx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &minNrPx);
            continue;
        }
		str = "MaxNrPx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &maxNrPx);
            continue;
        }
		str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ty);
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tz);
            continue;
        }
        str = "p0 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p0);
            continue;
        }
        str = "p1 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p1);
            continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy,
				&OmegaRanges[nOmeRanges][0], &OmegaRanges[nOmeRanges][1]);
            nOmeRanges++;
            continue;
        }
        str = "p2 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p2);
            continue;
        }
        str = "StartFileNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &StartFileNr);
            continue;
        }
        str = "DoFullImage ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &DoFullImage);
            continue;
        }
        str = "HeadSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &headSize);
            continue;
        }
        str = "NrFilesPerSweep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrFilesPerSweep);
            continue;
        }
        str = "Ext ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Ext);
            continue;
        }
        str = "RawFolder ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, RawFolder);
            continue;
        }
        str = "Folder ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Folder);
            continue;
        }
        str = "FileStem ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fs);
            continue;
        }
        str = "Dark ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, darkcurrentfilename);
            continue;
        }
        str = "Flood ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, floodfilename);
            continue;
        }
        str = "LowerBoundThreshold ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Thresh);
            continue;
        }
        str = "BeamCurrent ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &bc);
            continue;
        }
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &Ycen, &Zcen);
            continue;
        }
        str = "UpperBoundThreshold ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &IntSat);
            continue;
        }
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "OmegaFirstFile ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaFirstFile);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
        str = "Width ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Width);
            continue;
        }
        str = "LayerNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "NrPixels ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "Padding ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &Padding);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd);
            continue;
        }
        str = "StartNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &StartNr);
            continue;
        }
        str = "NrDarkFramesDataFile ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrDarkFramesDataFile);
            continue;
        }
        str = "BadPxIntensity ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lld", dummy, &BadPxIntensity);
            makeMap = 1;
            continue;
        }
        str = "MaxRingRad ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxRingRad);
            continue;
        }
        str = "RhoD ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &RhoD);
            continue;
        }
        str = "ImTransOpt ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "FrameOmeChange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d %lf %lf", dummy, &FrameNrOmeChange, &OmegaMissing, &MisDir);
            continue;
        }
        str = "FileOmegaOmeStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &FileOmegaOmeStep[fnr][0], &FileOmegaOmeStep[fnr][1]);
            fnr++;
            continue;
        }
        //~ str = "IsEDF";
        //~ LowNr = strncmp()
	}
	printf("Threshold: %f\n",Thresh);
	Width = Width/px;
	int i,j,k;
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 3){
			printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
			return 0;
		}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0)
			printf("No change.\n");
        else if (TransOpt[i] == 1)
			printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2)
			printf("Flip Top Bottom.\n");
        else
			printf("Transpose.\n");
    }
	sprintf(FileStem,"%s_%d",fs,LayerNr);
	fclose(fileParam);
	double MaxTtheta = rad2deg*atan(MaxRingRad/Lsd);
	double RingRad;
	char hklfn[2040];
	sprintf(hklfn,"%s/hkls.csv",Folder);
	FILE *hklf = fopen(hklfn,"r");
    if (hklf == NULL){
		printf("HKL file could not be read. Exiting\n");
		return 1;
	}
	fgets(aline,1000,hklf);
	int Rnr;
	double RRd;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %s %s %s %s %s %lf",dummy,dummy,dummy,dummy,&Rnr,dummy,dummy,dummy,dummy,dummy,&RRd);
		if (Rnr == RingNr){
			RingRad = RRd;
			break;
		}
	}
	RingRad = RingRad/px;
	printf("RingNr = %d RingRad = %f\n",RingNr, RingRad);
	double Rmin=RingRad-Width, Rmax=RingRad+Width;
	double Omega;
	int Nadditions;
	if (fnr == 0){
		if (FileNr - StartNr + 1 < FrameNrOmeChange){
			Omega = OmegaFirstFile + ((FileNr-StartNr)*OmegaStep);
		} else {
			Nadditions = (int) ((FileNr - StartNr + 1) / FrameNrOmeChange)  ;
			Omega = OmegaFirstFile + ((FileNr-StartNr)*OmegaStep) + MisDir*OmegaMissing*Nadditions;
		}
    }
    // Dark file reading from here.
	double *dark, *flood, *darkTemp;;
	dark = malloc(NrPixels*NrPixels*sizeof(*dark));
	darkTemp = malloc(NrPixels*NrPixels*sizeof(*darkTemp));
	flood = malloc(NrPixels*NrPixels*sizeof(*flood));
	FILE *darkfile=fopen(darkcurrentfilename,"rb");
	size_t sz;
	int nFrames;
	int SizeFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	long int Skip;
	for (i=0;i<(NrPixels*NrPixels);i++){dark[i]=0;darkTemp[i]=0;}
	pixelvalue *darkcontents;
	darkcontents = malloc(NrPixels*NrPixels*sizeof(*darkcontents));
	if (darkfile==NULL){printf("Could not read the dark file. Using no dark subtraction.\n");}
	else{
		fseek(darkfile,0L,SEEK_END);
		sz = ftell(darkfile);
		rewind(darkfile);
		nFrames = sz/(2*NrPixels*NrPixels);
		Skip = sz - (nFrames*2*NrPixels*NrPixels);
		fseek(darkfile,Skip,SEEK_SET);
		printf("Reading dark file: %s, nFrames: %d, skipping first %ld bytes.\n",darkcurrentfilename,nFrames,Skip);
		for (i=0;i<nFrames;i++){
			fread(darkcontents,SizeFile,1,darkfile);
			DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
			for (j=0;j<(NrPixels*NrPixels);j++){
				darkTemp[j] += darkcontents[j];
			}
		}
		fclose(darkfile);
		for (i=0;i<(NrPixels*NrPixels);i++){
			darkTemp[i] /= nFrames;
		}
	}
	Transposer(darkTemp,NrPixels,dark);
	free(darkcontents);
	//Finished reading dark file.

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
	double Rt;
	int *GoodCoords;
	GoodCoords = malloc(NrPixels*NrPixels*sizeof(*GoodCoords));
	memset(GoodCoords,0,NrPixels*NrPixels*sizeof(*GoodCoords));
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
	double Yc, Zc, n0=2, n1=4, n2=2;
	double ABC[3], ABCPr[3], XYZ[3];
	double Rad, Eta, RNorm, DistortFunc, EtaT;
	int nrCoords = 0;
	for (i=1;i<NrPixels;i++){
		for (j=1;j<NrPixels;j++){
			// Correct for tilts and Distortion here
			Yc = (-i + Ycen)*px;
			Zc =  (j - Zcen)*px;
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
			DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT)))) + (p2*(pow(RNorm,n2))) + 1;
			Rt = Rad * DistortFunc / px;
			if (Rt > Rmin && Rt < Rmax){
				GoodCoords[((i-1)*NrPixels)+(j-1)] = 1;
				nrCoords ++;
			} else {
				GoodCoords[((i-1)*NrPixels)+(j-1)] = 0;
			}
		}
	}
	printf("Number of coordinates: %d\n",nrCoords);
	if (DoFullImage == 1){
		for (i=0;i<NrPixels*NrPixels;i++) GoodCoords[i] = 1;
	}

	// Get nFrames:
	FILE *dummyFile;
	char dummyFN[2048];
	sprintf(dummyFN,"%s/%s_%0*d%s",RawFolder,fs,Padding,StartFileNr,Ext);
	dummyFile = fopen(dummyFN,"rb");
	if (dummyFile == NULL){
		printf("Could not read the input file %s. Exiting.\n",dummyFN);
		return 1;
	}
	fseek(dummyFile,0L,SEEK_END);
	sz = ftell(dummyFile);
	sz = sz - headSize;
	fclose(dummyFile);
	nFrames = sz/(2*NrPixels*NrPixels);

	char FN[2048];
	int ReadFileNr;
	ReadFileNr = StartFileNr + ((FileNr-1) / nFrames);
	int FramesToSkip = ((FileNr-1) % nFrames);

	if (fnr != 0){
		OmegaStep = FileOmegaOmeStep[ReadFileNr-StartFileNr][1];
		Omega = FileOmegaOmeStep[ReadFileNr-StartFileNr][0] + FramesToSkip*OmegaStep;
	}
	char OutFolderName[1024];
	sprintf(OutFolderName,"%s/%s",Folder,TmpFolder);
	int e = CheckDirectoryCreation(OutFolderName);
	if (e == 0){ return 1;}
	char OutFile[1024];
	sprintf(OutFile,"%s/%s_%0*d_%d_PS.csv",OutFolderName,FileStem,Padding,FileNr,RingNr);
	FILE *outfilewrite;
	outfilewrite = fopen(OutFile,"w");
	fprintf(outfilewrite,"SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta NrPixels TotalNrPixelsInPeakRegion nPeaks maxY maxZ diffY diffZ rawIMax returnCode\n");
	int KeepSpots = 0;
	for (i=0;i<nOmeRanges;i++){
		if (Omega >= OmegaRanges[i][0] && Omega <= OmegaRanges[i][1]) KeepSpots = 1;
	}
	if (KeepSpots == 0) return 0;
	sprintf(FN,"%s/%s_%0*d%s",RawFolder,fs,Padding,ReadFileNr,Ext);
	printf("Reading file: %s\n",FN);
	FILE *ImageFile = fopen(FN,"rb");
	if (ImageFile == NULL){
		printf("Could not read the input file. Exiting.\n");
		return 1;
	}
	pixelvalue *Image;
	Image = malloc(NrPixels*NrPixels*sizeof(*Image));
	fseek(ImageFile,0L,SEEK_END);
	sz = ftell(ImageFile);
	rewind(ImageFile);
	size_t temp = (nFrames-FramesToSkip);
	temp *= 2;
	temp *= NrPixels;
	temp *= NrPixels;
	Skip = sz - temp;
	printf("Now processing file: %s\n",FN);
	double beamcurr=1;
	fseek(ImageFile,Skip,SEEK_SET);
	fread(Image,SizeFile,1,ImageFile);
	if (makeMap == 1){
		int badPxCounter = 0;
		for (i=0;i<NrPixels*NrPixels;i++){
			if (Image[i] == (pixelvalue)BadPxIntensity){
				Image[i] = 0;
				badPxCounter++;
			}
		}
		printf("Number of badPixels %d\n",badPxCounter);
	}
	fclose(ImageFile);
	DoImageTransformations(NrTransOpt,TransOpt,Image,NrPixels);
	printf("Beam current this file: %f, Beam current scaling value: %f\n",beamcurr,bc);
	double *ImgCorrBCTemp, *ImgCorrBC;
	ImgCorrBC = malloc(NrPixels*NrPixels*sizeof(*ImgCorrBC));
	ImgCorrBCTemp = malloc(NrPixels*NrPixels*sizeof(*ImgCorrBCTemp));
	for (i=0;i<(NrPixels*NrPixels);i++)ImgCorrBCTemp[i]=Image[i];
	free(Image);
	Transposer(ImgCorrBCTemp,NrPixels,ImgCorrBC);
	for (i=0;i<(NrPixels*NrPixels);i++){
		ImgCorrBC[i] = (ImgCorrBC[i] - dark[i])/flood[i];
		ImgCorrBC[i] = ImgCorrBC[i]*bc/beamcurr;
		if (GoodCoords[i] == 0){
			ImgCorrBC[i] = 0;
		} else {
			if (ImgCorrBC[i] < Thresh){
				ImgCorrBC[i] = 0;
			}
		}
	}
	free(GoodCoords);
	free(ImgCorrBCTemp);
	free(dark);
	free(flood);
	int nOverlapsMaxPerImage = 10000;
	// Do Connected components
	int **BoolImage, **ConnectedComponents;
	BoolImage = allocMatrixInt(NrPixels,NrPixels);
	ConnectedComponents = allocMatrixInt(NrPixels,NrPixels);
	int **Positions;
	Positions = allocMatrixInt(nOverlapsMaxPerImage,NrPixels*4);
	int *PositionTrackers;
	PositionTrackers = malloc(nOverlapsMaxPerImage*sizeof(*PositionTrackers));
	for (i=0;i<nOverlapsMaxPerImage;i++)PositionTrackers[i] = 0;
	int NrOfReg;
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
	FreeMemMatrixInt(BoolImage,NrPixels);
	/*FILE *connectedcomps;
	char conncomp[1024];
	pixelvalue *CCP;
	CCP = malloc(NrPixels*NrPixels*sizeof(*CCP));
	memset(CCP,0,NrPixels*NrPixels*sizeof(*CCP));
	for (i=0;i<NrPixels;i++){
		for (j=0;j<NrPixels;j++){
			if (ConnectedComponents[i][j] != 0){
				CCP[(i*NrPixels)+j] = ConnectedComponents[i][j];
			}
		}
	}
	sprintf(conncomp,"%sccp",FN);
	connectedcomps = fopen(conncomp,"w");
	fwrite(CCP,NrPixels*NrPixels*sizeof(pixelvalue),1,connectedcomps);
	fclose(connectedcomps);
	free(CCP);*/
	FreeMemMatrixInt(ConnectedComponents,NrPixels);
	int RegNr,NrPixelsThisRegion;
	int **MaximaPositions;
	double *MaximaValues;
	int **UsefulPixels;
	double *z;
	MaximaPositions = allocMatrixInt(NrPixels*10,2);
	MaximaValues = malloc(NrPixels*10*sizeof(*MaximaValues));
	UsefulPixels = allocMatrixInt(NrPixels*10,2);
	z = malloc(NrPixels*10*sizeof(*z));
	int IsSaturated;
	int SpotIDStart = 1;
	int TotNrRegions = NrOfReg;
	struct timespec timer1, timer2;
	long double timex=0;
	for (RegNr=1;RegNr<=NrOfReg;RegNr++){
		NrPixelsThisRegion = PositionTrackers[RegNr];
		for (i=0;i<NrPixelsThisRegion;i++){
			UsefulPixels[i][0] = (int)(Positions[RegNr][i]/NrPixels);
			UsefulPixels[i][1] = (int)(Positions[RegNr][i]%NrPixels);
			z[i] = ImgCorrBC[((UsefulPixels[i][0])*NrPixels) + (UsefulPixels[i][1])];
		}
		unsigned nPeaks;
		nPeaks = FindRegionalMaxima(z,UsefulPixels,NrPixelsThisRegion,MaximaPositions,MaximaValues,&IsSaturated,IntSat);
		if (NrPixelsThisRegion <= minNrPx || NrPixelsThisRegion >= maxNrPx){
			printf("Removed peak with %d pixels, position: %d %d.\n",NrPixelsThisRegion,MaximaPositions[0][0],MaximaPositions[0][1]);
			TotNrRegions--;
			continue;
		}
		if (IsSaturated == 1){ //Saturated peaks removed
			printf("Saturated peak removed.\n");
			TotNrRegions--;
			continue;
		}
		if (nPeaks > maxNPeaks){
			// Sort peaks by MaxIntensity, remove the smallest peaks until maxNPeaks, arrays needed MaximaPositions, MaximaValues.
			printf("nPeaks = %d, will be reduced to %d.\n",nPeaks,maxNPeaks);
			int MaximaPositionsT[nPeaks][2];
			double MaximaValuesT[nPeaks];
			double maxIntMax;
			int maxPos;
			for (i=0;i<maxNPeaks;i++){
				maxIntMax = 0;
				for (j=0;j<nPeaks;j++){
					if (MaximaValues[j] > maxIntMax){
						maxPos = j;
						maxIntMax = MaximaValues[j];
					}
				}
				MaximaPositionsT[i][0] = MaximaPositions[maxPos][0];
				MaximaPositionsT[i][1] = MaximaPositions[maxPos][1];
				MaximaValuesT[i] = MaximaValues[maxPos];
				MaximaValues[maxPos] = 0;
			}
			nPeaks = maxNPeaks;
			for (i=0;i<nPeaks;i++){
				MaximaValues[i] = MaximaValuesT[i];
				MaximaPositions[i][0] = MaximaPositionsT[i][0];
				MaximaPositions[i][1] = MaximaPositionsT[i][1];
			}
		}
		double *IntegratedIntensity, *IMAX, *YCEN, *ZCEN, *Rads, *Etass, *OtherInfo;
		IntegratedIntensity = malloc(nPeaks*2*sizeof(*IntegratedIntensity));
		memset(IntegratedIntensity,0,nPeaks*2*sizeof(*IntegratedIntensity));
		IMAX = malloc(nPeaks*2*sizeof(*IMAX));
		YCEN = malloc(nPeaks*2*sizeof(*YCEN));
		ZCEN = malloc(nPeaks*2*sizeof(*ZCEN));
		Rads = malloc(nPeaks*2*sizeof(*Rads));
		Etass = malloc(nPeaks*2*sizeof(*Etass));
		OtherInfo = malloc(nPeaks*10*sizeof(*OtherInfo));
		int *NrPx;
		NrPx = malloc(nPeaks*2*sizeof(*NrPx));
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&timer1);
		int rc = Fit2DPeaks(nPeaks,NrPixelsThisRegion,z,UsefulPixels,MaximaValues,MaximaPositions,IntegratedIntensity,IMAX,YCEN,ZCEN,Rads,Etass,Ycen,Zcen,Thresh,NrPx,OtherInfo);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&timer2);
		printf("%d %d %d %d %llf\n",RegNr,NrOfReg,NrPixelsThisRegion,nPeaks,diff(timer1,timer2));
		timex += diff(timer1,timer2);
		for (i=0;i<nPeaks;i++){
			fprintf(outfilewrite,"%d %f %f %f %f %f %f %f ",(SpotIDStart+i),IntegratedIntensity[i],Omega,-YCEN[i]+Ycen,ZCEN[i]+Zcen,IMAX[i],Rads[i],Etass[i]);
			for (j=0;j<2;j++) fprintf(outfilewrite, "%f ",OtherInfo[2*i+j]);
			fprintf(outfilewrite,"%d %d %d %d %d %f %f %f %d\n",NrPx[i],NrPixelsThisRegion,nPeaks,MaximaPositions[i][0],MaximaPositions[i][1],(double)MaximaPositions[i][0]-YCEN[i]-Ycen,(double)MaximaPositions[i][1]-ZCEN[i]-Zcen,MaximaValues[i],rc);
		}
		SpotIDStart += nPeaks;
		free(IntegratedIntensity);
		free(IMAX);
		free(YCEN);
		free(ZCEN);
		free(Rads);
		free(Etass);
		free(NrPx);
	}
	printf("Time spent in fitting: %llf\n",timex);
	printf("Number of regions = %d\n",TotNrRegions);
	printf("Number of peaks = %d\n",SpotIDStart-1);
	fclose(outfilewrite);
	free(ImgCorrBC);
	free(PositionTrackers);
	free(z);
	free(MaximaValues);
	FreeMemMatrixInt(Positions,nOverlapsMaxPerImage);
	FreeMemMatrixInt(MaximaPositions,NrPixels*10);
	FreeMemMatrixInt(UsefulPixels,NrPixels*10);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n",diftotal);
	return 0;
}
