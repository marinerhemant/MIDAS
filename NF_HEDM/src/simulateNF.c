//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <nlopt.h>
#include <stdint.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define RealType double
#define float32_t float
#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define ClearBit(A,k) (A[(k/32)] &= ~(1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-5
#define MAX_N_SPOTS 500
#define MAX_N_OMEGA_RANGES 20
#define MAX_POINTS_GRID_GOOD 300000

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;
int SpaceGrp;

double**
allocMatrixF(int nrows, int ncols)
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

int**
allocMatrixIntF(int nrows, int ncols)
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

struct my_func_data{
	int NrOfFiles;
    int nLayers;
    double ExcludePoleAngle;
    long long int SizeObsSpots;
    double XGrain[3];
    double YGrain[3];
    double OmegaStart;
    double OmegaStep;
    double px;
    double gs;
    double hkls[5000][4];
    int n_hkls;
    double Thetas[5000];
    int NoOfOmegaRanges;
    int NrPixelsGrid;
    double OmegaRanges[MAX_N_OMEGA_RANGES][2];
    double BoxSizes[MAX_N_OMEGA_RANGES][4];
    double **P0;
    int *ObsSpotsInfo;
    double *Lsd;
    double RotMatTilts[3][3];
    double *ybc;
    double *zbc;
};

static
double problem_function(
    unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_func_data *f_data = (struct my_func_data *) f_data_trial;
	int i, j, count = 1;
	const int NrOfFiles = f_data->NrOfFiles;
    const int nLayers = f_data->nLayers;
    const double ExcludePoleAngle = f_data->ExcludePoleAngle;
    const long long int SizeObsSpots = f_data->SizeObsSpots;
    double XGrain[3];
    double YGrain[3];
    const double OmegaStart = f_data->OmegaStart;
    const double OmegaStep = f_data->OmegaStep;
    const double px = f_data->px;
    const double gs = f_data->gs;
    const int NoOfOmegaRanges = f_data->NoOfOmegaRanges;
    const int NrPixelsGrid = f_data->NrPixelsGrid;
    double P0[nLayers][3];
    double OmegaRanges[MAX_N_OMEGA_RANGES][2];
    double BoxSizes[MAX_N_OMEGA_RANGES][4];
    double hkls[5000][4];
    int n_hkls = f_data->n_hkls;
    double Thetas[5000];
    for (i=0;i<5000;i++){
		hkls[i][0] = f_data->hkls[i][0];
		hkls[i][1] = f_data->hkls[i][1];
		hkls[i][2] = f_data->hkls[i][2];
		hkls[i][3] = f_data->hkls[i][3];
		Thetas[i] = f_data->Thetas[i];
	}
    int *ObsSpotsInfo;
	ObsSpotsInfo = &(f_data->ObsSpotsInfo[0]);
	double *Lsd;
	Lsd = &(f_data->Lsd[0]);
	double *ybc;
	ybc = &(f_data->ybc[0]);
	double *zbc;
	zbc = &(f_data->zbc[0]);
	for (i=0;i<3;i++){
		XGrain[i] = f_data->XGrain[i];
		YGrain[i] = f_data->YGrain[i];
		for (j=0;j<nLayers;j++){
			P0[j][i] = f_data->P0[j][i];
		}
	}
	for (i=0;i<MAX_N_OMEGA_RANGES;i++){
		for (j=0;j<2;j++){
			OmegaRanges[i][j] = f_data->OmegaRanges[i][j];
		}
		for (j=0;j<4;j++){
			BoxSizes[i][j] = f_data->BoxSizes[i][j];
		}
	}
	double RotMatTilts[3][3];
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			RotMatTilts[i][j] = f_data->RotMatTilts[i][j];
		}
	}
    double OrientMatIn[3][3], FracOverlap, x2[3];
    x2[0] = x[0]; x2[1] = x[1]; x2[2] = x[2];
    Euler2OrientMat(x2,OrientMatIn);
    CalcOverlapAccOrient(NrOfFiles,nLayers,ExcludePoleAngle,Lsd,SizeObsSpots,XGrain,
		YGrain,RotMatTilts,OmegaStart,OmegaStep,px,ybc,zbc,gs,hkls,n_hkls,
		Thetas,OmegaRanges,NoOfOmegaRanges,BoxSizes,P0,NrPixelsGrid,
		ObsSpotsInfo,OrientMatIn,&FracOverlap);
    return (1 - FracOverlap);
}

void
FitOrientation(
    const int NrOfFiles,
    const int nLayers,
    const double ExcludePoleAngle,
    double Lsd[nLayers],
    const long long int SizeObsSpots,
    const double XGrain[3],
    const double YGrain[3],
    double RotMatTilts[3][3],
    const double OmegaStart,
    const double OmegaStep,
    const double px,
    double ybc[nLayers],
    double zbc[nLayers],
    const double gs,
    double OmegaRanges[MAX_N_OMEGA_RANGES][2],
    const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4],
    double P0[nLayers][3],
    const int NrPixelsGrid,
    int *ObsSpotsInfo,
    double EulerIn[3],
    double tol,
    double *EulerOutA,
    double *EulerOutB,
    double *EulerOutC,
    double *ResultFracOverlap,
    double hkls[5000][4],
    double Thetas[5000],
    int n_hkls)
{
	unsigned n;
    long int i,j;
    n  = 3;
    double x[n],xl[n],xu[n];
    for( i=0; i<n; i++)
    {
        x[i] = EulerIn[i];
        xl[i] = x[i] - (tol*M_PI/180);
        xu[i] = x[i] + (tol*M_PI/180);
    }
	struct my_func_data f_data;
	f_data.NrOfFiles = NrOfFiles;
	f_data.nLayers = nLayers;
	f_data.n_hkls = n_hkls;
	for (i=0;i<5000;i++){
		f_data.hkls[i][0] = hkls[i][0];
		f_data.hkls[i][1] = hkls[i][1];
		f_data.hkls[i][2] = hkls[i][2];
		f_data.hkls[i][3] = hkls[i][3];
		f_data.Thetas[i] = Thetas[i];
	}
	f_data.ExcludePoleAngle = ExcludePoleAngle;
	f_data.SizeObsSpots = SizeObsSpots;
	f_data.P0 = allocMatrixF(nLayers,3);
	for (i=0;i<3;i++){
		f_data.XGrain[i] = XGrain[i];
		f_data.YGrain[i] = YGrain[i];
		for (j=0;j<nLayers;j++){
			f_data.P0[j][i] = P0[j][i];
		}
		for (j=0;j<3;j++){
			f_data.RotMatTilts[i][j] = RotMatTilts[i][j];
		}
	}
	for (i=0;i<MAX_N_OMEGA_RANGES;i++){
		for (j=0;j<2;j++){
			f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		}
		for (j=0;j<4;j++){
			f_data.BoxSizes[i][j] = BoxSizes[i][j];
		}
	}
	f_data.ObsSpotsInfo = &ObsSpotsInfo[0];
	f_data.Lsd = &Lsd[0];
	f_data.ybc = &ybc[0];
	f_data.zbc = &zbc[0];
	f_data.OmegaStart = OmegaStart;
	f_data.OmegaStep = OmegaStep;
	f_data.px = px;
	f_data.gs = gs;
	f_data.NoOfOmegaRanges = NoOfOmegaRanges;
	f_data.NrPixelsGrid = NrPixelsGrid;
	struct my_func_data *f_datat;
	f_datat = &f_data;
	void* trp = (struct my_func_data *) f_datat;
	double tole = 1e-3;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function, trp);
	double minf=1;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
    *ResultFracOverlap = minf;
    *EulerOutA = x[0];
    *EulerOutB = x[1];
    *EulerOutC = x[2];
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

static inline void
QuatToOrientMat(
    double Quat[4],
    double OrientMat[3][3])
{
    double Q1_2,Q2_2,Q3_2,Q12,Q03,Q13,Q02,Q23,Q01;
    Q1_2 = Quat[1]*Quat[1];
    Q2_2 = Quat[2]*Quat[2];
    Q3_2 = Quat[3]*Quat[3];
    Q12  = Quat[1]*Quat[2];
    Q03  = Quat[0]*Quat[3];
    Q13  = Quat[1]*Quat[3];
    Q02  = Quat[0]*Quat[2];
    Q23  = Quat[2]*Quat[3];
    Q01  = Quat[0]*Quat[1];
    OrientMat[0][0] = 1 - 2*(Q2_2+Q3_2);
    OrientMat[0][1] = 2*(Q12-Q03);
    OrientMat[0][2] = 2*(Q13+Q02);
    OrientMat[1][0] = 2*(Q12+Q03);
    OrientMat[1][1] = 1 - 2*(Q1_2+Q3_2);
    OrientMat[1][2] = 2*(Q23-Q01);
    OrientMat[2][0] = 2*(Q13-Q02);
    OrientMat[2][1] = 2*(Q23+Q01);
    OrientMat[2][2] = 1 - 2*(Q1_2+Q2_2);
}

int
main(int argc, char *argv[])
{
	if (argc != 4){
		printf("Usage:\n FitOrientation params.txt InputMicFN OutputFN\n");
		return 1;
	}

    clock_t start, end;
    double diftotal;
    start = clock();

    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
	char cmmd[4096];
	sprintf(cmmd,"~/opt/MIDAS/NF_HEDM/bin/GetHKLList %s",ParamFN);
	system(cmmd);
	return 1;
    char *MicFN = argv[3];
    char *outputFN = argv[4];
    char aline[4096];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[4096];
    int LowNr,nLayers;
    double tx,ty,tz;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "nDistances ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nLayers);
            break;
        }
    }
    rewind(fileParam);
    double Lsd[nLayers],ybc[nLayers],zbc[nLayers],ExcludePoleAngle,
		LatticeConstant[6], minFracOverlap,doubledummy,
		MaxRingRad,MaxTtheta;
    double px, OmegaStart,OmegaStep,tol;
	char fn[1000];
	char fn2[1000];
	char direct[1000];
	char gridfn[1000];
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
    int cntr=0,countr=0,conter=0,StartNr,EndNr,intdummy,SpaceGroup, RingsToUse[100],nRingsToUse=0;
    int NoOfOmegaRanges=0;
    int nSaves = 1;
    int gridfnfound = 0;
    Wedge = 0;
    int MinMiso = 0;
    while (fgets(aline,1000,fileParam)!=NULL){
		str = "ReducedFileName ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fn2);
            continue;
        }
		str = "GridFileName ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, gridfn);
            gridfnfound = 1;
            continue;
        }
		str = "SaveNSolutions ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &nSaves);
            continue;
        }
		str = "DataDirectory ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, direct);
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd[cntr]);
            cntr++;
            continue;
        }
        str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SpaceGroup);
            continue;
        }
        str = "MaxRingRad ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxRingRad);
            continue;
        }
        str = "StartNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &StartNr);
            continue;
        }
        str = "EndNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &EndNr);
            continue;
        }
        str = "ExcludePoleAngle ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ExcludePoleAngle);
            continue;
        }
        str = "LatticeParameter ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf %lf %lf", dummy,
				&LatticeConstant[0],&LatticeConstant[1],
				&LatticeConstant[2],&LatticeConstant[3],
				&LatticeConstant[4],&LatticeConstant[5]);
            continue;
        }
        str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
        str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &ty);
            continue;
        }
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &ybc[conter], &zbc[conter]);
            conter++;
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tz);
            continue;
        }
        str = "OrientTol ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tol);
            continue;
        }
        str = "MinFracAccept ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &minFracOverlap);
            continue;
        }
        str = "OmegaStart ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStart);
            continue;
        }
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "Wedge ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wedge);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
		str = "RingsToUse ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingsToUse[nRingsToUse]);
            nRingsToUse++;
            continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &OmegaRanges[NoOfOmegaRanges][0],&OmegaRanges[NoOfOmegaRanges][1]);
            NoOfOmegaRanges++;
            continue;
        }
        str = "BoxSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf", dummy, &BoxSizes[countr][0], &BoxSizes[countr][1], &BoxSizes[countr][2], &BoxSizes[countr][3]);
            countr++;
            continue;
        }
        str = "Ice9Input ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            Flag = 1;
            continue;
        }
        str = "NearestMisorientation ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&MinMiso);
            continue;
        }
    }
    int i,j,m,nrFiles,nrPixels;
    for (i=0;i<NoOfOmegaRanges;i++){
		OmegaRang[i][0] = OmegaRanges[i][0];
		OmegaRang[i][1] = OmegaRanges[i][1];
	}
    nOmeRang = NoOfOmegaRanges;
    fclose(fileParam);
    MaxTtheta = rad2deg*atan(MaxRingRad/Lsd[0]);
    char *ext="bin";
    int *ObsSpotsInfo;
    nrFiles = EndNr - StartNr + 1;
    nrPixels = 2048*2048;
    long long int SizeObsSpots;
    SizeObsSpots = (nLayers);
    SizeObsSpots*=nrPixels;
    SizeObsSpots*=nrFiles;
    SizeObsSpots;
    printf("%lld\n",SizeObsSpots);
    ObsSpotsInfo = calloc(SizeObsSpots,sizeof(*ObsSpotsInfo));

	double RotMatTilts[3][3];
	RotationTilts(tx,ty,tz,RotMatTilts);
	double MatIn[3],P0[nLayers][3],P0T[3];
	double xs,ys,edgeLen,gs,ud,eulThis[3],origConf,XG[3],YG[3],dy1,dy2;
	MatIn[0]=0;
	MatIn[1]=0;
	MatIn[2]=0;
	for (i=0;i<nLayers;i++){
		MatIn[0] = -Lsd[i];
		MatrixMultF(RotMatTilts,MatIn,P0T);
		for (j=0;j<3;j++){
			P0[i][j] = P0T[j];
		}
	}
	int n_hkls = 0;
	double hkls[5000][4];
	double Thetas[5000];
	char hklfn[1024];
	sprintf(hklfn,"%s/hkls.csv",direct);
	FILE *hklf = fopen(hklfn,"r");
	fgets(aline,1000,hklf);
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s",dummy,dummy,dummy,
			dummy,&hkls[n_hkls][3],&hkls[n_hkls][0],&hkls[n_hkls][1],
			&hkls[n_hkls][2],&Thetas[n_hkls],dummy,dummy);
		n_hkls++;
	}
	if (nRingsToUse > 0){
		double hklTemps[n_hkls][4],thetaTemps[n_hkls];
		int totalHKLs=0;
		for (i=0;i<nRingsToUse;i++){
			for (j=0;j<n_hkls;j++){
				if ((int)hkls[j][3] == RingsToUse[i]){
					hklTemps[totalHKLs][0] = hkls[j][0];
					hklTemps[totalHKLs][1] = hkls[j][1];
					hklTemps[totalHKLs][2] = hkls[j][2];
					hklTemps[totalHKLs][3] = hkls[j][3];
					thetaTemps[totalHKLs] = Thetas[j];
					totalHKLs++;
				}
			}
		}
		for (i=0;i<totalHKLs;i++){
			hkls[i][0] = hklTemps[i][0];
			hkls[i][1] = hklTemps[i][1];
			hkls[i][2] = hklTemps[i][2];
			hkls[i][3] = hklTemps[i][3];
			Thetas[i] = thetaTemps[i];
		}
		n_hkls = totalHKLs;
	}
	double OMIn[3][3], FracCalc;
	FILE *InpMicF;
	InpMicF = fopen(MicFN,"r");
	char outFN[4096];
	sprintf(outFN,"%s.output.csv",MicFN);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	fgets(aline,4096,InpMicF);
	int lineNr=0;
	while (fgets(aline,4096,InpMicF)!= NULL){
		sscanf(aline,"%s %s %s %lf %lf %lf %lf %lf %lf %lf %lf %s",dummy,dummy,dummy,&xs,&ys,&edgeLen,&ud,&eulThis[0],&eulThis[1],&eulThis[2],&origConf,dummy);
		gs = edgeLen/2;
		dy1 = edgeLen/sqrt(3);
		dy2 = -edgeLen/(2*sqrt(3));
		if (ud < 0){
			dy1 *= -1;
			dy2 *= -1;
		}
		int NrPixelsGrid=2*(ceil((gs*2)/px))*(ceil((gs*2)/px));
		XG[0] = xs;
		XG[1] = xs-gs;
		XG[2] = xs+gs;
		YG[0] = ys+dy1;
		YG[1] = ys+dy2;
		YG[2] = ys+dy2;
		Euler2OrientMat(eulThis,OMIn);
		SimulateAccOrient(nrFiles,nLayers,ExcludePoleAngle,Lsd,SizeObsSpots,XG,YG,RotMatTilts,OmegaStart,OmegaStep,px,ybc,zbc,gs,hkls,n_hkls,Thetas,OmegaRanges,NoOfOmegaRanges,BoxSizes,P0,NrPixelsGrid,ObsSpotsInfo,OMIn);
	}
	FILE *OutputF;
	OutputF = fopen(outputFN,"wb");
	char dummychar[8192];
	fwrite(dummychar,8192,1,OutputF);
	fwrite(ObsSpotsInfo,SizeObsSpots*sizeof(*ObsSpotsInfo),1,OutputF);
	return 0;
}
