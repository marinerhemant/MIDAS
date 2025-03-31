//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitTiltBCLsdSample.c
//
//
//  Created by Hemant Sharma on 2014/06/18.
//
//
//  Important: row major, starting with y's and going up. Bottom right is 0,0.

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

//~ #define PRINTOPT
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef uint16_t pixelvalue;
int NrCalls;
#define MultFactor 1
#define MaxNSpots 2000000

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
double R4mTtheta(double Ttheta, double Lsd)
{
	return Lsd*tan(deg2rad*Ttheta);
}

static inline
double Ttheta4mR(double R, double Lsd)
{
	return rad2deg*atan(R/Lsd);
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
void CalcWeightedMean(int nIndices, int *NrEachIndexBin, int **Indices, double *Average, double *R, double *Eta, double *RMean, double *EtaMean){
	int i,j,k;
	double TotIntensities[nIndices];
	for (i=0;i<nIndices;i++) TotIntensities[i]=0;
	for (i=0;i<nIndices;i++){
		for (j=0;j<NrEachIndexBin[i];j++){
			TotIntensities[i] += Average[Indices[i][j]];
		}
	}
	for (i=0;i<nIndices;i++){
		for (j=0;j<NrEachIndexBin[i];j++){
			RMean[i] += (Average[Indices[i][j]]*R[Indices[i][j]])/TotIntensities[i];
			EtaMean[i] += (Average[Indices[i][j]]*Eta[Indices[i][j]])/TotIntensities[i];
		}
	}
}

struct my_func_data{
	int nIndices;
	double *YMean;
	double *ZMean;
	double *IdealTtheta;
	double MaxRad;
	double px;
	double tx;
	double p0;
	double p1;
	double p2;
};

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

static
double problem_function(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_func_data *f_data = (struct my_func_data *) f_data_trial;
	double MaxRad = f_data->MaxRad;
	int nIndices = f_data->nIndices;
	double *YMean, *ZMean, *IdealTtheta, px;
	YMean = &(f_data->YMean[0]);
	ZMean = &(f_data->ZMean[0]);
	IdealTtheta = &(f_data->IdealTtheta[0]);
	px = f_data->px;
	double Lsd,ybc,zbc,tx,ty,tz,p0,p1,p2,txr,tyr,tzr;
	Lsd = x[0];
	ybc = x[1];
	zbc = x[2];
	tx  = f_data->tx;
	ty  = x[3];
	tz  = x[4];
	p0  = f_data->p0;
	p1  = f_data->p1;
	p2  = f_data->p2;
	txr = deg2rad*tx;
	tyr = deg2rad*ty;
	tzr = deg2rad*tz;
	double Rx[3][3] = {{1,0,0},{0,cos(txr),-sin(txr)},{0,sin(txr),cos(txr)}};
	double Ry[3][3] = {{cos(tyr),0,sin(tyr)},{0,1,0},{-sin(tyr),0,cos(tyr)}};
	double Rz[3][3] = {{cos(tzr),-sin(tzr),0},{sin(tzr),cos(tzr),0},{0,0,1}};
	double TRint[3][3], TRs[3][3];
	MatrixMultF33(Ry,Rz,TRint);
	MatrixMultF33(Rx,TRint,TRs);
	int i,j,k;
	double n0=2,n1=4,n2=2,Yc,Zc;
	double Rad, Eta, RNorm, DistortFunc, Rcorr, Theta, Diff, IdealTheta, TotalDiff=0, RIdeal,EtaT;
	double MeanErrorEtas[72];
	int nMeanErrorEtas[72];
	for (i=0;i<72;i++) MeanErrorEtas[i] = 0;
	for (i=0;i<72;i++) nMeanErrorEtas[i] = 0;
	int idx;
	for (i=0;i<nIndices;i++){
		Yc = -(YMean[i]-ybc)*px;
		Zc =  (ZMean[i]-zbc)*px;
		double ABC[3] = {0,Yc,Zc};
		double ABCPr[3];
		MatrixMult(TRs,ABC,ABCPr);
		double XYZ[3] = {Lsd+ABCPr[0],ABCPr[1],ABCPr[2]};
		Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
		Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
		RNorm = Rad/MaxRad;
		EtaT = 90 - Eta;
		DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT)))) + (p2*(pow(RNorm,n2))) + 1;
		Rcorr = Rad * DistortFunc;
		RIdeal = Lsd*tan(deg2rad*IdealTtheta[i]);
		Diff = fabs(1- (Rcorr/RIdeal));
//~ #ifdef PRINTOPT
		//~ printf("%lf %lf %lf\n",Lsd,IdealTtheta[i],Diff);
//~ #endif
		//~ TotalDiff+=Diff;
		idx = (Eta + 180)/5;
		MeanErrorEtas[idx] += Diff;
		nMeanErrorEtas[idx] ++;
	}
	TotalDiff = 0;
	for (i=0;i<72;i++){
		if (nMeanErrorEtas[i] != 0){
			MeanErrorEtas[i] /= (double)nMeanErrorEtas[i];
			TotalDiff+= MeanErrorEtas[i];
		}
	}
	TotalDiff *= MultFactor;
	NrCalls++;
#ifdef PRINTOPT
	printf("Mean Strain: %0.40f ty: %lf tz: %lf bc: %lf %lf Lsd: %lf\n",TotalDiff/(MultFactor*nIndices),ty,tz,ybc,zbc,Lsd);
#endif
	return TotalDiff;
}

void FitTiltBCLsd(int nIndices, double *YMean, double *ZMean, double *IdealTtheta, double Lsd, double MaxRad,
				  double ybc, double zbc, double tx, double tyIn, double tzIn, double *ty, double *tz, double *LsdFit,
				  double *ybcFit, double *zbcFit, double p0, double p1, double p2, double *MeanDiff, double tolTilts, double tolLsd, double tolBC, double px){
	unsigned n=5;
	struct my_func_data f_data;
	f_data.nIndices = nIndices;
	f_data.YMean = &YMean[0];
	f_data.ZMean = &ZMean[0];
	f_data.IdealTtheta = &IdealTtheta[0];
	f_data.MaxRad = MaxRad;
	f_data.px = px;
	f_data.tx = tx;
	f_data.p0 = p0;
	f_data.p1 = p1;
	f_data.p2 = p2;
	double x[n], xl[n], xu[n];
	x[0] =  Lsd;  xl[0] = Lsd - tolLsd;   xu[0] = Lsd + tolLsd;
	x[1] =  ybc;  xl[1] = ybc - tolBC;    xu[1] = ybc + tolBC;
	x[2] =  zbc;  xl[2] = zbc - tolBC;    xu[2] = zbc + tolBC;
	x[3] = tyIn;  xl[3] = tyIn- tolTilts; xu[3] = tyIn+ tolTilts;
	x[4] = tzIn;  xl[4] = tzIn- tolTilts; xu[4] = tzIn+ tolTilts;
	struct my_func_data *f_datat;
	f_datat = &f_data;
	void* trp = (struct my_func_data *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function, trp);
	double minf;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
	*MeanDiff = minf/(MultFactor*nIndices);
	*LsdFit = x[0];
	*ybcFit = x[1];
	*zbcFit = x[2];
	*ty     = x[3];
	*tz     = x[4];
}

static inline
void CorrectTiltSpatialDistortion(int nIndices, double MaxRad, double *YMean, double *ZMean,
		double px, double Lsd, double ybc, double zbc, double tx, double ty, double tz, double p0, double p1,
		double p2, double *YCorrected, double *ZCorrected)
{
	double txr,tyr,tzr;
	txr = deg2rad*tx;
	tyr = deg2rad*ty;
	tzr = deg2rad*tz;
	double Rx[3][3] = {{1,0,0},{0,cos(txr),-sin(txr)},{0,sin(txr),cos(txr)}};
	double Ry[3][3] = {{cos(tyr),0,sin(tyr)},{0,1,0},{-sin(tyr),0,cos(tyr)}};
	double Rz[3][3] = {{cos(tzr),-sin(tzr),0},{sin(tzr),cos(tzr),0},{0,0,1}};
	double TRint[3][3], TRs[3][3];
	MatrixMultF33(Ry,Rz,TRint);
	MatrixMultF33(Rx,TRint,TRs);
	int i,j,k;
	double n0=2,n1=4,n2=2,Yc,Zc;
	double Rad, Eta, RNorm, DistortFunc, Rcorr, EtaT;
	for (i=0;i<nIndices;i++){
		Yc = -(YMean[i]-ybc)*px;
		Zc =  (ZMean[i]-zbc)*px;
		double ABC[3] = {0,Yc,Zc};
		double ABCPr[3];
		MatrixMult(TRs,ABC,ABCPr);
		double XYZ[3] = {Lsd+ABCPr[0],ABCPr[1],ABCPr[2]};
		Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]));
		Eta = CalcEtaAngle(XYZ[1],XYZ[2]);
		RNorm = Rad/MaxRad;
		EtaT = 90 - Eta;
		DistortFunc = (p0*(pow(RNorm,n0))*(cos(deg2rad*(2*EtaT)))) + (p1*(pow(RNorm,n1))*(cos(deg2rad*(4*EtaT)))) + (p2*(pow(RNorm,n2))) + 1;
		Rcorr = Rad * DistortFunc;
		YCorrected[i] = -Rcorr*sin(deg2rad*Eta);
		ZCorrected[i] =  Rcorr*cos(deg2rad*Eta);
	}
}

static inline
void CorrectWedge(double yc, double zc, double Lsd, double OmegaIni, double wl, double wedge, double *ysOut, double *zsOut, double *OmegaOut, double *EtaOut, double *TthetaOut)
{
	double ysi = yc, zsi = zc;
	double CosOme=cos(deg2rad*OmegaIni), SinOme=sin(deg2rad*OmegaIni);
	double eta = CalcEtaAngle(ysi,zsi);
	double RingRadius = sqrt((ysi*ysi)+(zsi*zsi));
	double tth = rad2deg*atan(RingRadius/Lsd);
	double theta = tth/2;
	double SinTheta = sin(deg2rad*theta);
	double CosTheta = cos(deg2rad*theta);
	double ds = 2*SinTheta/wl;
	double CosW = cos(deg2rad*wedge);
	double SinW = sin(deg2rad*wedge);
	double SinEta = sin(deg2rad*eta);
	double CosEta = cos(deg2rad*eta);
	double k1 = -ds*SinTheta;
	double k2 = -ds*CosTheta*SinEta;
	double k3 =  ds*CosTheta*CosEta;
	if (eta == 90){k3 = 0; k2 = -CosTheta;}
	else if (eta == -90) {k3 = 0; k2 = CosTheta;}
	double k1f = (k1*CosW) + (k3*SinW);
	double k2f = k2;
	double k3f = (k3*CosW) - (k1*SinW);
	double G1a = (k1f*CosOme) + (k2f*SinOme);
	double G2a = (k2f*CosOme) - (k1f*SinOme);
	double G3a = k3f;
	double LenGa = sqrt((G1a*G1a)+(G2a*G2a)+(G3a*G3a));
	double g1 = G1a*ds/LenGa;
	double g2 = G2a*ds/LenGa;
	double g3 = G3a*ds/LenGa;
	SinW = 0;
	CosW = 1;
	double LenG = sqrt((g1*g1)+(g2*g2)+(g3*g3));
	double k1i = -(LenG*LenG*wl)/2;
	tth = 2*rad2deg*asin(wl*LenG/2);
	RingRadius = Lsd*tan(deg2rad*tth);
	double A = (k1i+(g3*SinW))/(CosW);
	double a_Sin = (g1*g1) + (g2*g2);
	double b_Sin = 2*A*g2;
	double c_Sin = (A*A) - (g1*g1);
	double a_Cos = a_Sin;
	double b_Cos = -2*A*g1;
	double c_Cos = (A*A) - (g2*g2);
	double Par_Sin = (b_Sin*b_Sin) - (4*a_Sin*c_Sin);
	double Par_Cos = (b_Cos*b_Cos) - (4*a_Cos*c_Cos);
	double P_check_Sin = 0;
	double P_check_Cos = 0;
	double P_Sin,P_Cos;
	if (Par_Sin >=0) P_Sin=sqrt(Par_Sin);
	else {P_Sin=0;P_check_Sin=1;}
	if (Par_Cos>=0) P_Cos=sqrt(Par_Cos);
	else {P_Cos=0;P_check_Cos=1;}
	double SinOmega1 = (-b_Sin-P_Sin)/(2*a_Sin);
	double SinOmega2 = (-b_Sin+P_Sin)/(2*a_Sin);
	double CosOmega1 = (-b_Cos-P_Cos)/(2*a_Cos);
	double CosOmega2 = (-b_Cos+P_Cos)/(2*a_Cos);
	if      (SinOmega1 < -1) SinOmega1=-1;
	else if (SinOmega1 >  1) SinOmega1=1;
	else if (SinOmega2 < -1) SinOmega2=-1;
	else if (SinOmega2 >  1) SinOmega2=1;
	if      (CosOmega1 < -1) CosOmega1=-1;
	else if (CosOmega1 >  1) CosOmega1=1;
	else if (CosOmega2 < -1) CosOmega2=-1;
	else if (CosOmega2 >  1) CosOmega2=1;
	if (P_check_Sin == 1){SinOmega1=0;SinOmega2=0;}
	if (P_check_Cos == 1){CosOmega1=0;CosOmega2=0;}
	double Option1 = fabs((SinOmega1*SinOmega1)+(CosOmega1*CosOmega1)-1);
	double Option2 = fabs((SinOmega1*SinOmega1)+(CosOmega2*CosOmega2)-1);
	double Omega1, Omega2;
	if (Option1 < Option2){Omega1=rad2deg*atan2(SinOmega1,CosOmega1);Omega2=rad2deg*atan2(SinOmega2,CosOmega2);}
	else {Omega1=rad2deg*atan2(SinOmega1,CosOmega2);Omega2=rad2deg*atan2(SinOmega2,CosOmega1);}
	double OmeDiff1 = fabs(Omega1-OmegaIni);
	double OmeDiff2 = fabs(Omega2-OmegaIni);
	if (fabs(OmeDiff1-360) < 0.1){
		OmeDiff1 = 0; Omega1 *= -1;
	}
	if (fabs(OmeDiff2-360) < 0.1){
		OmeDiff2 = 0; Omega2 *= -1;
	}
	double Omega;
	if (OmeDiff1 < OmeDiff2)Omega=Omega1;
	else Omega=Omega2;
	double SinOmega=sin(deg2rad*Omega);
	double CosOmega=cos(deg2rad*Omega);
	double Fact = (g1*CosOmega) - (g2*SinOmega);
	double k2N  = (g1*SinOmega) + (g2*CosOmega);
	double k3N  = (SinW*Fact)   + (g3*CosW);
	double Eta = CalcEtaAngle(k2,k3);
	double Sin_Eta = sin(deg2rad*Eta);
	double Cos_Eta = cos(deg2rad*Eta);
	*ysOut = -RingRadius*Sin_Eta;
	*zsOut = RingRadius*Cos_Eta;
	*OmegaOut = Omega;
	*EtaOut = Eta;
	*TthetaOut = rad2deg*atan(RingRadius/Lsd);
}

struct SpotsData{
	double SpotID;
	double Omega;
	double Y;
	double Z;
	double RingNr;
	double Radius;
	double IntInt;
};

static int cmpfunc (const void * a, const void *b){
	struct SpotsData *ia = (struct SpotsData *)a;
	struct SpotsData *ib = (struct SpotsData *)b;
	return (int)(1000.f*ia->Omega - 1000.f*ib->Omega);
}

static inline void SortSpots(int nIndices, double **SpotsInfo){
	struct SpotsData *MyData;
    MyData = malloc(nIndices*sizeof(*MyData));
    int i,j,k;
    for (i=0;i<nIndices;i++){
		MyData[i].SpotID = SpotsInfo[i][0];
		MyData[i].Omega = SpotsInfo[i][1];
		MyData[i].Y = SpotsInfo[i][2];
		MyData[i].Z = SpotsInfo[i][3];
		MyData[i].RingNr = SpotsInfo[i][4];
		MyData[i].Radius = SpotsInfo[i][5];
		MyData[i].IntInt = SpotsInfo[i][6];
	}
	qsort(MyData, nIndices, sizeof(struct SpotsData), cmpfunc);
	for (i=0;i<nIndices;i++){
		SpotsInfo[i][0] = MyData[i].SpotID;
		SpotsInfo[i][1] = MyData[i].Omega;
		SpotsInfo[i][2] = MyData[i].Y;
		SpotsInfo[i][3] = MyData[i].Z;
		SpotsInfo[i][4] = MyData[i].RingNr;
		SpotsInfo[i][5] = MyData[i].Radius;
		SpotsInfo[i][6] = MyData[i].IntInt;
	}
	free(MyData);
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		printf("Usage: %s Params.txt\n", argv[0]);
		exit(EXIT_FAILURE);
	}
    clock_t start, end;
    double diftotal;
    start = clock();
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    char folder[1024],Folder[1024],*spotsfn,outfolder[1024],resultfolder[1024],*idfn,FileStem[1024],fs[1024];
    int StartNr, EndNr, LowNr, NrPixels,LayerNr;
    int SpaceGroup;
    double LatticeConstant[6],Wavelength,MaxRingRad,Lsd,MaxTtheta,TthetaTol,ybc,zbc,px,tyIn,tzIn, BeamSize = 0;
    double tx,tolTilts=1,tolLsd=5000,tolBC=1,p0,p1,p2,RhoD,wedge,MinEta,OmegaRanges[2000][2],BoxSizes[2000][4];
    int RingNumbers[200],cs=0,nOmeRanges=0,nBoxSizes=0,DoFit=0,RingToIndex;
    double Rsample, Hbeam,MinMatchesToAcceptFrac,MinOmeSpotIDsToIndex,MaxOmeSpotIDsToIndex,Width;
    int UseFriedelPairs=1;
	double t_int=1, t_gap=0;
    int NewType = 1, TopLayer = 0;
    int maxNFrames = 100000, SGnum = 225;
    spotsfn = "InputAll.csv";
    idfn = "SpotsToIndex.csv";
    double StepSizePos=5,StepSizeOrient=0.2,MarginRadius=500,MarginRadial=500,OmeBinSize=0.1,EtaBinSize=0.1,MarginEta=500,MarginOme=0.5,OmegaStep,MargABC=2.0,MargABG=2.0;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "OmegaStep ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmegaStep);
            continue;
        }
        str = "StepSizePos ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &StepSizePos);
            continue;
        }
        str = "MaxNFrames ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &maxNFrames);
            continue;
        }
        str = "NrPixels ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "tInt ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &t_int);
            continue;
        }
        str = "tGap ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &t_gap);
            continue;
        }
        str = "StepSizeOrient ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &StepSizeOrient);
            continue;
        }
        str = "MarginRadius ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MarginRadius);
            continue;
        }
        str = "MarginRadial ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MarginRadial);
            continue;
        }
        str = "MarginEta ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MarginEta);
            continue;
        }
        str = "MarginOme ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MarginOme);
            continue;
        }
        str = "MargABC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MargABC);
            continue;
        }
        str = "MargABG ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MargABG);
            continue;
        }
        str = "OmeBinSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &OmeBinSize);
            continue;
        }
        str = "EtaBinSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &EtaBinSize);
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
        str = "LayerNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "UseFriedelPairs ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &UseFriedelPairs);
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
        str = "DoFit ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &DoFit);
            continue;
        }
        str = "OverAllRingToIndex ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingToIndex);
            continue;
        }
        str = "LatticeConstant ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf %lf %lf", dummy,
				&LatticeConstant[0],&LatticeConstant[1],
				&LatticeConstant[2],&LatticeConstant[3],
				&LatticeConstant[4],&LatticeConstant[5]);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "MaxRingRad ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxRingRad);
            continue;
        }
        str = "MinEta ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MinEta);
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd);
            continue;
        }
        str = "MinOmeSpotIDsToIndex ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MinOmeSpotIDsToIndex);
            continue;
        }
        str = "MaxOmeSpotIDsToIndex ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxOmeSpotIDsToIndex);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
        str = "BeamSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &BeamSize);
            continue;
        }
        str = "Width ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Width);
            continue;
        }
        str = "BC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &ybc, &zbc);
            continue;
        }
        str = "tolTilts ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tolTilts);
            continue;
        }
        str = "tolBC ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tolBC);
            continue;
        }
        str = "tolLsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tolLsd);
            continue;
        }
        str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SGnum);
            continue;
        }
        str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
        str = "Hbeam ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Hbeam);
            continue;
        }
        str = "Rsample ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Rsample);
            continue;
        }
        str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tyIn);
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tzIn);
            continue;
        }
        str = "Wedge ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &wedge);
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
        str = "p2 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p2);
            continue;
        }
        str = "RhoD ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &RhoD);
            continue;
        }
        str = "Completeness ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MinMatchesToAcceptFrac);
            continue;
        }
        str = "RingThresh ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingNumbers[cs]);
            cs++;
            continue;
        }
        str = "NewType ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NewType);
            continue;
        }
        str = "TopLayer ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &TopLayer);
            continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy, &OmegaRanges[nOmeRanges][0], &OmegaRanges[nOmeRanges][1]);
            nOmeRanges++;
            continue;
        }
        str = "BoxSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf", dummy, &BoxSizes[nBoxSizes][0], &BoxSizes[nBoxSizes][1],
													  &BoxSizes[nBoxSizes][2], &BoxSizes[nBoxSizes][3]);
            nBoxSizes++;
            continue;
        }
	}
	sprintf(FileStem,"%s_%d",fs,LayerNr);
	if (nOmeRanges != nBoxSizes){printf("Number of omega ranges and number of box sizes don't match. Exiting!");return 1;}
	MaxTtheta = rad2deg*atan(MaxRingRad/Lsd);
	double **SpotsInfo;
	SpotsInfo = allocMatrix(MaxNSpots,7);
	char FileName[1024];
	int i,j,k;
	if (TopLayer == 1){
		for (i=0;i<nBoxSizes;i++){
			BoxSizes[i][3] = 0;
		}
	}
	FILE *fp;
	//sprintf(FileName,"%s/%s",folder,fn);
	sprintf(folder,"%s/",Folder);
	sprintf(FileName,"%s/Radius_StartNr_%d_EndNr_%d.csv",Folder,StartNr,EndNr);
	char line[5024];
	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(aline,1000,hklf);
	int Rnr;
	double tht;
	int PlaneNumbers[cs], donePlanes[cs];
	double Thetas[cs];
	double RingRadsIdeal[cs], ds[cs], rrdideal, dsthis;
	int n_hkls = cs,nhkls = 0;
	for (i=0;i<cs;i++) donePlanes[i] = 0;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %lf %d %s %s %s %lf %s %lf",dummy,dummy,dummy,&dsthis,&Rnr,dummy,dummy,dummy,&tht,dummy,&rrdideal);
		if (tht > MaxTtheta/2) break;
		for (i=0;i<cs;i++){
			if(Rnr == RingNumbers[i] && donePlanes[i] == 0){
				donePlanes[i] = 1;
				Thetas[nhkls] = tht;
				PlaneNumbers[nhkls] = Rnr;
				RingRadsIdeal[nhkls] = rrdideal;
				ds[nhkls] = dsthis;
				nhkls++;
				break;
			}
		}
	}
	TthetaTol = Ttheta4mR((MaxRingRad+Width),Lsd) - Ttheta4mR((MaxRingRad-Width),Lsd);
	double IdealTthetas[n_hkls], TthetaMins[n_hkls], TthetaMaxs[n_hkls];
	for (i=0;i<n_hkls;i++){IdealTthetas[i]=2*Thetas[i];TthetaMins[i]=IdealTthetas[i]-TthetaTol;TthetaMaxs[i]=IdealTthetas[i]+TthetaTol;}
	double IdealRs[n_hkls], Rmins[n_hkls], Rmaxs[n_hkls];
	for (i=0;i<n_hkls;i++){IdealRs[i]=R4mTtheta(IdealTthetas[i],Lsd);Rmins[i]=R4mTtheta(TthetaMins[i],Lsd);Rmaxs[i]=R4mTtheta(TthetaMaxs[i],Lsd);}
	int counter = 0;
	double nFramesThis;
	int nSpotsEachRing[n_hkls];
	for (i=0;i<n_hkls;i++) nSpotsEachRing[i] = 0;
	fp = fopen(FileName,"r");
	printf("Reading file: %s.\n",FileName);
	fgets(line,5000,fp);
	while (fgets(line,5000,fp) != NULL){
		sscanf(line,"%lf %lf %lf %lf %lf %s %s %s %s %s %s %s %lf %lf %s %lf %s %s %s",
			&SpotsInfo[counter][0],&SpotsInfo[counter][6],&SpotsInfo[counter][1],&SpotsInfo[counter][2],&SpotsInfo[counter][3],
			dummy,dummy,dummy,dummy,dummy,dummy,dummy,&nFramesThis,&SpotsInfo[counter][4],dummy,&SpotsInfo[counter][5],dummy,dummy,dummy);
		for (i=0;i<n_hkls;i++) if ((int)SpotsInfo[counter][4] == PlaneNumbers[i]) nSpotsEachRing[i]++;
		if ((int)nFramesThis > maxNFrames) continue; // Overwrite the spot if nFrames is greater than maxNFrames
		counter++;
	}
	//~ }else if (NewType == 2){ // Fable system
		//~ char *fltfn = argv[2];
		//~ FILE *fltfile = fopen(fltfn,"r");
		//~ fgets(line,5000,fltfile); // Skip header.
		//~ printf("Reading file: %s.\n",fltfn);
		//~ double y,z,ome,Rtemp,sumInt;
		//~ int IDtemp;
		//~ while (fgets(line,5000,fltfile) != NULL){
			//~ sscanf(line,"%lf %lf %lf %s %s %s %s %s %s %s %s %s %s %lf %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %d %s %s %s %s %s %s %s %s %s %s %s %s",
				//~ &z,&y,&ome,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,
				//~ dummy,dummy,&sumInt,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,
				//~ dummy,dummy,dummy,dummy,dummy,dummy,dummy,&IDtemp,dummy,dummy,dummy,
				//~ dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy);
			//~ //y = 2048.0 - y;
			//~ //z = 2048.0 - z;
			//~ Rtemp = sqrt((y-ybc)*(y-ybc) + (z-zbc)*(z-zbc)) * px;
			//~ //printf("%d %lf %lf %lf %lf %f %f %f %d\n",IDtemp,y,z,ome,Rtemp,Rmins[0],Rmaxs[0],IdealRs[0],n_hkls);
			//~ if (Rtemp > Rmins[0] && Rtemp < Rmaxs[0]){
				//~ SpotsInfo[counter][0] = IDtemp;
				//~ SpotsInfo[counter][1] = ome;
				//~ SpotsInfo[counter][2] = y;
				//~ SpotsInfo[counter][3] = z;
				//~ SpotsInfo[counter][4] = (int)PlaneNumbers[0];
				//~ //printf("%f\n",SpotsInfo[counter][4]);
				//~ SpotsInfo[counter][5] = sumInt;
				//~ counter++;
			//~ }
		//~ }
	//~ }
	printf("Number spots per ring: ");
	for (i=0;i<n_hkls;i++) printf("%d ",nSpotsEachRing[i]); printf("\n");
	int nIndices = counter;
	printf("Number of planes being considered: %d.\nNumber of spots: %d.\n",n_hkls,nIndices);
	for (i=0;i<nIndices;i++){
		// Omega correction
		SpotsInfo[i][1] = SpotsInfo[i][1] - (t_gap/(t_gap+t_int))*OmegaStep*(1.0 - fabs(2*SpotsInfo[i][3] - (double)NrPixels) /(double) NrPixels);
		if (SpotsInfo[i][1] < -180) SpotsInfo[i][1] += 360;
		if (SpotsInfo[i][1] > 180) SpotsInfo[i][1] -= 360;
	}
	// Sort spots per ring
    char fnidhsh[1024];
    FILE *idhsh;
    sprintf(fnidhsh,"%s/IDRings.csv",folder);
    idhsh = fopen(fnidhsh,"w");
    fprintf(idhsh,"RingNumber OriginalID NewID(RingsMerge)\n");
	FILE *idshashout;
	char fnidshash[1024];
    sprintf(fnidshash,"%s/IDsHash.csv",folder);
    printf("%s\n%s\n",fnidshash,fnidhsh);
    idshashout = fopen(fnidshash,"w");
	int nSpotsThis,nctr=0,colN,startrowN=0;
	double **spotsall;
	spotsall = allocMatrix(nIndices,7);
	for (i=0;i<n_hkls;i++){
		double **spotsTemp;
		nSpotsThis = nSpotsEachRing[i];
		spotsTemp = allocMatrix(nSpotsThis,7);
		nctr = 0;
		for (j=0;j<nIndices;j++){
			if (SpotsInfo[j][4] == PlaneNumbers[i]){
				for (colN=0;colN<7;colN++) spotsTemp[nctr][colN] = SpotsInfo[j][colN];
				nctr++;
			}
		}
		SortSpots(nSpotsThis,spotsTemp);
		for (j=0;j<nSpotsThis;j++){
			spotsall[j+startrowN][0] = j+startrowN+1;
			for (colN=1;colN<7;colN++){
				spotsall[j+startrowN][colN] = spotsTemp[j][colN];
			}
			fprintf(idhsh,"%d %d %d\n",PlaneNumbers[i],(int)spotsTemp[j][0],(int)spotsall[j+startrowN][0]);
		}
		fprintf(idshashout,"%d %d %d %lf\n",PlaneNumbers[i],startrowN+1,startrowN+nSpotsThis+1,ds[i]);
		startrowN += nSpotsThis;
		FreeMemMatrix(spotsTemp,nSpotsThis);
	}
	fclose(idhsh);
	fclose(idshashout);
	for (i=0;i<nIndices;i++) for (j=0;j<7;j++) SpotsInfo[i][j] = spotsall[i][j];
	FreeMemMatrix(spotsall,nIndices);
	double *Ys, *Zs, *IdealTtheta,omegaCorrTemp;
	Ys = malloc(nIndices*sizeof(*Ys));
	Zs = malloc(nIndices*sizeof(*Zs));
	IdealTtheta = malloc(nIndices*sizeof(*IdealTtheta));
	for (i=0;i<nIndices;i++){
		Ys[i]=SpotsInfo[i][2];
		Zs[i]=SpotsInfo[i][3];
		for (j=0;j<n_hkls;j++){
			if (PlaneNumbers[j] == (int)SpotsInfo[i][4]){
				IdealTtheta[i] = IdealTthetas[j];
				break;
			}
		}
	}
	double ty,tz,LsdFit,ybcFit,zbcFit,MeanDiff;
	if (DoFit == 1){
		printf("Fitting parameters.\n");
		FitTiltBCLsd(nIndices,Ys,Zs,IdealTtheta,Lsd,RhoD,ybc,zbc,tx,tyIn,tzIn,&ty,&tz,&LsdFit,&ybcFit,&zbcFit,p0,p1,p2,&MeanDiff,tolTilts,tolLsd,tolBC,px);
		printf("Number of function calls: %d\n",NrCalls);
		printf("LsdFit:\t\t%0.12f\nYBCFit:\t\t%0.12f\nZBCFit:\t\t%0.12f\ntyFit:\t\t%0.12f\ntzFit:\t\t%0.12f\nMeanStrain:\t%0.12lf\n",
			LsdFit,ybcFit,zbcFit,ty,tz,MeanDiff);
	} else {
		printf("Fitting not used. Using intial values for final results.\n");LsdFit = Lsd;ty = tyIn;tz = tzIn;ybcFit = ybc;zbcFit = zbc;
	}
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
	double *YCorrected, *ZCorrected;
	YCorrected = malloc(nIndices*sizeof(*YCorrected));
	ZCorrected = malloc(nIndices*sizeof(*ZCorrected));
	CorrectTiltSpatialDistortion(nIndices,RhoD,Ys,Zs,px,LsdFit,ybcFit,zbcFit,tx,ty,tz,p0,p1,p2,YCorrected,ZCorrected);
	double *YCorrWedge,*ZCorrWedge,*OmegaCorrWedge,*EtaCorrWedge,*TthetaCorrWedge,YCorrWedgeT,ZCorrWedgeT,OmegaCorrWedgeT,EtaCorrWedgeT,TthetaCorrWedgeT;
	YCorrWedge = malloc(nIndices*sizeof(*YCorrWedge));
	ZCorrWedge = malloc(nIndices*sizeof(*ZCorrWedge));
	OmegaCorrWedge = malloc(nIndices*sizeof(*OmegaCorrWedge));
	EtaCorrWedge = malloc(nIndices*sizeof(*EtaCorrWedge));
	TthetaCorrWedge = malloc(nIndices*sizeof(*TthetaCorrWedge));
	for (i=0;i<nIndices;i++){
		CorrectWedge(YCorrected[i],ZCorrected[i],LsdFit,SpotsInfo[i][1],Wavelength,wedge,&YCorrWedgeT,&ZCorrWedgeT,&OmegaCorrWedgeT,&EtaCorrWedgeT,&TthetaCorrWedgeT);
		YCorrWedge[i] = YCorrWedgeT;
		ZCorrWedge[i] = ZCorrWedgeT;
		OmegaCorrWedge[i] = OmegaCorrWedgeT;
		//~ if (fabs(OmegaCorrWedgeT-SpotsInfo[i][1])>0.1) printf("%lf %lf %lf %lf %lf %lf\n",YCorrected[i],YCorrWedgeT,ZCorrected[i],ZCorrWedgeT,SpotsInfo[i][1],OmegaCorrWedgeT);
		EtaCorrWedge[i] = EtaCorrWedgeT;
		TthetaCorrWedge[i] = TthetaCorrWedgeT;
	}
	//Useful arrays till now: SpotsInfo,YCorrected,ZCorrected,YCorrWedge,ZCorrWedge,OmegaCorrWedge,EtaCorrWedge
	int NumberSpotsToKeep=0;
	int *goodRows;
	goodRows = calloc(nIndices,sizeof(*goodRows));
	int KeepSpot,nSpotIDsToIndex=0,*SpotIDsToIndex;
	SpotIDsToIndex = malloc(nIndices*sizeof(*SpotIDsToIndex));
	int UniqueRingNumbers[200], nrUniqueRingNumbers=0,RingNumberThis,RingNumberPresent=0,nRejects = 0;
	for (i=0;i<nIndices;i++){
		if (((EtaCorrWedge[i] > (-180+MinEta)) && (EtaCorrWedge[i] < -MinEta))|| ((EtaCorrWedge[i] > MinEta) && (EtaCorrWedge[i] < (180-MinEta)))){
			KeepSpot = 0;
			for (j=0;j<nOmeRanges;j++){
				if ((OmegaCorrWedge[i]>=OmegaRanges[j][0])&&(OmegaCorrWedge[i]<=OmegaRanges[j][1])&&(YCorrWedge[i]>BoxSizes[j][0])
				  &&(YCorrWedge[i]<BoxSizes[j][1])&&(ZCorrWedge[i]>BoxSizes[j][2])&&(ZCorrWedge[i]<BoxSizes[j][3])){KeepSpot=1;break;}
			}
			if (KeepSpot == 1){
				goodRows[i] = 1;
				NumberSpotsToKeep++;
				RingNumberThis = (int)(SpotsInfo[i][4]);
				RingNumberPresent = 0;
				if (RingNumberThis == RingToIndex && OmegaCorrWedge[i]>=MinOmeSpotIDsToIndex && OmegaCorrWedge[i]<=MaxOmeSpotIDsToIndex){
					SpotIDsToIndex[nSpotIDsToIndex] = SpotsInfo[i][0];
					nSpotIDsToIndex++;
				}
				for (j=0;j<nrUniqueRingNumbers;j++){
					if (RingNumberThis == UniqueRingNumbers[j]){RingNumberPresent = 1; break;}
				}
				if (RingNumberPresent == 0){
					UniqueRingNumbers[nrUniqueRingNumbers] = RingNumberThis;
					nrUniqueRingNumbers++;
				}
			} else{
				nRejects++;
			}
		} else {
			nRejects++;
		}
	}
	printf("nRejects: %d, nIndices: %d, Spots to keep: %d, SpotIDsToIndex: %d\n",nRejects,nIndices,NumberSpotsToKeep,nSpotIDsToIndex);
	FILE *IndexAll, *IndexAllNoHeader, *ExtraInfo, *IDs, *PF;
	char fnIndexAll[2048],fnIndexAllNoHeader[2048],fnExtraInfo[2048],fnSpIds[1024],parfn[1024];
	sprintf(parfn,"%s/paramstest.txt",folder);
	sprintf(fnIndexAll,"%s/InputAll.csv",folder);
	sprintf(outfolder,"%s/Output",folder);
	sprintf(resultfolder,"%s/Results",folder);
	sprintf(fnIndexAllNoHeader,"%s/InputAllNoHeader.csv",folder);
	sprintf(fnExtraInfo,"%s/InputAllExtraInfoFittingAll.csv",folder);
	sprintf(fnSpIds,"%s/%s",folder,idfn);
	IDs = fopen(fnSpIds,"w");
	for (i=0;i<nSpotIDsToIndex;i++){
		fprintf(IDs,"%d\n",SpotIDsToIndex[i]);
	}
	fclose(IDs);
	IndexAll = fopen(fnIndexAll,"w");
	IndexAllNoHeader = fopen(fnIndexAllNoHeader,"w");
	ExtraInfo = fopen(fnExtraInfo,"w");
	fprintf(IndexAll,"%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta\n");
	fprintf(ExtraInfo,"%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) ZOrig(NoWedgeCorr) YOrig(DetCor) ZOrig(DetCor) OmegaOrig(DetCor) IntegratedIntensity(counts)\n");
	for (i=0;i<nIndices;i++){
		if (goodRows[i] == 1){
			fprintf(IndexAll,"%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",YCorrWedge[i],ZCorrWedge[i],OmegaCorrWedge[i],
				SpotsInfo[i][5],SpotsInfo[i][0],SpotsInfo[i][4],EtaCorrWedge[i],TthetaCorrWedge[i]);
			fprintf(IndexAllNoHeader,"%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",YCorrWedge[i],ZCorrWedge[i],OmegaCorrWedge[i],
				SpotsInfo[i][5],SpotsInfo[i][0],SpotsInfo[i][4],EtaCorrWedge[i],TthetaCorrWedge[i]);
			fprintf(ExtraInfo,"%12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n",YCorrWedge[i],
				ZCorrWedge[i],OmegaCorrWedge[i],SpotsInfo[i][5],SpotsInfo[i][0],SpotsInfo[i][4],EtaCorrWedge[i],TthetaCorrWedge[i],SpotsInfo[i][1],
				YCorrected[i],ZCorrected[i],SpotsInfo[i][2],SpotsInfo[i][3],SpotsInfo[i][1],SpotsInfo[i][6]);
		} else{
			fprintf(IndexAll,"0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000\n",SpotsInfo[i][0]);
			fprintf(IndexAllNoHeader,"0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000\n",SpotsInfo[i][0]);
			fprintf(ExtraInfo,"0.000 0.000 0.000 0.0000 %12.5f 0.0000 0.0000 0.0000 0.000 0.000 0.000 0.0000 0.000 0.000 0.000\n",SpotsInfo[i][0]);
		}
	}

	fclose(IndexAll);
	fclose(IndexAllNoHeader);
	fclose(ExtraInfo);
	PF = fopen(parfn,"w");
	//~ fprintf(PF,"LatticeConstant %f;\n",LatticeConstant[0]);
	fprintf(PF,"LatticeParameter %f %f %f %f %f %f;\n",LatticeConstant[0],LatticeConstant[1],LatticeConstant[2],LatticeConstant[3],LatticeConstant[4],LatticeConstant[5]);
	//~ fprintf(PF,"CellStruct %d;\n",2);
	fprintf(PF,"MaxRingRad %f;\n",MaxRingRad);
	fprintf(PF,"SpaceGroup %d\n",SGnum);
	fprintf(PF,"Wavelength %f;\n",Wavelength);
	fprintf(PF,"Distance %f;\n",LsdFit);
	fprintf(PF,"Rsample %f;\n",Rsample);
	fprintf(PF,"Hbeam %f;\n",Hbeam);
	fprintf(PF,"px %f;\n",px);
	fprintf(PF,"BeamSize %f;\n",BeamSize);
	fprintf(PF,"StepsizePos %f;\n",StepSizePos);
	fprintf(PF,"StepsizeOrient %f;\n",StepSizeOrient);
	fprintf(PF,"MarginRadius %f;\n",MarginRadius);
	fprintf(PF,"OmeBinSize %f;\n",OmeBinSize);
	fprintf(PF,"EtaBinSize %f;\n",EtaBinSize);
	fprintf(PF,"ExcludePoleAngle %f;\n",MinEta);
	for (i=0;i<nrUniqueRingNumbers;i++){
		fprintf(PF,"RingNumbers %d;\n",UniqueRingNumbers[i]);
	}
	for (i=0;i<nrUniqueRingNumbers;i++){
		fprintf(PF,"RingRadii %f;\n",RingRadsIdeal[i]);
	}
	fprintf(PF,"UseFriedelPairs %d;\n",UseFriedelPairs);
	fprintf(PF,"Wedge %f;\n",wedge);
	for (i=0;i<nOmeRanges;i++){
		fprintf(PF,"OmegaRange %f %f;\n",OmegaRanges[i][0],OmegaRanges[i][1]);
	}
	for (i=0;i<nOmeRanges;i++){
		fprintf(PF,"BoxSize %f %f %f %f;\n",BoxSizes[i][0],BoxSizes[i][1],BoxSizes[i][2],BoxSizes[i][3]);
	}
	fprintf(PF,"MarginEta %f;\n",MarginEta);
	fprintf(PF,"MarginOme %f;\n",MarginOme);
	fprintf(PF,"MargABC %f;\n",MargABC);
	fprintf(PF,"MargABG %f;\n",MargABG);
	fprintf(PF,"MarginRadial %f;\n",MarginRadial);
	fprintf(PF,"MinMatchesToAcceptFrac %f;\n",MinMatchesToAcceptFrac);
	fprintf(PF,"SpotsFileName %s\n",spotsfn);
	fprintf(PF,"RefinementFileName %s\n","InputAllExtraInfoFittingAll.csv");
	fprintf(PF,"OutputFolder %s\n",outfolder);
	fprintf(PF,"ResultFolder %s\n",resultfolder);
	fprintf(PF,"IDsFileName %s\n", idfn);
	fprintf(PF,"LsdFit %f\n",LsdFit);
	fprintf(PF,"YBCFit %f\n",ybcFit);
	fprintf(PF,"ZBCFit %f\n",zbcFit);
	fprintf(PF,"tyFit %f\n",ty);
	fprintf(PF,"tzFit %f\n",tz);
	fprintf(PF,"p0 %f\n",p0);
	fprintf(PF,"p1 %f\n",p1);
	fprintf(PF,"p2 %f\n",p2);
	fclose(PF);
	FreeMemMatrix(SpotsInfo,MaxNSpots);
	free(Ys);
	free(Zs);
	free(IdealTtheta);
	free(YCorrected);
	free(ZCorrected);
	free(YCorrWedge);
	free(ZCorrWedge);
	free(OmegaCorrWedge);
	free(EtaCorrWedge);
	free(TthetaCorrWedge);
	free(SpotIDsToIndex);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
