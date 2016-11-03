//
//  Calibrant.c
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

//#define PRINTOPT
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
typedef uint16_t pixelvalue;
long long int NrCalls;
long long int NrCallsProfiler;
#define MultFactor 1

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
void Car2Pol(int n_hkls, int nEtaBins, int y, int z, double ybc, double zbc, double px, double *R, double *Eta, double Rmins[n_hkls],
						   double Rmaxs[n_hkls], double EtaBinsLow[nEtaBins], double EtaBinsHigh[nEtaBins], int nIndices, int *NrEachIndexbin, int **Indices){
	int i, j, k, l, counter=0;
	for (i=0;i<nIndices;i++) NrEachIndexbin[i]=0;
	for (i=0;i<y;i++){
		for (j=0;j<z;j++){
			R[counter] = px*sqrt(((j-ybc)*(j-ybc))+((i-zbc)*(i-zbc)));
			Eta[counter] = CalcEtaAngle(-(j-ybc),(i-zbc));
			for (k=0;k<n_hkls;k++){
				if (R[counter] >= (Rmins[k]-px) && R[counter] <= (Rmaxs[k] + px)){
					for (l=0;l<nEtaBins;l++){
						if (Eta[counter] >= (EtaBinsLow[l] - px/R[counter]) && Eta[counter] <= (EtaBinsHigh[l] + px/R[counter])){
							Indices[(nEtaBins*k)+l][NrEachIndexbin[(nEtaBins*k)+l]] = (i*2048) + j;
							NrEachIndexbin[(nEtaBins*k)+l] += 1;
							break;
						}
					}
					break;
				}
			}
			counter++;
		}
	}
}

static inline 
void CalcWeightedMean(int nIndices, int *NrEachIndexBin, int **Indices, double *Average, double *R, double *Eta, double *RMean, double *EtaMean){
	int i,j,k;
	double TotIntensities[nIndices];
	for (i=0;i<nIndices;i++){TotIntensities[i]=0;EtaMean[i]=0;RMean[i]=0;}
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

struct my_profile_func_data{
	int NrPtsForFit;
	double *Rs;
	double *PeakShape;
};

static
double problem_function_profile(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_profile_func_data *f_data = (struct my_profile_func_data *) f_data_trial;
	int NrPtsForFit = f_data->NrPtsForFit;
	double *Rs, *PeakShape;
	Rs = &(f_data->Rs[0]);
	PeakShape = &(f_data->PeakShape[0]);
	double Rcen, Mu, SigmaG, SigmaL, Imax, BG;
	Rcen = x[0];
	Mu = x[1];
	SigmaG = x[2];
	SigmaL = x[3];
	Imax = x[4];
	BG = x[5];
	double TotalDifferenceIntensity=0,CalcIntensity;
	int i,j,k;
	double L, G;
	for (i=0;i<NrPtsForFit;i++){
		//L = (2/M_PI)*(SigmaL/((4*(Rs[i]-Rcen)*(Rs[i]-Rcen))+(SigmaL*SigmaL)));
		L = (1/(((Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaL*SigmaL))+(1)));
		//G = (sqrt(4*log(2))/(sqrt(M_PI)*SigmaG))*(exp(((-4*log(2))/(SigmaG*SigmaG))*(Rs[i]-Rcen)*(Rs[i]-Rcen)));
		G = (exp((-0.5)*(Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaG*SigmaG)));
		CalcIntensity = BG + Imax*((Mu*L)+((1-Mu)*G));
		TotalDifferenceIntensity += (CalcIntensity - PeakShape[i])*(CalcIntensity - PeakShape[i]);
	}
	NrCallsProfiler++;
#ifdef PRINTOPT
	printf("Peak profiler intensity difference: %f\n",TotalDifferenceIntensity);
#endif	
	return TotalDifferenceIntensity;
}

void FitPeakShape(int NrPtsForFit, double Rs[NrPtsForFit], double PeakShape[NrPtsForFit], 
				double *Rfit, double Rstep, double Rmean, double Etas[NrPtsForFit])
{
	unsigned n = 6;
	double x[n],xl[n],xu[n];
	struct my_profile_func_data f_data;
	f_data.NrPtsForFit = NrPtsForFit;
	f_data.Rs = &Rs[0];
	f_data.PeakShape = &PeakShape[0];
	double BG0 = (PeakShape[0]+PeakShape[NrPtsForFit-1])/2;
	if (BG0 < 0) BG0=0;
	double MaxI=-100000;
	int i;
	for (i=0;i<NrPtsForFit;i++){
		if (PeakShape[i] > MaxI){
			MaxI=PeakShape[i];
		}
	}
	x[0] = Rmean; xl[0] = Rs[0];    xu[0] = Rs[NrPtsForFit-1];
	x[1] = 0.5;   xl[1] = 0;        xu[1] = 1;
	x[2] = Rstep;     xl[2] = Rstep/2;  xu[2] = Rstep*NrPtsForFit/2;
	x[3] = Rstep;     xl[3] = Rstep/2;  xu[3] = Rstep*NrPtsForFit/2;
	x[4] = MaxI;  xl[4] = MaxI/100; xu[4] = MaxI*1.5;
	x[5] = BG0;   xl[5] = 0;        xu[5] = BG0*1.5;
	struct my_profile_func_data *f_datat;
	f_datat = &f_data;
	void* trp = (struct my_profile_func_data *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function_profile, trp);
	double minf,MeanDiff;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
	MeanDiff = sqrt(minf)/(NrPtsForFit);
	*Rfit = x[0];
}

void CalcFittedMean(int nIndices, int *NrEachIndexBin, int **Indices, double *Average,
	double *R, double *Eta, double *RMean, double *EtaMean, int NrPtsForFit, double *IdealRmins,
	double *IdealRmaxs,int nBinsPerRing,double ybc, double zbc, double px, int NrPixels){
	int i,j,k,BinNr;
	double PeakShape[NrPtsForFit], Rmin, Rmax, Rstep, Rthis, Rs[NrPtsForFit];
	double Rfit;
	int **Idxs;
	Idxs = allocMatrixInt(1,NrPtsForFit);
	double Etas[NrPtsForFit];
	double EtaMi,EtaMa,Rmi,Rma;
	double RetVal;
	for (i=0;i<NrPtsForFit;i++)Idxs[0][i]=i;
	double AllZero;
	for (i=0;i<nIndices;i++){
		//printf("%i of %i done.\n",i,nIndices);
		Rmin = IdealRmins[i];
		Rmax = IdealRmaxs[i];
		AllZero=1;
		Rstep = (Rmax-Rmin)/NrPtsForFit;
		BinNr = i % nBinsPerRing;
		EtaMi = -180 + BinNr*(360/nBinsPerRing);
		EtaMa = -180 + (BinNr+1)*(360/nBinsPerRing);
		//printf("%f %f\n",EtaMi,EtaMa);
		EtaMean[i] = (EtaMi+EtaMa)/2;
		/*for (j=0;j<NrPtsForFit;j++){PeakShape[j]=0;Rs[j]=(Rmin+(j*Rstep)+Rstep/2);}
		for (j=0;j<NrEachIndexBin[i];j++){
			Rthis = R[Indices[i][j]];
			BinNr = (int)(floor((-Rmin+Rthis)/Rstep));
			PeakShape[BinNr] += Average[Indices[i][j]];
			EtaMean[i] += Eta[Indices[i][j]]/NrEachIndexBin[i];
		}*/
		for (j=0;j<NrPtsForFit;j++){
			PeakShape[j]=0;
			Rs[j]=(Rmin+(j*Rstep)+Rstep/2);
			Rmi = Rs[j] - Rstep/2;
			Rma = Rs[j] + Rstep/2;
			CalcPeakProfile(Indices,NrEachIndexBin,i,Average,Rmi,Rma,EtaMi,EtaMa,ybc,zbc,px,NrPixels, &RetVal);
			PeakShape[j] = RetVal;
			if (RetVal != 0){
				AllZero = 0;
			}
		}
		for (j=0;j<NrPtsForFit;j++){
			Etas[j]=EtaMean[i];
		}
		double *Rm, *Etam;
		int *NrPts;
		NrPts = malloc(sizeof(*NrPts));
		Rm = malloc(sizeof(*Rm));
		Etam = malloc(sizeof(*Etam));
		NrPts[0] = NrPtsForFit;
		if (AllZero != 1){
			CalcWeightedMean(1, NrPts, Idxs, PeakShape, Rs, Etas, Rm, Etam);
			double Rmean=Rm[0];
			FitPeakShape(NrPtsForFit,Rs,PeakShape,&Rfit,Rstep,Rmean,Etas);
		}else{
			printf("All intensities were 0. i=%d of %d, %f %f\n",i,nIndices,IdealRmins[i],IdealRmaxs[i]);
			Rfit = (IdealRmins[i] + IdealRmaxs[i])/2;
		}
		RMean[i] = Rfit;
		free(NrPts);
		free(Rm);
		free(Etam);
	}
	FreeMemMatrixInt(Idxs,1);
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

struct my_func_data{
	int nIndices;
	double *YMean;
	double *ZMean;
	double *IdealTtheta;
	double MaxRad;
	double px;
	double tx;
};

static
double problem_function(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct my_func_data *f_data = (struct my_func_data *) f_data_trial;
	int MaxRad = f_data->MaxRad;
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
	p0  = x[5];
	p1  = x[6];
	p2  = x[7];
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
		Diff = fabs(1 - (Rcorr/RIdeal));
		TotalDiff+=Diff;
	}
	TotalDiff *= MultFactor;
	NrCalls++;
#ifdef PRINTOPT
	printf("Mean strain: %0.40f\n",TotalDiff/(MultFactor*nIndices));
#endif
	return TotalDiff;
}

void FitTiltBCLsd(int nIndices, double *YMean, double *ZMean, double *IdealTtheta, double Lsd, double MaxRad, 
				  double ybc, double zbc, double tx, double tyin, double tzin, double p0in, double p1in, double p2in, double *ty, double *tz, double *LsdFit, 
				  double *ybcFit, double *zbcFit, double *p0, double *p1, double *p2, double *MeanDiff, double tolTilts, double tolLsd, double tolBC, double tolP, double px)
{
	unsigned n=8;
	struct my_func_data f_data;
	f_data.nIndices = nIndices;
	f_data.YMean = &YMean[0];
	f_data.ZMean = &ZMean[0];
	f_data.IdealTtheta = &IdealTtheta[0];
	f_data.MaxRad = MaxRad;
	f_data.px = px;
	f_data.tx = tx;
	double x[n],  xl[n], xu[n];
	x[0] = Lsd;   xl[0] = Lsd - tolLsd;   xu[0] = Lsd + tolLsd;
	x[1] = ybc;   xl[1] = ybc - tolBC;    xu[1] = ybc + tolBC;
	x[2] = zbc;   xl[2] = zbc - tolBC;    xu[2] = zbc + tolBC;
	x[3] = tyin;  xl[3] = tyin- tolTilts; xu[3] = tyin+ tolTilts;
	x[4] = tzin;  xl[4] = tzin- tolTilts; xu[4] = tzin+ tolTilts;
	x[5] = p0in;  xl[5] = p0in- tolP;     xu[5] = p0in+ tolP;
	x[6] = p1in;  xl[6] = p1in- tolP;     xu[6] = p1in+ tolP;
	x[7] = p2in;  xl[7] = p2in- tolP;     xu[7] = p2in+ tolP;
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
	*p0     = x[5];
	*p1     = x[6];
	*p2     = x[7];
}

static inline
void CorrectTiltSpatialDistortion(int nIndices, double MaxRad, double *YMean, double *ZMean, double *IdealTtheta,
		double px, double Lsd, double ybc, double zbc, double tx, double ty, double tz, double p0, double p1,
		double p2, double *Etas, double *Diffs, double *RadOuts, double *StdDiff)
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
	double Rad,Eta,RNorm,DistortFunc,Rcorr,RIdeal,EtaT,Diff,MeanDiff;
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
		Diff = fabs(1 - (Rcorr/RIdeal));
		Etas[i] = Eta;
		Diffs[i] = Diff;
		MeanDiff += Diff;
		RadOuts[i] = Rcorr;
	}
	MeanDiff /= nIndices;
	double StdDiff2;
	for (i=0;i<nIndices;i++){
		StdDiff2 += (Diffs[i] - MeanDiff)*(Diffs[i] - MeanDiff);
	}
	*StdDiff = sqrt(StdDiff2/nIndices);
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
double CalcTheta(double a, double b, double c, double alpha,
	double beta, double gamma, double wavelength, int hkl[3])
{
	double cosAlpha = cos(alpha*(M_PI/180.0));
	double cosBeta  = cos(beta *(M_PI/180.0));
	double cosGamma = cos(gamma*(M_PI/180.0));
	double h = (double) hkl[0];
	double k = (double) hkl[1];
	double l = (double) hkl[2];
	double numer =  (((h*h)/(a*a))*(1-(cosAlpha*cosAlpha)))+
	                (((k*k)/(b*b))*(1-(cosBeta *cosBeta )))+
	                (((l*l)/(c*c))*(1-(cosGamma*cosGamma)))+
	                (2*h*k*(((cosAlpha*cosBeta )-cosGamma)/(a*b)))+
	                (2*h*l*(((cosAlpha*cosGamma)-cosBeta )/(a*c)))+
	                (2*k*l*(((cosGamma*cosBeta )-cosAlpha)/(b*c)));
	double denom =  1.0 - ((cosAlpha*cosAlpha) + (cosBeta*cosBeta) +
					(cosGamma*cosGamma)) + (2*cosAlpha*cosBeta*cosGamma);
	double sinThetaPerLambda = ((sqrt(numer))/(2.0*sqrt(denom)));
	return asin(wavelength*sinThetaPerLambda)*180/M_PI;
}

int PlanesLaB6[55][3] = {{1,0,0},
						 {1,1,0},
						 {1,1,1},
						 {2,0,0},
						 {2,1,0},
						 {2,1,1},
						 {2,2,0},
						 {3,0,0},
						 {3,1,0},
						 {3,1,1},
						 {2,2,2},
						 {3,2,0},
						 {3,2,1},
						 {4,0,0},
						 {4,1,0},
						 {4,1,1},
						 {3,3,1},
						 {4,2,0},
						 {4,2,1},
						 {3,3,2},
						 {4,2,2},
						 {4,3,0},
						 {4,3,1},
						 {3,3,3},
						 {4,3,2},
						 {5,2,1},
						 {4,4,0},
						 {5,2,2},
						 {4,3,3},
						 {5,3,1},
						 {4,4,2},
						 {6,1,0},
						 {5,3,2},
						 {6,2,0},
						 {5,4,0},
						 {5,4,1},
						 {5,3,3},
						 {6,2,2},
						 {5,4,2},
						 {6,3,1},
						 {4,4,4},
						 {6,3,2},
						 {5,4,3},
						 {5,5,1},
						 {6,4,0},
						 {6,4,1},
						 {5,5,2},
						 {6,4,2},
						 {7,2,2},
						 {7,3,0},
						 {5,5,3},
						 {6,5,0},
						 {6,5,1},
						 {8,0,0},
						 {8,1,0}};

int PlanesCeO2[35][3] = {{1,1,1},
						 {2,0,0},
						 {2,2,0},
						 {3,1,1},
						 {2,2,2},
						 {4,0,0},
						 {3,3,1},
						 {4,2,0},
						 {4,2,2},
						 {5,1,1},
						 {4,4,0},
						 {5,3,1},
						 {4,4,2},
						 {6,2,0},
						 {5,3,3},
						 {6,2,2},
						 {4,4,4},
						 {7,1,1},
						 {4,6,0},
						 {6,4,2},
						 {5,5,3},
						 {8,0,0},
						 {7,3,3},
						 {6,4,4},
						 {6,6,0},
						 {7,5,1},
						 {6,6,2},
						 {8,4,0},
						 {7,5,3},
						 {8,4,2},
						 {6,6,4},
						 {9,3,1},
						 {8,4,4},
						 {7,7,1},
						 {8,6,0}};

static inline 
int CalcThetas(int CalType, double a, double b, double c, 
	double alpha,double beta, double gamma, double wavelength,
	double MaxTtheta, double Thetas[30], int nRingsExclude, int RingsToExclude[20])
{
	int i,j;
	double Theta;
	int hkl[3], ThisRingToExclude, counter=0;
	if (CalType == 225){
		for (i=0;i<100;i++){
			ThisRingToExclude = 0;
			for (j=0;j<nRingsExclude;j++){
				if (i == RingsToExclude[j]-1){
					ThisRingToExclude = 1;
					break;
				}
			}
			if (ThisRingToExclude == 1){
				continue;
			}
			hkl[0] = PlanesCeO2[i][0];
			hkl[1] = PlanesCeO2[i][1];
			hkl[2] = PlanesCeO2[i][2];
			Theta = CalcTheta(a,b,c,alpha,beta,gamma,wavelength,hkl);
			//printf("%d %d %d %f %f\n",hkl[0],hkl[1],hkl[2],2*Theta,MaxTtheta);
			if (2*Theta > MaxTtheta){
				return counter;
			}
			Thetas[counter] = Theta;
			counter ++;
		}
	} else if (CalType == 221){
		for (i=0;i<100;i++){
			ThisRingToExclude = 0;
			for (j=0;j<nRingsExclude;j++){
				if (i == RingsToExclude[j]-1){
					ThisRingToExclude = 1;
					break;
				}
			}
			if (ThisRingToExclude == 1){
				continue;
			}
			hkl[0] = PlanesLaB6[i][0];
			hkl[1] = PlanesLaB6[i][1];
			hkl[2] = PlanesLaB6[i][2];
			Theta = CalcTheta(a,b,c,alpha,beta,gamma,wavelength,hkl);
			//printf("%d %d %d %f %f\n",hkl[0],hkl[1],hkl[2],2*Theta,MaxTtheta);
			if (2*Theta > MaxTtheta){
				return counter;
			}
			Thetas[counter] = Theta;
			counter++;
		}
	} else {
		printf("Wrong SpaceGroup. Exiting.\n");
		return 0;
	}
}

int main(int argc, char *argv[])
{
    clock_t start, end, start0, end0;
    start0 = clock();
    double diftotal;
    // Read params file.
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    char fn[1024],folder[1024],Ext[1024],Dark[1024];
    int StartNr, EndNr, LowNr, NrPixels;
    int SpaceGroup,FitWeightMean=0;
    double LatticeConstant[6], Wavelength, MaxRingRad, Lsd, MaxTtheta, TthetaTol, ybc, zbc, EtaBinSize, px,Width;
    double tx,tolTilts,tolLsd,tolBC,tolP,tyin=0,tzin=0,p0in=0,p1in=0,p2in=0;
    //int SkipHeader = 1;
    int Padding = 6;
    int NrTransOpt=0;
    int TransOpt[10], nRingsExclude=0, RingsExclude[20];
    while (fgets(aline,1000,fileParam)!=NULL){
		str = "FileStem ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, fn);
            continue;
        }
		str = "Folder ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, folder);
            continue;
        }
		str = "Ext ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Ext);
            continue;
        }
		str = "Dark ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %s", dummy, Dark);
            continue;
        }
        str = "Padding ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &Padding);
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
        str = "NrPixels ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "ImTransOpt ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SpaceGroup);
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
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Wavelength);
            continue;
        }
        str = "RhoD ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &MaxRingRad);
            continue;
        }
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Lsd);
            continue;
        }
        str = "px ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &px);
            continue;
        }
        str = "ty ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tyin);
            continue;
        }
        str = "tz ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tzin);
            continue;
        }
        str = "p0 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p0in);
            continue;
        }
        str = "p1 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p1in);
            continue;
        }
        str = "p2 ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &p2in);
            continue;
        }
        str = "Width ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Width);
            continue;
        }
        str = "EtaBinSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &EtaBinSize);
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
        str = "tolP ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tolP);
            continue;
        }
        str = "tx ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &tx);
            continue;
        }
        str = "FitOrWeightedMean ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &FitWeightMean);
            continue;
        }
        str = "RingsToExclude ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingsExclude[nRingsExclude]);
            nRingsExclude++;
            continue;
        }
	}
	int i,j,k;
	printf("NrTransOpt: %d\n",NrTransOpt);
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 3){printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");return 0;}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
        else printf("Transpose.\n");
    }
	MaxTtheta = rad2deg*atan(MaxRingRad/Lsd);
    double Thetas[100];
    for (i=0;i<100;i++) Thetas[i] = 0;
    int n_hkls = 0;
    n_hkls = CalcThetas(SpaceGroup,LatticeConstant[0],LatticeConstant[1],
			LatticeConstant[2],LatticeConstant[3],LatticeConstant[4],LatticeConstant[5],
			Wavelength,MaxTtheta,Thetas,nRingsExclude,RingsExclude);
	printf("Number of planes being considered: %d.\n",n_hkls);
	printf("The following rings will be excluded:");
	for (i=0;i<nRingsExclude;i++){
		printf(" %d",RingsExclude[i]);
	}
	//Width = Width/px;
	TthetaTol = Ttheta4mR((MaxRingRad+Width),Lsd) - Ttheta4mR((MaxRingRad-Width),Lsd);
	printf("\n2Theta Tolerance: %f \n",TthetaTol);
	pixelvalue *DarkFile;
	double *AverageDark;
	int SizeFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	int sz;
	char FileName[1024];
	long Skip;
	FILE *fp, *fd;
	int nFrames, TotFrames=0;
	double *Average;
	pixelvalue *Image;
	DarkFile = malloc(NrPixels*NrPixels*sizeof(*DarkFile));
	AverageDark = malloc(NrPixels*NrPixels*sizeof(*AverageDark));
	Average = malloc(NrPixels*NrPixels*sizeof(*Average));
	Image = malloc(NrPixels*NrPixels*sizeof(*Image));
	fd = fopen(Dark,"rb");
	if (fd == NULL){
		printf("Dark file %s could not be read. Making an empty array for dark.\n",Dark);
		for (j=0;j<(NrPixels*NrPixels);j++)AverageDark[j] = 0;
	}else{
		fseek(fd,0L,SEEK_END);
		sz = ftell(fd);
		rewind(fd);
		nFrames = sz/(8*1024*1024);
		Skip = sz - (nFrames*8*1024*1024);
		printf("Reading dark file:      %s, nFrames: %d, skipping first %ld bytes.\n",Dark,nFrames,Skip);
		fseek(fd,Skip,SEEK_SET);
		for (j=0;j<(NrPixels*NrPixels);j++)AverageDark[j]=0;
		for (i=0;i<nFrames;i++){
			fread(DarkFile,SizeFile,1,fd);
			DoImageTransformations(NrTransOpt,TransOpt,DarkFile,NrPixels);
			for(j=0;j<(NrPixels*NrPixels);j++)AverageDark[j]+=DarkFile[j];
		}
		printf("Dark file read.\n");
		for (j=0;j<(NrPixels*NrPixels);j++)AverageDark[j]=AverageDark[j]/nFrames;
		fclose(fd);
	}
	int a;
	for (a=StartNr;a<=EndNr;a++){
		start = clock();
		for (j=0;j<(NrPixels*NrPixels);j++)Average[j]=0;
		if (Padding == 2){sprintf(FileName,"%s/%s_%02d%s",folder,fn,a,Ext);}
		else if (Padding == 3){sprintf(FileName,"%s/%s_%03d%s",folder,fn,a,Ext);}
		else if (Padding == 4){sprintf(FileName,"%s/%s_%04d%s",folder,fn,a,Ext);}
		else if (Padding == 5){sprintf(FileName,"%s/%s_%05d%s",folder,fn,a,Ext);}
		else if (Padding == 6){sprintf(FileName,"%s/%s_%06d%s",folder,fn,a,Ext);}
		else if (Padding == 7){sprintf(FileName,"%s/%s_%07d%s",folder,fn,a,Ext);}
		else if (Padding == 8){sprintf(FileName,"%s/%s_%08d%s",folder,fn,a,Ext);}
		else if (Padding == 9){sprintf(FileName,"%s/%s_%09d%s",folder,fn,a,Ext);}
		fp = fopen(FileName,"rb");
		if (fp == NULL){
			printf("File %s could not be read. Continuing to next one.\n",FileName);
			continue;
		}
		fseek(fp,0L,SEEK_END);
		sz = ftell(fp);
		nFrames = sz/(8*1024*1024);
		Skip = sz - (nFrames*8*1024*1024);
		printf("Reading calibrant file: %s, nFrames: %d, skipping first %ld bytes.\n",FileName,nFrames,Skip);
		rewind(fp);
		fseek(fp,Skip,SEEK_SET);
		for (j=0;j<nFrames;j++){
			fread(Image,SizeFile,1,fp);
			DoImageTransformations(NrTransOpt,TransOpt,Image,NrPixels);
			for(k=0;k<(NrPixels*NrPixels);k++){
				Average[k]+=Image[k]-AverageDark[k];
			}
		}
		TotFrames+=nFrames;
		fclose(fp);
		double IdealTthetas[n_hkls], TthetaMins[n_hkls], TthetaMaxs[n_hkls];
		for (i=0;i<n_hkls;i++){IdealTthetas[i]=2*Thetas[i];TthetaMins[i]=IdealTthetas[i]-TthetaTol;TthetaMaxs[i]=IdealTthetas[i]+TthetaTol;}
		double IdealRs[n_hkls], Rmins[n_hkls], Rmaxs[n_hkls];
		for (i=0;i<n_hkls;i++){IdealRs[i]=R4mTtheta(IdealTthetas[i],Lsd);Rmins[i]=R4mTtheta(TthetaMins[i],Lsd);Rmaxs[i]=R4mTtheta(TthetaMaxs[i],Lsd);}
		int nEtaBins;
		nEtaBins = (int)ceil(360.0/EtaBinSize);
		printf("Number of eta bins: %d.\n",nEtaBins);
		double EtaBinsLow[nEtaBins], EtaBinsHigh[nEtaBins];
		for (i=0;i<nEtaBins;i++){
			EtaBinsLow[i] = EtaBinSize*i - 180;
			EtaBinsHigh[i] = EtaBinSize*(i+1) - 180;
		}
		double *R,*Eta;
		R = malloc(NrPixels*NrPixels*sizeof(*R));
		Eta = malloc(NrPixels*NrPixels*sizeof(*Eta));
		int **Indices, nIndices;
		nIndices = nEtaBins * n_hkls;
		int *NrEachIndexBin;
		NrEachIndexBin = malloc(nIndices*sizeof(*NrEachIndexBin));
		Indices = allocMatrixInt(nIndices,20000);
		Car2Pol(n_hkls,nEtaBins,NrPixels,NrPixels,ybc,zbc,px,R,Eta,Rmins,Rmaxs,EtaBinsLow,EtaBinsHigh,nIndices,NrEachIndexBin,Indices);
		double *RMean, *EtaMean, *IdealR, *IdealTtheta, *IdealRmins, *IdealRmaxs;
		IdealR = malloc(nIndices*sizeof(*IdealR));
		IdealRmins = malloc(nIndices*sizeof(*IdealRmins));
		IdealRmaxs = malloc(nIndices*sizeof(*IdealRmaxs));
		IdealTtheta = malloc(nIndices*sizeof(*IdealTtheta));
		RMean = malloc(nIndices*sizeof(*RMean));
		EtaMean = malloc(nIndices*sizeof(*EtaMean));
		int NrPtsForFit;
		NrPtsForFit = (int)((floor)((Rmaxs[0]-Rmins[0])/px))*4;
		for (i=0;i<nIndices;i++){
			IdealR[i] = IdealRs[(int)(floor(i/nEtaBins))];
			IdealRmins[i] = Rmins[(int)(floor(i/nEtaBins))];
			IdealRmaxs[i] = Rmaxs[(int)(floor(i/nEtaBins))];
			IdealTtheta[i]=rad2deg*atan(IdealR[i]/Lsd);
		}
		NrCallsProfiler = 0;
		if (FitWeightMean == 1) {
			CalcWeightedMean(nIndices,NrEachIndexBin,Indices,Average,R,Eta,RMean,EtaMean);
		} else {
			CalcFittedMean(nIndices,NrEachIndexBin,Indices,Average,R,Eta,RMean,EtaMean,NrPtsForFit,IdealRmins,IdealRmaxs,nEtaBins,ybc,zbc,px,NrPixels);
		}
		end = clock();
	    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	    if (FitWeightMean != 1){printf("Number of calls to profiler function: %lld\n",NrCallsProfiler);printf("Time elapsed in fitting peak profiles:\t%f s.\n",diftotal);}
	    else printf("Time elapsed in finding peak positions:\t%f s.\n",diftotal);
		double *YMean, *ZMean;
		YMean = malloc(nIndices*sizeof(*YMean));
		ZMean = malloc(nIndices*sizeof(*ZMean));
		YZ4mREta(nIndices,RMean,EtaMean,YMean,ZMean);
		double ty,tz,LsdFit,ybcFit,zbcFit,p0,p1,p2,MeanDiff,*Yc,*Zc,*EtaIns,*RadIns,*DiffIns,StdDiff;
		Yc=malloc(nIndices*sizeof(*Yc));
		Zc=malloc(nIndices*sizeof(*Zc));
		EtaIns = malloc(nIndices*sizeof(*EtaIns));
		RadIns = malloc(nIndices*sizeof(*RadIns));
		DiffIns = malloc(nIndices*sizeof(*DiffIns));
		for (i=0;i<nIndices;i++){Yc[i]=(ybc-(YMean[i]/px));Zc[i]=(zbc+(ZMean[i]/px));}
		CorrectTiltSpatialDistortion(nIndices,MaxRingRad,Yc,Zc,IdealTtheta,px,Lsd,ybc,zbc,tx,tyin,tzin,p0in,p1in,p2in,EtaIns,DiffIns,RadIns,&StdDiff);
		NrCalls = 0;
		FitTiltBCLsd(nIndices,Yc,Zc,IdealTtheta,Lsd,MaxRingRad,ybc,zbc,tx,tyin,tzin,p0in,p1in,p2in,&ty,&tz,&LsdFit,&ybcFit,&zbcFit,&p0,&p1,&p2,&MeanDiff,tolTilts,tolLsd,tolBC,tolP,px);
		printf("Number of function calls: %lld\n",NrCalls);
		printf("LsdFit:\t\t%0.12f\nYBCFit:\t\t%0.12f\nZBCFit:\t\t%0.12f\ntyFit:\t\t%0.12f\ntzFit:\t\t%0.12f\nP0Fit:\t\t%0.12f\nP1Fit:\t\t%0.12f\nP2Fit:\t\t%0.12f\nMeanStrain:\t%0.12lf\n",
				LsdFit,ybcFit,zbcFit,ty,tz,p0,p1,p2,MeanDiff);
		double *Etas, *Diffs, *RadOuts;
		Etas = malloc(nIndices*sizeof(*Etas));
		Diffs = malloc(nIndices*sizeof(*Diffs));
		RadOuts = malloc(nIndices*sizeof(*RadOuts));
		CorrectTiltSpatialDistortion(nIndices,MaxRingRad,Yc,Zc,IdealTtheta,px,LsdFit,ybcFit,zbcFit,tx,ty,tz,p0,p1,p2,Etas,Diffs,RadOuts,&StdDiff);
		printf("StdStrain:\t%0.12lf\n",StdDiff);
		FILE *Out;
		char OutFileName[1024];
		sprintf(OutFileName,"%s_%05d%s",fn,a,"corr.csv");
		Out = fopen(OutFileName,"w");
		for (i=0;i<nIndices;i++){fprintf(Out,"%f %10.8f %10.8f %f %10.8f %10.8f %f\n",Etas[i],Diffs[i],RadOuts[i],EtaIns[i],DiffIns[i],RadIns[i],IdealTtheta[i]);}
		fclose(Out);
		FreeMemMatrixInt(Indices,nIndices);
		free(R);
		free(Eta);
		free(NrEachIndexBin);
		free(IdealR);
		free(IdealRmins);
		free(IdealRmaxs);
		free(IdealTtheta);
		free(RMean);
		free(EtaMean);
		free(YMean);
		free(ZMean);
		free(Diffs);
		free(Etas);
		end = clock();
	    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	    printf("Time elapsed for this file:\t%f s.\n",diftotal);
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Total time elapsed:\t%f s.\n",diftotal);
	free(DarkFile);
	free(AverageDark);
	free(Average);
	free(Image);
    return 0;
}
