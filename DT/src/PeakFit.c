//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <nlopt.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdint.h>
#include <tiffio.h>
#include <libgen.h>

#define nFitVals 12

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
    SigmaL = x[2];
    Imax = x[3];
    BG = x[4];
    double TotalDifferenceIntensity=0,CalcIntensity;
    int i,j,k;
    double L, G;
    for (i=0;i<NrPtsForFit;i++){
        L = (1/(((Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaL*SigmaL))+(1)));
        G = (exp((-0.5)*(Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaG*SigmaG)));
        CalcIntensity = BG + Imax*((Mu*L)+((1-Mu)*G));
        TotalDifferenceIntensity += (CalcIntensity - PeakShape[i])*(CalcIntensity - PeakShape[i]);
    }
#ifdef PRINTOPT
    printf("Peak profiler intensity difference: %f\n",TotalDifferenceIntensity);
#endif
    return TotalDifferenceIntensity;
}

static
double CalcIntegratedIntensity(
    const double *x,
    void* f_data_trial)
{
    struct my_profile_func_data *f_data = (struct my_profile_func_data *) f_data_trial;
    int NrPtsForFit = f_data->NrPtsForFit;
    double *Rs;
    Rs = &(f_data->Rs[0]);
    double Rcen, Mu, SigmaG, SigmaL, Imax, BG;
    Rcen = x[0];
    Mu = x[1];
    SigmaG = x[2];
    SigmaL = x[2];
    Imax = x[3];
    BG = x[4];
    double TotalIntensity=0;
    int i,j,k;
    double L, G;
    for (i=0;i<NrPtsForFit;i++){
        L = (1/(((Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaL*SigmaL))+(1)));
        G = (exp((-0.5)*(Rs[i]-Rcen)*(Rs[i]-Rcen)/(SigmaG*SigmaG)));
        TotalIntensity += Imax*((Mu*L)+((1-Mu)*G));
    }
#ifdef PRINTOPT2
    printf("Peak fit intensity value: %lf\n",TotalIntensity);
#endif
    return TotalIntensity;
}

void FitPeakShape(int NrPtsForFit, double Rs[NrPtsForFit], double PeakShape[NrPtsForFit],
                double *Rfit, double Rstep)
{
    unsigned n = 5;
    double x[n],xl[n],xu[n];
    struct my_profile_func_data f_data;
    f_data.NrPtsForFit = NrPtsForFit;
    f_data.Rs = &Rs[0];
    f_data.PeakShape = &PeakShape[0];
    double BG0 = (PeakShape[0]+PeakShape[NrPtsForFit-1])/2;
    if (BG0 < 0) BG0=0;
    double MaxI=-100000, TotInt = 0;
    double Rmean = (Rs[0] + Rs[NrPtsForFit-1])/2;
    double RTemp;
    int i;
    for (i=0;i<NrPtsForFit;i++){
        TotInt += PeakShape[i];
        if (PeakShape[i] > MaxI){
            MaxI=PeakShape[i];
            RTemp = Rs[i];
        }
    }
    if (fabs(RTemp-Rmean)<2.0) Rmean = RTemp;
    MaxI -= BG0;
    x[0] = Rmean; xl[0] = Rs[0];    xu[0] = Rs[NrPtsForFit-1];
    x[1] = 0.5;   xl[1] = 0;        xu[1] = 1;
    x[2] = 1;     xl[2] = 0.05;  xu[2] = 100;
    //~ x[3] = 1;     xl[3] = 0.05;  xu[3] = 100;
    x[3] = MaxI;  xl[3] = MaxI/100; xu[3] = MaxI*3;
    x[4] = BG0;   xl[4] = 0;        xu[4] = BG0*3;
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
    Rfit[0] = x[0];
    Rfit[1] = x[1];
    Rfit[2] = x[2];
    Rfit[3] = x[2];
    Rfit[4] = x[3];
    Rfit[5] = MaxI; // Input Max Intensity
    Rfit[6] = x[4];
    Rfit[7] = BG0;
    Rfit[8] = MeanDiff;
    Rfit[9] = CalcIntegratedIntensity(x,trp); // Calculate integrated intensity
    Rfit[10] = TotInt; // Total intensity
    Rfit[11] = TotInt - (BG0*NrPtsForFit); // Total intensity after removing background
    //~ for (i=0;i<NrValsFitOutput;i++) printf("%lf ",Rfit[i]); printf("\n");
}

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}


int main(int argc, char **argv)
{
    if (argc != 3){
        printf("Usage: ./peakfit paramFN numProcs \n");
        return 1;
    }
    
    int numProcs = atoi(argv[2]);
    int nFits;
    double *ResultArrAll;
    double Rstep;
	double radiiToFit[200][6], etasToFit[200][4];
    
    char  peakfittingFN[4096], fitresultFN[4096], *paramFN = argv[1];
	FILE *paramFile;
	char aline[4096], dummy[4096], *str;
	paramFile = fopen(paramFN,"r");
	double Rstarts[200];
	int nRadFits = 0, ReconSize,nEtas=0,nElsPerRad;
	while (fgets(aline,4096,paramFile) != NULL){
		str = "RawDataPeakFN ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, peakfittingFN);
		}
		str = "PeakFitResultFN ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, fitresultFN);
		}
		str = "EtaToFit ";
		if (StartsWith(aline,str) == 1){
			nEtas++;
		}
		str = "RadiusToFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Rstarts[nRadFits]);
			nRadFits++;
		}
		str = "RBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Rstep);
		}
		str = "ReconSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &ReconSize);
		}
		str = "nElsPerRad ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &nElsPerRad);
		}
    }
	nFits = ReconSize*ReconSize*nEtas*nRadFits;
    
    ResultArrAll = calloc(nFitVals*nFits,sizeof(*ResultArrAll));
    double *rawData;
    rawData = calloc(nFits*nElsPerRad,sizeof(double));
    printf("%d %d %d %d %d\n",ReconSize,nEtas,nRadFits,nFits,nElsPerRad);
    FILE *peakFile,*fitresultFile;
    peakFile=fopen(peakfittingFN,"rb");
    fread(rawData,nFits*nElsPerRad*sizeof(double),1,peakFile);
    fitresultFile=fopen(fitresultFN,"w");
    int fitNr;
	double *RsAll;
	RsAll = calloc(nElsPerRad*nFits,sizeof(*RsAll));
	#pragma omp parallel for num_threads(numProcs) private(fitNr) schedule(dynamic)
	for (fitNr=0;fitNr<nFits;fitNr++){
		double *Peakshape, *ResultArr;
		Peakshape = &rawData[fitNr*nElsPerRad];
	    int i, iRad;
	    double *Rs;
	    Rs = &RsAll[fitNr*nElsPerRad];
	    memset(Rs,0,nElsPerRad*sizeof(double));
	    iRad = fitNr % nRadFits;
		for (i=0;i<nElsPerRad;i++){
			Rs[i]=Rstarts[iRad]+i*Rstep;
		}
		ResultArr = &ResultArrAll[fitNr*nFitVals];
		FitPeakShape(nElsPerRad,Rs,Peakshape,ResultArr,Rstep);
	}
	fwrite(ResultArrAll,nFitVals*nFits*sizeof(double),1,fitresultFile);
	fclose(fitresultFile);
}
