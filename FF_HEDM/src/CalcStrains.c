//
//
// CalcStrains.c
//
// Code to calculate strains according to Fable and according to P. Kenesei
//
// Hemant Sharma, 2014/10/25
// Update 2014/11/19: Added P. Kenesei definition
//

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <nlopt.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
#define NR_MAX_IDS_PER_GRAIN 5000

inline void
MatMult(double m[3][3], double n[3][3], double res[3][3]) 
{
   int r;
   for (r=0; r<3; r++) {
      res[r][0] = m[r][0]*n[0][0] +
                  m[r][1]*n[1][0] +
                  m[r][2]*n[2][0];
      res[r][1] = m[r][0]*n[0][1] +
                  m[r][1]*n[1][1] +
                  m[r][2]*n[2][1];
      res[r][2] = m[r][0]*n[0][2] +
                  m[r][1]*n[1][2] +
                  m[r][2]*n[2][2];
   }    
}

inline void
TransposeM(double m[3][3], double n[3][3])
{
	int i,j;
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			n[j][i] = m[i][j];
		}
	}
}

inline void
MatInv(double A[3][3], double AInv[3][3])
{
	double a = A[0][0];
	double b = A[0][1];
	double c = A[0][2];
	double d = A[1][0];
	double e = A[1][1];
	double f = A[1][2];
	double g = A[2][0];
	double h = A[2][1];
	double i = A[2][2];
	double DetA = (a*(e*i-f*h)) - (b*(i*d-f*g)) + (c*(d*h-e*g));
	AInv[0][0] =  (e*i-f*h)/DetA;
	AInv[0][1] = -(b*i-c*h)/DetA;
	AInv[0][2] =  (b*f-c*e)/DetA;
	AInv[1][0] = -(d*i-f*g)/DetA;
	AInv[1][1] =  (a*i-c*g)/DetA;
	AInv[1][2] = -(a*f-c*d)/DetA;
	AInv[2][0] =  (d*h-e*g)/DetA;
	AInv[2][1] = -(a*h-b*g)/DetA;
	AInv[2][2] =  (a*e-b*d)/DetA;
}

inline void
CalcStrainTensorFableBeaudoin(double LatCin[6],double LatticeParameterFit[6],
	double Orient[3][3], double StrainTensorSample[3][3])
{
	// unstrained parameters
	double a1 = LatCin[0];
	double b1 = LatCin[1];
	double c1 = LatCin[2];
	double alpha1 = deg2rad*LatCin[3];
	double beta1  = deg2rad*LatCin[4];
	double gamma1 = deg2rad*LatCin[5];
	// Strained parameters
	
	double a2 = LatticeParameterFit[0];
	double b2 = LatticeParameterFit[1];
	double c2 = LatticeParameterFit[2];
	double alpha2 = deg2rad*LatticeParameterFit[3];
	double beta2  = deg2rad*LatticeParameterFit[4];
	double gamma2 = deg2rad*LatticeParameterFit[5];
	
	// Calculate strain tensor in the grain coordinate system
	
	double CosGPr1 = (cos(alpha1)*cos(beta1)-cos(gamma1)/(sin(alpha1)*sin(beta1)));
	double SinGPr1 = sin(acos(CosGPr1));
	double A0[3][3] =  {{ a1*sin(beta1)*SinGPr1, 0, 			 0},
						{-a1*sin(beta1)*CosGPr1, b1*sin(alpha1), 0},
						{ a1*cos(beta1), 		 b1*cos(alpha1), c1}};
	
	double CosGPr2 = (cos(alpha2)*cos(beta2)-cos(gamma2)/(sin(alpha2)*sin(beta2)));
	double SinGPr2 = sin(acos(CosGPr2));
	double A[3][3] =   {{ a2*sin(beta2)*SinGPr2, 0, 			 0},
						{-a2*sin(beta2)*CosGPr2, b2*sin(alpha2), 0},
						{ a2*cos(beta2), 		 b2*cos(alpha2), c2}};
	
	double I[3][3] = {{1, 0, 0},
					  {0, 1, 0},
					  {0, 0, 1}};
	double A0Inv[3][3];
	int i,j;
	MatInv(A0,A0Inv);
	//for (i=0;i<3;i++) for (j=0;j<3;j++) printf("%.12f %.12f\n",A[i][j],A0Inv[i][j]);
	double Prod[3][3];
	MatMult(A,A0Inv,Prod);
	double ProdTranspose[3][3];
	TransposeM(Prod,ProdTranspose);
	double StrainTensorGr[3][3];
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			StrainTensorGr[i][j] = 0.5*(Prod[i][j] + ProdTranspose[i][j]) - I[i][j];
		}
	}
	double OrientTranspose[3][3], PreMult[3][3];
	TransposeM(Orient,OrientTranspose);
	MatMult(StrainTensorGr,OrientTranspose,PreMult);
	MatMult(Orient,PreMult,StrainTensorSample);
}

struct data_StrainFit{
	int nspots;
	double A[NR_MAX_IDS_PER_GRAIN][6];
	double B[NR_MAX_IDS_PER_GRAIN];
};

static
double problem_function(
    unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	struct data_StrainFit *f_data = (struct data_StrainFit *) f_data_trial;
	int nspots = f_data->nspots;
	int i,j;
	double TotDiff=0, InterDiff, Diff;
	for (i=0;i<nspots;i++){
		InterDiff = 0;
		for (j=0;j<6;j++){
			InterDiff += f_data->A[i][j]*x[j];
			
		}
		Diff = InterDiff - f_data->B[i];
		TotDiff += Diff*Diff;
	}
	return TotDiff;
}

inline int
StrainTensorKenesei(int nspots,double SpotsInfo[NR_MAX_IDS_PER_GRAIN][8], double Distance, double wavelength, double StrainTensorSample[3][3])
		/*SpotsInfo: 0,1,2 - Gobs, 3,4 - Y,Z spot, 5,6 - Y,Z sim, 7 - ID, G Vec should be normalized or not????*/
{
	int i,j;
	struct data_StrainFit mydata;
	double gobs[3],lenGobs;
	mydata.nspots = nspots;
	int id;
	int IDHash[NR_MAX_IDS_PER_GRAIN][3];
	double dspacings[NR_MAX_IDS_PER_GRAIN];
	int nRings=0;
	char aline[2000];
	char *hashfn = "IDsHash.csv";
	FILE *hashfile = fopen(hashfn,"r");
	while (fgets(aline,2000,hashfile)!=NULL){
		sscanf(aline,"%d %d %d %lf",&IDHash[nRings][0],&IDHash[nRings][0],&IDHash[nRings][0],&dspacings[nRings]);
		printf("%d %d %d %lf\n",IDHash[nRings][0],IDHash[nRings][0],IDHash[nRings][0],dspacings[nRings]);
		nRings++;
	}
	int ringNr;
	double ds0, dsObs;
	for (i=0;i<nspots;i++){
		lenGobs = CalcNorm3(SpotsInfo[i][0],SpotsInfo[i][1],SpotsInfo[i][2]);
		dsObs = wavelength/(2*sin(atan((CalcNorm2(SpotsInfo[i][3],SpotsInfo[i][4]))/Distance)/2));
		gobs[0] = SpotsInfo[i][0]/lenGobs;
		gobs[1] = SpotsInfo[i][1]/lenGobs;
		gobs[2] = SpotsInfo[i][2]/lenGobs;
		id = (int) SpotsInfo[i][7];
		printf("%d\n",id);
		for (j=0;j<nRings;j++){
			if (id >= IDHash[j][1] && id < IDHash[j][2]){
				ds0 = dspacings[j];
			}
		}
		if (ds0 == 0){
			return 0;
		}
		mydata.B[i] = (dsObs-ds0)/ds0;
		mydata.A[i][0] = gobs[0]*gobs[0];
		mydata.A[i][1] = gobs[1]*gobs[1];
		mydata.A[i][2] = gobs[2]*gobs[2];
		mydata.A[i][3] = gobs[0]*gobs[1]*2;
		mydata.A[i][4] = gobs[0]*gobs[2]*2;
		mydata.A[i][5] = gobs[1]*gobs[2]*2;
		ds0 = 0;
	}
	int n = 6;
	double x[n],xl[n],xu[n];
	for (i=0;i<n;i++){
		x[i] = 0;
		xl[i] = -0.01;
		xu[i] =  0.01;
	}
	struct data_StrainFit *f_datat;
	f_datat = &mydata;
	void* trp = (struct data_StrainFit *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function, trp);
	double minf=1;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
	StrainTensorSample[0][0] = x[0];
	StrainTensorSample[0][1] = x[3];
	StrainTensorSample[0][2] = x[4];
	StrainTensorSample[1][0] = x[3];
	StrainTensorSample[1][1] = x[1];
	StrainTensorSample[1][2] = x[5];
	StrainTensorSample[2][0] = x[4];
	StrainTensorSample[2][1] = x[5];
	StrainTensorSample[2][2] = x[2];
	return 1;
}
