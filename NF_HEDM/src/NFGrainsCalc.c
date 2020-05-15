//
// We will provide the orientTol, 3 euler angle arrays, dimensions of the arrays and fillVal, will get back grain IDs
//

#include<stdio.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

double Sym[24][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.70711,   0.70711,   0.00000,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.70711,  -0.70711,   0.00000,   0.00000},
   {0.70711,   0.00000,   0.70711,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.70711,   0.00000,  -0.70711,   0.00000},
   {0.70711,   0.00000,   0.00000,   0.70711},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.70711,   0.00000,   0.00000,  -0.70711},
   {0.50000,   0.50000,   0.50000,   0.50000},
   {0.50000,  -0.50000,  -0.50000,  -0.50000},
   {0.50000,  -0.50000,   0.50000,   0.50000},
   {0.50000,   0.50000,  -0.50000,  -0.50000},
   {0.50000,   0.50000,  -0.50000,   0.50000},
   {0.50000,  -0.50000,   0.50000,  -0.50000},
   {0.50000,  -0.50000,  -0.50000,   0.50000},
   {0.50000,   0.50000,   0.50000,  -0.50000},
   {0.00000,   0.70711,   0.70711,   0.00000},
   {0.00000,  -0.70711,   0.70711,   0.00000},
   {0.00000,   0.70711,   0.00000,   0.70711},
   {0.00000,   0.70711,   0.00000,  -0.70711},
   {0.00000,   0.00000,   0.70711,   0.70711},
   {0.00000,   0.00000,   0.70711,  -0.70711}};

int diffArr[3][26] = {{-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1},
			  {-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1},
			  {-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1}};

static inline
void QuaternionProduct(double q[4], double r[4], double Q[4])
{
	Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3];
	Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3];
	Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1];
	Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2];
	if (Q[0] < 0) {
		Q[0] = -Q[0];
		Q[1] = -Q[1];
		Q[2] = -Q[2];
		Q[3] = -Q[3];
	}
}

static inline
void BringDownToFundamentalRegionSym(double QuatIn[4], double QuatOut[4], int NrSymmetries)
{
	int i, maxCosRowNr;
	double qps[NrSymmetries][4], q2[4], qt[4], maxCos=-10000;
	for (i=0;i<NrSymmetries;i++){
		q2[0] = Sym[i][0];
		q2[1] = Sym[i][1];
		q2[2] = Sym[i][2];
		q2[3] = Sym[i][3];
		QuaternionProduct(QuatIn,q2,qt);
		qps[i][0] = qt[0];
		qps[i][1] = qt[1];
		qps[i][2] = qt[2];
		qps[i][3] = qt[3];
		if (maxCos < qt[0]){
			maxCos = qt[0];
			maxCosRowNr = i;
		}
	}
	QuatOut[0] = qps[maxCosRowNr][0];
	QuatOut[1] = qps[maxCosRowNr][1];
	QuatOut[2] = qps[maxCosRowNr][2];
	QuatOut[3] = qps[maxCosRowNr][3];
}

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

static inline
void Euler2Quat(double Euler[3],double Quat[4]){
	double psi, phi, theta, cps, cph, cth, sps, sph, sth;
	double OrientMat[9];
	psi = Euler[0];
	phi = Euler[1];
	theta = Euler[2];
	cps = cosd(psi) ; cph = cosd(phi); cth = cosd(theta);
	sps = sind(psi); sph = sind(phi); sth = sind(theta);
	OrientMat[0] = cth * cps - sth * cph * sps;
	OrientMat[1] = -cth * cph * sps - sth * cps;
	OrientMat[2] = sph * sps;
	OrientMat[3] = cth * sps + sth * cph * cps;
	OrientMat[4] = cth * cph * cps - sth * sps;
	OrientMat[5] = -sph * cps;
	OrientMat[6] = sth * sph;
	OrientMat[7] = cth * sph;
	OrientMat[8] = cph;
	double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
	if(trace > 0){
		double s = 0.5/sqrt(trace+1.0);
		Quat[0] = 0.25/s;
		Quat[1] = (OrientMat[7]-OrientMat[5])*s;
		Quat[2] = (OrientMat[2]-OrientMat[6])*s;
		Quat[3] = (OrientMat[3]-OrientMat[1])*s;
	}else{
		if (OrientMat[0]>OrientMat[4] && OrientMat[0]>OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8]);
			Quat[0] = (OrientMat[7]-OrientMat[5])/s;
			Quat[1] = 0.25*s;
			Quat[2] = (OrientMat[1]+OrientMat[3])/s;
			Quat[3] = (OrientMat[2]+OrientMat[6])/s;
		} else if (OrientMat[4] > OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8]);
			Quat[0] = (OrientMat[2]-OrientMat[6])/s;
			Quat[1] = (OrientMat[1]+OrientMat[3])/s;
			Quat[2] = 0.25*s;
			Quat[3] = (OrientMat[5]+OrientMat[7])/s;
		} else {
			double s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4]);
			Quat[0] = (OrientMat[3]-OrientMat[1])/s;
			Quat[1] = (OrientMat[2]+OrientMat[6])/s;
			Quat[2] = (OrientMat[5]+OrientMat[7])/s;
			Quat[3] = 0.25*s;
		}
	}
	if (Quat[0] < 0){
		Quat[0] = -Quat[0];
		Quat[1] = -Quat[1];
		Quat[2] = -Quat[2];
		Quat[3] = -Quat[3];
	}
	double QNorm = sqrt(Quat[0]*Quat[0] + Quat[1]*Quat[1] + Quat[2]*Quat[2] + Quat[3]*Quat[3]);
	Quat[0] /= QNorm;
	Quat[1] /= QNorm;
	Quat[2] /= QNorm;
	Quat[3] /= QNorm;
}

inline
double GetMisOrientationAngle(double Eul1[3], double Eul2[3], double *Angle, int NrSymmetries)
{
	double quat1[4], quat2[4];
	Euler2Quat(Eul1,quat1);
	Euler2Quat(Eul2,quat2);
	double q1FR[4], q2FR[4], QP[4], MisV[4];
	BringDownToFundamentalRegionSym(quat1,q1FR,NrSymmetries);
	BringDownToFundamentalRegionSym(quat2,q2FR,NrSymmetries);
	q1FR[0] = -q1FR[0];
	QuaternionProduct(q1FR,q2FR,QP);
	BringDownToFundamentalRegionSym(QP,MisV,NrSymmetries);
	if (MisV[0] > 1) MisV[0] = 1;
	double angle = 2*(acos(MisV[0]))*rad2deg;
	*Angle = angle;
	return angle;
}

inline long long int getIDX (int layerNr, int xpos, int ypos, int xMax, int yMax){
	long long int retval = layerNr;
	retval *= xMax;
	retval *= yMax;
	retval += xpos * yMax;
	retval += ypos;
	return retval;
}

inline void DFS(int a, int b, int c, int grainNr, int *dims, int NrSymmetries, double *Euler1, double *Euler2, double *Euler3, int *grains, double fillVal, double orientTol){
	long long int Pos = getIDX(a,b,c,dims[1],dims[2]);
	if (grains[Pos] != 0) return;
	grains[Pos] = grainNr;
	double Eul1[3],Eul2[3], quat1[4], quat2[4];
	Eul1[0] = Euler1[Pos];
	Eul1[1] = Euler2[Pos];
	Eul1[2] = Euler3[Pos];
	Euler2Quat(Eul1,quat1);
	int diff;
	double ang, miso;
	for (diff = 0; diff < 26; diff++){
		int a2 = a + diffArr[0][diff];
		int b2 = b + diffArr[1][diff];
		int c2 = c + diffArr[2][diff];
		if (a2 < 0 || a2 == dims[0]) continue;
		if (b2 < 0 || b2 == dims[1]) continue;
		if (c2 < 0 || c2 == dims[2]) continue;
		long long int Pos2 = getIDX(a2,b2,c2,dims[1],dims[2]);
		Eul2[0] = Euler1[Pos2];
		Eul2[1] = Euler2[Pos2];
		Eul2[2] = Euler3[Pos2];
		Euler2Quat(Eul2,quat2);
		if (quat2[0] == fillVal){
			grains[Pos2] = fillVal;
			continue;
		}
		miso = GetMisOrientationAngle(quat1,quat2,&ang,NrSymmetries);
		printf("%d %d %d %d %lf %lf\n",a2,b2,c2,grainNr,miso,ang);
		fflush(stdout);
		if (miso <= orientTol){
			DFS(a2,b2,c2,grainNr,dims,NrSymmetries,Euler1,Euler2,Euler3,grains,fillVal,orientTol);
		}
	}
}

void calcGrainNrs (double orientTol, double *Euler1, double *Euler2, double *Euler3, int nrLayers, int xMax, int yMax, double fillVal, int NrSymmetries, int *GrainNrs)
{
	int layernr,xpos,ypos;
	int grainNr = 0;
	int i,j;
	int dims[3] = {nrLayers,xMax,yMax};
	for (layernr = 0; layernr < nrLayers; layernr++){
		for (xpos = 0; xpos < xMax; xpos++){
			for (ypos = 0; ypos < yMax; ypos++){
				if (Euler1[getIDX(layernr,xpos,ypos,xMax,yMax)] == fillVal){
					GrainNrs[getIDX(layernr,xpos,ypos,xMax,yMax)] = (int)fillVal;
				} else {
					// call DFS here.
					grainNr ++;
					DFS(layernr,xpos,ypos,grainNr,dims,NrSymmetries,Euler1,Euler2,Euler3,GrainNrs,fillVal,orientTol);
				}
			}
		}
	}
}

int main(int argc,char *argv[]){
	// Read in Euler1, Euler2, Euler3, Symm, allocate: GrainNrs
	double orientTol = 5.0;
	int nrLayers = 3;
	int xMax = 900;
	int yMax = 900;
	double fillVal = -15;
	int nrSymmetries = 24;
	int *GrainNrs;
	GrainNrs = calloc(nrLayers*xMax*yMax,sizeof(*GrainNrs));
	FILE *f1 = fopen("EulerAngles1.bin","rb");
	FILE *f2 = fopen("EulerAngles2.bin","rb");
	FILE *f3 = fopen("EulerAngles3.bin","rb");
	double *Euler1, *Euler2, *Euler3;
	Euler1 = calloc(nrLayers*xMax*yMax,sizeof(*Euler1));
	Euler2 = calloc(nrLayers*xMax*yMax,sizeof(*Euler2));
	Euler3 = calloc(nrLayers*xMax*yMax,sizeof(*Euler3));
	fread(Euler1,nrLayers*xMax*yMax*sizeof(double),1,f1);
	fread(Euler2,nrLayers*xMax*yMax*sizeof(double),1,f2);
	fread(Euler3,nrLayers*xMax*yMax*sizeof(double),1,f3);
	calcGrainNrs (orientTol, Euler1, Euler2, Euler3, nrLayers, xMax, yMax, fillVal, nrSymmetries, GrainNrs);
}
