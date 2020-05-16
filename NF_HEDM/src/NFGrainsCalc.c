//
// We will provide the orientTol, 3 euler angle arrays, dimensions of the arrays and fillVal, will get back grain IDs
//

#include<stdio.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

double Sym[24][4];

double TricSym[2][4] = { // This is just for house keeping to make it 2 rows
   {1.00000,   0.00000,   0.00000,   0.00000},
   {1.00000,   0.00000,   0.00000,   0.00000}};

double MonoSym[2][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000}};

double OrtSym[4][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {1.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,   0.00000,   0.00000,   1.00000}};

double TetSym[8][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.70711,   0.00000,   0.00000,   0.70711},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.70711,  -0.00000,  -0.00000,  -0.70711},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,   0.70711,   0.70711,   0.00000},
   {0.00000,  -0.70711,   0.70711,   0.00000}};

double TrigSym[6][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.50000,   0.00000,   0.00000,   0.86603},
   {0.50000,  -0.00000,  -0.00000,  -0.86603},
   {0.00000,   0.50000,  -0.86603,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.50000,   0.86603,   0.00000}};

double HexSym[12][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.86603,   0.00000,   0.00000,   0.50000},
   {0.50000,   0.00000,   0.00000,   0.86603},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.50000,  -0.00000,  -0.00000,  -0.86603},
   {0.86603,  -0.00000,  -0.00000,  -0.50000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.86603,   0.50000,   0.00000},
   {0.00000,   0.50000,   0.86603,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,  -0.50000,   0.86603,   0.00000},
   {0.00000,  -0.86603,   0.50000,   0.00000}};

double CubSym[24][4] = {
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

inline
int MakeSymmetries(int SGNr)
{
	int i, j, NrSymmetries;;
	if (SGNr <= 2){ // Triclinic
		NrSymmetries = 1;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TricSym[i][j];
			}
		}
	}else if (SGNr > 2 && SGNr <= 15){  // Monoclinic
		NrSymmetries = 2;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = MonoSym[i][j];
			}
		}
	}else if (SGNr >= 16 && SGNr <= 74){ // Orthorhombic
		NrSymmetries = 4;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = OrtSym[i][j];
			}
		}
	}else if (SGNr >= 75 && SGNr <= 142){  // Tetragonal
		NrSymmetries = 8;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TetSym[i][j];
			}
		}
	}else if (SGNr >= 143 && SGNr <= 167){ // Trigonal
		NrSymmetries = 6;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TrigSym[i][j];
			}
		}
	}else if (SGNr >= 168 && SGNr <= 194){ // Hexagonal
		NrSymmetries = 12;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = HexSym[i][j];
			}
		}
	}else if (SGNr >= 195 && SGNr <= 230){ // Cubic
		NrSymmetries = 24;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = CubSym[i][j];
			}
		}
	}
	return NrSymmetries;
}

int diffArr[3][13] = {{-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0},
					  {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0},
					  {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1}};

static inline
void QuaternionProduct(double *q, double *r, double *Q)
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
void BringDownToFundamentalRegionSym(double *QuatIn, double *QuatOut, int NrSymmetries)
{
	int i, maxCosRowNr;
	double *qps, *q2, *qt, maxCos=-10000;
	qps = calloc(NrSymmetries*4,sizeof(*qps));
	q2 = calloc(4,sizeof(*q2));
	qt = calloc(4,sizeof(*qt));
	for (i=0;i<NrSymmetries;i++){
		q2[0] = Sym[i][0];
		q2[1] = Sym[i][1];
		q2[2] = Sym[i][2];
		q2[3] = Sym[i][3];
		QuaternionProduct(QuatIn,q2,qt);
		qps[i*4+0] = qt[0];
		qps[i*4+1] = qt[1];
		qps[i*4+2] = qt[2];
		qps[i*4+3] = qt[3];
		if (maxCos < qt[0]){
			maxCos = qt[0];
			maxCosRowNr = i;
		}
	}
	free(q2);
	free(qt);
	QuatOut[0] = qps[maxCosRowNr*4+0];
	QuatOut[1] = qps[maxCosRowNr*4+1];
	QuatOut[2] = qps[maxCosRowNr*4+2];
	QuatOut[3] = qps[maxCosRowNr*4+3];
	free(qps);
}


static inline
void Euler2Quat(double *Euler, double *Quat){
	double psi, phi, theta, cps, cph, cth, sps, sph, sth;
	double *OrientMat;
	OrientMat = calloc(9,sizeof(*OrientMat));
	psi = Euler[0];
	phi = Euler[1];
	theta = Euler[2];
	cps = cos(psi) ; cph = cos(phi); cth = cos(theta);
	sps = sin(psi); sph = sin(phi); sth = sin(theta);
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
	free(OrientMat);
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
double GetMisOrientationAngle(double *Eul1, double *Eul2, double *Angle, int NrSymmetries)
{
	double *quat1, *quat2;
	quat1 = calloc(4,sizeof(*quat1));
	quat2 = calloc(4,sizeof(*quat2));
	Euler2Quat(Eul1,quat1);
	Euler2Quat(Eul2,quat2);
	double *q1FR, *q2FR, *QP, *MisV;
	q1FR = calloc(4,sizeof(*q1FR));
	q2FR = calloc(4,sizeof(*q2FR));
	QP = calloc(4,sizeof(*QP));
	MisV = calloc(4,sizeof(*MisV));
	BringDownToFundamentalRegionSym(quat1,q1FR,NrSymmetries);
	BringDownToFundamentalRegionSym(quat2,q2FR,NrSymmetries);
	free(quat1);
	free(quat2);
	q1FR[0] = -q1FR[0];
	QuaternionProduct(q1FR,q2FR,QP);
	free(q1FR);
	free(q2FR);
	BringDownToFundamentalRegionSym(QP,MisV,NrSymmetries);
	free(QP);
	if (MisV[0] > 1) MisV[0] = 1;
	double angle = 2*(acos(MisV[0]))*rad2deg;
	free(MisV);
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

int Dims[3];
double fV, oT;
int NSym;
double *Euler1, *Euler2, *Euler3;
int *GrainNrs;

inline void DFS (int *Pos, int grainNr){
	long long int Pos1 = getIDX(Pos[0],Pos[1],Pos[2],Dims[1],Dims[2]);
	if (GrainNrs[Pos1] != 0) return;
	GrainNrs[Pos1] = grainNr;
	printf("Grain Nr: %d\n",grainNr);
	int i;
	double *Eul1,*Eul2, miso, ang;
	Eul1 = calloc(3,sizeof(*Eul1));
	Eul2 = calloc(3,sizeof(*Eul2));
	Eul1[0] = Euler1[Pos1];
	Eul1[1] = Euler2[Pos1];
	Eul1[2] = Euler3[Pos1];
	int a2,b2,c2;
	long long int Pos2;
	int *PosNext;
	PosNext = calloc(3,sizeof(int));
	for (i=0;i<13;i++){
		a2 = Pos[0] + diffArr[0][i];
		b2 = Pos[1] + diffArr[1][i];
		c2 = Pos[2] + diffArr[2][i];
		if (a2 < 0 || a2 == Dims[0]) continue;
		if (b2 < 0 || b2 == Dims[1]) continue;
		if (c2 < 0 || c2 == Dims[2]) continue;
		Pos2 = getIDX(a2,b2,c2,Dims[1],Dims[2]);
		if (Euler1[Pos2] == fV) continue;
		Eul2[0] = Euler1[Pos2];
		Eul2[1] = Euler2[Pos2];
		Eul2[2] = Euler3[Pos2];
		miso = GetMisOrientationAngle(Eul1,Eul2,&ang,NSym);
		if (miso < oT){
			PosNext[0] = a2;
			PosNext[1] = b2;
			PosNext[2] = c2;
			DFS(PosNext,grainNr);
		}
	}
	free(Eul1);
	free(Eul2);
	free(PosNext);
}

void calcGrainNrs (double orientTol, int nrLayers, int xMax, int yMax, double fillVal, int SGNum)
{
	int NrSymmetries;
	NrSymmetries = MakeSymmetries(SGNum);
	NSym = NrSymmetries;
	Dims[0] = nrLayers;
	Dims[1] = xMax;
	Dims[2] = yMax;
	fV = fillVal;
	oT = orientTol;
	GrainNrs = calloc(nrLayers*xMax*yMax,sizeof(*GrainNrs));
	FILE *f1 = fopen("EulerAngles1.bin","rb");
	FILE *f2 = fopen("EulerAngles2.bin","rb");
	FILE *f3 = fopen("EulerAngles3.bin","rb");
	Euler1 = calloc(nrLayers*xMax*yMax,sizeof(*Euler1));
	Euler2 = calloc(nrLayers*xMax*yMax,sizeof(*Euler2));
	Euler3 = calloc(nrLayers*xMax*yMax,sizeof(*Euler3));
	fread(Euler1,nrLayers*xMax*yMax*sizeof(double),1,f1);
	fread(Euler2,nrLayers*xMax*yMax*sizeof(double),1,f2);
	fread(Euler3,nrLayers*xMax*yMax*sizeof(double),1,f3);
	int layernr,xpos,ypos,a2,b2,c2;
	int grainNr = 0;
	int i,j, *Pos;
	Pos = calloc(3,sizeof(int));
	long long int Pos1;
	//~ int GrainFound;
	for (layernr = 0; layernr < nrLayers; layernr++){
		for (xpos = 0; xpos < xMax; xpos++){
			for (ypos = 0; ypos < yMax; ypos++){
				Pos[0] = layernr;
				Pos[1] = xpos;
				Pos[2] = ypos;
				Pos1 = getIDX(Pos[0],Pos[1],Pos[2],Dims[1],Dims[2]);
				if (Euler1[Pos1] == fillVal){
					GrainNrs[Pos1] = (int)fillVal;
				} else if (GrainNrs[Pos1] == 0){
					grainNr++;
					printf("Initial grainNr: %d\n",grainNr);
					DFS(Pos,grainNr);
				}
					//~ GrainFound = 0;
					//~ Eul1[0] = Euler1[Pos1];
					//~ Eul1[1] = Euler2[Pos1];
					//~ Eul1[2] = Euler3[Pos1];
					//~ // Calculate misorientation with the neighbors, whichever one fits, give that number and continue.
					//~ for (i=0;i<13;i++){
						//~ a2 = layernr + diffArr[0][i];
						//~ b2 = xpos + diffArr[1][i];
						//~ c2 = ypos + diffArr[2][i];
						//~ if (a2 < 0 || a2 == nrLayers) continue;
						//~ if (b2 < 0 || b2 == xMax) continue;
						//~ if (c2 < 0 || c2 == yMax) continue;
						//~ Pos2 = getIDX(a2,b2,c2,xMax,yMax);
						//~ if (Euler1[Pos2] == fillVal) continue;
						//~ Eul2[0] = Euler1[Pos2];
						//~ Eul2[1] = Euler2[Pos2];
						//~ Eul2[2] = Euler3[Pos2];
						//~ miso = GetMisOrientationAngle(Eul1,Eul2,&ang,NrSymmetries);
						//~ if (miso < orientTol){
							//~ GrainNrs[Pos1] = GrainNrs[Pos2];
							//~ GrainFound = 1;
							//~ break;
						//~ }
					//~ }
					//~ if (GrainFound == 0){
						//~ // No neighbor matched, new grain.
						//~ grainNr ++;
						//~ GrainNrs[Pos1] = grainNr;
					//~ }
				//~ }
			}
		}
	}
	printf("Total number of grains: %d\n",grainNr);
	FILE *f4 = fopen("GrainNrs.bin","wb");
	fwrite(GrainNrs,nrLayers*xMax*yMax*sizeof(int),1,f4);
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
	free(Euler1);
	free(Euler2);
	free(Euler3);
	free(GrainNrs);
}

//~ int main(int argc,char *argv[]){
	//~ // Read in Euler1, Euler2, Euler3, Symm, allocate: GrainNrs
	//~ double orientTol = 5.0;
	//~ int nrLayers = 3;
	//~ int xMax = 900;
	//~ int yMax = 900;
	//~ double fillVal = -15;
	//~ int nrSymmetries = 24;
	//~ int *GrainNrs;
	//~ GrainNrs = calloc(nrLayers*xMax*yMax,sizeof(*GrainNrs));
	//~ FILE *f1 = fopen("EulerAngles1.bin","rb");
	//~ FILE *f2 = fopen("EulerAngles2.bin","rb");
	//~ FILE *f3 = fopen("EulerAngles3.bin","rb");
	//~ double *Euler1, *Euler2, *Euler3;
	//~ Euler1 = calloc(nrLayers*xMax*yMax,sizeof(*Euler1));
	//~ Euler2 = calloc(nrLayers*xMax*yMax,sizeof(*Euler2));
	//~ Euler3 = calloc(nrLayers*xMax*yMax,sizeof(*Euler3));
	//~ fread(Euler1,nrLayers*xMax*yMax*sizeof(double),1,f1);
	//~ fread(Euler2,nrLayers*xMax*yMax*sizeof(double),1,f2);
	//~ fread(Euler3,nrLayers*xMax*yMax*sizeof(double),1,f3);
	//~ calcGrainNrs (orientTol, Euler1, Euler2, Euler3, nrLayers, xMax, yMax, fillVal, nrSymmetries, GrainNrs);
	//~ FILE *f4 = fopen("GrainNrs.bin","wb");
	//~ fwrite(GrainNrs,nrLayers*xMax*yMax*sizeof(int),1,f4);
	//~ fclose(f1);
	//~ fclose(f2);
	//~ fclose(f3);
	//~ fclose(f4);
//~ }
