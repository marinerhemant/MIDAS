//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
// GetMisorientation.c
//
//
// Hemant Sharma

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

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1e-9
static inline double sin_cos_to_angle (double s, double c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double sind(double x){return sin(deg2rad*x);}

static inline void normalizeQuat(double quat[4]){
	double norm = sqrt(quat[0]*quat[0]+quat[1]*quat[1]+quat[2]*quat[2]+quat[3]*quat[3]);
	quat[0] /= norm;
	quat[1] /= norm;
	quat[2] /= norm;
	quat[3] /= norm;
}

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
   {0.00000,   0.86603,  -0.50000,   0.00000},
   {0.50000,   0.00000,   0.00000,   0.86603},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.50000,  -0.00000,  -0.00000,  -0.86603},
   {0.00000,   0.86603,   0.50000,   0.00000}};

// double TrigSym[6][4] = {
//    {1.00000,   0.00000,   0.00000,   0.00000},
//    {0.50000,   0.00000,   0.00000,   0.86603},
//    {0.50000,  -0.00000,  -0.00000,  -0.86603},
//    {0.00000,   0.50000,  -0.86603,   0.00000},
//    {0.00000,   1.00000,   0.00000,   0.00000},
//    {0.00000,   0.50000,   0.86603,   0.00000}};

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
	normalizeQuat(Q);
}

inline
int MakeSymmetries(int SGNr, double Sym[24][4])
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

static inline
void BringDownToFundamentalRegionSym(double QuatIn[4], double QuatOut[4], int NrSymmetries, double Sym[24][4])
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
	normalizeQuat(QuatOut);
}

inline
void BringDownToFundamentalRegion(double QuatIn[4], double QuatOut[4],int SGNr)
{
	int i, j, maxCosRowNr=0, NrSymmetries;
	double Sym[24][4];
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
	normalizeQuat(QuatOut);
}

inline
double GetMisOrientation(double quat1[4], double quat2[4], double axis[3], double *Angle,int SGNr)
{
	double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];
	normalizeQuat(quat1);
	normalizeQuat(quat2);
	BringDownToFundamentalRegion(quat1,q1FR,SGNr);
	BringDownToFundamentalRegion(quat2,q2FR,SGNr);
	normalizeQuat(q1FR);
	normalizeQuat(q2FR);
	q1Inv[0] = -q1FR[0];
	q1Inv[1] =  q1FR[1];
	q1Inv[2] =  q1FR[2];
	q1Inv[3] =  q1FR[3];
	QuaternionProduct(q1Inv,q2FR,QP);
	BringDownToFundamentalRegion(QP,MisV,SGNr);
	if (MisV[0] > 1) MisV[0] = 1;
	double angle = 2*(acos(MisV[0]))*rad2deg;
	if (fabs(MisV[0] - 1) < 1E-10){
		axis[0] = 1;
		axis[1] = 0;
		axis[2] = 0;
	}else{
		axis[0] = MisV[1]/sqrt(1 - MisV[0]*MisV[0]);
		axis[1] = MisV[2]/sqrt(1 - MisV[0]*MisV[0]);
		axis[2] = MisV[3]/sqrt(1 - MisV[0]*MisV[0]);
	}
	*Angle = angle;
	return angle;
}

inline
double GetMisOrientationAngle(double quat1[4], double quat2[4], double *Angle, int NrSymmetries, double Sym[24][4])
{
	double q1FR[4], q2FR[4], QP[4], MisV[4];
	normalizeQuat(quat1);
	normalizeQuat(quat2);
	BringDownToFundamentalRegionSym(quat1,q1FR,NrSymmetries,Sym);
	BringDownToFundamentalRegionSym(quat2,q2FR,NrSymmetries,Sym);
	normalizeQuat(q1FR);
	normalizeQuat(q2FR);
	q1FR[0] = -q1FR[0];
	QuaternionProduct(q1FR,q2FR,QP);
	BringDownToFundamentalRegionSym(QP,MisV,NrSymmetries,Sym);
	if (MisV[0] > 1) MisV[0] = 1;
	double angle = 2*(acos(MisV[0]))*rad2deg;
	*Angle = angle;
	return angle;
}

inline 
void OrientMat2Quat(double OrientMat[9], double Quat[4]){
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
	normalizeQuat(Quat);
}

static inline
void OrientMat2Euler(double m[3][3],double Euler[3])
{
    double psi, phi, theta, sph;
	if (fabs(m[2][2] - 1.0) < EPS){
		phi = 0;
	}else{
	    phi = acos(m[2][2]);
	}
    sph = sin(phi);
    if (fabs(sph) < EPS)
    {
        psi = 0.0;
        theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0]) : sin_cos_to_angle(-m[1][0], m[0][0]);
    } else{
        psi = (fabs(-m[1][2] / sph) <= 1.0) ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph) : sin_cos_to_angle(m[0][2] / sph,1);
        theta = (fabs(m[2][1] / sph) <= 1.0) ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph) : sin_cos_to_angle(m[2][0] / sph,1);
    }
    Euler[0] = psi;
    Euler[1] = phi;
    Euler[2] = theta;
}

static inline
void Euler2OrientMat(
    double Euler[3],
    double m_out[3][3])
{
    double psi, phi, theta, cps, cph, cth, sps, sph, sth;
    psi = Euler[0];
    phi = Euler[1];
    theta = Euler[2];
    cps = cosd(psi) ; cph = cosd(phi); cth = cosd(theta);
    sps = sind(psi); sph = sind(phi); sth = sind(theta);
    m_out[0][0] = cth * cps - sth * cph * sps;
    m_out[0][1] = -cth * cph * sps - sth * cps;
    m_out[0][2] = sph * sps;
    m_out[1][0] = cth * sps + sth * cph * cps;
    m_out[1][1] = cth * cph * cps - sth * sps;
    m_out[1][2] = -sph * cps;
    m_out[2][0] = sth * sph;
    m_out[2][1] = cth * sph;
    m_out[2][2] = cph;
}
