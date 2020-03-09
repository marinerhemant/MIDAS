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
#include <stdint.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <nlopt.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-10
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}
static inline double sin_cos_to_angle (double s, double c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}

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
CalcEtaAngle(double y,double z,double *alpha) {
	*alpha = rad2deg * acos(z/sqrt(y*y+z*z));
	if (y > 0)    *alpha = -*alpha;
}


static inline void MatrixMult(double m[3][3], double v[3], double r[3]){
	int i;
	for (i=0; i<3; i++) {
		r[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2];
	}
}

static inline void CorrectHKLsLatC(double LatC[6], double *hklsIn,int nhkls,double Lsd,double Wavelength,double *hkls)
{
	double a=LatC[0],b=LatC[1],c=LatC[2],alpha=LatC[3],beta=LatC[4],gamma=LatC[5];
	int hklnr;
	double SinA = sind(alpha), SinB = sind(beta), SinG = sind(gamma), CosA = cosd(alpha), CosB = cosd(beta), CosG = cosd(gamma);
	double GammaPr = acosd((CosA*CosB - CosG)/(SinA*SinB)), BetaPr  = acosd((CosG*CosA - CosB)/(SinG*SinA)), SinBetaPr = sind(BetaPr);
	double Vol = (a*(b*(c*(SinA*(SinBetaPr*(SinG)))))), APr = b*c*SinA/Vol, BPr = c*a*SinB/Vol, CPr = a*b*SinG/Vol;
	double B[3][3]; B[0][0] = APr; B[0][1] = (BPr*cosd(GammaPr)), B[0][2] = (CPr*cosd(BetaPr)), B[1][0] = 0,
		B[1][1] = (BPr*sind(GammaPr)), B[1][2] = (-CPr*SinBetaPr*CosA), B[2][0] = 0, B[2][1] = 0, B[2][2] = (CPr*SinBetaPr*SinA);
	for (hklnr=0;hklnr<nhkls;hklnr++){
		double ginit[3]; ginit[0] = hklsIn[hklnr*4+0]; ginit[1] = hklsIn[hklnr*4+1]; ginit[2] = hklsIn[hklnr*4+2];
		double GCart[3];
		MatrixMult(B,ginit,GCart);
		double Ds = 1/(sqrt((GCart[0]*GCart[0])+(GCart[1]*GCart[1])+(GCart[2]*GCart[2])));
		hkls[hklnr*5+0] = GCart[0];
		hkls[hklnr*5+1] = GCart[1];
		hkls[hklnr*5+2] = GCart[2];
		hkls[hklnr*5+3] = asind((Wavelength)/(2*Ds)); // Theta
		hkls[hklnr*5+4] = hklsIn[hklnr*4+3]; // RingNr
	}
}

static inline
void Euler2OrientMat( double Euler[3], double m_out[3][3]) // Must be in degrees
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
    Euler[0] = rad2deg*psi;
    Euler[1] = rad2deg*phi;
    Euler[2] = rad2deg*theta;
}

static inline double intersectionCalc(double y, double y1, double y2, double x1, double x2)
{
	return x1+(y-y1)*(x2-x1)/(y2-y1);
}

static inline void
RotateAroundZ( double v1[3], double alpha, double v2[3]) {
	double cosa = cos(alpha*deg2rad);
	double sina = sin(alpha*deg2rad);
	double mat[3][3] = {{ cosa, -sina, 0 },
						{ sina,  cosa, 0 },
						{ 0,     0,    1}};
	MatrixMult(mat, v1, v2);
}

static inline void
CalcOmega(double x, double y, double z, double theta, double omegas[4], double etas[4], int * nsol)
{
	*nsol = 0;
	double ome;
	double len= sqrt(x*x + y*y + z*z);
	double v=sin(theta*deg2rad)*len;
	double almostzero = 1e-4;
	if ( fabs(y) < almostzero ) {
		if (x != 0) {
			double cosome1 = -v/x;
			if (fabs(cosome1 <= 1)) {
				ome = acos(cosome1)*rad2deg;
				omegas[*nsol] = ome;
				*nsol = *nsol + 1;
				omegas[*nsol] = -ome;
				*nsol = *nsol + 1;
			}
		}
	} else {
		double y2 = y*y;
		double a = 1 + ((x*x) / y2);
		double b = (2*v*x) / y2;
		double c = ((v*v) / y2) - 1;
		double discr = b*b - 4*a*c;
		double ome1a;
		double ome1b;
		double ome2a;
		double ome2b;
		double cosome1;
		double cosome2;
		double eqa, eqb, diffa, diffb;
		if (discr >= 0) {
			cosome1 = (-b + sqrt(discr))/(2*a);
			if (fabs(cosome1) <= 1) {
				ome1a = acos(cosome1);
				ome1b = -ome1a;
				eqa = -x*cos(ome1a) + y*sin(ome1a);
				diffa = fabs(eqa - v);
				eqb = -x*cos(ome1b) + y*sin(ome1b);
				diffb = fabs(eqb - v);
				if (diffa < diffb ) {
					omegas[*nsol] = ome1a*rad2deg;
					*nsol = *nsol + 1;
				} else {
					omegas[*nsol] = ome1b*rad2deg;
					*nsol = *nsol + 1;
				}
			}
			cosome2 = (-b - sqrt(discr))/(2*a);
			if (fabs(cosome2) <= 1) {
				ome2a = acos(cosome2);
				ome2b = -ome2a;
				eqa = -x*cos(ome2a) + y*sin(ome2a);
				diffa = fabs(eqa - v);
				eqb = -x*cos(ome2b) + y*sin(ome2b);
				diffb = fabs(eqb - v);
				if (diffa < diffb) {
					omegas[*nsol] = ome2a*rad2deg;
					*nsol = *nsol + 1;
				} else {
					omegas[*nsol] = ome2b*rad2deg;
					*nsol = *nsol + 1;
				}
			}
		}
	}
	double gw[3];
	double gv[3]={x,y,z};
	double eta;
	int indexOme;
	for (indexOme = 0; indexOme < *nsol; indexOme++) {
		RotateAroundZ(gv, omegas[indexOme], gw);
		CalcEtaAngle(gw[1],gw[2], &eta);
		etas[indexOme] = eta;
	}
}


double dx[4] = {-0.5,+0.5,+0.5,-0.5};
double dy[4] = {-0.5,-0.5,+0.5,+0.5};

// Function to calculate the fraction of a voxel in a beam profile. Omega[degrees]
// Assuming a gaussian beam profile.
static inline double IntensityFraction(double voxLen, double beamPosition, double beamFWHM, double voxelPosition[3], double Omega)
{
	double xy[4][2], xyPr[4][2], minY=1e6, maxY=-1e6, startY, endY, yStep, intX, volFr=0, sigma, thisPos, delX;
	int inSide=0, nrYs=200, i, j, splCase = 0;
	double omePr,etaPr,eta;
	sigma = beamFWHM/(2*sqrt(2*log(2)));
	// Convert from Omega to Eta (look in the computation notebook, 08/19/19 pg. 34 for calculation)
	if (Omega < 0) omePr = 360 + Omega;
	else omePr = Omega;
	if (abs(omePr) < 1e-5) splCase = 1;
	else if (abs(omePr-90) < 1e-5) splCase = 1;
	else if (abs(omePr-180) < 1e-5) splCase = 1;
	else {
		if (omePr < 90) etaPr = 90 - omePr;
		else if (omePr < 180) etaPr = 180 - omePr;
		else if (omePr < 270) etaPr = 270 - omePr;
		else etaPr = 360 - omePr;
		if (etaPr < 45) eta = 90 - etaPr;
		else eta = etaPr;
	}
	// What we need: minY, maxY, startY, endY
	for (i=0;i<4;i++) {
		xy[i][0] = voxelPosition[0] + dx[i]*voxLen;
		xy[i][1] = voxelPosition[1] + dy[i]*voxLen;
		xyPr[i][1] = xy[i][0]*sind(Omega) + xy[i][1]*cosd(Omega);
		if (xyPr[i][1] < minY) {minY = xyPr[i][1];}
		if (xyPr[i][1] > maxY) {maxY = xyPr[i][1];}
	}
	if (maxY >= beamPosition - beamFWHM && minY <= beamPosition + beamFWHM) inSide = 1;
	if (inSide == 1){
		startY = (minY > beamPosition - beamFWHM) ? minY : beamPosition - beamFWHM;
		endY = (maxY < beamPosition + beamFWHM) ? maxY : beamPosition + beamFWHM;
		yStep = (endY-startY)/((double)nrYs);
		for (i=0;i<=nrYs;i++){
			if (splCase == 1) delX = 1;
			else{
				thisPos = i*yStep;
				if (thisPos < voxLen*cosd(eta)) delX = thisPos * (tand(eta)+ (1/tand(eta)));
				else if (maxY-minY - thisPos < voxLen*cosd(eta)) delX = (maxY-minY-thisPos) * (tand(eta)+ (1/tand(eta)));
				else delX = voxLen*(sind(eta)+(cosd(eta)/tand(eta)));
			}
			thisPos = startY + i*yStep;
			intX = yStep*exp(-((thisPos-beamPosition)*(thisPos-beamPosition))/(2*sigma*sigma))/(sigma*sqrt(2*M_PI));
			volFr += intX * delX;
		}
	}
	return volFr;
}

static inline
void SpotToGv(double xi, double yi, double zi, double Omega, double theta, double *g1, double *g2, double *g3)
{
	double CosOme = cosd(Omega), SinOme = sind(Omega);
	double eta;
	CalcEtaAngle(yi,zi,&eta);
	double TanEta = tand(-eta), SinTheta = sind(theta);
    double CosTheta = cosd(theta), CosW = 1, SinW = 0, k3 = SinTheta*(1+xi)/((yi*TanEta)+zi), k2 = TanEta*k3, k1 = -SinTheta;
    if (eta == 90){
		k3 = 0;
		k2 = -CosTheta;
	} else if (eta == -90){
		k3 = 0;
		k2 = CosTheta;
	}
    double k1f = (k1*CosW) + (k3*SinW), k3f = (k3*CosW) - (k1*SinW), k2f = k2;
    *g1 = (k1f*CosOme) + (k2f*SinOme);
    *g2 = (k2f*CosOme) - (k1f*SinOme);
    *g3 = k3f;
}

// Function to calculate y,z,ome of diffraction spots, given euler angles (degrees), position and lattice parameter.
// Ideal hkls need to be provided, with 4 columns: h,k,l,ringNr
// Output for comparisonType 0: g1,g2,g3,eta,omega,y,z,2theta,nrhkls
static inline int CalcDiffractionSpots(double Lsd, double Wavelength,
			double position[3], double LatC[6], double EulerAngles[3],
			int nhkls, double *hklsIn, double *spotPos, int comparisonType)
{
	double *hkls; // We need h,k,l,theta,ringNr
	hkls = calloc(nhkls*5,sizeof(*hkls));
	CorrectHKLsLatC(LatC,hklsIn,nhkls,Lsd,Wavelength,hkls);
	double Gc[3],Ghkl[3],cosOme,sinOme,yspot,zspot,yprr;
	double omegas[4], etas[4], lenK, xs, ys, zs, th,g1,g2,g3;
	double yspots, zspots, xGr=position[0], yGr=position[1], zGr=position[2], xRot, yRot, yPrr, zPrr;
	double OM[3][3], theta, RingRadius, omega, eta, etanew, nrhkls;
	int hklnr, nspotsPlane, i;
	Euler2OrientMat(EulerAngles,OM);
	int spotNr = 0;
	for (hklnr=0;hklnr<nhkls;hklnr++){
		Ghkl[0] = hkls[hklnr*5+0];
		Ghkl[1] = hkls[hklnr*5+1];
		Ghkl[2] = hkls[hklnr*5+2];
		MatrixMult(OM,Ghkl, Gc);
		theta = hkls[hklnr*5+3];
		CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
		nrhkls = (double)hklnr*2 + 1;
		for (i=0;i<nspotsPlane;i++){
			omega = omegas[i];
			eta = etas[i];
			if (isnan(omega) || isnan(eta)) continue;
			cosOme = cosd(omega);
			sinOme = sind(omega);
			xRot = xGr*cosOme - yGr*sinOme;
			yRot = xGr*sinOme + yGr*cosOme;
			RingRadius = tand(2*theta)*(Lsd + xRot);
			yPrr = -(sind(eta)*RingRadius);
			zPrr = cosd(eta)*RingRadius;
			yspot = yprr + yRot;
			zspot = zPrr + zGr;
			RingRadius = sqrt(yspot*yspot + zspot*zspot);
			CalcEtaAngle(yspot,zspot,&etanew);
			th = atand(RingRadius/Lsd)/2;
			spotPos[spotNr*7+3] = etanew;
			spotPos[spotNr*7+4] = omega;
			spotPos[spotNr*7+5] = hkls[hklnr*5+4]; // ringNr
			spotPos[spotNr*7+6] = nrhkls;
			// Depending on comparisonType:
			// 1. Gvector:
			switch (comparisonType){
			case 1:
				xs = Lsd;
				ys = yspot;
				zs = zspot;
				lenK =CalcNorm3(xs,ys,zs);
				SpotToGv(xs/lenK,ys/lenK,zs/lenK,omega,th,&g1,&g2,&g3);
				spotPos[spotNr*7+0] = g1;
				spotPos[spotNr*7+1] = g2;
				spotPos[spotNr*7+2] = g3;
				break;
			// 2. 2theta, eta, omega
			case 2:
				spotPos[spotNr*7+0] = 2*th;
				spotPos[spotNr*7+1] = etanew;
				spotPos[spotNr*7+2] = omega;
				break;
			// 3. y, z, omega
			case 3:
				spotPos[spotNr*7+0] = yspot;
				spotPos[spotNr*7+1] = zspot;
				spotPos[spotNr*7+2] = etanew;
				break;
			default:
				spotPos[spotNr*7+6] = 0;
				spotNr = -1;
				nrhkls = -1;
				break;
			}
			nrhkls++;
			spotNr++;
		}
	}
	return spotNr;
}

// 3 cases:g1,g2,g3 or 2theta,eta,omega or y,z,omega
// 3 ways to compare spotPositions:
// 1. InternalAngles (maybe most representative??)
	// Convert to angles and length, average and then convert back
// 2. Unitless Angles (2theta, eta, omega) (divided by error)
	// Errors: 2theta: 0.1*px, eta: 0.1*px converted, omega: (OmegaStep/2)*(1+1/sin(eta))
// 3. Unitless Position (y, z, omega) (divided by error)
	// Errors: y: 0.1px, z: 0.1px, omega: (OmegaStep/2)*(1+1/sin(eta))
// thisSpotPos has: 0,1,2 acc to comparisonType, 3 is eta, 4 is Lsd
static inline double calcDiffs(double px, double omegaStep, double thisSpotPos[5],double obsSpotPos[3],int comparisonType){
	int i;
	double diffs, normParams[3], diff[3];
	switch (comparisonType)
	{
		case 1:
			break;
		case 2:
			normParams[0] = atand(0.1*px*cosd(thisSpotPos[0])/thisSpotPos[4]);
			normParams[1] = atand(0.1*px/(thisSpotPos[4]*tand(thisSpotPos[0])));
			normParams[2] = omegaStep*0.5*(1+1/sind(thisSpotPos[3]));
			for (i=0;i<3;i++) diff[i] = (thisSpotPos[i]-obsSpotPos[i])/normParams[i];
			break;
		case 3:
			normParams[0] = 0.1*px;
			normParams[1] = 0.1*px;
			normParams[2] = omegaStep*0.5*(1+1/sind(thisSpotPos[3]));
			for (i=0;i<3;i++) diff[i] = (thisSpotPos[i]-obsSpotPos[i])/normParams[i];
			break;
	}
	diffs = CalcNorm3(diff[0],diff[1],diff[2]);
	return diffs;
}

// hkls have h,k,l,ringNr
// obsSpotsInfo arrangement: for each yPosition, there are n spots per ring, already filtered for that grain. Only spots for the grain are supplied.
// obsSpotsInfo contains y,z,eta,omega,g1,g2,g3,2theta,nrhkls,spotID for each observed spot.
// IDsInfo has for each yPosition and ringNr, startingrowNr and endingRowNr for that yPosition and ringNr
// It must have all the rings upto nRings eg, if analysis is not taking ringNrs 1 and 2 into consideration, there must be zeros for those, but not skipped.
// obsSpotPos needs to be extracted from obsSpotsInfo using correct y-position and ringNr and omega.
// Given a list of voxels belonging to a grain - 1 in approx. case, multiple in other cases,
// We will return the weightedDiff.
// If firstPass is 1, we will read AllSpotsInfo, return obsSpotsInfo, weightedDiff will be 0
// AllSpotsInfo is the memory mapped file, has Y, Z, Ome, GrRadius, ID, RingNr, Eta, 2Theta .... (14 columns, 1D array)
// AllIDsInfo has startingRowNr, endingRowNr
static inline double CalcErrorAngles (double omegaStep, double px, int nVoxels, double *voxelList, double voxelLen,
									double beamFWHM, int nBeamPositions, double *beamPositions, double omeTol,
									double *EulLatC, int nhkls, double *hkls,
									double Lsd, double Wavelength, int comparisonType, double *obsSpotsInfo,
									long *IDsInfo, int nRings, int firstPass, double *AllSpotsInfo, long *AllIDsInfo,
									long totalNrSpots, long *nrMatchedSpots){
	long voxelNr, nSpots, i, j, spotNr, positionNr, ringNr;
	double thisPos[3], thisBeamPosition, thisOmega, thisEta, bestAngle, ys, zs, lenK, omeObs, *matchedMat;
	double LatCThis[6], EulersThis[3], thisSpotPos[5], obsSpotPos[3], gObs[3], gSim[3], IA, bestG1,bestG2,bestG3;
	double *spotInfo, yRot, voxelFraction, diff, weightedDiff=0, bestInfo[13], bestHKLNr;
	long startRowNr, endRowNr, maxNrSpots = 0, bestRow, spotsFilled;
	spotInfo = calloc(nhkls*2*7,sizeof(*spotInfo));
	if (firstPass == 1){
		matchedMat = calloc(totalNrSpots*5,sizeof(*matchedMat));
		comparisonType = 1;
	}
	for (voxelNr=0;voxelNr<nVoxels;voxelNr++){
		thisPos[0] = voxelList[voxelNr*2+0];
		thisPos[1] = voxelList[voxelNr*2+1];
		thisPos[2] = 0;
		for (i=0;i<6;i++) LatCThis[i]  = EulLatC[voxelNr*9 + 3 + i];
		for (i=0;i<3;i++) EulersThis[i] = EulLatC[voxelNr*9 + i];
		// Depending on what we want (comparisonType), we can return gVector, 2theta,eta,ome or y,z,ome
		// spotInfo columns acc to comparisonType:
		// 1. g1,		g2,		g3,		eta,	omega,	ringNr,		nrhkls
		// 2. 2theta,	eta,	omega,	eta,	omega,	ringNr,		nrhkls
		// 3. y,		z,		omega,	eta,	omega,	ringNr,		nrhkls
		nSpots = CalcDiffractionSpots(Lsd,Wavelength,thisPos,LatCThis,EulersThis,nhkls,hkls,spotInfo,comparisonType);
		for (spotNr=0;spotNr<nSpots;spotNr++){
			thisOmega = spotInfo[spotNr*7+4];
			thisEta = spotInfo[spotNr*7+3];
			yRot = thisPos[0]*sind(thisOmega) + thisPos[1]*cosd(thisOmega);
			ringNr = (int) spotInfo[spotNr*7+5];
			for (i=0;i<4;i++) thisSpotPos[i] = spotInfo[spotNr*7+i];
			gSim[0] = spotInfo[spotNr*7+0]; // Will only be used for firstPass;
			gSim[1] = spotInfo[spotNr*7+1]; // Will only be used for firstPass;
			gSim[2] = spotInfo[spotNr*7+2]; // Will only be used for firstPass;
			bestHKLNr = spotInfo[spotNr*7+6];
			thisSpotPos[4] = Lsd;
			for (positionNr=0;positionNr<nBeamPositions;positionNr++){
				thisBeamPosition = beamPositions[positionNr];
				voxelFraction = IntensityFraction(voxelLen,thisBeamPosition,beamFWHM,thisPos,thisOmega);
				if (voxelFraction ==0) continue;
				// Find and set obsSpotPos
				if (firstPass == 1){
					startRowNr = AllIDsInfo[(positionNr*nRings+ringNr)*2+0];
					endRowNr = AllIDsInfo[(positionNr*nRings+ringNr)*2+1];
					bestAngle = 1e10;
					for (i=startRowNr;i<=endRowNr;i++){
						if (matchedMat[i*5+0] == 1) continue;
						//Everything in AllSpotsInfo needs to have i-1
						omeObs = AllSpotsInfo[14*(i-1)+2];
						if (fabs(thisOmega-omeObs) < omeTol){
							ys = AllSpotsInfo[14*(i-1)+0];
							zs = AllSpotsInfo[14*(i-1)+1];
							lenK = CalcNorm3(Lsd,ys,zs);
							SpotToGv(Lsd/lenK,ys/lenK,zs/lenK,omeObs,AllSpotsInfo[14*(i-1)+7]/2,&gObs[0],&gObs[1],&gObs[2]);
							IA = fabs(acosd((gSim[0]*gObs[0]+gSim[1]*gObs[1]+gSim[2]*gObs[2])/
									(CalcNorm3(gSim[0],gSim[1],gSim[2])*CalcNorm3(gObs[0],gObs[1],gObs[2]))));
							if (IA < bestAngle) {
								// mark this Spot to be used!!!!
								bestAngle = IA;
								bestG1 = gObs[0];
								bestG2 = gObs[1];
								bestG3 = gObs[2];
								bestRow = i;
							}
						}
					}
					if (bestAngle < 1){ // Spot was found
						matchedMat[bestRow*5+0] = 1;
						matchedMat[bestRow*5+1] = bestG1;
						matchedMat[bestRow*5+2] = bestG2;
						matchedMat[bestRow*5+3] = bestG3;
						matchedMat[bestRow*5+4] = bestHKLNr;
						obsSpotPos[0] = AllSpotsInfo[14*(i-1)+7];
						obsSpotPos[1] = AllSpotsInfo[14*(i-1)+6];
						obsSpotPos[2] = AllSpotsInfo[14*(i-1)+2];
						diff = calcDiffs(px,omegaStep,thisSpotPos,obsSpotPos,2);
						weightedDiff += diff*voxelFraction;
					}
				} else {
					startRowNr = IDsInfo[(positionNr*nRings+ringNr)*2+0];
					endRowNr = IDsInfo[(positionNr*nRings+ringNr)*2+1];
					for (i=startRowNr;i<=endRowNr;i++){
						if ((int)obsSpotsInfo[i*9+8] == (int)bestHKLNr){
							switch (comparisonType)
							{
								case 1:
									obsSpotPos[0] =obsSpotsInfo[i*9+4]; // g1
									obsSpotPos[1] =obsSpotsInfo[i*9+5]; // g2
									obsSpotPos[2] =obsSpotsInfo[i*9+6]; // g3
									break;
								case 2:
									obsSpotPos[0] =obsSpotsInfo[i*9+7]; // 2theta
									obsSpotPos[1] =obsSpotsInfo[i*9+2]; // eta
									obsSpotPos[2] =obsSpotsInfo[i*9+3]; // omega
									break;
								case 3:
									obsSpotPos[0] =obsSpotsInfo[i*9+0]; // y
									obsSpotPos[1] =obsSpotsInfo[i*9+1]; // z
									obsSpotPos[2] =obsSpotsInfo[i*9+3]; // omega
									break;
							}
						}
					}
					diff = calcDiffs(px,omegaStep,thisSpotPos,obsSpotPos,comparisonType);
					weightedDiff += diff*voxelFraction;
				}
			}
		}
	}
	// If firstPass == 1, populate IDsInfo and obsSpotsInfo
	if (firstPass == 1){
		for (i=0;i<totalNrSpots;i++){
			ringNr = AllSpotsInfo[14*i+5];
			if (matchedMat[i*5+0] == 1){
				positionNr = 0;
				for (j=0;j<nBeamPositions;j++) {
					if (i+1 >= AllIDsInfo[(j*nRings+ringNr)*2+0] && i+1 <= AllIDsInfo[(j*nRings+ringNr)*2+1]) positionNr = j;
				}
				if (IDsInfo[(positionNr*nRings+ringNr)*2+0] == 0)
					IDsInfo[(positionNr*nRings+ringNr)*2+0] = maxNrSpots;
				IDsInfo[(positionNr*nRings+ringNr)*2+1] = maxNrSpots;
				obsSpotsInfo[maxNrSpots*9+0] = AllSpotsInfo[14*i+0]; // y
				obsSpotsInfo[maxNrSpots*9+1] = AllSpotsInfo[14*i+0]; // z
				obsSpotsInfo[maxNrSpots*9+2] = AllSpotsInfo[14*i+6]; // eta
				obsSpotsInfo[maxNrSpots*9+3] = AllSpotsInfo[14*i+2]; // omega
				obsSpotsInfo[maxNrSpots*9+4] = matchedMat[i*5+1]; // g1
				obsSpotsInfo[maxNrSpots*9+5] = matchedMat[i*5+2]; // g2
				obsSpotsInfo[maxNrSpots*9+6] = matchedMat[i*5+3]; // g3
				obsSpotsInfo[maxNrSpots*9+7] = AllSpotsInfo[14*i+7]; // 2Theta
				obsSpotsInfo[maxNrSpots*9+8] = matchedMat[i*5+4]; // nrhkls
				maxNrSpots++;
			}
		}
		*nrMatchedSpots = maxNrSpots;
		free(matchedMat);
	}
	free(spotInfo);
	return weightedDiff;
}

// Parameters to be passed by struct:
struct FITTING_PARAMS {
	double omegaStep,
		px,
		voxelLen,
		beamFWHM,
		omeTol,
		Lsd,
		Wavelength;
	double *voxelList,
		*beamPositions,
		*hkls,
		*obsSpotsInfo,
		*AllSpotsInfo;
	int nBeamPositions,
		nhkls,
		ComparisonType,
		nRings,
		FirstPass;
	long *IDsInfo,
		*AllIDsInfo;
	long totalNrSpots;
	long *nrMatchedSpots;
};

// EulerAngles in Degrees!!!!
static double problem_function(
	unsigned n,
	const double *x,
	double *grad,
	void* f_data_trial)
{
	int nVoxels = n / 9; // nVoxels*9 is the total number of parameters to be optimized.
	// x is arranged as EulerAngles, then LatC for each voxel. EulerAngle=x[voxelNr*9+{0,1,2}] and LatC=x[voxelNr*9+{3,4,5,6,7,8}].
	struct FITTING_PARAMS *f_data = (struct FITTING_PARAMS *) f_data_trial;
	double omegaStep = f_data->omegaStep, px = f_data->px, voxelLen = f_data->voxelLen, beamFWHM = f_data->beamFWHM, omeTol = f_data->omeTol;
	double Lsd = f_data->Lsd, Wavelength = f_data->Wavelength;
	double *voxelList = &(f_data->voxelList[0]), *beamPositions = &(f_data->beamPositions[0]), *hkls = &(f_data->hkls[0]);
	double *obsSpotsInfo = &(f_data->obsSpotsInfo[0]), *AllSpotsInfo = &(f_data->AllSpotsInfo[0]);
	int nBeamPositions = f_data->nBeamPositions, nhkls = f_data->nhkls, ComparisonType = f_data->ComparisonType, nRings = f_data->nRings, FirstPass = f_data->FirstPass;
	long totalNrSpots = f_data->totalNrSpots;
	long *IDsInfo = &(f_data->IDsInfo[0]);
	long *AllIDsInfo = &(f_data->AllIDsInfo[0]);
	long *nrMatchedSpots = &(f_data->nrMatchedSpots); // This is only single pointer, not array pointer.
	if (grad){
		int i, j;
		double h = 1e-5, *xNew;
		xNew = malloc(n*sizeof(*xNew));
		for (j=0;j<n;j++) xNew[j] = x[j];
		double eNeg, ePos;
		for (i=0;i<n;i++){
			xNew[i] -= h;
			eNeg = CalcErrorAngles (omegaStep, px, nVoxels, voxelList, voxelLen, beamFWHM, nBeamPositions, beamPositions, omeTol,
						xNew, nhkls, hkls, Lsd, Wavelength, ComparisonType, obsSpotsInfo, IDsInfo, nRings, FirstPass, AllSpotsInfo, AllIDsInfo,
						totalNrSpots, nrMatchedSpots);
			xNew[i] += 2*h;
			ePos = CalcErrorAngles (omegaStep, px, nVoxels, voxelList, voxelLen, beamFWHM, nBeamPositions, beamPositions, omeTol,
						xNew, nhkls, hkls, Lsd, Wavelength, ComparisonType, obsSpotsInfo, IDsInfo, nRings, FirstPass, AllSpotsInfo, AllIDsInfo,
						totalNrSpots, nrMatchedSpots);
			xNew[i] = x[i];
			grad[i] = (ePos-eNeg)/(2*h);
		}
	}
	double error = CalcErrorAngles (omegaStep, px, nVoxels, voxelList, voxelLen, beamFWHM, nBeamPositions, beamPositions, omeTol,
		x, nhkls, hkls, Lsd, Wavelength, ComparisonType, obsSpotsInfo, IDsInfo, nRings, FirstPass, AllSpotsInfo, AllIDsInfo,
		totalNrSpots, nrMatchedSpots);
	return error;
}

// We assume scans centered around 0, 163 scans with 7 microns mean
int main (int argc, char *argv[]){

	// Read omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength, nScans from PARAM file.
	char *paramFN;
	paramFN = argv[1];
	FILE *fileParam;
	fileParam = fopen(paramFN,"r");
	double omegaStep, px, voxelLen, beamFWHM, omeTol, Lsd, Wavelength;
	int nScans, rings[500], nRings=0;
	char aline[4096], dummy[4096];
	while(fgets(aline,4096,fileParam)!=NULL){
		if (strncmp(aline,"OmegaStep",strlen("OmegaStep"))==0){
			sscanf(aline,"%s %lf",dummy,&omegaStep);
		}
		if (strncmp(aline,"px",strlen("px"))==0){
			sscanf(aline,"%s %lf",dummy,&px);
		}
		if (strncmp(aline,"VoxelLength",strlen("VoxelLength"))==0){
			sscanf(aline,"%s %lf",dummy,&voxelLen);
		}
		if (strncmp(aline,"BeamFWHM",strlen("BeamFWHM"))==0){
			sscanf(aline,"%s %lf",dummy,&beamFWHM);
		}
		if (strncmp(aline,"OmegaTol",strlen("OmegaTol"))==0){
			sscanf(aline,"%s %lf",dummy,&omeTol);
		}
		if (strncmp(aline,"Lsd",strlen("Lsd"))==0){
			sscanf(aline,"%s %lf",dummy,&Lsd);
		}
		if (strncmp(aline,"Wavelength",strlen("Wavelength"))==0){
			sscanf(aline,"%s %lf",dummy,&Wavelength);
		}
		if (strncmp(aline,"nLayers",strlen("nLayers"))==0){
			sscanf(aline,"%s %d",dummy,&nScans);
		}
		if (strncmp(aline,"RingThresh",strlen("RingThresh"))==0){
			sscanf(aline,"%s %d",dummy,&rings[nRings]);
			nRings++;
		}
	}
	fclose(fileParam);

	// Read beamPositions from positions.csv file.
	int i,j,k;
	char *positionsFN;
	positionsFN = argv[2];
	FILE *positionsFile;
	positionsFile = fopen(positionsFN,"r");
	int nBeamPositions = nScans;
	double *beamPositions;
	beamPositions = calloc(nBeamPositions,sizeof(*beamPositions));
	fgets(aline,4096,positionsFile);
	for (i=0;i<nBeamPositions;i++){
		fgets(aline,4096,positionsFile);
		sscanf(aline,"%lf",&beamPositions[i]);
		beamPositions[i] *= 1000;
	}
	fclose(positionsFile);

	// Read hkls, nhkls, nRings (maxRingNr) from HKL file.
	// hkls has h,k,l,ringNr
	char hklfn[4096];
	sprintf(hklfn,"hkls.csv");
	FILE *hklf;
	hklf = fopen(hklfn,"r");
	double ht,kt,lt,ringT;
	double *hklTs;
	hklTs = calloc(500*4,sizeof(*hklTs));
	int nhkls = 0;
	while (fgets(aline,4096,hklf)!=NULL){
		sscanf(aline,"%lf %lf %lf %s %lf",&ht,&kt,&lt,dummy,&ringT);
		for (i=0;i<nRings;i++){
			if ((int)ringT == rings[i]){
				hklTs[nhkls*4+0] = ht;
				hklTs[nhkls*4+1] = kt;
				hklTs[nhkls*4+2] = lt;
				hklTs[nhkls*4+3] = ringT;
				nhkls++;
			}
		}
	}
	fclose(hklf);
	double *hkls;
	hkls = calloc(nhkls*4,sizeof(*hkls));
	for (i=0;i<nhkls*4;i++) hkls[i] = hklTs[i];
	nRings = (int)hkls[nhkls*4-1]; // Highest ring number
	free(hklTs);

	// Read voxelList from VOXELS file.
	char *voxelsFN;
	voxelsFN = argv[3];
	FILE *voxelsFile;
	voxelsFile = fopen(voxelsFN,"r");
	double *voxelsT;
	voxelsT = calloc(nBeamPositions*nBeamPositions,sizeof(*voxelsT));
	int nVoxels;
	while (fgets(aline,4096,voxelsFile)!=NULL){
		sscanf("%lf,%lf",&voxelsT[nVoxels*2+0],&voxelsT[nVoxels*2+1]);
		nVoxels++;
	}
	double *voxelList;
	voxelList = calloc(nVoxels*2,sizeof(*voxelList));
	for (i=0;i<nVoxels*2;i++) voxelList[i] = voxelsT[i];
	free(voxelsT);

	// Read AllSpotsInfo from ExtraInfo.bin
	const char *filename = "/dev/shm/ExtraInfo.bin";
	int rc;
	double *AllSpotsInfo;
	struct stat s;
	size_t size;
	int fd = open(filename,O_RDONLY);
	check(fd < 0, "open %s failed: %s", filename, strerror(errno));
	int status = fstat (fd , &s);
	check (status < 0, "stat %s failed: %s", filename, strerror(errno));
	size = s.st_size;
	AllSpotsInfo = mmap(0,size,PROT_READ,MAP_SHARED,fd,0);
	check (AllSpotsInfo == MAP_FAILED,"mmap %s failed: %s", filename, strerror(errno));
	long totalNrSpots = size/(14*sizeof(double));

	// AllIDsInfo is to be filled
	long *AllIDsInfo;
	AllIDsInfo = calloc(nBeamPositions*nRings,sizeof(*AllIDsInfo));
	FILE *idsfile;
	char *idsfn;
	idsfn = argv[4];
	idsfile = fopen(idsfn,"r");
	int positionNr, startNr, endNr, ringNr;
	while(fgets(aline,4096,idsfile)!=NULL){
		sscanf(aline,"%d %d %d %d",&positionNr,&ringNr,&startNr,&endNr);
		AllIDsInfo[((positionNr-1)*nRings+ringNr)*2+0] =startNr;
		AllIDsInfo[((positionNr-1)*nRings+ringNr)*2+1] =endNr;
	}
	fclose(idsfile);

	// GrainNr
	int GrainNr = atoi(argv[5]);
	char grainFN[4096];
	sprintf(grainFN,"Grains.csv");
	FILE *grainsFile;
	grainsFile = fopen(grainFN,"r");
	char line[20000];
	for (i=0;i<(9+GrainNr);i++) fgets(line,20000,grainsFile);
	double OM[3][3],LatC[6];
	sscanf("%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %lf %lf %lf %lf %lf %lf",dummy,
		&OM[0][0],&OM[0][1],&OM[0][2],&OM[1][0],&OM[1][1],&OM[1][2],&OM[2][0],&OM[2][1],&OM[2][2],
		&LatC[0],&LatC[1],&LatC[2],&LatC[3],&LatC[4],&LatC[5]);
	double Eul[3];
	OrientMat2Euler(OM,Eul);

	// Allocate obsSpotsInfo and IDsInfo and then provide them to problem_function with undefined grad!
	// obsSpotsInfo needs y,z,eta,omega,g1,g2,g3,2theta,nrhkls for each observed spot for all voxels of the grain.
	// IDsInfo has for each yPosition and ringNr, startingrowNr and endingRowNr for that yPosition and ringNr.
	double *obsSpotsInfo;
	obsSpotsInfo = calloc(totalNrSpots*9,sizeof(*obsSpotsInfo));
	long *IDsInfo;
	IDsInfo = calloc(nBeamPositions*nRings,sizeof(*IDsInfo));

	// Make struct
	long nrMatchedSpots;
	struct FITTING_PARAMS f_data;
	f_data.Wavelength = Wavelength;
	f_data.nBeamPositions = nBeamPositions;
	f_data.AllIDsInfo = &AllIDsInfo[0];
	f_data.AllSpotsInfo = &AllSpotsInfo[0];
	f_data.ComparisonType = 1;
	f_data.FirstPass = 1;
	f_data.IDsInfo = &IDsInfo[0];
	f_data.Lsd = Lsd;
	f_data.beamFWHM = beamFWHM;
	f_data.beamPositions = &beamPositions[0];
	f_data.nRings = nRings;
	f_data.nhkls = nhkls;
	f_data.nrMatchedSpots = &nrMatchedSpots;
	f_data.obsSpotsInfo = &obsSpotsInfo[0];
	f_data.omeTol = omeTol;
	f_data.omegaStep = omegaStep;
	f_data.px = px;
	f_data.totalNrSpots = totalNrSpots;
	f_data.voxelLen = voxelLen;
	f_data.voxelList = &voxelList[0];
	struct FITTING_PARAMS *f_datat;
	f_datat = &f_data;
	void* trp = (struct FITTING_PARAMS *) f_datat;

	// Call problem_function with f_data.
	int n = nVoxels * 9;
	double *x;
	double *xl;
	double *xu;
	double EulTol = 2; // Degrees
	double ABCTol = 2; // %
	double ABGTol = 2; // %
	x = calloc(n,sizeof(*x));
	xl = calloc(n,sizeof(*xl));
	xu = calloc(n,sizeof(*xu));
	for (i=0;i<nVoxels;i++){
		for (j=0;j<3;j++) x[i*9+j] = Eul[j];
		for (j=0;j<3;j++) xl[i*9+j] = Eul[j] - EulTol;
		for (j=0;j<3;j++) xu[i*9+j] = Eul[j] + EulTol;
		for (j=0;j<6;j++) x[i*9+3+j] = LatC[j];
		for (j=0;j<3;j++) xl[i*9+3+j] = LatC[j]*(100-ABCTol)/100;
		for (j=3;j<6;j++) xl[i*9+3+j] = LatC[j]*(100-ABGTol)/100;
		for (j=0;j<3;j++) xu[i*9+3+j] = LatC[j]*(100*ABCTol)/100;
		for (j=3;j<6;j++) xu[i*9+3+j] = LatC[j]*(100+ABGTol)/100;
	}
	double error = problem_function(n,x,NULL,f_datat);
	obsSpotsInfo = realloc(obsSpotsInfo,nrMatchedSpots*9*sizeof(*obsSpotsInfo));

	// Now we call the fitting function.
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LD_MMA, n);
	nlopt_set_lower_bounds(opt, xl);
	nlopt_set_upper_bounds(opt, xu);
	nlopt_set_min_objective(opt, problem_function, trp);
	double minf;
	nlopt_optimize(opt, x, &minf);
	nlopt_destroy(opt);
}
