// Copyright (c) 2014, UChicago Argonne, LLC. See LICENSE file.
// FitPosOrStrainsGPU.cu — GPU-accelerated FF-HEDM grain refinement.
// Ports FitPosOrStrainsOMP.c: 4-stage Nelder-Mead simplex fitting
// (position, orientation, strain, position) with dynamic spot reassignment.

#include <cuda_runtime.h>
#define RealType double
#include "midas_gpu_math.cuh"
#include "../../utils/gpu_simplex.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <libgen.h>
#include <time.h>
#include <errno.h>

extern "C" {
#include "MIDAS_ParamParser.h"
}

#include "midas_version.h"
#include "MIDAS_Limits.h"
#define MAXNOMEGARANGES MAX_N_OMEGA_RANGES
#define MaxNHKLS 5000
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); exit(1); } } while(0)

// ═══════════════════════════════════════════
//  Device helper functions
// ═══════════════════════════════════════════

__device__ static inline double gpu_CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y*y + z*z));
  if (y > 0) alpha = -alpha;
  return alpha;
}

__device__ static inline void gpu_Euler2OrientMat(const double euler[3], double m[3][3]) {
  double psi=euler[0]*deg2rad, phi=euler[1]*deg2rad, theta=euler[2]*deg2rad;
  double cps=cos(psi),sps=sin(psi),cph=cos(phi),sph=sin(phi),cth=cos(theta),sth=sin(theta);
  m[0][0]=cth*cps-sth*cph*sps; m[0][1]=-cth*cph*sps-sth*cps; m[0][2]=sph*sps;
  m[1][0]=cth*sps+sth*cph*cps; m[1][1]=cth*cph*cps-sth*sps;  m[1][2]=-sph*cps;
  m[2][0]=sth*sph;             m[2][1]=cth*sph;               m[2][2]=cph;
}

__device__ static inline void gpu_OrientMat2Euler(double m[3][3], double E[3]) {
  double phi,psi,theta,sph;
  if (fabs(m[2][2]-1.0)<1e-12) phi=0; else phi=acos(m[2][2]);
  sph=sin(phi);
  if (fabs(sph)<1e-12) {
    psi=0.0;
    double s=m[1][0],c=m[0][0];
    theta=(fabs(m[2][2]-1.0)<1e-12)? ((s>=0)?acos(c):2*M_PI-acos(c))
                                    : (((-m[1][0])>=0)?acos(m[0][0]):2*M_PI-acos(m[0][0]));
  } else {
    double r1=fabs(-m[1][2]/sph)<=1.0?-m[1][2]/sph:1.0;
    psi=(m[0][2]/sph>=0)?acos(r1):2*M_PI-acos(r1);
    double r2=fabs(m[2][1]/sph)<=1.0?m[2][1]/sph:1.0;
    theta=(m[2][0]/sph>=0)?acos(r2):2*M_PI-acos(r2);
  }
  E[0]=rad2deg*psi; E[1]=rad2deg*phi; E[2]=rad2deg*theta;
}

__device__ static int gpu_CalcOmega(double gx, double gy, double gz, double theta,
    double omegas[4], double etas[4]) {
  int nsol=0;
  double len=sqrt(gx*gx+gy*gy+gz*gz), v=sin(theta*deg2rad)*len;
  if (fabs(gy)<1e-4) {
    if (gx!=0) { double c=-v/gx; if(fabs(c)<=1){double o=acos(c)*rad2deg;omegas[nsol++]=o;omegas[nsol++]=-o;}}
  } else {
    double y2=gy*gy,a=1+(gx*gx)/y2,b=(2*v*gx)/y2,c=(v*v)/y2-1,D=b*b-4*a*c;
    if(D>=0){
      double sd=sqrt(D);
      double c1=(-b+sd)/(2*a);
      if(fabs(c1)<=1){double o1a=acos(c1),o1b=-o1a;
        double ea=-gx*cos(o1a)+gy*sin(o1a),eb=-gx*cos(o1b)+gy*sin(o1b);
        omegas[nsol++]=(fabs(ea-v)<fabs(eb-v))?o1a*rad2deg:o1b*rad2deg;}
      double c2=(-b-sd)/(2*a);
      if(fabs(c2)<=1){double o2a=acos(c2),o2b=-o2a;
        double ea=-gx*cos(o2a)+gy*sin(o2a),eb=-gx*cos(o2b)+gy*sin(o2b);
        omegas[nsol++]=(fabs(ea-v)<fabs(eb-v))?o2a*rad2deg:o2b*rad2deg;}
    }
  }
  for(int i=0;i<nsol;i++){
    double oR=omegas[i]*deg2rad,cO=cos(oR),sO=sin(oR);
    double gw2=gx*sO+gy*cO,gw3=gz;
    double r=sqrt(gw2*gw2+gw3*gw3);
    double eta=(r>1e-10)?rad2deg*acos(gw3/r):0;
    if(gw2>0)eta=-eta; etas[i]=eta;
  }
  return nsol;
}

__device__ static void gpu_DisplacementInTheSpot(double a,double b,double c,
    double xi,double yi,double zi,double omega,double wedge,double chi,
    double *Dy,double *Dz) {
  double sO=sin(omega*deg2rad),cO=cos(omega*deg2rad);
  double XN=a*cO-b*sO,YN=a*sO+b*cO,ZN=c;
  double wR=wedge*deg2rad,cW=cos(wR),sW=sin(wR);
  double XW=XN*cW-ZN*sW,YW=YN,ZW=XN*sW+ZN*cW;
  double cR=chi*deg2rad,cC=cos(cR),sC=sin(cR);
  double XC=XW,YC=cC*YW-sC*ZW,ZC=sC*YW+cC*ZW;
  double ik0=xi-XC,ik1=yi-YC,ik2=zi-ZC;
  double nk=sqrt(ik0*ik0+ik1*ik1+ik2*ik2);
  ik0/=nk;ik1/=nk;ik2/=nk;
  if(fabs(ik0)>1e-12){*Dy=YC-(XC*ik1/ik0);*Dz=ZC-(XC*ik2/ik0);}
}

// CorrectForOme: correct spot position for omega/wedge (120 lines in CPU)
__device__ static void gpu_CorrectForOme(double yc,double zc,double Lsd,
    double OmIni,double wl,double wedge,double *ysO,double *zsO,double *OmeO){
  double CosOme=cos(deg2rad*OmIni),SinOme=sin(deg2rad*OmIni);
  double eta=gpu_CalcEtaAngle(yc,zc);
  double RR=sqrt(yc*yc+zc*zc),tth=rad2deg*atan(RR/Lsd),th=tth/2;
  double SinTh=sin(deg2rad*th),CosTh=cos(deg2rad*th),ds=2*SinTh/wl;
  double CosW=cos(deg2rad*wedge),SinW=sin(deg2rad*wedge);
  double SinE=sin(deg2rad*eta),CosE=cos(deg2rad*eta);
  double k1=-ds*SinTh,k2=-ds*CosTh*SinE,k3=ds*CosTh*CosE;
  if(eta==90){k3=0;k2=-CosTh;}else if(eta==-90){k3=0;k2=CosTh;}
  double k1f=k1*CosW+k3*SinW,k2f=k2,k3f=k3*CosW-k1*SinW;
  double G1a=k1f*CosOme+k2f*SinOme,G2a=k2f*CosOme-k1f*SinOme,G3a=k3f;
  double LG=sqrt(G1a*G1a+G2a*G2a+G3a*G3a);
  double g1=G1a*ds/LG,g2=G2a*ds/LG,g3=G3a*ds/LG;
  SinW=0;CosW=1;
  double LenG=sqrt(g1*g1+g2*g2+g3*g3);
  double k1i=-(LenG*LenG*wl)/2;
  tth=2*rad2deg*asin(wl*LenG/2);
  RR=Lsd*tan(deg2rad*tth);
  double A=(k1i+g3*SinW)/CosW;
  double aS=g1*g1+g2*g2,bS=2*A*g2,cS=A*A-g1*g1;
  double aC=aS,bC=-2*A*g1,cC=A*A-g2*g2;
  double PS=bS*bS-4*aS*cS,PC=bC*bC-4*aC*cC;
  double pS=(PS>=0)?sqrt(PS):0,pC=(PC>=0)?sqrt(PC):0;
  int pchkS=(PS<0),pchkC=(PC<0);
  double SO1=(-bS-pS)/(2*aS),SO2=(-bS+pS)/(2*aS);
  double CO1=(-bC-pC)/(2*aC),CO2=(-bC+pC)/(2*aC);
  if(SO1<-1)SO1=0;else if(SO1>1)SO1=0;else if(SO2<-1)SO2=0;else if(SO2>1)SO2=0;
  if(CO1<-1)CO1=0;else if(CO1>1)CO1=0;else if(CO2<-1)CO2=0;else if(CO2>1)CO2=0;
  if(pchkS){SO1=0;SO2=0;} if(pchkC){CO1=0;CO2=0;}
  double Op1=fabs(SO1*SO1+CO1*CO1-1),Op2=fabs(SO1*SO1+CO2*CO2-1);
  double Om1,Om2;
  if(Op1<Op2){Om1=rad2deg*atan2(SO1,CO1);Om2=rad2deg*atan2(SO2,CO2);}
  else{Om1=rad2deg*atan2(SO1,CO2);Om2=rad2deg*atan2(SO2,CO1);}
  double Omega=(fabs(Om1-OmIni)<fabs(Om2-OmIni))?Om1:Om2;
  double Eta2=gpu_CalcEtaAngle(k2,k3);
  *ysO=-RR*sin(deg2rad*Eta2); *zsO=RR*cos(deg2rad*Eta2); *OmeO=Omega;
}

__device__ static void gpu_SpotToGv(double xi,double yi,double zi,double Omega,
    double theta,double *g1,double *g2,double *g3){
  double cO=cos(Omega*deg2rad),sO=sin(Omega*deg2rad);
  double eta=gpu_CalcEtaAngle(yi,zi),tE=tan(-eta*deg2rad),sTh=sin(theta*deg2rad);
  double cTh=cos(theta*deg2rad);
  double k3=sTh*(1+xi)/(yi*tE+zi),k2=tE*k3,k1=-sTh;
  if(eta==90){k3=0;k2=-cTh;}else if(eta==-90){k3=0;k2=cTh;}
  double k1f=k1,k3f=k3,k2f=k2; // wedge=0
  *g1=k1f*cO+k2f*sO; *g2=k2f*cO-k1f*sO; *g3=k3f;
}

__device__ static void gpu_CorrectHKLsLatC(const double LatC[6],const double *hklsRaw,
    int nhkls,double Lsd,double wl,double *hklsOut){
  double a=LatC[0],b=LatC[1],c=LatC[2],al=LatC[3],be=LatC[4],ga=LatC[5];
  double SA=sin(al*deg2rad),SB=sin(be*deg2rad),SG=sin(ga*deg2rad);
  double CA=cos(al*deg2rad),CB=cos(be*deg2rad),CG=cos(ga*deg2rad);
  double GP=acos((CA*CB-CG)/(SA*SB))*rad2deg,BP=acos((CG*CA-CB)/(SG*SA))*rad2deg;
  double SBP=sin(BP*deg2rad);
  double V=a*b*c*SA*SBP*SG;
  double AP=b*c*SA/V,BPr=c*a*SB/V,CP=a*b*SG/V;
  double B00=AP,B01=BPr*cos(GP*deg2rad),B02=CP*cos(BP*deg2rad);
  double B11=BPr*sin(GP*deg2rad),B12=-CP*SBP*CA,B22=CP*SBP*SA;
  for(int i=0;i<nhkls;i++){
    double h=hklsRaw[i*7+0],k=hklsRaw[i*7+1],l=hklsRaw[i*7+2];
    double g0=B00*h+B01*k+B02*l,g1=B11*k+B12*l,g2=B22*l;
    double Ds=1.0/sqrt(g0*g0+g1*g1+g2*g2);
    double Th=asin(wl/(2*Ds))*rad2deg;
    double Rad=Lsd*tan(2*Th*deg2rad);
    hklsOut[i*7+0]=g0; hklsOut[i*7+1]=g1; hklsOut[i*7+2]=g2;
    hklsOut[i*7+3]=Ds; hklsOut[i*7+4]=Th; hklsOut[i*7+5]=Rad;
    hklsOut[i*7+6]=hklsRaw[i*7+6];
  }
}

// CalcDiffrSpots: generate theoretical spots from orientation+HKLs
// Output: spots[nSpots*9] = {yl,zl,omega,gx,gy,gz,distance,ringNr,spotNr}
__device__ static int gpu_CalcDiffrSpots(double orient[3][3],double distance,
    const double *omeRanges,int nOmeRanges,const double *boxSizes,
    const double *hkls,int nhkls,double excludePole,double *spots){
  int spotnr=0;
  for(int ih=0;ih<nhkls;ih++){
    double Ghkl[3]={hkls[ih*7+0],hkls[ih*7+1],hkls[ih*7+2]};
    double Gc[3];
    Gc[0]=orient[0][0]*Ghkl[0]+orient[0][1]*Ghkl[1]+orient[0][2]*Ghkl[2];
    Gc[1]=orient[1][0]*Ghkl[0]+orient[1][1]*Ghkl[1]+orient[1][2]*Ghkl[2];
    Gc[2]=orient[2][0]*Ghkl[0]+orient[2][1]*Ghkl[1]+orient[2][2]*Ghkl[2];
    double theta=hkls[ih*7+4],RR=hkls[ih*7+5],RingNr=hkls[ih*7+6];
    double Ds=hkls[ih*7+3],NGc=sqrt(Gc[0]*Gc[0]+Gc[1]*Gc[1]+Gc[2]*Gc[2]);
    double GCr[3]={Ds*Gc[0]/NGc,Ds*Gc[1]/NGc,Ds*Gc[2]/NGc};
    double nrhkls=(double)ih*2+1;
    double omegas[4],etas[4];
    int nsol=gpu_CalcOmega(Gc[0],Gc[1],Gc[2],theta,omegas,etas);
    for(int i=0;i<nsol;i++){
      double Om=omegas[i],Eta=etas[i];
      if(isnan(Om)||isnan(Eta))continue;
      double EA=fabs(Eta);
      if(EA<excludePole||(180-EA)<excludePole)continue;
      double yl=-sin(Eta*deg2rad)*RR,zl=cos(Eta*deg2rad)*RR;
      int keep=0;
      for(int r=0;r<nOmeRanges;r++){
        if(Om>omeRanges[r*2]&&Om<omeRanges[r*2+1]&&
           yl>boxSizes[r*4]&&yl<boxSizes[r*4+1]&&
           zl>boxSizes[r*4+2]&&zl<boxSizes[r*4+3]){keep=1;break;}
      }
      if(keep){
        int b=spotnr*9;
        spots[b+0]=yl;spots[b+1]=zl;spots[b+2]=Om;
        spots[b+3]=GCr[0];spots[b+4]=GCr[1];spots[b+5]=GCr[2];
        spots[b+6]=distance;spots[b+7]=RingNr;spots[b+8]=nrhkls;
        nrhkls+=1; spotnr++;
      }
    }
  }
  return spotnr;
}

// ═══════════════════════════════════════════
//  GPU Params (constant memory)
// ═══════════════════════════════════════════
struct FitGPUParams {
  double Lsd, Wavelength, MinEta, wedge, chi;
  double OmegaRanges[MAXNOMEGARANGES*2];
  double BoxSizes[MAXNOMEGARANGES*4];
  int nOmeRanges;
  int nhkls;
  // Bin structure info for reassignment
  double EtaBinSize, OmeBinSize;
  int nRingBins, nEtaBins, nOmeBins, nSpotsBin;
};
__device__ FitGPUParams d_params;

// ═══════════════════════════════════════════
//  Objective function: shared evaluation core
// ═══════════════════════════════════════════

// Per-grain scratch layout (flat array):
//  [0 .. nhkls*7-1]           : corrected HKLs
//  [nhkls*7 .. nhkls*7+nMaxTheor*9-1] : theoretical spots
//  [nhkls*7+nMaxTheor*9 .. ]  : additional scratch

// Evaluate error for trial params x[12]={pos3,euler3,latc6}
// against observed spots. Returns sum of position or angle errors.
__device__ static double gpu_FitErrorsPosT(
    const double *x, // [12] = pos(3)+euler(3)+latc(6)
    const double *spotsYZO, int nSpots, // [nSpots*11]
    const double *hklsRaw, double *scratch, int nMaxTheor) {
  double *hkls = scratch;
  double *theorSpots = scratch + d_params.nhkls*7;
  // 1. Correct HKLs for trial lattice params
  gpu_CorrectHKLsLatC(&x[6], hklsRaw, d_params.nhkls, d_params.Lsd,
                       d_params.Wavelength, hkls);
  // 2. Euler → Orient matrix
  double orient[3][3];
  gpu_Euler2OrientMat(&x[3], orient);
  // 3. Generate theoretical spots
  int nTspots = gpu_CalcDiffrSpots(orient, d_params.Lsd,
      d_params.OmegaRanges, d_params.nOmeRanges, d_params.BoxSizes,
      hkls, d_params.nhkls, d_params.MinEta, theorSpots);
  // 4. For each observed spot: correct for grain pos, match against theor
  double Error = 0;
  for (int sp = 0; sp < nSpots; sp++) {
    const double *s = &spotsYZO[sp*11];
    double DisplY, DisplZ, ys, zs, Omega;
    gpu_DisplacementInTheSpot(x[0],x[1],x[2], d_params.Lsd,
        s[5],s[6],s[4], d_params.wedge,d_params.chi, &DisplY,&DisplZ);
    double yt=s[5]-DisplY, zt=s[6]-DisplZ;
    gpu_CorrectForOme(yt,zt,d_params.Lsd,s[4],d_params.Wavelength,
        d_params.wedge,&ys,&zs,&Omega);
    // Match against theoretical spots by sequence number
    int spnr = (int)s[8]; // theoretical spot sequence number
    for (int k = 0; k < nTspots; k++) {
      if ((int)theorSpots[k*9+8] == spnr) {
        double dy = ys - theorSpots[k*9+0];
        double dz = zs - theorSpots[k*9+1];
        Error += sqrt(dy*dy + dz*dz);
        break;
      }
    }
  }
  return Error;
}

// Orient+Strain objective (pos fixed)
__device__ static double gpu_FitErrorsOrientStrains(
    const double *x, // [9] = euler(3)+latc(6)
    const double *spotsYZO, int nSpots,
    const double *hklsRaw, double *scratch, int nMaxTheor,
    const double Pos[3]) {
  double *hkls = scratch;
  double *theorSpots = scratch + d_params.nhkls*7;
  gpu_CorrectHKLsLatC(&x[3], hklsRaw, d_params.nhkls, d_params.Lsd,
                       d_params.Wavelength, hkls);
  double orient[3][3];
  gpu_Euler2OrientMat(x, orient);
  int nTspots = gpu_CalcDiffrSpots(orient, d_params.Lsd,
      d_params.OmegaRanges, d_params.nOmeRanges, d_params.BoxSizes,
      hkls, d_params.nhkls, d_params.MinEta, theorSpots);
  double Error = 0;
  for (int sp = 0; sp < nSpots; sp++) {
    const double *s = &spotsYZO[sp*11];
    double DisplY,DisplZ,ys,zs,Omega;
    gpu_DisplacementInTheSpot(Pos[0],Pos[1],Pos[2], d_params.Lsd,
        s[5],s[6],s[4], d_params.wedge,d_params.chi, &DisplY,&DisplZ);
    double yt=s[5]-DisplY,zt=s[6]-DisplZ;
    gpu_CorrectForOme(yt,zt,d_params.Lsd,s[4],d_params.Wavelength,
        d_params.wedge,&ys,&zs,&Omega);
    double lenK=sqrt(d_params.Lsd*d_params.Lsd+ys*ys+zs*zs);
    double Radius=sqrt(ys*ys+zs*zs);
    double Theta=0.5*atan(Radius/d_params.Lsd)*rad2deg;
    double g1,g2,g3;
    gpu_SpotToGv(d_params.Lsd/lenK,ys/lenK,zs/lenK,Omega,Theta,&g1,&g2,&g3);
    double NormGObs=sqrt(g1*g1+g2*g2+g3*g3);
    int spnr=(int)s[8];
    for(int k=0;k<nTspots;k++){
      if((int)theorSpots[k*9+8]!=spnr) continue;
      double gt0=theorSpots[k*9+3],gt1=theorSpots[k*9+4],gt2=theorSpots[k*9+5];
      double NGt=sqrt(gt0*gt0+gt1*gt1+gt2*gt2);
      double dot=g1*gt0+g2*gt1+g3*gt2;
      double ratio=dot/(NormGObs*NGt);
      if(ratio>1)ratio=1;if(ratio<-1)ratio=-1;
      if(ratio>=0.9975640502598242){
        double angle=fabs(acos(ratio)*rad2deg);
        if(angle<4) Error+=angle;
      }
      break;
    }
  }
  return Error;
}

// Strain-only objective (pos+orient fixed)
__device__ static double gpu_FitErrorsStrains(
    const double *x, // [6] = latc(6)
    const double *spotsYZO, int nSpots,
    const double *hklsRaw, double *scratch, int nMaxTheor,
    const double Pos[3], const double Orient[3]) {
  double *hkls = scratch;
  double *theorSpots = scratch + d_params.nhkls*7;
  gpu_CorrectHKLsLatC(x, hklsRaw, d_params.nhkls, d_params.Lsd,
                       d_params.Wavelength, hkls);
  double orient[3][3];
  gpu_Euler2OrientMat(Orient, orient);
  int nTspots = gpu_CalcDiffrSpots(orient, d_params.Lsd,
      d_params.OmegaRanges, d_params.nOmeRanges, d_params.BoxSizes,
      hkls, d_params.nhkls, d_params.MinEta, theorSpots);
  // Position-based error (same as FitErrorsStrains in CPU)
  double Error = 0;
  for (int sp = 0; sp < nSpots; sp++) {
    const double *s = &spotsYZO[sp*11];
    double DisplY,DisplZ,ys,zs,Omega;
    gpu_DisplacementInTheSpot(Pos[0],Pos[1],Pos[2], d_params.Lsd,
        s[5],s[6],s[4], d_params.wedge,d_params.chi, &DisplY,&DisplZ);
    double yt=s[5]-DisplY,zt=s[6]-DisplZ;
    gpu_CorrectForOme(yt,zt,d_params.Lsd,s[4],d_params.Wavelength,
        d_params.wedge,&ys,&zs,&Omega);
    int spnr=(int)s[8];
    for(int k=0;k<nTspots;k++){
      if((int)theorSpots[k*9+8]==spnr){
        double dy=ys-theorSpots[k*9+0],dz=zs-theorSpots[k*9+1];
        Error+=sqrt(dy*dy+dz*dz); break;
      }
    }
  }
  return Error;
}

// ═══════════════════════════════════════════
//  Dynamic Spot Reassignment (device-side)
// ═══════════════════════════════════════════

// Reassign observed spots to theoretical spots using bin structure.
// Returns new nSpots (replaces spot data in-place).
__device__ static int gpu_ReassignSpotsFromBins(
    const double *x12,       // [12] current pos+euler+latc
    const double *hklsRaw, double *scratch,
    double *spotsYZO,        // [maxSpots*11] — will be REWRITTEN
    int maxSpots,
    const double *AllSpotsPtr, int totalNSpots,
    const double *ObsSpotsLab, int nSpotsBin,
    const int *BinData, const int *nBinData,
    int nMaxTheor) {

  double *hkls = scratch;
  double *theorSpots = scratch + d_params.nhkls * 7;
  // 1. Correct HKLs for current lattice
  gpu_CorrectHKLsLatC(&x12[6], hklsRaw, d_params.nhkls, d_params.Lsd,
                       d_params.Wavelength, hkls);
  // 2. Orient matrix from current Euler
  double orient[3][3];
  gpu_Euler2OrientMat(&x12[3], orient);
  // 3. Generate theoretical spots
  int nTspots = gpu_CalcDiffrSpots(orient, d_params.Lsd,
      d_params.OmegaRanges, d_params.nOmeRanges, d_params.BoxSizes,
      hkls, d_params.nhkls, d_params.MinEta, theorSpots);

  // 4. For each theoretical spot, search bin structure for best observed match
  int nMatched = 0;
  int usedSpotIDs[256]; // enough for typical grains
  for (int i = 0; i < 256; i++) usedSpotIDs[i] = -1;

  for (int sp = 0; sp < nTspots && nMatched < maxSpots; sp++) {
    int RingNr = (int)theorSpots[sp*9+7];
    double theorOmega = theorSpots[sp*9+2];
    double theorYl = theorSpots[sp*9+0], theorZl = theorSpots[sp*9+1];
    double theorEta = gpu_CalcEtaAngle(theorYl, theorZl);

    int iRing = RingNr - 1;
    if (iRing < 0 || iRing >= d_params.nRingBins) continue;
    int iEta = (int)floor((180.0 + theorEta) / d_params.EtaBinSize);
    int iOme = (int)floor((180.0 + theorOmega) / d_params.OmeBinSize);
    if (iEta < 0) iEta = 0;
    if (iEta >= d_params.nEtaBins) iEta = d_params.nEtaBins - 1;
    if (iOme < 0) iOme = 0;
    if (iOme >= d_params.nOmeBins) iOme = d_params.nOmeBins - 1;

    long long int Pos = (long long int)iRing * d_params.nEtaBins * d_params.nOmeBins
                        + iEta * d_params.nOmeBins + iOme;
    int nInBin = nBinData[Pos * 2];
    int DataPos = nBinData[Pos * 2 + 1];
    if (nInBin == 0) continue;

    double bestDiffOme = 1e9;
    int bestRow = -1;
    for (int iSpot = 0; iSpot < nInBin; iSpot++) {
      int spotRow = BinData[DataPos + iSpot];
      if (spotRow < 0 || spotRow >= nSpotsBin) continue;
      // Check if already used
      int alreadyUsed = 0;
      for (int u = 0; u < nMatched; u++) {
        if (usedSpotIDs[u] == spotRow) { alreadyUsed = 1; break; }
      }
      if (alreadyUsed) continue;
      double obsOmega = ObsSpotsLab[spotRow * 9 + 2];
      double diffOme = fabs(theorOmega - obsOmega);
      if (diffOme < 5.0 && diffOme < bestDiffOme) {
        bestDiffOme = diffOme;
        bestRow = spotRow;
      }
    }
    if (bestRow >= 0 && nMatched < 256) {
      usedSpotIDs[nMatched] = bestRow;
      size_t spos = (size_t)bestRow;
      if (spos < (size_t)totalNSpots) {
        double *s = &spotsYZO[nMatched * 11];
        s[0]  = AllSpotsPtr[spos*16+0];  // YLab
        s[1]  = AllSpotsPtr[spos*16+1];  // ZLab
        s[2]  = AllSpotsPtr[spos*16+2];  // Omega
        s[3]  = AllSpotsPtr[spos*16+4];  // SpotID
        s[4]  = AllSpotsPtr[spos*16+8];  // OmegaIni
        s[5]  = AllSpotsPtr[spos*16+9];  // YOrig
        s[6]  = AllSpotsPtr[spos*16+10]; // ZOrig
        s[7]  = AllSpotsPtr[spos*16+5];  // RingNr
        s[8]  = theorSpots[sp*9+8];      // nrhkls sequence number
        s[9]  = AllSpotsPtr[spos*16+14]; // maskTouched
        s[10] = AllSpotsPtr[spos*16+15]; // FitRMSE
        nMatched++;
      }
    }
  }
  return nMatched;
}

// ═══════════════════════════════════════════
//  Functors for gpu_simplex
// ═══════════════════════════════════════════

struct FunctorPosOrientStrain { // Stage 1: NDIM=12
  const double *spotsYZO; int nSpots;
  const double *hklsRaw; double *scratch; int nMaxTheor;
  __device__ double operator()(const double *x, int ndim) const {
    return gpu_FitErrorsPosT(x, spotsYZO, nSpots, hklsRaw, scratch, nMaxTheor);
  }
};

struct FunctorOrientStrain { // Stage 2: NDIM=9
  const double *spotsYZO; int nSpots;
  const double *hklsRaw; double *scratch; int nMaxTheor;
  double Pos[3];
  __device__ double operator()(const double *x, int ndim) const {
    return gpu_FitErrorsOrientStrains(x, spotsYZO, nSpots, hklsRaw, scratch,
                                       nMaxTheor, Pos);
  }
};

struct FunctorStrain { // Stage 3: NDIM=6
  const double *spotsYZO; int nSpots;
  const double *hklsRaw; double *scratch; int nMaxTheor;
  double Pos[3], Orient[3];
  __device__ double operator()(const double *x, int ndim) const {
    return gpu_FitErrorsStrains(x, spotsYZO, nSpots, hklsRaw, scratch,
                                 nMaxTheor, Pos, Orient);
  }
};

struct FunctorPos { // Stage 4: NDIM=3
  const double *spotsYZO; int nSpots;
  const double *hklsRaw; double *scratch; int nMaxTheor;
  double Orient[3], Strains[6];
  __device__ double operator()(const double *x, int ndim) const {
    // Build full x12 from x[3](pos) + fixed orient + fixed strains
    double x12[12];
    x12[0]=x[0];x12[1]=x[1];x12[2]=x[2];
    x12[3]=Orient[0];x12[4]=Orient[1];x12[5]=Orient[2];
    for(int i=0;i<6;i++) x12[6+i]=Strains[i];
    return gpu_FitErrorsPosT(x12, spotsYZO, nSpots, hklsRaw, scratch, nMaxTheor);
  }
};

// ═══════════════════════════════════════════
//  Main grain-fitting kernel
// ═══════════════════════════════════════════
__global__ void fitGrainsKernel(
    int nGrains,
    const double *d_initData,    // [nGrains*15] from IndexBest.bin
    double *d_spotData,             // [nGrains*MaxNHKLS*11] observed spots per grain (modified in-place)
    int *d_nSpotsPerGrain,       // [nGrains] number of spots per grain (modified by reassign)
    const double *d_hklsRaw,     // [nhkls*7] raw HKL data (shared)
    double *d_scratch,           // [nGrains*scratchSize]
    const double *d_LatCin,      // [6] nominal lattice params
    double d_Rsample, double d_Hbeam,
    double d_MargPos, double d_MargOme, double d_MargABC, double d_MargABG,
    double *d_results,           // [nGrains*27] output
    int scratchPerGrain, int nMaxTheor,
    // Bin data for dynamic reassignment
    const double *d_AllSpots, int d_totalNSpots,
    const double *d_ObsSpotsLab, int d_nSpotsBin,
    const int *d_BinData, const int *d_nBinData,
    int doDynReassign)
{
  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (gIdx >= nGrains) return;

  // Load initial data from IndexBest.bin
  const double *init = &d_initData[gIdx * 15];
  double Orient0[9], Pos0[3], Euler0[3];
  for (int i=0;i<9;i++) Orient0[i] = init[i+1];
  for (int i=0;i<3;i++) Pos0[i] = init[i+10];
  double NrExp = init[13], NrObs = init[14];
  if (NrObs == 0) return;

  // Orient matrix → Euler
  double O3[3][3];
  for(int i=0;i<3;i++)for(int j=0;j<3;j++) O3[i][j]=Orient0[i*3+j];
  gpu_OrientMat2Euler(O3, Euler0);
  gpu_Euler2OrientMat(Euler0, O3);

  // Pointers
  int nSpots = d_nSpotsPerGrain[gIdx];
  if (nSpots <= 0) return;
  double *spots = &d_spotData[gIdx * MaxNHKLS * 11];
  double *scratch = &d_scratch[gIdx * scratchPerGrain];

  double LatCin[6];
  for(int i=0;i<6;i++) LatCin[i]=d_LatCin[i];

  // ─── Initialize spot→HKL mapping ───
  // Compute theoretical spots from initial orientation and match against
  // observed spots. Sets spotsYZO cols 4(omega_theor), 5(yl_theor),
  // 6(zl_theor), 8(nrhkls sequence number). This mirrors the CPU's
  // CalcAngleErrors initial assignment.
  {
    double *initHkls = scratch;
    double *initTheorSpots = scratch + d_params.nhkls * 7;
    // Use raw HKLs (no lattice correction for initial matching)
    for(int i=0; i<d_params.nhkls*7; i++) initHkls[i] = d_hklsRaw[i];
    int nInitT = gpu_CalcDiffrSpots(O3, d_params.Lsd,
        d_params.OmegaRanges, d_params.nOmeRanges, d_params.BoxSizes,
        initHkls, d_params.nhkls, d_params.MinEta, initTheorSpots);

    for(int sp=0; sp<nSpots; sp++){
      double *s = &spots[sp*11];
      double obsY=s[0], obsZ=s[1], obsOme=s[2];
      double bestDist = 1e20;
      int bestIdx = -1;
      for(int t=0; t<nInitT; t++){
        double tyl=initTheorSpots[t*9+0], tzl=initTheorSpots[t*9+1];
        double tome=initTheorSpots[t*9+2];
        // Compute displaced theoretical position for initial grain pos
        double DisplY=0, DisplZ=0;
        gpu_DisplacementInTheSpot(Pos0[0],Pos0[1],Pos0[2],
            d_params.Lsd, tyl, tzl, tome,
            d_params.wedge, d_params.chi, &DisplY, &DisplZ);
        double dyl = tyl - DisplY, dzl = tzl - DisplZ;
        double dy = obsY - dyl, dz = obsZ - dzl;
        double dOme = obsOme - tome;
        double dist = dy*dy + dz*dz + dOme*dOme;
        if(dist < bestDist){ bestDist = dist; bestIdx = t; }
      }
      if(bestIdx >= 0){
        // Only set col 8 = theoretical spot sequence number (nrhkls).
        // Cols 4,5,6 are OBSERVED positions from AllSpots — they stay as loaded.
        // CalcAngleErrors in the CPU also only changes col 8, not 4,5,6.
        s[8] = initTheorSpots[bestIdx*9+8]; // nrhkls sequence number
      }
    }
  }

  // ─── Stage 1: Fit Pos+Orient+Strain (12D) ───
  double x0_12[12], lb12[12], ub12[12], res12[12];
  for(int i=0;i<3;i++){x0_12[i]=Pos0[i];x0_12[i+3]=Euler0[i];}
  for(int i=0;i<6;i++) x0_12[i+6]=LatCin[i];
  for(int i=0;i<3;i++){
    lb12[i]=Pos0[i]-d_MargPos; ub12[i]=Pos0[i]+d_MargPos;
    lb12[i+3]=Euler0[i]-0.01; ub12[i+3]=Euler0[i]+0.01;
  }
  // Clamp position bounds
  if(lb12[0]<-d_Rsample)lb12[0]=-d_Rsample; if(ub12[0]>d_Rsample)ub12[0]=d_Rsample;
  if(lb12[1]<-d_Rsample)lb12[1]=-d_Rsample; if(ub12[1]>d_Rsample)ub12[1]=d_Rsample;
  if(lb12[2]<-d_Hbeam/2)lb12[2]=-d_Hbeam/2; if(ub12[2]>d_Hbeam/2)ub12[2]=d_Hbeam/2;
  for(int i=0;i<6;i++){
    lb12[6+i]=LatCin[i]*(1-d_MargABC/100);
    ub12[6+i]=LatCin[i]*(1+d_MargABC/100);
  }
  for(int i=3;i<6;i++){
    lb12[6+i]=LatCin[i]*(1-d_MargABG/100);
    ub12[6+i]=LatCin[i]*(1+d_MargABG/100);
  }

  FunctorPosOrientStrain f1;
  f1.spotsYZO=spots; f1.nSpots=nSpots; f1.hklsRaw=d_hklsRaw;
  f1.scratch=scratch; f1.nMaxTheor=nMaxTheor;
  nm_optimize<12>(x0_12,lb12,ub12,res12,f1,1e-5,5000,0.05);
  nm_optimize<12>(res12,lb12,ub12,res12,f1,1e-5,5000,0.05);

  // Dynamic reassignment after Stage 1
  if (doDynReassign && d_BinData != nullptr) {
    double x12_cur[12] = {res12[0],res12[1],res12[2],
                          Euler0[0],Euler0[1],Euler0[2],
                          LatCin[0],LatCin[1],LatCin[2],LatCin[3],LatCin[4],LatCin[5]};
    int nNew = gpu_ReassignSpotsFromBins(x12_cur, d_hklsRaw, scratch,
        spots, MaxNHKLS, d_AllSpots, d_totalNSpots,
        d_ObsSpotsLab, d_nSpotsBin, d_BinData, d_nBinData, nMaxTheor);
    if (nNew > 0) { nSpots = nNew; d_nSpotsPerGrain[gIdx] = nNew; }
  }

  // Reset orient to initial, keep position from Stage 1
  for(int i=0;i<3;i++) res12[i+3]=Euler0[i];
  for(int i=0;i<6;i++) res12[i+6]=LatCin[i];

  // ─── Stage 2: Fit Orient+Strain (9D), pos fixed ───
  double x0_9[9], lb9[9], ub9[9], res9[9];
  for(int i=0;i<3;i++) x0_9[i]=Euler0[i];
  for(int i=0;i<6;i++) x0_9[i+3]=LatCin[i];
  double MargOme2=2.0;
  for(int i=0;i<3;i++){lb9[i]=Euler0[i]-MargOme2;ub9[i]=Euler0[i]+MargOme2;}
  for(int i=0;i<6;i++){lb9[3+i]=lb12[6+i];ub9[3+i]=ub12[6+i];}

  FunctorOrientStrain f2;
  f2.spotsYZO=spots; f2.nSpots=nSpots; f2.hklsRaw=d_hklsRaw;
  f2.scratch=scratch; f2.nMaxTheor=nMaxTheor;
  for(int i=0;i<3;i++) f2.Pos[i]=res12[i];
  nm_optimize<9>(x0_9,lb9,ub9,res9,f2,1e-5,5000,0.05);
  nm_optimize<9>(res9,lb9,ub9,res9,f2,1e-5,5000,0.05);

  // Dynamic reassignment after Stage 2
  if (doDynReassign && d_BinData != nullptr) {
    double x12_cur[12] = {res12[0],res12[1],res12[2],
                          res9[0],res9[1],res9[2],
                          res9[3],res9[4],res9[5],res9[6],res9[7],res9[8]};
    int nNew = gpu_ReassignSpotsFromBins(x12_cur, d_hklsRaw, scratch,
        spots, MaxNHKLS, d_AllSpots, d_totalNSpots,
        d_ObsSpotsLab, d_nSpotsBin, d_BinData, d_nBinData, nMaxTheor);
    if (nNew > 0) { nSpots = nNew; d_nSpotsPerGrain[gIdx] = nNew; }
  }

  // ─── Stage 3: Fit Strain (6D), pos+orient fixed ───
  double x0_6[6], lb6[6], ub6[6], res6[6];
  for(int i=0;i<6;i++) x0_6[i]=LatCin[i];
  for(int i=0;i<6;i++){lb6[i]=lb12[6+i];ub6[i]=ub12[6+i];}

  FunctorStrain f3;
  f3.spotsYZO=spots; f3.nSpots=nSpots; f3.hklsRaw=d_hklsRaw;
  f3.scratch=scratch; f3.nMaxTheor=nMaxTheor;
  for(int i=0;i<3;i++) f3.Pos[i]=res12[i];
  for(int i=0;i<3;i++) f3.Orient[i]=res9[i];
  nm_optimize<6>(x0_6,lb6,ub6,res6,f3,1e-5,5000,0.05);
  nm_optimize<6>(res6,lb6,ub6,res6,f3,1e-5,5000,0.05);

  // Dynamic reassignment after Stage 3
  if (doDynReassign && d_BinData != nullptr) {
    double x12_cur[12] = {res12[0],res12[1],res12[2],
                          res9[0],res9[1],res9[2],
                          res6[0],res6[1],res6[2],res6[3],res6[4],res6[5]};
    int nNew = gpu_ReassignSpotsFromBins(x12_cur, d_hklsRaw, scratch,
        spots, MaxNHKLS, d_AllSpots, d_totalNSpots,
        d_ObsSpotsLab, d_nSpotsBin, d_BinData, d_nBinData, nMaxTheor);
    if (nNew > 0) { nSpots = nNew; d_nSpotsPerGrain[gIdx] = nNew; }
  }

  // ─── Stage 4: Fit Pos (3D), orient+strain fixed ───
  double x0_3[3], lb3[3], ub3[3], res3[3];
  for(int i=0;i<3;i++) x0_3[i]=res12[i];
  for(int i=0;i<3;i++){
    lb3[i]=x0_3[i]-d_MargPos;ub3[i]=x0_3[i]+d_MargPos;
  }
  if(lb3[0]<-d_Rsample)lb3[0]=-d_Rsample; if(ub3[0]>d_Rsample)ub3[0]=d_Rsample;
  if(lb3[1]<-d_Rsample)lb3[1]=-d_Rsample; if(ub3[1]>d_Rsample)ub3[1]=d_Rsample;
  if(lb3[2]<-d_Hbeam/2)lb3[2]=-d_Hbeam/2; if(ub3[2]>d_Hbeam/2)ub3[2]=d_Hbeam/2;

  FunctorPos f4;
  f4.spotsYZO=spots; f4.nSpots=nSpots; f4.hklsRaw=d_hklsRaw;
  f4.scratch=scratch; f4.nMaxTheor=nMaxTheor;
  for(int i=0;i<3;i++) f4.Orient[i]=res9[i];
  for(int i=0;i<6;i++) f4.Strains[i]=res6[i];
  nm_optimize<3>(x0_3,lb3,ub3,res3,f4,1e-5,5000,0.05);
  nm_optimize<3>(res3,lb3,ub3,res3,f4,1e-5,5000,0.05);

  // ─── Write results ───
  // Final: pos=res3, orient=res9[0..2], strain=res6[0..5]
  double *out = &d_results[gIdx * 27];
  // OrientsFit: [SpID, orient9]
  double EulerFit[3]={res9[0],res9[1],res9[2]};
  double OF[3][3]; gpu_Euler2OrientMat(EulerFit, OF);
  out[0] = (double)(gIdx); // SpotID placeholder
  for(int i=0;i<3;i++)for(int j=0;j<3;j++) out[1+i*3+j]=OF[i][j];
  // PositionsFit: [SpID, pos3]
  out[10]=(double)(gIdx); out[11]=res3[0]; out[12]=res3[1]; out[13]=res3[2];
  // StrainsFit: [SpID, latc6]
  out[14]=(double)(gIdx);
  for(int i=0;i<6;i++) out[15+i]=res6[i];
  // ErrorsFin: [SpID, err3] — compute final error
  out[21]=(double)(gIdx);
  double finalX[12]={res3[0],res3[1],res3[2],res9[0],res9[1],res9[2],
                     res6[0],res6[1],res6[2],res6[3],res6[4],res6[5]};
  double finalErr = gpu_FitErrorsPosT(finalX, spots, nSpots, d_hklsRaw,
                                       scratch, nMaxTheor);
  out[22]=finalErr; out[23]=0; out[24]=0;
  out[25]=0; // meanRadius placeholder
  out[26]=NrObs/NrExp; // completeness
}

// ═══════════════════════════════════════════
//  Host helper: read hkls.csv → flat array
// ═══════════════════════════════════════════
static int ReadHKLs(const char *fn, double *hkls, int maxHKLs, int *RingNumbers,
                     int nRings, double *RingRadii, double MaxTtheta) {
  FILE *f = fopen(fn, "r");
  if (!f) { printf("Could not open %s\n", fn); return 0; }
  char line[1024]; fgets(line, sizeof(line), f); // skip header
  int nhkls = 0;
  int h, k, l, Rnr; double ds, tht; char dum[64];
  while (fgets(line, sizeof(line), f)) {
    sscanf(line, "%d %d %d %lf %d %s %s %s %lf", &h, &k, &l, &ds, &Rnr,
           dum, dum, dum, &tht);
    if (tht > MaxTtheta / 2) break;
    for (int i = 0; i < nRings; i++) {
      if (Rnr == RingNumbers[i]) {
        hkls[nhkls*7+0]=h; hkls[nhkls*7+1]=k; hkls[nhkls*7+2]=l;
        hkls[nhkls*7+3]=ds; hkls[nhkls*7+4]=tht;
        hkls[nhkls*7+5]=RingRadii[i]; hkls[nhkls*7+6]=RingNumbers[i];
        nhkls++;
        break;
      }
    }
    if (nhkls >= maxHKLs) break;
  }
  fclose(f);
  return nhkls;
}

// Host Euler functions removed — not needed (all fitting is on GPU)

static double getTimeSec() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
  printf("\t\tFitPosOrStrainsGPU (MIDAS %s)\n", MIDAS_VERSION_STRING);
  printf("GPU-accelerated FF-HEDM grain refinement.\n");
  printf("Contact hsharma@anl.gov for questions.\n\n");
  if (argc != 6) {
    printf("Usage: %s params.txt blockNr nBlocks nSpotsToIndex numProcs\n", argv[0]);
    return 1;
  }
  double t0 = getTimeSec();

  // ─── Parse parameters ───
  MIDASConfig cfg;
  if (midas_parse_params(argv[1], &cfg) != 0) return 1;
  int blockNr = atoi(argv[2]), nBlocks = atoi(argv[3]);
  int nSpotsToIndex = atoi(argv[4]);

  double Wavelength=cfg.Wavelength, Lsd=cfg.Lsd;
  double LatCinT[6]; memcpy(LatCinT, cfg.LatticeConstant, sizeof(LatCinT));
  double wedge=cfg.Wedge, MinEta=cfg.MinEta, chi=0;
  double OmegaRanges[MAXNOMEGARANGES][2], BoxSizes[MAXNOMEGARANGES][4];
  int nOmeRanges=cfg.nOmeRanges;
  for(int i=0;i<nOmeRanges&&i<MAXNOMEGARANGES;i++){
    OmegaRanges[i][0]=cfg.OmegaRanges[i][0]; OmegaRanges[i][1]=cfg.OmegaRanges[i][1];
    BoxSizes[i][0]=cfg.BoxSizes[i][0]; BoxSizes[i][1]=cfg.BoxSizes[i][1];
    BoxSizes[i][2]=cfg.BoxSizes[i][2]; BoxSizes[i][3]=cfg.BoxSizes[i][3];
  }
  int RingNumbers[200], nRings=cfg.nRingNumbers;
  for(int i=0;i<nRings;i++) RingNumbers[i]=cfg.RingNumbers[i];
  double RingRadii[200]; int nRingRadii=cfg.nRingRadii;
  for(int i=0;i<nRingRadii;i++) RingRadii[i]=cfg.RingRadii[i];
  double Rsample=cfg.Rsample, Hbeam=cfg.Hbeam;
  double MargABC=cfg.MargABC, MargABG=cfg.MargABG;
  double MaxRingRad=cfg.RhoD;
  char OutputFolder[1024], ResultFolder[1024];
  strcpy(OutputFolder, cfg.OutputFolder); strcpy(ResultFolder, cfg.ResultFolder);

  printf("Wavelength: %lf, Lsd: %lf, nHKL rings: %d\n", Wavelength, Lsd, nRings);

  // ─── Read HKLs ───
  double *hklsFlat = (double*)calloc(MaxNHKLS * 7, sizeof(double));
  double MaxTtheta = rad2deg * atan(MaxRingRad / Lsd);
  int nhkls = ReadHKLs("hkls.csv", hklsFlat, MaxNHKLS, RingNumbers, nRings,
                        RingRadii, MaxTtheta);
  printf("HKLs: %d\n", nhkls);

  // ─── Read AllSpots (ExtraInfo.bin) ───
  char tmpstr[2048]; sprintf(tmpstr, "%s", ResultFolder);
  char *cwd = dirname(tmpstr);
  char filename[2048];
  sprintf(filename, "%s/ExtraInfo.bin", cwd);
  int fd = open(filename, O_RDONLY);
  if(fd<0){printf("Cannot open %s\n",filename);return 1;}
  struct stat s; fstat(fd, &s);
  double *AllSpots = (double*)mmap(0, s.st_size, PROT_READ, MAP_SHARED, fd, 0);
  int nSpots = (int)(s.st_size / (16 * sizeof(double)));
  printf("AllSpots: %d spots from ExtraInfo.bin\n", nSpots);

  // ─── Read SpotsToIndex.csv ───
  int startRowNr = (int)(ceil((double)nSpotsToIndex/nBlocks)) * blockNr;
  int endRowNr = (int)(ceil((double)nSpotsToIndex/nBlocks)) * (blockNr+1);
  if (endRowNr > nSpotsToIndex-1) endRowNr = nSpotsToIndex-1;
  int nSptIDs = endRowNr - startRowNr + 1;
  int *SptIDs = (int*)malloc(nSptIDs * sizeof(int));
  {
    FILE *sf = fopen("SpotsToIndex.csv", "r");
    if(!sf){printf("Cannot open SpotsToIndex.csv\n");return 1;}
    char line[5024]; int it;
    for(it=0;it<startRowNr;it++) fgets(line,sizeof(line),sf);
    for(it=0;it<nSptIDs;it++){fgets(line,sizeof(line),sf);sscanf(line,"%d",&SptIDs[it]);}
    fclose(sf);
  }
  printf("SpotIDs to fit: %d (block %d/%d)\n", nSptIDs, blockNr, nBlocks);

  // ─── Build per-grain data on host ───
  // For each grain: read IndexBest.bin + IndexBestFull.bin → initial params + spots
  int nMaxTheor = nhkls * 2;
  int MaxNSpotsBest = MaxNHKLS;
  double MargPos = Rsample / 2;

  // Prepare per-grain arrays
  double *h_initData = (double*)calloc(nSptIDs * 15, sizeof(double));
  double *h_spotData = (double*)calloc((size_t)nSptIDs * MaxNSpotsBest * 11, sizeof(double));
  int *h_nSpotsPerGrain = (int*)calloc(nSptIDs, sizeof(int));
  int *h_SpotIDs = (int*)calloc(nSptIDs, sizeof(int)); // actual SpotIDs for output

  // Read all grains' initial data
  char InpFN[2048], InpFN2[2048];
  sprintf(InpFN, "%s/IndexBest.bin", OutputFolder);
  sprintf(InpFN2, "%s/IndexBestFull.bin", OutputFolder);

  // Read SpotsToIndex to find rowNr for each SpId
  int *allSpotIDs_csv = NULL; int totalCSVLines = 0;
  {
    FILE *sf2 = fopen("SpotsToIndex.csv", "r");
    char line[5024];
    while(fgets(line,sizeof(line),sf2)) totalCSVLines++;
    rewind(sf2);
    allSpotIDs_csv = (int*)malloc(totalCSVLines * sizeof(int));
    for(int i=0;i<totalCSVLines;i++){fgets(line,sizeof(line),sf2);sscanf(line,"%d",&allSpotIDs_csv[i]);}
    fclose(sf2);
  }

  int nValidGrains = 0;
  for (int g = 0; g < nSptIDs; g++) {
    int SpId = SptIDs[g];
    h_SpotIDs[g] = SpId;
    printf("  Grain %d: SpId=%d\n", g, SpId);
    if (SpId == -1) { printf("    SKIP: SpId==-1\n"); continue; }

    // Find rowNr for this SpId
    int rowNr = -1;
    for (int c = 0; c < totalCSVLines; c++) {
      if (allSpotIDs_csv[c] == SpId) { rowNr = c; break; }
    }
    if (rowNr < 0) { printf("    SKIP: rowNr not found in CSV (%d lines)\n", totalCSVLines); continue; }
    printf("    rowNr=%d\n", rowNr);

    // Read from IndexBest.bin
    int inpF = open(InpFN, O_RDONLY);
    if (inpF < 0) { printf("    SKIP: cannot open %s\n", InpFN); continue; }
    size_t offst = (size_t)rowNr * 15 * sizeof(double);
    double locArr[15] = {0};
    ssize_t rc = pread(inpF, locArr, sizeof(locArr), offst);
    close(inpF);
    printf("    pread: offset=%zu, rc=%zd (need %zu)\n", offst, rc, sizeof(locArr));
    if (rc < (ssize_t)sizeof(locArr)) { printf("    SKIP: short read\n"); continue; }
    printf("    locArr[0]=%.1f locArr[13]=%.1f locArr[14]=%.1f\n", locArr[0], locArr[13], locArr[14]);
    if (locArr[14] == 0) { printf("    SKIP: NrObs==0\n"); continue; }

    memcpy(&h_initData[g * 15], locArr, sizeof(locArr));

    int nObs = (int)locArr[14];
    if (nObs > MaxNSpotsBest || nObs <= 0) { printf("    SKIP: nObs=%d out of range\n", nObs); continue; }
    printf("    nObs=%d, reading IndexBestFull.bin\n", nObs);

    // Read from IndexBestFull.bin → get spot IDs
    // CPU format: interleaved [spotID, distance, spotID, distance, ...]
    int inpF2 = open(InpFN2, O_RDONLY);
    if (inpF2 < 0) { printf("    SKIP: cannot open %s\n", InpFN2); continue; }
    size_t offst2 = (size_t)rowNr * MaxNHKLS * 2 * sizeof(double);
    double *locArr2 = (double*)calloc(nObs * 2, sizeof(double));
    pread(inpF2, locArr2, nObs * 2 * sizeof(double), offst2);
    close(inpF2);

    // SpotIDs are at even indices: locArr2[i*2+0]=spotID, locArr2[i*2+1]=distance
    int nFound = 0;
    for (int i = 0; i < nObs && nFound < MaxNSpotsBest; i++) {
      int spotID = (int)locArr2[i * 2]; // interleaved format
      int spotPos = spotID - 1;
      if (spotPos < 0 || spotPos >= nSpots) { printf("    spot %d: invalid spotPos=%d\n", i, spotPos); continue; }
      double *sp = &h_spotData[(size_t)g * MaxNSpotsBest * 11 + nFound * 11];
      sp[0] = AllSpots[spotPos*16+0];
      sp[1] = AllSpots[spotPos*16+1];
      sp[2] = AllSpots[spotPos*16+2];
      sp[3] = AllSpots[spotPos*16+4];
      sp[4] = AllSpots[spotPos*16+8];
      sp[5] = AllSpots[spotPos*16+9];
      sp[6] = AllSpots[spotPos*16+10];
      sp[7] = AllSpots[spotPos*16+5];
      sp[8] = AllSpots[spotPos*16+14];
      sp[9] = AllSpots[spotPos*16+15];
      nFound++;
    }
    free(locArr2);
    h_nSpotsPerGrain[g] = nFound;
    printf("    nFound=%d spots\n", nFound);
    if (nFound > 0) nValidGrains++;
  }
  free(allSpotIDs_csv);
  printf("Valid grains: %d / %d\n", nValidGrains, nSptIDs);

  // ─── GPU setup ───
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (%d MB)\n", prop.name, (int)(prop.totalGlobalMem/(1024*1024)));

  // Upload constant params
  FitGPUParams params;
  params.Lsd = Lsd; params.Wavelength = Wavelength;
  params.MinEta = MinEta; params.wedge = wedge; params.chi = chi;
  params.nOmeRanges = nOmeRanges; params.nhkls = nhkls;
  for (int i = 0; i < nOmeRanges; i++) {
    params.OmegaRanges[i*2] = OmegaRanges[i][0];
    params.OmegaRanges[i*2+1] = OmegaRanges[i][1];
    params.BoxSizes[i*4] = BoxSizes[i][0]; params.BoxSizes[i*4+1] = BoxSizes[i][1];
    params.BoxSizes[i*4+2] = BoxSizes[i][2]; params.BoxSizes[i*4+3] = BoxSizes[i][3];
  }
  CUDA_CHECK(cudaMemcpyToSymbol(d_params, &params, sizeof(FitGPUParams)));

  // GPU memory allocation
  int scratchPerGrain = nhkls*7 + nMaxTheor*9 + nMaxTheor*2; // doubles
  double *d_initData, *d_spotData, *d_hklsRaw, *d_scratch, *d_LatCin, *d_results;
  int *d_nSpotsPerGrain;

  CUDA_CHECK(cudaMalloc(&d_hklsRaw, nhkls * 7 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_hklsRaw, hklsFlat, nhkls*7*sizeof(double), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_LatCin, 6 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_LatCin, LatCinT, 6*sizeof(double), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_initData, (size_t)nSptIDs * 15 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_initData, h_initData, (size_t)nSptIDs*15*sizeof(double), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_spotData, (size_t)nSptIDs * MaxNSpotsBest * 11 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_spotData, h_spotData, (size_t)nSptIDs*MaxNSpotsBest*11*sizeof(double), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_nSpotsPerGrain, nSptIDs * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_nSpotsPerGrain, h_nSpotsPerGrain, nSptIDs*sizeof(int), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)nSptIDs * scratchPerGrain * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_results, (size_t)nSptIDs * 27 * sizeof(double)));

  // Upload AllSpots for reassignment (already mmap'd)
  double *d_AllSpotsGPU = nullptr;
  CUDA_CHECK(cudaMalloc(&d_AllSpotsGPU, (size_t)nSpots * 16 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_AllSpotsGPU, AllSpots, (size_t)nSpots*16*sizeof(double), cudaMemcpyHostToDevice));

  // Bin data for dynamic reassignment — disabled for initial validation
  // TODO: Load ObsSpotsLab.bin, BinData.bin, nBinData.bin and upload
  double *d_ObsSpotsLabGPU = nullptr;
  int *d_BinDataGPU = nullptr, *d_nBinDataGPU = nullptr;
  int doDynReassign = 0; // set to 1 once bin data is loaded

  size_t totalGPUMB = ((size_t)nSptIDs*(15+MaxNSpotsBest*11+scratchPerGrain+27)*sizeof(double)
                       + nhkls*7*sizeof(double) + (size_t)nSpots*16*sizeof(double)) / (1024*1024);
  printf("GPU memory: ~%zu MB for %d grains\n", totalGPUMB, nSptIDs);

  // ─── Launch kernel ───
  int threadsPerBlock = 64; // low occupancy per thread due to register pressure
  int blocks = (nSptIDs + threadsPerBlock - 1) / threadsPerBlock;
  printf("Launching kernel: %d blocks × %d threads\n", blocks, threadsPerBlock);
  double tKernel = getTimeSec();

  fitGrainsKernel<<<blocks, threadsPerBlock>>>(
      nSptIDs, d_initData, d_spotData, d_nSpotsPerGrain, d_hklsRaw,
      d_scratch, d_LatCin, Rsample, Hbeam, MargPos, 0.01, MargABC, MargABG,
      d_results, scratchPerGrain, nMaxTheor,
      d_AllSpotsGPU, nSpots, d_ObsSpotsLabGPU, 0,
      d_BinDataGPU, d_nBinDataGPU, doDynReassign);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Kernel done: %.2f s\n", getTimeSec() - tKernel);

  // ─── Download results ───
  double *h_results = (double*)malloc((size_t)nSptIDs * 27 * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_results, d_results, (size_t)nSptIDs*27*sizeof(double), cudaMemcpyDeviceToHost));

  // ─── Write output files ───
  char KeyFN[1024]; sprintf(KeyFN, "%s/Key.bin", ResultFolder);
  int keyFD = open(KeyFN, O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
  char OutFN[1024]; sprintf(OutFN, "%s/OrientPosFit.bin", ResultFolder);

  for (int g = 0; g < nSptIDs; g++) {
    int SpId = h_SpotIDs[g];
    int nSpotsComp = h_nSpotsPerGrain[g];
    double *out = &h_results[g * 27];

    // Find rowNr
    int rowNr = -1;
    FILE *sf = fopen("SpotsToIndex.csv", "r");
    char line[5024]; int cnt=0;
    while(fgets(line,sizeof(line),sf)){int id;sscanf(line,"%d",&id);if(id==SpId){rowNr=cnt;break;}cnt++;}
    fclose(sf);
    if (rowNr < 0) rowNr = g;

    // Fix SpotID in output
    out[0] = SpId; out[10] = SpId; out[14] = SpId; out[21] = SpId;

    // Key.bin
    int SizeKey = 2 * sizeof(int);
    size_t offKey = (size_t)SizeKey * rowNr;
    int KeyInfo[2] = {(nSpotsComp > 0) ? SpId : 0, nSpotsComp};
    if (keyFD > 0) pwrite(keyFD, KeyInfo, SizeKey, offKey);

    // OrientPosFit.bin
    if (nSpotsComp > 0) {
      int outFD = open(OutFN, O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
      if (outFD > 0) {
        pwrite(outFD, out, 27*sizeof(double), (size_t)rowNr*27*sizeof(double));
        close(outFD);
      }
    }

    // Print per-grain result
    if (nSpotsComp > 0) {
      printf("SpotID %6d, %3d spots, Err: %7.2f, Pos: %7.2f %7.2f %7.2f, "
             "LatC: %6.4f %6.4f %6.4f %7.3f %7.3f %7.3f\n",
             SpId, nSpotsComp, out[22],
             out[11], out[12], out[13],
             out[15], out[16], out[17], out[18], out[19], out[20]);
    }
  }
  if (keyFD > 0) close(keyFD);

  // Cleanup
  free(h_results); free(h_initData); free(h_spotData);
  free(h_nSpotsPerGrain); free(h_SpotIDs); free(hklsFlat); free(SptIDs);
  cudaFree(d_initData); cudaFree(d_spotData); cudaFree(d_nSpotsPerGrain);
  cudaFree(d_hklsRaw); cudaFree(d_scratch); cudaFree(d_LatCin); cudaFree(d_results);
  cudaFree(d_AllSpotsGPU);
  if (d_ObsSpotsLabGPU) cudaFree(d_ObsSpotsLabGPU);
  if (d_BinDataGPU) cudaFree(d_BinDataGPU);
  if (d_nBinDataGPU) cudaFree(d_nBinDataGPU);

  printf("Finished, time elapsed: %.2f seconds.\n", getTimeSec() - t0);
  return 0;
}

