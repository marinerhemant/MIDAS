// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// Code to FitPosOrStrains using multiple datasets

// Include libraries
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// Define constants
#define MAX_SPOTS 5000
#define MAX_GRAIN_IDS 100000
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MaxNSpots 6000000
#define MaxNSpotsBest 5000 // MUST BE EQUAL TO MaxNHKLS
#define MaxNHKLS 5000
#define EPS 1E-12
#define CalcNorm3(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
#define CalcNorm2(x, y) sqrt((x) * (x) + (y) * (y))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define MAXNOMEGARANGES 2000

// Helper functions and structs
static void check(int test, const char *message, ...) {
  if (test) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}

static inline int **allocMatrixInt(int nrows, int ncols) {
  int **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline double **allocMatrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  double determinant;
  determinant = m[0][0] * ((m[1][1] * m[2][2]) - (m[2][1] * m[1][2])) -
                m[0][1] * (m[1][0] * m[2][2] - m[2][0] * m[1][2]) +
                m[0][2] * (m[1][0] * m[2][1] - m[2][0] * m[1][1]);

  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = rad2deg * psi;
  Euler[1] = rad2deg * phi;
  Euler[2] = rad2deg * theta;
}

static inline void Euler2OrientMat(double Euler[3], double m_out[3][3]) {
  double psi, phi, theta, cps, cph, cth, sps, sph, sth;
  psi = Euler[0];
  phi = Euler[1];
  theta = Euler[2];
  cps = cosd(psi);
  cph = cosd(phi);
  cth = cosd(theta);
  sps = sind(psi);
  sph = sind(phi);
  sth = sind(theta);
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

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void CorrectHKLsLatC(double LatC[6], double **hklsIn, int nhkls,
                                   double Lsd, double Wavelength,
                                   double **hkls) {
  double a = LatC[0], b = LatC[1], c = LatC[2], alpha = LatC[3], beta = LatC[4],
         gamma = LatC[5];
  int hklnr;
  double SinA = sind(alpha), SinB = sind(beta), SinG = sind(gamma),
         CosA = cosd(alpha), CosB = cosd(beta), CosG = cosd(gamma);
  double GammaPr = acosd((CosA * CosB - CosG) / (SinA * SinB)),
         BetaPr = acosd((CosG * CosA - CosB) / (SinG * SinA)),
         SinBetaPr = sind(BetaPr);
  double Vol = (a * (b * (c * (SinA * (SinBetaPr * (SinG)))))),
         APr = b * c * SinA / Vol, BPr = c * a * SinB / Vol,
         CPr = a * b * SinG / Vol;
  double B[3][3];
  B[0][0] = APr;
  B[0][1] = (BPr * cosd(GammaPr)), B[0][2] = (CPr * cosd(BetaPr)), B[1][0] = 0,
  B[1][1] = (BPr * sind(GammaPr)), B[1][2] = (-CPr * SinBetaPr * CosA),
  B[2][0] = 0, B[2][1] = 0, B[2][2] = (CPr * SinBetaPr * SinA);
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    double ginit[3];
    ginit[0] = hklsIn[hklnr][0];
    ginit[1] = hklsIn[hklnr][1];
    ginit[2] = hklsIn[hklnr][2];
    double GCart[3];
    MatrixMult(B, ginit, GCart);
    double Ds = 1 / (sqrt((GCart[0] * GCart[0]) + (GCart[1] * GCart[1]) +
                          (GCart[2] * GCart[2])));
    hkls[hklnr][0] = GCart[0];
    hkls[hklnr][1] = GCart[1];
    hkls[hklnr][2] = GCart[2];
    hkls[hklnr][3] = Ds;
    double Theta = (asind((Wavelength) / (2 * Ds)));
    hkls[hklnr][4] = Theta;
    double Rad = Lsd * (tand(2 * Theta));
    hkls[hklnr][5] = Rad;
    hkls[hklnr][6] = hklsIn[hklnr][6];
  }
}

static inline void DisplacementInTheSpot(double a, double b, double c,
                                         double xi, double yi, double zi,
                                         double omega, double wedge, double chi,
                                         double *Displ_y, double *Displ_z) {
  double sinOme = sind(omega), cosOme = cosd(omega), AcosOme = a * cosOme,
         BsinOme = b * sinOme;
  double XNoW = AcosOme - BsinOme, YNoW = (a * sinOme) + (b * cosOme), ZNoW = c;
  double WedgeRad = deg2rad * wedge, CosW = cos(WedgeRad), SinW = sin(WedgeRad),
         XW = XNoW * CosW - ZNoW * SinW, YW = YNoW;
  double ZW = (XNoW * SinW) + (ZNoW * CosW), ChiRad = deg2rad * chi,
         CosC = cos(ChiRad), SinC = sin(ChiRad), XC = XW;
  double YC = (CosC * YW) - (SinC * ZW), ZC = (SinC * YW) + (CosC * ZW);
  double IK[3], NormIK;
  IK[0] = xi - XC;
  IK[1] = yi - YC;
  IK[2] = zi - ZC;
  NormIK = sqrt((IK[0] * IK[0]) + (IK[1] * IK[1]) + (IK[2] * IK[2]));
  IK[0] = IK[0] / NormIK;
  IK[1] = IK[1] / NormIK;
  IK[2] = IK[2] / NormIK;
  *Displ_y = YC - ((XC * IK[1]) / (IK[0]));
  *Displ_z = ZC - ((XC * IK[2]) / (IK[0]));
}

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void CorrectForOme(double yc, double zc, double Lsd,
                                 double OmegaIni, double wl, double wedge,
                                 double *ysOut, double *zsOut,
                                 double *OmegaOut) {
  double ysi = yc, zsi = zc;
  double CosOme = cos(deg2rad * OmegaIni), SinOme = sin(deg2rad * OmegaIni);
  double eta = CalcEtaAngle(ysi, zsi);
  double RingRadius = sqrt((ysi * ysi) + (zsi * zsi));
  double tth = rad2deg * atan(RingRadius / Lsd);
  double theta = tth / 2;
  double SinTheta = sin(deg2rad * theta);
  double CosTheta = cos(deg2rad * theta);
  double ds = 2 * SinTheta / wl;
  double CosW = cos(deg2rad * wedge);
  double SinW = sin(deg2rad * wedge);
  double SinEta = sin(deg2rad * eta);
  double CosEta = cos(deg2rad * eta);
  double k1 = -ds * SinTheta;
  double k2 = -ds * CosTheta * SinEta;
  double k3 = ds * CosTheta * CosEta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW);
  double k2f = k2;
  double k3f = (k3 * CosW) - (k1 * SinW);
  double G1a = (k1f * CosOme) + (k2f * SinOme);
  double G2a = (k2f * CosOme) - (k1f * SinOme);
  double G3a = k3f;
  double LenGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / LenGa;
  double g2 = G2a * ds / LenGa;
  double g3 = G3a * ds / LenGa;
  SinW = 0;
  CosW = 1;
  double LenG = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(LenG * LenG * wl) / 2;
  tth = 2 * rad2deg * asin(wl * LenG / 2);
  RingRadius = Lsd * tan(deg2rad * tth);
  double A = (k1i + (g3 * SinW)) / (CosW);
  double a_Sin = (g1 * g1) + (g2 * g2);
  double b_Sin = 2 * A * g2;
  double c_Sin = (A * A) - (g1 * g1);
  double a_Cos = a_Sin;
  double b_Cos = -2 * A * g1;
  double c_Cos = (A * A) - (g2 * g2);
  double Par_Sin = (b_Sin * b_Sin) - (4 * a_Sin * c_Sin);
  double Par_Cos = (b_Cos * b_Cos) - (4 * a_Cos * c_Cos);
  double P_check_Sin = 0;
  double P_check_Cos = 0;
  double P_Sin, P_Cos;
  if (Par_Sin >= 0)
    P_Sin = sqrt(Par_Sin);
  else {
    P_Sin = 0;
    P_check_Sin = 1;
  }
  if (Par_Cos >= 0)
    P_Cos = sqrt(Par_Cos);
  else {
    P_Cos = 0;
    P_check_Cos = 1;
  }
  double SinOmega1 = (-b_Sin - P_Sin) / (2 * a_Sin);
  double SinOmega2 = (-b_Sin + P_Sin) / (2 * a_Sin);
  double CosOmega1 = (-b_Cos - P_Cos) / (2 * a_Cos);
  double CosOmega2 = (-b_Cos + P_Cos) / (2 * a_Cos);
  if (SinOmega1 < -1)
    SinOmega1 = 0;
  else if (SinOmega1 > 1)
    SinOmega1 = 0;
  else if (SinOmega2 < -1)
    SinOmega2 = 0;
  else if (SinOmega2 > 1)
    SinOmega2 = 0;
  if (CosOmega1 < -1)
    CosOmega1 = 0;
  else if (CosOmega1 > 1)
    CosOmega1 = 0;
  else if (CosOmega2 < -1)
    CosOmega2 = 0;
  else if (CosOmega2 > 1)
    CosOmega2 = 0;
  if (P_check_Sin == 1) {
    SinOmega1 = 0;
    SinOmega2 = 0;
  }
  if (P_check_Cos == 1) {
    CosOmega1 = 0;
    CosOmega2 = 0;
  }
  double Option1 = fabs((SinOmega1 * SinOmega1) + (CosOmega1 * CosOmega1) - 1);
  double Option2 = fabs((SinOmega1 * SinOmega1) + (CosOmega2 * CosOmega2) - 1);
  double Omega1, Omega2;
  if (Option1 < Option2) {
    Omega1 = rad2deg * atan2(SinOmega1, CosOmega1);
    Omega2 = rad2deg * atan2(SinOmega2, CosOmega2);
  } else {
    Omega1 = rad2deg * atan2(SinOmega1, CosOmega2);
    Omega2 = rad2deg * atan2(SinOmega2, CosOmega1);
  }
  double OmeDiff1 = fabs(Omega1 - OmegaIni);
  double OmeDiff2 = fabs(Omega2 - OmegaIni);
  double Omega;
  if (OmeDiff1 < OmeDiff2)
    Omega = Omega1;
  else
    Omega = Omega2;
  double SinOmega = sin(deg2rad * Omega);
  double CosOmega = cos(deg2rad * Omega);
  double Fact = (g1 * CosOmega) - (g2 * SinOmega);
  double Eta = CalcEtaAngle(k2, k3);
  double Sin_Eta = sin(deg2rad * Eta);
  double Cos_Eta = cos(deg2rad * Eta);
  *ysOut = -RingRadius * Sin_Eta;
  *zsOut = RingRadius * Cos_Eta;
  *OmegaOut = Omega;
}

static inline void SpotToGv(double xi, double yi, double zi, double Omega,
                            double theta, double *g1, double *g2, double *g3) {
  double CosOme = cosd(Omega), SinOme = sind(Omega), eta = CalcEtaAngle(yi, zi),
         TanEta = tand(-eta), SinTheta = sind(theta);
  double CosTheta = cosd(theta), CosW = 1, SinW = 0,
         k3 = SinTheta * (1 + xi) / ((yi * TanEta) + zi), k2 = TanEta * k3,
         k1 = -SinTheta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW), k3f = (k3 * CosW) - (k1 * SinW),
         k2f = k2;
  *g1 = (k1f * CosOme) + (k2f * SinOme);
  *g2 = (k2f * CosOme) - (k1f * SinOme);
  *g3 = k3f;
}

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAXNOMEGARANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAXNOMEGARANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots);

// Structs
struct data_FitPosIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  // Dataset 2
  int nSpotsComp2;
  double **spotsYZO2;
  double Lsd2;
  double Wavelength2;
  int nOmeRanges2;
  double OmegaRanges2[MAXNOMEGARANGES][2];
  double BoxSizes2[MAXNOMEGARANGES][4];
  double MinEta2;
  double wedge2;
  double chi2;
};

struct data_FitOrientIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Pos[3];
  // Dataset 2
  int nSpotsComp2;
  double **spotsYZO2;
  double Lsd2;
  double Wavelength2;
  int nOmeRanges2;
  double OmegaRanges2[MAXNOMEGARANGES][2];
  double BoxSizes2[MAXNOMEGARANGES][4];
  double MinEta2;
  double wedge2;
  double chi2;
};

struct data_FitStrainIni {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Pos[3];
  double Orient[3];
  // Dataset 2
  int nSpotsComp2;
  double **spotsYZO2;
  double Lsd2;
  double Wavelength2;
  int nOmeRanges2;
  double OmegaRanges2[MAXNOMEGARANGES][2];
  double BoxSizes2[MAXNOMEGARANGES][4];
  double MinEta2;
  double wedge2;
  double chi2;
};

struct data_FitPos {
  int nSpotsComp;
  double **spotsYZO;
  int nhkls;
  double **hkls;
  double Lsd;
  double Wavelength;
  int nOmeRanges;
  double OmegaRanges[MAXNOMEGARANGES][2];
  double BoxSizes[MAXNOMEGARANGES][4];
  double MinEta;
  double wedge;
  double chi;
  double Orient[3];
  double Strains[6];
  // Dataset 2
  int nSpotsComp2;
  double **spotsYZO2;
  double Lsd2;
  double Wavelength2;
  int nOmeRanges2;
  double OmegaRanges2[MAXNOMEGARANGES][2];
  double BoxSizes2[MAXNOMEGARANGES][4];
  double MinEta2;
  double wedge2;
  double chi2;
};

static void ReadParameters(char *folder, double *Lsd, double *Wavelength, double *wedge, double *MinEta, double *chi, 
                           double OmegaRanges[MAXNOMEGARANGES][2], int *nOmeRanges, 
                           double BoxSizes[MAXNOMEGARANGES][4], int *nBoxSizes,
                           int RingNumbers[200], int *nRingNumbers, double RingRadii[200])
{
    char filename[2048];
    sprintf(filename,"%s/parameters.txt",folder);
    FILE *fileParam = fopen(filename,"r");
    if (fileParam == NULL) {
        sprintf(filename,"%s/params.txt",folder);
        fileParam = fopen(filename,"r");
        if (fileParam == NULL){
             fprintf(stderr, "Could not open parameters.txt or params.txt in %s\n", folder);
             exit(1);
        }
    }
    char aline[1000];
    char *str, dummy[1000];
    int LowNr;
    *nOmeRanges = 0;
    *nBoxSizes = 0;
    *nRingNumbers = 0;
    int cs2 = 0; 
    
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, Lsd);
            continue;
        }
        str = "Distance ";
         LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, Lsd);
            continue;
        }
        str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, Wavelength);
            continue;
        }
        str = "Wedge ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, wedge);
            continue;
        }
        str = "MinEta ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, MinEta);
            continue;
        }
        str = "Chi "; 
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
             sscanf(aline,"%s %lf", dummy, chi);
             continue;
        }
        str = "OmegaRange ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf", dummy,
				&OmegaRanges[*nOmeRanges][0], &OmegaRanges[*nOmeRanges][1]);
            (*nOmeRanges)++;
            continue;
        }
        str = "BoxSize ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf", dummy,
				&BoxSizes[*nBoxSizes][0], &BoxSizes[*nBoxSizes][1],
				&BoxSizes[*nBoxSizes][2], &BoxSizes[*nBoxSizes][3]);
            (*nBoxSizes)++;
            continue;
        }
        str = "RingNumbers ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &RingNumbers[*nRingNumbers]);
            (*nRingNumbers)++;
            continue;
        }
        str = "RingRadii ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &RingRadii[cs2]);
            cs2++;
            continue;
        }
    }
    fclose(fileParam);
}

static int ReadHKLs(char *folder, double **hkls, int RingNumbers[200], int nRingNumbers, double RingRadii[200])
{
    char filename[2048];
    sprintf(filename,"%s/hkls.csv",folder);
    FILE *hklf = fopen(filename,"r");
    if (hklf == NULL){
        hklf = fopen("hkls.csv", "r");
        if (hklf == NULL) {
             printf("Could not read the hkl file. Exiting.\n");
             exit(1);
        }
    }
    char aline[1000];
    char dummy[1000];
    fgets(aline,1000,hklf); 
    
    int h,kt,l,Rnr, nhkls=0;
	double ds,tht;
    int iter;
    
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%d %d %d %lf %d %s %s %s %lf %s %s",&h,&kt,&l,&ds,&Rnr,dummy,dummy,dummy,&tht,dummy,dummy);
        for (iter=0;iter<nRingNumbers;iter++){
			if(Rnr == RingNumbers[iter]){
				hkls[nhkls][0] = h;
				hkls[nhkls][1] = kt;
				hkls[nhkls][2] = l;
				hkls[nhkls][3] = ds;
				hkls[nhkls][4] = tht;
				hkls[nhkls][5] = RingRadii[iter];
				hkls[nhkls][6] = RingNumbers[iter];
				nhkls++;
                break;
			}
		}
    }
    fclose(hklf);
    return nhkls;
}

static inline
double FitErrorsPosTSingle(double x[12],int nSpotsComp,double **spotsYZO,int nhkls,double **hklsIn,
					 double Lsd,double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],
					 double BoxSizes[MAXNOMEGARANGES][4],double MinEta,double wedge,double chi)
{
	int i;
	int nrMatchedIndexer = nSpotsComp;
	double LatC[6];
	for (i=0;i<6;i++)LatC[i] = x[6+i];
	double **hkls;hkls = allocMatrix(nhkls,7);CorrectHKLsLatC(LatC,hklsIn,nhkls,Lsd,Wavelength,hkls);
	double OrientMatrix[3][3],EulerIn[3];EulerIn[0]=x[3];EulerIn[1]=x[4];EulerIn[2]=x[5];
	Euler2OrientMat(EulerIn,OrientMatrix);
	int nTspots,nrSp;
	double **TheorSpots;TheorSpots=allocMatrix(MaxNSpotsBest,9);
	CalcDiffractionSpots(Lsd,MinEta,OmegaRanges,nOmeRanges,hkls,nhkls,BoxSizes,&nTspots,OrientMatrix,TheorSpots);
	double **SpotsYZOGCorr;SpotsYZOGCorr=allocMatrix(nrMatchedIndexer,3);
	double DisplY,DisplZ,ys,zs,Omega, yt, zt;
	for (nrSp=0;nrSp<nrMatchedIndexer;nrSp++){
		DisplacementInTheSpot(x[0],x[1],x[2],Lsd,spotsYZO[nrSp][5],spotsYZO[nrSp][6],spotsYZO[nrSp][4],wedge,chi,&DisplY,&DisplZ);
		yt = spotsYZO[nrSp][5]-DisplY;
		zt = spotsYZO[nrSp][6]-DisplZ;
		CorrectForOme(yt,zt,Lsd,spotsYZO[nrSp][4],Wavelength,wedge,&ys,&zs,&Omega);
		SpotsYZOGCorr[nrSp][0] = ys;
		SpotsYZOGCorr[nrSp][1] = zs;
		SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
	}
	double **TheorSpotsYZWE;TheorSpotsYZWE=allocMatrix(nTspots,3);
	for (i=0;i<nTspots;i++){
		TheorSpotsYZWE[i][0] = TheorSpots[i][0];
		TheorSpotsYZWE[i][1] = TheorSpots[i][1];
		TheorSpotsYZWE[i][2] = TheorSpots[i][8];
	}
	int sp;
	double PosObs[2], PosTheor[2], Spnr;
	double Error=0;
	for (sp=0;sp<nrMatchedIndexer;sp++){
		PosObs[0] = SpotsYZOGCorr[sp][0];
		PosObs[1] = SpotsYZOGCorr[sp][1];
		Spnr = SpotsYZOGCorr[sp][2];
		for (i=0;i<nTspots;i++){
			if ((int)TheorSpotsYZWE[i][2] == (int)Spnr){
				PosTheor[0] = TheorSpotsYZWE[i][0];
				PosTheor[1] = TheorSpotsYZWE[i][1];
				Error += CalcNorm2((PosObs[0]-PosTheor[0]),(PosObs[1]-PosTheor[1]));
				break;
			}
		}
	}
	FreeMemMatrix(hkls,nhkls);
	FreeMemMatrix(TheorSpots,MaxNSpotsBest);
	FreeMemMatrix(SpotsYZOGCorr,nrMatchedIndexer);
	FreeMemMatrix(TheorSpotsYZWE,nTspots);
	return Error;
}

static inline
double FitErrorsOrientStrainsSingle(double x[9],int nSpotsComp,double **spotsYZO,int nhkls,double **hklsIn,
					 double Lsd,double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],
					 double BoxSizes[MAXNOMEGARANGES][4],double MinEta,double wedge,double chi, double Pos[3])
{
	int i,j;
	int nrMatchedIndexer = nSpotsComp;
	double LatC[6];
	for (i=0;i<6;i++)LatC[i] = x[3+i];
	double **hkls;hkls = allocMatrix(nhkls,7);CorrectHKLsLatC(LatC,hklsIn,nhkls,Lsd,Wavelength,hkls);
	double OrientMatrix[3][3],EulerIn[3];EulerIn[0]=x[0];EulerIn[1]=x[1];EulerIn[2]=x[2];
	Euler2OrientMat(EulerIn,OrientMatrix);
	int nTspots,nrSp;
	double **TheorSpots;TheorSpots=allocMatrix(MaxNSpotsBest,9);
	CalcDiffractionSpots(Lsd,MinEta,OmegaRanges,nOmeRanges,hkls,nhkls,BoxSizes,&nTspots,OrientMatrix,TheorSpots);
	double **SpotsYZOGCorr;SpotsYZOGCorr=allocMatrix(nrMatchedIndexer,4);
	double DisplY,DisplZ,ys,zs,Omega,Radius,Theta,lenK,yt,zt;
	for (nrSp=0;nrSp<nrMatchedIndexer;nrSp++){
		DisplacementInTheSpot(Pos[0],Pos[1],Pos[2],Lsd,spotsYZO[nrSp][5],spotsYZO[nrSp][6],spotsYZO[nrSp][4],wedge,chi,&DisplY,&DisplZ);
		yt = spotsYZO[nrSp][5]-DisplY;
		zt = spotsYZO[nrSp][6]-DisplZ;
		CorrectForOme(yt,zt,Lsd,spotsYZO[nrSp][4],Wavelength,wedge,&ys,&zs,&Omega);
		lenK = sqrt((Lsd*Lsd)+(ys*ys)+(zs*zs));
		Radius = sqrt((ys*ys) + (zs*zs));
		Theta = 0.5*atand(Radius/Lsd);
		double g1,g2,g3;
		SpotToGv(Lsd/lenK,ys/lenK,zs/lenK,Omega,Theta,&g1,&g2,&g3);
		SpotsYZOGCorr[nrSp][0] = g1;
		SpotsYZOGCorr[nrSp][1] = g2;
		SpotsYZOGCorr[nrSp][2] = g3;
		SpotsYZOGCorr[nrSp][3] = spotsYZO[nrSp][8];
	}
	double **TheorSpotsYZWE;TheorSpotsYZWE=allocMatrix(nTspots,4);
	for (i=0;i<nTspots;i++){for (j=0;j<3;j++){TheorSpotsYZWE[i][j] = TheorSpots[i][j+3];}TheorSpotsYZWE[i][3] = TheorSpots[i][8];}
	int sp,nTheorSpotsYZWER;
    double Spnr;
	double GObs[3],GTheors[3],NormGObs,NormGTheors,DotGs,Numers,Denoms,*Angles,minAngle,Error=0;
	Angles=malloc(20*sizeof(*Angles));
	for (sp=0;sp<nrMatchedIndexer;sp++){
		nTheorSpotsYZWER=0;
		GObs[0]=SpotsYZOGCorr[sp][0];GObs[1]=SpotsYZOGCorr[sp][1];GObs[2]=SpotsYZOGCorr[sp][2];
		NormGObs = CalcNorm3(GObs[0],GObs[1],GObs[2]);
		Spnr = SpotsYZOGCorr[sp][3];
		for (i=0;i<nTspots;i++){
			if ((int)TheorSpotsYZWE[i][3]==(int)Spnr){
				GTheors[0]=TheorSpotsYZWE[i][0];
				GTheors[1]=TheorSpotsYZWE[i][1];
				GTheors[2]=TheorSpotsYZWE[i][2];
				DotGs = ((GTheors[0]*GObs[0])+(GTheors[1]*GObs[1])+(GTheors[2]*GObs[2]));
				NormGTheors = CalcNorm3(GTheors[0],GTheors[1],GTheors[2]);
				Numers = DotGs;
				Denoms = NormGObs*NormGTheors;
				Angles[nTheorSpotsYZWER] = fabs(acosd(Numers/Denoms));
				nTheorSpotsYZWER++;
			}
		}
		if (nTheorSpotsYZWER==0)continue;
		minAngle = 1000000;
		for (i=0;i<nTheorSpotsYZWER;i++){if (Angles[i]<minAngle){minAngle=Angles[i];}}
		if (minAngle > 4) continue;
		Error += minAngle;
	}
	FreeMemMatrix(hkls,nhkls);
	FreeMemMatrix(TheorSpots,MaxNSpotsBest);
	FreeMemMatrix(SpotsYZOGCorr,nrMatchedIndexer);
	FreeMemMatrix(TheorSpotsYZWE,nTspots);
	free(Angles);
	return Error;
}

static inline
double FitErrorsStrainsSingle(double x[6],int nSpotsComp,double **spotsYZO,int nhkls,double **hklsIn,
						double Lsd,double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],
						double BoxSizes[MAXNOMEGARANGES][4],double MinEta,double wedge,double chi, double Pos[3],double EulerIn[3])
{
	int i;
	int nrMatchedIndexer = nSpotsComp;
	double LatC[6];
	for (i=0;i<6;i++)LatC[i] = x[i];
	double **hkls;hkls = allocMatrix(nhkls,7);CorrectHKLsLatC(LatC,hklsIn,nhkls,Lsd,Wavelength,hkls);
	double OrientMatrix[3][3];
	Euler2OrientMat(EulerIn,OrientMatrix);
	int nTspots,nrSp;
	double **TheorSpots;TheorSpots=allocMatrix(MaxNSpotsBest,9);
	CalcDiffractionSpots(Lsd,MinEta,OmegaRanges,nOmeRanges,hkls,nhkls,BoxSizes,&nTspots,OrientMatrix,TheorSpots);
	double **SpotsYZOGCorr;SpotsYZOGCorr=allocMatrix(nrMatchedIndexer,3);
	double DisplY,DisplZ,ys,zs,Omega, yt, zt;
	for (nrSp=0;nrSp<nrMatchedIndexer;nrSp++){
		DisplacementInTheSpot(Pos[0],Pos[1],Pos[2],Lsd,spotsYZO[nrSp][5],spotsYZO[nrSp][6],spotsYZO[nrSp][4],wedge,chi,&DisplY,&DisplZ);
		yt = spotsYZO[nrSp][5]-DisplY;
		zt = spotsYZO[nrSp][6]-DisplZ;
		CorrectForOme(yt,zt,Lsd,spotsYZO[nrSp][4],Wavelength,wedge,&ys,&zs,&Omega);
		SpotsYZOGCorr[nrSp][0] = ys;
		SpotsYZOGCorr[nrSp][1] = zs;
		SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
	}
	double **TheorSpotsYZWE;TheorSpotsYZWE=allocMatrix(nTspots,3);
	for (i=0;i<nTspots;i++){
		TheorSpotsYZWE[i][0] = TheorSpots[i][0];
		TheorSpotsYZWE[i][1] = TheorSpots[i][1];
		TheorSpotsYZWE[i][2] = TheorSpots[i][8];
	}
	int sp;
	double PosObs[2], PosTheor[2], Spnr;
	double Error=0;
	for (sp=0;sp<nrMatchedIndexer;sp++){
		PosObs[0] = SpotsYZOGCorr[sp][0];
		PosObs[1] = SpotsYZOGCorr[sp][1];
		Spnr = SpotsYZOGCorr[sp][2];
		for (i=0;i<nTspots;i++){
			if ((int)TheorSpotsYZWE[i][2] == (int)Spnr){
				PosTheor[0] = TheorSpotsYZWE[i][0];
				PosTheor[1] = TheorSpotsYZWE[i][1];
				Error += CalcNorm2((PosObs[0]-PosTheor[0]),(PosObs[1]-PosTheor[1]));
				break;
			}
		}
	}
	FreeMemMatrix(hkls,nhkls);
	FreeMemMatrix(TheorSpots,MaxNSpotsBest);
	FreeMemMatrix(SpotsYZOGCorr,nrMatchedIndexer);
	FreeMemMatrix(TheorSpotsYZWE,nTspots);
	return Error;
}

static inline
double FitErrorsPosSecSingle(double x[3],int nSpotsComp,double **spotsYZO,int nhkls,double **hklsIn,
						double Lsd,double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],
						double BoxSizes[MAXNOMEGARANGES][4],double MinEta,double wedge,double chi,double EulerIn[3],double Strains[6])
{
	int i;
	int nrMatchedIndexer = nSpotsComp;
	double LatC[6];
	for (i=0;i<6;i++)LatC[i] = Strains[i];
	double **hkls;hkls = allocMatrix(nhkls,7);CorrectHKLsLatC(LatC,hklsIn,nhkls,Lsd,Wavelength,hkls);
	double OrientMatrix[3][3];
	Euler2OrientMat(EulerIn,OrientMatrix);
	int nTspots,nrSp;
	double **TheorSpots;TheorSpots=allocMatrix(MaxNSpotsBest,9);
	CalcDiffractionSpots(Lsd,MinEta,OmegaRanges,nOmeRanges,hkls,nhkls,BoxSizes,&nTspots,OrientMatrix,TheorSpots);
	double **SpotsYZOGCorr;SpotsYZOGCorr=allocMatrix(nrMatchedIndexer,3);
	double DisplY,DisplZ,ys,zs,Omega, yt, zt;
	for (nrSp=0;nrSp<nrMatchedIndexer;nrSp++){
		DisplacementInTheSpot(x[0],x[1],x[2],Lsd,spotsYZO[nrSp][5],spotsYZO[nrSp][6],spotsYZO[nrSp][4],wedge,chi,&DisplY,&DisplZ);
		yt = spotsYZO[nrSp][5]-DisplY;
		zt = spotsYZO[nrSp][6]-DisplZ;
		CorrectForOme(yt,zt,Lsd,spotsYZO[nrSp][4],Wavelength,wedge,&ys,&zs,&Omega);
		SpotsYZOGCorr[nrSp][0] = ys;
		SpotsYZOGCorr[nrSp][1] = zs;
		SpotsYZOGCorr[nrSp][2] = spotsYZO[nrSp][8];
	}
	double **TheorSpotsYZWE;TheorSpotsYZWE=allocMatrix(nTspots,3);
	for (i=0;i<nTspots;i++){
		TheorSpotsYZWE[i][0] = TheorSpots[i][0];
		TheorSpotsYZWE[i][1] = TheorSpots[i][1];
		TheorSpotsYZWE[i][2] = TheorSpots[i][8];
	}
	int sp;
	double PosObs[2], PosTheor[2], Spnr;
	double Error=0;
	for (sp=0;sp<nrMatchedIndexer;sp++){
		PosObs[0] = SpotsYZOGCorr[sp][0];
		PosObs[1] = SpotsYZOGCorr[sp][1];
		Spnr = SpotsYZOGCorr[sp][2];
		for (i=0;i<nTspots;i++){
			if ((int)TheorSpotsYZWE[i][2] == (int)Spnr){
				PosTheor[0] = TheorSpotsYZWE[i][0];
				PosTheor[1] = TheorSpotsYZWE[i][1];
				Error += CalcNorm2((PosObs[0]-PosTheor[0]),(PosObs[1]-PosTheor[1]));
				break;
			}
		}
	}
	FreeMemMatrix(hkls,nhkls);
	FreeMemMatrix(TheorSpots,MaxNSpotsBest);
	FreeMemMatrix(SpotsYZOGCorr,nrMatchedIndexer);
	FreeMemMatrix(TheorSpotsYZWE,nTspots);
	return Error;
}



static double problem_function_PosIni(unsigned n, const double *x, double *grad, void* f_data_trial)
{
	int i;
	struct data_FitPosIni *f_data = (struct data_FitPosIni *) f_data_trial;
	double XIn[n];
	for (i=0;i<n;i++) XIn[i]=x[i];
    
    double error = FitErrorsPosTSingle(XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
        f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges, f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi);
        
    if (f_data->nSpotsComp2 > 0 && f_data->spotsYZO2 != NULL) {
        error += FitErrorsPosTSingle(XIn, f_data->nSpotsComp2, f_data->spotsYZO2, f_data->nhkls, f_data->hkls,
            f_data->Lsd2, f_data->Wavelength2, f_data->nOmeRanges2, f_data->OmegaRanges2, f_data->BoxSizes2, f_data->MinEta2, f_data->wedge2, f_data->chi2);
    }
	return error;
}

void FitPositionIni(double X0[12],int nSpotsComp,double **spotsYZO,int nhkls,double **hkls,double Lsd,
					double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],double BoxSizes[MAXNOMEGARANGES][4],
					double MinEta,double wedge,double chi,
                    int nSpotsComp2, double **spotsYZO2, double Lsd2, double Wavelength2, int nOmeRanges2, double OmegaRanges2[MAXNOMEGARANGES][2], double BoxSizes2[MAXNOMEGARANGES][4], double MinEta2, double wedge2, double chi2,
                    double *XFit,double lb[12],double ub[12])
{
	unsigned n=12;
	double x[n],xl[n],xu[n];
	int i,j;
	struct data_FitPosIni f_data;
	f_data.nSpotsComp = nSpotsComp;
	f_data.spotsYZO = spotsYZO;
	f_data.nhkls = nhkls;
	f_data.hkls = hkls;
	f_data.Lsd = Lsd;
	f_data.Wavelength = Wavelength;
	f_data.nOmeRanges = nOmeRanges;
	for (i=0;i<nOmeRanges;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes[i][j] = BoxSizes[i][j];
	}
	f_data.MinEta = MinEta;
	f_data.wedge = wedge;
	f_data.chi = chi;
    
    // Dataset 2
    f_data.nSpotsComp2 = nSpotsComp2;
    f_data.spotsYZO2 = spotsYZO2;
    f_data.Lsd2 = Lsd2;
    f_data.Wavelength2 = Wavelength2;
    f_data.nOmeRanges2 = nOmeRanges2;
    for (i=0;i<nOmeRanges2;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges2[i][j] = OmegaRanges2[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes2[i][j] = BoxSizes2[i][j];
	}
    f_data.MinEta2 = MinEta2;
    f_data.wedge2 = wedge2;
    f_data.chi2 = chi2;

	for (i=0;i<n;i++){x[i]=X0[i];xl[i]=lb[i];xu[i]=ub[i];}
	struct data_FitPosIni *f_datat;
	f_datat = &f_data;
	void* trp = (struct data_FitPosIni *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_PosIni,trp);
	double minf;
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
    
    opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_PosIni,trp);
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);

	for (i=0;i<n;i++) XFit[i] = x[i];
}

static
double problem_function_OrientIni(unsigned n, const double *x, double *grad, void* f_data_trial)
{
	int i;
	struct data_FitOrientIni *f_data = (struct data_FitOrientIni *) f_data_trial;
	double XIn[n];
	for (i=0;i<n;i++) XIn[i]=x[i];
    
    double error = FitErrorsOrientStrainsSingle(XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
        f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges, f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi, f_data->Pos);
        
    if (f_data->nSpotsComp2 > 0 && f_data->spotsYZO2 != NULL) {
        error += FitErrorsOrientStrainsSingle(XIn, f_data->nSpotsComp2, f_data->spotsYZO2, f_data->nhkls, f_data->hkls,
            f_data->Lsd2, f_data->Wavelength2, f_data->nOmeRanges2, f_data->OmegaRanges2, f_data->BoxSizes2, f_data->MinEta2, f_data->wedge2, f_data->chi2, f_data->Pos);
    }
	return error;
}

void FitOrientIni(double X0[9],int nSpotsComp,double **spotsYZO,int nhkls,double **hkls,double Lsd,
				  double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],double BoxSizes[MAXNOMEGARANGES][4],
				  double MinEta,double wedge,double chi,
                  int nSpotsComp2, double **spotsYZO2, double Lsd2, double Wavelength2, int nOmeRanges2, double OmegaRanges2[MAXNOMEGARANGES][2], double BoxSizes2[MAXNOMEGARANGES][4], double MinEta2, double wedge2, double chi2,
                  double *XFit,double lb[9],double ub[9],double Pos[3])
{
	unsigned n=9;
	double x[n],xl[n],xu[n];
	int i,j;
	struct data_FitOrientIni f_data;
	f_data.nSpotsComp = nSpotsComp;
	f_data.spotsYZO = spotsYZO;
	f_data.nhkls = nhkls;
	f_data.hkls = hkls;
	f_data.Lsd = Lsd;
	f_data.Wavelength = Wavelength;
	f_data.nOmeRanges = nOmeRanges;
	for (i=0;i<nOmeRanges;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes[i][j] = BoxSizes[i][j];
	}
	f_data.MinEta = MinEta;
	f_data.wedge = wedge;
	f_data.chi = chi;
    for (i=0;i<3;i++) f_data.Pos[i] = Pos[i];
    
    // Dataset 2
    f_data.nSpotsComp2 = nSpotsComp2;
    f_data.spotsYZO2 = spotsYZO2;
    f_data.Lsd2 = Lsd2;
    f_data.Wavelength2 = Wavelength2;
    f_data.nOmeRanges2 = nOmeRanges2;
    for (i=0;i<nOmeRanges2;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges2[i][j] = OmegaRanges2[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes2[i][j] = BoxSizes2[i][j];
	}
    f_data.MinEta2 = MinEta2;
    f_data.wedge2 = wedge2;
    f_data.chi2 = chi2;
    
	for (i=0;i<n;i++){x[i]=X0[i];xl[i]=lb[i];xu[i]=ub[i];}
	struct data_FitOrientIni *f_datat;
	f_datat = &f_data;
	void* trp = (struct data_FitOrientIni *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_OrientIni,trp);
	double minf;
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
    
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_OrientIni,trp);
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
    
	for (i=0;i<n;i++) XFit[i] = x[i];
}

static
double problem_function_StrainIni(unsigned n, const double *x, double *grad, void* f_data_trial)
{
	int i;
	struct data_FitStrainIni *f_data = (struct data_FitStrainIni *) f_data_trial;
	double XIn[n];
	for (i=0;i<n;i++) XIn[i]=x[i];
    
    double error = FitErrorsStrainsSingle(XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
        f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges, f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi, f_data->Pos, f_data->Orient);
        
    if (f_data->nSpotsComp2 > 0 && f_data->spotsYZO2 != NULL) {
        error += FitErrorsStrainsSingle(XIn, f_data->nSpotsComp2, f_data->spotsYZO2, f_data->nhkls, f_data->hkls,
            f_data->Lsd2, f_data->Wavelength2, f_data->nOmeRanges2, f_data->OmegaRanges2, f_data->BoxSizes2, f_data->MinEta2, f_data->wedge2, f_data->chi2, f_data->Pos, f_data->Orient);
    }
	return error;
}

void FitStrainIni(double X0[6],int nSpotsComp,double **spotsYZO,int nhkls,double **hkls,double Lsd,
				  double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],double BoxSizes[MAXNOMEGARANGES][4],
				  double MinEta,double wedge, double chi,
                  int nSpotsComp2, double **spotsYZO2, double Lsd2, double Wavelength2, int nOmeRanges2, double OmegaRanges2[MAXNOMEGARANGES][2], double BoxSizes2[MAXNOMEGARANGES][4], double MinEta2, double wedge2, double chi2,
                  double *XFit,double lb[6],double ub[6],
				  double Pos[3],double Orient[3])
{
	unsigned n=6;
	double x[n],xl[n],xu[n];
	int i,j;
	struct data_FitStrainIni f_data;
	f_data.nSpotsComp = nSpotsComp;
	f_data.spotsYZO = spotsYZO;
	f_data.nhkls = nhkls;
	f_data.hkls = hkls;
	f_data.Lsd = Lsd;
	f_data.Wavelength = Wavelength;
	f_data.nOmeRanges = nOmeRanges;
	for (i=0;i<nOmeRanges;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes[i][j] = BoxSizes[i][j];
	}
	f_data.MinEta = MinEta;
	f_data.wedge = wedge;
	f_data.chi = chi;
    for (i=0;i<3;i++) f_data.Pos[i] = Pos[i];
	for (i=0;i<3;i++) f_data.Orient[i] = Orient[i];

    // Dataset 2
    f_data.nSpotsComp2 = nSpotsComp2;
    f_data.spotsYZO2 = spotsYZO2;
    f_data.Lsd2 = Lsd2;
    f_data.Wavelength2 = Wavelength2;
    f_data.nOmeRanges2 = nOmeRanges2;
    for (i=0;i<nOmeRanges2;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges2[i][j] = OmegaRanges2[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes2[i][j] = BoxSizes2[i][j];
	}
    f_data.MinEta2 = MinEta2;
    f_data.wedge2 = wedge2;
    f_data.chi2 = chi2;

	for (i=0;i<n;i++){x[i]=X0[i];xl[i]=lb[i];xu[i]=ub[i];}
	struct data_FitStrainIni *f_datat;
	f_datat = &f_data;
	void* trp = (struct data_FitStrainIni *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_StrainIni,trp);
	double minf;
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
    
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_StrainIni,trp);
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);

	for (i=0;i<n;i++) XFit[i] = x[i];
}

static
double problem_function_Pos(unsigned n, const double *x, double *grad, void* f_data_trial)
{
	int i;
	struct data_FitPos *f_data = (struct data_FitPos *) f_data_trial;
	double XIn[n];
	for (i=0;i<n;i++) XIn[i]=x[i];
    
    double error = FitErrorsPosSecSingle(XIn, f_data->nSpotsComp, f_data->spotsYZO, f_data->nhkls, f_data->hkls,
        f_data->Lsd, f_data->Wavelength, f_data->nOmeRanges, f_data->OmegaRanges, f_data->BoxSizes, f_data->MinEta, f_data->wedge, f_data->chi, f_data->Orient, f_data->Strains);
        
    if (f_data->nSpotsComp2 > 0 && f_data->spotsYZO2 != NULL) {
        error += FitErrorsPosSecSingle(XIn, f_data->nSpotsComp2, f_data->spotsYZO2, f_data->nhkls, f_data->hkls,
            f_data->Lsd2, f_data->Wavelength2, f_data->nOmeRanges2, f_data->OmegaRanges2, f_data->BoxSizes2, f_data->MinEta2, f_data->wedge2, f_data->chi2, f_data->Orient, f_data->Strains);
    }
	return error;
}

void FitPosSec(double X0[3],int nSpotsComp,double **spotsYZO,int nhkls,double **hkls,double Lsd,
				  double Wavelength,int nOmeRanges,double OmegaRanges[MAXNOMEGARANGES][2],double BoxSizes[MAXNOMEGARANGES][4],
				  double MinEta,double wedge,double chi,
                  int nSpotsComp2, double **spotsYZO2, double Lsd2, double Wavelength2, int nOmeRanges2, double OmegaRanges2[MAXNOMEGARANGES][2], double BoxSizes2[MAXNOMEGARANGES][4], double MinEta2, double wedge2, double chi2,
                  double *XFit,double lb[3],double ub[3],
				  double Orient[3],double Strains[6])
{
	unsigned n=3;
	double x[n],xl[n],xu[n];
	int i,j;
	struct data_FitPos f_data;
	f_data.nSpotsComp = nSpotsComp;
	f_data.spotsYZO = spotsYZO;
	f_data.nhkls = nhkls;
	f_data.hkls = hkls;
	f_data.Lsd = Lsd;
	f_data.Wavelength = Wavelength;
	f_data.nOmeRanges = nOmeRanges;
	for (i=0;i<nOmeRanges;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges[i][j] = OmegaRanges[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes[i][j] = BoxSizes[i][j];
	}
	f_data.MinEta = MinEta;
	f_data.wedge = wedge;
	f_data.chi = chi;
    
    // Dataset 2
    f_data.nSpotsComp2 = nSpotsComp2;
    f_data.spotsYZO2 = spotsYZO2;
    f_data.Lsd2 = Lsd2;
    f_data.Wavelength2 = Wavelength2;
    f_data.nOmeRanges2 = nOmeRanges2;
    for (i=0;i<nOmeRanges2;i++){
		for (j=0;j<2;j++) f_data.OmegaRanges2[i][j] = OmegaRanges2[i][j];
		for (j=0;j<4;j++) f_data.BoxSizes2[i][j] = BoxSizes2[i][j];
	}
    f_data.MinEta2 = MinEta2;
    f_data.wedge2 = wedge2;
    f_data.chi2 = chi2;
    
	for (i=0;i<3;i++) f_data.Orient[i] = Orient[i];
	for (i=0;i<6;i++) f_data.Strains[i] = Strains[i];

	for (i=0;i<n;i++){x[i]=X0[i];xl[i]=lb[i];xu[i]=ub[i];}
	struct data_FitPos *f_datat;
	f_datat = &f_data;
	void* trp = (struct data_FitPos *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_Pos,trp);
	double minf;
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
    
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function_Pos,trp);
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);

	for (i=0;i<n;i++) XFit[i] = x[i];
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <foldername> [foldername2]\n", argv[0]);
        return 1;
    }

    char grainsFilePath[256];
    snprintf(grainsFilePath, sizeof(grainsFilePath), "%s/Grains.csv", argv[1]);

    FILE *grainsFile = fopen(grainsFilePath, "r");
    if (grainsFile == NULL) {
        fprintf(stderr, "Error opening grains file: %s\n", grainsFilePath);
        return 1;
    }
    // Skip the first 9 lines (Header)
    char line[1024];
    for (int i = 0; i < 9; i++) {
        fgets(line, sizeof(line), grainsFile);
    }

    // Allocate memory for grainIDs and Initial Parameters
    int *grainIDs = (int *)malloc(MAX_GRAIN_IDS * sizeof(int));
    double **grainParams = allocMatrix(MAX_GRAIN_IDS, 12); // Pos(3) + Euler(3) + Lat(6)
    int grainID_count = 0;

    // Read Grains.csv
    while (fgets(line, sizeof(line), grainsFile)) {
        char *token = strtok(line, " \t,");
        if (token != NULL) {
            grainIDs[grainID_count] = atoi(token);
            for(int k=0; k<12; k++) {
                token = strtok(NULL, " \t,");
                if(token != NULL) grainParams[grainID_count][k] = atof(token);
                else grainParams[grainID_count][k] = 0;
            }
            grainID_count++;
        }
    }
    fclose(grainsFile);

    // MapDatasets reading
    int *mapDatasets = NULL;
    if (argc >= 3) {
        mapDatasets = (int *)malloc(MAX_GRAIN_IDS * MAX_SPOTS * sizeof(int));
        int mapDatasets_count = 0;
        char mapDatasetsFilePath[256];
        snprintf(mapDatasetsFilePath, sizeof(mapDatasetsFilePath), "%s/mapDatasets.txt", argv[1]);
        FILE *mapDatasetsFile = fopen(mapDatasetsFilePath, "r");
        if (mapDatasetsFile != NULL) {
            while (fgets(line, sizeof(line), mapDatasetsFile)) {
                char *token = strtok(line, " \t,");
                if (token != NULL) {
                    mapDatasets[mapDatasets_count++] = atoi(token);
                }
            }
            fclose(mapDatasetsFile);
            mapDatasets = (int *)realloc(mapDatasets, mapDatasets_count * sizeof(int));
        } else {
            // Warn but proceed?
            fprintf(stderr, "Warning: mapDatasets.txt not found, assuming no mapping for dataset 2.\n");
            free(mapDatasets);
            mapDatasets = NULL;
        }
    }

    // Read Parameters
    double Lsd, Wavelength, wedge, MinEta, chi=0;
    double OmegaRanges[MAXNOMEGARANGES][2];
    double BoxSizes[MAXNOMEGARANGES][4];
    int nOmeRanges, nBoxSizes;
    int RingNumbers[200], nRingNumbers;
    double RingRadii[200];
    
    ReadParameters(argv[1], &Lsd, &Wavelength, &wedge, &MinEta, &chi, OmegaRanges, &nOmeRanges, BoxSizes, &nBoxSizes, RingNumbers, &nRingNumbers, RingRadii);
    
    double Lsd2, Wavelength2, wedge2, MinEta2, chi2=0;
    double OmegaRanges2[MAXNOMEGARANGES][2];
    double BoxSizes2[MAXNOMEGARANGES][4];
    int nOmeRanges2, nBoxSizes2;
    int RingNumbers2[200], nRingNumbers2;
    double RingRadii2[200];
    
    int doubleDataset = 0;
    if (argc >= 3) {
        doubleDataset = 1;
        ReadParameters(argv[2], &Lsd2, &Wavelength2, &wedge2, &MinEta2, &chi2, OmegaRanges2, &nOmeRanges2, BoxSizes2, &nBoxSizes2, RingNumbers2, &nRingNumbers2, RingRadii2);
    }

    // Read HKLs
    double **hkls = allocMatrix(MaxNHKLS,7);
    int nhkls = ReadHKLs(argv[1], hkls, RingNumbers, nRingNumbers, RingRadii);

    // Load AllSpots1
    double *AllSpots1;
    int fd1;
    struct stat s1;
    char filename[2048];
    sprintf(filename, "%s/ExtraInfo.bin", argv[1]);
    fd1 = open(filename, O_RDONLY);
    check(fd1 < 0, "open %s failed: %s", filename, strerror(errno));
    fstat(fd1, &s1);
    AllSpots1 = mmap(0, s1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
    check(AllSpots1 == MAP_FAILED, "mmap %s failed: %s", filename, strerror(errno));

    // Load AllSpots2
    double *AllSpots2 = NULL;
    if (doubleDataset) {
        int fd2;
        struct stat s2;
        sprintf(filename, "%s/ExtraInfo.bin", argv[2]);
        fd2 = open(filename, O_RDONLY);
        if (fd2 >= 0) {
            fstat(fd2, &s2);
            AllSpots2 = mmap(0, s2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
            check(AllSpots2 == MAP_FAILED, "mmap %s failed: %s", filename, strerror(errno));
        } else {
            fprintf(stderr, "Warning: Could not open %s/ExtraInfo.bin\n", argv[2]);
            doubleDataset = 0;
        }
    }

    // Read SpotMatrix
    char spotMatrixFilePath[256];
    snprintf(spotMatrixFilePath, sizeof(spotMatrixFilePath), "%s/SpotMatrix.csv", argv[1]);
    FILE *spotMatrixFile = fopen(spotMatrixFilePath, "r");
    if (spotMatrixFile == NULL) {
        fprintf(stderr, "Error opening spot matrix file: %s\n", spotMatrixFilePath);
        return 1;
    }
    fgets(line, sizeof(line), spotMatrixFile); // Skip header
    int *spotIDLocations = (int *)malloc(grainID_count * 2 * sizeof(int));
    memset(spotIDLocations, 0, grainID_count * 2 * sizeof(int));
    int *spotIDs = (int *)malloc(MAX_SPOTS * grainID_count * sizeof(int));
    int totalSpots = 0;
    while (fgets(line, sizeof(line), spotMatrixFile)) {
        char *token = strtok(line, " \t,");
        if (token != NULL) {
            int grainID = atoi(token);
            int grainIndex = -1;
            for (int i = 0; i < grainID_count; i++) {
                if (grainIDs[i] == grainID) {
                    grainIndex = i;
                    break;
                }
            }
            if (grainIndex != -1) {
                if (spotIDLocations[grainIndex * 2 + 1] == 0) {
                    spotIDLocations[grainIndex * 2] = totalSpots;
                }
                spotIDLocations[grainIndex * 2 + 1]++;
            }
            token = strtok(NULL, " \t,");
            if (token != NULL) {
                spotIDs[totalSpots++] = atoi(token);
            }
        }
    }
    fclose(spotMatrixFile);

    // Output
    char outFileName[1024];
    sprintf(outFileName, "%s/Grains_Fitted.csv", argv[1]);
    FILE *fOut = fopen(outFileName, "w");
    fprintf(fOut, "GrainID, PosX, PosY, PosZ, Euler1, Euler2, Euler3, Lat1, Lat2, Lat3, Lat4, Lat5, Lat6, Error\n");

    // Parallel Loop
    int i;
    #pragma omp parallel for private(i) schedule(dynamic)
    for (i = 0; i < grainID_count; i++) {
        int start = spotIDLocations[i * 2];
        int count = spotIDLocations[i * 2 + 1];
        if (count == 0) continue;

        int *localSpotIDs = (int *)malloc(count * sizeof(int));
        memcpy(localSpotIDs, &spotIDs[start], count * sizeof(int));
        
        double **spotsYZO = allocMatrix(count, 9);
        double **spotsYZO2 = NULL;
        int count2 = 0;
        if (doubleDataset && mapDatasets != NULL) {
             // First pass to count valid mappings? No, realloc or alloc max.
             // We can alloc max count.
             spotsYZO2 = allocMatrix(count, 9); 
        }

        // Fill Data
        for (int j = 0; j < count; j++) {
            size_t locationSpotPos = localSpotIDs[j] - 1;
            size_t locationSpotPosAll = locationSpotPos * 14;
            spotsYZO[j][0] = AllSpots1[locationSpotPosAll + 0];
            spotsYZO[j][1] = AllSpots1[locationSpotPosAll + 1];
            spotsYZO[j][2] = AllSpots1[locationSpotPosAll + 2];
            spotsYZO[j][3] = AllSpots1[locationSpotPosAll + 4];
            spotsYZO[j][4] = AllSpots1[locationSpotPosAll + 8];
            spotsYZO[j][5] = AllSpots1[locationSpotPosAll + 9];
            spotsYZO[j][6] = AllSpots1[locationSpotPosAll + 10];
            spotsYZO[j][7] = AllSpots1[locationSpotPosAll + 5];
            spotsYZO[j][8] = j; 

            if (spotsYZO2 != NULL) {
                // Map
                size_t lineNr = mapDatasets[locationSpotPos]; // 1-based index from file
                if (lineNr > 0) {
                    size_t locationSpotPos2 = lineNr - 1;
                    size_t locationSpotPosAll2 = locationSpotPos2 * 14;
                    spotsYZO2[count2][0] = AllSpots2[locationSpotPosAll2 + 0];
                    spotsYZO2[count2][1] = AllSpots2[locationSpotPosAll2 + 1];
                    spotsYZO2[count2][2] = AllSpots2[locationSpotPosAll2 + 2];
                    spotsYZO2[count2][3] = AllSpots2[locationSpotPosAll2 + 4];
                    spotsYZO2[count2][4] = AllSpots2[locationSpotPosAll2 + 8];
                    spotsYZO2[count2][5] = AllSpots2[locationSpotPosAll2 + 9];
                    spotsYZO2[count2][6] = AllSpots2[locationSpotPosAll2 + 10];
                    spotsYZO2[count2][7] = AllSpots2[locationSpotPosAll2 + 5];
                    spotsYZO2[count2][8] = count2;
                    count2++;
                }
            }
        }

        // Fitting Logic
        double XFit[12];
        double lb[12], ub[12];
        double IniPos[12]; // 0-2 Pos, 3-5 Euler, 6-11 Lat
        
        // Initial Guess
        for(int k=0; k<12; k++) IniPos[k] = grainParams[i][k];
        
        double Rsample = 1.0; // Hardcoded or read? OMP reads Rsample.
        // Bounds
        for (int k=0; k<3; k++) { lb[k] = IniPos[k] - Rsample; ub[k] = IniPos[k] + Rsample; }
        for (int k=3; k<6; k++) { lb[k] = IniPos[k] - 5; ub[k] = IniPos[k] + 5; } // +/- 5 deg
        for (int k=6; k<12; k++) { lb[k] = IniPos[k] - 0.1; ub[k] = IniPos[k] + 0.1; } // Lattice +/- 0.1?

        // 1. Fit Init
        FitPositionIni(IniPos, count, spotsYZO, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                       count2, spotsYZO2, Lsd2, Wavelength2, nOmeRanges2, OmegaRanges2, BoxSizes2, MinEta2, wedge2, chi2,
                       XFit, lb, ub);
                       
        // 2. Fit Orient
        double X0_Orient[9];
        double lb_Orient[9], ub_Orient[9];
        // Pos fixed from XFit
        double FitPos[3] = {XFit[0], XFit[1], XFit[2]};
        // Init form XFit
        for(int k=0; k<3; k++) X0_Orient[k] = XFit[3+k]; // Euler
        for(int k=0; k<6; k++) X0_Orient[3+k] = XFit[6+k]; // Lat
        for(int k=0; k<3; k++) { lb_Orient[k] = X0_Orient[k]-2; ub_Orient[k] = X0_Orient[k]+2; }
        for(int k=3; k<9; k++) { lb_Orient[k] = X0_Orient[k]-0.05; ub_Orient[k] = X0_Orient[k]+0.05; }
        
        FitOrientIni(X0_Orient, count, spotsYZO, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                     count2, spotsYZO2, Lsd2, Wavelength2, nOmeRanges2, OmegaRanges2, BoxSizes2, MinEta2, wedge2, chi2,
                     XFit, lb_Orient, ub_Orient, FitPos); // Reusing XFit array (size 12, here writing 9)
                     
        // XFit now has 0-2 Euler, 3-8 Lat.
        double FitEuler[3] = {XFit[0], XFit[1], XFit[2]};
        double FitLat[6]; for(int k=0; k<6; k++) FitLat[k] = XFit[3+k];
        
        // 3. Fit Strain
        double X0_Strain[6];
        double lb_Strain[6], ub_Strain[6];
        for(int k=0; k<6; k++) X0_Strain[k] = FitLat[k];
        for(int k=0; k<6; k++) { lb_Strain[k] = X0_Strain[k]-0.02; ub_Strain[k] = X0_Strain[k]+0.02; }
        
        FitStrainIni(X0_Strain, count, spotsYZO, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                     count2, spotsYZO2, Lsd2, Wavelength2, nOmeRanges2, OmegaRanges2, BoxSizes2, MinEta2, wedge2, chi2,
                     XFit, lb_Strain, ub_Strain, FitPos, FitEuler);
        
        // XFit has 0-5 Lat.
        for(int k=0; k<6; k++) FitLat[k] = XFit[k];
        
        // 4. Fit Pos Final
        double X0_Pos[3];
        double lb_Pos[3], ub_Pos[3];
        for(int k=0; k<3; k++) X0_Pos[k] = FitPos[k];
        for(int k=0; k<3; k++) { lb_Pos[k] = X0_Pos[k]-0.5; ub_Pos[k] = X0_Pos[k]+0.5; }
        
        FitPosSec(X0_Pos, count, spotsYZO, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi,
                  count2, spotsYZO2, Lsd2, Wavelength2, nOmeRanges2, OmegaRanges2, BoxSizes2, MinEta2, wedge2, chi2,
                  XFit, lb_Pos, ub_Pos, FitEuler, FitLat);
                  
        // Final Pos
        for(int k=0; k<3; k++) FitPos[k] = XFit[k];
        
        // Calculate Final Error (using helper function or just storing 0 for now? OMP code calculates it).
        // I will calculate it using FitErrorsPosSecSingle sum.
        double FinalError = FitErrorsPosSecSingle(FitPos, count, spotsYZO, nhkls, hkls, Lsd, Wavelength, nOmeRanges, OmegaRanges, BoxSizes, MinEta, wedge, chi, FitEuler, FitLat);
        if (doubleDataset && spotsYZO2 != NULL) {
             FinalError += FitErrorsPosSecSingle(FitPos, count2, spotsYZO2, nhkls, hkls, Lsd2, Wavelength2, nOmeRanges2, OmegaRanges2, BoxSizes2, MinEta2, wedge2, chi2, FitEuler, FitLat);
        }
        
        #pragma omp critical
        {
            fprintf(fOut, "%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", 
                grainIDs[i], FitPos[0], FitPos[1], FitPos[2], FitEuler[0], FitEuler[1], FitEuler[2],
                FitLat[0], FitLat[1], FitLat[2], FitLat[3], FitLat[4], FitLat[5], FinalError);
        }

        FreeMemMatrix(spotsYZO, count);
        if (spotsYZO2 != NULL) FreeMemMatrix(spotsYZO2, count); // count? No, FreeMemMatrix takes Rows. Alloc was size `count`.
        free(localSpotIDs);
    }
    fclose(fOut);
    return 0;
}

