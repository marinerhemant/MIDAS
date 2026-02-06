//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// 04/26/2021
// Hemant Sharma
// OpenMP version of IndexerLinuxArgsShm code.

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

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

#define RealType double

// conversions constants
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

// max array sizes
#define MAX_N_SPOTS 100000000
#define MAX_N_OR 7200
#define MAX_N_MATCHES 1
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS 16
#define N_COL_OBSSPOTS 10
#define N_COL_GRAINSPOTS 17
#define N_COL_GRAINMATCHES 16
#define MAX_MIC_ROWS 50000000
#define EPS 1e-9

// Globals
RealType *ObsSpotsLab;
struct TParams {
  int RingNumbers[MAX_N_RINGS];
  int SpaceGroupNum;
  RealType LatticeConstant;
  RealType Wavelength;
  RealType Distance;
  RealType Rsample;
  RealType Hbeam;
  RealType StepsizePos;
  RealType StepsizeOrient;
  int NrOfRings;
  RealType RingRadii[MAX_N_RINGS];
  RealType RingRadiiUser[MAX_N_RINGS];
  RealType MarginOme;
  RealType MarginEta;
  RealType MarginRad;
  RealType MarginRadial;
  RealType EtaBinSize;
  RealType OmeBinSize;
  RealType InvEtaBinSize;
  RealType InvOmeBinSize;
  RealType ExcludePoleAngle;
  RealType MinMatchesToAcceptFrac;
  RealType BoxSizes[MAX_N_OMEGARANGES][4];
  RealType OmegaRanges[MAX_N_OMEGARANGES][2];
  char OutputFolder[4096];
  char MicFN[4096];
  char GrainsFN[4096];
  int NoOfOmegaRanges;
  char SpotsFileName[4096];
  char IDsFileName[4096];
  int UseFriedelPairs;
  int RingsToReject[MAX_N_RINGS];
  int nRingsToRejectCalc;
  int RingToIndex;
};

size_t n_spots = 0;

// hkls to use
double hkls[MAX_N_HKLS][10];
int n_hkls = 0;
int HKLints[MAX_N_HKLS][4];
double ABCABG[6];
//~ RealType RingTtheta[MAX_N_RINGS];
//~ int   RingMult[MAX_N_RINGS];
double RingHKL[MAX_N_RINGS][3];

// For detector mapping!
int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;
double BeamSize;
int numScans;
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

RealType omemargins[181];
RealType etamargins[MAX_N_RINGS];

size_t *data;
size_t *ndata;
int SGNum;
double *grid;
double *ypos;

// the number of elements of the data arrays above
int n_ring_bins;
int n_eta_bins;
int n_ome_bins;

// the binsizes used for the binning
RealType EtaBinSize = 0;
RealType OmeBinSize = 0;

// some macros for math calculations
#define crossProduct(a, b, c)                                                  \
  (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];                                  \
  (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];                                  \
  (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

#define dot(v, q) ((v)[0] * (q)[0] + (v)[1] * (q)[1] + (v)[2] * (q)[2])

#define CalcLength(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

void FindInMatrix(RealType *aMatrixp, int nrows, int ncols, int SearchColumn,
                  RealType aVal, int *idx) {
  int r, LIndex;
  *idx = -1;
  for (r = 0; r < nrows; r++) {
    LIndex = (r * ncols) + SearchColumn;
    if (aMatrixp[LIndex] == aVal) {
      *idx = r;
      break;
    }
  }
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
  Euler[0] = psi;
  Euler[1] = phi;
  Euler[2] = theta;
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

RealType **allocMatrix(int nrows, int ncols) {
  RealType **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  RealType *block = malloc(nrows * ncols * sizeof(RealType));
  if (block == NULL) {
    free(arr);
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = &block[i * ncols];
  }
  return arr;
}

void FreeMemMatrix(RealType **mat, int nrows) {
  if (mat != NULL) {
    if (mat[0] != NULL) {
      free(mat[0]);
    }
    free(mat);
  }
}

void MatrixMultF33(RealType m[3][3], RealType n[3][3], RealType res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

void MatrixMultF(RealType m[3][3], RealType v[3], RealType r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

void MatrixMult(RealType m[3][3], int v[3], RealType r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

RealType min(RealType a, RealType b) { return (a < b ? a : b); }

RealType max(RealType a, RealType b) { return (a > b ? a : b); }

void CalcInternalAngle(RealType x1, RealType y1, RealType z1, RealType x2,
                       RealType y2, RealType z2, RealType *ia) {
  RealType v1[3];
  RealType v2[3];
  v1[0] = x1;
  v1[1] = y1;
  v1[2] = z1;
  v2[0] = x2;
  v2[1] = y2;
  v2[2] = z2;
  RealType l1 = CalcLength(x1, y1, z1);
  RealType l2 = CalcLength(x2, y2, z2);
  RealType tmp = dot(v1, v2) / (l1 * l2);
  if (tmp > 1) {
    tmp = 1;
  }
  if (tmp < -1) {
    tmp = -1;
  }
  *ia = rad2deg * acos(tmp);
}

void RotateAroundZ(RealType v1[3], RealType alpha, RealType v2[3]) {
  RealType cosa = cos(alpha * deg2rad);
  RealType sina = sin(alpha * deg2rad);
  RealType mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};
  MatrixMultF(mat, v1, v2);
}

void CalcEtaAngle(RealType y, RealType z, RealType *alpha) {
  *alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    *alpha = -*alpha;
}

void CalcSpotPosition(RealType RingRadius, RealType eta, RealType *yl,
                      RealType *zl) {
  RealType etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

void CalcOmega(RealType x, RealType y, RealType z, RealType v, RealType vSq,
               RealType omegas[4], RealType etas[4], RealType cosOmes[4],
               RealType sinOmes[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  // RealType len = sqrt(x * x + y * y + z * z);
  // RealType v = sinTheta * len;
  RealType almostzero = 1e-4;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1 <= 1)) {
        ome = acos(cosome1) * rad2deg;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        omegas[*nsol] = ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = sinome1;
        *nsol = *nsol + 1;

        omegas[*nsol] = -ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = -sinome1;
        *nsol = *nsol + 1;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType inv_y2 = 1.0 / y2;
    RealType a = 1 + ((x * x) * inv_y2);
    RealType b = (2 * v * x) * inv_y2;
    RealType c = (vSq * inv_y2) - 1;
    RealType discr = b * b - 4 * a * c;
    RealType ome1a;
    RealType ome1b;
    RealType ome2a;
    RealType ome2b;
    RealType cosome1;
    RealType cosome2;
    RealType eqa, eqb, diffa, diffb;
    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1) {
        ome1a = acos(cosome1);
        ome1b = -ome1a;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        eqa = -x * cosome1 + y * sinome1;
        diffa = fabs(eqa - v);
        eqb = -x * cosome1 - y * sinome1;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome1a * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = sinome1;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome1b * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = -sinome1;
          *nsol = *nsol + 1;
        }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2);
        ome2b = -ome2a;
        RealType sinome2 = sqrt(1 - cosome2 * cosome2);
        eqa = -x * cosome2 + y * sinome2;
        diffa = fabs(eqa - v);
        eqb = -x * cosome2 - y * sinome2;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome2a * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = sinome2;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome2b * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = -sinome2;
          *nsol = *nsol + 1;
        }
      }
    }
  }
  RealType gw[3];
  RealType gv[3] = {x, y, z};
  RealType eta;
  int indexOme;
  for (indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

void CalcDiffrSpots_Furnace(RealType OrientMatrix[3][3],
                            RealType LatticeConstant, RealType Wavelength,
                            RealType distance, RealType RingRadii[],
                            RealType OmegaRange[][2], RealType BoxSizes[][4],
                            int NOmegaRanges, RealType ExcludePoleAngle,
                            RealType **spots, int *nspots) {
  int i, OmegaRangeNo;
  RealType ds;
  RealType theta;
  int KeepSpot;
  double Ghkl[3];
  int indexhkl;
  RealType Gc[3];
  RealType omegas[4];
  RealType cosOmes[4];
  RealType sinOmes[4];
  RealType etas[4];
  RealType yl;
  RealType zl;
  int nspotsPlane;
  int spotnr = 0;
  int spotid = 0;
  int OrientID = 0;
  int ringnr = 0;
  int YCInt, ZCInt;
  long long int idx;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    ringnr = (int)(hkls[indexhkl][3]);
    RealType RingRadius = RingRadii[ringnr];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    // ds    = hkls[indexhkl][4];
    theta = hkls[indexhkl][5];
    CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl][8], hkls[indexhkl][9], omegas,
              etas, cosOmes, sinOmes, &nspotsPlane);
    for (i = 0; i < nspotsPlane; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      RealType EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle))
        continue;
      CalcSpotPosition(RingRadius, etas[i], &(yl), &(zl));
      for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
        KeepSpot = 0;
        if ((Omega > OmegaRange[OmegaRangeNo][0]) &&
            (Omega < OmegaRange[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) &&
            (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) &&
            (zl < BoxSizes[OmegaRangeNo][3])) {
          KeepSpot = 1;
          break;
        }
      }
      if (BigDetSize != 0 && KeepSpot == 1) {
        YCInt = (int)floor((BigDetSize / 2) - (-yl / pixelsize));
        ZCInt = (int)floor(((zl / pixelsize + (BigDetSize / 2))));
        idx = (long long int)(YCInt + BigDetSize * ZCInt);
        if (!TestBit(BigDetector, idx)) {
          KeepSpot = 0;
        }
      }
      if (KeepSpot) {
        spots[spotnr][0] = OrientID;
        spots[spotnr][1] = spotid;
        spots[spotnr][2] = indexhkl;
        spots[spotnr][3] = distance;
        spots[spotnr][4] = yl;
        spots[spotnr][5] = zl;
        spots[spotnr][6] = omegas[i];
        spots[spotnr][7] = etas[i];
        spots[spotnr][8] = theta;
        spots[spotnr][9] = ringnr;
        spots[spotnr][14] = sinOmes[i];
        spots[spotnr][15] = cosOmes[i];
        spotnr++;
        spotid++;
      }
    }
  }
  *nspots = spotnr;
}

void CompareSpots(RealType **TheorSpots, int nTheorSpots, RealType RefRad,
                  RealType MarginOme, RealType MarginRad, RealType MarginRadial,
                  RealType etamargins[], RealType omemargins[], int *nMatch,
                  RealType **GrainSpots, RealType xThis, RealType yThis,
                  struct TParams *Params) {
  int nMatched = 0;
  int nNonMatched = 0;
  int sp;
  int RingNr;
  int iOme, iEta;
  size_t spotRow, spotRowBest;
  int MatchFound;
  RealType diffOme;
  RealType diffOmeBest;
  size_t iRing, iPos;
  size_t iSpot;
  size_t scannrobs, scannrtheor;
  RealType etamargin, omemargin, yRot, ySpot;
  for (sp = 0; sp < nTheorSpots; sp++) {
    RingNr = (int)TheorSpots[sp][9];
    iRing = RingNr - 1;
    // iEta = floor((180 + TheorSpots[sp][12]) / EtaBinSize);
    // iOme = floor((180 + TheorSpots[sp][6]) / OmeBinSize);
    iEta = floor((180 + TheorSpots[sp][12]) * Params->InvEtaBinSize);
    iOme = floor((180 + TheorSpots[sp][6]) * Params->InvOmeBinSize);
    // yRot = xThis * sin(deg2rad * TheorSpots[sp][6]) +
    //       yThis * cos(deg2rad * TheorSpots[sp][6]);
    yRot = xThis * TheorSpots[sp][14] + yThis * TheorSpots[sp][15];
    //~ printf("%lf %lf %lf %lf\n",yRot,xThis,yThis,TheorSpots[sp][6]);
    // use iOme, xpos and yPos to calculate scanNr
    etamargin = etamargins[RingNr];
    omemargin = omemargins[(int)floor(fabs(TheorSpots[sp][12]))];
    MatchFound = 0;
    diffOmeBest = MarginOme + 0.00001;
    size_t Pos = iRing;
    Pos *= n_eta_bins;
    Pos *= n_ome_bins;
    Pos += iEta * n_ome_bins;
    Pos += iOme;
    size_t nspots = ndata[Pos * 2];
    size_t DataPos = ndata[Pos * 2 + 1];
    for (iSpot = 0; iSpot < nspots; iSpot++) {
      spotRow = data[(DataPos + iSpot) * 2 + 0];
      scannrobs = data[(DataPos + iSpot) * 2 + 1];
      ySpot = ypos[scannrobs];
      if (fabs(yRot - ySpot) < BeamSize / 2) {
        if (fabs(TheorSpots[sp][13] - ObsSpotsLab[spotRow * 10 + 8]) <
            MarginRadial) {
          if (fabs(TheorSpots[sp][12] - ObsSpotsLab[spotRow * 10 + 6]) <
              etamargin) {
            diffOme = fabs(TheorSpots[sp][6] - ObsSpotsLab[spotRow * 10 + 2]);
            if (diffOme < diffOmeBest) {
              // printf("%lf %zu %zu %zu %zu %lf %lf %lf %lf %d %d %d %lf %lf
              // %lf %lf\n", 	diffOme,DataPos,nspots,iSpot,(DataPos +
              // iSpot)*2+0,ySpot,yRot, 	fabs(yRot-ySpot),
              // BeamSize/2,scannrobs,spotRow,spotRow*10+8,
              // 	TheorSpots[sp][13],ObsSpotsLab[spotRow*10+8],TheorSpots[sp][6],
              // 	ObsSpotsLab[spotRow*10+2]);
              // fflush(stdout);
              diffOmeBest = diffOme;
              spotRowBest = spotRow;
              MatchFound = 1;
            }
          }
        }
      }
    }
    if (MatchFound == 1) {
      GrainSpots[nMatched][0] = nMatched;
      GrainSpots[nMatched][1] = 999.0;
      GrainSpots[nMatched][2] = TheorSpots[sp][10];
      GrainSpots[nMatched][3] = ObsSpotsLab[spotRowBest * 10 + 0];
      GrainSpots[nMatched][4] =
          ObsSpotsLab[spotRowBest * 10 + 0] - TheorSpots[sp][10];
      GrainSpots[nMatched][5] = TheorSpots[sp][11];
      GrainSpots[nMatched][6] = ObsSpotsLab[spotRowBest * 10 + 1];
      GrainSpots[nMatched][7] =
          ObsSpotsLab[spotRowBest * 10 + 1] - TheorSpots[sp][11];
      GrainSpots[nMatched][8] = TheorSpots[sp][6];
      GrainSpots[nMatched][9] = ObsSpotsLab[spotRowBest * 10 + 2];
      GrainSpots[nMatched][10] =
          ObsSpotsLab[spotRowBest * 10 + 2] - TheorSpots[sp][6];
      GrainSpots[nMatched][11] = RefRad;
      GrainSpots[nMatched][12] = ObsSpotsLab[spotRowBest * 10 + 3];
      GrainSpots[nMatched][13] = ObsSpotsLab[spotRowBest * 10 + 3] - RefRad;
      GrainSpots[nMatched][14] = ObsSpotsLab[spotRowBest * 10 + 4];
      nMatched++;
    } else {
      nNonMatched++;
      int idx = nTheorSpots - nNonMatched;
      GrainSpots[idx][0] = -nNonMatched;
      GrainSpots[idx][1] = 999.0;
      GrainSpots[idx][2] = TheorSpots[sp][10];
      GrainSpots[idx][3] = 0;
      GrainSpots[idx][4] = 0;
      GrainSpots[idx][5] = TheorSpots[sp][11];
      GrainSpots[idx][6] = 0;
      GrainSpots[idx][7] = 0;
      GrainSpots[idx][8] = TheorSpots[sp][6];
      GrainSpots[idx][9] = 0;
      GrainSpots[idx][10] = 0;
      GrainSpots[idx][11] = 0;
      GrainSpots[idx][12] = 0;
      GrainSpots[idx][13] = 0;
      GrainSpots[idx][14] = 0;
    }
  }
  *nMatch = nMatched;
}

void AxisAngle2RotMatrix(RealType axis[3], RealType angle, RealType R[3][3]) {
  if ((axis[0] == 0) && (axis[1] == 0) && (axis[2] == 0)) {
    R[0][0] = 1;
    R[1][0] = 0;
    R[2][0] = 0;
    R[0][1] = 0;
    R[1][1] = 1;
    R[2][1] = 0;
    R[0][2] = 0;
    R[1][2] = 0;
    R[2][2] = 1;
    return;
  }
  RealType lenInv =
      1 / sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  RealType u = axis[0] * lenInv;
  RealType v = axis[1] * lenInv;
  RealType w = axis[2] * lenInv;
  RealType angleRad = deg2rad * angle;
  RealType rcos = cos(angleRad);
  RealType rsin = sin(angleRad);
  R[0][0] = rcos + u * u * (1 - rcos);
  R[1][0] = w * rsin + v * u * (1 - rcos);
  R[2][0] = -v * rsin + w * u * (1 - rcos);
  R[0][1] = -w * rsin + u * v * (1 - rcos);
  R[1][1] = rcos + v * v * (1 - rcos);
  R[2][1] = u * rsin + w * v * (1 - rcos);
  R[0][2] = v * rsin + u * w * (1 - rcos);
  R[1][2] = -u * rsin + v * w * (1 - rcos);
  R[2][2] = rcos + w * w * (1 - rcos);
}

double CalcRotationAngle(int RingNr) {
  int habs, kabs, labs;
  int i;
  for (i = 0; i < MAX_N_HKLS; i++) {
    if (HKLints[i][3] == RingNr) {
      habs = abs(HKLints[i][0]);
      kabs = abs(HKLints[i][1]);
      labs = abs(HKLints[i][2]);
      break;
    }
  }
  int nzeros = 0;
  if (habs == 0)
    nzeros++;
  if (kabs == 0)
    nzeros++;
  if (labs == 0)
    nzeros++;
  if (nzeros == 3)
    return 0;
  if (SGNum == 1 || SGNum == 2) { // Triclinic
    return 360;
  } else if (SGNum >= 3 && SGNum <= 15) { // Monoclinic
    if (nzeros != 2)
      return 360;
    else if (ABCABG[3] == 90 && ABCABG[4] == 90 && labs != 0) {
      return 180;
    } else if (ABCABG[3] == 90 && ABCABG[5] == 90 && habs != 0) {
      return 180;
    } else if (ABCABG[3] == 90 && ABCABG[5] == 90 && kabs != 0) {
      return 180;
    } else
      return 360;
  } else if (SGNum >= 16 && SGNum <= 74) { // Orthorhombic
    if (nzeros != 2)
      return 360;
    else
      return 180;
  } else if (SGNum >= 75 && SGNum <= 142) { // Tetragonal
    if (nzeros == 0)
      return 360;
    else if (nzeros == 1 && labs == 0 && habs == kabs) {
      return 180;
    } else if (nzeros == 2) {
      if (labs == 0) {
        return 180;
      } else {
        return 90;
      }
    } else
      return 360;
  } else if (SGNum >= 143 && SGNum <= 167) { // Trigonal
    if (nzeros == 0)
      return 360;
    else if (nzeros == 2 && labs != 0)
      return 120;
    else
      return 360;
  } else if (SGNum >= 168 && SGNum <= 194) { // Hexagonal
    if (nzeros == 2 && labs != 0)
      return 60;
    else
      return 360;
  } else if (SGNum >= 195 && SGNum <= 230) { // Cubic
    if (nzeros == 2)
      return 90;
    else if (nzeros == 1) {
      if (habs == kabs || kabs == labs || habs == labs)
        return 180;
    } else if (habs == kabs && kabs == labs)
      return 120;
    else
      return 360;
  } else
    return 0;
}

int GenerateCandidateOrientationsF(double hkl[3], RealType hklnormal[3],
                                   RealType stepsize, RealType OrMat[][3][3],
                                   int *nOrient, int RingNr) {
  RealType v[3];
  RealType MaxAngle = 0;
  crossProduct(v, hkl, hklnormal);
  RealType hkllen = sqrt(hkl[0] * hkl[0] + hkl[1] * hkl[1] + hkl[2] * hkl[2]);
  RealType hklnormallen =
      sqrt(hklnormal[0] * hklnormal[0] + hklnormal[1] * hklnormal[1] +
           hklnormal[2] * hklnormal[2]);
  RealType dotpr = dot(hkl, hklnormal);
  RealType angled = rad2deg * acos(dotpr / (hkllen * hklnormallen));
  RealType RotMat[3][3];
  RealType RotMat2[3][3];
  RealType RotMat3[3][3];
  AxisAngle2RotMatrix(v, angled, RotMat);
  MaxAngle = CalcRotationAngle(RingNr);
  RealType nsteps = (MaxAngle / stepsize);
  int nstepsi = (int)nsteps;
  int or;
  int row, col;
  RealType angle2;
  for (or = 0; or < nstepsi; or++) {
    angle2 = or * stepsize;
    AxisAngle2RotMatrix(hklnormal, angle2, RotMat2);
    MatrixMultF33(RotMat2, RotMat, RotMat3);
    for (row = 0; row < 3; row++) {
      for (col = 0; col < 3; col++) {
        OrMat[or][row][col] = RotMat3[row][col];
      }
    }
  }
  *nOrient = nstepsi;
  return 0;
}

void displacement_spot_needed_COM(RealType a, RealType b, RealType c,
                                  RealType xi, RealType yi, RealType zi,
                                  RealType sinOme, RealType cosOme,
                                  RealType *Displ_y, RealType *Displ_z) {
  RealType lenInv = 1 / sqrt(xi * xi + yi * yi + zi * zi);
  xi = xi * lenInv;
  yi = yi * lenInv;
  zi = zi * lenInv;
  // RealType OmegaRad = deg2rad * omega;
  // RealType sinOme = sin(OmegaRad);
  // RealType cosOme = cos(OmegaRad);
  RealType t = (a * cosOme - b * sinOme) / xi;
  *Displ_y = ((a * sinOme) + (b * cosOme)) - (t * yi);
  *Displ_z = c - t * zi;
}

void spot_to_gv(RealType xi, RealType yi, RealType zi, RealType Omega,
                RealType *g1, RealType *g2, RealType *g3) {
  RealType len = sqrt(xi * xi + yi * yi + zi * zi);
  if (len == 0) {
    *g1 = 0;
    *g2 = 0;
    *g3 = 0;
    printf("len o!\n");
    return;
  }
  RealType xn = xi / len;
  RealType yn = yi / len;
  RealType zn = zi / len;
  RealType g1r = (-1 + xn);
  RealType g2r = yn;
  RealType CosOme = cos(-Omega * deg2rad);
  RealType SinOme = sin(-Omega * deg2rad);
  *g1 = g1r * CosOme - g2r * SinOme;
  *g2 = g1r * SinOme + g2r * CosOme;
  *g3 = zn;
}

void spot_to_gv_pos(RealType xi, RealType yi, RealType zi, RealType Omega,
                    RealType cx, RealType cy, RealType cz, RealType *g1,
                    RealType *g2, RealType *g3) {
  RealType v[3], vr[3];
  v[0] = cx;
  v[1] = cy;
  v[2] = cz;
  RotateAroundZ(v, Omega, vr);
  xi = xi - vr[0];
  yi = yi - vr[1];
  zi = zi - vr[2];
  spot_to_gv(xi, yi, zi, Omega, g1, g2, g3);
}

int AddUnique(int *arr, int *n, int val) {
  int i;
  for (i = 0; i < *n; ++i) {
    if (arr[i] == val) {
      return 0;
    }
  }
  arr[*n] = val;
  (*n)++;
  return 1;
}

void MakeUnitLength(RealType x, RealType y, RealType z, RealType *xu,
                    RealType *yu, RealType *zu) {
  RealType len = CalcLength(x, y, z);
  if (len == 0) {
    *xu = 0;
    *yu = 0;
    *zu = 0;
    return;
  }
  *xu = x / len;
  *yu = y / len;
  *zu = z / len;
}

struct IndexerScratch {
  RealType **GrainMatches;
  RealType **GrainMatchesT;
  RealType **AllGrainSpots;
  RealType **AllGrainSpotsT;
  RealType **GrainSpots;
  RealType **TheorSpots;
  RealType *IAgrainspots;
};

size_t ReadBigDet(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/BigDetectorMask.bin", cwd);
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  BigDetector = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(BigDetector == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  return size;
}

int ReadParams(char FileName[], struct TParams *Params) {
#define MAX_LINE_LENGTH 4096
  sprintf(Params->MicFN, "0");
  sprintf(Params->GrainsFN, "0");
  FILE *fp;
  char line[MAX_LINE_LENGTH];
  char dummy[MAX_LINE_LENGTH];
  char *str;
  int NrOfBoxSizes = 0;
  int cmpres;
  int NoRingNumbers = 0;
  Params->NrOfRings = 0;
  Params->NoOfOmegaRanges = 0;
  fp = fopen(FileName, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s.\n", FileName);
    return (1);
  }
  totNrPixelsBigDetector = 0;
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
    str = "RingNumbers ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->RingNumbers[NoRingNumbers]));
      NoRingNumbers++;
      continue;
    }
    str = "RingToIndex ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->RingToIndex));
      NoRingNumbers++;
      continue;
    }
    str = "BigDetSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &BigDetSize);
      totNrPixelsBigDetector = BigDetSize;
      totNrPixelsBigDetector *= BigDetSize;
      totNrPixelsBigDetector /= 32;
      totNrPixelsBigDetector++;
      continue;
    }
    str = "px ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &pixelsize);
      continue;
    }
    str = "BeamSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &BeamSize);
      BeamSize += 0.1;
      continue;
    }
    str = "SpaceGroup ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->SpaceGroupNum));
      SGNum = Params->SpaceGroupNum;
      continue;
    }
    str = "LatticeParameter ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->LatticeConstant));
      sscanf(line, "%s %lf %lf %lf %lf %lf %lf", dummy, &ABCABG[0], &ABCABG[1],
             &ABCABG[2], &ABCABG[3], &ABCABG[4], &ABCABG[5]);
      continue;
    }
    str = "Wavelength ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Wavelength));
      continue;
    }
    str = "Distance ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Distance));
      continue;
    }
    str = "Rsample ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Rsample));
      continue;
    }
    str = "Hbeam ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->Hbeam));
      continue;
    }
    str = "StepsizePos ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->StepsizePos));
      continue;
    }
    str = "StepsizeOrient ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->StepsizeOrient));
      continue;
    }
    str = "MarginOme ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginOme));
      continue;
    }
    str = "MarginRadius ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginRad));
      continue;
    }
    str = "MarginRadial ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginRadial));
      continue;
    }
    str = "EtaBinSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->EtaBinSize));
      continue;
    }
    str = "OmeBinSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->OmeBinSize));
      continue;
    }
    str = "MinMatchesToAcceptFrac ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MinMatchesToAcceptFrac));
      continue;
    }
    str = "ExcludePoleAngle ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->ExcludePoleAngle));
      continue;
    }
    str = "RingRadii ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy,
             &(Params->RingRadiiUser[Params->NrOfRings]));
      Params->NrOfRings = Params->NrOfRings + 1;
      continue;
    }
    str = "OmegaRange ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf %lf", dummy,
             &(Params->OmegaRanges[Params->NoOfOmegaRanges][0]),
             &(Params->OmegaRanges[Params->NoOfOmegaRanges][1]));
      (Params->NoOfOmegaRanges)++;
      continue;
    }
    str = "BoxSize ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf %lf %lf %lf", dummy,
             &(Params->BoxSizes[NrOfBoxSizes][0]),
             &(Params->BoxSizes[NrOfBoxSizes][1]),
             &(Params->BoxSizes[NrOfBoxSizes][2]),
             &(Params->BoxSizes[NrOfBoxSizes][3]));
      NrOfBoxSizes++;
      continue;
    }
    str = "SpotsFileName ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->SpotsFileName);
      continue;
    }
    str = "IDsFileName ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->IDsFileName);
      continue;
    }
    str = "MicFile ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->MicFN);
      continue;
    }
    str = "GrainsFile";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->GrainsFN);
      continue;
    }
    str = "MarginEta ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %lf", dummy, &(Params->MarginEta));
      continue;
    }
    str = "UseFriedelPairs ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->UseFriedelPairs));
      continue;
    }
    str = "OutputFolder ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->OutputFolder);
      continue;
    }
    str = "";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      continue;
    }
    printf("Warning: skipping line in parameters file:\n");
    printf("%s\n", line);
  }
  if (totNrPixelsBigDetector != 0) {
    char *cwd = dirname(Params->OutputFolder);
    size_t sz = ReadBigDet(cwd);
  }
  int i;
  for (i = 0; i < MAX_N_RINGS; i++)
    Params->RingRadii[i] = 0;
  for (i = 0; i < Params->NrOfRings; i++)
    Params->RingRadii[Params->RingNumbers[i]] = Params->RingRadiiUser[i];
  return (0);
}

RealType CalcAvgIA(RealType *Arr, int n) {
  RealType total = 0;
  int nnum = 0;
  int i;
  for (i = 0; i < n; i++) {
    if (Arr[i] == 999)
      continue;
    total = total + fabs(Arr[i]);
    nnum++;
  }
  if (nnum == 0)
    return 0;
  else
    return total / nnum;
}

void CalcIA(RealType **GrainMatches, int ngrains, RealType **AllGrainSpots,
            RealType distance, struct IndexerScratch *scratch) {
  RealType *IAgrainspots = scratch->IAgrainspots;
  int r, g;
  RealType g1x, g1y, g1z;
  RealType x1, y1, z1, w1, x2, y2, z2, w2, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
  int nspots;
  int rt = 0;
  // IAgrainspots allocated in scratch
  for (g = 0; g < ngrains; g++) {
    nspots = GrainMatches[g][12];
    for (r = 0; r < nspots; r++) {
      if (AllGrainSpots[rt][0] < 0) {
        AllGrainSpots[rt][16] = 999;
        IAgrainspots[r] = AllGrainSpots[rt][16];
        rt++;
        continue;
      }
      x1 = distance;
      x2 = distance;
      y1 = AllGrainSpots[rt][2];
      y2 = AllGrainSpots[rt][3];
      z1 = AllGrainSpots[rt][5];
      z2 = AllGrainSpots[rt][6];
      w1 = AllGrainSpots[rt][8];
      w2 = AllGrainSpots[rt][9];
      g1x = GrainMatches[g][9];
      g1y = GrainMatches[g][10];
      g1z = GrainMatches[g][11];
      spot_to_gv_pos(x1, y1, z1, w1, g1x, g1y, g1z, &gv1x, &gv1y, &gv1z);
      spot_to_gv_pos(x2, y2, z2, w2, g1x, g1y, g1z, &gv2x, &gv2y, &gv2z);
      CalcInternalAngle(gv1x, gv1y, gv1z, gv2x, gv2y, gv2z,
                        &AllGrainSpots[rt][16]);
      IAgrainspots[r] = AllGrainSpots[rt][16];
      rt++;
    }
    GrainMatches[g][15] = CalcAvgIA(IAgrainspots, nspots);
  }
}

void MakeFullFileName(char *fullFileName, char *aPath, char *aFileName) {
  if (aPath[0] == '\0') {
    strcpy(fullFileName, aFileName);
  } else {
    strcpy(fullFileName, aPath);
    strcat(fullFileName, "/");
    strcat(fullFileName, aFileName);
  }
}

int DoIndexingSingle(int voxNr, double OM[3][3], double xThis, double yThis,
                     struct TParams Params, FILE *valsF, FILE *allF, FILE *keyF,
                     struct IndexerScratch *scratch) {
  RealType ga = xThis, gb = yThis, gc = 0;
  RealType **TheorSpots = scratch->TheorSpots;
  RealType **GrainSpots = scratch->GrainSpots;
  RealType **GrainMatches = scratch->GrainMatches;
  RealType **AllGrainSpots = scratch->AllGrainSpots;
  RealType **GrainMatchesT = scratch->GrainMatchesT;
  RealType **AllGrainSpotsT = scratch->AllGrainSpotsT;

  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  int nRowsPerGrain = 2 * n_hkls, nTspots;
  // Memory allocated in scratch
  int nMatches;
  int r, c, i, SpotID;
  int rownr = 0;
  int sp;
  RealType Displ_y, Displ_z;
  RealType FracThis;
  RealType RefRad = -1;
  CalcDiffrSpots_Furnace(OM, Params.LatticeConstant, Params.Wavelength,
                         Params.Distance, Params.RingRadii, Params.OmegaRanges,
                         Params.BoxSizes, Params.NoOfOmegaRanges,
                         Params.ExcludePoleAngle, TheorSpots, &nTspots);
  for (sp = 0; sp < nTspots; sp++) {
    displacement_spot_needed_COM(
        ga, gb, gc, TheorSpots[sp][3], TheorSpots[sp][4], TheorSpots[sp][5],
        TheorSpots[sp][14], TheorSpots[sp][15], &Displ_y, &Displ_z);
    TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
    TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
    CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12]);
    TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                              TheorSpots[sp][11] * TheorSpots[sp][11]) -
                         Params.RingRadii[(int)TheorSpots[sp][9]];
  }
  CompareSpots(TheorSpots, nTspots, RefRad, Params.MarginOme, Params.MarginRad,
               Params.MarginRadial, etamargins, omemargins, &nMatches,
               GrainSpots, xThis, yThis, &Params);
  FracThis = (double)nMatches / (double)nTspots;
  if (FracThis > Params.MinMatchesToAcceptFrac) {
    for (i = 0; i < 9; i++)
      GrainMatchesT[0][i] = OM[i / 3][i % 3];
    GrainMatchesT[0][9] = ga;
    GrainMatchesT[0][10] = gb;
    GrainMatchesT[0][11] = gc;
    GrainMatchesT[0][12] = nTspots;
    GrainMatchesT[0][13] = nMatches;
    GrainMatchesT[0][14] = 1;
    for (r = 0; r < nTspots; r++) {
      for (c = 0; c < 15; c++)
        AllGrainSpotsT[r][c] = GrainSpots[r][c];
      AllGrainSpotsT[r][15] = 1;
    }
    CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance, scratch);
    rownr = nTspots;
    for (i = 0; i < 17; i++)
      GrainMatches[0][i] = GrainMatchesT[0][i];
    for (r = 0; r < nTspots; r++)
      for (c = 0; c < 17; c++)
        AllGrainSpots[r][c] = AllGrainSpotsT[r][c];
    for (r = nTspots; r < nRowsOutput; r++)
      for (c = 0; c < 17; c++)
        AllGrainSpots[r][c] = 0;
  } else {
    return 0;
  }
  SpotID = AllGrainSpots[0][13];
  double outArr[16] = {
      (double)SpotID,      GrainMatches[0][15], GrainMatches[0][0],
      GrainMatches[0][1],  GrainMatches[0][2],  GrainMatches[0][3],
      GrainMatches[0][4],  GrainMatches[0][5],  GrainMatches[0][6],
      GrainMatches[0][7],  GrainMatches[0][8],  GrainMatches[0][9],
      GrainMatches[0][10], GrainMatches[0][11], GrainMatches[0][12],
      GrainMatches[0][13]};
  size_t locVals, locAll;
  locVals = ftell(valsF);
  locAll = ftell(allF);
  fwrite(outArr, 16 * sizeof(double), 1, valsF);
  int *outArr2;
  outArr2 = malloc(rownr * sizeof(*outArr2));
  for (i = 0; i < rownr; i++)
    outArr2[i] = (int)AllGrainSpots[i][14];
  fwrite(outArr2, rownr * sizeof(int), 1, allF);
  free(outArr2);
  fprintf(keyF, "%zu %zu %zu %zu\n", (size_t)SpotID, (size_t)rownr, locVals,
          locAll);
  printf("ID: %d, voxNr: %d, Confidence: %lf\n", SpotID, voxNr, FracThis);
  return 0;
}

int DoIndexing(int SpotID, int voxNr, double xThis, double yThis, double zThis,
               struct TParams Params, int SpotRowNo, FILE *valsF, FILE *allF,
               FILE *keyF, struct IndexerScratch *scratch) {
  int i;
  RealType ga = xThis, gb = yThis, gc = zThis;
  RealType y0 = ObsSpotsLab[SpotRowNo * 10 + 0];
  RealType z0 = ObsSpotsLab[SpotRowNo * 10 + 1];
  RealType omega = ObsSpotsLab[SpotRowNo * 10 + 2];
  int ringnr = (int)ObsSpotsLab[SpotRowNo * 10 + 5];
  RealType xi, yi, zi, g1, g2, g3, hklnormal[3], hkl[3];
  MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi);
  spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
  hklnormal[0] = g1;
  hklnormal[1] = g2;
  hklnormal[2] = g3;
  int nOrient, or = 0, orDelta = 1;
  RealType RefRad = ObsSpotsLab[SpotRowNo * 10 + 3];
  RealType OrMat[MAX_N_OR][3][3];
  hkl[0] = RingHKL[ringnr][0];
  hkl[1] = RingHKL[ringnr][1];
  hkl[2] = RingHKL[ringnr][2];
  GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient, OrMat,
                                 &nOrient, ringnr);
  RealType **TheorSpots = scratch->TheorSpots;
  RealType **GrainSpots = scratch->GrainSpots;
  RealType **GrainMatches = scratch->GrainMatches;
  RealType **AllGrainSpots = scratch->AllGrainSpots;
  RealType **GrainMatchesT = scratch->GrainMatchesT;
  RealType **AllGrainSpotsT = scratch->AllGrainSpotsT;

  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  int nRowsPerGrain = 2 * n_hkls, nTspots;
  // Memory allocated in scratch
  RealType MinMatchesToAccept;
  int bestnMatchesIsp = -1, bestnMatchesRot = -1, bestnMatchesPos;
  int bestnTspotsIsp = -1, bestnTspotsRot, bestnTspotsPos;
  int nMatches, bestMatchFound = 0;
  int r, c;
  int rownr = 0;
  int matchNr = 0, sp;
  RealType MinInternalAngle = 1000;
  RealType Displ_y, Displ_z;
  RealType fracMatches = 0;
  RealType bestConfidence = 0;
  RealType FracThis;
  while (or < nOrient) {
    CalcDiffrSpots_Furnace(
        OrMat[or], Params.LatticeConstant, Params.Wavelength, Params.Distance,
        Params.RingRadii, Params.OmegaRanges, Params.BoxSizes,
        Params.NoOfOmegaRanges, Params.ExcludePoleAngle, TheorSpots, &nTspots);
    MinMatchesToAccept = nTspots * Params.MinMatchesToAcceptFrac;
    bestnMatchesPos = -1;
    bestnTspotsPos = 0;
    for (sp = 0; sp < nTspots; sp++) {
      displacement_spot_needed_COM(
          ga, gb, gc, TheorSpots[sp][3], TheorSpots[sp][4], TheorSpots[sp][5],
          TheorSpots[sp][14], TheorSpots[sp][15], &Displ_y, &Displ_z);
      TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
      TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
      CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12]);
      TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                                TheorSpots[sp][11] * TheorSpots[sp][11]) -
                           Params.RingRadii[(int)TheorSpots[sp][9]];
    }
    CompareSpots(TheorSpots, nTspots, RefRad, Params.MarginOme,
                 Params.MarginRad, Params.MarginRadial, etamargins, omemargins,
                 &nMatches, GrainSpots, xThis, yThis, &Params);
    FracThis = (double)nMatches / (double)nTspots;
    if (FracThis > Params.MinMatchesToAcceptFrac) {
      if (FracThis >= bestConfidence) {
        bestConfidence = FracThis;
        bestMatchFound = 1;
        for (i = 0; i < 9; i++)
          GrainMatchesT[0][i] = OrMat[or][i / 3][i % 3];
        GrainMatchesT[0][9] = ga;
        GrainMatchesT[0][10] = gb;
        GrainMatchesT[0][11] = gc;
        GrainMatchesT[0][12] = nTspots;
        GrainMatchesT[0][13] = nMatches;
        GrainMatchesT[0][14] = 1;
        for (r = 0; r < nTspots; r++) {
          for (c = 0; c < 15; c++)
            AllGrainSpotsT[r][c] = GrainSpots[r][c];
          AllGrainSpotsT[r][15] = 1;
        }
        CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance, scratch);
        if (FracThis == bestConfidence &&
            GrainMatchesT[0][15] > MinInternalAngle) {

        } else {
          MinInternalAngle = GrainMatchesT[0][15];
          bestnMatchesIsp = nMatches;
          bestnTspotsIsp = nTspots;
          rownr = nTspots;
          matchNr = 1;
          for (i = 0; i < 17; i++)
            GrainMatches[0][i] = GrainMatchesT[0][i];
          for (r = 0; r < nTspots; r++)
            for (c = 0; c < 17; c++)
              AllGrainSpots[r][c] = AllGrainSpotsT[r][c];
          for (r = nTspots; r < nRowsOutput; r++)
            for (c = 0; c < 17; c++)
              AllGrainSpots[r][c] = 0;
        }
      }
    }
    or += orDelta;
  }
  fracMatches = (RealType)bestnMatchesIsp / bestnTspotsIsp;
  // printf("%lf %d %d\n",fracMatches,bestnMatchesIsp,bestnTspotsIsp);
  if (bestnMatchesIsp < 0 ||
      (fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 ||
       (int)bestnMatchesIsp == -1 || bestMatchFound == 0) ||
      fracMatches < Params.MinMatchesToAcceptFrac) {
    return 0;
  }
  double outArr[16] = {
      (double)SpotID,      GrainMatches[0][15], GrainMatches[0][0],
      GrainMatches[0][1],  GrainMatches[0][2],  GrainMatches[0][3],
      GrainMatches[0][4],  GrainMatches[0][5],  GrainMatches[0][6],
      GrainMatches[0][7],  GrainMatches[0][8],  GrainMatches[0][9],
      GrainMatches[0][10], GrainMatches[0][11], GrainMatches[0][12],
      GrainMatches[0][13]};
  size_t locVals, locAll;
  locVals = ftell(valsF);
  locAll = ftell(allF);
  fwrite(outArr, 16 * sizeof(double), 1, valsF);
  int *outArr2;
  int matchedNrSpots = GrainMatches[0][13];
  outArr2 = malloc(matchedNrSpots * sizeof(*outArr2));
  for (i = 0; i < matchedNrSpots; i++)
    outArr2[i] = (int)AllGrainSpots[i][14];
  fwrite(outArr2, matchedNrSpots * sizeof(int), 1, allF);
  free(outArr2);
  fprintf(keyF, "%zu %zu %zu %zu\n", (size_t)SpotID, (size_t)matchedNrSpots,
          locVals, locAll);
  printf("ID: %d, voxNr: %d, Confidence: %lf, IA: %lf\n", SpotID, voxNr,
         fracMatches, GrainMatches[0][15]);
}

int ReadBins(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char file_name[2048];
  sprintf(file_name, "%s/Data.bin", cwd);
  char cmmd[4096];
  // sprintf(cmmd,"cp %s /dev/shm/",file_name);
  // system(cmmd);
  // sprintf(file_name,"/dev/shm/Data.bin");
  int rc;
  fd = open(file_name, O_RDONLY);
  check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  data = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(data == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  int fd2;
  struct stat s2;
  int status2;
  char file_name2[2048];
  sprintf(file_name2, "%s/nData.bin", cwd);
  // sprintf(cmmd,"cp %s /dev/shm/",file_name2);
  // system(cmmd);
  // sprintf(file_name2,"/dev/shm/nData.bin");
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  ndata = mmap(0, size2, PROT_READ, MAP_SHARED, fd2, 0);
  printf("DataSize: %lld %d Nelems: %lld \n", (long long int)size,
         (int)sizeof(*data), (long long int)(size / sizeof(*data)));
  printf("nDataSize: %lld %d Nelems: %lld \n", (long long int)size2,
         (int)sizeof(*ndata), (long long int)(size2 / sizeof(*ndata)));
  fflush(stdout);
  check(ndata == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  return 1;
}

int ReadSpots(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/Spots.bin", cwd);
  char cmmd[4096];
  // sprintf(cmmd,"cp %s /dev/shm/",filename);
  // system(cmmd);
  // sprintf(filename,"/dev/shm/Spots.bin");
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  ObsSpotsLab = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(ObsSpotsLab == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  size_t nrsps;
  nrsps = size;
  nrsps /= sizeof(double);
  nrsps /= 10;
  return nrsps;
}

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tIndexerScanningOMP v6.0\nContact hsharma@anl.gov in case of "
         "questions about the MIDAS project.\n\n");
  int returncode;
  struct TParams Params;
  char *ParamFN;
  char fn[1024];
  if (argc < 6) {
    printf("Usage: %s paramtest.txt blockNr nBlocks numScans numProcs\n\n",
           argv[0]);
    exit(EXIT_FAILURE);
  }
  ParamFN = argv[1];
  printf("Reading parameters from file: %s.\n", ParamFN);
  returncode = ReadParams(ParamFN, &Params);
  if (returncode != 0) {
    printf("Error reading params file %s\n", ParamFN);
    exit(EXIT_FAILURE);
  }

  int i;
  for (i = 1; i < 180; i++)
    omemargins[i] = Params.MarginOme +
                    (0.5 * Params.StepsizeOrient / fabs(sin(i * deg2rad)));
  omemargins[0] = omemargins[1];
  omemargins[180] = omemargins[1];
  for (i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] == 0)
      etamargins[i] = 0;
    else
      etamargins[i] = rad2deg * atan(Params.MarginEta / Params.RingRadii[i]) +
                      0.5 * Params.StepsizeOrient;
  }

  printf("SpaceGroup: %d\n", Params.SpaceGroupNum);
  printf("Finished reading parameters.\n");
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  char aline[5024], dummy[1024];
  fgets(aline, 1000, hklf);
  int Rnr;
  int hi, ki, li;
  double hc, kc, lc, RRd, Ds, tht;
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf", &hi, &ki, &li, &Ds,
           &Rnr, &hc, &kc, &lc, &tht, dummy, &RRd);
    RingHKL[Rnr][0] = hc;
    RingHKL[Rnr][1] = kc;
    RingHKL[Rnr][2] = lc;
    for (i = 0; i < Params.NrOfRings; i++) {
      if (Rnr == Params.RingNumbers[i]) {
        HKLints[n_hkls][0] = hi;
        HKLints[n_hkls][1] = ki;
        HKLints[n_hkls][2] = li;
        HKLints[n_hkls][3] = Rnr;
        hkls[n_hkls][0] = hc;
        hkls[n_hkls][1] = kc;
        hkls[n_hkls][2] = lc;
        hkls[n_hkls][3] = (double)Rnr;
        hkls[n_hkls][4] = Ds;
        hkls[n_hkls][5] = tht;
        hkls[n_hkls][6] = RRd;
        hkls[n_hkls][7] = sin(tht * deg2rad);
        RealType len = sqrt(hc * hc + kc * kc + lc * lc);
        RealType v = hkls[n_hkls][7] * len;
        hkls[n_hkls][8] = v;
        hkls[n_hkls][9] = v * v;
        n_hkls++;
      }
    }
  }
  fclose(hklf);
  char tmpstr[2048];
  sprintf(tmpstr, "%s", Params.OutputFolder);
  char *cwdstr = dirname(tmpstr);
  printf("No of hkl's: %d\n", n_hkls);
  n_spots = ReadSpots(cwdstr);
  printf("nSpots = %d, Binned data...\n", n_spots);
  int rc = ReadBins(cwdstr);
  int HighestRingNo = 0;
  for (i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] != 0)
      HighestRingNo = i;
  }
  n_ring_bins = HighestRingNo;
  n_eta_bins = ceil(360.0 / Params.EtaBinSize);
  n_ome_bins = ceil(360.0 / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize;
  EtaBinSize = Params.EtaBinSize;
  OmeBinSize = Params.OmeBinSize;
  Params.InvEtaBinSize = 1.0 / EtaBinSize;
  Params.InvOmeBinSize = 1.0 / OmeBinSize;
  printf("No of bins for rings : %d\n", n_ring_bins);
  printf("No of bins for eta   : %d\n", n_eta_bins);
  printf("No of bins for omega : %d\n", n_ome_bins);
  printf("Total no of bins     : %d\n\n",
         n_ring_bins * n_eta_bins * n_ome_bins);
  printf("Finished binning.\n\n");
  // int *SpotIDs;
  // int nSpotIDs;
  int nBlocks = atoi(argv[3]);
  int blockNr = atoi(argv[2]);
  numScans = atoi(argv[4]);
  int numProcs = atoi(argv[5]);
  int hasMic = 0, nrMic = 0;
  double *mic;
  if (strncmp(Params.MicFN, "0", strlen("0"))) {
    hasMic = 1;
    mic = calloc(MAX_MIC_ROWS * 5, sizeof(*mic));
    FILE *micF;
    micF = fopen(Params.MicFN, "r");
    if (micF == NULL) {
      printf("Mic File could not be read, but a filename was provided. "
             "Exiting.\n");
      return 1;
    }
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    fgets(aline, 1000, micF);
    printf("%s\n", aline);
    while (fgets(aline, 5000, micF) != NULL) {
      sscanf(aline, "%s %s %s %lf %lf %s %s %lf %lf %lf %s %s", dummy, dummy,
             dummy, &mic[nrMic * 5 + 0], &mic[nrMic * 5 + 1], dummy, dummy,
             &mic[nrMic * 5 + 2], &mic[nrMic * 5 + 3], &mic[nrMic * 5 + 4],
             dummy, dummy);
      nrMic++;
    }
    realloc(mic, nrMic * 5 * sizeof(*mic));
  }

  int hasGrains = 0, nrGrains = 0;
  double *grainsOM;
  if (strncmp(Params.GrainsFN, "0", strlen("0"))) {
    hasGrains = 1;
    grainsOM = calloc(MAX_MIC_ROWS * 9, sizeof(*grainsOM));
    FILE *grainsF;
    grainsF = fopen(Params.GrainsFN, "r");
    if (grainsF == NULL) {
      printf("Grains File could not be read, but a filename was provided. "
             "Exiting.\n");
      return 1;
    }
    while (fgets(aline, 5000, grainsF) != NULL) {
      if (aline[0] == '%')
        continue;
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf", dummy,
             &grainsOM[nrGrains * 9 + 0], &grainsOM[nrGrains * 9 + 1],
             &grainsOM[nrGrains * 9 + 2], &grainsOM[nrGrains * 9 + 3],
             &grainsOM[nrGrains * 9 + 4], &grainsOM[nrGrains * 9 + 5],
             &grainsOM[nrGrains * 9 + 6], &grainsOM[nrGrains * 9 + 7],
             &grainsOM[nrGrains * 9 + 8]);
      nrGrains++;
    }
    realloc(grainsOM, nrGrains * 9 * sizeof(*grainsOM));
    printf("Read %d grains from %s\n", nrGrains, Params.GrainsFN);
  }

  int startRowNr;
  int endRowNr;

  int nVoxels = numScans * numScans;
  startRowNr = (int)(ceil((double)nVoxels / (double)nBlocks)) * blockNr;
  int tmp = (int)(ceil((double)nVoxels / (double)nBlocks)) * (blockNr + 1);
  endRowNr = tmp < (nVoxels) ? tmp : (nVoxels);
  // nSpotIDs = endRowNr-startRowNr+1;
  // SpotIDs = malloc(nSpotIDs*sizeof(*SpotIDs));
  printf("%d %d %d %d %d %d %d\n", numScans, nVoxels, nBlocks, blockNr,
         startRowNr, endRowNr, numProcs);

  FILE *positionsF = fopen("positions.csv", "r");
  if (positionsF == NULL) {
    printf("positions.csv file not found.\n");
    return 1;
  }
  grid = malloc(nVoxels * 2 * sizeof(*grid));
  ypos = malloc(numScans * sizeof(*ypos));
  for (i = 0; i < numScans; i++) {
    fgets(aline, 1000, positionsF);
    sscanf(aline, "%lf", &ypos[i]);
  }
  int j;
  for (i = 0; i < numScans; i++) {
    for (j = 0; j < numScans; j++) {
      grid[(i * numScans + j) * 2 + 0] = ypos[i];
      grid[(i * numScans + j) * 2 + 1] = ypos[j];
    }
  }

  int RingToIndex = Params.RingToIndex;
  size_t startRowNrSp = MAX_N_SPOTS, endRowNrSp = 0;
  for (i = 0; i < n_spots; i++) {
    // printf("%d\n",(int)ObsSpotsLab[i*10+5]);
    // TODO::::::: ADD A RANGE OF OMEGA FILTERING
    if ((int)ObsSpotsLab[i * 10 + 5] == RingToIndex && startRowNrSp > i)
      startRowNrSp = i;
    if ((int)ObsSpotsLab[i * 10 + 5] == RingToIndex && endRowNrSp < i)
      endRowNrSp = i;
  }

  int thisRowNr;
  printf("%s %d %d %d %d\n", Params.MicFN, nrMic, hasMic, startRowNrSp,
         endRowNrSp);
#pragma omp parallel num_threads(numProcs) private(thisRowNr)
  {
    struct IndexerScratch scratch;
    int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
    int nRowsPerGrain = 2 * n_hkls;
    scratch.GrainMatches = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
    scratch.GrainMatchesT = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
    scratch.AllGrainSpots = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
    scratch.AllGrainSpotsT = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
    scratch.GrainSpots = allocMatrix(nRowsPerGrain, N_COL_GRAINSPOTS);
    scratch.TheorSpots = allocMatrix(nRowsPerGrain, N_COL_THEORSPOTS);
    scratch.IAgrainspots = malloc(nRowsOutput * sizeof(double));

#pragma omp for schedule(dynamic)
    for (thisRowNr = startRowNr; thisRowNr < endRowNr; thisRowNr++) {
      FILE *valsF, *allF, *keyF;
      char valsFN[2048], allFN[2048], keyFN[2048];
      sprintf(valsFN, "%s/IndexBest_voxNr_%0*d.bin", Params.OutputFolder, 6,
              thisRowNr);
      valsF = fopen(valsFN, "wb");
      sprintf(allFN, "%s/IndexBest_IDs_voxNr_%0*d.bin", Params.OutputFolder, 6,
              thisRowNr);
      allF = fopen(allFN, "wb");
      sprintf(keyFN, "%s/IndexKey_voxNr_%0*d.txt", Params.OutputFolder, 6,
              thisRowNr);
      keyF = fopen(keyFN, "w");
      double xThis = 0, yThis = 0;
      xThis = grid[thisRowNr * 2 + 0];
      yThis = grid[thisRowNr * 2 + 1];
      if (hasMic == 1) {
        int iter;
        int bestRow = -1;
        double bestLen = 100000, lenThis;
        for (iter = 0; iter < nrMic; iter++) {
          lenThis =
              sqrt((xThis - mic[iter * 5 + 0]) * (xThis - mic[iter * 5 + 0]) +
                   (yThis - mic[iter * 5 + 1]) * (yThis - mic[iter * 5 + 1]));
          if (lenThis < bestLen) {
            bestLen = lenThis;
            bestRow = iter;
          }
        }
        double eulerThis[3], OMThis[3][3];
        if (bestRow != -1) {
          eulerThis[0] = mic[bestRow * 5 + 2] * rad2deg;
          eulerThis[1] = mic[bestRow * 5 + 3] * rad2deg;
          eulerThis[2] = mic[bestRow * 5 + 4] * rad2deg;
          Euler2OrientMat(eulerThis, OMThis);
        }
        DoIndexingSingle(thisRowNr, OMThis, xThis, yThis, Params, valsF, allF,
                         keyF, &scratch);
      } else if (hasGrains == 1) {
        int iter;
        double OMThis[3][3];
        for (iter = 0; iter < nrGrains; iter++) {
          OMThis[0][0] = grainsOM[iter * 9 + 0];
          OMThis[0][1] = grainsOM[iter * 9 + 1];
          OMThis[0][2] = grainsOM[iter * 9 + 2];
          OMThis[1][0] = grainsOM[iter * 9 + 3];
          OMThis[1][1] = grainsOM[iter * 9 + 4];
          OMThis[1][2] = grainsOM[iter * 9 + 5];
          OMThis[2][0] = grainsOM[iter * 9 + 6];
          OMThis[2][1] = grainsOM[iter * 9 + 7];
          OMThis[2][2] = grainsOM[iter * 9 + 8];
          DoIndexingSingle(thisRowNr, OMThis, xThis, yThis, Params, valsF, allF,
                           keyF, &scratch);
        }
      } else {
        double angle, newY;
        int idnr;
        int thisID;
        int nrRows = endRowNrSp - startRowNrSp + 1;
        printf("%d %lf %lf %d %d %d %d\n", thisRowNr, xThis, yThis, startRowNr,
               endRowNr, endRowNr - startRowNr, nrRows);
        for (idnr = startRowNrSp; idnr <= endRowNrSp; idnr++) {
          angle = ObsSpotsLab[idnr * 10 + 2];
          thisID = (int)ObsSpotsLab[idnr * 10 + 4];
          newY = xThis * sin(deg2rad * angle) + yThis * cos(deg2rad * angle);
          if (fabs(newY - ypos[(int)ObsSpotsLab[idnr * 10 + 9]]) <=
              BeamSize / 2) {
            // printf("%d %lf %lf\n",idnr,newY,angle);
            DoIndexing(thisID, thisRowNr, xThis, yThis, 0, Params, idnr, valsF,
                       allF, keyF, &scratch);
          }
        }
      }
      fclose(valsF);
      fclose(allF);
      fclose(keyF);
    }
    FreeMemMatrix(scratch.GrainMatches, MAX_N_MATCHES);
    FreeMemMatrix(scratch.GrainMatchesT, MAX_N_MATCHES);
    FreeMemMatrix(scratch.AllGrainSpots, nRowsOutput);
    FreeMemMatrix(scratch.AllGrainSpotsT, nRowsOutput);
    FreeMemMatrix(scratch.GrainSpots, nRowsPerGrain);
    FreeMemMatrix(scratch.TheorSpots, nRowsPerGrain);
    free(scratch.IAgrainspots);
  }
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
  return (0);
}
