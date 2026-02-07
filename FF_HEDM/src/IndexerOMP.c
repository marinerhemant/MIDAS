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
#include <unistd.h>

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
#define MAX_N_SPOTS 6000000
#define MAX_N_STEPS 1000
#define MAX_N_OR 36000
#define MAX_N_MATCHES 1
#define MAX_N_RINGS 500
#define MAX_N_HKLS 5000
#define MAX_N_OMEGARANGES 2000
#define N_COL_THEORSPOTS 14
#define N_COL_OBSSPOTS 9
#define N_COL_GRAINSPOTS 17
#define N_COL_GRAINMATCHES 16

// Globals
RealType *ObsSpotsLab;
int n_spots = 0;

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAX_N_OMEGARANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAX_N_OMEGARANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots);

// hkls to use
double hkls[MAX_N_HKLS][7];
int n_hkls = 0;
int HKLints[MAX_N_HKLS][4];
double ABCABG[6];
double RingHKL[MAX_N_RINGS][3];
RealType RingTtheta[MAX_N_RINGS];

// For detector mapping!
int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

int *data;
int *ndata;
int SGNum;

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

int GetBin(int ringno, RealType eta, RealType omega, int **spotRows,
           int *nspotRows) {
  int iRing, iEta, iOme, iSpot;
  iRing = ringno - 1;
  iEta = floor((180 + eta) / EtaBinSize);
  iOme = floor((180 + omega) / OmeBinSize);
  int Pos = iRing * n_eta_bins * n_ome_bins + iEta * n_ome_bins + iOme;
  int nspots = ndata[Pos * 2];
  int DataPos = ndata[Pos * 2 + 1];
  *spotRows = malloc(nspots * sizeof(**spotRows));
  if (spotRows == NULL) {
    printf("Memory error: could not allocate memory for spotRows matrix. "
           "Memory full?\n");
    return 1;
  }
  // calc the diff. Note: smallest diff in pos is choosen
  for (iSpot = 0; iSpot < nspots; iSpot++) {
    (*spotRows)[iSpot] = data[DataPos + iSpot];
  }
  *nspotRows = nspots;
  return 0;
}

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

RealType **allocMatrix(int nrows, int ncols) {
  RealType **arr;
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

void FreeMemMatrix(RealType **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
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

void CalcOmega(RealType x, RealType y, RealType z, RealType theta,
               RealType omegas[4], RealType etas[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  RealType len = sqrt(x * x + y * y + z * z);
  RealType v = sin(theta * deg2rad) * len;
  RealType almostzero = 1e-4;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * rad2deg;
        omegas[*nsol] = ome;
        *nsol = *nsol + 1;
        omegas[*nsol] = -ome;
        *nsol = *nsol + 1;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType a = 1 + ((x * x) / y2);
    RealType b = (2 * v * x) / y2;
    RealType c = ((v * v) / y2) - 1;
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
      if (fabs(cosome1) <= 1.0) {
        ome1a = acos(cosome1);
        ome1b = -ome1a;
        eqa = -x * cos(ome1a) + y * sin(ome1a);
        diffa = fabs(eqa - v);
        eqb = -x * cos(ome1b) + y * sin(ome1b);
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome1a * rad2deg;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome1b * rad2deg;
          *nsol = *nsol + 1;
        }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2);
        ome2b = -ome2a;
        eqa = -x * cos(ome2a) + y * sin(ome2a);
        diffa = fabs(eqa - v);
        eqb = -x * cos(ome2b) + y * sin(ome2b);
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome2a * rad2deg;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome2b * rad2deg;
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
                            RealType **spots, int *nspots,
                            int ringsToRejectCalc[], int nRingsToRejectCalc,
                            int *nSpotsFracCalc) {
  int i, OmegaRangeNo;
  RealType DSpacings[MAX_N_HKLS];
  RealType thetas[MAX_N_HKLS];
  RealType ds;
  RealType theta;
  int KeepSpot;
  for (i = 0; i < n_hkls; i++) {
    DSpacings[i] = hkls[i][4];
    thetas[i] = hkls[i][5];
  }
  double Ghkl[3];
  int indexhkl;
  RealType Gc[3];
  RealType omegas[4];
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
  int nSpotsForFracCalc = 0;
  int SpotToFrac = 0, ringFracCntr = 0;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    ringnr = (int)(hkls[indexhkl][3]);
    SpotToFrac = 1;
    for (ringFracCntr = 0; ringFracCntr < nRingsToRejectCalc; ringFracCntr++) {
      if (ringnr == ringsToRejectCalc[ringFracCntr]) {
        SpotToFrac = 0;
        break;
      }
    }
    RealType RingRadius = RingRadii[ringnr];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    ds = DSpacings[indexhkl];
    theta = thetas[indexhkl];
    CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
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
      if (BigDetSize != 0) {
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
        spotnr++;
        spotid++;
        if (SpotToFrac) {
          nSpotsForFracCalc++;
        }
      }
    }
  }
  *nspots = spotnr;
  *nSpotsFracCalc = nSpotsForFracCalc;
}

void CompareSpots(RealType **TheorSpots, int nTheorSpots, RealType *ObsSpots,
                  RealType RefRad, RealType MarginRad, RealType MarginRadial,
                  RealType etamargins[], RealType omemargins[], int *nMatch,
                  RealType **GrainSpots, int ringsToRejectCalc[],
                  int nRingsToRejectCalc, int *nMatchesFracCalc) {
  int nMatched = 0;
  int nNonMatched = 0;
  int sp;
  int RingNr;
  int iOme, iEta;
  int spotRow, spotRowBest;
  int MatchFound;
  RealType diffOme;
  RealType diffOmeBest;
  int iRing;
  int iSpot;
  RealType etamargin, omemargin;
  *nMatchesFracCalc = 0;
  for (sp = 0; sp < nTheorSpots; sp++) {
    RingNr = (int)TheorSpots[sp][9];
    iRing = RingNr - 1;
    iEta = floor((180 + TheorSpots[sp][12]) / EtaBinSize);
    iOme = floor((180 + TheorSpots[sp][6]) / OmeBinSize);
    etamargin = etamargins[RingNr];
    omemargin = omemargins[(int)floor(fabs(TheorSpots[sp][12]))];
    MatchFound = 0;
    diffOmeBest = 100000;
    long long int Pos =
        iRing * n_eta_bins * n_ome_bins + iEta * n_ome_bins + iOme;
    long long int nspots = ndata[Pos * 2];
    long long int DataPos = ndata[Pos * 2 + 1];
    for (iSpot = 0; iSpot < nspots; iSpot++) {
      spotRow = data[DataPos + iSpot];
      if (fabs(TheorSpots[sp][13] - ObsSpots[spotRow * 9 + 8]) < MarginRadial) {
        if (fabs(RefRad - ObsSpots[spotRow * 9 + 3]) < MarginRad) {
          if (fabs(TheorSpots[sp][12] - ObsSpots[spotRow * 9 + 6]) <
              etamargin) {
            diffOme = fabs(TheorSpots[sp][6] - ObsSpots[spotRow * 9 + 2]);
            if (diffOme < diffOmeBest) {
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
      GrainSpots[nMatched][3] = ObsSpots[spotRowBest * 9 + 0];
      GrainSpots[nMatched][4] =
          ObsSpots[spotRowBest * 9 + 0] - TheorSpots[sp][10];
      GrainSpots[nMatched][5] = TheorSpots[sp][11];
      GrainSpots[nMatched][6] = ObsSpots[spotRowBest * 9 + 1];
      GrainSpots[nMatched][7] =
          ObsSpots[spotRowBest * 9 + 1] - TheorSpots[sp][11];
      GrainSpots[nMatched][8] = TheorSpots[sp][6];
      GrainSpots[nMatched][9] = ObsSpots[spotRowBest * 9 + 2];
      GrainSpots[nMatched][10] =
          ObsSpots[spotRowBest * 9 + 2] - TheorSpots[sp][6];
      GrainSpots[nMatched][11] = RefRad;
      GrainSpots[nMatched][12] = ObsSpots[spotRowBest * 9 + 3];
      GrainSpots[nMatched][13] = ObsSpots[spotRowBest * 9 + 3] - RefRad;
      GrainSpots[nMatched][14] = ObsSpots[spotRowBest * 9 + 4];
      if (GrainSpots[nMatched][14] > 1e6 || GrainSpots[nMatched][14] < -1e6) {
        printf("IndexerOMP Debug: Suspicious value at spotRowBest=%d (File "
               "index %d). Val=%f\n",
               spotRowBest, spotRowBest * 9 + 4, GrainSpots[nMatched][14]);
        if (spotRowBest >= n_spots) {
          printf("IndexerOMP Error: spotRowBest %d >= n_spots %d!\n",
                 spotRowBest, n_spots);
        }
      }
      nMatched++;
      (*nMatchesFracCalc)++;
      for (int i = 0; i < nRingsToRejectCalc; i++) {
        if (RingNr == ringsToRejectCalc[i]) {
          (*nMatchesFracCalc)--;
          break;
        }
      }
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
                                   RealType stepsize, RealType *OrMat,
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
        OrMat[or * 9 + row * 3 + col] = RotMat3[row][col];
      }
    }
  }
  *nOrient = nstepsi;
  return 0;
}

void displacement_spot_needed_COM(RealType a, RealType b, RealType c,
                                  RealType xi, RealType yi, RealType zi,
                                  RealType omega, RealType *Displ_y,
                                  RealType *Displ_z) {
  RealType lenInv = 1 / sqrt(xi * xi + yi * yi + zi * zi);
  xi = xi * lenInv;
  yi = yi * lenInv;
  zi = zi * lenInv;
  RealType OmegaRad = deg2rad * omega;
  RealType sinOme = sin(OmegaRad);
  RealType cosOme = cos(OmegaRad);
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

void FriedelEtaCalculation(RealType ys, RealType zs, RealType ttheta,
                           RealType eta, RealType Ring_rad, RealType Rsample,
                           RealType Hbeam, RealType *EtaMinFr,
                           RealType *EtaMaxFr) {
  RealType quadr_coeff2 = 0;
  RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0,
                                   y0_min_z0, y0_max = 0, y0_min = 0,
                                   z0_min = 0, z0_max = 0;
  if (eta > 90)
    eta_Hbeam = 180 - eta;
  else if (eta < -90)
    eta_Hbeam = 180 - fabs(eta);
  else
    eta_Hbeam = 90 - fabs(eta);
  Hbeam = Hbeam +
          2 * (Rsample * tan(ttheta * deg2rad)) * (sin(eta_Hbeam * deg2rad));
  RealType eta_pole = 1 + rad2deg * acos(1 - (Hbeam / Ring_rad));
  RealType eta_equator = 1 + rad2deg * acos(1 - (Rsample / Ring_rad));
  if ((eta >= eta_pole) && (eta <= (90 - eta_equator))) {
    quadr_coeff = 1;
    coeff_y0 = -1;
    coeff_z0 = 1;
  } else if ((eta >= (90 + eta_equator)) && (eta <= (180 - eta_pole))) {
    quadr_coeff = 2;
    coeff_y0 = -1;
    coeff_z0 = -1;
  } else if ((eta >= (-90 + eta_equator)) && (eta <= -eta_pole)) {
    quadr_coeff = 2;
    coeff_y0 = 1;
    coeff_z0 = 1;
  } else if ((eta >= (-180 + eta_pole)) && (eta <= (-90 - eta_equator))) {
    quadr_coeff = 1;
    coeff_y0 = 1;
    coeff_z0 = -1;
  } else
    quadr_coeff = 0;
  RealType y0_max_Rsample = ys + Rsample;
  RealType y0_min_Rsample = ys - Rsample;
  RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
  RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
    y0_min_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
  } else if (quadr_coeff == 2) {
    y0_max_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
    y0_min_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
  }
  if (quadr_coeff > 0) {
    y0_max = min(y0_max_Rsample, y0_max_z0);
    y0_min = max(y0_min_Rsample, y0_min_z0);
  } else {
    if ((eta > -eta_pole) && (eta < eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = 1;
    } else if (eta < (-180 + eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = -1;
    } else if (eta > (180 - eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = -1;
    } else if ((eta > (90 - eta_equator)) && (eta < (90 + eta_equator))) {
      quadr_coeff2 = 1;
      z0_max = z0_max_Hbeam;
      z0_min = z0_min_Hbeam;
      coeff_y0 = -1;
    } else if ((eta > (-90 - eta_equator)) && (eta < (-90 + eta_equator))) {
      quadr_coeff2 = 1;
      z0_max = z0_max_Hbeam;
      z0_min = z0_min_Hbeam;
      coeff_y0 = 1;
    }
  }
  if (quadr_coeff2 == 0) {
    z0_min = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y0_min * y0_min));
    z0_max = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y0_max * y0_max));
  } else {
    y0_min = coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min * z0_min));
    y0_max = coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max * z0_max));
  }
  RealType dYMin = ys - y0_min;
  RealType dYMax = ys - y0_max;
  RealType dZMin = zs - z0_min;
  RealType dZMax = zs - z0_max;
  RealType YMinFrIdeal = y0_min;
  RealType YMaxFrIdeal = y0_max;
  RealType ZMinFrIdeal = -z0_min;
  RealType ZMaxFrIdeal = -z0_max;
  RealType YMinFr = YMinFrIdeal - dYMin;
  RealType YMaxFr = YMaxFrIdeal - dYMax;
  RealType ZMinFr = ZMinFrIdeal + dZMin;
  RealType ZMaxFr = ZMaxFrIdeal + dZMax;
  RealType Eta1, Eta2;
  CalcEtaAngle((YMinFr + ys), (ZMinFr - zs), &Eta1);
  CalcEtaAngle((YMaxFr + ys), (ZMaxFr - zs), &Eta2);
  *EtaMinFr = min(Eta1, Eta2);
  *EtaMaxFr = max(Eta1, Eta2);
}

void GenerateIdealSpots(RealType ys, RealType zs, RealType ttheta, RealType eta,
                        RealType Ring_rad, RealType Rsample, RealType Hbeam,
                        RealType step_size, RealType y0_vector[],
                        RealType z0_vector[], int *NoOfSteps) {
  int quadr_coeff2 = 0;
  RealType eta_Hbeam, quadr_coeff, coeff_y0 = 0, coeff_z0 = 0, y0_max_z0,
                                   y0_min_z0, y0_max = 0, y0_min = 0,
                                   z0_min = 0, z0_max = 0;
  RealType y01, z01, y02, z02, y_diff, z_diff, length;
  int nsteps;
  if (eta > 90)
    eta_Hbeam = 180 - eta;
  else if (eta < -90)
    eta_Hbeam = 180 - fabs(eta);
  else
    eta_Hbeam = 90 - fabs(eta);
  Hbeam = Hbeam +
          2 * (Rsample * tan(ttheta * deg2rad)) * (sin(eta_Hbeam * deg2rad));
  RealType eta_pole = 1 + rad2deg * acos(1 - (Hbeam / Ring_rad));
  RealType eta_equator = 1 + rad2deg * acos(1 - (Rsample / Ring_rad));
  if ((eta >= eta_pole) && (eta <= (90 - eta_equator))) {
    quadr_coeff = 1;
    coeff_y0 = -1;
    coeff_z0 = 1;
  } else if ((eta >= (90 + eta_equator)) && (eta <= (180 - eta_pole))) {
    quadr_coeff = 2;
    coeff_y0 = -1;
    coeff_z0 = -1;
  } else if ((eta >= (-90 + eta_equator)) && (eta <= -eta_pole)) {
    quadr_coeff = 2;
    coeff_y0 = 1;
    coeff_z0 = 1;
  } else if ((eta >= (-180 + eta_pole)) && (eta <= (-90 - eta_equator))) {
    quadr_coeff = 1;
    coeff_y0 = 1;
    coeff_z0 = -1;
  } else
    quadr_coeff = 0;
  RealType y0_max_Rsample = ys + Rsample;
  RealType y0_min_Rsample = ys - Rsample;
  RealType z0_max_Hbeam = zs + 0.5 * Hbeam;
  RealType z0_min_Hbeam = zs - 0.5 * Hbeam;
  if (quadr_coeff == 1) {
    y0_max_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
    y0_min_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
  } else if (quadr_coeff == 2) {
    y0_max_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_min_Hbeam * z0_min_Hbeam));
    y0_min_z0 =
        coeff_y0 * sqrt((Ring_rad * Ring_rad) - (z0_max_Hbeam * z0_max_Hbeam));
  }
  if (quadr_coeff > 0) {
    y0_max = min(y0_max_Rsample, y0_max_z0);
    y0_min = max(y0_min_Rsample, y0_min_z0);
  } else {
    if ((eta > -eta_pole) && (eta < eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = 1;
    } else if (eta < (-180 + eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = -1;
    } else if (eta > (180 - eta_pole)) {
      y0_max = y0_max_Rsample;
      y0_min = y0_min_Rsample;
      coeff_z0 = -1;
    } else if ((eta > (90 - eta_equator)) && (eta < (90 + eta_equator))) {
      quadr_coeff2 = 1;
      z0_max = z0_max_Hbeam;
      z0_min = z0_min_Hbeam;
      coeff_y0 = -1;
    } else if ((eta > (-90 - eta_equator)) && (eta < (-90 + eta_equator))) {
      quadr_coeff2 = 1;
      z0_max = z0_max_Hbeam;
      z0_min = z0_min_Hbeam;
      coeff_y0 = 1;
    }
  }
  if (quadr_coeff2 == 0) {
    y01 = y0_min;
    z01 = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y01 * y01));
    y02 = y0_max;
    z02 = coeff_z0 * sqrt((Ring_rad * Ring_rad) - (y02 * y02));
    y_diff = y01 - y02;
    z_diff = z01 - z02;
    length = sqrt(y_diff * y_diff + z_diff * z_diff);
    nsteps = ceil(length / step_size);
  } else {
    z01 = z0_min;
    y01 = coeff_y0 * sqrt((Ring_rad * Ring_rad) - ((z01 * z01)));
    z02 = z0_max;
    y02 = coeff_y0 * sqrt((Ring_rad * Ring_rad) - ((z02 * z02)));
    y_diff = y01 - y02;
    z_diff = z01 - z02;
    length = sqrt(y_diff * y_diff + z_diff * z_diff);
    nsteps = ceil(length / step_size);
  }
  if ((nsteps % 2) == 0) {
    nsteps = nsteps + 1;
  }
  if (nsteps == 1) {
    if (quadr_coeff2 == 0) {
      y0_vector[0] = (y0_max + y0_min) / 2;
      z0_vector[0] = coeff_z0 * sqrt((Ring_rad * Ring_rad) -
                                     (y0_vector[0] * y0_vector[0]));
    } else {
      z0_vector[0] = (z0_max + z0_min) / 2;
      y0_vector[0] = coeff_y0 * sqrt((Ring_rad * Ring_rad) -
                                     (z0_vector[0] * z0_vector[0]));
    }
  } else {
    int i;
    RealType stepsizeY = (y0_max - y0_min) / (nsteps - 1);
    RealType stepsizeZ = (z0_max - z0_min) / (nsteps - 1);
    if (quadr_coeff2 == 0) {
      for (i = 0; i < nsteps; i++) {
        y0_vector[i] = y0_min + i * stepsizeY;
        z0_vector[i] = coeff_z0 * sqrt((Ring_rad * Ring_rad) -
                                       (y0_vector[i] * y0_vector[i]));
      }
    } else {
      for (i = 0; i < nsteps; i++) {
        z0_vector[i] = z0_min + i * stepsizeZ;
        y0_vector[i] = coeff_y0 * sqrt((Ring_rad * Ring_rad) -
                                       (z0_vector[i] * z0_vector[i]));
      }
    }
  }
  *NoOfSteps = nsteps;
}

void calc_n_max_min(RealType xi, RealType yi, RealType ys, RealType y0,
                    RealType R_sample, int step_size, int *n_max, int *n_min) {
  RealType dy = ys - y0;
  RealType a = xi * xi + yi * yi;
  RealType b = 2 * yi * dy;
  RealType c = dy * dy - R_sample * R_sample;
  RealType D = b * b - 4 * a * c;
  RealType P = sqrt(D);
  RealType lambda_max = (-b + P) / (2 * a) + 20;
  *n_max = (int)((lambda_max * xi) / (step_size));
  *n_min = -*n_max;
}

void spot_to_unrotated_coordinates(RealType xi, RealType yi, RealType zi,
                                   RealType ys, RealType zs, RealType y0,
                                   RealType z0, RealType step_size_in_x, int n,
                                   RealType omega, RealType *a, RealType *b,
                                   RealType *c) {
  RealType lambda = (step_size_in_x) * (n / xi);
  RealType x1 = lambda * xi;
  RealType y1 = ys - y0 + lambda * yi;
  RealType z1 = zs - z0 + lambda * zi;
  RealType cosOme = cos(omega * deg2rad);
  RealType sinOme = sin(omega * deg2rad);
  *a = (x1 * cosOme) + (y1 * sinOme);
  *b = (y1 * cosOme) - (x1 * sinOme);
  *c = z1;
}

void GenerateIdealSpotsFriedel(RealType ys, RealType zs, RealType ttheta,
                               RealType eta, RealType omega, int ringno,
                               RealType Ring_rad, RealType Rsample,
                               RealType Hbeam, RealType OmeTol,
                               RealType RadiusTol, RealType y0_vector[],
                               RealType z0_vector[], int *NoOfSteps) {
  RealType EtaF;
  RealType OmeF;
  RealType EtaMinF, EtaMaxF, etaIdealF;
  RealType IdealYPos, IdealZPos;
  *NoOfSteps = 0;
  if (omega < 0)
    OmeF = omega + 180;
  else
    OmeF = omega - 180;
  if (eta < 0)
    EtaF = -180 - eta;
  else
    EtaF = 180 - eta;
  int r;
  int rno_obs;
  RealType ome_obs, eta_obs;
  for (r = 0; r < n_spots; r++) {
    rno_obs = round(ObsSpotsLab[r * 9 + 5]);
    ome_obs = ObsSpotsLab[r * 9 + 2];
    eta_obs = ObsSpotsLab[r * 9 + 6];
    if (rno_obs != ringno)
      continue;
    if (fabs(ome_obs - OmeF) > OmeTol)
      continue;
    RealType yf = ObsSpotsLab[r * 9 + 0];
    RealType zf = ObsSpotsLab[r * 9 + 1];
    RealType EtaTransf;
    CalcEtaAngle(yf + ys, zf - zs, &EtaTransf);
    RealType radius = sqrt((yf + ys) * (yf + ys) + (zf - zs) * (zf - zs));
    if (fabs(radius - 2 * Ring_rad) > RadiusTol)
      continue;
    FriedelEtaCalculation(ys, zs, ttheta, eta, Ring_rad, Rsample, Hbeam,
                          &EtaMinF, &EtaMaxF);
    if ((EtaTransf < EtaMinF) || (EtaTransf > EtaMaxF))
      continue;
    RealType ZPositionAccZ = zs - ((zf + zs) / 2);
    RealType YPositionAccY = ys - ((-yf + ys) / 2);
    CalcEtaAngle(YPositionAccY, ZPositionAccZ, &etaIdealF);
    CalcSpotPosition(Ring_rad, etaIdealF, &IdealYPos, &IdealZPos);
    y0_vector[*NoOfSteps] = IdealYPos;
    z0_vector[*NoOfSteps] = IdealZPos;
    (*NoOfSteps)++;
  }
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

void GenerateIdealSpotsFriedelMixed(
    RealType ys, RealType zs, RealType Ttheta, RealType Eta, RealType Omega,
    int RingNr, RealType Ring_rad, RealType Lsd, RealType Rsample,
    RealType Hbeam, RealType StepSizePos, RealType OmeTol, RealType RadialTol,
    RealType EtaTol, RealType spots_y[], RealType spots_z[], int *NoOfSteps) {
  const int MinEtaReject = 10;
  RealType omegasFP[4];
  RealType etasFP[4];
  int nsol;
  int nFPCandidates;
  RealType theta = Ttheta / 2;
  RealType SinMinEtaReject = sin(MinEtaReject * deg2rad);
  RealType y0_vector[2000];
  RealType z0_vector[2000];
  RealType G1, G2, G3;
  int SpOnRing, NoOfSpots;
  int FPCandidatesUnique[2000];
  RealType FPCandidates[2000][3];
  RealType xi, yi, zi;
  RealType y0, z0;
  RealType YFP1, ZFP1;
  int nMax, nMin;
  RealType EtaTolDeg;
  EtaTolDeg = rad2deg * atan(EtaTol / Ring_rad);
  *NoOfSteps = 0;
  nFPCandidates = 0;
  if (fabs(sin(Eta * deg2rad)) < SinMinEtaReject) {
    return;
  }
  GenerateIdealSpots(ys, zs, Ttheta, Eta, Ring_rad, Rsample, Hbeam, StepSizePos,
                     y0_vector, z0_vector, &NoOfSpots);
  for (SpOnRing = 0; SpOnRing < NoOfSpots; ++SpOnRing) {
    y0 = y0_vector[SpOnRing];
    z0 = z0_vector[SpOnRing];
    MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi);
    spot_to_gv(Lsd, y0, z0, Omega, &G1, &G2, &G3);
    CalcOmega(-G1, -G2, -G3, theta, omegasFP, etasFP, &nsol);
    if (nsol <= 1) {
      printf("no omega solutions. skipping plane.\n");
      continue;
    }
    RealType OmegaFP, EtaFP, diff0, diff1;
    diff0 = fabs(omegasFP[0] - Omega);
    if (diff0 > 180)
      diff0 = 360 - diff0;
    diff1 = fabs(omegasFP[1] - Omega);
    if (diff1 > 180)
      diff1 = 360 - diff1;
    if (diff0 < diff1) {
      OmegaFP = omegasFP[0];
      EtaFP = etasFP[0];
    } else {
      OmegaFP = omegasFP[1];
      EtaFP = etasFP[1];
    }
    CalcSpotPosition(Ring_rad, EtaFP, &YFP1, &ZFP1);
    calc_n_max_min(xi, yi, ys, y0, Rsample, StepSizePos, &nMax, &nMin);
    RealType a, b, c, YFP, ZFP, RadialPosFP, EtaFPCorr;
    int n;
    for (n = nMin; n <= nMax; ++n) {
      spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0, StepSizePos, n,
                                    Omega, &a, &b, &c);
      if (fabs(c) > Hbeam / 2)
        continue;
      RealType Dy, Dz;
      displacement_spot_needed_COM(a, b, c, Lsd, YFP1, ZFP1, OmegaFP, &Dy, &Dz);
      YFP = YFP1 + Dy;
      ZFP = ZFP1 + Dz;
      RadialPosFP = sqrt(YFP * YFP + ZFP * ZFP) - Ring_rad;
      CalcEtaAngle(YFP, ZFP, &EtaFPCorr);
      int *spotRows;
      int nspotRows, iSpot, spotRow;
      GetBin(RingNr, EtaFPCorr, OmegaFP, &spotRows, &nspotRows);
      RealType diffPos2, dy, dz;
      for (iSpot = 0; iSpot < nspotRows; iSpot++) {
        spotRow = spotRows[iSpot];
        if ((fabs(RadialPosFP - ObsSpotsLab[spotRow * 9 + 8]) < RadialTol) &&
            (fabs(OmegaFP - ObsSpotsLab[spotRow * 9 + 2]) < OmeTol) &&
            (fabs(EtaFPCorr - ObsSpotsLab[spotRow * 9 + 6]) < EtaTolDeg)) {
          dy = (YFP - ObsSpotsLab[spotRow * 9 + 0]);
          dz = (ZFP - ObsSpotsLab[spotRow * 9 + 1]);
          diffPos2 = dy * dy + dz * dz;
          int i;
          int idx = nFPCandidates;
          for (i = 0; i < nFPCandidates; ++i) {
            if (FPCandidates[i][0] == ObsSpotsLab[spotRow * 9 + 4]) {
              if (diffPos2 < FPCandidates[i][2]) {
                idx = i;
              } else {
                idx = -1;
              }
              break;
            }
          }
          if (idx >= 0) {
            FPCandidates[idx][0] = ObsSpotsLab[spotRow * 9 + 4];
            FPCandidates[idx][1] = SpOnRing;
            FPCandidates[idx][2] = diffPos2;
            if (idx == nFPCandidates)
              nFPCandidates++;
          }
        }
      }
      free(spotRows);
    }
  }
  int i;
  int nFPCandidatesUniq = 0;
  for (i = 0; i < nFPCandidates; ++i) {
    AddUnique(FPCandidatesUnique, &nFPCandidatesUniq, FPCandidates[i][1]);
  }
  int iFP;
  for (iFP = 0; iFP < nFPCandidatesUniq; ++iFP) {
    spots_y[iFP] = y0_vector[FPCandidatesUnique[iFP]];
    spots_z[iFP] = z0_vector[FPCandidatesUnique[iFP]];
  }
  *NoOfSteps = nFPCandidatesUniq;
}

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
  RealType ExcludePoleAngle;
  RealType MinMatchesToAcceptFrac;
  RealType BoxSizes[MAX_N_OMEGARANGES][4];
  RealType OmegaRanges[MAX_N_OMEGARANGES][2];
  char OutputFolder[4096];
  int NoOfOmegaRanges;
  int isGrainsInput;
  char GrainsFileName[4096];
  char SpotsFileName[4096];
  char IDsFileName[4096];
  int UseFriedelPairs;
  int RingsToReject[MAX_N_RINGS];
  int nRingsToRejectCalc;
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
  FILE *fp;
  char line[MAX_LINE_LENGTH];
  char dummy[MAX_LINE_LENGTH];
  char *str;
  int NrOfBoxSizes = 0;
  int cmpres;
  int NoRingNumbers = 0;
  Params->NrOfRings = 0;
  Params->NoOfOmegaRanges = 0;
  Params->isGrainsInput = 0;
  Params->nRingsToRejectCalc = 0;
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
    str = "RingsToExcludeFraction ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy,
             &(Params->RingsToReject[Params->nRingsToRejectCalc]));
      Params->nRingsToRejectCalc++;
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
    str = "LatticeConstant ";
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
    str = "Lsd ";
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
    str = "StepSizeOrient ";
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
    str = "Completeness ";
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
    str = "MinEta ";
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
    str = "GrainsFile ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      Params->isGrainsInput = 1;
      sscanf(line, "%s %s", dummy, Params->GrainsFileName);
      continue;
    }
    str = "IDsFileName ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %s", dummy, Params->IDsFileName);
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

void CreateNumberedFilenameW(char stem[1000], int aNumber, int numberOfDigits,
                             char ext[10],
                             char fn[1000 + 10 + numberOfDigits + 1]) {
  sprintf(fn, "%s%0*d%s", stem, numberOfDigits, aNumber, ext);
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

int WriteBestMatch(char *FileName, RealType **GrainMatches, int ngrains,
                   RealType **AllGrainSpots, int nrows, char *FileName2) {
  int r, g, c;
  RealType smallestIA = 99999;
  int bestGrainIdx = -1;
  for (g = 0; g < ngrains; g++) {
    if (GrainMatches[g][15] < smallestIA) {
      smallestIA = GrainMatches[g][15];
      bestGrainIdx = g;
    }
  }
  if (bestGrainIdx != -1) {
    FILE *fp2;
    fp2 = fopen(FileName2, "w");
    if (fp2 == NULL) {
      printf("Cannot open file: %s\n", FileName2);
      return (1);
    }
    RealType bestGrainID = GrainMatches[bestGrainIdx][14];
    fprintf(fp2, "%lf\n%lf\n", bestGrainID, bestGrainID);
    fprintf(fp2,
            "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, "
            "%lf, %lf\n",
            GrainMatches[bestGrainIdx][15], GrainMatches[bestGrainIdx][0],
            GrainMatches[bestGrainIdx][1], GrainMatches[bestGrainIdx][2],
            GrainMatches[bestGrainIdx][3], GrainMatches[bestGrainIdx][4],
            GrainMatches[bestGrainIdx][5], GrainMatches[bestGrainIdx][6],
            GrainMatches[bestGrainIdx][7], GrainMatches[bestGrainIdx][8],
            GrainMatches[bestGrainIdx][9], GrainMatches[bestGrainIdx][10],
            GrainMatches[bestGrainIdx][11], GrainMatches[bestGrainIdx][12],
            GrainMatches[bestGrainIdx][13]);
    for (r = 0; r < nrows; r++) {
      if (AllGrainSpots[r][15] == bestGrainID) {
        for (c = 0; c < N_COL_GRAINSPOTS; c++) {
          if (c != 1)
            fprintf(fp2, "%14lf, ", AllGrainSpots[r][c]);
        }
        fprintf(fp2, "\n");
      }
    }
    fclose(fp2);
  } else {
    FILE *fp2;
    fp2 = fopen(FileName2, "w");
    fclose(fp2);
  }
  return (0);
}

int WriteBestMatchBin(RealType **GrainMatches, RealType **AllGrainSpots,
                      int nrows, char *FolderName, int offsetLoc) {
  int r, g, c;
  int bestGrainIdx = 0;
  char outFN[2048];
  sprintf(outFN, "%s/IndexBest.bin", FolderName);
  int Ib = open(outFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  char outFN2[2048];
  sprintf(outFN2, "%s/IndexBestFull.bin", FolderName);
  int Ib2 = open(outFN2, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (Ib <= 0) {
    printf("Cannot open file: %s\n", outFN);
    return (1);
  }
  if (Ib2 <= 0) {
    printf("Cannot open file: %s\n", outFN2);
    return (1);
  }
  size_t offst1 = offsetLoc;
  offst1 *= 15;
  offst1 *= sizeof(double);
  double res[15] = {
      GrainMatches[bestGrainIdx][15], GrainMatches[bestGrainIdx][0],
      GrainMatches[bestGrainIdx][1],  GrainMatches[bestGrainIdx][2],
      GrainMatches[bestGrainIdx][3],  GrainMatches[bestGrainIdx][4],
      GrainMatches[bestGrainIdx][5],  GrainMatches[bestGrainIdx][6],
      GrainMatches[bestGrainIdx][7],  GrainMatches[bestGrainIdx][8],
      GrainMatches[bestGrainIdx][9],  GrainMatches[bestGrainIdx][10],
      GrainMatches[bestGrainIdx][11], GrainMatches[bestGrainIdx][12],
      GrainMatches[bestGrainIdx][13]};
  int rc1 = pwrite(Ib, res, 15 * sizeof(double), offst1);
  if (rc1 < 0) {
    printf("Could not write to output file %s.\n", outFN);
  }
  rc1 = close(Ib);
  double *outArr;
  outArr = malloc(MAX_N_HKLS * 2 * sizeof(*outArr));
  for (r = 0; r < nrows; r++) {
    if (AllGrainSpots[r][15] == 1.0) {
      int idxx = 0;
      outArr[r * 2 + 0] = AllGrainSpots[r][14];
      outArr[r * 2 + 1] = AllGrainSpots[r][12];
    }
  }
  size_t offst2 = offsetLoc;
  offst2 *= MAX_N_HKLS;
  offst2 *= 2;
  offst2 *= sizeof(double);
  int rc2 = pwrite(Ib2, outArr, MAX_N_HKLS * 2 * sizeof(double), offst2);
  free(outArr);
  if (rc2 < 0) {
    printf("Could not write to output file %s.\n", outFN2);
  }
  rc2 = close(Ib2);
  return (0);
}

void CalcIA(RealType **GrainMatches, int ngrains, RealType **AllGrainSpots,
            RealType distance) {
  RealType *IAgrainspots;
  int r, g;
  RealType g1x, g1y, g1z;
  RealType x1, y1, z1, w1, x2, y2, z2, w2, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
  int nspots;
  int rt = 0;
  IAgrainspots = malloc(5000 * sizeof(*IAgrainspots));
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
  free(IAgrainspots);
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

int DoIndexing(int SpotIDs, struct TParams Params, int offsetLoc, int idNr,
               int totalIDs, int ringsToRejectCalc[], int nRingsToRejectCalc) {
  double dif;
  RealType HalfBeam = Params.Hbeam / 2;
  RealType MinMatchesToAccept;
  RealType ga, gb, gc;
  int nTspots, nTspotsFracCalc, nMatchesFracCalc;
  int bestnMatchesIsp, bestnMatchesRot, bestnMatchesPos;
  int bestnTspotsIsp, bestnTspotsRot, bestnTspotsPos;
  int matchNr;
  int nOrient;
  RealType hklnormal[3];
  RealType Displ_y;
  RealType Displ_z;
  int or;
  int sp;
  int nMatches;
  int r, c, i, j, k;
  RealType y0_vector[MAX_N_STEPS];
  RealType z0_vector[MAX_N_STEPS];
  int nPlaneNormals;
  double hkl[3];
  RealType g1, g2, g3;
  int isp;
  RealType xi, yi, zi;
  int n_max, n_min, n;
  RealType y0, z0;
  int orDelta, ispDelta, nDelta;
  RealType fracMatches;
  int rownr;
  int SpotRowNo;
  int usingFriedelPair;
  RealType **BestMatches;

  RealType omemargins[181];
  RealType etamargins[MAX_N_RINGS];
  char fn[1000];
  char ffn[1000];
  char fn2[1000];
  char ffn2[1000];
  RealType **GrainMatches;
  RealType **TheorSpots;
  RealType **GrainSpots;
  RealType **AllGrainSpots;
  RealType **GrainMatchesT;
  RealType **AllGrainSpotsT;
  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  AllGrainSpots = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
  if (AllGrainSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  AllGrainSpotsT = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
  if (AllGrainSpotsT == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  GrainMatchesT = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
  if (GrainMatchesT == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  GrainMatches = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
  if (GrainMatches == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  int nRowsPerGrain = 2 * n_hkls;
  GrainSpots = allocMatrix(nRowsPerGrain, N_COL_GRAINSPOTS);
  TheorSpots = allocMatrix(nRowsPerGrain, N_COL_THEORSPOTS);
  if (TheorSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  BestMatches = allocMatrix(2, 5);
  if (BestMatches == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
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
  int SpotIDIdx = 0;
  RealType MinInternalAngle = 1000;
  matchNr = 0;
  rownr = 0;
  RealType SpotID = SpotIDs;
  FindInMatrix(&ObsSpotsLab[0 * 9 + 0], n_spots, N_COL_OBSSPOTS, 4, SpotID,
               &SpotRowNo);
  if (SpotRowNo == -1) {
    printf(
        "WARNING: SpotId %lf not found in spots file! Ignoring this spotID.\n",
        SpotID);
    fflush(stdout);
    return 1;
  }
  RealType ys = ObsSpotsLab[SpotRowNo * 9 + 0];
  RealType zs = ObsSpotsLab[SpotRowNo * 9 + 1];
  RealType omega = ObsSpotsLab[SpotRowNo * 9 + 2];
  RealType RefRad = ObsSpotsLab[SpotRowNo * 9 + 3];
  RealType eta = ObsSpotsLab[SpotRowNo * 9 + 6];
  RealType ttheta = ObsSpotsLab[SpotRowNo * 9 + 7];
  int ringnr = (int)ObsSpotsLab[SpotRowNo * 9 + 5];
  RealType *OrMat;
  OrMat = calloc(MAX_N_OR * 3 * 3, sizeof(*OrMat));
  hkl[0] = RingHKL[ringnr][0];
  hkl[1] = RingHKL[ringnr][1];
  hkl[2] = RingHKL[ringnr][2];
  long long int SpotID2 = (int)SpotIDs;
  nPlaneNormals = 0;
  usingFriedelPair = 0;
  if (Params.UseFriedelPairs == 1) {
    usingFriedelPair = 1;
    GenerateIdealSpotsFriedel(ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
                              Params.RingRadii[ringnr], Params.Rsample,
                              Params.Hbeam, Params.MarginOme,
                              Params.MarginRadial, y0_vector, z0_vector,
                              &nPlaneNormals);
    if (nPlaneNormals == 0) {
      GenerateIdealSpotsFriedelMixed(
          ys, zs, RingTtheta[ringnr], eta, omega, ringnr,
          Params.RingRadii[ringnr], Params.Distance, Params.Rsample,
          Params.Hbeam, Params.StepsizePos, Params.MarginOme,
          Params.MarginRadial, Params.MarginEta, y0_vector, z0_vector,
          &nPlaneNormals);
    }
  }
  if (nPlaneNormals == 0) {
    if (usingFriedelPair == 1) {
      printf("We were using Friedel Pairs, but did not find any for this spot "
             "%d\n.",
             SpotIDs);
      fflush(stdout);
      return 1;
    }
    usingFriedelPair = 0;
    GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                       Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam,
                       Params.StepsizePos, y0_vector, z0_vector,
                       &nPlaneNormals);
  }
  bestnMatchesIsp = -1;
  bestnTspotsIsp = 0;
  isp = 0;
  RealType bestFracTillNow = -1;
  int bestMatchFound = 0, bestNMatches = -1000;
  double sttm = omp_get_wtime();
  while (isp < nPlaneNormals) {
    y0 = y0_vector[isp];
    z0 = z0_vector[isp];
    MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi);
    spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
    hklnormal[0] = g1;
    hklnormal[1] = g2;
    hklnormal[2] = g3;
    GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient, OrMat,
                                   &nOrient, ringnr);
    bestnMatchesRot = -1;
    bestnTspotsRot = 0;
    or = 0;
    orDelta = 1;
    while (or < nOrient) {
      int t;
      RealType orThis[3][3];
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          orThis[i][j] = OrMat[or * 9 + i * 3 + j];
      CalcDiffrSpots_Furnace(
          orThis, Params.LatticeConstant, Params.Wavelength, Params.Distance,
          Params.RingRadii, Params.OmegaRanges, Params.BoxSizes,
          Params.NoOfOmegaRanges, Params.ExcludePoleAngle, TheorSpots, &nTspots,
          ringsToRejectCalc, nRingsToRejectCalc, &nTspotsFracCalc);
      MinMatchesToAccept = nTspotsFracCalc * Params.MinMatchesToAcceptFrac;
      bestnMatchesPos = -1;
      bestnTspotsPos = 0;
      calc_n_max_min(xi, yi, ys, y0, Params.Rsample, Params.StepsizePos, &n_max,
                     &n_min);
      n = n_min;
      while (n <= n_max) {
        spot_to_unrotated_coordinates(xi, yi, zi, ys, zs, y0, z0,
                                      Params.StepsizePos, n, omega, &ga, &gb,
                                      &gc);
        if (fabs(gc) > HalfBeam) {
          n++;
          continue;
        }
        for (sp = 0; sp < nTspots; sp++) {
          displacement_spot_needed_COM(ga, gb, gc, TheorSpots[sp][3],
                                       TheorSpots[sp][4], TheorSpots[sp][5],
                                       TheorSpots[sp][6], &Displ_y, &Displ_z);
          TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
          TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
          CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11],
                       &TheorSpots[sp][12]);
          TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                                    TheorSpots[sp][11] * TheorSpots[sp][11]) -
                               Params.RingRadii[(int)TheorSpots[sp][9]];
        }
        CompareSpots(TheorSpots, nTspots, ObsSpotsLab, RefRad, Params.MarginRad,
                     Params.MarginRadial, etamargins, omemargins, &nMatches,
                     GrainSpots, ringsToRejectCalc, nRingsToRejectCalc,
                     &nMatchesFracCalc);
        if (nMatchesFracCalc > bestnMatchesPos) {
          bestnMatchesPos = nMatchesFracCalc;
          bestnTspotsPos = nTspotsFracCalc;
        }
        double fracMatchesThis = (RealType)((RealType)nMatchesFracCalc) /
                                 ((RealType)nTspotsFracCalc);
        if (nMatchesFracCalc >= MinMatchesToAccept &&
            fracMatchesThis >= bestFracTillNow) {
          bestMatchFound = 1;
          for (i = 0; i < 9; i++)
            GrainMatchesT[0][i] = orThis[i / 3][i % 3];
          GrainMatchesT[0][9] = ga;
          GrainMatchesT[0][10] = gb;
          GrainMatchesT[0][11] = gc;
          GrainMatchesT[0][12] = (double)nTspots;
          GrainMatchesT[0][13] = (double)nMatches;
          GrainMatchesT[0][14] = 1;
          for (r = 0; r < nTspots; r++) {
            for (c = 0; c < 15; c++)
              AllGrainSpotsT[r][c] = GrainSpots[r][c];
            AllGrainSpotsT[r][15] = 1;
          }
          CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance);
          if (fracMatchesThis > bestFracTillNow ||
              (fracMatchesThis == bestFracTillNow &&
               GrainMatchesT[0][15] < MinInternalAngle)) {
            bestFracTillNow = fracMatchesThis;
            MinInternalAngle = GrainMatchesT[0][15];
            rownr = nTspots;
            matchNr = 1;
            for (i = 0; i < 16; i++)
              GrainMatches[0][i] = GrainMatchesT[0][i];
            for (r = 0; r < nTspots; r++)
              for (c = 0; c < 17; c++)
                AllGrainSpots[r][c] = AllGrainSpotsT[r][c];
            for (r = nTspots; r < nRowsOutput; r++)
              for (c = 0; c < 17; c++)
                AllGrainSpots[r][c] = 0;
          }
        }
        nDelta = 1;
        if (nTspotsFracCalc != 0) {
          fracMatches = (RealType)nMatchesFracCalc / nTspotsFracCalc;
          if (fracMatches < 0.5) {
            nDelta = 5 - round(fracMatches * (5 - 1) / 0.5);
          }
        }
        n = n + nDelta;
      }
      if (bestnMatchesPos > bestnMatchesRot) {
        bestnMatchesRot = bestnMatchesPos;
        bestnTspotsRot = bestnTspotsPos;
      }
      or = or + orDelta;
    }
    if (bestnMatchesRot > bestnMatchesIsp) {
      bestnMatchesIsp = bestnMatchesRot;
      bestnTspotsIsp = bestnTspotsRot;
    }
    ispDelta = 1;
    if ((!usingFriedelPair) && (bestnTspotsRot != 0)) {
      fracMatches = (RealType)bestnMatchesRot / bestnTspotsRot;
      if (fracMatches < 0.5)
        ispDelta = 3 - round(fracMatches * (3 - 1) / 0.5);
    }
    isp = isp + ispDelta;
  }
  free(OrMat);
  fracMatches = bestFracTillNow;
  if (fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 ||
      (int)bestnMatchesIsp == -1 || bestMatchFound == 0) {
    FreeMemMatrix(GrainMatches, MAX_N_MATCHES);
    FreeMemMatrix(GrainMatchesT, MAX_N_MATCHES);
    FreeMemMatrix(TheorSpots, nRowsPerGrain);
    FreeMemMatrix(GrainSpots, nRowsPerGrain);
    FreeMemMatrix(AllGrainSpots, nRowsOutput);
    FreeMemMatrix(AllGrainSpotsT, nRowsOutput);
    FreeMemMatrix(BestMatches, 2);
    printf("Nothing good found for ID: %d.\n", SpotIDs);
    fflush(stdout);
    return 1;
  }
  double enTm = omp_get_wtime() - sttm;
  BestMatches[SpotIDIdx][0] = SpotIDIdx + 1;
  BestMatches[SpotIDIdx][1] = SpotID;
  BestMatches[SpotIDIdx][2] = bestnTspotsIsp;
  BestMatches[SpotIDIdx][3] = bestnMatchesIsp;
  BestMatches[SpotIDIdx][4] = fracMatches;
  WriteBestMatchBin(GrainMatches, AllGrainSpots, rownr, Params.OutputFolder,
                    offsetLoc);
  printf("IDNr: %d, Total: %d, ID: %d, Confidence: %lf, nExp: %lf, nObs: %lf, "
         "nPlanes: %d, omega: %lf, time: %lfs.\n",
         idNr, totalIDs, SpotIDs, fracMatches, GrainMatches[0][12],
         GrainMatches[0][13], nPlaneNormals, omega, enTm);
  FreeMemMatrix(GrainMatches, MAX_N_MATCHES);
  FreeMemMatrix(GrainMatchesT, MAX_N_MATCHES);
  FreeMemMatrix(TheorSpots, nRowsPerGrain);
  FreeMemMatrix(GrainSpots, nRowsPerGrain);
  FreeMemMatrix(AllGrainSpots, nRowsOutput);
  FreeMemMatrix(AllGrainSpotsT, nRowsOutput);
  FreeMemMatrix(BestMatches, 2);
}

int DoIndexingSeed(double orMat[9], double posThis[3], double RefRad,
                   struct TParams Params, int offsetLoc, int idNr, int totalIDs,
                   int ringsToRejectCalc[], int nRingsToRejectCalc)
// We want to provide the orientation matrix as input, then compute the spots
// and do the rest.
{
  double sttm = omp_get_wtime();
  int nRowsPerGrain = 2 * n_hkls;
  int nRowsOutput = MAX_N_MATCHES * 2 * n_hkls;
  RealType orThis[3][3];
  int i, j, r, c;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      orThis[i][j] = orMat[i * 3 + j];
  int nTspots = 0, nTspotsFrac = 0, nMatchesFracCalc = 0;
  RealType **GrainSpots;
  GrainSpots = allocMatrix(nRowsPerGrain, N_COL_GRAINSPOTS);
  RealType **TheorSpots;
  TheorSpots = allocMatrix(nRowsPerGrain, N_COL_THEORSPOTS);
  RealType **GrainMatches;
  GrainMatches = allocMatrix(MAX_N_MATCHES, N_COL_GRAINMATCHES);
  RealType **AllGrainSpots;
  AllGrainSpots = allocMatrix(nRowsOutput, N_COL_GRAINSPOTS);
  if (AllGrainSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return -1;
  }
  if (GrainMatches == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return -1;
  }

  if (TheorSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return -1;
  }
  RealType omemargins[181];
  RealType etamargins[MAX_N_RINGS];
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
  double ga = posThis[0], gb = posThis[1], gc = posThis[2];
  double Displ_y, Displ_z;
  int sp;
  int nMatches = 0;
  int rownr = 0;
  // Calculate diffraction spots
  CalcDiffrSpots_Furnace(orThis, Params.LatticeConstant, Params.Wavelength,
                         Params.Distance, Params.RingRadii, Params.OmegaRanges,
                         Params.BoxSizes, Params.NoOfOmegaRanges,
                         Params.ExcludePoleAngle, TheorSpots, &nTspots,
                         ringsToRejectCalc, nRingsToRejectCalc, &nTspotsFrac);
  for (sp = 0; sp < nTspots; sp++) {
    displacement_spot_needed_COM(ga, gb, gc, TheorSpots[sp][3],
                                 TheorSpots[sp][4], TheorSpots[sp][5],
                                 TheorSpots[sp][6], &Displ_y, &Displ_z);
    TheorSpots[sp][10] = TheorSpots[sp][4] + Displ_y;
    TheorSpots[sp][11] = TheorSpots[sp][5] + Displ_z;
    CalcEtaAngle(TheorSpots[sp][10], TheorSpots[sp][11], &TheorSpots[sp][12]);
    TheorSpots[sp][13] = sqrt(TheorSpots[sp][10] * TheorSpots[sp][10] +
                              TheorSpots[sp][11] * TheorSpots[sp][11]) -
                         Params.RingRadii[(int)TheorSpots[sp][9]];
  }
  // Compare with observed diffraction spots.
  CompareSpots(TheorSpots, nTspots, ObsSpotsLab, RefRad, Params.MarginRad,
               Params.MarginRadial, etamargins, omemargins, &nMatches,
               GrainSpots, ringsToRejectCalc, nRingsToRejectCalc,
               &nMatchesFracCalc);
  double fracMatchesThis =
      (RealType)((RealType)nMatchesFracCalc) / ((RealType)nTspotsFrac);
  int grID = -1;
  if (fracMatchesThis < Params.MinMatchesToAcceptFrac) {
    printf("%lf %d\n", fracMatchesThis, idNr);
  }
  for (i = 0; i < 9; i++)
    GrainMatches[0][i] = orThis[i / 3][i % 3];
  GrainMatches[0][9] = ga;
  GrainMatches[0][10] = gb;
  GrainMatches[0][11] = gc;
  GrainMatches[0][12] = (double)nTspots;
  GrainMatches[0][13] = (double)nMatches;
  GrainMatches[0][14] = 1;
  for (r = 0; r < nTspots; r++) {
    for (c = 0; c < 15; c++)
      AllGrainSpots[r][c] = GrainSpots[r][c];
    AllGrainSpots[r][15] = 1;
  }
  CalcIA(GrainMatches, 1, AllGrainSpots, Params.Distance);
  rownr = nTspots;
  double enTm = omp_get_wtime() - sttm;
  WriteBestMatchBin(GrainMatches, AllGrainSpots, rownr, Params.OutputFolder,
                    offsetLoc);
  grID = (int)GrainSpots[0][14];
  printf("IDNr: %d, Total: %d, ID: %d, Confidence: %lf, nExp: %lf, nObs: %lf, "
         "time: %lfs.\n",
         idNr, totalIDs, grID, fracMatchesThis, GrainMatches[0][12],
         GrainMatches[0][13], enTm);
  FreeMemMatrix(GrainMatches, MAX_N_MATCHES);
  FreeMemMatrix(TheorSpots, nRowsPerGrain);
  FreeMemMatrix(GrainSpots, nRowsPerGrain);
  FreeMemMatrix(AllGrainSpots, nRowsOutput);
  return grID;
}

int ReadBins(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char file_name[2048];
  sprintf(file_name, "%s/Data.bin", cwd);
  int rc;
  fd = open(file_name, O_RDONLY);
  check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  data = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(data == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  printf("Data.bin read\n");
  int fd2;
  struct stat s2;
  int status2;
  char file_name2[2048];
  sprintf(file_name2, "%s/nData.bin", cwd);
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  ndata = mmap(0, size2, PROT_READ, MAP_SHARED, fd2, 0);
  check(ndata == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  printf("nData.bin read\n");
  printf("%lld %d %lld \n", (long long int)size2, (int)sizeof(int),
         (long long int)(size2 / sizeof(int)));
}

int ReadSpots(char *cwd) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char filename[2048];
  sprintf(filename, "%s/Spots.bin", cwd);
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  ObsSpotsLab = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(ObsSpotsLab == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  return (int)size / (9 * sizeof(double));
}

int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tIndexerOMP v6.0\nContact hsharma@anl.gov in case of "
         "questions about the MIDAS project.\n\n");
  int returncode;
  struct TParams Params;
  char *ParamFN;
  char fn[1024];
  if (argc != 6) {
    printf(
        "Supply a parameter file, blockNr, nBlocks, nSpotsToIndex, numProcs as "
        "arguments: ie %s param.txt blockNr nBlocks nSpotsToIndex numProcs\n\n",
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
  printf("SpaceGroup: %d\n", Params.SpaceGroupNum);
  printf("Finished reading parameters.\n");
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  char aline[4096], dummy[1024];
  fgets(aline, 1000, hklf);
  int Rnr, i;
  int hi, ki, li;
  double hc, kc, lc, RRd, Ds, tht, tth;
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %lf %lf", &hi, &ki, &li, &Ds,
           &Rnr, &hc, &kc, &lc, &tht, &tth, &RRd);
    RingHKL[Rnr][0] = hc;
    RingHKL[Rnr][1] = kc;
    RingHKL[Rnr][2] = lc;
    RingTtheta[Rnr] = tth;
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
  printf("Reading binned data from %s...\n", cwdstr);
  int rc = ReadBins(cwdstr);
  printf("Binned data read.\n");
  fflush(stdout);
  int HighestRingNo = 0;
  for (i = 0; i < MAX_N_RINGS; i++) {
    if (Params.RingRadii[i] != 0)
      HighestRingNo = i;
  }
  n_ring_bins = HighestRingNo;
  n_eta_bins = ceil(360.0 / Params.EtaBinSize);
  n_ome_bins = ceil(360.0 / Params.OmeBinSize);
  EtaBinSize = Params.EtaBinSize;
  OmeBinSize = Params.OmeBinSize;
  printf("No of bins for rings : %d\n", n_ring_bins);
  printf("No of bins for eta   : %d\n", n_eta_bins);
  printf("No of bins for omega : %d\n", n_ome_bins);
  printf("Total no of bins     : %d\n\n",
         n_ring_bins * n_eta_bins * n_ome_bins);
  printf("Finished binning.\n\n");
  fflush(stdout);
  int numProcs = atoi(argv[5]);
  if (Params.isGrainsInput == 0) {
    int *SpotIDs;
    int nSpotIDs;
    int nBlocks = atoi(argv[3]);
    int blockNr = atoi(argv[2]);
    int nSpotsToIndex = atoi(argv[4]);
    int startRowNr;
    int endRowNr;
    startRowNr = (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * blockNr;
    int tmp =
        (int)(ceil((double)nSpotsToIndex / (double)nBlocks)) * (blockNr + 1);
    endRowNr = tmp < (nSpotsToIndex - 1) ? tmp : (nSpotsToIndex - 1);
    nSpotIDs = endRowNr - startRowNr + 1;
    SpotIDs = malloc(nSpotIDs * sizeof(*SpotIDs));
    FILE *spotsFile = fopen("SpotsToIndex.csv", "r");
    for (i = 0; i < startRowNr; i++) {
      fgets(aline, 1000, spotsFile);
    }
    for (i = 0; i < nSpotIDs; i++) {
      fgets(aline, 1000, spotsFile);
      sscanf(aline, "%d", &SpotIDs[i]);
    }
    fclose(spotsFile);
    printf("Read spots info, nSpots = %d, %d %d\n", nSpotIDs, numProcs,
           omp_get_max_threads());
    fflush(stdout);
    printf("%d\n", omp_get_max_threads());
    int thisRowNr;
#pragma omp parallel for num_threads(numProcs) private(thisRowNr)              \
    schedule(dynamic)
    for (thisRowNr = 0; thisRowNr < nSpotIDs; thisRowNr++) {
      int thisSpotID = SpotIDs[thisRowNr];
      int idRow = thisRowNr + startRowNr;
      DoIndexing(thisSpotID, Params, idRow, thisRowNr, nSpotIDs,
                 Params.RingsToReject, Params.nRingsToRejectCalc);
    }
    free(SpotIDs);
  } else {
    // Read the orientations from the grains.csv file, then do forward
    // simulation and match. Assuming we will only use one node!!!!! Also we
    // need to write out the SpotsToIndex.csv file!!!
    int nGrains = 0, grainNr = 0;
    FILE *inpF;
    inpF = fopen(Params.GrainsFileName, "r");
    fgets(aline, 4096, inpF);
    sscanf(aline, "%s %d", dummy, &nGrains);
    RealType **orients;
    RealType **positions;
    RealType *radii;
    int *IDsFiles;
    IDsFiles = malloc(nGrains * sizeof(*IDsFiles));
    orients = allocMatrix(nGrains, 9);
    positions = allocMatrix(nGrains, 3);
    radii = calloc(nGrains, sizeof(*radii));
    for (i = 0; i < 8; i++)
      fgets(aline, 4096, inpF);
    while (fgets(aline, 4096, inpF) != NULL) {
      sscanf(aline,
             "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %s "
             "%s %s %s %s %s %lf ",
             &IDsFiles[grainNr], &orients[grainNr][0], &orients[grainNr][1],
             &orients[grainNr][2], &orients[grainNr][3], &orients[grainNr][4],
             &orients[grainNr][5], &orients[grainNr][6], &orients[grainNr][7],
             &orients[grainNr][8], &positions[grainNr][0],
             &positions[grainNr][1], &positions[grainNr][2], dummy, dummy,
             dummy, dummy, dummy, dummy, dummy, dummy, dummy, &radii[grainNr]);
      grainNr++;
    }
    // We have all the info, get back the results.
    int *spotsIndexed;
    spotsIndexed = malloc(nGrains * 2 * sizeof(*spotsIndexed));
    if (spotsIndexed == NULL) {
      printf("Memory error: could not allocate memory for spotsIndexed array. "
             "Memory full?\n");
      exit(EXIT_FAILURE);
    }
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (grainNr = 0; grainNr < nGrains; grainNr++) {
      // get the corresponding orient and position, feed to DoIndexingSeed
      spotsIndexed[grainNr * 2 + 0] = DoIndexingSeed(
          orients[grainNr], positions[grainNr], radii[grainNr], Params, grainNr,
          grainNr, nGrains, Params.RingsToReject, Params.nRingsToRejectCalc);
      spotsIndexed[grainNr * 2 + 1] = IDsFiles[grainNr];
    }
    // Write the SpotsToIndex.csv file, this will have 2 IDS, first will be the
    // newID, next is the original ID.
    printf("Writing SpotsToIndex.csv file with %d grains.\n", nGrains);
    fflush(stdout);
    FILE *spotsFile = fopen("SpotsToIndex.csv", "w");
    if (spotsFile == NULL) {
      printf("Error: Could not open SpotsToIndex.csv for writing.\n");
      exit(EXIT_FAILURE);
    }
    for (grainNr = 0; grainNr < nGrains; grainNr++) {
      fprintf(spotsFile, "%d %d\n", spotsIndexed[grainNr + 0],
              spotsIndexed[grainNr * 2 + 1]);
    }
    fclose(spotsFile);
    free(spotsIndexed);
    FreeMemMatrix(orients, nGrains);
    FreeMemMatrix(positions, nGrains);
    free(radii);
    free(IDsFiles);
  }
  double time = omp_get_wtime() - start_time;
  printf("Finished, time elapsed: %lf seconds.\n", time);
}
