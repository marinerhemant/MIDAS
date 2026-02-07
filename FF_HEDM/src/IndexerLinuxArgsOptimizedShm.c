//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
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

// To store the orientation matrices
RealType OrMat[MAX_N_OR][3][3];

// hkls to use
double hkls[MAX_N_HKLS][7];
int n_hkls = 0;
int HKLints[MAX_N_HKLS][4];
double ABCABG[6];

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
  // calc the diff. NOte: smallest diff in pos is choosen
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
      if (fabs(cosome1) <= 1) {
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
                            RealType **spots, int *nspots) {
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
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    ringnr = (int)(hkls[indexhkl][3]);
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
      }
    }
  }
  *nspots = spotnr;
}

void CompareSpots(RealType **TheorSpots, int nTheorSpots, RealType *ObsSpots,
                  RealType RefRad, RealType MarginRad, RealType MarginRadial,
                  RealType etamargins[], RealType omemargins[], int *nMatch,
                  RealType **GrainSpots) {
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
    printf("The spot is too close to the poles. This technique to find mixed "
           "friedel pair would not work satisfactorily. So don't use mixed "
           "friedel pair.\n");
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
  char SpotsFileName[4096];
  char IDsFileName[4096];
  int UseFriedelPairs;
};

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
  fp = fopen(FileName, "r");
  if (fp == NULL) {
    printf("Cannot open file: %s.\n", FileName);
    return (1);
  }
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) {
    str = "RingNumbers ";
    cmpres = strncmp(line, str, strlen(str));
    if (cmpres == 0) {
      sscanf(line, "%s %d", dummy, &(Params->RingNumbers[NoRingNumbers]));
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
      long long int sz = ReadBigDet();
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

void CalcIA(RealType **GrainMatches, int ngrains, RealType **AllGrainSpots,
            RealType distance) {
  RealType *IAgrainspots;
  int r, g;
  RealType g1x, g1y, g1z;
  RealType x1, y1, z1, w1, x2, y2, z2, w2, gv1x, gv1y, gv1z, gv2x, gv2y, gv2z;
  int nspots;
  int rt = 0;
  IAgrainspots = malloc(1000 * sizeof(*IAgrainspots));
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

int DoIndexing(int SpotIDs, int nSpotIDs, struct TParams Params) {
  clock_t start, end;
  double dif;
  RealType HalfBeam = Params.Hbeam / 2;
  RealType MinMatchesToAccept;
  RealType ga, gb, gc;
  int nTspots;
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
  int r, c, i;
  RealType y0_vector[MAX_N_STEPS];
  RealType z0_vector[MAX_N_STEPS];
  int nPlaneNormals;
  double hkl[3];
  RealType g1, g2, g3;
  int isp;
  RealType xi, yi, zi;
  int n_max, n_min, n;
  RealType y0, z0;
  RealType RingTtheta[MAX_N_RINGS];
  int RingMult[MAX_N_RINGS];
  double RingHKL[MAX_N_RINGS][3];
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
  BestMatches = allocMatrix(nSpotIDs, 5);
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
  int SpotIDIdx;
  printf("Starting indexing...\n");
  for (SpotIDIdx = 0; SpotIDIdx < nSpotIDs; SpotIDIdx++) {
    start = clock();
    RealType MinInternalAngle = 1000;
    matchNr = 0;
    rownr = 0;
    RealType SpotID = SpotIDs;
    FindInMatrix(&ObsSpotsLab[0 * 9 + 0], n_spots, N_COL_OBSSPOTS, 4, SpotID,
                 &SpotRowNo);
    if (SpotRowNo == -1) {
      printf("WARNING: SpotId %lf not found in spots file! Ignoring this "
             "spotID.\n",
             SpotID);
      continue;
    }
    RealType ys = ObsSpotsLab[SpotRowNo * 9 + 0];
    RealType zs = ObsSpotsLab[SpotRowNo * 9 + 1];
    RealType omega = ObsSpotsLab[SpotRowNo * 9 + 2];
    RealType RefRad = ObsSpotsLab[SpotRowNo * 9 + 3];
    RealType eta = ObsSpotsLab[SpotRowNo * 9 + 6];
    RealType ttheta = ObsSpotsLab[SpotRowNo * 9 + 7];
    int ringnr = (int)ObsSpotsLab[SpotRowNo * 9 + 5];
    char *hklfn = "hkls.csv";
    FILE *hklf = fopen(hklfn, "r");
    char aline[1024], dummy[1000];
    fgets(aline, 1000, hklf);
    int Rnr;
    double hc, kc, lc, tth;
    for (i = 0; i < MAX_N_RINGS; i++)
      RingMult[i] = 0;
    while (fgets(aline, 1000, hklf) != NULL) {
      sscanf(aline, "%s %s %s %s %d %lf %lf %lf %s %lf %s", dummy, dummy, dummy,
             dummy, &Rnr, &hc, &kc, &lc, dummy, &tth, dummy);
      RingMult[Rnr]++;
      RingHKL[Rnr][0] = hc;
      RingHKL[Rnr][1] = kc;
      RingHKL[Rnr][2] = lc;
      RingTtheta[Rnr] = tth;
    }
    hkl[0] = RingHKL[ringnr][0];
    hkl[1] = RingHKL[ringnr][1];
    hkl[2] = RingHKL[ringnr][2];
    printf("\n-----------------------------------------------------------------"
           "---------\n");
    printf("Spot number being processed %d of %d\n\n", SpotIDIdx, nSpotIDs);
    printf("%8s %10s %9s %9s %9s %9s %9s %7s\n", "SpotID", "SpotRowNo", "ys",
           "zs", "omega", "eta", "ttheta", "ringno");
    printf("%8.0f %10d %9.2f %9.2f %9.3f %9.3f %9.3f %7d\n\n", SpotID,
           SpotRowNo, ys, zs, omega, eta, ttheta, ringnr);
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
        printf("No Friedel pair found, will try everything.\n");
      }
      usingFriedelPair = 0;
      printf("Trying all plane normals.\n");
      GenerateIdealSpots(ys, zs, RingTtheta[ringnr], eta,
                         Params.RingRadii[ringnr], Params.Rsample, Params.Hbeam,
                         Params.StepsizePos, y0_vector, z0_vector,
                         &nPlaneNormals);
    }
    printf("No of Plane normals: %d\n\n", nPlaneNormals);
    bestnMatchesIsp = -1;
    bestnTspotsIsp = 0;
    isp = 0;
    int bestMatchFound = 0;
    while (isp < nPlaneNormals) {
      y0 = y0_vector[isp];
      z0 = z0_vector[isp];
      MakeUnitLength(Params.Distance, y0, z0, &xi, &yi, &zi);
      spot_to_gv(xi, yi, zi, omega, &g1, &g2, &g3);
      hklnormal[0] = g1;
      hklnormal[1] = g2;
      hklnormal[2] = g3;
      GenerateCandidateOrientationsF(hkl, hklnormal, Params.StepsizeOrient,
                                     OrMat, &nOrient, ringnr);
      bestnMatchesRot = -1;
      bestnTspotsRot = 0;
      or = 0;
      orDelta = 1;
      while (or < nOrient) {
        int t;
        CalcDiffrSpots_Furnace(OrMat[or], Params.LatticeConstant,
                               Params.Wavelength, Params.Distance,
                               Params.RingRadii, Params.OmegaRanges,
                               Params.BoxSizes, Params.NoOfOmegaRanges,
                               Params.ExcludePoleAngle, TheorSpots, &nTspots);
        MinMatchesToAccept = nTspots * Params.MinMatchesToAcceptFrac;
        bestnMatchesPos = -1;
        bestnTspotsPos = 0;
        calc_n_max_min(xi, yi, ys, y0, Params.Rsample, Params.StepsizePos,
                       &n_max, &n_min);
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
          CompareSpots(TheorSpots, nTspots, ObsSpotsLab, RefRad,
                       Params.MarginRad, Params.MarginRadial, etamargins,
                       omemargins, &nMatches, GrainSpots);
          if (nMatches > bestnMatchesPos) {
            bestnMatchesPos = nMatches;
            bestnTspotsPos = nTspots;
          }
          if ((nMatches > 0) && (matchNr < 100) &&
              (nMatches >= MinMatchesToAccept)) {
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
            CalcIA(GrainMatchesT, 1, AllGrainSpotsT, Params.Distance);
            if (GrainMatchesT[0][15] < MinInternalAngle) {
              MinInternalAngle = GrainMatchesT[0][15];
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
          nDelta = 1;
          if (nTspots != 0) {
            fracMatches = (RealType)nMatches / nTspots;
            if (fracMatches < 0.5) {
              nDelta = 10 - round(fracMatches * (10 - 1) / 0.5);
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
          ispDelta = 5 - round(fracMatches * (5 - 1) / 0.5);
      }
      printf(
          "==> planenormal #pns #or #pos #Theor #Matches: %d %d %d %d %d %d\n",
          isp, nPlaneNormals, nOrient, 2 * n_max + 1, bestnTspotsRot,
          bestnMatchesRot);
      isp = isp + ispDelta;
      if (matchNr >= 100) {
        printf("Warning: the number of grain matches exceeds maximum (%d). Not "
               "all output is saved!\n",
               MAX_N_MATCHES);
      }
    }

    fracMatches = (RealType)bestnMatchesIsp / bestnTspotsIsp;
    printf("\n==> Best Match: No_of_theoretical_spots No_of_spots_found "
           "fraction: %d %d %0.2f\n",
           bestnTspotsIsp, bestnMatchesIsp, fracMatches);
    if (fracMatches > 1 || fracMatches < 0 || (int)bestnTspotsIsp == 0 ||
        (int)bestnMatchesIsp == -1 || bestMatchFound == 0) {
      printf("Nothing good was found. Exiting.\n");
      exit(0);
    }
    BestMatches[SpotIDIdx][0] = SpotIDIdx + 1;
    BestMatches[SpotIDIdx][1] = SpotID;
    BestMatches[SpotIDIdx][2] = bestnTspotsIsp;
    BestMatches[SpotIDIdx][3] = bestnMatchesIsp;
    BestMatches[SpotIDIdx][4] = fracMatches;
    end = clock();
    dif = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed [s] [min]: %f %f\n", dif, dif / 60);
    CreateNumberedFilenameW("BestGrain_", (int)SpotID, 9, ".txt", fn);
    MakeFullFileName(ffn, Params.OutputFolder, fn);
    CreateNumberedFilenameW("BestPos_", (int)SpotID, 9, ".csv", fn2);
    MakeFullFileName(ffn2, Params.OutputFolder, fn2);
    WriteBestMatch(ffn, GrainMatches, matchNr, AllGrainSpots, rownr, ffn2);
  }
  FreeMemMatrix(GrainMatches, MAX_N_MATCHES);
  FreeMemMatrix(TheorSpots, nRowsPerGrain);
  FreeMemMatrix(GrainSpots, nRowsPerGrain);
  FreeMemMatrix(AllGrainSpots, nRowsOutput);
  FreeMemMatrix(BestMatches, nSpotIDs);
}

void ConcatStr(char *str1, char *str2, char *resStr) {
  strcpy(resStr, str1);
  strcat(resStr, str2);
}

int ReadBins() {
  int fd;
  struct stat s;
  int status;
  size_t size;
  const char *file_name = "/dev/shm/Data.bin";
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
  const char *file_name2 = "/dev/shm/nData.bin";
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  ndata = mmap(0, size2, PROT_READ, MAP_SHARED, fd2, 0);
  printf("%lld %d %lld \n", (long long int)size2, (int)sizeof(int),
         (long long int)(size2 / sizeof(int)));
  fflush(stdout);
  check(ndata == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  return 1;
}

int ReadSpots() {
  int fd;
  struct stat s;
  int status;
  size_t size;
  const char *filename = "/dev/shm/Spots.bin";
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

int ReadBigDet() {
  int fd;
  struct stat s;
  int status;
  size_t size;
  const char *filename = "/dev/shm/BigDetectorMask.bin";
  int rc;
  fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  BigDetector = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(BigDetector == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  return (long long int)size;
}

int UnMap() {
  int fd;
  struct stat s;
  int status;
  size_t size;
  const char *file_name = "/dev/shm/Data.bin";
  int rc;
  fd = open(file_name, O_RDONLY);
  check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  rc = munmap(data, size);
  int fd2;
  struct stat s2;
  int status2;
  const char *file_name2 = "/dev/shm/nData.bin";
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  rc = munmap(ndata, size2);
  int fd3;
  struct stat s3;
  int status3;
  const char *filename3 = "/dev/shm/Spots.bin";
  fd3 = open(filename3, O_RDONLY);
  check(fd3 < 0, "open %s failed: %s", filename3, strerror(errno));
  status3 = fstat(fd3, &s3);
  check(status3 < 0, "stat %s failed: %s", filename3, strerror(errno));
  size_t size3 = s3.st_size;
  rc = munmap(ObsSpotsLab, size3);
  return 1;
}

int main(int argc, char *argv[]) {
  printf("\n\n\t\tIndexer v5.0\nContact hsharma@anl.gov in case of questions "
         "about the MIDAS project.\n\n");
  clock_t end, start0;
  double diftotal;
  int returncode;
  struct TParams Params;
  int SpotIDs;
  int nSpotIDs;
  char *ParamFN;
  char fn[1024];
  if (argc != 3) {
    printf("Supply a parameter file and a spotID as argument: ie %s param.txt "
           "SpotID\n\n",
           argv[0]);
    exit(EXIT_FAILURE);
  }
  ParamFN = argv[1];
  SpotIDs = atoi(argv[2]);
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
  char aline[1024], dummy[1024];
  fgets(aline, 1000, hklf);
  int Rnr, i;
  int hi, ki, li;
  double hc, kc, lc, RRd, Ds, tht;
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %lf %d %lf %lf %lf %lf %s %lf", &hi, &ki, &li, &Ds,
           &Rnr, &hc, &kc, &lc, &tht, dummy, &RRd);
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
  printf("No of hkl's: %d\n", n_hkls);
  int t;
  n_spots = ReadSpots();
  nSpotIDs = 1;
  printf("Binned data...\n");
  int rc = ReadBins();
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
  start0 = clock();
  DoIndexing(SpotIDs, nSpotIDs, Params);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("\nTotal time elapsed [s] [min]: %f %f\n", diftotal, diftotal / 60);
  int tc = UnMap();
  return (0);
}
