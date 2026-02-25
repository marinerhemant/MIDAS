//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "nf_headers.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define RealType double
#define float32_t float
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

extern int Flag;
extern double Wedge;
extern double Wavelength;
extern double OmegaRang[MAX_N_OMEGA_RANGES][2];
extern int nOmeRang;

double **allocMatrix(int nrows, int ncols) {
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

int **allocMatrixInt(int nrows, int ncols) {
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

void FreeMemMatrix(RealType **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

void Convert9To3x3(double MatIn[9], double MatOut[3][3]) {
  int i, j, k = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      MatOut[i][j] = MatIn[k];
      k++;
    }
  }
}

void Convert3x3To9(double MatIn[3][3], double MatOut[9]) {
  int i, j, k = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      MatOut[k] = MatIn[i][j];
      k++;
    }
  }
}

struct Theader {
  uint32_t uBlockHeader;
  uint16_t BlockType;
  uint16_t DataFormat;
  uint16_t NumChildren;
  uint16_t NameSize;
  char BlockName[4096];
  uint32_t DataSize;
  uint16_t ChunkNumber;
  uint16_t TotalChunks;
};

void ReadHeader(FILE *fp, struct Theader *head) {
  fread(&head->uBlockHeader, sizeof(uint32_t), 1, fp);
  fread(&head->BlockType, sizeof(uint16_t), 1, fp);
  fread(&head->DataFormat, sizeof(uint16_t), 1, fp);
  fread(&head->NumChildren, sizeof(uint16_t), 1, fp);
  fread(&head->NameSize, sizeof(uint16_t), 1, fp);
  fread(&head->DataSize, sizeof(uint32_t), 1, fp);
  fread(&head->ChunkNumber, sizeof(uint16_t), 1, fp);
  fread(&head->TotalChunks, sizeof(uint16_t), 1, fp);
  fread(&head->BlockName, (sizeof(char) * (head->NameSize)), 1, fp);
}

static inline void realloc_buffers(int nElements, int nElements_previous,
                                   uint16_t **ys, uint16_t **zs,
                                   uint16_t **peakID, float32_t **intensity) {
  if (nElements > nElements_previous) {
    *ys = realloc(*ys, nElements * sizeof(**ys));
    *zs = realloc(*zs, nElements * sizeof(**zs));
    *peakID = realloc(*peakID, nElements * sizeof(**peakID));
    *intensity = realloc(*intensity, nElements * sizeof(**intensity));
  }
}

int ReadBinFiles(char FileStem[1000], char *ext, int StartNr, int EndNr,
                 int *ObsSpotsMat, int nLayers, long long int ObsSpotsSize,
                 int NrPixelsY, int NrPixelsZ) {
  int i, j, k, nElements = 0, nElements_previous, nCheck, ythis, zthis,
               NrOfFiles, NrOfPixels;
  long long int BinNr;
  long long int TempCntr;
  float32_t dummy;
  uint32_t dummy2;
  FILE *fp;
  char FileName[1024];
  struct Theader Header1;
  uint16_t *ys = NULL, *zs = NULL, *peakID = NULL;
  float32_t *intensity = NULL;
  int counter = 0;
  NrOfFiles = EndNr - StartNr + 1;
  NrOfPixels = NrPixelsY * NrPixelsZ;
  long long int kT;
  kT = ObsSpotsSize * 32;
  /*for (kT=0;kT<ObsSpotsSize;kT++){
      ObsSpotsMat[k] = 0;
  }*/
  for (k = 0; k < nLayers; k++) {
    for (i = StartNr; i <= EndNr; i++) {
      sprintf(FileName, "%s_%06d.%s%d", FileStem, i, ext, k);
      // printf("Reading file : %s\n",FileName);
      fp = fopen(FileName, "r");
      fread(&dummy, sizeof(float32_t), 1, fp);
      ReadHeader(fp, &Header1);
      fread(&dummy2, sizeof(uint32_t), 1, fp);
      fread(&dummy2, sizeof(uint32_t), 1, fp);
      fread(&dummy2, sizeof(uint32_t), 1, fp);
      fread(&dummy2, sizeof(uint32_t), 1, fp);
      fread(&dummy2, sizeof(uint32_t), 1, fp);
      ReadHeader(fp, &Header1);
      nElements_previous = nElements;
      nElements = (Header1.DataSize - Header1.NameSize) / 2;
      realloc_buffers(nElements, nElements_previous, &ys, &zs, &peakID,
                      &intensity);
      fread(ys, sizeof(uint16_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 2;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      fread(zs, sizeof(uint16_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 4;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      fread(intensity, sizeof(float32_t) * nElements, 1, fp);
      ReadHeader(fp, &Header1);
      nCheck = (Header1.DataSize - Header1.NameSize) / 2;
      if (nCheck != nElements) {
        printf("Number of elements mismatch.\n");
        return 0;
      }
      fread(peakID, sizeof(uint16_t) * nElements, 1, fp);
      for (j = 0; j < nElements; j++) {
        ythis = (int)ys[j];
        zthis = (int)zs[j];
        if (Flag == 1)
          zthis = NrPixelsZ - zthis;
        BinNr = k;
        BinNr *= NrOfFiles;
        BinNr *= NrOfPixels;
        TempCntr = NrOfPixels;
        TempCntr *= counter;
        BinNr += TempCntr;
        BinNr += (ythis * ((long long int)NrPixelsZ));
        BinNr += zthis;
        if (BinNr < 0 || BinNr > kT) {
          // printf("BinNr was out of bounds.\n");
          // printf("%lld %lld %d %d %d %d %lld %d %d %lld %lld\n",BinNr, k,
          // NrOfFiles, NrOfPixels, counter, TempCntr, ythis, zthis,
          // ObsSpotsSize, kT);fflush(stdout);
          return 0;
        }
        SetBit(ObsSpotsMat, BinNr);
      }
      fclose(fp);
      counter += 1;
    }
    counter = 0;
  }
  return 1;
  free(ys);
  free(zs);
  free(peakID);
  free(intensity);
}

void MatrixMultF(RealType m[3][3], RealType v[3], RealType r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
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

void RotationTilts(double tx, double ty, double tz, double RotMatOut[3][3]) {
  tx = deg2rad * tx;
  ty = deg2rad * ty;
  tz = deg2rad * tz;
  double r1[3][3];
  double r2[3][3];
  double r3[3][3];
  double r1r2[3][3];
  r1[0][0] = cos(tz);
  r1[0][1] = -sin(tz);
  r1[0][2] = 0;
  r1[1][0] = sin(tz);
  r1[1][1] = cos(tz);
  r1[1][2] = 0;
  r1[2][0] = 0;
  r1[2][1] = 0;
  r1[2][2] = 1;
  r2[0][0] = cos(ty);
  r2[0][1] = 0;
  r2[0][2] = sin(ty);
  r2[1][0] = 0;
  r2[1][1] = 1;
  r2[1][2] = 0;
  r2[2][0] = -sin(ty);
  r2[2][1] = 0;
  r2[2][2] = cos(ty);
  r3[0][0] = 1;
  r3[0][1] = 0;
  r3[0][2] = 0;
  r3[1][0] = 0;
  r3[1][1] = cos(tx);
  r3[1][2] = -sin(tx);
  r3[2][0] = 0;
  r3[2][1] = sin(tx);
  r3[2][2] = cos(tx);
  MatrixMultF33(r1, r2, r1r2);
  MatrixMultF33(r1r2, r3, RotMatOut);
}

void RotateAroundZ(RealType v1[3], RealType alpha, RealType v2[3]) {
  RealType cosa = cos(alpha * deg2rad);
  RealType sina = sin(alpha * deg2rad);
  RealType mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};
  MatrixMultF(mat, v1, v2);
}

void DisplacementSpots(RealType a, RealType b, RealType Lsd, RealType yi,
                       RealType zi, RealType omega, RealType *Displ_y,
                       RealType *Displ_z) {
  RealType OmegaRad = deg2rad * omega;
  RealType sinOme = sin(OmegaRad);
  RealType cosOme = cos(OmegaRad);
  RealType xa = a * cosOme - b * sinOme;
  RealType ya = a * sinOme + b * cosOme;
  RealType t = 1 - (xa / Lsd);
  *Displ_y = ya + (yi * t);
  *Displ_z = t * zi;
}

static inline void DisplacementSpotsPrecomp(RealType a, RealType b,
                                            RealType Lsd, RealType yi,
                                            RealType zi, RealType sinOme,
                                            RealType cosOme, RealType *Displ_y,
                                            RealType *Displ_z) {
  RealType xa = a * cosOme - b * sinOme;
  RealType ya = a * sinOme + b * cosOme;
  RealType t = 1 - (xa / Lsd);
  *Displ_y = ya + (yi * t);
  *Displ_z = t * zi;
}

struct Point2D {
  int x, y;
};

int orient2d(struct Point2D a, struct Point2D b, struct Point2D c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double distSq2d(struct Point2D a, struct Point2D b, struct Point2D c) {
  double num = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  double denSq = ((a.y - b.y) * (a.y - b.y)) + ((b.x - a.x) * (b.x - a.x));
  return (num * num) / denSq;
}

void CalcPixels2(double Edges[3][2], int **Pixels, int *counter) {
  int i;
  double minX = 10000000, maxX = -10000000, minY = 100000000, maxY = -100000000;
  for (i = 0; i < 3; i++) {
    Edges[i][0] = round(Edges[i][0]);
    Edges[i][1] = round(Edges[i][1]);
  }
  for (i = 0; i < 3; i++) {
    if (Edges[i][0] < minX) {
      minX = Edges[i][0];
    }
    if (Edges[i][0] > maxX) {
      maxX = Edges[i][0];
    }
    if (Edges[i][1] < minY) {
      minY = Edges[i][1];
    }
    if (Edges[i][1] > maxY) {
      maxY = Edges[i][1];
    }
  }
  *counter = 0;
  int A01 = Edges[0][1] - Edges[1][1], B01 = Edges[1][0] - Edges[0][0];
  int A12 = Edges[1][1] - Edges[2][1], B12 = Edges[2][0] - Edges[1][0];
  int A20 = Edges[2][1] - Edges[0][1], B20 = Edges[0][0] - Edges[2][0];
  struct Point2D p = {minX, minY};
  struct Point2D v0 = {Edges[0][0], Edges[0][1]};
  struct Point2D v1 = {Edges[1][0], Edges[1][1]};
  struct Point2D v2 = {Edges[2][0], Edges[2][1]};
  int w0_row = orient2d(v1, v2, p);
  int w1_row = orient2d(v2, v0, p);
  int w2_row = orient2d(v0, v1, p);
  for (p.y = minY; p.y <= maxY; p.y++) {
    int w0 = w0_row;
    int w1 = w1_row;
    int w2 = w2_row;
    for (p.x = minX; p.x <= maxX; p.x++) {
      if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        Pixels[*counter][0] = p.x;
        Pixels[*counter][1] = p.y;
        *counter += 1;
      } else if (distSq2d(v1, v2, p) < 0.9801) {
        Pixels[*counter][0] = p.x;
        Pixels[*counter][1] = p.y;
        *counter += 1;
      } else if (distSq2d(v2, v0, p) < 0.9801) {
        Pixels[*counter][0] = p.x;
        Pixels[*counter][1] = p.y;
        *counter += 1;
      } else if (distSq2d(v0, v1, p) < 0.9801) {
        Pixels[*counter][0] = p.x;
        Pixels[*counter][1] = p.y;
        *counter += 1;
      }
      w0 += A12;
      w1 += A20;
      w2 += A01;
    }
    w0_row += B12;
    w1_row += B20;
    w2_row += B01;
  }
}

static inline double CalcEta(RealType y, RealType z) {
  double alpha;
  alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline double CorrectWedge(double eta, double theta, double wl,
                                  double wedge) {
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
  double G1a = (k1f);
  double G2a = (k2f);
  double G3a = k3f;
  double LenGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / LenGa;
  double g2 = G2a * ds / LenGa;
  double g3 = G3a * ds / LenGa;
  SinW = 0;
  CosW = 1;
  double LenG = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(LenG * LenG * wl) / 2;
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
  double OmeDiff1 = fabs(Omega1);
  double OmeDiff2 = fabs(Omega2);
  double Omega;
  if (OmeDiff1 < OmeDiff2)
    Omega = Omega1;
  else
    Omega = Omega2;
  return Omega;
}

void CalcFracOverlap(const int NrOfFiles, const int nLayers, const int nTspots,
                     double *TheorSpots, double OmegaStart, double OmegaStep,
                     double XGrain[3], double YGrain[3],
                     const double Lsds[nLayers],
                     const long long int SizeObsSpots, double RotMatTilts[3][3],
                     const double px, const double ybcs[nLayers],
                     const double zbcs[nLayers], const double gs,
                     double P0All[nLayers][3], const int NrPixelsGrid,
                     int *ObsSpotsInfo, double OrientMatIn[3][3],
                     double *FracOver, int **InPixels, int NrPixelsY,
                     int NrPixelsZ) {
  int j, OmeBin, OutofBounds, k, l;
  double OmegaThis, ythis, zthis, XGT, YGT, Displ_Y, Displ_Z, ytemp, ztemp,
      xyz[3], P1[3], ABC[3], outxyz[3], YZSpots[3][2], Lsd, ybc, zbc, P0[3],
      YZSpotsTemp[2], YZSpotsT[3][2];
  int NrInPixels, OverlapPixels, Layer, omeRangNr;
  double eta, RingRadius, theta, omediff;
  long long int BinNr, TempCntr;
  int MultY, MultZ, AllDistsFound, TotalPixels;
  *FracOver = 0;
  // InPixels allocation hoisted to caller
  OverlapPixels = 0;
  TotalPixels = 0;
  for (k = 0; k < 3; k++) {
    P0[k] = P0All[0][k];
  }
  Lsd = Lsds[0];
  ybc = ybcs[0];
  zbc = zbcs[0];
  for (j = 0; j < nTspots; j++) {
    ythis = TheorSpots[j * 3 + 0];
    zthis = TheorSpots[j * 3 + 1];
    OutofBounds = 1;
    if (Wedge != 0) {
      eta = CalcEta(ythis, zthis);
      RingRadius = sqrt(ythis * ythis + zthis * zthis);
      theta = rad2deg * atan(RingRadius / Lsds[0]);
      omediff = CorrectWedge(eta, theta, Wavelength, Wedge);
      OmegaThis = TheorSpots[j * 3 + 2] - omediff;
      // Check if we just went outside the omegaRange, then make OutofBounds =
      // 1;
      if (OmegaThis >= 180) {
        OmegaThis -= 360;
      } else if (OmegaThis <= -180) {
        OmegaThis += 360;
      }
      for (omeRangNr = 0; omeRangNr < nOmeRang; omeRangNr++) {
        if (OmegaThis > OmegaRang[omeRangNr][0] &&
            OmegaThis < OmegaRang[omeRangNr][1]) {
          OutofBounds = 0;
          break;
        }
      }
    } else {
      OmegaThis = TheorSpots[j * 3 + 2];
      OutofBounds = 0;
    }
    OmeBin = (int)floor((-OmegaStart + OmegaThis) / OmegaStep);
    if (OmeBin < 0 || OmeBin >= NrOfFiles) {
      OutofBounds = 1;
    }
    double OmegaRad = deg2rad * OmegaThis;
    double sinOme = sin(OmegaRad);
    double cosOme = cos(OmegaRad);
    for (k = 0; k < 3; k++) {
      XGT = XGrain[k];
      YGT = YGrain[k];
      DisplacementSpotsPrecomp(XGT, YGT, Lsd, ythis, zthis, sinOme, cosOme,
                               &Displ_Y, &Displ_Z);
      ytemp = Displ_Y;
      ztemp = Displ_Z;
      xyz[0] = 0;
      xyz[1] = ytemp;
      xyz[2] = ztemp;
      MatrixMultF(RotMatTilts, xyz, P1);
      for (l = 0; l < 3; l++) {
        ABC[l] = P1[l] - P0[l];
      }
      outxyz[0] = 0;
      outxyz[1] = P0[1] - (ABC[1] * P0[0]) / (ABC[0]);
      outxyz[2] = P0[2] - (ABC[2] * P0[0]) / (ABC[0]);
      YZSpotsT[k][0] = (outxyz[1]) / px + ybc;
      YZSpotsT[k][1] = (outxyz[2]) / px + zbc;
      if (YZSpotsT[k][0] > NrPixelsY || YZSpotsT[k][0] < 0 ||
          YZSpotsT[k][1] > NrPixelsZ || YZSpotsT[k][1] < 0) {
        OutofBounds = 1;
        break;
      }
      if (k == 2) {
        xyz[0] = 0;
        xyz[1] = ythis;
        xyz[2] = zthis;
        MatrixMultF(RotMatTilts, xyz, P1);
        for (l = 0; l < 3; l++) {
          ABC[l] = P1[l] - P0[l];
        }
        outxyz[0] = 0;
        outxyz[1] = P0[1] - (ABC[1] * P0[0]) / (ABC[0]);
        outxyz[2] = P0[2] - (ABC[2] * P0[0]) / (ABC[0]);
        YZSpotsTemp[0] = (outxyz[1]) / px + ybc;
        YZSpotsTemp[1] = (outxyz[2]) / px + zbc;
        for (l = 0; l < 3; l++) {
          YZSpots[l][0] = YZSpotsT[l][0] - YZSpotsTemp[0];
          YZSpots[l][1] = YZSpotsT[l][1] - YZSpotsTemp[1];
        }
      }
    }
    if (OutofBounds == 1) {
      continue;
    }
    if (gs * 2 > px) {
      CalcPixels2(YZSpots, InPixels, &NrInPixels);
    } else {
      InPixels[0][0] =
          (int)round((YZSpots[0][0] + YZSpots[1][0] + YZSpots[2][0]) / 3);
      InPixels[0][1] =
          (int)round((YZSpots[0][1] + YZSpots[1][1] + YZSpots[2][1]) / 3);
      NrInPixels = 1;
    }
    for (k = 0; k < NrInPixels; k++) {
      AllDistsFound = 1;
      for (Layer = 0; Layer < nLayers; Layer++) {
        MultY = (int)floor(((((double)(YZSpotsTemp[0] - ybc)) * px) *
                            (Lsds[Layer] / Lsd)) /
                               px +
                           ybcs[Layer]) +
                InPixels[k][0];
        MultZ = (int)floor(((((double)(YZSpotsTemp[1] - zbc)) * px) *
                            (Lsds[Layer] / Lsd)) /
                               px +
                           zbcs[Layer]) +
                InPixels[k][1];
        if (MultY >= NrPixelsY || MultY < 0 || MultZ >= NrPixelsZ ||
            MultZ < 0) {
          OutofBounds = 1;
          break;
        }
        BinNr = Layer * NrOfFiles;
        BinNr *= NrPixelsY;
        BinNr *= NrPixelsZ;
        TempCntr = OmeBin;
        TempCntr *= NrPixelsY;
        TempCntr *= NrPixelsZ;
        BinNr += TempCntr;
        TempCntr = NrPixelsZ;
        TempCntr *= MultY;
        BinNr += TempCntr;
        BinNr += MultZ;
        if (TestBit(ObsSpotsInfo, BinNr)) {
          if (AllDistsFound == 1) {
            AllDistsFound = 1;
          }
        } else {
          AllDistsFound = 0;
        }
      }
      if (OutofBounds == 1) {
        continue;
      }
      if (AllDistsFound == 1) {
        OverlapPixels += 1;
      }
      TotalPixels += 1;
    }
  }
  if (TotalPixels > 0) {
    *FracOver = (double)((double)OverlapPixels) / ((double)TotalPixels);
  }
  // FreeMemMatrixInt(InPixels, NrPixelsGrid); // Hoisted to caller
}

void SimulateDiffractionImage(
    const int NrOfFiles, const int nLayers, const int nTspots,
    double *TheorSpots, double OmegaStart, double OmegaStep, double XGrain[3],
    double YGrain[3], const double Lsds[nLayers],
    const long long int SizeObsSpots, double RotMatTilts[3][3], const double px,
    const double ybcs[nLayers], const double zbcs[nLayers], const double gs,
    double P0All[nLayers][3], const int NrPixelsGrid, uint16_t *ObsSpotsInfo,
    double OrientMatIn[3][3], int voxNr, FILE *spF, int **InPixels,
    int NrPixelsY, int NrPixelsZ) {
  int j, OmeBin, OutofBounds, k, l;
  double OmegaThis, ythis, zthis, XGT, YGT, Displ_Y, Displ_Z, ytemp, ztemp,
      xyz[3], P1[3], ABC[3], outxyz[3], YZSpots[3][2], Lsd, ybc, zbc, P0[3],
      YZSpotsTemp[2], YZSpotsT[3][2];
  int NrInPixels, OverlapPixels, Layer, omeRangNr;
  double eta, RingRadius, theta, omediff;
  long long int BinNr, TempCntr;
  int MultY, MultZ, AllDistsFound, TotalPixels;
  OverlapPixels = 0;
  TotalPixels = 0;
  int NrSpots = 0;
  for (k = 0; k < 3; k++) {
    P0[k] = P0All[0][k];
  }
  Lsd = Lsds[0];
  ybc = ybcs[0];
  zbc = zbcs[0];
  for (j = 0; j < nTspots; j++) {
    ythis = TheorSpots[j * 3 + 0];
    zthis = TheorSpots[j * 3 + 1];
    OutofBounds = 1;
    // printf("%d %lf %lf\n",j,ythis,zthis);
    if (Wedge != 0) {
      eta = CalcEta(ythis, zthis);
      RingRadius = sqrt(ythis * ythis + zthis * zthis);
      theta = rad2deg * atan(RingRadius / Lsds[0]);
      omediff = CorrectWedge(eta, theta, Wavelength, Wedge);
      OmegaThis = TheorSpots[j * 3 + 2] - omediff;
      // Check if we just went outside the omegaRange, then make OutofBounds =
      // 1;
      if (OmegaThis >= 180) {
        OmegaThis -= 360;
      } else if (OmegaThis <= -180) {
        OmegaThis += 360;
      }
      for (omeRangNr = 0; omeRangNr < nOmeRang; omeRangNr++) {
        if (OmegaThis > OmegaRang[omeRangNr][0] &&
            OmegaThis < OmegaRang[omeRangNr][1]) {
          OutofBounds = 0;
          break;
        }
      }
    } else {
      OmegaThis = TheorSpots[j * 3 + 2];
      OutofBounds = 0;
    }
    // printf("%d %lf %lf %lf\n",j,ythis,zthis,OmegaThis);
    OmeBin = (int)floor((-OmegaStart + OmegaThis) / OmegaStep);
    if (OmeBin < 0 || OmeBin >= NrOfFiles) {
      OutofBounds = 1;
    }
    double OmegaRad = deg2rad * OmegaThis;
    double sinOme = sin(OmegaRad);
    double cosOme = cos(OmegaRad);
    for (k = 0; k < 3; k++) {
      XGT = XGrain[k];
      YGT = YGrain[k];
      DisplacementSpotsPrecomp(XGT, YGT, Lsd, ythis, zthis, sinOme, cosOme,
                               &Displ_Y, &Displ_Z);
      ytemp = Displ_Y;
      ztemp = Displ_Z;
      xyz[0] = 0;
      xyz[1] = ytemp;
      xyz[2] = ztemp;
      MatrixMultF(RotMatTilts, xyz, P1);
      for (l = 0; l < 3; l++) {
        ABC[l] = P1[l] - P0[l];
      }
      outxyz[0] = 0;
      outxyz[1] = P0[1] - (ABC[1] * P0[0]) / (ABC[0]);
      outxyz[2] = P0[2] - (ABC[2] * P0[0]) / (ABC[0]);
      YZSpotsT[k][0] = (outxyz[1]) / px + ybc;
      YZSpotsT[k][1] = (outxyz[2]) / px + zbc;
      // if (YZSpotsT[k][0] > 2048 || YZSpotsT[k][0] < 0 || YZSpotsT[k][1] >
      // 2048 || YZSpotsT[k][1] < 0){ 	OutofBounds = 1; 	break;
      // }
      if (k == 2) {
        xyz[0] = 0;
        xyz[1] = ythis;
        xyz[2] = zthis;
        MatrixMultF(RotMatTilts, xyz, P1);
        for (l = 0; l < 3; l++) {
          ABC[l] = P1[l] - P0[l];
        }
        outxyz[0] = 0;
        outxyz[1] = P0[1] - (ABC[1] * P0[0]) / (ABC[0]);
        outxyz[2] = P0[2] - (ABC[2] * P0[0]) / (ABC[0]);
        YZSpotsTemp[0] = (outxyz[1]) / px + ybc;
        YZSpotsTemp[1] = (outxyz[2]) / px + zbc;
        for (l = 0; l < 3; l++) {
          YZSpots[l][0] = YZSpotsT[l][0] - YZSpotsTemp[0];
          YZSpots[l][1] = YZSpotsT[l][1] - YZSpotsTemp[1];
        }
      }
    }
    if (OutofBounds == 1) {
      continue;
    }
    if (gs * 2 > px) {
      CalcPixels2(YZSpots, InPixels, &NrInPixels);
    } else {
      InPixels[0][0] =
          (int)round((YZSpots[0][0] + YZSpots[1][0] + YZSpots[2][0]) / 3);
      InPixels[0][1] =
          (int)round((YZSpots[0][1] + YZSpots[1][1] + YZSpots[2][1]) / 3);
      NrInPixels = 1;
    }
    for (k = 0; k < NrInPixels; k++) {
      AllDistsFound = 1;
      for (Layer = 0; Layer < nLayers; Layer++) {
        MultY = (int)floor(((((double)(YZSpotsTemp[0] - ybc)) * px) *
                            (Lsds[Layer] / Lsd)) /
                               px +
                           ybcs[Layer]) +
                InPixels[k][0];
        MultZ = (int)floor(((((double)(YZSpotsTemp[1] - zbc)) * px) *
                            (Lsds[Layer] / Lsd)) /
                               px +
                           zbcs[Layer]) +
                InPixels[k][1];
        if (MultY >= NrPixelsY || MultY < 0 || MultZ >= NrPixelsZ ||
            MultZ < 0) {
          break;
        }
        // printf("%lf\n",OmegaThis);
        if (spF != NULL) {
#pragma omp critical(spF_write)
          fprintf(spF, "%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n", voxNr, Layer,
                  OmeBin, MultY, MultZ, OmegaThis, ythis, zthis);
        }
        BinNr = Layer * NrOfFiles;
        BinNr *= NrPixelsY;
        BinNr *= NrPixelsZ;
        TempCntr = OmeBin;
        TempCntr *= NrPixelsY;
        TempCntr *= NrPixelsZ;
        BinNr += TempCntr;
        TempCntr = NrPixelsZ;
        TempCntr *= MultY;
        BinNr += TempCntr;
        BinNr += MultZ;
#pragma omp atomic
        ObsSpotsInfo[BinNr] += 1;
        if (Layer == 0)
          NrSpots++;
      }
    }
  }
  // printf("%d\n",NrSpots);
}

void CalcOverlapAccOrient(
    const int NrOfFiles, const int nLayers, const double ExcludePoleAngle,
    const double Lsd[nLayers], const long long int SizeObsSpots,
    const double XGrain[3], const double YGrain[3], double RotMatTilts[3][3],
    const double OmegaStart, const double OmegaStep, const double px,
    const double ybc[nLayers], const double zbc[nLayers], const double gs,
    double hkls[5000][4], int n_hkls, double Thetas[5000],
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4], double P0[nLayers][3],
    const int NrPixelsGrid, int *ObsSpotsInfo, double OrientMatIn[3][3],
    double *FracOverlap, double *TheorSpots, int **InPixels, double *Gs,
    int NrPixelsY, int NrPixelsZ) {
  int nTspots;
  // Compute spot positions.
  CalcDiffractionSpots(Lsd[0], ExcludePoleAngle, OmegaRanges, NoOfOmegaRanges,
                       hkls, n_hkls, Thetas, BoxSizes, &nTspots, OrientMatIn,
                       TheorSpots, Gs);
  double FracOver;
  CalcFracOverlap(NrOfFiles, nLayers, nTspots, TheorSpots, OmegaStart,
                  OmegaStep, XGrain, YGrain, Lsd, SizeObsSpots, RotMatTilts, px,
                  ybc, zbc, gs, P0, NrPixelsGrid, ObsSpotsInfo, OrientMatIn,
                  &FracOver, InPixels, NrPixelsY, NrPixelsZ);
  *FracOverlap = FracOver;
}

void SimulateAccOrient(
    const int NrOfFiles, const int nLayers, const double ExcludePoleAngle,
    const double Lsd[nLayers], const long long int SizeObsSpots,
    const double XGrain[3], const double YGrain[3], double RotMatTilts[3][3],
    const double OmegaStart, const double OmegaStep, const double px,
    const double ybc[nLayers], const double zbc[nLayers], const double gs,
    double hkls[5000][4], int n_hkls, double Thetas[5000],
    double OmegaRanges[MAX_N_OMEGA_RANGES][2], const int NoOfOmegaRanges,
    double BoxSizes[MAX_N_OMEGA_RANGES][4], double P0[nLayers][3],
    const int NrPixelsGrid, uint16_t *ObsSpotsInfo, double OrientMatIn[3][3],
    double *TheorSpots, int voxNr, FILE *spF, int **InPixels, double *Gs,
    int NrPixelsY, int NrPixelsZ) {
  int nTspots, i;
  // Compute spot positions.
  CalcDiffractionSpots(Lsd[0], ExcludePoleAngle, OmegaRanges, NoOfOmegaRanges,
                       hkls, n_hkls, Thetas, BoxSizes, &nTspots, OrientMatIn,
                       TheorSpots, Gs);
  // printf("#Spots: %d\n",nTspots);
  double XG[3], YG[3];
  for (i = 0; i < 3; i++) {
    XG[i] = XGrain[i];
    YG[i] = YGrain[i];
  }
  SimulateDiffractionImage(
      NrOfFiles, nLayers, nTspots, TheorSpots, OmegaStart, OmegaStep, XG, YG,
      Lsd, SizeObsSpots, RotMatTilts, px, ybc, zbc, gs, P0, NrPixelsGrid,
      ObsSpotsInfo, OrientMatIn, voxNr, spF, InPixels, NrPixelsY, NrPixelsZ);
}

void NormalizeMat(double OMIn[9], double OMOut[9]) {
  double determinant;
  int i;
  determinant = (OMIn[0] * ((OMIn[4] * OMIn[8]) - (OMIn[5] * OMIn[7]))) -
                (OMIn[1] * ((OMIn[3] * OMIn[8]) - (OMIn[5] * OMIn[6]))) +
                (OMIn[2] * ((OMIn[3] * OMIn[7]) - (OMIn[4] * OMIn[6])));
  for (i = 0; i < 9; i++) {
    OMOut[i] = OMIn[i] / determinant;
  }
}