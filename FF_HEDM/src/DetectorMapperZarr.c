//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  Created by Hemant Sharma on 2017/07/10.
//
//
// TODO: Add option to give QbinSize instead of RbinSize, look at 0,90,180,270

#include "FileReader.h"
#include "ZarrReader.h"
#include <blosc2.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <zip.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1E-6
double *distortionMapY;
double *distortionMapZ;
int distortionFile;
#include "Panel.h"
int numProcs;
static Panel *panels = NULL;
static int nPanels = 0;

static inline int BETWEEN(double val, double min, double max) {
  return ((val - EPS <= max && val + EPS >= min) ? 1 : 0);
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

static inline double signVal(double x) {
  if (x == 0)
    return 1.0;
  else
    return x / fabs(x);
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void REta4MYZ(double Y, double Z, double Ycen, double Zcen,
                            double TRs[3][3], double Lsd, double RhoD,
                            double p0, double p1, double p2, double p3,
                            double n0, double n1, double n2, double px,
                            double *RetVals, double p4, double dLsd,
                            double dP2) {
  double Yc, Zc, ABC[3], ABCPr[3], XYZ[3], Rad, Eta, RNorm, DistortFunc, EtaT,
      Rt;
  double panelLsd = Lsd + dLsd;
  double panelP2 = p2 + dP2;
  Yc = (-Y + Ycen) * px;
  Zc = (Z - Zcen) * px;
  ABC[0] = 0;
  ABC[1] = Yc;
  ABC[2] = Zc;
  MatrixMult(TRs, ABC, ABCPr);
  XYZ[0] = panelLsd + ABCPr[0];
  XYZ[1] = ABCPr[1];
  XYZ[2] = ABCPr[2];
  Rad = (panelLsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
  Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
  RNorm = Rad / RhoD;
  EtaT = 90 - Eta;
  DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT + p3)))) +
                (panelP2 * (pow(RNorm, n2)));
  DistortFunc += p4 * pow(RNorm, 6.0);
  DistortFunc += 1;
  Rt = Rad * DistortFunc / px; // in pixels
  Rt = Rt * (Lsd / panelLsd);  // re-project to global Lsd plane
  RetVals[0] = Eta;
  RetVals[1] = Rt;
}

static inline void YZ4mREta(double R, double Eta, double *YZ) {
  YZ[0] = -R * sin(Eta * deg2rad);
  YZ[1] = R * cos(Eta * deg2rad);
}

const double dy[2] = {-0.5, +0.5};
const double dz[2] = {-0.5, +0.5};

static inline void REtaMapper(double Rmin, double EtaMin, int nEtaBins,
                              int nRBins, double EtaBinSize, double RBinSize,
                              double *EtaBinsLow, double *EtaBinsHigh,
                              double *RBinsLow, double *RBinsHigh) {
  int i, j, k, l;
  for (i = 0; i < nEtaBins; i++) {
    EtaBinsLow[i] = EtaBinSize * i + EtaMin;
    EtaBinsHigh[i] = EtaBinSize * (i + 1) + EtaMin;
  }
  for (i = 0; i < nRBins; i++) {
    RBinsLow[i] = RBinSize * i + Rmin;
    RBinsHigh[i] = RBinSize * (i + 1) + Rmin;
  }
}

static inline int nOutside(double pos, int direction, double tempIntercepts[],
                           int nIntercepts) {
  int i;
  int nOD = 0;
  for (i = 0; i < nIntercepts; i++) {
    if (direction * pos < direction * tempIntercepts[i]) {
      nOD++;
    }
  }
  return nOD;
}

struct Point {
  double x;
  double y;
};

struct Point center;

static int cmpfunc(const void *ia, const void *ib) {
  struct Point *a = (struct Point *)ia;
  struct Point *b = (struct Point *)ib;
  if (a->x - center.x >= 0 && b->x - center.x < 0)
    return 1;
  if (a->x - center.x < 0 && b->x - center.x >= 0)
    return -1;
  if (a->x - center.x == 0 && b->x - center.x == 0) {
    if (a->y - center.y >= 0 || b->y - center.y >= 0) {
      return a->y > b->y ? 1 : -1;
    }
    return b->y > a->y ? 1 : -1;
  }
  double det = (a->x - center.x) * (b->y - center.y) -
               (b->x - center.x) * (a->y - center.y);
  if (det < 0)
    return 1;
  if (det > 0)
    return -1;
  int d1 = (a->x - center.x) * (a->x - center.x) +
           (a->y - center.y) * (a->y - center.y);
  int d2 = (b->x - center.x) * (b->x - center.x) +
           (b->y - center.y) * (b->y - center.y);
  return d1 > d2 ? 1 : -1;
}

double PosMatrix[4][2] = {{-0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}};

static inline double CalcAreaPolygon(double **Edges, int nEdges) {
  int i;
  struct Point *MyData;
  MyData = malloc(nEdges * sizeof(*MyData));
  center.x = 0;
  center.y = 0;
  for (i = 0; i < nEdges; i++) {
    center.x += Edges[i][0];
    center.y += Edges[i][1];
    MyData[i].x = Edges[i][0];
    MyData[i].y = Edges[i][1];
  }
  center.x /= nEdges;
  center.y /= nEdges;

  qsort(MyData, nEdges, sizeof(struct Point), cmpfunc);
  double **SortedEdges;
  SortedEdges = allocMatrix(nEdges + 1, 2);
  for (i = 0; i < nEdges; i++) {
    SortedEdges[i][0] = MyData[i].x;
    SortedEdges[i][1] = MyData[i].y;
  }
  SortedEdges[nEdges][0] = MyData[0].x;
  SortedEdges[nEdges][1] = MyData[0].y;

  double Area = 0;
  for (i = 0; i < nEdges; i++) {
    Area += 0.5 * ((SortedEdges[i][0] * SortedEdges[i + 1][1]) -
                   (SortedEdges[i + 1][0] * SortedEdges[i][1]));
  }
  free(MyData);
  FreeMemMatrix(SortedEdges, nEdges + 1);
  return Area;
}

static inline int FindUniques(double **EdgesIn, double **EdgesOut, int nEdgesIn,
                              double RMin, double RMax, double EtaMin,
                              double EtaMax) {
  int i, j, nEdgesOut = 0, duplicate;
  double Len, RT, ET;
  for (i = 0; i < nEdgesIn; i++) {
    duplicate = 0;
    for (j = i + 1; j < nEdgesIn; j++) {
      Len = sqrt(
          (EdgesIn[i][0] - EdgesIn[j][0]) * (EdgesIn[i][0] - EdgesIn[j][0]) +
          (EdgesIn[i][1] - EdgesIn[j][1]) * (EdgesIn[i][1] - EdgesIn[j][1]));
      if (Len == 0) {
        duplicate = 1;
      }
    }
    RT = sqrt(EdgesIn[i][0] * EdgesIn[i][0] + EdgesIn[i][1] * EdgesIn[i][1]);
    ET = CalcEtaAngle(EdgesIn[i][0], EdgesIn[i][1]);
    if (!BETWEEN(ET, EtaMin, EtaMax)) {
      if (BETWEEN(ET + 360, EtaMin, EtaMax)) {
        ET += 360;
      } else if (BETWEEN(ET - 360, EtaMin, EtaMax)) {
        ET -= 360;
      }
    }
    if (BETWEEN(RT, RMin, RMax) == 0) {
      duplicate = 1;
    }
    if (BETWEEN(ET, EtaMin, EtaMax) == 0) {
      duplicate = 1;
    }
    if (duplicate == 0) {
      EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
      EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
      nEdgesOut++;
    }
  }
  return nEdgesOut;
}

struct data {
  int y;
  int z;
  double frac;
};

static inline long long int
mapperfcn(double tx, double ty, double tz, int NrPixelsY, int NrPixelsZ,
          double pxY, double pxZ, double Ycen, double Zcen, double Lsd,
          double RhoD, double p0, double p1, double p2, double p3,
          double *EtaBinsLow, double *EtaBinsHigh, double *RBinsLow,
          double *RBinsHigh, int nRBins, int nEtaBins, struct data ***pxList,
          int **nPxList, int **maxnPx, double *mask, double p4) {
  double txr, tyr, tzr;
  txr = deg2rad * tx;
  tyr = deg2rad * ty;
  tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  double n0 = 2.0, n1 = 4.0, n2 = 2.0;
  double *RetVals, *RetVals2;
  RetVals = malloc(2 * sizeof(*RetVals));
  RetVals2 = malloc(2 * sizeof(*RetVals2));
  double Y, Z, Eta, Rt;
  int i, j, k, l, m, n;
  double EtaMi, EtaMa, RMi, RMa;
  int RChosen[500], EtaChosen[500];
  int nrRChosen, nrEtaChosen;
  double EtaMiTr, EtaMaTr;
  double YZ[2];
  double **Edges;
  Edges = allocMatrix(50, 2);
  double **EdgesOut;
  EdgesOut = allocMatrix(50, 2);
  int nEdges;
  double RMin, RMax, EtaMin, EtaMax;
  double yMin, yMax, zMin, zMax;
  double boxEdge[4][2];
  double Area;
  double RThis, EtaThis;
  double yTemp, zTemp, yTempMin, yTempMax, zTempMin, zTempMax;
  int maxnVal, nVal;
  struct data *oldarr, *newarr;
  long long int TotNrOfBins = 0;
  long long int sumNrBins = 0;
  long long int nrContinued = 0;
  long long int testPos;
  double ypr, zpr;
  double RT, ET;
  // This could be done with OMP Parallel for
  for (i = 0; i < NrPixelsY; i++) {
    for (j = 0; j < NrPixelsZ; j++) {
      testPos = j;
      testPos *= NrPixelsY;
      testPos += i;
      if (mask != NULL) {
        if (mask[testPos] == 1.0)
          continue;
      }
      EtaMi = 1800;
      EtaMa = -1800;
      RMi = 1E8; // In pixels
      RMa = -1000;
      // Calculate RMi, RMa, EtaMi, EtaMa
      testPos = j;
      testPos *= NrPixelsY;
      testPos += i;
      testPos += i;
      double pdY = 0, pdZ = 0;
      double dLsd = 0, dP2 = 0;
      int pIdx = GetPanelIndex((double)i, (double)j, nPanels, panels);
      if (pIdx >= 0) {
        ApplyPanelCorrection((double)i, (double)j, &panels[pIdx], &pdY, &pdZ);
        pdY -= (double)i; // convert back to delta
        pdZ -= (double)j;
        dLsd = panels[pIdx].dLsd;
        dP2 = panels[pIdx].dP2;
      }
      ypr = (double)i + distortionMapY[testPos] + pdY;
      zpr = (double)j + distortionMapZ[testPos] + pdZ;
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          Y = ypr + dy[k];
          Z = zpr + dz[l];
          REta4MYZ(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2,
                   pxY, RetVals, p4, dLsd, dP2);
          Eta = RetVals[0];
          Rt = RetVals[1]; // in pixels
          if (Eta < EtaMi)
            EtaMi = Eta;
          if (Eta > EtaMa)
            EtaMa = Eta;
          if (Rt < RMi)
            RMi = Rt;
          if (Rt > RMa)
            RMa = Rt;
        }
      }
      // Get corrected Y, Z for this position.
      REta4MYZ(ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, n0, n1, n2,
               pxY, RetVals, p4, dLsd, dP2);
      Eta = RetVals[0];
      Rt = RetVals[1]; // in pixels
      YZ4mREta(Rt, Eta, RetVals2);
      YZ[0] = RetVals2[0]; // Corrected Y position according to R, Eta, center
                           // at 0,0
      YZ[1] = RetVals2[1]; // Corrected Z position according to R, Eta, center
                           // at 0,0
      // Now check which eta, R ranges should have this pixel
      nrRChosen = 0;
      nrEtaChosen = 0;
      for (k = 0; k < nRBins; k++) {
        if (RBinsHigh[k] >= RMi && RBinsLow[k] <= RMa) {
          RChosen[nrRChosen] = k;
          nrRChosen++;
        }
      }
      for (k = 0; k < nEtaBins;
           k++) { // If Eta is smaller than 0, check for eta, eta+360, if eta is
                  // greater than 0, check for eta, eta-360
        // First check if the pixel is a special case
        if (EtaMa - EtaMi > 180) {
          EtaMiTr = EtaMa;
          EtaMaTr = 360 + EtaMi;
          EtaMa = EtaMaTr;
          EtaMi = EtaMiTr;
        }
        if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)) {
          EtaChosen[nrEtaChosen] = k;
          nrEtaChosen++;
          continue;
        }
        if (EtaMi < 0) {
          EtaMi += 360;
          EtaMa += 360;
        } else {
          EtaMi -= 360;
          EtaMa -= 360;
        }
        if ((EtaBinsHigh[k] >= EtaMi && EtaBinsLow[k] <= EtaMa)) {
          EtaChosen[nrEtaChosen] = k;
          nrEtaChosen++;
          continue;
        }
      }
      yMin = YZ[0] - 0.5;
      yMax = YZ[0] + 0.5;
      zMin = YZ[1] - 0.5;
      zMax = YZ[1] + 0.5;
      sumNrBins += nrRChosen * nrEtaChosen;
      double totPxArea = 0;
      // Line Intercepts ordering: RMin: ymin, ymax, zmin, zmax. RMax: ymin,
      // ymax, zmin, zmax
      //							 EtaMin: ymin,
      // ymax, zmin, zmax. EtaMax: ymin, ymax, zmin, zmax.
      for (k = 0; k < nrRChosen; k++) {
        RMin = RBinsLow[RChosen[k]];
        RMax = RBinsHigh[RChosen[k]];
        for (l = 0; l < nrEtaChosen; l++) {
          EtaMin = EtaBinsLow[EtaChosen[l]];
          EtaMax = EtaBinsHigh[EtaChosen[l]];
          // Find YZ of the polar mask.
          YZ4mREta(RMin, EtaMin, RetVals);
          boxEdge[0][0] = RetVals[0];
          boxEdge[0][1] = RetVals[1];
          YZ4mREta(RMin, EtaMax, RetVals);
          boxEdge[1][0] = RetVals[0];
          boxEdge[1][1] = RetVals[1];
          YZ4mREta(RMax, EtaMin, RetVals);
          boxEdge[2][0] = RetVals[0];
          boxEdge[2][1] = RetVals[1];
          YZ4mREta(RMax, EtaMax, RetVals);
          boxEdge[3][0] = RetVals[0];
          boxEdge[3][1] = RetVals[1];
          nEdges = 0;
          // Now check if any edge of the pixel is within the polar mask
          for (m = 0; m < 4; m++) {
            RThis = sqrt((YZ[0] + PosMatrix[m][0]) * (YZ[0] + PosMatrix[m][0]) +
                         (YZ[1] + PosMatrix[m][1]) * (YZ[1] + PosMatrix[m][1]));
            EtaThis =
                CalcEtaAngle(YZ[0] + PosMatrix[m][0], YZ[1] + PosMatrix[m][1]);
            if (EtaMin < -180 && signVal(EtaThis) != signVal(EtaMin))
              EtaThis -= 360;
            if (EtaMax > 180 && signVal(EtaThis) != signVal(EtaMax))
              EtaThis += 360;
            if (RThis >= RMin && RThis <= RMax && EtaThis >= EtaMin &&
                EtaThis <= EtaMax) {
              Edges[nEdges][0] = YZ[0] + PosMatrix[m][0];
              Edges[nEdges][1] = YZ[1] + PosMatrix[m][1];
              nEdges++;
            }
          }
          for (m = 0; m < 4; m++) { // Check if any edge of the polar mask is
                                    // within the pixel edges.
            if (boxEdge[m][0] >= yMin && boxEdge[m][0] <= yMax &&
                boxEdge[m][1] >= zMin && boxEdge[m][1] <= zMax) {
              Edges[nEdges][0] = boxEdge[m][0];
              Edges[nEdges][1] = boxEdge[m][1];
              nEdges++;
            }
          }
          if (nEdges < 4) {
            // Now go through Rmin, Rmax, EtaMin, EtaMax and calculate
            // intercepts and check if within the pixel.
            // RMin,Max and yMin,Max
            if (RMin >= yMin) {
              zTemp = signVal(YZ[1]) * sqrt(RMin * RMin - yMin * yMin);
              if (BETWEEN(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMin;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMin >= yMax) {
              zTemp = signVal(YZ[1]) * sqrt(RMin * RMin - yMax * yMax);
              if (BETWEEN(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMax;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMax >= yMin) {
              zTemp = signVal(YZ[1]) * sqrt(RMax * RMax - yMin * yMin);
              if (BETWEEN(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMin;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMax >= yMax) {
              zTemp = signVal(YZ[1]) * sqrt(RMax * RMax - yMax * yMax);
              if (BETWEEN(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMax;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            // RMin,Max and zMin,Max
            if (RMin >= zMin) {
              yTemp = signVal(YZ[0]) * sqrt(RMin * RMin - zMin * zMin);
              if (BETWEEN(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMin;
                nEdges++;
              }
            }
            if (RMin >= zMax) {
              yTemp = signVal(YZ[0]) * sqrt(RMin * RMin - zMax * zMax);
              if (BETWEEN(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMax;
                nEdges++;
              }
            }
            if (RMax >= zMin) {
              yTemp = signVal(YZ[0]) * sqrt(RMax * RMax - zMin * zMin);
              if (BETWEEN(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMin;
                nEdges++;
              }
            }
            if (RMax >= zMax) {
              yTemp = signVal(YZ[0]) * sqrt(RMax * RMax - zMax * zMax);
              if (BETWEEN(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMax;
                nEdges++;
              }
            }
            // EtaMin,Max and yMin,Max
            if (fabs(EtaMin) < 1E-5 || fabs(fabs(EtaMin) - 180) < 1E-5) {
              zTempMin = 0;
              zTempMax = 0;
            } else {
              zTempMin = -yMin / tan(EtaMin * deg2rad);
              zTempMax = -yMax / tan(EtaMin * deg2rad);
            }
            if (BETWEEN(zTempMin, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMin;
              Edges[nEdges][1] = zTempMin;
              nEdges++;
            }
            if (BETWEEN(zTempMax, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMax;
              Edges[nEdges][1] = zTempMax;
              nEdges++;
            }
            if (fabs(EtaMax) < 1E-5 || fabs(fabs(EtaMax) - 180) < 1E-5) {
              zTempMin = 0;
              zTempMax = 0;
            } else {
              zTempMin = -yMin / tan(EtaMax * deg2rad);
              zTempMax = -yMax / tan(EtaMax * deg2rad);
            }
            if (BETWEEN(zTempMin, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMin;
              Edges[nEdges][1] = zTempMin;
              nEdges++;
            }
            if (BETWEEN(zTempMax, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMax;
              Edges[nEdges][1] = zTempMax;
              nEdges++;
            }
            // EtaMin,Max and zMin,Max
            if (fabs(fabs(EtaMin) - 90) < 1E-5) {
              yTempMin = 0;
              yTempMax = 0;
            } else {
              yTempMin = -zMin * tan(EtaMin * deg2rad);
              yTempMax = -zMax * tan(EtaMin * deg2rad);
            }
            if (BETWEEN(yTempMin, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMin;
              Edges[nEdges][1] = zMin;
              nEdges++;
            }
            if (BETWEEN(yTempMax, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMax;
              Edges[nEdges][1] = zMax;
              nEdges++;
            }
            if (fabs(fabs(EtaMax) - 90) < 1E-5) {
              yTempMin = 0;
              yTempMax = 0;
            } else {
              yTempMin = -zMin * tan(EtaMax * deg2rad);
              yTempMax = -zMax * tan(EtaMax * deg2rad);
            }
            if (BETWEEN(yTempMin, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMin;
              Edges[nEdges][1] = zMin;
              nEdges++;
            }
            if (BETWEEN(yTempMax, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMax;
              Edges[nEdges][1] = zMax;
              nEdges++;
            }
          }
          if (nEdges < 3) {
            nrContinued++;
            continue;
          }
          nEdges =
              FindUniques(Edges, EdgesOut, nEdges, RMin, RMax, EtaMin, EtaMax);
          if (nEdges < 3) {
            nrContinued++;
            continue;
          }
          // Now we have all the edges, let's calculate the area.
          Area = CalcAreaPolygon(EdgesOut, nEdges);
          if (Area < 1E-5) {
            nrContinued++;
            continue;
          }
          // Populate the arrays, this needs to be done critical.
          maxnVal = maxnPx[RChosen[k]][EtaChosen[l]];
          nVal = nPxList[RChosen[k]][EtaChosen[l]];
          if (nVal >= maxnVal) {
            maxnVal += 2;
            oldarr = pxList[RChosen[k]][EtaChosen[l]];
            newarr = realloc(oldarr, maxnVal * sizeof(*newarr));
            if (newarr == NULL) {
              printf("Could not allocate array. Behavior undefined.\n");
            }
            pxList[RChosen[k]][EtaChosen[l]] = newarr;
            maxnPx[RChosen[k]][EtaChosen[l]] = maxnVal;
          }
          pxList[RChosen[k]][EtaChosen[l]][nVal].y = i;
          pxList[RChosen[k]][EtaChosen[l]][nVal].z = j;
          pxList[RChosen[k]][EtaChosen[l]][nVal].frac = Area;
          totPxArea += Area;
          (nPxList[RChosen[k]][EtaChosen[l]])++;
          TotNrOfBins++;
        }
      }
    }
  }
  return TotNrOfBins;
}

static inline int StartsWith(const char *a, const char *b) {
  if (strncmp(a, b, strlen(b)) == 0)
    return 1;
  return 0;
}

static inline void DoImageTransformations(int NrTransOpt, int TransOpt[10],
                                          double *ImageIn, double *ImageOut,
                                          int NrPixelsY, int NrPixelsZ) {
  int i, k, l;
  if (ImageIn != ImageOut) {
    memcpy(ImageOut, ImageIn, NrPixelsY * NrPixelsZ * sizeof(*ImageIn));
  }
  if (NrTransOpt == 0) {
    return;
  }

  double buffer;
  for (i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] == 1) { // Invert Y (columns)
      for (l = 0; l < NrPixelsZ; l++) {
        for (k = 0; k < NrPixelsY / 2; k++) {
          buffer = ImageOut[l * NrPixelsY + k];
          ImageOut[l * NrPixelsY + k] =
              ImageOut[l * NrPixelsY + (NrPixelsY - k - 1)];
          ImageOut[l * NrPixelsY + (NrPixelsY - k - 1)] = buffer;
        }
      }
    } else if (TransOpt[i] == 2) { // Invert Z (rows)
      for (l = 0; l < NrPixelsZ / 2; l++) {
        for (k = 0; k < NrPixelsY; k++) {
          buffer = ImageOut[l * NrPixelsY + k];
          ImageOut[l * NrPixelsY + k] =
              ImageOut[(NrPixelsZ - l - 1) * NrPixelsY + k];
          ImageOut[(NrPixelsZ - l - 1) * NrPixelsY + k] = buffer;
        }
      }
    } else if (TransOpt[i] == 3) { // Transpose
      if (NrPixelsY == NrPixelsZ) {
        for (l = 0; l < NrPixelsZ; l++) {
          for (k = l + 1; k < NrPixelsY; k++) {
            buffer = ImageOut[l * NrPixelsY + k];
            ImageOut[l * NrPixelsY + k] = ImageOut[k * NrPixelsY + l];
            ImageOut[k * NrPixelsY + l] = buffer;
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  clock_t start, end, start0, end0;
  start0 = clock();
  double diftotal;
  if (argc != 2) {
    printf(
        "******************Supply a Zarr file as argument.******************\n"
        "Zarr file must have all the parameters needed: tx, ty, tz, px, BC, "
        "Lsd, RhoD,"
        "\n\t\t   p0, p1, p2, EtaBinSize, EtaMin,\n\t\t   EtaMax, RBinSize, "
        "RMin, RMax,\n\t\t   NrPixels\n");
    return (1);
  }
  char *DataFN = argv[1];
  blosc2_init();
  // Read zarr config
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(DataFN, 0, &errorp);
  if (arch == NULL) {
    fprintf(stderr, "ERROR: Could not open zip archive '%s' (error code: %d)\n",
            DataFN, errorp);
    return 1;
  }
  struct zip_stat *finfo = NULL;
  finfo = calloc(16384, sizeof(int));
  zip_stat_init(finfo);
  int count = 0;
  double tx = 0.0, ty = 0.0, tz = 0.0, pxY = 200.0, pxZ = 200.0, yCen = 1024.0,
         zCen = 1024.0, Lsd = 1000000.0, RhoD = 200000.0, p0 = 0.0, p1 = 0.0,
         p2 = 0.0, p3 = 0.0, p4 = 0.0, EtaBinSize = 5, RBinSize = 0.25,
         RMax = 1524.0, RMin = 10.0, EtaMax = 180.0, EtaMin = -180.0;
  int NrPixelsY = 2048, NrPixelsZ = 2048;
  char aline[4096], dummy[4096], *str;
  distortionFile = 0;
  char *distortionFN = NULL;
  int NrTransOpt = 0;
  int TransOpt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  char *resultFolder = NULL;
  int locImTransOpt, nFrames;
  // Panel parameters
  int NPanelsY = 0;
  int NPanelsZ = 0;
  int PanelSizeY = 0;
  int PanelSizeZ = 0;
  int *PanelGapsY = NULL;
  int *PanelGapsZ = NULL;
  int locPanelGapsY = -1;
  int locPanelGapsZ = -1;
  char PanelShiftsFile[1024];
  PanelShiftsFile[0] = '\0';
  char MaskFN[4096];
  int useMask = 0;
  double *mask = NULL;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaskFile/0") != NULL) {
      ReadZarrChunk(arch, count, MaskFN, 4096);
      useMask = 1;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      ReadZarrString(arch, count, &resultFolder, 4096);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/DistortionFN/0") != NULL) {
      ReadZarrString(arch, count, &distortionFN, 4096);
      distortionFile = 1;
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p3/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &p3, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p4/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &p4, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/DistortionOrder/0") !=
        NULL) {
      ; // DistortionOrder ignored; p4 is always enabled
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p2/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &p2, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p1/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &p1, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/p0/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &p0, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tz/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &tz, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/ty/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &ty, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/tx/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &tx, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RhoD/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &RhoD, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/MaxRingRad/0") != NULL) {
      ReadZarrChunk(arch, count, &RhoD, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/EtaBinSize/0") != NULL) {
      ReadZarrChunk(arch, count, &EtaBinSize, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/RBinSize/0") != NULL) {
      ReadZarrChunk(arch, count, &RBinSize, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RMax/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &RMax, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RMin/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &RMin, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/EtaMax/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &EtaMax, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/EtaMin/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &EtaMin, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &Lsd, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSizeY/0") != NULL) {
      ReadZarrChunk(arch, count, &pxY, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSizeZ/0") != NULL) {
      ReadZarrChunk(arch, count, &pxZ, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSize/0") != NULL) {
      ReadZarrChunk(arch, count, &pxY, sizeof(double));
      pxZ = pxY;
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/YCen/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &yCen, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/ZCen/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &zCen, sizeof(double));
    }
    if (strstr(finfo->name, "exchange/data/.zarray") != NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
               &nFrames, &NrPixelsZ, &NrPixelsY);
      } else {
        free(s);
        return 1;
      }
      free(s);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/.zarray") !=
        NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &NrTransOpt);
      } else {
        free(s);
        return 1;
      }
      printf("nImTransOpt: %d\n", NrTransOpt);
      free(s);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/0") != NULL) {
      locImTransOpt = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/NPanelsY/0") != NULL) {
      ReadZarrChunk(arch, count, &NPanelsY, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/NPanelsZ/0") != NULL) {
      ReadZarrChunk(arch, count, &NPanelsZ, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PanelSizeY/0") != NULL) {
      ReadZarrChunk(arch, count, &PanelSizeY, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PanelSizeZ/0") != NULL) {
      ReadZarrChunk(arch, count, &PanelSizeZ, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PanelShiftsFile/0") !=
        NULL) {
      char *tmpStr = NULL;
      ReadZarrString(arch, count, &tmpStr, 1024);
      strcpy(PanelShiftsFile, tmpStr);
      free(tmpStr);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PanelGapsY/0") != NULL) {
      locPanelGapsY = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PanelGapsZ/0") != NULL) {
      locPanelGapsZ = count;
    }
    count++;
  }
  printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
         "%lf %lf %d %d\n",
         tx, ty, tz, pxY, pxZ, yCen, zCen, Lsd, RhoD, p0, p1, p2, p3,
         EtaBinSize, RBinSize, RMax, RMin, EtaMax, EtaMin, NrPixelsY,
         NrPixelsZ);
  int32_t imTransBufSize = NrTransOpt * sizeof(int);
  int *imTransData = (int *)malloc((size_t)imTransBufSize);
  ReadZarrChunk(arch, locImTransOpt, imTransData, imTransBufSize);
  int iter;
  for (iter = 0; iter < NrTransOpt; iter++)
    TransOpt[iter] = imTransData[iter];
  for (iter = 0; iter < NrTransOpt; iter++)
    printf("Transopt: %d\n", TransOpt[iter]);
  free(imTransData);

  if (locPanelGapsY >= 0 && NPanelsY > 1) {
    int32_t gapSize = (NPanelsY - 1) * sizeof(int);
    PanelGapsY = malloc(gapSize);
    ReadZarrChunk(arch, locPanelGapsY, PanelGapsY, gapSize);
  }
  if (locPanelGapsZ >= 0 && NPanelsZ > 1) {
    int32_t gapSize = (NPanelsZ - 1) * sizeof(int);
    PanelGapsZ = malloc(gapSize);
    ReadZarrChunk(arch, locPanelGapsZ, PanelGapsZ, gapSize);
  }

  // Generate Panels
  if (NPanelsY > 0 && NPanelsZ > 0) {
    if (GeneratePanels(NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ, PanelGapsY,
                       PanelGapsZ, &panels, &nPanels) != 0) {
      fprintf(stderr, "Fast generation failed.\n");
      return 1;
    }
    printf("Generated %d panels.\n", nPanels);

    if (PanelShiftsFile[0] != '\0') {
      if (LoadPanelShifts(PanelShiftsFile, nPanels, panels) == 0) {
        printf("Loaded panel shifts from %s\n", PanelShiftsFile);
      } else {
        fprintf(stderr, "Failed to load panel shifts from %s\n",
                PanelShiftsFile);
      }
    }
  }
  if (useMask == 1) {
    mask = calloc(NrPixelsY * NrPixelsZ, sizeof(double));
    if (ReadTiffFrame(MaskFN, 7, NrPixelsY * NrPixelsZ, mask, 0) == 0) {
      printf("Encoded mask from file: %s\n", MaskFN);
      DoImageTransformations(NrTransOpt, TransOpt, mask, mask, NrPixelsY,
                             NrPixelsZ);
    } else {
      printf("Failed to read mask file: %s\n", MaskFN);
      free(mask);
      mask = NULL;
    }
  }

  distortionMapY = calloc(NrPixelsY * NrPixelsZ, sizeof(double));
  distortionMapZ = calloc(NrPixelsY * NrPixelsZ, sizeof(double));
  if (distortionFile == 1) {
    FILE *distortionFileHandle = fopen(distortionFN, "rb");
    double *distortionMapTemp;
    distortionMapTemp = malloc(NrPixelsY * NrPixelsZ * sizeof(double));
    fread(distortionMapTemp, NrPixelsY * NrPixelsZ * sizeof(double), 1,
          distortionFileHandle);
    DoImageTransformations(NrTransOpt, TransOpt, distortionMapTemp,
                           distortionMapY, NrPixelsY, NrPixelsZ);
    fread(distortionMapTemp, NrPixelsY * NrPixelsZ * sizeof(double), 1,
          distortionFileHandle);
    DoImageTransformations(NrTransOpt, TransOpt, distortionMapTemp,
                           distortionMapZ, NrPixelsY, NrPixelsZ);
    printf("Distortion file %s was provided and read correctly.\n",
           distortionFN);
  }
  // Parameters needed: Rmax RMin RBinSize (px) EtaMax EtaMin EtaBinSize
  // (degrees)
  int nEtaBins, nRBins;
  nRBins = (int)ceil((RMax - RMin) / RBinSize);
  nEtaBins = (int)ceil((EtaMax - EtaMin) / EtaBinSize);
  printf("Creating a mapper for integration.\nNumber of eta bins: %d, number "
         "of R bins: %d.\n",
         nEtaBins, nRBins);
  double *EtaBinsLow, *EtaBinsHigh;
  double *RBinsLow, *RBinsHigh;
  EtaBinsLow = malloc(nEtaBins * sizeof(*EtaBinsLow));
  EtaBinsHigh = malloc(nEtaBins * sizeof(*EtaBinsHigh));
  RBinsLow = malloc(nRBins * sizeof(*RBinsLow));
  RBinsHigh = malloc(nRBins * sizeof(*RBinsHigh));
  REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow,
             EtaBinsHigh, RBinsLow, RBinsHigh);
  // Initialize arrays, need fraction array
  struct data ***pxList;
  int **nPxList;
  int **maxnPx;
  pxList = malloc(nRBins * sizeof(pxList));
  nPxList = malloc(nRBins * sizeof(nPxList));
  maxnPx = malloc(nRBins * sizeof(maxnPx));
  int i, j, k, l;
  for (i = 0; i < nRBins; i++) {
    pxList[i] = malloc(nEtaBins * sizeof(pxList[i]));
    nPxList[i] = malloc(nEtaBins * sizeof(nPxList[i]));
    maxnPx[i] = malloc(nEtaBins * sizeof(maxnPx[i]));
    for (j = 0; j < nEtaBins; j++) {
      pxList[i][j] = NULL;
      nPxList[i][j] = 0;
      maxnPx[i][j] = 0;
    }
  }
  // Parameters needed: tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen,
  // Lsd, RhoD, p0, p1, p2
  long long int TotNrOfBins =
      mapperfcn(tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen, Lsd,
                RhoD, p0, p1, p2, p3, EtaBinsLow, EtaBinsHigh, RBinsLow,
                RBinsHigh, nRBins, nEtaBins, pxList, nPxList, maxnPx, mask, p4);
  printf("Total Number of bins %lld\n", TotNrOfBins);
  fflush(stdout);
  long long int LengthNPxList = nRBins * nEtaBins;
  struct data *pxListStore;
  int *nPxListStore;
  pxListStore = malloc(TotNrOfBins * sizeof(*pxListStore));
  nPxListStore = malloc(LengthNPxList * 2 * sizeof(*nPxListStore));
  long long int Pos;
  int localNPxVal, localCounter = 0;
  for (i = 0; i < nRBins; i++) {
    for (j = 0; j < nEtaBins; j++) {
      localNPxVal = nPxList[i][j];
      Pos = i * nEtaBins;
      Pos += j;
      nPxListStore[(Pos * 2) + 0] = localNPxVal;
      nPxListStore[(Pos * 2) + 1] = localCounter;
      for (k = 0; k < localNPxVal; k++) {
        pxListStore[localCounter + k].y = pxList[i][j][k].y;
        pxListStore[localCounter + k].z = pxList[i][j][k].z;
        pxListStore[localCounter + k].frac = pxList[i][j][k].frac;
      }
      localCounter += localNPxVal;
    }
  }

  // Write out
  char mapfn[4096];
  sprintf(mapfn, "%s/Map.bin", resultFolder);
  char nmapfn[4096];
  sprintf(nmapfn, "%s/nMap.bin", resultFolder);
  FILE *mapfile = fopen(mapfn, "wb");
  FILE *nmapfile = fopen(nmapfn, "wb");
  // check if it can write file, if not, give the error:
  if (!mapfile || !nmapfile) {
    fprintf(stderr, "Error: Could not open output files for writing %s %s.\n",
            mapfn, nmapfn);
    exit(EXIT_FAILURE);
  }
  fwrite(pxListStore, TotNrOfBins * sizeof(*pxListStore), 1, mapfile);
  fwrite(nPxListStore, LengthNPxList * 2 * sizeof(*nPxListStore), 1, nmapfile);

  end0 = clock();
  diftotal = ((double)(end0 - start0)) / CLOCKS_PER_SEC;
  printf("Total time elapsed:\t%f s.\n", diftotal);
}
