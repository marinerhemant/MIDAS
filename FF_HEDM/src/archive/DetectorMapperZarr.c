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
#include "MapHeader.h"
#include "ZarrReader.h"
#include <blosc2.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <zip.h>

#include "DetectorGeometry.h"
#include "Panel.h"

double *distortionMapY;
double *distortionMapZ;
int distortionFile;
int numProcs;
static Panel *panels = NULL;
static int nPanels = 0;

struct data {
  int y;
  int z;
  double frac;
};

static long long int
mapperfcn(double tx, double ty, double tz, int NrPixelsY, int NrPixelsZ,
          double pxY, double pxZ, double Ycen, double Zcen, double Lsd,
          double RhoD, double p0, double p1, double p2, double p3,
          double *EtaBinsLow, double *EtaBinsHigh, double *RBinsLow,
          double *RBinsHigh, int nRBins, int nEtaBins, struct data ***pxList,
          int **nPxList, int **maxnPx, double *mask, double p4) {
  double TRs[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs);
  int i;
  long long int TotNrOfBins = 0;
  long long int sumNrBins = 0;
  long long int nrContinued = 0;
#pragma omp parallel for schedule(dynamic, 16)                                 \
    reduction(+ : TotNrOfBins, sumNrBins, nrContinued)
  for (i = 0; i < NrPixelsY; i++) {
    // Thread-private allocations
    double RetVals[2], RetVals2[2];
    double **Edges = dg_alloc_matrix(50, 2);
    double **EdgesOut = dg_alloc_matrix(50, 2);
    double boxEdge[4][2];
    double YZ[2];
    int RChosen[500], EtaChosen[500];
    int j, k, l, m;
    for (j = 0; j < NrPixelsZ; j++) {
      long long int testPos = j;
      testPos *= NrPixelsY;
      testPos += i;
      if (mask != NULL) {
        if (mask[testPos] == 1.0)
          continue;
      }
      double EtaMi = 1800;
      double EtaMa = -1800;
      double RMi = 1E8; // In pixels
      double RMa = -1000;
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
      double ypr = (double)i + distortionMapY[testPos] + pdY;
      double zpr = (double)j + distortionMapZ[testPos] + pdZ;
      double Eta, Rt;
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          double Y = ypr + dg_dy[k];
          double Z = zpr + dg_dz[l];
          dg_pixel_to_REta(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, p4,
                           pxY, dLsd, dP2, &Rt, &Eta, NULL);
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
      dg_pixel_to_REta(ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2, p3, p4,
                       pxY, dLsd, dP2, &Rt, &Eta, NULL);
      dg_REta_to_YZ(Rt, Eta, &YZ[0], &YZ[1]);
      // Now check which eta, R ranges should have this pixel
      int nrRChosen = 0;
      int nrEtaChosen = 0;
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
          double EtaMiTr = EtaMa;
          double EtaMaTr = 360 + EtaMi;
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
      double yMin = YZ[0] - 0.5;
      double yMax = YZ[0] + 0.5;
      double zMin = YZ[1] - 0.5;
      double zMax = YZ[1] + 0.5;
      sumNrBins += nrRChosen * nrEtaChosen;
      double totPxArea = 0;
      // Line Intercepts ordering: RMin: ymin, ymax, zmin, zmax. RMax: ymin,
      // ymax, zmin, zmax
      //							 EtaMin: ymin,
      // ymax, zmin, zmax. EtaMax: ymin, ymax, zmin, zmax.
      for (k = 0; k < nrRChosen; k++) {
        double RMin = RBinsLow[RChosen[k]];
        double RMax = RBinsHigh[RChosen[k]];
        for (l = 0; l < nrEtaChosen; l++) {
          double EtaMin = EtaBinsLow[EtaChosen[l]];
          double EtaMax = EtaBinsHigh[EtaChosen[l]];
          // Find YZ of the polar mask.
          dg_REta_to_YZ(RMin, EtaMin, &boxEdge[0][0], &boxEdge[0][1]);
          dg_REta_to_YZ(RMin, EtaMax, &boxEdge[1][0], &boxEdge[1][1]);
          dg_REta_to_YZ(RMax, EtaMin, &boxEdge[2][0], &boxEdge[2][1]);
          dg_REta_to_YZ(RMax, EtaMax, &boxEdge[3][0], &boxEdge[3][1]);
          int nEdges = 0;
          // Now check if any edge of the pixel is within the polar mask
          for (m = 0; m < 4; m++) {
            double RThis = sqrt(
                (YZ[0] + dg_PosMatrix[m][0]) * (YZ[0] + dg_PosMatrix[m][0]) +
                (YZ[1] + dg_PosMatrix[m][1]) * (YZ[1] + dg_PosMatrix[m][1]));
            double EtaThis = dg_calc_eta_angle(YZ[0] + dg_PosMatrix[m][0],
                                               YZ[1] + dg_PosMatrix[m][1]);
            if (EtaMin < -180 && dg_sign(EtaThis) != dg_sign(EtaMin))
              EtaThis -= 360;
            if (EtaMax > 180 && dg_sign(EtaThis) != dg_sign(EtaMax))
              EtaThis += 360;
            if (RThis >= RMin && RThis <= RMax && EtaThis >= EtaMin &&
                EtaThis <= EtaMax) {
              Edges[nEdges][0] = YZ[0] + dg_PosMatrix[m][0];
              Edges[nEdges][1] = YZ[1] + dg_PosMatrix[m][1];
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
            double yTemp, zTemp, yTempMin, yTempMax, zTempMin, zTempMax;
            if (RMin >= yMin) {
              zTemp = dg_sign(YZ[1]) * sqrt(RMin * RMin - yMin * yMin);
              if (dg_between(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMin;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMin >= yMax) {
              zTemp = dg_sign(YZ[1]) * sqrt(RMin * RMin - yMax * yMax);
              if (dg_between(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMax;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMax >= yMin) {
              zTemp = dg_sign(YZ[1]) * sqrt(RMax * RMax - yMin * yMin);
              if (dg_between(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMin;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            if (RMax >= yMax) {
              zTemp = dg_sign(YZ[1]) * sqrt(RMax * RMax - yMax * yMax);
              if (dg_between(zTemp, zMin, zMax) == 1) {
                Edges[nEdges][0] = yMax;
                Edges[nEdges][1] = zTemp;
                nEdges++;
              }
            }
            // RMin,Max and zMin,Max
            if (RMin >= zMin) {
              yTemp = dg_sign(YZ[0]) * sqrt(RMin * RMin - zMin * zMin);
              if (dg_between(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMin;
                nEdges++;
              }
            }
            if (RMin >= zMax) {
              yTemp = dg_sign(YZ[0]) * sqrt(RMin * RMin - zMax * zMax);
              if (dg_between(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMax;
                nEdges++;
              }
            }
            if (RMax >= zMin) {
              yTemp = dg_sign(YZ[0]) * sqrt(RMax * RMax - zMin * zMin);
              if (dg_between(yTemp, yMin, yMax) == 1) {
                Edges[nEdges][0] = yTemp;
                Edges[nEdges][1] = zMin;
                nEdges++;
              }
            }
            if (RMax >= zMax) {
              yTemp = dg_sign(YZ[0]) * sqrt(RMax * RMax - zMax * zMax);
              if (dg_between(yTemp, yMin, yMax) == 1) {
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
              zTempMin = -yMin / tan(EtaMin * DG_DEG2RAD);
              zTempMax = -yMax / tan(EtaMin * DG_DEG2RAD);
            }
            if (dg_between(zTempMin, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMin;
              Edges[nEdges][1] = zTempMin;
              nEdges++;
            }
            if (dg_between(zTempMax, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMax;
              Edges[nEdges][1] = zTempMax;
              nEdges++;
            }
            if (fabs(EtaMax) < 1E-5 || fabs(fabs(EtaMax) - 180) < 1E-5) {
              zTempMin = 0;
              zTempMax = 0;
            } else {
              zTempMin = -yMin / tan(EtaMax * DG_DEG2RAD);
              zTempMax = -yMax / tan(EtaMax * DG_DEG2RAD);
            }
            if (dg_between(zTempMin, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMin;
              Edges[nEdges][1] = zTempMin;
              nEdges++;
            }
            if (dg_between(zTempMax, zMin, zMax) == 1) {
              Edges[nEdges][0] = yMax;
              Edges[nEdges][1] = zTempMax;
              nEdges++;
            }
            // EtaMin,Max and zMin,Max
            if (fabs(fabs(EtaMin) - 90) < 1E-5) {
              yTempMin = 0;
              yTempMax = 0;
            } else {
              yTempMin = -zMin * tan(EtaMin * DG_DEG2RAD);
              yTempMax = -zMax * tan(EtaMin * DG_DEG2RAD);
            }
            if (dg_between(yTempMin, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMin;
              Edges[nEdges][1] = zMin;
              nEdges++;
            }
            if (dg_between(yTempMax, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMax;
              Edges[nEdges][1] = zMax;
              nEdges++;
            }
            if (fabs(fabs(EtaMax) - 90) < 1E-5) {
              yTempMin = 0;
              yTempMax = 0;
            } else {
              yTempMin = -zMin * tan(EtaMax * DG_DEG2RAD);
              yTempMax = -zMax * tan(EtaMax * DG_DEG2RAD);
            }
            if (dg_between(yTempMin, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMin;
              Edges[nEdges][1] = zMin;
              nEdges++;
            }
            if (dg_between(yTempMax, yMin, yMax) == 1) {
              Edges[nEdges][0] = yTempMax;
              Edges[nEdges][1] = zMax;
              nEdges++;
            }
          }
          if (nEdges < 3) {
            nrContinued++;
            continue;
          }
          nEdges = dg_find_unique_vertices(Edges, EdgesOut, nEdges, RMin, RMax,
                                           EtaMin, EtaMax);
          if (nEdges < 3) {
            nrContinued++;
            continue;
          }
          // Now we have all the edges, let's calculate the area.
          double Area = dg_polygon_area(EdgesOut, nEdges, RMin, RMax);
          if (Area < 1E-5) {
            nrContinued++;
            continue;
          }
          // Populate the arrays (thread-safe)
#pragma omp critical
          {
            int maxnVal = maxnPx[RChosen[k]][EtaChosen[l]];
            int nVal = nPxList[RChosen[k]][EtaChosen[l]];
            if (nVal >= maxnVal) {
              maxnVal += 2;
              struct data *oldarr = pxList[RChosen[k]][EtaChosen[l]];
              struct data *newarr = realloc(oldarr, maxnVal * sizeof(*newarr));
              if (newarr == NULL) {
                fprintf(stderr, "realloc failed in mapperfcn\n");
              } else {
                pxList[RChosen[k]][EtaChosen[l]] = newarr;
                maxnPx[RChosen[k]][EtaChosen[l]] = maxnVal;
              }
            }
            pxList[RChosen[k]][EtaChosen[l]][nVal].y = i;
            pxList[RChosen[k]][EtaChosen[l]][nVal].z = j;
            pxList[RChosen[k]][EtaChosen[l]][nVal].frac = Area;
            (nPxList[RChosen[k]][EtaChosen[l]])++;
          }
          totPxArea += Area;
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
  numProcs = omp_get_max_threads();
  // Check for -nCPUs flag
  for (int ai = 1; ai < argc; ai++) {
    if (strcmp(argv[ai], "-nCPUs") == 0 && ai + 1 < argc) {
      numProcs = atoi(argv[ai + 1]);
      ai++;
    }
  }
  if (argc < 2) {
    printf(
        "******************Supply a Zarr file as argument.******************\n"
        "Zarr file must have all the parameters needed: tx, ty, tz, px, BC, "
        "Lsd, RhoD,"
        "\n\t\t   p0, p1, p2, EtaBinSize, EtaMin,\n\t\t   EtaMax, RBinSize, "
        "RMin, RMax,\n\t\t   NrPixels\n"
        "Optional: -nCPUs N (default: all available)\n");
    return (1);
  }
  omp_set_num_threads(numProcs);
  printf("Using %d threads.\n", numProcs);
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
  memset(MaskFN, 0, sizeof(MaskFN));
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
  dg_build_bin_edges(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize,
                     EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);
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

  // Compute parameter hash header
  struct MapHeader map_hdr;
  map_header_compute(&map_hdr, Lsd, yCen, zCen, pxY, pxZ, tx, ty, tz, p0, p1,
                     p2, p3, p4, RhoD, RBinSize, EtaBinSize, RMin, RMax, EtaMin,
                     EtaMax, NrPixelsY, NrPixelsZ, 0, NULL, 0, 0.0);
  map_header_print("Map.bin", &map_hdr);

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
  map_header_write(mapfile, &map_hdr);
  map_header_write(nmapfile, &map_hdr);
  fwrite(pxListStore, TotNrOfBins * sizeof(*pxListStore), 1, mapfile);
  fwrite(nPxListStore, LengthNPxList * 2 * sizeof(*nPxListStore), 1, nmapfile);

  end0 = clock();
  diftotal = ((double)(end0 - start0)) / CLOCKS_PER_SEC;
  printf("Total time elapsed:\t%f s.\n", diftotal);
}
