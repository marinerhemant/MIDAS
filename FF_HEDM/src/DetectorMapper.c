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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <zip.h>

#include "DetectorGeometry.h"
#include "MapperCore.h"
#include "ImageUtils.h"
#include <omp.h>

double *distortionMapY;
double *distortionMapZ;
int distortionFile;
int numProcs;

#include "Panel.h"
#include "midas_version.h"
static Panel *panels = NULL;
static int nPanels = 0;

// struct data → struct MapPixelData (from MapperCore.h)
// mapperfcn → mapper_build_map (from MapperCore.c)
// inverse_transform_pixel → mapper_inverse_transform_pixel (from MapperCore.c)
// DoImageTransformations → midas_image_transform (from ImageUtils.c)

static inline int StartsWith(const char *a, const char *b) {
  if (strncmp(a, b, strlen(b)) == 0)
    return 1;
  return 0;
}

int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  double start0, end0, diftotal;
  start0 = omp_get_wtime();
  numProcs = omp_get_max_threads();

  // ── CLI parsing ──────────────────────────────────────────────────
  char *ParamFN = NULL; // text parameter file (positional)
  char *ZarrFN = NULL;  // Zarr archive (-zarrFN)
  char *resultFolder = NULL;
  for (int ai = 1; ai < argc; ai++) {
    if (strcmp(argv[ai], "-nCPUs") == 0 && ai + 1 < argc) {
      numProcs = atoi(argv[++ai]);
    } else if (strcmp(argv[ai], "-zarrFN") == 0 && ai + 1 < argc) {
      ZarrFN = argv[++ai];
    } else if (strcmp(argv[ai], "-resultFolder") == 0 && ai + 1 < argc) {
      resultFolder = argv[++ai];
    } else if (argv[ai][0] != '-' && ParamFN == NULL) {
      ParamFN = argv[ai];
    }
  }
  if (ParamFN == NULL && ZarrFN == NULL) {
    printf("Usage: DetectorMapper <parameters.txt> [-nCPUs N]\n"
           "       DetectorMapper -zarrFN <data.zarr> [-resultFolder dir] "
           "[-nCPUs N]\n");
    return 1;
  }
  omp_set_num_threads(numProcs);
  printf("Running with %d OpenMP threads.\n", numProcs);

  // ── Shared variables ─────────────────────────────────────────────
  double tx = 0, ty = 0, tz = 0, pxY = 200, pxZ = 200, yCen = 1024, zCen = 1024,
         Lsd = 1e6, RhoD = 2e5, p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0,
         p7 = 0, p8 = 0, p9 = 0, p10 = 0,
         EtaBinSize = 5, RBinSize = 0.25, RMax = 1524, RMin = 10, EtaMax = 180,
         EtaMin = -180;
  int NrPixelsY = 2048, NrPixelsZ = 2048;
  distortionFile = 0;
  char distortionFN[4096];
  distortionFN[0] = '\0';
  int NrTransOpt = 0;
  int TransOpt[10] = {0};
  int NPanelsY = 0, NPanelsZ = 0, PanelSizeY = 0, PanelSizeZ = 0;
  int *PanelGapsY = NULL, *PanelGapsZ = NULL;
  char PanelShiftsFile[1024];
  PanelShiftsFile[0] = '\0';
  char MaskFN[4096];
  MaskFN[0] = '\0';
  int useMask = 0;
  double *mask = NULL;
  // Q-spacing mode parameters
  double QBinSize = 0, QMin = 0, QMax = 0, Wavelength = 0;
  int SubPixelLevel = 1;
  double SubPixelCardinalWidth = 10.0;
  // Physical corrections
  double Parallax = 0.0;
  int SolidAngleCorrection = 0;
  int PolarizationCorrection = 0;
  double PolarizationFraction = 0.99;

  // ── Mode A: Read parameters from text file ─────────────────────
  if (ParamFN != NULL) {
    printf("readParamFile: %s\n", ParamFN);
    char aline[4096], dummy[4096], *str;
    FILE *paramFile = fopen(ParamFN, "r");
    if (!paramFile) {
      fprintf(stderr, "Error: cannot open %s\n", ParamFN);
      return 1;
    }
    while (fgets(aline, 4096, paramFile) != NULL) {
      str = "tx ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &tx);
      }
      str = "ty ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &ty);
      }
      str = "tz ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &tz);
      }
      str = "pxY ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &pxY);
      }
      str = "pxZ ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &pxZ);
      }
      str = "NPanelsY ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &NPanelsY);
        continue;
      }
      str = "NPanelsZ ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &NPanelsZ);
        continue;
      }
      str = "PanelSizeY ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &PanelSizeY);
        continue;
      }
      str = "PanelSizeZ ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &PanelSizeZ);
        continue;
      }
      str = "PanelShiftsFile ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %s", dummy, PanelShiftsFile);
        continue;
      }
      str = "PanelGapsY ";
      if (StartsWith(aline, str) == 1) {
        char *ptr = aline + strlen(str);
        if (NPanelsY > 1) {
          PanelGapsY = malloc((NPanelsY - 1) * sizeof(int));
          for (int k = 0; k < NPanelsY - 1; k++)
            PanelGapsY[k] = strtol(ptr, &ptr, 10);
        }
        continue;
      }
      str = "PanelGapsZ ";
      if (StartsWith(aline, str) == 1) {
        char *ptr = aline + strlen(str);
        if (NPanelsZ > 1) {
          PanelGapsZ = malloc((NPanelsZ - 1) * sizeof(int));
          for (int k = 0; k < NPanelsZ - 1; k++)
            PanelGapsZ[k] = strtol(ptr, &ptr, 10);
        }
        continue;
      }
      str = "px ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &pxY);
        pxZ = pxY;
      }
      str = "BC ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf %lf", dummy, &yCen, &zCen);
      }
      str = "Lsd ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &Lsd);
      }
      str = "RhoD ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &RhoD);
      }
      str = "MaxRingRad ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &RhoD);
      }
      str = "p0 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p0);
      }
      str = "p1 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p1);
      }
      str = "p2 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p2);
      }
      str = "p3 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p3);
      }
      str = "p4 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p4);
      }
      str = "p5 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p5);
      }
      str = "p6 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p6);
      }
      str = "p7 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p7);
      }
      str = "p8 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p8);
      }
      str = "p9 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p9);
      }
      str = "p10 ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &p10);
      }
      str = "DistortionOrder ";
      if (StartsWith(aline, str) == 1) {
        continue;
      }
      str = "EtaBinSize ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &EtaBinSize);
      }
      str = "RBinSize ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &RBinSize);
      }
      str = "RMax ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &RMax);
      }
      str = "RMin ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &RMin);
      }
      str = "EtaMax ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &EtaMax);
      }
      str = "EtaMin ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &EtaMin);
      }
      str = "QBinSize ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &QBinSize);
      }
      str = "SubPixelLevel ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &SubPixelLevel);
      }
      str = "SubPixelCardinalWidth ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &SubPixelCardinalWidth);
      }
      str = "QMin ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &QMin);
      }
      str = "QMax ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &QMax);
      }
      str = "Wavelength ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &Wavelength);
      }
      str = "NrPixelsY ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &NrPixelsY);
      }
      str = "NrPixelsZ ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      }
      str = "NrPixels ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &NrPixelsY);
        sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      }
      str = "DistortionFile ";
      if (StartsWith(aline, str) == 1) {
        distortionFile = 1;
        sscanf(aline, "%s %s", dummy, distortionFN);
      }
      str = "ImTransOpt ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &TransOpt[NrTransOpt]);
        NrTransOpt++;
        continue;
      }
      str = "MaskFile ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %s", dummy, MaskFN);
        useMask = 1;
      }
      str = "Parallax ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &Parallax);
      }
      str = "SolidAngleCorrection ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &SolidAngleCorrection);
      }
      str = "PolarizationCorrection ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %d", dummy, &PolarizationCorrection);
      }
      str = "PolarizationFraction ";
      if (StartsWith(aline, str) == 1) {
        sscanf(aline, "%s %lf", dummy, &PolarizationFraction);
      }
    }
    fclose(paramFile);
  }

  // ── Mode B: Read parameters from Zarr archive ──────────────────
  if (ZarrFN != NULL) {
    blosc2_init();
    int errorp = 0;
    zip_t *arch = zip_open(ZarrFN, 0, &errorp);
    if (arch == NULL) {
      fprintf(stderr,
              "ERROR: Could not open zip archive '%s' (error code: %d)\n",
              ZarrFN, errorp);
      return 1;
    }
    struct zip_stat *finfo = calloc(16384, sizeof(int));
    zip_stat_init(finfo);
    int count = 0;
    int locImTransOpt = -1, nFrames = 0;
    int locPanelGapsY = -1, locPanelGapsZ = -1;
    while (zip_stat_index(arch, count, 0, finfo) == 0) {
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/MaskFile/0") != NULL) {
        ReadZarrChunk(arch, count, MaskFN, 4096);
        useMask = 1;
      }
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/ResultFolder/0") !=
          NULL) {
        ReadZarrString(arch, count, &resultFolder, 4096);
      }
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/DistortionFN/0") !=
          NULL) {
        char *distFN = NULL;
        ReadZarrString(arch, count, &distFN, 4096);
        strncpy(distortionFN, distFN, sizeof(distortionFN) - 1);
        free(distFN);
        distortionFile = 1;
      }
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p3/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p3, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p4/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p4, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p5/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p5, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p6/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p6, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p7/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p7, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p8/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p8, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p9/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p9, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p10/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p10, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/DistortionOrder/0") !=
          NULL)
        ; // ignored
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p2/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p2, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p1/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p1, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/p0/0") !=
          NULL)
        ReadZarrChunk(arch, count, &p0, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/tz/0") !=
          NULL)
        ReadZarrChunk(arch, count, &tz, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/ty/0") !=
          NULL)
        ReadZarrChunk(arch, count, &ty, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/tx/0") !=
          NULL)
        ReadZarrChunk(arch, count, &tx, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/RhoD/0") !=
          NULL)
        ReadZarrChunk(arch, count, &RhoD, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/MaxRingRad/0") != NULL)
        ReadZarrChunk(arch, count, &RhoD, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/EtaBinSize/0") != NULL)
        ReadZarrChunk(arch, count, &EtaBinSize, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/RBinSize/0") != NULL)
        ReadZarrChunk(arch, count, &RBinSize, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/RMax/0") !=
          NULL)
        ReadZarrChunk(arch, count, &RMax, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/RMin/0") !=
          NULL)
        ReadZarrChunk(arch, count, &RMin, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/EtaMax/0") != NULL)
        ReadZarrChunk(arch, count, &EtaMax, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/EtaMin/0") != NULL)
        ReadZarrChunk(arch, count, &EtaMin, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/QBinSize/0") != NULL)
        ReadZarrChunk(arch, count, &QBinSize, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/QMin/0") !=
          NULL)
        ReadZarrChunk(arch, count, &QMin, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/QMax/0") !=
          NULL)
        ReadZarrChunk(arch, count, &QMax, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/Wavelength/0") != NULL)
        ReadZarrChunk(arch, count, &Wavelength, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
          NULL)
        ReadZarrChunk(arch, count, &Lsd, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PixelSizeY/0") != NULL)
        ReadZarrChunk(arch, count, &pxY, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PixelSizeZ/0") != NULL)
        ReadZarrChunk(arch, count, &pxZ, sizeof(double));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PixelSize/0") != NULL) {
        ReadZarrChunk(arch, count, &pxY, sizeof(double));
        pxZ = pxY;
      }
      if (strstr(finfo->name, "analysis/process/analysis_parameters/YCen/0") !=
          NULL)
        ReadZarrChunk(arch, count, &yCen, sizeof(double));
      if (strstr(finfo->name, "analysis/process/analysis_parameters/ZCen/0") !=
          NULL)
        ReadZarrChunk(arch, count, &zCen, sizeof(double));
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
        }
        free(s);
      }
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/ImTransOpt/0") != NULL)
        locImTransOpt = count;
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/NPanelsY/0") != NULL)
        ReadZarrChunk(arch, count, &NPanelsY, sizeof(int));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/NPanelsZ/0") != NULL)
        ReadZarrChunk(arch, count, &NPanelsZ, sizeof(int));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PanelSizeY/0") != NULL)
        ReadZarrChunk(arch, count, &PanelSizeY, sizeof(int));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PanelSizeZ/0") != NULL)
        ReadZarrChunk(arch, count, &PanelSizeZ, sizeof(int));
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PanelShiftsFile/0") !=
          NULL) {
        char *tmpStr = NULL;
        ReadZarrString(arch, count, &tmpStr, 1024);
        strncpy(PanelShiftsFile, tmpStr, sizeof(PanelShiftsFile) - 1);
        free(tmpStr);
      }
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PanelGapsY/0") != NULL)
        locPanelGapsY = count;
      if (strstr(finfo->name,
                 "analysis/process/analysis_parameters/PanelGapsZ/0") != NULL)
        locPanelGapsZ = count;
      count++;
    }
    printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
           "%lf %lf %lf %d %d\n",
           tx, ty, tz, pxY, pxZ, yCen, zCen, Lsd, RhoD, p0, p1, p2, p3,
           EtaBinSize, RBinSize, RMax, RMin, EtaMax, EtaMin, NrPixelsY,
           NrPixelsZ);
    if (NrTransOpt > 0 && locImTransOpt >= 0) {
      int32_t imTransBufSize = NrTransOpt * sizeof(int);
      int *imTransData = malloc((size_t)imTransBufSize);
      ReadZarrChunk(arch, locImTransOpt, imTransData, imTransBufSize);
      for (int iter = 0; iter < NrTransOpt; iter++)
        TransOpt[iter] = imTransData[iter];
      free(imTransData);
    }
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
    free(finfo);
  }

  // ── Common path: Panels, mask, distortion, mapping, output ─────
  printf("  NrPixels: %d x %d, Lsd: %.1f, px: %.4f\n", NrPixelsY, NrPixelsZ,
         Lsd, pxY);
  printf("  R: [%.1f, %.1f] bin=%.4f, Eta: [%.1f, %.1f] bin=%.2f\n", RMin, RMax,
         RBinSize, EtaMin, EtaMax, EtaBinSize);

  // Generate Panels
  if (NPanelsY > 0 && NPanelsZ > 0) {
    if (GeneratePanels(NPanelsY, NPanelsZ, PanelSizeY, PanelSizeZ, PanelGapsY,
                       PanelGapsZ, &panels, &nPanels) != 0) {
      fprintf(stderr, "Fast generation failed.\n");
      return 1;
    }
    printf("Generated %d panels.\n", nPanels);
    if (PanelShiftsFile[0] != '\0') {
      if (LoadPanelShifts(PanelShiftsFile, nPanels, panels) == 0)
        printf("Loaded panel shifts from %s\n", PanelShiftsFile);
      else
        fprintf(stderr, "Failed to load panel shifts from %s\n",
                PanelShiftsFile);
    }
  }

  if (useMask == 1) {
    mask = calloc(NrPixelsY * NrPixelsZ, sizeof(double));
    if (ReadTiffFrame(MaskFN, 7, NrPixelsY * NrPixelsZ, mask, 0) == 0) {
      printf("Encoded mask from file: %s\n", MaskFN);
      midas_image_transform(NrTransOpt, TransOpt, mask, mask, NrPixelsY,
                            NrPixelsZ);
      int maskedPixels = 0;
      for (int mi = 0; mi < NrPixelsY * NrPixelsZ; mi++)
        if (mask[mi] == 1.0)
          maskedPixels++;
      printf("Number of masked pixels: %d out of %d\n", maskedPixels,
             NrPixelsY * NrPixelsZ);
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
    double *distortionMapTemp = malloc(NrPixelsY * NrPixelsZ * sizeof(double));
    fread(distortionMapTemp, NrPixelsY * NrPixelsZ * sizeof(double), 1,
          distortionFileHandle);
    midas_image_transform(NrTransOpt, TransOpt, distortionMapTemp,
                          distortionMapY, NrPixelsY, NrPixelsZ);
    fread(distortionMapTemp, NrPixelsY * NrPixelsZ * sizeof(double), 1,
          distortionFileHandle);
    midas_image_transform(NrTransOpt, TransOpt, distortionMapTemp,
                          distortionMapZ, NrPixelsY, NrPixelsZ);
    printf("Distortion file %s was provided and read correctly.\n",
           distortionFN);
    free(distortionMapTemp);
    fclose(distortionFileHandle);
  }

  // Bin edges
  int nEtaBins = (int)ceil((EtaMax - EtaMin) / EtaBinSize);
  int qMode = (QBinSize > 0 && Wavelength > 0);
  int nRBins;
  if (qMode) {
    // Q-mode: compute RMin/RMax from QMin/QMax, then nRBins from QBinSize
    if (QMin > 0)
      RMin = (Lsd / pxY) * tan(2.0 * asin(QMin * Wavelength / (4.0 * M_PI)));
    if (QMax > 0)
      RMax = (Lsd / pxY) * tan(2.0 * asin(QMax * Wavelength / (4.0 * M_PI)));
    nRBins = (int)ceil((QMax - QMin) / QBinSize);
    printf("Q-mode: %d Q-bins [%.4f, %.4f] inv-Angstrom, step %.4f\n", nRBins,
           QMin, QMax, QBinSize);
    printf("  Wavelength: %.6f Angstrom\n", Wavelength);
    printf("  -> R range [%.2f, %.2f] pixels\n", RMin, RMax);
  } else {
    nRBins = (int)ceil((RMax - RMin) / RBinSize);
  }
  printf("Creating a mapper for integration.\nNumber of eta bins: %d, number "
         "of R bins: %d.\n",
         nEtaBins, nRBins);
  double *EtaBinsLow = malloc(nEtaBins * sizeof(*EtaBinsLow));
  double *EtaBinsHigh = malloc(nEtaBins * sizeof(*EtaBinsHigh));
  double *RBinsLow = malloc(nRBins * sizeof(*RBinsLow));
  double *RBinsHigh = malloc(nRBins * sizeof(*RBinsHigh));
  if (qMode) {
    // Equal Q bins -> non-uniform R bins
    for (int qi = 0; qi < nRBins; qi++) {
      double qLow = QMin + QBinSize * qi;
      double qHigh = QMin + QBinSize * (qi + 1);
      RBinsLow[qi] =
          (Lsd / pxY) * tan(2.0 * asin(qLow * Wavelength / (4.0 * M_PI)));
      RBinsHigh[qi] =
          (Lsd / pxY) * tan(2.0 * asin(qHigh * Wavelength / (4.0 * M_PI)));
    }
    // Build eta bins normally
    for (int ei = 0; ei < nEtaBins; ei++) {
      EtaBinsLow[ei] = EtaBinSize * ei + EtaMin;
      EtaBinsHigh[ei] = EtaBinSize * (ei + 1) + EtaMin;
    }
  } else {
    dg_build_bin_edges(RMin, EtaMin, nRBins, nEtaBins, RBinSize, EtaBinSize,
                       RBinsLow, RBinsHigh, EtaBinsLow, EtaBinsHigh);
  }

  // Initialize pixel-list arrays
  struct MapPixelData ***pxList = malloc(nRBins * sizeof(*pxList));
  int **nPxList = malloc(nRBins * sizeof(*nPxList));
  int **maxnPx = malloc(nRBins * sizeof(*maxnPx));
  int i, j, k;
  for (i = 0; i < nRBins; i++) {
    pxList[i] = malloc(nEtaBins * sizeof(*pxList[i]));
    nPxList[i] = malloc(nEtaBins * sizeof(*nPxList[i]));
    maxnPx[i] = malloc(nEtaBins * sizeof(*maxnPx[i]));
    for (j = 0; j < nEtaBins; j++) {
      pxList[i][j] = NULL;
      nPxList[i][j] = 0;
      maxnPx[i][j] = 0;
    }
  }

  // Allocate bin mask flag array (one int per R×Eta bin, zero-initialized)
  long long int nBinTotal = (long long int)nRBins * nEtaBins;
  int *binMaskFlag = calloc(nBinTotal, sizeof(int));

  // Initialize per-bin striped locks for thread-safe bin population
  omp_lock_t binLocks[MAPPER_N_BIN_LOCKS];
  for (int li = 0; li < MAPPER_N_BIN_LOCKS; li++)
    omp_init_lock(&binLocks[li]);

  // Run mapper
  if (SolidAngleCorrection)
    printf("  SolidAngleCorrection: ON (cos^3(2theta))\n");
  if (PolarizationCorrection)
    printf("  PolarizationCorrection: ON (fraction=%.4f)\n", PolarizationFraction);
  if (Parallax != 0.0)
    printf("  Parallax: %.2f µm\n", Parallax);
  long long int TotNrOfBins = mapper_build_map(
      tx, ty, tz, NrPixelsY, NrPixelsZ, pxY, pxZ, yCen, zCen, Lsd, RhoD, p0, p1,
      p2, p3, p4, p5, p6, p7, p8, p9, p10,
      EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh, nRBins, nEtaBins,
      pxList, nPxList, maxnPx, mask, binMaskFlag, NrTransOpt, TransOpt,
      binLocks, SubPixelLevel, SubPixelCardinalWidth,
      Parallax, SolidAngleCorrection, PolarizationCorrection, PolarizationFraction,
      distortionMapY, distortionMapZ, panels, nPanels);
  printf("Total Number of bins %lld\n", TotNrOfBins);
  fflush(stdout);

  // Destroy locks
  for (int li = 0; li < MAPPER_N_BIN_LOCKS; li++)
    omp_destroy_lock(&binLocks[li]);

  // Count flagged bins
  int nFlaggedBins = 0;
  for (long long int bi = 0; bi < nBinTotal; bi++)
    if (binMaskFlag[bi])
      nFlaggedBins++;
  printf("Bins contaminated by mask: %d out of %lld (%.2f%%)\n", nFlaggedBins,
         nBinTotal, 100.0 * nFlaggedBins / (nBinTotal > 0 ? nBinTotal : 1));

  // Serialize to flat arrays
  long long int LengthNPxList = (long long int)nRBins * nEtaBins;
  struct MapPixelData *pxListStore = malloc(TotNrOfBins * sizeof(*pxListStore));
  int *nPxListStore = malloc(LengthNPxList * 2 * sizeof(*nPxListStore));
  long long int Pos;
  int localNPxVal, localCounter = 0;
  for (i = 0; i < nRBins; i++) {
    for (j = 0; j < nEtaBins; j++) {
      localNPxVal = nPxList[i][j];
      Pos = (long long int)i * nEtaBins + j;
      nPxListStore[(Pos * 2) + 0] = localNPxVal;
      nPxListStore[(Pos * 2) + 1] = localCounter;
      for (k = 0; k < localNPxVal; k++) {
        pxListStore[localCounter + k].y = pxList[i][j][k].y;
        pxListStore[localCounter + k].z = pxList[i][j][k].z;
        pxListStore[localCounter + k].frac = pxList[i][j][k].frac;
        pxListStore[localCounter + k].deltaR = pxList[i][j][k].deltaR;
        pxListStore[localCounter + k].areaWeight = pxList[i][j][k].areaWeight;
      }
      localCounter += localNPxVal;
    }
  }

  // MapHeader
  struct MapHeader map_hdr;
  map_header_compute(&map_hdr, Lsd, yCen, zCen, pxY, pxZ, tx, ty, tz, p0, p1,
                     p2, p3, p4, p6, RhoD, RBinSize, EtaBinSize, RMin, RMax, EtaMin,
                     EtaMax, NrPixelsY, NrPixelsZ, NrTransOpt, TransOpt, qMode,
                     Wavelength);
  map_header_print("Map.bin", &map_hdr);

  // Write output (to resultFolder if set, else cwd)
  char mapfn[4096], nmapfn[4096];
  if (resultFolder != NULL) {
    snprintf(mapfn, sizeof(mapfn), "%s/Map.bin", resultFolder);
    snprintf(nmapfn, sizeof(nmapfn), "%s/nMap.bin", resultFolder);
  } else {
    snprintf(mapfn, sizeof(mapfn), "Map.bin");
    snprintf(nmapfn, sizeof(nmapfn), "nMap.bin");
  }
  FILE *mapfile = fopen(mapfn, "wb");
  FILE *nmapfile = fopen(nmapfn, "wb");
  if (!mapfile || !nmapfile) {
    fprintf(stderr, "Error: Could not open output files for writing %s %s.\n",
            mapfn, nmapfn);
    exit(EXIT_FAILURE);
  }
  map_header_write(mapfile, &map_hdr);
  map_header_write(nmapfile, &map_hdr);
  fwrite(pxListStore, TotNrOfBins * sizeof(*pxListStore), 1, mapfile);
  fwrite(nPxListStore, LengthNPxList * 2 * sizeof(*nPxListStore), 1, nmapfile);
  fclose(mapfile);
  fclose(nmapfile);

  // Write maskMap.bin (bin contamination flags)
  if (nFlaggedBins > 0) {
    char maskMapFN[4096];
    if (resultFolder != NULL)
      snprintf(maskMapFN, sizeof(maskMapFN), "%s/maskMap.bin", resultFolder);
    else
      snprintf(maskMapFN, sizeof(maskMapFN), "maskMap.bin");
    FILE *maskMapFile = fopen(maskMapFN, "wb");
    if (maskMapFile) {
      map_header_write(maskMapFile, &map_hdr);
      fwrite(binMaskFlag, nBinTotal * sizeof(int), 1, maskMapFile);
      fclose(maskMapFile);
      printf("Wrote %s (%d flagged bins)\n", maskMapFN, nFlaggedBins);
    }
  }
  free(binMaskFlag);

  end0 = omp_get_wtime();
  diftotal = end0 - start0;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  return 0;
}
