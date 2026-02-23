//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "midas_paths.h"
#include "nf_headers.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
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

#define RealType double
#define float32_t float
#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define ClearBit(A, k) (A[(k / 32)] &= ~(1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

int Flag = 0;
double Wedge;
double Wavelength;
double OmegaRang[MAX_N_OMEGA_RANGES][2];
int nOmeRang;
int SpaceGrp;

double **allocMatrixF(int nrows, int ncols) {
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

int **allocMatrixIntF(int nrows, int ncols) {
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

// check() is now provided by MIDAS_Limits.h (via nf_headers.h)

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage:\n compareNF params.txt InputMicFN ConfDiff\n Must have "
           "SpotInfo.bin in the same directory.\n");
    return 1;
  }

  clock_t start, end;
  double diftotal;
  start = clock();

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  run_midas_binary("GetHKLList", ParamFN);
  char cmmd[4096];
  // SpotsInfo.bin copy removed — will mmap directly from DataDirectory below
  printf("Will mmap SpotsInfo.bin directly from DataDirectory\n");
  char *MicFN = argv[2];
  double fracThresh = atof(argv[3]);
  char aline[4096];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[4096];
  int LowNr, nLayers;
  double tx, ty, tz;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "nDistances ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nLayers);
      break;
    }
  }
  rewind(fileParam);
  double Lsd[nLayers], ybc[nLayers], zbc[nLayers], ExcludePoleAngle,
      LatticeConstant[6], minFracOverlap, doubledummy, MaxRingRad, MaxTtheta;
  double px, OmegaStart, OmegaStep, tol;
  char fn[1000];
  char fn2[1000];
  char direct[1000] = ".";
  char gridfn[1000];
  double OmegaRanges[MAX_N_OMEGA_RANGES][2], BoxSizes[MAX_N_OMEGA_RANGES][4];
  int cntr = 0, countr = 0, conter = 0, StartNr, EndNr, intdummy, SpaceGroup,
      RingsToUse[100], nRingsToUse = 0;
  int NoOfOmegaRanges = 0;
  int nSaves = 1;
  int gridfnfound = 0;
  Wedge = 0;
  int MinMiso = 0;
  int NrPixelsY = 2048, NrPixelsZ = 2048;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "ReducedFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, fn2);
      continue;
    }
    str = "GridFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, gridfn);
      gridfnfound = 1;
      continue;
    }
    str = "SaveNSolutions ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nSaves);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Lsd[cntr]);
      cntr++;
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SpaceGroup);
      continue;
    }
    str = "MaxRingRad ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
      continue;
    }
    str = "StartNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &StartNr);
      continue;
    }
    str = "EndNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &EndNr);
      continue;
    }
    str = "ExcludePoleAngle ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &ExcludePoleAngle);
      continue;
    }
    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeConstant[0],
             &LatticeConstant[1], &LatticeConstant[2], &LatticeConstant[3],
             &LatticeConstant[4], &LatticeConstant[5]);
      continue;
    }
    str = "tx ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tx);
      continue;
    }
    str = "ty ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &ty);
      continue;
    }
    str = "BC ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &ybc[conter], &zbc[conter]);
      conter++;
      continue;
    }
    str = "tz ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tz);
      continue;
    }
    str = "OrientTol ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tol);
      continue;
    }
    str = "MinFracAccept ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &minFracOverlap);
      continue;
    }
    str = "OmegaStart ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStart);
      continue;
    }
    str = "OmegaStep ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &OmegaStep);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "Wedge ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wedge);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "RingsToUse ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRingsToUse]);
      nRingsToUse++;
      continue;
    }
    str = "OmegaRange ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf", dummy, &OmegaRanges[NoOfOmegaRanges][0],
             &OmegaRanges[NoOfOmegaRanges][1]);
      NoOfOmegaRanges++;
      continue;
    }
    str = "BoxSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf", dummy, &BoxSizes[countr][0],
             &BoxSizes[countr][1], &BoxSizes[countr][2], &BoxSizes[countr][3]);
      countr++;
      continue;
    }
    str = "Ice9Input ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      Flag = 1;
      continue;
    }
    str = "NearestMisorientation ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &MinMiso);
      continue;
    }
    str = "NrPixels ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      NrPixelsZ = NrPixelsY;
      continue;
    }
    str = "NrPixelsY ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      continue;
    }
    str = "NrPixelsZ ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
      continue;
    }
  }
  int i, j, m, nrFiles, nrPixels;
  for (i = 0; i < NoOfOmegaRanges; i++) {
    OmegaRang[i][0] = OmegaRanges[i][0];
    OmegaRang[i][1] = OmegaRanges[i][1];
  }
  nOmeRang = NoOfOmegaRanges;
  fclose(fileParam);
  MaxTtheta = rad2deg * atan(MaxRingRad / Lsd[0]);
  int *ObsSpotsInfo;
  nrFiles = EndNr - StartNr + 1;
  nrPixels = NrPixelsY * NrPixelsZ;
  long long int SizeObsSpots;
  SizeObsSpots = (nLayers);
  SizeObsSpots *= nrPixels;
  SizeObsSpots *= nrFiles;
  SizeObsSpots /= 32;
  // printf("%lld\n",SizeObsSpots*32);

  // Read spots info (mmap directly from DataDirectory)
  char file_name[1024];
  sprintf(file_name, "%s/SpotsInfo.bin", direct);
  int descp;
  struct stat s;
  int status;
  size_t size;
  int rc;
  descp = open(file_name, O_RDONLY);
  check(descp < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(descp, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  ObsSpotsInfo = mmap(0, size, PROT_READ, MAP_SHARED, descp, 0);
  check(ObsSpotsInfo == MAP_FAILED, "mmap %s failed: %s", file_name,
        strerror(errno));

  double RotMatTilts[3][3];
  RotationTilts(tx, ty, tz, RotMatTilts);
  double MatIn[3], P0[nLayers][3], P0T[3];
  double xs, ys, edgeLen, gs, ud, eulThis[3], origConf, XG[3], YG[3], dy1, dy2;
  MatIn[0] = 0;
  MatIn[1] = 0;
  MatIn[2] = 0;
  for (i = 0; i < nLayers; i++) {
    MatIn[0] = -Lsd[i];
    MatrixMultF(RotMatTilts, MatIn, P0T);
    for (j = 0; j < 3; j++) {
      P0[i][j] = P0T[j];
    }
  }
  int n_hkls = 0;
  double hkls[5000][4];
  double Thetas[5000];
  char hklfn[1024];
  sprintf(hklfn, "hkls.csv");
  FILE *hklf = fopen(hklfn, "r");
  fgets(aline, 1000, hklf);
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%s %s %s %s %lf %lf %lf %lf %lf %s %s", dummy, dummy, dummy,
           dummy, &hkls[n_hkls][3], &hkls[n_hkls][0], &hkls[n_hkls][1],
           &hkls[n_hkls][2], &Thetas[n_hkls], dummy, dummy);
    n_hkls++;
  }
  if (nRingsToUse > 0) {
    double hklTemps[n_hkls][4], thetaTemps[n_hkls];
    int totalHKLs = 0;
    for (i = 0; i < nRingsToUse; i++) {
      for (j = 0; j < n_hkls; j++) {
        if ((int)hkls[j][3] == RingsToUse[i]) {
          hklTemps[totalHKLs][0] = hkls[j][0];
          hklTemps[totalHKLs][1] = hkls[j][1];
          hklTemps[totalHKLs][2] = hkls[j][2];
          hklTemps[totalHKLs][3] = hkls[j][3];
          thetaTemps[totalHKLs] = Thetas[j];
          totalHKLs++;
        }
      }
    }
    for (i = 0; i < totalHKLs; i++) {
      hkls[i][0] = hklTemps[i][0];
      hkls[i][1] = hklTemps[i][1];
      hkls[i][2] = hklTemps[i][2];
      hkls[i][3] = hklTemps[i][3];
      Thetas[i] = thetaTemps[i];
    }
    n_hkls = totalHKLs;
  }
  // Precompute Gs for CalcDiffractionSpots optimization
  double Gs[n_hkls];
  for (i = 0; i < n_hkls; i++) {
    double len = sqrt(hkls[i][0] * hkls[i][0] + hkls[i][1] * hkls[i][1] +
                      hkls[i][2] * hkls[i][2]);
    Gs[i] = sin(Thetas[i] * deg2rad) * len;
  }
  double OMIn[3][3], FracCalc;
  FILE *InpMicF;
  InpMicF = fopen(MicFN, "r");
  char outFN[4096];
  sprintf(outFN, "%s.output.csv", MicFN);
  FILE *outMicF;
  outMicF = fopen(outFN, "w");
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  fgets(aline, 4096, InpMicF);
  int lineNr = 0;
  double *TheorSpots;
  TheorSpots = malloc(MAX_N_SPOTS * 3 * sizeof(*TheorSpots));
  while (fgets(aline, 4096, InpMicF) != NULL) {
    sscanf(aline, "%s %s %s %lf %lf %lf %lf %lf %lf %lf %lf %s", dummy, dummy,
           dummy, &xs, &ys, &edgeLen, &ud, &eulThis[0], &eulThis[1],
           &eulThis[2], &origConf, dummy);
    gs = edgeLen / 2;
    dy1 = edgeLen / sqrt(3);
    dy2 = -edgeLen / (2 * sqrt(3));
    if (ud < 0) {
      dy1 *= -1;
      dy2 *= -1;
    }
    int NrPixelsGrid = 2 * (ceil((gs * 2) / px)) * (ceil((gs * 2) / px));
    XG[0] = xs;
    XG[1] = xs - gs;
    XG[2] = xs + gs;
    YG[0] = ys + dy1;
    YG[1] = ys + dy2;
    YG[2] = ys + dy2;
    // We need to make diffraction spots now. Use eulThis to calc OM, then
    // calcDiffrSpots
    eulThis[0] = eulThis[0] * rad2deg;
    eulThis[1] = eulThis[1] * rad2deg;
    eulThis[2] = eulThis[2] * rad2deg;
    Euler2OrientMat(eulThis, OMIn);
    int **InPixels;
    InPixels = allocMatrixIntF(NrPixelsGrid, 2);
    CalcOverlapAccOrient(nrFiles, nLayers, ExcludePoleAngle, Lsd, SizeObsSpots,
                         XG, YG, RotMatTilts, OmegaStart, OmegaStep, px, ybc,
                         zbc, gs, hkls, n_hkls, Thetas, OmegaRanges,
                         NoOfOmegaRanges, BoxSizes, P0, NrPixelsGrid,
                         ObsSpotsInfo, OMIn, &FracCalc, TheorSpots, InPixels,
                         Gs, NrPixelsY, NrPixelsZ);
    FreeMemMatrixInt(InPixels, NrPixelsGrid);
    lineNr += 1;
    if (origConf - FracCalc > fracThresh) {
      fprintf(outMicF, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf 0\n",
              (double)lineNr, FracCalc, origConf - FracCalc, xs, ys, edgeLen,
              ud, eulThis[0], eulThis[1], eulThis[2], origConf);
    } else {
      fprintf(outMicF, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf 1\n",
              (double)lineNr, FracCalc, origConf - FracCalc, xs, ys, edgeLen,
              ud, eulThis[0], eulThis[1], eulThis[2], origConf);
    }
  }
  return 0;
}