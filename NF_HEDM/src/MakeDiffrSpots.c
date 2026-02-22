//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  MakeDiffrSpots.c
//
//
//  Created by Hemant Sharma on 2013/11/19
//
//

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define RealType double
#include "../../FF_HEDM/src/MIDAS_Limits.h"
#define EPS 0.000000001

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define crossProduct(a, b, c)                                                  \
  (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];                                  \
  (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];                                  \
  (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

#define dot(v, q) ((v)[0] * (q)[0] + (v)[1] * (q)[1] + (v)[2] * (q)[2])

#define CalcLength(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

int n_hkls = 0;
double hkls[MAX_N_HKLS][4];
double Thetas[MAX_N_HKLS];

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

static inline void MatrixMult(RealType m[3][3], int v[3], RealType r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void MatrixMultF(RealType m[3][3], RealType v[3], RealType r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void QuatToOrientMat(double Quat[4], double OrientMat[3][3]) {
  double Q1_2, Q2_2, Q3_2, Q12, Q03, Q13, Q02, Q23, Q01;
  Q1_2 = Quat[1] * Quat[1];
  Q2_2 = Quat[2] * Quat[2];
  Q3_2 = Quat[3] * Quat[3];
  Q12 = Quat[1] * Quat[2];
  Q03 = Quat[0] * Quat[3];
  Q13 = Quat[1] * Quat[3];
  Q02 = Quat[0] * Quat[2];
  Q23 = Quat[2] * Quat[3];
  Q01 = Quat[0] * Quat[1];
  OrientMat[0][0] = 1 - 2 * (Q2_2 + Q3_2);
  OrientMat[0][1] = 2 * (Q12 - Q03);
  OrientMat[0][2] = 2 * (Q13 + Q02);
  OrientMat[1][0] = 2 * (Q12 + Q03);
  OrientMat[1][1] = 1 - 2 * (Q1_2 + Q3_2);
  OrientMat[1][2] = 2 * (Q23 - Q01);
  OrientMat[2][0] = 2 * (Q13 - Q02);
  OrientMat[2][1] = 2 * (Q23 + Q01);
  OrientMat[2][2] = 1 - 2 * (Q1_2 + Q2_2);
}

static inline void RotateAroundZ(RealType v1[3], RealType alpha,
                                 RealType v2[3]) {
  RealType cosa = cos(alpha * deg2rad);
  RealType sina = sin(alpha * deg2rad);

  RealType mat[3][3] = {{cosa, -sina, 0}, {sina, cosa, 0}, {0, 0, 1}};

  MatrixMultF(mat, v1, v2);
}

static inline void CalcEtaAngle(RealType y, RealType z, RealType *alpha) {
  *alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    *alpha = -*alpha;
}

static inline void CalcSpotPosition(RealType RingRadius, RealType eta,
                                    RealType *yl, RealType *zl) {
  RealType etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}

static inline void CalcOmega(RealType x, RealType y, RealType z, RealType theta,
                             RealType omegas[4], RealType etas[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  RealType len = sqrt(x * x + y * y + z * z);
  RealType v = sin(theta * deg2rad) * len;

  RealType almostzero = 1e-4;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= 1) {
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

static inline void FreeMemMatrix(RealType **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

static inline void
CalcDiffrSpots_Furnace(RealType OrientMatrix[3][3], RealType distance,
                       RealType OmegaRange[][2], RealType BoxSizes[][4],
                       int NOmegaRanges, RealType ExcludePoleAngle,
                       int n_hkls_local, double hkls_local[][4],
                       double Thetas_local[], RealType *spots, int *nspots) {
  int i, OmegaRangeNo;
  RealType theta;
  int KeepSpot;
  double Ghkl[3];
  int indexhkl;
  RealType Gc[3];
  RealType omegas[4];
  RealType etas[4];
  RealType yl;
  RealType zl;
  int nspotsPlane;
  int spotnr = 0;
  for (indexhkl = 0; indexhkl < n_hkls_local; indexhkl++) {
    Ghkl[0] = hkls_local[indexhkl][0];
    Ghkl[1] = hkls_local[indexhkl][1];
    Ghkl[2] = hkls_local[indexhkl][2];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = Thetas_local[indexhkl];
    RealType RingRadius = distance * tan(2 * deg2rad * theta);
    // printf("%d %d %d %f %f\n",Ghkl[0],Ghkl[1],Ghkl[2],theta,RingRadius);
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
      if (KeepSpot == 1) {
        spots[spotnr * 3 + 0] = yl;
        spots[spotnr * 3 + 1] = zl;
        spots[spotnr * 3 + 2] = omegas[i];
        spotnr++;
        spotnr++;
      }
    }
  }
  *nspots = spotnr;
}

static inline void usage(void) {
  printf("Make diffraction spots: usage: ./MakeDiffrSpots <ParameterFile> "
         "[nCPUs]\n");
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 3) {
    usage();
    return 1;
  }
  clock_t start0, end;
  double diftotal;
  start0 = clock();
  int i, j, k, t, p;
  int nCPUs = 1;
  if (argc == 3) {
    nCPUs = atoi(argv[2]);
  }

  // Read params file.
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000], direct[1024], OF[1024];
  int LowNr, NrOrientations;
  RealType Distance;
  double Distances[10], ExcludePoleAngle, LatticeConstant, Wavelength;
  RealType OmegaRanges[20][2], BoxSizes[20][4], px, MaxTtheta, MaxRingRad, a, b,
      c, alpha, beta, gamma;
  int cntr = 0, countr = 0, RingsToUse[100], nRingsToUse = 0;
  int NoOfOmegaRanges = 0, SpaceGroup;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Distances[cntr]);
      cntr++;
      continue;
    }
    str = "RingsToUse ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &RingsToUse[nRingsToUse]);
      nRingsToUse++;
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %s", direct);
      continue;
    }
    str = "SeedOrientations ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%*s %s", OF);
      continue;
    }
    str = "NrOrientations ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NrOrientations);
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SpaceGroup);
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
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &a, &b, &c, &alpha,
             &beta, &gamma);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
      continue;
    }
    str = "px ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &px);
      continue;
    }
    str = "MaxRingRad ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &MaxRingRad);
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
  }
  fclose(fileParam);

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("          MakeDiffrSpots: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:      %s\n", direct);
  printf("  SeedOrientations:   %s\n", OF);
  printf("\n--- Geometry ---\n");
  printf("  SpaceGroup:           %d\n", SpaceGroup);
  printf("  Wavelength:           %f\n", Wavelength);
  printf("  px (pixel size):      %f\n", px);
  printf("  MaxRingRad:           %f\n", MaxRingRad);
  printf("  NrOrientations:       %d\n", NrOrientations);
  printf("  ExcludePoleAngle:     %f\n", ExcludePoleAngle);
  printf("\n--- Detector Distances (Lsd) ---\n");
  for (i = 0; i < cntr; i++) {
    printf("  Lsd[%d]:               %f\n", i, Distances[i]);
  }
  printf("\n--- Lattice Parameters ---\n");
  printf("  a=%f  b=%f  c=%f\n", a, b, c);
  printf("  alpha=%f  beta=%f  gamma=%f\n", alpha, beta, gamma);
  printf("\n--- Omega Ranges (%d) ---\n", NoOfOmegaRanges);
  for (i = 0; i < NoOfOmegaRanges; i++) {
    printf("  OmegaRange[%d]:       %f  %f\n", i, OmegaRanges[i][0],
           OmegaRanges[i][1]);
  }
  printf("\n--- Box Sizes ---\n");
  for (i = 0; i < countr; i++) {
    printf("  BoxSize[%d]:          %f  %f  %f  %f\n", i, BoxSizes[i][0],
           BoxSizes[i][1], BoxSizes[i][2], BoxSizes[i][3]);
  }
  printf("\n--- Rings To Use (%d) ---\n", nRingsToUse);
  for (i = 0; i < nRingsToUse; i++) {
    printf("  Ring: %d\n", RingsToUse[i]);
  }
  printf(
      "================================================================\n\n");

  MaxTtheta = (180 / M_PI) * atan(MaxRingRad / Distances[0]);
  Distance = Distances[0];
  FILE *fp;

  double quat1, quat2, quat3, quat4;
  double **randQ;
  randQ = allocMatrix(NrOrientations, 4);
  fp = fopen(OF, "r");
  if (fp == NULL) {
    printf("Could not read file %s\n", OF);
    return (1);
  }
  int count = 0;
  while (fgets(aline, 1000, fp) != NULL) {
    if (count > NrOrientations) {
      printf("Nr of Orientations mismatch. Exiting.\n");
      return (1);
    }
    sscanf(aline, "%lf,%lf,%lf,%lf", &quat1, &quat2, &quat3, &quat4);
    randQ[count][0] = quat1;
    randQ[count][1] = quat2;
    randQ[count][2] = quat3;
    randQ[count][3] = quat4;
    count += 1;
  }
  if (count != NrOrientations) {
    printf("Nr of Orientations mismatch. Exiting.\n");
    return (1);
  }

  // For each position and each orientation, generate spots and save.

  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  if (hklf == NULL) {
    printf("Could not read file %s\n", hklfn);
    return (1);
  }
  fgets(aline, 1000, hklf); // skip header line
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
  printf("Number of individual diffracting planes: %d\n", n_hkls);
  int nRowsPerGrain = 2 * n_hkls;

  // Allocate per-orientation storage for parallel computation
  // Allocate per-orientation storage for parallel computation
  int *allNTspots = malloc(NrOrientations * sizeof(*allNTspots));
  double *allOrientMatr = malloc(NrOrientations * 9 * sizeof(*allOrientMatr));

  // Compact memory: array of pointers to variable-sized arrays
  double **allSpots = malloc(NrOrientations * sizeof(double *));

  if (allNTspots == NULL || allOrientMatr == NULL || allSpots == NULL) {
    printf(
        "Memory error: could not allocate memory for output matrices. Memory "
        "full?\n");
    return 1;
  }

  // Parallel computation: each thread uses private buffers
#pragma omp parallel for num_threads(nCPUs)
  for (j = 0; j < NrOrientations; j++) {
    double QuatLocal[4], OrientLocal[3][3];
    int nTspotsLocal;

    // Safer allocation on heap instead of stack VLA
    RealType *SpotsLocal = malloc(nRowsPerGrain * 3 * sizeof(RealType));

    for (k = 0; k < 4; k++) {
      QuatLocal[k] = randQ[j][k];
    }
    QuatToOrientMat(QuatLocal, OrientLocal);
    CalcDiffrSpots_Furnace(OrientLocal, Distance, OmegaRanges, BoxSizes,
                           NoOfOmegaRanges, ExcludePoleAngle, n_hkls, hkls,
                           Thetas, SpotsLocal, &nTspotsLocal);

    // Store results into per-orientation arrays
    allNTspots[j] = nTspotsLocal;
    for (t = 0; t < 3; t++) {
      for (p = 0; p < 3; p++) {
        allOrientMatr[j * 9 + t * 3 + p] = OrientLocal[t][p];
      }
    }

    // Allocate exact size for this orientation
    if (nTspotsLocal > 0) {
      allSpots[j] = malloc(nTspotsLocal * 3 * sizeof(double));
      memcpy(allSpots[j], SpotsLocal, nTspotsLocal * 3 * sizeof(double));
    } else {
      allSpots[j] = NULL;
    }

    free(SpotsLocal);
  }

  // Sequential writes in j-loop order
  FILE *ft;
  char DSFN[1024];
  char KeyFN[1024];
  char OMFN[1024];
  sprintf(DSFN, "%s/DiffractionSpots.txt", direct);
  sprintf(KeyFN, "%s/Key.txt", direct);
  sprintf(OMFN, "%s/OrientMat.txt", direct);

  ft = fopen(DSFN, "w");
  FILE *fg = fopen(KeyFN, "w");
  FILE *fl = fopen(OMFN, "w");

  if (ft == NULL || fg == NULL || fl == NULL) {
    printf("Could not open one or more output files for writing.\n");
    if (ft)
      fclose(ft);
    if (fg)
      fclose(fg);
    if (fl)
      fclose(fl);
    return (1);
  }

  // Use large buffers for output streams (1MB)
  char buf_ft[1048576];
  char buf_fg[1048576];
  char buf_fl[1048576];
  setvbuf(ft, buf_ft, _IOFBF, sizeof(buf_ft));
  setvbuf(fg, buf_fg, _IOFBF, sizeof(buf_fg));
  setvbuf(fl, buf_fl, _IOFBF, sizeof(buf_fl));

  fprintf(fg, "%i\n", NrOrientations);
  for (j = 0; j < NrOrientations; j++) {
    fprintf(fg, "%i\n", allNTspots[j]);
    for (t = 0; t < 3; t++) {
      for (p = 0; p < 3; p++) {
        fprintf(fl, "%f ", allOrientMatr[j * 9 + t * 3 + p]);
      }
    }
    fprintf(fl, "\n");
    for (i = 0; i < allNTspots[j]; i++) {
      fprintf(ft, "%f %f %f\n", allSpots[j][i * 3 + 0], allSpots[j][i * 3 + 1],
              allSpots[j][i * 3 + 2]);
    }
  }

  // Cleanup
  fclose(ft);
  fclose(fg);
  fclose(fl);
  FreeMemMatrix(randQ, NrOrientations);

  free(allNTspots);
  free(allOrientMatr);

  for (j = 0; j < NrOrientations; j++) {
    if (allSpots[j] != NULL)
      free(allSpots[j]);
  }
  free(allSpots);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("Time elapsed in making diffraction spots: %f [s]\n", diftotal);
  return 0;
}
