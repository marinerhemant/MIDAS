//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  SimulateDiffractionSpots.c
//
//
//  Created by Hemant Sharma on 2014/10/22.
//
//

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define RealType double
#define MAX_N_HKLS 5000
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

static inline void CalcDiffrSpots_Furnace(RealType OrientMatrix[3][3],
                                          RealType distance, RealType **spots,
                                          int *nspots) {
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
  int spotid = 0;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0];
    Ghkl[1] = hkls[indexhkl][1];
    Ghkl[2] = hkls[indexhkl][2];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = Thetas[indexhkl];
    RealType RingRadius = distance * tan(2 * deg2rad * theta);
    CalcOmega(Gc[0], Gc[1], Gc[2], theta, omegas, etas, &nspotsPlane);
    for (i = 0; i < nspotsPlane; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      spots[spotnr][0] = RingRadius;
      spots[spotnr][1] = Eta;
      spots[spotnr][2] = omegas[i];
      spots[spotnr][3] = Ghkl[0];
      spots[spotnr][4] = Ghkl[1];
      spots[spotnr][5] = Ghkl[2];
      spotnr++;
      spotid++;
    }
  }
  *nspots = spotnr;
}

static inline void usage(void) {
  printf("Make diffraction spots: usage: ./MakeDiffrSpots "
         "<distance> <OrientationFile>\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    usage();
    return 1;
  }
  clock_t start0, end;
  double diftotal;
  start0 = clock();
  int i, j, k, t, p;
  double Distance = atof(argv[1]);
  char *OF, aline[1024];
  OF = argv[2];
  FILE *fp;
  double QuatThis[4], OrientMatr[3][3];
  double quat1, quat2, quat3, quat4;
  double **randQ;
  int NrOrientations = 1000000;
  randQ = allocMatrix(NrOrientations, 4);
  fp = fopen(OF, "r");
  if (fp == NULL) {
    printf("Could not read file %s\n", OF);
    return (1);
  }
  int count = 0;
  while (fgets(aline, 1000, fp) != NULL) {
    sscanf(aline, "%lf,%lf,%lf,%lf", &quat1, &quat2, &quat3, &quat4);
    randQ[count][0] = quat1;
    randQ[count][1] = quat2;
    randQ[count][2] = quat3;
    randQ[count][3] = quat4;
    count += 1;
  }
  NrOrientations = count;

  // For each position and each orientation, generate spots and save.
  RealType **TheorSpots;
  int nTspots;
  char *rc;
  char *hklfn = "hkls.csv";
  char dummy[1024];
  FILE *hklf = fopen(hklfn, "r");
  rc = fgets(aline, 1000, hklf);
  int hklsInts[MAX_N_HKLS][3];
  while (fgets(aline, 1000, hklf) != NULL) {
    sscanf(aline, "%d %d %d %s %lf %lf %lf %lf %lf %s %s", &hklsInts[n_hkls][0],
           &hklsInts[n_hkls][1], &hklsInts[n_hkls][2], dummy, &hkls[n_hkls][3],
           &hkls[n_hkls][0], &hkls[n_hkls][1], &hkls[n_hkls][2],
           &Thetas[n_hkls], dummy, dummy);
    n_hkls++;
  }
  printf("Number of individual diffracting planes: %d\n", n_hkls);
  printf("Number of orientations: %d\n", NrOrientations);
  int nRowsPerGrain = 2 * n_hkls;
  TheorSpots = allocMatrix(nRowsPerGrain, 6);
  if (TheorSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }
  FILE *ft;
  char DSFN[1024];
  char KeyFN[1024];
  char OMFN[1024];
  sprintf(DSFN, "SimulatedDiffractionSpots.txt");
  sprintf(KeyFN, "DiffractionSpotsKey.txt");
  sprintf(OMFN, "OrientationMatrices.txt");
  ft = fopen(DSFN, "w");
  FILE *fg;
  fg = fopen(KeyFN, "w");
  FILE *fl;
  fl = fopen(OMFN, "w");
  if (ft == NULL) {
    printf("Could not open DiffractionSpots.txt file for writing.\n");
    return (1);
  }
  fprintf(fg, "%i\n", NrOrientations);
  for (j = 0; j < NrOrientations; j++) {
    for (k = 0; k < 4; k++) {
      QuatThis[k] = randQ[j][k];
    }
    QuatToOrientMat(QuatThis, OrientMatr);
    CalcDiffrSpots_Furnace(OrientMatr, Distance, TheorSpots, &nTspots);
    fprintf(fg, "%i\n", nTspots);
    for (t = 0; t < 3; t++) {
      for (p = 0; p < 3; p++) {
        fprintf(fl, "%f ", OrientMatr[t][p]);
      }
    }
    fprintf(fl, "\n");
    for (i = 0; i < nTspots; i++) {
      for (k = 0; k < 6; k++) {
        fprintf(ft, "%f ", TheorSpots[i][k]);
      }
      for (k = 0; k < 3; k++) {
        fprintf(ft, "%d ", hklsInts[i / 2][k]);
      }
      fprintf(ft, "\n");
    }
  }
  fclose(ft);
  fclose(fg);
  fclose(fl);
  FreeMemMatrix(randQ, NrOrientations);
  FreeMemMatrix(TheorSpots, nRowsPerGrain);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("Time elapsed in making diffraction spots: %f [s]\n", diftotal);
  return 0;
}
