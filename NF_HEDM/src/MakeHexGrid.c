//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
//
//
//  Created by Hemant Sharma on 2013/11/13.
//
//

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define RealType double

static inline void HexGrid(double GridSize, double Rsample, double NrHex,
                           double HtTriangle, double ALast, double **XY,
                           double EdgeLength) {
  int i, j;
  int counter = 0;
  int NrRowElements;
  double ythis, xstart, ynext, ysmall, ybig, xt1, xt2;
  ysmall = HtTriangle * (1.0 / 3.0);
  ybig = HtTriangle * (2.0 / 3.0);
  xt1 = EdgeLength * sqrt(3) / 6;
  xt2 = EdgeLength * sqrt(3) * 2 / 6;
  for (i = -NrHex; i <= NrHex; i++) {
    if (i == 0) {
      continue;
    }
    if (i < 0) {
      ynext = ybig;
    } else {
      ynext = ysmall;
    }
    NrRowElements = (2 * ((2 * NrHex) - (fabs(i)))) + 1;
    ythis = HtTriangle * i;
    xstart = -ALast + (fabs(i) * GridSize * 0.5);
    for (j = 0; j < NrRowElements; j++) {
      if (ynext == ybig) {
        //~ XY[counter][0] = ynext;
        //~ XY[counter][1] = ysmall;
        XY[counter][0] = xt2;
        XY[counter][1] = xt1;
      } else {
        //~ XY[counter][0] = ynext;
        //~ XY[counter][1] = ybig;
        XY[counter][0] = xt1;
        XY[counter][1] = xt2;
      }
      XY[counter][2] = xstart + (GridSize * j) / 2;
      XY[counter][3] = ythis - (ynext * i / (fabs(i)));
      //~ XY[counter][4] = GridSize/2;
      XY[counter][4] = EdgeLength / 2;
      counter++;
      if (ynext == ybig) {
        ynext = ysmall;
      } else {
        ynext = ybig;
      }
    }
  }
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

static inline void FreeMemMatrix(RealType **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

int main(int argc, char *argv[]) {
  clock_t start, end;
  double diftotal;
  start = clock();

  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000], direct[1024], gridfn[1000];
  int LowNr;
  int gridfnfound = 0;
  double GridSize, Rsample, EdgeLength = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "GridSize ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &GridSize);
      continue;
    }
    str = "EdgeLength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &EdgeLength);
      continue;
    }

    str = "Rsample ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Rsample);
      continue;
    }
    str = "DataDirectory ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, direct);
      continue;
    }
    str = "GridFileName ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, gridfn);
      gridfnfound = 1;
      continue;
    }
  }
  fclose(fileParam);
  if (EdgeLength == 0)
    EdgeLength = GridSize;

  // Print all parameters read from parameter file
  printf("\n");
  printf("================================================================\n");
  printf("            MakeHexGrid: Parsed Parameters Summary\n");
  printf("================================================================\n");
  printf("\n--- File Paths ---\n");
  printf("  DataDirectory:      %s\n", direct);
  if (gridfnfound)
    printf("  GridFileName:       %s\n", gridfn);
  else
    printf("  GridFileName:       grid.txt (default)\n");
  printf("\n--- Grid Parameters ---\n");
  printf("  GridSize:             %f\n", GridSize);
  printf("  EdgeLength:           %f%s\n", EdgeLength,
         (EdgeLength == GridSize) ? " (defaulted to GridSize)" : "");
  printf("  Rsample:              %f\n", Rsample);
  printf(
      "================================================================\n\n");

  // Make grid.
  double NrHex, ALarge, ALast, HtTriangle;
  int i, j;
  int NrGridElements = 0;
  FILE *fp;
  ALarge = (2 * Rsample) / (sqrt(3));
  HtTriangle = ((sqrt(3)) * (GridSize)) / (2);
  NrHex = ceil(ALarge / GridSize);
  ALast = GridSize * NrHex;
  for (i = 1; i <= NrHex; i++) {
    NrGridElements += 2 * ((2 * ((2 * NrHex) - i)) + 1);
  }
  double **XYGrid;
  XYGrid = allocMatrix(NrGridElements, 5);
  HexGrid(GridSize, Rsample, NrHex, HtTriangle, ALast, XYGrid, EdgeLength);
  printf("Number of grid points: %d.\n", NrGridElements);
  char fn[1024];
  if (gridfnfound == 1)
    sprintf(fn, "%s/%s", direct, gridfn);
  else
    sprintf(fn, "%s/grid.txt", direct);
  // sprintf(fn,"%s/grid.txt",direct);
  fp = fopen(fn, "w");
  if (fp == NULL) {
    printf("Cannot open file, %s\n", fn);
    return (0);
  }
  fprintf(fp, "%d\n", NrGridElements);
  for (j = 0; j < NrGridElements; j++) {
    fprintf(fp, "%f %f %f %f %f\n", XYGrid[j][0], XYGrid[j][1], XYGrid[j][2],
            XYGrid[j][3], XYGrid[j][4]);
  }
  fclose(fp);
  FreeMemMatrix(XYGrid, NrGridElements);
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed in making HexGrid: %f [s]\n", diftotal);
  return 0;
}
