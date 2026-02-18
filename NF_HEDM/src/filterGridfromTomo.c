#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char gridFN[1024] = "grid.txt";
  double pxTomo = atof(argv[2]);
  char *tomoFN;
  tomoFN = argv[1];
  FILE *tomoF = fopen(tomoFN, "rb");
  fseek(tomoF, 0L, SEEK_END);
  size_t sz = ftell(tomoF);
  rewind(tomoF);
  size_t nrPxTomo = sqrt(sz);
  uint8_t *imTomo;
  imTomo = calloc(nrPxTomo * nrPxTomo, sizeof(uint8_t));
  fread(imTomo, sz * sizeof(uint8_t), 1, tomoF);
  fclose(tomoF);
  printf("Size of tomo image: %d Pixel dimension: %d\n", (int)sz,
         (int)nrPxTomo);
  FILE *gridF, *gridOut;
  gridF = fopen("grid.txt", "r");
  gridOut = fopen("gridInter.txt", "w");
  char aline[1000], dummy[1000];
  fgets(aline, 1000, gridF);
  size_t nrGridPoints;
  sscanf(aline, "%zu", &nrGridPoints);
  printf("Number of grid points: %zu\n", nrGridPoints);
  double *gridPoints, x, y;
  int xPos, yPos;
  gridPoints = calloc(3 * nrGridPoints, sizeof(*gridPoints));
  size_t i, nrRows = 0;
  for (i = 0; i < nrGridPoints; i++) {
    fgets(aline, 1000, gridF);
    sscanf(aline, "%s %s %lf %lf %s", dummy, dummy, &x, &y, dummy);
    xPos = (int)((x / pxTomo) + ((double)nrPxTomo / 2));
    yPos = (int)((y / pxTomo) + ((double)nrPxTomo / 2));
    if (xPos >= 0 && yPos >= 0 && xPos < (int)nrPxTomo &&
        yPos < (int)nrPxTomo &&
        imTomo[nrPxTomo * (nrPxTomo - yPos) + (xPos)] != 0) {
      fprintf(gridOut, "%s", aline);
      nrRows++;
    }
  }
  fclose(gridOut);
  fclose(gridF);
  gridF = fopen("gridNew.txt", "w");
  gridOut = fopen("gridInter.txt", "r");
  fprintf(gridF, "%d\n", (int)nrRows);
  for (i = 0; i < nrRows; i++) {
    fgets(aline, 1000, gridOut);
    fprintf(gridF, "%s", aline);
  }
  fclose(gridOut);
  fclose(gridF);
}
