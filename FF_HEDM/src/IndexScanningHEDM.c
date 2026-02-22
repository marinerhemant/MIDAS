//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
// IndexScanningHEDM.c
//
// Created by Hemant Sharma on 2017/08/07
//
//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
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

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#include "MIDAS_Limits.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// For detector mapping!
extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;

int BigDetSize = 0;
int *BigDetector;
long long int totNrPixelsBigDetector;
double pixelsize;

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }
static inline double CalcNorm3(double x, double y, double z) {
  return sqrt((x) * (x) + (y) * (y) + (z) * (z));
}

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

static inline void DisplacementInTheSpot(double a, double b, double c,
                                         double xi, double yi, double zi,
                                         double omega, double *Displ_y,
                                         double *Displ_z, double *xtr) {
  double sinOme = sind(omega), cosOme = cosd(omega);
  double XC = (a * cosOme) - (b * sinOme), YC = (a * sinOme) + (b * cosOme),
         ZC = c;
  double IK[3], NormIK;
  IK[0] = xi - XC;
  IK[1] = yi - YC;
  IK[2] = zi - ZC;
  NormIK = sqrt((IK[0] * IK[0]) + (IK[1] * IK[1]) + (IK[2] * IK[2]));
  IK[0] = IK[0] / NormIK;
  IK[1] = IK[1] / NormIK;
  IK[2] = IK[2] / NormIK;
  *Displ_y = YC - ((XC * IK[1]) / (IK[0]));
  *Displ_z = ZC - ((XC * IK[2]) / (IK[0]));
  *xtr = YC; // Y is the beam movement direction.
}

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

static inline void SpotToGv(double xi, double yi, double zi, double Omega,
                            double theta, double *g1, double *g2, double *g3) {
  double CosOme = cosd(Omega), SinOme = sind(Omega), eta = CalcEtaAngle(yi, zi),
         TanEta = tand(-eta), SinTheta = sind(theta);
  double CosTheta = cosd(theta), CosW = 1, SinW = 0,
         k3 = SinTheta * (1 + xi) / ((yi * TanEta) + zi), k2 = TanEta * k3,
         k1 = -SinTheta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW), k3f = (k3 * CosW) - (k1 * SinW),
         k2f = k2;
  *g1 = (k1f * CosOme) + (k2f * SinOme);
  *g2 = (k2f * CosOme) - (k1f * SinOme);
  *g3 = k3f;
}

static inline double CalcRad(double y, double z) { return sqrt(y * y + z * z); }

static inline int StartsWith(const char *a, const char *b) {
  if (strncmp(a, b, strlen(b)) == 0)
    return 1;
  return 0;
}

// Arguments: Parameters File, Grains file, GrainPosition (x and y)
int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: ./IndexScanningHEDM params.txt Grains.csv xpos ypos "
           "positionNr.\n"
           "HKL file should be generated already.\n");
    return 1;
  }
  clock_t start, end;
  double diftotal;
  start = clock();
  char aline[4096], dummy[4096], positionsFN[4096], outdirpath[4096];
  char *GrainsFN, *paramsFN;
  paramsFN = argv[1];
  GrainsFN = argv[2];
  double x, y, z, Lsd, MinEta, OmegaRanges[50][2], BoxSizes[50][4], Wavelength,
      MargOme, MargRad, MargEta, Completeness, LatC[6], completenessTol = 0;
  int nBoxSizes = 0, nOmeRanges = 0, nRings = 0, cs2 = 0, RingNumbers[200],
      nLayers;
  x = atof(argv[3]);
  y = atof(argv[4]);
  int posNr = atoi(argv[5]);
  z = 0;
  FILE *fileParam;
  fileParam = fopen(paramsFN, "r");
  while (fgets(aline, 4096, fileParam) != NULL) {
    if (StartsWith(aline, "Lsd ")) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    } else if (StartsWith(aline, "Distance ")) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    } else if (StartsWith(aline, "RingThresh ")) {
      sscanf(aline, "%s %d", dummy, &RingNumbers[nRings]);
      nRings++;
    } else if (StartsWith(aline, "OmegaRange ")) {
      sscanf(aline, "%s %lf %lf", dummy, &OmegaRanges[nOmeRanges][0],
             &OmegaRanges[nOmeRanges][1]);
      nOmeRanges++;
    } else if (StartsWith(aline, "BoxSize ")) {
      sscanf(aline, "%s %lf %lf %lf %lf", dummy, &BoxSizes[nBoxSizes][0],
             &BoxSizes[nBoxSizes][1], &BoxSizes[nBoxSizes][2],
             &BoxSizes[nBoxSizes][3]);
      nBoxSizes++;
    } else if (StartsWith(aline, "MinEta ")) {
      sscanf(aline, "%s %lf", dummy, &MinEta);
    } else if (StartsWith(aline, "Wavelength ")) {
      sscanf(aline, "%s %lf", dummy, &Wavelength);
    } else if (StartsWith(aline, "MarginOme ")) {
      sscanf(aline, "%s %lf", dummy, &MargOme);
    } else if (StartsWith(aline, "Completeness ")) {
      sscanf(aline, "%s %lf", dummy, &Completeness);
    } else if (StartsWith(aline, "MarginRadial ")) {
      sscanf(aline, "%s %lf", dummy, &MargRad);
    } else if (StartsWith(aline, "MarginEta ")) {
      sscanf(aline, "%s %lf", dummy, &MargEta);
    } else if (StartsWith(aline, "PositionsFile ")) {
      sscanf(aline, "%s %s", dummy, positionsFN);
    } else if (StartsWith(aline, "CompletenessTol ")) {
      sscanf(aline, "%s %lf", dummy, &completenessTol);
    } else if (StartsWith(aline, "OutDirPath ")) {
      sscanf(aline, "%s %s", dummy, outdirpath);
    } else if (StartsWith(aline, "nLayers ")) {
      sscanf(aline, "%s %d", dummy, &nLayers);
    } else if (StartsWith(aline, "LatticeConstant ")) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatC[0], &LatC[1],
             &LatC[2], &LatC[3], &LatC[4], &LatC[5]);
    }
  }
  fclose(fileParam);
  int i, j, k;

  // Read hkls file
  char *hklfn = "hkls.csv";
  FILE *hklf = fopen(hklfn, "r");
  int ringNr, nhkls = 0;
  double ds, theta, rad, g1, g2, g3, **hkls;
  hkls = allocMatrix(MAX_N_HKLS, 7);
  while (fgets(aline, 4096, hklf) != NULL) {
    sscanf(aline, "%s %s %s %lf %d %lf %lf %lf %lf %s %lf", dummy, dummy, dummy,
           &ds, &ringNr, &g1, &g2, &g3, &theta, dummy, &rad);
    for (i = 0; i < nRings; i++) {
      if (ringNr == RingNumbers[i]) {
        hkls[nhkls][0] = g1;
        hkls[nhkls][1] = g2;
        hkls[nhkls][2] = g3;
        hkls[nhkls][3] = ds;
        hkls[nhkls][4] = theta;
        hkls[nhkls][5] = rad;
        hkls[nhkls][6] = (double)ringNr;
        nhkls++;
      }
    }
  }
  fclose(hklf);
  printf("Number of individual diffracting planes: %d\n", nhkls);

  // Read PositionsFile
  FILE *positionsFile;
  double positions[nLayers], ptemp;
  int IDsInfo[nLayers * nRings][4];
  positionsFile = fopen(positionsFN, "r");
  fgets(aline, 4096, positionsFile);
  int count = 0;
  while (fgets(aline, 4096, positionsFile) != NULL) {
    sscanf(aline, "%lf %s", &ptemp, dummy);
    positions[count] = 1000 * ptemp;
    count++;
  }
  fclose(positionsFile);

  // Read IDsHash.csv
  FILE *idsfile;
  char idsfn[4096];
  sprintf(idsfn, "%s/IDsHash.csv", outdirpath);
  idsfile = fopen(idsfn, "r");
  count = 0;
  int maxID = 0;
  while (fgets(aline, 4096, idsfile) != NULL) {
    sscanf(aline, "%d %d %d %d", &IDsInfo[count][0], &IDsInfo[count][1],
           &IDsInfo[count][2], &IDsInfo[count][3]);
    if (IDsInfo[count][3] > maxID) {
      maxID = IDsInfo[count][3];
    }
    count++;
  }
  fclose(idsfile);

  // Read ExtraInfo.bin
  const char *filename = "/dev/shm/ExtraInfo.bin";
  int rc;
  double *AllSpots;
  struct stat s;
  size_t size;
  int fd = open(filename, O_RDONLY);
  check(fd < 0, "open %s failed: %s", filename, strerror(errno));
  int status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", filename, strerror(errno));
  size = s.st_size;
  AllSpots = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(AllSpots == MAP_FAILED, "mmap %s failed: %s", filename,
        strerror(errno));
  int maxID2 = size / (14 * sizeof(double));
  double **allSpotsYZO;
  allSpotsYZO = allocMatrix(maxID, 9);
  for (i = 0; i < maxID; i++) { // We save YLab ZLab Omega GrainRadius SpotID
                                // RingNumber Eta Theta Radius
    allSpotsYZO[i][0] = AllSpots[i * 14 + 0];
    allSpotsYZO[i][1] = AllSpots[i * 14 + 1];
    allSpotsYZO[i][2] = AllSpots[i * 14 + 2];
    allSpotsYZO[i][3] = AllSpots[i * 14 + 3];
    allSpotsYZO[i][4] = AllSpots[i * 14 + 4];
    allSpotsYZO[i][5] = AllSpots[i * 14 + 5];
    allSpotsYZO[i][6] = AllSpots[i * 14 + 6];
    allSpotsYZO[i][7] = AllSpots[i * 14 + 7] / 2;
    allSpotsYZO[i][8] = CalcRad(allSpotsYZO[i][0], allSpotsYZO[i][1]);
  }
  int tc2 = munmap(AllSpots, size);

  // Read orientations from grains file
  FILE *grainsF;
  grainsF = fopen(GrainsFN, "r");
  int nGrains, grainNr = 0;
  fgets(aline, 4096, grainsF);
  sscanf(aline, "%s %d", dummy, &nGrains);
  printf("nGrains: %d\n", nGrains);
  double OrientMatrix[nGrains * 9];
  while (fgets(aline, 4096, grainsF) != NULL) {
    if (aline[0] == '%')
      continue;
    sscanf(aline, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf", dummy,
           &OrientMatrix[grainNr * 9 + 0], &OrientMatrix[grainNr * 9 + 1],
           &OrientMatrix[grainNr * 9 + 2], &OrientMatrix[grainNr * 9 + 3],
           &OrientMatrix[grainNr * 9 + 4], &OrientMatrix[grainNr * 9 + 5],
           &OrientMatrix[grainNr * 9 + 6], &OrientMatrix[grainNr * 9 + 7],
           &OrientMatrix[grainNr * 9 + 8]);
    grainNr++;
  }

  // Go through each orientation
  double OM[3][3];
  int nSpots, bestLayer, startRowNr, endRowNr, ringNR, thisRing, layernr,
      ringTemp;
  double **TheorSpots;
  double DisplY, DisplZ, xtr, minDist, EtaMarg, EtaThis, RadThis, OmeThis,
      bestOR[9];
  double gT1, gT2, gT3, gO1, gO2, gO3, ys, zs, Omega, Theta, lenK,
      bestCompleteness;
  double NormGTheors, NormGObs, DotGs, Numers, Denoms, IAThis, bestIA, meanIA,
      bestMeanIA = 180;
  int bestSpotRow, bestSpotsTemp[2 * nhkls], bestSpots[2 * nhkls];
  TheorSpots = allocMatrix(2 * nhkls, 9);
  int nMatches, nMatchesBest = 0, bestYet, nSpotsFin;
  for (i = 0; i < nGrains; i++) {
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
        OM[j][k] = OrientMatrix[i * 9 + j * 3 + k];
    // For this orientation, make spots
    CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmeRanges, hkls, nhkls,
                         BoxSizes, &nSpots, OM, TheorSpots);
    nMatches = 0;
    meanIA = 0;
    bestYet = 0;
    for (j = 0; j < nSpots; j++) {
      DisplacementInTheSpot(x, y, z, Lsd, TheorSpots[j][0], TheorSpots[j][1],
                            TheorSpots[j][2], &DisplY, &DisplZ, &xtr);
      TheorSpots[j][0] += DisplY;
      TheorSpots[j][1] += DisplZ;
      OmeThis = TheorSpots[j][2];
      RadThis = CalcRad(TheorSpots[j][0], TheorSpots[j][1]);      // Radius
      EtaThis = CalcEtaAngle(TheorSpots[j][0], TheorSpots[j][1]); // Eta
      gT1 = TheorSpots[j][3];
      gT2 = TheorSpots[j][4];
      gT3 = TheorSpots[j][5];
      NormGTheors = CalcNorm3(gT1, gT2, gT3);
      ringNR = (int)TheorSpots[j][7];
      for (k = 0; k < nRings; k++) {
        if (ringNR == RingNumbers[k]) {
          thisRing = k;
        }
      }
      minDist = 1000000000;
      bestIA = 180;
      bestSpotRow = -1;
      for (k = 0; k < nLayers; k++) {
        if (fabs(positions[k] - xtr) < minDist) {
          minDist = fabs(positions[k] - xtr);
          bestLayer = k;
        }
      }
      EtaMarg = atand(MargEta / RadThis);
      if (IDsInfo[bestLayer * nRings + thisRing][0] != 0) {
        startRowNr = IDsInfo[bestLayer * nRings + thisRing][2];
        endRowNr = IDsInfo[bestLayer * nRings + thisRing][3];
        if (IDsInfo[bestLayer * nRings + thisRing][1] != ringNR) {
          printf("IDs order did not match with IDHash.\nExiting.\n");
          return 1;
        }
        // We have the row numbers (those start at 1!!!!!), we can now go
        // through our data to find the best match). Need to use MargOme,
        // MargEta, MargRad for initial sorting and then minIA for best Match.
        for (k = startRowNr - 1; k < endRowNr; k++) {
          if (fabs(allSpotsYZO[k][2] - OmeThis) < MargOme) {
            if (fabs(allSpotsYZO[k][6] - EtaThis) < MargEta) {
              if (fabs(allSpotsYZO[k][8] - RadThis) < MargRad) {
                ys = allSpotsYZO[k][0];
                zs = allSpotsYZO[k][1];
                Omega = allSpotsYZO[k][2];
                Theta = allSpotsYZO[k][7];
                lenK = sqrt((Lsd * Lsd) + (ys * ys) + (zs * zs));
                SpotToGv(Lsd / lenK, ys / lenK, zs / lenK, Omega, Theta, &gO1,
                         &gO2, &gO3);
                DotGs = ((gT1 * gO1) + (gT2 * gO2) + (gT3 * gO3));
                NormGObs = CalcNorm3(gO1, gO2, gO3);
                Numers = DotGs;
                Denoms = NormGObs * NormGTheors;
                IAThis = fabs(acosd(Numers / Denoms));
                if (IAThis < bestIA) {
                  bestIA = IAThis;
                  bestSpotRow = k;
                }
              }
            }
          }
        }
        if (bestSpotRow !=
            -1) { // We found the best match. Now save the info needed.
          meanIA += bestIA;
          bestSpotsTemp[nMatches] = bestSpotRow;
          nMatches++;
        }
      }
    }
    if (((double)nMatches) / ((double)nSpots) > Completeness) {
      meanIA /= nMatches;
      if (nMatchesBest < nMatches) {
        bestYet = 1;
      } else if (nMatches >= (nMatchesBest - completenessTol) &&
                 meanIA < bestMeanIA) {
        bestYet = 1;
      }
      if (bestYet == 1) {
        nMatchesBest = nMatches;
        bestMeanIA = meanIA;
        nSpotsFin = nSpots;
        bestCompleteness = ((double)nMatches) / ((double)nSpots);
        for (j = 0; j < nMatches; j++) {
          bestSpots[j] = bestSpotsTemp[j];
        }
        for (j = 0; j < 3; j++)
          for (k = 0; k < 3; k++)
            bestOR[j * 3 + k] = OM[j][k];
      }
    }
  }
  if (nMatchesBest > 0) {
    int GrainID = bestSpots[0];
    char outfilename[4096];
    sprintf(outfilename, "BestPos_%09d.csv",
            posNr); // Cannot be to grain ID, has to be to position number
    FILE *outfile = fopen(outfilename, "w");
    fprintf(outfile, "%d\n", GrainID);
    fprintf(outfile, "%lf, %lf, %lf, %lf, %lf, %lf\n", LatC[0], LatC[1],
            LatC[2], LatC[3], LatC[4], LatC[5]);
    fprintf(outfile, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
            bestCompleteness, bestOR[0], bestOR[1], bestOR[2], bestOR[3],
            bestOR[4], bestOR[5], bestOR[6], bestOR[7], bestOR[8]);
    for (i = 0; i < nMatchesBest; i++) {
      fprintf(outfile, "%d %lf\n", bestSpots[i] + 1,
              allSpotsYZO[bestSpots[i]][3]);
    }
  }
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
