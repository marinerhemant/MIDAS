//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  ProcessGrains.c
//
//
//  Created by Hemant Sharma on 2014/06/24.
//
//  New Features (2014/11/06):
//  - Twins were implemented in a previous version.
//  - Single file reading is implemented now.
//  New Features (2014/11/19):
//  - Strains!!
//

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
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
#include <unistd.h>

#include "MIDAS_Limits.h"
#define NR_MAX_IDS_PER_GRAIN 5000
#define IAColNr 20 // 20 for Internal Angle, 18 for position, 19 for omega
#define EPS 1E-12

inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);
inline double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);
inline void CalcStrainTensorFableBeaudoin(double LatCin[6],
                                          double LatticeParameterFit[6],
                                          double Orient[3][3],
                                          double StrainTensorSample[3][3]);
inline int StrainTensorKenesei(int nspots, double **SpotsInfo, double Distance,
                               double wavelength,
                               double StrainTensorSample[3][3], int **IDHash,
                               double *dspacings, int nRings,
                               int startSpotMatrix, double **SpotMatrix,
                               double *RetVal, double StrainTensorInput[3][3]);
inline void BringDownToFundamentalRegionSym(double QuatIn[4], double QuatOut[4],
                                            int NrSymmetries,
                                            double Sym[24][4]);
inline void BringDownToFundamentalRegion(double QuatIn[4], double QuatOut[4],
                                         int SGNr);

// check() is now provided by MIDAS_Limits.h

static inline int **allocMatrixInt(int nrows, int ncols) {
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

static inline void FreeMemMatrixInt(int **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
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

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = rad2deg * psi;
  Euler[1] = rad2deg * phi;
  Euler[2] = rad2deg * theta;
}

static inline void QuatToOrientMat(double Quat[4], double OrientMat[9]) {
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
  OrientMat[0] = 1 - 2 * (Q2_2 + Q3_2);
  OrientMat[1] = 2 * (Q12 - Q03);
  OrientMat[2] = 2 * (Q13 + Q02);
  OrientMat[3] = 2 * (Q12 + Q03);
  OrientMat[4] = 1 - 2 * (Q1_2 + Q3_2);
  OrientMat[5] = 2 * (Q23 - Q01);
  OrientMat[6] = 2 * (Q13 - Q02);
  OrientMat[7] = 2 * (Q23 + Q01);
  OrientMat[8] = 1 - 2 * (Q1_2 + Q2_2);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: ProcessGrains ParameterFile NrPoints\n");
    return 0;
  }
  clock_t start, end;
  double diftotal;
  start = clock();
  char BestFileName[1024], ResultFileName[1024];
  FILE *BestFile, *ResultFile;
  char line[5024];
  char *ParamFN;
  FILE *fileParam;
  ParamFN = argv[1];
  char aline[1000];
  fileParam = fopen(ParamFN, "r");
  char *str, dummy[1000];
  int LowNr;
  int SGNr;
  char OutDirPath[4096];
  double Distance, wavelength, LatCin[6];
  double BeamThickness = 0, GlobalPosition = 0;
  int NumPhases = 1, PhaseNr = 1;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &wavelength);
      continue;
    }
    str = "BeamThickness ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &BeamThickness);
      continue;
    }
    str = "GlobalPosition ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &GlobalPosition);
      continue;
    }
    str = "NumPhases ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &NumPhases);
      continue;
    }
    str = "PhaseNr ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &PhaseNr);
      continue;
    }
    str = "SpaceGroup ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &SGNr);
      continue;
    }
    str = "Lsd ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Distance);
      continue;
    }
    str = "Wavelength ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &wavelength);
      continue;
    }
    str = "OutDirPath ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, OutDirPath);
      continue;
    }
    str = "LatticeConstant ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
      continue;
    }
    str = "LatticeParameter ";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatCin[0], &LatCin[1],
             &LatCin[2], &LatCin[3], &LatCin[4], &LatCin[5]);
      continue;
    }
  }

  char fnkey[1024], fnopfit[1024], fnprocesskey[1024], fnfullinfo[4096];
  sprintf(fnkey, "%s/Key.bin", OutDirPath);
  sprintf(fnopfit, "%s/OrientPosFit.bin", OutDirPath);
  sprintf(fnprocesskey, "%s/Key.bin", OutDirPath);
  sprintf(fnfullinfo, "%s/FitBest.bin", OutDirPath);
  int fullInfoFile = open(fnfullinfo, O_RDONLY);
  FILE *fileKey = fopen(fnkey, "r");
  FILE *fileOPFit = fopen(fnopfit, "r");
  FILE *fileProcessKey = fopen(fnprocesskey, "r");
  if (fileKey == NULL) {
    printf("Key file was not found. Exiting.\n");
    return 1;
  }
  if (fileOPFit == NULL) {
    printf("OrientPos file was not found. Exiting.\n");
    return 1;
  }
  if (fileProcessKey == NULL) {
    printf("ProcessKey file was not found. Exiting.\n");
    return 1;
  }
  int i, j, k;
  int *IDs;
  int nrIDs = atoi(argv[2]);
  int *keyID;
  keyID = malloc(2 * sizeof(*keyID));
  double *OPThis;
  OPThis = malloc(27 * sizeof(*OPThis));
  char cmmd[4096];
  sprintf(cmmd, "cp %s/ExtraInfo.bin /dev/shm/", OutDirPath);
  system(cmmd);
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
  double *dummySampleInfo;
  dummySampleInfo =
      malloc(22 * NR_MAX_IDS_PER_GRAIN * sizeof(*dummySampleInfo));
  int OffSt, ReadSize;

  // Read IDsHash.csv
  int **IDHash;
  IDHash = allocMatrixInt(NR_MAX_IDS_PER_GRAIN * 2, 3);
  double dspacings[NR_MAX_IDS_PER_GRAIN * 2];
  FILE *idsfile;
  char idsfn[4096];
  sprintf(idsfn, "%s/IDsHash.csv", OutDirPath);
  idsfile = fopen(idsfn, "r");
  int count = 0;
  while (fgets(aline, 4096, idsfile) != NULL) {
    sscanf(aline, "%s %d %d %d", dummy, &IDHash[count][0], &IDHash[count][1],
           &IDHash[count][2]);
    count++;
  }
  int nRings = count;
  fclose(idsfile);
  FILE *hklf = fopen("hkls.csv", "r");
  char aline2[2048];
  double ds;
  int rnr;
  while (fgets(aline2, 2000, hklf) != NULL) {
    sscanf(aline2, "%s %s %s %lf %d %s %s %s %s %s %s", dummy, dummy, dummy,
           &ds, &rnr, dummy, dummy, dummy, dummy, dummy, dummy);
    for (i = 0; i < nRings; i++) {
      if (IDHash[i][0] == rnr) {
        dspacings[i] = ds;
      }
    }
  }

  double **SpotMatrix, **InputMatrix;
  SpotMatrix = allocMatrix(NR_MAX_IDS_PER_GRAIN, 12);
  int counterSpotMatrix = 0, startSpotMatrix, rowSpotID;
  double RetVal;
  double **SpotsInfo;
  SpotsInfo = allocMatrix(NR_MAX_IDS_PER_GRAIN, 8);
  int nspots;
  double LatticeParameterFit[6], Orient[3][3];
  double StrainTensorSampleKen[3][3];
  double StrainTensorSampleFab[3][3];
  double MultR = 1000000;
  double **FinalMatrix;
  double Eul[3], q1[4], q2[4], OR1[9];
  double BeamCenter = 0, FullVol = 0, VNorm;
  FinalMatrix = allocMatrix(nrIDs, 47);
  for (i = 0; i < nrIDs; i++) {
    for (j = 0; j < 47; j++)
      FinalMatrix[i][j] = 0;
  }
  for (j = 0; j < NR_MAX_IDS_PER_GRAIN; j++)
    for (k = 0; k < 12; k++)
      SpotMatrix[j][k] = 0;
  FILE *spotsfile = fopen("SpotMatrix.csv", "w");
  fprintf(spotsfile, "%%"
                     "GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
                     "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n");
  for (i = 0; i < nrIDs; i++) {
    if (i % 1000 == 0)
      printf("Processed point %d of %d.\n", i, nrIDs);
    fread(keyID, 2 * sizeof(int), 1, fileKey);
    fread(OPThis, 27 * sizeof(double), 1, fileOPFit);
    if (keyID[1] == 0)
      continue;
    OffSt = i * 22 * NR_MAX_IDS_PER_GRAIN * sizeof(double);
    ReadSize = 22 * keyID[1] * sizeof(double);
    pread(fullInfoFile, dummySampleInfo, ReadSize, OffSt);
    nspots = keyID[1];
    // Now we have all the info, calculate strains and be done.
    counterSpotMatrix = 0;
    for (j = 0; j < nspots; j++) {
      SpotsInfo[j][0] = dummySampleInfo[j * 22 + 4];
      SpotsInfo[j][1] = dummySampleInfo[j * 22 + 5];
      SpotsInfo[j][2] = dummySampleInfo[j * 22 + 6];
      SpotsInfo[j][3] = dummySampleInfo[j * 22 + 1];
      SpotsInfo[j][4] = dummySampleInfo[j * 22 + 2];
      SpotsInfo[j][5] = dummySampleInfo[j * 22 + 7];
      SpotsInfo[j][6] = dummySampleInfo[j * 22 + 8];
      SpotsInfo[j][7] = dummySampleInfo[j * 22 + 0]; // SpotID
      rowSpotID = (int)dummySampleInfo[j * 22 + 0] - 1;
      SpotMatrix[counterSpotMatrix][0] = (double)(i + 1);             // GrainID
      SpotMatrix[counterSpotMatrix][1] = dummySampleInfo[j * 22 + 0]; // SpotID
      SpotMatrix[counterSpotMatrix][2] = AllSpots[rowSpotID * 16 + 2];  // Omega
      SpotMatrix[counterSpotMatrix][3] = AllSpots[rowSpotID * 16 + 11]; // YRaw
      SpotMatrix[counterSpotMatrix][4] = AllSpots[rowSpotID * 16 + 12]; // ZRaw
      SpotMatrix[counterSpotMatrix][5] =
          AllSpots[rowSpotID * 16 + 13];                               // OmeRaw
      SpotMatrix[counterSpotMatrix][6] = AllSpots[rowSpotID * 16 + 6]; // Eta
      SpotMatrix[counterSpotMatrix][7] = AllSpots[rowSpotID * 16 + 5]; // RingNr
      SpotMatrix[counterSpotMatrix][8] = AllSpots[rowSpotID * 16 + 0]; // YLab
      SpotMatrix[counterSpotMatrix][9] = AllSpots[rowSpotID * 16 + 1]; // ZLab
      SpotMatrix[counterSpotMatrix][10] =
          AllSpots[rowSpotID * 16 + 7] / 2.0; // Theta
      counterSpotMatrix++;
    }
    LatticeParameterFit[0] = OPThis[15];
    LatticeParameterFit[1] = OPThis[16];
    LatticeParameterFit[2] = OPThis[17];
    LatticeParameterFit[3] = OPThis[18];
    LatticeParameterFit[4] = OPThis[19];
    LatticeParameterFit[5] = OPThis[20];
    Orient[0][0] = OPThis[1];
    Orient[0][1] = OPThis[2];
    Orient[0][2] = OPThis[3];
    Orient[1][0] = OPThis[4];
    Orient[1][1] = OPThis[5];
    Orient[1][2] = OPThis[6];
    Orient[2][0] = OPThis[7];
    Orient[2][1] = OPThis[8];
    Orient[2][2] = OPThis[9];
    CalcStrainTensorFableBeaudoin(LatCin, LatticeParameterFit, Orient,
                                  StrainTensorSampleFab);
    int retval = StrainTensorKenesei(
        nspots, SpotsInfo, Distance, wavelength, StrainTensorSampleKen, IDHash,
        dspacings, nRings, 0, SpotMatrix, &RetVal, StrainTensorSampleFab);
    FinalMatrix[i][0] = (double)(i + 1);
    // for (j=0;j<9;j++) FinalMatrix[i][j+1] = OPThis[j+1];
    //  Take orientation and bring down to FR
    for (j = 0; j < 9; j++)
      OR1[j] = OPThis[j + 1];
    OrientMat2Quat(OR1, q1);
    BringDownToFundamentalRegion(q1, q2, SGNr);
    QuatToOrientMat(q2, OR1);
    Orient[0][0] = OR1[0];
    Orient[0][1] = OR1[1];
    Orient[0][2] = OR1[2];
    Orient[1][0] = OR1[3];
    Orient[1][1] = OR1[4];
    Orient[1][2] = OR1[5];
    Orient[2][0] = OR1[6];
    Orient[2][1] = OR1[7];
    Orient[2][2] = OR1[8];
    for (j = 0; j < 9; j++) {
      FinalMatrix[i][j + 1] = OR1[j];
    }
    for (j = 0; j < 3; j++)
      FinalMatrix[i][j + 10] =
          OPThis[j + 11]; // Flip positions due to 180 rotation
    for (j = 0; j < 6; j++)
      FinalMatrix[i][j + 13] = OPThis[j + 15];
    for (j = 0; j < 5; j++)
      FinalMatrix[i][j + 19] = OPThis[j + 22];
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        FinalMatrix[i][24 + 3 * j + k] = MultR * StrainTensorSampleFab[j][k];
        FinalMatrix[i][33 + 3 * j + k] = MultR * StrainTensorSampleKen[j][k];
      }
    }
    FinalMatrix[i][42] = MultR * RetVal;
    FinalMatrix[i][43] = (double)PhaseNr;
    OrientMat2Euler(Orient, Eul);
    FinalMatrix[i][44] = Eul[0];
    FinalMatrix[i][45] = Eul[1];
    FinalMatrix[i][46] = Eul[2];
    VNorm = FinalMatrix[i][22] * FinalMatrix[i][22] * FinalMatrix[i][22];
    BeamCenter += (FinalMatrix[i][12]) * (VNorm);
    FullVol += VNorm;
    for (j = 0; j < counterSpotMatrix; j++) {
      fprintf(spotsfile,
              "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%d\t%lf\t%lf\t%lf\t%lf\n",
              (int)SpotMatrix[j][0], (int)SpotMatrix[j][1], SpotMatrix[j][2],
              SpotMatrix[j][3], SpotMatrix[j][4], SpotMatrix[j][5],
              SpotMatrix[j][6], (int)SpotMatrix[j][7], SpotMatrix[j][8],
              SpotMatrix[j][9], SpotMatrix[j][10], MultR * SpotMatrix[j][11]);
    }
  }
  int tc2 = munmap(AllSpots, size);
  char GrainsFileName[1024];
  sprintf(GrainsFileName, "Grains.csv");
  FILE *GrainsFile;
  GrainsFile = fopen(GrainsFileName, "w");
  printf("Number of points: %d.\n", nrIDs);
  BeamCenter /= FullVol;
  // Write file
  fclose(spotsfile);
  fprintf(GrainsFile, "%%NumGrains %d\n", nrIDs);
  fprintf(GrainsFile, "%%BeamCenter %f\n", BeamCenter);
  fprintf(GrainsFile, "%%BeamThickness %f\n", BeamThickness);
  fprintf(GrainsFile, "%%GlobalPosition %f\n", GlobalPosition);
  fprintf(GrainsFile, "%%NumPhases %d\n", NumPhases);
  fprintf(GrainsFile, "%%PhaseInfo\n%%\tSpaceGroup:%d\n", SGNr);
  fprintf(GrainsFile, "%%\tLattice Parameter: %lf %lf %lf %lf %lf %lf\n",
          LatCin[0], LatCin[1], LatCin[2], LatCin[3], LatCin[4], LatCin[5]);
  fprintf(
      GrainsFile,
      "%%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\ta\tb"
      "\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfi"
      "dence\t");
  fprintf(GrainsFile, "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\t"
                      "eFab32\teFab33\t");
  fprintf(GrainsFile,
          "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen"
          "33\tRMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n");
  for (i = 0; i < nrIDs; i++) {
    if (FinalMatrix[i][0] == 0)
      continue;
    fprintf(GrainsFile, "%d\t", (int)FinalMatrix[i][0]);
    for (j = 1; j < 47; j++) {
      fprintf(GrainsFile, "%lf\t", FinalMatrix[i][j]);
    }
    fprintf(GrainsFile, "\n");
  }
  end = clock();
  diftotal = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time elapsed: %f s.\n", diftotal);
  return 0;
}
