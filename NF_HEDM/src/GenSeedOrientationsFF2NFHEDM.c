//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
// GenSeedOrientationsFF2NFHEDM.c
//
// Generate seed orientations for NF_HEDM using orientations from FF_HEDM
//
// Created by Hemant Sharma on 2014/07/21.
//

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// External Functions from GetMisorientation.c with inline keyword
extern void OrientMat2Quat(double OrientMat[9], double Quat[4]);

static inline void usage(void) {
  printf("GenSeedOrientations usage: ./GenSeedOrientations <OringinalFile>"
         " <OutputFile>. Please provide full path.\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    usage();
    return 1;
  }
  clock_t start0, end;
  double diftotal;
  start0 = clock();
  char *outfilename;
  outfilename = argv[2];
  char *GrainFN;
  GrainFN = argv[1];
  char aline[1000];
  double OrientMatrix[9], Quaternion[4], LatC[6];
  char dummy[1024];
  FILE *GrainFile = fopen(GrainFN, "r");
  FILE *OutFile = fopen(outfilename, "w");
  int nOrientations = 0;
  int GrainID;
  while (fgets(aline, 1000, GrainFile) != NULL) {
    if (aline[0] == '%')
      continue;
    sscanf(aline,
           "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %lf %lf %lf %lf "
           "%lf %lf %s %s %s %s",
           &GrainID, &OrientMatrix[0], &OrientMatrix[1], &OrientMatrix[2],
           &OrientMatrix[3], &OrientMatrix[4], &OrientMatrix[5],
           &OrientMatrix[6], &OrientMatrix[7], &OrientMatrix[8], dummy, dummy,
           dummy, &LatC[0], &LatC[1], &LatC[2], &LatC[3], &LatC[4], &LatC[5],
           dummy, dummy, dummy, dummy);
    OrientMat2Quat(OrientMatrix, Quaternion);
    fprintf(OutFile, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n", Quaternion[0],
            Quaternion[1], Quaternion[2], Quaternion[3], LatC[0], LatC[1],
            LatC[2], LatC[3], LatC[4], LatC[5], GrainID);
    nOrientations++;
  }
  printf("Number of seed orientations: %d\n", nOrientations);
  fclose(GrainFile);
  end = clock();
  diftotal = ((double)(end - start0)) / CLOCKS_PER_SEC;
  printf("Time elapsed in making diffraction spots: %f [s]\n", diftotal);
  return 0;
}
