//
// Mic2GrainsList.c
//
// Generate a list of unique orientations from Mic file (output of ParseMic.c)
// Output format matches Grains.csv expected by GenSeedOrientationsFF2NFHEDM.c
//
// Created by Hemant Sharma on 2024/02/10
//

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define EPS 1e-9

// External Functions from GetMisorientation.c
extern double GetMisOrientation(double quat1[4], double quat2[4],
                                double axis[3], double *Angle, int SGNr);
extern void OrientMat2Quat(double OrientMat[9], double Quat[4]);
extern void Euler2OrientMat(double Euler[3], double m_out[3][3]);

// Data Structures
typedef struct {
  double eul[3];
  double confidence;
  double orientMat[9]; // Storing as 1D array of 9 doubles
  bool used;
} GrainInfo;

// Specific sorting
int compare_grains(const void *a, const void *b) {
  const GrainInfo *ga = (const GrainInfo *)a;
  const GrainInfo *gb = (const GrainInfo *)b;
  if (ga->confidence < gb->confidence)
    return 1;
  if (ga->confidence > gb->confidence)
    return -1;
  return 0;
}

static inline void usage(void) {
  printf("Mic2GrainsList usage: ./Mic2GrainsList <ParamFile> <MicFile> "
         "<OutputFile>\n");
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    usage();
    return 1;
  }

  char *paramFN = argv[1];
  char *micFN = argv[2];
  char *outFN = argv[3];

  // Default parameters
  int sgNr = 225; // Default to cubic if not found
  double maxAng = 1.0;
  double minConf = 0.04; // Default based on user example
  double latP[6] = {0};

  // Parse Parameter File
  FILE *pParam = fopen(paramFN, "r");
  if (!pParam) {
    printf("Error opening parameter file: %s\n", paramFN);
    return 1;
  }

  char aline[1024];
  char dummy[1024];
  while (fgets(aline, 1024, pParam) != NULL) {
    if (aline[0] == '#')
      continue;

    if (strncmp(aline, "LatticeParameter", 16) == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &latP[0], &latP[1],
             &latP[2], &latP[3], &latP[4], &latP[5]);
      continue;
    }
    if (strncmp(aline, "SpaceGroup", 10) == 0) {
      sscanf(aline, "%s %d", dummy, &sgNr);
      continue;
    }
    if (strncmp(aline, "MaxAngle", 8) == 0) { // User mentioned reading MaxAngle
      sscanf(aline, "%s %lf", dummy, &maxAng);
      continue;
    }
    // Check for both MinFracAccept (User example) and MinConfidence (Generic)
    if (strncmp(aline, "MinFracAccept", 13) == 0) {
      sscanf(aline, "%s %lf", dummy, &minConf);
      continue;
    }
    if (strncmp(aline, "MinConfidence", 13) == 0) {
      sscanf(aline, "%s %lf", dummy, &minConf);
      continue;
    }
  }
  fclose(pParam);

  printf("Parameters:\n");
  printf("  MicFile: %s\n", micFN);
  printf("  ParamFile: %s\n", paramFN);
  printf("  Output: %s\n", outFN);
  printf("  SpaceGroup: %d\n", sgNr);
  printf("  MaxAngle: %lf\n", maxAng);
  printf("  MinConfidence: %lf\n", minConf);
  printf("  LatticeParameter: %lf %lf %lf %lf %lf %lf\n", latP[0], latP[1],
         latP[2], latP[3], latP[4], latP[5]);

  FILE *fp = fopen(micFN, "r");
  if (!fp) {
    printf("Error opening input file %s\n", micFN);
    return 1;
  }

  // Estimate number of lines or realloc dynamic
  char line[4096];
  int capacity = 10000;
  int count = 0;
  GrainInfo *grains = malloc(capacity * sizeof(GrainInfo));

  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '%')
      continue;

    // Parse line based on ParseMic format
    // Format: OrientationRowNr(0) ID(1) Time(2) X(3) Y(4) Size(5) UD(6) Eul1(7)
    // Eul2(8) Eul3(9) Conf(10) Phase(11)
    double vals[12];
    // Only read up to confidence
    int nRead =
        sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &vals[0],
               &vals[1], &vals[2], &vals[3], &vals[4], &vals[5], &vals[6],
               &vals[7], &vals[8], &vals[9], &vals[10]);

    if (nRead < 11)
      continue; // Skip incomplete lines

    if (vals[10] < minConf)
      continue;

    if (count >= capacity) {
      capacity *= 2;
      grains = realloc(grains, capacity * sizeof(GrainInfo));
    }

    grains[count].eul[0] = vals[7];
    grains[count].eul[1] = vals[8];
    grains[count].eul[2] = vals[9];
    grains[count].confidence = vals[10];
    grains[count].used = false;

    Euler2OrientMat(grains[count].eul, (double (*)[3])grains[count].orientMat);

    count++;
  }
  fclose(fp);

  printf("Read %d valid orientations (confidence >= %lf)\n", count, minConf);

  // Sort by confidence
  qsort(grains, count, sizeof(GrainInfo), compare_grains);

  // Find unique
  FILE *fout = fopen(outFN, "w");
  if (!fout) {
    printf("Error opening output file %s\n", outFN);
    free(grains);
    return 1;
  }

  // Write header
  fprintf(fout, "%%GrainID OrientMat(9) dummies LatC(6) dummies\n");

  int grainID = 1;
  double quat1[4], quat2[4], axis[3], ang;
  int nUnique = 0;

  for (int i = 0; i < count; i++) {
    if (grains[i].used)
      continue;

    OrientMat2Quat(grains[i].orientMat, quat1);

    fprintf(fout,
            "%d %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf "
            "0 0 0 %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf 0 0 0 0\n",
            grainID, grains[i].orientMat[0], grains[i].orientMat[1],
            grains[i].orientMat[2], grains[i].orientMat[3],
            grains[i].orientMat[4], grains[i].orientMat[5],
            grains[i].orientMat[6], grains[i].orientMat[7],
            grains[i].orientMat[8], latP[0], latP[1], latP[2], latP[3], latP[4],
            latP[5]);

    grainID++;
    nUnique++;
    grains[i].used = true;

    // Mark duplicates
    for (int j = i + 1; j < count; j++) {
      if (grains[j].used)
        continue;

      OrientMat2Quat(grains[j].orientMat, quat2);
      GetMisOrientation(quat1, quat2, axis, &ang, sgNr);

      if (ang < maxAng) {
        grains[j].used = true; // Mark as merged
      }
    }
  }

  fclose(fout);
  free(grains);

  printf("Values written: %d unique grains found.\n", nUnique);

  return 0;
}
