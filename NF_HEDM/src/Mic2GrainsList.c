//
// Mic2GrainsList.c
//
// Generate a list of unique orientations from Mic file (output of ParseMic.c)
// Output format matches Grains.csv expected by GenSeedOrientationsFF2NFHEDM.c
//
// Modified to optionaly support Spatial Connectivity (Region Growing)
//
// Created by Hemant Sharma on 2024/02/10
//

#include "midas_version.h"
#include <ctype.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
  double x, y;
  int originalIndex;
  bool used;
} GrainInfo;

// Linked List for Spatial Bin
typedef struct Node {
  int grainIndex;
  struct Node *next;
} Node;

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
         "<OutputFile> [DoNeighborSearch (0/1, default 0)] [nCPUs (default "
         "all)] [MinConfidence (overrides param file)]\n");
}

// Queue for BFS
typedef struct {
  int *data;
  int front;
  int rear;
  int capacity;
} Queue;

Queue *createQueue(int capacity) {
  Queue *q = (Queue *)malloc(sizeof(Queue));
  q->capacity = capacity;
  q->data = (int *)malloc(capacity * sizeof(int));
  q->front = 0;
  q->rear = -1;
  return q;
}

void enqueue(Queue *q, int item) {
  if (q->rear >= q->capacity - 1)
    return; // Should not happen with proper sizing
  q->data[++q->rear] = item;
}

int dequeue(Queue *q) {
  if (q->front > q->rear)
    return -1;
  return q->data[q->front++];
}

bool isQueueEmpty(Queue *q) { return q->front > q->rear; }

void freeQueue(Queue *q) {
  free(q->data);
  free(q);
}

int main(int argc, char *argv[]) {
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  if (argc < 4) {
    usage();
    return 1;
  }

  char *paramFN = argv[1];
  char *micFN = argv[2];
  char *outFN = argv[3];
  int doNeighborSearch = 0;
  int nCPUs = omp_get_max_threads();
  if (argc >= 5) {
    doNeighborSearch = atoi(argv[4]);
  }
  if (argc >= 6) {
    nCPUs = atoi(argv[5]);
  }
  omp_set_num_threads(nCPUs);

  // Command-line MinConfidence override (applied after param file parsing)
  int hasMinConfOverride = 0;
  double minConfOverride = 0.0;
  if (argc >= 7) {
    hasMinConfOverride = 1;
    minConfOverride = atof(argv[6]);
  }

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
    if (strncmp(aline, "MaxAngle", 8) == 0) {
      sscanf(aline, "%s %lf", dummy, &maxAng);
      continue;
    }
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

  // Apply command-line MinConfidence override if given
  if (hasMinConfOverride) {
    printf("  MinConfidence override from command line: %lf -> %lf\n", minConf,
           minConfOverride);
    minConf = minConfOverride;
  }

  printf("Parameters:\n");
  printf("  MicFile: %s\n", micFN);
  printf("  ParamFile: %s\n", paramFN);
  printf("  Output: %s\n", outFN);
  printf("  SpaceGroup: %d\n", sgNr);
  printf("  MaxAngle: %lf\n", maxAng);
  printf("  MinConfidence: %lf\n", minConf);
  printf("  LatticeParameter: %lf %lf %lf %lf %lf %lf\n", latP[0], latP[1],
         latP[2], latP[3], latP[4], latP[5]);
  printf("  DoNeighborSearch: %d\n", doNeighborSearch);
  printf("  nCPUs: %d\n", nCPUs);

  FILE *fp = fopen(micFN, "r");
  if (!fp) {
    printf("Error opening input file %s\n", micFN);
    return 1;
  }

  // Header Parsing for TriEdgeSize
  double triEdgeSize = 0.0;
  char line[4096];

  // Read first few header lines
  long fpos_data = 0;
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '%') {
      if (strncmp(line, "%TriEdgeSize", 12) == 0) {
        sscanf(line, "%*s %lf", &triEdgeSize);
      }
      fpos_data = ftell(fp); // Mark position after last header
      continue;
    }
    // First non-% line is data, rewind to it
    fseek(fp, fpos_data, SEEK_SET);
    break;
  }

  if (doNeighborSearch && triEdgeSize <= 0.000001) {
    printf("Warning: Neighbor search requested but TriEdgeSize not found or "
           "invalid in header. Defaulting to 0 neighbor search.\n");
    doNeighborSearch = 0;
  }

  if (doNeighborSearch) {
    printf("  TriEdgeSize: %lf\n", triEdgeSize);
  }

  // Estimate number of lines or realloc dynamic
  int capacity = 10000;
  int count = 0;
  GrainInfo *grains = malloc(capacity * sizeof(GrainInfo));

  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '%')
      continue;

    double vals[12];
    int nRead =
        sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &vals[0],
               &vals[1], &vals[2], &vals[3], &vals[4], &vals[5], &vals[6],
               &vals[7], &vals[8], &vals[9], &vals[10]);

    if (nRead < 11)
      continue;

    if (vals[10] < minConf)
      continue;

    if (count >= capacity) {
      capacity *= 2;
      grains = realloc(grains, capacity * sizeof(GrainInfo));
    }

    grains[count].x = vals[3];
    grains[count].y = vals[4];
    grains[count].eul[0] = vals[7] * rad2deg;
    grains[count].eul[1] = vals[8] * rad2deg;
    grains[count].eul[2] = vals[9] * rad2deg;
    grains[count].confidence = vals[10];
    grains[count].used = false;
    grains[count].originalIndex = count;
    count++;
  }
  fclose(fp);

  // Parallel Euler -> OrientMat conversion
#pragma omp parallel for schedule(static)
  for (int k = 0; k < count; k++) {
    Euler2OrientMat(grains[k].eul, (double (*)[3])grains[k].orientMat);
  }

  printf("Read %d valid orientations (confidence >= %lf)\n", count, minConf);

  // Sort by confidence
  qsort(grains, count, sizeof(GrainInfo), compare_grains);

  // --- Buffer unique grains so we can write %NumGrains header ---
  // Each unique grain entry: orientMat[9], x, y, nVoxels
  typedef struct {
    double orientMat[9];
    double x, y;
    int nVoxels; // number of mic voxels merged into this grain
  } UniqueGrain;

  int grainID = 1;
  double quat1[4], quat2[4], axis[3], ang;
  int nUnique = 0;
  int uniqueCapacity = 1000;
  UniqueGrain *uniqueGrains = malloc(uniqueCapacity * sizeof(UniqueGrain));

  if (doNeighborSearch == 0) {
    // --- Traditional Global Merge (OpenMP-parallel inner loop) ---
    for (int i = 0; i < count; i++) {
      if (grains[i].used)
        continue;

      OrientMat2Quat(grains[i].orientMat, quat1);

      // Count merged voxels (this grain + all duplicates)
      int nVox = 1;
      grains[i].used = true;

      // Mark duplicates globally (parallel) and count them
      int nDups = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+:nDups)
      for (int j = i + 1; j < count; j++) {
        if (grains[j].used)
          continue;

        double q2[4], ax[3], a;
        OrientMat2Quat(grains[j].orientMat, q2);
        GetMisOrientation(quat1, q2, ax, &a, sgNr);

        if (a < maxAng) {
          grains[j].used = true;
          nDups++;
        }
      }
      nVox += nDups;

      // Buffer the unique grain
      if (nUnique >= uniqueCapacity) {
        uniqueCapacity *= 2;
        uniqueGrains = realloc(uniqueGrains, uniqueCapacity * sizeof(UniqueGrain));
      }
      memcpy(uniqueGrains[nUnique].orientMat, grains[i].orientMat, 9 * sizeof(double));
      uniqueGrains[nUnique].x = grains[i].x;
      uniqueGrains[nUnique].y = grains[i].y;
      uniqueGrains[nUnique].nVoxels = nVox;
      nUnique++;
    }
  } else {
    // --- Spatial Neighbor Search (Region Growing) ---

    // 1. Determine Grid Bounds
    double minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (int i = 0; i < count; i++) {
      if (grains[i].x < minX)
        minX = grains[i].x;
      if (grains[i].x > maxX)
        maxX = grains[i].x;
      if (grains[i].y < minY)
        minY = grains[i].y;
      if (grains[i].y > maxY)
        maxY = grains[i].y;
    }

    // 2. Set up Grid
    double binSize = triEdgeSize * 1.01;
    int dimX = (int)((maxX - minX) / binSize) + 2;
    int dimY = (int)((maxY - minY) / binSize) + 2;

    printf("Building Spatial Index: Grid %d x %d\n", dimX, dimY);

    Node **grid = (Node **)calloc(dimX * dimY, sizeof(Node *));

    // 3. Populate Grid
    for (int i = 0; i < count; i++) {
      int bx = (int)((grains[i].x - minX) / binSize);
      int by = (int)((grains[i].y - minY) / binSize);
      int idx = by * dimX + bx;

      Node *newNode = (Node *)malloc(sizeof(Node));
      newNode->grainIndex = i;
      newNode->next = grid[idx];
      grid[idx] = newNode;
    }

    // 4. BFS Clustering
    double distThresh = triEdgeSize * 2.0;
    double distThreshSq = distThresh * distThresh;

    Queue *q = createQueue(count);

    for (int i = 0; i < count; i++) {
      if (grains[i].used)
        continue;

      // New Grain Found (Seed)
      OrientMat2Quat(grains[i].orientMat, quat1);

      grains[i].used = true;
      enqueue(q, i);
      int nVox = 1;

      // Flood fill orientation
      while (!isQueueEmpty(q)) {
        int currIdx = dequeue(q);
        GrainInfo *curr = &grains[currIdx];

        int bx = (int)((curr->x - minX) / binSize);
        int by = (int)((curr->y - minY) / binSize);

        for (int ny = by - 1; ny <= by + 1; ny++) {
          for (int nx = bx - 1; nx <= bx + 1; nx++) {
            if (nx < 0 || nx >= dimX || ny < 0 || ny >= dimY)
              continue;

            Node *scan = grid[ny * dimX + nx];
            while (scan) {
              int neighborIdx = scan->grainIndex;
              if (!grains[neighborIdx].used) {
                double dx = grains[neighborIdx].x - curr->x;
                double dy = grains[neighborIdx].y - curr->y;
                if (dx * dx + dy * dy < distThreshSq) {
                  OrientMat2Quat(grains[neighborIdx].orientMat, quat2);
                  GetMisOrientation(quat1, quat2, axis, &ang, sgNr);

                  if (ang < maxAng) {
                    grains[neighborIdx].used = true;
                    enqueue(q, neighborIdx);
                    nVox++;
                  }
                }
              }
              scan = scan->next;
            }
          }
        }
      }

      // Buffer the unique grain
      if (nUnique >= uniqueCapacity) {
        uniqueCapacity *= 2;
        uniqueGrains = realloc(uniqueGrains, uniqueCapacity * sizeof(UniqueGrain));
      }
      memcpy(uniqueGrains[nUnique].orientMat, grains[i].orientMat, 9 * sizeof(double));
      uniqueGrains[nUnique].x = grains[i].x;
      uniqueGrains[nUnique].y = grains[i].y;
      uniqueGrains[nUnique].nVoxels = nVox;
      nUnique++;
    }

    // Cleanup
    freeQueue(q);
    for (int k = 0; k < dimX * dimY; k++) {
      Node *scan = grid[k];
      while (scan) {
        Node *temp = scan;
        scan = scan->next;
        free(temp);
      }
    }
    free(grid);
  }

  // --- Write output with 9-line header ---
  FILE *fout = fopen(outFN, "w");
  if (!fout) {
    printf("Error opening output file %s\n", outFN);
    free(grains);
    free(uniqueGrains);
    return 1;
  }

  // 9-line header (IndexerOMP skips line 1 for nGrains, then 8 more)
  fprintf(fout, "%%NumGrains %d\n", nUnique);
  fprintf(fout, "%%GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence\n");
  fprintf(fout, "%%Generated by Mic2GrainsList from %s\n", micFN);
  fprintf(fout, "%%SpaceGroup %d\n", sgNr);
  fprintf(fout, "%%MaxAngle %.4lf\n", maxAng);
  fprintf(fout, "%%MinConfidence %.4lf\n", minConf);
  fprintf(fout, "%%LatticeParameter %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf\n",
          latP[0], latP[1], latP[2], latP[3], latP[4], latP[5]);
  fprintf(fout, "%%DoNeighborSearch %d\n", doNeighborSearch);
  fprintf(fout, "%%TriEdgeSize %.6lf\n", triEdgeSize);

  // Voxel area for radius computation: equilateral triangle = side^2 * sqrt(3)/4
  double voxelArea = triEdgeSize * triEdgeSize * sqrt(3.0) / 4.0;

  // Write data rows: ID O(9) X Y 0 LatC(6) 0 0 0 radius 1
  for (int i = 0; i < nUnique; i++) {
    double area = uniqueGrains[i].nVoxels * voxelArea;
    double radius = sqrt(area / M_PI);
    fprintf(fout,
            "%d %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf "
            "%.6lf %.6lf 0 "
            "%.6lf %.6lf %.6lf %.6lf %.6lf %.6lf "
            "0 0 0 %.6lf 1\n",
            i + 1,
            uniqueGrains[i].orientMat[0], uniqueGrains[i].orientMat[1],
            uniqueGrains[i].orientMat[2], uniqueGrains[i].orientMat[3],
            uniqueGrains[i].orientMat[4], uniqueGrains[i].orientMat[5],
            uniqueGrains[i].orientMat[6], uniqueGrains[i].orientMat[7],
            uniqueGrains[i].orientMat[8],
            uniqueGrains[i].x, uniqueGrains[i].y,
            latP[0], latP[1], latP[2], latP[3], latP[4], latP[5],
            radius);
  }

  fclose(fout);
  free(uniqueGrains);
  free(grains);

  printf("Values written: %d unique grains found.\n", nUnique);

  return 0;
}

