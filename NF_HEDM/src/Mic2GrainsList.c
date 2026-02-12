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
         "<OutputFile> [DoNeighborSearch (0/1, default 0)]\n");
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
  if (argc < 4) {
    usage();
    return 1;
  }

  char *paramFN = argv[1];
  char *micFN = argv[2];
  char *outFN = argv[3];
  int doNeighborSearch = 0;
  if (argc >= 5) {
    doNeighborSearch = atoi(argv[4]);
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

    Euler2OrientMat(grains[count].eul, (double (*)[3])grains[count].orientMat);

    count++;
  }
  fclose(fp);

  printf("Read %d valid orientations (confidence >= %lf)\n", count, minConf);

  // Sort by confidence
  qsort(grains, count, sizeof(GrainInfo), compare_grains);

  FILE *fout = fopen(outFN, "w");
  if (!fout) {
    printf("Error opening output file %s\n", outFN);
    free(grains);
    return 1;
  }
  fprintf(fout, "%%GrainID OrientMat(9) dummies LatC(6) x y z dummies\n");

  int grainID = 1;
  double quat1[4], quat2[4], axis[3], ang;
  int nUnique = 0;

  if (doNeighborSearch == 0) {
    // --- Traditional Global Merge ---
    for (int i = 0; i < count; i++) {
      if (grains[i].used)
        continue;

      OrientMat2Quat(grains[i].orientMat, quat1);

      fprintf(
          fout,
          "%d %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf "
          "0 0 0 %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf 0 0\n",
          grainID, grains[i].orientMat[0], grains[i].orientMat[1],
          grains[i].orientMat[2], grains[i].orientMat[3],
          grains[i].orientMat[4], grains[i].orientMat[5],
          grains[i].orientMat[6], grains[i].orientMat[7],
          grains[i].orientMat[8], latP[0], latP[1], latP[2], latP[3], latP[4],
          latP[5], grains[i].x, grains[i].y);

      grainID++;
      nUnique++;
      grains[i].used = true;

      // Mark duplicates globally
      for (int j = i + 1; j < count; j++) {
        if (grains[j].used)
          continue;

        OrientMat2Quat(grains[j].orientMat, quat2);
        GetMisOrientation(quat1, quat2, axis, &ang, sgNr);

        if (ang < maxAng) {
          grains[j].used = true;
        }
      }
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
    // Use slightly larger bin size to catch neighbors
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
      newNode->grainIndex = i; // storing index into *sorted* array
      newNode->next = grid[idx];
      grid[idx] = newNode;
    }

    // 4. BFS Clustering
    double distThresh =
        triEdgeSize *
        2.0; // Neighbor tolerance (slightly generous for diagonals)
    double distThreshSq = distThresh * distThresh;

    Queue *q = createQueue(count);

    for (int i = 0; i < count; i++) {
      if (grains[i].used)
        continue;

      // New Grain Found (Seed)
      OrientMat2Quat(grains[i].orientMat, quat1);

      fprintf(
          fout,
          "%d %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf "
          "0 0 0 %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf 0 0\n",
          grainID, grains[i].orientMat[0], grains[i].orientMat[1],
          grains[i].orientMat[2], grains[i].orientMat[3],
          grains[i].orientMat[4], grains[i].orientMat[5],
          grains[i].orientMat[6], grains[i].orientMat[7],
          grains[i].orientMat[8], latP[0], latP[1], latP[2], latP[3], latP[4],
          latP[5], grains[i].x, grains[i].y);

      grains[i].used = true;
      enqueue(q, i);

      // Flood fill orientation
      while (!isQueueEmpty(q)) {
        int currIdx = dequeue(q);
        GrainInfo *curr = &grains[currIdx];

        // Determine search range in grid (3x3 neighborhood)
        int bx = (int)((curr->x - minX) / binSize);
        int by = (int)((curr->y - minY) / binSize);

        for (int ny = by - 1; ny <= by + 1; ny++) {
          for (int nx = bx - 1; nx <= bx + 1; nx++) {
            if (nx < 0 || nx >= dimX || ny < 0 || ny >= dimY)
              continue;

            // Check all points in this bin
            Node *scan = grid[ny * dimX + nx];
            while (scan) {
              int neighborIdx = scan->grainIndex;
              if (!grains[neighborIdx].used) {
                // Check Spatial Distance
                double dx = grains[neighborIdx].x - curr->x;
                double dy = grains[neighborIdx].y - curr->y;
                if (dx * dx + dy * dy < distThreshSq) {
                  // Check Misorientation relative to SEED (i) or CURRENT
                  // (currIdx)? Standard region growing usually checks relative
                  // to SEED to prevent drift, or CURRENT for connectivity.
                  // Given HEDM context, we want Grain Average. Since we printed
                  // the Seed orientation as the grain orientation, we should
                  // check if neighbor matches the Seed.

                  OrientMat2Quat(grains[neighborIdx].orientMat, quat2);
                  GetMisOrientation(quat1, quat2, axis, &ang, sgNr);

                  if (ang < maxAng) {
                    grains[neighborIdx].used = true;
                    enqueue(q, neighborIdx);
                  }
                }
              }
              scan = scan->next;
            }
          }
        }
      }

      grainID++;
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

  fclose(fout);
  free(grains);

  printf("Values written: %d unique grains found.\n", nUnique);

  return 0;
}
