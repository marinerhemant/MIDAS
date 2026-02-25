/**
 * findSingleSolutionPFRefactored.c - Finding Single Solution in PF-HEDM
 *
 * Purpose: This program identifies unique crystal orientations in
 * polycrystalline materials using far-field high-energy diffraction microscopy
 * (FF-HEDM) data. It processes data from multiple scans and voxels to find the
 * most probable grain orientations, then generates sinograms for visualization
 * and further analysis.
 *
 * Workflow:
 * 1. Process each voxel to find the best orientation solution
 * 2. Identify unique orientations across all voxels
 * 3. Process detected spots and associate them with grains
 * 4. Generate sinograms for visualization
 *
 * Command line usage:
 *   findSingleSolutionPF folderName sgNum maxAngle nScans nCPUs tolOme tolEta
 *
 * Author: Hemant Sharma
 * Improved version with better structure, memory management, error handling,
 * and performance.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
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

/* Zarr/Blosc/Zip includes for patch extraction */
#include <blosc2.h>
#include <zip.h>

/* Error codes for better error management */
#define SUCCESS 0
#define ERR_MEMORY_ALLOC -1
#define ERR_FILE_OPEN -2
#define ERR_FILE_READ -3
#define ERR_FILE_WRITE -4
#define ERR_INVALID_INPUT -5
#define ERR_PROCESSING -6

/* Constants for array dimensions and buffer sizes */
#define MAX_N_SOLUTIONS_PER_VOX                                                \
  1000000 /* Maximum number of solutions per voxel */
#define MAX_N_SPOTS_PER_GRAIN                                                  \
  5000 /* Maximum number of diffraction spots per grain */
#define MAX_N_SPOTS_TOTAL                                                      \
  100000000               /* Maximum total number of spots across all grains */
#define MAX_PATH_LEN 2048 /* Maximum length for file paths */

#define INVALID_VOX ((size_t)-1)

/* Constants for array column counts */
#define SPOTS_ARRAY_COLS                                                       \
  10 /* Number of columns in spots array:                                      \
        [x,y,omega,intensity,spotID,ringNum,eta,theta,dspacing,scanNum] */
#define KEY_ARRAY_COLS                                                         \
  4 /* Number of columns in key array:                                         \
       [grainID,nSpots,startRowNr,spotListStartPos] */
#define ORIENT_ARRAY_COLS                                                      \
  9 /* Number of columns in orientation matrix: [9 parameters of 3x3           \
       orientation matrix] */
#define CONF_IA_ARRAY_COLS                                                     \
  2 /* Number of columns in confidence/internal angle array:                   \
       [confidence,internalAngle] */
#define TMP_ARRAY_COLS 16 /* Number of columns in temporary data array */

/**
 * Structure to hold information about a diffraction spot
 *
 * Stores angular coordinates, spot metadata, and grain association
 */
typedef struct {
  double omega; /* Rotation angle of sample during measurement (radians) */
  double eta;   /* Azimuthal angle on detector (radians) */
  int ringNr; /* Diffraction ring number (related to crystallographic plane) */
  int mergedID; /* Unique identifier for the spot across all measurements */
  int scanNr;   /* Scan number in which the spot was detected */
  int grainNr;  /* Grain number to which this spot is assigned */
  int spotNr;   /* Index of this spot within its grain */
} SpotData;

/**
 * Structure used for sorting sinogram data by angle
 */
typedef struct {
  double *intensities; /* Array of intensity values across scans */
  double angle;        /* Angle value used for sorting */
} SinoSortData;

/**
 * Structure to hold the result of unique orientations processing
 */
typedef struct {
  size_t nUniques;         /* Number of unique orientations found */
  size_t *uniqueKeyArr;    /* Array of keys for unique orientations */
  double *uniqueOrientArr; /* Array of orientation matrices for unique
                              orientations */
} UniqueOrientationsResult;

/**
 * Structure to hold a list of diffraction spots
 */
typedef struct {
  SpotData *spotData; /* Array of spot data */
  size_t nSpots;      /* Number of spots in the list */
} SpotList;

/**
 * Log an error message to stderr without exiting
 *
 * @param message The format string for the error message
 * @param ... Additional arguments for the format string
 */
static void log_error(const char *message, ...) {
  va_list args;
  va_start(args, message);
  fprintf(stderr, "[ERROR] ");
  vfprintf(stderr, message, args);
  fprintf(stderr, "\n");
  va_end(args);
}

/**
 * Log a fatal error message to stderr and exit the program
 *
 * @param message The format string for the error message
 * @param ... Additional arguments for the format string
 */
static void fatal_error(const char *message, ...) {
  va_list args;
  va_start(args, message);
  fprintf(stderr, "[FATAL] ");
  vfprintf(stderr, message, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(EXIT_FAILURE);
}

/* Function declarations */
void print_usage(const char *program_name);
double *read_memory_mapped_file(const char *filename, size_t *size_out);
int compare_sino_data(const void *a, const void *b);
void process_voxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                   size_t *allKeyArr, double *allOrientationsArr, int ib);
UniqueOrientationsResult find_unique_orientations(size_t *allKeyArr,
                                                  double *allOrientationsArr,
                                                  size_t nScans, int sgNr,
                                                  double maxAng);
SpotList process_spots(UniqueOrientationsResult *uniqueResult,
                       const char *folderName, double *allSpots,
                       size_t nSpotsAll, double tolOme, double tolEta);
void generate_sinograms(SpotList *spotList,
                        UniqueOrientationsResult *uniqueResult,
                        double *allSpots, size_t nSpotsAll, int nScans,
                        double tolOme, double tolEta, const char *outputFolder,
                        int numProcs, int normalizeSino, int absTransform,
                        int *outMaxNHKLs);
void extract_patches(const char *topdir, const char *outputFolder,
                     const char *paramFile, size_t nGrs, int maxNHKLs,
                     int nScans, int numProcs);

/* FrameRequest struct and comparator for qsort (used in extract_patches) */
typedef struct {
  int frameIdx;
  double yCen;
  double zCen;
  int cellIdx;
} FrameRequest;

static int cmp_frame_requests(const void *a, const void *b) {
  return ((const FrameRequest *)a)->frameIdx -
         ((const FrameRequest *)b)->frameIdx;
}
void save_orientation_results(UniqueOrientationsResult *uniqueResult,
                              const char *outputFolder);
void free_resources(SpotList *spotList, UniqueOrientationsResult *uniqueResult);
int safe_realloc(void **ptr, size_t new_size);
void cleanup_shared_memory(const char *filename);

/* External function declarations - these are supplied from external files */

/**
 * Converts an orientation matrix to quaternion representation
 *
 * @param OM Input orientation matrix (3x3 array stored as 9-element array)
 * @param Quat Output quaternion (4-element array)
 */
extern void OrientMat2Quat(const double *OM, double *Quat);

/**
 * Calculates misorientation between two orientations
 *
 * @param Quat1 First orientation as quaternion
 * @param Quat2 Second orientation as quaternion
 * @param Axis Output rotation axis (3-element array)
 * @param ang Output rotation angle (pointer to double)
 * @param sgNr Space group number for crystallographic symmetry
 * @return Misorientation angle in degrees
 */
extern double GetMisOrientation(const double *Quat1, const double *Quat2,
                                double *Axis, double *ang, int sgNr);

/**
 * Main program entry point
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Program exit code
 */
int main(int argc, char *argv[]) {
  double start_time = omp_get_wtime();
  printf("\n\n\t\tFinding Single Solution in PF-HEDM.\n\n");

  /* Parse command line arguments */
  if (argc < 8 || argc > 10) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  /* Extract parameters from command line arguments */
  char folderName[MAX_PATH_LEN];
  sprintf(folderName, "%s/Output/", argv[1]);

  int sgNr =
      atoi(argv[2]); /* Space group number for crystallographic symmetry */
  double maxAng = atof(argv[3]); /* Maximum misorientation angle for considering
                                    orientations equivalent */
  int nScans = atoi(argv[4]);    /* Number of scans in the experiment */
  int numProcs =
      atoi(argv[5]); /* Number of CPU cores to use for parallel processing */
  double tolOme =
      atof(argv[6]); /* Tolerance for omega angle when matching spots */
  double tolEta =
      atof(argv[7]); /* Tolerance for eta angle when matching spots */
  const char *paramFile = (argc >= 9) ? argv[8] : "paramstest.txt";
  int normalizeSino = (argc >= 10) ? atoi(argv[9]) : 1;
  int absTransform = (argc >= 11) ? atoi(argv[10]) : 1;

  /* Validate input parameters */
  if (sgNr <= 0 || maxAng <= 0.0 || nScans <= 0 || numProcs <= 0 ||
      tolOme <= 0.0 || tolEta <= 0.0) {
    fatal_error(
        "Invalid input parameters. All numeric values must be positive.");
  }

  /* Initialize arrays for storing data from all voxels */
  size_t *allKeyArr =
      calloc(nScans * nScans * KEY_ARRAY_COLS, sizeof(*allKeyArr));
  double *allOrientationsArr = calloc(nScans * nScans * (ORIENT_ARRAY_COLS + 1),
                                      sizeof(*allOrientationsArr));

  if (!allKeyArr || !allOrientationsArr) {
    free(allKeyArr); /* Safe to call free with NULL */
    free(allOrientationsArr);
    fatal_error("Failed to allocate memory for key and orientation arrays");
  }

  /* Process each voxel in parallel */
  printf("Processing %d voxels with %d threads...\n", nScans * nScans,
         numProcs);

  char outKeyFN[MAX_PATH_LEN];
  sprintf(outKeyFN, "%s/UniqueIndexSingleKey.bin", folderName);
  int ib = open(outKeyFN, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (ib < 0) {
    fatal_error("Failed to open %s: %s", outKeyFN, strerror(errno));
  }

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
    process_voxel(voxNr, folderName, sgNr, maxAng, allKeyArr,
                  allOrientationsArr, ib);
  }
  close(ib);

  printf("Voxel processing complete. Finding unique orientations...\n");

  /* Find unique orientations across all voxels */
  UniqueOrientationsResult uniqueResult = find_unique_orientations(
      allKeyArr, allOrientationsArr, nScans, sgNr, maxAng);

  printf("Number of unique orientations found: %zu\n", uniqueResult.nUniques);

  /* Save unique orientations to file */
  save_orientation_results(&uniqueResult, argv[1]);

  /* Read all spots data */
  char spotsFilename[MAX_PATH_LEN];
  sprintf(spotsFilename, "%s/Spots.bin", argv[1]);

  /* Copy to shared memory for faster access */
  printf("Copying spots data to shared memory for faster access...\n");
  char command[MAX_PATH_LEN * 2];
  sprintf(command, "cp %s /dev/shm/", spotsFilename);

  int system_status = system(command);
  if (system_status != 0) {
    log_error("Failed to copy spots file to shared memory. Continuing with "
              "original file.");
    /* Continue with original file if shared memory copy fails */
  } else {
    sprintf(spotsFilename, "/dev/shm/Spots.bin");
  }

  /* Memory map the spots file for efficient access */
  size_t spotsDataSize;
  double *allSpots = read_memory_mapped_file(spotsFilename, &spotsDataSize);
  size_t nSpotsAll = spotsDataSize / (SPOTS_ARRAY_COLS * sizeof(double));
  printf("Total number of spots in dataset: %zu\n", nSpotsAll);

  /* Process spots for each unique orientation */
  printf("Processing spots for each unique orientation...\n");
  SpotList spotList = process_spots(&uniqueResult, folderName, allSpots,
                                    nSpotsAll, tolOme, tolEta);

  /* Generate sinograms for visualization */
  printf("Generating sinograms...\n");
  int maxNHKLs = 0;
  generate_sinograms(&spotList, &uniqueResult, allSpots, nSpotsAll, nScans,
                     tolOme, tolEta, argv[1], numProcs, normalizeSino,
                     absTransform, &maxNHKLs);

  /* Extract intensity patches for the viewer */
  printf("Extracting patches...\n");
  extract_patches(argv[1], argv[1], paramFile, uniqueResult.nUniques, maxNHKLs,
                  nScans, numProcs);

  /* Clean up resources */
  printf("Cleaning up resources...\n");
  free(allKeyArr);
  free(allOrientationsArr);
  free_resources(&spotList, &uniqueResult);

  /* Unmap memory mapped file */
  if (munmap(allSpots, spotsDataSize) != 0) {
    log_error("Failed to unmap spots file: %s", strerror(errno));
  }

  /* Clean up shared memory if used */
  if (strstr(spotsFilename, "/dev/shm/") != NULL) {
    cleanup_shared_memory(spotsFilename);
  }

  /* Report total processing time */
  double elapsed = omp_get_wtime() - start_time;
  printf("Total processing time: %.2f seconds\n", elapsed);

  return EXIT_SUCCESS;
}

/**
 * Print program usage information
 *
 * @param program_name Name of the executable
 */
void print_usage(const char *program_name) {
  printf(
      "Supply foldername spaceGroup, maxAng, NumberScans, nCPUs, tolOme, "
      "tolEta [normalizeSino] [absTransform] as arguments:\n"
      "%s foldername sgNum maxAngle nScans nCPUs tolOme tolEta [normalizeSino] "
      "[absTransform]\n"
      "\nWhere:\n"
      "  foldername:     Path to the folder containing input data\n"
      "  sgNum:          Space group number for crystallographic symmetry\n"
      "  maxAngle:       Maximum misorientation angle for grouping "
      "orientations "
      "(degrees)\n"
      "  nScans:         Number of scans in the experiment\n"
      "  nCPUs:          Number of CPU cores to use for parallel processing\n"
      "  tolOme:         Tolerance for omega angle when matching spots "
      "(degrees)\n"
      "  tolEta:         Tolerance for eta angle when matching spots "
      "(degrees)\n"
      "  normalizeSino:  0=off, 1=on (default: 1)\n"
      "  absTransform:   0=off, 1=on (default: 1)\n"
      "                  When on, applies exp(-I/Imax) so absorption CT codes\n"
      "                  can recover diffraction intensity via -ln().\n"
      "\nThe indexing results need to be in folderName/Output\n",
      program_name);
}

/**
 * Safely reallocate memory with error checking
 *
 * @param ptr Pointer to the memory block to be reallocated
 * @param new_size New size in bytes
 * @return 0 on success, error code on failure
 */
int safe_realloc(void **ptr, size_t new_size) {
  void *new_ptr = realloc(*ptr, new_size);
  if (new_ptr == NULL && new_size > 0) {
    return ERR_MEMORY_ALLOC;
  }
  *ptr = new_ptr;
  return SUCCESS;
}

/**
 * Clean up a file in shared memory
 *
 * @param filename Path to the file in shared memory
 */
void cleanup_shared_memory(const char *filename) {
  if (unlink(filename) != 0) {
    log_error("Failed to remove shared memory file %s: %s", filename,
              strerror(errno));
  } else {
    printf("Removed shared memory file: %s\n", filename);
  }
}

/**
 * Read a file into memory using memory mapping for efficient access
 *
 * @param filename Path to the file to be memory mapped
 * @param size_out Pointer to store the size of the mapped file
 * @return Pointer to the memory mapped region, or NULL on failure
 */
double *read_memory_mapped_file(const char *filename, size_t *size_out) {
  /* Open the file for reading */
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    fatal_error("open %s failed: %s", filename, strerror(errno));
  }

  /* Get file size */
  struct stat s;
  int status = fstat(fd, &s);
  if (status < 0) {
    close(fd);
    fatal_error("stat %s failed: %s", filename, strerror(errno));
  }

  /* Store file size in the output parameter */
  *size_out = s.st_size;

  /* Check for zero-sized file */
  if (*size_out == 0) {
    close(fd);
    fatal_error("File %s is empty", filename);
  }

  /* Map the file into memory */
  double *mapped_data = mmap(0, *size_out, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped_data == MAP_FAILED) {
    close(fd);
    fatal_error("mmap %s failed: %s", filename, strerror(errno));
  }

  /* Close the file descriptor (the mapping remains valid) */
  close(fd);

  return mapped_data;
}

/**
 * Comparison function for sorting sinogram data by angle
 *
 * @param a First element to compare
 * @param b Second element to compare
 * @return Comparison result (-1 if a<b, 1 if a>=b)
 */
int compare_sino_data(const void *a, const void *b) {
  const SinoSortData *ia = (const SinoSortData *)a;
  const SinoSortData *ib = (const SinoSortData *)b;

  /* Compare angles and return sort order */
  if (ia->angle >= ib->angle)
    return 1;
  else
    return -1;
}

/**
 * Process a single voxel to find the best orientation solution
 *
 * This function:
 * 1. Reads indexing results for the voxel
 * 2. Evaluates confidence and internal angle for each solution
 * 3. Finds the best solution based on confidence and internal angle
 * 4. Identifies unique orientations within the voxel
 * 5. Saves the results to output files
 *
 * @param voxNr Voxel number to process
 * @param folderName Path to the folder containing input data
 * @param sgNr Space group number for crystallographic symmetry
 * @param maxAng Maximum misorientation angle for grouping orientations
 */
void process_voxel(int voxNr, const char *folderName, int sgNr, double maxAng,
                   size_t *allKeyArr, double *allOrientationsArr, int ib) {
  /* Construct input filenames for this voxel */
  FILE *valsF = NULL, *keyF = NULL;
  char valsFN[MAX_PATH_LEN], keyFN[MAX_PATH_LEN];
  sprintf(valsFN, "%s/IndexBest_voxNr_%06d.bin", folderName, voxNr);
  sprintf(keyFN, "%s/IndexKey_voxNr_%06d.txt", folderName, voxNr);

  /* Open input files */
  valsF = fopen(valsFN, "rb");
  keyF = fopen(keyFN, "r");

  /* Check if files opened successfully */
  if (!keyF || !valsF) {
    if (!keyF)
      log_error("Could not open key file %s", keyFN);
    if (!valsF)
      log_error("Could not open vals file %s", valsFN);

    /* Write empty result */
    size_t outarr[5] = {0};
    pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
    allKeyArr[voxNr * KEY_ARRAY_COLS] = INVALID_VOX;

    /* Clean up */
    if (keyF)
      fclose(keyF);
    if (valsF)
      fclose(valsF);
    return;
  }

  /* Check if key file is empty */
  fseek(keyF, 0L, SEEK_END);
  size_t szt = ftell(keyF);
  rewind(keyF);

  if (szt == 0) {
    /* Write empty result for empty key file */
    fclose(keyF);
    fclose(valsF);

    size_t outarr[5] = {0};
    pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
    allKeyArr[voxNr * KEY_ARRAY_COLS] = INVALID_VOX;
    return;
  }

  /* Read key file */
  size_t *keys =
      calloc(MAX_N_SOLUTIONS_PER_VOX * KEY_ARRAY_COLS, sizeof(*keys));
  if (!keys) {
    log_error("Failed to allocate memory for keys");
    fclose(keyF);
    fclose(valsF);
    return;
  }

  char aline[MAX_PATH_LEN];
  int nIDs = 0;

  /* Parse each line in the key file */
  while (fgets(aline, MAX_PATH_LEN, keyF) != NULL &&
         nIDs < MAX_N_SOLUTIONS_PER_VOX) {
    /* Extract key values from each line */
    if (sscanf(aline, "%zu %zu %zu %zu",
               &keys[nIDs * KEY_ARRAY_COLS + 0], /* grainID */
               &keys[nIDs * KEY_ARRAY_COLS + 1], /* nSpots */
               &keys[nIDs * KEY_ARRAY_COLS + 2], /* startRowNr */
               &keys[nIDs * KEY_ARRAY_COLS + 3]) /* spotListStartPos */
        != 4) {
      /* Skip lines that don't have 4 values */
      continue;
    }
    nIDs++;
  }

  /* Resize keys array to actual size */
  if (nIDs > 0) {
    size_t *resized_keys = realloc(keys, nIDs * KEY_ARRAY_COLS * sizeof(*keys));
    if (resized_keys) {
      keys = resized_keys;
    }
    /* If realloc fails, we continue with the original larger array */
  }

  fclose(keyF);

  /* Read values file for orientations, confidence, and internal angle */
  double *OMArr = calloc(nIDs * ORIENT_ARRAY_COLS, sizeof(double));
  double *confIAArr = calloc(nIDs * CONF_IA_ARRAY_COLS, sizeof(double));
  double *tmpArr = calloc(nIDs * TMP_ARRAY_COLS, sizeof(double));

  if (!OMArr || !confIAArr || !tmpArr) {
    log_error("Failed to allocate memory for orientation arrays");
    free(keys);
    free(OMArr);     /* Safe to free NULL */
    free(confIAArr); /* Safe to free NULL */
    free(tmpArr);    /* Safe to free NULL */
    fclose(valsF);
    return;
  }

  /* Read orientation values from binary file */
  size_t items_read =
      fread(tmpArr, sizeof(double), nIDs * TMP_ARRAY_COLS, valsF);
  fclose(valsF);

  /* Check if we read the expected number of items */
  if (items_read != nIDs * TMP_ARRAY_COLS) {
    log_error("Failed to read expected number of values from %s. Expected %d, "
              "got %zu",
              valsFN, nIDs * TMP_ARRAY_COLS, items_read);
    /* Continue with partial data */
  }

  /* Extract confidence, internal angle, and orientation matrix values */
  for (int i = 0; i < nIDs; i++) {
    /* Calculate confidence as ratio of matched spots to total spots */
    confIAArr[i * CONF_IA_ARRAY_COLS + 0] =
        tmpArr[i * TMP_ARRAY_COLS + 15] / tmpArr[i * TMP_ARRAY_COLS + 14];

    /* Store internal angle (measure of solution quality) */
    confIAArr[i * CONF_IA_ARRAY_COLS + 1] = tmpArr[i * TMP_ARRAY_COLS + 1];

    /* Extract orientation matrix (9 values) */
    for (int k = 0; k < ORIENT_ARRAY_COLS; k++) {
      OMArr[i * ORIENT_ARRAY_COLS + k] = tmpArr[i * TMP_ARRAY_COLS + 2 + k];
    }
  }

  /* Find best orientation based on confidence and internal angle */
  bool *markArr = calloc(nIDs, sizeof(*markArr));
  if (!markArr) {
    log_error("Failed to allocate memory for mark array");
    free(keys);
    free(OMArr);
    free(confIAArr);
    free(tmpArr);
    return;
  }

  /* Initialize with all orientations unmarked */
  memset(markArr, 0, nIDs * sizeof(*markArr));

  /* Find the orientation with highest confidence and lowest internal angle */
  int bestRow = -1;
  double bestConf = -1, bestIA = 100;

  for (int i = 0; i < nIDs; i++) {
    if (markArr[i])
      continue; /* Skip marked orientations */

    /* Compare with current best */
    if (confIAArr[i * CONF_IA_ARRAY_COLS + 0] < bestConf)
      continue;
    if (confIAArr[i * CONF_IA_ARRAY_COLS + 0] == bestConf &&
        confIAArr[i * CONF_IA_ARRAY_COLS + 1] > bestIA)
      continue;

    /* Update best values */
    bestConf = confIAArr[i * CONF_IA_ARRAY_COLS + 0];
    bestIA = confIAArr[i * CONF_IA_ARRAY_COLS + 1];
    bestRow = i;
  }

  if (bestRow == -1) {
    /* No valid orientation found */
    size_t outarr[5] = {0};
    pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
    allKeyArr[voxNr * KEY_ARRAY_COLS] = INVALID_VOX;
  } else {
    /* Process unique orientations within this voxel */
    for (int i = 0; i < nIDs; i++)
      markArr[i] = false;

    /* Arrays for quaternion comparison and unique orientations */
    double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3], ang;
    size_t *uniqueArrThis =
        calloc(nIDs * KEY_ARRAY_COLS, sizeof(*uniqueArrThis));
    double *uniqueOrientArrThis =
        calloc(nIDs * ORIENT_ARRAY_COLS, sizeof(*uniqueOrientArrThis));

    if (!uniqueArrThis || !uniqueOrientArrThis) {
      log_error("Failed to allocate memory for unique arrays");
      free(keys);
      free(OMArr);
      free(confIAArr);
      free(tmpArr);
      free(markArr);
      free(uniqueArrThis);       /* Safe to free NULL */
      free(uniqueOrientArrThis); /* Safe to free NULL */
      return;
    }

    int nUniquesThis = 0;

    /* Find unique orientations by comparing misorientation angles */
    for (int i = 0; i < nIDs; i++) {
      if (markArr[i])
        continue; /* Skip marked orientations */

      /* Copy orientation matrix and convert to quaternion */
      memcpy(OMThis, &OMArr[i * ORIENT_ARRAY_COLS],
             ORIENT_ARRAY_COLS * sizeof(double));
      OrientMat2Quat(OMThis, Quat1);

      /* Initialize best values from current orientation */
      double bCon = confIAArr[i * CONF_IA_ARRAY_COLS + 0];
      double bIA = confIAArr[i * CONF_IA_ARRAY_COLS + 1];
      int bRN = i;

      /* Compare with other orientations to find similar ones */
      for (int j = i + 1; j < nIDs; j++) {
        if (markArr[j])
          continue; /* Skip marked orientations */

        /* Convert orientation to quaternion */
        memcpy(OMInside, &OMArr[j * ORIENT_ARRAY_COLS],
               ORIENT_ARRAY_COLS * sizeof(double));
        OrientMat2Quat(OMInside, Quat2);

        /* Get confidence and internal angle */
        double conIn = confIAArr[j * CONF_IA_ARRAY_COLS + 0];
        double iaIn = confIAArr[j * CONF_IA_ARRAY_COLS + 1];

        /* Calculate misorientation angle */
        GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);

        /* Group similar orientations */
        if (ang < maxAng) {
          /* Keep track of the best orientation in this group */
          if (bCon < conIn) {
            bCon = conIn;
            bIA = iaIn;
            bRN = j;
          } else if (bCon == conIn && bIA > iaIn) {
            bCon = conIn;
            bIA = iaIn;
            bRN = j;
          }

          /* Mark this orientation as processed */
          markArr[j] = true;
        }
      }

      /* Store the best orientation from this group */
      memcpy(&uniqueArrThis[nUniquesThis * KEY_ARRAY_COLS],
             &keys[bRN * KEY_ARRAY_COLS], KEY_ARRAY_COLS * sizeof(size_t));

      memcpy(&uniqueOrientArrThis[nUniquesThis * ORIENT_ARRAY_COLS],
             &OMArr[bRN * ORIENT_ARRAY_COLS],
             ORIENT_ARRAY_COLS * sizeof(double));

      nUniquesThis++;
    }

    /* Save the overall best orientation for this voxel */
    size_t outarr[5] = {voxNr, keys[bestRow * KEY_ARRAY_COLS + 0],
                        keys[bestRow * KEY_ARRAY_COLS + 1],
                        keys[bestRow * KEY_ARRAY_COLS + 2],
                        keys[bestRow * KEY_ARRAY_COLS + 3]};

    /* Write to the output file at the correct position */
    pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);

    /* Save best solution to global arrays for find_unique_orientations */
    size_t keyIdx = voxNr * KEY_ARRAY_COLS;
    size_t orientIdx = voxNr * (ORIENT_ARRAY_COLS + 1);

    /* Copy keys: [grainID, nSpots, startRowNr, spotListStartPos] */
    memcpy(&allKeyArr[keyIdx], &keys[bestRow * KEY_ARRAY_COLS],
           KEY_ARRAY_COLS * sizeof(size_t));

    /* Copy orientation matrix */
    memcpy(&allOrientationsArr[orientIdx], &OMArr[bestRow * ORIENT_ARRAY_COLS],
           ORIENT_ARRAY_COLS * sizeof(double));

    /* Copy confidence (stored in last column) */
    allOrientationsArr[orientIdx + ORIENT_ARRAY_COLS] =
        confIAArr[bestRow * CONF_IA_ARRAY_COLS + 0];

    /* Save unique orientations to a separate file for this voxel */
    char outKeyFN[MAX_PATH_LEN];
    sprintf(outKeyFN, "%s/UniqueIndexKeyOrientAll_voxNr_%06d.txt", folderName,
            voxNr);

    FILE *outKeyF = fopen(outKeyFN, "w");
    if (outKeyF) {
      /* Write each unique orientation with its key values */
      for (int i = 0; i < nUniquesThis; i++) {
        /* Write key values */
        for (int j = 0; j < KEY_ARRAY_COLS; j++) {
          fprintf(outKeyF, "%zu ", uniqueArrThis[i * KEY_ARRAY_COLS + j]);
        }

        /* Write orientation values */
        for (int j = 0; j < ORIENT_ARRAY_COLS; j++) {
          fprintf(outKeyF, "%lf ",
                  uniqueOrientArrThis[i * ORIENT_ARRAY_COLS + j]);
        }

        fprintf(outKeyF, "\n");
      }
      fclose(outKeyF);
    } else {
      log_error("Failed to open %s for writing", outKeyFN);
    }

    /* Free unique orientation arrays */
    free(uniqueArrThis);
    free(uniqueOrientArrThis);
  }

  /* Clean up resources */
  free(keys);
  free(OMArr);
  free(confIAArr);
  free(tmpArr);
  free(markArr);
}

/**
 * Find unique orientations across all voxels
 *
 * This function identifies unique crystal orientations across all voxels
 * by comparing misorientation angles between orientation matrices.
 *
 * @param allKeyArr Array of key values for all orientations
 * @param allOrientationsArr Array of orientation matrices for all orientations
 * @param nScans Number of scans in the experiment
 * @param sgNr Space group number for crystallographic symmetry
 * @param maxAng Maximum misorientation angle for grouping orientations
 * @return Structure containing unique orientations information
 */
UniqueOrientationsResult find_unique_orientations(size_t *allKeyArr,
                                                  double *allOrientationsArr,
                                                  size_t nScans, int sgNr,
                                                  double maxAng) {

  UniqueOrientationsResult result;
  result.nUniques = 0;

  /* Allocate memory for results */
  result.uniqueKeyArr =
      calloc(nScans * nScans * 5, sizeof(*(result.uniqueKeyArr)));
  result.uniqueOrientArr = calloc(nScans * nScans * ORIENT_ARRAY_COLS,
                                  sizeof(*(result.uniqueOrientArr)));

  if (!result.uniqueKeyArr || !result.uniqueOrientArr) {
    free(result.uniqueKeyArr);    /* Safe to free NULL */
    free(result.uniqueOrientArr); /* Safe to free NULL */
    fatal_error("Failed to allocate memory for unique orientation results");
  }

  /* Mark invalid orientations */
  bool *markArr = malloc(nScans * nScans * sizeof(*markArr));
  if (!markArr) {
    free(result.uniqueKeyArr);
    free(result.uniqueOrientArr);
    fatal_error("Failed to allocate mark array");
  }

  /* Initialize mark array - mark entries with invalid key values */
  for (size_t i = 0; i < nScans * nScans; i++) {
    markArr[i] = (allKeyArr[i * KEY_ARRAY_COLS] == INVALID_VOX);
  }

  /* Find unique orientations */
  double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3];
  double ang, fracInside, bestFrac;
  size_t bestOrientationRowNr;

  for (size_t i = 0; i < nScans * nScans; i++) {
    if (markArr[i])
      continue; /* Skip marked orientations */

    /* Copy orientation matrix and convert to quaternion */
    memcpy(OMThis, &allOrientationsArr[i * 10], 9 * sizeof(double));
    bestFrac = allOrientationsArr[i * 10 + 9]; /* Quality metric */
    OrientMat2Quat(OMThis, Quat1);
    bestOrientationRowNr = i;

    /* Compare with other orientations */
    for (size_t j = i + 1; j < nScans * nScans; j++) {
      if (markArr[j])
        continue; /* Skip marked orientations */

      /* Get quality metric and orientation matrix */
      fracInside = allOrientationsArr[j * 10 + 9];
      memcpy(OMInside, &allOrientationsArr[j * 10], 9 * sizeof(double));
      OrientMat2Quat(OMInside, Quat2);

      /* Calculate misorientation angle */
      GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);

      /* Group similar orientations */
      if (ang < maxAng) {
        /* Keep track of the best orientation in this group */
        if (bestFrac < fracInside) {
          bestFrac = fracInside;
          bestOrientationRowNr = j;
        }

        /* Mark this orientation as processed */
        markArr[j] = true;
      }
    }

    /* Store the best orientation from this group */
    result.uniqueKeyArr[result.nUniques * 5] = bestOrientationRowNr;

    memcpy(&result.uniqueKeyArr[result.nUniques * 5 + 1],
           &allKeyArr[bestOrientationRowNr * KEY_ARRAY_COLS],
           KEY_ARRAY_COLS * sizeof(size_t));

    memcpy(&result.uniqueOrientArr[result.nUniques * ORIENT_ARRAY_COLS],
           &allOrientationsArr[bestOrientationRowNr * 10],
           ORIENT_ARRAY_COLS * sizeof(double));

    result.nUniques++;
  }

  free(markArr);

  /* Resize arrays to actual size - handle possible failure gracefully */
  if (result.nUniques > 0 && result.nUniques < nScans * nScans) {
    size_t *resized_keys =
        realloc(result.uniqueKeyArr,
                result.nUniques * 5 * sizeof(*(result.uniqueKeyArr)));
    if (resized_keys) {
      result.uniqueKeyArr = resized_keys;
    }

    double *resized_orients =
        realloc(result.uniqueOrientArr, result.nUniques * ORIENT_ARRAY_COLS *
                                            sizeof(*(result.uniqueOrientArr)));
    if (resized_orients) {
      result.uniqueOrientArr = resized_orients;
    }
  }

  return result;
}

/**
 * Save unique orientations to output file
 *
 * @param uniqueResult Structure containing unique orientations information
 * @param outputFolder Path to the output folder
 */
void save_orientation_results(UniqueOrientationsResult *uniqueResult,
                              const char *outputFolder) {
  char uniqueOrientsFN[MAX_PATH_LEN];
  sprintf(uniqueOrientsFN, "%s/UniqueOrientations.csv", outputFolder);

  FILE *uniqueOrientationsF = fopen(uniqueOrientsFN, "w");
  if (!uniqueOrientationsF) {
    log_error("Failed to open %s for writing", uniqueOrientsFN);
    return;
  }

  /* Write header */
  fprintf(uniqueOrientationsF, "# GrainID RowNr nSpots StartRowNr ListStartPos "
                               "OM1 OM2 OM3 OM4 OM5 OM6 OM7 OM8 OM9\n");

  /* Write each unique orientation with its metadata */
  for (size_t i = 0; i < uniqueResult->nUniques; i++) {
    /* Write key values */
    for (int j = 0; j < 5; j++) {
      fprintf(uniqueOrientationsF, "%zu ",
              uniqueResult->uniqueKeyArr[i * 5 + j]);
    }

    /* Write orientation matrix values */
    for (int j = 0; j < ORIENT_ARRAY_COLS; j++) {
      fprintf(uniqueOrientationsF, "%lf ",
              uniqueResult->uniqueOrientArr[i * ORIENT_ARRAY_COLS + j]);
    }

    fprintf(uniqueOrientationsF, "\n");
  }

  fclose(uniqueOrientationsF);
}

/**
 * Process spots for each unique grain orientation
 *
 * This function identifies and associates diffraction spots with each
 * unique grain orientation, removing duplicate spots.
 *
 * @param uniqueResult Structure containing unique orientations information
 * @param folderName Path to the folder containing input data
 * @param allSpots Array of all spot data
 * @param nSpotsAll Number of spots in the dataset
 * @param tolOme Tolerance for omega angle when matching spots
 * @param tolEta Tolerance for eta angle when matching spots
 * @return Structure containing processed spot data
 */
SpotList process_spots(UniqueOrientationsResult *uniqueResult,
                       const char *folderName, double *allSpots,
                       size_t nSpotsAll, double tolOme, double tolEta) {

  printf("Processing spots for %zu unique orientations...\n",
         uniqueResult->nUniques);

  SpotList result;
  result.spotData = calloc(MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques,
                           sizeof(*(result.spotData)));

  if (!result.spotData) {
    fatal_error("Failed to allocate memory for spot data (%zu bytes)",
                MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques *
                    sizeof(*(result.spotData)));
  }

  /* Array to track non-unique spots */
  bool *isNotUniqueSpot = calloc(MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques,
                                 sizeof(*isNotUniqueSpot));
  if (!isNotUniqueSpot) {
    free(result.spotData);
    fatal_error("Failed to allocate memory for spot uniqueness flags");
  }

  /* Initialize with all spots being unique */
  memset(isNotUniqueSpot, 0,
         MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques *
             sizeof(*isNotUniqueSpot));

  /* Count spots per grain */
  int *nrHKLsFilled = calloc(uniqueResult->nUniques, sizeof(*nrHKLsFilled));
  if (!nrHKLsFilled) {
    free(result.spotData);
    free(isNotUniqueSpot);
    fatal_error("Failed to allocate memory for HKL counts");
  }

  size_t nAllSpots = 0;

  /* Process each grain */
  for (size_t i = 0; i < uniqueResult->nUniques; i++) {
    size_t thisVoxNr = uniqueResult->uniqueKeyArr[i * 5];
    size_t nSpots = uniqueResult->uniqueKeyArr[i * 5 + 2];
    size_t startPos = uniqueResult->uniqueKeyArr[i * 5 + 4];

    /* Check for invalid values to prevent buffer overflows */
    if (nSpots > MAX_N_SPOTS_PER_GRAIN) {
      log_error("Grain %zu has too many spots (%zu), limiting to %d", i, nSpots,
                MAX_N_SPOTS_PER_GRAIN);
      nSpots = MAX_N_SPOTS_PER_GRAIN;
    }

    if (nAllSpots + nSpots > MAX_N_SPOTS_TOTAL) {
      log_error("Maximum total spots exceeded, stopping at grain %zu", i);
      break;
    }

    /* Read spot IDs */
    char IDsFNThis[MAX_PATH_LEN];
    sprintf(IDsFNThis, "%s/IndexBest_IDs_voxNr_%06zu.bin", folderName,
            thisVoxNr);

    FILE *IDF = fopen(IDsFNThis, "rb");
    if (!IDF) {
      log_error("Failed to open %s", IDsFNThis);
      continue;
    }

    /* Seek to the correct position in the file */
    if (fseek(IDF, startPos, SEEK_SET) != 0) {
      log_error("Failed to seek to position %zu in %s", startPos, IDsFNThis);
      fclose(IDF);
      continue;
    }

    /* Allocate memory for spot IDs */
    int *IDArrThis = malloc(nSpots * sizeof(*IDArrThis));
    if (!IDArrThis) {
      log_error("Failed to allocate memory for ID array");
      fclose(IDF);
      continue;
    }

    /* Read spot IDs */
    size_t items_read = fread(IDArrThis, sizeof(int), nSpots, IDF);
    // for (size_t i = 0; i < nSpots; i++) {
    //   printf("%d ", IDArrThis[i]);
    // }
    // printf("\n");
    fclose(IDF);

    /* Check if we read the expected number of IDs */
    if (items_read != nSpots) {
      log_error("Failed to read expected number of spot IDs from %s. Expected "
                "%zu, got %zu",
                IDsFNThis, nSpots, items_read);
      free(IDArrThis);
      continue;
    }

    int uniqueSpotCount = 0;

    /* Process each spot */
    for (size_t j = 0; j < nSpots; j++) {
      /* Ensure spot ID is valid to prevent out-of-bounds access */
      if (IDArrThis[j] < 1 || IDArrThis[j] > (int)nSpotsAll) {
        log_error("Invalid spot ID %d (range 1-%zu) at grain %zu, spot %zu, "
                  "total spots: %zu, file: %s",
                  IDArrThis[j], nSpotsAll, i, j, nSpots, IDsFNThis);
        continue;
      }

      /* Verify data alignment */
      size_t spotIdx = (size_t)(IDArrThis[j] - 1);
      if (allSpots[SPOTS_ARRAY_COLS * spotIdx + 4] != (double)IDArrThis[j]) {
        log_error("Data is not aligned. Spot ID %d at index %zu doesn't match "
                  "expected value",
                  IDArrThis[j], spotIdx);
        continue;
      }

      /* Store spot data */
      result.spotData[nAllSpots + j].mergedID = IDArrThis[j];
      result.spotData[nAllSpots + j].omega =
          allSpots[SPOTS_ARRAY_COLS * spotIdx + 2];
      result.spotData[nAllSpots + j].eta =
          allSpots[SPOTS_ARRAY_COLS * spotIdx + 6];
      result.spotData[nAllSpots + j].ringNr =
          (int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 5];
      result.spotData[nAllSpots + j].grainNr = i;
      result.spotData[nAllSpots + j].spotNr = j;

      /* Check if this spot is a duplicate of any previous spot */
      bool isDuplicate = false;
      for (size_t k = 0; k < nAllSpots + j; k++) {
        if (result.spotData[k].ringNr ==
                result.spotData[nAllSpots + j].ringNr &&
            fabs(result.spotData[nAllSpots + j].omega -
                 result.spotData[k].omega) < tolOme &&
            fabs(result.spotData[nAllSpots + j].eta - result.spotData[k].eta) <
                tolEta) {

          isNotUniqueSpot[k] = true;
          isNotUniqueSpot[nAllSpots + j] = true;
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate) {
        uniqueSpotCount++;
      }
    }

    /* Debug output */
    printf("Grain %zu: %d unique spots out of %zu total\n", i, uniqueSpotCount,
           nSpots);

    free(IDArrThis);
    nAllSpots += nSpots;
  }

  /* Extract only unique spots */
  SpotData *uniqueSpots = calloc(nAllSpots, sizeof(*uniqueSpots));
  if (!uniqueSpots) {
    free(result.spotData);
    free(isNotUniqueSpot);
    free(nrHKLsFilled);
    fatal_error("Failed to allocate memory for unique spots");
  }

  /* Copy only unique spots to the new array */
  size_t nUniqueSpots = 0;
  for (size_t i = 0; i < nAllSpots; i++) {
    if (!isNotUniqueSpot[i]) {
      /* Update spot number to be sequential within each grain */
      uniqueSpots[nUniqueSpots] = result.spotData[i];
      uniqueSpots[nUniqueSpots].spotNr =
          nrHKLsFilled[uniqueSpots[nUniqueSpots].grainNr]++;
      nUniqueSpots++;
    }
  }

  /* Find maximum number of spots per grain */
  int maxNHKLs = 0;
  for (size_t i = 0; i < uniqueResult->nUniques; i++) {
    if (nrHKLsFilled[i] > maxNHKLs) {
      maxNHKLs = nrHKLsFilled[i];
    }
  }

  /* Replace the original spot data with unique spots */
  free(result.spotData);
  result.spotData = uniqueSpots;
  result.nSpots = nUniqueSpots;

  printf("Total spots: %zu, Unique spots: %zu, Max spots per grain: %d\n",
         nAllSpots, nUniqueSpots, maxNHKLs);

  /* Save unique spots to file */
  char fnUniqueSpots[MAX_PATH_LEN];
  sprintf(fnUniqueSpots, "%s/UniqueOrientationSpots.csv", folderName);

  FILE *fUniqueSpots = fopen(fnUniqueSpots, "w");
  if (fUniqueSpots) {
    /* Write header */
    fprintf(fUniqueSpots, "ID,GrainNr,SpotNr,RingNr,Omega,Eta\n");

    /* Write spot data */
    for (size_t i = 0; i < nUniqueSpots; i++) {
      fprintf(fUniqueSpots, "%d,%d,%d,%d,%lf,%lf\n", uniqueSpots[i].mergedID,
              uniqueSpots[i].grainNr, uniqueSpots[i].spotNr,
              uniqueSpots[i].ringNr, uniqueSpots[i].omega, uniqueSpots[i].eta);
    }

    fclose(fUniqueSpots);
  } else {
    log_error("Failed to open %s for writing", fnUniqueSpots);
  }

  /* Clean up */
  free(isNotUniqueSpot);
  free(nrHKLsFilled);

  return result;
}

/**
 * Generate sinograms for each unique grain orientation
 *
 * This function:
 * 1. Identifies spots associated with each grain
 * 2. Creates sinograms showing spot intensity as a function of scan number
 * 3. Sorts spots by omega angle
 * 4. Normalizes intensities
 * 5. Saves results to binary files
 *
 * @param spotList Structure containing processed spot data
 * @param uniqueResult Structure containing unique orientations information
 * @param allSpots Array of all spot data
 * @param nSpotsAll Number of spots in the dataset
 * @param nScans Number of scans in the experiment
 * @param tolOme Tolerance for omega angle when matching spots
 * @param tolEta Tolerance for eta angle when matching spots
 * @param outputFolder Path to the output folder
 * @param numProcs Number of CPU cores to use for parallel processing
 */
void generate_sinograms(SpotList *spotList,
                        UniqueOrientationsResult *uniqueResult,
                        double *allSpots, size_t nSpotsAll, int nScans,
                        double tolOme, double tolEta, const char *outputFolder,
                        int numProcs, int normalizeSino, int absTransform,
                        int *outMaxNHKLs) {

  /* Find maximum number of spots per grain */
  int maxNHKLs = 0;
  int *nrHKLsPerGrain = calloc(uniqueResult->nUniques, sizeof(*nrHKLsPerGrain));

  if (!nrHKLsPerGrain) {
    fatal_error("Failed to allocate memory for HKL counts");
  }

  /* Count spots per grain */
  for (size_t i = 0; i < spotList->nSpots; i++) {
    int grainNr = spotList->spotData[i].grainNr;
    int spotNr = spotList->spotData[i].spotNr;

    /* Update maximum spot number for each grain */
    if (grainNr >= 0 && grainNr < (int)uniqueResult->nUniques) {
      if (spotNr + 1 > nrHKLsPerGrain[grainNr]) {
        nrHKLsPerGrain[grainNr] = spotNr + 1;
      }
    }
  }

  /* Find overall maximum */
  for (size_t i = 0; i < uniqueResult->nUniques; i++) {
    if (nrHKLsPerGrain[i] > maxNHKLs) {
      maxNHKLs = nrHKLsPerGrain[i];
    }
  }

  /* Allocate memory for sinograms */
  size_t szSino = uniqueResult->nUniques * maxNHKLs * nScans;
  double *sinoArr = calloc(szSino, sizeof(*sinoArr));

  /* Allocate spotID mapping array (same shape as sinoArr) */
  int *spotIDArr = calloc(szSino, sizeof(*spotIDArr));

/* Allocate per-cell metadata (4 doubles per cell: eta, 2theta, yCen, zCen) */
#define SPOT_META_COLS 4
  double *spotMetaArr = calloc(szSino * SPOT_META_COLS, sizeof(double));

  double *sumOmeArr =
      calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*sumOmeArr));
  int *countOmeArr =
      calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*countOmeArr));

  if (!sinoArr || !spotIDArr || !spotMetaArr || !sumOmeArr || !countOmeArr) {
    free(nrHKLsPerGrain);
    free(sinoArr); /* Safe to free NULL */
    free(spotIDArr);
    free(sumOmeArr);
    free(countOmeArr);
    fatal_error("Failed to allocate memory for sinograms");
  }

  /* Initialize arrays */
  memset(sinoArr, 0, szSino * sizeof(*sinoArr));
  /* Initialize spotID to -1 (no match) */
  for (size_t i = 0; i < szSino; i++)
    spotIDArr[i] = -1;
  /* Initialize spotMeta to NaN */
  for (size_t i = 0; i < szSino * SPOT_META_COLS; i++)
    spotMetaArr[i] = NAN;

  double *omeArr = calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*omeArr));
  double *maxIntArr =
      calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*maxIntArr));

  if (!omeArr || !maxIntArr) {
    free(nrHKLsPerGrain);
    free(sinoArr);
    free(sumOmeArr);
    free(countOmeArr);
    free(omeArr);    /* Safe to free NULL */
    free(maxIntArr); /* Safe to free NULL */
    fatal_error("Failed to allocate memory for omega arrays");
  }

  /* Initialize omega array with invalid values */
  for (size_t i = 0; i < uniqueResult->nUniques * maxNHKLs; i++) {
    omeArr[i] = -10000.0;
  }

  /* Process each scan in parallel */
  printf("Processing sinograms in parallel with %d threads...\n", numProcs);

#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int scanNr = 0; scanNr < nScans; scanNr++) {
    /* Loop through all spots in the scan */
    for (size_t spotIdx = 0; spotIdx < nSpotsAll; spotIdx++) {
      /* Check if this spot belongs to the current scan */
      if ((int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 9] != scanNr) {
        continue;
      }

      /* Compare with all unique spots */
      for (size_t uniqueSpotIdx = 0; uniqueSpotIdx < spotList->nSpots;
           uniqueSpotIdx++) {
        SpotData *spot = &spotList->spotData[uniqueSpotIdx];

        /* Check if ring number matches */
        if ((int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 5] != spot->ringNr) {
          continue;
        }

        /* Check if omega and eta are within tolerance */
        if (fabs(allSpots[SPOTS_ARRAY_COLS * spotIdx + 2] - spot->omega) <
                tolOme &&
            fabs(allSpots[SPOTS_ARRAY_COLS * spotIdx + 6] - spot->eta) <
                tolEta) {

          /* Calculate array indices with bounds checking */
          if (spot->grainNr < 0 ||
              spot->grainNr >= (int)uniqueResult->nUniques ||
              spot->spotNr < 0 || spot->spotNr >= maxNHKLs || scanNr < 0 ||
              scanNr >= nScans) {
            continue;
          }

          size_t locThis = (size_t)spot->grainNr * maxNHKLs * nScans +
                           (size_t)spot->spotNr * nScans + scanNr;
          size_t maxIntIdx =
              (size_t)spot->grainNr * maxNHKLs + (size_t)spot->spotNr;
          double currentIntensity = allSpots[SPOTS_ARRAY_COLS * spotIdx + 3];
          double currentOmega = allSpots[SPOTS_ARRAY_COLS * spotIdx + 2];

          /* Store intensity and spotID mapping */
          sinoArr[locThis] = currentIntensity;
          spotIDArr[locThis] = (int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 4];

          /* Store per-cell metadata: eta, 2theta, yCen_det, zCen_det */
          spotMetaArr[locThis * SPOT_META_COLS + 0] =
              allSpots[SPOTS_ARRAY_COLS * spotIdx + 6]; /* eta */
          spotMetaArr[locThis * SPOT_META_COLS + 1] =
              allSpots[SPOTS_ARRAY_COLS * spotIdx + 7] * 2.0; /* 2theta */
          spotMetaArr[locThis * SPOT_META_COLS + 2] =
              allSpots[SPOTS_ARRAY_COLS * spotIdx + 0]; /* yCen_det */
          spotMetaArr[locThis * SPOT_META_COLS + 3] =
              allSpots[SPOTS_ARRAY_COLS * spotIdx + 1]; /* zCen_det */

/* Critical section to prevent race conditions */
#pragma omp critical
          {
            if (maxIntArr[maxIntIdx] < currentIntensity) {
              maxIntArr[maxIntIdx] = currentIntensity;
            }
            if (currentIntensity > 0) {
              sumOmeArr[maxIntIdx] += currentOmega;
              countOmeArr[maxIntIdx]++;
            }
          }
        }
      }
    }
  }

  /* --- Diagnostic: match statistics per grain --- */
  printf("\n=== Sinogram match diagnostics ===\n");
  for (size_t g = 0; g < uniqueResult->nUniques; g++) {
    int nSp = nrHKLsPerGrain[g];
    int filled = 0;
    for (int s = 0; s < nSp; s++) {
      for (int sc = 0; sc < nScans; sc++) {
        size_t loc = g * maxNHKLs * nScans + s * nScans + sc;
        if (sinoArr[loc] > 0)
          filled++;
      }
    }
    printf("  Grain %zu: %d/%d cells filled (%.1f%%)\n", g, filled,
           nSp * nScans, 100.0 * filled / (nSp * nScans));
  }

  /* Show per-scan spot counts in Spots.bin */
  {
    int *scanCounts = calloc(nScans, sizeof(int));
    for (size_t i = 0; i < nSpotsAll; i++)
      scanCounts[(int)allSpots[SPOTS_ARRAY_COLS * i + 9]]++;
    printf("  Spots per scan (first 5): ");
    for (int i = 0; i < 5 && i < nScans; i++)
      printf("scan%d=%d ", i, scanCounts[i]);
    printf("...\n");
    free(scanCounts);
  }

  /* Show match details for grain 0 spot 0 */
  if (spotList->nSpots > 0) {
    SpotData *sp0 = &spotList->spotData[0];
    printf("  Grain 0 Spot 0: omega=%.3f eta=%.3f ring=%d\n", sp0->omega,
           sp0->eta, sp0->ringNr);
    printf("  Matches in first 3 scans:\n");
    for (int sc = 0; sc < 3 && sc < nScans; sc++) {
      int nMatch = 0;
      for (size_t i = 0; i < nSpotsAll; i++) {
        if ((int)allSpots[SPOTS_ARRAY_COLS * i + 9] != sc)
          continue;
        if ((int)allSpots[SPOTS_ARRAY_COLS * i + 5] != sp0->ringNr)
          continue;
        double dOme = fabs(allSpots[SPOTS_ARRAY_COLS * i + 2] - sp0->omega);
        double dEta = fabs(allSpots[SPOTS_ARRAY_COLS * i + 6] - sp0->eta);
        if (dOme < tolOme && dEta < tolEta) {
          printf("    scan=%d spotIdx=%zu omega=%.3f eta=%.3f I=%.1f "
                 "dOme=%.3f dEta=%.3f\n",
                 sc, i, allSpots[SPOTS_ARRAY_COLS * i + 2],
                 allSpots[SPOTS_ARRAY_COLS * i + 6],
                 allSpots[SPOTS_ARRAY_COLS * i + 3], dOme, dEta);
          nMatch++;
        }
      }
      if (nMatch == 0)
        printf("    scan=%d: NO MATCH\n", sc);
    }
  }
  printf("=================================\n\n");

  /* Calculate average omega angles */
  for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
    for (size_t spotIdx = 0; spotIdx < (size_t)maxNHKLs; spotIdx++) {
      size_t maxIntIdx = grainIdx * maxNHKLs + spotIdx;
      if (countOmeArr[maxIntIdx] > 0) {
        omeArr[maxIntIdx] = sumOmeArr[maxIntIdx] / countOmeArr[maxIntIdx];
      }
    }
  }

  /* Sort spots by omega angle for each grain */
  for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
    SinoSortData *sortData = malloc(maxNHKLs * sizeof(*sortData));
    if (!sortData) {
      log_error("Failed to allocate memory for sorting data for grain %zu",
                grainIdx);
      continue;
    }

    /* Also allocate parallel arrays for spotID and metadata sorting */
    int **sortSpotIDs = malloc(maxNHKLs * sizeof(int *));
    double **sortSpotMeta = malloc(maxNHKLs * sizeof(double *));

    /* Count valid spots */
    int nValidSpots = 0;
    for (int spotIdx = 0; spotIdx < maxNHKLs; spotIdx++) {
      /* Check if this is a valid spot (omega value set) */
      if (omeArr[grainIdx * maxNHKLs + spotIdx] > -9999.0) {
        sortData[nValidSpots].angle = omeArr[grainIdx * maxNHKLs + spotIdx];
        sortData[nValidSpots].intensities = calloc(nScans, sizeof(double));
        sortSpotIDs[nValidSpots] = calloc(nScans, sizeof(int));
        sortSpotMeta[nValidSpots] =
            calloc(nScans * SPOT_META_COLS, sizeof(double));

        if (!sortData[nValidSpots].intensities || !sortSpotIDs[nValidSpots] ||
            !sortSpotMeta[nValidSpots]) {
          log_error("Failed to allocate memory for sort data for grain "
                    "%zu, spot %d",
                    grainIdx, spotIdx);

          /* Clean up previously allocated arrays */
          for (int k = 0; k < nValidSpots; k++) {
            free(sortData[k].intensities);
            free(sortSpotIDs[k]);
            free(sortSpotMeta[k]);
          }
          free(sortData);
          free(sortSpotIDs);
          free(sortSpotMeta);
          continue;
        }

        /* Copy intensities and spotIDs for this spot across all scans */
        for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
          size_t loc =
              grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx;
          sortData[nValidSpots].intensities[scanIdx] = sinoArr[loc];
          sortSpotIDs[nValidSpots][scanIdx] = spotIDArr[loc];
          for (int m = 0; m < SPOT_META_COLS; m++)
            sortSpotMeta[nValidSpots][scanIdx * SPOT_META_COLS + m] =
                spotMetaArr[loc * SPOT_META_COLS + m];
        }

        nValidSpots++;
      }
    }

    /* Sort by omega angle */
    qsort(sortData, nValidSpots, sizeof(SinoSortData), compare_sino_data);

    /* Copy sorted data back to original arrays */
    for (int spotIdx = 0; spotIdx < nValidSpots; spotIdx++) {
      /* Store sorted omega values */
      omeArr[grainIdx * maxNHKLs + spotIdx] = sortData[spotIdx].angle;

      /* Store sorted intensity and spotID values */
      for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
        size_t loc = grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx;
        sinoArr[loc] = sortData[spotIdx].intensities[scanIdx];
        spotIDArr[loc] = sortSpotIDs[spotIdx][scanIdx];
        for (int m = 0; m < SPOT_META_COLS; m++)
          spotMetaArr[loc * SPOT_META_COLS + m] =
              sortSpotMeta[spotIdx][scanIdx * SPOT_META_COLS + m];
      }

      /* Free arrays */
      free(sortData[spotIdx].intensities);
      free(sortSpotIDs[spotIdx]);
      free(sortSpotMeta[spotIdx]);
    }

    /* Set remaining spots to invalid */
    for (int spotIdx = nValidSpots; spotIdx < maxNHKLs; spotIdx++) {
      omeArr[grainIdx * maxNHKLs + spotIdx] = -10000.0;
      for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
        spotIDArr[grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx] =
            -1;
        for (int m = 0; m < SPOT_META_COLS; m++)
          spotMetaArr[(grainIdx * maxNHKLs * nScans + spotIdx * nScans +
                       scanIdx) *
                          SPOT_META_COLS +
                      m] = NAN;
      }
    }

    /* Free sort data arrays */
    free(sortData);
    free(sortSpotIDs);
    free(sortSpotMeta);
  }

  /* --- Make a raw copy before any transforms --- */
  double *rawSinoArr = malloc(szSino * sizeof(double));
  memcpy(rawSinoArr, sinoArr, szSino * sizeof(double));

  /* --- Apply user-requested transforms to sinoArr --- */
  for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
    for (size_t spotIdx = 0; spotIdx < (size_t)maxNHKLs; spotIdx++) {
      double maxIntensity = maxIntArr[grainIdx * maxNHKLs + spotIdx];
      for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
        size_t locThis =
            grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx;
        if (sinoArr[locThis] > 0) {
          if (normalizeSino && maxIntensity > 0) {
            sinoArr[locThis] /= maxIntensity;
          }
          if (absTransform) {
            sinoArr[locThis] *= -1;
            sinoArr[locThis] = exp(sinoArr[locThis]);
          }
        }
      }
    }
  }

  /* Save results to files */
  char sinoFN[MAX_PATH_LEN], omeFN[MAX_PATH_LEN], HKLsFN[MAX_PATH_LEN];
  sprintf(sinoFN, "%s/sinos_%zu_%d_%d.bin", outputFolder,
          uniqueResult->nUniques, maxNHKLs, nScans);
  sprintf(omeFN, "%s/omegas_%zu_%d.bin", outputFolder, uniqueResult->nUniques,
          maxNHKLs);
  sprintf(HKLsFN, "%s/nrHKLs_%zu.bin", outputFolder, uniqueResult->nUniques);

  FILE *sinoF = fopen(sinoFN, "wb");
  FILE *omeF = fopen(omeFN, "wb");
  FILE *HKLsF = fopen(HKLsFN, "wb");

  if (sinoF && omeF && HKLsF) {
    /* Write binary data to output files */
    fwrite(sinoArr,
           uniqueResult->nUniques * maxNHKLs * nScans * sizeof(*sinoArr), 1,
           sinoF);
    fwrite(omeArr, uniqueResult->nUniques * maxNHKLs * sizeof(*omeArr), 1,
           omeF);
    fwrite(nrHKLsPerGrain, uniqueResult->nUniques * sizeof(*nrHKLsPerGrain), 1,
           HKLsF);

    printf("Sinogram data saved to:\n");
    printf("  %s\n", sinoFN);
    printf("  %s\n", omeFN);
    printf("  %s\n", HKLsFN);
  } else {
    log_error("Failed to open one or more output files for writing");
    if (!sinoF)
      log_error("Could not open %s", sinoFN);
    if (!omeF)
      log_error("Could not open %s", omeFN);
    if (!HKLsF)
      log_error("Could not open %s", HKLsFN);
  }

  /* Clean up */
  if (sinoF)
    fclose(sinoF);
  if (omeF)
    fclose(omeF);
  if (HKLsF)
    fclose(HKLsF);

  /* --- Generate and save all 4 combinations for visualization --- */
  // Combination labels: raw, norm, abs, normabs
  const char *comboNames[4] = {"raw", "norm", "abs", "normabs"};
  int comboNorm[4] = {0, 1, 0, 1};
  int comboAbs[4] = {0, 0, 1, 1};

  for (int combo = 0; combo < 4; combo++) {
    double *comboArr = malloc(szSino * sizeof(double));
    memcpy(comboArr, rawSinoArr, szSino * sizeof(double));

    // Apply this combo's transforms
    for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
      for (size_t spotIdx = 0; spotIdx < (size_t)maxNHKLs; spotIdx++) {
        double maxIntensity = maxIntArr[grainIdx * maxNHKLs + spotIdx];
        for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
          size_t locThis =
              grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx;
          if (comboArr[locThis] > 0) {
            if (comboNorm[combo] && maxIntensity > 0)
              comboArr[locThis] /= maxIntensity;
            if (comboAbs[combo]) {
              comboArr[locThis] *= -1;
              comboArr[locThis] = exp(comboArr[locThis]);
            }
          }
        }
      }
    }

    char comboFN[MAX_PATH_LEN];
    sprintf(comboFN, "%s/sinos_%s_%zu_%d_%d.bin", outputFolder,
            comboNames[combo], uniqueResult->nUniques, maxNHKLs, nScans);
    FILE *comboF = fopen(comboFN, "wb");
    if (comboF) {
      fwrite(comboArr, szSino * sizeof(double), 1, comboF);
      fclose(comboF);
    }
    free(comboArr);
  }
  printf("Saved all 4 sinogram combinations.\n");

  /* Save spotID mapping (same shape as sinograms, for patch extraction) */
  char spotMapFN[MAX_PATH_LEN];
  sprintf(spotMapFN, "%s/spotMapping_%zu_%d_%d.bin", outputFolder,
          uniqueResult->nUniques, maxNHKLs, nScans);
  FILE *spotMapF = fopen(spotMapFN, "wb");
  if (spotMapF) {
    fwrite(spotIDArr, szSino * sizeof(int), 1, spotMapF);
    fclose(spotMapF);
    printf("Saved spot ID mapping: %s\n", spotMapFN);
  }

  /* Save per-cell metadata (eta, 2theta, yCen, zCen) for viewer hover */
  char spotMetaFN[MAX_PATH_LEN];
  sprintf(spotMetaFN, "%s/spotMeta_%zu_%d_%d.bin", outputFolder,
          uniqueResult->nUniques, maxNHKLs, nScans);
  FILE *smF = fopen(spotMetaFN, "wb");
  if (smF) {
    fwrite(spotMetaArr, szSino * SPOT_META_COLS * sizeof(double), 1, smF);
    fclose(smF);
    printf("Saved spot metadata: %s\n", spotMetaFN);
  }

  /* Return maxNHKLs to caller */
  if (outMaxNHKLs)
    *outMaxNHKLs = maxNHKLs;

  free(rawSinoArr);
  free(sinoArr);
  free(spotIDArr);
  free(spotMetaArr);
  free(omeArr);
  free(sumOmeArr);
  free(countOmeArr);
  free(maxIntArr);
  free(nrHKLsPerGrain);
}

/* ======================================================================
 * PATCH EXTRACTION — Helper functions (from PeaksFittingOMPZarrRefactor)
 * ====================================================================== */

#define PATCH_HALF_SIZE 10
#define PATCH_SIZE (2 * PATCH_HALF_SIZE + 1) /* 21 */

/* Pixel type enum matching PeaksFitting */
typedef enum {
  PX_UINT16 = 0,
  PX_UINT32 = 1,
  PX_INT32 = 2,
  PX_FLOAT = 3,
  PX_DOUBLE = 4
} PatchPixelType;

static inline void patch_convertPixelsToDouble(const void *rawData,
                                               double *dest, int nPixels,
                                               PatchPixelType pxType) {
  switch (pxType) {
  case PX_UINT32:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const uint32_t *)rawData)[i];
    break;
  case PX_INT32:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const int32_t *)rawData)[i];
    break;
  case PX_FLOAT:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const float *)rawData)[i];
    break;
  case PX_DOUBLE:
    memcpy(dest, rawData, (size_t)nPixels * sizeof(double));
    break;
  case PX_UINT16:
  default:
    for (int i = 0; i < nPixels; i++)
      dest[i] = (double)((const uint16_t *)rawData)[i];
    break;
  }
}

static inline void patch_makeSquareImage(int nrPixels, int nrPixelsY,
                                         int nrPixelsZ, double *inImage,
                                         double *outImage) {
  memset(outImage, 0, (size_t)nrPixels * nrPixels * sizeof(double));
  if (nrPixelsY == nrPixelsZ) {
    memcpy(outImage, inImage, (size_t)nrPixels * nrPixels * sizeof(double));
  } else if (nrPixelsY > nrPixelsZ) {
    memcpy(outImage, inImage, (size_t)nrPixelsY * nrPixelsZ * sizeof(double));
  } else {
    for (int i = 0; i < nrPixelsZ; i++) {
      memcpy(outImage + (size_t)i * nrPixels, inImage + (size_t)i * nrPixelsY,
             (size_t)nrPixelsY * sizeof(double));
    }
  }
}

static inline void patch_applyImTransOpt(int nrTransOpt, int transOpt[10],
                                         double *image, int nrPixels,
                                         double *temp1, double *temp2) {
  size_t nSq = (size_t)nrPixels * nrPixels;
  memcpy(temp1, image, nSq * sizeof(double));

  for (int k = 0; k < nrTransOpt; k++) {
    switch (transOpt[k]) {
    case 1: /* Flip horizontal (Y) */
      for (int l = 0; l < nrPixels; l++)
        for (int m = 0; m < nrPixels; m++)
          temp2[l * nrPixels + m] = temp1[l * nrPixels + (nrPixels - m - 1)];
      break;
    case 2: /* Flip vertical (Z) */
      for (int l = 0; l < nrPixels; l++)
        for (int m = 0; m < nrPixels; m++)
          temp2[l * nrPixels + m] = temp1[(nrPixels - l - 1) * nrPixels + m];
      break;
    case 3: /* Transpose */
      for (int l = 0; l < nrPixels; l++)
        for (int m = 0; m < nrPixels; m++)
          temp2[l * nrPixels + m] = temp1[m * nrPixels + l];
      break;
    case 0:
    default:
      memcpy(temp2, temp1, nSq * sizeof(double));
      break;
    }
    double *swap = temp1;
    temp1 = temp2;
    temp2 = swap;
  }
  memcpy(image, temp1, nSq * sizeof(double));
}

static inline void patch_transposeMatrix(double *x, int n, double *y) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      y[i * n + j] = x[j * n + i];
}

/* ======================================================================
 * extract_patches — Read zarr frames and extract 21x21 patches
 * ====================================================================== */

/**
 * Extract intensity patches for each sinogram cell.
 *
 * For each (grain, spot, scan) with a matched spot:
 * 1. Translate globalSpotID → per-scan origID via IDsMergedScanning.csv
 * 2. Look up pixel position from Result_*.csv
 * 3. Read the zarr frame, apply PeaksFitting transforms
 * 4. Extract 21×21 patch centered on pixel position
 * 5. Save all patches as a float32 binary
 */
void extract_patches(const char *topdir, const char *outputFolder,
                     const char *paramFile, size_t nGrs, int maxNHKLs,
                     int nScans, int numProcs) {
  printf("\n=== Extracting intensity patches ===\n");

  /* --- Read spot mapping from file --- */
  char spotMapFN[MAX_PATH_LEN];
  sprintf(spotMapFN, "%s/spotMapping_%zu_%d_%d.bin", outputFolder, nGrs,
          maxNHKLs, nScans);
  size_t szSino = nGrs * maxNHKLs * nScans;
  int *spotIDArr = malloc(szSino * sizeof(int));
  if (!spotIDArr) {
    log_error("Failed to allocate spotIDArr");
    return;
  }

  FILE *smf = fopen(spotMapFN, "rb");
  if (!smf) {
    log_error("Cannot open %s", spotMapFN);
    free(spotIDArr);
    return;
  }
  fread(spotIDArr, sizeof(int), szSino, smf);
  fclose(smf);

  /* --- Read IDsMergedScanning.csv --- */
  char idMapFN[MAX_PATH_LEN];
  sprintf(idMapFN, "%s/IDsMergedScanning.csv", topdir);
  FILE *idMapF = fopen(idMapFN, "r");
  if (!idMapF) {
    log_error("Cannot open %s", idMapFN);
    free(spotIDArr);
    return;
  }

  /* Count lines */
  int nIDEntries = 0;
  char line[1024];
  while (fgets(line, sizeof(line), idMapF)) {
    if (line[0] != 'N')
      nIDEntries++; /* Skip header */
  }
  rewind(idMapF);

  /* Allocate mapping arrays */
  int *idMap_origID = calloc(nIDEntries + 1, sizeof(int)); /* 1-indexed */
  int *idMap_scanNr = calloc(nIDEntries + 1, sizeof(int));
  if (!idMap_origID || !idMap_scanNr) {
    log_error("Failed to allocate ID mapping arrays");
    fclose(idMapF);
    free(spotIDArr);
    free(idMap_origID);
    free(idMap_scanNr);
    return;
  }

  /* Parse the CSV */
  while (fgets(line, sizeof(line), idMapF)) {
    int newID, origID, scanNr;
    if (sscanf(line, "%d,%d,%d", &newID, &origID, &scanNr) == 3) {
      if (newID >= 0 && newID <= nIDEntries) {
        idMap_origID[newID] = origID;
        idMap_scanNr[newID] = scanNr;
      }
    }
  }
  fclose(idMapF);
  printf("  Loaded IDsMergedScanning: %d entries\n", nIDEntries);

  /* Quick diagnostic: show a sample of idMap values */
  {
    int sampleCount = 0;
    for (int i = 1; i <= nIDEntries && sampleCount < 3; i++) {
      if (idMap_origID[i] != 0 || idMap_scanNr[i] != 0) {
        printf("    IDMap[%d]: origID=%d scanNr=%d\n", i, idMap_origID[i],
               idMap_scanNr[i]);
        sampleCount++;
      }
    }
  }

  /* --- Read paramstest.txt for StartFileNrFirstLayer, NrFilesPerSweep, etc.
   * --- */
  char paramsFN[MAX_PATH_LEN];
  sprintf(paramsFN, "%s/%s", topdir, paramFile);
  FILE *pf = fopen(paramsFN, "r");
  if (!pf) {
    log_error("Cannot open %s", paramsFN);
    free(spotIDArr);
    free(idMap_origID);
    free(idMap_scanNr);
    return;
  }

  int startFileNr = 0, nrFilesPerSweep = 0;
  int startNr = 1, endNr = 0; /* StartNr/EndNr for Result CSV naming */
  double omegaStart = 0, omegaStep = 0;
  char fileStem[MAX_PATH_LEN] = "";
  int imTransOpt[10] = {0};
  int nImTransOpt = 0;
  int nrPixels = 0, nrPixelsY = 0, nrPixelsZ = 0;
  int skipFrame = 0;
  int padding = 6; /* default zero-padding for file numbers */
  char dummy[1024];

  while (fgets(line, sizeof(line), pf)) {
    if (line[0] == '#')
      continue; /* skip comments */
    if (strncmp(line, "StartFileNrFirstLayer ", 21) == 0)
      sscanf(line, "%s %d", dummy, &startFileNr);
    else if (strncmp(line, "NrFilesPerSweep ", 16) == 0)
      sscanf(line, "%s %d", dummy, &nrFilesPerSweep);
    else if (strncmp(line, "StartNr ", 8) == 0)
      sscanf(line, "%s %d", dummy, &startNr);
    else if (strncmp(line, "EndNr ", 6) == 0)
      sscanf(line, "%s %d", dummy, &endNr);
    else if (strncmp(line, "OmegaFirstFile ", 15) == 0 ||
             strncmp(line, "OmegaStart ", 11) == 0)
      sscanf(line, "%s %lf", dummy, &omegaStart);
    else if (strncmp(line, "OmegaStep ", 10) == 0)
      sscanf(line, "%s %lf", dummy, &omegaStep);
    else if (strncmp(line, "FileStem ", 9) == 0)
      sscanf(line, "%s %s", dummy, fileStem);
    else if (strncmp(line, "ImTransOpt ", 11) == 0) {
      if (nImTransOpt < 10)
        sscanf(line, "%s %d", dummy, &imTransOpt[nImTransOpt++]);
    } else if (strncmp(line, "NrPixels ", 9) == 0)
      sscanf(line, "%s %d", dummy, &nrPixels);
    else if (strncmp(line, "NrPixelsY ", 10) == 0 ||
             strncmp(line, "numPxY ", 7) == 0)
      sscanf(line, "%s %d", dummy, &nrPixelsY);
    else if (strncmp(line, "NrPixelsZ ", 10) == 0 ||
             strncmp(line, "numPxZ ", 7) == 0)
      sscanf(line, "%s %d", dummy, &nrPixelsZ);
    else if (strncmp(line, "SkipFrame ", 10) == 0)
      sscanf(line, "%s %d", dummy, &skipFrame);
    else if (strncmp(line, "Padding ", 8) == 0)
      sscanf(line, "%s %d", dummy, &padding);
  }
  fclose(pf);

  if (nrPixels == 0)
    nrPixels = (nrPixelsY > nrPixelsZ) ? nrPixelsY : nrPixelsZ;
  printf("  Params: StartNr=%d NrFiles=%d OmeStart=%.1f OmeStep=%.3f "
         "NrPx=%d(%dx%d) nImTrans=%d Padding=%d\n",
         startFileNr, nrFilesPerSweep, omegaStart, omegaStep, nrPixels,
         nrPixelsY, nrPixelsZ, nImTransOpt, padding);

  /* --- Allocate output patches array (float32) --- */
  size_t patchPixels = PATCH_SIZE * PATCH_SIZE; /* 21*21 = 441 */
  size_t totalPatches = (size_t)nGrs * maxNHKLs * nScans;
  float *patchesArr = calloc(totalPatches * patchPixels, sizeof(float));
  double *spotPosArr = calloc(totalPatches * 2, sizeof(double));
  if (!patchesArr || !spotPosArr) {
    log_error("Failed to allocate patches array (%.1f MB)",
              (double)(totalPatches * patchPixels * sizeof(float)) /
                  (1024.0 * 1024.0));
    free(spotIDArr);
    free(idMap_origID);
    free(idMap_scanNr);
    free(patchesArr);
    free(spotPosArr);
    return;
  }
  /* Initialize spotPos to -1 */
  for (size_t i = 0; i < totalPatches * 2; i++)
    spotPosArr[i] = -1.0;

  printf("  Allocated patches: %.1f MB\n",
         (double)(totalPatches * patchPixels * sizeof(float)) /
             (1024.0 * 1024.0));

  /* --- Per-scan loop: read Result_*.csv and zarr frames --- */
  int patchesExtracted = 0;

  /* Pre-allocate per-thread frame buffers outside the loop */
  size_t rawFrameSize = (size_t)nrPixelsY * nrPixelsZ;
  int maxPxSize = 8; /* worst case: double */
  size_t maxRawBytes = rawFrameSize * maxPxSize;
  int sqPx = nrPixels;

  /* Allocate one set of buffers per thread */
  int nThreads = numProcs;
  void **t_rawBuf = calloc(nThreads, sizeof(void *));
  double **t_asymBuf = calloc(nThreads, sizeof(double *));
  double **t_squareBuf = calloc(nThreads, sizeof(double *));
  double **t_transBuf = calloc(nThreads, sizeof(double *));
  double **t_temp1 = calloc(nThreads, sizeof(double *));
  double **t_temp2 = calloc(nThreads, sizeof(double *));
  for (int t = 0; t < nThreads; t++) {
    t_rawBuf[t] = malloc(maxRawBytes + 4096);
    t_asymBuf[t] = malloc(rawFrameSize * sizeof(double));
    t_squareBuf[t] = malloc((size_t)sqPx * sqPx * sizeof(double));
    t_transBuf[t] = malloc((size_t)sqPx * sqPx * sizeof(double));
    t_temp1[t] = malloc((size_t)sqPx * sqPx * sizeof(double));
    t_temp2[t] = malloc((size_t)sqPx * sqPx * sizeof(double));
  }

#pragma omp parallel for schedule(dynamic) num_threads(numProcs)               \
    reduction(+ : patchesExtracted)
  for (int scanNr = 0; scanNr < nScans; scanNr++) {
    char tline[1024]; /* Thread-private line buffer for CSV reading */
    int thisStartNr = startFileNr + scanNr * nrFilesPerSweep;

    /* Collect all sinogram cells for this scan that have a matched spot */
    typedef struct {
      int grainNr;
      int spotNr;
      int globalID;
    } CellInfo;
    CellInfo *cells = malloc(nGrs * maxNHKLs * sizeof(CellInfo));
    int nCells = 0;

    for (size_t g = 0; g < nGrs; g++) {
      for (int s = 0; s < maxNHKLs; s++) {
        size_t loc = g * maxNHKLs * nScans + s * nScans + scanNr;
        int gid = spotIDArr[loc];
        if (gid > 0 && gid <= nIDEntries) {
          /* Verify this spot belongs to this scan */
          if (idMap_scanNr[gid] == scanNr) {
            cells[nCells].grainNr = (int)g;
            cells[nCells].spotNr = s;
            cells[nCells].globalID = gid;
            nCells++;
          }
        }
      }
    }

    if (nCells == 0) {
      if (scanNr < 3) {
        /* Diagnostic: why no cells? */
        int nSpotIDs = 0, nInRange = 0, nScanMatch = 0;
        for (size_t g = 0; g < nGrs; g++) {
          for (int s = 0; s < maxNHKLs; s++) {
            size_t loc = g * maxNHKLs * nScans + s * nScans + scanNr;
            int gid = spotIDArr[loc];
            if (gid > 0)
              nSpotIDs++;
            if (gid > 0 && gid <= nIDEntries) {
              nInRange++;
              if (idMap_scanNr[gid] == scanNr)
                nScanMatch++;
            }
          }
        }
        printf("  Scan %d diag: spotIDs>0=%d inRange=%d scanMatch=%d (looking "
               "for scanNr=%d)\n",
               scanNr, nSpotIDs, nInRange, nScanMatch, scanNr);
        if (nInRange > 0 && nScanMatch == 0) {
          /* Show what scanNr values these IDs actually have */
          for (size_t g = 0; g < nGrs && g < 1; g++) {
            for (int s = 0; s < maxNHKLs && s < 5; s++) {
              size_t loc = g * maxNHKLs * nScans + s * nScans + scanNr;
              int gid = spotIDArr[loc];
              if (gid > 0 && gid <= nIDEntries) {
                printf("    gid=%d → origID=%d scanNr=%d\n", gid,
                       idMap_origID[gid], idMap_scanNr[gid]);
              }
            }
          }
        }
      }
      free(cells);
      continue;
    }

    /* --- Read per-scan Result_*.csv for pixel positions --- */
    char resultFN[MAX_PATH_LEN];
    int resEndNr = (endNr > 0) ? endNr : (startNr + nrFilesPerSweep - 1);
    sprintf(resultFN, "%s/%d/Result_StartNr_%d_EndNr_%d.csv", topdir,
            thisStartNr, startNr, resEndNr);
    FILE *rf = fopen(resultFN, "r");
    if (!rf) {
      /* Try with skipFrame */
      resEndNr =
          (endNr > 0) ? endNr : (startNr + nrFilesPerSweep - 1 - skipFrame);
      sprintf(resultFN, "%s/%d/Result_StartNr_%d_EndNr_%d.csv", topdir,
              thisStartNr, startNr, resEndNr);
      rf = fopen(resultFN, "r");
    }
    if (!rf) {
      if (scanNr < 3)
        log_error("Cannot open Result CSV for scan %d: %s", scanNr, resultFN);
      free(cells);
      continue;
    }

    /* Parse Result CSV: spotID → (yCen, zCen, omega) */
    /* Use a simple array indexed by spotID (max ~20k per scan) */
    int maxResultID = 0;
    /* First pass: find max ID */
    while (fgets(tline, sizeof(tline), rf)) {
      if (tline[0] == 'S' || tline[0] == '#')
        continue;
      int sid;
      if (sscanf(tline, "%d", &sid) == 1 && sid > maxResultID)
        maxResultID = sid;
    }
    rewind(rf);

    double *rYCen = calloc(maxResultID + 1, sizeof(double));
    double *rZCen = calloc(maxResultID + 1, sizeof(double));
    double *rOmega = calloc(maxResultID + 1, sizeof(double));
    int *rValid = calloc(maxResultID + 1, sizeof(int));

    while (fgets(tline, sizeof(tline), rf)) {
      if (tline[0] == 'S' || tline[0] == '#')
        continue;
      int sid;
      double intInt, omega, ycen, zcen;
      if (sscanf(tline, "%d %lf %lf %lf %lf", &sid, &intInt, &omega, &ycen,
                 &zcen) == 5) {
        if (sid >= 0 && sid <= maxResultID) {
          rYCen[sid] = ycen;
          rZCen[sid] = zcen;
          rOmega[sid] = omega;
          rValid[sid] = 1;
        }
      }
    }
    fclose(rf);

    if (scanNr < 3)
      printf("  Scan %d: nCells=%d maxResultID=%d nRequests pending...\n",
             scanNr, nCells, maxResultID);

    /* --- Determine which frames we need --- */
    FrameRequest *requests = malloc(nCells * sizeof(FrameRequest));
    int nRequests = 0;

    for (int c = 0; c < nCells; c++) {
      int origID = idMap_origID[cells[c].globalID];
      if (origID >= 0 && origID <= maxResultID && rValid[origID]) {
        double omega = rOmega[origID];
        int frameIdx = 0;
        if (fabs(omegaStep) > 1e-9) {
          frameIdx = (int)round((omega - omegaStart) / omegaStep);
        }
        if (frameIdx < 0)
          frameIdx += nrFilesPerSweep;
        if (frameIdx < 0)
          frameIdx = 0;
        if (frameIdx >= nrFilesPerSweep)
          frameIdx = nrFilesPerSweep - 1;

        requests[nRequests].frameIdx = frameIdx;
        requests[nRequests].yCen = rYCen[origID];
        requests[nRequests].zCen = rZCen[origID];
        requests[nRequests].cellIdx = c;
        nRequests++;

        /* Store pixel position in output */
        size_t posLoc =
            ((size_t)cells[c].grainNr * maxNHKLs + cells[c].spotNr) * nScans +
            scanNr;
        spotPosArr[posLoc * 2 + 0] = rYCen[origID];
        spotPosArr[posLoc * 2 + 1] = rZCen[origID];
      }
    }

    free(rYCen);
    free(rZCen);
    free(rOmega);
    free(rValid);

    if (nRequests == 0) {
      free(cells);
      free(requests);
      continue;
    }

    /* --- Open zarr zip file --- */
    char zarFN[MAX_PATH_LEN];
    sprintf(zarFN, "%s/%d/%s_%0*d.MIDAS.zip", topdir, thisStartNr, fileStem,
            padding, thisStartNr);
    int errorp = 0;
    zip_t *archive = zip_open(zarFN, 0, &errorp);
    if (!archive) {
      if (scanNr < 3)
        log_error("Cannot open zarr %s (err=%d)", zarFN, errorp);
      free(cells);
      free(requests);
      continue;
    }
    int dataLoc = -1;
    int nEntries = (int)zip_get_num_entries(archive, 0);
    for (int e = 0; e < nEntries; e++) {
      const char *name = zip_get_name(archive, e, 0);
      if (name && strstr(name, "exchange/data/0.0.0")) {
        dataLoc = e;
        break;
      }
    }
    if (dataLoc < 0) {
      /* Try single-chunk format */
      for (int e = 0; e < nEntries; e++) {
        const char *name = zip_get_name(archive, e, 0);
        if (name && strstr(name, "exchange/data/0")) {
          dataLoc = e;
          break;
        }
      }
    }

    /* Determine pixel type from zarr metadata */
    PatchPixelType pxType = PX_UINT16;
    for (int e = 0; e < nEntries; e++) {
      const char *name = zip_get_name(archive, e, 0);
      if (name && strstr(name, "exchange/data/.zarray")) {
        zip_file_t *zf = zip_fopen_index(archive, e, 0);
        if (zf) {
          char buf[4096];
          int nr = (int)zip_fread(zf, buf, sizeof(buf) - 1);
          buf[nr] = '\0';
          zip_fclose(zf);
          if (strstr(buf, "float32") || strstr(buf, "<f4"))
            pxType = PX_FLOAT;
          else if (strstr(buf, "float64") || strstr(buf, "<f8"))
            pxType = PX_DOUBLE;
          else if (strstr(buf, "uint32") || strstr(buf, "<u4"))
            pxType = PX_UINT32;
          else if (strstr(buf, "int32") || strstr(buf, "<i4"))
            pxType = PX_INT32;
        }
        break;
      }
    }

    if (dataLoc < 0) {
      log_error("Cannot find data in zarr %s", zarFN);
      zip_close(archive);
      free(cells);
      free(requests);
      continue;
    }

    /* Process each unique frame needed */
    int lastFrame = -1;
    /* Sort requests by frameIdx to minimize re-reads */
    qsort(requests, nRequests, sizeof(FrameRequest), cmp_frame_requests);

    /* Get per-thread buffers */
    int tid = omp_get_thread_num();
    void *rawBuf = t_rawBuf[tid];
    double *asymBuf = t_asymBuf[tid];
    double *squareBuf = t_squareBuf[tid];
    double *transBuf = t_transBuf[tid];
    double *temp1 = t_temp1[tid];
    double *temp2 = t_temp2[tid];

    for (int r = 0; r < nRequests; r++) {
      int frameIdx = requests[r].frameIdx;

      /* Read and transform frame only if different from last */
      if (frameIdx != lastFrame) {
        int zipIdx = dataLoc + frameIdx;
        if (zipIdx >= nEntries) {
          continue;
        }

        zip_stat_t fileStat;
        zip_stat_init(&fileStat);
        if (zip_stat_index(archive, zipIdx, 0, &fileStat) != 0)
          continue;

        char *compressedBuf = malloc(fileStat.size);
        if (!compressedBuf)
          continue;

        zip_file_t *zf = zip_fopen_index(archive, zipIdx, 0);
        if (!zf) {
          free(compressedBuf);
          continue;
        }
        zip_fread(zf, compressedBuf, fileStat.size);
        zip_fclose(zf);

        /* Decompress with blosc */
        int decompSize =
            blosc1_decompress(compressedBuf, rawBuf, maxRawBytes + 4096);
        free(compressedBuf);
        if (decompSize <= 0)
          continue;

        /* Convert to double */
        patch_convertPixelsToDouble(rawBuf, asymBuf, rawFrameSize, pxType);

        /* Make square image */
        patch_makeSquareImage(nrPixels, nrPixelsY, nrPixelsZ, asymBuf,
                              squareBuf);

        /* Apply ImTransOpt */
        patch_applyImTransOpt(nImTransOpt, imTransOpt, squareBuf, nrPixels,
                              temp1, temp2);

        /* Transpose (always applied after ImTransOpt in PeaksFitting) */
        patch_transposeMatrix(squareBuf, nrPixels, transBuf);

        lastFrame = frameIdx;
      }

      /* Extract 21×21 patch from transBuf */
      int yCen = (int)round(requests[r].yCen);
      int zCen = (int)round(requests[r].zCen);

      int ci = requests[r].cellIdx;
      size_t patchLoc =
          ((size_t)cells[ci].grainNr * maxNHKLs + cells[ci].spotNr) * nScans +
          scanNr;
      float *outPatch = &patchesArr[patchLoc * patchPixels];

      for (int dz = -PATCH_HALF_SIZE; dz <= PATCH_HALF_SIZE; dz++) {
        for (int dy = -PATCH_HALF_SIZE; dy <= PATCH_HALF_SIZE; dy++) {
          int pz = zCen + dz;
          int py = yCen + dy;
          int patchRow = dz + PATCH_HALF_SIZE;
          int patchCol = dy + PATCH_HALF_SIZE;
          if (pz >= 0 && pz < nrPixels && py >= 0 && py < nrPixels) {
            outPatch[patchRow * PATCH_SIZE + patchCol] =
                (float)transBuf[pz * nrPixels + py];
          }
        }
      }
      patchesExtracted++;
    }

    zip_close(archive);
    free(cells);
    free(requests);

    if (scanNr % 10 == 0 || scanNr == nScans - 1)
      printf("  Scan %d/%d: %d patches extracted so far\n", scanNr + 1, nScans,
             patchesExtracted);
  } /* end OMP parallel for */

  /* Free per-thread buffers */
  for (int t = 0; t < nThreads; t++) {
    free(t_rawBuf[t]);
    free(t_asymBuf[t]);
    free(t_squareBuf[t]);
    free(t_transBuf[t]);
    free(t_temp1[t]);
    free(t_temp2[t]);
  }
  free(t_rawBuf);
  free(t_asymBuf);
  free(t_squareBuf);
  free(t_transBuf);
  free(t_temp1);
  free(t_temp2);

  printf("  Total patches extracted: %d\n", patchesExtracted);

  /* --- Save patches binary --- */
  char patchesFN[MAX_PATH_LEN];
  sprintf(patchesFN, "%s/patches_%zu_%d_%d.bin", outputFolder, nGrs, maxNHKLs,
          nScans);
  FILE *pf2 = fopen(patchesFN, "wb");
  if (pf2) {
    fwrite(patchesArr, sizeof(float), totalPatches * patchPixels, pf2);
    fclose(pf2);
    printf("  Saved patches: %s (%.1f MB)\n", patchesFN,
           (double)(totalPatches * patchPixels * sizeof(float)) /
               (1024.0 * 1024.0));
  }

  /* --- Save spot positions binary --- */
  char spotPosFN[MAX_PATH_LEN];
  sprintf(spotPosFN, "%s/spotPositions_%zu_%d_%d.bin", outputFolder, nGrs,
          maxNHKLs, nScans);
  FILE *spf = fopen(spotPosFN, "wb");
  if (spf) {
    fwrite(spotPosArr, sizeof(double), totalPatches * 2, spf);
    fclose(spf);
    printf("  Saved spot positions: %s\n", spotPosFN);
  }

  free(spotIDArr);
  free(idMap_origID);
  free(idMap_scanNr);
  free(patchesArr);
  free(spotPosArr);
  printf("=== Patch extraction complete ===\n");
}

/**
 * Free resources allocated for SpotList and UniqueOrientationsResult
 *
 * @param spotList Structure containing processed spot data
 * @param uniqueResult Structure containing unique orientations information
 */
void free_resources(SpotList *spotList,
                    UniqueOrientationsResult *uniqueResult) {
  if (spotList) {
    free(spotList->spotData);
    spotList->spotData = NULL;
    spotList->nSpots = 0;
  }

  if (uniqueResult) {
    free(uniqueResult->uniqueKeyArr);
    free(uniqueResult->uniqueOrientArr);
    uniqueResult->uniqueKeyArr = NULL;
    uniqueResult->uniqueOrientArr = NULL;
    uniqueResult->nUniques = 0;
  }
}