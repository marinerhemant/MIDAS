/**
 * findSingleSolutionPF.c - Finding Single Solution in PF-HEDM
 * 
 * Improved version with better structure, memory management, 
 * error handling, and performance.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <libgen.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* Constants */
#define MAX_N_SOLUTIONS_PER_VOX 1000000
#define MAX_N_SPOTS_PER_GRAIN 5000
#define MAX_N_SPOTS_TOTAL 100000000
#define MAX_PATH_LEN 2048
#define SPOTS_ARRAY_COLS 10
#define KEY_ARRAY_COLS 4
#define ORIENT_ARRAY_COLS 9
#define CONF_IA_ARRAY_COLS 2
#define TMP_ARRAY_COLS 16

/* Structure definitions */
typedef struct {
    double omega;
    double eta;
    int ringNr;
    int mergedID;
    int scanNr;
    int grainNr;
    int spotNr;
} SpotData;

typedef struct {
    double *intensities;
    double angle;
} SinoSortData;

typedef struct {
    size_t nUniques;
    size_t *uniqueKeyArr;
    double *uniqueOrientArr;
} UniqueOrientationsResult;

typedef struct {
    SpotData *spotData;
    size_t nSpots;
} SpotList;

/* Global error handler */
static void log_error(const char *message, ...) {
    va_list args;
    va_start(args, message);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, message, args);
    fprintf(stderr, "\n");
    va_end(args);
}

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
double* read_memory_mapped_file(const char *filename, size_t *size_out);
int compare_sino_data(const void *a, const void *b);
void process_voxel(int voxNr, const char *folderName, int sgNr, double maxAng);
UniqueOrientationsResult find_unique_orientations(size_t *allKeyArr, double *allOrientationsArr, 
                                             size_t nScans, int sgNr, double maxAng);
SpotList process_spots(UniqueOrientationsResult *uniqueResult, const char *folderName, 
                      double *allSpots, size_t nSpotsAll, double tolOme, double tolEta);
void generate_sinograms(SpotList *spotList, UniqueOrientationsResult *uniqueResult, 
                       double *allSpots, size_t nSpotsAll, int nScans, double tolOme, 
                       double tolEta, const char *outputFolder, int numProcs);
void save_orientation_results(UniqueOrientationsResult *uniqueResult, const char *outputFolder);
void free_resources(SpotList *spotList, UniqueOrientationsResult *uniqueResult);

/* External function declarations - these are supplied from external files */
extern void OrientMat2Quat(const double *OM, double *Quat);
extern double GetMisOrientation(const double *Quat1, const double *Quat2, double *Axis, double *ang, int sgNr);

/* Entry point */
int main(int argc, char *argv[]) {
    double start_time = omp_get_wtime();
    printf("\n\n\t\tFinding Single Solution in PF-HEDM.\n\n");
    
    /* Parse command line arguments */
    if (argc != 8) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    char folderName[MAX_PATH_LEN];
    sprintf(folderName, "%s/Output/", argv[1]);
    
    int sgNr = atoi(argv[2]);
    double maxAng = atof(argv[3]);
    int nScans = atoi(argv[4]);
    int numProcs = atoi(argv[5]);
    double tolOme = atof(argv[6]);
    double tolEta = atof(argv[7]);
    
    /* Initialize arrays for storing data from all voxels */
    size_t *allKeyArr = calloc(nScans * nScans * KEY_ARRAY_COLS, sizeof(*allKeyArr));
    double *allOrientationsArr = calloc(nScans * nScans * (ORIENT_ARRAY_COLS + 1), sizeof(*allOrientationsArr));
    
    if (!allKeyArr || !allOrientationsArr) {
        fatal_error("Failed to allocate memory for key and orientation arrays");
    }
    
    /* Process each voxel in parallel */
    #pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
        process_voxel(voxNr, folderName, sgNr, maxAng);
    }
    
    /* Find unique orientations across all voxels */
    UniqueOrientationsResult uniqueResult = find_unique_orientations(
        allKeyArr, allOrientationsArr, nScans, sgNr, maxAng);
    
    printf("Number of unique orientations: %zu\n", uniqueResult.nUniques);
    
    /* Save unique orientations to file */
    save_orientation_results(&uniqueResult, argv[1]);
    
    /* Read all spots data */
    char spotsFilename[MAX_PATH_LEN];
    sprintf(spotsFilename, "%s/Spots.bin", argv[1]);
    
    /* Copy to shared memory for faster access */
    char command[MAX_PATH_LEN * 2];
    sprintf(command, "cp %s /dev/shm/", spotsFilename);
    system(command);
    sprintf(spotsFilename, "/dev/shm/Spots.bin");
    
    size_t spotsDataSize;
    double *allSpots = read_memory_mapped_file(spotsFilename, &spotsDataSize);
    size_t nSpotsAll = spotsDataSize / (SPOTS_ARRAY_COLS * sizeof(double));
    printf("nSpotsAll: %zu\n", nSpotsAll);
    
    /* Process spots for each unique orientation */
    SpotList spotList = process_spots(&uniqueResult, folderName, allSpots, 
                                     nSpotsAll, tolOme, tolEta);
    
    /* Generate sinograms */
    generate_sinograms(&spotList, &uniqueResult, allSpots, nSpotsAll, 
                      nScans, tolOme, tolEta, argv[1], numProcs);
    
    /* Clean up */
    free(allKeyArr);
    free(allOrientationsArr);
    free_resources(&spotList, &uniqueResult);
    
    munmap(allSpots, spotsDataSize);
    
    double elapsed = omp_get_wtime() - start_time;
    printf("Total processing time: %.2f seconds\n", elapsed);
    
    return EXIT_SUCCESS;
}

void print_usage(const char *program_name) {
    printf("Supply foldername spaceGroup, maxAng, NumberScans, nCPUs, tolOme, tolEta as arguments:\n"
           "%s foldername sgNum maxAngle nScans nCPUs tolOme tolEta\n"
           "\nThe indexing results need to be in folderName/Output\n", program_name);
}

double* read_memory_mapped_file(const char *filename, size_t *size_out) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fatal_error("open %s failed: %s", filename, strerror(errno));
    }
    
    struct stat s;
    int status = fstat(fd, &s);
    if (status < 0) {
        close(fd);
        fatal_error("stat %s failed: %s", filename, strerror(errno));
    }
    
    *size_out = s.st_size;
    double *mapped_data = mmap(0, *size_out, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd);
        fatal_error("mmap %s failed: %s", filename, strerror(errno));
    }
    
    return mapped_data;
}

int compare_sino_data(const void *a, const void *b) {
    const SinoSortData *ia = (const SinoSortData *)a;
    const SinoSortData *ib = (const SinoSortData *)b;
    
    if (ia->angle >= ib->angle) return 1;
    else return -1;
}

void process_voxel(int voxNr, const char *folderName, int sgNr, double maxAng) {
    char outKeyFN[MAX_PATH_LEN];
    sprintf(outKeyFN, "%s/UniqueIndexSingleKey.bin", folderName);
    
    int ib = open(outKeyFN, O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
    if (ib < 0) {
        log_error("Failed to open %s: %s", outKeyFN, strerror(errno));
        return;
    }
    
    FILE *valsF, *keyF;
    char valsFN[MAX_PATH_LEN], keyFN[MAX_PATH_LEN];
    sprintf(valsFN, "%s/IndexBest_voxNr_%06d.bin", folderName, voxNr);
    sprintf(keyFN, "%s/IndexKey_voxNr_%06d.txt", folderName, voxNr);
    
    valsF = fopen(valsFN, "rb");
    keyF = fopen(keyFN, "r");
    
    if (!keyF || !valsF) {
        if (!keyF) log_error("Could not open key file %s", keyFN);
        if (!valsF) log_error("Could not open vals file %s", valsFN);
        
        size_t outarr[5] = {0};
        pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
        close(ib);
        
        if (keyF) fclose(keyF);
        if (valsF) fclose(valsF);
        return;
    }
    
    /* Check if key file is empty */
    fseek(keyF, 0L, SEEK_END);
    size_t szt = ftell(keyF);
    rewind(keyF);
    
    if (szt == 0) {
        fclose(keyF);
        fclose(valsF);
        
        size_t outarr[5] = {0};
        pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
        close(ib);
        return;
    }
    
    /* Read key file */
    size_t *keys = calloc(MAX_N_SOLUTIONS_PER_VOX * KEY_ARRAY_COLS, sizeof(*keys));
    if (!keys) {
        log_error("Failed to allocate memory for keys");
        fclose(keyF);
        fclose(valsF);
        close(ib);
        return;
    }
    
    char aline[MAX_PATH_LEN];
    int nIDs = 0;
    
    while (fgets(aline, MAX_PATH_LEN, keyF) != NULL && nIDs < MAX_N_SOLUTIONS_PER_VOX) {
        sscanf(aline, "%zu %zu %zu %zu", 
               &keys[nIDs * KEY_ARRAY_COLS + 0],
               &keys[nIDs * KEY_ARRAY_COLS + 1],
               &keys[nIDs * KEY_ARRAY_COLS + 2],
               &keys[nIDs * KEY_ARRAY_COLS + 3]);
        nIDs++;
    }
    
    if (nIDs > 0) {
        size_t *resized_keys = realloc(keys, nIDs * KEY_ARRAY_COLS * sizeof(*keys));
        if (resized_keys) {
            keys = resized_keys;
        }
    }
    
    fclose(keyF);
    
    /* Read values file */
    double *OMArr = calloc(nIDs * ORIENT_ARRAY_COLS, sizeof(double));
    double *confIAArr = calloc(nIDs * CONF_IA_ARRAY_COLS, sizeof(double));
    double *tmpArr = calloc(nIDs * TMP_ARRAY_COLS, sizeof(double));
    
    if (!OMArr || !confIAArr || !tmpArr) {
        log_error("Failed to allocate memory for orientation arrays");
        free(keys);
        free(OMArr);
        free(confIAArr);
        free(tmpArr);
        fclose(valsF);
        close(ib);
        return;
    }
    
    fread(tmpArr, nIDs * TMP_ARRAY_COLS * sizeof(double), 1, valsF);
    fclose(valsF);
    
    /* Extract confidence and internal angle values */
    for (int i = 0; i < nIDs; i++) {
        confIAArr[i * CONF_IA_ARRAY_COLS + 0] = tmpArr[i * TMP_ARRAY_COLS + 15] / tmpArr[i * TMP_ARRAY_COLS + 14];
        confIAArr[i * CONF_IA_ARRAY_COLS + 1] = tmpArr[i * TMP_ARRAY_COLS + 1];
        
        for (int k = 0; k < ORIENT_ARRAY_COLS; k++) {
            OMArr[i * ORIENT_ARRAY_COLS + k] = tmpArr[i * TMP_ARRAY_COLS + 2 + k];
        }
    }
    
    /* Find best orientation */
    bool *markArr = calloc(nIDs, sizeof(*markArr));
    if (!markArr) {
        log_error("Failed to allocate memory for mark array");
        free(keys);
        free(OMArr);
        free(confIAArr);
        free(tmpArr);
        close(ib);
        return;
    }
    
    int bestRow = -1;
    double bestConf = -1, bestIA = 100;
    
    for (int i = 0; i < nIDs; i++) {
        if (markArr[i]) continue;
        
        if (confIAArr[i * CONF_IA_ARRAY_COLS + 0] < bestConf) continue;
        if (confIAArr[i * CONF_IA_ARRAY_COLS + 0] == bestConf && 
            confIAArr[i * CONF_IA_ARRAY_COLS + 1] > bestIA) continue;
        
        bestConf = confIAArr[i * CONF_IA_ARRAY_COLS + 0];
        bestIA = confIAArr[i * CONF_IA_ARRAY_COLS + 1];
        bestRow = i;
    }
    
    if (bestRow == -1) {
        /* No valid orientation found */
        size_t outarr[5] = {0};
        pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
    } else {
        /* Process unique orientations */
        for (int i = 0; i < nIDs; i++) markArr[i] = false;
        
        double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3], ang;
        size_t *uniqueArrThis = calloc(nIDs * KEY_ARRAY_COLS, sizeof(*uniqueArrThis));
        double *uniqueOrientArrThis = calloc(nIDs * ORIENT_ARRAY_COLS, sizeof(*uniqueOrientArrThis));
        
        if (!uniqueArrThis || !uniqueOrientArrThis) {
            log_error("Failed to allocate memory for unique arrays");
            free(keys);
            free(OMArr);
            free(confIAArr);
            free(tmpArr);
            free(markArr);
            free(uniqueArrThis);
            free(uniqueOrientArrThis);
            close(ib);
            return;
        }
        
        int nUniquesThis = 0;
        
        for (int i = 0; i < nIDs; i++) {
            if (markArr[i]) continue;
            
            /* Copy orientation matrix and convert to quaternion */
            memcpy(OMThis, &OMArr[i * ORIENT_ARRAY_COLS], ORIENT_ARRAY_COLS * sizeof(double));
            OrientMat2Quat(OMThis, Quat1);
            
            double bCon = confIAArr[i * CONF_IA_ARRAY_COLS + 0];
            double bIA = confIAArr[i * CONF_IA_ARRAY_COLS + 1];
            int bRN = i;
            
            /* Compare with other orientations */
            for (int j = i + 1; j < nIDs; j++) {
                if (markArr[j]) continue;
                
                memcpy(OMInside, &OMArr[j * ORIENT_ARRAY_COLS], ORIENT_ARRAY_COLS * sizeof(double));
                OrientMat2Quat(OMInside, Quat2);
                
                double conIn = confIAArr[j * CONF_IA_ARRAY_COLS + 0];
                double iaIn = confIAArr[j * CONF_IA_ARRAY_COLS + 1];
                
                /* Check misorientation angle */
                GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
                
                if (ang < maxAng) {
                    if (bCon < conIn) {
                        bCon = conIn;
                        bIA = iaIn;
                        bRN = j;
                    } else if (bCon == conIn && bIA > iaIn) {
                        bCon = conIn;
                        bIA = iaIn;
                        bRN = j;
                    }
                    markArr[j] = true;
                }
            }
            
            /* Store the best orientation */
            memcpy(&uniqueArrThis[nUniquesThis * KEY_ARRAY_COLS], 
                   &keys[bRN * KEY_ARRAY_COLS], 
                   KEY_ARRAY_COLS * sizeof(size_t));
                   
            memcpy(&uniqueOrientArrThis[nUniquesThis * ORIENT_ARRAY_COLS], 
                   &OMArr[bRN * ORIENT_ARRAY_COLS], 
                   ORIENT_ARRAY_COLS * sizeof(double));
                   
            nUniquesThis++;
        }
        
        /* Save results */
        size_t outarr[5] = {
            voxNr,
            keys[bestRow * KEY_ARRAY_COLS + 0],
            keys[bestRow * KEY_ARRAY_COLS + 1],
            keys[bestRow * KEY_ARRAY_COLS + 2],
            keys[bestRow * KEY_ARRAY_COLS + 3]
        };
        
        pwrite(ib, outarr, 5 * sizeof(size_t), 5 * sizeof(size_t) * voxNr);
        
        /* Save unique orientations to file */
        char outKeyFN[MAX_PATH_LEN];
        sprintf(outKeyFN, "%s/UniqueIndexKeyOrientAll_voxNr_%06d.txt", folderName, voxNr);
        
        FILE *outKeyF = fopen(outKeyFN, "w");
        if (outKeyF) {
            for (int i = 0; i < nUniquesThis; i++) {
                for (int j = 0; j < KEY_ARRAY_COLS; j++) {
                    fprintf(outKeyF, "%zu ", uniqueArrThis[i * KEY_ARRAY_COLS + j]);
                }
                
                for (int j = 0; j < ORIENT_ARRAY_COLS; j++) {
                    fprintf(outKeyF, "%lf ", uniqueOrientArrThis[i * ORIENT_ARRAY_COLS + j]);
                }
                
                fprintf(outKeyF, "\n");
            }
            fclose(outKeyF);
        }
        
        free(uniqueArrThis);
        free(uniqueOrientArrThis);
    }
    
    /* Clean up */
    free(keys);
    free(OMArr);
    free(confIAArr);
    free(tmpArr);
    free(markArr);
    close(ib);
}

UniqueOrientationsResult find_unique_orientations(
    size_t *allKeyArr, double *allOrientationsArr, size_t nScans, int sgNr, double maxAng) {
    
    UniqueOrientationsResult result;
    result.nUniques = 0;
    
    /* Allocate memory for results */
    result.uniqueKeyArr = calloc(nScans * nScans * 5, sizeof(*(result.uniqueKeyArr)));
    result.uniqueOrientArr = calloc(nScans * nScans * ORIENT_ARRAY_COLS, sizeof(*(result.uniqueOrientArr)));
    
    if (!result.uniqueKeyArr || !result.uniqueOrientArr) {
        fatal_error("Failed to allocate memory for unique orientation results");
    }
    
    /* Mark invalid orientations */
    bool *markArr = malloc(nScans * nScans * sizeof(*markArr));
    if (!markArr) {
        fatal_error("Failed to allocate mark array");
    }
    
    for (size_t i = 0; i < nScans * nScans; i++) {
        markArr[i] = (allKeyArr[i * KEY_ARRAY_COLS] == (size_t)-1);
    }
    
    /* Find unique orientations */
    double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Axis[3];
    double ang, fracInside, bestFrac;
    size_t bestOrientationRowNr;
    
    for (size_t i = 0; i < nScans * nScans; i++) {
        if (markArr[i]) continue;
        
        /* Copy orientation matrix and convert to quaternion */
        memcpy(OMThis, &allOrientationsArr[i * 10], 9 * sizeof(double));
        bestFrac = allOrientationsArr[i * 10 + 9];
        OrientMat2Quat(OMThis, Quat1);
        bestOrientationRowNr = i;
        
        /* Compare with other orientations */
        for (size_t j = i + 1; j < nScans * nScans; j++) {
            if (markArr[j]) continue;
            
            fracInside = allOrientationsArr[j * 10 + 9];
            memcpy(OMInside, &allOrientationsArr[j * 10], 9 * sizeof(double));
            OrientMat2Quat(OMInside, Quat2);
            
            /* Check misorientation angle */
            GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
            
            if (ang < maxAng) {
                if (bestFrac < fracInside) {
                    bestFrac = fracInside;
                    bestOrientationRowNr = j;
                }
                markArr[j] = true;
            }
        }
        
        /* Store the best orientation */
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
    
    /* Resize arrays to actual size */
    size_t *resized_keys = realloc(result.uniqueKeyArr, 
                                  result.nUniques * 5 * sizeof(*(result.uniqueKeyArr)));
    if (resized_keys) {
        result.uniqueKeyArr = resized_keys;
    }
    
    double *resized_orients = realloc(result.uniqueOrientArr, 
                                     result.nUniques * ORIENT_ARRAY_COLS * sizeof(*(result.uniqueOrientArr)));
    if (resized_orients) {
        result.uniqueOrientArr = resized_orients;
    }
    
    return result;
}

void save_orientation_results(UniqueOrientationsResult *uniqueResult, const char *outputFolder) {
    char uniqueOrientsFN[MAX_PATH_LEN];
    sprintf(uniqueOrientsFN, "%s/UniqueOrientations.csv", outputFolder);
    
    FILE *uniqueOrientationsF = fopen(uniqueOrientsFN, "w");
    if (!uniqueOrientationsF) {
        log_error("Failed to open %s for writing", uniqueOrientsFN);
        return;
    }
    
    for (size_t i = 0; i < uniqueResult->nUniques; i++) {
        /* Write key values */
        for (int j = 0; j < 5; j++) {
            fprintf(uniqueOrientationsF, "%zu ", uniqueResult->uniqueKeyArr[i * 5 + j]);
        }
        
        /* Write orientation values */
        for (int j = 0; j < ORIENT_ARRAY_COLS; j++) {
            fprintf(uniqueOrientationsF, "%lf ", uniqueResult->uniqueOrientArr[i * ORIENT_ARRAY_COLS + j]);
        }
        
        fprintf(uniqueOrientationsF, "\n");
    }
    
    fclose(uniqueOrientationsF);
}

void generate_sinograms(SpotList *spotList, UniqueOrientationsResult *uniqueResult, 
                       double *allSpots, size_t nSpotsAll, int nScans, double tolOme, 
                       double tolEta, const char *outputFolder, int numProcs) {
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
        
        if (grainNr >= 0 && grainNr < (int)uniqueResult->nUniques) {
            nrHKLsPerGrain[grainNr] = spotNr + 1 > nrHKLsPerGrain[grainNr] ? 
                                      spotNr + 1 : nrHKLsPerGrain[grainNr];
        }
    }
    
    /* Find maximum */
    for (size_t i = 0; i < uniqueResult->nUniques; i++) {
        if (nrHKLsPerGrain[i] > maxNHKLs) {
            maxNHKLs = nrHKLsPerGrain[i];
        }
    }
    
    /* Allocate memory for sinograms */
    size_t szSino = uniqueResult->nUniques * maxNHKLs * nScans;
    double *sinoArr = calloc(szSino, sizeof(*sinoArr));
    double *allOmeArr = calloc(szSino, sizeof(*allOmeArr));
    
    if (!sinoArr || !allOmeArr) {
        free(nrHKLsPerGrain);
        free(sinoArr);
        free(allOmeArr);
        fatal_error("Failed to allocate memory for sinograms (size: %zu bytes)",
                   szSino * sizeof(*sinoArr));
    }
    
    /* Initialize arrays */
    memset(sinoArr, 0, szSino * sizeof(*sinoArr));
    
    double *omeArr = calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*omeArr));
    double *maxIntArr = calloc(uniqueResult->nUniques * maxNHKLs, sizeof(*maxIntArr));
    
    if (!omeArr || !maxIntArr) {
        free(nrHKLsPerGrain);
        free(sinoArr);
        free(allOmeArr);
        free(omeArr);
        free(maxIntArr);
        fatal_error("Failed to allocate memory for omega arrays");
    }
    
    /* Initialize omega array */
    for (size_t i = 0; i < uniqueResult->nUniques * maxNHKLs; i++) {
        omeArr[i] = -10000.0;
    }
    
    /* Process each scan in parallel */
    #pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int scanNr = 0; scanNr < nScans; scanNr++) {
        /* Loop through all spots in the scan */
        for (size_t spotIdx = 0; spotIdx < nSpotsAll; spotIdx++) {
            /* Check if this spot belongs to the current scan */
            if ((int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 9] != scanNr) {
                continue;
            }
            
            /* Compare with all unique spots */
            for (size_t uniqueSpotIdx = 0; uniqueSpotIdx < spotList->nSpots; uniqueSpotIdx++) {
                SpotData *spot = &spotList->spotData[uniqueSpotIdx];
                
                /* Check if ring number matches */
                if ((int)allSpots[SPOTS_ARRAY_COLS * spotIdx + 5] != spot->ringNr) {
                    continue;
                }
                
                /* Check if omega and eta are within tolerance */
                if (fabs(allSpots[SPOTS_ARRAY_COLS * spotIdx + 2] - spot->omega) < tolOme &&
                    fabs(allSpots[SPOTS_ARRAY_COLS * spotIdx + 6] - spot->eta) < tolEta) {
                    
                    /* Calculate array indices */
                    size_t locThis = (size_t)spot->grainNr * maxNHKLs * nScans +
                                     (size_t)spot->spotNr * nScans +
                                     scanNr;
                    
                    /* Store intensity and omega values */
                    sinoArr[locThis] = allSpots[SPOTS_ARRAY_COLS * spotIdx + 3];
                    allOmeArr[locThis] = allSpots[SPOTS_ARRAY_COLS * spotIdx + 2];
                    
                    /* Update maximum intensity */
                    size_t maxIntIdx = (size_t)spot->grainNr * maxNHKLs + (size_t)spot->spotNr;
                    
                    #pragma omp critical
                    {
                        if (maxIntArr[maxIntIdx] < allSpots[SPOTS_ARRAY_COLS * spotIdx + 3]) {
                            maxIntArr[maxIntIdx] = allSpots[SPOTS_ARRAY_COLS * spotIdx + 3];
                        }
                    }
                }
            }
        }
    }
    
    /* Calculate average omega angles and normalize intensities */
    for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
        for (size_t spotIdx = 0; spotIdx < (size_t)maxNHKLs; spotIdx++) {
            double avgOmega = 0.0;
            int nAngles = 0;
            double maxIntensity = maxIntArr[grainIdx * maxNHKLs + spotIdx];
            
            /* Skip spots with no intensity */
            if (maxIntensity <= 0) {
                continue;
            }
            
            /* Process each scan */
            for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
                size_t index = grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx;
                
                /* Normalize intensity if it's positive */
                if (sinoArr[index] > 0) {
                    sinoArr[index] /= maxIntensity;
                    avgOmega += allOmeArr[index];
                    nAngles++;
                }
            }
            
            /* Store average omega if we have measurements */
            if (nAngles > 0) {
                omeArr[grainIdx * maxNHKLs + spotIdx] = avgOmega / nAngles;
            }
        }
    }
    
    /* Sort spots by omega angle for each grain */
    for (size_t grainIdx = 0; grainIdx < uniqueResult->nUniques; grainIdx++) {
        SinoSortData *sortData = malloc(maxNHKLs * sizeof(*sortData));
        if (!sortData) {
            log_error("Failed to allocate memory for sorting data");
            continue;
        }
        
        /* Count valid spots */
        int nValidSpots = 0;
        for (int spotIdx = 0; spotIdx < maxNHKLs; spotIdx++) {
            if (omeArr[grainIdx * maxNHKLs + spotIdx] > -9999.0) {
                sortData[nValidSpots].angle = omeArr[grainIdx * maxNHKLs + spotIdx];
                sortData[nValidSpots].intensities = calloc(nScans, sizeof(double));
                
                if (!sortData[nValidSpots].intensities) {
                    log_error("Failed to allocate memory for intensity data");
                    for (int k = 0; k < nValidSpots; k++) {
                        free(sortData[k].intensities);
                    }
                    free(sortData);
                    continue;
                }
                
                /* Copy intensities */
                for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
                    sortData[nValidSpots].intensities[scanIdx] = 
                        sinoArr[grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx];
                }
                
                nValidSpots++;
            }
        }
        
        /* Sort by omega angle */
        qsort(sortData, nValidSpots, sizeof(SinoSortData), compare_sino_data);
        
        /* Copy sorted data back */
        for (int spotIdx = 0; spotIdx < nValidSpots; spotIdx++) {
            omeArr[grainIdx * maxNHKLs + spotIdx] = sortData[spotIdx].angle;
            
            for (int scanIdx = 0; scanIdx < nScans; scanIdx++) {
                sinoArr[grainIdx * maxNHKLs * nScans + spotIdx * nScans + scanIdx] = 
                    sortData[spotIdx].intensities[scanIdx];
            }
            
            free(sortData[spotIdx].intensities);
        }
        
        free(sortData);
    }
    
    /* Save results to files */
    char sinoFN[MAX_PATH_LEN], omeFN[MAX_PATH_LEN], HKLsFN[MAX_PATH_LEN];
    sprintf(sinoFN, "%s/sinos_%zu_%d_%d.bin", outputFolder, uniqueResult->nUniques, maxNHKLs, nScans);
    sprintf(omeFN, "%s/omegas_%zu_%d.bin", outputFolder, uniqueResult->nUniques, maxNHKLs);
    sprintf(HKLsFN, "%s/nrHKLs_%zu.bin", outputFolder, uniqueResult->nUniques);
    
    FILE *sinoF = fopen(sinoFN, "wb");
    FILE *omeF = fopen(omeFN, "wb");
    FILE *HKLsF = fopen(HKLsFN, "wb");
    
    if (sinoF && omeF && HKLsF) {
        fwrite(sinoArr, uniqueResult->nUniques * maxNHKLs * nScans * sizeof(*sinoArr), 1, sinoF);
        fwrite(omeArr, uniqueResult->nUniques * maxNHKLs * sizeof(*omeArr), 1, omeF);
        fwrite(nrHKLsPerGrain, uniqueResult->nUniques * sizeof(*nrHKLsPerGrain), 1, HKLsF);
    } else {
        log_error("Failed to open output files for writing");
    }
    
    /* Clean up */
    if (sinoF) fclose(sinoF);
    if (omeF) fclose(omeF);
    if (HKLsF) fclose(HKLsF);
    
    free(sinoArr);
    free(omeArr);
    free(allOmeArr);
    free(maxIntArr);
    free(nrHKLsPerGrain);
}

void free_resources(SpotList *spotList, UniqueOrientationsResult *uniqueResult) {
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

SpotList process_spots(UniqueOrientationsResult *uniqueResult, const char *folderName, 
                      double *allSpots, size_t nSpotsAll, double tolOme, double tolEta) {
    
    printf("Processing spots for %zu unique orientations...\n", uniqueResult->nUniques);
    
    SpotList result;
    result.spotData = calloc(MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques, sizeof(*(result.spotData)));
    
    if (!result.spotData) {
        fatal_error("Failed to allocate memory for spot data");
    }
    
    bool *isNotUniqueSpot = calloc(MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques, sizeof(*isNotUniqueSpot));
    if (!isNotUniqueSpot) {
        free(result.spotData);
        fatal_error("Failed to allocate memory for spot uniqueness flags");
    }
    
    /* Initialize with all spots being unique */
    memset(isNotUniqueSpot, 0, MAX_N_SPOTS_PER_GRAIN * uniqueResult->nUniques * sizeof(*isNotUniqueSpot));
    
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
        
        /* Read spot IDs */
        char IDsFNThis[MAX_PATH_LEN];
        sprintf(IDsFNThis, "%s/IndexBest_IDs_voxNr_%06zu.bin", folderName, thisVoxNr);
        
        FILE *IDF = fopen(IDsFNThis, "rb");
        if (!IDF) {
            log_error("Failed to open %s", IDsFNThis);
            continue;
        }
        
        fseek(IDF, startPos, SEEK_SET);
        
        int *IDArrThis = malloc(nSpots * sizeof(*IDArrThis));
        if (!IDArrThis) {
            log_error("Failed to allocate memory for ID array");
            fclose(IDF);
            continue;
        }
        
        fread(IDArrThis, nSpots * sizeof(int), 1, IDF);
        fclose(IDF);
        
        int uniqueSpotCount = 0;
        
        /* Process each spot */
        for (size_t j = 0; j < nSpots; j++) {
            /* Verify data alignment */
            if (allSpots[SPOTS_ARRAY_COLS * (IDArrThis[j] - 1) + 4] != (double)IDArrThis[j]) {
                log_error("Data is not aligned. Please check. ID %d doesn't match expected value", IDArrThis[j]);
                free(IDArrThis);
                continue;
            }
            
            /* Store spot data */
            result.spotData[nAllSpots + j].mergedID = IDArrThis[j];
            result.spotData[nAllSpots + j].omega = allSpots[SPOTS_ARRAY_COLS * (IDArrThis[j] - 1) + 2];
            result.spotData[nAllSpots + j].eta = allSpots[SPOTS_ARRAY_COLS * (IDArrThis[j] - 1) + 6];
            result.spotData[nAllSpots + j].ringNr = (int)allSpots[SPOTS_ARRAY_COLS * (IDArrThis[j] - 1) + 5];
            result.spotData[nAllSpots + j].grainNr = i;
            result.spotData[nAllSpots + j].spotNr = j;
            
            /* Check if this spot is a duplicate of any previous spot */
            bool isDuplicate = false;
            for (size_t k = 0; k < nAllSpots + j; k++) {
                if (result.spotData[k].ringNr == result.spotData[nAllSpots + j].ringNr &&
                    fabs(result.spotData[nAllSpots + j].omega - result.spotData[k].omega) < tolOme &&
                    fabs(result.spotData[nAllSpots + j].eta - result.spotData[k].eta) < tolEta) {
                    
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
        printf("Grain %zu: %d unique spots out of %zu total\n", i, uniqueSpotCount, nSpots);
        
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
    
    size_t nUniqueSpots = 0;
    for (size_t i = 0; i < nAllSpots; i++) {
        if (!isNotUniqueSpot[i]) {
            uniqueSpots[nUniqueSpots] = result.spotData[i];
            uniqueSpots[nUniqueSpots].spotNr = nrHKLsFilled[uniqueSpots[nUniqueSpots].grainNr]++;
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
        fprintf(fUniqueSpots, "ID,GrainNr,SpotNr,RingNr,Omega,Eta\n");
        
        for (size_t i = 0; i < nUniqueSpots; i++) {
            fprintf(fUniqueSpots, "%d,%d,%d,%d,%lf,%lf\n",
                    uniqueSpots[i].mergedID, uniqueSpots[i].grainNr,
                    uniqueSpots[i].spotNr, uniqueSpots[i].ringNr,
                    uniqueSpots[i].omega, uniqueSpots[i].eta);
        }
        
        fclose(fUniqueSpots);
    } else {
        log_error("Failed to open %s for writing", fnUniqueSpots);
    }
    
    free(isNotUniqueSpot);
    free(nrHKLsFilled);
    return result;
}