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

#define MAX_N_SOLUTIONS_PER_VOX 1000000
#define BUFFER_SIZE 2048 // Replace hardcoded buffer size with a macro

// Function prototypes
void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng, int nScans);
void writeSpotsToIndex(const char *folderName, const char *originalFolder, int nScans);

int main(int argc, char *argv[]) {
    double start_time = omp_get_wtime();
    printf("\n\n\t\tFinding Multiple Solutions in PF-HEDM.\n\n");

    if (argc < 6) {
        printf("Supply folder, spaceGroup, maxAng, NumberScans and nCPUs as arguments: ie %s sgNum maxAngle folderName nScans nCPUs (optional)minConf\n\n"
               "The indexing results must be in folderName/Output\n", argv[0]);
        return 1;
    }

    char folderName[BUFFER_SIZE];
    sprintf(folderName, "%s/Output/", argv[1]);
    int sgNr = atoi(argv[2]);
    double maxAng = atof(argv[3]);
    int nScans = atoi(argv[4]);
    int numProcs = atoi(argv[5]);
    double minConf = 0.0;
    if (argc > 6) {
        minConf = atof(argv[6]);
        printf("Minimum confidence: %lf\n", minConf);
    }

    #pragma omp parallel for num_threads(numProcs) schedule(dynamic)
    for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
        processVoxel(voxNr, folderName, sgNr, maxAng, nScans,minConf);
    }

    writeSpotsToIndex(folderName, argv[1], nScans);

    printf("Execution completed in %.2f seconds.\n", omp_get_wtime() - start_time);
    return 0;
}

void processVoxel(int voxNr, const char *folderName, int sgNr, double maxAng, int nScans, double minConf) {
    char valsFN[BUFFER_SIZE], keyFN[BUFFER_SIZE];
    sprintf(valsFN, "%s/IndexBest_voxNr_%0*d.bin", folderName, 6, voxNr);
    sprintf(keyFN, "%s/IndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);

    FILE *valsF = fopen(valsFN, "rb");
    FILE *keyF = fopen(keyFN, "r");
    if (!valsF || !keyF) {
        if (valsF) fclose(valsF);
        if (keyF) fclose(keyF);
        return;
    }

    size_t *keys = calloc(MAX_N_SOLUTIONS_PER_VOX * 4, sizeof(*keys));
    if (!keys) {
        fclose(valsF);
        fclose(keyF);
        return;
    }

    char aline[BUFFER_SIZE];
    int nIDs = 0;
    while (fgets(aline, BUFFER_SIZE, keyF) != NULL) {
        sscanf(aline, "%zu %zu %zu %zu", &keys[nIDs * 4 + 0], &keys[nIDs * 4 + 1], &keys[nIDs * 4 + 2], &keys[nIDs * 4 + 3]);
        nIDs++;
    }
    fclose(keyF);

    if (nIDs == 0) {
        free(keys);
        fclose(valsF);
        return;
    }

    keys = realloc(keys, nIDs * 4 * sizeof(*keys));
    double *OMArr = calloc(nIDs * 9, sizeof(double));
    double *confIAArr = calloc(nIDs * 2, sizeof(double));
    double *tmpArr = calloc(nIDs * 16, sizeof(double));
    if (!OMArr || !confIAArr || !tmpArr) {
        free(keys);
        free(OMArr);
        free(confIAArr);
        free(tmpArr);
        fclose(valsF);
        return;
    }

    fread(tmpArr, nIDs * 16 * sizeof(double), 1, valsF);
    fclose(valsF);

    for (int i = 0; i < nIDs; i++) {
        confIAArr[i * 2 + 0] = tmpArr[i * 16 + 15] / tmpArr[i * 16 + 14];
        confIAArr[i * 2 + 1] = tmpArr[i * 16 + 1];
        for (int j = 0; j < 9; j++) OMArr[i * 9 + j] = tmpArr[i * 16 + 2 + j];
    }
    free(tmpArr);

    bool *markArr = calloc(nIDs, sizeof(*markArr));
    size_t *uniqueArr = calloc(nIDs * 4, sizeof(*uniqueArr));
    if (!markArr || !uniqueArr) {
        free(keys);
        free(OMArr);
        free(confIAArr);
        free(markArr);
        free(uniqueArr);
        return;
    }

    int nUniques = 0;
    double OMThis[9], OMInside[9], Quat1[4], Quat2[4], Angle, Axis[3], ang;
    int bRN;
    double bCon, bIA, conIn, iaIn;
    for (int i = 0; i < nIDs; i++) {
        if (markArr[i] == true) continue;
        if (confIAArr[i * 2 + 0] < minConf) {
            markArr[i] = true;
            continue;
        }
        for (int j = 0; j < 9; j++) OMThis[j] = OMArr[i * 9 + j];
        OrientMat2Quat(OMThis, Quat1);
        bCon = confIAArr[i * 2 + 0];
        bIA = confIAArr[i * 2 + 1];
        bRN = i;
        for (int j = i + 1; j < nIDs; j++) {
            if (markArr[j] == true) continue;
            for (int k = 0; k < 9; k++) OMInside[k] = OMArr[j * 9 + k];
            OrientMat2Quat(OMInside, Quat2);
            Angle = GetMisOrientation(Quat1, Quat2, Axis, &ang, sgNr);
            conIn = confIAArr[j * 2 + 0];
            if (conIn < minConf) {
                markArr[j] = true;
                continue;
            }
            iaIn = confIAArr[j * 2 + 1];
            if (ang < maxAng) {
                if (bCon < conIn) {
                    bCon = conIn;
                    bIA = iaIn;
                    bRN = j;
                } else if (bCon == conIn) {
                    if (bIA > iaIn) {
                        bCon = conIn;
                        bIA = iaIn;
                        bRN = j;
                    }
                }
                markArr[j] = true;
            }
        }
        for (int j = 0; j < 4; j++) uniqueArr[nUniques * 4 + j] = keys[bRN * 4 + j];
        nUniques++;
    }

    char outKeyFN[BUFFER_SIZE];
    sprintf(outKeyFN, "%s/UniqueIndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);
    FILE *outKeyF = fopen(outKeyFN, "w");
    if (outKeyF) {
        for (int i = 0; i < nUniques; i++) {
            fprintf(outKeyF, "%zu %zu %zu %zu\n", uniqueArr[i * 4 + 0], uniqueArr[i * 4 + 1], uniqueArr[i * 4 + 2], uniqueArr[i * 4 + 3]);
        }
        fclose(outKeyF);
    }

    free(markArr);
    free(keys);
    free(OMArr);
    free(uniqueArr);
}

void writeSpotsToIndex(const char *folderName, const char *originalFolder, int nScans) {
    char outFN[BUFFER_SIZE];
    sprintf(outFN, "%s/SpotsToIndex.csv", originalFolder);
    FILE *outF = fopen(outFN, "w");
    if (!outF) return;

    for (int voxNr = 0; voxNr < nScans * nScans; voxNr++) {
        char outKeyFN[BUFFER_SIZE], aline[BUFFER_SIZE];
        sprintf(outKeyFN, "%s/UniqueIndexKey_voxNr_%0*d.txt", folderName, 6, voxNr);
        FILE *outKeyF = fopen(outKeyFN, "r");
        if (!outKeyF) continue;

        while (fgets(aline, BUFFER_SIZE, outKeyF) != NULL) {
            fprintf(outF, "%d %s", voxNr, aline);
        }
        fclose(outKeyF);
    }
    fclose(outF);
}