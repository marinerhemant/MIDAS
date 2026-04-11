/*
 * GenerateSeedLookupTables.c
 *
 * Reads the master quaternion CSV (triclinicSeed.txt, covering all of SO(3)/Z2)
 * and generates:
 *   1. orientations_master.bin  — binary double[N][4] (w,x,y,z)
 *   2. lookup_<type>.bin        — binary int32 index arrays for each of 12
 *                                  MIDAS symmetry types
 *
 * Usage: GenerateSeedLookupTables <master_csv> <output_dir> [nCPUs]
 *
 * Links against: libmidas_orientation (for MakeSymmetries, BringDownToFundamentalRegionSym)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "GetMisorientation.h"

#define MAX_ORIENTATIONS 6000000
#define FZ_TOL 1e-6

/* ── Symmetry type table ─────────────────────────────────────────── */

typedef struct {
    const char *label;
    int representative_sg;
    const char *filename;
} SymType;

static const SymType SYM_TYPES[] = {
    { "triclinic",       1,   "lookup_triclinic.bin"       },
    { "monoclinic",      15,  "lookup_monoclinic.bin"      },
    { "orthorhombic",    74,  "lookup_orthorhombic.bin"    },
    { "tetragonal_low",  88,  "lookup_tetragonal_low.bin"  },
    { "tetragonal_high", 142, "lookup_tetragonal_high.bin" },
    { "trigonal_low",    148, "lookup_trigonal_low.bin"     },
    { "trigonal_type1",  155, "lookup_trigonal_type1.bin"   },
    { "trigonal_type2",  149, "lookup_trigonal_type2.bin"   },
    { "hexagonal_low",   176, "lookup_hexagonal_low.bin"   },
    { "hexagonal_high",  194, "lookup_hexagonal_high.bin"  },
    { "cubic_low",       206, "lookup_cubic_low.bin"       },
    { "cubic_high",      225, "lookup_cubic_high.bin"      },
};

#define N_SYM_TYPES 12

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <master_csv> <output_dir> [nCPUs]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  master_csv  : Path to triclinicSeed.txt (CSV w,x,y,z quaternions)\n");
    fprintf(stderr, "  output_dir  : Directory for output binary files\n");
    fprintf(stderr, "  nCPUs       : Number of OpenMP threads (default: all available)\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const char *master_csv = argv[1];
    const char *output_dir = argv[2];
    int nCPUs = 0;

    if (argc >= 4) {
        nCPUs = atoi(argv[3]);
    }

#ifdef _OPENMP
    if (nCPUs > 0) {
        omp_set_num_threads(nCPUs);
    }
    printf("Using %d OpenMP threads.\n", omp_get_max_threads());
#endif

    /* ── Step 1: Read master quaternion CSV ─────────────────────── */

    printf("Reading master quaternions from %s ...\n", master_csv);

    double *quats = (double *)malloc(MAX_ORIENTATIONS * 4 * sizeof(double));
    if (quats == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for quaternions.\n");
        return 1;
    }

    FILE *fp = fopen(master_csv, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open %s\n", master_csv);
        free(quats);
        return 1;
    }

    int N = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (N >= MAX_ORIENTATIONS) {
            fprintf(stderr, "Error: Exceeded MAX_ORIENTATIONS (%d)\n", MAX_ORIENTATIONS);
            fclose(fp);
            free(quats);
            return 1;
        }
        double w, x, y, z;
        if (sscanf(line, "%lf,%lf,%lf,%lf", &w, &x, &y, &z) == 4) {
            quats[N * 4 + 0] = w;
            quats[N * 4 + 1] = x;
            quats[N * 4 + 2] = y;
            quats[N * 4 + 3] = z;
            N++;
        }
    }
    fclose(fp);
    printf("  Read %d quaternions.\n", N);

    /* Ensure w >= 0 for all quaternions */
    for (int i = 0; i < N; i++) {
        if (quats[i * 4] < 0) {
            quats[i * 4 + 0] = -quats[i * 4 + 0];
            quats[i * 4 + 1] = -quats[i * 4 + 1];
            quats[i * 4 + 2] = -quats[i * 4 + 2];
            quats[i * 4 + 3] = -quats[i * 4 + 3];
        }
    }

    /* ── Step 2: Write binary master file ──────────────────────── */

    char path[2048];
    snprintf(path, sizeof(path), "%s/orientations_master.bin", output_dir);
    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not create %s\n", path);
        free(quats);
        return 1;
    }
    fwrite(quats, sizeof(double), (size_t)N * 4, fp);
    fclose(fp);
    printf("  Wrote %s (%d x 4 doubles, %.1f MB)\n",
           path, N, (double)N * 4 * sizeof(double) / (1024.0 * 1024.0));

    /* ── Step 3: Generate lookup tables ────────────────────────── */

    /* Allocate flag array (reused for each type) */
    char *in_fz = (char *)calloc(N, sizeof(char));
    if (in_fz == NULL) {
        fprintf(stderr, "Error: Could not allocate flag array.\n");
        free(quats);
        return 1;
    }

    printf("\nGenerating lookup tables for %d symmetry types:\n", N_SYM_TYPES);
    printf("%-20s %6s %10s %10s\n", "Type", "NrSym", "N_in_FZ", "Expected");
    printf("%-20s %6s %10s %10s\n", "----", "-----", "-------", "--------");

    for (int t = 0; t < N_SYM_TYPES; t++) {
        const SymType *st = &SYM_TYPES[t];

        /* Get symmetry operators */
        double Sym[24][4];
        int NrSym = MakeSymmetries(st->representative_sg, Sym);

        /* Clear flags */
        memset(in_fz, 0, N * sizeof(char));

        /* Parallel FZ membership test */
        #pragma omp parallel for schedule(dynamic, 1000)
        for (int i = 0; i < N; i++) {
            double qin[4], qout[4];
            qin[0] = quats[i * 4 + 0];
            qin[1] = quats[i * 4 + 1];
            qin[2] = quats[i * 4 + 2];
            qin[3] = quats[i * 4 + 3];

            BringDownToFundamentalRegionSym(qin, qout, NrSym, Sym);

            /* Check if output matches input (within tolerance) */
            double dw = fabs(qin[0] - qout[0]);
            double dx = fabs(qin[1] - qout[1]);
            double dy = fabs(qin[2] - qout[2]);
            double dz = fabs(qin[3] - qout[3]);

            if (dw < FZ_TOL && dx < FZ_TOL && dy < FZ_TOL && dz < FZ_TOL) {
                in_fz[i] = 1;
            }
        }

        /* Count and collect indices */
        int n_in_fz = 0;
        for (int i = 0; i < N; i++) {
            if (in_fz[i]) n_in_fz++;
        }

        int32_t *indices = (int32_t *)malloc(n_in_fz * sizeof(int32_t));
        if (indices == NULL) {
            fprintf(stderr, "Error: Could not allocate index array for %s.\n",
                    st->label);
            free(in_fz);
            free(quats);
            return 1;
        }

        int idx = 0;
        for (int i = 0; i < N; i++) {
            if (in_fz[i]) {
                indices[idx++] = (int32_t)i;
            }
        }

        /* Write binary lookup file */
        snprintf(path, sizeof(path), "%s/%s", output_dir, st->filename);
        fp = fopen(path, "wb");
        if (fp == NULL) {
            fprintf(stderr, "Error: Could not create %s\n", path);
            free(indices);
            free(in_fz);
            free(quats);
            return 1;
        }
        fwrite(indices, sizeof(int32_t), n_in_fz, fp);
        fclose(fp);

        printf("%-20s %6d %10d %10d\n",
               st->label, NrSym, n_in_fz, N / NrSym);

        free(indices);
    }

    printf("\nDone. All lookup tables written to %s\n", output_dir);

    free(in_fz);
    free(quats);
    return 0;
}
