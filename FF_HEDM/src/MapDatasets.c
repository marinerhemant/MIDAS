// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
//
// Creator: Hemant Sharma   
//
// MapDatasets.c (v1.1 - Hardened)
//
// OpenMP-parallelized version with a command-line argument to set the core count.
// Maps diffraction spots by converting them to g-vectors and finding the
// smallest angle between them.
//
// Usage: ./MapDatasets <source_folder> <target_folder> <num_cores>
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>   // For DBL_MAX
#include <limits.h>  // For PATH_MAX
#include <omp.h>     // OpenMP header

#define MAX_LINE_LENGTH 4096
#define MAX_N_RINGS 500
#define N_COL_OBSSPOTS 9
#define N_COL_EXTRAINFO 14
#define DEG_TO_RAD 0.0174532925199433

// --- Data Structures ---
typedef struct {
    double eta_bin_size, ome_bin_size, distance;
    int ring_numbers[MAX_N_RINGS], num_ring_numbers;
} AppParams;

typedef struct {
    double y, z, omega;
    int ring;
    double eta;
} Point3D;

// --- Function Prototypes ---
int read_parameters(const char* filepath, AppParams* params);
long read_binary_file(const char* filepath, void** buffer, size_t element_size);
void convert_to_g_vector(double y, double z, double omega, double distance, double* gx, double* gy, double* gz);

// --- Main Program ---
int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <source_folder> <target_folder> <num_cores>\n", argv[0]);
        fprintf(stderr, "  - source_folder: Contains ExtraInfo.bin.\n");
        fprintf(stderr, "  - target_folder: Contains the reference dataset and its hash table.\n");
        fprintf(stderr, "  - num_cores: The number of CPU cores to use for processing.\n");
        return 1;
    }
    char* source_folder = argv[1];
    char* target_folder = argv[2];
    
    int num_cores = atoi(argv[3]);
    if (num_cores <= 0) {
        fprintf(stderr, "Warning: Invalid number of cores '%s'. Using all available cores.\n", argv[3]);
    } else {
        omp_set_num_threads(num_cores);
    }

    char param_path[PATH_MAX], spots_path[PATH_MAX], ndata_path[PATH_MAX];
    char data_path[PATH_MAX], extra_info_path[PATH_MAX], output_path[PATH_MAX];

    snprintf(param_path, sizeof(param_path), "%s/paramstest.txt", target_folder);
    snprintf(spots_path, sizeof(spots_path), "%s/Spots.bin", target_folder);
    snprintf(ndata_path, sizeof(ndata_path), "%s/nData.bin", target_folder);
    snprintf(data_path, sizeof(data_path), "%s/Data.bin", target_folder);
    snprintf(extra_info_path, sizeof(extra_info_path), "%s/ExtraInfo.bin", source_folder);
    snprintf(output_path, sizeof(output_path), "%s/mapDatasets_indexed.txt", source_folder);

    AppParams params = {0};
    if (read_parameters(param_path, &params) != 0) return 1;
    if (params.distance == 0.0) {
        fprintf(stderr, "ERROR: 'Distance' or 'Lsd' parameter not found in %s.\n", param_path);
        return 1;
    }

    int highest_ring_no = 0;
    for (int i = 0; i < params.num_ring_numbers; ++i) {
        if (params.ring_numbers[i] > highest_ring_no) highest_ring_no = params.ring_numbers[i];
    }
    int n_ring_bins = highest_ring_no + 1;
    int n_eta_bins = (int)ceil(360.0 / params.eta_bin_size);
    int n_ome_bins = (int)ceil(360.0 / params.ome_bin_size);

    double *spots_mat = NULL, *extra_mat = NULL;
    int *ndata_store = NULL, *data_store = NULL;
    long n_spots = read_binary_file(spots_path, (void**)&spots_mat, sizeof(double)) / N_COL_OBSSPOTS;
    long n_source_points = read_binary_file(extra_info_path, (void**)&extra_mat, sizeof(double)) / N_COL_EXTRAINFO;
    long ndata_len = read_binary_file(ndata_path, (void**)&ndata_store, sizeof(int));
    long data_len = read_binary_file(data_path, (void**)&data_store, sizeof(int));
    
    // **FIX #1:** Added ndata_len and data_len to the check
    if (n_spots <= 0 || n_source_points <= 0 || ndata_len <= 0 || data_len <= 0) {
        fprintf(stderr, "ERROR: Failed to load one or more essential .bin data files. Check paths and file contents.\n");
        free(spots_mat); free(extra_mat); free(ndata_store); free(data_store);
        return 1;
    }

    printf("Successfully loaded:\n- %ld source spots\n- %ld target spots\n- Using Distance = %.2f\n\n", 
           n_source_points, n_spots, params.distance);
    printf("Starting parallel processing using up to %d threads...\n", num_cores > 0 ? num_cores : omp_get_max_threads());

    int* results = malloc(n_source_points * sizeof(int));
    if (!results) { fprintf(stderr, "ERROR: Failed to allocate memory for results array.\n"); return 1; }

    long progress_counter = 0;
    
    #pragma omp parallel for schedule(dynamic, 100)
    for (long i = 0; i < n_source_points; ++i) {
        if (omp_get_thread_num() == 0) {
            long current_progress = 0;
            #pragma omp atomic read
            current_progress = progress_counter;
            if ((long)(current_progress / 10000) != (long)((current_progress - 1) / 10000)) {
                printf("Processing... %ld / %ld (%.1f%%)\r", current_progress, n_source_points, 100.0 * current_progress / n_source_points);
                fflush(stdout);
            }
        }

        results[i] = -1;
        Point3D query_point = {
            .y = extra_mat[i * N_COL_EXTRAINFO + 0], .z = extra_mat[i * N_COL_EXTRAINFO + 1],
            .omega = extra_mat[i * N_COL_EXTRAINFO + 2], .ring = (int)extra_mat[i * N_COL_EXTRAINFO + 5],
            .eta = extra_mat[i * N_COL_EXTRAINFO + 6]
        };
        
        if (query_point.ring == 0) {
            #pragma omp atomic update
            progress_counter++;
            continue;
        }

        int best_match_idx = -1;
        double max_cos_angle = -2.0;
        double g_source_x, g_source_y, g_source_z;
        convert_to_g_vector(query_point.y, query_point.z, query_point.omega, params.distance, &g_source_x, &g_source_y, &g_source_z);
        double len_sq_source = g_source_x * g_source_x + g_source_y * g_source_y + g_source_z * g_source_z;

        if (len_sq_source == 0) {
            #pragma omp atomic update
            progress_counter++;
            continue;
        }

        int iRing = query_point.ring - 1;
        if (iRing < 0 || iRing >= n_ring_bins) {
            #pragma omp atomic update
            progress_counter++;
            continue;
        }

        int iEta_center = floor((query_point.eta + 180.0) / params.eta_bin_size);
        int iOme_center = floor((query_point.omega + 180.0) / params.ome_bin_size);

        for (int dr = -1; dr <= 1; ++dr) {
            for (int de = -1; de <= 1; ++de) {
                for (int dw = -1; dw <= 1; ++dw) {
                    int current_ring = iRing + dr;
                    if (current_ring < 0 || current_ring >= n_ring_bins) continue;
                    int current_eta = (iEta_center + de) % n_eta_bins;
                    if (current_eta < 0) current_eta += n_eta_bins;
                    int current_ome = (iOme_center + dw) % n_ome_bins;
                    if (current_ome < 0) current_ome += n_ome_bins;
                    
                    long long pos = (long long)current_ring * n_eta_bins * n_ome_bins + (long long)current_eta * n_ome_bins + current_ome;
                    int num_points_in_bin = ndata_store[pos * 2];
                    int start_offset = ndata_store[pos * 2 + 1];

                    for (int k = 0; k < num_points_in_bin; ++k) {
                        printf("Checking candidate %d in bin (query_point.ring=%d, current_ring=%d, eta=%d, ome=%d, offset=%d, len=%d, pos=%lld, num_points=%d, ndata_len=%d, start_offset_loc=%d)\n", k, query_point.ring, current_ring, current_eta, current_ome, start_offset, data_len, pos, num_points_in_bin, ndata_len, pos * 2 + 1);
                        int candidate_idx = data_store[start_offset + k];
                        double y_cand = spots_mat[candidate_idx * N_COL_OBSSPOTS + 0];
                        double z_cand = spots_mat[candidate_idx * N_COL_OBSSPOTS + 1];
                        double omega_cand = spots_mat[candidate_idx * N_COL_OBSSPOTS + 2];
                        double g_cand_x, g_cand_y, g_cand_z;
                        // print y, z, omega, distance
                        printf("%lf %lf %lf %lf\n", y_cand, z_cand, omega_cand, params.distance);
                        convert_to_g_vector(y_cand, z_cand, omega_cand, params.distance, &g_cand_x, &g_cand_y, &g_cand_z);
                        double dot_product = g_source_x * g_cand_x + g_source_y * g_cand_y + g_source_z * g_cand_z;
                        double len_sq_cand = g_cand_x * g_cand_x + g_cand_y * g_cand_y + g_cand_z * g_cand_z;
                        if (len_sq_cand > 0) {
                            double cos_angle = dot_product / sqrt(len_sq_source * len_sq_cand);
                            if (cos_angle > max_cos_angle) {
                                max_cos_angle = cos_angle;
                                best_match_idx = candidate_idx;
                            }
                        }
                    }
                }
            }
        }
        results[i] = best_match_idx;
        #pragma omp atomic update
        progress_counter++;
    }
    printf("Processing... %ld / %ld (100.0%%)\n", n_source_points, n_source_points);

    FILE* out_file = fopen(output_path, "w");
    if (!out_file) {
        fprintf(stderr, "ERROR: Could not open output file '%s' for writing.\n", output_path);
        // **FIX #2:** Free all buffers on this error path
        free(results);
        free(spots_mat);
        free(extra_mat);
        free(ndata_store);
        free(data_store);
        return 1;
    }
    for (long i = 0; i < n_source_points; ++i) {
        fprintf(out_file, "%d\n", results[i]);
    }
    printf("Processing complete. Indexed mapping saved to '%s'\n", output_path);

    fclose(out_file); free(results); free(spots_mat); free(extra_mat); free(ndata_store); free(data_store);
    return 0;
}


// --- Function Implementations (No changes) ---
void convert_to_g_vector(double y, double z, double omega, double distance, double* gx, double* gy, double* gz) {
    double xi = distance, yi = y, zi = z;
    double len = sqrt(xi*xi + yi*yi + zi*zi);
    if (len == 0) { 
        *gx = 0; 
        *gy = 0; 
        *gz = 0; 
        return; 
    }
    double xn = xi / len, yn = yi / len, zn = zi / len;
    double qr_x = xn - 1.0, qr_y = yn;
    double omega_rad = -omega * DEG_TO_RAD;
    // printf("Omega (deg): %lf, Omega (rad): %lf\n", omega, omega_rad);
    double cos_ome = cos(omega_rad), sin_ome = sin(omega_rad);
    *gx = qr_x * cos_ome - qr_y * sin_ome;
    *gy = qr_x * sin_ome + qr_y * cos_ome;
    *gz = zn;
}

int read_parameters(const char* filepath, AppParams* params) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open parameter file: %s\n", filepath);
        return 1;
    }
    char line[MAX_LINE_LENGTH], key[128], dummy[128];
    params->num_ring_numbers = 0; params->distance = 0.0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%s", key) != 1) continue;
        if (strcmp(key, "EtaBinSize") == 0) sscanf(line, "%s %lf", dummy, &params->eta_bin_size);
        else if (strcmp(key, "OmeBinSize") == 0) sscanf(line, "%s %lf", dummy, &params->ome_bin_size);
        else if (strcmp(key, "Distance") == 0 || strcmp(key, "Lsd") == 0) sscanf(line, "%s %lf", dummy, &params->distance);
        else if (strcmp(key, "RingNumbers") == 0) {
            if (params->num_ring_numbers < MAX_N_RINGS) {
                sscanf(line, "%s %d", dummy, &params->ring_numbers[params->num_ring_numbers++]);
            }
        }
    }
    fclose(fp);
    return 0;
}

long read_binary_file(const char* filepath, void** buffer, size_t element_size) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not open binary file %s\n", filepath);
        return -1;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (file_size <= 0) {
        fclose(file); *buffer = NULL; return 0;
    }
    *buffer = malloc(file_size);
    if (!*buffer) {
        fprintf(stderr, "ERROR: Memory allocation failed for reading %s\n", filepath);
        fclose(file); return -1;
    }
    fread(*buffer, 1, file_size, file);
    fclose(file);
    return file_size / element_size;
}
