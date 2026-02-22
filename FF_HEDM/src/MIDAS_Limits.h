//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef MIDAS_LIMITS_H
#define MIDAS_LIMITS_H

// --- Global Physics & Grid Bounds ---
// Unifying varying declarations into comprehensive maximums
#define MAX_N_HKLS 5000
#define MAX_N_RINGS 500
#define MAX_N_OMEGA_RANGES 2000

// --- Voxel / Search Limits ---
#define MAX_N_SOLUTIONS_PER_VOX 1000000
#define MAX_POINTS_GRID_GOOD 300000
#define MAX_N_OVERLAPS 355000
#define MAX_N_SPOTS 6000000
#define MAX_N_IDS 6000000
#define MAX_ID_IA_MAT 5000000

// --- File I/O & Buffers ---
#define MAX_FILENAME_LENGTH 2048
#define MAX_BUFFER_SIZE 4096
#define MAX_LINE_LENGTH 4096

// --- Networking & Queues (GPU Streams) ---
#define MAX_CONNECTIONS 10
#define MAX_QUEUE_SIZE 100
#define MAX_TRANSFORM_OPS 10

// --- Centralized error-exit helper ---
// Previously copy-pasted into ~15 source files; now defined once here.
#ifndef MIDAS_CHECK_DEFINED
#define MIDAS_CHECK_DEFINED
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
static inline void check(int test, const char *message, ...) {
  if (test) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
  }
}
#endif /* MIDAS_CHECK_DEFINED */

#endif /* MIDAS_LIMITS_H */
