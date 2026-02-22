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

#endif /* MIDAS_LIMITS_H */
