// =========================================================================
// IntegratorFitPeaksGPUStream.cu (CSR Input Version)
//
// Copyright (c) 2014, UChicago Argonne, LLC; 2024 modifications.
// See LICENSE file (if applicable).
//
// Purpose: Integrates 2D detector data streamed over a socket, performs
//          image transformations, optionally fits peaks to the resulting
//          1D lineout, and saves results. Uses GPU acceleration and CSR map.
//
// Features:
//  - Socket data receiving (multi-threaded)
//  - Thread-safe queue for buffering frames
//  - CSR map file reading (MapCSR.*)
//  - GPU pipeline: H->D Copy -> Transform -> Cast -> Dark Subtract
//  - GPU kernel for CSR-based area calculation and integration
//  - Pinned host memory for H<->D transfers and CSR reading
//  - NLopt peak fitting with global background parameter
//  - Signal handling for graceful shutdown
//  - CUDA Event API for accurate GPU timing
//  - Optional saving of results (including single large 2D output file)
//
// Example compile command (adjust paths and architecture flags):
/*
~/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStreamCSR.cu -o bin/IntegratorFitPeaksGPUStreamCSR \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_90,code=sm_90 \
  -Xcompiler -g \
  -I/path/to/midas/includes \
  -L/path/to/midas/libs \
  -O3 -lnlopt -lz -ldl -lm -lpthread
*/
// =========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>   // For mmap (used by original ReadBins, kept for potential future use)
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/types.h>
#include <libgen.h>     // For basename (optional)
#include <assert.h>
#include <signal.h>     // For signal handling
#include <sys/time.h>   // For gettimeofday
#include <cuda.h>
#include <cuda_runtime.h>
// #include <blosc2.h>     // Include if blosc compression is used
#include <nlopt.h>      // For non-linear optimization

// --- Constants ---
#define SERVER_IP "127.0.0.1"
#define PORT 60439               // Port for receiving image data
#define MAX_CONNECTIONS 10      // Max simultaneous client connections
#define MAX_QUEUE_SIZE 100      // Max image frames buffered before processing
#define HEADER_SIZE sizeof(uint16_t) // Size of frame number header
#define BYTES_PER_PIXEL 8       // Size of each pixel (int64_t)
#define MAX_FILENAME_LENGTH 1024
#define THREADS_PER_BLOCK_TRANSFORM 512 // CUDA block size for transform/processing
#define THREADS_PER_BLOCK_INTEGRATE 512 // CUDA block size for integration
#define THREADS_PER_BLOCK_PROFILE 512  // CUDA block size for 1D profile reduction
#define MAX_TRANSFORM_OPS 10    // Max number of sequential transforms allowed
#define MAX_PEAK_LOCATIONS 100  // Max peaks specifiable in param file
#define AREA_THRESHOLD 1e-9     // Minimum accumulated area considered valid in profiling/normalization

// Global variables (initialized in main)
size_t CHUNK_SIZE;
size_t TOTAL_MSG_SIZE;
// --- CSR globals ---
int csr_num_rows = 0;         // Number of integration bins (R*Eta)
int csr_num_cols = 0;         // Number of detector pixels (Y*Z)
long long csr_num_nonzeros = 0; // Total non-zero elements in map
double    *h_csr_values = NULL; // Pinned host memory
int       *h_csr_col_indices = NULL; // Pinned host memory
int       *h_csr_row_ptr = NULL; // Pinned host memory
double    *d_csr_values = NULL; // Device memory
int       *d_csr_col_indices = NULL; // Device memory
int       *d_csr_row_ptr = NULL; // Device memory
// --- End CSR globals ---
volatile sig_atomic_t keep_running = 1; // Flag for graceful shutdown

// --- Data Structures ---
typedef struct {
    uint16_t dataset_num;
    int64_t *data; // Still receives int64_t raw data (pinned host mem)
    size_t size;   // Number of pixels in the chunk (Y*Z)
} DataChunk;

typedef struct {
    DataChunk chunks[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} ProcessQueue;

// --- Global Variables ---
ProcessQueue process_queue;

// --- CUDA Error Handling ---
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
#define gpuWarnchk(ans) { gpuAssert((ans), __FILE__, __LINE__, false); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abortflag) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abortflag) {
            exit(code);
        }
    }
}

// --- General Error Handling ---
static void check (int test, const char * message, ...) {
    if (test) {
        va_list args;
        va_start (args, message);
        fprintf(stderr, "Fatal Error: ");
        vfprintf(stderr, message, args);
        va_end (args);
        fprintf(stderr, " (errno: %s)\n", test > 0 ? strerror(errno) : "N/A");
        exit(EXIT_FAILURE);
    }
}

// --- Signal Handler ---
void sigint_handler(int signum) {
    if (keep_running) {
        printf("\nCaught signal %d, requesting shutdown...\n", signum);
        keep_running = 0;
    }
}

// --- Timing Function ---
static inline double get_wall_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// --- Queue Functions ---
void queue_init(ProcessQueue *queue) {
    queue->front = 0;
    queue->rear = -1;
    queue->count = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
}

int queue_push(ProcessQueue *queue, uint16_t dataset_num, int64_t *data, size_t num_values) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->count >= MAX_QUEUE_SIZE && keep_running) {
        printf("Queue full, waiting...\n");
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        pthread_cond_timedwait(&queue->not_full, &queue->mutex, &ts);
    }
    if (!keep_running) {
        pthread_mutex_unlock(&queue->mutex);
        return -1; // Indicate shutdown request
    }
    queue->rear = (queue->rear + 1) % MAX_QUEUE_SIZE;
    queue->chunks[queue->rear].dataset_num = dataset_num;
    queue->chunks[queue->rear].data = data;
    queue->chunks[queue->rear].size = num_values;
    queue->count++;
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    return 0; // Success
}

int queue_pop(ProcessQueue *queue, DataChunk *chunk) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->count <= 0 && keep_running) {
        // Wait for data or shutdown signal
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1; // Wait for 1 second max
        pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &ts);
    }
    if (!keep_running && queue->count <= 0) {
        // Shutdown requested while queue is empty
        pthread_mutex_unlock(&queue->mutex);
        return -1; // Indicate shutdown
    }
    // Check again in case timedwait woke up but count is still 0
    if (queue->count <= 0) {
        pthread_mutex_unlock(&queue->mutex);
        // This case might happen if timedwait expires without signal
        // Let the main loop check keep_running again
        return -2; // Indicate empty queue without shutdown
    }

    *chunk = queue->chunks[queue->front];
    queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
    queue->count--;
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    return 0; // Success
}

void queue_destroy(ProcessQueue *queue) {
     pthread_mutex_destroy(&queue->mutex);
     pthread_cond_destroy(&queue->not_empty);
     pthread_cond_destroy(&queue->not_full);
     printf("Cleaning up remaining %d queue entries...\n", queue->count);
     while (queue->count > 0) {
        DataChunk chunk;
        // Directly access without lock, as other threads should be stopped
        chunk.data = queue->chunks[queue->front].data;
        queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
        queue->count--;
        if (chunk.data) {
            // Use cudaFreeHost for pinned memory allocated in handle_client
            gpuWarnchk(cudaFreeHost(chunk.data));
        }
     }
     printf("Queue cleanup complete.\n");
}


// --- Socket Handling ---
void* handle_client(void *arg) {
    int client_socket = *((int*)arg);
    free(arg); // Free the malloc'd socket descriptor container
    uint8_t *buffer = NULL;
    int64_t *data = NULL; // Pinned host memory for image data
    size_t num_pixels = 0;
    int bytes_read = 0;
    int total_bytes_read = 0;

    buffer = (uint8_t*)malloc(TOTAL_MSG_SIZE);
    check(buffer == NULL, "handle_client: Failed alloc recv buffer");
    num_pixels = CHUNK_SIZE / BYTES_PER_PIXEL;

    printf("Client handler started for socket %d.\n", client_socket);

    while (keep_running) {
        total_bytes_read = 0;
        while (total_bytes_read < TOTAL_MSG_SIZE && keep_running) {
            bytes_read = recv(client_socket, buffer + total_bytes_read, TOTAL_MSG_SIZE - total_bytes_read, 0);
            if (bytes_read <= 0) {
                 goto connection_closed; // Exit inner loop, then outer loop check handles keep_running
            }
            total_bytes_read += bytes_read;
        }

        if (!keep_running) {
             break; // Exit outer loop if shutdown requested during recv
        }

        // Check if we received the full message after the inner loop
        if (total_bytes_read != TOTAL_MSG_SIZE) {
             fprintf(stderr, "Warn: Incomplete message received on socket %d (%d/%zu bytes). Discarding.\n",
                     client_socket, total_bytes_read, TOTAL_MSG_SIZE);
             continue; // Try receiving the next message
        }

        uint16_t dataset_num;
        memcpy(&dataset_num, buffer, HEADER_SIZE);

        // Allocate pinned memory for the frame data
        data = NULL; // Reset pointer before allocation
        gpuWarnchk(cudaMallocHost((void**)&data, num_pixels * sizeof(int64_t)));
        if (!data) {
            perror("handle_client: Pinned memory alloc failed");
            break; // Exit handler thread on critical error
        }

        memcpy(data, buffer + HEADER_SIZE, CHUNK_SIZE);

        if (queue_push(&process_queue, dataset_num, data, num_pixels) < 0) {
            printf("handle_client: queue_push failed, likely shutdown. Discarding frame %d.\n", dataset_num);
            gpuWarnchk(cudaFreeHost(data)); // Free the allocated buffer since queue didn't take it
            goto connection_closed; // Assume shutdown if queue push fails
        }
        // data pointer is now owned by the queue, do not free here
        data = NULL;
    }

connection_closed:
    if (bytes_read == 0 && keep_running) {
        printf("Client disconnected (socket %d)\n", client_socket);
    } else if (bytes_read < 0) {
        // Avoid printing error if shutdown was requested (errno might be EINTR)
        if (keep_running) {
            perror("Receive error");
        }
    }

    if (buffer) {
        free(buffer);
    }
    // If loop exited while data was allocated but not pushed, free it
    if (data) {
        gpuWarnchk(cudaFreeHost(data));
    }
    close(client_socket);
    printf("Client handler finished (socket %d).\n", client_socket);
    return NULL;
}

void* accept_connections(void *server_fd_ptr) {
    int server_fd = *((int*)server_fd_ptr);
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    pthread_t thread_id;
    int create_rc;

    printf("Accept thread started, listening for connections.\n");

    while (keep_running) {
        int *client_socket_ptr = (int*) malloc(sizeof(int));
        check(client_socket_ptr == NULL, "accept_connections: Failed alloc client socket ptr");

        *client_socket_ptr = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);

        if (!keep_running) {
             if (*client_socket_ptr >= 0) {
                 close(*client_socket_ptr); // Close accepted socket if shutting down
             }
             free(client_socket_ptr);
             break; // Exit accept loop
        }

        if (*client_socket_ptr < 0) {
            if (errno == EINTR) {
                // Interrupted by signal (likely SIGINT/SIGTERM), loop again to check keep_running
                free(client_socket_ptr);
                continue;
            }
            perror("Accept failed");
            free(client_socket_ptr);
            sleep(1); // Avoid busy-waiting on persistent errors
            continue;
        }

        printf("Connection accepted from %s:%d (socket %d)\n",
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), *client_socket_ptr);

        create_rc = pthread_create(&thread_id, NULL, handle_client, (void*)client_socket_ptr);
        if (create_rc != 0) {
            fprintf(stderr, "Thread creation failed: %s\n", strerror(create_rc));
            close(*client_socket_ptr); // Close socket if thread creation failed
            free(client_socket_ptr);
        } else {
            pthread_detach(thread_id); // Don't need to join client threads
        }
    }

    printf("Accept thread exiting.\n");
    return NULL;
}

// --- Bit Manipulation Macros ---
#define SetBit(A,k)   (A[((k)/32)] |=  (1U << ((k)%32)))
#define TestBit(A,k)  (A[((k)/32)] &   (1U << ((k)%32)))
#define rad2deg 57.2957795130823

// --- CSR Map File I/O ---
int ReadCSRMaps(int expected_rows, size_t expected_cols) {
    const char *hdr_fn = "MapCSR.hdr";
    const char *val_fn = "MapCSR_values.bin";
    const char *col_fn = "MapCSR_col_indices.bin";
    const char *row_fn = "MapCSR_row_ptr.bin";
    FILE *f_hdr = NULL;
    FILE *f_val = NULL;
    FILE *f_col = NULL;
    FILE *f_row = NULL;
    size_t read_val, read_col, read_row;
    int read_rows;
    long long read_cols_ll; // Read potentially large number from file
    long long read_nnz;

    printf("Reading CSR map files...\n");

    // Read Header
    f_hdr = fopen(hdr_fn, "r");
    check(f_hdr == NULL, "Failed to open CSR header file '%s'", hdr_fn);
    // Read as long long for cols, then check against size_t expected_cols
    check(fscanf(f_hdr, "%d\n%lld\n%lld\n", &read_rows, &read_cols_ll, &read_nnz) != 3,
          "Failed to parse CSR header file '%s'", hdr_fn);
    fclose(f_hdr);

    printf(" Header: rows=%d, cols=%lld, nnz=%lld\n", read_rows, read_cols_ll, read_nnz);

    // Validate dimensions
    check(read_rows <= 0, "Invalid number of rows (%d) in CSR header", read_rows);
    check(read_cols_ll <= 0, "Invalid number of cols (%lld) in CSR header", read_cols_ll);
    check(read_nnz < 0, "Invalid non-zero count (%lld) in CSR header", read_nnz);
    check(read_rows != expected_rows, "CSR map row count (%d) mismatch with expected (%d from parameters)", read_rows, expected_rows);
    check((size_t)read_cols_ll != expected_cols, "CSR map column count (%lld) mismatch with expected (%zu from parameters)", read_cols_ll, expected_cols);

    csr_num_rows = read_rows;
    csr_num_cols = (int)expected_cols; // Store as int if it fits, validation passed
    csr_num_nonzeros = read_nnz;

    if (csr_num_nonzeros == 0) {
        printf("Warning: CSR map contains zero non-zero entries. Integration will yield zeros.\n");
        h_csr_values = NULL;
        h_csr_col_indices = NULL;
        // Allocate row_ptr even if nnz=0 (pinned host memory)
        gpuErrchk(cudaMallocHost((void**)&h_csr_row_ptr, (csr_num_rows + 1) * sizeof(int)));
        check(h_csr_row_ptr == NULL, "CSR Read: Failed cudaMallocHost for zero-NNZ row_ptr");
        // Initialize row pointers to 0
        for(int i=0; i <= csr_num_rows; ++i) {
            h_csr_row_ptr[i] = 0;
        }
        return 1; // Success, but map is empty
    }

    // Allocate Pinned Host Memory for CSR data
    printf(" Allocating pinned host memory for CSR arrays (NNZ=%lld)...\n", csr_num_nonzeros);
    gpuErrchk(cudaMallocHost((void**)&h_csr_values, csr_num_nonzeros * sizeof(double)));
    gpuErrchk(cudaMallocHost((void**)&h_csr_col_indices, csr_num_nonzeros * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_csr_row_ptr, (csr_num_rows + 1) * sizeof(int)));
    check(!h_csr_values || !h_csr_col_indices || !h_csr_row_ptr, "CSR Read: Failed cudaMallocHost for CSR arrays");

    // Read Values
    printf(" Reading %s...\n", val_fn);
    f_val = fopen(val_fn, "rb");
    check(f_val == NULL, "Failed to open CSR values file '%s'", val_fn);
    read_val = fread(h_csr_values, sizeof(double), csr_num_nonzeros, f_val);
    check(read_val != (size_t)csr_num_nonzeros, "Failed read CSR values file (read %zu/%lld elements)", read_val, csr_num_nonzeros);
    fclose(f_val);

    // Read Column Indices
    printf(" Reading %s...\n", col_fn);
    f_col = fopen(col_fn, "rb");
    check(f_col == NULL, "Failed to open CSR col_indices file '%s'", col_fn);
    read_col = fread(h_csr_col_indices, sizeof(int), csr_num_nonzeros, f_col);
    check(read_col != (size_t)csr_num_nonzeros, "Failed read CSR col_indices file (read %zu/%lld elements)", read_col, csr_num_nonzeros);
    fclose(f_col);

    // Read Row Pointer
    printf(" Reading %s...\n", row_fn);
    f_row = fopen(row_fn, "rb");
    check(f_row == NULL, "Failed to open CSR row_ptr file '%s'", row_fn);
    read_row = fread(h_csr_row_ptr, sizeof(int), csr_num_rows + 1, f_row);
    check(read_row != (size_t)(csr_num_rows + 1), "Failed read CSR row_ptr file (read %zu/%d elements)", read_row, csr_num_rows + 1);
    fclose(f_row);

    printf("CSR map files read successfully.\n");
    fflush(stdout);
    return 1;
}

void FreeCSRMaps() {
    printf("Freeing host CSR map buffers...\n");
    if(h_csr_values) {
        gpuWarnchk(cudaFreeHost(h_csr_values));
        h_csr_values = NULL;
    }
    if(h_csr_col_indices) {
        gpuWarnchk(cudaFreeHost(h_csr_col_indices));
        h_csr_col_indices = NULL;
    }
    if(h_csr_row_ptr) {
        gpuWarnchk(cudaFreeHost(h_csr_row_ptr));
        h_csr_row_ptr = NULL;
    }
    csr_num_rows = 0;
    csr_num_cols = 0;
    csr_num_nonzeros = 0;
    printf("Host CSR buffers freed.\n");
}


// --- Binning Setup ---
static inline void REtaMapper(double Rmin, double EtaMin, int nEta, int nR, double EtaStep, double RStep, double *EtaLo, double *EtaHi, double *RLo, double *RHi) {
    for(int i = 0; i < nEta; ++i){
        EtaLo[i] = EtaStep * i + EtaMin;
        EtaHi[i] = EtaStep * (i + 1) + EtaMin;
    }
    for(int i = 0; i < nR; ++i){
        RLo[i] = RStep * i + Rmin;
        RHi[i] = RStep * (i + 1) + Rmin;
    }
}

// --- Sequential CPU Image Transformation (for dark processing) ---
static inline void DoImageTransformationsSequential(int Nopt, const int Topt[MAX_TRANSFORM_OPS], const int64_t *In, int64_t *Out, int NY, int NZ) {
    size_t N = (size_t)NY * NZ;
    size_t B = N * sizeof(int64_t);
    bool any = false;
    int64_t *tmp = NULL;
    const int64_t* rB = NULL;
    int64_t* wB = NULL;
    int cY = NY;
    int cZ = NZ;
    size_t cB;
    int nY, nZ;
    size_t fB;

    if(Nopt > 0){
        for(int i = 0; i < Nopt; ++i){
            if(Topt[i] < 0 || Topt[i] > 3){
                fprintf(stderr, "CPU Err: Inv opt %d\n", Topt[i]);
                return;
            }
            if(Topt[i] != 0) {
                any = true;
            }
        }
    }
    if(!any){
        if(Out != In) {
            memcpy(Out, In, B);
        }
        return;
    }

    tmp = (int64_t*)malloc(B);
    if(!tmp){
        fprintf(stderr, "CPU Err: Alloc tmp fail\n");
        if(Out != In) {
            memcpy(Out, In, B); // Try to copy if allocation fails
        }
        return;
    }

    rB = NULL;
    wB = NULL;

    for(int i = 0; i < Nopt; ++i){
        int opt = Topt[i];
        cB = (size_t)cY * cZ * sizeof(int64_t);
        if(i == 0){
            rB = In;
            wB = tmp;
        } else if(i % 2 == 1){
            rB = tmp;
            wB = Out;
        } else {
            rB = Out;
            wB = tmp;
        }
        nY = cY;
        nZ = cZ;
        switch(opt){
            case 0: // No-op
                if(wB != rB) {
                    memcpy(wB, rB, cB);
                }
                break;
            case 1: // Flip Horizontal (around Y axis)
                for(int l = 0; l < cZ; ++l) {
                    for(int k = 0; k < cY; ++k) {
                        wB[l * cY + k] = rB[l * cY + (cY - 1 - k)];
                    }
                }
                break;
            case 2: // Flip Vertical (around Z axis)
                for(int l = 0; l < cZ; ++l) {
                    for(int k = 0; k < cY; ++k) {
                        wB[l * cY + k] = rB[(cZ - 1 - l) * cY + k];
                    }
                }
                break;
            case 3: // Transpose
                if(cY != cZ){
                    fprintf(stderr, "CPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i);
                    if(wB != rB) {
                        memcpy(wB, rB, cB); // Still copy if treating as no-op
                    }
                } else {
                    nY = cZ;
                    nZ = cY;
                    for(int l = 0; l < nZ; ++l) {
                        for(int k = 0; k < nY; ++k) {
                            wB[l * nY + k] = rB[k * cY + l];
                        }
                    }
                }
                break;
        }
        cY = nY; // Update dimensions for next step
        cZ = nZ;
    }

    // Copy result to final destination if needed
    if(Nopt % 2 != 0){ // If odd number of transforms, result is in tmp
        fB = (size_t)cY * cZ * sizeof(int64_t);
        if(fB > B){
            fprintf(stderr, "CPU Err: Final buffer size > Original\n");
            fB = B; // Prevent buffer overflow on copy
        }
        memcpy(Out, tmp, fB);
    } else { // If even number (>0), result is already in Out
        if((size_t)cY * cZ != N) {
             fprintf(stderr, "CPU Warn: Final image size != Original size\n");
        }
    }
    free(tmp);
}

// --- GPU Kernels ---

// Kernel to calculate initial R, TTh, Eta and *masked* Area for each bin using CSR map
__global__ void initialize_PerFrameArr_Area_CSR_kernel(
    double *dPerFrameArr, size_t bigArrSize, // Output: R, TTh, Eta, Area arrays (size 4*bigArrSize)
    int nRBins, int nEtaBins,
    const double *dRBinsLow, const double *dRBinsHigh,
    const double *dEtaBinsLow, const double *dEtaBinsHigh,
    // CSR Map Input
    const double *d_csr_values,
    const int *d_csr_col_indices,
    const int *d_csr_row_ptr,
    // Detector/Mask Info
    int NrPixelsY, int NrPixelsZ,            // Detector dimensions
    const int *dMapMask, size_t mapMaskWordCount, // Pixel mask (optional, dMapMask can be NULL)
    double px, double Lsd)                    // Geometry params
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // Integration bin index
    if (idx >= bigArrSize) {
        return;
    }

    // --- Calculate Static R, TTh, Eta ---
    double RMean = 0.0;
    double EtaMean = 0.0;
    double TwoTheta = 0.0;

    if (nEtaBins > 0 && nRBins > 0) { // Basic check
        int j = idx / nEtaBins; // R bin index
        int k = idx % nEtaBins; // Eta bin index
        if (j < nRBins && k < nEtaBins) { // Bounds check
            RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5;
            EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5;
            if (Lsd > 1e-9) { // Avoid division by zero if Lsd is zero
                TwoTheta = rad2deg * atan(RMean * px / Lsd);
            }
        }
    }

    // --- Calculate Static Total Area for this bin using CSR map & mask ---
    double totArea = 0.0;
    int start_offset = d_csr_row_ptr[idx];
    int end_offset = d_csr_row_ptr[idx + 1];
    size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ; // Cast needed below

    for (int l = start_offset; l < end_offset; ++l) {
        double frac = d_csr_values[l];
        int pixel_idx = d_csr_col_indices[l]; // Linear pixel index

        // Apply Mask if provided
        bool isMasked = false;
        if (dMapMask != NULL && mapMaskWordCount > 0) {
            // Ensure pixel_idx is within range for TestBit
            if (pixel_idx >= 0 && (size_t)pixel_idx < totalPixels) { // Safe cast for comparison
                 if (TestBit(dMapMask, pixel_idx)) {
                    isMasked = true; // Masked pixel
                 }
            } else {
                isMasked = true; // Treat out-of-bounds as masked
            }
        }

        if (!isMasked) {
            totArea += frac; // Accumulate area only if not masked
        }
    }

    // --- Write ALL static values to dPerFrameArr ---
    dPerFrameArr[0 * bigArrSize + idx] = RMean;    // R values start at index 0
    dPerFrameArr[1 * bigArrSize + idx] = TwoTheta; // TwoTheta values start at bigArrSize
    dPerFrameArr[2 * bigArrSize + idx] = EtaMean;  // Eta values start at 2*bigArrSize
    dPerFrameArr[3 * bigArrSize + idx] = totArea;  // Area values start at 3*bigArrSize
}


// Integration kernel using CSR map, NO dynamic detector mask applied
// Relies on Area pre-calculated by initialize_PerFrameArr_Area_CSR_kernel
__global__ void integrate_CSR_noPixelMask(
    size_t bigArrSize, int Normalize, int sumImages, // Control flags
    // CSR Map Input
    const double *d_csr_values,
    const int *d_csr_col_indices,
    const int *d_csr_row_ptr,
    // Input Image & Output Buffers
    const double *dImage,        // Processed input image (double)
    double *dIntArrPerFrame, // Output 2D integrated pattern for this frame
    double *dSumMatrix,      // Optional: Output summed 2D pattern
    const double *dPerFrameArr)  // Input: Pre-calculated Area array (at offset 3*bigArrSize)
{
	const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // Integration bin index
	if (idx >= bigArrSize) {
        return;
    }

	double Intensity = 0.0;
    // Get the pre-calculated area for this bin
    double totArea = dPerFrameArr[3 * bigArrSize + idx];

    if (totArea > AREA_THRESHOLD) { // Only integrate if the bin has significant *unmasked* area
        int start_offset = d_csr_row_ptr[idx];
        int end_offset = d_csr_row_ptr[idx + 1];

        for (int l = start_offset; l < end_offset; ++l) {
            double frac = d_csr_values[l];
            int pixel_idx = d_csr_col_indices[l];
            // Assumes pixel mask was already applied during area pre-calculation
            Intensity += dImage[pixel_idx] * frac;
        }

        // Normalize using the pre-calculated (masked) area if requested
        if (Normalize) {
            Intensity /= totArea;
        }
    } else {
        Intensity = 0.0; // Bin has no area or is fully masked
    }

	dIntArrPerFrame[idx] = Intensity;
	if (sumImages && dSumMatrix) {
		atomicAdd(&dSumMatrix[idx], Intensity);
	}
}


// Transformation Kernels (unchanged from previous versions)
__global__ void sequential_transform_kernel(const int64_t *r, int64_t *w, int cY, int cZ, int nY, int nZ, int opt) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1;
    int zs = -1;

    switch(opt){
        case 0: ys = yo; zs = zo; break; // No-op
        case 1: ys = cY - 1 - yo; zs = zo; break; // Flip Horizontal
        case 2: ys = yo; zs = cZ - 1 - zo; break; // Flip Vertical
        case 3: ys = zo; zs = yo; break; // Transpose (caller ensures square)
        default: return; // Invalid option
    }

    // Read from source location (ys, zs) in input buffer 'r' (dimensions cY, cZ)
    // Write to target location (yo, zo) in output buffer 'w' (dimensions nY, nZ)
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
        w[i] = r[(size_t)zs * cY + ys];
    } else {
        w[i] = 0; // Default: write 0 if source is out of bounds
    }
}

__global__ void final_transform_process_kernel(const int64_t *r, double *o, const double *d, int cY, int cZ, int nY, int nZ, int opt, bool sub) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1;
    int zs = -1;
    double pv = 0.0;

    switch(opt){
        case 0: ys = yo; zs = zo; break; // No-op
        case 1: ys = cY - 1 - yo; zs = zo; break; // Flip Horizontal
        case 2: ys = yo; zs = cZ - 1 - zo; break; // Flip Vertical
        case 3: ys = zo; zs = yo; break; // Transpose
        default: o[i] = 0.0; return; // Invalid option
    }

    // Read from source location (ys, zs) in input buffer 'r' (dimensions cY, cZ)
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
        const int64_t rv = r[(size_t)zs * cY + ys];
        pv = (double)rv; // Cast to double
        // Subtract dark if enabled and dark buffer exists
        if (sub && d) {
            // Assuming dark 'd' has the same dimensions as output 'o' (nY, nZ)
            pv -= d[i];
        }
    }
    // Write final processed value to output buffer 'o' at location i (yo, zo)
    o[i] = pv;
}

__global__ void process_direct_kernel(const int64_t *r, double *o, const double *d, size_t N, bool sub) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int64_t rv = r[i];
        double pv = (double)rv; // Cast
        if (sub && d) {
            pv -= d[i]; // Subtract dark
        }
        o[i] = pv;
    }
}

// 1D Profile Calculation Kernel (unchanged)
// Works on the output of integration (dIntArrPerFrame) and pre-calculated Area
__global__ void calculate_1D_profile_kernel(const double *d_IntArrPerFrame, const double *d_PerFrameArr, double *d_int1D, int nRBins, int nEtaBins, size_t bigArrSize) {
    extern __shared__ double sdata[];
    double * sIntArea = sdata;
    double * sArea    = &sdata[blockDim.x / 32]; // Assumes warpSize 32

    const int r_bin = blockIdx.x;
    if (r_bin >= nRBins) {
        return;
    }

    const int tid = threadIdx.x;
    const int warpSize = 32;
    const int lane = tid % warpSize;
    const int warpId = tid / warpSize;

    if (lane == 0) {
         sIntArea[warpId] = 0.0;
         sArea[warpId] = 0.0;
    }

    double mySumIntArea = 0.0;
    double mySumArea = 0.0;
    for (int eta_bin = tid; eta_bin < nEtaBins; eta_bin += blockDim.x) {
        size_t idx2d = (size_t)r_bin * nEtaBins + eta_bin;
        if (idx2d < bigArrSize) {
            // Access area from the dPerFrameArr (offset 3*bigArrSize)
            double area = d_PerFrameArr[3 * bigArrSize + idx2d];
            if (area > AREA_THRESHOLD) {
                 mySumIntArea += d_IntArrPerFrame[idx2d] * area;
                 mySumArea += area;
            }
        }
    }

    // Warp Level Reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySumIntArea += __shfl_down_sync(0xFFFFFFFF, mySumIntArea, offset);
        mySumArea += __shfl_down_sync(0xFFFFFFFF, mySumArea, offset);
    }

    // Write Warp Result to Shared Memory
    if (lane == 0) {
        atomicAdd(&sIntArea[warpId], mySumIntArea);
        atomicAdd(&sArea[warpId], mySumArea);
    }

    // Block Level Reduction
    __syncthreads();

    // Thread 0 sums results from shared memory
    if (tid == 0) {
        double finalSumIntArea = 0.0;
        double finalSumArea = 0.0;
        int numWarps = blockDim.x / warpSize;
        if (blockDim.x % warpSize != 0) {
            numWarps++;
        }

        for (int i = 0; i < numWarps; ++i) {
             finalSumIntArea += sIntArea[i];
             finalSumArea += sArea[i];
        }

        // Calculate final average intensity for this R bin
        if (finalSumArea > AREA_THRESHOLD) {
            d_int1D[r_bin] = finalSumIntArea / finalSumArea;
        } else {
            d_int1D[r_bin] = 0.0; // Avoid division by zero
        }
    }
}


// --- Host Wrapper for Full GPU Processing Pipeline ---
void ProcessImageGPU(const int64_t *hRaw, double *dProc, const double *dAvgDark, int Nopt, const int Topt[MAX_TRANSFORM_OPS], int NY, int NZ, bool doSub,
	int64_t* d_b1, int64_t* d_b2) {
    const size_t N = (size_t)NY * NZ;
    const size_t B64 = N * sizeof(int64_t);
    const int TPB = THREADS_PER_BLOCK_TRANSFORM;
    bool anyT = false;
    unsigned long long nBUL;
    int mGDX;
    dim3 nB;
    dim3 th(TPB);
    const int64_t* rP = NULL;
    int64_t* wP = NULL;
    int cY = NY;
    int cZ = NZ;
    int nY, nZ;
    size_t sON;
    int fOpt;
    size_t fON;


    // Check for invalid transform options first
    if(Nopt > 0){
        for(int i = 0; i < Nopt; ++i){
            if(Topt[i] < 0 || Topt[i] > 3){
                fprintf(stderr,"GPU Err: Inv opt %d\n", Topt[i]);
                gpuErrchk(cudaMemset(dProc, 0, N * sizeof(double))); // Zero output
                return;
            }
            if(Topt[i] != 0){
                anyT = true;
            }
        }
    }

    // --- Case 1: No non-identity transformations ---
    if(!anyT){
        gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));

        nBUL = (N + TPB - 1) / TPB;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){
            fprintf(stderr, "Block count %llu exceeds max grid dim %d\n", nBUL, mGDX);
            exit(1);
        }
        nB.x = (unsigned int)nBUL; // Set grid dim

        process_direct_kernel<<<nB, th>>>(d_b1, dProc, dAvgDark, N, doSub);
        gpuErrchk(cudaPeekAtLastError());
        return;
    }

    // --- Case 2: One or more transformations needed ---
    gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));

    rP = NULL;
    wP = NULL;
    cY = NY;
    cZ = NZ;

    // --- Apply intermediate transformations sequentially ---
    for(int i = 0; i < Nopt - 1; ++i){
        int opt = Topt[i];
        if(i % 2 == 0){ // Even step (0, 2, ...)
            rP = d_b1;
            wP = d_b2;
        } else { // Odd step (1, 3, ...)
            rP = d_b2;
            wP = d_b1;
        }

        nY = cY;
        nZ = cZ;

        if(opt == 3){ // Handle transpose
            if(cY != cZ){
                fprintf(stderr,"GPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i);
                opt = 0; // Treat as no-op
            } else {
                nY = cZ; // Swap dimensions
                nZ = cY;
            }
        }

        sON = (size_t)nY * nZ;
        nBUL = (sON + TPB - 1) / TPB;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){
            fprintf(stderr,"Blk %llu > max %d (step %d)\n", nBUL, mGDX, i);
            // No cudaFree here, use persistent buffers
            exit(1);
        }
        nB.x = (unsigned int)nBUL;

        sequential_transform_kernel<<<nB, th>>>(rP, wP, cY, cZ, nY, nZ, opt);
        gpuErrchk(cudaPeekAtLastError());

        cY = nY; // Update current dimensions
        cZ = nZ;
    }

    // --- Apply the FINAL transformation and processing ---
    fOpt = Topt[Nopt - 1];
    // Determine the final read buffer
    if((Nopt - 1) % 2 == 0){ // Last intermediate write was to d_b2 (Nopt=1, 3, 5...) -> read from d_b1
        rP = d_b1;
    } else { // Last intermediate write was to d_b1 (Nopt=2, 4...) -> read from d_b2
        rP = d_b2;
    }

    nY = cY; // Final dimensions start as current dimensions
    nZ = cZ;
    if(fOpt == 3){ // Handle final transpose
        if(cY != cZ){
            fprintf(stderr,"GPU Warn: Skip Tpose %dx%d final st %d\n", cY, cZ, Nopt - 1);
            fOpt = 0; // Treat as no-op
        } else {
            nY = cZ; // Swap dimensions
            nZ = cY;
        }
    }

    fON = (size_t)nY * nZ;
    if(fON != N){
        fprintf(stderr, "GPU Warn: Final image size %zu != Original size %zu\n", fON, N);
        // Proceed with caution, dProc might be wrong size if transforms change dims
    }

    nBUL = (fON + TPB - 1) / TPB;
    gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
    if(nBUL > (unsigned long long)mGDX){
        fprintf(stderr,"Blk %llu > max %d (final step)\n", nBUL, mGDX);
        exit(1);
    }
    nB.x = (unsigned int)nBUL;

    final_transform_process_kernel<<<nB, th>>>(rP, dProc, dAvgDark, cY, cZ, nY, nZ, fOpt, doSub);
    gpuErrchk(cudaPeekAtLastError());
}


// --- Peak Fitting Data Structures and Functions ---
typedef struct {
    int nrBins;
    const double *R; // Radial positions (X-axis)
    const double *Int; // Intensity values (Y-axis)
} dataFit;

typedef struct {
    int index;       // Index in the R/Int array
    double radius;   // R value at the peak
    double intensity;// Intensity value at the peak
} Peak;

// Objective function for NLopt: Calculates sum of squared errors for multiple Pseudo-Voigt peaks + global background
static double problem_function_global_bg(unsigned n, const double *x, double *grad, void *fdat) {
    const dataFit *d = (const dataFit*)fdat;
    const int Np_bins = d->nrBins;    // Number of data points (bins)
    const double *Rs = d->R;     // R values (data)
    const double *Is = d->Int;   // Intensity values (data)
    const int nP_peaks = (n - 1) / 4; // Number of peaks
    const double bg_g = x[n - 1]; // Global background parameter
    double total_sq_error = 0.0;
    double *calculated_I = NULL;

    // Gradient calculation is not implemented
    if(grad){
        for(unsigned i = 0; i < n; ++i){
            grad[i] = 0.0;
        }
    }

    if(nP_peaks <= 0 || (4 * nP_peaks + 1 != n)){
        fprintf(stderr, "Obj Func Err: Invalid number of parameters %u for %d peaks\n", n, nP_peaks);
        return INFINITY; // Return infinity for invalid input
    }

    calculated_I = (double*)calloc(Np_bins, sizeof(double));
    if(!calculated_I){
        fprintf(stderr, "Obj Func Err: Failed to allocate memory for calculated profile\n");
        return INFINITY; // Return infinity on allocation failure
    }

    // Calculate the contribution of each peak
    for(int pN = 0; pN < nP_peaks; ++pN){
        // Extract parameters for the current peak (pN)
        double Amplitude = x[pN * 4 + 0];
        double Mix       = fmax(0.0, fmin(1.0, x[pN * 4 + 1])); // Constrain mix [0, 1]
        double Center    = x[pN * 4 + 2];
        double Sigma     = fmax(1e-9, x[pN * 4 + 3]); // Constrain sigma > 0

        // Add this peak's contribution
        for(int i = 0; i < Np_bins; ++i){
            double diff = Rs[i] - Center;
            double diff_sq = diff * diff;
            double sigma_sq = Sigma * Sigma;
            double gaussian = 0.0;
            double lorentzian = 0.0;

            // Gaussian component (avoid division by zero/small sigma)
            if (Sigma > 1e-9) {
                 gaussian = exp(-diff_sq / (2.0 * sigma_sq)) / (Sigma * sqrt(2.0 * M_PI));
            }
            // Lorentzian component (avoid division by zero/small sigma)
            if (Sigma > 1e-9) {
                lorentzian = (1.0 / (M_PI * Sigma)) / (1.0 + diff_sq / sigma_sq);
            }

            calculated_I[i] += Amplitude * (Mix * gaussian + (1.0 - Mix) * lorentzian);
        }
    }

    // Add global background and calculate squared error
    for(int i = 0; i < Np_bins; ++i){
        calculated_I[i] += bg_g; // Add global background
        double error = calculated_I[i] - Is[i];
        total_sq_error += error * error; // Accumulate squared error
    }

    free(calculated_I); // Free the temporary buffer
    return total_sq_error; // Return the total squared error
}

// Apply Savitzky-Golay smoothing filter
void smoothData(const double *in, double *out, int N, int W) {
    double *coeffs = NULL;
    double norm = 0.0;
    int H;

    // W = Window size (must be odd, >= 3)
    if(W < 3 || W % 2 == 0){
        memcpy(out, in, N * sizeof(double)); // No smoothing
        return;
    }
    H = W / 2; // Half-window size

    coeffs = (double*)malloc(W * sizeof(double));
    check(!coeffs, "smoothData: Malloc failed for coefficients");

    switch(W){
        case 5: norm = 35.0; coeffs[0]=-3; coeffs[1]=12; coeffs[2]=17; coeffs[3]=12; coeffs[4]=-3; break;
        case 7: norm = 21.0; coeffs[0]=-2; coeffs[1]= 3; coeffs[2]= 6; coeffs[3]= 7; coeffs[4]= 6; coeffs[5]= 3; coeffs[6]=-2; break;
        case 9: norm = 231.0; coeffs[0]=-21; coeffs[1]=14; coeffs[2]=39; coeffs[3]=54; coeffs[4]=59; coeffs[5]=54; coeffs[6]=39; coeffs[7]=14; coeffs[8]=-21; break;
        default:
            fprintf(stderr, "smoothData Warn: Unsupported window size %d. No smoothing applied.\n", W);
            memcpy(out, in, N * sizeof(double));
            free(coeffs);
            return;
    }

    // Normalize coefficients
    for(int i = 0; i < W; ++i){
        coeffs[i] /= norm;
    }

    // Apply the filter
    for(int i = 0; i < N; ++i){
        if(i < H || i >= N - H){
            out[i] = in[i]; // Handle boundaries: copy original data
        } else {
            double smoothed_value = 0.0;
            for(int j = 0; j < W; ++j){
                smoothed_value += coeffs[j] * in[i - H + j];
            }
            out[i] = smoothed_value;
        }
    }
    free(coeffs);
}

// Simple peak finding: detects local maxima above a threshold and applies minimum distance constraint
int findPeaks(const double *data, const double *r_values, int N, Peak **foundPeaks, double minHeight, int minDistance) {
    int maxPossiblePeaks;
    Peak* preliminaryPeaks = NULL;
    int peakCount = 0;
    bool* isSuppressed = NULL;
    Peak* filteredPeaks = NULL;
    int filteredCount = 0;
    Peak* finalPeaks = NULL;

    if(N < 3){
        *foundPeaks = NULL;
        return 0; // Cannot find peaks in less than 3 points
    }

    maxPossiblePeaks = N / 2 + 1;
    preliminaryPeaks = (Peak*)malloc(maxPossiblePeaks * sizeof(Peak));
    check(!preliminaryPeaks, "findPeaks: Malloc failed for preliminaryPeaks");

    // --- Step 1: Find all local maxima above minHeight ---
    for(int i = 1; i < N - 1; ++i){
        if(data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] >= minHeight){
            if(peakCount < maxPossiblePeaks){
                preliminaryPeaks[peakCount].index = i;
                preliminaryPeaks[peakCount].radius = r_values[i];
                preliminaryPeaks[peakCount].intensity = data[i];
                peakCount++;
            } else {
                fprintf(stderr, "Peak find warn: Exceeded max possible peaks buffer.\n");
                break;
            }
        }
    }

    if(peakCount == 0 || minDistance <= 1){
        // No peaks found, or no distance filtering needed
        finalPeaks = (Peak*)realloc(preliminaryPeaks, peakCount * sizeof(Peak));
        if(peakCount > 0 && finalPeaks == NULL) {
             *foundPeaks = preliminaryPeaks; // realloc failed, return original buffer
             fprintf(stderr, "findPeaks Warn: realloc failed, returning potentially oversized buffer\n");
        } else {
             *foundPeaks = finalPeaks; // Can be NULL if peakCount is 0
        }
        return peakCount;
    }

    // --- Step 2: Apply minimum distance constraint ---
    isSuppressed = (bool*)calloc(peakCount, sizeof(bool));
    check(!isSuppressed, "findPeaks: Calloc failed for isSuppressed");

    for(int i = 0; i < peakCount; ++i){
        if(isSuppressed[i]) {
            continue; // Skip if already marked for suppression
        }
        for(int j = i + 1; j < peakCount; ++j){
            if(isSuppressed[j]) {
                continue;
            }
            int distance = abs(preliminaryPeaks[i].index - preliminaryPeaks[j].index);
            if(distance < minDistance){
                // Suppress the smaller peak
                if(preliminaryPeaks[j].intensity <= preliminaryPeaks[i].intensity) {
                    isSuppressed[j] = true;
                } else {
                    isSuppressed[i] = true;
                    break; // Peak i is suppressed, move to next i
                }
            }
        }
    }

    // --- Step 3: Create the final filtered list ---
    filteredPeaks = (Peak*)malloc(peakCount * sizeof(Peak));
    check(!filteredPeaks, "findPeaks: Malloc failed for filteredPeaks");
    filteredCount = 0;
    for(int i = 0; i < peakCount; ++i){
        if(!isSuppressed[i]){
            filteredPeaks[filteredCount++] = preliminaryPeaks[i];
        }
    }

    free(preliminaryPeaks);
    free(isSuppressed);

    // Reallocate final array to the exact size
    finalPeaks = (Peak*)realloc(filteredPeaks, filteredCount * sizeof(Peak));
     if(filteredCount > 0 && finalPeaks == NULL) {
         *foundPeaks = filteredPeaks; // Realloc failed, return potentially oversized buffer
         fprintf(stderr, "findPeaks Warn: realloc failed, returning potentially oversized buffer\n");
     } else {
        *foundPeaks = finalPeaks; // Can be NULL if filteredCount is 0
     }
    return filteredCount;
}


// =========================================================================
// ============================ MAIN FUNCTION ============================
// =========================================================================
int main(int argc, char *argv[]){
    // --- Declarations ---
    double t_start_main, t_start_map, t_start_params, t_start_dark, t_end_setup;
    int dev_id = 0;
    cudaDeviceProp prop;
    double RMax = 0, RMin = 0, RBinSize = 0;
    double EtaMax = 0, EtaMin = 0, EtaBinSize = 0;
    double Lsd = 0, px = 0;
    int NrPixelsY = 0, NrPixelsZ = 0;
    int Normalize = 1;
    int nEtaBins = 0, nRBins = 0;
    char *ParamFN = NULL;
    FILE *pF = NULL;
    char line[4096], key[1024], val_str[3072];
    int Nopt = 0;
    long long GapI = 0, BadPxI = 0;
    int Topt[MAX_TRANSFORM_OPS] = {0};
    int mkMap = 0;
    int sumI = 0;
    int doSm = 0;
    int multiP = 0;
    int pkFit = 0;
    int nSpecP = 0;
    int wr2D = 0;
    double pkLoc[MAX_PEAK_LOCATIONS];
    size_t bigArrSize;
    size_t totalPixels;
    double *hEtaLo = NULL, *hEtaHi = NULL, *hRLo = NULL, *hRHi = NULL;
    double *hAvgDark = NULL;
    int64_t *hDarkInT = NULL;
    int64_t *hDarkIn = NULL;
    size_t SizeFile;
    double *dAvgDark = NULL;
    double *dProcessedImage = NULL;
    double *d_int1D = NULL;
    int *dMapMask = NULL;
    size_t mapMaskWC = 0;
    double *dSumMatrix = NULL;
    double *dIntArrFrame = NULL;
    double *dPerFrame = NULL;
    double *dEtaLo = NULL, *dEtaHi = NULL, *dRLo = NULL, *dRHi = NULL;
    int64_t *g_dTempTransformBuf1 = NULL;
    int64_t *g_dTempTransformBuf2 = NULL;
    bool darkSubEnabled = false;
    int nDarkFramesRead = 0;
    int *hMapMask = NULL;
    int server_fd;
    struct sockaddr_in server_addr;
    pthread_t accept_thread;
    FILE *fLineout = NULL;
    FILE *fFit = NULL;
    FILE *f2D = NULL;
    cudaEvent_t ev_proc_start, ev_proc_stop;
    cudaEvent_t ev_integ_start, ev_integ_stop;
    cudaEvent_t ev_prof_start, ev_prof_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    float t_proc_gpu = 0, t_integ_gpu = 0, t_prof_gpu = 0, t_d2h_gpu = 0;
    double t_qp_cpu = 0, t_write1d_cpu = 0, t_fit_cpu = 0, t_writefit_cpu = 0, t_write2d_cpu = 0, t_loop_cpu = 0;
    double t_start_loop, t_end_loop;
    int firstFrame = 1;
    double *hIntArrFrame = NULL;
    double *hPerFrame = NULL;
    double *h_int1D = NULL;
    double *hR = NULL;
    double *hEta = NULL;
    double *hLineout = NULL;
    int frameCounter = 0;
    DataChunk chunk;
    int queue_rc;
    int currFidx;
    int integTPB;
    int nrVox;
    size_t profileSharedMem;
    size_t written;
    double maxInt;
    int maxIntLoc;
    int currentPeakCount;
    double *sendFitParams = NULL;
    Peak *pks = NULL;
    int validPeakCount;
    int bestBin;
    double minDiff;
    double *dataToFindPeaks = NULL;
    double *smoothedData = NULL;
    int nFitParams;
    double *fitParams = NULL;
    double *lowerBounds = NULL;
    double *upperBounds = NULL;
    double maxOverallIntensity;
    dataFit fitData;
    nlopt_opt opt;
    double minObjectiveValue;
    int nlopt_rc;
    double globalBG;
    int sock_opt;
    int cancel_ret;
    void *thread_result;
    int join_ret;


    // --- Argument parsing ---
    if (argc < 2){
        printf("Usage: %s ParamFN [DarkAvgFN]\n", argv[0]);
        printf(" Args:\n");
        printf("  ParamFN:   Path to parameter file.\n");
        printf("  DarkAvgFN: Optional path to dark frame file (binary int64_t, averaged if multiple frames).\n");
        return 1;
    }
    ParamFN = argv[1]; // Store parameter file name
    printf("[%s] - Starting...\n", argv[0]);
    t_start_main = get_wall_time_ms();

    // --- Setup Signal Handling ---
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // --- Initialize GPU ---
    dev_id = 0;
    gpuErrchk(cudaSetDevice(dev_id));
    gpuErrchk(cudaGetDeviceProperties(&prop, dev_id));
    printf("GPU Device %d: %s (CC %d.%d)\n", dev_id, prop.name, prop.major, prop.minor);
    printf("Init GPU: %.3f ms\n", get_wall_time_ms() - t_start_main);

    // --- Read Parameters ---
    t_start_params = get_wall_time_ms();
    pF = fopen(ParamFN, "r");
    check(!pF, "Failed open param file: %s", ParamFN);

    while(fgets(line, sizeof(line), pF)){
        if(line[0] == '#' || isspace(line[0]) || strlen(line) < 3) {
            continue;
        }
        if (sscanf(line, "%1023s %[^\n]", key, val_str) == 2) {
             if (strcmp(key, "EtaBinSize") == 0) { sscanf(val_str, "%lf", &EtaBinSize); }
             else if (strcmp(key, "RBinSize") == 0) { sscanf(val_str, "%lf", &RBinSize); }
             else if (strcmp(key, "RMax") == 0) { sscanf(val_str, "%lf", &RMax); }
             else if (strcmp(key, "RMin") == 0) { sscanf(val_str, "%lf", &RMin); }
             else if (strcmp(key, "EtaMax") == 0) { sscanf(val_str, "%lf", &EtaMax); }
             else if (strcmp(key, "EtaMin") == 0) { sscanf(val_str, "%lf", &EtaMin); }
             else if (strcmp(key, "Lsd") == 0) { sscanf(val_str, "%lf", &Lsd); }
             else if (strcmp(key, "px") == 0) { sscanf(val_str, "%lf", &px); }
             else if (strcmp(key, "NrPixelsY") == 0) { sscanf(val_str, "%d", &NrPixelsY); }
             else if (strcmp(key, "NrPixelsZ") == 0) { sscanf(val_str, "%d", &NrPixelsZ); }
             else if (strcmp(key, "NrPixels") == 0) { sscanf(val_str, "%d", &NrPixelsY); NrPixelsZ = NrPixelsY; }
             else if (strcmp(key, "Normalize") == 0) { sscanf(val_str, "%d", &Normalize); }
             else if (strcmp(key, "GapIntensity") == 0) { sscanf(val_str, "%lld", &GapI); mkMap = 1; }
             else if (strcmp(key, "BadPxIntensity") == 0) { sscanf(val_str, "%lld", &BadPxI); mkMap = 1; }
             else if (strcmp(key, "ImTransOpt") == 0) {
                 if(Nopt < MAX_TRANSFORM_OPS) { sscanf(val_str, "%d", &Topt[Nopt++]); }
                 else { printf("Warn: Max %d ImTransOpt reached, ignoring further options.\n", MAX_TRANSFORM_OPS); }
             }
             else if (strcmp(key, "SumImages") == 0) { sscanf(val_str, "%d", &sumI); }
             else if (strcmp(key, "Write2D") == 0) { sscanf(val_str, "%d", &wr2D); }
             else if (strcmp(key, "DoSmoothing") == 0) { sscanf(val_str, "%d", &doSm); }
             else if (strcmp(key, "MultiplePeaks") == 0) { sscanf(val_str, "%d", &multiP); }
             else if (strcmp(key, "DoPeakFit") == 0) { sscanf(val_str, "%d", &pkFit); }
             else if (strcmp(key, "PeakLocation") == 0) {
                 if(nSpecP < MAX_PEAK_LOCATIONS) {
                     sscanf(val_str, "%lf", &pkLoc[nSpecP++]);
                     multiP = 1; // Implicitly enable multi-peak if locations given
                     pkFit = 1;  // Implicitly enable fitting
                     doSm = 0;   // Disable smoothing
                 } else {
                     printf("Warn: Max %d PeakLocation reached, ignoring further locations.\n", MAX_PEAK_LOCATIONS);
                 }
             }
        }
    }
    fclose(pF);

    // Validate essential parameters
    check(NrPixelsY <= 0 || NrPixelsZ <= 0, "NrPixelsY/Z invalid or not set in parameter file.");
    check(Lsd <= 0 || px <= 0, "Lsd/px invalid or not set in parameter file.");
    if(pkFit && nSpecP > 0) {
        multiP = 1;
        if (doSm) {
            printf("Warn: Smoothing disabled because specific PeakLocations were provided.\n");
            doSm = 0; // Don't smooth if fitting specific locations
        }
    }
    nRBins = (RBinSize > 1e-9) ? (int)ceil((RMax - RMin) / RBinSize) : 0;
    nEtaBins = (EtaBinSize > 1e-9) ? (int)ceil((EtaMax - EtaMin) / EtaBinSize) : 0;
    check(nRBins <= 0 || nEtaBins <= 0, "Invalid bin parameters. R bins=%d, Eta bins=%d", nRBins, nEtaBins);
    bigArrSize = (size_t)nRBins * nEtaBins;
    totalPixels = (size_t)NrPixelsY * NrPixelsZ;

    printf("Parameters Loaded:\n");
    printf(" R Bins:    [%.3f .. %.3f], %d bins (step %.4f)\n", RMin, RMax, nRBins, RBinSize);
    printf(" Eta Bins:  [%.3f .. %.3f], %d bins (step %.4f)\n", EtaMin, EtaMax, nEtaBins, EtaBinSize);
    printf(" Detector:  %d x %d pixels (%zu total)\n", NrPixelsY, NrPixelsZ, totalPixels);
    printf(" Geometry:  Lsd=%.4f, px=%.6f\n", Lsd, px);
    printf(" Transforms(%d):", Nopt);
    for(int i = 0; i < Nopt; ++i) { printf(" %d", Topt[i]); }
    printf("\n");
    printf(" Options:   Normalize=%d, SumIntegrations=%d, Write2D=%d\n", Normalize, sumI, wr2D);
    printf(" Peak Fit:  Enabled=%d, MultiPeak=%d, Smooth=%d, NumSpecifiedPeaks=%d\n", pkFit, multiP, doSm, nSpecP);
    if (mkMap) { printf(" Masking:   Will generate from Gap=%lld, BadPx=%lld in Dark Frame\n", GapI, BadPxI); }
    printf("Read Params: %.3f ms\n", get_wall_time_ms() - t_start_params);
    fflush(stdout);

    // --- Read CSR Pixel Mapping Files ---
    t_start_map = get_wall_time_ms();
    check(ReadCSRMaps(bigArrSize, totalPixels) != 1, "Failed read/validate CSR map files");
    printf("Read CSR Maps: %.3f ms\n", get_wall_time_ms() - t_start_map);

    // --- Setup Bin Edges (Host) ---
    hEtaLo = (double*)malloc(nEtaBins * sizeof(double));
    hEtaHi = (double*)malloc(nEtaBins * sizeof(double));
    hRLo   = (double*)malloc(nRBins * sizeof(double));
    hRHi   = (double*)malloc(nRBins * sizeof(double));
    check(!hEtaLo || !hEtaHi || !hRLo || !hRHi, "Allocation failed for host bin edge arrays");
    REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, hEtaLo, hEtaHi, hRLo, hRHi);

    // --- Host Memory Allocations ---
    hAvgDark = (double*)calloc(totalPixels, sizeof(double));
    check(!hAvgDark, "Allocation failed for hAvgDark");
    SizeFile = totalPixels * BYTES_PER_PIXEL;
    hDarkInT = (int64_t*)malloc(SizeFile);
    check(!hDarkInT, "Allocation failed for hDarkInT");
    hDarkIn = (int64_t*)malloc(SizeFile);
    check(!hDarkIn, "Allocation failed for hDarkIn");

    // --- Device Memory Allocations (Persistent) ---
    darkSubEnabled = (argc > 2);
    gpuErrchk(cudaMalloc(&dProcessedImage, totalPixels * sizeof(double)));
    // Allocate CSR map on GPU (pointers declared globally)
    if (csr_num_nonzeros > 0) {
        printf("Allocating GPU memory for CSR arrays (NNZ=%lld)...\n", csr_num_nonzeros);
        gpuErrchk(cudaMalloc(&d_csr_values, csr_num_nonzeros * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_csr_col_indices, csr_num_nonzeros * sizeof(int)));
    } else {
        d_csr_values = NULL;
        d_csr_col_indices = NULL;
    }
    gpuErrchk(cudaMalloc(&d_csr_row_ptr, (csr_num_rows + 1) * sizeof(int))); // Always needed
    gpuErrchk(cudaMalloc(&dIntArrFrame, bigArrSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&dPerFrame, bigArrSize * 4 * sizeof(double))); // R, TTh, Eta, Area
    gpuErrchk(cudaMalloc(&dEtaLo, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dEtaHi, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRLo, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRHi, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_int1D, nRBins * sizeof(double)));
    size_t tempBufferSize = totalPixels * sizeof(int64_t);
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf1, tempBufferSize));
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf2, tempBufferSize));

    // Copy map data and bin edges to GPU
    if (csr_num_nonzeros > 0 && h_csr_values && h_csr_col_indices) {
         printf("Copying CSR arrays from host (pinned) to GPU...\n");
         gpuErrchk(cudaMemcpy(d_csr_values, h_csr_values, csr_num_nonzeros * sizeof(double), cudaMemcpyHostToDevice));
         gpuErrchk(cudaMemcpy(d_csr_col_indices, h_csr_col_indices, csr_num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    }
    check(h_csr_row_ptr != NULL, "Host CSR row pointer (h_csr_row_ptr) is NULL before GPU copy");
    gpuErrchk(cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (csr_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    printf("CSR map copied to GPU.\n");
    gpuErrchk(cudaMemcpy(dEtaLo, hEtaLo, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaHi, hEtaHi, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRLo, hRLo, nRBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRHi, hRHi, nRBins * sizeof(double), cudaMemcpyHostToDevice));

    // --- Process Dark Frame (Mask generation happens here) ---
    t_start_dark = get_wall_time_ms();
    nDarkFramesRead = 0;
    hMapMask = NULL;
    dMapMask = NULL; // Ensure it's NULL initially
    mapMaskWC = 0;

    if(darkSubEnabled){
        char* darkFN = argv[2];
        FILE* fD = fopen(darkFN, "rb");
        check(!fD, "Failed to open dark frame file: %s", darkFN);
        fseek(fD, 0, SEEK_END);
        size_t szD = ftell(fD);
        rewind(fD);
        int nFD = szD / SizeFile;
        check(nFD == 0 || szD % SizeFile != 0, "Dark file %s incomplete (size %zu, frame %zu). Found %d frames.", darkFN, szD, SizeFile, nFD);
        printf("Reading dark file: %s, Found %d frames.\n", darkFN, nFD);

        for(int i = 0; i < nFD; ++i){
            check(fread(hDarkInT, 1, SizeFile, fD) != SizeFile, "Read failed for dark frame %d from %s", i, darkFN);
            DoImageTransformationsSequential(Nopt, Topt, hDarkInT, hDarkIn, NrPixelsY, NrPixelsZ);

            if(mkMap == 1 && i == 0){
                mapMaskWC = (totalPixels + 31) / 32;
                hMapMask = (int*)calloc(mapMaskWC, sizeof(int));
                check(!hMapMask, "Allocation failed for host mask buffer");
                int maskedPixelCount = 0;
                for(size_t j = 0; j < totalPixels; ++j){
                    if(hDarkIn[j] == GapI || hDarkIn[j] == BadPxI){
                        SetBit(hMapMask, j);
                        maskedPixelCount++;
                    }
                }
                printf("Mask generated from first dark frame: %d pixels masked.\n", maskedPixelCount);
                gpuErrchk(cudaMalloc(&dMapMask, mapMaskWC * sizeof(int)));
                gpuErrchk(cudaMemcpy(dMapMask, hMapMask, mapMaskWC * sizeof(int), cudaMemcpyHostToDevice));
                mkMap = 0; // Mask generation done
            }
            for(size_t j = 0; j < totalPixels; ++j){
                hAvgDark[j] += (double)hDarkIn[j];
            }
        }
        fclose(fD);
        nDarkFramesRead = nFD;

        if(nDarkFramesRead > 0){
            for(size_t j = 0; j < totalPixels; ++j){
                hAvgDark[j] /= (double)nDarkFramesRead;
            }
            printf("Averaged %d dark frames.\n", nDarkFramesRead);
            gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
            gpuErrchk(cudaMemcpy(dAvgDark, hAvgDark, totalPixels * sizeof(double), cudaMemcpyHostToDevice));
            printf("Average dark frame copied to GPU.\n");
        } else {
             gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
             gpuErrchk(cudaMemset(dAvgDark, 0, totalPixels * sizeof(double)));
             printf("Dark file specified but no frames read/averaged. Using zeros on GPU.\n");
        }
    } else {
        gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
        gpuErrchk(cudaMemset(dAvgDark, 0, totalPixels * sizeof(double)));
        printf("No dark frame provided, using zeros on GPU.\n");
    }

    // Initialize static R, TTh, Eta, and Area using the CSR map and pixel mask
    printf("Initializing static PerFrame array (R, TTh, Eta, Area) using CSR map on GPU...\n");
    int initTPB = 256;
    int initBlocks = (bigArrSize + initTPB - 1) / initTPB;
    initialize_PerFrameArr_Area_CSR_kernel<<<initBlocks, initTPB>>>(
        dPerFrame, bigArrSize,
        nRBins, nEtaBins,
        dRLo, dRHi, dEtaLo, dEtaHi,
        d_csr_values, d_csr_col_indices, d_csr_row_ptr, // CSR Map Data
        NrPixelsY, NrPixelsZ, // Detector Info
        dMapMask, mapMaskWC,  // Pixel mask (dMapMask might be NULL)
        px, Lsd               // Geometry
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("GPU PerFrame array initialization complete (Area pre-calculated %s pixel mask).\n",
           (dMapMask != NULL ? "with" : "without"));

    if(sumI){
        gpuErrchk(cudaMalloc(&dSumMatrix, bigArrSize * sizeof(double)));
        gpuErrchk(cudaMemset(dSumMatrix, 0, bigArrSize * sizeof(double)));
    }
    printf("Processed dark/mask: %.3f ms\n", get_wall_time_ms() - t_start_dark);
    fflush(stdout);

    // --- Network Setup ---
    CHUNK_SIZE = SizeFile;
    TOTAL_MSG_SIZE = HEADER_SIZE + CHUNK_SIZE;
    printf("Network: Expecting %zu B header + %zu B data = %zu B total per message.\n", HEADER_SIZE, CHUNK_SIZE, TOTAL_MSG_SIZE);
    queue_init(&process_queue);
    check((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0, "Socket creation failed");
    sock_opt = 1;
    check(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &sock_opt, sizeof(sock_opt)), "setsockopt failed");
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    check(bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0, "Bind failed for port %d", PORT);
    check(listen(server_fd, MAX_CONNECTIONS) < 0, "Listen failed");
    printf("Server listening on port %d\n", PORT);
    check(pthread_create(&accept_thread, NULL, accept_connections, &server_fd) != 0, "Failed to create accept thread");

    // --- Prepare for Main Loop ---
    fLineout = fopen("lineout.bin", "wb");
    check(!fLineout, "Error opening lineout.bin for writing");
    if(pkFit){
        fFit = fopen("fit.bin", "wb");
        check(!fFit, "Error opening fit.bin for writing");
    }
    if (wr2D) {
        printf("Will write all 2D integrated patterns to single file: Int2D.bin\n");
        f2D = fopen("Int2D.bin", "wb");
        check(!f2D, "Error opening Int2D.bin for writing");
    }

    // CUDA Events
    gpuErrchk(cudaEventCreate(&ev_proc_start)); gpuErrchk(cudaEventCreate(&ev_proc_stop));
    gpuErrchk(cudaEventCreate(&ev_integ_start)); gpuErrchk(cudaEventCreate(&ev_integ_stop));
    gpuErrchk(cudaEventCreate(&ev_prof_start)); gpuErrchk(cudaEventCreate(&ev_prof_stop));
    gpuErrchk(cudaEventCreate(&ev_d2h_start)); gpuErrchk(cudaEventCreate(&ev_d2h_stop));

    // Host result buffers (pinned for async copies)
    firstFrame = 1;
    if (wr2D) {
        gpuErrchk(cudaMallocHost((void**)&hIntArrFrame, bigArrSize * sizeof(double)));
        check(!hIntArrFrame, "Allocation failed for pinned hIntArrFrame");
    }
    gpuErrchk(cudaMallocHost((void**)&hPerFrame, bigArrSize * 4 * sizeof(double)));
    check(!hPerFrame, "Allocation failed for pinned hPerFrame");
    gpuErrchk(cudaMallocHost((void**)&h_int1D, nRBins * sizeof(double)));
    check(!h_int1D, "Allocation failed for pinned host buffer h_int1D");
    hR = (double*)calloc(nRBins, sizeof(double));
    check(!hR, "Allocation failed for hR");
    hEta = (double*)calloc(nEtaBins, sizeof(double));
    check(!hEta, "Allocation failed for hEta");
    hLineout = (double*)malloc(nRBins * 2 * sizeof(double));
    check(!hLineout, "Allocation failed for hLineout");

    printf("Setup complete. Starting main processing loop...\n");
    t_end_setup = get_wall_time_ms();
    printf("Total setup time: %.3f ms\n", t_end_setup - t_start_main);
    fflush(stdout);

    // =========================== Main Processing Loop ===========================
    frameCounter = 0;
    while (keep_running) {
        t_start_loop = get_wall_time_ms();

        // --- Get next data chunk from queue ---
        double t_qp_start = get_wall_time_ms();
        queue_rc = queue_pop(&process_queue, &chunk);
        if(queue_rc == -1){ // Shutdown requested
            break;
        }
        if(queue_rc == -2){ // Queue empty, but no shutdown
            usleep(10000); // Sleep briefly to avoid busy-waiting
            continue;
        }
        t_qp_cpu = get_wall_time_ms() - t_qp_start;

        // --- GPU Processing Stage (Transform, Cast, Subtract Dark) ---
        gpuErrchk(cudaEventRecord(ev_proc_start, 0));
        ProcessImageGPU(chunk.data, dProcessedImage, dAvgDark, Nopt, Topt, NrPixelsY, NrPixelsZ, darkSubEnabled,
                        g_dTempTransformBuf1, g_dTempTransformBuf2);
        gpuErrchk(cudaEventRecord(ev_proc_stop, 0));

        // --- GPU Integration Stage (2D Integration using CSR) ---
        currFidx = chunk.dataset_num;
        integTPB = THREADS_PER_BLOCK_INTEGRATE;
        nrVox = (bigArrSize + integTPB - 1) / integTPB;

        gpuErrchk(cudaEventRecord(ev_integ_start, 0));
        // Use kernel that relies on pre-calculated area (including mask effects)
        integrate_CSR_noPixelMask<<<nrVox, integTPB>>>(
                bigArrSize, Normalize, sumI,
                d_csr_values, d_csr_col_indices, d_csr_row_ptr,
                dProcessedImage, dIntArrFrame, dSumMatrix,
                dPerFrame); // Pass pre-calculated area array
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_integ_stop, 0));

        // --- GPU 1D Profile Stage (Reduction from 2D integrated) ---
        profileSharedMem = (THREADS_PER_BLOCK_PROFILE / 32) * sizeof(double) * 2;
        gpuErrchk(cudaEventRecord(ev_prof_start, 0));
        calculate_1D_profile_kernel<<<nRBins, THREADS_PER_BLOCK_PROFILE, profileSharedMem>>>(
            dIntArrFrame, dPerFrame, d_int1D, nRBins, nEtaBins, bigArrSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_prof_stop, 0));

        // --- D->H Copy Stage (Asynchronous) ---
        gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
        gpuErrchk(cudaMemcpyAsync(h_int1D, d_int1D, nRBins * sizeof(double), cudaMemcpyDeviceToHost, 0));
        if(firstFrame == 1){
             gpuErrchk(cudaMemcpyAsync(hPerFrame, dPerFrame, bigArrSize * 4 * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        if(wr2D && hIntArrFrame){
            gpuErrchk(cudaMemcpyAsync(hIntArrFrame, dIntArrFrame, bigArrSize * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        gpuErrchk(cudaEventRecord(ev_d2h_stop, 0));

        // --- Synchronize GPU Events and Get Timings ---
        gpuErrchk(cudaEventSynchronize(ev_proc_stop));
        gpuErrchk(cudaEventElapsedTime(&t_proc_gpu, ev_proc_start, ev_proc_stop));
        gpuErrchk(cudaEventSynchronize(ev_integ_stop));
        gpuErrchk(cudaEventElapsedTime(&t_integ_gpu, ev_integ_start, ev_integ_stop));
        gpuErrchk(cudaEventSynchronize(ev_prof_stop));
        gpuErrchk(cudaEventElapsedTime(&t_prof_gpu, ev_prof_start, ev_prof_stop));
        gpuErrchk(cudaEventSynchronize(ev_d2h_stop));
        gpuErrchk(cudaEventElapsedTime(&t_d2h_gpu, ev_d2h_start, ev_d2h_stop));

        // --- CPU Processing Stage ---

        // Initialize Host R/Eta Arrays (First Frame Only)
        if(firstFrame == 1){
            for(int r = 0; r < nRBins; ++r){
                hR[r] = hPerFrame[r * nEtaBins + 0 * bigArrSize]; // R is at offset 0
            }
            for(int e = 0; e < nEtaBins; ++e){
                 hEta[e] = hPerFrame[e + 2 * bigArrSize]; // Eta is at offset 2*bigArrSize
            }
            for(int r = 0; r < nRBins; ++r){
                hLineout[r * 2] = hR[r]; // R value
                hLineout[r * 2 + 1] = 0.0; // Initialize intensity
            }
            printf("Initialized host R/Eta arrays from first frame D->H copy.\n");
            firstFrame = 0;
        }

        // Write 2D Integrated Data
        double t_write2d_start = get_wall_time_ms();
        if (wr2D && f2D && hIntArrFrame) {
            written = fwrite(hIntArrFrame, sizeof(double), bigArrSize, f2D);
            if (written != bigArrSize) {
                fprintf(stderr, "Warn: Failed write 2D frame %d (wrote %zu/%zu)\n", currFidx, written, bigArrSize);
            }
        }
        t_write2d_cpu = get_wall_time_ms() - t_write2d_start;

        // Prepare 1D Lineout Data
        maxInt = -1.0;
        maxIntLoc = -1;
        if(!multiP){ // Find single peak if not in multi-peak mode
            for(int r = 0; r < nRBins; ++r){
                if(h_int1D[r] > maxInt){
                    maxInt = h_int1D[r];
                    maxIntLoc = r;
                }
            }
        }
        for(int r = 0; r < nRBins; ++r){
            hLineout[r * 2 + 1] = h_int1D[r]; // Intensity value
        }

        // Write 1D Lineout Data
        double t_write1d_start = get_wall_time_ms();
        if (fLineout) {
             written = fwrite(hLineout, sizeof(double), nRBins * 2, fLineout);
             if (written != (size_t)nRBins * 2) {
                  fprintf(stderr, "Warn: Failed write lineout frame %d (wrote %zu/%d)\n", currFidx, written, nRBins * 2);
             } else {
                  fflush(fLineout);
             }
        }
        t_write1d_cpu = get_wall_time_ms() - t_write1d_start;

        // --- Peak Finding and Fitting ---
        double t_fit_start = get_wall_time_ms();
        currentPeakCount = 0;
        sendFitParams = NULL;
        pks = NULL;

        if(pkFit){
            // --- Step 1: Identify Peak Candidates ---
            if(multiP){ // Multi-peak mode
                if(nSpecP > 0){ // Specific peak locations provided
                    pks = (Peak*)malloc(nSpecP * sizeof(Peak));
                    check(!pks, "pkFit: Malloc failed for specified peaks array");
                    validPeakCount = 0;
                    for(int p = 0; p < nSpecP; ++p){
                        bestBin = -1;
                        minDiff = 1e10;
                        for(int r = 0; r < nRBins; ++r){
                            double diff = fabs(hR[r] - pkLoc[p]);
                            if(diff < minDiff){
                                minDiff = diff;
                                bestBin = r;
                            }
                        }
                        if(bestBin != -1 && minDiff < RBinSize * 2.0){
                            pks[validPeakCount].index = bestBin;
                            pks[validPeakCount].radius = hR[bestBin];
                            pks[validPeakCount].intensity = h_int1D[bestBin];
                            validPeakCount++;
                        } else {
                            printf("Warn: Specified peak R=%.4f ignored Frame %d (too far from bins).\n", pkLoc[p], currFidx);
                        }
                    }
                    currentPeakCount = validPeakCount;
                    if(validPeakCount == 0){
                        free(pks);
                        pks = NULL;
                    } else if (validPeakCount < nSpecP) {
                        Peak * reallocPks = (Peak*)realloc(pks, validPeakCount * sizeof(Peak));
                        if(reallocPks) { pks = reallocPks; } // If realloc fails, pks is still valid (but larger)
                    }
                } else { // Find peaks automatically
                    dataToFindPeaks = h_int1D;
                    smoothedData = NULL;
                    if(doSm){
                        smoothedData = (double*)malloc(nRBins * sizeof(double));
                        check(!smoothedData, "pkFit: Malloc failed for smoothedData buffer");
                        smoothData(h_int1D, smoothedData, nRBins, 7); // Window size 7
                        dataToFindPeaks = smoothedData;
                    }
                    // Find peaks using the (potentially smoothed) data
                    currentPeakCount = findPeaks(dataToFindPeaks, hR, nRBins, &pks, 0.0, 5);
                    if(smoothedData) {
                        free(smoothedData);
                        smoothedData = NULL; // Reset pointer
                    }
                }
            } else { // Single peak mode
                if(maxIntLoc != -1){
                    currentPeakCount = 1;
                    pks = (Peak*)malloc(sizeof(Peak));
                    check(!pks, "pkFit: Malloc failed for single peak");
                    pks[0].index = maxIntLoc;
                    pks[0].radius = hR[maxIntLoc];
                    pks[0].intensity = maxInt;
                } else {
                    currentPeakCount = 0;
                    pks = NULL;
                }
            }

            // --- Step 2: Perform Fit if Peaks were Found ---
            if (currentPeakCount > 0 && pks != NULL) {
                nFitParams = currentPeakCount * 4 + 1;
                fitParams = (double*)malloc(nFitParams * sizeof(double)); check(!fitParams, "pkFit: Malloc fitParams");
                lowerBounds = (double*)malloc(nFitParams * sizeof(double)); check(!lowerBounds, "pkFit: Malloc lowerBounds");
                upperBounds = (double*)malloc(nFitParams * sizeof(double)); check(!upperBounds, "pkFit: Malloc upperBounds");

                maxOverallIntensity = 0.0;
                for(int r = 0; r < nRBins; ++r) {
                    if(h_int1D[r] > maxOverallIntensity) { maxOverallIntensity = h_int1D[r]; }
                }
                if(maxOverallIntensity <= 0) { maxOverallIntensity = 1.0; }

                for(int p = 0; p < currentPeakCount; ++p){
                    int base = p * 4;
                    double initialCenter = pks[p].radius;
                    double initialIntensity = pks[p].intensity;
                    double initialSigma = RBinSize * 2.0;

                    fitParams[base + 0] = initialIntensity; // Amplitude
                    fitParams[base + 1] = 0.5;             // Mix
                    fitParams[base + 2] = initialCenter;   // Center
                    fitParams[base + 3] = initialSigma;    // Sigma

                    lowerBounds[base + 0] = 0.0;
                    lowerBounds[base + 1] = 0.0;
                    lowerBounds[base + 2] = initialCenter - RBinSize * 5.0;
                    lowerBounds[base + 3] = RBinSize * 0.5;

                    upperBounds[base + 0] = maxOverallIntensity * 2.0;
                    upperBounds[base + 1] = 1.0;
                    upperBounds[base + 2] = initialCenter + RBinSize * 5.0;
                    upperBounds[base + 3] = (RMax - RMin) / 4.0;
                }
                fitParams[nFitParams - 1] = 0.0; // Initial BG guess
                lowerBounds[nFitParams - 1] = -maxOverallIntensity;
                upperBounds[nFitParams - 1] = maxOverallIntensity;

                fitData.nrBins = nRBins;
                fitData.R = hR;
                fitData.Int = h_int1D;

                opt = nlopt_create(NLOPT_LN_NELDERMEAD, nFitParams);
                nlopt_set_lower_bounds(opt, lowerBounds);
                nlopt_set_upper_bounds(opt, upperBounds);
                nlopt_set_min_objective(opt, problem_function_global_bg, &fitData);
                nlopt_set_xtol_rel(opt, 1e-4);
                nlopt_set_maxeval(opt, 500 * nFitParams);

                nlopt_rc = nlopt_optimize(opt, fitParams, &minObjectiveValue);

                if(nlopt_rc < 0){
                    printf("F#%d: NLopt optimization failed with error code %d\n", currFidx, nlopt_rc);
                    currentPeakCount = 0; // Indicate no successful fit
                    // fitParams will be freed below
                } else {
                    sendFitParams = (double*)malloc(currentPeakCount * 5 * sizeof(double));
                    check(!sendFitParams, "pkFit: Malloc failed for sendFitParams buffer");
                    globalBG = fitParams[nFitParams - 1];

                    for(int p = 0; p < currentPeakCount; ++p){
                         sendFitParams[p * 5 + 0] = fitParams[p * 4 + 0]; // Amp
                         sendFitParams[p * 5 + 1] = globalBG;             // BG
                         sendFitParams[p * 5 + 2] = fitParams[p * 4 + 1]; // Mix
                         sendFitParams[p * 5 + 3] = fitParams[p * 4 + 2]; // Cen
                         sendFitParams[p * 5 + 4] = fitParams[p * 4 + 3]; // Sig
                    }
                }
                nlopt_destroy(opt);
                free(fitParams); fitParams = NULL; // Free intermediate buffer
                free(lowerBounds); lowerBounds = NULL;
                free(upperBounds); upperBounds = NULL;
            } // End if (currentPeakCount > 0)

            if(pks) {
                free(pks); // Free the peak candidate array
                pks = NULL;
            }
        } // End if(pkFit)
        t_fit_cpu = get_wall_time_ms() - t_fit_start;

        // --- Write Peak Fit Results ---
        double t_writefit_start = get_wall_time_ms();
        if(pkFit && currentPeakCount > 0 && sendFitParams != NULL && fFit){
            written = fwrite(sendFitParams, sizeof(double), currentPeakCount * 5, fFit);
            if (written != (size_t)currentPeakCount * 5) {
                 fprintf(stderr, "Warn: Failed write fit frame %d (wrote %zu/%d)\n", currFidx, written, currentPeakCount * 5);
            } else {
                fflush(fFit);
            }
        }
        if (sendFitParams != NULL) {
             free(sendFitParams);
             sendFitParams = NULL;
        }
        t_writefit_cpu = get_wall_time_ms() - t_writefit_start;

        // --- Free received data buffer ---
        gpuWarnchk(cudaFreeHost(chunk.data));
        chunk.data = NULL; // Avoid double free risk

        // --- Timing and Output ---
        t_end_loop = get_wall_time_ms();
        t_loop_cpu = t_end_loop - t_start_loop;
        printf("F#%d: Ttl:%.2f| QPop:%.2f GPU(Proc:%.2f Int:%.2f Prof:%.2f D2H:%.2f) CPU(Wr2D:%.2f Wr1D:%.2f Fit:%.2f WrFit:%.2f)\n",
               currFidx, t_loop_cpu, t_qp_cpu,
               t_proc_gpu, t_integ_gpu, t_prof_gpu, t_d2h_gpu,
               t_write2d_cpu, t_write1d_cpu, t_fit_cpu, t_writefit_cpu);
        fflush(stdout);

        frameCounter++;
    } // ======================== End Main Processing Loop ========================

    printf("Processing loop finished (keep_running=%d). Processed %d frames. Cleaning up...\n", keep_running, frameCounter);

    // --- Cleanup ---

    if(fLineout) { fclose(fLineout); }
    if(fFit) { fclose(fFit); }
    if(f2D) { fclose(f2D); }

    if(hAvgDark) { free(hAvgDark); }
    if(hDarkInT) { free(hDarkInT); }
    if(hDarkIn) { free(hDarkIn); }
    if(hIntArrFrame) { gpuWarnchk(cudaFreeHost(hIntArrFrame)); }
    if(hPerFrame) { gpuWarnchk(cudaFreeHost(hPerFrame)); }
    if(h_int1D) { gpuWarnchk(cudaFreeHost(h_int1D)); }
    if(hR) { free(hR); }
    if(hEta) { free(hEta); }
    if(hLineout) { free(hLineout); }
    if(hEtaLo) { free(hEtaLo); }
    if(hEtaHi) { free(hEtaHi); }
    if(hRLo) { free(hRLo); }
    if(hRHi) { free(hRHi); }
    if(hMapMask) { free(hMapMask); }

    FreeCSRMaps(); // Free host CSR buffers

    // Free GPU memory
    if(dAvgDark) { gpuWarnchk(cudaFree(dAvgDark)); }
    if(dProcessedImage) { gpuWarnchk(cudaFree(dProcessedImage)); }
    if(d_int1D) { gpuWarnchk(cudaFree(d_int1D)); }
    if(dMapMask) { gpuWarnchk(cudaFree(dMapMask)); }
    if(d_csr_values) { gpuWarnchk(cudaFree(d_csr_values)); } // Free CSR GPU buffers
    if(d_csr_col_indices) { gpuWarnchk(cudaFree(d_csr_col_indices)); }
    if(d_csr_row_ptr) { gpuWarnchk(cudaFree(d_csr_row_ptr)); }
    if(dSumMatrix) { gpuWarnchk(cudaFree(dSumMatrix)); }
    if(dIntArrFrame) { gpuWarnchk(cudaFree(dIntArrFrame)); }
    if(dPerFrame) { gpuWarnchk(cudaFree(dPerFrame)); }
    if(dEtaLo) { gpuWarnchk(cudaFree(dEtaLo)); }
    if(dEtaHi) { gpuWarnchk(cudaFree(dEtaHi)); }
    if(dRLo) { gpuWarnchk(cudaFree(dRLo)); }
    if(dRHi) { gpuWarnchk(cudaFree(dRHi)); }
    if(g_dTempTransformBuf1) { gpuWarnchk(cudaFree(g_dTempTransformBuf1)); }
    if(g_dTempTransformBuf2) { gpuWarnchk(cudaFree(g_dTempTransformBuf2)); }

    // Destroy CUDA events
    gpuWarnchk(cudaEventDestroy(ev_proc_start)); gpuWarnchk(cudaEventDestroy(ev_proc_stop));
    gpuWarnchk(cudaEventDestroy(ev_integ_start)); gpuWarnchk(cudaEventDestroy(ev_integ_stop));
    gpuWarnchk(cudaEventDestroy(ev_prof_start)); gpuWarnchk(cudaEventDestroy(ev_prof_stop));
    gpuWarnchk(cudaEventDestroy(ev_d2h_start)); gpuWarnchk(cudaEventDestroy(ev_d2h_stop));

    // --- Shutdown Accept Thread Gracefully ---
    printf("Attempting to shut down network acceptor thread...\n");
    if (server_fd >= 0) {
         printf("Closing server listening socket %d...\n", server_fd);
         shutdown(server_fd, SHUT_RDWR);
         close(server_fd);
         server_fd = -1;
    }

    printf("Sending cancellation request to accept thread...\n");
    cancel_ret = pthread_cancel(accept_thread);
    if (cancel_ret != 0) {
        fprintf(stderr, "Warning: Failed to send cancel request to accept thread: %s\n", strerror(cancel_ret));
    }

    printf("Joining accept thread (waiting for it to exit)...\n");
    join_ret = pthread_join(accept_thread, &thread_result);
    if (join_ret != 0) {
         fprintf(stderr, "Warning: Failed to join accept thread: %s\n", strerror(join_ret));
    } else {
        if (thread_result == PTHREAD_CANCELED) {
            printf("Accept thread successfully canceled and joined.\n");
        } else {
            printf("Accept thread joined normally (result: %p).\n", thread_result);
        }
    }

    // Destroy queue (frees remaining pinned data buffers)
    queue_destroy(&process_queue);

    printf("[%s] - Exiting cleanly.\n", argv[0]);
    return 0;
}