// =========================================================================
// IntegratorFitPeaksGPUStream_c.cu
//
// Copyright (c) 2014, UChicago Argonne, LLC
// Uses CSR map format.
// =========================================================================
/*
~/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStreamCSR.cu -o bin/IntegratorFitPeaksGPUStreamCSR \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_90,code=sm_90 \
  -Xcompiler -g \
  -I/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/include \
  -L/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/lib \
  -O3 -lnlopt -lz -ldl -lm -lpthread

*/


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
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/types.h>
#include <libgen.h>
#include <assert.h>
#include <signal.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>    // For bool type
#include <blosc2.h>
#include <nlopt.h>

// --- Constants ---
#define SERVER_IP "127.0.0.1"
#define PORT 60439
#define MAX_CONNECTIONS 10
#define MAX_QUEUE_SIZE 100
#define HEADER_SIZE sizeof(uint16_t)
#define BYTES_PER_PIXEL 8 // Input data is int64_t
#define MAX_FILENAME_LENGTH 1024
#define THREADS_PER_BLOCK_TRANSFORM 512
#define THREADS_PER_BLOCK_INTEGRATE 512 // Kernel iterates over pixels now
#define THREADS_PER_BLOCK_NORMALIZE 512
#define THREADS_PER_BLOCK_PROFILE 512
#define MAX_TRANSFORM_OPS 10
#define MAX_PEAK_LOCATIONS 100
#define AREA_THRESHOLD_F 1e-7f      // Minimum area (float) for normalization/profile
#define EPS_F 1e-5f                 // Epsilon for float compares
#define rad2degf 57.2957795130823f
#define rad2deg 57.2957795130823 // Double for fitting

// --- Forward Declarations (C style) ---
static inline void DoImageTransformationsSequential(int Nopt, const int Topt[MAX_TRANSFORM_OPS], const int64_t *In, int64_t *Out, int NY, int NZ);
static inline void REtaMapperF(float Rmin, float EtaMin, int nEta, int nR, float EtaStep, float RStep, float *EtaLo, float *EtaHi, float *RLo, float *RHi);
void smoothData(const double *in, double *out, int N, int W);
int findPeaks(const double *data, const double *r_values, int N, Peak **foundPeaks, double minHeight, int minDistance);
static double problem_function_global_bg(unsigned n, const double *x, double *grad, void *fdat);

// Global variables
size_t CHUNK_SIZE;
size_t TOTAL_MSG_SIZE;
volatile sig_atomic_t keep_running = 1;

// --- Data Structures ---
typedef struct { uint16_t dataset_num; int64_t *data; size_t size; } DataChunk;
typedef struct { DataChunk chunks[MAX_QUEUE_SIZE]; int front; int rear; int count; pthread_mutex_t mutex; pthread_cond_t not_empty; pthread_cond_t not_full; } ProcessQueue;
typedef struct { int nrBins; const double *R; const double *Int; } dataFit; // Peak fitting
typedef struct { int index; double radius; double intensity; } Peak;    // Peak fitting

// --- CSR Map Data Globals ---
int g_NrPixelsY = 0;
int g_NrPixelsZ = 0;
int g_nRBins = 0;
int g_nEtaBins = 0;
long long g_TotalContributions = 0;
size_t g_total_pixels = 0;
size_t g_bigArrSize = 0;
long long *d_csr_row_offsets = NULL;
int *d_csr_col_indices = NULL;
float *d_csr_values = NULL;
float *d_total_area_per_bin = NULL;

// --- Global Variables ---
ProcessQueue process_queue;

// --- CUDA Error Handling ---
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
#define gpuWarnchk(ans) { gpuAssert((ans), __FILE__, __LINE__, false); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abortflag)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abortflag)
        {
            exit(code);
        }
    }
}

// --- General Error Handling ---
static void check (int test, const char * message, ...)
{
    if (test)
    {
        va_list args;
        va_start (args, message);
        fprintf(stderr, "Fatal Error: ");
        vfprintf(stderr, message, args);
        va_end (args);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}

// --- Signal Handler ---
void sigint_handler(int signum)
{
    if (keep_running)
    {
        printf("\nCaught signal %d, requesting shutdown...\n", signum);
        keep_running = 0;
    }
}

// --- Timing Function ---
static inline double get_wall_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// --- Queue Functions ---
void queue_init(ProcessQueue *queue)
{
    queue->front = 0;
    queue->rear = -1;
    queue->count = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
}

int queue_push(ProcessQueue *queue, uint16_t dataset_num, int64_t *data, size_t num_values)
{
    pthread_mutex_lock(&queue->mutex);
    while (queue->count >= MAX_QUEUE_SIZE && keep_running)
    {
        printf("Queue full, waiting...\n");
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        pthread_cond_timedwait(&queue->not_full, &queue->mutex, &ts);
    }
    if (!keep_running)
    {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    queue->rear = (queue->rear + 1) % MAX_QUEUE_SIZE;
    queue->chunks[queue->rear].dataset_num = dataset_num;
    queue->chunks[queue->rear].data = data;
    queue->chunks[queue->rear].size = num_values;
    queue->count++;
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

int queue_pop(ProcessQueue *queue, DataChunk *chunk)
{
    pthread_mutex_lock(&queue->mutex);
    while (queue->count <= 0 && keep_running)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &ts);
    }
    if (!keep_running && queue->count <= 0)
    {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    *chunk = queue->chunks[queue->front];
    queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
    queue->count--;
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

void queue_destroy(ProcessQueue *queue)
{
     pthread_mutex_destroy(&queue->mutex);
     pthread_cond_destroy(&queue->not_empty);
     pthread_cond_destroy(&queue->not_full);
     printf("Cleaning up remaining %d queue entries...\n", queue->count);
     while (queue->count > 0)
     {
        DataChunk chunk;
        chunk.data = queue->chunks[queue->front].data;
        queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
        queue->count--;
        if (chunk.data)
        {
            gpuWarnchk(cudaFreeHost(chunk.data)); // Free pinned memory
        }
     }
}

// --- Socket Handling ---
void* handle_client(void *arg)
{
    int client_socket = *((int*)arg);
    free(arg);
    uint8_t *buffer = (uint8_t*)malloc(TOTAL_MSG_SIZE);
    check(buffer == NULL, "handle_client: Failed alloc recv buffer");
    int bytes_read;
    printf("Client handler started for socket %d.\n", client_socket);

    while (keep_running)
    {
        int total_bytes_read = 0;
        while (total_bytes_read < TOTAL_MSG_SIZE)
        {
            bytes_read = recv(client_socket, buffer + total_bytes_read, TOTAL_MSG_SIZE - total_bytes_read, 0);
            if (bytes_read <= 0)
            {
                 goto connection_closed;
            }
            total_bytes_read += bytes_read;
            if (!keep_running)
            {
                goto connection_closed;
            }
        }
        uint16_t dataset_num;
        memcpy(&dataset_num, buffer, HEADER_SIZE);
        size_t num_pixels = CHUNK_SIZE / BYTES_PER_PIXEL;
        int64_t *data = NULL;
        gpuWarnchk(cudaMallocHost((void**)&data, num_pixels * sizeof(int64_t))); // Pinned memory
        if (!data)
        {
            perror("handle_client: Pinned memory alloc failed");
            break;
        }
        memcpy(data, buffer + HEADER_SIZE, CHUNK_SIZE);
        if (queue_push(&process_queue, dataset_num, data, num_pixels) < 0)
        {
            printf("handle_client: queue_push failed, likely shutdown. Discarding frame %d.\n", dataset_num);
            gpuWarnchk(cudaFreeHost(data));
            goto connection_closed;
        }
    }
connection_closed:
    if (bytes_read == 0 && keep_running)
    {
        printf("Client disconnected (socket %d)\n", client_socket);
    }
    else if (bytes_read < 0)
    {
        perror("Receive error");
    }
    free(buffer);
    close(client_socket);
    printf("Client handler finished (socket %d).\n", client_socket);
    return NULL;
}

void* accept_connections(void *server_fd_ptr)
{
    int server_fd = *((int*)server_fd_ptr);
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    printf("Accept thread started, listening for connections.\n");
    while (keep_running)
    {
        int *client_socket_ptr = (int*) malloc(sizeof(int));
        check(client_socket_ptr == NULL, "accept_connections: Failed alloc client socket ptr");
        *client_socket_ptr = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (!keep_running)
        {
             if (*client_socket_ptr >= 0)
             {
                 close(*client_socket_ptr);
             }
             free(client_socket_ptr);
             break;
        }
        if (*client_socket_ptr < 0)
        {
            if (errno == EINTR)
            {
                continue; // Interrupted by signal, check keep_running and loop again
            }
            perror("Accept failed");
            free(client_socket_ptr);
            sleep(1); // Avoid busy-waiting on persistent errors
            continue;
        }
        printf("Connection accepted from %s:%d (socket %d)\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), *client_socket_ptr);
        pthread_t thread_id;
        int create_rc = pthread_create(&thread_id, NULL, handle_client, (void*)client_socket_ptr);
        if (create_rc != 0)
        {
            fprintf(stderr, "Thread creation failed: %s\n", strerror(create_rc));
            close(*client_socket_ptr);
            free(client_socket_ptr);
        }
        else
        {
            pthread_detach(thread_id); // Don't need to join client threads
        }
    }
    printf("Accept thread exiting.\n");
    return NULL;
}

// --- Bit Manipulation Macros ---
#define SetBit(A,k)   (A[((k)/32)] |=  (1U << ((k)%32)))
#define TestBit(A,k)  (A[((k)/32)] &   (1U << ((k)%32)))

// --- File I/O & Mapping (CSR Reader - C Version) ---
bool ReadBinaryFileC(const char* filename, void** data_ptr, size_t* num_bytes_ptr, size_t element_size)
{
    FILE* f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error opening file: %s (%s)\n", filename, strerror(errno));
        return false;
    }
    fseek(f, 0, SEEK_END);
    long file_size_bytes = ftell(f);
    rewind(f);
    if (file_size_bytes <= 0 || file_size_bytes % element_size != 0)
    {
        fprintf(stderr, "Error: Invalid file size %ld for %s (element size %zu)\n", file_size_bytes, filename, element_size);
        fclose(f);
        return false;
    }
    size_t num_elements = file_size_bytes / element_size;
    void* buffer = malloc(file_size_bytes);
    if (!buffer)
    {
        fprintf(stderr, "Error: malloc failed for %s\n", filename);
        fclose(f);
        return false;
    }
    size_t elements_read = fread(buffer, element_size, num_elements, f);
    fclose(f);
    if (elements_read != num_elements)
    {
        fprintf(stderr, "Error reading data from %s (read %zu / %zu elements)\n", filename, elements_read, num_elements);
        free(buffer);
        return false;
    }
    *data_ptr = buffer;
    *num_bytes_ptr = file_size_bytes;
    printf("Read %zu bytes (%zu elements) from %s.\n", file_size_bytes, num_elements, filename);
    return true;
}

bool ReadCSRMapC()
{
    const char *metaFN = "Map_CSR_Meta.txt";
    const char *offsFN = "Map_CSR_Offsets.bin";
    const char *indsFN = "Map_CSR_Indices.bin";
    const char *valsFN = "Map_CSR_Values.bin";
    const char *areaFN = "Map_CSR_AreaPerBin.bin";

    printf("Reading CSR Map Metadata: %s\n", metaFN);
    FILE *metafile = fopen(metaFN, "r");
    if (!metafile)
    {
        fprintf(stderr, "Error opening CSR meta file: %s (%s)\n", metaFN, strerror(errno));
        return false;
    }

    char line[256];
    char key[100];
    char value_str[150];
    while (fgets(line, sizeof(line), metafile))
    {
        // Basic parsing, skip comments/empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (sscanf(line, "%99s %149s", key, value_str) == 2)
        {
            errno = 0; // Reset errno before conversion
            if (strcmp(key, "NrPixelsY") == 0) g_NrPixelsY = atoi(value_str);
            else if (strcmp(key, "NrPixelsZ") == 0) g_NrPixelsZ = atoi(value_str);
            else if (strcmp(key, "nRBins") == 0) g_nRBins = atoi(value_str);
            else if (strcmp(key, "nEtaBins") == 0) g_nEtaBins = atoi(value_str);
            else if (strcmp(key, "TotalContributions") == 0) g_TotalContributions = strtoll(value_str, NULL, 10);
            // Add other metadata if needed
            if (errno != 0)
            {
                fprintf(stderr, "Error converting metadata value for key '%s': %s\n", key, strerror(errno));
            }
        }
        else
        {
            fprintf(stderr, "Warning: Could not parse metadata line: %s", line);
        }
    }
    fclose(metafile);

    // Validate essential metadata
    if (g_NrPixelsY <= 0 || g_NrPixelsZ <= 0 || g_nRBins <= 0 || g_nEtaBins <= 0 || g_TotalContributions < 0)
    {
        fprintf(stderr, "Error: Invalid metadata read from %s.\n", metaFN);
        return false;
    }
    g_total_pixels = (size_t)g_NrPixelsY * g_NrPixelsZ;
    g_bigArrSize = (size_t)g_nRBins * g_nEtaBins;
    printf(" CSR Map Metadata: %dx%d detector, %dx%d bins, %lld contributions.\n", g_NrPixelsY, g_NrPixelsZ, g_nRBins, g_nEtaBins, g_TotalContributions);

    // Read Binary Data into Host Buffers
    printf("Reading CSR Map Binary Files...\n");
    long long* h_csr_row_offsets_c = NULL;
    size_t offsets_bytes = 0;
    int*       h_csr_col_indices_c = NULL;
    size_t indices_bytes = 0;
    float*     h_csr_values_c      = NULL;
    size_t values_bytes = 0;
    float*     h_total_area_per_bin_c = NULL;
    size_t areas_bytes = 0;
    bool ok = true;

    ok &= ReadBinaryFileC(offsFN, (void**)&h_csr_row_offsets_c, &offsets_bytes, sizeof(long long));
    ok &= ReadBinaryFileC(indsFN, (void**)&h_csr_col_indices_c, &indices_bytes, sizeof(int));
    ok &= ReadBinaryFileC(valsFN, (void**)&h_csr_values_c, &values_bytes, sizeof(float));
    ok &= ReadBinaryFileC(areaFN, (void**)&h_total_area_per_bin_c, &areas_bytes, sizeof(float));

    if (!ok)
    {
        fprintf(stderr, "Error reading one or more CSR binary map files.\n");
        // Free any buffers that were successfully allocated before returning
        free(h_csr_row_offsets_c);
        free(h_csr_col_indices_c);
        free(h_csr_values_c);
        free(h_total_area_per_bin_c);
        return false;
    }

    // Validate sizes against metadata
    if (offsets_bytes != (g_total_pixels + 1) * sizeof(long long)) { fprintf(stderr, "CSR offsets size mismatch\n"); ok = false; }
    if (indices_bytes != (size_t)g_TotalContributions * sizeof(int)) { fprintf(stderr, "CSR indices size mismatch\n"); ok = false; }
    if (values_bytes != (size_t)g_TotalContributions * sizeof(float)) { fprintf(stderr, "CSR values size mismatch\n"); ok = false; }
    if (areas_bytes != g_bigArrSize * sizeof(float)) { fprintf(stderr, "CSR areas size mismatch\n"); ok = false; }

    if (!ok)
    {
        free(h_csr_row_offsets_c);
        free(h_csr_col_indices_c);
        free(h_csr_values_c);
        free(h_total_area_per_bin_c);
        return false;
    }

    // Allocate GPU Memory and Copy
    printf("Allocating and copying CSR map data to GPU...\n");
    gpuErrchk(cudaMalloc(&d_csr_row_offsets, offsets_bytes));
    gpuErrchk(cudaMalloc(&d_csr_col_indices, indices_bytes));
    gpuErrchk(cudaMalloc(&d_csr_values, values_bytes));
    gpuErrchk(cudaMalloc(&d_total_area_per_bin, areas_bytes));

    gpuErrchk(cudaMemcpy(d_csr_row_offsets, h_csr_row_offsets_c, offsets_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_csr_col_indices, h_csr_col_indices_c, indices_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_csr_values, h_csr_values_c, values_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_total_area_per_bin, h_total_area_per_bin_c, areas_bytes, cudaMemcpyHostToDevice));

    // Free host buffers now that data is on GPU
    free(h_csr_row_offsets_c);
    free(h_csr_col_indices_c);
    free(h_csr_values_c);
    free(h_total_area_per_bin_c);

    printf("Successfully read and copied CSR map files to GPU.\n");
    fflush(stdout);
    return true;
}


// --- Binning Setup (float, C version) ---
static inline void REtaMapperF(float Rmin, float EtaMin, int nEta, int nR, float EtaStep, float RStep, float *EtaLo, float *EtaHi, float *RLo, float *RHi)
{
    int i;
    for(i = 0; i < nEta; ++i)
    {
        EtaLo[i] = EtaStep * i + EtaMin;
        EtaHi[i] = EtaStep * (i + 1) + EtaMin;
    }
    for(i = 0; i < nR; ++i)
    {
        RLo[i] = RStep * i + Rmin;
        RHi[i] = RStep * (i + 1) + Rmin;
    }
}

// --- Sequential CPU Image Transformation ---
static inline void DoImageTransformationsSequential(int Nopt, const int Topt[MAX_TRANSFORM_OPS], const int64_t *In, int64_t *Out, int NY, int NZ)
{
    size_t N = (size_t)NY * NZ;
    size_t B = N * sizeof(int64_t);
    int any = 0, i, k, l;
    if(Nopt > 0)
    {
        for(i = 0; i < Nopt; ++i)
        {
            if(Topt[i] < 0 || Topt[i] > 3)
            {
                fprintf(stderr, "CPU Err: Inv opt %d\n", Topt[i]);
                return;
            }
            if(Topt[i] != 0) any = 1;
        }
    }
    if(!any)
    {
        if(Out != In) memcpy(Out, In, B);
        return;
    }
    int64_t *tmp = (int64_t*)malloc(B);
    if(!tmp)
    {
        fprintf(stderr, "CPU Err: Alloc tmp fail\n");
        if(Out != In) memcpy(Out, In, B);
        return;
    }
    const int64_t* rB = NULL;
    int64_t* wB = NULL;
    int cY = NY;
    int cZ = NZ;
    for(i = 0; i < Nopt; ++i)
    {
        int opt = Topt[i];
        size_t cB = (size_t)cY * cZ * sizeof(int64_t);
        if(i == 0){ rB = In; wB = tmp; }
        else if(i % 2 == 1){ rB = tmp; wB = Out; }
        else { rB = Out; wB = tmp; }
        int nY = cY;
        int nZ = cZ;
        switch(opt)
        {
            case 0: if(wB != rB) memcpy(wB, rB, cB); break;
            case 1: for(l=0;l<cZ;++l){for(k=0;k<cY;++k){wB[l*cY+k]=rB[l*cY+(cY-1-k)];}} break;
            case 2: for(l=0;l<cZ;++l){for(k=0;k<cY;++k){wB[l*cY+k]=rB[(cZ-1-l)*cY+k];}} break;
            case 3: if(cY != cZ){ fprintf(stderr,"CPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i); if(wB != rB) memcpy(wB, rB, cB); } else { nY = cZ; nZ = cY; for(l=0;l<nZ;++l){for(k=0;k<nY;++k){wB[l*nY+k]=rB[k*cY+l];}}} break;
        }
        cY = nY;
        cZ = nZ;
    }
    if(Nopt % 2 != 0)
    {
        size_t fB = (size_t)cY * cZ * sizeof(int64_t);
        if(fB > B){ fprintf(stderr,"CPU Err: Final buf size > Orig\n"); fB=B; }
        memcpy(Out, tmp, fB);
    }
    free(tmp);
}


// --- GPU Kernels ---
__global__ void initialize_RTE_kernel( float *dRTE, size_t bigArrSize, int nRBins, int nEtaBins, const float *dRBinsLow, const float *dRBinsHigh, const float *dEtaBinsLow, const float *dEtaBinsHigh, float px, float Lsd)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bigArrSize) return;
    float RMean = 0.0f;
    float EtaMean = 0.0f;
    float TwoTheta = 0.0f;
    if (nEtaBins > 0)
    {
        int j = idx / nEtaBins;
        int k = idx % nEtaBins;
        if (j < nRBins && k < nEtaBins)
        {
            RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5f;
            EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5f;
            TwoTheta = (Lsd > EPS_F) ? (rad2degf * atanf(RMean * px / Lsd)) : 0.0f;
        }
    }
    dRTE[0 * bigArrSize + idx] = RMean;
    dRTE[1 * bigArrSize + idx] = TwoTheta;
    dRTE[2 * bigArrSize + idx] = EtaMean;
}

__global__ void integrate_kernel_csr( const float *dProcessedImage, const long long *d_csr_row_offsets, const int *d_csr_col_indices, const float *d_csr_values, float *dIntArrFrame, size_t total_pixels, const int *dMapMask, size_t mapMaskWordCount)
{
    const size_t pix_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pix_idx >= total_pixels) return;
    if (dMapMask != NULL)
    {
        if (TestBit(dMapMask, pix_idx))
        {
            return;
        }
    }
    float image_value = dProcessedImage[pix_idx];
    long long start_offset = d_csr_row_offsets[pix_idx];
    long long end_offset = d_csr_row_offsets[pix_idx + 1];
    long long k;
    for (k = start_offset; k < end_offset; ++k)
    {
        int bin_idx = d_csr_col_indices[k];
        float frac = d_csr_values[k];
        atomicAdd(&dIntArrFrame[bin_idx], image_value * frac);
    }
}

__global__ void normalize_kernel( float *dIntArrFrame, const float *d_total_area_per_bin, size_t bigArrSize)
{
    const size_t bin_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= bigArrSize) return;
    float intensity_sum = dIntArrFrame[bin_idx];
    float total_area = d_total_area_per_bin[bin_idx];
    if (total_area > AREA_THRESHOLD_F)
    {
        dIntArrFrame[bin_idx] = intensity_sum / total_area;
    }
    else
    {
        dIntArrFrame[bin_idx] = 0.0f;
    }
}

__global__ void sequential_transform_kernel(const int64_t *r, int64_t *w, int cY, int cZ, int nY, int nZ, int opt)
{
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1;
    int zs = -1;
    switch(opt){ case 0: ys=yo; zs=zo; break; case 1: ys=cY-1-yo; zs=zo; break; case 2: ys=yo; zs=cZ-1-zo; break; case 3: ys=zo; zs=yo; break; default: return; }
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ)
    {
        w[i] = r[(size_t)zs * cY + ys];
    }
    else
    {
        w[i] = 0;
    }
}

__global__ void final_transform_process_kernel(const int64_t *r, float *o, const float *d, int cY, int cZ, int nY, int nZ, int opt, bool sub)
{
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1;
    int zs = -1;
    switch(opt){ case 0: ys=yo; zs=zo; break; case 1: ys=cY-1-yo; zs=zo; break; case 2: ys=yo; zs=cZ-1-zo; break; case 3: ys=zo; zs=yo; break; default: o[i] = 0.0f; return; }
    float pv = 0.0f;
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ)
    {
        const int64_t rv = r[(size_t)zs * cY + ys];
        pv = (float)rv;
        if (sub && d)
        {
            pv -= d[i];
        }
    }
    o[i] = pv;
}

__global__ void process_direct_kernel(const int64_t *r, float *o, const float *d, size_t N, bool sub)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const int64_t rv = r[i];
        float pv = (float)rv;
        if (sub && d)
        {
            pv -= d[i];
        }
        o[i] = pv;
    }
}

__global__ void calculate_1D_profile_kernel( const float *d_IntArrPerFrame, const float *d_total_area_per_bin, float *d_int1D, int nRBins, int nEtaBins, size_t bigArrSize)
{
    extern __shared__ float sdata[];
    float * sIntArea = sdata;
    float * sArea = &sdata[blockDim.x / 32];
    const int r_bin = blockIdx.x;
    if (r_bin >= nRBins) return;
    const int tid = threadIdx.x;
    const int warpSize = 32;
    const int lane = tid % warpSize;
    const int warpId = tid / warpSize;
    if (lane == 0)
    {
        sIntArea[warpId] = 0.0f;
        sArea[warpId] = 0.0f;
    }
    float mySumIntArea = 0.0f;
    float mySumArea = 0.0f;
    int eta_bin;
    for (eta_bin = tid; eta_bin < nEtaBins; eta_bin += blockDim.x)
    {
        size_t idx2d = (size_t)r_bin * nEtaBins + eta_bin;
        if (idx2d < bigArrSize)
        {
            float area = d_total_area_per_bin[idx2d];
            if (area > AREA_THRESHOLD_F)
            {
                 float intensity = d_IntArrPerFrame[idx2d];
                 mySumIntArea += intensity * area;
                 mySumArea += area;
            }
        }
    }
    int offset;
    #pragma unroll
    for (offset = warpSize / 2; offset > 0; offset /= 2)
    {
        mySumIntArea += __shfl_down_sync(0xFFFFFFFF, mySumIntArea, offset);
        mySumArea += __shfl_down_sync(0xFFFFFFFF, mySumArea, offset);
    }
    if (lane == 0)
    {
        atomicAdd(&sIntArea[warpId], mySumIntArea);
        atomicAdd(&sArea[warpId], mySumArea);
    }
    __syncthreads();
    if (tid == 0)
    {
        float finalSumIntArea = 0.0f;
        float finalSumArea = 0.0f;
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        int i;
        for (i = 0; i < numWarps; ++i)
        {
             finalSumIntArea += sIntArea[i];
             finalSumArea += sArea[i];
        }
        if (finalSumArea > AREA_THRESHOLD_F)
        {
            d_int1D[r_bin] = finalSumIntArea / finalSumArea;
        }
        else
        {
            d_int1D[r_bin] = 0.0f;
        }
    }
}

// --- Host Wrapper for GPU Processing ---
void ProcessImageGPU(const int64_t *hRaw, float *dProc, const float *dAvgDark, int Nopt, const int Topt[MAX_TRANSFORM_OPS], int NY, int NZ, bool doSub, int64_t* d_b1, int64_t* d_b2)
{
    const size_t N = (size_t)NY * NZ;
    const size_t B64 = N * sizeof(int64_t);
    const size_t BFlt = N * sizeof(float);
    const int TPB = THREADS_PER_BLOCK_TRANSFORM;
    int anyT = 0, i;
    if(Nopt > 0)
    {
        for(i = 0; i < Nopt; ++i)
        {
            if(Topt[i] < 0 || Topt[i] > 3)
            {
                fprintf(stderr,"GPU Err: Inv opt %d\n", Topt[i]);
                gpuErrchk(cudaMemset(dProc, 0, BFlt));
                return;
            }
            if(Topt[i] != 0) anyT = 1;
        }
    }
    if(!anyT)
    {
        gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));
        dim3 nB = {(unsigned int)((N + TPB - 1) / TPB)};
        dim3 th = {TPB};
        process_direct_kernel<<<nB, th>>>(d_b1, dProc, dAvgDark, N, doSub);
        gpuErrchk(cudaPeekAtLastError());
        return;
    }
    gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));
    const int64_t* rP = NULL;
    int64_t* wP = NULL;
    int cY = NY;
    int cZ = NZ;
    for(i = 0; i < Nopt - 1; ++i)
    {
        int opt = Topt[i];
        if(i % 2 == 0){ rP = d_b1; wP = d_b2; } else { rP = d_b2; wP = d_b1; }
        int nY = cY;
        int nZ = cZ;
        if(opt == 3){ if(cY != cZ){ fprintf(stderr,"GPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i); opt = 0; } else { nY = cZ; nZ = cY; } }
        size_t sON = (size_t)nY * nZ;
        dim3 nB = {(unsigned int)((sON + TPB - 1) / TPB)};
        dim3 th = {TPB};
        sequential_transform_kernel<<<nB, th>>>(rP, wP, cY, cZ, nY, nZ, opt);
        gpuErrchk(cudaPeekAtLastError());
        cY = nY;
        cZ = nZ;
    }
    int fOpt = Topt[Nopt - 1];
    if((Nopt - 1) % 2 == 0){ rP = d_b1; } else { rP = d_b2; }
    int nY = cY;
    int nZ = cZ;
    if(fOpt == 3){ if(cY != cZ){ fprintf(stderr,"GPU Warn: Skip Tpose %dx%d final st %d\n", cY, cZ, Nopt - 1); fOpt = 0; } else { nY = cZ; nZ = cY; } }
    size_t fON = (size_t)nY * nZ;
    if(fON != N){ fprintf(stderr, "GPU Warn: Final image size %zu != Original size %zu\n", fON, N); }
    dim3 nB = {(unsigned int)((fON + TPB - 1) / TPB)};
    dim3 th = {TPB};
    final_transform_process_kernel<<<nB, th>>>(rP, dProc, dAvgDark, cY, cZ, nY, nZ, fOpt, doSub);
    gpuErrchk(cudaPeekAtLastError());
}

// --- Peak Fitting ---
static double problem_function_global_bg(unsigned n, const double *x, double *grad, void *fdat)
{
    if(grad){ unsigned i; for(i = 0; i < n; ++i){ grad[i] = 0.0; } }
    const dataFit *d = (const dataFit*)fdat;
    const int Np = d->nrBins;
    const double *Rs = d->R;
    const double *Is = d->Int;
    const int nP = (n - 1) / 4;
    if(nP <= 0 || (4 * nP + 1 != n)){ fprintf(stderr, "Obj Func Err: Invalid params %u for %d peaks\n", n, nP); return INFINITY; }
    const double bg_g = x[n - 1];
    double total_sq_error = 0.0;
    double *calculated_I = (double*)calloc(Np, sizeof(double));
    if(!calculated_I){ fprintf(stderr, "Obj Func Err: Alloc fail\n"); return INFINITY; }
    int pN, i;
    for(pN = 0; pN < nP; ++pN)
    {
        double Amplitude = x[pN * 4 + 0];
        double Mix = fmax(0.0, fmin(1.0, x[pN * 4 + 1]));
        double Center = x[pN * 4 + 2];
        double Sigma = fmax(1e-9, x[pN * 4 + 3]);
        for(i = 0; i < Np; ++i)
        {
             double diff = Rs[i] - Center;
             double diff_sq = diff * diff;
             double sigma_sq = Sigma * Sigma;
             double gaussian = exp(-diff_sq / (2.0 * sigma_sq)) / (Sigma * sqrt(2.0 * M_PI));
             double lorentzian = (1.0 / (M_PI * Sigma)) / (1.0 + diff_sq / sigma_sq);
             calculated_I[i] += Amplitude * (Mix * gaussian + (1.0 - Mix) * lorentzian);
        }
    }
    for(i = 0; i < Np; ++i)
    {
        calculated_I[i] += bg_g;
        double error = calculated_I[i] - Is[i];
        total_sq_error += error * error;
    }
    free(calculated_I);
    return total_sq_error;
}

void smoothData(const double *in, double *out, int N, int W)
{
    if(W < 3 || W % 2 == 0){ memcpy(out, in, N * sizeof(double)); return; }
    int H = W / 2;
    int i, j;
    double *coeffs = (double*)malloc(W * sizeof(double));
    check(!coeffs, "smoothData: Malloc coeffs");
    double norm = 0.0;
    switch(W)
    {
        case 5: norm=35.0; coeffs[0]=-3; coeffs[1]=12; coeffs[2]=17; coeffs[3]=12; coeffs[4]=-3; break;
        case 7: norm=21.0; coeffs[0]=-2; coeffs[1]= 3; coeffs[2]= 6; coeffs[3]= 7; coeffs[4]= 6; coeffs[5]= 3; coeffs[6]=-2; break;
        case 9: norm=231.0; coeffs[0]=-21; coeffs[1]=14; coeffs[2]=39; coeffs[3]=54; coeffs[4]=59; coeffs[5]=54; coeffs[6]=39; coeffs[7]=14; coeffs[8]=-21; break;
        default: fprintf(stderr, "smoothData Warn: Unsupp win %d\n", W); memcpy(out, in, N * sizeof(double)); free(coeffs); return;
    }
    for(i = 0; i < W; ++i){ coeffs[i] /= norm; }
    for(i = 0; i < N; ++i)
    {
        if(i < H || i >= N - H){ out[i] = in[i]; }
        else
        {
            double smoothed_value = 0.0;
            for(j = 0; j < W; ++j){ smoothed_value += coeffs[j] * in[i - H + j]; }
            out[i] = smoothed_value;
        }
    }
    free(coeffs);
}

int findPeaks(const double *data, const double *r_values, int N, Peak **foundPeaks, double minHeight, int minDistance)
{
    if(N < 3){ *foundPeaks = NULL; return 0; }
    int maxPossiblePeaks = N / 2 + 1;
    int i, j;
    Peak* preliminaryPeaks = (Peak*)malloc(maxPossiblePeaks * sizeof(Peak));
    check(!preliminaryPeaks, "findPeaks: Malloc prelim");
    int peakCount = 0;
    for(i = 1; i < N - 1; ++i)
    {
        if(data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] >= minHeight)
        {
            if(peakCount < maxPossiblePeaks)
            {
                preliminaryPeaks[peakCount].index = i;
                preliminaryPeaks[peakCount].radius = r_values[i];
                preliminaryPeaks[peakCount].intensity = data[i];
                peakCount++;
            }
            else
            {
                fprintf(stderr, "Peak find warn: Exceeded max buffer\n"); break;
            }
        }
    }
    if(peakCount == 0 || minDistance <= 1)
    {
        Peak* finalPeaks = (Peak*)realloc(preliminaryPeaks, peakCount * sizeof(Peak));
        if(peakCount > 0 && finalPeaks == NULL) { *foundPeaks = preliminaryPeaks; fprintf(stderr, "findPeaks Warn: realloc failed prelim\n"); }
        else { *foundPeaks = finalPeaks; }
        return peakCount;
    }
    bool* isSuppressed = (bool*)calloc(peakCount, sizeof(bool));
    check(!isSuppressed, "findPeaks: Calloc suppressed");
    for(i = 0; i < peakCount; ++i)
    {
        if(isSuppressed[i]) continue;
        for(j = i + 1; j < peakCount; ++j)
        {
            if(isSuppressed[j]) continue;
            int distance = abs(preliminaryPeaks[i].index - preliminaryPeaks[j].index);
            if(distance < minDistance)
            {
                if(preliminaryPeaks[j].intensity <= preliminaryPeaks[i].intensity) { isSuppressed[j] = true; }
                else { isSuppressed[i] = true; break; }
            }
        }
    }
    Peak* filteredPeaks = (Peak*)malloc(peakCount * sizeof(Peak));
    check(!filteredPeaks, "findPeaks: Malloc filtered");
    int filteredCount = 0;
    for(i = 0; i < peakCount; ++i){ if(!isSuppressed[i]){ filteredPeaks[filteredCount++] = preliminaryPeaks[i]; } }
    free(preliminaryPeaks);
    free(isSuppressed);
    Peak* finalPeaks = (Peak*)realloc(filteredPeaks, filteredCount * sizeof(Peak));
    if(filteredCount > 0 && finalPeaks == NULL) { *foundPeaks = filteredPeaks; fprintf(stderr, "findPeaks Warn: realloc failed final\n"); }
    else { *foundPeaks = finalPeaks; }
    return filteredCount;
}


// =========================================================================
// ============================ MAIN FUNCTION ============================
// =========================================================================
int main(int argc, char *argv[])
{
    // --- Basic Setup ---
    if (argc < 2)
    {
        printf("Usage: %s ParamFN [DarkAvgFN]\n", argv[0]);
        return 1;
    }
    printf("[%s] - Starting...\n", argv[0]);
    double t_start_main = get_wall_time_ms();
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    int dev_id = 0;
    gpuErrchk(cudaSetDevice(dev_id));
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, dev_id));
    printf("GPU %d: %s (CC %d.%d)\n", dev_id, prop.name, prop.major, prop.minor);

    // --- Read CSR Pixel Mapping Files (C Version) ---
    double t_start_map = get_wall_time_ms();
    check(!ReadCSRMapC(), "Failed read/validate CSR map files");
    printf("Read CSR Maps: %.3f ms\n", get_wall_time_ms() - t_start_map);

    // --- Read Parameters ---
    double t_start_params = get_wall_time_ms();
    double RMax_d=0, RMin_d=0, RBinSize_d=0, EtaMax_d=0, EtaMin_d=0, EtaBinSize_d=0, Lsd_d=0, px_d=0;
    int Normalize=1;
    char *ParamFN = argv[1];
    FILE *pF = fopen(ParamFN, "r");
    check(!pF, "Failed open param file: %s", ParamFN);
    char line[4096], key[1024], val_str[3072];
    int Nopt=0;
    long long GapI=0, BadPxI=0;
    int Topt[MAX_TRANSFORM_OPS]={0};
    int mkMap=0;
    int sumI=0;
    int doSm=0;
    int multiP=0;
    int pkFit=0;
    int nSpecP=0;
    int wr2D=0;
    double pkLoc[MAX_PEAK_LOCATIONS];
    while(fgets(line, sizeof(line), pF))
    {
        if(line[0]=='#'||isspace(line[0])||strlen(line)<3) continue;
        if (sscanf(line, "%1023s %[^\n]", key, val_str) == 2)
        {
            if (strcmp(key, "EtaBinSize") == 0) sscanf(val_str, "%lf", &EtaBinSize_d);
            else if (strcmp(key, "RBinSize") == 0) sscanf(val_str, "%lf", &RBinSize_d);
            else if (strcmp(key, "RMax") == 0) sscanf(val_str, "%lf", &RMax_d);
            else if (strcmp(key, "RMin") == 0) sscanf(val_str, "%lf", &RMin_d);
            else if (strcmp(key, "EtaMax") == 0) sscanf(val_str, "%lf", &EtaMax_d);
            else if (strcmp(key, "EtaMin") == 0) sscanf(val_str, "%lf", &EtaMin_d);
            else if (strcmp(key, "Lsd") == 0) sscanf(val_str, "%lf", &Lsd_d);
            else if (strcmp(key, "px") == 0) sscanf(val_str, "%lf", &px_d);
            else if (strcmp(key, "Normalize") == 0) sscanf(val_str, "%d", &Normalize);
            else if (strcmp(key, "GapIntensity") == 0) { sscanf(val_str, "%lld", &GapI); mkMap = 1; }
            else if (strcmp(key, "BadPxIntensity") == 0) { sscanf(val_str, "%lld", &BadPxI); mkMap = 1; }
            else if (strcmp(key, "ImTransOpt") == 0) { if(Nopt < MAX_TRANSFORM_OPS) sscanf(val_str, "%d", &Topt[Nopt++]); }
            else if (strcmp(key, "SumImages") == 0) sscanf(val_str, "%d", &sumI);
            else if (strcmp(key, "Write2D") == 0) sscanf(val_str, "%d", &wr2D);
            else if (strcmp(key, "DoSmoothing") == 0) sscanf(val_str, "%d", &doSm);
            else if (strcmp(key, "MultiplePeaks") == 0) sscanf(val_str, "%d", &multiP);
            else if (strcmp(key, "DoPeakFit") == 0) sscanf(val_str, "%d", &pkFit);
            else if (strcmp(key, "PeakLocation") == 0) { if(nSpecP < MAX_PEAK_LOCATIONS) { sscanf(val_str, "%lf", &pkLoc[nSpecP++]); multiP=1; pkFit=1; doSm=0; } else { printf("Warn: Max PeakLoc reached\n"); } }
        }
    }
    fclose(pF);
    float Lsd = (float)Lsd_d;
    float px = (float)px_d;
    float RMin = (float)RMin_d;
    float RMax = (float)RMax_d;
    float RBinSize = (float)RBinSize_d;
    float EtaMin = (float)EtaMin_d;
    float EtaMax = (float)EtaMax_d;
    float EtaBinSize = (float)EtaBinSize_d;
    check(g_total_pixels <= 0, "NrPixels invalid from map");
    check(Lsd <= 0 || px <= 0, "Lsd/px invalid");
    check(g_bigArrSize <= 0, "Invalid bin count");
    if(pkFit && nSpecP > 0)
    {
        multiP = 1;
        if (doSm) { printf("Warn: Smoothing disabled for specified peaks.\n"); doSm = 0; }
    }
    printf("Parameters Loaded/Validated (using CSR map info):\n");
    printf(" R Bins:    [%.3f .. %.3f], %d bins (step %.4f)\n", RMin, RMax, g_nRBins, RBinSize);
    printf(" Eta Bins:  [%.3f .. %.3f], %d bins (step %.4f)\n", EtaMin, EtaMax, g_nEtaBins, EtaBinSize);
    printf(" Detector:  %d x %d pixels\n", g_NrPixelsY, g_NrPixelsZ);
    printf(" Geometry:  Lsd=%.4f, px=%.6f\n", Lsd, px);
    printf(" Transforms(%d):", Nopt); {int i; for(i=0;i<Nopt;++i) printf(" %d", Topt[i]);} printf("\n");
    printf(" Options:   Normalize=%d, SumIntegrations=%d, Write2D=%d\n", Normalize, sumI, wr2D);
    printf(" Peak Fit:  Enabled=%d, MultiPeak=%d, Smooth=%d, NumSpecifiedPeaks=%d\n", pkFit, multiP, doSm, nSpecP);
    if (mkMap) printf(" Masking:   Will gen from Gap=%lld, BadPx=%lld\n", GapI, BadPxI);
    printf("Read Params: %.3f ms\n", get_wall_time_ms() - t_start_params);
    fflush(stdout);

    // --- Setup Bin Edges (Host, float) ---
    float *hEtaLo, *hEtaHi, *hRLo, *hRHi;
    hEtaLo = (float*)malloc(g_nEtaBins * sizeof(float)); check(!hEtaLo, "alloc hEtaLo");
    hEtaHi = (float*)malloc(g_nEtaBins * sizeof(float)); check(!hEtaHi, "alloc hEtaHi");
    hRLo   = (float*)malloc(g_nRBins * sizeof(float));   check(!hRLo, "alloc hRLo");
    hRHi   = (float*)malloc(g_nRBins * sizeof(float));   check(!hRHi, "alloc hRHi");
    REtaMapperF(RMin, EtaMin, g_nEtaBins, g_nRBins, EtaBinSize, RBinSize, hEtaLo, hEtaHi, hRLo, hRHi);

    // --- Host Memory Allocations ---
    float *hAvgDark = NULL;
    int64_t *hDarkInT = NULL;
    int64_t *hDarkIn = NULL;
    size_t SizeFile = g_total_pixels * BYTES_PER_PIXEL;
    hAvgDark = (float*)calloc(g_total_pixels, sizeof(float)); check(!hAvgDark, "alloc hAvgDark");
    hDarkInT = (int64_t*)malloc(SizeFile); check(!hDarkInT, "alloc hDarkInT");
    hDarkIn = (int64_t*)malloc(SizeFile); check(!hDarkIn, "alloc hDarkIn");

    // --- Device Memory Allocations ---
    float *dAvgDark = NULL;
    float *dProcessedImage = NULL;
    float *d_int1D = NULL;
    int *dMapMask = NULL;
    size_t mapMaskWC = 0;
    float *dSumMatrix = NULL;
    float *dIntArrFrame = NULL;
    float *dRTE = NULL;
    float *dEtaLo = NULL, *dEtaHi = NULL, *dRLo = NULL, *dRHi = NULL;
    int64_t *g_dTempTransformBuf1 = NULL;
    int64_t *g_dTempTransformBuf2 = NULL;
    bool darkSubEnabled = (argc > 2);

    gpuErrchk(cudaMalloc(&dProcessedImage, g_total_pixels * sizeof(float)));
    gpuErrchk(cudaMalloc(&dIntArrFrame, g_bigArrSize * sizeof(float)));
    gpuErrchk(cudaMalloc(&dRTE, g_bigArrSize * 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&dEtaLo, g_nEtaBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&dEtaHi, g_nEtaBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&dRLo, g_nRBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&dRHi, g_nRBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_int1D, g_nRBins * sizeof(float)));
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf1, g_total_pixels * sizeof(int64_t)));
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf2, g_total_pixels * sizeof(int64_t)));

    // Copy bin edges to GPU
    gpuErrchk(cudaMemcpy(dEtaLo, hEtaLo, g_nEtaBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaHi, hEtaHi, g_nEtaBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRLo, hRLo, g_nRBins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRHi, hRHi, g_nRBins * sizeof(float), cudaMemcpyHostToDevice));

    // --- Process Dark Frame ---
    double t_start_dark = get_wall_time_ms();
    int nDarkFramesRead = 0;
    int *hMapMask = NULL;
    if(darkSubEnabled)
    {
        char* darkFN = argv[2];
        FILE* fD = fopen(darkFN, "rb");
        check(!fD, "Failed open dark: %s", darkFN);
        fseek(fD, 0, SEEK_END);
        size_t szD = ftell(fD);
        rewind(fD);
        int nFD = szD / SizeFile;
        check(nFD == 0 || szD % SizeFile != 0, "Dark file %s incomplete", darkFN);
        printf("Reading dark file: %s, %d frames.\n", darkFN, nFD);
        int i;
        size_t j;
        for(i = 0; i < nFD; ++i)
        {
            check(fread(hDarkInT, 1, SizeFile, fD) != SizeFile, "Read failed dark frame %d", i);
            DoImageTransformationsSequential(Nopt, Topt, hDarkInT, hDarkIn, g_NrPixelsY, g_NrPixelsZ);
            if(mkMap == 1 && i == 0)
            {
                mapMaskWC = (g_total_pixels + 31) / 32;
                hMapMask = (int*)calloc(mapMaskWC, sizeof(int));
                check(!hMapMask, "alloc host mask");
                int maskedPixelCount = 0;
                for(j = 0; j < g_total_pixels; ++j)
                {
                    if(hDarkIn[j] == GapI || hDarkIn[j] == BadPxI)
                    {
                        SetBit(hMapMask, j);
                        maskedPixelCount++;
                    }
                }
                printf("Mask generated: %d pixels masked.\n", maskedPixelCount);
                gpuErrchk(cudaMalloc(&dMapMask, mapMaskWC * sizeof(int)));
                gpuErrchk(cudaMemcpy(dMapMask, hMapMask, mapMaskWC * sizeof(int), cudaMemcpyHostToDevice));
                mkMap = 0; // Done
            }
            for(j = 0; j < g_total_pixels; ++j)
            {
                hAvgDark[j] += (float)hDarkIn[j]; // Accumulate as float
            }
        }
        fclose(fD);
        nDarkFramesRead = nFD;
        if(nDarkFramesRead > 0)
        {
            size_t j;
            for(j = 0; j < g_total_pixels; ++j)
            {
                hAvgDark[j] /= (float)nDarkFramesRead;
            }
            gpuErrchk(cudaMalloc(&dAvgDark, g_total_pixels * sizeof(float)));
            gpuErrchk(cudaMemcpy(dAvgDark, hAvgDark, g_total_pixels * sizeof(float), cudaMemcpyHostToDevice));
            printf("Avg float dark copied to GPU.\n");
        }
    }
    if (!dAvgDark)
    { // If no dark file OR error reading dark
        gpuErrchk(cudaMalloc(&dAvgDark, g_total_pixels * sizeof(float)));
        gpuErrchk(cudaMemset(dAvgDark, 0, g_total_pixels * sizeof(float)));
        printf("Using zero float dark on GPU.\n");
    }

    // --- Initialize R, TTh, Eta array on GPU ---
    printf("Initializing static RTE array on GPU...\n");
    int initTPB = 256;
    int initBlocks = (g_bigArrSize + initTPB - 1) / initTPB;
    initialize_RTE_kernel<<<initBlocks, initTPB>>>( dRTE, g_bigArrSize, g_nRBins, g_nEtaBins, dRLo, dRHi, dEtaLo, dEtaHi, px, Lsd );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("GPU RTE init complete.\n");
    if(sumI)
    {
        gpuErrchk(cudaMalloc(&dSumMatrix, g_bigArrSize * sizeof(float)));
        gpuErrchk(cudaMemset(dSumMatrix, 0, g_bigArrSize * sizeof(float)));
    }
    printf("Processed dark/mask: %.3f ms\n", get_wall_time_ms() - t_start_dark);

    // --- Network Setup ---
    CHUNK_SIZE = SizeFile;
    TOTAL_MSG_SIZE = HEADER_SIZE + CHUNK_SIZE;
    int server_fd;
    struct sockaddr_in server_addr;
    queue_init(&process_queue);
    check((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0, "Socket fail");
    int sock_opt = 1;
    check(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &sock_opt, sizeof(sock_opt)), "setsockopt fail");
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    check(bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0, "Bind fail");
    check(listen(server_fd, MAX_CONNECTIONS) < 0, "Listen fail");
    printf("Server listening on port %d\n", PORT);
    pthread_t accept_thread;
    check(pthread_create(&accept_thread, NULL, accept_connections, &server_fd) != 0, "Accept thread fail");

    // --- Prepare for Main Loop ---
    FILE *fLineout = fopen("lineout.bin", "wb");
    check(!fLineout, "Open lineout.bin fail");
    FILE *fFit = NULL;
    if(pkFit)
    {
        fFit = fopen("fit.bin", "wb");
        check(!fFit, "Open fit.bin fail");
    }
    FILE *f2D = NULL;
    if (wr2D)
    {
        printf("Will write 2D patterns to Int2D.bin\n");
        f2D = fopen("Int2D.bin", "wb");
        check(!f2D, "Open Int2D.bin fail");
    }
    cudaEvent_t ev_proc_start, ev_proc_stop, ev_integ_start, ev_integ_stop;
    cudaEvent_t ev_norm_start, ev_norm_stop, ev_prof_start, ev_prof_stop, ev_d2h_start, ev_d2h_stop;
    gpuErrchk(cudaEventCreate(&ev_proc_start)); gpuErrchk(cudaEventCreate(&ev_proc_stop));
    gpuErrchk(cudaEventCreate(&ev_integ_start)); gpuErrchk(cudaEventCreate(&ev_integ_stop));
    gpuErrchk(cudaEventCreate(&ev_norm_start)); gpuErrchk(cudaEventCreate(&ev_norm_stop));
    gpuErrchk(cudaEventCreate(&ev_prof_start)); gpuErrchk(cudaEventCreate(&ev_prof_stop));
    gpuErrchk(cudaEventCreate(&ev_d2h_start)); gpuErrchk(cudaEventCreate(&ev_d2h_stop));

    float t_proc_gpu=0, t_integ_gpu=0, t_norm_gpu=0, t_prof_gpu=0, t_d2h_gpu=0;
    double t_qp_cpu=0, t_write1d_cpu=0, t_fit_cpu=0, t_writefit_cpu=0, t_write2d_cpu=0, t_loop_cpu=0;
    double t_start_loop, t_end_loop;
    int firstFrame = 1;
    float *hIntArrFrame = NULL;
    float *hRTE = NULL;
    float *h_int1D_f = NULL;
    if (wr2D)
    {
        gpuErrchk(cudaMallocHost((void**)&hIntArrFrame, g_bigArrSize * sizeof(float)));
        check(!hIntArrFrame, "alloc pinned hIntArrFrame");
    }
    gpuErrchk(cudaMallocHost((void**)&hRTE, g_bigArrSize * 3 * sizeof(float)));
    check(!hRTE, "alloc pinned hRTE");
    gpuErrchk(cudaMallocHost((void**)&h_int1D_f, g_nRBins * sizeof(float)));
    check(!h_int1D_f, "alloc pinned h_int1D_f");
    double *hR_d = (double*)calloc(g_nRBins, sizeof(double));
    check(!hR_d, "alloc hR_d");
    double *hLineout_d = (double*)malloc(g_nRBins * 2 * sizeof(double));
    check(!hLineout_d, "alloc hLineout_d");
    printf("Setup complete. Starting main processing loop...\n");
    double t_end_setup = get_wall_time_ms();
    printf("Total setup time: %.3f ms\n", t_end_setup - t_start_main);
    fflush(stdout);

    // =========================== Main Processing Loop ===========================
    int frameCounter = 0;
    while (keep_running)
    {
        t_start_loop = get_wall_time_ms();
        DataChunk chunk;
        if(queue_pop(&process_queue, &chunk) < 0) break;
        t_qp_cpu = get_wall_time_ms() - t_start_loop;

        // --- GPU Processing Stage ---
        gpuErrchk(cudaEventRecord(ev_proc_start, 0));
        ProcessImageGPU(chunk.data, dProcessedImage, dAvgDark, Nopt, Topt, g_NrPixelsY, g_NrPixelsZ, darkSubEnabled, g_dTempTransformBuf1, g_dTempTransformBuf2);
        gpuErrchk(cudaEventRecord(ev_proc_stop, 0));

        // --- GPU Integration Stage (CSR) ---
        gpuErrchk(cudaMemsetAsync(dIntArrFrame, 0, g_bigArrSize * sizeof(float), 0));
        int integTPB = THREADS_PER_BLOCK_INTEGRATE;
        int integBlocks = (g_total_pixels + integTPB - 1) / integTPB;
        gpuErrchk(cudaEventRecord(ev_integ_start, 0));
        integrate_kernel_csr<<<integBlocks, integTPB>>>( dProcessedImage, d_csr_row_offsets, d_csr_col_indices, d_csr_values, dIntArrFrame, g_total_pixels, dMapMask, mapMaskWC);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_integ_stop, 0));

        // --- GPU Normalization Stage ---
        gpuErrchk(cudaEventRecord(ev_norm_start, 0));
        if (Normalize)
        {
             int normTPB = THREADS_PER_BLOCK_NORMALIZE;
             int normBlocks = (g_bigArrSize + normTPB - 1) / normTPB;
             normalize_kernel<<<normBlocks, normTPB>>>( dIntArrFrame, d_total_area_per_bin, g_bigArrSize);
             gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaEventRecord(ev_norm_stop, 0));

        // --- GPU 1D Profile Stage ---
        size_t profileSharedMem = (THREADS_PER_BLOCK_PROFILE / 32) * sizeof(float) * 2;
        gpuErrchk(cudaEventRecord(ev_prof_start, 0));
        calculate_1D_profile_kernel<<<g_nRBins, THREADS_PER_BLOCK_PROFILE, profileSharedMem>>>( dIntArrFrame, d_total_area_per_bin, d_int1D, g_nRBins, g_nEtaBins, g_bigArrSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_prof_stop, 0));

        // --- D->H Copy Stage (Asynchronous) ---
        gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
        gpuErrchk(cudaMemcpyAsync(h_int1D_f, d_int1D, g_nRBins * sizeof(float), cudaMemcpyDeviceToHost, 0));
        if(firstFrame == 1)
        {
             gpuErrchk(cudaMemcpyAsync(hRTE, dRTE, g_bigArrSize * 3 * sizeof(float), cudaMemcpyDeviceToHost, 0));
        }
        if(wr2D && hIntArrFrame)
        {
            gpuErrchk(cudaMemcpyAsync(hIntArrFrame, dIntArrFrame, g_bigArrSize * sizeof(float), cudaMemcpyDeviceToHost, 0));
        }
        gpuErrchk(cudaEventRecord(ev_d2h_stop, 0));

        // --- Synchronize GPU Events ---
        gpuErrchk(cudaEventSynchronize(ev_proc_stop)); gpuErrchk(cudaEventElapsedTime(&t_proc_gpu, ev_proc_start, ev_proc_stop));
        gpuErrchk(cudaEventSynchronize(ev_integ_stop)); gpuErrchk(cudaEventElapsedTime(&t_integ_gpu, ev_integ_start, ev_integ_stop));
        gpuErrchk(cudaEventSynchronize(ev_norm_stop)); gpuErrchk(cudaEventElapsedTime(&t_norm_gpu, ev_norm_start, ev_norm_stop));
        gpuErrchk(cudaEventSynchronize(ev_prof_stop)); gpuErrchk(cudaEventElapsedTime(&t_prof_gpu, ev_prof_start, ev_prof_stop));
        gpuErrchk(cudaEventSynchronize(ev_d2h_stop)); gpuErrchk(cudaEventElapsedTime(&t_d2h_gpu, ev_d2h_start, ev_d2h_stop));

        // --- CPU Processing Stage ---
        int r; // Loop variable for C99
        if(firstFrame == 1)
        {
            for(r = 0; r < g_nRBins; ++r)
            {
                 hR_d[r] = (double)hRTE[r * g_nEtaBins];
                 hLineout_d[r * 2] = hR_d[r];
                 hLineout_d[r * 2 + 1] = 0.0;
            }
            printf("Initialized host R array (double).\n");
            firstFrame = 0;
        }

        // --- Write 2D Integrated Data ---
        double t_write2d_start = get_wall_time_ms();
        if (wr2D && f2D && hIntArrFrame)
        {
            size_t written = fwrite(hIntArrFrame, sizeof(float), g_bigArrSize, f2D);
            if (written != g_bigArrSize)
            {
                fprintf(stderr, "Warn: Failed write 2D frame %d\n", chunk.dataset_num);
            }
        }
        t_write2d_cpu = get_wall_time_ms() - t_write2d_start;

        // --- Prepare and Write 1D Lineout Data ---
        double maxInt_d = -1.0;
        int maxIntLoc = -1;
        for(r = 0; r < g_nRBins; ++r)
        {
            double intensity_d = (double)h_int1D_f[r]; // Convert float GPU result to double
            hLineout_d[r * 2 + 1] = intensity_d;
            if(!multiP && intensity_d > maxInt_d)
            {
                 maxInt_d = intensity_d; maxIntLoc = r;
            }
        }
        double t_write1d_start = get_wall_time_ms();
        if (fLineout)
        {
             size_t written = fwrite(hLineout_d, sizeof(double), g_nRBins * 2, fLineout);
             if (written != (size_t)g_nRBins * 2)
             {
                  fprintf(stderr, "Warn: Failed write lineout frame %d\n", chunk.dataset_num);
             }
             else
             {
                  fflush(fLineout);
             }
        }
        t_write1d_cpu = get_wall_time_ms() - t_write1d_start;

        // --- Peak Finding and Fitting ---
        double t_fit_start = get_wall_time_ms();
        int currentPeakCount = 0;
        double *sendFitParams = NULL;
        if(pkFit)
        {
            Peak *pks = NULL;
            double *h_int1D_d = (double*)malloc(g_nRBins * sizeof(double));
            check(!h_int1D_d, "alloc h_int1D_d");
            for(r=0; r<g_nRBins; ++r) h_int1D_d[r] = hLineout_d[r*2+1];

            if(multiP)
            {
                if(nSpecP > 0)
                {
                    pks=(Peak*)malloc(nSpecP*sizeof(Peak));
                    check(!pks,"pkFit malloc spec pks");
                    int validPeakCount=0;
                    int p;
                    for(p=0;p<nSpecP;++p)
                    {
                        int bestBin=-1;
                        double minDiff=1e10;
                        for(r=0;r<g_nRBins;++r)
                        {
                            double diff=fabs(hR_d[r]-pkLoc[p]);
                            if(diff<minDiff){ minDiff=diff; bestBin=r; }
                        }
                        if(bestBin!=-1&&minDiff<RBinSize*2.0)
                        {
                            pks[validPeakCount].index=bestBin;
                            pks[validPeakCount].radius=hR_d[bestBin];
                            pks[validPeakCount].intensity=h_int1D_d[bestBin];
                            validPeakCount++;
                        }
                        else
                        {
                            printf("Warn: Spec peak R=%.4f ignored\n",pkLoc[p]);
                        }
                    }
                    currentPeakCount=validPeakCount;
                    if(validPeakCount==0){free(pks); pks=NULL;}
                    else if(validPeakCount<nSpecP){ Peak*reallocPks=(Peak*)realloc(pks,validPeakCount*sizeof(Peak)); if(reallocPks) pks=reallocPks; }
                }
                else
                {
                    double *dataToFindPeaks=h_int1D_d;
                    double *smoothedData=NULL;
                    if(doSm)
                    {
                        smoothedData=(double*)malloc(g_nRBins*sizeof(double));
                        check(!smoothedData,"pkFit malloc smooth");
                        smoothData(h_int1D_d,smoothedData,g_nRBins,7);
                        dataToFindPeaks=smoothedData;
                    }
                    currentPeakCount = findPeaks(dataToFindPeaks, hR_d, g_nRBins, &pks, 0.0, 5);
                    if(smoothedData) free(smoothedData);
                }
            }
            else
            {
                if(maxIntLoc!=-1)
                {
                    currentPeakCount=1;
                    pks=(Peak*)malloc(sizeof(Peak));
                    check(!pks,"pkFit malloc single");
                    pks[0].index=maxIntLoc;
                    pks[0].radius=hR_d[maxIntLoc];
                    pks[0].intensity=maxInt_d;
                } else { currentPeakCount=0; pks=NULL; }
            }

            if (currentPeakCount > 0 && pks != NULL)
            {
                int nFitParams=currentPeakCount*4+1;
                double *fitParams=(double*)malloc(nFitParams*sizeof(double)); check(!fitParams, "malloc fitParams");
                double *lowerBounds=(double*)malloc(nFitParams*sizeof(double)); check(!lowerBounds, "malloc lowerB");
                double *upperBounds=(double*)malloc(nFitParams*sizeof(double)); check(!upperBounds, "malloc upperB");
                double maxOverallIntensity=0.0;
                for(r=0;r<g_nRBins;++r) if(h_int1D_d[r]>maxOverallIntensity) maxOverallIntensity=h_int1D_d[r];
                if(maxOverallIntensity<=0) maxOverallIntensity=1.0;
                int p;
                for(p=0;p<currentPeakCount;++p)
                {
                    int base=p*4; double initialCenter=pks[p].radius; double initialIntensity=pks[p].intensity; double initialSigma=RBinSize*2.0;
                    fitParams[base+0]=initialIntensity; fitParams[base+1]=0.5; fitParams[base+2]=initialCenter; fitParams[base+3]=initialSigma;
                    lowerBounds[base+0]=0.0; lowerBounds[base+1]=0.0; lowerBounds[base+2]=initialCenter-RBinSize*5.0; lowerBounds[base+3]=RBinSize*0.5;
                    upperBounds[base+0]=maxOverallIntensity*2.0; upperBounds[base+1]=1.0; upperBounds[base+2]=initialCenter+RBinSize*5.0; upperBounds[base+3]=(RMax-RMin)/4.0;
                }
                fitParams[nFitParams-1]=0.0; lowerBounds[nFitParams-1]=-maxOverallIntensity; upperBounds[nFitParams-1]=maxOverallIntensity;
                dataFit fitData; fitData.nrBins=g_nRBins; fitData.R=hR_d; fitData.Int=h_int1D_d;
                nlopt_opt opt=nlopt_create(NLOPT_LN_NELDERMEAD,nFitParams);
                nlopt_set_lower_bounds(opt,lowerBounds); nlopt_set_upper_bounds(opt,upperBounds); nlopt_set_min_objective(opt,problem_function_global_bg,&fitData); nlopt_set_xtol_rel(opt,1e-4); nlopt_set_maxeval(opt,500*nFitParams);
                double minObjectiveValue;
                int nlopt_rc=nlopt_optimize(opt,fitParams,&minObjectiveValue);
                if(nlopt_rc<0)
                {
                    printf("F#%d: NLopt failed %d\n",chunk.dataset_num,nlopt_rc); currentPeakCount=0; free(fitParams);
                }
                else
                {
                    sendFitParams=(double*)malloc(currentPeakCount*5*sizeof(double)); check(!sendFitParams,"pkFit malloc send");
                    double globalBG=fitParams[nFitParams-1];
                    for(p=0;p<currentPeakCount;++p){ sendFitParams[p*5+0]=fitParams[p*4+0]; sendFitParams[p*5+1]=globalBG; sendFitParams[p*5+2]=fitParams[p*4+1]; sendFitParams[p*5+3]=fitParams[p*4+2]; sendFitParams[p*5+4]=fitParams[p*4+3]; }
                    free(fitParams);
                }
                nlopt_destroy(opt); free(lowerBounds); free(upperBounds);
            }
            if(pks) free(pks);
            free(h_int1D_d);
        }
        t_fit_cpu = get_wall_time_ms() - t_fit_start;

        // --- Write Peak Fit Results ---
        double t_writefit_start = get_wall_time_ms();
        if(pkFit && currentPeakCount > 0 && sendFitParams != NULL && fFit)
        {
            size_t written = fwrite(sendFitParams, sizeof(double), currentPeakCount * 5, fFit);
            if (written != (size_t)currentPeakCount * 5)
            {
                 fprintf(stderr, "Warn: Failed write fit frame %d\n", chunk.dataset_num);
            }
            else
            {
                fflush(fFit);
            }
        }
        if (sendFitParams) free(sendFitParams);
        t_writefit_cpu = get_wall_time_ms() - t_writefit_start;

        // --- Free received data buffer ---
        gpuWarnchk(cudaFreeHost(chunk.data));

        // --- Timing and Output ---
        t_end_loop = get_wall_time_ms();
        t_loop_cpu = t_end_loop - t_start_loop;
        printf("F#%d: Ttl:%.2f| QP:%.2f GPU(Pr:%.2f Int:%.2f Nrm:%.2f Pf:%.2f D2H:%.2f) CPU(W2D:%.2f W1D:%.2f Fit:%.2f WFit:%.2f)\n",
               chunk.dataset_num, t_loop_cpu, t_qp_cpu,
               t_proc_gpu, t_integ_gpu, t_norm_gpu, t_prof_gpu, t_d2h_gpu,
               t_write2d_cpu, t_write1d_cpu, t_fit_cpu, t_writefit_cpu);
        fflush(stdout);

        frameCounter++;
    } // ======================== End Main Processing Loop ========================

    printf("Processing loop finished. Processed %d frames. Cleaning up...\n", frameCounter);

    // --- Cleanup ---
    if(fLineout) fclose(fLineout);
    if(fFit) fclose(fFit);
    if(f2D) fclose(f2D);
    free(hAvgDark);
    free(hDarkInT);
    free(hDarkIn);
    if(hIntArrFrame) gpuWarnchk(cudaFreeHost(hIntArrFrame));
    if(hRTE) gpuWarnchk(cudaFreeHost(hRTE));
    if(h_int1D_f) gpuWarnchk(cudaFreeHost(h_int1D_f));
    free(hR_d);
    free(hLineout_d);
    free(hEtaLo);
    free(hEtaHi);
    free(hRLo);
    free(hRHi);
    if(hMapMask) free(hMapMask);

    if(dAvgDark) gpuWarnchk(cudaFree(dAvgDark));
    if(dProcessedImage) gpuWarnchk(cudaFree(dProcessedImage));
    if(d_int1D) gpuWarnchk(cudaFree(d_int1D));
    if(dMapMask) gpuWarnchk(cudaFree(dMapMask));
    if(dSumMatrix) gpuWarnchk(cudaFree(dSumMatrix));
    if(dIntArrFrame) gpuWarnchk(cudaFree(dIntArrFrame));
    if(dRTE) gpuWarnchk(cudaFree(dRTE));
    if(dEtaLo) gpuWarnchk(cudaFree(dEtaLo));
    if(dEtaHi) gpuWarnchk(cudaFree(dEtaHi));
    if(dRLo) gpuWarnchk(cudaFree(dRLo));
    if(dRHi) gpuWarnchk(cudaFree(dRHi));
    if(g_dTempTransformBuf1) gpuWarnchk(cudaFree(g_dTempTransformBuf1));
    if(g_dTempTransformBuf2) gpuWarnchk(cudaFree(g_dTempTransformBuf2));
    if(d_csr_row_offsets) gpuWarnchk(cudaFree(d_csr_row_offsets));
    if(d_csr_col_indices) gpuWarnchk(cudaFree(d_csr_col_indices));
    if(d_csr_values) gpuWarnchk(cudaFree(d_csr_values));
    if(d_total_area_per_bin) gpuWarnchk(cudaFree(d_total_area_per_bin));

    gpuWarnchk(cudaEventDestroy(ev_proc_start)); gpuWarnchk(cudaEventDestroy(ev_proc_stop));
    gpuWarnchk(cudaEventDestroy(ev_integ_start)); gpuWarnchk(cudaEventDestroy(ev_integ_stop));
    gpuWarnchk(cudaEventDestroy(ev_norm_start)); gpuWarnchk(cudaEventDestroy(ev_norm_stop));
    gpuWarnchk(cudaEventDestroy(ev_prof_start)); gpuWarnchk(cudaEventDestroy(ev_prof_stop));
    gpuWarnchk(cudaEventDestroy(ev_d2h_start)); gpuWarnchk(cudaEventDestroy(ev_d2h_stop));

    printf("Attempting to shut down network acceptor thread...\n");
    if (server_fd >= 0)
    {
         printf("Closing server listening socket %d...\n", server_fd);
         shutdown(server_fd, SHUT_RDWR);
         close(server_fd);
         server_fd = -1;
    }
    printf("Sending cancellation request to accept thread...\n");
    int cancel_ret = pthread_cancel(accept_thread);
    if (cancel_ret != 0)
    {
        fprintf(stderr, "Warning: Failed cancel accept thread: %s\n", strerror(cancel_ret));
    }
    printf("Joining accept thread...\n");
    void *thread_result;
    int join_ret = pthread_join(accept_thread, &thread_result);
    if (join_ret != 0)
    {
         fprintf(stderr, "Warning: Failed join accept thread: %s\n", strerror(join_ret));
    }
    else
    {
        if (thread_result == PTHREAD_CANCELED) { printf("Accept thread canceled and joined.\n"); }
        else { printf("Accept thread joined normally.\n"); }
    }

    queue_destroy(&process_queue);
    printf("[%s] - Exiting cleanly.\n", argv[0]);
    return 0;
}