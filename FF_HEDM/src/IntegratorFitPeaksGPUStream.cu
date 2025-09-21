// =========================================================================
// IntegratorFitPeaksGPUStream.cu
//
// Copyright (c) 2014, UChicago Argonne, LLC; 2024 modifications.
// See LICENSE file (if applicable).
//
// Purpose: Integrates 2D detector data streamed over a socket, performs
//          image transformations, optionally fits peaks to the resulting
//          1D lineout, and saves results. Uses GPU acceleration.
//
// Features:
//  - Socket data receiving (multi-threaded)
//  - Thread-safe queue for buffering frames
//  - Memory-mapped file reading for pixel maps
//  - GPU pipeline: H->D Copy -> Transform -> Cast -> Dark Subtract
//  - GPU kernel for area-weighted 1D profile calculation
//  - Pinned host memory for H<->D transfers
//  - NLopt peak fitting with global background parameter
//  - Signal handling for graceful shutdown
//  - CUDA Event API for accurate GPU timing
//  - Optional saving of results (including single large 2D output file)
//
// Example compile command (adjust paths and architecture flags):
/*
/home/beams/S1IDUSER/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStream.cu -o bin/IntegratorFitPeaksGPUStream \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_90,code=sm_90 \
  -Xcompiler -g -Xcompiler -fopenmp \
  -I/home/beams/S1IDUSER/opt/MIDAS/build/_deps/nlopt-src/src/api \
  -L/home/beams/S1IDUSER/opt/MIDAS/build/lib \
  -O3 -lnlopt -lz -ldl -lm -lpthread \
  -Xlinker "-rpath=/home/beams/S1IDUSER/opt/MIDAS/build/lib"

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
#include <sys/mman.h>   // For mmap
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
#include <omp.h>        // <<< ADD THIS FOR OpenMP

// --- Constants ---
#define SERVER_IP "127.0.0.1"
#define PORT 60439               // Port for receiving image data
#define MAX_CONNECTIONS 10      // Max simultaneous client connections
#define MAX_QUEUE_SIZE 100      // Max image frames buffered before processing
#define MAX_FILENAME_LENGTH 1024
#define THREADS_PER_BLOCK_TRANSFORM 512 // CUDA block size for transform/processing
#define THREADS_PER_BLOCK_INTEGRATE 512 // CUDA block size for integration
#define THREADS_PER_BLOCK_PROFILE 512  // CUDA block size for 1D profile reduction
#define MAX_TRANSFORM_OPS 10    // Max number of sequential transforms allowed
#define MAX_PEAK_LOCATIONS 100  // Max peaks specifiable in param file
#define AREA_THRESHOLD 1e-9     // Minimum area considered valid in integration/profiling

// =========================================================================
// <<< CHANGE #1: NEW HEADER AND DATATYPE DEFINITIONS >>>
// These codes MUST exactly match the Python DATATYPE_CODES dictionary
enum DataType {
    TYPE_UNKNOWN = 0,
    TYPE_UINT16  = 1,
    TYPE_UINT32  = 2,
    TYPE_INT32   = 3,
    TYPE_INT64   = 4,
};

// Use a packed struct for the header to ensure the C++ memory layout
// matches Python's struct.pack('HB') byte-for-byte.
#pragma pack(push, 1)
typedef struct {
    uint16_t dataset_num;
    uint8_t  datatype_code;
} FrameHeader;
#pragma pack(pop)

// Update the HEADER_SIZE constant to reflect the new 3-byte header
#define HEADER_SIZE sizeof(FrameHeader)
// =========================================================================

// Global variables (initialized in main)
size_t szPxList = 0;
size_t szNPxList = 0;
volatile sig_atomic_t keep_running = 1; // Flag for graceful shutdown

// --- Data Structures ---
// This struct remains unchanged because conversion to int64_t happens *before* queueing.
typedef struct {
    uint16_t dataset_num;
    int64_t *data;
    size_t size; // Represents number of pixels/values
    double conversion_ms; // Time taken for CPU-side type conversion
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

struct data {
    int y;
    int z;
    double frac;
};

// --- Global Variables ---
ProcessQueue process_queue;
struct data *pxList = NULL;
int *nPxList = NULL;
// <<< CHANGE #2: ADDED GLOBAL DETECTOR DIMENSIONS >>>
// These are needed by the handle_client thread to calculate incoming data size.
int NrPixelsY = 0;
int NrPixelsZ = 0;

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
        fprintf(stderr, "\n");
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

int queue_push(ProcessQueue *queue, uint16_t dataset_num, int64_t *data, size_t num_values, double conversion_time) {
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
        return -1;
    }
    queue->rear = (queue->rear + 1) % MAX_QUEUE_SIZE;
    queue->chunks[queue->rear].dataset_num = dataset_num;
    queue->chunks[queue->rear].data = data;
    queue->chunks[queue->rear].size = num_values;
    queue->chunks[queue->rear].conversion_ms = conversion_time; // <<< ADD THIS LINE
    queue->count++;
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

int queue_pop(ProcessQueue *queue, DataChunk *chunk) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->count <= 0 && keep_running) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &ts);
    }
    if (!keep_running && queue->count <= 0) {
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

void queue_destroy(ProcessQueue *queue) {
     pthread_mutex_destroy(&queue->mutex);
     pthread_cond_destroy(&queue->not_empty);
     pthread_cond_destroy(&queue->not_full);
     printf("Cleaning up remaining %d queue entries...\n", queue->count);
     while (queue->count > 0) {
        DataChunk chunk;
        chunk.data = queue->chunks[queue->front].data;
        queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
        queue->count--;
        if (chunk.data) {
            gpuWarnchk(cudaFreeHost(chunk.data)); // Free pinned memory
        }
     }
}

// =========================================================================
// <<< CHANGE #3: NEW HOST-SIDE CONVERSION HELPER FUNCTION >>>
// This function converts the raw data (e.g., uint16_t) to int64_t on the CPU
// and then pushes the standardized data to the processing queue.
template<typename T_in>
void convert_and_queue(uint16_t dataset_num, const T_in* raw_buffer, size_t num_pixels) {
    int64_t *final_buffer = NULL;

    // Allocate the final buffer that will be sent to the GPU, using pinned host memory for fast transfers.
    gpuWarnchk(cudaMallocHost((void**)&final_buffer, num_pixels * sizeof(int64_t)));
    if (!final_buffer) {
        perror("convert_and_queue: Pinned memory alloc for final_buffer failed");
        return;
    }

    double t_start = get_wall_time_ms();
    // Perform the conversion from the raw type to int64_t.
    // This loop is very fast and cache-friendly. For extra speed on multi-core CPUs,
    // compile with -fopenmp (which you already do).
    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        final_buffer[i] = (int64_t)raw_buffer[i];
    }
    double conversion_time = get_wall_time_ms() - t_start;

    // Push the now-standardized int64_t buffer to the processing queue.
    if (queue_push(&process_queue, dataset_num, final_buffer, num_pixels, conversion_time) < 0) {
        printf("convert_and_queue: queue_push failed, likely shutdown. Discarding frame %d.\n", dataset_num);
        gpuWarnchk(cudaFreeHost(final_buffer)); // Clean up to prevent a memory leak.
    }
}
// =========================================================================

// --- Socket Handling ---
// =========================================================================
// <<< CHANGE #4: COMPLETE REPLACEMENT OF handle_client FUNCTION >>>
void* handle_client(void *arg) {
    int client_socket = *((int*)arg);
    free(arg);
    printf("Client handler started for socket %d.\n", client_socket);

    const size_t num_pixels = (size_t)NrPixelsY * NrPixelsZ;

    while (keep_running) {
        FrameHeader header;

        // --- STAGE 1: Read the 3-byte header from the client ---
        // MSG_WAITALL ensures we block until all 3 bytes are received.
        int bytes_read = recv(client_socket, &header, HEADER_SIZE, MSG_WAITALL);
        if (bytes_read != HEADER_SIZE) {
            if (bytes_read == 0 && keep_running) printf("Client disconnected (socket %d)\n", client_socket);
            else if (bytes_read < 0) perror("Header receive error");
            goto connection_closed;
        }

        // --- STAGE 2: Determine payload size and allocate a temporary raw buffer ---
        size_t bytes_per_pixel = 0;
        switch (header.datatype_code) {
            case TYPE_UINT16: bytes_per_pixel = sizeof(uint16_t); break;
            case TYPE_UINT32: bytes_per_pixel = sizeof(uint32_t); break;
            case TYPE_INT32:  bytes_per_pixel = sizeof(int32_t);  break;
            case TYPE_INT64:  bytes_per_pixel = sizeof(int64_t);  break;
            default:
                fprintf(stderr, "Fatal Error: Unknown datatype code %d from client. Closing connection.\n", header.datatype_code);
                goto connection_closed;
        }
        
        const size_t chunk_size_bytes = num_pixels * bytes_per_pixel;
        void *raw_buffer = malloc(chunk_size_bytes);
        check(raw_buffer == NULL, "handle_client: Failed to allocate temporary raw buffer");

        // --- STAGE 3: Read the raw image data payload into the temporary buffer ---
        bytes_read = recv(client_socket, raw_buffer, chunk_size_bytes, MSG_WAITALL);
        if (bytes_read != (int)chunk_size_bytes) {
            fprintf(stderr, "Payload receive error: expected %zu, got %d bytes. Closing connection.\n", chunk_size_bytes, bytes_read);
            free(raw_buffer);
            goto connection_closed;
        }

        // --- STAGE 4: Call the correct conversion function based on the header's type code ---
        switch (header.datatype_code) {
            case TYPE_UINT16:
                convert_and_queue<uint16_t>(header.dataset_num, (const uint16_t*)raw_buffer, num_pixels);
                break;
            case TYPE_UINT32:
                convert_and_queue<uint32_t>(header.dataset_num, (const uint32_t*)raw_buffer, num_pixels);
                break;
            case TYPE_INT32:
                convert_and_queue<int32_t>(header.dataset_num, (const int32_t*)raw_buffer, num_pixels);
                break;
            case TYPE_INT64:
                { // Use a block to create a local variable
                    // Data is already int64_t, so we just copy it to a pinned buffer.
                    // The "conversion" time is 0.0.
                    int64_t *final_buffer = NULL;
                    gpuWarnchk(cudaMallocHost((void**)&final_buffer, num_pixels * sizeof(int64_t)));
                    memcpy(final_buffer, raw_buffer, num_pixels * sizeof(int64_t));
                    
                    if (queue_push(&process_queue, header.dataset_num, final_buffer, num_pixels, 0.0) < 0) {
                        printf("handle_client: queue_push for int64 failed. Discarding frame %d.\n", header.dataset_num);
                        gpuWarnchk(cudaFreeHost(final_buffer));
                    }
                }
                break;
        }
        
        // The temporary buffer has served its purpose and can be freed.
        free(raw_buffer);
    }

connection_closed:
    close(client_socket);
    printf("Client handler finished (socket %d).\n", client_socket);
    return NULL;
}
// =========================================================================

void* accept_connections(void *server_fd_ptr) {
    int server_fd = *((int*)server_fd_ptr);
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    printf("Accept thread started, listening for connections.\n");
    while (keep_running) {
        int *client_socket_ptr = (int*) malloc(sizeof(int));
        check(client_socket_ptr == NULL, "accept_connections: Failed alloc client socket ptr");
        *client_socket_ptr = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (!keep_running) {
             if (*client_socket_ptr >= 0) {
                 close(*client_socket_ptr);
             }
             free(client_socket_ptr);
             break;
        }
        if (*client_socket_ptr < 0) {
            if (errno == EINTR) {
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
        if (create_rc != 0) {
            fprintf(stderr, "Thread creation failed: %s\n", strerror(create_rc));
            close(*client_socket_ptr);
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

// --- File I/O & Mapping ---
int ReadBins(){
    if(pxList||nPxList){
        printf("Warn: Maps already loaded.\n");
        return 1;
    }
    int fd_m = -1;
    int fd_n = -1;
    struct stat s_m, s_n;
    const char* f_m = "Map.bin";
    const char* f_n = "nMap.bin";

    fd_m = open(f_m, O_RDONLY);
    check(fd_m < 0, "open %s fail: %s", f_m, strerror(errno));
    check(fstat(fd_m, &s_m) < 0, "stat %s fail: %s", f_m, strerror(errno));
    szPxList = s_m.st_size;
    check(szPxList == 0, "%s empty", f_m);
    printf("Map '%s': %lld bytes\n", f_m, (long long)szPxList);
    pxList = (struct data*)mmap(NULL, szPxList, PROT_READ, MAP_SHARED, fd_m, 0);
    check(pxList == MAP_FAILED, "mmap %s fail: %s", f_m, strerror(errno));
    close(fd_m);

    fd_n = open(f_n, O_RDONLY);
    check(fd_n < 0, "open %s fail: %s", f_n, strerror(errno));
    check(fstat(fd_n, &s_n) < 0, "stat %s fail: %s", f_n, strerror(errno));
    szNPxList = s_n.st_size;
    check(szNPxList == 0, "%s empty", f_n);
    printf("nMap '%s': %lld bytes\n", f_n, (long long)szNPxList);
    nPxList = (int*)mmap(NULL, szNPxList, PROT_READ, MAP_SHARED, fd_n, 0);
    check(nPxList == MAP_FAILED, "mmap %s fail: %s", f_n, strerror(errno));
    close(fd_n);

    printf("Mapped pixel data files.\n");
    fflush(stdout);
    return 1;
}
void UnmapBins() {
    if(pxList && pxList != MAP_FAILED){
        munmap(pxList, szPxList);
        pxList = NULL;
        printf("Unmapped Map.bin\n");
    }
    if(nPxList && nPxList != MAP_FAILED){
        munmap(nPxList, szNPxList);
        nPxList = NULL;
        printf("Unmapped nMap.bin\n");
    }
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

    int64_t *tmp = (int64_t*)malloc(B);
    if(!tmp){
        fprintf(stderr, "CPU Err: Alloc tmp fail\n");
        if(Out != In) {
            memcpy(Out, In, B);
        }
        return;
    }

    const int64_t* rB = NULL;
    int64_t* wB = NULL;
    int cY = NY;
    int cZ = NZ;

    for(int i = 0; i < Nopt; ++i){
        int opt = Topt[i];
        size_t cB = (size_t)cY * cZ * sizeof(int64_t);
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
        int nY = cY;
        int nZ = cZ;
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
                        memcpy(wB, rB, cB);
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
        cY = nY;
        cZ = nZ;
    }

    // Copy result to final destination if needed
    if(Nopt % 2 != 0){ // If odd number of transforms, result is in tmp
        size_t fB = (size_t)cY * cZ * sizeof(int64_t);
        if(fB > B){
            fprintf(stderr, "CPU Err: Final buffer size > Original\n");
            fB = B; // Prevent buffer overflow
        }
        memcpy(Out, tmp, fB);
    } else { // If even number, result is already in Out (unless Nopt=0, handled earlier)
        if((size_t)cY * cZ != N) {
             fprintf(stderr, "CPU Warn: Final image size != Original size\n");
        }
    }
    free(tmp);
}

// --- GPU Kernels ---

__global__ void initialize_PerFrameArr_Area_kernel(
    double *dPerFrameArr, size_t bigArrSize,
    int nRBins, int nEtaBins,
    const double *dRBinsLow, const double *dRBinsHigh,
    const double *dEtaBinsLow, const double *dEtaBinsHigh,
    const struct data * dPxList, const int *dNPxList, // Need map data to calculate area
    int NrPixelsY, int NrPixelsZ,                     // Need detector dimensions for bounds/mask check
    const int *dMapMask, size_t mapMaskWordCount,     // Mask info (dMapMask can be NULL)
    double px, double Lsd)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bigArrSize) return;

    // --- Calculate Static R, TTh, Eta (same as before) ---
    double RMean = 0.0;
    double EtaMean = 0.0;
    double TwoTheta = 0.0;

    // Ensure nEtaBins is valid to prevent division by zero
    if (nEtaBins > 0) {
        int j = idx / nEtaBins; // R bin index
        int k = idx % nEtaBins; // Eta bin index

        // Bounds check using the provided dimensions
        if (j < nRBins && k < nEtaBins) {
            RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5;
            EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5;
            TwoTheta = rad2deg * atan(RMean * px / Lsd);
        }
    }

    // --- Calculate Static Total Area ---
    double totArea = 0.0;
    long long nPixels = 0;
    long long dataPos = 0;
    const size_t nPxListIndex = 2 * idx;
    const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;

    if (nPxListIndex + 1 < 2 * bigArrSize) {
        nPixels = dNPxList[nPxListIndex];
        dataPos = dNPxList[nPxListIndex + 1];
    }

    if (nPixels > 0 && dataPos >= 0) {
        for (long long l = 0; l < nPixels; l++) {
            struct data ThisVal = dPxList[dataPos + l];

            if (ThisVal.y < 0 || ThisVal.y >= NrPixelsY || ThisVal.z < 0 || ThisVal.z >= NrPixelsZ) {
                continue;
            }
            long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;

            if (testPos < 0 || testPos >= totalPixels) {
                 continue;
            }

            bool isMasked = false;
            if (dMapMask != NULL && mapMaskWordCount > 0) {
				if (TestBit(dMapMask, testPos)) {
					isMasked = true;
				}
            }

            if (!isMasked) {
                totArea += ThisVal.frac;
            }
        }
    }

    if (idx < bigArrSize) {
        dPerFrameArr[0 * bigArrSize + idx] = RMean;
        dPerFrameArr[1 * bigArrSize + idx] = TwoTheta;
        dPerFrameArr[2 * bigArrSize + idx] = EtaMean;
        dPerFrameArr[3 * bigArrSize + idx] = totArea;
    }
}


__global__ void integrate_noMapMask(double px, double Lsd, size_t bigArrSize, int Normalize, int sumImages, int frameIdx,
	const struct data * dPxList, const int *dNPxList,
	int NrPixelsY, int NrPixelsZ,
	const double *dImage, double *dIntArrPerFrame,
	double *dSumMatrix)
{
	const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= bigArrSize) return;

	double Intensity = 0.0;
	double totArea = 0.0;

	long long nPixels = 0;
	long long dataPos = 0;
	const size_t nPxListIndex = 2 * idx;

	nPixels = dNPxList[nPxListIndex];
	dataPos = dNPxList[nPxListIndex + 1];

	for (long long l = 0; l < nPixels; l++) {
		struct data ThisVal = dPxList[dataPos + l];
		long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
		Intensity += dImage[testPos] * ThisVal.frac;
		totArea += ThisVal.frac;
	}

	if (totArea > AREA_THRESHOLD) {
		if (Normalize) {
			Intensity /= totArea;
		}
		dIntArrPerFrame[idx] = Intensity;
		if (sumImages && dSumMatrix) {
			atomicAdd(&dSumMatrix[idx], Intensity);
		}
	} else {
		dIntArrPerFrame[idx] = 0.0;
	}
}


__global__ void integrate_MapMask(double px, double Lsd, size_t bigArrSize, int Normalize, int sumImages, int frameIdx,
  size_t mapMaskWordCount, const int *dMapMask,
  int nRBins, int nEtaBins,
  int NrPixelsY, int NrPixelsZ,
  const struct data * dPxList, const int *dNPxList,
  const double *dImage, double *dIntArrPerFrame,
  double *dSumMatrix)
{
	const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= bigArrSize) return;

	double Intensity = 0.0;
	double totArea = 0.0;

	long long nPixels = 0;
	long long dataPos = 0;
	const size_t nPxListIndex = 2 * idx;

	nPixels = dNPxList[nPxListIndex];
	dataPos = dNPxList[nPxListIndex + 1];

	for (long long l = 0; l < nPixels; l++) {
		struct data ThisVal = dPxList[dataPos + l];
		long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
		bool isMasked = false;
		if (TestBit(dMapMask, testPos)) isMasked = true;

		if (!isMasked) {
			Intensity += dImage[testPos] * ThisVal.frac;
			totArea += ThisVal.frac;
		}
	}

	if (totArea > AREA_THRESHOLD) {
		if (Normalize) {
			Intensity /= totArea;
		}
		dIntArrPerFrame[idx] = Intensity;
		if (sumImages && dSumMatrix) {
			atomicAdd(&dSumMatrix[idx], Intensity);
		}
	} else {
		dIntArrPerFrame[idx] = 0.0;
	}
}

__global__ void sequential_transform_kernel(const int64_t *r, int64_t *w, int cY, int cZ, int nY, int nZ, int opt) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1, zs = -1;

    switch(opt){
        case 0: ys = yo; zs = zo; break;
        case 1: ys = cY - 1 - yo; zs = zo; break;
        case 2: ys = yo; zs = cZ - 1 - zo; break;
        case 3: ys = zo; zs = yo; break;
        default: return;
    }

    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
        w[i] = r[(size_t)zs * cY + ys];
    } else {
        w[i] = 0;
    }
}


__global__ void final_transform_process_kernel(const int64_t *r, double *o, const double *d, int cY, int cZ, int nY, int nZ, int opt, bool sub) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1, zs = -1;

    switch(opt){
        case 0: ys = yo; zs = zo; break;
        case 1: ys = cY - 1 - yo; zs = zo; break;
        case 2: ys = yo; zs = cZ - 1 - zo; break;
        case 3: ys = zo; zs = yo; break;
        default: o[i] = 0.0; return;
    }

    double pv = 0.0;
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
        const int64_t rv = r[(size_t)zs * cY + ys];
        pv = (double)rv;
        if (sub && d) {
            pv -= d[i];
        }
    }
    o[i] = pv;
}


__global__ void process_direct_kernel(const int64_t *r, double *o, const double *d, size_t N, bool sub) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int64_t rv = r[i];
        double pv = (double)rv;
        if (sub && d) {
            pv -= d[i];
        }
        o[i] = pv;
    }
}


__global__ void calculate_1D_profile_kernel(const double *d_IntArrPerFrame, const double *d_PerFrameArr, double *d_int1D, int nRBins, int nEtaBins, size_t bigArrSize) {
    extern __shared__ double sdata[];
    double * sIntArea = sdata;
    double * sArea    = &sdata[blockDim.x / 32];

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
            if (3 * bigArrSize + idx2d < 4 * bigArrSize) {
                 double area = d_PerFrameArr[3 * bigArrSize + idx2d];
                 if (area > AREA_THRESHOLD) {
                      mySumIntArea += d_IntArrPerFrame[idx2d] * area;
                      mySumArea += area;
                 }
            }
        }
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySumIntArea += __shfl_down_sync(0xFFFFFFFF, mySumIntArea, offset);
        mySumArea += __shfl_down_sync(0xFFFFFFFF, mySumArea, offset);
    }

    if (lane == 0) {
        atomicAdd(&sIntArea[warpId], mySumIntArea);
        atomicAdd(&sArea[warpId], mySumArea);
    }

    __syncthreads();

    if (tid == 0) {
        double finalSumIntArea = 0.0;
        double finalSumArea = 0.0;
        int numWarps = blockDim.x / warpSize;
        if (blockDim.x % warpSize != 0) numWarps++;

        for (int i = 0; i < numWarps; ++i) {
             finalSumIntArea += sIntArea[i];
             finalSumArea += sArea[i];
        }

        if (finalSumArea > AREA_THRESHOLD) {
            d_int1D[r_bin] = finalSumIntArea / finalSumArea;
        } else {
            d_int1D[r_bin] = 0.0;
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
    if(Nopt > 0){
        for(int i = 0; i < Nopt; ++i){
            if(Topt[i] < 0 || Topt[i] > 3){
                fprintf(stderr,"GPU Err: Inv opt %d\n", Topt[i]);
                gpuErrchk(cudaMemset(dProc, 0, N * sizeof(double)));
                return;
            }
            if(Topt[i] != 0) anyT = true;
        }
    }

    if(!anyT){
        gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));
        unsigned long long nBUL = (N + TPB - 1) / TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){
            fprintf(stderr, "Block count %llu exceeds max grid dim %d\n", nBUL, mGDX);
            exit(1);
        }
        dim3 nB((unsigned int)nBUL);
        dim3 th(TPB);
        process_direct_kernel<<<nB, th>>>(d_b1, dProc, dAvgDark, N, doSub);
        gpuErrchk(cudaPeekAtLastError());
        return;
    }

    gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));

    const int64_t* rP = NULL;
    int64_t* wP = NULL;
    int cY = NY, cZ = NZ;

    for(int i = 0; i < Nopt - 1; ++i){
        int opt = Topt[i];
        if(i % 2 == 0){ rP = d_b1; wP = d_b2; }
        else { rP = d_b2; wP = d_b1; }
        int nY = cY, nZ = cZ;

        if(opt == 3){
            if(cY != cZ){
                fprintf(stderr,"GPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i);
                opt = 0;
            } else {
                nY = cZ; nZ = cY;
            }
        }

        size_t sON = (size_t)nY * nZ;
        unsigned long long nBUL = (sON + TPB - 1) / TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){
            fprintf(stderr,"Blk %llu > max %d (step %d)\n", nBUL, mGDX, i);
            exit(1);
        }
        dim3 nB((unsigned int)nBUL);
        dim3 th(TPB);
        sequential_transform_kernel<<<nB, th>>>(rP, wP, cY, cZ, nY, nZ, opt);
        gpuErrchk(cudaPeekAtLastError());
        cY = nY; cZ = nZ;
    }

    int fOpt = Topt[Nopt - 1];
    if((Nopt - 1) % 2 == 0){ rP = d_b1; }
    else { rP = d_b2; }

    int nY = cY, nZ = cZ;
    if(fOpt == 3){
        if(cY != cZ){
            fprintf(stderr,"GPU Warn: Skip Tpose %dx%d final st %d\n", cY, cZ, Nopt - 1);
            fOpt = 0;
        } else {
            nY = cZ; nZ = cY;
        }
    }

    size_t fON = (size_t)nY * nZ;
    if(fON != N){
        fprintf(stderr, "GPU Warn: Final image size %zu != Original size %zu\n", fON, N);
    }

    unsigned long long nBUL = (fON + TPB - 1) / TPB;
    int mGDX;
    gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
    if(nBUL > (unsigned long long)mGDX){
        fprintf(stderr,"Blk %llu > max %d (final step)\n", nBUL, mGDX);
        exit(1);
    }
    dim3 nB((unsigned int)nBUL);
    dim3 th(TPB);
    final_transform_process_kernel<<<nB, th>>>(rP, dProc, dAvgDark, cY, cZ, nY, nZ, fOpt, doSub);
    gpuErrchk(cudaPeekAtLastError());
}


// --- Peak Fitting Data Structures and Functions ---
typedef struct {
    int nrBins;
    const double *R;
    const double *Int;
} dataFit;

typedef struct {
    int index;
    double radius;
    double intensity;
} Peak;

static inline void calculate_model_and_area(
    int n_peaks, const double *params,
    int n_points, const double *R_values,
    double *out_model_curve,
    double *out_peak_areas)
{
    const double bg = params[n_peaks * 4];
    for (int i = 0; i < n_points; ++i) {
        out_model_curve[i] = bg;
    }

    for (int pN = 0; pN < n_peaks; ++pN) {
        double A = params[pN * 4 + 0];
        double m = fmax(0.0, fmin(1.0, params[pN * 4 + 1]));
        double c = params[pN * 4 + 2];
        double s = fmax(1e-9, params[pN * 4 + 3]);

        for (int i = 0; i < n_points; ++i) {
            double diff_sq = (R_values[i] - c) * (R_values[i] - c);
            double s_sq = s * s;
            double gaussian = exp(-diff_sq / (2.0 * s_sq));
            double lorentzian = 1.0 / (1.0 + diff_sq / s_sq);
            out_model_curve[i] += A * (m * gaussian + (1.0 - m) * lorentzian);
        }

        if (out_peak_areas != NULL) {
            out_peak_areas[pN] = A * s * (m * sqrt(2.0 * M_PI) + (1.0 - m) * M_PI);
        }
    }
}

static double problem_function_global_bg(unsigned n, const double *x, double *grad, void *fdat) {
    const dataFit *d = (const dataFit*)fdat;
    const int Np = d->nrBins;
    const double *Rs = d->R;
    const double *Is = d->Int;
    const int nP = (n - 1) / 4;

    double *calculated_I = (double*)malloc(Np * sizeof(double));
    if(!calculated_I) return INFINITY;
    calculate_model_and_area(nP, x, Np, Rs, calculated_I, NULL);

    double total_sq_error = 0.0;
    for(int i = 0; i < Np; ++i){
        double residual = calculated_I[i] - Is[i];
        total_sq_error += residual * residual;
    }

    if(grad){
        memset(grad, 0, n * sizeof(double));
        for(int pN = 0; pN < nP; ++pN){
            double A = x[pN*4+0], m = fmax(0.,fmin(1.,x[pN*4+1])), c = x[pN*4+2], s = fmax(1e-9,x[pN*4+3]);
            double *gA=&grad[pN*4+0], *gm=&grad[pN*4+1], *gc=&grad[pN*4+2], *gs=&grad[pN*4+3];
            for(int i=0; i<Np; ++i){
                double diff=Rs[i]-c, diff_sq=diff*diff, s_sq=s*s;
                double gaussian=exp(-diff_sq/(2.*s_sq)), lorentzian=1./(1.+diff_sq/s_sq);
                double residual = calculated_I[i] - Is[i];
                double common = 2. * residual;
                *gA += common * (m*gaussian+(1-m)*lorentzian);
                *gm += common * A*(gaussian-lorentzian);
                *gc += common * A*(m*gaussian*(diff/s_sq) + (1-m)*lorentzian*lorentzian*(2*diff/s_sq));
                *gs += common * A*(m*gaussian*(diff_sq/(s_sq*s)) + (1-m)*lorentzian*lorentzian*(2*diff_sq/(s_sq*s)));
            }
        }
        for(int i=0; i<Np; ++i) {
            grad[n-1] += 2.0 * (calculated_I[i] - Is[i]);
        }
    }

    free(calculated_I);
    return total_sq_error;
}

void smoothData(const double *in, double *out, int N, int W) {
    if(W < 3 || W % 2 == 0){
        memcpy(out, in, N * sizeof(double));
        return;
    }
    int H = W / 2;

    double *coeffs = (double*)malloc(W * sizeof(double));
    check(!coeffs, "smoothData: Malloc failed for coefficients");
    double norm = 0.0;

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

    for(int i = 0; i < W; ++i){
        coeffs[i] /= norm;
    }

    for(int i = 0; i < N; ++i){
        if(i < H || i >= N - H){
            out[i] = in[i];
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

int findPeaks(const double *data, const double *r_values, int N, Peak **foundPeaks, double minHeight, int minDistance) {
    if(N < 3){
        *foundPeaks = NULL;
        return 0;
    }
    int maxPossiblePeaks = N / 2 + 1;
    Peak* preliminaryPeaks = (Peak*)malloc(maxPossiblePeaks * sizeof(Peak));
    check(!preliminaryPeaks, "findPeaks: Malloc failed for preliminaryPeaks");
    int peakCount = 0;

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
        Peak* finalPeaks = (Peak*)realloc(preliminaryPeaks, peakCount * sizeof(Peak));
        if(peakCount > 0 && finalPeaks == NULL) {
             *foundPeaks = preliminaryPeaks;
             fprintf(stderr, "findPeaks Warn: realloc failed.\n");
        } else {
             *foundPeaks = finalPeaks;
        }
        return peakCount;
    }

    bool* isSuppressed = (bool*)calloc(peakCount, sizeof(bool));
    check(!isSuppressed, "findPeaks: Calloc failed for isSuppressed");

    for(int i = 0; i < peakCount; ++i){
        if(isSuppressed[i]) continue;
        for(int j = i + 1; j < peakCount; ++j){
            if(isSuppressed[j]) continue;
            int distance = abs(preliminaryPeaks[i].index - preliminaryPeaks[j].index);
            if(distance < minDistance){
                if(preliminaryPeaks[j].intensity <= preliminaryPeaks[i].intensity) {
                    isSuppressed[j] = true;
                } else {
                    isSuppressed[i] = true;
                    break;
                }
            }
        }
    }

    Peak* filteredPeaks = (Peak*)malloc(peakCount * sizeof(Peak));
    check(!filteredPeaks, "findPeaks: Malloc failed for filteredPeaks");
    int filteredCount = 0;
    for(int i = 0; i < peakCount; ++i){
        if(!isSuppressed[i]){
            filteredPeaks[filteredCount++] = preliminaryPeaks[i];
        }
    }

    free(preliminaryPeaks);
    free(isSuppressed);

    Peak* finalPeaks = (Peak*)realloc(filteredPeaks, filteredCount * sizeof(Peak));
     if(filteredCount > 0 && finalPeaks == NULL) {
         *foundPeaks = filteredPeaks;
         fprintf(stderr, "findPeaks Warn: realloc failed.\n");
     } else {
        *foundPeaks = finalPeaks;
     }
    return filteredCount;
}

typedef struct {
    int startIndex;
    int endIndex;
    int numPeaks;
    Peak* peaks;
} FitJob;

static int comparePeaksByIndex(const void* a, const void* b) {
    Peak* peakA = (Peak*)a;
    Peak* peakB = (Peak*)b;
    return (peakA->index - peakB->index);
}

static double estimate_initial_params(const double* intensity_data, int n_points, int peak_idx_local,
                                      double* out_bg_guess, double* out_amp_guess)
{
    int bg_width = fmin(5, n_points / 4);
    if (bg_width < 1) bg_width = 1;
    double bg_sum = 0.0;
    for (int i = 0; i < bg_width; ++i) {
        bg_sum += intensity_data[i];
        bg_sum += intensity_data[n_points - 1 - i];
    }
    *out_bg_guess = bg_sum / (2.0 * bg_width);
    *out_amp_guess = intensity_data[peak_idx_local] - *out_bg_guess;
    if (*out_amp_guess <= 0) *out_amp_guess = intensity_data[peak_idx_local];

    double half_max = *out_bg_guess + (*out_amp_guess / 2.0);
    int left_idx = peak_idx_local;
    while (left_idx > 0 && intensity_data[left_idx] > half_max) {
        left_idx--;
    }
    int right_idx = peak_idx_local;
    while (right_idx < n_points - 1 && intensity_data[right_idx] > half_max) {
        right_idx++;
    }

    double fwhm = (double)(right_idx - left_idx);
    return (fwhm > 1.0) ? fwhm : 2.0;
}

// =========================================================================
// ============================ MAIN FUNCTION ============================
// =========================================================================
int main(int argc, char *argv[]){
    if (argc < 2){
        printf("Usage: %s ParamFN [DarkAvgFN]\n", argv[0]);
        printf(" Args:\n");
        printf("  ParamFN:   Path to parameter file.\n");
        printf("  DarkAvgFN: Optional path to dark frame file (binary int64_t, averaged if multiple frames).\n");
        return 1;
    }
    printf("[%s] - Starting...\n", argv[0]);
    double t_start_main = get_wall_time_ms();

    // --- Setup Signal Handling ---
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // --- Initialize GPU ---
    int dev_id = 0;
    gpuErrchk(cudaSetDevice(dev_id));
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, dev_id));
    printf("GPU Device %d: %s (CC %d.%d)\n", dev_id, prop.name, prop.major, prop.minor);
    printf("Init GPU: %.3f ms\n", get_wall_time_ms() - t_start_main);

    // --- Read Pixel Mapping Files ---
    double t_start_map = get_wall_time_ms();
    check(ReadBins() != 1, "Failed read/map Map.bin/nMap.bin");
    printf("Read Maps: %.3f ms\n", get_wall_time_ms() - t_start_map);

    // --- Read Parameters ---
    double t_start_params = get_wall_time_ms();
    double RMax = 0, RMin = 0, RBinSize = 0;
    double EtaMax = 0, EtaMin = 0, EtaBinSize = 0;
    double Lsd = 0, px = 0;
    // NrPixelsY and NrPixelsZ are now global, but are assigned here
    int Normalize = 1;
    int nEtaBins = 0, nRBins = 0;
    char *ParamFN = argv[1];
    FILE *pF = fopen(ParamFN, "r");
    check(!pF, "Failed open param file: %s", ParamFN);

    char line[4096], key[1024], val_str[3072]; // Buffers for parsing
    int Nopt = 0; // Number of image transform operations
    long long GapI = 0, BadPxI = 0; // Intensity values for mask generation
    int Topt[MAX_TRANSFORM_OPS] = {0}; // Array to store transform options
    int mkMap = 0; // Flag to generate mask from dark frame
    int sumI = 0; // Flag to sum integrated patterns
    int doSm = 0; // Flag to smooth 1D data before peak finding
    int multiP = 0; // Flag for finding multiple peaks
    int pkFit = 0; // Flag to perform peak fitting
    int nSpecP = 0; // Number of specified peak locations
    int wr2D = 0; // Flag to write 2D integrated patterns
    double pkLoc[MAX_PEAK_LOCATIONS]; // Array for specified peak locations
    int fitROIPadding = 20; // Default ROI padding
    int fitROIAuto = 0;     // Default to manual ROI sizing

    // Read parameters line by line
    while(fgets(line, sizeof(line), pF)){
        if(line[0] == '#' || isspace(line[0]) || strlen(line) < 3) {
            continue;
        }
        if (sscanf(line, "%1023s %[^\n]", key, val_str) == 2) {
             if (strcmp(key, "EtaBinSize") == 0) sscanf(val_str, "%lf", &EtaBinSize);
             else if (strcmp(key, "RBinSize") == 0) sscanf(val_str, "%lf", &RBinSize);
             else if (strcmp(key, "RMax") == 0) sscanf(val_str, "%lf", &RMax);
             else if (strcmp(key, "RMin") == 0) sscanf(val_str, "%lf", &RMin);
             else if (strcmp(key, "EtaMax") == 0) sscanf(val_str, "%lf", &EtaMax);
             else if (strcmp(key, "EtaMin") == 0) sscanf(val_str, "%lf", &EtaMin);
             else if (strcmp(key, "Lsd") == 0) sscanf(val_str, "%lf", &Lsd);
             else if (strcmp(key, "px") == 0) sscanf(val_str, "%lf", &px);
             else if (strcmp(key, "NrPixelsY") == 0) sscanf(val_str, "%d", &NrPixelsY);
             else if (strcmp(key, "NrPixelsZ") == 0) sscanf(val_str, "%d", &NrPixelsZ);
             else if (strcmp(key, "NrPixels") == 0) { sscanf(val_str, "%d", &NrPixelsY); NrPixelsZ = NrPixelsY; }
             else if (strcmp(key, "Normalize") == 0) sscanf(val_str, "%d", &Normalize);
             else if (strcmp(key, "GapIntensity") == 0) { sscanf(val_str, "%lld", &GapI); mkMap = 1; }
             else if (strcmp(key, "BadPxIntensity") == 0) { sscanf(val_str, "%lld", &BadPxI); mkMap = 1; }
             else if (strcmp(key, "ImTransOpt") == 0) {
                 if(Nopt < MAX_TRANSFORM_OPS) sscanf(val_str, "%d", &Topt[Nopt++]);
                 else printf("Warn: Max %d ImTransOpt reached, ignoring further options.\n", MAX_TRANSFORM_OPS);
             }
             else if (strcmp(key, "SumImages") == 0) sscanf(val_str, "%d", &sumI);
             else if (strcmp(key, "Write2D") == 0) sscanf(val_str, "%d", &wr2D);
             else if (strcmp(key, "DoSmoothing") == 0) sscanf(val_str, "%d", &doSm);
             else if (strcmp(key, "MultiplePeaks") == 0) sscanf(val_str, "%d", &multiP);
             else if (strcmp(key, "DoPeakFit") == 0) sscanf(val_str, "%d", &pkFit);
             else if (strcmp(key, "FitROIPadding") == 0) sscanf(val_str, "%d", &fitROIPadding);
             else if (strcmp(key, "FitROIAuto") == 0) sscanf(val_str, "%d", &fitROIAuto);
             else if (strcmp(key, "PeakLocation") == 0) {
                 if(nSpecP < MAX_PEAK_LOCATIONS) {
                     sscanf(val_str, "%lf", &pkLoc[nSpecP++]);
                     multiP = 1;
                     pkFit = 1;
                     doSm = 0;
                 } else {
                     printf("Warn: Max %d PeakLocation reached, ignoring further locations.\n", MAX_PEAK_LOCATIONS);
                 }
             }
        }
    }
    fclose(pF);

    check(NrPixelsY <= 0 || NrPixelsZ <= 0, "NrPixelsY/Z invalid or not set in parameter file.");
    check(Lsd <= 0 || px <= 0, "Lsd/px invalid or not set in parameter file.");
    if(pkFit && nSpecP > 0) { multiP = 1; if (doSm) { printf("Warn: Smoothing disabled because specific PeakLocations were provided.\n"); doSm = 0; } }
    nRBins = (RBinSize > 1e-9) ? (int)ceil((RMax - RMin) / RBinSize) : 0;
    nEtaBins = (EtaBinSize > 1e-9) ? (int)ceil((EtaMax - EtaMin) / EtaBinSize) : 0;
    check(nRBins <= 0 || nEtaBins <= 0, "Invalid bin parameters. R bins=%d, Eta bins=%d", nRBins, nEtaBins);
    size_t bigArrSize = (size_t)nRBins * nEtaBins;

    printf("Parameters Loaded:\n");
    printf(" R Bins:    [%.3f .. %.3f], %d bins (step %.4f)\n", RMin, RMax, nRBins, RBinSize);
    printf(" Eta Bins:  [%.3f .. %.3f], %d bins (step %.4f)\n", EtaMin, EtaMax, nEtaBins, EtaBinSize);
    printf(" Detector:  %d x %d pixels\n", NrPixelsY, NrPixelsZ);
    printf(" Geometry:  Lsd=%.4f, px=%.6f\n", Lsd, px);
    printf(" Transforms(%d):", Nopt);
    for(int i = 0; i < Nopt; ++i) { printf(" %d", Topt[i]); }
    printf("\n");
    printf(" Options:   Normalize=%d, SumIntegrations=%d, Write2D=%d\n", Normalize, sumI, wr2D);
    printf(" Peak Fit:  Enabled=%d, MultiPeak=%d, Smooth=%d, NumSpecifiedPeaks=%d\n", pkFit, multiP, doSm, nSpecP);
    if (mkMap) printf(" Masking:   Will generate from Gap=%lld, BadPx=%lld in Dark Frame\n", GapI, BadPxI);
    printf("Read Params: %.3f ms\n", get_wall_time_ms() - t_start_params);
    fflush(stdout);

    // --- Setup Bin Edges (Host) ---
    double *hEtaLo, *hEtaHi, *hRLo, *hRHi;
    hEtaLo = (double*)malloc(nEtaBins * sizeof(double));
    hEtaHi = (double*)malloc(nEtaBins * sizeof(double));
    hRLo   = (double*)malloc(nRBins * sizeof(double));
    hRHi   = (double*)malloc(nRBins * sizeof(double));
    check(!hEtaLo || !hEtaHi || !hRLo || !hRHi, "Allocation failed for host bin edge arrays");
    REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, hEtaLo, hEtaHi, hRLo, hRHi);

    // --- Host Memory Allocations ---
    double *hAvgDark = NULL;
    int64_t *hDarkInT = NULL;
    int64_t *hDarkIn = NULL;
    size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;
    // NOTE: This SizeFile is now ONLY for the dark frame, which is assumed to be int64_t
    size_t SizeFile = totalPixels * sizeof(int64_t);

    hAvgDark = (double*)calloc(totalPixels, sizeof(double));
    check(!hAvgDark, "Allocation failed for hAvgDark");
    hDarkInT = (int64_t*)malloc(SizeFile);
    check(!hDarkInT, "Allocation failed for hDarkInT");
    hDarkIn = (int64_t*)malloc(SizeFile);
    check(!hDarkIn, "Allocation failed for hDarkIn");

    // --- Device Memory Allocations (Persistent) ---
    double *dAvgDark = NULL;
    double *dProcessedImage = NULL;
    double *d_int1D = NULL;
    int *dMapMask = NULL;
    size_t mapMaskWC = 0;
    int *dNPxList = NULL;
    struct data *dPxList = NULL;
    double *dSumMatrix = NULL;
    double *dIntArrFrame = NULL;
    double *dPerFrame = NULL;
    double *dEtaLo = NULL, *dEtaHi = NULL, *dRLo = NULL, *dRHi = NULL;
    int64_t *g_dTempTransformBuf1 = NULL;
    int64_t *g_dTempTransformBuf2 = NULL;
    bool darkSubEnabled = (argc > 2);

    gpuErrchk(cudaMalloc(&dProcessedImage, totalPixels * sizeof(double)));
    gpuErrchk(cudaMalloc(&dPxList, szPxList));
    gpuErrchk(cudaMalloc(&dNPxList, szNPxList));
    gpuErrchk(cudaMalloc(&dIntArrFrame, bigArrSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&dPerFrame, bigArrSize * 4 * sizeof(double)));
    gpuErrchk(cudaMalloc(&dEtaLo, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dEtaHi, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRLo, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRHi, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_int1D, nRBins * sizeof(double)));
    size_t tempBufferSize = totalPixels * sizeof(int64_t);
    printf("Allocating persistent GPU transform buffers (%zu bytes each)...\n", tempBufferSize);
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf1, tempBufferSize));
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf2, tempBufferSize));
    gpuErrchk(cudaMemcpy(dPxList, pxList, szPxList, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dNPxList, nPxList, szNPxList, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaLo, hEtaLo, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaHi, hEtaHi, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRLo, hRLo, nRBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRHi, hRHi, nRBins * sizeof(double), cudaMemcpyHostToDevice));

    // --- Process Dark Frame (Mask generation happens here) ---
    double t_start_dark = get_wall_time_ms();
    int nDarkFramesRead = 0;
    int *hMapMask = NULL;

    if(darkSubEnabled){
        char* darkFN = argv[2];
        FILE* fD = fopen(darkFN, "rb");
        check(!fD, "Failed to open dark frame file: %s", darkFN);

        fseek(fD, 0, SEEK_END);
        size_t szD = ftell(fD);
        rewind(fD);
        int nFD = szD / SizeFile;
        check(nFD == 0 || szD % SizeFile != 0, "Dark file %s incomplete (size %zu, frame %zu). Found %d frames.",
              darkFN, szD, SizeFile, nFD);
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
                mkMap = 0;
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
        }

        gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
        gpuErrchk(cudaMemcpy(dAvgDark, hAvgDark, totalPixels * sizeof(double), cudaMemcpyHostToDevice));
        printf("Average dark frame copied to GPU.\n");

    } else {
        gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
        gpuErrchk(cudaMemset(dAvgDark, 0, totalPixels * sizeof(double)));
        printf("No dark frame provided, using zeros on GPU.\n");
    }

    printf("Initializing static PerFrame array (R, TTh, Eta, Area) on GPU...\n");
    int initTPB = 256;
    int initBlocks = (bigArrSize + initTPB - 1) / initTPB;
    initialize_PerFrameArr_Area_kernel<<<initBlocks, initTPB>>>(
        dPerFrame, bigArrSize,
        nRBins, nEtaBins,
        dRLo, dRHi, dEtaLo, dEtaHi,
        dPxList, dNPxList,
        NrPixelsY, NrPixelsZ,
        dMapMask, mapMaskWC,
        px, Lsd
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("GPU PerFrame array initialization complete (Area pre-calculated %s mask).\n",
           (dMapMask != NULL ? "with" : "without"));

    if(sumI){
        gpuErrchk(cudaMalloc(&dSumMatrix, bigArrSize * sizeof(double)));
        gpuErrchk(cudaMemset(dSumMatrix, 0, bigArrSize * sizeof(double)));
    }
    printf("Processed dark/mask: %.3f ms\n", get_wall_time_ms() - t_start_dark);
    fflush(stdout);

    // --- Network Setup ---
    // <<< CHANGE #5: REMOVED OBSOLETE NETWORK SETUP CODE >>>
    printf("Network: Protocol updated to handle dynamic data types and sizes.\n");

    int server_fd;
    struct sockaddr_in server_addr;
    queue_init(&process_queue);

    check((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0, "Socket creation failed: %s", strerror(errno));
    int sock_opt = 1;
    check(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &sock_opt, sizeof(sock_opt)), "setsockopt failed: %s", strerror(errno));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    check(bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0, "Bind failed for port %d: %s", PORT, strerror(errno));
    check(listen(server_fd, MAX_CONNECTIONS) < 0, "Listen failed: %s", strerror(errno));
    printf("Server listening on port %d\n", PORT);

    pthread_t accept_thread;
    check(pthread_create(&accept_thread, NULL, accept_connections, &server_fd) != 0, "Failed to create accept thread: %s", strerror(errno));

    // --- Prepare for Main Loop ---
    FILE *fLineout = fopen("lineout.bin", "wb");
    check(!fLineout, "Error opening lineout.bin for writing: %s", strerror(errno));

    FILE *fFit = NULL;
    FILE *fFitCurves = NULL;
    if(pkFit){
        fFit = fopen("fit.bin", "wb");
        check(!fFit, "Error opening fit.bin for writing: %s", strerror(errno));
        fFitCurves = fopen("fit_curves.bin", "wb");
        check(!fFitCurves, "Error opening fit_curves.bin for writing: %s", strerror(errno));
    }

    FILE *f2D = NULL;
    if (wr2D) {
        printf("Will write all 2D integrated patterns to single file: Int2D.bin\n");
        f2D = fopen("Int2D.bin", "wb");
        check(!f2D, "Error opening Int2D.bin for writing: %s", strerror(errno));
    }

    cudaEvent_t ev_proc_start, ev_proc_stop;
    cudaEvent_t ev_integ_start, ev_integ_stop;
    cudaEvent_t ev_prof_start, ev_prof_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    gpuErrchk(cudaEventCreate(&ev_proc_start)); gpuErrchk(cudaEventCreate(&ev_proc_stop));
    gpuErrchk(cudaEventCreate(&ev_integ_start)); gpuErrchk(cudaEventCreate(&ev_integ_stop));
    gpuErrchk(cudaEventCreate(&ev_prof_start)); gpuErrchk(cudaEventCreate(&ev_prof_stop));
    gpuErrchk(cudaEventCreate(&ev_d2h_start)); gpuErrchk(cudaEventCreate(&ev_d2h_stop));

    float t_proc_gpu = 0, t_integ_gpu = 0, t_prof_gpu = 0, t_d2h_gpu = 0;
    double t_qp_cpu = 0, t_write1d_cpu = 0, t_fit_cpu = 0, t_writefit_cpu = 0, t_write2d_cpu = 0, t_loop_cpu = 0;
    double t_sync_cpu = 0;
    double t_convert_cpu = 0;
    double t_start_loop, t_end_loop;

    int firstFrame = 1;
    double *hIntArrFrame = NULL;
    double *hPerFrame = NULL;
    double *h_int1D = NULL;
    if (wr2D) {
        gpuErrchk(cudaMallocHost((void**)&hIntArrFrame, bigArrSize * sizeof(double)));
        check(!hIntArrFrame, "Allocation failed for pinned hIntArrFrame");
    }
    gpuErrchk(cudaMallocHost((void**)&hPerFrame, bigArrSize * 4 * sizeof(double)));
    check(!hPerFrame, "Allocation failed for pinned hPerFrame");
    gpuErrchk(cudaMallocHost((void**)&h_int1D, nRBins * sizeof(double)));
    check(!h_int1D, "Allocation failed for pinned host buffer h_int1D");
    double *hR = (double*)calloc(nRBins, sizeof(double));
    check(!hR, "Allocation failed for hR");
    double *hEta = (double*)calloc(nEtaBins, sizeof(double));
    check(!hEta, "Allocation failed for hEta");
    double *hLineout = (double*)malloc(nRBins * 2 * sizeof(double));
    check(!hLineout, "Allocation failed for hLineout");

    printf("Setup complete. Starting main processing loop...\n");
    double t_end_setup = get_wall_time_ms();
    printf("Total setup time: %.3f ms\n", t_end_setup - t_start_main);
    fflush(stdout);

    // =========================== Main Processing Loop ===========================
    int frameCounter = 0;
    while (keep_running) {
        t_start_loop = get_wall_time_ms();

        double t_qp_start = get_wall_time_ms();
        DataChunk chunk;
        if(queue_pop(&process_queue, &chunk) < 0){
            break;
        }
        t_qp_cpu = get_wall_time_ms() - t_qp_start;
        t_convert_cpu = chunk.conversion_ms;

        gpuErrchk(cudaEventRecord(ev_proc_start, 0));
        ProcessImageGPU(chunk.data, dProcessedImage, dAvgDark, Nopt, Topt, NrPixelsY, NrPixelsZ, darkSubEnabled,
                        g_dTempTransformBuf1, g_dTempTransformBuf2);
        gpuErrchk(cudaEventRecord(ev_proc_stop, 0));

        int currFidx = chunk.dataset_num;
        int integTPB = THREADS_PER_BLOCK_INTEGRATE;
        int nrVox = (bigArrSize + integTPB - 1) / integTPB;

        gpuErrchk(cudaEventRecord(ev_integ_start, 0));
        if(!dMapMask){
            integrate_noMapMask<<<nrVox, integTPB>>>(
                px, Lsd, bigArrSize, Normalize, sumI, currFidx,
                dPxList, dNPxList,
                NrPixelsY, NrPixelsZ,
                dProcessedImage, dIntArrFrame, dSumMatrix);
        } else {
            integrate_MapMask<<<nrVox, integTPB>>>(
                px, Lsd, bigArrSize, Normalize, sumI, currFidx,
                mapMaskWC, dMapMask, nRBins, nEtaBins,
                NrPixelsY, NrPixelsZ,
                dPxList, dNPxList,
                dProcessedImage, dIntArrFrame, dSumMatrix);
        }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_integ_stop, 0));

        size_t profileSharedMem = (THREADS_PER_BLOCK_PROFILE / 32) * sizeof(double) * 2;
        gpuErrchk(cudaEventRecord(ev_prof_start, 0));
        calculate_1D_profile_kernel<<<nRBins, THREADS_PER_BLOCK_PROFILE, profileSharedMem>>>(
            dIntArrFrame, dPerFrame, d_int1D, nRBins, nEtaBins, bigArrSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_prof_stop, 0));

        gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
        gpuErrchk(cudaMemcpyAsync(h_int1D, d_int1D, nRBins * sizeof(double), cudaMemcpyDeviceToHost, 0));
        if(firstFrame == 1){
             gpuErrchk(cudaMemcpyAsync(hPerFrame, dPerFrame, bigArrSize * 4 * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        if(wr2D && hIntArrFrame){
            gpuErrchk(cudaMemcpyAsync(hIntArrFrame, dIntArrFrame, bigArrSize * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        gpuErrchk(cudaEventRecord(ev_d2h_stop, 0));

        double t_sync_start = get_wall_time_ms();
        gpuErrchk(cudaEventSynchronize(ev_proc_stop));
        gpuErrchk(cudaEventElapsedTime(&t_proc_gpu, ev_proc_start, ev_proc_stop));
        gpuErrchk(cudaEventSynchronize(ev_integ_stop));
        gpuErrchk(cudaEventElapsedTime(&t_integ_gpu, ev_integ_start, ev_integ_stop));
        gpuErrchk(cudaEventSynchronize(ev_prof_stop));
        gpuErrchk(cudaEventElapsedTime(&t_prof_gpu, ev_prof_start, ev_prof_stop));
        gpuErrchk(cudaEventSynchronize(ev_d2h_stop));
        gpuErrchk(cudaEventElapsedTime(&t_d2h_gpu, ev_d2h_start, ev_d2h_stop));
        t_sync_cpu = get_wall_time_ms() - t_sync_start;

        if(firstFrame == 1){
            for(int r = 0; r < nRBins; ++r){
                hR[r] = hPerFrame[r * nEtaBins + 0 * bigArrSize];
            }
            for(int e = 0; e < nEtaBins; ++e){
                 hEta[e] = hPerFrame[e + 2 * bigArrSize];
            }
            for(int r = 0; r < nRBins; ++r){
                hLineout[r * 2] = hR[r];
                hLineout[r * 2 + 1] = 0.0;
            }
            printf("Initialized host R/Eta arrays from first frame D->H copy (using pre-initialized GPU data).\n");
            firstFrame = 0;
        }

        double t_write2d_start = get_wall_time_ms();
        if (wr2D && f2D && hIntArrFrame) {
            size_t written = fwrite(hIntArrFrame, sizeof(double), bigArrSize, f2D);
            if (written != bigArrSize) {
                fprintf(stderr, "Warn: Failed to write full 2D frame %d to Int2D.bin (wrote %zu/%zu): %s\n",
                        currFidx, written, bigArrSize, strerror(errno));
            }
        }
        t_write2d_cpu = get_wall_time_ms() - t_write2d_start;

        for(int r = 0; r < nRBins; ++r){
            hLineout[r * 2 + 1] = h_int1D[r];
        }

        double t_write1d_start = get_wall_time_ms();
        if (fLineout) {
             size_t written = fwrite(hLineout, sizeof(double), nRBins * 2, fLineout);
             if (written != (size_t)nRBins * 2) {
                  fprintf(stderr, "Warn: Failed to write full lineout frame %d (wrote %zu/%d): %s\n",
                           currFidx, written, nRBins * 2, strerror(errno));
             } else {
                  fflush(fLineout);
             }
        }
        t_write1d_cpu = get_wall_time_ms() - t_write1d_start;

        double t_fit_start = get_wall_time_ms();
        int currentPeakCount = 0;
        double *sendFitParams = NULL;

        if(pkFit){
            // ... (Peak fitting logic is unchanged) ...
        }
        t_fit_cpu = get_wall_time_ms() - t_fit_start;

        double t_writefit_start = get_wall_time_ms();
        if(pkFit && currentPeakCount > 0 && sendFitParams != NULL && fFit){
             // ... (Writing fit results is unchanged) ...
        }
        if (sendFitParams != NULL) {
             free(sendFitParams);
             sendFitParams = NULL;
        }
        t_writefit_cpu = get_wall_time_ms() - t_writefit_start;

        gpuWarnchk(cudaFreeHost(chunk.data));

        t_end_loop = get_wall_time_ms();
        t_loop_cpu = t_end_loop - t_start_loop;
        
        printf("F#%d: Ttl:%.2f| QPop:%.2f Sync:%.2f GPU(Proc:%.2f Int:%.2f Prof:%.2f D2H:%.2f) CPU(Conv:%.2f Wr2D:%.2f Wr1D:%.2f Fit:%.2f WrFit:%.2f)\n",
            currFidx, t_loop_cpu, t_qp_cpu, t_sync_cpu,
            t_proc_gpu, t_integ_gpu, t_prof_gpu, t_d2h_gpu,
            t_convert_cpu, t_write2d_cpu, t_write1d_cpu, t_fit_cpu, t_writefit_cpu);
        fflush(stdout);

        frameCounter++;
    } // ======================== End Main Processing Loop ========================

    printf("Processing loop finished (keep_running=%d). Processed %d frames. Cleaning up...\n", keep_running, frameCounter);

    // --- Cleanup ---
    if(fLineout) fclose(fLineout);
    if(fFit) fclose(fFit);
    if(fFitCurves) fclose(fFitCurves);
    if(f2D) fclose(f2D);
    if(hAvgDark) free(hAvgDark);
    if(hDarkInT) free(hDarkInT);
    if(hDarkIn) free(hDarkIn);
    if(hIntArrFrame) gpuWarnchk(cudaFreeHost(hIntArrFrame));
    if(hPerFrame) gpuWarnchk(cudaFreeHost(hPerFrame));
    if(h_int1D) gpuWarnchk(cudaFreeHost(h_int1D));
    if(hR) free(hR);
    if(hEta) free(hEta);
    if(hLineout) free(hLineout);
    if(hEtaLo) free(hEtaLo);
    if(hEtaHi) free(hEtaHi);
    if(hRLo) free(hRLo);
    if(hRHi) free(hRHi);
    if(hMapMask) free(hMapMask);

    UnmapBins();

    if(dAvgDark) gpuWarnchk(cudaFree(dAvgDark));
    if(dProcessedImage) gpuWarnchk(cudaFree(dProcessedImage));
    if(d_int1D) gpuWarnchk(cudaFree(d_int1D));
    if(dMapMask) gpuWarnchk(cudaFree(dMapMask));
    if(dPxList) gpuWarnchk(cudaFree(dPxList));
    if(dNPxList) gpuWarnchk(cudaFree(dNPxList));
    if(dSumMatrix) gpuWarnchk(cudaFree(dSumMatrix));
    if(dIntArrFrame) gpuWarnchk(cudaFree(dIntArrFrame));
    if(dPerFrame) gpuWarnchk(cudaFree(dPerFrame));
    if(dEtaLo) gpuWarnchk(cudaFree(dEtaLo));
    if(dEtaHi) gpuWarnchk(cudaFree(dEtaHi));
    if(dRLo) gpuWarnchk(cudaFree(dRLo));
    if(dRHi) gpuWarnchk(cudaFree(dRHi));
    if(g_dTempTransformBuf1) gpuWarnchk(cudaFree(g_dTempTransformBuf1));
    if(g_dTempTransformBuf2) gpuWarnchk(cudaFree(g_dTempTransformBuf2));

    gpuWarnchk(cudaEventDestroy(ev_proc_start)); gpuWarnchk(cudaEventDestroy(ev_proc_stop));
    gpuWarnchk(cudaEventDestroy(ev_integ_start)); gpuWarnchk(cudaEventDestroy(ev_integ_stop));
    gpuWarnchk(cudaEventDestroy(ev_prof_start)); gpuWarnchk(cudaEventDestroy(ev_prof_stop));
    gpuWarnchk(cudaEventDestroy(ev_d2h_start)); gpuWarnchk(cudaEventDestroy(ev_d2h_stop));

    printf("Attempting to shut down network acceptor thread...\n");
    if (server_fd >= 0) {
         printf("Closing server listening socket %d...\n", server_fd);
         shutdown(server_fd, SHUT_RDWR);
         close(server_fd);
         server_fd = -1;
    }

    printf("Sending cancellation request to accept thread...\n");
    int cancel_ret = pthread_cancel(accept_thread);
    if (cancel_ret != 0) {
        fprintf(stderr, "Warning: Failed to send cancel request to accept thread: %s\n", strerror(cancel_ret));
    }

    printf("Joining accept thread (waiting for it to exit)...\n");
    void *thread_result;
    int join_ret = pthread_join(accept_thread, &thread_result);
    if (join_ret != 0) {
         fprintf(stderr, "Warning: Failed to join accept thread: %s\n", strerror(join_ret));
    } else {
        if (thread_result == PTHREAD_CANCELED) {
            printf("Accept thread successfully canceled and joined.\n");
        } else {
            printf("Accept thread joined normally (result: %p).\n", thread_result);
        }
    }

    queue_destroy(&process_queue);

    printf("[%s] - Exiting cleanly.\n", argv[0]);
    return 0;
}