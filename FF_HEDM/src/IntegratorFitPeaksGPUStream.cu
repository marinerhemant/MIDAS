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
   ~/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStream.cu -o bin/IntegratorFitPeaksGPUStream \
   -Xcompiler -g -arch sm_90   -gencode=arch=compute_90,code=sm_90 -I/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/include \
   -L/home/beams/S1IDUSER/opt/MIDAS/FF_HEDM/build/lib   -O3 -lnlopt -lblosc2 -lhdf5 -lhdf5_hl -lz -ldl -lm -lpthread -lzip
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
#include <blosc2.h>     // Include if blosc compression is used
#include <nlopt.h>      // For non-linear optimization

// --- Constants ---
#define SERVER_IP "127.0.0.1"
#define PORT 5000               // Port for receiving image data
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
#define AREA_THRESHOLD 1e-9     // Minimum area considered valid in integration/profiling

// Global variables (initialized in main)
size_t CHUNK_SIZE;
size_t TOTAL_MSG_SIZE;
size_t szPxList = 0;
size_t szNPxList = 0;
volatile sig_atomic_t keep_running = 1; // Flag for graceful shutdown

// --- Data Structures ---
typedef struct {
    uint16_t dataset_num;
    int64_t *data;
    size_t size;
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

// --- Socket Handling ---
void* handle_client(void *arg) {
    int client_socket = *((int*)arg);
    free(arg);
    uint8_t *buffer = (uint8_t*)malloc(TOTAL_MSG_SIZE);
    check(buffer == NULL, "handle_client: Failed alloc recv buffer");
    int bytes_read;
    printf("Client handler started for socket %d.\n", client_socket);

    while (keep_running) {
        int total_bytes_read = 0;
        while (total_bytes_read < TOTAL_MSG_SIZE) {
            bytes_read = recv(client_socket, buffer + total_bytes_read, TOTAL_MSG_SIZE - total_bytes_read, 0);
            if (bytes_read <= 0) {
                 goto connection_closed;
            }
            total_bytes_read += bytes_read;
            if (!keep_running) {
                goto connection_closed;
            }
        }
        uint16_t dataset_num;
        memcpy(&dataset_num, buffer, HEADER_SIZE);
        size_t num_pixels = CHUNK_SIZE / BYTES_PER_PIXEL;
        int64_t *data = NULL;
        gpuWarnchk(cudaMallocHost((void**)&data, num_pixels * sizeof(int64_t))); // Pinned memory
        if (!data) {
            perror("handle_client: Pinned memory alloc failed");
            break;
        }
        memcpy(data, buffer + HEADER_SIZE, CHUNK_SIZE);
        if (queue_push(&process_queue, dataset_num, data, num_pixels) < 0) {
            printf("handle_client: queue_push failed, likely shutdown. Discarding frame %d.\n", dataset_num);
            gpuWarnchk(cudaFreeHost(data));
            goto connection_closed;
        }
    }
connection_closed:
    if (bytes_read == 0 && keep_running) {
        printf("Client disconnected (socket %d)\n", client_socket);
    } else if (bytes_read < 0) {
        perror("Receive error");
    }
    free(buffer);
    close(client_socket);
    printf("Client handler finished (socket %d).\n", client_socket);
    return NULL;
}

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
        // else: Index out of range, values remain 0.0
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
    } // else: nPixels remains 0

    if (nPixels > 0 && dataPos >= 0) {
        for (long long l = 0; l < nPixels; l++) {
            struct data ThisVal = dPxList[dataPos + l];

            // Pixel bounds check
            if (ThisVal.y < 0 || ThisVal.y >= NrPixelsY || ThisVal.z < 0 || ThisVal.z >= NrPixelsZ) {
                continue;
            }
            long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;

            // Linear index bounds check (redundant but safe)
            if (testPos < 0 || testPos >= totalPixels) {
                 continue;
            }

            // --- Apply Mask if provided ---
            bool isMasked = false;
            if (dMapMask != NULL && mapMaskWordCount > 0) {
				if (TestBit(dMapMask, testPos)) {
					isMasked = true; // Masked pixel
				}
            }
            // --- End Mask Application ---

            if (!isMasked) {
                totArea += ThisVal.frac; // Accumulate area only if not masked
            }
        }
    } // else: No pixels for this bin, totArea remains 0.0

    // --- Write ALL static values to dPerFrameArr ---
    // Ensure index is valid before writing (although already checked at start)
    if (idx < bigArrSize) {
        // R values start at index 0
        dPerFrameArr[0 * bigArrSize + idx] = RMean;
        // TwoTheta values start at index bigArrSize
        dPerFrameArr[1 * bigArrSize + idx] = TwoTheta;
        // Eta values start at index 2 * bigArrSize
        dPerFrameArr[2 * bigArrSize + idx] = EtaMean;
        // Area values start at index 3 * bigArrSize
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
	const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;
	if (idx >= bigArrSize) return;

	double Intensity = 0.0;
	double totArea = 0.0; // <<< RE-INTRODUCED local area calculation

	long long nPixels = 0;
	long long dataPos = 0;
	const size_t nPxListIndex = 2 * idx;

	nPixels = dNPxList[nPxListIndex];
	dataPos = dNPxList[nPxListIndex + 1];

	for (long long l = 0; l < nPixels; l++) {
		struct data ThisVal = dPxList[dataPos + l];
		long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
		Intensity += dImage[testPos] * ThisVal.frac;
		totArea += ThisVal.frac; // <<< Accumulate area locally
	}

	// Use the *locally calculated* totArea for threshold and normalization
	if (totArea > AREA_THRESHOLD) {
		if (Normalize) {
			Intensity /= totArea; // <<< Normalize with local area
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
	const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;
	if (idx >= bigArrSize) return;

	double Intensity = 0.0;
	double totArea = 0.0; // <<< RE-INTRODUCED local area calculation

	long long nPixels = 0;
	long long dataPos = 0;
	const size_t nPxListIndex = 2 * idx;

	nPixels = dNPxList[nPxListIndex];
	dataPos = dNPxList[nPxListIndex + 1];

	// <<< RE-INTRODUCED area and intensity calculation loop (with mask) >>>
	for (long long l = 0; l < nPixels; l++) {
		struct data ThisVal = dPxList[dataPos + l];
		long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
		bool isMasked = false;
		if (TestBit(dMapMask, testPos)) isMasked = true;

		if (!isMasked) {
			Intensity += dImage[testPos] * ThisVal.frac;
			totArea += ThisVal.frac; // <<< Accumulate area locally (only for non-masked)
		}
	}

	// Use the *locally calculated* totArea for threshold and normalization
	if (totArea > AREA_THRESHOLD) {
		if (Normalize) {
			Intensity /= totArea; // <<< Normalize with local area
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
    int ys = -1;
    int zs = -1;

    switch(opt){
        case 0: ys = yo; zs = zo; break; // No-op
        case 1: ys = cY - 1 - yo; zs = zo; break; // Flip Horizontal
        case 2: ys = yo; zs = cZ - 1 - zo; break; // Flip Vertical
        case 3: ys = zo; zs = yo; break; // Transpose (only valid if cY=cZ=nY=nZ handled by host)
        default: return; // Invalid option
    }

    // Read from source location (ys, zs) in input buffer 'r' (dimensions cY, cZ)
    // Write to target location (yo, zo) in output buffer 'w' (dimensions nY, nZ)
    if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
        w[i] = r[(size_t)zs * cY + ys];
    } else {
        // Handle out-of-bounds reads if necessary (e.g., padding)
        w[i] = 0; // Default: write 0 if source is out of bounds
    }
}


__global__ void final_transform_process_kernel(const int64_t *r, double *o, const double *d, int cY, int cZ, int nY, int nZ, int opt, bool sub) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int yo = i % nY;
    const int zo = i / nY;
    int ys = -1;
    int zs = -1;

    switch(opt){
        case 0: ys = yo; zs = zo; break; // No-op
        case 1: ys = cY - 1 - yo; zs = zo; break; // Flip Horizontal
        case 2: ys = yo; zs = cZ - 1 - zo; break; // Flip Vertical
        case 3: ys = zo; zs = yo; break; // Transpose
        default: o[i] = 0.0; return; // Invalid option
    }

    double pv = 0.0;
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


__global__ void calculate_1D_profile_kernel(const double *d_IntArrPerFrame, const double *d_PerFrameArr, double *d_int1D, int nRBins, int nEtaBins, size_t bigArrSize) {
    // Shared memory for reduction within a block (one warp processes multiple Eta bins)
    extern __shared__ double sdata[]; // Expects size >= (blockDim.x / warpSize) * 2
    double * sIntArea = sdata; // Buffer for sum(Intensity * Area) per warp
    double * sArea    = &sdata[blockDim.x / 32]; // Buffer for sum(Area) per warp

    const int r_bin = blockIdx.x; // Each block processes one R bin
    if (r_bin >= nRBins) {
        return; // Block out of range
    }

    const int tid = threadIdx.x;
    const int warpSize = 32;
    const int lane = tid % warpSize; // Lane index within the warp (0-31)
    const int warpId = tid / warpSize; // Warp index within the block

    // Initialize shared memory for this warp (only first thread in warp needs to do it)
    if (lane == 0) {
         sIntArea[warpId] = 0.0;
         sArea[warpId] = 0.0;
    }
    // __syncthreads(); // Sync within block needed if initialization isn't guaranteed before use (warp execution guarantees this for lane 0)

    // Each thread processes a subset of Eta bins for the current R bin
    double mySumIntArea = 0.0;
    double mySumArea = 0.0;
    for (int eta_bin = tid; eta_bin < nEtaBins; eta_bin += blockDim.x) {
        size_t idx2d = (size_t)r_bin * nEtaBins + eta_bin;
        // Bounds check for safety, though loop condition should prevent overshoot if bigArrSize is correct
        if (idx2d < bigArrSize) {
            // Access area from the dPerFrameArr (assuming layout R, TTh, Eta, Area)
            // Index is 3*bigArrSize (start of Area block) + idx2d
            if (3 * bigArrSize + idx2d < 4 * bigArrSize) { // Check bounds for area read
                 double area = d_PerFrameArr[3 * bigArrSize + idx2d];
                 if (area > AREA_THRESHOLD) {
                      mySumIntArea += d_IntArrPerFrame[idx2d] * area;
                      mySumArea += area;
                 }
            }
        }
    }

    // --- Warp Level Reduction using shfl_down_sync ---
    // Sum contributions across all threads in the warp
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySumIntArea += __shfl_down_sync(0xFFFFFFFF, mySumIntArea, offset);
        mySumArea += __shfl_down_sync(0xFFFFFFFF, mySumArea, offset);
    }

    // --- Write Warp Result to Shared Memory ---
    // The thread at lane 0 now holds the sum for the entire warp
    if (lane == 0) {
        atomicAdd(&sIntArea[warpId], mySumIntArea); // Use atomicAdd for safety if multiple warps write
        atomicAdd(&sArea[warpId], mySumArea);
    }

    // --- Block Level Reduction ---
    __syncthreads(); // Ensure all warps have written to shared memory

    // One thread (e.g., tid 0) sums the results from all warps in the block
    if (tid == 0) {
        double finalSumIntArea = 0.0;
        double finalSumArea = 0.0;
        int numWarps = blockDim.x / warpSize; // Should match shared memory allocation
        if (blockDim.x % warpSize != 0) numWarps++; // Account for partial warps if applicable

        for (int i = 0; i < numWarps; ++i) {
             finalSumIntArea += sIntArea[i];
             finalSumArea += sArea[i];
        }

        // Calculate final average intensity for this R bin and write to global memory
        if (finalSumArea > AREA_THRESHOLD) {
            d_int1D[r_bin] = finalSumIntArea / finalSumArea;
        } else {
            d_int1D[r_bin] = 0.0; // Avoid division by zero or small number
        }
    }
}


// --- Host Wrapper for Full GPU Processing Pipeline ---
void ProcessImageGPU(const int64_t *hRaw, double *dProc, const double *dAvgDark, int Nopt, const int Topt[MAX_TRANSFORM_OPS], int NY, int NZ, bool doSub,
	int64_t* d_b1, int64_t* d_b2) {
    const size_t N = (size_t)NY * NZ;
    const size_t B64 = N * sizeof(int64_t);
    const int TPB = THREADS_PER_BLOCK_TRANSFORM;

    // Check for invalid transform options first
    bool anyT = false;
    if(Nopt > 0){
        for(int i = 0; i < Nopt; ++i){
            if(Topt[i] < 0 || Topt[i] > 3){
                fprintf(stderr,"GPU Err: Inv opt %d\n", Topt[i]);
                // Zero out the output buffer and return on error
                gpuErrchk(cudaMemset(dProc, 0, N * sizeof(double)));
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

        // Calculate grid dimensions
        unsigned long long nBUL = (N + TPB - 1) / TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){ // Cast mGDX for comparison
            fprintf(stderr, "Block count %llu exceeds max grid dim %d\n", nBUL, mGDX);
            exit(1);
        }
        dim3 nB((unsigned int)nBUL); // Ensure cast to unsigned int
        dim3 th(TPB);

        // Launch kernel for direct processing (cast + optional dark subtract)
        process_direct_kernel<<<nB, th>>>(d_b1, dProc, dAvgDark, N, doSub);
        gpuErrchk(cudaPeekAtLastError()); // Check for launch errors
        return;
    }

    // --- Case 2: One or more transformations needed ---
    // Copy initial raw data to the first buffer
    gpuErrchk(cudaMemcpy(d_b1, hRaw, B64, cudaMemcpyHostToDevice));

    const int64_t* rP = NULL; // Read pointer
    int64_t* wP = NULL; // Write pointer
    int cY = NY; // Current Y dimension
    int cZ = NZ; // Current Z dimension

    // --- Apply intermediate transformations sequentially ---
    for(int i = 0; i < Nopt - 1; ++i){
        int opt = Topt[i];
        // Determine read/write buffers for this step
        if(i % 2 == 0){ // Even step (0, 2, ...)
            rP = d_b1;
            wP = d_b2;
        } else { // Odd step (1, 3, ...)
            rP = d_b2;
            wP = d_b1;
        }

        int nY = cY; // Next Y dimension
        int nZ = cZ; // Next Z dimension

        // Handle transpose: check dimensions and update next dimensions
        if(opt == 3){
            if(cY != cZ){
                // Transpose on non-square image is complex, skipping as in CPU version
                fprintf(stderr,"GPU Warn: Skip Tpose %dx%d st %d\n", cY, cZ, i);
                opt = 0; // Treat as no-op for this step
            } else {
                nY = cZ; // Swap dimensions for transpose
                nZ = cY;
            }
        }

        size_t sON = (size_t)nY * nZ; // Size of the output for this step
        unsigned long long nBUL = (sON + TPB - 1) / TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
        if(nBUL > (unsigned long long)mGDX){
            fprintf(stderr,"Blk %llu > max %d (step %d)\n", nBUL, mGDX, i);
            cudaFree(d_b1);
            cudaFree(d_b2);
            exit(1);
        }
        dim3 nB((unsigned int)nBUL);
        dim3 th(TPB);

        // Launch kernel for sequential transform
        sequential_transform_kernel<<<nB, th>>>(rP, wP, cY, cZ, nY, nZ, opt);
        gpuErrchk(cudaPeekAtLastError());
        //gpuErrchk(cudaDeviceSynchronize()); // Sync only if strictly needed between steps

        // Update current dimensions for the next iteration
        cY = nY;
        cZ = nZ;
    }

    // --- Apply the FINAL transformation and processing (cast + subtract) ---
    int fOpt = Topt[Nopt - 1]; // Get the last transform operation
    // Determine the final read buffer based on the number of steps
    if((Nopt - 1) % 2 == 0){ // If Nopt-1 is even, last write was to d_b2, so read from d_b1
        rP = d_b1;
    } else { // If Nopt-1 is odd, last write was to d_b1, so read from d_b2
        rP = d_b2;
    }

    int nY = cY; // Final Y dimension
    int nZ = cZ; // Final Z dimension
    // Handle transpose for the final step
    if(fOpt == 3){
        if(cY != cZ){
            fprintf(stderr,"GPU Warn: Skip Tpose %dx%d final st %d\n", cY, cZ, Nopt - 1);
            fOpt = 0; // Treat as no-op
        } else {
            nY = cZ; // Swap dimensions
            nZ = cY;
        }
    }

    size_t fON = (size_t)nY * nZ; // Final output size
    // Check if the final size matches the original size (expected for dProc buffer)
    if(fON != N){
        fprintf(stderr, "GPU Warn: Final image size %zu != Original size %zu\n", fON, N);
        // This implies dProc might not be the correct size for the output.
        // If transforms change dimensions permanently, dProc allocation needs adjustment.
        // Assuming dProc is always allocated with the original N size.
        // If fON > N, this will write out of bounds. If fON < N, part of dProc won't be written.
        // For now, proceed but log the warning. A robust solution might require resizing dProc or erroring out.
    }

    unsigned long long nBUL = (fON + TPB - 1) / TPB; // Grid calc based on final size
    int mGDX;
    gpuErrchk(cudaDeviceGetAttribute(&mGDX, cudaDevAttrMaxGridDimX, 0));
    if(nBUL > (unsigned long long)mGDX){
        fprintf(stderr,"Blk %llu > max %d (final step)\n", nBUL, mGDX);
        cudaFree(d_b1);
        cudaFree(d_b2);
        exit(1);
    }
    dim3 nB((unsigned int)nBUL);
    dim3 th(TPB);

    // Launch the final kernel: transforms, casts to double, subtracts dark, writes to dProc
    final_transform_process_kernel<<<nB, th>>>(rP, dProc, dAvgDark, cY, cZ, nY, nZ, fOpt, doSub);
    gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize()); // Sync after final kernel if needed
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
    // Gradient calculation is not implemented (NLopt methods like Nelder-Mead don't require it)
    if(grad){
        for(unsigned i = 0; i < n; ++i){
            grad[i] = 0.0;
        }
    }

    const dataFit *d = (const dataFit*)fdat;
    const int Np = d->nrBins;    // Number of data points (bins)
    const double *Rs = d->R;     // R values (data)
    const double *Is = d->Int;   // Intensity values (data)

    // Parameters `x` structure: [A1, m1, c1, s1, A2, m2, c2, s2, ..., AnP, mnP, cnP, snP, bg_global]
    // n = 4 * nP + 1
    const int nP = (n - 1) / 4; // Number of peaks
    if(nP <= 0 || (4 * nP + 1 != n)){
        fprintf(stderr, "Obj Func Err: Invalid number of parameters %u for %d peaks\n", n, nP);
        return INFINITY; // Return infinity for invalid input
    }

    const double bg_g = x[n - 1]; // Extract the global background parameter

    double total_sq_error = 0.0;

    // Allocate buffer for the calculated intensity profile (model)
    // Use calloc to initialize to zero
    double *calculated_I = (double*)calloc(Np, sizeof(double));
    if(!calculated_I){
        fprintf(stderr, "Obj Func Err: Failed to allocate memory for calculated profile\n");
        return INFINITY; // Return infinity on allocation failure
    }

    // --- Calculate the contribution of each peak ---
    for(int pN = 0; pN < nP; ++pN){
        // Extract parameters for the current peak (pN)
        double Amplitude = x[pN * 4 + 0]; // Peak Amplitude (A)
        double Mix       = fmax(0.0, fmin(1.0, x[pN * 4 + 1])); // Gaussian/Lorentzian mix factor (m), constrained [0, 1]
        double Center    = x[pN * 4 + 2]; // Peak Center (c)
        double Sigma     = fmax(1e-9, x[pN * 4 + 3]); // Peak Width (s), constrained > 0

        // Add this peak's contribution to the calculated profile
        for(int i = 0; i < Np; ++i){
            double diff = Rs[i] - Center;
            double diff_sq = diff * diff;
            double sigma_sq = Sigma * Sigma;

            // Gaussian component (normalized)
            double gaussian = exp(-diff_sq / (2.0 * sigma_sq)) / (Sigma * sqrt(2.0 * M_PI));

            // Lorentzian component (normalized)
            double lorentzian = (1.0 / (M_PI * Sigma)) / (1.0 + diff_sq / sigma_sq);

            // Pseudo-Voigt profile contribution for this peak
            calculated_I[i] += Amplitude * (Mix * gaussian + (1.0 - Mix) * lorentzian);
        }
    }

    // --- Add global background and calculate squared error ---
    for(int i = 0; i < Np; ++i){
        calculated_I[i] += bg_g; // Add global background to the model intensity
        double error = calculated_I[i] - Is[i]; // Difference between model and data
        total_sq_error += error * error; // Accumulate squared error
    }

    free(calculated_I); // Free the temporary buffer
    return total_sq_error; // Return the total squared error (objective function value)
}

// Apply Savitzky-Golay smoothing filter (coefficients for specific window sizes)
void smoothData(const double *in, double *out, int N, int W) {
    // W = Window size (must be odd, >= 3)
    if(W < 3 || W % 2 == 0){
        // Invalid window size, just copy input to output
        memcpy(out, in, N * sizeof(double));
        return;
    }
    int H = W / 2; // Half-window size

    // Savitzky-Golay coefficients (pre-calculated, for smoothing, polynomial order 2)
    // Source: Numerical Recipes or similar resources
    double *coeffs = (double*)malloc(W * sizeof(double));
    check(!coeffs, "smoothData: Malloc failed for coefficients");
    double norm = 0.0;

    switch(W){
        case 5: norm = 35.0; coeffs[0]=-3; coeffs[1]=12; coeffs[2]=17; coeffs[3]=12; coeffs[4]=-3; break;
        case 7: norm = 21.0; coeffs[0]=-2; coeffs[1]= 3; coeffs[2]= 6; coeffs[3]= 7; coeffs[4]= 6; coeffs[5]= 3; coeffs[6]=-2; break;
        case 9: norm = 231.0; coeffs[0]=-21; coeffs[1]=14; coeffs[2]=39; coeffs[3]=54; coeffs[4]=59; coeffs[5]=54; coeffs[6]=39; coeffs[7]=14; coeffs[8]=-21; break;
        // Add more cases for other window sizes if needed
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
            // Handle boundaries: just copy original data (could use reflection or other methods)
            out[i] = in[i];
        } else {
            // Apply convolution in the center
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
    if(N < 3){
        *foundPeaks = NULL;
        return 0; // Cannot find peaks in less than 3 points
    }

    // Allocate space for the maximum possible number of peaks found initially
    int maxPossiblePeaks = N / 2 + 1;
    Peak* preliminaryPeaks = (Peak*)malloc(maxPossiblePeaks * sizeof(Peak));
    check(!preliminaryPeaks, "findPeaks: Malloc failed for preliminaryPeaks");
    int peakCount = 0;

    // --- Step 1: Find all local maxima above minHeight ---
    for(int i = 1; i < N - 1; ++i){
        if(data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] >= minHeight){
            // Found a local maximum satisfying height condition
            if(peakCount < maxPossiblePeaks){
                preliminaryPeaks[peakCount].index = i;
                preliminaryPeaks[peakCount].radius = r_values[i];
                preliminaryPeaks[peakCount].intensity = data[i];
                peakCount++;
            } else {
                fprintf(stderr, "Peak find warn: Exceeded max possible peaks buffer. Some peaks might be missed.\n");
                break; // Stop finding more peaks if buffer overflows
            }
        }
    }

    if(peakCount == 0 || minDistance <= 1){
        // No peaks found, or no distance filtering needed
        // Realloc to exact size (or NULL if 0 peaks)
        Peak* finalPeaks = (Peak*)realloc(preliminaryPeaks, peakCount * sizeof(Peak));
        if(peakCount > 0 && finalPeaks == NULL) {
             // realloc failed but we still have the original buffer
             *foundPeaks = preliminaryPeaks;
             fprintf(stderr, "findPeaks Warn: realloc failed, returning potentially oversized buffer\n");
        } else {
             *foundPeaks = finalPeaks; // Can be NULL if peakCount is 0
        }
        return peakCount;
    }

    // --- Step 2: Apply minimum distance constraint ---
    // Keep track of peaks to suppress
    bool* isSuppressed = (bool*)calloc(peakCount, sizeof(bool));
    check(!isSuppressed, "findPeaks: Calloc failed for isSuppressed");

    // Iterate through preliminary peaks (sorting by intensity first can be more efficient, but not done here)
    for(int i = 0; i < peakCount; ++i){
        if(isSuppressed[i]) {
            continue; // Skip if already marked for suppression
        }
        // Check peaks after peak 'i'
        for(int j = i + 1; j < peakCount; ++j){
            if(isSuppressed[j]) {
                continue;
            }
            // Calculate distance between peaks 'i' and 'j'
            int distance = abs(preliminaryPeaks[i].index - preliminaryPeaks[j].index);

            if(distance < minDistance){
                // Peaks are too close, suppress the smaller one
                if(preliminaryPeaks[j].intensity <= preliminaryPeaks[i].intensity) {
                    isSuppressed[j] = true; // Suppress peak j
                } else {
                    isSuppressed[i] = true; // Suppress peak i
                    break; // Peak i is suppressed, no need to compare it further in the inner loop
                }
            }
        }
    }

    // --- Step 3: Create the final filtered list of peaks ---
    // Allocate space for the filtered peaks (worst case, same as peakCount)
    Peak* filteredPeaks = (Peak*)malloc(peakCount * sizeof(Peak));
    check(!filteredPeaks, "findPeaks: Malloc failed for filteredPeaks");
    int filteredCount = 0;
    for(int i = 0; i < peakCount; ++i){
        if(!isSuppressed[i]){
            filteredPeaks[filteredCount++] = preliminaryPeaks[i];
        }
    }

    // Free temporary arrays
    free(preliminaryPeaks);
    free(isSuppressed);

    // Reallocate final array to the exact size
    Peak* finalPeaks = (Peak*)realloc(filteredPeaks, filteredCount * sizeof(Peak));
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
    int NrPixelsY = 0, NrPixelsZ = 0;
    int Normalize = 1;
    int nEtaBins = 0, nRBins = 0;
    char *ParamFN = argv[1];
    FILE *pF = fopen(ParamFN, "r");
    check(!pF, "Failed open param file: %s", ParamFN);

    char line[4096], key[1024], val_str[3072]; // Buffers for parsing
    const char *s; // Pointer for StartsWith checks
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

    // Read parameters line by line
    while(fgets(line, sizeof(line), pF)){
        // Skip comments, blank lines, and lines that are too short
        if(line[0] == '#' || isspace(line[0]) || strlen(line) < 3) {
            continue;
        }

        // Use sscanf to parse "key value" pairs, robust against extra whitespace
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
             else if (strcmp(key, "NrPixels") == 0) { sscanf(val_str, "%d", &NrPixelsY); NrPixelsZ = NrPixelsY; } // Shortcut
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
             else if (strcmp(key, "PeakLocation") == 0) {
                 if(nSpecP < MAX_PEAK_LOCATIONS) {
                     sscanf(val_str, "%lf", &pkLoc[nSpecP++]);
                     multiP = 1;                                                          // Implicitly enable multi-peak etc.
                     pkFit = 1;
                     doSm = 0;
                 } else {
                     printf("Warn: Max %d PeakLocation reached, ignoring further locations.\n", MAX_PEAK_LOCATIONS);
                 }
             }
             // Add other parameters here if needed
        }
    }
    fclose(pF);

    // Validate essential parameters
    check(NrPixelsY <= 0 || NrPixelsZ <= 0, "NrPixelsY/Z invalid or not set in parameter file.");
    check(Lsd <= 0 || px <= 0, "Lsd/px invalid or not set in parameter file.");

    // Ensure multi-peak finding is enabled if specific peaks are given for fitting
    if(pkFit && nSpecP > 0) {
        multiP = 1;
        if (doSm) {
            printf("Warn: Smoothing disabled because specific PeakLocations were provided.\n");
            doSm = 0; // Don't smooth if fitting specific locations
        }
    }

    // Calculate number of bins
    nRBins = (RBinSize > 1e-9) ? (int)ceil((RMax - RMin) / RBinSize) : 0;
    nEtaBins = (EtaBinSize > 1e-9) ? (int)ceil((EtaMax - EtaMin) / EtaBinSize) : 0;
    check(nRBins <= 0 || nEtaBins <= 0, "Invalid bin parameters. R bins=%d, Eta bins=%d", nRBins, nEtaBins);
    size_t bigArrSize = (size_t)nRBins * nEtaBins; // Total number of R-Eta bins

    // Print summary of parameters
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
    double *hAvgDark = NULL;     // Averaged dark frame (double) on host
    int64_t *hDarkInT = NULL;   // Temporary buffer for reading raw dark frame (int64)
    int64_t *hDarkIn = NULL;    // Buffer for transformed dark frame (int64)

    size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;
    size_t SizeFile = totalPixels * BYTES_PER_PIXEL; // Size of one frame in bytes

    hAvgDark = (double*)calloc(totalPixels, sizeof(double)); // Initialize avg dark to zeros
    check(!hAvgDark, "Allocation failed for hAvgDark");
    hDarkInT = (int64_t*)malloc(SizeFile);
    check(!hDarkInT, "Allocation failed for hDarkInT");
    hDarkIn = (int64_t*)malloc(SizeFile);
    check(!hDarkIn, "Allocation failed for hDarkIn");

    // --- Device Memory Allocations (Persistent) ---
    double *dAvgDark = NULL;         // Averaged dark frame on GPU
    double *dProcessedImage = NULL;  // Transformed, dark-subtracted image on GPU (double)
    double *d_int1D = NULL;          // Final 1D integrated profile on GPU
    int *dMapMask = NULL;            // Pixel mask on GPU (optional)
    size_t mapMaskWC = 0;            // Word count for the mask array
    int *dNPxList = NULL;            // nMap data (pixel counts, offsets) on GPU
    struct data *dPxList = NULL;     // Map data (pixel coords, fractions) on GPU
    double *dSumMatrix = NULL;       // Accumulated 2D integrated pattern on GPU (optional)
    double *dIntArrFrame = NULL;     // 2D integrated pattern for the current frame on GPU
    double *dPerFrame = NULL;        // R, TTh, Eta, Area values per R-Eta bin on GPU
    double *dEtaLo = NULL, *dEtaHi = NULL, *dRLo = NULL, *dRHi = NULL; // Bin edges on GPU

    // <<< ADDED: Persistent temporary buffers for transformations >>>
    int64_t *g_dTempTransformBuf1 = NULL;
    int64_t *g_dTempTransformBuf2 = NULL;
    // <<< END ADDED >>>

    bool darkSubEnabled = (argc > 2); // Dark subtraction is enabled if DarkAvgFN is provided

    // Allocate essential GPU buffers
    gpuErrchk(cudaMalloc(&dProcessedImage, totalPixels * sizeof(double)));
    gpuErrchk(cudaMalloc(&dPxList, szPxList));
    gpuErrchk(cudaMalloc(&dNPxList, szNPxList));
    gpuErrchk(cudaMalloc(&dIntArrFrame, bigArrSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&dPerFrame, bigArrSize * 4 * sizeof(double))); // R, TTh, Eta, Area
    gpuErrchk(cudaMalloc(&dEtaLo, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dEtaHi, nEtaBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRLo, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&dRHi, nRBins * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_int1D, nRBins * sizeof(double)));

    // <<< ADDED: Allocate persistent transform buffers >>>
    size_t tempBufferSize = totalPixels * sizeof(int64_t);
    printf("Allocating persistent GPU transform buffers (%zu bytes each)...\n", tempBufferSize);
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf1, tempBufferSize));
    gpuErrchk(cudaMalloc(&g_dTempTransformBuf2, tempBufferSize));
    // <<< END ADDED >>>

    // Copy map data and bin edges to GPU
    gpuErrchk(cudaMemcpy(dPxList, pxList, szPxList, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dNPxList, nPxList, szNPxList, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaLo, hEtaLo, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaHi, hEtaHi, nEtaBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRLo, hRLo, nRBins * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRHi, hRHi, nRBins * sizeof(double), cudaMemcpyHostToDevice));

    // --- Process Dark Frame (Mask generation happens here) ---
    double t_start_dark = get_wall_time_ms();
    int nDarkFramesRead = 0;
    int *hMapMask = NULL; // Host-side mask buffer
    // dMapMask is allocated and filled inside this block if mkMap=1

    if(darkSubEnabled){
        char* darkFN = argv[2];
        FILE* fD = fopen(darkFN, "rb");
        check(!fD, "Failed to open dark frame file: %s", darkFN);

        // Check file size to determine number of frames
        fseek(fD, 0, SEEK_END);
        size_t szD = ftell(fD);
        rewind(fD);
        int nFD = szD / SizeFile; // Number of full frames in the file
        check(nFD == 0 || szD % SizeFile != 0, "Dark file %s incomplete (size %zu, frame %zu). Found %d frames.",
              darkFN, szD, SizeFile, nFD);
        printf("Reading dark file: %s, Found %d frames.\n", darkFN, nFD);

        // Process each frame in the dark file
        for(int i = 0; i < nFD; ++i){
            check(fread(hDarkInT, 1, SizeFile, fD) != SizeFile, "Read failed for dark frame %d from %s", i, darkFN);
            // Apply the same transformations to the dark frame as the data frames (using CPU version)
            DoImageTransformationsSequential(Nopt, Topt, hDarkInT, hDarkIn, NrPixelsY, NrPixelsZ);

            // Generate mask from the first transformed dark frame if requested
            if(mkMap == 1 && i == 0){
                mapMaskWC = (totalPixels + 31) / 32; // Calculate word count needed for bitmap
                hMapMask = (int*)calloc(mapMaskWC, sizeof(int)); // Allocate and zero host mask
                check(!hMapMask, "Allocation failed for host mask buffer");
                int maskedPixelCount = 0; // Local counter for this block
                for(size_t j = 0; j < totalPixels; ++j){
                    // Check if pixel value matches Gap or Bad Pixel intensity
                    if(hDarkIn[j] == GapI || hDarkIn[j] == BadPxI){
                        SetBit(hMapMask, j); // Set the corresponding bit in the mask
                        maskedPixelCount++; // Increment count of masked pixels
                    }
                }
                printf("Mask generated from first dark frame: %d pixels masked.\n", maskedPixelCount);
                // Allocate mask on GPU and copy from host
                gpuErrchk(cudaMalloc(&dMapMask, mapMaskWC * sizeof(int)));                // dMapMask allocated here
                gpuErrchk(cudaMemcpy(dMapMask, hMapMask, mapMaskWC * sizeof(int), cudaMemcpyHostToDevice));
                mkMap = 0; // Mask generation done
            }

            // Accumulate the transformed dark frame into the host average buffer
            for(size_t j = 0; j < totalPixels; ++j){
                hAvgDark[j] += (double)hDarkIn[j];
            }
        }
        fclose(fD);
        nDarkFramesRead = nFD; // Store the actual number of frames read

        // Calculate the average dark frame on host
        if(nDarkFramesRead > 0){
            for(size_t j = 0; j < totalPixels; ++j){
                hAvgDark[j] /= (double)nDarkFramesRead;
            }
            printf("Averaged %d dark frames.\n", nDarkFramesRead);
        }

        // Allocate GPU buffer for average dark and copy from host
        gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
        gpuErrchk(cudaMemcpy(dAvgDark, hAvgDark, totalPixels * sizeof(double), cudaMemcpyHostToDevice));
        printf("Average dark frame copied to GPU.\n");

    } else {
        // No dark frame provided, use zeros on GPU for average
        gpuErrchk(cudaMalloc(&dAvgDark, totalPixels * sizeof(double)));
        gpuErrchk(cudaMemset(dAvgDark, 0, totalPixels * sizeof(double)));
        printf("No dark frame provided, using zeros on GPU.\n");
        // dMapMask remains NULL, mapMaskWC remains 0
    }
    // --- End Dark Processing ---

    // At this point, dMapMask is either NULL or points to the mask on the GPU
    // mapMaskWC is either 0 or the word count for the mask

    printf("Initializing static PerFrame array (R, TTh, Eta, Area) on GPU...\n");
    int initTPB = 256; // Choose a reasonable block size
    int initBlocks = (bigArrSize + initTPB - 1) / initTPB;
    initialize_PerFrameArr_Area_kernel<<<initBlocks, initTPB>>>(
        dPerFrame, bigArrSize,
        nRBins, nEtaBins,
        dRLo, dRHi, dEtaLo, dEtaHi,
        dPxList, dNPxList,                      // Pass map data
        NrPixelsY, NrPixelsZ,                   // Pass detector dimensions
        dMapMask, mapMaskWC,                    // Pass mask info (dMapMask might be NULL)
        px, Lsd
    );
    gpuErrchk(cudaPeekAtLastError()); // Check launch
    gpuErrchk(cudaDeviceSynchronize()); // Ensure initialization completes before proceeding
    printf("GPU PerFrame array initialization complete (Area pre-calculated %s mask).\n",
           (dMapMask != NULL ? "with" : "without"));

    // Allocate and initialize summation buffer if needed
    if(sumI){
        gpuErrchk(cudaMalloc(&dSumMatrix, bigArrSize * sizeof(double)));
        gpuErrchk(cudaMemset(dSumMatrix, 0, bigArrSize * sizeof(double)));
    }
    printf("Processed dark/mask: %.3f ms\n", get_wall_time_ms() - t_start_dark);
    fflush(stdout);

    // --- Network Setup ---
    CHUNK_SIZE = SizeFile; // Size of image data payload
    TOTAL_MSG_SIZE = HEADER_SIZE + CHUNK_SIZE; // Total expected message size
    printf("Network: Expecting %zu B header + %zu B data = %zu B total per message.\n", HEADER_SIZE, CHUNK_SIZE, TOTAL_MSG_SIZE);

    int server_fd;
    struct sockaddr_in server_addr;
    queue_init(&process_queue); // Initialize the processing queue

    // Create socket
    check((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0, "Socket creation failed: %s", strerror(errno));

    // Set socket options (allow reuse of address/port)
    int sock_opt = 1;
    check(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &sock_opt, sizeof(sock_opt)), "setsockopt failed: %s", strerror(errno));

    // Prepare server address structure
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY; // Bind to any local address
    server_addr.sin_port = htons(PORT);

    // Bind the socket to the address and port
    check(bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0, "Bind failed for port %d: %s", PORT, strerror(errno));

    // Start listening for incoming connections
    check(listen(server_fd, MAX_CONNECTIONS) < 0, "Listen failed: %s", strerror(errno));
    printf("Server listening on port %d\n", PORT);

    // Create and detach the thread that accepts incoming connections
    pthread_t accept_thread;
    check(pthread_create(&accept_thread, NULL, accept_connections, &server_fd) != 0, "Failed to create accept thread: %s", strerror(errno));

    // --- Prepare for Main Loop ---

    // Open output files (use append binary mode "ab" to add to existing files if run multiple times)
    FILE *fLineout = fopen("lineout.bin", "wb"); // Overwrite lineout file each run
    check(!fLineout, "Error opening lineout.bin for writing: %s", strerror(errno));

    FILE *fFit = NULL; // File handle for fit results
    if(pkFit){
        fFit = fopen("fit.bin", "wb"); // Overwrite fit file each run
        check(!fFit, "Error opening fit.bin for writing: %s", strerror(errno));
    }

    FILE *f2D = NULL; // File handle for the single 2D integrated data file
    if (wr2D) {
        printf("Will write all 2D integrated patterns to single file: Int2D.bin\n");
        f2D = fopen("Int2D.bin", "wb"); // Open in write binary mode (overwrites)
        check(!f2D, "Error opening Int2D.bin for writing: %s", strerror(errno));
    }

    // CUDA Events for timing GPU stages
    cudaEvent_t ev_proc_start, ev_proc_stop;   // Image processing (transform, subtract)
    cudaEvent_t ev_integ_start, ev_integ_stop; // Integration (2D)
    cudaEvent_t ev_prof_start, ev_prof_stop;  // 1D Profile calculation
    cudaEvent_t ev_d2h_start, ev_d2h_stop;    // Device to Host copies
    gpuErrchk(cudaEventCreate(&ev_proc_start)); gpuErrchk(cudaEventCreate(&ev_proc_stop));
    gpuErrchk(cudaEventCreate(&ev_integ_start)); gpuErrchk(cudaEventCreate(&ev_integ_stop));
    gpuErrchk(cudaEventCreate(&ev_prof_start)); gpuErrchk(cudaEventCreate(&ev_prof_stop));
    gpuErrchk(cudaEventCreate(&ev_d2h_start)); gpuErrchk(cudaEventCreate(&ev_d2h_stop));

    // Variables for timing results (milliseconds)
    float t_proc_gpu = 0, t_integ_gpu = 0, t_prof_gpu = 0, t_d2h_gpu = 0; // GPU times
    double t_qp_cpu = 0, t_write1d_cpu = 0, t_fit_cpu = 0, t_writefit_cpu = 0, t_write2d_cpu = 0, t_loop_cpu = 0; // CPU times
    double t_start_loop, t_end_loop; // Wall clock time per loop

    // Host buffers for results
    int firstFrame = 1; // Flag to handle initialization tasks on the first frame
    double *hIntArrFrame = NULL; // Host buffer for 2D integrated frame (if wr2D)
    double *hPerFrame = NULL; // Host buffer for R, TTh, Eta, Area
    double *h_int1D = NULL; // Pinned host buffer for 1D profile
    // Allocate pinned/regular host buffers for results
    if (wr2D) {
        gpuErrchk(cudaMallocHost((void**)&hIntArrFrame, bigArrSize * sizeof(double)));    // Pinned for async copy
        check(!hIntArrFrame, "Allocation failed for pinned hIntArrFrame");
    }
    gpuErrchk(cudaMallocHost((void**)&hPerFrame, bigArrSize * 4 * sizeof(double)));       // Pinned for async copy
    check(!hPerFrame, "Allocation failed for pinned hPerFrame");
    gpuErrchk(cudaMallocHost((void**)&h_int1D, nRBins * sizeof(double)));
    check(!h_int1D, "Allocation failed for pinned host buffer h_int1D");
    double *hR = (double*)calloc(nRBins, sizeof(double)); // Host buffer for R bin centers
    check(!hR, "Allocation failed for hR");
    double *hEta = (double*)calloc(nEtaBins, sizeof(double)); // Host buffer for Eta bin centers
    check(!hEta, "Allocation failed for hEta");
    double *hLineout = (double*)malloc(nRBins * 2 * sizeof(double)); // Host buffer for writing lineout (R, Intensity pairs)
    check(!hLineout, "Allocation failed for hLineout");

    printf("Setup complete. Starting main processing loop...\n");
    double t_end_setup = get_wall_time_ms();
    printf("Total setup time: %.3f ms\n", t_end_setup - t_start_main);
    fflush(stdout);

    // =========================== Main Processing Loop ===========================
    int frameCounter = 0;
    while (keep_running) {
        t_start_loop = get_wall_time_ms(); // Start timing the loop iteration

        // --- Get next data chunk from queue ---
        double t_qp_start = get_wall_time_ms();
        DataChunk chunk;
        if(queue_pop(&process_queue, &chunk) < 0){
            // Shutdown requested while queue was empty
            break;
        }
        t_qp_cpu = get_wall_time_ms() - t_qp_start; // Time spent waiting for/getting data

        // --- GPU Processing Stage (Transform, Cast, Subtract Dark) ---
        gpuErrchk(cudaEventRecord(ev_proc_start, 0)); // Stream 0
        ProcessImageGPU(chunk.data, dProcessedImage, dAvgDark, Nopt, Topt, NrPixelsY, NrPixelsZ, darkSubEnabled,
                        g_dTempTransformBuf1, g_dTempTransformBuf2);
        gpuErrchk(cudaEventRecord(ev_proc_stop, 0)); // Stream 0

        // --- GPU Integration Stage (2D Integration) ---
        int currFidx = chunk.dataset_num; // Frame index/number from the data source
        int integTPB = THREADS_PER_BLOCK_INTEGRATE;
        int nrVox = (bigArrSize + integTPB - 1) / integTPB; // Number of blocks for integration kernel

        gpuErrchk(cudaEventRecord(ev_integ_start, 0)); // Stream 0
        if(!dMapMask){ // Launch kernel without mask support
            integrate_noMapMask<<<nrVox, integTPB>>>(
                px, Lsd, bigArrSize, Normalize, sumI, currFidx,
                dPxList, dNPxList,
                NrPixelsY, NrPixelsZ,
                dProcessedImage, dIntArrFrame, dSumMatrix);
        } else { // Launch kernel with mask support
            integrate_MapMask<<<nrVox, integTPB>>>(
                px, Lsd, bigArrSize, Normalize, sumI, currFidx,
                mapMaskWC, dMapMask, nRBins, nEtaBins, // Keep nRBins/nEtaBins here if needed
                NrPixelsY, NrPixelsZ,
                dPxList, dNPxList,
                dProcessedImage, dIntArrFrame, dSumMatrix);
        }
        gpuErrchk(cudaPeekAtLastError()); // Check for kernel launch errors immediately
        gpuErrchk(cudaEventRecord(ev_integ_stop, 0)); // Stream 0

        // --- GPU 1D Profile Stage (Reduction from 2D integrated) ---
        size_t profileSharedMem = (THREADS_PER_BLOCK_PROFILE / 32) * sizeof(double) * 2; // Shared mem per warp * 2 buffers
        gpuErrchk(cudaEventRecord(ev_prof_start, 0)); // Stream 0
        calculate_1D_profile_kernel<<<nRBins, THREADS_PER_BLOCK_PROFILE, profileSharedMem>>>(
            dIntArrFrame, dPerFrame, d_int1D, nRBins, nEtaBins, bigArrSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(ev_prof_stop, 0)); // Stream 0

        // --- D->H Copy Stage (Asynchronous) ---
        gpuErrchk(cudaEventRecord(ev_d2h_start, 0)); // Stream 0
        // Copy 1D profile result to pinned host memory
        gpuErrchk(cudaMemcpyAsync(h_int1D, d_int1D, nRBins * sizeof(double), cudaMemcpyDeviceToHost, 0));
        // On the very first frame, also copy the R, TTh, Eta, Area array (all pre-calculated now)
        if(firstFrame == 1){
             // This copy now brings R, TTh, Eta, Area (all calculated once by init kernel)
             gpuErrchk(cudaMemcpyAsync(hPerFrame, dPerFrame, bigArrSize * 4 * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        // If writing 2D data, copy the 2D integrated frame to host memory (use pinned buffer)
        if(wr2D && hIntArrFrame){
            gpuErrchk(cudaMemcpyAsync(hIntArrFrame, dIntArrFrame, bigArrSize * sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        gpuErrchk(cudaEventRecord(ev_d2h_stop, 0)); // Stream 0

        // --- Synchronize GPU Events and Get Timings ---
        // Synchronize *after* launching all GPU work for the frame
        gpuErrchk(cudaEventSynchronize(ev_proc_stop)); // Wait for processing to finish
        gpuErrchk(cudaEventElapsedTime(&t_proc_gpu, ev_proc_start, ev_proc_stop));

        gpuErrchk(cudaEventSynchronize(ev_integ_stop)); // Wait for integration to finish
        gpuErrchk(cudaEventElapsedTime(&t_integ_gpu, ev_integ_start, ev_integ_stop));

        gpuErrchk(cudaEventSynchronize(ev_prof_stop)); // Wait for 1D profile to finish
        gpuErrchk(cudaEventElapsedTime(&t_prof_gpu, ev_prof_start, ev_prof_stop));

        // *** Crucially, synchronize the D->H copy event BEFORE using the host data ***
        gpuErrchk(cudaEventSynchronize(ev_d2h_stop));
        gpuErrchk(cudaEventElapsedTime(&t_d2h_gpu, ev_d2h_start, ev_d2h_stop));

        // --- CPU Processing Stage (using results from D->H copy) ---

        // --- Initialize Host R/Eta Arrays (First Frame Only) ---
        if(firstFrame == 1){
            // Now we extract R/Eta from the hPerFrame copied *after* the first frame's D->H copy
            // This buffer now contains the static R, TTh, Eta, Area initialized by the separate kernel
            for(int r = 0; r < nRBins; ++r){
                // Access R using the correct offset (0 * bigArrSize)
                hR[r] = hPerFrame[r * nEtaBins + 0 * bigArrSize];
            }
            for(int e = 0; e < nEtaBins; ++e){
                // Access Eta using the correct offset (2 * bigArrSize)
                 hEta[e] = hPerFrame[e + 2 * bigArrSize];
            }
            // Prepare the R values in the lineout buffer (Intensity will be filled later)
            for(int r = 0; r < nRBins; ++r){
                hLineout[r * 2] = hR[r]; // R value
                hLineout[r * 2 + 1] = 0.0; // Initialize intensity
            }
            printf("Initialized host R/Eta arrays from first frame D->H copy (using pre-initialized GPU data).\n");
            firstFrame = 0; // Don't run this block again
        }

        // --- Write 2D Integrated Data (if enabled) ---
        double t_write2d_start = get_wall_time_ms();
        if (wr2D && f2D && hIntArrFrame) { // Check flag, file handle, and buffer validity
            // Write the copied 2D data (hIntArrFrame) to the file
            size_t written = fwrite(hIntArrFrame, sizeof(double), bigArrSize, f2D);
            if (written != bigArrSize) {
                fprintf(stderr, "Warn: Failed to write full 2D frame %d to Int2D.bin (wrote %zu/%zu): %s\n",
                        currFidx, written, bigArrSize, strerror(errno));
            }
            // Optional: Flush periodically to reduce data loss risk on crash
            // if (frameCounter % 100 == 0) { fflush(f2D); }
        }
        t_write2d_cpu = get_wall_time_ms() - t_write2d_start;

        // --- Prepare 1D Lineout Data ---
        double maxInt = -1.0;
        int maxIntLoc = -1;
        // Find max intensity and its location if only fitting a single peak (legacy?)
        if(!multiP){
            for(int r = 0; r < nRBins; ++r){
                if(h_int1D[r] > maxInt){
                    maxInt = h_int1D[r];
                    maxIntLoc = r;
                }
            }
        }
        // Fill the intensity part of the lineout buffer
        for(int r = 0; r < nRBins; ++r){
            hLineout[r * 2 + 1] = h_int1D[r]; // Intensity value
        }

        // --- Write 1D Lineout Data ---
        double t_write1d_start = get_wall_time_ms();
        if (fLineout) {
             size_t written = fwrite(hLineout, sizeof(double), nRBins * 2, fLineout);
             if (written != (size_t)nRBins * 2) {
                  fprintf(stderr, "Warn: Failed to write full lineout frame %d (wrote %zu/%d): %s\n",
                           currFidx, written, nRBins * 2, strerror(errno));
             } else {
                  fflush(fLineout); // Flush after each write to ensure data is saved
             }
        }
        t_write1d_cpu = get_wall_time_ms() - t_write1d_start;

        // --- Peak Finding and Fitting (if enabled) ---
        double t_fit_start = get_wall_time_ms();
        int currentPeakCount = 0; // Number of peaks found/fitted in this frame
        double *sendFitParams = NULL; // Buffer for formatted fit parameters to be saved/sent

        if(pkFit){
            Peak *pks = NULL; // Array to store found/specified peaks

            // --- Step 1: Identify Peak Candidates ---
            if(multiP){ // Multi-peak mode
                if(nSpecP > 0){ // Specific peak locations provided
                    // Allocate space for specified peaks
                    pks = (Peak*)malloc(nSpecP * sizeof(Peak));
                    check(!pks, "pkFit: Malloc failed for specified peaks array");
                    int validPeakCount = 0;
                    // Find the nearest bin index for each specified R location
                    for(int p = 0; p < nSpecP; ++p){
                        int bestBin = -1;
                        double minDiff = 1e10;
                        for(int r = 0; r < nRBins; ++r){
                            double diff = fabs(hR[r] - pkLoc[p]);
                            if(diff < minDiff){
                                minDiff = diff;
                                bestBin = r;
                            }
                        }
                        // Include peak if it's reasonably close to a bin center
                        if(bestBin != -1 && minDiff < RBinSize * 2.0){ // Threshold: within 2 bins
                            pks[validPeakCount].index = bestBin;
                            pks[validPeakCount].radius = hR[bestBin];
                            pks[validPeakCount].intensity = h_int1D[bestBin]; // Use actual intensity
                            validPeakCount++;
                        } else {
                            printf("Warn: Specified peak R=%.4f ignored Frame %d (too far from bins).\n", pkLoc[p], currFidx);
                        }
                    }
                    currentPeakCount = validPeakCount;
                    if(validPeakCount == 0){ // If no specified peaks were valid, free
                        free(pks);
                        pks = NULL;
                    } else if (validPeakCount < nSpecP) { // Realloc if some were ignored
                        Peak * reallocPks = (Peak*)realloc(pks, validPeakCount * sizeof(Peak));
                        if(reallocPks) pks = reallocPks;
                        else { /* Keep original potentially larger buffer */ }
                    }
                } else { // No specific locations, find peaks automatically
                    double *dataToFindPeaks = h_int1D; // Use raw data by default
                    double *smoothedData = NULL;
                    if(doSm){ // Apply smoothing if enabled
                        smoothedData = (double*)malloc(nRBins * sizeof(double));
                        check(!smoothedData, "pkFit: Malloc failed for smoothedData buffer");
                        smoothData(h_int1D, smoothedData, nRBins, 7); // Example: Window size 7
                        dataToFindPeaks = smoothedData; // Use smoothed data for peak finding
                    }
                    // Find peaks using the (potentially smoothed) data
                    currentPeakCount = findPeaks(dataToFindPeaks, hR, nRBins, &pks, 0.0, 5); // Example params
                    if(smoothedData) {
                        free(smoothedData); // Free smoothing buffer if it was used
                    }
                }
            } else { // Single peak mode (find highest intensity)
                if(maxIntLoc != -1){ // If a maximum was found earlier
                    currentPeakCount = 1;
                    pks = (Peak*)malloc(sizeof(Peak));
                    check(!pks, "pkFit: Malloc failed for single peak");
                    pks[0].index = maxIntLoc;
                    pks[0].radius = hR[maxIntLoc];
                    pks[0].intensity = maxInt;
                } else {
                    currentPeakCount = 0; // No peak found
                    pks = NULL;
                }
            }

            // --- Step 2: Perform Fit if Peaks were Found ---
            if (currentPeakCount > 0 && pks != NULL) {
                int nFitParams = currentPeakCount * 4 + 1; // 4 params/peak + 1 global BG
                double *fitParams = (double*)malloc(nFitParams * sizeof(double)); check(!fitParams, "pkFit: Malloc fitParams");
                double *lowerBounds = (double*)malloc(nFitParams * sizeof(double)); check(!lowerBounds, "pkFit: Malloc lowerBounds");
                double *upperBounds = (double*)malloc(nFitParams * sizeof(double)); check(!upperBounds, "pkFit: Malloc upperBounds");

                // --- Set Initial Guesses and Bounds ---
                double maxOverallIntensity = 0.0; // Find max intensity for bounds
                for(int r = 0; r < nRBins; ++r) {
                    if(h_int1D[r] > maxOverallIntensity) maxOverallIntensity = h_int1D[r];
                }
                if(maxOverallIntensity <= 0) maxOverallIntensity = 1.0; // Avoid zero bounds

                for(int p = 0; p < currentPeakCount; ++p){
                    int base = p * 4;
                    double initialCenter = pks[p].radius;
                    double initialIntensity = pks[p].intensity;
                    double initialSigma = RBinSize * 2.0; // Initial guess for width

                    // Initial Guesses (x)
                    fitParams[base + 0] = initialIntensity; // Amplitude
                    fitParams[base + 1] = 0.5;             // Mix (50% Gaussian/Lorentzian)
                    fitParams[base + 2] = initialCenter;   // Center
                    fitParams[base + 3] = initialSigma;    // Sigma

                    // Lower Bounds (lb)
                    lowerBounds[base + 0] = 0.0;                         // Amplitude >= 0
                    lowerBounds[base + 1] = 0.0;                         // Mix >= 0
                    lowerBounds[base + 2] = initialCenter - RBinSize * 5.0; // Center +/- 5 bins
                    lowerBounds[base + 3] = RBinSize * 0.5;              // Sigma >= half bin

                    // Upper Bounds (ub)
                    upperBounds[base + 0] = maxOverallIntensity * 2.0;    // Amplitude <= 2*max
                    upperBounds[base + 1] = 1.0;                         // Mix <= 1
                    upperBounds[base + 2] = initialCenter + RBinSize * 5.0; // Center +/- 5 bins
                    upperBounds[base + 3] = (RMax - RMin) / 4.0;         // Sigma <= R range/4
                }
                // Global Background (last parameter)
                fitParams[nFitParams - 1] = 0.0;                         // Initial BG guess = 0
                lowerBounds[nFitParams - 1] = -maxOverallIntensity;      // BG lower bound
                upperBounds[nFitParams - 1] = maxOverallIntensity;       // BG upper bound

                // --- Setup NLopt ---
                dataFit fitData;
                fitData.nrBins = nRBins;
                fitData.R = hR;
                fitData.Int = h_int1D; // Fit against the original (unsmoothed) data

                nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, nFitParams);
                nlopt_set_lower_bounds(opt, lowerBounds);
                nlopt_set_upper_bounds(opt, upperBounds);
                nlopt_set_min_objective(opt, problem_function_global_bg, &fitData);
                nlopt_set_xtol_rel(opt, 1e-4); // Relative tolerance
                nlopt_set_maxeval(opt, 500 * nFitParams); // Limit evaluations

                // --- Run Optimization ---
                double minObjectiveValue; // Stores objective value at minimum
                int nlopt_rc = nlopt_optimize(opt, fitParams, &minObjectiveValue);

                // --- Process Results ---
                if(nlopt_rc < 0){ // Check for NLopt errors
                    printf("F#%d: NLopt optimization failed with error code %d\n", currFidx, nlopt_rc);
                    currentPeakCount = 0; // Indicate no successful fit
                    free(fitParams); // Free buffer
                } else {
                    // Optimization succeeded (or stopped due to limits)
                    sendFitParams = (double*)malloc(currentPeakCount * 5 * sizeof(double)); // Format: Amp, BG, Mix, Cen, Sig
                    check(!sendFitParams, "pkFit: Malloc failed for sendFitParams buffer");
                    double globalBG = fitParams[nFitParams - 1]; // Get fitted global BG

                    for(int p = 0; p < currentPeakCount; ++p){
                         sendFitParams[p * 5 + 0] = fitParams[p * 4 + 0]; // Amplitude
                         sendFitParams[p * 5 + 1] = globalBG;             // Background (Global)
                         sendFitParams[p * 5 + 2] = fitParams[p * 4 + 1]; // Mix factor
                         sendFitParams[p * 5 + 3] = fitParams[p * 4 + 2]; // Center
                         sendFitParams[p * 5 + 4] = fitParams[p * 4 + 3]; // Sigma
                    }
                    free(fitParams); // Free the raw optimization parameter buffer
                }

                // Cleanup NLopt resources
                nlopt_destroy(opt);
                free(lowerBounds);
                free(upperBounds);
            } // End if (currentPeakCount > 0 && pks != NULL)

            if(pks) {
                free(pks); // Free the peak candidate array
            }
        } // End if(pkFit)
        t_fit_cpu = get_wall_time_ms() - t_fit_start;

        // --- Write Peak Fit Results ---
        double t_writefit_start = get_wall_time_ms();
        // Check if fitting was enabled, successful, parameters exist, and file open
        if(pkFit && currentPeakCount > 0 && sendFitParams != NULL && fFit){
            // Write the formatted parameters (Amp, BG, Mix, Cen, Sig per peak)
            size_t written = fwrite(sendFitParams, sizeof(double), currentPeakCount * 5, fFit);
            if (written != (size_t)currentPeakCount * 5) {
                 fprintf(stderr, "Warn: Failed write fit frame %d (wrote %zu/%d): %s\n",
                           currFidx, written, currentPeakCount * 5, strerror(errno));
            } else {
                fflush(fFit); // Flush after writing fit results
            }
        }
        // Free the formatted fit parameter buffer AFTER writing/sending it
        if (sendFitParams != NULL) {
             free(sendFitParams);
             sendFitParams = NULL; // Avoid double free if loop exits unexpectedly
        }
        t_writefit_cpu = get_wall_time_ms() - t_writefit_start;

        // --- Free received data buffer ---
        // Free the pinned host buffer for the received chunk AFTER all processing is done
        gpuWarnchk(cudaFreeHost(chunk.data));

        // --- Timing and Output ---
        t_end_loop = get_wall_time_ms();
        t_loop_cpu = t_end_loop - t_start_loop; // Total wall time for the loop iteration

        // Print detailed timing information for the frame
        printf("F#%d: Ttl:%.2f| QPop:%.2f GPU(Proc:%.2f Int:%.2f Prof:%.2f D2H:%.2f) CPU(Wr2D:%.2f Wr1D:%.2f Fit:%.2f WrFit:%.2f)\n",
               currFidx, t_loop_cpu, t_qp_cpu,
               t_proc_gpu, t_integ_gpu, t_prof_gpu, t_d2h_gpu,
               t_write2d_cpu, t_write1d_cpu, t_fit_cpu, t_writefit_cpu);
        fflush(stdout); // Ensure output is visible immediately

        frameCounter++;
    } // ======================== End Main Processing Loop ========================

    printf("Processing loop finished (keep_running=%d). Processed %d frames. Cleaning up...\n", keep_running, frameCounter);

    // --- Cleanup ---

    // Close output files
    if(fLineout) fclose(fLineout);
    if(fFit) fclose(fFit);
    if(f2D) fclose(f2D); //

    // Free host memory allocations
    if(hAvgDark) free(hAvgDark);
    if(hDarkInT) free(hDarkInT);
    if(hDarkIn) free(hDarkIn);
    // chunk.data is freed inside the loop or by queue_destroy
    if(hIntArrFrame) gpuWarnchk(cudaFreeHost(hIntArrFrame)); // Free pinned host memory
    if(hPerFrame) gpuWarnchk(cudaFreeHost(hPerFrame));       // Free pinned host memory
    if(h_int1D) gpuWarnchk(cudaFreeHost(h_int1D));           // Free pinned host memory
    if(hR) free(hR);
    if(hEta) free(hEta);
    if(hLineout) free(hLineout);
    if(hEtaLo) free(hEtaLo);
    if(hEtaHi) free(hEtaHi);
    if(hRLo) free(hRLo);
    if(hRHi) free(hRHi);
    if(hMapMask) free(hMapMask); // Free host mask buffer if it was allocated

    UnmapBins(); // Unmap memory-mapped input files (Map.bin, nMap.bin)

    // Free GPU memory (use gpuWarnchk for cleanup)
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
    // <<< ADDED: Free persistent transform buffers >>>
    if(g_dTempTransformBuf1) gpuWarnchk(cudaFree(g_dTempTransformBuf1));
    if(g_dTempTransformBuf2) gpuWarnchk(cudaFree(g_dTempTransformBuf2));
    // <<< END ADDED >>>

    // Destroy CUDA events
    gpuWarnchk(cudaEventDestroy(ev_proc_start)); gpuWarnchk(cudaEventDestroy(ev_proc_stop));
    gpuWarnchk(cudaEventDestroy(ev_integ_start)); gpuWarnchk(cudaEventDestroy(ev_integ_stop));
    gpuWarnchk(cudaEventDestroy(ev_prof_start)); gpuWarnchk(cudaEventDestroy(ev_prof_stop));
    gpuWarnchk(cudaEventDestroy(ev_d2h_start)); gpuWarnchk(cudaEventDestroy(ev_d2h_stop));

    // --- Shutdown Accept Thread Gracefully ---
    printf("Attempting to shut down network acceptor thread...\n");
    if (server_fd >= 0) {
         printf("Closing server listening socket %d...\n", server_fd);
         shutdown(server_fd, SHUT_RDWR); // Shut down read/write ends first
         close(server_fd); // Close the socket file descriptor
         server_fd = -1; // Mark as closed
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
    // --- End Shutdown Accept Thread ---

    // Destroy queue
    queue_destroy(&process_queue);

    printf("[%s] - Exiting cleanly.\n", argv[0]);
    return 0;
}