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
//  - Optional saving of results
//
// Example compile command (adjust paths and architecture flags):
/*
  source ~/.MIDAS/paths # Set up environment variables for libraries
  ~/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStream.cu -o bin/IntegratorFitPeaksGPUStream \
  -Xcompiler -g -arch sm_90 -gencode=arch=compute_90,code=sm_90 \
  -I/path/to/nlopt/include -L/path/to/nlopt/lib -lnlopt \
  -I/path/to/blosc/include -L/path/to/blosc/lib64 -lblosc2 \
  -I/path/to/hdf5/include -L/path/to/hdf5/lib -lhdf5 -lhdf5_hl -lz -ldl -lm -lpthread \
  -I/path/to/libzip/include -L/path/to/libzip/lib64 -lzip \
  -O3
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
// #define PORTSENDFIT 5001        // Removed - not used
// #define PORTSENDLINEOUT 5002    // Removed - not used
#define PORT 5000               // Port for receiving image data
#define MAX_CONNECTIONS 10      // Max simultaneous client connections
#define MAX_QUEUE_SIZE 100      // Max image frames buffered before processing
#define HEADER_SIZE sizeof(uint16_t) // Size of frame number header
#define BYTES_PER_PIXEL 8       // Size of each pixel (int64_t)
#define MAX_FILENAME_LENGTH 1024
#define THREADS_PER_BLOCK_TRANSFORM 512 // CUDA block size for transform/processing
#define THREADS_PER_BLOCK_INTEGRATE 512 // CUDA block size for integration
#define THREADS_PER_BLOCK_PROFILE 256  // CUDA block size for 1D profile reduction
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
                continue;
            }
            perror("Accept failed");
            free(client_socket_ptr);
            sleep(1);
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
            pthread_detach(thread_id);
        }
    }
    printf("Accept thread exiting.\n");
    return NULL;
}

// --- Data Sending Functions Removed ---
// Functions send_data_to_server, send_fit_result, send_lineouts removed as requested.

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
    int fd_m=-1,fd_n=-1;
    struct stat s_m,s_n;
    const char* f_m="Map.bin";
    const char* f_n="nMap.bin";
    fd_m=open(f_m,O_RDONLY);
    check(fd_m<0,"open %s fail:%s",f_m,strerror(errno));
    check(fstat(fd_m,&s_m)<0,"stat %s fail:%s",f_m,strerror(errno));
    szPxList=s_m.st_size;
    check(szPxList==0,"%s empty",f_m);
    printf("Map '%s': %lld bytes\n",f_m,(long long)szPxList);
    pxList=(struct data*)mmap(NULL,szPxList,PROT_READ,MAP_SHARED,fd_m,0);
    check(pxList==MAP_FAILED,"mmap %s fail:%s",f_m,strerror(errno));
    close(fd_m);
    fd_n=open(f_n,O_RDONLY);
    check(fd_n<0,"open %s fail:%s",f_n,strerror(errno));
    check(fstat(fd_n,&s_n)<0,"stat %s fail:%s",f_n,strerror(errno));
    szNPxList=s_n.st_size;
    check(szNPxList==0,"%s empty",f_n);
    printf("nMap '%s': %lld bytes\n",f_n,(long long)szNPxList);
    nPxList=(int*)mmap(NULL,szNPxList,PROT_READ,MAP_SHARED,fd_n,0);
    check(nPxList==MAP_FAILED,"mmap %s fail:%s",f_n,strerror(errno));
    close(fd_n);
    printf("Mapped pixel data files.\n");
    fflush(stdout);
    return 1;
}
void UnmapBins() {
    if(pxList&&pxList!=MAP_FAILED){
        munmap(pxList,szPxList);
        pxList=NULL;
        printf("Unmapped Map.bin\n");
    }
    if(nPxList&&nPxList!=MAP_FAILED){
        munmap(nPxList,szNPxList);
        nPxList=NULL;
        printf("Unmapped nMap.bin\n");
    }
}

// --- Binning Setup ---
static inline void REtaMapper(double Rmin,double EtaMin,int nEta,int nR,double EtaStep,double RStep,double *EtaLo,double *EtaHi,double *RLo,double *RHi) {
    for(int i=0; i<nEta; ++i){
        EtaLo[i]=EtaStep*i+EtaMin;
        EtaHi[i]=EtaStep*(i+1)+EtaMin;
    }
    for(int i=0; i<nR; ++i){
        RLo[i]=RStep*i+Rmin;
        RHi[i]=RStep*(i+1)+Rmin;
    }
}

// --- Sequential CPU Image Transformation (for dark processing) ---
static inline void DoImageTransformationsSequential(int Nopt, const int Topt[MAX_TRANSFORM_OPS], const int64_t *In, int64_t *Out, int NY, int NZ) {
    size_t N=(size_t)NY*NZ;
    size_t B=N*sizeof(int64_t);
    bool any=false;
    if(Nopt>0){
        for(int i=0;i<Nopt;++i){
            if(Topt[i]<0||Topt[i]>3){
                fprintf(stderr,"CPU Err: Inv opt %d\n",Topt[i]);
                return;
            }
            if(Topt[i]!=0) {
                any=true;
            }
        }
    }
    if(!any){
        if(Out!=In) {
            memcpy(Out,In,B);
        }
        return;
    }
    int64_t *tmp=(int64_t*)malloc(B);
    if(!tmp){
        fprintf(stderr,"CPU Err: Alloc tmp fail\n");
        if(Out!=In) {
            memcpy(Out,In,B);
        }
        return;
    }
    const int64_t* rB=NULL;
    int64_t* wB=NULL;
    int cY=NY;
    int cZ=NZ;
    for(int i=0;i<Nopt;++i){
        int opt=Topt[i];
        size_t cB=(size_t)cY*cZ*sizeof(int64_t);
        if(i==0){
            rB=In;
            wB=tmp;
        } else if(i%2==1){
            rB=tmp;
            wB=Out;
        } else {
            rB=Out;
            wB=tmp;
        }
        int nY=cY;
        int nZ=cZ;
        switch(opt){
            case 0:
                if(wB!=rB) {
                    memcpy(wB,rB,cB);
                }
                break;
            case 1:
                for(int l=0;l<cZ;++l) {
                    for(int k=0;k<cY;++k) {
                        wB[l*cY+k]=rB[l*cY+(cY-1-k)];
                    }
                }
                break;
            case 2:
                for(int l=0;l<cZ;++l) {
                    for(int k=0;k<cY;++k) {
                        wB[l*cY+k]=rB[(cZ-1-l)*cY+k];
                    }
                }
                break;
            case 3:
                if(cY!=cZ){
                    fprintf(stderr,"CPU Warn: Skip Tpose %dx%d st %d\n",cY,cZ,i);
                    if(wB!=rB) {
                        memcpy(wB,rB,cB);
                    }
                } else {
                    nY=cZ;
                    nZ=cY;
                    for(int l=0;l<nZ;++l) {
                        for(int k=0;k<nY;++k) {
                            wB[l*nY+k]=rB[k*cY+l];
                        }
                    }
                }
                break;
        }
        cY=nY;
        cZ=nZ;
    }
    if(Nopt%2!=0){
        size_t fB=(size_t)cY*cZ*sizeof(int64_t);
        if(fB>B){
            fprintf(stderr,"CPU Err:Final>Orig\n");
            fB=B;
        }
        memcpy(Out,tmp,fB);
    } else {
        if((size_t)cY*cZ!=N) {
            fprintf(stderr,"CPU Warn:Final!=Orig\n");
        }
    }
    free(tmp);
}

// --- String Utility ---
static inline int StartsWith(const char *a, const char *b) {
    // Check if string 'a' starts with string 'b'
    return (strncmp(a, b, strlen(b)) == 0);
}

// --- GPU Kernels ---
__global__ void integrate_noMapMask(double px, double Lsd, size_t bigArrSize, int Normalize, int sumImages, int frameIdx,
                                    const struct data * dPxList, const int *dNPxList,
                                    const double *dRBinsLow, const double *dRBinsHigh,
                                    const double *dEtaBinsLow, const double *dEtaBinsHigh,
                                    int NrPixelsY, int NrPixelsZ, // Pass dimensions
                                    const double *dImage, double *dIntArrPerFrame,
                                    double *dPerFrameArr, double *dSumMatrix)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ; // Use parameters
    if (idx < bigArrSize) {
        double Intensity = 0.0;
        double totArea = 0.0;
        struct data ThisVal;
        long long nPixels = 0;
        long long dataPos = 0;
        const size_t nPxListIndex = 2 * idx;
        if (nPxListIndex + 1 < 2 * bigArrSize) { // Basic bounds check
            nPixels = dNPxList[nPxListIndex];
            dataPos = dNPxList[nPxListIndex + 1];
        } else {
            return;
        }
        if (nPixels < 0 || dataPos < 0) {
            return;
        }
        for (long long l = 0; l < nPixels; l++) {
            ThisVal = dPxList[dataPos + l];
            if (ThisVal.y < 0 || ThisVal.y >= NrPixelsY || ThisVal.z < 0 || ThisVal.z >= NrPixelsZ) { // Use parameters
                continue;
            }
            long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y; // Use parameter
            if (testPos < 0 || testPos >= totalPixels) {
                 continue;
            }
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
        if (frameIdx == 0 && dPerFrameArr) {
            // Determine nEtaBins and nRBins from bigArrSize and one dimension (assuming consistent mapping)
            // This requires NrPixelsZ to be passed or known, assume it's equivalent to nEtaBins if square R-Eta grid
            int nEtaBins = bigArrSize / (bigArrSize / NrPixelsZ); // Assuming NrPixelsZ corresponds to Eta bins in mapping
            int nRBins = bigArrSize / nEtaBins;
            int j = idx / nEtaBins; // Use calculated nEtaBins
            int k = idx % nEtaBins; // Use calculated nEtaBins
            if (j < nRBins && k < nEtaBins) {
                double RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5;
                double EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5;
                if ((3 * bigArrSize + idx) < (bigArrSize * 4)) {
                    dPerFrameArr[0 * bigArrSize + idx] = RMean;
                    dPerFrameArr[1 * bigArrSize + idx] = rad2deg * atan(RMean * px / Lsd);
                    dPerFrameArr[2 * bigArrSize + idx] = EtaMean;
                    dPerFrameArr[3 * bigArrSize + idx] = totArea;
                }
            }
        }
    }
}

__global__ void integrate_MapMask(double px, double Lsd, size_t bigArrSize, int Normalize, int sumImages, int frameIdx,
                                  size_t mapMaskWordCount, const int *dMapMask,
                                  int nRBins, int nEtaBins, // Pass dimensions explicitly
                                  int NrPixelsY, int NrPixelsZ, // Pass dimensions
                                  const struct data * dPxList, const int *dNPxList,
                                  const double *dRBinsLow, const double *dRBinsHigh,
                                  const double *dEtaBinsLow, const double *dEtaBinsHigh,
                                  const double *dImage, double *dIntArrPerFrame,
                                  double *dPerFrameArr, double *dSumMatrix)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ; // Use parameters
    if (idx < bigArrSize) {
        double Intensity = 0.0;
        double totArea = 0.0;
        struct data ThisVal;
        long long nPixels = 0;
        long long dataPos = 0;
        const size_t nPxListIndex = 2 * idx;
        if (nPxListIndex + 1 < 2 * bigArrSize) {
            nPixels = dNPxList[nPxListIndex];
            dataPos = dNPxList[nPxListIndex + 1];
        } else {
            return;
        }
        if (nPixels < 0 || dataPos < 0) {
            return;
        }
        for (long long l = 0; l < nPixels; l++) {
            ThisVal = dPxList[dataPos + l];
            if (ThisVal.y < 0 || ThisVal.y >= NrPixelsY || ThisVal.z < 0 || ThisVal.z >= NrPixelsZ) { // Use parameters
                continue;
            }
            long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y; // Use parameter
            if (testPos < 0 || testPos >= totalPixels) {
                continue;
            }
            size_t wordIndex = testPos / 32;
            if (wordIndex < mapMaskWordCount) {
                if (TestBit(dMapMask, testPos)) {
                    continue; // Masked pixel
                }
            } else {
                 continue; // Out of mask bounds
            }
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
        if (frameIdx == 0 && dPerFrameArr) {
             int j = idx / nEtaBins; // Use passed nEtaBins
             int k = idx % nEtaBins; // Use passed nEtaBins
            if (j < nRBins && k < nEtaBins) { // Use passed nRBins
                double RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5;
                double EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5;
                if ((3 * bigArrSize + idx) < (bigArrSize * 4)) {
                    dPerFrameArr[0 * bigArrSize + idx] = RMean;
                    dPerFrameArr[1 * bigArrSize + idx] = rad2deg * atan(RMean * px / Lsd);
                    dPerFrameArr[2 * bigArrSize + idx] = EtaMean;
                    dPerFrameArr[3 * bigArrSize + idx] = totArea;
                }
            }
        }
    }
}

__global__ void sequential_transform_kernel(const int64_t *r, int64_t *w, int cY, int cZ, int nY, int nZ, int opt) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int yo = i % nY;
        const int zo = i / nY;
        int ys = -1;
        int zs = -1;
        switch(opt){
            case 0: ys=yo; zs=zo; break;
            case 1: ys=cY-1-yo; zs=zo; break;
            case 2: ys=yo; zs=cZ-1-zo; break;
            case 3: ys=zo; zs=yo; break;
            default: return;
        }
        if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
            w[i] = r[(size_t)zs * cY + ys];
        } else {
            w[i] = 0;
        }
    }
}

__global__ void final_transform_process_kernel(const int64_t *r, double *o, const double *d, int cY, int cZ, int nY, int nZ, int opt, bool sub) {
    const size_t N = (size_t)nY * nZ;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int yo = i % nY;
        const int zo = i / nY;
        int ys = -1;
        int zs = -1;
        switch(opt){
            case 0: ys=yo; zs=zo; break;
            case 1: ys=cY-1-yo; zs=zo; break;
            case 2: ys=yo; zs=cZ-1-zo; break;
            case 3: ys=zo; zs=yo; break;
            default: o[i]=0.0; return;
        }
        double pv=0.0;
        if (ys >= 0 && ys < cY && zs >= 0 && zs < cZ) {
            const int64_t rv = r[(size_t)zs * cY + ys];
            pv = (double)rv;
            if (sub && d) {
                pv -= d[i];
            }
        }
        o[i] = pv;
    }
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
    double * sArea = &sdata[blockDim.x / 32];
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
    __syncthreads();
    double mySumIntArea = 0.0;
    double mySumArea = 0.0;
    for (int eta_bin = tid; eta_bin < nEtaBins; eta_bin += blockDim.x) {
        size_t idx2d = (size_t)r_bin * nEtaBins + eta_bin;
        double area = d_PerFrameArr[3 * bigArrSize + idx2d];
        if (area > AREA_THRESHOLD) {
             mySumIntArea += d_IntArrPerFrame[idx2d] * area;
             mySumArea += area;
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
void ProcessImageGPU(const int64_t *hRaw, double *dProc, const double *dAvgDark, int Nopt, const int Topt[MAX_TRANSFORM_OPS], int NY, int NZ, bool doSub) {
    const size_t N=(size_t)NY*NZ;
    const size_t B64=N*sizeof(int64_t);
    const int TPB=THREADS_PER_BLOCK_TRANSFORM;
    bool anyT=false;
    if(Nopt>0){
        for(int i=0;i<Nopt;++i){
            if(Topt[i]<0||Topt[i]>3){
                fprintf(stderr,"GPU Err: Inv opt %d\n",Topt[i]);
                gpuErrchk(cudaMemset(dProc,0,N*sizeof(double)));
                return;
            }
            if(Topt[i]!=0){
                anyT=true;
            }
        }
    }
    if(!anyT){
        int64_t *dTmp=NULL;
        gpuErrchk(cudaMalloc(&dTmp,B64));
        gpuErrchk(cudaMemcpy(dTmp,hRaw,B64,cudaMemcpyHostToDevice));
        unsigned long long nBUL=(N+TPB-1)/TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX,cudaDevAttrMaxGridDimX,0));
        if(nBUL>(unsigned long long)mGDX){ // Cast mGDX for comparison
            fprintf(stderr,"Blk %llu>max %d\n",nBUL,mGDX);
            exit(1);
        }
        dim3 nB((unsigned)nBUL);
        dim3 th(TPB);
        process_direct_kernel<<<nB,th>>>(dTmp,dProc,dAvgDark,N,doSub);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaFree(dTmp));
        return;
    }
    int64_t *d_b1=NULL,*d_b2=NULL;
    gpuErrchk(cudaMalloc(&d_b1,B64));
    gpuErrchk(cudaMalloc(&d_b2,B64));
    gpuErrchk(cudaMemcpy(d_b1,hRaw,B64,cudaMemcpyHostToDevice));
    const int64_t* rP=NULL;
    int64_t* wP=NULL;
    int cY=NY;
    int cZ=NZ;
    for(int i=0;i<Nopt-1;++i){
        int opt=Topt[i];
        if(i%2==0){
            rP=d_b1;
            wP=d_b2;
        } else {
            rP=d_b2;
            wP=d_b1;
        }
        int nY=cY;
        int nZ=cZ;
        if(opt==3){
            if(cY!=cZ){
                fprintf(stderr,"GPU Warn: Skip Tpose %dx%d st %d\n",cY,cZ,i);
                opt=0;
            } else {
                nY=cZ;
                nZ=cY;
            }
        }
        size_t sON=(size_t)nY*nZ;
        unsigned long long nBUL=(sON+TPB-1)/TPB;
        int mGDX;
        gpuErrchk(cudaDeviceGetAttribute(&mGDX,cudaDevAttrMaxGridDimX,0));
        if(nBUL>(unsigned long long)mGDX){
            fprintf(stderr,"Blk %llu>max %d\n",nBUL,mGDX);
            cudaFree(d_b1);
            cudaFree(d_b2);
            exit(1);
        }
        dim3 nB((unsigned)nBUL);
        dim3 th(TPB);
        sequential_transform_kernel<<<nB,th>>>(rP,wP,cY,cZ,nY,nZ,opt);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cY=nY;
        cZ=nZ;
    }
    int fOpt=Topt[Nopt-1];
    if((Nopt-1)%2==0){
        rP=d_b1;
    } else {
        rP=d_b2;
    }
    int nY=cY;
    int nZ=cZ;
    if(fOpt==3){
        if(cY!=cZ){
            fprintf(stderr,"GPU Warn: Skip Tpose %dx%d final st %d\n",cY,cZ,Nopt-1);
            fOpt=0;
        } else {
            nY=cZ;
            nZ=cY;
        }
    }
    size_t fON=(size_t)nY*nZ;
    if(fON!=N){
        fprintf(stderr,"GPU Warn: Final N %zu!=Orig %zu\n",fON,N);
    }
    unsigned long long nBUL=(fON+TPB-1)/TPB;
    int mGDX;
    gpuErrchk(cudaDeviceGetAttribute(&mGDX,cudaDevAttrMaxGridDimX,0));
    if(nBUL>(unsigned long long)mGDX){
        fprintf(stderr,"Blk %llu>max %d\n",nBUL,mGDX);
        cudaFree(d_b1);
        cudaFree(d_b2);
        exit(1);
    }
    dim3 nB((unsigned)nBUL);
    dim3 th(TPB);
    final_transform_process_kernel<<<nB,th>>>(rP,dProc,dAvgDark,cY,cZ,nY,nZ,fOpt,doSub);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_b1));
    gpuErrchk(cudaFree(d_b2));
}

// --- Peak Fitting Data Structures and Functions ---
typedef struct { int nrBins; const double *R; const double *Int; } dataFit;
typedef struct { int index; double radius; double intensity; } Peak;
static double problem_function_global_bg(unsigned n, const double *x, double *grad, void *fdat) {
    if(grad){
        for(unsigned i=0;i<n;++i){
            grad[i]=0.0;
        }
    }
    const dataFit *d=(const dataFit*)fdat;
    const int Np=d->nrBins;
    const double *Rs=d->R;
    const double *Is=d->Int;
    const int nP=(n-1)/4;
    if(nP<=0||(4*nP+1!=n)){
        fprintf(stderr,"Obj Err: Inv params %u\n",n);
        return INFINITY;
    }
    const double bg_g=x[n-1];
    double t_err_sq=0.0;
    double *cI=(double*)calloc(Np, sizeof(double)); // Use sizeof(double)
    if(!cI){
        fprintf(stderr,"Obj Err: Alloc fail\n");
        return INFINITY;
    }
    for(int pN=0;pN<nP;++pN){
        double a=x[pN*4+0];
        double m=fmax(0.0,fmin(1.0,x[pN*4+1]));
        double c=x[pN*4+2];
        double s=fmax(1e-9,x[pN*4+3]);
        for(int i=0;i<Np;++i){
            double df=Rs[i]-c;
            double df2=df*df;
            double s2=s*s;
            double g=exp(-df2/(2.0*s2))/(s*sqrt(2.0*M_PI));
            double l=(1.0/(M_PI*s))/(1.0+df2/s2);
            cI[i]+=a*(m*g+(1.0-m)*l);
        }
    }
    for(int i=0;i<Np;++i){
        cI[i]+=bg_g;
        double err=cI[i]-Is[i];
        t_err_sq+=err*err;
    }
    free(cI);
    return t_err_sq;
}
void smoothData(const double *in, double *out, int N, int W) {
    if(W<3||W%2==0){
        memcpy(out,in,N*sizeof(double));
        return;
    }
    int H=W/2;
    double *c=(double*)malloc(W*sizeof(double));
    check(!c,"Malloc fail smooth c");
    double norm=0.0;
    switch(W){
        case 5: norm=35.0; c[0]=-3; c[1]=12; c[2]=17; c[3]=12; c[4]=-3; break;
        case 7: norm=21.0; c[0]=-2; c[1]= 3; c[2]= 6; c[3]= 7; c[4]= 6; c[5]= 3; c[6]=-2; break;
        case 9: norm=231.0; c[0]=-21; c[1]=14; c[2]=39; c[3]=54; c[4]=59; c[5]=54; c[6]=39; c[7]=14; c[8]=-21; break;
        default: fprintf(stderr,"Smooth Warn: Unsup Win %d\n",W); memcpy(out,in,N*sizeof(double)); free(c); return;
    }
    for(int i=0;i<W;++i){
        c[i]/=norm;
    }
    for(int i=0;i<N;++i){
        if(i<H||i>=N-H){
            out[i]=in[i];
        }else{
            double s=0;
            for(int j=0;j<W;++j){
                s+=c[j]*in[i-H+j];
            }
            out[i]=s;
        }
    }
    free(c);
}
int findPeaks(const double *d, const double *r, int N, Peak **fP, double minH, int minD) {
    if(N<3){
        *fP=NULL;
        return 0;
    }
    int maxP=N/2+1;
    Peak*pP=(Peak*)malloc(maxP*sizeof(Peak));
    check(!pP,"Malloc fail pP");
    int pC=0;
    for(int i=1;i<N-1;++i){
        if(d[i]>d[i-1]&&d[i]>d[i+1]&&d[i]>=minH){
            if(pC<maxP){
                pP[pC++]=(Peak){i,r[i],d[i]};
            } else {
                fprintf(stderr,"Peak find warn: Exceed maxP\n");
                break;
            }
        }
    }
    if(pC==0||minD<=1){
        *fP=pP;
        return pC;
    }
    Peak*filtP=(Peak*)malloc(pC*sizeof(Peak));
    check(!filtP,"Malloc fail filtP");
    int filtC=0;
    bool*isS=(bool*)calloc(pC,sizeof(bool));
    check(!isS,"Calloc fail isS");
    for(int i=0;i<pC;++i){
        if(isS[i]) {
            continue;
        }
        filtP[filtC++]=pP[i];
        for(int j=0;j<pC;++j){
            if(i==j||isS[j]) {
                continue;
            }
            int dist=abs(pP[i].index-pP[j].index);
            if(dist<minD){
                if(pP[j].intensity<=pP[i].intensity) {
                    isS[j]=true;
                } else {
                    isS[i]=true;
                    filtC--; // Decrement count as we removed the current 'i' peak
                    break; // Exit inner loop as 'i' is suppressed
                }
            }
        }
    }
    free(pP);
    free(isS);
    Peak*finalP=(Peak*)realloc(filtP,filtC*sizeof(Peak));
    *fP=(finalP!=NULL)?finalP:filtP;
    return filtC;
}


// =========================================================================
// ============================ MAIN FUNCTION ============================
// =========================================================================
int main(int argc, char *argv[]){
    if (argc < 2){
        printf("Usage: %s ParamFN [DarkAvgFN]\n Args:\n  ParamFN: Path to parameter file.\n  DarkAvgFN: Optional path to dark frame file (binary int64_t).\n", argv[0]);
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
    double RMax=0,RMin=0,RBinSize=0,EtaMax=0,EtaMin=0,EtaBinSize=0,Lsd=0,px=0;
    int NrPixelsY=0,NrPixelsZ=0,Normalize=1;
    int nEtaBins=0,nRBins=0;
    char *ParamFN=argv[1];
    FILE *pF=fopen(ParamFN,"r");
    check(!pF,"Failed open param file: %s",ParamFN);
    char line[4096],dum[4096];
    const char *s;
    int Nopt=0;
    long long GapI=0,BadPxI=0;
    int Topt[MAX_TRANSFORM_OPS]={0};
    int mkMap=0, sumI=0, doSm=0, multiP=0, pkFit=0, nSpecP=0, wr2D=0;
    double pkLoc[MAX_PEAK_LOCATIONS];
    while(fgets(line,4096,pF)){
        if(line[0]=='#'||isspace(line[0])||strlen(line)<3){
            continue;
        }
        s="EtaBinSize "; if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&EtaBinSize);
        s="RBinSize ";   if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&RBinSize);
        s="RMax ";       if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&RMax);
        s="RMin ";       if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&RMin);
        s="EtaMax ";     if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&EtaMax);
        s="EtaMin ";     if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&EtaMin);
        s="Lsd ";        if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&Lsd);
        s="px ";         if(StartsWith(line,s)) sscanf(line,"%s %lf",dum,&px);
        s="NrPixelsY ";  if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&NrPixelsY);
        s="NrPixelsZ ";  if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&NrPixelsZ);
        s="NrPixels ";   if(StartsWith(line,s)){ sscanf(line,"%s %d",dum,&NrPixelsY); NrPixelsZ=NrPixelsY; }
        s="Normalize ";  if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&Normalize);
        s="GapIntensity "; if(StartsWith(line,s)){ sscanf(line,"%s %lld",dum,&GapI); mkMap=1; }
        s="BadPxIntensity "; if(StartsWith(line,s)){ sscanf(line,"%s %lld",dum,&BadPxI); mkMap=1; }
        s="ImTransOpt "; if(StartsWith(line,s)){ if(Nopt<MAX_TRANSFORM_OPS) sscanf(line,"%s %d",dum,&Topt[Nopt++]); else printf("Warn:Max %d ImTransOpt\n",MAX_TRANSFORM_OPS); }
        s="SumImages ";  if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&sumI);
        s="Write2D ";    if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&wr2D);
        s="DoSmoothing ";if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&doSm);
        s="MultiplePeaks ";if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&multiP);
        s="DoPeakFit ";  if(StartsWith(line,s)) sscanf(line,"%s %d",dum,&pkFit);
        s="PeakLocation ";if(StartsWith(line,s)){ if(nSpecP<MAX_PEAK_LOCATIONS) sscanf(line,"%s %lf",dum,&pkLoc[nSpecP++]); else printf("Warn:Max %d PeakLoc\n",MAX_PEAK_LOCATIONS); multiP=1; pkFit=1; doSm=0; }
    }
    fclose(pF);
    check(NrPixelsY<=0||NrPixelsZ<=0,"NrPixelsY/Z invalid");
    // global_NrPixelsY=NrPixelsY; // Store globally
    // global_NrPixelsZ=NrPixelsZ; // Store globally
    if(pkFit&&nSpecP>0) {
        multiP=1;
    }
    nRBins=(RBinSize>1e-9)?(int)ceil((RMax-RMin)/RBinSize):0;
    nEtaBins=(EtaBinSize>1e-9)?(int)ceil((EtaMax-EtaMin)/EtaBinSize):0;
    check(nRBins<=0||nEtaBins<=0,"Invalid bin params. R=%d, Eta=%d",nRBins,nEtaBins);
    size_t bigArrSize=(size_t)nRBins*nEtaBins;
    printf("Params: R:[%.2f..%.2f],%d(%.3f); Eta:[%.2f..%.2f],%d(%.3f)\n",RMin,RMax,nRBins,RBinSize,EtaMin,EtaMax,nEtaBins,EtaBinSize);
    printf(" Det:%dx%d Lsd=%.2f px=%.4f\n",NrPixelsY,NrPixelsZ,Lsd,px);
    printf(" T(%d):",Nopt);
    for(int i=0;i<Nopt;++i){
        printf(" %d",Topt[i]);
    }
    printf("\n");
    printf(" Norm=%d Sum=%d Wr2D=%d\n",Normalize,sumI,wr2D);
    printf(" Fit=%d MultiP=%d Smooth=%d NSpecP=%d\n",pkFit,multiP,doSm,nSpecP);
    printf("Read Params: %.3f ms\n", get_wall_time_ms() - t_start_params);

    // --- Setup Bin Edges (Host) ---
    double *hEtaLo,*hEtaHi,*hRLo,*hRHi;
    hEtaLo=(double*)malloc(nEtaBins*sizeof(double));
    hEtaHi=(double*)malloc(nEtaBins*sizeof(double));
    hRLo=(double*)malloc(nRBins*sizeof(double));
    hRHi=(double*)malloc(nRBins*sizeof(double));
    check(!hEtaLo||!hEtaHi||!hRLo||!hRHi,"Alloc fail host bin edges");
    REtaMapper(RMin,EtaMin,nEtaBins,nRBins,EtaBinSize,RBinSize,hEtaLo,hEtaHi,hRLo,hRHi);

    // --- Host Memory Allocations ---
    double *hAvgDark=NULL;
    int64_t *hDarkInT=NULL,*hDarkIn=NULL;
    // hImageInT is allocated in handle_client using cudaMallocHost
    size_t totalPixels=(size_t)NrPixelsY*NrPixelsZ;
    size_t SizeFile=totalPixels*BYTES_PER_PIXEL;
    hAvgDark=(double*)calloc(totalPixels,sizeof(double));
    check(!hAvgDark,"Alloc fail hAvgDark");
    hDarkInT=(int64_t*)malloc(SizeFile);
    check(!hDarkInT,"Alloc fail hDarkInT");
    hDarkIn=(int64_t*)malloc(SizeFile);
    check(!hDarkIn,"Alloc fail hDarkIn");

    // --- Device Memory Allocations (Persistent) ---
    double *dAvgDark=NULL, *dProcessedImage=NULL, *d_int1D=NULL;
    int *dMapMask=NULL;
    size_t mapMaskWC=0;
    int *dNPxList=NULL;
    struct data *dPxList=NULL;
    double *dSumMatrix=NULL, *dIntArrFrame=NULL, *dPerFrame=NULL;
    double *dEtaLo=NULL,*dEtaHi=NULL,*dRLo=NULL,*dRHi=NULL;
    bool darkSubEnabled=(argc>2);
    gpuErrchk(cudaMalloc(&dProcessedImage,totalPixels*sizeof(double)));
    gpuErrchk(cudaMalloc(&dPxList,szPxList));
    gpuErrchk(cudaMalloc(&dNPxList,szNPxList));
    gpuErrchk(cudaMemcpy(dPxList,pxList,szPxList,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dNPxList,nPxList,szNPxList,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dIntArrFrame,bigArrSize*sizeof(double)));
    gpuErrchk(cudaMalloc(&dPerFrame,bigArrSize*4*sizeof(double)));
    if(sumI){
        gpuErrchk(cudaMalloc(&dSumMatrix,bigArrSize*sizeof(double)));
        gpuErrchk(cudaMemset(dSumMatrix,0,bigArrSize*sizeof(double)));
    }
    gpuErrchk(cudaMalloc(&dEtaLo,nEtaBins*sizeof(double)));
    gpuErrchk(cudaMalloc(&dEtaHi,nEtaBins*sizeof(double)));
    gpuErrchk(cudaMalloc(&dRLo,nRBins*sizeof(double)));
    gpuErrchk(cudaMalloc(&dRHi,nRBins*sizeof(double)));
    gpuErrchk(cudaMemcpy(dEtaLo,hEtaLo,nEtaBins*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEtaHi,hEtaHi,nEtaBins*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRLo,hRLo,nRBins*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dRHi,hRHi,nRBins*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_int1D, nRBins * sizeof(double)));

    // --- Process Dark Frame ---
    double t_start_dark = get_wall_time_ms();
    int nrdone=0;
    int *hMapMask=NULL;
    if(darkSubEnabled){
        char* darkFN=argv[2];
        FILE* fD=fopen(darkFN,"rb");
        check(!fD,"Failed open dark: %s",darkFN);
        fseek(fD,0,SEEK_END);
        size_t szD=ftell(fD);
        rewind(fD);
        int nFD=szD/SizeFile;
        check(nFD==0||szD%SizeFile!=0,"Dark %s empty/bad size (%zu), %d frames",darkFN,SizeFile,nFD);
        printf("Read dark: %s, Frames: %d.\n",darkFN,nFD);
        for(int i=0;i<nFD;++i){
            check(fread(hDarkInT,1,SizeFile,fD)!=SizeFile,"Read fail dark fr %d",i);
            DoImageTransformationsSequential(Nopt,Topt,hDarkInT,hDarkIn,NrPixelsY,NrPixelsZ); // CPU transform dark
            if(mkMap==1&&i==0){
                mapMaskWC=totalPixels/32+1;
                hMapMask=(int*)calloc(mapMaskWC,sizeof(int));
                check(!hMapMask,"Alloc host mask fail");
                nrdone=0;
                for(size_t j=0;j<totalPixels;++j){
                    if(hDarkIn[j]==GapI||hDarkIn[j]==BadPxI){
                        SetBit(hMapMask,j);
                        nrdone++;
                    }
                }
                printf("Mask gen: %d masked\n",nrdone);
                gpuErrchk(cudaMalloc(&dMapMask,mapMaskWC*sizeof(int)));
                gpuErrchk(cudaMemcpy(dMapMask,hMapMask,mapMaskWC*sizeof(int),cudaMemcpyHostToDevice));
                mkMap=0;
            }
            for(size_t j=0;j<totalPixels;++j){
                hAvgDark[j]+=(double)hDarkIn[j]; // Accumulate transformed dark
            }
        }
        fclose(fD);
        if(nFD>0){
            for(size_t j=0;j<totalPixels;++j){
                hAvgDark[j]/=nFD; // Finalize average
            }
        }
        gpuErrchk(cudaMalloc(&dAvgDark,totalPixels*sizeof(double)));
        gpuErrchk(cudaMemcpy(dAvgDark,hAvgDark,totalPixels*sizeof(double),cudaMemcpyHostToDevice));
        printf("Avg dark copied GPU\n");
    } else {
        gpuErrchk(cudaMalloc(&dAvgDark,totalPixels*sizeof(double)));
        gpuErrchk(cudaMemset(dAvgDark,0,totalPixels*sizeof(double)));
        printf("No dark, use zeros GPU\n");
    }
    if(!dMapMask){
        mapMaskWC=0;
        printf("No mask used\n");
    }
    printf("Processed dark/mask: %.3f ms\n", get_wall_time_ms() - t_start_dark);

    // --- Network Setup ---
    CHUNK_SIZE=SizeFile;
    TOTAL_MSG_SIZE=HEADER_SIZE+CHUNK_SIZE;
    printf("Net: Expect %zu H + %zu D = %zu tot bytes\n",HEADER_SIZE,CHUNK_SIZE,TOTAL_MSG_SIZE);
    int server_fd;
    struct sockaddr_in server_addr;
    queue_init(&process_queue);
    check((server_fd=socket(AF_INET,SOCK_STREAM,0))==0,"Socket fail");
    int sock_opt=1; // Renamed variable
    check(setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR|SO_REUSEPORT,&sock_opt,sizeof(sock_opt)),"setsockopt fail");
    server_addr.sin_family=AF_INET;
    server_addr.sin_addr.s_addr=INADDR_ANY;
    server_addr.sin_port=htons(PORT);
    check(bind(server_fd,(struct sockaddr*)&server_addr,sizeof(server_addr))<0,"Bind fail port %d",PORT);
    check(listen(server_fd,MAX_CONNECTIONS)<0,"Listen fail");
    printf("Server listening on port %d\n",PORT);
    pthread_t accept_thread;
    check(pthread_create(&accept_thread,NULL,accept_connections,&server_fd)!=0,"Fail create accept thread");

    // --- Prepare for Main Loop ---
    FILE *fLineout=fopen("lineout.bin","ab");
    check(!fLineout,"Error open lineout.bin");
    FILE *fFit=NULL;
    if(pkFit){
        fFit=fopen("fit.bin","ab");
        check(!fFit,"Error open fit.bin");
    }
    if(wr2D){
        printf("Will write 2D patterns\n");
    }
    // CUDA Events for timing
    cudaEvent_t ev_proc_start, ev_proc_stop, ev_integ_start, ev_integ_stop;
    cudaEvent_t ev_prof_start, ev_prof_stop, ev_d2h_start, ev_d2h_stop;
    gpuErrchk(cudaEventCreate(&ev_proc_start)); gpuErrchk(cudaEventCreate(&ev_proc_stop)); // Use default flags (blocking sync)
    gpuErrchk(cudaEventCreate(&ev_integ_start)); gpuErrchk(cudaEventCreate(&ev_integ_stop));
    gpuErrchk(cudaEventCreate(&ev_prof_start)); gpuErrchk(cudaEventCreate(&ev_prof_stop));
    gpuErrchk(cudaEventCreate(&ev_d2h_start)); gpuErrchk(cudaEventCreate(&ev_d2h_stop));
    float t_proc_gpu=0, t_integ_gpu=0, t_prof_gpu=0, t_d2h_gpu=0;

    double t_qp_cpu=0, t_write1d_cpu=0, t_fit_cpu=0, t_writefit_cpu=0, t_write2d_cpu=0, t_loop_cpu=0;
    double t_start_loop, t_end_loop;

    int firstFrame=1;
    double *hIntArrFrame=(double*)calloc(bigArrSize,sizeof(double)); check(!hIntArrFrame,"Alloc fail hIntArrFrame");
    double *hPerFrame=(double*)malloc(bigArrSize*4*sizeof(double)); check(!hPerFrame,"Alloc fail hPerFrame");
    double *h_int1D = NULL; gpuErrchk(cudaMallocHost((void**)&h_int1D, nRBins * sizeof(double))); check(!h_int1D,"Alloc fail pinned h_int1D");
    double *hR=(double*)calloc(nRBins,sizeof(double)); check(!hR,"Alloc fail hR");
    double *hEta=(double*)calloc(nEtaBins,sizeof(double)); check(!hEta,"Alloc fail hEta");
    double *hLineout=(double*)malloc(nRBins*2*sizeof(double)); check(!hLineout,"Alloc fail hLineout");

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
        if(queue_pop(&process_queue,&chunk)<0){
            break; // Shutdown requested while queue was empty
        }
        t_qp_cpu = get_wall_time_ms() - t_qp_start;

        // --- GPU Processing Stage ---
        gpuErrchk(cudaEventRecord(ev_proc_start, 0));
        ProcessImageGPU(chunk.data,dProcessedImage,dAvgDark,Nopt,Topt,NrPixelsY,NrPixelsZ,darkSubEnabled);
        gpuErrchk(cudaEventRecord(ev_proc_stop, 0));

        // --- GPU Integration Stage ---
        gpuErrchk(cudaMemsetAsync(dIntArrFrame,0,bigArrSize*sizeof(double),0));
        int currFidx=chunk.dataset_num;
        int integTPB=THREADS_PER_BLOCK_INTEGRATE;
        int nrVox=(bigArrSize+integTPB-1)/integTPB;
        gpuErrchk(cudaEventRecord(ev_integ_start, 0));
        if(!dMapMask){
            integrate_noMapMask<<<nrVox,integTPB, 0, 0>>>(px,Lsd,bigArrSize,Normalize,sumI,currFidx,dPxList,dNPxList,dRLo,dRHi,dEtaLo,dEtaHi,NrPixelsY, NrPixelsZ, dProcessedImage,dIntArrFrame,dPerFrame,dSumMatrix);
        } else {
            integrate_MapMask<<<nrVox,integTPB, 0, 0>>>(px,Lsd,bigArrSize,Normalize,sumI,currFidx,mapMaskWC,dMapMask,nRBins,nEtaBins,NrPixelsY, NrPixelsZ, dPxList,dNPxList,dRLo,dRHi,dEtaLo,dEtaHi,dProcessedImage,dIntArrFrame,dPerFrame,dSumMatrix);
        }
        gpuErrchk(cudaEventRecord(ev_integ_stop, 0));

        // --- GPU 1D Profile Stage ---
        size_t profileSharedMem = (THREADS_PER_BLOCK_PROFILE / 32) * sizeof(double) * 2; // Shared mem per warp * 2 buffers (intensity*area, area)
        gpuErrchk(cudaEventRecord(ev_prof_start, 0));
        calculate_1D_profile_kernel<<<nRBins, THREADS_PER_BLOCK_PROFILE, profileSharedMem, 0>>>(dIntArrFrame, dPerFrame, d_int1D, nRBins, nEtaBins, bigArrSize);
        gpuErrchk(cudaEventRecord(ev_prof_stop, 0));

        // --- D->H Copy Stage ---
        gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
        gpuErrchk(cudaMemcpyAsync(h_int1D, d_int1D, nRBins*sizeof(double), cudaMemcpyDeviceToHost, 0));
        if(firstFrame==1){
            gpuErrchk(cudaMemcpyAsync(hPerFrame,dPerFrame,bigArrSize*4*sizeof(double),cudaMemcpyDeviceToHost, 0));
        }
        if(wr2D){
            gpuErrchk(cudaMemcpyAsync(hIntArrFrame, dIntArrFrame, bigArrSize*sizeof(double), cudaMemcpyDeviceToHost, 0));
        }
        gpuErrchk(cudaEventRecord(ev_d2h_stop, 0));

        // --- Synchronize and Get GPU Timings ---
        gpuErrchk(cudaEventSynchronize(ev_proc_stop));
        gpuErrchk(cudaEventElapsedTime(&t_proc_gpu, ev_proc_start, ev_proc_stop));
        gpuErrchk(cudaEventSynchronize(ev_integ_stop));
        gpuErrchk(cudaEventElapsedTime(&t_integ_gpu, ev_integ_start, ev_integ_stop));
        gpuErrchk(cudaEventSynchronize(ev_prof_stop));
        gpuErrchk(cudaEventElapsedTime(&t_prof_gpu, ev_prof_start, ev_prof_stop));
        gpuErrchk(cudaEventSynchronize(ev_d2h_stop)); // Sync D->H copies before CPU processing
        gpuErrchk(cudaEventElapsedTime(&t_d2h_gpu, ev_d2h_start, ev_d2h_stop));

        // --- CPU Processing Stage ---
        if(firstFrame==1){
            for(int r=0;r<nRBins;++r){
                hR[r]=hPerFrame[r*nEtaBins]; // R stored at start of each row's data
            }
            for(int e=0;e<nEtaBins;++e){
                hEta[e]=hPerFrame[e+2*bigArrSize]; // Eta stored in 3rd block
            }
            for(int r=0;r<nRBins;++r){
                hLineout[r*2]=hR[r]; // Set R values for lineout buffer
            }
            printf("Init R/Eta arrays GPU\n");
            firstFrame=0;
        }

        double t_write2d_start = get_wall_time_ms();
        if(wr2D){
            char fn[MAX_FILENAME_LENGTH];
            snprintf(fn,MAX_FILENAME_LENGTH,"Int2D_f%05d.bin",currFidx);
            FILE*f2d=fopen(fn,"wb");
            if(f2d){
                fwrite(hIntArrFrame,sizeof(double),bigArrSize,f2d);
                fclose(f2d);
            } else {
                fprintf(stderr,"Warn:Fail open %s\n",fn);
            }
        }
        t_write2d_cpu = get_wall_time_ms() - t_write2d_start;

        double maxInt=-1.0;
        int maxIntLoc=-1;
        if(!multiP){
            for(int r=0; r<nRBins; ++r){
                if(h_int1D[r]>maxInt){
                    maxInt=h_int1D[r];
                    maxIntLoc=r;
                }
            }
        }
        for(int r=0; r<nRBins; ++r){
            hLineout[r*2+1]=h_int1D[r]; // Prepare lineout buffer
        }

        double t_write1d_start = get_wall_time_ms();
        check(fwrite(hLineout,sizeof(double),nRBins*2,fLineout)!=(size_t)nRBins*2,"Err write lineout");
        fflush(fLineout);
        // Optional Send: send_lineouts(currFidx, nRBins, hLineout);
        t_write1d_cpu = get_wall_time_ms() - t_write1d_start;

        double t_fit_start = get_wall_time_ms();
        int currentPeakCount=0;
        double *sendFitParams=NULL;
        if(pkFit){
            Peak *pks=NULL;
            if(multiP){
                if(nSpecP>0){
                    pks=(Peak*)malloc(nSpecP*sizeof(Peak));
                    check(!pks,"Malloc fail specP");
                    int vPC=0;
                    for(int p=0;p<nSpecP;++p){
                        int bL=-1;
                        double mD=1e10;
                        for(int r=0;r<nRBins;++r){
                            double df=fabs(hR[r]-pkLoc[p]);
                            if(df<mD){
                                mD=df;
                                bL=r;
                            }
                        }
                        if(bL!=-1&&mD<RBinSize*2.0){
                            pks[vPC++]=(Peak){bL,hR[bL],h_int1D[bL]};
                        } else {
                            printf("Warn: Peak %.2f ign F%d\n",pkLoc[p],currFidx);
                        }
                    }
                    currentPeakCount=vPC;
                    if(vPC==0){
                        free(pks);
                        pks=NULL;
                    }
                } else {
                    double *d2FP=h_int1D;
                    double *smD=NULL;
                    if(doSm){
                        smD=(double*)malloc(nRBins*sizeof(double));
                        check(!smD,"Malloc smD");
                        smoothData(h_int1D,smD,nRBins,7); // Use window size 7 (example)
                        d2FP=smD;
                    }
                    currentPeakCount=findPeaks(d2FP,hR,nRBins,&pks,0.0,5); // minHeight=0, minDist=5
                    if(smD) {
                        free(smD);
                    }
                }
            } else {
                if(maxIntLoc!=-1){
                    currentPeakCount=1;
                    pks=(Peak*)malloc(sizeof(Peak));
                    check(!pks,"Malloc sglP");
                    pks[0]=(Peak){maxIntLoc,hR[maxIntLoc],maxInt};
                } else {
                    currentPeakCount=0;
                    pks=NULL;
                }
            }
            if (currentPeakCount > 0 && pks != NULL) {
                int nFP=currentPeakCount*4+1; // 4 params per peak + 1 global bg
                double *fitP=(double*)malloc(nFP*sizeof(double)); check(!fitP,"Malloc fitP");
                double *lb=(double*)malloc(nFP*sizeof(double)); check(!lb,"Malloc lb");
                double *ub=(double*)malloc(nFP*sizeof(double)); check(!ub,"Malloc ub");
                double maxOI=0.0; for(int r=0;r<nRBins;++r) if(h_int1D[r]>maxOI) maxOI=h_int1D[r]; if(maxOI<=0) maxOI=1.0;
                for(int p=0;p<currentPeakCount;++p){
                    int b=p*4;
                    fitP[b+0]=pks[p].intensity; fitP[b+1]=0.5; fitP[b+2]=pks[p].radius; fitP[b+3]=RBinSize*2.0; // Initial guess
                    lb[b+0]=0.0; lb[b+1]=0.0; lb[b+2]=pks[p].radius-RBinSize*5.0; lb[b+3]=RBinSize*0.5; // Lower bounds
                    ub[b+0]=maxOI*2.0; ub[b+1]=1.0; ub[b+2]=pks[p].radius+RBinSize*5.0; ub[b+3]=(RMax-RMin)/4.0; // Upper bounds
                }
                fitP[nFP-1]=0.0; lb[nFP-1]=-maxOI; ub[nFP-1]=maxOI; // Global BG init/bounds
                dataFit fD; fD.nrBins=nRBins; fD.R=hR; fD.Int=h_int1D;
                nlopt_opt opt=nlopt_create(NLOPT_LN_NELDERMEAD,nFP);
                nlopt_set_lower_bounds(opt,lb);
                nlopt_set_upper_bounds(opt,ub);
                nlopt_set_min_objective(opt,problem_function_global_bg,&fD); // USE GLOBAL BG FUNCTION
                nlopt_set_xtol_rel(opt,1e-4);
                nlopt_set_maxeval(opt,500*nFP); // Limit evaluations
                double minO;
                int nlopt_rc=nlopt_optimize(opt,fitP,&minO);
                if(nlopt_rc<0){
                    printf("F#%d: NLopt fail %d\n",currFidx,nlopt_rc);
                    currentPeakCount=0;
                    free(fitP);
                } else {
                    sendFitParams=(double*)malloc(currentPeakCount*5*sizeof(double)); // Format for 5 params/peak
                    check(!sendFitParams,"Malloc sendFP");
                    double gBG=fitP[nFP-1]; // Get the fitted global bg
                    for(int p=0;p<currentPeakCount;++p){
                         sendFitParams[p*5+0]=fitP[p*4+0]; // amp
                         sendFitParams[p*5+1]=gBG;         // bg (global)
                         sendFitParams[p*5+2]=fitP[p*4+1]; // mix
                         sendFitParams[p*5+3]=fitP[p*4+2]; // cen
                         sendFitParams[p*5+4]=fitP[p*4+3]; // sig
                    }
                    free(fitP); // Free the raw fit parameters buffer
                }
                nlopt_destroy(opt);
                free(lb);
                free(ub);
            }
            if(pks) {
                free(pks); // Free peak array allocated by findPeaks or manually
            }
        }
        t_fit_cpu = get_wall_time_ms() - t_fit_start;

        double t_writefit_start = get_wall_time_ms();
        if(pkFit && currentPeakCount > 0 && sendFitParams != NULL){
            check(fwrite(sendFitParams,sizeof(double),currentPeakCount*5,fFit)!=(size_t)currentPeakCount*5,"Err write fit");
            fflush(fFit);
            // Optional Send: send_fit_result(currFidx,currentPeakCount,5,sendFitParams);
            free(sendFitParams); // Free the formatted buffer *after* using it
        }
        t_writefit_cpu = get_wall_time_ms() - t_writefit_start;

        double t_end_loop = get_wall_time_ms();
        t_loop_cpu = t_end_loop - t_start_loop; // Total wall time

        printf("F#%d: Ttl:%.2f| QPop:%.2f GPU(Proc:%.2f Int:%.2f Prof:%.2f D2H:%.2f) CPU(Wr2D:%.2f Wr1D:%.2f Fit:%.2f WrFit:%.2f) (ms)\n",
               currFidx, t_loop_cpu, t_qp_cpu, t_proc_gpu, t_integ_gpu, t_prof_gpu, t_d2h_gpu, t_write2d_cpu, t_write1d_cpu, t_fit_cpu, t_writefit_cpu);
        fflush(stdout);

        gpuWarnchk(cudaFreeHost(chunk.data)); // Free pinned host buffer for the received chunk
        frameCounter++;
    } // ======================== End Main Processing Loop ========================

    printf("Processing loop finished (keep_running=%d). Cleaning up...\n", keep_running);

    // --- Cleanup ---
    if(fLineout) fclose(fLineout);
    if(fFit) fclose(fFit);
    free(hAvgDark);
    free(hDarkInT);
    free(hDarkIn);
    // hImageInT is freed in queue_destroy
    free(hIntArrFrame);
    free(hPerFrame);
    if(h_int1D) gpuWarnchk(cudaFreeHost(h_int1D)); // Free pinned
    free(hR);
    free(hEta);
    free(hLineout);
    free(hEtaLo);
    free(hEtaHi);
    free(hRLo);
    free(hRHi);
    if(hMapMask) free(hMapMask);

    UnmapBins(); // Unmap mmap'd files

    // Free GPU memory (use gpuWarnchk for cleanup to avoid exiting on minor errors)
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

    // Destroy CUDA events
    gpuWarnchk(cudaEventDestroy(ev_proc_start)); gpuWarnchk(cudaEventDestroy(ev_proc_stop));
    gpuWarnchk(cudaEventDestroy(ev_integ_start)); gpuWarnchk(cudaEventDestroy(ev_integ_stop));
    gpuWarnchk(cudaEventDestroy(ev_prof_start)); gpuWarnchk(cudaEventDestroy(ev_prof_stop));
    gpuWarnchk(cudaEventDestroy(ev_d2h_start)); gpuWarnchk(cudaEventDestroy(ev_d2h_stop));

    // --- Shutdown Accept Thread ---
    printf("Closing server socket %d...\n", server_fd);
    if (server_fd >= 0) {
         // Closing the socket first *might* help cancellation succeed faster
         // if accept() reacts to it, though cancellation is the primary mechanism here.
         shutdown(server_fd, SHUT_RDWR); // More forceful than just close
         close(server_fd);
         server_fd = -1;
    }

    printf("Sending cancellation request to accept thread...\n");
    int cancel_ret = pthread_cancel(accept_thread);
    if (cancel_ret != 0) {
        fprintf(stderr, "Warning: Failed to send cancel request to accept thread: %s\n", strerror(cancel_ret));
    }

    // Join the thread to wait for it to actually terminate and clean up resources
    printf("Joining accept thread (waiting for cancellation)...\n");
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

    // Destroy queue (should happen after all threads using it are joined/finished)
    queue_destroy(&process_queue);

    printf("[%s] - Exiting cleanly.\n", argv[0]);
    return 0;
}
