//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// export LD_LIBRARY_PATH=/scratch/s1iduser/sharma_tests/HDF5/lib:/scratch/s1iduser/sharma_tests/LIBTIFF/lib::$LD_LIBRARY_PATH
/* Compiling info:
  source ~/.MIDAS/paths
  ~/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStream.cu -o bin/IntegratorFitPeaksGPUStream -Xcompiler -g -arch sm_90 \
  -gencode=arch=compute_90,code=sm_90 -I/home/beams/S1IDUSER/.MIDAS/NLOPT/include -L/home/beams/S1IDUSER/.MIDAS/NLOPT/lib \
  -O3 -lnlopt -I/home/beams/S1IDUSER/.MIDAS/BLOSC/include -L/home/beams/S1IDUSER/.MIDAS/BLOSC/lib64 -lblosc2 \
  -I/home/beams/S1IDUSER/.MIDAS/HDF5/include -L/home/beams/S1IDUSER/.MIDAS/HDF5/lib -lhdf5 -lhdf5_hl -lz -ldl -lm -lpthread \
  -I/home/beams/S1IDUSER/.MIDAS/LIBZIP/include -L/home/beams/S1IDUSER/.MIDAS/LIBZIP/lib64 -lzip
  */

/* Compiling for s16idbuser
  source ~/.MIDAS/paths
  /home/beams/S1IDUSER/opt/midascuda/cuda/bin/nvcc src/IntegratorFitPeaksGPUStream.cu -o bin/IntegratorFitPeaksGPUStream -Xcompiler -g -arch sm_90 \
  -gencode=arch=compute_90,code=sm_90 -I/home/beams/S16IDBUSER/.MIDAS/NLOPT/include -L/home/beams/S16IDBUSER/.MIDAS/NLOPT/lib \
  -O3 -lnlopt -I/home/beams/S16IDBUSER/.MIDAS/BLOSC/include -L/home/beams/S16IDBUSER/.MIDAS/BLOSC/lib64 -lblosc2 \
  -I/home/beams/S16IDBUSER/.MIDAS/HDF5/include -L/home/beams/S16IDBUSER/.MIDAS/HDF5/lib -lhdf5 -lhdf5_hl -lz -ldl -lm -lpthread \
  -I/home/beams/S16IDBUSER/.MIDAS/LIBZIP/include -L/home/beams/S16IDBUSER/.MIDAS/LIBZIP/lib64 -lzip

*/

// Always expectes the data to be in int32_t format.
// Expected to be run on a GPU with at least 16GB of memory.
// Expected to be run on a machine with at least 16GB of memory.
// Localhost:5000 port is used to get the images from a server.
// The server is expected to send the images in int32_t format.
// There is an imagenum in the header, which is used to keep track of the image number.
// The images are expected to be in a binary format.
//
// The code expects a Map.bin and nMap.bin file to be present in the same directory.
// The Map.bin file is expected to contain the pixel information in the following format:
// struct data{
//     int y;
//     int z;
//     double frac;
// };
// The nMap.bin file is expected to contain the number of pixels in each pixel.
// The code expects a ParamFN file to be present in the same directory.
// The ParamFN file is expected to contain the following information:
// EtaBinSize 0.1
// RBinSize 0.1
// RMax 100
// RMin 10
// EtaMax 180
// EtaMin -180
// Lsd 1000
// px 0.172
// NrPixelsY 2048
// NrPixelsZ 2048
// Normalize 1
// NrPixels 2048
// GapIntensity 0
// BadPxIntensity 0
// ImTransOpt 0
// SumImages 0

// The code expects a(n optional) dark file to be present in the same directory.
// The dark file is expected to be in the same format as the images.
// The dark file is expected to be in a binary format.
// The dark file is expected to contain the average of all the dark frames.
// The dark file is expected to be in int32_t format.

// Benchmarks: 
// 11.8s using H100, 38.2s using CPU. 
// Overhead (I/O, prepare): 10.5s, 1.19s using H100, 27.7s using CPU, 23x speedup. 
// 1469MPx/s using H100, 208MPx/s using CPU
// Integrator.cu
//
// Hemant Sharma
// Dt: 2023/12/15
//

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
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdint.h>
#include <libgen.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <blosc2.h>
//#include <zip.h> 
#include <nlopt.h>

#define SERVER_IP "127.0.0.1"
#define PORTSENDFIT 5001
#define PORTSENDLINEOUT 5002
#define PORT 5000           // Changed port to 5000
#define MAX_CONNECTIONS 10
size_t CHUNK_SIZE;
#define MAX_QUEUE_SIZE 100  // Maximum number of chunks in the queue, should not be too large, else segfaults.
#define HEADER_SIZE sizeof(uint16_t)  // Size of dataset number
size_t TOTAL_MSG_SIZE;
#define BYTES_PER_PIXEL 4

// Structure for our data chunks
typedef struct {
    uint16_t dataset_num;  // Dataset number
    int32_t *data; // int32 data
    size_t size;
} DataChunk;

// Thread-safe queue for data processing
typedef struct {
    DataChunk chunks[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} ProcessQueue;

// Global processing queue
ProcessQueue process_queue;

// Initialize the processing queue
void queue_init(ProcessQueue *queue) {
    queue->front = 0;
    queue->rear = -1;
    queue->count = 0;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
}

// Add a data chunk to the queue
int queue_push(ProcessQueue *queue, uint16_t dataset_num, int32_t *data, size_t num_values) {
    pthread_mutex_lock(&queue->mutex);
    
    // Wait if the queue is full
    while (queue->count >= MAX_QUEUE_SIZE) {
        printf("Queue full, waiting...\n");
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    // Add the chunk to the queue
    queue->rear = (queue->rear + 1) % MAX_QUEUE_SIZE;
    queue->chunks[queue->rear].dataset_num = dataset_num;
    queue->chunks[queue->rear].data = data;
    queue->chunks[queue->rear].size = num_values;
    queue->count++;
    
    // Signal that the queue is not empty
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

// Get a data chunk from the queue
int queue_pop(ProcessQueue *queue, DataChunk *chunk) {
    pthread_mutex_lock(&queue->mutex);
    
    // Wait if the queue is empty
    while (queue->count <= 0) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    // Get the chunk from the queue
    *chunk = queue->chunks[queue->front];
    queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
    queue->count--;
    
    // Signal that the queue is not full
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return 0;
}

// Thread function to handle client connection
void* handle_client(void *arg) {
    int client_socket = *((int*)arg);
    free(arg);  // Free the memory allocated for the argument
    
    // Buffer for receiving raw bytes (header + data)
    uint8_t *buffer = (uint8_t*)malloc(TOTAL_MSG_SIZE);
    int bytes_read;
    
    // Continuously read chunks
    while (1) {
        // Reset total bytes read for this message
        int total_bytes_read = 0;
        
        // Read until we get a complete message
        while (total_bytes_read < TOTAL_MSG_SIZE) {
            bytes_read = recv(client_socket, buffer + total_bytes_read, TOTAL_MSG_SIZE - total_bytes_read, 0);
            
            if (bytes_read <= 0) {
                // Connection closed or error
                goto connection_closed;
            }
            
            total_bytes_read += bytes_read;
        }
        
        // Extract dataset number from header
        uint16_t dataset_num;
        memcpy(&dataset_num, buffer, HEADER_SIZE);
        
        // Allocate memory for the data
        int32_t *data = (int32_t*)malloc(CHUNK_SIZE/BYTES_PER_PIXEL * sizeof(int32_t));
        if (!data) {
            perror("Memory allocation failed");
            break;
        }
        for (int i = 0; i < CHUNK_SIZE/BYTES_PER_PIXEL; i++) {
			int32_t value;
			memcpy(&value, buffer + HEADER_SIZE + (i * sizeof(int32_t)), sizeof(int32_t));
			data[i] = value;
		}
        
        // Add the data to the processing queue
        queue_push(&process_queue, dataset_num, data, CHUNK_SIZE/BYTES_PER_PIXEL);
        printf("Received dataset #%u with %d int32_t values\n", dataset_num, CHUNK_SIZE/BYTES_PER_PIXEL);
    }
    
connection_closed:
    if (bytes_read == 0) {
        printf("Client disconnected gracefully\n");
    } else if (bytes_read < 0) {
        perror("Receive error");
    }
    
    close(client_socket);
    return NULL;
}

// Thread function for accepting connections
void* accept_connections(void *server_fd_ptr) {
    int server_fd = *((int*)server_fd_ptr);
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    while (1) {
        // Accept connection
        int *client_socket = (int*) malloc(sizeof(int));
        if ((*client_socket = accept(server_fd, (struct sockaddr *)&client_addr, &client_len)) < 0) {
            perror("Accept failed");
            free(client_socket);
            continue;
        }
        
        printf("Connection accepted from %s:%d\n", 
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        
        // Create thread to handle client
        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, (void*)client_socket) != 0) {
            perror("Thread creation failed");
            close(*client_socket);
            free(client_socket);
        } else {
            // Detach thread to free resources automatically when it terminates
            pthread_detach(thread_id);
        }
    }
    
    return NULL;
}

int send_fit_result(uint16_t frame_num, int n, double *values) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    
    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        return -1;
    }
    
    // Set up server address
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORTSENDFIT);
    
    // Convert IPv4 address from text to binary form
    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock);
        return -1;
    }
    
    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        close(sock);
        return -1;
    }
    
    // Prepare headers: frame number and count, both in network byte order
    uint16_t frame_header = htons(frame_num);
    uint16_t count_header = htons((uint16_t)n);
    
    // Calculate total number of doubles (n sets of 5 doubles each)
    int total_doubles = n * 5;
    
    // Send frame number header
    if (send(sock, &frame_header, sizeof(frame_header), 0) != sizeof(frame_header)) {
        perror("Failed to send frame number header");
        close(sock);
        return -1;
    }
    
    // Send count header (number of sets)
    if (send(sock, &count_header, sizeof(count_header), 0) != sizeof(count_header)) {
        perror("Failed to send count header");
        close(sock);
        return -1;
    }
    
    // Send doubles one by one, converting to network byte order
    for (int i = 0; i < total_doubles; i++) {
        // Network byte order for doubles needs manual handling
        // For portability, we'll convert the double bits
        uint64_t network_value;
        memcpy(&network_value, &values[i], sizeof(double));
        network_value = htobe64(network_value);  // Host to big-endian
        
        if (send(sock, &network_value, sizeof(double), 0) != sizeof(double)) {
            perror("Failed to send value");
            close(sock);
            return -1;
        }
    }
    
    printf("Sent results for frameNr %d with %d peak (%d doubles) successfully\n", frame_num, n, total_doubles);
    close(sock);
    return 0;
}

int send_lineouts(uint16_t frame_num, int n, double *values) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    
    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        return -1;
    }
    
    // Set up server address
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    
    // Convert IPv4 address from text to binary form
    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock);
        return -1;
    }
    
    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        close(sock);
        return -1;
    }
    
    // Prepare headers: frame number and count, both in network byte order
    uint16_t frame_header = htons(frame_num);
    uint16_t count_header = htons((uint16_t)n);
    
    // Send frame number header
    if (send(sock, &frame_header, sizeof(frame_header), 0) != sizeof(frame_header)) {
        perror("Failed to send frame number header");
        close(sock);
        return -1;
    }
    
    // Send count header (number of sets)
    if (send(sock, &count_header, sizeof(count_header), 0) != sizeof(count_header)) {
        perror("Failed to send count header");
        close(sock);
        return -1;
    }
    
    // Send doubles one by one, converting to network byte order
    for (int i = 0; i < n; i++) {
        // Network byte order for doubles needs manual handling
        // For portability, we'll convert the double bits
        uint64_t network_value;
        memcpy(&network_value, &values[i], sizeof(double));
        network_value = htobe64(network_value);  // Host to big-endian
        
        if (send(sock, &network_value, sizeof(double), 0) != sizeof(double)) {
            perror("Failed to send value");
            close(sock);
            return -1;
        }
    }
    
    printf("Sent frame lineout %d with %d doubles successfully\n", frame_num, n);
    close(sock);
    return 0;
}

#define SetBit(A,k)   (A[(k/32)] |=  (1 << (k%32)))
#define TestBit(A,k)  (A[(k/32)] &   (1 << (k%32)))
#define rad2deg 57.2957795130823

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void
check (int test, const char * message, ...)
{
    if (test) {
        va_list args;
        va_start (args, message);
        vfprintf (stderr, message, args);
        va_end (args);
        fprintf (stderr, "\n");
        exit (EXIT_FAILURE);
    }
}

struct data {
	int y;
	int z;
	double frac;
};

struct data *pxList;
int *nPxList;
size_t szPxList;
size_t szNPxList;

int ReadBins(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "Map.bin";
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    int sizelen = 2*(int)sizeof(int) + (int)sizeof(double);
    printf("Map size in bytes: %lld, each element size: %d, total elements: %lld. \n",(long long int)size,sizelen,(long long int)(size/sizelen));
    pxList = (data *) mmap (0, size, PROT_READ, MAP_SHARED, fd, 0);
	szPxList = size;
    check (pxList == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));

    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "nMap.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    nPxList = (int *) mmap (0, size2, PROT_READ, MAP_SHARED, fd2, 0);
	szNPxList = size2;
    printf("nMap size in bytes: %lld, each element size: %d, total elements: %lld. \n",
		(long long int)size2,2*(int)sizeof(int),2*(long long int)(size2/sizeof(int)));
    fflush(stdout);
    check (nPxList == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));
	return 1;
}

__global__
void integrate_noMapMask (double px, double Lsd, int bigArrSize, int Normalize, int sumImages, int i, int NrPixelsY, 
		int mapMaskSize, int *MapMask, int nRBins, int nEtaBins, struct data * dPxList, 
		int *dNPxList, double *RBinsLow, double *RBinsHigh, double *EtaBinsLow, double *EtaBinsHigh, 
		double *dImage, double *IntArrPerFrame, double *PerFrameArr, double *SumMatrix)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < bigArrSize){
		int l;
		double Intensity=0, totArea=0;
		struct data ThisVal;
		long long int nPixels, dataPos, testPos;
		nPixels = dNPxList[2*idx + 0];
		dataPos = dNPxList[2*idx + 1];
		for (l=0;l<nPixels;l++){
			ThisVal = dPxList[dataPos + l];
			testPos = ThisVal.z;
			testPos *= NrPixelsY;
			testPos += ThisVal.y;
			Intensity += dImage[testPos]*ThisVal.frac;
			totArea += ThisVal.frac;
		}
		if (Intensity != 0){
			if (Normalize == 1){
				Intensity /= totArea;
			}
			IntArrPerFrame[idx] = Intensity;
			if (sumImages==1){
				SumMatrix[idx] += Intensity;
			}
		}
		if (i==0){
			int j = idx/nEtaBins;
			int k = idx%nEtaBins;
			double RMean = (RBinsLow[j]+RBinsHigh[j])/2;
			double EtaMean = (EtaBinsLow[k]+EtaBinsHigh[k])/2;
			PerFrameArr[0*bigArrSize+(j*nEtaBins+k)] = RMean;
			PerFrameArr[1*bigArrSize+(j*nEtaBins+k)] = 57.2957795130823*atan(RMean*px/Lsd);
			PerFrameArr[2*bigArrSize+(j*nEtaBins+k)] = EtaMean;
			PerFrameArr[3*bigArrSize+(j*nEtaBins+k)] = totArea;
		}
	}
}

__global__
void integrate_MapMask (double px, double Lsd, int bigArrSize, int Normalize, int sumImages, int i, int NrPixelsY, 
		int mapMaskSize, int *MapMask, int nRBins, int nEtaBins, struct data * dPxList, 
		int *dNPxList, double *RBinsLow, double *RBinsHigh, double *EtaBinsLow, double *EtaBinsHigh, 
		double *dImage, double *IntArrPerFrame, double *PerFrameArr, double *SumMatrix)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < bigArrSize){
		int l;
		double Intensity=0, totArea=0;
		struct data ThisVal;
		long long int nPixels, dataPos, testPos;
		nPixels = dNPxList[2*idx + 0];
		dataPos = dNPxList[2*idx + 1];
		for (l=0;l<nPixels;l++){
			ThisVal = dPxList[dataPos + l];
			testPos = ThisVal.z;
			testPos *= NrPixelsY;
			testPos += ThisVal.y;
			if (TestBit(MapMask,testPos)){
				continue;
			}
			Intensity += dImage[testPos]*ThisVal.frac;
			totArea += ThisVal.frac;
		}
		if (Intensity != 0){
			if (Normalize == 1){
				Intensity /= totArea;
			}
			IntArrPerFrame[idx] = Intensity;
			if (sumImages==1){
				SumMatrix[idx] += Intensity;
			}
		}
		if (i==0){
			int j = idx/nEtaBins;
			int k = idx%nEtaBins;
			double RMean = (RBinsLow[j]+RBinsHigh[j])/2;
			double EtaMean = (EtaBinsLow[k]+EtaBinsHigh[k])/2;
			PerFrameArr[0*bigArrSize+(j*nEtaBins+k)] = RMean;
			PerFrameArr[1*bigArrSize+(j*nEtaBins+k)] = 57.2957795130823*atan(RMean*px/Lsd);
			PerFrameArr[2*bigArrSize+(j*nEtaBins+k)] = EtaMean;
			PerFrameArr[3*bigArrSize+(j*nEtaBins+k)] = totArea;
		}
	}
}

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

static inline
void
REtaMapper(
	double Rmin,
	double EtaMin,
	int nEtaBins,
	int nRBins,
	double EtaBinSize,
	double RBinSize,
	double *EtaBinsLow,
	double *EtaBinsHigh,
	double *RBinsLow,
	double *RBinsHigh)
{
	int i;
	for (i=0;i<nEtaBins;i++){
		EtaBinsLow[i] = EtaBinSize*i      + EtaMin;
		EtaBinsHigh[i] = EtaBinSize*(i+1) + EtaMin;
	}
	for (i=0;i<nRBins;i++){
		RBinsLow[i] =  RBinSize * i     + Rmin;
		RBinsHigh[i] = RBinSize * (i+1) + Rmin;
	}
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], int32_t *ImageIn, int32_t *ImageOut, int NrPixelsY, int NrPixelsZ)
{
	int i,k,l;
	if (NrTransOpt == 0 || (NrTransOpt==1 && TransOpt[0]==0)){
		memcpy(ImageOut,ImageIn,NrPixelsY*NrPixelsZ*sizeof(*ImageIn)); // Nothing to do
		return;
	}
    for (i=0;i<NrTransOpt;i++){
		if (TransOpt[i] == 1){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[l*NrPixelsY+(NrPixelsY-k-1)]; // Invert Y
				}
			}
		}else if (TransOpt[i] == 2){
			for (k=0;k<NrPixelsY;k++){
				for (l=0;l<NrPixelsZ;l++){
					ImageOut[l*NrPixelsY+k] = ImageIn[(NrPixelsZ-l-1)*NrPixelsY+k]; // Invert Z
				}
			}
		}
	}
}

struct dataFit{
	int nrBins;
	double *R;
	double *Int;
};

static double problem_function(unsigned n,
	const double *x,
	double *grad,
	void *my_func_data)
{
	int nPeaks = n/5;
	struct dataFit *d = (struct dataFit *) my_func_data;
	double error = 0;
	int nrPoints = d->nrBins;
	double *Rs;
	Rs = &(d->R[0]);
	double *Ints;
	Ints = &(d->Int[0]);
	double *thisInt;
	thisInt = (double *)calloc(nrPoints,sizeof(double));
	for (int PeakNr=0;PeakNr<nPeaks;PeakNr++){
		double amp = x[PeakNr*5+0];
		double bg = x[PeakNr*5+1];
		double mix = x[PeakNr*5+2];
		double cen = x[PeakNr*5+3];
		double sig = x[PeakNr*5+4];
		int i;
		double diff, gauss, lorentz;
		for (i=0;i<nrPoints;i++){
			diff = Rs[i] - cen;
			gauss = exp(-diff*diff/(2*sig*sig))/(sig*sqrt(2*M_PI));
			lorentz = 1/((M_PI*sig)*(1+diff*diff/(sig*sig)));
			thisInt[i] += bg + amp*(mix*gauss + (1-mix)*lorentz);
		}
	}
	for (int i=0;i<nrPoints;i++){
		error += (thisInt[i] - Ints[i])*(thisInt[i] - Ints[i]);
	}
	free(thisInt);
	return error;
}

// Structure to store peak information
typedef struct {
    int index;       // Index of the peak in the data array
    double radius;   // Radius value corresponding to the peak
    double intensity; // Intensity value at the peak
} Peak;

// Function to calculate factorial
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Function to calculate binomial coefficient
int binomial(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// Function to perform Savitzky-Golay smoothing
void smoothData(const double *input, double *output, int size, int windowSize) {
    if (windowSize % 2 == 0) windowSize++; // Ensure window size is odd
    
    int halfWindow = windowSize / 2;
    int degree = 2; // Polynomial degree (quadratic by default)
    
    // Ensure degree is less than window size
    if (degree >= windowSize) {
        degree = windowSize - 1;
        if (degree < 1) degree = 1;
    }
    
    // Pre-compute Savitzky-Golay coefficients
    // For a quadratic polynomial (degree=2), we can use simplified coefficients
    double *coeffs = (double *)malloc(windowSize * sizeof(double));
    
    if (degree == 2) {
        // For quadratic fit, we can use predefined coefficients for common window sizes
        switch (windowSize) {
            case 5:
                // 5-point quadratic coefficients: [-3, 12, 17, 12, -3] / 35
                coeffs[0] = -3.0/35.0; coeffs[1] = 12.0/35.0; coeffs[2] = 17.0/35.0;
                coeffs[3] = 12.0/35.0; coeffs[4] = -3.0/35.0;
                break;
            case 7:
                // 7-point quadratic coefficients: [-2, 3, 6, 7, 6, 3, -2] / 21
                coeffs[0] = -2.0/21.0; coeffs[1] = 3.0/21.0; coeffs[2] = 6.0/21.0;
                coeffs[3] = 7.0/21.0; coeffs[4] = 6.0/21.0; coeffs[5] = 3.0/21.0;
                coeffs[6] = -2.0/21.0;
                break;
            case 9:
                // 9-point quadratic coefficients: [-21, 14, 39, 54, 59, 54, 39, 14, -21] / 231
                coeffs[0] = -21.0/231.0; coeffs[1] = 14.0/231.0; coeffs[2] = 39.0/231.0;
                coeffs[3] = 54.0/231.0; coeffs[4] = 59.0/231.0; coeffs[5] = 54.0/231.0;
                coeffs[6] = 39.0/231.0; coeffs[7] = 14.0/231.0; coeffs[8] = -21.0/231.0;
                break;
            default:
                // For other window sizes, calculate coefficients using a general formula
                // This is a simplified version - full implementation would use matrix operations
                int norm = 0;
                for (int i = -halfWindow; i <= halfWindow; i++) {
                    coeffs[i + halfWindow] = (3 * (halfWindow*halfWindow) - i*i);
                    norm += coeffs[i + halfWindow];
                }
                for (int i = 0; i < windowSize; i++) {
                    coeffs[i] /= norm;
                }
        }
    } else {
        // For non-quadratic polynomials, we'd need a more general approach
        // Using a simplified approach for linear fit (degree=1)
        int norm = windowSize;
        for (int i = 0; i < windowSize; i++) {
            coeffs[i] = 1.0 / norm;  // Simplifies to moving average for degree=1
        }
    }
    
    // Apply Savitzky-Golay filter to the main part of the data
    for (int i = halfWindow; i < size - halfWindow; i++) {
        double sum = 0;
        for (int j = 0; j < windowSize; j++) {
            sum += coeffs[j] * input[i - halfWindow + j];
        }
        output[i] = sum;
    }
    
    // Handle boundary cases
    // For simplicity, use the original input values near the edges
    for (int i = 0; i < halfWindow; i++) {
        output[i] = input[i];
    }
    for (int i = size - halfWindow; i < size; i++) {
        output[i] = input[i];
    }
    
    free(coeffs);
}

// Function to find local maxima peaks
int findPeaks(const double *data, const double *radii, int size, Peak **peaks, double minHeight, int minDistance) {
    int maxPeaks = size / 2; // Maximum possible number of peaks
    *peaks = (Peak *)malloc(maxPeaks * sizeof(Peak));
    
    int peakCount = 0;
    
    // Find all potential peaks
    for (int i = 1; i < size - 1; i++) {
        if (data[i] > data[i-1] && data[i] > data[i+1] && data[i] >= minHeight) {
            // Found a potential peak
            (*peaks)[peakCount].index = i;
            (*peaks)[peakCount].intensity = data[i];
            // Assuming radius data is available or can be calculated
            (*peaks)[peakCount].radius = radii[i]; // Replace with actual radius calculation if available
            peakCount++;
            
            if (peakCount >= maxPeaks) {
                break;
            }
        }
    }
    
    // Apply minimum distance filter
    if (minDistance > 1 && peakCount > 1) {
        int j = 0;
        Peak *filteredPeaks = (Peak *)malloc(peakCount * sizeof(Peak));
        
        // Add the first peak
        filteredPeaks[j++] = (*peaks)[0];
        
        for (int i = 1; i < peakCount; i++) {
            // Check if this peak is far enough from the last added peak
            if ((*peaks)[i].index - filteredPeaks[j-1].index >= minDistance) {
                filteredPeaks[j++] = (*peaks)[i];
            } else if ((*peaks)[i].intensity > filteredPeaks[j-1].intensity) {
                // If not far enough but higher, replace the previous peak
                filteredPeaks[j-1] = (*peaks)[i];
            }
        }
        
        // Replace the original peak array
        free(*peaks);
        *peaks = filteredPeaks;
        peakCount = j;
    }
    
    return peakCount;
}

int main(int argc, char *argv[]){
    if (argc < 2){
		printf("Usage: ./Integrator ParamFN (optional)DarkName\n"
		"Optional:\n\tDark file: dark correction with average of all dark frames"
		".\n\tDark must be in a binary format, with int32 dataType."
        "\n\nSteams data from a socket and processes it.\n");
		return(1);
	}
	printf("[%s] - Starting...\n", argv[0]);
	clock_t start0, end0;
	start0 = clock();
    double diftotal;
	int device_id = 0;
	gpuErrchk(cudaSetDevice(device_id));
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Initialized the GPU:\t%f s.\n",diftotal);
	int rc = ReadBins();
	double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
	int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
	int nEtaBins, nRBins;
    char *ParamFN;
    FILE *paramFile;
    ParamFN = argv[1];
	char aline[4096], dummy[4096];
	const char *str;
	paramFile = fopen(ParamFN,"r");
    int NrTransOpt=0;
    long long int GapIntensity=0, BadPxIntensity=0;
    int TransOpt[10];
    int makeMap = 0;
    size_t mapMaskSize = 0;
	int *mapMask;
	int sumImages=0;
	int doSmoothing = 0, multiplePeaks = 0, peakFit = 0;
	int maxNPeaks = 100, nPeaks = 0;
	double peakLocations[maxNPeaks];
	while (fgets(aline,4096,paramFile) != NULL){
		str = "EtaBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaBinSize);
		}
		str = "RBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RBinSize);
		}
		str = "RMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMax);
		}
		str = "RMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RMin);
		}
		str = "EtaMax ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMax);
		}
		str = "EtaMin ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaMin);
		}
		str = "PeakLocation ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &peakLocations[nPeaks]);
			multiplePeaks = 1;
			peakFit = 1;
			doSmoothing = 0;
			nPeaks++;
		}
		str = "Lsd ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Lsd);
		}
		str = "px ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &px);
		}
		str = "NrPixelsY ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
		}
		str = "NrPixelsZ ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "DoSmoothing ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &doSmoothing);
		}
		str = "MultiplePeaks ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &multiplePeaks);
		}
		str = "DoPeakFit ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &peakFit);
		}
		str = "Normalize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &Normalize);
		}
		str = "NrPixels ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &NrPixelsY);
			sscanf(aline,"%s %d", dummy, &NrPixelsZ);
		}
		str = "GapIntensity ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %lld", dummy, &GapIntensity);
            makeMap = 1;
            continue;
        }
		str = "BadPxIntensity ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %lld", dummy, &BadPxIntensity);
            makeMap = 1;
            continue;
        }
        str = "ImTransOpt ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "SumImages ";
        if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &sumImages);
			continue;
        }
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Read config file, time elapsed:\t%f s.\n",diftotal);
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int)ceil((EtaMax - EtaMin)/EtaBinSize);
	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = (double *) malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = (double *) malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = (double *) malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = (double *) malloc(nRBins*sizeof(*RBinsHigh));
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);

	int i,j;
	printf("NrTransOpt: %d\n",NrTransOpt);
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 2){printf("TransformationOptions can only be 0, 1, 2.\nExiting.\n");return 0;}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
    }

    /*Allocations!*/
	double *Image;
	double *AverageDark;
	cudaMallocHost((void **) &Image,NrPixelsY*NrPixelsZ*sizeof(*Image));
	AverageDark = (double *) calloc(NrPixelsY*NrPixelsZ,sizeof(*AverageDark));
	int32_t *ImageIn;
	int32_t *DarkIn;
	int32_t *ImageInT;
	int32_t *DarkInT;
	DarkIn = (int32_t *) malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkIn));
	DarkInT = (int32_t *) malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkInT));
	ImageIn = (int32_t *) malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageIn));
	ImageInT = (int32_t *) malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageInT));
	size_t pxSize = BYTES_PER_PIXEL;
	size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
	int nFrames;
	size_t sz;
	FILE *fd;
	char *darkFN;
	int nrdone = 0;
	if (argc > 2){
		darkFN = argv[2];
		fd = fopen(darkFN,"rb");
		fseek(fd,0L,SEEK_END);
		sz = ftell(fd);
		rewind(fd);
		nFrames = sz / (SizeFile);
		printf("Reading dark file:      %s, nFrames: %d.\n",darkFN,nFrames);
		for (i=0;i<nFrames;i++){
            fread(DarkInT,pxSize,NrPixelsY*NrPixelsZ,fd);
			DoImageTransformations(NrTransOpt,TransOpt,DarkInT,DarkIn,NrPixelsY,NrPixelsZ);
			if (makeMap == 1){
				mapMaskSize = NrPixelsY;
				mapMaskSize *= NrPixelsZ;
				mapMaskSize /= 32;
				mapMaskSize ++;
				mapMask = (int *) calloc(mapMaskSize,sizeof(*mapMask));
				for (j=0;j<NrPixelsY*NrPixelsZ;j++){
					if (DarkIn[j] == (int32_t) GapIntensity || DarkIn[j] == (int32_t) BadPxIntensity){
						SetBit(mapMask,j);
						nrdone++;
					}
				}
				printf("Nr mask pixels: %d\n",nrdone);
				makeMap = 0;
			}
			for(j=0;j<NrPixelsY*NrPixelsZ;j++) AverageDark[j] += (double)DarkIn[j]/nFrames;
		}
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Done reading the dark file, time elapsed till now:\t%f s.\n",diftotal);
	printf("Number of eta bins: %d, number of R bins: %d.\n",nEtaBins,nRBins);
    CHUNK_SIZE = SizeFile;
	TOTAL_MSG_SIZE = HEADER_SIZE + CHUNK_SIZE;
    printf("Image chunk size: %d bytes.\n",CHUNK_SIZE);
	size_t bigArrSize = nEtaBins*nRBins;
	double *devSumMatrix;
	double *IntArrPerFrame, *devIntArrPerFrame;
	double *PerFrameArr, *devPerFrameArr;
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Initializing device, getting allocation, time elapsed till now:\t%f s.\n",diftotal);
	if (sumImages == 1){
		gpuErrchk(cudaMalloc(&devSumMatrix,bigArrSize*sizeof(double)));
		gpuErrchk(cudaMemset(devSumMatrix,0,bigArrSize*sizeof(double)));
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Starting to allocate small arrays, time elapsed till now:\t%f s.\n",diftotal);
	IntArrPerFrame = (double *) calloc(bigArrSize,sizeof(*IntArrPerFrame));
	gpuErrchk(cudaMalloc(&devIntArrPerFrame,bigArrSize*sizeof(double)));
	PerFrameArr = (double *) malloc(bigArrSize*4*sizeof(*PerFrameArr));
	gpuErrchk(cudaMalloc(&devPerFrameArr,bigArrSize*4*sizeof(double)));
	double *devEtaBinsLow, *devEtaBinsHigh;
	double *devRBinsLow, *devRBinsHigh;
	gpuErrchk(cudaMalloc(&devEtaBinsLow,nEtaBins*sizeof(double)));
	gpuErrchk(cudaMalloc(&devEtaBinsHigh,nEtaBins*sizeof(double)));
	gpuErrchk(cudaMalloc(&devRBinsLow,nRBins*sizeof(double)));
	gpuErrchk(cudaMalloc(&devRBinsHigh,nRBins*sizeof(double)));
	gpuErrchk(cudaMemcpy(devEtaBinsLow,EtaBinsLow,nEtaBins*sizeof(double),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devEtaBinsHigh,EtaBinsHigh,nEtaBins*sizeof(double),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devRBinsLow,RBinsLow,nRBins*sizeof(double),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devRBinsHigh,RBinsHigh,nRBins*sizeof(double),cudaMemcpyHostToDevice));
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Allocated small arrays on device, will move the mapping information to device, time elapsed:\t%f s.\n",diftotal);
	// Move pxList and nPxList over to device.
	struct data *devPxList;
	int *devNPxList;
	gpuErrchk(cudaMalloc(&devPxList,szPxList));
	gpuErrchk(cudaMalloc(&devNPxList,szNPxList));
	gpuErrchk(cudaMemcpy(devPxList,pxList,szPxList,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devNPxList,nPxList,szNPxList,cudaMemcpyHostToDevice));
	double *devImage;
	cudaMalloc(&devImage,NrPixelsY*NrPixelsZ*sizeof(double));
	int *devMapMask;
	if (mapMaskSize !=0){
		gpuErrchk(cudaMalloc(&devMapMask,mapMaskSize*sizeof(int)));
		gpuErrchk(cudaMemcpy(devMapMask,mapMask,mapMaskSize*sizeof(int),cudaMemcpyHostToDevice));
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Starting frames now, time elapsed:\t%f s.\n",diftotal);

    /*Socket processing from here on.*/
    int server_fd;
    struct sockaddr_in server_addr;
    
    // Initialize the processing queue
    queue_init(&process_queue);
    
    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        exit(EXIT_FAILURE);
    }
    
    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    // Bind socket
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    
    // Listen for connections
    if (listen(server_fd, MAX_CONNECTIONS) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }
    
    printf("Server listening on port %d\n", PORT);
    printf("Expecting messages with %d-byte header and %d bytes (%d bytes total)\n", 
           HEADER_SIZE, CHUNK_SIZE, TOTAL_MSG_SIZE);
    
    // Create a thread for accepting new connections
    pthread_t accept_thread;
    pthread_create(&accept_thread, NULL, accept_connections, &server_fd);
    
	// Open two files: one to save the 1D lineout and the second to save the fitted peak parameters.
	// We want to append to both the files and they are binary.
	FILE *lineoutFile = fopen("lineout.bin", "ab");
	FILE *fitFile = fopen("fit.bin", "ab");
	if (lineoutFile == NULL || fitFile == NULL) {
		perror("Error opening output files");
		exit(EXIT_FAILURE);
	}

    // Main thread processes data from the queue
	clock_t t1, t2;
	double diffT=0;
	int firstFrame = 1;
	double *int1D;
	int1D = (double *) calloc(nRBins,sizeof(*int1D));
	double *R;
	R = (double *) calloc(nRBins,sizeof(*R));
	double *lineout;
	lineout = (double *)malloc(nRBins*2*sizeof(*lineout));
	// Main processing loop
    while (1) {
		t1 = clock();
        DataChunk chunk;
        queue_pop(&process_queue, &chunk);
        // Process the data
        memcpy(ImageInT,chunk.data,chunk.size*BYTES_PER_PIXEL);
		if ((NrTransOpt==0) || (NrTransOpt==1 && TransOpt[0]==0)){
			if (argc > 2){
				for (j=0;j<NrPixelsY*NrPixelsZ;j++){
					Image[j] = (double)ImageInT[j] - AverageDark[j];
				}
			} else {
				for (j=0;j<NrPixelsY*NrPixelsZ;j++){
					Image[j] = (double)ImageInT[j];
				}
			}
		} else {
			DoImageTransformations(NrTransOpt,TransOpt,ImageInT,ImageIn,NrPixelsY,NrPixelsZ);
			for (j=0;j<NrPixelsY*NrPixelsZ;j++){
				Image[j] = (double)ImageIn[j] - AverageDark[j];
			}
		}
		gpuErrchk(cudaMemset(devIntArrPerFrame,0,bigArrSize*sizeof(double)));
		gpuErrchk(cudaMemcpy(devImage,Image,NrPixelsY*NrPixelsZ*sizeof(double),cudaMemcpyHostToDevice));
		gpuErrchk(cudaDeviceSynchronize());
		int tPB = 512;
		int nrVox = (bigArrSize+tPB-1)/tPB;
		i = chunk.dataset_num;
		if (mapMaskSize==0)
			integrate_noMapMask <<<tPB,nrVox>>> (px,Lsd,bigArrSize,Normalize,sumImages,i,NrPixelsY, 
													mapMaskSize,devMapMask,nRBins,nEtaBins,devPxList, 
													devNPxList,devRBinsLow,devRBinsHigh,devEtaBinsLow,devEtaBinsHigh, 
													devImage,devIntArrPerFrame,devPerFrameArr,devSumMatrix);
		else 
			integrate_MapMask <<<tPB,nrVox>>> (px,Lsd,bigArrSize,Normalize,sumImages,i,NrPixelsY, 
													mapMaskSize,devMapMask,nRBins,nEtaBins,devPxList, 
													devNPxList,devRBinsLow,devRBinsHigh,devEtaBinsLow,devEtaBinsHigh, 
													devImage,devIntArrPerFrame,devPerFrameArr,devSumMatrix);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(IntArrPerFrame,devIntArrPerFrame,bigArrSize*sizeof(double),cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize());
		if (chunk.dataset_num==0 || firstFrame == 1){
			firstFrame = 0;
			gpuErrchk(cudaMemcpy(PerFrameArr,devPerFrameArr,bigArrSize*4*sizeof(double),cudaMemcpyDeviceToHost));
			gpuErrchk(cudaDeviceSynchronize());
			for (i=0;i<nRBins;i++){
				R[i] = (RBinsLow[i]+RBinsHigh[i])/2;
			}
			for (int r=0;r<nRBins;r++){
				lineout[r*2] = R[r];
			}
		}
		// Now we have IntArrPerFrame, we need to make it into a 1D.
		gpuErrchk(cudaDeviceSynchronize());
		memset(int1D,0,nRBins*sizeof(*int1D));
		double maxInt=-1;
		int maxIntLoc;
		int nNonZeros;
		if (multiplePeaks == 1){
			for (j=0;j<nRBins;j++){
				nNonZeros = 0;
				for (i=0;i<nEtaBins;i++){
					int1D[j] += IntArrPerFrame[j*nEtaBins+i];
					if (PerFrameArr[3*bigArrSize+(j*nEtaBins+i)] != 0)	nNonZeros ++;
				}
				if (nNonZeros != 0) int1D[j] /= (double)nNonZeros;
				if (int1D[j] > maxInt){
					maxInt = int1D[j];
					maxIntLoc = j;
				}
			}
		} else {
			for (j=0;j<nRBins;j++){
				nNonZeros = 0;
				for (i=0;i<nEtaBins;i++){
					int1D[j] += IntArrPerFrame[j*nEtaBins+i];
					if (PerFrameArr[3*bigArrSize+(j*nEtaBins+i)] != 0)	nNonZeros ++;
				}
				if (nNonZeros != 0) int1D[j] /= (double)nNonZeros;
			}
		}
		for (int r=0;r<nRBins;r++){
			lineout[r*2+1] = int1D[r];
		}
		// DSave linout to the file.
		fwrite(lineout,sizeof(double),nRBins*2,lineoutFile);
		fflush(lineoutFile);
		// // Send the lineout to the server
		// send_lineouts(chunk.dataset_num,nRBins*2,lineout);
		if (peakFit == 1){
			int peakCount;
			Peak *peaks = NULL;
			if (multiplePeaks ==1){
				if (nPeaks >0){
					peakCount = nPeaks;
					peaks = (Peak *)malloc(nPeaks*sizeof(Peak));
					for (int p = 0; p < nPeaks; p++){
						if (peakLocations[p] < R[0] || peakLocations[p] > R[nRBins-1]){
							printf("Warning: Peak location %f is out of bounds, R range was: %f %f.\n", peakLocations[p],R[0],R[nRBins-1]);
							continue;
						}
						// Find the closest radius value corresponding to the peak location
						int bestLoc = -1;
						double bestDiff = 1e6;
						for (int r = 0; r < nRBins; r++){
							double diff = fabs(R[r] - peakLocations[p]);
							if (diff < bestDiff){
								bestDiff = diff;
								bestLoc = r;
							}
						}
						if (bestLoc != -1){
							peaks[p].index = bestLoc;
							peaks[p].radius = R[bestLoc];
							peaks[p].intensity = int1D[bestLoc];
						} else {
							printf("Warning: Peak location %f not found in radius array.\n", peakLocations[p]);
						}
						printf("Peak %d: Index = %d, Radius = %f, Intensity = %f\n", 
							p, peaks[p].index, peaks[p].radius, peaks[p].intensity);
					}
				} else{
					double *smoothedData = (double *)malloc(nRBins*sizeof(double));
					peakCount = 0;
					// We need to find the number of peaks.
					if (doSmoothing == 1){
						int windowSize = 5;
						double minHeight = 0.0;
						int minDistance = 10;
						smoothData(int1D, smoothedData, nRBins, windowSize);
						peakCount = findPeaks(smoothedData, R, nRBins, &peaks, minHeight, minDistance);
					} else {
						peakCount = findPeaks(int1D, R, nRBins, &peaks, 0.0, 10); // No smoothing
					}
					printf("Found %d peaks in the data.\n", peakCount);
					for (int p = 0; p < peakCount; p++) {
						printf("Peak %d: Index = %d, Radius = %f, Intensity = %f\n", 
							p, peaks[p].index, peaks[p].radius, peaks[p].intensity);
					}
				}
			} else {
				// If not multiple peaks, just use the maximum intensity
				peaks = (Peak *)malloc(sizeof(Peak));
				peaks[0].index = maxIntLoc;
				peaks[0].radius = R[maxIntLoc];
				peaks[0].intensity = maxInt;
				peakCount = 1;
				printf("Single peak found at index %d, radius %f, intensity %f\n", 
					peaks[0].index, peaks[0].radius, peaks[0].intensity);
			}
			// Now do peak fitting for each peak in the data:
			int n = peakCount * 5;
			double *x, *lb, *ub;
			x = (double *)malloc(n*sizeof(double));
			lb = (double *)malloc(n*sizeof(double));
			ub = (double *)malloc(n*sizeof(double));
			for (int peakNr = 0; peakNr<peakCount; peakNr++){
				x[peakNr*5 + 0] = peaks[peakNr].intensity; // amplitude
				x[peakNr*5 + 1] = 0.0; // background
				x[peakNr*5 + 2] = 0.5; // mix
				x[peakNr*5 + 3] = peaks[peakNr].radius; // center
				x[peakNr*5 + 4] = 2; // sigma
				lb[peakNr*5 + 0] = 0.0; // lower bound for amplitude
				lb[peakNr*5 + 1] = -1.0; // lower bound for background
				lb[peakNr*5 + 2] = 0.0; // lower bound for mix
				lb[peakNr*5 + 3] = peaks[peakNr].radius - 10; // lower bound for center
				lb[peakNr*5 + 4] = 0.1; // lower bound for sigma
				ub[peakNr*5 + 0] = maxInt*2; // upper bound for amplitude
				ub[peakNr*5 + 1] = maxInt; // upper bound for background
				ub[peakNr*5 + 2] = 1.0; // upper bound for mix
				ub[peakNr*5 + 3] = peaks[peakNr].radius + 10; // upper bound for center
				ub[peakNr*5 + 4] = (R[nRBins-1]-R[0])/2; // upper bound for sigma
			}
			struct dataFit d;
			d.nrBins = nRBins;
			d.R = &R[0];
			d.Int = &int1D[0];
			struct dataFit *fitD;
			fitD = &d;
			void *trp = (struct dataFit *) fitD;
			nlopt_opt opt;
			opt = nlopt_create(NLOPT_LN_NELDERMEAD,5);
			nlopt_set_lower_bounds(opt,lb);
			nlopt_set_upper_bounds(opt,ub);
			nlopt_set_maxeval(opt,1000);
			nlopt_set_min_objective(opt,problem_function,trp);
			double minf;
			if (nlopt_optimize(opt,x,&minf) < 0){
				printf("nlopt failed!\n");
			} else {
				// Print the results
				for (int peakNr = 0; peakNr<peakCount; peakNr++){
					printf("Peak %d: Optimized parameters: amplitude = %f, background = %f, mix = %f, center = %f, sigma = %f\n", 
						peakNr, x[peakNr*5 + 0], x[peakNr*5 + 1], x[peakNr*5 + 2], x[peakNr*5 + 3], x[peakNr*5 + 4]);
				}
				// Print minf too.
				printf("Minimum objective function value: %f\n", minf);
			}
			nlopt_destroy(opt);
			// Save x to the fit file.
			fwrite(x,sizeof(double),n,fitFile);
			fflush(fitFile);
			// // Send the fit result to the server
			// send_fit_result(chunk.dataset_num,peakCount,x);
		}

		t2 = clock();
		diffT = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Did integration, took %lf s for this frame, frameNr: %d.\n",diffT,chunk.dataset_num);
        free(chunk.data);
    }
    
    pthread_join(accept_thread, NULL);
    close(server_fd);
    
    return 0;
}