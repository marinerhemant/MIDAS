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

#define PORT 5000           // Changed port to 5000
#define MAX_CONNECTIONS 10
size_t CHUNK_SIZE;
#define MAX_QUEUE_SIZE 100  // Maximum number of chunks in the queue, should not be too large, else segfaults.
#define HEADER_SIZE sizeof(uint16_t)  // Size of dataset number
size_t TOTAL_MSG_SIZE;
#define BYTES_PER_PIXEL 2

// Structure for our data chunks
typedef struct {
    uint16_t dataset_num;  // Dataset number
    uint16_t *data;
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
int queue_push(ProcessQueue *queue, uint16_t dataset_num, uint16_t *data, size_t num_values) {
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
        // dataset_num = ntohs(dataset_num);  // Convert from network to host byte order
        
        // Allocate memory for the data
        uint16_t *data = (uint16_t*)malloc(CHUNK_SIZE * sizeof(uint16_t));
        if (!data) {
            perror("Memory allocation failed");
            break;
        }
        int maxInt = -1;
        // Convert data from network byte order to host byte order
        for (int i = 0; i < CHUNK_SIZE/BYTES_PER_PIXEL; i++) {
            // uint16_t network_value;
            // memcpy(&network_value, buffer + HEADER_SIZE + (i * sizeof(uint16_t)), sizeof(uint16_t));
            // data[i] = ntohs(network_value);
			uint16_t value;
			memcpy(&value, buffer + HEADER_SIZE + (i * sizeof(uint16_t)), sizeof(uint16_t));
			data[i] = value;  // No conversion
			if (data[i] > maxInt) maxInt = data[i];
		}
		printf("Max intensity: %d\n",maxInt);
        
        // Add the data to the processing queue
        queue_push(&process_queue, dataset_num, data, CHUNK_SIZE);
        printf("Received dataset #%u with %d uint16_t values\n", dataset_num, CHUNK_SIZE);
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

typedef double pixelvalue;

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
	printf("BigArrSize: %d Idx: %d\n",(int)idx,bigArrSize);
	if (idx < bigArrSize){
		int l;
		double Intensity=0, totArea=0;
		struct data ThisVal;
		long long int nPixels, dataPos, testPos;
		nPixels = dNPxList[2*idx + 0];
		dataPos = dNPxList[2*idx + 1];
		printf("nPixels: %d, dataPos: %d\n",(int)nPixels,(int)dataPos);
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

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *ImageIn, pixelvalue *ImageOut, int NrPixelsY, int NrPixelsZ)
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
	struct dataFit *d = (struct dataFit *) my_func_data;
	double amp = x[0];
	double bg = x[1];
	double mix = x[2];
	double cen = x[3];
	double sig = x[4];
	int nrPoints = d->nrBins;
	double *Rs;
	Rs = &(d->R[0]);
	double *Ints;
	Ints = &(d->Int[0]);
	int i;
	double error = 0;
	double diff, gauss, lorentz, thisInt;
	for (i=0;i<nrPoints;i++){
		diff = Rs[i] - cen;
		gauss = exp(-diff*diff/(2*sig*sig))/(sig*sqrt(2*M_PI));
		lorentz = 1/((M_PI*sig)*(1+diff*diff/(sig*sig)));
		thisInt = bg + amp*(mix*gauss + (1-mix)*lorentz);
		error += (thisInt - Ints[i])*(thisInt - Ints[i]);
	}
	return error;
}

int main(int argc, char *argv[]){
    if (argc < 2){
		printf("Usage: ./Integrator ParamFN (optional)DarkName\n"
		"Optional:\n\tDark file: dark correction with average of all dark frames"
		".\n\tDark must be in a binary format, with uint16 dataType."
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
	int dType = 1;
	int sumImages=0;
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
	pixelvalue *ImageIn;
	pixelvalue *DarkIn;
	pixelvalue *ImageInT;
	pixelvalue *DarkInT;
	double *AverageDark;
	DarkIn = (pixelvalue *) malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkIn));
	DarkInT = (pixelvalue *) malloc(NrPixelsY*NrPixelsZ*sizeof(*DarkInT));
	AverageDark = (pixelvalue *) calloc(NrPixelsY*NrPixelsZ,sizeof(*AverageDark));
	ImageIn = (pixelvalue *) malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageIn));
	ImageInT = (pixelvalue *) malloc(NrPixelsY*NrPixelsZ*sizeof(*ImageInT));
	cudaMallocHost((void **) &Image,NrPixelsY*NrPixelsZ*sizeof(*Image));
	size_t pxSize = BYTES_PER_PIXEL;
	size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
	int nFrames;
	size_t sz;
	FILE *fd;
	char *darkFN;
	int nrdone = 0;
	if (argc > 3){
		darkFN = argv[3];
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
					if (DarkIn[j] == (pixelvalue) GapIntensity || DarkIn[j] == (pixelvalue) BadPxIntensity){
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
    printf("Expecting messages with %d-byte header and %d uint16_t values (%d bytes total)\n", 
           HEADER_SIZE, CHUNK_SIZE, TOTAL_MSG_SIZE);
    
    // Create a thread for accepting new connections
    pthread_t accept_thread;
    pthread_create(&accept_thread, NULL, accept_connections, &server_fd);
    
    // Main thread processes data from the queue
	clock_t t1, t2;
	double diffT=0;
	int firstFrame = 1;
	double *int1D;
	int1D = (double *) calloc(nRBins,sizeof(*int1D));
	double *area1D;
	area1D = (double *) calloc(nRBins,sizeof(*area1D));
	double *R;
	R = (double *) calloc(nRBins,sizeof(*R));
    while (1) {
		t1 = clock();
        DataChunk chunk;
        queue_pop(&process_queue, &chunk);
        // Process the data
        memcpy(ImageInT,chunk.data,chunk.size);
		if ((NrTransOpt==0) || (NrTransOpt==1 && TransOpt[0]==0)){
			if (argc > 3 && dType!=8){
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
		}
		// Now we have IntArrPerFrame, we need to make it into a 1D.
		gpuErrchk(cudaDeviceSynchronize());
		memset(int1D,0,nRBins*sizeof(*int1D));
		memset(area1D,0,nRBins*sizeof(*area1D));
		memset(R,0,nRBins*sizeof(*R));
		double maxInt=-1;
		int maxIntLoc;
		for (j=0;j<nRBins;j++){
			for (i=0;i<nEtaBins;i++){
				int1D[j] += IntArrPerFrame[j*nEtaBins+i];
				area1D[j] += PerFrameArr[3*bigArrSize+(j*nEtaBins+i)];
			}
			if (area1D[j] != 0) int1D[j] /= area1D[j];
			// printf("%lf %lf\n",int1D[j],area1D[j]);
			if (int1D[j] > maxInt){
				maxInt = int1D[j];
				maxIntLoc = j;
			}
			R[j] = (RBinsLow[j]+RBinsHigh[j])/2;
		}
		printf("Max intensity: %lf at R: %lf\n",maxInt,R[maxIntLoc]);
		// We have the 1D array, now fit it with a peak shape.
		double x[5] = {maxInt,(int1D[0]+int1D[nRBins-1])/2,0.5,R[maxIntLoc],0.1}; // amp, bg, mix, cen, sig
		double lb[5] = {0.0,-1.0,0.0,R[0],0.0};
		double ub[5] = {maxInt*2,maxInt,1.0,R[nRBins-1],R[nRBins-1]-R[0]};
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
		nlopt_set_maxeval(opt,10);
		nlopt_set_min_objective(opt,problem_function,trp);
		double minf;
		if (nlopt_optimize(opt,x,&minf) < 0){
			printf("nlopt failed!\n");
		} else {
			printf("found minimum at f(%f,%f,%f,%f,%f) = %0.10f\n",x[0],x[1],x[2],x[3],x[4],minf);
		}
		nlopt_destroy(opt);

		t2 = clock();
		diffT = ((double)(t2-t1))/CLOCKS_PER_SEC;
		printf("Did integration, took %lf s for this frame, frameNr: %d.\n",diffT,chunk.dataset_num);
        
        // Free the data
        free(chunk.data);
    }
    
    // This code is not reached in this example
    pthread_join(accept_thread, NULL);
    close(server_fd);
    
    return 0;
}