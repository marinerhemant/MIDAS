//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// ~/opt/midascuda/cuda/bin/nvcc integrator.cu -o integrator -Xcompiler -g -arch sm_90 -gencode=arch=compute_90,code=sm_90 -I/scratch/s1iduser/sharma_tests/HDF5/include -L/scratch/s1iduser/sharma_tests/HDF5/lib -lhdf5_hl -lhdf5 -I/scratch/s1iduser/sharma_tests/LIBTIFF/include -L/scratch/s1iduser/sharma_tests/LIBTIFF/lib -ltiff -O3
// ~/opt/midascuda/cuda/bin/nvcc src/Integrator.cu -o ../bin/IntegratorGPU -Xcompiler -g -arch sm_90 -gencode=arch=compute_90,code=sm_90 -I/home/beams/S1IDUSER/.MIDAS/HDF5/include -L/home/beams/S1IDUSER/.MIDAS/HDF5/lib -lhdf5_hl -lhdf5 -I/home/beams/S1IDUSER/.MIDAS/LIBTIFF/include -L/home/beams/S1IDUSER/.MIDAS/LIBTIFF/lib -ltiff -O3
// ~/opt/midascuda/cuda_RHEL8/bin/nvcc src/Integrator.cu -o bin/IntegratorGPU -Xcompiler -g -arch sm_90 -gencode=arch=compute_90,code=sm_90 -I/home/beams/S1IDUSER/.MIDAS/HDF5/include -L/home/beams/S1IDUSER/.MIDAS/HDF5/lib -lhdf5_hl -lhdf5 -I/home/beams/S1IDUSER/.MIDAS/LIBTIFF/include -L/home/beams/S1IDUSER/.MIDAS/LIBTIFF/lib -ltiff -O3
// export LD_LIBRARY_PATH=/scratch/s1iduser/sharma_tests/HDF5/lib:/scratch/s1iduser/sharma_tests/LIBTIFF/lib:$LD_LIBRARY_PATH
// export LD_LIBRARY_PATH=/home/beams/S1IDUSER/.MIDAS/HDF5/lib:/home/beams/S1IDUSER/.MIDAS/LIBTIFF/lib:$LD_LIBRARY_PATH

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
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdint.h>
#include <tiffio.h>
#include <libgen.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <assert.h>
#include <cuda.h>
#include <sys/socket.h>
#include <sys/un.h>

int cuda_client_init();
int cuda_execute(const char* command);
void cuda_client_close();

#define SOCKET_PATH "/tmp/cuda_server_socket"

static int client_fd = -1;

int cuda_client_init() {
    client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (client_fd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    struct sockaddr_un address;
    memset(&address, 0, sizeof(struct sockaddr_un));
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, SOCKET_PATH, sizeof(address.sun_path) - 1);
    
    if (connect(client_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Connection failed");
        close(client_fd);
        client_fd = -1;
        return -1;
    }
    
    return 0;
}

int cuda_execute(const char* command) {
    if (client_fd < 0) {
        fprintf(stderr, "Not connected to CUDA server\n");
        return -1;
    }
    
    write(client_fd, command, strlen(command));
    
    char buffer[256];
    int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        printf("Server response: %s\n", buffer);
        return 0;
    }
    
    return -1;
}

void cuda_client_close() {
    if (client_fd >= 0) {
        close(client_fd);
        client_fd = -1;
    }
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

int fileReader (FILE *f,char fn[], int dType, int NrPixels, double *returnArr)
{
	int i;
	if (dType == 1){ // Binary with uint16
		uint16_t *readData;
		readData = (uint16_t *) calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 2){ // Binary with double
		double *readData;
		readData = (double *) calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 3){ // Binary with float
		float *readData;
		readData = (float *) calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 4){ // Binary with uint32
		uint32_t *readData;
		readData = (uint32_t *) calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 5){ // Binary with int32
		int32_t *readData;
		readData = (int32_t *) calloc(NrPixels,sizeof(*readData));
		fread(readData,NrPixels*sizeof(*readData),1,f);
		for (i=0;i<NrPixels;i++){
			returnArr[i] = (double) readData[i];
		}
		free(readData);
		return 0;
	} else if (dType == 6){ // TIFF with uint32 format
		TIFFErrorHandler oldhandler;
		oldhandler = TIFFSetWarningHandler(NULL);
		printf("%s\n",fn);
		TIFF* tif = TIFFOpen(fn, "r");
		TIFFSetWarningHandler(oldhandler);
		if (tif){
			uint32_t imagelength;
			tsize_t scanline;
			TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imagelength);
			scanline = TIFFScanlineSize(tif);
			tdata_t buf;
			buf = _TIFFmalloc(scanline);
			uint32_t *datar;
			int rnr;
			for (rnr=0;rnr<imagelength;rnr++){
				TIFFReadScanline(tif,buf,rnr,1);
				datar = (uint32_t*)buf;
				for (i=0;i<scanline/sizeof(uint32_t);i++){
					returnArr[rnr*(scanline/sizeof(uint32_t)) + i] = (double) datar[i];
				}
			}
			_TIFFfree(buf);
		}
		return 0;
	} else if (dType == 7){ // TIFF with uint8 format
		TIFFErrorHandler oldhandler;
		oldhandler = TIFFSetWarningHandler(NULL);
		printf("%s\n",fn);
		TIFF* tif = TIFFOpen(fn, "r");
		TIFFSetWarningHandler(oldhandler);
		if (tif){
			uint32_t imagelength;
			tsize_t scanline;
			TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imagelength);
			scanline = TIFFScanlineSize(tif);
			tdata_t buf;
			buf = _TIFFmalloc(scanline);
			uint8_t *datar;
			int rnr;
			for (rnr=0;rnr<imagelength;rnr++){
				TIFFReadScanline(tif,buf,rnr,1);
				datar = (uint8_t*)buf;
				for (i=0;i<scanline/sizeof(uint8_t);i++){
					if (datar[i] == 1){
						returnArr[rnr*(scanline/sizeof(uint8_t)) + i] = 1;
					}
				}
			}
			_TIFFfree(buf);
		}
		return 0;
	} else {
		return 127;
	}
}

int main(int argc, char **argv)
{
	if (cuda_client_init() < 0) {
        fprintf(stderr, "Failed to connect to CUDA server. Start the server at ~/opt/MIDAS/FF_HEDM/bin/midaS_cuda_server \n");
        return 1;
    }
	// cudaSetDevice(0);
	// printf("[%s] - Starting...\n", argv[0]);
	clock_t start0, end0;
	start0 = clock();
    double diftotal;
    if (argc < 3){
		printf("Usage: ./Integrator ParamFN ImageName (optional)DarkName\n"
		"Optional:\n\tDark file: dark correction with average of all dark frames"
		".\n");
		return(1);
	}
    //~ system("cp Map.bin nMap.bin /dev/shm");
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
	int HeadSize = 8192;
    int NrTransOpt=0;
    long long int GapIntensity=0, BadPxIntensity=0;
    int TransOpt[10];
    int makeMap = 0;
    size_t mapMaskSize = 0;
	int *mapMask, skipFrame=0;
	int dType = 1;
	char GapFN[4096], BadPxFN[4096], outputFolder[4096];
	int sumImages=0, separateFolder=0, newOutput=0;
	int haveOmegas = 0, chunkFiles=0, individualSave=1;
	double omeStart, omeStep;
	double Lam=0.172978, Polariz=0.99, SHpL=0.002, U=1.163, V=-0.126, W=0.063, X=0.0, Y=0.0, Z=0.0;
	while (fgets(aline,4096,paramFile) != NULL){
		str = "Z ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Z);
		}
		str = "Y ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Y);
		}
		str = "X ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &X);
		}
		str = "W ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &W);
		}
		str = "V ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &V);
		}
		str = "U ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &U);
		}
		str = "SH/L ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &SHpL);
		}
		str = "Polariz ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Polariz);
		}
		str = "Wavelength ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &Lam);
		}
		str = "GapFile ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, GapFN);
			makeMap = 2;
		}
		str = "OutFolder ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, outputFolder);
			separateFolder = 1;
		}
		str = "BadPxFile ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %s", dummy, BadPxFN);
			makeMap = 2;
		}
		str = "EtaBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &EtaBinSize);
		}
		str = "OmegaStart ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &omeStart);
		}
		str = "OmegaStep ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &omeStep);
			haveOmegas = 1;
		}
		str = "OmegaSumFrames ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &chunkFiles);
		}
		str = "SaveIndividualFrames ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &individualSave);
		}
		str = "NewOutput ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &newOutput);
		}
		str = "RBinSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %lf", dummy, &RBinSize);
		}
		str = "DataType ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &dType);
		}
		str = "HeadSize ";
		if (StartsWith(aline,str) == 1){
			sscanf(aline,"%s %d", dummy, &HeadSize);
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
		str = "SkipFrame ";
        if (StartsWith(aline,str) == 1){
            sscanf(aline,"%s %d", dummy, &skipFrame);
            continue;
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
	if (separateFolder!=0){
		struct stat st = {0};
		if (stat(outputFolder,&st)==-1){
			printf("Output folder '%s' did not exit. Making now.\n",outputFolder);
			mkdir(outputFolder,0700);
		}
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Read config file, time elapsed:\t%f s.\n",diftotal);
	if (newOutput!=2){
		printf("Only works with newOutput == 2. Will exit now.\n");
		return 1;
	}
	nRBins = (int) ceil((RMax-RMin)/RBinSize);
	nEtaBins = (int)ceil((EtaMax - EtaMin)/EtaBinSize);
	double *EtaBinsLow, *EtaBinsHigh;
	double *RBinsLow, *RBinsHigh;
	EtaBinsLow = (double *) malloc(nEtaBins*sizeof(*EtaBinsLow));
	EtaBinsHigh = (double *) malloc(nEtaBins*sizeof(*EtaBinsHigh));
	RBinsLow = (double *) malloc(nRBins*sizeof(*RBinsLow));
	RBinsHigh = (double *) malloc(nRBins*sizeof(*RBinsHigh));
	REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);

	int i,j,k,p;
	printf("NrTransOpt: %d\n",NrTransOpt);
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 2){printf("TransformationOptions can only be 0, 1, 2.\nExiting.\n");return 0;}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
    }
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
	// Image = (double *) malloc(NrPixelsY*NrPixelsZ*sizeof(*Image));
	size_t pxSize;
	if (dType == 1){ // Uint16
		pxSize = sizeof(uint16_t);
	} else if (dType == 2){ // Double
		pxSize = sizeof(double);
	} else if (dType == 3){ // Float
		pxSize = sizeof(float);
	} else if (dType == 4){ // Uint32
		pxSize = sizeof(uint32_t);
	} else if (dType == 5){ // Int32
		pxSize = sizeof(int32_t);
	} else if (dType == 6){ // Tiff Uint32
		pxSize = sizeof(uint32_t);
		HeadSize = 0;
	} else if (dType == 7){ // Tiff Uint8
		pxSize = sizeof(uint8_t);
		HeadSize = 0;
	} else if (dType == 8){ // HDF Unit16
		pxSize = sizeof(uint16_t);
		HeadSize = 0;
	}
	size_t SizeFile = pxSize * NrPixelsY * NrPixelsZ;
	int nFrames;
	size_t sz;
	int Skip = HeadSize;
	FILE *fp, *fd;
	char *darkFN;
	double *omeArr;
	int nrdone = 0;
	if (argc > 3 && dType!=8){
		darkFN = argv[3];
		fd = fopen(darkFN,"rb");
		fseek(fd,0L,SEEK_END);
		sz = ftell(fd);
		rewind(fd);
		nFrames = sz / (SizeFile);
		printf("Reading dark file:      %s, nFrames: %d, skipping first %d bytes.\n",darkFN,nFrames,Skip);
		fseek(fd,Skip,SEEK_SET);
		for (i=0;i<nFrames;i++){
			rc = fileReader(fd,darkFN,dType,NrPixelsY*NrPixelsZ,DarkInT);
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
	const char *DATASETNAME;
    hid_t file;
    hid_t dataset;  
	hid_t dataspace;
    hsize_t dims[3];
    int ndims;
    char *fn_hdf;
    uint16_t *all_images;
    int frame_dims2[3];
	if (dType == 8){
	    fn_hdf = argv[2];
	    file = H5Fopen(fn_hdf,H5F_ACC_RDONLY, H5P_DEFAULT);
		// READ DARK
		DATASETNAME = "exchange/dark";
	    dataset = H5Dopen(file, DATASETNAME,H5P_DEFAULT);
	    dataspace = H5Dget_space(dataset);
	    ndims  = H5Sget_simple_extent_dims(dataspace, dims, NULL);
	    printf("ndims: %d, dimensions %lu x %lu x %lu. Allocating big array.\n",
		   ndims, (unsigned long)(dims[0]), (unsigned long)(dims[1]), (unsigned long)(dims[2]));
		int frame_dims[3] = {(int)dims[0],(int)dims[1],(int)dims[2]};	
		uint16_t *data = (uint16_t *) calloc(frame_dims[0]*frame_dims[1]*frame_dims[2],sizeof(uint16_t));
		printf("Reading file: %lu bytes from dataset: %s.\n",(unsigned long) dims[0]*dims[1]*dims[2]*2,DATASETNAME);
		herr_t status_n = H5Dread(dataset,H5T_STD_U16LE,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
		for (i=skipFrame;i<frame_dims[0];i++){
			for (j=0;j<frame_dims[1];j++){
				for (k=0;k<frame_dims[2];k++){
					DarkInT[j*frame_dims[2]+k] = ((double)data[i*(frame_dims[1]*frame_dims[2])+j*frame_dims[2]+k])/frame_dims[0];
				}
			}
		}
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
		// READ DATA, STORED IN all_images array
		DATASETNAME = "exchange/data";
	    dataset = H5Dopen(file, DATASETNAME,H5P_DEFAULT);
	    dataspace = H5Dget_space(dataset);
	    ndims  = H5Sget_simple_extent_dims(dataspace, dims, NULL);
	    printf("ndims: %d, dimensions %lu x %lu x %lu. Allocating big array.\n",
		   ndims, (unsigned long)(dims[0]), (unsigned long)(dims[1]), (unsigned long)(dims[2]));
		frame_dims2[0] = dims[0];
		frame_dims2[1] = dims[1];
		frame_dims2[2] = dims[2];
		all_images = (uint16_t *) calloc(frame_dims2[0]*frame_dims2[1]*frame_dims2[2],sizeof(uint16_t));
		printf("Reading file: %lu bytes from dataset: %s.\n",(unsigned long) dims[0]*dims[1]*dims[2]*2,DATASETNAME);
		status_n = H5Dread(dataset,H5T_STD_U16LE,H5S_ALL,H5S_ALL,H5P_DEFAULT,all_images);
	}
	if (makeMap == 2){
		mapMaskSize = NrPixelsY;
		mapMaskSize *= NrPixelsZ;
		mapMaskSize /= 32;
		mapMaskSize ++;
		mapMask = (int *) calloc(mapMaskSize,sizeof(*mapMask));
		double *mapper;
		mapper = (double *) calloc(NrPixelsY*NrPixelsZ,sizeof(*mapper));
		double *mapperOut;
		mapperOut = (double *) calloc(NrPixelsY*NrPixelsZ,sizeof(*mapperOut));
		fileReader(fd,GapFN,7,NrPixelsY*NrPixelsZ,mapper);
		DoImageTransformations(NrTransOpt,TransOpt,mapper,mapperOut,NrPixelsY,NrPixelsZ);
		for (i=0;i<NrPixelsY*NrPixelsZ;i++){
			if (mapperOut[i] != 0){
				SetBit(mapMask,i);
				mapperOut[i] = 0;
				nrdone++;
			}
		}
		fileReader(fd,BadPxFN,7,NrPixelsY*NrPixelsZ,mapper);
		DoImageTransformations(NrTransOpt,TransOpt,mapper,mapperOut,NrPixelsY,NrPixelsZ);
		for (i=0;i<NrPixelsY*NrPixelsZ;i++){
			if (mapperOut[i] != 0){
				SetBit(mapMask,i);
				mapperOut[i] = 0;
				nrdone++;
			}
		}
		printf("Nr mask pixels: %d\n",nrdone);
	}
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	printf("Looking at data file, time elapsed till now:\t%f s.\n",diftotal);
	char *imageFN;
	imageFN = argv[2];
	fp = fopen(imageFN,"rb");
	fseek(fp,0L,SEEK_END);
	sz = ftell(fp);
	rewind(fp);
	fseek(fp,Skip,SEEK_SET);
	nFrames = (sz-Skip) / SizeFile;
	printf("Number of eta bins: %d, number of R bins: %d. Number of frames in the file: %d\n",nEtaBins,nRBins,(int)nFrames);
	char outfn2[4096];
	hid_t file_id;
	size_t bigArrSize = nEtaBins*nRBins;
	double firstOme;
	double *chunkArr;
	if (chunkFiles>0){
		chunkArr = (double *) calloc(bigArrSize,sizeof(*chunkArr));
	}
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


	if (haveOmegas==1){
		omeArr = (double *) malloc(nFrames*sizeof(*omeArr));
		for (i=0;i<nFrames;i++){
			omeArr[i] = omeStart + i*omeStep;
		}
	}

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
	clock_t t1, t2,t3,t4,t5,t6;
	double diffT=0, diffT2=0,diffT3=0;
	for (i=0;i<nFrames;i++){
		if (chunkFiles>0){
			if ((i%chunkFiles) == 0){
				memset(chunkArr,0,bigArrSize*sizeof(*chunkArr));
				firstOme = omeArr[i];
			}
		}
		printf("Processing frame number: %d of %d of file %s.\n",i+1,nFrames,imageFN);
		t3 = clock();
		if (dType!=8){
			rc = fileReader(fp,imageFN,dType,NrPixelsY*NrPixelsZ,ImageInT);
		} else {
			for (j=0;j<frame_dims2[1];j++){
				for (k=0;k<frame_dims2[2];k++){
					ImageInT[j*frame_dims2[2]+k] = ((pixelvalue)all_images[(i+skipFrame)*(frame_dims2[1]*frame_dims2[2])+j*frame_dims2[2]+k]);
				}
			}
		}
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
		if (i==0){
			if (separateFolder==0) sprintf(outfn2,"%s.caked.hdf",imageFN);
			else{
				char fn2[4096];
				sprintf(fn2,"%s",imageFN);
				char *bnname;
				bnname = basename(fn2);
				sprintf(outfn2,"%s/%s.caked.hdf",outputFolder,bnname);
			}
			file_id = H5Fcreate(outfn2, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			if (individualSave==1) H5Gcreate(file_id,"/IntegrationResult", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			if (chunkFiles>0) {
				char gName [2048];
				sprintf(gName,"/OmegaSumFrame",chunkFiles);
				H5Gcreate(file_id,gName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			}
		}
		t4 = clock();
		diffT2 += ((double)(t4-t3))/CLOCKS_PER_SEC;
		t1 = clock();
		gpuErrchk(cudaMemset(devIntArrPerFrame,0,bigArrSize*sizeof(double)));
		gpuErrchk(cudaMemcpy(devImage,Image,NrPixelsY*NrPixelsZ*sizeof(double),cudaMemcpyHostToDevice));
		gpuErrchk(cudaDeviceSynchronize());
		int nrVox = (bigArrSize+2047)/2048, tPB = 2048;
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
		t2 = clock();
		diffT += ((double)(t2-t1))/CLOCKS_PER_SEC;
		t5 = clock();
		if (i==0){
			gpuErrchk(cudaMemcpy(PerFrameArr,devPerFrameArr,bigArrSize*4*sizeof(double),cudaMemcpyDeviceToHost));
			gpuErrchk(cudaDeviceSynchronize());
			hsize_t dims[3] = {(unsigned long long)4,(unsigned long long)nRBins,(unsigned long long)nEtaBins};
			herr_t status_f = H5LTmake_dataset_double(file_id, "/REtaMap", 3, dims, PerFrameArr);
			H5LTset_attribute_int(file_id, "/REtaMap", "nEtaBins", &nEtaBins, 1);
			H5LTset_attribute_int(file_id, "/REtaMap", "nRBins", &nRBins, 1);
			H5LTset_attribute_string(file_id, "/REtaMap", "Header", "Radius,2Theta,Eta,BinArea");
			H5LTset_attribute_string(file_id, "/REtaMap", "Units", "Pixels,Degrees,Degrees,Pixels");
		}
		gpuErrchk(cudaDeviceSynchronize());
		hsize_t dim[2] = {(unsigned long long)nRBins,(unsigned long long)nEtaBins};
		char dsetName[1024];
		if (individualSave==1) {
			sprintf(dsetName,"/IntegrationResult/FrameNr_%d",i);
			H5LTmake_dataset_double(file_id, dsetName, 2, dim, IntArrPerFrame);
			H5LTset_attribute_double(file_id, dsetName, "omega", &omeArr[i], 1);
			H5LTset_attribute_string(file_id, dsetName, "Header", "Radius,Eta");
			H5LTset_attribute_string(file_id, dsetName, "Units", "Pixels,Degrees");
		}
		if (chunkFiles>0){
			for (p=0;p<bigArrSize;p++) chunkArr[p] += IntArrPerFrame[p];
			if (((i+1)%chunkFiles) == 0 || i==(nFrames-1)) {
				hsize_t dim_chunk[2] = {(unsigned long long)nRBins,(unsigned long long)nEtaBins};
				char chunkSetName[1024];
				sprintf(chunkSetName,"/OmegaSumFrame/LastFrameNumber_%d",i);
				H5LTmake_dataset_double(file_id, chunkSetName, 2, dim_chunk, chunkArr);
				H5LTset_attribute_int(file_id, chunkSetName, "LastFrameNumber", &i, 1);
				int nSum = (int)((omeArr[i] - firstOme)/omeStep)  + 1;
				H5LTset_attribute_int(file_id, chunkSetName, "Number Of Frames Summed", &nSum, 1);
				H5LTset_attribute_double(file_id, chunkSetName, "FirstOme", &firstOme, 1);
				H5LTset_attribute_double(file_id, chunkSetName, "LastOme", &omeArr[i], 1);
			}
		}
		t6 = clock();
		diffT3 += ((double)(t6-t5))/CLOCKS_PER_SEC;
	}
	if (haveOmegas==1){
		hsize_t dimome[1] = {(unsigned long long)nFrames};
		H5LTmake_dataset_double(file_id, "/Omegas", 1, dimome,omeArr);
		H5LTset_attribute_string(file_id, "/Omegas", "Units", "Degrees");
	}
	if (sumImages == 1){
		double *sumArr;
		sumArr = (double *) malloc(bigArrSize*sizeof(*sumArr));
		gpuErrchk(cudaMemcpy(sumArr,devSumMatrix,bigArrSize*sizeof(double),cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize());
		hsize_t dimsum[2] = {(unsigned long long)nRBins,(unsigned long long)nEtaBins};
		H5LTmake_dataset_double(file_id, "/SumFrames", 2, dimsum,sumArr);
		H5LTset_attribute_string(file_id, "/SumFrames", "Header", "Radius,Eta");
		H5LTset_attribute_string(file_id, "/SumFrames", "Units", "Pixels,Degrees");
		H5LTset_attribute_int(file_id, "/SumFrames", "nFrames", &nFrames,1);
		free(sumArr);
	}
	H5Gcreate(file_id,"InstrumentParameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimval[1] = {1};
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/Polariz", 1, dimval, &Polariz);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/Lam", 1, dimval, &Lam);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/SH_L", 1, dimval, &SHpL);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/U", 1, dimval, &U);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/V", 1, dimval, &V);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/W", 1, dimval, &W);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/X", 1, dimval, &X);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/Y", 1, dimval, &Y);
	H5LTmake_dataset_double(file_id, "/InstrumentParameters/Z", 1, dimval, &Z);
	herr_t status_f2 = H5Fclose (file_id);
	end0 = clock();
	diftotal = ((double)(end0-start0))/CLOCKS_PER_SEC;
	double TP = NrPixelsY*NrPixelsZ*nFrames;
	TP /= 1000000;
	TP /= diffT;
	printf("Time taken in reading and preparing files:\t%lfs, time taken in writing files:\t%lfs.\n",diffT2,diffT3);
	printf("Integration throughput:\t%zu MPixels/s, time taken for integration:\t%lfs, total time elapsed:\t%f s.\n",(size_t) TP,diffT,diftotal);
	return 0;
}
