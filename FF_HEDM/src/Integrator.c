//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Integrator.c
//
// Hemant Sharma
// Dt: 2017/07/26
//

#include "FileReader.h"
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef double pixelvalue;

#define SetBit(A, k) (A[(k / 32)] |= (1 << (k % 32)))
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
#define rad2deg 57.2957795130823

static inline double atand(double x) { return rad2deg * (atan(x)); }

// check() - using MIDAS_CHECK_DEFINED guard
#ifndef MIDAS_CHECK_DEFINED
#define MIDAS_CHECK_DEFINED
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
#endif

static inline double **allocMatrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

struct data {
  int y;
  int z;
  double frac;
};

struct data *pxList;
int *nPxList;

static int CopyToShm(const char *srcFn, char *destFn, size_t destLen) {
  int srcFd = open(srcFn, O_RDONLY);
  if (srcFd < 0) {
    // If we can't open local file, return error
    return -1;
  }

  // Create unique filename in /dev/shm to prevent collisions
  // Use PID and timestamp for uniqueness
  snprintf(destFn, destLen, "/dev/shm/Integrator_%s_%d_%ld.bin", srcFn,
           getpid(), time(NULL));

  // Open destination in /dev/shm
  // O_EXCL ensures we don't clobber an existing file (unlikely with PID/Time)
  int destFd = open(destFn, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (destFd < 0) {
    // If /dev/shm is full or not accessible, close src and return error
    close(srcFd);
    return -1;
  }

  // IMMEDIATELY unlink the file!
  // This is the key safety feature. The file remains available via destFd,
  // but is removed from the filesystem namespace. The OS will automatically
  // reclaim the space when this process exits (cleanly or crash).
  unlink(destFn);

  // Copy data
  char buffer[8192];
  ssize_t bytesRead, bytesWritten;
  while ((bytesRead = read(srcFd, buffer, sizeof(buffer))) > 0) {
    char *ptr = buffer;
    while (bytesRead > 0) {
      bytesWritten = write(destFd, ptr, bytesRead);
      if (bytesWritten < 0) {
        // Write error (disk full?)
        close(srcFd);
        close(destFd);
        return -1;
      }
      bytesRead -= bytesWritten;
      ptr += bytesWritten;
    }
  }

  close(srcFd);

  // Rewind destFd for subsequent mmap/reading
  lseek(destFd, 0, SEEK_SET);

  return destFd;
}

int ReadBins() {
  int fd;
  struct stat s;
  int status;
  size_t size;
  const char *file_name = "Map.bin";
  char shm_name[256];

  // Try to copy to /dev/shm first
  fd = CopyToShm(file_name, shm_name, sizeof(shm_name));

  if (fd >= 0) {
    printf("Using optimized /dev/shm cache for %s (internal: %s)\n", file_name,
           shm_name);
  } else {
    printf("Warning: Could not cache %s to /dev/shm. Falling back to disk.\n",
           file_name);
    fd = open(file_name, O_RDONLY);
    check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  }

  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  int sizelen = 2 * (int)sizeof(int) + (int)sizeof(double);
  printf("Map size in bytes: %lld, each element size: %d, total elements: "
         "%lld. \n",
         (long long int)size, sizelen, (long long int)(size / sizelen));
  pxList = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(pxList == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));
  // We can close fd now, mmap keeps the reference
  close(fd);

  int fd2;
  struct stat s2;
  int status2;
  const char *file_name2 = "nMap.bin";

  // Try to copy nMap to /dev/shm
  fd2 = CopyToShm(file_name2, shm_name, sizeof(shm_name));

  if (fd2 >= 0) {
    printf("Using optimized /dev/shm cache for %s (internal: %s)\n", file_name2,
           shm_name);
  } else {
    printf("Warning: Could not cache %s to /dev/shm. Falling back to disk.\n",
           file_name2);
    fd2 = open(file_name2, O_RDONLY);
    check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  }

  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2 = s2.st_size;
  nPxList = mmap(0, size2, PROT_READ, MAP_SHARED, fd2, 0);
  printf("nMap size in bytes: %lld, each element size: %d, total elements: "
         "%lld. \n",
         (long long int)size2, 2 * (int)sizeof(int),
         2 * (long long int)(size2 / sizeof(int)));
  fflush(stdout);
  check(nPxList == MAP_FAILED, "mmap %s failed: %s", file_name,
        strerror(errno));
  close(fd2);
  return 1;
}

static inline int StartsWith(const char *a, const char *b) {
  if (strncmp(a, b, strlen(b)) == 0)
    return 1;
  return 0;
}

static inline void Transposer(double *x, int n1, int n2, double *y) {
  int i, j;
  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      y[(i * n2) + j] = x[(j * n1) + i];
    }
  }
}

static inline void REtaMapper(double Rmin, double EtaMin, int nEtaBins,
                              int nRBins, double EtaBinSize, double RBinSize,
                              double *EtaBinsLow, double *EtaBinsHigh,
                              double *RBinsLow, double *RBinsHigh) {
  int i, j, k, l;
  for (i = 0; i < nEtaBins; i++) {
    EtaBinsLow[i] = EtaBinSize * i + EtaMin;
    EtaBinsHigh[i] = EtaBinSize * (i + 1) + EtaMin;
  }
  for (i = 0; i < nRBins; i++) {
    RBinsLow[i] = RBinSize * i + Rmin;
    RBinsHigh[i] = RBinSize * (i + 1) + Rmin;
  }
}

static inline void DoImageTransformations(int NrTransOpt, int TransOpt[10],
                                          pixelvalue *ImageIn,
                                          pixelvalue *ImageOut, int NrPixelsY,
                                          int NrPixelsZ) {
  int i, j, k, l, m;
  if (NrTransOpt == 0 || (NrTransOpt == 1 && TransOpt[0] == 0)) {
    memcpy(ImageOut, ImageIn,
           NrPixelsY * NrPixelsZ * sizeof(*ImageIn)); // Nothing to do
    return;
  }
  for (i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] == 1) {
      for (k = 0; k < NrPixelsY; k++) {
        for (l = 0; l < NrPixelsZ; l++) {
          ImageOut[l * NrPixelsY + k] =
              ImageIn[l * NrPixelsY + (NrPixelsY - k - 1)]; // Invert Y
        }
      }
    } else if (TransOpt[i] == 2) {
      for (k = 0; k < NrPixelsY; k++) {
        for (l = 0; l < NrPixelsZ; l++) {
          ImageOut[l * NrPixelsY + k] =
              ImageIn[(NrPixelsZ - l - 1) * NrPixelsY + k]; // Invert Z
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  clock_t start, end, start0, end0;
  start0 = clock();
  double diftotal;
  if (argc < 3) {
    printf("Usage: ./Integrator ParamFN ImageName (optional)DarkName\n"
           "Optional:\n\tDark file: dark correction with average of all dark "
           "frames"
           ".\n");
    return (1);
  }
  //~ system("cp Map.bin nMap.bin /dev/shm");
  int rc = ReadBins();
  double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
  int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
  int nEtaBins, nRBins;
  char *ParamFN;
  FILE *paramFile;
  ParamFN = argv[1];
  char aline[4096], dummy[4096], *str;
  paramFile = fopen(ParamFN, "r");
  int HeadSize = 8192;
  int NrTransOpt = 0;
  long long int GapIntensity = 0, BadPxIntensity = 0;
  int TransOpt[10];
  int makeMap = 0;
  size_t mapMaskSize = 0;
  int *mapMask, skipFrame = 0;
  int dType = 1;
  char GapFN[4096], BadPxFN[4096], outputFolder[4096];
  int sumImages = 0, separateFolder = 0, newOutput = 0;
  int haveOmegas = 0, chunkFiles = 0, individualSave = 1;
  double omeStart, omeStep;
  double Lam = 0.172978, Polariz = 0.99, SHpL = 0.002, U = 1.163, V = -0.126,
         W = 0.063, X = 0.0, Y = 0.0, Z = 0.0;
  char darkDataset[1024];
  char dataDataset[1024];
  sprintf(darkDataset, "exchange/dark");
  sprintf(dataDataset, "exchange/data");

  while (fgets(aline, 4096, paramFile) != NULL) {
    str = "darkDataset ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %s", dummy, darkDataset);
    }
    str = "dataDataset ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %s", dummy, dataDataset);
    }
    str = "Z ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &Z);
    }
    str = "Y ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &Y);
    }
    str = "X ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &X);
    }
    str = "W ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &W);
    }
    str = "V ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &V);
    }
    str = "U ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &U);
    }
    str = "SH/L ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &SHpL);
    }
    str = "Polariz ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &Polariz);
    }
    str = "Wavelength ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &Lam);
    }
    str = "GapFile ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %s", dummy, GapFN);
      makeMap = 2;
    }
    str = "OutFolder ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %s", dummy, outputFolder);
      separateFolder = 1;
    }
    str = "BadPxFile ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %s", dummy, BadPxFN);
      makeMap = 2;
    }
    str = "EtaBinSize ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &EtaBinSize);
    }
    str = "OmegaStart ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &omeStart);
    }
    str = "OmegaStep ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &omeStep);
      haveOmegas = 1;
    }
    str = "OmegaSumFrames ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &chunkFiles);
    }
    str = "SaveIndividualFrames ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &individualSave);
    }
    str = "NewOutput ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &newOutput);
    }
    str = "RBinSize ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &RBinSize);
    }
    str = "DataType ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &dType);
    }
    str = "HeadSize ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &HeadSize);
    }
    str = "RMax ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &RMax);
    }
    str = "RMin ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &RMin);
    }
    str = "EtaMax ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &EtaMax);
    }
    str = "EtaMin ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &EtaMin);
    }
    str = "Lsd ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &Lsd);
    }
    str = "px ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lf", dummy, &px);
    }
    str = "NrPixelsY ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
    }
    str = "NrPixelsZ ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
    }
    str = "Normalize ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &Normalize);
    }
    str = "SkipFrame ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &skipFrame);
      continue;
    }
    str = "NrPixels ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &NrPixelsY);
      sscanf(aline, "%s %d", dummy, &NrPixelsZ);
    }
    str = "GapIntensity ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lld", dummy, &GapIntensity);
      makeMap = 1;
      continue;
    }
    str = "BadPxIntensity ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %lld", dummy, &BadPxIntensity);
      makeMap = 1;
      continue;
    }
    str = "ImTransOpt ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &TransOpt[NrTransOpt]);
      NrTransOpt++;
      continue;
    }
    str = "SumImages ";
    if (StartsWith(aline, str) == 1) {
      sscanf(aline, "%s %d", dummy, &sumImages);
      continue;
    }
  }
  if (separateFolder != 0) {
    struct stat st = {0};
    if (stat(outputFolder, &st) == -1) {
      printf("Output folder '%s' did not exit. Making now.\n", outputFolder);
      mkdir(outputFolder, 0700);
    }
  }
  nRBins = (int)ceil((RMax - RMin) / RBinSize);
  nEtaBins = (int)ceil((EtaMax - EtaMin) / EtaBinSize);
  double *EtaBinsLow, *EtaBinsHigh;
  double *RBinsLow, *RBinsHigh;
  EtaBinsLow = malloc(nEtaBins * sizeof(*EtaBinsLow));
  EtaBinsHigh = malloc(nEtaBins * sizeof(*EtaBinsHigh));
  RBinsLow = malloc(nRBins * sizeof(*RBinsLow));
  RBinsHigh = malloc(nRBins * sizeof(*RBinsHigh));
  REtaMapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize, EtaBinsLow,
             EtaBinsHigh, RBinsLow, RBinsHigh);

  int i, j, k, l, p;
  printf("NrTransOpt: %d\n", NrTransOpt);
  for (i = 0; i < NrTransOpt; i++) {
    if (TransOpt[i] < 0 || TransOpt[i] > 2) {
      printf("TransformationOptions can only be 0, 1, 2.\nExiting.\n");
      return 0;
    }
    printf("TransformationOptions: %d ", TransOpt[i]);
    if (TransOpt[i] == 0)
      printf("No change.\n");
    else if (TransOpt[i] == 1)
      printf("Flip Left Right.\n");
    else if (TransOpt[i] == 2)
      printf("Flip Top Bottom.\n");
  }
  double *Image;
  pixelvalue *ImageIn;
  pixelvalue *DarkIn;
  pixelvalue *ImageInT;
  pixelvalue *DarkInT;
  double *AverageDark;
  DarkIn = malloc(NrPixelsY * NrPixelsZ * sizeof(*DarkIn));
  DarkInT = malloc(NrPixelsY * NrPixelsZ * sizeof(*DarkInT));
  AverageDark = calloc(NrPixelsY * NrPixelsZ, sizeof(*AverageDark));
  ImageIn = malloc(NrPixelsY * NrPixelsZ * sizeof(*ImageIn));
  ImageInT = malloc(NrPixelsY * NrPixelsZ * sizeof(*ImageInT));
  Image = malloc(NrPixelsY * NrPixelsZ * sizeof(*Image));
  size_t pxSize;
  if (dType == 1) { // Uint16
    pxSize = sizeof(uint16_t);
  } else if (dType == 2) { // Double
    pxSize = sizeof(double);
  } else if (dType == 3) { // Float
    pxSize = sizeof(float);
  } else if (dType == 4) { // Uint32
    pxSize = sizeof(uint32_t);
  } else if (dType == 5) { // Int32
    pxSize = sizeof(int32_t);
  } else if (dType == 6) { // Tiff Uint32
    pxSize = sizeof(uint32_t);
    HeadSize = 0;
  } else if (dType == 7) { // Tiff Uint8
    pxSize = sizeof(uint8_t);
    HeadSize = 0;
  } else if (dType == 8) { // HDF Unit16
    pxSize = sizeof(uint16_t);
    HeadSize = 0;
  } else if (dType == 9) { // Tiff Uint16
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
  if (argc > 3 || dType == 8) {
    int loadDark = 0;
    if (argc > 3) {
      darkFN = argv[3];
      loadDark = 1;
    } else if (dType == 8) {
      // Fallback: Check if dark dataset exists in data file
      darkFN = argv[2];
      hsize_t dims[3];
      // Check existence by trying to get dims
      if (GetHDF5Dimensions(darkFN, darkDataset, dims) == FR_SUCCESS) {
        printf(
            "No separate dark file provided. Found dark dataset '%s' in data "
            "file '%s'. Using it.\n",
            darkDataset, darkFN);
        loadDark = 1;
      } else {
        printf("No separate dark file provided and dark dataset '%s' not found "
               "in data file '%s'. Processing without dark subtraction.\n",
               darkDataset, darkFN);
        loadDark = 0;
      }
    }

    if (loadDark) {
      if (dType == 8) {
        hsize_t dims[3];
        GetHDF5Dimensions(darkFN, darkDataset, dims);
        nFrames = dims[0] - skipFrame;
        printf("Reading dark file: %s (HDF5), nFrames: %d (dims: %llu, skip: "
               "%d)\n",
               darkFN, nFrames, (unsigned long long)dims[0], skipFrame);
      } else {
        fd = fopen(darkFN, "rb");
        fseek(fd, 0L, SEEK_END);
        sz = ftell(fd);
        rewind(fd);
        nFrames = sz / (SizeFile);
        printf("Reading dark file:      %s, nFrames: %d, skipping first %d "
               "bytes.\n",
               darkFN, nFrames, Skip);
        fseek(fd, Skip, SEEK_SET);
      }

      for (i = 0; i < nFrames; i++) {
        if (dType == 8) {
          rc = ReadHDF5Frame(darkFN, darkDataset, NrPixelsY * NrPixelsZ,
                             DarkInT, i + skipFrame);
        } else if (dType == 6 || dType == 7 || dType == 9) {
          rc = ReadTiffFrame(darkFN, dType, NrPixelsY * NrPixelsZ, DarkInT, i);
        } else {
          rc = ReadBinaryFrame(fd, dType, NrPixelsY * NrPixelsZ, DarkInT);
        }

        DoImageTransformations(NrTransOpt, TransOpt, DarkInT, DarkIn, NrPixelsY,
                               NrPixelsZ);
        if (makeMap == 1) {
          mapMaskSize = NrPixelsY;
          mapMaskSize *= NrPixelsZ;
          mapMaskSize /= 32;
          mapMaskSize++;
          mapMask = calloc(mapMaskSize, sizeof(*mapMask));
          for (j = 0; j < NrPixelsY * NrPixelsZ; j++) {
            if (DarkIn[j] == (pixelvalue)GapIntensity ||
                DarkIn[j] == (pixelvalue)BadPxIntensity) {
              SetBit(mapMask, j);
              nrdone++;
            }
          }
          printf("Nr mask pixels: %d\n", nrdone);
          makeMap = 0;
        }
        for (j = 0; j < NrPixelsY * NrPixelsZ; j++)
          AverageDark[j] += (double)DarkIn[j] / nFrames;
      }
    }
  }

  // HDF5 Data Setup
  hsize_t dims[3];
  if (dType == 8) {
    // Read dims
    GetHDF5Dimensions(argv[2], dataDataset, dims);
    printf("HDF5 Dimensions: %llu x %llu x %llu\n", (unsigned long long)dims[0],
           (unsigned long long)dims[1], (unsigned long long)dims[2]);
    nFrames = dims[0] - skipFrame;
  }

  if (makeMap == 2) {
    mapMaskSize = NrPixelsY;
    mapMaskSize *= NrPixelsZ;
    mapMaskSize /= 32;
    mapMaskSize++;
    mapMask = calloc(mapMaskSize, sizeof(*mapMask));
    double *mapper;
    mapper = calloc(NrPixelsY * NrPixelsZ, sizeof(*mapper));
    double *mapperOut;
    mapperOut = calloc(NrPixelsY * NrPixelsZ, sizeof(*mapperOut));
    ReadTiffFrame(GapFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    DoImageTransformations(NrTransOpt, TransOpt, mapper, mapperOut, NrPixelsY,
                           NrPixelsZ);
    for (i = 0; i < NrPixelsY * NrPixelsZ; i++) {
      if (mapperOut[i] != 0) {
        SetBit(mapMask, i);
        mapperOut[i] = 0;
        nrdone++;
      }
    }
    ReadTiffFrame(BadPxFN, 7, NrPixelsY * NrPixelsZ, mapper, 0);
    DoImageTransformations(NrTransOpt, TransOpt, mapper, mapperOut, NrPixelsY,
                           NrPixelsZ);
    for (i = 0; i < NrPixelsY * NrPixelsZ; i++) {
      if (mapperOut[i] != 0) {
        SetBit(mapMask, i);
        mapperOut[i] = 0;
        nrdone++;
      }
    }
    printf("Nr mask pixels: %d\n", nrdone);
  }
  char *imageFN;
  imageFN = argv[2];
  if (dType != 8) {
    fp = fopen(imageFN, "rb");
    fseek(fp, 0L, SEEK_END);
    sz = ftell(fp);
    rewind(fp);
    fseek(fp, Skip, SEEK_SET);
    nFrames = (sz - Skip) / SizeFile;
  }
  printf("Number of eta bins: %d, number of R bins: %d. Number of frames in "
         "the file: %d\n",
         nEtaBins, nRBins, (int)nFrames);
  long long int Pos;
  int nPixels, dataPos;
  struct data ThisVal;
  char outfn[4096];
  char outfn2[4096];
  FILE *out, *out2;
  hid_t file_id;
  herr_t status_f;
  double *PerFrameArr;
  char outFN1d[4096];
  char dmyt[10000];
  FILE *out1d;
  double Intensity, totArea, ThisInt;
  size_t testPos;
  double RMean, EtaMean;
  double Int1d;
  int n1ds;
  size_t bigArrSize = nEtaBins * nRBins;
  double *chunkArr;
  if (chunkFiles > 0) {
    chunkArr = calloc(bigArrSize, sizeof(*chunkArr));
  }
  double *sumMatrix;
  if (sumImages == 1) {
    sumMatrix = calloc(bigArrSize * 5, sizeof(*sumMatrix));
  }
  double *outArr, *outThisArr, *out1dArr;
  char *outext;
  outext = ".csv";
  double *IntArrPerFrame;
  double firstOme, lastOme;
  IntArrPerFrame = calloc(bigArrSize, sizeof(*IntArrPerFrame));
  FILE *out3;
  if (newOutput == 1) {
    char outfnAll[4096];
    char fn3[4096];
    sprintf(fn3, "%s", imageFN);
    char *bname3;
    bname3 = basename(fn3);
    sprintf(outfnAll, "%s/%s_integrated.bin", outputFolder, bname3, outext);
    printf("%s\n", outfnAll);
    out3 = fopen(outfnAll, "wb");
  }
  if (haveOmegas == 1) {
    omeArr = malloc(nFrames * sizeof(*omeArr));
    for (i = 0; i < nFrames; i++) {
      omeArr[i] = omeStart + i * omeStep;
    }
  }
  for (i = 0; i < nFrames; i++) {
    if (chunkFiles > 0) {
      if ((i % chunkFiles) == 0) {
        memset(chunkArr, 0, bigArrSize * sizeof(*chunkArr));
        firstOme = omeArr[i];
      }
    }
    printf("Processing frame number: %d of %d of file %s.\n", i + 1, nFrames,
           imageFN);
    if (dType == 8) {
      rc = ReadHDF5Frame(imageFN, "exchange/data", NrPixelsY * NrPixelsZ,
                         ImageInT, i + skipFrame);
    } else if (dType == 6 || dType == 7 || dType == 9) {
      rc = ReadTiffFrame(imageFN, dType, NrPixelsY * NrPixelsZ, ImageInT, i);
    } else {
      rc = ReadBinaryFrame(fp, dType, NrPixelsY * NrPixelsZ, ImageInT);
    }
    DoImageTransformations(NrTransOpt, TransOpt, ImageInT, ImageIn, NrPixelsY,
                           NrPixelsZ);
    for (j = 0; j < NrPixelsY * NrPixelsZ; j++) {
      Image[j] = (double)ImageIn[j] - AverageDark[j];
    }
    if (newOutput == 0) {
      if (separateFolder == 0) {
        sprintf(outfn, "%s_integrated_framenr_%d%s", imageFN, i, outext);
        sprintf(outFN1d, "%s_integrated_framenr_%d.1d%s", imageFN, i, outext);
      } else {
        char fn2[4096];
        sprintf(fn2, "%s", imageFN);
        char *bname;
        bname = basename(fn2);
        sprintf(outfn, "%s/%s_integrated_framenr_%d%s", outputFolder, bname, i,
                outext);
        sprintf(outFN1d, "%s/%s_integrated_framenr_%d.1d%s", outputFolder,
                bname, i, outext);
      }
      out = fopen(outfn, "w");
      out1d = fopen(outFN1d, "w");
      fprintf(
          out1d,
          "%%nRBins:\t%d\n%%Radius(px)\t2Theta(degrees)\tIntensity(counts)\n",
          nRBins);
      fprintf(out,
              "%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)"
              "\tEta(degrees)\tIntensity(counts)\tBinArea\n",
              nEtaBins, nRBins);
    }
    if (i == 0 && newOutput == 1) {
      if (separateFolder == 0)
        sprintf(outfn2, "%s.REtaAreaMap.csv", imageFN);
      else {
        char fn2[4096];
        sprintf(fn2, "%s", imageFN);
        char *bnname;
        bnname = basename(fn2);
        sprintf(outfn2, "%s/%s.REtaAreaMap.csv", outputFolder, bnname);
      }
      out2 = fopen(outfn2, "w");
      fprintf(out2,
              "%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)"
              "\tEta(degrees)\tBinArea\n",
              nEtaBins, nRBins);
    } else if (i == 0 && newOutput == 2) {
      if (separateFolder == 0)
        sprintf(outfn2, "%s.caked.hdf", imageFN);
      else {
        char fn2[4096];
        sprintf(fn2, "%s", imageFN);
        char *bnname;
        bnname = basename(fn2);
        sprintf(outfn2, "%s/%s.caked.hdf", outputFolder, bnname);
      }
      file_id = H5Fcreate(outfn2, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (individualSave == 1)
        H5Gcreate(file_id, "/IntegrationResult", H5P_DEFAULT, H5P_DEFAULT,
                  H5P_DEFAULT);
      if (chunkFiles > 0) {
        char gName[2048];
        sprintf(gName, "/OmegaSumFrame", chunkFiles);
        H5Gcreate(file_id, gName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }
      PerFrameArr = malloc(bigArrSize * 4 * sizeof(*PerFrameArr));
    }
    memset(IntArrPerFrame, 0, bigArrSize * sizeof(double));
    for (j = 0; j < nRBins; j++) {
      RMean = (RBinsLow[j] + RBinsHigh[j]) / 2;
      Int1d = 0;
      n1ds = 0;
      for (k = 0; k < nEtaBins; k++) {
        Pos = j * nEtaBins + k;
        nPixels = nPxList[2 * Pos + 0];
        dataPos = nPxList[2 * Pos + 1];
        Intensity = 0;
        totArea = 0;
        for (l = 0; l < nPixels; l++) {
          ThisVal = pxList[dataPos + l];
          testPos = ThisVal.z;
          testPos *= NrPixelsY;
          testPos += ThisVal.y;
          if (mapMaskSize != 0) {
            if (TestBit(mapMask, testPos)) {
              continue;
            }
          }
          ThisInt = Image[testPos]; // The data is arranged as y(fast) and then
                                    // z(slow)
          Intensity += ThisInt * ThisVal.frac;
          totArea += ThisVal.frac;
        }
        if (Intensity != 0) {
          if (Normalize == 1) {
            Intensity /= totArea;
          }
        }
        EtaMean = (EtaBinsLow[k] + EtaBinsHigh[k]) / 2;
        Int1d += Intensity;
        n1ds++;
        if (newOutput == 0) {
          fprintf(out, "%lf\t%lf\t%lf\t%lf\t%lf\n", RMean,
                  atand(RMean * px / Lsd), EtaMean, Intensity, totArea);
        } else {
          if (newOutput == 1) {
            if (i == 0) {
              fprintf(out2, "%lf\t%lf\t%lf\t%lf\n", RMean,
                      atand(RMean * px / Lsd), EtaMean, totArea);
            }
          } else if (newOutput == 2) {
            if (i == 0) {
              PerFrameArr[0 * bigArrSize + (j * nEtaBins + k)] = RMean;
              PerFrameArr[1 * bigArrSize + (j * nEtaBins + k)] =
                  atand(RMean * px / Lsd);
              PerFrameArr[2 * bigArrSize + (j * nEtaBins + k)] = EtaMean;
              PerFrameArr[3 * bigArrSize + (j * nEtaBins + k)] = totArea;
            }
          }
          IntArrPerFrame[j * nEtaBins + k] = Intensity;
        }
        if (sumImages == 1) {
          if (i == 0) {
            sumMatrix[j * nEtaBins * 5 + k * 5 + 0] = RMean;
            sumMatrix[j * nEtaBins * 5 + k * 5 + 1] = atand(RMean * px / Lsd);
            sumMatrix[j * nEtaBins * 5 + k * 5 + 2] = EtaMean;
            sumMatrix[j * nEtaBins * 5 + k * 5 + 4] = totArea;
          }
          sumMatrix[j * nEtaBins * 5 + k * 5 + 3] += Intensity;
        }
      }
      Int1d /= n1ds;
      if (newOutput == 0)
        fprintf(out1d, "%lf\t%lf\t%lf\n", RMean, atand(RMean * px / Lsd),
                Int1d);
    }
    if (newOutput == 2 && i == 0) {
      hsize_t dims[3] = {4, nRBins, nEtaBins};
      status_f =
          H5LTmake_dataset_double(file_id, "/REtaMap", 3, dims, PerFrameArr);
      H5LTset_attribute_int(file_id, "/REtaMap", "nEtaBins", &nEtaBins, 1);
      H5LTset_attribute_int(file_id, "/REtaMap", "nRBins", &nRBins, 1);
      H5LTset_attribute_string(file_id, "/REtaMap", "Header",
                               "Radius,2Theta,Eta,BinArea");
      H5LTset_attribute_string(file_id, "/REtaMap", "Units",
                               "Pixels,Degrees,Degrees,Pixels");
    }
    if (newOutput > 0) {
      if (newOutput == 1)
        fwrite(IntArrPerFrame, bigArrSize * sizeof(*IntArrPerFrame), 1, out3);
      else if (newOutput == 2) {
        hsize_t dim[2] = {nRBins, nEtaBins};
        char dsetName[1024];
        if (individualSave == 1) {
          sprintf(dsetName, "/IntegrationResult/FrameNr_%d", i);
          H5LTmake_dataset_double(file_id, dsetName, 2, dim, IntArrPerFrame);
          H5LTset_attribute_double(file_id, dsetName, "omega", &omeArr[i], 1);
          H5LTset_attribute_string(file_id, dsetName, "Header", "Radius,Eta");
          H5LTset_attribute_string(file_id, dsetName, "Units",
                                   "Pixels,Degrees");
        }
        if (chunkFiles > 0)
          for (p = 0; p < bigArrSize; p++)
            chunkArr[p] += IntArrPerFrame[p];
      }
      if (i == 0 && newOutput == 1) {
        fclose(out2);
      }
    } else {
      fclose(out);
      fclose(out1d);
    }
    if (chunkFiles > 0 && newOutput == 2) {
      if (((i + 1) % chunkFiles) == 0 || i == (nFrames - 1)) {
        hsize_t dim_chunk[2] = {nRBins, nEtaBins};
        char chunkSetName[1024];
        sprintf(chunkSetName, "/OmegaSumFrame/LastFrameNumber_%d", i);
        H5LTmake_dataset_double(file_id, chunkSetName, 2, dim_chunk, chunkArr);
        H5LTset_attribute_int(file_id, chunkSetName, "LastFrameNumber", &i, 1);
        int nSum = (int)((omeArr[i] - firstOme) / omeStep) + 1;
        H5LTset_attribute_int(file_id, chunkSetName, "Number Of Frames Summed",
                              &nSum, 1);
        H5LTset_attribute_double(file_id, chunkSetName, "FirstOme", &firstOme,
                                 1);
        H5LTset_attribute_double(file_id, chunkSetName, "LastOme", &omeArr[i],
                                 1);
      }
    }
  }
  if (newOutput == 2) {
    if (haveOmegas == 1) {
      hsize_t dimome[1] = {nFrames};
      H5LTmake_dataset_double(file_id, "/Omegas", 1, dimome, omeArr);
      H5LTset_attribute_string(file_id, "/Omegas", "Units", "Degrees");
    }
  }
  if (newOutput == 1)
    fclose(out3);
  if (sumImages == 1) {
    if (newOutput < 2) {
      FILE *sumFile;
      char sumFN[4096];
      if (separateFolder == 0) {
        sprintf(sumFN, "%s_sum%s", imageFN, outext);
      } else {
        char fn2[4096];
        sprintf(fn2, "%s", imageFN);
        char *bname;
        bname = basename(fn2);
        sprintf(sumFN, "%s/%s_sum%s", outputFolder, bname, outext);
      }
      sumFile = fopen(sumFN, "w");
      if (newOutput == 0) {
        fprintf(sumFile,
                "%%nEtaBins:\t%d\tnRBins:\t%d\n%%Radius(px)\t2Theta(degrees)"
                "\tEta(degrees)\tIntensity(counts)\tBinArea\n");
        for (i = 0; i < bigArrSize; i++) {
          for (k = 0; k < 5; k++)
            fprintf(sumFile, "%lf\t", sumMatrix[i * 5 + k]);
          fprintf(sumFile, "\n");
        }
      } else if (newOutput == 1) {
        fprintf(sumFile, "%%Intensity(counts)\n");
        for (i = 0; i < bigArrSize; i++) {
          fprintf(sumFile, "%lf\n", sumMatrix[i * 5 + 3]);
        }
      }
    } else if (newOutput == 2) {
      double *sumArr;
      sumArr = malloc(bigArrSize * sizeof(*sumArr));
      for (i = 0; i < bigArrSize; i++) {
        sumArr[i] = sumMatrix[i * 5 + 3];
      }
      hsize_t dimsum[2] = {nRBins, nEtaBins};
      H5LTmake_dataset_double(file_id, "/SumFrames", 2, dimsum, sumArr);
      H5LTset_attribute_string(file_id, "/SumFrames", "Header", "Radius,Eta");
      H5LTset_attribute_string(file_id, "/SumFrames", "Units",
                               "Pixels,Degrees");
      H5LTset_attribute_int(file_id, "/SumFrames", "nFrames", &nFrames, 1);
      free(sumArr);
    }
  }
  if (newOutput == 2) {
    H5Gcreate(file_id, "InstrumentParameters", H5P_DEFAULT, H5P_DEFAULT,
              H5P_DEFAULT);
    hsize_t dimval[1] = {1};
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/Polariz", 1, dimval,
                            &Polariz);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/Lam", 1, dimval,
                            &Lam);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/SH_L", 1, dimval,
                            &SHpL);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/U", 1, dimval, &U);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/V", 1, dimval, &V);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/W", 1, dimval, &W);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/X", 1, dimval, &X);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/Y", 1, dimval, &Y);
    H5LTmake_dataset_double(file_id, "/InstrumentParameters/Z", 1, dimval, &Z);
    status_f = H5Fclose(file_id);
  }
  end0 = clock();
  diftotal = ((double)(end0 - start0)) / CLOCKS_PER_SEC;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  return 0;
}
