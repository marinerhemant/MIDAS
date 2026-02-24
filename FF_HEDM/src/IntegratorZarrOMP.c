//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Integrator.c
//
// Hemant Sharma
// Dt: 2017/07/26
//

#include "ZarrReader.h"
#include <assert.h>
#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
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
#include <tiffio.h>
#include <time.h>
#include <unistd.h>
#include <zip.h>

typedef double pixelvalue;

#define SetBit(A, k) (A[(k >> 5)] |= (1 << (k & 31)))
#define TestBit(A, k) (A[(k >> 5)] & (1 << (k & 31)))
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

int ReadBins(char *resultFolder) {
  int fd;
  struct stat s;
  int status;
  size_t size;
  char file_name[4096];
  sprintf(file_name, "%s/Map.bin", resultFolder);
  int rc;
  fd = open(file_name, O_RDONLY);
  check(fd < 0, "open %s failed: %s", file_name, strerror(errno));
  status = fstat(fd, &s);
  check(status < 0, "stat %s failed: %s", file_name, strerror(errno));
  size = s.st_size;
  int sizelen = 2 * (int)sizeof(int) + (int)sizeof(double);
  printf("Map size in bytes: %lld, each element size: %d, total elements: "
         "%lld. \n",
         (long long int)size, sizelen, (long long int)(size / sizelen));
  pxList = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(pxList == MAP_FAILED, "mmap %s failed: %s", file_name, strerror(errno));

  int fd2;
  struct stat s2;
  int status2;
  char file_name2[4096];
  sprintf(file_name2, "%s/nMap.bin", resultFolder);
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
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

int fileReader(FILE *f, char fn[], int dType, int NrPixels, double *returnArr) {
  int i;
  if (dType == 1) { // Binary with uint16
    uint16_t *readData;
    readData = calloc(NrPixels, sizeof(*readData));
    fread(readData, NrPixels * sizeof(*readData), 1, f);
    for (i = 0; i < NrPixels; i++) {
      returnArr[i] = (double)readData[i];
    }
    free(readData);
    return 0;
  } else if (dType == 2) { // Binary with double
    double *readData;
    readData = calloc(NrPixels, sizeof(*readData));
    fread(readData, NrPixels * sizeof(*readData), 1, f);
    for (i = 0; i < NrPixels; i++) {
      returnArr[i] = (double)readData[i];
    }
    free(readData);
    return 0;
  } else if (dType == 3) { // Binary with float
    float *readData;
    readData = calloc(NrPixels, sizeof(*readData));
    fread(readData, NrPixels * sizeof(*readData), 1, f);
    for (i = 0; i < NrPixels; i++) {
      returnArr[i] = (double)readData[i];
    }
    free(readData);
    return 0;
  } else if (dType == 4) { // Binary with uint32
    uint32_t *readData;
    readData = calloc(NrPixels, sizeof(*readData));
    fread(readData, NrPixels * sizeof(*readData), 1, f);
    for (i = 0; i < NrPixels; i++) {
      returnArr[i] = (double)readData[i];
    }
    free(readData);
    return 0;
  } else if (dType == 5) { // Binary with int32
    int32_t *readData;
    readData = calloc(NrPixels, sizeof(*readData));
    fread(readData, NrPixels * sizeof(*readData), 1, f);
    for (i = 0; i < NrPixels; i++) {
      returnArr[i] = (double)readData[i];
    }
    free(readData);
    return 0;
  } else if (dType == 6) { // TIFF with uint32 format
    TIFFErrorHandler oldhandler;
    oldhandler = TIFFSetWarningHandler(NULL);
    printf("%s\n", fn);
    TIFF *tif = TIFFOpen(fn, "r");
    TIFFSetWarningHandler(oldhandler);
    if (tif) {
      uint32 imagelength;
      tsize_t scanline;
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
      scanline = TIFFScanlineSize(tif);
      tdata_t buf;
      buf = _TIFFmalloc(scanline);
      uint32_t *datar;
      int rnr;
      for (rnr = 0; rnr < imagelength; rnr++) {
        TIFFReadScanline(tif, buf, rnr, 1);
        datar = (uint32_t *)buf;
        for (i = 0; i < scanline / sizeof(uint32_t); i++) {
          returnArr[rnr * (scanline / sizeof(uint32_t)) + i] = (double)datar[i];
        }
      }
      _TIFFfree(buf);
    }
    return 0;
  } else if (dType == 7) { // TIFF with uint8 format
    TIFFErrorHandler oldhandler;
    oldhandler = TIFFSetWarningHandler(NULL);
    printf("%s\n", fn);
    TIFF *tif = TIFFOpen(fn, "r");
    TIFFSetWarningHandler(oldhandler);
    if (tif) {
      uint32 imagelength;
      tsize_t scanline;
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
      scanline = TIFFScanlineSize(tif);
      tdata_t buf;
      buf = _TIFFmalloc(scanline);
      uint8_t *datar;
      int rnr;
      for (rnr = 0; rnr < imagelength; rnr++) {
        TIFFReadScanline(tif, buf, rnr, 1);
        datar = (uint8_t *)buf;
        for (i = 0; i < scanline / sizeof(uint8_t); i++) {
          if (datar[i] == 1) {
            returnArr[rnr * (scanline / sizeof(uint8_t)) + i] = 1;
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

static inline void rawToDouble(const char *raw, double *out, int nPixels,
                               int dType) {
  int i;
  if (dType == 4) { // uint32
    const uint32_t *p = (const uint32_t *)raw;
    for (i = 0; i < nPixels; i++)
      out[i] = (double)p[i];
  } else if (dType == 5) { // int32
    const int32_t *p = (const int32_t *)raw;
    for (i = 0; i < nPixels; i++)
      out[i] = (double)p[i];
  } else if (dType == 3) { // float32
    const float *p = (const float *)raw;
    for (i = 0; i < nPixels; i++)
      out[i] = (double)p[i];
  } else if (dType == 2) { // float64
    const double *p = (const double *)raw;
    for (i = 0; i < nPixels; i++)
      out[i] = p[i];
  } else { // default: uint16 (dType == 1 or 9)
    const uint16_t *p = (const uint16_t *)raw;
    for (i = 0; i < nPixels; i++)
      out[i] = (double)p[i];
  }
}

static inline void MakeSquare(int NrPixels, int NrPixelsY, int NrPixelsZ,
                              pixelvalue *InImage, pixelvalue *OutImage) {
  int i, j, k;
  if (NrPixelsY == NrPixelsZ) {
    memcpy(OutImage, InImage, NrPixels * NrPixels * sizeof(*InImage));
  } else {
    if (NrPixelsY > NrPixelsZ) { // Filling along the slow direction // easy
      memcpy(OutImage, InImage, NrPixelsY * NrPixelsZ * sizeof(*InImage));
    } else {
      for (i = 0; i < NrPixelsZ; i++) {
        memcpy(OutImage + i * NrPixelsZ, InImage + i * NrPixelsY,
               NrPixelsY * sizeof(*InImage));
      }
    }
  }
}

int main(int argc, char **argv) {
  clock_t start, end, start0, end0;
  start0 = clock();
  double diftotal;
  int nCPUs = 4;
  if (argc == 3) {
    nCPUs = atoi(argv[2]);
  } else if (argc != 2) {
    printf("Usage: %s ZarrName.zip [nCPUs]\n", argv[0]);
    return (1);
  }
  omp_set_num_threads(nCPUs);
  printf("Running with %d OpenMP threads.\n", nCPUs);

  double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
  int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
  int nEtaBins, nRBins;
  char aline[4096], dummy[4096], *str;
  int HeadSize = 8192;
  int NrTransOpt = 0;
  long long int GapIntensity = 0, BadPxIntensity = 0;
  int TransOpt[10];
  int makeMap = 0;
  size_t mapMaskSize = 0;
  int *mapMask, skipFrame = 0;
  int dType = 1; // default uint16
  char *GapFN = NULL, *BadPxFN = NULL, outputFolder[4096];
  int sumImages = 0, separateFolder = 1, newOutput = 2;
  int haveOmegas = 0, chunkFiles = 1, individualSave = 1;
  double omeStart, omeStep;
  double Lam = 0.172978, Polariz = 0.99, SHpL = 0.002, U = 1.163, V = -0.126,
         W = 0.063, X = 0.0, Y = 0.0, Z = 0.0;
  char *DataFN = argv[1];
  blosc2_init();
  // Read zarr config
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(DataFN, 0, &errorp);
  if (arch == NULL) {
    fprintf(stderr, "ERROR: Could not open zip archive '%s' (error code: %d)\n",
            DataFN, errorp);
    return 1;
  }
  struct zip_stat *finfo = NULL;
  finfo = calloc(16384, sizeof(int));
  zip_stat_init(finfo);
  int count = 0;
  char *resultFolder = NULL;
  int locImTransOpt, locTemp = -1, locPres = -1, locI = -1, locI0 = -1, nFrames,
                     nDarks, nFloods, bytesPerPx = 2;
  int darkLoc = -1, dataLoc = -1, floodLoc = -1;
  double *Temperature, *Pressure, *I, *I0;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ResultFolder/0") != NULL) {
      ReadZarrString(arch, count, &resultFolder, 4096);
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/GapFile/0") !=
        NULL) {
      ReadZarrString(arch, count, &GapFN, 4096);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BadPxFile/0") != NULL) {
      ReadZarrString(arch, count, &BadPxFN, 4096);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/Wavelength/0") != NULL) {
      ReadZarrChunk(arch, count, &Lam, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/X/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &X, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Y/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &Y, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Z/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &Z, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/U/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &U, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/V/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &V, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/W/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &W, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/SHpL/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &SHpL, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Polariz/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &Polariz, sizeof(double));
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/start/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &omeStart, sizeof(double));
      haveOmegas = 1;
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/step/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &omeStep, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/EtaBinSize/0") != NULL) {
      ReadZarrChunk(arch, count, &EtaBinSize, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/RBinSize/0") != NULL) {
      ReadZarrChunk(arch, count, &RBinSize, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RMax/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &RMax, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/RMin/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &RMin, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/EtaMax/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &EtaMax, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/EtaMin/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &EtaMin, sizeof(double));
    }
    if (strstr(finfo->name, "analysis/process/analysis_parameters/Lsd/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &Lsd, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSizeY/0") != NULL) {
      ReadZarrChunk(arch, count, &px, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSizeZ/0") != NULL) {
      ReadZarrChunk(arch, count, &px, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/PixelSize/0") != NULL) {
      ReadZarrChunk(arch, count, &px, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/OmegaSumFrames/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &chunkFiles, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SaveIndividualFrames/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &individualSave, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SkipFrame/0") != NULL) {
      ReadZarrChunk(arch, count, &skipFrame, sizeof(int));
      printf("SkipFrame: %d\n", skipFrame);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/Normalize/0") != NULL) {
      ReadZarrChunk(arch, count, &Normalize, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SumImages/0") != NULL) {
      ReadZarrChunk(arch, count, &sumImages, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/GapIntensity/0") != NULL) {
      ReadZarrChunk(arch, count, &GapIntensity, sizeof(long long int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BadPxIntensity/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &BadPxIntensity, sizeof(long long int));
    }
    if (strstr(finfo->name, "exchange/data/.zarray") != NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
               &nFrames, &NrPixelsZ, &NrPixelsY);
        printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d\n", nFrames, NrPixelsZ,
               NrPixelsY);
      } else {
        free(s);
        return 1;
      }
      free(s);
    }
    if (strstr(finfo->name, "exchange/dark/.zarray") != NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        if (3 == sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &nDarks, &NrPixelsZ, &NrPixelsY)) {
          printf("nDarks: %d nrPixelsZ: %d nrPixelsY: %d\n", nDarks, NrPixelsZ,
                 NrPixelsY);
        } else {
          free(s);
          return 1;
        }
      } else {
        free(s);
        return 1;
      }
      free(s);
    }
    if (strstr(finfo->name, "exchange/flood/.zarray") != NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        if (3 == sscanf(ptr3,
                        "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                        &nFloods, &NrPixelsZ, &NrPixelsY)) {
          printf("nFloods: %d nrPixelsZ: %d nrPixelsY: %d\n", nFloods,
                 NrPixelsZ, NrPixelsY);
        } else {
          free(s);
          return 1;
        }
      } else {
        free(s);
        return 1;
      }
      free(s);
    }
    if (strstr(finfo->name, "exchange/data/0.0.0") != NULL) {
      dataLoc = count;
    }
    if (strstr(finfo->name, "exchange/dark/0.0.0") != NULL) {
      darkLoc = count;
    }
    if (strstr(finfo->name, "exchange/flood/0.0.0") != NULL) {
      floodLoc = count;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/.zarray") !=
        NULL) {
      char *s = NULL;
      size_t sSize;
      ReadZarrRaw(arch, count, &s, &sSize);
      char *ptr = strstr(s, "shape");
      if (ptr != NULL) {
        char *ptrt = strstr(ptr, "[");
        char *ptr2 = strstr(ptrt, "]");
        int loc = (int)(ptr2 - ptrt);
        char ptr3[2048];
        strncpy(ptr3, ptrt, loc + 1);
        sscanf(ptr3, "%*[^0123456789]%d", &NrTransOpt);
      } else {
        free(s);
        return 1;
      }
      printf("nImTransOpt: %d\n", NrTransOpt);
      free(s);
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/ImTransOpt/0") != NULL) {
      locImTransOpt = count;
    }
    if (strstr(finfo->name,
               "measurement/process/scan_parameters/Temperature/0") != NULL) {
      locTemp = count;
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/Pressure/0") !=
        NULL) {
      locPres = count;
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/I/0") !=
        NULL) {
      locI = count;
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/I0/0") !=
        NULL) {
      locI0 = count;
    }
    if (strstr(finfo->name, "measurement/process/scan_parameters/datatype/0") !=
        NULL) {
      char *typeName = NULL;
      size_t typeSize;
      if (ReadZarrRaw(arch, count, &typeName, &typeSize) >= 0 && typeName) {
        if (strcasecmp(typeName, "uint32") == 0) {
          bytesPerPx = 4;
          dType = 4;
        } else if (strcasecmp(typeName, "int32") == 0) {
          bytesPerPx = 4;
          dType = 5;
        } else if (strcasecmp(typeName, "float32") == 0) {
          bytesPerPx = 4;
          dType = 3;
        } else if (strcasecmp(typeName, "float64") == 0) {
          bytesPerPx = 8;
          dType = 2;
        } else {
          bytesPerPx = 2;
          dType = 1;
        }
        printf("Detected datatype: %s (bytesPerPx=%d)\n", typeName, bytesPerPx);
        free(typeName);
      }
    }
    count++;
  }
  if (chunkFiles == 0)
    chunkFiles = 1;
  int rc = ReadBins(resultFolder);
  int32_t imTransBufSize = NrTransOpt * sizeof(int);
  int *imTransData = (int *)malloc((size_t)imTransBufSize);
  ReadZarrChunk(arch, locImTransOpt, imTransData, imTransBufSize);
  int iter;
  for (iter = 0; iter < NrTransOpt; iter++)
    TransOpt[iter] = imTransData[iter];
  for (iter = 0; iter < NrTransOpt; iter++)
    printf("Transopt: %d\n", TransOpt[iter]);
  free(imTransData);
  if (separateFolder != 0) {
    struct stat st = {0};
    sprintf(outputFolder, "%s", resultFolder);
    if (stat(outputFolder, &st) == -1) {
      printf("Output folder '%s' did not exit. Making now.\n", outputFolder);
      mkdir(outputFolder, 0700);
    }
  }
  nFrames = nFrames - skipFrame; // This ensures we don't over-read.
  dataLoc += skipFrame;
  darkLoc += skipFrame;
  Pressure = (double *)calloc(nFrames, sizeof(*Pressure));
  Temperature = (double *)calloc(nFrames, sizeof(*Temperature));
  I = (double *)calloc(nFrames, sizeof(*I));
  I0 = (double *)calloc(nFrames, sizeof(*I0));
  if (locPres > 0) {
    int32_t bufSize = (nFrames + skipFrame) * sizeof(double);
    double *tmpData = (double *)malloc((size_t)bufSize);
    ReadZarrChunk(arch, locPres, tmpData, bufSize);
    for (iter = skipFrame; iter < nFrames + skipFrame; iter++)
      Pressure[iter - skipFrame] = tmpData[iter];
    free(tmpData);
  }
  if (locTemp > 0) {
    int32_t bufSize = (nFrames + skipFrame) * sizeof(double);
    double *tmpData = (double *)malloc((size_t)bufSize);
    ReadZarrChunk(arch, locTemp, tmpData, bufSize);
    for (iter = skipFrame; iter < nFrames + skipFrame; iter++)
      Temperature[iter - skipFrame] = tmpData[iter];
    free(tmpData);
  }
  if (locI > 0) {
    int32_t bufSize = (nFrames + skipFrame) * sizeof(double);
    double *tmpData = (double *)malloc((size_t)bufSize);
    ReadZarrChunk(arch, locI, tmpData, bufSize);
    for (iter = skipFrame; iter < nFrames + skipFrame; iter++)
      I[iter - skipFrame] = tmpData[iter];
    free(tmpData);
  }
  if (locI0 > 0) {
    int32_t bufSize = (nFrames + skipFrame) * sizeof(double);
    double *tmpData = (double *)malloc((size_t)bufSize);
    ReadZarrChunk(arch, locI0, tmpData, bufSize);
    for (iter = skipFrame; iter < nFrames + skipFrame; iter++)
      I0[iter - skipFrame] = tmpData[iter];
    free(tmpData);
  }
  double *Image;
  int a, b;
  pixelvalue *ImageIn;
  pixelvalue *DarkIn;
  pixelvalue *ImageInT;
  pixelvalue *DarkInT;
  double *AverageDark;
  int nPixels = NrPixelsY * NrPixelsZ;
  DarkIn = malloc(nPixels * sizeof(*DarkIn));
  DarkInT = malloc(nPixels * sizeof(*DarkInT));
  AverageDark = calloc(nPixels, sizeof(*AverageDark));
  ImageIn = malloc(nPixels * sizeof(*ImageIn));
  ImageInT = malloc(nPixels * sizeof(*ImageInT));
  Image = malloc(nPixels * sizeof(*Image));
  // printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d, dataLoc: %d\n", nFrames,
  // NrPixelsZ, NrPixelsY,dataLoc); Now we read the size of data for each file
  // pointer.
  size_t *sizeArr;
  sizeArr = calloc(nFrames * 2, sizeof(*sizeArr)); // Number StartLoc
  size_t cntr = 0;
  if (dataLoc < 0) {
    printf("Error: Missing primary data chunk 0.0.0! Cannot proceed with "
           "integrations.\n");
    zip_close(arch);
    return 1;
  }
  printf("Reading compressed image data.\n");
  for (iter = 0; iter < nFrames; iter++) {
    zip_stat_index(arch, dataLoc + iter, 0, finfo);
    sizeArr[iter * 2 + 0] = finfo->size;
    sizeArr[iter * 2 + 1] = cntr;
    cntr += finfo->size;
  }
  // allocate arr
  char *allData;
  allData = calloc(cntr + 1, sizeof(*allData));
  for (iter = 0; iter < nFrames; iter++) {
    zip_file_t *fLoc = NULL;
    fLoc = zip_fopen_index(arch, dataLoc + iter, 0);
    zip_fread(fLoc, &allData[sizeArr[iter * 2 + 1]], sizeArr[iter * 2 + 0]);
    zip_fclose(fLoc);
  }
  omeStart += skipFrame * omeStep;
  // Dark file reading from here.
  size_t pxSize = sizeof(uint16_t);
  HeadSize = 0;
  int darkIter;
  // Local variables for frame-by-frame image reading (not parameter reads)
  char *arr;
  zip_file_t *fd;
  char *data;
  int32_t dsize;
  dsize = bytesPerPx * NrPixelsZ * NrPixelsY;
  data = (char *)malloc((size_t)dsize);
  int32_t expected_dsize =
      bytesPerPx * NrPixelsZ * NrPixelsY; // Buffer max capacity
  for (darkIter = skipFrame; darkIter < nDarks; darkIter++) {
    int current_darkLoc = darkLoc + (darkIter - skipFrame);
    if (current_darkLoc < 0) {
      printf("Error: Invalid darkLoc. Skipping dark frames.\n");
      break;
    }
    if (zip_stat_index(arch, current_darkLoc, 0, finfo) != 0) {
      printf("Error: Failed to stat dark frame at index %d\n", current_darkLoc);
      break;
    }
    // Go to the right location in the zip file and read frames.
    arr = calloc(finfo->size, sizeof(char));
    fd = zip_fopen_index(arch, current_darkLoc, 0);
    if (!fd) {
      printf("Error: Failed to open dark frame inside zip.\n");
      free(arr);
      break;
    }
    int numRead = zip_fread(fd, arr, finfo->size);
    // print arr size
    printf("Dark frame compressed size: %ld\n", finfo->size);
    printf("Number of bytes read: %d\n", numRead);
    // print
    dsize = expected_dsize; // Reset buffer capacity
    dsize = blosc1_decompress(arr, data, dsize);
    free(arr);
    zip_fclose(fd);

    if (dsize <= 0) {
      printf("Error: Failed to decompress dark frame data! dsize: %d\n", dsize);
      memset(DarkInT, 0, nPixels * sizeof(*DarkInT));
    } else {
      rawToDouble(data, DarkInT, nPixels, dType);
    }
    DoImageTransformations(NrTransOpt, TransOpt, DarkInT, DarkIn, NrPixelsY,
                           NrPixelsZ);
    for (b = 0; b < (NrPixelsY * NrPixelsZ); b++) {
      AverageDark[b] += ((double)DarkIn[b]) / (nDarks - skipFrame);
    }
  }
  // for (b=0;b<NrPixelsY*NrPixelsZ;b++) printf("%lf\n",AverageDark[b]);
  free(data);
  zip_close(arch);

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
  double *omeArr;
  int nrdone = 0;
  FILE *fdd;
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
    fileReader(fdd, GapFN, 7, NrPixelsY * NrPixelsZ, mapper);
    DoImageTransformations(NrTransOpt, TransOpt, mapper, mapperOut, NrPixelsY,
                           NrPixelsZ);
    for (i = 0; i < NrPixelsY * NrPixelsZ; i++) {
      if (mapperOut[i] != 0) {
        SetBit(mapMask, i);
        mapperOut[i] = 0;
        nrdone++;
      }
    }
    fileReader(fdd, BadPxFN, 7, NrPixelsY * NrPixelsZ, mapper);
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
  printf("Number of eta bins: %d, number of R bins: %d. Number of frames in "
         "the file: %d\n",
         nEtaBins, nRBins, (int)nFrames);
  long long int Pos;
  int dataPos;
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
  if (haveOmegas == 1) {
    omeArr = malloc(nFrames * sizeof(*omeArr));
    for (i = 0; i < nFrames; i++) {
      omeArr[i] = omeStart + i * omeStep;
    }
  } else
    omeArr = calloc(nFrames, sizeof(*omeArr));
  char *locData;
  locData = calloc(NrPixelsY * NrPixelsZ * bytesPerPx, sizeof(*locData));
  int32_t dsz = NrPixelsY * NrPixelsZ * bytesPerPx;
  double presThis, tempThis, iThis, i0This;
  double t_integration = 0, t_0;
  for (i = 0; i < nFrames; i++) {
    if (chunkFiles > 0) {
      if ((i % chunkFiles) == 0) {
        memset(chunkArr, 0, bigArrSize * sizeof(*chunkArr));
        firstOme = omeArr[i];
        presThis = 0;
        tempThis = 0;
        iThis = 0;
        i0This = 0;
      }
    }
    printf("Processing frame number: %d of %d of file %s.\n", i + 1, nFrames,
           DataFN);
    presThis += Pressure[i];
    tempThis += Temperature[i];
    iThis += I[i];
    i0This = I0[i];
    dsz = NrPixelsY * NrPixelsZ * bytesPerPx; // Reset buffer capacity
    dsz = blosc1_decompress(&allData[sizeArr[i * 2 + 1]], locData, dsz);
    if (dsz <= 0) {
      printf("Error: Failed to decompress frame data at index %d! dsize: %d\n",
             i, dsz);
      exit(1);
    }
    rawToDouble(locData, ImageInT, nPixels, dType);
    DoImageTransformations(NrTransOpt, TransOpt, ImageInT, ImageIn, NrPixelsY,
                           NrPixelsZ);
    for (j = 0; j < NrPixelsY * NrPixelsZ; j++) {
      Image[j] = (double)ImageIn[j] - AverageDark[j];
    }
    if (i == 0) {
      char fn2[4096];
      sprintf(fn2, "%s", DataFN);
      char *bnname;
      bnname = basename(fn2);
      sprintf(outfn2, "%s/%s.caked.hdf", outputFolder, bnname);
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
    t_0 = omp_get_wtime();
#pragma omp parallel for private(j, k, l, Pos, nPixels, dataPos, Intensity,    \
                                     totArea, ThisVal, testPos, ThisInt,       \
                                     RMean, EtaMean)
    for (j = 0; j < nRBins; j++) {
      RMean = (RBinsLow[j] + RBinsHigh[j]) / 2;
      for (k = 0; k < nEtaBins; k++) {
        Pos = j * nEtaBins + k;
        nPixels = nPxList[2 * Pos + 0];
        dataPos = nPxList[2 * Pos + 1];
        Intensity = 0;
        totArea = 0;
        if (mapMaskSize != 0) {
          for (l = 0; l < nPixels; l++) {
            ThisVal = pxList[dataPos + l];
            testPos = (size_t)ThisVal.z * NrPixelsY + ThisVal.y;
            if (TestBit(mapMask, testPos)) {
              continue;
            }
            Intensity += Image[testPos] * ThisVal.frac;
            totArea += ThisVal.frac;
          }
        } else {
          for (l = 0; l < nPixels; l++) {
            ThisVal = pxList[dataPos + l];
            Intensity +=
                Image[(size_t)ThisVal.z * NrPixelsY + ThisVal.y] * ThisVal.frac;
            totArea += ThisVal.frac;
          }
        }
        if (Intensity != 0) {
          if (Normalize == 1) {
            Intensity /= totArea;
          }
        }
        EtaMean = (EtaBinsLow[k] + EtaBinsHigh[k]) / 2;
        if (i == 0) {
          PerFrameArr[0 * bigArrSize + (j * nEtaBins + k)] = RMean;
          PerFrameArr[1 * bigArrSize + (j * nEtaBins + k)] =
              atand(RMean * px / Lsd);
          PerFrameArr[2 * bigArrSize + (j * nEtaBins + k)] = EtaMean;
          PerFrameArr[3 * bigArrSize + (j * nEtaBins + k)] = totArea;
        }
        IntArrPerFrame[j * nEtaBins + k] = Intensity;
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
    }
    t_integration += (omp_get_wtime() - t_0);
    if (i == 0) {
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
    hsize_t dim[2] = {nRBins, nEtaBins};
    char dsetName[1024];
    if (individualSave == 1) {
      sprintf(dsetName, "/IntegrationResult/FrameNr_%d", i);
      H5LTmake_dataset_double(file_id, dsetName, 2, dim, IntArrPerFrame);
      H5LTset_attribute_double(file_id, dsetName, "omega", &omeArr[i], 1);
      H5LTset_attribute_string(file_id, dsetName, "Header", "Radius,Eta");
      H5LTset_attribute_string(file_id, dsetName, "Units", "Pixels,Degrees");
    }
    if (chunkFiles > 0)
      for (p = 0; p < bigArrSize; p++)
        chunkArr[p] += IntArrPerFrame[p];
    if (chunkFiles > 0) {
      if (((i + 1) % chunkFiles) == 0 || i == (nFrames - 1)) {
        hsize_t dim_chunk[2] = {nRBins, nEtaBins};
        char chunkSetName[1024];
        sprintf(chunkSetName, "/OmegaSumFrame/LastFrameNumber_%d", i);
        H5LTmake_dataset_double(file_id, chunkSetName, 2, dim_chunk, chunkArr);
        H5LTset_attribute_int(file_id, chunkSetName, "LastFrameNumber", &i, 1);
        int nSum = (int)((omeArr[i] - firstOme) / omeStep) + 1;
        presThis /= nSum;
        tempThis /= nSum;
        iThis /= nSum;
        i0This /= nSum;
        H5LTset_attribute_int(file_id, chunkSetName, "Number Of Frames Summed",
                              &nSum, 1);
        H5LTset_attribute_double(file_id, chunkSetName, "FirstOme", &firstOme,
                                 1);
        H5LTset_attribute_double(file_id, chunkSetName, "LastOme", &omeArr[i],
                                 1);
        H5LTset_attribute_double(file_id, chunkSetName, "Temperature",
                                 &tempThis, 1);
        H5LTset_attribute_double(file_id, chunkSetName, "Pressure", &presThis,
                                 1);
        H5LTset_attribute_double(file_id, chunkSetName, "I", &iThis, 1);
        H5LTset_attribute_double(file_id, chunkSetName, "I0", &i0This, 1);
      }
    }
  }
  printf("Time for integration: %f seconds.\n", t_integration);
  if (haveOmegas == 1) {
    hsize_t dimome[1] = {nFrames};
    H5LTmake_dataset_double(file_id, "/Omegas", 1, dimome, omeArr);
    H5LTset_attribute_string(file_id, "/Omegas", "Units", "Degrees");
  }
  if (sumImages == 1) {
    double *sumArr;
    sumArr = malloc(bigArrSize * sizeof(*sumArr));
    for (i = 0; i < bigArrSize; i++) {
      sumArr[i] = sumMatrix[i * 5 + 3];
    }
    hsize_t dimsum[2] = {nRBins, nEtaBins};
    H5LTmake_dataset_double(file_id, "/SumFrames", 2, dimsum, sumArr);
    H5LTset_attribute_string(file_id, "/SumFrames", "Header", "Radius,Eta");
    H5LTset_attribute_string(file_id, "/SumFrames", "Units", "Pixels,Degrees");
    H5LTset_attribute_int(file_id, "/SumFrames", "nFrames", &nFrames, 1);
    free(sumArr);
  }
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
  H5LTmake_dataset_double(file_id, "/InstrumentParameters/Distance", 1, dimval,
                          &Lsd);
  status_f = H5Fclose(file_id);
  end0 = clock();
  diftotal = ((double)(end0 - start0)) / CLOCKS_PER_SEC;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  return 0;
}
