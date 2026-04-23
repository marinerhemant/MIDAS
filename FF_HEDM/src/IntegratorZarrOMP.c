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
// CalibPeakFit.h removed — peak fitting unified in PeakFit.h
#include "MapHeader.h"
#include "PeakFit.h"
#include "PeakFitIO.h"
#include "ZarrReader.h"
#include "midas_version.h"
#include "ImageUtils.h"
#include <assert.h>
#include <blosc2.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <libgen.h>
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
#define rad2deg (180.0 / M_PI)

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
  float y;
  float z;
  double frac;       /* corrected weight: Area / C  (C = solid-angle × polarization) */
  float deltaR;      /* R_sub_centroid - R_bin_center (gradient correction) */
  float areaWeight;  /* uncorrected geometric area weight */
};

struct data *pxList;
int *nPxList;
int *binMaskFlag; // per-bin contamination flags from maskMap.bin (NULL if
                  // absent)

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

  /* Detect header */
  struct MapHeader map_hdr;
  int has_header = map_header_read_fd(fd, &map_hdr);
  size_t data_offset = 0;
  if (has_header) {
    data_offset = MAP_HEADER_SIZE;
    map_header_print("Map.bin", &map_hdr);
  } else {
    printf("WARNING: Map.bin has no parameter header (legacy format).\n");
    printf("  Consider regenerating with latest DetectorMapper.\n");
  }
  size_t data_size = size - data_offset;

  int sizelen = 2 * (int)sizeof(int) + (int)sizeof(double);
  printf("Map size in bytes: %lld, each element size: %d, total elements: "
         "%lld. \n",
         (long long int)data_size, sizelen,
         (long long int)(data_size / sizelen));
  /* mmap entire file, then adjust pointer past header */
  void *map_base = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
  check(map_base == MAP_FAILED, "mmap %s failed: %s", file_name,
        strerror(errno));
  pxList = (struct data *)((char *)map_base + data_offset);

  int fd2;
  struct stat s2;
  int status2;
  char file_name2[4096];
  sprintf(file_name2, "%s/nMap.bin", resultFolder);
  fd2 = open(file_name2, O_RDONLY);
  check(fd2 < 0, "open %s failed: %s", file_name2, strerror(errno));
  status2 = fstat(fd2, &s2);
  check(status2 < 0, "stat %s failed: %s", file_name2, strerror(errno));
  size_t size2_total = s2.st_size;

  /* Detect header in nMap.bin too */
  struct MapHeader nmap_hdr;
  int has_header2 = map_header_read_fd(fd2, &nmap_hdr);
  size_t data_offset2 = has_header2 ? MAP_HEADER_SIZE : 0;
  size_t data_size2 = size2_total - data_offset2;

  void *nmap_base = mmap(0, size2_total, PROT_READ, MAP_SHARED, fd2, 0);
  check(nmap_base == MAP_FAILED, "mmap %s failed: %s", file_name2,
        strerror(errno));
  nPxList = (int *)((char *)nmap_base + data_offset2);
  printf("nMap size in bytes: %lld, each element size: %d, total elements: "
         "%lld. \n",
         (long long int)data_size2, 2 * (int)sizeof(int),
         2 * (long long int)(data_size2 / sizeof(int)));

  // Optionally read maskMap.bin (bin contamination flags)
  binMaskFlag = NULL;
  char mask_map_fn[4096];
  sprintf(mask_map_fn, "%s/maskMap.bin", resultFolder);
  int fd3 = open(mask_map_fn, O_RDONLY);
  if (fd3 >= 0) {
    struct stat s3;
    if (fstat(fd3, &s3) == 0 && s3.st_size > 0) {
      size_t size3 = s3.st_size;
      struct MapHeader mask_hdr;
      int has_hdr3 = map_header_read_fd(fd3, &mask_hdr);
      size_t off3 = has_hdr3 ? MAP_HEADER_SIZE : 0;
      void *mask_base = mmap(0, size3, PROT_READ, MAP_SHARED, fd3, 0);
      if (mask_base != MAP_FAILED) {
        binMaskFlag = (int *)((char *)mask_base + off3);
        int nFlagged = 0;
        long long int nBins = (long long int)(size3 - off3) / sizeof(int);
        for (long long int bi = 0; bi < nBins; bi++)
          if (binMaskFlag[bi])
            nFlagged++;
        printf("maskMap.bin: %lld bins, %d flagged as contaminated\n", nBins,
               nFlagged);
      }
    } else {
      close(fd3);
    }
  }

  fflush(stdout);
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

// REtaMapper, DoImageTransformations, MakeSquare removed — now in shared library
// (ImageUtils.h: midas_reta_mapper, midas_image_transform, midas_make_square)

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

// MakeSquare removed — now midas_make_square() in ImageUtils.h

// ─────────────────────────────────────────────────────────────────
// readParamFile: parse MIDAS parameter file into integrator vars
// ─────────────────────────────────────────────────────────────────
struct IntegratorParams {
  double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize;
  double Lsd, px, Lam;
  double QBinSize, QMin, QMax;
  double Polariz, SHpL, U, V, W, X, Y, Z;
  double omeStart, omeStep;
  double BC_y, BC_z; // beam center for bilinear interpolation
  int NrPixelsY, NrPixelsZ, Normalize, skipFrame;
  int GradientCorrection; /* 0=off, 1=apply radial gradient correction */
  int NrTransOpt;
  int TransOpt[10];
  int sumImages, chunkFiles, individualSave;
  int haveOmegas;
  int doPeakFit, nPeakLocations, fitROIPadding;
  int autoDetectPeaks; // 0 = disabled, >0 = max peaks to auto-detect
  int snipIterations;  // SNIP background iterations (default 50)
  double doubletSeparation; // px: adjacent peaks within this are fit jointly (0=off)
  double peakLocations[MAX_PEAK_LOCATIONS_PF];
  char resultFolder[4096];
  char GapFN[4096];
  char BadPxFN[4096];
  char MaskFN[4096];
  int hasGapFN, hasBadPxFN, hasMaskFN;
};

static void initIntegratorParams(struct IntegratorParams *p) {
  memset(p, 0, sizeof(*p));
  p->NrPixelsY = 2048;
  p->NrPixelsZ = 2048;
  p->Normalize = 1;
  p->GradientCorrection = 0;
  p->Lam = 0.172978;
  p->Polariz = 0.99;
  p->SHpL = 0.002;
  p->U = 1.163;
  p->V = -0.126;
  p->W = 0.063;
  p->chunkFiles = 1;
  p->individualSave = 1;
  p->fitROIPadding = 20;
  p->autoDetectPeaks = 0;
  p->snipIterations = 50;
  p->doubletSeparation = 0.0;
}

static int readParamFile(const char *filename, struct IntegratorParams *p) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: Cannot open param file: %s\n", filename);
    return -1;
  }
  char line[4096], key[256], rest[3840];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
      continue;
    if (sscanf(line, "%255s %[^\n]", key, rest) < 2)
      continue;
    /* Geometry */
    if (strcmp(key, "RMin") == 0)
      p->RMin = atof(rest);
    else if (strcmp(key, "RMax") == 0)
      p->RMax = atof(rest);
    else if (strcmp(key, "RBinSize") == 0)
      p->RBinSize = atof(rest);
    else if (strcmp(key, "BC") == 0)
      sscanf(rest, "%lf %lf", &p->BC_y, &p->BC_z);
    else if (strcmp(key, "EtaMin") == 0 || strcmp(key, "MinEta") == 0)
      p->EtaMin = atof(rest);
    else if (strcmp(key, "EtaMax") == 0)
      p->EtaMax = atof(rest);
    else if (strcmp(key, "EtaBinSize") == 0)
      p->EtaBinSize = atof(rest);
    else if (strcmp(key, "Lsd") == 0 || strcmp(key, "DetDist") == 0)
      p->Lsd = atof(rest);
    else if (strcmp(key, "px") == 0 || strcmp(key, "PixelSize") == 0)
      p->px = atof(rest);
    else if (strcmp(key, "YPixelSize") == 0)
      p->px = atof(rest);
    else if (strcmp(key, "Wavelength") == 0)
      p->Lam = atof(rest);
    else if (strcmp(key, "QBinSize") == 0)
      p->QBinSize = atof(rest);
    else if (strcmp(key, "QMin") == 0)
      p->QMin = atof(rest);
    else if (strcmp(key, "QMax") == 0)
      p->QMax = atof(rest);
    else if (strcmp(key, "NrPixels") == 0) {
      p->NrPixelsY = atoi(rest);
      p->NrPixelsZ = atoi(rest);
    } else if (strcmp(key, "NrPixelsY") == 0)
      p->NrPixelsY = atoi(rest);
    else if (strcmp(key, "NrPixelsZ") == 0)
      p->NrPixelsZ = atoi(rest);
    /* Tilts / distortion — not needed for integration, but keep for future */
    /* Output */
    else if (strcmp(key, "Folder") == 0 || strcmp(key, "ResultFolder") == 0)
      sscanf(rest, "%4095s", p->resultFolder);
    /* Transforms */
    else if (strcmp(key, "ImTransOpt") == 0) {
      /* Space-separated ints on one line */
      char *tok = strtok(rest, " \t");
      p->NrTransOpt = 0;
      while (tok && p->NrTransOpt < 10) {
        p->TransOpt[p->NrTransOpt++] = atoi(tok);
        tok = strtok(NULL, " \t");
      }
    } else if (strcmp(key, "Normalize") == 0)
      p->Normalize = atoi(rest);
    else if (strcmp(key, "GradientCorrection") == 0)
      p->GradientCorrection = atoi(rest);
    else if (strcmp(key, "SkipFrame") == 0)
      p->skipFrame = atoi(rest);
    else if (strcmp(key, "OmegaStart") == 0 ||
             strcmp(key, "OmegaFirstFile") == 0) {
      p->omeStart = atof(rest);
      p->haveOmegas = 1;
    } else if (strcmp(key, "OmegaStep") == 0)
      p->omeStep = atof(rest);
    else if (strcmp(key, "OmegaSumFrames") == 0)
      p->chunkFiles = atoi(rest);
    else if (strcmp(key, "SumImages") == 0)
      p->sumImages = atoi(rest);
    else if (strcmp(key, "SaveIndividualFrames") == 0)
      p->individualSave = atoi(rest);
    /* Masking */
    else if (strcmp(key, "GapFile") == 0) {
      sscanf(rest, "%4095s", p->GapFN);
      p->hasGapFN = 1;
    } else if (strcmp(key, "BadPxFile") == 0) {
      sscanf(rest, "%4095s", p->BadPxFN);
      p->hasBadPxFN = 1;
    } else if (strcmp(key, "MaskFile") == 0 || strcmp(key, "MaskFN") == 0) {
      sscanf(rest, "%4095s", p->MaskFN);
      p->hasMaskFN = 1;
    }
    /* Instrument */
    else if (strcmp(key, "Polariz") == 0)
      p->Polariz = atof(rest);
    else if (strcmp(key, "SHpL") == 0)
      p->SHpL = atof(rest);
    else if (strcmp(key, "U") == 0)
      p->U = atof(rest);
    else if (strcmp(key, "V") == 0)
      p->V = atof(rest);
    else if (strcmp(key, "W") == 0)
      p->W = atof(rest);
    else if (strcmp(key, "X") == 0)
      p->X = atof(rest);
    else if (strcmp(key, "Y") == 0)
      p->Y = atof(rest);
    else if (strcmp(key, "Z") == 0)
      p->Z = atof(rest);
    /* Peak fitting */
    else if (strcmp(key, "DoPeakFit") == 0)
      p->doPeakFit = atoi(rest);
    else if (strcmp(key, "FitROIPadding") == 0)
      p->fitROIPadding = atoi(rest);
    else if (strcmp(key, "AutoDetectPeaks") == 0) {
      p->autoDetectPeaks = atoi(rest);
      p->doPeakFit = 1;
    } else if (strcmp(key, "SNIPIterations") == 0) {
      p->snipIterations = atoi(rest);
    } else if (strcmp(key, "DoubletSeparation") == 0) {
      p->doubletSeparation = atof(rest);
    } else if (strcmp(key, "PeakLocation") == 0) {
      if (p->nPeakLocations < MAX_PEAK_LOCATIONS_PF) {
        p->peakLocations[p->nPeakLocations++] = atof(rest);
        p->doPeakFit = 1;
      }
    }
  }
  fclose(f);
  if (p->chunkFiles == 0)
    p->chunkFiles = 1;
  printf("readParamFile: %s\n", filename);
  printf("  NrPixels: %d x %d, Lsd: %.1f, px: %.4f, Wavelength: %.6f\n",
         p->NrPixelsY, p->NrPixelsZ, p->Lsd, p->px, p->Lam);
  printf("  R: [%.1f, %.1f] bin=%.4f, Eta: [%.1f, %.1f] bin=%.2f\n", p->RMin,
         p->RMax, p->RBinSize, p->EtaMin, p->EtaMax, p->EtaBinSize);
  return 0;
}

// ─────────────────────────────────────────────────────────────────
// readDataFile: read a single frame via FileReader.h
// ─────────────────────────────────────────────────────────────────
static int readDataFile(const char *filename, size_t NrPixels,
                        double *returnArr, const char *h5Loc) {
  const char *ext = strrchr(filename, '.');
  if (!ext) {
    fprintf(stderr, "ERROR: No extension in '%s'\n", filename);
    return -1;
  }
  if (strcasecmp(ext, ".tif") == 0 || strcasecmp(ext, ".tiff") == 0) {
    return ReadTiffFrame(filename, 0, NrPixels, returnArr, 0);
  } else if (strcasecmp(ext, ".h5") == 0 || strcasecmp(ext, ".hdf5") == 0 ||
             strcasecmp(ext, ".hdf") == 0) {
    const char *loc = (h5Loc && h5Loc[0]) ? h5Loc : "exchange/data";
    return ReadHDF5Frame(filename, loc, NrPixels, returnArr, 0);
  } else if (strcasecmp(ext, ".cbf") == 0) {
    return ReadCBFFrame(filename, NrPixels, returnArr, NULL, NULL);
  } else {
    /* Binary fallback */
    FILE *f = fopen(filename, "rb");
    if (!f) {
      fprintf(stderr, "ERROR: Cannot open '%s'\n", filename);
      return -1;
    }
    fseek(f, 8192, SEEK_SET); /* skip GE header */
    int rc = ReadBinaryFrame(f, 1, NrPixels, returnArr);
    fclose(f);
    return rc;
  }
}

int main(int argc, char **argv) {
  clock_t start, end, start0, end0;
  start0 = clock();
  double diftotal;
  int nCPUs = 4;
  char *PeakParamsFN = NULL;
  int benchmarkIters = 0; /* -benchmark N: repeat integration kernel N times */

  // ── Mode detection ────────────────────────────────────────────
  int useParamFile = 0;
  char *paramFN = NULL, *dataFN_arg = NULL, *darkFN_arg = NULL;
  char *dataLoc_arg = NULL, *darkLoc_arg = NULL;

  if (argc >= 2 && strcmp(argv[1], "-paramFN") == 0) {
    useParamFile = 1;
    /* Flag-based argument parsing */
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-paramFN") == 0 && i + 1 < argc)
        paramFN = argv[++i];
      else if (strcmp(argv[i], "-dataFN") == 0 && i + 1 < argc)
        dataFN_arg = argv[++i];
      else if (strcmp(argv[i], "-darkFN") == 0 && i + 1 < argc)
        darkFN_arg = argv[++i];
      else if (strcmp(argv[i], "-dataLoc") == 0 && i + 1 < argc)
        dataLoc_arg = argv[++i];
      else if (strcmp(argv[i], "-darkLoc") == 0 && i + 1 < argc)
        darkLoc_arg = argv[++i];
      else if (strcmp(argv[i], "-nCPUs") == 0 && i + 1 < argc)
        nCPUs = atoi(argv[++i]);
      else if (strcmp(argv[i], "-PeakParamsFN") == 0 && i + 1 < argc)
        PeakParamsFN = argv[++i];
      else if (strcmp(argv[i], "-benchmark") == 0 && i + 1 < argc)
        benchmarkIters = atoi(argv[++i]);
    }
    if (!paramFN || !dataFN_arg) {
      fprintf(stderr,
              "Usage: %s -paramFN params.txt -dataFN image.tif "
              "[-darkFN dark.tif] [-dataLoc path/in/h5] [-darkLoc path/in/h5] "
              "[-nCPUs N] [-PeakParamsFN peaks.txt]\n",
              argv[0]);
      return 1;
    }
  } else if (argc >= 2) {
    /* Existing zarr mode: positional args */
    dataFN_arg = argv[1];
    if (argc == 3) {
      if (atoi(argv[2]) > 0)
        nCPUs = atoi(argv[2]);
      else
        PeakParamsFN = argv[2];
    } else if (argc == 4) {
      nCPUs = atoi(argv[2]);
      PeakParamsFN = argv[3];
    } else if (argc > 4) {
      printf("Usage: %s ZarrName.zip [nCPUs] [PeakParamsFN]\n", argv[0]);
      printf("       %s -paramFN params.txt -dataFN image.tif [...]\n",
             argv[0]);
      return 1;
    }
  } else {
    printf("Usage: %s ZarrName.zip [nCPUs] [PeakParamsFN]\n", argv[0]);
    printf("       %s -paramFN params.txt -dataFN image.tif "
           "[-darkFN dark.tif] [-dataLoc path/in/h5] [-darkLoc path/in/h5] "
           "[-nCPUs N] [-PeakParamsFN peaks.txt]\n",
           argv[0]);
    return 1;
  }

  omp_set_num_threads(nCPUs);
  printf("Version: %s\n", MIDAS_VERSION_STRING);
  printf("Running with %d OpenMP threads.\n", nCPUs);

  double RMax, RMin, RBinSize, EtaMax, EtaMin, EtaBinSize, Lsd, px;
  double BC_y = 0, BC_z = 0;
  int NrPixelsY = 2048, NrPixelsZ = 2048, Normalize = 1;
  int GradientCorrection = 0;
  double *dIdR = NULL; /* radial gradient image (pre-computed per frame) */
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
  double omeStart = 0, omeStep = 0;
  double Lam = 0.172978, Polariz = 0.99, SHpL = 0.002, U = 1.163, V = -0.126,
         W = 0.063, X = 0.0, Y = 0.0, Z = 0.0;
  double QBinSize = 0.0, QMin = 0.0, QMax = 0.0;
  // Peak fitting parameters
  int doPeakFit = 0, nPeakLocations = 0, fitROIPadding = 20;
  int autoDetectPeaks = 0, snipIterations = 50;
  double doubletSeparation = 0.0;
  double peakLocations[MAX_PEAK_LOCATIONS_PF];

  char *DataFN = dataFN_arg;
  char *resultFolder = NULL;
  int nFrames = 0, nDarks = 0, nFloods = 0, bytesPerPx = 2;
  int paramFN_frames_summed = 1;  // actual count of frames summed in paramFN mode
  double *AverageDark = NULL;
  pixelvalue *ImageInT = NULL;
  double *Image = NULL;
  int nPixels;

  blosc2_init();

  if (useParamFile) {
    // ── PARAM-FILE MODE ──────────────────────────────────────
    struct IntegratorParams ip;
    initIntegratorParams(&ip);
    if (readParamFile(paramFN, &ip) != 0)
      return 1;

    // Override with PeakParamsFN if provided
    if (PeakParamsFN != NULL) {
      FILE *pf = fopen(PeakParamsFN, "r");
      if (pf) {
        char pfline[4096], pfkey[256], pfval[3072];
        while (fgets(pfline, sizeof(pfline), pf)) {
          if (pfline[0] == '#' || pfline[0] == '\n')
            continue;
          if (sscanf(pfline, "%255s %[^\n]", pfkey, pfval) == 2) {
            if (strcmp(pfkey, "DoPeakFit") == 0)
              ip.doPeakFit = atoi(pfval);
            else if (strcmp(pfkey, "FitROIPadding") == 0)
              ip.fitROIPadding = atoi(pfval);
            else if (strcmp(pfkey, "AutoDetectPeaks") == 0) {
              ip.autoDetectPeaks = atoi(pfval);
              ip.doPeakFit = 1;
            } else if (strcmp(pfkey, "SNIPIterations") == 0)
              ip.snipIterations = atoi(pfval);
            else if (strcmp(pfkey, "DoubletSeparation") == 0)
              ip.doubletSeparation = atof(pfval);
            else if (strcmp(pfkey, "PeakLocation") == 0) {
              if (ip.nPeakLocations < MAX_PEAK_LOCATIONS_PF) {
                ip.peakLocations[ip.nPeakLocations++] = atof(pfval);
                ip.doPeakFit = 1;
              }
            }
          }
        }
        fclose(pf);
        printf("Read peak params from %s: DoPeakFit=%d, nPeaks=%d, "
               "AutoDetect=%d, SNIP=%d, ROIPadding=%d\n",
               PeakParamsFN, ip.doPeakFit, ip.nPeakLocations,
               ip.autoDetectPeaks, ip.snipIterations, ip.fitROIPadding);
      }
    }

    // Copy params to local vars used by integration loop
    RMax = ip.RMax;
    RMin = ip.RMin;
    RBinSize = ip.RBinSize;
    if (ip.QBinSize > 0)
      QBinSize = ip.QBinSize;
    if (ip.QMin > 0)
      QMin = ip.QMin;
    if (ip.QMax > 0)
      QMax = ip.QMax;
    EtaMax = ip.EtaMax;
    EtaMin = ip.EtaMin;
    EtaBinSize = ip.EtaBinSize;
    Lsd = ip.Lsd;
    px = ip.px;
    Lam = ip.Lam;
    BC_y = ip.BC_y;
    BC_z = ip.BC_z;
    NrPixelsY = ip.NrPixelsY;
    NrPixelsZ = ip.NrPixelsZ;
    Normalize = ip.Normalize;
    GradientCorrection = ip.GradientCorrection;
    skipFrame = ip.skipFrame;
    NrTransOpt = ip.NrTransOpt;
    for (int i = 0; i < NrTransOpt; i++)
      TransOpt[i] = ip.TransOpt[i];
    omeStart = ip.omeStart;
    omeStep = ip.omeStep;
    haveOmegas = ip.haveOmegas;
    chunkFiles = ip.chunkFiles;
    sumImages = ip.sumImages;
    individualSave = ip.individualSave;
    Polariz = ip.Polariz;
    SHpL = ip.SHpL;
    U = ip.U;
    V = ip.V;
    W = ip.W;
    X = ip.X;
    Y = ip.Y;
    Z = ip.Z;
    doPeakFit = ip.doPeakFit;
    nPeakLocations = ip.nPeakLocations;
    fitROIPadding = ip.fitROIPadding;
    autoDetectPeaks = ip.autoDetectPeaks;
    snipIterations = ip.snipIterations;
    doubletSeparation = ip.doubletSeparation;
    for (int i = 0; i < nPeakLocations; i++)
      peakLocations[i] = ip.peakLocations[i];
    GapFN = ip.hasGapFN ? ip.GapFN : NULL;
    BadPxFN = ip.hasBadPxFN ? ip.BadPxFN : NULL;

    // In paramFN mode, the caller sets cwd to the work directory,
    // so resultFolder should always be "." to avoid double-pathing.
    resultFolder = strdup(".");
    sprintf(outputFolder, "%s", resultFolder);
    struct stat st = {0};
    if (stat(outputFolder, &st) == -1) {
      printf("Output folder '%s' did not exist. Creating now.\n", outputFolder);
      mkdir(outputFolder, 0700);
    }

    nPixels = NrPixelsY * NrPixelsZ;
    nFrames = 1; // Loop runs once; if chunkFiles > 1 we pre-sum into ImageInT
    nDarks = 0;

    // OmegaSumFrames (params key) > 1 means "sum the first N frames of the
    // HDF5 dataset into one image before integration/peak-fitting". The
    // peak-fit step then operates on the summed frame. Dark is scaled
    // accordingly so background subtraction stays correct.
    int framesToSum = (chunkFiles > 1) ? chunkFiles : 1;

    // Read Map.bin / nMap.bin
    int rc = ReadBins(resultFolder);

    // Read data frame(s)
    ImageInT = malloc(nPixels * sizeof(*ImageInT));
    AverageDark = calloc(nPixels, sizeof(*AverageDark));

    const char *dataExt = strrchr(DataFN, '.');
    int dataIsH5 = dataExt && (strcasecmp(dataExt, ".h5") == 0 ||
                               strcasecmp(dataExt, ".hdf5") == 0 ||
                               strcasecmp(dataExt, ".hdf") == 0);
    const char *dataLocUsed =
        (dataLoc_arg && dataLoc_arg[0]) ? dataLoc_arg : "exchange/data";

    // Probe how many frames the data dataset actually has so we can cap
    // framesToSum (if the user asks for more than exist) and so we know
    // whether a single-frame read is needed.
    int dataFramesAvail = 1;
    if (dataIsH5) {
      hid_t _fid = H5Fopen(DataFN, H5F_ACC_RDONLY, H5P_DEFAULT);
      if (_fid >= 0) {
        hid_t _did = H5Dopen2(_fid, dataLocUsed, H5P_DEFAULT);
        if (_did >= 0) {
          hid_t _sid = H5Dget_space(_did);
          hsize_t _dims[3] = {0, 0, 0};
          int _nd = H5Sget_simple_extent_dims(_sid, _dims, NULL);
          if (_nd == 3) dataFramesAvail = (int)_dims[0];
          H5Sclose(_sid);
          H5Dclose(_did);
        }
        H5Fclose(_fid);
      }
    }
    if (framesToSum > dataFramesAvail) framesToSum = dataFramesAvail;
    if (framesToSum < 1) framesToSum = 1;
    paramFN_frames_summed = framesToSum;

    double *rawFrame = calloc(nPixels, sizeof(double));
    if (!rawFrame) {
      fprintf(stderr, "ERROR: calloc(rawFrame) failed\n");
      return 1;
    }

    if (dataIsH5 && framesToSum > 1) {
      printf("Summing %d frames from HDF5 data file: %s (loc: %s)\n",
             framesToSum, DataFN, dataLocUsed);
      double *frameBuf = calloc(nPixels, sizeof(double));
      if (!frameBuf) {
        fprintf(stderr, "ERROR: calloc(frameBuf) failed\n");
        free(rawFrame);
        return 1;
      }
      for (int fi = 0; fi < framesToSum; fi++) {
        if (ReadHDF5Frame(DataFN, dataLocUsed, nPixels, frameBuf, fi) != FR_SUCCESS) {
          fprintf(stderr, "ERROR: Failed to read frame %d of %s\n", fi, DataFN);
          free(frameBuf);
          free(rawFrame);
          return 1;
        }
        for (size_t p = 0; p < nPixels; p++)
          rawFrame[p] += frameBuf[p];
      }
      free(frameBuf);
    } else {
      printf("Reading data file: %s (HDF5 loc: %s)\n", DataFN, dataLocUsed);
      if (readDataFile(DataFN, nPixels, rawFrame, dataLoc_arg) != FR_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to read data file: %s\n", DataFN);
        free(rawFrame);
        return 1;
      }
    }

    // Copy summed (or single) frame to ImageInT
    for (int i = 0; i < nPixels; i++)
      ImageInT[i] = rawFrame[i];

    // Dark subtraction — scale to framesToSum so sum(data) - N*mean(dark)
    if (darkFN_arg) {
      double *darkFrame = calloc(nPixels, sizeof(double));
      const char *darkLocUsed =
          (darkLoc_arg && darkLoc_arg[0])
              ? darkLoc_arg
              : ((dataLoc_arg && dataLoc_arg[0]) ? dataLoc_arg : "exchange/data");
      const char *darkExt = strrchr(darkFN_arg, '.');
      int darkIsH5 = darkExt && (strcasecmp(darkExt, ".h5") == 0 ||
                                 strcasecmp(darkExt, ".hdf5") == 0 ||
                                 strcasecmp(darkExt, ".hdf") == 0);
      printf("Reading dark file: %s (HDF5 loc: %s)\n", darkFN_arg, darkLocUsed);

      int darkOK = 0;
      if (darkIsH5) {
        // Use SumHDF5Frames which returns the mean across all dark frames.
        if (SumHDF5Frames(darkFN_arg, darkLocUsed, nPixels, darkFrame, 0) ==
            FR_SUCCESS) {
          darkOK = 1;
        }
      } else {
        if (readDataFile(darkFN_arg, nPixels, darkFrame, darkLocUsed) ==
            FR_SUCCESS) {
          darkOK = 1;
        }
      }
      if (darkOK) {
        // sum(data) = N × mean(data); subtract N × mean(dark) so each pixel
        // is "sum of background-corrected frames".
        for (int i = 0; i < nPixels; i++)
          AverageDark[i] = darkFrame[i] * (double)framesToSum;
      } else {
        printf("Warning: Failed to read dark file, proceeding without dark.\n");
      }
      free(darkFrame);
    }
    free(rawFrame);

    // ImageInT is now used directly (no transform needed)

    // Mask setup
    if (GapFN || BadPxFN) {
      makeMap = 2;
    }

    printf("ParamFile mode: ready for integration (%d x %d, %d frame)\n",
           NrPixelsY, NrPixelsZ, nFrames);

    // Skip to integration (goto label after zarr reading)
    goto integration_start;
  }

  // ── ZARR MODE (existing code) ────────────────────────────────
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
  resultFolder = NULL;
  int locImTransOpt, locTemp = -1, locPres = -1, locI = -1, locI0 = -1;
  nDarks = 0;
  nFloods = 0;
  bytesPerPx = 2;
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
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/QBinSize/0") != NULL) {
      ReadZarrChunk(arch, count, &QBinSize, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/QMin/0") != NULL) {
      ReadZarrChunk(arch, count, &QMin, sizeof(double));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/QMax/0") != NULL) {
      ReadZarrChunk(arch, count, &QMax, sizeof(double));
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
               "analysis/process/analysis_parameters/DoPeakFit/0") != NULL) {
      ReadZarrChunk(arch, count, &doPeakFit, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/FitROIPadding/0") !=
        NULL) {
      ReadZarrChunk(arch, count, &fitROIPadding, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/SumImages/0") != NULL) {
      ReadZarrChunk(arch, count, &sumImages, sizeof(int));
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/GapIntensity/0") != NULL) {
      double tmpGap;
      ReadZarrChunk(arch, count, &tmpGap, sizeof(double));
      GapIntensity = (long long int)tmpGap;
    }
    if (strstr(finfo->name,
               "analysis/process/analysis_parameters/BadPxIntensity/0") !=
        NULL) {
      double tmpBad;
      ReadZarrChunk(arch, count, &tmpBad, sizeof(double));
      BadPxIntensity = (long long int)tmpBad;
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
      // Parse dtype from .zarray to set bytesPerPx/dType as fallback
      ptr = strstr(s, "dtype");
      if (ptr != NULL) {
        char *q1 = strchr(ptr, '"');
        char *q2 = q1 ? strchr(q1 + 1, '"') : NULL;
        char *q3 = q2 ? strchr(q2 + 1, '"') : NULL;
        if (q2 && q3) {
          size_t dlen = q3 - q2 - 1;
          char dtStr[32] = {0};
          if (dlen < sizeof(dtStr)) {
            strncpy(dtStr, q2 + 1, dlen);
            printf("Parsed dtype from .zarray: '%s'\n", dtStr);
            if (strstr(dtStr, "u2")) {
              bytesPerPx = 2;
              dType = 1;
            } else if (strstr(dtStr, "i4")) {
              bytesPerPx = 4;
              dType = 5;
            } else if (strstr(dtStr, "u4")) {
              bytesPerPx = 4;
              dType = 4;
            } else if (strstr(dtStr, "f4")) {
              bytesPerPx = 4;
              dType = 3;
            } else if (strstr(dtStr, "f8")) {
              bytesPerPx = 8;
              dType = 2;
            }
          }
        }
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

  // Read peak fitting parameters from companion file (if provided)
  if (PeakParamsFN != NULL) {
    FILE *pf = fopen(PeakParamsFN, "r");
    if (pf) {
      char pfline[4096], pfkey[256], pfval[3072];
      while (fgets(pfline, sizeof(pfline), pf)) {
        if (pfline[0] == '#' || pfline[0] == '\n')
          continue;
        if (sscanf(pfline, "%255s %[^\n]", pfkey, pfval) == 2) {
          if (strcmp(pfkey, "DoPeakFit") == 0)
            sscanf(pfval, "%d", &doPeakFit);
          else if (strcmp(pfkey, "FitROIPadding") == 0)
            sscanf(pfval, "%d", &fitROIPadding);
          else if (strcmp(pfkey, "AutoDetectPeaks") == 0) {
            sscanf(pfval, "%d", &autoDetectPeaks);
            doPeakFit = 1;
          } else if (strcmp(pfkey, "SNIPIterations") == 0) {
            sscanf(pfval, "%d", &snipIterations);
          } else if (strcmp(pfkey, "DoubletSeparation") == 0) {
            sscanf(pfval, "%lf", &doubletSeparation);
          } else if (strcmp(pfkey, "PeakLocation") == 0) {
            if (nPeakLocations < MAX_PEAK_LOCATIONS_PF) {
              sscanf(pfval, "%lf", &peakLocations[nPeakLocations]);
              nPeakLocations++;
              doPeakFit = 1; // Implicitly enable
            }
          }
        }
      }
      fclose(pf);
      printf("Read peak params from %s: DoPeakFit=%d, nPeaks=%d, "
             "AutoDetect=%d, SNIP=%d, ROIPadding=%d\n",
             PeakParamsFN, doPeakFit, nPeakLocations, autoDetectPeaks,
             snipIterations, fitROIPadding);
    } else {
      printf("Warning: Could not open peak params file: %s\n", PeakParamsFN);
    }
  }

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
  int a, b;
  pixelvalue *DarkInT;
  nPixels = NrPixelsY * NrPixelsZ;
  DarkInT = malloc(nPixels * sizeof(*DarkInT));
  AverageDark = calloc(nPixels, sizeof(*AverageDark));
  ImageInT = malloc(nPixels * sizeof(*ImageInT));
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
    printf("Dark frame compressed size: %ld\n", (long int)finfo->size);
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
    // Map.bin now stores raw pixel coordinates, so skip transformation
    for (b = 0; b < (NrPixelsY * NrPixelsZ); b++) {
      AverageDark[b] += ((double)DarkInT[b]) / (nDarks - skipFrame);
    }
  }
  // for (b=0;b<NrPixelsY*NrPixelsZ;b++) printf("%lf\n",AverageDark[b]);
  free(data);
  zip_close(arch);

integration_start:
  if (!nPixels)
    nPixels = NrPixelsY * NrPixelsZ;
  Image = malloc(nPixels * sizeof(*Image));
  int qMode = (QBinSize > 0 && Lam > 0 && QMin > 0 && QMax > 0);
  if (qMode) {
    nRBins = (int)ceil((QMax - QMin) / QBinSize);
    printf("Q-mode: nQBins=%d, QMin=%.4f, QMax=%.4f, QBinSize=%.6f\n",
           nRBins, QMin, QMax, QBinSize);
  } else {
    nRBins = (int)ceil((RMax - RMin) / RBinSize);
  }
  nEtaBins = (int)ceil((EtaMax - EtaMin) / EtaBinSize);
  double *EtaBinsLow, *EtaBinsHigh;
  double *RBinsLow, *RBinsHigh;
  EtaBinsLow = malloc(nEtaBins * sizeof(*EtaBinsLow));
  EtaBinsHigh = malloc(nEtaBins * sizeof(*EtaBinsHigh));
  RBinsLow = malloc(nRBins * sizeof(*RBinsLow));
  RBinsHigh = malloc(nRBins * sizeof(*RBinsHigh));
  if (qMode) {
    // Eta bins are still uniform
    for (int i = 0; i < nEtaBins; i++) {
      EtaBinsLow[i] = EtaBinSize * i + EtaMin;
      EtaBinsHigh[i] = EtaBinSize * (i + 1) + EtaMin;
    }
    // R bins from uniform Q spacing: R(Q) = (Lsd/px) * tan(2*asin(Q*Lam/(4*PI)))
    for (int i = 0; i < nRBins; i++) {
      double QLow = QMin + QBinSize * i;
      double QHigh = QMin + QBinSize * (i + 1);
      RBinsLow[i] = (Lsd / px) * tan(2.0 * asin(QLow * Lam / (4.0 * M_PI)));
      RBinsHigh[i] = (Lsd / px) * tan(2.0 * asin(QHigh * Lam / (4.0 * M_PI)));
    }
  } else {
    midas_reta_mapper(RMin, EtaMin, nEtaBins, nRBins, EtaBinSize, RBinSize,
                      EtaBinsLow, EtaBinsHigh, RBinsLow, RBinsHigh);
  }

  // Compute R bin centers for peak fitting and lineout output
  double *RBinCenters = malloc(nRBins * sizeof(double));
  for (int rb = 0; rb < nRBins; rb++) {
    RBinCenters[rb] = (RBinsLow[rb] + RBinsHigh[rb]) / 2.0;
  }

  // Build data-file-based prefix for output files (e.g. "map1_00001")
  char _dfn_buf[4096];
  sprintf(_dfn_buf, "%s", DataFN);
  char *_bn = basename(_dfn_buf);
  char dataPrefix[4096];
  sprintf(dataPrefix, "%s", _bn);
  // Strip extension from prefix
  char *_dot = strrchr(dataPrefix, '.');
  if (_dot)
    *_dot = '\0';

  // Open binary output files (named per data file for parallel safety)
  FILE *fLineout = NULL, *fFitBin = NULL;
  char lineoutFN[4096];
  sprintf(lineoutFN, "%s_lineout.bin", dataPrefix);
  fLineout = fopen(lineoutFN, "wb");
  if (fLineout)
    printf("Opened %s for binary output.\n", lineoutFN);
  char lineoutXYFN[4096];
  sprintf(lineoutXYFN, "%s_lineout.xy", dataPrefix);
  FILE *fLineoutXY = fopen(lineoutXYFN, "w");
  if (fLineoutXY) {
    if (qMode)
      fprintf(fLineoutXY, "# Q_invA  intensity\n");
    else
      fprintf(fLineoutXY, "# 2theta_deg  intensity\n");
    printf("Opened %s for text lineout output.\n", lineoutXYFN);
  }
  FILE *fFitPerEta = NULL;
  int peakFitActive = doPeakFit && (nPeakLocations > 0 || autoDetectPeaks > 0);
  if (peakFitActive) {
    int nPeaksForOutput =
        (autoDetectPeaks > 0) ? autoDetectPeaks : nPeakLocations;
    char fitBinFN[4096];
    sprintf(fitBinFN, "%s_fit.bin", dataPrefix);
    fFitBin = fopen(fitBinFN, "wb");
    if (fFitBin)
      printf("Opened %s for peak fitting output (%d peaks%s).\n", fitBinFN,
             nPeaksForOutput, autoDetectPeaks > 0 ? ", auto-detect" : "");
    char fitPerEtaFN[4096];
    sprintf(fitPerEtaFN, "%s_fit_per_eta.csv", dataPrefix);
    fFitPerEta = fopen(fitPerEtaFN, "w");
    if (fFitPerEta) {
      fprintf(fFitPerEta,
              "Frame,EtaBin,EtaCen,PeakIdx,R_px,R_um,TwoTheta_deg,"
              "Area,Sig_centideg2,Gam_centideg,FWHM_deg,Eta,ChiSq,Area2\n");
      printf("Opened %s for per-eta peak fitting output.\n", fitPerEtaFN);
    }
    printf("Peak fit enabled: %d peaks, ROI padding: %d bins\n", nPeakLocations,
           fitROIPadding);
  }

  // HDF5 peak output buffer
  PeakH5Buffer h5buf;
  if (peakFitActive) {
    pfio_init_buffer(&h5buf, 4096);
  }

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
  // Per-pixel masking removed: DetectorMapper's maskMap.bin provides
  // bin-level mask via binMaskFlag (applied at L1793).
  (void)makeMap; (void)mapMask; (void)mapMaskSize;
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
  // Use float accumulators when FLOAT32_ACCUM is defined (precision test)
#ifdef FLOAT32_ACCUM
  float Intensity, totArea;
  double ThisInt;
#else
  double Intensity, totArea, ThisInt;
#endif
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
  double presThis = 0, tempThis = 0, iThis = 0, i0This = 0;
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
    if (useParamFile) {
      // In paramFN mode: ImageInT already holds the raw frame,
      // AverageDark is set. Just compute Image = ImageInT - dark.
      for (j = 0; j < NrPixelsY * NrPixelsZ; j++) {
        Image[j] = (double)ImageInT[j] - AverageDark[j];
      }
    } else {
      presThis += Pressure[i];
      tempThis += Temperature[i];
      iThis += I[i];
      i0This = I0[i];
      dsz = NrPixelsY * NrPixelsZ * bytesPerPx; // Reset buffer capacity
      dsz = blosc1_decompress(&allData[sizeArr[i * 2 + 1]], locData, dsz);
      if (dsz <= 0) {
        printf(
            "Error: Failed to decompress frame data at index %d! dsize: %d\n",
            i, dsz);
        exit(1);
      }
      rawToDouble(locData, ImageInT, nPixels, dType);
      for (j = 0; j < NrPixelsY * NrPixelsZ; j++) {
        Image[j] = (double)ImageInT[j] - AverageDark[j];
      }
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
      PerFrameArr = malloc(bigArrSize * 5 * sizeof(*PerFrameArr));
    }
    memset(IntArrPerFrame, 0, bigArrSize * sizeof(double));
    // Diagnostic: count mapped pixels with value -1 or -2
    {
      long neg1_total = 0, neg2_total = 0;
      for (j = 0; j < nRBins; j++) {
        for (k = 0; k < nEtaBins; k++) {
          long long int pos = (long long int)j * nEtaBins + k;
          int npx = nPxList[2 * pos + 0];
          int dp = nPxList[2 * pos + 1];
          for (l = 0; l < npx; l++) {
            size_t tp = (size_t)pxList[dp + l].z * NrPixelsY + pxList[dp + l].y;
            double val = Image[tp];
            if (val == -1.0) neg1_total++;
            else if (val == -2.0) neg2_total++;
          }
        }
      }
      printf("  Pixel diagnostic (mapped only): "
             "val=-1: %ld in map; val=-2: %ld in map\n",
             neg1_total, neg2_total);
    }
    t_0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64) private(j, k, l, Pos, nPixels, dataPos, Intensity,    \
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
        for (l = 0; l < nPixels; l++) {
          ThisVal = pxList[dataPos + l];
          double read_y = ThisVal.y, read_z = ThisVal.z;
          if (GradientCorrection && ThisVal.deltaR != 0.0f) {
            double dy = ThisVal.y - BC_y;
            double dz = ThisVal.z - BC_z;
            double R = sqrt(dy * dy + dz * dz);
            if (R > 1.0) {
              read_y -= ThisVal.deltaR * dy / R;
              read_z -= ThisVal.deltaR * dz / R;
            }
          }
          int iy = (int)floorf(read_y);
          int iz = (int)floorf(read_z);
          double fy = read_y - iy, fz = read_z - iz;
          if (iy < 0) { iy = 0; fy = 0; }
          if (iy >= NrPixelsY - 1) { iy = NrPixelsY - 2; fy = 1; }
          if (iz < 0) { iz = 0; fz = 0; }
          if (iz >= NrPixelsZ - 1) { iz = NrPixelsZ - 2; fz = 1; }
          double pixVal =
              Image[(size_t)iz * NrPixelsY + iy] * (1 - fy) * (1 - fz) +
              Image[(size_t)iz * NrPixelsY + iy + 1] * fy * (1 - fz) +
              Image[(size_t)(iz + 1) * NrPixelsY + iy] * (1 - fy) * fz +
              Image[(size_t)(iz + 1) * NrPixelsY + iy + 1] * fy * fz;
          Intensity += pixVal * ThisVal.frac;
          totArea += ThisVal.areaWeight;
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
          double twoTheta_rad = atan(RMean * px / Lsd);
          PerFrameArr[4 * bigArrSize + (j * nEtaBins + k)] =
              (Lam > 0) ? (4.0 * M_PI / Lam) * sin(twoTheta_rad / 2.0) : 0.0;
        }
        IntArrPerFrame[j * nEtaBins + k] = Intensity;
        // If this bin is flagged as contaminated by a masked pixel, force NAN
        if (binMaskFlag != NULL && binMaskFlag[j * nEtaBins + k])
          IntArrPerFrame[j * nEtaBins + k] = NAN;
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

    /* ── Benchmark mode: repeat kernel N-1 more times (first iter above) ── */
    if (benchmarkIters > 1 && i == 0) {
      printf("BENCHMARK_CSV_HEADER,iter,wall_sec\n");
      printf("BENCHMARK_CSV,0,%.9f\n", omp_get_wtime() - t_0);
      for (int bm = 1; bm < benchmarkIters; bm++) {
        memset(IntArrPerFrame, 0, bigArrSize * sizeof(double));
        double bm_t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64) private(j, k, l, Pos, nPixels,  \
                                         dataPos, Intensity, totArea, ThisVal, \
                                         testPos, ThisInt, RMean, EtaMean)
        for (j = 0; j < nRBins; j++) {
          RMean = (RBinsLow[j] + RBinsHigh[j]) / 2;
          for (k = 0; k < nEtaBins; k++) {
            Pos = j * nEtaBins + k;
            nPixels = nPxList[2 * Pos + 0];
            dataPos = nPxList[2 * Pos + 1];
            Intensity = 0;
            totArea = 0;
            for (l = 0; l < nPixels; l++) {
              ThisVal = pxList[dataPos + l];
              double read_y = ThisVal.y, read_z = ThisVal.z;
              int iy = (int)floorf(read_y);
              int iz = (int)floorf(read_z);
              double fy = read_y - iy, fz = read_z - iz;
              if (iy < 0) { iy = 0; fy = 0; }
              if (iy >= NrPixelsY - 1) { iy = NrPixelsY - 2; fy = 1; }
              if (iz < 0) { iz = 0; fz = 0; }
              if (iz >= NrPixelsZ - 1) { iz = NrPixelsZ - 2; fz = 1; }
              double pixVal =
                  Image[(size_t)iz * NrPixelsY + iy] * (1 - fy) * (1 - fz) +
                  Image[(size_t)iz * NrPixelsY + iy + 1] * fy * (1 - fz) +
                  Image[(size_t)(iz + 1) * NrPixelsY + iy] * (1 - fy) * fz +
                  Image[(size_t)(iz + 1) * NrPixelsY + iy + 1] * fy * fz;
              Intensity += pixVal * ThisVal.frac;
              totArea += ThisVal.areaWeight;
            }
            if (Intensity != 0 && Normalize == 1) {
              Intensity /= totArea;
            }
            IntArrPerFrame[j * nEtaBins + k] = Intensity;
          }
        }
        double bm_dt = omp_get_wtime() - bm_t0;
        printf("BENCHMARK_CSV,%d,%.9f\n", bm, bm_dt);
      }
      fflush(stdout);
    }

    // --- Compute eta-summed 1D lineout and write binary output ---
    double *lineout1D = (double *)calloc(nRBins, sizeof(double));
#pragma omp parallel for schedule(static)
    for (j = 0; j < nRBins; j++) {
      double sum = 0;
      int cnt = 0;
      for (k = 0; k < nEtaBins; k++) {
        double val = IntArrPerFrame[j * nEtaBins + k];
        if (val != 0 && !isnan(val)) {
          sum += val;
          cnt++;
        }
      }
      lineout1D[j] = (cnt > 0) ? sum / cnt : 0;
    }
    // Write lineout.bin: nRBins doubles per frame (eta-averaged lineout)
    if (fLineout) {
      fwrite(lineout1D, sizeof(double), nRBins, fLineout);
      fflush(fLineout);
    }
    // Write lineout.xy: 2theta (deg) + intensity per R bin
    if (fLineoutXY) {
      for (j = 0; j < nRBins; j++) {
        if (isnan(lineout1D[j]))
          continue; // skip contaminated bins
        if (qMode) {
          double qVal = QMin + QBinSize * (j + 0.5);
          fprintf(fLineoutXY, "%.6f  %.6f\n", qVal, lineout1D[j]);
        } else {
          double tth = atand(RBinCenters[j] * px / Lsd);
          fprintf(fLineoutXY, "%.6f  %.6f\n", tth, lineout1D[j]);
        }
      }
      fflush(fLineoutXY);
    }
    // Peak fitting
    if (peakFitActive) {
      int nPeaksThisFit =
          (autoDetectPeaks > 0) ? autoDetectPeaks : nPeakLocations;
      double *fitResults =
          (double *)calloc(nPeaksThisFit * PF_PARAMS_PER_PEAK, sizeof(double));
      int nFitted;
      if (autoDetectPeaks > 0) {
        nFitted = fitPeaksAutoDetect(RBinCenters, lineout1D, nRBins,
                                     autoDetectPeaks, RBinSize, fitROIPadding,
                                     fitResults, snipIterations);
      } else {
        nFitted = fitPeaks(RBinCenters, lineout1D, nRBins, peakLocations,
                           nPeakLocations, RBinSize, fitROIPadding, fitResults);
      }
      if (nFitted > 0 && fFitBin) {
        fwrite(fitResults, sizeof(double), nFitted * PF_PARAMS_PER_PEAK,
               fFitBin);
        fflush(fFitBin);
      }
      if (i == 0) {
        if (nFitted > 0) {
          printf("  Peak fit results (frame 0): %d/%d peaks fitted%s\n",
                 nFitted, nPeaksThisFit,
                 autoDetectPeaks > 0 ? " (auto-detect)" : "");
          for (int pp = 0; pp < nFitted; pp++) {
            int base = pp * PF_PARAMS_PER_PEAK;
            printf("    Peak %d: Center=%.4f, Area=%.2f, FWHM=%.4f, eta=%.2f\n",
                   pp, fitResults[base + 1], fitResults[base + 0],
                   fitResults[base + 4], fitResults[base + 5]);
          }
        } else {
          printf("  Peak fit FAILED (frame 0): nFitted=0 out of %d peaks\n",
                 nPeaksThisFit);
        }
      }
      free(fitResults);
    }

    // --- Per-eta-bin peak fitting ---
    if (peakFitActive && fFitPerEta) {
      int nPeaksThisFit =
          (autoDetectPeaks > 0) ? autoDetectPeaks : nPeakLocations;

      // Precompute doublet pairs for manual peak locations: adjacent peaks
      // whose |ΔR| < doubletSeparation are fit jointly via pf_fit_doublet_peak
      // to avoid each peak's ROI being contaminated by its neighbor.
      int *dbFlag = NULL, *dbPair = NULL;
      if (doubletSeparation > 0.0 && autoDetectPeaks == 0 && nPeakLocations > 1) {
        dbFlag = (int *)calloc(nPeakLocations, sizeof(int));
        dbPair = (int *)malloc(nPeakLocations * sizeof(int));
        for (int pp = 0; pp < nPeakLocations; pp++) dbPair[pp] = -1;
        for (int pp = 0; pp < nPeakLocations - 1; pp++) {
          if (dbFlag[pp] != 0) continue;
          if (fabs(peakLocations[pp + 1] - peakLocations[pp]) <
              doubletSeparation) {
            dbFlag[pp] = 1; dbFlag[pp + 1] = 2;
            dbPair[pp] = pp + 1; dbPair[pp + 1] = pp;
          }
        }
        int nDoublets = 0;
        for (int pp = 0; pp < nPeakLocations; pp++)
          if (dbFlag[pp] == 1) nDoublets++;
        if (i == 0 && nDoublets > 0) {
          printf("  Per-eta doublet detection: %d doublet pair(s) "
                 "at sep<%.3f px\n", nDoublets, doubletSeparation);
        }
      }

      double *etaProfile = (double *)malloc(nRBins * sizeof(double));
      double *etaFitResults =
          (double *)calloc(nPeaksThisFit * PF_PARAMS_PER_PEAK, sizeof(double));
      int totalPerEtaFitted = 0;
      for (int eb = 0; eb < nEtaBins; eb++) {
        // Extract the 1D R-profile for this eta bin
        int nonZero = 0;
        for (int rb = 0; rb < nRBins; rb++) {
          etaProfile[rb] = IntArrPerFrame[rb * nEtaBins + eb];
          if (etaProfile[rb] != 0)
            nonZero++;
        }
        if (nonZero < 5)
          continue; // skip nearly-empty eta bins

        memset(etaFitResults, 0,
               nPeaksThisFit * PF_PARAMS_PER_PEAK * sizeof(double));
        int nf = 0;
        if (autoDetectPeaks > 0) {
          nf = fitPeaksAutoDetect(RBinCenters, etaProfile, nRBins,
                                  autoDetectPeaks, RBinSize, fitROIPadding,
                                  etaFitResults, snipIterations);
        } else {
          // Use CI's peak fitting for each peak (joint if part of a doublet)
          for (int pp = 0; pp < nPeakLocations; pp++) {
            // Skip the trailing member of a doublet — handled via its partner
            if (dbFlag && dbFlag[pp] == 2) continue;

            double peakR = peakLocations[pp];
            int pairIdx = (dbFlag && dbFlag[pp] == 1) ? dbPair[pp] : -1;
            double peakR2 = (pairIdx >= 0) ? peakLocations[pairIdx] : 0.0;

            // ROI bounds: expand to cover both peaks if a doublet
            double roiLoR, roiHiR;
            if (pairIdx >= 0) {
              double plo = (peakR < peakR2) ? peakR : peakR2;
              double phi = (peakR < peakR2) ? peakR2 : peakR;
              roiLoR = plo - fitROIPadding * RBinSize;
              roiHiR = phi + fitROIPadding * RBinSize;
            } else {
              roiLoR = peakR - fitROIPadding * RBinSize;
              roiHiR = peakR + fitROIPadding * RBinSize;
            }

            int lo = -1, hi = -1;
            for (int rb = 0; rb < nRBins; rb++) {
              if (RBinCenters[rb] >= roiLoR && lo < 0) lo = rb;
              if (RBinCenters[rb] <= roiHiR) hi = rb;
            }
            if (lo < 0 || hi < 0 || hi - lo < 5) continue;
            int roiN = hi - lo + 1;
            double *roiR = &RBinCenters[lo];
            double *roiI = (double *)malloc(roiN * sizeof(double));
            for (int rb = 0; rb < roiN; rb++)
              roiI[rb] = etaProfile[lo + rb];

            if (pairIdx >= 0) {
              // Joint doublet fit
              double Rfit1, Rfit2, snr1, snr2, fwhm1, fwhm2;
              double lowR = (peakR < peakR2) ? peakR : peakR2;
              double highR = (peakR < peakR2) ? peakR2 : peakR;
              int lowIdx = (peakR < peakR2) ? pp : pairIdx;
              int highIdx = (peakR < peakR2) ? pairIdx : pp;
              double Rmid = (lowR + highR) * 0.5;
              pf_fit_doublet_peak(PF_MODE_PV, roiN, roiR, roiI,
                                   &Rfit1, &Rfit2, &snr1, &snr2,
                                   &fwhm1, &fwhm2,
                                   RBinSize, lowR, highR, Rmid);
              free(roiI);
              if (snr1 > 1.0) {
                int base = lowIdx * PF_PARAMS_PER_PEAK;
                etaFitResults[base + 1] = Rfit1;
                etaFitResults[base + 6] = snr1;
              }
              if (snr2 > 1.0) {
                int base = highIdx * PF_PARAMS_PER_PEAK;
                etaFitResults[base + 1] = Rfit2;
                etaFitResults[base + 6] = snr2;
              }
            } else {
              // Singleton fit
              double Rfit, snrFit;
              pf_fit_single_peak(PF_MODE_PV, roiN, roiR, roiI, &Rfit, &snrFit, NULL, RBinSize, peakR);
              free(roiI);
              if (snrFit > 1.0) {
                int base = pp * PF_PARAMS_PER_PEAK;
                etaFitResults[base + 0] = 0; // area
                etaFitResults[base + 1] = Rfit; // center
                etaFitResults[base + 2] = 0; // sig
                etaFitResults[base + 3] = 0; // gam
                etaFitResults[base + 4] = 0; // FWHM
                etaFitResults[base + 5] = 0; // eta
                etaFitResults[base + 6] = snrFit; // SNR
              }
            }
          }
          nf = nPeakLocations; // report all slots (unfitted ones remain 0)
        }
        if (nf > 0) {
          totalPerEtaFitted += nf;
          double etaCen = (EtaBinsLow[eb] + EtaBinsHigh[eb]) / 2.0;
          for (int pp = 0; pp < nf; pp++) {
            int base = pp * PF_PARAMS_PER_PEAK;
            double center_px = etaFitResults[base + 1]; // center
            if (center_px == 0)
              continue; // peak not fitted
            double r_um = center_px * px;
            double tth_deg = atan(r_um / Lsd) * 180.0 / M_PI;
            double fwhm_px = etaFitResults[base + 4]; // FWHM
            fprintf(fFitPerEta,
                    "%d,%d,%.4f,%d,%.6f,%.4f,%.6f,%.4f,%.6f,%.6f,%.4f,%.4f,"
                    "%.4f,%.4f\n",
                    i, eb, etaCen, pp, center_px, r_um, tth_deg,
                    etaFitResults[base + 0], // area
                    etaFitResults[base + 2], // sig
                    etaFitResults[base + 3], // gam
                    fwhm_px,
                    etaFitResults[base + 5], // eta
                    etaFitResults[base + 6], // chi_sq
                    etaFitResults[base + 0]  // area (for backward compat)
            );
            // Append to HDF5 buffer
            PeakH5Row h5row = pfio_make_row(i, eb, pp, etaCen,
                                            &etaFitResults[base], px, Lsd, Lam);
            pfio_append_row(&h5buf, &h5row);
          }
        }
      }
      if (i == 0) {
        printf("  Per-eta peak fit (frame 0): %d total fits across %d eta "
               "bins\n",
               totalPerEtaFitted, nEtaBins);
      }
      free(etaProfile);
      free(etaFitResults);
      if (dbFlag) free(dbFlag);
      if (dbPair) free(dbPair);
    }
    free(lineout1D);

    if (i == 0) {
      hsize_t dims[3] = {5, nRBins, nEtaBins};
      status_f =
          H5LTmake_dataset_double(file_id, "/REtaMap", 3, dims, PerFrameArr);
      H5LTset_attribute_int(file_id, "/REtaMap", "nEtaBins", &nEtaBins, 1);
      H5LTset_attribute_int(file_id, "/REtaMap", "nRBins", &nRBins, 1);
      H5LTset_attribute_string(file_id, "/REtaMap", "Header",
                               "Radius,2Theta,Eta,BinArea,Q");
      H5LTset_attribute_string(file_id, "/REtaMap", "Units",
                               "Pixels,Degrees,Degrees,Pixels,InvAngstrom");
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
        int nSum;
        if (useParamFile) {
          // paramFN mode: nFrames is forced to 1 (one loop iteration over a
          // pre-summed ImageInT), so the omega-based formula is wrong.
          nSum = paramFN_frames_summed;
        } else {
          nSum = (omeStep != 0.0)
                     ? (int)((omeArr[i] - firstOme) / omeStep) + 1
                     : 1;
        }
        if (nSum <= 0) nSum = 1;
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
  if (fLineout)
    fclose(fLineout);
  if (fLineoutXY)
    fclose(fLineoutXY);
  if (fFitBin)
    fclose(fFitBin);
  if (fFitPerEta)
    fclose(fFitPerEta);
  if (peakFitActive) {
    // Write HDF5 peaks file
    char peaksH5FN[4096];
    sprintf(peaksH5FN, "%s_caked_peaks.h5", dataPrefix);
    // Compute tth_axis for HDF5 metadata
    double *tthAxis = (double *)malloc(nRBins * sizeof(double));
    double *etaAxis = (double *)malloc(nEtaBins * sizeof(double));
    for (int ri = 0; ri < nRBins; ri++) {
      tthAxis[ri] = atan(RBinCenters[ri] * px / Lsd) * 180.0 / M_PI;
    }
    for (int ei = 0; ei < nEtaBins; ei++) {
      etaAxis[ei] = (EtaBinsLow[ei] + EtaBinsHigh[ei]) / 2.0;
    }
    pfio_write_peaks_h5(peaksH5FN, &h5buf, tthAxis, nRBins, etaAxis, nEtaBins,
                        NrTransOpt, DataFN);
    free(tthAxis);
    free(etaAxis);
    pfio_free_buffer(&h5buf);
  }
  free(RBinCenters);
  end0 = clock();
  diftotal = ((double)(end0 - start0)) / CLOCKS_PER_SEC;
  printf("Total time elapsed:\t%f s.\n", diftotal);
  return 0;
}
