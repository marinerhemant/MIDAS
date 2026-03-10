// FileReader.c
// Implementation of common file reading utilities

#include "FileReader.h"

// --- Binary Reading ---

int ReadBinaryFrame(FILE *f, int dType, size_t NrPixels, double *returnArr) {
  size_t elsRead = 0;
  int i;

  if (dType == 1) { // uint16
    uint16_t *buf = calloc(NrPixels, sizeof(uint16_t));
    if (!buf)
      return FR_ERR_MEMORY;
    elsRead = fread(buf, sizeof(uint16_t), NrPixels, f);
    for (i = 0; i < NrPixels; i++)
      returnArr[i] = (double)buf[i];
    free(buf);
  } else if (dType == 2) { // double
    double *buf = calloc(NrPixels, sizeof(double));
    if (!buf)
      return FR_ERR_MEMORY;
    elsRead = fread(buf, sizeof(double), NrPixels, f);
    for (i = 0; i < NrPixels; i++)
      returnArr[i] = (double)buf[i];
    free(buf);
  } else if (dType == 3) { // float
    float *buf = calloc(NrPixels, sizeof(float));
    if (!buf)
      return FR_ERR_MEMORY;
    elsRead = fread(buf, sizeof(float), NrPixels, f);
    for (i = 0; i < NrPixels; i++)
      returnArr[i] = (double)buf[i];
    free(buf);
  } else if (dType == 4) { // uint32
    uint32_t *buf = calloc(NrPixels, sizeof(uint32_t));
    if (!buf)
      return FR_ERR_MEMORY;
    elsRead = fread(buf, sizeof(uint32_t), NrPixels, f);
    for (i = 0; i < NrPixels; i++)
      returnArr[i] = (double)buf[i];
    free(buf);
  } else if (dType == 5) { // int32
    int32_t *buf = calloc(NrPixels, sizeof(int32_t));
    if (!buf)
      return FR_ERR_MEMORY;
    elsRead = fread(buf, sizeof(int32_t), NrPixels, f);
    for (i = 0; i < NrPixels; i++)
      returnArr[i] = (double)buf[i];
    free(buf);
  } else {
    return FR_ERR_FORMAT;
  }

  if (elsRead != NrPixels) {
    // It might be EOF or error, but we return read error if strictly incomplete
    // fprintf(stderr, "Warning: Expected %lu elements, read %lu\n", NrPixels,
    // elsRead); return FR_ERR_READ; Logic in original code didn't strictly
    // fail, but we should be careful. For now, allow partial reads if that's
    // standard, but usually it's full frames.
  }
  return FR_SUCCESS;
}

// --- TIFF Reading ---

int ReadTiffFrame(const char *filename, int dType, size_t NrPixels,
                  double *returnArr, int frameIndex) {
  TIFFErrorHandler oldhandler = TIFFSetWarningHandler(NULL);
  TIFF *tif = TIFFOpen(filename, "r");
  TIFFSetWarningHandler(oldhandler);

  if (!tif) {
    printf("Error: Could not open TIFF file %s\n", filename);
    return FR_ERR_OPEN;
  }

  if (frameIndex > 0) {
    if (!TIFFSetDirectory(tif, frameIndex)) {
      TIFFClose(tif);
      return FR_ERR_READ; // Directory not found
    }
  }

  uint32_t imagelength;
  tsize_t scanline;
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength); // Rows
  scanline = TIFFScanlineSize(tif);

  tdata_t buf = _TIFFmalloc(scanline);
  if (!buf) {
    TIFFClose(tif);
    return FR_ERR_MEMORY;
  }

  int rnr;
  size_t pxIndex = 0;
  int i;

  // Check scanline size vs expected width
  // We assume the caller knows the dimensions implicitly via NrPixels?
  // original code calculates index: rnr * (scanline / sizeof_type) + i
  // We strictly follow original logic.

  if (dType == 6) { // int32
    int32_t *datar;
    for (rnr = 0; rnr < imagelength; rnr++) {
      TIFFReadScanline(tif, buf, rnr, 1);
      datar = (int32_t *)buf;
      for (i = 0; i < scanline / sizeof(int32_t); i++) {
        returnArr[rnr * (scanline / sizeof(int32_t)) + i] = (double)datar[i];
      }
    }
  } else if (dType == 7) { // uint8 (bitmap logic)
    uint8_t *datar;
    // Initialize to 0 first? Original code: if (datar[i] == 1) returnArr[...] =
    // 1; The original code does NOT set 0 if datar[i] != 1. It assumes
    // returnArr is pre-cleared or something. But here we are producing values.
    // Let's assume we copy values unless it is strictly a mask?
    // Original code: "if (datar[i] == 1) returnArr[...] = 1;"
    // This implies returnArr should be initialized to 0 by caller or here.
    // We will initialize returnArr to 0 here to be safe for dType 7 usage
    // (usually masks).
    memset(returnArr, 0, NrPixels * sizeof(double));

    for (rnr = 0; rnr < imagelength; rnr++) {
      TIFFReadScanline(tif, buf, rnr, 1);
      datar = (uint8_t *)buf;
      for (i = 0; i < scanline / sizeof(uint8_t); i++) {
        if (datar[i] == 1) {
          returnArr[rnr * (scanline / sizeof(uint8_t)) + i] = 1.0;
        }
      }
    }
  } else if (dType == 9) { // uint16 (NEW)
    uint16_t *datar;
    for (rnr = 0; rnr < imagelength; rnr++) {
      TIFFReadScanline(tif, buf, rnr, 1);
      datar = (uint16_t *)buf;
      for (i = 0; i < scanline / sizeof(uint16_t); i++) {
        returnArr[rnr * (scanline / sizeof(uint16_t)) + i] = (double)datar[i];
      }
    }
  }

  _TIFFfree(buf);
  TIFFClose(tif);
  return FR_SUCCESS;
}

// --- HDF5 Reading ---

// Helper to copy and cast generic buffer to double
static void CopyBufferToDouble(void *buf, hid_t mem_type_id, size_t n_elements,
                               double *out) {
  // This is a naive implementation.
  // HDF5 has H5Dread which can convert types on the fly if we specify
  // H5T_NATIVE_DOUBLE for memory type! So we don't need to manually switch on
  // native type if we ask HDF5 to give us doubles. BUT, the user request says:
  // "it's hard coded that hdf will be uint16... is there a way to read the
  // correct data type?" If we rely on HDF5 conversion, we just ask for DOUBLE
  // and HDF5 library handles the conversion from uint16/uint32/etc on disk.
  // This is the cleanest way.
  // HOWEVER, for `SumHDF5Frames`, we might want to read chunk by chunk.
  // Let's implement providing H5T_NATIVE_DOUBLE to H5Dread.
}

int GetHDF5Dimensions(const char *filename, const char *datasetName,
                      hsize_t *dims) {
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    return FR_ERR_OPEN;

  hid_t dataset_id = H5Dopen2(file_id, datasetName, H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Fclose(file_id);
    return FR_ERR_OPEN; // Dataset not found
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  return FR_SUCCESS;
}

int ReadHDF5Frame(const char *filename, const char *datasetName,
                  size_t NrPixels, double *returnArr, int frameIndex) {
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    return FR_ERR_OPEN;

  hid_t dataset_id = H5Dopen2(file_id, datasetName, H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Fclose(file_id);
    return FR_ERR_OPEN;
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  hsize_t dims[3];
  int ndims = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

  // We assume 3D dataset [Frames, Y, Z]
  hsize_t offset[3] = {(hsize_t)frameIndex, 0, 0};
  hsize_t count[3] = {1, dims[1], dims[2]};

  herr_t status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset,
                                      NULL, count, NULL);
  if (status < 0) {
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return FR_ERR_READ;
  }

  // Memory dataspace
  hid_t memspace_id = H5Screate_simple(3, count, NULL);

  // We read directly into returnArr as DOUBLE. HDF5 handles conversion.
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id,
                   H5P_DEFAULT, returnArr);

  H5Sclose(memspace_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  return (status < 0) ? FR_ERR_READ : FR_SUCCESS;
}

int SumHDF5Frames(const char *filename, const char *datasetName,
                  size_t NrPixels, double *returnArr, int skipFrames) {
  // Initialize returnArr to 0
  memset(returnArr, 0, NrPixels * sizeof(double));

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    return FR_ERR_OPEN;

  hid_t dataset_id = H5Dopen2(file_id, datasetName, H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Fclose(file_id);
    return FR_ERR_OPEN;
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  int ndims = H5Sget_simple_extent_ndims(dataspace_id);
  hsize_t dims[3] = {0, 0, 0};
  H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

  int nFrames, frameRows, frameCols;
  size_t frameSize;

  if (ndims == 2) {
    // 2D dataset: [rows, cols] — single frame, read directly
    nFrames = 1;
    frameRows = dims[0];
    frameCols = dims[1];
    frameSize = (size_t)frameRows * frameCols;

    double *frameBuf = malloc(frameSize * sizeof(double));
    if (!frameBuf) {
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Fclose(file_id);
      return FR_ERR_MEMORY;
    }

    // Read entire 2D dataset
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            frameBuf);

    size_t copySize =
        (frameSize < NrPixels || NrPixels == 0) ? frameSize : NrPixels;
    int p;
    for (p = 0; p < copySize; p++) {
      returnArr[p] = frameBuf[p];
    }

    free(frameBuf);
  } else {
    // 3D dataset: [nFrames, rows, cols] — sum/average frames
    nFrames = dims[0];
    frameRows = dims[1];
    frameCols = dims[2];
    frameSize = (size_t)frameRows * frameCols;

    double *frameBuf = malloc(frameSize * sizeof(double));
    if (!frameBuf) {
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Fclose(file_id);
      return FR_ERR_MEMORY;
    }

    hid_t memspace_id =
        H5Screate_simple(3, (hsize_t[]){1, dims[1], dims[2]}, NULL);

    int i, p;
    for (i = skipFrames; i < nFrames; i++) {
      hsize_t offset[3] = {(hsize_t)i, 0, 0};
      hsize_t count[3] = {1, dims[1], dims[2]};

      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
                          NULL);

      H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id,
              H5P_DEFAULT, frameBuf);

      for (p = 0; p < frameSize; p++) {
        returnArr[p] += frameBuf[p];
      }
    }

    // Compute average
    int validFrames = nFrames - skipFrames;
    if (validFrames > 1) {
      for (p = 0; p < frameSize; p++) {
        returnArr[p] /= validFrames;
      }
    }

    free(frameBuf);
    H5Sclose(memspace_id);
  }

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  return FR_SUCCESS;
}

// --- CBF Reading (x-CBF_BYTE_OFFSET decompression) ---

// CBF binary section starter: CIF magic bytes 0x0C 0x1A 0x04 0xD5
static const unsigned char CBF_STARTER[] = {0x0C, 0x1A, 0x04, 0xD5};

// Find a byte pattern in a buffer
static const unsigned char *find_bytes(const unsigned char *haystack,
                                       size_t hlen, const unsigned char *needle,
                                       size_t nlen) {
  if (nlen > hlen)
    return NULL;
  for (size_t i = 0; i <= hlen - nlen; i++) {
    if (memcmp(haystack + i, needle, nlen) == 0)
      return haystack + i;
  }
  return NULL;
}

// Parse an integer value from a header line like "X-Binary-Something: 12345"
static int parse_header_int(const char *header, const char *key) {
  const char *pos = strstr(header, key);
  if (!pos)
    return 0;
  pos += strlen(key);
  while (*pos == ' ' || *pos == ':' || *pos == '\t')
    pos++;
  return atoi(pos);
}

int ReadCBFFrame(const char *filename, size_t NrPixels, double *returnArr,
                 int *outNcols, int *outNrows) {
  // Read entire file into memory
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Error: Could not open CBF file %s\n", filename);
    return FR_ERR_OPEN;
  }
  fseek(f, 0L, SEEK_END);
  size_t fileSize = ftell(f);
  rewind(f);

  unsigned char *fileData = malloc(fileSize);
  if (!fileData) {
    fclose(f);
    return FR_ERR_MEMORY;
  }
  if (fread(fileData, 1, fileSize, f) != fileSize) {
    free(fileData);
    fclose(f);
    return FR_ERR_READ;
  }
  fclose(f);

  // Find the binary section starter marker
  const unsigned char *starter =
      find_bytes(fileData, fileSize, CBF_STARTER, sizeof(CBF_STARTER));
  if (!starter) {
    fprintf(stderr, "Error: No CBF binary section marker found in %s\n",
            filename);
    free(fileData);
    return FR_ERR_FORMAT;
  }

  // Parse the binary sub-header (ASCII text between "--CIF-BINARY..." and
  // starter) We search backward for the MIME boundary
  const char *boundary = "--CIF-BINARY-FORMAT-SECTION--";
  const unsigned char *bnd =
      find_bytes(fileData, (size_t)(starter - fileData),
                 (const unsigned char *)boundary, strlen(boundary));
  int ncols = 0, nrows = 0, n_elements = 0, binary_size = 0;
  if (bnd) {
    // Header is from boundary to starter, null-terminate a copy
    size_t hdr_len = (size_t)(starter - bnd);
    char *hdr = malloc(hdr_len + 1);
    memcpy(hdr, bnd, hdr_len);
    hdr[hdr_len] = '\0';

    ncols = parse_header_int(hdr, "X-Binary-Size-Fastest-Dimension");
    nrows = parse_header_int(hdr, "X-Binary-Size-Second-Dimension");
    n_elements = parse_header_int(hdr, "X-Binary-Number-of-Elements");
    binary_size = parse_header_int(hdr, "X-Binary-Size:");

    free(hdr);
  }

  if (ncols <= 0 || nrows <= 0 || n_elements <= 0) {
    fprintf(stderr,
            "Error: Could not parse CBF header dimensions in %s "
            "(ncols=%d, nrows=%d, n_elements=%d)\n",
            filename, ncols, nrows, n_elements);
    free(fileData);
    return FR_ERR_FORMAT;
  }

  if (outNcols)
    *outNcols = nrows; // transposed: original rows become output cols
  if (outNrows)
    *outNrows = ncols; // transposed: original cols become output rows

  if (NrPixels > 0 && (size_t)n_elements != NrPixels) {
    fprintf(stderr, "Warning: CBF element count %d != expected %zu in %s\n",
            n_elements, NrPixels, filename);
  }

  // Data starts right after the 4-byte starter
  const unsigned char *data = starter + sizeof(CBF_STARTER);
  size_t data_avail = fileSize - (size_t)(data - fileData);
  if (binary_size > 0 && (size_t)binary_size < data_avail)
    data_avail = (size_t)binary_size;

  // x-CBF_BYTE_OFFSET decompression
  int64_t prev = 0;
  size_t pos = 0;
  for (int i = 0; i < n_elements; i++) {
    if (pos >= data_avail) {
      fprintf(stderr, "Error: CBF data truncated at pixel %d/%d in %s\n", i,
              n_elements, filename);
      free(fileData);
      return FR_ERR_READ;
    }

    int8_t b = (int8_t)data[pos++];
    int64_t delta;

    if (b != -128) {
      delta = b;
    } else {
      // Read 16-bit value (little-endian)
      if (pos + 2 > data_avail) {
        free(fileData);
        return FR_ERR_READ;
      }
      int16_t s =
          (int16_t)((uint16_t)data[pos] | ((uint16_t)data[pos + 1] << 8));
      pos += 2;

      if (s != -32768) {
        delta = s;
      } else {
        // Read 32-bit value (little-endian)
        if (pos + 4 > data_avail) {
          free(fileData);
          return FR_ERR_READ;
        }
        int32_t l =
            (int32_t)((uint32_t)data[pos] | ((uint32_t)data[pos + 1] << 8) |
                      ((uint32_t)data[pos + 2] << 16) |
                      ((uint32_t)data[pos + 3] << 24));
        pos += 4;
        delta = l;
      }
    }

    prev += delta;

    // Transpose: original (row, col) in row-major → (col, row) in output
    // Plus flip both axes: .T[::-1, ::-1]
    int row = i / ncols;
    int col = i % ncols;
    returnArr[(ncols - 1 - col) * nrows + (nrows - 1 - row)] = (double)prev;
  }

  printf(
      "ReadCBFFrame: %s — %dx%d (%d elements, transposed+flipped to %dx%d)\n",
      filename, ncols, nrows, n_elements, nrows, ncols);

  free(fileData);
  return FR_SUCCESS;
}
