// ZarrReader.c
// Implementation of shared Zarr (ZIP + blosc) chunk-reading utilities.
// See ZarrReader.h for API documentation.

#include "ZarrReader.h"

int ReadZarrChunk(zip_t *arch, int entryIndex, void *dest, size_t destSize) {
  struct zip_stat finfo;
  zip_stat_init(&finfo);

  if (zip_stat_index(arch, entryIndex, 0, &finfo) != 0) {
    fprintf(stderr, "ZarrReader ERROR: zip_stat_index failed for index %d\n",
            entryIndex);
    return ZR_ERR_STAT;
  }

  char *compressed = (char *)calloc(finfo.size + 1, sizeof(char));
  if (compressed == NULL) {
    fprintf(stderr,
            "ZarrReader ERROR: Failed to allocate %llu bytes for compressed "
            "buffer (index %d)\n",
            (unsigned long long)finfo.size, entryIndex);
    return ZR_ERR_ALLOC;
  }

  zip_file_t *fd = zip_fopen_index(arch, entryIndex, 0);
  if (fd == NULL) {
    fprintf(stderr, "ZarrReader ERROR: zip_fopen_index failed for index %d\n",
            entryIndex);
    free(compressed);
    return ZR_ERR_OPEN;
  }

  zip_int64_t bytesRead = zip_fread(fd, compressed, finfo.size);
  if (bytesRead < 0 || (zip_uint64_t)bytesRead != finfo.size) {
    fprintf(stderr,
            "ZarrReader ERROR: zip_fread short read for index %d "
            "(expected %llu, got %lld)\n",
            entryIndex, (unsigned long long)finfo.size, (long long)bytesRead);
    free(compressed);
    zip_fclose(fd);
    return ZR_ERR_READ;
  }

  int decompSize = blosc1_decompress(compressed, dest, (int)destSize);
  if (decompSize < 0) {
    fprintf(stderr,
            "ZarrReader ERROR: blosc1_decompress failed for index %d "
            "(error code: %d)\n",
            entryIndex, decompSize);
    free(compressed);
    zip_fclose(fd);
    return ZR_ERR_DECOMP;
  }

  free(compressed);
  zip_fclose(fd);
  return decompSize;
}

int ReadZarrRaw(zip_t *arch, int entryIndex, char **outBuf, size_t *outSize) {
  struct zip_stat finfo;
  zip_stat_init(&finfo);

  if (zip_stat_index(arch, entryIndex, 0, &finfo) != 0) {
    fprintf(stderr, "ZarrReader ERROR: zip_stat_index failed for index %d\n",
            entryIndex);
    return ZR_ERR_STAT;
  }

  char *buf = (char *)calloc(finfo.size + 1, sizeof(char));
  if (buf == NULL) {
    fprintf(stderr,
            "ZarrReader ERROR: Failed to allocate %llu bytes for raw buffer "
            "(index %d)\n",
            (unsigned long long)finfo.size, entryIndex);
    return ZR_ERR_ALLOC;
  }

  zip_file_t *fd = zip_fopen_index(arch, entryIndex, 0);
  if (fd == NULL) {
    fprintf(stderr, "ZarrReader ERROR: zip_fopen_index failed for index %d\n",
            entryIndex);
    free(buf);
    return ZR_ERR_OPEN;
  }

  zip_int64_t bytesRead = zip_fread(fd, buf, finfo.size);
  if (bytesRead < 0 || (zip_uint64_t)bytesRead != finfo.size) {
    fprintf(stderr,
            "ZarrReader ERROR: zip_fread short read for index %d "
            "(expected %llu, got %lld)\n",
            entryIndex, (unsigned long long)finfo.size, (long long)bytesRead);
    free(buf);
    zip_fclose(fd);
    return ZR_ERR_READ;
  }

  buf[finfo.size] = '\0'; // NUL-terminate for JSON parsing convenience
  zip_fclose(fd);

  *outBuf = buf;
  *outSize = (size_t)finfo.size;
  return ZR_SUCCESS;
}

int ReadZarrString(zip_t *arch, int entryIndex, char **outStr, size_t maxLen) {
  char *buf = (char *)malloc(maxLen + 1);
  if (buf == NULL) {
    fprintf(stderr,
            "ZarrReader ERROR: Failed to allocate %zu bytes for string buffer "
            "(index %d)\n",
            maxLen + 1, entryIndex);
    return ZR_ERR_ALLOC;
  }

  int decompSize = ReadZarrChunk(arch, entryIndex, buf, maxLen);
  if (decompSize < 0) {
    free(buf);
    return decompSize; // propagate error
  }

  buf[decompSize] = '\0';
  *outStr = buf;
  return decompSize;
}
