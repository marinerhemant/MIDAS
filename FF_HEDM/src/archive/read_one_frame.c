// gcc src/read_one_frame.c -o bin/read_one_frame -L${HOME}/.MIDAS/BLOSC/lib
// -I${HOME}/.MIDAS/BLOSC/include -lblosc2 install_name_tool -id
// '${HOME}/.MIDAS/BLOSC/lib/libblosc2.2.dylib'
// ${HOME}/.MIDAS/BLOSC/lib/libblosc2.2.dylib install_name_tool -id
// '${HOME}/.MIDAS/LIBZIP/lib/libzip.5.dylib'
// ${HOME}/.MIDAS/LIBZIP/lib/libzip.5.dylib
#include <blosc2.h>
#include <stdio.h>
#include <stdlib.h>
#include <zip.h>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: read_one_frame zarr_zip_file frameNr output_file\n");
    return -1;
  }
  blosc2_init();
  char *data = NULL;
  // Read zarr config
  char *zipFN = argv[1];
  char *folder;
  int errorp = 0;
  zip_t *arch = NULL;
  arch = zip_open(zipFN, 0, &errorp);
  struct zip_stat *finfo = NULL;
  finfo = calloc(2048, sizeof(int));
  zip_stat_init(finfo);
  zip_file_t *fd = NULL;
  char *txt = NULL;
  int count = 0;
  char *s = NULL;
  char fnStr[4096];
  int frameNr = atoi(argv[2]);
  sprintf(fnStr, "exchange/data/%d.0.0", frameNr);
  char *arr = NULL;
  while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
    if (strstr(finfo->name, "exchange/data/.zarray") != NULL) {
      s = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, s, finfo->size);
      zip_fclose(fd);
    }
    if (strstr(finfo->name, fnStr) != NULL) {
      arr = calloc(finfo->size + 1, sizeof(char));
      fd = zip_fopen_index(arch, count, 0);
      zip_fread(fd, arr, finfo->size);
      zip_fclose(fd);
    }
    count++;
  }
  if (arr == NULL)
    return 1;
  char *ptr = strstr(s, "shape");
  int nFrames, NrPixelsZ, NrPixelsY, bytesPerPx;
  if (ptr != NULL) {
    char *ptrt = strstr(ptr, "[");
    char *ptr2 = strstr(ptrt, "]");
    int loc = (int)(ptr2 - ptrt);
    char ptr3[2048];
    strncpy(ptr3, ptrt, loc + 1);
    if (3 == sscanf(ptr3, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d",
                    &nFrames, &NrPixelsZ, &NrPixelsY)) {
      printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d\n", nFrames, NrPixelsZ,
             NrPixelsY);
    } else
      return 1;
  } else
    return 1;
  ptr = strstr(s, "dtype");
  if (ptr != NULL) {
    char *ptrt = strstr(ptr, ":");
    char *ptr2 = strstr(ptrt, ",");
    int loc = (int)(ptr2 - ptrt);
    char ptr3[2048];
    strncpy(ptr3, ptrt + 3, loc - 4);
    if (strncmp(ptr3, "<u2", 3) == 0) {
      bytesPerPx = 2;
    } else
      return 1;
  } else
    return 1;
  int32_t dsize = bytesPerPx * NrPixelsZ * NrPixelsY;
  data = (char *)malloc((size_t)dsize);
  dsize = blosc1_decompress(arr, data, dsize);
  FILE *foutput = fopen(argv[3], "wb");
  fwrite(data, dsize, 1, foutput);
  fclose(foutput);
  // t = *(uint16_t *)&data[i*2]; //typecast to int16 if wanted, but not
  // required if writing to a file.

  /* Free resources */
  free(data);
  blosc2_destroy();
  return 0;
}