// gcc src/read_one_frame.c -o bin/read_one_frame -L${HOME}/.MIDAS/BLOSC/lib -I${HOME}/.MIDAS/BLOSC/include -lblosc2
// install_name_tool -id '${HOME}/.MIDAS/BLOSC/lib/libblosc2.2.dylib' ${HOME}/.MIDAS/BLOSC/lib/libblosc2.2.dylib
#include <stdio.h>
#include <blosc2.h>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: read_one_frame zarr_folder_with_data frameNr output_file\n");
    return -1;
  }
  blosc2_init();
  static char* data;
  // Read zarr config
  char *folder = argv[1];
  char *s, fn[2048], fnJSON[2048];
  sprintf(fnJSON,"%s/.zarray",folder);
  FILE *fjson = fopen(fnJSON,"rb");
  fseek(fjson,0L,SEEK_END);
  size_t szJSON = ftell(fjson);
  fseek(fjson,0L,SEEK_SET);
  s = (char *)malloc(szJSON);
  fread(s,szJSON,1,fjson);
  fclose(fjson);
  char *ptr = strstr(s,"shape");
  int nFrames,NrPixelsZ,NrPixelsY,bytesPerPx;
  if (ptr!=NULL){
    if (3 == sscanf(ptr, "%*[^0123456789]%d%*[^0123456789]%d%*[^0123456789]%d", &nFrames, &NrPixelsZ, &NrPixelsY)){
            printf("nFrames: %d nrPixelsZ: %d nrPixelsY: %d\n", nFrames, NrPixelsZ, NrPixelsY);
        } else return 1;
  } else return 1;
  ptr = strstr(s,"dtype");
  if (ptr!=NULL){
    char *ptr2 = ptr + 9;
    if (strncmp(ptr2,"<u2",3) == 0){
        bytesPerPx=2;
    } else return 1;
  } else return 1;

  int32_t dsize = bytesPerPx*NrPixelsZ*NrPixelsY;
  int frameNr = atoi(argv[2]);
  if (frameNr > nFrames-1) return 1;
  sprintf(fn,"%s/%d.0.0",folder,frameNr);
  int64_t nbytes, cbytes;
  blosc_timestamp_t last, current;
  double ttotal;

  printf("Blosc version info: %s (%s)\n",
         BLOSC2_VERSION_STRING, BLOSC2_VERSION_DATE);
  
  char *arr;
  FILE *f = fopen(fn,"rb");
  fseek(f,0L,SEEK_END);
  size_t sz = ftell(f);
  fseek(f,0L,SEEK_SET);
  arr = (char *)malloc(sz);
  fread(arr,sz,1,f);
  fclose(f);

  data = (char*)malloc((size_t)dsize);
  blosc_set_timestamp(&last);
  dsize = blosc1_decompress(arr,data,dsize);
  FILE* foutput = fopen(argv[3], "wb");
  fwrite(data, dsize, 1, foutput);
  fclose(foutput);
  // t = *(uint16_t *)&data[i*2]; //typecast to int16 if wanted, but not required if writing to a file.

  /* Free resources */
  free(data);
  blosc2_destroy();
  return 0;
}