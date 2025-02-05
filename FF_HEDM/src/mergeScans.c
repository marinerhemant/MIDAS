//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  mergeScans.c
//  code to do random merging in omega or scan directions.
//
//
//  Created by Hemant Sharma on 2024/02/27.
//
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <nlopt.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>
#include <libgen.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/resource.h>
#include <blosc2.h>
#include <stdlib.h> 
#include <zip.h> 

int main(int argc, char *argv[]){
    double start_time = omp_get_wtime();
    if (argc < 11){
        printf("Usage: %s FolderName FileStem Extension(with.) nScans nScansMerge nFramesMerge nPixelsY nPixelsZ nFrames nCPUs (optional)omegaFileName (optional)omegaStep\n",argv[0]);
        return 1;
    }
    char *folder = argv[1];
    char *fileStem = argv[2];
    char *ext = argv[3];
    int nScans = atoi(argv[4]);
    int nScansMerge = atoi(argv[5]);
    int nFramesMerge = atoi(argv[6]);
    int nPxY = atoi(argv[7]);
    int nPxZ = atoi(argv[8]);
    int nFrames = atoi(argv[9]);
    int nCPUs = atoi(argv[10]);
    double *omegas, omegaStep=0;
    omegas = calloc(nScans,sizeof(*omegas));
    char *omegaFile;
    int omegaNr=0;
    if (argc==13){
        printf("Omega file was provided, it will rotate scans to start at -180 degrees.\n");
        omegaFile = argv[11];
        omegaStep = atof(argv[12]);
        FILE *omegaF = fopen(omegaFile,"r");
        char aline[1000];
        for (omegaNr=0;omegaNr<nScans;omegaNr++){
            fgets(aline,1000,omegaF);
            sscanf(aline,"%lf",&omegas[omegaNr]);
        }
    }
    blosc2_init();

    int nJobs = nScans / nScansMerge;
    int nFramesOut = nFrames / nFramesMerge;
    int jobNr;
    #pragma omp parallel for num_threads(nCPUs) private(jobNr) schedule(dynamic)
    for (jobNr=0;jobNr<nJobs;jobNr++){
        int startScanNr = jobNr*nScansMerge + 1;
        int endScanNr = startScanNr + nScansMerge;
        int fileNr;
        double *outArr;
        size_t szarr = nFramesOut;
        szarr *= nPxY;
        szarr *= nPxZ;
        outArr = calloc(szarr,sizeof(*outArr));
        for (fileNr=startScanNr; fileNr<endScanNr; fileNr++){
            char DataFN[2048];
            sprintf(DataFN,"%s/%d/%s_%06d%s",folder,fileNr,fileStem,fileNr,ext);
            int errorp = 0;
            zip_t* arch = NULL;
            arch = zip_open(DataFN,0,&errorp);
            if (errorp!=NULL){
                printf("Input was not a zarr zip.\n");
                continue;
            }
            struct zip_stat* finfo = NULL;
            finfo = calloc(16384, sizeof(int));
            zip_stat_init(finfo);
            zip_file_t* fd = NULL;
            int count = 0;
            char* s = NULL;
            char* arr = NULL;
            int dataLoc;
            printf("%s\n",DataFN);
            while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
                if (strstr(finfo->name,"exchange/data/0.0.0")!=NULL){
                    dataLoc = count;
                    break;
                }
                count ++;
            }
            double t_1 = omp_get_wtime();
            size_t *sizeArr;
            sizeArr = calloc(nFrames*2,sizeof(*sizeArr)); // Number StartLoc
            size_t cntr=0;
            int iter;
            for (iter=0;iter<nFrames;iter++){
                zip_stat_index(arch,dataLoc+iter,0,finfo);
                sizeArr[iter*2+0] = finfo->size;
                sizeArr[iter*2+1] = cntr;
                cntr += finfo->size;
            }
            // allocate arr
            char * allData;
            allData = calloc(cntr+1,sizeof(*allData));
            for (iter=0;iter<nFrames;iter++){
                zip_file_t *fLoc = NULL;
                fLoc = zip_fopen_index(arch,dataLoc+iter,0);
                zip_fread(fLoc,&allData[sizeArr[iter*2+1]],sizeArr[iter*2+0]);
                zip_fclose(fLoc);
            }
            double t_0 = omp_get_wtime();
            int frameNr;
            char *rawImage;
            rawImage = malloc(nPxY*nPxZ*2*sizeof(*rawImage));
            uint16_t *ImageAsym;
            ImageAsym = malloc(nPxY*nPxZ*sizeof(*ImageAsym));
            int32_t dsz = nPxY*nPxZ*2;
            uint16_t maxInt;
            for (frameNr=0;frameNr<nFrames;frameNr++){
                int frameToPut;
                if (omegaNr>0){
                    double thisOmega = omegas[fileNr-1];
                    double signTO;
                    if (thisOmega!=0) signTO = thisOmega/fabs(thisOmega);
                    double delOmega = signTO * (fmod(fabs(thisOmega),360.0));
                    delOmega = delOmega *(fmod(fabs(delOmega), 360.0))/fabs(delOmega);
                    double currentOmega = delOmega + (frameNr)*omegaStep;
                    if (currentOmega - 180.0 > 0.00001) currentOmega -= 360.0;
                    double recalcFrameNr = (180.0 + currentOmega)/(omegaStep*nFramesMerge);
                    frameToPut = (int)floor(recalcFrameNr);
                    if (frameToPut >= nFrames/nFramesMerge) frameNr = 0;
                } else frameToPut = frameNr / nFramesMerge;
                dsz = blosc1_decompress(&allData[sizeArr[frameNr*2+1]],rawImage,dsz);
                memcpy(ImageAsym,rawImage,dsz);
                size_t offset;
                // if (frameNr%nFramesMerge == 0) printf("FrameNr: %d\n",frameNr);
                for (cntr=0;cntr<dsz/2;cntr++){
                    offset = frameToPut;
                    offset *= nPxY;
                    offset *= nPxZ;
                    offset += cntr;
                    outArr[offset] += (double)ImageAsym[cntr];
                }
                
            }
            t_1 = omp_get_wtime();
            printf("Frames processed, time taken: %lf seconds.\n",t_1-t_0);
        }
        free(outArr);
    }
}