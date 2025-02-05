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
        printf("Usage: %s FolderName FileStem Extension(with.) nScans nScansMerge nFramesMerge nFrames nPixelsY nPixelsZ nCPUs.\n",argv[0]);
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
        outArr = calloc(nFramesOut*nPxY*nPxZ,sizeof(*outArr));
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
            while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
                if (strstr(finfo->name,"exchange/data/0.0.0")!=NULL){
                    dataLoc = count;
                }
            }
            printf("Reading compressed image data.\n");
            fflush(stdout);
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
            printf("Data read completely. Total size: %zu bytes, total time taken: %lf seconds.\n",cntr,t_0-t_1);

        }
        free(outArr);
    }
}