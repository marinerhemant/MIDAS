#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <libgen.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_N_SOLUTIONS_PER_VOX 1000000
// We will read the files for each voxel in parallel, then compute unique orientations for each voxel, then spit out the unique IDs for that voxel.
int
main(int argc, char *argv[])
{
	double start_time = omp_get_wtime();
	printf("\n\n\t\tFinding Multiple Solutions in PF-HEDM.\n\n");
	int returncode;
	if (argc != 6) {
		printf("Supply folder, spaceGroup, maxAng, NumberScans and nCPUs as arguments: ie %s sgNum maxAngle folderName nScans nCPUs\n\n"
            "The indexing results must be in folderName/Output\n", argv[0]);
		return 1;
	}
	char folderName[2048]; 
    sprintf(folderName,"%s/Output/",argv[1]);
    int sgNr = atoi(argv[2]);
    double maxAng = atof(argv[3]);
    int nScans = atoi(argv[4]);
    int numProcs = atoi(argv[5]);
    int voxNr;
    # pragma omp parallel for num_threads(numProcs) private(voxNr) schedule(dynamic)
    for (voxNr=0;voxNr<nScans*nScans;voxNr++){
        FILE *valsF, *keyF;
        char valsFN[2048], keyFN[2048];
        sprintf(valsFN,"%s/IndexBest_voxNr_%0*d.bin",folderName,6,voxNr);
        sprintf(keyFN,"%s/IndexKey_voxNr_%0*d.txt",folderName,6,voxNr);
        valsF = fopen(valsFN,"rb");
        keyF = fopen(keyFN,"r");
        size_t *keys;
        keys = calloc(MAX_N_SOLUTIONS_PER_VOX*4,sizeof(*keys));
        char aline[2048];
        int nIDs = 0;
        while(fgets(aline,2048,keyF)!=NULL){
            sscanf(aline,"%zu %zu %zu %zu",&keys[nIDs*4+0],&keys[nIDs*4+1],&keys[nIDs*4+2],&keys[nIDs*4+3]);
            nIDs++;
        }
        realloc(keys,nIDs*4*sizeof(*keys));
        fclose(keyF);
        double *OMArr, *tmpArr;
        OMArr = calloc(nIDs*9,sizeof(double));
        tmpArr = calloc(nIDs*16,sizeof(double));
        fread(tmpArr,nIDs*16*sizeof(double),1,valsF);
        fclose(valsF);
        int i,j,k;
        for (i=0;i<nIDs;i++) for (j=0;j<9;j++) OMArr[i*9+j] = tmpArr[i*16+2+j];
        free(tmpArr);
        bool *markArr;
        markArr = malloc(nIDs*sizeof(*markArr));
        for (i=0;i<nIDs;i++) markArr[i] = false;
        double OMThis[9], OMInside[9], Quat1[4],Quat2[4], Angle, Axis[3],ang;
        size_t *uniqueArr;
        uniqueArr = calloc(nIDs*4,sizeof(*uniqueArr));
        int nUniques = 0;
        for (i=0;i<nIDs;i++){
            if (markArr[i]==true) continue;
            for (j=0;j<9;j++) OMThis[j] = OMArr[i*9+j];
            OrientMat2Quat(OMThis,Quat1);
            for (j=i+1;j<nIDs;j++){
                if (markArr[j]==true) continue;
                for (k=0;k<9;k++) OMInside[k] = OMArr[j*9+k];
                OrientMat2Quat(OMInside,Quat2);
                Angle = GetMisOrientation(Quat1,Quat2,Axis,&ang,sgNr);
                if (ang<maxAng) markArr[j] = true;
            }
            for (j=0;j<4;j++) uniqueArr[nUniques*4+j] = keys[i*4+j];
            nUniques++;
        }
        char outKeyFN[2048];
        FILE *outKeyF;
        sprintf(outKeyFN,"%s/UniqueIndexKey_voxNr_%0*d.txt",folderName,6,voxNr);
        outKeyF = fopen(outKeyFN,"w");
        for (i=0;i<nUniques;i++) fprintf(outKeyF,"%zu %zu %zu %zu\n",uniqueArr[i*4+0],uniqueArr[i*4+1],uniqueArr[i*4+2],uniqueArr[i*4+3]);
        fclose(outKeyF);
        free(markArr);
        free(keys);
        free(OMArr);
        free(uniqueArr);
    }
    // Now generate a spotsToIndex.csv with all the info from all the voxels
    char *originalFolder = argv[1], outFN[2048];
    sprintf(outFN,"%s/SpotsToIndex.csv");
    FILE *outF;
    outF = fopen(outFN,"w");
    for (voxNr=0;voxNr<nScans*nScans;voxNr++){
        char outKeyFN[2048],aline[2048];
        FILE *outKeyF;
        sprintf(outKeyFN,"%s/UniqueIndexKey_voxNr_%0*d.txt",folderName,6,voxNr);
        outKeyF = fopen(outKeyFN,"r");
        while (fgets(aline,2048,outKeyF)!=NULL){
            fprintf(outF,"%d %s",voxNr,aline);
        }
    }
    fclose(outF);
}