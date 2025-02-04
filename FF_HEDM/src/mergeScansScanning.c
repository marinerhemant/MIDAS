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

#define MAX_N_SPOTS 100000000

int
main(int argc, char *argv[])
{
	double start_time = omp_get_wtime();
	printf("\n\n\tMerging Scans in scanning in PF-HEDM.\n\n");
	int returncode;
	if (argc != 6) {
		printf("Supply %s nScans, nMerges, pxTol, omeTol, nCPUs as arguments.\n",argv[0]);
		return 1;
	}
    int nScans = atoi(argv[1]);
    int nMerges = atoi(argv[2]);
    double tolPx = atof(argv[3]);
    double tolOme = atof(argv[4]);
    int numProcs = atoi(argv[5]);
    int nFinScans = (int)floor((double)nScans / (double)nMerges);
    double *positions, *positionsNew;
    positions = calloc(nScans,sizeof(*positions));
    positionsNew = calloc(nFinScans,sizeof(*positionsNew));
    FILE *posF = fopen("original_positions.csv","r");
    int iter;
    char aline[2048];
    for (iter=0;iter<nScans;iter++){
        fgets(aline,2048,posF);
        sscanf(aline,"%lf",&positions[iter]);
    }
    fclose(posF);

    int finScanNr;
    # pragma omp parallel for num_threads(numProcs) private(finScanNr) schedule(dynamic)
    for (finScanNr=0;finScanNr<nFinScans;finScanNr++){
        int startScanNr = finScanNr*nMerges;
        double thisPosition = positions[startScanNr];
        double *thisSpots, *allSpots;
        thisSpots = calloc(MAX_N_SPOTS*14,sizeof(*thisSpots));
        allSpots = calloc(MAX_N_SPOTS*14,sizeof(*allSpots));
        // Read the first fileNr
        char thisFN[2048], thisLine[2048], headThis[2048];
        sprintf(thisFN,"original_InputAllExtraInfoFittingAll%d.csv",startScanNr);
        FILE *thisF;
        thisF = fopen(thisFN,"r");
        fgets(thisLine,2048,thisF);
        sprintf(headThis,"%s",thisLine);
        size_t nAll=0;
        while (fgets(thisLine,2048,thisF)!=NULL){
            sscanf(thisLine,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                &allSpots[nAll*14+0],&allSpots[nAll*14+1],&allSpots[nAll*14+2],
                &allSpots[nAll*14+3],&allSpots[nAll*14+4],&allSpots[nAll*14+5],
                &allSpots[nAll*14+6],&allSpots[nAll*14+7],&allSpots[nAll*14+8],
                &allSpots[nAll*14+9],&allSpots[nAll*14+10],&allSpots[nAll*14+11],
                &allSpots[nAll*14+12],&allSpots[nAll*14+13]);
            if (allSpots[nAll*14+3]<0.01) continue;
            nAll++;
        }
        fclose(thisF);
        int scanNr, thisScanNr;
        int i,j,k,l,found;
        double origWeight, newWeight;
        int nSpotsLastScan;
        int *lastScansSpots, *thisScansSpots;
        lastScansSpots = calloc(MAX_N_SPOTS,sizeof(*lastScansSpots));
        thisScansSpots = calloc(MAX_N_SPOTS,sizeof(*thisScansSpots));
        for (scanNr=1;scanNr<nMerges;scanNr++){
            printf("ScanNr: %d, nAll: %zu\n",scanNr,nAll);
            thisScanNr = startScanNr + scanNr;
            thisPosition += positions[thisScanNr];
            int nThis = 0;
            sprintf(thisFN,"original_InputAllExtraInfoFittingAll%d.csv",thisScanNr);
            thisF = fopen(thisFN,"r");
            fgets(thisLine,2048,thisF);
            while (fgets(thisLine,2048,thisF)!=NULL){
                sscanf(thisLine,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &thisSpots[nThis*14+0],&thisSpots[nThis*14+1],&thisSpots[nThis*14+2],
                    &thisSpots[nThis*14+3],&thisSpots[nThis*14+4],&thisSpots[nThis*14+5],
                    &thisSpots[nThis*14+6],&thisSpots[nThis*14+7],&thisSpots[nThis*14+8],
                    &thisSpots[nThis*14+9],&thisSpots[nThis*14+10],&thisSpots[nThis*14+11],
                    &thisSpots[nThis*14+12],&thisSpots[nThis*14+13]);
                if (thisSpots[nThis*14+3]<0.01) continue;
                nThis++;
            }
            fclose(thisF);
            nSpotsLastScan = nThis;
            // Go through each spot in both files, if found a match add weighted values, if not, add to the original array
            for (i=0;i<nThis;i++){
                found = 0;
                for (l=0;l<nSpotsLastScan;l++){
                    j = lastScansSpots[l];
                    if (fabs(thisSpots[i*14+5] - allSpots[j*14+5])<0.01){
                        if (fabs(thisSpots[i*14+0] - allSpots[j*14+0])<tolPx){
                            if (fabs(thisSpots[i*14+1] - allSpots[j*14+1])<tolPx){
                                if (fabs(thisSpots[i*14+2] - allSpots[j*14+2])<tolOme){
                                    found = 1;
                                    origWeight = allSpots[j*14+3];
                                    newWeight = thisSpots[i*14+3];
                                    thisScansSpots[i] = j;
                                    for (k=0;k<14;k++) {
                                        allSpots[j*14+k] = (allSpots[j*14+k]*origWeight + thisSpots[i*14+k]*newWeight)/(origWeight+newWeight);
                                    }
                                }
                            }
                        }
                    }
                }
                if (found == 0){
                    thisScansSpots[i] = nAll;
                    for (j=0;j<14;j++) allSpots[nAll*14+j] = thisSpots[i*14+j];
                    nAll++;
                }
            }
            for (i=0;i<nSpotsLastScan;i++) lastScansSpots[i] = thisScansSpots[i];
        }
        for (i=0;i<nAll;i++) allSpots[i*14+4] = i+1;
        sprintf(thisFN,"InputAllExtraInfoFittingAll%d.csv",finScanNr);
        thisF = fopen(thisFN,"w");
        fprintf(thisF,"%s",headThis);
        for (i=0;i<nAll;i++){
            for (j=0;j<14;j++){
                fprintf(thisF,"%lf ",allSpots[i*14+j]);
            }
            fprintf(thisF,"\n");
        }
        thisPosition /= nMerges;
        positionsNew[finScanNr] = thisPosition;
    }
    posF = fopen("positions.csv","w");
    for (iter=0;iter<nFinScans;iter++){
        fprintf(posF,"%lf\n",positionsNew[iter]);
    }
    fclose(posF);
}
