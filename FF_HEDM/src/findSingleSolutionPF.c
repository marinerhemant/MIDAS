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
#define MAX_N_SPOTS_PER_GRAIN 5000
#define MAX_N_SPOTS_TOTAL 100000000

struct InputData{
	int mergedID;
	int originalID;
	int scanNr;
	int grainNr;
    int spotNr;
};

static int cmpfunc (const void * a, const void *b){
	struct InputData *ia = (struct InputData *)a;
	struct InputData *ib = (struct InputData *)b;
    if (ia->scanNr > ib->scanNr) return 1;
    if (ia->scanNr < ib->scanNr) return -1;
	return (int)(ia->originalID - ib->originalID);
}

int
main(int argc, char *argv[])
{
	double start_time = omp_get_wtime();
	printf("\n\n\t\tFinding Single Solution in PF-HEDM.\n\n");
	int returncode;
	if (argc != 8) {
		printf("Supply spaceGroup, maxAng, FolderName, NumberScans, nCPUs, tolOme, tolEta as arguments: ie %s sgNum maxAngle folderName nScans nCPUs tolOme tolEta\n"
                "\nThe indexing results need to be in folderName/Output\n", argv[0]);
		return 1;
	}
	char folderName[2048];
    sprintf(folderName,"%s/Output/",argv[1]);
    int sgNr = atoi(argv[2]);
    double maxAng = atof(argv[3]);
    int nScans = atoi(argv[4]);
    int numProcs = atoi(argv[5]);
    double tolOme = atof(argv[6]);
    double tolEta = atof(argv[7]);
    size_t *allKeyArr, *uniqueKeyArr;
    allKeyArr = calloc(nScans*nScans*4,sizeof(*allKeyArr));
    uniqueKeyArr = calloc(nScans*nScans*5,sizeof(*uniqueKeyArr));
    double *allOrientationsArr;
    allOrientationsArr = calloc(nScans*nScans*9,sizeof(*allOrientationsArr));
    int voxNr;
    # pragma omp parallel for num_threads(numProcs) private(voxNr) schedule(dynamic)
    for (voxNr=0;voxNr<nScans*nScans;voxNr++){
        FILE *valsF, *keyF;
        char valsFN[2048], keyFN[2048];
        sprintf(valsFN,"%s/IndexBest_voxNr_%0*d.bin",folderName,6,voxNr);
        sprintf(keyFN,"%s/IndexKey_voxNr_%0*d.txt",folderName,6,voxNr);
        printf("%s\n",keyFN);
        valsF = fopen(valsFN,"rb");
        keyF = fopen(keyFN,"r");
        fseek(keyF,0L,SEEK_END);
	    size_t szt = ftell(keyF);
        if (szt==0){
            fclose(keyF);
            fclose(valsF);
            continue;
        }
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
        double *OMArr, *tmpArr, *confIAArr;
        OMArr = calloc(nIDs*9,sizeof(double));
        confIAArr = calloc(nIDs*2,sizeof(double));
        tmpArr = calloc(nIDs*16,sizeof(double));
        fread(tmpArr,nIDs*16*sizeof(double),1,valsF);
        fclose(valsF);
        int i,j,k;
        for (i=0;i<nIDs;i++){
            confIAArr[i*2+0] = tmpArr[i*16+15]/tmpArr[i*16+14];
            confIAArr[i*2+1] = tmpArr[i*16+1];
            for (i=0;i<nIDs;i++) for (j=0;j<9;j++) OMArr[i*9+j] = tmpArr[i*16+2+j];
        }
        bool *markArr;
        markArr = malloc(nIDs*sizeof(*markArr));
        for (i=0;i<nIDs;i++) markArr[i] = false;
        int bestRow=-1;
        double bestConf = -1, bestIA = 100;
        for (i=0;i<nIDs;i++){
            if (markArr[i]==true) continue;
            if (confIAArr[i*2+0] < bestConf) continue;
            if (confIAArr[i*2+0] == bestConf && confIAArr[i*2+1] < bestIA) continue;
            bestConf = confIAArr[i*2+0];
            bestIA = confIAArr[i*2+1];
            bestRow = i;
        }
        if (bestRow==-1) {
            allKeyArr[voxNr*4+0] = -1;
            free(confIAArr);
            free(keys);
            free(markArr);
            free(tmpArr);
            free(OMArr);
            continue;
        }
        free(confIAArr);
        for (i=0;i<nIDs;i++) markArr[i] = false;
        double OMThis[9], OMInside[9], Quat1[4],Quat2[4], Angle, Axis[3],ang;
        size_t *uniqueArrThis;
        uniqueArrThis = calloc(nIDs*4,sizeof(*uniqueArrThis));
        double *uniqueOrientArrThis;
        uniqueOrientArrThis = calloc(nIDs*9,sizeof(*uniqueOrientArrThis));
        int nUniquesThis = 0;
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
            for (j=0;j<4;j++) uniqueArrThis[nUniquesThis*4+j] = keys[i*4+j];
            for (j=0;j<9;j++) uniqueOrientArrThis[nUniquesThis*9+j] = OMArr[j];
            nUniquesThis++;
        }
        FILE *outKeyF;
        char outKeyFN[2048];
        sprintf(outKeyFN,"%s/UniqueIndexSingleKey.bin",folderName);
        int ib = open(outKeyFN, O_CREAT|O_WRONLY, S_IRUSR|S_IWUSR);
        int rc = pwrite(ib,&keys[bestRow*4+0],4*sizeof(size_t),4*sizeof(size_t)*voxNr);
        rc = close(ib);
        sprintf(outKeyFN,"%s/UniqueIndexKeyOrientAll_voxNr_%0*d.txt",folderName,6,voxNr);
        outKeyF = fopen(outKeyFN,"w");
        for (i=0;i<nUniquesThis;i++) {
            for (j=0;j<4;j++) fprintf(outKeyF,"%zu ",uniqueArrThis[i*4+j]);
            for (j=0;j<9;j++) fprintf(outKeyF,"%lf ",uniqueOrientArrThis[i*9+j]);
            fprintf(outKeyF,"\n");
        }
        fclose(outKeyF);
        for (i=0;i<4;i++) allKeyArr[voxNr*4+i] = keys[bestRow*4+i];
        for (i=0;i<9;i++) allOrientationsArr[voxNr*9+i] = tmpArr[bestRow*16+2+i];
        free(uniqueArrThis);
        free(uniqueOrientArrThis);
        free(OMArr);
        free(tmpArr);
        free(markArr);
        free(keys);
    }
    int nUniques = 0;
    int i,j,k;
    bool *markArr2;
    markArr2 = malloc(nScans*nScans*sizeof(*markArr2));
    for (i=0;i<nScans*nScans;i++){
        if (allKeyArr[i*4+0]==-1) markArr2[i] = true;
        markArr2[i] = false;
    }
    double OMThis[9], OMInside[9], Quat1[4],Quat2[4], Angle, Axis[3],ang;
    double *uniqueOrientArr;
    uniqueOrientArr = calloc(nScans*nScans*9,sizeof(*uniqueOrientArr));
    for (i=0;i<nScans*nScans;i++){
        if (markArr2[i]==true) continue;
        for (j=0;j<9;j++) OMThis[j] = allOrientationsArr[i*9+j];
        OrientMat2Quat(OMThis,Quat1);
        for (j=i+1;j<nScans*nScans;j++){
            if (markArr2[j]==true) continue;
            for (k=0;k<9;k++) OMInside[k] = allOrientationsArr[j*9+k];
            OrientMat2Quat(OMInside,Quat2);
            Angle = GetMisOrientation(Quat1,Quat2,Axis,&ang,sgNr);
            if (ang<maxAng) markArr2[j] = true;
        }
        uniqueKeyArr[nUniques*5+0] = i;
        for (j=0;j<4;j++) uniqueKeyArr[nUniques*5+1+j] = allKeyArr[i*4+j];
        for (j=0;j<9;j++) uniqueOrientArr[nUniques*9+j] = OMThis[j];
        nUniques++;
    }
    free(markArr2);
    free(allKeyArr);
    free(allOrientationsArr);
    char *originalFolder = argv[1], mergedIDsFN[2048];
    // Write out
    FILE *uniqueOrientationsF;
    char uniqueOrientsFN[2048];
    sprintf(uniqueOrientsFN,"%s/UniqueOrientations.csv",originalFolder);
    uniqueOrientationsF = fopen(uniqueOrientsFN,"w");
    for (i=0;i<nUniques;i++){
        for (j=0;j<5;j++) fprintf(uniqueOrientationsF,"%zu ",uniqueKeyArr[i*5+j]);
        for (j=0;j<9;j++) fprintf(uniqueOrientationsF,"%lf ",uniqueOrientArr[i*9+j]);
        fprintf(uniqueOrientationsF,"\n");
    }
    sprintf(mergedIDsFN,"%s/IDsMergedScanning.csv",originalFolder);
    FILE *fIDsMerged;
    fIDsMerged = fopen(mergedIDsFN,"r");
    char aline[2048], dummy[1000];
    int nIDsTot = 0;
    int *mergeMap;
    mergeMap = calloc(MAX_N_SPOTS_TOTAL*3,sizeof(*mergeMap));
    fgets(aline,2048,fIDsMerged);
    while(fgets(aline,2048,fIDsMerged)!=NULL){
        sscanf(aline,"%d,%d,%d",&mergeMap[nIDsTot*3+0],&mergeMap[nIDsTot*3+1],&mergeMap[nIDsTot*3+2]);
        nIDsTot++;
    }
    realloc(mergeMap,nIDsTot*3*sizeof(*mergeMap));
    fclose(fIDsMerged);
    struct InputData *allSpotIDs;
    double *allSpots;
    allSpotIDs = calloc(MAX_N_SPOTS_PER_GRAIN*nUniques,sizeof(*allSpotIDs));
    size_t nAllSpots=0, thisVoxNr;
    size_t startPos, nSpots;
    for (i=0;i<nUniques;i++){
        thisVoxNr = uniqueKeyArr[i*5+0];
        nSpots = uniqueKeyArr[i*5+2];
        startPos = uniqueKeyArr[i*5+4];
        char IDsFNThis[2048];
        sprintf(IDsFNThis,"%s/IndexBest_IDs_voxNr_%0*d.bin",folderName,6,thisVoxNr);
        FILE *IDF;
        IDF = fopen(IDsFNThis,"rb");
        fseek(IDF,startPos,SEEK_SET);
        int *IDArrThis;
        IDArrThis = malloc(nSpots*sizeof(*IDArrThis));
        fread(IDArrThis,nSpots*sizeof(int),1,IDF);
        fclose(IDF);
        for (j=0;j<nSpots;j++){
            allSpotIDs[nAllSpots+j].mergedID = IDArrThis[j];
            allSpotIDs[nAllSpots+j].originalID = mergeMap[(IDArrThis[j]-1)*3+1];
            allSpotIDs[nAllSpots+j].scanNr = mergeMap[(IDArrThis[j]-1)*3+2];
            allSpotIDs[nAllSpots+j].grainNr = i;
            allSpotIDs[nAllSpots+j].spotNr = j;
        }
        free(IDArrThis);
        nAllSpots+=nSpots;
    }
    free(uniqueKeyArr);
    free(mergeMap);
    realloc(allSpotIDs,nAllSpots*sizeof(*allSpotIDs));
    qsort(allSpotIDs,nAllSpots,sizeof(struct InputData),cmpfunc);
    allSpots = calloc(nAllSpots*5,sizeof(*allSpots));
    int rowNrThis, rowNrPrevious, rowsToSkip;
    for (i=0;i<nScans;i++){
        rowNrPrevious = 0;
        char fnInpThis[2048];
        sprintf(fnInpThis,"%s/InputAllExtraInfoFittingAll%d.csv",originalFolder,i);
        FILE *inpExtraF;
        inpExtraF = fopen(fnInpThis,"r");
        fgets(aline,2048,inpExtraF);
        for (j=0;j<nAllSpots;j++){
            if (allSpotIDs[j].scanNr != i) continue;
            rowNrThis = allSpotIDs[j].originalID - 1;
            rowsToSkip = rowNrThis-rowNrPrevious;
            rowNrPrevious = rowNrThis;
            for (k=0;k<rowsToSkip;k++) fgets(aline,2048,inpExtraF);
            sscanf(aline,"%s %s %lf %s %s %lf %lf %s",dummy,dummy,&allSpots[j*5+0],dummy,dummy,&allSpots[j*5+1],&allSpots[j*5+2],dummy);
            allSpots[j*5+3] = allSpotIDs[j].grainNr;
            allSpots[j*5+4] = allSpotIDs[j].spotNr;
        }
        fclose(inpExtraF);
    }
    free(allSpotIDs);
    int nDuplicates = 0;
    bool *dupArr;
    dupArr = malloc(nAllSpots*sizeof(*dupArr));
    for (i=0;i<nAllSpots;i++) dupArr[i] = false;
    for (i=0;i<nAllSpots;i++){
        if (dupArr[i] == true) continue;
        for (j=i+1;j<nAllSpots;j++){
            if (fabs(allSpots[i*5+0]-allSpots[j*5+0]) < tolOme && fabs(allSpots[i*5+2]-allSpots[j*5+2]) < tolOme && fabs(allSpots[i*5+1]-allSpots[j*5+1]) < 0.01){
                nDuplicates++;
                dupArr[i] = true;
                }
        }
    }
    int nAllSpotsFin=0;
    double *allSpotsFin;
    allSpotsFin = calloc((nAllSpots-nDuplicates)*5,sizeof(*allSpotsFin)); // Omega, RingNr, Eta
    for (i=0;i<nAllSpots;i++){
        if (dupArr[i]==true) continue;
        for (j=0;j<4;j++) allSpotsFin[nAllSpotsFin*5+j] = allSpots[i*5+j]; 
        nAllSpotsFin++;
    }
    free(dupArr);
    free(allSpots);
    int *nrHKLsFilled, maxNHKLs=-1;
    nrHKLsFilled = calloc(nUniques,sizeof(*nrHKLsFilled));
    for (i=0;i<nUniques;i++){
        for (j=0;j<nAllSpotsFin;j++){
            if (allSpotsFin[j*5+3]==i){
                allSpotsFin[j+8+4] = nrHKLsFilled[j];
                nrHKLsFilled[j]++;
                if (nrHKLsFilled[j]>maxNHKLs) maxNHKLs=nrHKLsFilled[j];
            }
        }
    }
    double *sinoArr, *omeArr;
    sinoArr = calloc(nUniques*maxNHKLs*nScans,sizeof(*sinoArr));
    omeArr = calloc(nUniques*maxNHKLs,sizeof(*omeArr));
    # pragma omp parallel for num_threads(numProcs) private(i) schedule(dynamic)
    for (i=0;i<nScans;i++){
        char fnInpThis[2048];
        sprintf(fnInpThis,"%s/InputAllExtraInfoFittingAll%d.csv",originalFolder,i);
        FILE *inpExtraF;
        inpExtraF = fopen(fnInpThis,"r");
        char line[2048];
        fgets(line,2048,inpExtraF);
        int nSpotsThisScan=0;
        double *ArrThis;
        char dThis[1000];
        ArrThis = calloc(MAX_N_SPOTS_TOTAL*5,sizeof(*ArrThis));
        int nSptsThis=0;
        while (fgets(line,2048,inpExtraF)!=NULL){
            sscanf("%s %s %lf %lf %lf %lf %lf %s",dThis,dThis,&ArrThis[nSptsThis*5+0],
                        &ArrThis[nSptsThis*5+1],&ArrThis[nSptsThis*5+2],&ArrThis[nSptsThis*5+3],
                        &ArrThis[nSptsThis*5+4],dThis);
            nSptsThis++;
        }
        realloc(ArrThis,nSptsThis*5*sizeof(*ArrThis));
        for (j=0;j<nSptsThis;j++){
            for (k=0;k<nAllSpotsFin;k++){
                if (allSpotsFin[k*5+3]==(double)i){
                    if (fabs(allSpotsFin[k*5+0]-ArrThis[j*5+0])<tolOme){
                        if (fabs(allSpotsFin[k*5+1]-ArrThis[j*5+3])<0.01){
                            if (fabs(allSpotsFin[k*5+2]-ArrThis[j*5+4])<tolEta){
                                size_t locThis;
                                locThis = ((size_t)allSpotsFin[k*5+3])*maxNHKLs*nScans;
                                locThis += ((size_t)allSpotsFin[k*5+4])*nScans;
                                locThis += i;
                                sinoArr[locThis] = ArrThis[j*5+1];
                                locThis = ((size_t)allSpotsFin[k*5+3])*maxNHKLs + ((size_t)allSpotsFin[k*5+4]);
                                if (omeArr[locThis]==0) {
                                    omeArr[locThis] = allSpotsFin[k*5+0];
                                }
                            }
                        }
                    }
                }
            }
        }
        free(ArrThis);
        fclose(inpExtraF);
    }
    free(allSpotsFin);
    char sinoFN[2048], omeFN[2048], HKLsFN[2048];
    sprintf(sinoFN,"%s/sinos_%d_%d_%d.bin",originalFolder,nUniques,maxNHKLs,nScans);
    sprintf(omeFN,"%s/omegas_%d_%d.bin",originalFolder,nUniques,maxNHKLs);
    sprintf(HKLsFN,"%s/nrHKLs_%d.bin",originalFolder,nUniques);
    FILE *sinoF, *omeF, *HKLsF;
    sinoF = fopen(sinoFN,"wb");
    omeF = fopen(omeFN,"wb");
    HKLsF = fopen(HKLsFN,"wb");
    fwrite(sinoArr,nUniques*maxNHKLs*nScans*sizeof(*sinoArr),1,sinoF);
    fwrite(omeArr,nUniques*maxNHKLs*sizeof(*omeArr),1,omeF);
    fwrite(nrHKLsFilled,nUniques*sizeof(*nrHKLsFilled),1,HKLsF);
    fclose(sinoF);
    fclose(omeF);
    fclose(HKLsF);
    free(sinoArr);
    free(omeArr);
    free(nrHKLsFilled);
}