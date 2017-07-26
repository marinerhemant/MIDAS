//
//
// MatchGrains.c
//
// Author: Hemant Sharma
// Dt: 2017/07/12

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_N_GRAINS 1000000

static inline
double Len3d(double x, double y, double z)
{
	return sqrt(x*x + y*y + z*z);
}

static inline
int**
allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrixInt(int **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline
double**
allocMatrix(int nrows, int ncols)
{
    double** arr;
    int i;
    arr = malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline
void
FreeMemMatrix(double **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

struct sortArrayType{
	double angle;
	int x; // Maps to j or 1
	int y; // Maps to i or 2
};

static int cmpfunc (const void *a, const void *b){
	struct sortArrayType *ia = (struct sortArrayType *)a;
	struct sortArrayType *ib = (struct sortArrayType *)b;
	return (int)(1000000.f*ia->angle - 1000000.f*ib->angle);
}

static inline
int StartsWith(const char *a, const char *b)
{
	if (strncmp(a,b,strlen(b)) == 0) return 1;
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc < 12){
		printf("Usage: MatchGrains   OutFileName   state1.txt state2.txt"
		"     SGNr      offset[3]"
		"     matchMode    beamThickness1 beamThickness2    matchDuplicates    (optional)weights\n");
		printf("                                (list of Grains.csv files)"
		"         [microns][3vals]"
		" (next line)     [microns]      [microns]       binary[0 or 1]  [degrees and microns]\n");
		printf("MatchMode: \n\t0: Match orientations only\n\t1: Match "
		"positions only\n\t2: Match according to both orientation and "
		"position using supplied weights\n");
		printf("Offset: 1 value each in x(along beam), y(out the door), z(up)"
		" directions. Going from State1 to State2.\n");
		printf("stateN.txt can be a file containing a list of Grains.csv files or a Grains.csv file directly.\n");
		printf("matchDuplicates must be 1 for now.\n");
		return 1;
	}
	clock_t start, end;
	double diftotal;
	start = clock();
	char *fn1, *fn2;
	char *outfn;
	FILE *f1, *f2;
	outfn = argv[1];
	fn1 = argv[2];
	fn2 = argv[3];
	f1 = fopen(fn1,"r");
	f2 = fopen(fn2,"r");
	double offset[3], beamThickness1, beamThickness2;
	int matchMode;
	int SGNr;
	SGNr = atoi(argv[4]);
	offset[0] = atof(argv[5]);
	offset[1] = atof(argv[6]);
	offset[2] = atof(argv[7]);
	matchMode = atoi(argv[8]);
	beamThickness1 = atof(argv[9]);
	beamThickness2 = atof(argv[10]);
	int removeDuplicates;
	removeDuplicates = atoi(argv[11]);
	double weights[2];
	if (matchMode == 2){
		weights[0] = atof(argv[12]);
		weights[1] = atof(argv[13]);
	}
	char aline[4096],bline[4096];
	FILE *grainsF;
	double **Quats1, **Quats2, **Pos1, **Pos2;
	int **IDs1, **IDs2; // Final ID, FNr, InitialID.
	int FNr = 0;
	int ThisID = 0;
	int GrainID;
	Quats1 = allocMatrix(MAX_N_GRAINS,4);
	Quats2 = allocMatrix(MAX_N_GRAINS,4);
	Pos1 = allocMatrix(MAX_N_GRAINS,3);
	Pos2 = allocMatrix(MAX_N_GRAINS,3);
	IDs1 = allocMatrixInt(MAX_N_GRAINS,3);
	IDs2 = allocMatrixInt(MAX_N_GRAINS,3);
	double OrientMatrix[9];
	double QuatTemp[4];
	int len;
	int grainsSupplied = 0;
	while (fgets(aline,4096,f1)!=NULL){
		if (StartsWith(aline,"%NumGrains")){
			grainsSupplied = 1;
		}
		if (aline[0] == '%') continue;
		if (grainsSupplied == 1){ // Was supplied a grains file directly.
			sscanf(aline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&GrainID,&OrientMatrix[0],
				&OrientMatrix[1],&OrientMatrix[2],&OrientMatrix[3],&OrientMatrix[4],&OrientMatrix[5],&OrientMatrix[6],
				&OrientMatrix[7],&OrientMatrix[8],&Pos1[ThisID][0],&Pos1[ThisID][1],&Pos1[ThisID][2]);
				OrientMat2Quat(OrientMatrix,QuatTemp);
				Pos1[ThisID][2] += FNr*beamThickness1;
				Quats1[ThisID][0] = QuatTemp[0];
				Quats1[ThisID][1] = QuatTemp[1];
				Quats1[ThisID][2] = QuatTemp[2];
				Quats1[ThisID][3] = QuatTemp[3];
				IDs1[ThisID][0] = ThisID;
				IDs1[ThisID][1] = FNr;
				IDs1[ThisID][2] = GrainID;
				ThisID++;
				continue;
		}
		// List of grains file was supplied.
		if (aline[0] == '#') continue;
		len = strlen(aline);
		aline[len-1] = '\0';
		grainsF = fopen(aline,"r");
		printf("Reading file: %s\n",aline);fflush(stdout);
		while (fgets(bline,4096,grainsF)!=NULL){
			if (bline[0] == '%') continue;
			sscanf(bline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&GrainID,&OrientMatrix[0],
				&OrientMatrix[1],&OrientMatrix[2],&OrientMatrix[3],&OrientMatrix[4],&OrientMatrix[5],&OrientMatrix[6],
				&OrientMatrix[7],&OrientMatrix[8],&Pos1[ThisID][0],&Pos1[ThisID][1],&Pos1[ThisID][2]);
				OrientMat2Quat(OrientMatrix,QuatTemp);
				Pos1[ThisID][2] += FNr*beamThickness1;
				Quats1[ThisID][0] = QuatTemp[0];
				Quats1[ThisID][1] = QuatTemp[1];
				Quats1[ThisID][2] = QuatTemp[2];
				Quats1[ThisID][3] = QuatTemp[3];
				IDs1[ThisID][0] = ThisID;
				IDs1[ThisID][1] = FNr;
				IDs1[ThisID][2] = GrainID;
				ThisID++;
		}
		fclose(grainsF);
		FNr++;
	}
	grainsSupplied = 0;
	fclose(f1);
	int totIDs1, totIDs2;
	totIDs1 = ThisID;
	FNr = 0;
	ThisID = 0;
	while (fgets(aline,4096,f2)!=NULL){
		if (StartsWith(aline,"%NumGrains")){
			grainsSupplied = 1;
		}
		if (aline[0] == '%') continue;
		if (grainsSupplied == 1){ // Was supplied a grains file directly.
			sscanf(aline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&GrainID,&OrientMatrix[0],
				&OrientMatrix[1],&OrientMatrix[2],&OrientMatrix[3],&OrientMatrix[4],&OrientMatrix[5],&OrientMatrix[6],
				&OrientMatrix[7],&OrientMatrix[8],&Pos2[ThisID][0],&Pos2[ThisID][1],&Pos2[ThisID][2]);
				Pos2[ThisID][0] += offset[0];
				Pos2[ThisID][1] += offset[1];
				Pos2[ThisID][2] += offset[2] + FNr*beamThickness2;
				OrientMat2Quat(OrientMatrix,QuatTemp);
				Quats2[ThisID][0] = QuatTemp[0];
				Quats2[ThisID][1] = QuatTemp[1];
				Quats2[ThisID][2] = QuatTemp[2];
				Quats2[ThisID][3] = QuatTemp[3];
				IDs2[ThisID][0] = ThisID;
				IDs2[ThisID][1] = FNr;
				IDs2[ThisID][2] = GrainID;
				ThisID++;
				continue;
		}
		// List of grains file was supplied.
		if (aline[0] == '#') continue;
		len = strlen(aline);
		aline[len-1] = '\0';
		grainsF = fopen(aline,"r");
		printf("Reading file: %s\n",aline);fflush(stdout);
		while (fgets(bline,4096,grainsF)!=NULL){
			if (bline[0] == '%') continue;
			sscanf(bline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&GrainID,&OrientMatrix[0],
				&OrientMatrix[1],&OrientMatrix[2],&OrientMatrix[3],&OrientMatrix[4],&OrientMatrix[5],&OrientMatrix[6],
				&OrientMatrix[7],&OrientMatrix[8],&Pos2[ThisID][0],&Pos2[ThisID][1],&Pos2[ThisID][2]);
				Pos2[ThisID][0] += offset[0];
				Pos2[ThisID][1] += offset[1];
				Pos2[ThisID][2] += offset[2] + FNr*beamThickness2;
				OrientMat2Quat(OrientMatrix,QuatTemp);
				Quats2[ThisID][0] = QuatTemp[0];
				Quats2[ThisID][1] = QuatTemp[1];
				Quats2[ThisID][2] = QuatTemp[2];
				Quats2[ThisID][3] = QuatTemp[3];
				IDs2[ThisID][0] = ThisID;
				IDs2[ThisID][1] = FNr;
				IDs2[ThisID][2] = GrainID;
				ThisID++;
		}
		fclose(grainsF);
		FNr++;
	}
	fclose(f2);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time to read files: %f s.\n",diftotal);
	totIDs2 = ThisID;
	int i,j,k;
	double Q1[4], Q2[4], Axis[3], Angle, ang, **Angles, difflen;
	double posT1[3], posT2[3];
	Angles = allocMatrix(totIDs1,totIDs2);
	struct sortArrayType *SortMatrix;
	SortMatrix = malloc(totIDs1*totIDs2*sizeof(*SortMatrix));
	int **doneMatrix;
	doneMatrix = allocMatrixInt(totIDs1,totIDs2);
	if (matchMode == 0){
		for (i=0;i<totIDs2;i++){
			Q2[0] = Quats2[i][0];
			Q2[1] = Quats2[i][1];
			Q2[2] = Quats2[i][2];
			Q2[3] = Quats2[i][3];
			for (j=0;j<totIDs1;j++){
				Q1[0] = Quats1[j][0];
				Q1[1] = Quats1[j][1];
				Q1[2] = Quats1[j][2];
				Q1[3] = Quats1[j][3];
				Angle = GetMisOrientation(Q1,Q2,Axis,&ang,SGNr);
				Angles[j][i] = ang;
				SortMatrix[j*totIDs2+i].angle = Angles[j][i];
				SortMatrix[j*totIDs2+i].x = j;
				SortMatrix[j*totIDs2+i].y = i;
				doneMatrix[j][i] = 0;
			}
		}
	}else if (matchMode == 1){
		for (i=0;i<totIDs2;i++){
			posT2[0] = Pos2[i][0];
			posT2[1] = Pos2[i][1];
			posT2[2] = Pos2[i][2];
			for (j=0;j<totIDs1;j++){
				posT1[0] = Pos1[j][0];
				posT1[1] = Pos1[j][1];
				posT1[2] = Pos1[j][2];
				difflen = Len3d(posT2[0]-posT1[0],posT2[1]-posT1[1],posT2[2]-posT1[2]);
				Angles[j][i] = difflen;
				SortMatrix[j*totIDs2+i].angle = Angles[j][i];
				SortMatrix[j*totIDs2+i].x = j;
				SortMatrix[j*totIDs2+i].y = i;
				doneMatrix[j][i] = 0;
			}
		}
	}else if (matchMode == 2){
		for (i=0;i<totIDs2;i++){
			Q2[0] = Quats2[i][0];
			Q2[1] = Quats2[i][1];
			Q2[2] = Quats2[i][2];
			Q2[3] = Quats2[i][3];
			posT2[0] = Pos2[i][0];
			posT2[1] = Pos2[i][1];
			posT2[2] = Pos2[i][2];
			for (j=0;j<totIDs1;j++){
				Q1[0] = Quats1[j][0];
				Q1[1] = Quats1[j][1];
				Q1[2] = Quats1[j][2];
				Q1[3] = Quats1[j][3];
				posT1[0] = Pos1[j][0];
				posT1[1] = Pos1[j][1];
				posT1[2] = Pos1[j][2];
				Angle = GetMisOrientation(Q1,Q2,Axis,&ang,SGNr);
				difflen = Len3d(posT2[0]-posT1[0],posT2[1]-posT1[1],posT2[2]-posT1[2]);
				Angles[j][i] = ang/weights[0] + difflen/weights[1];
				SortMatrix[j*totIDs2+i].angle = Angles[j][i];
				SortMatrix[j*totIDs2+i].x = j;
				SortMatrix[j*totIDs2+i].y = i;
				doneMatrix[j][i] = 0;
			}
		}
	}
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time to make bigArray: %f s.\n",diftotal);
	qsort(SortMatrix,totIDs1*totIDs2,sizeof(struct sortArrayType),cmpfunc);
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time to sort bigArray: %f s.\n",diftotal);
	int posX, posY;
	double **Matches;
	Matches = allocMatrix(totIDs2,26);
	int counter = 0;
	for (i=0;i<totIDs1*totIDs2;i++){
		posX = SortMatrix[i].x; // State1
		posY = SortMatrix[i].y; // State2
		if (doneMatrix[posX][posY] == 1){
			continue;
		}
		if (removeDuplicates == 1){
			for (j=0;j<totIDs1;j++){
				doneMatrix[j][posY] = 1;
			}
			for (j=0;j<totIDs2;j++){
				doneMatrix[posX][j] = 1;
			}
		}
		//printf("%d %d %lf\n", SortMatrix[i].x,SortMatrix[i].y,SortMatrix[i].angle);
		Q1[0] = Quats1[posX][0];
		Q1[1] = Quats1[posX][1];
		Q1[2] = Quats1[posX][2];
		Q1[3] = Quats1[posX][3];
		Q2[0] = Quats2[posY][0];
		Q2[1] = Quats2[posY][1];
		Q2[2] = Quats2[posY][2];
		Q2[3] = Quats2[posY][3];
		Angle = GetMisOrientation(Q1,Q2,Axis,&ang,SGNr);
		Matches[counter][0] = IDs2[posY][0];
		Matches[counter][1] = IDs2[posY][1];
		Matches[counter][2] = IDs2[posY][2];
		Matches[counter][3] = IDs1[posX][0];
		Matches[counter][4] = IDs1[posX][1];
		Matches[counter][5] = IDs1[posX][2];
		Matches[counter][6] = Quats2[posY][0];
		Matches[counter][7] = Quats2[posY][1];
		Matches[counter][8] = Quats2[posY][2];
		Matches[counter][9] = Quats2[posY][3];
		Matches[counter][10] = Quats1[posX][0];
		Matches[counter][11] = Quats1[posX][1];
		Matches[counter][12] = Quats1[posX][2];
		Matches[counter][13] = Quats1[posX][3];
		Matches[counter][14] = Pos2[posY][0];
		Matches[counter][15] = Pos2[posY][1];
		Matches[counter][16] = Pos2[posY][2];
		Matches[counter][17] = Pos1[posX][0];
		Matches[counter][18] = Pos1[posX][1];
		Matches[counter][19] = Pos1[posX][2];
		Matches[counter][20] = Angles[posX][posY];
		Matches[counter][21] = ang;
		Matches[counter][22] = Pos2[posY][0] - Pos1[posX][0];
		Matches[counter][23] = Pos2[posY][1] - Pos1[posX][1];
		Matches[counter][24] = Pos2[posY][2] - Pos1[posX][2];
		Matches[counter][25] = Len3d(Matches[counter][22],Matches[counter][23],Matches[counter][24]);
		counter ++;
	}

	FILE *outfile = fopen(outfn,"w");
	fprintf(outfile,"%%NewIDState2\tFNrState2\tOrigIDState2\tNewIDState1\tFNrState1\tOrigIDState1\t"
	"Quat0State2\tQuat1State2\tQuat2State2\tQuat3State2\tQuat0State1\tQuat1State1\tQuat2State1\tQuat3State1\t"
	"Pos0State2\tPos1State2\tPos2State2\tPos0State1\tPos1State1\tPos2State1\t"
	"selectionCriteriaVal\tminAngle\tdiffPosX\tdiffPosY\tdiffPosZ\tEuclideanDistt\n");
	for (i=0;i<totIDs2;i++){
		for (j=0;j<26;j++) fprintf(outfile,"%lf\t",Matches[i][j]);
		fprintf(outfile,"\n");
	}
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n",diftotal);
	return 0;
}
