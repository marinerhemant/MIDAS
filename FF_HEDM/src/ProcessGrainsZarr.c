//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  ProcessGrains.c
//
//
//  Created by Hemant Sharma on 2014/06/24.
//
//  New Features (2014/11/06):
//  - Twins were implemented in a previous version.
//  - Single file reading is implemented now.
//  New Features (2014/11/19):
//  - Strains!!
//

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
#include <blosc2.h>
#include <stdlib.h> 
#include <zip.h> 

#define MAX_N_IDS 6000000
#define MAX_ID_IA_MAT 5000000
#define NR_MAX_IDS_PER_GRAIN 5000 // Nr spots per grain max.
#define IAColNr 20 // 20 for Internal Angle, 18 for position, 19 for omega

#define EPS 1E-12
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
static inline double sin_cos_to_angle (double s, double c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}

inline
double GetMisOrientation(double quat1[4], double quat2[4], double axis[3], double *Angle,int SGNr);

static inline
void OrientMat2Euler(double m[3][3],double Euler[3])
{
    double psi, phi, theta, sph;
	if (fabs(m[2][2] - 1.0) < EPS){
		phi = 0;
	}else{
	    phi = acos(m[2][2]);
	}
    sph = sin(phi);
    if (fabs(sph) < EPS)
    {
        psi = 0.0;
        theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0]) : sin_cos_to_angle(-m[1][0], m[0][0]);
    } else{
        psi = (fabs(-m[1][2] / sph) <= 1.0) ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph) : sin_cos_to_angle(m[0][2] / sph,1);
        theta = (fabs(m[2][1] / sph) <= 1.0) ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph) : sin_cos_to_angle(m[2][0] / sph,1);
    }
    Euler[0] = psi;
    Euler[1] = phi;
    Euler[2] = theta;
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

inline 
void OrientMat2Quat(double OrientMat[9], double Quat[4]);

static inline
int
FindInternalAnglesTwins(int nrIDs, int *IDs, int *IDsPerGrain,
				   int *NrIDsPerID, bool *IDsChecked,
				   double **OPs, double *ID_IA_Mat, int counter, int Pos,
				   int StartingID, double *Radiuses, int SGNr)
{
	int i,j,k,ThisID,ThisID2;
	bool AreTwins=false;
	ID_IA_Mat[(counter*4)] = (double) StartingID;
	ID_IA_Mat[(counter*4)+1] = (double) Pos;
	ID_IA_Mat[(counter*4)+2] = OPs[Pos][IAColNr];
	ID_IA_Mat[(counter*4)+3] = Radiuses[Pos];
	IDsChecked[Pos] = true;
	counter++;
	double Angle, Axis[3],q1[4],q2[4],ang;
	double OR1[9], OR2[9];
	for (i=0;i<9;i++){
		OR1[i] = OPs[Pos][i];
	}
	OrientMat2Quat(OR1,q1);
	for (i=0;i<NrIDsPerID[Pos];i++){
		ThisID = IDsPerGrain[(Pos*NR_MAX_IDS_PER_GRAIN)+i];
		for (j=0;j<nrIDs;j++){
			ThisID2 = IDs[j];
			if (ThisID == ThisID2 && IDsChecked[j] == false){
				for (k=0;k<9;k++){
					OR2[k] = OPs[j][k];
				}
				OrientMat2Quat(OR2,q2);
				Angle = GetMisOrientation(q1,q2,Axis,&ang,SGNr);
				AreTwins = ( fabs(ang - 60) < 0.4 ) &&
						 ( fabs(Axis[0]) - fabs(Axis[1]) ) < 0.01 &&
						 ( fabs(Axis[2]) - fabs(Axis[1]) ) < 0.01;
				if (fabs(ang) < 0.4 || AreTwins){
					counter = FindInternalAnglesTwins(nrIDs,IDs,IDsPerGrain,NrIDsPerID,IDsChecked,
							OPs,ID_IA_Mat,counter,j,ThisID,Radiuses,SGNr);
					break;
				}
			}
		}
	}
	int counte = counter;
	return counte;
}

static inline
int
FindInternalAngles(int nrIDs, int *IDs, int *IDsPerGrain,
				   int *NrIDsPerID, bool *IDsChecked,
				   double **OPs, double *ID_IA_Mat, int counter, int Pos,
				   int StartingID, double *Radiuses,int SGNr)
{
	int i,j,k,ThisID,ThisID2;
	ID_IA_Mat[(counter*4)] = (double) StartingID;
	ID_IA_Mat[(counter*4)+1] = (double) Pos;
	ID_IA_Mat[(counter*4)+2] = OPs[Pos][IAColNr];
	ID_IA_Mat[(counter*4)+3] = Radiuses[Pos];
	IDsChecked[Pos] = true;
	counter++;
	double Angle, Axis[3],q1[4],q2[4],ang;
	double OR1[9], OR2[9];
	for (i=0;i<9;i++){
		OR1[i] = OPs[Pos][i];
	}
	OrientMat2Quat(OR1,q1);
	size_t posSize = Pos;
	posSize *= NR_MAX_IDS_PER_GRAIN;
	for (i=0;i<NrIDsPerID[Pos];i++){
		ThisID = IDsPerGrain[(posSize)+i];
		for (j=0;j<nrIDs;j++){
			ThisID2 = IDs[j];
			if (ThisID == ThisID2 && IDsChecked[j] == false){
				for (k=0;k<9;k++){
					OR2[k] = OPs[j][k];
				}
				OrientMat2Quat(OR2,q2);
				Angle = GetMisOrientation(q1,q2,Axis,&ang,SGNr);
				if (fabs(ang) < 0.4){
					counter = FindInternalAngles(nrIDs,IDs,IDsPerGrain,NrIDsPerID,IDsChecked,
							OPs,ID_IA_Mat,counter,j,ThisID,Radiuses,SGNr);
					break;
				}
			}
		}
	}
	int counte = counter;
	return counte;
}

static inline void
QuatToOrientMat(
    double Quat[4],
    double OrientMat[9])
{
    double Q1_2,Q2_2,Q3_2,Q12,Q03,Q13,Q02,Q23,Q01;
    Q1_2 = Quat[1]*Quat[1];
    Q2_2 = Quat[2]*Quat[2];
    Q3_2 = Quat[3]*Quat[3];
    Q12  = Quat[1]*Quat[2];
    Q03  = Quat[0]*Quat[3];
    Q13  = Quat[1]*Quat[3];
    Q02  = Quat[0]*Quat[2];
    Q23  = Quat[2]*Quat[3];
    Q01  = Quat[0]*Quat[1];
    OrientMat[0] = 1 - 2*(Q2_2+Q3_2);
    OrientMat[1] = 2*(Q12-Q03);
    OrientMat[2] = 2*(Q13+Q02);
    OrientMat[3] = 2*(Q12+Q03);
    OrientMat[4] = 1 - 2*(Q1_2+Q3_2);
    OrientMat[5] = 2*(Q23-Q01);
    OrientMat[6] = 2*(Q13-Q02);
    OrientMat[7] = 2*(Q23+Q01);
    OrientMat[8] = 1 - 2*(Q1_2+Q2_2);
}

inline void
CalcStrainTensorFableBeaudoin(double LatCin[6],double LatticeParameterFit[6],
	double Orient[3][3], double StrainTensorSample[3][3]);

inline int
StrainTensorKenesei(int nspots,double **SpotsInfo, double Distance, double wavelength,
		double StrainTensorSample[3][3], int **IDHash,
		double *dspacings, int nRings, int startSpotMatrix, double **SpotMatrix, double *RetVal,
		double StrainTensorInput[3][3]);


int main(int argc, char *argv[])
{
	if (argc < 2){
		printf("Usage: ProcessGrainsZarr ZarrZip (optionally)TrackGrains (0 or 1)\n");
		return 1;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
    char line[5024];

    char aline[1000];
    char *str, dummy[1000];
    int Twin = 0, MinNrSpots = 1, SGNr = 225;
    double Distance, wavelength, LatCin[6];
    double BeamThickness = 0, GlobalPosition = 0;
    int NumPhases = 1, PhaseNr = 1;

	char *DataFN = argv[1];
    blosc2_init();
    // Read zarr config
    int errorp = 0;
    zip_t* arch = NULL;
    arch = zip_open(DataFN,0,&errorp);
    if (errorp!=NULL) return 1;
    struct zip_stat* finfo = NULL;
    finfo = calloc(16384, sizeof(int));
    zip_stat_init(finfo);
    zip_file_t* fd = NULL;
    int count = 0;
    char* data = NULL;
    char* s = NULL;
    char* arr;
    int32_t dsize;
    while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
        if (strstr(finfo->name,"analysis/process/analysis_parameters/SpaceGroup/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            SGNr = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Twins/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Twin = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/MinNrSpots/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            MinNrSpots = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/NumPhases/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            NumPhases = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/PhaseNr/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            PhaseNr = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/GlobalPosition/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            GlobalPosition = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/BeamThickness/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            BeamThickness = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/LatticeParameter/0")!=NULL){
            s = calloc(finfo->size + 1, sizeof(char));
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, s, finfo->size); 
            int32_t dsize = 6*sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(s,data,dsize);
            int iter;
            for (iter=0;iter<6;iter++) LatCin[iter] = *(double *)&data[iter*sizeof(double)];
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Lsd/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Distance = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Wavelength/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            wavelength = *(double *)&data[0];
            free(arr);
            free(data);
        }
        count++;
    }


	int i,j,k,ThisID,counter;
	int *IDs;
	IDs = malloc(MAX_N_IDS*sizeof(*IDs));
	for (i=0;i<MAX_N_IDS;i++) IDs[i] = 0;
	int nrIDs=0;
    char IDsFileName[1024];
    FILE *IDsFile;
    sprintf(IDsFileName,"SpotsToIndex.csv");
    printf("Reading IDs file: %s\n",IDsFileName);
    IDsFile = fopen(IDsFileName,"r");
	if (IDsFile == NULL)printf("Could not open spots file.\n");
	while (fgets(line,5024,IDsFile) != NULL){
		sscanf(line,"%d",&IDs[nrIDs]);
		if (IDs[nrIDs]<0) continue;
		nrIDs++;
	}
	if (nrIDs == 0){
		printf("No ID was found in SpotsToIndex.csv file. Please check your parameters file.\n");
		return 1;
	}
	fclose(IDsFile);
	printf("Total of %d IDs will be sorted into grains now.\n",nrIDs);
	bool *IDsToKeep;
	IDsToKeep = malloc(nrIDs*sizeof(*IDsToKeep));
	double *Radiuses;
	Radiuses = malloc(nrIDs*sizeof(*Radiuses));
	double *OPThis,**OPs;
	OPThis = malloc(27*sizeof(*OPThis));
	for (i=0;i<27;i++) OPThis[i] = 0;
	OPs = allocMatrix(nrIDs,23);
	int *IDsPerGrain,*NrIDsPerID;
	NrIDsPerID = malloc(nrIDs*sizeof(*NrIDsPerID));
	size_t sizeMat = NR_MAX_IDS_PER_GRAIN;
	sizeMat *= nrIDs;
	sizeMat *= sizeof(*IDsPerGrain);
	IDsPerGrain = malloc(sizeMat);
	for (i=0;i<nrIDs;i++){
		IDsToKeep[i] = false;
		Radiuses[i] = 0;
		for (j=0;j<23;j++){
			OPs[i][j] = 0;
		}
		for (j=0;j<NR_MAX_IDS_PER_GRAIN;j++){
			IDsPerGrain[(NR_MAX_IDS_PER_GRAIN*i) + j] = 0;
		}
		NrIDsPerID[i] = 0;
	}
	FILE *fileKey = fopen("Results/Key.bin","r");
	FILE *fileOPFit = fopen("Results/OrientPosFit.bin","r");
	FILE *fileProcessKey = fopen("Results/ProcessKey.bin","r");
	if (fileKey == NULL){
		printf("Key file was not found. This means nothing was indexed in the previous step.\nTypically this means parameters were not correct. Please check.\nExiting.\n");
		return 1;
	}
	if (fileOPFit == NULL){
		printf("OrientPos file was not found. This means nothing was indexed in the previous step.\nTypically this means parameters were not correct. Please check.\nExiting.\n");
		return 1;
	}
	if (fileProcessKey == NULL){
		printf("ProcessKey file was not found. This means nothing was indexed in the previous step.\nTypically this means parameters were not correct. Please check.\nExiting.\n");
		return 1;
	}
	int *keyID;
	keyID = malloc(2*sizeof(*keyID));
	size_t readKey, readOP, readProcess;
	readProcess = fread(IDsPerGrain,NR_MAX_IDS_PER_GRAIN*nrIDs*sizeof(int),1,fileProcessKey);
	fclose(fileProcessKey);
	for (i=0;i<nrIDs;i++){
		readKey = fread(keyID,2*sizeof(int),1,fileKey);
		IDsToKeep[i] = true;
		if (keyID[0] == 0){
			IDsToKeep[i] = false;
		}
		NrIDsPerID[i] = keyID[1];
		readOP = fread(OPThis,27*sizeof(double),1,fileOPFit);
		counter = 0;
		for (j=0;j<27;j++){
			if (j == 0 || j == 10 || j == 14 || j == 21){
				continue;
			}
			OPs[i][counter] = OPThis[j];
			counter++;
		}
		Radiuses[i] = OPThis[25];
	}
	fclose(fileKey);
	fclose(fileOPFit);
	int StartingID,ThisID1,ThisID2;
	int nGrainPositions = 0,BestGrainPos, bestGrainID;
	int *GrainPositions,*nGrainsMatched;
	GrainPositions = malloc(nrIDs*sizeof(*GrainPositions));
	nGrainsMatched = malloc(nrIDs*sizeof(*nGrainsMatched));
	double minIA,maxRadThis;
	printf("Read all grain files.\n");
	bool *IDsChecked;
	IDsChecked = malloc(nrIDs*sizeof(*IDsChecked));
	for (i=0;i<nrIDs;i++) IDsChecked[i] = false;
	for (i=0;i<nrIDs;i++){
		GrainPositions[i] = 0;
		nGrainsMatched[i] = 0;
		if (IDsToKeep[i] == false){
			IDsChecked[i] = true;
		}
	}
	double *ID_IA_MAT;
	double ang, Angle, Axis[3],DiffPos,OR1[9],q1[4],OR2[9],q2[4],q3[4];
	int counte,counten,totcount=0;
	ID_IA_MAT = calloc(MAX_ID_IA_MAT*4,sizeof(*ID_IA_MAT));
	FILE *fIDs = fopen("GrainIDsKey.csv","w");
	int trackGrains = 0;
	if (argc==3) trackGrains = atoi(argv[2]);
	for (i=0;i<nrIDs;i++){
		if (i%1000 == 0) printf("Processed %d of %d IDs.\n",i,nrIDs);
		if (IDsChecked[i] == false){
			counte = 0;
			StartingID = IDs[i];
			maxRadThis = Radiuses[i];
			minIA = OPs[i][IAColNr];
			BestGrainPos = i;
			if (trackGrains == 0){
				if (Twin ==0){
					counten = FindInternalAngles     (nrIDs,IDs,IDsPerGrain,NrIDsPerID,
					IDsChecked,OPs,ID_IA_MAT,counte,i,StartingID,Radiuses,SGNr);
				}else{
					counten = FindInternalAnglesTwins(nrIDs,IDs,IDsPerGrain,NrIDsPerID,
					IDsChecked,OPs,ID_IA_MAT,counte,i,StartingID,Radiuses,SGNr);
				}
			} else {
				counten = 0;
				ID_IA_MAT[(counten*4)] = (double) StartingID;
				ID_IA_MAT[(counten*4)+1] = (double) i;
				ID_IA_MAT[(counten*4)+2] = OPs[i][IAColNr];
				ID_IA_MAT[(counten*4)+3] = Radiuses[i];
				counten = 1;
			}
			totcount+=counten;
			nGrainsMatched[i] = counten;
			if (counten < MinNrSpots){
				continue;
			}
			for (j=0;j<counten;j++){
				// Write out the following information: for each counten, save the IDs Matched, so that we can use them later
				// ID_IA_MAT[(j*4)+1] has the row Number, ID_IA_MAT[(j*4)] has the ID, these are the two things we need......
				if (ID_IA_MAT[(j*4)+2] < minIA){
					minIA = ID_IA_MAT[(j*4)+2];
					BestGrainPos = (int)ID_IA_MAT[(j*4)+1];
					bestGrainID = (int)ID_IA_MAT[(j*4)];
					maxRadThis = ID_IA_MAT[(j*4)+3];
				}
			}
			fprintf(fIDs,"%d %d ",bestGrainID,BestGrainPos);
			for (j=0;j<counten;j++){
				// Write out ID_IA_MAT[(j*4)] (0,1) along with BestGrainPos and corresponding ID
				if ((int)ID_IA_MAT[(j*4)+1] == BestGrainPos) continue;
				fprintf(fIDs,"%d %d ",(int)ID_IA_MAT[(j*4)],(int)ID_IA_MAT[(j*4)+1]);
			}
			fprintf(fIDs,"\n");
			GrainPositions[nGrainPositions] = BestGrainPos;
			Radiuses[BestGrainPos] = maxRadThis;
			nGrainPositions ++;
		}
	}
	printf("%d\n ",nGrainPositions);
	fclose(fIDs);
	//Write out
	int nGrains=0;
	int *IDsDone;
	IDsDone = malloc((nGrainPositions*2)*sizeof(*IDsDone)); // Fix for now.
	int cres=0;
	int DoneAlready = 0;
	double StrainTensorSampleKen[3][3];
	double StrainTensorSampleFab[3][3];
	double *dummySampleInfo;
	dummySampleInfo = malloc(22*NR_MAX_IDS_PER_GRAIN*sizeof(*dummySampleInfo));
	double LatticeParameterFit[6],Orient[3][3];
	double **SpotsInfo;
	SpotsInfo = allocMatrix(NR_MAX_IDS_PER_GRAIN,8);
	int nspots, rown;
	// Calculate Strains Now
	int fullInfoFile = open("Output/FitBest.bin",O_RDONLY);
	size_t OffSt;
	size_t ReadSize;
	double MultR=1000000.0;
	double BeamCenter = 0, FullVol = 0,VNorm;
	int rown2;
	int **IDHash;
	IDHash = allocMatrixInt(NR_MAX_IDS_PER_GRAIN,3);
	double *dspacings;
	dspacings = malloc(NR_MAX_IDS_PER_GRAIN*sizeof(*dspacings));
	int nRings=0;
	char *hashfn = "IDsHash.csv";
	FILE *hashfile = fopen(hashfn,"r");
	int MakeHash = 0;
	if (hashfile != NULL){
		while (fgets(aline,2000,hashfile)!=NULL){
			sscanf(aline,"%d %d %d %lf",&IDHash[nRings][0],&IDHash[nRings][1],&IDHash[nRings][2],&dspacings[nRings]);
			nRings++;
		}
	}else{
		MakeHash = 1;
	}
	fclose(hashfile);
	double **SpotMatrix, **InputMatrix;
	SpotMatrix = allocMatrix(NR_MAX_IDS_PER_GRAIN,12);
	InputMatrix = allocMatrix(MAX_N_IDS,10);
	int counterSpotMatrix = 0;
	char *inputallfn = "InputAllExtraInfoFittingAll.csv";
	FILE *inpfile = fopen(inputallfn,"r");
	int counterIF=0;
	fgets(aline,2000,inpfile);
	int currentRing;
	while (fgets(aline,2000,inpfile)!=NULL){
		sscanf(aline,"%lf %lf %lf %s %lf %lf %lf %lf %s %s %s %lf %lf %lf",&InputMatrix[counterIF][6], &InputMatrix[counterIF][7], &InputMatrix[counterIF][0],
			dummy, &InputMatrix[counterIF][1], &InputMatrix[counterIF][5], &InputMatrix[counterIF][4], &InputMatrix[counterIF][8], dummy, dummy, dummy,
			&InputMatrix[counterIF][2], &InputMatrix[counterIF][3],&InputMatrix[counterIF][9]);
		if ((int)InputMatrix[counterIF][1] != counterIF+1){
			printf("IDs dont match.\nExiting\n");
			return(1);
		}
		// Write Hash Matrix if needed.
		if (MakeHash == 1){
			if (counterIF == 0){ // First Spot
				IDHash[nRings][0] = (int) InputMatrix[counterIF][5];
				IDHash[nRings][1] = counterIF + 1;
				currentRing = (int) InputMatrix[counterIF][5];
				nRings++;
			}else{
				if ((int) InputMatrix[counterIF][5] != currentRing){ // Each time ring number changes
					IDHash[nRings][0] = (int) InputMatrix[counterIF][5];
					IDHash[nRings][1] = counterIF + 1;
					IDHash[nRings-1][2] = counterIF;
					currentRing = (int) InputMatrix[counterIF][5];
					nRings++;
				}
			}
		}
		counterIF++;
	}
	fclose(inpfile);
	IDHash[nRings-1][2] = counterIF; // Write the max for last ring last ID.
	if (MakeHash == 1){ // Get dspacings from hkls.csv file
		FILE *hklf = fopen("hkls.csv","r");
		char aline2[2048];
		double ds;
		int rnr;
		while (fgets(aline2,2000,hklf)!=NULL){
			sscanf(aline2,"%s %s %s %lf %d %s %s %s %s %s %s", dummy, dummy,
				dummy, &ds, &rnr, dummy, dummy, dummy, dummy, dummy, dummy);
			for (i=0;i<nRings;i++){
				if (IDHash[i][0] == rnr){
					dspacings[i] = ds;
				}
			}
		}
		fclose(hklf);
	}
	for (j=0;j<NR_MAX_IDS_PER_GRAIN;j++) for (k=0;k<12;k++) SpotMatrix[j][k] = 0;
	int rowSpotID, startSpotMatrix;
	double RetVal, Eul[3];
	FILE *spotsfile = fopen("SpotMatrix.csv","w");
	fprintf(spotsfile, "%%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n");
	double **FinalMatrix;
	FinalMatrix = allocMatrix(nGrainPositions,47);
	for (i=0;i<nGrainPositions;i++){
		rown = GrainPositions[i];
		if (rown >= nrIDs){
			printf("Something is wrong. Please check.\n");
			return 1;
		}
		if (trackGrains == 0){
			DoneAlready = 0;
			for (j=0;j<cres;j++){
				if (IDsDone[j] == IDs[rown]){
					DoneAlready = 1;
				}
			}
			if (DoneAlready == 1){
				continue;
			}else{
				IDsDone[cres] = IDs[rown];
				cres++;
				if (cres >= (nGrainPositions*2)){
					printf("Something went wrong with cres %d out of %d alloc at %d. nGrains: %d Please check the ProcessGrains code.\n",cres,nGrainPositions, i,nGrains);
					return 1;
				}
			}
			for (k=0;k<9;k++){
				OR1[k] = OPs[rown][k];
			}
			OrientMat2Quat(OR1,q1);
			for (j=i+1;j<nGrainPositions;j++){
				rown2 = GrainPositions[j];
				if (rown2 >= nrIDs){
					printf("Something is wrong. Please check.\n");
					return 1;
				}
				int DA = 0;
				for (k=0;k<cres;k++){
					if (IDsDone[k] == IDs[rown2]){
						DA = 1;
					}
				}
				if (DA == 1) {
					printf("Here11!");
					continue;
				}
				for (k=0;k<9;k++){
					OR2[k] = OPs[rown2][k];
				}
				OrientMat2Quat(OR2,q2);
				Angle = GetMisOrientation(q1,q2,Axis,&ang,SGNr);
				DiffPos = sqrt((OPs[rown][9]- OPs[rown2][9])*( OPs[rown][9]- OPs[rown2][9])
							+ (OPs[rown][10]-OPs[rown2][10])*(OPs[rown][10]-OPs[rown2][10])
							+ (OPs[rown][11]-OPs[rown2][11])*(OPs[rown][11]-OPs[rown2][11]));
				if (ang < 0.1 && DiffPos < 5){
					IDsDone[cres] = IDs[rown2];
					cres++;
					if (cres >= (nGrainPositions*2)){
						printf("Something went wrong with cres %d out of %d allocated at %d. nGrains: %d Please check the ProcessGrains code.\n",cres,nGrainPositions, i,nGrains);
						return 1;
					}
				}
			}
		}
		if (OPs[rown][22] < 0.05){
			printf("Here!");
			continue;
		}

		nspots = NrIDsPerID[rown];
		OffSt = rown;
		OffSt *= 22;
		OffSt *= NR_MAX_IDS_PER_GRAIN;
		OffSt *= sizeof(double);
		ReadSize = 22*nspots*sizeof(double);
		int rc = pread(fullInfoFile,dummySampleInfo,ReadSize,OffSt);
		counterSpotMatrix = 0;
		startSpotMatrix = counterSpotMatrix;
		double GrainIDThis = (double)IDs[rown];
		for (j=0;j<nspots;j++){
			SpotsInfo[j][0] = dummySampleInfo[j*22+4];
			SpotsInfo[j][1] = dummySampleInfo[j*22+5];
			SpotsInfo[j][2] = dummySampleInfo[j*22+6];
			SpotsInfo[j][3] = dummySampleInfo[j*22+1];
			SpotsInfo[j][4] = dummySampleInfo[j*22+2];
			SpotsInfo[j][5] = dummySampleInfo[j*22+7];
			SpotsInfo[j][6] = dummySampleInfo[j*22+8];
			SpotsInfo[j][7] = dummySampleInfo[j*22+0]; // SpotID
			rowSpotID = (int) dummySampleInfo[j*22+0] - 1;
			if (rowSpotID >= counterIF){
				printf("Looking at the wrong info. Please check.\n");
				return 1;
			}
			SpotMatrix[counterSpotMatrix][0] = GrainIDThis; // GrainID
			SpotMatrix[counterSpotMatrix][1] = dummySampleInfo[j*22+0]; //SpotID
			SpotMatrix[counterSpotMatrix][2] = InputMatrix[rowSpotID][0]; //Omega
			SpotMatrix[counterSpotMatrix][3] = InputMatrix[rowSpotID][2]; //YRaw
			SpotMatrix[counterSpotMatrix][4] = InputMatrix[rowSpotID][3]; //ZRaw
			SpotMatrix[counterSpotMatrix][5] = InputMatrix[rowSpotID][9]; //OmeRaw
			SpotMatrix[counterSpotMatrix][6] = InputMatrix[rowSpotID][4]; //Eta
			SpotMatrix[counterSpotMatrix][7] = InputMatrix[rowSpotID][5]; //RingNr
			SpotMatrix[counterSpotMatrix][8] = InputMatrix[rowSpotID][6]; //YLab
			SpotMatrix[counterSpotMatrix][9] = InputMatrix[rowSpotID][7]; //ZLab
			SpotMatrix[counterSpotMatrix][10] = InputMatrix[rowSpotID][8]/2.0; //Theta
			counterSpotMatrix++;
		}
		LatticeParameterFit[0] = OPs[rown][12];
		LatticeParameterFit[1] = OPs[rown][13];
		LatticeParameterFit[2] = OPs[rown][14];
		LatticeParameterFit[3] = OPs[rown][15];
		LatticeParameterFit[4] = OPs[rown][16];
		LatticeParameterFit[5] = OPs[rown][17];
		Orient[0][0] = OPs[rown][0];
		Orient[0][1] = OPs[rown][1];
		Orient[0][2] = OPs[rown][2];
		Orient[1][0] = OPs[rown][3];
		Orient[1][1] = OPs[rown][4];
		Orient[1][2] = OPs[rown][5];
		Orient[2][0] = OPs[rown][6];
		Orient[2][1] = OPs[rown][7];
		Orient[2][2] = OPs[rown][8];
		CalcStrainTensorFableBeaudoin(LatCin,LatticeParameterFit,Orient,StrainTensorSampleFab);
		int retval = StrainTensorKenesei(nspots,SpotsInfo,Distance,wavelength,
			StrainTensorSampleKen,IDHash,dspacings,nRings,startSpotMatrix,SpotMatrix,&RetVal,StrainTensorSampleFab);
		for (j=0;j<counterSpotMatrix;j++){
			for (k=0;k<2;k++) fprintf(spotsfile,"%d\t",(int)SpotMatrix[j][k]);
			for (k=2;k<7;k++) fprintf(spotsfile,"%lf\t",SpotMatrix[j][k]);
			fprintf(spotsfile,"%d\t",(int)SpotMatrix[j][7]);
			for (k=8;k<12;k++) fprintf(spotsfile,"%lf\t",SpotMatrix[j][k]);
			fprintf(spotsfile,"\n");
		}
		if (retval == 0){
			printf("Did not read correct hash table for IDs. Exiting\n");
			return 1;
		}
		FinalMatrix[nGrains][0] = GrainIDThis;
		for (j=0;j<21;j++){
			FinalMatrix[nGrains][j+1] = OPs[rown][j];
		}
		FinalMatrix[nGrains][22] = Radiuses[rown];
		FinalMatrix[nGrains][23] = OPs[rown][22];
		for (j=0;j<3;j++){
			for (k=0;k<3;k++){
				FinalMatrix[nGrains][24+3*j+k] = MultR*StrainTensorSampleFab[j][k];
				FinalMatrix[nGrains][33+3*j+k] = MultR*StrainTensorSampleKen[j][k];
			}
		}
		FinalMatrix[nGrains][42] = MultR * RetVal;
		FinalMatrix[nGrains][43] = (double)PhaseNr;
		OrientMat2Euler(Orient,Eul);
		FinalMatrix[i][44] = Eul[0];
		FinalMatrix[i][45] = Eul[1];
		FinalMatrix[i][46] = Eul[2];
		VNorm = FinalMatrix[nGrains][22]*FinalMatrix[nGrains][22]*FinalMatrix[nGrains][22];
		BeamCenter += (FinalMatrix[nGrains][12])*(VNorm);
		FullVol += VNorm;
		nGrains++;
	}
	printf("Number of grains: %d.\n",nGrains);
	BeamCenter /= FullVol;
	// Write file
	fclose(spotsfile);
	char GrainsFileName[1024];
	sprintf(GrainsFileName,"Grains.csv");
	FILE *GrainsFile;
	GrainsFile = fopen(GrainsFileName,"w");
	if (GrainsFile == NULL) {
		printf("Could not write to Grains.csv. Please check.\n");
		return 1;
	}
	fprintf(GrainsFile,"%%NumGrains %d\n",nGrains);
	fprintf(GrainsFile, "%%BeamCenter %f\n",BeamCenter);
	fprintf(GrainsFile, "%%BeamThickness %f\n",BeamThickness);
	fprintf(GrainsFile, "%%GlobalPosition %f\n",GlobalPosition);
	fprintf(GrainsFile, "%%NumPhases %d\n",NumPhases);
	fprintf(GrainsFile, "%%PhaseInfo\n%%\tSpaceGroup:%d\n",SGNr);
	fprintf(GrainsFile, "%%\tLattice Parameter: %lf %lf %lf %lf %lf %lf\n", LatCin[0], LatCin[1],
							LatCin[2], LatCin[3], LatCin[4], LatCin[5]);
	fprintf(GrainsFile,"%%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\ta\tb"
						"\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfidence\t");
	fprintf(GrainsFile,"eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\teFab32\teFab33\t");
	fprintf(GrainsFile,"eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\tRMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n");
	for (i=0;i<nGrains;i++){
		fprintf(GrainsFile,"%d\t",(int)FinalMatrix[i][0]);
		for (j=1;j<47;j++){
			fprintf(GrainsFile,"%lf\t",FinalMatrix[i][j]);
		}
		fprintf(GrainsFile,"\n");
	}
	fclose(GrainsFile);
    end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
