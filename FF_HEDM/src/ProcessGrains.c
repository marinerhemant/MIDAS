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

#define MAX_N_IDS 6000000
#define NR_MAX_IDS_PER_GRAIN 5000
#define IAColNr 20 // 20 for Internal Angle, 18 for position, 19 for omega

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

static inline
int
FindInternalAnglesTwins(int nrIDs, int *IDs, int *IDsPerGrain,
				   int *NrIDsPerID, bool *IDsChecked,
				   double **OPs, double *ID_IA_Mat, int counter, int Pos,
				   int StartingID, double *Radiuses, int SGNr)
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
	for (i=0;i<NrIDsPerID[Pos];i++){
		ThisID = IDsPerGrain[(Pos*NR_MAX_IDS_PER_GRAIN)+i];
		for (j=0;j<nrIDs;j++){
			ThisID2 = IDs[j];
			if (ThisID != ThisID2 && IDsChecked[j] == false){
				for (k=0;k<9;k++){
					OR2[k] = OPs[j][k];
				}
				OrientMat2Quat(OR2,q2);
				Angle = GetMisOrientation(q1,q2,Axis,&ang,SGNr);
				//printf("%lf %lf %lf %lf\n",ang,Axis[0],Axis[1],Axis[2]);
				if (fabs(ang) < 0.1 || 
					(fabs(ang - 60) < 0.1 && 
					 fabs(Axis[0])-fabs(Axis[1]) < 0.01 && 
					 fabs(Axis[2])-fabs(Axis[1]) < 0.01)){
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
				if (fabs(ang) < 0.1){
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

int main(int argc, char *argv[])
{
	if (argc != 2){
		printf("Usage: ProcessGrains ParameterFile\n");
		return;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
    char IDsFileName[1024], BestFileName[1024], ResultFileName[1024];
    FILE *IDsFile, *BestFile, *ResultFile;
    sprintf(IDsFileName,"SpotsToIndex.csv");
    printf("Reading IDs file: %s\n",IDsFileName);
    IDsFile = fopen(IDsFileName,"r");
    char line[5024];
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char aline[1000];
    fileParam = fopen(ParamFN,"r");
    char *str, dummy[1000];
    int LowNr;
    int Twin, MinNrSpots, SGNr;
    double Distance, wavelength, LatCin[6];
    double BeamThickness = 0, GlobalPosition = 0;
    int NumPhases = 1, PhaseNr = 1;
    while (fgets(aline,1000,fileParam)!=NULL){
        str = "Twins ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &Twin);
            continue;
		}
		str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &wavelength);
            continue;
		}
		str = "BeamThickness ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &BeamThickness);
            continue;
		}
		str = "GlobalPosition ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &GlobalPosition);
            continue;
		}
		str = "NumPhases ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &NumPhases);
            continue;
		}
		str = "PhaseNr ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &PhaseNr);
            continue;
		}
		str = "MinNrSpots ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &MinNrSpots);
            continue;
		}
		str = "SpaceGroup ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %d", dummy, &SGNr);
            continue;
		}
		str = "Lsd ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &Distance);
            continue;
		}
		str = "Wavelength ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf", dummy, &wavelength);
            continue;
		}
		str = "LatticeConstant ";
        LowNr = strncmp(aline,str,strlen(str));
        if (LowNr==0){
            sscanf(aline,"%s %lf %lf %lf %lf %lf %lf", dummy, 
						&LatCin[0], &LatCin[1], &LatCin[2],
						&LatCin[3], &LatCin[4], &LatCin[5]);
            continue;
		}
	}

    
	int i,j,k,ThisID,counter;
    int *IDs;
    IDs = malloc(MAX_N_IDS*sizeof(*IDs));
    for (i=0;i<MAX_N_IDS;i++) IDs[i] = 0;
    int nrIDs=0;
    if (IDsFile == NULL)printf("Could not open spots file.\n");
    while (fgets(line,5024,IDsFile) != NULL){
		sscanf(line,"%d",&IDs[nrIDs]);
		nrIDs++;
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
	IDsPerGrain = malloc(NR_MAX_IDS_PER_GRAIN*nrIDs*sizeof(*IDsPerGrain));
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
		printf("Key file was not found. Exiting.\n");
		return 1;
	}
	if (fileOPFit == NULL){
		printf("OrientPos file was not found. Exiting.\n");
		return 1;
	}
	if (fileProcessKey == NULL){
		printf("ProcessKey file was not found. Exiting.\n");
		return 1;
	}
	int *keyID;
	keyID = malloc(2*sizeof(*keyID));
	size_t readKey, readOP, readProcess;
	readProcess = fread(IDsPerGrain,NR_MAX_IDS_PER_GRAIN*nrIDs*sizeof(int),1,fileProcessKey);
	for (i=0;i<nrIDs;i++){
		readKey = fread(keyID,2*sizeof(int),1,fileKey);
		if (keyID[0] == 0){
			IDsToKeep[i] = false;
		}
		IDsToKeep[i] = true;
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
	int StartingID,ThisID1,ThisID2;
	int nGrainPositions = 0,BestGrainPos;
	int *GrainPositions,*nGrainsMatched;
	GrainPositions = malloc(nrIDs*sizeof(*GrainPositions));
	nGrainsMatched = malloc(nrIDs*sizeof(*nGrainsMatched));
	double minIA,maxRadThis;
	printf("Read all grain files.\n");
	bool *IDsChecked;
	IDsChecked = malloc(MAX_N_IDS*sizeof(*IDsChecked));
	for (i=0;i<MAX_N_IDS;i++) IDsChecked[i] = false;
	for (i=0;i<nrIDs;i++){
		GrainPositions[i] = 0;
		nGrainsMatched[i] = 0;
		if (IDsToKeep[i] == false){
			IDsChecked[i] = true;
		}
	}
	double *ID_IA_MAT;
	double ang, Angle, Axis[3],DiffPos,OR1[9],q1[9],OR2[9],q2[4];
	int counte,counten,totcount=0;
	ID_IA_MAT = malloc(50000*4*sizeof(*ID_IA_MAT));
	for (i=0;i<50000*4;i++) ID_IA_MAT[i] = 0;
	for (i=0;i<nrIDs;i++){
		if (i%1000 == 0) printf("Processed %d of %d IDs.\n",i,nrIDs);
		if (IDsChecked[i] == false){
			counte = 0;
			StartingID = IDs[i];
			maxRadThis = Radiuses[i];
			minIA = OPs[i][IAColNr];
			BestGrainPos = i;
			if (Twin ==0){
				counten = FindInternalAngles(nrIDs,IDs,IDsPerGrain,NrIDsPerID,
				IDsChecked,OPs,ID_IA_MAT,counte,i,StartingID,Radiuses,SGNr);
				printf("%d\n",counten);
			}else{
				counten = FindInternalAnglesTwins(nrIDs,IDs,IDsPerGrain,NrIDsPerID,
				IDsChecked,OPs,ID_IA_MAT,counte,i,StartingID,Radiuses,SGNr);
				printf("%d\n",counten);
			}
			totcount+=counten;
			nGrainsMatched[i] = counten;
			if (counten < MinNrSpots) continue;
			for (j=0;j<counten;j++){
				if (ID_IA_MAT[(j*4)+2] < minIA){
					minIA = ID_IA_MAT[(j*4)+2];
					BestGrainPos = (int)ID_IA_MAT[(j*4)+1];
					maxRadThis = ID_IA_MAT[(j*4)+3];
				}
			}
			GrainPositions[nGrainPositions] = BestGrainPos;
			Radiuses[BestGrainPos] = maxRadThis;
			nGrainPositions ++;
		}
	}
	
	//Write out
	char GrainsFileName[1024];
	sprintf(GrainsFileName,"Grains.csv");
	FILE *GrainsFile;
	GrainsFile = fopen(GrainsFileName,"w");
	int nGrains=0;
	int *IDsDone;
	IDsDone = malloc(nGrainPositions*sizeof(*IDsDone));
	int cres=0;
	int DoneAlready = 0;
	double StrainTensorSampleKen[3][3];
	double StrainTensorSampleFab[3][3];
	double *dummySampleInfo;
	dummySampleInfo = malloc(22*NR_MAX_IDS_PER_GRAIN*sizeof(*dummySampleInfo));
	double LatticeParameterFit[6],Orient[3][3],SpotsInfo[NR_MAX_IDS_PER_GRAIN][8];
	int nspots, rown;
	// Calculate Strains Now
	int fullInfoFile = open("Output/FitBest.bin",O_RDONLY);
	int OffSt, ReadSize;
	double MultR=1000000;
	double **FinalMatrix;
	double BeamCenter = 0, FullVol = 0,VNorm;
	FinalMatrix = allocMatrix(nGrainPositions,43);
	int rown2;
	int IDHash[NR_MAX_IDS_PER_GRAIN][3];
	double dspacings[NR_MAX_IDS_PER_GRAIN];
	int nRings=0;
	char *hashfn = "IDsHash.csv";
	FILE *hashfile = fopen(hashfn,"r");
	while (fgets(aline,2000,hashfile)!=NULL){
		sscanf(aline,"%d %d %d %lf",&IDHash[nRings][0],&IDHash[nRings][1],&IDHash[nRings][2],&dspacings[nRings]);
		nRings++;
	}
	double **SpotMatrix, **InputMatrix;
	SpotMatrix = allocMatrix(NR_MAX_IDS_PER_GRAIN*nGrainPositions,5);
	InputMatrix = allocMatrix(MAX_N_IDS,4);
	int counterSpotMatrix = 0;
	char *inputallfn = "InputAllExtraInfoFittingAll.csv";
	FILE *inpfile = fopen(inputallfn,"r");
	int counterIF=0;
	FILE *spotsfile = fopen("SpotMatrix.csv","w");
	fgets(aline,2000,inpfile);
	while (fgets(aline,2000,inpfile)!=NULL){
		sscanf(aline,"%s %s %lf %s %lf %s %s %s %s %s %s %lf %lf %s",dummy, dummy, &InputMatrix[counterIF][0],
			dummy, &InputMatrix[counterIF][1], dummy, dummy, dummy, dummy, dummy, dummy, &InputMatrix[counterIF][2],
			&InputMatrix[counterIF][3],dummy);
		if ((int)InputMatrix[counterIF][1] != counterIF+1){
			printf("IDs dont match.\nExiting\n"); 
			return(1);
		}
		counterIF++;
	}
	int rowSpotID;
	for (i=0;i<nGrainPositions;i++){
		rown = GrainPositions[i];
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
		}
		for (k=0;k<9;k++){
			OR1[k] = OPs[rown][k];
		}
		OrientMat2Quat(OR1,q1);
		for (j=i+1;j<nGrainPositions;j++){
			rown2 = GrainPositions[j];
			for (k=0;k<9;k++){
				OR2[k] = OPs[rown2][k];
			}
			OrientMat2Quat(OR2,q2);
			Angle = GetMisOrientation(q1,q2,Axis,&ang,SGNr);
			DiffPos = sqrt((OPs[rown][9]-OPs[rown2][9])*(OPs[rown][9]-OPs[rown2][9]) 
						 + (OPs[rown][10]-OPs[rown2][10])*(OPs[rown][10]-OPs[rown2][10]) 
						 + (OPs[rown][11]-OPs[rown2][11])*(OPs[rown][11]-OPs[rown2][11]));
			if (ang < 0.1 && DiffPos < 5){
				IDsDone[cres] = IDs[rown2];
				cres++;
			}
		}
		if (OPs[rown][22] < 0.1) continue;
		
		nspots = NrIDsPerID[rown];
		OffSt = rown*22*NR_MAX_IDS_PER_GRAIN*sizeof(double);
		ReadSize = 22*nspots*sizeof(double);
		int rc = pread(fullInfoFile,dummySampleInfo,ReadSize,OffSt);
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
			SpotMatrix[counterSpotMatrix][0] = (double)IDs[rown];
			SpotMatrix[counterSpotMatrix][1] = dummySampleInfo[j*22+0];
			SpotMatrix[counterSpotMatrix][2] = InputMatrix[rowSpotID][0];
			SpotMatrix[counterSpotMatrix][3] = InputMatrix[rowSpotID][2];
			SpotMatrix[counterSpotMatrix][4] = InputMatrix[rowSpotID][3];
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
		int retval = StrainTensorKenesei(nspots,SpotsInfo,Distance,wavelength,StrainTensorSampleKen,IDHash,dspacings,nRings);
		if (retval == 0){
			printf("Did not read correct hash table for IDs. Exiting\n");
			return;
		}
		CalcStrainTensorFableBeaudoin(LatCin,LatticeParameterFit,Orient,StrainTensorSampleFab);
		FinalMatrix[nGrains][0] = (double)IDs[rown];
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
		FinalMatrix[nGrains][42] = (double)PhaseNr;
		VNorm = FinalMatrix[nGrains][22]*FinalMatrix[nGrains][22]*FinalMatrix[nGrains][22];
		BeamCenter += (FinalMatrix[nGrains][12])*(VNorm);
		FullVol += VNorm;
		nGrains++;
	}
	printf("Number of grains: %d.\n",nGrains);
	BeamCenter /= FullVol;
	// Write file
	fprintf(spotsfile, "%%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\n");
	for (i=0;i<counterSpotMatrix;i++){
		fprintf(spotsfile,"%d\t%d\t%lf\t%lf\t%lf\n",(int)SpotMatrix[i][0],(int)SpotMatrix[i][1],
			SpotMatrix[i][2],SpotMatrix[i][3],SpotMatrix[i][4]);
	}
	fclose(spotsfile);
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
	fprintf(GrainsFile,"eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\tPhaseNr\n");
	for (i=0;i<nGrains;i++){
		fprintf(GrainsFile,"%d\t",(int)FinalMatrix[i][0]);
		for (j=1;j<43;j++){
			fprintf(GrainsFile,"%lf\t",FinalMatrix[i][j]);
		}
		fprintf(GrainsFile,"\n");
	}
    end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
