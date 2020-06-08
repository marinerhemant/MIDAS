//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

// Code to take output from previous "stage" and check for same grains in
// the next stage
//
//
// Author: Hemant Sharma
//
//
// Dated: 2016/06/16
//
//
// Data Model: (loop over grains)
// 0.	Read GrainsOld.csv (first line) to get idea about nGrains.
// 1.	Read SpotMatrixOld.csv -> Pick spots for a grain and
//		the positions (y,z,ome,RingNr).
// 2.	Read Data.bin (row number in Spots.bin), nData.bin (number of
//		spots in a certain bin in Data.bin) and Spots.bin.
//			NOTE: All 3 bin files are for
// 3.	Calculate best match, based on Internal Angle.
// 4.	Write out a .csv file for grain having IDs matched, abc, alpha,
//		beta, gamma, position, orientation, radius per spot.
// 5.	Add a random ID to SpotsToIndex.csv for use during optimization.
// 6.	Write a csv file to match previous IDs with new IDs.


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
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>

#define MAX_LINE_LENGTH 4096
#define RealType double
#define MAX_N_RINGS 500       // max nr of rings that can be stored (applies to the arrays ringttheta, ringhkl, etc)
#define MAX_N_SPOTS 5000
#define MAX_N_MATCHES 200
// conversions constants
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))

static inline void Convert9To3x3(double MatIn[9],double MatOut[3][3]){int i,j,k=0;for (i=0;i<3;i++){for (j=0;j<3;j++){MatOut[i][j] = MatIn[k];k++;}}}
static inline void Convert3x3To9(double MatIn[3][3],double MatOut[9]){int i,j; for (i=0;i<3;i++) for (j=0;j<3;j++) MatOut[(i*3)+j] = MatIn[i][j];}
static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}
static inline double sin_cos_to_angle (double s, double c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}

static inline
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

//Global variables
int *data;
int *ndata;
RealType *ObsSpotsLab;

static void
check (int test, const char * message, ...){
    if (test) {
        va_list args;
        va_start (args, message);
        vfprintf (stderr, message, args);
        va_end (args);
        fprintf (stderr, "\n");
        exit (EXIT_FAILURE);
    }
}

static inline
double**
allocMatrix(int nrows, int ncols){
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

int ReadBins(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "/dev/shm/Data.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    data = mmap (0, size, PROT_READ, MAP_SHARED, fd, 0);
    check (data == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));

    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "/dev/shm/nData.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    ndata = mmap (0, size2, PROT_READ, MAP_SHARED, fd2, 0);
    check (ndata == MAP_FAILED, "mmap %s failed: %s",file_name, strerror (errno));
	return 1;
}

int ReadSpots(){
	int fd;
	struct stat s;
	int status;
	size_t size;
	const char *filename = "/dev/shm/Spots.bin";
	int rc;
	fd = open(filename,O_RDONLY);
	check(fd < 0, "open %s failed: %s", filename, strerror(errno));
	status = fstat (fd , &s);
	check (status < 0, "stat %s failed: %s", filename, strerror(errno));
	size = s.st_size;
	ObsSpotsLab = mmap(0,size,PROT_READ,MAP_SHARED,fd,0);
	check (ObsSpotsLab == MAP_FAILED,"mmap %s failed: %s", filename, strerror(errno));
	return (int) size/(9*sizeof(double));
}

int UnMap(){
	int fd;
    struct stat s;
    int status;
    size_t size;
    const char * file_name = "/dev/shm/Data.bin";
    int rc;
    fd = open (file_name, O_RDONLY);
    check (fd < 0, "open %s failed: %s", file_name, strerror (errno));
    status = fstat (fd, & s);
    check (status < 0, "stat %s failed: %s", file_name, strerror (errno));
    size = s.st_size;
    rc = munmap (data,size);
    int fd2;
    struct stat s2;
    int status2;
    const char* file_name2 = "/dev/shm/nData.bin";
    fd2 = open (file_name2, O_RDONLY);
    check (fd2 < 0, "open %s failed: %s", file_name2, strerror (errno));
    status2 = fstat (fd2, & s2);
    check (status2 < 0, "stat %s failed: %s", file_name2, strerror (errno));
    size_t size2 = s2.st_size;
    rc = munmap (ndata,size2);

	int fd3;
	struct stat s3;
	int status3;
	const char *filename3 = "/dev/shm/Spots.bin";
	fd3 = open(filename3,O_RDONLY);
	check(fd3 < 0, "open %s failed: %s", filename3, strerror(errno));
	status3 = fstat (fd3 , &s3);
	check (status3 < 0, "stat %s failed: %s", filename3, strerror(errno));
	size_t size3 = s3.st_size;
	rc = munmap(ObsSpotsLab,size3);
	return 1;
}

static inline
void SpotToGv(double xi, double yi, double zi, double Omega, double theta, double *g1, double *g2, double *g3)
{
	double CosOme = cosd(Omega), SinOme = sind(Omega), eta = CalcEtaAngle(yi,zi), TanEta = tand(-eta), SinTheta = sind(theta);
    double CosTheta = cosd(theta), CosW = 1, SinW = 0, k3 = SinTheta*(1+xi)/((yi*TanEta)+zi), k2 = TanEta*k3, k1 = -SinTheta;
    if (eta == 90){
		k3 = 0;
		k2 = -CosTheta;
	} else if (eta == -90){
		k3 = 0;
		k2 = CosTheta;
	}
    double k1f = (k1*CosW) + (k3*SinW), k3f = (k3*CosW) - (k1*SinW), k2f = k2;
    *g1 = (k1f*CosOme) + (k2f*SinOme);
    *g2 = (k2f*CosOme) - (k1f*SinOme);
    *g3 = k3f;
}

int main(int argc, char *argv[]) // Arguments: OldFolder, NewFolder, ParametersFile
{
	int i,j,k;
	char *oldFolder, *ParamsFN;
	oldFolder = argv[1];
	ParamsFN = argv[2];
	char GrainsOldFN[MAX_LINE_LENGTH], spotsMatrixOldFN[MAX_LINE_LENGTH];
	sprintf(GrainsOldFN,"%s/Grains.csv",oldFolder);
	printf("Grains file: %s\n",GrainsOldFN); fflush(stdout);
	char aline[MAX_LINE_LENGTH];
	FILE *GrainsFile = fopen(GrainsOldFN,"r");
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	int nGrains;
	char dummy[MAX_LINE_LENGTH];
	sscanf(aline, "%s %d",dummy, &nGrains);
	int GrainIDsOld[nGrains];
	double **GrainInfo;
	GrainInfo = allocMatrix(nGrains,19);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	fgets(aline,MAX_LINE_LENGTH,GrainsFile);
	int grainNr=0;
	while(fgets(aline,MAX_LINE_LENGTH,GrainsFile)!=NULL){
		sscanf(aline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
			"%lf %lf %lf %lf %lf %lf",&GrainIDsOld[grainNr],
			&GrainInfo[grainNr][0],&GrainInfo[grainNr][1],&GrainInfo[grainNr][2],
			&GrainInfo[grainNr][3],&GrainInfo[grainNr][4],&GrainInfo[grainNr][5],
			&GrainInfo[grainNr][6],&GrainInfo[grainNr][7],&GrainInfo[grainNr][8],
			&GrainInfo[grainNr][9],&GrainInfo[grainNr][10],&GrainInfo[grainNr][11],
			&GrainInfo[grainNr][12],&GrainInfo[grainNr][13],&GrainInfo[grainNr][14],
			&GrainInfo[grainNr][15],&GrainInfo[grainNr][16],&GrainInfo[grainNr][17]);
		grainNr++;
	}
	if (grainNr != nGrains){
		printf("Number of grains from Grains.csv file do not match.\nExiting\n.");
		return 1;
	}
	// Read bin files
	int n_spots = ReadSpots();
	int rc = ReadBins();
	// Necessary parameters: EtaBinSize, OmeBinSize
	FILE *ParamsFile = fopen(ParamsFN,"r");
    int LowNr;
	char *str;
	double etabinsize, omebinsize,Distance;
	int n_eta_bins, n_ome_bins;
	char outfolder[MAX_LINE_LENGTH];
	while(fgets(aline,MAX_LINE_LENGTH,ParamsFile)!=NULL){
		str = "EtaBinSize ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &etabinsize);
			continue;
		}
        str = "OmeBinSize ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &omebinsize);
			continue;
		}
        str = "Distance ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %lf", dummy, &Distance);
			continue;
		}
        str = "OutputFolder ";
		LowNr = strncmp(aline, str, strlen(str));
		if (LowNr == 0) {
			sscanf(aline, "%s %s", dummy, outfolder);
			continue;
		}
	}
	n_ome_bins = ceil(360.0 / omebinsize);
	n_eta_bins = ceil(360.0 / etabinsize);

	// Go through all grains, reading Ome, RingNr, Eta
	sprintf(spotsMatrixOldFN,"%s/SpotMatrix.csv",oldFolder);
	FILE *SpotMatrixFile = fopen(spotsMatrixOldFN,"r");
	fgets(aline,MAX_LINE_LENGTH,SpotMatrixFile);
	fgets(aline,MAX_LINE_LENGTH,SpotMatrixFile);
	int spotNr = 0;
	double Omegas[MAX_N_SPOTS],Etas[MAX_N_SPOTS], YLab[MAX_N_SPOTS], ZLab[MAX_N_SPOTS],Thetas[MAX_N_SPOTS];
	int RingNrs[MAX_N_SPOTS];
	int ID;
	int iRing, iOme, iEta, iSpot, bestID;
	long long int Pos, nspots, DataPos, spotRow;
	char *str2;
	double g01,g02,g03;
	double g11,g12,g13;
	double y1, z1, ome1, theta1, bestRadius;
	double minAngle = 100000, lenK, NormG0, NormG1, Angle, DotGs;
	int IDs[MAX_N_SPOTS];
	double Rads[MAX_N_SPOTS];
	char outfilename[MAX_LINE_LENGTH];
	char *spotsfilename = "SpotsToIndex.csv";
	char *GrainMatchesFileName = "GrainMatches.csv";
	FILE *GrainMatchesFile = fopen(GrainMatchesFileName,"w");
	fprintf(GrainMatchesFile,"OldID\tNewID\n");
	FILE *spotsfile = fopen(spotsfilename,"w");
	int GrainID,nrFilled;
	double mult;
	for (i=0;i<nGrains;i++){
		printf("Trying to track %d grain out of %d grains.\n",i+1,nGrains);
		do{ // Check for both EOF and ID matching GrainID
			sscanf(aline,"%d %s %s %$s %s %s %s",&ID,dummy,dummy,dummy,dummy,dummy,dummy);
			if (ID != GrainIDsOld[i]){
				break;
			}
			sscanf(aline,"%s %s %lf %s %s %s %lf %d %lf %lf %lf %s",dummy, dummy, &Omegas[spotNr],dummy, dummy, dummy, &Etas[spotNr],&RingNrs[spotNr],&YLab[spotNr],&ZLab[spotNr],&Thetas[spotNr],dummy);
			spotNr ++;
		}while(fgets(aline,MAX_LINE_LENGTH,SpotMatrixFile)!=NULL);
		if (spotNr == 0) continue;
		printf("Nr of spots for grain %d: %d\n",i+1, spotNr);
		nrFilled = 0;
		for (j=0;j<spotNr;j++){
			lenK = CalcNorm3(Distance,YLab[j],ZLab[j]);
			SpotToGv(Distance/lenK,YLab[j]/lenK,ZLab[j]/lenK,Omegas[j],Thetas[j],&g01,&g02,&g03);
			NormG0 = CalcNorm3(g01,g02,g03);
			iRing = RingNrs[j] - 1;
			iOme = floor((180+Omegas[j])/omebinsize);
			iEta = floor((180+Etas[j])/etabinsize);
			Pos = iRing*n_eta_bins*n_ome_bins + iEta*n_ome_bins + iOme;
			nspots = ndata[Pos*2];
			if (nspots == 0) continue;
			DataPos = ndata[Pos*2+1];
			minAngle = 100000;
			for ( iSpot = 0 ; iSpot < nspots; iSpot++ ) { // For each potential match, calculate angle between gvectors
				spotRow = data[DataPos + iSpot];
				y1 = ObsSpotsLab[spotRow*9+0];
				z1 = ObsSpotsLab[spotRow*9+1];
				ome1 = ObsSpotsLab[spotRow*9+2];
				theta1 = ObsSpotsLab[spotRow*9+7] / 2.0;
				lenK = CalcNorm3(Distance,y1,z1);
				SpotToGv(Distance/lenK,y1/lenK,z1/lenK,ome1,theta1,&g11,&g12,&g13);
				NormG1 = CalcNorm3(g11,g12,g13);
				DotGs = (g01*g11) + (g02*g12) + (g03*g13);
				mult = DotGs/(NormG0*NormG1);
				if (mult > 1) mult = 1;
				if (mult < -1) mult = -1;
				Angle = fabs(acosd(mult));
				//~ Angle = fabs(acosd(DotGs/(NormG0*NormG1)));
				if (Angle < minAngle){
					minAngle = Angle;
					bestID = (int) ObsSpotsLab[spotRow*9 + 4];
					//~ printf("%d ",bestID);
					bestRadius = ObsSpotsLab[spotRow*9+3];
				}
			}
			//~ printf("%d\n",bestID);
			IDs[nrFilled] = bestID;
			Rads[nrFilled] = bestRadius;
			nrFilled ++;
		}
		GrainInfo[i][18] = (double)nrFilled/ (double)spotNr;
		// Write CSV files and we are done.
		//~ for (iSpot=0;iSpot<n_spots;iSpot++) printf("%d\n",(int)ObsSpotsLab[iSpot*9+4]);
		GrainID = IDs[0];
		sprintf(outfilename,"%s/BestPos_%09d.csv",outfolder,GrainID);
		FILE *outfile = fopen(outfilename,"w");
		printf("GrainID %d\n",GrainID);
		fprintf(spotsfile,"%d\n",GrainID);
		fprintf(outfile,"%d\n",GrainID);
		fprintf(outfile,"%lf, %lf, %lf, %lf, %lf, %lf\n",GrainInfo[i][12],
			GrainInfo[i][13],GrainInfo[i][14],GrainInfo[i][15],GrainInfo[i][16],
			GrainInfo[i][17]);
		fprintf(outfile,"%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
			GrainInfo[i][18],GrainInfo[i][0],GrainInfo[i][1],GrainInfo[i][2],GrainInfo[i][3],
			GrainInfo[i][4],GrainInfo[i][5],GrainInfo[i][6],GrainInfo[i][7],GrainInfo[i][8],
			GrainInfo[i][9],GrainInfo[i][10],GrainInfo[i][11]);
		for (j=0;j<spotNr;j++){
			fprintf(outfile,"%d %lf\n",IDs[j],Rads[j]);
		}
		fprintf(GrainMatchesFile,"%d\t%d\n",GrainIDsOld[i],GrainID);
		spotNr = 0;
		fclose(outfile);
	}
}
