//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//
//
// CalcRadius.c
//
//
// Created by Hemant Sharma on 2024/02/27
//
//


#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/types.h>
#include <errno.h>
#include <blosc2.h>
#include <stdlib.h> 
#include <zip.h> 

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y)   sqrt((x)*(x) + (y)*(y))
#define MAXNRINGS 500

#define MAX_N_SPOTS 2000000

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
double CalcEtaAngle(double y, double z){
	double alpha = rad2deg*acos(z/sqrt(y*y+z*z));
	if (y>0) alpha = -alpha;
	return alpha;
}

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

int main(int argc, char *argv[]){
	if (argc < 2){
		printf("Usage:\n %s ZarrZip (optional)ResultFolder\n",argv[0]);
		return 1;
	}
	clock_t start, end;
    double diftotal;
    start = clock();
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
    char *resultFolder;
    int skipFrame=0;

    // Read params file.
    char aline[1000], *str, dummy[1000];
    char *Folder = NULL;
    int StartNr=1, EndNr, LayerNr;
    double Ycen, Zcen, OmegaStep, OmegaFirstFile, Lsd, px, Wavelength,Rsample,Hbeam;
    double PowderIntIn = 0;
    int DiscModel = 0;
    double DiscArea = 0, Vsample = 0, width=-1,widthOrig;
    int locRingThresh, nRings=0;
    while ((zip_stat_index(arch, count, 0, finfo)) == 0) {
        if (strstr(finfo->name,"analysis/process/analysis_parameters/RingThresh/0.0")!=NULL){
            locRingThresh = count;
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/RingThresh/.zarray")!=NULL){
            s = calloc(finfo->size + 1, sizeof(char));
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, s, finfo->size);
            char *ptr = strstr(s,"shape");
            if (ptr != NULL){
                char *ptrt = strstr(ptr,"[");
                char *ptr2 = strstr(ptrt,"]");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[2048];
                strncpy(ptr3,ptrt,loc+1);
                sscanf(ptr3,"%*[^0123456789]%d",&nRings);
            } else return 1;
            free(s);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/ResultFolder/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = 4096;
            resultFolder = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,resultFolder,dsize);
            resultFolder[dsize] = '\0';
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/DiscModel/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            DiscModel = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/DiscArea/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            DiscArea = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Lsd/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Lsd = *(double *)&data[0];
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
            Wavelength = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Vsample/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Vsample = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Hbeam/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Hbeam = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Rsample/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Rsample = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/PixelSize/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            px = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"measurement/process/scan_parameters/step/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            OmegaStep = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/WidthTthPx/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            width = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/Width/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            widthOrig = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/YCen/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Ycen = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/ZCen/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(double);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            Zcen = *(double *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/ResultFolder/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = 4096;
            Folder = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,Folder,dsize);
            Folder[dsize] = '\0';
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"exchange/data/.zarray")!=NULL){
            s = calloc(finfo->size + 1, sizeof(char));
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, s, finfo->size);
            char *ptr = strstr(s,"shape");
            if (ptr != NULL){
                char *ptrt = strstr(ptr,"[");
                char *ptr2 = strstr(ptrt,"]");
                int loc = (int)(ptr2 - ptrt);
                char ptr3[2048];
                strncpy(ptr3,ptrt,loc+1);
                sscanf(ptr3,"%*[^0123456789]%d",&EndNr);
            } else return 1;
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/LayerNr/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            LayerNr = *(int *)&data[0];
            free(arr);
            free(data);
        }
        if (strstr(finfo->name,"analysis/process/analysis_parameters/SkipFrame/0")!=NULL){
            arr = calloc(finfo->size + 1, sizeof(char)); 
            fd = zip_fopen_index(arch, count, 0);
            zip_fread(fd, arr, finfo->size);
            dsize = sizeof(int);
            data = (char*)malloc((size_t)dsize);
            dsize = blosc1_decompress(arr,data,dsize);
            skipFrame = *(int *)&data[0];
            free(arr);
            free(data);
        }
        count++;
    }
    if (width==-1) width = widthOrig;
	if (argc==3) Folder = argv[2];
	if (argc==3) resultFolder = argv[2];
    EndNr = EndNr - skipFrame; // This ensures we don't over-read.
	int RingNrs[nRings];
    double Thresholds[nRings];
    zip_stat_index(arch, locRingThresh, 0, finfo);
    s = calloc(finfo->size + 1, sizeof(char));
    fd = zip_fopen_index(arch, locRingThresh, 0);
    zip_fread(fd, s, finfo->size); 
    dsize = nRings*2*sizeof(double);
    data = (char*)malloc((size_t)dsize);
    dsize = blosc1_decompress(s,data,dsize);
    int iter;
    for (iter=0;iter<nRings;iter++){
        RingNrs[iter]    = (int) *(double *)&data[(iter*2+0)*sizeof(double)];
    }
    free(s);
    free(data);

    int TopLayer=0;

	char InputFile[2048];
    sprintf(InputFile,"%s/Result_StartNr_%d_EndNr_%d.csv",Folder,StartNr,EndNr);
	FILE *Infile;
	Infile = fopen(InputFile,"r");
    if (Infile == NULL){
		printf("Could not read file %s\n",InputFile);
		return 1;
	}
	fgets(aline,1000,Infile);
	int counter = 0, RingNr;
	double **SpotsMat;
	SpotsMat = allocMatrix(MAX_N_SPOTS,16);
	double PowderInt[nRings];
	int i;
	char hklfn[2048];
    sprintf(hklfn,"%s/hkls.csv",resultFolder);
	FILE *hklf = fopen(hklfn,"r");
	int mhkl[nRings];
	for (i=0;i<nRings;i++){
		mhkl[i] = 0;
		PowderInt[i] = 0;
	}
	int RN;
	fgets(aline,1000,hklf);
	double RingRads[nRings], rrd;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline, "%s %s %s %s %d %s %s %s %s %s %lf",dummy, dummy, dummy, dummy,
				&RN, dummy, dummy, dummy, dummy, dummy, &rrd);
		for (i=0;i<nRings;i++){
			RingNr = RingNrs[i];
			if (RN == RingNr){
				RingRads[i] = rrd;
				mhkl[i]++;
				break;
			}
		}
	}
    char header[2048] = "SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px)"
					" IMax MinOme(degrees) MaxOme(degress) Radius(px) Theta(degrees) Eta(degrees)"
					" DeltaOmega NImgs RingNr GrainVolume GrainRadius PowderIntensity SigmaR SigmaEta NrPx NrPxTot\n";
	char OutFile[2048];
	sprintf(OutFile,"%s/Radius_StartNr_%d_EndNr_%d.csv",Folder,StartNr,EndNr);
	FILE *outfile;
	outfile = fopen(OutFile,"w");
	fprintf(outfile,"%s",header);
	double **Sigmas;
	Sigmas = allocMatrix(MAX_N_SPOTS,2);
	double **NrPx;
	NrPx = allocMatrix(MAX_N_SPOTS,2);
	double MinOme=100000, MaxOme=-100000;
	int thisRings[nRings][2];
	double tempArr[13],dummyDouble;
    int found;
	while (fgets(aline,1000,Infile)!=NULL){
		sscanf(aline,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",&dummyDouble,&tempArr[0],&tempArr[1],&tempArr[2],&tempArr[3],
			&tempArr[4],&tempArr[5],&tempArr[6],&tempArr[7],&tempArr[8],&tempArr[9],&tempArr[10],&tempArr[11],&tempArr[12]);
		rrd = tempArr[11]*px;
        SpotsMat[counter][13] = -1;
        found = 0;
		for (i=0;i<nRings;i++){
			if (fabs(rrd-RingRads[i]) < width){
				SpotsMat[counter][0] = counter+1;
				SpotsMat[counter][1] = tempArr[0];
				SpotsMat[counter][2] = tempArr[1];
				SpotsMat[counter][3] = tempArr[2];
				SpotsMat[counter][4] = tempArr[3];
				SpotsMat[counter][5] = tempArr[4];
				SpotsMat[counter][6] = tempArr[5];
				SpotsMat[counter][7] = tempArr[6];
				Sigmas[counter][0] = tempArr[7];
				Sigmas[counter][1] = tempArr[8];
				NrPx[counter][0] = tempArr[9];
				NrPx[counter][1] = tempArr[10];
				SpotsMat[counter][8] = tempArr[11];
				SpotsMat[counter][10] = tempArr[12];
				//~ if (SpotsMat[counter][2] < MinOme) MinOme = SpotsMat[counter][2];
				//~ if (SpotsMat[counter][2] > MaxOme) MaxOme = SpotsMat[counter][2];
				PowderInt[i] += SpotsMat[counter][1];
				SpotsMat[counter][9] = 0.5*(atand(SpotsMat[counter][8]*px/Lsd));
				SpotsMat[counter][11] = fabs(OmegaStep) + SpotsMat[counter][7] - SpotsMat[counter][6];
				SpotsMat[counter][12] = SpotsMat[counter][11]/fabs(OmegaStep);
				SpotsMat[counter][13] = RingNrs[i];
				if (TopLayer == 1 && fabs(SpotsMat[counter][10]) < 90)
				{}
				else {
                    found = 1;
					counter++;
				}
			}
		}
        if (SpotsMat[counter][13] == -1 && found==1) counter--;
	}
	for (i=0;i<nRings;i++){
		PowderInt[i] /= (EndNr-StartNr+1);
	}
	double Vgauge = Hbeam * M_PI * Rsample * Rsample;
	if (Vsample != 0){
		Vgauge = Vsample;
	}
	if (DiscModel == 1){
		Vgauge = DiscArea;
	}
	int j,ctr;
	double deltaTheta;
	for (i=0;i<counter;i++){
		RingNr = SpotsMat[i][13];
		for (j=0;j<nRings;j++){
			if (RingNrs[j] == RingNr){
				ctr = j;
			}
		}
		deltaTheta = deg2rad*(asind(((sind(SpotsMat[i][9]))*(cosd(SpotsMat[i][11])))+((cosd(SpotsMat[i][9]))
						   *(fabs(sind(SpotsMat[i][10])))*(sind(SpotsMat[i][11])))) - SpotsMat[i][9]);
		SpotsMat[i][14] = 0.5*((double)mhkl[ctr])*deltaTheta*cosd(SpotsMat[i][9])*Vgauge*SpotsMat[i][1]/(SpotsMat[i][12]*PowderInt[ctr]);
		//~ totVol += SpotsMat[i][14];
		SpotsMat[i][15] = cbrt(3*SpotsMat[i][14]/(4*M_PI));
		if (DiscModel == 1){
			SpotsMat[i][15] = sqrt(SpotsMat[i][14]/M_PI);
		}
		for (j=0;j<16;j++){
			fprintf(outfile,"%f ",SpotsMat[i][j]);
		}
		fprintf(outfile,"%f %f %f %f %f\n",PowderInt[ctr],Sigmas[i][0],Sigmas[i][1],NrPx[i][0],NrPx[i][1]);
	}
	FreeMemMatrix(SpotsMat,MAX_N_SPOTS);
	end = clock();
    diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Time elapsed: %f s.\n",diftotal);
    return 0;
}
