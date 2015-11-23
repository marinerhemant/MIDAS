//
//  Peaks.cu
//
//
//  Created by Hemant Sharma on 2015/07/04.
//

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "nldrmd.cuh"

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define MAX_LINE_LENGTH 10240
#define MAX_N_RINGS 5000
#define CalcNorm3(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define CalcNorm2(x,y) sqrt((x)*(x) + (y)*(y))
typedef uint16_t pixelvalue;

static inline double sind(double x){return sin(deg2rad*x);}
static inline double cosd(double x){return cos(deg2rad*x);}
static inline double tand(double x){return tan(deg2rad*x);}
static inline double asind(double x){return rad2deg*(asin(x));}
static inline double acosd(double x){return rad2deg*(acos(x));}
static inline double atand(double x){return rad2deg*(atan(x));}

static inline pixelvalue** allocMatrixPX(int nrows, int ncols)
{
    pixelvalue** arr;
    int i;
    arr = (pixelvalue **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (pixelvalue *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}

static inline int** allocMatrixInt(int nrows, int ncols)
{
    int** arr;
    int i;
    arr = (int **) malloc(nrows * sizeof(*arr));
    if (arr == NULL ) {
        return NULL;
    }
    for ( i = 0 ; i < nrows ; i++) {
        arr[i] = (int *) malloc(ncols * sizeof(*arr[i]));
        if (arr[i] == NULL ) {
            return NULL;
        }
    }
    return arr;
}


static inline void FreeMemMatrixPx(pixelvalue **mat,int nrows)
{
    int r;
    for ( r = 0 ; r < nrows ; r++) {
        free(mat[r]);
    }
    free(mat);
}

static inline void DoImageTransformations (int NrTransOpt, int TransOpt[10], pixelvalue *Image, int NrPixels)
{
	int i,j,k,l,m;
    pixelvalue **ImageTemp1, **ImageTemp2;
    ImageTemp1 = allocMatrixPX(NrPixels,NrPixels);
    ImageTemp2 = allocMatrixPX(NrPixels,NrPixels);
	for (k=0;k<NrPixels;k++) {
		for (l=0;l<NrPixels;l++) {
			ImageTemp1[k][l] = Image[(NrPixels*k)+l];
		}
	}
	for (k=0;k<NrTransOpt;k++) {
		if (TransOpt[k] == 1){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][NrPixels-m-1]; //Inverting Y.
		} else if (TransOpt[k] == 2){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[NrPixels-l-1][m]; //Inverting Z.
		} else if (TransOpt[k] == 3){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[m][l];
		} else if (TransOpt[k] == 0){
			for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp2[l][m] = ImageTemp1[l][m];
		}
		for (l=0;l<NrPixels;l++) for (m=0;m<NrPixels;m++) ImageTemp1[l][m] = ImageTemp2[l][m];
	}
	for (k=0;k<NrPixels;k++) for (l=0;l<NrPixels;l++) Image[(NrPixels*k)+l] = ImageTemp2[k][l];
	FreeMemMatrixPx(ImageTemp1,NrPixels);
	FreeMemMatrixPx(ImageTemp2,NrPixels);
}

static inline void Transposer (double *x, int n, double *y)
{
	int i,j;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			y[(i*n)+j] = x[(j*n)+i];
		}
	}
}

const int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
const int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

static inline void DepthFirstSearch(int x, int y, int current_label, int NrPixels, int **BoolImage, int **ConnectedComponents,int **Positions, int *PositionTrackers)
{
	if (x < 0 || x == NrPixels) return;
	if (y < 0 || y == NrPixels) return;
	if ((ConnectedComponents[x][y]!=0)||(BoolImage[x][y]==0)) return;
	
	ConnectedComponents[x][y] = current_label;
	Positions[current_label][PositionTrackers[current_label]] = (x*NrPixels) + y;
	PositionTrackers[current_label] += 1;
	int direction;
	for (direction=0;direction<8;++direction){
		DepthFirstSearch(x + dx[direction], y + dy[direction], current_label, NrPixels, BoolImage, ConnectedComponents,Positions,PositionTrackers);
		
	}
}

static inline int FindConnectedComponents(int **BoolImage, int NrPixels, int **ConnectedComponents, int **Positions, int *PositionTrackers){
	int i,j;
	for (i=0;i<NrPixels;i++){
		for (j=0;j<NrPixels;j++){
			ConnectedComponents[i][j] = 0;
		}
	}
	int component = 0;
	for (i=0;i<NrPixels;++i) {
		for (j=0;j<NrPixels;++j) {
			if ((ConnectedComponents[i][j]==0) && (BoolImage[i][j] == 1)){
				DepthFirstSearch(i,j,++component,NrPixels,BoolImage,ConnectedComponents,Positions,PositionTrackers);
			}
		}
	}
	return component;
}

int main(int argc, char *argv[]){ // Arguments: parameter file name
	if (argc != 2){
		printf("Not enough arguments, exiting. Use as:\n\t\t%s %s\n",argv[0],argv[1]);
		return 1;
	}
	//Read params file
    char *ParamFN;
    FILE *fileParam;
    ParamFN = argv[1];
    char line[MAX_LINE_LENGTH];
    fileParam = fopen(ParamFN,"r");
    if (fileParam == NULL){
		printf("Parameter file: %s could not be read. Exiting\n",argv[1]);
		return 1;
	}
	char *str;
	int cmpres, StartFileNr, NrFilesPerSweep, NumDarkBegin=0, NumDarkEnd=0,
		ColBeamCurrent, NrOfRings=0, RingNumbers[MAX_N_RINGS], TransOpt[10], 
		NrTransOpt=0, DoFullImage=0, Padding, NrPixels, LayerNr, FrameNumberToDo=-1;
	double OmegaOffset = 0, bc=0, RingSizeThreshold[MAX_N_RINGS][4], px, 
		Width, IntSat, Ycen, Zcen;
	char dummy[MAX_LINE_LENGTH], ParFilePath[MAX_LINE_LENGTH], 
		FileStem[MAX_LINE_LENGTH], RawFolder[MAX_LINE_LENGTH], 
		OutputFolder[MAX_LINE_LENGTH], darkcurrentfilename[MAX_LINE_LENGTH], 
		floodfilename[MAX_LINE_LENGTH], Ext[MAX_LINE_LENGTH];
	while (fgets(line, MAX_LINE_LENGTH, fileParam) != NULL) {
		str = "ParFilePath ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, ParFilePath);
			continue;
		}
		str = "RingThresh ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d %lf", dummy, RingNumbers[NrOfRings], 
				RingSizeThreshold[NrOfRings][1]);
			NrOfRings++;
			continue;
		}
		str = "FileStem ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %s", dummy, FileStem);
			continue;
		}
		str = "ParFileColBeamCurrent ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &ColBeamCurrent);
			continue;
		}
		str = "StartFileNr ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &StartFileNr);
			continue;
		}
		str = "NrFilesPerSweep ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NrFilesPerSweep);
			continue;
		}
		str = "NumDarkBegin ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkBegin);
			continue;
		}
		str = "NumDarkEnd ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %d", dummy, &NumDarkEnd);
			continue;
		}
		str = "OmegaOffset ";
		cmpres = strncmp(line, str, strlen(str));
		if (cmpres == 0) {
			sscanf(line, "%s %lf", dummy, &OmegaOffset);
			continue;
		}
		str = "BeamCurrent ";
		cmpres = strncmp(line,str,strlen(str));
		if (cmpres==0){
			sscanf(line,"%s %lf", dummy, &bc);
			continue;
		}
        str = "Width ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &Width);
            continue;
        }
        str = "px ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &px);
            continue;
        }
        str = "ImTransOpt ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &TransOpt[NrTransOpt]);
            NrTransOpt++;
            continue;
        }
        str = "DoFullImage ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &DoFullImage);
            continue;
        }
        str = "RawFolder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, RawFolder);
            continue;
        }
        str = "Folder ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, OutputFolder);
            continue;
        }
        str = "Dark ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, darkcurrentfilename);
            continue;
        }
        str = "Flood ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, floodfilename);
            continue;
        }
        str = "BC ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf %lf", dummy, &Ycen, &Zcen);
            continue;
        }
        str = "UpperBoundThreshold ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %lf", dummy, &IntSat);
            continue;
        }
        str = "LayerNr ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &LayerNr);
            continue;
        }
        str = "NrPixels ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &NrPixels);
            continue;
        }
        str = "Padding ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &Padding);
            continue;
        }
        str = "SingleFrameNumber ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %d", dummy, &FrameNumberToDo);
            continue;
        }
        str = "Ext ";
        cmpres = strncmp(line,str,strlen(str));
        if (cmpres==0){
            sscanf(line,"%s %s", dummy, Ext);
            continue;
        }
	}
	if (DoFullImage == 1 && FrameNumberToDo == -1){
		printf("For processing the full image you need to provide a single"
			" Frame Number using the FrameNumberToDo parameter in the"
			" param file.\n Exiting\n");
		return (1);
	}
	Width = Width/px;
	FILE *ParFile;
	ParFile = fopen(ParFilePath,"r");
	if (ParFile == NULL){
		printf("ParFile could not be read");
		return 1;
	}
	int i, j, k;
	int NrFramesPerFile[NrFilesPerSweep],CurrFileNrOffset;
	for (i=0;i<NrFilesPerSweep;i++){
		NrFramesPerFile[i] = -(NumDarkBegin+NumDarkEnd);
	}
	char *token, *saveptr;
	int OmegaSign=1, goodLine, omegafound;
	double Omegas[NrFilesPerSweep][300],BeamCurrents[NrFilesPerSweep][300],
			maxBC=0;
	while (fgets(line, MAX_LINE_LENGTH, ParFile) != NULL) {
		strncpy(line,line,strlen(line));
		goodLine = 0;
		for (str = line; ; str=NULL){
			token = strtok_r(str, " ", &saveptr);
			if (token == NULL) break;
			if (!strncmp(token,FileStem,strlen(FileStem))){
				token = strtok_r(str, " ", &saveptr);
				token = strtok_r(str, " ", &saveptr);
				CurrFileNrOffset = atoi(token)-StartFileNr;
				if (CurrFileNrOffset >=0 && CurrFileNrOffset < NrFilesPerSweep){
					NrFramesPerFile[CurrFileNrOffset]++;
					goodLine = 1;
				}
			}
		}
		if (NrFramesPerFile[CurrFileNrOffset] < -NumDarkBegin + 1) continue;
		if (goodLine){
			strncpy(line,line,strlen(line));
			omegafound = 0;
			for (i=1, str = line; ; i++, str = NULL){
				token = strtok_r(str, " ", &saveptr);
				if (token == NULL) break;
				if (!strncmp(token,"ramsrot",strlen("ramsrot"))){
					omegafound = 1;
					OmegaSign = 1;
				} else if (!strncmp(token,"aero",strlen("aero"))){
					omegafound = 1;
					OmegaSign = -1;
				} else if (!strncmp(token,"preci",strlen("preci"))){
					omegafound = 1;
					OmegaSign = 1;
				}
				if (omegafound){
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					token  = strtok_r(str," ", &saveptr);
					i+=3;
					Omegas[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] 
								= atof(token) * OmegaSign + OmegaOffset;
					omegafound = 0;
				}
				if (i == ColBeamCurrent){
					BeamCurrents[CurrFileNrOffset][NrFramesPerFile
							[CurrFileNrOffset]+NumDarkBegin-1] = atof(token);
					maxBC = (maxBC > atof(token)) ? maxBC : atof(token);
				}
			}
		}
	}
	int TotalNrFrames = 0;
	for (i=0;i<NrFilesPerSweep;i++){
		TotalNrFrames += NrFramesPerFile[i];
	}
	bc = (bc > maxBC) ? bc : maxBC;
	// Read hkls.csv
   	char *hklfn = "hkls.csv";
	FILE *hklf = fopen(hklfn,"r");
	fgets(line,1000,hklf);
	int Rnr;
	double RRd;
	while (fgets(line,1000,hklf)!=NULL){
		sscanf(line, "%s %s %s %s %d %s %s %s %s %s %lf", dummy, dummy, 
			dummy, dummy, &Rnr, dummy, dummy, dummy, dummy ,dummy, &RRd);
		for (i=0;i<NrOfRings;i++){
			if (Rnr == RingNumbers[i]){
				RingSizeThreshold[i][0] = RRd/px;
				RingSizeThreshold[i][2] = RRd/px - Width;
				RingSizeThreshold[i][3] = RRd/px + Width;
			}
		}
	}
    for (i=0;i<NrTransOpt;i++){
        if (TransOpt[i] < 0 || TransOpt[i] > 3){
			printf("TransformationOptions can only be 0, 1, 2 or 3.\nExiting.\n");
			return 1;
		}
        printf("TransformationOptions: %d ",TransOpt[i]);
        if (TransOpt[i] == 0) printf("No change.\n");
        else if (TransOpt[i] == 1) printf("Flip Left Right.\n");
        else if (TransOpt[i] == 2) printf("Flip Top Bottom.\n");
        else printf("Transpose.\n");
    }
    int *GoodCoords, *RingInfoImage, TotalGoodPixels=0, ythis, zthis;
    double Rmin, Rmax, Rt;
    GoodCoords = (int*) malloc(NrPixels*NrPixels*sizeof(*GoodCoords));
    RingInfoImage = (int*) malloc(NrPixels*NrPixels*sizeof(*RingInfoImage));
	for (i=1;i<NrPixels;i++){
		for (j=1;j<NrPixels;j++){
			Rt = sqrt((i-Ycen)*(i-Ycen)+(j-Zcen)*(j-Zcen));
			for (k=0;k<NrOfRings;k++){
				Rmin = RingSizeThreshold[k][2];
				Rmax = RingSizeThreshold[k][3];
				if (Rt > Rmin && Rt < Rmax){
					GoodCoords[((i-1)*NrPixels)+(j-1)] = 1;
					RingInfoImage[((i-1)*NrPixels)+(j-1)] = RingNumbers[k];
					TotalGoodPixels++;
				}else {
					GoodCoords[((i-1)*NrPixels)+(j-1)] = 0;
					RingInfoImage[((i-1)*NrPixels)+(j-1)] = 0;
				}
			}
		}
	}
	if (DoFullImage == 1){
		TotalNrFrames = 1;
		for (i=0;i<NrPixels*NrPixels;i++) {
			GoodCoords[i] = 1;
		}
		TotalGoodPixels = NrPixels*NrPixels;
	}
	double *dark,*flood, *darkTemp;;
	dark = (double *) malloc(NrPixels*NrPixels*sizeof(*dark));
	darkTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*darkTemp));
	flood = (double *) malloc(NrPixels*NrPixels*sizeof(*flood));
	
	// If a darkfile is specified.
	FILE *darkfile=fopen(darkcurrentfilename,"rb");
	int sz, nFrames;
	int SizeFile = sizeof(pixelvalue) * NrPixels * NrPixels;
	long int Skip;
	for (i=0;i<(NrPixels*NrPixels);i++){
		dark[i]=0;
		darkTemp[i]=0;
	}
	pixelvalue *darkcontents;
	darkcontents = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*darkcontents));
	if (darkfile==NULL){
		printf("No dark file was specified, will use %d frames at the beginning of each file for dark calculation.", NumDarkBegin);
	}else{
		fseek(darkfile,0L,SEEK_END);
		sz = ftell(darkfile);
		rewind(darkfile);
		nFrames = sz/(8*1024*1024);
		Skip = sz - (nFrames*8*1024*1024);
		fseek(darkfile,Skip,SEEK_SET);
		printf("Reading dark file: %s, nFrames: %d, skipping first %ld bytes.\n",darkcurrentfilename,nFrames,Skip);
		for (i=0;i<nFrames;i++){
			fread(darkcontents,SizeFile,1,darkfile);
			DoImageTransformations(NrTransOpt,TransOpt,darkcontents,NrPixels);
			for (j=0;j<(NrPixels*NrPixels);j++){
				darkTemp[j] += darkcontents[j];
			}
		}
		fclose(darkfile);
		for (i=0;i<(NrPixels*NrPixels);i++){
			darkTemp[i] /= nFrames;
		}
	}
	Transposer(darkTemp,NrPixels,dark);
	free(darkcontents);
	FILE *floodfile=fopen(floodfilename,"rb");
	if (floodfile==NULL){
		printf("Could not read the flood file. Using no flood correction.\n");
		for(i=0;i<(NrPixels*NrPixels);i++){
			flood[i]=1;
		}
	}
	else{
		fread(flood,sizeof(double)*NrPixels*NrPixels, 1, floodfile);
		fclose(floodfile);
	}
	int FrameNr = 0, FramesToSkip, CurrentFileNr, CurrentRingNr;
	double beamcurr, Thresh;
	pixelvalue *Image;
	Image = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(*Image));
	char FN[MAX_LINE_LENGTH];
	double *ImgCorrBCTemp, *ImgCorrBC;
	ImgCorrBC = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBC));
	ImgCorrBCTemp = (double *) malloc(NrPixels*NrPixels*sizeof(*ImgCorrBCTemp));
	char outfoldername[MAX_LINE_LENGTH];
	sprintf(outfoldername,"%s/Temp",OutputFolder);
	char extcmd[MAX_LINE_LENGTH];
	sprintf(extcmd,"mkdir -p %s",outfoldername);
	system(extcmd);
	int nOverlapsMaxPerImage = 10000;
	int **BoolImage, **ConnectedComponents, **Positions, *PositionTrackers, NrOfReg;
	BoolImage = allocMatrixInt(NrPixels,NrPixels);
	ConnectedComponents = allocMatrixInt(NrPixels,NrPixels);
	Positions = allocMatrixInt(nOverlapsMaxPerImage,NrPixels*4);
	PositionTrackers = (int *) malloc(nOverlapsMaxPerImage*sizeof(*PositionTrackers));
	int RegNr, NrPixelsThisRegion, **MaximaPositions, **UsefulPixels, IsSaturated,
		SpotIDStart;
	double *MaximaValues, *z;
	MaximaPositions = allocMatrixInt(2, NrPixels*NrPixels);
	MaximaValues = (double *) malloc(NrPixels*NrPixels*sizeof(*MaximaValues));
	UsefulPixels = allocMatrixInt(2, NrPixels*NrPixels);
	z = (double *) malloc(NrPixels*NrPixels*sizeof(*z));
	char OutFile[MAX_LINE_LENGTH];
	int TotNrRegions;
	while (FrameNr < TotalNrFrames){
		if (TotalNrFrames == 1){
			FrameNr = FrameNumberToDo;
			for (i=0;i<NrFilesPerSweep;i++){
				if (NrFramesPerFile[i]/FrameNumberToDo > 0){
					FrameNumberToDo -= NrFramesPerFile[i];
				}else{
					CurrentFileNr = StartFileNr + i;
					FramesToSkip = FrameNumberToDo;
					break;
				}
			}
		}else{
			FramesToSkip = FrameNr;
			for (i=0;i<NrFilesPerSweep;i++){
				if (NrFramesPerFile[i]/FramesToSkip > 0){
					FramesToSkip -= NrFramesPerFile[i];
				}else{
					CurrentFileNr = StartFileNr + i;
					break;
				}
			}
		}
		if (Padding == 2){sprintf(FN,"%s/%s_%02d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 3){sprintf(FN,"%s/%s_%03d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 4){sprintf(FN,"%s/%s_%04d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 5){sprintf(FN,"%s/%s_%05d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 6){sprintf(FN,"%s/%s_%06d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 7){sprintf(FN,"%s/%s_%07d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 8){sprintf(FN,"%s/%s_%08d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		else if (Padding == 9){sprintf(FN,"%s/%s_%09d%s",RawFolder,FileStem,CurrentFileNr,Ext);}
		FILE *ImageFile = fopen(FN,"rb");
		if (ImageFile == NULL){
			printf("Could not read the input file. Exiting.\n");
			return 1;
		}
		fseek(ImageFile,0L,SEEK_END);
		sz = ftell(ImageFile);
		rewind(ImageFile);
		Skip = sz - ((NrFramesPerFile[StartFileNr-CurrentFileNr] + NumDarkEnd - FramesToSkip) * 8*1024*1024);
		printf("Now processing file: %s, Frame: %d\n",FN, FramesToSkip);
		fseek(ImageFile,Skip,SEEK_SET);
		fread(Image,SizeFile,1,ImageFile);
		fclose(ImageFile);
		DoImageTransformations(NrTransOpt,TransOpt,Image,NrPixels);
		beamcurr = BeamCurrents[StartFileNr - CurrentFileNr][FramesToSkip];
		printf("Beam current this file: %f, Beam current scaling value: %f\n",beamcurr,bc);
		for (i=0;i<NrPixels*NrPixels;i++)
			ImgCorrBCTemp[i]=Image[i];
		Transposer(ImgCorrBCTemp,NrPixels,ImgCorrBC);
		for (i=0;i<NrPixels*NrPixels;i++){
			ImgCorrBC[i] = (ImgCorrBC[i] - dark[i])/flood[i];
			ImgCorrBC[i] = ImgCorrBC[i]*bc/beamcurr;
			CurrentRingNr = RingInfoImage[i];
			Thresh = RingSizeThreshold[CurrentRingNr][1];
			if (ImgCorrBC[i] < Thresh){
				ImgCorrBC[i] = 0;
			}
			if (GoodCoords[i] == 0){
				ImgCorrBC[i] = 0;
			}
		}
		for (i=0;i<nOverlapsMaxPerImage;i++)
			PositionTrackers[i] = 0;
		for (i=0;i<NrPixels;i++){
			for (j=0;j<NrPixels;j++){
				if (ImgCorrBC[(i*NrPixels)+j] != 0){
					BoolImage[i][j] = 1;
				}else{
					BoolImage[i][j] = 0;
				}
			}
		}
		NrOfReg = FindConnectedComponents(BoolImage,NrPixels,ConnectedComponents,Positions,PositionTrackers);
		SpotIDStart = 1;
		if (Padding == 2) {sprintf(OutFile,"%s/%s_%d_%02ds_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 3) {sprintf(OutFile,"%s/%s_%03d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 4) {sprintf(OutFile,"%s/%s_%04d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 5) {sprintf(OutFile,"%s/%s_%05d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 6) {sprintf(OutFile,"%s/%s_%06d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 7) {sprintf(OutFile,"%s/%s_%07d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 8) {sprintf(OutFile,"%s/%s_%08d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		else if (Padding == 9) {sprintf(OutFile,"%s/%s_%09d_%d_PS.csv",outfoldername,FileStem,LayerNr,FrameNr);}
		FILE *outfilewrite;
		outfilewrite = fopen(OutFile,"w");
		fprintf(outfilewrite,"SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta\n");
		TotNrRegions = NrOfReg;
		for (RegNr=1;RegNr<=NrOfReg;RegNr++){
			NrPixelsThisRegion = PositionTrackers[RegNr];
			if (NrPixelsThisRegion == 1){
				TotNrRegions--;
				continue;
			}

		
		fclose(outfilewrite);
		FrameNr++;
	}
}
