//
// We will provide the orientTol, 3 euler angle arrays, dimensions of the arrays and fillVal, will get back grain IDs
// TODO: Do things in 2D instead of 3D.

#include<stdio.h>
#include<stdlib.h>
#include "nf_headers.h"

#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)
#define maxNGrains 1000000

int Dims[3];
double fV, oT;
int NSym;
int grainSize;
double *Euler1, *Euler2, *Euler3;
int *GrainNrs;
double Sym[24][4];


int diffArr[3][26] = {{-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
					  {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1},
					  {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1}};

inline long long int getIDX (int layerNr, int xpos, int ypos, int xMax, int yMax){
	long long int retval = layerNr;
	retval *= xMax;
	retval *= yMax;
	retval += xpos * yMax;
	retval += ypos;
	return retval;
}

inline void DFS (int *Pos, int grainNr){
	long long int Pos1 = getIDX(Pos[0],Pos[1],Pos[2],Dims[1],Dims[2]);
	if (GrainNrs[Pos1] != 0) return;
	GrainNrs[Pos1] = grainNr;
	grainSize++;
	int i;
	double eul1[3], eul2[3], miso, ang;
	double mat1[3][3], mat2[3][3], quat1[4], quat2[4];
	eul1[0] = Euler1[Pos1];
	eul1[1] = Euler2[Pos1];
	eul1[2] = Euler3[Pos1];
	Euler2OrientMat(eul1, mat1);
	OrientMat2Quat(&mat1[0][0], quat1);
	int a2,b2,c2;
	long long int Pos2;
	int *PosNext;
	PosNext = calloc(3,sizeof(int));
	for (i=0;i<26;i++){
		a2 = Pos[0] + diffArr[0][i];
		b2 = Pos[1] + diffArr[1][i];
		c2 = Pos[2] + diffArr[2][i];
		if (a2 < 0 || a2 == Dims[0]) continue;
		if (b2 < 0 || b2 == Dims[1]) continue;
		if (c2 < 0 || c2 == Dims[2]) continue;
		Pos2 = getIDX(a2,b2,c2,Dims[1],Dims[2]);
		if (Euler1[Pos2] == fV) continue;
		eul2[0] = Euler1[Pos2];
		eul2[1] = Euler2[Pos2];
		eul2[2] = Euler3[Pos2];
		Euler2OrientMat(eul2, mat2);
		OrientMat2Quat(&mat2[0][0], quat2);
		miso = GetMisOrientationAngle(quat1,quat2,&ang,NSym,Sym);
		if (miso < oT){
			//~ printf("%d %d %d %d %d %d %d Found!\n",Pos[0],Pos[1],Pos[2],a2,b2,c2,grainNr);
			PosNext[0] = a2;
			PosNext[1] = b2;
			PosNext[2] = c2;
			DFS(PosNext,grainNr);
		}
	}
	free(PosNext);
}

void calcGrainNrs (double orientTol, int nrLayers, int xMax, int yMax, double fillVal, int SGNum){
	int NrSymmetries;
	NrSymmetries = MakeSymmetries(SGNum,Sym);
	NSym = NrSymmetries;
	Dims[0] = nrLayers;
	Dims[1] = xMax;
	Dims[2] = yMax;
	fV = fillVal;
	oT = orientTol * deg2rad;  // orientTol is in degrees, convert for radians-native API
	GrainNrs = calloc(nrLayers*xMax*yMax,sizeof(*GrainNrs));
	FILE *f1 = fopen("EulerAngles1.bin","rb");
	FILE *f2 = fopen("EulerAngles2.bin","rb");
	FILE *f3 = fopen("EulerAngles3.bin","rb");
	Euler1 = calloc(nrLayers*xMax*yMax,sizeof(*Euler1));
	Euler2 = calloc(nrLayers*xMax*yMax,sizeof(*Euler2));
	Euler3 = calloc(nrLayers*xMax*yMax,sizeof(*Euler3));
	fread(Euler1,nrLayers*xMax*yMax*sizeof(double),1,f1);
	fread(Euler2,nrLayers*xMax*yMax*sizeof(double),1,f2);
	fread(Euler3,nrLayers*xMax*yMax*sizeof(double),1,f3);
	int layernr,xpos,ypos,a2,b2,c2;
	int grainNr = 0;
	int i,j, *Pos;
	Pos = calloc(3,sizeof(int));
	long long int Pos1, Pos2;
	double miso, ang;
	int *grainSizes;
	grainSizes = calloc(maxNGrains,sizeof(*grainSizes));
	for (layernr = 0; layernr < nrLayers; layernr++){
		for (xpos = 0; xpos < xMax; xpos++){
			for (ypos = 0; ypos < yMax; ypos++){
				Pos[0] = layernr;
				Pos[1] = xpos;
				Pos[2] = ypos;
				Pos1 = getIDX(Pos[0],Pos[1],Pos[2],Dims[1],Dims[2]);
				if (Euler1[Pos1] == fillVal){
					GrainNrs[Pos1] = (int)fillVal;
				} else if (GrainNrs[Pos1] == 0){
					grainNr++;
					grainSize = 0;
					DFS(Pos,grainNr);
					grainSizes[grainNr] = grainSize;
				}
			}
		}
	}
	int thisGrainNr;
	int *GSArr;
	GSArr = calloc(nrLayers*xMax*yMax,sizeof(*GSArr));
	double *kamArr;
	kamArr = calloc(nrLayers*xMax*yMax,sizeof(*kamArr));
	int nrKAM;
	for (layernr = 0; layernr < nrLayers; layernr++){
		for (xpos = 0; xpos < xMax; xpos++){
			for (ypos = 0; ypos < yMax; ypos++){
				Pos1 = getIDX(layernr,xpos,ypos,Dims[1],Dims[2]);
				thisGrainNr = GrainNrs[Pos1];
				if (thisGrainNr == fV){
					GSArr[Pos1] = (int)fV;
				} else {
					// put grain sizes
					GSArr[Pos1] = grainSizes[thisGrainNr];
					// Calculate kam
					nrKAM = 0;
					double kamEul1[3] = {Euler1[Pos1], Euler2[Pos1], Euler3[Pos1]};
					double kamMat1[3][3], kamQ1[4];
					Euler2OrientMat(kamEul1, kamMat1);
					OrientMat2Quat(&kamMat1[0][0], kamQ1);
					for (i=0;i<26;i++){
						a2 = layernr + diffArr[0][i];
						b2 = xpos + diffArr[1][i];
						c2 = ypos + diffArr[2][i];
						if (a2 < 0 || a2 == Dims[0]) continue;
						if (b2 < 0 || b2 == Dims[1]) continue;
						if (c2 < 0 || c2 == Dims[2]) continue;
						Pos2 = getIDX(a2,b2,c2,Dims[1],Dims[2]);
						if (GrainNrs[Pos2] !=fV){
							double kamEul2[3] = {Euler1[Pos2], Euler2[Pos2], Euler3[Pos2]};
							double kamMat2[3][3], kamQ2[4];
							Euler2OrientMat(kamEul2, kamMat2);
							OrientMat2Quat(&kamMat2[0][0], kamQ2);
							miso = GetMisOrientationAngle(kamQ1,kamQ2,&ang,NSym,Sym);
							nrKAM ++;
							kamArr[Pos1] += miso;
						}
					}
					if (nrKAM > 0) kamArr[Pos1] /= nrKAM;
					else kamArr[Pos1] = fV;
				}
			}
		}
	}
	printf("Total number of grains: %d\n",grainNr);
	FILE *f4 = fopen("GrainNrs.bin","wb");
	FILE *f5 = fopen("GrainSizes.bin","wb");
	FILE *f6 = fopen("KAMArr.bin","wb");
	fwrite(GrainNrs,nrLayers*xMax*yMax*sizeof(int),1,f4);
	fwrite(GSArr,nrLayers*xMax*yMax*sizeof(int),1,f5);
	fwrite(kamArr,nrLayers*xMax*yMax*sizeof(double),1,f6);
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
	fclose(f5);
	fclose(f6);
	free(Euler1);
	free(Euler2);
	free(Euler3);
	free(GrainNrs);
	free(GSArr);
	free(kamArr);
}
