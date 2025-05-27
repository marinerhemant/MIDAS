//
// We will provide the orientTol, 3 euler angle arrays, dimensions of the arrays and fillVal, will get back grain IDs
// TODO: Do things in 2D instead of 3D.

#include<stdio.h>
#include<stdlib.h>
#include "nf_headers.h"

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
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
	double *Eul1,*Eul2, miso, ang;
	Eul1 = calloc(3,sizeof(*Eul1));
	Eul2 = calloc(3,sizeof(*Eul2));
	Eul1[0] = Euler1[Pos1];
	Eul1[1] = Euler2[Pos1];
	Eul1[2] = Euler3[Pos1];
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
		Eul2[0] = Euler1[Pos2];
		Eul2[1] = Euler2[Pos2];
		Eul2[2] = Euler3[Pos2];
		miso = GetMisOrientationAngle(Eul1,Eul2,&ang,NSym,Sym);
		if (miso < oT){
			//~ printf("%d %d %d %d %d %d %d Found!\n",Pos[0],Pos[1],Pos[2],a2,b2,c2,grainNr);
			PosNext[0] = a2;
			PosNext[1] = b2;
			PosNext[2] = c2;
			DFS(PosNext,grainNr);
		}
	}
	free(Eul1);
	free(Eul2);
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
	oT = orientTol;
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
	double *Eul1,*Eul2, miso, ang;
	Eul1 = calloc(3,sizeof(*Eul1));
	Eul2 = calloc(3,sizeof(*Eul2));
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
					Eul1[0] = Euler1[Pos1];
					Eul1[1] = Euler2[Pos1];
					Eul1[2] = Euler3[Pos1];
					for (i=0;i<26;i++){
						a2 = layernr + diffArr[0][i];
						b2 = xpos + diffArr[1][i];
						c2 = ypos + diffArr[2][i];
						if (a2 < 0 || a2 == Dims[0]) continue;
						if (b2 < 0 || b2 == Dims[1]) continue;
						if (c2 < 0 || c2 == Dims[2]) continue;
						Pos2 = getIDX(a2,b2,c2,Dims[1],Dims[2]);
						if (GrainNrs[Pos2] !=fV){
							Eul2[0] = Euler1[Pos2];
							Eul2[1] = Euler2[Pos2];
							Eul2[2] = Euler3[Pos2];
							miso = GetMisOrientationAngle(Eul1,Eul2,&ang,NSym,Sym);
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
