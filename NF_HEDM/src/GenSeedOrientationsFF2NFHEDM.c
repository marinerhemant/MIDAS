//
//
// GenSeedOrientationsFF2NFHEDM.c
//
// Generate seed orientations for NF_HEDM using orientations from FF_HEDM
//
// Created by Hemant Sharma on 2014/07/21.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

static inline void OrientMat2Quat(double OrientMat[9], double Quat[4]){
	double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
	if(trace > 0){
		double s = 0.5/sqrt(trace+1.0);
		Quat[0] = 0.25/s;
		Quat[1] = (OrientMat[7]-OrientMat[5])*s;
		Quat[2] = (OrientMat[2]-OrientMat[6])*s;
		Quat[3] = (OrientMat[3]-OrientMat[1])*s;
	}else{
		if (OrientMat[0]>OrientMat[4] && OrientMat[0]>OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8]);
			Quat[0] = (OrientMat[7]-OrientMat[5])/s;
			Quat[1] = 0.25*s;
			Quat[2] = (OrientMat[1]+OrientMat[3])/s;
			Quat[3] = (OrientMat[2]+OrientMat[6])/s;
		} else if (OrientMat[4] > OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8]);
			Quat[0] = (OrientMat[2]-OrientMat[6])/s;
			Quat[1] = (OrientMat[1]+OrientMat[3])/s;
			Quat[2] = 0.25*s;
			Quat[3] = (OrientMat[5]+OrientMat[7])/s;
		} else {
			double s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4]);
			Quat[0] = (OrientMat[3]-OrientMat[1])/s;
			Quat[1] = (OrientMat[2]+OrientMat[6])/s;
			Quat[2] = (OrientMat[5]+OrientMat[7])/s;
			Quat[3] = 0.25*s;
		}
	}
	if (Quat[0] < 0){
		Quat[0] = -Quat[0];
		Quat[1] = -Quat[1];
		Quat[2] = -Quat[2];
		Quat[3] = -Quat[3];
	}
	double QNorm = sqrt(Quat[0]*Quat[0] + Quat[1]*Quat[1] + Quat[2]*Quat[2] + Quat[3]*Quat[3]);
	Quat[0] /= QNorm;
	Quat[1] /= QNorm;
	Quat[2] /= QNorm;
	Quat[3] /= QNorm;
}

static inline void
usage(void)
{
    printf("GenSeedOrientations usage: ./GenSeedOrientations <OringinalFile>"
    " <OutputFile>. Please provide full path.\n");
}

int main(int argc, char *argv[])
{
	if (argc != 3)
    {
        usage();
        return 1;
    }
    clock_t start0, end;
    double diftotal;
    start0 = clock();
	char *outfilename;
    outfilename = argv[2];
    char *GrainFN;
    GrainFN = argv[1];
    char aline[1000];
	double OrientMatrix[9],Quaternion[4],LatC[6];
	char dummy[1024];
	FILE *GrainFile = fopen(GrainFN,"r");
	FILE *OutFile = fopen(outfilename,"w");
	int nOrientations = 0;
	int GrainID;
    while(fgets(aline,1000,GrainFile)!=NULL){
		if (aline[0] == '%') continue;
        sscanf(aline,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %s %s %s %lf %lf %lf %lf %lf %lf %s %s %s %s",&GrainID,&OrientMatrix[0],
				&OrientMatrix[1],&OrientMatrix[2],&OrientMatrix[3],&OrientMatrix[4],&OrientMatrix[5],&OrientMatrix[6],
				&OrientMatrix[7],&OrientMatrix[8],dummy,dummy,dummy,&LatC[0],&LatC[1],&LatC[2],&LatC[3],&LatC[4],&LatC[5],dummy,dummy,dummy,dummy);
		OrientMat2Quat(OrientMatrix,Quaternion);
		fprintf(OutFile,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n",Quaternion[0],Quaternion[1],Quaternion[2],Quaternion[3],LatC[0],LatC[1],LatC[2],LatC[3],LatC[4],LatC[5],GrainID);
        nOrientations++;
	}
	printf("Number of seed orientations: %d\n",nOrientations);
	fclose(GrainFile);
    end = clock();
    diftotal = ((double)(end-start0))/CLOCKS_PER_SEC;
    printf("Time elapsed in making diffraction spots: %f [s]\n",diftotal);
    return 0;
}
