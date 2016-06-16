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
// Data Model: 
// 1.	Read SpotMatrixOld.csv -> Pick spots for a grain and find 
//		the positions (y,z,ome) from ExtraOld.bin.
// 2.	Read Data.bin (row number in Spots.bin), nData.bin (number of
//		spots in a certain bin in Data.bin) and Spots.bin.
// 3.	Calculate best match, based on Internal Angle.
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
#include <errno.h>
#include <stdarg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h> 

static void
check (int test, const char * message, ...)
{
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

int main(int argc, char *argv[])
{
	int i;
	double **AllSpotsYZO;
	double *AllSpots;
	int fd;
	struct stat s;
	int status;
	size_t size;
	const char *filename = "/dev/shm/ExtraInfo.bin";
	int rc;
	fd = open(filename,O_RDONLY);
	check(fd < 0, "open %s failed: %s", filename, strerror(errno));
	status = fstat (fd , &s);
	check (status < 0, "stat %s failed: %s", filename, strerror(errno));
	size = s.st_size;
	AllSpots = mmap(0,size,PROT_READ,MAP_SHARED,fd,0);
	check (AllSpots == MAP_FAILED,"mmap %s failed: %s", filename, strerror(errno));
	int nSpots =  (int) size/(14*sizeof(double));
	AllSpotsYZO = allocMatrix(nSpots,8);
	for (i=0;i<nSpots;i++){
		AllSpotsYZO[i][0] = AllSpots[i*14+0];
		AllSpotsYZO[i][1] = AllSpots[i*14+1];
		AllSpotsYZO[i][2] = AllSpots[i*14+2];
		AllSpotsYZO[i][3] = AllSpots[i*14+4];
		AllSpotsYZO[i][4] = AllSpots[i*14+8];
		AllSpotsYZO[i][5] = AllSpots[i*14+9];
		AllSpotsYZO[i][6] = AllSpots[i*14+10];
		AllSpotsYZO[i][7] = AllSpots[i*14+5];
	}
	int tc2 = munmap(AllSpots,size);
}
