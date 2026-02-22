//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// HDF and OpenMP implementation. Will do all distances in a single process using OpenMP.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <hdf5.h>
#include <omp.h>

typedef uint16_t pixelvalue;
pixelvalue quick_select(pixelvalue a[], int n) ;
omp_lock_t lock;

#define PIX_SWAP(a,b) { pixelvalue temp=(a);(a)=(b);(b)=temp; }
pixelvalue quick_select(pixelvalue a[], int n)
{
    int low, high ;
    int median;
    int middle, ll, hh;
    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low)
            return a[median] ;
        if (high == low + 1) {
            if (a[low] > a[high])
                PIX_SWAP(a[low], a[high]) ;
            return a[median] ;
        }
        middle = (low + high) / 2;
        if (a[middle] > a[high])    PIX_SWAP(a[middle], a[high]) ;
        if (a[low] > a[high])       PIX_SWAP(a[low], a[high]) ;
        if (a[middle] > a[low])     PIX_SWAP(a[middle], a[low]) ;
        PIX_SWAP(a[middle], a[low+1]) ;
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (a[low] > a[ll]) ;
            do hh--; while (a[hh]  > a[low]) ;
            if (hh < ll)
            break;
            PIX_SWAP(a[ll], a[hh]) ;
        }
        PIX_SWAP(a[low], a[hh]) ;
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}
#undef PIX_SWAP

static inline void CalcMedian(hid_t file, int NrFilesPerDistance, int NrDistances, int NrPixels, int DistanceNr)
{
	char buffer[4096];
	int i, j, k, frameNr;
	hid_t dataset, dcpl;
	size_t nelmts;
	unsigned int flags, filter_info;
	H5Z_filter_t filter_type;
	herr_t status;
	pixelvalue **filedata;
	filedata = (pixelvalue **) malloc(NrPixels*sizeof(pixelvalue*));
	filedata[0] = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(pixelvalue));
	for (i=0;i<NrPixels;i++) filedata[i] = filedata[0] + i*NrPixels;
	pixelvalue **AllIntensities, *MedianArray;
	MedianArray = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(pixelvalue));
	AllIntensities = (pixelvalue **) malloc(NrFilesPerDistance*sizeof(pixelvalue*));
	AllIntensities[0] = (pixelvalue *) malloc(NrPixels*NrPixels*NrFilesPerDistance*sizeof(pixelvalue));
	for (i=0;i<NrFilesPerDistance;i++) AllIntensities[i] = AllIntensities[0] + i*NrPixels*NrPixels;
    for (i=0;i<NrFilesPerDistance;i++){
		frameNr = i + NrFilesPerDistance*DistanceNr;
		sprintf(buffer,"/exchange/data/%d",frameNr);
		nelmts = 0;
		#pragma omp critical
		{
			dataset = H5Dopen(file,buffer,H5P_DEFAULT);
			dcpl = H5Dget_create_plist (dataset);
			filter_type = H5Pget_filter(dcpl,0,&flags,&nelmts,NULL,0,NULL,&filter_info);
			status = H5Dread(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		}
		printf("Frame Nr %d of %d DistanceNr %d\n",i,NrFilesPerDistance,DistanceNr);
		for (j=0;j<NrPixels;j++)
			for (k=0;k<NrPixels;k++)
				AllIntensities[i][j*NrPixels+k] = filedata[j][k];
	}
	pixelvalue *MaxIntArr, *MaxIntMedianArr;
	MaxIntArr = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(pixelvalue));
	MaxIntMedianArr = (pixelvalue *) malloc(NrPixels*NrPixels*sizeof(pixelvalue));
	pixelvalue SubArr[NrFilesPerDistance];
	int tempVal;
	for (i=0;i<NrPixels*NrPixels;i++){
		MaxIntArr[i] = 0;
		MaxIntMedianArr[i] = 0;
		for (j=0;j<NrFilesPerDistance;j++){
			SubArr[j] = AllIntensities[j][i];
			if (AllIntensities[j][i] > MaxIntArr[i]){
				MaxIntArr[i] = AllIntensities[j][i];
			}
		}
		MedianArray[i] = quick_select(SubArr,NrFilesPerDistance);
		tempVal =  (MaxIntArr[i] - MedianArray[i]);
		MaxIntMedianArr[i] = (pixelvalue) (tempVal > 0 ? tempVal : 0);
	}

	int NrZeros = 0;
	for (i=0;i<NrPixels;i++){
		for (j=0;j<NrPixels;j++){
			filedata[i][j] = MedianArray[i*NrPixels+j];
			if (MedianArray[i*NrPixels+j] == 0){
				NrZeros++;
			}
		}
	}
	omp_set_lock(&lock);
	char dsetname[2048];
	sprintf(dsetname,"/analysis/median_images/distance_%d",DistanceNr);
	hsize_t dims[2]  = {NrPixels,NrPixels},
			chunk[2] = {NrPixels/8,NrPixels/8};
	hid_t space = H5Screate_simple(2,dims,NULL);
	dcpl = H5Pcreate(H5P_DATASET_CREATE);
	status = H5Pset_deflate(dcpl,4);
	status = H5Pset_chunk(dcpl,2,chunk);
	if (H5Lexists(file,dsetname,H5P_DEFAULT)==1){
		printf("Dataset found, will update: %s\n",dsetname);
		dataset = H5Dopen(file,dsetname,H5P_DEFAULT);
		status = H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}else{
		printf("Creating dataset: %s\n",dsetname);
		dataset = H5Dcreate(file,dsetname,H5T_NATIVE_USHORT,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
		H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}
	omp_unset_lock(&lock);

	for (i=0;i<NrPixels;i++)
		for (j=0;j<NrPixels;j++)
			filedata[i][j] = MaxIntArr[i*NrPixels+j];
	omp_set_lock(&lock);
	sprintf(dsetname,"/analysis/max_images/distance_%d",DistanceNr);
	if (H5Lexists(file,dsetname,H5P_DEFAULT)==1){
		printf("Dataset found, will update: %s\n",dsetname);
		dataset = H5Dopen(file,dsetname,H5P_DEFAULT);
		status = H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}else{
		printf("Creating dataset: %s\n",dsetname);
		dataset = H5Dcreate(file,dsetname,H5T_NATIVE_USHORT,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
		H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}
	omp_unset_lock(&lock);

	for (i=0;i<NrPixels;i++)
		for (j=0;j<NrPixels;j++)
			filedata[i][j] = MaxIntMedianArr[i*NrPixels+j];
	omp_set_lock(&lock);
	sprintf(dsetname,"/analysis/max_median_images/distance_%d",DistanceNr);
	if (H5Lexists(file,dsetname,H5P_DEFAULT)==1){
		printf("Dataset found, will update: %s\n",dsetname);
		dataset = H5Dopen(file,dsetname,H5P_DEFAULT);
		status = H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}else{
		printf("Creating dataset: %s\n",dsetname);
		dataset = H5Dcreate(file,dsetname,H5T_NATIVE_USHORT,space,H5P_DEFAULT,dcpl,H5P_DEFAULT);
		H5Dwrite(dataset,H5T_NATIVE_USHORT,H5S_ALL,H5S_ALL,H5P_DEFAULT,filedata[0]);
		H5Dclose(dataset);
	}
	omp_unset_lock(&lock);
}

static void
usage(void)
{
    printf("MedianImage: usage: ./MedianImageHDF <DataSet.hdf>\n");
}

int
main(int argc, char *argv[])
{
	if (argc != 2)
	{
		usage();
		return 1;
	}
		clock_t start, end;
	double diftotal;
	start = clock();
	herr_t status, status_n;
	htri_t avail;
	unsigned int filter_info;
	avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
	if (!avail) {
		printf("GZIP filter is not available. Will not be able to read the data file.\n");
		return 1;
	}
	status = H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
	if ( !(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
			!(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) ) {
		printf ("gzip filter not available for encoding and decoding.\n");
		return 1;
	}
	hid_t file, dataset;
	file = H5Fopen(argv[1],H5F_ACC_RDWR,H5P_DEFAULT);
	int NrFilesPerDistance;
	dataset = H5Dopen(file,"/measurement/nr_files_per_distance",H5P_DEFAULT);
	status = H5Dread(dataset,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&NrFilesPerDistance);
	printf("NrFilesPerDistance %d\n",NrFilesPerDistance);
	int NrDistances;
	dataset = H5Dopen(file,"/measurement/nr_distances",H5P_DEFAULT);
	status = H5Dread(dataset,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&NrDistances);
	printf("NrDistances %d\n",NrDistances);
	int NrPixels;
	dataset = H5Dopen(file,"/measurement/instrument/detector/geometry/nr_pixels",H5P_DEFAULT);
	status = H5Dread(dataset,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&NrPixels);
	printf("NrPixels %d\n",NrPixels);
	int DistanceNr;
	# pragma omp parallel num_threads(NrDistances) shared(file,NrFilesPerDistance,NrDistances,NrPixels) private(DistanceNr)
	{
		DistanceNr = omp_get_thread_num();
		CalcMedian(file,NrFilesPerDistance,NrDistances,NrPixels,DistanceNr);
	}
	end = clock();
	diftotal = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("Time elapsed in reading files, computing median and saving data to original HDF: %f [s]\n",diftotal);
	return 0;
}
