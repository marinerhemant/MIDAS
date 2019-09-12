//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#include "tomo_heads.h"

void LogProj(float *data, int xdim, int ydim) { 
	int i, k; 
	float mean, max; 
	for (i=0;i<ydim;i++) {
		max = data[i*xdim]; 
		for (k=0;k<xdim;k++) {
			if (data[i*xdim+k] > max) 
				max = data[i*xdim+k]; 
		}
		for (k=0;k<xdim;k++) { 
			if (data[i*xdim+k] <= 0.0) 
				data[i*xdim+k] = 1.0;
			data[i*xdim+k] = log (max/data[i*xdim+k]); 
		} 
	}
} 

void LogSinogram (float *data, int xdim, int ydim){
	int i, k;
	for (i=0;i<ydim;i++){
		for (k=0;k<xdim;k++){
			if (data[i*xdim+k] > 0)
				data[i*xdim+k] = -1 * log (data[i*xdim+k]);
			else
				data[i*xdim+k] = 0;
		}
	}
}

void RingCorrectionSingle (float *data, float ring_coeff, LOCAL_CONFIG_OPTS *information, GLOBAL_CONFIG_OPTS *recon_info_record) { 
	int         i, j, m; 
	float       mean_total; 
	float       tmp; 
	for (m=0;m<20;m++) {
		for (i=0;i<recon_info_record->sinogram_ydim;i++)  information->mean_vect[i] = 0.0; 
		mean_total = 0.0; 
		for (i=0;i<recon_info_record->sinogram_ydim;i++) {
			for (j=0;j<information->sinogram_adjusted_xdim;j++) {
				information->mean_vect[i] += data[i*information->sinogram_adjusted_xdim+j]; 
			}
			information->mean_vect[i] /= information->sinogram_adjusted_xdim; 
			mean_total += information->mean_vect[i]; 
		} 
		mean_total /= recon_info_record->sinogram_ydim; 
		for (i=0;i<recon_info_record->sinogram_ydim;i++) {
			for (j=0;j<information->sinogram_adjusted_xdim;j++) {
				if (information->mean_vect[i] != 0.0) {
					data[i*information->sinogram_adjusted_xdim+j] = data[i*information->sinogram_adjusted_xdim+j]*mean_total/information->mean_vect[i];
				}
			}
		}
		for (i=0;i<information->sinogram_adjusted_xdim;i++)  information->mean_sino_line_data[i] = 0.0; 
		for (i=0;i<recon_info_record->sinogram_ydim;i++) 
			for (j=0;j<information->sinogram_adjusted_xdim;j++) 
				information->mean_sino_line_data[j] += data[i*information->sinogram_adjusted_xdim+j]; 
		for (i=0;i<information->sinogram_adjusted_xdim;i++)  information->mean_sino_line_data[i] /= recon_info_record->sinogram_ydim; 
		for (j=1;j<information->sinogram_adjusted_xdim-1;j++) {
			information->low_pass_sino_lines_data[j] = (information->mean_sino_line_data[j-1]+information->mean_sino_line_data[j]+information->mean_sino_line_data[j+1])/3.0; 
		}
		information->low_pass_sino_lines_data[0] = information->mean_sino_line_data[0]; 
		information->low_pass_sino_lines_data[information->sinogram_adjusted_xdim-1] = information->mean_sino_line_data[information->sinogram_adjusted_xdim-1]; 
		for (i=0;i<recon_info_record->sinogram_ydim;i++) {
			for (j=0;j<information->sinogram_adjusted_xdim;j++) {
				tmp = information->mean_sino_line_data[j]-information->low_pass_sino_lines_data[j]; 
				if ((data[i*information->sinogram_adjusted_xdim+j] - (tmp * ring_coeff) ) > 0.0) 
					data[i*information->sinogram_adjusted_xdim+j] -= (tmp * ring_coeff); 
				else 
					data[i*information->sinogram_adjusted_xdim+j] = 0.0; 
			} 
		} 
	}
}

/* This is the definition: 
 * 1 dark(D_x), 
 * 2 whites (W1_x, W2_x) and 
 * y Images (I_x_y), 
 * the intensity should be 
 * I'_x_y = (I_x_y-D_x)/(W_x-D_x), where
 * W_x = p*W1_x + (1-p)*W2_x and
 * p = y/nr_y
*/
// This function assumes the short_sino is the proper sinogram, white_field_sino is two rows of first and last wf image slice, dark_field_sino_ave is a single slice. Size of each sino is recon_info_record->sinogram_xdim, output norm_sino is information->sinogram_adjusted_xdim (padded)

void Normalize (SINO_READ_OPTS *readStruct, GLOBAL_CONFIG_OPTS *recon_info_record){
	int pad_size = readStruct->sinogram_adjusted_xdim - recon_info_record->sinogram_xdim,
		front_pad_size = pad_size / 2,
		back_pad_size = pad_size - front_pad_size;
	int frameNr, pxNr, colNr;
	float front_pad_denom, front_pad_numer, temp_front, back_pad_denom, back_pad_numer, temp_back, white_temp, factor;
	front_pad_denom = ((float)(readStruct->white_field_sino[0]+readStruct->white_field_sino[recon_info_record->sinogram_xdim]))/2 - readStruct->dark_field_sino_ave[0];
	back_pad_denom = ((float)(readStruct->white_field_sino[recon_info_record->sinogram_xdim-1]+readStruct->white_field_sino[recon_info_record->sinogram_xdim*2-1]))/2 - readStruct->dark_field_sino_ave[recon_info_record->sinogram_xdim-1];
	for (frameNr=0;frameNr<recon_info_record->sinogram_ydim;frameNr++){
		front_pad_numer = (float)readStruct->short_sinogram[0] - readStruct->dark_field_sino_ave[0];
		back_pad_numer = (float)readStruct->short_sinogram[recon_info_record->sinogram_xdim-1] - readStruct->dark_field_sino_ave[recon_info_record->sinogram_xdim-1];
		temp_front = front_pad_numer / front_pad_denom;
		temp_back = back_pad_numer / back_pad_denom;
		factor = frameNr / recon_info_record->theta_list_size;
		if (temp_front==0) temp_front = 1e-3;
		if (temp_back==0) temp_back = 1e-3;
		for (pxNr=0;pxNr<readStruct->sinogram_adjusted_xdim;pxNr ++){
			if (pxNr<front_pad_size){ // front padding
				readStruct->norm_sino[frameNr*readStruct->sinogram_adjusted_xdim+pxNr] = temp_front;
			} else if (pxNr >=front_pad_size+recon_info_record->sinogram_xdim){ // back padding
				readStruct->norm_sino[frameNr*readStruct->sinogram_adjusted_xdim+pxNr] = temp_back;
			} else {
				// Apply our formula
				colNr = pxNr - front_pad_size;
				white_temp = factor * (float) readStruct->white_field_sino[colNr] + (1-factor) * (float) readStruct->white_field_sino[colNr+recon_info_record->sinogram_xdim];
				readStruct->norm_sino[frameNr*readStruct->sinogram_adjusted_xdim+pxNr] = ((float)readStruct->short_sinogram[colNr] - readStruct->dark_field_sino_ave[colNr]) /(white_temp-readStruct->dark_field_sino_ave[colNr]);
			}
		}
		
	}
}

void Pad (SINO_READ_OPTS *readStruct, GLOBAL_CONFIG_OPTS *recon_info_record){ // Take the sino directly read (init_sinogram) and pad it, return norm_sino.
	int pad_size = readStruct->sinogram_adjusted_xdim - recon_info_record->sinogram_xdim,
		front_pad_size = pad_size / 2,
		back_pad_size = pad_size - front_pad_size;
	int colNr, frameNr;
	for (frameNr=0;frameNr<recon_info_record->sinogram_ydim;frameNr++){
		for (colNr=0;colNr<readStruct->sinogram_adjusted_xdim;colNr++){
			if (colNr<front_pad_size) readStruct->norm_sino[colNr] = readStruct->init_sinogram[0];
			else if (colNr>=front_pad_size+recon_info_record->sinogram_xdim) readStruct->norm_sino[colNr] = readStruct->init_sinogram[recon_info_record->sinogram_xdim-1];
			else readStruct->norm_sino[colNr] = readStruct->init_sinogram[colNr-front_pad_size];
		}
	}
}

int setGlobalOpts(char *inputFN, GLOBAL_CONFIG_OPTS *recon_info_record){
	/* Input file is a text file name with a data link: sino data is a !!!single!!! binary file with darks, whites and tomo data in that order.
		* The rest of the file consists of the parameters required.
		* Parameters to be supplied:
			* dataFileName: [char*] name of the file with the raw data or sino data
			* reconFileName: [char*] Name of the file for saving the reconstruction
			* areSinos: If the input is a sinogram instead of raw (cleaned) images [0 or 1]
			* The data can be one of two types: 
			* 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. 
			* 							dark[float], whites (2,floats) and then raw images[shorts]. Using number of angles, we know how many images are there. 
			* 							The scaling with white should be proportional to the distance from a white and appropriate dark value.
			* detXdim - [uint]
			* detYdim - [uint]
			* Thetas can either be given as a range:
			* 	thetaRange: startAngle endAngle angleInterval - [floats]
			* or a File:
			* 	thetaFileName [char*] with each line having an angle value [float].
			* filter - [int] set to * 0: default
									* 1: Shepp / Logan
									* 2: Hann
									* 3: Hamming
									* 4: Ramp
			* shiftValues: start_shift end_shift shift_interval [floats] In case of 1 shift, give start_shift=end_shift, shift_interval doesn't matter
			* ringRemovalCoefficient - If given, will do ringRemoval, otherwise comment or remove line [float] default 1.0
			* slicesToProcess - -1 for all or FileName
	*/
	int arbThetas = 0;
	FILE *fileParam;
	fileParam = fopen(inputFN,"r");
	if (fileParam==NULL) return 1;
	char dummy[4096],aline[4096], slices[4096];
	int temp;
	recon_info_record->use_ring_removal = 0;
	recon_info_record->debug = 0;
	while(fgets(aline,4096,fileParam)!=NULL){
		if (strncmp(aline,"dataFileName",strlen("dataFileName"))==0){
			sscanf(aline,"%s %s",dummy,recon_info_record->DataFileName);
		}
		if (strncmp(aline,"reconFileName",strlen("reconFileName"))==0){
			sscanf(aline,"%s %s",dummy,recon_info_record->ReconFileName);
		}
		if (strncmp(aline,"areSinos",strlen("areSinos"))==0){
			sscanf(aline,"%s %ud",dummy,&recon_info_record->are_sinos);
		}
		if (strncmp(aline,"detXdim",strlen("detXdim"))==0){
			sscanf(aline,"%s %ud",dummy,&recon_info_record->det_xdim);
		}
		if (strncmp(aline,"detYdim",strlen("detYdim"))==0){
			sscanf(aline,"%s %ud",dummy,&recon_info_record->det_ydim);
		}
		if (strncmp(aline,"filter",strlen("filter"))==0){
			sscanf(aline,"%s %d",dummy,&recon_info_record->filter);
		}
		if (strncmp(aline,"debug",strlen("debug"))==0){
			sscanf(aline,"%s %d",dummy,&recon_info_record->debug);
		}
		if (strncmp(aline,"thetaRange",strlen("thetaRange"))==0){
			sscanf(aline,"%s %f %f %f",dummy,&recon_info_record->start_angle,&recon_info_record->end_angle,&recon_info_record->angle_interval);
		}
		if (strncmp(aline,"thetaFileName",strlen("thetaFileName"))==0){
			arbThetas = 1;
			sscanf(aline,"%s %s",dummy,recon_info_record->thetaFileName);
		}
		if (strncmp(aline,"shiftValues",strlen("shiftValues"))==0){
			sscanf(aline,"%s %f %f %f",dummy,&recon_info_record->start_shift,&recon_info_record->end_shift,&recon_info_record->shift_interval);
		}
		if (strncmp(aline,"ringRemovalCoeff",strlen("ringRemovalCoeff"))==0){
			recon_info_record->use_ring_removal = 1;
			sscanf(aline,"%s %f",dummy,&recon_info_record->ring_removal_coeff);
		}
		if (strncmp(aline,"slicesToProcess",strlen("slicesToProcess"))==0){
			sscanf(aline,"%s %s %s",dummy,slices,dummy);
		}
	}
	recon_info_record->auto_centering = 1; // ALWAYS DONE
	fseek(fileParam,0,SEEK_SET);
	if (arbThetas == 0){
		recon_info_record->theta_list_size = abs((recon_info_record->end_angle-recon_info_record->start_angle)/recon_info_record->angle_interval) + 1;
		recon_info_record->theta_list = (float *) malloc(recon_info_record->theta_list_size*sizeof(float));
		int i;
		for (i=0;i<recon_info_record->theta_list_size;i++){
			recon_info_record->theta_list[i] = recon_info_record->start_angle + i*recon_info_record->angle_interval;
		}
	} else {
		recon_info_record->theta_list_size = 0;
		recon_info_record->theta_list = (float *) malloc(MAX_N_THETAS*sizeof(float));
		FILE *fileTheta = fopen(recon_info_record->thetaFileName,"r");
		while (fgets (aline,4096,fileTheta)!=NULL){
			recon_info_record->theta_list[recon_info_record->theta_list_size] = atoi(aline);
			recon_info_record->theta_list_size ++;
		}
	}
	recon_info_record->n_shifts = abs((recon_info_record->end_shift-recon_info_record->start_shift))/recon_info_record->shift_interval+1;
	recon_info_record->shift_values = (float *) malloc(sizeof(float)*(recon_info_record->n_shifts));
	int i;
	for (i=0;i<recon_info_record->n_shifts;i++){
		recon_info_record->shift_values[i] = recon_info_record->start_shift + i*recon_info_record->shift_interval;
	}
	long val;
	char *endptr;
	val = strtol(slices,&endptr,10);
	if (endptr == slices){ // filename with slices, doesn't start with an integer
		sprintf(recon_info_record->SliceFileName,"%s",slices);
		FILE *slicesFile = fopen(recon_info_record->SliceFileName,"r");
		recon_info_record->n_slices = 0;
		recon_info_record->slices_to_process = (uint *) malloc(sizeof(uint)*recon_info_record->det_ydim);
		printf("We are reading the slices file: %s",slices);
		while(fgets(aline,4096,slicesFile)!=NULL){
			recon_info_record->slices_to_process[recon_info_record->n_slices] = atoi(aline);
			recon_info_record->n_slices++;
		}
	}else{
		if (strncmp(slices,"-1",strlen("-1"))==0){
			printf("We are doing all slices.");
			recon_info_record->slices_to_process = (uint *) malloc(sizeof(uint)*recon_info_record->det_ydim);
			for (i=0;i<recon_info_record->det_ydim;i++)
				recon_info_record->slices_to_process[i] = i;
			recon_info_record->n_slices = recon_info_record->det_ydim;
		} else {
			printf("We are doing only 1 slice: %s",slices);
			recon_info_record->slices_to_process = (uint *) malloc(sizeof(uint)*1);
			recon_info_record->slices_to_process[0] = atoi(slices);
		}
	}
	recon_info_record->sinogram_ydim = recon_info_record->theta_list_size; // Equal to number of files
	recon_info_record->sinogram_xdim = recon_info_record->det_xdim;
	return 0;
}

void setReadStructSize (GLOBAL_CONFIG_OPTS *recon_info_record, SINO_READ_OPTS *readStruct){
	int power, size;
	bool still_smaller;
	still_smaller = true;
	power = 0;
	while (still_smaller){
		if (recon_info_record->sinogram_xdim > pow (2, power)){
			power++;
			still_smaller = true;
		} else {
			still_smaller = false;
		}
	}
	if (recon_info_record->sinogram_xdim == pow (2, power)){
		readStruct->sinogram_adjusted_xdim = recon_info_record->sinogram_xdim;
		recon_info_record->reconstruction_xdim = recon_info_record->sinogram_xdim;
		recon_info_record->reconstruction_ydim = recon_info_record->sinogram_xdim;
		readStruct->sinogram_adjusted_size = readStruct->sinogram_adjusted_xdim * recon_info_record->sinogram_ydim;
		readStruct->reconstruction_size = recon_info_record->reconstruction_xdim*recon_info_record->reconstruction_ydim;
		printf ("Sinograms are a power of 2!\n");
	}else{
		size = (int) pow (2, power);
		readStruct->sinogram_adjusted_xdim = size;
		readStruct->sinogram_adjusted_size = readStruct->sinogram_adjusted_xdim * recon_info_record->sinogram_ydim;
		recon_info_record->reconstruction_xdim = size;
		recon_info_record->reconstruction_ydim = size;
		readStruct->reconstruction_size = recon_info_record->reconstruction_xdim*recon_info_record->reconstruction_ydim;
		printf ("Sinograms are not a power of 2.  They will be increased to %d\n", readStruct->sinogram_adjusted_xdim);
	}
}

void setSinoSize (LOCAL_CONFIG_OPTS *information, GLOBAL_CONFIG_OPTS *recon_info_record, SINO_READ_OPTS *readStruct){
	information->sinogram_adjusted_xdim = readStruct->sinogram_adjusted_xdim;
	information->sinogram_adjusted_size = readStruct->sinogram_adjusted_size;
	information->reconstruction_size = readStruct->reconstruction_size;
	printf("shifted_recon: %ld\n",(long)(sizeof (float)*information->reconstruction_size));
	printf("shifted_sinogram %ld\n",(long)(sizeof (float)*information->sinogram_adjusted_size));
	printf("sinograms_boundary_padding %ld\n",(long)(sizeof(float)*information->sinogram_adjusted_size*2));
	printf("reconstructions_boundary_padding %ld\n",(long)(sizeof(float)*information->reconstruction_size*4));
	printf("recon_calc_buffer %ld\n",(long)(sizeof(float)*information->reconstruction_size*2));
	information->shifted_recon = (float *) malloc (sizeof (float)*information->reconstruction_size);
	information->shifted_sinogram = (float *) malloc (sizeof (float)*information->sinogram_adjusted_size);
	information->sinograms_boundary_padding = (float *) malloc (sizeof(float)*information->sinogram_adjusted_size*2);
	information->reconstructions_boundary_padding = (float *) malloc (sizeof(float)*information->reconstruction_size*4);
	information->recon_calc_buffer = (float *) malloc (sizeof(float)*information->reconstruction_size*2);
	if (recon_info_record->auto_centering){
		printf("mean_vect %ld\n",(long)(sizeof (float)*recon_info_record->sinogram_ydim));
		printf("mean_sino_line_data %ld\n",(long)(sizeof (float)*information->sinogram_adjusted_xdim));
		printf("low_pass_sino_lines_data %ld\n",(long)(sizeof(float) *information->sinogram_adjusted_xdim));
		information->mean_vect = (float *) malloc (sizeof (float)*recon_info_record->sinogram_ydim);
		information->mean_sino_line_data = (float *) malloc (sizeof (float)*information->sinogram_adjusted_xdim);
		information->low_pass_sino_lines_data = (float  *) malloc (sizeof(float) *information->sinogram_adjusted_xdim); 
	}
}

void readSino(int sliceNr,GLOBAL_CONFIG_OPTS recon_info_record, SINO_READ_OPTS *readStruct){
	FILE *dataFile;
	dataFile = fopen(recon_info_record.DataFileName,"rb");
	size_t offset = sizeof(float)*sliceNr*recon_info_record.det_xdim*recon_info_record.det_ydim;
	fseek(dataFile,offset,SEEK_SET);
	size_t SizeSino = sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim;
	printf("init_sinogram %ld\n",(long)SizeSino);
	printf("norm_sino %ld\n",(long)(sizeof(float)*readStruct->sinogram_adjusted_xdim*recon_info_record.det_ydim));
	readStruct->init_sinogram = (float *) malloc(SizeSino);
	readStruct->norm_sino = (float *) malloc(sizeof(float)*readStruct->sinogram_adjusted_xdim*recon_info_record.det_ydim);
	fread(readStruct->init_sinogram,SizeSino,1,dataFile);
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"init_sinogram_%s",recon_info_record.DataFileName);
		fwrite(readStruct->init_sinogram,SizeSino,1,outfn);
	}
	Pad(readStruct,&recon_info_record);
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"norm_sino_%s",recon_info_record.DataFileName);
		fwrite(readStruct->norm_sino,sizeof(float)*readStruct->sinogram_adjusted_xdim*recon_info_record.det_ydim,1,outfn);
	}
}

void readRaw(int sliceNr,GLOBAL_CONFIG_OPTS recon_info_record,SINO_READ_OPTS *readStruct) {
	FILE *dataFile;
	dataFile = fopen(recon_info_record.DataFileName,"rb");
	size_t offset, SizeDark, SizeWhite, SizeSino, SizeNormSino;
	// Dark
	SizeDark = sizeof(float)*recon_info_record.det_xdim;
	printf("dark_field_sino_ave %ld\n",(long)SizeDark);
	readStruct->dark_field_sino_ave = (float *) malloc(SizeDark);
	offset = sizeof(float*)*sliceNr*recon_info_record.det_xdim;
	fseek(dataFile,offset,SEEK_SET);
	fread(readStruct->dark_field_sino_ave,SizeDark,1,dataFile);
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"dark_field_%s",recon_info_record.DataFileName);
		fwrite(readStruct->dark_field_sino_ave,SizeDark,1,outfn);
	}
	// 2 Whites
	SizeWhite = sizeof(float)*recon_info_record.det_xdim*2;
	printf("white_field_sino %ld\n",(long)SizeWhite);
	readStruct->white_field_sino = (float *) malloc(SizeWhite);
	offset = sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim // dark
				+ sizeof(float)*recon_info_record.det_xdim*sliceNr; // Partial white
	fseek(dataFile,offset,SEEK_SET);
	fread(readStruct->white_field_sino,SizeWhite/2,1,dataFile); // One Row
	offset = sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim // dark
				+ sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim // One full white
				+ sizeof(float)*recon_info_record.det_xdim*sliceNr; // Partial white
	fseek(dataFile,offset,SEEK_SET);
	fread((readStruct->white_field_sino)+recon_info_record.det_xdim,SizeWhite/2,1,dataFile); // Second Row
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"whites_%s",recon_info_record.DataFileName);
		fwrite(readStruct->white_field_sino,SizeWhite,1,outfn);
	}
	// Sino start
	SizeSino = sizeof(unsigned short int)*recon_info_record.det_xdim*recon_info_record.det_ydim;
	printf("short_sinogram %ld\n",(long)SizeSino);
	readStruct->short_sinogram = (unsigned short int *) malloc(SizeSino);
	offset = sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim // dark
				+ sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim // One full white
				+ sizeof(float)*recon_info_record.det_xdim*recon_info_record.det_ydim; // Second full white
	fseek(dataFile,offset,SEEK_SET);
	// We are now at the beginning of the image data.
	offset = sizeof(unsigned short int)*recon_info_record.det_xdim*sliceNr;
	fseek(dataFile,offset,SEEK_CUR);
	fread(readStruct->short_sinogram,sizeof(unsigned short int)*recon_info_record.det_xdim,1,dataFile); // One row
	int frameNr;
	for (frameNr=1;frameNr<recon_info_record.sinogram_ydim;frameNr++){
		offset = sizeof(unsigned short int)*recon_info_record.det_xdim*(recon_info_record.det_ydim-1);
		fseek(dataFile,offset,SEEK_CUR);
		fread((readStruct->short_sinogram)+recon_info_record.det_xdim*frameNr,sizeof(unsigned short int)*recon_info_record.det_xdim,1,dataFile); // One row each at the next subsequent place
	}
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"short_sinogram_%s",recon_info_record.DataFileName);
		fwrite(readStruct->short_sinogram,SizeSino,1,outfn);
	}
	SizeNormSino = sizeof(float)*readStruct->sinogram_adjusted_xdim*recon_info_record.det_ydim;
	printf("norm_sino %ld\n",(long)SizeNormSino);
	readStruct->norm_sino = (float *) malloc(SizeNormSino);
	Normalize(readStruct,&recon_info_record);
	if (recon_info_record.debug == 1){
		char outfn[4096];
		sprintf(outfn,"norm_sino_%s",recon_info_record.DataFileName);
		fwrite(readStruct->norm_sino,SizeNormSino,1,outfn);
	}
}

void reconCentering(LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS *recon_info_record){
	int j, k;
	LogProj(information->sino_calc_buffer, information->sinogram_adjusted_xdim, recon_info_record->sinogram_ydim);
	for( j = 0; j < recon_info_record->sinogram_ydim; j++ ){
		for( k = 0; k < information->sinogram_adjusted_xdim; k++ ){
			information->shifted_recon[j * information->sinogram_adjusted_xdim+ k] = 0.0f;
		}
	}
	for( j = 0; j < recon_info_record->sinogram_ydim; j++ ){
		for( k = 0; k < information->sinogram_adjusted_xdim; k++ ){
			float kk = k - information->shift; 
			int nkk = (int)floor(kk);
			float fInterpPixel = 0.0f;
			float fInterpWeight = 0.0f;
			if( nkk >= 0 && nkk < information->sinogram_adjusted_xdim ){
				fInterpPixel += information->sino_calc_buffer[j * information->sinogram_adjusted_xdim + nkk ] * (nkk + 1 - kk);
				fInterpWeight = nkk + 1 - kk;
			}
			if( nkk + 1 >= 0 && nkk + 1 < information->sinogram_adjusted_xdim ){
				fInterpPixel += information->sino_calc_buffer[j * information->sinogram_adjusted_xdim + nkk + 1] * (kk - nkk);
				fInterpWeight += kk - nkk;
			}
			if( fInterpWeight < 1e-5 )
				fInterpPixel = 0.0f;
			else
				fInterpPixel /= fInterpWeight;
			information->shifted_sinogram[ j * information->sinogram_adjusted_xdim + k ] = fInterpPixel;
		}
	}
	memcpy(&information->sino_calc_buffer[0], information->shifted_sinogram, sizeof(float) * information->sinogram_adjusted_size);
	if (recon_info_record->use_ring_removal){
		RingCorrectionSingle (&information->sino_calc_buffer[0],recon_info_record->ring_removal_coeff,information,recon_info_record);
	}
	for( j = 0; j < recon_info_record->sinogram_ydim; j++ ){
		memcpy( &information->sinograms_boundary_padding[j * information->sinogram_adjusted_xdim * 2 + information->sinogram_adjusted_xdim / 2 ],&information->sino_calc_buffer[j * information->sinogram_adjusted_xdim ], sizeof(float) * information->sinogram_adjusted_xdim);
		for( k = 0; k < information->sinogram_adjusted_xdim /2; k++ ){
			information->sinograms_boundary_padding[j * information->sinogram_adjusted_xdim * 2 + k ] = information->sinograms_boundary_padding[j * information->sinogram_adjusted_xdim * 2 + information->sinogram_adjusted_xdim / 2 ];
		}
		for( k = 0; k < information->sinogram_adjusted_xdim /2; k++ ){
			information->sinograms_boundary_padding[j * information->sinogram_adjusted_xdim * 2 + information->sinogram_adjusted_xdim / 2 + information->sinogram_adjusted_xdim + k ] = information->sinograms_boundary_padding[j * information->sinogram_adjusted_xdim * 2 + information->sinogram_adjusted_xdim / 2 + information->sinogram_adjusted_xdim - 1];
		}
	}
}

void getRecons(LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS *recon_info_record,gridrecParams *param){
	int j,k;
	for ( j=0;j<recon_info_record->reconstruction_ydim;j++){
		if (information->shift >= 0){
			memcpy(&information->recon_calc_buffer[j * recon_info_record->reconstruction_xdim ],&information->reconstructions_boundary_padding[ ( j + recon_info_record->reconstruction_xdim / 2 ) * recon_info_record->reconstruction_xdim * 2 + recon_info_record->reconstruction_xdim / 2 ], sizeof(float) * (recon_info_record->reconstruction_xdim) ); 
		}else{
			memcpy(&information->recon_calc_buffer[j * recon_info_record->reconstruction_xdim ],&information->reconstructions_boundary_padding[ ( j + recon_info_record->reconstruction_xdim / 2 ) * recon_info_record->reconstruction_xdim * 2 + recon_info_record->reconstruction_xdim / 2 ], sizeof(float) * (recon_info_record->reconstruction_xdim) );
		}
	}
	for( j = 0; j < recon_info_record->sinogram_ydim; j++ ){
		for( k = 0; k < recon_info_record->reconstruction_xdim; k++ ){
			information->shifted_recon[j * recon_info_record->reconstruction_xdim + k] = 0.0f;
		}
	}
	float *recon_buffer;
	if (recon_info_record->auto_centering){
		recon_buffer = &information->recon_calc_buffer[0];
		if (information->shift >= 0){
			for ( j=0;j<recon_info_record->reconstruction_ydim;j++)
				memcpy (&information->shifted_recon[j*recon_info_record->reconstruction_xdim], (void *) &recon_buffer[(j*recon_info_record->reconstruction_xdim)+ (int)round(information->shift) ], sizeof(float)*(recon_info_record->reconstruction_xdim- (int)round(information->shift) ));
		} else {
			for ( j=0;j<recon_info_record->reconstruction_ydim;j++)
				memcpy (&information->shifted_recon[(j*recon_info_record->reconstruction_xdim)+abs ((int)round(information->shift))], (void *) &recon_buffer[j*recon_info_record->reconstruction_xdim], sizeof(float)*(recon_info_record->reconstruction_xdim-abs ((int)round(information->shift) )));
		}
		memcpy ((void *) recon_buffer, information->shifted_recon, sizeof(float)*information->reconstruction_size);
	}
}

void writeRecon(int sliceNr,LOCAL_CONFIG_OPTS *information,GLOBAL_CONFIG_OPTS *recon_info_record){
	// The results are in information.recon_calc_buffer
	// Output file: float with reconstruction_xdim*reconstruction_xdim size
	// OutputFileName: {recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_reconstruction_xdim_float_4byte.bin
	char outFileName[4096];
	sprintf(outFileName,"%s_%d_%d_%d_float_4byte.bin",recon_info_record->ReconFileName,sliceNr,recon_info_record->reconstruction_xdim,recon_info_record->reconstruction_xdim);
	FILE *outfile;
	outfile = fopen(outFileName,"wb");
	fwrite(information->recon_calc_buffer,sizeof(float)*information->reconstruction_size,1,outfile);
	fclose(outfile);
}
