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

/* 
 * The data can be one of two types: 
 * 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. 
 * 							dark, whites (2) and then raw images. Using number of angles, we know how many images are there. The scaling with white should be proportional to the distance from a white and appropriate dark value.
 * The functions assume one sinogram being processed for now with one fixed_shift.
*/

// TODO:
//	1. read Array of slices_to_process.
//	2. Make array with slices_to_process and shifts, it will be the (number of times we reconstruct)/2. Gridrec can process 2 slices at once.
//	3. set shift before calling the reconCentering function.
//	4. Set two slices when we call setSinoAndReconBuffers
//	5. Check difference between lines 1192 and 1198 in tomompi_client.cpp
//	6. Change output file to single file.
//	7. Safe malloc to check NULL ptrs.
//	8. Check memory usage and determine the number of processes we would be able to run per node.

void usage(){
	printf("MIDAS-TOMO Code to do tomo recon using Gridrec. Based on tomompi implementation from Brian Tiemann, APS. Maintained by Hemant Sharma, APS (hsharma@anl.gov).\nUsage is: \n"
		"tomo ParamsFile.txt\n"
		"Params file must have the following parameters:\n"
		"Input file is a text file name with a data link: sino data is a !!!single!!! binary file with darks, whites and tomo data in that order.\n"
		"* The rest of the file consists of the parameters required.\n"
		"* Parameters to be supplied:\n"
		"	* DataFileName: [char*] name of the file with the raw data or sino data\n"
		"	* ReconFileName: [char*] Name of the file for saving the reconstruction\n"
		"	* areSinos: If the input is a sinogram instead of raw (cleaned) images [0 or 1]\n"
		"	* The data can be one of two types: \n"
		"	* 							sinogram already with float data type, directly give to reconstruct code with some additional centering etc. \n"
		"	* 							dark, whites (2) and then raw images. Using number of angles, we know how many images are there. The scaling with white should be proportional to the distance from a white and appropriate dark value.\n"
		"	* detXdim - [uint]\n"
		"	* detYdim - [uint]\n"
		"	* Thetas can either be given as a range:\n"
		"	* 	thetaRange: startAngle endAngle angleInterval - [floats]\n"
		"	* or a File:\n"
		"	* 	thetaFileName [char*] with each line having an angle value [float].\n"
		"	* filter - [int] set to * 0: default\n"
		"							* 1: Shepp / Logan\n"
		"							* 2: Hann\n"
		"							* 3: Hamming\n"
		"							* 4: Ramp\n"
		"	* shift_values: start_shift end_shift shift_interval [floats] In case of 1 shift, give start_shift=end_shift, shift_interval doesn't matter\n"
		"	* ringRemovalCoefficient - If given, will do ringRemoval, otherwise comment or remove line [float] default 1.0\n"
		"	* slicesToProcess - -1 for all or FileName\n"
		"Output file: float with reconstruction_xdim*reconstruction_xdim size\n"
		"OutputFileName: {recon_info_record.ReconFileName}_sliceNr_reconstruction_xdim_reconstruction_xdim_float_4byte.bin\n");
}

int main(int argc, char *argv[])
{
	if (argc!=2){
		usage();
		return 1;
	}
	GLOBAL_CONFIG_OPTS recon_info_record;
	char *fileName;
	fileName = argv[1];
	int RC;
	RC = setGlobalOpts(fileName, &recon_info_record);
	if (RC!=0){
		printf("Parameter file could not be read. Exiting.\n");
		return 1;
	}
	// Make groups of two, read two slices for each omp job, 
			// private: information, param, readStruct, sliceNr
			// public recon_info_record
	SINO_READ_OPTS readStruct;
	setReadStructSize(&recon_info_record,&readStruct);
	int sliceNr;
	sliceNr = 0; // TEMPORARY
	if (recon_info_record.are_sinos){
		printf("We have sinograms.\n");
		readSino(sliceNr,recon_info_record,&readStruct);
	} else {
		printf("We were provided with Dark, Whites (2) and images. We will do pre-processing ourselves.\n");
		readRaw(sliceNr,recon_info_record,&readStruct);
	}
	// Do till here for each slice, next step is when we have multiple shifts
	// define shift here
	LOCAL_CONFIG_OPTS information;
	printf("sino_calc_buffer %ld\n",(long)(sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.det_ydim));
	information.sino_calc_buffer = (float *) malloc(sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.det_ydim);
	memcpy(information.sino_calc_buffer,readStruct.norm_sino,sizeof(float)*information.sinogram_adjusted_xdim*recon_info_record.det_ydim);
	gridrecParams param;
	setSinoSize(&information,&recon_info_record,&readStruct);
	setGridRecPSWF(&param);
	param.sinogram_x_dim = information.sinogram_adjusted_xdim * 2; // Always GRIDREC_PADDING_HALF is assumed.
	param.theta_list = recon_info_record.theta_list;
	param.filter_type = recon_info_record.filter;
	param.theta_list_size = recon_info_record.theta_list_size;
	initGridRec(&param);
	information.shift = 0; // TEMPORARY
	reconCentering(&information,&recon_info_record); // We can probably clean up the arrays after this if not needed.
	// Placeholder to do the same slice twice
	setSinoAndReconBuffers(1, &information.sinograms_boundary_padding[0], &information.reconstructions_boundary_padding[0],&param);
	setSinoAndReconBuffers(2, &information.sinograms_boundary_padding[0], &information.reconstructions_boundary_padding[0],&param);
	reconstruct(&param);
	getRecons(&information,&recon_info_record,&param);
	writeRecon(sliceNr,&information,&recon_info_record);
	return 0;
}
